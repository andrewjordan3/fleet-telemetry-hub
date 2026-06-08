"""DuckDB-backed incremental delete-then-append merge for ``UtilizationPipeline``.

Companion module to ``utilization_pipeline.py``. Owns the read + window
delete + append + global sort + write of the output parquet entirely in
DuckDB, so pandas never materializes the whole (unbounded) on-disk file:
DuckDB streams ``read_parquet``, the window delete is a ``WHERE``, the
append is ``UNION ALL`` over the registered window-sized new frame, the
global sort is ``ORDER BY`` (spilling to the destination's parent
directory when it does not fit RAM), and the result is written with
``COPY``.

The merge normalizes every provider to start-anchored: both the incoming
frame and the existing file are filtered on ``start_time_utc`` in the
half-open window ``[W_start, W_end)``. Samsara's ``/v1/fleet/trips``
endpoint is overlap-anchored (it returns any trip intersecting the query
window), so filtering on start keeps a cross-boundary event from being
duplicated at a window's leading edge. Events starting before the window
are dropped from the incoming frame by design -- their single
authoritative copy already lives under the earlier window that owns
their start.

The parquet DuckDB writes is Arrow-native (microsecond timestamps, no
pandas extension metadata). Consumers restore the locked in-memory
``DTYPES`` via ``unifier.schema.read_unified_parquet``.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from uuid import uuid4

import duckdb
import pandas as pd

from fleet_telemetry_hub.unifier.schema import COLUMNS

__all__: list[str] = [
    'CorruptUtilizationParquetError',
    'MergeStats',
    'merge_incremental_to_parquet',
]

logger: logging.Logger = logging.getLogger(__name__)

# SQL fragments built once from the locked column tuple.
_COLUMN_LIST: str = ', '.join(COLUMNS)

# Half-open window predicate on ``start_time_utc``; the two ``?`` bind to
# ``(W_start, W_end)``. Used verbatim for the incoming filter and, negated,
# for the existing-row deletion, so the two predicates can never drift.
_WINDOW_PREDICATE: str = 'start_time_utc >= ? AND start_time_utc < ?'

# Total deterministic ordering key. SQL ``ORDER BY`` is not stable, so the
# key must cover every column to make the on-disk byte order reproducible
# run to run. ``NULLS FIRST`` on ``company`` matches pandas
# ``na_position='first'``.
_ORDER_BY: str = (
    'company ASC NULLS FIRST, start_time_utc ASC, event_type ASC, '
    'driver_id, driver_name, vin, end_time_utc, duration_seconds, distance_miles'
)


class CorruptUtilizationParquetError(Exception):
    """Raised when an existing data.parquet is unreadable or schema-mismatched."""


@dataclass(frozen=True, slots=True)
class MergeStats:
    """Row-delta counts for one merge, for logging and the run result.

    Attributes:
        incoming: Rows in ``new_frame``.
        kept_in_window: ``new_frame`` rows with start in ``[W_start, W_end)``.
        existing_deleted: Existing rows removed (0 on a first run).
        existing_retained: Existing rows kept (outside the window).
        final_row_count: Rows written to the output parquet.
    """

    incoming: int
    kept_in_window: int
    existing_deleted: int
    existing_retained: int
    final_row_count: int


def merge_incremental_to_parquet(
    new_frame: pd.DataFrame,
    start_date: date,
    end_date: date,
    destination_path: Path,
    *,
    compression: str,
) -> MergeStats:
    """Merge a freshly fetched window into ``destination_path``, in DuckDB.

    Deletes the half-open UTC window ``[W_start, W_end)`` from any existing
    file at ``destination_path`` and appends the rows of ``new_frame``
    whose ``start_time_utc`` falls inside that window, globally sorts the
    union by the total deterministic key, and writes it with ``COPY``. The
    whole pipeline streams in DuckDB; pandas only holds the window-sized
    ``new_frame``. The function owns the full write transaction: it creates
    the destination's parent directory, writes to a unique temp file in
    that directory, and atomically renames the temp file onto
    ``destination_path``. A failure before the rename leaves any existing
    destination untouched and leaks no temp file.

    Args:
        new_frame: The freshly fetched, window-sized unified frame.
        start_date: Inclusive UTC start date of the fetch window.
        end_date: Inclusive UTC end date of the fetch window.
        destination_path: Final parquet path. Read for the existing rows
            (when present) and atomically replaced with the merged result.
        compression: Parquet codec for ``COPY`` (e.g. ``'snappy'``,
            ``'zstd'``, ``'uncompressed'``).

    Returns:
        ``MergeStats`` with the incoming / kept / deleted / retained /
        final row counts.

    Raises:
        ValueError: If ``start_date > end_date``.
        CorruptUtilizationParquetError: If an existing ``destination_path``
            is unreadable or its column set does not match ``COLUMNS``.
            Never silently treated as a first run.

    Side Effects:
        Creates ``destination_path.parent``; writes via a temp file in that
        directory and atomically renames it onto ``destination_path``; the
        temp file is always removed, so a failure leaks no ``*.tmp``. Emits
        one ``DEBUG`` row-delta line.
    """
    if start_date > end_date:
        raise ValueError(f'start_date ({start_date}) must be <= end_date ({end_date})')

    window: tuple[datetime, datetime] = _window_bounds(start_date, end_date)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    existing_path: Path | None = (
        destination_path if destination_path.exists() else None
    )
    temp_path: Path = destination_path.parent / f'data-{uuid4().hex}.parquet.tmp'

    try:
        with duckdb.connect() as connection:
            connection.execute(
                f'SET temp_directory = {_sql_literal(str(destination_path.parent))}'
            )

            if existing_path is not None:
                _validate_existing_columns(connection, existing_path)

            connection.register('new_frame', new_frame)
            _copy_merged(connection, existing_path, temp_path, window, compression)
            stats: MergeStats = _compute_stats(
                connection, existing_path, len(new_frame), window
            )
        temp_path.replace(destination_path)
    finally:
        temp_path.unlink(missing_ok=True)

    logger.debug(
        'merge_incremental: incoming=%d kept_in_window=%d '
        'existing_deleted=%d existing_retained=%d final=%d',
        stats.incoming,
        stats.kept_in_window,
        stats.existing_deleted,
        stats.existing_retained,
        stats.final_row_count,
    )
    return stats


def _window_bounds(start_date: date, end_date: date) -> tuple[datetime, datetime]:
    """Half-open UTC window ``[W_start, W_end)`` as tz-aware ``datetime`` bounds.

    Args:
        start_date: Inclusive UTC start date of the fetch window.
        end_date: Inclusive UTC end date of the fetch window.

    Returns:
        ``(window_start, window_end)``: midnight UTC of ``start_date`` and
        midnight UTC of the day after ``end_date`` (exclusive). These are
        bound to DuckDB as parameters, not string-formatted into SQL.
    """
    window_start = datetime.combine(start_date, time.min, tzinfo=UTC)
    window_end = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=UTC)
    return window_start, window_end


def _validate_existing_columns(
    connection: duckdb.DuckDBPyConnection, path: Path
) -> None:
    """Validate the existing parquet's readability and column set, no data scan.

    Uses ``DESCRIBE SELECT *`` so only the schema is read. An unreadable
    file (``duckdb.Error``, e.g. ``InvalidInputException``) or a column set
    that does not match ``COLUMNS`` raises ``CorruptUtilizationParquetError``
    rather than being treated as a first run.

    Args:
        connection: Open DuckDB connection.
        path: Path to the existing parquet.

    Raises:
        CorruptUtilizationParquetError: On an unreadable file or a column
            mismatch (naming ``unexpected=`` / ``missing=``).
    """
    try:
        described = connection.execute(
            'DESCRIBE SELECT * FROM read_parquet(?)', [str(path)]
        ).fetchall()
    except duckdb.Error as describe_error:
        raise CorruptUtilizationParquetError(
            f'Existing parquet at {path} is unreadable'
        ) from describe_error

    actual_columns = {row[0] for row in described}
    expected_columns = set(COLUMNS)
    if actual_columns != expected_columns:
        unexpected = sorted(actual_columns - expected_columns)
        missing = sorted(expected_columns - actual_columns)
        raise CorruptUtilizationParquetError(
            f'Existing parquet at {path} has a mismatched column set: '
            f'unexpected={unexpected}, missing={missing}'
        )


def _copy_merged(
    connection: duckdb.DuckDBPyConnection,
    existing_path: Path | None,
    output_path: Path,
    window: tuple[datetime, datetime],
    compression: str,
) -> None:
    """Build and execute the ``COPY`` that writes the merged, sorted parquet.

    The COPY target path and compression are escaped string literals
    (DuckDB binds the COPY target before the inner query's parameters, so
    it cannot be a ``?``); the ``read_parquet`` path and the window bounds
    are bound parameters.

    Args:
        connection: Open DuckDB connection with ``new_frame`` registered.
        existing_path: Existing parquet path, or ``None`` on a first run.
        output_path: Destination parquet the COPY writes.
        window: The half-open ``(window_start, window_end)`` bounds.
        compression: Parquet codec for the COPY.

    Side Effects:
        Writes ``output_path``.
    """
    window_start, window_end = window
    if existing_path is None:
        select_sql = f'SELECT {_COLUMN_LIST} FROM new_frame WHERE {_WINDOW_PREDICATE}'
        params: list[object] = [window_start, window_end]
    else:
        select_sql = (
            f'SELECT {_COLUMN_LIST} FROM read_parquet(?) '
            f'WHERE NOT ({_WINDOW_PREDICATE}) '
            f'UNION ALL '
            f'SELECT {_COLUMN_LIST} FROM new_frame WHERE {_WINDOW_PREDICATE}'
        )
        params = [
            str(existing_path),
            window_start,
            window_end,
            window_start,
            window_end,
        ]

    copy_sql = (
        f'COPY ({select_sql} ORDER BY {_ORDER_BY}) '
        f'TO {_sql_literal(str(output_path))} '
        f'(FORMAT parquet, COMPRESSION {_sql_literal(compression)})'
    )
    connection.execute(copy_sql, params)


def _compute_stats(
    connection: duckdb.DuckDBPyConnection,
    existing_path: Path | None,
    incoming: int,
    window: tuple[datetime, datetime],
) -> MergeStats:
    """Compute the row-delta counts with bounded ``count(*)`` aggregates.

    Args:
        connection: Open DuckDB connection with ``new_frame`` registered.
        existing_path: Existing parquet path, or ``None`` on a first run.
        incoming: Row count of ``new_frame`` (already known to the caller).
        window: The half-open ``(window_start, window_end)`` bounds.

    Returns:
        ``MergeStats`` for the merge. ``final_row_count`` is
        ``kept_in_window + existing_retained`` (the ``UNION ALL`` does not
        deduplicate), avoiding a re-scan of the written file.
    """
    window_start, window_end = window
    kept_in_window = _count(
        connection,
        f'SELECT count(*) FROM new_frame WHERE {_WINDOW_PREDICATE}',
        [window_start, window_end],
    )

    if existing_path is None:
        existing_deleted = 0
        existing_retained = 0
    else:
        existing_deleted = _count(
            connection,
            f'SELECT count(*) FROM read_parquet(?) WHERE {_WINDOW_PREDICATE}',
            [str(existing_path), window_start, window_end],
        )
        existing_retained = _count(
            connection,
            f'SELECT count(*) FROM read_parquet(?) WHERE NOT ({_WINDOW_PREDICATE})',
            [str(existing_path), window_start, window_end],
        )

    return MergeStats(
        incoming=incoming,
        kept_in_window=kept_in_window,
        existing_deleted=existing_deleted,
        existing_retained=existing_retained,
        final_row_count=kept_in_window + existing_retained,
    )


def _count(
    connection: duckdb.DuckDBPyConnection, sql: str, params: list[object]
) -> int:
    """Run a single-column ``count(*)`` query and return the scalar as ``int``.

    A ``count(*)`` aggregate always returns exactly one row, so the
    ``fetchone() is None`` branch is unreachable in practice; it satisfies
    the typed (``Optional``) DuckDB cursor API without a spurious cast.
    """
    row = connection.execute(sql, params).fetchone()
    return 0 if row is None else int(row[0])


def _sql_literal(value: str) -> str:
    """Return ``value`` as a single-quoted SQL string literal, quotes escaped."""
    return "'" + value.replace("'", "''") + "'"
