"""Metadata-JSON derivation and persistence for ``UtilizationPipeline``.

Companion module to ``utilization_pipeline.py``. The pure layer
(``compute_parquet_aggregates``, ``build_metadata_dict``) derives the
locked metadata-dict shape from whole-file aggregates plus the run
context, never loading the parquet into pandas. ``MetadataStore`` wraps
that layer with the on-disk concerns -- loading the prior metadata,
computing the aggregates over the sibling ``data.parquet``, and writing
the result atomically -- so the pipeline module stays close to its size
target and owns no persistence logic.
"""

import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import duckdb

__all__: list[str] = [
    'MetadataAggregates',
    'MetadataBuildContext',
    'MetadataStore',
    'build_metadata_dict',
    'compute_parquet_aggregates',
]

logger: logging.Logger = logging.getLogger(__name__)

# Schema version literal embedded in the metadata file. Bump on a
# breaking metadata-shape change so downstream tooling can detect it.
_METADATA_SCHEMA_VERSION: int = 1

# Display key for the ``None`` company bucket in ``by_company`` -- a real
# company string will never use these literal parens, so the choice
# avoids any collision.
_NULL_COMPANY_KEY: str = '(null)'

# Anchor for rebuilding ``max(end_time_utc)`` from epoch microseconds.
_EPOCH_UTC: datetime = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class MetadataAggregates:
    """Whole-file aggregates derived by a streaming DuckDB pass.

    Attributes:
        row_count: Total rows in the written parquet.
        by_company: Per-company row counts; the ``None`` company bucket
            lands under the ``'(null)'`` key.
        latest_event_end: The exact ``max(end_time_utc)`` as a tz-aware
            UTC ``datetime``, or ``None`` when the file is empty.
    """

    row_count: int
    by_company: dict[str, int]
    latest_event_end: datetime | None


@dataclass(frozen=True, slots=True)
class MetadataBuildContext:
    """Frozen run-context bundle for ``build_metadata_dict``.

    Carries everything the metadata shape needs *except* the whole-file
    aggregates, which ``MetadataStore`` derives from the written parquet
    and passes alongside this context. Bundling keeps the builder signature
    under the PLR0913 five-parameter cap.

    Attributes:
        prior_metadata: The previous run's metadata dict, or ``None`` on
            a first run. Used to preserve ``latest_data_date`` when the
            file has no events.
        run_started: Wall-clock UTC instant when ``run()`` began.
        run_completed: Wall-clock UTC instant just after the parquet
            write succeeded.
        start_date: Inclusive UTC start date of the fetch window.
        end_date: Inclusive UTC end date of the fetch window.
        providers_present: Provider names whose bundles were fetched and
            transformed successfully, in fixed (motive, samsara) order.
        providers_skipped: Provider names skipped because they were
            disabled or absent from config, in fixed order.
        providers_failed: Provider names that raised during fetch or
            transform, in fixed order.
    """

    prior_metadata: dict[str, Any] | None
    run_started: datetime
    run_completed: datetime
    start_date: date
    end_date: date
    providers_present: list[str]
    providers_skipped: list[str]
    providers_failed: list[str]


def compute_parquet_aggregates(path: Path) -> MetadataAggregates:
    """Derive whole-file metadata aggregates from a parquet via DuckDB.

    Streams ``count(*)``, ``max(end_time_utc)`` and a per-company count
    over ``path`` without materializing it in pandas. ``max(end_time_utc)``
    is fetched as epoch microseconds (a plain integer), which sidesteps
    DuckDB's pytz requirement for materializing a ``TIMESTAMPTZ`` value and
    preserves microsecond precision exactly; it is rebuilt into a UTC
    ``datetime`` here.

    Args:
        path: Path to the written utilization parquet.

    Returns:
        ``MetadataAggregates`` with whole-file ``row_count`` /
        ``by_company`` and the exact ``latest_event_end`` (``None`` when
        the file has zero rows).

    Side Effects:
        Reads ``path`` from disk via DuckDB. Logs the aggregates at DEBUG.
    """
    with duckdb.connect() as connection:
        # A two-aggregate query always returns exactly one row; the
        # ``or (0, None)`` fallback only guards the typed (Optional) cursor
        # API and happens to match the empty-file semantics anyway.
        row_count, latest_end_epoch_us = connection.execute(
            'SELECT count(*), epoch_us(max(end_time_utc)) FROM read_parquet(?)',
            [str(path)],
        ).fetchone() or (0, None)
        company_counts = connection.execute(
            'SELECT company, count(*) FROM read_parquet(?) GROUP BY company',
            [str(path)],
        ).fetchall()

    latest_event_end = (
        None
        if latest_end_epoch_us is None
        else _EPOCH_UTC + timedelta(microseconds=latest_end_epoch_us)
    )
    by_company: dict[str, int] = {}
    for company, count in company_counts:
        key = _NULL_COMPANY_KEY if company is None else str(company)
        by_company[key] = int(count)

    aggregates = MetadataAggregates(
        row_count=int(row_count),
        by_company=by_company,
        latest_event_end=latest_event_end,
    )
    logger.debug(
        'parquet aggregates: row_count=%d by_company=%s latest_event_end=%s',
        aggregates.row_count,
        aggregates.by_company,
        aggregates.latest_event_end,
    )
    return aggregates


def build_metadata_dict(
    aggregates: MetadataAggregates, ctx: MetadataBuildContext
) -> dict[str, Any]:
    """Derive the metadata JSON dict from whole-file aggregates and run context.

    Args:
        aggregates: Whole-file aggregates over the written parquet.
        ctx: Bundled run inputs (prior metadata, timing, window,
            provider-status lists).

    Returns:
        Dict matching the locked metadata-JSON shape, ready for
        ``json.dump``.
    """
    latest_event_end = aggregates.latest_event_end
    return {
        'last_run_started_utc': _iso_z(ctx.run_started),
        'last_run_completed_utc': _iso_z(ctx.run_completed),
        'fetch_window_start_utc': _iso_z(
            datetime.combine(ctx.start_date, time.min, tzinfo=UTC)
        ),
        'fetch_window_end_utc': _iso_z(
            datetime.combine(ctx.end_date + timedelta(days=1), time.min, tzinfo=UTC)
        ),
        'latest_event_end_utc': (
            None if latest_event_end is None else _iso_z(latest_event_end)
        ),
        'latest_data_date': _latest_data_date(latest_event_end, ctx.prior_metadata),
        'row_count': aggregates.row_count,
        'by_company': aggregates.by_company,
        'providers_present': ctx.providers_present,
        'providers_skipped': ctx.providers_skipped,
        'providers_failed': ctx.providers_failed,
        'schema_version': _METADATA_SCHEMA_VERSION,
    }


class MetadataStore:
    """Owns ``metadata.json`` for one utilization output directory.

    Reads the prior metadata, computes whole-file aggregates over the
    sibling ``data.parquet``, builds the locked metadata shape, and writes
    it atomically. The pure functions it calls
    (``compute_parquet_aggregates``, ``build_metadata_dict``) stay
    side-effect-free; this class is the only side-effecting layer.

    Attributes:
        parquet_dir: The output directory holding ``data.parquet`` and
            ``metadata.json`` (read-only).
    """

    def __init__(self, parquet_dir: Path) -> None:
        """Bind the store to ``parquet_dir`` and derive its file paths.

        Args:
            parquet_dir: The ``{parquet_path}/utilization/`` directory that
                holds (or will hold) ``data.parquet`` and ``metadata.json``.
        """
        self._parquet_dir: Path = parquet_dir
        self._metadata_path: Path = parquet_dir / 'metadata.json'
        self._parquet_path: Path = parquet_dir / 'data.parquet'

    def load(self) -> dict[str, Any] | None:
        """Return the prior metadata dict, or ``None`` if no file exists.

        Returns:
            The parsed metadata dict, or ``None`` on a first run with no
            metadata file yet.

        Raises:
            json.JSONDecodeError: If the metadata file exists but is
                malformed. Never silently treated as a first run.

        Side Effects:
            Reads ``metadata.json`` from disk when present.
        """
        if not self._metadata_path.exists():
            return None
        with self._metadata_path.open('r', encoding='utf-8') as metadata_file:
            loaded: dict[str, Any] = json.load(metadata_file)
            return loaded

    def write(self, ctx: MetadataBuildContext) -> None:
        """Build and atomically write the metadata JSON for this run.

        Computes whole-file aggregates over the sibling ``data.parquet``,
        builds the metadata dict, and writes it via a temp file + rename so
        a failed write leaves any prior metadata intact.

        Args:
            ctx: The run-context bundle (timing, window, provider lists,
                prior metadata) for this run.

        Side Effects:
            Reads ``data.parquet``; creates the output directory; writes
            ``metadata.json`` via a temp file and atomic rename. Logs the
            write at INFO.
        """
        aggregates: MetadataAggregates = compute_parquet_aggregates(self._parquet_path)
        metadata: dict[str, Any] = build_metadata_dict(aggregates, ctx)
        self._parquet_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json.tmp',
            dir=self._parquet_dir,
            delete=False,
            encoding='utf-8',
        ) as tmp_file:
            tmp_path: Path = Path(tmp_file.name)
            json.dump(metadata, tmp_file, indent=2, sort_keys=True)
        tmp_path.replace(self._metadata_path)
        logger.info('Wrote metadata to %s', self._metadata_path)


def _iso_z(value: datetime) -> str:
    """Return ISO-8601 with the ``Z`` UTC suffix (not ``+00:00``)."""
    return value.isoformat().replace('+00:00', 'Z')


def _latest_data_date(
    latest_event_end: datetime | None, prior_metadata: dict[str, Any] | None
) -> str | None:
    """The UTC date of ``latest_event_end``, or the prior anchor when empty.

    ``latest_data_date`` is the next run's window anchor. When the file
    has no events (``latest_event_end is None``) the prior metadata's
    anchor is preserved so an empty run does not reset it.
    """
    if latest_event_end is not None:
        return latest_event_end.date().isoformat()
    if prior_metadata is not None:
        prior_value = prior_metadata.get('latest_data_date')
        return prior_value if isinstance(prior_value, str) else None
    return None
