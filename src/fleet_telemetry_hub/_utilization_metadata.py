"""Metadata-JSON derivation for ``UtilizationPipeline``.

Companion module to ``utilization_pipeline.py``: derives the locked
metadata-dict shape from precomputed whole-file aggregates plus the run
context, so the pipeline module stays close to its size target. The
aggregates come from a streaming DuckDB pass over the written parquet
(``compute_parquet_aggregates``), so metadata derivation never loads the
file into pandas.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import duckdb

__all__: list[str] = [
    'MetadataAggregates',
    'MetadataBuildContext',
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
    """Frozen bundle of inputs to ``build_metadata_dict``.

    Bundling keeps the metadata-builder signature under the PLR0913
    five-parameter cap and makes the writer side a single-argument call
    site.

    Attributes:
        aggregates: Whole-file aggregates over the written parquet.
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

    aggregates: MetadataAggregates
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


def build_metadata_dict(ctx: MetadataBuildContext) -> dict[str, Any]:
    """Derive the metadata JSON dict from a run context.

    Args:
        ctx: Bundled run inputs (whole-file aggregates, prior metadata,
            timing, window, provider-status lists).

    Returns:
        Dict matching the locked metadata-JSON shape, ready for
        ``json.dump``.
    """
    latest_event_end = ctx.aggregates.latest_event_end
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
        'row_count': ctx.aggregates.row_count,
        'by_company': ctx.aggregates.by_company,
        'providers_present': ctx.providers_present,
        'providers_skipped': ctx.providers_skipped,
        'providers_failed': ctx.providers_failed,
        'schema_version': _METADATA_SCHEMA_VERSION,
    }


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
