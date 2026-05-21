# pyright: reportUnknownVariableType=false
"""Metadata-JSON derivation for ``UtilizationPipeline``.

Internal companion module to ``utilization_pipeline.py``: extracts the
pure derivation logic (``_MetadataBuildContext`` plus the helpers that
turn a DataFrame + run context into the locked metadata-dict shape) so
the pipeline module itself stays close to the 150-200 line target.

The module-level pyright suppression mirrors the convention used in
``models/motive_responses.py``: pandas Series indexing returns
partially-Unknown types in strict mode, and the legible alternative is
a swarm of ``cast`` calls that CLAUDE.md explicitly discourages. The
pandas boundary in this file is small -- a few ``.max()`` / ``.tolist()``
calls -- so the trade-off is favorable.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import pandas as pd

__all__: list[str] = ['MetadataBuildContext', 'build_metadata_dict']

# Schema version literal embedded in the metadata file. Bump on a
# breaking metadata-shape change so downstream tooling can detect it.
_METADATA_SCHEMA_VERSION: int = 1

# Display key for the ``None`` company bucket in ``by_company`` -- a
# real company string will never use these literal parens, so the
# choice avoids any collision.
_NULL_COMPANY_KEY: str = '(null)'


@dataclass(frozen=True, slots=True)
class MetadataBuildContext:
    """Frozen bundle of inputs to ``build_metadata_dict``.

    Bundling keeps the metadata-builder signature under the PLR0913
    five-parameter cap and makes the writer side a single-argument
    call site.

    Attributes:
        df: The unified DataFrame about to be written to parquet.
        prior_metadata: The previous run's metadata dict, or ``None``
            on a first run. Used to preserve ``latest_data_date``
            across an empty result.
        run_started: Wall-clock UTC instant when ``run()`` began.
        run_completed: Wall-clock UTC instant just after the parquet
            write succeeded.
        start_date: Inclusive UTC start date of the fetch window.
        end_date: Inclusive UTC end date of the fetch window.
        providers_present: Provider names whose bundles were fetched
            and transformed successfully, in fixed (motive, samsara)
            order.
        providers_skipped: Provider names skipped because they were
            disabled or absent from config, in fixed order.
        providers_failed: Provider names that raised during fetch or
            transform, in fixed order.
    """

    df: pd.DataFrame
    prior_metadata: dict[str, Any] | None
    run_started: datetime
    run_completed: datetime
    start_date: date
    end_date: date
    providers_present: list[str]
    providers_skipped: list[str]
    providers_failed: list[str]


def build_metadata_dict(ctx: MetadataBuildContext) -> dict[str, Any]:
    """Derive the metadata JSON dict from a run context.

    Args:
        ctx: Bundled run inputs (DataFrame, prior metadata, timing,
            window, provider-status lists).

    Returns:
        Dict matching the locked metadata-JSON shape, ready for
        ``json.dump``.
    """
    return {
        'last_run_started_utc': _iso_z(ctx.run_started),
        'last_run_completed_utc': _iso_z(ctx.run_completed),
        'fetch_window_start_utc': _iso_z(
            datetime.combine(ctx.start_date, time.min, tzinfo=UTC)
        ),
        'fetch_window_end_utc': _iso_z(
            datetime.combine(
                ctx.end_date + timedelta(days=1), time.min, tzinfo=UTC
            )
        ),
        'latest_event_end_utc': _latest_event_end(ctx.df),
        'latest_data_date': _latest_data_date(ctx.df, ctx.prior_metadata),
        'row_count': len(ctx.df),
        'by_company': _by_company(ctx.df),
        'providers_present': ctx.providers_present,
        'providers_skipped': ctx.providers_skipped,
        'providers_failed': ctx.providers_failed,
        'schema_version': _METADATA_SCHEMA_VERSION,
    }


def _iso_z(value: datetime) -> str:
    """Return ISO-8601 with the ``Z`` UTC suffix (not ``+00:00``)."""
    return value.isoformat().replace('+00:00', 'Z')


def _latest_event_end(df: pd.DataFrame) -> str | None:
    """``df['end_time_utc'].max()`` as an ISO-Z string, or ``None`` if empty."""
    if len(df) == 0:
        return None
    return _iso_z(df['end_time_utc'].max().to_pydatetime())


def _latest_data_date(
    df: pd.DataFrame, prior_metadata: dict[str, Any] | None
) -> str | None:
    """Latest ``start_time_utc`` date, or the prior metadata's anchor on empty."""
    if len(df) > 0:
        return df['start_time_utc'].max().to_pydatetime().date().isoformat()
    if prior_metadata is not None:
        prior_value = prior_metadata.get('latest_data_date')
        return prior_value if isinstance(prior_value, str) else None
    return None


def _by_company(df: pd.DataFrame) -> dict[str, int]:
    """Per-company row counts; ``pd.NA`` entries land under ``'(null)'``."""
    counter: dict[str, int] = {}
    for value in df['company'].tolist():
        key = _NULL_COMPANY_KEY if pd.isna(value) else str(value)
        counter[key] = counter.get(key, 0) + 1
    return counter
