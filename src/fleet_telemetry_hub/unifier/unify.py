"""Top-level orchestrator: combine per-provider transforms into a single DataFrame.

The orchestrator is intentionally thin: it composes the existing
``transform_motive_bundle`` and ``transform_samsara_bundle`` outputs,
applies a stable ``(company, start_time_utc, event_type)`` sort, and
hands the result to ``build_dataframe``. Business logic stays in the
per-provider transforms.

Both bundles are required (possibly empty). Each contributes zero rows
when empty, so a both-empty call still yields the locked empty-schema
DataFrame -- downstream parquet consumers never see a different shape on
a no-data day.
"""

import logging
from collections import Counter

import pandas as pd

from fleet_telemetry_hub.unifier.motive_transform import transform_motive_bundle
from fleet_telemetry_hub.unifier.samsara_transform import transform_samsara_bundle
from fleet_telemetry_hub.unifier.schema import (
    UnifiedEventRow,
    build_dataframe,
    sort_unified_frame,
)
from fleet_telemetry_hub.utilization.motive_fetcher import MotiveUtilizationBundle
from fleet_telemetry_hub.utilization.samsara_fetcher import SamsaraUtilizationBundle

__all__: list[str] = ['unify']

logger: logging.Logger = logging.getLogger(__name__)

# Display sentinel for the ``None`` company key in the per-company
# count log; kept distinct from the literal string ``'(null)'`` a
# real company would never use.
_NULL_COMPANY_DISPLAY: str = '(null)'


def unify(
    motive_bundle: MotiveUtilizationBundle,
    samsara_bundle: SamsaraUtilizationBundle,
) -> pd.DataFrame:
    """
    Combine Motive and Samsara utilization bundles into a single typed DataFrame.

    Both bundles are required. An empty bundle contributes zero rows via
    its transform, so a both-empty call returns an empty DataFrame with
    the correct schema.

    Rows are sorted by ``(company, start_time_utc, event_type)``
    ascending. The sort is stable; within a tied key, Motive rows
    precede Samsara rows (the order rows are concatenated in).
    ``company=None`` rows sort before any non-null-company string.

    Args:
        motive_bundle: Motive utilization bundle (possibly empty).
        samsara_bundle: Samsara utilization bundle (possibly empty).

    Returns:
        Typed pandas DataFrame matching the unified schema. Empty
        (zero-row, correct columns and dtypes) when both bundles produce
        no rows.
    """
    _log_entry(motive_bundle, samsara_bundle)

    rows: list[UnifiedEventRow] = []
    rows.extend(transform_motive_bundle(motive_bundle))
    rows.extend(transform_samsara_bundle(samsara_bundle))

    _log_exit(rows)
    return sort_unified_frame(build_dataframe(rows))


def _log_entry(
    motive_bundle: MotiveUtilizationBundle,
    samsara_bundle: SamsaraUtilizationBundle,
) -> None:
    """Log INFO at the unify boundary with both bundles' date ranges."""
    logger.info(
        'unify called with both bundles: motive_date_range=%s, '
        'samsara_date_range=%s',
        motive_bundle.date_range,
        samsara_bundle.date_range,
    )


def _log_exit(rows: list[UnifiedEventRow]) -> None:
    """Log INFO with total row count and a per-company breakdown."""
    raw_counts = Counter(row.company for row in rows)
    by_company: dict[str, int] = {
        (company if company is not None else _NULL_COMPANY_DISPLAY): count
        for company, count in raw_counts.items()
    }
    logger.info(
        'unify produced %d rows, by_company=%s', len(rows), by_company
    )
