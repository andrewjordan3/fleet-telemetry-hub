"""Unified utilization-event schema: row type, columns, dtypes, builder.

This module defines the single flat fact-table shape that every
per-provider transform must emit. The output is self-contained -- no
star-schema joins downstream -- so the column set is locked and the
dtypes are pinned to pandas extension types that survive a parquet
round-trip without silent coercion.

The ``UnifiedEventRow`` dataclass validates each row at construction
time so transforms cannot accidentally emit naive timestamps,
reversed windows, empty VINs, or negative durations.
``build_dataframe`` materializes a typed pandas DataFrame from a
sequence of validated rows; empty input still produces a zero-row
DataFrame with the correct columns and dtypes so downstream consumers
never see schema drift.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

import pandas as pd

__all__: list[str] = [
    'COLUMNS',
    'DTYPES',
    'SORT_COLUMNS',
    'EventType',
    'UnifiedEventRow',
    'build_dataframe',
    'sort_unified_frame',
]


class EventType(StrEnum):
    """Discriminator for unified event rows."""

    DRIVING = 'driving'
    IDLE = 'idle'


@dataclass(frozen=True, slots=True)
class UnifiedEventRow:
    """
    Single row in the unified fleet telemetry table.

    Field order matches the output column order in ``COLUMNS``. The
    per-provider transforms construct one instance per valid source
    event; instances are immutable.

    Validation on construction:
        - ``start_time_utc`` and ``end_time_utc`` must be tz-aware.
        - ``start_time_utc <= end_time_utc``.
        - ``vin`` must be a non-empty string.
        - ``duration_seconds >= 0`` (zero-duration is allowed).
        - ``distance_miles is None`` or ``distance_miles >= 0.0``.

    Attributes:
        company: Identifier of the company the event was sourced from.
            ``None`` is acceptable when the upstream Provider had no
            company configured.
        event_type: ``EventType.DRIVING`` or ``EventType.IDLE``.
        driver_id: Provider-internal driver identifier, normalized;
            ``None`` when the event is unattributed.
        driver_name: Normalized driver display name; ``None`` when
            unattributed.
        vin: Vehicle Identification Number, NFKC-normalized and
            stripped; required (transforms drop rows without a VIN
            before construction).
        start_time_utc: Event start instant, tz-aware UTC.
        end_time_utc: Event end instant, tz-aware UTC.
        duration_seconds: Event duration in integer seconds. For
            driving rows this is the total span minus overlapping
            idle. For idle rows this is the raw span.
        distance_miles: Driving distance in miles rounded to one
            decimal place; ``None`` for idle rows (no distance
            concept).
    """

    company: str | None
    event_type: EventType
    driver_id: str | None
    driver_name: str | None
    vin: str
    start_time_utc: datetime
    end_time_utc: datetime
    duration_seconds: int
    distance_miles: float | None

    def __post_init__(self) -> None:
        """Validate the constructed row against the locked schema invariants."""
        if self.start_time_utc.tzinfo is None:
            raise TypeError(
                f'start_time_utc must be tz-aware, got naive: {self.start_time_utc!r}'
            )
        if self.end_time_utc.tzinfo is None:
            raise TypeError(
                f'end_time_utc must be tz-aware, got naive: {self.end_time_utc!r}'
            )
        if self.start_time_utc > self.end_time_utc:
            raise ValueError(
                f'start_time_utc ({self.start_time_utc}) must be <= '
                f'end_time_utc ({self.end_time_utc})'
            )
        if not self.vin:
            raise ValueError('vin must be a non-empty string')
        if self.duration_seconds < 0:
            raise ValueError(
                f'duration_seconds must be >= 0, got {self.duration_seconds}'
            )
        if self.distance_miles is not None and self.distance_miles < 0.0:
            raise ValueError(
                f'distance_miles must be >= 0.0 or None, got {self.distance_miles}'
            )


COLUMNS: tuple[str, ...] = (
    'company',
    'event_type',
    'driver_id',
    'driver_name',
    'vin',
    'start_time_utc',
    'end_time_utc',
    'duration_seconds',
    'distance_miles',
)
"""Column order of the unified output table."""


SORT_COLUMNS: tuple[str, ...] = ('company', 'start_time_utc', 'event_type')
"""Canonical sort key for the unified output table, in priority order."""


def sort_unified_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Sort a unified frame by the canonical key, nulls-first and stably.

    This is the single definition of the unified sort order, shared by
    the per-provider orchestrator and the incremental merge so the two
    can never drift. ``company`` nulls sort before any non-null company
    string (matching the orchestrator's historical ``None`` -> ``''``
    substitution). The sort is stable, so rows sharing a key keep their
    input order -- this preserves the Motive-before-Samsara concatenation
    order on ties. Column set and dtypes are unchanged.

    Args:
        frame: A unified DataFrame carrying the ``SORT_COLUMNS``.

    Returns:
        A new DataFrame sorted ascending by ``SORT_COLUMNS`` with a clean
        positional ``RangeIndex``.
    """
    return frame.sort_values(
        by=list(SORT_COLUMNS),
        ascending=True,
        na_position='first',
        kind='stable',
    ).reset_index(drop=True)


DTYPES: dict[str, Any] = {
    'company': pd.StringDtype(),
    'event_type': pd.StringDtype(),
    'driver_id': pd.StringDtype(),
    'driver_name': pd.StringDtype(),
    'vin': pd.StringDtype(),
    'start_time_utc': 'datetime64[ns, UTC]',
    'end_time_utc': 'datetime64[ns, UTC]',
    'duration_seconds': pd.Int64Dtype(),
    'distance_miles': pd.Float64Dtype(),
}
"""Pandas dtypes for the unified output table, keyed by column name."""


def build_dataframe(rows: Sequence[UnifiedEventRow]) -> pd.DataFrame:
    """
    Materialize a typed pandas DataFrame from validated unified rows.

    Empty input yields a zero-row DataFrame whose columns and dtypes
    still match the locked schema, so downstream parquet writers and
    BigQuery loaders see the same shape on no-data days.

    Args:
        rows: Sequence of pre-validated ``UnifiedEventRow`` instances.
            Order is preserved -- this layer does no sorting.

    Returns:
        DataFrame with the nine schema columns in ``COLUMNS`` order
        and dtypes from ``DTYPES``. ``EventType`` values appear as
        their underlying string value (``'driving'`` / ``'idle'``),
        not as enum instances.
    """
    data: dict[str, list[Any]] = {
        'company': [row.company for row in rows],
        'event_type': [row.event_type.value for row in rows],
        'driver_id': [row.driver_id for row in rows],
        'driver_name': [row.driver_name for row in rows],
        'vin': [row.vin for row in rows],
        'start_time_utc': [row.start_time_utc for row in rows],
        'end_time_utc': [row.end_time_utc for row in rows],
        'duration_seconds': [row.duration_seconds for row in rows],
        'distance_miles': [row.distance_miles for row in rows],
    }
    frame = pd.DataFrame(data, columns=list(COLUMNS))
    return frame.astype(DTYPES)
