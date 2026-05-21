"""Pure helpers and per-provider transforms for the unified utilization output.

This package exposes the building blocks the per-provider transforms
compose into the unified utilization output: text normalization for
driver names / IDs / VINs, interval-overlap math for idle/driving
event reconciliation, distance-unit conversion, and the locked
output schema with its DataFrame builder. The Motive transform lives
here too; the Samsara transform and the top-level ``unify`` function
land in a follow-on prompt.
"""

from fleet_telemetry_hub.unifier.distance import km_to_miles, meters_to_miles
from fleet_telemetry_hub.unifier.motive_transform import transform_motive_bundle
from fleet_telemetry_hub.unifier.overlap import (
    DriverIdentity,
    DrivingWindow,
    attribute_idle_driver,
    clip_overlap_seconds,
    compute_driving_duration_seconds,
    sum_overlap_seconds,
)
from fleet_telemetry_hub.unifier.schema import (
    COLUMNS,
    DTYPES,
    EventType,
    UnifiedEventRow,
    build_dataframe,
)
from fleet_telemetry_hub.unifier.text_normalization import (
    nfkc_strip,
    normalize_driver_name,
    nullify_tokens,
)

__all__: list[str] = [
    'COLUMNS',
    'DTYPES',
    'DriverIdentity',
    'DrivingWindow',
    'EventType',
    'UnifiedEventRow',
    'attribute_idle_driver',
    'build_dataframe',
    'clip_overlap_seconds',
    'compute_driving_duration_seconds',
    'km_to_miles',
    'meters_to_miles',
    'nfkc_strip',
    'normalize_driver_name',
    'nullify_tokens',
    'sum_overlap_seconds',
    'transform_motive_bundle',
]
