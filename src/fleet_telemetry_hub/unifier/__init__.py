"""Pure helpers for the unifier transforms.

This package exposes the pure-function building blocks the per-provider
transforms compose into the unified utilization output: text
normalization for driver names / IDs / VINs, and interval-overlap
math for idle/driving-event reconciliation. The top-level ``unify``
function and the per-provider transforms themselves live in a
follow-on prompt.
"""

from fleet_telemetry_hub.unifier.overlap import (
    DriverIdentity,
    DrivingWindow,
    attribute_idle_driver,
    clip_overlap_seconds,
    compute_driving_duration_seconds,
    sum_overlap_seconds,
)
from fleet_telemetry_hub.unifier.text_normalization import (
    nfkc_strip,
    normalize_driver_name,
    nullify_tokens,
)

__all__: list[str] = [
    'DriverIdentity',
    'DrivingWindow',
    'attribute_idle_driver',
    'clip_overlap_seconds',
    'compute_driving_duration_seconds',
    'nfkc_strip',
    'normalize_driver_name',
    'nullify_tokens',
    'sum_overlap_seconds',
]
