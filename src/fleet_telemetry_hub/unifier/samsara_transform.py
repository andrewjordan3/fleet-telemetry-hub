"""Transform a Samsara utilization bundle into unified event rows.

Mirrors the Motive transform's shape but adds two Samsara-specific
resolution steps: VIN comes from the bundle's ``vehicles`` dimension
(with an ``'unknown_vin'`` fallback so transient dim-data staleness
does not silently drop events) and driver names come from the
``drivers`` dimension (with an unresolvable-name warning when the
event's driver_id has no entry).

Trips emit driving rows; idling events emit idle rows. Idle
subtraction, idle gap-fill, and meters-to-miles distance conversion
follow the same patterns as the Motive transform so the unified
output is provider-agnostic from the orchestrator's perspective.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from fleet_telemetry_hub.models.samsara_responses import (
    IdlingEvent,
    SamsaraDriver,
    SamsaraVehicle,
)
from fleet_telemetry_hub.unifier.distance import meters_to_miles
from fleet_telemetry_hub.unifier.overlap import (
    DriverIdentity,
    DrivingWindow,
    attribute_idle_driver,
    compute_driving_duration_seconds,
)
from fleet_telemetry_hub.unifier.schema import EventType, UnifiedEventRow
from fleet_telemetry_hub.unifier.text_normalization import (
    nfkc_strip,
    normalize_driver_name,
)
from fleet_telemetry_hub.utilization.samsara_fetcher import SamsaraUtilizationBundle
from fleet_telemetry_hub.utilization.vehicle_trip import VehicleTrip

__all__: list[str] = ['transform_samsara_bundle']

logger: logging.Logger = logging.getLogger(__name__)

# Sentinel used in the unified output when a Samsara event's vehicle
# cannot be resolved to a VIN (either the vehicle_id is not in the
# dimension table at all or the dim entry's vin is null/empty). The
# row is still emitted so we don't lose events to transient dim
# staleness; the loud WARNING and the literal column value let
# downstream observers see and triage the gap.
_UNKNOWN_VIN_SENTINEL: str = 'unknown_vin'

_COUNTER_KEYS: tuple[str, ...] = (
    'unknown_vin_fallbacks',
    'unresolvable_drivers',
    'non_positive_durations_dropped',
)


@dataclass(frozen=True, slots=True)
class _SamsaraContext:
    """Per-bundle lookup maps, counters, and company name for row builders."""

    vin_by_vehicle_id: dict[str, str | None]
    name_by_driver_id: dict[str, str]
    idling_by_vehicle_id: dict[str, list[IdlingEvent]]
    trips_by_vehicle_id: dict[str, list[VehicleTrip]]
    company: str | None
    counters: dict[str, int]


def transform_samsara_bundle(
    bundle: SamsaraUtilizationBundle,
) -> list[UnifiedEventRow]:
    """
    Transform a Samsara utilization bundle into unified event rows.

    Implements VIN resolution with ``unknown_vin`` fallback, driver
    name resolution with an unresolvable-name warning, idle
    subtraction from trip durations, idle gap-fill via
    ``attribute_idle_driver``, and meters-to-miles distance
    conversion. Trips emit driving rows; idling events emit idle
    rows.

    Args:
        bundle: ``SamsaraUtilizationBundle`` from
            ``SamsaraUtilizationFetcher``.

    Returns:
        List of ``UnifiedEventRow`` instances. An empty bundle yields
        an empty list.
    """
    logger.info(
        'Starting Samsara transform: date_range=%s, %d vehicles, '
        '%d drivers, %d trips, %d idling_events',
        bundle.date_range,
        len(bundle.vehicles),
        len(bundle.drivers),
        len(bundle.trips),
        len(bundle.idling_events),
    )

    ctx = _SamsaraContext(
        vin_by_vehicle_id=_build_vin_map(bundle.vehicles),
        name_by_driver_id=_build_driver_name_map(bundle.drivers),
        idling_by_vehicle_id=_index_idling_by_vehicle(bundle.idling_events),
        trips_by_vehicle_id=_index_trips_by_vehicle(bundle.trips),
        company=bundle.company,
        counters=dict.fromkeys(_COUNTER_KEYS, 0),
    )

    rows: list[UnifiedEventRow] = []

    for vehicle_trip in bundle.trips:
        trip_row = _trip_to_row(vehicle_trip, ctx)
        if trip_row is not None:
            rows.append(trip_row)

    for idling_event in bundle.idling_events:
        idle_row = _idling_event_to_row(idling_event, ctx)
        if idle_row is not None:
            rows.append(idle_row)

    logger.info(
        'Samsara transform complete: %d rows emitted, counters=%s',
        len(rows),
        ctx.counters,
    )
    return rows


def _build_vin_map(vehicles: list[SamsaraVehicle]) -> dict[str, str | None]:
    """Map Samsara ``vehicle_id`` → ``vin`` (which itself may be ``None``)."""
    return {vehicle.vehicle_id: vehicle.vin for vehicle in vehicles}


def _build_driver_name_map(drivers: list[SamsaraDriver]) -> dict[str, str]:
    """Map Samsara ``driver_id`` → driver name."""
    return {driver.driver_id: driver.name for driver in drivers}


def _index_idling_by_vehicle(
    events: list[IdlingEvent],
) -> dict[str, list[IdlingEvent]]:
    """Index idling events by ``asset.asset_id`` for O(1) lookup during trip conversion."""
    index: dict[str, list[IdlingEvent]] = {}
    for event in events:
        index.setdefault(event.asset.asset_id, []).append(event)
    return index


def _index_trips_by_vehicle(
    trips: list[VehicleTrip],
) -> dict[str, list[VehicleTrip]]:
    """Index trips by ``vehicle_id`` for O(1) lookup during idle gap-fill."""
    index: dict[str, list[VehicleTrip]] = {}
    for vehicle_trip in trips:
        index.setdefault(vehicle_trip.vehicle_id, []).append(vehicle_trip)
    return index


def _resolve_vin(
    vehicle_id: str,
    ctx: _SamsaraContext,
    *,
    event_kind: str,
    event_identifier: str,
    event_window: tuple[datetime, datetime],
) -> str:
    """
    Resolve a Samsara ``vehicle_id`` to a VIN, falling back to ``'unknown_vin'``.

    Two failure modes both fall back to the sentinel:
        1. ``vehicle_id`` not in ``vin_by_vehicle_id``
        2. ``vehicle_id`` is in the map but the looked-up VIN is
           null/empty after NFKC-strip.

    Increments ``counters['unknown_vin_fallbacks']`` on either
    failure and logs a WARNING; the row is still emitted by the
    caller.
    """
    in_dim_table = vehicle_id in ctx.vin_by_vehicle_id
    raw_vin = ctx.vin_by_vehicle_id.get(vehicle_id)
    normalized = nfkc_strip(raw_vin) if raw_vin is not None else None
    if normalized is None:
        ctx.counters['unknown_vin_fallbacks'] += 1
        event_start, event_end = event_window
        logger.warning(
            'Samsara VIN unresolvable, falling back to unknown_vin: '
            'event_kind=%s, event_identifier=%s, vehicle_id=%s, '
            'in_dim_table=%s, event_window=[%s, %s]',
            event_kind,
            event_identifier,
            vehicle_id,
            in_dim_table,
            event_start,
            event_end,
        )
        return _UNKNOWN_VIN_SENTINEL
    return normalized


def _resolve_driver(
    source_driver_id: str | None,
    ctx: _SamsaraContext,
    *,
    event_kind: str,
    event_identifier: str,
) -> DriverIdentity:
    """
    Resolve a Samsara ``driver_id`` to ``(driver_id, driver_name)``.

    - ``source_driver_id`` ``None`` or empty after NFKC-strip →
      ``(None, None)``, no warning.
    - Normalized id present in ``name_by_driver_id`` →
      ``(id, normalize_driver_name(name))``. The returned name may
      still be ``None`` if the dim's name itself normalizes to
      ``'unknown'`` or empty.
    - Normalized id non-null but absent from the dim →
      ``(id, None)``, WARNING logged, counter incremented.
    """
    normalized_id = nfkc_strip(source_driver_id)
    if normalized_id is None:
        return (None, None)
    if normalized_id not in ctx.name_by_driver_id:
        ctx.counters['unresolvable_drivers'] += 1
        logger.warning(
            'Samsara driver name unresolvable: event_kind=%s, '
            'event_identifier=%s, driver_id=%s',
            event_kind,
            event_identifier,
            normalized_id,
        )
        return (normalized_id, None)
    driver_name = normalize_driver_name(ctx.name_by_driver_id[normalized_id])
    return (normalized_id, driver_name)


def _trip_to_row(
    vehicle_trip: VehicleTrip, ctx: _SamsaraContext
) -> UnifiedEventRow | None:
    """Convert a single ``VehicleTrip`` into a unified driving row, or drop with WARNING."""
    trip = vehicle_trip.trip
    vehicle_id = vehicle_trip.vehicle_id
    event_identifier = trip.trip_id if trip.trip_id is not None else '<no_trip_id>'
    vin = _resolve_vin(
        vehicle_id,
        ctx,
        event_kind='trip',
        event_identifier=event_identifier,
        event_window=(trip.start_time, trip.end_time),
    )
    driver_id, driver_name = _resolve_driver(
        trip.driver_id,
        ctx,
        event_kind='trip',
        event_identifier=event_identifier,
    )

    idling_candidates = ctx.idling_by_vehicle_id.get(vehicle_id, [])
    idle_windows = [
        (
            event.start_time,
            event.start_time + timedelta(milliseconds=event.duration_milliseconds),
        )
        for event in idling_candidates
    ]
    duration_seconds = compute_driving_duration_seconds(
        trip.start_time, trip.end_time, idle_windows
    )
    if duration_seconds <= 0:
        ctx.counters['non_positive_durations_dropped'] += 1
        logger.warning(
            'Dropping Samsara trip: idle fully covers trip duration. '
            'trip_id=%s, vehicle_id=%s, vin=%s, start=%s, end=%s, '
            'computed_seconds=%d',
            event_identifier,
            vehicle_id,
            vin,
            trip.start_time,
            trip.end_time,
            duration_seconds,
        )
        return None

    return UnifiedEventRow(
        company=ctx.company,
        event_type=EventType.DRIVING,
        driver_id=driver_id,
        driver_name=driver_name,
        vin=vin,
        start_time_utc=trip.start_time,
        end_time_utc=trip.end_time,
        duration_seconds=duration_seconds,
        distance_miles=meters_to_miles(trip.distance_meters),
    )


def _idling_event_to_row(
    event: IdlingEvent, ctx: _SamsaraContext
) -> UnifiedEventRow | None:
    """Convert a single ``IdlingEvent`` into a unified idle row."""
    end_time_utc = event.start_time + timedelta(
        milliseconds=event.duration_milliseconds
    )
    vehicle_id = event.asset.asset_id
    vin = _resolve_vin(
        vehicle_id,
        ctx,
        event_kind='idling_event',
        event_identifier=event.event_uuid,
        event_window=(event.start_time, end_time_utc),
    )
    driver_id, driver_name = _resolve_driver(
        event.operator.operator_id,
        ctx,
        event_kind='idling_event',
        event_identifier=event.event_uuid,
    )
    if driver_id is None and driver_name is None:
        driver_id, driver_name = _gap_fill_idle_driver(
            event=event,
            end_time_utc=end_time_utc,
            ctx=ctx,
        )

    duration_seconds = event.duration_milliseconds // 1000

    return UnifiedEventRow(
        company=ctx.company,
        event_type=EventType.IDLE,
        driver_id=driver_id,
        driver_name=driver_name,
        vin=vin,
        start_time_utc=event.start_time,
        end_time_utc=end_time_utc,
        duration_seconds=duration_seconds,
        distance_miles=None,
    )


def _gap_fill_idle_driver(
    *,
    event: IdlingEvent,
    end_time_utc: datetime,
    ctx: _SamsaraContext,
) -> DriverIdentity:
    """Attribute a driver to an unattributed idling event via overlap math."""
    vehicle_id = event.asset.asset_id
    candidate_trips = ctx.trips_by_vehicle_id.get(vehicle_id, [])
    driving_windows = [
        DrivingWindow(
            start=vt.trip.start_time,
            end=vt.trip.end_time,
            driver=_resolve_driver(
                vt.trip.driver_id,
                ctx,
                event_kind='trip',
                event_identifier=(
                    vt.trip.trip_id
                    if vt.trip.trip_id is not None
                    else '<no_trip_id>'
                ),
            ),
        )
        for vt in candidate_trips
    ]
    winner, distribution, warn_flag = attribute_idle_driver(
        event.start_time, end_time_utc, driving_windows
    )
    if warn_flag:
        logger.warning(
            'Multiple drivers overlap idling event on same vehicle. '
            'event_uuid=%s, vehicle_id=%s, idle_start=%s, idle_end=%s, '
            'bucket_distribution=%s',
            event.event_uuid,
            vehicle_id,
            event.start_time,
            end_time_utc,
            distribution,
        )
    return winner
