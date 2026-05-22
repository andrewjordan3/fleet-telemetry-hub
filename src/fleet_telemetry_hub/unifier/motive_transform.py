"""Transform a Motive utilization bundle into unified event rows.

Composes the unifier's pure helpers -- text normalization,
interval-overlap math, distance conversion -- into the Motive-side of
the per-provider transform layer. The function is pure aside from
logging: no I/O, no mutation of the input bundle, no global state.

Transformation rules (locked):
    - VIN is required; rows missing it after NFKC+strip are dropped
      with a WARNING.
    - Driver fields are normalized independently
      (NFKC+strip on the ID, plus ``unknown`` nullification on the
      name).
    - ``DrivingPeriod.type`` values ``'driving'``, ``'PC'``, and
      ``'YM'`` all map to ``EventType.DRIVING``.
    - Driving duration = total span minus the sum of overlapping idle
      windows on the same vehicle. Rows with ``<= 0`` computed
      duration are dropped with a WARNING.
    - Idle events with both driver fields null after normalization
      are gap-filled by attributing to the driver covering the most
      overlap; a null winner is allowed (the uncovered-time bucket).
    - Idle distance is always ``None``.
"""

import logging

from fleet_telemetry_hub.models.motive_responses import (
    DriverSummary,
    DrivingPeriod,
    IdleEvent,
)
from fleet_telemetry_hub.unifier.distance import km_to_miles
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
from fleet_telemetry_hub.utilization.motive_fetcher import MotiveUtilizationBundle

__all__: list[str] = ['transform_motive_bundle']

logger: logging.Logger = logging.getLogger(__name__)

_DROP_REASONS: tuple[str, ...] = ('null_vin', 'non_positive_duration')

# Soft-warning categories track events that produced a row but with
# one or more degraded enrichment fields (e.g. null odometer ->
# null distance). Distinct from ``drops``, which counts events that
# could not be represented in the unified schema at all.
_SOFT_WARNING_REASONS: tuple[str, ...] = ('null_odometer',)


def transform_motive_bundle(bundle: MotiveUtilizationBundle) -> list[UnifiedEventRow]:
    """
    Transform a Motive utilization bundle into unified event rows.

    Emits driving rows in source order first, then idle rows in source
    order. The top-level ``unify()`` orchestrator applies any final
    cross-provider sort.

    Args:
        bundle: Output of ``MotiveUtilizationFetcher.fetch``.

    Returns:
        List of ``UnifiedEventRow`` instances, one per valid source
        event. Rows that fail the VIN-required or
        non-positive-duration checks are dropped (with a WARNING) and
        do not appear in the output.
    """
    logger.info(
        'Starting Motive transform: date_range=%s, '
        '%d driving_periods, %d idle_events',
        bundle.date_range,
        len(bundle.driving_periods),
        len(bundle.idle_events),
    )

    idle_by_vehicle = _index_idle_by_vehicle(bundle.idle_events)
    driving_by_vehicle = _index_driving_by_vehicle(bundle.driving_periods)

    rows: list[UnifiedEventRow] = []
    drops: dict[str, int] = dict.fromkeys(_DROP_REASONS, 0)
    soft_warnings: dict[str, int] = dict.fromkeys(_SOFT_WARNING_REASONS, 0)

    for period in bundle.driving_periods:
        driving_row = _driving_period_to_row(
            period, idle_by_vehicle, bundle.company, drops, soft_warnings
        )
        if driving_row is not None:
            rows.append(driving_row)

    for event in bundle.idle_events:
        idle_row = _idle_event_to_row(
            event, driving_by_vehicle, bundle.company, drops
        )
        if idle_row is not None:
            rows.append(idle_row)

    logger.info(
        'Motive transform complete: %d rows emitted, drops=%s, soft_warnings=%s',
        len(rows),
        drops,
        soft_warnings,
    )
    return rows


def _index_idle_by_vehicle(events: list[IdleEvent]) -> dict[int, list[IdleEvent]]:
    """Index idle events by Motive-internal ``vehicle_id`` for O(1) lookup."""
    index: dict[int, list[IdleEvent]] = {}
    for event in events:
        index.setdefault(event.vehicle.vehicle_id, []).append(event)
    return index


def _index_driving_by_vehicle(
    periods: list[DrivingPeriod],
) -> dict[int, list[DrivingPeriod]]:
    """Index driving periods by Motive-internal ``vehicle_id`` for O(1) lookup."""
    index: dict[int, list[DrivingPeriod]] = {}
    for period in periods:
        index.setdefault(period.vehicle.vehicle_id, []).append(period)
    return index


def _extract_driver_identity(
    source_driver: DriverSummary | None,
) -> DriverIdentity:
    """
    Return normalized ``(driver_id, driver_name)`` from a Motive ``DriverSummary``.

    Motive's ``DriverSummary.driver_id`` is an integer; it is rendered
    to its decimal string form for the unified schema's string-typed
    ``driver_id`` column. The display name is composed from the
    ``full_name`` property and then run through ``normalize_driver_name``
    so the ``unknown`` sentinel (in any case / unicode variant) maps
    to ``None``.
    """
    if source_driver is None:
        return (None, None)
    driver_id = nfkc_strip(str(source_driver.driver_id))
    driver_name = normalize_driver_name(source_driver.full_name)
    return (driver_id, driver_name)


def _driving_period_to_row(
    period: DrivingPeriod,
    idle_by_vehicle: dict[int, list[IdleEvent]],
    company: str | None,
    drops: dict[str, int],
    soft_warnings: dict[str, int],
) -> UnifiedEventRow | None:
    """
    Convert a single ``DrivingPeriod`` into a unified row.

    Returns ``None`` (and increments ``drops``) for events that
    cannot be represented in the unified schema -- null VIN or
    non-positive computed duration. Returns a row (and increments
    ``soft_warnings``) for events that can be emitted but with a
    degraded enrichment field, currently only null odometer ->
    null distance.
    """
    vin = nfkc_strip(period.vehicle.vin)
    if not vin:
        logger.warning(
            'Dropping Motive driving period with null/empty VIN: '
            'vehicle_id=%d, number=%s, start=%s, end=%s',
            period.vehicle.vehicle_id,
            period.vehicle.number,
            period.start_time,
            period.end_time,
        )
        drops['null_vin'] += 1
        return None

    idle_candidates = idle_by_vehicle.get(period.vehicle.vehicle_id, [])
    idle_windows = [(event.start_time, event.end_time) for event in idle_candidates]
    duration_seconds = compute_driving_duration_seconds(
        period.start_time, period.end_time, idle_windows
    )
    if duration_seconds <= 0:
        logger.warning(
            'Dropping Motive driving period: idle fully covers driving. '
            'vehicle_id=%d, vin=%s, start=%s, end=%s, computed_seconds=%d',
            period.vehicle.vehicle_id,
            vin,
            period.start_time,
            period.end_time,
            duration_seconds,
        )
        drops['non_positive_duration'] += 1
        return None

    driver_id, driver_name = _extract_driver_identity(period.driver)

    kilometers_traveled = period.kilometers_traveled
    distance_miles: float | None
    if kilometers_traveled is None:
        logger.warning(
            'Motive driving period has null odometer reading(s); '
            'emitting row with null distance. '
            'vehicle_id=%d, vin=%s, start=%s, end=%s, '
            'start_kilometers=%s, end_kilometers=%s',
            period.vehicle.vehicle_id,
            vin,
            period.start_time,
            period.end_time,
            period.start_kilometers,
            period.end_kilometers,
        )
        soft_warnings['null_odometer'] += 1
        distance_miles = None
    else:
        distance_miles = km_to_miles(kilometers_traveled)

    return UnifiedEventRow(
        company=company,
        event_type=EventType.DRIVING,
        driver_id=driver_id,
        driver_name=driver_name,
        vin=vin,
        start_time_utc=period.start_time,
        end_time_utc=period.end_time,
        duration_seconds=duration_seconds,
        distance_miles=distance_miles,
    )


def _idle_event_to_row(
    event: IdleEvent,
    driving_by_vehicle: dict[int, list[DrivingPeriod]],
    company: str | None,
    drops: dict[str, int],
) -> UnifiedEventRow | None:
    """Convert a single ``IdleEvent`` into a unified row, or drop with a WARNING."""
    vin = nfkc_strip(event.vehicle.vin)
    if not vin:
        logger.warning(
            'Dropping Motive idle event with null/empty VIN: '
            'vehicle_id=%d, number=%s, start=%s, end=%s',
            event.vehicle.vehicle_id,
            event.vehicle.number,
            event.start_time,
            event.end_time,
        )
        drops['null_vin'] += 1
        return None

    driver_id, driver_name = _extract_driver_identity(event.driver)
    if driver_id is None and driver_name is None:
        driver_id, driver_name = _gap_fill_idle_driver(
            event, driving_by_vehicle.get(event.vehicle.vehicle_id, []), vin
        )

    duration_seconds = int((event.end_time - event.start_time).total_seconds())

    return UnifiedEventRow(
        company=company,
        event_type=EventType.IDLE,
        driver_id=driver_id,
        driver_name=driver_name,
        vin=vin,
        start_time_utc=event.start_time,
        end_time_utc=event.end_time,
        duration_seconds=duration_seconds,
        distance_miles=None,
    )


def _gap_fill_idle_driver(
    event: IdleEvent,
    driving_periods: list[DrivingPeriod],
    vin: str,
) -> DriverIdentity:
    """Attribute a driver to an unattributed idle event via overlap math."""
    driving_windows = [
        DrivingWindow(
            start=period.start_time,
            end=period.end_time,
            driver=_extract_driver_identity(period.driver),
        )
        for period in driving_periods
    ]
    winner, distribution, warn_flag = attribute_idle_driver(
        event.start_time, event.end_time, driving_windows
    )
    if warn_flag:
        logger.warning(
            'Multiple drivers overlap idle event on same vehicle. '
            'vehicle_id=%d, vin=%s, idle_start=%s, idle_end=%s, '
            'bucket_distribution=%s',
            event.vehicle.vehicle_id,
            vin,
            event.start_time,
            event.end_time,
            distribution,
        )
    return winner
