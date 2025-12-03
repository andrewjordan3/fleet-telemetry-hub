# fleet_telemetry_hub/utils/samsara_funcs.py

import logging
from datetime import datetime
from typing import Any

from ..models import (
    DriverVehicleAssignment,
    GpsRecord,
    VehicleStatsHistoryRecord,
)

logger: logging.Logger = logging.getLogger(__name__)

# The timestamps for a location record and odometer reading must be below this
# amount to be merged
GPS_ODO_THRESHOLD_SEC: int = 1800  # Half hour

# =============================================================================
# Samsara-Specific Transformation Functions
# =============================================================================


def _find_engine_state_at_time(
    engine_states: list[dict[str, Any]],
    timestamp: datetime,
) -> str | None:
    """
    Find the engine state at a specific timestamp.

    Engine states are sparse - we need to find the most recent state
    before or at the given timestamp.

    Uses *linear* scan with early break. For typical ELD data (~10-50 engine
    state changes per day), this is faster than binary search due to:
    1. Small state lists (log2(50) ≈ 5.6 comparisons)
    2. Chronological GPS processing (early break after ~2-5 states)
    3. No overhead of extracting timestamps list

    If profiling shows this is a bottleneck (unlikely), consider bisect.

    Args:
        engine_states: List of {'time': datetime, 'state': str} records
                       (sorted by time ascending).
        timestamp: Target timestamp.

    Returns:
        Engine state string ('On'/'Off'/'Idle') or None if no data available.
    """
    if not engine_states:
        return None

    # Find the most recent state <= timestamp
    current_state: str | None = None
    for state in engine_states:
        if state['time'] <= timestamp:
            current_state = state['state']
        else:
            break  # States are sorted, no need to continue

    return current_state


def _find_odometer_at_time(
    odometer_readings: list[dict[str, Any]],
    timestamp: datetime,
) -> float | None:
    """
    Find the odometer reading closest to a specific timestamp.

    Odometer readings are sparse - we find the closest reading within
    a reasonable time window.

    Args:
        odometer_readings: List of {'time': datetime, 'value_miles': float} records.
        timestamp: Target timestamp.

    Returns:
        Odometer reading in miles or None if no reading within tolerance window.
    """
    if not odometer_readings:
        return None

    # Find the closest reading (before or after)
    closest: dict[str, Any] = min(
        odometer_readings,
        key=lambda r: abs((r['time'] - timestamp).total_seconds()),
    )

    # Only use if within 30 minutes of the GPS reading
    # (tighter tolerance than 1 hour since odometer updates are typically
    # more frequent than originally assumed)
    time_diff_seconds: float = abs((closest['time'] - timestamp).total_seconds())
    if time_diff_seconds <= GPS_ODO_THRESHOLD_SEC:  # 30 minutes
        return float(closest['value_miles'])

    return None


def _find_driver_for_timestamp(
    assignments: list[DriverVehicleAssignment],
    vehicle_id: str,
    timestamp: datetime,
) -> tuple[str | None, str | None]:
    """
    Find the driver assigned to a vehicle at a specific timestamp.

    Args:
        assignments: List of DriverVehicleAssignment records.
        vehicle_id: Samsara vehicle ID to search for.
        timestamp: Timestamp to check.

    Returns:
        Tuple of (driver_id, driver_name) or (None, None) if no assignment found.

    Notes:
        Filters out passenger assignments (only returns primary driver).
    """
    for assignment in assignments:
        if (
            assignment.vehicle.vehicle_id == vehicle_id
            and assignment.contains_timestamp(timestamp)
            and not assignment.is_passenger
        ):
            return assignment.driver.driver_id, assignment.driver.name

    return None, None


def flatten_samsara_gps(
    gps_record: GpsRecord,
    vehicle_stats: VehicleStatsHistoryRecord,
    engine_states: list[dict[str, Any]],
    odometer_readings: list[dict[str, Any]],
    assignments: list[DriverVehicleAssignment],
) -> dict[str, Any]:
    """
    Flatten a Samsara GPS record into the target schema.

    Args:
        gps_record: GPS record from vehicle stats history.
        vehicle_stats: Parent vehicle stats record for context.
        engine_states: Pre-processed engine state records (sorted by time).
        odometer_readings: Pre-processed odometer records.
        assignments: All driver-vehicle assignments for the time period.

    Returns:
        Dictionary with flattened fields matching target schema.
    """
    # Find engine state and odometer at this timestamp
    engine_state: str | None = _find_engine_state_at_time(
        engine_states, gps_record.time
    )
    odometer: float | None = _find_odometer_at_time(odometer_readings, gps_record.time)

    # Find driver assignment
    driver_id: str | None
    driver_name: str | None
    driver_id, driver_name = _find_driver_for_timestamp(
        assignments,
        vehicle_stats.vehicle_id,
        gps_record.time,
    )

    return {
        'provider': 'samsara',
        'provider_vehicle_id': vehicle_stats.vehicle_id,
        'vin': vehicle_stats.get_vin(),
        'fleet_number': vehicle_stats.name,
        'timestamp': gps_record.time,
        'latitude': gps_record.latitude,
        'longitude': gps_record.longitude,
        'speed_mph': gps_record.speed_miles_per_hour,
        'heading_degrees': gps_record.heading_degrees,
        'engine_state': engine_state,
        'driver_id': driver_id,
        'driver_name': driver_name,
        'location_description': gps_record.formatted_location,
        'odometer': odometer,
    }
