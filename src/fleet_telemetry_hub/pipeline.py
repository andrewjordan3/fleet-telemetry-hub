# fleet_telemetry_hub/pipeline.py
"""
Vehicle data pipeline for fetching and transforming telemetry data.

This module provides a pipeline to fetch vehicle location and telemetry data
from multiple providers (Motive and Samsara), flatten nested structures, and
combine them into a unified pandas DataFrame.
"""

import logging
from datetime import date, datetime
from typing import Any

import pandas as pd

from .models.motive_responses import Vehicle, VehicleLocation, VehicleLocationType
from .models.samsara_responses import (
    DriverVehicleAssignment,
    GpsRecord,
    VehicleStatsHistoryRecord,
)
from .provider import Provider

logger: logging.Logger = logging.getLogger(__name__)

# Threshold for GPS drift while still considering a truck idle
IDLING_THRESHOLD_MPH: float = 3.0
# The timestamps for a location record and odometer reading must be below this
# amount to be merged
GPS_ODO_THRESHOLD_SEC: int = 1800  # Half hour
# =============================================================================
# Motive-Specific Transformation Functions
# =============================================================================


def _convert_motive_speed_to_mph(
    speed: float | None,
    metric_units: bool,
) -> float | None:
    """
    Convert Motive speed to MPH for unified schema.

    Args:
        speed: Speed value from Motive API.
        metric_units: Whether vehicle reports in metric (km/h).

    Returns:
        Speed in MPH, or None if input was None.
    """
    if speed is None:
        return None

    if metric_units:
        return speed * 0.621371  # km/h to mph

    return speed  # Already in mph


def _derive_motive_engine_state(
    location_type: VehicleLocationType | None,
    speed_mph: float | None,
) -> str | None:
    """
    Derive engine state from Motive location_type and speed.

    Args:
        location_type: Motive's location event type.
        speed_mph: Vehicle speed in MPH.

    Returns:
        'On', 'Off', 'Idle', or None if indeterminate.

    Logic:
        The following priority determines the state:
        - If inputs are None, result is None.
        - vehicle_stopped event implies 'Idle'.
        - ignition_off event implies 'Off'.
        - Any event with Speed > Threshold implies 'On'.
        - Any event with Speed <= Threshold implies 'Idle'.
        - vehicle_moving/driving events (without speed data) imply 'On'.
        - ignition_on event (without speed data) implies 'Idle'.
        - All other cases default to 'Off'.
    """
    # Pre-check: If both passed variables are None then return None.
    if location_type is None and speed_mph is None:
        return None

    type_value: str | None = location_type.value if location_type is not None else None
    # Initialize the return variable
    derived_status: str = 'Off'

    # We match on a tuple of (type_value, speed_mph)
    match (type_value, speed_mph):
        # If type_value is 'vehicle_stopped' return Idle.
        case ('vehicle_stopped', _):
            derived_status = 'Idle'

        # If type_value is 'ignition_off' return Off.
        case ('ignition_off', _):
            derived_status = 'Off'

        # If the speed_mph is greater than IDLING_THRESHOLD return On.
        # We use a 'guard' (if condition) here. We must ensure s is not None before comparing.
        case (_, s) if s is not None and s > IDLING_THRESHOLD_MPH:
            derived_status = 'On'

        # If the speed_mph is less than or equal to IDLING_THRESHOLD return Idle.
        # We use a 'guard' (if condition) here. We must ensure s is not None before comparing.
        case (_, s) if s is not None and s <= IDLING_THRESHOLD_MPH:
            derived_status = 'Idle'

        # If type_value is 'vehicle_moving' or 'vehicle_driving' return On.
        # We can use the pipe | character to match multiple strings.
        case ('vehicle_moving' | 'vehicle_driving', _):
            derived_status = 'On'

        # If type_value is 'ignition_on' return Idle.
        case ('ignition_on', _):
            derived_status = 'Idle'

        # Default/Catch-all: Return 'Off'.
        # The underscore _ acts as a wildcard that matches anything not caught above.
        case _:
            derived_status = 'Off'

    return derived_status


def _get_best_motive_odometer(location: VehicleLocation) -> float | None:
    """
    Get the best available odometer reading from Motive.

    Prefers ECM-reported value (true_odometer) over calculated value.

    Args:
        location: VehicleLocation record.

    Returns:
        Odometer reading in miles, or None if unavailable.
    """
    # Prefer ECM-reported odometer (more accurate)
    if location.true_odometer is not None:
        return location.true_odometer

    # Fall back to calculated odometer
    return location.odometer


def _flatten_motive_location(
    location: VehicleLocation,
    vehicle: Vehicle,
) -> dict[str, Any]:
    """
    Flatten a Motive VehicleLocation into the target schema.

    Args:
        location: VehicleLocation record from Motive.
        vehicle: Vehicle record for context (provides VIN, fleet number, units).

    Returns:
        Dictionary with flattened fields matching target schema.
    """
    # Convert speed to MPH if needed
    speed_mph: float | None = _convert_motive_speed_to_mph(
        location.speed, vehicle.metric_units
    )

    # Derive engine state
    engine_state: str | None = _derive_motive_engine_state(
        location.location_type, speed_mph
    )

    # Get best odometer reading
    odometer: float | None = _get_best_motive_odometer(location)

    return {
        'provider': 'motive',
        'provider_vehicle_id': str(vehicle.vehicle_id),
        'vin': vehicle.vin,
        'fleet_number': vehicle.number,
        'timestamp': location.located_at,
        'latitude': location.latitude,
        'longitude': location.longitude,
        'speed_mph': speed_mph,
        'heading_degrees': location.bearing,
        'engine_state': engine_state,
        'driver_id': str(location.driver.driver_id) if location.driver else None,
        'driver_name': location.driver.full_name if location.driver else None,
        'location_description': location.description,
        'odometer': odometer,
    }


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


def _flatten_samsara_gps(
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


# =============================================================================
# Provider-Specific Pipeline Functions
# =============================================================================


def _fetch_motive_data(
    provider: Provider,
    start_datetime: datetime,
    end_datetime: datetime,
) -> list[dict[str, Any]]:
    """
    Fetch and transform Motive vehicle location data.

    Args:
        provider: Motive provider instance.
        start_datetime: Start of time range (inclusive).
        end_datetime: End of time range (inclusive).

    Returns:
        List of flattened records matching target schema.
    """
    logger.info('Fetching Motive data from %s to %s', start_datetime, end_datetime)

    # Convert datetime to date for Motive API (uses date-only parameters)
    start_date: date = start_datetime.date()
    end_date: date = end_datetime.date()

    records: list[dict[str, Any]] = []

    # Fetch all vehicles first
    logger.info('Fetching Motive vehicles...')
    vehicles: list[Vehicle] = list(provider.fetch_all('vehicles'))
    logger.info('Found %d Motive vehicles', len(vehicles))

    # For each vehicle, fetch location history
    for vehicle in vehicles:
        logger.debug(
            'Fetching locations for Motive vehicle %s (%s)',
            vehicle.number,
            vehicle.vehicle_id,
        )

        try:
            locations: list[VehicleLocation] = list(
                provider.fetch_all(
                    'vehicle_locations',
                    vehicle_id=vehicle.vehicle_id,
                    start_date=start_date,
                    end_date=end_date,
                )
            )

            logger.debug(
                'Found %d locations for vehicle %s', len(locations), vehicle.number
            )

            # Flatten each location record
            for location in locations:
                # Filter by datetime (Motive API uses dates, so we filter to exact times)
                if (
                    location.located_at < start_datetime
                    or location.located_at > end_datetime
                ):
                    continue

                records.append(_flatten_motive_location(location, vehicle))

        except Exception:
            logger.exception(
                'Error fetching locations for Motive vehicle %s', vehicle.vehicle_id
            )
            continue

    logger.info('Fetched %d Motive location records', len(records))
    return records


def _fetch_samsara_data(
    provider: Provider,
    start_datetime: datetime,
    end_datetime: datetime,
) -> list[dict[str, Any]]:
    """
    Fetch and transform Samsara vehicle telemetry data.

    Args:
        provider: Samsara provider instance.
        start_datetime: Start of time range (inclusive).
        end_datetime: End of time range (inclusive).

    Returns:
        List of flattened records matching target schema.
    """
    logger.info('Fetching Samsara data from %s to %s', start_datetime, end_datetime)

    records: list[dict[str, Any]] = []

    # Fetch vehicle stats history (includes GPS, engine states, odometer)
    logger.info('Fetching Samsara vehicle stats history...')
    vehicle_stats_list: list[VehicleStatsHistoryRecord] = list(
        provider.fetch_all(
            'vehicle_stats_history',
            start_time=start_datetime,
            end_time=end_datetime,
            types='engineStates,gps,obdOdometerMeters',
        )
    )
    logger.info('Found stats for %d Samsara vehicles', len(vehicle_stats_list))

    if not vehicle_stats_list:
        logger.warning('No Samsara vehicle stats found for time range')
        return records

    # Extract vehicle IDs for driver-vehicle assignment filtering
    vehicle_ids: str = ','.join(str(vs.vehicle_id) for vs in vehicle_stats_list)

    # Fetch driver-vehicle assignments for these specific vehicles
    logger.info('Fetching Samsara driver-vehicle assignments...')
    try:
        assignments: list[DriverVehicleAssignment] = list(
            provider.fetch_all(
                'driver_vehicle_assignments',
                filter_by='vehicles',
                start_time=start_datetime,
                end_time=end_datetime,
                vehicle_ids=vehicle_ids,
            )
        )
        logger.info('Found %d driver-vehicle assignments', len(assignments))
    except Exception:
        logger.exception('Error fetching driver-vehicle assignments')
        assignments = []

    # Process each vehicle's stats
    for vehicle_stats in vehicle_stats_list:
        logger.debug(
            'Processing Samsara vehicle %s (%d GPS records)',
            vehicle_stats.name,
            len(vehicle_stats.gps),
        )

        # Pre-process engine states for efficient lookup (sort by time)
        engine_states: list[dict[str, datetime | str]] = sorted(
            [
                {'time': es.time, 'state': es.value.value}
                for es in vehicle_stats.engine_states
            ],
            key=lambda x: x['time'],
        )

        # Pre-process odometer readings for efficient lookup
        odometer_readings: list[dict[str, datetime | float]] = [
            {'time': odom.time, 'value_miles': odom.value_miles}
            for odom in vehicle_stats.obd_odometer_meters
        ]

        # Flatten each GPS record
        for gps_record in vehicle_stats.gps:
            records.append(
                _flatten_samsara_gps(
                    gps_record,
                    vehicle_stats,
                    engine_states,
                    odometer_readings,
                    assignments,
                )
            )

    logger.info('Fetched %d Samsara GPS records', len(records))
    return records


# =============================================================================
# Main Pipeline Function
# =============================================================================


def create_vehicle_data_pipeline(
    motive_provider: Provider | None,
    samsara_provider: Provider | None,
    start_datetime: datetime,
    end_datetime: datetime,
) -> pd.DataFrame:
    """
    Create a unified vehicle data pipeline from multiple providers.

    Fetches vehicle location and telemetry data from Motive and Samsara,
    flattens nested structures, and combines into a single DataFrame.

    Args:
        motive_provider: Motive provider instance (or None to skip).
        samsara_provider: Samsara provider instance (or None to skip).
        start_datetime: Start of time range (inclusive).
        end_datetime: End of time range (inclusive).

    Returns:
        pandas DataFrame with columns:
            - provider: Provider name ('motive' or 'samsara')
            - provider_vehicle_id: Provider-specific vehicle ID
            - vin: Vehicle Identification Number
            - fleet_number: Fleet/unit number
            - timestamp: Timestamp of the record (UTC)
            - latitude: GPS latitude (decimal degrees)
            - longitude: GPS longitude (decimal degrees)
            - speed_mph: Speed in miles per hour
            - heading_degrees: Compass heading (0-360)
            - engine_state: Engine state ('On'/'Off'/'Idle'/None)
            - driver_id: Driver identifier
            - driver_name: Driver full name
            - location_description: Human-readable location
            - odometer: Odometer reading in miles

    Raises:
        ValueError: If both providers are None.

    Example:
        >>> from fleet_telemetry_hub.config.loader import load_config
        >>> from fleet_telemetry_hub.provider import Provider
        >>> from datetime import datetime, timezone
        >>>
        >>> config = load_config("config.yaml")
        >>> motive = Provider.from_config("motive", config)
        >>> samsara = Provider.from_config("samsara", config)
        >>>
        >>> df = create_vehicle_data_pipeline(
        ...     motive,
        ...     samsara,
        ...     start_datetime=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        ...     end_datetime=datetime(2025, 1, 7, 23, 59, 59, tzinfo=timezone.utc),
        ... )
        >>>
        >>> print(f"Records: {len(df)}, Providers: {df['provider'].nunique()}")
    """
    if motive_provider is None and samsara_provider is None:
        raise ValueError('At least one provider must be configured')

    logger.info(
        'Starting vehicle data pipeline: %s to %s',
        start_datetime,
        end_datetime,
    )

    all_records: list[dict[str, Any]] = []

    # Fetch Motive data
    if motive_provider:
        try:
            motive_records: list[dict[str, Any]] = _fetch_motive_data(
                motive_provider,
                start_datetime,
                end_datetime,
            )
            all_records.extend(motive_records)
        except Exception:
            logger.exception('Failed to fetch Motive data')
    else:
        logger.info('Skipping Motive (provider not configured)')

    # Fetch Samsara data
    if samsara_provider:
        try:
            samsara_records: list[dict[str, Any]] = _fetch_samsara_data(
                samsara_provider,
                start_datetime,
                end_datetime,
            )
            all_records.extend(samsara_records)
        except Exception:
            logger.exception('Failed to fetch Samsara data')
    else:
        logger.info('Skipping Samsara (provider not configured)')

    # Create DataFrame
    if not all_records:
        logger.warning('No records found from any provider')
        # Return empty DataFrame with correct schema
        return pd.DataFrame(
            columns=[
                'provider',
                'provider_vehicle_id',
                'vin',
                'fleet_number',
                'timestamp',
                'latitude',
                'longitude',
                'speed_mph',
                'heading_degrees',
                'engine_state',
                'driver_id',
                'driver_name',
                'location_description',
                'odometer',
            ]
        )

    df: pd.DataFrame = pd.DataFrame(all_records)

    # Enforce schema types
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['latitude'] = df['latitude'].astype('float64')
    df['longitude'] = df['longitude'].astype('float64')
    df['speed_mph'] = pd.to_numeric(df['speed_mph'], errors='coerce')
    df['heading_degrees'] = pd.to_numeric(df['heading_degrees'], errors='coerce')
    df['odometer'] = pd.to_numeric(df['odometer'], errors='coerce')
    df['provider'] = df['provider'].astype('category')
    df['engine_state'] = df['engine_state'].astype('category')

    # Sort by VIN and timestamp for temporal analysis
    return_df: pd.DataFrame = df.sort_values(['vin', 'timestamp']).reset_index(
        drop=True
    )

    logger.info(
        'Pipeline complete: %d total records from %d provider(s)',
        len(return_df),
        return_df['provider'].nunique(),
    )

    return return_df
