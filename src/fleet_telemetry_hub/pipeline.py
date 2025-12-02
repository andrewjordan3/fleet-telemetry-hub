# fleet_telemetry_hub/pipeline.py
"""
Vehicle data pipeline for fetching and transforming telemetry data.

This module provides a pipeline to fetch vehicle location and telemetry data
from multiple providers (Motive and Samsara), flatten nested structures, and
combine them into a unified pandas DataFrame.
"""

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from .models.motive_responses import Vehicle, VehicleLocation
from .models.samsara_responses import (
    DriverVehicleAssignment,
    EngineState,
    GpsRecord,
    VehicleStatsHistoryRecord,
)
from .provider import Provider

logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# Data Transformation Functions
# =============================================================================


def _flatten_motive_location(
    location: VehicleLocation,
    vehicle: Vehicle,
) -> dict[str, Any]:
    """
    Flatten a Motive VehicleLocation into the target schema.

    Args:
        location: VehicleLocation record from Motive.
        vehicle: Vehicle record for context (provides VIN, fleet number).

    Returns:
        Dictionary with flattened fields matching target schema.
    """
    return {
        'provider': 'motive',
        'provider_vehicle_id': str(vehicle.vehicle_id),
        'vin': vehicle.vin,
        'fleet_number': vehicle.number,
        'timestamp': location.located_at,
        'latitude': location.latitude,
        'longitude': location.longitude,
        'speed_mph': location.speed,
        'heading_degrees': location.bearing,
        'engine_state': None,  # Motive doesn't provide engine state in location data
        'driver_id': str(location.driver.driver_id) if location.driver else None,
        'driver_name': location.driver.full_name if location.driver else None,
        'location_description': location.description,
        'odometer': location.odometer,
    }


def _find_engine_state_at_time(
    engine_states: list[dict[str, Any]],
    timestamp: datetime,
) -> str | None:
    """
    Find the engine state at a specific timestamp.

    Engine states are sparse - we need to find the most recent state
    before or at the given timestamp.

    Args:
        engine_states: List of {'time': datetime, 'state': str} records.
        timestamp: Target timestamp.

    Returns:
        Engine state string (On/Off/Idle) or None if no data available.
    """
    if not engine_states:
        return None

    # Find the most recent state <= timestamp
    relevant_states = [s for s in engine_states if s['time'] <= timestamp]
    if not relevant_states:
        return None

    return max(relevant_states, key=lambda s: s['time'])['state']


def _find_odometer_at_time(
    odometer_readings: list[dict[str, Any]],
    timestamp: datetime,
) -> float | None:
    """
    Find the odometer reading closest to a specific timestamp.

    Odometer readings are sparse - we interpolate by finding the closest reading.

    Args:
        odometer_readings: List of {'time': datetime, 'value_miles': float} records.
        timestamp: Target timestamp.

    Returns:
        Odometer reading in miles or None if no data available.
    """
    if not odometer_readings:
        return None

    # Find the closest reading (before or after)
    closest = min(
        odometer_readings,
        key=lambda r: abs((r['time'] - timestamp).total_seconds()),
    )

    # Only use if within 1 hour of the GPS reading
    time_diff = abs((closest['time'] - timestamp).total_seconds())
    if time_diff <= 3600:  # 1 hour tolerance
        return closest['value_miles']

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
        engine_states: Pre-processed engine state records.
        odometer_readings: Pre-processed odometer records.
        assignments: All driver-vehicle assignments for the time period.

    Returns:
        Dictionary with flattened fields matching target schema.
    """
    # Find engine state and odometer at this timestamp
    engine_state = _find_engine_state_at_time(engine_states, gps_record.time)
    odometer = _find_odometer_at_time(odometer_readings, gps_record.time)

    # Find driver assignment
    driver_id, driver_name = _find_driver_for_timestamp(
        assignments,
        vehicle_stats.vehicle_id,
        gps_record.time,
    )

    return {
        'provider': 'samsara',
        'provider_vehicle_id': vehicle_stats.vehicle_id,
        'vin': vehicle_stats.get_vin(),
        'fleet_number': vehicle_stats.name,  # Samsara uses 'name' as identifier
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
        start_datetime: Start of time range.
        end_datetime: End of time range.

    Returns:
        List of flattened records matching target schema.
    """
    logger.info('Fetching Motive data from %s to %s', start_datetime, end_datetime)

    # Convert datetime to date for Motive API (uses date-only parameters)
    start_date = start_datetime.date()
    end_date = end_datetime.date()

    records: list[dict[str, Any]] = []

    # Fetch all vehicles first
    logger.info('Fetching Motive vehicles...')
    vehicles = list(provider.fetch_all('vehicles'))
    logger.info('Found %d Motive vehicles', len(vehicles))

    # For each vehicle, fetch location history
    for vehicle in vehicles:
        logger.debug(
            'Fetching locations for Motive vehicle %s (%s)',
            vehicle.number,
            vehicle.vehicle_id,
        )

        try:
            locations = list(
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
                # Filter by datetime if location has timestamp
                if location.located_at < start_datetime or location.located_at > end_datetime:
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
        start_datetime: Start of time range.
        end_datetime: End of time range.

    Returns:
        List of flattened records matching target schema.
    """
    logger.info('Fetching Samsara data from %s to %s', start_datetime, end_datetime)

    records: list[dict[str, Any]] = []

    # Fetch vehicle stats history (includes GPS, engine states, odometer)
    logger.info('Fetching Samsara vehicle stats history...')
    vehicle_stats_list = list(
        provider.fetch_all(
            'vehicle_stats_history',
            start_time=start_datetime,
            end_time=end_datetime,
            types='engineStates,gps,obdOdometerMeters',
        )
    )
    logger.info('Found stats for %d Samsara vehicles', len(vehicle_stats_list))

    # Fetch driver-vehicle assignments
    logger.info('Fetching Samsara driver-vehicle assignments...')
    assignments = list(
        provider.fetch_all(
            'driver_vehicle_assignments',
            filter_by='vehicles',
            start_time=start_datetime,
            end_time=end_datetime,
        )
    )
    logger.info('Found %d driver-vehicle assignments', len(assignments))

    # Process each vehicle's stats
    for vehicle_stats in vehicle_stats_list:
        logger.debug(
            'Processing Samsara vehicle %s (%s GPS records)',
            vehicle_stats.name,
            len(vehicle_stats.gps),
        )

        # Pre-process engine states for efficient lookup
        engine_states = [
            {'time': es.time, 'state': es.value.value}
            for es in vehicle_stats.engine_states
        ]

        # Pre-process odometer readings for efficient lookup
        odometer_readings = [
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
            - timestamp: Timestamp of the record
            - latitude: GPS latitude
            - longitude: GPS longitude
            - speed_mph: Speed in miles per hour
            - heading_degrees: Compass heading (0-360)
            - engine_state: Engine state (On/Off/Idle)
            - driver_id: Driver identifier
            - driver_name: Driver full name
            - location_description: Human-readable location
            - odometer: Odometer reading in miles

    Example:
        >>> from fleet_telemetry_hub.config.loader import load_config
        >>> from fleet_telemetry_hub.provider import Provider
        >>> from datetime import datetime
        >>>
        >>> config = load_config("config.yaml")
        >>> motive = Provider.from_config("motive", config)
        >>> samsara = Provider.from_config("samsara", config)
        >>>
        >>> df = create_vehicle_data_pipeline(
        ...     motive,
        ...     samsara,
        ...     start_datetime=datetime(2025, 1, 1, 0, 0, 0),
        ...     end_datetime=datetime(2025, 1, 7, 23, 59, 59),
        ... )
        >>>
        >>> print(df.shape)
        >>> print(df.head())
    """
    logger.info(
        'Starting vehicle data pipeline: %s to %s',
        start_datetime,
        end_datetime,
    )

    all_records: list[dict[str, Any]] = []

    # Fetch Motive data
    if motive_provider:
        try:
            motive_records = _fetch_motive_data(
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
            samsara_records = _fetch_samsara_data(
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

    df = pd.DataFrame(all_records)

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    logger.info(
        'Pipeline complete: %d total records from %d provider(s)',
        len(df),
        df['provider'].nunique(),
    )

    return df
