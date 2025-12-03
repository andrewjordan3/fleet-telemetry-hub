# fleet_telemetry_hub/utils/fetch_data.py

import logging
from datetime import date, datetime
from typing import Any

from ..models import (
    DriverVehicleAssignment,
    Vehicle,
    VehicleLocation,
    VehicleStatsHistoryRecord,
)
from ..provider import Provider
from .motive_funcs import flatten_motive_location
from .samsara_funcs import flatten_samsara_gps

logger: logging.Logger = logging.getLogger(__name__)


def fetch_motive_data(
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

                records.append(flatten_motive_location(location, vehicle))

        except Exception:
            logger.exception(
                'Error fetching locations for Motive vehicle %s', vehicle.vehicle_id
            )
            continue

    logger.info('Fetched %d Motive location records', len(records))
    return records


def fetch_samsara_data(
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
                flatten_samsara_gps(
                    gps_record,
                    vehicle_stats,
                    engine_states,
                    odometer_readings,
                    assignments,
                )
            )

    logger.info('Fetched %d Samsara GPS records', len(records))
    return records
