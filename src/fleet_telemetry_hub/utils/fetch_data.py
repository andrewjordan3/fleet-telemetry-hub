# fleet_telemetry_hub/utils/fetch_data.py
"""
Data fetching functions for ELD telemetry providers.

This module provides provider-specific fetch functions that retrieve raw
telemetry data from Motive and Samsara APIs and transform it into the
unified schema format.

Each function handles:
- API pagination via the Provider abstraction
- Provider-specific data transformations
- Flattening nested structures to tabular format

The output format matches TELEMETRY_COLUMNS from the schema module, enabling
direct concatenation of data from different providers.
"""

import logging
from datetime import date, datetime
from typing import Any

from fleet_telemetry_hub.models import (
    DriverVehicleAssignment,
    Vehicle,
    VehicleLocation,
    VehicleStatsHistoryRecord,
)
from fleet_telemetry_hub.provider import Provider
from fleet_telemetry_hub.schema import TELEMETRY_COLUMNS

from .motive_funcs import flatten_motive_location
from .samsara_funcs import flatten_samsara_gps

__all__: list[str] = ['fetch_motive_data', 'fetch_samsara_data']

logger: logging.Logger = logging.getLogger(__name__)


def _validate_record_columns(record: dict[str, Any], source: str) -> None:
    """
    Validate that a flattened record contains all required columns.

    This is a debug-time check to catch schema drift between flatten functions
    and the canonical schema. Only runs at DEBUG log level for performance.

    Args:
        record: Flattened record dictionary from a flatten function.
        source: Description of source for error messages (e.g., 'Motive location').

    Raises:
        ValueError: If record is missing required columns (only in debug mode).
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    missing_columns: set[str] = set(TELEMETRY_COLUMNS) - set(record.keys())
    if missing_columns:
        raise ValueError(
            f'{source} record missing required columns: {sorted(missing_columns)}'
        )


def fetch_motive_data(
    provider: Provider,
    start_datetime: datetime,
    end_datetime: datetime,
) -> list[dict[str, Any]]:
    """
    Fetch and transform Motive vehicle location data.

    Retrieves all vehicles from Motive, then fetches location history for each
    vehicle within the specified time range. The Motive API accepts date-only
    parameters, so additional datetime filtering is applied client-side.

    Args:
        provider: Initialized Motive Provider instance.
        start_datetime: Start of time range (inclusive, timezone-aware UTC).
        end_datetime: End of time range (inclusive, timezone-aware UTC).

    Returns:
        List of flattened record dictionaries matching TELEMETRY_COLUMNS schema.
        Empty list if no vehicles found or no locations in range.

    Raises:
        Exception: Propagates API errors from provider.fetch_all().
            Individual vehicle fetch errors are logged and skipped.
    """
    logger.info(
        'Fetching Motive data: %s to %s',
        start_datetime.isoformat(),
        end_datetime.isoformat(),
    )

    # Motive API uses date-only parameters; we filter to exact datetimes later
    start_date: date = start_datetime.date()
    end_date: date = end_datetime.date()

    records: list[dict[str, Any]] = []

    # Fetch all vehicles first (required to get vehicle IDs for location queries)
    logger.debug('Fetching Motive vehicle list...')
    vehicles: list[Vehicle] = list(provider.fetch_all('vehicles'))
    logger.info('Found %d Motive vehicles', len(vehicles))

    # Fetch location history for each vehicle
    for vehicle in vehicles:
        logger.debug(
            'Fetching locations for Motive vehicle %s (id=%s)',
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
                'Vehicle %s: %d raw locations fetched',
                vehicle.number,
                len(locations),
            )

            # Filter to exact datetime range (Motive returns full days)
            filtered_count: int = 0
            for location in locations:
                if location.located_at < start_datetime:
                    continue
                if location.located_at > end_datetime:
                    continue

                record: dict[str, Any] = flatten_motive_location(location, vehicle)
                _validate_record_columns(record, 'Motive location')
                records.append(record)
                filtered_count += 1

            logger.debug(
                'Vehicle %s: %d locations after datetime filter',
                vehicle.number,
                filtered_count,
            )

        except Exception:
            logger.exception(
                'Error fetching locations for Motive vehicle %s (id=%s), skipping',
                vehicle.number,
                vehicle.vehicle_id,
            )
            continue

    logger.info('Fetched %d total Motive location records', len(records))
    return records


def fetch_samsara_data(
    provider: Provider,
    start_datetime: datetime,
    end_datetime: datetime,
) -> list[dict[str, Any]]:
    """
    Fetch and transform Samsara vehicle telemetry data.

    Retrieves vehicle stats history (GPS, engine states, odometer) and
    driver-vehicle assignments from Samsara, then flattens into the unified
    schema format.

    Args:
        provider: Initialized Samsara Provider instance.
        start_datetime: Start of time range (inclusive, timezone-aware UTC).
        end_datetime: End of time range (inclusive, timezone-aware UTC).

    Returns:
        List of flattened record dictionaries matching TELEMETRY_COLUMNS schema.
        Empty list if no vehicle stats found in range.

    Raises:
        Exception: Propagates API errors from provider.fetch_all().
            Driver assignment fetch errors are logged and skipped.
    """
    logger.info(
        'Fetching Samsara data: %s to %s',
        start_datetime.isoformat(),
        end_datetime.isoformat(),
    )

    records: list[dict[str, Any]] = []

    # Fetch vehicle stats history (includes GPS, engine states, odometer)
    logger.debug('Fetching Samsara vehicle stats history...')
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

    # Build vehicle ID list for driver assignment filtering
    vehicle_ids_csv: str = ','.join(
        str(vehicle_stats.vehicle_id) for vehicle_stats in vehicle_stats_list
    )

    # Fetch driver-vehicle assignments (optional, continue on failure)
    assignments: list[DriverVehicleAssignment] = []
    logger.debug('Fetching Samsara driver-vehicle assignments...')
    try:
        assignments = list(
            provider.fetch_all(
                'driver_vehicle_assignments',
                filter_by='vehicles',
                start_time=start_datetime,
                end_time=end_datetime,
                vehicle_ids=vehicle_ids_csv,
            )
        )
        logger.info('Found %d driver-vehicle assignments', len(assignments))
    except Exception:
        logger.exception(
            'Error fetching driver-vehicle assignments, continuing without driver data'
        )

    # Process each vehicle's stats
    for vehicle_stats in vehicle_stats_list:
        gps_count: int = len(vehicle_stats.gps)
        logger.debug(
            'Processing Samsara vehicle %s (id=%s): %d GPS records',
            vehicle_stats.name,
            vehicle_stats.vehicle_id,
            gps_count,
        )

        # Pre-process engine states for efficient temporal lookup
        # Sorting enables binary search in flatten function if needed
        engine_states: list[dict[str, datetime | str]] = sorted(
            [
                {'time': engine_state.time, 'state': engine_state.value.value}
                for engine_state in vehicle_stats.engine_states
            ],
            key=lambda x: x['time'],
        )

        # Pre-process odometer readings for temporal lookup
        odometer_readings: list[dict[str, datetime | float]] = [
            {'time': odometer.time, 'value_miles': odometer.value_miles}
            for odometer in vehicle_stats.obd_odometer_meters
        ]

        # Flatten each GPS record into unified schema
        for gps_record in vehicle_stats.gps:
            record: dict[str, Any] = flatten_samsara_gps(
                gps_record,
                vehicle_stats,
                engine_states,
                odometer_readings,
                assignments,
            )
            _validate_record_columns(record, 'Samsara GPS')
            records.append(record)

    logger.info('Fetched %d total Samsara GPS records', len(records))
    return records
