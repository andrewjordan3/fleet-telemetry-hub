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
from abc import abstractmethod
from collections.abc import Callable, Iterator
from datetime import date, datetime
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel

from fleet_telemetry_hub.models import (
    DriverVehicleAssignment,
    Vehicle,
    VehicleLocation,
    VehicleStatsHistoryRecord,
)
from fleet_telemetry_hub.operations.motive_funcs import flatten_motive_location
from fleet_telemetry_hub.operations.samsara_funcs import flatten_samsara_gps
from fleet_telemetry_hub.provider import Provider
from fleet_telemetry_hub.schema import TELEMETRY_COLUMNS

__all__: list[str] = [
    'PROVIDER_FETCHER_CLASSES',
    'DataFetcher',
    'MotiveDataFetcher',
    'SamsaraDataFetcher',
]

logger: logging.Logger = logging.getLogger(__name__)

# Maximum number of vehicle IDs per request to avoid URL length limits.
# With 15-digit IDs + commas, 50 IDs ≈ 800 chars, well under typical limits.
VEHICLE_ID_CHUNK_SIZE: int = 50

# =============================================================================
# Protocol Definition
# =============================================================================


@runtime_checkable
class DataFetcher(Protocol):
    """
    Protocol defining the interface for provider data fetchers.

    Implementations must provide a fetch_data method that retrieves telemetry
    records for a given time range. The protocol enables type-safe handling
    of different provider fetchers through a common interface.

    Example:
        fetcher: DataFetcher = MotiveDataFetcher(provider)
        records = fetcher.fetch_data(start_dt, end_dt)
    """

    @abstractmethod
    def fetch_data(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> list[dict[str, Any]]:
        """
        Fetch telemetry data for the specified time range.

        Args:
            start_datetime: Start of time range (inclusive, timezone-aware UTC).
            end_datetime: End of time range (inclusive, timezone-aware UTC).

        Returns:
            List of flattened record dictionaries matching TELEMETRY_COLUMNS schema.
        """
        ...


# =============================================================================
# Helper Functions
# =============================================================================


def _chunk_list[T](items: list[T], chunk_size: int) -> Iterator[list[T]]:
    """
    Yield successive chunks of a specified size from a list.

    Args:
        items: The list to chunk.
        chunk_size: Maximum number of items per chunk.

    Yields:
        Lists of at most chunk_size items.

    Example:
        >>> list(_chunk_list([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
    """
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


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


# =============================================================================
# Motive Fetcher
# =============================================================================


class MotiveDataFetcher:
    """
    Fetcher for Motive ELD telemetry data.

    Caches the vehicle list on first fetch, reusing it for subsequent calls.
    This avoids redundant API calls when processing multiple time batches
    in a single pipeline run.

    The vehicle cache persists for the lifetime of the fetcher instance.
    Create a new instance to force a vehicle list refresh.

    Attributes:
        provider: The Motive Provider instance (read-only).
        vehicles: Cached vehicle list, or None if not yet fetched (read-only).
        vehicle_count: Number of cached vehicles, or 0 if not loaded (read-only).

    Example:
        fetcher = MotiveDataFetcher(motive_provider)

        # First call fetches and caches vehicles
        batch1_records = fetcher.fetch_data(batch1_start, batch1_end)

        # Subsequent calls reuse cached vehicles
        batch2_records = fetcher.fetch_data(batch2_start, batch2_end)
    """

    def __init__(self, provider: Provider) -> None:
        """
        Initialize the Motive data fetcher.

        Does not make any network calls. Vehicle list is fetched lazily
        on first fetch_data() call.

        Args:
            provider: Initialized Motive Provider instance.
        """
        self._provider: Provider = provider
        self._vehicles: list[Vehicle] | None = None

        logger.debug('Initialized MotiveDataFetcher (vehicles will be lazy-loaded)')

    @property
    def provider(self) -> Provider:
        """The Motive Provider instance."""
        return self._provider

    @property
    def vehicles(self) -> list[Vehicle] | None:
        """Cached vehicle list, or None if not yet fetched."""
        return self._vehicles

    @property
    def vehicle_count(self) -> int:
        """Number of cached vehicles, or 0 if not loaded."""
        return len(self._vehicles) if self._vehicles is not None else 0

    def _ensure_vehicles_loaded(self) -> list[Vehicle]:
        """
        Ensure vehicle list is loaded, fetching if necessary.

        This method is idempotent - subsequent calls return the cached list
        without making additional API calls.

        Returns:
            List of Vehicle objects from Motive.

        Raises:
            Exception: Propagates API errors from provider.fetch_all().
        """
        if self._vehicles is not None:
            logger.debug(
                'Using cached Motive vehicle list (%d vehicles)',
                len(self._vehicles),
            )
            return self._vehicles

        logger.info('Fetching Motive vehicle list (first call, will be cached)...')

        with self._provider.client() as client:
            vehicles_raw: list[BaseModel] = list(
                client.fetch_all(self._provider.endpoint('vehicles'))
            )
            self._vehicles = cast(list[Vehicle], vehicles_raw)

        logger.info(
            'Cached %d Motive vehicles for subsequent batch fetches',
            len(self._vehicles),
        )

        return self._vehicles

    def clear_vehicle_cache(self) -> None:
        """
        Clear the cached vehicle list.

        Next fetch_data() call will re-fetch the vehicle list from the API.
        Useful if you need to pick up fleet changes mid-run (rare).
        """
        vehicle_count: int = self.vehicle_count
        self._vehicles = None
        logger.info('Cleared Motive vehicle cache (%d vehicles)', vehicle_count)

    def fetch_data(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> list[dict[str, Any]]:
        """
        Fetch and transform Motive vehicle location data.

        On first call, fetches and caches the vehicle list. Subsequent calls
        reuse the cached list, only fetching location data for the new time range.

        The Motive API accepts date-only parameters for locations, so additional
        datetime filtering is applied client-side.

        Args:
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

        # Get vehicles (cached after first call)
        vehicles: list[Vehicle] = self._ensure_vehicles_loaded()

        if not vehicles:
            logger.warning('No Motive vehicles available')
            return records

        # Fetch location history for each vehicle
        with self._provider.client() as client:
            for vehicle in vehicles:
                logger.debug(
                    'Fetching locations for Motive vehicle %s (id=%s)',
                    vehicle.number,
                    vehicle.vehicle_id,
                )

                try:
                    locations_raw: list[BaseModel] = list(
                        client.fetch_all(
                            self._provider.endpoint('vehicle_locations'),
                            vehicle_id=vehicle.vehicle_id,
                            start_date=start_date,
                            end_date=end_date,
                        )
                    )
                    locations: list[VehicleLocation] = cast(
                        list[VehicleLocation], locations_raw
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

                        record: dict[str, Any] = flatten_motive_location(
                            location, vehicle
                        )
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
                        'Error fetching locations for Motive vehicle %s (id=%s), '
                        'skipping',
                        vehicle.number,
                        vehicle.vehicle_id,
                    )
                    continue

        logger.info('Fetched %d total Motive location records', len(records))
        return records


# =============================================================================
# Samsara Fetcher
# =============================================================================


class SamsaraDataFetcher:
    """
    Fetcher for Samsara ELD telemetry data.

    Unlike Motive, Samsara's API returns vehicle stats directly filtered by
    time range, so there's no vehicle list to cache. This class exists for
    interface consistency with MotiveDataFetcher.

    Uses chunked requests for driver assignments to avoid URL length limits
    with large fleets.

    Attributes:
        provider: The Samsara Provider instance (read-only).

    Example:
        fetcher = SamsaraDataFetcher(samsara_provider)
        records = fetcher.fetch_data(start_dt, end_dt)
    """

    def __init__(self, provider: Provider) -> None:
        """
        Initialize the Samsara data fetcher.

        Args:
            provider: Initialized Samsara Provider instance.
        """
        self._provider: Provider = provider

        logger.debug('Initialized SamsaraDataFetcher')

    @property
    def provider(self) -> Provider:
        """The Samsara Provider instance."""
        return self._provider

    def _fetch_driver_assignments_chunked(
        self,
        vehicle_ids: list[str],
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> list[DriverVehicleAssignment]:
        """
        Fetch driver-vehicle assignments in chunks to avoid URL length limits.

        Samsara's driver-vehicle-assignments endpoint accepts a comma-separated
        list of vehicle IDs in the query string. With large fleets, this can
        exceed URL length limits (typically 2000-8000 characters). This method
        batches requests to stay under those limits.

        Args:
            vehicle_ids: List of vehicle ID strings to fetch assignments for.
            start_datetime: Start of time range.
            end_datetime: End of time range.

        Returns:
            Combined list of driver-vehicle assignments from all chunks.
            Returns empty list if all chunks fail.
        """
        if not vehicle_ids:
            return []

        assignments: list[DriverVehicleAssignment] = []
        chunks: list[list[str]] = list(_chunk_list(vehicle_ids, VEHICLE_ID_CHUNK_SIZE))

        logger.debug(
            'Fetching driver assignments for %d vehicles in %d chunk(s)',
            len(vehicle_ids),
            len(chunks),
        )

        with self._provider.client() as client:
            for chunk_index, vehicle_id_chunk in enumerate(chunks, start=1):
                vehicle_ids_csv: str = ','.join(vehicle_id_chunk)

                try:
                    chunk_assignments_raw: list[BaseModel] = list(
                        client.fetch_all(
                            self._provider.endpoint('driver_vehicle_assignments'),
                            filter_by='vehicles',
                            start_time=start_datetime,
                            end_time=end_datetime,
                            vehicle_ids=vehicle_ids_csv,
                        )
                    )
                    chunk_assignments: list[DriverVehicleAssignment] = cast(
                        list[DriverVehicleAssignment], chunk_assignments_raw
                    )
                    assignments.extend(chunk_assignments)

                    logger.debug(
                        'Assignment chunk %d/%d: %d assignments for %d vehicles',
                        chunk_index,
                        len(chunks),
                        len(chunk_assignments),
                        len(vehicle_id_chunk),
                    )

                except Exception:
                    logger.exception(
                        'Error fetching driver assignments for chunk %d/%d '
                        '(%d vehicles), continuing with partial data',
                        chunk_index,
                        len(chunks),
                        len(vehicle_id_chunk),
                    )
                    continue

        logger.debug(
            'Fetched %d total driver assignments across %d chunks',
            len(assignments),
            len(chunks),
        )

        return assignments

    def fetch_data(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> list[dict[str, Any]]:
        """
        Fetch and transform Samsara vehicle telemetry data.

        Retrieves vehicle stats history (GPS, engine states, odometer) and
        driver-vehicle assignments from Samsara, then flattens into the unified
        schema format.

        Args:
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

        with self._provider.client() as client:
            vehicle_stats_list_raw: list[BaseModel] = list(
                client.fetch_all(
                    self._provider.endpoint('vehicle_stats_history'),
                    start_time=start_datetime,
                    end_time=end_datetime,
                    types='engineStates,gps,obdOdometerMeters',
                )
            )

        vehicle_stats_list: list[VehicleStatsHistoryRecord] = cast(
            list[VehicleStatsHistoryRecord], vehicle_stats_list_raw
        )
        logger.info('Found stats for %d Samsara vehicles', len(vehicle_stats_list))

        if not vehicle_stats_list:
            logger.warning('No Samsara vehicle stats found for time range')
            return records

        # Build vehicle ID list for driver assignment filtering
        vehicle_ids: list[str] = [
            str(vehicle_stats.vehicle_id) for vehicle_stats in vehicle_stats_list
        ]

        # Fetch driver-vehicle assignments in chunks (handles URL length limits)
        logger.debug('Fetching Samsara driver-vehicle assignments...')
        assignments: list[DriverVehicleAssignment] = (
            self._fetch_driver_assignments_chunked(
                vehicle_ids=vehicle_ids,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
            )
        )
        logger.info('Found %d driver-vehicle assignments', len(assignments))

        # Process each vehicle's stats
        for vehicle_stats in vehicle_stats_list:
            gps_count: int = len(vehicle_stats.gps)
            logger.debug(
                'Vehicle %s: %d raw locations fetched',
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


# =============================================================================
# Fetcher Registry
# =============================================================================

# Maps provider names to their fetcher classes.
# Used by the pipeline to instantiate the appropriate fetcher for each provider.
PROVIDER_FETCHER_CLASSES: dict[str, Callable[[Provider], DataFetcher]] = {
    'motive': MotiveDataFetcher,
    'samsara': SamsaraDataFetcher,
}
