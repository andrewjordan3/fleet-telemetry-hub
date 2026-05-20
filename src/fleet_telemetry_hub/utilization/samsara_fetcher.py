"""Samsara utilization fetcher and its typed bundle dataclass."""

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from fleet_telemetry_hub.models.samsara_requests import SamsaraEndpoints
from fleet_telemetry_hub.models.samsara_responses import (
    DriverFuelEnergyReport,
    DriverVehicleAssignment,
    FuelEnergyVehicleReport,
    IdlingEvent,
)
from fleet_telemetry_hub.provider import Provider

__all__: list[str] = ['SamsaraUtilizationBundle', 'SamsaraUtilizationFetcher']

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SamsaraUtilizationBundle:
    """
    Typed container for Samsara utilization data fetched over a date range.

    Mirrors MotiveUtilizationBundle in structure: aggregate-grain
    records grouped into by-date dicts, event-grain records as flat
    lists, plus the inclusive ``date_range``. Samsara has one
    additional event-grain field (``idling_events``) that the Motive
    bundle does not.

    Aggregate-grain endpoints (vehicle_fuel_energy, driver_fuel_energy)
    return per-window totals without per-record timestamps. The
    bundle groups them into dicts keyed by the UTC date they were
    fetched for. Every date in ``date_range`` is present as a key,
    even if the value is an empty list.

    Event-grain endpoints (driver_vehicle_assignments, idling_events)
    return records carrying their own start/end timestamps. The
    bundle stores these flat. Cross-midnight events are preserved
    unclipped -- the unifier handles clipping.

    Attributes:
        vehicle_fuel_energy_by_date: Per-vehicle aggregates, keyed by
            UTC date. Every date in date_range is present.
        driver_fuel_energy_by_date: Per-driver aggregates, keyed by
            UTC date. Every date in date_range is present.
        driver_vehicle_assignments: Bipartite (driver, vehicle,
            time-window) records spanning the full date range,
            including any that straddle the range boundaries. Order
            matches the API response order (no client-side sorting).
        idling_events: Idling event records spanning the full date
            range. Each event has a startTime and durationMilliseconds
            but no endTime. Order matches the API response order
            (no client-side sorting).
        date_range: Inclusive (start_date, end_date) the bundle was
            fetched for. Useful for downstream code to verify coverage
            without recomputing from dict keys.
    """

    vehicle_fuel_energy_by_date: dict[date, list[FuelEnergyVehicleReport]]
    driver_fuel_energy_by_date: dict[date, list[DriverFuelEnergyReport]]
    driver_vehicle_assignments: list[DriverVehicleAssignment]
    idling_events: list[IdlingEvent]
    date_range: tuple[date, date]


class SamsaraUtilizationFetcher:
    """
    Fetcher for Samsara utilization data across a UTC date range.

    Wraps a configured Samsara Provider and orchestrates the four
    endpoint calls needed for utilization attribution:

        - vehicle_fuel_energy:        per-day call
        - driver_fuel_energy:         per-day call
        - driver_vehicle_assignments: single call across the full range
        - idling_events:              single call across the full range

    The fetcher does no transformation of returned records -- it
    fetches and assembles them into a typed bundle. Unit conversions,
    timezone interpretation, attribution math, and cross-midnight
    clipping all happen downstream in the unifier.

    Satisfies the UtilizationFetcher[SamsaraUtilizationBundle] Protocol.

    Attributes:
        provider: The configured Samsara Provider instance (read-only).
    """

    def __init__(self, provider: Provider) -> None:
        """
        Initialize the fetcher with a configured Samsara Provider.

        Args:
            provider: A Provider configured for the Samsara API.
        """
        self._provider: Provider = provider

    @property
    def provider(self) -> Provider:
        """Return the configured Samsara Provider."""
        return self._provider

    def fetch(
        self,
        start_date: date,
        end_date: date,
    ) -> SamsaraUtilizationBundle:
        """
        Fetch all Samsara utilization data for the inclusive UTC date range.

        Args:
            start_date: First UTC day to fetch (inclusive).
            end_date: Last UTC day to fetch (inclusive).

        Returns:
            SamsaraUtilizationBundle holding per-day vehicle and driver
            fuel-energy rollups plus the full-range lists of
            driver-vehicle assignments and idling events.

        Raises:
            ValueError: If start_date > end_date.
        """
        if start_date > end_date:
            raise ValueError(
                f'start_date ({start_date}) must be <= end_date ({end_date})'
            )

        logger.info(
            'Fetching Samsara utilization for %s through %s',
            start_date,
            end_date,
        )

        target_dates: list[date] = self._enumerate_dates(start_date, end_date)
        range_start, range_end = self._utc_range_window(start_date, end_date)

        vehicle_fuel_energy_by_date: dict[date, list[FuelEnergyVehicleReport]] = {}
        driver_fuel_energy_by_date: dict[date, list[DriverFuelEnergyReport]] = {}

        with self._provider.client() as client:
            for target_date in target_dates:
                window_start, window_end = self._utc_day_window(target_date)

                vehicle_rows: list[FuelEnergyVehicleReport] = list(
                    client.fetch_all(
                        SamsaraEndpoints.VEHICLE_FUEL_ENERGY,
                        start_date=window_start,
                        end_date=window_end,
                    )
                )
                vehicle_fuel_energy_by_date[target_date] = vehicle_rows
                logger.debug(
                    'Day %s vehicle_fuel_energy: %d records',
                    target_date,
                    len(vehicle_rows),
                )

                driver_rows: list[DriverFuelEnergyReport] = list(
                    client.fetch_all(
                        SamsaraEndpoints.DRIVER_FUEL_ENERGY,
                        start_date=window_start,
                        end_date=window_end,
                    )
                )
                driver_fuel_energy_by_date[target_date] = driver_rows
                logger.debug(
                    'Day %s driver_fuel_energy: %d records',
                    target_date,
                    len(driver_rows),
                )

            driver_vehicle_assignments: list[DriverVehicleAssignment] = list(
                client.fetch_all(
                    SamsaraEndpoints.DRIVER_VEHICLE_ASSIGNMENTS,
                    filter_by='drivers',
                    start_time=range_start,
                    end_time=range_end,
                )
            )
            logger.debug(
                'driver_vehicle_assignments: %d records',
                len(driver_vehicle_assignments),
            )

            idling_events: list[IdlingEvent] = list(
                client.fetch_all(
                    SamsaraEndpoints.IDLING_EVENTS,
                    start_time=range_start,
                    end_time=range_end,
                )
            )
            logger.debug('idling_events: %d records', len(idling_events))

        total_vehicle_rows: int = sum(
            len(rows) for rows in vehicle_fuel_energy_by_date.values()
        )
        total_driver_rows: int = sum(
            len(rows) for rows in driver_fuel_energy_by_date.values()
        )
        logger.info(
            'Samsara fetch complete: %d vehicle rows, %d driver rows, '
            '%d assignments, %d idling events',
            total_vehicle_rows,
            total_driver_rows,
            len(driver_vehicle_assignments),
            len(idling_events),
        )

        return SamsaraUtilizationBundle(
            vehicle_fuel_energy_by_date=vehicle_fuel_energy_by_date,
            driver_fuel_energy_by_date=driver_fuel_energy_by_date,
            driver_vehicle_assignments=driver_vehicle_assignments,
            idling_events=idling_events,
            date_range=(start_date, end_date),
        )

    @staticmethod
    def _enumerate_dates(start_date: date, end_date: date) -> list[date]:
        """Return every UTC date in [start_date, end_date] inclusive."""
        day_count: int = (end_date - start_date).days + 1
        return [start_date + timedelta(days=offset) for offset in range(day_count)]

    @staticmethod
    def _utc_day_window(target_date: date) -> tuple[datetime, datetime]:
        """Return the [00:00:00Z, next-day 00:00:00Z) UTC window for a date."""
        window_start: datetime = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            tzinfo=UTC,
        )
        window_end: datetime = window_start + timedelta(days=1)
        return window_start, window_end

    @staticmethod
    def _utc_range_window(
        start_date: date,
        end_date: date,
    ) -> tuple[datetime, datetime]:
        """
        Return the half-open UTC window covering an inclusive date range.

        For [start_date, end_date] inclusive, returns
        ``(start_date 00:00:00Z, (end_date + 1 day) 00:00:00Z)`` so the
        upper bound excludes the day after end_date but includes all of
        end_date itself.
        """
        range_start: datetime = datetime(
            start_date.year,
            start_date.month,
            start_date.day,
            tzinfo=UTC,
        )
        range_end: datetime = datetime(
            end_date.year,
            end_date.month,
            end_date.day,
            tzinfo=UTC,
        ) + timedelta(days=1)
        return range_start, range_end
