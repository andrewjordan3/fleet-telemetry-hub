"""Motive utilization fetcher and its typed bundle dataclass."""

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from fleet_telemetry_hub.models.motive_requests import MotiveEndpoints
from fleet_telemetry_hub.models.motive_responses import (
    DriverIdleRollup,
    DrivingPeriod,
    IdleEvent,
    VehicleUtilization,
)
from fleet_telemetry_hub.provider import Provider

__all__: list[str] = ['MotiveUtilizationBundle', 'MotiveUtilizationFetcher']

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MotiveUtilizationBundle:
    """
    Typed container for Motive utilization data fetched over a date range.

    Aggregate-grain endpoints (vehicle_utilization, driver_utilization)
    return per-window totals without per-record timestamps. To preserve
    the date-of-fetch association, those records are grouped into dicts
    keyed by the UTC date they were fetched for. Every date in the
    requested range has a key, even if the value is an empty list.

    Event-grain endpoints (driving_periods) return records with their
    own start_time and end_time fields, so per-day association is
    intrinsic to the records themselves and no by-date grouping is
    applied at the bundle layer. Cross-midnight periods are preserved
    unclipped -- the unifier handles clipping when it computes
    per-day attribution.

    Note on bundle field consumption: ``vehicle_utilizations_by_date``
    and ``driver_idle_rollups_by_date`` are populated by the fetcher
    but are not currently consumed by the downstream unifier path,
    which sources its event-grain output from ``driving_periods`` and
    ``idle_events``. The aggregate-grain fields are retained for now
    so downstream callers that previously depended on them keep
    working; they will be revisited when the unifier work consolidates.

    Attributes:
        vehicle_utilizations_by_date: Per-vehicle aggregates, keyed by
            UTC date. Every date in date_range is present as a key.
        driver_idle_rollups_by_date: Per-driver aggregates, keyed by UTC
            date. Includes the null-driver bucket as an entry in each
            day's list when the API returns it. Every date in
            date_range is present as a key.
        driving_periods: Driving period records spanning the full date
            range, including any cross-midnight periods that started
            within the range. Order matches the API response order
            (no client-side sorting).
        idle_events: Idle event records spanning the full date range,
            including any cross-midnight events that started within
            the range. Order matches the API response order (no
            client-side sorting).
        date_range: Inclusive (start_date, end_date) the bundle was
            fetched for. Useful for downstream code to verify coverage
            without recomputing from dict keys.
    """

    vehicle_utilizations_by_date: dict[date, list[VehicleUtilization]]
    driver_idle_rollups_by_date: dict[date, list[DriverIdleRollup]]
    driving_periods: list[DrivingPeriod]
    idle_events: list[IdleEvent]
    date_range: tuple[date, date]


class MotiveUtilizationFetcher:
    """
    Fetcher for Motive utilization data across a UTC date range.

    Wraps a configured Motive Provider and orchestrates the four
    endpoint calls needed for utilization attribution:

        - vehicle_utilization: per-day call
        - driver_utilization:  per-day call
        - driving_periods:     single call across the full range
        - idle_events:         single call across the full range

    The fetcher does no transformation of returned records -- it
    fetches and assembles them into a typed bundle. Unit conversions,
    timezone interpretation, attribution math, and cross-midnight
    clipping all happen downstream in the unifier.

    Satisfies the UtilizationFetcher[MotiveUtilizationBundle] Protocol.

    Attributes:
        provider: The configured Motive Provider instance (read-only).
    """

    def __init__(self, provider: Provider) -> None:
        """
        Initialize the fetcher with a configured Motive Provider.

        Args:
            provider: A Provider configured for the Motive API.
        """
        self._provider: Provider = provider

    @property
    def provider(self) -> Provider:
        """Return the configured Motive Provider."""
        return self._provider

    def fetch(
        self,
        start_date: date,
        end_date: date,
    ) -> MotiveUtilizationBundle:
        """
        Fetch all Motive utilization data for the inclusive UTC date range.

        Args:
            start_date: First UTC day to fetch (inclusive).
            end_date: Last UTC day to fetch (inclusive).

        Returns:
            MotiveUtilizationBundle holding per-day vehicle and driver
            rollups plus the full-range list of driving periods.

        Raises:
            ValueError: If start_date > end_date.
        """
        if start_date > end_date:
            raise ValueError(
                f'start_date ({start_date}) must be <= end_date ({end_date})'
            )

        logger.info(
            'Fetching Motive utilization for %s through %s',
            start_date,
            end_date,
        )

        target_dates: list[date] = self._enumerate_dates(start_date, end_date)

        vehicle_utilizations_by_date: dict[date, list[VehicleUtilization]] = {}
        driver_idle_rollups_by_date: dict[date, list[DriverIdleRollup]] = {}

        with self._provider.client() as client:
            for target_date in target_dates:
                window_start, window_end = self._utc_day_window(target_date)

                vehicle_rows: list[VehicleUtilization] = list(
                    client.fetch_all(
                        MotiveEndpoints.VEHICLE_UTILIZATION,
                        start_at=window_start,
                        end_at=window_end,
                    )
                )
                vehicle_utilizations_by_date[target_date] = vehicle_rows
                logger.debug(
                    'Day %s vehicle_utilization: %d records',
                    target_date,
                    len(vehicle_rows),
                )

                driver_rows: list[DriverIdleRollup] = list(
                    client.fetch_all(
                        MotiveEndpoints.DRIVER_UTILIZATION,
                        start_date=window_start,
                        end_date=window_end,
                    )
                )
                driver_idle_rollups_by_date[target_date] = driver_rows
                logger.debug(
                    'Day %s driver_utilization: %d records',
                    target_date,
                    len(driver_rows),
                )

            driving_periods: list[DrivingPeriod] = list(
                client.fetch_all(
                    MotiveEndpoints.DRIVING_PERIODS,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
            logger.debug('driving_periods: %d records', len(driving_periods))

            idle_events: list[IdleEvent] = list(
                client.fetch_all(
                    MotiveEndpoints.IDLE_EVENTS,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
            logger.debug('idle_events: %d records', len(idle_events))

        total_vehicle_rows: int = sum(
            len(rows) for rows in vehicle_utilizations_by_date.values()
        )
        total_driver_rows: int = sum(
            len(rows) for rows in driver_idle_rollups_by_date.values()
        )
        logger.info(
            'Motive fetch complete: %d vehicle rows, %d driver rows, '
            '%d periods, %d idle events',
            total_vehicle_rows,
            total_driver_rows,
            len(driving_periods),
            len(idle_events),
        )

        return MotiveUtilizationBundle(
            vehicle_utilizations_by_date=vehicle_utilizations_by_date,
            driver_idle_rollups_by_date=driver_idle_rollups_by_date,
            driving_periods=driving_periods,
            idle_events=idle_events,
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
