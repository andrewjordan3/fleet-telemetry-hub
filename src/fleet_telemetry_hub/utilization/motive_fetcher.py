"""Motive utilization fetcher and its typed bundle dataclass."""

import logging
from dataclasses import dataclass
from datetime import date

from fleet_telemetry_hub.models.motive_requests import MotiveEndpoints
from fleet_telemetry_hub.models.motive_responses import (
    DrivingPeriod,
    IdleEvent,
)
from fleet_telemetry_hub.provider import Provider
from fleet_telemetry_hub.utilization.date_chunking import iter_chunks

__all__: list[str] = ['MotiveUtilizationBundle', 'MotiveUtilizationFetcher']

logger: logging.Logger = logging.getLogger(__name__)

# Motive caps both /v1/driving_periods and /v1/idle_events at 30
# days per request. 28 days is a deliberate safety margin and also
# matches the chunking constant used by ``samsara_fetcher`` for
# cross-provider parity.
_MAX_CHUNK_DAYS: int = 28


@dataclass(frozen=True, slots=True)
class MotiveUtilizationBundle:
    """
    Typed container for Motive utilization data fetched over a date range.

    The fetched endpoints are event-grain (driving_periods, idle_events):
    each record carries its own start_time and end_time, so per-day
    association is intrinsic to the records themselves and no by-date
    grouping is applied at the bundle layer. Cross-midnight events are
    preserved unclipped -- the unifier handles clipping when it computes
    per-day attribution.

    Attributes:
        driving_periods: Driving period records spanning the full date
            range, including any cross-midnight periods that started
            within the range. Order matches the API response order
            (no client-side sorting).
        idle_events: Idle event records spanning the full date range,
            including any cross-midnight events that started within
            the range. Order matches the API response order (no
            client-side sorting).
        date_range: Inclusive (start_date, end_date) the bundle was
            fetched for.
        company: Company identifier configured on the source Provider,
            propagated into the unified output's ``company`` column.
            ``None`` when the Provider was constructed without one.
    """

    driving_periods: list[DrivingPeriod]
    idle_events: list[IdleEvent]
    date_range: tuple[date, date]
    company: str | None


class MotiveUtilizationFetcher:
    """
    Fetcher for Motive utilization data across a UTC date range.

    Wraps a configured Motive Provider and orchestrates the two
    event-grain endpoint calls the unifier consumes:

        - driving_periods: chunked into <=28-day windows
                           (Motive caps the endpoint at 30 days)
        - idle_events:     chunked into <=28-day windows
                           (Motive caps the endpoint at 30 days)

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
            MotiveUtilizationBundle holding the full-range lists of
            driving periods and idle events.

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

        driving_periods: list[DrivingPeriod] = []
        idle_events: list[IdleEvent] = []
        with self._provider.client() as client:
            for chunk_start_date, chunk_end_date in iter_chunks(
                start_date, end_date, _MAX_CHUNK_DAYS, chunk_format='date'
            ):
                driving_periods.extend(
                    client.fetch_all(
                        MotiveEndpoints.DRIVING_PERIODS,
                        start_date=chunk_start_date,
                        end_date=chunk_end_date,
                    )
                )
                idle_events.extend(
                    client.fetch_all(
                        MotiveEndpoints.IDLE_EVENTS,
                        start_date=chunk_start_date,
                        end_date=chunk_end_date,
                    )
                )

        logger.info(
            'Motive fetch complete: %d periods, %d idle events',
            len(driving_periods),
            len(idle_events),
        )

        return MotiveUtilizationBundle(
            driving_periods=driving_periods,
            idle_events=idle_events,
            date_range=(start_date, end_date),
            company=self._provider.company,
        )
