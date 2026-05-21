"""Samsara utilization fetcher and its typed bundle dataclass.

This module switches Samsara to an event-grain shape. Per-vehicle
trips (from ``/v1/fleet/trips``) and per-window idling events (from
``/idling/events``) are the new source of utilization signal,
chunked at 28 days to stay well under Samsara's 90-day per-call
cap on the trips endpoint. Dimension data (``vehicles``,
``drivers``) flows in the same bundle so the downstream unifier
can resolve ``vehicleId -> vin`` and ``driverId -> driver_name``
from a single fetch.
"""

import logging
from dataclasses import dataclass
from datetime import date

from fleet_telemetry_hub.models.samsara_requests import SamsaraEndpoints
from fleet_telemetry_hub.models.samsara_responses import (
    DriverActivationStatus,
    IdlingEvent,
    SamsaraDriver,
    SamsaraVehicle,
    Trip,
)
from fleet_telemetry_hub.provider import Provider
from fleet_telemetry_hub.utilization.date_chunking import iter_chunks

__all__: list[str] = ['SamsaraUtilizationBundle', 'SamsaraUtilizationFetcher']

logger: logging.Logger = logging.getLogger(__name__)

# Samsara caps /v1/fleet/trips at 90 days per call; 28 days is a
# deliberate safety margin that also matches the chunking used by the
# idling-events fetch for parity.
_MAX_CHUNK_DAYS: int = 28


@dataclass(frozen=True, slots=True)
class SamsaraUtilizationBundle:
    """
    Typed container for Samsara utilization data fetched over a date range.

    The bundle pairs the two event-grain streams the unifier consumes
    (``trips``, ``idling_events``) with the two dimension lists the
    unifier needs to resolve identifiers to human-readable values
    (``vehicles``, ``drivers``). Holding the dimensions alongside the
    events means one fetch yields everything required for a unified
    utilization view of the requested date range.

    Trip and idling-event ordering is whatever the API returns within
    each chunk. The trips list is also ordered by vehicle iteration
    order on the outer dimension (matches the order of ``vehicles``).
    No client-side sorting is applied.

    Attributes:
        vehicles: All vehicles known to the account, in the order the
            ``/fleet/vehicles`` endpoint returned them. Includes
            historical (deactivated) vehicles so backfill joins work.
        drivers: All drivers known to the account, deduplicated by
            driver identifier. Active drivers are listed first; any
            deactivated drivers that share an ID with an active driver
            are dropped in favor of the active record.
        trips: Per-vehicle trip records spanning the full date range,
            collected by looping over ``vehicles`` and over the 28-day
            chunks the time window was split into.
        idling_events: Idling event records spanning the full date
            range, collected by looping over the same 28-day chunks
            (no per-vehicle inner loop -- the endpoint accepts the
            time window directly).
        date_range: Inclusive ``(start_date, end_date)`` the bundle was
            fetched for. Useful for downstream code to verify coverage
            without recomputing from chunk arithmetic.
        company: Company identifier configured on the source Provider,
            propagated into the unified output's ``company`` column.
            ``None`` when the Provider was constructed without one.
    """

    vehicles: list[SamsaraVehicle]
    drivers: list[SamsaraDriver]
    trips: list[Trip]
    idling_events: list[IdlingEvent]
    date_range: tuple[date, date]
    company: str | None


class SamsaraUtilizationFetcher:
    """
    Fetcher for Samsara utilization data across a UTC date range.

    Wraps a configured Samsara Provider and orchestrates the four
    endpoint calls needed for the event-grain unifier path:

        - vehicles:       single call, no time window
        - drivers:        two calls (active + deactivated), deduplicated
        - trips:          per-vehicle, per-chunk loop
        - idling_events:  per-chunk loop (no per-vehicle dimension)

    The fetcher does no transformation of returned records -- it
    fetches and assembles them into a typed bundle. Unit
    conversions, timezone interpretation, attribution math, and
    cross-midnight clipping all happen downstream in the unifier.

    Satisfies the ``UtilizationFetcher[SamsaraUtilizationBundle]``
    Protocol.

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
            ``SamsaraUtilizationBundle`` carrying the vehicles and
            drivers dimension lists plus the chunked event-grain
            trips and idling events.

        Raises:
            ValueError: If ``start_date > end_date``.
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

        chunks = list(
            iter_chunks(
                start_date,
                end_date,
                _MAX_CHUNK_DAYS,
                chunk_format='datetime',
            )
        )

        with self._provider.client() as client:
            vehicles: list[SamsaraVehicle] = list(
                client.fetch_all(SamsaraEndpoints.VEHICLES)
            )
            logger.debug('vehicles: %d records', len(vehicles))

            active_drivers: list[SamsaraDriver] = list(
                client.fetch_all(
                    SamsaraEndpoints.DRIVERS,
                    driver_activation_status=DriverActivationStatus.ACTIVE.value,
                )
            )
            deactivated_drivers: list[SamsaraDriver] = list(
                client.fetch_all(
                    SamsaraEndpoints.DRIVERS,
                    driver_activation_status=DriverActivationStatus.DEACTIVATED.value,
                )
            )
            drivers: list[SamsaraDriver] = _dedup_drivers(
                active_drivers,
                deactivated_drivers,
            )
            logger.debug(
                'drivers: %d active + %d deactivated -> %d after dedup',
                len(active_drivers),
                len(deactivated_drivers),
                len(drivers),
            )

            trips: list[Trip] = []
            for vehicle in vehicles:
                for chunk_start_dt, chunk_end_dt in chunks:
                    logger.debug(
                        'trips fetch: vehicle=%s chunk=%s..%s',
                        vehicle.vehicle_id,
                        chunk_start_dt,
                        chunk_end_dt,
                    )
                    trips.extend(
                        client.fetch_all(
                            SamsaraEndpoints.TRIPS,
                            vehicle_id=vehicle.vehicle_id,
                            start_time=chunk_start_dt,
                            end_time=chunk_end_dt,
                        )
                    )

            idling_events: list[IdlingEvent] = []
            for chunk_start_dt, chunk_end_dt in chunks:
                logger.debug(
                    'idling_events fetch: chunk=%s..%s',
                    chunk_start_dt,
                    chunk_end_dt,
                )
                idling_events.extend(
                    client.fetch_all(
                        SamsaraEndpoints.IDLING_EVENTS,
                        start_time=chunk_start_dt,
                        end_time=chunk_end_dt,
                    )
                )

        logger.info(
            'Samsara fetch complete: %d vehicles, %d drivers, %d trips, %d idling events',
            len(vehicles),
            len(drivers),
            len(trips),
            len(idling_events),
        )

        return SamsaraUtilizationBundle(
            vehicles=vehicles,
            drivers=drivers,
            trips=trips,
            idling_events=idling_events,
            date_range=(start_date, end_date),
            company=self._provider.company,
        )


def _dedup_drivers(
    active: list[SamsaraDriver],
    deactivated: list[SamsaraDriver],
) -> list[SamsaraDriver]:
    """
    Deduplicate ``active`` and ``deactivated`` drivers by primary ID.

    Active drivers are inserted first; deactivated drivers that
    share an ID with an active driver are dropped. Order within
    each group is preserved.

    Args:
        active: Drivers returned by the ``DRIVERS`` endpoint with
            ``driver_activation_status='active'``.
        deactivated: Drivers returned with
            ``driver_activation_status='deactivated'``.

    Returns:
        Combined, deduplicated list with active records winning on
        ID collisions.
    """
    seen: dict[str, SamsaraDriver] = {}
    for driver in active:
        seen[driver.driver_id] = driver
    for driver in deactivated:
        seen.setdefault(driver.driver_id, driver)
    return list(seen.values())
