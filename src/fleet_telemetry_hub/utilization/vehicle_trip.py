"""``VehicleTrip`` -- wrapper pairing a Samsara ``Trip`` with its known vehicle_id.

The Samsara ``/v1/fleet/trips`` endpoint queries by ``vehicleId`` but
does not echo that field back per-trip in the response body. This
wrapper restores the vehicle association at the fetcher boundary so
downstream consumers (the unifier transform, in particular) can rely
on a non-nullable ``vehicle_id`` by construction rather than
threading it as a side-channel through the call stack.

The wrapper is intentionally a frozen+slotted dataclass, not a
Pydantic model: it carries post-validation, in-process data, never
crosses an external boundary, and never needs JSON
serialization. Pydantic stays reserved for unvalidated input at the
API surface.
"""

from dataclasses import dataclass
from typing import Self

from fleet_telemetry_hub.models.samsara_responses import Trip

__all__: list[str] = ['VehicleTrip']


@dataclass(frozen=True, slots=True)
class VehicleTrip:
    """
    A Samsara ``Trip`` paired with the queried vehicle's identifier.

    Constructed at the ``SamsaraUtilizationFetcher`` boundary,
    immediately after the ``/v1/fleet/trips`` API call returns.
    Replaces the raw ``list[Trip]`` previously held by
    ``SamsaraUtilizationBundle``.

    Attributes:
        trip: The Samsara ``Trip`` record as returned by the API. The
            inner ``Trip`` has no ``vehicle_id`` field of its own;
            that association lives on this wrapper.
        vehicle_id: Samsara vehicle identifier for the queried
            vehicle. Non-nullable by construction; validated as a
            non-empty string in ``__post_init__``.
    """

    trip: Trip
    vehicle_id: str

    def __post_init__(self) -> None:
        """Validate that ``vehicle_id`` is a non-empty string."""
        if not self.vehicle_id:
            raise ValueError(
                f'vehicle_id must be a non-empty string, got {self.vehicle_id!r}'
            )

    @classmethod
    def from_trip(cls, trip: Trip, vehicle_id: str) -> Self:
        """
        Build a ``VehicleTrip`` by pairing a ``Trip`` with its queried ``vehicle_id``.

        This is the canonical stamping site referenced by the
        fetcher; ``cls(trip=trip, vehicle_id=vehicle_id)`` is
        equivalent but reads less well at the call site.

        Args:
            trip: The ``Trip`` record from the API response.
            vehicle_id: The Samsara vehicle identifier used in the
                query that returned this trip.

        Returns:
            A new ``VehicleTrip`` wrapping the inputs.

        Raises:
            ValueError: If ``vehicle_id`` is empty.
        """
        return cls(trip=trip, vehicle_id=vehicle_id)
