"""Tests for ``VehicleTrip``: construction, validation, frozen-ness, slots, classmethod.

``VehicleTrip`` is the small frozen+slotted wrapper the Samsara
fetcher uses to stamp a non-nullable ``vehicle_id`` onto each parsed
``Trip``. The Samsara ``/v1/fleet/trips`` endpoint queries by
``vehicleId`` but does not echo it back in the response body, so this
wrapper restores the association at the fetcher boundary.
"""

import dataclasses
from datetime import UTC, datetime

import pytest

from fleet_telemetry_hub.models.samsara_responses import Trip
from fleet_telemetry_hub.utilization.vehicle_trip import VehicleTrip

_VEHICLE_ID = '999999900000001'
_DRIVER_ID_INT = 7046697

_TRIP_START_UTC = datetime(2026, 5, 14, 13, 0, 0, tzinfo=UTC)
_TRIP_END_UTC = datetime(2026, 5, 14, 14, 0, 0, tzinfo=UTC)


def _make_trip() -> Trip:
    """Build a plausible Trip with no vehicle_id (matches the API shape)."""
    return Trip.model_validate(
        {
            'id': '00000000-0000-0000-0000-000000001001',
            'driverId': _DRIVER_ID_INT,
            'startMs': int(_TRIP_START_UTC.timestamp() * 1000),
            'endMs': int(_TRIP_END_UTC.timestamp() * 1000),
            'distanceMeters': 1609,
        }
    )


class TestVehicleTripConstruction:
    """Direct constructor accepts a Trip and a non-empty vehicle_id."""

    def test_direct_construction_populates_attributes(self) -> None:
        """The wrapper exposes ``trip`` and ``vehicle_id`` as given."""

        trip = _make_trip()
        wrapper = VehicleTrip(trip=trip, vehicle_id=_VEHICLE_ID)

        assert wrapper.trip is trip
        assert wrapper.vehicle_id == _VEHICLE_ID

    def test_from_trip_equivalent_to_direct_constructor(self) -> None:
        """``VehicleTrip.from_trip(t, vid)`` matches ``VehicleTrip(trip=t, vehicle_id=vid)``."""

        trip = _make_trip()
        from_classmethod = VehicleTrip.from_trip(trip, _VEHICLE_ID)
        from_constructor = VehicleTrip(trip=trip, vehicle_id=_VEHICLE_ID)

        assert from_classmethod == from_constructor
        # Identity check on the inner trip: the wrapper does not copy.
        assert from_classmethod.trip is trip


class TestVehicleTripValidation:
    """``__post_init__`` rejects an empty ``vehicle_id`` on both construction paths."""

    def test_empty_vehicle_id_via_constructor_raises_value_error(self) -> None:
        """Direct constructor with ``vehicle_id=''`` raises ``ValueError``."""

        with pytest.raises(ValueError, match='non-empty string'):
            VehicleTrip(trip=_make_trip(), vehicle_id='')

    def test_empty_vehicle_id_via_from_trip_raises_value_error(self) -> None:
        """``from_trip`` with an empty id raises -- the validator runs on both paths."""

        with pytest.raises(ValueError, match='non-empty string'):
            VehicleTrip.from_trip(_make_trip(), '')


class TestVehicleTripImmutability:
    """The wrapper is a frozen+slotted dataclass."""

    def test_assigning_to_trip_raises_frozen_instance_error(self) -> None:
        """Reassigning ``trip`` after construction is rejected by ``frozen=True``."""

        wrapper = VehicleTrip.from_trip(_make_trip(), _VEHICLE_ID)
        with pytest.raises(dataclasses.FrozenInstanceError):
            wrapper.trip = _make_trip()  # type: ignore[misc]

    def test_assigning_to_vehicle_id_raises_frozen_instance_error(self) -> None:
        """Reassigning ``vehicle_id`` after construction is rejected by ``frozen=True``."""

        wrapper = VehicleTrip.from_trip(_make_trip(), _VEHICLE_ID)
        with pytest.raises(dataclasses.FrozenInstanceError):
            wrapper.vehicle_id = '999999900000002'  # type: ignore[misc]

    def test_class_declares_slots(self) -> None:
        """``slots=True`` materializes as ``__slots__`` on the class itself."""

        # ``frozen=True`` already blocks every ``__setattr__``, so the
        # behavioral slots check (assigning a new attribute) is
        # indistinguishable from the frozen check above. Assert on
        # the structural attribute instead.
        assert hasattr(VehicleTrip, '__slots__')
        assert set(VehicleTrip.__slots__) == {'trip', 'vehicle_id'}
