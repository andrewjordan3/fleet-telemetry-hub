"""Tests for ``unifier.samsara_transform.transform_samsara_bundle``.

Uses real Pydantic instances for vehicles, drivers, trips, and
idling events so the transform's actual field-access paths and
Pydantic validation are exercised. Helpers build a default
``Trip`` / ``IdlingEvent`` with sensible defaults and accept
keyword overrides for the fields each test cares about.
"""

import logging
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from fleet_telemetry_hub.models.samsara_responses import (
    IdlingEvent,
    SamsaraDriver,
    SamsaraVehicle,
    Trip,
)
from fleet_telemetry_hub.unifier.samsara_transform import transform_samsara_bundle
from fleet_telemetry_hub.unifier.schema import EventType
from fleet_telemetry_hub.utilization.samsara_fetcher import SamsaraUtilizationBundle
from fleet_telemetry_hub.utilization.vehicle_trip import VehicleTrip

# Date constants shared across tests (keep cross-test consistency).
_MAY_14 = date(2026, 5, 14)
_MAY_20 = date(2026, 5, 20)


def _at(hour: int, minute: int = 0, second: int = 0) -> datetime:
    """Build a tz-aware UTC datetime on 2026-05-14."""
    return datetime(2026, 5, 14, hour, minute, second, tzinfo=UTC)


_VEHICLE_A_ID = '999999900000001'
_VEHICLE_B_ID = '999999900000002'
_VIN_A = 'TESTVIN0000000001'
_VIN_B = 'TESTVIN0000000002'
_DRIVER_SAM_ID = '1000001'
_DRIVER_SUZY_ID = '1000002'
_DRIVER_SAMMY_ID = '1000003'
_DRIVER_SAM_NAME = 'Sam Snowflake'
_DRIVER_SUZY_NAME = 'Suzy Snowflake'
_DRIVER_SAMMY_NAME = 'Sammy Snowflake'

# Common landmarks pinned once so the assertions read like a spec.
_ONE_HOUR_SECONDS = 3600
_FIFTEEN_MIN_SECONDS = 900
_FIVE_MIN_SECONDS = 300
_ONE_MILE_IN_METERS = 1609
_TWO_DRIVER_PLUS_IDLE_ROWS = 2
_TWO_WARN_RECORDS = 2


def _make_vehicle(
    *,
    vehicle_id: str = _VEHICLE_A_ID,
    vin: str | None = _VIN_A,
    name: str = 'TEST-001',
) -> SamsaraVehicle:
    return SamsaraVehicle.model_validate(
        {'id': vehicle_id, 'name': name, 'vin': vin}
    )


def _make_driver(
    *,
    driver_id: str = _DRIVER_SAM_ID,
    name: str = _DRIVER_SAM_NAME,
    activation_status: str = 'active',
) -> SamsaraDriver:
    return SamsaraDriver.model_validate(
        {
            'id': driver_id,
            'name': name,
            'driverActivationStatus': activation_status,
        }
    )


def _make_trip(  # noqa: PLR0913 -- test factory; one knob per field
    *,
    trip_id: str = '00000000-0000-0000-0000-000000001001',
    vehicle_id: str = _VEHICLE_A_ID,
    driver_id: str | None = _DRIVER_SAM_ID,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    distance_meters: int | None = _ONE_MILE_IN_METERS,
) -> VehicleTrip:
    """
    Build a ``VehicleTrip`` -- the wrapper the bundle's ``trips`` list
    actually holds. The inner ``Trip`` has no ``vehicleId`` in the API
    response shape; the queried ``vehicle_id`` is stamped on the
    wrapper instead. Pass ``distance_meters=None`` to exercise the
    unifier's null-distance soft-fallback path.
    """
    if start_dt is None:
        start_dt = _at(hour=8)
    if end_dt is None:
        end_dt = _at(hour=9)
    trip = Trip.model_validate(
        {
            'id': trip_id,
            'driverId': driver_id,
            'startMs': int(start_dt.timestamp() * 1000),
            'endMs': int(end_dt.timestamp() * 1000),
            'distanceMeters': distance_meters,
        }
    )
    return VehicleTrip.from_trip(trip, vehicle_id)


def _make_idling_event(  # noqa: PLR0913 -- test factory; one knob per field
    *,
    event_uuid: str = '00000000-0000-0000-0000-000000002001',
    vehicle_id: str = _VEHICLE_A_ID,
    operator_id: str = _DRIVER_SAM_ID,
    start_dt: datetime | None = None,
    duration_ms: int = 30 * 60 * 1000,  # 30 minutes
    pto_state: str = 'inactive',
    null_operator: bool = False,
) -> IdlingEvent:
    """
    Construct a valid ``IdlingEvent`` with sensible defaults.

    Defaults pin every nested object and required scalar so each
    behavioral test only has to override the field it cares about.

    ``null_operator=True`` omits the ``operator`` key entirely from
    the payload (mirroring the actual production shape for
    unattributed idle events) so the unifier's missing-operator
    handling path can be exercised. Use this rather than passing
    ``operator: None`` -- the unifier path is identical, but the
    key-absent shape is what the live API actually sends.
    """
    if start_dt is None:
        start_dt = _at(hour=10)
    payload: dict[str, Any] = {
        'asset': {'id': vehicle_id},
        'durationMilliseconds': duration_ms,
        'eventUuid': event_uuid,
        'fuelConsumedMilliliters': 0.5,
        'fuelCost': {'amount': '0.66', 'currency': 'usd'},
        'gaseousFuelConsumedGrams': 0,
        'gaseousFuelCost': {'amount': '0', 'currency': 'usd'},
        'ptoState': pto_state,
        'startTime': start_dt.isoformat().replace('+00:00', 'Z'),
        'latitude': 30.0,
        'longitude': -90.0,
    }
    if not null_operator:
        payload['operator'] = {'id': operator_id}
    return IdlingEvent.model_validate(payload)


def _make_bundle(  # noqa: PLR0913 -- bundles mirror the six SamsaraUtilizationBundle fields
    *,
    vehicles: list[SamsaraVehicle] | None = None,
    drivers: list[SamsaraDriver] | None = None,
    trips: list[Trip] | None = None,
    idling_events: list[IdlingEvent] | None = None,
    company: str | None = 'test_co',
    date_range: tuple[date, date] = (_MAY_14, _MAY_20),
) -> SamsaraUtilizationBundle:
    return SamsaraUtilizationBundle(
        vehicles=vehicles or [],
        drivers=drivers or [],
        trips=trips or [],
        idling_events=idling_events or [],
        date_range=date_range,
        company=company,
    )


def _default_vehicles_and_drivers() -> (
    tuple[list[SamsaraVehicle], list[SamsaraDriver]]
):
    """The most-common single-vehicle, single-driver dim setup."""
    return (
        [_make_vehicle()],
        [_make_driver()],
    )


class TestVinResolution:
    """VIN dim lookup with the locked ``unknown_vin`` fallback semantics."""

    def test_trip_vin_resolved_from_dim(self) -> None:
        """A trip whose vehicle_id is in the dim gets the dim's VIN."""

        vehicles, drivers = _default_vehicles_and_drivers()
        rows = transform_samsara_bundle(
            _make_bundle(vehicles=vehicles, drivers=drivers, trips=[_make_trip()])
        )

        assert len(rows) == 1
        assert rows[0].vin == _VIN_A

    def test_trip_vehicle_not_in_dim_falls_back(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A trip on an unknown vehicle_id uses the ``unknown_vin`` sentinel."""

        # Empty vehicles dim -> the trip's vehicle_id is not present.
        trip = _make_trip(vehicle_id='ghost_vehicle_id')
        with caplog.at_level(logging.WARNING):
            rows = transform_samsara_bundle(_make_bundle(trips=[trip]))

        assert rows[0].vin == 'unknown_vin'
        warn_records = [
            record
            for record in caplog.records
            if 'VIN unresolvable' in record.message
        ]
        assert len(warn_records) == 1
        message = warn_records[0].message
        assert 'in_dim_table=False' in message
        assert 'vehicle_id=ghost_vehicle_id' in message
        assert trip.trip.trip_id is not None
        assert trip.trip.trip_id in message

    def test_trip_vehicle_vin_is_none_falls_back(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Vehicle in dim but with ``vin=None`` still triggers fallback + warn."""

        vehicles = [_make_vehicle(vin=None)]
        drivers = [_make_driver()]
        with caplog.at_level(logging.WARNING):
            rows = transform_samsara_bundle(
                _make_bundle(
                    vehicles=vehicles, drivers=drivers, trips=[_make_trip()]
                )
            )

        assert rows[0].vin == 'unknown_vin'
        warn_records = [
            record
            for record in caplog.records
            if 'VIN unresolvable' in record.message
        ]
        assert len(warn_records) == 1
        assert 'in_dim_table=True' in warn_records[0].message

    def test_trip_vehicle_vin_is_whitespace_falls_back(self) -> None:
        """A whitespace-only VIN normalizes to None and triggers fallback."""

        vehicles = [_make_vehicle(vin='   ')]
        drivers = [_make_driver()]
        rows = transform_samsara_bundle(
            _make_bundle(vehicles=vehicles, drivers=drivers, trips=[_make_trip()])
        )

        assert rows[0].vin == 'unknown_vin'

    def test_idling_event_unknown_vehicle_falls_back(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Same fallback semantics apply to idling events."""

        event = _make_idling_event(vehicle_id='ghost_vehicle_id')
        with caplog.at_level(logging.WARNING):
            rows = transform_samsara_bundle(_make_bundle(idling_events=[event]))

        assert rows[0].vin == 'unknown_vin'
        warn_records = [
            record
            for record in caplog.records
            if 'VIN unresolvable' in record.message
        ]
        assert len(warn_records) == 1
        assert 'event_kind=idling_event' in warn_records[0].message

    def test_repeated_unresolvable_vehicle_increments_counter_each_time(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two trips on the same unknown vehicle yield two WARNINGs, not one."""

        trip_a = _make_trip(
            trip_id='00000000-0000-0000-0000-000000001001',
            vehicle_id='ghost_vehicle_id',
        )
        trip_b = _make_trip(
            trip_id='00000000-0000-0000-0000-000000001002',
            vehicle_id='ghost_vehicle_id',
            start_dt=_at(hour=14),
            end_dt=_at(hour=15),
        )
        with caplog.at_level(logging.WARNING):
            transform_samsara_bundle(_make_bundle(trips=[trip_a, trip_b]))

        warn_records = [
            record
            for record in caplog.records
            if 'VIN unresolvable' in record.message
        ]
        assert len(warn_records) == _TWO_WARN_RECORDS


class TestDriverResolution:
    """Driver-name dim lookup with the locked partial-null / warning semantics."""

    def test_driver_resolved_from_dim(self) -> None:
        """A trip with a known driver_id picks up the dim's name."""

        vehicles, drivers = _default_vehicles_and_drivers()
        rows = transform_samsara_bundle(
            _make_bundle(vehicles=vehicles, drivers=drivers, trips=[_make_trip()])
        )

        assert rows[0].driver_id == _DRIVER_SAM_ID
        assert rows[0].driver_name == _DRIVER_SAM_NAME

    def test_trip_with_null_driver_id_yields_null_pair(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``driver_id=None`` yields ``(None, None)`` with no warning."""

        vehicles = [_make_vehicle()]
        with caplog.at_level(logging.WARNING):
            rows = transform_samsara_bundle(
                _make_bundle(vehicles=vehicles, trips=[_make_trip(driver_id=None)])
            )

        assert rows[0].driver_id is None
        assert rows[0].driver_name is None
        assert not any(
            'driver name unresolvable' in record.message for record in caplog.records
        )

    def test_trip_with_empty_driver_id_yields_null_pair(self) -> None:
        """``driver_id=''`` normalizes to None and yields ``(None, None)``."""

        vehicles = [_make_vehicle()]
        rows = transform_samsara_bundle(
            _make_bundle(vehicles=vehicles, trips=[_make_trip(driver_id='')])
        )

        assert rows[0].driver_id is None
        assert rows[0].driver_name is None

    def test_trip_driver_not_in_dim_warns_and_sets_name_null(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unresolvable driver_id keeps its id and nulls the name, with WARNING."""

        vehicles = [_make_vehicle()]
        # Empty drivers dim -> the trip's driver_id is not present.
        with caplog.at_level(logging.WARNING):
            rows = transform_samsara_bundle(
                _make_bundle(vehicles=vehicles, trips=[_make_trip()])
            )

        assert rows[0].driver_id == _DRIVER_SAM_ID
        assert rows[0].driver_name is None
        warn_records = [
            record
            for record in caplog.records
            if 'driver name unresolvable' in record.message
        ]
        assert len(warn_records) == 1
        assert f'driver_id={_DRIVER_SAM_ID}' in warn_records[0].message

    def test_driver_name_unknown_in_dim_normalizes_to_null(self) -> None:
        """A dim name of 'Unknown' normalizes to None on lookup."""

        vehicles = [_make_vehicle()]
        drivers = [_make_driver(name='Unknown')]
        rows = transform_samsara_bundle(
            _make_bundle(vehicles=vehicles, drivers=drivers, trips=[_make_trip()])
        )

        assert rows[0].driver_id == _DRIVER_SAM_ID
        assert rows[0].driver_name is None

    def test_idling_event_resolves_driver_from_dim(self) -> None:
        """Idling events use ``operator.operator_id`` for the dim lookup."""

        vehicles, drivers = _default_vehicles_and_drivers()
        rows = transform_samsara_bundle(
            _make_bundle(
                vehicles=vehicles,
                drivers=drivers,
                idling_events=[_make_idling_event()],
            )
        )

        assert rows[0].driver_id == _DRIVER_SAM_ID
        assert rows[0].driver_name == _DRIVER_SAM_NAME


class TestTripToDrivingRow:
    """Trip → driving row, including idle-subtraction math and distance."""

    def test_single_trip_no_idling(self) -> None:
        """No idling on the same vehicle means duration is the full span."""

        vehicles, drivers = _default_vehicles_and_drivers()
        rows = transform_samsara_bundle(
            _make_bundle(vehicles=vehicles, drivers=drivers, trips=[_make_trip()])
        )

        assert len(rows) == 1
        row = rows[0]
        assert row.event_type is EventType.DRIVING
        assert row.duration_seconds == _ONE_HOUR_SECONDS
        assert row.distance_miles == 1.0

    def test_contained_idling_subtracts(self) -> None:
        """A 15-minute idling inside a 1-hour trip leaves 45 minutes."""

        vehicles, drivers = _default_vehicles_and_drivers()
        trip = _make_trip(start_dt=_at(hour=8), end_dt=_at(hour=9))
        idling = _make_idling_event(
            start_dt=_at(hour=8, minute=20),
            duration_ms=15 * 60 * 1000,
        )
        rows = transform_samsara_bundle(
            _make_bundle(
                vehicles=vehicles,
                drivers=drivers,
                trips=[trip],
                idling_events=[idling],
            )
        )

        trip_row = next(r for r in rows if r.event_type is EventType.DRIVING)
        assert trip_row.duration_seconds == _ONE_HOUR_SECONDS - _FIFTEEN_MIN_SECONDS

    def test_idling_on_other_vehicle_does_not_subtract(self) -> None:
        """Idling on a different vehicle_id does not subtract from this trip."""

        vehicles = [
            _make_vehicle(vehicle_id=_VEHICLE_A_ID, vin=_VIN_A),
            _make_vehicle(vehicle_id=_VEHICLE_B_ID, vin=_VIN_B, name='TEST-002'),
        ]
        drivers = [_make_driver()]
        trip = _make_trip(start_dt=_at(hour=8), end_dt=_at(hour=9))
        # Idling 8:20-8:35 but on vehicle B, not vehicle A.
        idling = _make_idling_event(
            vehicle_id=_VEHICLE_B_ID,
            start_dt=_at(hour=8, minute=20),
            duration_ms=15 * 60 * 1000,
        )
        rows = transform_samsara_bundle(
            _make_bundle(
                vehicles=vehicles,
                drivers=drivers,
                trips=[trip],
                idling_events=[idling],
            )
        )
        trip_row = next(r for r in rows if r.event_type is EventType.DRIVING)

        assert trip_row.duration_seconds == _ONE_HOUR_SECONDS

    def test_idling_fully_covering_trip_drops_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A trip whose computed duration is <= 0 is dropped; idle still emits."""

        vehicles, drivers = _default_vehicles_and_drivers()
        trip = _make_trip(start_dt=_at(hour=8), end_dt=_at(hour=9))
        # 3-hour idling 7:00-10:00 fully covers the trip.
        idling = _make_idling_event(
            start_dt=_at(hour=7),
            duration_ms=3 * 60 * 60 * 1000,
        )
        with caplog.at_level(logging.WARNING):
            rows = transform_samsara_bundle(
                _make_bundle(
                    vehicles=vehicles,
                    drivers=drivers,
                    trips=[trip],
                    idling_events=[idling],
                )
            )

        assert all(row.event_type is EventType.IDLE for row in rows)
        assert len(rows) == 1
        warn_records = [
            record
            for record in caplog.records
            if 'idle fully covers trip' in record.message
        ]
        assert len(warn_records) == 1

    def test_zero_distance_trip_yields_zero_miles(self) -> None:
        """``distance_meters=0`` yields ``distance_miles == 0.0``."""

        vehicles, drivers = _default_vehicles_and_drivers()
        trip = _make_trip(distance_meters=0)
        rows = transform_samsara_bundle(
            _make_bundle(vehicles=vehicles, drivers=drivers, trips=[trip])
        )

        assert rows[0].distance_miles == 0.0


class TestIdlingToIdleRow:
    """Idling → idle row, including end-time computation and floor-div duration."""

    def test_idle_row_has_no_distance_and_floor_div_duration(self) -> None:
        """The idle row has ``distance_miles=None`` and floor-div duration."""

        vehicles, drivers = _default_vehicles_and_drivers()
        # 30 min = 1_800_000 ms exactly -> 1800 s.
        idling = _make_idling_event(duration_ms=1_800_000)
        rows = transform_samsara_bundle(
            _make_bundle(
                vehicles=vehicles, drivers=drivers, idling_events=[idling]
            )
        )

        idle_row = rows[0]
        assert idle_row.event_type is EventType.IDLE
        assert idle_row.distance_miles is None
        thirty_minutes_seconds = 1800
        assert idle_row.duration_seconds == thirty_minutes_seconds

    def test_idle_end_time_computed_from_start_plus_duration(self) -> None:
        """``end_time_utc = start_time + duration_milliseconds``."""

        vehicles, drivers = _default_vehicles_and_drivers()
        start = datetime(2026, 5, 14, 13, 0, 0, tzinfo=UTC)
        idling = _make_idling_event(start_dt=start, duration_ms=3000)
        rows = transform_samsara_bundle(
            _make_bundle(
                vehicles=vehicles, drivers=drivers, idling_events=[idling]
            )
        )
        idle_row = rows[0]

        assert idle_row.start_time_utc == start
        assert idle_row.end_time_utc == start + timedelta(milliseconds=3000)
        three_seconds = 3
        assert idle_row.duration_seconds == three_seconds

    def test_sub_second_duration_floors_to_zero(self) -> None:
        """A 500ms idling event has ``duration_seconds == 0`` (floor division)."""

        vehicles, drivers = _default_vehicles_and_drivers()
        idling = _make_idling_event(duration_ms=500)
        rows = transform_samsara_bundle(
            _make_bundle(
                vehicles=vehicles, drivers=drivers, idling_events=[idling]
            )
        )

        assert rows[0].duration_seconds == 0


class TestIdleGapFill:
    """``attribute_idle_driver`` runs when both driver fields are null."""

    def test_null_operator_no_trips_stays_null(self) -> None:
        """Null operator_id with no trips on the same vehicle stays null, no warn."""

        vehicles = [_make_vehicle()]
        idling = _make_idling_event(operator_id='')
        rows = transform_samsara_bundle(
            _make_bundle(vehicles=vehicles, idling_events=[idling])
        )

        assert rows[0].driver_id is None
        assert rows[0].driver_name is None

    def test_null_operator_single_trip_gap_fills(self) -> None:
        """A single fully-overlapping trip populates both driver fields."""

        vehicles, drivers = _default_vehicles_and_drivers()
        idling = _make_idling_event(
            operator_id='',
            start_dt=_at(hour=10),
            duration_ms=30 * 60 * 1000,
        )
        # Sam's trip overlaps the idling window entirely.
        trip = _make_trip(start_dt=_at(hour=10), end_dt=_at(hour=11))
        rows = transform_samsara_bundle(
            _make_bundle(
                vehicles=vehicles,
                drivers=drivers,
                trips=[trip],
                idling_events=[idling],
            )
        )
        idle_row = next(row for row in rows if row.event_type is EventType.IDLE)

        assert idle_row.driver_id == _DRIVER_SAM_ID
        assert idle_row.driver_name == _DRIVER_SAM_NAME

    def test_yard_hand_keeps_null_driver(self) -> None:
        """5 min of trip inside 30 min idling: uncovered time wins -> null."""

        vehicles, drivers = _default_vehicles_and_drivers()
        idling = _make_idling_event(
            operator_id='',
            start_dt=_at(hour=10),
            duration_ms=30 * 60 * 1000,
        )
        trip = _make_trip(
            start_dt=_at(hour=10),
            end_dt=_at(hour=10, minute=5),
        )
        rows = transform_samsara_bundle(
            _make_bundle(
                vehicles=vehicles,
                drivers=drivers,
                trips=[trip],
                idling_events=[idling],
            )
        )
        idle_row = next(row for row in rows if row.event_type is EventType.IDLE)

        assert idle_row.driver_id is None
        assert idle_row.driver_name is None

    def test_two_drivers_overlap_warns_and_picks_winner(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two distinct drivers overlapping idling: most-overlap wins; warn logged."""

        vehicles = [_make_vehicle()]
        drivers = [
            _make_driver(driver_id=_DRIVER_SAM_ID, name=_DRIVER_SAM_NAME),
            _make_driver(driver_id=_DRIVER_SUZY_ID, name=_DRIVER_SUZY_NAME),
        ]
        idling = _make_idling_event(
            operator_id='',
            start_dt=_at(hour=10),
            duration_ms=30 * 60 * 1000,
        )
        sam_trip = _make_trip(
            trip_id='00000000-0000-0000-0000-000000001001',
            driver_id=_DRIVER_SAM_ID,
            start_dt=_at(hour=10),
            end_dt=_at(hour=10, minute=20),
        )
        suzy_trip = _make_trip(
            trip_id='00000000-0000-0000-0000-000000001002',
            driver_id=_DRIVER_SUZY_ID,
            start_dt=_at(hour=10, minute=20),
            end_dt=_at(hour=10, minute=30),
        )
        with caplog.at_level(logging.WARNING):
            rows = transform_samsara_bundle(
                _make_bundle(
                    vehicles=vehicles,
                    drivers=drivers,
                    trips=[sam_trip, suzy_trip],
                    idling_events=[idling],
                )
            )
        idle_row = next(row for row in rows if row.event_type is EventType.IDLE)

        assert idle_row.driver_id == _DRIVER_SAM_ID  # 20 min Sam beats 10 min Suzy
        warn_records = [
            record
            for record in caplog.records
            if 'Multiple drivers overlap' in record.message
        ]
        assert len(warn_records) == 1
        message = warn_records[0].message
        assert f'vehicle_id={_VEHICLE_A_ID}' in message
        assert 'bucket_distribution' in message

    def test_partial_null_does_not_trigger_gap_fill(self) -> None:
        """Set operator_id with unresolvable driver_name: no gap-fill from trips."""

        vehicles = [_make_vehicle()]
        # Drivers dim has Sammy only; the idling event's operator_id is
        # SAM_ID (not in the dim), so name resolves to None but driver_id
        # stays set. A trip with Sammy on the same vehicle exists; gap-fill
        # must NOT overwrite either field.
        drivers = [_make_driver(driver_id=_DRIVER_SAMMY_ID, name=_DRIVER_SAMMY_NAME)]
        idling = _make_idling_event(operator_id=_DRIVER_SAM_ID)
        trip = _make_trip(
            driver_id=_DRIVER_SAMMY_ID,
            start_dt=_at(hour=10),
            end_dt=_at(hour=11),
        )
        rows = transform_samsara_bundle(
            _make_bundle(
                vehicles=vehicles,
                drivers=drivers,
                trips=[trip],
                idling_events=[idling],
            )
        )
        idle_row = next(row for row in rows if row.event_type is EventType.IDLE)

        assert idle_row.driver_id == _DRIVER_SAM_ID
        assert idle_row.driver_name is None


class TestBundleLevelBehavior:
    """Bundle-wide behavior: company propagation, empty input, ordering."""

    def test_empty_bundle_returns_empty_list(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A bundle with no events returns an empty list with no warnings."""

        with caplog.at_level(logging.WARNING):
            rows = transform_samsara_bundle(_make_bundle())

        assert rows == []
        assert not any(record.levelno == logging.WARNING for record in caplog.records)

    def test_company_none_propagates(self) -> None:
        """``bundle.company=None`` flows into every emitted row."""

        vehicles, drivers = _default_vehicles_and_drivers()
        bundle = _make_bundle(
            vehicles=vehicles,
            drivers=drivers,
            trips=[_make_trip()],
            idling_events=[_make_idling_event()],
            company=None,
        )
        rows = transform_samsara_bundle(bundle)

        assert len(rows) == _TWO_DRIVER_PLUS_IDLE_ROWS
        assert all(row.company is None for row in rows)

    def test_company_value_propagates(self) -> None:
        """A populated ``bundle.company`` flows into every emitted row."""

        vehicles, drivers = _default_vehicles_and_drivers()
        bundle = _make_bundle(
            vehicles=vehicles,
            drivers=drivers,
            trips=[_make_trip()],
            idling_events=[_make_idling_event()],
            company='patriot',
        )
        rows = transform_samsara_bundle(bundle)

        assert all(row.company == 'patriot' for row in rows)

    def test_emission_order_is_trips_then_idling(self) -> None:
        """Trips appear first (in input order), then idling events (in input order)."""

        vehicles = [
            _make_vehicle(vehicle_id=_VEHICLE_A_ID, vin=_VIN_A),
            _make_vehicle(vehicle_id=_VEHICLE_B_ID, vin=_VIN_B, name='TEST-002'),
        ]
        drivers = [_make_driver()]
        trip_a = _make_trip(
            trip_id='00000000-0000-0000-0000-000000001001',
            start_dt=_at(hour=8),
            end_dt=_at(hour=9),
        )
        trip_b = _make_trip(
            trip_id='00000000-0000-0000-0000-000000001002',
            vehicle_id=_VEHICLE_B_ID,
            start_dt=_at(hour=13),
            end_dt=_at(hour=14),
        )
        idling_a = _make_idling_event(
            event_uuid='00000000-0000-0000-0000-000000002001',
            start_dt=_at(hour=10),
            duration_ms=15 * 60 * 1000,
        )
        idling_b = _make_idling_event(
            event_uuid='00000000-0000-0000-0000-000000002002',
            vehicle_id=_VEHICLE_B_ID,
            start_dt=_at(hour=12),
            duration_ms=10 * 60 * 1000,
        )
        rows = transform_samsara_bundle(
            _make_bundle(
                vehicles=vehicles,
                drivers=drivers,
                trips=[trip_a, trip_b],
                idling_events=[idling_a, idling_b],
            )
        )

        assert [row.event_type for row in rows] == [
            EventType.DRIVING,
            EventType.DRIVING,
            EventType.IDLE,
            EventType.IDLE,
        ]
        assert rows[0].start_time_utc == _at(hour=8)
        assert rows[1].start_time_utc == _at(hour=13)
        assert rows[2].start_time_utc == _at(hour=10)
        assert rows[3].start_time_utc == _at(hour=12)


class TestFinalInfoLog:
    """The final INFO log exposes the counter dict so dashboards can chart it."""

    def test_final_info_contains_all_counter_keys(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The summary INFO line shows all three counters (zero or non-zero)."""

        # Scope the level lift to this module's logger so a prior test
        # that pinned the ``fleet_telemetry_hub`` package logger to
        # WARNING (e.g. pipeline tests) does not filter INFO before
        # caplog sees it.
        with caplog.at_level(
            logging.INFO, logger='fleet_telemetry_hub.unifier.samsara_transform'
        ):
            transform_samsara_bundle(_make_bundle())

        complete_records = [
            record
            for record in caplog.records
            if 'Samsara transform complete' in record.message
        ]
        assert len(complete_records) == 1
        message = complete_records[0].message
        assert 'unknown_vin_fallbacks' in message
        assert 'unresolvable_drivers' in message
        assert 'non_positive_durations_dropped' in message
        assert 'null_distance_meters_fallback' in message


class TestIdlingEventMissingOperator:
    """Samsara omits the ``operator`` block for unattributed idle events."""

    def test_idling_event_with_missing_operator_emits_row(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No operator + no overlapping trip -> row emitted with null driver fields."""

        vehicles = [_make_vehicle()]
        drivers = [_make_driver()]
        idling = _make_idling_event(null_operator=True)

        with caplog.at_level(logging.WARNING):
            rows = transform_samsara_bundle(
                _make_bundle(
                    vehicles=vehicles, drivers=drivers, idling_events=[idling]
                )
            )

        assert len(rows) == 1
        row = rows[0]
        assert row.event_type is EventType.IDLE
        assert row.driver_id is None
        assert row.driver_name is None
        # No "missing operator" warning fired -- matches Motive's
        # ``IdleEvent.driver is None`` quiet path.
        assert not any(
            'missing operator' in record.message.lower()
            or 'null operator' in record.message.lower()
            for record in caplog.records
        )

    def test_idling_event_with_missing_operator_gap_fills_from_overlapping_trip(
        self,
    ) -> None:
        """Missing operator + covering trip -> idle row picks up the trip's driver."""

        vehicles = [_make_vehicle()]
        drivers = [_make_driver()]
        idling = _make_idling_event(
            null_operator=True,
            start_dt=_at(hour=10),
            duration_ms=30 * 60 * 1000,
        )
        # Sam's trip overlaps the idling window entirely.
        covering_trip = _make_trip(start_dt=_at(hour=10), end_dt=_at(hour=11))

        rows = transform_samsara_bundle(
            _make_bundle(
                vehicles=vehicles,
                drivers=drivers,
                trips=[covering_trip],
                idling_events=[idling],
            )
        )
        idle_row = next(row for row in rows if row.event_type is EventType.IDLE)

        assert idle_row.driver_id == _DRIVER_SAM_ID
        assert idle_row.driver_name == _DRIVER_SAM_NAME


class TestTripNullDistanceMetersSoftFallback:
    """Null ``distance_meters`` triggers the soft-fallback path, not a drop."""

    def test_trip_with_null_distance_meters_emits_row_with_null_distance(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Null distance -> row emitted with ``distance_miles=None`` + WARNING fires."""

        vehicles = [_make_vehicle()]
        drivers = [_make_driver()]
        trip = _make_trip(distance_meters=None)

        with caplog.at_level(logging.WARNING):
            rows = transform_samsara_bundle(
                _make_bundle(vehicles=vehicles, drivers=drivers, trips=[trip])
            )

        assert len(rows) == 1
        row = rows[0]
        assert row.event_type is EventType.DRIVING
        assert row.distance_miles is None
        # Other fields are still populated from the trip.
        assert row.vin == _VIN_A
        assert row.driver_id == _DRIVER_SAM_ID
        warn_records = [
            record
            for record in caplog.records
            if 'null distance_meters' in record.message
        ]
        assert len(warn_records) == 1

    def test_null_distance_meters_increments_counter_in_final_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Final INFO log reports ``null_distance_meters_fallback`` was incremented."""

        vehicles = [_make_vehicle()]
        drivers = [_make_driver()]
        trip = _make_trip(distance_meters=None)

        with caplog.at_level(
            logging.INFO, logger='fleet_telemetry_hub.unifier.samsara_transform'
        ):
            transform_samsara_bundle(
                _make_bundle(vehicles=vehicles, drivers=drivers, trips=[trip])
            )

        complete_records = [
            record
            for record in caplog.records
            if 'Samsara transform complete' in record.message
        ]
        assert len(complete_records) == 1
        assert "'null_distance_meters_fallback': 1" in complete_records[0].message

    def test_trip_with_present_distance_meters_does_not_increment_counter(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A normal trip leaves the counter at ``0`` and emits no fallback WARNING."""

        vehicles = [_make_vehicle()]
        drivers = [_make_driver()]
        trip = _make_trip()  # default distance_meters is _ONE_MILE_IN_METERS

        with caplog.at_level(
            logging.INFO, logger='fleet_telemetry_hub.unifier.samsara_transform'
        ):
            transform_samsara_bundle(
                _make_bundle(vehicles=vehicles, drivers=drivers, trips=[trip])
            )

        complete_records = [
            record
            for record in caplog.records
            if 'Samsara transform complete' in record.message
        ]
        assert len(complete_records) == 1
        assert "'null_distance_meters_fallback': 0" in complete_records[0].message
        assert not any(
            'null distance_meters' in record.message for record in caplog.records
        )
