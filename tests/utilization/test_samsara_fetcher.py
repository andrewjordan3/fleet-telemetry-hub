"""Tests for the event-grain SamsaraUtilizationFetcher.

Uses ``unittest.mock`` for the Provider/TelemetryClient context
pair but constructs real Pydantic record instances for the data
the fake client yields. Tests assert the fetcher orchestrates the
four endpoints (VEHICLES, DRIVERS x2, TRIPS, IDLING_EVENTS)
correctly without performing any transformation: identity equality
is used for records flowing through the bundle, and ``is`` identity
is used for endpoint constants on the call list.

All identifiers in test data are synthetic.
"""

import dataclasses
from datetime import UTC, date, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from fleet_telemetry_hub.client import TelemetryClient
from fleet_telemetry_hub.models.samsara_requests import SamsaraEndpoints
from fleet_telemetry_hub.models.samsara_responses import (
    IdlingEvent,
    SamsaraDriver,
    SamsaraVehicle,
    Trip,
)
from fleet_telemetry_hub.provider import Provider
from fleet_telemetry_hub.utilization import (
    SamsaraUtilizationBundle,
    SamsaraUtilizationFetcher,
    UtilizationFetcher,
    VehicleTrip,
)

# Date constants. _MAY_14..16 is a single 28-day chunk; the 30-day
# range below exercises the multi-chunk path.
_MAY_14 = date(2026, 5, 14)
_MAY_20 = date(2026, 5, 20)
_JUN_12 = date(2026, 6, 12)  # 30 days after MAY_14 inclusive -> 2 chunks


def _make_samsara_vehicle(vehicle_id: str, vin: str, name: str) -> SamsaraVehicle:
    return SamsaraVehicle.model_validate({'id': vehicle_id, 'name': name, 'vin': vin})


def _make_samsara_driver(
    driver_id: str,
    name: str,
    activation_status: str,
) -> SamsaraDriver:
    return SamsaraDriver.model_validate(
        {
            'id': driver_id,
            'name': name,
            'driverActivationStatus': activation_status,
        }
    )


def _make_trip(
    trip_id: str,
    driver_id: str | None,
    start_dt: datetime,
    end_dt: datetime,
    distance_meters: int,
) -> Trip:
    """Build a ``Trip`` with the on-wire shape -- no ``vehicleId`` in the payload."""
    return Trip.model_validate(
        {
            'id': trip_id,
            'driverId': driver_id,
            'startMs': int(start_dt.timestamp() * 1000),
            'endMs': int(end_dt.timestamp() * 1000),
            'distanceMeters': distance_meters,
        }
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _FakeBackend:
    """Routes fake_client.fetch_all calls to the right preconfigured iterator."""

    vehicles: list[SamsaraVehicle]
    active_drivers: list[SamsaraDriver]
    deactivated_drivers: list[SamsaraDriver]
    trips_by_vehicle_chunk: dict[tuple[str, datetime, datetime], list[Trip]]
    idling_by_chunk: dict[tuple[datetime, datetime], list[IdlingEvent]]

    def dispatch(self, endpoint: Any, **params: Any) -> Any:
        records: list[Any] = []
        if endpoint is SamsaraEndpoints.VEHICLES:
            records = list(self.vehicles)
        elif endpoint is SamsaraEndpoints.DRIVERS:
            status = params['driver_activation_status']
            if status == 'active':
                records = list(self.active_drivers)
            elif status == 'deactivated':
                records = list(self.deactivated_drivers)
        elif endpoint is SamsaraEndpoints.TRIPS:
            trips_key = (
                params['vehicle_id'],
                params['start_time'],
                params['end_time'],
            )
            records = list(self.trips_by_vehicle_chunk.get(trips_key, []))
        elif endpoint is SamsaraEndpoints.IDLING_EVENTS:
            idling_key = (params['start_time'], params['end_time'])
            records = list(self.idling_by_chunk.get(idling_key, []))
        return iter(records)


def _build_fake_provider_and_client(
    *,
    vehicles: list[SamsaraVehicle] | None = None,
    active_drivers: list[SamsaraDriver] | None = None,
    deactivated_drivers: list[SamsaraDriver] | None = None,
    trips_by_vehicle_chunk: dict[tuple[str, datetime, datetime], list[Trip]]
    | None = None,
    idling_by_chunk: dict[tuple[datetime, datetime], list[IdlingEvent]] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Construct a fake (Provider, TelemetryClient) pair wired to a ``_FakeBackend``."""

    backend = _FakeBackend(
        vehicles=vehicles or [],
        active_drivers=active_drivers or [],
        deactivated_drivers=deactivated_drivers or [],
        trips_by_vehicle_chunk=trips_by_vehicle_chunk or {},
        idling_by_chunk=idling_by_chunk or {},
    )

    fake_client = MagicMock(spec=TelemetryClient)
    fake_client.__enter__.return_value = fake_client
    fake_client.__exit__.return_value = None
    fake_client.fetch_all.side_effect = backend.dispatch

    fake_provider = MagicMock(spec=Provider)
    fake_provider.client.return_value = fake_client

    return fake_provider, fake_client


def _single_chunk(start_date: date, end_date: date) -> tuple[datetime, datetime]:
    """Return the half-open UTC datetime chunk for a sub-28-day range."""
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=UTC)
    one_past = end_date.toordinal() + 1
    end_dt_date = date.fromordinal(one_past)
    end_dt = datetime(end_dt_date.year, end_dt_date.month, end_dt_date.day, tzinfo=UTC)
    return start_dt, end_dt


class TestSamsaraUtilizationFetcherProtocolConformance:
    """The concrete fetcher satisfies the runtime Protocol."""

    def test_isinstance_of_utilization_fetcher_protocol(self) -> None:
        """Should pass isinstance against the runtime_checkable Protocol."""

        fake_provider, _ = _build_fake_provider_and_client()
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        assert isinstance(fetcher, UtilizationFetcher)


class TestSamsaraUtilizationBundleShape:
    """Bundle field set and frozen-dataclass behavior."""

    def test_field_set_is_exact(self) -> None:
        """Bundle declares exactly the documented field set."""

        fields = {f.name for f in dataclasses.fields(SamsaraUtilizationBundle)}

        assert fields == {
            'vehicles',
            'drivers',
            'trips',
            'idling_events',
            'date_range',
            'company',
        }

    def test_bundle_is_frozen_dataclass(self) -> None:
        """Assigning to a bundle attribute raises FrozenInstanceError."""

        bundle = SamsaraUtilizationBundle(
            vehicles=[],
            drivers=[],
            trips=[],
            idling_events=[],
            date_range=(_MAY_14, _MAY_14),
            company=None,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            bundle.vehicles = []  # pyright: ignore[reportAttributeAccessIssue]


class TestSamsaraUtilizationFetcherValidation:
    """Date-range validation: reversed range raises before any API call."""

    def test_reversed_range_raises_value_error(self) -> None:
        """``start_date > end_date`` raises ValueError with both dates in the message."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        with pytest.raises(ValueError, match='must be <=') as exc_info:
            fetcher.fetch(_MAY_20, _MAY_14)

        assert str(_MAY_20) in str(exc_info.value)
        assert str(_MAY_14) in str(exc_info.value)
        assert fake_client.fetch_all.call_count == 0


class TestSamsaraUtilizationFetcherDateRange:
    """The bundle's date_range reflects the fetch arguments."""

    def test_date_range_matches_input(self) -> None:
        """``bundle.date_range`` is the inclusive ``(start, end)`` tuple."""

        fake_provider, _ = _build_fake_provider_and_client()
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_20)

        assert bundle.date_range == (_MAY_14, _MAY_20)


class TestSamsaraUtilizationFetcherDimensionPassThrough:
    """Vehicles and active drivers flow into the bundle by identity."""

    def test_vehicles_pass_through_by_identity(self) -> None:
        """Bundle's ``vehicles`` list is the same instances the API returned."""

        v_a = _make_samsara_vehicle('999999900000001', 'TESTVIN0000000001', 'TEST-001')
        v_b = _make_samsara_vehicle('999999900000002', 'TESTVIN0000000002', 'TEST-002')
        fake_provider, _ = _build_fake_provider_and_client(vehicles=[v_a, v_b])
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_20)

        assert bundle.vehicles[0] is v_a
        assert bundle.vehicles[1] is v_b

    def test_active_drivers_pass_through_by_identity(self) -> None:
        """Active driver instances appear in the bundle unchanged."""

        d_sam = _make_samsara_driver('1000001', 'Sam Snowflake', 'active')
        d_suzy = _make_samsara_driver('1000002', 'Suzy Snowflake', 'active')
        fake_provider, _ = _build_fake_provider_and_client(
            active_drivers=[d_sam, d_suzy],
        )
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_20)

        assert d_sam in bundle.drivers
        assert d_suzy in bundle.drivers
        bundle_by_id = {d.driver_id: d for d in bundle.drivers}
        assert bundle_by_id['1000001'] is d_sam
        assert bundle_by_id['1000002'] is d_suzy


class TestSamsaraUtilizationFetcherDriverDedup:
    """Active drivers win when the same ID appears in both status calls."""

    def test_dedup_prefers_active_instance(self) -> None:
        """A driver in both lists appears once with the active-version instance."""

        d_sam_active = _make_samsara_driver('1000001', 'Sam Snowflake', 'active')
        d_sam_deactivated = _make_samsara_driver(
            '1000001', 'Sam Snowflake (Deactivated)', 'deactivated'
        )
        d_sammy = _make_samsara_driver('1000003', 'Sammy Snowflake', 'deactivated')

        fake_provider, _ = _build_fake_provider_and_client(
            active_drivers=[d_sam_active],
            deactivated_drivers=[d_sam_deactivated, d_sammy],
        )
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_20)

        bundle_by_id = {d.driver_id: d for d in bundle.drivers}
        assert bundle_by_id['1000001'] is d_sam_active
        assert d_sam_deactivated not in bundle.drivers
        assert bundle_by_id['1000003'] is d_sammy
        assert len(bundle.drivers) == 2  # noqa: PLR2004


class TestSamsaraUtilizationFetcherTripsAndIdlingChunking:
    """The fetcher loops per-vehicle x per-chunk over TRIPS; per-chunk over IDLING_EVENTS."""

    def _vehicles(self) -> list[SamsaraVehicle]:
        return [
            _make_samsara_vehicle('999999900000001', 'TESTVIN0000000001', 'TEST-001'),
            _make_samsara_vehicle('999999900000002', 'TESTVIN0000000002', 'TEST-002'),
        ]

    def test_single_chunk_two_vehicles(self) -> None:
        """7-day range yields one chunk: trips called n_vehicles times, idling once."""

        vehicles = self._vehicles()
        fake_provider, fake_client = _build_fake_provider_and_client(vehicles=vehicles)
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        fetcher.fetch(_MAY_14, _MAY_20)

        trips_calls = [
            call
            for call in fake_client.fetch_all.call_args_list
            if call.args[0] is SamsaraEndpoints.TRIPS
        ]
        idling_calls = [
            call
            for call in fake_client.fetch_all.call_args_list
            if call.args[0] is SamsaraEndpoints.IDLING_EVENTS
        ]
        assert len(trips_calls) == len(vehicles)
        assert len(idling_calls) == 1

    def test_thirty_day_range_yields_two_chunks(self) -> None:
        """30-day range -> 2 chunks: trips n_vehicles x 2 calls; idling 2 calls."""

        vehicles = self._vehicles()
        fake_provider, fake_client = _build_fake_provider_and_client(vehicles=vehicles)
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        fetcher.fetch(_MAY_14, _JUN_12)

        trips_calls = [
            call
            for call in fake_client.fetch_all.call_args_list
            if call.args[0] is SamsaraEndpoints.TRIPS
        ]
        idling_calls = [
            call
            for call in fake_client.fetch_all.call_args_list
            if call.args[0] is SamsaraEndpoints.IDLING_EVENTS
        ]
        expected_trip_calls = len(vehicles) * 2
        expected_idling_calls = 2
        assert len(trips_calls) == expected_trip_calls
        assert len(idling_calls) == expected_idling_calls

    def test_trips_call_params_match_vehicle_and_chunk(self) -> None:
        """Each trips call carries one vehicle's ID and one chunk's datetime bounds."""

        vehicles = self._vehicles()
        chunk_start, chunk_end = _single_chunk(_MAY_14, _MAY_20)
        fake_provider, fake_client = _build_fake_provider_and_client(vehicles=vehicles)
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        fetcher.fetch(_MAY_14, _MAY_20)

        trips_calls = [
            call
            for call in fake_client.fetch_all.call_args_list
            if call.args[0] is SamsaraEndpoints.TRIPS
        ]
        observed = [
            (
                call.kwargs['vehicle_id'],
                call.kwargs['start_time'],
                call.kwargs['end_time'],
            )
            for call in trips_calls
        ]
        assert observed == [
            (vehicles[0].vehicle_id, chunk_start, chunk_end),
            (vehicles[1].vehicle_id, chunk_start, chunk_end),
        ]


class TestSamsaraUtilizationFetcherTripOrdering:
    """Bundle's trip list orders by vehicle iteration, then by chunk chronology."""

    def test_trip_ordering_outer_by_vehicle_inner_by_chunk(self) -> None:
        """Trips appear in (vehicle_index, chunk_index) order."""

        v_a = _make_samsara_vehicle('999999900000001', 'TESTVIN0000000001', 'TEST-001')
        v_b = _make_samsara_vehicle('999999900000002', 'TESTVIN0000000002', 'TEST-002')

        # 30-day range -> two chunks.
        chunks = [
            (datetime(2026, 5, 14, tzinfo=UTC), datetime(2026, 6, 11, tzinfo=UTC)),
            (datetime(2026, 6, 11, tzinfo=UTC), datetime(2026, 6, 13, tzinfo=UTC)),
        ]
        trip_a1 = _make_trip(
            '00000000-0000-0000-0000-000000001001',
            '1000001',
            chunks[0][0],
            chunks[0][0],
            100,
        )
        trip_a2 = _make_trip(
            '00000000-0000-0000-0000-000000001002',
            '1000001',
            chunks[1][0],
            chunks[1][0],
            200,
        )
        trip_b1 = _make_trip(
            '00000000-0000-0000-0000-000000001003',
            '1000002',
            chunks[0][0],
            chunks[0][0],
            300,
        )
        trip_b2 = _make_trip(
            '00000000-0000-0000-0000-000000001004',
            '1000002',
            chunks[1][0],
            chunks[1][0],
            400,
        )

        trips_map: dict[tuple[str, datetime, datetime], list[Trip]] = {
            (v_a.vehicle_id, chunks[0][0], chunks[0][1]): [trip_a1],
            (v_a.vehicle_id, chunks[1][0], chunks[1][1]): [trip_a2],
            (v_b.vehicle_id, chunks[0][0], chunks[0][1]): [trip_b1],
            (v_b.vehicle_id, chunks[1][0], chunks[1][1]): [trip_b2],
        }
        fake_provider, _ = _build_fake_provider_and_client(
            vehicles=[v_a, v_b],
            trips_by_vehicle_chunk=trips_map,
        )
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        bundle = fetcher.fetch(_MAY_14, _JUN_12)

        # Bundle.trips is now list[VehicleTrip]; the inner Trip is
        # identity-equal to the canned record and the wrapper is
        # stamped with the queried vehicle_id.
        assert all(isinstance(item, VehicleTrip) for item in bundle.trips)
        assert [vt.trip for vt in bundle.trips] == [
            trip_a1,
            trip_a2,
            trip_b1,
            trip_b2,
        ]
        assert [vt.vehicle_id for vt in bundle.trips] == [
            v_a.vehicle_id,
            v_a.vehicle_id,
            v_b.vehicle_id,
            v_b.vehicle_id,
        ]


class TestSamsaraUtilizationFetcherEmptyResults:
    """When every endpoint returns no records, the bundle has empty lists."""

    def test_empty_everywhere_produces_empty_bundle_lists(self) -> None:
        """No vehicles, no drivers, no trips, no idling events -> empty lists, no crash."""

        fake_provider, _ = _build_fake_provider_and_client()
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_20)

        assert bundle.vehicles == []
        assert bundle.drivers == []
        assert bundle.trips == []
        assert bundle.idling_events == []


class TestSamsaraUtilizationFetcherClientLifecycle:
    """The fetcher opens and closes the client context exactly once."""

    def test_client_enter_and_exit_called_once(self) -> None:
        """``__enter__`` and ``__exit__`` are each invoked exactly once per fetch."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        fetcher.fetch(_MAY_14, _MAY_20)

        assert fake_client.__enter__.call_count == 1
        assert fake_client.__exit__.call_count == 1
