"""Tests for SamsaraUtilizationFetcher.

Mirrors the structure of test_motive_fetcher.py. Uses ``unittest.mock``
for the Provider/TelemetryClient context-manager pair but constructs
real Pydantic record instances (FuelEnergyVehicleReport,
DriverFuelEnergyReport, DriverVehicleAssignment, IdlingEvent) for the
data the fake client yields. Tests assert the fetcher orchestrates
the four Samsara endpoints correctly without performing any
transformation: identity equality is used for records flowing through
the bundle, and ``is`` identity is used for endpoint constants on the
call list.

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
    DriverFuelEnergyReport,
    DriverVehicleAssignment,
    FuelEnergyVehicleReport,
    IdlingEvent,
)
from fleet_telemetry_hub.provider import Provider
from fleet_telemetry_hub.utilization import (
    SamsaraUtilizationBundle,
    SamsaraUtilizationFetcher,
    UtilizationFetcher,
)

# Single-day and multi-day date constants used across tests.
_MAY_14 = date(2026, 5, 14)
_MAY_15 = date(2026, 5, 15)
_MAY_16 = date(2026, 5, 16)

_MAY_14_START = datetime(2026, 5, 14, 0, 0, 0, tzinfo=UTC)
_MAY_15_START = datetime(2026, 5, 15, 0, 0, 0, tzinfo=UTC)
_MAY_16_START = datetime(2026, 5, 16, 0, 0, 0, tzinfo=UTC)
_MAY_17_START = datetime(2026, 5, 17, 0, 0, 0, tzinfo=UTC)


def _make_vehicle_fuel_energy_report(
    vehicle_id: str = '999999900000001',
) -> FuelEnergyVehicleReport:
    return FuelEnergyVehicleReport.model_validate(
        {
            'vehicle': {
                'energyType': 'fuel',
                'id': vehicle_id,
                'name': f'TEST-{vehicle_id} (Tractor)',
                'externalIds': {
                    'samsara.serial': f'TESTSERIAL{vehicle_id[-2:]}',
                    'samsara.vin': f'TESTVIN{vehicle_id[-11:]}',
                },
            },
            'efficiencyMpge': 5.5,
            'energyUsedKwh': 0,
            'fuelConsumedMl': 200000,
            'distanceTraveledMeters': 500000,
            'estCarbonEmissionsKg': 540.0,
            'estFuelEnergyCost': {'amount': 300.0, 'currencyCode': 'USD'},
            'engineRunTimeDurationMs': 72000000,
            'engineIdleTimeDurationMs': 40000000,
        }
    )


def _make_driver_fuel_energy_report(
    driver_id: str = '1000001',
) -> DriverFuelEnergyReport:
    return DriverFuelEnergyReport.model_validate(
        {
            'driver': {'id': driver_id, 'name': f'Test Driver {driver_id}'},
            'efficiencyMpge': 6.5,
            'energyUsedKwh': 0,
            'fuelConsumedMl': 35000,
            'distanceTraveledMeters': 100000,
            'estCarbonEmissionsKg': 93.0,
            'estFuelEnergyCost': {'amount': 51.0, 'currencyCode': 'USD'},
            'engineRunTimeDurationMs': 11000000,
            'engineIdleTimeDurationMs': 700000,
        }
    )


def _make_driver_vehicle_assignment(
    driver_id: str = '1000001',
    vehicle_id: str = '999999900000001',
) -> DriverVehicleAssignment:
    return DriverVehicleAssignment.model_validate(
        {
            'startTime': '2026-05-14T07:00:00Z',
            'endTime': '2026-05-14T19:00:00Z',
            'isPassenger': False,
            'assignedAtTime': '',
            'assignmentType': 'HOS',
            'driver': {'id': driver_id, 'name': f'Test Driver {driver_id}'},
            'vehicle': {
                'id': vehicle_id,
                'name': f'TEST-{vehicle_id}',
                'externalIds': {
                    'samsara.vin': f'TESTVIN{vehicle_id[-11:]}',
                    'samsara.serial': f'TESTSERIAL{vehicle_id[-2:]}',
                },
            },
        }
    )


def _make_idling_event(
    event_uuid: str = '00000000-0000-0000-0000-000000000001',
) -> IdlingEvent:
    return IdlingEvent.model_validate(
        {
            'airTemperatureMillicelsius': 12938,
            'asset': {'id': 999999900000005},
            'durationMilliseconds': 331886,
            'eventUuid': event_uuid,
            'fuelConsumedMilliliters': 451.07,
            'fuelCost': {'amount': '0.66', 'currency': 'usd'},
            'gaseousFuelConsumedGrams': 0,
            'gaseousFuelCost': {'amount': '0.00', 'currency': 'usd'},
            'operator': {'id': 1000006},
            'ptoState': 'inactive',
            'startTime': '2026-05-14T13:13:01.078Z',
            'latitude': 30.0,
            'longitude': -90.0,
        }
    )


def _build_fake_provider_and_client(
    *,
    vehicle_records_by_window: dict[
        tuple[datetime, datetime], list[FuelEnergyVehicleReport]
    ]
    | None = None,
    driver_records_by_window: dict[
        tuple[datetime, datetime], list[DriverFuelEnergyReport]
    ]
    | None = None,
    assignment_records: list[DriverVehicleAssignment] | None = None,
    idling_event_records: list[IdlingEvent] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """
    Construct a fake (Provider, TelemetryClient) pair wired together.

    The fake client's ``fetch_all`` dispatches on the endpoint constant:
    VEHICLE_FUEL_ENERGY / DRIVER_FUEL_ENERGY lookups use the
    (start_date, end_date) window as the key into the per-window
    record maps; DRIVER_VEHICLE_ASSIGNMENTS and IDLING_EVENTS each
    return their configured flat list regardless of window.

    Returns (fake_provider, fake_client) so tests can also assert on
    the client side (e.g. context-manager invocation counts).
    """

    vehicle_by_window: dict[
        tuple[datetime, datetime], list[FuelEnergyVehicleReport]
    ] = vehicle_records_by_window or {}
    driver_by_window: dict[tuple[datetime, datetime], list[DriverFuelEnergyReport]] = (
        driver_records_by_window or {}
    )
    assignments: list[DriverVehicleAssignment] = assignment_records or []
    idling_events: list[IdlingEvent] = idling_event_records or []

    fake_client = MagicMock(spec=TelemetryClient)
    fake_client.__enter__.return_value = fake_client
    fake_client.__exit__.return_value = None

    def fake_fetch_all(endpoint: Any, **params: Any) -> Any:
        if endpoint is SamsaraEndpoints.VEHICLE_FUEL_ENERGY:
            window = (params['start_date'], params['end_date'])
            return iter(vehicle_by_window.get(window, []))
        if endpoint is SamsaraEndpoints.DRIVER_FUEL_ENERGY:
            window = (params['start_date'], params['end_date'])
            return iter(driver_by_window.get(window, []))
        if endpoint is SamsaraEndpoints.DRIVER_VEHICLE_ASSIGNMENTS:
            return iter(assignments)
        if endpoint is SamsaraEndpoints.IDLING_EVENTS:
            return iter(idling_events)
        return iter([])

    fake_client.fetch_all.side_effect = fake_fetch_all

    fake_provider = MagicMock(spec=Provider)
    fake_provider.client.return_value = fake_client

    return fake_provider, fake_client


class TestSamsaraUtilizationFetcherProtocolConformance:
    """The concrete fetcher satisfies the runtime Protocol."""

    def test_isinstance_of_utilization_fetcher_protocol(self) -> None:
        """Should pass isinstance against the runtime_checkable Protocol."""

        fake_provider, _ = _build_fake_provider_and_client()
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        assert isinstance(fetcher, UtilizationFetcher)


class TestSamsaraUtilizationFetcherValidation:
    """Date-range validation."""

    def test_reversed_range_raises_value_error(self) -> None:
        """Should reject start_date > end_date with both dates in the message."""

        fake_provider, _ = _build_fake_provider_and_client()
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        with pytest.raises(ValueError, match='must be <=') as exc_info:
            fetcher.fetch(_MAY_16, _MAY_14)

        assert str(_MAY_16) in str(exc_info.value)
        assert str(_MAY_14) in str(exc_info.value)


class TestSamsaraUtilizationFetcherSingleDay:
    """Single-day range fetches all four endpoints exactly once."""

    def test_one_key_per_by_date_dict_and_four_calls(self) -> None:
        """Should produce one by-date key per dict and 4 total fetch_all calls."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_14)

        assert list(bundle.vehicle_fuel_energy_by_date) == [_MAY_14]
        assert list(bundle.driver_fuel_energy_by_date) == [_MAY_14]
        expected_call_count = 4
        assert fake_client.fetch_all.call_count == expected_call_count


class TestSamsaraUtilizationFetcherMultiDay:
    """Multi-day range loops per-day endpoints; full-range event endpoints once each."""

    def test_three_day_range_keys_and_call_count(self) -> None:
        """Should fan out 3 days x 2 per-day + 2 full-range endpoints = 8."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_16)

        assert list(bundle.vehicle_fuel_energy_by_date) == [
            _MAY_14,
            _MAY_15,
            _MAY_16,
        ]
        assert list(bundle.driver_fuel_energy_by_date) == [
            _MAY_14,
            _MAY_15,
            _MAY_16,
        ]
        expected_call_count = 8
        assert fake_client.fetch_all.call_count == expected_call_count


class TestSamsaraUtilizationFetcherEmptyDaysPreserved:
    """Days with zero records still appear in the by-date dicts."""

    def test_empty_vehicle_day_still_has_key(self) -> None:
        """Should preserve a key with value [] when an aggregate day is empty."""

        vehicle_records_by_window = {
            (_MAY_14_START, _MAY_15_START): [
                _make_vehicle_fuel_energy_report('999999900000001'),
            ],
            # No entry for May 15 -> fake_fetch_all returns iter([]).
            (_MAY_16_START, _MAY_17_START): [
                _make_vehicle_fuel_energy_report('999999900000002'),
            ],
        }
        fake_provider, _ = _build_fake_provider_and_client(
            vehicle_records_by_window=vehicle_records_by_window,
        )
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_16)

        assert _MAY_15 in bundle.vehicle_fuel_energy_by_date
        assert bundle.vehicle_fuel_energy_by_date[_MAY_15] == []
        assert len(bundle.vehicle_fuel_energy_by_date[_MAY_14]) == 1
        assert len(bundle.vehicle_fuel_energy_by_date[_MAY_16]) == 1


class TestSamsaraUtilizationFetcherEndpointAndParameterShapes:
    """Endpoint identity and parameter types per call."""

    def _fetch_and_partition_calls(
        self,
    ) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
        """Run a 2-day fetch and partition calls by endpoint constant."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        fetcher.fetch(_MAY_14, _MAY_15)

        vehicle_calls: list[Any] = []
        driver_calls: list[Any] = []
        assignment_calls: list[Any] = []
        idling_calls: list[Any] = []
        for call in fake_client.fetch_all.call_args_list:
            endpoint = call.args[0]
            if endpoint is SamsaraEndpoints.VEHICLE_FUEL_ENERGY:
                vehicle_calls.append(call)
            elif endpoint is SamsaraEndpoints.DRIVER_FUEL_ENERGY:
                driver_calls.append(call)
            elif endpoint is SamsaraEndpoints.DRIVER_VEHICLE_ASSIGNMENTS:
                assignment_calls.append(call)
            elif endpoint is SamsaraEndpoints.IDLING_EVENTS:
                idling_calls.append(call)

        return vehicle_calls, driver_calls, assignment_calls, idling_calls

    def test_endpoint_identity_per_call_class(self) -> None:
        """Each call's positional endpoint argument matches the expected constant."""

        vehicle_calls, driver_calls, assignment_calls, idling_calls = (
            self._fetch_and_partition_calls()
        )

        expected_per_day_calls = 2
        assert len(vehicle_calls) == expected_per_day_calls
        assert len(driver_calls) == expected_per_day_calls
        assert len(assignment_calls) == 1
        assert len(idling_calls) == 1

        for call in vehicle_calls:
            assert call.args[0] is SamsaraEndpoints.VEHICLE_FUEL_ENERGY
        for call in driver_calls:
            assert call.args[0] is SamsaraEndpoints.DRIVER_FUEL_ENERGY
        assert (
            assignment_calls[0].args[0] is SamsaraEndpoints.DRIVER_VEHICLE_ASSIGNMENTS
        )
        assert idling_calls[0].args[0] is SamsaraEndpoints.IDLING_EVENTS

    def test_vehicle_fuel_energy_params_are_utc_datetimes(self) -> None:
        """VEHICLE_FUEL_ENERGY receives start_date / end_date as UTC datetimes."""

        vehicle_calls, _, _, _ = self._fetch_and_partition_calls()

        for call in vehicle_calls:
            assert isinstance(call.kwargs['start_date'], datetime)
            assert isinstance(call.kwargs['end_date'], datetime)
            assert call.kwargs['start_date'].tzinfo is UTC
            assert call.kwargs['end_date'].tzinfo is UTC

    def test_driver_fuel_energy_params_are_utc_datetimes(self) -> None:
        """DRIVER_FUEL_ENERGY receives start_date / end_date as UTC datetimes."""

        _, driver_calls, _, _ = self._fetch_and_partition_calls()

        for call in driver_calls:
            assert isinstance(call.kwargs['start_date'], datetime)
            assert isinstance(call.kwargs['end_date'], datetime)
            assert call.kwargs['start_date'].tzinfo is UTC
            assert call.kwargs['end_date'].tzinfo is UTC

    def test_assignments_params_are_utc_datetimes_with_filter_by(self) -> None:
        """DRIVER_VEHICLE_ASSIGNMENTS receives start_time/end_time + filter_by='drivers'."""

        _, _, assignment_calls, _ = self._fetch_and_partition_calls()

        assignment_call = assignment_calls[0]
        assert isinstance(assignment_call.kwargs['start_time'], datetime)
        assert isinstance(assignment_call.kwargs['end_time'], datetime)
        assert assignment_call.kwargs['start_time'].tzinfo is UTC
        assert assignment_call.kwargs['end_time'].tzinfo is UTC
        assert assignment_call.kwargs['filter_by'] == 'drivers'

    def test_idling_events_params_are_utc_datetimes_without_filter_by(self) -> None:
        """IDLING_EVENTS receives start_time/end_time only; no filter_by kwarg."""

        _, _, _, idling_calls = self._fetch_and_partition_calls()

        idling_call = idling_calls[0]
        assert isinstance(idling_call.kwargs['start_time'], datetime)
        assert isinstance(idling_call.kwargs['end_time'], datetime)
        assert idling_call.kwargs['start_time'].tzinfo is UTC
        assert idling_call.kwargs['end_time'].tzinfo is UTC
        assert 'filter_by' not in idling_call.kwargs


class TestSamsaraUtilizationFetcherWindowBoundaries:
    """Exact window boundaries for both aggregate and full-range endpoints."""

    def test_three_day_windows_are_exact(self) -> None:
        """May 14 per-day window plus full-range [May 14 00Z, May 17 00Z) bounds."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        fetcher.fetch(_MAY_14, _MAY_16)

        # First per-day vehicle and driver calls cover May 14.
        first_vehicle_call = fake_client.fetch_all.call_args_list[0]
        first_driver_call = fake_client.fetch_all.call_args_list[1]
        assert first_vehicle_call.args[0] is SamsaraEndpoints.VEHICLE_FUEL_ENERGY
        assert first_vehicle_call.kwargs['start_date'] == _MAY_14_START
        assert first_vehicle_call.kwargs['end_date'] == _MAY_15_START
        assert first_driver_call.args[0] is SamsaraEndpoints.DRIVER_FUEL_ENERGY
        assert first_driver_call.kwargs['start_date'] == _MAY_14_START
        assert first_driver_call.kwargs['end_date'] == _MAY_15_START

        # Full-range calls span the half-open window [May 14 00Z, May 17 00Z).
        calls_by_endpoint: dict[Any, Any] = {
            call.args[0]: call for call in fake_client.fetch_all.call_args_list
        }
        assignment_call = calls_by_endpoint[SamsaraEndpoints.DRIVER_VEHICLE_ASSIGNMENTS]
        assert assignment_call.kwargs['start_time'] == _MAY_14_START
        assert assignment_call.kwargs['end_time'] == _MAY_17_START

        idling_call = calls_by_endpoint[SamsaraEndpoints.IDLING_EVENTS]
        assert idling_call.kwargs['start_time'] == _MAY_14_START
        assert idling_call.kwargs['end_time'] == _MAY_17_START


class TestSamsaraUtilizationFetcherPassThrough:
    """Records flow through the bundle unchanged."""

    def test_records_pass_through_by_identity(self) -> None:
        """Bundle entries are the exact instances the fake client yielded."""

        v_rec_14 = _make_vehicle_fuel_energy_report('999999900000001')
        v_rec_15 = _make_vehicle_fuel_energy_report('999999900000002')
        d_rec_14 = _make_driver_fuel_energy_report('1000001')
        assignment_a = _make_driver_vehicle_assignment('1000001', '999999900000001')
        assignment_b = _make_driver_vehicle_assignment('1000002', '999999900000002')
        idling_a = _make_idling_event('00000000-0000-0000-0000-000000000001')
        idling_b = _make_idling_event('00000000-0000-0000-0000-000000000002')

        fake_provider, _ = _build_fake_provider_and_client(
            vehicle_records_by_window={
                (_MAY_14_START, _MAY_15_START): [v_rec_14],
                (_MAY_15_START, _MAY_16_START): [v_rec_15],
            },
            driver_records_by_window={
                (_MAY_14_START, _MAY_15_START): [d_rec_14],
            },
            assignment_records=[assignment_a, assignment_b],
            idling_event_records=[idling_a, idling_b],
        )
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_15)

        assert bundle.vehicle_fuel_energy_by_date[_MAY_14][0] is v_rec_14
        assert bundle.vehicle_fuel_energy_by_date[_MAY_15][0] is v_rec_15
        assert bundle.driver_fuel_energy_by_date[_MAY_14][0] is d_rec_14
        assert bundle.driver_vehicle_assignments[0] is assignment_a
        assert bundle.driver_vehicle_assignments[1] is assignment_b
        assert bundle.idling_events[0] is idling_a
        assert bundle.idling_events[1] is idling_b


class TestSamsaraUtilizationFetcherBundleMetadata:
    """date_range and bundle immutability."""

    def test_date_range_matches_input(self) -> None:
        """Bundle.date_range equals the (start, end) tuple originally passed."""

        fake_provider, _ = _build_fake_provider_and_client()
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_16)

        assert bundle.date_range == (_MAY_14, _MAY_16)

    def test_bundle_is_frozen_dataclass(self) -> None:
        """Assigning to a bundle attribute raises FrozenInstanceError."""

        bundle = SamsaraUtilizationBundle(
            vehicle_fuel_energy_by_date={},
            driver_fuel_energy_by_date={},
            driver_vehicle_assignments=[],
            idling_events=[],
            date_range=(_MAY_14, _MAY_14),
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            bundle.vehicle_fuel_energy_by_date = {}  # pyright: ignore[reportAttributeAccessIssue]


class TestSamsaraUtilizationFetcherClientLifecycle:
    """The fetcher opens and closes the client context exactly once."""

    def test_client_enter_and_exit_called_once(self) -> None:
        """__enter__ and __exit__ are each invoked exactly once during fetch."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = SamsaraUtilizationFetcher(fake_provider)

        fetcher.fetch(_MAY_14, _MAY_15)

        assert fake_client.__enter__.call_count == 1
        assert fake_client.__exit__.call_count == 1
