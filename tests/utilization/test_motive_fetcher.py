"""Tests for MotiveUtilizationFetcher.

Uses ``unittest.mock`` to stand in for the Provider/TelemetryClient
context-manager pair, but constructs real Pydantic record instances
(VehicleUtilization, DriverIdleRollup, DrivingPeriod, IdleEvent, etc.)
for the data the fake client yields. Tests assert the fetcher
orchestrates the four Motive endpoints correctly without performing
any transformation: identity equality is used for records flowing
through the bundle, and ``is`` identity is used for endpoint
constants on the call list.

All identifiers in test data are synthetic.
"""

import dataclasses
from datetime import UTC, date, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from fleet_telemetry_hub.client import TelemetryClient
from fleet_telemetry_hub.models.motive_requests import MotiveEndpoints
from fleet_telemetry_hub.models.motive_responses import (
    DriverIdleRollup,
    DriverSummary,
    DrivingPeriod,
    EldDeviceInfo,
    IdleEvent,
    VehicleSummary,
    VehicleUtilization,
)
from fleet_telemetry_hub.provider import Provider
from fleet_telemetry_hub.utilization import (
    MotiveUtilizationBundle,
    MotiveUtilizationFetcher,
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


def _make_vehicle_summary(vehicle_id: int = 8000001) -> VehicleSummary:
    return VehicleSummary.model_validate(
        {
            'id': vehicle_id,
            'number': f'TEST-{vehicle_id}',
            'year': '2020',
            'make': 'TestMake',
            'model': 'TestModel',
            'vin': f'TESTVIN{vehicle_id:011d}',
            'metric_units': False,
        }
    )


def _make_driver_summary(driver_id: int = 9000001) -> DriverSummary:
    return DriverSummary.model_validate(
        {
            'id': driver_id,
            'first_name': 'Test',
            'last_name': f'Driver{driver_id}',
            'username': f'test.driver{driver_id}',
            'email': f'test.driver{driver_id}@example.com',
            'driver_company_id': f'TEST-{driver_id}-OTR',
            'status': 'active',
            'role': 'driver',
        }
    )


def _make_vehicle_utilization(vehicle_id: int = 8000001) -> VehicleUtilization:
    return VehicleUtilization.model_validate(
        {
            'message': '',
            'last_located_at': '2026-05-14T12:00:00Z',
            'utilization_percentage': 50.0,
            'idle_time': 1000,
            'idle_fuel': 0.5,
            'driving_time': 2000,
            'driving_fuel': 5.0,
            'total_fuel': 5.5,
            'total_distance': 100.0,
            'vehicle': _make_vehicle_summary(vehicle_id).model_dump(by_alias=True),
        }
    )


def _make_driver_idle_rollup(driver_id: int = 9000001) -> DriverIdleRollup:
    return DriverIdleRollup.model_validate(
        {
            'utilization': 66.6,
            'idle_time': 500,
            'driving_time': 1000,
            'driver': _make_driver_summary(driver_id).model_dump(by_alias=True),
            'idle_fuel': 0.25,
            'driving_fuel': 2.5,
        }
    )


def _make_driving_period(period_id: int = 4550000001) -> DrivingPeriod:
    return DrivingPeriod.model_validate(
        {
            'id': period_id,
            'start_time': '2026-05-14T10:00:00Z',
            'end_time': '2026-05-14T11:00:00Z',
            'status': 'complete',
            'type': 'driving',
            'annotation_status': None,
            'notes': None,
            'duration': 3600,
            'start_kilometers': 100.0,
            'end_kilometers': 150.0,
            'source': 1,
            'driver': _make_driver_summary().model_dump(by_alias=True),
            'vehicle': _make_vehicle_summary().model_dump(by_alias=True),
            'origin': '100 Test St',
            'origin_lat': 30.0,
            'origin_lon': -90.0,
            'destination': '200 Test Ave',
            'destination_lat': 30.1,
            'destination_lon': -90.1,
            'distance': '31.1 mi',
            'start_hvb_state_of_charge': None,
            'end_hvb_state_of_charge': None,
            'start_hvb_lifetime_energy_output': None,
            'end_hvb_lifetime_energy_output': None,
        }
    )


def _make_eld_device(device_id: int = 8800001) -> EldDeviceInfo:
    return EldDeviceInfo.model_validate(
        {
            'id': device_id,
            'identifier': f'TESTELD{device_id:04d}',
            'model': 'lbb-3.6ca',
        }
    )


def _make_idle_event(event_id: int = 4860000001) -> IdleEvent:
    return IdleEvent.model_validate(
        {
            'id': event_id,
            'start_time': '2026-05-14T07:00:00Z',
            'end_time': '2026-05-14T07:10:00Z',
            'veh_fuel_start': 100.0,
            'veh_fuel_end': 100.5,
            'lat': 30.0,
            'lon': -90.0,
            'city': 'Testville',
            'state': 'TX',
            'rg_brg': 62.0,
            'rg_km': 1.6,
            'rg_match': True,
            'end_type': 'vehicle_moving',
            'driver': _make_driver_summary().model_dump(by_alias=True),
            'vehicle': _make_vehicle_summary().model_dump(by_alias=True),
            'eld_device': _make_eld_device().model_dump(by_alias=True),
            'location': 'Testville, TX',
        }
    )


def _build_fake_provider_and_client(
    *,
    vehicle_records_by_window: dict[tuple[datetime, datetime], list[VehicleUtilization]]
    | None = None,
    driver_records_by_window: dict[tuple[datetime, datetime], list[DriverIdleRollup]]
    | None = None,
    driving_period_records: list[DrivingPeriod] | None = None,
    idle_event_records: list[IdleEvent] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """
    Construct a fake (Provider, TelemetryClient) pair wired together.

    The fake client's ``fetch_all`` dispatches on the endpoint constant:
    VEHICLE_UTILIZATION / DRIVER_UTILIZATION lookups use the
    (start_at, end_at) or (start_date, end_date) window as the key into
    the per-window record maps; DRIVING_PERIODS and IDLE_EVENTS each
    return their configured flat list regardless of window.

    Returns (fake_provider, fake_client) so tests can also assert on
    the client side (e.g. context-manager invocation counts).
    """

    vehicle_by_window: dict[tuple[datetime, datetime], list[VehicleUtilization]] = (
        vehicle_records_by_window or {}
    )
    driver_by_window: dict[tuple[datetime, datetime], list[DriverIdleRollup]] = (
        driver_records_by_window or {}
    )
    periods: list[DrivingPeriod] = driving_period_records or []
    idle_events: list[IdleEvent] = idle_event_records or []

    fake_client = MagicMock(spec=TelemetryClient)
    fake_client.__enter__.return_value = fake_client
    fake_client.__exit__.return_value = None

    def fake_fetch_all(endpoint: Any, **params: Any) -> Any:
        if endpoint is MotiveEndpoints.VEHICLE_UTILIZATION:
            window = (params['start_at'], params['end_at'])
            return iter(vehicle_by_window.get(window, []))
        if endpoint is MotiveEndpoints.DRIVER_UTILIZATION:
            window = (params['start_date'], params['end_date'])
            return iter(driver_by_window.get(window, []))
        if endpoint is MotiveEndpoints.DRIVING_PERIODS:
            return iter(periods)
        if endpoint is MotiveEndpoints.IDLE_EVENTS:
            return iter(idle_events)
        return iter([])

    fake_client.fetch_all.side_effect = fake_fetch_all

    fake_provider = MagicMock(spec=Provider)
    fake_provider.client.return_value = fake_client

    return fake_provider, fake_client


class TestMotiveUtilizationFetcherProtocolConformance:
    """The concrete fetcher satisfies the runtime Protocol."""

    def test_isinstance_of_utilization_fetcher_protocol(self) -> None:
        """Should pass isinstance against the runtime_checkable Protocol."""

        fake_provider, _ = _build_fake_provider_and_client()
        fetcher = MotiveUtilizationFetcher(fake_provider)

        assert isinstance(fetcher, UtilizationFetcher)


class TestMotiveUtilizationFetcherValidation:
    """Date-range validation."""

    def test_reversed_range_raises_value_error(self) -> None:
        """Should reject start_date > end_date with both dates in the message."""

        fake_provider, _ = _build_fake_provider_and_client()
        fetcher = MotiveUtilizationFetcher(fake_provider)

        with pytest.raises(ValueError, match='must be <=') as exc_info:
            fetcher.fetch(_MAY_16, _MAY_14)

        assert str(_MAY_16) in str(exc_info.value)
        assert str(_MAY_14) in str(exc_info.value)


class TestMotiveUtilizationFetcherSingleDay:
    """Single-day range fetches all four endpoints exactly once."""

    def test_one_key_per_by_date_dict_and_four_calls(self) -> None:
        """Should produce one by-date key per dict and 4 total fetch_all calls."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = MotiveUtilizationFetcher(fake_provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_14)

        assert list(bundle.vehicle_utilizations_by_date) == [_MAY_14]
        assert list(bundle.driver_idle_rollups_by_date) == [_MAY_14]
        expected_call_count = 4
        assert fake_client.fetch_all.call_count == expected_call_count


class TestMotiveUtilizationFetcherMultiDay:
    """Multi-day range loops per-day endpoints; full-range event endpoints once each."""

    def test_three_day_range_keys_and_call_count(self) -> None:
        """Should fan out 3 days x 2 per-day endpoints + driving_periods + idle_events = 8."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = MotiveUtilizationFetcher(fake_provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_16)

        assert list(bundle.vehicle_utilizations_by_date) == [
            _MAY_14,
            _MAY_15,
            _MAY_16,
        ]
        assert list(bundle.driver_idle_rollups_by_date) == [
            _MAY_14,
            _MAY_15,
            _MAY_16,
        ]
        expected_call_count = 8
        assert fake_client.fetch_all.call_count == expected_call_count


class TestMotiveUtilizationFetcherEmptyDaysPreserved:
    """Days with zero records still appear in the by-date dicts."""

    def test_empty_vehicle_day_still_has_key(self) -> None:
        """Should preserve a key with value [] when an aggregate day is empty."""

        vehicle_records_by_window = {
            (_MAY_14_START, _MAY_15_START): [_make_vehicle_utilization(8000001)],
            # No entry for May 15 -> fake_fetch_all returns iter([]).
            (_MAY_16_START, _MAY_17_START): [_make_vehicle_utilization(8000002)],
        }
        fake_provider, _ = _build_fake_provider_and_client(
            vehicle_records_by_window=vehicle_records_by_window,
        )
        fetcher = MotiveUtilizationFetcher(fake_provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_16)

        assert _MAY_15 in bundle.vehicle_utilizations_by_date
        assert bundle.vehicle_utilizations_by_date[_MAY_15] == []
        assert len(bundle.vehicle_utilizations_by_date[_MAY_14]) == 1
        assert len(bundle.vehicle_utilizations_by_date[_MAY_16]) == 1


class TestMotiveUtilizationFetcherEndpointAndParameterShapes:
    """Endpoint identity and parameter types per call."""

    def _fetch_and_partition_calls(
        self,
    ) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
        """Run a 2-day fetch and partition calls by endpoint constant."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = MotiveUtilizationFetcher(fake_provider)

        fetcher.fetch(_MAY_14, _MAY_15)

        vehicle_calls: list[Any] = []
        driver_calls: list[Any] = []
        period_calls: list[Any] = []
        idle_calls: list[Any] = []
        for call in fake_client.fetch_all.call_args_list:
            endpoint = call.args[0]
            if endpoint is MotiveEndpoints.VEHICLE_UTILIZATION:
                vehicle_calls.append(call)
            elif endpoint is MotiveEndpoints.DRIVER_UTILIZATION:
                driver_calls.append(call)
            elif endpoint is MotiveEndpoints.DRIVING_PERIODS:
                period_calls.append(call)
            elif endpoint is MotiveEndpoints.IDLE_EVENTS:
                idle_calls.append(call)

        return vehicle_calls, driver_calls, period_calls, idle_calls

    def test_endpoint_identity_per_call_class(self) -> None:
        """Each call's positional endpoint argument matches the expected constant."""

        vehicle_calls, driver_calls, period_calls, idle_calls = (
            self._fetch_and_partition_calls()
        )

        expected_per_day_calls = 2
        assert len(vehicle_calls) == expected_per_day_calls
        assert len(driver_calls) == expected_per_day_calls
        assert len(period_calls) == 1
        assert len(idle_calls) == 1

        for call in vehicle_calls:
            assert call.args[0] is MotiveEndpoints.VEHICLE_UTILIZATION
        for call in driver_calls:
            assert call.args[0] is MotiveEndpoints.DRIVER_UTILIZATION
        assert period_calls[0].args[0] is MotiveEndpoints.DRIVING_PERIODS
        assert idle_calls[0].args[0] is MotiveEndpoints.IDLE_EVENTS

    def test_vehicle_utilization_params_are_utc_datetimes(self) -> None:
        """VEHICLE_UTILIZATION receives start_at / end_at as UTC datetimes."""

        vehicle_calls, _, _, _ = self._fetch_and_partition_calls()

        for call in vehicle_calls:
            assert isinstance(call.kwargs['start_at'], datetime)
            assert isinstance(call.kwargs['end_at'], datetime)
            assert call.kwargs['start_at'].tzinfo is UTC
            assert call.kwargs['end_at'].tzinfo is UTC

    def test_driver_utilization_params_are_utc_datetimes(self) -> None:
        """DRIVER_UTILIZATION receives start_date / end_date as UTC datetimes."""

        _, driver_calls, _, _ = self._fetch_and_partition_calls()

        for call in driver_calls:
            assert isinstance(call.kwargs['start_date'], datetime)
            assert isinstance(call.kwargs['end_date'], datetime)
            assert call.kwargs['start_date'].tzinfo is UTC
            assert call.kwargs['end_date'].tzinfo is UTC

    def test_driving_periods_params_are_bare_dates(self) -> None:
        """DRIVING_PERIODS receives start_date / end_date as bare date instances."""

        _, _, period_calls, _ = self._fetch_and_partition_calls()

        period_call = period_calls[0]
        assert isinstance(period_call.kwargs['start_date'], date)
        assert not isinstance(period_call.kwargs['start_date'], datetime)
        assert isinstance(period_call.kwargs['end_date'], date)
        assert not isinstance(period_call.kwargs['end_date'], datetime)

    def test_idle_events_params_are_bare_dates(self) -> None:
        """IDLE_EVENTS receives start_date / end_date as bare date instances."""

        _, _, _, idle_calls = self._fetch_and_partition_calls()

        idle_call = idle_calls[0]
        assert isinstance(idle_call.kwargs['start_date'], date)
        assert not isinstance(idle_call.kwargs['start_date'], datetime)
        assert isinstance(idle_call.kwargs['end_date'], date)
        assert not isinstance(idle_call.kwargs['end_date'], datetime)


class TestMotiveUtilizationFetcherWindowBoundaries:
    """Exact window boundaries for both aggregate and event endpoints."""

    def test_may_14_windows_are_exact(self) -> None:
        """A May 14 fetch produces midnight-to-midnight UTC windows for aggregates."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = MotiveUtilizationFetcher(fake_provider)

        fetcher.fetch(_MAY_14, _MAY_14)

        calls_by_endpoint: dict[Any, Any] = {
            call.args[0]: call for call in fake_client.fetch_all.call_args_list
        }

        vehicle_call = calls_by_endpoint[MotiveEndpoints.VEHICLE_UTILIZATION]
        assert vehicle_call.kwargs['start_at'] == _MAY_14_START
        assert vehicle_call.kwargs['end_at'] == _MAY_15_START

        driver_call = calls_by_endpoint[MotiveEndpoints.DRIVER_UTILIZATION]
        assert driver_call.kwargs['start_date'] == _MAY_14_START
        assert driver_call.kwargs['end_date'] == _MAY_15_START

        period_call = calls_by_endpoint[MotiveEndpoints.DRIVING_PERIODS]
        assert period_call.kwargs['start_date'] == _MAY_14
        assert period_call.kwargs['end_date'] == _MAY_14

        idle_call = calls_by_endpoint[MotiveEndpoints.IDLE_EVENTS]
        assert idle_call.kwargs['start_date'] == _MAY_14
        assert idle_call.kwargs['end_date'] == _MAY_14


class TestMotiveUtilizationFetcherPassThrough:
    """Records flow through the bundle unchanged."""

    def test_records_pass_through_by_identity(self) -> None:
        """Bundle entries are the exact instances the fake client yielded."""

        v_rec_14 = _make_vehicle_utilization(8000001)
        v_rec_15 = _make_vehicle_utilization(8000002)
        d_rec_14 = _make_driver_idle_rollup(9000001)
        period_a = _make_driving_period(4550000001)
        period_b = _make_driving_period(4550000002)
        idle_a = _make_idle_event(4860000001)
        idle_b = _make_idle_event(4860000002)

        fake_provider, _ = _build_fake_provider_and_client(
            vehicle_records_by_window={
                (_MAY_14_START, _MAY_15_START): [v_rec_14],
                (_MAY_15_START, _MAY_16_START): [v_rec_15],
            },
            driver_records_by_window={
                (_MAY_14_START, _MAY_15_START): [d_rec_14],
            },
            driving_period_records=[period_a, period_b],
            idle_event_records=[idle_a, idle_b],
        )
        fetcher = MotiveUtilizationFetcher(fake_provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_15)

        assert bundle.vehicle_utilizations_by_date[_MAY_14][0] is v_rec_14
        assert bundle.vehicle_utilizations_by_date[_MAY_15][0] is v_rec_15
        assert bundle.driver_idle_rollups_by_date[_MAY_14][0] is d_rec_14
        assert bundle.driving_periods[0] is period_a
        assert bundle.driving_periods[1] is period_b
        assert bundle.idle_events[0] is idle_a
        assert bundle.idle_events[1] is idle_b


class TestMotiveUtilizationFetcherBundleMetadata:
    """date_range and bundle immutability."""

    def test_date_range_matches_input(self) -> None:
        """Bundle.date_range equals the (start, end) tuple originally passed."""

        fake_provider, _ = _build_fake_provider_and_client()
        fetcher = MotiveUtilizationFetcher(fake_provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_16)

        assert bundle.date_range == (_MAY_14, _MAY_16)

    def test_bundle_is_frozen_dataclass(self) -> None:
        """Assigning to a bundle attribute raises FrozenInstanceError."""

        bundle = MotiveUtilizationBundle(
            vehicle_utilizations_by_date={},
            driver_idle_rollups_by_date={},
            driving_periods=[],
            idle_events=[],
            date_range=(_MAY_14, _MAY_14),
            company=None,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            bundle.vehicle_utilizations_by_date = {}  # pyright: ignore[reportAttributeAccessIssue]


class TestMotiveUtilizationFetcherClientLifecycle:
    """The fetcher opens and closes the client context exactly once."""

    def test_client_enter_and_exit_called_once(self) -> None:
        """__enter__ and __exit__ are each invoked exactly once during fetch."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = MotiveUtilizationFetcher(fake_provider)

        fetcher.fetch(_MAY_14, _MAY_15)

        assert fake_client.__enter__.call_count == 1
        assert fake_client.__exit__.call_count == 1
