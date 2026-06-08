"""Tests for MotiveUtilizationFetcher.

Uses ``unittest.mock`` to stand in for the Provider/TelemetryClient
context-manager pair, but constructs real Pydantic record instances
(DrivingPeriod, IdleEvent, etc.) for the data the fake client yields.
Tests assert the fetcher orchestrates the two event-grain Motive
endpoints (driving_periods, idle_events) correctly without performing
any transformation: identity equality is used for records flowing
through the bundle, and ``is`` identity is used for endpoint constants
on the call list. The fetcher must make no aggregate-endpoint
(VEHICLE_UTILIZATION / DRIVER_UTILIZATION) calls.

All identifiers in test data are synthetic.
"""

import dataclasses
from datetime import date, datetime, timedelta
from itertools import pairwise
from typing import Any
from unittest.mock import MagicMock

import pytest

from fleet_telemetry_hub.client import TelemetryClient
from fleet_telemetry_hub.models.motive_requests import MotiveEndpoints
from fleet_telemetry_hub.models.motive_responses import (
    DriverSummary,
    DrivingPeriod,
    EldDeviceInfo,
    IdleEvent,
    VehicleSummary,
)
from fleet_telemetry_hub.provider import Provider
from fleet_telemetry_hub.utilization import (
    MotiveUtilizationBundle,
    MotiveUtilizationFetcher,
    UtilizationFetcher,
)
from fleet_telemetry_hub.utilization.motive_fetcher import _MAX_CHUNK_DAYS

# Single-day and multi-day date constants used across tests.
_MAY_14 = date(2026, 5, 14)
_MAY_15 = date(2026, 5, 15)
_MAY_16 = date(2026, 5, 16)

# The two event-grain endpoints the fetcher is allowed to call. Any call
# outside this set is a regression (the aggregate endpoints were dropped).
_ALLOWED_ENDPOINTS = frozenset(
    {MotiveEndpoints.DRIVING_PERIODS, MotiveEndpoints.IDLE_EVENTS}
)


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


@dataclasses.dataclass(frozen=True, slots=True)
class _FakeBackendRecords:
    """
    Frozen fixture bundling the records the fake backend dispatches.

    ``driving_period_records`` / ``idle_event_records`` are flat lists
    returned regardless of window; tests that want chunk-window dispatch
    populate the ``*_by_window`` maps (keyed on the chunk's
    ``(start_date, end_date)``) instead, and a missing window key yields
    an empty iterator.

    Frozen+slots is structural only -- the held collections remain
    mutable Python dicts/lists, which is fine for a test fixture.
    """

    driving_period_records: list[DrivingPeriod] = dataclasses.field(
        default_factory=list
    )
    idle_event_records: list[IdleEvent] = dataclasses.field(default_factory=list)
    driving_period_records_by_window: dict[
        tuple[date, date], list[DrivingPeriod]
    ] = dataclasses.field(default_factory=dict)
    idle_event_records_by_window: dict[
        tuple[date, date], list[IdleEvent]
    ] = dataclasses.field(default_factory=dict)


def _build_fake_provider_and_client(
    *,
    records: _FakeBackendRecords | None = None,
) -> tuple[MagicMock, MagicMock]:
    """
    Construct a fake (Provider, TelemetryClient) pair wired together.

    The fake client's ``fetch_all`` dispatches on the endpoint constant:
    DRIVING_PERIODS / IDLE_EVENTS first check the ``*_by_window`` maps
    keyed on the chunk's ``(start_date, end_date)``; an empty map falls
    back to the flat ``*_records`` list. Any other endpoint yields an
    empty iterator -- the fetcher should never request one.

    Returns ``(fake_provider, fake_client)`` so tests can also assert on
    the client side (e.g. context-manager invocation counts, call-list
    dispatch).
    """

    backend_records = records or _FakeBackendRecords()

    fake_client = MagicMock(spec=TelemetryClient)
    fake_client.__enter__.return_value = fake_client
    fake_client.__exit__.return_value = None

    def dispatch_driving_periods(**params: Any) -> Any:
        date_window = (params['start_date'], params['end_date'])
        if backend_records.driving_period_records_by_window:
            return iter(
                backend_records.driving_period_records_by_window.get(date_window, [])
            )
        return iter(backend_records.driving_period_records)

    def dispatch_idle_events(**params: Any) -> Any:
        date_window = (params['start_date'], params['end_date'])
        if backend_records.idle_event_records_by_window:
            return iter(
                backend_records.idle_event_records_by_window.get(date_window, [])
            )
        return iter(backend_records.idle_event_records)

    def fake_fetch_all(endpoint: Any, **params: Any) -> Any:
        if endpoint is MotiveEndpoints.DRIVING_PERIODS:
            return dispatch_driving_periods(**params)
        if endpoint is MotiveEndpoints.IDLE_EVENTS:
            return dispatch_idle_events(**params)
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


class TestMotiveUtilizationFetcherCallShape:
    """Only the two event-grain endpoints are called -- no aggregate fetches."""

    def test_single_day_makes_only_event_calls(self) -> None:
        """A single-day fetch calls DRIVING_PERIODS + IDLE_EVENTS once each."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = MotiveUtilizationFetcher(fake_provider)

        fetcher.fetch(_MAY_14, _MAY_14)

        endpoints_called = [
            call.args[0] for call in fake_client.fetch_all.call_args_list
        ]
        # One chunk (single day) -> exactly one call per event endpoint.
        expected_call_count = 2
        assert len(endpoints_called) == expected_call_count
        assert set(endpoints_called) == set(_ALLOWED_ENDPOINTS)

    def test_no_aggregate_endpoint_is_ever_called(self) -> None:
        """The fetcher never calls VEHICLE_UTILIZATION or DRIVER_UTILIZATION.

        A multi-day range previously fanned out a per-day aggregate call
        pair; that path is gone, so every recorded call must target one of
        the two allowed event-grain endpoints.
        """

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = MotiveUtilizationFetcher(fake_provider)

        fetcher.fetch(_MAY_14, _MAY_16)

        endpoints_called = {
            call.args[0] for call in fake_client.fetch_all.call_args_list
        }
        assert endpoints_called <= set(_ALLOWED_ENDPOINTS)
        assert MotiveEndpoints.VEHICLE_UTILIZATION not in endpoints_called
        assert MotiveEndpoints.DRIVER_UTILIZATION not in endpoints_called


class TestMotiveUtilizationFetcherEndpointAndParameterShapes:
    """Endpoint identity and parameter types per call."""

    def _fetch_and_partition_calls(
        self,
    ) -> tuple[list[Any], list[Any]]:
        """Run a 2-day fetch and partition calls into (period, idle)."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = MotiveUtilizationFetcher(fake_provider)

        fetcher.fetch(_MAY_14, _MAY_15)

        period_calls: list[Any] = []
        idle_calls: list[Any] = []
        for call in fake_client.fetch_all.call_args_list:
            endpoint = call.args[0]
            if endpoint is MotiveEndpoints.DRIVING_PERIODS:
                period_calls.append(call)
            elif endpoint is MotiveEndpoints.IDLE_EVENTS:
                idle_calls.append(call)

        return period_calls, idle_calls

    def test_endpoint_identity_per_call_class(self) -> None:
        """Only DRIVING_PERIODS and IDLE_EVENTS are called, once each (one chunk)."""

        period_calls, idle_calls = self._fetch_and_partition_calls()

        assert len(period_calls) == 1
        assert len(idle_calls) == 1
        assert period_calls[0].args[0] is MotiveEndpoints.DRIVING_PERIODS
        assert idle_calls[0].args[0] is MotiveEndpoints.IDLE_EVENTS

    def test_driving_periods_params_are_bare_dates(self) -> None:
        """DRIVING_PERIODS receives start_date / end_date as bare date instances."""

        period_calls, _ = self._fetch_and_partition_calls()

        period_call = period_calls[0]
        assert isinstance(period_call.kwargs['start_date'], date)
        assert not isinstance(period_call.kwargs['start_date'], datetime)
        assert isinstance(period_call.kwargs['end_date'], date)
        assert not isinstance(period_call.kwargs['end_date'], datetime)

    def test_idle_events_params_are_bare_dates(self) -> None:
        """IDLE_EVENTS receives start_date / end_date as bare date instances."""

        _, idle_calls = self._fetch_and_partition_calls()

        idle_call = idle_calls[0]
        assert isinstance(idle_call.kwargs['start_date'], date)
        assert not isinstance(idle_call.kwargs['start_date'], datetime)
        assert isinstance(idle_call.kwargs['end_date'], date)
        assert not isinstance(idle_call.kwargs['end_date'], datetime)


class TestMotiveUtilizationFetcherWindowBoundaries:
    """Exact window boundaries for the event endpoints."""

    def test_may_14_event_windows_are_exact(self) -> None:
        """A single-day fetch passes the bare start/end date to both endpoints."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = MotiveUtilizationFetcher(fake_provider)

        fetcher.fetch(_MAY_14, _MAY_14)

        calls_by_endpoint: dict[Any, Any] = {
            call.args[0]: call for call in fake_client.fetch_all.call_args_list
        }

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

        period_a = _make_driving_period(4550000001)
        period_b = _make_driving_period(4550000002)
        idle_a = _make_idle_event(4860000001)
        idle_b = _make_idle_event(4860000002)

        fake_provider, _ = _build_fake_provider_and_client(
            records=_FakeBackendRecords(
                driving_period_records=[period_a, period_b],
                idle_event_records=[idle_a, idle_b],
            ),
        )
        fetcher = MotiveUtilizationFetcher(fake_provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_15)

        assert bundle.driving_periods[0] is period_a
        assert bundle.driving_periods[1] is period_b
        assert bundle.idle_events[0] is idle_a
        assert bundle.idle_events[1] is idle_b


class TestMotiveUtilizationFetcherBundleMetadata:
    """date_range, exact field set, and bundle immutability."""

    def test_date_range_matches_input(self) -> None:
        """Bundle.date_range equals the (start, end) tuple originally passed."""

        fake_provider, _ = _build_fake_provider_and_client()
        fetcher = MotiveUtilizationFetcher(fake_provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_16)

        assert bundle.date_range == (_MAY_14, _MAY_16)

    def test_field_set_is_exact(self) -> None:
        """The bundle carries exactly the four event-grain fields."""

        field_names = {field.name for field in dataclasses.fields(MotiveUtilizationBundle)}
        assert field_names == {
            'driving_periods',
            'idle_events',
            'date_range',
            'company',
        }

    def test_bundle_is_frozen_dataclass(self) -> None:
        """Assigning to a bundle attribute raises FrozenInstanceError."""

        bundle = MotiveUtilizationBundle(
            driving_periods=[],
            idle_events=[],
            date_range=(_MAY_14, _MAY_14),
            company=None,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            bundle.driving_periods = []  # pyright: ignore[reportAttributeAccessIssue]


class TestMotiveUtilizationFetcherClientLifecycle:
    """The fetcher opens and closes the client context exactly once."""

    def test_client_enter_and_exit_called_once(self) -> None:
        """__enter__ and __exit__ are each invoked exactly once during fetch."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = MotiveUtilizationFetcher(fake_provider)

        fetcher.fetch(_MAY_14, _MAY_15)

        assert fake_client.__enter__.call_count == 1
        assert fake_client.__exit__.call_count == 1


# ============================================================
# Chunking tests
# ============================================================
#
# Motive caps /v1/driving_periods and /v1/idle_events at 30 days per
# request. The fetcher chunks longer ranges via ``iter_chunks`` with
# ``_MAX_CHUNK_DAYS``; these tests exercise that path and pin the
# expected call shape.

# Multi-chunk landmarks: a 30-day inclusive range from May 14 should
# produce two chunks (days 1..28 + days 29..30), and a 142-day
# range from Jan 1 should produce six chunks. Both bracket the
# production failure mode.
_JAN_1 = date(2026, 1, 1)
_MAY_22 = date(2026, 5, 22)  # 142 days inclusive from Jan 1
_JUN_12 = date(2026, 6, 12)  # 30 days inclusive from May 14
_TWO_CHUNKS = 2
_SIX_CHUNKS = 6


class TestMotiveUtilizationFetcherChunking:
    """``DRIVING_PERIODS`` and ``IDLE_EVENTS`` are chunked into <=28-day windows."""

    @staticmethod
    def _driving_periods_calls(fake_client: MagicMock) -> list[Any]:
        """Filter the fake client's recorded calls down to DRIVING_PERIODS."""
        return [
            call
            for call in fake_client.fetch_all.call_args_list
            if call.args[0] is MotiveEndpoints.DRIVING_PERIODS
        ]

    @staticmethod
    def _idle_events_calls(fake_client: MagicMock) -> list[Any]:
        """Filter the fake client's recorded calls down to IDLE_EVENTS."""
        return [
            call
            for call in fake_client.fetch_all.call_args_list
            if call.args[0] is MotiveEndpoints.IDLE_EVENTS
        ]

    def test_thirty_day_range_yields_two_chunks_per_event_endpoint(self) -> None:
        """A 30-day inclusive range chunks into two calls each for both endpoints."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = MotiveUtilizationFetcher(fake_provider)

        fetcher.fetch(_MAY_14, _JUN_12)

        assert len(self._driving_periods_calls(fake_client)) == _TWO_CHUNKS
        assert len(self._idle_events_calls(fake_client)) == _TWO_CHUNKS

    def test_long_range_chunks_match_iter_chunks_output(self) -> None:
        """A 142-day backfill produces six chunks, matching ``iter_chunks``."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = MotiveUtilizationFetcher(fake_provider)

        fetcher.fetch(_JAN_1, _MAY_22)

        assert len(self._driving_periods_calls(fake_client)) == _SIX_CHUNKS
        assert len(self._idle_events_calls(fake_client)) == _SIX_CHUNKS

    def test_no_single_chunk_exceeds_max_chunk_days(self) -> None:
        """Every chunk's inclusive span is <= ``_MAX_CHUNK_DAYS`` days.

        This is the assertion that pins the production-failure
        invariant: no single call against ``DRIVING_PERIODS`` or
        ``IDLE_EVENTS`` may span a window wider than
        ``_MAX_CHUNK_DAYS`` inclusive days, regardless of the
        requested range.
        """

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = MotiveUtilizationFetcher(fake_provider)

        fetcher.fetch(_JAN_1, _MAY_22)

        for call in (
            *self._driving_periods_calls(fake_client),
            *self._idle_events_calls(fake_client),
        ):
            chunk_start: date = call.kwargs['start_date']
            chunk_end: date = call.kwargs['end_date']
            inclusive_days = (chunk_end - chunk_start).days + 1
            assert inclusive_days <= _MAX_CHUNK_DAYS, (
                f'Chunk {chunk_start}..{chunk_end} spans {inclusive_days} '
                f'days, exceeding _MAX_CHUNK_DAYS={_MAX_CHUNK_DAYS}'
            )

    def test_chunks_are_contiguous_and_non_overlapping(self) -> None:
        """Adjacent chunks meet at day-grain (``chunk[N].end + 1 == chunk[N+1].start``)."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = MotiveUtilizationFetcher(fake_provider)

        fetcher.fetch(_JAN_1, _MAY_22)

        windows: list[tuple[date, date]] = [
            (call.kwargs['start_date'], call.kwargs['end_date'])
            for call in self._driving_periods_calls(fake_client)
        ]
        # First chunk starts at the requested start; last chunk ends
        # at the requested end; every interior boundary is contiguous.
        assert windows[0][0] == _JAN_1
        assert windows[-1][1] == _MAY_22
        one_day = timedelta(days=1)
        for previous_window, next_window in pairwise(windows):
            assert next_window[0] == previous_window[1] + one_day

    def test_records_concatenate_in_chunk_chronological_order(self) -> None:
        """Each chunk's records appear in the bundle in chunk-chronological order."""

        # Two-chunk run with distinct period records per chunk window.
        chunk_one = (_MAY_14, date(2026, 6, 10))  # inclusive 28-day chunk
        chunk_two = (date(2026, 6, 11), _JUN_12)  # remaining 2 days
        period_chunk_one = _make_driving_period(4550000001)
        period_chunk_two = _make_driving_period(4550000002)
        idle_chunk_one = _make_idle_event(4860000001)
        idle_chunk_two = _make_idle_event(4860000002)

        fake_provider, _ = _build_fake_provider_and_client(
            records=_FakeBackendRecords(
                driving_period_records_by_window={
                    chunk_one: [period_chunk_one],
                    chunk_two: [period_chunk_two],
                },
                idle_event_records_by_window={
                    chunk_one: [idle_chunk_one],
                    chunk_two: [idle_chunk_two],
                },
            ),
        )
        fetcher = MotiveUtilizationFetcher(fake_provider)

        bundle = fetcher.fetch(_MAY_14, _JUN_12)

        assert bundle.driving_periods[0] is period_chunk_one
        assert bundle.driving_periods[1] is period_chunk_two
        assert bundle.idle_events[0] is idle_chunk_one
        assert bundle.idle_events[1] is idle_chunk_two

    def test_short_range_still_makes_exactly_one_chunk(self) -> None:
        """A range that fits in a single chunk produces exactly one call per endpoint."""

        fake_provider, fake_client = _build_fake_provider_and_client()
        fetcher = MotiveUtilizationFetcher(fake_provider)

        fetcher.fetch(_MAY_14, _MAY_16)  # 3 inclusive days, well under 28

        assert len(self._driving_periods_calls(fake_client)) == 1
        assert len(self._idle_events_calls(fake_client)) == 1
