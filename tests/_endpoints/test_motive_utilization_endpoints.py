"""Tests for the Motive driver_utilization and driving_periods endpoints.

Covers the two Motive utilization endpoints added alongside the Patriot
fleet integration:

- ``MotiveEndpoints.DRIVER_UTILIZATION`` (/v2/driver_utilization), the
  per-driver-day analog of /v2/vehicle_utilization.
- ``MotiveEndpoints.DRIVING_PERIODS`` (/v1/driving_periods), the
  bipartite driver/vehicle linkage analog of Samsara's
  /fleet/driver-vehicle-assignments.

Each section asserts the endpoint surface (path, method, pagination,
response wiring), the declared ``QueryParameterSpec`` shape, the
expected subclass (Z-suffix for driver_utilization, base for
driving_periods), the ``build_query_params`` round-trip with type-
appropriate serialization (Z-suffix DATETIME vs bare DATE), and
response-model parsing on anonymized fixtures including edge cases
(null-driver bucket, cross-midnight periods, same-day periods).
Registry-resolution tests confirm the dynamic pickup via
``MotiveEndpoints.get_all_endpoints()``.

All identifiers in test data are synthetic. No real driver IDs, names,
emails, vehicle IDs, VINs, addresses, or lat/lons from any production
fleet appear in this file.
"""

from datetime import UTC, date, datetime
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic.main import BaseModel

from fleet_telemetry_hub.models import EndpointDefinition
from fleet_telemetry_hub.models.motive_requests import (
    MotiveEndpointDefinition,
    MotiveEndpoints,
    MotiveZSuffixDatetimeEndpointDefinition,
)
from fleet_telemetry_hub.models.motive_responses import (
    DriverIdleRollup,
    DriverIdleRollupWrapper,
    DriverSummary,
    DriverUtilizationsResponse,
    DrivingPeriod,
    DrivingPeriodsResponse,
    DrivingPeriodWrapper,
    EldDeviceInfo,
    IdleEvent,
    IdleEventsResponse,
    VehicleSummary,
)
from fleet_telemetry_hub.models.shared_request_models import HTTPMethod
from fleet_telemetry_hub.models.shared_response_models import ParameterType
from fleet_telemetry_hub.registry import EndpointRegistry

_MAX_PER_PAGE = 100

_DRIVER_UTILIZATION_FIXTURE: dict[str, Any] = {
    'driver_idle_rollups': [
        {
            'driver_idle_rollup': {
                'utilization': 37.8138731617499,
                'idle_time': 1311761,
                'driving_time': 797650,
                'driver': None,
                'idle_fuel': 320.7518131468578,
                'driving_fuel': 1248.0842902216707,
            },
        },
        {
            'driver_idle_rollup': {
                'utilization': 72.0731424597195,
                'idle_time': 9637,
                'driving_time': 24871,
                'driver': {
                    'id': 9000001,
                    'first_name': 'Sam',
                    'last_name': 'Snowflake',
                    'username': 'sam.snowflake',
                    'email': 'sam.snowflake@example.com',
                    'driver_company_id': 'TEST-001-OTR',
                    'status': 'active',
                    'role': 'driver',
                },
                'idle_fuel': 1.5396274375,
                'driving_fuel': 51.525923062500006,
            },
        },
        {
            'driver_idle_rollup': {
                'utilization': 90.4479946374707,
                'idle_time': 2565,
                'driving_time': 24288,
                'driver': {
                    'id': 9000002,
                    'first_name': 'Suzy',
                    'last_name': 'Snowflake',
                    'username': None,
                    'email': 'suzy.snowflake@example.com',
                    'driver_company_id': 'TEST-002-SSR',
                    'status': 'active',
                    'role': 'driver',
                },
                'idle_fuel': 0.636695796875,
                'driving_fuel': 40.473007859375,
            },
        },
        {
            'driver_idle_rollup': {
                'utilization': 0,
                'idle_time': 0,
                'driving_time': 0,
                'driver': {
                    'id': 9000003,
                    'first_name': 'Sammy',
                    'last_name': 'Snowflake',
                    'username': 'sammy.snowflake',
                    'email': 'sammy.snowflake@example.com',
                    'driver_company_id': 'TEST-003-OTR',
                    'status': 'active',
                    'role': 'driver',
                },
                'idle_fuel': 0,
                'driving_fuel': 0,
            },
        },
    ],
    'pagination': {'per_page': 25, 'page_no': 1, 'total': 634},
}

_DRIVING_PERIODS_FIXTURE: dict[str, Any] = {
    'driving_periods': [
        {
            'driving_period': {
                'id': 4550263305,
                'start_time': '2026-05-15T23:55:27Z',
                'end_time': '2026-05-16T00:34:36Z',
                'status': 'complete',
                'type': 'driving',
                'annotation_status': None,
                'notes': None,
                'duration': 2349,
                'start_kilometers': 306319.7429,
                'end_kilometers': 306355.68866,
                'source': 1,
                'driver': None,
                'vehicle': {
                    'id': 8000001,
                    'number': 'TEST-100',
                    'year': '2011',
                    'make': 'TestMake',
                    'model': 'TestModel',
                    'vin': 'TESTVIN0000000100',
                    'metric_units': False,
                },
                'origin': '100 Test St, Testville, TX 99999',
                'origin_lat': 30.0,
                'origin_lon': -90.0,
                'destination_lat': 30.1,
                'destination_lon': -90.1,
                'destination': '200 Test Ave, Testville, TX 99999',
                'distance': '22.3 mi',
                'start_hvb_state_of_charge': None,
                'end_hvb_state_of_charge': None,
                'start_hvb_lifetime_energy_output': None,
                'end_hvb_lifetime_energy_output': None,
            },
        },
        {
            'driving_period': {
                'id': 4550079336,
                'start_time': '2026-05-15T23:54:12Z',
                'end_time': '2026-05-16T00:24:02Z',
                'status': 'complete',
                'type': 'driving',
                'annotation_status': None,
                'notes': None,
                'duration': 1790,
                'start_kilometers': 348536.17697,
                'end_kilometers': 348566.67233,
                'source': 1,
                'driver': {
                    'id': 9000001,
                    'first_name': 'Sam',
                    'last_name': 'Snowflake',
                    'username': 'sam.snowflake',
                    'email': 'sam.snowflake@example.com',
                    'driver_company_id': 'TEST-001-OTR',
                    'status': 'active',
                    'role': 'driver',
                },
                'vehicle': {
                    'id': 8000002,
                    'number': 'TEST-101',
                    'year': '2019',
                    'make': 'TestMake',
                    'model': 'TestModel',
                    'vin': 'TESTVIN0000000101',
                    'metric_units': False,
                },
                'origin': '300 Test Rd, Testville, TX 99999',
                'origin_lat': 30.2,
                'origin_lon': -90.2,
                'destination_lat': 30.3,
                'destination_lon': -90.3,
                'destination': '400 Test Blvd, Testville, TX 99999',
                'distance': '18.9 mi',
                'start_hvb_state_of_charge': None,
                'end_hvb_state_of_charge': None,
                'start_hvb_lifetime_energy_output': None,
                'end_hvb_lifetime_energy_output': None,
            },
        },
        {
            'driving_period': {
                'id': 4557689477,
                'start_time': '2026-05-15T22:40:19Z',
                'end_time': '2026-05-15T23:30:26Z',
                'status': 'complete',
                'type': 'driving',
                'annotation_status': None,
                'notes': '',
                'duration': 3007,
                'start_kilometers': 185252.61072,
                'end_kilometers': 185306.70761,
                'source': 4,
                'driver': {
                    'id': 9000002,
                    'first_name': 'Suzy',
                    'last_name': 'Snowflake',
                    'username': None,
                    'email': 'suzy.snowflake@example.com',
                    'driver_company_id': 'TEST-002-SSR',
                    'status': 'active',
                    'role': 'driver',
                },
                'vehicle': {
                    'id': 8000003,
                    'number': 'TEST-102',
                    'year': '2018',
                    'make': 'TestMake',
                    'model': 'TestModel',
                    'vin': 'TESTVIN0000000102',
                    'metric_units': False,
                },
                'origin': '500 Test Way, Testville, TX 99999',
                'origin_lat': 30.4,
                'origin_lon': -90.4,
                'destination_lat': 30.5,
                'destination_lon': -90.5,
                'destination': '600 Test Ct, Testville, TX 99999',
                'distance': '33.6 mi',
                'start_hvb_state_of_charge': None,
                'end_hvb_state_of_charge': None,
                'start_hvb_lifetime_energy_output': None,
                'end_hvb_lifetime_energy_output': None,
            },
        },
    ],
    'pagination': {'per_page': 25, 'page_no': 1, 'total': 10242},
}

_FIRST_ROLLUP_RAW: dict[str, Any] = _DRIVER_UTILIZATION_FIXTURE['driver_idle_rollups'][
    0
]['driver_idle_rollup']
_SECOND_ROLLUP_RAW: dict[str, Any] = _DRIVER_UTILIZATION_FIXTURE['driver_idle_rollups'][
    1
]['driver_idle_rollup']
_EXPECTED_ROLLUP_COUNT: int = len(_DRIVER_UTILIZATION_FIXTURE['driver_idle_rollups'])

_FIRST_PERIOD_RAW: dict[str, Any] = _DRIVING_PERIODS_FIXTURE['driving_periods'][0][
    'driving_period'
]
_SECOND_PERIOD_RAW: dict[str, Any] = _DRIVING_PERIODS_FIXTURE['driving_periods'][1][
    'driving_period'
]
_THIRD_PERIOD_RAW: dict[str, Any] = _DRIVING_PERIODS_FIXTURE['driving_periods'][2][
    'driving_period'
]
_EXPECTED_PERIOD_COUNT: int = len(_DRIVING_PERIODS_FIXTURE['driving_periods'])


class TestDriverUtilizationEndpointDefinition:
    """Surface assertions on MotiveEndpoints.DRIVER_UTILIZATION."""

    def test_endpoint_path(self) -> None:
        """Should expose the /v2/driver_utilization path."""

        assert (
            MotiveEndpoints.DRIVER_UTILIZATION.endpoint_path == '/v2/driver_utilization'
        )

    def test_http_method_is_get(self) -> None:
        """Should use HTTP GET."""

        assert MotiveEndpoints.DRIVER_UTILIZATION.http_method == HTTPMethod.GET

    def test_endpoint_is_paginated(self) -> None:
        """Should be marked paginated."""

        assert MotiveEndpoints.DRIVER_UTILIZATION.is_paginated is True

    def test_max_per_page(self) -> None:
        """Should cap page size at Motive's documented maximum of 100."""

        assert MotiveEndpoints.DRIVER_UTILIZATION.max_per_page == _MAX_PER_PAGE

    def test_response_model(self) -> None:
        """Should parse responses with DriverUtilizationsResponse."""

        assert (
            MotiveEndpoints.DRIVER_UTILIZATION.response_model
            is DriverUtilizationsResponse
        )

    def test_item_extractor_method(self) -> None:
        """Should extract items via get_driver_idle_rollups."""

        assert (
            MotiveEndpoints.DRIVER_UTILIZATION.item_extractor_method
            == 'get_driver_idle_rollups'
        )

    def test_description_is_non_empty(self) -> None:
        """Should declare a non-empty description for registry introspection."""

        assert MotiveEndpoints.DRIVER_UTILIZATION.description

    def test_is_z_suffix_subclass(self) -> None:
        """Should be an instance of the Z-suffix datetime subclass."""

        assert isinstance(
            MotiveEndpoints.DRIVER_UTILIZATION,
            MotiveZSuffixDatetimeEndpointDefinition,
        )

    def test_query_parameter_shape(self) -> None:
        """Should declare start_date and end_date as required DATETIME params."""

        specs_by_name = {
            spec.name: spec
            for spec in MotiveEndpoints.DRIVER_UTILIZATION.query_parameters
        }
        assert set(specs_by_name) == {'start_date', 'end_date'}
        for spec in specs_by_name.values():
            assert spec.required is True
            assert spec.parameter_type == ParameterType.DATETIME

    def test_build_query_params_emits_z_suffix(self) -> None:
        """Should serialize tz-aware UTC datetimes with a Z suffix."""

        utc_start = datetime(2026, 5, 14, 0, 0, 0, tzinfo=UTC)
        utc_end = datetime(2026, 5, 15, 0, 0, 0, tzinfo=UTC)

        query = MotiveEndpoints.DRIVER_UTILIZATION.build_query_params(
            start_date=utc_start,
            end_date=utc_end,
        )

        assert query['start_date'] == '2026-05-14T00:00:00Z'
        assert query['end_date'] == '2026-05-15T00:00:00Z'


class TestDriverUtilizationResponseParsing:
    """Tests against DriverUtilizationsResponse.model_validate."""

    def test_full_fixture_parses_without_error(self) -> None:
        """Should parse the full fixture without raising ValidationError."""

        DriverUtilizationsResponse.model_validate(_DRIVER_UTILIZATION_FIXTURE)

    def test_get_driver_idle_rollups_unwraps(self) -> None:
        """Should unwrap into a flat list of DriverIdleRollup instances."""

        parsed = DriverUtilizationsResponse.model_validate(
            _DRIVER_UTILIZATION_FIXTURE,
        )

        items = parsed.get_driver_idle_rollups()

        assert len(items) == _EXPECTED_ROLLUP_COUNT
        assert all(isinstance(item, DriverIdleRollup) for item in items)

    def test_wrappers_remain_typed(self) -> None:
        """Should preserve DriverIdleRollupWrapper typing on the wrapper list."""

        parsed = DriverUtilizationsResponse.model_validate(
            _DRIVER_UTILIZATION_FIXTURE,
        )

        assert all(
            isinstance(wrapper, DriverIdleRollupWrapper)
            for wrapper in parsed.driver_idle_rollups
        )

    def test_pagination_metadata_parsed(self) -> None:
        """Should parse pagination metadata."""

        parsed = DriverUtilizationsResponse.model_validate(
            _DRIVER_UTILIZATION_FIXTURE,
        )

        assert (
            parsed.pagination.per_page
            == _DRIVER_UTILIZATION_FIXTURE['pagination']['per_page']
        )
        assert (
            parsed.pagination.page_no
            == _DRIVER_UTILIZATION_FIXTURE['pagination']['page_no']
        )
        assert (
            parsed.pagination.total
            == _DRIVER_UTILIZATION_FIXTURE['pagination']['total']
        )


class TestDriverIdleRollupEdgeCases:
    """Edge-case assertions on the parsed rollup records."""

    def test_null_driver_record_parses(self) -> None:
        """Should parse the unattributed-activity bucket with driver=None."""

        items = DriverUtilizationsResponse.model_validate(
            _DRIVER_UTILIZATION_FIXTURE,
        ).get_driver_idle_rollups()

        first = items[0]

        assert first.driver is None
        assert first.idle_time == _FIRST_ROLLUP_RAW['idle_time']
        assert first.driving_time == _FIRST_ROLLUP_RAW['driving_time']

    def test_named_driver_record_reuses_driver_summary(self) -> None:
        """Should parse the driver block into the existing DriverSummary model."""

        items = DriverUtilizationsResponse.model_validate(
            _DRIVER_UTILIZATION_FIXTURE,
        ).get_driver_idle_rollups()

        second = items[1]

        assert isinstance(second.driver, DriverSummary)
        assert second.driver.driver_id == _SECOND_ROLLUP_RAW['driver']['id']
        assert second.driver.first_name == _SECOND_ROLLUP_RAW['driver']['first_name']
        assert second.driver.last_name == _SECOND_ROLLUP_RAW['driver']['last_name']
        assert second.driver.username == _SECOND_ROLLUP_RAW['driver']['username']

    def test_null_username_record_parses(self) -> None:
        """Should accept username=None on the embedded driver."""

        items = DriverUtilizationsResponse.model_validate(
            _DRIVER_UTILIZATION_FIXTURE,
        ).get_driver_idle_rollups()

        third = items[2]

        assert third.driver is not None
        assert third.driver.username is None

    def test_zero_activity_record_parses(self) -> None:
        """Should parse the all-zeros record without raising."""

        items = DriverUtilizationsResponse.model_validate(
            _DRIVER_UTILIZATION_FIXTURE,
        ).get_driver_idle_rollups()

        fourth = items[3]

        assert fourth.utilization == 0.0
        assert fourth.idle_time == 0
        assert fourth.driving_time == 0
        assert fourth.idle_fuel == 0.0
        assert fourth.driving_fuel == 0.0


class TestDriverIdleRollupConvenienceProperties:
    """Tests for is_unattributed_bucket and total_engine_seconds."""

    def test_is_unattributed_bucket_true_for_null_driver(self) -> None:
        """Should return True for the null-driver bucket."""

        items = DriverUtilizationsResponse.model_validate(
            _DRIVER_UTILIZATION_FIXTURE,
        ).get_driver_idle_rollups()

        assert items[0].is_unattributed_bucket is True

    def test_is_unattributed_bucket_false_for_named_driver(self) -> None:
        """Should return False when a driver is attached."""

        items = DriverUtilizationsResponse.model_validate(
            _DRIVER_UTILIZATION_FIXTURE,
        ).get_driver_idle_rollups()

        assert items[1].is_unattributed_bucket is False

    def test_total_engine_seconds_sums_idle_and_driving(self) -> None:
        """Should equal idle_time + driving_time."""

        items = DriverUtilizationsResponse.model_validate(
            _DRIVER_UTILIZATION_FIXTURE,
        ).get_driver_idle_rollups()

        second = items[1]

        assert second.total_engine_seconds == (second.idle_time + second.driving_time)


class TestDrivingPeriodsEndpointDefinition:
    """Surface assertions on MotiveEndpoints.DRIVING_PERIODS."""

    def test_endpoint_path(self) -> None:
        """Should expose the /v1/driving_periods path."""

        assert MotiveEndpoints.DRIVING_PERIODS.endpoint_path == '/v1/driving_periods'

    def test_http_method_is_get(self) -> None:
        """Should use HTTP GET."""

        assert MotiveEndpoints.DRIVING_PERIODS.http_method == HTTPMethod.GET

    def test_endpoint_is_paginated(self) -> None:
        """Should be marked paginated."""

        assert MotiveEndpoints.DRIVING_PERIODS.is_paginated is True

    def test_max_per_page(self) -> None:
        """Should cap page size at Motive's documented maximum of 100."""

        assert MotiveEndpoints.DRIVING_PERIODS.max_per_page == _MAX_PER_PAGE

    def test_response_model(self) -> None:
        """Should parse responses with DrivingPeriodsResponse."""

        assert MotiveEndpoints.DRIVING_PERIODS.response_model is DrivingPeriodsResponse

    def test_item_extractor_method(self) -> None:
        """Should extract items via get_driving_periods."""

        assert (
            MotiveEndpoints.DRIVING_PERIODS.item_extractor_method
            == 'get_driving_periods'
        )

    def test_description_is_non_empty(self) -> None:
        """Should declare a non-empty description for registry introspection."""

        assert MotiveEndpoints.DRIVING_PERIODS.description

    def test_is_not_z_suffix_subclass(self) -> None:
        """Should NOT be a Z-suffix subclass instance (bare DATE params only)."""

        assert not isinstance(
            MotiveEndpoints.DRIVING_PERIODS,
            MotiveZSuffixDatetimeEndpointDefinition,
        )
        assert isinstance(MotiveEndpoints.DRIVING_PERIODS, MotiveEndpointDefinition)

    def test_query_parameter_shape(self) -> None:
        """Should declare start_date and end_date as required DATE params."""

        specs_by_name = {
            spec.name: spec for spec in MotiveEndpoints.DRIVING_PERIODS.query_parameters
        }
        assert set(specs_by_name) == {'start_date', 'end_date'}
        for spec in specs_by_name.values():
            assert spec.required is True
            assert spec.parameter_type == ParameterType.DATE

    def test_build_query_params_emits_bare_date(self) -> None:
        """Should serialize date values as YYYY-MM-DD (no time, no Z suffix)."""

        query = MotiveEndpoints.DRIVING_PERIODS.build_query_params(
            start_date=date(2026, 5, 14),
            end_date=date(2026, 5, 15),
        )

        assert query['start_date'] == '2026-05-14'
        assert query['end_date'] == '2026-05-15'


class TestDrivingPeriodsResponseParsing:
    """Tests against DrivingPeriodsResponse.model_validate."""

    def test_full_fixture_parses_without_error(self) -> None:
        """Should parse the full fixture without raising ValidationError."""

        DrivingPeriodsResponse.model_validate(_DRIVING_PERIODS_FIXTURE)

    def test_get_driving_periods_unwraps(self) -> None:
        """Should unwrap into a flat list of DrivingPeriod instances."""

        parsed = DrivingPeriodsResponse.model_validate(_DRIVING_PERIODS_FIXTURE)

        items = parsed.get_driving_periods()

        assert len(items) == _EXPECTED_PERIOD_COUNT
        assert all(isinstance(item, DrivingPeriod) for item in items)

    def test_wrappers_remain_typed(self) -> None:
        """Should preserve DrivingPeriodWrapper typing on the wrapper list."""

        parsed = DrivingPeriodsResponse.model_validate(_DRIVING_PERIODS_FIXTURE)

        assert all(
            isinstance(wrapper, DrivingPeriodWrapper)
            for wrapper in parsed.driving_periods
        )

    def test_pagination_metadata_parsed(self) -> None:
        """Should parse pagination metadata."""

        parsed = DrivingPeriodsResponse.model_validate(_DRIVING_PERIODS_FIXTURE)

        assert (
            parsed.pagination.per_page
            == _DRIVING_PERIODS_FIXTURE['pagination']['per_page']
        )
        assert (
            parsed.pagination.total == _DRIVING_PERIODS_FIXTURE['pagination']['total']
        )


class TestDrivingPeriodEdgeCases:
    """Edge-case assertions on cross-midnight, attributed, and same-day periods."""

    def test_cross_midnight_null_driver_record(self) -> None:
        """Should parse the cross-midnight period with driver=None."""

        items = DrivingPeriodsResponse.model_validate(
            _DRIVING_PERIODS_FIXTURE,
        ).get_driving_periods()

        first = items[0]

        assert first.driver is None
        assert first.period_id == _FIRST_PERIOD_RAW['id']
        assert isinstance(first.start_time, datetime)
        assert isinstance(first.end_time, datetime)
        assert first.start_time.tzinfo is not None
        assert first.end_time.tzinfo is not None
        assert first.end_time.date() != first.start_time.date()

    def test_cross_midnight_attributed_record(self) -> None:
        """Should parse the cross-midnight period with a logged-in driver."""

        items = DrivingPeriodsResponse.model_validate(
            _DRIVING_PERIODS_FIXTURE,
        ).get_driving_periods()

        second = items[1]

        assert isinstance(second.driver, DriverSummary)
        assert second.driver.driver_id == _SECOND_PERIOD_RAW['driver']['id']
        assert isinstance(second.vehicle, VehicleSummary)
        assert second.vehicle.vehicle_id == _SECOND_PERIOD_RAW['vehicle']['id']
        assert second.end_time.date() != second.start_time.date()

    def test_same_day_attributed_record(self) -> None:
        """Should parse the same-day attributed period."""

        items = DrivingPeriodsResponse.model_validate(
            _DRIVING_PERIODS_FIXTURE,
        ).get_driving_periods()

        third = items[2]

        assert third.driver is not None
        assert third.start_time.date() == third.end_time.date()
        assert third.notes == _THIRD_PERIOD_RAW['notes']
        assert third.source == _THIRD_PERIOD_RAW['source']

    def test_hvb_fields_are_none_for_fuel_vehicles(self) -> None:
        """Should leave all HVB fields as None on every fuel-vehicle record."""

        items = DrivingPeriodsResponse.model_validate(
            _DRIVING_PERIODS_FIXTURE,
        ).get_driving_periods()

        for period in items:
            assert period.start_hvb_state_of_charge is None
            assert period.end_hvb_state_of_charge is None
            assert period.start_hvb_lifetime_energy_output is None
            assert period.end_hvb_lifetime_energy_output is None

    def test_distance_string_preserved_verbatim(self) -> None:
        """Should preserve Motive's formatted distance string for fidelity."""

        items = DrivingPeriodsResponse.model_validate(
            _DRIVING_PERIODS_FIXTURE,
        ).get_driving_periods()

        assert items[0].distance == _FIRST_PERIOD_RAW['distance']


class TestDrivingPeriodConvenienceProperties:
    """Tests for kilometers_traveled."""

    def test_kilometers_traveled_is_odometer_delta(self) -> None:
        """Should equal end_kilometers - start_kilometers."""

        items = DrivingPeriodsResponse.model_validate(
            _DRIVING_PERIODS_FIXTURE,
        ).get_driving_periods()

        first = items[0]

        assert first.kilometers_traveled == (
            first.end_kilometers - first.start_kilometers
        )


class TestMotiveUtilizationEndpointsRegistryResolution:
    """Verifies EndpointRegistry picks up each new endpoint by name."""

    def test_driver_utilization_resolves(self) -> None:
        """Should resolve 'driver_utilization' from the motive provider."""

        registry = EndpointRegistry()

        endpoint: EndpointDefinition[BaseModel] = registry.get(
            'motive', 'driver_utilization'
        )

        assert isinstance(endpoint, EndpointDefinition)
        assert isinstance(endpoint, MotiveZSuffixDatetimeEndpointDefinition)
        assert endpoint.endpoint_path == '/v2/driver_utilization'

    def test_driving_periods_resolves(self) -> None:
        """Should resolve 'driving_periods' from the motive provider."""

        registry = EndpointRegistry()

        endpoint: EndpointDefinition[BaseModel] = registry.get(
            'motive', 'driving_periods'
        )

        assert isinstance(endpoint, EndpointDefinition)
        assert isinstance(endpoint, MotiveEndpointDefinition)
        assert not isinstance(endpoint, MotiveZSuffixDatetimeEndpointDefinition)
        assert endpoint.endpoint_path == '/v1/driving_periods'

    def test_idle_events_resolves(self) -> None:
        """Should resolve 'idle_events' from the motive provider."""

        registry = EndpointRegistry()

        endpoint: EndpointDefinition[BaseModel] = registry.get('motive', 'idle_events')

        assert isinstance(endpoint, EndpointDefinition)
        assert isinstance(endpoint, MotiveEndpointDefinition)
        assert not isinstance(endpoint, MotiveZSuffixDatetimeEndpointDefinition)
        assert endpoint.endpoint_path == '/v1/idle_events'


_IDLE_EVENTS_FIXTURE: dict[str, Any] = {
    'idle_events': [
        {
            'idle_event': {
                'id': 4864227475,
                'start_time': '2026-05-14T07:05:29Z',
                'end_time': '2026-05-14T07:09:52Z',
                'veh_fuel_start': 132572.46875,
                'veh_fuel_end': 132572.734375,
                'lat': 30.0,
                'lon': -90.0,
                'city': 'Testville',
                'state': 'TX',
                'rg_brg': 62.084,
                'rg_km': 1.614,
                'rg_match': True,
                'end_type': 'vehicle_moving',
                'driver': None,
                'vehicle': {
                    'id': 8000004,
                    'number': 'TEST-103',
                    'year': '2020',
                    'make': 'TestMake',
                    'model': 'TestModel',
                    'vin': 'TESTVIN0000000103',
                    'metric_units': False,
                },
                'eld_device': {
                    'id': 8800004,
                    'identifier': 'TESTELD0004',
                    'model': 'lbb-3.6ca',
                },
                'location': 'Testville, TX',
            },
        },
        {
            'idle_event': {
                'id': 4900000001,
                'start_time': '2026-05-15T23:55:00Z',
                'end_time': '2026-05-16T00:15:30Z',
                'veh_fuel_start': 250000.0,
                'veh_fuel_end': 250000.875,
                'lat': 30.2,
                'lon': -90.2,
                'city': 'Testville',
                'state': 'TX',
                'rg_brg': 124.819,
                'rg_km': 3.721,
                'rg_match': True,
                'end_type': 'vehicle_moving',
                'driver': {
                    'id': 9000002,
                    'first_name': 'Suzy',
                    'last_name': 'Snowflake',
                    'username': None,
                    'email': 'suzy.snowflake@example.com',
                    'driver_company_id': 'TEST-002-SSR',
                    'status': 'active',
                    'role': 'driver',
                },
                'vehicle': {
                    'id': 8000003,
                    'number': 'TEST-102',
                    'year': '2018',
                    'make': 'TestMake',
                    'model': 'TestModel',
                    'vin': 'TESTVIN0000000102',
                    'metric_units': False,
                },
                'eld_device': {
                    'id': 8800003,
                    'identifier': 'TESTELD0003',
                    'model': 'lbb-3.6ca',
                },
                'location': 'Testville, TX',
            },
        },
        {
            'idle_event': {
                'id': 4864274598,
                'start_time': '2026-05-14T07:00:09Z',
                'end_time': '2026-05-14T07:53:50Z',
                'veh_fuel_start': 48106.73828125,
                'veh_fuel_end': 48108.54296875,
                'lat': 30.1,
                'lon': -90.1,
                'city': 'Testville',
                'state': 'TX',
                'rg_brg': 63.786,
                'rg_km': 1.574,
                'rg_match': True,
                'end_type': 'vehicle_moving',
                'driver': {
                    'id': 9000001,
                    'first_name': 'Sam',
                    'last_name': 'Snowflake',
                    'username': 'sam.snowflake',
                    'email': 'sam.snowflake@example.com',
                    'driver_company_id': 'TEST-001-OTR',
                    'status': 'active',
                    'role': 'driver',
                },
                'vehicle': {
                    'id': 8000001,
                    'number': 'TEST-100',
                    'year': '2011',
                    'make': 'TestMake',
                    'model': 'TestModel',
                    'vin': 'TESTVIN0000000100',
                    'metric_units': False,
                },
                'eld_device': {
                    'id': 8800001,
                    'identifier': 'TESTELD0001',
                    'model': 'lbb-3.6ca',
                },
                'location': 'Testville, TX',
            },
        },
    ],
    'pagination': {'per_page': 25, 'page_no': 1, 'total': 11871},
}

_FIRST_IDLE_RAW: dict[str, Any] = _IDLE_EVENTS_FIXTURE['idle_events'][0]['idle_event']
_SECOND_IDLE_RAW: dict[str, Any] = _IDLE_EVENTS_FIXTURE['idle_events'][1]['idle_event']
_THIRD_IDLE_RAW: dict[str, Any] = _IDLE_EVENTS_FIXTURE['idle_events'][2]['idle_event']
_EXPECTED_IDLE_COUNT: int = len(_IDLE_EVENTS_FIXTURE['idle_events'])

# Tolerances for floating-point comparisons in convenience-property tests.
_DURATION_SECONDS_TOLERANCE = 1e-6
_FUEL_DELTA_TOLERANCE = 1e-9


class TestIdleEventsEndpointDefinition:
    """Surface assertions on MotiveEndpoints.IDLE_EVENTS."""

    def test_endpoint_path(self) -> None:
        """Should expose the /v1/idle_events path."""

        assert MotiveEndpoints.IDLE_EVENTS.endpoint_path == '/v1/idle_events'

    def test_http_method_is_get(self) -> None:
        """Should use HTTP GET."""

        assert MotiveEndpoints.IDLE_EVENTS.http_method == HTTPMethod.GET

    def test_endpoint_is_paginated(self) -> None:
        """Should be marked paginated."""

        assert MotiveEndpoints.IDLE_EVENTS.is_paginated is True

    def test_max_per_page(self) -> None:
        """Should cap page size at Motive's documented maximum of 100."""

        assert MotiveEndpoints.IDLE_EVENTS.max_per_page == _MAX_PER_PAGE

    def test_response_model(self) -> None:
        """Should parse responses with IdleEventsResponse."""

        assert MotiveEndpoints.IDLE_EVENTS.response_model is IdleEventsResponse

    def test_item_extractor_method(self) -> None:
        """Should extract items via get_idle_events."""

        assert MotiveEndpoints.IDLE_EVENTS.item_extractor_method == 'get_idle_events'

    def test_description_is_non_empty(self) -> None:
        """Should declare a non-empty description for registry introspection."""

        assert MotiveEndpoints.IDLE_EVENTS.description

    def test_is_not_z_suffix_subclass(self) -> None:
        """Should NOT be a Z-suffix subclass instance (bare DATE params only)."""

        assert not isinstance(
            MotiveEndpoints.IDLE_EVENTS,
            MotiveZSuffixDatetimeEndpointDefinition,
        )
        assert isinstance(MotiveEndpoints.IDLE_EVENTS, MotiveEndpointDefinition)

    def test_query_parameter_shape(self) -> None:
        """Should declare start_date and end_date as required DATE params."""

        specs_by_name = {
            spec.name: spec for spec in MotiveEndpoints.IDLE_EVENTS.query_parameters
        }
        assert set(specs_by_name) == {'start_date', 'end_date'}
        for spec in specs_by_name.values():
            assert spec.required is True
            assert spec.parameter_type == ParameterType.DATE

    def test_build_query_params_emits_bare_date(self) -> None:
        """Should serialize date values as YYYY-MM-DD (no time, no Z suffix)."""

        query = MotiveEndpoints.IDLE_EVENTS.build_query_params(
            start_date=date(2026, 5, 14),
            end_date=date(2026, 5, 15),
        )

        assert query['start_date'] == '2026-05-14'
        assert query['end_date'] == '2026-05-15'


class TestIdleEventsResponseParsing:
    """Tests against IdleEventsResponse.model_validate."""

    def test_full_fixture_parses_without_error(self) -> None:
        """Should parse the full fixture without raising ValidationError."""

        IdleEventsResponse.model_validate(_IDLE_EVENTS_FIXTURE)

    def test_get_idle_events_unwraps(self) -> None:
        """Should unwrap into a flat list of IdleEvent instances."""

        parsed = IdleEventsResponse.model_validate(_IDLE_EVENTS_FIXTURE)

        items = parsed.get_idle_events()

        assert len(items) == _EXPECTED_IDLE_COUNT
        assert all(isinstance(item, IdleEvent) for item in items)

    def test_pagination_metadata_parsed(self) -> None:
        """Should parse pagination metadata."""

        parsed = IdleEventsResponse.model_validate(_IDLE_EVENTS_FIXTURE)

        assert (
            parsed.pagination.per_page == _IDLE_EVENTS_FIXTURE['pagination']['per_page']
        )
        assert parsed.pagination.total == _IDLE_EVENTS_FIXTURE['pagination']['total']


class TestIdleEventEdgeCases:
    """Edge-case assertions on null-driver, cross-midnight, and same-day events."""

    def test_null_driver_same_day_record(self) -> None:
        """Should parse the null-driver same-day record."""

        items = IdleEventsResponse.model_validate(
            _IDLE_EVENTS_FIXTURE,
        ).get_idle_events()

        first = items[0]

        assert first.driver is None
        assert first.event_id == _FIRST_IDLE_RAW['id']
        assert isinstance(first.start_time, datetime)
        assert isinstance(first.end_time, datetime)
        assert first.start_time.tzinfo is not None
        assert first.end_time.tzinfo is not None
        assert first.start_time.date() == first.end_time.date()

    def test_cross_midnight_attributed_record(self) -> None:
        """Should parse the cross-midnight attributed record."""

        items = IdleEventsResponse.model_validate(
            _IDLE_EVENTS_FIXTURE,
        ).get_idle_events()

        second = items[1]

        assert isinstance(second.driver, DriverSummary)
        assert second.driver.driver_id == _SECOND_IDLE_RAW['driver']['id']
        assert second.end_time.date() != second.start_time.date()

    def test_same_day_attributed_record(self) -> None:
        """Should parse the long-duration same-day attributed record."""

        items = IdleEventsResponse.model_validate(
            _IDLE_EVENTS_FIXTURE,
        ).get_idle_events()

        third = items[2]

        assert third.driver is not None
        assert third.start_time.date() == third.end_time.date()
        assert third.event_id == _THIRD_IDLE_RAW['id']

    def test_nested_models_reuse_existing_types(self) -> None:
        """Should reuse VehicleSummary and EldDeviceInfo on every record."""

        items = IdleEventsResponse.model_validate(
            _IDLE_EVENTS_FIXTURE,
        ).get_idle_events()

        for event in items:
            assert isinstance(event.vehicle, VehicleSummary)
            assert isinstance(event.eld_device, EldDeviceInfo)


class TestIdleEventConvenienceProperties:
    """Tests for is_unattributed, duration_seconds, and fuel_consumed."""

    def test_is_unattributed_true_for_null_driver(self) -> None:
        """Should return True for the null-driver record."""

        items = IdleEventsResponse.model_validate(
            _IDLE_EVENTS_FIXTURE,
        ).get_idle_events()

        assert items[0].is_unattributed is True

    def test_is_unattributed_false_for_attributed_records(self) -> None:
        """Should return False whenever a driver is attached."""

        items = IdleEventsResponse.model_validate(
            _IDLE_EVENTS_FIXTURE,
        ).get_idle_events()

        assert items[1].is_unattributed is False
        assert items[2].is_unattributed is False

    def test_duration_seconds_matches_elapsed_time(self) -> None:
        """Should equal (end_time - start_time).total_seconds()."""

        items = IdleEventsResponse.model_validate(
            _IDLE_EVENTS_FIXTURE,
        ).get_idle_events()

        first = items[0]
        elapsed = (first.end_time - first.start_time).total_seconds()
        assert abs(first.duration_seconds - elapsed) < _DURATION_SECONDS_TOLERANCE

    def test_fuel_consumed_is_cumulative_delta(self) -> None:
        """Should equal veh_fuel_end - veh_fuel_start."""

        items = IdleEventsResponse.model_validate(
            _IDLE_EVENTS_FIXTURE,
        ).get_idle_events()

        first = items[0]
        expected = _FIRST_IDLE_RAW['veh_fuel_end'] - _FIRST_IDLE_RAW['veh_fuel_start']
        assert abs(first.fuel_consumed - expected) < _FUEL_DELTA_TOLERANCE


# A minimal valid ``DriverSummary`` payload used by the audit tests
# below. Kept inline rather than promoted to a module-level fixture
# because only the drift tests need it.
_BASE_DRIVER_SUMMARY_RAW: dict[str, Any] = {
    'id': 9000099,
    'first_name': 'Drift',
    'last_name': 'Tester',
    'username': 'drift.tester',
    'email': 'drift.tester@example.com',
    'driver_company_id': 'TEST-DRIFT',
    'status': 'active',
    'role': 'driver',
}


class TestMotiveModelDriftRegressions:
    """Confirmed-shape regressions surfaced by the live ``/v1/driving_periods`` run."""

    def test_driving_period_accepts_int_annotation_status(self) -> None:
        """Live API emits ``annotation_status`` as an int (e.g. 1); model must accept."""

        payload = dict(_THIRD_PERIOD_RAW)
        payload['annotation_status'] = 1

        parsed = DrivingPeriod.model_validate(payload)

        assert parsed.annotation_status == 1

    def test_driving_period_accepts_null_source(self) -> None:
        """Live API emits ``source: null`` on some records; model must accept."""

        payload = dict(_THIRD_PERIOD_RAW)
        payload['source'] = None

        parsed = DrivingPeriod.model_validate(payload)

        assert parsed.source is None


class TestMotiveModelDriftAuditCoverage:
    """Audit-pass coverage: previously-strict unused fields now widen to nullable."""

    @pytest.mark.parametrize('field_name', ['status', 'type', 'duration'])
    def test_driving_period_accepts_null_in_unused_strict_fields(
        self, field_name: str
    ) -> None:
        """Newly-nullable unused fields on ``DrivingPeriod`` accept ``None``."""

        payload = dict(_THIRD_PERIOD_RAW)
        payload[field_name] = None

        parsed = DrivingPeriod.model_validate(payload)

        assert getattr(parsed, field_name) is None

    @pytest.mark.parametrize(
        'field_name',
        [
            'veh_fuel_start',
            'veh_fuel_end',
            'lat',
            'lon',
            'city',
            'state',
            'rg_brg',
            'rg_km',
            'rg_match',
            'end_type',
            'eld_device',
            'location',
        ],
    )
    def test_idle_event_accepts_null_in_unused_strict_fields(
        self, field_name: str
    ) -> None:
        """Newly-nullable unused fields on ``IdleEvent`` accept ``None``."""

        payload = dict(_THIRD_IDLE_RAW)
        payload[field_name] = None

        parsed = IdleEvent.model_validate(payload)

        assert getattr(parsed, field_name) is None

    @pytest.mark.parametrize(
        'status_value', ['supervisor', 'fleet_manager_advanced', 'archived']
    )
    def test_driver_summary_accepts_arbitrary_status(self, status_value: str) -> None:
        """``DriverSummary.status`` accepts strings outside the historical enum set."""

        payload = dict(_BASE_DRIVER_SUMMARY_RAW)
        payload['status'] = status_value

        parsed = DriverSummary.model_validate(payload)

        assert parsed.status == status_value

    @pytest.mark.parametrize(
        'role_value', ['supervisor', 'fleet_manager_advanced', 'archived']
    )
    def test_driver_summary_accepts_arbitrary_role(self, role_value: str) -> None:
        """``DriverSummary.role`` accepts strings outside the historical enum set."""

        payload = dict(_BASE_DRIVER_SUMMARY_RAW)
        payload['role'] = role_value

        parsed = DriverSummary.model_validate(payload)

        assert parsed.role == role_value


class TestMotiveModelDriftConsumedFieldsStrictness:
    """Fields the unifier consumes stay strict -- loud failure on drift is intentional."""

    @pytest.mark.parametrize(
        'field_name',
        ['period_id', 'start_time', 'end_time', 'start_kilometers', 'end_kilometers'],
    )
    def test_driving_period_rejects_null_in_consumed_fields(
        self, field_name: str
    ) -> None:
        """Setting a unifier-consumed ``DrivingPeriod`` field to None raises."""

        # The alias for ``period_id`` is ``id``; the rest use the same
        # snake_case key as the attribute name.
        payload = dict(_THIRD_PERIOD_RAW)
        payload_key = 'id' if field_name == 'period_id' else field_name
        payload[payload_key] = None

        with pytest.raises(ValidationError):
            DrivingPeriod.model_validate(payload)

    @pytest.mark.parametrize('field_name', ['event_id', 'start_time', 'end_time'])
    def test_idle_event_rejects_null_in_consumed_fields(
        self, field_name: str
    ) -> None:
        """Setting a unifier-consumed ``IdleEvent`` field to None raises."""

        payload = dict(_THIRD_IDLE_RAW)
        payload_key = 'id' if field_name == 'event_id' else field_name
        payload[payload_key] = None

        with pytest.raises(ValidationError):
            IdleEvent.model_validate(payload)
