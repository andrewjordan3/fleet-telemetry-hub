"""Tests for the Motive /v2/vehicle_utilization endpoint definition.

Covers the endpoint surface, query-parameter serialization (including the
Z-suffix datetime conversion enforced by
MotiveZSuffixDatetimeEndpointDefinition), response parsing, pagination
state computation, convenience properties on VehicleUtilization, and the
frozen behavior inherited from FrozenResponseModelBase.

All identifiers in test data are synthetic. No real VINs, vehicle IDs,
or fleet numbers from any production fleet appear in this file.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from fleet_telemetry_hub.models.motive_requests import (
    MotiveEndpoints,
    MotiveZSuffixDatetimeEndpointDefinition,
)
from fleet_telemetry_hub.models.motive_responses import (
    VehicleSummary,
    VehicleUtilization,
    VehicleUtilizationsResponse,
)
from fleet_telemetry_hub.models.shared_request_models import HTTPMethod
from fleet_telemetry_hub.models.shared_response_models import (
    ParameterType,
    ProviderCredentials,
)


_SAMPLE_RESPONSE_JSON: dict[str, Any] = {
    'vehicle_utilizations': [
        {
            'vehicle_utilization': {
                'message': (
                    'the vehicle has not communicated with us yet. '
                    'Please check if this vehicle is connected and in '
                    'coverage area.'
                ),
                'last_located_at': None,
                'utilization_percentage': 0,
                'idle_time': 0,
                'idle_fuel': 0,
                'driving_time': 0,
                'driving_fuel': 0,
                'total_fuel': 0,
                'total_distance': 0,
                'vehicle': {
                    'id': 1001,
                    'number': 'TEST-001',
                    'year': '2020',
                    'make': 'TestMake',
                    'model': 'TypeA',
                    'vin': 'TESTVIN0000000001',
                    'metric_units': False,
                },
            },
        },
        {
            'vehicle_utilization': {
                'message': '',
                'last_located_at': '2026-05-07T12:11:17+09:00',
                'utilization_percentage': 47.47,
                'idle_time': 12087,
                'idle_fuel': 2.75935909375,
                'driving_time': 10921,
                'driving_fuel': 19.375365125000002,
                'total_fuel': 22.13472421875,
                'total_distance': 129.245168,
                'vehicle': {
                    'id': 1002,
                    'number': 'TEST-002',
                    'year': '2022',
                    'make': 'TestMake',
                    'model': 'TypeB',
                    'vin': 'TESTVIN0000000002',
                    'metric_units': False,
                },
            },
        },
    ],
    'pagination': {'page_no': 1, 'per_page': 25, 'total': 1455},
}


def _build_minimal_record(
    *,
    idle_time: int = 0,
    driving_time: int = 0,
    message: str | None = None,
    include_message_field: bool = True,
) -> VehicleUtilization:
    """Construct a minimal synthetic VehicleUtilization for property tests."""

    payload: dict[str, Any] = {
        'last_located_at': None,
        'utilization_percentage': 0,
        'idle_time': idle_time,
        'idle_fuel': 0,
        'driving_time': driving_time,
        'driving_fuel': 0,
        'total_fuel': 0,
        'total_distance': 0,
        'vehicle': {
            'id': 9999,
            'number': 'TEST-099',
            'vin': 'TESTVIN0000000099',
            'metric_units': False,
        },
    }

    if include_message_field:
        payload['message'] = message

    return VehicleUtilization.model_validate(payload)


def _build_paginated_response_json(
    *,
    page_no: int,
    per_page: int,
    total: int,
) -> dict[str, Any]:
    """Construct a response payload with one minimal record for pagination tests."""

    return {
        'vehicle_utilizations': [
            {
                'vehicle_utilization': {
                    'message': '',
                    'last_located_at': None,
                    'utilization_percentage': 0,
                    'idle_time': 0,
                    'idle_fuel': 0,
                    'driving_time': 0,
                    'driving_fuel': 0,
                    'total_fuel': 0,
                    'total_distance': 0,
                    'vehicle': {
                        'id': 9999,
                        'number': 'TEST-099',
                        'vin': 'TESTVIN0000000099',
                        'metric_units': False,
                    },
                },
            },
        ],
        'pagination': {
            'page_no': page_no,
            'per_page': per_page,
            'total': total,
        },
    }


class TestVehicleUtilizationEndpointDefinition:
    """Surface assertions on MotiveEndpoints.VEHICLE_UTILIZATION."""

    def test_endpoint_path_is_v2_vehicle_utilization(self) -> None:
        """Should expose the /v2/vehicle_utilization path."""

        assert (
            MotiveEndpoints.VEHICLE_UTILIZATION.endpoint_path
            == '/v2/vehicle_utilization'
        )

    def test_http_method_is_get(self) -> None:
        """Should use HTTP GET."""

        assert MotiveEndpoints.VEHICLE_UTILIZATION.http_method == HTTPMethod.GET

    def test_endpoint_is_paginated(self) -> None:
        """Should be marked paginated."""

        assert MotiveEndpoints.VEHICLE_UTILIZATION.is_paginated is True

    def test_max_per_page_is_100(self) -> None:
        """Should cap page size at Motive's documented maximum of 100."""

        assert MotiveEndpoints.VEHICLE_UTILIZATION.max_per_page == 100

    def test_response_model_is_vehicle_utilizations_response(self) -> None:
        """Should parse responses with VehicleUtilizationsResponse."""

        assert (
            MotiveEndpoints.VEHICLE_UTILIZATION.response_model
            is VehicleUtilizationsResponse
        )

    def test_item_extractor_method_name(self) -> None:
        """Should extract items via get_vehicle_utilizations."""

        assert (
            MotiveEndpoints.VEHICLE_UTILIZATION.item_extractor_method
            == 'get_vehicle_utilizations'
        )

    def test_endpoint_is_z_suffix_subclass(self) -> None:
        """Should be an instance of the Z-suffix datetime subclass."""

        assert isinstance(
            MotiveEndpoints.VEHICLE_UTILIZATION,
            MotiveZSuffixDatetimeEndpointDefinition,
        )

    def test_required_query_params(self) -> None:
        """Should declare start_at and end_at as required DATETIME params."""

        specs_by_name = {
            spec.name: spec
            for spec in MotiveEndpoints.VEHICLE_UTILIZATION.query_parameters
        }
        assert set(specs_by_name) == {'start_at', 'end_at'}
        for spec in specs_by_name.values():
            assert spec.required is True
            assert spec.parameter_type == ParameterType.DATETIME

    def test_no_path_parameters(self) -> None:
        """Should declare no path parameters."""

        assert MotiveEndpoints.VEHICLE_UTILIZATION.path_parameters == ()


class TestVehicleUtilizationQueryParamSerialization:
    """Serialization tests for build_query_params on VEHICLE_UTILIZATION."""

    def test_tz_aware_utc_datetime_serializes_with_z_suffix(self) -> None:
        """Should serialize tz-aware UTC datetimes with a Z suffix."""

        utc_value = datetime(2026, 5, 6, 0, 0, 0, tzinfo=timezone.utc)

        query = MotiveEndpoints.VEHICLE_UTILIZATION.build_query_params(
            start_at=utc_value,
            end_at=utc_value,
        )

        assert query['start_at'] == '2026-05-06T00:00:00Z'
        assert query['end_at'] == '2026-05-06T00:00:00Z'

    def test_tz_aware_non_utc_datetime_converts_to_utc(self) -> None:
        """Should convert tz-aware non-UTC datetimes to UTC before formatting."""

        chicago_offset = timezone(timedelta(hours=-5))
        chicago_value = datetime(2026, 5, 6, 17, 0, 0, tzinfo=chicago_offset)

        query = MotiveEndpoints.VEHICLE_UTILIZATION.build_query_params(
            start_at=chicago_value,
            end_at=chicago_value,
        )

        assert query['start_at'] == '2026-05-06T22:00:00Z'

    def test_naive_datetime_assumed_utc(self) -> None:
        """Should format naive datetimes as-is and append Z."""

        naive_value = datetime(2026, 5, 6, 0, 0, 0)

        query = MotiveEndpoints.VEHICLE_UTILIZATION.build_query_params(
            start_at=naive_value,
            end_at=naive_value,
        )

        assert query['start_at'] == '2026-05-06T00:00:00Z'

    def test_missing_start_at_raises_value_error(self) -> None:
        """Should raise ValueError mentioning start_at when missing."""

        with pytest.raises(ValueError, match='start_at'):
            MotiveEndpoints.VEHICLE_UTILIZATION.build_query_params(
                end_at=datetime(2026, 5, 6, 0, 0, 0, tzinfo=timezone.utc),
            )

    def test_missing_end_at_raises_value_error(self) -> None:
        """Should raise ValueError mentioning end_at when missing."""

        with pytest.raises(ValueError, match='end_at'):
            MotiveEndpoints.VEHICLE_UTILIZATION.build_query_params(
                start_at=datetime(2026, 5, 6, 0, 0, 0, tzinfo=timezone.utc),
            )

    def test_pagination_params_included(self) -> None:
        """Should include first-page pagination params when none supplied."""

        utc_value = datetime(2026, 5, 6, 0, 0, 0, tzinfo=timezone.utc)
        endpoint = MotiveEndpoints.VEHICLE_UTILIZATION

        query = endpoint.build_query_params(
            pagination_state=endpoint.get_initial_pagination_state(),
            start_at=utc_value,
            end_at=utc_value,
        )

        assert query['page_no'] == '1'
        assert query['per_page'] == '100'


class TestVehicleUtilizationRequestSpec:
    """Tests for build_request_spec on VEHICLE_UTILIZATION."""

    def test_url_composition(
        self, provider_credentials: ProviderCredentials
    ) -> None:
        """Should compose URL from base_url and endpoint path."""

        utc_value = datetime(2026, 5, 6, 0, 0, 0, tzinfo=timezone.utc)

        spec = MotiveEndpoints.VEHICLE_UTILIZATION.build_request_spec(
            provider_credentials,
            start_at=utc_value,
            end_at=utc_value,
        )

        assert (
            spec.url
            == f'{provider_credentials.base_url}/v2/vehicle_utilization'
        )

    def test_x_api_key_header_set_from_credentials(
        self, provider_credentials: ProviderCredentials
    ) -> None:
        """Should populate X-API-Key from credentials."""

        utc_value = datetime(2026, 5, 6, 0, 0, 0, tzinfo=timezone.utc)

        spec = MotiveEndpoints.VEHICLE_UTILIZATION.build_request_spec(
            provider_credentials,
            start_at=utc_value,
            end_at=utc_value,
        )

        assert (
            spec.headers['X-API-Key']
            == provider_credentials.api_key.get_secret_value()
        )

    def test_query_params_include_z_suffix_datetimes(
        self, provider_credentials: ProviderCredentials
    ) -> None:
        """Should serialize start_at and end_at with Z suffix in the spec."""

        utc_value = datetime(2026, 5, 6, 0, 0, 0, tzinfo=timezone.utc)

        spec = MotiveEndpoints.VEHICLE_UTILIZATION.build_request_spec(
            provider_credentials,
            start_at=utc_value,
            end_at=utc_value,
        )

        assert spec.query_params['start_at'].endswith('Z')
        assert spec.query_params['end_at'].endswith('Z')

    def test_request_spec_includes_pagination_params(
        self, provider_credentials: ProviderCredentials
    ) -> None:
        """Should include page_no and per_page in the spec query params."""

        utc_value = datetime(2026, 5, 6, 0, 0, 0, tzinfo=timezone.utc)

        spec = MotiveEndpoints.VEHICLE_UTILIZATION.build_request_spec(
            provider_credentials,
            start_at=utc_value,
            end_at=utc_value,
        )

        assert 'page_no' in spec.query_params
        assert 'per_page' in spec.query_params


class TestVehicleUtilizationResponseParsing:
    """Tests against VehicleUtilizationsResponse.model_validate."""

    def test_parses_full_response(self) -> None:
        """Should parse the sample response into two wrapper entries."""

        parsed = VehicleUtilizationsResponse.model_validate(_SAMPLE_RESPONSE_JSON)

        assert len(parsed.vehicle_utilizations) == 2

    def test_get_vehicle_utilizations_unwraps_records(self) -> None:
        """Should unwrap two VehicleUtilization instances from the response."""

        parsed = VehicleUtilizationsResponse.model_validate(_SAMPLE_RESPONSE_JSON)

        items = parsed.get_vehicle_utilizations()

        assert len(items) == 2
        assert all(isinstance(item, VehicleUtilization) for item in items)

    def test_pagination_metadata_parsed(self) -> None:
        """Should parse page_no, per_page, and total from pagination block."""

        parsed = VehicleUtilizationsResponse.model_validate(_SAMPLE_RESPONSE_JSON)

        assert parsed.pagination.page_no == 1
        assert parsed.pagination.per_page == 25
        assert parsed.pagination.total == 1455

    def test_populated_message_is_preserved(self) -> None:
        """Should preserve the populated diagnostic message verbatim."""

        items = VehicleUtilizationsResponse.model_validate(
            _SAMPLE_RESPONSE_JSON
        ).get_vehicle_utilizations()

        assert items[0].message is not None
        assert items[0].message.startswith('the vehicle has not communicated')

    def test_empty_message_normalizes_to_none(self) -> None:
        """Should normalize Motive's empty-string message to None."""

        items = VehicleUtilizationsResponse.model_validate(
            _SAMPLE_RESPONSE_JSON
        ).get_vehicle_utilizations()

        assert items[1].message is None

    def test_last_located_at_parsed_when_present(self) -> None:
        """Should parse last_located_at as a tz-aware datetime."""

        items = VehicleUtilizationsResponse.model_validate(
            _SAMPLE_RESPONSE_JSON
        ).get_vehicle_utilizations()

        assert items[1].last_located_at is not None
        assert items[1].last_located_at.tzinfo is not None

    def test_last_located_at_is_none_when_null(self) -> None:
        """Should leave last_located_at as None when absent in payload."""

        items = VehicleUtilizationsResponse.model_validate(
            _SAMPLE_RESPONSE_JSON
        ).get_vehicle_utilizations()

        assert items[0].last_located_at is None

    def test_vehicle_summary_fields_parsed(self) -> None:
        """Should parse the embedded VehicleSummary fields."""

        items = VehicleUtilizationsResponse.model_validate(
            _SAMPLE_RESPONSE_JSON
        ).get_vehicle_utilizations()

        assert items[1].vehicle.vin == 'TESTVIN0000000002'
        assert items[1].vehicle.year == '2022'
        assert items[1].vehicle.make == 'TestMake'
        assert items[1].vehicle.metric_units is False


class TestVehicleUtilizationConvenienceProperties:
    """Tests for engine_on_seconds, engine_on_hours, has_communication_issue."""

    def test_engine_on_seconds_sums_idle_and_driving(self) -> None:
        """Should sum idle_time and driving_time."""

        record = _build_minimal_record(idle_time=100, driving_time=200)

        assert record.engine_on_seconds == 300

    def test_engine_on_hours_divides_seconds_by_3600(self) -> None:
        """Should return engine_on_seconds / 3600."""

        record = _build_minimal_record(idle_time=1800, driving_time=1800)

        assert abs(record.engine_on_hours - 1.0) < 1e-9

    def test_has_communication_issue_true_when_message_populated(self) -> None:
        """Should return True when message contains a diagnostic string."""

        record = _build_minimal_record(message='vehicle offline')

        assert record.has_communication_issue is True

    def test_has_communication_issue_false_when_message_empty_string(
        self,
    ) -> None:
        """Should return False when Motive's empty-string message normalizes to None."""

        record = _build_minimal_record(message='')

        assert record.has_communication_issue is False

    def test_has_communication_issue_false_when_message_field_absent(
        self,
    ) -> None:
        """Should return False when message is omitted from the payload."""

        record = _build_minimal_record(include_message_field=False)

        assert record.has_communication_issue is False


class TestVehicleUtilizationPaginationState:
    """Tests for pagination state computation via parse_response."""

    def test_first_page_advances_to_next(self) -> None:
        """Should produce next_page_params advancing to page 2."""

        response_json = _build_paginated_response_json(
            page_no=1, per_page=100, total=200
        )

        parsed = MotiveEndpoints.VEHICLE_UTILIZATION.parse_response(
            response_json
        )

        assert parsed.pagination.has_next_page is True
        assert parsed.pagination.next_page_params == {
            'page_no': 2,
            'per_page': 100,
        }

    def test_last_page_has_no_next(self) -> None:
        """Should report has_next_page=False on the final page."""

        response_json = _build_paginated_response_json(
            page_no=2, per_page=100, total=200
        )

        parsed = MotiveEndpoints.VEHICLE_UTILIZATION.parse_response(
            response_json
        )

        assert parsed.pagination.has_next_page is False

    def test_single_page_has_no_next(self) -> None:
        """Should report has_next_page=False when total fits in one page."""

        response_json = _build_paginated_response_json(
            page_no=1, per_page=100, total=50
        )

        parsed = MotiveEndpoints.VEHICLE_UTILIZATION.parse_response(
            response_json
        )

        assert parsed.pagination.has_next_page is False

    def test_pagination_state_includes_total_items(self) -> None:
        """Should propagate the response's total into PaginationState."""

        response_json = _build_paginated_response_json(
            page_no=1, per_page=100, total=200
        )

        parsed = MotiveEndpoints.VEHICLE_UTILIZATION.parse_response(
            response_json
        )

        assert parsed.pagination.total_items == 200


class TestVehicleUtilizationFrozen:
    """Tests for frozen=True behavior on the new response models."""

    def test_cannot_mutate_vehicle_summary(self) -> None:
        """Should reject attribute assignment on a parsed VehicleSummary."""

        summary = VehicleSummary.model_validate(
            {
                'id': 9999,
                'number': 'TEST-099',
                'vin': 'TESTVIN0000000099',
                'metric_units': False,
            }
        )

        with pytest.raises((TypeError, ValidationError)):
            summary.year = '2099'  # pyright: ignore[reportAttributeAccessIssue]

    def test_cannot_mutate_vehicle_utilization(self) -> None:
        """Should reject attribute assignment on a parsed VehicleUtilization."""

        record = _build_minimal_record(idle_time=10, driving_time=20)

        with pytest.raises((TypeError, ValidationError)):
            record.idle_time = 999  # pyright: ignore[reportAttributeAccessIssue]

    def test_vehicle_utilization_is_hashable(self) -> None:
        """Should be hashable since all fields are hashable."""

        record = _build_minimal_record(idle_time=10, driving_time=20)

        assert isinstance(hash(record), int)
