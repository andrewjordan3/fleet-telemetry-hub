"""Tests for the Samsara utilization endpoint definitions.

Covers the three new endpoints added alongside the Patriot utilization
integration:

- ``SamsaraEndpoints.VEHICLE_FUEL_ENERGY`` (/fleet/reports/vehicles/fuel-energy)
- ``SamsaraEndpoints.DRIVER_FUEL_ENERGY`` (/fleet/reports/drivers/fuel-energy)
- ``SamsaraEndpoints.IDLING_EVENTS`` (/idling/events)

Each section asserts the endpoint surface (path, method, pagination,
response wiring), the declared ``QueryParameterSpec`` shape, and the
``build_query_params`` round-trip including Z-suffix UTC datetime
serialization. Registry-resolution tests confirm the dynamic pickup
via ``SamsaraEndpoints.get_all_endpoints()``.
"""

from datetime import UTC, datetime

from pydantic.main import BaseModel

from fleet_telemetry_hub.models import EndpointDefinition
from fleet_telemetry_hub.models.samsara_requests import (
    SamsaraEndpointDefinition,
    SamsaraEndpoints,
)
from fleet_telemetry_hub.models.samsara_responses import (
    DriverFuelEnergyResponse,
    FuelEnergyResponse,
    IdlingEventsResponse,
)
from fleet_telemetry_hub.models.shared_request_models import HTTPMethod
from fleet_telemetry_hub.models.shared_response_models import ParameterType
from fleet_telemetry_hub.registry import EndpointRegistry

_UTC_START = datetime(2026, 5, 14, 0, 0, 0, tzinfo=UTC)
_UTC_END = datetime(2026, 5, 15, 0, 0, 0, tzinfo=UTC)


class TestVehicleFuelEnergyEndpointDefinition:
    """Surface assertions on SamsaraEndpoints.VEHICLE_FUEL_ENERGY."""

    def test_endpoint_path(self) -> None:
        """Should expose the /fleet/reports/vehicles/fuel-energy path."""

        assert (
            SamsaraEndpoints.VEHICLE_FUEL_ENERGY.endpoint_path
            == '/fleet/reports/vehicles/fuel-energy'
        )

    def test_http_method_is_get(self) -> None:
        """Should use HTTP GET."""

        assert SamsaraEndpoints.VEHICLE_FUEL_ENERGY.http_method == HTTPMethod.GET

    def test_endpoint_is_paginated(self) -> None:
        """Should be marked paginated."""

        assert SamsaraEndpoints.VEHICLE_FUEL_ENERGY.is_paginated is True

    def test_response_model(self) -> None:
        """Should parse responses with FuelEnergyResponse."""

        assert SamsaraEndpoints.VEHICLE_FUEL_ENERGY.response_model is FuelEnergyResponse

    def test_item_extractor_method(self) -> None:
        """Should extract items via get_items."""

        assert SamsaraEndpoints.VEHICLE_FUEL_ENERGY.item_extractor_method == 'get_items'

    def test_description_is_non_empty(self) -> None:
        """Should declare a non-empty description for registry introspection."""

        assert SamsaraEndpoints.VEHICLE_FUEL_ENERGY.description

    def test_query_parameter_shape(self) -> None:
        """Should declare start_date and end_date as required DATETIME params."""

        specs_by_name = {
            spec.name: spec
            for spec in SamsaraEndpoints.VEHICLE_FUEL_ENERGY.query_parameters
        }
        assert set(specs_by_name) == {'start_date', 'end_date'}
        assert specs_by_name['start_date'].api_name == 'startDate'
        assert specs_by_name['end_date'].api_name == 'endDate'
        for spec in specs_by_name.values():
            assert spec.required is True
            assert spec.parameter_type == ParameterType.DATETIME

    def test_build_query_params_serializes_with_z_suffix(self) -> None:
        """Should emit Z-suffix UTC datetimes under the API-side keys."""

        query = SamsaraEndpoints.VEHICLE_FUEL_ENERGY.build_query_params(
            start_date=_UTC_START,
            end_date=_UTC_END,
        )

        assert query['startDate'] == '2026-05-14T00:00:00Z'
        assert query['endDate'] == '2026-05-15T00:00:00Z'


class TestDriverFuelEnergyEndpointDefinition:
    """Surface assertions on SamsaraEndpoints.DRIVER_FUEL_ENERGY."""

    def test_endpoint_path(self) -> None:
        """Should expose the /fleet/reports/drivers/fuel-energy path."""

        assert (
            SamsaraEndpoints.DRIVER_FUEL_ENERGY.endpoint_path
            == '/fleet/reports/drivers/fuel-energy'
        )

    def test_http_method_is_get(self) -> None:
        """Should use HTTP GET."""

        assert SamsaraEndpoints.DRIVER_FUEL_ENERGY.http_method == HTTPMethod.GET

    def test_endpoint_is_paginated(self) -> None:
        """Should be marked paginated."""

        assert SamsaraEndpoints.DRIVER_FUEL_ENERGY.is_paginated is True

    def test_response_model(self) -> None:
        """Should parse responses with DriverFuelEnergyResponse."""

        assert (
            SamsaraEndpoints.DRIVER_FUEL_ENERGY.response_model
            is DriverFuelEnergyResponse
        )

    def test_item_extractor_method(self) -> None:
        """Should extract items via get_items."""

        assert SamsaraEndpoints.DRIVER_FUEL_ENERGY.item_extractor_method == 'get_items'

    def test_description_is_non_empty(self) -> None:
        """Should declare a non-empty description for registry introspection."""

        assert SamsaraEndpoints.DRIVER_FUEL_ENERGY.description

    def test_query_parameter_shape(self) -> None:
        """Should declare start_date and end_date as required DATETIME params."""

        specs_by_name = {
            spec.name: spec
            for spec in SamsaraEndpoints.DRIVER_FUEL_ENERGY.query_parameters
        }
        assert set(specs_by_name) == {'start_date', 'end_date'}
        assert specs_by_name['start_date'].api_name == 'startDate'
        assert specs_by_name['end_date'].api_name == 'endDate'
        for spec in specs_by_name.values():
            assert spec.required is True
            assert spec.parameter_type == ParameterType.DATETIME

    def test_build_query_params_serializes_with_z_suffix(self) -> None:
        """Should emit Z-suffix UTC datetimes under the API-side keys."""

        query = SamsaraEndpoints.DRIVER_FUEL_ENERGY.build_query_params(
            start_date=_UTC_START,
            end_date=_UTC_END,
        )

        assert query['startDate'] == '2026-05-14T00:00:00Z'
        assert query['endDate'] == '2026-05-15T00:00:00Z'


class TestIdlingEventsEndpointDefinition:
    """Surface assertions on SamsaraEndpoints.IDLING_EVENTS."""

    def test_endpoint_path(self) -> None:
        """Should expose the literal /idling/events path (no /fleet prefix)."""

        assert SamsaraEndpoints.IDLING_EVENTS.endpoint_path == '/idling/events'

    def test_http_method_is_get(self) -> None:
        """Should use HTTP GET."""

        assert SamsaraEndpoints.IDLING_EVENTS.http_method == HTTPMethod.GET

    def test_endpoint_is_paginated(self) -> None:
        """Should be marked paginated."""

        assert SamsaraEndpoints.IDLING_EVENTS.is_paginated is True

    def test_response_model(self) -> None:
        """Should parse responses with IdlingEventsResponse."""

        assert SamsaraEndpoints.IDLING_EVENTS.response_model is IdlingEventsResponse

    def test_item_extractor_method(self) -> None:
        """Should extract items via get_items."""

        assert SamsaraEndpoints.IDLING_EVENTS.item_extractor_method == 'get_items'

    def test_description_is_non_empty(self) -> None:
        """Should declare a non-empty description for registry introspection."""

        assert SamsaraEndpoints.IDLING_EVENTS.description

    def test_query_parameter_shape(self) -> None:
        """Should declare required time range plus optional ID list filters."""

        specs_by_name = {
            spec.name: spec for spec in SamsaraEndpoints.IDLING_EVENTS.query_parameters
        }
        assert set(specs_by_name) == {
            'start_time',
            'end_time',
            'operator_ids',
            'asset_ids',
        }

        assert specs_by_name['start_time'].api_name == 'startTime'
        assert specs_by_name['start_time'].parameter_type == ParameterType.DATETIME
        assert specs_by_name['start_time'].required is True

        assert specs_by_name['end_time'].api_name == 'endTime'
        assert specs_by_name['end_time'].parameter_type == ParameterType.DATETIME
        assert specs_by_name['end_time'].required is True

        assert specs_by_name['operator_ids'].api_name == 'operatorIds'
        assert specs_by_name['operator_ids'].parameter_type == ParameterType.STRING_LIST
        assert specs_by_name['operator_ids'].required is False

        assert specs_by_name['asset_ids'].api_name == 'assetIds'
        assert specs_by_name['asset_ids'].parameter_type == ParameterType.STRING_LIST
        assert specs_by_name['asset_ids'].required is False

    def test_build_query_params_serializes_with_z_suffix(self) -> None:
        """Should emit Z-suffix UTC datetimes under startTime/endTime."""

        query = SamsaraEndpoints.IDLING_EVENTS.build_query_params(
            start_time=_UTC_START,
            end_time=_UTC_END,
        )

        assert query['startTime'] == '2026-05-14T00:00:00Z'
        assert query['endTime'] == '2026-05-15T00:00:00Z'

    def test_operator_ids_serialized_as_comma_separated(self) -> None:
        """Should serialize operator_ids as Samsara's comma-separated format."""

        query = SamsaraEndpoints.IDLING_EVENTS.build_query_params(
            start_time=_UTC_START,
            end_time=_UTC_END,
            operator_ids=['1000001', '1000002'],
        )

        assert query['operatorIds'] == '1000001,1000002'

    def test_asset_ids_serialized_as_comma_separated(self) -> None:
        """Should serialize asset_ids as Samsara's comma-separated format."""

        query = SamsaraEndpoints.IDLING_EVENTS.build_query_params(
            start_time=_UTC_START,
            end_time=_UTC_END,
            asset_ids=['999999900000005', '999999900000006'],
        )

        assert query['assetIds'] == '999999900000005,999999900000006'

    def test_optional_id_filters_absent_when_not_passed(self) -> None:
        """Should omit operatorIds/assetIds entirely when caller skips them."""

        query = SamsaraEndpoints.IDLING_EVENTS.build_query_params(
            start_time=_UTC_START,
            end_time=_UTC_END,
        )

        assert 'operatorIds' not in query
        assert 'assetIds' not in query


class TestUtilizationEndpointsRegistryResolution:
    """Verifies EndpointRegistry picks up each new endpoint by name."""

    def test_vehicle_fuel_energy_resolves(self) -> None:
        """Should resolve 'vehicle_fuel_energy' from the samsara provider."""

        registry = EndpointRegistry()

        endpoint: EndpointDefinition[BaseModel] = registry.get(
            'samsara', 'vehicle_fuel_energy'
        )

        assert isinstance(endpoint, EndpointDefinition)
        assert isinstance(endpoint, SamsaraEndpointDefinition)
        assert endpoint.endpoint_path == '/fleet/reports/vehicles/fuel-energy'

    def test_driver_fuel_energy_resolves(self) -> None:
        """Should resolve 'driver_fuel_energy' from the samsara provider."""

        registry = EndpointRegistry()

        endpoint: EndpointDefinition[BaseModel] = registry.get(
            'samsara', 'driver_fuel_energy'
        )

        assert isinstance(endpoint, EndpointDefinition)
        assert isinstance(endpoint, SamsaraEndpointDefinition)
        assert endpoint.endpoint_path == '/fleet/reports/drivers/fuel-energy'

    def test_idling_events_resolves(self) -> None:
        """Should resolve 'idling_events' from the samsara provider."""

        registry = EndpointRegistry()

        endpoint: EndpointDefinition[BaseModel] = registry.get(
            'samsara', 'idling_events'
        )

        assert isinstance(endpoint, EndpointDefinition)
        assert isinstance(endpoint, SamsaraEndpointDefinition)
        assert endpoint.endpoint_path == '/idling/events'
