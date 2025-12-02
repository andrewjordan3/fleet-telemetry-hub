# fleet_telemetry_hub/models/samsara_requests.py

import logging
from datetime import datetime
from typing import Any, Generic

from pydantic import ConfigDict, Field

from .samsara_responses import (
    AddressesResponse,
    DriversResponse,
    DriverVehicleAssignment,
    DriverVehicleAssignmentsResponse,
    LocationStreamRecord,
    LocationStreamResponse,
    SamsaraAddress,
    SamsaraDriver,
    SamsaraPaginationInfo,
    SamsaraVehicle,
    VehiclesResponse,
    VehicleStatsHistoryRecord,
    VehicleStatsHistoryResponse,
)
from .shared_request_models import HTTPMethod, RequestSpec
from .shared_response_models import (
    EndpointDefinition,
    ItemT,
    PaginationState,
    ParameterType,
    ParsedResponse,
    ProviderCredentials,
    QueryParameterSpec,
    ResponseModelT,
)

logger: logging.Logger = logging.getLogger(__name__)


class SamsaraEndpointDefinition(
    EndpointDefinition[ResponseModelT, ItemT],
    Generic[ResponseModelT, ItemT],
):
    """
    Samsara-specific endpoint definition with cursor-based pagination.

    Samsara uses 'after' query parameter with opaque cursor tokens.
    Responses include pagination object with 'endCursor' and 'hasNextPage'.
    """

    model_config = ConfigDict(extra='forbid', frozen=True, arbitrary_types_allowed=True)

    response_model: type[ResponseModelT]
    item_extractor_method: str = Field(
        default='get_items',
        description='Method name on response model to extract items',
    )

    def build_request_spec(
        self,
        credentials: ProviderCredentials,
        pagination_state: PaginationState | None = None,
        **params: Any,
    ) -> RequestSpec:
        """
        Build complete request specification for Samsara API.

        Injects Bearer token authentication header and handles cursor-based
        pagination via the 'after' query parameter.

        Args:
            credentials: Samsara API credentials and connection settings.
            pagination_state: Optional pagination state with cursor.
            **params: Path and query parameters.

        Returns:
            RequestSpec ready for HTTP execution.
        """
        # Separate path params from query params
        path_param_names: set[str] = {p.name for p in self.path_parameters}
        path_params: dict[str, Any] = {
            k: v for k, v in params.items() if k in path_param_names
        }
        query_params_input: dict[str, Any] = {
            k: v for k, v in params.items() if k not in path_param_names
        }

        # Use initial pagination if none provided for paginated endpoints
        effective_pagination: PaginationState | None = pagination_state
        if effective_pagination is None and self.is_paginated:
            effective_pagination = self.get_initial_pagination_state()

        # Build URL and query params
        url: str = self.build_url(credentials.base_url, **path_params)
        query_params: dict[str, str] = self.build_query_params(
            pagination_state=effective_pagination,
            **query_params_input,
        )

        # Build Samsara-specific headers (Bearer token auth)
        headers: dict[str, str] = {
            'Authorization': f'Bearer {credentials.api_key.get_secret_value()}',
            'Accept': 'application/json',
        }

        logger.debug(
            'Built Samsara request: %s %s params=%r',
            self.http_method.value,
            url,
            list(query_params.keys()),
        )

        return RequestSpec(
            url=url,
            method=self.http_method,
            headers=headers,
            query_params=query_params,
            timeout=credentials.timeout,
            max_retries=credentials.max_retries,
            retry_backoff_factor=credentials.retry_backoff_factor,
            verify_ssl=credentials.verify_ssl,
        )

    def parse_response(self, response_json: dict[str, Any]) -> ParsedResponse[ItemT]:
        """
        Parse Samsara API response into uniform ParsedResponse.

        Args:
            response_json: Raw JSON from Samsara API.

        Returns:
            ParsedResponse with extracted items and pagination state.
        """
        # Parse full response into typed model
        parsed_response: ResponseModelT = self.response_model.model_validate(
            response_json
        )

        # Extract items using the configured method
        extractor = getattr(parsed_response, self.item_extractor_method)
        items: list[ItemT] = extractor()

        # Compute pagination state
        pagination_state: PaginationState = self._compute_pagination_state(
            parsed_response
        )

        logger.debug(
            'Parsed %d items from %r, has_more=%r',
            len(items),
            self.endpoint_path,
            pagination_state.has_next_page,
        )

        return ParsedResponse(
            items=items,
            pagination=pagination_state,
            raw_response=parsed_response,
        )

    def _compute_pagination_state(
        self,
        parsed_response: ResponseModelT,
    ) -> PaginationState:
        """
        Extract pagination state from parsed Samsara response.

        Args:
            parsed_response: The fully parsed response model.

        Returns:
            PaginationState with cursor for next page if available.
        """
        if not self.is_paginated:
            return PaginationState.finished()

        # Access pagination attribute
        pagination_info: SamsaraPaginationInfo | None = getattr(
            parsed_response, 'pagination', None
        )

        if pagination_info is None:
            logger.warning(
                'Expected pagination metadata for %s but found none',
                self.endpoint_path,
            )
            return PaginationState.finished()

        if not pagination_info.has_next_page:
            return PaginationState(has_next_page=False)

        # Samsara uses 'after' parameter for cursor
        next_cursor: str | None = pagination_info.next_cursor
        if next_cursor:
            return PaginationState(
                has_next_page=True,
                next_page_params={'after': next_cursor},
            )

        return PaginationState.finished()

    def get_initial_pagination_state(self) -> PaginationState:
        """
        Get initial pagination state for Samsara endpoints.

        Samsara doesn't need any special params for first page.
        """
        if not self.is_paginated:
            return PaginationState.finished()

        return PaginationState.initial_cursor()

    def _serialize_parameter_value(
        self,
        value: Any,
        parameter_type: ParameterType,
    ) -> str:
        """
        Override serialization to enforce Samsara's strict RFC 3339 format (Z suffix).

        Standard isoformat() often produces +00:00, but Samsara explicitly
        documents 'Z' for UTC timestamps.
        """
        if parameter_type == ParameterType.DATETIME and isinstance(value, datetime):
            # Force format: 2024-01-01T00:00:00Z
            return value.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Fallback to shared serialization for other types (int, date, bool)
        return super()._serialize_parameter_value(value, parameter_type)


# =============================================================================
# Samsara Endpoint Registry
# =============================================================================


class SamsaraEndpoints:
    """
    Registry of all Samsara API endpoint definitions.

    Each endpoint is fully self-describing. Client code interacts
    with all endpoints identically through the uniform interface.

    Usage:
        >>> endpoint = SamsaraEndpoints.VEHICLES
        >>> url = endpoint.build_url("https://api.samsara.com")
        >>> query = endpoint.build_query_params()
        >>> parsed = endpoint.parse_response(response_json)
        >>> for vehicle in parsed.items:
        ...     print(vehicle.name)
    """

    VEHICLES: SamsaraEndpointDefinition[VehiclesResponse, SamsaraVehicle] = (
        SamsaraEndpointDefinition(
            endpoint_path='/fleet/vehicles',
            http_method=HTTPMethod.GET,
            description='List all vehicles in the fleet',
            is_paginated=True,
            response_model=VehiclesResponse,
            item_extractor_method='get_items',
        )
    )

    DRIVERS: SamsaraEndpointDefinition[DriversResponse, SamsaraDriver] = (
        SamsaraEndpointDefinition(
            endpoint_path='/fleet/drivers',
            http_method=HTTPMethod.GET,
            description='List drivers with optional activation status filter',
            query_parameters=(
                QueryParameterSpec(
                    name='driver_activation_status',
                    parameter_type=ParameterType.STRING,
                    required=True,
                    api_name='driverActivationStatus',
                    description='Filter by activation status (active, deactivated), Samsara defaults to active',
                ),
            ),
            is_paginated=True,
            response_model=DriversResponse,
            item_extractor_method='get_items',
        )
    )

    ADDRESSES: SamsaraEndpointDefinition[AddressesResponse, SamsaraAddress] = (
        SamsaraEndpointDefinition(
            endpoint_path='/addresses',
            http_method=HTTPMethod.GET,
            description='List all addresses/geofences',
            is_paginated=True,
            response_model=AddressesResponse,
            item_extractor_method='get_items',
        )
    )

    VEHICLE_STATS_HISTORY: SamsaraEndpointDefinition[
        VehicleStatsHistoryResponse,
        VehicleStatsHistoryRecord,
    ] = SamsaraEndpointDefinition(
        endpoint_path='/fleet/vehicles/stats/history',
        http_method=HTTPMethod.GET,
        description='Get historical vehicle stats (engine states, GPS) for time range',
        query_parameters=(
            QueryParameterSpec(
                name='start_time',
                parameter_type=ParameterType.DATETIME,
                required=True,
                api_name='startTime',
                description='Start of time range (ISO-8601 UTC)',
            ),
            QueryParameterSpec(
                name='end_time',
                parameter_type=ParameterType.DATETIME,
                required=True,
                api_name='endTime',
                description='End of time range (ISO-8601 UTC)',
            ),
            QueryParameterSpec(
                name='types',
                parameter_type=ParameterType.STRING_LIST,
                required=True,
                description='Comma-separated stat types (engineStates, gps)',
            ),
        ),
        is_paginated=True,
        response_model=VehicleStatsHistoryResponse,
        item_extractor_method='get_items',
    )

    LOCATION_STREAM: SamsaraEndpointDefinition[
        LocationStreamResponse,
        LocationStreamRecord,
    ] = SamsaraEndpointDefinition(
        endpoint_path='/assets/location-and-speed/stream',
        http_method=HTTPMethod.GET,
        description='Get high-frequency location stream for specific assets',
        query_parameters=(
            QueryParameterSpec(
                name='start_time',
                parameter_type=ParameterType.DATETIME,
                required=True,
                api_name='startTime',
                description='Start of time range (ISO-8601 UTC)',
            ),
            QueryParameterSpec(
                name='end_time',
                parameter_type=ParameterType.DATETIME,
                required=True,
                api_name='endTime',
                description='End of time range (ISO-8601 UTC)',
            ),
            QueryParameterSpec(
                name='vehicle_ids',
                parameter_type=ParameterType.STRING_LIST,
                required=True,
                api_name='ids',
                description='Comma-separated vehicle IDs',
            ),
        ),
        is_paginated=True,
        response_model=LocationStreamResponse,
        item_extractor_method='get_items',
    )

    DRIVER_VEHICLE_ASSIGNMENTS: SamsaraEndpointDefinition[
        DriverVehicleAssignmentsResponse,
        DriverVehicleAssignment,
    ] = SamsaraEndpointDefinition(
        endpoint_path='/fleet/driver-vehicle-assignments',
        http_method=HTTPMethod.GET,
        description='Get driver-vehicle assignments for a time period to correlate drivers with GPS data',
        query_parameters=(
            QueryParameterSpec(
                name='filter_by',
                parameter_type=ParameterType.STRING,
                required=True,
                api_name='filterBy',
                description='Filter mode: "vehicles" or "drivers"',
            ),
            QueryParameterSpec(
                name='start_time',
                parameter_type=ParameterType.DATETIME,
                required=True,
                api_name='startTime',
                description='Start of time range (ISO-8601 UTC)',
            ),
            QueryParameterSpec(
                name='end_time',
                parameter_type=ParameterType.DATETIME,
                required=True,
                api_name='endTime',
                description='End of time range (ISO-8601 UTC)',
            ),
            QueryParameterSpec(
                name='vehicle_ids',
                parameter_type=ParameterType.STRING_LIST,
                required=False,
                api_name='vehicleIds',
                description='Comma-separated vehicle IDs (required if filterBy=vehicles)',
            ),
            QueryParameterSpec(
                name='driver_ids',
                parameter_type=ParameterType.STRING_LIST,
                required=False,
                api_name='driverIds',
                description='Comma-separated driver IDs (required if filterBy=drivers)',
            ),
        ),
        is_paginated=True,
        response_model=DriverVehicleAssignmentsResponse,
        item_extractor_method='get_items',
    )

    @classmethod
    def get_all_endpoints(
        cls,
    ) -> dict[str, SamsaraEndpointDefinition[Any, Any]]:
        """Return all endpoint definitions as a dictionary."""
        return {
            name: value
            for name, value in vars(cls).items()
            if isinstance(value, SamsaraEndpointDefinition)
        }
