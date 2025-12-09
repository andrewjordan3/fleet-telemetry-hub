# fleet_telemetry_hub/models/motive_requests.py
"""
Unified endpoint definitions for Motive API.

This module provides self-describing endpoint objects that encapsulate all
knowledge required to interact with an endpoint: URL construction, request
parameters, response parsing, data extraction, and pagination handling.

The client code receives an EndpointDefinition and interacts with it through
a uniform interface, never needing to know endpoint-specific details.
"""

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from fleet_telemetry_hub.models.motive_responses import (
    Group,
    GroupsResponse,
    MotivePaginationInfo,
    ResponseModelBase,
    User,
    UsersResponse,
    Vehicle,
    VehicleLocation,
    VehicleLocationsResponse,
    VehiclesResponse,
)
from fleet_telemetry_hub.models.shared_request_models import HTTPMethod, RequestSpec
from fleet_telemetry_hub.models.shared_response_models import (
    EndpointDefinition,
    PaginationState,
    ParameterType,
    ParsedResponse,
    PathParameterSpec,
    ProviderCredentials,
    QueryParameterSpec,
)

logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# Motive-Specific Endpoint Definition
# =============================================================================


class MotiveEndpointDefinition[ResponseModelT: BaseModel, ItemT: ResponseModelBase](
    EndpointDefinition[ItemT],
):
    """
    Motive-specific endpoint definition with page-number pagination.

    Motive uses offset-based pagination with 'page_no' and 'per_page'
    query parameters. Responses include pagination metadata with total
    count for progress tracking.

    Attributes:
        response_model: Pydantic model class for parsing full response.
        item_extractor_method: Name of method on response model that extracts items.
        max_per_page: Maximum results per page (Motive caps at 100).
    """

    model_config = ConfigDict(extra='forbid', frozen=True, arbitrary_types_allowed=True)

    response_model: type[ResponseModelT]
    item_extractor_method: str = Field(
        description='Method name on response model to extract items (e.g., "get_vehicles")'
    )
    max_per_page: int = Field(default=100, ge=1, le=100)

    def build_request_spec(
        self,
        credentials: ProviderCredentials,
        pagination_state: PaginationState | None = None,
        **params: Any,
    ) -> RequestSpec:
        """
        Build complete request specification for Motive API.

        Injects X-API-Key authentication header and handles page-number
        based pagination parameters.

        Args:
            credentials: Motive API credentials and connection settings.
            pagination_state: Optional pagination state with page number.
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

        # Build Motive-specific headers
        headers: dict[str, str] = {
            'X-API-Key': credentials.api_key.get_secret_value(),
            'Accept': 'application/json',
        }

        logger.debug(
            'Built Motive request: %s %s params=%r',
            self.http_method.value,
            url,
            query_params,
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
        Parse Motive API response into uniform ParsedResponse.

        Args:
            response_json: Raw JSON from Motive API.

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
        Extract pagination state from parsed Motive response.

        Args:
            parsed_response: The fully parsed response model.

        Returns:
            PaginationState reflecting current position and next page params.
        """
        if not self.is_paginated:
            return PaginationState.finished()

        # Access pagination attribute (all paginated Motive responses have this)
        pagination_info: MotivePaginationInfo | None = getattr(
            parsed_response, 'pagination', None
        )

        if pagination_info is None:
            logger.warning(
                'Expected pagination metadata for %r but found none',
                self.endpoint_path,
            )
            return PaginationState.finished()

        if not pagination_info.has_next_page:
            return PaginationState(
                has_next_page=False,
                total_items=pagination_info.total,
                current_page=pagination_info.page_no,
            )

        return PaginationState(
            has_next_page=True,
            next_page_params={
                'page_no': pagination_info.page_no + 1,
                'per_page': pagination_info.per_page,
            },
            total_items=pagination_info.total,
            current_page=pagination_info.page_no,
        )

    def get_initial_pagination_state(self) -> PaginationState:
        """
        Get initial pagination state for Motive endpoints.

        Returns:
            PaginationState with page_no=1 for paginated endpoints,
            or finished state for non-paginated endpoints.
        """
        if not self.is_paginated:
            return PaginationState.finished()

        return PaginationState.first_page(per_page=self.max_per_page)


# =============================================================================
# Motive Endpoint Registry
# =============================================================================


class MotiveEndpoints:
    """
    Registry of all Motive API endpoint definitions.

    Each endpoint is a fully self-describing object. The client code
    interacts with all endpoints identically through the uniform interface
    defined by EndpointDefinition.

    Usage:
        >>> endpoint = MotiveEndpoints.VEHICLES
        >>> url = endpoint.build_url("https://api.gomotive.com")
        >>> query = endpoint.build_query_params()
        >>> # ... make HTTP request ...
        >>> parsed = endpoint.parse_response(response_json)
        >>> for vehicle in parsed.items:
        ...     print(vehicle.number)
    """

    VEHICLES: MotiveEndpointDefinition[VehiclesResponse, Vehicle] = (
        MotiveEndpointDefinition(
            endpoint_path='/v1/vehicles',
            http_method=HTTPMethod.GET,
            description='List all vehicles in the fleet with current driver and device info',
            is_paginated=True,
            response_model=VehiclesResponse,
            item_extractor_method='get_vehicles',
            max_per_page=100,
        )
    )

    VEHICLE_LOCATIONS: MotiveEndpointDefinition[
        VehicleLocationsResponse, VehicleLocation
    ] = MotiveEndpointDefinition(
        endpoint_path='/v3/vehicle_locations/{vehicle_id}',
        http_method=HTTPMethod.GET,
        description='Get location history (breadcrumbs) for a specific vehicle',
        path_parameters=(
            PathParameterSpec(
                name='vehicle_id',
                parameter_type=ParameterType.INTEGER,
                description='Motive internal vehicle ID',
            ),
        ),
        query_parameters=(
            QueryParameterSpec(
                name='start_date',
                parameter_type=ParameterType.DATE,
                required=False,
                description='Start of date range (YYYY-MM-DD)',
            ),
            QueryParameterSpec(
                name='end_date',
                parameter_type=ParameterType.DATE,
                required=False,
                description='End of date range (YYYY-MM-DD)',
            ),
            QueryParameterSpec(
                name='updated_after',
                parameter_type=ParameterType.DATETIME,
                required=False,
                description='Filter to locations updated after this timestamp',
            ),
        ),
        is_paginated=False,
        response_model=VehicleLocationsResponse,
        item_extractor_method='get_locations',
    )

    GROUPS: MotiveEndpointDefinition[GroupsResponse, Group] = MotiveEndpointDefinition(
        endpoint_path='/v1/groups',
        http_method=HTTPMethod.GET,
        description='List all groups (organizational units) in the company',
        is_paginated=True,
        response_model=GroupsResponse,
        item_extractor_method='get_groups',
        max_per_page=100,
    )

    USERS: MotiveEndpointDefinition[UsersResponse, User] = MotiveEndpointDefinition(
        endpoint_path='/v1/users',
        http_method=HTTPMethod.GET,
        description='List all users (drivers and admins) in the company',
        is_paginated=True,
        response_model=UsersResponse,
        item_extractor_method='get_users',
        max_per_page=100,
    )

    @classmethod
    def get_all_endpoints(
        cls,
    ) -> dict[str, MotiveEndpointDefinition[ResponseModelBase, ResponseModelBase]]:
        """Return all endpoint definitions as a dictionary."""
        return {
            name: value
            for name, value in vars(cls).items()
            if isinstance(value, MotiveEndpointDefinition)
        }
