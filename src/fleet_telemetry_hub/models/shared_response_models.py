# fleet_telemetry_hub/models/shared_response_models.py
"""
Shared abstractions for API endpoint definitions.

This module provides the foundational building blocks for the "Active Endpoint"
architecture. It contains the generic types, parameter specifications, and
abstract base classes that enable uniform interaction with diverse APIs
(Motive, Samsara) without code duplication.

Key Components:
    - PaginationState: A provider-agnostic container for "next page" logic.
    - EndpointDefinition: The abstract base class that all provider-specific
      definitions must inherit from.
    - ParameterType: Enumerations for strict type handling in URLs.
"""

import logging
from abc import ABC, abstractmethod
from datetime import date, datetime
from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from .shared_request_models import HTTPMethod, RequestSpec

logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# Type Variables for Generic Response/Item Typing
# =============================================================================

# The full response model type (e.g., VehiclesResponse)
ResponseModelT = TypeVar('ResponseModelT', bound=BaseModel)

# The individual item type extracted from responses (e.g., Vehicle)
ItemT = TypeVar('ItemT', bound=BaseModel)

# =============================================================================
# Parameter Specifications
# =============================================================================

class ParameterType(str, Enum):
    """Supported parameter types for automatic serialization."""

    STRING = 'string'
    INTEGER = 'integer'
    FLOAT = 'float'
    DATE = 'date'
    DATETIME = 'datetime'
    BOOLEAN = 'boolean'
    STRING_LIST = 'string_list'


class PathParameterSpec(BaseModel):
    """Specification for a URL path parameter placeholder."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    name: str
    parameter_type: ParameterType
    description: str = ''


class QueryParameterSpec(BaseModel):
    """Specification for a URL query parameter."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    name: str
    parameter_type: ParameterType
    required: bool = False
    description: str = ''
    api_name: str | None = Field(
        default=None,
        description='Parameter name sent to API if different from internal name',
    )

    def get_api_parameter_name(self) -> str:
        """Return the parameter name to use in API requests."""
        return self.api_name if self.api_name is not None else self.name


# =============================================================================
# Pagination State Container
# =============================================================================


class PaginationState(BaseModel):
    """
    Encapsulates pagination state for iteration.

    This container holds everything needed to request the next page,
    abstracting away provider-specific pagination mechanics. It supports
    both Offset-based (Motive) and Cursor-based (Samsara) patterns via
    the flexible `next_page_params` dictionary.

    Attributes:
        has_next_page: Whether more data is available.
        next_page_params: Query parameters to add for the next request.
                          Motive: {'page_no': 2, 'per_page': 100}
                          Samsara: {'after': 'endCursorString...'}
        total_items: Total item count (if known, typical for Offset).
        current_page: Current page number (if applicable, typical for Offset).
        current_cursor: Current cursor string (if applicable, typical for Cursor).
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    has_next_page: bool
    next_page_params: dict[str, int | str] = Field(default_factory=dict)
    total_items: int | None = None
    current_page: int | None = None
    current_cursor: str | None = None

    @classmethod
    def finished(cls) -> 'PaginationState':
        """Factory for terminal pagination state (no more pages)."""
        return cls(has_next_page=False)

    @classmethod
    def first_page(cls, per_page: int = 100) -> 'PaginationState':
        """Factory for initial pagination state."""
        return cls(
            has_next_page=True,
            next_page_params={'page_no': 1, 'per_page': per_page},
            current_page=0,
        )

    @classmethod
    def initial_cursor(cls) -> 'PaginationState':
        """Factory for initial Cursor-based pagination (Samsara)."""
        # Samsara requires NO 'after' param for the first page
        return cls(has_next_page=True, next_page_params={}, current_cursor=None)

    @classmethod
    def next_cursor(cls, cursor_token: str) -> 'PaginationState':
        """Factory for subsequent Cursor-based pages (Samsara)."""
        return cls(
            has_next_page=True,
            next_page_params={'after': cursor_token},
            current_cursor=cursor_token,
        )


# =============================================================================
# Parsed Response Container
# =============================================================================


class ParsedResponse(BaseModel, Generic[ItemT]):
    """
    Uniform container for parsed API responses.

    This is what the client receives after calling endpoint.parse_response().
    It provides a consistent interface regardless of the underlying endpoint.

    Attributes:
        items: Extracted data items (e.g., list of Vehicle objects).
        pagination: Current pagination state for fetching more data.
        raw_response: Original parsed response model (for advanced use).

    Type Parameters:
        ItemT: The type of individual items (e.g., Vehicle, User).
    """

    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)

    items: list[ItemT]
    pagination: PaginationState
    raw_response: BaseModel | None = Field(default=None, exclude=True)

    @property
    def item_count(self) -> int:
        """Number of items in this response page."""
        return len(self.items)

    @property
    def has_more(self) -> bool:
        """Whether more pages are available."""
        return self.pagination.has_next_page

# =============================================================================
# Provider Configuration Protocol
# =============================================================================


class ProviderCredentials(BaseModel):
    """
    Credentials and connection settings for a provider.

    This is passed to build_request_spec() to inject authentication
    and connection configuration into the request.

    Attributes:
        base_url: API base URL (no trailing slash).
        api_key: API key or token (SecretStr for security).
        timeout: Tuple of (connect_timeout, read_timeout) in seconds.
        max_retries: Maximum retry attempts.
        retry_backoff_factor: Exponential backoff multiplier.
        verify_ssl: SSL verification (bool or path to CA bundle).
        request_delay: delay between requests in seconds.
        use_truststore: Use the Windows credential manager.
    """

    model_config = ConfigDict(extra='forbid')

    base_url: str
    api_key: SecretStr
    timeout: tuple[int, int] = (30, 120)
    max_retries: int = 3
    retry_backoff_factor: float = 1.0
    verify_ssl: bool | str = True
    request_delay: float = 0.0
    use_truststore: bool = False


# =============================================================================
# Abstract Endpoint Definition
# =============================================================================


class EndpointDefinition(ABC, BaseModel, Generic[ResponseModelT, ItemT]):
    """
    Abstract base for self-describing API endpoints.

    An EndpointDefinition encapsulates everything needed to interact with
    an API endpoint through a uniform interface:

    - URL construction (path + parameters)
    - Query parameter serialization
    - Response parsing into typed models
    - Data extraction from nested response structures
    - Pagination state management

    The client interacts with all endpoints identically:

        endpoint = MotiveEndpoints.VEHICLES
        url = endpoint.build_url(base_url, **params)
        query = endpoint.build_query_params(**params)
        parsed = endpoint.parse_response(json_data)
        for item in parsed.items:
            ...

    Type Parameters:
        ResponseModelT: The Pydantic model for the full API response.
        ItemT: The Pydantic model for individual data items.

    Attributes:
        endpoint_path: Relative URL path (may contain {placeholders}).
        http_method: HTTP verb for requests.
        description: Human-readable endpoint description.
        path_parameters: Specifications for path placeholders.
        query_parameters: Specifications for query string parameters.
        is_paginated: Whether endpoint returns paginated results.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    endpoint_path: str

    http_method: HTTPMethod = HTTPMethod.GET
    description: str

    path_parameters: tuple[PathParameterSpec, ...] = Field(default_factory=tuple)
    query_parameters: tuple[QueryParameterSpec, ...] = Field(default_factory=tuple)

    is_paginated: bool

    @field_validator('endpoint_path')
    @classmethod
    def validate_endpoint_path_format(cls, endpoint_path: str) -> str:
        """Ensure endpoint path starts with forward slash."""
        if not endpoint_path:
            raise ValueError('endpoint_path cannot be empty')
        if not endpoint_path.startswith('/'):
            endpoint_path = f'/{endpoint_path}'
        return endpoint_path

    @field_validator('path_parameters', mode='before')
    @classmethod
    def convert_path_params_to_tuple(
        cls,
        value: list[PathParameterSpec] | tuple[PathParameterSpec],
    ) -> tuple[PathParameterSpec, ...]:
        """Convert list input to immutable tuple."""
        return tuple(value)

    @field_validator('query_parameters', mode='before')
    @classmethod
    def convert_query_params_to_tuple(
        cls,
        value: list[QueryParameterSpec] | tuple[QueryParameterSpec],
    ) -> tuple[QueryParameterSpec, ...]:
        """Convert list input to immutable tuple."""
        return tuple(value)

    # -------------------------------------------------------------------------
    # URL Construction (Uniform Interface)
    # -------------------------------------------------------------------------

    def build_url(self, base_url: str, **path_params: Any) -> str:
        """
        Construct the full URL for this endpoint.

        Args:
            base_url: API base URL (e.g., "https://api.gomotive.com").
            **path_params: Values for path parameter placeholders.

        Returns:
            Complete URL ready for HTTP request.

        Raises:
            ValueError: If required path parameters are missing.

        Example:
            >>> endpoint.build_url(
            ...     "https://api.gomotive.com",
            ...     vehicle_id=543179
            ... )
            'https://api.gomotive.com/v3/vehicle_locations/543179'
        """
        resolved_path: str = self.endpoint_path

        for param_spec in self.path_parameters:
            placeholder: str = f'{{{param_spec.name}}}'

            if param_spec.name not in path_params:
                raise ValueError(f'Missing required path parameter: {param_spec.name}')

            value = path_params[param_spec.name]
            serialized_value: str = self._serialize_parameter_value(
                value, param_spec.parameter_type
            )
            resolved_path = resolved_path.replace(placeholder, serialized_value)

        # base_url should not have trailing slash (enforced in config)
        return f'{base_url}{resolved_path}'

    def build_query_params(
        self,
        pagination_state: PaginationState | None = None,
        **user_params: Any,
    ) -> dict[str, str]:
        """
        Build query parameter dictionary for this endpoint.

        Combines user-provided parameters with pagination parameters,
        serializing all values to strings for HTTP transmission.

        Args:
            pagination_state: Optional pagination state with next page params.
            **user_params: User-provided query parameter values.

        Returns:
            Dictionary of query parameters (all values as strings).

        Raises:
            ValueError: If required query parameters are missing.

        Example:
            >>> endpoint.build_query_params(
            ...     start_date=date(2025, 1, 1),
            ...     end_date=date(2025, 1, 31)
            ... )
            {'start_date': '2025-01-01', 'end_date': '2025-01-31'}
        """
        query_params: dict[str, str] = {}

        # Add pagination parameters first (if provided)
        if pagination_state is not None:
            for key, value in pagination_state.next_page_params.items():
                query_params[key] = str(value)

        # Process defined query parameters
        for param_spec in self.query_parameters:
            param_name: str = param_spec.name
            api_name: str = param_spec.get_api_parameter_name()

            if param_name in user_params:
                value = user_params[param_name]
                if value is not None:
                    query_params[api_name] = self._serialize_parameter_value(
                        value, param_spec.parameter_type
                    )
            elif param_spec.required:
                raise ValueError(f'Missing required query parameter: {param_name}')

        return query_params

    def _serialize_parameter_value(
        self,
        value: Any,
        parameter_type: ParameterType,
    ) -> str:
        """
        Serialize a parameter value to string for HTTP transmission.

        Args:
            value: The value to serialize.
            parameter_type: Expected type for format selection.

        Returns:
            String representation suitable for URL/query string.
        """
        formatted_value: str
        if parameter_type == ParameterType.DATE:
            if isinstance(value, date):
                formatted_value = value.isoformat()
            formatted_value = str(value)

        if parameter_type == ParameterType.DATETIME:
            if isinstance(value, datetime):
                formatted_value = value.isoformat()
            formatted_value = str(value)

        if parameter_type == ParameterType.BOOLEAN:
            formatted_value = 'true' if value else 'false'

        if parameter_type == ParameterType.STRING_LIST:
            if isinstance(value, (list, tuple)):
                formatted_value = ','.join(str(v) for v in value)  # pyright: ignore[reportUnknownVariableType]
            formatted_value = str(value)

        formatted_value = str(value)

        return formatted_value

    # -------------------------------------------------------------------------
    # Response Parsing (Abstract - Provider-Specific)
    # -------------------------------------------------------------------------

    @abstractmethod
    def build_request_spec(
        self,
        credentials: ProviderCredentials,
        pagination_state: PaginationState | None = None,
        **params: Any,
    ) -> RequestSpec:
        """
        Build a complete request specification ready for HTTP execution.

        This method produces a fully-formed RequestSpec that the client
        can execute without knowing provider-specific details like auth
        patterns or pagination mechanics.

        Args:
            credentials: Provider credentials and connection settings.
            pagination_state: Optional pagination state for subsequent pages.
            **params: Path parameters and query parameters combined.

        Returns:
            RequestSpec ready for HTTP client execution.
        """
        raise NotImplementedError('Subclasses must implement build_request_spec')

    @abstractmethod
    def parse_response(self, response_json: dict[str, Any]) -> ParsedResponse[ItemT]:
        """
        Parse raw JSON response into typed ParsedResponse.

        This method handles:
        1. Validating JSON against the response model
        2. Extracting data items from nested structures
        3. Computing pagination state for next request

        Args:
            response_json: Raw JSON dictionary from HTTP response.

        Returns:
            ParsedResponse containing typed items and pagination state.

        Raises:
            ValidationError: If response doesn't match expected schema.
        """
        raise NotImplementedError('Subclasses must implement parse_response')

    @abstractmethod
    def get_initial_pagination_state(self) -> PaginationState:
        """
        Get the initial pagination state for first request.

        For paginated endpoints, returns state with first-page parameters.
        For non-paginated endpoints, returns finished state.

        Returns:
            PaginationState for initiating requests.
        """
        raise NotImplementedError(
            'Subclasses must implement get_initial_pagination_state'
        )
