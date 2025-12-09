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
from collections.abc import Callable
from datetime import date, datetime
from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from fleet_telemetry_hub.models.shared_request_models import HTTPMethod, RequestSpec

logger: logging.Logger = logging.getLogger(__name__)


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
    def finished(cls) -> Self:
        """Factory for terminal pagination state (no more pages)."""
        return cls(has_next_page=False)

    @classmethod
    def first_page(cls, per_page: int = 100) -> Self:
        """Factory for initial pagination state."""
        return cls(
            has_next_page=True,
            next_page_params={'page_no': 1, 'per_page': per_page},
            current_page=0,
        )

    @classmethod
    def initial_cursor(cls) -> Self:
        """Factory for initial Cursor-based pagination (Samsara)."""
        # Samsara requires NO 'after' param for the first page
        return cls(has_next_page=True, next_page_params={}, current_cursor=None)

    @classmethod
    def next_cursor(cls, cursor_token: str) -> Self:
        """Factory for subsequent Cursor-based pages (Samsara)."""
        return cls(
            has_next_page=True,
            next_page_params={'after': cursor_token},
            current_cursor=cursor_token,
        )


# =============================================================================
# Parsed Response Container
# =============================================================================


class ParsedResponse[ItemT: BaseModel](BaseModel):
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


class EndpointDefinition[ItemT: BaseModel](ABC, BaseModel):
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

            value: Any = path_params[param_spec.name]
            serialized_value: str = self._serialize_parameter_value(
                value, param_spec.parameter_type
            )
            resolved_path = resolved_path.replace(placeholder, serialized_value)

        # base_url should not have trailing slash (enforced in config)
        return f'{base_url}{resolved_path}'

    def build_resource_path(self, **path_params: Any) -> str:
        """
        Construct the relative resource path with placeholders replaced.

        This method is distinct from build_url because it returns ONLY the path
        portion (e.g., '/v1/vehicles/55') without the base URL. This is critical
        for structured logging where you want to identify the resource being
        accessed without the noise of the full absolute URL.

        Args:
            **path_params: Values for path parameter placeholders.

        Returns:
            str: The relative path with actual values substituted.

        Raises:
            ValueError: If required path parameters are missing.
        """
        resolved_path: str = self.endpoint_path

        for param_spec in self.path_parameters:
            placeholder: str = f'{{{param_spec.name}}}'

            # Strict check: You cannot log a path if you don't have the ID
            if param_spec.name not in path_params:
                raise ValueError(f'Missing required path parameter: {param_spec.name}')

            value: Any = path_params[param_spec.name]

            # Use the existing serializer to ensure consistent formatting
            # (e.g., Dates become ISO strings)
            serialized_value: str = self._serialize_parameter_value(
                value, param_spec.parameter_type
            )
            resolved_path = resolved_path.replace(placeholder, serialized_value)

        return resolved_path

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
                param_value: Any = user_params[param_name]
                if param_value is not None:
                    query_params[api_name] = self._serialize_parameter_value(
                        param_value, param_spec.parameter_type
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
        # Dispatch table maps parameter types to handler methods
        handler_map: dict[ParameterType, Callable[[Any], str]] = {
            ParameterType.DATE: self._serialize_date,
            ParameterType.DATETIME: self._serialize_datetime,
            ParameterType.BOOLEAN: self._serialize_boolean,
            ParameterType.STRING_LIST: self._serialize_string_list,
        }

        # Use handler if registered, otherwise default to string conversion
        handler: Callable[[Any], str] = handler_map.get(parameter_type, str)
        formatted_value: str = handler(value)

        return formatted_value

    def _serialize_date(self, value: Any) -> str:
        """Serialize date values to ISO format string."""
        return value.isoformat() if isinstance(value, date) else str(value)

    def _serialize_datetime(self, value: Any) -> str:
        """Serialize datetime values to ISO format string."""
        return value.isoformat() if isinstance(value, datetime) else str(value)

    def _serialize_boolean(self, value: Any) -> str:
        """Serialize boolean values to lowercase string literals."""
        return 'true' if value else 'false'

    def _serialize_string_list(self, value: Any) -> str:
        """Serialize list/tuple values to comma-separated string."""
        if isinstance(value, (list, tuple)):
            return ','.join(str(v) for v in value)  # pyright: ignore[reportUnknownVariableType]
        return str(value)

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
