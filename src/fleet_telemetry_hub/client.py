# fleet_telemetry_hub/providers/core/client.py
"""
Provider-agnostic HTTP client.

This client executes RequestSpec objects without knowing anything about
the provider that created them. All provider-specific logic (auth, pagination)
is handled by the EndpointDefinition.
"""

import logging
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    import pandas as pd

from .models.shared_request_models import RateLimitInfo, RequestSpec
from .models.shared_response_models import (
    EndpointDefinition,
    ItemT,
    PaginationState,
    ParsedResponse,
    ProviderCredentials,
)
from .utils import build_truststore_ssl_context

logger: logging.Logger = logging.getLogger(__name__)

RATE_LIMIT_HTTP_CODE: int = 429

class APIError(Exception):
    """Base exception for API errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code: int | None = status_code
        self.response_body: str | None = response_body


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""

    def __init__(self, rate_limit_info: RateLimitInfo) -> None:
        super().__init__(
            f'Rate limit exceeded, retry after {rate_limit_info.retry_after_seconds}s',
            status_code=429,
        )
        self.rate_limit_info: RateLimitInfo = rate_limit_info


class TelemetryClient:
    """
    Provider-agnostic HTTP client for telemetry APIs.

    This client is completely decoupled from provider-specific knowledge.
    It receives RequestSpec objects (built by EndpointDefinitions) and
    executes them. The same client instance can be used for both Motive
    and Samsara endpoints.

    The client handles:
    - HTTP transport with connection pooling
    - Automatic retries with exponential backoff
    - Rate limit handling with Retry-After support
    - SSL verification (including Zscaler bypass)

    Attributes:
        credentials: Provider credentials for building request specs.

    Example:
        >>> # Works with any provider's endpoints
        >>> client = TelemetryClient(motive_credentials)
        >>> for vehicle in client.fetch_all(MotiveEndpoints.VEHICLES):
        ...     print(vehicle.number)
        >>>
        >>> client = TelemetryClient(samsara_credentials)
        >>> for vehicle in client.fetch_all(SamsaraEndpoints.VEHICLES):
        ...     print(vehicle.name)
    """

    def __init__(
        self,
        credentials: ProviderCredentials,
        pool_connections: int = 5,
        pool_maxsize: int = 10,
    ) -> None:
        """
        Initialize telemetry API client.

        Args:
            credentials: Provider credentials and connection settings.
            pool_connections: Max keepalive connections in pool.
            pool_maxsize: Max total connections in pool.
        """
        self._credentials: ProviderCredentials = credentials

        # Default timeout from credentials, can be overridden per-request
        default_timeout: tuple[int, int] = credentials.timeout

        self._http_client: httpx.Client = httpx.Client(
            timeout=httpx.Timeout(
                connect=default_timeout[0],
                read=default_timeout[1],
                write=default_timeout[0],
                pool=default_timeout[0],
            ),
            verify=credentials.verify_ssl,
            limits=httpx.Limits(
                max_keepalive_connections=pool_connections,
                max_connections=pool_maxsize,
            ),
        )

        logger.info(
            f'Initialized TelemetryClient for {credentials.base_url} '
            f'(SSL verify: {credentials.verify_ssl})'
        )

    def close(self) -> None:
        """Close HTTP client and release resources."""
        self._http_client.close()
        logger.debug('TelemetryClient closed')

    def __enter__(self) -> 'TelemetryClient':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    # -------------------------------------------------------------------------
    # Core Fetch Methods
    # -------------------------------------------------------------------------

    def fetch(
        self,
        endpoint: EndpointDefinition[Any, ItemT],
        pagination_state: PaginationState | None = None,
        **params: Any,
    ) -> ParsedResponse[ItemT]:
        """
        Fetch a single page from any endpoint.

        This method is completely endpoint-agnostic. The endpoint definition
        builds the request spec, and this client executes it.

        Args:
            endpoint: Self-describing endpoint definition.
            pagination_state: Optional pagination state for subsequent pages.
            **params: Path parameters and query parameters.

        Returns:
            ParsedResponse containing typed items and pagination state.

        Raises:
            APIError: For non-retryable API errors.
            RateLimitError: When rate limit exceeded (after retries).
        """
        # Let the endpoint build the complete request spec
        request_spec: RequestSpec = endpoint.build_request_spec(
            credentials=self._credentials,
            pagination_state=pagination_state,
            **params,
        )

        # Execute the request
        response_json: dict[str, Any] = self._execute_request(request_spec)

        # Let the endpoint parse the response
        return endpoint.parse_response(response_json)

    def fetch_all(
        self,
        endpoint: EndpointDefinition[Any, ItemT],
        request_delay_seconds: float = 0.0,
        **params: Any,
    ) -> Iterator[ItemT]:
        """
        Iterate through all items across all pages.

        Automatically handles pagination, yielding individual items
        as they are retrieved.

        Args:
            endpoint: Self-describing endpoint definition.
            request_delay_seconds: Delay between page requests (rate limiting).
            **params: Path and query parameters for the endpoint.

        Yields:
            Individual items of type ItemT from each page.
        """
        pagination_state: PaginationState | None = None
        page_count: int = 0
        total_items: int = 0

        while True:
            response: ParsedResponse[ItemT] = self.fetch(
                endpoint,
                pagination_state=pagination_state,
                **params,
            )

            page_count += 1
            total_items += response.item_count

            logger.debug(
                f'Page {page_count}: retrieved {response.item_count} items '
                f'(total: {total_items})'
            )

            yield from response.items

            if not response.has_more:
                logger.info(
                    f'Completed fetching {endpoint.endpoint_path}: '
                    f'{total_items} items across {page_count} pages'
                )
                break

            pagination_state = response.pagination

            if request_delay_seconds > 0:
                time.sleep(request_delay_seconds)

    def fetch_all_pages(
        self,
        endpoint: EndpointDefinition[Any, ItemT],
        request_delay_seconds: float = 0.0,
        **params: Any,
    ) -> Iterator[ParsedResponse[ItemT]]:
        """
        Iterate through all pages, yielding full ParsedResponse objects.

        Useful when you need pagination metadata or batch processing.

        Args:
            endpoint: Self-describing endpoint definition.
            request_delay_seconds: Delay between page requests.
            **params: Path and query parameters.

        Yields:
            ParsedResponse objects for each page.
        """
        pagination_state: PaginationState | None = None
        page_count: int = 0

        while True:
            response: ParsedResponse[ItemT] = self.fetch(
                endpoint,
                pagination_state=pagination_state,
                **params,
            )

            page_count += 1
            yield response

            if not response.has_more:
                logger.info(
                    f'Completed {page_count} pages from {endpoint.endpoint_path}'
                )
                break

            pagination_state = response.pagination

            if request_delay_seconds > 0:
                time.sleep(request_delay_seconds)

    def to_dataframe(
        self,
        endpoint: EndpointDefinition[Any, ItemT],
        request_delay_seconds: float = 0.0,
        **params: Any,
    ) -> 'pd.DataFrame':
        """
        Fetch all data from an endpoint and return as a pandas DataFrame.

        This is a convenience method that fetches all items, converts them
        from Pydantic models to dictionaries, and creates a DataFrame.

        Args:
            endpoint: Self-describing endpoint definition.
            request_delay_seconds: Delay between page requests.
            **params: Path and query parameters.

        Returns:
            pandas DataFrame containing all fetched data.

        Raises:
            ImportError: If pandas is not installed.

        Example:
            >>> from fleet_telemetry_hub import TelemetryClient, MotiveEndpoints
            >>> from fleet_telemetry_hub.models.shared_response_models import ProviderCredentials
            >>>
            >>> credentials = ProviderCredentials(...)
            >>> endpoint = MotiveEndpoints.VEHICLES
            >>>
            >>> with TelemetryClient(credentials) as client:
            ...     df = client.to_dataframe(endpoint)
            ...     print(df.head())
            ...
            ...     # Save to file
            ...     df.to_parquet("vehicles.parquet")
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                'pandas is required for to_dataframe(). '
                'Install it with: pip install pandas'
            ) from e

        # Fetch all items
        items: list[ItemT] = list(
            self.fetch_all(
                endpoint,
                request_delay_seconds=request_delay_seconds,
                **params,
            )
        )

        if not items:
            logger.warning(
                f'No items found for endpoint {endpoint.endpoint_path} '
                f'with params {params}'
            )
            return pd.DataFrame()

        # Convert Pydantic models to dictionaries
        data = [
            item.model_dump() if hasattr(item, 'model_dump') else dict(item)
            for item in items
        ]

        # Create DataFrame
        df = pd.DataFrame(data)

        logger.info(
            f'Created DataFrame from {len(items)} items from {endpoint.endpoint_path} '
            f'with {len(df.columns)} columns'
        )

        return df

    # -------------------------------------------------------------------------
    # HTTP Execution Layer
    # -------------------------------------------------------------------------

    def _execute_request(self, request_spec: RequestSpec) -> dict[str, Any]:
        """
        Execute an HTTP request from a RequestSpec.

        Args:
            request_spec: Complete request specification.

        Returns:
            Parsed JSON response as dictionary.

        Raises:
            APIError: For non-retryable errors.
            RateLimitError: When rate limit exceeded.
        """
        return self._execute_with_retry(request_spec)

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _execute_with_retry(self, request_spec: RequestSpec) -> dict[str, Any]:
        """
        Execute request with automatic retry on rate limits.

        Uses tenacity for exponential backoff retry logic.
        """
        logger.debug(
            f'{request_spec.method.value} {request_spec.url} '
            f'params={list(request_spec.query_params.keys())}'
        )

        try:
            # Build timeout from request spec
            timeout = httpx.Timeout(
                connect=request_spec.timeout[0],
                read=request_spec.timeout[1],
                write=request_spec.timeout[0],
                pool=request_spec.timeout[0],
            )

            response: httpx.Response = self._http_client.request(
                method=request_spec.method.value,
                url=request_spec.url,
                params=request_spec.query_params,
                headers=request_spec.headers,
                json=request_spec.body,
                timeout=timeout,
            )
        except httpx.TimeoutException as error:
            logger.error(f'Request timeout: {request_spec.url}')
            raise APIError(f'Request timeout: {error}') from error
        except httpx.RequestError as error:
            logger.error(f'Request failed: {request_spec.url} - {error}')
            raise APIError(f'Request failed: {error}') from error

        # Handle rate limiting
        if response.status_code == RATE_LIMIT_HTTP_CODE:
            rate_limit_info: RateLimitInfo = RateLimitInfo.from_response_headers(
                dict(response.headers)
            )
            logger.warning(
                f'Rate limited, retry after {rate_limit_info.retry_after_seconds}s '
                f'(remaining: {rate_limit_info.remaining})'
            )
            raise RateLimitError(rate_limit_info)

        # Handle other errors
        if not response.is_success:
            logger.error(f'API error {response.status_code}: {response.text[:500]}')
            raise APIError(
                message=f'API returned {response.status_code}',
                status_code=response.status_code,
                response_body=response.text,
            )

        return response.json()
