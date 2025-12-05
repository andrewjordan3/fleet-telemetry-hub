# fleet_telemetry_hub/client.py
"""
Provider-agnostic HTTP client for telemetry APIs.

This client executes RequestSpec objects without knowing anything about
the provider that created them. All provider-specific logic (authentication,
pagination, response parsing) is handled by the EndpointDefinition.

Retry Behavior:
---------------
The client automatically retries requests on transient failures:
- Rate limits (429): Respects Retry-After header, falls back to exponential backoff
- Server errors (5xx): Exponential backoff
- Timeouts: Exponential backoff
- Connection errors: Exponential backoff

Non-retryable errors (4xx except 429) fail immediately.

SSL/TLS Handling:
-----------------
Supports three SSL verification modes via ProviderCredentials:
- Standard verification (verify_ssl=True)
- Disabled verification (verify_ssl=False) for development
- Custom CA bundle (verify_ssl='/path/to/cert.pem') for Zscaler environments
- Truststore integration (use_truststore=True) for Windows system CA store
"""

import logging
import time
from collections.abc import Iterator
from ssl import SSLContext
from types import TracebackType
from typing import Any, Final, Self, cast

import httpx
import pandas as pd
from pydantic import BaseModel
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
)

from fleet_telemetry_hub.common import build_truststore_ssl_context
from fleet_telemetry_hub.models import (
    EndpointDefinition,
    PaginationState,
    ParsedResponse,
    ProviderCredentials,
    RateLimitInfo,
    RequestSpec,
)

__all__: list[str] = [
    'APIError',
    'RateLimitError',
    'TelemetryClient',
    'TransientAPIError',
]

logger: logging.Logger = logging.getLogger(__name__)

# HTTP status codes
HTTP_STATUS_RATE_LIMITED: Final[int] = 429
HTTP_STATUS_SERVER_ERROR_MIN: Final[int] = 500
HTTP_STATUS_SERVER_ERROR_MAX: Final[int] = 599

# Retry configuration
MAX_RETRY_ATTEMPTS: Final[int] = 5
RETRY_BACKOFF_MULTIPLIER: Final[float] = 1.0
RETRY_BACKOFF_MIN_SECONDS: Final[float] = 1.0
RETRY_BACKOFF_MAX_SECONDS: Final[float] = 60.0


# =============================================================================
# Exception Hierarchy
# =============================================================================


class APIError(Exception):
    """
    Base exception for API errors.

    This is the root of the API error hierarchy. Catch this to handle all
    API-related failures. For more specific handling, catch subclasses.

    Attributes:
        status_code: HTTP status code if available, None for connection errors.
        response_body: Raw response body for debugging, None if unavailable.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code: int | None = status_code
        self.response_body: str | None = response_body


class TransientAPIError(APIError):
    """
    Raised for transient errors that should be retried.

    This includes timeouts, connection errors, and server errors (5xx).
    The retry decorator catches this exception type for automatic retry.
    """

    pass


class RateLimitError(TransientAPIError):
    """
    Raised when API rate limit is exceeded (HTTP 429).

    Contains rate limit metadata from response headers, including the
    server-suggested retry delay.

    Attributes:
        rate_limit_info: Parsed rate limit headers including retry_after_seconds.
    """

    def __init__(self, rate_limit_info: RateLimitInfo) -> None:
        super().__init__(
            f'Rate limit exceeded, retry after {rate_limit_info.retry_after_seconds}s',
            status_code=HTTP_STATUS_RATE_LIMITED,
        )
        self.rate_limit_info: RateLimitInfo = rate_limit_info


# =============================================================================
# Custom Wait Strategy for Rate Limits
# =============================================================================


def _wait_for_rate_limit_or_exponential(retry_state: RetryCallState) -> float:
    """
    Custom wait strategy that respects Retry-After header for rate limits.

    If the exception is a RateLimitError with a retry_after_seconds value,
    use that. Otherwise, fall back to exponential backoff.

    Args:
        retry_state: Tenacity retry state containing exception info.

    Returns:
        Number of seconds to wait before next retry attempt.
    """
    exception: BaseException | None = (
        retry_state.outcome.exception() if retry_state.outcome else None
    )

    # If it's a rate limit error, use the server's suggested wait time
    if isinstance(exception, RateLimitError):
        wait_seconds: float = exception.rate_limit_info.retry_after_seconds
        # Add small buffer to avoid hitting the limit again immediately
        return wait_seconds + 0.5

    # Fall back to exponential backoff for other transient errors
    attempt_number: int = retry_state.attempt_number
    exponential_wait: float = RETRY_BACKOFF_MULTIPLIER * (2 ** (attempt_number - 1))
    return min(exponential_wait, RETRY_BACKOFF_MAX_SECONDS)


# =============================================================================
# HTTP Client
# =============================================================================


class TelemetryClient:
    """
    Provider-agnostic HTTP client for telemetry APIs.

    This client is completely decoupled from provider-specific knowledge.
    It receives RequestSpec objects (built by EndpointDefinitions) and
    executes them. The same client instance can be used for any provider's
    endpoints as long as the credentials match.

    The client handles:
    - HTTP transport with connection pooling
    - Automatic retries with exponential backoff for transient errors
    - Rate limit handling that respects Retry-After headers
    - SSL verification (including Zscaler/truststore bypass)

    Thread Safety:
        The underlying httpx.Client is thread-safe for concurrent requests.
        However, this class is designed for single-threaded use. For concurrent
        fetching, create separate TelemetryClient instances per thread.

    Example:
        >>> credentials = ProviderCredentials(...)
        >>> with TelemetryClient(credentials) as client:
        ...     for vehicle in client.fetch_all(MotiveEndpoints.VEHICLES):
        ...         print(vehicle.number)
    """

    def __init__(
        self,
        credentials: ProviderCredentials,
        pool_connections: int = 5,
        pool_maxsize: int = 10,
        request_delay_seconds: float = 0.0,
    ) -> None:
        """
        Initialize telemetry API client.

        Args:
            credentials: Provider credentials including base URL, API key,
                timeout settings, and SSL configuration.
            pool_connections: Maximum number of keepalive connections to
                maintain in the connection pool.
            pool_maxsize: Maximum total connections allowed in the pool.
            request_delay_seconds: Default delay between sequential requests.
                Can be overridden per-method call. Use to stay under rate limits.

        Raises:
            OSError: If truststore SSLContext cannot be built (when use_truststore=True).
        """
        self._credentials: ProviderCredentials = credentials
        self._request_delay_seconds: float = request_delay_seconds

        # Build SSL verification context
        ssl_verify: SSLContext | bool | str = self._build_ssl_context()

        # Build default timeout from credentials
        connect_timeout: int
        read_timeout: int
        connect_timeout, read_timeout = credentials.timeout
        default_timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=connect_timeout,
            pool=connect_timeout,
        )

        # Initialize HTTP client with connection pooling
        self._http_client: httpx.Client = httpx.Client(
            timeout=default_timeout,
            verify=ssl_verify,
            limits=httpx.Limits(
                max_keepalive_connections=pool_connections,
                max_connections=pool_maxsize,
            ),
        )

        logger.info(
            'Initialized TelemetryClient: base_url=%r, pool_size=%d',
            credentials.base_url,
            pool_maxsize,
        )

    def _build_ssl_context(self) -> SSLContext | bool | str:
        """
        Build SSL verification context from credentials.

        Returns:
            SSLContext for truststore, bool for enable/disable, or str path to CA bundle.
        """
        if self._credentials.use_truststore:
            logger.debug('Building SSLContext from truststore (Windows system CA)')
            return build_truststore_ssl_context()

        logger.debug('Using SSL verification setting: %r', self._credentials.verify_ssl)
        return self._credentials.verify_ssl

    def _get_effective_delay(self, override: float | None) -> float:
        """
        Get the effective request delay, preferring override if provided.

        Args:
            override: Caller-specified delay, or None to use instance default.

        Returns:
            Delay in seconds to use between requests.
        """
        return self._request_delay_seconds if override is None else override

    # -------------------------------------------------------------------------
    # Context Manager Protocol
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """
        Close HTTP client and release connection pool resources.

        Safe to call multiple times. After closing, the client cannot be used
        for further requests.
        """
        self._http_client.close()
        logger.debug('TelemetryClient closed')

    def __enter__(self) -> Self:
        """Enter context manager, returning self."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager, closing the HTTP client."""
        self.close()

    # -------------------------------------------------------------------------
    # Core Fetch Methods
    # -------------------------------------------------------------------------

    def fetch[ItemT: BaseModel](
        self,
        endpoint: EndpointDefinition[ItemT],
        pagination_state: PaginationState | None = None,
        **params: Any,
    ) -> ParsedResponse[ItemT]:
        """
        Fetch a single page from any endpoint.

        This method is completely endpoint-agnostic. The endpoint definition
        builds the request spec (URL, headers, auth), and this client executes it.

        Args:
            endpoint: Self-describing endpoint definition that knows how to
                build requests and parse responses.
            pagination_state: Optional pagination state from a previous response
                for fetching subsequent pages. None for the first page.
            **params: Path parameters (e.g., vehicle_id) and query parameters
                (e.g., start_date, end_date) for the endpoint.

        Returns:
            ParsedResponse containing typed items, pagination state for the
            next page (if any), and metadata.

        Raises:
            APIError: For non-retryable API errors (4xx except 429).
            TransientAPIError: After exhausting retries on transient errors.
            RateLimitError: After exhausting retries on rate limits.
        """
        # Delegate request building to the endpoint definition
        request_spec: RequestSpec = endpoint.build_request_spec(
            credentials=self._credentials,
            pagination_state=pagination_state,
            **params,
        )

        # Execute with retry logic
        response_json: dict[str, Any] = self._execute_request(request_spec)

        # Delegate response parsing to the endpoint definition
        return endpoint.parse_response(response_json)

    def fetch_all[ItemT: BaseModel](
        self,
        endpoint: EndpointDefinition[ItemT],
        request_delay_seconds: float | None = None,
        **params: Any,
    ) -> Iterator[ItemT]:
        """
        Iterate through all items across all pages.

        Automatically handles pagination, yielding individual items as they
        are retrieved. Memory-efficient for large result sets since items
        are yielded one at a time rather than collected into a list.

        Args:
            endpoint: Self-describing endpoint definition.
            request_delay_seconds: Delay between page requests. None uses
                the instance default.
            **params: Path and query parameters for the endpoint.

        Yields:
            Individual items of type ItemT from each page.

        Raises:
            APIError: For non-retryable API errors.
            TransientAPIError: After exhausting retries on transient errors.
        """
        for page_response in self._paginate(endpoint, request_delay_seconds, **params):
            yield from page_response.items

    def fetch_all_pages[ItemT: BaseModel](
        self,
        endpoint: EndpointDefinition[ItemT],
        request_delay_seconds: float | None = None,
        **params: Any,
    ) -> Iterator[ParsedResponse[ItemT]]:
        """
        Iterate through all pages, yielding full ParsedResponse objects.

        Useful when you need access to pagination metadata, item counts,
        or want to process items in page-sized batches.

        Args:
            endpoint: Self-describing endpoint definition.
            request_delay_seconds: Delay between page requests. None uses
                the instance default.
            **params: Path and query parameters.

        Yields:
            ParsedResponse objects for each page, containing items and metadata.

        Raises:
            APIError: For non-retryable API errors.
            TransientAPIError: After exhausting retries on transient errors.
        """
        yield from self._paginate(endpoint, request_delay_seconds, **params)

    def _paginate[ItemT: BaseModel](
        self,
        endpoint: EndpointDefinition[ItemT],
        request_delay_seconds: float | None,
        **params: Any,
    ) -> Iterator[ParsedResponse[ItemT]]:
        """
        Internal pagination loop shared by fetch_all and fetch_all_pages.

        Args:
            endpoint: Self-describing endpoint definition.
            request_delay_seconds: Delay between page requests.
            **params: Path and query parameters.

        Yields:
            ParsedResponse objects for each page.
        """
        pagination_state: PaginationState | None = None
        page_count: int = 0
        total_items: int = 0
        effective_delay: float = self._get_effective_delay(request_delay_seconds)

        while True:
            response: ParsedResponse[ItemT] = self.fetch(
                endpoint,
                pagination_state=pagination_state,
                **params,
            )

            page_count += 1
            total_items += response.item_count

            logger.debug(
                'Page %d: %d items (running total: %d)',
                page_count,
                response.item_count,
                total_items,
            )

            yield response

            if not response.has_more:
                rel_path: str = endpoint.build_resource_path(**params)
                logger.info(
                    'Pagination complete for %r: %d items across %d pages',
                    rel_path,
                    total_items,
                    page_count,
                )
                break

            pagination_state = response.pagination

            if effective_delay > 0:
                time.sleep(effective_delay)

    def to_dataframe[ItemT: BaseModel](
        self,
        endpoint: EndpointDefinition[ItemT],
        request_delay_seconds: float | None = None,
        **params: Any,
    ) -> pd.DataFrame:
        """
        Fetch all data from an endpoint and return as a pandas DataFrame.

        Convenience method that fetches all items, converts Pydantic models
        to dictionaries, and creates a DataFrame. Note that this loads all
        data into memory; for large datasets, consider using fetch_all()
        and processing items incrementally.

        Args:
            endpoint: Self-describing endpoint definition.
            request_delay_seconds: Delay between page requests.
            **params: Path and query parameters.

        Returns:
            DataFrame containing all fetched data. Returns empty DataFrame
            with no columns if no items were found.

        Example:
            >>> with TelemetryClient(credentials) as client:
            ...     vehicles_df = client.to_dataframe(MotiveEndpoints.VEHICLES)
            ...     vehicles_df.to_parquet('vehicles.parquet')
        """
        effective_delay: float = self._get_effective_delay(request_delay_seconds)

        items: list[ItemT] = list(
            self.fetch_all(
                endpoint,
                request_delay_seconds=effective_delay,
                **params,
            )
        )

        if not items:
            logger.warning(
                'No items found for endpoint %r with params %r',
                endpoint.endpoint_path,
                params,
            )
            return pd.DataFrame()

        # Convert Pydantic models to dictionaries for DataFrame construction
        records: list[dict[str, Any]] = [
            item.model_dump() if hasattr(item, 'model_dump') else dict(item)
            for item in items
        ]

        dataframe = pd.DataFrame(records)

        logger.info(
            'Created DataFrame: %d rows, %d columns from %r',
            len(dataframe),
            len(dataframe.columns),
            endpoint.endpoint_path,
        )

        return dataframe

    # -------------------------------------------------------------------------
    # HTTP Execution Layer
    # -------------------------------------------------------------------------

    def _execute_request(self, request_spec: RequestSpec) -> dict[str, Any]:
        """
        Execute an HTTP request with automatic retry on transient failures.

        Args:
            request_spec: Complete request specification including URL,
                method, headers, query params, and body.

        Returns:
            Parsed JSON response as dictionary.

        Raises:
            APIError: For non-retryable errors (4xx except 429).
            TransientAPIError: After exhausting retries.
            RateLimitError: After exhausting rate limit retries.
        """
        return self._execute_with_retry(request_spec)

    @retry(
        retry=retry_if_exception_type(TransientAPIError),
        wait=_wait_for_rate_limit_or_exponential,
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        reraise=True,
    )
    def _execute_with_retry(self, request_spec: RequestSpec) -> dict[str, Any]:
        """
        Execute request with tenacity retry wrapper.

        Retries on TransientAPIError (which includes RateLimitError).
        Uses custom wait strategy that respects Retry-After for rate limits.
        """
        # Execute HTTP request
        response: httpx.Response = self._send_http_request(request_spec)

        # Handle response based on status code
        return self._handle_response(response)

    def _send_http_request(self, request_spec: RequestSpec) -> httpx.Response:
        """
        Send the HTTP request, converting transport errors to TransientAPIError.

        Args:
            request_spec: Complete request specification.

        Returns:
            httpx Response object.

        Raises:
            TransientAPIError: On timeout or connection errors (retryable).
        """
        timeout = httpx.Timeout(
            connect=request_spec.timeout[0],
            read=request_spec.timeout[1],
            write=request_spec.timeout[0],
            pool=request_spec.timeout[0],
        )

        try:
            return self._http_client.request(
                method=request_spec.method.value,
                url=request_spec.url,
                params=request_spec.query_params,
                headers=request_spec.headers,
                json=request_spec.body,
                timeout=timeout,
            )
        except httpx.TimeoutException as error:
            logger.warning('Request timeout (will retry): %s', request_spec.url)
            raise TransientAPIError(f'Request timeout: {error}') from error
        except httpx.RequestError as error:
            logger.warning(
                'Connection error (will retry): %s - %s', request_spec.url, error
            )
            raise TransientAPIError(f'Connection error: {error}') from error

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """
        Handle HTTP response, raising appropriate exceptions for errors.

        Args:
            response: httpx Response object.

        Returns:
            Parsed JSON response body.

        Raises:
            RateLimitError: On HTTP 429 (retryable).
            TransientAPIError: On 5xx server errors (retryable).
            APIError: On 4xx client errors or malformed responses (not retryable).
        """
        status_code: int = response.status_code

        # Rate limit - retryable with Retry-After
        if status_code == HTTP_STATUS_RATE_LIMITED:
            rate_limit_info: RateLimitInfo = RateLimitInfo.from_response_headers(
                dict(response.headers)
            )
            logger.warning(
                'Rate limited (will retry after %.1fs): remaining=%d',
                rate_limit_info.retry_after_seconds,
                rate_limit_info.remaining,
            )
            raise RateLimitError(rate_limit_info)

        # Server errors - retryable
        if HTTP_STATUS_SERVER_ERROR_MIN <= status_code <= HTTP_STATUS_SERVER_ERROR_MAX:
            logger.warning(
                'Server error %d (will retry): %s',
                status_code,
                response.text[:200],
            )
            raise TransientAPIError(
                message=f'Server error: HTTP {status_code}',
                status_code=status_code,
                response_body=response.text,
            )

        # Client errors (4xx except 429) - not retryable
        if not response.is_success:
            logger.error(
                'Client error %d (not retryable): %s',
                status_code,
                response.text[:500],
            )
            raise APIError(
                message=f'Client error: HTTP {status_code}',
                status_code=status_code,
                response_body=response.text,
            )

        # Parse JSON and validate it's a dictionary
        try:
            json_body: Any = response.json()
        except ValueError as parse_error:
            raise APIError(
                message=f'Invalid JSON in response: {parse_error}',
                status_code=status_code,
                response_body=response.text[:500],
            ) from parse_error

        if not isinstance(json_body, dict):
            raise APIError(
                message=(
                    f'Expected JSON object in response, got {type(json_body).__name__}. '
                    f'Content: {response.text[:200]}'
                ),
                status_code=status_code,
                response_body=response.text[:500],
            )

        # Type narrowing: isinstance check above guarantees this is dict[str, Any]
        # JSON spec requires all object keys to be strings, so this is safe.
        validated_response: dict[str, Any] = cast(dict[str, Any], json_body)

        return validated_response
