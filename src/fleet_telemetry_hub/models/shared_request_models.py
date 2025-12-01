# fleet_telemetry_hub/models/shared_request_models.py
"""
Provider-agnostic request specification models.

This module defines the contract between EndpointDefinitions (which build
request specs) and Clients (which execute them). The client never needs
to know about provider-specific auth patterns or pagination mechanics.
"""

import logging
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger: logging.Logger = logging.getLogger(__name__)


class HTTPMethod(str, Enum):
    """Supported HTTP methods for API requests."""

    GET = 'GET'
    POST = 'POST'
    PUT = 'PUT'
    DELETE = 'DELETE'


class RequestSpec(BaseModel):
    """
    Complete specification for an HTTP request.

    This is the contract between EndpointDefinition (producer) and Client
    (consumer). The client receives this fully-formed spec and executes it
    without needing to know provider-specific details.

    The EndpointDefinition is responsible for:
    - Building the full URL
    - Serializing query parameters
    - Injecting authentication headers
    - Setting provider-appropriate timeouts and retry config

    The Client is responsible for:
    - Executing the HTTP request
    - Handling retries and rate limits
    - Passing the response back to the endpoint for parsing

    Attributes:
        url: Complete URL ready for HTTP request.
        method: HTTP method (GET, POST, etc.).
        headers: All headers including authentication.
        query_params: Serialized query parameters (all strings).
        body: Request body for POST/PUT requests (None for GET).
        timeout: Tuple of (connect_timeout, read_timeout) in seconds.
        max_retries: Maximum retry attempts on failure.
        retry_backoff_factor: Exponential backoff multiplier.
        verify_ssl: SSL verification setting (bool or path to CA bundle).
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    url: str
    method: HTTPMethod = HTTPMethod.GET
    headers: dict[str, str] = Field(default_factory=dict)
    query_params: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] | None = None

    # Execution configuration
    timeout: tuple[int, int] = Field(
        default=(30, 120),
        description='(connect_timeout, read_timeout) in seconds',
    )
    max_retries: int = Field(default=3, ge=1)
    retry_backoff_factor: float = Field(default=1.0, gt=0)
    verify_ssl: bool | str = True


class RateLimitInfo(BaseModel):
    """
    Rate limit metadata extracted from HTTP response headers.

    Used by clients to implement intelligent throttling when 429 errors occur.

    Attributes:
        retry_after_seconds: Seconds to wait before retrying.
        limit: Maximum requests allowed in the rate limit window.
        remaining: Requests remaining in current window.
        reset_at_unix: Unix timestamp when the rate limit resets.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    retry_after_seconds: float = 1.0
    limit: int | None = None
    remaining: int | None = None
    reset_at_unix: int | None = None

    @classmethod
    def from_response_headers(cls, headers: dict[str, str]) -> 'RateLimitInfo':
        """
        Extract rate limit information from HTTP response headers.

        Handles case-insensitive header lookup and common header names
        across different APIs.

        Args:
            headers: HTTP response headers dictionary.

        Returns:
            RateLimitInfo with parsed values, defaults to 1 second retry
            if Retry-After header is missing.
        """
        # Normalize header names to lowercase for case-insensitive lookup
        normalized_headers: dict[str, str] = {
            key.lower(): value for key, value in headers.items()
        }

        # Standard Retry-After header (RFC 7231)
        retry_after_raw: str = normalized_headers.get('retry-after', '1')

        # Common rate limit headers (used by many APIs including Samsara)
        limit_raw: str | None = normalized_headers.get('x-ratelimit-limit')
        remaining_raw: str | None = normalized_headers.get('x-ratelimit-remaining')
        reset_raw: str | None = normalized_headers.get('x-ratelimit-reset')

        return cls(
            retry_after_seconds=float(retry_after_raw),
            limit=int(limit_raw) if limit_raw is not None else None,
            remaining=int(remaining_raw) if remaining_raw is not None else None,
            reset_at_unix=int(reset_raw) if reset_raw is not None else None,
        )

    @property
    def is_exhausted(self) -> bool:
        """Check if rate limit is completely exhausted."""
        return self.remaining is not None and self.remaining <= 0
