# fleet_telemetry_hub/provider.py
"""
High-level provider facade for streamlined API access.

This module provides a convenient object-oriented interface that combines
provider configuration, endpoint access, and client operations into a
unified API. It eliminates the need to manually manage client instances,
endpoint lookups, and credential handling.

Design Decisions:
-----------------
- Provider wraps credentials + registry + client factory into one object
- Convenience methods (fetch_all, to_dataframe) auto-manage client lifecycle
- For repeated operations, use client() context manager for connection reuse
- ProviderManager handles multi-provider scenarios from a single config

Usage:
------
    from fleet_telemetry_hub.config import load_config
    from fleet_telemetry_hub.provider import Provider, ProviderManager

    config = load_config('config.yaml')

    # Single provider usage
    motive = Provider.from_config('motive', config)
    for vehicle in motive.fetch_all('vehicles'):
        print(vehicle.number)

    # Multi-provider usage
    manager = ProviderManager.from_config(config)
    for name, provider in manager.items():
        print(f'{name}: {len(list(provider.fetch_all("vehicles")))} vehicles')
"""

import logging
from collections.abc import Iterator
from typing import Any, Self

import pandas as pd
from pydantic import BaseModel

from fleet_telemetry_hub.client import TelemetryClient
from fleet_telemetry_hub.config import ProviderConfig, TelemetryConfig
from fleet_telemetry_hub.models import (
    EndpointDefinition,
    ParsedResponse,
    ProviderCredentials,
)
from fleet_telemetry_hub.registry import EndpointRegistry, ProviderNotFoundError

__all__: list[str] = [
    'Provider',
    'ProviderConfigurationError',
    'ProviderManager',
]

logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class ProviderConfigurationError(Exception):
    """
    Raised when provider configuration is missing or invalid.

    Attributes:
        provider_name: The provider that failed to configure.
        available_providers: List of configured provider names, if known.
    """

    def __init__(
        self,
        provider_name: str,
        message: str,
        available_providers: list[str] | None = None,
    ) -> None:
        self.provider_name: str = provider_name
        self.available_providers: list[str] | None = available_providers
        super().__init__(message)


# =============================================================================
# Provider Facade
# =============================================================================


class Provider:
    """
    High-level facade for provider API access.

    Combines provider configuration, endpoint registry, and client operations
    into a single convenient interface. Eliminates the need to manually
    manage client instances and endpoint lookups.

    The Provider offers two usage patterns:

    1. Convenience methods (fetch_all, to_dataframe): Automatically create
       and close a client for each call. Simple but creates new connections.

    2. Context manager (client()): Reuse a single client for multiple
       operations. Better performance for batch operations.

    Attributes:
        name: Provider name (lowercase, e.g., 'motive', 'samsara').
        credentials: Provider credentials and connection settings.
        config: Full telemetry config (optional, used for rate limit calculation).

    Example:
        >>> config = load_config('config.yaml')
        >>> motive = Provider.from_config('motive', config)
        >>>
        >>> # Simple: auto-managed client per call
        >>> for vehicle in motive.fetch_all('vehicles'):
        ...     print(vehicle.number)
        >>>
        >>> # Efficient: reuse client for multiple operations
        >>> with motive.client() as client:
        ...     vehicles = list(client.fetch_all(motive.endpoint('vehicles')))
        ...     for v in vehicles:
        ...         locations = list(client.fetch_all(
        ...             motive.endpoint('vehicle_locations'),
        ...             vehicle_id=v.vehicle_id,
        ...         ))
    """

    def __init__(
        self,
        name: str,
        credentials: ProviderCredentials,
        config: TelemetryConfig | None = None,
        registry: EndpointRegistry | None = None,
    ) -> None:
        """
        Initialize provider facade.

        Args:
            name: Provider name (e.g., 'motive', 'samsara'). Normalized to lowercase.
            credentials: Provider credentials and connection settings.
            config: Optional full config, used for rate limit calculations.
            registry: Optional custom endpoint registry. Uses singleton if not provided.

        Raises:
            ProviderNotFoundError: If provider doesn't exist in the registry.
        """
        self._name: str = name.lower()
        self._credentials: ProviderCredentials = credentials
        self._config: TelemetryConfig | None = config
        self._registry: EndpointRegistry = registry or EndpointRegistry.instance()

        # Validate provider exists in registry (raises ProviderNotFoundError if not)
        # We call list_endpoints to trigger validation without fetching a specific endpoint
        self._registry.list_endpoints(self._name)

        logger.info(
            'Initialized Provider: name=%r, base_url=%r',
            self._name,
            credentials.base_url,
        )

    @property
    def name(self) -> str:
        """Provider name (lowercase)."""
        return self._name

    @property
    def credentials(self) -> ProviderCredentials:
        """Provider credentials and connection settings."""
        return self._credentials

    @property
    def config(self) -> TelemetryConfig | None:
        """Full telemetry config, if provided."""
        return self._config

    @classmethod
    def from_config(
        cls,
        provider_name: str,
        config: TelemetryConfig,
        registry: EndpointRegistry | None = None,
    ) -> Self:
        """
        Create provider from configuration object.

        Extracts the provider-specific configuration, converts it to
        ProviderCredentials, and initializes the provider facade.

        Args:
            provider_name: Provider name (e.g., 'motive', 'samsara').
            config: Full TelemetryConfig from config file.
            registry: Optional custom endpoint registry.

        Returns:
            Initialized Provider instance.

        Raises:
            ProviderConfigurationError: If provider not found in config.
            ProviderNotFoundError: If provider not found in registry.

        Example:
            >>> config = load_config('config.yaml')
            >>> motive = Provider.from_config('motive', config)
        """
        provider_config: ProviderConfig | None = config.providers.get(
            provider_name.lower()
        )

        if provider_config is None:
            available: list[str] = sorted(config.providers.keys())
            raise ProviderConfigurationError(
                provider_name=provider_name,
                message=(
                    f"Provider '{provider_name}' not found in configuration. "
                    f'Available: {", ".join(available)}'
                ),
                available_providers=available,
            )

        # Convert ProviderConfig to ProviderCredentials
        # The tuple cast is needed because ProviderConfig stores it as tuple[int, int]
        # but ProviderCredentials expects the same type
        timeout: tuple[int, int] = (
            provider_config.request_timeout[0],
            provider_config.request_timeout[1],
        )

        credentials = ProviderCredentials(
            base_url=provider_config.base_url,
            api_key=provider_config.api_key,
            timeout=timeout,
            max_retries=provider_config.max_retries,
            retry_backoff_factor=provider_config.retry_backoff_factor,
            verify_ssl=provider_config.verify_ssl,
            use_truststore=config.pipeline.use_truststore,
        )

        return cls(provider_name, credentials, config, registry)

    # -------------------------------------------------------------------------
    # Endpoint Access
    # -------------------------------------------------------------------------

    def endpoint(self, endpoint_name: str) -> EndpointDefinition[BaseModel]:
        """
        Get an endpoint definition by name.

        Args:
            endpoint_name: Endpoint name (case-insensitive).

        Returns:
            Endpoint definition ready for use with TelemetryClient.

        Raises:
            EndpointNotFoundError: If endpoint doesn't exist for this provider.
        """
        return self._registry.get(self._name, endpoint_name)

    def list_endpoints(self) -> list[str]:
        """
        List all available endpoint names for this provider.

        Returns:
            Sorted list of endpoint names.
        """
        return self._registry.list_endpoints(self._name)

    def has_endpoint(self, endpoint_name: str) -> bool:
        """
        Check if an endpoint exists for this provider.

        Args:
            endpoint_name: Endpoint name to check (case-insensitive).

        Returns:
            True if endpoint exists, False otherwise.
        """
        return self._registry.has(self._name, endpoint_name)

    # -------------------------------------------------------------------------
    # Client Management
    # -------------------------------------------------------------------------

    def client(
        self,
        pool_connections: int = 5,
        pool_maxsize: int = 10,
    ) -> TelemetryClient:
        """
        Create a TelemetryClient for this provider.

        Returns a client configured with this provider's credentials.
        Use as a context manager to ensure proper resource cleanup.

        Args:
            pool_connections: Max keepalive connections in pool.
            pool_maxsize: Max total connections in pool.

        Returns:
            TelemetryClient configured for this provider.

        Example:
            >>> with motive.client() as client:
            ...     for vehicle in client.fetch_all(motive.endpoint('vehicles')):
            ...         print(vehicle.number)
        """
        return TelemetryClient(
            credentials=self._credentials,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            request_delay_seconds=self._calculate_request_delay(),
        )

    def _calculate_request_delay(self) -> float:
        """
        Calculate the effective request delay for this provider.

        Returns the larger of:
        - Pipeline-level request_delay_seconds (global throttle)
        - 1/rate_limit_requests_per_second (provider-specific rate limit)

        Returns:
            Delay in seconds between requests.
        """
        if self._config is None:
            return 0.0

        pipeline_delay: float = self._config.pipeline.request_delay_seconds

        # Get provider-specific rate limit
        provider_config: ProviderConfig | None = self._config.providers.get(self._name)
        if provider_config is None:
            return pipeline_delay

        # Convert rate limit to delay: 10 req/sec = 0.1 sec/req
        rate_limit: int = provider_config.rate_limit_requests_per_second
        rate_based_delay: float = 1.0 / rate_limit if rate_limit > 0 else 0.0

        # Use the more conservative (larger) delay
        return max(pipeline_delay, rate_based_delay)

    # -------------------------------------------------------------------------
    # Convenience Methods (Auto-managed Client)
    # -------------------------------------------------------------------------

    def fetch(
        self,
        endpoint_name: str,
        **params: Any,
    ) -> ParsedResponse[BaseModel]:
        """
        Fetch a single page from an endpoint.

        Creates a client, fetches one page, and closes the client.
        For repeated operations, use client() context manager instead.

        Args:
            endpoint_name: Endpoint name (e.g., 'vehicles').
            **params: Path and query parameters.

        Returns:
            ParsedResponse containing typed items and pagination state.
        """
        endpoint: EndpointDefinition[BaseModel] = self.endpoint(endpoint_name)

        with self.client() as client:
            return client.fetch(endpoint, **params)

    def fetch_all(
        self,
        endpoint_name: str,
        request_delay_seconds: float | None = None,
        **params: Any,
    ) -> Iterator[BaseModel]:
        """
        Iterate through all items across all pages.

        Creates a client, paginates through all results, and closes the client.
        Items are yielded one at a time for memory efficiency.

        Args:
            endpoint_name: Endpoint name (e.g., 'vehicles').
            request_delay_seconds: Override delay between page requests.
                None uses the calculated default from rate limits.
            **params: Path and query parameters.

        Yields:
            Individual items from each page.

        Example:
            >>> for vehicle in motive.fetch_all('vehicles'):
            ...     print(f'{vehicle.number}: {vehicle.make} {vehicle.model}')
        """
        endpoint: EndpointDefinition[BaseModel] = self.endpoint(endpoint_name)

        with self.client() as client:
            yield from client.fetch_all(
                endpoint,
                request_delay_seconds=request_delay_seconds,
                **params,
            )

    def fetch_all_pages(
        self,
        endpoint_name: str,
        request_delay_seconds: float | None = None,
        **params: Any,
    ) -> Iterator[ParsedResponse[BaseModel]]:
        """
        Iterate through all pages, yielding full ParsedResponse objects.

        Useful when you need pagination metadata or want to process
        items in page-sized batches.

        Args:
            endpoint_name: Endpoint name (e.g., 'vehicles').
            request_delay_seconds: Override delay between page requests.
            **params: Path and query parameters.

        Yields:
            ParsedResponse objects for each page.
        """
        endpoint: EndpointDefinition[BaseModel] = self.endpoint(endpoint_name)

        with self.client() as client:
            yield from client.fetch_all_pages(
                endpoint,
                request_delay_seconds=request_delay_seconds,
                **params,
            )

    def to_dataframe(
        self,
        endpoint_name: str,
        request_delay_seconds: float | None = None,
        **params: Any,
    ) -> pd.DataFrame:
        """
        Fetch all data from an endpoint and return as a DataFrame.

        Convenience method that fetches all items, converts Pydantic models
        to dictionaries, and creates a DataFrame. Loads all data into memory.

        Args:
            endpoint_name: Endpoint name (e.g., 'vehicles').
            request_delay_seconds: Override delay between page requests.
            **params: Path and query parameters.

        Returns:
            DataFrame containing all fetched data. Empty DataFrame if no items.

        Example:
            >>> df = motive.to_dataframe('vehicles')
            >>> df.to_parquet('vehicles.parquet')
        """
        endpoint: EndpointDefinition[BaseModel] = self.endpoint(endpoint_name)

        with self.client() as client:
            return client.to_dataframe(
                endpoint,
                request_delay_seconds=request_delay_seconds,
                **params,
            )

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def describe(self) -> str:
        """
        Generate human-readable description of this provider's endpoints.

        Returns:
            Formatted multi-line string with endpoint details.
        """
        return self._registry.describe(self._name)

    def __repr__(self) -> str:
        """String representation for debugging."""
        endpoint_count: int = len(self.list_endpoints())
        return (
            f'Provider(name={self._name!r}, '
            f'base_url={self._credentials.base_url!r}, '
            f'endpoints={endpoint_count})'
        )


# =============================================================================
# Multi-Provider Manager
# =============================================================================


class ProviderManager:
    """
    Manages multiple provider instances from configuration.

    Provides unified access to all enabled providers from a single config.
    Only providers with `enabled: true` in configuration are loaded.

    Example:
        >>> config = load_config('config.yaml')
        >>> manager = ProviderManager.from_config(config)
        >>>
        >>> # Get specific provider (returns None if not enabled)
        >>> motive = manager.get('motive')
        >>> if motive:
        ...     for vehicle in motive.fetch_all('vehicles'):
        ...         print(vehicle.number)
        >>>
        >>> # Iterate all enabled providers
        >>> for name, provider in manager.items():
        ...     print(f'{name}: {len(list(provider.fetch_all("vehicles")))} vehicles')
    """

    def __init__(self, providers: dict[str, Provider]) -> None:
        """
        Initialize provider manager.

        Args:
            providers: Dictionary mapping provider names to Provider instances.
                Only enabled providers should be included.
        """
        self._providers: dict[str, Provider] = providers

        logger.info(
            'ProviderManager initialized: %d provider(s) [%s]',
            len(providers),
            ', '.join(sorted(providers.keys())),
        )

    @classmethod
    def from_config(
        cls,
        config: TelemetryConfig,
        registry: EndpointRegistry | None = None,
    ) -> Self:
        """
        Create provider manager from configuration.

        Only initializes providers that are marked as enabled in config.

        Args:
            config: TelemetryConfig instance from config file.
            registry: Optional custom endpoint registry.

        Returns:
            Initialized ProviderManager with enabled providers.

        Example:
            >>> config = load_config('config.yaml')
            >>> manager = ProviderManager.from_config(config)
            >>> print(manager.list_providers())
            ['motive', 'samsara']
        """
        providers: dict[str, Provider] = {}

        for provider_name, provider_config in config.providers.items():
            if provider_config.enabled:
                try:
                    providers[provider_name] = Provider.from_config(
                        provider_name,
                        config,
                        registry,
                    )
                    logger.info('Loaded enabled provider: %r', provider_name)
                except ProviderNotFoundError:
                    # Provider is configured but not in registry (unknown provider)
                    logger.warning(
                        'Provider %r is enabled but not found in registry, skipping',
                        provider_name,
                    )
            else:
                logger.debug('Skipped disabled provider: %r', provider_name)

        return cls(providers)

    def get(self, provider_name: str) -> Provider | None:
        """
        Get a provider by name, or None if not available.

        Returns None for providers that are disabled, not configured,
        or not found in the registry. Use this when you want to gracefully
        handle missing providers.

        Args:
            provider_name: Provider name (case-sensitive as stored).

        Returns:
            Provider instance if available, None otherwise.

        Example:
            >>> motive = manager.get('motive')
            >>> if motive:
            ...     process_motive_data(motive)
        """
        return self._providers.get(provider_name)

    def require(self, provider_name: str) -> Provider:
        """
        Get a provider by name, raising if not available.

        Use this when a provider is required and its absence is an error.

        Args:
            provider_name: Provider name (case-sensitive).

        Returns:
            Provider instance.

        Raises:
            KeyError: If provider is not available.

        Example:
            >>> motive = manager.require('motive')  # Raises if not enabled
        """
        provider: Provider | None = self._providers.get(provider_name)

        if provider is None:
            available: str = ', '.join(sorted(self._providers.keys()))
            raise KeyError(
                f"Provider '{provider_name}' not available. "
                f'Enabled providers: {available or "(none)"}'
            )

        return provider

    def has(self, provider_name: str) -> bool:
        """
        Check if a provider is available.

        Args:
            provider_name: Provider name to check.

        Returns:
            True if provider is loaded and enabled.
        """
        return provider_name in self._providers

    def list_providers(self) -> list[str]:
        """
        Get list of all enabled provider names.

        Returns:
            Sorted list of provider names.
        """
        return sorted(self._providers.keys())

    def items(self) -> Iterator[tuple[str, Provider]]:
        """
        Iterate through all enabled providers.

        Yields providers in sorted order by name for deterministic iteration.

        Yields:
            Tuples of (provider_name, provider_instance).

        Example:
            >>> for name, provider in manager.items():
            ...     print(f'Processing {name}...')
            ...     for vehicle in provider.fetch_all('vehicles'):
            ...         process(vehicle)
        """
        for name in sorted(self._providers.keys()):
            yield name, self._providers[name]

    def __len__(self) -> int:
        """Number of enabled providers."""
        return len(self._providers)

    def __bool__(self) -> bool:
        """True if at least one provider is enabled."""
        return len(self._providers) > 0

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f'ProviderManager(providers={sorted(self._providers.keys())})'
