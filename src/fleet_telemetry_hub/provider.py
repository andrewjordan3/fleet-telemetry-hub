# fleet_telemetry_hub/provider.py
"""
High-level provider facade for streamlined API access.

This module provides a convenient object-oriented interface that combines
provider configuration, endpoint access, and client operations into a
unified API.

Example:
    >>> # Configure provider
    >>> config = ProviderConfig(
    ...     enabled=True,
    ...     base_url="https://api.gomotive.com",
    ...     api_key="your-key",
    ...     request_timeout=(10, 30),
    ...     max_retries=5,
    ...     retry_backoff_factor=2.0,
    ...     verify_ssl=True,
    ...     rate_limit_requests_per_second=10
    ... )
    >>>
    >>> # Create provider interface
    >>> motive = Provider.from_config("motive", config)
    >>>
    >>> # Access endpoints directly
    >>> for vehicle in motive.fetch_all("vehicles"):
    ...     print(vehicle.number)
"""

import logging
from collections.abc import Iterator
from typing import Any, Self

import pandas as pd

from .client import TelemetryClient
from .config import ProviderConfig, TelemetryConfig
from .models.shared_response_models import (
    EndpointDefinition,
    ParsedResponse,
    ProviderCredentials,
)
from .registry import EndpointRegistry

logger: logging.Logger = logging.getLogger(__name__)


class Provider:
    """
    High-level facade for provider API access.

    Combines provider configuration, endpoint registry, and client operations
    into a single convenient interface. Eliminates the need to manually
    manage client instances and endpoint lookups.

    This class provides both direct endpoint access and pass-through methods
    for common client operations.

    Attributes:
        name: Provider name (e.g., 'motive', 'samsara').
        credentials: Provider credentials and connection settings.

    Example:
        >>> # Create provider from config
        >>> config = load_config("config.yaml")
        >>> motive = Provider.from_config("motive", config.providers["motive"])
        >>>
        >>> # Fetch all vehicles (automatic client management)
        >>> for vehicle in motive.fetch_all("vehicles"):
        ...     print(vehicle.number)
        >>>
        >>> # Or use context manager for custom client usage
        >>> with motive.client() as client:
        ...     response = client.fetch(motive.endpoint("vehicles"))
        ...     print(f"Found {response.item_count} vehicles")
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
            name: Provider name (e.g., 'motive', 'samsara').
            credentials: Provider credentials and connection settings.
            registry: Optional custom endpoint registry. Uses global if not provided.
        """
        self.name: str = name.lower()
        self.credentials: ProviderCredentials = credentials
        self.config: TelemetryConfig | None = config
        self._registry: EndpointRegistry = registry or EndpointRegistry.instance()

        # Validate provider exists in registry
        if self.name not in self._registry.list_providers():
            available: str = ', '.join(self._registry.list_providers())
            raise ValueError(
                f"Provider '{name}' not found in registry. "
                f'Available providers: {available}'
            )

        logger.info(f"Initialized Provider facade for '{self.name}'")

    @classmethod
    def from_config(
        cls,
        provider_name: str,
        config: TelemetryConfig,
        registry: EndpointRegistry | None = None,
    ) -> Self:
        """
        Create provider from configuration object.

        Converts ProviderConfig into ProviderCredentials and initializes
        the provider facade.

        Args:
            provider_name: Provider name (e.g., 'motive', 'samsara').
            config: Package level onfiguration from config file.
            registry: Optional custom endpoint registry.

        Returns:
            Initialized Provider instance.

        Example:
            >>> from fleet_telemetry_hub.config.loader import load_config
            >>>
            >>> config = load_config("config.yaml")
            >>> motive = Provider.from_config("motive", config)
        """
        provider_config: ProviderConfig | None = config.providers.get(provider_name)
        if not provider_config:
            logger.error(f'No configuration found for provider {provider_name}.')
            raise NotImplementedError

        credentials = ProviderCredentials(
            base_url=provider_config.base_url,
            api_key=provider_config.api_key,
            timeout=tuple(provider_config.request_timeout),  # type: ignore[arg-type]
            max_retries=provider_config.max_retries,
            retry_backoff_factor=provider_config.retry_backoff_factor,
            verify_ssl=provider_config.verify_ssl,
            use_truststore=config.pipeline.use_truststore
        )

        return cls(provider_name, credentials, config, registry)

    def endpoint(self, endpoint_name: str) -> EndpointDefinition[Any, Any]:
        """
        Get an endpoint definition by name.

        Args:
            endpoint_name: Endpoint name (case-insensitive: 'vehicles', 'drivers').

        Returns:
            Endpoint definition ready for use with client.

        Raises:
            EndpointNotFoundError: If endpoint doesn't exist for this provider.

        Example:
            >>> motive = Provider.from_config("motive", config)
            >>> vehicles_endpoint = motive.endpoint("vehicles")
            >>> print(vehicles_endpoint.description)
        """
        return self._registry.get(self.name, endpoint_name)

    def list_endpoints(self) -> list[str]:
        """
        List all available endpoint names for this provider.

        Returns:
            List of endpoint names.

        Example:
            >>> motive = Provider.from_config("motive", config)
            >>> print(motive.list_endpoints())
            ['vehicles', 'vehicle_locations', 'groups', 'users']
        """
        return self._registry.list_endpoints(self.name)

    def has_endpoint(self, endpoint_name: str) -> bool:
        """
        Check if an endpoint exists for this provider.

        Args:
            endpoint_name: Endpoint name to check.

        Returns:
            True if endpoint exists, False otherwise.

        Example:
            >>> if motive.has_endpoint("vehicles"):
            ...     data = motive.fetch_all("vehicles")
        """
        return self._registry.has(self.name, endpoint_name)

    def client(
        self,
        pool_connections: int = 5,
        pool_maxsize: int = 10,
    ) -> TelemetryClient:
        """
        Create a telemetry client for this provider.

        Returns a context manager that handles client lifecycle.

        Args:
            pool_connections: Max keepalive connections in pool.
            pool_maxsize: Max total connections in pool.

        Returns:
            TelemetryClient configured for this provider.

        Example:
            >>> with motive.client() as client:
            ...     for vehicle in client.fetch_all(motive.endpoint("vehicles")):
            ...         print(vehicle.number)
        """
        request_delay_seconds: float = self.get_request_delay()
        return TelemetryClient(
            credentials=self.credentials,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            request_delay_seconds=request_delay_seconds,
        )

    # -------------------------------------------------------------------------
    # Convenience Methods (Auto-managed Client)
    # -------------------------------------------------------------------------

    def fetch(
        self,
        endpoint_name: str,
        **params: Any,
    ) -> ParsedResponse[Any]:
        """
        Fetch a single page from an endpoint.

        Creates and manages a client automatically. For repeated operations,
        use the client() context manager for better performance.

        Args:
            endpoint_name: Endpoint name (e.g., 'vehicles').
            **params: Path parameters and query parameters.

        Returns:
            ParsedResponse containing typed items and pagination state.

        Example:
            >>> motive = Provider.from_config("motive", config)
            >>> response = motive.fetch("vehicles")
            >>> print(f"Found {response.item_count} vehicles on this page")
        """
        endpoint: EndpointDefinition[Any, Any] = self.endpoint(endpoint_name)

        with self.client() as client:
            return client.fetch(endpoint, **params)

    def fetch_all(
        self,
        endpoint_name: str,
        request_delay_seconds: float = 0.0,
        **params: Any,
    ) -> Iterator[Any]:
        """
        Iterate through all items across all pages.

        Creates and manages a client automatically. Yields individual items
        as they are retrieved.

        Args:
            endpoint_name: Endpoint name (e.g., 'vehicles').
            request_delay_seconds: Delay between page requests.
            **params: Path and query parameters.

        Yields:
            Individual items from each page.

        Example:
            >>> motive = Provider.from_config("motive", config)
            >>> for vehicle in motive.fetch_all("vehicles"):
            ...     print(f"{vehicle.number}: {vehicle.make} {vehicle.model}")
        """
        endpoint: EndpointDefinition[Any, Any] = self.endpoint(endpoint_name)

        with self.client() as client:
            yield from client.fetch_all(
                endpoint,
                request_delay_seconds=request_delay_seconds,
                **params,
            )

    def fetch_all_pages(
        self,
        endpoint_name: str,
        request_delay_seconds: float = 0.0,
        **params: Any,
    ) -> Iterator[ParsedResponse[Any]]:
        """
        Iterate through all pages, yielding full ParsedResponse objects.

        Useful when you need pagination metadata or batch processing.

        Args:
            endpoint_name: Endpoint name (e.g., 'vehicles').
            request_delay_seconds: Delay between page requests.
            **params: Path and query parameters.

        Yields:
            ParsedResponse objects for each page.

        Example:
            >>> motive = Provider.from_config("motive", config)
            >>> for page in motive.fetch_all_pages("vehicles"):
            ...     print(f"Processing page with {page.item_count} items")
            ...     process_batch(page.items)
        """
        endpoint: EndpointDefinition[Any, Any] = self.endpoint(endpoint_name)

        with self.client() as client:
            yield from client.fetch_all_pages(
                endpoint,
                request_delay_seconds=request_delay_seconds,
                **params,
            )

    def to_dataframe(
        self,
        endpoint_name: str,
        request_delay_seconds: float = 0.0,
        **params: Any,
    ) -> pd.DataFrame:
        """
        Fetch all data from an endpoint and return as a pandas DataFrame.

        This is a convenience method that fetches all items, converts them
        from Pydantic models to dictionaries, and creates a DataFrame.

        Args:
            endpoint_name: Endpoint name (e.g., 'vehicles').
            request_delay_seconds: Delay between page requests.
            **params: Path and query parameters.

        Returns:
            pandas DataFrame containing all fetched data.

        Raises:
            ImportError: If pandas is not installed.

        Example:
            >>> motive = Provider.from_config("motive", config)
            >>>
            >>> # Get all vehicles as DataFrame
            >>> df = motive.to_dataframe("vehicles")
            >>> print(df.head())
            >>>
            >>> # With parameters
            >>> df = motive.to_dataframe(
            ...     "vehicle_locations",
            ...     vehicle_id=12345,
            ...     start_date=date(2025, 1, 1),
            ... )
            >>>
            >>> # Save to file
            >>> df.to_parquet("vehicles.parquet")
            >>> df.to_csv("vehicles.csv")
        """
        # Fetch all items
        items: list[Any] = list(
            self.fetch_all(
                endpoint_name,
                request_delay_seconds=request_delay_seconds,
                **params,
            )
        )

        if not items:
            logger.warning(
                f'No items found for endpoint "{endpoint_name}" with params {params}'
            )
            return pd.DataFrame()

        # Convert Pydantic models to dictionaries
        data: list[Any | dict[str, Any]] = [
            item.model_dump() if hasattr(item, 'model_dump') else dict(item)
            for item in items
        ]

        # Create DataFrame
        df = pd.DataFrame(data)

        logger.info(
            f'Created DataFrame from {len(items)} items with {len(df.columns)} columns'
        )

        return df

    def get_request_delay(self) -> float:
        """
        Get the effective request delay, considering both pipeline global delay
        and provider-specific rate limits.

        Returns the larger of:
        - Pipeline global delay
        - 1/rate_limit (time per request based on rate limit)
        """
        if not self.config:
            return 0.0

        pipeline_delay: float = self.config.pipeline.request_delay_seconds
        provider_config: ProviderConfig | None = self.config.providers.get(self.name)
        if not provider_config:
            return pipeline_delay

        # Calculate minimum delay based on rate limit
        # e.g., 10 req/sec = 0.1 sec/req minimum
        rate_limit: int = provider_config.rate_limit_requests_per_second
        rate_delay: float = 1.0 / rate_limit if rate_limit > 0 else 0.0

        # Use the more conservative (larger) delay
        return max(pipeline_delay, rate_delay)

    def describe(self) -> str:
        """
        Generate human-readable description of this provider's endpoints.

        Returns:
            Formatted string with endpoint descriptions.

        Example:
            >>> motive = Provider.from_config("motive", config)
            >>> print(motive.describe())
        """
        return self._registry.describe(self.name)

    def __repr__(self) -> str:
        """String representation of Provider."""
        endpoint_count: int = len(self.list_endpoints())
        return (
            f"Provider(name='{self.name}', "
            f"base_url='{self.credentials.base_url}', "
            f'endpoints={endpoint_count})'
        )


# =============================================================================
# Multi-Provider Facade
# =============================================================================


class ProviderManager:
    """
    Manages multiple provider instances from configuration.

    Provides unified access to multiple providers (Motive, Samsara, etc.)
    from a single configuration file.

    Example:
        >>> from fleet_telemetry_hub.config.loader import load_config
        >>>
        >>> config = load_config("config.yaml")
        >>> manager = ProviderManager.from_config(config)
        >>>
        >>> # Access specific provider
        >>> motive = manager.get("motive")
        >>> for vehicle in motive.fetch_all("vehicles"):
        ...     print(vehicle.number)
        >>>
        >>> # Or iterate all enabled providers
        >>> for provider_name, provider in manager.enabled_providers():
        ...     print(f"\nFetching from {provider_name}...")
        ...     for vehicle in provider.fetch_all("vehicles"):
        ...         print(vehicle.number)
    """

    def __init__(
        self,
        providers: dict[str, Provider],
    ) -> None:
        """
        Initialize provider manager.

        Args:
            providers: Dictionary mapping provider names to Provider instances.
        """
        self._providers: dict[str, Provider] = providers

        logger.info(
            f'ProviderManager initialized with {len(providers)} provider(s): '
            f'{", ".join(providers.keys())}'
        )

    @classmethod
    def from_config(
        cls,
        config: TelemetryConfig,
        registry: EndpointRegistry | None = None,
    ) -> Self:
        """
        Create provider manager from configuration.

        Only initializes providers that are marked as enabled.

        Args:
            config: TelemetryConfig instance from config file.
            registry: Optional custom endpoint registry.

        Returns:
            Initialized ProviderManager.

        Example:
            >>> from fleet_telemetry_hub.config.loader import load_config
            >>>
            >>> config = load_config("config.yaml")
            >>> manager = ProviderManager.from_config(config)
        """
        providers: dict[str, Provider] = {}

        for provider_name, provider_config in config.providers.items():
            if provider_config.enabled:
                providers[provider_name] = Provider.from_config(
                    provider_name,
                    config,
                    registry,
                )
                logger.info(f'Loaded provider: {provider_name}')
            else:
                logger.info(f'Skipped disabled provider: {provider_name}')

        return cls(providers)

    def get(self, provider_name: str) -> Provider:
        """
        Get a provider by name.

        Args:
            provider_name: Provider name (case-sensitive).

        Returns:
            Provider instance.

        Raises:
            KeyError: If provider not found or not enabled.

        Example:
            >>> manager = ProviderManager.from_config(config)
            >>> motive = manager.get("motive")
        """
        if provider_name not in self._providers:
            available: str = ', '.join(self._providers.keys())
            raise KeyError(
                f"Provider '{provider_name}' not found or not enabled. "
                f'Available providers: {available}'
            )

        return self._providers[provider_name]

    def has(self, provider_name: str) -> bool:
        """
        Check if a provider is available.

        Args:
            provider_name: Provider name to check.

        Returns:
            True if provider is loaded and enabled.

        Example:
            >>> if manager.has("motive"):
            ...     motive = manager.get("motive")
        """
        return provider_name in self._providers

    def list_providers(self) -> list[str]:
        """
        Get list of all loaded provider names.

        Returns:
            List of provider names.

        Example:
            >>> manager = ProviderManager.from_config(config)
            >>> print(manager.list_providers())
            ['motive', 'samsara']
        """
        return list(self._providers.keys())

    def enabled_providers(self) -> Iterator[tuple[str, Provider]]:
        """
        Iterate through all enabled providers.

        Yields:
            Tuples of (provider_name, provider_instance).

        Example:
            >>> for name, provider in manager.enabled_providers():
            ...     print(f"Processing {name}...")
            ...     for vehicle in provider.fetch_all("vehicles"):
            ...         process(vehicle)
        """
        yield from self._providers.items()

    def __repr__(self) -> str:
        """String representation of ProviderManager."""
        return f'ProviderManager(providers={list(self._providers.keys())})'
