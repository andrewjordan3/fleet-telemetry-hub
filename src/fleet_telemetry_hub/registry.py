# fleet_telemetry_hub/registry.py
"""
Unified endpoint registry for dynamic endpoint access across providers.

This module provides a centralized registry that allows string-based endpoint
lookup, enabling dynamic endpoint access without hardcoding provider-specific
class references throughout the codebase.

Design Decisions:
-----------------
- Case-insensitive lookups: Provider and endpoint names are normalized to
  lowercase internally, so 'MOTIVE', 'Motive', and 'motive' all work.

- Immutable after initialization: The registry contents cannot be modified
  after construction, preventing accidental corruption.

- Singleton convenience: The `instance()` class method provides a shared
  registry for simple use cases, but you can also create independent instances.

Usage:
------
    from fleet_telemetry_hub.registry import EndpointRegistry

    # Using the singleton
    registry = EndpointRegistry.instance()
    endpoint = registry.get('motive', 'vehicles')

    # Or create your own instance
    registry = EndpointRegistry()
    endpoint = registry.get('samsara', 'vehicle_stats_history')
"""

import logging
from types import MappingProxyType
from typing import Any, Self

from pydantic import BaseModel

from fleet_telemetry_hub.models import (
    EndpointDefinition,
    MotiveEndpointDefinition,
    MotiveEndpoints,
    SamsaraEndpointDefinition,
    SamsaraEndpoints,
)

__all__: list[str] = [
    'EndpointNotFoundError',
    'EndpointRegistry',
    'ProviderNotFoundError',
]

logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class ProviderNotFoundError(Exception):
    """
    Raised when requested provider doesn't exist in registry.

    Attributes:
        provider: The provider name that was not found.
        available_providers: List of valid provider names.
    """

    def __init__(self, provider: str, available_providers: list[str]) -> None:
        self.provider: str = provider
        self.available_providers: list[str] = available_providers
        super().__init__(
            f"Provider '{provider}' not found. "
            f'Available: {", ".join(sorted(available_providers))}'
        )


class EndpointNotFoundError(Exception):
    """
    Raised when requested endpoint doesn't exist for a provider.

    Attributes:
        provider: The provider name.
        endpoint_name: The endpoint name that was not found.
        available_endpoints: List of valid endpoint names for this provider.
    """

    def __init__(
        self,
        provider: str,
        endpoint_name: str,
        available_endpoints: list[str],
    ) -> None:
        self.provider: str = provider
        self.endpoint_name: str = endpoint_name
        self.available_endpoints: list[str] = available_endpoints
        super().__init__(
            f"Endpoint '{endpoint_name}' not found for provider '{provider}'. "
            f'Available: {", ".join(sorted(available_endpoints))}'
        )


# =============================================================================
# Registry
# =============================================================================


class EndpointRegistry:
    """
    Centralized registry for all API endpoints across providers.

    Provides string-based access to endpoint definitions without requiring
    direct references to provider-specific classes like MotiveEndpoints
    or SamsaraEndpoints. This enables dynamic endpoint selection based on
    configuration or user input.

    The registry is immutable after initialization. All lookups are
    case-insensitive.

    Example:
        >>> registry = EndpointRegistry()
        >>>
        >>> # Dynamic endpoint access
        >>> endpoint = registry.get('motive', 'vehicles')
        >>> with TelemetryClient(credentials) as client:
        ...     for vehicle in client.fetch_all(endpoint):
        ...         print(vehicle.number)
        >>>
        >>> # Discover available endpoints
        >>> for provider in registry.list_providers():
        ...     print(f'{provider}: {registry.list_endpoints(provider)}')
    """

    _instance: Self | None = None

    def __init__(self) -> None:
        """
        Initialize registry with all known providers and endpoints.

        Endpoint names are normalized to lowercase for case-insensitive lookup.
        The registry is frozen after initialization using MappingProxyType.
        """
        # Build mutable registry during initialization
        mutable_registry: dict[str, dict[str, EndpointDefinition[BaseModel]]] = {
            'motive': self._build_motive_endpoints(),
            'samsara': self._build_samsara_endpoints(),
        }

        # Freeze the registry to prevent modification after initialization.
        # MappingProxyType provides a read-only view of the underlying dict.
        self._registry: MappingProxyType[
            str, MappingProxyType[str, EndpointDefinition[BaseModel]]
        ] = MappingProxyType(
            {
                provider: MappingProxyType(endpoints)
                for provider, endpoints in mutable_registry.items()
            }
        )

        total_endpoints: int = sum(len(eps) for eps in self._registry.values())
        logger.info(
            'EndpointRegistry initialized: %d providers, %d endpoints',
            len(self._registry),
            total_endpoints,
        )

    @classmethod
    def instance(cls) -> Self:
        """
        Get the shared registry instance (singleton pattern).

        Convenience method that returns a pre-initialized global registry.
        For testing or isolated usage, create your own instance instead.

        Returns:
            The shared EndpointRegistry instance.

        Note:
            This method is not thread-safe for initial creation. In multi-threaded
            applications, create the singleton during startup before spawning threads.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -------------------------------------------------------------------------
    # Endpoint Lookup
    # -------------------------------------------------------------------------

    def get(
        self,
        provider: str,
        endpoint_name: str,
    ) -> EndpointDefinition[BaseModel]:
        """
        Get an endpoint definition by provider and name.

        Args:
            provider: Provider name (case-insensitive).
            endpoint_name: Endpoint name (case-insensitive).

        Returns:
            The endpoint definition ready for use with TelemetryClient.

        Raises:
            ProviderNotFoundError: If provider doesn't exist. Exception includes
                list of available providers.
            EndpointNotFoundError: If endpoint doesn't exist for provider.
                Exception includes list of available endpoints.

        Example:
            >>> registry = EndpointRegistry()
            >>> endpoint = registry.get('motive', 'vehicles')
            >>> print(endpoint.description)
        """
        provider_lower: str = provider.lower()
        endpoint_lower: str = endpoint_name.lower()

        if provider_lower not in self._registry:
            raise ProviderNotFoundError(
                provider=provider,
                available_providers=list(self._registry.keys()),
            )

        provider_endpoints: MappingProxyType[str, EndpointDefinition[BaseModel]] = (
            self._registry[provider_lower]
        )

        if endpoint_lower not in provider_endpoints:
            raise EndpointNotFoundError(
                provider=provider,
                endpoint_name=endpoint_name,
                available_endpoints=list(provider_endpoints.keys()),
            )

        return provider_endpoints[endpoint_lower]

    def has(self, provider: str, endpoint_name: str) -> bool:
        """
        Check if an endpoint exists in the registry.

        Args:
            provider: Provider name (case-insensitive).
            endpoint_name: Endpoint name (case-insensitive).

        Returns:
            True if endpoint exists, False otherwise.
        """
        provider_lower: str = provider.lower()
        endpoint_lower: str = endpoint_name.lower()

        if provider_lower not in self._registry:
            return False

        return endpoint_lower in self._registry[provider_lower]

    # -------------------------------------------------------------------------
    # Discovery Methods
    # -------------------------------------------------------------------------

    def list_providers(self) -> list[str]:
        """
        Get list of all registered provider names.

        Returns:
            Sorted list of provider names (lowercase).
        """
        return sorted(self._registry.keys())

    def list_endpoints(self, provider: str) -> list[str]:
        """
        Get list of all endpoint names for a provider.

        Args:
            provider: Provider name (case-insensitive).

        Returns:
            Sorted list of endpoint names (lowercase).

        Raises:
            ProviderNotFoundError: If provider doesn't exist.
        """
        provider_lower: str = provider.lower()

        if provider_lower not in self._registry:
            raise ProviderNotFoundError(
                provider=provider,
                available_providers=list(self._registry.keys()),
            )

        return sorted(self._registry[provider_lower].keys())

    def get_all_endpoints(
        self,
        provider: str,
    ) -> dict[str, EndpointDefinition[BaseModel]]:
        """
        Get all endpoints for a provider as a dictionary.

        Returns a mutable copy of the internal endpoint mapping. Modifying
        the returned dict does not affect the registry.

        Args:
            provider: Provider name (case-insensitive).

        Returns:
            Dictionary mapping endpoint names to definitions.

        Raises:
            ProviderNotFoundError: If provider doesn't exist.
        """
        provider_lower: str = provider.lower()

        if provider_lower not in self._registry:
            raise ProviderNotFoundError(
                provider=provider,
                available_providers=list(self._registry.keys()),
            )

        # Return a mutable copy (MappingProxyType -> dict)
        return dict(self._registry[provider_lower])

    def find_by_path(self, path_fragment: str) -> list[tuple[str, str]]:
        """
        Find endpoints whose API path contains the given fragment.

        Useful for discovering which endpoint corresponds to a known API path.

        Args:
            path_fragment: Substring to search for in endpoint paths.
                Case-sensitive since API paths are case-sensitive.

        Returns:
            Sorted list of (provider, endpoint_name) tuples for matching endpoints.
            Empty list if no matches found.

        Example:
            >>> registry = EndpointRegistry()
            >>> matches = registry.find_by_path('/vehicles')
            >>> print(matches)
            [('motive', 'vehicles'), ('samsara', 'vehicles')]
        """
        matches: list[tuple[str, str]] = []

        for provider_name in self._registry:
            for endpoint_name, endpoint in self._registry[provider_name].items():
                if path_fragment in endpoint.endpoint_path:
                    matches.append((provider_name, endpoint_name))

        return sorted(matches)

    def describe(self, provider: str | None = None) -> str:
        """
        Generate human-readable description of registry contents.

        Useful for debugging, documentation, or CLI help output.

        Args:
            provider: Provider name to describe. If None, describes all providers.

        Returns:
            Formatted multi-line string with endpoint details.

        Raises:
            ProviderNotFoundError: If specified provider doesn't exist.
        """
        lines: list[str] = []

        if provider is not None:
            provider_lower: str = provider.lower()
            if provider_lower not in self._registry:
                raise ProviderNotFoundError(
                    provider=provider,
                    available_providers=list(self._registry.keys()),
                )
            providers_to_describe: list[str] = [provider_lower]
        else:
            providers_to_describe = self.list_providers()

        for provider_name in providers_to_describe:
            endpoints: MappingProxyType[str, EndpointDefinition[BaseModel]] = (
                self._registry[provider_name]
            )
            lines.append(f'Provider: {provider_name} ({len(endpoints)} endpoints)')
            lines.append('')

            for endpoint_name in sorted(endpoints.keys()):
                endpoint: EndpointDefinition[BaseModel] = endpoints[endpoint_name]
                pagination_indicator: str = (
                    ' [paginated]' if endpoint.is_paginated else ''
                )

                lines.append(f'  {endpoint_name}{pagination_indicator}')
                lines.append(f'    Path: {endpoint.endpoint_path}')
                lines.append(f'    Description: {endpoint.description}')
                lines.append('')

        return '\n'.join(lines).rstrip()

    # -------------------------------------------------------------------------
    # Internal Registration
    # -------------------------------------------------------------------------

    def _build_motive_endpoints(
        self,
    ) -> dict[str, EndpointDefinition[BaseModel]]:
        """Build Motive endpoint mapping with normalized (lowercase) names."""
        motive_endpoints: dict[str, MotiveEndpointDefinition[Any, Any]] = (
            MotiveEndpoints.get_all_endpoints()
        )

        return {name.lower(): endpoint for name, endpoint in motive_endpoints.items()}

    def _build_samsara_endpoints(
        self,
    ) -> dict[str, EndpointDefinition[BaseModel]]:
        """Build Samsara endpoint mapping with normalized (lowercase) names."""
        samsara_endpoints: dict[str, SamsaraEndpointDefinition[Any, Any]] = (
            SamsaraEndpoints.get_all_endpoints()
        )

        return {name.lower(): endpoint for name, endpoint in samsara_endpoints.items()}
