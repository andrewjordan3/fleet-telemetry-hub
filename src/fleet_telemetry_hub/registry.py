# fleet_telemetry_hub/registry.py
"""
Unified endpoint registry for dynamic endpoint access across providers.

This module provides a centralized registry that allows string-based endpoint
lookup, making it easy to work with endpoints dynamically without hardcoding
provider-specific class references.

Example:
    >>> registry = EndpointRegistry()
    >>>
    >>> # Get endpoint by provider and name
    >>> endpoint = registry.get("motive", "vehicles")
    >>>
    >>> # List all endpoints for a provider
    >>> motive_endpoints = registry.list_endpoints("motive")
    >>>
    >>> # Check if endpoint exists
    >>> if registry.has("samsara", "drivers"):
    ...     endpoint = registry.get("samsara", "drivers")
"""

import logging
from typing import Any, Literal

from .models import (
    EndpointDefinition,
    MotiveEndpointDefinition,
    MotiveEndpoints,
    SamsaraEndpointDefinition,
    SamsaraEndpoints,
)

logger: logging.Logger = logging.getLogger(__name__)


class EndpointNotFoundError(Exception):
    """Raised when requested endpoint doesn't exist in registry."""

    def __init__(self, provider: str, endpoint_name: str) -> None:
        super().__init__(
            f"Endpoint '{endpoint_name}' not found for provider '{provider}'"
        )
        self.provider: str = provider
        self.endpoint_name: str = endpoint_name


class ProviderNotFoundError(Exception):
    """Raised when requested provider doesn't exist in registry."""

    def __init__(self, provider: str) -> None:
        super().__init__(f"Provider '{provider}' not found in registry")
        self.provider: str = provider


class EndpointRegistry:
    """
    Centralized registry for all API endpoints across providers.

    Provides string-based access to endpoint definitions without requiring
    direct references to provider-specific classes like MotiveEndpoints
    or SamsaraEndpoints.

    Thread-safe and immutable after initialization.

    Example:
        >>> registry = EndpointRegistry()
        >>>
        >>> # Dynamic endpoint access
        >>> endpoint = registry.get("motive", "vehicles")
        >>> with TelemetryClient(credentials) as client:
        ...     for vehicle in client.fetch_all(endpoint):
        ...         print(vehicle.number)
        >>>
        >>> # Discover available endpoints
        >>> for provider_name in registry.list_providers():
        ...     print(f"\n{provider_name}:")
        ...     for endpoint_name in registry.list_endpoints(provider_name):
        ...         endpoint = registry.get(provider_name, endpoint_name)
        ...         print(f"  - {endpoint_name}: {endpoint.description}")
    """

    _instance: 'EndpointRegistry | None' = None

    def __init__(self) -> None:
        """Initialize registry with all known providers and endpoints."""
        self._registry: dict[str, dict[str, EndpointDefinition[Any, Any]]] = {
            'motive': self._register_motive_endpoints(),
            'samsara': self._register_samsara_endpoints(),
        }

        logger.info(
            'EndpointRegistry initialized with %d endpoints across %d providers',
            self._count_total_endpoints(),
            len(self._registry),
        )

    @classmethod
    def instance(cls) -> 'EndpointRegistry':
        """
        Get the shared registry instance (singleton pattern).

        This is a convenience method that returns a pre-initialized global
        registry. You can also create your own instances with EndpointRegistry().

        Returns:
            The shared EndpointRegistry instance.

        Example:
            >>> from fleet_telemetry_hub import EndpointRegistry
            >>>
            >>> registry = EndpointRegistry.instance()
            >>> endpoint = registry.get("motive", "vehicles")
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _register_motive_endpoints(
        self,
    ) -> dict[str, EndpointDefinition[Any, Any]]:
        """Register all Motive endpoints with normalized names."""
        motive_endpoints: dict[str, MotiveEndpointDefinition[Any, Any]] = (
            MotiveEndpoints.get_all_endpoints()
        )

        # Normalize names: VEHICLES -> vehicles, VEHICLE_LOCATIONS -> vehicle_locations
        return {name.lower(): endpoint for name, endpoint in motive_endpoints.items()}

    def _register_samsara_endpoints(
        self,
    ) -> dict[str, EndpointDefinition[Any, Any]]:
        """Register all Samsara endpoints with normalized names."""
        samsara_endpoints: dict[str, SamsaraEndpointDefinition[Any, Any]] = (
            SamsaraEndpoints.get_all_endpoints()
        )

        return {name.lower(): endpoint for name, endpoint in samsara_endpoints.items()}

    def get(
        self,
        provider: str,
        endpoint_name: str,
    ) -> EndpointDefinition[Any, Any]:
        """
        Get an endpoint definition by provider and name.

        Args:
            provider: Provider name (case-insensitive: 'motive', 'samsara').
            endpoint_name: Endpoint name (case-insensitive: 'vehicles', 'drivers').

        Returns:
            The endpoint definition ready for use with TelemetryClient.

        Raises:
            ProviderNotFoundError: If provider doesn't exist.
            EndpointNotFoundError: If endpoint doesn't exist for provider.

        Example:
            >>> registry = EndpointRegistry()
            >>> endpoint = registry.get("motive", "vehicles")
            >>> print(endpoint.description)
            'List all vehicles in the fleet with current driver and device info'
        """
        provider_lower: str = provider.lower()
        endpoint_lower: str = endpoint_name.lower()

        if provider_lower not in self._registry:
            raise ProviderNotFoundError(provider)

        provider_endpoints: dict[str, EndpointDefinition[Any, Any]] = self._registry[
            provider_lower
        ]

        if endpoint_lower not in provider_endpoints:
            raise EndpointNotFoundError(provider, endpoint_name)

        return provider_endpoints[endpoint_lower]

    def has(self, provider: str, endpoint_name: str) -> bool:
        """
        Check if an endpoint exists in the registry.

        Args:
            provider: Provider name (case-insensitive).
            endpoint_name: Endpoint name (case-insensitive).

        Returns:
            True if endpoint exists, False otherwise.

        Example:
            >>> registry = EndpointRegistry()
            >>> if registry.has("samsara", "drivers"):
            ...     endpoint = registry.get("samsara", "drivers")
        """
        provider_lower: str = provider.lower()
        endpoint_lower: str = endpoint_name.lower()

        return (
            provider_lower in self._registry
            and endpoint_lower in self._registry[provider_lower]
        )

    def list_providers(self) -> list[str]:
        """
        Get list of all registered provider names.

        Returns:
            List of provider names in lowercase.

        Example:
            >>> registry = EndpointRegistry()
            >>> print(registry.list_providers())
            ['motive', 'samsara']
        """
        return list(self._registry.keys())

    def list_endpoints(self, provider: str) -> list[str]:
        """
        Get list of all endpoint names for a provider.

        Args:
            provider: Provider name (case-insensitive).

        Returns:
            List of endpoint names in lowercase.

        Raises:
            ProviderNotFoundError: If provider doesn't exist.

        Example:
            >>> registry = EndpointRegistry()
            >>> endpoints = registry.list_endpoints("motive")
            >>> print(endpoints)
            ['vehicles', 'vehicle_locations', 'groups', 'users']
        """
        provider_lower: str = provider.lower()

        if provider_lower not in self._registry:
            raise ProviderNotFoundError(provider)

        return list(self._registry[provider_lower].keys())

    def get_all_endpoints(
        self,
        provider: str,
    ) -> dict[str, EndpointDefinition[Any, Any]]:
        """
        Get all endpoints for a provider as a dictionary.

        Args:
            provider: Provider name (case-insensitive).

        Returns:
            Dictionary mapping endpoint names to definitions.

        Raises:
            ProviderNotFoundError: If provider doesn't exist.

        Example:
            >>> registry = EndpointRegistry()
            >>> motive_endpoints = registry.get_all_endpoints("motive")
            >>> for name, endpoint in motive_endpoints.items():
            ...     print(f"{name}: {endpoint.description}")
        """
        provider_lower: str = provider.lower()

        if provider_lower not in self._registry:
            raise ProviderNotFoundError(provider)

        return self._registry[provider_lower].copy()

    def find_by_path(self, endpoint_path: str) -> list[tuple[str, str]]:
        """
        Find endpoints by their API path.

        Useful for discovering which endpoint to use for a given API path.

        Args:
            endpoint_path: Full or partial endpoint path (e.g., '/v1/vehicles').

        Returns:
            List of (provider, endpoint_name) tuples matching the path.

        Example:
            >>> registry = EndpointRegistry()
            >>> matches = registry.find_by_path('/vehicles')
            >>> print(matches)
            [('motive', 'vehicles'), ('samsara', 'vehicles')]
        """
        matches: list[tuple[str, str]] = []

        for provider_name, endpoints in self._registry.items():
            for endpoint_name, endpoint in endpoints.items():
                if endpoint_path in endpoint.endpoint_path:
                    matches.append((provider_name, endpoint_name))

        return matches

    def describe(self, provider: str | None = None) -> str:
        """
        Generate human-readable description of registry contents.

        Args:
            provider: Optional provider name to describe. If None, describes all.

        Returns:
            Formatted string with endpoint descriptions.

        Example:
            >>> registry = EndpointRegistry()
            >>> print(registry.describe("motive"))
            Provider: motive (4 endpoints)
              vehicles: List all vehicles in the fleet with current driver and device info
              vehicle_locations: Get location history (breadcrumbs) for a specific vehicle
              groups: List all groups (organizational units) in the company
              users: List all users (drivers and admins) in the company
        """
        lines: list[str] = []

        providers: list[str] = [provider.lower()] if provider else self.list_providers()

        for provider_name in providers:
            if provider_name not in self._registry:
                lines.append(f'Provider: {provider_name} (not found)')
                continue

            endpoints: dict[str, EndpointDefinition[Any, Any]] = self._registry[
                provider_name
            ]
            lines.append(f'\nProvider: {provider_name} ({len(endpoints)} endpoints)')

            for endpoint_name, endpoint in sorted(endpoints.items()):
                # Show path and description
                path: str = endpoint.endpoint_path
                desc: str = endpoint.description
                paginated: Literal[' [paginated]'] | Literal[''] = (
                    ' [paginated]' if endpoint.is_paginated else ''
                )
                lines.append(f'  {endpoint_name}:{paginated}')
                lines.append(f'    Path: {path}')
                lines.append(f'    Description: {desc}')

        return '\n'.join(lines)

    def _count_total_endpoints(self) -> int:
        """Count total number of registered endpoints across all providers."""
        return sum(len(endpoints) for endpoints in self._registry.values())
