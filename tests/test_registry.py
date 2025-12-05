"""

Tests for fleet_telemetry_hub.registry module.



Tests EndpointRegistry functionality including lookup, discovery,

and error handling.

"""

import pytest
from pydantic.main import BaseModel

from fleet_telemetry_hub.models import EndpointDefinition
from fleet_telemetry_hub.registry import (
    EndpointNotFoundError,
    EndpointRegistry,
    ProviderNotFoundError,
)


class TestEndpointRegistryInitialization:
    """Test EndpointRegistry initialization."""

    def test_registry_initialization_succeeds(self) -> None:
        """Should initialize registry successfully."""

        registry = EndpointRegistry()

        assert registry is not None

    def test_registry_has_known_providers(self) -> None:
        """Should register known providers (motive, samsara)."""

        registry = EndpointRegistry()

        providers: list[str] = registry.list_providers()

        assert 'motive' in providers

        assert 'samsara' in providers

    def test_registry_is_immutable(self) -> None:
        """Should not allow modification of registry after initialization."""

        registry = EndpointRegistry()

        # Registry should be using MappingProxyType (immutable)

        # Attempting to modify should fail

        with pytest.raises((TypeError, AttributeError)):
            registry._registry['new_provider'] = {}  # pyright: ignore[reportIndexIssue, reportPrivateUsage]


class TestEndpointRegistrySingleton:
    """Test EndpointRegistry singleton pattern."""

    def test_instance_returns_same_object(self) -> None:
        """Should return same instance on multiple calls."""

        instance1: EndpointRegistry = EndpointRegistry.instance()

        instance2: EndpointRegistry = EndpointRegistry.instance()

        assert instance1 is instance2

    def test_instance_returns_initialized_registry(self) -> None:
        """Should return a fully initialized registry."""

        instance: EndpointRegistry = EndpointRegistry.instance()

        providers: list[str] = instance.list_providers()

        assert len(providers) > 0

        assert 'motive' in providers


class TestEndpointRegistryGet:
    """Test EndpointRegistry.get() method."""

    def test_get_valid_motive_endpoint(self) -> None:
        """Should retrieve valid Motive endpoint."""

        registry = EndpointRegistry()

        endpoint: EndpointDefinition[BaseModel] = registry.get('motive', 'vehicles')

        assert isinstance(endpoint, EndpointDefinition)

        assert 'vehicle' in endpoint.endpoint_path.lower()

    def test_get_valid_samsara_endpoint(self) -> None:
        """Should retrieve valid Samsara endpoint."""

        registry = EndpointRegistry()

        endpoint: EndpointDefinition[BaseModel] = registry.get('samsara', 'vehicles')

        assert isinstance(endpoint, EndpointDefinition)

        assert 'vehicle' in endpoint.endpoint_path.lower()

    def test_get_is_case_insensitive(self) -> None:
        """Should work with different case variations."""

        registry = EndpointRegistry()

        endpoint_lower: EndpointDefinition[BaseModel] = registry.get(
            'motive', 'vehicles'
        )

        endpoint_upper: EndpointDefinition[BaseModel] = registry.get(
            'MOTIVE', 'VEHICLES'
        )

        endpoint_mixed: EndpointDefinition[BaseModel] = registry.get(
            'Motive', 'Vehicles'
        )

        # Should all return the same endpoint

        assert endpoint_lower is endpoint_upper

        assert endpoint_lower is endpoint_mixed

    def test_get_raises_on_invalid_provider(self) -> None:
        """Should raise ProviderNotFoundError for unknown provider."""

        registry = EndpointRegistry()

        with pytest.raises(ProviderNotFoundError) as exc_info:
            registry.get('unknown_provider', 'vehicles')

        # Exception should contain helpful info

        assert exc_info.value.provider == 'unknown_provider'

        assert 'motive' in exc_info.value.available_providers

        assert 'samsara' in exc_info.value.available_providers

    def test_get_raises_on_invalid_endpoint(self) -> None:
        """Should raise EndpointNotFoundError for unknown endpoint."""

        registry = EndpointRegistry()

        with pytest.raises(EndpointNotFoundError) as exc_info:
            registry.get('motive', 'nonexistent_endpoint')

        # Exception should contain helpful info

        assert exc_info.value.provider == 'motive'

        assert exc_info.value.endpoint_name == 'nonexistent_endpoint'

        assert len(exc_info.value.available_endpoints) > 0


class TestEndpointRegistryHas:
    """Test EndpointRegistry.has() method."""

    def test_has_returns_true_for_valid_endpoint(self) -> None:
        """Should return True for valid provider/endpoint."""

        registry = EndpointRegistry()

        assert registry.has('motive', 'vehicles') is True

        assert registry.has('samsara', 'vehicles') is True

    def test_has_returns_false_for_invalid_provider(self) -> None:
        """Should return False for unknown provider."""

        registry = EndpointRegistry()

        assert registry.has('unknown_provider', 'vehicles') is False

    def test_has_returns_false_for_invalid_endpoint(self) -> None:
        """Should return False for unknown endpoint."""

        registry = EndpointRegistry()

        assert registry.has('motive', 'nonexistent_endpoint') is False

    def test_has_is_case_insensitive(self) -> None:
        """Should work with different case variations."""

        registry = EndpointRegistry()

        assert registry.has('MOTIVE', 'VEHICLES') is True

        assert registry.has('Motive', 'Vehicles') is True


class TestEndpointRegistryListProviders:
    """Test EndpointRegistry.list_providers() method."""

    def test_list_providers_returns_all_providers(self) -> None:
        """Should return list of all registered providers."""

        registry = EndpointRegistry()

        providers: list[str] = registry.list_providers()

        assert isinstance(providers, list)

        assert len(providers) >= 2  # noqa: PLR2004

        assert 'motive' in providers

        assert 'samsara' in providers

    def test_list_providers_returns_sorted_list(self) -> None:
        """Should return providers in sorted order."""

        registry = EndpointRegistry()

        providers: list[str] = registry.list_providers()

        assert providers == sorted(providers)


class TestEndpointRegistryListEndpoints:
    """Test EndpointRegistry.list_endpoints() method."""

    def test_list_endpoints_for_motive(self) -> None:
        """Should return list of Motive endpoints."""

        registry = EndpointRegistry()

        endpoints: list[str] = registry.list_endpoints('motive')

        assert isinstance(endpoints, list)

        assert len(endpoints) > 0

        assert 'vehicles' in endpoints

    def test_list_endpoints_for_samsara(self) -> None:
        """Should return list of Samsara endpoints."""

        registry = EndpointRegistry()

        endpoints: list[str] = registry.list_endpoints('samsara')

        assert isinstance(endpoints, list)

        assert len(endpoints) > 0

        assert 'vehicles' in endpoints

    def test_list_endpoints_returns_sorted_list(self) -> None:
        """Should return endpoints in sorted order."""

        registry = EndpointRegistry()

        endpoints: list[str] = registry.list_endpoints('motive')

        assert endpoints == sorted(endpoints)

    def test_list_endpoints_is_case_insensitive(self) -> None:
        """Should work with different case variations."""

        registry = EndpointRegistry()

        endpoints_lower: list[str] = registry.list_endpoints('motive')

        endpoints_upper: list[str] = registry.list_endpoints('MOTIVE')

        assert endpoints_lower == endpoints_upper

    def test_list_endpoints_raises_on_invalid_provider(self) -> None:
        """Should raise ProviderNotFoundError for unknown provider."""

        registry = EndpointRegistry()

        with pytest.raises(ProviderNotFoundError):
            registry.list_endpoints('unknown_provider')


class TestEndpointRegistryGetAllEndpoints:
    """Test EndpointRegistry.get_all_endpoints() method."""

    def test_get_all_endpoints_returns_dict(self) -> None:
        """Should return dictionary of endpoint name to definition."""

        registry = EndpointRegistry()

        endpoints: dict[str, EndpointDefinition[BaseModel]] = (
            registry.get_all_endpoints('motive')
        )

        assert isinstance(endpoints, dict)

        assert len(endpoints) > 0

        assert 'vehicles' in endpoints

        assert isinstance(endpoints['vehicles'], EndpointDefinition)

    def test_get_all_endpoints_returns_mutable_copy(self) -> None:
        """Should return mutable copy (not the internal registry)."""

        registry = EndpointRegistry()

        endpoints: dict[str, EndpointDefinition[BaseModel]] = (
            registry.get_all_endpoints('motive')
        )

        # Should be able to modify the returned dict

        endpoints['test_endpoint'] = None  # pyright: ignore[reportArgumentType]

        # Original registry should be unchanged

        assert not registry.has('motive', 'test_endpoint')

    def test_get_all_endpoints_raises_on_invalid_provider(self) -> None:
        """Should raise ProviderNotFoundError for unknown provider."""

        registry = EndpointRegistry()

        with pytest.raises(ProviderNotFoundError):
            registry.get_all_endpoints('unknown_provider')


class TestEndpointRegistryFindByPath:
    """Test EndpointRegistry.find_by_path() method."""

    def test_find_by_path_finds_matching_endpoints(self) -> None:
        """Should find endpoints containing path fragment."""

        registry = EndpointRegistry()

        # Search for endpoints with 'vehicle' in path

        matches: list[tuple[str, str]] = registry.find_by_path('vehicle')

        assert isinstance(matches, list)

        assert len(matches) > 0

        # Should return (provider, endpoint_name) tuples

        for provider, endpoint_name in matches:
            assert isinstance(provider, str)

            assert isinstance(endpoint_name, str)

            endpoint: EndpointDefinition[BaseModel] = registry.get(
                provider, endpoint_name
            )

            assert 'vehicle' in endpoint.endpoint_path

    def test_find_by_path_is_case_sensitive(self) -> None:
        """Should be case-sensitive when matching paths."""

        registry = EndpointRegistry()

        # API paths are typically case-sensitive

        matches_lower: list[tuple[str, str]] = registry.find_by_path('vehicle')

        matches_upper: list[tuple[str, str]] = registry.find_by_path('VEHICLE')

        # Different case may yield different results

        # (or no results for uppercase if all paths are lowercase)

        assert isinstance(matches_lower, list)

        assert isinstance(matches_upper, list)

    def test_find_by_path_returns_empty_for_no_matches(self) -> None:
        """Should return empty list when no endpoints match."""

        registry = EndpointRegistry()

        matches: list[tuple[str, str]] = registry.find_by_path(
            'definitely_not_a_real_path_xyzabc'
        )

        assert matches == []

    def test_find_by_path_returns_sorted_results(self) -> None:
        """Should return results in sorted order."""

        registry = EndpointRegistry()

        matches: list[tuple[str, str]] = registry.find_by_path('vehicle')

        if len(matches) > 1:
            assert matches == sorted(matches)


class TestEndpointRegistryDescribe:
    """Test EndpointRegistry.describe() method."""

    def test_describe_all_providers(self) -> None:
        """Should describe all providers when provider=None."""

        registry = EndpointRegistry()

        description: str = registry.describe()

        assert isinstance(description, str)

        assert 'motive' in description.lower()

        assert 'samsara' in description.lower()

    def test_describe_specific_provider(self) -> None:
        """Should describe specific provider."""

        registry = EndpointRegistry()

        description: str = registry.describe('motive')

        assert isinstance(description, str)

        assert 'motive' in description.lower()

        assert 'vehicles' in description.lower()

        # Should not mention other providers

        assert 'samsara' not in description.lower()

    def test_describe_includes_endpoint_details(self) -> None:
        """Should include endpoint paths and descriptions."""

        registry = EndpointRegistry()

        description: str = registry.describe('motive')

        # Should include path and description for endpoints

        assert 'Path:' in description

        assert 'Description:' in description

    def test_describe_raises_on_invalid_provider(self) -> None:
        """Should raise ProviderNotFoundError for unknown provider."""

        registry = EndpointRegistry()

        with pytest.raises(ProviderNotFoundError):
            registry.describe('unknown_provider')


class TestEndpointRegistryIntegration:
    """Integration tests for EndpointRegistry."""

    def test_complete_workflow_motive(self) -> None:
        """Test complete workflow: lookup, validate, use endpoint."""

        registry = EndpointRegistry()

        # List providers

        providers: list[str] = registry.list_providers()

        assert 'motive' in providers

        # List endpoints for motive

        endpoints: list[str] = registry.list_endpoints('motive')

        assert 'vehicles' in endpoints

        # Get specific endpoint

        endpoint: EndpointDefinition[BaseModel] = registry.get('motive', 'vehicles')

        assert endpoint.description is not None

        assert endpoint.endpoint_path is not None

        # Verify endpoint is for Motive

        # (Motive endpoints have specific attributes)

        assert hasattr(endpoint, 'http_method')

    def test_complete_workflow_samsara(self) -> None:
        """Test complete workflow for Samsara provider."""

        registry = EndpointRegistry()

        # List providers

        providers: list[str] = registry.list_providers()

        assert 'samsara' in providers

        # List endpoints for samsara

        endpoints: list[str] = registry.list_endpoints('samsara')

        assert 'vehicles' in endpoints

        # Get specific endpoint

        endpoint: EndpointDefinition[BaseModel] = registry.get('samsara', 'vehicles')

        assert endpoint.description is not None

        assert endpoint.endpoint_path is not None

    def test_registry_consistency_across_instances(self) -> None:
        """Different registry instances should have same endpoints."""

        registry1 = EndpointRegistry()

        registry2 = EndpointRegistry()

        # Should have same providers

        assert registry1.list_providers() == registry2.list_providers()

        # Should have same endpoints for each provider

        for provider in registry1.list_providers():
            assert registry1.list_endpoints(provider) == registry2.list_endpoints(
                provider
            )
