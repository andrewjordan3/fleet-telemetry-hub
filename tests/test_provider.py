"""

Tests for fleet_telemetry_hub.provider module.



Tests Provider and ProviderManager functionality.

"""

import pytest
from pydantic.main import BaseModel

from fleet_telemetry_hub.client import TelemetryClient
from fleet_telemetry_hub.config import TelemetryConfig
from fleet_telemetry_hub.config.config_models import ProviderConfig
from fleet_telemetry_hub.models.shared_response_models import EndpointDefinition
from fleet_telemetry_hub.provider import (
    Provider,
    ProviderConfigurationError,
    ProviderManager,
)
from fleet_telemetry_hub.registry import EndpointNotFoundError, ProviderNotFoundError


class TestProviderInitialization:
    """Test Provider initialization."""

    def test_from_config_succeeds_for_valid_provider(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should initialize provider from valid config."""

        provider: Provider = Provider.from_config('motive', telemetry_config)

        assert provider.name == 'motive'

        assert provider.credentials.base_url == 'https://api.gomotive.com'

        assert provider.credentials.api_key.get_secret_value() == 'test_motive_api_key'

    def test_from_config_normalizes_provider_name(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should normalize provider name to lowercase."""

        provider: Provider = Provider.from_config('MOTIVE', telemetry_config)

        assert provider.name == 'motive'

    def test_from_config_raises_on_missing_provider(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should raise ProviderConfigurationError for unconfigured provider."""

        with pytest.raises(ProviderConfigurationError) as exc_info:
            Provider.from_config('unknown_provider', telemetry_config)

        assert exc_info.value.provider_name == 'unknown_provider'

        assert 'motive' in exc_info.value.available_providers  # pyright: ignore[reportOperatorIssue]

    def test_from_config_raises_on_unknown_provider_in_registry(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should raise ProviderNotFoundError if provider not in registry."""

        # Add a provider to config that's not in registry

        telemetry_config.providers['unknown'] = telemetry_config.providers['motive']

        with pytest.raises(ProviderNotFoundError):
            Provider.from_config('unknown', telemetry_config)


class TestProviderEndpointAccess:
    """Test Provider endpoint access methods."""

    def test_endpoint_returns_valid_endpoint(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should return endpoint definition by name."""

        provider: Provider = Provider.from_config('motive', telemetry_config)

        endpoint: EndpointDefinition[BaseModel] = provider.endpoint('vehicles')

        assert endpoint is not None

        assert endpoint.description is not None

    def test_endpoint_is_case_insensitive(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should work with different case variations."""

        provider: Provider = Provider.from_config('motive', telemetry_config)

        endpoint_lower: EndpointDefinition[BaseModel] = provider.endpoint('vehicles')

        endpoint_upper: EndpointDefinition[BaseModel] = provider.endpoint('VEHICLES')

        assert endpoint_lower is endpoint_upper

    def test_endpoint_raises_on_invalid_endpoint(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should raise EndpointNotFoundError for invalid endpoint."""

        provider: Provider = Provider.from_config('motive', telemetry_config)

        with pytest.raises(EndpointNotFoundError):
            provider.endpoint('nonexistent_endpoint')

    def test_list_endpoints_returns_all_endpoints(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should list all endpoints for provider."""

        provider: Provider = Provider.from_config('motive', telemetry_config)

        endpoints: list[str] = provider.list_endpoints()

        assert isinstance(endpoints, list)

        assert len(endpoints) > 0

        assert 'vehicles' in endpoints

    def test_has_endpoint_returns_true_for_valid_endpoint(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should return True for valid endpoint."""

        provider: Provider = Provider.from_config('motive', telemetry_config)

        assert provider.has_endpoint('vehicles') is True

    def test_has_endpoint_returns_false_for_invalid_endpoint(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should return False for invalid endpoint."""

        provider: Provider = Provider.from_config('motive', telemetry_config)

        assert provider.has_endpoint('nonexistent_endpoint') is False


class TestProviderClientManagement:
    """Test Provider client creation and management."""

    def test_client_creates_telemetry_client(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should create TelemetryClient instance."""

        provider: Provider = Provider.from_config('motive', telemetry_config)

        client: TelemetryClient = provider.client()

        assert isinstance(client, TelemetryClient)

        # Clean up

        client.close()

    def test_client_context_manager(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should work as context manager."""

        provider: Provider = Provider.from_config('motive', telemetry_config)

        with provider.client() as client:
            assert client is not None


class TestProviderDescribe:
    """Test Provider.describe() method."""

    def test_describe_returns_endpoint_descriptions(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should return formatted endpoint descriptions."""

        provider: Provider = Provider.from_config('motive', telemetry_config)

        description: str = provider.describe()

        assert isinstance(description, str)

        assert 'motive' in description.lower()

        assert 'vehicles' in description.lower()


class TestProviderRepresentation:
    """Test Provider string representation."""

    def test_repr_includes_key_info(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should include provider name and base URL in repr."""

        provider: Provider = Provider.from_config('motive', telemetry_config)

        repr_str: str = repr(provider)

        assert 'motive' in repr_str

        assert 'https://api.gomotive.com' in repr_str


class TestProviderManagerInitialization:
    """Test ProviderManager initialization."""

    def test_from_config_creates_manager(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should create ProviderManager from config."""

        manager: ProviderManager = ProviderManager.from_config(telemetry_config)

        assert manager is not None

        assert len(manager) > 0

    def test_from_config_loads_enabled_providers_only(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should load only enabled providers."""

        # Disable samsara

        telemetry_config.providers['samsara'].enabled = False

        manager: ProviderManager = ProviderManager.from_config(telemetry_config)

        assert manager.has('motive') is True

        assert manager.has('samsara') is False

    def test_from_config_skips_unknown_providers(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should skip providers not in registry."""

        # Create a dummy config with the required fields
        # We set enabled=True to prove that the Manager skips it
        # based on the NAME ('unknown'), not the enabled flag.
        dummy_config = ProviderConfig(
            base_url='https://ignore.me',
            api_key='ignore_me',  # pyright: ignore[reportArgumentType]
            enabled=True,
            request_timeout=(10, 30),
            max_retries=3,
            retry_backoff_factor=1.5,
            verify_ssl=True,
            rate_limit_requests_per_second=10,
        )

        # Assign it to the dictionary
        telemetry_config.providers['unknown'] = dummy_config

        # Should not raise, just skip unknown provider

        manager: ProviderManager = ProviderManager.from_config(telemetry_config)

        assert not manager.has('unknown')


class TestProviderManagerGet:
    """Test ProviderManager.get() method."""

    def test_get_returns_provider_when_available(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should return provider instance when available."""

        manager: ProviderManager = ProviderManager.from_config(telemetry_config)

        provider: Provider | None = manager.get('motive')

        assert provider is not None

        assert isinstance(provider, Provider)

        assert provider.name == 'motive'

    def test_get_returns_none_when_not_available(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should return None for unavailable provider."""

        # Disable all providers

        for provider_config in telemetry_config.providers.values():
            provider_config.enabled = False

        manager: ProviderManager = ProviderManager.from_config(telemetry_config)

        provider: Provider | None = manager.get('motive')

        assert provider is None


class TestProviderManagerRequire:
    """Test ProviderManager.require() method."""

    def test_require_returns_provider_when_available(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should return provider instance when available."""

        manager: ProviderManager = ProviderManager.from_config(telemetry_config)

        provider: Provider = manager.require('motive')

        assert isinstance(provider, Provider)

        assert provider.name == 'motive'

    def test_require_raises_when_not_available(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should raise KeyError when provider not available."""

        # Disable motive

        telemetry_config.providers['motive'].enabled = False

        manager: ProviderManager = ProviderManager.from_config(telemetry_config)

        with pytest.raises(KeyError) as exc_info:
            manager.require('motive')

        assert 'motive' in str(exc_info.value)


class TestProviderManagerHas:
    """Test ProviderManager.has() method."""

    def test_has_returns_true_for_enabled_provider(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should return True for enabled provider."""

        manager: ProviderManager = ProviderManager.from_config(telemetry_config)

        assert manager.has('motive') is True

    def test_has_returns_false_for_disabled_provider(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should return False for disabled provider."""

        telemetry_config.providers['motive'].enabled = False

        manager: ProviderManager = ProviderManager.from_config(telemetry_config)

        assert manager.has('motive') is False


class TestProviderManagerListProviders:
    """Test ProviderManager.list_providers() method."""

    def test_list_providers_returns_enabled_providers(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should list all enabled providers."""

        manager: ProviderManager = ProviderManager.from_config(telemetry_config)

        providers: list[str] = manager.list_providers()

        assert isinstance(providers, list)

        assert 'motive' in providers

        assert 'samsara' in providers

    def test_list_providers_returns_sorted_list(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should return providers in sorted order."""

        manager: ProviderManager = ProviderManager.from_config(telemetry_config)

        providers: list[str] = manager.list_providers()

        assert providers == sorted(providers)


class TestProviderManagerItems:
    """Test ProviderManager.items() method."""

    def test_items_yields_provider_tuples(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should yield (name, provider) tuples."""

        manager: ProviderManager = ProviderManager.from_config(telemetry_config)

        items: list[tuple[str, Provider]] = list(manager.items())

        assert len(items) > 0

        for name, provider in items:
            assert isinstance(name, str)

            assert isinstance(provider, Provider)

            assert provider.name == name

    def test_items_yields_in_sorted_order(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should yield providers in sorted order."""

        manager: ProviderManager = ProviderManager.from_config(telemetry_config)

        names: list[str] = [name for name, _ in manager.items()]

        assert names == sorted(names)


class TestProviderManagerMagicMethods:
    """Test ProviderManager magic methods."""

    def test_len_returns_provider_count(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should return number of enabled providers."""

        manager: ProviderManager = ProviderManager.from_config(telemetry_config)

        assert len(manager) == 2  # motive and samsara  # noqa: PLR2004

    def test_bool_true_when_providers_enabled(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should be True when providers are enabled."""

        manager: ProviderManager = ProviderManager.from_config(telemetry_config)

        assert bool(manager) is True

    def test_bool_false_when_no_providers(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should be False when no providers enabled."""

        # Disable all providers

        for provider_config in telemetry_config.providers.values():
            provider_config.enabled = False

        manager: ProviderManager = ProviderManager.from_config(telemetry_config)

        assert bool(manager) is False

    def test_repr_includes_provider_names(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should include provider names in repr."""

        manager: ProviderManager = ProviderManager.from_config(telemetry_config)

        repr_str: str = repr(manager)

        assert 'ProviderManager' in repr_str

        assert 'motive' in repr_str

        assert 'samsara' in repr_str
