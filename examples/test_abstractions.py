#!/usr/bin/env python3
"""
Test script to verify abstraction layers work correctly.

This script doesn't make actual API calls but verifies that all the
abstraction layers can be instantiated and used correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_layer_1_endpoints() -> None:
    """Test Layer 1: Direct endpoint definitions."""
    print('\n=== Testing Layer 1: Endpoint Definitions ===')

    from fleet_telemetry_hub.models.motive_requests import MotiveEndpoints
    from fleet_telemetry_hub.models.samsara_requests import SamsaraEndpoints

    # Test Motive endpoint
    motive_vehicles = MotiveEndpoints.VEHICLES
    print(f'✓ Motive VEHICLES endpoint: {motive_vehicles.endpoint_path}')
    print(f'  Description: {motive_vehicles.description}')
    print(f'  Paginated: {motive_vehicles.is_paginated}')
    print(f'  HTTP Method: {motive_vehicles.http_method.value}')

    # Test Samsara endpoint
    samsara_vehicles = SamsaraEndpoints.VEHICLES
    print(f'✓ Samsara VEHICLES endpoint: {samsara_vehicles.endpoint_path}')
    print(f'  Description: {samsara_vehicles.description}')

    # Test endpoint with parameters
    vehicle_locations = MotiveEndpoints.VEHICLE_LOCATIONS
    print(f'✓ VEHICLE_LOCATIONS has {len(vehicle_locations.path_parameters)} path params')
    print(f'  and {len(vehicle_locations.query_parameters)} query params')

    print('✓ Layer 1 tests passed!')


def test_layer_2_client() -> None:
    """Test Layer 2: TelemetryClient."""
    print('\n=== Testing Layer 2: Telemetry Client ===')

    from fleet_telemetry_hub import TelemetryClient
    from fleet_telemetry_hub.models.shared_response_models import ProviderCredentials

    # Create credentials
    credentials = ProviderCredentials(
        base_url='https://api.example.com',
        api_key='test-key-12345',
        timeout=(10, 30),
        max_retries=3,
        verify_ssl=True,
    )

    print(f'✓ Created credentials for: {credentials.base_url}')

    # Create client
    client = TelemetryClient(credentials)
    print('✓ Created TelemetryClient')

    # Test context manager
    with TelemetryClient(credentials) as client:
        print('✓ Context manager works')

    client.close()
    print('✓ Layer 2 tests passed!')


def test_layer_3_registry() -> None:
    """Test Layer 3: Endpoint Registry."""
    print('\n=== Testing Layer 3: Endpoint Registry ===')

    from fleet_telemetry_hub import EndpointRegistry, get_registry

    # Test global registry
    registry = get_registry()
    print(f'✓ Got global registry')

    # List providers
    providers = registry.list_providers()
    print(f'✓ Found {len(providers)} providers: {", ".join(providers)}')

    # List endpoints for each provider
    for provider in providers:
        endpoints = registry.list_endpoints(provider)
        print(f'  {provider}: {len(endpoints)} endpoints')

    # Test endpoint lookup
    motive_vehicles = registry.get('motive', 'vehicles')
    print(f'✓ Retrieved motive vehicles endpoint: {motive_vehicles.endpoint_path}')

    samsara_vehicles = registry.get('samsara', 'vehicles')
    print(f'✓ Retrieved samsara vehicles endpoint: {samsara_vehicles.endpoint_path}')

    # Test has() method
    assert registry.has('motive', 'vehicles')
    assert registry.has('samsara', 'drivers')
    assert not registry.has('motive', 'nonexistent')
    print('✓ has() method works correctly')

    # Test find_by_path
    matches = registry.find_by_path('/vehicles')
    print(f'✓ Found {len(matches)} endpoints with "/vehicles" in path')

    # Test describe
    description = registry.describe('motive')
    assert 'motive' in description.lower()
    print('✓ describe() generates documentation')

    # Test custom registry
    custom_registry = EndpointRegistry()
    print('✓ Can create custom registry instances')

    print('✓ Layer 3 tests passed!')


def test_layer_4_provider() -> None:
    """Test Layer 4: Provider facade."""
    print('\n=== Testing Layer 4: Provider Facade ===')

    from fleet_telemetry_hub import Provider
    from fleet_telemetry_hub.config.config_models import ProviderConfig
    from fleet_telemetry_hub.models.shared_response_models import ProviderCredentials
    from pydantic import SecretStr

    # Create config
    config = ProviderConfig(
        enabled=True,
        base_url='https://api.example.com',
        api_key=SecretStr('test-key-12345'),
        request_timeout=[10, 30],
        max_retries=5,
        retry_backoff_factor=2.0,
        verify_ssl=True,
        rate_limit_requests_per_second=10,
    )

    print('✓ Created ProviderConfig')

    # Create provider from config
    provider = Provider.from_config('motive', config)
    print(f'✓ Created Provider: {provider}')

    # Test endpoint access
    endpoint = provider.endpoint('vehicles')
    print(f'✓ Got endpoint via provider: {endpoint.endpoint_path}')

    # Test list endpoints
    endpoints = provider.list_endpoints()
    print(f'✓ Provider has {len(endpoints)} endpoints: {", ".join(endpoints[:3])}...')

    # Test has_endpoint
    assert provider.has_endpoint('vehicles')
    assert provider.has_endpoint('groups')
    assert not provider.has_endpoint('nonexistent')
    print('✓ has_endpoint() works correctly')

    # Test describe
    description = provider.describe()
    assert 'motive' in description.lower()
    print('✓ Provider describe() works')

    # Test client creation
    with provider.client() as client:
        print('✓ Provider can create client')

    print('✓ Layer 4 tests passed!')


def test_provider_manager() -> None:
    """Test ProviderManager."""
    print('\n=== Testing ProviderManager ===')

    from fleet_telemetry_hub import ProviderManager
    from fleet_telemetry_hub.config.config_models import ProviderConfig, TelemetryConfig, PipelineConfig, StorageConfig, LoggingConfig
    from pydantic import SecretStr

    # Create test config
    motive_config = ProviderConfig(
        enabled=True,
        base_url='https://api.gomotive.com',
        api_key=SecretStr('test-key-1'),
        request_timeout=[10, 30],
        max_retries=5,
        retry_backoff_factor=2.0,
        verify_ssl=True,
        rate_limit_requests_per_second=10,
    )

    samsara_config = ProviderConfig(
        enabled=True,
        base_url='https://api.samsara.com',
        api_key=SecretStr('test-key-2'),
        request_timeout=[10, 30],
        max_retries=5,
        retry_backoff_factor=2.0,
        verify_ssl=True,
        rate_limit_requests_per_second=5,
    )

    config = TelemetryConfig(
        providers={'motive': motive_config, 'samsara': samsara_config},
        pipeline=PipelineConfig(
            default_start_date='2024-01-01',
            lookback_days=7,
            request_delay_seconds=0.5,
        ),
        storage=StorageConfig(
            parquet_path='data/test.parquet',
            parquet_compression='snappy',
        ),
        logging=LoggingConfig(
            file_path='logs/test.log',
            console_level='INFO',
            file_level='DEBUG',
        ),
    )

    print('✓ Created test TelemetryConfig')

    # Create manager
    manager = ProviderManager.from_config(config)
    print(f'✓ Created ProviderManager: {manager}')

    # Test provider access
    motive = manager.get('motive')
    print(f'✓ Got motive provider: {motive}')

    samsara = manager.get('samsara')
    print(f'✓ Got samsara provider: {samsara}')

    # Test list providers
    providers = manager.list_providers()
    print(f'✓ Manager has {len(providers)} providers: {", ".join(providers)}')

    # Test has()
    assert manager.has('motive')
    assert manager.has('samsara')
    assert not manager.has('nonexistent')
    print('✓ has() works correctly')

    # Test enabled_providers()
    enabled = list(manager.enabled_providers())
    print(f'✓ Can iterate {len(enabled)} enabled providers')

    print('✓ ProviderManager tests passed!')


def test_request_spec_building() -> None:
    """Test that endpoints can build request specs correctly."""
    print('\n=== Testing Request Spec Building ===')

    from fleet_telemetry_hub.models.motive_requests import MotiveEndpoints
    from fleet_telemetry_hub.models.samsara_requests import SamsaraEndpoints
    from fleet_telemetry_hub.models.shared_response_models import ProviderCredentials
    from pydantic import SecretStr

    # Create credentials
    motive_creds = ProviderCredentials(
        base_url='https://api.gomotive.com',
        api_key=SecretStr('test-motive-key'),
    )

    samsara_creds = ProviderCredentials(
        base_url='https://api.samsara.com',
        api_key=SecretStr('test-samsara-key'),
    )

    # Test Motive request spec building
    motive_vehicles = MotiveEndpoints.VEHICLES
    motive_spec = motive_vehicles.build_request_spec(motive_creds)

    print(f'✓ Built Motive request spec:')
    print(f'  URL: {motive_spec.url}')
    print(f'  Method: {motive_spec.method.value}')
    print(f'  Headers: {list(motive_spec.headers.keys())}')
    print(f'  Query params: {motive_spec.query_params}')

    assert 'X-API-Key' in motive_spec.headers
    assert 'page_no' in motive_spec.query_params
    assert 'per_page' in motive_spec.query_params

    # Test Samsara request spec building
    samsara_vehicles = SamsaraEndpoints.VEHICLES
    samsara_spec = samsara_vehicles.build_request_spec(samsara_creds)

    print(f'✓ Built Samsara request spec:')
    print(f'  URL: {samsara_spec.url}')
    print(f'  Method: {samsara_spec.method.value}')
    print(f'  Headers: {list(samsara_spec.headers.keys())}')
    print(f'  Query params: {samsara_spec.query_params}')

    assert 'Authorization' in samsara_spec.headers
    assert samsara_spec.headers['Authorization'].startswith('Bearer ')

    print('✓ Request spec building tests passed!')


def main() -> None:
    """Run all tests."""
    print('=' * 80)
    print('Fleet Telemetry Hub - Abstraction Layer Tests')
    print('=' * 80)

    try:
        test_layer_1_endpoints()
        test_layer_2_client()
        test_layer_3_registry()
        test_layer_4_provider()
        test_provider_manager()
        test_request_spec_building()

        print('\n' + '=' * 80)
        print('✅ ALL TESTS PASSED!')
        print('=' * 80)
        print('\nThe abstraction layers are working correctly.')
        print('You can now use them in your application.')
        print('\nRecommended starting point: See examples/usage_examples.py')

    except Exception as e:
        print(f'\n❌ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
