#!/usr/bin/env python3
"""
Usage examples for fleet_telemetry_hub abstraction layers.

This file demonstrates the different ways to interact with the API,
from low-level to high-level abstractions.
"""

from datetime import date, datetime

# =============================================================================
# Level 1: Direct Endpoint Usage (Most Control)
# =============================================================================


def example_1_direct_endpoint_usage() -> None:
    """
    Lowest level: Direct endpoint and client usage.

    Use this when you need maximum control or are working with
    a single endpoint repeatedly.
    """
    from fleet_telemetry_hub.client import TelemetryClient
    from fleet_telemetry_hub.models.motive_requests import MotiveEndpoints
    from fleet_telemetry_hub.models.shared_response_models import ProviderCredentials
    from pydantic import SecretStr

    # Create credentials
    credentials = ProviderCredentials(
        base_url='https://api.gomotive.com',
        api_key=SecretStr('your-api-key-here'),
        timeout=(10, 30),
        max_retries=5,
        retry_backoff_factor=2.0,
        verify_ssl=True,
    )

    # Get endpoint definition
    vehicles_endpoint = MotiveEndpoints.VEHICLES

    # Create client and fetch
    with TelemetryClient(credentials) as client:
        # Fetch single page
        response = client.fetch(vehicles_endpoint)
        print(f'Found {response.item_count} vehicles on page 1')

        # Or fetch all items across all pages
        for vehicle in client.fetch_all(vehicles_endpoint, request_delay_seconds=0.5):
            print(f'{vehicle.number}: {vehicle.make} {vehicle.model}')


# =============================================================================
# Level 2: Registry-Based Access (String-Based Lookup)
# =============================================================================


def example_2_registry_access() -> None:
    """
    Mid-level: Use registry for dynamic endpoint lookup.

    Use this when you need to select endpoints at runtime
    based on string names (e.g., from config or user input).
    """
    from fleet_telemetry_hub.client import TelemetryClient
    from fleet_telemetry_hub.models.shared_response_models import ProviderCredentials
    from fleet_telemetry_hub import EndpointRegistry
    from pydantic import SecretStr

    # Get shared registry instance
    registry = EndpointRegistry.instance()

    # Discover available endpoints
    print('Available providers:', registry.list_providers())
    print('Motive endpoints:', registry.list_endpoints('motive'))

    # Get endpoint by string name
    vehicles_endpoint = registry.get('motive', 'vehicles')
    groups_endpoint = registry.get('motive', 'groups')

    # Create credentials
    credentials = ProviderCredentials(
        base_url='https://api.gomotive.com',
        api_key=SecretStr('your-api-key-here'),
    )

    # Use with client
    with TelemetryClient(credentials) as client:
        # Fetch vehicles
        for vehicle in client.fetch_all(vehicles_endpoint):
            print(vehicle.number)

        # Fetch groups
        for group in client.fetch_all(groups_endpoint):
            print(group.name)


def example_2b_registry_introspection() -> None:
    """Discover and explore endpoints programmatically."""
    from fleet_telemetry_hub import EndpointRegistry

    registry = EndpointRegistry.instance()

    # Generate full documentation
    print(registry.describe())

    # Find endpoints by path
    matches = registry.find_by_path('/vehicles')
    print(f'Endpoints with "/vehicles" in path: {matches}')

    # Check if endpoint exists before using
    if registry.has('samsara', 'drivers'):
        endpoint = registry.get('samsara', 'drivers')
        print(f'Found endpoint: {endpoint.description}')


# =============================================================================
# Level 3: Provider Facade (Recommended for Most Use Cases)
# =============================================================================


def example_3_provider_facade() -> None:
    """
    High-level: Use Provider facade for cleaner API.

    This is the recommended approach for most applications.
    Combines configuration, endpoint access, and client management.
    """
    from fleet_telemetry_hub.config.loader import load_config
    from fleet_telemetry_hub.provider import Provider

    # Load configuration
    config = load_config('config/telemetry_config.yaml')

    # Create provider from config
    motive = Provider.from_config('motive', config)

    # Simple fetch - client managed automatically
    for vehicle in motive.fetch_all('vehicles'):
        print(f'{vehicle.number}: {vehicle.make} {vehicle.model}')

    # Fetch with parameters
    for location in motive.fetch_all(
        'vehicle_locations',
        vehicle_id=12345,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
    ):
        print(f'Location at {location.recorded_at}: {location.latitude}, {location.longitude}')

    # Or use context manager for multiple operations (better performance)
    with motive.client() as client:
        # Fetch multiple endpoints with same client
        vehicles = list(client.fetch_all(motive.endpoint('vehicles')))
        groups = list(client.fetch_all(motive.endpoint('groups')))
        users = list(client.fetch_all(motive.endpoint('users')))

    print(f'Fetched {len(vehicles)} vehicles, {len(groups)} groups, {len(users)} users')


# =============================================================================
# Level 4: Multi-Provider Manager (For Multi-Provider Apps)
# =============================================================================


def example_4_provider_manager() -> None:
    """
    Highest level: Manage multiple providers from config.

    Use this when working with multiple providers (Motive + Samsara).
    """
    from fleet_telemetry_hub.config.loader import load_config
    from fleet_telemetry_hub.provider import ProviderManager

    # Load configuration
    config = load_config('config/telemetry_config.yaml')

    # Create manager for all enabled providers
    manager = ProviderManager.from_config(config)

    # Access specific provider
    motive = manager.get('motive')
    for vehicle in motive.fetch_all('vehicles'):
        print(f'Motive: {vehicle.number}')

    # Or iterate all enabled providers
    for provider_name, provider in manager.enabled_providers():
        print(f'\n=== Fetching from {provider_name} ===')

        # Fetch vehicles from each provider
        for vehicle in provider.fetch_all('vehicles'):
            print(f'{vehicle.number if hasattr(vehicle, "number") else vehicle.name}')


# =============================================================================
# Advanced Examples
# =============================================================================


def example_5_pagination_control() -> None:
    """Work with pagination explicitly for progress tracking."""
    from fleet_telemetry_hub.config.loader import load_config
    from fleet_telemetry_hub.provider import Provider

    config = load_config('config/telemetry_config.yaml')
    motive = Provider.from_config('motive', config)

    # Fetch page by page with progress tracking
    total_items = 0
    for page_num, page in enumerate(motive.fetch_all_pages('vehicles'), start=1):
        total_items += page.item_count
        print(f'Page {page_num}: {page.item_count} items (total so far: {total_items})')

        if page.pagination.total_items:
            progress = (total_items / page.pagination.total_items) * 100
            print(f'Progress: {progress:.1f}%')

        # Process batch
        for vehicle in page.items:
            process_vehicle(vehicle)

        if not page.has_more:
            print(f'Completed! Total items: {total_items}')
            break


def example_6_error_handling() -> None:
    """Proper error handling for API operations."""
    from fleet_telemetry_hub.client import APIError, RateLimitError, TelemetryClient
    from fleet_telemetry_hub.models.shared_response_models import ProviderCredentials
    from fleet_telemetry_hub import EndpointRegistry
    from fleet_telemetry_hub.registry import EndpointNotFoundError

    registry = EndpointRegistry.instance()
    credentials = ProviderCredentials(
        base_url='https://api.gomotive.com',
        api_key=SecretStr('your-api-key-here'),
    )

    try:
        # Check if endpoint exists
        if not registry.has('motive', 'vehicles'):
            raise EndpointNotFoundError('motive', 'vehicles')

        endpoint = registry.get('motive', 'vehicles')

        with TelemetryClient(credentials) as client:
            for vehicle in client.fetch_all(endpoint):
                print(vehicle.number)

    except EndpointNotFoundError as e:
        print(f'Endpoint not found: {e.provider}/{e.endpoint_name}')

    except RateLimitError as e:
        print(f'Rate limited! Retry after {e.rate_limit_info.retry_after_seconds}s')
        print(f'Remaining: {e.rate_limit_info.remaining}')

    except APIError as e:
        print(f'API error {e.status_code}: {e}')
        if e.response_body:
            print(f'Response: {e.response_body[:500]}')


def example_7_samsara_specific() -> None:
    """Samsara-specific examples (cursor pagination, time ranges)."""
    from fleet_telemetry_hub.config.loader import load_config
    from fleet_telemetry_hub.provider import Provider

    config = load_config('config/telemetry_config.yaml')
    samsara = Provider.from_config('samsara', config.providers['samsara'])

    # Fetch vehicles (cursor-based pagination handled automatically)
    for vehicle in samsara.fetch_all('vehicles'):
        print(f'{vehicle.name}: {vehicle.id}')

    # Fetch drivers with filter
    for driver in samsara.fetch_all(
        'drivers',
        driver_activation_status='active',
    ):
        print(f'{driver.name}: {driver.id}')

    # Fetch historical vehicle stats with time range
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    end_time = datetime(2025, 1, 31, 23, 59, 59)

    for stat in samsara.fetch_all(
        'vehicle_stats_history',
        start_time=start_time,
        end_time=end_time,
        types=['engineStates', 'gps'],
        request_delay_seconds=1.0,  # Be nice to API
    ):
        print(f'Vehicle {stat.id}: {stat.engine_states}')


def example_8_custom_registry() -> None:
    """Create custom endpoint registry (for testing or extensions)."""
    from fleet_telemetry_hub.provider import Provider, ProviderManager
    from fleet_telemetry_hub.registry import EndpointRegistry

    # Create custom registry (useful for testing)
    custom_registry = EndpointRegistry()

    # Use with providers
    from fleet_telemetry_hub.config.loader import load_config

    config = load_config('config/telemetry_config.yaml')

    # Single provider with custom registry
    motive = Provider.from_config(
        'motive',
        config.providers['motive'],
        registry=custom_registry,
    )

    # Or manager with custom registry
    manager = ProviderManager.from_config(config, registry=custom_registry)


def example_9_to_dataframe() -> None:
    """Convert API data to pandas DataFrame for analysis."""
    from fleet_telemetry_hub.config.loader import load_config
    from fleet_telemetry_hub.provider import Provider

    config = load_config('config/telemetry_config.yaml')
    motive = Provider.from_config('motive', config)

    # Get all vehicles as DataFrame
    vehicles_df = motive.to_dataframe('vehicles')
    print(f'Loaded {len(vehicles_df)} vehicles')
    print(vehicles_df.head())

    # Get vehicle locations with parameters
    from datetime import date

    locations_df = motive.to_dataframe(
        'vehicle_locations',
        vehicle_id=12345,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
    )

    # Analyze data
    print(f'\nLocation statistics:')
    print(locations_df.describe())

    # Save to various formats
    vehicles_df.to_parquet('data/vehicles.parquet', compression='snappy')
    vehicles_df.to_csv('data/vehicles.csv', index=False)
    vehicles_df.to_excel('data/vehicles.xlsx', index=False)

    # Or work with multiple endpoints
    import pandas as pd

    vehicles = motive.to_dataframe('vehicles')
    groups = motive.to_dataframe('groups')
    users = motive.to_dataframe('users')

    # Combine or analyze together
    print(f'\nDataset summary:')
    print(f'  Vehicles: {len(vehicles)} rows, {len(vehicles.columns)} columns')
    print(f'  Groups: {len(groups)} rows, {len(groups.columns)} columns')
    print(f'  Users: {len(users)} rows, {len(users.columns)} columns')


def example_10_dataframe_with_client() -> None:
    """Use to_dataframe with TelemetryClient for lower-level control."""
    from fleet_telemetry_hub import TelemetryClient
    from fleet_telemetry_hub.models.motive_requests import MotiveEndpoints
    from fleet_telemetry_hub.models.shared_response_models import ProviderCredentials

    credentials = ProviderCredentials(
        base_url='https://api.gomotive.com',
        api_key=SecretStr('your-api-key-here'),
    )

    with TelemetryClient(credentials) as client:
        # Get DataFrame using client
        df = client.to_dataframe(MotiveEndpoints.VEHICLES)
        print(f'Fetched {len(df)} vehicles')

        # Process multiple endpoints with same client
        vehicles_df = client.to_dataframe(MotiveEndpoints.VEHICLES)
        groups_df = client.to_dataframe(MotiveEndpoints.GROUPS)
        users_df = client.to_dataframe(MotiveEndpoints.USERS)

        # Save combined data
        vehicles_df.to_parquet('data/vehicles.parquet')
        groups_df.to_parquet('data/groups.parquet')
        users_df.to_parquet('data/users.parquet')


# =============================================================================
# Helper Functions
# =============================================================================


def process_vehicle(vehicle: object) -> None:
    """Placeholder for vehicle processing logic."""
    pass


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run example demonstrating recommended usage."""
    print('=' * 80)
    print('Fleet Telemetry Hub - Usage Examples')
    print('=' * 80)

    # Show recommended approach (Level 3)
    print('\n--- Recommended Approach: Provider Facade ---\n')

    from fleet_telemetry_hub.config.loader import load_config
    from fleet_telemetry_hub.provider import Provider

    try:
        config = load_config('config/telemetry_config.yaml')
        motive = Provider.from_config('motive', config)

        # Show available endpoints
        print(f'Provider: {motive.name}')
        print(f'Base URL: {motive.credentials.base_url}')
        print(f'Available endpoints: {", ".join(motive.list_endpoints())}')

        print('\nEndpoint Details:')
        print(motive.describe())

        # Example fetch (commented out to avoid actual API calls)
        # for vehicle in motive.fetch_all('vehicles'):
        #     print(f'{vehicle.number}: {vehicle.make} {vehicle.model}')

    except FileNotFoundError:
        print('Config file not found. Please create config/telemetry_config.yaml')
    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
