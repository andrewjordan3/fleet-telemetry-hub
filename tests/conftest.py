"""

Shared pytest fixtures for fleet_telemetry_hub tests.



This module provides reusable fixtures for common test scenarios across

all test modules. Fixtures are automatically discovered by pytest.

"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pandas as pd
import pytest
from pydantic import BaseModel

from fleet_telemetry_hub.config import (
    LoggingConfig,
    PipelineConfig,
    ProviderConfig,
    StorageConfig,
    TelemetryConfig,
)
from fleet_telemetry_hub.models import (
    HTTPMethod,
    PaginationState,
    ParsedResponse,
    ProviderCredentials,
    RequestSpec,
)
from fleet_telemetry_hub.schema import TELEMETRY_COLUMNS, enforce_telemetry_schema

# =============================================================================

# Configuration Fixtures

# =============================================================================


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """

    Provide a temporary directory for test files.



    Args:

        tmp_path: pytest built-in fixture providing unique temp directory.



    Returns:

        Path to temporary directory that is automatically cleaned up.

    """

    return tmp_path


@pytest.fixture
def temp_parquet_file(temp_dir: Path) -> Path:
    """

    Provide path to a temporary Parquet file.



    Returns:

        Path to .parquet file in temp directory (file not created).

    """

    return temp_dir / 'test_telemetry.parquet'


@pytest.fixture
def temp_log_file(temp_dir: Path) -> Path:
    """

    Provide path to a temporary log file.



    Returns:

        Path to .log file in temp directory (file not created).

    """

    return temp_dir / 'test.log'


@pytest.fixture
def storage_config(temp_parquet_file: Path) -> StorageConfig:
    """

    Provide StorageConfig for testing.



    Returns:

        StorageConfig with temp parquet path and snappy compression.

    """

    return StorageConfig(
        parquet_path=temp_parquet_file,
        parquet_compression='snappy',
    )


@pytest.fixture
def logging_config(temp_log_file: Path) -> LoggingConfig:
    """

    Provide LoggingConfig for testing.



    Returns:

        LoggingConfig with temp log file and INFO level.

    """

    return LoggingConfig(
        file_path=temp_log_file,
        console_level='INFO',
        file_level='DEBUG',
    )


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    """

    Provide PipelineConfig for testing.



    Returns:

        PipelineConfig with default test values.

    """

    return PipelineConfig(
        default_start_date='2025-01-01',
        lookback_days=7,
        batch_increment_days=1.0,
        request_delay_seconds=0.1,
        use_truststore=False,
    )


@pytest.fixture
def motive_provider_config() -> ProviderConfig:
    """

    Provide Motive ProviderConfig for testing.



    Returns:

        ProviderConfig with mock Motive settings.

    """

    return ProviderConfig(
        enabled=True,
        base_url='https://api.gomotive.com',
        api_key='test_motive_api_key',  # pyright: ignore[reportArgumentType]
        request_timeout=(10, 30),
        max_retries=3,
        retry_backoff_factor=2.0,
        verify_ssl=True,
        rate_limit_requests_per_second=10,
    )


@pytest.fixture
def samsara_provider_config() -> ProviderConfig:
    """

    Provide Samsara ProviderConfig for testing.



    Returns:

        ProviderConfig with mock Samsara settings.

    """

    return ProviderConfig(
        enabled=True,
        base_url='https://api.samsara.com',
        api_key='test_samsara_api_key',  # pyright: ignore[reportArgumentType]
        request_timeout=(10, 30),
        max_retries=3,
        retry_backoff_factor=2.0,
        verify_ssl=True,
        rate_limit_requests_per_second=10,
    )


@pytest.fixture
def telemetry_config(
    motive_provider_config: ProviderConfig,
    samsara_provider_config: ProviderConfig,
    pipeline_config: PipelineConfig,
    storage_config: StorageConfig,
    logging_config: LoggingConfig,
) -> TelemetryConfig:
    """

    Provide complete TelemetryConfig for testing.



    Returns:

        TelemetryConfig with all sub-configs populated.

    """

    return TelemetryConfig(
        providers={
            'motive': motive_provider_config,
            'samsara': samsara_provider_config,
        },
        pipeline=pipeline_config,
        storage=storage_config,
        logging=logging_config,
    )


@pytest.fixture
def provider_credentials() -> ProviderCredentials:
    """

    Provide ProviderCredentials for testing.



    Returns:

        ProviderCredentials with mock values.

    """

    return ProviderCredentials(
        base_url='https://api.example.com',
        api_key='test_api_key_12345',  # pyright: ignore[reportArgumentType]
        timeout=(10, 30),
        max_retries=3,
        retry_backoff_factor=2.0,
        verify_ssl=True,
        use_truststore=False,
    )


# =============================================================================

# Sample Data Fixtures

# =============================================================================


@pytest.fixture
def sample_timestamp() -> datetime:
    """

    Provide a fixed timestamp for deterministic testing.



    Returns:

        Timezone-aware UTC datetime: 2025-12-05 10:00:00.

    """

    return datetime(2025, 12, 5, 10, 0, 0, tzinfo=UTC)


@pytest.fixture
def sample_telemetry_record(sample_timestamp: datetime) -> dict[str, Any]:
    """

    Provide a single valid telemetry record as dict.



    Returns:

        Dictionary with all TELEMETRY_COLUMNS populated.

    """

    return {
        'provider': 'motive',
        'provider_vehicle_id': 'motive_12345',
        'vin': 'ABC123XYZ45678901',
        'fleet_number': 'TRUCK-001',
        'timestamp': sample_timestamp,
        'latitude': 37.7749,
        'longitude': -122.4194,
        'speed_mph': 55.5,
        'heading_degrees': 270.0,
        'engine_state': 'On',
        'driver_id': 'driver_001',
        'driver_name': 'John Doe',
        'location_description': 'San Francisco, CA',
        'odometer': 125000.5,
    }


@pytest.fixture
def sample_telemetry_records(
    sample_timestamp: datetime,
) -> list[dict[str, Any]]:
    """

    Provide multiple telemetry records for testing.



    Returns:

        List of 5 telemetry records with different VINs and timestamps.

    """

    records: list[dict[str, Any]] = []

    for i in range(5):
        timestamp: datetime = sample_timestamp + timedelta(minutes=i * 10)

        record: dict[str, str | datetime | float] = {
            'provider': 'motive' if i % 2 == 0 else 'samsara',
            'provider_vehicle_id': f'vehicle_{i:05d}',
            'vin': f'VIN{i:017d}',
            'fleet_number': f'TRUCK-{i:03d}',
            'timestamp': timestamp,
            'latitude': 37.7749 + (i * 0.01),
            'longitude': -122.4194 + (i * 0.01),
            'speed_mph': 50.0 + (i * 5.0),
            'heading_degrees': float(i * 45 % 360),
            'engine_state': 'On' if i % 2 == 0 else 'Off',
            'driver_id': f'driver_{i:03d}',
            'driver_name': f'Driver {i}',
            'location_description': f'Location {i}',
            'odometer': 100000.0 + (i * 1000.0),
        }

        records.append(record)

    return records


@pytest.fixture
def sample_telemetry_dataframe(
    sample_telemetry_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """

    Provide a sample DataFrame with telemetry data.



    Returns:

        DataFrame with schema-enforced telemetry records.

    """

    df = pd.DataFrame(sample_telemetry_records)

    return enforce_telemetry_schema(df)


@pytest.fixture
def duplicate_telemetry_records(
    sample_timestamp: datetime,
) -> list[dict[str, Any]]:
    """

    Provide telemetry records with intentional duplicates.



    Returns:

        List of 10 records where records are duplicated on (vin, timestamp).

    """

    records: list[dict[str, Any]] = []

    # Create 5 unique records

    for i in range(5):
        record = {
            'provider': 'motive',
            'provider_vehicle_id': f'vehicle_{i:05d}',
            'vin': f'VIN{i:017d}',
            'fleet_number': f'TRUCK-{i:03d}',
            'timestamp': sample_timestamp + timedelta(hours=i),
            'latitude': 37.7749,
            'longitude': -122.4194,
            'speed_mph': 50.0,
            'heading_degrees': 0.0,
            'engine_state': 'On',
            'driver_id': 'driver_001',
            'driver_name': 'John Doe',
            'location_description': 'Test Location',
            'odometer': 100000.0,
        }

        records.append(record)

    # Add duplicates with slightly different data

    for i in range(5):
        duplicate = records[i].copy()

        duplicate['speed_mph'] = 60.0  # Different speed

        duplicate['provider'] = 'samsara'  # Different provider

        records.append(duplicate)

    return records


# =============================================================================

# HTTP Mock Fixtures

# =============================================================================


@pytest.fixture
def mock_httpx_response() -> MagicMock:
    """

    Provide a mock httpx.Response object.



    Returns:

        MagicMock configured to act like a successful httpx.Response.

    """

    mock_response = MagicMock(spec=httpx.Response)

    mock_response.status_code = 200

    mock_response.is_success = True

    mock_response.json.return_value = {'data': []}

    mock_response.headers = {}

    mock_response.text = '{"data": []}'

    return mock_response


@pytest.fixture
def mock_paginated_response() -> dict[str, Any]:
    """

    Provide a mock paginated API response.



    Returns:

        Dictionary representing a paginated response with next page.

    """

    return {
        'data': [
            {'id': 1, 'name': 'Item 1'},
            {'id': 2, 'name': 'Item 2'},
        ],
        'pagination': {
            'hasMore': True,
            'endCursor': 'cursor_abc123',
        },
    }


# =============================================================================

# Model Fixtures

# =============================================================================


class SampleItem(BaseModel):
    """Sample Pydantic model for testing."""

    id: int

    name: str


@pytest.fixture
def sample_parsed_response() -> ParsedResponse[SampleItem]:
    """

    Provide a sample ParsedResponse for testing.



    Returns:

        ParsedResponse with sample items and pagination.

    """

    items: list[SampleItem] = [
        SampleItem(id=1, name='Item 1'),
        SampleItem(id=2, name='Item 2'),
    ]

    pagination = PaginationState(
        has_next_page=True,
        current_page=1,
        total_items=2,
        current_cursor='cursor_abc123',
    )

    return ParsedResponse[SampleItem](
        items=items,
        pagination=pagination,
    )


@pytest.fixture
def sample_request_spec(provider_credentials: ProviderCredentials) -> RequestSpec:
    """

    Provide a sample RequestSpec for testing.



    Returns:

        RequestSpec for a GET request to /vehicles.

    """

    return RequestSpec(
        url=f'{provider_credentials.base_url}/vehicles',
        method=HTTPMethod.GET,
        headers={'X-API-Key': provider_credentials.api_key.get_secret_value()},
        query_params={'page': '1', 'per_page': '10'},
        body=None,
        timeout=provider_credentials.timeout,
    )


# =============================================================================

# Utility Functions

# =============================================================================


@pytest.fixture
def assert_dataframe_valid_telemetry() -> Any:
    """

    Provide assertion function to validate telemetry DataFrame schema.



    Returns:

        Callable that asserts DataFrame has valid telemetry schema.

    """

    def _assert(df: pd.DataFrame) -> None:
        """Assert DataFrame has valid telemetry schema."""

        # Check all required columns present

        assert list(df.columns) == TELEMETRY_COLUMNS, (
            f'DataFrame columns mismatch. '
            f'Expected: {TELEMETRY_COLUMNS}, Got: {list(df.columns)}'
        )

        # Check timestamp is timezone-aware UTC

        assert df['timestamp'].dtype == 'datetime64[ns, UTC]', (
            f"Timestamp column should be 'datetime64[ns, UTC]', "
            f'got {df["timestamp"].dtype}'
        )

        # Check numeric columns

        numeric_cols: list[str] = [
            'latitude',
            'longitude',
            'speed_mph',
            'heading_degrees',
            'odometer',
        ]

        for col in numeric_cols:
            assert pd.api.types.is_float_dtype(df[col]), (
                f"Column '{col}' should be float64, got {df[col].dtype}"
            )

        # Check categorical columns

        categorical_cols: list[str] = ['provider', 'engine_state']

        for col in categorical_cols:
            assert isinstance(df[col], pd.CategoricalDtype), (
                f"Column '{col}' should be category, got {df[col].dtype}"
            )

    return _assert
