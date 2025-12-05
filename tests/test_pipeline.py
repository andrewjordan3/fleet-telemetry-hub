"""

Tests for fleet_telemetry_hub.pipeline module.



Tests TelemetryPipeline orchestration, batch processing, and error handling.

"""

# pyright: reportPrivateUsage=false
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from fleet_telemetry_hub.config import TelemetryConfig
from fleet_telemetry_hub.pipeline import PipelineError, TelemetryPipeline

config_type = dict[
    str,
    dict[str, dict[str, bool | str | list[int] | int | float]]
    | dict[str, str | int | float | bool]
    | dict[str, str],
]


class TestTelemetryPipelineInitialization:
    """Test TelemetryPipeline initialization."""

    def test_initialization_from_config_file(
        self,
        temp_dir: Path,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """Should initialize from YAML config file."""

        # Write config to file

        config_file: Path = temp_dir / 'config.yaml'

        config_data: config_type = {
            'providers': {
                'motive': {
                    'enabled': True,
                    'base_url': 'https://api.gomotive.com',
                    'api_key': 'test_key',
                    'request_timeout': [10, 30],
                    'max_retries': 3,
                    'retry_backoff_factor': 2.0,
                    'verify_ssl': True,
                    'rate_limit_requests_per_second': 10,
                },
            },
            'pipeline': {
                'default_start_date': '2025-01-01',
                'lookback_days': 7,
                'batch_increment_days': 1.0,
                'request_delay_seconds': 0.0,
                'use_truststore': False,
            },
            'storage': {
                'parquet_path': str(temp_dir / 'telemetry.parquet'),
                'parquet_compression': 'snappy',
            },
            'logging': {
                'file_path': str(temp_dir / 'telemetry.log'),
                'console_level': 'INFO',
                'file_level': 'DEBUG',
            },
        }

        with config_file.open('w') as f:
            yaml.dump(config_data, f)

        # Initialize pipeline

        pipeline = TelemetryPipeline(config_file)

        assert pipeline is not None

        assert pipeline.config.providers['motive'].enabled is True

    def test_initialization_raises_on_no_enabled_providers(
        self,
        temp_dir: Path,
    ) -> None:
        """Should raise PipelineError when no providers enabled."""

        config_file: Path = temp_dir / 'config.yaml'

        config_data: config_type = {
            'providers': {
                'motive': {
                    'enabled': False,  # Disabled
                    'base_url': 'https://api.gomotive.com',
                    'api_key': 'test_key',
                    'request_timeout': [10, 30],
                    'max_retries': 3,
                    'retry_backoff_factor': 2.0,
                    'verify_ssl': True,
                    'rate_limit_requests_per_second': 10,
                },
            },
            'pipeline': {
                'default_start_date': '2025-01-01',
                'lookback_days': 7,
                'batch_increment_days': 1.0,
                'request_delay_seconds': 0.0,
                'use_truststore': False,
            },
            'storage': {
                'parquet_path': str(temp_dir / 'telemetry.parquet'),
                'parquet_compression': 'snappy',
            },
            'logging': {
                'file_path': str(temp_dir / 'telemetry.log'),
                'console_level': 'INFO',
                'file_level': 'DEBUG',
            },
        }

        with config_file.open('w') as f:
            yaml.dump(config_data, f)

        with pytest.raises(PipelineError, match='No providers are enabled'):
            TelemetryPipeline(config_file)


class TestTelemetryPipelineProperties:
    """Test TelemetryPipeline properties."""

    def test_config_property_returns_config(
        self,
        temp_dir: Path,
    ) -> None:
        """Should return config via property."""

        config_file: Path = temp_dir / 'config.yaml'

        config_data: config_type = {
            'providers': {
                'motive': {
                    'enabled': True,
                    'base_url': 'https://api.gomotive.com',
                    'api_key': 'test_key',
                    'request_timeout': [10, 30],
                    'max_retries': 3,
                    'retry_backoff_factor': 2.0,
                    'verify_ssl': True,
                    'rate_limit_requests_per_second': 10,
                },
            },
            'pipeline': {
                'default_start_date': '2025-01-01',
                'lookback_days': 7,
                'batch_increment_days': 1.0,
                'request_delay_seconds': 0.0,
                'use_truststore': False,
            },
            'storage': {
                'parquet_path': str(temp_dir / 'telemetry.parquet'),
                'parquet_compression': 'snappy',
            },
            'logging': {
                'file_path': str(temp_dir / 'telemetry.log'),
                'console_level': 'INFO',
                'file_level': 'DEBUG',
            },
        }

        with config_file.open('w') as f:
            yaml.dump(config_data, f)

        pipeline = TelemetryPipeline(config_file)

        assert pipeline.config is not None

        assert isinstance(pipeline.config, TelemetryConfig)

    def test_dataframe_property_none_before_run(
        self,
        temp_dir: Path,
    ) -> None:
        """Should return None before run() is called."""

        config_file: Path = temp_dir / 'config.yaml'

        config_data: config_type = {
            'providers': {
                'motive': {
                    'enabled': True,
                    'base_url': 'https://api.gomotive.com',
                    'api_key': 'test_key',
                    'request_timeout': [10, 30],
                    'max_retries': 3,
                    'retry_backoff_factor': 2.0,
                    'verify_ssl': True,
                    'rate_limit_requests_per_second': 10,
                },
            },
            'pipeline': {
                'default_start_date': '2025-01-01',
                'lookback_days': 7,
                'batch_increment_days': 1.0,
                'request_delay_seconds': 0.0,
                'use_truststore': False,
            },
            'storage': {
                'parquet_path': str(temp_dir / 'telemetry.parquet'),
                'parquet_compression': 'snappy',
            },
            'logging': {
                'file_path': str(temp_dir / 'telemetry.log'),
                'console_level': 'INFO',
                'file_level': 'DEBUG',
            },
        }

        with config_file.open('w') as f:
            yaml.dump(config_data, f)

        pipeline = TelemetryPipeline(config_file)

        assert pipeline.dataframe is None


class TestTelemetryPipelineDetermineStartDatetime:
    """Test _determine_start_datetime logic."""

    def test_uses_default_start_date_on_first_run(
        self,
        temp_dir: Path,
    ) -> None:
        """Should use default_start_date when no existing data."""

        config_file: Path = temp_dir / 'config.yaml'

        config_data: config_type = {
            'providers': {
                'motive': {
                    'enabled': True,
                    'base_url': 'https://api.gomotive.com',
                    'api_key': 'test_key',
                    'request_timeout': [10, 30],
                    'max_retries': 3,
                    'retry_backoff_factor': 2.0,
                    'verify_ssl': True,
                    'rate_limit_requests_per_second': 10,
                },
            },
            'pipeline': {
                'default_start_date': '2025-01-01',
                'lookback_days': 7,
                'batch_increment_days': 1.0,
                'request_delay_seconds': 0.0,
                'use_truststore': False,
            },
            'storage': {
                'parquet_path': str(temp_dir / 'telemetry.parquet'),
                'parquet_compression': 'snappy',
            },
            'logging': {
                'file_path': str(temp_dir / 'telemetry.log'),
                'console_level': 'INFO',
                'file_level': 'DEBUG',
            },
        }

        with config_file.open('w') as f:
            yaml.dump(config_data, f)

        pipeline = TelemetryPipeline(config_file)

        start_datetime: datetime = pipeline._determine_start_datetime()

        # Should be 2025-01-01 at midnight UTC

        assert start_datetime.year == 2025  # noqa: PLR2004

        assert start_datetime.month == 1

        assert start_datetime.day == 1

        assert start_datetime.hour == 0

        assert start_datetime.tzinfo == UTC


class TestTelemetryPipelineGenerateBatches:
    """Test _generate_batches logic."""

    def test_generates_single_batch_for_short_range(
        self,
        temp_dir: Path,
    ) -> None:
        """Should generate single batch when range < increment."""

        config_file: Path = temp_dir / 'config.yaml'

        config_data: config_type = {
            'providers': {
                'motive': {
                    'enabled': True,
                    'base_url': 'https://api.gomotive.com',
                    'api_key': 'test_key',
                    'request_timeout': [10, 30],
                    'max_retries': 3,
                    'retry_backoff_factor': 2.0,
                    'verify_ssl': True,
                    'rate_limit_requests_per_second': 10,
                },
            },
            'pipeline': {
                'default_start_date': '2025-01-01',
                'lookback_days': 7,
                'batch_increment_days': 1.0,
                'request_delay_seconds': 0.0,
                'use_truststore': False,
            },
            'storage': {
                'parquet_path': str(temp_dir / 'telemetry.parquet'),
                'parquet_compression': 'snappy',
            },
            'logging': {
                'file_path': str(temp_dir / 'telemetry.log'),
                'console_level': 'INFO',
                'file_level': 'DEBUG',
            },
        }

        with config_file.open('w') as f:
            yaml.dump(config_data, f)

        pipeline = TelemetryPipeline(config_file)

        start = datetime(2025, 1, 1, tzinfo=UTC)

        end = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)  # 12 hours later

        batches: list[tuple[datetime, datetime]] = pipeline._generate_batches(
            start, end
        )

        assert len(batches) == 1

        assert batches[0][0] == start

        assert batches[0][1] == end

    def test_generates_multiple_batches_for_long_range(
        self,
        temp_dir: Path,
    ) -> None:
        """Should generate multiple batches when range > increment."""

        config_file: Path = temp_dir / 'config.yaml'

        config_data: config_type = {
            'providers': {
                'motive': {
                    'enabled': True,
                    'base_url': 'https://api.gomotive.com',
                    'api_key': 'test_key',
                    'request_timeout': [10, 30],
                    'max_retries': 3,
                    'retry_backoff_factor': 2.0,
                    'verify_ssl': True,
                    'rate_limit_requests_per_second': 10,
                },
            },
            'pipeline': {
                'default_start_date': '2025-01-01',
                'lookback_days': 7,
                'batch_increment_days': 1.0,
                'request_delay_seconds': 0.0,
                'use_truststore': False,
            },
            'storage': {
                'parquet_path': str(temp_dir / 'telemetry.parquet'),
                'parquet_compression': 'snappy',
            },
            'logging': {
                'file_path': str(temp_dir / 'telemetry.log'),
                'console_level': 'INFO',
                'file_level': 'DEBUG',
            },
        }

        with config_file.open('w') as f:
            yaml.dump(config_data, f)

        pipeline = TelemetryPipeline(config_file)

        start = datetime(2025, 1, 1, tzinfo=UTC)

        end = datetime(2025, 1, 4, tzinfo=UTC)  # 3 days later

        batches: list[tuple[datetime, datetime]] = pipeline._generate_batches(
            start, end
        )

        # Should generate 3 batches (1 day each)

        assert len(batches) == 3  # noqa: PLR2004

    def test_generates_empty_list_when_start_after_end(
        self,
        temp_dir: Path,
    ) -> None:
        """Should return empty list when start >= end."""

        config_file: Path = temp_dir / 'config.yaml'

        config_data: config_type = {
            'providers': {
                'motive': {
                    'enabled': True,
                    'base_url': 'https://api.gomotive.com',
                    'api_key': 'test_key',
                    'request_timeout': [10, 30],
                    'max_retries': 3,
                    'retry_backoff_factor': 2.0,
                    'verify_ssl': True,
                    'rate_limit_requests_per_second': 10,
                },
            },
            'pipeline': {
                'default_start_date': '2025-01-01',
                'lookback_days': 7,
                'batch_increment_days': 1.0,
                'request_delay_seconds': 0.0,
                'use_truststore': False,
            },
            'storage': {
                'parquet_path': str(temp_dir / 'telemetry.parquet'),
                'parquet_compression': 'snappy',
            },
            'logging': {
                'file_path': str(temp_dir / 'telemetry.log'),
                'console_level': 'INFO',
                'file_level': 'DEBUG',
            },
        }

        with config_file.open('w') as f:
            yaml.dump(config_data, f)

        pipeline = TelemetryPipeline(config_file)

        start = datetime(2025, 1, 4, tzinfo=UTC)

        end = datetime(2025, 1, 1, tzinfo=UTC)  # Before start

        batches: list[tuple[datetime, datetime]] = pipeline._generate_batches(
            start, end
        )

        assert batches == []


class TestTelemetryPipelineRun:
    """Test TelemetryPipeline.run() method."""

    def test_run_with_mocked_fetch_functions(
        self,
        temp_dir: Path,
        sample_telemetry_records: list[dict[str, Any]],
    ) -> None:
        """Should run pipeline with mocked fetch functions."""

        config_file: Path = temp_dir / 'config.yaml'

        config_data: config_type = {
            'providers': {
                'motive': {
                    'enabled': True,
                    'base_url': 'https://api.gomotive.com',
                    'api_key': 'test_key',
                    'request_timeout': [10, 30],
                    'max_retries': 3,
                    'retry_backoff_factor': 2.0,
                    'verify_ssl': True,
                    'rate_limit_requests_per_second': 10,
                },
            },
            'pipeline': {
                'default_start_date': '2025-01-01',
                'lookback_days': 0,  # No lookback for test
                'batch_increment_days': 30.0,  # Large batch to avoid multiple batches
                'request_delay_seconds': 0.0,
                'use_truststore': False,
            },
            'storage': {
                'parquet_path': str(temp_dir / 'telemetry.parquet'),
                'parquet_compression': 'snappy',
            },
            'logging': {
                'file_path': str(temp_dir / 'telemetry.log'),
                'console_level': 'WARNING',  # Reduce logging noise
                'file_level': 'WARNING',
            },
        }

        with config_file.open('w') as f:
            yaml.dump(config_data, f)

        pipeline = TelemetryPipeline(config_file)

        # Mock the fetch function to return sample data

        def mock_fetch_motive(
            provider: Any, start: datetime, end: datetime
        ) -> list[dict[str, Any]]:
            return sample_telemetry_records

        with patch(
            'fleet_telemetry_hub.pipeline.PROVIDER_FETCH_FUNCTIONS',
            {'motive': mock_fetch_motive},
        ):
            pipeline.run()

        # Should have data now

        assert pipeline.dataframe is not None

        assert len(pipeline.dataframe) > 0

    def test_run_raises_when_all_providers_fail(
        self,
        temp_dir: Path,
    ) -> None:
        """Should raise PipelineError when all providers fail."""

        config_file: Path = temp_dir / 'config.yaml'

        config_data: config_type = {
            'providers': {
                'motive': {
                    'enabled': True,
                    'base_url': 'https://api.gomotive.com',
                    'api_key': 'test_key',
                    'request_timeout': [10, 30],
                    'max_retries': 3,
                    'retry_backoff_factor': 2.0,
                    'verify_ssl': True,
                    'rate_limit_requests_per_second': 10,
                },
            },
            'pipeline': {
                'default_start_date': '2025-01-01',
                'lookback_days': 0,
                'batch_increment_days': 1.0,
                'request_delay_seconds': 0.0,
                'use_truststore': False,
            },
            'storage': {
                'parquet_path': str(temp_dir / 'telemetry.parquet'),
                'parquet_compression': 'snappy',
            },
            'logging': {
                'file_path': str(temp_dir / 'telemetry.log'),
                'console_level': 'WARNING',
                'file_level': 'WARNING',
            },
        }

        with config_file.open('w') as f:
            yaml.dump(config_data, f)

        pipeline = TelemetryPipeline(config_file)

        # Mock fetch function to always raise

        def mock_fetch_fails(
            provider: Any, start: datetime, end: datetime
        ) -> list[dict[str, Any]]:
            raise Exception('Fetch failed')

        with (
            patch(
                'fleet_telemetry_hub.pipeline.PROVIDER_FETCH_FUNCTIONS',
                {'motive': mock_fetch_fails},
            ),
            pytest.raises(PipelineError, match='All providers failed'),
        ):
            pipeline.run()


class TestTelemetryPipelineErrorHandling:
    """Test TelemetryPipeline error handling."""

    def test_pipeline_error_includes_batch_info(self) -> None:
        """PipelineError should include batch index and partial data info."""

        error = PipelineError(
            message='Test error',
            batch_index=5,
            partial_data_saved=True,
        )

        assert error.batch_index == 5  # noqa: PLR2004

        assert error.partial_data_saved is True

        assert 'Test error' in str(error)
