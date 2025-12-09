"""
Tests for fleet_telemetry_hub.pipeline_partitioned module.

Tests PartitionedTelemetryPipeline orchestration, batch processing,
date partitioning, and error handling.
"""

# pyright: reportPrivateUsage=false
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml
from pandas import DataFrame

from fleet_telemetry_hub.config import TelemetryConfig
from fleet_telemetry_hub.pipeline_partitioned import (
    PartitionedPipelineError,
    PartitionedTelemetryPipeline,
)

config_type = dict[
    str,
    dict[str, dict[str, bool | str | list[int] | int | float]]
    | dict[str, str | int | float | bool]
    | dict[str, str],
]


class TestPartitionedTelemetryPipelineInitialization:
    """Test PartitionedTelemetryPipeline initialization."""

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
                'parquet_path': str(temp_dir / 'telemetry'),  # Directory, not file
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

        pipeline = PartitionedTelemetryPipeline(config_file)

        assert pipeline is not None

        assert pipeline.config.providers['motive'].enabled is True

    def test_initialization_raises_on_no_enabled_providers(
        self,
        temp_dir: Path,
    ) -> None:
        """
        Should raise ValueError when no providers enabled. (load_config catches
        pydantic's ValidationError and re-raises with ValueError)
        """

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
                'parquet_path': str(temp_dir / 'telemetry'),
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

        with pytest.raises(ValueError, match='Configuration validation failed'):
            PartitionedTelemetryPipeline(config_file)


class TestPartitionedTelemetryPipelineProperties:
    """Test PartitionedTelemetryPipeline properties."""

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
                'parquet_path': str(temp_dir / 'telemetry'),
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

        pipeline = PartitionedTelemetryPipeline(config_file)

        assert pipeline.config is not None

        assert isinstance(pipeline.config, TelemetryConfig)

    def test_file_handler_property_returns_handler(
        self,
        temp_dir: Path,
    ) -> None:
        """Should return file_handler via property."""

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
                'parquet_path': str(temp_dir / 'telemetry'),
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

        pipeline = PartitionedTelemetryPipeline(config_file)

        assert pipeline.file_handler is not None

        assert pipeline.file_handler.partition_count == 0


class TestPartitionedTelemetryPipelineDetermineStartDatetime:
    """Test _determine_start_datetime logic."""

    def test_uses_default_start_date_on_first_run(
        self,
        temp_dir: Path,
    ) -> None:
        """Should use default_start_date when no existing partitions."""

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
                'parquet_path': str(temp_dir / 'telemetry'),
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

        pipeline = PartitionedTelemetryPipeline(config_file)

        start_datetime: datetime = pipeline._determine_start_datetime()

        # Should be 2025-01-01 at midnight UTC

        assert start_datetime.year == 2025  # noqa: PLR2004

        assert start_datetime.month == 1

        assert start_datetime.day == 1

        assert start_datetime.hour == 0

        assert start_datetime.tzinfo == UTC


class TestPartitionedTelemetryPipelineGenerateBatches:
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
                'parquet_path': str(temp_dir / 'telemetry'),
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

        pipeline = PartitionedTelemetryPipeline(config_file)

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
                'parquet_path': str(temp_dir / 'telemetry'),
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

        pipeline = PartitionedTelemetryPipeline(config_file)

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
                'parquet_path': str(temp_dir / 'telemetry'),
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

        pipeline = PartitionedTelemetryPipeline(config_file)

        start = datetime(2025, 1, 4, tzinfo=UTC)

        end = datetime(2025, 1, 1, tzinfo=UTC)  # Before start

        batches: list[tuple[datetime, datetime]] = pipeline._generate_batches(
            start, end
        )

        assert batches == []


class TestPartitionedTelemetryPipelineRun:
    """Test PartitionedTelemetryPipeline.run() method."""

    def test_run_with_mocked_fetchers(
        self,
        temp_dir: Path,
        sample_telemetry_records: list[dict[str, Any]],
    ) -> None:
        """Should run pipeline with mocked fetchers."""

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
                'batch_increment_days': 7,  # Large batch to avoid multiple batches
                'request_delay_seconds': 0.0,
                'use_truststore': False,
            },
            'storage': {
                'parquet_path': str(temp_dir / 'telemetry'),
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

        pipeline = PartitionedTelemetryPipeline(config_file)

        # Create a mock fetcher class

        class MockFetcher:
            def __init__(self, provider: Any) -> None:
                self.provider = provider

            def fetch_data(
                self, start: datetime, end: datetime
            ) -> list[dict[str, Any]]:
                return sample_telemetry_records

        # Mock the PROVIDER_FETCHER_CLASSES

        with patch(
            'fleet_telemetry_hub.pipeline_partitioned.PROVIDER_FETCHER_CLASSES',
            {'motive': MockFetcher},
        ):
            pipeline.run()

        # Should have created partitions

        assert pipeline.file_handler.partition_count > 0

    def test_run_raises_when_all_providers_fail(
        self,
        temp_dir: Path,
    ) -> None:
        """Should raise PartitionedPipelineError when all providers fail."""

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
                'parquet_path': str(temp_dir / 'telemetry'),
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

        pipeline = PartitionedTelemetryPipeline(config_file)

        # Create mock fetcher that always fails

        class FailingFetcher:
            def __init__(self, provider: Any) -> None:
                self.provider = provider

            def fetch_data(
                self, start: datetime, end: datetime
            ) -> list[dict[str, Any]]:
                raise Exception('Fetch failed')

        with (
            patch(
                'fleet_telemetry_hub.pipeline_partitioned.PROVIDER_FETCHER_CLASSES',
                {'motive': FailingFetcher},
            ),
            pytest.raises(PartitionedPipelineError, match='All providers failed'),
        ):
            pipeline.run()

    def test_run_saves_to_date_partitions(
        self,
        temp_dir: Path,
        sample_telemetry_records: list[dict[str, Any]],
    ) -> None:
        """Should save records to date-partitioned directories."""

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
                'batch_increment_days': 7,
                'request_delay_seconds': 0.0,
                'use_truststore': False,
            },
            'storage': {
                'parquet_path': str(temp_dir / 'telemetry'),
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

        pipeline = PartitionedTelemetryPipeline(config_file)

        class MockFetcher:
            def __init__(self, provider: Any) -> None:
                self.provider = provider

            def fetch_data(
                self, start: datetime, end: datetime
            ) -> list[dict[str, Any]]:
                return sample_telemetry_records

        with patch(
            'fleet_telemetry_hub.pipeline_partitioned.PROVIDER_FETCHER_CLASSES',
            {'motive': MockFetcher},
        ):
            pipeline.run()

        # Check that partition directories were created

        telemetry_dir: Path = temp_dir / 'telemetry'

        partition_dirs: list[Path] = [
            d
            for d in telemetry_dir.iterdir()
            if d.is_dir() and d.name.startswith('date=')
        ]

        assert len(partition_dirs) > 0

        # Verify Hive-style naming

        assert all(d.name.startswith('date=') for d in partition_dirs)


class TestPartitionedTelemetryPipelineErrorHandling:
    """Test PartitionedTelemetryPipeline error handling."""

    def test_pipeline_error_includes_batch_info(self) -> None:
        """PartitionedPipelineError should include batch index and partial data info."""

        error = PartitionedPipelineError(
            message='Test error',
            batch_index=5,
            partial_data_saved=True,
            affected_partitions=[date(2024, 1, 15), date(2024, 1, 16)],
        )

        assert error.batch_index == 5  # noqa: PLR2004

        assert error.partial_data_saved is True

        assert error.affected_partitions == [date(2024, 1, 15), date(2024, 1, 16)]

        assert 'Test error' in str(error)


class TestPartitionedTelemetryPipelineDataRetention:
    """Test data retention and partition management features."""

    def test_delete_old_partitions_removes_old_data(
        self,
        temp_dir: Path,
        sample_telemetry_records: list[dict[str, Any]],
    ) -> None:
        """Should delete partitions older than retention period."""

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
                'default_start_date': '2024-01-01',
                'lookback_days': 0,
                'batch_increment_days': 7,
                'request_delay_seconds': 0.0,
                'use_truststore': False,
            },
            'storage': {
                'parquet_path': str(temp_dir / 'telemetry'),
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

        pipeline = PartitionedTelemetryPipeline(config_file)

        # Create old partitions manually

        for day in [1, 2, 3, 15, 16, 17]:
            import pandas as pd  # noqa: PLC0415

            from fleet_telemetry_hub.schema import (  # noqa: PLC0415
                enforce_telemetry_schema,
            )

            df = pd.DataFrame(sample_telemetry_records)
            df: DataFrame = enforce_telemetry_schema(df)

            partition_date = date(2024, 1, day)

            pipeline.file_handler.save_partition(df, partition_date)

        initial_count: int = pipeline.file_handler.partition_count

        assert initial_count == 6  # noqa: PLR2004

        # Delete partitions older than 10 days (should delete days 1, 2, 3)

        deleted: int = pipeline.delete_old_partitions(retention_days=10)

        assert deleted == 3  # noqa: PLR2004

        assert pipeline.file_handler.partition_count == 3  # noqa: PLR2004
