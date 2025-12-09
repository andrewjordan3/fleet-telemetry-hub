"""
Tests for fleet_telemetry_hub.common.partitioned_file_io module.

Tests PartitionedParquetHandler operations including partitioned storage,
date range loading, atomic writes, and partition management.
"""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fleet_telemetry_hub.common import PartitionedParquetHandler
from fleet_telemetry_hub.config import StorageConfig
from fleet_telemetry_hub.schema import enforce_telemetry_schema


class TestPartitionedParquetHandlerInitialization:
    """Test PartitionedParquetHandler initialization."""

    def test_initialization_creates_base_directory(
        self,
        temp_dir: Path,
    ) -> None:
        """Should create base directory on initialization."""

        # Use nested path to test directory creation

        nested_path: Path = temp_dir / 'data' / 'telemetry'

        storage_config = StorageConfig(
            parquet_path=nested_path,
            parquet_compression='snappy',
        )

        PartitionedParquetHandler(storage_config)

        # Base directory should be created

        assert nested_path.exists()

        assert nested_path.is_dir()

    def test_initialization_properties(
        self,
        storage_config: StorageConfig,
    ) -> None:
        """Should set properties correctly."""

        handler = PartitionedParquetHandler(storage_config)

        assert handler.base_path == storage_config.parquet_path

        assert handler.partition_count == 0  # No partitions yet


class TestPartitionedParquetHandlerPartitionOperations:
    """Test partition save and load operations."""

    def test_save_partition_creates_partition_directory(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should create partition directory when saving."""

        handler = PartitionedParquetHandler(storage_config)

        partition_date = date(2024, 1, 15)

        # Remove partition_date column if it exists
        df = sample_telemetry_dataframe.copy()
        if 'partition_date' in df.columns:
            df = df.drop(columns=['partition_date'])

        handler.save_partition(df, partition_date)

        # Should create date=2024-01-15 directory

        partition_dir = storage_config.parquet_path / 'date=2024-01-15'

        assert partition_dir.exists()

        assert partition_dir.is_dir()

        # Should create data.parquet file

        parquet_file = partition_dir / 'data.parquet'

        assert parquet_file.exists()

    def test_load_partition_returns_none_when_missing(
        self,
        storage_config: StorageConfig,
    ) -> None:
        """Should return None when partition doesn't exist."""

        handler = PartitionedParquetHandler(storage_config)

        result: pd.DataFrame | None = handler.load_partition(date(2024, 1, 15))

        assert result is None

    def test_load_partition_returns_dataframe_when_exists(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should load DataFrame from existing partition."""

        handler = PartitionedParquetHandler(storage_config)

        partition_date = date(2024, 1, 15)

        # Remove partition_date column if it exists
        df = sample_telemetry_dataframe.copy()
        if 'partition_date' in df.columns:
            df = df.drop(columns=['partition_date'])

        # Save

        handler.save_partition(df, partition_date)

        # Load

        result: pd.DataFrame | None = handler.load_partition(partition_date)

        assert result is not None

        assert isinstance(result, pd.DataFrame)

        assert len(result) == len(sample_telemetry_dataframe)

    def test_partition_exists_returns_correct_value(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should correctly report partition existence."""

        handler = PartitionedParquetHandler(storage_config)

        partition_date = date(2024, 1, 15)

        assert handler.partition_exists(partition_date) is False

        # Remove partition_date column if it exists
        df = sample_telemetry_dataframe.copy()
        if 'partition_date' in df.columns:
            df = df.drop(columns=['partition_date'])

        handler.save_partition(df, partition_date)

        assert handler.partition_exists(partition_date) is True


class TestPartitionedParquetHandlerDateRangeOperations:
    """Test date range loading operations."""

    def test_load_date_range_returns_none_when_no_partitions(
        self,
        storage_config: StorageConfig,
    ) -> None:
        """Should return None when no partitions exist in range."""

        handler = PartitionedParquetHandler(storage_config)

        result: pd.DataFrame | None = handler.load_date_range(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert result is None

    def test_load_date_range_loads_multiple_partitions(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should load and concatenate multiple partitions."""

        handler = PartitionedParquetHandler(storage_config)

        # Save 3 partitions

        for day in [15, 16, 17]:
            partition_date = date(2024, 1, day)

            df = sample_telemetry_dataframe.copy()
            if 'partition_date' in df.columns:
                df = df.drop(columns=['partition_date'])

            handler.save_partition(df, partition_date)

        # Load date range

        result: pd.DataFrame | None = handler.load_date_range(
            start_date=date(2024, 1, 15),
            end_date=date(2024, 1, 17),
        )

        assert result is not None

        # Should have 3x the records (3 partitions)

        assert len(result) == len(sample_telemetry_dataframe) * 3

    def test_load_date_range_filters_by_date(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should only load partitions within the specified range."""

        handler = PartitionedParquetHandler(storage_config)

        # Save 5 partitions

        for day in [13, 14, 15, 16, 17]:
            partition_date = date(2024, 1, day)

            df = sample_telemetry_dataframe.copy()
            if 'partition_date' in df.columns:
                df = df.drop(columns=['partition_date'])

            handler.save_partition(df, partition_date)

        # Load only middle 3 partitions

        result: pd.DataFrame | None = handler.load_date_range(
            start_date=date(2024, 1, 14),
            end_date=date(2024, 1, 16),
        )

        assert result is not None

        # Should have 3x the records (3 partitions: 14, 15, 16)

        assert len(result) == len(sample_telemetry_dataframe) * 3


class TestPartitionedParquetHandlerSavePartitioned:
    """Test save_partitioned method with auto-partitioning."""

    def test_save_partitioned_groups_by_date(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should automatically partition records by date column."""

        handler = PartitionedParquetHandler(storage_config)

        # Add partition dates to sample data

        df = sample_telemetry_dataframe.copy()

        # Assign records to different dates

        df['partition_date'] = [
            date(2024, 1, 15),
            date(2024, 1, 15),
            date(2024, 1, 16),
            date(2024, 1, 16),
            date(2024, 1, 17),
        ]

        # Save with auto-partitioning

        records_saved: dict[date, int] = handler.save_partitioned(
            df,
            date_column='partition_date',
            deduplicate=False,
        )

        # Should create 3 partitions

        assert len(records_saved) == 3

        assert records_saved[date(2024, 1, 15)] == 2

        assert records_saved[date(2024, 1, 16)] == 2

        assert records_saved[date(2024, 1, 17)] == 1

    def test_save_partitioned_deduplicates_within_partition(
        self,
        storage_config: StorageConfig,
    ) -> None:
        """Should deduplicate records within each partition."""

        handler = PartitionedParquetHandler(storage_config)

        # Create dataframe with duplicate records

        records: list[dict[str, Any]] = [
            {
                'provider': 'motive',
                'provider_vehicle_id': 'vehicle_1',
                'vin': 'ABC123',
                'fleet_number': '001',
                'timestamp': datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),
                'latitude': 37.7749,
                'longitude': -122.4194,
                'speed_mph': 45.0,
                'heading_degrees': 90.0,
                'engine_state': 'running',
                'driver_id': 'driver_1',
                'driver_name': 'John Doe',
                'location_description': 'San Francisco',
                'odometer': 10000.0,
                'partition_date': date(2024, 1, 15),
            },
            {
                'provider': 'motive',
                'provider_vehicle_id': 'vehicle_1',
                'vin': 'ABC123',
                'fleet_number': '001',
                'timestamp': datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC),  # Duplicate
                'latitude': 37.7749,
                'longitude': -122.4194,
                'speed_mph': 50.0,  # Different speed (updated)
                'heading_degrees': 90.0,
                'engine_state': 'running',
                'driver_id': 'driver_1',
                'driver_name': 'John Doe',
                'location_description': 'San Francisco',
                'odometer': 10000.0,
                'partition_date': date(2024, 1, 15),
            },
        ]

        df = pd.DataFrame(records)

        df = enforce_telemetry_schema(df)

        # Save with deduplication

        records_saved: dict[date, int] = handler.save_partitioned(
            df,
            date_column='partition_date',
            deduplicate=True,
            dedup_columns=['provider', 'provider_vehicle_id', 'timestamp'],
        )

        # Should save only 1 record (deduplicated)

        assert records_saved[date(2024, 1, 15)] == 1


class TestPartitionedParquetHandlerPartitionManagement:
    """Test partition deletion and management operations."""

    def test_delete_partition_removes_partition(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should delete partition directory and file."""

        handler = PartitionedParquetHandler(storage_config)

        partition_date = date(2024, 1, 15)

        df = sample_telemetry_dataframe.copy()
        if 'partition_date' in df.columns:
            df = df.drop(columns=['partition_date'])

        # Create partition

        handler.save_partition(df, partition_date)

        assert handler.partition_exists(partition_date)

        # Delete it

        result: bool = handler.delete_partition(partition_date)

        assert result is True

        assert not handler.partition_exists(partition_date)

    def test_delete_partition_returns_false_when_missing(
        self,
        storage_config: StorageConfig,
    ) -> None:
        """Should return False when partition doesn't exist."""

        handler = PartitionedParquetHandler(storage_config)

        result: bool = handler.delete_partition(date(2024, 1, 15))

        assert result is False

    def test_delete_partitions_before_deletes_old_partitions(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should delete all partitions before cutoff date."""

        handler = PartitionedParquetHandler(storage_config)

        # Create 5 partitions

        for day in [13, 14, 15, 16, 17]:
            partition_date = date(2024, 1, day)

            df = sample_telemetry_dataframe.copy()
            if 'partition_date' in df.columns:
                df = df.drop(columns=['partition_date'])

            handler.save_partition(df, partition_date)

        # Delete partitions before Jan 16

        deleted_count: int = handler.delete_partitions_before(date(2024, 1, 16))

        # Should delete 3 partitions (13, 14, 15)

        assert deleted_count == 3

        # Verify partitions 16 and 17 still exist

        assert handler.partition_exists(date(2024, 1, 16))

        assert handler.partition_exists(date(2024, 1, 17))

        # Verify partitions 13, 14, 15 are gone

        assert not handler.partition_exists(date(2024, 1, 13))

        assert not handler.partition_exists(date(2024, 1, 14))

        assert not handler.partition_exists(date(2024, 1, 15))


class TestPartitionedParquetHandlerMetadata:
    """Test metadata and partition listing operations."""

    def test_list_partition_dates_returns_empty_list_initially(
        self,
        storage_config: StorageConfig,
    ) -> None:
        """Should return empty list when no partitions exist."""

        handler = PartitionedParquetHandler(storage_config)

        result: list[date] = handler.list_partition_dates()

        assert result == []

    def test_list_partition_dates_returns_sorted_dates(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should return partition dates in chronological order."""

        handler = PartitionedParquetHandler(storage_config)

        # Create partitions in random order

        for day in [17, 13, 15, 14, 16]:
            partition_date = date(2024, 1, day)

            df = sample_telemetry_dataframe.copy()
            if 'partition_date' in df.columns:
                df = df.drop(columns=['partition_date'])

            handler.save_partition(df, partition_date)

        result: list[date] = handler.list_partition_dates()

        # Should be sorted

        expected = [date(2024, 1, d) for d in [13, 14, 15, 16, 17]]

        assert result == expected

    def test_get_latest_partition_date_returns_none_when_empty(
        self,
        storage_config: StorageConfig,
    ) -> None:
        """Should return None when no partitions exist."""

        handler = PartitionedParquetHandler(storage_config)

        result: date | None = handler.get_latest_partition_date()

        assert result is None

    def test_get_latest_partition_date_returns_most_recent(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should return the most recent partition date."""

        handler = PartitionedParquetHandler(storage_config)

        # Create partitions

        for day in [13, 14, 15, 16, 17]:
            partition_date = date(2024, 1, day)

            df = sample_telemetry_dataframe.copy()
            if 'partition_date' in df.columns:
                df = df.drop(columns=['partition_date'])

            handler.save_partition(df, partition_date)

        result: date | None = handler.get_latest_partition_date()

        assert result == date(2024, 1, 17)

    def test_get_earliest_partition_date_returns_oldest(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should return the earliest partition date."""

        handler = PartitionedParquetHandler(storage_config)

        # Create partitions

        for day in [13, 14, 15, 16, 17]:
            partition_date = date(2024, 1, day)

            df = sample_telemetry_dataframe.copy()
            if 'partition_date' in df.columns:
                df = df.drop(columns=['partition_date'])

            handler.save_partition(df, partition_date)

        result: date | None = handler.get_earliest_partition_date()

        assert result == date(2024, 1, 13)

    def test_get_statistics_returns_summary(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should return summary statistics about partitions."""

        handler = PartitionedParquetHandler(storage_config)

        # Create partitions

        for day in [15, 16, 17]:
            partition_date = date(2024, 1, day)

            df = sample_telemetry_dataframe.copy()
            if 'partition_date' in df.columns:
                df = df.drop(columns=['partition_date'])

            handler.save_partition(df, partition_date)

        stats: dict[str, int | float | str | None] = handler.get_statistics()

        assert stats['partition_count'] == 3

        assert stats['earliest_date'] == '2024-01-15'

        assert stats['latest_date'] == '2024-01-17'

        assert isinstance(stats['total_size_mb'], float)

        assert stats['total_size_mb'] > 0


class TestPartitionedParquetHandlerSizeOperations:
    """Test partition and total size operations."""

    def test_get_partition_size_returns_none_when_missing(
        self,
        storage_config: StorageConfig,
    ) -> None:
        """Should return None when partition doesn't exist."""

        handler = PartitionedParquetHandler(storage_config)

        result: float | None = handler.get_partition_size(date(2024, 1, 15))

        assert result is None

    def test_get_partition_size_returns_size(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should return partition size in requested unit."""

        handler = PartitionedParquetHandler(storage_config)

        partition_date = date(2024, 1, 15)

        df = sample_telemetry_dataframe.copy()
        if 'partition_date' in df.columns:
            df = df.drop(columns=['partition_date'])

        handler.save_partition(df, partition_date)

        size_mb: float | None = handler.get_partition_size(partition_date, 'mb')

        assert size_mb is not None

        assert isinstance(size_mb, float)

        assert size_mb > 0

    def test_get_total_size_sums_all_partitions(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should return total size across all partitions."""

        handler = PartitionedParquetHandler(storage_config)

        # Create multiple partitions

        for day in [15, 16, 17]:
            partition_date = date(2024, 1, day)

            df = sample_telemetry_dataframe.copy()
            if 'partition_date' in df.columns:
                df = df.drop(columns=['partition_date'])

            handler.save_partition(df, partition_date)

        total_size: float = handler.get_total_size('mb')

        assert total_size > 0

        # Total should be roughly 3x individual partition size

        partition_size: float | None = handler.get_partition_size(date(2024, 1, 15), 'mb')

        assert partition_size is not None

        # Allow some variance due to compression

        assert total_size > partition_size * 2.5

        assert total_size < partition_size * 4.0


class TestPartitionedParquetHandlerAtomicWrites:
    """Test atomic write behavior at partition level."""

    def test_atomic_write_uses_temp_file(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should use atomic write (temp file + rename) for each partition."""

        handler = PartitionedParquetHandler(storage_config)

        partition_date = date(2024, 1, 15)

        df = sample_telemetry_dataframe.copy()
        if 'partition_date' in df.columns:
            df = df.drop(columns=['partition_date'])

        # Save should succeed without errors

        handler.save_partition(df, partition_date)

        # Verify partition exists and is readable

        loaded = handler.load_partition(partition_date)

        assert loaded is not None

        assert len(loaded) == len(sample_telemetry_dataframe)

    def test_save_overwrites_existing_partition(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should overwrite existing partition atomically."""

        handler = PartitionedParquetHandler(storage_config)

        partition_date = date(2024, 1, 15)

        # Save initial data

        df1 = sample_telemetry_dataframe.iloc[:2].copy()
        if 'partition_date' in df1.columns:
            df1 = df1.drop(columns=['partition_date'])

        handler.save_partition(df1, partition_date)

        # Overwrite with different data

        df2 = sample_telemetry_dataframe.iloc[2:4].copy()
        if 'partition_date' in df2.columns:
            df2 = df2.drop(columns=['partition_date'])

        handler.save_partition(df2, partition_date)

        # Load should return new data

        loaded = handler.load_partition(partition_date)

        assert loaded is not None

        assert len(loaded) == 2

    def test_partition_count_property_updates(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """partition_count property should reflect current partition count."""

        handler = PartitionedParquetHandler(storage_config)

        assert handler.partition_count == 0

        # Add partitions

        for day in [15, 16, 17]:
            partition_date = date(2024, 1, day)

            df = sample_telemetry_dataframe.copy()
            if 'partition_date' in df.columns:
                df = df.drop(columns=['partition_date'])

            handler.save_partition(df, partition_date)

        assert handler.partition_count == 3

        # Delete one

        handler.delete_partition(date(2024, 1, 16))

        assert handler.partition_count == 2
