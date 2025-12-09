# fleet_telemetry_hub/common/partitioned_file_io.py
"""
Partitioned Parquet file handler for large-scale time-series telemetry data.

This module provides date-partitioned storage using Hive-style directory
structure (date=YYYY-MM-DD/) which is natively compatible with BigQuery
external tables and GCS loading.

Design Philosophy:
------------------
- Each date partition is an independent Parquet file
- Operations only touch partitions within the specified date range
- Atomic writes at the partition level (temp file + rename)
- Memory pressure scales with lookback window, not total dataset size

Directory Structure:
--------------------
    base_path/
    ├── date=2024-01-15/
    │   └── data.parquet
    ├── date=2024-01-16/
    │   └── data.parquet
    └── _metadata.json  (optional, for faster startup)

BigQuery Compatibility:
-----------------------
The Hive-style partitioning (date=YYYY-MM-DD/) is automatically recognized by:
- BigQuery external tables with `hive_partition_uri_prefix`
- BigQuery `LOAD DATA` with auto-schema detection
- Most Spark/Dask/Polars readers

Thread Safety:
--------------
NOT thread-safe. Concurrent writes to the same partition will cause corruption.
Concurrent writes to different partitions are safe. Implement external locking
if you need concurrent access to the same partition.

Usage:
------
    from datetime import date
    from fleet_telemetry_hub.config import PartitionedStorageConfig
    from fleet_telemetry_hub.common.partitioned_file_io import PartitionedParquetHandler

    config = PartitionedStorageConfig(base_path='data/telemetry/')
    handler = PartitionedParquetHandler(config)

    # Load specific date range (for lookback deduplication)
    lookback_data = handler.load_date_range(
        start_date=date(2024, 1, 10),
        end_date=date(2024, 1, 17),
    )

    # Save records to their respective date partitions
    handler.save_partitioned(dataframe_with_date_column)

    # Get latest partition date (for determining pipeline start)
    latest = handler.get_latest_partition_date()
"""

import json
import logging
import tempfile
from contextlib import suppress
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
from pandas.api.typing import DataFrameGroupBy
from pyarrow import (
    ArrowInvalid as _ArrowInvalid,  # pyright: ignore[reportUnknownVariableType]
    ArrowIOError as _ArrowIOError,  # pyright: ignore[reportUnknownVariableType]
)

from fleet_telemetry_hub.config import StorageConfig

# PyArrow exception types for proper error handling.
# These are used in except clauses, so we need them at runtime.
# The type stubs are incomplete, hence we cast for cleaner code.
ArrowInvalid: type[Exception] = cast(type[Exception], _ArrowInvalid)
ArrowIOError: type[Exception] = cast(type[Exception], _ArrowIOError)

__all__: list[str] = ['PartitionedParquetHandler']

logger: logging.Logger = logging.getLogger(__name__)

FileSizeUnit = Literal['bytes', 'kb', 'mb', 'gb']

# Hive-style partition directory format
PARTITION_DIR_FORMAT: str = 'date={date}'
PARTITION_FILE_NAME: str = 'data.parquet'
METADATA_FILE_NAME: str = '_metadata.json'


class PartitionedParquetHandler:
    """
    Manages a directory of date-partitioned Parquet files.

    Each partition is a directory named `date=YYYY-MM-DD` containing a single
    `data.parquet` file. This structure is compatible with Hive partitioning
    and recognized by BigQuery, Spark, Dask, and other big data tools.

    Partition-Level Atomicity:
        Each partition file is written atomically using temp file + rename.
        If the process crashes mid-write, only that one partition may be
        affected; other partitions remain intact.

    Attributes:
        base_path: Root directory containing all partitions (read-only).
        partition_count: Number of partition directories present (read-only).
    """

    def __init__(self, storage_config: StorageConfig) -> None:
        """
        Initialize the partitioned Parquet handler.

        Creates the base directory if it doesn't exist.

        Args:
            storage_config: Configuration containing base path and compression
                settings. A reference is stored, not copied.

        Raises:
            OSError: If base directory cannot be created (permissions, etc).
            PermissionError: If base directory exists but is not writable.
        """
        self._config: StorageConfig = storage_config

        # Ensure base directory exists at initialization time
        self._config.parquet_path.mkdir(parents=True, exist_ok=True)

        # Initialize metadata cache
        self._metadata_cache: PartitionMetadataCache = PartitionMetadataCache(
            self._config.parquet_path
        )

        logger.info(
            'Initialized PartitionedParquetHandler: base_path=%r, compression=%r',
            self._config.parquet_path,
            self._config.parquet_compression,
        )

    @property
    def base_path(self) -> Path:
        """Root directory containing all partition directories."""
        return self._config.parquet_path

    @property
    def partition_count(self) -> int:
        """Number of partition directories currently present."""
        return len(self._list_partition_directories())

    def _list_partition_directories(self) -> list[Path]:
        """
        List all partition directories in sorted order (oldest first).

        Returns:
            List of partition directory paths matching the Hive pattern.
        """
        if not self._config.parquet_path.exists():
            return []

        partition_dirs: list[Path] = [
            path
            for path in self._config.parquet_path.iterdir()
            if path.is_dir() and path.name.startswith('date=')
        ]

        # Sort by date (lexicographic sort works for ISO dates)
        return sorted(partition_dirs)

    def _get_partition_path(self, partition_date: date) -> Path:
        """
        Get the full path to a partition's Parquet file.

        Args:
            partition_date: The date for this partition.

        Returns:
            Path to the data.parquet file within the partition directory.
        """
        partition_dir_name: str = PARTITION_DIR_FORMAT.format(
            date=partition_date.isoformat()
        )
        return self._config.parquet_path / partition_dir_name / PARTITION_FILE_NAME

    def _parse_partition_date(self, partition_dir: Path) -> date | None:
        """
        Extract the date from a partition directory name.

        Args:
            partition_dir: Path to a partition directory (e.g., .../date=2024-01-15)

        Returns:
            Parsed date object, or None if the directory name is malformed.
        """
        # Expected format: date=YYYY-MM-DD
        dir_name: str = partition_dir.name
        if not dir_name.startswith('date='):
            return None

        date_str: str = dir_name[5:]  # Remove 'date=' prefix
        try:
            return date.fromisoformat(date_str)
        except ValueError:
            logger.warning(
                'Invalid partition directory name (cannot parse date): %r',
                partition_dir,
            )
            return None

    def get_latest_partition_date(self) -> date | None:
        """
        Get the most recent partition date, using cache if available.

        The cache avoids directory scanning on repeated calls. Cache is
        automatically invalidated on partition writes/deletes.

        Returns:
            The latest partition date, or None if no partitions exist.
        """
        # Try cache first
        cached_metadata: dict[str, Any] | None = self._metadata_cache.read()
        if cached_metadata is not None and 'latest_date' in cached_metadata:
            try:
                return date.fromisoformat(cached_metadata['latest_date'])
            except (ValueError, TypeError):
                # Cache corrupted, fall through to directory scan
                logger.warning('Invalid cached latest_date, rescanning directories')

        # Cache miss or invalid: scan directories
        partition_dirs: list[Path] = self._list_partition_directories()

        if not partition_dirs:
            return None

        latest_dir: Path = partition_dirs[-1]
        latest_date: date | None = self._parse_partition_date(latest_dir)

        # Update cache for next time
        if latest_date is not None:
            self._metadata_cache.write(
                latest_date=latest_date,
                partition_count=len(partition_dirs),
            )

        return latest_date

    def get_earliest_partition_date(self) -> date | None:
        """
        Get the earliest partition date without loading any data.

        Useful for determining the full date range of available data.

        Returns:
            The earliest partition date, or None if no partitions exist.
        """
        partition_dirs: list[Path] = self._list_partition_directories()

        if not partition_dirs:
            return None

        # Directories are sorted oldest-first, so take the first one
        earliest_dir: Path = partition_dirs[0]
        return self._parse_partition_date(earliest_dir)

    def list_partition_dates(self) -> list[date]:
        """
        List all partition dates in chronological order.

        Returns:
            List of dates for which partitions exist, sorted ascending.
        """
        partition_dirs: list[Path] = self._list_partition_directories()
        dates: list[date] = []

        for partition_dir in partition_dirs:
            partition_date: date | None = self._parse_partition_date(partition_dir)
            if partition_date is not None:
                dates.append(partition_date)

        return dates

    def partition_exists(self, partition_date: date) -> bool:
        """Check whether a partition exists for the given date."""
        partition_path: Path = self._get_partition_path(partition_date)
        return partition_path.exists()

    def load_partition(self, partition_date: date) -> pd.DataFrame | None:
        """
        Load a single partition's data.

        Returns None for missing or corrupt partitions rather than raising,
        allowing callers to treat this as "no data for this date."

        Args:
            partition_date: The date of the partition to load.

        Returns:
            DataFrame containing the partition's data, or None if the
            partition is missing, unreadable, or corrupt.
        """
        partition_path: Path = self._get_partition_path(partition_date)

        if not partition_path.exists():
            return None

        try:
            dataframe: pd.DataFrame = pd.read_parquet(partition_path)
            logger.debug(
                'Loaded partition %s: %d records',
                partition_date.isoformat(),
                len(dataframe),
            )
            return dataframe

        except (OSError, ArrowInvalid, ArrowIOError) as read_error:
            logger.exception(
                'Failed to read partition %s from %r: %s',
                partition_date.isoformat(),
                partition_path,
                read_error,
            )
            return None

    def load_date_range(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame | None:
        """
        Load all partitions within a date range (inclusive).

        Only loads partitions that exist within the range. Missing dates
        are silently skipped—this is expected when data is sparse or when
        fetching a range that extends before the first available data.

        This is the primary method for loading lookback data for deduplication.

        Args:
            start_date: First date to include (inclusive).
            end_date: Last date to include (inclusive).

        Returns:
            Combined DataFrame from all partitions in the range, or None
            if no partitions exist in the range. The DataFrame is NOT
            sorted; caller should sort if order matters.

        Example:
            # Load 7 days of lookback for deduplication
            lookback_data = handler.load_date_range(
                start_date=date(2024, 1, 10),
                end_date=date(2024, 1, 17),
            )
        """
        if start_date > end_date:
            logger.warning(
                'start_date %s is after end_date %s. Returning None.',
                start_date.isoformat(),
                end_date.isoformat(),
            )
            return None

        # Find existing partitions within the date range
        all_partition_dates: list[date] = self.list_partition_dates()
        range_partition_dates: list[date] = [
            partition_date
            for partition_date in all_partition_dates
            if start_date <= partition_date <= end_date
        ]

        if not range_partition_dates:
            logger.debug(
                'No partitions found in range %s to %s',
                start_date.isoformat(),
                end_date.isoformat(),
            )
            return None

        logger.debug(
            'Loading %d partitions from %s to %s',
            len(range_partition_dates),
            start_date.isoformat(),
            end_date.isoformat(),
        )

        # Load each partition and concatenate
        dataframes: list[pd.DataFrame] = []
        for partition_date in range_partition_dates:
            partition_df: pd.DataFrame | None = self.load_partition(partition_date)
            if partition_df is not None and not partition_df.empty:
                dataframes.append(partition_df)

        if not dataframes:
            return None

        combined_dataframe: pd.DataFrame = pd.concat(dataframes, ignore_index=True)
        logger.info(
            'Loaded %d partitions (%d total records) from %s to %s',
            len(dataframes),
            len(combined_dataframe),
            start_date.isoformat(),
            end_date.isoformat(),
        )

        return combined_dataframe

    def save_partition(
        self,
        dataframe: pd.DataFrame,
        partition_date: date,
    ) -> None:
        """
        Save a DataFrame to a specific partition atomically.

        If the partition already exists, it is overwritten. Use this for
        incremental updates where you've already handled deduplication
        by loading the existing partition, merging, and deduplicating.

        Atomic Write:
            Writes to a temporary file in the partition directory, then
            atomically renames. The original partition remains intact until
            the new file is fully written.

        Args:
            dataframe: The DataFrame to persist. All records in this DataFrame
                should belong to the specified partition_date.
            partition_date: The date partition to write to.

        Raises:
            OSError: File system errors (permissions, disk full, etc).
            ArrowInvalid: DataFrame contains types that cannot be serialized.
            ArrowIOError: I/O errors during write.
        """
        partition_path: Path = self._get_partition_path(partition_date)
        partition_dir: Path = partition_path.parent

        # Ensure partition directory exists
        partition_dir.mkdir(parents=True, exist_ok=True)

        record_count: int = len(dataframe)

        try:
            # Write to temp file in the same directory for atomic rename
            with tempfile.NamedTemporaryFile(
                mode='wb',
                suffix='.parquet.tmp',
                dir=partition_dir,
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)

            logger.debug(
                'Writing partition %s: %d records to temp file %r',
                partition_date.isoformat(),
                record_count,
                temp_path,
            )

            dataframe.to_parquet(
                temp_path,
                index=False,
                compression=self._config.parquet_compression,
            )

            # Atomic rename
            temp_path.replace(partition_path)

            logger.info(
                'Saved partition %s: %d records',
                partition_date.isoformat(),
                record_count,
            )

        except (OSError, ArrowInvalid, ArrowIOError) as write_error:
            logger.exception(
                'Failed to save partition %s (%d records): %s',
                partition_date.isoformat(),
                record_count,
                write_error,
            )
            # Clean up temp file if it exists
            if 'temp_path' in locals() and temp_path.exists():  # pyright: ignore[reportPossiblyUnboundVariable]
                with suppress(OSError):
                    temp_path.unlink()  # pyright: ignore[reportPossiblyUnboundVariable]
            raise

        # After successful save, update cache if this is the new latest
        cached_metadata: dict[str, Any] | None = self._metadata_cache.read()
        cached_latest_str: str | None = (
            cached_metadata.get('latest_date') if cached_metadata else None
        )

        if cached_latest_str is None:
            # No cache, write fresh
            self._metadata_cache.write(
                latest_date=partition_date,
                partition_count=self.partition_count,
                total_records=record_count,
            )
        else:
            try:
                cached_latest: date = date.fromisoformat(cached_latest_str)
                if partition_date >= cached_latest:
                    self._metadata_cache.write(
                        latest_date=partition_date,
                        partition_count=self.partition_count,
                        total_records=record_count,
                    )
            except ValueError:
                # Corrupted cache, rewrite
                self._metadata_cache.write(
                    latest_date=partition_date,
                    partition_count=self.partition_count,
                    total_records=record_count,
                )

    def save_partitioned(
        self,
        dataframe: pd.DataFrame,
        date_column: str = 'partition_date',
        deduplicate: bool = True,
        dedup_columns: list[str] | None = None,
    ) -> dict[date, int]:
        """
        Save a DataFrame by partitioning on a date column.

        Automatically groups records by date and saves each group to its
        respective partition. If partitions already exist, optionally merges
        and deduplicates with existing data.

        This is the primary save method for pipeline batch operations.

        Args:
            dataframe: DataFrame containing records to save. Must have a column
                matching `date_column` with date or datetime values.
            date_column: Name of the column containing partition dates.
                If the column contains datetimes, they are converted to dates.
            deduplicate: If True, load existing partition data and deduplicate
                after merging. If False, overwrite partitions entirely.
            dedup_columns: Columns to use for deduplication (passed to
                drop_duplicates). If None, uses all columns.

        Returns:
            Dictionary mapping partition dates to record counts saved.

        Raises:
            ValueError: If date_column is not present in the DataFrame.
            OSError: If any partition write fails.

        Example:
            # Save batch with automatic partitioning and deduplication
            records_saved = handler.save_partitioned(
                dataframe=batch_df,
                date_column='partition_date',
                deduplicate=True,
                dedup_columns=['vin', 'timestamp'],
            )
        """
        if date_column not in dataframe.columns:
            raise ValueError(f'DataFrame missing required date column: {date_column!r}')

        # Ensure we have date objects, not datetimes
        working_df: pd.DataFrame = dataframe.copy()

        # Convert datetime column to date if needed
        if pd.api.types.is_datetime64_any_dtype(working_df[date_column]):
            working_df[date_column] = working_df[date_column].dt.date  # pyright: ignore[reportAttributeAccessIssue]
        else:
            # Ensure it's proper date objects
            working_df[date_column] = pd.to_datetime(working_df[date_column]).dt.date

        # Group by partition date
        grouped: DataFrameGroupBy[date, pd.DataFrame] = working_df.groupby(  # pyright: ignore[reportAssignmentType, reportInvalidTypeArguments]
            date_column, observed=True
        )

        records_saved: dict[date, int] = {}

        for partition_date, group_df in grouped:
            # Remove the partition date column before saving
            # (it's encoded in the directory structure)
            partition_df: pd.DataFrame = group_df.drop(columns=[date_column])

            if deduplicate:
                # Load existing partition data if present
                existing_df: pd.DataFrame | None = self.load_partition(partition_date)

                if existing_df is not None and not existing_df.empty:
                    # Merge and deduplicate
                    combined_df: pd.DataFrame = pd.concat(
                        [existing_df, partition_df],
                        ignore_index=True,
                    )

                    pre_dedup_count: int = len(combined_df)
                    combined_df = combined_df.drop_duplicates(
                        subset=dedup_columns,
                        keep='last',  # Fresh data overwrites stale
                    )
                    post_dedup_count: int = len(combined_df)

                    logger.debug(
                        'Partition %s: merged %d existing + %d new = %d, '
                        '%d duplicates removed',
                        partition_date.isoformat(),
                        len(existing_df),
                        len(partition_df),
                        post_dedup_count,
                        pre_dedup_count - post_dedup_count,
                    )

                    partition_df = combined_df

            self.save_partition(partition_df, partition_date)
            records_saved[partition_date] = len(partition_df)

        logger.info(
            'Saved %d partitions, %d total records',
            len(records_saved),
            sum(records_saved.values()),
        )

        return records_saved

    def delete_partition(self, partition_date: date) -> bool:
        """
        Delete a partition and its directory.

        Useful for data retention policies or correcting bad data.

        Args:
            partition_date: The date of the partition to delete.

        Returns:
            True if partition was deleted, False if it didn't exist.

        Raises:
            OSError: If partition exists but cannot be deleted.
        """
        partition_path: Path = self._get_partition_path(partition_date)
        partition_dir: Path = partition_path.parent

        if not partition_path.exists():
            logger.debug(
                'Delete requested but partition does not exist: %s',
                partition_date.isoformat(),
            )
            return False

        # Delete the file, then the directory
        partition_path.unlink()

        # Only delete directory if it's now empty
        if partition_dir.exists() and not any(partition_dir.iterdir()):
            partition_dir.rmdir()

        # Invalidate cache since we don't know if this was the latest
        # Next get_latest_partition_date() call will rescan and rebuild
        self._metadata_cache.invalidate()

        logger.info('Deleted partition: %s', partition_date.isoformat())

        return True

    def delete_partitions_before(self, cutoff_date: date) -> int:
        """
        Delete all partitions older than the cutoff date.

        Useful for implementing data retention policies.

        Args:
            cutoff_date: Delete all partitions with dates strictly before
                this date.

        Returns:
            Number of partitions deleted.

        Raises:
            OSError: If any partition deletion fails.
        """
        partition_dates: list[date] = self.list_partition_dates()
        dates_to_delete: list[date] = [
            partition_date
            for partition_date in partition_dates
            if partition_date < cutoff_date
        ]

        deleted_count: int = 0
        for partition_date in dates_to_delete:
            if self.delete_partition(partition_date):
                deleted_count += 1

        if deleted_count > 0:
            logger.info(
                'Deleted %d partitions before %s',
                deleted_count,
                cutoff_date.isoformat(),
            )
            self._metadata_cache.invalidate()

        return deleted_count

    def get_total_size(self, unit: FileSizeUnit = 'mb') -> float:
        """
        Get the total size of all partitions.

        Args:
            unit: Size unit to return ('bytes', 'kb', 'mb', 'gb').

        Returns:
            Total size across all partition files.
        """
        total_bytes: int = 0

        for partition_dir in self._list_partition_directories():
            partition_file: Path = partition_dir / PARTITION_FILE_NAME
            if partition_file.exists():
                total_bytes += partition_file.stat().st_size

        divisors: dict[FileSizeUnit, float] = {
            'bytes': 1.0,
            'kb': 1024.0,
            'mb': 1024.0**2,
            'gb': 1024.0**3,
        }

        return total_bytes / divisors[unit]

    def get_partition_size(
        self,
        partition_date: date,
        unit: FileSizeUnit = 'mb',
    ) -> float | None:
        """
        Get the size of a specific partition.

        Args:
            partition_date: The date of the partition.
            unit: Size unit to return ('bytes', 'kb', 'mb', 'gb').

        Returns:
            Partition size in the requested unit, or None if partition
            doesn't exist.
        """
        partition_path: Path = self._get_partition_path(partition_date)

        if not partition_path.exists():
            return None

        size_bytes: int = partition_path.stat().st_size

        divisors: dict[FileSizeUnit, float] = {
            'bytes': 1.0,
            'kb': 1024.0,
            'mb': 1024.0**2,
            'gb': 1024.0**3,
        }

        return size_bytes / divisors[unit]

    def get_statistics(self) -> dict[str, int | float | str | None]:
        """
        Get summary statistics about the partitioned dataset.

        Returns:
            Dictionary containing partition count, date range, and size info.
        """
        partition_dates: list[date] = self.list_partition_dates()

        if not partition_dates:
            return {
                'partition_count': 0,
                'earliest_date': None,
                'latest_date': None,
                'total_size_mb': 0.0,
            }

        return {
            'partition_count': len(partition_dates),
            'earliest_date': partition_dates[0].isoformat(),
            'latest_date': partition_dates[-1].isoformat(),
            'total_size_mb': round(self.get_total_size('mb'), 2),
        }


# =============================================================================
# Metadata Caching (Optional Optimization)
# =============================================================================


class PartitionMetadataCache:
    """
    Optional metadata cache for faster startup.

    Stores the latest partition date and record counts in a JSON file,
    avoiding the need to scan all partition directories on startup.

    Usage:
        This is an optional optimization. The PartitionedParquetHandler
        works correctly without it, but may be slower to initialize with
        thousands of partitions.
    """

    def __init__(self, base_path: Path) -> None:
        self._metadata_path: Path = base_path / METADATA_FILE_NAME

    def read(self) -> dict[str, Any] | None:
        """Read cached metadata, or None if not present/valid."""
        if not self._metadata_path.exists():
            return None

        try:
            with self._metadata_path.open('r') as file_handle:
                output: dict[str, Any] = cast(dict[str, Any], json.load(file_handle))
                return output
        except (OSError, json.JSONDecodeError) as error:
            logger.warning('Failed to read metadata cache: %s', error)
            return None

    def write(
        self,
        latest_date: date,
        partition_count: int,
        total_records: int | None = None,
    ) -> None:
        """Write metadata to cache file."""
        metadata: dict[str, Any] = {
            'latest_date': latest_date.isoformat(),
            'partition_count': partition_count,
            'total_records': total_records,
            'updated_at': datetime.now(UTC).isoformat(),
        }

        try:
            with self._metadata_path.open('w') as file_handle:
                json.dump(metadata, file_handle, indent=2)
        except OSError as error:
            logger.warning('Failed to write metadata cache: %s', error)

    def invalidate(self) -> None:
        """Delete the metadata cache file."""
        if self._metadata_path.exists():
            with suppress(OSError):
                self._metadata_path.unlink()
