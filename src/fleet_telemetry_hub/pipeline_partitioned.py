# fleet_telemetry_hub/pipeline_partitioned.py
"""
Partitioned telemetry data pipeline for large-scale fleet data.

This is a modified version of the TelemetryPipeline that uses date-partitioned
Parquet storage instead of a single monolithic file. This design supports
datasets with billions of rows while maintaining reasonable memory usage.

Key Differences from Single-File Pipeline:
------------------------------------------
1. **Storage**: Each day's data is stored in a separate partition file
   (data/telemetry/date=2024-01-15/data.parquet)

2. **Memory Usage**: Only loads partitions within the lookback window,
   not the entire dataset

3. **Deduplication**: Handled per-partition, comparing new records only
   against overlapping date partitions

4. **BigQuery Integration**: Hive-style partitioning is natively compatible
   with BigQuery external tables and load jobs

Usage:
------
    from fleet_telemetry_hub.pipeline_partitioned import PartitionedTelemetryPipeline

    # One-liner for cron jobs
    PartitionedTelemetryPipeline('config.yaml').run()
"""

import logging
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from fleet_telemetry_hub.common import PartitionedParquetHandler, setup_logger
from fleet_telemetry_hub.config import TelemetryConfig, load_config
from fleet_telemetry_hub.operations import PROVIDER_FETCHER_CLASSES, DataFetcher
from fleet_telemetry_hub.provider import Provider, ProviderManager
from fleet_telemetry_hub.schema import (
    DEDUP_COLUMNS,
    SORT_COLUMNS,
    enforce_telemetry_schema,
)

__all__: list[str] = ['PartitionedPipelineError', 'PartitionedTelemetryPipeline']

logger: logging.Logger = logging.getLogger(__name__)

# Type alias for fetch functions
FetchFunction = Callable[[Provider, datetime, datetime], list[dict[str, Any]]]

# PROVIDER_FETCH_FUNCTIONS: dict[str, FetchFunction] = {
#     'motive': fetch_motive_data,
#     'samsara': fetch_samsara_data,
# }


class PartitionedPipelineError(Exception):
    """
    Raised when the partitioned pipeline encounters a fatal error.

    Attributes:
        message: Human-readable error description.
        batch_index: Which batch failed (1-indexed), if applicable.
        partial_data_saved: Whether any data was saved before the error.
        affected_partitions: Which date partitions were being processed.
    """

    def __init__(
        self,
        message: str,
        batch_index: int | None = None,
        partial_data_saved: bool = False,
        affected_partitions: list[date] | None = None,
    ) -> None:
        super().__init__(message)
        self.batch_index: int | None = batch_index
        self.partial_data_saved: bool = partial_data_saved
        self.affected_partitions: list[date] | None = affected_partitions


class PartitionedTelemetryPipeline:
    """
    Orchestrates telemetry extraction to date-partitioned Parquet storage.

    This pipeline is designed for large-scale datasets (millions/billions of rows)
    where loading the entire dataset into memory is not feasible. Data is
    partitioned by date, with each day stored in a separate Parquet file.

    Partition Strategy:
        Records are assigned to partitions based on their timestamp's DATE
        (UTC). A record at 2024-01-15T23:59:59Z goes into the 2024-01-15
        partition, while 2024-01-16T00:00:01Z goes into 2024-01-16.

    Lookback and Deduplication:
        Late-arriving records (records with timestamps in the past that arrive
        after their date has already been processed) are handled by the
        lookback_days configuration. On each run:
        1. Find the latest partition date
        2. Fetch data starting from (latest_date - lookback_days)
        3. For each affected partition, merge new records with existing data
        4. Deduplicate on (vin, timestamp), keeping the latest record

    Attributes:
        config: The loaded TelemetryConfig instance (read-only).
        file_handler: The partitioned file handler (read-only).
    """

    def __init__(self, config_path: Path | str) -> None:
        """
        Initialize the partitioned telemetry pipeline.

        Args:
            config_path: Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValidationError: If the config file fails Pydantic validation.
            OSError: If the partition base directory cannot be created.
            PartitionedPipelineError: If no providers are enabled.
        """
        # Normalize to Path for consistent handling
        config_path = Path(config_path)

        # Load and validate configuration
        self._config: TelemetryConfig = load_config(config_path)

        # Configure logging first so all subsequent operations are logged
        setup_logger(config=self._config.logging)

        logger.info(
            'Initializing PartitionedTelemetryPipeline from config: %s',
            config_path,
        )

        # Initialize partitioned storage handler
        self._file_handler: PartitionedParquetHandler = PartitionedParquetHandler(
            self._config.storage
        )

        self._provider_manager: ProviderManager = ProviderManager.from_config(
            self._config
        )

        # Initialize fetchers once (enables vehicle caching for Motive)
        self._fetchers: dict[str, DataFetcher] = {}
        for provider_name, provider in self._provider_manager.items():
            fetcher_class: Callable[[Provider], DataFetcher] | None = (
                PROVIDER_FETCHER_CLASSES.get(provider_name)
            )
            if fetcher_class is not None:
                self._fetchers[provider_name] = fetcher_class(provider)
                logger.debug('Initialized %s fetcher', provider_name)
            else:
                logger.warning(
                    'No fetcher class for provider %r, will be skipped',
                    provider_name,
                )

        # Validate we have at least one provider
        self._validate_providers()

        logger.info(
            'Pipeline initialized: providers=%s, storage=%s, partitions=%d',
            list(self._provider_manager.list_providers()),
            self._file_handler.base_path,
            self._file_handler.partition_count,
        )

    def _validate_providers(self) -> None:
        """
        Validate that at least one provider has a working fetcher.

        Raises:
            PartitionedPipelineError: If no providers are enabled or none have fetchers.
        """
        if not self._provider_manager:
            raise PartitionedPipelineError(
                'No providers are enabled in configuration. '
                'Enable at least one provider (motive, samsara) to run the pipeline.'
            )

        enabled_providers: list[str] = self._provider_manager.list_providers()
        if not self._fetchers:
            raise PartitionedPipelineError(
                f'No enabled providers have fetcher classes implemented. '
                f'Enabled: {enabled_providers}. '
                f'Supported: {list(PROVIDER_FETCHER_CLASSES.keys())}.'
            )

        # Log any enabled providers that don't have fetchers
        skipped_providers: set[str] = set(enabled_providers) - set(
            self._fetchers.keys()
        )
        if skipped_providers:
            logger.warning(
                'Some enabled providers have no fetcher class and will be skipped: %s',
                sorted(skipped_providers),
            )

    @property
    def config(self) -> TelemetryConfig:
        """The loaded configuration (read-only)."""
        return self._config

    @property
    def file_handler(self) -> PartitionedParquetHandler:
        """The partitioned file handler (read-only)."""
        return self._file_handler

    def run(self) -> None:
        """
        Execute the partitioned telemetry data pipeline.

        Performs:
        1. Determine start datetime (from latest partition or config default)
        2. Generate time-based batches from start to now
        3. For each batch, fetch from all enabled providers
        4. Group records by date, save to respective partitions
        5. Deduplicate within each partition
        6. Log summary statistics

        Each partition is saved independently, so partial progress is preserved
        if execution is interrupted.

        Raises:
            PartitionedPipelineError: If all providers fail for any batch.
        """
        run_start_time: datetime = datetime.now(UTC)
        start_datetime: datetime = self._determine_start_datetime()
        end_datetime: datetime = run_start_time

        logger.info(
            'Starting partitioned pipeline run: %s to %s',
            start_datetime.isoformat(),
            end_datetime.isoformat(),
        )

        batches: list[tuple[datetime, datetime]] = self._generate_batches(
            start_datetime,
            end_datetime,
        )

        if not batches:
            logger.info('No batches to process (start >= end). Pipeline complete.')
            return

        logger.info(
            'Processing %d batches (increment=%.2f days)',
            len(batches),
            self._config.pipeline.batch_increment_days,
        )

        total_records_fetched: int = 0
        batches_with_data: int = 0
        partitions_updated: set[date] = set()

        for batch_index, (batch_start, batch_end) in enumerate(batches, start=1):
            logger.info(
                'Processing batch %d/%d: %s to %s',
                batch_index,
                len(batches),
                batch_start.isoformat(),
                batch_end.isoformat(),
            )

            # Fetch from all providers (independent failures)
            batch_records: list[dict[str, Any]] | None = self._fetch_batch(
                batch_start,
                batch_end,
            )

            # If ALL providers failed, abort the pipeline
            if batch_records is None:
                raise PartitionedPipelineError(
                    f'All providers failed for batch {batch_index}/{len(batches)} '
                    f'({batch_start.isoformat()} to {batch_end.isoformat()}). '
                    f'Partial data from previous batches is preserved.',
                    batch_index=batch_index,
                    partial_data_saved=batches_with_data > 0,
                    affected_partitions=sorted(partitions_updated),
                )

            # Empty batch (no records, but no errors) is fine - just skip saving
            if not batch_records:
                logger.info(
                    'Batch %d/%d returned no records, skipping save',
                    batch_index,
                    len(batches),
                )
                continue

            # Save to partitions (handles grouping by date internally)
            batch_partitions: dict[date, int] = self._save_batch_partitioned(
                batch_records
            )

            total_records_fetched += len(batch_records)
            batches_with_data += 1
            partitions_updated.update(batch_partitions.keys())

            logger.info(
                'Batch %d/%d complete: %d records saved to %d partition(s)',
                batch_index,
                len(batches),
                len(batch_records),
                len(batch_partitions),
            )

        self._log_run_summary(
            run_start_time=run_start_time,
            batches_processed=len(batches),
            batches_with_data=batches_with_data,
            records_fetched=total_records_fetched,
            partitions_updated=len(partitions_updated),
        )

    def _determine_start_datetime(self) -> datetime:
        """
        Determine the start datetime for this pipeline run.

        Uses the latest partition date (not the actual latest timestamp in the data)
        for efficiency. We don't need to load any Parquet files, just scan
        directory names.

        Returns:
            Timezone-aware UTC datetime for the start of data fetching.
        """
        latest_partition_date: date | None = (
            self._file_handler.get_latest_partition_date()
        )

        if latest_partition_date is None:
            # Initial run: use config default (midnight UTC)
            start_datetime: datetime = datetime.fromisoformat(
                self._config.pipeline.default_start_date
            ).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)

            logger.info(
                'No existing partitions found. Starting from config default: %s',
                start_datetime.isoformat(),
            )
            return start_datetime

        # Incremental run: latest partition date minus lookback
        lookback: timedelta = timedelta(days=self._config.pipeline.lookback_days)

        # Convert date to datetime at start of day
        latest_datetime: datetime = datetime(
            year=latest_partition_date.year,
            month=latest_partition_date.month,
            day=latest_partition_date.day,
            tzinfo=UTC,
        )

        start_datetime = latest_datetime - lookback

        logger.info(
            'Latest partition: %s. Starting from %s (lookback=%d days). '
            'Total partitions: %d',
            latest_partition_date.isoformat(),
            start_datetime.isoformat(),
            self._config.pipeline.lookback_days,
            self._file_handler.partition_count,
        )

        return start_datetime

    def _generate_batches(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Generate time-based batches for incremental processing.

        Divides the time range into chunks based on the configured
        batch_increment_days. The final batch may be smaller than the
        increment if it reaches end_datetime.

        Args:
            start_datetime: Start of the overall time range (inclusive).
            end_datetime: End of the overall time range (inclusive).

        Returns:
            List of (batch_start, batch_end) datetime tuples. Empty list if
            start_datetime >= end_datetime.
        """
        if start_datetime >= end_datetime:
            logger.warning(
                'Start datetime %s is not before end datetime %s.',
                start_datetime.isoformat(),
                end_datetime.isoformat(),
            )
            return []

        increment: timedelta = timedelta(
            days=self._config.pipeline.batch_increment_days
        )
        batches: list[tuple[datetime, datetime]] = []

        current_start: datetime = start_datetime
        while current_start < end_datetime:
            # Batch end is either start + increment or end_datetime, whichever is earlier
            current_end: datetime = min(current_start + increment, end_datetime)
            batches.append((current_start, current_end))
            current_start = current_end

        return batches

    def _fetch_batch(
        self,
        batch_start: datetime,
        batch_end: datetime,
    ) -> list[dict[str, Any]] | None:
        """
        Fetch telemetry data from all enabled providers for a single time batch.

        Iterates through initialized fetchers (one per enabled provider) and
        collects records from each. Providers are fetched independently—if one
        fails, the others still contribute their data.

        The fetchers are instantiated once at pipeline init, enabling stateful
        optimizations like Motive's vehicle list caching across batch calls.

        Args:
            batch_start: Start of the batch time range (inclusive, timezone-aware UTC).
            batch_end: End of the batch time range (inclusive, timezone-aware UTC).

        Returns:
            List of record dictionaries if at least one provider succeeded.
            Records match TELEMETRY_COLUMNS schema and can be directly converted
            to DataFrame. Returns None if ALL providers failed, signaling the
            pipeline should abort.

        Raises:
            Does not raise directly. Provider exceptions are caught and logged;
            the method returns None only when all providers fail.

        Side Effects:
            - Makes HTTP requests to provider APIs
            - Logs progress at DEBUG level, errors at ERROR level with traceback
            - May populate fetcher caches (e.g., Motive vehicle list on first call)
        """
        all_records: list[dict[str, Any]] = []
        providers_attempted: int = 0
        providers_succeeded: int = 0

        for provider_name, fetcher in self._fetchers.items():
            providers_attempted += 1

            try:
                records: list[dict[str, Any]] = fetcher.fetch_data(
                    batch_start,
                    batch_end,
                )
                all_records.extend(records)
                providers_succeeded += 1
                logger.debug(
                    'Provider %r returned %d records',
                    provider_name,
                    len(records),
                )
            except Exception:
                logger.exception(
                    'Provider %r fetch failed for batch %s to %s',
                    provider_name,
                    batch_start.isoformat(),
                    batch_end.isoformat(),
                )

        # If no providers were even attempted, something is misconfigured
        if providers_attempted == 0:
            logger.error('No providers with fetch functions were available')
            return None

        # Return None only if ALL attempted providers failed (abort signal)
        if providers_succeeded == 0:
            logger.error('All %d provider(s) failed for batch', providers_attempted)
            return None

        logger.debug(
            'Batch fetch complete: %d/%d providers succeeded, %d total records',
            providers_succeeded,
            providers_attempted,
            len(all_records),
        )

        return all_records

    def _save_batch_partitioned(
        self,
        records: list[dict[str, Any]],
    ) -> dict[date, int]:
        """
        Save a batch of records to date-partitioned Parquet files.

        Groups records by their timestamp's date (UTC), then saves each group
        to its respective partition. Existing partition data is merged and
        deduplicated.

        Args:
            records: List of record dictionaries from fetch functions.

        Returns:
            Dictionary mapping partition dates to record counts saved.

        Side Effects:
            - Reads existing partition files (if present)
            - Writes updated partition files
        """
        # Convert to DataFrame and enforce schema
        new_dataframe: pd.DataFrame = pd.DataFrame(records)
        new_dataframe = enforce_telemetry_schema(new_dataframe)

        # Extract partition date from timestamp
        # The timestamp column is UTC datetime after enforce_telemetry_schema
        new_dataframe['partition_date'] = new_dataframe['timestamp'].dt.date  # pyright: ignore[reportAttributeAccessIssue]

        # Use the handler's partitioned save method
        # This handles grouping by date, loading existing partitions,
        # merging, deduplicating, and saving
        records_saved: dict[date, int] = self._file_handler.save_partitioned(
            dataframe=new_dataframe,
            date_column='partition_date',
            deduplicate=True,
            dedup_columns=DEDUP_COLUMNS,
        )

        return records_saved

    def _log_run_summary(
        self,
        run_start_time: datetime,
        batches_processed: int,
        batches_with_data: int,
        records_fetched: int,
        partitions_updated: int,
    ) -> None:
        """Log summary statistics for the pipeline run."""
        run_duration: timedelta = datetime.now(UTC) - run_start_time

        stats: dict[str, int | float | str | None] = self._file_handler.get_statistics()

        logger.info(
            'Partitioned pipeline run complete: '
            '%d total partitions, date range %s to %s. '
            'This run: %d batches, %d with data, %d records fetched, '
            '%d partitions updated. '
            'Total size: %.2f MB. Duration: %s',
            stats['partition_count'],
            stats['earliest_date'],
            stats['latest_date'],
            batches_processed,
            batches_with_data,
            records_fetched,
            partitions_updated,
            stats['total_size_mb'],
            run_duration,
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def load_date_range(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame | None:
        """
        Load data for a specific date range.

        Convenience method for ad-hoc analysis or debugging. For production
        queries at scale, use BigQuery or another query engine directly.

        Args:
            start_date: First date to include (inclusive).
            end_date: Last date to include (inclusive).

        Returns:
            Combined DataFrame from all partitions in the range, sorted by
            (vin, timestamp), or None if no data exists in the range.
        """
        dataframe: pd.DataFrame | None = self._file_handler.load_date_range(
            start_date=start_date,
            end_date=end_date,
        )

        if dataframe is not None and not dataframe.empty:
            dataframe = dataframe.sort_values(SORT_COLUMNS).reset_index(drop=True)

        return dataframe

    def delete_old_partitions(self, retention_days: int) -> int:
        """
        Delete partitions older than the retention period.

        Useful for implementing data retention policies.

        Args:
            retention_days: Keep partitions from the last N days.

        Returns:
            Number of partitions deleted.
        """
        cutoff_date: date = (datetime.now(UTC) - timedelta(days=retention_days)).date()

        deleted_count: int = self._file_handler.delete_partitions_before(cutoff_date)

        if deleted_count > 0:
            logger.info(
                'Deleted %d partitions older than %s (%d day retention)',
                deleted_count,
                cutoff_date.isoformat(),
                retention_days,
            )

        return deleted_count
