# fleet_telemetry_hub/pipeline.py
"""
Telemetry data pipeline for fetching, transforming, and storing fleet data.

This module provides the main pipeline orchestration for extracting vehicle
telemetry from multiple ELD providers (Motive and Samsara), combining them
into a unified schema, and persisting to Parquet storage.

Usage:
------
    from fleet_telemetry_hub.pipeline import TelemetryPipeline

    # One-liner for cron jobs
    TelemetryPipeline('config.yaml').run()

    # Or with access to resulting data
    pipeline = TelemetryPipeline('config.yaml')
    pipeline.run()
    print(f"Total records: {len(pipeline.dataframe)}")

Design Decisions:
-----------------
- Batching: Data is fetched in configurable time increments to provide
  incremental progress and reduce memory pressure on large date ranges.

- Incremental saves: Each batch is saved immediately after fetch, ensuring
  partial progress is preserved if the pipeline is interrupted.

- Provider independence: Motive and Samsara are fetched independently.
  If one fails, the other's data is still saved. The pipeline only aborts
  if ALL providers fail for a given batch.

- Deduplication: Records are deduplicated on (vin, timestamp) when appending
  to existing data. The 'last' record is kept, meaning fresh data overwrites
  stale data from previous runs with lookback overlap.
"""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from fleet_telemetry_hub.config import (
    TelemetryConfig,
    load_config,
)
from fleet_telemetry_hub.provider import Provider, ProviderManager
from fleet_telemetry_hub.schema import (
    DEDUP_COLUMNS,
    SORT_COLUMNS,
    enforce_telemetry_schema,
)
from fleet_telemetry_hub.utils import (
    ParquetFileHandler,
    fetch_motive_data,
    fetch_samsara_data,
    setup_logger,
)

__all__: list[str] = ['TelemetryPipeline']

logger: logging.Logger = logging.getLogger(__name__)


class TelemetryPipeline:
    """
    Orchestrates telemetry data extraction from ELD providers to Parquet storage.

    This class manages the complete data pipeline lifecycle: configuration loading,
    logging setup, provider initialization, date range calculation, batched fetching,
    schema enforcement, deduplication, and incremental persistence.

    The pipeline is designed for scheduled execution (e.g., cron) to keep the
    Parquet file continuously updated with fresh telemetry data.

    Attributes:
        config: The loaded TelemetryConfig instance (read-only).
        dataframe: The current state of telemetry data after run() completes,
            or None if run() has not been called.

    Example:
        # Simple scheduled job usage
        pipeline = TelemetryPipeline('/path/to/config.yaml')
        pipeline.run()

        # Access data after run
        if pipeline.dataframe is not None:
            print(f"Vehicles: {pipeline.dataframe['vin'].nunique()}")
            print(f"Records: {len(pipeline.dataframe)}")
    """

    def __init__(self, config_path: Path | str) -> None:
        """
        Initialize the telemetry pipeline from a configuration file.

        Performs all setup required before run(): loads configuration, configures
        logging, initializes the file handler, and creates the provider manager.

        Args:
            config_path: Path to the YAML configuration file. Can be a string
                or Path object. Relative paths are resolved from the current
                working directory.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValidationError: If the config file fails Pydantic validation.
            OSError: If the Parquet output directory cannot be created.
        """
        # Normalize to Path for consistent handling
        config_path = Path(config_path)

        # Load and validate configuration
        self._config: TelemetryConfig = load_config(config_path)

        # Configure logging first so all subsequent operations are logged
        setup_logger(config=self._config.logging)

        logger.info(
            'Initializing TelemetryPipeline from config: %s',
            config_path,
        )

        # Initialize storage handler (creates parent directories)
        self._file_handler: ParquetFileHandler = ParquetFileHandler(
            self._config.storage
        )

        # Initialize provider manager with all configured providers
        self._provider_manager: ProviderManager = ProviderManager.from_config(
            self._config
        )

        # Will hold the final DataFrame after run() completes
        self._dataframe: pd.DataFrame | None = None

        logger.info(
            'Pipeline initialized: providers=%s, storage=%s',
            list(self._provider_manager.list_providers()),
            self._file_handler.path,
        )

    @property
    def config(self) -> TelemetryConfig:
        """The loaded configuration (read-only)."""
        return self._config

    @property
    def dataframe(self) -> pd.DataFrame | None:
        """
        The current telemetry DataFrame, or None if run() has not been called.

        This property returns the in-memory state after the most recent run().
        For the persisted state, use the file handler directly or call run()
        again to refresh.
        """
        return self._dataframe

    def run(self) -> None:
        """
        Execute the telemetry data pipeline.

        This is the main entry point for pipeline execution. It performs:
        1. Determine start datetime (from existing data or config default)
        2. Generate time-based batches from start to now
        3. For each batch, fetch from all enabled providers
        4. Deduplicate and save incrementally

        The pipeline saves after each batch, so partial progress is preserved
        if execution is interrupted.

        Raises:
            RuntimeError: If all providers fail for any batch. Partial data
                from successful batches is preserved in the Parquet file.

        Side Effects:
            - Writes to the configured Parquet file
            - Updates self._dataframe with the final state
            - Logs progress at INFO level, details at DEBUG level
        """
        start_datetime: datetime = self._determine_start_datetime()
        end_datetime: datetime = datetime.now(UTC)

        logger.info(
            'Starting pipeline run: %s to %s',
            start_datetime.isoformat(),
            end_datetime.isoformat(),
        )

        # Generate batches based on configured increment
        batches: list[tuple[datetime, datetime]] = self._generate_batches(
            start_datetime,
            end_datetime,
        )

        logger.info(
            'Processing %d batches (increment=%.2f days)',
            len(batches),
            self._config.pipeline.batch_increment_days,
        )

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
                error_message: str = (
                    f'All providers failed for batch {batch_index}/{len(batches)} '
                    f'({batch_start.isoformat()} to {batch_end.isoformat()}). '
                    f'Aborting pipeline. Partial data from previous batches is preserved.'
                )
                logger.error(error_message)
                raise RuntimeError(error_message)

            # Empty batch (no records, but no errors) is fine - just skip saving
            if not batch_records:
                logger.info(
                    'Batch %d/%d returned no records, skipping save',
                    batch_index,
                    len(batches),
                )
                continue

            # Save incrementally
            self._save_batch(batch_records)

            logger.info(
                'Batch %d/%d complete: %d records saved',
                batch_index,
                len(batches),
                len(batch_records),
            )

        # Load final state into memory
        self._dataframe = self._file_handler.load()

        final_record_count: int = (
            len(self._dataframe) if self._dataframe is not None else 0
        )
        logger.info('Pipeline run complete: %d total records', final_record_count)

    def _determine_start_datetime(self) -> datetime:
        """
        Determine the start datetime for this pipeline run.

        If existing data exists, start from (max_timestamp - lookback_days) to
        catch any late-arriving records. Otherwise, use the configured
        default_start_date for initial backfill.

        Returns:
            Timezone-aware UTC datetime for the start of data fetching.
        """
        existing_data: pd.DataFrame | None = self._file_handler.load()

        if existing_data is None or existing_data.empty:
            # Initial run: use config default (midnight UTC)
            start_datetime: datetime = datetime.fromisoformat(
                self._config.pipeline.default_start_date
            ).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)

            logger.info(
                'No existing data found. Starting from config default: %s',
                start_datetime.isoformat(),
            )
            return start_datetime

        # Incremental run: max timestamp minus lookback
        max_timestamp: datetime = existing_data['timestamp'].max().to_pydatetime()
        lookback: timedelta = timedelta(days=self._config.pipeline.lookback_days)
        start_datetime = max_timestamp - lookback

        logger.info(
            'Existing data found (max_timestamp=%s). Starting from %s (lookback=%d days)',
            max_timestamp.isoformat(),
            start_datetime.isoformat(),
            self._config.pipeline.lookback_days,
        )

        return start_datetime

    def _generate_batches(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """
        Generate time-based batches for incremental processing.

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
                'Start datetime %s is not before end datetime %s. No batches generated.',
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
        Fetch telemetry data from all enabled providers for a single batch.

        Each provider is fetched independently. If one provider fails, the
        other's data is still collected. Only returns None if ALL providers
        fail, signaling that the pipeline should abort.

        Args:
            batch_start: Start of the batch time range (inclusive).
            batch_end: End of the batch time range (inclusive).

        Returns:
            List of record dictionaries if at least one provider succeeded,
            None if all providers failed (signals abort condition).
        """
        all_records: list[dict[str, Any]] = []
        any_provider_succeeded: bool = False

        # Fetch from Motive
        motive_provider: Provider = self._provider_manager.get('motive')
        try:
            motive_records: list[dict[str, Any]] = fetch_motive_data(
                motive_provider,
                batch_start,
                batch_end,
            )
            all_records.extend(motive_records)
            any_provider_succeeded = True
            logger.debug('Motive returned %d records', len(motive_records))
        except Exception:
            logger.exception('Motive fetch failed for batch')

        # Fetch from Samsara
        samsara_provider: Provider = self._provider_manager.get('samsara')
        try:
            samsara_records: list[dict[str, Any]] = fetch_samsara_data(
                samsara_provider,
                batch_start,
                batch_end,
            )
            all_records.extend(samsara_records)
            any_provider_succeeded = True
            logger.debug('Samsara returned %d records', len(samsara_records))
        except Exception:
            logger.exception('Samsara fetch failed for batch')

        # Return None only if ALL providers failed (abort signal)
        if not any_provider_succeeded:
            return None

        return all_records

    def _save_batch(self, records: list[dict[str, Any]]) -> None:
        """
        Save a batch of records to the Parquet file.

        Converts records to DataFrame, enforces schema types, appends to
        existing data (if any), deduplicates on (vin, timestamp), and saves.

        Args:
            records: List of record dictionaries from fetch functions.
                Must contain all columns defined in TELEMETRY_COLUMNS.

        Side Effects:
            - Reads existing Parquet file (if present)
            - Writes combined/deduplicated data to Parquet file
            - Updates self._dataframe with current state
        """
        # Convert to DataFrame and enforce schema
        new_dataframe: pd.DataFrame = pd.DataFrame(records)
        new_dataframe = enforce_telemetry_schema(new_dataframe)

        # Load existing data for append
        existing_dataframe: pd.DataFrame | None = self._file_handler.load()

        if existing_dataframe is not None and not existing_dataframe.empty:
            # Combine and deduplicate
            combined_dataframe: pd.DataFrame = pd.concat(
                [existing_dataframe, new_dataframe],
                ignore_index=True,
            )

            # Deduplicate: keep='last' means fresh data overwrites stale
            combined_dataframe = combined_dataframe.drop_duplicates(
                subset=DEDUP_COLUMNS,
                keep='last',
            )

            logger.debug(
                'Combined %d existing + %d new = %d after dedup',
                len(existing_dataframe),
                len(new_dataframe),
                len(combined_dataframe),
            )
        else:
            combined_dataframe = new_dataframe

        # Sort for efficient downstream queries and consistent output
        combined_dataframe = combined_dataframe.sort_values(SORT_COLUMNS).reset_index(
            drop=True
        )

        # Persist to storage
        self._file_handler.save(combined_dataframe)

        # Update in-memory state
        self._dataframe = combined_dataframe
