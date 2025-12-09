# fleet_telemetry_hub/common/file_io.py
"""
File input/output utilities for the fleet_telemetry_hub package.

This module handles the low-level details of reading and writing Parquet files,
abstracting storage format details away from the core business logic.

Design Philosophy:
------------------
- load() returns None on errors (missing/corrupt data is recoverable)
- save() raises on errors (filesystem issues require explicit handling)
- Config injected at initialization defines file path and compression
- Writes are atomic (temp file + rename) to prevent corruption on crash

Thread Safety:
--------------
This class is NOT thread-safe. Concurrent reads are generally safe, but
concurrent writes or read-during-write will cause undefined behavior.
If you need concurrent access, implement external locking or use separate
file paths per writer.

Usage:
------
    from fleet_telemetry_hub.config import StorageConfig
    from fleet_telemetry_hub.common.file_io import ParquetFileHandler

    storage_config = StorageConfig(parquet_path='data/telemetry.parquet')
    handler = ParquetFileHandler(storage_config)

    # Load existing data (returns None if missing/corrupt)
    existing_data = handler.load()

    # Save new data (raises on failure)
    handler.save(new_dataframe)
"""

import logging
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Literal, cast

import pandas as pd
from pyarrow import (
    ArrowInvalid as _ArrowInvalid,  # pyright: ignore[reportUnknownVariableType]
    ArrowIOError as _ArrowIOError,  # pyright: ignore[reportUnknownVariableType]
)

from fleet_telemetry_hub.config import CompressionType, StorageConfig

# PyArrow exception types for proper error handling.
# These are used in except clauses, so we need them at runtime.
# The type stubs are incomplete, hence we cast for cleaner code.
ArrowInvalid: type[Exception] = cast(type[Exception], _ArrowInvalid)
ArrowIOError: type[Exception] = cast(type[Exception], _ArrowIOError)


__all__: list[str] = ['ParquetFileHandler']

logger: logging.Logger = logging.getLogger(__name__)

FileSizeUnit = Literal['bytes', 'kb', 'mb', 'gb']


class ParquetFileHandler:
    """
    Handles reading and writing a single Parquet file.

    Operates on a file path and compression setting defined in the injected
    StorageConfig. Each handler manages exactly one file; instantiate multiple
    handlers if you need to work with multiple files.

    Atomic Write Guarantee:
        The save() method writes to a temporary file in the same directory,
        then performs an atomic rename. This prevents data corruption if the
        process crashes mid-write. The original file remains intact until the
        new file is completely written and flushed to disk.

    Attributes:
        path: The configured Parquet file path (read-only property).
        exists: Whether the Parquet file currently exists (read-only property).
    """

    def __init__(self, storage_config: StorageConfig) -> None:
        """
        Initialize the Parquet file handler.

        Creates the parent directory for the configured file path if it doesn't
        exist. The file itself is created on first save().

        Args:
            storage_config: Storage configuration containing file path and
                compression settings. A reference is stored, not copied.

        Raises:
            OSError: If parent directory cannot be created (permissions, etc).
            PermissionError: If parent directory exists but is not writable.
        """
        self._storage_config: StorageConfig = storage_config

        # Ensure parent directory exists at initialization time.
        # This fails fast on permission issues rather than waiting until save().
        self._storage_config.parquet_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            'Initialized ParquetFileHandler: path=%r, compression=%r',
            self._storage_config.parquet_path,
            self._storage_config.parquet_compression,
        )

    @property
    def path(self) -> Path:
        """The configured Parquet file path."""
        return self._storage_config.parquet_path

    @property
    def compression(self) -> CompressionType:
        """The configured compression codec."""
        return self._storage_config.parquet_compression

    @property
    def exists(self) -> bool:
        """Whether the Parquet file currently exists on disk."""
        return self._storage_config.parquet_path.exists()

    def load(self) -> pd.DataFrame | None:
        """
        Load the configured Parquet file into a DataFrame.

        Returns None for missing or corrupt files rather than raising, allowing
        callers to treat this as "no cached data available" without try/except
        blocks. File absence is expected on first run or after cache clear.

        Returns:
            DataFrame containing the file contents if successful, None if the
            file is missing, unreadable, or corrupt.

        Side Effects:
            Reads from filesystem. Logs at DEBUG level on success, ERROR level
            with full traceback on read failures.
        """
        file_path: Path = self._storage_config.parquet_path

        if not file_path.exists():
            # Intentionally no logging: file absence is expected (first run, cache cleared)
            return None

        try:
            logger.debug('Loading data from %r', file_path)
            dataframe: pd.DataFrame = pd.read_parquet(file_path)
            logger.debug(
                'Loaded %d records (%d columns) from %r',
                len(dataframe),
                len(dataframe.columns),
                file_path,
            )
            return dataframe

        except (OSError, ArrowInvalid, ArrowIOError) as read_error:
            # OSError: File permissions, disk errors
            # ArrowInvalid: Corrupt or malformed Parquet file
            # ArrowIOError: I/O errors during read (incomplete file, etc)
            logger.exception(
                'Failed to read Parquet file %r: %s',
                file_path,
                read_error,
            )
            return None

    def save(self, dataframe: pd.DataFrame) -> None:
        """
        Save a DataFrame to the configured Parquet file atomically.

        Writes to a temporary file in the same directory, then atomically
        renames to the target path. This ensures the target file is never
        left in a corrupt state, even if the process crashes mid-write.

        Unlike load(), this method raises on errors because save failures
        indicate serious issues (disk full, permissions) that require
        explicit handling by the caller.

        Args:
            dataframe: The DataFrame to persist. Empty DataFrames are allowed
                but trigger a warning log.

        Raises:
            OSError: File system errors (permissions, disk full, etc).
            ArrowInvalid: DataFrame contains types that cannot be serialized.
            ArrowIOError: I/O errors during write.

        Side Effects:
            Writes to filesystem. Creates a temporary file during write, then
            renames atomically. Logs at INFO level on success, ERROR level
            with traceback on failure before re-raising.
        """
        if dataframe.empty:
            logger.warning('Saving empty DataFrame to %r', self.path)

        file_path: Path = self._storage_config.parquet_path
        compression: CompressionType = self._storage_config.parquet_compression
        record_count: int = len(dataframe)

        # Write to temp file in the same directory, then rename atomically.
        # Same directory ensures we're on the same filesystem (required for atomic rename).
        # delete=False because we need the file to persist for the rename operation.
        try:
            with tempfile.NamedTemporaryFile(
                mode='wb',
                suffix='.parquet.tmp',
                dir=file_path.parent,
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)

            logger.debug(
                'Writing %d records to temp file %r (compression=%r)',
                record_count,
                temp_path,
                compression,
            )

            # Write to the temp file
            dataframe.to_parquet(
                temp_path,
                index=False,
                compression=compression,
            )

            # Atomic rename: either fully succeeds or target is unchanged
            temp_path.replace(file_path)

            logger.info(
                'Saved %d records (%d columns) to %r',
                record_count,
                len(dataframe.columns),
                file_path,
            )

        except (OSError, ArrowInvalid, ArrowIOError) as write_error:
            logger.exception(
                'Failed to save %d records to %r: %s',
                record_count,
                file_path,
                write_error,
            )
            # Clean up temp file if it exists
            if 'temp_path' in locals() and temp_path.exists():  # pyright: ignore[reportPossiblyUnboundVariable]
                with suppress(OSError):
                    temp_path.unlink()  # pyright: ignore[reportPossiblyUnboundVariable]
            raise

    def delete(self) -> bool:
        """
        Delete the Parquet file if it exists.

        Useful for cache invalidation, testing cleanup, or forcing a fresh
        data fetch on next pipeline run.

        Returns:
            True if file was deleted, False if file did not exist.

        Raises:
            OSError: If file exists but cannot be deleted (permissions, etc).
            PermissionError: If file is locked or user lacks delete permission.

        Side Effects:
            Removes file from filesystem. Logs at INFO level on deletion,
            DEBUG level if file was already absent.
        """
        file_path: Path = self._storage_config.parquet_path

        if not file_path.exists():
            logger.debug('Delete requested but file does not exist: %r', file_path)
            return False

        file_path.unlink()
        logger.info('Deleted Parquet file: %r', file_path)
        return True

    def get_file_size(
        self,
        unit: FileSizeUnit = 'mb',
    ) -> float | None:
        """
        Get the size of the Parquet file in the specified unit.

        Useful for monitoring, logging, or deciding whether to load the file
        into memory based on available resources.

        Args:
            unit: Size unit to return. One of 'bytes', 'kb', 'mb', 'gb'.
                Defaults to 'mb' for typical telemetry file sizes.

        Returns:
            File size in the requested unit (float for fractional sizes),
            or None if file does not exist.
        """
        if not self.exists:
            return None

        size_bytes: int = self._storage_config.parquet_path.stat().st_size

        divisors: dict[FileSizeUnit, float] = {
            'bytes': 1.0,
            'kb': 1024.0,
            'mb': 1024.0**2,
            'gb': 1024.0**3,
        }

        return size_bytes / divisors[unit]
