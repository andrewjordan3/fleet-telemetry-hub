"""

Tests for fleet_telemetry_hub.utils.file_io module.



Tests ParquetFileHandler operations including load, save, delete,

atomic writes, and error handling.

"""

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fleet_telemetry_hub.common import ParquetFileHandler
from fleet_telemetry_hub.config import StorageConfig
from fleet_telemetry_hub.config.config_models import CompressionType
from fleet_telemetry_hub.schema import enforce_telemetry_schema


class TestParquetFileHandlerInitialization:
    """Test ParquetFileHandler initialization."""

    def test_initialization_creates_parent_directory(
        self,
        temp_dir: Path,
    ) -> None:
        """Should create parent directory on initialization."""

        # Use nested path to test parent directory creation

        nested_path: Path = temp_dir / 'data' / 'telemetry' / 'test.parquet'

        storage_config = StorageConfig(
            parquet_path=nested_path,
            parquet_compression='snappy',
        )

        ParquetFileHandler(storage_config)

        # Parent directories should be created

        assert nested_path.parent.exists()

        assert nested_path.parent.is_dir()

        # File itself should not exist yet

        assert not nested_path.exists()

    def test_initialization_properties(
        self,
        storage_config: StorageConfig,
    ) -> None:
        """Should set properties correctly."""

        handler = ParquetFileHandler(storage_config)

        assert handler.path == storage_config.parquet_path

        assert handler.compression == storage_config.parquet_compression

        assert handler.exists is False  # File doesn't exist yet


class TestParquetFileHandlerLoad:
    """Test ParquetFileHandler.load() method."""

    def test_load_returns_none_when_file_missing(
        self,
        storage_config: StorageConfig,
    ) -> None:
        """Should return None when file doesn't exist."""

        handler = ParquetFileHandler(storage_config)

        result: pd.DataFrame | None = handler.load()

        assert result is None

    def test_load_returns_dataframe_when_file_exists(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should load DataFrame from existing Parquet file."""

        handler = ParquetFileHandler(storage_config)

        # First save some data

        handler.save(sample_telemetry_dataframe)

        # Then load it

        result: pd.DataFrame | None = handler.load()

        assert result is not None

        assert isinstance(result, pd.DataFrame)

        assert len(result) == len(sample_telemetry_dataframe)

        pd.testing.assert_frame_equal(result, sample_telemetry_dataframe)

    def test_load_returns_none_on_corrupt_file(
        self,
        storage_config: StorageConfig,
    ) -> None:
        """Should return None when Parquet file is corrupt."""

        handler = ParquetFileHandler(storage_config)

        # Create a corrupt file (not valid Parquet)

        storage_config.parquet_path.write_text('This is not a valid Parquet file')

        # Should return None instead of raising

        result: pd.DataFrame | None = handler.load()

        assert result is None

    def test_load_handles_permission_errors(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should return None when file exists but cannot be read."""

        handler = ParquetFileHandler(storage_config)

        # Save data first

        handler.save(sample_telemetry_dataframe)

        # Mock pd.read_parquet to raise OSError

        with patch('pandas.read_parquet', side_effect=OSError('Permission denied')):
            result: pd.DataFrame | None = handler.load()

            assert result is None


class TestParquetFileHandlerSave:
    """Test ParquetFileHandler.save() method."""

    def test_save_creates_file(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should create Parquet file when saving."""

        handler = ParquetFileHandler(storage_config)

        assert not handler.exists

        handler.save(sample_telemetry_dataframe)

        assert handler.exists

        assert storage_config.parquet_path.exists()

    def test_save_preserves_data(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should save data that can be loaded back identically."""

        handler = ParquetFileHandler(storage_config)

        handler.save(sample_telemetry_dataframe)

        loaded: pd.DataFrame | None = handler.load()

        assert loaded is not None

        pd.testing.assert_frame_equal(loaded, sample_telemetry_dataframe)

    def test_save_overwrites_existing_file(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should overwrite existing file when saving."""

        handler = ParquetFileHandler(storage_config)

        # Save initial data

        initial_df: pd.DataFrame = sample_telemetry_dataframe.iloc[
            :2
        ]  # First 2 records

        handler.save(initial_df)

        # Save new data (overwrites)

        new_df: pd.DataFrame = sample_telemetry_dataframe.iloc[
            2:4
        ]  # Different 2 records

        handler.save(new_df)

        # Load should return new data

        loaded: pd.DataFrame | None = handler.load()

        assert loaded is not None

        assert len(loaded) == 2  # noqa: PLR2004

        pd.testing.assert_frame_equal(
            loaded.reset_index(drop=True), new_df.reset_index(drop=True)
        )

    def test_save_handles_empty_dataframe(
        self,
        storage_config: StorageConfig,
    ) -> None:
        """Should handle saving empty DataFrame with warning."""

        handler = ParquetFileHandler(storage_config)

        empty_df = pd.DataFrame()

        # Should not raise, but may log warning

        handler.save(empty_df)

        # Should be able to load it back

        loaded: pd.DataFrame | None = handler.load()

        assert loaded is not None

        assert len(loaded) == 0

    def test_save_atomic_write_on_success(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should use atomic write (temp file + rename)."""

        handler = ParquetFileHandler(storage_config)

        # Mock to verify temp file is used

        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            # Create a real temp file path for the mock

            temp_path: Path = storage_config.parquet_path.parent / 'temp.parquet.tmp'

            # Configure mock

            mock_temp_context = MagicMock()

            mock_temp_context.__enter__.return_value.name = str(temp_path)

            mock_temp.return_value = mock_temp_context

            # Mock the actual file operations

            with (
                patch('pandas.DataFrame.to_parquet') as mock_to_parquet,
                patch('pathlib.Path.replace') as mock_replace,
            ):
                handler.save(sample_telemetry_dataframe)

                # Should call to_parquet with temp path

                mock_to_parquet.assert_called_once()

                # Should rename temp to final path

                mock_replace.assert_called_once()

    def test_save_cleans_up_temp_file_on_error(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should clean up temp file if save fails."""

        handler = ParquetFileHandler(storage_config)

        # Mock to_parquet to raise error

        with (
            patch('pandas.DataFrame.to_parquet', side_effect=OSError('Disk full')),
            pytest.raises(OSError, match='Disk full'),
        ):
            handler.save(sample_telemetry_dataframe)

        # Final file should not exist

        assert not storage_config.parquet_path.exists()

    def test_save_applies_compression(
        self,
        temp_parquet_file: Path,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should use configured compression codec."""

        # Test with different compression settings
        for compression in ['snappy', 'gzip', 'brotli']:
            config = StorageConfig(
                parquet_path=temp_parquet_file.parent / f'test_{compression}.parquet',
                parquet_compression=cast(CompressionType, compression),
            )

            handler = ParquetFileHandler(config)

            handler.save(sample_telemetry_dataframe)

            # Verify file was created and can be loaded

            loaded: pd.DataFrame | None = handler.load()

            assert loaded is not None

            assert len(loaded) == len(sample_telemetry_dataframe)


class TestParquetFileHandlerDelete:
    """Test ParquetFileHandler.delete() method."""

    def test_delete_removes_existing_file(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should delete file if it exists."""

        handler = ParquetFileHandler(storage_config)

        # Create file

        handler.save(sample_telemetry_dataframe)

        assert handler.exists

        # Delete it

        result: bool = handler.delete()

        assert result is True

        assert not handler.exists

        assert not storage_config.parquet_path.exists()

    def test_delete_returns_false_when_file_missing(
        self,
        storage_config: StorageConfig,
    ) -> None:
        """Should return False when file doesn't exist."""

        handler = ParquetFileHandler(storage_config)

        assert not handler.exists

        result: bool = handler.delete()

        assert result is False

    def test_delete_raises_on_permission_error(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should raise if file cannot be deleted."""

        handler = ParquetFileHandler(storage_config)

        # Create file

        handler.save(sample_telemetry_dataframe)

        # Mock unlink to raise permission error

        with (
            patch('pathlib.Path.unlink', side_effect=PermissionError('Access denied')),
            pytest.raises(PermissionError, match='Access denied'),
        ):
            handler.delete()


class TestParquetFileHandlerGetFileSize:
    """Test ParquetFileHandler.get_file_size() method."""

    def test_get_file_size_returns_none_when_missing(
        self,
        storage_config: StorageConfig,
    ) -> None:
        """Should return None when file doesn't exist."""

        handler = ParquetFileHandler(storage_config)

        result: float | None = handler.get_file_size()

        assert result is None

    def test_get_file_size_returns_size_in_mb(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should return file size in MB by default."""

        handler = ParquetFileHandler(storage_config)

        handler.save(sample_telemetry_dataframe)

        result: float | None = handler.get_file_size()

        assert result is not None

        assert isinstance(result, float)

        assert result > 0  # File should have some size

    def test_get_file_size_supports_different_units(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should support bytes, kb, mb, gb units."""

        handler = ParquetFileHandler(storage_config)

        handler.save(sample_telemetry_dataframe)

        size_bytes: float | None = handler.get_file_size('bytes')

        size_kb: float | None = handler.get_file_size('kb')

        size_mb: float | None = handler.get_file_size('mb')

        size_gb: float | None = handler.get_file_size('gb')

        # All should be non-None

        assert all(s is not None for s in [size_bytes, size_kb, size_mb, size_gb])

        # Relationships should hold

        assert size_bytes > size_kb  # pyright: ignore[reportOperatorIssue]

        assert size_kb > size_mb  # pyright: ignore[reportOperatorIssue]

        assert size_mb > size_gb  # pyright: ignore[reportOperatorIssue]

        # Conversions should be accurate

        assert abs(size_bytes / 1024 - size_kb) < 0.01  # pyright: ignore[reportOperatorIssue, reportOptionalOperand]  # noqa: PLR2004

        assert abs(size_kb / 1024 - size_mb) < 0.01  # pyright: ignore[reportOperatorIssue, reportOptionalOperand]  # noqa: PLR2004

        assert abs(size_mb / 1024 - size_gb) < 0.01  # pyright: ignore[reportOperatorIssue, reportOptionalOperand]  # noqa: PLR2004


class TestParquetFileHandlerProperties:
    """Test ParquetFileHandler property accessors."""

    def test_exists_property_false_initially(
        self,
        storage_config: StorageConfig,
    ) -> None:
        """exists should be False before save."""

        handler = ParquetFileHandler(storage_config)

        assert handler.exists is False

    def test_exists_property_true_after_save(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """exists should be True after save."""

        handler = ParquetFileHandler(storage_config)

        handler.save(sample_telemetry_dataframe)

        assert handler.exists is True

    def test_exists_property_false_after_delete(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """exists should be False after delete."""

        handler = ParquetFileHandler(storage_config)

        handler.save(sample_telemetry_dataframe)

        handler.delete()

        assert handler.exists is False

    def test_path_property_returns_configured_path(
        self,
        storage_config: StorageConfig,
    ) -> None:
        """path property should return configured path."""

        handler = ParquetFileHandler(storage_config)

        assert handler.path == storage_config.parquet_path

    def test_compression_property_returns_configured_compression(
        self,
        storage_config: StorageConfig,
    ) -> None:
        """compression property should return configured compression."""

        handler = ParquetFileHandler(storage_config)

        assert handler.compression == storage_config.parquet_compression


class TestParquetFileHandlerRoundTrip:
    """Test complete save/load round trips with real data."""

    def test_round_trip_preserves_all_column_types(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
        assert_dataframe_valid_telemetry: Any,
    ) -> None:
        """Should preserve all column types through save/load."""

        handler = ParquetFileHandler(storage_config)

        # Save

        handler.save(sample_telemetry_dataframe)

        # Load

        loaded: pd.DataFrame | None = handler.load()

        # Verify schema is preserved

        assert loaded is not None

        assert_dataframe_valid_telemetry(loaded)

        # Verify data equality

        pd.testing.assert_frame_equal(
            loaded.reset_index(drop=True),
            sample_telemetry_dataframe.reset_index(drop=True),
        )

    def test_round_trip_with_multiple_saves(
        self,
        storage_config: StorageConfig,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Should handle multiple save/load cycles."""

        handler = ParquetFileHandler(storage_config)

        for i in range(3):
            # Modify data slightly

            df: pd.DataFrame = sample_telemetry_dataframe.copy()

            df['speed_mph'] = df['speed_mph'] + i

            # Save

            handler.save(df)

            # Load and verify

            loaded: pd.DataFrame | None = handler.load()

            assert loaded is not None

            pd.testing.assert_frame_equal(loaded, df)

    def test_round_trip_with_large_dataset(
        self,
        storage_config: StorageConfig,
        sample_telemetry_record: dict[str, Any],
    ) -> None:
        """Should handle larger datasets efficiently."""

        # Create a larger dataset (1000 records)

        records: list[dict[str, Any]] = [
            sample_telemetry_record.copy() for _ in range(1000)
        ]

        df = pd.DataFrame(records)

        df: pd.DataFrame = enforce_telemetry_schema(df)

        handler = ParquetFileHandler(storage_config)

        # Save

        handler.save(df)

        # Verify file size is reasonable

        size_mb: float | None = handler.get_file_size('mb')

        assert size_mb is not None

        assert (
            size_mb < 5.0  # noqa: PLR2004
        )  # Should be well under 5MB with compression

        # Load

        loaded: pd.DataFrame | None = handler.load()

        assert loaded is not None

        assert len(loaded) == 1000  # noqa: PLR2004

        pd.testing.assert_frame_equal(loaded, df)
