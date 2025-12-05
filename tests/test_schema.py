"""

Tests for fleet_telemetry_hub.schema module.



Tests schema enforcement, type coercion, deduplication logic, and constants.

"""

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from fleet_telemetry_hub.schema import (
    DEDUP_COLUMNS,
    SORT_COLUMNS,
    TELEMETRY_COLUMNS,
    enforce_telemetry_schema,
)

NUM_OF_TELEMETRY_COLUMNS: int = 14


class TestSchemaConstants:
    """Test schema constant definitions."""

    def test_telemetry_columns_count(self) -> None:
        """TELEMETRY_COLUMNS should have exactly 14 columns."""

        assert len(TELEMETRY_COLUMNS) == NUM_OF_TELEMETRY_COLUMNS

    def test_telemetry_columns_includes_required_fields(self) -> None:
        """TELEMETRY_COLUMNS should include all required fields."""

        required_fields: list[str] = [
            'provider',
            'provider_vehicle_id',
            'vin',
            'fleet_number',
            'timestamp',
            'latitude',
            'longitude',
        ]

        for field in required_fields:
            assert field in TELEMETRY_COLUMNS, f'Missing required field: {field}'

    def test_dedup_columns(self) -> None:
        """DEDUP_COLUMNS should be (vin, timestamp)."""

        assert DEDUP_COLUMNS == ['vin', 'timestamp']

        assert all(col in TELEMETRY_COLUMNS for col in DEDUP_COLUMNS)

    def test_sort_columns(self) -> None:
        """SORT_COLUMNS should be (vin, timestamp)."""

        assert SORT_COLUMNS == ['vin', 'timestamp']

        assert all(col in TELEMETRY_COLUMNS for col in SORT_COLUMNS)


class TestEnforceTelemetrySchema:
    """Test enforce_telemetry_schema function."""

    def test_enforce_schema_with_valid_data(
        self,
        sample_telemetry_records: list[dict[str, Any]],
        assert_dataframe_valid_telemetry: Any,
    ) -> None:
        """Should enforce schema on valid data successfully."""

        df = pd.DataFrame(sample_telemetry_records)

        result: pd.DataFrame = enforce_telemetry_schema(df)

        assert_dataframe_valid_telemetry(result)

        assert len(result) == len(sample_telemetry_records)

    def test_enforce_schema_coerces_timestamp_to_utc(self) -> None:
        """Should convert timestamp strings to timezone-aware UTC datetime."""

        df = pd.DataFrame(
            [
                {
                    'provider': 'motive',
                    'provider_vehicle_id': 'vehicle_001',
                    'vin': 'ABC123XYZ45678901',
                    'fleet_number': 'TRUCK-001',
                    'timestamp': '2025-12-05T10:00:00Z',  # String timestamp
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
            ]
        )

        result: pd.DataFrame = enforce_telemetry_schema(df)

        # Should be timezone-aware UTC

        assert result['timestamp'].dtype == 'datetime64[ns, UTC]'

        assert result['timestamp'].iloc[0].tzinfo is not None

        assert result['timestamp'].iloc[0].tzinfo.tzname(None) == 'UTC'

    def test_enforce_schema_coerces_numeric_columns(self) -> None:
        """Should coerce numeric columns to float64."""

        df = pd.DataFrame(
            [
                {
                    'provider': 'motive',
                    'provider_vehicle_id': 'vehicle_001',
                    'vin': 'ABC123XYZ45678901',
                    'fleet_number': 'TRUCK-001',
                    'timestamp': datetime(2025, 12, 5, 10, 0, 0, tzinfo=UTC),
                    'latitude': '37.7749',  # String that should be converted
                    'longitude': '-122.4194',  # String
                    'speed_mph': '55',  # Integer as string
                    'heading_degrees': '270',  # Integer as string
                    'engine_state': 'On',
                    'driver_id': 'driver_001',
                    'driver_name': 'John Doe',
                    'location_description': 'San Francisco, CA',
                    'odometer': '125000',  # Integer as string
                }
            ]
        )

        result: pd.DataFrame = enforce_telemetry_schema(df)

        # All numeric columns should be float64

        numeric_cols: list[str] = [
            'latitude',
            'longitude',
            'speed_mph',
            'heading_degrees',
            'odometer',
        ]

        for col in numeric_cols:
            assert result[col].dtype == 'float64', f'{col} should be float64'

            assert isinstance(result[col].iloc[0], float)

    def test_enforce_schema_creates_categorical_columns(self) -> None:
        """Should convert provider and engine_state to categorical."""

        df = pd.DataFrame(
            [
                {
                    'provider': 'motive',  # Should become categorical
                    'provider_vehicle_id': 'vehicle_001',
                    'vin': 'ABC123XYZ45678901',
                    'fleet_number': 'TRUCK-001',
                    'timestamp': datetime(2025, 12, 5, 10, 0, 0, tzinfo=UTC),
                    'latitude': 37.7749,
                    'longitude': -122.4194,
                    'speed_mph': 55.5,
                    'heading_degrees': 270.0,
                    'engine_state': 'On',  # Should become categorical
                    'driver_id': 'driver_001',
                    'driver_name': 'John Doe',
                    'location_description': 'San Francisco, CA',
                    'odometer': 125000.5,
                }
            ]
        )

        result: pd.DataFrame = enforce_telemetry_schema(df)

        assert isinstance(result['provider'].dtype, pd.CategoricalDtype)

        assert isinstance(result['engine_state'].dtype, pd.CategoricalDtype)

    def test_enforce_schema_handles_invalid_numeric_values(self) -> None:
        """Should convert invalid numeric values to NaN."""

        df = pd.DataFrame(
            [
                {
                    'provider': 'motive',
                    'provider_vehicle_id': 'vehicle_001',
                    'vin': 'ABC123XYZ45678901',
                    'fleet_number': 'TRUCK-001',
                    'timestamp': datetime(2025, 12, 5, 10, 0, 0, tzinfo=UTC),
                    'latitude': 'invalid',  # Invalid number
                    'longitude': -122.4194,
                    'speed_mph': 'N/A',  # Invalid number
                    'heading_degrees': 270.0,
                    'engine_state': 'On',
                    'driver_id': 'driver_001',
                    'driver_name': 'John Doe',
                    'location_description': 'San Francisco, CA',
                    'odometer': 125000.5,
                }
            ]
        )

        result: pd.DataFrame = enforce_telemetry_schema(df)

        # Invalid values should become NaN

        assert pd.isna(result['latitude'].iloc[0])

        assert pd.isna(result['speed_mph'].iloc[0])

        # Valid values should remain

        assert result['longitude'].iloc[0] == -122.4194  # noqa: PLR2004

        assert result['heading_degrees'].iloc[0] == 270.0  # noqa: PLR2004

    def test_enforce_schema_enforces_column_order(self) -> None:
        """Should reorder columns to match TELEMETRY_COLUMNS."""

        # Create DataFrame with columns in wrong order

        df = pd.DataFrame(
            [
                {
                    'vin': 'ABC123XYZ45678901',  # Should be 3rd, not 1st
                    'timestamp': datetime(2025, 12, 5, 10, 0, 0, tzinfo=UTC),
                    'provider': 'motive',  # Should be 1st
                    'provider_vehicle_id': 'vehicle_001',
                    'fleet_number': 'TRUCK-001',
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
            ]
        )

        result: pd.DataFrame = enforce_telemetry_schema(df)

        # Column order should match TELEMETRY_COLUMNS

        assert list(result.columns) == TELEMETRY_COLUMNS

    def test_enforce_schema_raises_on_missing_columns(self) -> None:
        """Should raise ValueError if required columns are missing."""

        df = pd.DataFrame(
            [
                {
                    'provider': 'motive',
                    'vin': 'ABC123XYZ45678901',
                    'timestamp': datetime(2025, 12, 5, 10, 0, 0, tzinfo=UTC),
                    # Missing many required columns
                }
            ]
        )

        with pytest.raises(ValueError, match='missing required columns'):
            enforce_telemetry_schema(df)

    def test_enforce_schema_is_idempotent(
        self,
        sample_telemetry_dataframe: pd.DataFrame,
    ) -> None:
        """Calling enforce_telemetry_schema twice should produce same result."""

        # First enforcement

        first_result: pd.DataFrame = enforce_telemetry_schema(
            sample_telemetry_dataframe
        )

        # Second enforcement on already-enforced data

        second_result: pd.DataFrame = enforce_telemetry_schema(first_result)

        # Results should be equal

        pd.testing.assert_frame_equal(first_result, second_result)

    def test_enforce_schema_works_with_empty_dataframe(self) -> None:
        """Should handle empty DataFrame with all columns present."""

        # Create empty DataFrame with correct columns but no rows

        df = pd.DataFrame(columns=TELEMETRY_COLUMNS)

        result: pd.DataFrame = enforce_telemetry_schema(df)

        # Should have correct column order

        assert list(result.columns) == TELEMETRY_COLUMNS

        # Should be empty

        assert len(result) == 0

        # Should have correct dtypes (where possible to determine from empty data)

        assert isinstance(result['provider'].dtype, pd.CategoricalDtype)

        assert isinstance(result['engine_state'].dtype, pd.CategoricalDtype)

    def test_enforce_schema_preserves_data_values(
        self,
        sample_telemetry_record: dict[str, Any],
    ) -> None:
        """Should preserve actual data values during schema enforcement."""

        df = pd.DataFrame([sample_telemetry_record])

        result: pd.DataFrame = enforce_telemetry_schema(df)

        # Check key values are preserved

        assert result['vin'].iloc[0] == sample_telemetry_record['vin']

        assert result['fleet_number'].iloc[0] == sample_telemetry_record['fleet_number']

        assert result['latitude'].iloc[0] == sample_telemetry_record['latitude']

        assert result['longitude'].iloc[0] == sample_telemetry_record['longitude']

        assert result['speed_mph'].iloc[0] == sample_telemetry_record['speed_mph']


class TestDeduplication:
    """Test deduplication behavior using schema constants."""

    def test_deduplication_removes_exact_duplicates(
        self,
        sample_timestamp: datetime,
    ) -> None:
        """Should remove records with same VIN and timestamp."""

        records: list[dict[str, str | datetime | float]] = [
            {
                'provider': 'motive',
                'provider_vehicle_id': 'vehicle_001',
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
            },
            {
                'provider': 'samsara',  # Different provider
                'provider_vehicle_id': 'vehicle_002',  # Different vehicle ID
                'vin': 'ABC123XYZ45678901',  # Same VIN
                'fleet_number': 'TRUCK-001',
                'timestamp': sample_timestamp,  # Same timestamp
                'latitude': 37.8,  # Different location
                'longitude': -122.5,
                'speed_mph': 60.0,  # Different speed
                'heading_degrees': 180.0,
                'engine_state': 'On',
                'driver_id': 'driver_002',
                'driver_name': 'Jane Smith',
                'location_description': 'Oakland, CA',
                'odometer': 126000.0,
            },
        ]

        df = pd.DataFrame(records)

        df: pd.DataFrame = enforce_telemetry_schema(df)

        # Deduplicate using schema constants

        result: pd.DataFrame = df.drop_duplicates(subset=DEDUP_COLUMNS, keep='last')

        # Should keep only the last record (samsara)

        assert len(result) == 1

        assert result['provider'].iloc[0] == 'samsara'

    def test_deduplication_keeps_different_timestamps(
        self,
        sample_timestamp: datetime,
    ) -> None:
        """Should keep records with same VIN but different timestamps."""

        records: list[dict[str, str | datetime | float]] = [
            {
                'provider': 'motive',
                'provider_vehicle_id': 'vehicle_001',
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
            },
            {
                'provider': 'motive',
                'provider_vehicle_id': 'vehicle_001',
                'vin': 'ABC123XYZ45678901',  # Same VIN
                'fleet_number': 'TRUCK-001',
                'timestamp': sample_timestamp
                + pd.Timedelta(minutes=10),  # Different timestamp
                'latitude': 37.8,
                'longitude': -122.5,
                'speed_mph': 60.0,
                'heading_degrees': 180.0,
                'engine_state': 'On',
                'driver_id': 'driver_001',
                'driver_name': 'John Doe',
                'location_description': 'Oakland, CA',
                'odometer': 125010.0,
            },
        ]

        df = pd.DataFrame(records)

        df: pd.DataFrame = enforce_telemetry_schema(df)

        # Deduplicate

        result: pd.DataFrame = df.drop_duplicates(subset=DEDUP_COLUMNS, keep='last')

        # Should keep both records (different timestamps)

        assert len(result) == 2  # noqa: PLR2004


class TestSorting:
    """Test sorting behavior using schema constants."""

    def test_sorting_by_schema_columns(
        self,
        sample_telemetry_records: list[dict[str, Any]],
    ) -> None:
        """Should sort by VIN then timestamp."""

        df = pd.DataFrame(sample_telemetry_records)

        df: pd.DataFrame = enforce_telemetry_schema(df)

        # Shuffle the DataFrame

        shuffled: pd.DataFrame = df.sample(frac=1.0, random_state=42).reset_index(
            drop=True
        )

        # Sort using schema constants

        sorted_df: pd.DataFrame = shuffled.sort_values(SORT_COLUMNS).reset_index(
            drop=True
        )

        # Check sorting

        # VINs should be in ascending order

        assert sorted_df['vin'].is_monotonic_increasing

        # Within each VIN, timestamps should be in ascending order

        for vin in sorted_df['vin'].unique():
            vin_records: pd.DataFrame = sorted_df[sorted_df['vin'] == vin]  # pyright: ignore[reportUnknownVariableType]

            assert vin_records['timestamp'].is_monotonic_increasing
