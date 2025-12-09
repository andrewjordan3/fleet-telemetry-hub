# fleet_telemetry_hub/schema.py
"""
Unified schema definitions for fleet telemetry data.

This module provides the canonical column definitions and DataFrame schema
enforcement for telemetry records. All modules that create or manipulate
telemetry DataFrames should use these definitions to ensure consistency.

Design Rationale:
-----------------
Centralizing schema here prevents column name/type mismatches between:
- fetch_motive_data() and fetch_samsara_data() output
- Pipeline DataFrame construction
- Parquet file storage
- Downstream analysis code

The schema is intentionally flat (no nested structures) to optimize for:
- Parquet columnar storage efficiency
- Direct querying without JSON parsing
- Compatibility with BI tools (Power BI, Tableau)
"""

import logging
from collections.abc import Hashable
from typing import Any, Final

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)

__all__: list[str] = [
    'DEDUP_COLUMNS',
    'SORT_COLUMNS',
    'TELEMETRY_COLUMNS',
    'enforce_telemetry_schema',
]

# =============================================================================
# Schema Constants
# =============================================================================

# Canonical column order for telemetry records.
# All fetch functions and pipeline operations must produce DataFrames
# with exactly these columns in this order.
TELEMETRY_COLUMNS: Final[list[str]] = [
    'provider',  # Source system: 'motive' or 'samsara'
    'provider_vehicle_id',  # Provider-specific vehicle identifier
    'vin',  # Vehicle Identification Number (17 chars)
    'fleet_number',  # Internal fleet/unit number
    'timestamp',  # Record timestamp (UTC, timezone-aware)
    'latitude',  # GPS latitude (decimal degrees, WGS84)
    'longitude',  # GPS longitude (decimal degrees, WGS84)
    'speed_mph',  # Speed in miles per hour
    'heading_degrees',  # Compass heading (0-360, 0=North)
    'engine_state',  # Engine state: 'On', 'Off', 'Idle', or None
    'driver_id',  # Driver identifier from provider
    'driver_name',  # Driver full name
    'location_description',  # Human-readable location (reverse geocoded)
    'odometer',  # Odometer reading in miles
]

# Columns used for deduplication.
# A record is considered duplicate if it has the same provider_vehicle_id and timestamp,
# and provider is used in case each provider uses and identical id.
# provider_vehicle_id is guaranteed to exist and is unique within each provider.
# This avoids issues with missing VINs while preserving data from multiple
# providers reporting the same physical vehicle.
DEDUP_COLUMNS: Final[list[str]] = ['provider', 'provider_vehicle_id', 'timestamp']

# Columns used for sorting the final output.
# Provider-first groups each provider's data; vehicle-second enables
# per-vehicle analysis; timestamp-third for temporal ordering.
SORT_COLUMNS: Final[list[str]] = ['provider', 'provider_vehicle_id', 'timestamp']


# =============================================================================
# Schema Functions
# =============================================================================


def enforce_telemetry_schema(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce correct column types on a telemetry DataFrame.

    Applies type coercion to ensure consistency regardless of how the data
    was originally constructed. Handles missing columns gracefully by
    raising a clear error.

    This function is idempotent: calling it multiple times on the same
    DataFrame produces the same result.

    Args:
        dataframe: Raw DataFrame with telemetry columns. May have incorrect
            types (e.g., timestamp as string, lat/lon as object).

    Returns:
        DataFrame with enforced types:
            - timestamp: datetime64[ns, UTC] (timezone-aware)
            - latitude/longitude: float64
            - speed_mph/heading_degrees/odometer: float64 (nullable via NaN)
            - provider/engine_state: category (memory efficient)
            - All others: object (string)

    Raises:
        ValueError: If required columns are missing from the input DataFrame.
    """
    # Validate all required columns are present
    missing_columns: set[str] = set(TELEMETRY_COLUMNS) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f'DataFrame missing required columns: {sorted(missing_columns)}'
        )

    # Work on a copy to avoid mutating the input
    result: pd.DataFrame = dataframe.copy()

    # Replace empty strings ('') and whitespace-only strings ('   ') with np.nan.
    result = result.replace(to_replace=r'^\s*$', value=np.nan, regex=True)

    # Timestamp: ensure timezone-aware UTC datetime
    # pd.to_datetime handles strings, timestamps, and already-converted values
    result['timestamp'] = pd.to_datetime(result['timestamp'], utc=True, errors='coerce')

    # Numeric columns: coerce to float64, invalid values become NaN
    numeric_columns: list[str] = [
        'latitude',
        'longitude',
        'speed_mph',
        'heading_degrees',
        'odometer',
    ]
    for column_name in numeric_columns:
        result[column_name] = pd.to_numeric(
            result[column_name], errors='coerce'
        ).astype(np.float64)

    # Categorical columns: memory-efficient storage for low-cardinality strings
    categorical_columns: list[str] = ['provider', 'engine_state']
    for column_name in categorical_columns:
        result[column_name] = result[column_name].astype('category')

    # String Identifiers: Force to string to prevent "101" (int) vs "101" (str) mismatches.
    # We calculate this dynamically by subtracting known non-string columns from the master list.
    non_string_columns: set[str] = (
        set(numeric_columns) | set(categorical_columns) | {'timestamp'}
    )
    # Use list comprehension to preserve the order from TELEMETRY_COLUMNS
    string_columns: list[str] = [
        col for col in TELEMETRY_COLUMNS if col not in non_string_columns
    ]

    for column_name in string_columns:
        # Check if the column is not empty before attempting conversion
        if not result[column_name].empty:
            # Vectorized approach: Create a boolean mask for non-null values
            # This avoids the lambda loop and keeps the code strictly vectorized.
            valid_mask: pd.Series[bool] = result[column_name].notna()

            # Apply conversion only to the valid (non-null) rows.
            # This ensures NaNs remain as NaNs and are not converted to the string "nan".
            result.loc[valid_mask, column_name] = result.loc[
                valid_mask, column_name
            ].astype(str)

    # Ensure column order matches schema
    result = result[TELEMETRY_COLUMNS]

    # Identify records with missing VINs
    missing_vin_mask: pd.Series = result['vin'].isna()
    missing_vin_count: int = missing_vin_mask.sum()

    if missing_vin_count > 0:
        # 1. Filter to just the bad rows
        # 2. Select only the identity columns
        # 3. Drop duplicates (so we see "Truck 101" once, not 500 times)
        unique_missing_identities: pd.DataFrame = result.loc[
            missing_vin_mask, ['provider', 'fleet_number', 'provider_vehicle_id']
        ].drop_duplicates()

        # Convert to a list of dictionaries for clean logging
        details_list: list[dict[Hashable, Any]] = unique_missing_identities.to_dict(
            orient='records'
        )

        logger.warning(
            'Found %d rows missing VINs. Unique affected vehicles: %r',
            missing_vin_count,
            details_list,
        )

    return result
