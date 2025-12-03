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

from typing import Final

import pandas as pd

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
# A record is considered duplicate if it has the same VIN and timestamp,
# regardless of which provider reported it.
DEDUP_COLUMNS: Final[list[str]] = ['vin', 'timestamp']

# Columns used for sorting the final output.
# VIN-first enables efficient filtering by vehicle; timestamp-second
# enables temporal analysis within each vehicle.
SORT_COLUMNS: Final[list[str]] = ['vin', 'timestamp']


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

    # Timestamp: ensure timezone-aware UTC datetime
    # pd.to_datetime handles strings, timestamps, and already-converted values
    result['timestamp'] = pd.to_datetime(result['timestamp'], utc=True)

    # Numeric columns: coerce to float64, invalid values become NaN
    numeric_columns: list[str] = [
        'latitude',
        'longitude',
        'speed_mph',
        'heading_degrees',
        'odometer',
    ]
    for column_name in numeric_columns:
        result[column_name] = pd.to_numeric(result[column_name], errors='coerce')

    # Categorical columns: memory-efficient storage for low-cardinality strings
    categorical_columns: list[str] = ['provider', 'engine_state']
    for column_name in categorical_columns:
        result[column_name] = result[column_name].astype('category')

    # Ensure column order matches schema
    result = result[TELEMETRY_COLUMNS]

    return result
