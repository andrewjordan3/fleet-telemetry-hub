# fleet_telemetry_hub/pipeline.py
"""
Vehicle data pipeline for fetching and transforming telemetry data.

This module provides a pipeline to fetch vehicle location and telemetry data
from multiple providers (Motive and Samsara), flatten nested structures, and
combine them into a unified pandas DataFrame.
"""

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from .provider import Provider
from .utils import fetch_motive_data, fetch_samsara_data

logger: logging.Logger = logging.getLogger(__name__)

# =============================================================================
# Main Pipeline Function
# =============================================================================


def create_vehicle_data_pipeline(
    motive_provider: Provider | None,
    samsara_provider: Provider | None,
    start_datetime: datetime,
    end_datetime: datetime,
) -> pd.DataFrame:
    """
    Create a unified vehicle data pipeline from multiple providers.

    Fetches vehicle location and telemetry data from Motive and Samsara,
    flattens nested structures, and combines into a single DataFrame.

    Args:
        motive_provider: Motive provider instance (or None to skip).
        samsara_provider: Samsara provider instance (or None to skip).
        start_datetime: Start of time range (inclusive).
        end_datetime: End of time range (inclusive).

    Returns:
        pandas DataFrame with columns:
            - provider: Provider name ('motive' or 'samsara')
            - provider_vehicle_id: Provider-specific vehicle ID
            - vin: Vehicle Identification Number
            - fleet_number: Fleet/unit number
            - timestamp: Timestamp of the record (UTC)
            - latitude: GPS latitude (decimal degrees)
            - longitude: GPS longitude (decimal degrees)
            - speed_mph: Speed in miles per hour
            - heading_degrees: Compass heading (0-360)
            - engine_state: Engine state ('On'/'Off'/'Idle'/None)
            - driver_id: Driver identifier
            - driver_name: Driver full name
            - location_description: Human-readable location
            - odometer: Odometer reading in miles

    Raises:
        ValueError: If both providers are None.

    Example:
        >>> from fleet_telemetry_hub.config.loader import load_config
        >>> from fleet_telemetry_hub.provider import Provider
        >>> from datetime import datetime, timezone
        >>>
        >>> config = load_config("config.yaml")
        >>> motive = Provider.from_config("motive", config)
        >>> samsara = Provider.from_config("samsara", config)
        >>>
        >>> df = create_vehicle_data_pipeline(
        ...     motive,
        ...     samsara,
        ...     start_datetime=datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        ...     end_datetime=datetime(2025, 1, 7, 23, 59, 59, tzinfo=timezone.utc),
        ... )
        >>>
        >>> print(f"Records: {len(df)}, Providers: {df['provider'].nunique()}")
    """
    if motive_provider is None and samsara_provider is None:
        raise ValueError('At least one provider must be configured')

    logger.info(
        'Starting vehicle data pipeline: %s to %s',
        start_datetime,
        end_datetime,
    )

    all_records: list[dict[str, Any]] = []

    # Fetch Motive data
    if motive_provider:
        try:
            motive_records: list[dict[str, Any]] = fetch_motive_data(
                motive_provider,
                start_datetime,
                end_datetime,
            )
            all_records.extend(motive_records)
        except Exception:
            logger.exception('Failed to fetch Motive data')
    else:
        logger.info('Skipping Motive (provider not configured)')

    # Fetch Samsara data
    if samsara_provider:
        try:
            samsara_records: list[dict[str, Any]] = fetch_samsara_data(
                samsara_provider,
                start_datetime,
                end_datetime,
            )
            all_records.extend(samsara_records)
        except Exception:
            logger.exception('Failed to fetch Samsara data')
    else:
        logger.info('Skipping Samsara (provider not configured)')

    # Create DataFrame
    if not all_records:
        logger.warning('No records found from any provider')
        # Return empty DataFrame with correct schema
        return pd.DataFrame(
            columns=[
                'provider',
                'provider_vehicle_id',
                'vin',
                'fleet_number',
                'timestamp',
                'latitude',
                'longitude',
                'speed_mph',
                'heading_degrees',
                'engine_state',
                'driver_id',
                'driver_name',
                'location_description',
                'odometer',
            ]
        )

    df: pd.DataFrame = pd.DataFrame(all_records)

    # Enforce schema types
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['latitude'] = df['latitude'].astype('float64')
    df['longitude'] = df['longitude'].astype('float64')
    df['speed_mph'] = pd.to_numeric(df['speed_mph'], errors='coerce')
    df['heading_degrees'] = pd.to_numeric(df['heading_degrees'], errors='coerce')
    df['odometer'] = pd.to_numeric(df['odometer'], errors='coerce')
    df['provider'] = df['provider'].astype('category')
    df['engine_state'] = df['engine_state'].astype('category')

    # Sort by VIN and timestamp for temporal analysis
    return_df: pd.DataFrame = df.sort_values(['vin', 'timestamp']).reset_index(
        drop=True
    )

    logger.info(
        'Pipeline complete: %d total records from %d provider(s)',
        len(return_df),
        return_df['provider'].nunique(),
    )

    return return_df
