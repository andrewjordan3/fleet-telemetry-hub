#!/usr/bin/env python3
"""
Advanced example usage of the vehicle data pipeline.

This script demonstrates various advanced use cases including:
- Custom date ranges
- Filtering and analysis
- Saving to different formats
- Handling individual providers
"""

import logging
from datetime import datetime, timedelta

import pandas as pd

from fleet_telemetry_hub.config.loader import load_config
from fleet_telemetry_hub.pipeline import create_vehicle_data_pipeline
from fleet_telemetry_hub.provider import Provider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


def example_custom_date_range() -> pd.DataFrame:
    """Example: Fetch data for a specific date range."""
    logger.info('=== Example: Custom Date Range ===')

    config = load_config('config/telemetry_config.yaml')
    motive = Provider.from_config('motive', config)
    samsara = Provider.from_config('samsara', config)

    # Fetch data for January 2025
    df = create_vehicle_data_pipeline(
        motive_provider=motive,
        samsara_provider=samsara,
        start_datetime=datetime(2025, 1, 1, 0, 0, 0),
        end_datetime=datetime(2025, 1, 31, 23, 59, 59),
    )

    logger.info('Fetched %d records for January 2025', len(df))
    return df


def example_single_provider() -> pd.DataFrame:
    """Example: Fetch data from only one provider."""
    logger.info('=== Example: Single Provider (Samsara only) ===')

    config = load_config('config/telemetry_config.yaml')
    samsara = Provider.from_config('samsara', config)

    # Only fetch from Samsara by passing None for Motive
    df = create_vehicle_data_pipeline(
        motive_provider=None,  # Skip Motive
        samsara_provider=samsara,
        start_datetime=datetime.now() - timedelta(days=1),
        end_datetime=datetime.now(),
    )

    logger.info('Fetched %d records from Samsara only', len(df))
    return df


def example_data_analysis(df: pd.DataFrame) -> None:
    """Example: Analyze the pipeline data."""
    logger.info('=== Example: Data Analysis ===')

    if df.empty:
        logger.warning('No data to analyze')
        return

    # Vehicle activity summary
    logger.info('\n--- Vehicle Activity Summary ---')
    vehicle_summary = df.groupby(['provider', 'fleet_number']).agg(
        {
            'timestamp': ['min', 'max', 'count'],
            'speed_mph': 'mean',
            'odometer': 'max',
        }
    )
    print(vehicle_summary)

    # Driver activity
    logger.info('\n--- Driver Activity ---')
    driver_summary = (
        df[df['driver_name'].notna()]
        .groupby('driver_name')
        .agg({'timestamp': 'count', 'speed_mph': 'mean'})
        .sort_values('timestamp', ascending=False)
    )
    print(driver_summary.head(10))

    # Engine state distribution (Samsara only)
    logger.info('\n--- Engine State Distribution ---')
    if 'engine_state' in df.columns:
        engine_states = df['engine_state'].value_counts()
        print(engine_states)

    # Speed analysis
    logger.info('\n--- Speed Analysis ---')
    speed_stats = df['speed_mph'].describe()
    print(speed_stats)

    # Records per provider
    logger.info('\n--- Records per Provider ---')
    provider_counts = df['provider'].value_counts()
    print(provider_counts)


def example_save_to_disk(df: pd.DataFrame) -> None:
    """Example: Save data to various formats."""
    logger.info('=== Example: Saving Data ===')

    if df.empty:
        logger.warning('No data to save')
        return

    # Save to Parquet (recommended for large datasets)
    parquet_file = 'output/vehicle_data.parquet'
    df.to_parquet(parquet_file, compression='snappy', index=False)
    logger.info('Saved to Parquet: %s', parquet_file)

    # Save to CSV
    csv_file = 'output/vehicle_data.csv'
    df.to_csv(csv_file, index=False)
    logger.info('Saved to CSV: %s', csv_file)

    # Save to JSON
    json_file = 'output/vehicle_data.json'
    df.to_json(json_file, orient='records', date_format='iso', indent=2)
    logger.info('Saved to JSON: %s', json_file)

    # Save provider-specific subsets
    for provider in df['provider'].unique():
        provider_df = df[df['provider'] == provider]
        provider_file = f'output/vehicle_data_{provider}.parquet'
        provider_df.to_parquet(provider_file, compression='snappy', index=False)
        logger.info('Saved %s data: %s (%d records)', provider, provider_file, len(provider_df))


def example_filter_by_vehicle(df: pd.DataFrame, fleet_number: str) -> pd.DataFrame:
    """Example: Filter data for a specific vehicle."""
    logger.info('=== Example: Filter by Vehicle (%s) ===', fleet_number)

    vehicle_df = df[df['fleet_number'] == fleet_number].copy()
    logger.info('Found %d records for vehicle %s', len(vehicle_df), fleet_number)

    if not vehicle_df.empty:
        # Sort by timestamp
        vehicle_df = vehicle_df.sort_values('timestamp')

        # Show summary
        logger.info('Time range: %s to %s', vehicle_df['timestamp'].min(), vehicle_df['timestamp'].max())
        logger.info('Total distance: %.2f miles', vehicle_df['odometer'].max() - vehicle_df['odometer'].min())
        logger.info('Average speed: %.2f mph', vehicle_df['speed_mph'].mean())

        # Show drivers
        drivers = vehicle_df['driver_name'].dropna().unique()
        logger.info('Drivers: %s', ', '.join(drivers))

    return vehicle_df


def example_time_series_analysis(df: pd.DataFrame) -> None:
    """Example: Time series analysis of vehicle data."""
    logger.info('=== Example: Time Series Analysis ===')

    if df.empty:
        logger.warning('No data for time series analysis')
        return

    # Ensure timestamp is datetime and set as index
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Resample to hourly averages
    hourly_avg = df.groupby('provider')['speed_mph'].resample('1H').mean()
    logger.info('\n--- Hourly Average Speed ---')
    print(hourly_avg.head(24))

    # Daily activity counts
    daily_counts = df.groupby('provider').resample('1D').size()
    logger.info('\n--- Daily Record Counts ---')
    print(daily_counts)


def main() -> None:
    """Run all advanced examples."""
    # Example 1: Custom date range
    df = example_custom_date_range()

    # Example 2: Single provider
    # df_samsara = example_single_provider()

    # Example 3: Data analysis
    if not df.empty:
        example_data_analysis(df)

    # Example 4: Save to disk
    if not df.empty:
        example_save_to_disk(df)

    # Example 5: Filter by vehicle (adjust fleet_number as needed)
    if not df.empty and len(df) > 0:
        first_vehicle = df['fleet_number'].iloc[0]
        vehicle_df = example_filter_by_vehicle(df, first_vehicle)

    # Example 6: Time series analysis
    if not df.empty:
        example_time_series_analysis(df)

    logger.info('\n=== All examples complete! ===')


if __name__ == '__main__':
    main()
