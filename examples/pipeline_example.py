#!/usr/bin/env python3
"""
Example usage of the vehicle data pipeline.

This script demonstrates how to use the pipeline to fetch and combine
vehicle telemetry data from Motive and Samsara into a unified DataFrame.
"""

import logging
from datetime import datetime, timedelta

from fleet_telemetry_hub.config.loader import load_config
from fleet_telemetry_hub.pipeline import create_vehicle_data_pipeline
from fleet_telemetry_hub.provider import Provider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the vehicle data pipeline example."""
    # Load configuration
    logger.info('Loading configuration...')
    config = load_config('config/telemetry_config.yaml')

    # Initialize providers
    logger.info('Initializing providers...')
    motive_provider = Provider.from_config('motive', config)
    samsara_provider = Provider.from_config('samsara', config)

    # Define time range (last 7 days)
    end_datetime = datetime.now()
    start_datetime = end_datetime - timedelta(days=7)

    logger.info('Time range: %s to %s', start_datetime, end_datetime)

    # Run pipeline
    logger.info('Running vehicle data pipeline...')
    df = create_vehicle_data_pipeline(
        motive_provider=motive_provider,
        samsara_provider=samsara_provider,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    # Display results
    logger.info('Pipeline complete!')
    logger.info('Total records: %d', len(df))
    logger.info('Columns: %s', list(df.columns))

    if not df.empty:
        logger.info('\nData summary:')
        logger.info('Providers: %s', df['provider'].unique())
        logger.info('Vehicles: %d unique', df['provider_vehicle_id'].nunique())
        logger.info('Date range: %s to %s', df['timestamp'].min(), df['timestamp'].max())

        logger.info('\nFirst 5 records:')
        print(df.head())

        logger.info('\nData types:')
        print(df.dtypes)

        logger.info('\nMissing data:')
        print(df.isnull().sum())

        # Example: Save to various formats
        # df.to_parquet('vehicle_data.parquet', compression='snappy')
        # df.to_csv('vehicle_data.csv', index=False)
        # df.to_json('vehicle_data.json', orient='records', date_format='iso')

        logger.info('\nPipeline example complete!')
    else:
        logger.warning('No data returned from pipeline')


if __name__ == '__main__':
    main()
