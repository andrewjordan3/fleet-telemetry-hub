# fleet_telemetry_hub/__init__.py
"""
Fleet Telemetry Hub - Unified telemetry data pipeline and API client.

This package provides two complementary systems for working with fleet telematics data:

1. **Data Pipeline System**: Automated data collection and storage
   - PartitionedTelemetryPipeline orchestrates scheduled data fetching
   - Date-partitioned Parquet storage (Hive-style: date=YYYY-MM-DD/)
   - Native BigQuery compatibility via Hive partitioning
   - Automatic deduplication on (provider, provider_vehicle_id, timestamp)
   - Built-in data retention with delete_old_partitions()

2. **API Abstraction Layer**: Direct provider access
   - Type-safe, provider-agnostic API interface
   - Paginated fetching with automatic retry logic
   - For custom workflows and ad-hoc queries

Quick Start - Pipeline System (Scheduled Data Collection):
    >>> from fleet_telemetry_hub import PartitionedTelemetryPipeline
    >>>
    >>> # One-liner for cron jobs
    >>> PartitionedTelemetryPipeline('config/telemetry_config.yaml').run()
    >>>
    >>> # Load data for analysis
    >>> from datetime import date
    >>> pipeline = PartitionedTelemetryPipeline('config/telemetry_config.yaml')
    >>> df = pipeline.load_date_range(
    ...     start_date=date(2024, 1, 1),
    ...     end_date=date(2024, 1, 31),
    ... )

Quick Start - API Abstraction (Direct Provider Access):
    >>> from fleet_telemetry_hub import Provider
    >>> from fleet_telemetry_hub.config import load_config
    >>>
    >>> config = load_config("config.yaml")
    >>> motive = Provider.from_config("motive", config.providers["motive"])
    >>>
    >>> for vehicle in motive.fetch_all("vehicles"):
    ...     print(vehicle.number)

Features:
    - Multi-provider support (Motive, Samsara, extensible)
    - Scalable date-partitioned storage (billions of records)
    - BigQuery direct query support via Hive partitioning
    - Automatic deduplication and data quality checks
    - Configurable data retention policies
    - Rate limiting and exponential backoff retry logic
    - Type-safe provider interfaces with runtime validation

For more information, see README.md and ARCHITECTURE.md.
"""

__version__ = '0.1.0'

from fleet_telemetry_hub.client import (
    APIError,
    RateLimitError,
    TelemetryClient,
    TransientAPIError,
)
from fleet_telemetry_hub.common import (
    PartitionedParquetHandler,
    setup_logger,
)
from fleet_telemetry_hub.config import load_config
from fleet_telemetry_hub.pipeline_partitioned import (
    PartitionedPipelineError,
    PartitionedTelemetryPipeline,
)
from fleet_telemetry_hub.provider import (
    Provider,
    ProviderConfigurationError,
    ProviderManager,
)
from fleet_telemetry_hub.registry import (
    EndpointNotFoundError,
    EndpointRegistry,
    ProviderNotFoundError,
)

__all__: list[str] = [
    'APIError',
    'EndpointNotFoundError',
    'EndpointRegistry',
    'PartitionedParquetHandler',
    'PartitionedPipelineError',
    'PartitionedTelemetryPipeline',
    'Provider',
    'ProviderConfigurationError',
    'ProviderManager',
    'ProviderNotFoundError',
    'RateLimitError',
    'TelemetryClient',
    'TransientAPIError',
    '__version__',
    'load_config',
    'setup_logger',
]
