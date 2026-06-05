# fleet_telemetry_hub/__init__.py
"""
Fleet Telemetry Hub - Unified telemetry data pipeline and API client.

This package provides three complementary systems for working with fleet
telematics data, two ETL pipelines for different grains of output plus
a shared low-level API layer:

1. **Partitioned Telemetry Pipeline** (breadcrumb grain): Automated
   collection of high-volume location point data.
   - PartitionedTelemetryPipeline orchestrates scheduled data fetching
   - Date-partitioned Parquet storage (Hive-style: date=YYYY-MM-DD/)
   - Native BigQuery compatibility via Hive partitioning
   - Automatic deduplication on (provider, provider_vehicle_id, timestamp)
   - Built-in data retention with delete_old_partitions()

2. **Utilization Pipeline** (event grain): driving and idle event
   rollups normalized across Motive and Samsara.
   - UtilizationPipeline produces a single Parquet plus a companion
     metadata JSON, loadable directly into BigQuery and suitable for
     self-service Power BI semantic models
   - 9-column event-grain schema; driving and idle rows on one table
   - Automatic fetch-window resolution from prior metadata; today-minus-one
     end cutoff so no incomplete UTC day is ever fetched
   - Per-provider failure isolation: one provider's outage produces a
     partial DataFrame instead of failing the whole run
   - Idle-adjusted driving durations and null-driver gap-fill out of the box

3. **API Abstraction Layer**: Direct provider access for custom workflows.
   - Type-safe, provider-agnostic API interface
   - Paginated fetching with automatic retry logic
   - For custom workflows and ad-hoc queries

Quick Start - Partitioned Telemetry Pipeline (Scheduled Data Collection):
    >>> from fleet_telemetry_hub import PartitionedTelemetryPipeline
    >>>
    >>> # Invoke once per run from your scheduler
    >>> PartitionedTelemetryPipeline('config/telemetry_config.yaml').run()
    >>>
    >>> # Load data for analysis
    >>> from datetime import date
    >>> pipeline = PartitionedTelemetryPipeline('config/telemetry_config.yaml')
    >>> df = pipeline.load_date_range(
    ...     start_date=date(2024, 1, 1),
    ...     end_date=date(2024, 1, 31),
    ... )

Quick Start - Utilization Pipeline (Driving and Idle Events):
    >>> from fleet_telemetry_hub.utilization_pipeline import UtilizationPipeline
    >>>
    >>> # Invoked once per run by your scheduler
    >>> UtilizationPipeline('config/telemetry_config.yaml').run()

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
    - Two pipelines for different grains: scalable date-partitioned
      breadcrumbs (billions of records) and single-file event-grain
      utilization (driving + idle, BigQuery- and Power BI-ready)
    - BigQuery direct query support via Hive partitioning on the
      breadcrumb pipeline
    - Atomic Parquet and metadata writes on both pipelines
    - Configurable data retention policies on the breadcrumb pipeline
    - Today-minus-one end cutoff and automatic window resolution on
      the utilization pipeline
    - Rate limiting and exponential backoff retry logic
    - Type-safe provider interfaces with runtime validation

For more information, see README.md and ARCHITECTURE.md.
"""

__version__ = '0.1.0'

from fleet_telemetry_hub._utilization_merge import CorruptUtilizationParquetError
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
from fleet_telemetry_hub.utilization_pipeline import UtilizationPipeline

__all__: list[str] = [
    'APIError',
    'CorruptUtilizationParquetError',
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
    'UtilizationPipeline',
    '__version__',
    'load_config',
    'setup_logger',
]
