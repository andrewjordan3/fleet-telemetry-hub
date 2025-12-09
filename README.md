# Fleet Telemetry Hub

A comprehensive Python framework for fleet telemetry data. Provides both a **type-safe API client** for direct provider interaction and an **automated ETL pipeline** for building unified datasets from multiple telematics platforms (Motive, Samsara).

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Type Checked: mypy](https://img.shields.io/badge/type_checked-mypy-blue.svg)](http://mypy-lang.org/)
[![Code Style: Ruff](https://img.shields.io/badge/code_style-ruff-black.svg)](https://github.com/astral-sh/ruff)

## Overview

Fleet Telemetry Hub is a **dual-purpose system**:

1. **API Abstraction Framework** - Type-safe, provider-agnostic interface for direct API interaction
2. **Data Pipeline System** - Automated ETL for collecting, normalizing, and persisting telemetry data

Whether you need one-off API queries or continuous data collection, Fleet Telemetry Hub provides the right abstraction.

## Features

### API Client Features
- **Multi-Provider Support**: Unified interface for Motive and Samsara APIs
- **Type-Safe**: Full type hints and Pydantic models for request/response validation
- **Automatic Pagination**: Seamlessly iterate through paginated results (offset & cursor-based)
- **Smart Retry Logic**: Exponential backoff with configurable retry strategies
- **Rate Limit Handling**: Automatic throttling and retry-after support
- **Network Resilience**: Designed to work behind corporate proxies (Zscaler, etc.)
- **SSL Flexibility**: Configurable SSL verification with custom CA bundle support
- **Connection Pooling**: Efficient HTTP connection management

### Data Pipeline Features
- **Unified Schema**: Normalizes data from all providers to common format
- **Incremental Updates**: Intelligent lookback for late-arriving data
- **Batch Processing**: Configurable time-based batches for memory efficiency
- **Atomic Writes**: No data corruption even if process crashes mid-write
- **Automatic Deduplication**: Handles overlapping data from multiple runs
- **Parquet Storage**: Efficient columnar storage with compression
- **Date Partitioning**: Daily partitions for BigQuery compatibility and scalable storage
- **Independent Provider Fetching**: One provider's failure doesn't block others
- **Comprehensive Logging**: Track progress and debug issues

## Installation

### From PyPI (when published)

```bash
pip install fleet-telemetry-hub
```

### From Source

```bash
git clone https://github.com/andrewjordan3/fleet_telemetry_hub.git
cd fleet_telemetry_hub
pip install -e .
```

### With Optional Dependencies

```bash
# For development tools (ruff, mypy, pytest)
pip install -e ".[dev]"

# For TLS/truststore support (Windows + Zscaler)
pip install -e ".[tls]"

# Install everything
pip install -e ".[all]"
```

## Quick Start

### Option 1: Data Pipeline (Recommended for Continuous Data Collection)

The pipeline automatically collects, normalizes, and stores data from all configured providers.

**Choose Your Storage Strategy:**
- **Single-File Pipeline**: Best for datasets under 10M records or simple deployments
- **Partitioned Pipeline**: Recommended for large-scale datasets (10M+ records), BigQuery integration, or data retention policies

**1. Create a configuration file** at `config/telemetry_config.yaml`:

```yaml
providers:
  motive:
    enabled: true
    base_url: "https://api.gomotive.com"
    api_key: "your-motive-api-key"
    request_timeout: [10, 30]
    max_retries: 5
    retry_backoff_factor: 2.0
    verify_ssl: true

  samsara:
    enabled: true
    base_url: "https://api.samsara.com"
    api_key: "your-samsara-api-key"
    request_timeout: [10, 30]
    max_retries: 5
    retry_backoff_factor: 2.0
    verify_ssl: true

pipeline:
  default_start_date: "2024-01-01"
  lookback_days: 7
  batch_increment_days: 1.0

storage:
  parquet_path: "data/fleet_telemetry.parquet"
  parquet_compression: "snappy"

logging:
  file_path: "logs/fleet_telemetry.log"
  console_level: "INFO"
  file_level: "DEBUG"
```

**2a. Run the single-file pipeline** (for smaller datasets):

```python
from fleet_telemetry_hub.pipeline import TelemetryPipeline

# One-liner for scheduled jobs (cron, etc.)
TelemetryPipeline('config/telemetry_config.yaml').run()

# Or access the resulting data
pipeline = TelemetryPipeline('config/telemetry_config.yaml')
pipeline.run()

# Work with unified DataFrame
df = pipeline.dataframe
print(f"Collected {len(df)} records from {df['vin'].nunique()} vehicles")
print(f"Providers: {df['provider'].unique()}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Export to various formats
df.to_csv('telemetry.csv', index=False)
df.to_excel('telemetry.xlsx', index=False)
```

**2b. Run the partitioned pipeline** (for large-scale datasets):

```python
from fleet_telemetry_hub.pipeline_partitioned import PartitionedTelemetryPipeline

# One-liner for scheduled jobs (cron, etc.)
PartitionedTelemetryPipeline('config/telemetry_config.yaml').run()

# Or work with specific date ranges
pipeline = PartitionedTelemetryPipeline('config/telemetry_config.yaml')
pipeline.run()

# Load data for specific date range (for analysis)
from datetime import date
df = pipeline.load_date_range(
    start_date=date(2024, 1, 1),
    end_date=date(2024, 1, 31),
)
print(f"Loaded {len(df)} records from January 2024")

# Implement data retention (delete old partitions)
deleted = pipeline.delete_old_partitions(retention_days=90)
print(f"Deleted {deleted} partitions older than 90 days")
```

**Partitioned Storage Structure:**

The partitioned pipeline creates a Hive-style directory structure compatible with BigQuery:

```
data/telemetry/
├── date=2024-01-15/
│   └── data.parquet
├── date=2024-01-16/
│   └── data.parquet
├── date=2024-01-17/
│   └── data.parquet
└── _metadata.json
```

**3. Schedule it** (optional):

```bash
# Single-file pipeline - run daily at 2 AM
0 2 * * * cd /path/to/project && python -c "from fleet_telemetry_hub.pipeline import TelemetryPipeline; TelemetryPipeline('config.yaml').run()"

# Partitioned pipeline - run daily at 2 AM
0 2 * * * cd /path/to/project && python -c "from fleet_telemetry_hub.pipeline_partitioned import PartitionedTelemetryPipeline; PartitionedTelemetryPipeline('config.yaml').run()"
```

The pipeline will:
- Fetch data from all enabled providers
- Normalize to unified schema (14 columns: VIN, timestamp, GPS, speed, driver, etc.)
- Deduplicate on (VIN, timestamp)
- Save incrementally to Parquet with atomic writes
- Resume from last run with configurable lookback
- **Partitioned only**: Organize data by date for efficient BigQuery loading and retention management

### Option 2: Direct API Access (For Custom Integrations)

For one-off queries or custom integrations, use the Provider interface:

```python
from fleet_telemetry_hub import Provider
from fleet_telemetry_hub.config.loader import load_config

# Load config
config = load_config('config/telemetry_config.yaml')

# Create provider
motive = Provider.from_config('motive', config.providers['motive'])

# Fetch data
for vehicle in motive.fetch_all('vehicles'):
    print(f"Vehicle: {vehicle.number} - VIN: {vehicle.vin}")

# Or convert directly to DataFrame
df = motive.to_dataframe('vehicles')
print(df.head())
```

## Configuration

### Provider Configuration

Each provider requires the following settings:

- `enabled`: Whether to use this provider (boolean)
- `base_url`: API base URL (no trailing slash)
- `api_key`: Your API token/key
- `request_timeout`: Tuple of [connect_timeout, read_timeout] in seconds
- `max_retries`: Maximum retry attempts for failed requests
- `retry_backoff_factor`: Exponential backoff multiplier
- `verify_ssl`: SSL verification (true/false or path to CA bundle)
- `rate_limit_requests_per_second`: Client-side rate limiting

### SSL/TLS Configuration

For corporate environments with MITM proxies (Zscaler, etc.):

```yaml
# Disable SSL verification (not recommended for production)
verify_ssl: false

# Or provide custom CA bundle
verify_ssl: "/path/to/ca-bundle.pem"

# Or use system trust store (Windows)
pipeline:
  use_truststore: true
```

### Timeout Configuration

The `request_timeout` tuple has two values:

1. **Connect Timeout**: Time to establish TCP handshake (default: 10s)
2. **Read Timeout**: Time to wait for server response (default: 30s)

```yaml
request_timeout: [10, 30]  # [connect, read]
```

## Advanced Usage

### Pipeline: BigQuery Integration (Partitioned Storage Only)

The partitioned pipeline creates Hive-style date partitions that are natively compatible with BigQuery:

```python
from fleet_telemetry_hub.pipeline_partitioned import PartitionedTelemetryPipeline

# Run pipeline to populate partitioned storage
pipeline = PartitionedTelemetryPipeline('config.yaml')
pipeline.run()
```

**Option 1: BigQuery External Table (recommended for GCS)**

1. Upload partitioned data to Google Cloud Storage:
```bash
gsutil -m cp -r data/telemetry/* gs://your-bucket/telemetry/
```

2. Create external table in BigQuery:
```sql
CREATE EXTERNAL TABLE `project.dataset.fleet_telemetry`
WITH PARTITION COLUMNS (
  date DATE
)
OPTIONS (
  format = 'PARQUET',
  uris = ['gs://your-bucket/telemetry/date=*/*.parquet'],
  hive_partition_uri_prefix = 'gs://your-bucket/telemetry/'
);
```

**Option 2: BigQuery LOAD DATA (for one-time imports)**

```sql
LOAD DATA INTO `project.dataset.fleet_telemetry`
FROM FILES (
  format = 'PARQUET',
  uris = ['gs://your-bucket/telemetry/date=*/*.parquet']
)
WITH PARTITION COLUMNS (date DATE);
```

**Benefits of Partitioned Storage:**
- Query only specific date ranges (reduces cost and latency)
- Automatic partition pruning in BigQuery
- Efficient data retention (delete old partitions easily)
- Scales to billions of records without memory constraints

### Pipeline: Historical Backfill

Fetch historical data by overriding the default start date:

```python
from fleet_telemetry_hub.pipeline import TelemetryPipeline
from fleet_telemetry_hub.config.loader import load_config

config = load_config('config.yaml')
config.pipeline.default_start_date = '2023-01-01'
config.pipeline.batch_increment_days = 7.0  # Larger batches for backfill

pipeline = TelemetryPipeline.from_config(config)
pipeline.run()
```

### Pipeline: Production Deployment with Error Handling

```python
from fleet_telemetry_hub.pipeline import TelemetryPipeline, PipelineError
import logging

logger = logging.getLogger(__name__)

try:
    pipeline = TelemetryPipeline('config.yaml')
    pipeline.run()
    logger.info(f"Pipeline success: {len(pipeline.dataframe)} records")
except PipelineError as e:
    logger.error(f"Pipeline failed: {e}")
    if e.partial_data_saved:
        logger.warning(f"Partial data saved up to batch {e.batch_index}")
    # Send alert, retry with different config, etc.
```

### Pipeline: Working with Unified Data

The pipeline produces a DataFrame with standardized columns across all providers:

```python
from fleet_telemetry_hub.pipeline import TelemetryPipeline
import pandas as pd

pipeline = TelemetryPipeline('config.yaml')
pipeline.run()

df = pipeline.dataframe

# Columns available (14 total):
# provider, provider_vehicle_id, vin, fleet_number, timestamp,
# latitude, longitude, speed_mph, heading_degrees, engine_state,
# driver_id, driver_name, location_description, odometer

# Filter by VIN
vehicle_data = df[df['vin'] == 'ABC123XYZ']

# Analyze by provider
print(df.groupby('provider')['vin'].nunique())

# Calculate average speed by vehicle
avg_speeds = df.groupby('fleet_number')['speed_mph'].mean()

# Export subsets
df[df['timestamp'] >= '2025-01-01'].to_csv('recent_data.csv')
```

### API: Pagination

The client handles pagination automatically:

```python
from fleet_telemetry_hub import Provider

provider = Provider.from_config('motive', config.providers['motive'])

# Iterate through all items (automatic pagination)
with provider.client() as client:
    for item in client.fetch_all(provider.endpoint('vehicles')):
        process(item)
```

### API: DataFrame Export

Convert API data to pandas DataFrame for analysis:

```python
from fleet_telemetry_hub import Provider
from datetime import date

config = load_config("config.yaml")
motive = Provider.from_config("motive", config.providers["motive"])

# Get all vehicles as DataFrame
df = motive.to_dataframe("vehicles")
print(df.head())

# With parameters
locations_df = motive.to_dataframe(
    "vehicle_locations",
    vehicle_id=12345,
    start_date=date(2025, 1, 1),
)

# Save to various formats
df.to_parquet("vehicles.parquet", compression="snappy")
df.to_csv("vehicles.csv", index=False)
df.to_excel("vehicles.xlsx", index=False)
```

### API: Multi-Endpoint Batch with Connection Reuse

```python
with motive.client() as client:
    # Single SSL handshake, connection pooling
    vehicles = list(client.fetch_all(motive.endpoint('vehicles')))
    groups = list(client.fetch_all(motive.endpoint('groups')))
    users = list(client.fetch_all(motive.endpoint('users')))
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/andrewjordan3/fleet_telemetry_hub.git
cd fleet_telemetry_hub

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"
```

### Code Quality Tools

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/

# Run tests
pytest

# Run tests with coverage
pytest --cov=fleet_telemetry_hub --cov-report=html
```

### Project Structure

```
fleet-telemetry-hub/
├── src/
│   └── fleet_telemetry_hub/
│       ├── pipeline.py                    # Single-file pipeline orchestrator
│       ├── pipeline_partitioned.py        # Date-partitioned pipeline orchestrator
│       ├── schema.py                      # Unified telemetry schema
│       ├── client.py                      # HTTP client (API abstraction)
│       ├── provider.py                    # Provider facade (API abstraction)
│       ├── registry.py                    # Endpoint discovery
│       │
│       ├── common/                        # Common utilities
│       │   ├── partitioned_file_io.py     # Date-partitioned Parquet handler
│       │   └── logger.py                  # Centralized logging setup
│       │
│       ├── config/                        # Configuration models and loader
│       │   ├── config_models.py           # Pydantic config models
│       │   └── loader.py                  # YAML config loader
│       │
│       ├── models/                        # Request/response models
│       │   ├── shared_request_models.py   # RequestSpec, HTTPMethod
│       │   ├── shared_response_models.py  # EndpointDefinition, ParsedResponse
│       │   ├── motive_requests.py         # Motive endpoint definitions
│       │   ├── motive_responses.py        # Motive Pydantic models
│       │   ├── samsara_requests.py        # Samsara endpoint definitions
│       │   └── samsara_responses.py       # Samsara Pydantic models
│       │
│       └── operations/                    # Data fetcher implementations
│           ├── motive_fetcher.py          # Motive data fetcher
│           └── samsara_fetcher.py         # Samsara data fetcher
│
├── config/
│   └── telemetry_config.yaml     # Example configuration
├── examples/                      # Example scripts
├── tests/                         # Test suite
├── pyproject.toml                 # Project metadata and dependencies
├── ARCHITECTURE.md                # Detailed architecture documentation
└── README.md                      # This file
```

## Requirements

- **Python 3.12 or higher**
- Dependencies:
  - `pydantic>=2.0.0` - Data validation and type safety
  - `httpx>=0.28.0` - Modern HTTP client
  - `tenacity>=9.1.0` - Retry logic with exponential backoff
  - `pyyaml>=6.0.0` - YAML configuration parsing
  - `pandas>=2.0.0` - DataFrame manipulation
  - `pyarrow>=14.0.0` - Parquet I/O and columnar storage

## Architecture

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

Key architectural highlights:
- **Two-tier system**: Data Pipeline (high-level ETL) + API Abstraction Framework (low-level access)
- **Self-describing endpoints**: All API knowledge encapsulated in endpoint definitions
- **Unified schema**: 14-column normalized format for cross-provider analytics
- **Provider independence**: Add/remove providers without changing core logic
- **Type safety**: Pydantic models from API → DataFrame
- **Atomic writes**: Temp file + rename guarantees no data corruption

## Supported Providers

### Motive (formerly KeepTruckin)

- API Documentation: https://developer.gomotive.com/
- Features: Vehicle tracking, driver logs, fuel data, ELD compliance
- Pagination: Offset-based (page_no, per_page)
- Authentication: X-API-Key header

### Samsara

- API Documentation: https://developers.samsara.com/
- Features: Fleet management, vehicle health, driver safety, route optimization
- Pagination: Cursor-based (after parameter)
- Authentication: Bearer token

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`pytest`, `ruff check`, `mypy`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Please ensure:
- All tests pass
- Code is formatted with Ruff
- Type hints are complete and mypy passes
- Documentation is updated

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

**Andrew Jordan** - andrewjordan3@gmail.com

Project Link: https://github.com/andrewjordan3/fleet-telemetry-hub

## Acknowledgments

- Built with [Pydantic](https://docs.pydantic.dev/) for data validation
- HTTP client powered by [HTTPX](https://www.python-httpx.org/)
- Retry logic using [Tenacity](https://tenacity.readthedocs.io/)
- Type checking with [mypy](http://mypy-lang.org/)
- Code quality with [Ruff](https://github.com/astral-sh/ruff)

## Roadmap

### Completed ✅
- [x] Unified schema for multi-provider data normalization
- [x] Automated ETL pipeline with incremental updates
- [x] Parquet storage with atomic writes
- [x] Date-partitioned storage for BigQuery compatibility
- [x] Python 3.12+ with modern type syntax
- [x] Comprehensive logging system
- [x] Batch processing with configurable time windows
- [x] Automatic deduplication

### Planned 🎯
- [ ] Add support for additional providers (Geotab, Verizon Connect)
- [ ] Async client support with `asyncio`
- [ ] CLI tool for common operations
- [ ] Real-time data streaming mode
- [ ] GraphQL API support
- [ ] Webhook integration for push notifications
- [ ] Data quality metrics and validation reports

## Troubleshooting

### SSL Certificate Errors

If you encounter SSL errors behind a corporate proxy:

```yaml
# Option 1: Use system trust store (Windows)
pipeline:
  use_truststore: true

# Option 2: Provide CA bundle
providers:
  motive:
    verify_ssl: "/path/to/ca-bundle.pem"

# Option 3: Disable verification (development only)
providers:
  motive:
    verify_ssl: false
```

### Rate Limiting

If you're hitting rate limits frequently:

1. Increase `request_delay_seconds` in pipeline config
2. Reduce `rate_limit_requests_per_second` for the provider
3. Check provider documentation for current rate limits

### Timeout Issues

For slow API responses, increase the read timeout:

```yaml
providers:
  motive:
    request_timeout: [10, 60]  # Increase read timeout to 60s
```
