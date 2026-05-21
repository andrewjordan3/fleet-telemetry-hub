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

The pipeline layer ships **two ETL pipelines for different grains of data**, which coexist and serve different downstream consumers:

- **`PartitionedTelemetryPipeline`** — breadcrumb-grain (location point) data, date-partitioned Parquet output. Designed for scale (billions of records) and BigQuery external-table consumption. See [Quick Start Option 1](#option-1-data-pipeline-recommended-for-continuous-data-collection).
- **`UtilizationPipeline`** — event-grain driving and idle utilization data, single-file Parquet output with a companion metadata JSON. Sized for self-service Power BI semantic models with no Power Query transformations. See [Quick Start Option 2](#option-2-utilization-pipeline-event-grain-driving-and-idle-data).

Whether you need one-off API queries, scalable breadcrumb collection, or daily utilization rollups, Fleet Telemetry Hub provides the right abstraction.

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
- **Date-Partitioned Parquet Storage**: Daily partitions for BigQuery compatibility and scalable storage
- **Independent Provider Fetching**: One provider's failure doesn't block others
- **Comprehensive Logging**: Track progress and debug issues

### Utilization Pipeline Features
- **Event-Grain Unified Schema**: Driving and idle events from Motive and Samsara normalized to a single 9-column table
- **Single-File Output**: One Parquet file plus a companion metadata JSON — suitable for self-service Power BI semantic models with no Power Query transformations
- **Automatic Window Resolution**: Pipeline derives its fetch window from the previous run's metadata; no scheduling logic needed inside the codebase
- **Today-Minus-One End Cutoff**: Never fetches the current incomplete UTC day, eliminating partial-day artifacts
- **Per-Provider Failure Isolation**: A single provider's API outage produces a partial DataFrame rather than failing the run
- **Idle-Adjusted Driving Durations**: True driving time computed by subtracting overlapping idle from raw driving-period or trip duration
- **Null-Driver Gap-Fill**: Idle events without an attributed driver are filled by most-overlap matching against driving events on the same vehicle
- **Loud VIN Fallback**: Samsara vehicles whose VIN can't be resolved get the literal value `unknown_vin` with a WARNING log, preserving data while flagging the gap

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

The pipeline automatically collects, normalizes, and stores data from all configured providers in date-partitioned Parquet files.

**1. Create a configuration file** at `config/telemetry_config.yaml`:

```yaml
providers:
  motive:
    enabled: true
    base_url: "https://api.gomotive.com"
    api_key: "your-motive-api-key"
    # Identifier emitted as the `company` column value in the unified
    # utilization output. Leave null to emit null company values.
    company: null
    request_timeout: [10, 30]
    max_retries: 5
    retry_backoff_factor: 2.0
    verify_ssl: true

  samsara:
    enabled: true
    base_url: "https://api.samsara.com"
    api_key: "your-samsara-api-key"
    # Identifier emitted as the `company` column value in the unified
    # utilization output. Leave null to emit null company values.
    company: null
    request_timeout: [10, 30]
    max_retries: 5
    retry_backoff_factor: 2.0
    verify_ssl: true

pipeline:
  default_start_date: "2024-01-01"
  lookback_days: 7
  batch_increment_days: 1.0

storage:
  parquet_path: "data/telemetry"
  parquet_compression: "snappy"

logging:
  file_path: "logs/fleet_telemetry.log"
  console_level: "INFO"
  file_level: "DEBUG"
```

**2. Run the pipeline**:

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

**Storage Structure:**

The pipeline creates a Hive-style directory structure compatible with BigQuery:

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
# Run daily at 2 AM
0 2 * * * cd /path/to/project && python -c "from fleet_telemetry_hub.pipeline_partitioned import PartitionedTelemetryPipeline; PartitionedTelemetryPipeline('config.yaml').run()"
```

The pipeline will:
- Fetch data from all enabled providers
- Normalize to unified schema (14 columns: VIN, timestamp, GPS, speed, driver, etc.)
- Deduplicate on (provider, provider_vehicle_id, timestamp)
- Save incrementally to date-partitioned Parquet files with atomic writes
- Resume from last run with configurable lookback
- Organize data by date for efficient BigQuery loading and retention management

### Option 2: Utilization Pipeline (Event-Grain Driving and Idle Data)

The utilization pipeline produces a single Parquet file plus a metadata JSON describing driving and idle events normalized across Motive and Samsara. It runs as a cron-invoked daily job and figures out its own fetch window from prior metadata.

**1. Configuration** — uses the same `config/telemetry_config.yaml` file shown in Option 1. The fields the utilization pipeline cares about:

- `providers.motive.company` and `providers.samsara.company` — emitted as the `company` column value in the output. Leave null to emit null company values.
- `pipeline.default_start_date` — used only on the first run, when no metadata file exists yet.
- `pipeline.lookback_days` — on subsequent runs, the fetch window starts at `latest_data_date - lookback_days` to catch late-arriving data.
- `storage.parquet_path` — base directory. Utilization output lands in `{parquet_path}/utilization/`.
- `storage.parquet_compression` — shared with the legacy pipeline.

The remaining `pipeline` fields (`batch_increment_days`, `request_delay_seconds`, `use_truststore`) are consumed by the legacy `PartitionedTelemetryPipeline` and ignored by the utilization pipeline.

**2. Run the pipeline**:

```python
from fleet_telemetry_hub.utilization_pipeline import UtilizationPipeline

# One-liner for scheduled jobs (cron, etc.)
UtilizationPipeline('config/telemetry_config.yaml').run()
```

The pipeline determines its own fetch window from the metadata file — no command-line arguments needed.

**3. Schedule it** (cron example):

```bash
# Run daily at 2 AM (host time)
0 2 * * * cd /path/to/project && python -c "from fleet_telemetry_hub.utilization_pipeline import UtilizationPipeline; UtilizationPipeline('config/telemetry_config.yaml').run()"
```

### Option 3: Direct API Access (For Custom Integrations)

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

## Utilization Pipeline: Output Layout

The utilization pipeline writes two files under the `utilization/` subdirectory of `storage.parquet_path`:

```
{parquet_path}/utilization/
├── data.parquet
└── metadata.json
```

Both files are atomically written (temp file + rename), so a crash mid-write leaves the previous version intact. There are no date partitions — the entire window's events live in the single `data.parquet`.

## Unified Utilization Schema

`data.parquet` is a 9-column event-grain table. Both pipelines emit pandas extension types so the schema survives a Parquet round-trip without silent coercion.

| Column | Pandas Dtype | Nullable | Description |
|---|---|---|---|
| `company` | StringDtype | yes | Per-provider company identifier from config; null when unconfigured |
| `event_type` | StringDtype | no | `'driving'` or `'idle'` |
| `driver_id` | StringDtype | yes | Provider's internal driver identifier; null when no driver was logged in |
| `driver_name` | StringDtype | yes | Driver's full name; null when unattributed or unresolvable |
| `vin` | StringDtype | no | Vehicle Identification Number; the literal value `unknown_vin` when the Samsara vehicle ID could not be resolved |
| `start_time_utc` | datetime64[ns, UTC] | no | Event start time, always UTC |
| `end_time_utc` | datetime64[ns, UTC] | no | Event end time, always UTC |
| `duration_seconds` | Int64Dtype | no | For driving rows: trip/period total minus overlapping idle. For idle rows: full duration as reported by the provider |
| `distance_miles` | Float64Dtype | yes | Driving rows only; rounded to one decimal place. Null for idle rows |

Semantic notes:

- **Grain:** one row per event; no aggregation, no date column. Consumers derive any date or business-day grouping in their own tools.
- **Row order:** sorted by `(company, start_time_utc, event_type)` ascending. Null-company rows sort before any non-null-company string.
- **PC and YM driving types** (Motive HoS classifications for personal conveyance and yard moves) are emitted as `event_type='driving'` rather than filtered or differentiated. Real-world misuse of these statuses is handled in downstream analysis, not at the pipeline boundary.
- **Multi-driver overlap on idle events** triggers a WARNING log with the bucket distribution; the row's driver fields are filled with the most-overlap winner.

## Utilization Pipeline: Metadata

Each run writes a `metadata.json` alongside the Parquet. The next run reads it to compute its own fetch window.

```json
{
  "last_run_started_utc": "2026-05-21T02:00:00Z",
  "last_run_completed_utc": "2026-05-21T02:03:42Z",
  "fetch_window_start_utc": "2026-05-14T00:00:00Z",
  "fetch_window_end_utc": "2026-05-21T00:00:00Z",
  "latest_event_end_utc": "2026-05-20T22:47:13Z",
  "latest_data_date": "2026-05-20",
  "row_count": 4837,
  "by_company": {"crystal_clean": 3201, "patriot": 1636},
  "providers_present": ["motive", "samsara"],
  "providers_skipped": [],
  "providers_failed": [],
  "schema_version": 1
}
```

Field notes:

- `latest_data_date` is the operational anchor for the next run's window start (subtracts `lookback_days`). On an empty run, the prior value is preserved so a failed run doesn't reset the anchor.
- `providers_present` / `providers_skipped` / `providers_failed` reflect each provider's outcome for the run. Motive always precedes Samsara in every list.
- `by_company` maps each `company` value to its row count. Null company values appear under the key `"(null)"`.
- All timestamps are ISO-8601 UTC with the `Z` suffix.
- `schema_version` is a literal integer for future schema evolution; bumped on a breaking metadata-shape change.

## Utilization Pipeline: Backfill

To bootstrap or re-run the utilization pipeline over a historical range:

1. Set `pipeline.default_start_date` in the config to the desired backfill start date (e.g., `"2024-01-01"` for a two-year backfill).
2. Delete the existing utilization output directory if any:

   ```bash
   rm -rf {parquet_path}/utilization/
   ```

3. Run the pipeline once. It will fetch from `default_start_date` through `today - 1` in a single invocation, write a single Parquet, and create the metadata file.
4. Subsequent runs use the metadata's `latest_data_date` minus `lookback_days` as the window start, so steady-state daily operation resumes automatically.

A multi-year backfill can take 10-30 minutes depending on fleet size due to the per-vehicle Samsara trips loop, but completes in a single run.

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

### Pipeline: BigQuery Integration

The pipeline creates Hive-style date partitions that are natively compatible with BigQuery:

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
from fleet_telemetry_hub.pipeline_partitioned import PartitionedTelemetryPipeline
from fleet_telemetry_hub.config.loader import load_config

config = load_config('config.yaml')
config.pipeline.default_start_date = '2023-01-01'
config.pipeline.batch_increment_days = 7.0  # Larger batches for backfill

pipeline = PartitionedTelemetryPipeline.from_config(config)
pipeline.run()
```

### Pipeline: Production Deployment with Error Handling

```python
from fleet_telemetry_hub.pipeline_partitioned import PartitionedTelemetryPipeline, PartitionedPipelineError
import logging

logger = logging.getLogger(__name__)

try:
    pipeline = PartitionedTelemetryPipeline('config.yaml')
    pipeline.run()
    stats = pipeline.file_handler.get_statistics()
    logger.info(f"Pipeline success: {stats['partition_count']} partitions, {stats['total_size_mb']} MB")
except PartitionedPipelineError as e:
    logger.error(f"Pipeline failed: {e}")
    if e.partial_data_saved:
        logger.warning(f"Partial data saved up to batch {e.batch_index}")
    # Send alert, retry with different config, etc.
```

### Pipeline: Working with Unified Data

The pipeline produces a DataFrame with standardized columns across all providers:

```python
from fleet_telemetry_hub.pipeline_partitioned import PartitionedTelemetryPipeline
from datetime import date
import pandas as pd

pipeline = PartitionedTelemetryPipeline('config.yaml')
pipeline.run()

# Load specific date range
df = pipeline.load_date_range(
    start_date=date(2025, 1, 1),
    end_date=date(2025, 1, 31),
)

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
df.to_csv('january_data.csv', index=False)
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
│       ├── pipeline_partitioned.py        # Pipeline orchestrator
│       ├── schema.py                      # Unified telemetry schema
│       ├── client.py                      # HTTP client (API abstraction)
│       ├── provider.py                    # Provider facade (API abstraction)
│       ├── registry.py                    # Endpoint discovery
│       │
│       ├── common/                        # Common utilities
│       │   ├── partitioned_file_io.py     # Date-partitioned Parquet handler
│       │   ├── logger.py                  # Centralized logging setup
│       │   └── truststore_context.py      # SSL/TLS utilities
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
│       └── operations/                    # Data transformation functions
│           ├── fetch_data.py              # Provider fetch orchestration
│           ├── motive_funcs.py            # Motive flatten functions
│           └── samsara_funcs.py           # Samsara flatten functions
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
