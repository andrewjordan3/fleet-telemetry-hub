# Fleet Telemetry Hub - Architecture Documentation

This document explains the complete architecture and design philosophy of the Fleet Telemetry Hub.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [API Abstraction Layers](#api-abstraction-layers)
4. [Data Pipeline Architecture](#data-pipeline-architecture)
5. [Schema and Type System](#schema-and-type-system)
6. [Data Transformation Layer](#data-transformation-layer)
7. [Storage Layer](#storage-layer)
8. [Design Principles](#design-principles)
9. [Usage Patterns](#usage-patterns)
10. [Extension Guide](#extension-guide)

---

## Overview

Fleet Telemetry Hub is a **dual-purpose system** that provides:

1. **API Abstraction Framework**: Type-safe, provider-agnostic interface for interacting with fleet telemetry APIs (Motive, Samsara)
2. **Data Pipeline System**: Automated ETL pipeline for extracting, transforming, and persisting unified telemetry data

The system is designed around two core principles:
- **Endpoints are self-describing objects** that encapsulate all API knowledge
- **Data flows through a unified schema** that normalizes provider-specific formats

### Key Design Goals

- ✅ **Provider Agnostic**: Client code works identically with any provider
- ✅ **Type Safe**: Full type hints and Pydantic validation throughout
- ✅ **Zero Duplication**: Shared logic lives in base classes
- ✅ **Pagination Transparent**: Different pagination styles (offset, cursor) handled uniformly
- ✅ **Auth Transparent**: Authentication injected by endpoint definitions
- ✅ **Progressive Disclosure**: Simple tasks are simple, complex tasks possible
- ✅ **Unified Schema**: All provider data normalized to common format
- ✅ **Incremental Processing**: Batched fetching with automatic progress preservation
- ✅ **Data Integrity**: Atomic writes, deduplication, and schema enforcement

---

## System Architecture

The Fleet Telemetry Hub consists of **two major subsystems** that can be used independently or together:

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Fleet Telemetry Hub                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Data Pipeline System (High-Level)               │  │
│  │                                                              │  │
│  │  TelemetryPipeline → Schema → Parquet Storage               │  │
│  │         ↓                                                    │  │
│  │    Batch Orchestration                                       │  │
│  │    Deduplication                                             │  │
│  │    Incremental Updates                                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                           ↓ uses ↓                                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │          API Abstraction Framework (Low-Level)               │  │
│  │                                                              │  │
│  │  Provider → Client → Endpoints → HTTP Requests               │  │
│  │      ↑                                                       │  │
│  │   Registry (Discovery)                                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                           ↓ talks to ↓                              │
└─────────────────────────────────────────────────────────────────────┘
                               │
        ┌──────────────────────┴──────────────────────┐
        ↓                                             ↓
  ┌──────────┐                                  ┌──────────┐
  │  Motive  │                                  │ Samsara  │
  │   API    │                                  │   API    │
  └──────────┘                                  └──────────┘
```

### Subsystem 1: API Abstraction Framework

**Purpose**: Provides a clean, type-safe interface for making API calls to any fleet telemetry provider.

**Use Cases**:
- One-off data queries
- Building custom integrations
- Exploring API capabilities
- Direct access to provider-specific features

**Key Components**:
- `EndpointDefinition` - Self-describing API endpoints
- `TelemetryClient` - Provider-agnostic HTTP client
- `Provider` - High-level facade combining config, endpoints, and client
- `EndpointRegistry` - Dynamic endpoint discovery

### Subsystem 2: Data Pipeline System

**Purpose**: Automated ETL system for continuously collecting, normalizing, and persisting telemetry data.

**Use Cases**:
- Scheduled data collection (cron jobs)
- Building unified datasets from multiple providers
- Historical data backfill
- Incremental updates with lookback

**Key Components**:
- `TelemetryPipeline` - Main orchestrator
- `schema.py` - Unified telemetry schema
- `fetch_data.py` - Provider-specific data extraction
- `ParquetFileHandler` - Atomic file I/O

---

## API Abstraction Layers

The system provides 4 levels of abstraction, from lowest (most control) to highest (most convenient):

### Layer 1: Endpoint Definitions (Lowest Level)

**Location**: `models/motive_requests.py`, `models/samsara_requests.py`

**Purpose**: Self-describing endpoint objects that know how to build requests and parse responses.

```python
from fleet_telemetry_hub.models.motive_requests import MotiveEndpoints

# Endpoint knows everything about itself
endpoint = MotiveEndpoints.VEHICLES
print(endpoint.endpoint_path)  # '/v1/vehicles'
print(endpoint.is_paginated)   # True
print(endpoint.description)    # 'List all vehicles...'

# Build complete request
request_spec = endpoint.build_request_spec(credentials, page_no=1)
# request_spec contains: URL, headers (with auth), query params, timeouts, etc.

# Parse response
parsed = endpoint.parse_response(response_json)
# Returns: ParsedResponse[Vehicle] with typed items and pagination state
```

**Key Classes**:
- `EndpointDefinition` - Abstract base class
- `MotiveEndpointDefinition` - Motive-specific (X-API-Key auth, page-number pagination)
- `SamsaraEndpointDefinition` - Samsara-specific (Bearer token, cursor pagination)

**When to Use**: When you need maximum control or are building low-level tooling.

---

### Layer 2: Telemetry Client (Provider-Agnostic HTTP)

**Location**: `client.py`

**Purpose**: Provider-agnostic HTTP client that executes requests built by endpoints.

```python
from fleet_telemetry_hub import TelemetryClient, MotiveEndpoints

credentials = ProviderCredentials(...)
endpoint = MotiveEndpoints.VEHICLES

with TelemetryClient(credentials) as client:
    # Fetch single page
    response = client.fetch(endpoint)

    # Or iterate all items (automatic pagination)
    for vehicle in client.fetch_all(endpoint):
        print(vehicle.number)
```

**Key Features**:
- Works with ANY EndpointDefinition (Motive, Samsara, future providers)
- Automatic retry with exponential backoff
- Rate limit handling (respects Retry-After headers)
- Connection pooling
- Context manager for resource cleanup

**When to Use**: When working directly with multiple endpoints using a single client instance.

---

### Layer 3: Endpoint Registry (Dynamic Lookup)

**Location**: `registry.py`

**Purpose**: String-based endpoint discovery and lookup across all providers.

```python
from fleet_telemetry_hub import EndpointRegistry

registry = EndpointRegistry.instance()

# Discover what's available
print(registry.list_providers())           # ['motive', 'samsara']
print(registry.list_endpoints('motive'))   # ['vehicles', 'groups', ...]

# Get endpoint by string name
endpoint = registry.get('motive', 'vehicles')

# Check existence
if registry.has('samsara', 'drivers'):
    endpoint = registry.get('samsara', 'drivers')

# Find by API path
matches = registry.find_by_path('/vehicles')  # Returns all matching endpoints

# Generate documentation
print(registry.describe('motive'))
```

**Key Features**:
- Case-insensitive lookup
- Introspection and discovery
- Documentation generation
- Useful for CLI tools, config-driven applications

**When to Use**: When you need to select endpoints dynamically at runtime.

---

### Layer 4: Provider Facade (Highest Level - **Recommended**)

**Location**: `provider.py`

**Purpose**: Complete abstraction combining config, endpoints, and client into a clean API.

```python
from fleet_telemetry_hub import Provider
from fleet_telemetry_hub.config.loader import load_config

# Load from config file
config = load_config('config.yaml')
motive = Provider.from_config('motive', config.providers['motive'])

# Simple fetch - client managed automatically
for vehicle in motive.fetch_all('vehicles'):
    print(vehicle.number)

# Fetch with parameters
for location in motive.fetch_all(
    'vehicle_locations',
    vehicle_id=12345,
    start_date=date(2025, 1, 1),
):
    print(location)

# Or use context manager for better performance
with motive.client() as client:
    vehicles = list(client.fetch_all(motive.endpoint('vehicles')))
    groups = list(client.fetch_all(motive.endpoint('groups')))
```

**Multi-Provider Management**:

```python
from fleet_telemetry_hub import ProviderManager

manager = ProviderManager.from_config(config)

# Access specific provider
motive = manager.get('motive')

# Or iterate all enabled providers
for provider_name, provider in manager.enabled_providers():
    for vehicle in provider.fetch_all('vehicles'):
        print(f'{provider_name}: {vehicle}')
```

**When to Use**: This is the **recommended approach** for most applications.

---

## Data Pipeline Architecture

The Data Pipeline System orchestrates the end-to-end process of extracting telemetry from multiple providers, transforming it to a unified schema, and persisting it to Parquet storage.

### Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TelemetryPipeline                              │
│                     (pipeline.py:228)                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Initialize                                                      │
│     ├─ Load config from YAML                                        │
│     ├─ Setup logging (console + file)                               │
│     ├─ Initialize ProviderManager (enabled providers)               │
│     └─ Create ParquetFileHandler                                    │
│                                                                     │
│  2. Determine Start DateTime                                        │
│     ├─ Load existing Parquet file (if exists)                       │
│     ├─ Calculate start: max_timestamp - lookback_days               │
│     └─ Or use default_start_date for first run                      │
│                                                                     │
│  3. Generate Time Batches                                           │
│     └─ Split time range into configurable increments               │
│        (default: 1 day chunks for memory efficiency)                │
│                                                                     │
│  4. For Each Batch:                                                 │
│     │                                                               │
│     ├─ Fetch from ALL Providers (parallel, independent)             │
│     │  ├─ Motive: fetch_motive_data()                               │
│     │  │   └─ vehicles → locations → flatten                        │
│     │  └─ Samsara: fetch_samsara_data()                             │
│     │      └─ vehicle_stats + driver_assignments → flatten          │
│     │                                                               │
│     ├─ Combine Records (list[dict])                                 │
│     │  └─ If ALL providers fail → abort pipeline                    │
│     │                                                               │
│     ├─ Convert to DataFrame                                         │
│     │  └─ enforce_telemetry_schema() - type coercion               │
│     │                                                               │
│     ├─ Append to Existing Data                                      │
│     │  ├─ Load existing Parquet (if any)                            │
│     │  ├─ pd.concat([existing, new])                                │
│     │  └─ Deduplicate on (vin, timestamp), keep='last'              │
│     │                                                               │
│     ├─ Save Incrementally (atomic write)                            │
│     │  ├─ Write to temp file                                        │
│     │  └─ Atomic rename                                             │
│     │                                                               │
│     └─ Continue to next batch                                       │
│                                                                     │
│  5. Log Summary Statistics                                          │
│     └─ Total records, unique VINs, date range, file size            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

#### 1. Batching Strategy

**Why**: Fetching months/years of data in one go leads to:
- Memory exhaustion with large fleets
- No visibility into progress
- All-or-nothing failure mode

**Solution**: Time-based batches (configurable, default 1 day)
- Each batch is independently fetched, processed, and saved
- Progress preserved even if pipeline crashes mid-run
- Memory usage stays constant regardless of total time range

#### 2. Independent Provider Fetching

**Why**: One provider's API issues shouldn't block data from other providers.

**Solution**: Each provider is wrapped in try/except
- Individual provider failures are logged and skipped
- Pipeline only aborts if **ALL** providers fail for a batch
- Partial data is better than no data

**Example**: If Motive API is down but Samsara succeeds, you still get Samsara data.

#### 3. Incremental Updates with Lookback

**Why**: Telemetry data can arrive late (e.g., offline vehicles syncing later).

**Solution**: On each run, start from `max_timestamp - lookback_days`
- Default lookback: 7 days
- Overlapping records are deduplicated with `keep='last'`
- Fresh data overwrites stale data from previous runs

**Example**:
```
Run 1 (Jan 1): Fetch Jan 1-7    → Save 100 records
Run 2 (Jan 8): Fetch Jan 1-8    → 7-day lookback catches late records
               Dedupe on (vin, timestamp) → Stale data replaced
```

#### 4. Atomic Writes

**Why**: Crashes during Parquet write can corrupt files.

**Solution**: Write to temp file, then atomic rename
- Original file untouched until new file is complete
- If crash occurs, original data remains intact
- Filesystem guarantees atomicity of rename operation

#### 5. Provider-Agnostic Pipeline

The pipeline doesn't know about Motive or Samsara specifics:

```python
# Pipeline just calls registered fetch functions
PROVIDER_FETCH_FUNCTIONS: dict[str, FetchFunction] = {
    'motive': fetch_motive_data,
    'samsara': fetch_samsara_data,
}

# Add new provider: just register a fetch function
PROVIDER_FETCH_FUNCTIONS['geotab'] = fetch_geotab_data
```

Each fetch function is responsible for:
1. Using the Provider abstraction to make API calls
2. Flattening nested provider responses
3. Returning `list[dict]` matching `TELEMETRY_COLUMNS`

### Pipeline Configuration

**Location**: `config/telemetry_config.yaml`

```yaml
pipeline:
  default_start_date: "2024-01-01"     # For initial backfill
  lookback_days: 7                      # Overlap for late records
  batch_increment_days: 1.0             # Time per batch (memory/progress tradeoff)

storage:
  parquet_path: "data/telemetry.parquet"
  parquet_compression: "snappy"         # Fast compression

logging:
  file_path: "logs/pipeline.log"
  console_level: "INFO"
  file_level: "DEBUG"
```

### Usage Example

```python
from fleet_telemetry_hub.pipeline import TelemetryPipeline

# One-liner for cron jobs
TelemetryPipeline('config.yaml').run()

# Or with access to results
pipeline = TelemetryPipeline('config.yaml')
pipeline.run()
print(f"Records: {len(pipeline.dataframe)}")
print(f"VINs: {pipeline.dataframe['vin'].nunique()}")
```

---

## Schema and Type System

The unified schema is the contract between data extraction (fetch functions) and data consumption (analytics, BI tools).

### Schema Definition

**Location**: `schema.py:41`

All telemetry data must conform to `TELEMETRY_COLUMNS`:

```python
TELEMETRY_COLUMNS: Final[list[str]] = [
    'provider',              # 'motive' or 'samsara'
    'provider_vehicle_id',   # Provider-specific ID
    'vin',                   # Vehicle Identification Number (17 chars)
    'fleet_number',          # Internal fleet/unit number
    'timestamp',             # Record time (UTC, timezone-aware)
    'latitude',              # GPS (decimal degrees, WGS84)
    'longitude',             # GPS (decimal degrees, WGS84)
    'speed_mph',             # Speed in miles per hour
    'heading_degrees',       # Compass heading (0-360, 0=North)
    'engine_state',          # 'On', 'Off', 'Idle', or None
    'driver_id',             # Driver identifier
    'driver_name',           # Driver full name
    'location_description',  # Human-readable location
    'odometer',              # Odometer reading in miles
]
```

### Schema Enforcement

**Function**: `enforce_telemetry_schema(dataframe: pd.DataFrame) -> pd.DataFrame`

**Purpose**: Type coercion and validation

```python
# Applied after fetching, before saving
df = pd.DataFrame(records)
df = enforce_telemetry_schema(df)  # Coerce types, validate columns
```

**Type Enforcement**:
- `timestamp`: `datetime64[ns, UTC]` (timezone-aware)
- `latitude/longitude`: `float64`
- `speed_mph/heading_degrees/odometer`: `float64` (nullable via NaN)
- `provider/engine_state`: `category` (memory efficient)
- Others: `object` (string)

**Validation**: Raises `ValueError` if required columns are missing

### Deduplication and Sorting

```python
# Deduplication keys
DEDUP_COLUMNS = ['vin', 'timestamp']

# Sort order (VIN-first for efficient filtering, timestamp for temporal analysis)
SORT_COLUMNS = ['vin', 'timestamp']
```

### Design Rationale

**Why Flat Schema?**
- ✅ Optimized for Parquet columnar storage
- ✅ No JSON parsing needed for queries
- ✅ Compatible with BI tools (Power BI, Tableau)
- ✅ Simple filtering: `df[df['vin'] == 'ABC123']`

**Why Standardized Names?**
- Motive calls it "number", Samsara calls it "name" → Both map to `fleet_number`
- Consistent schema = provider-agnostic analytics
- Add/remove providers without changing downstream code

---

## Data Transformation Layer

The transformation layer bridges provider-specific API responses and the unified schema.

### Architecture

```
Provider API Response (Nested JSON)
         ↓
   Pydantic Models (Type Validation)
         ↓
   Fetch Functions (Provider-Specific)
         ↓
   Flatten Functions (Provider-Specific)
         ↓
   list[dict[str, Any]] (Unified Schema)
         ↓
   enforce_telemetry_schema()
         ↓
   pd.DataFrame (Typed, Validated)
```

### Fetch Functions

**Location**: `utils/fetch_data.py`

Each provider has a fetch function with signature:

```python
def fetch_provider_data(
    provider: Provider,
    start_datetime: datetime,
    end_datetime: datetime,
) -> list[dict[str, Any]]:
    """
    Fetch and transform provider data to unified schema.

    Returns list of dicts, each matching TELEMETRY_COLUMNS.
    """
```

#### Motive Fetch Flow

```python
def fetch_motive_data(provider, start_datetime, end_datetime):
    with provider.client() as client:
        # 1. Fetch vehicle list
        vehicles = client.fetch_all(provider.endpoint('vehicles'))

        # 2. For each vehicle, fetch locations
        for vehicle in vehicles:
            locations = client.fetch_all(
                provider.endpoint('vehicle_locations'),
                vehicle_id=vehicle.vehicle_id,
                start_date=start_datetime.date(),
                end_date=end_datetime.date(),
            )

            # 3. Flatten each location to unified schema
            for location in locations:
                record = flatten_motive_location(location, vehicle)
                records.append(record)

    return records
```

**Motive Flatten**: `utils/motive_funcs.py:flatten_motive_location()`
- Combines `VehicleLocation` + `Vehicle` → single dict
- Maps Motive-specific fields to schema columns
- Handles missing fields (nullable columns)

#### Samsara Fetch Flow

```python
def fetch_samsara_data(provider, start_datetime, end_datetime):
    # 1. Fetch vehicle stats (GPS, engine states, odometer)
    vehicle_stats_list = provider.fetch_all(
        'vehicle_stats_history',
        start_time=start_datetime,
        end_time=end_datetime,
        types='engineStates,gps,obdOdometerMeters',
    )

    # 2. Fetch driver assignments (chunked to avoid URL length limits)
    assignments = _fetch_driver_assignments_chunked(
        provider, vehicle_ids, start_datetime, end_datetime
    )

    # 3. For each vehicle, flatten GPS records
    for vehicle_stats in vehicle_stats_list:
        for gps_record in vehicle_stats.gps:
            record = flatten_samsara_gps(
                gps_record, vehicle_stats, engine_states, odometer, assignments
            )
            records.append(record)

    return records
```

**Samsara Flatten**: `utils/samsara_funcs.py:flatten_samsara_gps()`
- Temporal join of GPS + engine states + odometer + driver assignments
- All joined on nearest timestamp
- Maps Samsara-specific fields to schema columns

### Key Challenges Solved

#### 1. Temporal Data Joining (Samsara)

Samsara returns separate time series for GPS, engine states, and odometer:

```python
gps_records:      [10:00, 10:01, 10:02, ...]
engine_states:    [09:55, 10:03, 10:07, ...]  # Sparse!
odometer:         [10:00, 10:05, 10:10, ...]  # Sparse!
```

Solution: For each GPS record, find nearest engine state and odometer by timestamp.

#### 2. URL Length Limits (Samsara Driver Assignments)

Samsara endpoint takes comma-separated vehicle IDs in query string:
```
/fleet/drivers/assignments?vehicleIds=id1,id2,...,id500
```

With large fleets, this exceeds URL length limits (2000-8000 chars).

**Solution**: `_fetch_driver_assignments_chunked()`
- Split vehicle IDs into chunks of 50
- Multiple requests, combined results
- Defined in `utils/fetch_data.py:195`

#### 3. Provider-Specific Naming

| Unified Schema    | Motive Field      | Samsara Field           |
|-------------------|-------------------|-------------------------|
| `fleet_number`    | `vehicle.number`  | `vehicle_stats.name`    |
| `provider_vehicle_id` | `vehicle.vehicle_id` | `vehicle_stats.vehicle_id` |
| `latitude`        | `location.latitude` | `gps.latitude`          |
| `engine_state`    | `location.state`  | `engine_states.value`   |

Flatten functions handle this mapping.

---

## Storage Layer

The storage layer provides atomic, efficient persistence for telemetry DataFrames.

### ParquetFileHandler

**Location**: `utils/file_io.py:65`

**Responsibilities**:
1. Load existing Parquet files (returns `None` on error/missing)
2. Save DataFrames with atomic writes
3. Provide file metadata (size, existence)

### Atomic Write Implementation

**Why**: Parquet writes are NOT atomic by default. If the process crashes mid-write, the file is corrupted.

**Solution**: Temp file + atomic rename

```python
def save(self, dataframe: pd.DataFrame) -> None:
    # 1. Write to temp file in same directory
    with tempfile.NamedTemporaryFile(
        suffix='.parquet.tmp',
        dir=parquet_path.parent,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    # 2. Write complete file
    dataframe.to_parquet(temp_path, compression='snappy')

    # 3. Atomic rename (POSIX guarantees atomicity)
    temp_path.replace(parquet_path)
```

**Guarantees**:
- Original file untouched until new file is complete
- If crash occurs at any point, original data remains intact
- No partial/corrupt files

### Parquet Format Benefits

**Why Parquet over CSV/JSON?**

| Feature              | Parquet | CSV  | JSON |
|----------------------|---------|------|------|
| Columnar storage     | ✅      | ❌   | ❌   |
| Compression          | ✅ High | ⚠️ Low | ⚠️ Low |
| Schema enforcement   | ✅      | ❌   | ❌   |
| Fast column queries  | ✅      | ❌   | ❌   |
| Type preservation    | ✅      | ❌   | ⚠️ Partial |
| BI tool compatible   | ✅      | ✅   | ❌   |

**Typical Compression**: 1M records ≈ 50MB Parquet vs 500MB CSV

### Configuration

```yaml
storage:
  parquet_path: "data/telemetry.parquet"
  parquet_compression: "snappy"  # Or 'gzip', 'brotli', 'zstd'
```

**Compression Codecs**:
- `snappy` - **Recommended**: Fast compression/decompression, moderate ratio
- `gzip` - Higher compression, slower
- `zstd` - Best of both worlds (requires PyArrow >= 0.16)
- `brotli` - Highest compression, slowest

---

## Core Concepts

### 1. EndpointDefinition (Abstract Base)

**Location**: `models/shared_response_models.py:220`

The foundation of the entire system. Every endpoint inherits from this base class.

```python
class EndpointDefinition(ABC, BaseModel, Generic[ResponseModelT, ItemT]):
    """
    Abstract base for self-describing API endpoints.

    Type Parameters:
        ResponseModelT: Full response model (e.g., VehiclesResponse)
        ItemT: Individual item type (e.g., Vehicle)
    """

    endpoint_path: str                    # e.g., '/v1/vehicles'
    http_method: HTTPMethod               # GET, POST, etc.
    description: str
    path_parameters: tuple[PathParameterSpec, ...]
    query_parameters: tuple[QueryParameterSpec, ...]
    is_paginated: bool

    # Uniform interface (implemented by subclasses)
    @abstractmethod
    def build_request_spec(...) -> RequestSpec: ...

    @abstractmethod
    def parse_response(...) -> ParsedResponse[ItemT]: ...

    @abstractmethod
    def get_initial_pagination_state() -> PaginationState: ...
```

**Key Methods**:
- `build_url()` - Constructs full URLs with path parameters
- `build_query_params()` - Serializes query params (handles dates, lists, etc.)
- `build_request_spec()` - **Abstract** - Provider-specific request building
- `parse_response()` - **Abstract** - Provider-specific response parsing

---

### 2. Provider-Specific Endpoint Definitions

#### Motive

```python
class MotiveEndpointDefinition(EndpointDefinition[ResponseModelT, ItemT]):
    """
    Motive uses:
    - Authentication: X-API-Key header
    - Pagination: Offset-based (page_no, per_page)
    - Response metadata: total count, current page
    """

    response_model: type[ResponseModelT]
    item_extractor_method: str  # Method name to call on response
    max_per_page: int = 100

    def build_request_spec(self, credentials, pagination_state, **params):
        # Inject X-API-Key header
        headers = {'X-API-Key': credentials.api_key.get_secret_value()}
        # ... build URL, query params, etc.
        return RequestSpec(...)

    def parse_response(self, response_json):
        # Parse into typed model
        parsed = self.response_model.model_validate(response_json)
        # Extract items
        items = getattr(parsed, self.item_extractor_method)()
        # Compute next page state
        return ParsedResponse(items=items, pagination=...)
```

#### Samsara

```python
class SamsaraEndpointDefinition(EndpointDefinition[ResponseModelT, ItemT]):
    """
    Samsara uses:
    - Authentication: Bearer token
    - Pagination: Cursor-based (after parameter)
    - Response metadata: endCursor, hasNextPage
    """

    def build_request_spec(self, credentials, pagination_state, **params):
        # Inject Bearer token
        headers = {'Authorization': f'Bearer {credentials.api_key.get_secret_value()}'}
        # ... cursor pagination handling
        return RequestSpec(...)

    def parse_response(self, response_json):
        # Similar to Motive but handles cursor pagination
        return ParsedResponse(...)
```

---

### 3. RequestSpec (The Contract)

**Location**: `models/shared_request_models.py:28`

The "contract" between endpoint definitions (producers) and clients (consumers).

```python
@dataclass(frozen=True)
class RequestSpec:
    """
    Complete specification for an HTTP request.

    EndpointDefinition builds this.
    TelemetryClient executes it.
    """
    url: str                          # Full URL ready to request
    method: HTTPMethod
    headers: dict[str, str]           # Including auth!
    query_params: dict[str, str]      # All values serialized to strings
    body: dict[str, Any] | None

    # Execution config
    timeout: tuple[int, int]          # (connect, read)
    max_retries: int
    retry_backoff_factor: float
    verify_ssl: bool | str
```

**Key Insight**: The client never needs to know about:
- How to authenticate (auth headers are pre-built)
- How to paginate (pagination params are pre-built)
- How to serialize parameters (already done)

---

### 4. PaginationState (Uniform Pagination)

**Location**: `models/shared_response_models.py:90`

Abstracts different pagination styles into a uniform interface.

```python
@dataclass(frozen=True)
class PaginationState:
    """
    Provider-agnostic pagination state.

    Supports:
    - Offset-based (Motive): {'page_no': 2, 'per_page': 100}
    - Cursor-based (Samsara): {'after': 'cursor_token_here'}
    """
    has_next_page: bool
    next_page_params: dict[str, int | str]  # Flexible!
    total_items: int | None                  # For offset pagination
    current_page: int | None                 # For offset pagination
    current_cursor: str | None               # For cursor pagination

    @classmethod
    def finished(cls) -> 'PaginationState': ...

    @classmethod
    def first_page(cls, per_page: int = 100) -> 'PaginationState': ...

    @classmethod
    def initial_cursor(cls) -> 'PaginationState': ...
```

---

### 5. ParsedResponse (Uniform Output)

**Location**: `models/shared_response_models.py:152`

What you get back from `endpoint.parse_response()` or `client.fetch()`.

```python
@dataclass
class ParsedResponse(Generic[ItemT]):
    """
    Uniform container for parsed API responses.
    """
    items: list[ItemT]              # Typed items!
    pagination: PaginationState      # How to get next page
    raw_response: BaseModel | None   # Full response model

    @property
    def item_count(self) -> int: ...

    @property
    def has_more(self) -> bool: ...
```

---

## Design Principles

### 1. Separation of Concerns

| Component | Responsibility |
|-----------|---------------|
| **EndpointDefinition** | Knows API schema, builds requests, parses responses |
| **TelemetryClient** | Executes HTTP, handles retries, manages connections |
| **Provider** | Combines config + endpoints + client into convenient API |
| **Registry** | Provides discovery and string-based lookup |

### 2. Open-Closed Principle

The system is **open for extension, closed for modification**:

- Adding a new provider: Create new `XyzEndpointDefinition` subclass
- Adding a new endpoint: Add to provider's endpoint registry class
- Client code doesn't change

### 3. Dependency Inversion

```
High-level (Provider) → depends on → EndpointDefinition (interface)
                                              ↑
                                    implements by
                                              ↑
                        MotiveEndpointDefinition, SamsaraEndpointDefinition
```

Client code depends on abstractions (EndpointDefinition), not concrete implementations.

### 4. Type Safety Throughout

```python
# Full type inference
endpoint: MotiveEndpointDefinition[VehiclesResponse, Vehicle]
response: ParsedResponse[Vehicle] = endpoint.parse_response(...)
items: list[Vehicle] = response.items
vehicle: Vehicle = items[0]
vehicle.number  # Type-safe access!
```

---

## Usage Patterns

### Pipeline Patterns (Recommended for Data Collection)

#### Pattern 1: Scheduled Data Collection

**Use Case**: Cron job to keep Parquet file up-to-date

```python
from fleet_telemetry_hub.pipeline import TelemetryPipeline

# One-liner for cron jobs
TelemetryPipeline('config/telemetry_config.yaml').run()
```

**Crontab Example**:
```bash
# Run every day at 2 AM
0 2 * * * cd /path/to/project && python -m fleet_telemetry_hub.pipeline config/telemetry_config.yaml
```

#### Pattern 2: Pipeline with Result Access

**Use Case**: ETL jobs that need to process results immediately

```python
from fleet_telemetry_hub.pipeline import TelemetryPipeline

pipeline = TelemetryPipeline('config.yaml')
pipeline.run()

# Access the resulting DataFrame
df = pipeline.dataframe
print(f"Fetched {len(df)} records from {df['vin'].nunique()} vehicles")

# Export to various formats
df.to_csv('telemetry.csv', index=False)
df.to_excel('telemetry.xlsx', index=False)

# Or analyze directly
recent_data = df[df['timestamp'] >= '2025-01-01']
avg_speed = recent_data.groupby('vin')['speed_mph'].mean()
```

#### Pattern 3: Custom Date Range Backfill

**Use Case**: One-time historical data extraction

```python
from datetime import datetime
from fleet_telemetry_hub.config import load_config, TelemetryConfig
from fleet_telemetry_hub.pipeline import TelemetryPipeline

# Temporarily override config for backfill
config = load_config('config.yaml')
config.pipeline.default_start_date = '2023-01-01'
config.pipeline.batch_increment_days = 7.0  # Larger batches for backfill

# Run pipeline (will fetch from 2023-01-01 to now)
pipeline = TelemetryPipeline.from_config(config)
pipeline.run()
```

#### Pattern 4: Pipeline Error Handling

**Use Case**: Production deployments with monitoring

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
except Exception as e:
    logger.exception("Unexpected error in pipeline")
    raise
```

---

### API Patterns (Direct Provider Interaction)

#### Pattern 5: Simple Data Extraction

**Use Case**: One-off queries, exploratory analysis

```python
from fleet_telemetry_hub import Provider
from fleet_telemetry_hub.config.loader import load_config

config = load_config('config.yaml')
motive = Provider.from_config('motive', config.providers['motive'])

# Fetch and process vehicles
for vehicle in motive.fetch_all('vehicles'):
    print(f"Vehicle {vehicle.number}: {vehicle.vin}")
```

#### Pattern 6: Multi-Endpoint Batch (Connection Reuse)

**Use Case**: Fetching related data from multiple endpoints efficiently

```python
with motive.client() as client:
    # Single SSL handshake, connection pooling
    vehicles = list(client.fetch_all(motive.endpoint('vehicles')))
    groups = list(client.fetch_all(motive.endpoint('groups')))
    users = list(client.fetch_all(motive.endpoint('users')))

    # Process all together
    process_fleet_data(vehicles, groups, users)
```

#### Pattern 7: Dynamic Endpoint Selection

**Use Case**: CLI tools, config-driven applications

```python
from fleet_telemetry_hub import EndpointRegistry

registry = EndpointRegistry.instance()
config = get_user_config()  # From CLI args, config file, etc.

provider_name = config['provider']
endpoint_name = config['endpoint']

if registry.has(provider_name, endpoint_name):
    endpoint = registry.get(provider_name, endpoint_name)
    with provider.client() as client:
        for item in client.fetch_all(endpoint):
            print(item)
else:
    print(f"Unknown endpoint: {provider_name}/{endpoint_name}")
```

#### Pattern 8: Multi-Provider Aggregation

**Use Case**: Combining data from multiple providers manually

```python
from fleet_telemetry_hub import ProviderManager

manager = ProviderManager.from_config(config)

all_vehicles = []
for provider_name, provider in manager.enabled_providers():
    vehicles = list(provider.fetch_all('vehicles'))
    all_vehicles.extend(vehicles)

# Now have vehicles from all providers
analyze_fleet(all_vehicles)
```

#### Pattern 9: DataFrame Export with Custom Parameters

**Use Case**: Exporting specific data slices for analysis

```python
from fleet_telemetry_hub import Provider
from datetime import date

config = load_config('config.yaml')
samsara = Provider.from_config('samsara', config.providers['samsara'])

# Fetch with parameters and convert to DataFrame
df = samsara.to_dataframe(
    'vehicle_locations',
    vehicle_id=12345,
    start_date=date(2025, 1, 1),
    end_date=date(2025, 1, 31),
)

# Analyze
print(df.describe())
print(df[['timestamp', 'latitude', 'longitude', 'speed_mph']].head())

# Export
df.to_parquet('january_vehicle_12345.parquet', compression='snappy')
```

---

## Extension Guide

### Adding a New Provider to the Pipeline

To add a new provider (e.g., Geotab) to the complete system, you need to:

1. **Add API abstraction** (for direct API access)
2. **Add fetch function** (for pipeline integration)
3. **Add flatten function** (for schema transformation)

#### Step 1: Create Response Models

**File**: `models/geotab_responses.py`

```python
from pydantic import BaseModel
from datetime import datetime

class GeotabVehicle(BaseModel):
    id: str
    name: str
    vin: str
    # ... provider-specific fields

class GeotabLocation(BaseModel):
    device_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    speed: float
    # ... provider-specific fields
```

#### Step 2: Create Endpoint Definitions

**File**: `models/geotab_requests.py`

```python
from fleet_telemetry_hub.models.shared_response_models import EndpointDefinition

class GeotabEndpointDefinition(EndpointDefinition[ResponseModelT, ItemT]):
    """
    Geotab-specific endpoint implementation.
    - Authentication: Session token in headers
    - Pagination: Results parameter with fromVersion
    """

    def build_request_spec(self, credentials, pagination_state, **params):
        # Implement Geotab auth and pagination
        headers = {'Authorization': f'Session {credentials.session_token}'}
        # ... build request
        return RequestSpec(...)

    def parse_response(self, response_json):
        # Parse Geotab response format
        parsed = self.response_model.model_validate(response_json)
        items = parsed.results
        # ... compute pagination
        return ParsedResponse(items=items, pagination=...)

class GeotabEndpoints:
    VEHICLES: GeotabEndpointDefinition[GeotabVehiclesResponse, GeotabVehicle] = ...
    LOCATIONS: GeotabEndpointDefinition[GeotabLocationsResponse, GeotabLocation] = ...
```

#### Step 3: Register in EndpointRegistry

**File**: `registry.py`

```python
def _register_geotab_endpoints(self):
    from .models.geotab_requests import GeotabEndpoints
    geotab_endpoints = GeotabEndpoints.get_all_endpoints()
    return {name.lower(): endpoint for name, endpoint in geotab_endpoints.items()}

# In __init__:
self._registry = {
    'motive': self._register_motive_endpoints(),
    'samsara': self._register_samsara_endpoints(),
    'geotab': self._register_geotab_endpoints(),  # Add here
}
```

#### Step 4: Create Flatten Function

**File**: `utils/geotab_funcs.py`

```python
from fleet_telemetry_hub.models.geotab_responses import GeotabLocation, GeotabVehicle
from typing import Any

def flatten_geotab_location(
    location: GeotabLocation,
    vehicle: GeotabVehicle,
) -> dict[str, Any]:
    """
    Flatten Geotab location and vehicle to unified schema.

    Maps Geotab-specific field names to TELEMETRY_COLUMNS.
    """
    return {
        'provider': 'geotab',
        'provider_vehicle_id': vehicle.id,
        'vin': vehicle.vin,
        'fleet_number': vehicle.name,
        'timestamp': location.timestamp,
        'latitude': location.latitude,
        'longitude': location.longitude,
        'speed_mph': location.speed * 0.621371,  # km/h to mph
        'heading_degrees': location.bearing,
        'engine_state': None,  # If not available
        'driver_id': None,     # If not available
        'driver_name': None,
        'location_description': None,
        'odometer': location.odometer_km * 0.621371,  # km to miles
    }
```

#### Step 5: Create Fetch Function

**File**: `utils/fetch_data.py` (add to existing file)

```python
def fetch_geotab_data(
    provider: Provider,
    start_datetime: datetime,
    end_datetime: datetime,
) -> list[dict[str, Any]]:
    """Fetch and transform Geotab telemetry data."""
    logger.info('Fetching Geotab data: %s to %s', start_datetime, end_datetime)

    records: list[dict[str, Any]] = []

    with provider.client() as client:
        # 1. Fetch vehicles
        vehicles = list(client.fetch_all(provider.endpoint('vehicles')))

        # 2. Fetch locations for each vehicle
        for vehicle in vehicles:
            locations = client.fetch_all(
                provider.endpoint('locations'),
                device_id=vehicle.id,
                from_date=start_datetime,
                to_date=end_datetime,
            )

            # 3. Flatten to unified schema
            for location in locations:
                record = flatten_geotab_location(location, vehicle)
                records.append(record)

    logger.info('Fetched %d Geotab records', len(records))
    return records
```

#### Step 6: Register Fetch Function

**File**: `pipeline.py`

```python
from fleet_telemetry_hub.utils import fetch_motive_data, fetch_samsara_data, fetch_geotab_data

PROVIDER_FETCH_FUNCTIONS: dict[str, FetchFunction] = {
    'motive': fetch_motive_data,
    'samsara': fetch_samsara_data,
    'geotab': fetch_geotab_data,  # Add here
}
```

#### Step 7: Add Configuration

**File**: `config/telemetry_config.yaml`

```yaml
providers:
  geotab:
    enabled: true
    base_url: "https://api.geotab.com"
    api_key: "your-geotab-api-key"
    request_timeout: [10, 30]
    max_retries: 5
    retry_backoff_factor: 2.0
    verify_ssl: true
```

#### Done!

The new provider now works with:
- Direct API access via `Provider`
- Automatic pipeline integration
- Unified schema output

```python
# Direct API usage
geotab = Provider.from_config('geotab', config.providers['geotab'])
vehicles = list(geotab.fetch_all('vehicles'))

# Pipeline usage (automatic)
pipeline = TelemetryPipeline('config.yaml')
pipeline.run()  # Includes Geotab data now!
```

---

### Adding a New API Endpoint (Existing Provider)

1. **Create response models** (in `models/xyz_responses.py`):

```python
class XyzVehicle(BaseModel):
    id: str
    name: str
    # ... provider-specific fields

class XyzVehiclesResponse(BaseModel):
    data: list[XyzVehicle]
    pagination: XyzPaginationInfo

    def get_vehicles(self) -> list[XyzVehicle]:
        return self.data
```

2. **Create endpoint definition** (in `models/xyz_requests.py`):

```python
class XyzEndpointDefinition(EndpointDefinition[ResponseModelT, ItemT]):
    """Provider-specific implementation."""

    def build_request_spec(self, credentials, pagination_state, **params):
        # Implement provider's auth pattern
        # Implement provider's pagination
        return RequestSpec(...)

    def parse_response(self, response_json):
        # Parse response
        # Extract items
        # Compute pagination state
        return ParsedResponse(...)

    def get_initial_pagination_state(self):
        # Return initial state for this provider
        return PaginationState(...)
```

3. **Create endpoint registry**:

```python
class XyzEndpoints:
    VEHICLES: XyzEndpointDefinition[XyzVehiclesResponse, XyzVehicle] = (
        XyzEndpointDefinition(
            endpoint_path='/api/v1/vehicles',
            http_method=HTTPMethod.GET,
            description='List all vehicles',
            is_paginated=True,
            response_model=XyzVehiclesResponse,
            item_extractor_method='get_vehicles',
        )
    )

    @classmethod
    def get_all_endpoints(cls):
        return {
            name: value
            for name, value in vars(cls).items()
            if isinstance(value, XyzEndpointDefinition)
        }
```

4. **Register in global registry** (in `registry.py`):

```python
def _register_xyz_endpoints(self):
    from .models.xyz_requests import XyzEndpoints
    xyz_endpoints = XyzEndpoints.get_all_endpoints()
    return {name.lower(): endpoint for name, endpoint in xyz_endpoints.items()}

# In __init__:
self._registry = {
    'motive': self._register_motive_endpoints(),
    'samsara': self._register_samsara_endpoints(),
    'xyz': self._register_xyz_endpoints(),  # Add here
}
```

5. **Done!** The new provider now works with all abstraction layers:

```python
# Works immediately
provider = Provider.from_config('xyz', xyz_config)
for vehicle in provider.fetch_all('vehicles'):
    print(vehicle.name)
```

---

## Summary

The Fleet Telemetry Hub provides a **complete, layered architecture** for working with fleet telemetry data:

### Two-Tier System

**Tier 1: Data Pipeline System** (High-Level, Automated ETL)
- `TelemetryPipeline` - Main orchestrator for scheduled data collection
- `schema.py` - Unified schema for cross-provider data normalization
- `fetch_data.py` - Provider-specific data extraction and transformation
- `ParquetFileHandler` - Atomic, efficient data persistence
- **Use when**: Building scheduled data collection, creating unified datasets, historical backfills

**Tier 2: API Abstraction Framework** (Low-Level, Direct Access)
- **Level 1 (Endpoints)**: Self-describing API endpoint objects
- **Level 2 (Client)**: Provider-agnostic HTTP client
- **Level 3 (Registry)**: Dynamic endpoint discovery
- **Level 4 (Provider)**: High-level facade for direct API interaction
- **Use when**: One-off queries, custom integrations, exploratory analysis

### Core Principles

1. **Endpoints are self-describing** - All API knowledge lives in endpoint definitions
2. **Unified schema** - All provider data flows through `TELEMETRY_COLUMNS`
3. **Provider independence** - Add/remove providers without changing core logic
4. **Type safety throughout** - Pydantic models from API → DataFrame
5. **Incremental processing** - Batched fetching with automatic progress preservation
6. **Data integrity** - Atomic writes, deduplication, schema enforcement

### Project Structure

```
fleet_telemetry_hub/
├── pipeline.py              # Main pipeline orchestrator
├── schema.py                # Unified telemetry schema
├── client.py                # HTTP client (API abstraction)
├── provider.py              # Provider facade (API abstraction)
├── registry.py              # Endpoint discovery (API abstraction)
│
├── config/
│   ├── config_models.py     # Configuration Pydantic models
│   └── loader.py            # YAML config loader
│
├── models/
│   ├── shared_request_models.py   # RequestSpec, HTTPMethod, etc.
│   ├── shared_response_models.py  # EndpointDefinition, ParsedResponse
│   ├── motive_requests.py         # Motive endpoint definitions
│   ├── motive_responses.py        # Motive Pydantic models
│   ├── samsara_requests.py        # Samsara endpoint definitions
│   └── samsara_responses.py       # Samsara Pydantic models
│
└── utils/
    ├── fetch_data.py        # Provider fetch functions (pipeline)
    ├── file_io.py           # Parquet I/O handler (pipeline)
    ├── logger.py            # Centralized logging setup
    ├── motive_funcs.py      # Motive flatten functions
    ├── samsara_funcs.py     # Samsara flatten functions
    └── truststore_context.py # SSL/TLS utilities
```

### When to Use What

| Use Case | Recommended Approach |
|----------|---------------------|
| Scheduled data collection (cron) | `TelemetryPipeline` |
| Historical backfill | `TelemetryPipeline` with custom dates |
| Unified multi-provider dataset | `TelemetryPipeline` |
| One-off API query | `Provider.fetch_all()` |
| Custom integration | `TelemetryClient` + endpoints |
| Building CLI tools | `EndpointRegistry` + `Provider` |
| Exploratory analysis | `Provider.to_dataframe()` |

### Technology Stack

- **Python 3.12+** - Modern type syntax, performance
- **Pydantic 2.x** - Data validation, type safety
- **Pandas** - DataFrame manipulation
- **PyArrow** - Parquet I/O, columnar storage
- **HTTPX** - Modern HTTP client
- **Tenacity** - Retry logic

### Key Features

✅ Multi-provider support (Motive, Samsara, extensible)
✅ Type-safe throughout (Pydantic + type hints)
✅ Automatic pagination handling
✅ Smart retry logic with backoff
✅ Rate limit handling
✅ Network resilience (proxies, SSL)
✅ Incremental updates with lookback
✅ Atomic file writes
✅ Deduplication on (VIN, timestamp)
✅ Comprehensive logging
✅ Zero-downtime data collection

The architecture ensures that whether you're doing ad-hoc API exploration or running production data pipelines, you have the right abstraction at the right level.
