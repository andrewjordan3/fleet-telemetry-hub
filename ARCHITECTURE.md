# Fleet Telemetry Hub - Architecture Documentation

This document explains the abstraction layers and design philosophy of the Fleet Telemetry Hub.

## Table of Contents

1. [Overview](#overview)
2. [Abstraction Layers](#abstraction-layers)
3. [Core Concepts](#core-concepts)
4. [Design Principles](#design-principles)
5. [Usage Patterns](#usage-patterns)
6. [Extension Guide](#extension-guide)

---

## Overview

Fleet Telemetry Hub uses a **layered abstraction architecture** that progressively simplifies API interaction. The system is designed around the principle that **endpoints should be self-describing objects** that encapsulate all knowledge needed to interact with them.

### Key Design Goals

- ✅ **Provider Agnostic**: Client code works identically with any provider
- ✅ **Type Safe**: Full type hints and Pydantic validation throughout
- ✅ **Zero Duplication**: Shared logic lives in base classes
- ✅ **Pagination Transparent**: Different pagination styles (offset, cursor) handled uniformly
- ✅ **Auth Transparent**: Authentication injected by endpoint definitions
- ✅ **Progressive Disclosure**: Simple tasks are simple, complex tasks possible

---

## Abstraction Layers

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
from fleet_telemetry_hub import get_registry

registry = get_registry()

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

### Pattern 1: Simple Data Extraction

```python
from fleet_telemetry_hub import Provider
from fleet_telemetry_hub.config.loader import load_config

config = load_config('config.yaml')
motive = Provider.from_config('motive', config.providers['motive'])

# Just fetch and process
for vehicle in motive.fetch_all('vehicles'):
    save_to_database(vehicle)
```

### Pattern 2: Multi-Endpoint Batch

```python
with motive.client() as client:
    vehicles = list(client.fetch_all(motive.endpoint('vehicles')))
    groups = list(client.fetch_all(motive.endpoint('groups')))
    users = list(client.fetch_all(motive.endpoint('users')))

    # Process all together
    process_fleet_data(vehicles, groups, users)
```

### Pattern 3: Dynamic Endpoint Selection

```python
from fleet_telemetry_hub import get_registry

registry = get_registry()
config = get_user_config()  # From CLI args, config file, etc.

provider_name = config['provider']
endpoint_name = config['endpoint']

if registry.has(provider_name, endpoint_name):
    endpoint = registry.get(provider_name, endpoint_name)
    # ... fetch data
```

### Pattern 4: Multi-Provider Aggregation

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

---

## Extension Guide

### Adding a New Provider

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

The Fleet Telemetry Hub architecture provides **multiple levels of abstraction** so you can choose the right level for your needs:

- **Level 1 (Endpoints)**: Maximum control, direct endpoint usage
- **Level 2 (Client)**: Provider-agnostic HTTP execution
- **Level 3 (Registry)**: Dynamic endpoint lookup and discovery
- **Level 4 (Provider)**: Complete abstraction, recommended for most use cases

All layers share the same core principle: **endpoints are self-describing objects** that encapsulate all provider-specific knowledge, allowing the rest of the system to be completely provider-agnostic.
