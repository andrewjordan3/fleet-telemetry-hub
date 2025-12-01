# Fleet Telemetry Hub

A robust, type-safe Python client for fleet telemetry provider APIs. Seamlessly integrate with multiple telematics platforms (Motive, Samsara) through a unified, provider-agnostic interface.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Type Checked: mypy](https://img.shields.io/badge/type_checked-mypy-blue.svg)](http://mypy-lang.org/)
[![Code Style: Ruff](https://img.shields.io/badge/code_style-ruff-black.svg)](https://github.com/astral-sh/ruff)

## Features

- **Multi-Provider Support**: Unified interface for Motive and Samsara APIs
- **Type-Safe**: Full type hints and Pydantic models for request/response validation
- **Automatic Pagination**: Seamlessly iterate through paginated results
- **Smart Retry Logic**: Exponential backoff with configurable retry strategies
- **Rate Limit Handling**: Automatic throttling and retry-after support
- **Network Resilience**: Designed to work behind corporate proxies (Zscaler, etc.)
- **SSL Flexibility**: Configurable SSL verification with custom CA bundle support
- **Connection Pooling**: Efficient HTTP connection management
- **Data Export**: Built-in Parquet export with pandas integration

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

### Basic Usage

```python
from fleet_telemetry_hub.client import TelemetryClient
from fleet_telemetry_hub.models.shared_response_models import ProviderCredentials

# Configure credentials
credentials = ProviderCredentials(
    base_url="https://api.gomotive.com",
    api_token="your-api-token-here",
    timeout=(10, 30),
    verify_ssl=True
)

# Create client
with TelemetryClient(credentials) as client:
    # Fetch all vehicles with automatic pagination
    for vehicle in client.fetch_all(endpoint):
        print(f"Vehicle: {vehicle.name}")
```

### Using Configuration File

Create a configuration file at `config/telemetry_config.yaml`:

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
    rate_limit_requests_per_second: 10

  samsara:
    enabled: true
    base_url: "https://api.samsara.com"
    api_key: "your-samsara-api-key"
    request_timeout: [10, 30]
    max_retries: 5
    retry_backoff_factor: 2.0
    verify_ssl: true
    rate_limit_requests_per_second: 5

pipeline:
  default_start_date: "2024-01-01"
  lookback_days: 7
  request_delay_seconds: 0.5
  use_truststore: false

storage:
  parquet_path: "data/fleet_telemetry.parquet"
  parquet_compression: "snappy"

logging:
  file_path: "logs/fleet_telemetry.log"
  console_level: "INFO"
  file_level: "DEBUG"
```

Then load and use the configuration:

```python
from fleet_telemetry_hub.config.loader import load_config

# Load configuration
config = load_config("config/telemetry_config.yaml")

# Access provider configs
motive_config = config.providers["motive"]
print(f"Motive API URL: {motive_config.base_url}")
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

### Pagination

The client handles pagination automatically:

```python
# Iterate through all items (automatic pagination)
for item in client.fetch_all(endpoint, request_delay_seconds=0.5):
    process(item)

# Or work with full pages
for page in client.fetch_all_pages(endpoint):
    print(f"Page has {page.item_count} items")
    process_batch(page.items)
```

### DataFrame Export

Convert API data to pandas DataFrame for analysis:

```python
from fleet_telemetry_hub import Provider
from fleet_telemetry_hub.config.loader import load_config

config = load_config("config.yaml")
motive = Provider.from_config("motive", config.providers["motive"])

# Get all vehicles as DataFrame
df = motive.to_dataframe("vehicles")
print(df.head())

# With parameters
from datetime import date

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

### Rate Limiting

Built-in rate limit handling with exponential backoff:

```python
# Client automatically retries on 429 responses
# Respects Retry-After headers
# Configurable retry attempts and backoff
```

### Error Handling

```python
from fleet_telemetry_hub.client import APIError, RateLimitError

try:
    response = client.fetch(endpoint)
except RateLimitError as e:
    print(f"Rate limited, retry after {e.rate_limit_info.retry_after_seconds}s")
except APIError as e:
    print(f"API error {e.status_code}: {e.response_body}")
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
│       ├── client.py              # Main HTTP client
│       ├── config/                # Configuration models and loader
│       │   ├── config_models.py
│       │   └── loader.py
│       ├── models/                # Request/response models
│       │   ├── motive_requests.py
│       │   ├── motive_responses.py
│       │   ├── samsara_requests.py
│       │   ├── samsara_responses.py
│       │   ├── shared_request_models.py
│       │   └── shared_response_models.py
│       └── utils/                 # Utility functions
│           └── truststore_context.py
├── config/
│   └── telemetry_config.yaml     # Example configuration
├── tests/                         # Test suite
├── pyproject.toml                # Project metadata and dependencies
└── README.md                     # This file
```

## Requirements

- Python 3.11 or higher
- Dependencies:
  - `pydantic>=2.0.0` - Data validation
  - `httpx>=0.28.0` - HTTP client
  - `tenacity>=9.1.0` - Retry logic
  - `pyyaml>=6.0.0` - Configuration parsing
  - `pandas>=2.0.0` - Data manipulation
  - `pyarrow>=14.0.0` - Parquet support

## Supported Providers

### Motive (formerly KeepTruckin)

- API Documentation: https://developer.gomotive.com/
- Features: Vehicle tracking, driver logs, fuel data, ELD compliance

### Samsara

- API Documentation: https://developers.samsara.com/
- Features: Fleet management, vehicle health, driver safety, route optimization

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

Project Link: https://github.com/andrewjordan3/fleet_telemetry_hub

## Acknowledgments

- Built with [Pydantic](https://docs.pydantic.dev/) for data validation
- HTTP client powered by [HTTPX](https://www.python-httpx.org/)
- Retry logic using [Tenacity](https://tenacity.readthedocs.io/)
- Type checking with [mypy](http://mypy-lang.org/)
- Code quality with [Ruff](https://github.com/astral-sh/ruff)

## Roadmap

- [ ] Add support for additional providers (Geotab, Verizon Connect)
- [ ] Async client support with `asyncio`
- [ ] CLI tool for common operations
- [ ] GraphQL API support
- [ ] Webhook integration
- [ ] Real-time event streaming

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
