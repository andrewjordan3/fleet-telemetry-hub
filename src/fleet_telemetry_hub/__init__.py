# fleet_telemetry_hub/__init__.py
"""
Fleet Telemetry Hub - Unified API client for fleet telematics providers.

This package provides a type-safe, provider-agnostic interface for accessing
fleet telemetry APIs from multiple providers (Motive, Samsara, etc.).

Quick Start:
    >>> from fleet_telemetry_hub import Provider
    >>> from fleet_telemetry_hub.config.loader import load_config
    >>>
    >>> config = load_config("config.yaml")
    >>> motive = Provider.from_config("motive", config.providers["motive"])
    >>>
    >>> for vehicle in motive.fetch_all("vehicles"):
    ...     print(vehicle.number)
"""

__version__ = '0.1.0'

# Mid-level API
from .client import APIError, RateLimitError, TelemetryClient

# Configuration
from .config.config_models import (
    LoggingConfig,
    PipelineConfig,
    ProviderConfig,
    StorageConfig,
    TelemetryConfig,
)
from .config.loader import load_config

# Low-level API (for advanced usage)
from .models import (
    EndpointDefinition,
    HTTPMethod,
    MotiveEndpointDefinition,
    MotiveEndpoints,
    PaginationState,
    ParsedResponse,
    ProviderCredentials,
    RateLimitInfo,
    RequestSpec,
    SamsaraEndpointDefinition,
    SamsaraEndpoints,
)

# Pipeline
from .pipeline import TelemetryPipeline
from .provider import Provider, ProviderManager
from .registry import (
    EndpointNotFoundError,
    EndpointRegistry,
    ProviderNotFoundError,
)

__all__: list[str] = [
    # Exceptions
    'APIError',
    # Low-level types (for advanced usage)
    'EndpointDefinition',
    'EndpointNotFoundError',
    'EndpointRegistry',
    'HTTPMethod',
    'LoggingConfig',
    'MotiveEndpointDefinition',
    # Provider-specific endpoints
    'MotiveEndpoints',
    'PaginationState',
    'ParsedResponse',
    'PipelineConfig',
    # High-level API (recommended for most users)
    'Provider',
    'ProviderConfig',
    'ProviderCredentials',
    'ProviderManager',
    'ProviderNotFoundError',
    'RateLimitError',
    'RateLimitInfo',
    'RequestSpec',
    'SamsaraEndpointDefinition',
    'SamsaraEndpoints',
    'StorageConfig',
    # Client
    'TelemetryClient',
    'TelemetryConfig',
    # Pipeline
    'TelemetryPipeline',
    # Version
    '__version__',
    # Configuration
    'load_config',
]
