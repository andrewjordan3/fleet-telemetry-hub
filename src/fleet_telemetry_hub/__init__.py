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

# High-level API (recommended)
from .provider import Provider, ProviderManager
from .registry import EndpointNotFoundError, EndpointRegistry, ProviderNotFoundError, get_registry

# Mid-level API
from .client import APIError, RateLimitError, TelemetryClient

# Low-level API (for advanced usage)
from .models.motive_requests import MotiveEndpointDefinition, MotiveEndpoints
from .models.samsara_requests import SamsaraEndpointDefinition, SamsaraEndpoints
from .models.shared_request_models import HTTPMethod, RateLimitInfo, RequestSpec
from .models.shared_response_models import (
    EndpointDefinition,
    PaginationState,
    ParsedResponse,
    ProviderCredentials,
)

# Configuration
from .config.config_models import (
    LoggingConfig,
    PipelineConfig,
    ProviderConfig,
    StorageConfig,
    TelemetryConfig,
)
from .config.loader import load_config

__all__ = [
    # Version
    '__version__',
    # High-level API (recommended for most users)
    'Provider',
    'ProviderManager',
    'EndpointRegistry',
    'get_registry',
    # Exceptions
    'APIError',
    'RateLimitError',
    'EndpointNotFoundError',
    'ProviderNotFoundError',
    # Client
    'TelemetryClient',
    # Configuration
    'load_config',
    'TelemetryConfig',
    'ProviderConfig',
    'PipelineConfig',
    'StorageConfig',
    'LoggingConfig',
    # Low-level types (for advanced usage)
    'EndpointDefinition',
    'ProviderCredentials',
    'RequestSpec',
    'ParsedResponse',
    'PaginationState',
    'RateLimitInfo',
    'HTTPMethod',
    # Provider-specific endpoints
    'MotiveEndpoints',
    'MotiveEndpointDefinition',
    'SamsaraEndpoints',
    'SamsaraEndpointDefinition',
]
