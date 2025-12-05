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

from fleet_telemetry_hub.client import (
    APIError,
    RateLimitError,
    TelemetryClient,
    TransientAPIError,
)
from fleet_telemetry_hub.common import ParquetFileHandler, setup_logger
from fleet_telemetry_hub.config import load_config
from fleet_telemetry_hub.pipeline import PipelineError, TelemetryPipeline
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
    'ParquetFileHandler',
    'PipelineError',
    'Provider',
    'ProviderConfigurationError',
    'ProviderManager',
    'ProviderNotFoundError',
    'RateLimitError',
    'TelemetryClient',
    'TelemetryPipeline',
    'TransientAPIError',
    '__version__',
    'load_config',
    'setup_logger',
]
