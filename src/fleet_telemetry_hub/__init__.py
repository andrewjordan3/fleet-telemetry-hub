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

# Pipeline
from fleet_telemetry_hub.pipeline import TelemetryPipeline

__all__: list[str] = [
    # Pipeline
    'TelemetryPipeline',
    # Version
    '__version__',
]
