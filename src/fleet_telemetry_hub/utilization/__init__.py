"""Public surface for the utilization fetcher package."""

from fleet_telemetry_hub.utilization.fetcher_protocol import UtilizationFetcher
from fleet_telemetry_hub.utilization.motive_fetcher import (
    MotiveUtilizationBundle,
    MotiveUtilizationFetcher,
)
from fleet_telemetry_hub.utilization.samsara_fetcher import (
    SamsaraUtilizationBundle,
    SamsaraUtilizationFetcher,
)
from fleet_telemetry_hub.utilization.vehicle_trip import VehicleTrip

__all__: list[str] = [
    'MotiveUtilizationBundle',
    'MotiveUtilizationFetcher',
    'SamsaraUtilizationBundle',
    'SamsaraUtilizationFetcher',
    'UtilizationFetcher',
    'VehicleTrip',
]
