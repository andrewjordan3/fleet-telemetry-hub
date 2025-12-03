# fleet_telemetry_hub/models/__init__.py

from .motive_requests import MotiveEndpointDefinition, MotiveEndpoints
from .motive_responses import Vehicle, VehicleLocation, VehicleLocationType
from .samsara_requests import SamsaraEndpointDefinition, SamsaraEndpoints
from .samsara_responses import (
    DriverVehicleAssignment,
    GpsRecord,
    VehicleStatsHistoryRecord,
)
from .shared_request_models import HTTPMethod, RateLimitInfo, RequestSpec
from .shared_response_models import (
    EndpointDefinition,
    ItemT,
    PaginationState,
    ParsedResponse,
    ProviderCredentials,
)

__all__: list[str] = [
    'DriverVehicleAssignment',
    'EndpointDefinition',
    'GpsRecord',
    'HTTPMethod',
    'ItemT',
    'MotiveEndpointDefinition',
    'MotiveEndpoints',
    'PaginationState',
    'ParsedResponse',
    'ProviderCredentials',
    'RateLimitInfo',
    'RequestSpec',
    'SamsaraEndpointDefinition',
    'SamsaraEndpoints',
    'Vehicle',
    'VehicleLocation',
    'VehicleLocationType',
    'VehicleStatsHistoryRecord',
]
