# fleet_telemetry_hub/models/__init__.py

from fleet_telemetry_hub.models.motive_requests import (
    MotiveEndpointDefinition,
    MotiveEndpoints,
)
from fleet_telemetry_hub.models.motive_responses import (
    Vehicle,
    VehicleLocation,
    VehicleLocationType,
)
from fleet_telemetry_hub.models.samsara_requests import (
    SamsaraEndpointDefinition,
    SamsaraEndpoints,
)
from fleet_telemetry_hub.models.samsara_responses import (
    DriverVehicleAssignment,
    GpsRecord,
    VehicleStatsHistoryRecord,
)
from fleet_telemetry_hub.models.shared_request_models import (
    HTTPMethod,
    RateLimitInfo,
    RequestSpec,
)
from fleet_telemetry_hub.models.shared_response_models import (
    EndpointDefinition,
    PaginationState,
    ParsedResponse,
    ProviderCredentials,
)

__all__: list[str] = [
    'DriverVehicleAssignment',
    'EndpointDefinition',
    'GpsRecord',
    'HTTPMethod',
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
