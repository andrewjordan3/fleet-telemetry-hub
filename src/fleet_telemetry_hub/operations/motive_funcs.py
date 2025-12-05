# fleet_telemetry_hub/utils/motive_funcs.py

import logging
from typing import Any

from fleet_telemetry_hub.models import Vehicle, VehicleLocation, VehicleLocationType

logger: logging.Logger = logging.getLogger(__name__)

# Threshold for GPS drift while still considering a truck idle
IDLING_THRESHOLD_MPH: float = 3.0

__all__: list[str] = ['flatten_motive_location']

# =============================================================================
# Motive-Specific Transformation Functions
# =============================================================================


def _convert_motive_speed_to_mph(
    speed: float | None,
    metric_units: bool,
) -> float | None:
    """
    Convert Motive speed to MPH for unified schema.

    Args:
        speed: Speed value from Motive API.
        metric_units: Whether vehicle reports in metric (km/h).

    Returns:
        Speed in MPH, or None if input was None.
    """
    if speed is None:
        return None

    if metric_units:
        return speed * 0.621371  # km/h to mph

    return speed  # Already in mph


def _derive_motive_engine_state(
    location_type: VehicleLocationType | None,
    speed_mph: float | None,
) -> str | None:
    """
    Derive engine state from Motive location_type and speed.

    Args:
        location_type: Motive's location event type.
        speed_mph: Vehicle speed in MPH.

    Returns:
        'On', 'Off', 'Idle', or None if indeterminate.

    Logic:
        The following priority determines the state:
        - If inputs are None, result is None.
        - vehicle_stopped event implies 'Idle'.
        - ignition_off event implies 'Off'.
        - Any event with Speed > Threshold implies 'On'.
        - Any event with Speed <= Threshold implies 'Idle'.
        - vehicle_moving/driving events (without speed data) imply 'On'.
        - ignition_on event (without speed data) implies 'Idle'.
        - All other cases default to 'Off'.
    """
    # Pre-check: If both passed variables are None then return None.
    if location_type is None and speed_mph is None:
        return None

    type_value: str | None = location_type.value if location_type is not None else None
    # Initialize the return variable
    derived_status: str = 'Off'

    # We match on a tuple of (type_value, speed_mph)
    match (type_value, speed_mph):
        # If type_value is 'vehicle_stopped' or 'gps_stopped' return Idle.
        case ('vehicle_stopped' | 'gps_stopped', _):
            derived_status = 'Idle'

        # If type_value is 'ignition_off' or 'engine_stop' return Off.
        case ('ignition_off' | 'engine_stop', _):
            derived_status = 'Off'

        # If the speed_mph is greater than IDLING_THRESHOLD return On.
        # We use a 'guard' (if condition) here. We must ensure s is not None before comparing.
        case (_, s) if s is not None and s > IDLING_THRESHOLD_MPH:
            derived_status = 'On'

        # If the speed_mph is less than or equal to IDLING_THRESHOLD return Idle.
        # We use a 'guard' (if condition) here. We must ensure s is not None before comparing.
        case (_, s) if s is not None and s <= IDLING_THRESHOLD_MPH:
            derived_status = 'Idle'

        # If type_value is 'vehicle_moving' or 'vehicle_driving' return On.
        # We can use the pipe | character to match multiple strings.
        case ('vehicle_moving' | 'vehicle_driving' | 'gps_moving', _):
            derived_status = 'On'

        # If type_value is 'ignition_on' or 'engine_start' return Idle.
        case ('ignition_on' | 'engine_start', _):
            derived_status = 'Idle'

        # Default/Catch-all: Return 'Off'.
        # The underscore _ acts as a wildcard that matches anything not caught above.
        case _:
            derived_status = 'Off'

    return derived_status


def _get_best_motive_odometer(location: VehicleLocation) -> float | None:
    """
    Get the best available odometer reading from Motive.

    Prefers ECM-reported value (true_odometer) over calculated value.

    Args:
        location: VehicleLocation record.

    Returns:
        Odometer reading in miles, or None if unavailable.
    """
    # Prefer ECM-reported odometer (more accurate)
    if location.true_odometer is not None:
        return location.true_odometer

    # Fall back to calculated odometer
    return location.odometer


def flatten_motive_location(
    location: VehicleLocation,
    vehicle: Vehicle,
) -> dict[str, Any]:
    """
    Flatten a Motive VehicleLocation into the target schema.

    Args:
        location: VehicleLocation record from Motive.
        vehicle: Vehicle record for context (provides VIN, fleet number, units).

    Returns:
        Dictionary with flattened fields matching target schema.
    """
    # Convert speed to MPH if needed
    speed_mph: float | None = _convert_motive_speed_to_mph(
        location.speed, vehicle.metric_units
    )

    # Derive engine state
    engine_state: str | None = _derive_motive_engine_state(
        location.location_type, speed_mph
    )

    # Get best odometer reading
    odometer: float | None = _get_best_motive_odometer(location)

    return {
        'provider': 'motive',
        'provider_vehicle_id': str(vehicle.vehicle_id),
        'vin': vehicle.vin,
        'fleet_number': vehicle.number,
        'timestamp': location.located_at,
        'latitude': location.latitude,
        'longitude': location.longitude,
        'speed_mph': speed_mph,
        'heading_degrees': location.bearing,
        'engine_state': engine_state,
        'driver_id': str(location.driver.driver_id) if location.driver else None,
        'driver_name': location.driver.full_name if location.driver else None,
        'location_description': location.description,
        'odometer': odometer,
    }
