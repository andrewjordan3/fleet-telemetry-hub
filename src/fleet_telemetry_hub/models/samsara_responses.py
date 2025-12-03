# fleet_telemetry_hub/models/samsara_responses.py
"""
Pydantic response models for Samsara API data structures.

Samsara uses a cleaner response structure than Motive:
- Data is directly in {"data": [...]} without wrapper objects
- Fields use camelCase (mapped to snake_case via aliases)
- Pagination uses cursor-based approach with endCursor/hasNextPage
"""
# pyright: reportUnknownVariableType=false

import logging
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class DriverActivationStatus(str, Enum):
    """Driver account activation status."""

    ACTIVE = 'active'
    DEACTIVATED = 'deactivated'


class VehicleRegulationMode(str, Enum):
    """ELD regulation mode for vehicle."""

    REGULATED = 'regulated'
    UNREGULATED = 'unregulated'


class EngineState(str, Enum):
    """Vehicle engine state values."""

    ON = 'On'
    OFF = 'Off'
    IDLE = 'Idle'


class HarshAccelerationSettingType(str, Enum):
    """Harsh acceleration detection setting."""

    AUTOMATIC = 'automatic'
    MANUAL = 'manual'
    OFF = 'off'


class AssignmentType(str, Enum):
    """Driver-vehicle assignment type."""

    HOS = 'HOS'  # Hours of Service assignment
    DISPATCH = 'Dispatch'  # Dispatch assignment
    MANUAL = 'Manual'  # Manual assignment


class FilterBy(str, Enum):
    """Filter mode for driver-vehicle assignments."""

    VEHICLES = 'vehicles'
    DRIVERS = 'drivers'


# =============================================================================
# Base Configuration
# =============================================================================


class SamsaraModelBase(BaseModel):
    """
    Base class for all Samsara API response models.

    Configuration:
        - extra='ignore': Silently ignore unknown fields from API.
        - populate_by_name=True: Allow both alias (camelCase) and field name (snake_case).
        - str_strip_whitespace=True: Trim whitespace from strings.
    """

    model_config = ConfigDict(
        extra='ignore',
        populate_by_name=True,
        str_strip_whitespace=True,
    )


# =============================================================================
# Embedded/Shared Models
# =============================================================================


class SamsaraTag(SamsaraModelBase):
    """
    Tag for organizing Samsara entities.

    Tags form a hierarchy for grouping vehicles, drivers, and addresses.

    Attributes:
        tag_id: Samsara's internal tag identifier.
        name: Display name for the tag.
        parent_tag_id: Parent tag ID if nested (None for top-level).
    """

    tag_id: str = Field(alias='id')
    name: str
    parent_tag_id: str | None = Field(default=None, alias='parentTagId')


class SamsaraAttribute(SamsaraModelBase):
    """
    Custom attribute attached to Samsara entities.

    Attributes allow flexible metadata on vehicles and other objects.

    Attributes:
        attribute_id: UUID for this attribute definition.
        name: Attribute name (e.g., "Asset Status").
        string_values: List of string values for this attribute.
    """

    attribute_id: str = Field(alias='id')
    name: str
    string_values: list[str] = Field(default_factory=list, alias='stringValues')


class SamsaraGateway(SamsaraModelBase):
    """
    Telematics gateway device information.

    Attributes:
        serial: Device serial number.
        model: Gateway model (e.g., "VG34").
    """

    serial: str
    model: str


class DriverReference(SamsaraModelBase):
    """
    Abbreviated driver reference embedded in vehicle records.

    Attributes:
        driver_id: Samsara's internal driver identifier.
        name: Driver's full name.
    """

    driver_id: str = Field(alias='id')
    name: str


class VehicleReference(SamsaraModelBase):
    """
    Abbreviated vehicle reference embedded in driver records.

    Attributes:
        vehicle_id: Samsara's internal vehicle identifier.
        name: Vehicle display name.
    """

    vehicle_id: str = Field(alias='id')
    name: str


class CarrierSettings(SamsaraModelBase):
    """
    Carrier/company information for HOS compliance.

    Attributes:
        carrier_name: Legal company name.
        main_office_address: Company headquarters address.
        dot_number: USDOT number.
        home_terminal_name: Driver's home terminal name.
        home_terminal_address: Home terminal street address.
    """

    carrier_name: str | None = Field(default=None, alias='carrierName')
    main_office_address: str | None = Field(default=None, alias='mainOfficeAddress')
    dot_number: int | None = Field(default=None, alias='dotNumber')
    home_terminal_name: str | None = Field(default=None, alias='homeTerminalName')
    home_terminal_address: str | None = Field(default=None, alias='homeTerminalAddress')


class EldRuleset(SamsaraModelBase):
    """
    Hours of Service ruleset configuration.

    Attributes:
        cycle: HOS cycle (e.g., "USA 70 hour / 8 day").
        shift: Shift rule (e.g., "US Interstate Property").
        restart: Restart rule (e.g., "34-hour Restart").
        break_rule: Break rule (e.g., "Property (off-duty/sleeper)").
    """

    cycle: str | None = None
    shift: str | None = None
    restart: str | None = None
    break_rule: str | None = Field(default=None, alias='break')


class EldSettings(SamsaraModelBase):
    """
    ELD configuration settings for a driver.

    Attributes:
        rulesets: List of applicable HOS rulesets.
    """

    rulesets: list[EldRuleset] = Field(default_factory=list)


class HosSettings(SamsaraModelBase):
    """
    Hours of Service feature toggles.

    Attributes:
        heavy_haul_exemption_toggle_enabled: Whether heavy haul exemption is available.
    """

    heavy_haul_exemption_toggle_enabled: bool = Field(
        default=False,
        alias='heavyHaulExemptionToggleEnabled',
    )


class AssetReference(SamsaraModelBase):
    """
    Abbreviated asset reference in location stream records.

    Attributes:
        asset_id: Samsara's internal asset/vehicle identifier.
    """

    asset_id: str = Field(alias='id')


class AddressReference(SamsaraModelBase):
    """
    Abbreviated address reference in GPS records.

    Attributes:
        address_id: Samsara's internal address identifier.
        name: Address display name.
    """

    address_id: str = Field(alias='id')
    name: str


class ReverseGeocode(SamsaraModelBase):
    """
    Reverse geocoded location description.

    Attributes:
        formatted_location: Human-readable address string.
    """

    formatted_location: str = Field(alias='formattedLocation')


class GpsLocation(SamsaraModelBase):
    """
    GPS coordinates with optional geofence context.

    Attributes:
        latitude: GPS latitude in decimal degrees.
        longitude: GPS longitude in decimal degrees.
        heading_degrees: Compass heading (0-360).
        accuracy_meters: GPS accuracy radius.
        geofence: Geofence context if vehicle is within one.
    """

    latitude: float
    longitude: float
    heading_degrees: float | None = Field(default=None, alias='headingDegrees')
    accuracy_meters: float | None = Field(default=None, alias='accuracyMeters')
    geofence: dict[str, Any] | None = None  # Complex nested structure, kept flexible

    @property
    def coordinates(self) -> tuple[float, float]:
        """Return (latitude, longitude) tuple."""
        return (self.latitude, self.longitude)


class PolygonVertex(SamsaraModelBase):
    """
    Single vertex in a geofence polygon.

    Attributes:
        latitude: Vertex latitude.
        longitude: Vertex longitude.
    """

    latitude: float
    longitude: float


class GeofencePolygon(SamsaraModelBase):
    """
    Polygon definition for geofence boundaries.

    Attributes:
        vertices: Ordered list of polygon vertices.
    """

    vertices: list[PolygonVertex] = Field(default_factory=list)


class GeofenceSettings(SamsaraModelBase):
    """
    Geofence display and behavior settings.

    Attributes:
        show_addresses: Whether to display addresses in geofence.
    """

    show_addresses: bool = Field(default=True, alias='showAddresses')


class Geofence(SamsaraModelBase):
    """
    Complete geofence definition.

    Attributes:
        polygon: Polygon boundary definition.
        settings: Display and behavior settings.
    """

    polygon: GeofencePolygon | None = None
    settings: GeofenceSettings | None = None


class AssignmentDriverReference(SamsaraModelBase):
    """
    Driver reference in assignment record.

    Attributes:
        driver_id: Samsara's internal driver identifier.
        name: Driver's full name.
    """

    driver_id: str = Field(alias='id')
    name: str


class AssignmentVehicleReference(SamsaraModelBase):
    """
    Vehicle reference in assignment record with external IDs.

    Attributes:
        vehicle_id: Samsara's internal vehicle identifier.
        name: Vehicle display name.
        external_ids: External system identifiers (includes VIN).
    """

    vehicle_id: str = Field(alias='id')
    name: str
    external_ids: dict[str, str] = Field(default_factory=dict, alias='externalIds')

    def get_vin(self) -> str | None:
        """Extract VIN from external IDs if present."""
        return self.external_ids.get('samsara.vin')

    def get_serial(self) -> str | None:
        """Extract device serial from external IDs if present."""
        return self.external_ids.get('samsara.serial')


# =============================================================================
# Primary Entity Models
# =============================================================================


class SamsaraVehicle(SamsaraModelBase):
    """
    Complete vehicle record from Samsara.

    Attributes:
        vehicle_id: Samsara's internal vehicle identifier (string, not int).
        name: Vehicle display name (often includes description like "PT-275 (Supervisor Truck)").
        vin: Vehicle Identification Number.
        make: Vehicle manufacturer.
        model: Vehicle model name.
        year: Model year as string.
        serial: Gateway serial number.
        notes: Free-form notes.
        camera_serial: Dash camera serial if installed.
        vehicle_regulation_mode: ELD regulation status.
        harsh_acceleration_setting_type: Harsh acceleration detection mode.
        gateway: Telematics gateway device info.
        static_assigned_driver: Permanently assigned driver.
        tags: Organization tags.
        attributes: Custom attributes.
        external_ids: External system identifiers (dict).
        created_at_time: When vehicle was added.
        updated_at_time: Last modification timestamp.
    """

    vehicle_id: str = Field(alias='id')
    name: str
    vin: str | None = None
    make: str | None = None
    model: str | None = None
    year: str | None = None
    serial: str | None = None
    notes: str | None = None
    camera_serial: str | None = Field(default=None, alias='cameraSerial')

    vehicle_regulation_mode: VehicleRegulationMode | None = Field(
        default=None,
        alias='vehicleRegulationMode',
    )
    harsh_acceleration_setting_type: HarshAccelerationSettingType | None = Field(
        default=None,
        alias='harshAccelerationSettingType',
    )

    gateway: SamsaraGateway | None = None
    static_assigned_driver: DriverReference | None = Field(
        default=None,
        alias='staticAssignedDriver',
    )

    tags: list[SamsaraTag] = Field(default_factory=list)
    attributes: list[SamsaraAttribute] = Field(default_factory=list)
    external_ids: dict[str, str] = Field(default_factory=dict, alias='externalIds')

    created_at_time: datetime | None = Field(default=None, alias='createdAtTime')
    updated_at_time: datetime | None = Field(default=None, alias='updatedAtTime')

    @property
    def has_gateway(self) -> bool:
        """Check if vehicle has telematics gateway installed."""
        return self.gateway is not None

    @property
    def is_regulated(self) -> bool:
        """Check if vehicle is ELD-regulated."""
        return self.vehicle_regulation_mode == VehicleRegulationMode.REGULATED

    def get_external_id(self, key: str) -> str | None:
        """
        Get external ID by key.

        Args:
            key: External ID key (e.g., "samsara.vin").

        Returns:
            External ID value or None if not found.
        """
        return self.external_ids.get(key)


class SamsaraDriver(SamsaraModelBase):
    """
    Complete driver record from Samsara.

    Attributes:
        driver_id: Samsara's internal driver identifier.
        name: Driver's full name.
        username: Login username.
        phone: Phone number.
        license_number: Driver's license number.
        license_state: License issuing state.
        timezone: Driver's timezone.
        driver_activation_status: Active or deactivated.
        is_deactivated: Boolean deactivation flag.
        eld_exempt: Whether driver is ELD-exempt.
        eld_exempt_reason: Reason for exemption.
        eld_big_day_exemption_enabled: Big day exemption toggle.
        eld_adverse_weather_exemption_enabled: Adverse weather exemption.
        eld_pc_enabled: Personal conveyance enabled.
        eld_ym_enabled: Yard moves enabled.
        waiting_time_duty_status_enabled: Waiting time as on-duty.
        eld_settings: HOS ruleset configuration.
        carrier_settings: Carrier/terminal information.
        hos_setting: HOS feature toggles.
        static_assigned_vehicle: Permanently assigned vehicle.
        tags: Organization tags.
        has_vehicle_unpinning_enabled: Can driver unpin from vehicle.
        created_at_time: Account creation timestamp.
        updated_at_time: Last modification timestamp.
    """

    driver_id: str = Field(alias='id')
    name: str
    username: str | None = None
    phone: str | None = None

    license_number: str | None = Field(default=None, alias='licenseNumber')
    license_state: str | None = Field(default=None, alias='licenseState')
    timezone: str | None = None

    driver_activation_status: DriverActivationStatus | None = Field(
        default=None,
        alias='driverActivationStatus',
    )
    is_deactivated: bool | None = Field(default=None, alias='isDeactivated')

    # ELD exemptions
    eld_exempt: bool | None = Field(default=None, alias='eldExempt')
    eld_exempt_reason: str | None = Field(default=None, alias='eldExemptReason')
    eld_big_day_exemption_enabled: bool | None = Field(
        default=None,
        alias='eldBigDayExemptionEnabled',
    )
    eld_adverse_weather_exemption_enabled: bool | None = Field(
        default=None,
        alias='eldAdverseWeatherExemptionEnabled',
    )
    eld_pc_enabled: bool | None = Field(default=None, alias='eldPcEnabled')
    eld_ym_enabled: bool | None = Field(default=None, alias='eldYmEnabled')
    waiting_time_duty_status_enabled: bool | None = Field(
        default=None,
        alias='waitingTimeDutyStatusEnabled',
    )

    eld_settings: EldSettings | None = Field(default=None, alias='eldSettings')
    carrier_settings: CarrierSettings | None = Field(
        default=None,
        alias='carrierSettings',
    )
    hos_setting: HosSettings | None = Field(default=None, alias='hosSetting')

    static_assigned_vehicle: VehicleReference | None = Field(
        default=None,
        alias='staticAssignedVehicle',
    )
    tags: list[SamsaraTag] = Field(default_factory=list)

    has_vehicle_unpinning_enabled: bool | None = Field(
        default=None,
        alias='hasVehicleUnpinningEnabled',
    )

    created_at_time: datetime | None = Field(default=None, alias='createdAtTime')
    updated_at_time: datetime | None = Field(default=None, alias='updatedAtTime')

    @property
    def is_active(self) -> bool:
        """Check if driver account is active."""
        return self.driver_activation_status == DriverActivationStatus.ACTIVE

    @property
    def home_terminal(self) -> str | None:
        """Get home terminal name if available."""
        if self.carrier_settings:
            return self.carrier_settings.home_terminal_name
        return None


class SamsaraAddress(SamsaraModelBase):
    """
    Address/location with geofence from Samsara.

    Addresses define named locations with optional geofence boundaries
    for arrival/departure tracking.

    Attributes:
        address_id: Samsara's internal address identifier.
        name: Display name for address.
        formatted_address: Full street address.
        latitude: Center point latitude.
        longitude: Center point longitude.
        geofence: Geofence boundary definition.
        address_types: Types like "yard", "customer", etc.
        tags: Organization tags.
        created_at_time: Creation timestamp.
    """

    address_id: str = Field(alias='id')
    name: str
    formatted_address: str | None = Field(default=None, alias='formattedAddress')
    latitude: float | None = None
    longitude: float | None = None
    geofence: Geofence | None = None
    address_types: list[str] = Field(default_factory=list, alias='addressTypes')
    tags: list[SamsaraTag] = Field(default_factory=list)
    created_at_time: datetime | None = Field(default=None, alias='createdAtTime')

    @property
    def coordinates(self) -> tuple[float, float] | None:
        """Return (latitude, longitude) tuple if available."""
        if self.latitude is not None and self.longitude is not None:
            return (self.latitude, self.longitude)
        return None

    @property
    def has_geofence(self) -> bool:
        """Check if address has a geofence defined."""
        return self.geofence is not None and self.geofence.polygon is not None


class DriverVehicleAssignment(SamsaraModelBase):
    """
    Driver-vehicle assignment record from Samsara.

    Represents a time period when a specific driver was assigned to a vehicle.
    Used for correlating driver identity with GPS/telemetry data.

    Attributes:
        start_time: When assignment started (driver logged into vehicle).
        end_time: When assignment ended (driver logged out).
        is_passenger: Whether driver was logged in as passenger (not primary operator).
        assigned_at_time: Administrative assignment timestamp (may be empty for HOS).
        assignment_type: How assignment was created (HOS, Dispatch, Manual).
        driver: Driver who was assigned.
        vehicle: Vehicle that was assigned.
    """

    start_time: datetime = Field(alias='startTime')
    end_time: datetime = Field(alias='endTime')
    is_passenger: bool = Field(alias='isPassenger')
    assigned_at_time: str | None = Field(default=None, alias='assignedAtTime')
    assignment_type: AssignmentType = Field(alias='assignmentType')
    driver: AssignmentDriverReference
    vehicle: AssignmentVehicleReference

    @property
    def duration_seconds(self) -> float:
        """Calculate assignment duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def duration_hours(self) -> float:
        """Calculate assignment duration in hours."""
        return self.duration_seconds / 3600.0

    @property
    def vin(self) -> str | None:
        """Get vehicle VIN if available."""
        return self.vehicle.get_vin()

    def contains_timestamp(self, timestamp: datetime) -> bool:
        """
        Check if a timestamp falls within this assignment period.

        Args:
            timestamp: Timestamp to check.

        Returns:
            True if timestamp is between start_time and end_time (inclusive).
        """
        return self.start_time <= timestamp <= self.end_time


# =============================================================================
# Vehicle Stats History Models
# =============================================================================


class EngineStateRecord(SamsaraModelBase):
    """
    Single engine state change record.

    Attributes:
        time: Timestamp of state change.
        value: Engine state (On, Off, Idle).
    """

    time: datetime
    value: EngineState

    @field_validator('value', mode='before')
    @classmethod
    def normalize_engine_state(cls, value: str) -> EngineState:
        """Handle case variations in engine state values."""
        try:
            # Try exact match first (On, Off, Idle)
            return EngineState(value)
        except ValueError:
            # Try case-insensitive match
            for state in EngineState:
                if state.value.lower() == value.lower():
                    return state
            logger.warning('Unknown engine state: %r', value)
            raise ValueError(f'Unknown engine state: {value}') from None


class ObdOdometerRecord(SamsaraModelBase):
    """
    OBD-II odometer reading from vehicle ECU.

    Represents the dashboard odometer value reported by the vehicle's
    onboard diagnostics system. Readings are sparse (not every GPS point)
    and must be correlated temporally with GPS records.

    Attributes:
        time: Timestamp when odometer was read from ECU.
        value: Odometer reading in meters (Samsara's standard unit).
    """

    time: datetime
    value: int = Field(description='Odometer reading in meters')

    @property
    def value_miles(self) -> float:
        """Convert meters to miles for US reporting."""
        return self.value / 1609.344

    @property
    def value_kilometers(self) -> float:
        """Convert meters to kilometers."""
        return self.value / 1000.0


class GpsRecord(SamsaraModelBase):
    """
    Single GPS location record from vehicle stats history.

    Attributes:
        time: Timestamp of GPS reading.
        latitude: GPS latitude.
        longitude: GPS longitude.
        heading_degrees: Compass heading.
        speed_miles_per_hour: Current speed in MPH.
        reverse_geo: Reverse geocoded location description.
        address: Samsara address if within a geofence.
        is_ecu_speed: Whether speed came from ECU (vs GPS-derived).
    """

    time: datetime
    latitude: float
    longitude: float
    heading_degrees: float | None = Field(default=None, alias='headingDegrees')
    speed_miles_per_hour: float | None = Field(default=None, alias='speedMilesPerHour')
    reverse_geo: ReverseGeocode | None = Field(default=None, alias='reverseGeo')
    address: AddressReference | None = None
    is_ecu_speed: bool | None = Field(default=None, alias='isEcuSpeed')

    @property
    def coordinates(self) -> tuple[float, float]:
        """Return (latitude, longitude) tuple."""
        return (self.latitude, self.longitude)

    @property
    def formatted_location(self) -> str | None:
        """Get human-readable location if available."""
        if self.reverse_geo:
            return self.reverse_geo.formatted_location
        return None

    @property
    def is_moving(self) -> bool:
        """Determine if vehicle was moving at this reading."""
        if self.speed_miles_per_hour is not None:
            return self.speed_miles_per_hour > 0
        return False


class VehicleStatsHistoryRecord(SamsaraModelBase):
    """
    Vehicle stats history for a single vehicle.

    Contains arrays of engine states, GPS readings, and odometer readings
    for the requested time period. Each vehicle in the response gets one
    of these records.

    Attributes:
        vehicle_id: Samsara's internal vehicle identifier.
        name: Vehicle display name.
        external_ids: External system identifiers.
        engine_states: List of engine state changes.
        gps: List of GPS readings.
        obd_odometer_meters: List of OBD-II odometer readings (sparse).
    """

    vehicle_id: str = Field(alias='id')
    name: str
    external_ids: dict[str, str] = Field(default_factory=dict, alias='externalIds')
    engine_states: list[EngineStateRecord] = Field(
        default_factory=list,
        alias='engineStates',
    )
    gps: list[GpsRecord] = Field(default_factory=list)
    obd_odometer_meters: list[ObdOdometerRecord] = Field(
        default_factory=list,
        alias='obdOdometerMeters',
    )

    @property
    def has_engine_data(self) -> bool:
        """Check if engine state data was returned."""
        return len(self.engine_states) > 0

    @property
    def has_gps_data(self) -> bool:
        """Check if GPS data was returned."""
        return len(self.gps) > 0

    @property
    def has_odometer_data(self) -> bool:
        """Check if odometer data was returned."""
        return len(self.obd_odometer_meters) > 0

    def get_vin(self) -> str | None:
        """Extract VIN from external IDs if present."""
        return self.external_ids.get('samsara.vin')


# =============================================================================
# Location Stream Models
# =============================================================================


class LocationStreamRecord(SamsaraModelBase):
    """
    Single location/speed record from asset location stream.

    The location stream provides high-frequency GPS updates for assets.

    Attributes:
        happened_at_time: Timestamp of the location reading.
        asset: Reference to the asset/vehicle.
        location: GPS location with accuracy and heading.
    """

    happened_at_time: datetime = Field(alias='happenedAtTime')
    asset: AssetReference
    location: GpsLocation

    @property
    def vehicle_id(self) -> str:
        """Get the vehicle/asset ID."""
        return self.asset.asset_id

    @property
    def coordinates(self) -> tuple[float, float]:
        """Return (latitude, longitude) tuple."""
        return self.location.coordinates


# =============================================================================
# Pagination Model
# =============================================================================


class SamsaraPaginationInfo(SamsaraModelBase):
    """
    Cursor-based pagination metadata from Samsara API.

    Attributes:
        end_cursor: Opaque cursor for fetching next page.
        has_next_page: Whether more data is available.
    """

    end_cursor: str = Field(alias='endCursor')
    has_next_page: bool = Field(alias='hasNextPage')

    @property
    def next_cursor(self) -> str | None:
        """Get cursor for next page, or None if no more pages."""
        if self.has_next_page and self.end_cursor:
            return self.end_cursor
        return None


# =============================================================================
# Full API Response Models
# =============================================================================


class VehiclesResponse(SamsaraModelBase):
    """
    Complete response from GET /fleet/vehicles.

    Attributes:
        data: List of vehicle records.
        pagination: Cursor-based pagination metadata.
    """

    data: list[SamsaraVehicle]
    pagination: SamsaraPaginationInfo | None = None

    def get_items(self) -> list[SamsaraVehicle]:
        """Extract vehicle list (uniform interface method)."""
        return self.data


class DriversResponse(SamsaraModelBase):
    """
    Complete response from GET /fleet/drivers.

    Attributes:
        data: List of driver records.
        pagination: Cursor-based pagination metadata.
    """

    data: list[SamsaraDriver]
    pagination: SamsaraPaginationInfo | None = None

    def get_items(self) -> list[SamsaraDriver]:
        """Extract driver list (uniform interface method)."""
        return self.data


class AddressesResponse(SamsaraModelBase):
    """
    Complete response from GET /addresses.

    Attributes:
        data: List of address records.
        pagination: Cursor-based pagination metadata.
    """

    data: list[SamsaraAddress]
    pagination: SamsaraPaginationInfo | None = None

    def get_items(self) -> list[SamsaraAddress]:
        """Extract address list (uniform interface method)."""
        return self.data


class VehicleStatsHistoryResponse(SamsaraModelBase):
    """
    Complete response from GET /fleet/vehicles/stats/history.

    Attributes:
        data: List of vehicle stats records (one per vehicle).
        pagination: Cursor-based pagination metadata.
    """

    data: list[VehicleStatsHistoryRecord]
    pagination: SamsaraPaginationInfo | None = None

    def get_items(self) -> list[VehicleStatsHistoryRecord]:
        """Extract stats records list (uniform interface method)."""
        return self.data


class LocationStreamResponse(SamsaraModelBase):
    """
    Complete response from GET /assets/location-and-speed/stream.

    Attributes:
        data: List of location stream records.
        pagination: Cursor-based pagination metadata.
    """

    data: list[LocationStreamRecord]
    pagination: SamsaraPaginationInfo | None = None

    def get_items(self) -> list[LocationStreamRecord]:
        """Extract location records list (uniform interface method)."""
        return self.data


class DriverVehicleAssignmentsResponse(SamsaraModelBase):
    """
    Complete response from GET /fleet/driver-vehicle-assignments.

    Attributes:
        data: List of driver-vehicle assignment records.
        pagination: Cursor-based pagination metadata.
    """

    data: list[DriverVehicleAssignment]
    pagination: SamsaraPaginationInfo | None = None

    def get_items(self) -> list[DriverVehicleAssignment]:
        """Extract assignment list (uniform interface method)."""
        return self.data

    def get_assignments_for_vehicle(
        self, vehicle_id: str
    ) -> list[DriverVehicleAssignment]:
        """
        Filter assignments for a specific vehicle.

        Args:
            vehicle_id: Samsara vehicle ID to filter by.

        Returns:
            List of assignments for the specified vehicle.
        """
        return [a for a in self.data if a.vehicle.vehicle_id == vehicle_id]

    def get_assignments_for_driver(
        self, driver_id: str
    ) -> list[DriverVehicleAssignment]:
        """
        Filter assignments for a specific driver.

        Args:
            driver_id: Samsara driver ID to filter by.

        Returns:
            List of assignments for the specified driver.
        """
        return [a for a in self.data if a.driver.driver_id == driver_id]

    def find_driver_at_timestamp(
        self,
        vehicle_id: str,
        timestamp: datetime,
    ) -> str | None:
        """
        Find which driver was operating a vehicle at a specific timestamp.

        Args:
            vehicle_id: Samsara vehicle ID to check.
            timestamp: Timestamp to query.

        Returns:
            Driver name if an assignment covers the timestamp, else None.
        """
        for assignment in self.data:
            if (
                assignment.vehicle.vehicle_id == vehicle_id
                and assignment.contains_timestamp(timestamp)
                and not assignment.is_passenger
            ):
                return assignment.driver.name
        return None
