# fleet_telemetry_hub/models/motive_responses.py
"""
Pydantic response models for Motive API data structures.

This module defines typed models for parsing Motive API responses. Models are
organized hierarchically from embedded/shared objects up to full response
containers.

Design Notes:
    - Motive uses double-nesting: {"vehicles": [{"vehicle": {...}}, ...]}
    - Many fields are nullable (API returns null for unset values)
    - Some integer fields use -1 as "not configured" sentinel
    - All timestamps are ISO-8601 UTC format
    - Response models use extra='ignore' to handle API additions gracefully
"""

# pyright: reportUnknownVariableType=false
import logging
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations for Constrained String Fields
# =============================================================================


class UserRole(str, Enum):
    """Valid roles for Motive users."""

    DRIVER = 'driver'
    ADMIN = 'admin'
    FLEET_MANAGER = 'fleet_manager'
    SAFETY_MANAGER = 'safety_manager'


class UserStatus(str, Enum):
    """Account status for Motive users."""

    ACTIVE = 'active'
    INACTIVE = 'inactive'
    DEACTIVATED = 'deactivated'


class VehicleStatus(str, Enum):
    """Operational status for vehicles."""

    ACTIVE = 'active'
    INACTIVE = 'inactive'
    DEACTIVATED = 'deactivated'


class AvailabilityStatus(str, Enum):
    """Vehicle availability status."""

    IN_SERVICE = 'in_service'
    OUT_OF_SERVICE = 'out_of_service'


class DutyStatus(str, Enum):
    """Driver's current Hours of Service duty status."""

    OFF_DUTY = 'off_duty'
    SLEEPER = 'sleeper'
    DRIVING = 'driving'
    ON_DUTY = 'on_duty'
    YARD_MOVES = 'yard_moves'
    PERSONAL_CONVEYANCE = 'personal_conveyance'


class EldMode(str, Enum):
    """ELD operational mode."""

    LOGS = 'logs'
    EXEMPT = 'exempt'


class VehicleLocationType(str, Enum):
    """Type of location record."""

    BREADCRUMB = 'breadcrumb'
    VEHICLE_STOPPED = 'vehicle_stopped'
    VEHICLE_MOVING = 'vehicle_moving'
    IGNITION_ON = 'ignition_on'
    IGNITION_OFF = 'ignition_off'


class FuelType(str, Enum):
    """Vehicle fuel types."""

    DIESEL = 'diesel'
    GASOLINE = 'gasoline'
    ELECTRIC = 'electric'
    HYBRID = 'hybrid'
    CNG = 'cng'  # Compressed Natural Gas
    LNG = 'lng'  # Liquefied Natural Gas
    PROPANE = 'propane'


# =============================================================================
# Base Configuration for Response Models
# =============================================================================


class ResponseModelBase(BaseModel):
    """
    Base class for all Motive API response models.

    Configuration:
        - extra='ignore': Silently ignore unknown fields from API responses.
          This prevents breakage when Motive adds new fields.
        - populate_by_name=True: Allow initialization by field name OR alias.
        - str_strip_whitespace=True: Trim whitespace from string fields.
    """

    model_config = ConfigDict(
        extra='ignore',
        populate_by_name=True,
        str_strip_whitespace=True,
    )


# =============================================================================
# Embedded/Shared Models (Used Across Multiple Endpoints)
# =============================================================================


class EldDeviceInfo(ResponseModelBase):
    """
    ELD (Electronic Logging Device) hardware information.

    This embedded object appears in Vehicle and VehicleLocation responses,
    identifying the physical telematics device installed in the vehicle.

    Attributes:
        device_id: Motive's internal device identifier.
        identifier: Device serial number or hardware ID (e.g., "AABL36SE164048").
        model: Device model name (e.g., "lbb-3.6ca" for Motive's LBB device).
    """

    device_id: int = Field(alias='id')
    identifier: str
    model: str


class DriverSummary(ResponseModelBase):
    """
    Abbreviated driver information embedded in other responses.

    This compact representation appears when a driver is referenced from
    another entity (e.g., current_driver on a Vehicle). For full driver
    details, query the /v1/users endpoint.

    Attributes:
        driver_id: Motive's internal driver identifier.
        first_name: Driver's first name.
        last_name: Driver's last name.
        username: Login username (may be null if not set).
        email: Driver's email address.
        driver_company_id: Company-assigned driver ID (e.g., "34311-AFSR").
        status: Account status (active, inactive, deactivated).
        role: User role (typically "driver" in this context).
    """

    driver_id: int = Field(alias='id')
    first_name: str
    last_name: str
    username: str | None = None
    email: str | None = None
    driver_company_id: str | None = None
    status: UserStatus
    role: UserRole

    @property
    def full_name(self) -> str:
        """Return driver's full name as 'First Last'."""
        return f'{self.first_name} {self.last_name}'


class AvailabilityDetails(ResponseModelBase):
    """
    Vehicle availability status information.

    Tracks whether a vehicle is in-service or out-of-service, along with
    when and by whom the status was last updated.

    Attributes:
        availability_status: Current availability (in_service, out_of_service).
        updated_at: Timestamp of last status change.
        updated_by_user: User who changed the status (null if system-set).
    """

    availability_status: AvailabilityStatus
    updated_at: datetime
    updated_by_user: dict[str, Any] | None = (
        None  # Nested user object, rarely populated
    )


class GroupUserSummary(ResponseModelBase):
    """
    User summary embedded in Group responses.

    Represents the manager or owner of a group/organizational unit.

    Attributes:
        user_id: Motive's internal user identifier.
        first_name: User's first name.
        last_name: User's last name.
        username: Login username (may be null).
        email: User's email address (may be null).
        driver_company_id: Company-assigned ID (null for non-drivers).
        status: Account status.
        role: User role (typically "admin" for group owners).
    """

    user_id: int = Field(alias='id')
    first_name: str
    last_name: str
    username: str | None = None
    email: str | None = None
    driver_company_id: str | None = None
    status: UserStatus
    role: UserRole


# =============================================================================
# Primary Entity Models
# =============================================================================


class Vehicle(ResponseModelBase):
    """
    Complete vehicle record from Motive.

    Represents a single vehicle in the fleet with all associated metadata
    including current driver assignment, ELD device info, and availability.

    Attributes:
        vehicle_id: Motive's internal vehicle identifier.
        company_id: Parent company identifier in Motive.
        number: Fleet number/unit number (user-assigned).
        status: Vehicle operational status.
        ifta: Whether vehicle is IFTA-reportable.
        vin: Vehicle Identification Number (17 characters).
        make: Vehicle manufacturer (e.g., "Kenworth").
        model: Vehicle model name (e.g., "AF Tanker").
        year: Model year as string.
        license_plate_state: State/province of registration.
        license_plate_number: License plate number.
        metric_units: Whether vehicle displays metric units.
        fuel_type: Primary fuel type.
        prevent_auto_odometer_entry: Disable automatic odometer capture.
        notes: Free-form notes field.
        group_ids: List of group IDs this vehicle belongs to.
        created_at: When vehicle was added to Motive.
        updated_at: Last modification timestamp.
        permanent_driver: Permanently assigned driver (if any).
        availability_details: Current availability status.
        eld_device: Installed ELD hardware info.
        current_driver: Currently logged-in driver (if any).
        external_ids: External system identifiers.
    """

    vehicle_id: int = Field(alias='id')
    company_id: int
    number: str  # Fleet/unit number
    status: VehicleStatus
    ifta: bool
    vin: str
    make: str | None = None
    model: str | None = None
    year: str | None = None
    license_plate_state: str | None = None
    license_plate_number: str | None = None
    metric_units: bool = False
    fuel_type: FuelType | None = None
    prevent_auto_odometer_entry: bool = False
    notes: str | None = None

    # Sentinel value handling: Motive uses -1 for "not configured"
    incab_alert_live_stream_enable: int = -1
    driver_facing_camera: int = -1
    incab_audio_recording: int = -1

    group_ids: list[int] = Field(default_factory=list)

    created_at: datetime
    updated_at: datetime

    permanent_driver: DriverSummary | None = None
    availability_details: AvailabilityDetails | None = None
    eld_device: EldDeviceInfo | None = None
    current_driver: DriverSummary | None = None

    external_ids: list[dict[str, Any]] = Field(default_factory=list)

    # CARB (California Air Resources Board) compliance fields
    carb_ctc_test_enabled: bool | None = None
    carb_ctc_emission_status: str | None = None
    registration_expiry_date: str | None = None

    @field_validator('fuel_type', mode='before')
    @classmethod
    def normalize_fuel_type(cls, fuel_type_value: str | None) -> FuelType | None:
        """
        Handle case-insensitive fuel type matching.

        Args:
            fuel_type_value: Raw fuel type string from API.

        Returns:
            Normalized FuelType enum or None if not provided.
        """
        if fuel_type_value is None:
            return None

        try:
            return FuelType(fuel_type_value.lower())
        except ValueError:
            logger.warning(f'Unknown fuel type encountered: {fuel_type_value}')
            return None

    @property
    def has_current_driver(self) -> bool:
        """Check if a driver is currently assigned to this vehicle."""
        return self.current_driver is not None

    @property
    def is_active(self) -> bool:
        """Check if vehicle is in active status."""
        return self.status == VehicleStatus.ACTIVE


class VehicleLocation(ResponseModelBase):
    """
    Single location record (breadcrumb) from vehicle telemetry.

    Represents a point-in-time snapshot of vehicle position, speed, and
    engine metrics. The /v3/vehicle_locations endpoint returns arrays
    of these records for historical analysis.

    Attributes:
        location_id: Unique identifier for this location record (UUID string).
        located_at: Timestamp when location was recorded.
        latitude: GPS latitude in decimal degrees.
        longitude: GPS longitude in decimal degrees.
        location_type: Type of location event (breadcrumb, stopped, etc.).
        description: Human-readable location description (city, state).
        speed: Vehicle speed (mph or km/h based on vehicle's metric_units).
        bearing: Compass heading in degrees (0-360).
        battery_voltage: Vehicle battery voltage.
        odometer: Calculated odometer reading (miles or km).
        true_odometer: ECM-reported odometer if available.
        engine_hours: Total engine run time in hours.
        true_engine_hours: ECM-reported engine hours if available.
        fuel: Cumulative fuel consumption.
        fuel_primary_remaining_percentage: Primary fuel tank level (0-100).
        fuel_secondary_remaining_percentage: Secondary tank level (0-100).
        driver: Driver logged in at time of location capture.
        eld_device: ELD device that captured this location.
    """

    location_id: str = Field(alias='id')  # UUID string
    located_at: datetime
    latitude: float = Field(alias='lat')
    longitude: float = Field(alias='lon')
    location_type: VehicleLocationType = Field(alias='type')
    description: str | None = None

    # Motion metrics
    speed: float | None = None
    bearing: float | None = None

    # Vehicle metrics
    battery_voltage: float | None = None
    odometer: float | None = None
    true_odometer: float | None = None
    engine_hours: float | None = None
    true_engine_hours: float | None = None

    # Fuel metrics
    fuel: float | None = None
    fuel_primary_remaining_percentage: float | None = None
    fuel_secondary_remaining_percentage: float | None = None

    # Electric vehicle fields (null for diesel/gas vehicles)
    veh_range: float | None = None
    hvb_state_of_charge: float | None = None
    hvb_charge_status: str | None = None
    hvb_charge_source: str | None = None
    hvb_lifetime_energy_output: float | None = None

    # Related entities
    driver: DriverSummary | None = None
    eld_device: EldDeviceInfo | None = None

    @property
    def coordinates(self) -> tuple[float, float]:
        """Return (latitude, longitude) tuple for GIS operations."""
        return (self.latitude, self.longitude)

    @property
    def is_moving(self) -> bool:
        """Determine if vehicle was moving at this location."""
        if self.speed is not None:
            return self.speed > 0
        return self.location_type not in (
            VehicleLocationType.VEHICLE_STOPPED,
            VehicleLocationType.IGNITION_OFF,
        )


class Group(ResponseModelBase):
    """
    Organizational group/unit within the company.

    Groups form a hierarchy for organizing vehicles and drivers by region,
    division, or other business structure. Vehicles and users can belong
    to multiple groups.

    Attributes:
        group_id: Motive's internal group identifier.
        name: Display name (e.g., "D1 - R1 - Chicago South").
        company_id: Parent company identifier.
        parent_id: Parent group ID for hierarchy (null if top-level).
        user: Manager/owner of this group.
    """

    group_id: int = Field(alias='id')
    name: str
    company_id: int
    parent_id: int | None = None
    user: GroupUserSummary | None = None

    @property
    def is_top_level(self) -> bool:
        """Check if this group has no parent (root level)."""
        return self.parent_id is None


class User(ResponseModelBase):
    """
    Complete user record from Motive (drivers and administrative users).

    Contains full profile information including HOS (Hours of Service)
    settings, terminal information, and activity timestamps.

    Attributes:
        user_id: Motive's internal user identifier.
        email: User's email address.
        first_name: First name.
        last_name: Last name.
        username: Login username.
        driver_company_id: Company-assigned driver ID.
        phone: Phone number.
        phone_country_code: Country code (e.g., "+1").
        phone_ext: Phone extension.
        time_zone: User's time zone for display purposes.
        metric_units: Whether user prefers metric units.
        role: User role (driver, admin, etc.).
        status: Account status.
        duty_status: Current HOS duty status (drivers only).
        eld_mode: ELD operational mode.
        group_ids: Groups this user belongs to.
        drivers_license_number: CDL number.
        drivers_license_state: CDL issuing state.
        cycle: HOS cycle rule (e.g., "70_8_2020").
        created_at: Account creation timestamp.
        updated_at: Last modification timestamp.
    """

    user_id: int = Field(alias='id')
    email: str | None = None
    first_name: str
    last_name: str
    username: str | None = None
    driver_company_id: str | None = None

    # Contact info
    phone: str | None = None
    phone_country_code: str | None = None
    phone_ext: str | None = None

    # Preferences
    time_zone: str | None = None
    metric_units: bool = False

    # Role and status
    role: UserRole
    status: UserStatus
    duty_status: DutyStatus | None = None
    eld_mode: EldMode | None = None

    # Organization
    group_ids: list[int] = Field(default_factory=list)
    company_reference_id: str | None = None

    # License info
    drivers_license_number: str | None = None
    drivers_license_state: str | None = None

    # Carrier info (company headquarters)
    carrier_name: str | None = None
    carrier_street: str | None = None
    carrier_city: str | None = None
    carrier_state: str | None = None
    carrier_zip: str | None = None

    # Terminal info (driver's home base)
    terminal_street: str | None = None
    terminal_city: str | None = None
    terminal_state: str | None = None
    terminal_zip: str | None = None

    # HOS cycle and exceptions
    cycle: str | None = None
    cycle2: str | None = None
    exception_24_hour_restart: bool = False
    exception_8_hour_break: bool = False
    exception_wait_time: bool = False
    exception_short_haul: bool = False
    exception_ca_farm_school_bus: bool = False
    exception_adverse_driving: bool = False

    # Secondary cycle exceptions (for split operations)
    exception_24_hour_restart2: bool = False
    exception_8_hour_break2: bool = False
    exception_wait_time2: bool = False
    exception_short_haul2: bool = False
    exception_ca_farm_school_bus2: bool = False
    exception_adverse_driving2: bool = False

    # Export preferences
    export_combined: bool = True
    export_recap: bool = True
    export_odometers: bool = True
    minute_logs: bool = True

    # Feature flags
    yard_moves_enabled: bool = False
    personal_conveyance_enabled: bool = False
    manual_driving_enabled: bool = False

    # Violation settings
    violation_alerts: str | None = None

    # Activity timestamps
    mobile_last_active_at: datetime | None = None
    mobile_current_sign_in_at: datetime | None = None
    mobile_last_sign_in_at: datetime | None = None
    web_last_active_at: datetime | None = None
    web_current_sign_in_at: datetime | None = None
    web_last_sign_in_at: datetime | None = None

    created_at: datetime
    updated_at: datetime

    external_ids: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def full_name(self) -> str:
        """Return user's full name as 'First Last'."""
        return f'{self.first_name} {self.last_name}'

    @property
    def is_driver(self) -> bool:
        """Check if user has driver role."""
        return self.role == UserRole.DRIVER

    @property
    def terminal_location(self) -> str | None:
        """Return formatted terminal address if available."""
        if not self.terminal_city or not self.terminal_state:
            return None
        return f'{self.terminal_city}, {self.terminal_state}'


# =============================================================================
# Wrapper Models for API Response Unpacking
# =============================================================================
# Motive uses double-nesting: {"vehicles": [{"vehicle": {...}}, ...]}
# These wrappers handle the inner nesting layer.


class VehicleWrapper(ResponseModelBase):
    """Wrapper for single vehicle in response array."""

    vehicle: Vehicle


class VehicleLocationWrapper(ResponseModelBase):
    """Wrapper for single location in response array."""

    vehicle_location: VehicleLocation


class GroupWrapper(ResponseModelBase):
    """Wrapper for single group in response array."""

    group: Group


class UserWrapper(ResponseModelBase):
    """Wrapper for single user in response array."""

    user: User


# =============================================================================
# Full API Response Models (with Pagination)
# =============================================================================


class MotivePaginationInfo(ResponseModelBase):
    """
    Pagination metadata from Motive API responses.

    Attributes:
        per_page: Number of results per page.
        page_no: Current page number (1-indexed).
        total: Total number of records across all pages.
    """

    per_page: int
    page_no: int
    total: int

    @property
    def total_pages(self) -> int:
        """Calculate total number of pages."""
        if self.per_page == 0:
            return 0
        return (self.total + self.per_page - 1) // self.per_page

    @property
    def has_next_page(self) -> bool:
        """Check if more pages are available."""
        return self.page_no < self.total_pages

    @property
    def next_page_number(self) -> int | None:
        """Get next page number, or None if on last page."""
        if self.has_next_page:
            return self.page_no + 1
        return None


class VehiclesResponse(ResponseModelBase):
    """
    Complete response from GET /v1/vehicles.

    Attributes:
        vehicles: List of vehicle wrappers.
        pagination: Pagination metadata.
    """

    vehicles: list[VehicleWrapper]
    pagination: MotivePaginationInfo

    def get_vehicles(self) -> list[Vehicle]:
        """
        Extract unwrapped Vehicle objects from response.

        Returns:
            List of Vehicle objects without wrapper nesting.
        """
        return [wrapper.vehicle for wrapper in self.vehicles]


class VehicleLocationsResponse(ResponseModelBase):
    """
    Complete response from GET /v3/vehicle_locations/{id}.

    Note: This endpoint is NOT paginated.

    Attributes:
        vehicle_locations: List of location wrappers.
    """

    vehicle_locations: list[VehicleLocationWrapper]

    def get_locations(self) -> list[VehicleLocation]:
        """
        Extract unwrapped VehicleLocation objects from response.

        Returns:
            List of VehicleLocation objects without wrapper nesting.
        """
        return [wrapper.vehicle_location for wrapper in self.vehicle_locations]


class GroupsResponse(ResponseModelBase):
    """
    Complete response from GET /v1/groups.

    Attributes:
        groups: List of group wrappers.
        pagination: Pagination metadata.
    """

    groups: list[GroupWrapper]
    pagination: MotivePaginationInfo

    def get_groups(self) -> list[Group]:
        """
        Extract unwrapped Group objects from response.

        Returns:
            List of Group objects without wrapper nesting.
        """
        return [wrapper.group for wrapper in self.groups]


class UsersResponse(ResponseModelBase):
    """
    Complete response from GET /v1/users.

    Attributes:
        users: List of user wrappers.
        pagination: Pagination metadata.
    """

    users: list[UserWrapper]
    pagination: MotivePaginationInfo

    def get_users(self) -> list[User]:
        """
        Extract unwrapped User objects from response.

        Returns:
            List of User objects without wrapper nesting.
        """
        return [wrapper.user for wrapper in self.users]
