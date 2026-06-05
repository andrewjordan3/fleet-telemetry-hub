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
from collections import Counter
from datetime import datetime, timedelta
from enum import StrEnum
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


class UserRole(StrEnum):
    """Valid roles for Motive users."""

    DRIVER = 'driver'
    ADMIN = 'admin'
    FLEET_MANAGER = 'fleet_manager'
    SAFETY_MANAGER = 'safety_manager'


class UserStatus(StrEnum):
    """Account status for Motive users."""

    ACTIVE = 'active'
    INACTIVE = 'inactive'
    DEACTIVATED = 'deactivated'


class VehicleStatus(StrEnum):
    """Operational status for vehicles."""

    ACTIVE = 'active'
    INACTIVE = 'inactive'
    DEACTIVATED = 'deactivated'


class AvailabilityStatus(StrEnum):
    """Vehicle availability status."""

    IN_SERVICE = 'in_service'
    OUT_OF_SERVICE = 'out_of_service'


class DutyStatus(StrEnum):
    """Driver's current Hours of Service duty status."""

    OFF_DUTY = 'off_duty'
    SLEEPER = 'sleeper'
    DRIVING = 'driving'
    ON_DUTY = 'on_duty'
    YARD_MOVES = 'yard_moves'
    PERSONAL_CONVEYANCE = 'personal_conveyance'


class EldMode(StrEnum):
    """ELD operational mode."""

    LOGS = 'logs'
    EXEMPT = 'exempt'


class VehicleLocationType(StrEnum):
    """Type of location record."""

    BREADCRUMB = 'breadcrumb'
    VEHICLE_STOPPED = 'vehicle_stopped'
    VEHICLE_MOVING = 'vehicle_moving'
    IGNITION_ON = 'ignition_on'
    IGNITION_OFF = 'ignition_off'
    ENGINE_START = 'engine_start'
    ENGINE_STOP = 'engine_stop'
    GPS_MOVING = 'gps_moving'
    GPS_STOPPED = 'gps_stopped'


class FuelType(StrEnum):
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


class FrozenResponseModelBase(ResponseModelBase):
    """
    Base class for immutable Motive API response models.

    Inherits all behavior from ResponseModelBase (extra='ignore',
    populate_by_name=True, str_strip_whitespace=True) and adds frozen=True
    to make instances hashable and prevent accidental mutation.

    Use this for newly-added response models. Existing models continue to
    inherit from ResponseModelBase to avoid disturbing their callers.
    """

    model_config = ConfigDict(
        extra='ignore',
        populate_by_name=True,
        str_strip_whitespace=True,
        frozen=True,
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
        identifier: Device serial number or hardware ID (e.g., "XXXX99XX999999").
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
        driver_company_id: Company-assigned driver ID (e.g., "12345-WXYZ").
        status: Free-form account status string from the Motive API
            (e.g. ``"active"``, ``"inactive"``, ``"deactivated"``).
            Modeled as ``str`` rather than a constrained enum since
            the API documents this as a plain String type, and the
            unifier does not consume the field.
        role: Free-form user role string (typically ``"driver"`` in
            this context). Modeled as ``str`` rather than a
            constrained enum for the same reason as ``status``.
    """

    driver_id: int = Field(alias='id')
    first_name: str
    last_name: str
    username: str | None = None
    email: str | None = None
    driver_company_id: str | None = None
    status: str | None = None
    role: str | None = None

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


class VehicleSummary(FrozenResponseModelBase):
    """
    Abbreviated vehicle information embedded in other responses.

    This compact representation appears when a vehicle is referenced from
    another entity (e.g., the `vehicle` block on a VehicleUtilization). For
    full vehicle metadata, query the /v1/vehicles endpoint.

    Attributes:
        vehicle_id: Motive's internal vehicle identifier.
        number: Fleet/unit number (user-assigned).
        year: Model year as string (Motive's convention).
        make: Vehicle manufacturer (e.g., "Kenworth").
        model: Vehicle model name.
        vin: Vehicle Identification Number (17 characters).
        metric_units: Whether vehicle reports metric units.
    """

    vehicle_id: int = Field(alias='id')
    number: str
    year: str | None = None
    make: str | None = None
    model: str | None = None
    vin: str | None = None
    metric_units: bool = False


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
    vin: str | None = None
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
            logger.warning('Unknown fuel type encountered: %s', fuel_type_value)
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
        name: Display name (e.g., "Region A - District 1 - Headquarters").
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


class VehicleUtilization(FrozenResponseModelBase):
    """
    Aggregate utilization metrics for a single vehicle over the requested window.

    Returned by /v2/vehicle_utilization. Numeric metrics are pre-calculated by
    Motive and should not be recomputed downstream. Fuel and distance units
    depend on the embedded vehicle's `metric_units` flag.

    Attributes:
        message: Diagnostic string describing a communication gap; normalized
            from Motive's empty-string default to None when the vehicle is
            communicating normally.
        last_located_at: Timezone-aware timestamp of the most recent location
            fix in the window (None if the vehicle never reported).
        utilization_percentage: Percentage of the window the vehicle was
            utilized, pre-calculated by Motive.
        idle_time: Engine-on-but-idle duration in seconds.
        idle_fuel: Fuel consumed while idle (gallons when
            vehicle.metric_units=False, liters when True).
        driving_time: Engine-on-and-moving duration in seconds.
        driving_fuel: Fuel consumed while driving (same units as idle_fuel).
        total_fuel: Total fuel consumed in the window (same units as idle_fuel).
        total_distance: Distance traveled in the window (miles when
            vehicle.metric_units=False, kilometers when True).
        vehicle: Embedded summary of the vehicle this record describes.
    """

    message: str | None = None
    last_located_at: datetime | None = None
    utilization_percentage: float
    idle_time: int
    idle_fuel: float
    driving_time: int
    driving_fuel: float
    total_fuel: float
    total_distance: float
    vehicle: VehicleSummary

    @field_validator('message', mode='before')
    @classmethod
    def normalize_empty_message(cls, message_value: str | None) -> str | None:
        """
        Convert Motive's empty-string default for `message` to None.

        Motive returns `message=""` when the vehicle is communicating normally
        and a populated diagnostic string when there's a communication gap.
        Normalizing the empty case to None means downstream code can use a
        single truthiness check instead of comparing to two sentinel values.
        """
        if message_value is None or message_value == '':
            return None
        return message_value

    @property
    def engine_on_seconds(self) -> int:
        """Total engine-on duration (idle + driving) in seconds."""
        return self.idle_time + self.driving_time

    @property
    def engine_on_hours(self) -> float:
        """Total engine-on duration in hours."""
        return self.engine_on_seconds / 3600

    @property
    def has_communication_issue(self) -> bool:
        """True when Motive reported a diagnostic message for this vehicle."""
        return self.message is not None


class DriverIdleRollup(FrozenResponseModelBase):
    """
    Aggregate idle/driving metrics for a single driver over the requested window.

    Returned by /v2/driver_utilization. Motive emits one record per driver who
    operated a vehicle in the window, plus a single ``driver=None`` bucket
    aggregating activity that could not be attributed to any logged-in driver.

    Durations are reported in seconds (note: this differs from Samsara's
    ``*DurationMs`` fields). Fuel amounts are in whatever unit the vehicle
    reports (gallons when the underlying vehicle's ``metric_units=False``,
    liters when True); the model does not normalize.

    Attributes:
        utilization: Percentage of the window the driver was utilized,
            pre-calculated by Motive (0-100).
        idle_time: Engine-on-but-idle duration in seconds.
        driving_time: Engine-on-and-moving duration in seconds.
        driver: Embedded driver summary, or None for the unattributed-activity
            bucket aggregating periods with no logged-in driver.
        idle_fuel: Fuel consumed while idle.
        driving_fuel: Fuel consumed while driving.
    """

    utilization: float
    idle_time: int
    driving_time: int
    driver: DriverSummary | None = None
    idle_fuel: float
    driving_fuel: float

    @property
    def is_unattributed_bucket(self) -> bool:
        """True for the null-driver bucket aggregating unattributed activity."""
        return self.driver is None

    @property
    def total_engine_seconds(self) -> int:
        """Total engine-on duration (idle + driving) in seconds."""
        return self.idle_time + self.driving_time


class DrivingPeriod(FrozenResponseModelBase):
    """
    One contiguous driving interval for a (driver-or-null, vehicle) pair.

    Returned by /v1/driving_periods. Periods can cross UTC midnight; callers
    aggregating per day must clip to target-day boundaries themselves -- this
    model does not clip.

    The ``distance`` field is a formatted string emitted by Motive (e.g.
    ``"22.3 mi"``) and is not useful for arithmetic. Callers wanting real
    distance should compute it from ``end_kilometers - start_kilometers``
    (also exposed as ``kilometers_traveled``). Both readings can be
    ``null`` when the ELD did not report odometer data; in that case
    ``kilometers_traveled`` propagates ``None`` and the unifier emits
    the row with null ``distance_miles``.

    The HVB (high-voltage battery) fields are EV-only; they are always null
    for fuel vehicles in the current fleet.

    Both ``start_time`` and ``end_time`` are nullable at the model
    layer: Motive emits null on at least one timestamp for ongoing
    or unterminated periods. The unifier-side guarantee that records
    arrive with valid timestamps has not been weakened -- it has
    been relocated one layer up. ``DrivingPeriodsResponse.get_driving_periods()``
    applies the recovery / filter decision and emits a counter-summary
    log line per response, so downstream consumers (including the
    unifier) never see null-timestamp records.

    Attributes:
        period_id: Motive's internal identifier for this driving period.
        start_time: Period start timestamp (timezone-aware UTC).
            Nullable: Motive emits null on ongoing or unterminated
            periods. The wrapper layer recovers from ``end_time -
            duration`` when ``duration`` is present and plausible,
            otherwise drops the record.
        end_time: Period end timestamp (timezone-aware UTC).
            Nullable: same semantics as ``start_time``; recovery
            via ``start_time + duration`` when feasible.
        status: Lifecycle status (e.g., ``"complete"``). Not consumed
            by the unifier; modeled permissively (may be null).
        type: Period classification (e.g., ``"driving"``). Not
            consumed by the unifier; modeled permissively (may be null).
        annotation_status: Integer annotation status code from Motive
            (e.g., ``1``). Not consumed by the unifier; modeled
            permissively (may be null).
        notes: Free-form driver notes attached to the period.
        duration: Period length in seconds as reported by Motive,
            real-valued (the live API has been observed sending
            fractional-second values like ``6420700.43``). Used by
            the wrapper layer to recover a missing ``start_time`` or
            ``end_time`` when the other endpoint is present.
            Otherwise not consumed by the unifier -- duration is
            computed from the recovered timestamps directly. Modeled
            permissively (may be null).
        start_kilometers: Vehicle odometer reading at period start (km).
            Null when the ELD did not report a reading; the unifier
            emits the row with null distance in that case.
        end_kilometers: Vehicle odometer reading at period end (km).
            Null when the ELD did not report a reading; the unifier
            emits the row with null distance in that case.
        source: Numeric source code identifying how the period was
            recorded. May be null.
        driver: Embedded driver summary, or None when no driver was logged in.
        vehicle: Embedded summary of the vehicle that operated the period.
        origin: Reverse-geocoded origin description.
        origin_lat: Origin latitude in decimal degrees.
        origin_lon: Origin longitude in decimal degrees.
        destination: Reverse-geocoded destination description.
        destination_lat: Destination latitude in decimal degrees.
        destination_lon: Destination longitude in decimal degrees.
        distance: Formatted distance string from Motive (not arithmetic-safe).
        start_hvb_state_of_charge: EV battery state of charge at period start.
        end_hvb_state_of_charge: EV battery state of charge at period end.
        start_hvb_lifetime_energy_output: EV battery lifetime energy at start.
        end_hvb_lifetime_energy_output: EV battery lifetime energy at end.
    """

    period_id: int = Field(alias='id')
    start_time: datetime | None = None
    end_time: datetime | None = None
    status: str | None = None
    type: str | None = None
    annotation_status: int | None = None
    notes: str | None = None
    duration: float | None = None
    start_kilometers: float | None = None
    end_kilometers: float | None = None
    source: int | None = None
    driver: DriverSummary | None = None
    vehicle: VehicleSummary
    origin: str | None = None
    origin_lat: float | None = None
    origin_lon: float | None = None
    destination: str | None = None
    destination_lat: float | None = None
    destination_lon: float | None = None
    distance: str | None = None
    start_hvb_state_of_charge: float | None = None
    end_hvb_state_of_charge: float | None = None
    start_hvb_lifetime_energy_output: float | None = None
    end_hvb_lifetime_energy_output: float | None = None

    @property
    def kilometers_traveled(self) -> float | None:
        """
        Odometer-delta distance for this period, in kilometers.

        Returns ``None`` when either ``start_kilometers`` or
        ``end_kilometers`` is null -- the ELD did not report the
        reading and no odometer delta is computable. The unifier
        emits the row with null distance in that case rather than
        dropping it: time, driver, and vehicle are still meaningful
        even when distance is not.
        """
        if self.start_kilometers is None or self.end_kilometers is None:
            return None
        return self.end_kilometers - self.start_kilometers


class IdleEvent(FrozenResponseModelBase):
    """
    Single contiguous idle event from /v1/idle_events.

    Each event captures an interval during which a vehicle was idling.
    Cross-midnight events are possible; callers aggregating per day
    must clip to target-day boundaries themselves -- this model does
    not clip.

    ``veh_fuel_start`` and ``veh_fuel_end`` are ELD-derived cumulative
    fuel readings (like a fuel odometer) -- estimates, not
    authoritative consumption data. The authoritative fuel-data
    pipeline lives outside this repo and reconciles against
    fuel-card transactions separately. ``fuel_consumed`` exposes the
    delta when both endpoints are present. The ``rg_*`` fields are
    Motive-internal reverse-geocode metadata and are preserved
    verbatim without interpretation. ``end_type`` is a free-form
    string code (e.g. ``"vehicle_moving"``) and is modeled as
    ``str`` rather than an enum because the full universe of values
    is not documented.

    Apart from ``event_id``, ``start_time``, ``end_time``,
    ``driver``, and ``vehicle``, the fields below are not consumed
    by the unifier; they are modeled permissively (nullable) so a
    single drifted field cannot kill the whole page during
    validation.

    ``start_time`` and ``end_time`` are also nullable at the model
    layer. Idle events have no ``duration`` field of their own
    (``duration_seconds`` is a derived property from the two
    timestamps), so recovery is not possible. The unifier-side
    guarantee that records arrive with valid timestamps still
    holds, but the boundary that enforces it has moved up:
    ``IdleEventsResponse.get_idle_events()`` drops any record with
    a missing timestamp and emits a counter-summary log line per
    response.

    Attributes:
        event_id: Motive's internal identifier for this idle event.
        start_time: Event start timestamp (timezone-aware UTC).
            Nullable: Motive emits null on at least one timestamp
            for ELD anomalies. The wrapper layer drops any such
            record; downstream consumers never see one.
        end_time: Event end timestamp (timezone-aware UTC).
            Nullable: same semantics as ``start_time``.
        veh_fuel_start: Cumulative ELD-derived fuel reading at event
            start. ELD readings are estimates, not authoritative
            fuel data; the authoritative fuel-data pipeline lives
            outside this repo. Retained for possible future use but
            not consumed by the unifier (may be null).
        veh_fuel_end: Cumulative ELD-derived fuel reading at event
            end. See ``veh_fuel_start`` for caveat (may be null).
        lat: Event latitude in decimal degrees. Not consumed by the
            unifier; may be null.
        lon: Event longitude in decimal degrees. Not consumed by the
            unifier; may be null.
        city: Reverse-geocoded city name. May be null in unmapped
            regions; not consumed by the unifier.
        state: Reverse-geocoded state/province code. May be null in
            unmapped regions; not consumed by the unifier.
        rg_brg: Motive-internal reverse-geocode bearing metric.
            Not consumed by the unifier; may be null.
        rg_km: Motive-internal reverse-geocode distance metric.
            Not consumed by the unifier; may be null.
        rg_match: Motive-internal reverse-geocode match flag.
            Not consumed by the unifier; may be null.
        end_type: Code describing how the idle ended
            (e.g. ``"vehicle_moving"``). Not consumed by the
            unifier; may be null.
        driver: Embedded driver summary, or None when no driver was
            logged in.
        vehicle: Embedded summary of the vehicle that was idling.
        eld_device: Embedded ELD device hardware information.
            Not consumed by the unifier; may be null.
        location: Human-readable location string (typically
            ``"<city>, <state>"``). Not consumed by the unifier;
            may be null.
    """

    event_id: int = Field(alias='id')
    start_time: datetime | None = None
    end_time: datetime | None = None
    veh_fuel_start: float | None = None
    veh_fuel_end: float | None = None
    lat: float | None = None
    lon: float | None = None
    city: str | None = None
    state: str | None = None
    rg_brg: float | None = None
    rg_km: float | None = None
    rg_match: bool | None = None
    end_type: str | None = None
    driver: DriverSummary | None = None
    vehicle: VehicleSummary
    eld_device: EldDeviceInfo | None = None
    location: str | None = None

    @property
    def is_unattributed(self) -> bool:
        """True when the event has no logged-in driver."""
        return self.driver is None

    @property
    def duration_seconds(self) -> float:
        """
        Elapsed event duration in seconds (``end_time - start_time``).

        Raises:
            ValueError: If either ``start_time`` or ``end_time`` is
                ``None``. Idle events have no separate ``duration``
                field for recovery, so a missing timestamp leaves
                the duration genuinely undefined. The wrapper layer
                drops such records before they reach consumers, so
                this raise should be unreachable in normal flow;
                callers that bypass the wrapper and need to handle
                absent timestamps should guard on the attributes
                directly. Mirrors the ``fuel_consumed`` precedent
                on this class.
        """
        if self.start_time is None or self.end_time is None:
            raise ValueError(
                f'duration_seconds requires both start_time and end_time; '
                f'got start_time={self.start_time!r}, end_time={self.end_time!r}'
            )
        return (self.end_time - self.start_time).total_seconds()

    @property
    def fuel_consumed(self) -> float:
        """
        Cumulative-fuel delta over the event (``veh_fuel_end - veh_fuel_start``).

        The result is an ELD-derived estimate, not an authoritative
        consumption figure. The authoritative fuel-data pipeline
        lives outside this repo and reconciles against fuel-card
        transactions separately.

        Raises:
            TypeError: If either ``veh_fuel_start`` or ``veh_fuel_end``
                is ``None``. Callers that need to handle absent
                readings should guard on those attributes directly.
        """
        # Both operands are nullable after the model-strictness audit
        # (see class docstring). The property intentionally surfaces
        # absent readings as a runtime ``TypeError`` rather than
        # silently returning a misleading value; the type-checker
        # cannot model that contract.
        return self.veh_fuel_end - self.veh_fuel_start  # type: ignore[operator]


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


class VehicleUtilizationWrapper(FrozenResponseModelBase):
    """Wrapper for single utilization record in response array."""

    vehicle_utilization: VehicleUtilization


class DriverIdleRollupWrapper(FrozenResponseModelBase):
    """Wrapper for single driver idle rollup record in response array."""

    driver_idle_rollup: DriverIdleRollup


class DrivingPeriodWrapper(FrozenResponseModelBase):
    """Wrapper for single driving period record in response array."""

    driving_period: DrivingPeriod


class IdleEventWrapper(FrozenResponseModelBase):
    """Wrapper for a single idle event in the response array."""

    idle_event: IdleEvent


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


class VehicleUtilizationsResponse(FrozenResponseModelBase):
    """
    Complete response from GET /v2/vehicle_utilization.

    Attributes:
        vehicle_utilizations: List of utilization wrappers (one per vehicle).
        pagination: Pagination metadata.
    """

    vehicle_utilizations: list[VehicleUtilizationWrapper]
    pagination: MotivePaginationInfo

    def get_vehicle_utilizations(self) -> list[VehicleUtilization]:
        """
        Extract unwrapped VehicleUtilization objects from response.

        Returns:
            List of VehicleUtilization objects without wrapper nesting.
        """
        return [wrapper.vehicle_utilization for wrapper in self.vehicle_utilizations]


class DriverUtilizationsResponse(FrozenResponseModelBase):
    """
    Complete response from GET /v2/driver_utilization.

    Attributes:
        driver_idle_rollups: List of rollup wrappers (one per driver, plus
            an optional null-driver bucket aggregating unattributed activity).
        pagination: Pagination metadata.
    """

    driver_idle_rollups: list[DriverIdleRollupWrapper]
    pagination: MotivePaginationInfo

    def get_driver_idle_rollups(self) -> list[DriverIdleRollup]:
        """
        Extract unwrapped DriverIdleRollup objects from response.

        Returns:
            List of DriverIdleRollup objects without wrapper nesting.
        """
        return [wrapper.driver_idle_rollup for wrapper in self.driver_idle_rollups]


# -----------------------------------------------------------------
# Wrapper-layer timestamp recovery / filter for /v1/driving_periods
# and /v1/idle_events
# -----------------------------------------------------------------
#
# DOT Hours-of-Service caps a single driver at 11 hours of driving
# per duty cycle; cross-midnight is normal in long-haul telematics
# but multi-day is not. 24 hours is a conservative threshold for
# "this duration is a plausible driving-period length, recover the
# missing timestamp" vs. "this is the shape of an ongoing or
# unterminated period -- drop it." Domain-derived; not an
# environment-tunable knob.
_MAX_PLAUSIBLE_DRIVING_PERIOD_SECONDS: int = 24 * 3600


class _DrivingPeriodRecoveryOutcome(StrEnum):
    """Per-record outcome categories for driving-period timestamp recovery."""

    KEPT = 'kept'
    RECOVERED = 'recovered'
    DROPPED_IMPLAUSIBLE = 'dropped_implausible'
    DROPPED_UNRECOVERABLE = 'dropped_unrecoverable'


class _IdleEventFilterOutcome(StrEnum):
    """Per-record outcome categories for idle-event timestamp filtering."""

    KEPT = 'kept'
    DROPPED_UNRECOVERABLE = 'dropped_unrecoverable'


def _recover_driving_period_timestamps(
    period: DrivingPeriod,
) -> tuple[DrivingPeriod | None, _DrivingPeriodRecoveryOutcome]:
    """
    Decide whether a ``DrivingPeriod`` survives the wrapper boundary, and how.

    Returns ``(maybe_period, outcome)`` where ``maybe_period`` is
    ``None`` for the two dropped categories and a (possibly
    timestamp-recovered) ``DrivingPeriod`` for ``KEPT`` /
    ``RECOVERED``.

    Decision matrix (see prompt):

    - both timestamps present -> ``KEPT`` (returned unchanged)
    - both null -> ``DROPPED_UNRECOVERABLE``
    - one null, ``duration`` null -> ``DROPPED_UNRECOVERABLE``
    - one null, ``duration > _MAX_PLAUSIBLE_DRIVING_PERIOD_SECONDS``
      -> ``DROPPED_IMPLAUSIBLE``
    - one null, ``duration`` present and within threshold ->
      ``RECOVERED`` (missing endpoint computed from the other +/-
      ``timedelta(seconds=duration)``)
    """
    start_time = period.start_time
    end_time = period.end_time
    if start_time is not None and end_time is not None:
        return period, _DrivingPeriodRecoveryOutcome.KEPT
    if start_time is None and end_time is None:
        return None, _DrivingPeriodRecoveryOutcome.DROPPED_UNRECOVERABLE
    duration_seconds = period.duration
    if duration_seconds is None:
        return None, _DrivingPeriodRecoveryOutcome.DROPPED_UNRECOVERABLE
    if duration_seconds > _MAX_PLAUSIBLE_DRIVING_PERIOD_SECONDS:
        return None, _DrivingPeriodRecoveryOutcome.DROPPED_IMPLAUSIBLE
    delta = timedelta(seconds=duration_seconds)
    if end_time is None:
        # ``start_time`` is non-null (the both-null branch above
        # returned). Assert for the type-checker -- the narrowing
        # cannot be expressed structurally.
        assert start_time is not None
        recovered = period.model_copy(update={'end_time': start_time + delta})
    else:
        # ``end_time`` is non-null; recover ``start_time``.
        recovered = period.model_copy(update={'start_time': end_time - delta})
    return recovered, _DrivingPeriodRecoveryOutcome.RECOVERED


def _filter_idle_event_timestamps(
    event: IdleEvent,
) -> tuple[IdleEvent | None, _IdleEventFilterOutcome]:
    """
    Decide whether an ``IdleEvent`` survives the wrapper boundary.

    Idle events have no separate ``duration`` field that could be
    used to reconstruct a missing timestamp, so any null in
    ``start_time`` or ``end_time`` is unrecoverable.

    Returns ``(maybe_event, outcome)``; ``maybe_event`` is ``None``
    for the dropped category.
    """
    if event.start_time is None or event.end_time is None:
        return None, _IdleEventFilterOutcome.DROPPED_UNRECOVERABLE
    return event, _IdleEventFilterOutcome.KEPT


class DrivingPeriodsResponse(FrozenResponseModelBase):
    """
    Complete response from GET /v1/driving_periods.

    Attributes:
        driving_periods: List of period wrappers (one per (driver-or-null,
            vehicle, time-window) record).
        pagination: Pagination metadata.
    """

    driving_periods: list[DrivingPeriodWrapper]
    pagination: MotivePaginationInfo

    def get_driving_periods(self) -> list[DrivingPeriod]:
        """
        Extract unwrapped ``DrivingPeriod`` objects, applying timestamp recovery.

        Records with both timestamps present are returned as-is.
        Records missing exactly one timestamp are recovered via
        ``duration`` when it is present and within the plausibility
        threshold; otherwise dropped. Records missing both
        timestamps are always dropped. See
        ``_recover_driving_period_timestamps`` for the full
        decision matrix.

        A single ``WARNING`` log line is emitted per call iff any
        record was recovered or dropped; the line carries the full
        per-outcome breakdown so operators can spot drift in a
        single grep.

        Returns:
            List of ``DrivingPeriod`` objects, all with
            non-null ``start_time`` and ``end_time``.
        """
        outcomes: Counter[_DrivingPeriodRecoveryOutcome] = Counter()
        recovered_periods: list[DrivingPeriod] = []
        for wrapper in self.driving_periods:
            recovered_period, recovery_outcome = _recover_driving_period_timestamps(
                wrapper.driving_period
            )
            outcomes[recovery_outcome] += 1
            if recovered_period is not None:
                recovered_periods.append(recovered_period)
        # Fill in any outcomes that did not occur so the log line is
        # shape-stable regardless of input distribution.
        for outcome_key in _DrivingPeriodRecoveryOutcome:
            outcomes.setdefault(outcome_key, 0)
        if any(
            outcomes[key] > 0
            for key in _DrivingPeriodRecoveryOutcome
            if key is not _DrivingPeriodRecoveryOutcome.KEPT
        ):
            logger.warning(
                'Driving-period unwrap: %d kept, %d recovered, '
                '%d dropped (implausible), %d dropped (unrecoverable)',
                outcomes[_DrivingPeriodRecoveryOutcome.KEPT],
                outcomes[_DrivingPeriodRecoveryOutcome.RECOVERED],
                outcomes[_DrivingPeriodRecoveryOutcome.DROPPED_IMPLAUSIBLE],
                outcomes[_DrivingPeriodRecoveryOutcome.DROPPED_UNRECOVERABLE],
            )
        return recovered_periods


class IdleEventsResponse(FrozenResponseModelBase):
    """
    Complete response from GET /v1/idle_events.

    Attributes:
        idle_events: List of idle event wrappers (one per idling
            interval; cross-midnight events are not clipped).
        pagination: Pagination metadata.
    """

    idle_events: list[IdleEventWrapper]
    pagination: MotivePaginationInfo

    def get_idle_events(self) -> list[IdleEvent]:
        """
        Extract unwrapped ``IdleEvent`` objects, filtering out null timestamps.

        Idle events have no separate ``duration`` field for
        recovery, so any record with a null ``start_time`` or
        ``end_time`` is dropped. A single ``WARNING`` log line is
        emitted per call iff any record was dropped.

        Returns:
            List of ``IdleEvent`` objects, all with non-null
            ``start_time`` and ``end_time``.
        """
        outcomes: Counter[_IdleEventFilterOutcome] = Counter()
        filtered_events: list[IdleEvent] = []
        for wrapper in self.idle_events:
            filtered_event, filter_outcome = _filter_idle_event_timestamps(
                wrapper.idle_event
            )
            outcomes[filter_outcome] += 1
            if filtered_event is not None:
                filtered_events.append(filtered_event)
        for outcome_key in _IdleEventFilterOutcome:
            outcomes.setdefault(outcome_key, 0)
        if outcomes[_IdleEventFilterOutcome.DROPPED_UNRECOVERABLE] > 0:
            logger.warning(
                'Idle-event unwrap: %d kept, %d dropped (unrecoverable)',
                outcomes[_IdleEventFilterOutcome.KEPT],
                outcomes[_IdleEventFilterOutcome.DROPPED_UNRECOVERABLE],
            )
        return filtered_events
