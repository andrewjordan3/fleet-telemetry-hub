"""Tests for ``unifier.unify.unify``: the per-provider orchestration boundary.

Mixes unit tests of the orchestrator's sort/log/schema invariants
with integration tests that drive real Motive and Samsara bundles
through the per-provider transforms and into the final DataFrame.

Fixture helpers are local to this file rather than imported from
the per-provider test modules: those helpers are underscore-private
to their files, and the integration scenarios here need only a
slimmer subset.
"""

import logging
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from fleet_telemetry_hub.models.motive_responses import (
    DriverSummary,
    DrivingPeriod,
    EldDeviceInfo,
    IdleEvent,
    VehicleSummary,
)
from fleet_telemetry_hub.models.samsara_responses import (
    IdlingEvent,
    SamsaraDriver,
    SamsaraVehicle,
    Trip,
)
from fleet_telemetry_hub.unifier.schema import COLUMNS, DTYPES
from fleet_telemetry_hub.unifier.unify import unify
from fleet_telemetry_hub.utilization.motive_fetcher import MotiveUtilizationBundle
from fleet_telemetry_hub.utilization.samsara_fetcher import SamsaraUtilizationBundle
from fleet_telemetry_hub.utilization.vehicle_trip import VehicleTrip

_MAY_14 = date(2026, 5, 14)
_MAY_20 = date(2026, 5, 20)

# Conversion landmarks pinned for the cross-provider distance
# assertions: 16.09 km == 10.0 mi (Motive); 1609 m == 1.0 mi (Samsara).
_MOTIVE_DEFAULT_MILES = 10.0
_SAMSARA_DEFAULT_MILES = 1.0


def _at(hour: int, minute: int = 0) -> datetime:
    """Build a tz-aware UTC datetime on 2026-05-14."""
    return datetime(2026, 5, 14, hour, minute, 0, tzinfo=UTC)


# -----------------------------------------------------------------------------
# Motive fixture helpers
# -----------------------------------------------------------------------------

_MOTIVE_VEHICLE_ID = 8000001
_MOTIVE_VIN = 'TESTVIN0000000100'
_MOTIVE_DRIVER_ID = 9000001


def _make_motive_vehicle(
    *, vehicle_id: int = _MOTIVE_VEHICLE_ID, vin: str | None = _MOTIVE_VIN
) -> VehicleSummary:
    return VehicleSummary.model_validate(
        {
            'id': vehicle_id,
            'number': f'TEST-{vehicle_id}',
            'year': '2020',
            'make': 'TestMake',
            'model': 'TestModel',
            'vin': vin,
            'metric_units': False,
        }
    )


def _make_motive_driver(
    *,
    driver_id: int = _MOTIVE_DRIVER_ID,
    first_name: str = 'Sam',
    last_name: str = 'Snowflake',
) -> DriverSummary:
    return DriverSummary.model_validate(
        {
            'id': driver_id,
            'first_name': first_name,
            'last_name': last_name,
            'username': None,
            'email': None,
            'driver_company_id': None,
            'status': 'active',
            'role': 'driver',
        }
    )


_DEFAULT_MOTIVE_DRIVER = _make_motive_driver()


def _make_motive_driving_period(  # noqa: PLR0913 -- test factory; one knob per field
    *,
    period_id: int = 4550000001,
    vehicle: VehicleSummary | None = None,
    driver: DriverSummary | None = _DEFAULT_MOTIVE_DRIVER,
    start: datetime | None = None,
    end: datetime | None = None,
    distance_km: float = 16.09,
) -> DrivingPeriod:
    if vehicle is None:
        vehicle = _make_motive_vehicle()
    if start is None:
        start = _at(hour=8)
    if end is None:
        end = _at(hour=9)
    return DrivingPeriod.model_validate(
        {
            'id': period_id,
            'start_time': start,
            'end_time': end,
            'status': 'complete',
            'type': 'driving',
            'duration': int((end - start).total_seconds()),
            'start_kilometers': 100.0,
            'end_kilometers': 100.0 + distance_km,
            'source': 1,
            'driver': (
                driver.model_dump(by_alias=True) if driver is not None else None
            ),
            'vehicle': vehicle.model_dump(by_alias=True),
        }
    )


def _make_motive_idle_event(
    *,
    event_id: int = 4860000001,
    vehicle: VehicleSummary | None = None,
    driver: DriverSummary | None = _DEFAULT_MOTIVE_DRIVER,
    start: datetime | None = None,
    end: datetime | None = None,
) -> IdleEvent:
    if vehicle is None:
        vehicle = _make_motive_vehicle()
    if start is None:
        start = _at(hour=10)
    if end is None:
        end = _at(hour=10, minute=30)
    return IdleEvent.model_validate(
        {
            'id': event_id,
            'start_time': start,
            'end_time': end,
            'veh_fuel_start': 100.0,
            'veh_fuel_end': 100.5,
            'lat': 30.0,
            'lon': -90.0,
            'city': 'Testville',
            'state': 'TX',
            'rg_brg': 62.0,
            'rg_km': 1.6,
            'rg_match': True,
            'end_type': 'vehicle_moving',
            'driver': (
                driver.model_dump(by_alias=True) if driver is not None else None
            ),
            'vehicle': vehicle.model_dump(by_alias=True),
            'eld_device': EldDeviceInfo.model_validate(
                {'id': 8800001, 'identifier': 'TESTELD0001', 'model': 'lbb-3.6ca'}
            ).model_dump(by_alias=True),
            'location': 'Testville, TX',
        }
    )


def _make_motive_bundle(
    *,
    driving_periods: list[DrivingPeriod] | None = None,
    idle_events: list[IdleEvent] | None = None,
    company: str | None = 'motive_co',
    date_range: tuple[date, date] = (_MAY_14, _MAY_20),
) -> MotiveUtilizationBundle:
    return MotiveUtilizationBundle(
        vehicle_utilizations_by_date={},
        driver_idle_rollups_by_date={},
        driving_periods=driving_periods or [],
        idle_events=idle_events or [],
        date_range=date_range,
        company=company,
    )


# -----------------------------------------------------------------------------
# Samsara fixture helpers
# -----------------------------------------------------------------------------

_SAMSARA_VEHICLE_ID = '999999900000001'
_SAMSARA_VIN = 'TESTVIN0000000001'
_SAMSARA_DRIVER_ID = '1000001'
_SAMSARA_DRIVER_NAME = 'Suzy Snowflake'


def _make_samsara_vehicle(
    *,
    vehicle_id: str = _SAMSARA_VEHICLE_ID,
    vin: str | None = _SAMSARA_VIN,
    name: str = 'TEST-001',
) -> SamsaraVehicle:
    return SamsaraVehicle.model_validate(
        {'id': vehicle_id, 'name': name, 'vin': vin}
    )


def _make_samsara_driver(
    *,
    driver_id: str = _SAMSARA_DRIVER_ID,
    name: str = _SAMSARA_DRIVER_NAME,
) -> SamsaraDriver:
    return SamsaraDriver.model_validate(
        {'id': driver_id, 'name': name, 'driverActivationStatus': 'active'}
    )


def _make_trip(  # noqa: PLR0913 -- test factory; one knob per field
    *,
    trip_id: str = '00000000-0000-0000-0000-000000001001',
    vehicle_id: str = _SAMSARA_VEHICLE_ID,
    driver_id: str | None = _SAMSARA_DRIVER_ID,
    start: datetime | None = None,
    end: datetime | None = None,
    distance_meters: int = 1609,
) -> VehicleTrip:
    """Build a ``VehicleTrip`` -- the wrapper the Samsara bundle's trips list now holds."""
    if start is None:
        start = _at(hour=8)
    if end is None:
        end = _at(hour=9)
    trip = Trip.model_validate(
        {
            'id': trip_id,
            'driverId': driver_id,
            'startMs': int(start.timestamp() * 1000),
            'endMs': int(end.timestamp() * 1000),
            'distanceMeters': distance_meters,
        }
    )
    return VehicleTrip.from_trip(trip, vehicle_id)


def _make_idling_event(
    *,
    event_uuid: str = '00000000-0000-0000-0000-000000002001',
    vehicle_id: str = _SAMSARA_VEHICLE_ID,
    operator_id: str = _SAMSARA_DRIVER_ID,
    start: datetime | None = None,
    duration_ms: int = 30 * 60 * 1000,
) -> IdlingEvent:
    if start is None:
        start = _at(hour=10)
    return IdlingEvent.model_validate(
        {
            'asset': {'id': vehicle_id},
            'durationMilliseconds': duration_ms,
            'eventUuid': event_uuid,
            'fuelConsumedMilliliters': 0.5,
            'fuelCost': {'amount': '0.66', 'currency': 'usd'},
            'gaseousFuelConsumedGrams': 0,
            'gaseousFuelCost': {'amount': '0', 'currency': 'usd'},
            'operator': {'id': operator_id},
            'ptoState': 'inactive',
            'startTime': start.isoformat().replace('+00:00', 'Z'),
            'latitude': 30.0,
            'longitude': -90.0,
        }
    )


def _make_samsara_bundle(  # noqa: PLR0913 -- mirrors SamsaraUtilizationBundle fields
    *,
    vehicles: list[SamsaraVehicle] | None = None,
    drivers: list[SamsaraDriver] | None = None,
    trips: list[Trip] | None = None,
    idling_events: list[IdlingEvent] | None = None,
    company: str | None = 'samsara_co',
    date_range: tuple[date, date] = (_MAY_14, _MAY_20),
) -> SamsaraUtilizationBundle:
    return SamsaraUtilizationBundle(
        vehicles=vehicles or [_make_samsara_vehicle()],
        drivers=drivers or [_make_samsara_driver()],
        trips=trips or [],
        idling_events=idling_events or [],
        date_range=date_range,
        company=company,
    )


# -----------------------------------------------------------------------------
# Unit tests of orchestration boundary
# -----------------------------------------------------------------------------


class TestEmptyAndPartialBundles:
    """Behavior when one or both bundles are empty."""

    def test_empty_samsara_emits_only_motive_rows(self) -> None:
        """An empty Samsara bundle still produces the Motive rows."""

        df = unify(
            _make_motive_bundle(driving_periods=[_make_motive_driving_period()]),
            _make_samsara_bundle(),
        )

        assert len(df) == 1
        assert df.at[0, 'company'] == 'motive_co'
        assert df.at[0, 'event_type'] == 'driving'

    def test_empty_motive_emits_only_samsara_rows(self) -> None:
        """An empty Motive bundle still produces the Samsara rows."""

        df = unify(_make_motive_bundle(), _make_samsara_bundle(trips=[_make_trip()]))

        assert len(df) == 1
        assert df.at[0, 'company'] == 'samsara_co'
        assert df.at[0, 'event_type'] == 'driving'

    def test_both_present_both_empty_returns_zero_row_correct_schema(self) -> None:
        """Both empty bundles produce the same empty-DataFrame shape as both-None."""

        df = unify(_make_motive_bundle(), _make_samsara_bundle())

        assert len(df) == 0
        assert list(df.columns) == list(COLUMNS)
        assert df.dtypes.to_dict() == DTYPES


class TestSortBehavior:
    """``(company, start_time_utc, event_type)`` sort with stable ordering."""

    def test_company_is_non_decreasing(self) -> None:
        """``company`` column is non-decreasing across the output."""

        motive = _make_motive_bundle(
            driving_periods=[_make_motive_driving_period()],
            company='aaa_co',
        )
        samsara = _make_samsara_bundle(trips=[_make_trip()], company='zzz_co')
        df = unify(motive, samsara)

        companies = df['company'].tolist()
        assert companies == sorted(companies)

    def test_null_company_sorts_first(self) -> None:
        """``company=None`` rows precede any non-null-company rows."""

        motive = _make_motive_bundle(
            driving_periods=[_make_motive_driving_period()],
            company=None,
        )
        samsara = _make_samsara_bundle(trips=[_make_trip()], company='zzz_co')
        df = unify(motive, samsara)

        assert pd.isna(df.at[0, 'company'])
        assert df.at[1, 'company'] == 'zzz_co'

    def test_within_company_sort_by_start_time(self) -> None:
        """Within a company group, ``start_time_utc`` is non-decreasing."""

        early = _make_motive_driving_period(
            period_id=4550000001, start=_at(hour=8), end=_at(hour=9)
        )
        late = _make_motive_driving_period(
            period_id=4550000002, start=_at(hour=14), end=_at(hour=15)
        )
        # Pass them in reverse order to prove the sort, not concat order, wins.
        motive = _make_motive_bundle(driving_periods=[late, early])
        df = unify(motive, _make_samsara_bundle())

        assert df.at[0, 'start_time_utc'] == pd.Timestamp(_at(hour=8))
        assert df.at[1, 'start_time_utc'] == pd.Timestamp(_at(hour=14))

    def test_driving_precedes_idle_on_tied_time(self) -> None:
        """``event_type`` 'driving' precedes 'idle' alphabetically when other keys tie."""

        # Same vehicle, same start instant for driving and idle is a real
        # scenario (idle can begin exactly when a driving period ends or
        # share a boundary). Force the tie explicitly here.
        same_start = _at(hour=8)
        driving = _make_motive_driving_period(
            start=same_start, end=_at(hour=8, minute=30)
        )
        idle = _make_motive_idle_event(
            start=same_start, end=_at(hour=8, minute=15)
        )
        df = unify(
            _make_motive_bundle(driving_periods=[driving], idle_events=[idle]),
            _make_samsara_bundle(),
        )

        assert df.at[0, 'event_type'] == 'driving'
        assert df.at[1, 'event_type'] == 'idle'

    def test_tied_keys_break_motive_before_samsara(self) -> None:
        """Stable sort: with all keys equal, Motive rows precede Samsara rows."""

        # Force company, start_time_utc, and event_type all equal across
        # the two providers' driving rows. Sort stability + the
        # Motive-first concatenation order in unify means Motive lands first.
        shared_company = 'shared_co'
        shared_start = _at(hour=8)
        shared_end = _at(hour=9)
        motive = _make_motive_bundle(
            driving_periods=[
                _make_motive_driving_period(start=shared_start, end=shared_end)
            ],
            company=shared_company,
        )
        samsara = _make_samsara_bundle(
            trips=[_make_trip(start=shared_start, end=shared_end)],
            company=shared_company,
        )
        df = unify(motive, samsara)

        # Motive distance is 10 mi (16.09 km), Samsara is 1.0 mi (1609 m).
        # Use that to confirm which row landed first.
        assert df.at[0, 'distance_miles'] == _MOTIVE_DEFAULT_MILES
        assert df.at[1, 'distance_miles'] == _SAMSARA_DEFAULT_MILES


class TestSchemaIntegrity:
    """Output DataFrame still satisfies the locked schema after orchestration."""

    def test_non_empty_dtypes_match_schema(self) -> None:
        """Dtypes match ``DTYPES`` even when rows actually flow through."""

        df = unify(
            _make_motive_bundle(driving_periods=[_make_motive_driving_period()]),
            _make_samsara_bundle(trips=[_make_trip()]),
        )

        assert df.dtypes.to_dict() == DTYPES

    def test_event_type_column_holds_string_values(self) -> None:
        """The ``event_type`` column holds string values, not Enum instances."""

        df = unify(
            _make_motive_bundle(
                driving_periods=[_make_motive_driving_period()],
                idle_events=[_make_motive_idle_event()],
            ),
            _make_samsara_bundle(),
        )

        types_in_output = set(df['event_type'].tolist())
        assert types_in_output == {'driving', 'idle'}

    def test_datetime_columns_are_utc_tz_aware(self) -> None:
        """Both timestamp columns preserve their tz-aware UTC zone."""

        df = unify(
            _make_motive_bundle(driving_periods=[_make_motive_driving_period()]),
            _make_samsara_bundle(),
        )

        start_dtype = df.dtypes['start_time_utc']
        end_dtype = df.dtypes['end_time_utc']
        assert isinstance(start_dtype, pd.DatetimeTZDtype)
        assert isinstance(end_dtype, pd.DatetimeTZDtype)
        assert str(start_dtype.tz) == 'UTC'
        assert str(end_dtype.tz) == 'UTC'


# -----------------------------------------------------------------------------
# Integration tests (full transform-then-unify pipeline)
# -----------------------------------------------------------------------------


class TestFullPipelineIntegration:
    """Real bundles through real transforms into the final DataFrame."""

    def test_mixed_providers_full_shape(self) -> None:
        """A two-event Motive bundle + two-event Samsara bundle yields four rows."""

        motive = _make_motive_bundle(
            driving_periods=[
                _make_motive_driving_period(start=_at(hour=8), end=_at(hour=9))
            ],
            idle_events=[
                _make_motive_idle_event(
                    start=_at(hour=11), end=_at(hour=11, minute=30)
                )
            ],
        )
        samsara = _make_samsara_bundle(
            trips=[_make_trip(start=_at(hour=8), end=_at(hour=9))],
            idling_events=[_make_idling_event(start=_at(hour=12))],
        )
        df = unify(motive, samsara)

        expected_row_count = 4
        assert len(df) == expected_row_count
        # Schema preserved at the orchestrator boundary.
        assert list(df.columns) == list(COLUMNS)
        assert df.dtypes.to_dict() == DTYPES

    def test_different_companies_sort_by_company_first(self) -> None:
        """Crystal Clean (Motive) rows precede Patriot (Samsara) rows in sort."""

        motive = _make_motive_bundle(
            driving_periods=[_make_motive_driving_period()],
            company='crystal_clean',
        )
        samsara = _make_samsara_bundle(
            trips=[_make_trip()], company='patriot'
        )
        df = unify(motive, samsara)

        assert df.at[0, 'company'] == 'crystal_clean'
        assert df.at[1, 'company'] == 'patriot'

    def test_same_company_sorts_within_by_time(self) -> None:
        """When both bundles share a company, rows interleave by ``start_time_utc``."""

        early_motive = _make_motive_driving_period(
            period_id=4550000001, start=_at(hour=8), end=_at(hour=9)
        )
        late_motive = _make_motive_driving_period(
            period_id=4550000002, start=_at(hour=14), end=_at(hour=15)
        )
        mid_samsara_trip = _make_trip(
            trip_id='00000000-0000-0000-0000-000000001001',
            start=_at(hour=10),
            end=_at(hour=11),
        )
        df = unify(
            _make_motive_bundle(
                driving_periods=[early_motive, late_motive], company='shared_co'
            ),
            _make_samsara_bundle(trips=[mid_samsara_trip], company='shared_co'),
        )

        start_times = df['start_time_utc'].tolist()
        assert start_times == [
            pd.Timestamp(_at(hour=8)),
            pd.Timestamp(_at(hour=10)),
            pd.Timestamp(_at(hour=14)),
        ]

    def test_samsara_gap_fill_flows_through_unify(self) -> None:
        """An idle event with null operator gets gap-filled from a covering trip."""

        unattributed_idle = _make_idling_event(
            operator_id='', start=_at(hour=10), duration_ms=30 * 60 * 1000
        )
        covering_trip = _make_trip(start=_at(hour=10), end=_at(hour=11))
        df = unify(
            _make_motive_bundle(),
            _make_samsara_bundle(
                trips=[covering_trip], idling_events=[unattributed_idle]
            ),
        )

        # Sort order: trip (driving) first, then idle.
        idle_row_index = df.index[df['event_type'] == 'idle'][0]
        assert df.at[idle_row_index, 'driver_id'] == _SAMSARA_DRIVER_ID
        assert df.at[idle_row_index, 'driver_name'] == _SAMSARA_DRIVER_NAME

    def test_motive_vin_drop_and_samsara_unknown_vin_coexist(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Motive null-VIN drops; Samsara unresolvable VIN survives with sentinel."""

        # Motive driving period with no VIN -> gets dropped entirely.
        motive_period_no_vin = _make_motive_driving_period(
            vehicle=_make_motive_vehicle(vin=None)
        )
        # Samsara trip whose vehicle_id isn't in the dim -> 'unknown_vin'.
        samsara_trip = _make_trip(vehicle_id='ghost_vehicle_id')

        with caplog.at_level(logging.WARNING):
            df = unify(
                _make_motive_bundle(driving_periods=[motive_period_no_vin]),
                _make_samsara_bundle(vehicles=[], trips=[samsara_trip]),
            )

        # Only the Samsara row survived -- the Motive row was dropped.
        assert len(df) == 1
        assert df.at[0, 'vin'] == 'unknown_vin'
        assert any(
            'null/empty VIN' in record.message for record in caplog.records
        )
        assert any(
            'VIN unresolvable' in record.message for record in caplog.records
        )


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


class TestEntryAndExitLogs:
    """INFO logs at the unify boundary."""

    def test_entry_log_includes_both_date_ranges(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When both bundles are present, the entry-INFO names both date ranges."""

        motive = _make_motive_bundle()
        samsara = _make_samsara_bundle()
        with caplog.at_level(logging.INFO, logger='fleet_telemetry_hub.unifier.unify'):
            unify(motive, samsara)

        entry_records = [
            record
            for record in caplog.records
            if 'unify called with both bundles' in record.message
        ]
        assert len(entry_records) == 1
        message = entry_records[0].message
        assert 'motive_date_range' in message
        assert 'samsara_date_range' in message

    def test_exit_log_includes_row_count_and_by_company(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The exit-INFO includes the row count and a per-company breakdown."""

        motive = _make_motive_bundle(
            driving_periods=[_make_motive_driving_period()], company='crystal_clean'
        )
        samsara = _make_samsara_bundle(
            trips=[_make_trip()], company='patriot'
        )
        with caplog.at_level(logging.INFO, logger='fleet_telemetry_hub.unifier.unify'):
            unify(motive, samsara)

        exit_records = [
            record
            for record in caplog.records
            if 'unify produced' in record.message
        ]
        assert len(exit_records) == 1
        message = exit_records[0].message
        assert 'unify produced 2 rows' in message
        assert 'crystal_clean' in message
        assert 'patriot' in message

    def test_null_company_appears_in_exit_log_as_sentinel(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A ``None`` company key is rendered as a readable sentinel string."""

        motive = _make_motive_bundle(
            driving_periods=[_make_motive_driving_period()], company=None
        )
        with caplog.at_level(logging.INFO, logger='fleet_telemetry_hub.unifier.unify'):
            unify(motive, _make_samsara_bundle())

        exit_records = [
            record
            for record in caplog.records
            if 'unify produced' in record.message
        ]
        assert len(exit_records) == 1
        assert '(null)' in exit_records[0].message

    def test_unify_emits_no_warning_records_itself(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The orchestrator does not emit WARNINGs on its own; only transforms do."""

        with caplog.at_level(logging.WARNING, logger='fleet_telemetry_hub.unifier.unify'):
            unify(_make_motive_bundle(), _make_samsara_bundle())

        assert not any(
            record.name == 'fleet_telemetry_hub.unifier.unify'
            and record.levelno == logging.WARNING
            for record in caplog.records
        )
