"""Tests for ``unifier.motive_transform.transform_motive_bundle``.

Uses real Pydantic instances for every source record so the
transform's actual field-access paths and Pydantic validation are
exercised. Helpers build a default ``DrivingPeriod`` / ``IdleEvent``
and accept keyword overrides for the fields each test cares about.
"""

import logging
from datetime import UTC, date, datetime

import pytest

from fleet_telemetry_hub.models.motive_responses import (
    DriverSummary,
    DrivingPeriod,
    EldDeviceInfo,
    IdleEvent,
    VehicleSummary,
)
from fleet_telemetry_hub.unifier.motive_transform import transform_motive_bundle
from fleet_telemetry_hub.unifier.schema import EventType
from fleet_telemetry_hub.utilization.motive_fetcher import MotiveUtilizationBundle

# Date constants shared across tests (keep cross-test consistency).
_MAY_14 = date(2026, 5, 14)
_MAY_20 = date(2026, 5, 20)


def _dt(hour: int = 8, minute: int = 0) -> datetime:
    """Build a tz-aware UTC datetime on 2026-05-14."""
    return datetime(2026, 5, 14, hour, minute, 0, tzinfo=UTC)


_DEFAULT_VEHICLE_ID = 8000001
_DEFAULT_VIN = 'TESTVIN0000000100'
_DEFAULT_DRIVER_ID = 9000001
_DEFAULT_DRIVER_FIRST_NAME = 'Sam'
_DEFAULT_DRIVER_LAST_NAME = 'Snowflake'
_DEFAULT_DRIVER_FULL_NAME = f'{_DEFAULT_DRIVER_FIRST_NAME} {_DEFAULT_DRIVER_LAST_NAME}'

# Common assertion landmarks reused across tests.
_ONE_HOUR_SECONDS = 3600
_TWO_HOURS_SECONDS = 7200
_FIFTEEN_MIN_SECONDS = 900
_TEN_MIN_SECONDS = 600
_DEFAULT_DISTANCE_KM = 16.09
_DEFAULT_DISTANCE_MILES = 10.0
_DRIVING_PLUS_IDLE_ROW_COUNT = 2


def _make_vehicle(
    *,
    vehicle_id: int = _DEFAULT_VEHICLE_ID,
    vin: str | None = _DEFAULT_VIN,
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


def _make_driver(
    *,
    driver_id: int = _DEFAULT_DRIVER_ID,
    first_name: str = _DEFAULT_DRIVER_FIRST_NAME,
    last_name: str = _DEFAULT_DRIVER_LAST_NAME,
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


def _make_eld_device() -> EldDeviceInfo:
    return EldDeviceInfo.model_validate(
        {'id': 8800001, 'identifier': 'TESTELD0001', 'model': 'lbb-3.6ca'}
    )


# Module-level default DriverSummary used as the in-signature default
# for the period/event factories below. DriverSummary is a frozen
# Pydantic model, so sharing the instance across calls is safe and
# avoids the mutable-default-argument footgun while still allowing
# ``driver=None`` to mean an unattributed event.
_DEFAULT_DRIVER: DriverSummary = _make_driver()


def _make_driving_period(  # noqa: PLR0913 -- test factory; one knob per field
    *,
    vehicle: VehicleSummary | None = None,
    driver: DriverSummary | None = _DEFAULT_DRIVER,
    start: datetime | None = None,
    end: datetime | None = None,
    distance_km: float = _DEFAULT_DISTANCE_KM,
    type_value: str = 'driving',
    period_id: int = 4550000001,
    null_start_kilometers: bool = False,
    null_end_kilometers: bool = False,
) -> DrivingPeriod:
    """
    Build a ``DrivingPeriod`` with sensible defaults.

    ``driver=None`` produces an unattributed period; pass a
    ``DriverSummary`` (or accept the default) to attribute the
    period. ``null_start_kilometers`` / ``null_end_kilometers``
    swap the respective odometer reading for ``None`` so tests can
    exercise the unifier's null-odometer soft-warning path. The
    boolean-knob shape avoids overloading ``None`` as a sentinel
    here, since ``None`` is a legitimate field value.
    """
    if vehicle is None:
        vehicle = _make_vehicle()
    if start is None:
        start = _dt(hour=8)
    if end is None:
        end = _dt(hour=9)
    start_kilometers: float | None = None if null_start_kilometers else 100.0
    end_kilometers: float | None = (
        None if null_end_kilometers else 100.0 + distance_km
    )
    return DrivingPeriod.model_validate(
        {
            'id': period_id,
            'start_time': start,
            'end_time': end,
            'status': 'complete',
            'type': type_value,
            'annotation_status': None,
            'notes': None,
            'duration': int((end - start).total_seconds()),
            'start_kilometers': start_kilometers,
            'end_kilometers': end_kilometers,
            'source': 1,
            'driver': (
                driver.model_dump(by_alias=True) if driver is not None else None
            ),
            'vehicle': vehicle.model_dump(by_alias=True),
        }
    )


def _make_idle_event(
    *,
    vehicle: VehicleSummary | None = None,
    driver: DriverSummary | None = _DEFAULT_DRIVER,
    start: datetime | None = None,
    end: datetime | None = None,
    event_id: int = 4860000001,
) -> IdleEvent:
    """Build an ``IdleEvent`` with sensible defaults (``driver=None`` is unattributed)."""
    if vehicle is None:
        vehicle = _make_vehicle()
    if start is None:
        start = _dt(hour=10)
    if end is None:
        end = _dt(hour=10, minute=30)
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
            'eld_device': _make_eld_device().model_dump(by_alias=True),
            'location': 'Testville, TX',
        }
    )


def _make_bundle(
    *,
    driving_periods: list[DrivingPeriod] | None = None,
    idle_events: list[IdleEvent] | None = None,
    company: str | None = 'test_co',
    date_range: tuple[date, date] = (_MAY_14, _MAY_20),
) -> MotiveUtilizationBundle:
    return MotiveUtilizationBundle(
        driving_periods=driving_periods or [],
        idle_events=idle_events or [],
        date_range=date_range,
        company=company,
    )


# Vehicle B used for cross-vehicle tests.
_OTHER_VEHICLE_ID = 8000002
_OTHER_VIN = 'TESTVIN0000000101'


def _make_other_vehicle() -> VehicleSummary:
    return _make_vehicle(vehicle_id=_OTHER_VEHICLE_ID, vin=_OTHER_VIN)


class TestDrivingPeriodConversion:
    """Driving-period → unified row, including idle-subtraction math."""

    def test_single_driving_period_no_idle(self) -> None:
        """No idle on the same vehicle means duration is the full span."""

        period = _make_driving_period()
        rows = transform_motive_bundle(_make_bundle(driving_periods=[period]))

        assert len(rows) == 1
        row = rows[0]
        assert row.event_type is EventType.DRIVING
        assert row.vin == _DEFAULT_VIN
        assert row.duration_seconds == _ONE_HOUR_SECONDS
        assert row.distance_miles == _DEFAULT_DISTANCE_MILES
        assert row.driver_id == str(_DEFAULT_DRIVER_ID)
        assert row.driver_name == _DEFAULT_DRIVER_FULL_NAME
        assert row.company == 'test_co'

    def test_fully_contained_idle_subtracts_its_duration(self) -> None:
        """A 15-minute idle inside a 60-minute driving leaves 45 minutes."""

        period = _make_driving_period(start=_dt(hour=8), end=_dt(hour=9))
        idle = _make_idle_event(
            start=_dt(hour=8, minute=20),
            end=_dt(hour=8, minute=35),
        )
        rows = transform_motive_bundle(
            _make_bundle(driving_periods=[period], idle_events=[idle])
        )
        driving_row = next(row for row in rows if row.event_type is EventType.DRIVING)

        assert driving_row.duration_seconds == _ONE_HOUR_SECONDS - _FIFTEEN_MIN_SECONDS

    def test_idle_on_other_vehicle_does_not_affect_driving(self) -> None:
        """Idle on a different vehicle does not subtract from this driving."""

        period = _make_driving_period(start=_dt(hour=8), end=_dt(hour=9))
        other_idle = _make_idle_event(
            vehicle=_make_other_vehicle(),
            start=_dt(hour=8, minute=20),
            end=_dt(hour=8, minute=35),
        )
        rows = transform_motive_bundle(
            _make_bundle(driving_periods=[period], idle_events=[other_idle])
        )
        driving_row = next(row for row in rows if row.event_type is EventType.DRIVING)

        assert driving_row.duration_seconds == _ONE_HOUR_SECONDS

    def test_multiple_overlapping_idle_events_subtract_their_sum(self) -> None:
        """Two idle windows inside one driving subtract the sum of overlaps."""

        period = _make_driving_period(start=_dt(hour=8), end=_dt(hour=10))
        idle_a = _make_idle_event(
            start=_dt(hour=8), end=_dt(hour=8, minute=10), event_id=4860000001
        )
        idle_b = _make_idle_event(
            start=_dt(hour=9), end=_dt(hour=9, minute=15), event_id=4860000002
        )
        rows = transform_motive_bundle(
            _make_bundle(driving_periods=[period], idle_events=[idle_a, idle_b])
        )
        driving_row = next(row for row in rows if row.event_type is EventType.DRIVING)

        assert (
            driving_row.duration_seconds
            == _TWO_HOURS_SECONDS - _TEN_MIN_SECONDS - _FIFTEEN_MIN_SECONDS
        )

    def test_null_vin_drops_driving_row_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A driving period whose vehicle has ``vin=None`` is dropped with WARNING."""

        period = _make_driving_period(vehicle=_make_vehicle(vin=None))
        with caplog.at_level(logging.WARNING):
            rows = transform_motive_bundle(_make_bundle(driving_periods=[period]))

        assert rows == []
        assert any(
            'null/empty VIN' in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        )

    @pytest.mark.parametrize('empty_vin', ['', '   '])
    def test_empty_or_whitespace_vin_drops_driving_row(self, empty_vin: str) -> None:
        """Empty or whitespace-only VIN is dropped just like ``None``."""

        period = _make_driving_period(vehicle=_make_vehicle(vin=empty_vin))
        rows = transform_motive_bundle(_make_bundle(driving_periods=[period]))

        assert rows == []

    @pytest.mark.parametrize('type_value', ['PC', 'YM'])
    def test_pc_and_ym_types_map_to_driving(self, type_value: str) -> None:
        """``type='PC'`` and ``type='YM'`` both emit ``EventType.DRIVING``."""

        period = _make_driving_period(type_value=type_value)
        rows = transform_motive_bundle(_make_bundle(driving_periods=[period]))

        assert len(rows) == 1
        assert rows[0].event_type is EventType.DRIVING

    def test_idle_fully_covers_driving_drops_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When idle subtracts to <= 0 the driving row drops and the idle still emits."""

        period = _make_driving_period(start=_dt(hour=8), end=_dt(hour=9))
        idle = _make_idle_event(start=_dt(hour=7), end=_dt(hour=10))
        with caplog.at_level(logging.WARNING):
            rows = transform_motive_bundle(
                _make_bundle(driving_periods=[period], idle_events=[idle])
            )

        assert all(row.event_type is EventType.IDLE for row in rows)
        assert len(rows) == 1
        assert any(
            'idle fully covers driving' in record.message
            and record.levelno == logging.WARNING
            for record in caplog.records
        )


class TestIdleEventConversion:
    """Idle-event → unified row, including the null-driver gap-fill rules."""

    def test_idle_with_null_driver_no_overlap_stays_null(self) -> None:
        """No driving on the same vehicle means both driver fields stay null."""

        idle = _make_idle_event(driver=None)
        rows = transform_motive_bundle(_make_bundle(idle_events=[idle]))

        assert len(rows) == 1
        assert rows[0].event_type is EventType.IDLE
        assert rows[0].driver_id is None
        assert rows[0].driver_name is None
        assert rows[0].distance_miles is None

    def test_idle_with_null_driver_single_overlap_is_gap_filled(self) -> None:
        """A single fully-overlapping driving period populates both driver fields."""

        idle = _make_idle_event(
            driver=None, start=_dt(hour=10), end=_dt(hour=10, minute=30)
        )
        covering_period = _make_driving_period(start=_dt(hour=10), end=_dt(hour=11))
        rows = transform_motive_bundle(
            _make_bundle(
                driving_periods=[covering_period],
                idle_events=[idle],
            )
        )
        idle_row = next(row for row in rows if row.event_type is EventType.IDLE)

        assert idle_row.driver_id == str(_DEFAULT_DRIVER_ID)
        assert idle_row.driver_name == 'Sam Snowflake'

    def test_yard_hand_idle_keeps_null_driver(self) -> None:
        """5 min of driving inside a 30 min idle: uncovered time wins -> null."""

        idle = _make_idle_event(
            driver=None, start=_dt(hour=10), end=_dt(hour=10, minute=30)
        )
        # Single 5-minute driving slice inside the idle.
        period = _make_driving_period(
            start=_dt(hour=10), end=_dt(hour=10, minute=5)
        )
        rows = transform_motive_bundle(
            _make_bundle(driving_periods=[period], idle_events=[idle])
        )
        idle_row = next(row for row in rows if row.event_type is EventType.IDLE)

        assert idle_row.driver_id is None
        assert idle_row.driver_name is None

    def test_multi_driver_overlap_warns_with_bucket_distribution(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two distinct drivers overlapping idle: most-overlap wins; warn logged."""

        idle = _make_idle_event(
            driver=None,
            start=_dt(hour=10),
            end=_dt(hour=10, minute=30),
        )
        sam_period = _make_driving_period(
            start=_dt(hour=10),
            end=_dt(hour=10, minute=20),
            driver=_make_driver(driver_id=9000001, first_name='Sam'),
            period_id=4550000001,
        )
        suzy_period = _make_driving_period(
            start=_dt(hour=10, minute=20),
            end=_dt(hour=10, minute=30),
            driver=_make_driver(driver_id=9000002, first_name='Suzy'),
            period_id=4550000002,
        )
        with caplog.at_level(logging.WARNING):
            rows = transform_motive_bundle(
                _make_bundle(
                    driving_periods=[sam_period, suzy_period],
                    idle_events=[idle],
                )
            )
        idle_row = next(row for row in rows if row.event_type is EventType.IDLE)

        assert idle_row.driver_id == '9000001'  # Sam (20 min) > Suzy (10 min)
        warn_records = [
            record
            for record in caplog.records
            if 'Multiple drivers overlap' in record.message
        ]
        assert len(warn_records) == 1
        message = warn_records[0].message
        assert f'vehicle_id={_DEFAULT_VEHICLE_ID}' in message
        assert 'bucket_distribution' in message

    def test_partial_null_driver_does_not_trigger_gap_fill(self) -> None:
        """driver_id present, driver_name nullified to None: fields stay as-is."""

        # ``last_name='Unknown'`` makes full_name 'Real Unknown' which does NOT
        # match the 'unknown' token. Use first_name='Unknown', last_name=''
        # so full_name normalizes to None while driver_id stays set.
        idle = _make_idle_event(
            driver=_make_driver(first_name='Unknown', last_name=''),
            start=_dt(hour=10),
            end=_dt(hour=10, minute=30),
        )
        covering_period = _make_driving_period(
            start=_dt(hour=10),
            end=_dt(hour=11),
            driver=_make_driver(driver_id=9999999, first_name='Other'),
        )
        rows = transform_motive_bundle(
            _make_bundle(
                driving_periods=[covering_period],
                idle_events=[idle],
            )
        )
        idle_row = next(row for row in rows if row.event_type is EventType.IDLE)

        # driver_id stayed because the int was non-null; driver_name normalized to None.
        assert idle_row.driver_id == str(_DEFAULT_DRIVER_ID)
        assert idle_row.driver_name is None

    def test_driver_name_unknown_normalizes_to_none(self) -> None:
        """A first_name 'Unknown' (case-sensitive) normalizes the full name to None."""

        idle = _make_idle_event(
            driver=_make_driver(first_name='Unknown', last_name=''),
        )
        rows = transform_motive_bundle(_make_bundle(idle_events=[idle]))

        assert rows[0].driver_name is None

    def test_driver_name_lowercase_unknown_normalizes_to_none(self) -> None:
        """Lowercase 'unknown' folds case-insensitively to None."""

        idle = _make_idle_event(
            driver=_make_driver(first_name='unknown', last_name=''),
        )
        rows = transform_motive_bundle(_make_bundle(idle_events=[idle]))

        assert rows[0].driver_name is None

    def test_driver_name_fullwidth_unknown_normalizes_to_none(self) -> None:
        """Full-width unicode spelling of 'unknown' NFKC-folds and then nullifies."""

        idle = _make_idle_event(
            driver=_make_driver(
                first_name='ｕｎｋｎｏｗｎ',  # noqa: RUF001 -- intentional full-width
                last_name='',
            ),
        )
        rows = transform_motive_bundle(_make_bundle(idle_events=[idle]))

        assert rows[0].driver_name is None

    def test_idle_vin_null_drops_row(self) -> None:
        """An idle event with null VIN is dropped (driving row unaffected if absent)."""

        idle = _make_idle_event(vehicle=_make_vehicle(vin=None))
        rows = transform_motive_bundle(_make_bundle(idle_events=[idle]))

        assert rows == []


class TestBundleLevelBehavior:
    """Bundle-wide behavior: company propagation, empty input, ordering."""

    def test_company_none_propagates_to_all_rows(self) -> None:
        """``bundle.company=None`` flows into every emitted row."""

        bundle = _make_bundle(
            driving_periods=[_make_driving_period()],
            idle_events=[_make_idle_event()],
            company=None,
        )
        rows = transform_motive_bundle(bundle)

        assert len(rows) == _DRIVING_PLUS_IDLE_ROW_COUNT
        assert all(row.company is None for row in rows)

    def test_company_value_propagates_to_all_rows(self) -> None:
        """A populated ``bundle.company`` flows into every emitted row."""

        bundle = _make_bundle(
            driving_periods=[_make_driving_period()],
            idle_events=[_make_idle_event()],
            company='crystal_clean',
        )
        rows = transform_motive_bundle(bundle)

        assert all(row.company == 'crystal_clean' for row in rows)

    def test_empty_bundle_returns_empty_list(self) -> None:
        """A bundle with no driving and no idle returns an empty list."""

        rows = transform_motive_bundle(_make_bundle())

        assert rows == []

    def test_emission_order_is_driving_then_idle_in_source_order(self) -> None:
        """Driving rows emit first in source order, then idle rows in source order."""

        period_a = _make_driving_period(period_id=4550000001)
        period_b = _make_driving_period(
            period_id=4550000002,
            start=_dt(hour=13),
            end=_dt(hour=14),
        )
        idle_a = _make_idle_event(event_id=4860000001)
        idle_b = _make_idle_event(
            event_id=4860000002,
            start=_dt(hour=12),
            end=_dt(hour=12, minute=30),
        )
        bundle = _make_bundle(
            driving_periods=[period_a, period_b],
            idle_events=[idle_a, idle_b],
        )
        rows = transform_motive_bundle(bundle)

        assert [row.event_type for row in rows] == [
            EventType.DRIVING,
            EventType.DRIVING,
            EventType.IDLE,
            EventType.IDLE,
        ]
        # First driving row's start matches period_a; second matches period_b.
        assert rows[0].start_time_utc == _dt(hour=8)
        assert rows[1].start_time_utc == _dt(hour=13)
        assert rows[2].start_time_utc == _dt(hour=10)
        assert rows[3].start_time_utc == _dt(hour=12)


class TestDrivingPeriodNullOdometerSoftWarning:
    """Null-odometer rows emit with ``distance_miles=None`` rather than dropping."""

    def test_driving_period_with_null_odometer_emits_row_with_null_distance(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Null both readings -> row still emitted; ``distance_miles`` is null."""

        period = _make_driving_period(
            null_start_kilometers=True, null_end_kilometers=True
        )
        with caplog.at_level(logging.WARNING):
            rows = transform_motive_bundle(_make_bundle(driving_periods=[period]))

        assert len(rows) == 1
        row = rows[0]
        assert row.event_type is EventType.DRIVING
        assert row.distance_miles is None
        # Other fields still reflect the input -- only distance degraded.
        assert row.vin == _DEFAULT_VIN
        assert row.driver_id == str(_DEFAULT_DRIVER_ID)
        assert row.driver_name == _DEFAULT_DRIVER_FULL_NAME
        assert row.start_time_utc == _dt(hour=8)
        assert row.end_time_utc == _dt(hour=9)
        assert row.duration_seconds == _ONE_HOUR_SECONDS
        # WARNING fires with the documented substring.
        assert any(
            'null odometer reading' in record.message for record in caplog.records
        )

    def test_null_start_only_still_emits_row_with_null_distance(self) -> None:
        """Only ``start_kilometers`` null -> still soft-degrade, not a drop."""

        period = _make_driving_period(null_start_kilometers=True)
        rows = transform_motive_bundle(_make_bundle(driving_periods=[period]))

        assert len(rows) == 1
        assert rows[0].distance_miles is None

    def test_null_end_only_still_emits_row_with_null_distance(self) -> None:
        """Only ``end_kilometers`` null -> still soft-degrade, not a drop."""

        period = _make_driving_period(null_end_kilometers=True)
        rows = transform_motive_bundle(_make_bundle(driving_periods=[period]))

        assert len(rows) == 1
        assert rows[0].distance_miles is None

    def test_null_odometer_increments_soft_warnings_counter(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Final INFO log reports ``soft_warnings={'null_odometer': 1}``."""

        period = _make_driving_period(
            null_start_kilometers=True, null_end_kilometers=True
        )
        # Scope to the transform's logger so a prior test that pinned
        # the package logger to WARNING does not filter the final
        # INFO line before caplog sees it.
        with caplog.at_level(
            logging.INFO, logger='fleet_telemetry_hub.unifier.motive_transform'
        ):
            transform_motive_bundle(_make_bundle(driving_periods=[period]))

        complete_records = [
            record
            for record in caplog.records
            if 'Motive transform complete' in record.message
        ]
        assert len(complete_records) == 1
        message = complete_records[0].message
        assert "soft_warnings={'null_odometer': 1}" in message

    def test_present_odometer_does_not_increment_soft_warnings(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Normal records keep ``null_odometer`` at zero in the final summary."""

        period = _make_driving_period()  # defaults: both readings present
        with caplog.at_level(
            logging.INFO, logger='fleet_telemetry_hub.unifier.motive_transform'
        ):
            transform_motive_bundle(_make_bundle(driving_periods=[period]))

        complete_records = [
            record
            for record in caplog.records
            if 'Motive transform complete' in record.message
        ]
        assert len(complete_records) == 1
        assert "soft_warnings={'null_odometer': 0}" in complete_records[0].message
