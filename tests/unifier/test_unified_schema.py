"""Tests for ``unifier.schema``: row dataclass, columns/dtypes, DataFrame builder.

Covers ``UnifiedEventRow`` construction-time validation (tz-awareness,
ordered timestamps, non-empty VIN, non-negative duration, non-negative
distance), the locked ``COLUMNS`` / ``DTYPES`` shape, and the
``build_dataframe`` materialization for both empty and populated
inputs.
"""

from dataclasses import FrozenInstanceError
from datetime import datetime

import pandas as pd
import pytest

from fleet_telemetry_hub.unifier.schema import (
    COLUMNS,
    DTYPES,
    SORT_COLUMNS,
    EventType,
    UnifiedEventRow,
    build_dataframe,
    sort_unified_frame,
)
from tests.unifier._fixtures import dt

# Default field values used to build a valid driving row in
# construction tests. Helpers below override one field at a time.
_DEFAULT_VIN = 'TESTVIN0000000100'
_DEFAULT_DRIVER_ID = '9000001'
_DEFAULT_DRIVER_NAME = 'Sam Snowflake'
_DEFAULT_COMPANY = 'test_co'
_ONE_HOUR_SECONDS = 3600
_TEN_MILES = 10.0
# A distance distinct from ``_TEN_MILES`` used only to tell two
# equal-sort-key rows apart in the stable-sort test.
_OTHER_MILES = 99.0

# Expected pandas dtype name strings used by the DataFrame builder
# tests; pinned here rather than recomputed from ``DTYPES`` so a
# silent dtype change loudly fails the test instead of being
# tautologically accepted.
_EXPECTED_DTYPE_NAMES: dict[str, str] = {
    'company': 'string',
    'event_type': 'string',
    'driver_id': 'string',
    'driver_name': 'string',
    'vin': 'string',
    'start_time_utc': 'datetime64[ns, UTC]',
    'end_time_utc': 'datetime64[ns, UTC]',
    'duration_seconds': 'Int64',
    'distance_miles': 'Float64',
}


def _driving_row(
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    vin: str = _DEFAULT_VIN,
    duration_seconds: int = _ONE_HOUR_SECONDS,
    distance_miles: float | None = _TEN_MILES,
) -> UnifiedEventRow:
    """Build a valid driving ``UnifiedEventRow`` with overridable fields."""
    return UnifiedEventRow(
        company=_DEFAULT_COMPANY,
        event_type=EventType.DRIVING,
        driver_id=_DEFAULT_DRIVER_ID,
        driver_name=_DEFAULT_DRIVER_NAME,
        vin=vin,
        start_time_utc=start if start is not None else dt(hour=8),
        end_time_utc=end if end is not None else dt(hour=9),
        duration_seconds=duration_seconds,
        distance_miles=distance_miles,
    )


def _idle_row(
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    duration_seconds: int = _ONE_HOUR_SECONDS,
) -> UnifiedEventRow:
    """Build a valid idle ``UnifiedEventRow`` (no distance) with overridable timings."""
    return UnifiedEventRow(
        company=_DEFAULT_COMPANY,
        event_type=EventType.IDLE,
        driver_id=_DEFAULT_DRIVER_ID,
        driver_name=_DEFAULT_DRIVER_NAME,
        vin=_DEFAULT_VIN,
        start_time_utc=start if start is not None else dt(hour=10),
        end_time_utc=end if end is not None else dt(hour=11),
        duration_seconds=duration_seconds,
        distance_miles=None,
    )


class TestUnifiedEventRowConstruction:
    """Construction-time validation rules on ``UnifiedEventRow``."""

    def test_valid_driving_row_constructs_successfully(self) -> None:
        """A normal driving row constructs and preserves field values."""

        row = _driving_row()

        assert row.event_type is EventType.DRIVING
        assert row.vin == _DEFAULT_VIN
        assert row.duration_seconds == _ONE_HOUR_SECONDS
        assert row.distance_miles == _TEN_MILES

    def test_valid_idle_row_with_null_distance_constructs(self) -> None:
        """An idle row with ``distance_miles=None`` is valid."""

        row = _idle_row()

        assert row.event_type is EventType.IDLE
        assert row.distance_miles is None

    def test_zero_duration_idle_row_is_allowed(self) -> None:
        """A zero-duration idle row is allowed (start == end, duration == 0)."""

        instant = dt(hour=8)
        row = _idle_row(start=instant, end=instant, duration_seconds=0)

        assert row.duration_seconds == 0

    def test_reversed_times_raise_value_error(self) -> None:
        """``start_time_utc > end_time_utc`` raises ``ValueError``."""

        with pytest.raises(ValueError, match='must be <='):
            _driving_row(start=dt(hour=9), end=dt(hour=8))

    def test_naive_start_time_raises_type_error(self) -> None:
        """A naive ``start_time_utc`` raises ``TypeError``."""

        naive = datetime(2026, 5, 14, 8, 0, 0)  # noqa: DTZ001 -- intentionally naive
        with pytest.raises(TypeError, match='tz-aware'):
            _driving_row(start=naive)

    def test_naive_end_time_raises_type_error(self) -> None:
        """A naive ``end_time_utc`` raises ``TypeError``."""

        naive = datetime(2026, 5, 14, 9, 0, 0)  # noqa: DTZ001 -- intentionally naive
        with pytest.raises(TypeError, match='tz-aware'):
            _driving_row(end=naive)

    def test_empty_vin_raises_value_error(self) -> None:
        """Empty VIN raises ``ValueError`` (transform must drop these rows earlier)."""

        with pytest.raises(ValueError, match='vin'):
            _driving_row(vin='')

    def test_negative_duration_raises_value_error(self) -> None:
        """Negative ``duration_seconds`` raises ``ValueError``."""

        with pytest.raises(ValueError, match='duration_seconds'):
            _driving_row(duration_seconds=-1)

    def test_negative_distance_raises_value_error(self) -> None:
        """Negative ``distance_miles`` raises ``ValueError``."""

        with pytest.raises(ValueError, match='distance_miles'):
            _driving_row(distance_miles=-0.1)

    def test_row_is_frozen(self) -> None:
        """Reassigning a field raises ``FrozenInstanceError``."""

        row = _driving_row()
        with pytest.raises(FrozenInstanceError):
            row.duration_seconds = 0  # type: ignore[misc]


class TestColumnsAndDtypes:
    """Locked-schema constants: column order and dtype map."""

    def test_columns_is_expected_tuple(self) -> None:
        """``COLUMNS`` is exactly the nine column names in the locked order."""

        expected = (
            'company',
            'event_type',
            'driver_id',
            'driver_name',
            'vin',
            'start_time_utc',
            'end_time_utc',
            'duration_seconds',
            'distance_miles',
        )
        assert expected == COLUMNS

    def test_dtypes_keys_match_columns(self) -> None:
        """``DTYPES`` covers exactly the schema columns -- no extras, no gaps."""

        assert set(DTYPES.keys()) == set(COLUMNS)


class TestBuildDataframe:
    """``build_dataframe`` materializes the locked schema for empty and populated inputs."""

    def test_empty_input_returns_zero_row_frame_with_correct_columns(self) -> None:
        """Empty input still produces the locked column order."""

        frame = build_dataframe([])

        assert len(frame) == 0
        assert list(frame.columns) == list(COLUMNS)

    def test_empty_input_dtypes_match_expected_names(self) -> None:
        """Empty input frame still carries the locked dtypes."""

        frame = build_dataframe([])

        actual_dtype_names = {col: dtype.name for col, dtype in frame.dtypes.items()}
        assert actual_dtype_names == _EXPECTED_DTYPE_NAMES

    def test_single_driving_row_populates_each_column(self) -> None:
        """One row in, one row out, with all nine columns populated."""

        row = _driving_row()
        frame = build_dataframe([row])

        assert len(frame) == 1
        assert frame.at[0, 'company'] == _DEFAULT_COMPANY
        assert frame.at[0, 'event_type'] == 'driving'
        assert frame.at[0, 'driver_id'] == _DEFAULT_DRIVER_ID
        assert frame.at[0, 'driver_name'] == _DEFAULT_DRIVER_NAME
        assert frame.at[0, 'vin'] == _DEFAULT_VIN
        assert frame.at[0, 'start_time_utc'] == pd.Timestamp(dt(hour=8))
        assert frame.at[0, 'end_time_utc'] == pd.Timestamp(dt(hour=9))
        assert frame.at[0, 'duration_seconds'] == _ONE_HOUR_SECONDS
        assert frame.at[0, 'distance_miles'] == _TEN_MILES

    def test_idle_row_distance_is_missing_marker(self) -> None:
        """An idle row carries the Float64Dtype missing marker for distance."""

        frame = build_dataframe([_idle_row()])

        assert pd.isna(frame.at[0, 'distance_miles'])

    def test_mixed_rows_preserve_input_order(self) -> None:
        """Rows appear in the DataFrame in the order they were passed."""

        driving = _driving_row()
        idle = _idle_row()
        frame = build_dataframe([driving, idle])

        assert frame.at[0, 'event_type'] == 'driving'
        assert frame.at[1, 'event_type'] == 'idle'

    def test_event_type_values_are_strings_not_enum_instances(self) -> None:
        """The ``event_type`` column holds the underlying string values."""

        frame = build_dataframe([_driving_row(), _idle_row()])

        # Cells must be plain strings, not ``EventType`` instances.
        assert frame.at[0, 'event_type'] == 'driving'
        assert frame.at[1, 'event_type'] == 'idle'
        assert not isinstance(frame.at[0, 'event_type'], EventType)

    def test_timestamps_preserve_utc_tz(self) -> None:
        """tz-aware UTC timestamps survive into the DataFrame."""

        frame = build_dataframe([_driving_row()])

        start_dtype = frame.dtypes['start_time_utc']
        end_dtype = frame.dtypes['end_time_utc']
        assert str(start_dtype) == 'datetime64[ns, UTC]'
        assert str(end_dtype) == 'datetime64[ns, UTC]'


def _sortable_row(
    *,
    company: str | None,
    start: datetime,
    event_type: EventType = EventType.DRIVING,
) -> UnifiedEventRow:
    """Build a row whose sort-key fields are explicit and others defaulted."""
    return UnifiedEventRow(
        company=company,
        event_type=event_type,
        driver_id=_DEFAULT_DRIVER_ID,
        driver_name=_DEFAULT_DRIVER_NAME,
        vin=_DEFAULT_VIN,
        start_time_utc=start,
        end_time_utc=start,
        duration_seconds=0,
        distance_miles=None if event_type is EventType.IDLE else _TEN_MILES,
    )


class TestSortUnifiedFrame:
    """``sort_unified_frame`` orders by ``SORT_COLUMNS`` nulls-first and stably."""

    def test_sort_columns_constant(self) -> None:
        """The shared sort key is ``(company, start_time_utc, event_type)``."""

        assert SORT_COLUMNS == ('company', 'start_time_utc', 'event_type')

    def test_null_company_sorts_before_non_null(self) -> None:
        """``company`` nulls land ahead of any non-null company string."""

        frame = build_dataframe(
            [
                _sortable_row(company='aaa_co', start=dt(hour=8)),
                _sortable_row(company=None, start=dt(hour=8)),
            ]
        )

        result = sort_unified_frame(frame)

        assert pd.isna(result.at[0, 'company'])
        assert result.at[1, 'company'] == 'aaa_co'

    def test_ties_broken_by_start_then_event_type(self) -> None:
        """Equal company sorts by ``start_time_utc`` then ``event_type``."""

        frame = build_dataframe(
            [
                _sortable_row(
                    company='co', start=dt(hour=9), event_type=EventType.IDLE
                ),
                _sortable_row(
                    company='co', start=dt(hour=8), event_type=EventType.IDLE
                ),
                _sortable_row(
                    company='co', start=dt(hour=8), event_type=EventType.DRIVING
                ),
            ]
        )

        result = sort_unified_frame(frame)

        assert result['start_time_utc'].tolist() == [
            pd.Timestamp(dt(hour=8)),
            pd.Timestamp(dt(hour=8)),
            pd.Timestamp(dt(hour=9)),
        ]
        assert result['event_type'].tolist() == ['driving', 'idle', 'idle']

    def test_stable_on_equal_keys(self) -> None:
        """Rows sharing the full sort key keep their input order."""

        # Two rows with an identical sort key, distinguishable only by a
        # non-key field (distance). Input order must survive the sort.
        first = _sortable_row(company='co', start=dt(hour=8))
        second = UnifiedEventRow(
            company='co',
            event_type=EventType.DRIVING,
            driver_id=_DEFAULT_DRIVER_ID,
            driver_name=_DEFAULT_DRIVER_NAME,
            vin=_DEFAULT_VIN,
            start_time_utc=dt(hour=8),
            end_time_utc=dt(hour=8),
            duration_seconds=0,
            distance_miles=_OTHER_MILES,
        )
        frame = build_dataframe([first, second])

        result = sort_unified_frame(frame)

        assert result.at[0, 'distance_miles'] == _TEN_MILES
        assert result.at[1, 'distance_miles'] == _OTHER_MILES

    def test_returns_range_index(self) -> None:
        """The sorted frame carries a clean positional ``RangeIndex``."""

        frame = build_dataframe(
            [
                _sortable_row(company='b_co', start=dt(hour=8)),
                _sortable_row(company='a_co', start=dt(hour=8)),
            ]
        )

        result = sort_unified_frame(frame)

        assert isinstance(result.index, pd.RangeIndex)
        assert result.index.tolist() == [0, 1]

    def test_columns_and_dtypes_unchanged(self) -> None:
        """Sorting preserves the locked column order and dtypes."""

        frame = build_dataframe(
            [
                _sortable_row(company='b_co', start=dt(hour=8)),
                _sortable_row(company='a_co', start=dt(hour=9)),
            ]
        )

        result = sort_unified_frame(frame)

        assert list(result.columns) == list(COLUMNS)
        assert result.dtypes.to_dict() == DTYPES
