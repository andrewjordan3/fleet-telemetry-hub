"""Tests for ``_utilization_merge.merge_incremental``: the pure delete-then-append merge.

Frames are built from ``UnifiedEventRow`` + ``build_dataframe`` so the
cases exercise the real locked schema and its construction-time
validation. All identifiers are synthetic. The window under test is the
single UTC day 2026-05-14, i.e. ``W_start = 2026-05-14T00:00Z`` and
``W_end = 2026-05-15T00:00Z`` (exclusive).
"""

import logging
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from fleet_telemetry_hub._utilization_merge import merge_incremental
from fleet_telemetry_hub.unifier.schema import (
    COLUMNS,
    DTYPES,
    EventType,
    UnifiedEventRow,
    build_dataframe,
)

_VIN = 'TESTVIN0000000001'
_DRIVER_ID = 'TEST-DRIVER-01'
_DRIVER_NAME = 'Test Driver'
_COMPANY = 'test_co'

# The single-day window every case merges into.
_START_DATE = date(2026, 5, 14)
_END_DATE = date(2026, 5, 14)
_W_START = pd.Timestamp(2026, 5, 14, tz='UTC')  # inclusive lower bound
_W_END = pd.Timestamp(2026, 5, 15, tz='UTC')  # exclusive upper bound

# Distinguishing distances let a test prove which copy of a same-key
# event survived the merge (old existing copy vs. new appended copy).
_OLD_DISTANCE = 5.0
_NEW_DISTANCE = 9.0


def _at(day: int, hour: int = 0) -> datetime:
    """Build a tz-aware UTC datetime in May 2026 for the given day/hour."""
    return datetime(2026, 5, day, hour, 0, 0, tzinfo=UTC)


def _row(
    start: datetime,
    *,
    company: str | None = _COMPANY,
    event_type: EventType = EventType.DRIVING,
    end: datetime | None = None,
    distance_miles: float | None = 1.0,
) -> UnifiedEventRow:
    """Build a single ``UnifiedEventRow`` keyed on ``start`` with sane defaults."""
    return UnifiedEventRow(
        company=company,
        event_type=event_type,
        driver_id=_DRIVER_ID,
        driver_name=_DRIVER_NAME,
        vin=_VIN,
        start_time_utc=start,
        end_time_utc=end if end is not None else start,
        duration_seconds=0,
        distance_miles=None if event_type is EventType.IDLE else distance_miles,
    )


def _frame(rows: list[UnifiedEventRow]) -> pd.DataFrame:
    """Materialize a schema-correct frame from the given rows."""
    return build_dataframe(rows)


class TestFirstRun:
    """``existing is None`` -- no file yet."""

    def test_non_empty_in_window_new_equals_new_rows(self) -> None:
        """First run with in-window rows returns exactly those rows."""

        new = _frame([_row(_at(14, 8)), _row(_at(14, 20))])

        result = merge_incremental(None, new, _START_DATE, _END_DATE)

        assert len(result) == len(new)
        assert result['start_time_utc'].tolist() == [
            pd.Timestamp(_at(14, 8)),
            pd.Timestamp(_at(14, 20)),
        ]

    def test_empty_new_returns_empty_schema_frame(self) -> None:
        """First run with an empty new frame yields the locked empty schema."""

        result = merge_incremental(None, _frame([]), _START_DATE, _END_DATE)

        assert len(result) == 0
        assert list(result.columns) == list(COLUMNS)
        assert result.dtypes.to_dict() == DTYPES


class TestWindowDeletion:
    """Existing rows kept or deleted based on the window predicate."""

    def test_out_of_window_existing_all_retained(self) -> None:
        """Existing rows entirely outside the window survive; new rows append."""

        existing = _frame([_row(_at(10, 8)), _row(_at(20, 8))])
        new = _frame([_row(_at(14, 9))])

        result = merge_incremental(existing, new, _START_DATE, _END_DATE)

        starts = result['start_time_utc'].tolist()
        assert pd.Timestamp(_at(10, 8)) in starts
        assert pd.Timestamp(_at(20, 8)) in starts
        assert pd.Timestamp(_at(14, 9)) in starts
        assert len(result) == len(existing) + len(new)

    def test_in_window_existing_replaced_by_new(self) -> None:
        """In-window existing rows are deleted and replaced by the new ones."""

        existing = _frame([_row(_at(14, 8), distance_miles=_OLD_DISTANCE)])
        new = _frame([_row(_at(14, 8), distance_miles=_NEW_DISTANCE)])

        result = merge_incremental(existing, new, _START_DATE, _END_DATE)

        assert len(result) == 1
        assert result.at[0, 'distance_miles'] == _NEW_DISTANCE

    def test_empty_new_deletes_window_keeps_outside(self) -> None:
        """Empty new frame deletes the window; out-of-window existing rows stay."""

        existing = _frame([_row(_at(14, 8)), _row(_at(20, 8))])

        result = merge_incremental(existing, _frame([]), _START_DATE, _END_DATE)

        assert len(result) == 1
        assert result.at[0, 'start_time_utc'] == pd.Timestamp(_at(20, 8))


class TestStartAnchoredNormalization:
    """The incoming filter normalizes overlap-anchored fetches to start-anchored."""

    def test_pre_window_new_row_is_dropped(self) -> None:
        """A new row starting before ``W_start`` is absent; in-window rows remain."""

        new = _frame([_row(_at(13, 8)), _row(_at(14, 9))])

        result = merge_incremental(None, new, _START_DATE, _END_DATE)

        assert len(result) == 1
        assert result.at[0, 'start_time_utc'] == pd.Timestamp(_at(14, 9))

    def test_cross_midnight_trailing_event_kept_once(self) -> None:
        """A new row starting in-window but ending after ``W_end`` is kept once."""

        new = _frame([_row(_at(14, 23), end=_at(15, 1))])

        result = merge_incremental(None, new, _START_DATE, _END_DATE)

        assert len(result) == 1
        assert result.at[0, 'start_time_utc'] == pd.Timestamp(_at(14, 23))
        assert result.at[0, 'end_time_utc'] == pd.Timestamp(_at(15, 1))


class TestBoundaryExactness:
    """Half-open ``[W_start, W_end)`` boundary handling for new and existing rows."""

    def test_new_row_at_w_start_kept(self) -> None:
        """A new row at exactly ``W_start`` is inside the window."""

        new = _frame([_row(_at(14, 0))])

        result = merge_incremental(None, new, _START_DATE, _END_DATE)

        assert len(result) == 1
        assert result.at[0, 'start_time_utc'] == _W_START

    def test_new_row_at_w_end_dropped(self) -> None:
        """A new row at exactly ``W_end`` is outside the half-open window."""

        new = _frame([_row(_at(15, 0))])

        result = merge_incremental(None, new, _START_DATE, _END_DATE)

        assert len(result) == 0

    def test_existing_at_w_start_deleted_at_w_end_retained(self) -> None:
        """Existing row at ``W_start`` is deleted; one at ``W_end`` is retained."""

        existing = _frame([_row(_at(14, 0)), _row(_at(15, 0))])

        result = merge_incremental(existing, _frame([]), _START_DATE, _END_DATE)

        assert len(result) == 1
        assert result.at[0, 'start_time_utc'] == _W_END


class TestDuplicateAcrossFrames:
    """Same event present in both frames must not duplicate."""

    def test_same_in_window_event_kept_once(self) -> None:
        """An event with the same in-window start in both frames yields one row."""

        existing = _frame([_row(_at(14, 8), distance_miles=_OLD_DISTANCE)])
        new = _frame([_row(_at(14, 8), distance_miles=_NEW_DISTANCE)])

        result = merge_incremental(existing, new, _START_DATE, _END_DATE)

        # Existing copy deleted, new copy appended -- exactly one survives.
        assert len(result) == 1
        assert result.at[0, 'distance_miles'] == _NEW_DISTANCE


class TestOutputShapeAndOrder:
    """The merged frame respects the locked schema and sort order."""

    def test_column_order_and_dtypes_match_schema(self) -> None:
        """Output columns are in ``COLUMNS`` order with ``DTYPES`` dtypes."""

        existing = _frame([_row(_at(20, 8))])
        new = _frame([_row(_at(14, 9))])

        result = merge_incremental(existing, new, _START_DATE, _END_DATE)

        assert list(result.columns) == list(COLUMNS)
        assert result.dtypes.to_dict() == DTYPES

    def test_result_sorted_with_clean_range_index(self) -> None:
        """Output is sorted null-company-first, then start, then event_type."""

        new = _frame(
            [
                _row(_at(14, 9), company='zzz_co'),
                _row(_at(14, 10), company=None),
                _row(_at(14, 8), company=None),
            ]
        )

        result = merge_incremental(None, new, _START_DATE, _END_DATE)

        assert pd.isna(result.at[0, 'company'])
        assert pd.isna(result.at[1, 'company'])
        assert result.at[2, 'company'] == 'zzz_co'
        # Null-company rows ordered by start_time among themselves.
        assert result.at[0, 'start_time_utc'] == pd.Timestamp(_at(14, 8))
        assert result.at[1, 'start_time_utc'] == pd.Timestamp(_at(14, 10))
        assert isinstance(result.index, pd.RangeIndex)
        assert result.index.tolist() == [0, 1, 2]


class TestContractValidation:
    """The pure function validates its own window contract."""

    def test_start_after_end_raises_value_error(self) -> None:
        """``start_date > end_date`` raises ``ValueError``."""

        with pytest.raises(ValueError, match='start_date'):
            merge_incremental(None, _frame([]), date(2026, 5, 15), date(2026, 5, 14))


class TestLogging:
    """The merge emits a single auditable DEBUG row-delta line."""

    def test_debug_line_reports_row_deltas(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A DEBUG line records incoming/kept/deleted/retained/final counts."""

        existing = _frame([_row(_at(14, 8)), _row(_at(20, 8))])
        new = _frame([_row(_at(13, 8)), _row(_at(14, 9))])

        with caplog.at_level(
            logging.DEBUG, logger='fleet_telemetry_hub._utilization_merge'
        ):
            merge_incremental(existing, new, _START_DATE, _END_DATE)

        debug_records = [
            record
            for record in caplog.records
            if 'merge_incremental:' in record.message
        ]
        assert len(debug_records) == 1
        message = debug_records[0].message
        assert 'incoming=2' in message
        assert 'kept_in_window=1' in message
        assert 'existing_deleted=1' in message
        assert 'existing_retained=1' in message
        assert 'final=2' in message
