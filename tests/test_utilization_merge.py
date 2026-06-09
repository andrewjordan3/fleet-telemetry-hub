"""Tests for ``_utilization_merge.merge_incremental_to_parquet``: the DuckDB merge.

The merge reads the existing parquet at ``destination_path``, deletes the
run window, appends the window-sized new frame, globally sorts, and
atomically rewrites ``destination_path`` -- all in DuckDB, owning its own
temp file and rename. Cases write the "existing" frame straight to
``destination_path`` via ``build_dataframe(...).to_parquet(...)``, run the
merge, then read the result back from ``destination_path`` with the
canonical ``read_unified_parquet`` (the on-disk format is Arrow-native,
microsecond timestamps, no pandas metadata, so the canonical reader
restores the locked ``DTYPES``).

All identifiers are synthetic. The window under test is the single UTC
day 2026-05-14, i.e. ``W_start = 2026-05-14T00:00Z`` and
``W_end = 2026-05-15T00:00Z`` (exclusive).
"""

import logging
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from fleet_telemetry_hub._utilization_merge import (
    MergeStats,
    merge_incremental_to_parquet,
)
from fleet_telemetry_hub.unifier.schema import (
    COLUMNS,
    DTYPES,
    EventType,
    UnifiedEventRow,
    build_dataframe,
    read_unified_parquet,
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


def _write_existing(rows: list[UnifiedEventRow], destination_path: Path) -> None:
    """Write an existing-file frame straight to ``destination_path``.

    The merge now owns existing-file detection (it reads whatever sits at
    ``destination_path``), so the "existing" rows are laid down at the very
    path the merge will read from and atomically rewrite.
    """
    build_dataframe(rows).to_parquet(destination_path, index=False)


def _merge(
    existing_rows: list[UnifiedEventRow] | None,
    new_rows: list[UnifiedEventRow],
    tmp_path: Path,
) -> tuple[pd.DataFrame, MergeStats]:
    """Run a merge over the single-day window and read the result back.

    ``existing_rows is None`` models a first run (no existing file); ``[]``
    models an existing file with zero rows. The result is read via the
    canonical reader so dtype assertions hold against the Arrow-native
    on-disk format.
    """
    destination_path = tmp_path / 'data.parquet'
    if existing_rows is not None:
        _write_existing(existing_rows, destination_path)
    stats = merge_incremental_to_parquet(
        build_dataframe(new_rows),
        _START_DATE,
        _END_DATE,
        destination_path,
        compression='snappy',
    )
    return read_unified_parquet(destination_path), stats


class TestFirstRun:
    """``existing_path is None`` -- no file yet."""

    def test_non_empty_in_window_new_equals_new_rows(self, tmp_path: Path) -> None:
        """First run with in-window rows writes exactly those rows."""

        new_rows = [_row(_at(14, 8)), _row(_at(14, 20))]
        result, _ = _merge(None, new_rows, tmp_path)

        assert len(result) == len(new_rows)
        assert result['start_time_utc'].tolist() == [
            pd.Timestamp(_at(14, 8)),
            pd.Timestamp(_at(14, 20)),
        ]

    def test_empty_new_returns_empty_schema_frame(self, tmp_path: Path) -> None:
        """First run with an empty new frame writes a valid 0-row schema parquet."""

        result, _ = _merge(None, [], tmp_path)

        assert len(result) == 0
        assert list(result.columns) == list(COLUMNS)
        assert result.dtypes.to_dict() == DTYPES


class TestWindowDeletion:
    """Existing rows kept or deleted based on the window predicate."""

    def test_out_of_window_existing_all_retained(self, tmp_path: Path) -> None:
        """Existing rows entirely outside the window survive; new rows append."""

        existing_rows = [_row(_at(10, 8)), _row(_at(20, 8))]
        new_rows = [_row(_at(14, 9))]
        result, _ = _merge(existing_rows, new_rows, tmp_path)

        starts = result['start_time_utc'].tolist()
        assert pd.Timestamp(_at(10, 8)) in starts
        assert pd.Timestamp(_at(20, 8)) in starts
        assert pd.Timestamp(_at(14, 9)) in starts
        assert len(result) == len(existing_rows) + len(new_rows)

    def test_in_window_existing_replaced_by_new(self, tmp_path: Path) -> None:
        """In-window existing rows are deleted and replaced by the new ones."""

        result, _ = _merge(
            [_row(_at(14, 8), distance_miles=_OLD_DISTANCE)],
            [_row(_at(14, 8), distance_miles=_NEW_DISTANCE)],
            tmp_path,
        )

        assert len(result) == 1
        assert result.at[0, 'distance_miles'] == _NEW_DISTANCE

    def test_empty_new_deletes_window_keeps_outside(self, tmp_path: Path) -> None:
        """Empty new frame deletes the window; out-of-window existing rows stay."""

        result, _ = _merge([_row(_at(14, 8)), _row(_at(20, 8))], [], tmp_path)

        assert len(result) == 1
        assert result.at[0, 'start_time_utc'] == pd.Timestamp(_at(20, 8))


class TestStartAnchoredNormalization:
    """The incoming filter normalizes overlap-anchored fetches to start-anchored."""

    def test_pre_window_new_row_is_dropped(self, tmp_path: Path) -> None:
        """A new row starting before ``W_start`` is absent; in-window rows remain."""

        result, _ = _merge(None, [_row(_at(13, 8)), _row(_at(14, 9))], tmp_path)

        assert len(result) == 1
        assert result.at[0, 'start_time_utc'] == pd.Timestamp(_at(14, 9))

    def test_cross_midnight_trailing_event_kept_once(self, tmp_path: Path) -> None:
        """A new row starting in-window but ending after ``W_end`` is kept once."""

        result, _ = _merge(None, [_row(_at(14, 23), end=_at(15, 1))], tmp_path)

        assert len(result) == 1
        assert result.at[0, 'start_time_utc'] == pd.Timestamp(_at(14, 23))
        assert result.at[0, 'end_time_utc'] == pd.Timestamp(_at(15, 1))


class TestBoundaryExactness:
    """Half-open ``[W_start, W_end)`` boundary handling for new and existing rows."""

    def test_new_row_at_w_start_kept(self, tmp_path: Path) -> None:
        """A new row at exactly ``W_start`` is inside the window."""

        result, _ = _merge(None, [_row(_at(14, 0))], tmp_path)

        assert len(result) == 1
        assert result.at[0, 'start_time_utc'] == _W_START

    def test_new_row_at_w_end_dropped(self, tmp_path: Path) -> None:
        """A new row at exactly ``W_end`` is outside the half-open window."""

        result, _ = _merge(None, [_row(_at(15, 0))], tmp_path)

        assert len(result) == 0

    def test_existing_at_w_start_deleted_at_w_end_retained(
        self, tmp_path: Path
    ) -> None:
        """Existing row at ``W_start`` is deleted; one at ``W_end`` is retained."""

        result, _ = _merge([_row(_at(14, 0)), _row(_at(15, 0))], [], tmp_path)

        assert len(result) == 1
        assert result.at[0, 'start_time_utc'] == _W_END


class TestDuplicateAcrossFrames:
    """Same event present in both frames must not duplicate."""

    def test_same_in_window_event_kept_once(self, tmp_path: Path) -> None:
        """An event with the same in-window start in both frames yields one row."""

        result, _ = _merge(
            [_row(_at(14, 8), distance_miles=_OLD_DISTANCE)],
            [_row(_at(14, 8), distance_miles=_NEW_DISTANCE)],
            tmp_path,
        )

        # Existing copy deleted, new copy appended -- exactly one survives.
        assert len(result) == 1
        assert result.at[0, 'distance_miles'] == _NEW_DISTANCE


class TestIncomingDeduplication:
    """Exact duplicates within one incoming frame collapse to a single copy.

    Overlap-anchored chunk seams and pagination drift can put two
    identical copies of an event into a single run's ``new_frame``; the
    merge dedups the incoming side with ``SELECT DISTINCT`` so the output
    parquet never contains two identical rows.
    """

    def test_exact_duplicate_pair_written_once(self, tmp_path: Path) -> None:
        """A duplicate pair plus distinct rows writes one copy of each row."""

        duplicated = _row(_at(14, 8))
        distinct_rows = [duplicated, _row(_at(14, 9)), _row(_at(14, 20))]
        result, stats = _merge(None, [duplicated, *distinct_rows], tmp_path)

        assert len(result) == len(distinct_rows)
        assert result['start_time_utc'].tolist() == [
            pd.Timestamp(_at(14, 8)),
            pd.Timestamp(_at(14, 9)),
            pd.Timestamp(_at(14, 20)),
        ]
        assert stats.incoming_duplicates_dropped == 1
        assert stats.kept_in_window == len(distinct_rows)
        # final_row_count must match the file actually written, not just
        # the kept + retained formula.
        assert stats.final_row_count == len(result)

    def test_merge_branch_dedups_and_keeps_outside_rows(self, tmp_path: Path) -> None:
        """With an existing file, the duplicate pair still collapses and
        existing rows outside the window are untouched."""

        duplicated = _row(_at(14, 8))
        existing_rows = [_row(_at(10, 8)), _row(_at(20, 8))]
        result, stats = _merge(
            existing_rows,
            [duplicated, duplicated],
            tmp_path,
        )

        starts = result['start_time_utc'].tolist()
        assert starts.count(pd.Timestamp(_at(14, 8))) == 1
        assert pd.Timestamp(_at(10, 8)) in starts
        assert pd.Timestamp(_at(20, 8)) in starts
        assert len(result) == len(existing_rows) + 1
        assert stats.incoming_duplicates_dropped == 1
        assert stats.final_row_count == len(result)

    def test_near_duplicate_rows_both_survive(self, tmp_path: Path) -> None:
        """Rows identical except one column are not duplicates -- both survive.

        Pins the exact-duplicate-only scope: same-event-different-payload
        pairs (e.g. a provider-side edit shifting ``end_time_utc``) are
        intentionally not collapsed here.
        """

        new_rows = [
            _row(_at(14, 8), end=_at(14, 9)),
            _row(_at(14, 8), end=_at(14, 10)),
        ]
        result, stats = _merge(None, new_rows, tmp_path)

        assert len(result) == len(new_rows)
        assert stats.incoming_duplicates_dropped == 0
        assert stats.kept_in_window == len(new_rows)


class TestOutputShapeAndOrder:
    """The written parquet respects the locked schema and sort order."""

    def test_column_order_and_dtypes_match_schema(self, tmp_path: Path) -> None:
        """Output columns are in ``COLUMNS`` order with ``DTYPES`` dtypes.

        This is the gate proving the canonical reader bridges the
        Arrow-native / microsecond drift of a DuckDB-written file back to
        the locked in-memory ``DTYPES``.
        """

        result, _ = _merge([_row(_at(20, 8))], [_row(_at(14, 9))], tmp_path)

        assert list(result.columns) == list(COLUMNS)
        assert result.dtypes.to_dict() == DTYPES

    def test_result_sorted_with_clean_range_index(self, tmp_path: Path) -> None:
        """Output is sorted null-company-first, then start, then event_type."""

        result, _ = _merge(
            None,
            [
                _row(_at(14, 9), company='zzz_co'),
                _row(_at(14, 10), company=None),
                _row(_at(14, 8), company=None),
            ],
            tmp_path,
        )

        assert pd.isna(result.at[0, 'company'])
        assert pd.isna(result.at[1, 'company'])
        assert result.at[2, 'company'] == 'zzz_co'
        # Null-company rows ordered by start_time among themselves.
        assert result.at[0, 'start_time_utc'] == pd.Timestamp(_at(14, 8))
        assert result.at[1, 'start_time_utc'] == pd.Timestamp(_at(14, 10))
        assert isinstance(result.index, pd.RangeIndex)
        assert result.index.tolist() == [0, 1, 2]

    def test_dtype_round_trip_after_merge(self, tmp_path: Path) -> None:
        """A merged file read via the canonical reader matches ``DTYPES`` exactly."""

        result, _ = _merge(
            [_row(_at(20, 8))],
            [
                _row(_at(14, 8), company=None),
                _row(_at(14, 9), event_type=EventType.IDLE),
            ],
            tmp_path,
        )

        assert result.dtypes.to_dict() == DTYPES

    def test_sort_is_input_order_independent(self, tmp_path: Path) -> None:
        """Same rows in different input orders produce byte-identical output."""

        rows_forward = [
            _row(_at(14, 8), company='b_co'),
            _row(_at(14, 9), company='a_co'),
            _row(_at(14, 7), company=None),
        ]
        rows_reversed = list(reversed(rows_forward))

        out_a = tmp_path / 'a.parquet'
        out_b = tmp_path / 'b.parquet'
        merge_incremental_to_parquet(
            build_dataframe(rows_forward),
            _START_DATE,
            _END_DATE,
            out_a,
            compression='snappy',
        )
        merge_incremental_to_parquet(
            build_dataframe(rows_reversed),
            _START_DATE,
            _END_DATE,
            out_b,
            compression='snappy',
        )

        # The total-key ORDER BY makes the on-disk byte order reproducible.
        assert out_a.read_bytes() == out_b.read_bytes()


class TestContractValidation:
    """The function validates its own window contract."""

    def test_start_after_end_raises_value_error(self, tmp_path: Path) -> None:
        """``start_date > end_date`` raises ``ValueError``."""

        with pytest.raises(ValueError, match='start_date'):
            merge_incremental_to_parquet(
                build_dataframe([]),
                date(2026, 5, 15),
                date(2026, 5, 14),
                tmp_path / 'out.parquet',
                compression='snappy',
            )


class TestMergeStats:
    """The returned ``MergeStats`` counts match the scenario."""

    def test_stats_report_row_deltas(self, tmp_path: Path) -> None:
        """incoming / kept / deleted / retained / final reflect the merge."""

        # existing: one in-window (deleted), one out-of-window (retained).
        # new: one pre-window (dropped), one in-window (kept).
        _, stats = _merge(
            [_row(_at(14, 8)), _row(_at(20, 8))],
            [_row(_at(13, 8)), _row(_at(14, 9))],
            tmp_path,
        )

        assert stats == MergeStats(
            incoming=2,
            kept_in_window=1,
            incoming_duplicates_dropped=0,
            existing_deleted=1,
            existing_retained=1,
            final_row_count=2,
        )

    def test_first_run_stats_have_no_existing(self, tmp_path: Path) -> None:
        """A first run reports zero deleted / retained existing rows."""

        new_rows = [_row(_at(14, 8)), _row(_at(14, 9))]
        _, stats = _merge(None, new_rows, tmp_path)

        assert stats.existing_deleted == 0
        assert stats.existing_retained == 0
        assert stats.kept_in_window == len(new_rows)
        assert stats.final_row_count == len(new_rows)


class TestLogging:
    """The merge emits a single auditable DEBUG row-delta line."""

    def test_debug_line_reports_row_deltas(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A DEBUG line records incoming/kept/deleted/retained/final counts."""

        destination_path = tmp_path / 'data.parquet'
        _write_existing([_row(_at(14, 8)), _row(_at(20, 8))], destination_path)
        new_frame = build_dataframe([_row(_at(13, 8)), _row(_at(14, 9))])

        with caplog.at_level(
            logging.DEBUG, logger='fleet_telemetry_hub._utilization_merge'
        ):
            merge_incremental_to_parquet(
                new_frame,
                _START_DATE,
                _END_DATE,
                destination_path,
                compression='snappy',
            )

        debug_records = [
            record
            for record in caplog.records
            if 'merge_incremental:' in record.message
        ]
        assert len(debug_records) == 1
        message = debug_records[0].message
        assert 'incoming=2' in message
        assert 'kept_in_window=1' in message
        assert 'incoming_duplicates_dropped=0' in message
        assert 'existing_deleted=1' in message
        assert 'existing_retained=1' in message
        assert 'final=2' in message

    def test_warning_emitted_when_duplicates_dropped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A planted exact duplicate produces one WARNING naming count and window."""

        duplicated = _row(_at(14, 8))
        new_frame = build_dataframe([duplicated, duplicated])

        with caplog.at_level(
            logging.WARNING, logger='fleet_telemetry_hub._utilization_merge'
        ):
            merge_incremental_to_parquet(
                new_frame,
                _START_DATE,
                _END_DATE,
                tmp_path / 'data.parquet',
                compression='snappy',
            )

        warning_records = [
            record for record in caplog.records if record.levelno == logging.WARNING
        ]
        assert len(warning_records) == 1
        message = warning_records[0].message
        assert 'dropped 1 exact duplicate' in message
        assert '2026-05-14T00:00:00+00:00' in message
        assert '2026-05-15T00:00:00+00:00' in message


class TestWriteTransaction:
    """The merge owns its own temp file, atomic rename, and directory creation."""

    def test_mid_write_failure_preserves_existing_and_leaves_no_tmp(
        self, tmp_path: Path
    ) -> None:
        """A failure after the COPY but before the rename leaves things pristine.

        The COPY has written the temp file by the time stats are computed;
        patching ``_compute_stats`` to raise exercises the ``try/finally``,
        which must remove that temp file and leave the existing destination
        byte-identical (the rename never ran).
        """

        destination_path = tmp_path / 'data.parquet'
        _write_existing([_row(_at(20, 8))], destination_path)
        original_bytes = destination_path.read_bytes()

        with (
            patch(
                'fleet_telemetry_hub._utilization_merge._compute_stats',
                side_effect=RuntimeError('mid-write boom'),
            ),
            pytest.raises(RuntimeError, match='mid-write boom'),
        ):
            merge_incremental_to_parquet(
                build_dataframe([_row(_at(14, 9))]),
                _START_DATE,
                _END_DATE,
                destination_path,
                compression='snappy',
            )

        assert destination_path.read_bytes() == original_bytes
        assert list(tmp_path.glob('*.tmp')) == []

    def test_creates_destination_parent_when_absent(self, tmp_path: Path) -> None:
        """The merge creates ``destination_path.parent`` when it does not exist."""

        destination_path = tmp_path / 'nested' / 'sub' / 'data.parquet'
        assert not destination_path.parent.exists()

        merge_incremental_to_parquet(
            build_dataframe([_row(_at(14, 8))]),
            _START_DATE,
            _END_DATE,
            destination_path,
            compression='snappy',
        )

        assert destination_path.is_file()
        result = read_unified_parquet(destination_path)
        assert len(result) == 1
