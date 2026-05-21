"""Tests for the pure date-range chunking helper.

The helper is provider-agnostic: no I/O, no logging, no global
state. Tests cover both ``chunk_format`` output forms, the
inclusive-end-at-date-grain contract, contiguity at instant grain
for the datetime form, edge cases (single-day range, exact-boundary
range), input validation, and a year-long backfill realism case.
"""

from datetime import UTC, date, datetime
from itertools import pairwise

import pytest

from fleet_telemetry_hub.utilization.date_chunking import iter_chunks

# Backfill-shaped landmarks reused across tests.
_JAN_1 = date(2024, 1, 1)
_JAN_7 = date(2024, 1, 7)
_JAN_28 = date(2024, 1, 28)
_JAN_29 = date(2024, 1, 29)
_FEB_1 = date(2024, 2, 1)
_DEC_31 = date(2024, 12, 31)
_NEXT_JAN_1 = date(2025, 1, 1)

_MAX_28 = 28
_DAYS_PER_YEAR_2024 = 366  # Leap year; used to compute expected chunk count.


class TestIterChunksDateFormat:
    """``chunk_format='date'`` yields inclusive ``(date, date)`` tuples."""

    def test_single_chunk_when_range_fits(self) -> None:
        """A short range produces exactly one tuple covering the whole range."""

        chunks = list(iter_chunks(_JAN_1, _JAN_7, _MAX_28, chunk_format='date'))

        assert chunks == [(_JAN_1, _JAN_7)]

    def test_exact_boundary_is_single_chunk(self) -> None:
        """A range exactly ``max_days`` long fits in one chunk."""

        chunks = list(iter_chunks(_JAN_1, _JAN_28, _MAX_28, chunk_format='date'))

        assert chunks == [(_JAN_1, _JAN_28)]

    def test_multi_chunk_is_contiguous_and_inclusive(self) -> None:
        """Multi-chunk ranges are contiguous at the date grain with inclusive ends."""

        chunks = list(iter_chunks(_JAN_1, _FEB_1, _MAX_28, chunk_format='date'))

        assert chunks == [(_JAN_1, _JAN_28), (_JAN_29, _FEB_1)]

    def test_same_day_range_yields_single_tuple(self) -> None:
        """``start == end`` produces exactly one ``(start, end)`` chunk."""

        chunks = list(iter_chunks(_JAN_1, _JAN_1, _MAX_28, chunk_format='date'))

        assert chunks == [(_JAN_1, _JAN_1)]


class TestIterChunksDatetimeFormat:
    """``chunk_format='datetime'`` yields tz-aware UTC half-open windows."""

    def test_single_chunk_end_is_day_after_last_date(self) -> None:
        """``chunk_end_dt`` is ``00:00 UTC`` of the day after the chunk's last date."""

        chunks = list(iter_chunks(_JAN_1, _JAN_7, _MAX_28, chunk_format='datetime'))

        assert chunks == [
            (
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 8, tzinfo=UTC),
            ),
        ]

    def test_multi_chunk_contiguous_at_instant_grain(self) -> None:
        """Adjacent chunks' end_dt and start_dt are equal at the instant grain."""

        chunks = list(iter_chunks(_JAN_1, _FEB_1, _MAX_28, chunk_format='datetime'))

        assert len(chunks) > 1
        for previous_chunk, next_chunk in pairwise(chunks):
            assert previous_chunk[1] == next_chunk[0]

    def test_all_datetimes_are_tz_aware_utc(self) -> None:
        """Every yielded datetime carries ``tzinfo`` identical to ``UTC``."""

        chunks = list(iter_chunks(_JAN_1, _FEB_1, _MAX_28, chunk_format='datetime'))

        for chunk_start, chunk_end in chunks:
            assert chunk_start.tzinfo is UTC
            assert chunk_end.tzinfo is UTC


class TestIterChunksValidation:
    """Input validation: bad ranges and invalid ``max_days`` raise ``ValueError``."""

    def test_reversed_range_raises(self) -> None:
        """``start > end`` raises ``ValueError`` with both dates in the message."""

        with pytest.raises(ValueError, match='must be <='):
            list(iter_chunks(_FEB_1, _JAN_1, _MAX_28, chunk_format='date'))

    @pytest.mark.parametrize('bad_max_days', [0, -1])
    def test_non_positive_max_days_raises(self, bad_max_days: int) -> None:
        """``max_days <= 0`` raises ``ValueError``."""

        with pytest.raises(ValueError, match='max_days'):
            list(iter_chunks(_JAN_1, _FEB_1, bad_max_days, chunk_format='date'))

    def test_invalid_chunk_format_raises(self) -> None:
        """An out-of-band ``chunk_format`` value defensively raises ``ValueError``."""

        # Bypass the static Literal check to exercise the runtime guard.
        with pytest.raises(ValueError, match='chunk_format'):
            list(iter_chunks(_JAN_1, _JAN_7, _MAX_28, chunk_format='weeks'))  # pyright: ignore[reportArgumentType, reportCallIssue]


class TestIterChunksYearLongBackfill:
    """A year-long realistic backfill chunks cleanly under the 28-day cap."""

    def test_year_long_range_chunks_cleanly(self) -> None:
        """First chunk starts at Jan 1 00Z; last chunk ends at next Jan 1 00Z."""

        chunks = list(iter_chunks(_JAN_1, _DEC_31, _MAX_28, chunk_format='datetime'))

        # First chunk start anchors to Jan 1 00Z; last chunk ends at next Jan 1 00Z.
        assert chunks[0][0] == datetime(2024, 1, 1, tzinfo=UTC)
        assert chunks[-1][1] == datetime(_NEXT_JAN_1.year, 1, 1, tzinfo=UTC)

        # Contiguous at instant grain.
        for previous_chunk, next_chunk in pairwise(chunks):
            assert previous_chunk[1] == next_chunk[0]

        # 366 days / 28 days-per-chunk = 14 chunks (last is short).
        expected_chunk_count = (_DAYS_PER_YEAR_2024 + _MAX_28 - 1) // _MAX_28
        assert len(chunks) == expected_chunk_count

    def test_year_long_range_no_chunk_exceeds_max_days(self) -> None:
        """At date grain, no chunk spans more than ``max_days`` inclusive days."""

        chunks = list(iter_chunks(_JAN_1, _DEC_31, _MAX_28, chunk_format='datetime'))

        for chunk_start_dt, chunk_end_dt in chunks:
            # End is exclusive at instant grain; inclusive date span is delta.days.
            inclusive_day_span = (chunk_end_dt - chunk_start_dt).days
            assert inclusive_day_span <= _MAX_28
