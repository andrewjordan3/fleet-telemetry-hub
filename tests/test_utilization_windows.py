"""Tests for ``_utilization_windows.iter_windows``: the pure range-slicing generator.

The generator covers an inclusive ``[range_start, range_end]`` UTC-date
range as contiguous bounded windows that partition the range. It is pure
arithmetic -- no I/O -- so these tests assert window shape, contiguity,
and coverage directly from the yielded tuples.
"""

from datetime import date, timedelta
from itertools import pairwise

from fleet_telemetry_hub._utilization_windows import iter_windows

_ANCHOR = date(2026, 1, 1)


class TestSingleWindow:
    """Ranges that fit in one window yield exactly one ``(start, end)``."""

    def test_window_days_exceeds_span_yields_one_window(self) -> None:
        """A ``window_days`` larger than the span yields the whole range once."""

        range_end = _ANCHOR + timedelta(days=5)
        windows = list(iter_windows(_ANCHOR, range_end, window_days=100))

        assert windows == [(_ANCHOR, range_end)]

    def test_uncapped_whole_range_window_yields_one_window(self) -> None:
        """The uncapped case (``window_days`` == span) yields a single window."""

        range_end = _ANCHOR + timedelta(days=30)
        span_days = (range_end - _ANCHOR).days
        windows = list(iter_windows(_ANCHOR, range_end, window_days=span_days))

        assert windows == [(_ANCHOR, range_end)]

    def test_equal_start_and_end_yields_one_degenerate_window(self) -> None:
        """``range_start == range_end`` yields exactly one ``(d, d)`` window."""

        windows = list(iter_windows(_ANCHOR, _ANCHOR, window_days=28))

        assert windows == [(_ANCHOR, _ANCHOR)]


class TestEmptyRange:
    """An inverted range covers nothing."""

    def test_start_after_end_yields_nothing(self) -> None:
        """``range_start > range_end`` yields no windows."""

        windows = list(
            iter_windows(_ANCHOR, _ANCHOR - timedelta(days=1), window_days=28)
        )

        assert windows == []


class TestMultipleWindows:
    """A far-past start under a cap yields contiguous, advancing windows."""

    def test_contiguity_span_and_final_clamp(self) -> None:
        """Consecutive windows are contiguous; non-final span window_days exactly."""

        window_days = 28
        range_end = _ANCHOR + timedelta(days=70)
        windows = list(iter_windows(_ANCHOR, range_end, window_days))

        # More than one window for a span well past the cap.
        assert len(windows) > 1
        # First window opens at range_start.
        assert windows[0][0] == _ANCHOR
        # Consecutive windows are contiguous: the next window's start is
        # exactly one day after the prior window's end -- no overlap, no gap.
        for (_, prior_end), (next_start, _) in pairwise(windows):
            assert next_start == prior_end + timedelta(days=1)
        # Each non-final window spans exactly window_days.
        for window_start, window_end in windows[:-1]:
            assert (window_end - window_start).days == window_days
        # The final window's end is clamped exactly to range_end.
        assert windows[-1][1] == range_end

    def test_leading_edge_advances_by_window_plus_one(self) -> None:
        """Every non-final window advances the leading edge by window_days + 1."""

        window_days = 28
        range_end = _ANCHOR + timedelta(days=200)
        windows = list(iter_windows(_ANCHOR, range_end, window_days))

        # Compare each window's end to the next window's end: a full window
        # plus the one-day contiguous step is window_days + 1 of advance.
        ends = [window_end for _, window_end in windows]
        for prior_end, next_end in pairwise(ends):
            if next_end == range_end:
                break  # final window is clamped, not a full step
            assert (next_end - prior_end).days == window_days + 1

    def test_windows_partition_range_exactly_once(self) -> None:
        """The windows cover every day in [range_start, range_end] exactly once."""

        window_days = 14
        range_end = _ANCHOR + timedelta(days=90)
        windows = list(iter_windows(_ANCHOR, range_end, window_days))

        covered: set[date] = set()
        total_day_count: int = 0
        for window_start, window_end in windows:
            day = window_start
            while day <= window_end:
                covered.add(day)
                total_day_count += 1
                day += timedelta(days=1)

        expected = {
            _ANCHOR + timedelta(days=offset)
            for offset in range((range_end - _ANCHOR).days + 1)
        }
        # No gaps: every expected day is covered.
        assert covered == expected
        # No overlap: the summed per-window day counts equal the day count,
        # so no day was counted twice.
        assert total_day_count == len(expected)
