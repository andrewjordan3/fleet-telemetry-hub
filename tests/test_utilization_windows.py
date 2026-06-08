"""Tests for ``_utilization_windows.iter_windows``: the pure range-slicing generator.

The generator covers an inclusive ``[range_start, range_end]`` UTC-date
range as bounded windows that overlap by ``lookback_days``. It is pure
arithmetic -- no I/O -- so these tests assert window shape, overlap, and
coverage directly from the yielded tuples.
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
        windows = list(iter_windows(_ANCHOR, range_end, window_days=100, lookback_days=7))

        assert windows == [(_ANCHOR, range_end)]

    def test_uncapped_whole_range_window_yields_one_window(self) -> None:
        """The uncapped case (``window_days`` == span) yields a single window."""

        range_end = _ANCHOR + timedelta(days=30)
        span_days = (range_end - _ANCHOR).days
        windows = list(
            iter_windows(_ANCHOR, range_end, window_days=span_days, lookback_days=7)
        )

        assert windows == [(_ANCHOR, range_end)]

    def test_equal_start_and_end_yields_one_degenerate_window(self) -> None:
        """``range_start == range_end`` yields exactly one ``(d, d)`` window."""

        windows = list(
            iter_windows(_ANCHOR, _ANCHOR, window_days=28, lookback_days=7)
        )

        assert windows == [(_ANCHOR, _ANCHOR)]


class TestEmptyRange:
    """An inverted range covers nothing."""

    def test_start_after_end_yields_nothing(self) -> None:
        """``range_start > range_end`` yields no windows."""

        windows = list(
            iter_windows(
                _ANCHOR, _ANCHOR - timedelta(days=1), window_days=28, lookback_days=7
            )
        )

        assert windows == []


class TestMultipleWindows:
    """A far-past start under a cap yields overlapping, advancing windows."""

    def test_overlap_span_and_final_clamp(self) -> None:
        """Consecutive windows overlap by lookback; non-final span window_days exactly."""

        window_days = 28
        lookback_days = 7
        range_end = _ANCHOR + timedelta(days=70)
        windows = list(
            iter_windows(_ANCHOR, range_end, window_days, lookback_days)
        )

        # More than one window for a span well past the cap.
        assert len(windows) > 1
        # First window opens at range_start.
        assert windows[0][0] == _ANCHOR
        # Each non-final window spans exactly window_days; the next window's
        # start steps back lookback_days from the prior window's end.
        for (_, prior_end), (next_start, _) in pairwise(windows):
            assert next_start == prior_end - timedelta(days=lookback_days)
        for window_start, window_end in windows[:-1]:
            assert (window_end - window_start).days == window_days
        # The final window's end is clamped exactly to range_end.
        assert windows[-1][1] == range_end

    def test_leading_edge_advances_by_window_minus_lookback(self) -> None:
        """Every window after the first advances the leading edge by the net step."""

        window_days = 28
        lookback_days = 7
        range_end = _ANCHOR + timedelta(days=200)
        windows = list(
            iter_windows(_ANCHOR, range_end, window_days, lookback_days)
        )

        net_advance = window_days - lookback_days
        # Compare each non-final window's end to the next non-final window's
        # end: the leading edge moves forward by exactly window_days - lookback.
        ends = [window_end for _, window_end in windows]
        for prior_end, next_end in pairwise(ends):
            if next_end == range_end:
                break  # final window is clamped, not a full net step
            assert (next_end - prior_end).days == net_advance

    def test_union_covers_range_with_no_gaps(self) -> None:
        """The union of all windows covers every day in [range_start, range_end]."""

        window_days = 14
        lookback_days = 3
        range_end = _ANCHOR + timedelta(days=90)
        windows = list(
            iter_windows(_ANCHOR, range_end, window_days, lookback_days)
        )

        covered: set[date] = set()
        for window_start, window_end in windows:
            day = window_start
            while day <= window_end:
                covered.add(day)
                day += timedelta(days=1)

        expected = {
            _ANCHOR + timedelta(days=offset)
            for offset in range((range_end - _ANCHOR).days + 1)
        }
        assert covered == expected
