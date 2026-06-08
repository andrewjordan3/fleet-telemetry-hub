"""Pure windowing generator for the utilization pipeline's range march.

A single run covers an inclusive ``[range_start, range_end]`` UTC-date
range as a sequence of contiguous bounded windows. This module owns only
the arithmetic that slices the range; it performs no I/O and imports
nothing from the pipeline. The pipeline resolves the range from persisted
metadata and config, then walks the windows this generator yields.

Termination is structural, not guarded: each non-final window's leading
edge advances by ``window_days + 1`` days (the full window width plus the
one-day step to the next contiguous window) and is bounded above by
``range_end``, so the generator yields finitely many windows and exhausts
on its own.
"""

from collections.abc import Iterator
from datetime import date, timedelta

__all__: list[str] = ['iter_windows']


def iter_windows(
    range_start: date,
    range_end: date,
    window_days: int,
) -> Iterator[tuple[date, date]]:
    """Yield inclusive (start, end) UTC-date windows covering [range_start, range_end].

    Each window spans at most ``window_days`` days. Consecutive windows are
    contiguous -- the next window's start is the day after the prior
    window's end -- so the windows partition the range: every day is
    fetched exactly once across the march, with no overlap and no gap.
    Every window after the first thus advances the leading edge by
    ``window_days + 1`` days. The final window's end is clamped to
    ``range_end``.

    Yields nothing when ``range_start > range_end`` (nothing to cover).

    Precondition: ``window_days >= 1``, so ``current_end`` strictly
    increases and, bounded above by ``range_end``, the loop terminates. A
    set ``max_window_days`` is a positive int per the config validator, and
    the uncapped effective window is ``max(span, 1)``, so this always holds.
    """
    if range_start > range_end:
        return
    current_start: date = range_start
    while True:
        current_end: date = min(
            current_start + timedelta(days=window_days), range_end
        )
        yield (current_start, current_end)
        if current_end >= range_end:
            return
        current_start = current_end + timedelta(days=1)

