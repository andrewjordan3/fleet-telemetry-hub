"""Pure windowing generator for the utilization pipeline's range march.

A single run covers an inclusive ``[range_start, range_end]`` UTC-date
range as a sequence of bounded windows. This module owns only the
arithmetic that slices the range; it performs no I/O and imports nothing
from the pipeline. The pipeline resolves the range from persisted
metadata and config, then walks the windows this generator yields.

Termination is structural, not guarded: each non-final window's leading
edge advances by ``window_days - lookback_days`` days (guaranteed
positive for a set ``max_window_days`` by the config validator) and is
bounded above by ``range_end``, so the generator yields finitely many
windows and exhausts on its own.
"""

from collections.abc import Iterator
from datetime import date, timedelta

__all__: list[str] = ['iter_windows']


def iter_windows(
    range_start: date,
    range_end: date,
    window_days: int,
    lookback_days: int,
) -> Iterator[tuple[date, date]]:
    """Yield inclusive (start, end) UTC-date windows covering [range_start, range_end].

    Each window spans at most ``window_days`` days. Consecutive windows
    overlap by ``lookback_days`` -- the next window's start steps back
    ``lookback_days`` from the prior window's end -- so late-arriving edits in
    the trailing lookback are re-fetched and re-merged. Every window after the
    first thus advances the leading edge by ``window_days - lookback_days``
    days of new ground. The final window's end is clamped to ``range_end``.

    Yields nothing when ``range_start > range_end`` (nothing to cover).

    Precondition (multi-window ranges only): ``window_days > lookback_days``,
    so the leading edge advances. The config validator guarantees this for a
    set ``max_window_days``; the single-window case (``window_days`` spanning
    the whole range) terminates after the first yield regardless and is safe
    for any ``lookback_days``.
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
        current_start = current_end - timedelta(days=lookback_days)
