# pyright: reportUnknownVariableType=false
"""Incremental delete-then-append merge for ``UtilizationPipeline``.

Internal companion module to ``utilization_pipeline.py``: holds the
pure, side-effect-free logic that merges a freshly fetched unified
frame into the existing on-disk frame under the half-open UTC fetch
window. The orchestrator owns reading and writing parquet; this module
only transforms in-memory DataFrames so the merge semantics can be
exercised in isolation.

The module-level pyright suppression mirrors the convention in the
sibling ``_utilization_metadata.py``: boolean-mask row selection and
``pd.concat`` return partially-Unknown types in strict mode, and the
legible alternative is a swarm of ``cast`` calls that CLAUDE.md
discourages. The pandas boundary here is small -- one mask, one concat,
one ``astype`` -- so the trade-off is favorable.

The merge normalizes every provider to start-anchored on our side: the
incoming frame is filtered to ``start_time_utc`` inside the window
before it is appended, and the same window is deleted from the existing
frame. Samsara's ``/v1/fleet/trips`` endpoint is overlap-anchored -- it
returns any trip intersecting the query window, including trips that
started before it -- while Motive ``driving_periods`` and Samsara
``/idling/events`` are start-anchored. Filtering on start time is what
keeps a cross-boundary event from being duplicated at the leading edge
of each run. Events that start before the window are dropped from the
incoming frame by design: their single authoritative copy already lives
in the file under the earlier window that owns their start.
"""

import logging
from datetime import date, timedelta

import pandas as pd

from fleet_telemetry_hub.unifier.schema import (
    COLUMNS,
    DTYPES,
    build_dataframe,
    sort_unified_frame,
)

__all__: list[str] = ['merge_incremental']

logger: logging.Logger = logging.getLogger(__name__)


def merge_incremental(
    existing: pd.DataFrame | None,
    new_frame: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Merge a freshly fetched unified frame into the existing one by window.

    Implements the incremental delete-then-append model: the half-open
    UTC window ``[W_start, W_end)`` is deleted from ``existing`` and
    replaced by the rows of ``new_frame`` whose ``start_time_utc`` falls
    inside that window. Filtering the incoming frame to in-window starts
    normalizes every provider to start-anchored, so an overlap-anchored
    fetch cannot duplicate a cross-boundary event at the leading edge.
    Events starting before ``W_start`` are dropped from the incoming
    frame by design -- their authoritative copy already lives in the file
    under the earlier window that owns their start.

    Both inputs are assumed to already carry the schema ``COLUMNS`` with
    the schema dtypes; validating an on-disk frame's shape is the
    orchestrator's responsibility. An empty ``new_frame`` is valid input:
    the window is deleted from ``existing`` and nothing is appended.

    Args:
        existing: The current on-disk unified frame, or ``None`` on a
            first run with no file yet.
        new_frame: The freshly fetched unified frame for this run.
        start_date: Inclusive UTC start date of the fetch window.
        end_date: Inclusive UTC end date of the fetch window.

    Returns:
        A new unified DataFrame with the window's rows replaced, sorted
        by ``SORT_COLUMNS`` with a clean ``RangeIndex`` and normalized to
        the locked ``COLUMNS`` order and ``DTYPES``.

    Raises:
        ValueError: If ``start_date > end_date``.

    Side Effects:
        Emits a single ``DEBUG`` log line recording the row deltas
        (incoming, kept, deleted, retained, final). Performs no I/O.
    """
    if start_date > end_date:
        raise ValueError(f'start_date ({start_date}) must be <= end_date ({end_date})')

    window_start, window_end = _window_bounds(start_date, end_date)

    filtered_new = new_frame.loc[
        _start_in_window_mask(new_frame, window_start, window_end)
    ]

    if existing is None:
        retained = build_dataframe([])
    else:
        retained = existing.loc[
            ~_start_in_window_mask(existing, window_start, window_end)
        ]

    merged = pd.concat([retained, filtered_new], ignore_index=True)
    sorted_frame = sort_unified_frame(merged)
    normalized = sorted_frame[list(COLUMNS)].astype(DTYPES)

    existing_deleted = 0 if existing is None else len(existing) - len(retained)
    logger.debug(
        'merge_incremental: incoming=%d kept_in_window=%d '
        'existing_deleted=%d existing_retained=%d final=%d',
        len(new_frame),
        len(filtered_new),
        existing_deleted,
        len(retained),
        len(normalized),
    )
    return normalized


def _window_bounds(
    start_date: date, end_date: date
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Compute the half-open UTC window ``[W_start, W_end)`` for the fetch dates.

    Args:
        start_date: Inclusive UTC start date of the fetch window.
        end_date: Inclusive UTC end date of the fetch window.

    Returns:
        ``(window_start, window_end)`` where ``window_start`` is midnight
        UTC of ``start_date`` and ``window_end`` is midnight UTC of the
        day after ``end_date`` (the exclusive upper bound).
    """
    window_start = pd.Timestamp(start_date, tz='UTC')
    window_end = pd.Timestamp(end_date + timedelta(days=1), tz='UTC')
    return window_start, window_end


def _start_in_window_mask(
    frame: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.Series:
    """Boolean mask of rows whose ``start_time_utc`` lies in ``[window_start, window_end)``.

    Shared by both the incoming-frame filter and the existing-row
    deletion so the two window predicates can never drift apart.

    Args:
        frame: A unified frame carrying the ``start_time_utc`` column.
        window_start: Inclusive lower bound (tz-aware UTC).
        window_end: Exclusive upper bound (tz-aware UTC).

    Returns:
        Boolean Series aligned to ``frame``, ``True`` where
        ``start_time_utc >= window_start`` and ``< window_end``.
    """
    start_times: pd.Series = frame['start_time_utc']
    return (start_times >= window_start) & (start_times < window_end)
