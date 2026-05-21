"""Pure interval-overlap math for the unifier's per-provider transforms.

These helpers do no I/O, no logging, and no global state. They take
tz-aware datetimes (or raise) and return integer seconds. The
transform layer composes them into "compute driving event duration
minus overlapping idle" and "attribute an idle event to a driver."
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

__all__: list[str] = [
    'DriverIdentity',
    'DrivingWindow',
    'attribute_idle_driver',
    'clip_overlap_seconds',
    'compute_driving_duration_seconds',
    'sum_overlap_seconds',
]

DriverIdentity = tuple[str | None, str | None]
"""(driver_id, driver_name). Either or both may be None."""

_NULL_IDENTITY: DriverIdentity = (None, None)


def _require_tz_aware(name: str, value: datetime) -> None:
    """Raise TypeError if ``value`` is a naive datetime."""
    if value.tzinfo is None:
        raise TypeError(f'{name} must be a tz-aware datetime, got naive: {value!r}')


@dataclass(frozen=True, slots=True)
class DrivingWindow:
    """
    Anonymous driving-event window for overlap math.

    Provider-agnostic intermediate. The transform layer constructs
    these from Motive ``DrivingPeriod`` and Samsara ``Trip`` records
    so the overlap helpers can stay structurally identical for both
    providers.

    Validates on construction:
        - ``start`` and ``end`` must both be tz-aware datetimes
          (``TypeError`` on naive input).
        - ``start <= end`` (``ValueError`` on reversed input).
        - ``start == end`` (zero-duration) is allowed and contributes
          ``0`` to all overlap math.

    Attributes:
        start: Window start instant; must be tz-aware.
        end: Window end instant; must be tz-aware. Inclusive of start,
            exclusive of end for overlap purposes.
        driver: ``(driver_id, driver_name)`` for the driver attributed
            to this window. Either or both may be ``None``.
    """

    start: datetime
    end: datetime
    driver: DriverIdentity

    def __post_init__(self) -> None:
        """Validate the constructed window."""
        _require_tz_aware('DrivingWindow.start', self.start)
        _require_tz_aware('DrivingWindow.end', self.end)
        if self.start > self.end:
            raise ValueError(
                f'DrivingWindow.start ({self.start}) must be <= end ({self.end})'
            )


def clip_overlap_seconds(
    a_start: datetime,
    a_end: datetime,
    b_start: datetime,
    b_end: datetime,
) -> int:
    """
    Integer seconds of overlap between two half-open intervals ``[start, end)``.

    Returns ``0`` for disjoint or merely adjacent intervals
    (``a.end == b.start`` or vice versa). Zero-duration windows
    (``start == end``) are allowed and contribute ``0``. Microsecond
    precision is truncated via ``int(...)``.

    Args:
        a_start: Start of the first interval (tz-aware).
        a_end: End of the first interval (tz-aware), ``>= a_start``.
        b_start: Start of the second interval (tz-aware).
        b_end: End of the second interval (tz-aware), ``>= b_start``.

    Returns:
        Non-negative integer seconds of overlap (truncated).

    Raises:
        TypeError: If any input is a naive datetime.
        ValueError: If either interval has ``start > end``.
    """
    _require_tz_aware('a_start', a_start)
    _require_tz_aware('a_end', a_end)
    _require_tz_aware('b_start', b_start)
    _require_tz_aware('b_end', b_end)
    if a_start > a_end:
        raise ValueError(f'a_start ({a_start}) must be <= a_end ({a_end})')
    if b_start > b_end:
        raise ValueError(f'b_start ({b_start}) must be <= b_end ({b_end})')

    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    if overlap_end <= overlap_start:
        return 0
    return int((overlap_end - overlap_start).total_seconds())


def sum_overlap_seconds(
    target_start: datetime,
    target_end: datetime,
    other_windows: Iterable[tuple[datetime, datetime]],
) -> int:
    """
    Sum of clipped overlap between a target window and each other window.

    Each ``(start, end)`` tuple in ``other_windows`` is clipped to
    the target window, and the clipped seconds are summed.
    ``other_windows`` is consumed once.

    **No deduplication.** If two windows in ``other_windows`` overlap
    each other within the target, their overlap with the target is
    counted twice. This is intentional for V1: well-formed ELD data
    does not emit overlapping idle events on the same vehicle, and
    union-based interval merging is deferred. The transform layer's
    "drop if computed driving duration <= 0" rule absorbs the harm
    in the rare malformed-data case.

    Args:
        target_start: Start of the window to clip against (tz-aware).
        target_end: End of the window to clip against (tz-aware),
            ``>= target_start``.
        other_windows: Iterable of ``(start, end)`` tuples to overlap
            with the target. Consumed once.

    Returns:
        Non-negative integer total of clipped overlap seconds.

    Raises:
        TypeError: If any datetime is naive.
        ValueError: If any interval has ``start > end``.
    """
    return sum(
        clip_overlap_seconds(target_start, target_end, other_start, other_end)
        for other_start, other_end in other_windows
    )


def compute_driving_duration_seconds(
    driving_start: datetime,
    driving_end: datetime,
    idle_windows: Iterable[tuple[datetime, datetime]],
) -> int:
    """
    Driving event duration minus overlapping idle, in integer seconds.

    Computes ``int((driving_end - driving_start).total_seconds())``
    minus the sum of clipped idle overlap. The result may be zero or
    negative if idle fully covers the driving window (pathological
    data); this helper does not clamp. The caller's transform layer
    decides whether to drop such rows.

    Args:
        driving_start: Start of the driving event (tz-aware).
        driving_end: End of the driving event (tz-aware),
            ``>= driving_start``.
        idle_windows: Iterable of ``(start, end)`` tuples for idle
            events on the same vehicle that may overlap the driving
            event. Consumed once.

    Returns:
        Integer seconds of driving time after subtracting overlapping
        idle. May be zero or negative.

    Raises:
        TypeError: If any datetime is naive.
        ValueError: If any interval has ``start > end``.
    """
    _require_tz_aware('driving_start', driving_start)
    _require_tz_aware('driving_end', driving_end)
    if driving_start > driving_end:
        raise ValueError(
            f'driving_start ({driving_start}) must be <= driving_end ({driving_end})'
        )
    total_seconds = int((driving_end - driving_start).total_seconds())
    overlap_seconds = sum_overlap_seconds(driving_start, driving_end, idle_windows)
    return total_seconds - overlap_seconds


def attribute_idle_driver(
    idle_start: datetime,
    idle_end: datetime,
    driving_windows: Iterable[DrivingWindow],
) -> tuple[DriverIdentity, dict[DriverIdentity, int], bool]:
    """
    Pick the driver to attribute to an idle event by most overlap.

    Partitions the idle window's duration into buckets keyed by
    ``DriverIdentity``, including an implicit ``(None, None)`` bucket
    for the portion of the idle that no driving window covered. The
    driver with the largest total overlap wins; the ``(None, None)``
    bucket is a valid winner if uncovered time exceeds every other
    candidate.

    Ties among non-null candidates are broken by the earliest
    ``DrivingWindow.start`` among the tied candidates. The
    ``(None, None)`` bucket is never preferred in a tie against a
    non-null candidate (it has no start time to compare on).

    Args:
        idle_start: Start of the idle event (tz-aware).
        idle_end: End of the idle event (tz-aware),
            ``>= idle_start``.
        driving_windows: Iterable of ``DrivingWindow`` instances on
            the same vehicle that may overlap the idle event.
            Consumed once.

    Returns:
        ``(winner_identity, full_bucket_distribution, warn_flag)``:

        - ``winner_identity``: The ``DriverIdentity`` attributed to
          the idle. May be ``(None, None)``.
        - ``full_bucket_distribution``: ``dict[DriverIdentity, int]``
          mapping every encountered identity (including
          ``(None, None)`` if non-zero) to its overlap seconds.
          Useful for logging context at the caller.
        - ``warn_flag``: ``True`` if two or more **unique non-null**
          ``DriverIdentity`` keys in the bucket distribution have
          non-zero overlap with the idle. The ``(None, None)``
          bucket is never counted toward this flag. ``False`` for
          zero or one unique non-null candidate.

    Raises:
        TypeError: If any datetime is naive.
        ValueError: If any interval has ``start > end``.
    """
    _require_tz_aware('idle_start', idle_start)
    _require_tz_aware('idle_end', idle_end)
    if idle_start > idle_end:
        raise ValueError(f'idle_start ({idle_start}) must be <= idle_end ({idle_end})')

    idle_total_seconds = int((idle_end - idle_start).total_seconds())
    seconds_by_driver: dict[DriverIdentity, int] = {}
    earliest_start_by_driver: dict[DriverIdentity, datetime] = {}

    for window in driving_windows:
        overlap = clip_overlap_seconds(idle_start, idle_end, window.start, window.end)
        if overlap == 0:
            continue
        seconds_by_driver[window.driver] = (
            seconds_by_driver.get(window.driver, 0) + overlap
        )
        existing_earliest = earliest_start_by_driver.get(window.driver)
        if existing_earliest is None or window.start < existing_earliest:
            earliest_start_by_driver[window.driver] = window.start

    covered_seconds = sum(seconds_by_driver.values())
    uncovered_seconds = idle_total_seconds - covered_seconds
    if uncovered_seconds > 0:
        seconds_by_driver[_NULL_IDENTITY] = uncovered_seconds

    non_null_candidate_count = sum(
        1
        for identity, seconds in seconds_by_driver.items()
        if identity != _NULL_IDENTITY and seconds > 0
    )
    warn_flag = non_null_candidate_count >= 2  # noqa: PLR2004 -- "two or more"

    winner = _pick_winner(seconds_by_driver, earliest_start_by_driver)
    return winner, seconds_by_driver, warn_flag


def _pick_winner(
    seconds_by_driver: dict[DriverIdentity, int],
    earliest_start_by_driver: dict[DriverIdentity, datetime],
) -> DriverIdentity:
    """
    Choose the winning identity from a populated bucket distribution.

    Most overlap wins. Non-null candidates beat ``(None, None)`` on
    ties (``(None, None)`` has no start time and so cannot win a
    tie-break). Among tied non-null candidates, the earliest
    ``DrivingWindow.start`` wins.
    """
    if not seconds_by_driver:
        return _NULL_IDENTITY

    max_seconds = max(seconds_by_driver.values())
    tied = [
        identity
        for identity, seconds in seconds_by_driver.items()
        if seconds == max_seconds
    ]
    if len(tied) == 1:
        return tied[0]

    non_null_tied = [identity for identity in tied if identity != _NULL_IDENTITY]
    if not non_null_tied:
        return _NULL_IDENTITY
    return min(non_null_tied, key=lambda identity: earliest_start_by_driver[identity])
