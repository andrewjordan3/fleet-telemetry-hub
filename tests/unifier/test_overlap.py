"""Tests for ``unifier.overlap`` helpers and the ``DrivingWindow`` value type.

Covers interval-overlap math at the second grain, ``DrivingWindow``
construction-time validation, driving-duration-minus-idle
composition, and the idle-driver attribution rules including the
multi-driver warn flag.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from fleet_telemetry_hub import unifier
from fleet_telemetry_hub.unifier.overlap import (
    DrivingWindow,
    attribute_idle_driver,
    clip_overlap_seconds,
    compute_driving_duration_seconds,
    sum_overlap_seconds,
)
from tests.unifier._fixtures import dt

# Semantic time constants used across the overlap assertions. Tests
# build idle/driving windows on round-hour boundaries so these
# constants stay readable next to the input bounds.
_FIVE_MINUTES_S = 5 * 60
_TEN_MINUTES_S = 10 * 60
_FIFTEEN_MINUTES_S = 15 * 60
_THIRTY_MINUTES_S = 30 * 60
_FORTY_FIVE_MINUTES_S = 45 * 60
_SIXTY_MINUTES_S = 60 * 60
_SIXTY_TRUNCATED_S = 60
_NINETY_MINUTES_S = 90 * 60
_TWO_HOURS_S = 2 * 60 * 60

_DRIVER_SAM = ('1000001', 'Sam Snowflake')
_DRIVER_SUZY = ('1000002', 'Suzy Snowflake')
_DRIVER_SAMMY = ('1000003', 'Sammy Snowflake')
_DRIVER_NAME_ONLY_SANDRA = (None, 'Sandra Snowflake')
_DRIVER_NULL_WINDOW = (None, None)


class TestClipOverlapSeconds:
    """``clip_overlap_seconds`` over half-open ``[start, end)`` intervals."""

    def test_disjoint_a_before_b(self) -> None:
        """``a`` entirely before ``b`` yields no overlap."""

        assert (
            clip_overlap_seconds(dt(hour=8), dt(hour=9), dt(hour=10), dt(hour=11)) == 0
        )

    def test_disjoint_a_after_b(self) -> None:
        """``a`` entirely after ``b`` yields no overlap."""

        assert (
            clip_overlap_seconds(dt(hour=10), dt(hour=11), dt(hour=8), dt(hour=9)) == 0
        )

    def test_adjacent_intervals_have_zero_overlap(self) -> None:
        """``a.end == b.start`` is adjacency, not overlap."""

        assert (
            clip_overlap_seconds(dt(hour=8), dt(hour=9), dt(hour=9), dt(hour=10)) == 0
        )

    def test_identical_intervals_return_full_duration(self) -> None:
        """Identical intervals overlap fully."""

        one_hour_seconds = 3600
        assert (
            clip_overlap_seconds(dt(hour=8), dt(hour=9), dt(hour=8), dt(hour=9))
            == one_hour_seconds
        )

    def test_a_strictly_inside_b_returns_a_duration(self) -> None:
        """``a`` fully inside ``b`` overlaps for ``a``'s full duration."""

        one_hour_seconds = 3600
        assert (
            clip_overlap_seconds(dt(hour=9), dt(hour=10), dt(hour=8), dt(hour=11))
            == one_hour_seconds
        )

    def test_b_strictly_inside_a_returns_b_duration(self) -> None:
        """``b`` fully inside ``a`` overlaps for ``b``'s full duration."""

        one_hour_seconds = 3600
        assert (
            clip_overlap_seconds(dt(hour=8), dt(hour=11), dt(hour=9), dt(hour=10))
            == one_hour_seconds
        )

    def test_partial_overlap_is_intersection_duration(self) -> None:
        """Partial overlap returns ``(min(a.end, b.end) - max(a.start, b.start))``."""

        thirty_minutes_seconds = 1800
        assert (
            clip_overlap_seconds(
                dt(hour=8),
                dt(hour=9, minute=30),
                dt(hour=9),
                dt(hour=10),
            )
            == thirty_minutes_seconds
        )

    def test_naive_datetime_raises_type_error(self) -> None:
        """A naive datetime anywhere in the inputs raises ``TypeError``."""

        naive = datetime(2026, 5, 14, 8, 0, 0)  # noqa: DTZ001 -- intentionally naive
        with pytest.raises(TypeError, match='tz-aware'):
            clip_overlap_seconds(naive, dt(hour=9), dt(hour=8), dt(hour=10))

    def test_reversed_interval_raises_value_error(self) -> None:
        """``start > end`` on either input raises ``ValueError``."""

        with pytest.raises(ValueError, match='must be <='):
            clip_overlap_seconds(dt(hour=9), dt(hour=8), dt(hour=8), dt(hour=10))

    def test_microsecond_overlap_truncates_to_integer_seconds(self) -> None:
        """Microsecond-level overlap is truncated via ``int(...)``."""

        a_start = dt(hour=8)
        # 60.5-second overlap.
        a_end = a_start + timedelta(seconds=60, microseconds=500_000)
        truncated_seconds = 60
        assert clip_overlap_seconds(a_start, a_end, a_start, a_end) == truncated_seconds

    def test_zero_duration_a_against_nonzero_b_returns_zero(self) -> None:
        """A zero-duration interval has no overlap to contribute."""

        assert clip_overlap_seconds(dt(hour=8), dt(hour=8), dt(hour=8), dt(hour=9)) == 0

    def test_both_zero_duration_at_same_instant_returns_zero(self) -> None:
        """Two zero-duration intervals at the same instant overlap by zero."""

        assert clip_overlap_seconds(dt(hour=8), dt(hour=8), dt(hour=8), dt(hour=8)) == 0


class TestDrivingWindowConstruction:
    """``DrivingWindow`` validates tz-awareness and start <= end."""

    def test_valid_construction(self) -> None:
        """A normal tz-aware UTC window constructs successfully."""

        window = DrivingWindow(start=dt(hour=8), end=dt(hour=9), driver=_DRIVER_SAM)

        assert window.start == dt(hour=8)
        assert window.end == dt(hour=9)
        assert window.driver == _DRIVER_SAM

    def test_zero_duration_window_is_allowed(self) -> None:
        """A zero-duration window (``start == end``) constructs successfully."""

        DrivingWindow(start=dt(hour=8), end=dt(hour=8), driver=_DRIVER_SAM)

    def test_reversed_window_raises_value_error(self) -> None:
        """``start > end`` raises ``ValueError``."""

        with pytest.raises(ValueError, match='must be <='):
            DrivingWindow(start=dt(hour=9), end=dt(hour=8), driver=_DRIVER_SAM)

    def test_naive_start_raises_type_error(self) -> None:
        """A naive ``start`` raises ``TypeError``."""

        naive = datetime(2026, 5, 14, 8, 0, 0)  # noqa: DTZ001 -- intentionally naive
        with pytest.raises(TypeError, match='tz-aware'):
            DrivingWindow(start=naive, end=dt(hour=9), driver=_DRIVER_SAM)

    def test_naive_end_raises_type_error(self) -> None:
        """A naive ``end`` raises ``TypeError``."""

        naive = datetime(2026, 5, 14, 9, 0, 0)  # noqa: DTZ001 -- intentionally naive
        with pytest.raises(TypeError, match='tz-aware'):
            DrivingWindow(start=dt(hour=8), end=naive, driver=_DRIVER_SAM)

    def test_mixed_tz_aware_zones_are_accepted(self) -> None:
        """Non-UTC tz-aware datetimes are accepted; UTC convention is enforced elsewhere."""

        chicago = ZoneInfo('America/Chicago')
        DrivingWindow(
            start=dt(hour=13),
            end=datetime(2026, 5, 14, 9, 0, 0, tzinfo=chicago),
            driver=_DRIVER_SAM,
        )


class TestSumOverlapSeconds:
    """``sum_overlap_seconds`` accumulates clipped overlaps without dedup."""

    def test_empty_iterable_returns_zero(self) -> None:
        """No other windows means zero overlap."""

        assert sum_overlap_seconds(dt(hour=8), dt(hour=10), []) == 0

    def test_single_non_overlapping_other_returns_zero(self) -> None:
        """A single disjoint other window contributes zero."""

        assert (
            sum_overlap_seconds(
                dt(hour=8),
                dt(hour=9),
                [(dt(hour=10), dt(hour=11))],
            )
            == 0
        )

    def test_single_fully_overlapping_other_returns_clipped(self) -> None:
        """A fully-contained other window contributes its full duration."""

        one_hour_seconds = 3600
        assert (
            sum_overlap_seconds(
                dt(hour=8),
                dt(hour=11),
                [(dt(hour=9), dt(hour=10))],
            )
            == one_hour_seconds
        )

    def test_multiple_disjoint_others_sum_independently(self) -> None:
        """Two non-overlapping other windows sum their clipped overlaps."""

        two_hours_seconds = 7200
        result = sum_overlap_seconds(
            dt(hour=8),
            dt(hour=12),
            [
                (dt(hour=9), dt(hour=10)),
                (dt(hour=10), dt(hour=11)),
            ],
        )
        assert result == two_hours_seconds

    def test_overlapping_other_windows_are_double_counted(self) -> None:
        """Two other windows overlapping each other are summed, not unioned."""

        # Both other windows are identical 1-hour intervals fully inside target.
        # Result is 2*3600, not 3600. This is the documented behavior.
        two_hours_seconds = 7200
        result = sum_overlap_seconds(
            dt(hour=8),
            dt(hour=11),
            [
                (dt(hour=9), dt(hour=10)),
                (dt(hour=9), dt(hour=10)),
            ],
        )
        assert result == two_hours_seconds


class TestComputeDrivingDurationSeconds:
    """``compute_driving_duration_seconds`` is total minus clipped idle."""

    def test_no_idle_returns_full_driving_duration(self) -> None:
        """With no idle windows the result is the driving duration."""

        one_hour_seconds = 3600
        assert (
            compute_driving_duration_seconds(dt(hour=8), dt(hour=9), [])
            == one_hour_seconds
        )

    def test_idle_fully_contained_subtracts_its_duration(self) -> None:
        """Idle fully inside driving subtracts its full duration."""

        one_hour_minus_15_minutes = 2700
        result = compute_driving_duration_seconds(
            dt(hour=8),
            dt(hour=9),
            [(dt(hour=8, minute=20), dt(hour=8, minute=35))],
        )
        assert result == one_hour_minus_15_minutes

    def test_idle_leading_edge_overlap_subtracts_partial(self) -> None:
        """Idle straddling the leading edge subtracts only the overlap."""

        one_hour_minus_10_minutes = 3000
        result = compute_driving_duration_seconds(
            dt(hour=8),
            dt(hour=9),
            [(dt(hour=7, minute=50), dt(hour=8, minute=10))],
        )
        assert result == one_hour_minus_10_minutes

    def test_idle_fully_covers_driving_yields_zero_or_negative(self) -> None:
        """Idle entirely covering driving may produce zero or negative seconds."""

        result = compute_driving_duration_seconds(
            dt(hour=8),
            dt(hour=9),
            [(dt(hour=7), dt(hour=10))],
        )
        # Driving = 3600, overlap = 3600 -> 0.
        assert result == 0

    def test_idle_disjoint_from_driving_returns_full_duration(self) -> None:
        """A disjoint idle window does not affect driving duration."""

        one_hour_seconds = 3600
        result = compute_driving_duration_seconds(
            dt(hour=8),
            dt(hour=9),
            [(dt(hour=10), dt(hour=11))],
        )
        assert result == one_hour_seconds

    def test_multiple_idle_windows_subtract_their_sum(self) -> None:
        """Multiple disjoint idle windows subtract the sum of their overlaps."""

        # Driving 8-10 (7200s). Two idle windows inside: 8:00-8:10 (600s) and 9:00-9:15 (900s).
        expected = 7200 - 600 - 900
        result = compute_driving_duration_seconds(
            dt(hour=8),
            dt(hour=10),
            [
                (dt(hour=8), dt(hour=8, minute=10)),
                (dt(hour=9), dt(hour=9, minute=15)),
            ],
        )
        assert result == expected

    def test_microsecond_driving_window_truncates_to_integer(self) -> None:
        """A 60.5-second driving window with no idle returns 60 (truncated)."""

        start = dt(hour=8)
        end = start + timedelta(seconds=60, microseconds=500_000)
        truncated_seconds = 60
        assert compute_driving_duration_seconds(start, end, []) == truncated_seconds


class TestAttributeIdleDriver:
    """``attribute_idle_driver`` partitions an idle by driver and picks a winner."""

    def test_empty_driving_windows_yields_null_winner(self) -> None:
        """No driving windows means the null bucket wins."""

        winner, buckets, warn = attribute_idle_driver(dt(hour=8), dt(hour=9), [])

        assert winner == _DRIVER_NULL_WINDOW
        assert buckets == {_DRIVER_NULL_WINDOW: 3600}
        assert warn is False

    def test_single_window_fully_covers_idle(self) -> None:
        """A single covering window wins with no null bucket."""

        window = DrivingWindow(start=dt(hour=7), end=dt(hour=10), driver=_DRIVER_SAM)

        winner, buckets, warn = attribute_idle_driver(dt(hour=8), dt(hour=9), [window])

        assert winner == _DRIVER_SAM
        assert buckets == {_DRIVER_SAM: 3600}
        assert _DRIVER_NULL_WINDOW not in buckets
        assert warn is False

    def test_yard_hand_scenario_null_wins(self) -> None:
        """5 minutes of driver D inside a 90-minute idle: null wins."""

        # Idle 8:00-9:30 (90 min = 5400s).
        # Driving 8:00-8:05 with Sam (5 min = 300s).
        window = DrivingWindow(
            start=dt(hour=8), end=dt(hour=8, minute=5), driver=_DRIVER_SAM
        )

        winner, buckets, warn = attribute_idle_driver(
            dt(hour=8), dt(hour=9, minute=30), [window]
        )

        assert winner == _DRIVER_NULL_WINDOW
        assert buckets[_DRIVER_SAM] == _FIVE_MINUTES_S
        assert buckets[_DRIVER_NULL_WINDOW] == _NINETY_MINUTES_S - _FIVE_MINUTES_S
        assert warn is False

    def test_single_driver_majority_wins(self) -> None:
        """60 of 90 minutes covered by D: D wins."""

        # Idle 8:00-9:30 (5400s); driving 8:00-9:00 with Sam (3600s); uncovered 1800s.
        window = DrivingWindow(start=dt(hour=8), end=dt(hour=9), driver=_DRIVER_SAM)

        winner, buckets, warn = attribute_idle_driver(
            dt(hour=8), dt(hour=9, minute=30), [window]
        )

        assert winner == _DRIVER_SAM
        assert buckets[_DRIVER_SAM] == _SIXTY_MINUTES_S
        assert buckets[_DRIVER_NULL_WINDOW] == _THIRTY_MINUTES_S
        assert warn is False

    def test_two_distinct_drivers_warn_flag_true(self) -> None:
        """Two different drivers in one idle: warn_flag is True."""

        w1 = DrivingWindow(start=dt(hour=8), end=dt(hour=9), driver=_DRIVER_SAM)
        w2 = DrivingWindow(
            start=dt(hour=9), end=dt(hour=9, minute=30), driver=_DRIVER_SUZY
        )

        winner, buckets, warn = attribute_idle_driver(
            dt(hour=8), dt(hour=9, minute=30), [w1, w2]
        )

        assert winner == _DRIVER_SAM
        assert buckets[_DRIVER_SAM] == _SIXTY_MINUTES_S
        assert buckets[_DRIVER_SUZY] == _THIRTY_MINUTES_S
        assert warn is True

    def test_tied_drivers_break_by_earliest_start(self) -> None:
        """Equal overlap broken by earliest ``DrivingWindow.start``."""

        # Both 45 min within a 90-min idle 8:00-9:30. Sam's window starts at 8:00,
        # Suzy's at 8:45 -- Sam wins the tie.
        sam_window = DrivingWindow(
            start=dt(hour=8), end=dt(hour=8, minute=45), driver=_DRIVER_SAM
        )
        suzy_window = DrivingWindow(
            start=dt(hour=8, minute=45),
            end=dt(hour=9, minute=30),
            driver=_DRIVER_SUZY,
        )

        winner, _, warn = attribute_idle_driver(
            dt(hour=8), dt(hour=9, minute=30), [sam_window, suzy_window]
        )

        assert winner == _DRIVER_SAM
        assert warn is True

    def test_null_driver_window_does_not_trigger_warn(self) -> None:
        """A (None, None) driver window is not a non-null candidate."""

        sam_window = DrivingWindow(
            start=dt(hour=8), end=dt(hour=8, minute=30), driver=_DRIVER_SAM
        )
        null_window = DrivingWindow(
            start=dt(hour=8, minute=30),
            end=dt(hour=9),
            driver=_DRIVER_NULL_WINDOW,
        )

        winner, buckets, warn = attribute_idle_driver(
            dt(hour=8), dt(hour=9), [sam_window, null_window]
        )

        # Sam and null tied at 1800 each; non-null beats null in ties.
        assert winner == _DRIVER_SAM
        assert buckets[_DRIVER_SAM] == _THIRTY_MINUTES_S
        assert buckets[_DRIVER_NULL_WINDOW] == _THIRTY_MINUTES_S
        assert warn is False

    def test_name_only_driver_counts_as_non_null_candidate(self) -> None:
        """A ``(None, name)`` driver counts as non-null for warn purposes."""

        sam_window = DrivingWindow(
            start=dt(hour=8), end=dt(hour=8, minute=45), driver=_DRIVER_SAM
        )
        sandra_window = DrivingWindow(
            start=dt(hour=8, minute=45),
            end=dt(hour=9),
            driver=_DRIVER_NAME_ONLY_SANDRA,
        )

        _, buckets, warn = attribute_idle_driver(
            dt(hour=8), dt(hour=9), [sam_window, sandra_window]
        )

        assert _DRIVER_NAME_ONLY_SANDRA in buckets
        assert warn is True

    def test_same_driver_across_windows_is_one_candidate(self) -> None:
        """Multiple windows with the same driver merge into one bucket."""

        # Three Sam windows inside a 90-min idle: 5+10+15 = 30 min covered.
        windows = [
            DrivingWindow(
                start=dt(hour=8),
                end=dt(hour=8, minute=5),
                driver=_DRIVER_SAM,
            ),
            DrivingWindow(
                start=dt(hour=8, minute=10),
                end=dt(hour=8, minute=20),
                driver=_DRIVER_SAM,
            ),
            DrivingWindow(
                start=dt(hour=8, minute=30),
                end=dt(hour=8, minute=45),
                driver=_DRIVER_SAM,
            ),
        ]

        winner, buckets, warn = attribute_idle_driver(
            dt(hour=8), dt(hour=9, minute=30), windows
        )

        # 30 min covered by Sam, 60 min uncovered -> null wins.
        assert winner == _DRIVER_NULL_WINDOW
        assert buckets[_DRIVER_SAM] == _THIRTY_MINUTES_S
        assert buckets[_DRIVER_NULL_WINDOW] == _SIXTY_MINUTES_S
        assert warn is False  # one unique non-null candidate

    def test_two_drivers_with_one_appearing_multiple_times(self) -> None:
        """Multiple windows for D1 + one window for D2: both non-null counted once."""

        # Idle 8:00-9:00 (3600s).
        # Sam: 20 min + 25 min = 45 min total.
        # Suzy: 15 min.
        windows = [
            DrivingWindow(
                start=dt(hour=8),
                end=dt(hour=8, minute=20),
                driver=_DRIVER_SAM,
            ),
            DrivingWindow(
                start=dt(hour=8, minute=20),
                end=dt(hour=8, minute=45),
                driver=_DRIVER_SAM,
            ),
            DrivingWindow(
                start=dt(hour=8, minute=45),
                end=dt(hour=9),
                driver=_DRIVER_SUZY,
            ),
        ]

        winner, buckets, warn = attribute_idle_driver(dt(hour=8), dt(hour=9), windows)

        assert winner == _DRIVER_SAM
        assert buckets[_DRIVER_SAM] == _FORTY_FIVE_MINUTES_S
        assert buckets[_DRIVER_SUZY] == _FIFTEEN_MINUTES_S
        assert warn is True

    def test_bucket_distribution_sums_to_idle_duration(self) -> None:
        """Every bucket value sums to the idle total (within integer seconds)."""

        w1 = DrivingWindow(
            start=dt(hour=8), end=dt(hour=8, minute=20), driver=_DRIVER_SAM
        )
        w2 = DrivingWindow(
            start=dt(hour=8, minute=30),
            end=dt(hour=8, minute=45),
            driver=_DRIVER_SUZY,
        )

        idle_start, idle_end = dt(hour=8), dt(hour=9)
        _, buckets, _ = attribute_idle_driver(idle_start, idle_end, [w1, w2])

        expected_total = int((idle_end - idle_start).total_seconds())
        assert sum(buckets.values()) == expected_total

    def test_real_driving_window_instances_used(self) -> None:
        """The helper accepts real ``DrivingWindow`` instances, not mocks."""

        windows = [
            DrivingWindow(start=dt(hour=8), end=dt(hour=9), driver=_DRIVER_SAMMY),
        ]

        winner, _, _ = attribute_idle_driver(dt(hour=8), dt(hour=9), windows)

        assert winner == _DRIVER_SAMMY

    def test_zero_duration_driving_window_is_ignored(self) -> None:
        """A zero-duration driving window contributes nothing."""

        zero_window = DrivingWindow(
            start=dt(hour=8), end=dt(hour=8), driver=_DRIVER_SAM
        )

        winner, buckets, warn = attribute_idle_driver(
            dt(hour=8), dt(hour=9), [zero_window]
        )

        # No coverage from Sam -> Sam not in buckets; null wins.
        assert winner == _DRIVER_NULL_WINDOW
        assert _DRIVER_SAM not in buckets
        assert buckets[_DRIVER_NULL_WINDOW] == _SIXTY_MINUTES_S
        assert warn is False

    def test_reversed_idle_window_raises_value_error(self) -> None:
        """``idle_start > idle_end`` raises ``ValueError`` before iteration."""

        with pytest.raises(ValueError, match='must be <='):
            attribute_idle_driver(dt(hour=9), dt(hour=8), [])

    def test_naive_idle_start_raises_type_error(self) -> None:
        """A naive ``idle_start`` raises ``TypeError``."""

        naive = datetime(2026, 5, 14, 8, 0, 0)  # noqa: DTZ001 -- intentionally naive
        with pytest.raises(TypeError, match='tz-aware'):
            attribute_idle_driver(naive, dt(hour=9), [])

    def test_non_utc_tz_aware_inputs_accepted(self) -> None:
        """Tz-aware non-UTC datetimes are accepted; UTC convention is enforced upstream."""

        chicago = ZoneInfo('America/Chicago')
        idle_start = datetime(2026, 5, 14, 3, 0, 0, tzinfo=chicago)
        idle_end = datetime(2026, 5, 14, 4, 0, 0, tzinfo=chicago)

        # No driving windows -> null wins; the call succeeds because tz-aware.
        winner, _, _ = attribute_idle_driver(idle_start, idle_end, [])

        assert winner == _DRIVER_NULL_WINDOW


def test_imports_from_unifier_package_re_export() -> None:
    """The unifier package re-exports the public helper surface."""

    expected_exports = {
        'DriverIdentity',
        'DrivingWindow',
        'attribute_idle_driver',
        'clip_overlap_seconds',
        'compute_driving_duration_seconds',
        'nfkc_strip',
        'normalize_driver_name',
        'nullify_tokens',
        'sum_overlap_seconds',
    }
    assert expected_exports.issubset(set(unifier.__all__))
