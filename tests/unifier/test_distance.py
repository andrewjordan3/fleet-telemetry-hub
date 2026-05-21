"""Tests for ``unifier.distance``: km/m-to-miles conversion with one-decimal rounding."""

import pytest

from fleet_telemetry_hub.unifier.distance import km_to_miles, meters_to_miles

# Conversion landmarks pinned once for the assertions below.
_ONE_KM_IN_MILES = 0.6
_ONE_HUNDRED_KM_IN_MILES = 62.1
_ONE_MILE_IN_KM = 1.609
_ONE_HUNDRED_THOUSAND_M_IN_MILES = 62.1


class TestKmToMiles:
    """``km_to_miles`` converts kilometers to miles, rounded to one decimal."""

    def test_zero_kilometers_returns_zero_miles(self) -> None:
        """0 km is exactly 0.0 miles."""

        assert km_to_miles(0) == 0.0

    def test_one_kilometer_returns_point_six(self) -> None:
        """1 km rounds to 0.6 mi."""

        assert km_to_miles(1) == _ONE_KM_IN_MILES

    def test_one_hundred_kilometers_returns_sixty_two_point_one(self) -> None:
        """100 km rounds to 62.1 mi."""

        assert km_to_miles(100) == _ONE_HUNDRED_KM_IN_MILES

    def test_one_mile_worth_of_km_rounds_back_to_one(self) -> None:
        """1.609 km (one mile, approximately) rounds back to 1.0 mi."""

        assert km_to_miles(_ONE_MILE_IN_KM) == 1.0

    def test_negative_input_raises_value_error(self) -> None:
        """Negative input raises ``ValueError``."""

        with pytest.raises(ValueError, match='Negative kilometers'):
            km_to_miles(-1)

    @pytest.mark.parametrize(
        ('km', 'expected'),
        [
            (0.0, 0.0),
            (1.0, 0.6),
            (5.0, 3.1),
            (10.0, 6.2),
            (16.09, 10.0),
            (100.0, 62.1),
            (1609.344, 1000.0),
        ],
    )
    def test_result_is_rounded_to_one_decimal(
        self, km: float, expected: float
    ) -> None:
        """Output is always rounded to one decimal place."""

        assert km_to_miles(km) == expected

    def test_result_is_float(self) -> None:
        """The return type is ``float`` even for integer-like inputs."""

        assert isinstance(km_to_miles(1), float)


class TestMetersToMiles:
    """``meters_to_miles`` converts meters to miles, rounded to one decimal."""

    def test_zero_meters_returns_zero_miles(self) -> None:
        """0 m is exactly 0.0 miles."""

        assert meters_to_miles(0) == 0.0

    def test_one_mile_worth_of_meters_returns_one(self) -> None:
        """1609 m rounds to 1.0 mi."""

        assert meters_to_miles(1609) == 1.0

    def test_one_hundred_thousand_meters_returns_sixty_two_point_one(self) -> None:
        """100,000 m rounds to 62.1 mi."""

        assert meters_to_miles(100_000) == _ONE_HUNDRED_THOUSAND_M_IN_MILES

    def test_negative_input_raises_value_error(self) -> None:
        """Negative input raises ``ValueError``."""

        with pytest.raises(ValueError, match='Negative meters'):
            meters_to_miles(-1)

    def test_result_is_float(self) -> None:
        """The return type is ``float``."""

        assert isinstance(meters_to_miles(1609), float)
