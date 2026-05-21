"""Distance-unit conversion helpers for the unifier transforms.

Motive emits distances as kilometers (float); Samsara emits meters
(int). Both providers funnel into the unified output's
``distance_miles`` column, so the per-provider transforms call into
these helpers to keep the conversion factor and rounding policy in
one place.
"""

__all__: list[str] = ['km_to_miles', 'meters_to_miles']

# Standard six-significant-digit conversion factors. Output is
# rounded to one decimal place, so this precision is more than
# sufficient and avoids the rounding drift of a coarser constant.
_MILES_PER_KILOMETER: float = 0.621371
_MILES_PER_METER: float = 0.000621371


def km_to_miles(km: float) -> float:
    """
    Convert kilometers to miles, rounded to one decimal place.

    Args:
        km: Non-negative distance in kilometers.

    Returns:
        Distance in miles rounded to one decimal place.

    Raises:
        ValueError: If ``km`` is negative.
    """
    if km < 0:
        raise ValueError(f'Negative kilometers not allowed: {km}')
    return round(km * _MILES_PER_KILOMETER, 1)


def meters_to_miles(meters: int) -> float:
    """
    Convert meters to miles, rounded to one decimal place.

    Args:
        meters: Non-negative distance in meters.

    Returns:
        Distance in miles rounded to one decimal place.

    Raises:
        ValueError: If ``meters`` is negative.
    """
    if meters < 0:
        raise ValueError(f'Negative meters not allowed: {meters}')
    return round(meters * _MILES_PER_METER, 1)
