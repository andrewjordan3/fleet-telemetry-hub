"""Shared fixtures for unifier helper tests."""

from datetime import UTC, datetime


def dt(  # noqa: PLR0913 -- six small date/time components, all defaulted
    year: int = 2026,
    month: int = 5,
    day: int = 14,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
) -> datetime:
    """Build a tz-aware UTC datetime with sensible test defaults."""
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)
