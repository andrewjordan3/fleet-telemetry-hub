"""Generic Protocol for provider-specific utilization fetchers."""

from datetime import date
from typing import Protocol, runtime_checkable

__all__: list[str] = ['UtilizationFetcher']


@runtime_checkable
class UtilizationFetcher[BundleT](Protocol):
    """
    Protocol for provider-specific utilization fetchers.

    Implementations fetch all utilization-relevant endpoints for a
    UTC-anchored date range and return a typed bundle. The bundle
    structure is provider-specific but follows a consistent convention:
    aggregate-grain records (per-vehicle-day, per-driver-day) are
    grouped by date in a dict; event-grain records (driving periods,
    assignments, idling events) are flat lists.

    Implementations must:
        - Anchor all time math to UTC. Callers pass dates; the fetcher
          translates each date to its UTC midnight-to-midnight window.
        - Day-loop internally over aggregate-grain endpoints. The
          caller does not know or care that some endpoints need
          per-day calls and others span the full range.
        - Return a fully-populated bundle. Every requested date has a
          key in each by-date dict, even if the value is an empty list.
        - Validate that start_date <= end_date and raise ValueError
          otherwise.
        - Perform no transformation on returned records. Unit
          conversion, timezone interpretation, attribution math, and
          cross-midnight clipping all happen downstream in the unifier.

    Type Parameters:
        BundleT: The provider-specific bundle dataclass returned by
            fetch.
    """

    def fetch(self, start_date: date, end_date: date) -> BundleT:
        """
        Fetch all utilization data for the date range.

        Args:
            start_date: First UTC day to fetch (inclusive).
            end_date: Last UTC day to fetch (inclusive).

        Returns:
            Provider-specific bundle containing all fetched records.

        Raises:
            ValueError: If start_date > end_date.
        """
        ...
