"""Pure date-range chunking helper for paged backfill fetches.

Many provider endpoints (e.g. Samsara's ``/v1/fleet/trips``) cap the
time window per request. ``iter_chunks`` produces contiguous,
non-overlapping sub-ranges that respect a caller-supplied maximum
chunk length, so backfill code can iterate over chunks without
reimplementing the arithmetic at each call site.

The helper is intentionally pure: no I/O, no global state, no logging.
"""

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Literal, overload

__all__: list[str] = ['iter_chunks']


@overload
def iter_chunks(
    start: date,
    end: date,
    max_days: int,
    chunk_format: Literal['date'],
) -> Iterator[tuple[date, date]]: ...


@overload
def iter_chunks(
    start: date,
    end: date,
    max_days: int,
    chunk_format: Literal['datetime'],
) -> Iterator[tuple[datetime, datetime]]: ...


def iter_chunks(
    start: date,
    end: date,
    max_days: int,
    chunk_format: Literal['date', 'datetime'],
) -> Iterator[tuple[date, date]] | Iterator[tuple[datetime, datetime]]:
    """
    Yield contiguous, non-overlapping sub-ranges of ``[start, end]``.

    Each chunk covers at most ``max_days`` consecutive UTC days
    (inclusive at both ends, at date grain). When ``chunk_format``
    is ``'date'``, chunks are ``(date, date)`` with both ends
    inclusive. When ``chunk_format`` is ``'datetime'``, chunks are
    ``(datetime, datetime)`` of tz-aware UTC instants where the
    start is ``00:00:00 UTC`` of the chunk's first date and the
    end is ``00:00:00 UTC`` of the day *after* the chunk's last
    covered date -- exclusive at instant grain, inclusive at date
    grain. The datetime form is what HTTP APIs typically expect for
    a half-open ``[start, end)`` time window.

    Args:
        start: First UTC date in the overall range (inclusive).
        end: Last UTC date in the overall range (inclusive).
        max_days: Maximum number of inclusive days per chunk. Must
            be a positive integer.
        chunk_format: ``'date'`` for inclusive date tuples,
            ``'datetime'`` for the half-open UTC datetime form.

    Yields:
        ``(chunk_start, chunk_end)`` tuples in chronological order.
        The exact element type depends on ``chunk_format`` -- see
        the overloads.

    Raises:
        ValueError: If ``start > end``, if ``max_days <= 0``, or if
            ``chunk_format`` is not one of the two literal values.
    """
    if start > end:
        raise ValueError(f'start ({start}) must be <= end ({end})')
    if max_days <= 0:
        raise ValueError(f'max_days must be positive, got {max_days}')
    if chunk_format not in ('date', 'datetime'):
        raise ValueError(
            f"chunk_format must be 'date' or 'datetime', got {chunk_format!r}"
        )

    if chunk_format == 'date':
        return _iter_chunks_date(start, end, max_days)
    return _iter_chunks_datetime(start, end, max_days)


def _iter_chunks_date(
    start: date,
    end: date,
    max_days: int,
) -> Iterator[tuple[date, date]]:
    """Yield inclusive ``(date, date)`` chunks."""
    chunk_start = start
    step = timedelta(days=max_days - 1)
    one_day = timedelta(days=1)
    while chunk_start <= end:
        chunk_end = min(chunk_start + step, end)
        yield (chunk_start, chunk_end)
        chunk_start = chunk_end + one_day


def _iter_chunks_datetime(
    start: date,
    end: date,
    max_days: int,
) -> Iterator[tuple[datetime, datetime]]:
    """Yield half-open ``(datetime, datetime)`` UTC chunks."""
    one_day = timedelta(days=1)
    for chunk_start_date, chunk_end_date in _iter_chunks_date(start, end, max_days):
        chunk_start_dt = datetime(
            chunk_start_date.year,
            chunk_start_date.month,
            chunk_start_date.day,
            tzinfo=UTC,
        )
        chunk_end_dt = (
            datetime(
                chunk_end_date.year,
                chunk_end_date.month,
                chunk_end_date.day,
                tzinfo=UTC,
            )
            + one_day
        )
        yield (chunk_start_dt, chunk_end_dt)
