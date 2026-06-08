"""Tests for ``_utilization_metadata.MetadataStore``: load + atomic write.

The store owns ``metadata.json`` for one utilization directory: it reads
the prior metadata, computes whole-file aggregates over the sibling
``data.parquet`` via the pure ``compute_parquet_aggregates``, builds the
locked shape with the pure ``build_metadata_dict``, and writes the result
atomically (temp file + rename). These tests exercise both the on-disk
load paths and the write transaction.
"""

import json
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from fleet_telemetry_hub._utilization_metadata import (
    MetadataAggregates,
    MetadataBuildContext,
    MetadataStore,
    build_metadata_dict,
    compute_parquet_aggregates,
)
from fleet_telemetry_hub.unifier.schema import (
    EventType,
    UnifiedEventRow,
    build_dataframe,
)

_RUN_STARTED = datetime(2026, 5, 14, 0, 0, 0, tzinfo=UTC)
_RUN_COMPLETED = datetime(2026, 5, 14, 0, 5, 0, tzinfo=UTC)
_START_DATE = date(2026, 5, 14)
_END_DATE = date(2026, 5, 14)


def _row(company: str | None, hour: int) -> UnifiedEventRow:
    """A single in-window driving row keyed on a May-14 hour."""
    start = datetime(2026, 5, 14, hour, 0, 0, tzinfo=UTC)
    return UnifiedEventRow(
        company=company,
        event_type=EventType.DRIVING,
        driver_id='TEST-DRIVER-01',
        driver_name='Test Driver',
        vin='TESTVIN0000000001',
        start_time_utc=start,
        end_time_utc=start,
        duration_seconds=0,
        distance_miles=1.0,
    )


def _write_parquet(parquet_dir: Path, rows: list[UnifiedEventRow]) -> Path:
    """Lay down a ``data.parquet`` in ``parquet_dir`` and return its path."""
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / 'data.parquet'
    build_dataframe(rows).to_parquet(parquet_path, index=False)
    return parquet_path


def _context() -> MetadataBuildContext:
    """A fixed run-context bundle (no prior metadata, both providers absent)."""
    return MetadataBuildContext(
        prior_metadata=None,
        run_started=_RUN_STARTED,
        run_completed=_RUN_COMPLETED,
        start_date=_START_DATE,
        end_date=_END_DATE,
        providers_present=['motive', 'samsara'],
        providers_skipped=[],
        providers_failed=[],
    )


class TestLoad:
    """``load()`` parses the prior metadata or signals its absence/corruption."""

    def test_absent_file_returns_none(self, tmp_path: Path) -> None:
        """No ``metadata.json`` -> ``None`` (a first run)."""

        store = MetadataStore(tmp_path)

        assert store.load() is None

    def test_valid_file_returns_parsed_dict(self, tmp_path: Path) -> None:
        """A well-formed ``metadata.json`` parses back to its dict."""

        payload: dict[str, object] = {'latest_data_date': '2026-05-10', 'row_count': 3}
        (tmp_path / 'metadata.json').write_text(json.dumps(payload), encoding='utf-8')
        store = MetadataStore(tmp_path)

        assert store.load() == payload

    def test_malformed_file_raises_json_decode_error(self, tmp_path: Path) -> None:
        """A corrupt ``metadata.json`` raises rather than being treated as absent."""

        (tmp_path / 'metadata.json').write_text('{ not json', encoding='utf-8')
        store = MetadataStore(tmp_path)

        with pytest.raises(json.JSONDecodeError):
            store.load()


class TestWrite:
    """``write(ctx)`` computes aggregates, builds, and atomically persists."""

    def test_written_content_matches_pure_build(self, tmp_path: Path) -> None:
        """The persisted JSON equals ``build_metadata_dict`` over the same parquet."""

        parquet_path = _write_parquet(
            tmp_path, [_row('motive_co', 8), _row('samsara_co', 12)]
        )
        store = MetadataStore(tmp_path)
        ctx = _context()

        store.write(ctx)

        expected = build_metadata_dict(compute_parquet_aggregates(parquet_path), ctx)
        with (tmp_path / 'metadata.json').open(encoding='utf-8') as handle:
            written = json.load(handle)
        assert written == expected

    def test_computes_aggregates_over_own_parquet(self, tmp_path: Path) -> None:
        """Aggregates reflect the ``data.parquet`` sitting in the store's own dir."""

        rows = [_row('motive_co', 8), _row('motive_co', 9)]
        _write_parquet(tmp_path, rows)
        store = MetadataStore(tmp_path)

        store.write(_context())

        with (tmp_path / 'metadata.json').open(encoding='utf-8') as handle:
            written = json.load(handle)
        assert written['row_count'] == len(rows)
        assert written['by_company'] == {'motive_co': len(rows)}

    def test_leaves_no_tmp_after_success(self, tmp_path: Path) -> None:
        """A clean write leaves no ``*.tmp`` in the directory."""

        _write_parquet(tmp_path, [_row('motive_co', 8)])
        store = MetadataStore(tmp_path)

        store.write(_context())

        assert list(tmp_path.glob('*.tmp')) == []

    def test_creates_directory_when_absent(self, tmp_path: Path) -> None:
        """A missing output directory is created before the metadata is written.

        ``write`` computes aggregates before the ``mkdir``; patching that
        pure step lets this test exercise the directory-creation guard in
        isolation, without a parquet needing to pre-exist.
        """

        parquet_dir = tmp_path / 'utilization'
        assert not parquet_dir.exists()
        store = MetadataStore(parquet_dir)
        canned = MetadataAggregates(
            row_count=0, by_company={}, latest_event_end=None
        )

        with patch(
            'fleet_telemetry_hub._utilization_metadata.compute_parquet_aggregates',
            return_value=canned,
        ):
            store.write(_context())

        assert parquet_dir.is_dir()
        assert (parquet_dir / 'metadata.json').is_file()
