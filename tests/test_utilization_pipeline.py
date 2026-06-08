"""Tests for ``UtilizationPipeline``: window determination, per-provider isolation,
atomic write, metadata content, and the end-to-end happy path.

Mocks via ``unittest.mock.patch`` on module-level class references in
``fleet_telemetry_hub.utilization_pipeline`` -- mirrors the pattern in
``tests/test_pipeline.py``. No real APIs are hit.
"""

# pyright: reportPrivateUsage=false

import json
import logging
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from fleet_telemetry_hub._utilization_merge import CorruptUtilizationParquetError
from fleet_telemetry_hub.models.motive_responses import (
    DriverSummary,
    DrivingPeriod,
    VehicleSummary,
)
from fleet_telemetry_hub.models.samsara_responses import (
    SamsaraDriver,
    SamsaraVehicle,
    Trip,
)
from fleet_telemetry_hub.unifier.schema import COLUMNS, DTYPES, read_unified_parquet
from fleet_telemetry_hub.utilization.motive_fetcher import MotiveUtilizationBundle
from fleet_telemetry_hub.utilization.samsara_fetcher import SamsaraUtilizationBundle
from fleet_telemetry_hub.utilization.vehicle_trip import VehicleTrip
from fleet_telemetry_hub.utilization_pipeline import (
    BackfillStalledError,
    BackfillSummary,
    UtilizationPipeline,
    UtilizationRunResult,
)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


_VALID_PROVIDER_BASE = {
    'motive': {
        'enabled': True,
        'base_url': 'https://api.gomotive.com',
        'api_key': 'test_motive_key',
        'request_timeout': [10, 30],
        'max_retries': 3,
        'retry_backoff_factor': 2.0,
        'verify_ssl': True,
        'rate_limit_requests_per_second': 10,
        'company': 'motive_co',
    },
    'samsara': {
        'enabled': True,
        'base_url': 'https://api.samsara.com',
        'api_key': 'test_samsara_key',
        'request_timeout': [10, 30],
        'max_retries': 3,
        'retry_backoff_factor': 2.0,
        'verify_ssl': True,
        'rate_limit_requests_per_second': 10,
        'company': 'samsara_co',
    },
}


def _write_config(  # noqa: PLR0913 -- one knob per dimension we vary across tests
    tmp_path: Path,
    *,
    default_start_date: str = '2026-05-14',
    lookback_days: int = 7,
    max_window_days: int | None = None,
    parquet_root: Path | None = None,
    motive_enabled: bool = True,
    samsara_enabled: bool = True,
    include_motive: bool = True,
    include_samsara: bool = True,
) -> Path:
    """Write a minimal valid telemetry_config.yaml and return its path."""
    if parquet_root is None:
        parquet_root = tmp_path / 'telemetry'
    providers: dict[str, Any] = {}
    if include_motive:
        providers['motive'] = {**_VALID_PROVIDER_BASE['motive']}
        providers['motive']['enabled'] = motive_enabled
    if include_samsara:
        providers['samsara'] = {**_VALID_PROVIDER_BASE['samsara']}
        providers['samsara']['enabled'] = samsara_enabled
    pipeline_config: dict[str, Any] = {
        'default_start_date': default_start_date,
        'lookback_days': lookback_days,
        'batch_increment_days': 1.0,
        'request_delay_seconds': 0.0,
        'use_truststore': False,
    }
    if max_window_days is not None:
        pipeline_config['max_window_days'] = max_window_days
    config_data: dict[str, Any] = {
        'providers': providers,
        'pipeline': pipeline_config,
        'storage': {
            'parquet_path': str(parquet_root),
            'parquet_compression': 'snappy',
        },
        'logging': {
            'file_path': str(tmp_path / 'telemetry.log'),
            'console_level': 'WARNING',
            'file_level': 'WARNING',
        },
    }
    config_path = tmp_path / 'config.yaml'
    with config_path.open('w') as handle:
        yaml.dump(config_data, handle)
    return config_path


# ---------------------------------------------------------------------------
# Bundle helpers
# ---------------------------------------------------------------------------

_MAY_14 = date(2026, 5, 14)
_MAY_20 = date(2026, 5, 20)
_TWO_PROVIDERS_PRESENT_ROW_COUNT = 2


def _at(hour: int) -> datetime:
    return datetime(2026, 5, 14, hour, 0, 0, tzinfo=UTC)


def _motive_vehicle(vin: str = 'TESTVIN0000000100') -> VehicleSummary:
    return VehicleSummary.model_validate(
        {
            'id': 8000001,
            'number': 'TEST-001',
            'year': '2020',
            'make': 'TestMake',
            'model': 'TestModel',
            'vin': vin,
            'metric_units': False,
        }
    )


def _motive_driver() -> DriverSummary:
    return DriverSummary.model_validate(
        {
            'id': 9000001,
            'first_name': 'Sam',
            'last_name': 'Snowflake',
            'username': None,
            'email': None,
            'driver_company_id': None,
            'status': 'active',
            'role': 'driver',
        }
    )


def _motive_period() -> DrivingPeriod:
    return DrivingPeriod.model_validate(
        {
            'id': 4550000001,
            'start_time': _at(hour=8),
            'end_time': _at(hour=9),
            'status': 'complete',
            'type': 'driving',
            'duration': 3600,
            'start_kilometers': 100.0,
            'end_kilometers': 116.09,
            'source': 1,
            'driver': _motive_driver().model_dump(by_alias=True),
            'vehicle': _motive_vehicle().model_dump(by_alias=True),
        }
    )


def _empty_motive_bundle(
    date_range: tuple[date, date] = (_MAY_14, _MAY_20),
    company: str | None = 'motive_co',
) -> MotiveUtilizationBundle:
    return MotiveUtilizationBundle(
        vehicle_utilizations_by_date={},
        driver_idle_rollups_by_date={},
        driving_periods=[],
        idle_events=[],
        date_range=date_range,
        company=company,
    )


def _motive_bundle_with_one_period(
    date_range: tuple[date, date] = (_MAY_14, _MAY_20),
    company: str | None = 'motive_co',
) -> MotiveUtilizationBundle:
    return MotiveUtilizationBundle(
        vehicle_utilizations_by_date={},
        driver_idle_rollups_by_date={},
        driving_periods=[_motive_period()],
        idle_events=[],
        date_range=date_range,
        company=company,
    )


def _at_date(day: date, hour: int) -> datetime:
    """Build a tz-aware UTC datetime on a given date (for merge-window tests)."""
    return datetime(day.year, day.month, day.day, hour, 0, 0, tzinfo=UTC)


def _motive_period_at(
    day: date, *, period_id: int, distance_km: float = 16.09
) -> DrivingPeriod:
    """A one-hour Motive driving period on ``day``, with a controllable distance."""
    start = _at_date(day, 8)
    end = _at_date(day, 9)
    return DrivingPeriod.model_validate(
        {
            'id': period_id,
            'start_time': start,
            'end_time': end,
            'status': 'complete',
            'type': 'driving',
            'duration': int((end - start).total_seconds()),
            'start_kilometers': 100.0,
            'end_kilometers': 100.0 + distance_km,
            'source': 1,
            'driver': _motive_driver().model_dump(by_alias=True),
            'vehicle': _motive_vehicle().model_dump(by_alias=True),
        }
    )


def _motive_bundle_with_periods(
    periods: list[DrivingPeriod], company: str | None = 'motive_co'
) -> MotiveUtilizationBundle:
    """A Motive bundle carrying an explicit list of driving periods."""
    return MotiveUtilizationBundle(
        vehicle_utilizations_by_date={},
        driver_idle_rollups_by_date={},
        driving_periods=periods,
        idle_events=[],
        date_range=(_MAY_14, _MAY_20),
        company=company,
    )


def _daily_motive_bundle(start: date, end: date) -> MotiveUtilizationBundle:
    """A Motive bundle with one driving period per UTC day in ``[start, end]``.

    The mock fetcher returns this whole bundle on every batch regardless of
    window; the merge's per-window filter is what slices it across a backfill
    march, so daily coverage lets a test assert the file ends up with exactly
    one row per day and no gaps or duplicates at batch boundaries.
    """
    periods = [
        _motive_period_at(start + timedelta(days=offset), period_id=4550000000 + offset)
        for offset in range((end - start).days + 1)
    ]
    return _motive_bundle_with_periods(periods)


def _samsara_vehicle() -> SamsaraVehicle:
    return SamsaraVehicle.model_validate(
        {
            'id': '999999900000001',
            'name': 'TEST-001',
            'vin': 'TESTVIN0000000001',
        }
    )


def _samsara_driver() -> SamsaraDriver:
    return SamsaraDriver.model_validate(
        {
            'id': '1000001',
            'name': 'Suzy Snowflake',
            'driverActivationStatus': 'active',
        }
    )


def _samsara_trip() -> VehicleTrip:
    """Build a ``VehicleTrip`` -- the wrapper the Samsara bundle's trips list now holds."""
    trip = Trip.model_validate(
        {
            'id': '00000000-0000-0000-0000-000000001001',
            'driverId': '1000001',
            'startMs': int(_at(hour=12).timestamp() * 1000),
            'endMs': int(_at(hour=13).timestamp() * 1000),
            'distanceMeters': 1609,
        }
    )
    return VehicleTrip.from_trip(trip, '999999900000001')


def _empty_samsara_bundle(
    date_range: tuple[date, date] = (_MAY_14, _MAY_20),
    company: str | None = 'samsara_co',
) -> SamsaraUtilizationBundle:
    return SamsaraUtilizationBundle(
        vehicles=[],
        drivers=[],
        trips=[],
        idling_events=[],
        date_range=date_range,
        company=company,
    )


def _samsara_bundle_with_one_trip(
    date_range: tuple[date, date] = (_MAY_14, _MAY_20),
    company: str | None = 'samsara_co',
) -> SamsaraUtilizationBundle:
    return SamsaraUtilizationBundle(
        vehicles=[_samsara_vehicle()],
        drivers=[_samsara_driver()],
        trips=[_samsara_trip()],
        idling_events=[],
        date_range=date_range,
        company=company,
    )


# ---------------------------------------------------------------------------
# Mock fetcher factory
# ---------------------------------------------------------------------------


def _mock_fetcher_class(
    canned_bundle: Any | None = None,
    *,
    raises: type[Exception] | None = None,
) -> type[Any]:
    """
    Return a class usable as a ``patch()`` target for a fetcher.

    The pipeline calls ``FetcherClass(provider)`` then ``.fetch(start, end)``.
    Tests pre-configure either a canned bundle or an exception to raise.
    """

    class _MockFetcher:
        def __init__(self, provider: Any) -> None:
            self._provider = provider

        def fetch(self, start_date: date, end_date: date) -> Any:
            del start_date, end_date  # unused; canned response
            if raises is not None:
                raise raises('mock fetcher failure')
            return canned_bundle

    return _MockFetcher


# ---------------------------------------------------------------------------
# Patch helper bundling Motive + Samsara + Provider.from_config
# ---------------------------------------------------------------------------


class _PatchedFetchers:
    """Context-manager that patches the three fetcher-relevant module globals."""

    def __init__(
        self,
        *,
        motive_bundle: MotiveUtilizationBundle | None = None,
        samsara_bundle: SamsaraUtilizationBundle | None = None,
        motive_raises: type[Exception] | None = None,
        samsara_raises: type[Exception] | None = None,
    ) -> None:
        self._motive_class = _mock_fetcher_class(motive_bundle, raises=motive_raises)
        self._samsara_class = _mock_fetcher_class(samsara_bundle, raises=samsara_raises)
        self._patches: list[Any] = []

    def __enter__(self) -> '_PatchedFetchers':
        self._patches = [
            patch(
                'fleet_telemetry_hub.utilization_pipeline.MotiveUtilizationFetcher',
                self._motive_class,
            ),
            patch(
                'fleet_telemetry_hub.utilization_pipeline.SamsaraUtilizationFetcher',
                self._samsara_class,
            ),
            patch(
                'fleet_telemetry_hub.utilization_pipeline.Provider.from_config',
                return_value=MagicMock(),
            ),
        ]
        for patcher in self._patches:
            patcher.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        for patcher in self._patches:
            patcher.stop()


def _seed_two_provider_file(tmp_path: Path, parquet_root: Path) -> tuple[bytes, str]:
    """Lay down a good ``data.parquet`` + ``metadata.json`` via a both-present run.

    Returns the captured ``(parquet_bytes, metadata_text)`` so a follow-up
    skip run can assert byte-for-byte preservation.
    """
    config_path = _write_config(tmp_path, parquet_root=parquet_root)
    pipeline = UtilizationPipeline(config_path)
    with _PatchedFetchers(
        motive_bundle=_motive_bundle_with_one_period(),
        samsara_bundle=_samsara_bundle_with_one_trip(),
    ):
        pipeline.run()
    parquet_bytes = (pipeline.parquet_dir / 'data.parquet').read_bytes()
    metadata_text = (pipeline.parquet_dir / 'metadata.json').read_text()
    return parquet_bytes, metadata_text


# ---------------------------------------------------------------------------
# Window determination
# ---------------------------------------------------------------------------


class TestDetermineWindow:
    """``_determine_window`` derives ``(start_date, end_date)`` per the spec."""

    def test_first_run_uses_default_start_date(self, tmp_path: Path) -> None:
        """No prior metadata -> start_date == config.default_start_date."""

        config_path = _write_config(tmp_path, default_start_date='2026-04-01')
        pipeline = UtilizationPipeline(config_path)

        start, end = pipeline._determine_window(None)

        assert start == date(2026, 4, 1)
        assert end == (datetime.now(UTC) - timedelta(days=1)).date()

    def test_first_run_metadata_present_but_no_anchor(self, tmp_path: Path) -> None:
        """Metadata missing ``latest_data_date`` is treated as a first run."""

        config_path = _write_config(tmp_path, default_start_date='2026-04-01')
        pipeline = UtilizationPipeline(config_path)
        stale_metadata: dict[str, Any] = {'row_count': 0}

        start, _ = pipeline._determine_window(stale_metadata)

        assert start == date(2026, 4, 1)

    def test_subsequent_run_with_lookback_days(self, tmp_path: Path) -> None:
        """``start = latest_data_date - lookback_days``."""

        config_path = _write_config(tmp_path, lookback_days=7)
        pipeline = UtilizationPipeline(config_path)
        prior: dict[str, Any] = {'latest_data_date': '2026-05-20'}

        start, _ = pipeline._determine_window(prior)

        assert start == date(2026, 5, 13)

    def test_zero_lookback_anchors_at_latest_data_date(self, tmp_path: Path) -> None:
        """``lookback_days=0`` -> ``start_date == latest_data_date`` exactly."""

        config_path = _write_config(tmp_path, lookback_days=0)
        pipeline = UtilizationPipeline(config_path)

        start, _ = pipeline._determine_window({'latest_data_date': '2026-05-20'})

        assert start == date(2026, 5, 20)

    def test_thirty_day_lookback(self, tmp_path: Path) -> None:
        """``lookback_days=30`` -> ``start = latest_data_date - 30``."""

        config_path = _write_config(tmp_path, lookback_days=30)
        pipeline = UtilizationPipeline(config_path)

        start, _ = pipeline._determine_window({'latest_data_date': '2026-05-20'})

        assert start == date(2026, 4, 20)

    def test_cap_inert_in_steady_state(self, tmp_path: Path) -> None:
        """A recent anchor + a set cap leaves ``end_date == today-1`` (cap no-op)."""

        today_minus_one = (datetime.now(UTC) - timedelta(days=1)).date()
        # A recent anchor: start = (today-1) - lookback, well within the cap.
        recent_latest = today_minus_one.isoformat()
        config_path = _write_config(tmp_path, lookback_days=7, max_window_days=28)
        pipeline = UtilizationPipeline(config_path)

        start, end = pipeline._determine_window({'latest_data_date': recent_latest})

        assert start == today_minus_one - timedelta(days=7)
        assert end == today_minus_one

    def test_cap_bites_on_far_past_start(self, tmp_path: Path) -> None:
        """A far-past first-run start + cap -> ``end == start + max_window_days``."""

        config_path = _write_config(
            tmp_path, default_start_date='2025-01-01', max_window_days=28
        )
        pipeline = UtilizationPipeline(config_path)

        start, end = pipeline._determine_window(None)

        assert start == date(2025, 1, 1)
        assert end == start + timedelta(days=28)

    def test_no_cap_uses_today_minus_one_even_for_far_past_start(
        self, tmp_path: Path
    ) -> None:
        """With no cap, a far-past start still ends at ``today-1`` (old behavior)."""

        config_path = _write_config(tmp_path, default_start_date='2025-01-01')
        pipeline = UtilizationPipeline(config_path)

        start, end = pipeline._determine_window(None)

        assert start == date(2025, 1, 1)
        assert end == (datetime.now(UTC) - timedelta(days=1)).date()

    def test_start_after_end_skips_run_without_fetch_or_metadata_write(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Future ``latest_data_date`` (clock skew) -> WARNING and no fetch."""

        config_path = _write_config(tmp_path, lookback_days=0)
        pipeline = UtilizationPipeline(config_path)
        # latest_data_date is in the year 3000 -> start > end -> skip.
        future_metadata = {'latest_data_date': '3000-01-01'}
        pipeline.parquet_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = pipeline.parquet_dir / 'metadata.json'
        with metadata_path.open('w') as handle:
            json.dump(future_metadata, handle)

        # Patch fetchers so we can assert they were not called.
        motive_class = _mock_fetcher_class(_empty_motive_bundle())
        samsara_class = _mock_fetcher_class(_empty_samsara_bundle())
        original_metadata_text = metadata_path.read_text()

        with (
            patch(
                'fleet_telemetry_hub.utilization_pipeline.MotiveUtilizationFetcher',
                motive_class,
            ),
            patch(
                'fleet_telemetry_hub.utilization_pipeline.SamsaraUtilizationFetcher',
                samsara_class,
            ),
            patch(
                'fleet_telemetry_hub.utilization_pipeline.Provider.from_config',
                return_value=MagicMock(),
            ),
            caplog.at_level(
                logging.WARNING,
                logger='fleet_telemetry_hub.utilization_pipeline',
            ),
        ):
            result = pipeline.run()

        assert result.written is False
        # WARNING fired and the prior metadata file was NOT overwritten.
        assert any('after end_date' in record.message for record in caplog.records)
        assert metadata_path.read_text() == original_metadata_text


# ---------------------------------------------------------------------------
# Per-provider fetch isolation
# ---------------------------------------------------------------------------


class TestProviderIsolation:
    """The binary provider model: both required at construction, fetch fails loud."""

    def test_both_present_when_both_succeed(self, tmp_path: Path) -> None:
        """Both fetchers succeed -> ``providers_present == ['motive', 'samsara']``."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)

        with _PatchedFetchers(
            motive_bundle=_empty_motive_bundle(),
            samsara_bundle=_empty_samsara_bundle(),
        ):
            pipeline.run()

        metadata_path = pipeline.parquet_dir / 'metadata.json'
        with metadata_path.open() as handle:
            metadata = json.load(handle)

        assert metadata['providers_present'] == ['motive', 'samsara']
        assert metadata['providers_failed'] == []
        assert metadata['providers_skipped'] == []

    def test_motive_disabled_raises_on_construction(self, tmp_path: Path) -> None:
        """Disabled Motive -> ``UtilizationPipeline`` refuses to construct."""

        config_path = _write_config(tmp_path, motive_enabled=False)

        with pytest.raises(ValueError, match=r"\['motive'\]"):
            UtilizationPipeline(config_path)

    def test_samsara_disabled_raises_on_construction(self, tmp_path: Path) -> None:
        """Symmetric: disabled Samsara -> construction raises."""

        config_path = _write_config(tmp_path, samsara_enabled=False)

        with pytest.raises(ValueError, match=r"\['samsara'\]"):
            UtilizationPipeline(config_path)

    def test_provider_missing_from_config_raises_on_construction(
        self, tmp_path: Path
    ) -> None:
        """A provider key absent from config -> construction raises (missing)."""

        config_path = _write_config(tmp_path, include_samsara=False)

        with pytest.raises(ValueError, match=r"\['samsara'\]"):
            UtilizationPipeline(config_path)

    def test_motive_fetch_failure_raises_and_skips_write(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A raising Motive fetcher propagates (fail loud); nothing is written.

        Motive failing means an enabled provider's data is missing, so the
        run must not write -- writing a Samsara-only frame would destroy
        prior Motive data. The failure is logged at ERROR, then re-raised;
        no parquet or metadata file is produced (first run -> no files).
        """

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)
        with (
            _PatchedFetchers(
                motive_raises=ConnectionError,
                samsara_bundle=_empty_samsara_bundle(),
            ),
            caplog.at_level(
                logging.ERROR, logger='fleet_telemetry_hub.utilization_pipeline'
            ),
            pytest.raises(ConnectionError),
        ):
            pipeline.run()

        assert not (pipeline.parquet_dir / 'data.parquet').exists()
        assert not (pipeline.parquet_dir / 'metadata.json').exists()
        # The failure was logged at ERROR before propagating.
        assert any('Motive fetch failed' in record.message for record in caplog.records)

    def test_samsara_fetch_failure_raises_and_skips_write(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Symmetric: a raising Samsara fetcher propagates; nothing is written."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)
        with (
            _PatchedFetchers(
                motive_bundle=_empty_motive_bundle(),
                samsara_raises=ConnectionError,
            ),
            caplog.at_level(
                logging.ERROR, logger='fleet_telemetry_hub.utilization_pipeline'
            ),
            pytest.raises(ConnectionError),
        ):
            pipeline.run()

        assert not (pipeline.parquet_dir / 'data.parquet').exists()
        assert not (pipeline.parquet_dir / 'metadata.json').exists()
        assert any(
            'Samsara fetch failed' in record.message for record in caplog.records
        )

    def test_both_providers_failing_raises_on_first_run(
        self, tmp_path: Path
    ) -> None:
        """Both providers raising -> the run raises and neither file is written.

        The data-preservation companion (a failing run leaving a
        pre-existing good file untouched) lives in
        ``TestFailureSkipPreservesData``.
        """

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)

        with (
            _PatchedFetchers(motive_raises=RuntimeError, samsara_raises=RuntimeError),
            pytest.raises(RuntimeError),
        ):
            pipeline.run()

        assert not (pipeline.parquet_dir / 'data.parquet').exists()
        assert not (pipeline.parquet_dir / 'metadata.json').exists()


# ---------------------------------------------------------------------------
# Atomic write behavior
# ---------------------------------------------------------------------------


class TestAtomicWrites:
    """Both parquet and metadata writes use the temp-file + rename pattern."""

    def test_no_temp_files_left_after_success(self, tmp_path: Path) -> None:
        """A clean run leaves no ``*.tmp`` files in the parquet directory."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)

        with _PatchedFetchers(
            motive_bundle=_motive_bundle_with_one_period(),
            samsara_bundle=_samsara_bundle_with_one_trip(),
        ):
            pipeline.run()

        leftover_tmp = list(pipeline.parquet_dir.glob('*.tmp'))
        assert leftover_tmp == []

    def test_parquet_write_failure_preserves_existing_file(
        self, tmp_path: Path
    ) -> None:
        """The DuckDB merge raising -> the pre-existing parquet stays intact.

        The merge writes a temp file and atomically renames it onto
        ``data.parquet`` itself; if the merge raises before the rename, the
        existing file must survive untouched and no ``*.tmp`` may leak.
        """

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)

        # First successful run lays down a parquet we can later check.
        with _PatchedFetchers(
            motive_bundle=_motive_bundle_with_one_period(),
            samsara_bundle=_empty_samsara_bundle(),
        ):
            pipeline.run()
        first_bytes = (pipeline.parquet_dir / 'data.parquet').read_bytes()

        # Second run: the merge raises before the atomic rename, so the
        # original parquet must survive the failure.
        with (
            _PatchedFetchers(
                motive_bundle=_motive_bundle_with_one_period(),
                samsara_bundle=_empty_samsara_bundle(),
            ),
            patch(
                'fleet_telemetry_hub.utilization_pipeline.merge_incremental_to_parquet',
                side_effect=OSError('disk full'),
            ),
            pytest.raises(OSError, match='disk full'),
        ):
            pipeline.run()

        assert (pipeline.parquet_dir / 'data.parquet').read_bytes() == first_bytes
        assert list(pipeline.parquet_dir.glob('*.tmp')) == []

    def test_metadata_write_failure_preserves_existing_metadata(
        self, tmp_path: Path
    ) -> None:
        """``json.dump`` raising -> the pre-existing metadata file stays intact."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)

        # First successful run lays down a metadata file.
        with _PatchedFetchers(
            motive_bundle=_motive_bundle_with_one_period(),
            samsara_bundle=_empty_samsara_bundle(),
        ):
            pipeline.run()
        first_metadata_text = (pipeline.parquet_dir / 'metadata.json').read_text()

        # Second run patches ``json.dump`` to raise.
        with (
            _PatchedFetchers(
                motive_bundle=_motive_bundle_with_one_period(),
                samsara_bundle=_empty_samsara_bundle(),
            ),
            patch(
                'fleet_telemetry_hub._utilization_metadata.json.dump',
                side_effect=OSError('disk full'),
            ),
            pytest.raises(OSError, match='disk full'),
        ):
            pipeline.run()

        assert (
            pipeline.parquet_dir / 'metadata.json'
        ).read_text() == first_metadata_text

    def test_malformed_metadata_propagates_json_decode_error(
        self, tmp_path: Path
    ) -> None:
        """Corrupted ``metadata.json`` -> ``json.JSONDecodeError``, not silent."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)
        pipeline.parquet_dir.mkdir(parents=True, exist_ok=True)
        (pipeline.parquet_dir / 'metadata.json').write_text('{ not json')

        with (
            _PatchedFetchers(motive_bundle=_empty_motive_bundle()),
            pytest.raises(json.JSONDecodeError),
        ):
            pipeline.run()


# ---------------------------------------------------------------------------
# Metadata content
# ---------------------------------------------------------------------------


class TestMetadataContent:
    """``_write_metadata`` derives the locked JSON shape from run inputs."""

    def test_successful_run_populates_all_fields(self, tmp_path: Path) -> None:
        """Each documented field appears with the right type / shape."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)

        with _PatchedFetchers(
            motive_bundle=_motive_bundle_with_one_period(),
            samsara_bundle=_samsara_bundle_with_one_trip(),
        ):
            pipeline.run()

        metadata_path = pipeline.parquet_dir / 'metadata.json'
        with metadata_path.open() as handle:
            metadata = json.load(handle)

        # Motive period starts 08:00, Samsara trip starts 12:00; latest is 12.
        assert metadata['latest_data_date'] == '2026-05-14'
        assert metadata['latest_event_end_utc'] == '2026-05-14T13:00:00Z'
        assert metadata['row_count'] == _TWO_PROVIDERS_PRESENT_ROW_COUNT
        assert metadata['by_company'] == {'motive_co': 1, 'samsara_co': 1}
        assert metadata['providers_present'] == ['motive', 'samsara']
        assert metadata['providers_skipped'] == []
        assert metadata['providers_failed'] == []
        assert metadata['schema_version'] == 1
        # Timestamps use the 'Z' suffix, not '+00:00'.
        assert metadata['last_run_started_utc'].endswith('Z')
        assert metadata['last_run_completed_utc'].endswith('Z')
        assert metadata['fetch_window_start_utc'].endswith('Z')
        assert metadata['fetch_window_end_utc'].endswith('Z')

    def test_empty_result_preserves_prior_latest_data_date(
        self, tmp_path: Path
    ) -> None:
        """Empty DataFrame with prior metadata -> ``latest_data_date`` preserved."""

        config_path = _write_config(tmp_path, lookback_days=0)
        pipeline = UtilizationPipeline(config_path)
        pipeline.parquet_dir.mkdir(parents=True, exist_ok=True)
        prior_anchor = '2026-05-10'
        prior = {'latest_data_date': prior_anchor}
        with (pipeline.parquet_dir / 'metadata.json').open('w') as handle:
            json.dump(prior, handle)

        with _PatchedFetchers(
            motive_bundle=_empty_motive_bundle(date_range=(date(2026, 5, 10), _MAY_20)),
            samsara_bundle=_empty_samsara_bundle(
                date_range=(date(2026, 5, 10), _MAY_20)
            ),
        ):
            pipeline.run()

        with (pipeline.parquet_dir / 'metadata.json').open() as handle:
            new_metadata = json.load(handle)

        assert new_metadata['latest_data_date'] == prior_anchor
        assert new_metadata['latest_event_end_utc'] is None

    def test_empty_result_with_no_prior_metadata_yields_null_anchor(
        self, tmp_path: Path
    ) -> None:
        """First-run empty result -> ``latest_data_date`` is JSON null."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)

        with _PatchedFetchers(
            motive_bundle=_empty_motive_bundle(),
            samsara_bundle=_empty_samsara_bundle(),
        ):
            pipeline.run()

        with (pipeline.parquet_dir / 'metadata.json').open() as handle:
            metadata = json.load(handle)

        assert metadata['latest_data_date'] is None

    def test_null_company_appears_in_by_company_under_sentinel(
        self, tmp_path: Path
    ) -> None:
        """A null-company row shows up under ``'(null)'`` in ``by_company``."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)

        # Motive bundle has company=None on every row it produces.
        with _PatchedFetchers(
            motive_bundle=_motive_bundle_with_one_period(company=None),
            samsara_bundle=_empty_samsara_bundle(),
        ):
            pipeline.run()

        with (pipeline.parquet_dir / 'metadata.json').open() as handle:
            metadata = json.load(handle)

        assert metadata['by_company'] == {'(null)': 1}

    def test_provider_ordering_is_deterministic_in_lists(self, tmp_path: Path) -> None:
        """Motive precedes Samsara in the persisted provider lists.

        Driven on a write path (both present) so the lists are actually
        persisted -- a failed/skipped-only run no longer writes metadata,
        so ordering must be asserted where metadata exists.
        """

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)
        with _PatchedFetchers(
            motive_bundle=_empty_motive_bundle(),
            samsara_bundle=_empty_samsara_bundle(),
        ):
            pipeline.run()

        with (pipeline.parquet_dir / 'metadata.json').open() as handle:
            metadata = json.load(handle)

        # Both present -> the populated list keeps Motive ahead of Samsara.
        assert metadata['providers_present'] == ['motive', 'samsara']
        assert metadata['providers_skipped'] == []
        assert metadata['providers_failed'] == []


# ---------------------------------------------------------------------------
# Output schema integrity
# ---------------------------------------------------------------------------


class TestOutputSchema:
    """The written parquet restores the locked schema via the canonical reader."""

    def test_round_trip_preserves_columns_and_dtypes(self, tmp_path: Path) -> None:
        """``read_unified_parquet`` restores ``COLUMNS`` and ``DTYPES`` from disk."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)
        with _PatchedFetchers(
            motive_bundle=_motive_bundle_with_one_period(),
            samsara_bundle=_samsara_bundle_with_one_trip(),
        ):
            pipeline.run()

        df = read_unified_parquet(pipeline.parquet_dir / 'data.parquet')
        assert list(df.columns) == list(COLUMNS)
        assert df.dtypes.to_dict() == DTYPES

    def test_empty_result_parquet_still_has_correct_schema(
        self, tmp_path: Path
    ) -> None:
        """An empty result still produces a schema-correct readable parquet."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)
        with _PatchedFetchers(
            motive_bundle=_empty_motive_bundle(),
            samsara_bundle=_empty_samsara_bundle(),
        ):
            pipeline.run()

        df = read_unified_parquet(pipeline.parquet_dir / 'data.parquet')
        assert len(df) == 0
        assert list(df.columns) == list(COLUMNS)
        assert df.dtypes.to_dict() == DTYPES


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------


class TestDirectoryCreation:
    """The ``utilization/`` directory is created lazily on first run."""

    def test_first_run_creates_utilization_subdirectory(self, tmp_path: Path) -> None:
        """Fresh ``parquet_path`` -> the ``utilization/`` subdir appears after ``run()``."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)
        assert not pipeline.parquet_dir.exists()

        with _PatchedFetchers(
            motive_bundle=_empty_motive_bundle(),
            samsara_bundle=_empty_samsara_bundle(),
        ):
            pipeline.run()

        assert pipeline.parquet_dir.is_dir()

    def test_second_run_does_not_error_on_existing_directory(
        self, tmp_path: Path
    ) -> None:
        """Re-running with the directory present is a no-op for the mkdir step."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)
        # Two successive runs.
        with _PatchedFetchers(
            motive_bundle=_empty_motive_bundle(),
            samsara_bundle=_empty_samsara_bundle(),
        ):
            pipeline.run()
            pipeline.run()  # must not raise


# ---------------------------------------------------------------------------
# End-to-end happy path
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """One integration test that exercises the entire flow."""

    def test_full_happy_path(self, tmp_path: Path) -> None:
        """Fresh config + canned bundles -> result, on-disk file, and metadata agree."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)

        with _PatchedFetchers(
            motive_bundle=_motive_bundle_with_one_period(),
            samsara_bundle=_samsara_bundle_with_one_trip(),
        ):
            result = pipeline.run()

        # 1) The run wrote, and the result row count matches the file.
        assert result.written is True
        assert result.row_count == _TWO_PROVIDERS_PRESENT_ROW_COUNT

        # 2) The on-disk file restores the locked schema and is globally sorted
        #    by (company, start_time_utc, event_type).
        on_disk = read_unified_parquet(pipeline.parquet_dir / 'data.parquet')
        assert len(on_disk) == result.row_count
        assert on_disk.dtypes.to_dict() == DTYPES
        companies = on_disk['company'].tolist()
        assert companies == sorted(companies)

        # 3) Metadata reflects the actual content.
        with (pipeline.parquet_dir / 'metadata.json').open() as handle:
            metadata = json.load(handle)
        assert metadata['row_count'] == _TWO_PROVIDERS_PRESENT_ROW_COUNT
        assert metadata['providers_present'] == ['motive', 'samsara']


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLogging:
    """The pipeline emits the documented INFO / WARNING / ERROR records."""

    def test_run_emits_start_window_and_completion_info(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """INFO records cover start time, fetch window, and completion."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)

        with (
            _PatchedFetchers(
                motive_bundle=_empty_motive_bundle(),
                samsara_bundle=_empty_samsara_bundle(),
            ),
            caplog.at_level(
                logging.INFO, logger='fleet_telemetry_hub.utilization_pipeline'
            ),
        ):
            pipeline.run()

        messages = [record.message for record in caplog.records]
        assert any('run starting at' in message for message in messages)
        assert any('Fetch window' in message for message in messages)
        assert any('run complete' in message for message in messages)


# ---------------------------------------------------------------------------
# Fetch failure preserves existing data (the direct incident regression)
# ---------------------------------------------------------------------------


class TestFailureSkipPreservesData:
    """A fetch failure raises before any write, leaving prior files intact.

    This is the direct regression for the incident: a partial fetch must
    never overwrite ``data.parquet`` with a short/empty frame. Under the
    binary provider model the mechanism is a raise (not a skip-and-return),
    but the guarantee -- existing files stay byte-identical -- is unchanged.
    """

    def test_samsara_failure_leaves_existing_files_byte_identical(
        self, tmp_path: Path
    ) -> None:
        """Motive present + Samsara failing -> run raises, files untouched."""

        parquet_root = tmp_path / 'telemetry'
        parquet_bytes, metadata_text = _seed_two_provider_file(tmp_path, parquet_root)

        config_path = _write_config(tmp_path, parquet_root=parquet_root)
        pipeline = UtilizationPipeline(config_path)
        with (
            _PatchedFetchers(
                motive_bundle=_motive_bundle_with_one_period(),
                samsara_raises=ConnectionError,
            ),
            pytest.raises(ConnectionError),
        ):
            pipeline.run()

        assert (pipeline.parquet_dir / 'data.parquet').read_bytes() == parquet_bytes
        assert (pipeline.parquet_dir / 'metadata.json').read_text() == metadata_text

    def test_both_failing_leaves_existing_files_byte_identical(
        self, tmp_path: Path
    ) -> None:
        """Both providers failing -> run raises, files untouched."""

        parquet_root = tmp_path / 'telemetry'
        parquet_bytes, metadata_text = _seed_two_provider_file(tmp_path, parquet_root)

        config_path = _write_config(tmp_path, parquet_root=parquet_root)
        pipeline = UtilizationPipeline(config_path)
        with (
            _PatchedFetchers(motive_raises=RuntimeError, samsara_raises=RuntimeError),
            pytest.raises(RuntimeError),
        ):
            pipeline.run()

        assert (pipeline.parquet_dir / 'data.parquet').read_bytes() == parquet_bytes
        assert (pipeline.parquet_dir / 'metadata.json').read_text() == metadata_text


# ---------------------------------------------------------------------------
# Incremental merge across two runs (union semantics)
# ---------------------------------------------------------------------------


class TestIncrementalMerge:
    """Two sequential runs prove delete-by-window + append, not whole-file replace."""

    def test_second_run_retains_old_replaces_window_and_appends_new(
        self, tmp_path: Path
    ) -> None:
        """Run 2 keeps pre-window rows, replaces the in-window row, appends new ones."""

        may_10 = date(2026, 5, 10)
        may_12 = date(2026, 5, 12)
        may_13 = date(2026, 5, 13)
        old_window_distance_miles = 10.0  # 16.09 km
        new_window_distance_miles = 20.0  # 32.18 km

        # default_start_date well before the data; lookback_days=1 so run 2's
        # window opens at 2026-05-11 (latest_data_date 2026-05-12 minus 1).
        # Both providers are required; Samsara contributes an empty bundle so
        # only the Motive rows under test land in the file.
        config_path = _write_config(
            tmp_path,
            default_start_date='2026-05-01',
            lookback_days=1,
        )
        pipeline = UtilizationPipeline(config_path)

        # Run 1 (first run): two driving periods, 2026-05-10 and 2026-05-12.
        with _PatchedFetchers(
            motive_bundle=_motive_bundle_with_periods(
                [
                    _motive_period_at(may_10, period_id=4550000010),
                    _motive_period_at(
                        may_12,
                        period_id=4550000012,
                        distance_km=16.09,
                    ),
                ]
            ),
            samsara_bundle=_empty_samsara_bundle(),
        ):
            pipeline.run()

        # Run 2: window opens 2026-05-11. An *updated* 2026-05-12 event
        # (different distance) plus a new 2026-05-13 event.
        with _PatchedFetchers(
            motive_bundle=_motive_bundle_with_periods(
                [
                    _motive_period_at(
                        may_12,
                        period_id=4550000012,
                        distance_km=32.18,
                    ),
                    _motive_period_at(may_13, period_id=4550000013),
                ]
            ),
            samsara_bundle=_empty_samsara_bundle(),
        ):
            pipeline.run()

        on_disk = read_unified_parquet(pipeline.parquet_dir / 'data.parquet')
        start_dates = on_disk['start_time_utc'].dt.date.tolist()

        # 2026-05-10 started before the run-2 window -> retained from run 1.
        assert may_10 in start_dates
        # 2026-05-13 is the freshly appended event.
        assert may_13 in start_dates
        # 2026-05-12 appears exactly once -- old copy deleted, new appended.
        assert start_dates.count(may_12) == 1
        # ...and it is the *new* copy (distance updated), not the stale one.
        may_12_row = on_disk[on_disk['start_time_utc'].dt.date == may_12]
        assert may_12_row['distance_miles'].iloc[0] == new_window_distance_miles
        # Sanity: the retained 2026-05-10 row kept its original distance.
        may_10_row = on_disk[on_disk['start_time_utc'].dt.date == may_10]
        assert may_10_row['distance_miles'].iloc[0] == old_window_distance_miles


# ---------------------------------------------------------------------------
# Corrupt existing parquet raises (never silently overwrite)
# ---------------------------------------------------------------------------


class TestCorruptParquetRaises:
    """An unreadable or schema-mismatched existing parquet aborts the run."""

    def test_unreadable_parquet_raises_and_is_not_overwritten(
        self, tmp_path: Path
    ) -> None:
        """Non-parquet garbage bytes -> ``CorruptUtilizationParquetError``, file intact."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)
        pipeline.parquet_dir.mkdir(parents=True, exist_ok=True)
        garbage = b'this is not a parquet file'
        (pipeline.parquet_dir / 'data.parquet').write_bytes(garbage)

        with (
            _PatchedFetchers(
                motive_bundle=_empty_motive_bundle(),
                samsara_bundle=_empty_samsara_bundle(),
            ),
            pytest.raises(CorruptUtilizationParquetError),
        ):
            pipeline.run()

        assert (pipeline.parquet_dir / 'data.parquet').read_bytes() == garbage

    def test_schema_mismatched_parquet_raises_and_is_not_overwritten(
        self, tmp_path: Path
    ) -> None:
        """A valid parquet with the wrong columns -> raises, file untouched."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)
        pipeline.parquet_dir.mkdir(parents=True, exist_ok=True)
        # Valid parquet, but the column set does not match COLUMNS.
        wrong_columns = pd.DataFrame({'company': ['x'], 'event_type': ['driving']})
        wrong_columns.to_parquet(pipeline.parquet_dir / 'data.parquet', index=False)
        before_bytes = (pipeline.parquet_dir / 'data.parquet').read_bytes()

        with (
            _PatchedFetchers(
                motive_bundle=_empty_motive_bundle(),
                samsara_bundle=_empty_samsara_bundle(),
            ),
            pytest.raises(CorruptUtilizationParquetError),
        ):
            pipeline.run()

        assert (pipeline.parquet_dir / 'data.parquet').read_bytes() == before_bytes


# ---------------------------------------------------------------------------
# Batched backfill driver
# ---------------------------------------------------------------------------


def _run_result(end_date: date, *, written: bool = True) -> UtilizationRunResult:
    """A minimal ``UtilizationRunResult`` for driving the backfill loop directly."""
    return UtilizationRunResult(
        written=written,
        row_count=0,
        start_date=end_date,
        end_date=end_date,
        providers_present=['motive'] if written else [],
        providers_skipped=[],
        providers_failed=[] if written else ['motive'],
    )


class TestBackfillToPresent:
    """``backfill_to_present`` marches capped batches forward to the present."""

    def test_marches_across_multiple_batches_to_present(self, tmp_path: Path) -> None:
        """A far-past start + cap runs >1 batch, catches up, covers the full range."""

        today_minus_one = (datetime.now(UTC) - timedelta(days=1)).date()
        span_days = 34
        default_start = today_minus_one - timedelta(days=span_days)

        config_path = _write_config(
            tmp_path,
            default_start_date=default_start.isoformat(),
            lookback_days=7,
            max_window_days=28,
        )
        pipeline = UtilizationPipeline(config_path)

        with _PatchedFetchers(
            motive_bundle=_daily_motive_bundle(default_start, today_minus_one),
            samsara_bundle=_empty_samsara_bundle(),
        ):
            summary = pipeline.backfill_to_present()

        assert isinstance(summary, BackfillSummary)
        assert summary.caught_up is True
        assert summary.batches_run > 1
        assert summary.final_end_date >= today_minus_one

        # The file covers every UTC day in [default_start, today-1], once each.
        on_disk = read_unified_parquet(pipeline.parquet_dir / 'data.parquet')
        covered_dates = sorted(set(on_disk['start_time_utc'].dt.date.tolist()))
        expected_dates = [
            default_start + timedelta(days=offset) for offset in range(span_days + 1)
        ]
        assert covered_dates == expected_dates
        # No duplicates across batch-boundary overlaps.
        assert len(on_disk) == len(expected_dates)
        assert summary.final_row_count == len(expected_dates)

    def test_anchor_advances_each_batch(self, tmp_path: Path) -> None:
        """The fetch-window start moves strictly forward batch to batch."""

        today_minus_one = (datetime.now(UTC) - timedelta(days=1)).date()
        default_start = today_minus_one - timedelta(days=34)

        config_path = _write_config(
            tmp_path,
            default_start_date=default_start.isoformat(),
            lookback_days=7,
            max_window_days=28,
        )
        pipeline = UtilizationPipeline(config_path)

        captured: list[UtilizationRunResult] = []
        real_run = pipeline.run

        def _recording_run() -> UtilizationRunResult:
            result = real_run()
            captured.append(result)
            return result

        with (
            _PatchedFetchers(
                motive_bundle=_daily_motive_bundle(default_start, today_minus_one),
                samsara_bundle=_empty_samsara_bundle(),
            ),
            patch.object(pipeline, 'run', side_effect=_recording_run),
        ):
            pipeline.backfill_to_present()

        starts = [result.start_date for result in captured]
        assert len(starts) > 1
        # Strictly increasing: never an infinite re-fetch of the earliest data.
        assert all(later > earlier for earlier, later in pairwise(starts))

    def test_stops_and_raises_on_failed_provider(self, tmp_path: Path) -> None:
        """A batch where a provider raises -> the march stops, propagating the error.

        Under the binary provider model a fetch failure is fail-loud: the
        batch's ``run()`` raises the provider exception, which propagates
        out of the march rather than being converted to a ``written=False``
        stall. The data-preservation guarantee is unchanged -- nothing was
        written, so a later backfill resumes cleanly.
        """

        today_minus_one = (datetime.now(UTC) - timedelta(days=1)).date()
        default_start = today_minus_one - timedelta(days=60)

        config_path = _write_config(
            tmp_path,
            default_start_date=default_start.isoformat(),
            lookback_days=7,
            max_window_days=28,
        )
        pipeline = UtilizationPipeline(config_path)

        with (
            _PatchedFetchers(
                motive_bundle=_daily_motive_bundle(default_start, today_minus_one),
                samsara_raises=ConnectionError,
            ),
            pytest.raises(ConnectionError),
        ):
            pipeline.backfill_to_present()

        # Nothing was written, so a later backfill can resume cleanly.
        assert not (pipeline.parquet_dir / 'data.parquet').exists()

    def test_requires_max_window_days(self, tmp_path: Path) -> None:
        """Calling without a cap raises -- an uncapped backfill is the OOM case."""

        config_path = _write_config(tmp_path)  # max_window_days defaults to None
        pipeline = UtilizationPipeline(config_path)

        with pytest.raises(ValueError, match='max_window_days'):
            pipeline.backfill_to_present()

    def test_no_progress_guard_terminates(self, tmp_path: Path) -> None:
        """A non-advancing ``end_date`` stops the loop instead of spinning forever."""

        config_path = _write_config(
            tmp_path,
            default_start_date='2025-01-01',
            lookback_days=7,
            max_window_days=28,
        )
        pipeline = UtilizationPipeline(config_path)

        # Every batch reports the same (written) end_date -> no forward progress.
        stuck_date = date(2025, 2, 1)
        with (
            patch.object(pipeline, 'run', side_effect=lambda: _run_result(stuck_date)),
            pytest.raises(BackfillStalledError, match='no forward progress'),
        ):
            pipeline.backfill_to_present()

    def test_iteration_ceiling_terminates(self, tmp_path: Path) -> None:
        """An advance slower than the configured rate aborts at the ceiling."""

        config_path = _write_config(
            tmp_path,
            default_start_date='2025-01-01',
            lookback_days=7,
            max_window_days=28,
        )
        pipeline = UtilizationPipeline(config_path)

        # Advance end_date by only one day per batch -- far slower than the
        # ceiling assumes (~21 days/batch) -- so the ceiling trips before the
        # present is ever reached. end_date advances, so the no-progress guard
        # does not fire first.
        batch_end_dates = (
            date(2025, 1, 1) + timedelta(days=offset) for offset in range(1, 10_000)
        )
        with (
            patch.object(
                pipeline, 'run', side_effect=lambda: _run_result(next(batch_end_dates))
            ),
            pytest.raises(BackfillStalledError, match='iteration ceiling'),
        ):
            pipeline.backfill_to_present()
