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
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

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
from fleet_telemetry_hub.unifier.schema import COLUMNS, DTYPES
from fleet_telemetry_hub.utilization.motive_fetcher import MotiveUtilizationBundle
from fleet_telemetry_hub.utilization.samsara_fetcher import SamsaraUtilizationBundle
from fleet_telemetry_hub.utilization_pipeline import UtilizationPipeline

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
    config_data: dict[str, Any] = {
        'providers': providers,
        'pipeline': {
            'default_start_date': default_start_date,
            'lookback_days': lookback_days,
            'batch_increment_days': 1.0,
            'request_delay_seconds': 0.0,
            'use_truststore': False,
        },
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


def _samsara_trip() -> Trip:
    return Trip.model_validate(
        {
            'id': '00000000-0000-0000-0000-000000001001',
            'vehicleId': '999999900000001',
            'driverId': '1000001',
            'startMs': int(_at(hour=12).timestamp() * 1000),
            'endMs': int(_at(hour=13).timestamp() * 1000),
            'distanceMeters': 1609,
        }
    )


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
        self._samsara_class = _mock_fetcher_class(
            samsara_bundle, raises=samsara_raises
        )
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
            df = pipeline.run()

        assert len(df) == 0
        # WARNING fired and the prior metadata file was NOT overwritten.
        assert any(
            'after end_date' in record.message for record in caplog.records
        )
        assert metadata_path.read_text() == original_metadata_text


# ---------------------------------------------------------------------------
# Per-provider fetch isolation
# ---------------------------------------------------------------------------


class TestProviderIsolation:
    """``_fetch_motive`` / ``_fetch_samsara`` isolate per-provider failures."""

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

    def test_motive_disabled_is_skipped(self, tmp_path: Path) -> None:
        """Disabled Motive -> ``providers_skipped = ['motive']``; Samsara still runs."""

        config_path = _write_config(tmp_path, motive_enabled=False)
        pipeline = UtilizationPipeline(config_path)

        with _PatchedFetchers(samsara_bundle=_empty_samsara_bundle()):
            pipeline.run()

        metadata_path = pipeline.parquet_dir / 'metadata.json'
        with metadata_path.open() as handle:
            metadata = json.load(handle)

        assert metadata['providers_skipped'] == ['motive']
        assert metadata['providers_present'] == ['samsara']

    def test_samsara_disabled_is_skipped(self, tmp_path: Path) -> None:
        """Symmetric: disabled Samsara -> Motive still processed."""

        config_path = _write_config(tmp_path, samsara_enabled=False)
        pipeline = UtilizationPipeline(config_path)

        with _PatchedFetchers(motive_bundle=_empty_motive_bundle()):
            pipeline.run()

        metadata_path = pipeline.parquet_dir / 'metadata.json'
        with metadata_path.open() as handle:
            metadata = json.load(handle)

        assert metadata['providers_skipped'] == ['samsara']
        assert metadata['providers_present'] == ['motive']

    def test_provider_missing_from_config_is_skipped(self, tmp_path: Path) -> None:
        """A provider key absent from config -> skipped (treated as missing)."""

        config_path = _write_config(tmp_path, include_samsara=False)
        pipeline = UtilizationPipeline(config_path)

        with _PatchedFetchers(motive_bundle=_empty_motive_bundle()):
            pipeline.run()

        metadata_path = pipeline.parquet_dir / 'metadata.json'
        with metadata_path.open() as handle:
            metadata = json.load(handle)

        assert metadata['providers_skipped'] == ['samsara']
        assert metadata['providers_present'] == ['motive']

    def test_motive_fetch_failure_is_isolated(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A raising Motive fetcher -> ``providers_failed`` + ERROR + Samsara still runs."""


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
        ):
            pipeline.run()

        metadata_path = pipeline.parquet_dir / 'metadata.json'
        with metadata_path.open() as handle:
            metadata = json.load(handle)

        assert metadata['providers_failed'] == ['motive']
        assert metadata['providers_present'] == ['samsara']
        assert any(
            'Motive fetch failed' in record.message for record in caplog.records
        )

    def test_both_providers_failing_still_writes_parquet_and_metadata(
        self, tmp_path: Path
    ) -> None:
        """Both providers raise -> empty DataFrame written, metadata updated."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)

        with _PatchedFetchers(
            motive_raises=RuntimeError, samsara_raises=RuntimeError
        ):
            df = pipeline.run()

        assert len(df) == 0
        metadata_path = pipeline.parquet_dir / 'metadata.json'
        assert metadata_path.exists()
        with metadata_path.open() as handle:
            metadata = json.load(handle)
        assert metadata['providers_failed'] == ['motive', 'samsara']
        assert metadata['row_count'] == 0


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
        """``to_parquet`` raising -> the pre-existing parquet stays intact."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)

        # First successful run lays down a parquet we can later check.
        with _PatchedFetchers(
            motive_bundle=_motive_bundle_with_one_period(),
            samsara_bundle=_empty_samsara_bundle(),
        ):
            pipeline.run()
        first_bytes = (pipeline.parquet_dir / 'data.parquet').read_bytes()

        # Second run patches ``to_parquet`` to raise and the original
        # parquet must survive the failure.
        with (
            _PatchedFetchers(
                motive_bundle=_motive_bundle_with_one_period(),
                samsara_bundle=_empty_samsara_bundle(),
            ),
            patch.object(pd.DataFrame, 'to_parquet', side_effect=OSError('disk full')),
            pytest.raises(OSError, match='disk full'),
        ):
            pipeline.run()

        assert (pipeline.parquet_dir / 'data.parquet').read_bytes() == first_bytes

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
                'fleet_telemetry_hub.utilization_pipeline.json.dump',
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
        """Motive precedes Samsara in every provider list."""

        config_path = _write_config(tmp_path, samsara_enabled=False)
        pipeline = UtilizationPipeline(config_path)
        with _PatchedFetchers(motive_raises=ValueError):
            pipeline.run()

        with (pipeline.parquet_dir / 'metadata.json').open() as handle:
            metadata = json.load(handle)

        # motive failed, samsara skipped -- both appear, in fixed order.
        assert metadata['providers_failed'] == ['motive']
        assert metadata['providers_skipped'] == ['samsara']


# ---------------------------------------------------------------------------
# Output schema integrity
# ---------------------------------------------------------------------------


class TestOutputSchema:
    """The written parquet round-trips with the locked unified schema."""

    def test_round_trip_preserves_columns_and_dtypes(self, tmp_path: Path) -> None:
        """``pd.read_parquet`` returns a DataFrame with ``COLUMNS`` and ``DTYPES``."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)
        with _PatchedFetchers(
            motive_bundle=_motive_bundle_with_one_period(),
            samsara_bundle=_samsara_bundle_with_one_trip(),
        ):
            pipeline.run()

        df = pd.read_parquet(pipeline.parquet_dir / 'data.parquet')
        assert list(df.columns) == list(COLUMNS)
        assert df.dtypes.to_dict() == DTYPES

    def test_empty_result_parquet_still_has_correct_schema(self, tmp_path: Path) -> None:
        """An empty result still produces a schema-correct readable parquet."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)
        with _PatchedFetchers(
            motive_bundle=_empty_motive_bundle(),
            samsara_bundle=_empty_samsara_bundle(),
        ):
            pipeline.run()

        df = pd.read_parquet(pipeline.parquet_dir / 'data.parquet')
        assert len(df) == 0
        assert list(df.columns) == list(COLUMNS)
        assert df.dtypes.to_dict() == DTYPES


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------


class TestDirectoryCreation:
    """The ``utilization/`` directory is created lazily on first run."""

    def test_first_run_creates_utilization_subdirectory(
        self, tmp_path: Path
    ) -> None:
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
        """Fresh config + canned bundles -> returned DataFrame matches parquet on disk."""

        config_path = _write_config(tmp_path)
        pipeline = UtilizationPipeline(config_path)

        with _PatchedFetchers(
            motive_bundle=_motive_bundle_with_one_period(),
            samsara_bundle=_samsara_bundle_with_one_trip(),
        ):
            returned_df = pipeline.run()

        # 1) Returned DataFrame is sorted by (company, start_time_utc, event_type).
        companies = returned_df['company'].tolist()
        assert companies == sorted(companies)

        # 2) Parquet on disk matches the returned DataFrame.
        on_disk = pd.read_parquet(pipeline.parquet_dir / 'data.parquet')
        pd.testing.assert_frame_equal(on_disk, returned_df)

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
