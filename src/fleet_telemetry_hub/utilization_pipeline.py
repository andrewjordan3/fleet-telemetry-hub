"""Cron-driven daily entrypoint for the unified utilization pipeline.

External scheduling (cron) invokes ``UtilizationPipeline(config_path).run()``.
The pipeline determines its own fetch window from prior metadata and
config -- no command-line arguments. Output is a single parquet plus
a metadata JSON, both atomically written to ``{parquet_path}/utilization/``.

Per-provider failure is isolated: if one provider's fetch or transform
raises, the other side still completes. Both providers failing still
produces an empty schema-correct parquet so downstream consumers
always see a fresh file.

The metadata-derivation logic lives in the sibling private module
``_utilization_metadata`` so this file stays focused on the
class-shaped orchestration surface.
"""

import json
import logging
import tempfile
import time as time_module
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from fleet_telemetry_hub._utilization_metadata import (
    MetadataBuildContext,
    build_metadata_dict,
)
from fleet_telemetry_hub.common import setup_logger
from fleet_telemetry_hub.config import TelemetryConfig, load_config
from fleet_telemetry_hub.provider import Provider
from fleet_telemetry_hub.unifier.schema import build_dataframe
from fleet_telemetry_hub.unifier.unify import unify
from fleet_telemetry_hub.utilization.motive_fetcher import (
    MotiveUtilizationBundle,
    MotiveUtilizationFetcher,
)
from fleet_telemetry_hub.utilization.samsara_fetcher import (
    SamsaraUtilizationBundle,
    SamsaraUtilizationFetcher,
)

__all__: list[str] = ['UtilizationPipeline']

logger: logging.Logger = logging.getLogger(__name__)

ProviderStatus = Literal['present', 'skipped', 'failed']


class UtilizationPipeline:
    """
    Daily-run pipeline that fetches Motive + Samsara utilization, unifies,
    and writes a single parquet plus a metadata JSON.

    External scheduling (cron) invokes ``run()``. The pipeline figures
    out its own fetch window from prior metadata and config.

    Per-provider failure is isolated -- one provider's outage does not
    block the other. Both providers failing still produces an empty
    schema-correct parquet so downstream consumers always see a fresh
    file.

    Attributes:
        config: The loaded ``TelemetryConfig`` instance (read-only).
        parquet_dir: The ``{parquet_path}/utilization/`` directory the
            pipeline writes to (read-only).
    """

    def __init__(self, config_path: str | Path) -> None:
        """
        Load and validate the YAML config at ``config_path``.

        Args:
            config_path: Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the config fails Pydantic validation.
        """
        config_path = Path(config_path)
        self._config: TelemetryConfig = load_config(config_path)
        setup_logger(config=self._config.logging)
        logger.info(
            'Initializing UtilizationPipeline from config: %s', config_path
        )
        self._parquet_dir: Path = (
            Path(self._config.storage.parquet_path) / 'utilization'
        )
        self._metadata_path: Path = self._parquet_dir / 'metadata.json'
        self._parquet_path: Path = self._parquet_dir / 'data.parquet'

    @property
    def config(self) -> TelemetryConfig:
        """Return the loaded ``TelemetryConfig``."""
        return self._config

    @property
    def parquet_dir(self) -> Path:
        """Return the ``{parquet_path}/utilization/`` output directory."""
        return self._parquet_dir

    def run(self) -> pd.DataFrame:
        """
        Execute one full daily run.

        Determines the fetch window from prior metadata and config,
        fetches each provider's bundle (isolating failures), runs the
        per-provider transforms, unifies, writes parquet, writes
        metadata.

        Returns:
            The unified DataFrame that was written to parquet. Empty
            (zero rows, correct schema) if both providers failed or no
            events fell in the window.

        Raises:
            json.JSONDecodeError: If an existing metadata file is
                malformed (do not silently treat as a first run).
            OSError: If the parquet or metadata write fails. The
                originals (if any) remain intact in this case.
        """
        run_started = datetime.now(UTC)
        wall_start = time_module.monotonic()
        logger.info(
            'UtilizationPipeline run starting at %s', run_started.isoformat()
        )

        prior_metadata = self._load_metadata()
        start_date, end_date = self._determine_window(prior_metadata)

        if start_date > end_date:
            logger.warning(
                'Computed start_date %s is after end_date %s; skipping run',
                start_date,
                end_date,
            )
            return build_dataframe([])

        logger.info(
            'Fetch window: %s to %s (inclusive)', start_date, end_date
        )

        motive_bundle, motive_status = self._fetch_motive(start_date, end_date)
        samsara_bundle, samsara_status = self._fetch_samsara(start_date, end_date)
        statuses: list[tuple[str, ProviderStatus]] = [
            ('motive', motive_status),
            ('samsara', samsara_status),
        ]
        providers_present = [name for name, status in statuses if status == 'present']
        providers_skipped = [name for name, status in statuses if status == 'skipped']
        providers_failed = [name for name, status in statuses if status == 'failed']

        df = unify(motive_bundle, samsara_bundle)
        self._write_parquet_atomic(df)

        run_completed = datetime.now(UTC)
        self._write_metadata(
            MetadataBuildContext(
                df=df,
                prior_metadata=prior_metadata,
                run_started=run_started,
                run_completed=run_completed,
                start_date=start_date,
                end_date=end_date,
                providers_present=providers_present,
                providers_skipped=providers_skipped,
                providers_failed=providers_failed,
            )
        )

        logger.info(
            'UtilizationPipeline run complete: %d rows, took %.2f seconds',
            len(df),
            time_module.monotonic() - wall_start,
        )
        return df

    # --------------------------------------------------------------
    # Window determination
    # --------------------------------------------------------------

    def _determine_window(
        self, prior_metadata: dict[str, Any] | None
    ) -> tuple[date, date]:
        """
        Compute ``(start_date, end_date)`` inclusive for the fetch.

        ``end_date`` is always ``today_utc - 1`` (never the current
        incomplete UTC day). ``start_date`` is the configured default
        on a first run, otherwise ``latest_data_date - lookback_days``
        from prior metadata.
        """
        end_date = (datetime.now(UTC) - timedelta(days=1)).date()
        prior_latest = (
            prior_metadata.get('latest_data_date')
            if prior_metadata is not None
            else None
        )
        if prior_latest is None:
            start_date = date.fromisoformat(
                self._config.pipeline.default_start_date
            )
        else:
            latest = date.fromisoformat(prior_latest)
            start_date = latest - timedelta(
                days=self._config.pipeline.lookback_days
            )
        return start_date, end_date

    # --------------------------------------------------------------
    # Per-provider fetch wrappers
    # --------------------------------------------------------------

    def _fetch_motive(
        self, start_date: date, end_date: date
    ) -> tuple[MotiveUtilizationBundle | None, ProviderStatus]:
        """Fetch Motive bundle, isolating failures into a ``failed`` status."""
        motive_config = self._config.providers.get('motive')
        if motive_config is None or not motive_config.enabled:
            logger.info('Motive provider not configured or disabled; skipping')
            return None, 'skipped'
        try:
            provider = Provider.from_config('motive', self._config)
            fetcher = MotiveUtilizationFetcher(provider)
            return fetcher.fetch(start_date, end_date), 'present'
        except Exception:
            logger.exception('Motive fetch failed; treating as missing bundle')
            return None, 'failed'

    def _fetch_samsara(
        self, start_date: date, end_date: date
    ) -> tuple[SamsaraUtilizationBundle | None, ProviderStatus]:
        """Fetch Samsara bundle, isolating failures into a ``failed`` status."""
        samsara_config = self._config.providers.get('samsara')
        if samsara_config is None or not samsara_config.enabled:
            logger.info('Samsara provider not configured or disabled; skipping')
            return None, 'skipped'
        try:
            provider = Provider.from_config('samsara', self._config)
            fetcher = SamsaraUtilizationFetcher(provider)
            return fetcher.fetch(start_date, end_date), 'present'
        except Exception:
            logger.exception('Samsara fetch failed; treating as missing bundle')
            return None, 'failed'

    # --------------------------------------------------------------
    # Atomic parquet write
    # --------------------------------------------------------------

    def _write_parquet_atomic(self, df: pd.DataFrame) -> None:
        """Write ``df`` to ``data.parquet`` via temp-file + rename."""
        self._parquet_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode='wb',
            suffix='.parquet.tmp',
            dir=self._parquet_dir,
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
        df.to_parquet(
            tmp_path,
            index=False,
            compression=self._config.storage.parquet_compression,
        )
        tmp_path.replace(self._parquet_path)
        logger.info('Wrote %d rows to %s', len(df), self._parquet_path)

    # --------------------------------------------------------------
    # Metadata read / write
    # --------------------------------------------------------------

    def _load_metadata(self) -> dict[str, Any] | None:
        """Return the prior metadata dict, or ``None`` if no file exists."""
        if not self._metadata_path.exists():
            return None
        with self._metadata_path.open('r', encoding='utf-8') as metadata_file:
            return json.load(metadata_file)

    def _write_metadata(self, ctx: MetadataBuildContext) -> None:
        """Build and atomically write the metadata JSON for this run."""
        metadata = build_metadata_dict(ctx)
        self._parquet_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json.tmp',
            dir=self._parquet_dir,
            delete=False,
            encoding='utf-8',
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            json.dump(metadata, tmp_file, indent=2, sort_keys=True)
        tmp_path.replace(self._metadata_path)
        logger.info('Wrote metadata to %s', self._metadata_path)
