"""Entrypoint for the unified utilization pipeline.

An external scheduler you supply (e.g. cron or a systemd timer) invokes
``UtilizationPipeline(config_path).run()`` -- the package ships no
scheduler of its own, and the cadence (daily is typical, not required) is
your choice. The pipeline resolves its own date range from prior metadata
and config on each invocation -- no command-line arguments -- and covers
it as a sequence of bounded windows (one in steady state, many for a
backfill or outage gap; see ``_utilization_windows``). Output is a single
parquet plus a metadata JSON, both atomically written to
``{parquet_path}/utilization/``.

Each window updates the file incrementally and with bounded memory: the
window is deleted from the existing parquet, the freshly-fetched window
is appended, the union is globally sorted, and the result is written --
all in DuckDB, so pandas never materializes the whole (unbounded) file
(see ``_utilization_merge``). Whole-file metadata aggregates come from a
second streaming DuckDB pass, written per window. Both Motive and
Samsara are required (the pipeline refuses to construct otherwise), and a
fetch failure aborts the run by raising before any write, so a partial
fetch can never destroy prior data; the next successful run re-fetches
the whole lookback window and rewrites correctly. An existing parquet
that is unreadable or whose columns do not match the schema aborts the
run rather than being silently overwritten.

The merge and metadata-derivation logic live in the sibling private
modules ``_utilization_merge`` and ``_utilization_metadata`` so this file
stays focused on the class-shaped orchestration surface.
"""

import logging
import time as time_module
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from fleet_telemetry_hub._utilization_merge import merge_incremental_to_parquet
from fleet_telemetry_hub._utilization_metadata import (
    MetadataBuildContext,
    MetadataStore,
)
from fleet_telemetry_hub._utilization_windows import iter_windows
from fleet_telemetry_hub.common import setup_logger
from fleet_telemetry_hub.config import ProviderConfig, TelemetryConfig, load_config
from fleet_telemetry_hub.provider import Provider
from fleet_telemetry_hub.unifier.unify import unify
from fleet_telemetry_hub.utilization.motive_fetcher import (
    MotiveUtilizationBundle,
    MotiveUtilizationFetcher,
)
from fleet_telemetry_hub.utilization.samsara_fetcher import (
    SamsaraUtilizationBundle,
    SamsaraUtilizationFetcher,
)

__all__: list[str] = [
    'UtilizationPipeline',
    'UtilizationRunResult',
]

logger: logging.Logger = logging.getLogger(__name__)

# The two providers ``UtilizationPipeline`` requires, in the fixed order
# the metadata JSON lists them. Both must be enabled to construct the
# pipeline; a written window always reports exactly these as present.
_UTILIZATION_PROVIDERS: tuple[str, str] = ('motive', 'samsara')


@dataclass(frozen=True, slots=True)
class UtilizationRunResult:
    """Summary of one ``UtilizationPipeline.run()`` march over its range.

    A run covers ``[range_start, today_utc - 1]`` as one or more bounded
    windows (see ``iter_windows``). The run never holds the whole file in
    memory, so callers receive these aggregate counts rather than a frame.

    Attributes:
        windows_run: Number of windows fetched, merged, and persisted.
            ``0`` when the resolved range was empty (``range_start`` after
            ``range_end``) -- nothing was fetched and any existing files
            are unchanged (not necessarily empty).
        final_start_date: Inclusive UTC start of the last window run, or
            ``None`` when ``windows_run == 0``.
        final_end_date: Inclusive UTC end of the last window run, or
            ``None`` when ``windows_run == 0``.
        final_row_count: Whole-file row count after the last window; ``0``
            when ``windows_run == 0`` (the existing file, if any, is
            untouched -- this is not a measured count of it).
    """

    windows_run: int
    final_start_date: date | None
    final_end_date: date | None
    final_row_count: int

    @property
    def written(self) -> bool:
        """True when at least one window was fetched and written."""
        return self.windows_run > 0


@dataclass(frozen=True, slots=True)
class UtilizationBatch:
    """One window's fetched bundles, ready to unify and merge.

    The binary provider model guarantees both bundles are present -- a
    fetch failure raises rather than producing a partial batch.

    Attributes:
        batch_start_date: Inclusive UTC start date of the window.
        batch_end_date: Inclusive UTC end date of the window.
        motive_bundle: The Motive bundle fetched for the window.
        samsara_bundle: The Samsara bundle fetched for the window.
    """

    batch_start_date: date
    batch_end_date: date
    motive_bundle: MotiveUtilizationBundle
    samsara_bundle: SamsaraUtilizationBundle


class UtilizationPipeline:
    """
    Pipeline that fetches Motive + Samsara utilization, unifies,
    and writes a single parquet plus a metadata JSON.

    An external scheduler you supply (e.g. cron or a systemd timer)
    invokes ``run()``; the package ships no scheduler and imposes no
    cadence. The pipeline figures out its own fetch window from prior
    metadata and config on each invocation.

    Both Motive and Samsara must be enabled: the pipeline refuses to
    construct otherwise (``__init__`` raises ``ValueError``). A fetch
    failure on either provider aborts the run and propagates the
    exception, before any write -- so a partial fetch can never overwrite
    good data, and any existing files are left byte-identical.

    The read + window-delete + append + global-sort + write run in DuckDB
    and metadata aggregates stream from the written file, so the whole
    on-disk parquet is never loaded into pandas. A single ``run()`` covers
    its resolved range as a sequence of bounded windows: one window in
    steady state (a recent anchor, or ``max_window_days`` unset), and many
    when a far-past ``default_start_date`` or an outage gap is capped by
    ``pipeline.max_window_days``. Each window is fetched, merged, and has
    its metadata written before the next begins, so a large backfill runs
    in bounded memory and a mid-march failure resumes on the next ``run()``
    from the last persisted window.

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
            ValueError: If the config fails Pydantic validation, or if
                either ``motive`` or ``samsara`` is missing or disabled --
                this pipeline requires both providers enabled.
        """
        config_path = Path(config_path)
        self._config: TelemetryConfig = load_config(config_path)
        setup_logger(config=self._config.logging)
        logger.info('Initializing UtilizationPipeline from config: %s', config_path)

        enabled_providers: dict[str, ProviderConfig] = (
            self._config.get_enabled_providers()
        )
        missing_providers: set[str] = set(_UTILIZATION_PROVIDERS) - set(
            enabled_providers
        )
        if missing_providers:
            raise ValueError(
                'UtilizationPipeline requires both "motive" and "samsara" enabled; '
                f'missing or disabled: {sorted(missing_providers)}'
            )

        self._parquet_dir: Path = (
            Path(self._config.storage.parquet_path) / 'utilization'
        )
        self._parquet_path: Path = self._parquet_dir / 'data.parquet'
        self._metadata_store: MetadataStore = MetadataStore(self._parquet_dir)

    @property
    def config(self) -> TelemetryConfig:
        """Return the loaded ``TelemetryConfig``."""
        return self._config

    @property
    def parquet_dir(self) -> Path:
        """Return the ``{parquet_path}/utilization/`` output directory."""
        return self._parquet_dir

    def run(self) -> UtilizationRunResult:
        """
        Cover the resolved range as a sequence of bounded windows.

        Resolves ``[range_start, today_utc - 1]`` from prior metadata and
        config, then walks ``iter_windows`` -- one window in steady state,
        many for a far-past ``default_start_date`` or outage gap capped by
        ``max_window_days``. For each window it fetches both providers,
        unifies, merges the window into the existing parquet (DuckDB
        delete-then-append), and writes metadata from a streaming aggregate
        over the written file, before the next window begins.

        Returns:
            A ``UtilizationRunResult`` summarizing the march. When the
            resolved range is empty (``range_start`` after ``range_end``)
            ``windows_run`` is ``0`` and ``written`` is ``False`` -- no
            fetch occurred and existing files are unchanged. Otherwise
            ``windows_run`` is the number of windows persisted and the
            ``final_*`` fields describe the last one.

        Raises:
            Exception: If either provider's fetch fails. The exception
                propagates and aborts the march; windows already written
                stay persisted (a re-run resumes from the last one), and
                the in-flight window leaves existing files intact (the
                fetch is logged at ERROR first).
            CorruptUtilizationParquetError: If the existing parquet is
                unreadable or its columns do not match the schema. The
                window aborts before any write, leaving the file intact.
            json.JSONDecodeError: If an existing metadata file is
                malformed (do not silently treat as a first run).
            OSError: If a parquet or metadata write fails. The originals
                (if any) remain intact in this case.
        """
        run_started: datetime = datetime.now(UTC)
        wall_start: float = time_module.monotonic()
        logger.info('UtilizationPipeline run starting at %s', run_started.isoformat())

        range_start, range_end = self._resolve_range(self._metadata_store.load())
        window_days: int = self._effective_window_days(range_start, range_end)

        windows_run: int = 0
        final_start: date | None = None
        final_end: date | None = None
        final_row_count: int = 0

        for window_start, window_end in iter_windows(
            range_start, range_end, window_days
        ):
            prior_metadata: dict[str, Any] | None = self._metadata_store.load()
            logger.info(
                'Fetch window: %s to %s (inclusive)', window_start, window_end
            )

            batch: UtilizationBatch = self._fetch_batch(window_start, window_end)
            df: pd.DataFrame = unify(batch.motive_bundle, batch.samsara_bundle)
            stats = merge_incremental_to_parquet(
                df,
                window_start,
                window_end,
                self._parquet_path,
                compression=self._config.storage.parquet_compression or 'uncompressed',
            )
            self._metadata_store.write(
                MetadataBuildContext(
                    prior_metadata=prior_metadata,
                    run_started=run_started,
                    run_completed=datetime.now(UTC),
                    start_date=window_start,
                    end_date=window_end,
                    providers_present=list(_UTILIZATION_PROVIDERS),
                    providers_skipped=[],
                    providers_failed=[],
                )
            )

            windows_run += 1
            final_start, final_end, final_row_count = (
                window_start,
                window_end,
                stats.final_row_count,
            )
            logger.info(
                'Window %d complete: %s..%s, file rows=%d',
                windows_run,
                window_start,
                window_end,
                stats.final_row_count,
            )

        if windows_run == 0:
            logger.warning(
                'Resolved range start %s is after end %s; nothing to fetch',
                range_start,
                range_end,
            )

        logger.info(
            'UtilizationPipeline run complete: %d window(s), took %.2f seconds',
            windows_run,
            time_module.monotonic() - wall_start,
        )
        return UtilizationRunResult(
            windows_run=windows_run,
            final_start_date=final_start,
            final_end_date=final_end,
            final_row_count=final_row_count,
        )

    def _fetch_batch(self, start_date: date, end_date: date) -> UtilizationBatch:
        """Fetch both providers for one window into a batch (fetch failure raises)."""
        motive_bundle: MotiveUtilizationBundle = self._fetch_motive(
            start_date, end_date
        )
        samsara_bundle: SamsaraUtilizationBundle = self._fetch_samsara(
            start_date, end_date
        )
        return UtilizationBatch(start_date, end_date, motive_bundle, samsara_bundle)

    # --------------------------------------------------------------
    # Range resolution
    # --------------------------------------------------------------

    def _resolve_range(
        self, prior_metadata: dict[str, Any] | None
    ) -> tuple[date, date]:
        """Resolve the inclusive [range_start, range_end] the run must cover.

        ``range_end`` is always ``today_utc - 1`` (never the current
        incomplete UTC day). ``range_start`` is ``default_start_date`` on a
        first run (no prior metadata), else ``latest_data_date -
        lookback_days`` so the trailing lookback is re-fetched.
        """
        range_end: date = (datetime.now(UTC) - timedelta(days=1)).date()
        prior_latest: str | None = (
            prior_metadata.get('latest_data_date')
            if prior_metadata is not None
            else None
        )
        if prior_latest is None:
            range_start: date = date.fromisoformat(
                self._config.pipeline.default_start_date
            )
        else:
            range_start = date.fromisoformat(prior_latest) - timedelta(
                days=self._config.pipeline.lookback_days
            )
        return range_start, range_end

    def _effective_window_days(self, range_start: date, range_end: date) -> int:
        """Per-window span: the configured cap, or the whole range when uncapped.

        When ``max_window_days`` is unset (steady-state default), the run
        uses a single window spanning the whole resolved range -- identical
        to the pre-refactor uncapped behavior. A historical backfill must
        therefore set ``max_window_days`` to march in bounded windows.
        """
        cap: int | None = self._config.pipeline.max_window_days
        if cap is not None:
            return cap
        return max((range_end - range_start).days, 1)

    # --------------------------------------------------------------
    # Per-provider fetch wrappers
    # --------------------------------------------------------------

    def _fetch_motive(
        self, start_date: date, end_date: date
    ) -> MotiveUtilizationBundle:
        """Fetch the Motive bundle; log and re-raise on failure (fail loud).

        ``__init__`` guarantees Motive is enabled, so there is no
        disabled/missing branch here.
        """
        try:
            provider: Provider = Provider.from_config('motive', self._config)
            fetcher: MotiveUtilizationFetcher = MotiveUtilizationFetcher(provider)
            return fetcher.fetch(start_date, end_date)
        except Exception:
            logger.exception('Motive fetch failed')
            raise

    def _fetch_samsara(
        self, start_date: date, end_date: date
    ) -> SamsaraUtilizationBundle:
        """Fetch the Samsara bundle; log and re-raise on failure (fail loud).

        ``__init__`` guarantees Samsara is enabled, so there is no
        disabled/missing branch here.
        """
        try:
            provider: Provider = Provider.from_config('samsara', self._config)
            fetcher: SamsaraUtilizationFetcher = SamsaraUtilizationFetcher(provider)
            return fetcher.fetch(start_date, end_date)
        except Exception:
            logger.exception('Samsara fetch failed')
            raise
