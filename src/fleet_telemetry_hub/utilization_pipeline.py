"""Entrypoint for the unified utilization pipeline.

An external scheduler you supply (e.g. cron or a systemd timer) invokes
``UtilizationPipeline(config_path).run()`` -- the package ships no
scheduler of its own, and the cadence (daily is typical, not required) is
your choice. The pipeline determines its own fetch window from prior
metadata and config on each invocation -- no command-line arguments.
Output is a single parquet plus a metadata JSON, both atomically written
to ``{parquet_path}/utilization/``.

Each run updates the file incrementally and with bounded memory: the
fetch window is deleted from the existing parquet, the freshly-fetched
window is appended, the union is globally sorted, and the result is
written -- all in DuckDB, so pandas never materializes the whole
(unbounded) file (see ``_utilization_merge``). Whole-file metadata
aggregates come from a second streaming DuckDB pass. A run writes only
when every enabled provider succeeded; if any enabled provider's fetch
raised, the run is skipped and the existing files are left untouched, so
a partial fetch can never destroy prior data. This is self-healing -- the
next run where all enabled providers succeed re-fetches the whole
lookback window and rewrites correctly. An existing parquet that is
unreadable or whose columns do not match the schema aborts the run
rather than being silently overwritten.

The merge and metadata-derivation logic live in the sibling private
modules ``_utilization_merge`` and ``_utilization_metadata`` so this file
stays focused on the class-shaped orchestration surface.
"""

import logging
import math
import time as time_module
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from fleet_telemetry_hub._utilization_merge import merge_incremental_to_parquet
from fleet_telemetry_hub._utilization_metadata import (
    MetadataBuildContext,
    MetadataStore,
)
from fleet_telemetry_hub.common import setup_logger
from fleet_telemetry_hub.config import TelemetryConfig, load_config
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
    'BackfillStalledError',
    'BackfillSummary',
    'UtilizationPipeline',
    'UtilizationRunResult',
]

logger: logging.Logger = logging.getLogger(__name__)

ProviderStatus = Literal['present', 'skipped', 'failed']

# Slack added to the computed backfill iteration ceiling, so off-by-one
# edge effects (the final batch landing exactly on today-1) never trip the
# safety abort. The ceiling is itself only a backstop against a logic bug.
_BACKFILL_ITERATION_BUFFER: int = 2


class BackfillStalledError(Exception):
    """Raised when ``backfill_to_present`` cannot reach the present.

    Distinct from a clean catch-up: a stall means a batch wrote nothing (an
    enabled provider failed), the window stopped advancing, or the iteration
    ceiling was hit. Existing data is preserved, so a later
    ``backfill_to_present`` resumes from where the file left off.
    """


@dataclass(frozen=True, slots=True)
class UtilizationRunResult:
    """Outcome of one ``UtilizationPipeline.run()``.

    The run no longer returns a DataFrame -- the whole file is never held
    in memory -- so callers receive this small summary instead.

    Attributes:
        written: ``True`` when the run wrote parquet + metadata; ``False``
            on a skipped run (start-after-end, or a failed/insufficient
            provider run), in which case existing files are unchanged.
        row_count: Whole-file row count after the write; ``0`` on a skip.
        start_date: Inclusive UTC start date of the computed fetch window.
        end_date: Inclusive UTC end date of the computed fetch window.
        providers_present: Provider names fetched successfully, in fixed
            order (empty on a start-after-end skip, before any fetch).
        providers_skipped: Provider names disabled or absent from config.
        providers_failed: Provider names that raised during fetch.
    """

    written: bool
    row_count: int
    start_date: date
    end_date: date
    providers_present: list[str]
    providers_skipped: list[str]
    providers_failed: list[str]


@dataclass(frozen=True, slots=True)
class BackfillSummary:
    """Outcome of a ``backfill_to_present`` march.

    Attributes:
        batches_run: Number of ``run()`` invocations the march made.
        final_end_date: ``end_date`` of the last batch that ran.
        caught_up: ``True`` when the march reached ``today_utc - 1``. A
            stalled march raises ``BackfillStalledError`` rather than
            returning ``caught_up=False``, so on return this is always
            ``True``; the field documents the success contract.
        final_row_count: Whole-file row count after the last batch.
    """

    batches_run: int
    final_end_date: date
    caught_up: bool
    final_row_count: int


class UtilizationPipeline:
    """
    Pipeline that fetches Motive + Samsara utilization, unifies,
    and writes a single parquet plus a metadata JSON.

    An external scheduler you supply (e.g. cron or a systemd timer)
    invokes ``run()``; the package ships no scheduler and imposes no
    cadence. The pipeline figures out its own fetch window from prior
    metadata and config on each invocation.

    Per-provider failure is isolated during the fetch -- one provider's
    outage does not block the other. The run only writes when every
    enabled provider succeeded, though: if any enabled provider failed
    (or none is present), the run skips both writes and preserves the
    existing files, recovering on the next all-success run.

    The read + window-delete + append + global-sort + write run in DuckDB
    and metadata aggregates stream from the written file, so the whole
    on-disk parquet is never loaded into pandas. The optional
    ``pipeline.max_window_days`` cap bounds each run's fetch span; the
    ``backfill_to_present`` driver marches the capped batches forward, so a
    large backfill or outage gap runs in bounded memory across many small
    batches instead of one giant run.

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
        logger.info('Initializing UtilizationPipeline from config: %s', config_path)
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
        Execute one full run.

        Determines the fetch window from prior metadata and config,
        fetches each provider's bundle (isolating failures), runs the
        per-provider transforms, unifies, then -- only if every enabled
        provider succeeded -- merges the window into the existing parquet
        (DuckDB delete-then-append), writes parquet, and writes metadata
        from a streaming aggregate over the written file.

        Returns:
            A ``UtilizationRunResult`` describing the run. ``written`` is
            ``False`` (and ``row_count`` is 0) on a skipped run -- either
            ``start_date`` computed after ``end_date`` (clock skew), or a
            run where an enabled provider failed or no provider was
            present -- in which case the on-disk parquet and metadata are
            left unchanged. ``written`` is ``True`` with the whole-file
            ``row_count`` when the merge wrote the file.

        Raises:
            CorruptUtilizationParquetError: If the existing parquet is
                unreadable or its columns do not match the schema. The
                run aborts before any write, leaving the file intact.
            json.JSONDecodeError: If an existing metadata file is
                malformed (do not silently treat as a first run).
            OSError: If the parquet or metadata write fails. The
                originals (if any) remain intact in this case.
        """
        run_started = datetime.now(UTC)
        wall_start = time_module.monotonic()
        logger.info('UtilizationPipeline run starting at %s', run_started.isoformat())

        prior_metadata = self._metadata_store.load()
        start_date, end_date = self._determine_window(prior_metadata)

        if start_date > end_date:
            logger.warning(
                'Computed start_date %s is after end_date %s; skipping run',
                start_date,
                end_date,
            )
            return UtilizationRunResult(
                written=False,
                row_count=0,
                start_date=start_date,
                end_date=end_date,
                providers_present=[],
                providers_skipped=[],
                providers_failed=[],
            )

        logger.info('Fetch window: %s to %s (inclusive)', start_date, end_date)

        motive_bundle, motive_status = self._fetch_motive(start_date, end_date)
        samsara_bundle, samsara_status = self._fetch_samsara(start_date, end_date)
        statuses: list[tuple[str, ProviderStatus]] = [
            ('motive', motive_status),
            ('samsara', samsara_status),
        ]
        providers_present = [name for name, status in statuses if status == 'present']
        providers_skipped = [name for name, status in statuses if status == 'skipped']
        providers_failed = [name for name, status in statuses if status == 'failed']

        df: pd.DataFrame = unify(motive_bundle, samsara_bundle)

        should_write = bool(providers_present) and not providers_failed
        if not should_write:
            logger.warning(
                'Skipping write: providers_failed=%s, providers_skipped=%s. '
                'Parquet and metadata writes are skipped and existing data is '
                'preserved; the next run where every enabled provider succeeds '
                'will re-fetch the full lookback window and rewrite it.',
                providers_failed,
                providers_skipped,
            )
            return UtilizationRunResult(
                written=False,
                row_count=0,
                start_date=start_date,
                end_date=end_date,
                providers_present=providers_present,
                providers_skipped=providers_skipped,
                providers_failed=providers_failed,
            )

        stats = merge_incremental_to_parquet(
            df,
            start_date,
            end_date,
            self._parquet_path,
            compression=self._config.storage.parquet_compression or 'uncompressed',
        )
        row_count = stats.final_row_count
        logger.info('Wrote %d rows to %s', row_count, self._parquet_path)

        run_completed = datetime.now(UTC)
        self._metadata_store.write(
            MetadataBuildContext(
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
            row_count,
            time_module.monotonic() - wall_start,
        )
        return UtilizationRunResult(
            written=True,
            row_count=row_count,
            start_date=start_date,
            end_date=end_date,
            providers_present=providers_present,
            providers_skipped=providers_skipped,
            providers_failed=providers_failed,
        )

    # --------------------------------------------------------------
    # Batched backfill driver
    # --------------------------------------------------------------

    def backfill_to_present(self) -> BackfillSummary:
        """March bounded batched runs forward until the file reaches the present.

        Calls ``run()`` repeatedly. Each call reads the metadata the prior
        batch wrote, so ``latest_data_date`` -- and therefore the fetch
        window -- advances batch to batch, while the ``max_window_days`` cap
        keeps every batch's fetch + transform bounded in memory. Because each
        window is delete-then-appended into the growing file (prompts 1-2 + A),
        overlapping batch boundaries neither duplicate nor overwrite data.

        Each batch re-fetches the prior ``lookback_days`` of overlap; that is
        correct (the merge dedupes by window) and is the price of a one-time
        backfill. A smaller ``lookback_days`` trims the redundant API volume.

        Returns:
            A ``BackfillSummary`` for a march that reached ``today_utc - 1``
            (``caught_up=True``).

        Raises:
            ValueError: If ``max_window_days`` is unset -- an uncapped backfill
                is the single-shot out-of-memory failure this driver avoids.
            BackfillStalledError: If a batch wrote nothing (an enabled provider
                failed), the window stopped advancing, or the iteration ceiling
                was hit. Existing data is preserved; re-running resumes.

        Side Effects:
            Performs repeated fetches and parquet/metadata writes -- one set
            per batch. Logs per-batch progress at INFO.
        """
        max_window_days = self._config.pipeline.max_window_days
        if max_window_days is None:
            raise ValueError(
                'backfill_to_present requires pipeline.max_window_days to be set; '
                'an uncapped backfill fetches the entire span in a single run -- '
                'the out-of-memory failure this driver exists to avoid'
            )

        iteration_ceiling = self._backfill_iteration_ceiling(max_window_days)
        batches_run = 0
        prior_end_date: date | None = None

        while batches_run < iteration_ceiling:
            result = self.run()
            batches_run += 1
            logger.info(
                'Backfill batch %d/%d: window %s..%s, written=%s, file rows=%d',
                batches_run,
                iteration_ceiling,
                result.start_date,
                result.end_date,
                result.written,
                result.row_count,
            )

            if not result.written:
                raise BackfillStalledError(
                    f'Backfill stalled at batch {batches_run}: the run wrote '
                    f'nothing (providers_failed={result.providers_failed}). '
                    f'Existing data is preserved; resolve the provider and '
                    f're-run backfill_to_present to resume.'
                )

            today_minus_one = (datetime.now(UTC) - timedelta(days=1)).date()
            if result.end_date >= today_minus_one:
                return BackfillSummary(
                    batches_run=batches_run,
                    final_end_date=result.end_date,
                    caught_up=True,
                    final_row_count=result.row_count,
                )

            if prior_end_date is not None and result.end_date <= prior_end_date:
                raise BackfillStalledError(
                    f'Backfill made no forward progress at batch {batches_run}: '
                    f'end_date {result.end_date} did not advance past '
                    f'{prior_end_date}.'
                )
            prior_end_date = result.end_date

        raise BackfillStalledError(
            f'Backfill exceeded its iteration ceiling ({iteration_ceiling}) '
            f'without reaching the present; aborting to avoid an unbounded loop.'
        )

    def _backfill_iteration_ceiling(self, max_window_days: int) -> int:
        """Upper bound on backfill batches: total span / per-batch advance + buffer.

        A belt-and-suspenders ceiling against a window-advance logic bug, so
        the march can never loop unboundedly even if the per-batch progress
        check is somehow defeated. Each batch advances the anchor by roughly
        ``max_window_days - lookback_days`` days.

        Args:
            max_window_days: The configured cap (already known to be set).

        Returns:
            The maximum number of batches the march may attempt.
        """
        advance_per_batch = max(
            1, max_window_days - self._config.pipeline.lookback_days
        )
        first_start, _ = self._determine_window(self._metadata_store.load())
        today_minus_one = (datetime.now(UTC) - timedelta(days=1)).date()
        total_days = max((today_minus_one - first_start).days, 0)
        return math.ceil(total_days / advance_per_batch) + _BACKFILL_ITERATION_BUFFER

    # --------------------------------------------------------------
    # Window determination
    # --------------------------------------------------------------

    def _determine_window(
        self, prior_metadata: dict[str, Any] | None
    ) -> tuple[date, date]:
        """
        Compute ``(start_date, end_date)`` inclusive for the fetch.

        ``start_date`` is the configured default on a first run, otherwise
        ``latest_data_date - lookback_days`` from prior metadata.

        ``end_date`` is ``today_utc - 1`` (never the current incomplete UTC
        day), optionally capped to ``start_date + max_window_days`` when
        ``max_window_days`` is set. The cap is a no-op in steady state (a
        recent ``start_date`` plus the span already reaches ``today-1``) and
        bites only when a large gap -- a fresh backfill or an outage
        recovery -- would otherwise fetch the whole span in one run.
        ``backfill_to_present`` marches the capped batches forward.
        """
        uncapped_end = (datetime.now(UTC) - timedelta(days=1)).date()
        prior_latest = (
            prior_metadata.get('latest_data_date')
            if prior_metadata is not None
            else None
        )
        if prior_latest is None:
            start_date = date.fromisoformat(self._config.pipeline.default_start_date)
        else:
            latest = date.fromisoformat(prior_latest)
            start_date = latest - timedelta(days=self._config.pipeline.lookback_days)

        max_window_days = self._config.pipeline.max_window_days
        if max_window_days is None:
            end_date = uncapped_end
        else:
            end_date = min(uncapped_end, start_date + timedelta(days=max_window_days))
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
