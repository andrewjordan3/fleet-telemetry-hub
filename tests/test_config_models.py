"""Tests for ``config.config_models`` validators.

Currently focused on ``PipelineConfig.max_window_days``: the optional
fetch-window cap and its ``> lookback_days`` invariant. Pydantic raises a
``ValidationError`` (a ``ValueError`` subclass) on a bad config, so the
cases assert ``pytest.raises(ValueError)``.
"""

import pytest

from fleet_telemetry_hub.config.config_models import PipelineConfig

# Baseline kwargs for a valid PipelineConfig; individual cases override
# ``max_window_days`` / ``lookback_days``.
_BASE_CONFIG = {
    'default_start_date': '2025-01-01',
    'lookback_days': 7,
    'request_delay_seconds': 0.1,
}

# A cap comfortably greater than the baseline ``lookback_days`` of 7.
_VALID_CAP = 28


class TestMaxWindowDays:
    """The optional ``max_window_days`` cap and its validator."""

    def test_default_is_none(self) -> None:
        """Omitting ``max_window_days`` leaves it ``None`` (uncapped)."""

        config = PipelineConfig(**_BASE_CONFIG)

        assert config.max_window_days is None

    def test_greater_than_lookback_constructs(self) -> None:
        """A cap strictly greater than ``lookback_days`` is accepted."""

        config = PipelineConfig(**_BASE_CONFIG, max_window_days=_VALID_CAP)

        assert config.max_window_days == _VALID_CAP

    def test_equal_to_lookback_raises(self) -> None:
        """A cap equal to ``lookback_days`` raises (no per-batch advance)."""

        with pytest.raises(ValueError, match='strictly'):
            PipelineConfig(**_BASE_CONFIG, max_window_days=7)

    def test_less_than_lookback_raises(self) -> None:
        """A cap below ``lookback_days`` raises (the march would regress)."""

        with pytest.raises(ValueError, match='strictly'):
            PipelineConfig(**_BASE_CONFIG, max_window_days=5)

    def test_zero_raises(self) -> None:
        """A zero cap raises (must be a positive int)."""

        with pytest.raises(ValueError, match='positive'):
            PipelineConfig(**_BASE_CONFIG, max_window_days=0)

    def test_negative_raises(self) -> None:
        """A negative cap raises (must be a positive int)."""

        with pytest.raises(ValueError, match='positive'):
            PipelineConfig(**_BASE_CONFIG, max_window_days=-1)
