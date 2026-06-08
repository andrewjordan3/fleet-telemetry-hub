"""Tests for the ``company`` config field plumbing.

Covers the ``company`` value's path from ``ProviderConfig`` (set or
default-None) through ``Provider`` (kwarg + property + ``from_config``)
into the utilization bundles (``MotiveUtilizationBundle`` and
``SamsaraUtilizationBundle``).

The unifier owns any normalization of the company string; these tests
only verify the value flows through unchanged.
"""

import dataclasses
from datetime import UTC, date, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from pydantic import SecretStr

from fleet_telemetry_hub.client import TelemetryClient
from fleet_telemetry_hub.config.config_models import (
    ProviderConfig,
    TelemetryConfig,
)
from fleet_telemetry_hub.models.shared_response_models import ProviderCredentials
from fleet_telemetry_hub.provider import Provider
from fleet_telemetry_hub.utilization import (
    MotiveUtilizationBundle,
    MotiveUtilizationFetcher,
    SamsaraUtilizationBundle,
    SamsaraUtilizationFetcher,
)

_TEST_COMPANY = 'test_fleet_co'
_MAY_14 = date(2026, 5, 14)
_MAY_14_START = datetime(2026, 5, 14, 0, 0, 0, tzinfo=UTC)
_MAY_15_START = datetime(2026, 5, 15, 0, 0, 0, tzinfo=UTC)


def _make_credentials() -> ProviderCredentials:
    return ProviderCredentials(
        base_url='https://api.example.com',
        api_key=SecretStr('test_key'),
    )


def _build_inert_fake_client_and_provider(
    *, company: str | None = None
) -> tuple[MagicMock, Provider]:
    """Return a (fake_client, real Provider) pair wired together.

    The fake client returns empty iterators for every endpoint so the
    fetchers run to completion with no records. We use a real
    ``Provider`` instance here (not a ``MagicMock(spec=Provider)``)
    so the ``company`` property returns the configured value rather
    than a mock attribute.
    """

    fake_client = MagicMock(spec=TelemetryClient)
    fake_client.__enter__.return_value = fake_client
    fake_client.__exit__.return_value = None
    fake_client.fetch_all.return_value = iter([])

    provider = Provider(
        name='motive',
        credentials=_make_credentials(),
        company=company,
    )
    # Bypass the real HTTP client construction by patching the bound method.
    provider.client = lambda *_args, **_kwargs: fake_client  # type: ignore[method-assign]

    return fake_client, provider


class TestProviderConfigCompany:
    """``ProviderConfig.company`` defaults to ``None`` and accepts a string."""

    @staticmethod
    def _minimum_provider_config(**overrides: Any) -> ProviderConfig:
        base: dict[str, Any] = {
            'enabled': True,
            'base_url': 'https://api.example.com',
            'api_key': SecretStr('key'),
            'request_timeout': (10, 30),
            'max_retries': 3,
            'retry_backoff_factor': 2.0,
            'verify_ssl': True,
            'rate_limit_requests_per_second': 10,
        }
        base.update(overrides)
        return ProviderConfig(**base)

    def test_default_company_is_none(self) -> None:
        """A ``ProviderConfig`` constructed without ``company`` has ``company is None``."""

        config = self._minimum_provider_config()

        assert config.company is None

    def test_explicit_company_is_stored(self) -> None:
        """A ``ProviderConfig`` constructed with ``company`` stores the string."""

        config = self._minimum_provider_config(company=_TEST_COMPANY)

        assert config.company == _TEST_COMPANY


class TestProviderConfigYamlLoading:
    """``TelemetryConfig.model_validate`` round-trips company through YAML."""

    @staticmethod
    def _minimum_yaml_dict(*, company_line: str | None) -> dict[str, Any]:
        provider_block: dict[str, Any] = {
            'enabled': True,
            'base_url': 'https://api.example.com',
            'api_key': 'k',
            'request_timeout': [10, 30],
            'max_retries': 3,
            'retry_backoff_factor': 2.0,
            'verify_ssl': True,
            'rate_limit_requests_per_second': 10,
        }
        if company_line is not None:
            provider_block['company'] = company_line
        return {
            'providers': {'motive': provider_block, 'samsara': provider_block},
            'pipeline': {
                'default_start_date': '2024-01-01',
                'lookback_days': 7,
                'batch_increment_days': 1.0,
                'request_delay_seconds': 0.5,
                'use_truststore': False,
            },
            'storage': {'parquet_path': 'data/', 'parquet_compression': 'snappy'},
            'logging': {
                'file_path': 'logs/x.log',
                'console_level': 'INFO',
                'file_level': 'DEBUG',
            },
        }

    def test_yaml_without_company_defaults_to_none(self) -> None:
        """A YAML block omitting ``company`` parses with ``company is None``."""

        raw = yaml.safe_dump(self._minimum_yaml_dict(company_line=None))
        config = TelemetryConfig.model_validate(yaml.safe_load(raw))

        assert config.providers['motive'].company is None

    def test_yaml_with_company_carries_through(self) -> None:
        """A YAML block setting ``company`` parses with that string preserved."""

        raw = yaml.safe_dump(self._minimum_yaml_dict(company_line=_TEST_COMPANY))
        config = TelemetryConfig.model_validate(yaml.safe_load(raw))

        assert config.providers['motive'].company == _TEST_COMPANY


class TestProviderCompany:
    """``Provider`` stores ``company`` and exposes it via property."""

    def test_provider_without_company_kwarg_defaults_to_none(self) -> None:
        """Constructing ``Provider`` without ``company`` yields ``.company is None``."""

        provider = Provider(name='motive', credentials=_make_credentials())

        assert provider.company is None

    def test_provider_with_company_kwarg_stores_value(self) -> None:
        """Constructing ``Provider(company=X)`` exposes ``.company == X``."""

        provider = Provider(
            name='motive',
            credentials=_make_credentials(),
            company=_TEST_COMPANY,
        )

        assert provider.company == _TEST_COMPANY

    def test_from_config_propagates_configured_company(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """``Provider.from_config`` reads ``ProviderConfig.company`` through."""

        # Inject a company onto the existing motive ProviderConfig.
        original = telemetry_config.providers['motive']
        updated = original.model_copy(update={'company': _TEST_COMPANY})
        telemetry_config.providers['motive'] = updated

        provider = Provider.from_config('motive', telemetry_config)

        assert provider.company == _TEST_COMPANY

    def test_from_config_propagates_default_none(
        self,
        telemetry_config: TelemetryConfig,
    ) -> None:
        """``Provider.from_config`` with no ``company`` configured yields ``None``."""

        # conftest's telemetry_config fixture does not set company,
        # so the default should propagate.
        provider = Provider.from_config('motive', telemetry_config)

        assert provider.company is None


class TestBundleCompanyField:
    """Both utilization bundles accept and surface ``company``."""

    def test_motive_bundle_company_none_round_trip(self) -> None:
        """``MotiveUtilizationBundle(company=None)`` exposes ``.company is None``."""

        bundle = MotiveUtilizationBundle(
            driving_periods=[],
            idle_events=[],
            date_range=(_MAY_14, _MAY_14),
            company=None,
        )

        assert bundle.company is None

    def test_motive_bundle_company_string_round_trip(self) -> None:
        """``MotiveUtilizationBundle(company=X)`` exposes ``.company == X``."""

        bundle = MotiveUtilizationBundle(
            driving_periods=[],
            idle_events=[],
            date_range=(_MAY_14, _MAY_14),
            company=_TEST_COMPANY,
        )

        assert bundle.company == _TEST_COMPANY

    def test_samsara_bundle_company_none_round_trip(self) -> None:
        """``SamsaraUtilizationBundle(company=None)`` exposes ``.company is None``."""

        bundle = SamsaraUtilizationBundle(
            vehicles=[],
            drivers=[],
            trips=[],
            idling_events=[],
            date_range=(_MAY_14, _MAY_14),
            company=None,
        )

        assert bundle.company is None

    def test_samsara_bundle_company_string_round_trip(self) -> None:
        """``SamsaraUtilizationBundle(company=X)`` exposes ``.company == X``."""

        bundle = SamsaraUtilizationBundle(
            vehicles=[],
            drivers=[],
            trips=[],
            idling_events=[],
            date_range=(_MAY_14, _MAY_14),
            company=_TEST_COMPANY,
        )

        assert bundle.company == _TEST_COMPANY

    def test_motive_bundle_company_is_frozen(self) -> None:
        """Assigning to ``bundle.company`` raises ``FrozenInstanceError``."""

        bundle = MotiveUtilizationBundle(
            driving_periods=[],
            idle_events=[],
            date_range=(_MAY_14, _MAY_14),
            company=None,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            bundle.company = 'x'  # pyright: ignore[reportAttributeAccessIssue]

    def test_samsara_bundle_company_is_frozen(self) -> None:
        """Assigning to ``bundle.company`` raises ``FrozenInstanceError``."""

        bundle = SamsaraUtilizationBundle(
            vehicles=[],
            drivers=[],
            trips=[],
            idling_events=[],
            date_range=(_MAY_14, _MAY_14),
            company=None,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            bundle.company = 'x'  # pyright: ignore[reportAttributeAccessIssue]


class TestFetcherCompanyPropagation:
    """Both fetchers populate ``bundle.company`` from ``provider.company``."""

    def test_motive_fetcher_propagates_none_company(self) -> None:
        """Provider without company → Motive bundle has ``.company is None``."""

        _, provider = _build_inert_fake_client_and_provider(company=None)
        fetcher = MotiveUtilizationFetcher(provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_14)

        assert bundle.company is None

    def test_motive_fetcher_propagates_configured_company(self) -> None:
        """Provider with company → Motive bundle carries the same string."""

        _, provider = _build_inert_fake_client_and_provider(company=_TEST_COMPANY)
        fetcher = MotiveUtilizationFetcher(provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_14)

        assert bundle.company == _TEST_COMPANY

    def test_samsara_fetcher_propagates_none_company(self) -> None:
        """Provider without company → Samsara bundle has ``.company is None``."""

        _, provider = _build_inert_fake_client_and_provider(company=None)
        fetcher = SamsaraUtilizationFetcher(provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_14)

        assert bundle.company is None

    def test_samsara_fetcher_propagates_configured_company(self) -> None:
        """Provider with company → Samsara bundle carries the same string."""

        _, provider = _build_inert_fake_client_and_provider(company=_TEST_COMPANY)
        fetcher = SamsaraUtilizationFetcher(provider)

        bundle = fetcher.fetch(_MAY_14, _MAY_14)

        assert bundle.company == _TEST_COMPANY
