"""Tests for ParameterType.UNIX_MS and EndpointDefinition._serialize_unix_ms.

These tests exercise the base-class serializer (registered for
``ParameterType.UNIX_MS`` in ``EndpointDefinition._serialize_parameter_value``)
through a real endpoint instance -- ``SamsaraEndpoints.TRIPS`` -- which
is the first endpoint in the codebase to use it. The serializer is
provider-agnostic: it lives on the base class, not on
``SamsaraEndpointDefinition``.
"""

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from fleet_telemetry_hub.models.samsara_requests import SamsaraEndpoints
from fleet_telemetry_hub.models.shared_response_models import ParameterType

# May 14 2026 13:00:00 UTC -- chosen so the equivalent America/Chicago
# instant (CDT, UTC-5) is 08:00 local, which makes the cross-zone
# equivalence test obvious to read.
_REF_UTC = datetime(2026, 5, 14, 13, 0, 0, tzinfo=UTC)
_REF_MS = int(_REF_UTC.timestamp() * 1000)


class TestSerializeUnixMs:
    """Direct unit tests for the _serialize_unix_ms handler."""

    def test_tz_aware_utc_datetime_round_trips(self) -> None:
        """A tz-aware UTC datetime serializes to its expected epoch-ms string."""

        out = SamsaraEndpoints.TRIPS._serialize_unix_ms(_REF_UTC)  # pyright: ignore[reportPrivateUsage]

        assert out == str(_REF_MS)

    def test_tz_aware_non_utc_equals_equivalent_utc(self) -> None:
        """A tz-aware non-UTC datetime serializes to the same ms as the UTC instant."""

        chicago_local = _REF_UTC.astimezone(ZoneInfo('America/Chicago'))

        out_chicago = SamsaraEndpoints.TRIPS._serialize_unix_ms(chicago_local)  # pyright: ignore[reportPrivateUsage]
        out_utc = SamsaraEndpoints.TRIPS._serialize_unix_ms(_REF_UTC)  # pyright: ignore[reportPrivateUsage]

        assert out_chicago == out_utc
        assert chicago_local.tzinfo is not None  # sanity: actually tz-aware
        assert chicago_local.tzinfo is not UTC

    def test_naive_datetime_assumed_utc(self) -> None:
        """A naive datetime is treated as UTC, matching the SamsaraEndpoint DATETIME convention."""

        # A naive datetime is intentional here: the serializer's contract is
        # to assume UTC for naive input, and this test pins that behavior.
        naive = datetime(2026, 5, 14, 13, 0, 0)  # noqa: DTZ001

        out = SamsaraEndpoints.TRIPS._serialize_unix_ms(naive)  # pyright: ignore[reportPrivateUsage]

        assert out == str(_REF_MS)

    def test_output_has_no_decimal_point(self) -> None:
        """Output is an integer-ms string, never a fractional/float form."""

        out = SamsaraEndpoints.TRIPS._serialize_unix_ms(_REF_UTC)  # pyright: ignore[reportPrivateUsage]

        assert '.' not in out

    @pytest.mark.parametrize('bad_value', [123456, '2026-05-14T13:00:00Z', None])
    def test_non_datetime_input_raises_type_error(self, bad_value: object) -> None:
        """Non-datetime input raises TypeError with a clear message."""

        with pytest.raises(
            TypeError, match='UNIX_MS parameter requires datetime input'
        ):
            SamsaraEndpoints.TRIPS._serialize_unix_ms(bad_value)  # pyright: ignore[reportPrivateUsage]


class TestSerializeParameterValueUnixMsDispatch:
    """UNIX_MS is reachable through the public _serialize_parameter_value dispatch."""

    def test_dispatch_routes_unix_ms_to_handler(self) -> None:
        """_serialize_parameter_value(..., UNIX_MS) produces the integer-ms string."""

        out = SamsaraEndpoints.TRIPS._serialize_parameter_value(  # pyright: ignore[reportPrivateUsage]
            _REF_UTC,
            ParameterType.UNIX_MS,
        )

        assert out == str(_REF_MS)


class TestSerializeUnixMsCrossZoneEquivalence:
    """Cross-zone instants that name the same moment produce identical ms output."""

    def test_offset_naming_same_instant(self) -> None:
        """A fixed-offset datetime equivalent to the UTC reference round-trips identically."""

        offset_value = _REF_UTC.astimezone(timezone(timedelta(hours=-5)))

        out_offset = SamsaraEndpoints.TRIPS._serialize_unix_ms(offset_value)  # pyright: ignore[reportPrivateUsage]
        out_utc = SamsaraEndpoints.TRIPS._serialize_unix_ms(_REF_UTC)  # pyright: ignore[reportPrivateUsage]

        assert out_offset == out_utc
