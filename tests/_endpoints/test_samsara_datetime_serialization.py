"""Regression tests for SamsaraEndpointDefinition._serialize_parameter_value.

These tests lock in the tzinfo-aware UTC conversion fix introduced
alongside the /v2/vehicle_utilization work. The previous implementation
called strftime directly on the supplied datetime, silently emitting
local-time numerals with a 'Z' suffix for any non-UTC tz-aware input.
A revert of that fix would be caught by
test_tz_aware_non_utc_datetime_converts_to_utc below.
"""

from datetime import datetime, timedelta, timezone

from fleet_telemetry_hub.models.samsara_requests import SamsaraEndpoints


class TestSamsaraDatetimeSerialization:
    """Tests for Samsara's RFC 3339 (Z suffix) datetime serialization."""

    def test_tz_aware_utc_datetime_serializes_with_z_suffix(self) -> None:
        """Should serialize tz-aware UTC datetimes with a Z suffix."""

        utc_value = datetime(2026, 5, 6, 0, 0, 0, tzinfo=timezone.utc)

        query = SamsaraEndpoints.VEHICLE_STATS_HISTORY.build_query_params(
            start_time=utc_value,
            end_time=utc_value,
            types=['gps'],
        )

        assert query['startTime'] == '2026-05-06T00:00:00Z'

    def test_tz_aware_non_utc_datetime_converts_to_utc(self) -> None:
        """Should convert tz-aware non-UTC datetimes to UTC before formatting."""

        chicago_offset = timezone(timedelta(hours=-5))
        chicago_value = datetime(2026, 5, 6, 17, 0, 0, tzinfo=chicago_offset)

        query = SamsaraEndpoints.VEHICLE_STATS_HISTORY.build_query_params(
            start_time=chicago_value,
            end_time=chicago_value,
            types=['gps'],
        )

        # Regression assertion: a revert would emit '2026-05-06T17:00:00Z'.
        assert query['startTime'] == '2026-05-06T22:00:00Z'

    def test_naive_datetime_assumed_utc(self) -> None:
        """Should format naive datetimes as-is and append Z."""

        naive_value = datetime(2026, 5, 6, 0, 0, 0)

        query = SamsaraEndpoints.VEHICLE_STATS_HISTORY.build_query_params(
            start_time=naive_value,
            end_time=naive_value,
            types=['gps'],
        )

        assert query['startTime'] == '2026-05-06T00:00:00Z'

    def test_string_list_param_still_delegates_to_parent(self) -> None:
        """Should serialize STRING_LIST params as comma-separated values."""

        utc_value = datetime(2026, 5, 6, 0, 0, 0, tzinfo=timezone.utc)

        query = SamsaraEndpoints.VEHICLE_STATS_HISTORY.build_query_params(
            start_time=utc_value,
            end_time=utc_value,
            types=['gps', 'engineStates'],
        )

        assert query['types'] == 'gps,engineStates'
