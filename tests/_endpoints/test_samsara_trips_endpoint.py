"""Tests for the Samsara /v1/fleet/trips endpoint and its Trip model.

Covers:
- Trip model parsing with epoch-ms → tz-aware UTC datetime coercion,
  optional fields, ignored extras, and missing-required failures.
- SamsaraEndpoints.TRIPS surface (path, pagination, query-parameter
  shape) and registry resolution.
- build_query_params round-trip emitting integer Unix epoch-ms
  strings under the camelCase API names.
- parse_response producing a ParsedResponse with typed Trip items.

All identifiers in test data are synthetic. No real driver IDs,
vehicle IDs, or trip UUIDs from any production fleet appear in this
file.
"""

import logging
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic.main import BaseModel

from fleet_telemetry_hub.models import EndpointDefinition
from fleet_telemetry_hub.models.samsara_requests import (
    SamsaraEndpointDefinition,
    SamsaraEndpoints,
)
from fleet_telemetry_hub.models.samsara_responses import Trip, TripsResponse
from fleet_telemetry_hub.models.shared_request_models import HTTPMethod
from fleet_telemetry_hub.models.shared_response_models import ParameterType
from fleet_telemetry_hub.registry import EndpointRegistry

# Anchor time fixtures to the same date constants used elsewhere in
# the suite. UTC epoch-ms values precomputed for clarity.
_TRIP_A_START_UTC = datetime(2026, 5, 14, 13, 0, 0, tzinfo=UTC)
_TRIP_A_END_UTC = datetime(2026, 5, 14, 15, 0, 0, tzinfo=UTC)
_TRIP_B_START_UTC = datetime(2026, 5, 14, 15, 5, 0, tzinfo=UTC)
_TRIP_B_END_UTC = datetime(2026, 5, 14, 15, 20, 0, tzinfo=UTC)

_TRIP_A_START_MS = int(_TRIP_A_START_UTC.timestamp() * 1000)
_TRIP_A_END_MS = int(_TRIP_A_END_UTC.timestamp() * 1000)
_TRIP_B_START_MS = int(_TRIP_B_START_UTC.timestamp() * 1000)
_TRIP_B_END_MS = int(_TRIP_B_END_UTC.timestamp() * 1000)

_TRIP_A_ID = '00000000-0000-0000-0000-000000001001'
_TRIP_B_ID = '00000000-0000-0000-0000-000000001002'

_VEHICLE_ID = '999999900000001'
# Samsara returns ``driverId`` as an integer; the Trip model coerces
# to string. Both forms are kept here so assertions can read against
# the post-coercion value while the fixture uses the on-wire shape.
_DRIVER_ID_INT = 7046697
_DRIVER_ID_STR = str(_DRIVER_ID_INT)

_EXPECTED_TRIP_COUNT = 2
_TRIP_A_DISTANCE_METERS = 12500
_TRIP_B_DISTANCE_METERS = 800

TRIPS_FIXTURE: dict[str, Any] = {
    'data': [
        {
            # The real API does not echo ``vehicleId`` back per-trip;
            # the fetcher boundary stamps it on via ``VehicleTrip``.
            'id': _TRIP_A_ID,
            'driverId': _DRIVER_ID_INT,
            'startMs': _TRIP_A_START_MS,
            'endMs': _TRIP_A_END_MS,
            'distanceMeters': _TRIP_A_DISTANCE_METERS,
            # Extras Samsara returns that V1 intentionally does not model
            # but that exercise the ``extra='ignore'`` config. Kept here
            # so a future field-shape regression in pydantic or the base
            # model would surface as a parse failure.
            'startLocation': 'Roselawn Street, Pomona, CA',
            'endLocation': 'Lowell Avenue, Claremont, CA',
            'startCoordinates': {'latitude': 34.056405, 'longitude': -117.791372},
            'endCoordinates': {'latitude': 34.114833, 'longitude': -117.711276},
            'fuelConsumedMl': 6000,
            'tollMeters': 0,
            'codriverIds': [],
            'startOdometer': 123933425,
            'endOdometer': 123947925,
            'assetIds': [],
        },
        {
            'id': _TRIP_B_ID,
            'driverId': None,
            'startMs': _TRIP_B_START_MS,
            'endMs': _TRIP_B_END_MS,
            'distanceMeters': _TRIP_B_DISTANCE_METERS,
        },
    ],
    # No ``pagination`` key: the legacy /v1/fleet/trips endpoint does
    # not return pagination metadata; everything for the queried
    # vehicle/window comes back in one response.
}


class TestTripModelParsing:
    """Trip model parsing covers epoch-ms coercion, optionals, and extras."""

    def test_full_record_parses_with_all_v1_fields(self) -> None:
        """A populated record produces a Trip with every V1 field set."""

        trip = Trip.model_validate(TRIPS_FIXTURE['data'][0])

        assert trip.trip_id == _TRIP_A_ID
        assert trip.driver_id == _DRIVER_ID_STR
        assert trip.distance_meters == _TRIP_A_DISTANCE_METERS

    def test_start_time_parses_to_tz_aware_utc(self) -> None:
        """startMs (int) is coerced into tz-aware UTC datetime matching the input."""

        trip = Trip.model_validate(TRIPS_FIXTURE['data'][0])

        assert trip.start_time.tzinfo is UTC
        assert trip.end_time.tzinfo is UTC
        assert trip.start_time == _TRIP_A_START_UTC
        assert trip.end_time == _TRIP_A_END_UTC

    def test_driver_id_none_round_trips(self) -> None:
        """driverId: None in the payload yields trip.driver_id is None."""

        trip = Trip.model_validate(TRIPS_FIXTURE['data'][1])

        assert trip.driver_id is None

    def test_missing_id_key_leaves_trip_id_none(self) -> None:
        """Omitting the id key leaves trip_id as None (default)."""

        payload = dict(TRIPS_FIXTURE['data'][0])
        del payload['id']

        trip = Trip.model_validate(payload)

        assert trip.trip_id is None

    def test_extras_are_silently_ignored(self) -> None:
        """Non-modeled fields don't raise and don't appear as attributes."""

        trip = Trip.model_validate(TRIPS_FIXTURE['data'][0])

        for unmodeled in (
            'startLocation',
            'endLocation',
            'startCoordinates',
            'endCoordinates',
            'fuelConsumedMl',
            'tollMeters',
            'codriverIds',
            'startOdometer',
            'endOdometer',
            'assetIds',
            'start_odometer',
        ):
            assert not hasattr(trip, unmodeled)

    def test_datetime_input_passes_through(self) -> None:
        """Direct datetime input (vs. epoch-ms int) is accepted unchanged."""

        payload = {
            'id': _TRIP_A_ID,
            'driverId': _DRIVER_ID_INT,
            'startMs': _TRIP_A_START_UTC,
            'endMs': _TRIP_A_END_UTC,
            'distanceMeters': _TRIP_A_DISTANCE_METERS,
        }

        trip = Trip.model_validate(payload)

        assert trip.start_time == _TRIP_A_START_UTC
        assert trip.end_time == _TRIP_A_END_UTC

    @pytest.mark.parametrize(
        'missing_key',
        ['startMs', 'endMs', 'distanceMeters'],
    )
    def test_missing_required_field_raises_validation_error(
        self, missing_key: str
    ) -> None:
        """Removing any required key produces a ValidationError."""

        payload = dict(TRIPS_FIXTURE['data'][0])
        del payload[missing_key]

        with pytest.raises(ValidationError):
            Trip.model_validate(payload)


class TestTripDriverIdCoercion:
    """``Trip._coerce_driver_id`` normalizes the on-wire shape to ``str | None``."""

    @staticmethod
    def _payload(driver_id_value: int | str | None) -> dict[str, Any]:
        """Build a minimal-but-valid Trip payload with the given ``driverId`` value."""
        return {
            'id': _TRIP_A_ID,
            'driverId': driver_id_value,
            'startMs': _TRIP_A_START_MS,
            'endMs': _TRIP_A_END_MS,
            'distanceMeters': _TRIP_A_DISTANCE_METERS,
        }

    def test_realistic_int_driver_id_is_coerced_to_string(self) -> None:
        """A large int (the real Samsara shape) round-trips as its decimal string."""

        trip = Trip.model_validate(self._payload(_DRIVER_ID_INT))

        assert trip.driver_id == _DRIVER_ID_STR

    def test_zero_int_driver_id_is_coerced_to_zero_string(self) -> None:
        """``driverId=0`` is a plain coercion, not a sentinel for null."""

        trip = Trip.model_validate(self._payload(0))

        assert trip.driver_id == '0'

    def test_none_driver_id_passes_through_as_none(self) -> None:
        """``driverId=None`` is preserved (unattributed trip)."""

        trip = Trip.model_validate(self._payload(None))

        assert trip.driver_id is None

    def test_string_driver_id_passes_through_unchanged(self) -> None:
        """Defensive symmetry: an already-string id is not re-stringified."""

        trip = Trip.model_validate(self._payload('1000001'))

        assert trip.driver_id == '1000001'


class TestTripsResponseShape:
    """TripsResponse mirrors the IdlingEventsResponse container shape."""

    def test_get_items_unwraps_to_flat_trip_list(self) -> None:
        """response.get_items() returns the data list typed as Trip."""

        parsed = TripsResponse.model_validate(TRIPS_FIXTURE)

        items = parsed.get_items()

        assert len(items) == _EXPECTED_TRIP_COUNT
        assert all(isinstance(item, Trip) for item in items)


class TestSamsaraTripsEndpointDefinition:
    """Surface assertions on SamsaraEndpoints.TRIPS."""

    def test_endpoint_path_is_v1_fleet_trips(self) -> None:
        """Should expose the /v1/fleet/trips path."""

        assert SamsaraEndpoints.TRIPS.endpoint_path == '/v1/fleet/trips'

    def test_http_method_is_get(self) -> None:
        """Should use HTTP GET."""

        assert SamsaraEndpoints.TRIPS.http_method == HTTPMethod.GET

    def test_endpoint_is_not_paginated(self) -> None:
        """The legacy /v1/fleet/trips endpoint does not paginate."""

        assert SamsaraEndpoints.TRIPS.is_paginated is False

    def test_response_model_and_item_extractor(self) -> None:
        """Should wire TripsResponse and the uniform get_items extractor."""

        assert SamsaraEndpoints.TRIPS.response_model is TripsResponse
        assert SamsaraEndpoints.TRIPS.item_extractor_method == 'get_items'

    def test_query_parameter_shape(self) -> None:
        """Should declare vehicle_id, start_time, end_time as required params."""

        specs_by_name = {
            spec.name: spec for spec in SamsaraEndpoints.TRIPS.query_parameters
        }
        assert set(specs_by_name) == {'vehicle_id', 'start_time', 'end_time'}
        for spec in specs_by_name.values():
            assert spec.required is True

        assert specs_by_name['vehicle_id'].api_name == 'vehicleId'
        assert specs_by_name['vehicle_id'].parameter_type == ParameterType.STRING

        assert specs_by_name['start_time'].api_name == 'startMs'
        assert specs_by_name['start_time'].parameter_type == ParameterType.UNIX_MS

        assert specs_by_name['end_time'].api_name == 'endMs'
        assert specs_by_name['end_time'].parameter_type == ParameterType.UNIX_MS

    def test_build_query_params_emits_integer_ms_under_camel_case_keys(self) -> None:
        """build_query_params produces camelCase keys with integer-ms time values."""

        query = SamsaraEndpoints.TRIPS.build_query_params(
            vehicle_id=_VEHICLE_ID,
            start_time=_TRIP_A_START_UTC,
            end_time=_TRIP_A_END_UTC,
        )

        assert query == {
            'vehicleId': _VEHICLE_ID,
            'startMs': str(_TRIP_A_START_MS),
            'endMs': str(_TRIP_A_END_MS),
        }

    def test_parse_response_returns_typed_trip_items(self) -> None:
        """parse_response yields a ParsedResponse with two Trip instances."""

        parsed = SamsaraEndpoints.TRIPS.parse_response(TRIPS_FIXTURE)

        assert len(parsed.items) == _EXPECTED_TRIP_COUNT
        assert all(isinstance(item, Trip) for item in parsed.items)
        assert parsed.items[0].driver_id == _DRIVER_ID_STR
        assert parsed.items[1].driver_id is None


class TestSamsaraTripsRegistryResolution:
    """EndpointRegistry picks up TRIPS via SamsaraEndpoints.get_all_endpoints()."""

    def test_registry_resolves_to_trips_constant(self) -> None:
        """Registry returns a SamsaraEndpointDefinition that IS the TRIPS constant."""

        registry = EndpointRegistry()

        endpoint: EndpointDefinition[BaseModel] = registry.get('samsara', 'trips')

        assert isinstance(endpoint, EndpointDefinition)
        assert isinstance(endpoint, SamsaraEndpointDefinition)
        assert endpoint is SamsaraEndpoints.TRIPS
        assert endpoint.endpoint_path == '/v1/fleet/trips'


class TestPaginationMetadataWarning:
    """The missing-pagination WARNING fires only for endpoints that declare paginated."""

    _REQUESTS_LOGGER = 'fleet_telemetry_hub.models.samsara_requests'
    _WARNING_NEEDLE = 'Expected pagination metadata'

    def test_non_paginated_endpoint_without_metadata_is_silent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``TRIPS`` (non-paginated) + payload with no ``pagination`` key -> no WARNING."""

        with caplog.at_level(logging.WARNING, logger=self._REQUESTS_LOGGER):
            SamsaraEndpoints.TRIPS.parse_response(TRIPS_FIXTURE)

        assert not any(
            self._WARNING_NEEDLE in record.message for record in caplog.records
        )

    def test_paginated_endpoint_without_metadata_emits_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A paginated endpoint receiving a response without metadata still warns."""

        # VEHICLES is paginated and its response model carries an
        # optional ``pagination`` field, so omitting the key produces
        # the exact scenario the WARNING was designed to surface.
        vehicles_payload: dict[str, Any] = {'data': []}

        with caplog.at_level(logging.WARNING, logger=self._REQUESTS_LOGGER):
            SamsaraEndpoints.VEHICLES.parse_response(vehicles_payload)

        warn_records = [
            record
            for record in caplog.records
            if self._WARNING_NEEDLE in record.message
        ]
        assert len(warn_records) == 1
        # The WARNING includes the offending endpoint's path.
        assert SamsaraEndpoints.VEHICLES.endpoint_path in warn_records[0].message

    def test_paginated_endpoint_with_metadata_is_silent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Pagination present on a paginated endpoint -> no WARNING (regression guard)."""

        vehicles_payload: dict[str, Any] = {
            'data': [],
            'pagination': {'endCursor': '', 'hasNextPage': False},
        }

        with caplog.at_level(logging.WARNING, logger=self._REQUESTS_LOGGER):
            SamsaraEndpoints.VEHICLES.parse_response(vehicles_payload)

        assert not any(
            self._WARNING_NEEDLE in record.message for record in caplog.records
        )
