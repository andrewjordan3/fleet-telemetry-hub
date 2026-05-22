"""Tests for the Samsara /idling/events response models.

Covers parsing of the anonymized fixture below through
``IdlingEventsResponse`` and its nested models, including the
endpoint-specific quirks documented on the production API:

- ``asset.id`` and ``operator.id`` arrive as JSON integers and are
  coerced to strings.
- ``fuelCost.amount`` arrives as a JSON string and is coerced to float.
- ``airTemperatureMillicelsius`` is optional on individual events.

All identifiers in test data are synthetic. No real vehicle IDs,
driver IDs, or event UUIDs from any production fleet appear in this
file.
"""

from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from fleet_telemetry_hub.models.samsara_responses import (
    IdlingAsset,
    IdlingEvent,
    IdlingEventsResponse,
    IdlingFuelCost,
    IdlingOperator,
)

_IDLING_FIXTURE: dict[str, Any] = {
    'data': [
        {
            'airTemperatureMillicelsius': 12938,
            'asset': {'id': 999999900000005},
            'durationMilliseconds': 331886,
            'eventUuid': '00000000-0000-0000-0000-000000000001',
            'fuelConsumedMilliliters': 451.0712847800302,
            'fuelCost': {'amount': '0.66', 'currency': 'usd'},
            'gaseousFuelConsumedGrams': 0,
            'gaseousFuelCost': {'amount': '0.00', 'currency': 'usd'},
            'operator': {'id': 1000006},
            'ptoState': 'inactive',
            'startTime': '2026-05-14T13:13:01.078Z',
            'latitude': 30.0,
            'longitude': -90.0,
        },
        {
            'asset': {'id': 999999900000005},
            'durationMilliseconds': 122217,
            'eventUuid': '00000000-0000-0000-0000-000000000002',
            'fuelConsumedMilliliters': 161.48796672383585,
            'fuelCost': {'amount': '0.23', 'currency': 'usd'},
            'gaseousFuelConsumedGrams': 0,
            'gaseousFuelCost': {'amount': '0.00', 'currency': 'usd'},
            'operator': {'id': 1000006},
            'ptoState': 'inactive',
            'startTime': '2026-05-14T14:03:07.381Z',
            'latitude': 30.1,
            'longitude': -90.1,
        },
        {
            'airTemperatureMillicelsius': 26688,
            'asset': {'id': 999999900000005},
            'durationMilliseconds': 216216,
            'eventUuid': '00000000-0000-0000-0000-000000000003',
            'fuelConsumedMilliliters': 357.7484364141765,
            'fuelCost': {'amount': '0.52', 'currency': 'usd'},
            'gaseousFuelConsumedGrams': 0,
            'gaseousFuelCost': {'amount': '0.00', 'currency': 'usd'},
            'operator': {'id': 1000006},
            'ptoState': 'inactive',
            'startTime': '2026-05-14T21:46:21.286Z',
            'latitude': 30.2,
            'longitude': -90.2,
        },
    ],
    'pagination': {'endCursor': '', 'hasNextPage': False},
}

_FIRST_EVENT: dict[str, Any] = _IDLING_FIXTURE['data'][0]
_SECOND_EVENT: dict[str, Any] = _IDLING_FIXTURE['data'][1]
_THIRD_EVENT: dict[str, Any] = _IDLING_FIXTURE['data'][2]
_EXPECTED_EVENT_COUNT: int = len(_IDLING_FIXTURE['data'])


class TestIdlingEventsResponseParsing:
    """Parse-and-shape assertions against the anonymized fixture."""

    def test_full_fixture_parses_without_error(self) -> None:
        """Should parse the full fixture without raising ValidationError."""

        IdlingEventsResponse.model_validate(_IDLING_FIXTURE)

    def test_get_items_returns_all_events(self) -> None:
        """Should expose every event via get_items()."""

        parsed = IdlingEventsResponse.model_validate(_IDLING_FIXTURE)

        items = parsed.get_items()

        assert len(items) == _EXPECTED_EVENT_COUNT
        assert all(isinstance(item, IdlingEvent) for item in items)

    def test_pagination_metadata_parsed_for_last_page(self) -> None:
        """Should parse the empty-cursor / hasNextPage=False terminal page."""

        parsed = IdlingEventsResponse.model_validate(_IDLING_FIXTURE)

        assert parsed.pagination is not None
        assert (
            parsed.pagination.end_cursor == _IDLING_FIXTURE['pagination']['endCursor']
        )
        assert (
            parsed.pagination.has_next_page
            == _IDLING_FIXTURE['pagination']['hasNextPage']
        )


class TestIdlingEventFirstEventFields:
    """Field-level assertions on the first idling event."""

    def test_scalar_fields(self) -> None:
        """Should parse scalar event fields with snake_case names."""

        first = IdlingEventsResponse.model_validate(_IDLING_FIXTURE).get_items()[0]

        assert first.event_uuid == _FIRST_EVENT['eventUuid']
        assert first.duration_milliseconds == _FIRST_EVENT['durationMilliseconds']
        assert first.pto_state == _FIRST_EVENT['ptoState']
        assert (
            first.fuel_consumed_milliliters == _FIRST_EVENT['fuelConsumedMilliliters']
        )
        assert (
            first.gaseous_fuel_consumed_grams
            == _FIRST_EVENT['gaseousFuelConsumedGrams']
        )
        assert first.latitude == _FIRST_EVENT['latitude']
        assert first.longitude == _FIRST_EVENT['longitude']

    def test_start_time_is_tz_aware(self) -> None:
        """Should parse startTime as a tz-aware datetime."""

        first = IdlingEventsResponse.model_validate(_IDLING_FIXTURE).get_items()[0]

        assert isinstance(first.start_time, datetime)
        assert first.start_time.tzinfo is not None

    def test_asset_id_coerced_to_str(self) -> None:
        """Should coerce the JSON integer asset.id to a Python str."""

        first = IdlingEventsResponse.model_validate(_IDLING_FIXTURE).get_items()[0]

        assert isinstance(first.asset, IdlingAsset)
        assert isinstance(first.asset.asset_id, str)
        assert first.asset.asset_id == str(_FIRST_EVENT['asset']['id'])

    def test_operator_id_coerced_to_str(self) -> None:
        """Should coerce the JSON integer operator.id to a Python str."""

        first = IdlingEventsResponse.model_validate(_IDLING_FIXTURE).get_items()[0]

        assert isinstance(first.operator, IdlingOperator)
        assert isinstance(first.operator.operator_id, str)
        assert first.operator.operator_id == str(_FIRST_EVENT['operator']['id'])

    def test_fuel_cost_amount_coerced_to_float(self) -> None:
        """Should coerce the JSON string fuelCost.amount to a Python float."""

        first = IdlingEventsResponse.model_validate(_IDLING_FIXTURE).get_items()[0]

        assert isinstance(first.fuel_cost, IdlingFuelCost)
        assert isinstance(first.fuel_cost.amount, float)
        assert first.fuel_cost.amount == float(_FIRST_EVENT['fuelCost']['amount'])
        assert first.fuel_cost.currency == _FIRST_EVENT['fuelCost']['currency']

    def test_gaseous_fuel_cost_amount_coerced_to_float(self) -> None:
        """Should coerce gaseousFuelCost.amount the same way as fuelCost."""

        first = IdlingEventsResponse.model_validate(_IDLING_FIXTURE).get_items()[0]

        assert isinstance(first.gaseous_fuel_cost.amount, float)
        assert first.gaseous_fuel_cost.amount == float(
            _FIRST_EVENT['gaseousFuelCost']['amount'],
        )

    def test_air_temperature_present_when_reported(self) -> None:
        """Should parse airTemperatureMillicelsius as int when present."""

        first = IdlingEventsResponse.model_validate(_IDLING_FIXTURE).get_items()[0]

        assert (
            first.air_temperature_millicelsius
            == _FIRST_EVENT['airTemperatureMillicelsius']
        )


class TestIdlingEventOptionalFields:
    """Edge cases for optional fields on idling events."""

    def test_air_temperature_is_none_when_omitted(self) -> None:
        """Should leave air_temperature_millicelsius as None for event 2."""

        second = IdlingEventsResponse.model_validate(_IDLING_FIXTURE).get_items()[1]

        assert 'airTemperatureMillicelsius' not in _SECOND_EVENT
        assert second.air_temperature_millicelsius is None

    def test_air_temperature_present_on_third_event(self) -> None:
        """Should parse air_temperature_millicelsius on event 3 too."""

        third = IdlingEventsResponse.model_validate(_IDLING_FIXTURE).get_items()[2]

        assert (
            third.air_temperature_millicelsius
            == _THIRD_EVENT['airTemperatureMillicelsius']
        )


class TestIdlingFuelCostDirectInstantiation:
    """Direct validation tests for the IdlingFuelCost coercion logic."""

    def test_accepts_int_amount(self) -> None:
        """Should accept JSON integer amounts and coerce to float."""

        int_amount = 1
        cost = IdlingFuelCost.model_validate(
            {'amount': int_amount, 'currency': 'usd'},
        )

        assert isinstance(cost.amount, float)
        assert cost.amount == float(int_amount)

    def test_accepts_float_amount(self) -> None:
        """Should accept JSON float amounts as-is."""

        float_amount = 1.5
        cost = IdlingFuelCost.model_validate(
            {'amount': float_amount, 'currency': 'usd'},
        )

        assert isinstance(cost.amount, float)
        assert cost.amount == float_amount


class TestIdlingEventMissingOperator:
    """Samsara omits the ``operator`` key for unattributed idle events."""

    def test_idling_event_accepts_missing_operator_key(self) -> None:
        """The actual production failure shape: ``operator`` key absent entirely."""

        payload = dict(_FIRST_EVENT)
        # Confirm the fixture has the key today (regression guard if
        # someone changes the base fixture later), then drop it.
        assert 'operator' in payload
        del payload['operator']

        parsed = IdlingEvent.model_validate(payload)

        assert parsed.operator is None

    def test_idling_event_accepts_explicit_null_operator(self) -> None:
        """Explicit ``operator: null`` (key present, value null) also parses."""

        payload = dict(_FIRST_EVENT)
        payload['operator'] = None

        parsed = IdlingEvent.model_validate(payload)

        assert parsed.operator is None


class TestIdlingEventDriftAuditCoverage:
    """Strict-but-unused ``IdlingEvent`` fields accept null after the audit pass."""

    @pytest.mark.parametrize(
        ('field_name', 'expected_attr'),
        [
            ('fuelConsumedMilliliters', 'fuel_consumed_milliliters'),
            ('fuelCost', 'fuel_cost'),
            ('gaseousFuelConsumedGrams', 'gaseous_fuel_consumed_grams'),
            ('gaseousFuelCost', 'gaseous_fuel_cost'),
            ('ptoState', 'pto_state'),
            ('latitude', 'latitude'),
            ('longitude', 'longitude'),
        ],
    )
    def test_idling_event_accepts_null_in_unused_strict_fields(
        self, field_name: str, expected_attr: str
    ) -> None:
        """Each newly-nullable unused field accepts ``None``."""

        payload = dict(_FIRST_EVENT)
        payload[field_name] = None

        parsed = IdlingEvent.model_validate(payload)

        assert getattr(parsed, expected_attr) is None


class TestIdlingEventConsumedFieldsStrictness:
    """Fields the unifier consumes stay strict -- loud failure on drift is intentional."""

    @pytest.mark.parametrize(
        'field_name',
        ['startTime', 'durationMilliseconds', 'eventUuid', 'asset'],
    )
    def test_idling_event_rejects_null_in_consumed_fields(
        self, field_name: str
    ) -> None:
        """Setting a unifier-consumed ``IdlingEvent`` field to ``None`` raises."""

        payload = dict(_FIRST_EVENT)
        payload[field_name] = None

        with pytest.raises(ValidationError):
            IdlingEvent.model_validate(payload)
