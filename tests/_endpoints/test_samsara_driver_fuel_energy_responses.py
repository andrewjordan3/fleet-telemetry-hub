"""Tests for the Samsara /fleet/reports/drivers/fuel-energy response models.

Covers parsing of the anonymized fixture below through
``DriverFuelEnergyResponse`` and its nested models, including the
double-wrapped ``data.driverReports`` structure that mirrors the
sibling vehicle fuel-energy endpoint, and the all-zeros edge case
where a driver did not drive in the requested window.

All identifiers in test data are synthetic. No real driver IDs or
names from any production fleet appear in this file.
"""

from typing import Any

from fleet_telemetry_hub.models.samsara_responses import (
    DriverFuelEnergyData,
    DriverFuelEnergyReport,
    DriverFuelEnergyResponse,
    EstFuelEnergyCost,
    FuelEnergyDriver,
)

_DRIVER_FUEL_ENERGY_FIXTURE: dict[str, Any] = {
    'data': {
        'driverReports': [
            {
                'driver': {'id': '1000001', 'name': 'Sam Snowflake'},
                'efficiencyMpge': 7.0101372132901485,
                'energyUsedKwh': 0,
                'fuelConsumedMl': 34497,
                'distanceTraveledMeters': 102812,
                'estCarbonEmissionsKg': 93.038409,
                'estFuelEnergyCost': {
                    'amount': 51.39811644251562,
                    'currencyCode': 'USD',
                },
                'engineRunTimeDurationMs': 11899489,
                'engineIdleTimeDurationMs': 677738,
            },
            {
                'driver': {'id': '1000002', 'name': 'Suzy Snowflake'},
                'efficiencyMpge': 6.7025665323334085,
                'energyUsedKwh': 0,
                'fuelConsumedMl': 120988,
                'distanceTraveledMeters': 344762,
                'estCarbonEmissionsKg': 326.3046360000001,
                'estFuelEnergyCost': {
                    'amount': 180.26365687700076,
                    'currencyCode': 'USD',
                },
                'engineRunTimeDurationMs': 34554638,
                'engineIdleTimeDurationMs': 8482403,
            },
            {
                'driver': {'id': '1000003', 'name': 'Sammy Snowflake'},
                'efficiencyMpge': 0,
                'energyUsedKwh': 0,
                'fuelConsumedMl': 1998,
                'distanceTraveledMeters': 224,
                'estCarbonEmissionsKg': 5.388605999999999,
                'estFuelEnergyCost': {
                    'amount': 2.9768802598530275,
                    'currencyCode': 'USD',
                },
                'engineRunTimeDurationMs': 1734387,
                'engineIdleTimeDurationMs': 1302013,
            },
            {
                'driver': {'id': '1000004', 'name': 'Sandra Snowflake'},
                'efficiencyMpge': 0,
                'energyUsedKwh': 0,
                'fuelConsumedMl': 0,
                'distanceTraveledMeters': 0,
                'estCarbonEmissionsKg': 0,
                'estFuelEnergyCost': {'amount': 0, 'currencyCode': 'USD'},
                'engineRunTimeDurationMs': 0,
                'engineIdleTimeDurationMs': 0,
            },
        ],
    },
    'pagination': {
        'endCursor': '00000000-0000-0000-0000-000000000003',
        'hasNextPage': True,
    },
}

_FIRST_REPORT: dict[str, Any] = _DRIVER_FUEL_ENERGY_FIXTURE['data']['driverReports'][0]
_THIRD_REPORT: dict[str, Any] = _DRIVER_FUEL_ENERGY_FIXTURE['data']['driverReports'][2]
_FOURTH_REPORT: dict[str, Any] = _DRIVER_FUEL_ENERGY_FIXTURE['data']['driverReports'][3]
_EXPECTED_REPORT_COUNT: int = len(
    _DRIVER_FUEL_ENERGY_FIXTURE['data']['driverReports'],
)


class TestDriverFuelEnergyResponseParsing:
    """Parse-and-shape assertions against the anonymized fixture."""

    def test_full_fixture_parses_without_error(self) -> None:
        """Should parse the full fixture without raising ValidationError."""

        DriverFuelEnergyResponse.model_validate(_DRIVER_FUEL_ENERGY_FIXTURE)

    def test_data_is_container_holding_driver_reports(self) -> None:
        """Should expose data as a DriverFuelEnergyData container with reports."""

        parsed = DriverFuelEnergyResponse.model_validate(_DRIVER_FUEL_ENERGY_FIXTURE)

        assert isinstance(parsed.data, DriverFuelEnergyData)
        assert len(parsed.data.driver_reports) == _EXPECTED_REPORT_COUNT

    def test_get_items_returns_flat_list_of_reports(self) -> None:
        """Should flatten through the data wrapper to a list of reports."""

        parsed = DriverFuelEnergyResponse.model_validate(_DRIVER_FUEL_ENERGY_FIXTURE)

        items = parsed.get_items()

        assert len(items) == _EXPECTED_REPORT_COUNT
        assert all(isinstance(item, DriverFuelEnergyReport) for item in items)

    def test_pagination_metadata_parsed(self) -> None:
        """Should parse endCursor and hasNextPage from the pagination block."""

        parsed = DriverFuelEnergyResponse.model_validate(_DRIVER_FUEL_ENERGY_FIXTURE)

        assert parsed.pagination is not None
        assert (
            parsed.pagination.end_cursor
            == _DRIVER_FUEL_ENERGY_FIXTURE['pagination']['endCursor']
        )
        assert (
            parsed.pagination.has_next_page
            == _DRIVER_FUEL_ENERGY_FIXTURE['pagination']['hasNextPage']
        )


class TestDriverFuelEnergyFirstReportFields:
    """Field-level assertions on the first driver report."""

    def test_driver_scalar_fields(self) -> None:
        """Should parse driver scalar fields with snake_case names."""

        first = DriverFuelEnergyResponse.model_validate(
            _DRIVER_FUEL_ENERGY_FIXTURE,
        ).get_items()[0]

        assert isinstance(first.driver, FuelEnergyDriver)
        assert first.driver.driver_id == _FIRST_REPORT['driver']['id']
        assert first.driver.name == _FIRST_REPORT['driver']['name']

    def test_report_numeric_fields(self) -> None:
        """Should parse numeric report fields with the documented types."""

        first = DriverFuelEnergyResponse.model_validate(
            _DRIVER_FUEL_ENERGY_FIXTURE,
        ).get_items()[0]

        assert first.efficiency_mpge == _FIRST_REPORT['efficiencyMpge']
        assert first.fuel_consumed_ml == _FIRST_REPORT['fuelConsumedMl']
        assert first.distance_traveled_meters == _FIRST_REPORT['distanceTraveledMeters']
        assert (
            first.engine_run_time_duration_ms
            == _FIRST_REPORT['engineRunTimeDurationMs']
        )
        assert (
            first.engine_idle_time_duration_ms
            == _FIRST_REPORT['engineIdleTimeDurationMs']
        )

    def test_report_est_fuel_energy_cost(self) -> None:
        """Should parse the nested cost block via the reused EstFuelEnergyCost."""

        first = DriverFuelEnergyResponse.model_validate(
            _DRIVER_FUEL_ENERGY_FIXTURE,
        ).get_items()[0]

        assert isinstance(first.est_fuel_energy_cost, EstFuelEnergyCost)
        assert (
            first.est_fuel_energy_cost.amount
            == _FIRST_REPORT['estFuelEnergyCost']['amount']
        )
        assert (
            first.est_fuel_energy_cost.currency_code
            == _FIRST_REPORT['estFuelEnergyCost']['currencyCode']
        )


class TestDriverFuelEnergyEdgeCases:
    """Edge-case assertions on the low- and zero-activity records."""

    def test_zero_efficiency_low_activity_record(self) -> None:
        """Should parse the low-activity record (efficiencyMpge=0) cleanly."""

        items = DriverFuelEnergyResponse.model_validate(
            _DRIVER_FUEL_ENERGY_FIXTURE,
        ).get_items()

        third = items[2]

        assert third.efficiency_mpge == 0.0
        assert third.fuel_consumed_ml == _THIRD_REPORT['fuelConsumedMl']
        assert third.distance_traveled_meters == _THIRD_REPORT['distanceTraveledMeters']

    def test_all_zeros_record_parses(self) -> None:
        """Should parse the all-zeros (no-drive) record without raising."""

        items = DriverFuelEnergyResponse.model_validate(
            _DRIVER_FUEL_ENERGY_FIXTURE,
        ).get_items()

        fourth = items[3]

        assert fourth.driver.name == _FOURTH_REPORT['driver']['name']
        assert fourth.efficiency_mpge == 0.0
        assert fourth.fuel_consumed_ml == 0
        assert fourth.distance_traveled_meters == 0
        assert fourth.est_fuel_energy_cost.amount == 0.0
        assert fourth.est_carbon_emissions_kg == 0.0
        assert fourth.engine_run_time_duration_ms == 0
        assert fourth.engine_idle_time_duration_ms == 0


class TestDriverFuelEnergyDataDirectInstantiation:
    """Direct instantiation tests for the intermediate DriverFuelEnergyData wrapper."""

    def test_data_wrapper_can_be_built_from_alias(self) -> None:
        """Should accept the driverReports alias when validating directly."""

        wrapper = DriverFuelEnergyData.model_validate(
            {'driverReports': [_FIRST_REPORT]},
        )

        assert len(wrapper.driver_reports) == 1
        assert isinstance(wrapper.driver_reports[0], DriverFuelEnergyReport)
