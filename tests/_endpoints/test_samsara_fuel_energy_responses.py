"""Tests for the Samsara /fleet/reports/vehicles/fuel-energy response models.

Covers parsing of the anonymized fixture below through
``FuelEnergyResponse`` and its nested models, including the
double-wrapped ``data.vehicleReports`` structure unique to this endpoint
and the all-zeros edge case where a vehicle barely moved.

All identifiers in test data are synthetic. No real VINs, vehicle IDs,
or fleet numbers from any production fleet appear in this file.
"""

from typing import Any

from fleet_telemetry_hub.models.samsara_responses import (
    EstFuelEnergyCost,
    FuelEnergyData,
    FuelEnergyResponse,
    FuelEnergyVehicle,
    FuelEnergyVehicleReport,
    SamsaraExternalIds,
)

_FUEL_ENERGY_FIXTURE: dict[str, Any] = {
    'data': {
        'vehicleReports': [
            {
                'vehicle': {
                    'energyType': 'fuel',
                    'id': '999999900000001',
                    'name': 'TEST-001 (Tractor)',
                    'externalIds': {
                        'samsara.serial': 'TESTSERIAL01',
                        'samsara.vin': 'TESTVIN0000000001',
                    },
                },
                'efficiencyMpge': 5.470090139159123,
                'energyUsedKwh': 0,
                'fuelConsumedMl': 208993,
                'distanceTraveledMeters': 486029,
                'estCarbonEmissionsKg': 563.654121,
                'estFuelEnergyCost': {
                    'amount': 311.3849550217656,
                    'currencyCode': 'USD',
                },
                'engineRunTimeDurationMs': 72129151,
                'engineIdleTimeDurationMs': 40774691,
            },
            {
                'vehicle': {
                    'energyType': 'fuel',
                    'id': '999999900000002',
                    'name': 'TEST-002 (Sleeper Cab w/ Wet Kit)',
                    'externalIds': {
                        'samsara.serial': 'TESTSERIAL02',
                        'samsara.vin': 'TESTVIN0000000002',
                    },
                },
                'efficiencyMpge': 5.171554543073868,
                'energyUsedKwh': 0,
                'fuelConsumedMl': 506285,
                'distanceTraveledMeters': 1113146,
                'estCarbonEmissionsKg': 1365.4506450000001,
                'estFuelEnergyCost': {
                    'amount': 754.3292366605518,
                    'currencyCode': 'USD',
                },
                'engineRunTimeDurationMs': 85993233,
                'engineIdleTimeDurationMs': 23455450,
            },
            {
                'vehicle': {
                    'energyType': 'fuel',
                    'id': '999999900000003',
                    'name': 'TEST-003 (Tractor w/ Wet Kit)',
                    'externalIds': {
                        'samsara.serial': 'TESTSERIAL03',
                        'samsara.vin': 'TESTVIN0000000003',
                    },
                },
                'efficiencyMpge': 0,
                'energyUsedKwh': 0,
                'fuelConsumedMl': 1500,
                'distanceTraveledMeters': 60,
                'estCarbonEmissionsKg': 4.0455,
                'estFuelEnergyCost': {
                    'amount': 2.23489512,
                    'currencyCode': 'USD',
                },
                'engineRunTimeDurationMs': 1278261,
                'engineIdleTimeDurationMs': 1105330,
            },
            {
                'vehicle': {
                    'energyType': 'fuel',
                    'id': '999999900000004',
                    'name': 'TEST-004 (Tractor w/ Wet Kit)',
                    'externalIds': {
                        'samsara.serial': 'TESTSERIAL04',
                        'samsara.vin': 'TESTVIN0000000004',
                    },
                },
                'efficiencyMpge': 0,
                'energyUsedKwh': 0,
                'fuelConsumedMl': 0,
                'distanceTraveledMeters': 152,
                'estCarbonEmissionsKg': 0,
                'estFuelEnergyCost': {
                    'amount': 0,
                    'currencyCode': 'USD',
                },
                'engineRunTimeDurationMs': 0,
                'engineIdleTimeDurationMs': 0,
            },
        ],
    },
    'pagination': {
        'endCursor': '00000000-0000-0000-0000-000000000001',
        'hasNextPage': True,
    },
}

_FIRST_REPORT: dict[str, Any] = _FUEL_ENERGY_FIXTURE['data']['vehicleReports'][0]
_THIRD_REPORT: dict[str, Any] = _FUEL_ENERGY_FIXTURE['data']['vehicleReports'][2]
_FOURTH_REPORT: dict[str, Any] = _FUEL_ENERGY_FIXTURE['data']['vehicleReports'][3]
_EXPECTED_REPORT_COUNT: int = len(_FUEL_ENERGY_FIXTURE['data']['vehicleReports'])


class TestFuelEnergyResponseParsing:
    """Parse-and-shape assertions against the anonymized fixture."""

    def test_full_fixture_parses_without_error(self) -> None:
        """Should parse the full fixture without raising ValidationError."""

        FuelEnergyResponse.model_validate(_FUEL_ENERGY_FIXTURE)

    def test_data_is_container_holding_vehicle_reports(self) -> None:
        """Should expose data as a FuelEnergyData container with reports."""

        parsed = FuelEnergyResponse.model_validate(_FUEL_ENERGY_FIXTURE)

        assert isinstance(parsed.data, FuelEnergyData)
        assert len(parsed.data.vehicle_reports) == _EXPECTED_REPORT_COUNT

    def test_get_items_returns_flat_list_of_reports(self) -> None:
        """Should flatten through the data wrapper to a list of reports."""

        parsed = FuelEnergyResponse.model_validate(_FUEL_ENERGY_FIXTURE)

        items = parsed.get_items()

        assert len(items) == _EXPECTED_REPORT_COUNT
        assert all(isinstance(item, FuelEnergyVehicleReport) for item in items)

    def test_pagination_metadata_parsed(self) -> None:
        """Should parse endCursor and hasNextPage from the pagination block."""

        parsed = FuelEnergyResponse.model_validate(_FUEL_ENERGY_FIXTURE)

        assert parsed.pagination is not None
        assert (
            parsed.pagination.end_cursor
            == _FUEL_ENERGY_FIXTURE['pagination']['endCursor']
        )
        assert (
            parsed.pagination.has_next_page
            == _FUEL_ENERGY_FIXTURE['pagination']['hasNextPage']
        )


class TestFuelEnergyFirstReportFields:
    """Field-level assertions on the first vehicle report."""

    def test_vehicle_scalar_fields(self) -> None:
        """Should parse vehicle scalar fields with snake_case names."""

        first = FuelEnergyResponse.model_validate(_FUEL_ENERGY_FIXTURE).get_items()[0]

        assert isinstance(first.vehicle, FuelEnergyVehicle)
        assert first.vehicle.vehicle_id == _FIRST_REPORT['vehicle']['id']
        assert first.vehicle.name == _FIRST_REPORT['vehicle']['name']
        assert first.vehicle.energy_type == _FIRST_REPORT['vehicle']['energyType']

    def test_vehicle_external_ids_parsed_from_dotted_keys(self) -> None:
        """Should parse externalIds via dotted aliases into SamsaraExternalIds."""

        first = FuelEnergyResponse.model_validate(_FUEL_ENERGY_FIXTURE).get_items()[0]

        assert isinstance(first.vehicle.external_ids, SamsaraExternalIds)
        assert (
            first.vehicle.external_ids.samsara_vin
            == _FIRST_REPORT['vehicle']['externalIds']['samsara.vin']
        )
        assert (
            first.vehicle.external_ids.samsara_serial
            == _FIRST_REPORT['vehicle']['externalIds']['samsara.serial']
        )

    def test_report_numeric_fields(self) -> None:
        """Should parse numeric report fields with the documented types."""

        first = FuelEnergyResponse.model_validate(_FUEL_ENERGY_FIXTURE).get_items()[0]

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
        """Should parse the nested cost block with uppercase currencyCode."""

        first = FuelEnergyResponse.model_validate(_FUEL_ENERGY_FIXTURE).get_items()[0]

        assert isinstance(first.est_fuel_energy_cost, EstFuelEnergyCost)
        assert (
            first.est_fuel_energy_cost.amount
            == _FIRST_REPORT['estFuelEnergyCost']['amount']
        )
        assert (
            first.est_fuel_energy_cost.currency_code
            == _FIRST_REPORT['estFuelEnergyCost']['currencyCode']
        )


class TestFuelEnergyEdgeCases:
    """Edge-case assertions on the low- and zero-activity records."""

    def test_zero_efficiency_low_activity_record(self) -> None:
        """Should parse the low-activity record (efficiencyMpge=0) cleanly."""

        items = FuelEnergyResponse.model_validate(_FUEL_ENERGY_FIXTURE).get_items()

        third = items[2]

        assert third.efficiency_mpge == 0.0
        assert third.fuel_consumed_ml == _THIRD_REPORT['fuelConsumedMl']
        assert third.distance_traveled_meters == _THIRD_REPORT['distanceTraveledMeters']

    def test_all_zeros_record_parses(self) -> None:
        """Should parse the all-zeros record without raising."""

        items = FuelEnergyResponse.model_validate(_FUEL_ENERGY_FIXTURE).get_items()

        fourth = items[3]

        assert fourth.efficiency_mpge == 0.0
        assert fourth.engine_run_time_duration_ms == 0
        assert fourth.fuel_consumed_ml == 0
        assert fourth.est_fuel_energy_cost.amount == 0.0
        assert fourth.est_carbon_emissions_kg == 0.0
        assert (
            fourth.distance_traveled_meters == _FOURTH_REPORT['distanceTraveledMeters']
        )


class TestFuelEnergyDataDirectInstantiation:
    """Direct instantiation tests for the intermediate FuelEnergyData wrapper."""

    def test_data_wrapper_can_be_built_from_alias(self) -> None:
        """Should accept the vehicleReports alias when validating directly."""

        wrapper = FuelEnergyData.model_validate(
            {'vehicleReports': [_FIRST_REPORT]},
        )

        assert len(wrapper.vehicle_reports) == 1
        assert isinstance(wrapper.vehicle_reports[0], FuelEnergyVehicleReport)
