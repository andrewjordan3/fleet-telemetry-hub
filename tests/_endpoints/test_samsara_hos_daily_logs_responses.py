"""Tests for the Samsara /fleet/hos/daily-logs response models.

Covers parsing of the anonymized fixture below through
``HosDailyLogsResponse`` and its nested models, including the
default-list behavior for absent ``vehicles`` / ``trailerNames`` keys
and the optional ``certifiedAtTime`` field.

All identifiers in test data are synthetic. No real VINs, driver IDs,
vehicle IDs, carrier names, or addresses from any production fleet
appear in this file.
"""

from datetime import datetime
from typing import Any

from fleet_telemetry_hub.models.samsara_responses import (
    EldSettings,
    HosDailyLog,
    HosDailyLogsResponse,
    HosDistanceTraveled,
    HosDriver,
    HosDutyStatusDurations,
    HosLogMetaData,
    HosVehicleReference,
    SamsaraExternalIds,
)

_HOS_FIXTURE: dict[str, Any] = {
    'data': [
        {
            'driver': {
                'timezone': 'America/Los_Angeles',
                'eldSettings': {
                    'rulesets': [
                        {
                            'break': 'Property (off-duty/sleeper)',
                            'cycle': 'USA 70 hour / 8 day',
                            'restart': '34-hour Restart',
                            'shift': 'US Interstate Property',
                        },
                    ],
                },
                'id': '1000001',
                'name': 'Sam Snowflake',
            },
            'startTime': '2026-05-14T07:00:00.000Z',
            'endTime': '2026-05-15T07:00:00.000Z',
            'logMetaData': {
                'shippingDocs': 'TEST-DOC-001',
                'vehicles': [
                    {
                        'id': '999999900000003',
                        'name': 'TEST-003 (Tractor)',
                        'externalIds': {
                            'samsara.serial': 'TESTSERIAL03',
                            'samsara.vin': 'TESTVIN0000000003',
                        },
                    },
                ],
                'isCertified': True,
                'certifiedAtTime': '2026-05-18T13:30:46.920Z',
                'adverseDrivingClaimed': False,
                'bigDayClaimed': False,
                'isUsShortHaulActive': False,
                'carrierName': 'Test Carrier Inc.',
                'carrierFormattedAddress': '100 Test St, Testville, TX 99999',
                'carrierUsDotNumber': 999999,
                'homeTerminalName': 'Test Terminal - 01',
                'homeTerminalFormattedAddress': '200 Test Ave, Testville, TX 99999',
            },
            'distanceTraveled': {'driveDistanceMeters': 102987},
            'dutyStatusDurations': {
                'activeDurationMs': 32680000,
                'onDutyDurationMs': 21186428,
                'driveDurationMs': 11493572,
                'offDutyDurationMs': 53719999,
                'sleeperBerthDurationMs': 0,
                'yardMoveDurationMs': 0,
                'personalConveyanceDurationMs': 0,
                'waitingTimeDurationMs': 0,
            },
            'pendingDutyStatusDurations': {
                'activeDurationMs': 32680000,
                'onDutyDurationMs': 21186428,
                'driveDurationMs': 11493572,
                'offDutyDurationMs': 53719999,
                'sleeperBerthDurationMs': 0,
                'yardMoveDurationMs': 0,
                'personalConveyanceDurationMs': 0,
                'waitingTimeDurationMs': 0,
            },
        },
        {
            'driver': {
                'timezone': 'America/Los_Angeles',
                'eldSettings': {
                    'rulesets': [
                        {
                            'break': 'Property (off-duty/sleeper)',
                            'cycle': 'USA 70 hour / 8 day',
                            'restart': '34-hour Restart',
                            'shift': 'US Interstate Property',
                        },
                    ],
                },
                'id': '1000002',
                'name': 'Suzy Snowflake',
            },
            'startTime': '2026-05-14T07:00:00.000Z',
            'endTime': '2026-05-15T07:00:00.000Z',
            'logMetaData': {
                'shippingDocs': '',
                'isCertified': False,
                'adverseDrivingClaimed': False,
                'bigDayClaimed': False,
                'isUsShortHaulActive': False,
                'carrierName': '',
                'carrierFormattedAddress': '',
                'carrierUsDotNumber': 0,
                'homeTerminalName': '',
                'homeTerminalFormattedAddress': '',
            },
            'distanceTraveled': {'driveDistanceMeters': 0},
            'dutyStatusDurations': {
                'activeDurationMs': 0,
                'onDutyDurationMs': 0,
                'driveDurationMs': 0,
                'offDutyDurationMs': 86399999,
                'sleeperBerthDurationMs': 0,
                'yardMoveDurationMs': 0,
                'personalConveyanceDurationMs': 0,
                'waitingTimeDurationMs': 0,
            },
            'pendingDutyStatusDurations': {
                'activeDurationMs': 0,
                'onDutyDurationMs': 0,
                'driveDurationMs': 0,
                'offDutyDurationMs': 86399999,
                'sleeperBerthDurationMs': 0,
                'yardMoveDurationMs': 0,
                'personalConveyanceDurationMs': 0,
                'waitingTimeDurationMs': 0,
            },
        },
        {
            'driver': {
                'timezone': 'America/Los_Angeles',
                'eldSettings': {
                    'rulesets': [
                        {
                            'break': 'Property (off-duty/sleeper)',
                            'cycle': 'USA 70 hour / 8 day',
                            'restart': '34-hour Restart',
                            'shift': 'US Interstate Property',
                        },
                    ],
                },
                'id': '1000003',
                'name': 'Sammy Snowflake',
            },
            'startTime': '2026-05-14T07:00:00.000Z',
            'endTime': '2026-05-15T07:00:00.000Z',
            'logMetaData': {
                'shippingDocs': 'TEST-DOC-002',
                'trailerNames': ['TEST-TRAILER-01'],
                'vehicles': [
                    {
                        'id': '999999900000004',
                        'name': 'TEST-004 (Tractor w/ Oil Cooled Wet Kit)',
                        'externalIds': {
                            'samsara.serial': 'TESTSERIAL04',
                            'samsara.vin': 'TESTVIN0000000004',
                        },
                    },
                ],
                'isCertified': True,
                'certifiedAtTime': '2026-05-15T12:27:00.573Z',
                'adverseDrivingClaimed': False,
                'bigDayClaimed': False,
                'isUsShortHaulActive': False,
                'carrierName': 'Test Carrier Inc.',
                'carrierFormattedAddress': '100 Test St, Testville, TX 99999',
                'carrierUsDotNumber': 999999,
                'homeTerminalName': 'Test Terminal - 02',
                'homeTerminalFormattedAddress': '300 Test Blvd, Testville, TX 99999',
            },
            'distanceTraveled': {'driveDistanceMeters': 226336},
            'dutyStatusDurations': {
                'activeDurationMs': 31751925,
                'onDutyDurationMs': 15083633,
                'driveDurationMs': 16668292,
                'offDutyDurationMs': 54648074,
                'sleeperBerthDurationMs': 0,
                'yardMoveDurationMs': 0,
                'personalConveyanceDurationMs': 0,
                'waitingTimeDurationMs': 0,
            },
            'pendingDutyStatusDurations': {
                'activeDurationMs': 31751925,
                'onDutyDurationMs': 15083633,
                'driveDurationMs': 16668292,
                'offDutyDurationMs': 54648074,
                'sleeperBerthDurationMs': 0,
                'yardMoveDurationMs': 0,
                'personalConveyanceDurationMs': 0,
                'waitingTimeDurationMs': 0,
            },
        },
    ],
    'pagination': {
        'endCursor': '00000000-0000-0000-0000-000000000002',
        'hasNextPage': True,
    },
}

_FIRST_LOG: dict[str, Any] = _HOS_FIXTURE['data'][0]
_SECOND_LOG: dict[str, Any] = _HOS_FIXTURE['data'][1]
_THIRD_LOG: dict[str, Any] = _HOS_FIXTURE['data'][2]
_EXPECTED_LOG_COUNT: int = len(_HOS_FIXTURE['data'])


class TestHosDailyLogsResponseParsing:
    """Parse-and-shape assertions against the anonymized fixture."""

    def test_full_fixture_parses_without_error(self) -> None:
        """Should parse the full fixture without raising ValidationError."""

        HosDailyLogsResponse.model_validate(_HOS_FIXTURE)

    def test_get_items_returns_all_daily_logs(self) -> None:
        """Should expose every log via get_items()."""

        parsed = HosDailyLogsResponse.model_validate(_HOS_FIXTURE)

        items = parsed.get_items()

        assert len(items) == _EXPECTED_LOG_COUNT
        assert all(isinstance(item, HosDailyLog) for item in items)

    def test_pagination_metadata_parsed(self) -> None:
        """Should parse endCursor and hasNextPage from the pagination block."""

        parsed = HosDailyLogsResponse.model_validate(_HOS_FIXTURE)

        assert parsed.pagination is not None
        assert parsed.pagination.end_cursor == _HOS_FIXTURE['pagination']['endCursor']
        assert (
            parsed.pagination.has_next_page == _HOS_FIXTURE['pagination']['hasNextPage']
        )


class TestHosDailyLogFirstLogFields:
    """Field-level assertions on Sam's (the first) daily log."""

    def test_driver_block_parsed(self) -> None:
        """Should parse driver scalars and the embedded EldSettings."""

        first = HosDailyLogsResponse.model_validate(_HOS_FIXTURE).get_items()[0]

        expected_driver = _FIRST_LOG['driver']
        assert isinstance(first.driver, HosDriver)
        assert first.driver.driver_id == expected_driver['id']
        assert first.driver.name == expected_driver['name']
        assert first.driver.timezone == expected_driver['timezone']
        assert isinstance(first.driver.eld_settings, EldSettings)
        assert len(first.driver.eld_settings.rulesets) == len(
            expected_driver['eldSettings']['rulesets']
        )
        assert (
            first.driver.eld_settings.rulesets[0].cycle
            == expected_driver['eldSettings']['rulesets'][0]['cycle']
        )

    def test_start_and_end_time_are_tz_aware(self) -> None:
        """Should parse startTime and endTime as tz-aware datetimes."""

        first = HosDailyLogsResponse.model_validate(_HOS_FIXTURE).get_items()[0]

        assert isinstance(first.start_time, datetime)
        assert first.start_time.tzinfo is not None
        assert isinstance(first.end_time, datetime)
        assert first.end_time.tzinfo is not None

    def test_log_meta_data_scalar_fields(self) -> None:
        """Should parse carrier and home terminal scalars verbatim."""

        first = HosDailyLogsResponse.model_validate(_HOS_FIXTURE).get_items()[0]

        meta = first.log_meta_data
        expected_meta = _FIRST_LOG['logMetaData']
        assert isinstance(meta, HosLogMetaData)
        assert meta.shipping_docs == expected_meta['shippingDocs']
        assert meta.is_certified is expected_meta['isCertified']
        assert meta.carrier_name == expected_meta['carrierName']
        assert meta.carrier_us_dot_number == expected_meta['carrierUsDotNumber']
        assert meta.home_terminal_name == expected_meta['homeTerminalName']

    def test_log_meta_data_vehicles_parsed(self) -> None:
        """Should parse the vehicles array including dotted-key externalIds."""

        first = HosDailyLogsResponse.model_validate(_HOS_FIXTURE).get_items()[0]

        vehicles = first.log_meta_data.vehicles
        expected_vehicles = _FIRST_LOG['logMetaData']['vehicles']
        assert len(vehicles) == len(expected_vehicles)
        vehicle = vehicles[0]
        expected_vehicle = expected_vehicles[0]
        assert isinstance(vehicle, HosVehicleReference)
        assert vehicle.vehicle_id == expected_vehicle['id']
        assert vehicle.name == expected_vehicle['name']
        assert isinstance(vehicle.external_ids, SamsaraExternalIds)
        assert (
            vehicle.external_ids.samsara_vin
            == expected_vehicle['externalIds']['samsara.vin']
        )

    def test_certified_at_time_is_tz_aware_when_present(self) -> None:
        """Should parse certifiedAtTime as a tz-aware datetime for Sam's log."""

        first = HosDailyLogsResponse.model_validate(_HOS_FIXTURE).get_items()[0]

        assert isinstance(first.log_meta_data.certified_at_time, datetime)
        assert first.log_meta_data.certified_at_time.tzinfo is not None

    def test_distance_traveled_parsed(self) -> None:
        """Should parse driveDistanceMeters as int."""

        first = HosDailyLogsResponse.model_validate(_HOS_FIXTURE).get_items()[0]

        assert isinstance(first.distance_traveled, HosDistanceTraveled)
        assert (
            first.distance_traveled.drive_distance_meters
            == _FIRST_LOG['distanceTraveled']['driveDistanceMeters']
        )

    def test_duty_status_durations_parsed(self) -> None:
        """Should parse duty status durations into HosDutyStatusDurations."""

        first = HosDailyLogsResponse.model_validate(_HOS_FIXTURE).get_items()[0]

        durations = first.duty_status_durations
        expected = _FIRST_LOG['dutyStatusDurations']
        assert isinstance(durations, HosDutyStatusDurations)
        assert durations.drive_duration_ms == expected['driveDurationMs']
        assert durations.on_duty_duration_ms == expected['onDutyDurationMs']
        assert durations.off_duty_duration_ms == expected['offDutyDurationMs']
        assert durations.sleeper_berth_duration_ms == expected['sleeperBerthDurationMs']


class TestHosDailyLogEmptyLog:
    """Edge-case assertions on Suzy's uncertified, equipment-less log."""

    def test_vehicles_defaults_to_empty_list(self) -> None:
        """Should default vehicles to [] when the key is absent."""

        second = HosDailyLogsResponse.model_validate(_HOS_FIXTURE).get_items()[1]

        assert 'vehicles' not in _SECOND_LOG['logMetaData']
        assert second.log_meta_data.vehicles == []

    def test_trailer_names_defaults_to_empty_list(self) -> None:
        """Should default trailer_names to [] when trailerNames is absent."""

        second = HosDailyLogsResponse.model_validate(_HOS_FIXTURE).get_items()[1]

        assert 'trailerNames' not in _SECOND_LOG['logMetaData']
        assert second.log_meta_data.trailer_names == []

    def test_certified_at_time_is_none_when_uncertified(self) -> None:
        """Should default certified_at_time to None for uncertified logs."""

        second = HosDailyLogsResponse.model_validate(_HOS_FIXTURE).get_items()[1]

        assert second.log_meta_data.is_certified is False
        assert second.log_meta_data.certified_at_time is None


class TestHosDailyLogWithTrailer:
    """Sanity assertions on Sammy's certified log with a trailer."""

    def test_trailer_names_populated(self) -> None:
        """Should preserve the trailerNames list verbatim."""

        third = HosDailyLogsResponse.model_validate(_HOS_FIXTURE).get_items()[2]

        assert (
            third.log_meta_data.trailer_names
            == _THIRD_LOG['logMetaData']['trailerNames']
        )

    def test_vehicle_external_ids_parsed(self) -> None:
        """Should parse the assigned vehicle's external IDs."""

        third = HosDailyLogsResponse.model_validate(_HOS_FIXTURE).get_items()[2]

        vehicle = third.log_meta_data.vehicles[0]
        expected_external = _THIRD_LOG['logMetaData']['vehicles'][0]['externalIds']
        assert vehicle.external_ids is not None
        assert vehicle.external_ids.samsara_vin == expected_external['samsara.vin']
        assert (
            vehicle.external_ids.samsara_serial == expected_external['samsara.serial']
        )
