"""

Tests for fleet_telemetry_hub.client module.



Tests TelemetryClient HTTP operations, pagination, retries, and error handling.

"""
# pyright: reportPrivateUsage=false

from collections.abc import Mapping
from typing import Any
from unittest.mock import Mock, patch

import httpx
import pytest
from pandas import DataFrame
from pydantic import BaseModel

from fleet_telemetry_hub.client import (
    APIError,
    RateLimitError,
    TelemetryClient,
    TransientAPIError,
)
from fleet_telemetry_hub.models import (
    EndpointDefinition,
    HTTPMethod,
    PaginationState,
    ParsedResponse,
    ProviderCredentials,
    RequestSpec,
)

# Sample model for testing


class Vehicle(BaseModel):
    """Sample vehicle model."""

    id: int

    number: str

    vin: str | None = None


# Sample endpoint definition for testing


class MockEndpointDefinition(EndpointDefinition[Vehicle]):
    """Mock endpoint definition for testing."""

    endpoint_path: str = '/vehicles'

    http_method: HTTPMethod = HTTPMethod.GET

    description: str = 'Test endpoint'

    is_paginated: bool = True

    def build_resource_path(self, **params: Any) -> str:
        """Build resource path."""

        return self.endpoint_path

    def build_request_spec(
        self,
        credentials: ProviderCredentials,
        pagination_state: PaginationState | None = None,
        **params: Any,
    ) -> RequestSpec:
        """Build request spec."""

        query_params: dict[Any, Any] = {}

        if pagination_state and pagination_state.current_page:
            query_params['page'] = str(pagination_state.current_page)

        return RequestSpec(
            url=f'{credentials.base_url}{self.endpoint_path}',
            method=self.http_method,
            headers={'X-API-Key': credentials.api_key.get_secret_value()},
            query_params=query_params,
            body=None,
            timeout=credentials.timeout,
        )

    def parse_response(self, response_json: dict[str, Any]) -> ParsedResponse[Vehicle]:
        """Parse response."""

        items: list[Vehicle] = [
            Vehicle(**item) for item in response_json.get('data', [])
        ]

        has_more: bool = response_json.get('has_next_page', False)

        current_page: int = response_json.get('current_page', 1)

        if has_more:
            current_page += 1

        pagination = PaginationState(
            has_next_page=has_more,
            current_page=current_page + 1,
            total_items=len(items),
        )

        return ParsedResponse[Vehicle](
            items=items,
            pagination=pagination,
        )

    def get_initial_pagination_state(self) -> PaginationState: ...


@pytest.fixture
def mock_endpoint() -> MockEndpointDefinition:
    """Provide mock endpoint definition."""

    return MockEndpointDefinition()


class TestTelemetryClientInitialization:
    """Test TelemetryClient initialization."""

    def test_initialization_succeeds(
        self,
        provider_credentials: ProviderCredentials,
    ) -> None:
        """Should initialize client successfully."""

        client = TelemetryClient(provider_credentials)

        assert client is not None

        assert client._credentials == provider_credentials

    def test_initialization_with_custom_pool_settings(
        self,
        provider_credentials: ProviderCredentials,
    ) -> None:
        """Should accept custom connection pool settings."""

        client = TelemetryClient(
            credentials=provider_credentials,
            pool_connections=10,
            pool_maxsize=20,
        )

        assert client is not None

    def test_initialization_with_request_delay(
        self,
        provider_credentials: ProviderCredentials,
    ) -> None:
        """Should accept custom request delay."""

        client = TelemetryClient(
            credentials=provider_credentials,
            request_delay_seconds=0.5,
        )

        assert client._request_delay_seconds == 0.5  # noqa: PLR2004


class TestTelemetryClientContextManager:
    """Test TelemetryClient context manager."""

    def test_context_manager_opens_and_closes(
        self,
        provider_credentials: ProviderCredentials,
    ) -> None:
        """Should work as context manager."""

        with TelemetryClient(provider_credentials) as client:
            assert client is not None

        # Client should be closed after context

        # (httpx.Client.close() should have been called)

    def test_manual_close(
        self,
        provider_credentials: ProviderCredentials,
    ) -> None:
        """Should allow manual close."""

        client = TelemetryClient(provider_credentials)

        # Should not raise

        client.close()


class TestTelemetryClientFetch:
    """Test TelemetryClient.fetch() method."""

    def test_fetch_successful_response(
        self,
        provider_credentials: ProviderCredentials,
        mock_endpoint: MockEndpointDefinition,
    ) -> None:
        """Should fetch and parse successful response."""

        mock_response = Mock(spec=httpx.Response)

        mock_response.status_code = 200

        mock_response.is_success = True

        mock_response.json.return_value = {
            'data': [
                {'id': 1, 'number': 'V001', 'vin': 'ABC123'},
                {'id': 2, 'number': 'V002', 'vin': 'DEF456'},
            ],
            'has_more': False,
        }

        mock_response.headers = {}

        with (
            TelemetryClient(provider_credentials) as client,
            patch.object(client._http_client, 'request', return_value=mock_response),
        ):
            response: ParsedResponse[Vehicle] = client.fetch(mock_endpoint)

            assert isinstance(response, ParsedResponse)

            assert len(response.items) == 2  # noqa: PLR2004

            assert response.items[0].number == 'V001'

            assert response.has_more is False

    def test_fetch_with_pagination_state(
        self,
        provider_credentials: ProviderCredentials,
        mock_endpoint: MockEndpointDefinition,
    ) -> None:
        """Should include pagination state in request."""

        pagination_state = PaginationState(
            has_next_page=True, current_page=2, total_items=10
        )

        mock_response = Mock(spec=httpx.Response)

        mock_response.status_code = 200

        mock_response.is_success = True

        mock_response.json.return_value = {
            'data': [{'id': 3, 'number': 'V003'}],
            'has_more': False,
        }

        mock_response.headers = {}

        with (
            TelemetryClient(provider_credentials) as client,
            patch.object(
                client._http_client,
                'request',
                return_value=mock_response,
            ) as mock_request,
        ):
            client.fetch(mock_endpoint, pagination_state=pagination_state)

            # Should include page parameter in request

            call_kwargs: Mapping[str, Any] | Any = mock_request.call_args.kwargs

            assert 'params' in call_kwargs

            assert call_kwargs['params'].get('page') == '2'

    def test_fetch_raises_on_client_error(
        self,
        provider_credentials: ProviderCredentials,
        mock_endpoint: MockEndpointDefinition,
    ) -> None:
        """Should raise APIError on client errors (4xx)."""

        mock_response = Mock(spec=httpx.Response)

        mock_response.status_code = 400

        mock_response.is_success = False

        mock_response.text = 'Bad Request'

        with (
            TelemetryClient(provider_credentials) as client,
            patch.object(client._http_client, 'request', return_value=mock_response),
        ):
            with pytest.raises(APIError) as exc_info:
                client.fetch(mock_endpoint)

            assert exc_info.value.status_code == 400  # noqa: PLR2004

    def test_fetch_raises_on_rate_limit(
        self,
        provider_credentials: ProviderCredentials,
        mock_endpoint: MockEndpointDefinition,
    ) -> None:
        """Should raise RateLimitError on HTTP 429."""

        mock_response = Mock(spec=httpx.Response)

        mock_response.status_code = 429

        mock_response.headers = {
            'Retry-After': '5',
            'X-RateLimit-Remaining': '0',
        }

        with (
            TelemetryClient(provider_credentials) as client,
            patch.object(client._http_client, 'request', return_value=mock_response),
            pytest.raises(RateLimitError),
        ):
            # Should retry and eventually raise
            client.fetch(mock_endpoint)


class TestTelemetryClientFetchAll:
    """Test TelemetryClient.fetch_all() method."""

    def test_fetch_all_single_page(
        self,
        provider_credentials: ProviderCredentials,
        mock_endpoint: MockEndpointDefinition,
    ) -> None:
        """Should fetch all items from single page."""

        mock_response = Mock(spec=httpx.Response)

        mock_response.status_code = 200

        mock_response.is_success = True

        mock_response.json.return_value = {
            'data': [
                {'id': 1, 'number': 'V001'},
                {'id': 2, 'number': 'V002'},
            ],
            'has_more': False,
        }

        mock_response.headers = {}

        with (
            TelemetryClient(provider_credentials) as client,
            patch.object(client._http_client, 'request', return_value=mock_response),
        ):
            items: list[Vehicle] = list(client.fetch_all(mock_endpoint))

            assert len(items) == 2  # noqa: PLR2004

            assert all(isinstance(item, Vehicle) for item in items)

    def test_fetch_all_multiple_pages(
        self,
        provider_credentials: ProviderCredentials,
        mock_endpoint: MockEndpointDefinition,
    ) -> None:
        """Should paginate through multiple pages."""

        # Mock responses for multiple pages

        page1_response = Mock(spec=httpx.Response)

        page1_response.status_code = 200

        page1_response.is_success = True

        page1_response.json.return_value = {
            'data': [{'id': 1, 'number': 'V001'}],
            'has_more': True,
            'page': 1,
        }

        page1_response.headers = {}

        page2_response = Mock(spec=httpx.Response)

        page2_response.status_code = 200

        page2_response.is_success = True

        page2_response.json.return_value = {
            'data': [{'id': 2, 'number': 'V002'}],
            'has_more': False,
            'page': 2,
        }

        page2_response.headers = {}

        with (
            TelemetryClient(provider_credentials) as client,
            patch.object(
                client._http_client,
                'request',
                side_effect=[page1_response, page2_response],
            ) as mock_request,
        ):
            items: list[Vehicle] = list(client.fetch_all(mock_endpoint))

            # Should have fetched all items

            assert len(items) == 2  # noqa: PLR2004

            assert items[0].number == 'V001'

            assert items[1].number == 'V002'

            # Should have made 2 requests

            assert mock_request.call_count == 2  # noqa: PLR2004


class TestTelemetryClientToDataFrame:
    """Test TelemetryClient.to_dataframe() method."""

    def test_to_dataframe_converts_items(
        self,
        provider_credentials: ProviderCredentials,
        mock_endpoint: MockEndpointDefinition,
    ) -> None:
        """Should convert items to DataFrame."""

        mock_response = Mock(spec=httpx.Response)

        mock_response.status_code = 200

        mock_response.is_success = True

        mock_response.json.return_value = {
            'data': [
                {'id': 1, 'number': 'V001', 'vin': 'ABC123'},
                {'id': 2, 'number': 'V002', 'vin': 'DEF456'},
            ],
            'has_more': False,
        }

        mock_response.headers = {}

        with (
            TelemetryClient(provider_credentials) as client,
            patch.object(client._http_client, 'request', return_value=mock_response),
        ):
            df: DataFrame = client.to_dataframe(mock_endpoint)

            assert len(df) == 2  # noqa: PLR2004

            assert 'id' in df.columns

            assert 'number' in df.columns

            assert df.iloc[0]['number'] == 'V001'

    def test_to_dataframe_returns_empty_when_no_items(
        self,
        provider_credentials: ProviderCredentials,
        mock_endpoint: MockEndpointDefinition,
    ) -> None:
        """Should return empty DataFrame when no items found."""

        mock_response = Mock(spec=httpx.Response)

        mock_response.status_code = 200

        mock_response.is_success = True

        mock_response.json.return_value = {
            'data': [],
            'has_more': False,
        }

        mock_response.headers = {}

        with (
            TelemetryClient(provider_credentials) as client,
            patch.object(client._http_client, 'request', return_value=mock_response),
        ):
            df: DataFrame = client.to_dataframe(mock_endpoint)

            assert len(df) == 0


class TestTelemetryClientErrorHandling:
    """Test TelemetryClient error handling and retries."""

    def test_retries_on_transient_error(
        self,
        provider_credentials: ProviderCredentials,
        mock_endpoint: MockEndpointDefinition,
    ) -> None:
        """Should retry on transient errors."""

        # First call fails with server error, second succeeds

        error_response = Mock(spec=httpx.Response)

        error_response.status_code = 500

        error_response.is_success = False

        error_response.text = 'Internal Server Error'

        success_response = Mock(spec=httpx.Response)

        success_response.status_code = 200

        success_response.is_success = True

        success_response.json.return_value = {
            'data': [{'id': 1, 'number': 'V001'}],
            'has_more': False,
        }

        success_response.headers = {}

        with (
            TelemetryClient(provider_credentials) as client,
            patch.object(
                client._http_client,
                'request',
                side_effect=[error_response, success_response],
            ) as mock_request,
        ):
            response: ParsedResponse[Vehicle] = client.fetch(mock_endpoint)

            # Should succeed after retry

            assert response.item_count == 1

            # Should have made 2 requests (1 failure + 1 retry)

            assert mock_request.call_count == 2  # noqa: PLR2004

    def test_retries_on_timeout(
        self,
        provider_credentials: ProviderCredentials,
        mock_endpoint: MockEndpointDefinition,
    ) -> None:
        """Should retry on timeout errors."""

        success_response = Mock(spec=httpx.Response)

        success_response.status_code = 200

        success_response.is_success = True

        success_response.json.return_value = {
            'data': [{'id': 1, 'number': 'V001'}],
            'has_more': False,
        }

        success_response.headers = {}

        with (
            TelemetryClient(provider_credentials) as client,
            patch.object(
                client._http_client,
                'request',
                side_effect=[httpx.TimeoutException('timeout'), success_response],
            ) as mock_request,
        ):
            response: ParsedResponse[Vehicle] = client.fetch(mock_endpoint)

            # Should succeed after retry

            assert response.item_count == 1

            assert mock_request.call_count == 2  # noqa: PLR2004

    def test_raises_after_max_retries(
        self,
        provider_credentials: ProviderCredentials,
        mock_endpoint: MockEndpointDefinition,
    ) -> None:
        """Should raise after exhausting retries."""

        error_response = Mock(spec=httpx.Response)

        error_response.status_code = 500

        error_response.is_success = False

        error_response.text = 'Internal Server Error'

        with (
            TelemetryClient(provider_credentials) as client,
            patch.object(
                client._http_client,
                'request',
                return_value=error_response,
            ),
            pytest.raises(TransientAPIError),
        ):
            client.fetch(mock_endpoint)


class TestTelemetryClientSSLConfiguration:
    """Test TelemetryClient SSL/TLS configuration."""

    def test_ssl_verification_enabled_by_default(
        self,
        provider_credentials: ProviderCredentials,
    ) -> None:
        """Should enable SSL verification by default."""

        assert provider_credentials.verify_ssl is True

        with TelemetryClient(provider_credentials) as client:
            # Client should be initialized successfully

            assert client is not None

    def test_ssl_verification_can_be_disabled(self) -> None:
        """Should allow disabling SSL verification."""

        credentials = ProviderCredentials(
            base_url='https://api.example.com',
            api_key='test_key', # pyright: ignore[reportArgumentType]
            timeout=(10, 30),
            max_retries=3,
            retry_backoff_factor=2.0,
            verify_ssl=False,  # Disable SSL
            use_truststore=False,
        )

        with TelemetryClient(credentials) as client:
            assert client is not None
