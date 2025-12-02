# fleet_telemetry_hub/config/config_models.py
"""
Configuration management for Fleet Telemetry Integration.

This module provides Pydantic models and loading functionality for the master
configuration file that controls dual-source ELD data extraction from Motive
and Samsara providers.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

logger: logging.Logger = logging.getLogger(__name__)

# =============================================================================
# Type Aliases for Clarity
# =============================================================================

# Valid logging level names recognized by Python's logging module
LogLevelName = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

# Valid compression algorithms supported by pandas.to_parquet()
CompressionType = Literal['snappy', 'gzip', 'brotli', 'lz4', 'zstd'] | None


class ProviderConfig(BaseModel):
    """Configuration for a single telemetry provider (Motive or Samsara).

    Attributes:
        enabled: Whether this provider should be queried during extraction.
        base_url: Root API endpoint URL (no trailing slash).
        api_key: Actual API key.
        request_timeout: Tuple of [connect_timeout, read_timeout] in seconds.
        max_retries: Maximum number of retry attempts for failed requests.
        retry_backoff_factor: Exponential backoff multiplier (delay = factor * (2 ** attempt)).
        verify_ssl: SSL certificate verification. False disables (Zscaler bypass),
                   or path to CA bundle as string.
        rate_limit_requests_per_second: Maximum requests allowed per second.
    """

    enabled: bool
    base_url: str
    api_key: SecretStr = Field(description='Actual API Token (masked in logs)')
    request_timeout: tuple[int, int] = Field(
        description='[connect_timeout, read_timeout] in seconds'
    )
    max_retries: int = Field(ge=1, le=10)
    retry_backoff_factor: float = Field(gt=0.0)
    verify_ssl: bool | str = Field(
        description='False to disable SSL verification, or path to CA certificate bundle'
    )
    rate_limit_requests_per_second: int = Field(gt=0)

    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, base_url: str) -> str:
        """Ensure base_url does not have trailing slash.

        Args:
            base_url: The API base URL to validate.

        Returns:
            The validated URL without trailing slash.

        Raises:
            ValueError: If URL is empty or malformed.
        """
        if not base_url:
            raise ValueError('base_url cannot be empty')

        # Remove trailing slash for consistency
        return base_url.rstrip('/')

    @field_validator('request_timeout')
    @classmethod
    def validate_timeout(cls, timeout: tuple[int, int]) -> tuple[int, int]:
        """Ensure both timeout values are positive.

        Args:
            timeout: Tuple of [connect_timeout, read_timeout].

        Returns:
            The validated timeout tuple.

        Raises:
            ValueError: If either timeout value is non-positive.
        """
        connect_timeout: int
        read_timeout: int
        connect_timeout, read_timeout = timeout
        if connect_timeout <= 0 or read_timeout <= 0:
            raise ValueError('Both timeout values must be positive')
        return timeout

    @field_validator('verify_ssl')
    @classmethod
    def validate_ssl_config(cls, verify_ssl: bool | str) -> bool | str:
        """Validate SSL verification setting.

        Args:
            verify_ssl: Either False (disabled), True, or path to CA bundle.

        Returns:
            The validated SSL configuration.

        Raises:
            ValueError: If string path is provided but file does not exist.
        """
        # If it's a string path, verify the file exists
        if isinstance(verify_ssl, str):
            cert_path = Path(verify_ssl)
            if not cert_path.exists():
                raise ValueError(f'SSL certificate file not found: {verify_ssl}')
        return verify_ssl


class PipelineConfig(BaseModel):
    """Configuration for data extraction pipeline execution.

    Attributes:
        default_start_date: ISO-8601 date string for initial backfill (YYYY-MM-DD).
        lookback_days: Number of days to overlap on incremental runs (catches late data).
        request_delay_seconds: Artificial delay between API requests (rate limiting).
        use_truststore: Use the truststore library for windows environments behind ZScaler or similar.
    """

    default_start_date: str = Field(
        pattern=r'^\d{4}-\d{2}-\d{2}$',
        description='ISO-8601 date (YYYY-MM-DD) for initial backfill',
    )
    lookback_days: int = Field(ge=0, le=30)
    request_delay_seconds: float = Field(ge=0.0)
    use_truststore: bool = Field(
        default=False,
        description=(
            'If true, build an SSLContext via truststore.SSLContext and pass it '
            'to httpx as verify=. If false, use httpx default certificate handling.'
        ),
    )

    @field_validator('default_start_date')
    @classmethod
    def validate_start_date(cls, date_string: str) -> str:
        """Ensure default_start_date is a valid ISO-8601 date.

        Args:
            date_string: Date string in YYYY-MM-DD format.

        Returns:
            The validated date string.

        Raises:
            ValueError: If date string cannot be parsed.
        """
        try:
            datetime.fromisoformat(date_string)
        except ValueError as error:
            raise ValueError(
                f'default_start_date must be valid ISO-8601 date (YYYY-MM-DD): {error}'
            ) from error
        return date_string


class StorageConfig(BaseModel):
    """Configuration for Parquet file storage and deduplication.

    Attributes:
        parquet_path: Relative or absolute path to the output Parquet file.
        parquet_compression: Compression codec for Parquet writer.
    """

    parquet_path: str
    parquet_compression: CompressionType = 'snappy'

    @field_validator('parquet_path')
    @classmethod
    def validate_parquet_path(cls, path_string: str) -> str:
        """Ensure parquet_path has .parquet extension.

        Args:
            path_string: Path to parquet file.

        Returns:
            The validated path string.

        Raises:
            ValueError: If path does not end with .parquet extension.
        """
        if not path_string.endswith('.parquet'):
            raise ValueError('parquet_path must have .parquet extension')
        return path_string


class LoggingConfig(BaseModel):
    """Configuration for application logging output.

    Attributes:
        file_path: Path to log file (directories created automatically).
        console_level: Minimum log level for console output.
        file_level: Minimum log level for file output.
    """

    file_path: str
    console_level: LogLevelName = 'INFO'
    file_level: LogLevelName = 'DEBUG'

    @field_validator('file_path')
    @classmethod
    def validate_log_path(cls, path_string: str) -> str:
        """Ensure log file path has .log extension.

        Args:
            path_string: Path to log file.

        Returns:
            The validated path string.

        Raises:
            ValueError: If path does not end with .log extension.
        """
        if not path_string.endswith('.log'):
            raise ValueError('file_path must have .log extension')
        return path_string


class TelemetryConfig(BaseModel):
    """Root configuration model for Fleet Telemetry Integration.

    This model enforces strict validation and prevents extraneous fields in the
    configuration YAML. All provider-specific, pipeline, storage, and logging
    settings are nested under this root.

    Attributes:
        providers: Dictionary of provider names to their configurations.
        pipeline: Pipeline execution settings.
        storage: Parquet file storage configuration.
        logging: Application logging configuration.
    """

    model_config = ConfigDict(extra='forbid')  # Reject unknown fields in YAML

    providers: dict[str, ProviderConfig]
    pipeline: PipelineConfig
    storage: StorageConfig
    logging: LoggingConfig

    @model_validator(mode='after')
    def validate_at_least_one_provider_enabled(self) -> 'TelemetryConfig':
        """Ensure at least one provider is enabled.

        Returns:
            The validated config instance.

        Raises:
            ValueError: If all providers are disabled.
        """
        enabled_providers: list[str] = [
            name for name, config in self.providers.items() if config.enabled
        ]
        if not enabled_providers:
            raise ValueError('At least one provider must be enabled')

        logger.debug('Enabled providers: %r', ', '.join(enabled_providers))
        return self
