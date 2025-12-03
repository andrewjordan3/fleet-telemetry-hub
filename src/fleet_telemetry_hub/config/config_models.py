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
from typing import Literal, Self, cast

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

    file_path: Path | None
    console_level: LogLevelName | int = 'INFO'
    file_level: LogLevelName | int | None = 'DEBUG'

    @field_validator('file_path', mode='before')
    @classmethod
    def validate_log_path(cls, path_string: str | None) -> Path | None:
        """
        Normalize incoming string to a Path, and ensure log file path has .log extension.

        Args:
            path_string: Path to log file.

        Returns:
            The validated path string.
        """
        if path_string is None:
            return None
        if not path_string.endswith('.log'):
            path_string = f'{path_string}.log'
        return Path(path_string)

    @field_validator('console_level', 'file_level')
    @classmethod
    def validate_log_level(
        cls, v: LogLevelName | int | None
    ) -> LogLevelName | int | None:
        """
        Validate that log levels are either valid string names or valid numeric values.

        Python's logging module uses these numeric values internally:
        - DEBUG: 10
        - INFO: 20
        - WARNING: 30
        - ERROR: 40
        - CRITICAL: 50

        We accept both formats and validate them here.
        """
        if v is None:
            return v

        # If it's a string, Pydantic's Literal type already validated it's one of the allowed names
        if isinstance(v, str):
            return v

        # Not None or str, so must be integer, validate it's a standard logging level
        valid_levels: set[int] = {10, 20, 30, 40, 50}
        if v not in valid_levels:
            raise ValueError(
                f'Numeric log level must be one of {valid_levels}, got {v}'
            )
        return v

    @model_validator(mode='after')
    def validate_file_logging_consistency(self) -> Self:
        """
        Ensure that if file_path is provided, file_level is also provided, and vice versa.

        This prevents misconfiguration where someone enables file logging but doesn't
        specify what level to log at (or vice versa).

        Model validators run after all field validators, so we can safely access
        multiple fields at once.
        """
        has_file_path: bool = self.file_path is not None
        has_file_level: bool = self.file_level is not None

        if has_file_path and not has_file_level:
            # Default to DEBUG if path is provided but level is missing
            self.file_level = 'DEBUG'
            logger.warning(
                'file_path provided without file_level. Defaulting to DEBUG for file logging.'
            )

        if has_file_level and not has_file_path:
            raise ValueError(
                'file_level is specified but file_path is missing. '
                'Both must be provided to enable file logging.'
            )

        return self

    def get_console_level_int(self) -> int:
        """
        Convert console_level to the integer value used by Python's logging module.

        Returns:
            The integer logging level (10, 20, 30, 40, or 50)
        """
        if isinstance(self.console_level, int):
            return self.console_level
        return cast(int, getattr(logging, self.console_level))

    def get_file_level_int(self) -> int | None:
        """
        Convert file_level to the integer value used by Python's logging module.

        Returns:
            The integer logging level, or None if file logging is disabled
        """
        if self.file_level is None:
            return None
        if isinstance(self.file_level, int):
            return self.file_level
        return cast(int, getattr(logging, self.file_level))


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
