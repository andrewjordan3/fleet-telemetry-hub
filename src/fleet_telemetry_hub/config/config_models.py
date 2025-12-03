# fleet_telemetry_hub/config/config_models.py
"""
Configuration management for Fleet Telemetry Integration.

This module provides Pydantic models and loading functionality for the master
configuration file that controls dual-source ELD data extraction from Motive
and Samsara providers.

Design Decisions:
-----------------
- All models use `extra='forbid'` to catch typos and invalid fields in YAML
  configuration files early, preventing silent misconfiguration.

- No logging occurs within this module because the logging configuration itself
  is defined here. Logging must be configured by the caller after loading config.

- SSL verification supports three modes to handle corporate proxy environments:
  1. `True` - Standard verification using system CA bundle
  2. `False` - Disabled verification (use with caution, required for some proxies)
  3. String path - Custom CA bundle (e.g., exported Zscaler root certificate)

- SecretStr is used for API keys to prevent accidental exposure in logs, repr(),
  or error messages. The actual value must be accessed via `.get_secret_value()`.

Usage:
------
    import yaml
    from fleet_telemetry_hub.config.config_models import TelemetryConfig

    with open('config.yaml', 'r') as config_file:
        raw_config = yaml.safe_load(config_file)

    config = TelemetryConfig.model_validate(raw_config)
"""

from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

# =============================================================================
# Public API
# =============================================================================

__all__: list[str] = [
    'CompressionType',
    'LogLevelName',
    'LoggingConfig',
    'PipelineConfig',
    'ProviderConfig',
    'StorageConfig',
    'TelemetryConfig',
]

# =============================================================================
# Type Aliases
# =============================================================================

# Valid logging level names recognized by Python's logging module.
# Using Literal rather than an Enum because these map directly to stdlib names.
LogLevelName = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

# Numeric equivalents of log level names for validation purposes.
# These are the only valid integer values accepted when specifying log levels.
LOG_LEVEL_VALUES: frozenset[int] = frozenset({10, 20, 30, 40, 50})

# Mapping from level name to numeric value, avoiding import of logging module
# in the model layer to maintain separation of concerns.
LOG_LEVEL_NAME_TO_INT: dict[LogLevelName, int] = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50,
}

# Valid compression algorithms supported by pandas.to_parquet() and pyarrow.
# None means no compression (fastest writes, largest files).
# 'snappy' is the default: good balance of speed and compression ratio.
CompressionType = Literal['snappy', 'gzip', 'brotli', 'lz4', 'zstd'] | None


# =============================================================================
# Provider Configuration
# =============================================================================


class ProviderConfig(BaseModel):
    """Configuration for a single telemetry provider (Motive or Samsara).

    This model encapsulates all connection and retry settings for an ELD API
    provider. Each provider can be independently enabled/disabled, allowing
    gradual rollout or temporary exclusion during outages.

    Network Resilience:
        The retry configuration uses exponential backoff to handle transient
        failures gracefully. The delay between retries follows the formula:
        `delay = retry_backoff_factor * (2 ** attempt_number)`

        Example with backoff_factor=0.5:
          Attempt 1: 0.5 * 2^0 = 0.5 seconds
          Attempt 2: 0.5 * 2^1 = 1.0 seconds
          Attempt 3: 0.5 * 2^2 = 2.0 seconds

    SSL/TLS Handling:
        Corporate proxy environments (e.g., Zscaler) often perform TLS
        interception, which breaks standard certificate verification.
        The verify_ssl field supports three modes:
          - True: Standard verification (default, use in production)
          - False: Disabled verification (insecure, use only when necessary)
          - Path string: Custom CA bundle path (preferred for proxy environments)

    Attributes:
        enabled: Whether this provider should be queried during extraction.
            Disabled providers are completely skipped, not just rate-limited.
        base_url: Root API endpoint URL. Must include scheme (https://) and
            must not have a trailing slash (normalized automatically).
        api_key: API authentication token. Stored as SecretStr to prevent
            accidental logging. Access via api_key.get_secret_value().
        request_timeout: Connection and read timeout as [connect, read] seconds.
            Connect timeout controls TCP handshake; read timeout controls
            response body download. Both must be positive integers.
        max_retries: Maximum retry attempts for failed requests (1-10 range).
            Total attempts = 1 (initial) + max_retries.
        retry_backoff_factor: Multiplier for exponential backoff calculation.
            Higher values increase delay between retries.
        verify_ssl: SSL certificate verification mode. False disables (insecure),
            True uses system CA store, or provide path to custom CA bundle.
        rate_limit_requests_per_second: Maximum API requests per second.
            Enforced client-side to respect provider rate limits.
    """

    model_config = ConfigDict(
        extra='forbid',  # Reject unknown fields to catch YAML typos early
    )

    enabled: bool = Field(
        description='Whether this provider should be queried during extraction',
    )
    base_url: str = Field(
        description='Root API endpoint URL with scheme, without trailing slash',
    )
    api_key: SecretStr = Field(
        description='API authentication token (masked in logs and repr)',
    )
    request_timeout: tuple[int, int] = Field(
        description='[connect_timeout, read_timeout] in seconds; both must be positive',
    )
    max_retries: int = Field(
        ge=1,
        le=10,
        description='Maximum retry attempts for failed requests (1-10)',
    )
    retry_backoff_factor: float = Field(
        gt=0.0,
        le=60.0,
        description='Exponential backoff multiplier; delay = factor * (2 ** attempt)',
    )
    verify_ssl: bool | str = Field(
        description='False to disable SSL, True for system CA, or path to CA bundle',
    )
    rate_limit_requests_per_second: int = Field(
        gt=0,
        le=100,
        description='Maximum requests per second (client-side rate limiting)',
    )

    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, base_url: str) -> str:
        """Validate and normalize the API base URL.

        Ensures the URL has a valid HTTP/HTTPS scheme and removes any trailing
        slash for consistent path joining downstream.

        Args:
            base_url: The API base URL to validate.

        Returns:
            Normalized URL without trailing slash.

        Raises:
            ValueError: If URL is empty or missing http/https scheme.
        """
        if not base_url:
            raise ValueError('base_url cannot be empty')

        # Require explicit scheme to prevent accidental http:// usage
        if not base_url.startswith(('http://', 'https://')):
            raise ValueError(
                f"base_url must start with 'http://' or 'https://', got: {base_url!r}"
            )

        # Normalize by removing trailing slash for consistent path construction
        return base_url.rstrip('/')

    @field_validator('api_key')
    @classmethod
    def validate_api_key_not_empty(cls, api_key: SecretStr) -> SecretStr:
        """Ensure API key is not empty or whitespace-only.

        Args:
            api_key: The API key to validate.

        Returns:
            The validated API key.

        Raises:
            ValueError: If API key is empty or contains only whitespace.
        """
        secret_value: str = api_key.get_secret_value()
        if not secret_value or not secret_value.strip():
            raise ValueError('api_key cannot be empty or whitespace-only')
        return api_key

    @field_validator('request_timeout')
    @classmethod
    def validate_timeout_values_positive(
        cls, timeout: tuple[int, int]
    ) -> tuple[int, int]:
        """Ensure both timeout values are positive integers.

        The connect timeout limits how long to wait for TCP connection
        establishment. The read timeout limits how long to wait for the
        complete response body after connection is established.

        Args:
            timeout: Tuple of [connect_timeout, read_timeout] in seconds.

        Returns:
            The validated timeout tuple.

        Raises:
            ValueError: If either timeout value is non-positive.
        """
        connect_timeout, read_timeout = timeout

        if connect_timeout <= 0:
            raise ValueError(
                f'connect_timeout must be positive, got: {connect_timeout}'
            )
        if read_timeout <= 0:
            raise ValueError(f'read_timeout must be positive, got: {read_timeout}')

        return timeout

    @field_validator('verify_ssl')
    @classmethod
    def validate_ssl_configuration(cls, verify_ssl: bool | str) -> bool | str:
        """Validate SSL verification configuration.

        When a string path is provided (for custom CA bundles), verifies
        the file exists and is a regular file (not a directory).

        Args:
            verify_ssl: Boolean or path to CA certificate bundle file.

        Returns:
            The validated SSL configuration.

        Raises:
            ValueError: If string path does not exist or is not a file.
        """
        if isinstance(verify_ssl, str):
            cert_path = Path(verify_ssl)

            if not cert_path.exists():
                raise ValueError(f'SSL certificate bundle file not found: {verify_ssl}')
            if not cert_path.is_file():
                raise ValueError(
                    f'SSL certificate path must be a file, not directory: {verify_ssl}'
                )

        return verify_ssl


# =============================================================================
# Pipeline Configuration
# =============================================================================


class PipelineConfig(BaseModel):
    """Configuration for data extraction pipeline execution.

    Controls the temporal boundaries and network behavior of the extraction
    pipeline. The lookback_days parameter is critical for data completeness
    since ELD providers may have delayed data availability.

    Batching Strategy:
        The pipeline processes data in time-based batches to provide incremental
        progress, reduce memory pressure, and enable partial saves on long runs.
        The batch_increment_days parameter controls batch size:
          - 1.0: One day per batch (default, good balance)
          - 0.5: Half-day batches (for high-volume fleets)
          - 2.0: Two-day batches (for smaller fleets or faster runs)

    Incremental Extraction Strategy:
        On subsequent runs, the pipeline uses the last successful extraction
        timestamp minus lookback_days to catch any late-arriving data. This
        overlap strategy ensures data completeness at the cost of some
        reprocessing. Deduplication downstream handles the overlap.

    Truststore Integration:
        When use_truststore=True, the pipeline uses the `truststore` library
        to build an SSLContext from the Windows certificate store. This is
        essential for corporate environments where Zscaler or similar proxies
        inject their own root CA that isn't in Python's bundled certificates.

    Attributes:
        default_start_date: ISO-8601 date (YYYY-MM-DD) for initial historical
            backfill. Used only when no prior extraction state exists.
        lookback_days: Number of days to overlap when calculating incremental
            extraction start date. Compensates for delayed data availability.
            Range: 0-30 days.
        batch_increment_days: Size of each processing batch in days. Supports
            fractional values (e.g., 0.5 for 12-hour batches). Smaller batches
            save more frequently but have more overhead. Range: 0.25-7.0 days.
        request_delay_seconds: Artificial delay between sequential API requests.
            Use to stay well under rate limits or reduce load on provider APIs.
        use_truststore: When True, use truststore library to build SSLContext
            from Windows system certificate store. Required for Zscaler
            environments. When False, use httpx default certificate handling.
    """

    model_config = ConfigDict(extra='forbid')

    default_start_date: str = Field(
        pattern=r'^\d{4}-\d{2}-\d{2}$',
        description='ISO-8601 date (YYYY-MM-DD) for initial backfill start',
    )
    lookback_days: int = Field(
        ge=0,
        le=30,
        description='Days to overlap on incremental runs to catch late data (0-30)',
    )
    batch_increment_days: float = Field(
        default=1.0,
        ge=0.25,
        le=7.0,
        description='Batch size in days; supports fractional (0.5 = 12 hours)',
    )
    request_delay_seconds: float = Field(
        ge=0.0,
        le=30.0,
        description='Delay between API requests in seconds (0-30)',
    )
    use_truststore: bool = Field(
        default=False,
        description='Use truststore library for Windows system CA certificates',
    )

    @field_validator('default_start_date')
    @classmethod
    def validate_start_date_is_valid_iso_date(cls, date_string: str) -> str:
        """Validate default_start_date is a parseable ISO-8601 date.

        The regex pattern in Field ensures format, but this validator ensures
        the date itself is valid (e.g., rejects 2024-02-30).

        Args:
            date_string: Date string in YYYY-MM-DD format.

        Returns:
            The validated date string.

        Raises:
            ValueError: If date is invalid (e.g., February 30).
        """
        try:
            datetime.fromisoformat(date_string)
        except ValueError as parse_error:
            raise ValueError(
                f"default_start_date '{date_string}' is not a valid date: {parse_error}"
            ) from parse_error

        return date_string


# =============================================================================
# Storage Configuration
# =============================================================================


class StorageConfig(BaseModel):
    """Configuration for Parquet file storage.

    Controls where extracted telemetry data is persisted and how it's
    compressed. The Parquet format provides efficient columnar storage
    with excellent compression ratios for telemetry data.

    Compression Trade-offs:
        - snappy: Fast compression/decompression, moderate ratio (default)
        - gzip: Slower but better ratio, good for archival
        - lz4: Fastest, lower ratio, good for frequently-read data
        - zstd: Best ratio with reasonable speed, good general choice
        - brotli: Highest ratio, slowest, best for cold storage
        - None: No compression, fastest writes, largest files

    Attributes:
        parquet_path: Path to the output Parquet file. Relative paths are
            resolved from the working directory. Extension .parquet is
            appended automatically if missing.
        parquet_compression: Compression codec for Parquet writer. Choose
            based on read/write frequency and storage constraints.
    """

    model_config = ConfigDict(extra='forbid')

    parquet_path: Path = Field(
        description='Output Parquet file path (.parquet extension auto-added)',
    )
    parquet_compression: CompressionType = Field(
        default='snappy',
        description="Compression codec: 'snappy', 'gzip', 'brotli', 'lz4', 'zstd', or None",
    )

    @field_validator('parquet_path', mode='before')
    @classmethod
    def normalize_parquet_path(cls, path_value: str | Path) -> Path:
        """Normalize path and ensure .parquet extension.

        Converts string paths to Path objects and appends .parquet extension
        if not already present for consistent file naming.

        Args:
            path_value: Path to parquet file as string or Path.

        Returns:
            Normalized Path with .parquet extension.
        """
        path_string: str = str(path_value)

        if not path_string.lower().endswith('.parquet'):
            path_string = f'{path_string}.parquet'

        return Path(path_string)


# =============================================================================
# Logging Configuration
# =============================================================================


class LoggingConfig(BaseModel):
    """Configuration for application logging output.

    Supports dual-destination logging: console (always enabled) and optional
    file output. Console output is typically set to INFO for operational
    visibility, while file output captures DEBUG-level detail for debugging.

    File Logging:
        File logging is enabled by providing a file_path. The file_level
        defaults to DEBUG if not specified, capturing maximum detail.
        Parent directories are created automatically by the logging setup.

    Level Specification:
        Levels can be specified as names ('DEBUG', 'INFO', etc.) or as
        their numeric equivalents (10, 20, etc.). String names are
        preferred for readability in YAML configuration files.

    Attributes:
        file_path: Path to log file. None disables file logging.
            Extension .log is appended automatically if missing.
            Parent directories should be created by logging setup.
        console_level: Minimum log level for console (stderr) output.
            Accepts level name or numeric value.
        file_level: Minimum log level for file output. None disables
            file logging. Defaults to DEBUG if file_path is provided.
    """

    model_config = ConfigDict(extra='forbid')

    file_path: Path | None = Field(
        default=None,
        description='Log file path (.log extension auto-added). None disables file logging.',
    )
    console_level: LogLevelName | int = Field(
        default='INFO',
        description="Console log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', or int",
    )
    file_level: LogLevelName | int | None = Field(
        default=None,
        description='File log level. None disables file logging.',
    )

    @field_validator('file_path', mode='before')
    @classmethod
    def normalize_log_file_path(cls, path_value: str | Path | None) -> Path | None:
        """Normalize path and ensure .log extension.

        Args:
            path_value: Path to log file, or None to disable file logging.

        Returns:
            Normalized Path with .log extension, or None.
        """
        if path_value is None:
            return None

        path_string: str = str(path_value)

        if not path_string.lower().endswith('.log'):
            path_string = f'{path_string}.log'

        return Path(path_string)

    @field_validator('console_level', 'file_level', mode='after')
    @classmethod
    def validate_numeric_log_level(
        cls, level_value: LogLevelName | int | None
    ) -> LogLevelName | int | None:
        """Validate numeric log levels are standard Python logging values.

        String level names are validated by the Literal type annotation.
        This validator ensures numeric values match Python's logging constants.

        Args:
            level_value: Log level as name string, integer, or None.

        Returns:
            The validated log level.

        Raises:
            ValueError: If numeric level is not a standard logging value.
        """
        if level_value is None or isinstance(level_value, str):
            return level_value

        # Integer level must be a standard Python logging level
        if level_value not in LOG_LEVEL_VALUES:
            raise ValueError(
                f'Numeric log level must be one of {sorted(LOG_LEVEL_VALUES)}, '
                f'got: {level_value}'
            )

        return level_value

    @model_validator(mode='after')
    def ensure_file_logging_configuration_consistency(self) -> Self:
        """Ensure file_path and file_level are consistently configured.

        If file_path is provided without file_level, defaults to DEBUG.
        If file_level is provided without file_path, raises an error since
        there's nowhere to write the logs.

        Returns:
            The validated model with consistent file logging settings.

        Raises:
            ValueError: If file_level is set but file_path is missing.
        """
        has_file_path: bool = self.file_path is not None
        has_file_level: bool = self.file_level is not None

        if has_file_path and not has_file_level:
            # Sensible default: capture everything to file for debugging
            # Note: Cannot log warning here as logging isn't configured yet
            self.file_level = 'DEBUG'

        if has_file_level and not has_file_path:
            raise ValueError(
                'file_level is specified but file_path is missing. '
                'Provide file_path to enable file logging, or remove file_level.'
            )

        return self

    def get_console_level_int(self) -> int:
        """Convert console_level to numeric value for logging module.

        Returns:
            Integer logging level (10=DEBUG through 50=CRITICAL).
        """
        if isinstance(self.console_level, int):
            return self.console_level
        return LOG_LEVEL_NAME_TO_INT[self.console_level]

    def get_file_level_int(self) -> int | None:
        """Convert file_level to numeric value for logging module.

        Returns:
            Integer logging level, or None if file logging is disabled.
        """
        if self.file_level is None:
            return None
        if isinstance(self.file_level, int):
            return self.file_level
        return LOG_LEVEL_NAME_TO_INT[self.file_level]


# =============================================================================
# Root Configuration
# =============================================================================


class TelemetryConfig(BaseModel):
    """Root configuration model for Fleet Telemetry Integration.

        This is the top-level model that aggregates all configuration sections.
        It enforces strict validation across the entire configuration tree and
        ensures at least one provider is enabled for extraction.

        Validation Philosophy:
            All models in this hierarchy use `extra='forbid'` to reject unknown
            fields. This catches configuration typos immediately at load time
            rather than silently ignoring misconfigured options.

        Loading Example:
    ```python
            import yaml
            from pathlib import Path

            config_path = Path('config.yaml')
            with config_path.open('r', encoding='utf-8') as config_file:
                raw_config = yaml.safe_load(config_file)

            config = TelemetryConfig.model_validate(raw_config)
    ```

        Attributes:
            providers: Dictionary mapping provider names (e.g., 'motive', 'samsara')
                to their ProviderConfig instances. At least one must be enabled.
            pipeline: Pipeline execution settings including date ranges and delays.
            storage: Parquet file storage configuration.
            logging: Application logging configuration for console and file output.
    """

    model_config = ConfigDict(
        extra='forbid',  # Reject unknown top-level fields
    )

    providers: dict[str, ProviderConfig] = Field(
        description='Provider name to configuration mapping. At least one must be enabled.',
    )
    pipeline: PipelineConfig = Field(
        description='Pipeline execution settings (dates, delays, SSL handling)',
    )
    storage: StorageConfig = Field(
        description='Parquet file storage configuration',
    )
    logging: LoggingConfig = Field(
        description='Application logging configuration',
    )

    @field_validator('providers')
    @classmethod
    def validate_provider_names_are_lowercase(
        cls, providers: dict[str, ProviderConfig]
    ) -> dict[str, ProviderConfig]:
        """Ensure provider names follow lowercase naming convention.

        Consistent naming prevents issues with case-sensitive file systems
        and makes configuration files more predictable.

        Args:
            providers: Dictionary of provider configurations.

        Returns:
            The validated providers dictionary.

        Raises:
            ValueError: If any provider name contains uppercase characters.
        """
        for provider_name in providers:
            if provider_name != provider_name.lower():
                raise ValueError(
                    f"Provider name must be lowercase: '{provider_name}'. "
                    f"Use '{provider_name.lower()}' instead."
                )

        return providers

    @model_validator(mode='after')
    def validate_at_least_one_provider_enabled(self) -> Self:
        """Ensure at least one provider is enabled for extraction.

        A configuration with all providers disabled would result in no data
        extraction, which is almost certainly a misconfiguration.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If all providers have enabled=False.
        """
        enabled_providers: list[str] = [
            name for name, config in self.providers.items() if config.enabled
        ]

        if not enabled_providers:
            all_providers: str = ', '.join(sorted(self.providers.keys()))
            raise ValueError(
                f'At least one provider must be enabled. '
                f'All providers are currently disabled: {all_providers}'
            )

        return self

    def get_enabled_providers(self) -> dict[str, ProviderConfig]:
        """Return only the enabled provider configurations.

        Convenience method for pipeline code that needs to iterate over
        active providers without checking the enabled flag each time.

        Returns:
            Dictionary containing only providers with enabled=True.
        """
        return {
            name: config for name, config in self.providers.items() if config.enabled
        }
