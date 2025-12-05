# fleet_telemetry_hub/config/loader.py
"""
Configuration Loading Logic.

This module handles the physical retrieval, parsing, and initial validation of
the application configuration. It serves as the bridge between raw YAML files
on the disk and the strictly typed Pydantic models defined in `model.py`.

Responsibilities:
    1.  File I/O: Safely locating and reading the configuration file.
    2.  Parsing: Converting YAML text into Python dictionaries.
    3.  Validation: Instantiating the `TelemetryConfig` model to enforce types.
    4.  Error Handling: Capturing low-level I/O or parsing errors and logging
        them with context before raising.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from fleet_telemetry_hub.config.config_models import TelemetryConfig

logger: logging.Logger = logging.getLogger(__name__)


def load_config(config_path: Path | str | None = None) -> TelemetryConfig:
    """Load and validate telemetry configuration from YAML file.

    This function reads the YAML configuration file, parses it, and validates
    all fields using the Pydantic model hierarchy. Detailed validation errors
    are logged and re-raised with context.

    Args:
        config_path: Path to the YAML configuration file (relative or absolute).
                    If None, defaults to 'config/telemetry_config.yaml' relative
                    to the current working directory.

    Returns:
        Validated TelemetryConfig instance ready for use.

    Raises:
        FileNotFoundError: If config file does not exist at the specified path.
        yaml.YAMLError: If YAML file is malformed or cannot be parsed.
        ValueError: If configuration fails Pydantic validation (detailed in error message).

    Side Effects:
        Logs configuration loading status at INFO level.
        Logs detailed validation errors at ERROR level if validation fails.

    Example:
        >>> config = load_config("config/telemetry_config.yaml")
        >>> print(config.providers["motive"].base_url)
        'https://api.gomotive.com'
    """
    # Use default path if none provided
    if config_path is None:
        config_path = Path('config/telemetry_config.yaml')
        logger.debug(
            'No config path provided, using default: config/telemetry_config.yaml'
        )
    else:
        config_path = Path(config_path)

    logger.info('Loading telemetry configuration from: %s', config_path)

    # Check file existence
    if not config_path.exists():
        error_message: str = f'Configuration file not found: {config_path}'
        logger.error(error_message)
        raise FileNotFoundError(error_message)

    # Parse YAML
    try:
        with Path.open(config_path, encoding='utf-8') as config_file:
            raw_config_data: Any = yaml.safe_load(config_file)
    except yaml.YAMLError as error:
        error_message = f'Failed to parse YAML configuration: {error}'
        logger.error(error_message)
        raise yaml.YAMLError(error_message) from error

    # Validate with Pydantic
    try:
        validated_config = TelemetryConfig(**raw_config_data)
    except ValueError as error:
        error_message = f'Configuration validation failed: {error}'
        logger.error(error_message)
        raise ValueError(error_message) from error

    logger.info('Configuration loaded and validated successfully')
    return validated_config
