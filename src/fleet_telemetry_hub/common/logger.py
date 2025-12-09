# fleet_telemetry_hub/utils/logger.py
"""
Logging configuration for the fleet_telemetry_hub package.

Provides centralized logging setup to ensure consistent log formatting
and output across all modules in the package.
"""

import logging
import sys
from pathlib import Path

from fleet_telemetry_hub.config import LoggingConfig

__all__: list[str] = ['setup_logger']


def setup_logger(
    logging_level: int | None = None,
    config: LoggingConfig | None = None,
) -> logging.Logger:
    """
    Set up logging for the fuelsync package.

    This function configures the package-level logger so that all modules
    inherit the same log level and handler configuration. This ensures
    consistent logging throughout the package.

    The function is idempotent - calling it multiple times will completely
    reset and reconfigure the handlers based on the provided arguments.

    Args:
        logging_level: Default logging level (e.g., logging.INFO) to use for
                      the console if NO config object is provided.
        config: Optional validated configuration object. If provided:
                - Console logging uses config.console_level
                - File logging is enabled if config.file_path is set
                - The 'logging_level' argument is ignored.

    Returns:
        The package-level logger ('fleet_telemetry_hub') for reference. All module-level
        loggers created with logging.getLogger(__name__) will automatically
        inherit this configuration.

    Example:
        >>> # Simple console logging (default INFO)
        >>> setup_logger()

        >>> # Simple console logging (DEBUG)
        >>> setup_logger(logging_level=logging.DEBUG)

        >>> # Full config (Console + File)
        >>> config = load_config()
        >>> setup_logger(config=config)
    """
    # Get the package-level logger (parent of all module loggers)
    package_logger: logging.Logger = logging.getLogger('fleet_telemetry_hub')

    # Clear existing handlers to allow reconfiguration/idempotency
    # This prevents duplicate logs if setup_logger is called multiple times
    package_logger.handlers.clear()

    # Define consistent log format for all handlers
    log_format: logging.Formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)-8s - [%(name)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    # --- 1. Configure Console Handler ---
    # Determine console level: Config priority > Argument fallback
    if logging_level is None:
        logging_level = logging.INFO
    if config:
        console_level: int = config.get_console_level_int()
    else:
        console_level = logging_level

    console_handler: logging.Handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(console_level)
    package_logger.addHandler(console_handler)

    # --- 2. Configure File Handler (Config Only) ---
    file_level: int | None = None

    if config and config.file_path and config.get_file_level_int():
        log_file_path: Path = config.file_path
        file_level = config.get_file_level_int()

        # Ensure parent directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler: logging.FileHandler = logging.FileHandler(
            filename=str(log_file_path),
            mode='a',  # Append mode
            encoding='utf-8',
        )
        file_handler.setFormatter(log_format)
        # We know file_level is int because of the check above, but type checker
        # might need help or we rely on runtime correctness from Pydantic
        if file_level is not None:
            file_handler.setLevel(file_level)

        package_logger.addHandler(file_handler)

        # Log the location of the log file to the console for visibility
        # We use a direct print or a temporary log to ensure it's seen
        # logging.info might not show up if console_level is WARNING
        if console_level <= logging.INFO:
            print(f'Logging to file: {log_file_path}', file=sys.stderr)

    # --- 3. Set Package Logger Level ---
    # The logger's level must be the lowest (most verbose) of all its handlers.
    # If the logger is set to INFO, a DEBUG handler will never receive messages.
    effective_level: int = console_level
    if file_level is not None:
        effective_level = min(console_level, file_level)

    package_logger.setLevel(effective_level)

    # Return the package logger for reference
    # Module-level loggers (created with logging.getLogger(__name__))
    # will automatically inherit this configuration
    return package_logger
