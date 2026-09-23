"""Application logging configured explicitly by an entry point."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.schema import LoggingConfig

_LOGGER_NAMESPACE = "local_requirement_extractor"
_HANDLER_NAMES = {
    "local_requirement_extractor.console",
    "local_requirement_extractor.file",
}


def _application_logger() -> logging.Logger:
    """Return the application parent, silent until explicitly configured."""
    logger = logging.getLogger(_LOGGER_NAMESPACE)
    logger.propagate = False
    if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        logger.addHandler(logging.NullHandler())
    return logger


def setup_logger(name: str = "myapp", log_file: str = "myapp.log") -> logging.Logger:
    """Get an application logger without creating files or output handlers.

    Args:
        name: Module or component name within the application namespace.
        log_file: Retained for compatibility. Configure file output through
            ``configure_logging`` and ``LoggingConfig.file_path`` instead.

    Returns:
        A logger inheriting the application's explicit logging configuration.
    """
    parent = _application_logger()
    return parent.getChild(name)


def configure_logging(config: LoggingConfig) -> None:
    """Apply logging settings to application loggers, preserving other loggers.

    Repeated calls replace only this module's output handlers and close old file
    handles. Module imports and ``setup_logger`` never open log files. Relative
    file paths are resolved from the process working directory.

    Args:
        config: Severity threshold and optional console and rotating file sinks.

    Raises:
        ValueError: The configured severity is not a valid logging level.
        OSError: The requested file output cannot be created.
    """
    level = logging.getLevelName(config.level.upper())
    if not isinstance(level, int):
        raise ValueError(f"Invalid logging.level: {config.level!r}")

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    handlers: list[logging.Handler] = []
    if config.to_file:
        file_path = Path(config.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            file_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.name = "local_requirement_extractor.file"
        handlers.append(file_handler)
    if config.to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.name = "local_requirement_extractor.console"
        handlers.append(console_handler)

    logger = _application_logger()
    for old_handler in list(logger.handlers):
        if old_handler.name in _HANDLER_NAMES:
            logger.removeHandler(old_handler)
            old_handler.close()
    logger.setLevel(level)
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
