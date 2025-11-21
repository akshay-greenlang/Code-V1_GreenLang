# -*- coding: utf-8 -*-
"""
Logging Configuration for CSRD Pipeline

Provides centralized logging setup with file and console handlers.

Author: GreenLang AI Team
Date: 2025-10-18
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from greenlang.determinism import DeterministicClock


def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for CSRD pipeline

    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional file path for log output
        log_format: Optional custom log format

    Returns:
        Configured root logger
    """
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers = []

    # Default format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # More detailed in file

        # More detailed format for file
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.info(f"Logging configured - Level: {log_level}, File: {log_file}")

    return logger


def get_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for a specific module

    Args:
        name: Logger name (typically __name__)
        log_level: Optional specific log level for this logger

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    if log_level:
        logger.setLevel(getattr(logging, log_level.upper()))

    return logger


def setup_pipeline_logging(pipeline_run_id: str) -> logging.Logger:
    """
    Setup logging for a specific pipeline run

    Args:
        pipeline_run_id: Unique identifier for this pipeline run

    Returns:
        Configured logger
    """
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = DeterministicClock.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"pipeline_{pipeline_run_id}_{timestamp}.log"

    return setup_logging(
        log_level='INFO',
        log_file=str(log_file)
    )


class LogContext:
    """Context manager for temporary log level changes"""

    def __init__(self, logger: logging.Logger, level: str):
        """
        Initialize log context

        Args:
            logger: Logger to modify
            level: Temporary log level
        """
        self.logger = logger
        self.level = getattr(logging, level.upper())
        self.original_level = logger.level

    def __enter__(self):
        """Set temporary log level"""
        self.logger.setLevel(self.level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original log level"""
        self.logger.setLevel(self.original_level)


# Example usage
if __name__ == "__main__":
    # Setup basic logging
    logger = setup_logging(log_level='INFO')

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Setup with file output
    logger = setup_logging(
        log_level='DEBUG',
        log_file='logs/test.log'
    )

    logger.info("Logging to both console and file")

    # Use log context
    with LogContext(logger, 'DEBUG'):
        logger.debug("This debug message will be shown")

    logger.debug("This debug message will not be shown (back to INFO)")
