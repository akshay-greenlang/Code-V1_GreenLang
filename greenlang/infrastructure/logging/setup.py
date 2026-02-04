"""
Logging Setup - INFRA-009

Main entry point for configuring the GreenLang structured logging system.
Wires together structlog, stdlib logging, redaction, and formatting into a
cohesive processor chain. Call :func:`configure_logging` once at application
startup (e.g. in your FastAPI ``lifespan`` or ``__main__`` block).

The processor chain is:

1. ``merge_contextvars`` -- injects correlation IDs from context
2. ``add_log_level`` -- adds the ``level`` field
3. ``add_logger_name`` -- adds the ``logger`` field
4. ``TimeStamper`` -- adds ISO 8601 UTC timestamp
5. ``StackInfoRenderer`` -- renders ``stack_info`` if present
6. ``format_exc_info`` -- renders exception tracebacks
7. ``add_service_info`` -- injects service_name, version, environment
8. ``add_caller_info`` -- injects module/function/line for ERROR/CRITICAL
9. ``RedactionProcessor`` -- redacts PII/secrets (if enabled)
10. ``JSONRenderer`` (production) or ``ConsoleRenderer`` (development)

Functions:
    - configure_logging: Configure structlog + stdlib logging.
    - get_logger: Obtain a structlog bound logger by name.
    - add_service_info: Custom processor that injects service metadata.
    - add_caller_info: Custom processor that injects caller location.

Example:
    >>> from greenlang.infrastructure.logging.setup import configure_logging, get_logger
    >>> configure_logging()
    >>> log = get_logger("my_module")
    >>> log.info("application_started", version="1.0.0")
"""

from __future__ import annotations

import inspect
import logging
import logging.handlers
import random
import sys
from typing import Any, Optional

import structlog

from greenlang.infrastructure.logging.config import LoggingConfig, get_config
from greenlang.infrastructure.logging.redaction import RedactionProcessor

logger = logging.getLogger(__name__)

# Module-level flag to prevent double-configuration
_configured: bool = False


# ---------------------------------------------------------------------------
# Custom Processors
# ---------------------------------------------------------------------------


def add_service_info(
    logger_: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Inject service metadata into every log entry.

    Reads ``service_name``, ``service_version``, and ``environment`` from
    the logging configuration singleton and adds them to the event dict.
    This ensures every log record is tagged with the emitting service for
    downstream aggregation and filtering in Loki/Grafana.

    Args:
        logger_: The wrapped logger object (unused).
        method_name: The name of the log method called (unused).
        event_dict: The structured log event dictionary.

    Returns:
        The event dict with service metadata injected.
    """
    config = get_config()
    event_dict["service_name"] = config.service_name
    event_dict["service_version"] = config.service_version
    event_dict["environment"] = config.environment
    return event_dict


def add_caller_info(
    logger_: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add caller location (module, function, line) for ERROR and CRITICAL logs.

    Only injects caller information for ERROR and CRITICAL log levels
    to avoid the performance overhead of frame inspection on every log call.
    The caller info helps engineers locate the exact code path that produced
    an error during incident response.

    Args:
        logger_: The wrapped logger object (unused).
        method_name: The name of the log method called (unused).
        event_dict: The structured log event dictionary.

    Returns:
        The event dict, potentially with ``caller_module``, ``caller_function``,
        and ``caller_lineno`` fields added.
    """
    level = event_dict.get("level", "").upper()
    if level not in ("ERROR", "CRITICAL"):
        return event_dict

    # Walk up the call stack to find the actual caller (skip structlog frames)
    frame = inspect.currentframe()
    try:
        # Walk up frames to find the caller outside structlog internals
        caller_frame = frame
        for _ in range(10):  # safety limit
            if caller_frame is None:
                break
            caller_frame = caller_frame.f_back
            if caller_frame is None:
                break
            module = caller_frame.f_globals.get("__name__", "")
            # Skip structlog and logging internals
            if not module.startswith(("structlog", "logging", __name__)):
                event_dict["caller_module"] = module
                event_dict["caller_function"] = caller_frame.f_code.co_name
                event_dict["caller_lineno"] = caller_frame.f_lineno
                break
    finally:
        del frame

    return event_dict


def _sample_filter(
    logger_: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Drop a fraction of DEBUG-level log entries based on sample_rate.

    This processor checks the configured ``sample_rate`` and randomly
    drops DEBUG messages to reduce log volume in high-throughput
    environments. Non-DEBUG messages are never sampled.

    Args:
        logger_: The wrapped logger object (unused).
        method_name: The name of the log method called.
        event_dict: The structured log event dictionary.

    Returns:
        The event dict (unmodified).

    Raises:
        structlog.DropEvent: If the message is sampled out.
    """
    config = get_config()

    # Only sample DEBUG level
    level = event_dict.get("level", "").upper()
    if level != "DEBUG":
        return event_dict

    # sample_rate == 1.0 means keep everything
    if config.sample_rate >= 1.0:
        return event_dict

    # sample_rate == 0.0 means drop all DEBUG
    if config.sample_rate <= 0.0:
        raise structlog.DropEvent()

    # Random sampling
    if random.random() > config.sample_rate:  # noqa: S311
        raise structlog.DropEvent()

    return event_dict


def _truncate_message(
    logger_: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Truncate the event message if it exceeds max_message_length.

    Guards against accidentally logging enormous payloads (e.g. full
    HTTP response bodies) that would bloat log storage.

    Args:
        logger_: The wrapped logger object (unused).
        method_name: The name of the log method called (unused).
        event_dict: The structured log event dictionary.

    Returns:
        The event dict with the ``event`` field truncated if needed.
    """
    config = get_config()
    event = event_dict.get("event", "")
    if isinstance(event, str) and len(event) > config.max_message_length:
        event_dict["event"] = (
            event[: config.max_message_length] + "... [TRUNCATED]"
        )
        event_dict["_truncated"] = True
    return event_dict


# ---------------------------------------------------------------------------
# Main Configuration
# ---------------------------------------------------------------------------


def configure_logging(config: Optional[LoggingConfig] = None) -> None:
    """Configure structlog and stdlib logging for the application.

    This function should be called once at application startup. It sets up
    the structlog processor chain, configures stdlib logging with the
    appropriate formatter, and optionally adds file and async handlers.

    If called multiple times, subsequent calls are no-ops unless the
    module-level ``_configured`` flag is reset (for testing).

    Args:
        config: Optional logging configuration. If None, the singleton
            from :func:`get_config` is used.

    Example:
        >>> from greenlang.infrastructure.logging.config import LoggingConfig
        >>> configure_logging(LoggingConfig(level="DEBUG", format="console"))
    """
    global _configured

    if _configured:
        logger.debug("Logging already configured, skipping re-configuration.")
        return

    if config is None:
        config = get_config()

    log_level = config.get_structlog_level()

    # ----- Build shared processors -----
    # These processors run for BOTH structlog-native loggers and stdlib
    # loggers that are funnelled through ProcessorFormatter. They must
    # NOT include a renderer -- the renderer lives in the formatter.

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        add_service_info,
        add_caller_info,
        _truncate_message,
        _sample_filter,
    ]

    # Optionally add redaction
    if config.enable_redaction:
        redaction_processor = RedactionProcessor(
            patterns=config.redaction_patterns if config.redaction_patterns else None,
        )
        shared_processors.append(redaction_processor)

    # ----- Choose renderer -----

    if config.format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    # ----- Configure structlog -----
    # structlog loggers use stdlib.LoggerFactory so that log records flow
    # through stdlib handlers. The shared_processors prepare the event_dict
    # and the stdlib.ProcessorFormatter renders the final output.

    structlog.configure(
        processors=shared_processors + [
            # This processor converts the structlog event_dict into stdlib
            # LogRecord attributes so ProcessorFormatter can pick them up.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # ----- Configure stdlib logging -----
    # Use ProcessorFormatter so that BOTH structlog and stdlib log records
    # are rendered identically (JSON or console). The "foreign_pre_chain"
    # handles stdlib records that did NOT originate from structlog.

    foreign_pre_chain: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        add_service_info,
    ]

    # Add redaction to foreign pre-chain too
    if config.enable_redaction:
        foreign_redactor = RedactionProcessor(
            patterns=config.redaction_patterns if config.redaction_patterns else None,
        )
        foreign_pre_chain.append(foreign_redactor)

    processor_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=foreign_pre_chain,
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicate output
    root_logger.handlers.clear()

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(processor_formatter)

    if config.async_logging:
        # Wrap the console handler in a QueueHandler for non-blocking I/O
        queue_handler = _create_queue_handler(console_handler)
        root_logger.addHandler(queue_handler)
    else:
        root_logger.addHandler(console_handler)

    # Optional file handler (always JSON for machine parsing)
    if config.log_file:
        file_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
            foreign_pre_chain=foreign_pre_chain,
        )

        file_handler = logging.FileHandler(
            config.log_file, encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)

        if config.async_logging:
            queue_file_handler = _create_queue_handler(file_handler)
            root_logger.addHandler(queue_file_handler)
        else:
            root_logger.addHandler(file_handler)

    _configured = True

    logger.info(
        "Structured logging configured: level=%s, format=%s, env=%s, "
        "redaction=%s, async=%s, file=%s",
        config.level,
        config.format,
        config.environment,
        config.enable_redaction,
        config.async_logging,
        config.log_file or "none",
    )


def _create_queue_handler(
    target_handler: logging.Handler,
) -> logging.handlers.QueueHandler:
    """Create a QueueHandler + QueueListener pair for non-blocking log I/O.

    The QueueHandler enqueues log records without blocking the caller.
    A background QueueListener thread dequeues and dispatches them to
    the target handler.

    Args:
        target_handler: The actual handler that processes log records.

    Returns:
        A QueueHandler that enqueues records for async processing.
    """
    import queue

    log_queue: queue.Queue[Any] = queue.Queue(maxsize=10000)
    queue_handler = logging.handlers.QueueHandler(log_queue)

    listener = logging.handlers.QueueListener(
        log_queue,
        target_handler,
        respect_handler_level=True,
    )
    listener.start()

    return queue_handler


# ---------------------------------------------------------------------------
# Logger Factory
# ---------------------------------------------------------------------------


def get_logger(name: str) -> structlog.BoundLogger:
    """Obtain a structlog bound logger by name.

    The returned logger automatically includes all processors configured
    by :func:`configure_logging`, including context variable merging,
    service metadata injection, and redaction.

    Args:
        name: The logger name, typically ``__name__`` of the calling module.

    Returns:
        A structlog :class:`BoundLogger` instance.

    Example:
        >>> log = get_logger("greenlang.agents.emission")
        >>> log.info("calculation_complete", scope="scope1", tonnes_co2e=42.5)
    """
    return structlog.get_logger(name)


# ---------------------------------------------------------------------------
# Testing Support
# ---------------------------------------------------------------------------


def reset_logging() -> None:
    """Reset the module-level configuration flag.

    Used in tests to allow re-configuration between test cases. This does
    NOT undo the stdlib logging configuration -- callers should also reset
    the root logger handlers if needed.
    """
    global _configured
    _configured = False
