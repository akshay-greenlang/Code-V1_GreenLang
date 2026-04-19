# Deprecated: Use greenlang.tests.templates instead
# This module provides backwards compatibility shims for template imports

# Re-export from tests.templates
from greenlang.tests.templates.agent_monitoring import (
    OperationalMonitoringMixin,
)

__all__ = [
    "OperationalMonitoringMixin",
]
