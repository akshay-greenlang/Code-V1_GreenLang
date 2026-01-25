"""
GL-013 PREDICTMAINT - Grafana Subpackage

Grafana dashboard definitions and specifications for predictive maintenance
visualization.

This subpackage provides:
    - Dashboard specifications document
    - Grafana dashboard JSON definition
    - Panel configurations for equipment monitoring

Files:
    - DASHBOARD_SPECIFICATIONS.md: Detailed dashboard design specifications
    - gl013_predictive_maintenance.json: Grafana dashboard JSON

Dashboard Features:
    - Executive summary with KPIs
    - Equipment health overview
    - Failure prediction panels
    - Vibration analysis (ISO 10816 compliant)
    - Temperature monitoring
    - Anomaly detection
    - Maintenance scheduling
    - System performance

Example:
    >>> import json
    >>> from pathlib import Path
    >>>
    >>> # Load dashboard JSON
    >>> dashboard_path = Path(__file__).parent / "gl013_predictive_maintenance.json"
    >>> with open(dashboard_path) as f:
    ...     dashboard = json.load(f)
    >>> print(f"Dashboard: {dashboard['title']}")

Author: GL-MonitoringEngineer
Version: 1.0.0
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

__version__ = "1.0.0"

# Dashboard file paths
_PACKAGE_DIR = Path(__file__).parent
DASHBOARD_JSON_PATH = _PACKAGE_DIR / "gl013_predictive_maintenance.json"
SPECIFICATIONS_PATH = _PACKAGE_DIR / "DASHBOARD_SPECIFICATIONS.md"


def load_dashboard() -> Dict[str, Any]:
    """
    Load the Grafana dashboard JSON.

    Returns:
        Dictionary containing Grafana dashboard definition

    Example:
        >>> dashboard = load_dashboard()
        >>> print(f"Title: {dashboard['title']}")
        >>> print(f"Panels: {len(dashboard['panels'])}")
    """
    with open(DASHBOARD_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dashboard_uid() -> str:
    """
    Get the dashboard UID.

    Returns:
        Dashboard UID string
    """
    dashboard = load_dashboard()
    return dashboard.get("uid", "gl013-predictive-maintenance")


def get_dashboard_title() -> str:
    """
    Get the dashboard title.

    Returns:
        Dashboard title string
    """
    dashboard = load_dashboard()
    return dashboard.get("title", "GL-013 Predictive Maintenance")


def get_panel_count() -> int:
    """
    Get the number of panels in the dashboard.

    Returns:
        Number of panels
    """
    dashboard = load_dashboard()
    return len(dashboard.get("panels", []))


def export_dashboard_for_provisioning(
    folder: str = "GreenLang",
    overwrite: bool = True
) -> Dict[str, Any]:
    """
    Export dashboard in Grafana provisioning format.

    Args:
        folder: Grafana folder name
        overwrite: Whether to overwrite existing dashboard

    Returns:
        Dictionary in provisioning format
    """
    dashboard = load_dashboard()
    return {
        "dashboard": dashboard,
        "folderId": 0,
        "folderUid": "",
        "message": "Provisioned by GL-013 monitoring package",
        "overwrite": overwrite
    }


__all__ = [
    "DASHBOARD_JSON_PATH",
    "SPECIFICATIONS_PATH",
    "load_dashboard",
    "get_dashboard_uid",
    "get_dashboard_title",
    "get_panel_count",
    "export_dashboard_for_provisioning",
]
