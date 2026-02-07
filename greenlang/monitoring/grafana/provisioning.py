# -*- coding: utf-8 -*-
"""
Grafana Dashboard Provisioning
================================

Dashboard validation and end-to-end provisioning utilities for the
GreenLang Grafana SDK. Validates dashboard JSON against Grafana schema
requirements and provides bulk provisioning from files or directories.

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-002 Grafana Dashboards - Python SDK
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Optional

from greenlang.monitoring.grafana.client import GrafanaClient, GrafanaNotFoundError
from greenlang.monitoring.grafana.datasource_manager import DataSourceManager
from greenlang.monitoring.grafana.folder_manager import FolderManager
from greenlang.monitoring.grafana.alert_manager import GrafanaAlertManager
from greenlang.monitoring.grafana.models import Dashboard

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Valid panel types for validation
# ---------------------------------------------------------------------------

VALID_PANEL_TYPES = frozenset({
    "timeseries", "stat", "gauge", "table", "barchart", "piechart",
    "heatmap", "logs", "text", "bargauge", "histogram", "news",
    "dashlist", "row", "canvas", "nodeGraph", "traces", "flamegraph",
    "geomap", "candlestick", "trend", "xychart", "datagrid",
    "state-timeline", "status-history",
})


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of a dashboard validation check.

    Attributes:
        is_valid: Whether the dashboard passed all checks.
        errors: List of error messages (validation failures).
        warnings: List of warning messages (non-fatal issues).
        dashboard_title: Title of the validated dashboard.
        file_path: Path to the source file, if applicable.
    """

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dashboard_title: str = ""
    file_path: str = ""

    def add_error(self, message: str) -> None:
        """Add an error and mark the result as invalid."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning (does not affect validity)."""
        self.warnings.append(message)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class DashboardValidator:
    """Validate Grafana dashboard JSON against schema requirements.

    Performs structural, content, and best-practice checks on dashboard
    definitions to catch errors before deployment.

    Usage::

        validator = DashboardValidator()
        result = validator.validate(dashboard_dict)
        if not result.is_valid:
            for error in result.errors:
                print(f"ERROR: {error}")
    """

    def validate(self, dashboard: dict[str, Any]) -> ValidationResult:
        """Validate a dashboard dict.

        Runs all validation checks and returns a consolidated result.

        Args:
            dashboard: Dashboard JSON dict (the inner 'dashboard' object,
                       not the API wrapper).

        Returns:
            ValidationResult with errors and warnings.
        """
        result = ValidationResult()
        result.dashboard_title = dashboard.get("title", "<untitled>")

        self._check_required_fields(dashboard, result)
        self._check_uid(dashboard, result)
        self._check_panels(dashboard, result)
        self._check_panel_ids(dashboard, result)
        self._check_targets(dashboard, result)
        self._check_variables(dashboard, result)
        self._check_schema_version(dashboard, result)
        self._check_refresh(dashboard, result)

        return result

    def validate_file(self, file_path: str | pathlib.Path) -> ValidationResult:
        """Validate a dashboard JSON file.

        Args:
            file_path: Path to the dashboard JSON file.

        Returns:
            ValidationResult with file_path set.
        """
        path = pathlib.Path(file_path)
        result = ValidationResult(file_path=str(path))

        if not path.exists():
            result.add_error(f"File not found: {path}")
            return result

        if not path.suffix == ".json":
            result.add_error(f"Expected .json file, got: {path.suffix}")
            return result

        try:
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            result.add_error(f"Invalid JSON: {exc}")
            return result

        # Handle both raw dashboard and API wrapper formats
        dashboard = data.get("dashboard", data)
        file_result = self.validate(dashboard)
        file_result.file_path = str(path)
        return file_result

    def validate_directory(
        self,
        directory: str | pathlib.Path,
        recursive: bool = True,
    ) -> list[ValidationResult]:
        """Validate all dashboard JSON files in a directory.

        Args:
            directory: Directory path to scan.
            recursive: Whether to recurse into subdirectories.

        Returns:
            List of ValidationResult, one per file.
        """
        path = pathlib.Path(directory)
        pattern = "**/*.json" if recursive else "*.json"
        results: list[ValidationResult] = []

        for json_file in sorted(path.glob(pattern)):
            result = self.validate_file(json_file)
            results.append(result)

        valid_count = sum(1 for r in results if r.is_valid)
        logger.info(
            "Directory validation: %d/%d valid in %s",
            valid_count, len(results), directory,
        )
        return results

    # -- Individual checks ---------------------------------------------------

    def _check_required_fields(
        self, dashboard: dict[str, Any], result: ValidationResult
    ) -> None:
        """Check that required fields are present."""
        if not dashboard.get("title"):
            result.add_error("Missing required field: 'title'")
        if not dashboard.get("panels") and dashboard.get("panels") != []:
            result.add_warning("Dashboard has no 'panels' field")

    def _check_uid(
        self, dashboard: dict[str, Any], result: ValidationResult
    ) -> None:
        """Check UID format and length."""
        uid = dashboard.get("uid", "")
        if uid and len(uid) > 40:
            result.add_error(f"UID exceeds 40 characters: '{uid}' ({len(uid)} chars)")
        if uid and not uid.replace("-", "").replace("_", "").isalnum():
            result.add_warning(f"UID contains non-alphanumeric characters: '{uid}'")

    def _check_panels(
        self, dashboard: dict[str, Any], result: ValidationResult
    ) -> None:
        """Check all panels have valid types."""
        panels = dashboard.get("panels", [])
        for i, panel in enumerate(panels):
            panel_type = panel.get("type", "")
            if not panel_type:
                result.add_error(f"Panel {i}: missing 'type' field")
            elif panel_type not in VALID_PANEL_TYPES:
                result.add_error(
                    f"Panel {i} ('{panel.get('title', '')}'): "
                    f"unknown type '{panel_type}'"
                )

            # Check nested panels in collapsed rows
            if panel_type == "row" and panel.get("panels"):
                for j, nested in enumerate(panel["panels"]):
                    nested_type = nested.get("type", "")
                    if nested_type and nested_type not in VALID_PANEL_TYPES:
                        result.add_error(
                            f"Panel {i} row nested panel {j}: "
                            f"unknown type '{nested_type}'"
                        )

    def _check_panel_ids(
        self, dashboard: dict[str, Any], result: ValidationResult
    ) -> None:
        """Check that panel IDs are unique."""
        panels = dashboard.get("panels", [])
        seen_ids: set[int] = set()
        for panel in panels:
            panel_id = panel.get("id", 0)
            if panel_id in seen_ids and panel_id != 0:
                result.add_error(
                    f"Duplicate panel ID: {panel_id} "
                    f"(panel '{panel.get('title', '')}')"
                )
            if panel_id != 0:
                seen_ids.add(panel_id)

    def _check_targets(
        self, dashboard: dict[str, Any], result: ValidationResult
    ) -> None:
        """Check that panel targets reference valid datasources."""
        panels = dashboard.get("panels", [])
        for panel in panels:
            panel_type = panel.get("type", "")
            if panel_type == "row":
                continue

            targets = panel.get("targets", [])
            if not targets and panel_type not in ("text", "news", "dashlist", "row"):
                result.add_warning(
                    f"Panel '{panel.get('title', '')}' has no query targets"
                )

            for target in targets:
                expr = target.get("expr", "")
                if not expr:
                    result.add_warning(
                        f"Panel '{panel.get('title', '')}' target "
                        f"{target.get('refId', '?')} has empty expr"
                    )

    def _check_variables(
        self, dashboard: dict[str, Any], result: ValidationResult
    ) -> None:
        """Check template variable definitions."""
        templating = dashboard.get("templating", {})
        variables = templating.get("list", [])
        seen_names: set[str] = set()

        for var in variables:
            name = var.get("name", "")
            if not name:
                result.add_error("Variable missing 'name' field")
                continue
            if name in seen_names:
                result.add_error(f"Duplicate variable name: '{name}'")
            seen_names.add(name)

    def _check_schema_version(
        self, dashboard: dict[str, Any], result: ValidationResult
    ) -> None:
        """Check schema version is reasonable."""
        version = dashboard.get("schemaVersion", 0)
        if version and version < 30:
            result.add_warning(
                f"Schema version {version} is outdated. Consider upgrading to 39+"
            )

    def _check_refresh(
        self, dashboard: dict[str, Any], result: ValidationResult
    ) -> None:
        """Check refresh interval is not too aggressive."""
        refresh = dashboard.get("refresh", "")
        if refresh:
            # Parse simple intervals like '5s', '10s', '1m'
            try:
                if refresh.endswith("s"):
                    seconds = int(refresh[:-1])
                    if seconds < 10:
                        result.add_warning(
                            f"Refresh interval '{refresh}' is below the "
                            f"recommended minimum of 10s"
                        )
            except ValueError:
                pass  # Non-standard format, skip check


# ---------------------------------------------------------------------------
# Provisioner
# ---------------------------------------------------------------------------

class GrafanaProvisioner:
    """End-to-end Grafana provisioning for the GreenLang platform.

    Orchestrates folder creation, data source provisioning, contact point
    setup, notification policy configuration, and dashboard deployment.

    Usage::

        async with GrafanaClient(base_url, api_key) as client:
            provisioner = GrafanaProvisioner(client)
            report = await provisioner.provision_all()
    """

    def __init__(self, client: GrafanaClient) -> None:
        """Initialise the provisioner.

        Args:
            client: Connected GrafanaClient instance.
        """
        self._client = client
        self._folder_manager = FolderManager(client)
        self._datasource_manager = DataSourceManager(client)
        self._alert_manager = GrafanaAlertManager(client)
        self._validator = DashboardValidator()

    async def provision_all(
        self,
        dashboard_dir: Optional[str | pathlib.Path] = None,
        validate_first: bool = True,
    ) -> dict[str, Any]:
        """Run the complete provisioning pipeline.

        Steps:
          1. Create folder hierarchy
          2. Provision data sources
          3. Provision contact points
          4. Provision notification policies
          5. Provision mute timings
          6. Deploy dashboards from directory (if provided)

        Args:
            dashboard_dir: Optional directory containing dashboard JSON files.
            validate_first: Whether to validate dashboards before deploying.

        Returns:
            Provisioning report dict with results for each step.
        """
        report: dict[str, Any] = {}

        # Step 1: Folders
        logger.info("Provisioning step 1/6: Folders")
        folders = await self._folder_manager.create_folder_hierarchy()
        report["folders"] = {uid: f.title for uid, f in folders.items()}

        # Step 2: Data sources
        logger.info("Provisioning step 2/6: Data sources")
        datasources = await self._datasource_manager.provision_datasources()
        report["datasources"] = {uid: ds.name for uid, ds in datasources.items()}

        # Step 3: Contact points
        logger.info("Provisioning step 3/6: Contact points")
        contact_points = await self._alert_manager.provision_contact_points()
        report["contact_points"] = [cp.name for cp in contact_points]

        # Step 4: Notification policies
        logger.info("Provisioning step 4/6: Notification policies")
        policy = await self._alert_manager.provision_notification_policies()
        report["notification_policy"] = {
            "receiver": policy.receiver,
            "routes": len(policy.routes),
        }

        # Step 5: Mute timings
        logger.info("Provisioning step 5/6: Mute timings")
        mute_timings = await self._alert_manager.provision_mute_timings()
        report["mute_timings"] = [mt.get("name", "") for mt in mute_timings]

        # Step 6: Dashboards
        if dashboard_dir:
            logger.info("Provisioning step 6/6: Dashboards from %s", dashboard_dir)
            dashboard_results = await self.provision_dashboards_from_directory(
                directory=dashboard_dir,
                validate_first=validate_first,
            )
            report["dashboards"] = dashboard_results
        else:
            report["dashboards"] = "skipped (no directory provided)"

        logger.info("Provisioning complete. Report: %s", json.dumps(report, default=str))
        return report

    async def provision_dashboards_from_directory(
        self,
        directory: str | pathlib.Path,
        folder_uid: str = "",
        validate_first: bool = True,
    ) -> dict[str, Any]:
        """Deploy all dashboard JSON files from a directory.

        Each JSON file is loaded, optionally validated, and deployed
        to Grafana via the API.

        Args:
            directory: Directory containing dashboard JSON files.
            folder_uid: Target folder UID (empty for General folder).
            validate_first: Whether to validate before deploying.

        Returns:
            Dict with 'deployed', 'skipped', and 'failed' counts and details.
        """
        path = pathlib.Path(directory)
        results: dict[str, Any] = {
            "deployed": [],
            "skipped": [],
            "failed": [],
        }

        for json_file in sorted(path.glob("**/*.json")):
            try:
                content = json_file.read_text(encoding="utf-8")
                data = json.loads(content)
                dashboard_data = data.get("dashboard", data)

                # Validate
                if validate_first:
                    validation = self._validator.validate(dashboard_data)
                    if not validation.is_valid:
                        results["skipped"].append({
                            "file": str(json_file),
                            "errors": validation.errors,
                        })
                        logger.warning(
                            "Dashboard skipped (validation failed): %s errors=%s",
                            json_file, validation.errors,
                        )
                        continue

                # Deploy
                dashboard = Dashboard.model_validate(dashboard_data)
                api_result = await self._client.create_dashboard(
                    dashboard=dashboard,
                    folder_uid=folder_uid or self._infer_folder_uid(json_file),
                    overwrite=True,
                    message=f"Provisioned from {json_file.name}",
                )
                results["deployed"].append({
                    "file": str(json_file),
                    "uid": api_result.get("uid", ""),
                    "title": dashboard.title,
                })

            except Exception as exc:
                results["failed"].append({
                    "file": str(json_file),
                    "error": str(exc),
                })
                logger.error(
                    "Dashboard deployment failed: %s error=%s",
                    json_file, exc, exc_info=True,
                )

        logger.info(
            "Dashboard deployment: %d deployed, %d skipped, %d failed",
            len(results["deployed"]),
            len(results["skipped"]),
            len(results["failed"]),
        )
        return results

    def validate_dashboard(self, dashboard: dict[str, Any]) -> ValidationResult:
        """Validate a single dashboard dict.

        Args:
            dashboard: Dashboard JSON dict.

        Returns:
            ValidationResult.
        """
        return self._validator.validate(dashboard)

    def validate_all_dashboards(
        self,
        directory: str | pathlib.Path,
    ) -> list[ValidationResult]:
        """Validate all dashboard JSON files in a directory.

        Args:
            directory: Directory path to scan.

        Returns:
            List of ValidationResult.
        """
        return self._validator.validate_directory(directory)

    async def export_dashboard(
        self,
        uid: str,
        output_path: str | pathlib.Path,
    ) -> pathlib.Path:
        """Export a dashboard from Grafana to a JSON file.

        Args:
            uid: Dashboard UID.
            output_path: Output file path.

        Returns:
            Path to the written file.
        """
        dashboard = await self._client.get_dashboard(uid)
        path = pathlib.Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        dashboard_dict = dashboard.model_dump(exclude_none=True)
        path.write_text(
            json.dumps(dashboard_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Dashboard exported: uid=%s path=%s", uid, path)
        return path

    async def diff_dashboard(
        self,
        uid: str,
        local_path: str | pathlib.Path,
    ) -> dict[str, Any]:
        """Compare a local dashboard file against the deployed version.

        Args:
            uid: Dashboard UID in Grafana.
            local_path: Path to local dashboard JSON file.

        Returns:
            Dict with 'identical' bool and 'differences' list.
        """
        # Fetch remote
        remote = await self._client.get_dashboard(uid)
        remote_dict = remote.model_dump(exclude_none=True)

        # Load local
        path = pathlib.Path(local_path)
        local_data = json.loads(path.read_text(encoding="utf-8"))
        local_dict = local_data.get("dashboard", local_data)

        # Simple key-level comparison
        differences: list[str] = []
        all_keys = set(remote_dict.keys()) | set(local_dict.keys())

        for key in sorted(all_keys):
            remote_val = remote_dict.get(key)
            local_val = local_dict.get(key)
            if remote_val != local_val:
                differences.append(key)

        return {
            "identical": len(differences) == 0,
            "differences": differences,
            "remote_title": remote.title,
            "local_title": local_dict.get("title", ""),
        }

    # -- Internal helpers ----------------------------------------------------

    def _infer_folder_uid(self, file_path: pathlib.Path) -> str:
        """Infer the target folder UID from the file path.

        Maps directory names to GreenLang folder UIDs based on naming
        conventions (e.g. 'infrastructure' -> 'gl-infrastructure').

        Args:
            file_path: Path to the dashboard JSON file.

        Returns:
            Inferred folder UID, or empty string for General.
        """
        parent_name = file_path.parent.name.lower()

        folder_mapping = {
            "executive": "gl-executive",
            "infrastructure": "gl-infrastructure",
            "data-stores": "gl-data-stores",
            "datastores": "gl-data-stores",
            "observability": "gl-observability",
            "security": "gl-security",
            "applications": "gl-applications",
            "alerts": "gl-alerts",
        }

        return folder_mapping.get(parent_name, "")
