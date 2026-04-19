"""
Regulatory Tracking Bridge - PACK-008 EU Taxonomy Alignment

This module monitors Delegated Act updates and manages criteria version
transitions for the EU Taxonomy. It tracks all DA versions with effective
dates, detects changes, and supports migration planning.

Tracked regulations:
- Climate Delegated Act (EU) 2021/2139 (CCM + CCA criteria)
- Environmental Delegated Act (EU) 2023/2486 (WTR, CE, PPC, BIO criteria)
- Complementary Climate DA (EU) 2022/1214 (Nuclear + Gas)
- Disclosures Delegated Act (EU) 2021/2178 (Article 8 templates)
- Omnibus Simplification Package 2025

Example:
    >>> config = RegulatoryTrackingConfig(check_interval_hours=24)
    >>> bridge = RegulatoryTrackingBridge(config)
    >>> updates = await bridge.check_for_updates()
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)


class RegulatoryTrackingConfig(BaseModel):
    """Configuration for Regulatory Tracking Bridge."""

    check_interval_hours: int = Field(
        default=24,
        ge=1,
        description="Interval between update checks in hours"
    )
    auto_migrate: bool = Field(
        default=False,
        description="Automatically migrate to new DA versions"
    )
    track_draft_acts: bool = Field(
        default=True,
        description="Track draft Delegated Acts not yet in force"
    )
    notification_channels: List[str] = Field(
        default=["dashboard", "email"],
        description="Channels for regulatory update notifications"
    )
    current_climate_da_version: str = Field(
        default="2023-06-27",
        description="Currently applied Climate DA version date"
    )
    current_environmental_da_version: str = Field(
        default="2023-11-21",
        description="Currently applied Environmental DA version date"
    )


# Delegated Act version registry
DA_VERSION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "climate_da_2021": {
        "regulation": "(EU) 2021/2139",
        "title": "Climate Delegated Act",
        "type": "climate",
        "objectives": ["CCM", "CCA"],
        "adoption_date": "2021-06-04",
        "effective_date": "2022-01-01",
        "status": "in_force",
        "activities_covered": 88,
        "oj_reference": "OJ L 442, 9.12.2021"
    },
    "climate_da_2022_amendment": {
        "regulation": "(EU) 2022/1214",
        "title": "Complementary Climate Delegated Act (Nuclear and Gas)",
        "type": "complementary",
        "objectives": ["CCM"],
        "adoption_date": "2022-03-09",
        "effective_date": "2023-01-01",
        "status": "in_force",
        "activities_covered": 4,
        "oj_reference": "OJ L 188, 15.7.2022"
    },
    "climate_da_2023_amendment": {
        "regulation": "(EU) 2023/2485",
        "title": "Climate DA Amendment 2023",
        "type": "climate_amendment",
        "objectives": ["CCM", "CCA"],
        "adoption_date": "2023-06-27",
        "effective_date": "2024-01-01",
        "status": "in_force",
        "activities_covered": 12,
        "oj_reference": "OJ L 2023/2485, 21.11.2023"
    },
    "environmental_da_2023": {
        "regulation": "(EU) 2023/2486",
        "title": "Environmental Delegated Act",
        "type": "environmental",
        "objectives": ["WTR", "CE", "PPC", "BIO"],
        "adoption_date": "2023-06-27",
        "effective_date": "2024-01-01",
        "status": "in_force",
        "activities_covered": 45,
        "oj_reference": "OJ L 2023/2486, 21.11.2023"
    },
    "disclosures_da_2021": {
        "regulation": "(EU) 2021/2178",
        "title": "Disclosures Delegated Act",
        "type": "disclosures",
        "objectives": [],
        "adoption_date": "2021-07-06",
        "effective_date": "2022-01-01",
        "status": "in_force",
        "activities_covered": 0,
        "oj_reference": "OJ L 443, 10.12.2021"
    },
    "disclosures_da_2023_amendment": {
        "regulation": "(EU) 2023/2772",
        "title": "Disclosures DA Amendment 2023",
        "type": "disclosures_amendment",
        "objectives": [],
        "adoption_date": "2023-10-17",
        "effective_date": "2024-01-01",
        "status": "in_force",
        "activities_covered": 0,
        "oj_reference": "OJ L 2023/2772, 22.12.2023"
    },
    "simplification_omnibus_2025": {
        "regulation": "COM(2025) Draft",
        "title": "Omnibus Simplification Package 2025",
        "type": "simplification",
        "objectives": ["CCM", "CCA", "WTR", "CE", "PPC", "BIO"],
        "adoption_date": "2025-02-26",
        "effective_date": "2026-01-01",
        "status": "draft",
        "activities_covered": 0,
        "oj_reference": "Pending"
    }
}


class RegulatoryTrackingBridge:
    """
    Bridge for monitoring EU Taxonomy Delegated Act updates and versions.

    Tracks all DA versions, detects changes, and supports migration planning
    for taxonomy criteria transitions.

    Example:
        >>> config = RegulatoryTrackingConfig()
        >>> bridge = RegulatoryTrackingBridge(config)
        >>> updates = await bridge.check_for_updates()
    """

    def __init__(self, config: RegulatoryTrackingConfig):
        """Initialize regulatory tracking bridge."""
        self.config = config
        self._service: Any = None
        self._last_check: Optional[datetime] = None
        logger.info("RegulatoryTrackingBridge initialized")

    def inject_service(self, service: Any) -> None:
        """Inject real regulatory monitoring service."""
        self._service = service
        logger.info("Injected regulatory tracking service")

    async def check_for_updates(self) -> Dict[str, Any]:
        """
        Check for Delegated Act updates since last check.

        Returns:
            Update status with any new or amended DAs
        """
        try:
            if self._service and hasattr(self._service, "check_for_updates"):
                return await self._service.check_for_updates()

            # Analyze registry for pending or recently effective DAs
            updates = []
            for da_id, da_info in DA_VERSION_REGISTRY.items():
                effective_date = da_info.get("effective_date", "")
                status = da_info.get("status", "")

                if status == "draft":
                    updates.append({
                        "da_id": da_id,
                        "regulation": da_info["regulation"],
                        "title": da_info["title"],
                        "status": "draft",
                        "effective_date": effective_date,
                        "action_required": "review"
                    })

            self._last_check = datetime.utcnow()

            return {
                "updates_found": len(updates),
                "updates": updates,
                "last_check": self._last_check.isoformat(),
                "current_versions": {
                    "climate_da": self.config.current_climate_da_version,
                    "environmental_da": self.config.current_environmental_da_version
                },
                "next_check_hours": self.config.check_interval_hours,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Update check failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def get_current_da(
        self,
        objective: str
    ) -> Dict[str, Any]:
        """
        Get the current active Delegated Act for an environmental objective.

        Args:
            objective: Environmental objective code (CCM, CCA, WTR, CE, PPC, BIO)

        Returns:
            Current active DA details for the objective
        """
        try:
            if self._service and hasattr(self._service, "get_current_da"):
                return await self._service.get_current_da(objective)

            objective_upper = objective.upper()
            matching_das = []

            for da_id, da_info in DA_VERSION_REGISTRY.items():
                if (objective_upper in da_info.get("objectives", []) and
                        da_info.get("status") == "in_force"):
                    matching_das.append({
                        "da_id": da_id,
                        "regulation": da_info["regulation"],
                        "title": da_info["title"],
                        "effective_date": da_info["effective_date"],
                        "activities_covered": da_info["activities_covered"],
                        "oj_reference": da_info["oj_reference"]
                    })

            # Sort by effective date (most recent first)
            matching_das.sort(key=lambda x: x["effective_date"], reverse=True)

            return {
                "objective": objective_upper,
                "current_da": matching_das[0] if matching_das else None,
                "all_applicable_das": matching_das,
                "total_applicable": len(matching_das),
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Current DA lookup failed: {str(e)}")
            return {"objective": objective, "error": str(e)}

    async def compare_versions(
        self,
        old_version: str,
        new_version: str
    ) -> Dict[str, Any]:
        """
        Compare two DA versions and identify changes.

        Args:
            old_version: Old DA version identifier
            new_version: New DA version identifier

        Returns:
            Version comparison with change summary
        """
        try:
            if self._service and hasattr(self._service, "compare_versions"):
                return await self._service.compare_versions(old_version, new_version)

            old_da = DA_VERSION_REGISTRY.get(old_version)
            new_da = DA_VERSION_REGISTRY.get(new_version)

            if not old_da:
                return {"error": f"Version {old_version} not found in registry"}
            if not new_da:
                return {"error": f"Version {new_version} not found in registry"}

            changes = {
                "old_version": {
                    "id": old_version,
                    "regulation": old_da["regulation"],
                    "effective_date": old_da["effective_date"],
                    "activities_covered": old_da["activities_covered"]
                },
                "new_version": {
                    "id": new_version,
                    "regulation": new_da["regulation"],
                    "effective_date": new_da["effective_date"],
                    "activities_covered": new_da["activities_covered"]
                },
                "activities_added": max(
                    0, new_da["activities_covered"] - old_da["activities_covered"]
                ),
                "objectives_changed": (
                    set(new_da.get("objectives", [])) !=
                    set(old_da.get("objectives", []))
                ),
                "type_change": old_da.get("type") != new_da.get("type"),
                "provenance_hash": self._calculate_hash({
                    "old": old_version, "new": new_version
                }),
                "timestamp": datetime.utcnow().isoformat()
            }

            return changes

        except Exception as e:
            logger.error(f"Version comparison failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def plan_migration(
        self,
        from_version: str,
        to_version: str
    ) -> Dict[str, Any]:
        """
        Plan migration from one DA version to another.

        Args:
            from_version: Current DA version identifier
            to_version: Target DA version identifier

        Returns:
            Migration plan with steps and impact assessment
        """
        try:
            if self._service and hasattr(self._service, "plan_migration"):
                return await self._service.plan_migration(from_version, to_version)

            # Compare versions
            comparison = await self.compare_versions(from_version, to_version)

            migration_plan = {
                "from_version": from_version,
                "to_version": to_version,
                "comparison": comparison,
                "migration_steps": [
                    {
                        "step": 1,
                        "action": "Review new TSC requirements",
                        "description": "Analyze changes in Technical Screening Criteria",
                        "estimated_effort_hours": 16
                    },
                    {
                        "step": 2,
                        "action": "Update activity mappings",
                        "description": "Map new or modified activities to company portfolio",
                        "estimated_effort_hours": 8
                    },
                    {
                        "step": 3,
                        "action": "Reassess alignment",
                        "description": "Re-evaluate SC, DNSH, and MS for affected activities",
                        "estimated_effort_hours": 40
                    },
                    {
                        "step": 4,
                        "action": "Update evidence",
                        "description": "Collect new evidence for changed criteria",
                        "estimated_effort_hours": 24
                    },
                    {
                        "step": 5,
                        "action": "Recalculate KPIs",
                        "description": "Recalculate Turnover/CapEx/OpEx ratios",
                        "estimated_effort_hours": 8
                    },
                    {
                        "step": 6,
                        "action": "Update disclosures",
                        "description": "Regenerate Article 8 / EBA Pillar 3 templates",
                        "estimated_effort_hours": 8
                    }
                ],
                "total_estimated_effort_hours": 104,
                "risk_level": "medium",
                "auto_migrate": self.config.auto_migrate,
                "provenance_hash": self._calculate_hash({
                    "from": from_version,
                    "to": to_version
                }),
                "timestamp": datetime.utcnow().isoformat()
            }

            return migration_plan

        except Exception as e:
            logger.error(f"Migration planning failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def get_da_registry(self) -> Dict[str, Dict[str, Any]]:
        """Return the complete DA version registry for inspection."""
        return DA_VERSION_REGISTRY.copy()

    def _calculate_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance."""
        import json
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
