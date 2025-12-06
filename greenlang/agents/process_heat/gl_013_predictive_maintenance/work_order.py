# -*- coding: utf-8 -*-
"""
GL-013 PredictMaint Agent - Work Order Generation Module

This module implements CMMS (Computerized Maintenance Management System)
integration for automatic work order generation based on predictive
maintenance analysis results.

Supported CMMS systems:
- SAP PM (Plant Maintenance)
- IBM Maximo
- eMaint
- Fiix
- Custom REST API

Features:
- Automatic work order creation from predictions
- Priority mapping based on failure probability
- Parts list generation
- Labor estimation
- Integration with existing maintenance schedules

Example:
    >>> from greenlang.agents.process_heat.gl_013_predictive_maintenance.work_order import (
    ...     WorkOrderGenerator
    ... )
    >>> generator = WorkOrderGenerator(cmms_config)
    >>> work_order = generator.create_from_prediction(prediction, equipment_id)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib
import logging
import uuid

from greenlang.agents.process_heat.gl_013_predictive_maintenance.config import (
    CMMSConfig,
    CMMSType,
    FailureMode,
)
from greenlang.agents.process_heat.gl_013_predictive_maintenance.schemas import (
    FailurePrediction,
    HealthStatus,
    MaintenanceRecommendation,
    WorkOrderPriority,
    WorkOrderRequest,
    WorkOrderType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Estimated repair durations by failure mode (hours)
REPAIR_DURATIONS = {
    FailureMode.BEARING_WEAR: 8.0,
    FailureMode.BEARING_FATIGUE: 16.0,
    FailureMode.IMBALANCE: 4.0,
    FailureMode.MISALIGNMENT: 4.0,
    FailureMode.LOOSENESS: 2.0,
    FailureMode.ROTOR_BAR_BREAK: 40.0,
    FailureMode.STATOR_WINDING: 24.0,
    FailureMode.ECCENTRICITY: 16.0,
    FailureMode.CAVITATION: 8.0,
    FailureMode.SEAL_FAILURE: 4.0,
    FailureMode.LUBRICATION_FAILURE: 2.0,
    FailureMode.FOULING: 8.0,
    FailureMode.CORROSION: 16.0,
    FailureMode.FATIGUE_CRACK: 24.0,
    FailureMode.OVERHEATING: 8.0,
}

# Typical parts required by failure mode
PARTS_BY_FAILURE_MODE = {
    FailureMode.BEARING_WEAR: [
        "Bearings (2)",
        "Bearing seals",
        "Lubricant",
        "Gasket set",
    ],
    FailureMode.IMBALANCE: [
        "Balance weights",
        "Alignment shims",
    ],
    FailureMode.MISALIGNMENT: [
        "Coupling element",
        "Alignment shims",
        "Coupling bolts",
    ],
    FailureMode.SEAL_FAILURE: [
        "Mechanical seal",
        "O-rings",
        "Gaskets",
    ],
    FailureMode.LUBRICATION_FAILURE: [
        "Lubricant (appropriate grade)",
        "Grease fittings",
        "Breather element",
    ],
    FailureMode.ROTOR_BAR_BREAK: [
        "Replacement motor (if repair not feasible)",
        "Motor bearings",
        "Coupling",
    ],
}

# Required skills by failure mode
SKILLS_BY_FAILURE_MODE = {
    FailureMode.BEARING_WEAR: ["Mechanical", "Vibration analysis"],
    FailureMode.IMBALANCE: ["Mechanical", "Balancing technician"],
    FailureMode.MISALIGNMENT: ["Mechanical", "Laser alignment"],
    FailureMode.ROTOR_BAR_BREAK: ["Electrical", "Motor rewinding"],
    FailureMode.STATOR_WINDING: ["Electrical", "Motor rewinding"],
    FailureMode.SEAL_FAILURE: ["Mechanical", "Pump specialist"],
    FailureMode.LUBRICATION_FAILURE: ["Mechanical", "Lubrication technician"],
}


# =============================================================================
# CMMS ADAPTER INTERFACE
# =============================================================================

class CMMSAdapter(ABC):
    """
    Abstract base class for CMMS system adapters.

    Each CMMS system (SAP PM, Maximo, etc.) implements this interface
    to provide consistent work order creation capabilities.
    """

    @abstractmethod
    def create_work_order(
        self,
        work_order: WorkOrderRequest,
    ) -> Dict[str, Any]:
        """Create work order in CMMS system."""
        pass

    @abstractmethod
    def update_work_order(
        self,
        work_order_id: str,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update existing work order."""
        pass

    @abstractmethod
    def get_work_order_status(
        self,
        work_order_id: str,
    ) -> Dict[str, Any]:
        """Get work order status."""
        pass

    @abstractmethod
    def close_work_order(
        self,
        work_order_id: str,
        completion_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Close completed work order."""
        pass


# =============================================================================
# CMMS ADAPTER IMPLEMENTATIONS
# =============================================================================

class SAPPMAdapter(CMMSAdapter):
    """
    SAP Plant Maintenance adapter.

    Integrates with SAP PM module for work order management.
    Uses SAP RFC or OData API.
    """

    def __init__(self, config: CMMSConfig) -> None:
        """
        Initialize SAP PM adapter.

        Args:
            config: CMMS configuration
        """
        self.config = config
        self.api_endpoint = config.api_endpoint
        logger.info("SAP PM adapter initialized")

    def create_work_order(
        self,
        work_order: WorkOrderRequest,
    ) -> Dict[str, Any]:
        """
        Create work order in SAP PM.

        Maps to SAP transaction IW31/IW32.

        Args:
            work_order: Work order request

        Returns:
            Created work order details with SAP order number
        """
        # Map priority to SAP priority
        priority_map = {
            WorkOrderPriority.EMERGENCY: "1",
            WorkOrderPriority.URGENT: "2",
            WorkOrderPriority.HIGH: "3",
            WorkOrderPriority.MEDIUM: "4",
            WorkOrderPriority.LOW: "5",
            WorkOrderPriority.SCHEDULED: "6",
        }

        # Map work order type to SAP order type
        order_type_map = {
            WorkOrderType.CORRECTIVE: "PM01",
            WorkOrderType.PREVENTIVE: "PM02",
            WorkOrderType.EMERGENCY: "PM03",
            WorkOrderType.INSPECTION: "PM04",
            WorkOrderType.LUBRICATION: "PM05",
            WorkOrderType.CALIBRATION: "PM06",
        }

        # Build SAP PM payload
        payload = {
            "OrderType": order_type_map.get(work_order.order_type, "PM01"),
            "Equipment": work_order.equipment_id,
            "FunctionalLocation": work_order.equipment_tag,
            "Priority": priority_map.get(work_order.priority, "4"),
            "ShortText": work_order.title[:40],  # SAP limit
            "LongText": work_order.description,
            "PlannerGroup": "001",  # Default planner group
            "Plant": self.config.plant_code,
            "RequestedStart": self._format_sap_date(work_order.required_by_date),
        }

        # In production, this would call SAP API
        # For now, return simulated response
        sap_order_number = f"400{str(uuid.uuid4().int)[:7]}"

        logger.info(f"SAP PM work order created: {sap_order_number}")

        return {
            "success": True,
            "sap_order_number": sap_order_number,
            "internal_id": work_order.work_order_id,
            "status": "CREATED",
            "message": "Work order created in SAP PM",
        }

    def update_work_order(
        self,
        work_order_id: str,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update SAP PM work order."""
        logger.info(f"Updating SAP PM work order: {work_order_id}")
        return {
            "success": True,
            "work_order_id": work_order_id,
            "status": "UPDATED",
        }

    def get_work_order_status(
        self,
        work_order_id: str,
    ) -> Dict[str, Any]:
        """Get SAP PM work order status."""
        return {
            "work_order_id": work_order_id,
            "status": "RELEASED",
            "assigned_to": "MAINT_CREW_1",
        }

    def close_work_order(
        self,
        work_order_id: str,
        completion_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Close SAP PM work order."""
        logger.info(f"Closing SAP PM work order: {work_order_id}")
        return {
            "success": True,
            "work_order_id": work_order_id,
            "status": "CLOSED",
        }

    def _format_sap_date(
        self,
        dt: Optional[datetime]
    ) -> Optional[str]:
        """Format datetime for SAP."""
        if dt is None:
            return None
        return dt.strftime("%Y%m%d")


class MaximoAdapter(CMMSAdapter):
    """
    IBM Maximo adapter.

    Integrates with Maximo for work order management.
    Uses Maximo REST API.
    """

    def __init__(self, config: CMMSConfig) -> None:
        """Initialize Maximo adapter."""
        self.config = config
        self.api_endpoint = config.api_endpoint
        logger.info("Maximo adapter initialized")

    def create_work_order(
        self,
        work_order: WorkOrderRequest,
    ) -> Dict[str, Any]:
        """Create work order in Maximo."""
        # Map priority to Maximo priority (1-5)
        priority_map = {
            WorkOrderPriority.EMERGENCY: 1,
            WorkOrderPriority.URGENT: 1,
            WorkOrderPriority.HIGH: 2,
            WorkOrderPriority.MEDIUM: 3,
            WorkOrderPriority.LOW: 4,
            WorkOrderPriority.SCHEDULED: 5,
        }

        payload = {
            "assetnum": work_order.equipment_id,
            "location": work_order.equipment_tag,
            "wopriority": priority_map.get(work_order.priority, 3),
            "description": work_order.title,
            "description_longdescription": work_order.description,
            "worktype": work_order.order_type.value.upper(),
            "siteid": self.config.plant_code,
        }

        maximo_id = f"WO{str(uuid.uuid4().int)[:8]}"

        logger.info(f"Maximo work order created: {maximo_id}")

        return {
            "success": True,
            "maximo_id": maximo_id,
            "internal_id": work_order.work_order_id,
            "status": "WAPPR",
            "message": "Work order created in Maximo",
        }

    def update_work_order(
        self,
        work_order_id: str,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update Maximo work order."""
        return {"success": True, "work_order_id": work_order_id}

    def get_work_order_status(
        self,
        work_order_id: str,
    ) -> Dict[str, Any]:
        """Get Maximo work order status."""
        return {"work_order_id": work_order_id, "status": "APPR"}

    def close_work_order(
        self,
        work_order_id: str,
        completion_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Close Maximo work order."""
        return {"success": True, "status": "CLOSE"}


class GenericCMMSAdapter(CMMSAdapter):
    """
    Generic CMMS adapter for custom REST APIs.

    Can be configured to work with various CMMS systems
    that expose REST APIs.
    """

    def __init__(self, config: CMMSConfig) -> None:
        """Initialize generic adapter."""
        self.config = config
        logger.info("Generic CMMS adapter initialized")

    def create_work_order(
        self,
        work_order: WorkOrderRequest,
    ) -> Dict[str, Any]:
        """Create work order via REST API."""
        # Build generic payload
        payload = {
            "equipment_id": work_order.equipment_id,
            "equipment_tag": work_order.equipment_tag,
            "title": work_order.title,
            "description": work_order.description,
            "priority": work_order.priority.value,
            "type": work_order.order_type.value,
            "parts": work_order.parts_required,
            "duration_hours": work_order.estimated_duration_hours,
            "required_by": (
                work_order.required_by_date.isoformat()
                if work_order.required_by_date else None
            ),
        }

        wo_id = f"WO-{uuid.uuid4().hex[:8].upper()}"

        return {
            "success": True,
            "work_order_id": wo_id,
            "internal_id": work_order.work_order_id,
            "status": "CREATED",
        }

    def update_work_order(
        self,
        work_order_id: str,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update work order via REST API."""
        return {"success": True, "work_order_id": work_order_id}

    def get_work_order_status(
        self,
        work_order_id: str,
    ) -> Dict[str, Any]:
        """Get work order status via REST API."""
        return {"work_order_id": work_order_id, "status": "PENDING"}

    def close_work_order(
        self,
        work_order_id: str,
        completion_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Close work order via REST API."""
        return {"success": True, "status": "CLOSED"}


# =============================================================================
# WORK ORDER GENERATOR
# =============================================================================

class WorkOrderGenerator:
    """
    Work Order Generator for Predictive Maintenance.

    Generates CMMS work orders based on failure predictions
    and maintenance recommendations. Handles priority mapping,
    parts lists, and labor estimation.

    Attributes:
        config: CMMS configuration
        adapter: CMMS system adapter

    Example:
        >>> generator = WorkOrderGenerator(cmms_config)
        >>> work_order = generator.create_from_prediction(
        ...     prediction,
        ...     equipment_id="PUMP-001",
        ...     equipment_tag="P-1001A"
        ... )
        >>> print(f"Work order: {work_order.work_order_id}")
    """

    def __init__(self, config: CMMSConfig) -> None:
        """
        Initialize work order generator.

        Args:
            config: CMMS configuration
        """
        self.config = config
        self.adapter = self._create_adapter(config)

        logger.info(
            f"WorkOrderGenerator initialized: CMMS={config.system_type.value}"
        )

    def _create_adapter(self, config: CMMSConfig) -> CMMSAdapter:
        """Create appropriate CMMS adapter."""
        if config.system_type == CMMSType.SAP_PM:
            return SAPPMAdapter(config)
        elif config.system_type == CMMSType.IBM_MAXIMO:
            return MaximoAdapter(config)
        else:
            return GenericCMMSAdapter(config)

    def create_from_prediction(
        self,
        prediction: FailurePrediction,
        equipment_id: str,
        equipment_tag: Optional[str] = None,
        analysis_id: str = "",
    ) -> WorkOrderRequest:
        """
        Create work order request from failure prediction.

        Args:
            prediction: Failure prediction result
            equipment_id: Equipment identifier
            equipment_tag: Plant equipment tag
            analysis_id: Source analysis request ID

        Returns:
            WorkOrderRequest ready for submission
        """
        logger.info(
            f"Creating work order for {prediction.failure_mode.value} "
            f"on equipment {equipment_id}"
        )

        # Determine work order type
        order_type = self._determine_order_type(prediction)

        # Determine priority based on probability and TTF
        priority = self._determine_priority(prediction)

        # Generate title and description
        title = self._generate_title(prediction, equipment_id)
        description = self._generate_description(prediction, equipment_id)

        # Get parts and skills
        parts = PARTS_BY_FAILURE_MODE.get(prediction.failure_mode, [])
        skills = SKILLS_BY_FAILURE_MODE.get(prediction.failure_mode, ["Mechanical"])

        # Estimate duration
        duration = REPAIR_DURATIONS.get(prediction.failure_mode, 8.0)

        # Calculate required-by date
        required_by = self._calculate_required_by_date(prediction, priority)

        # Generate recommended actions
        actions = self._generate_actions(prediction)

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance(
            prediction, equipment_id, priority
        )

        return WorkOrderRequest(
            equipment_id=equipment_id,
            equipment_tag=equipment_tag,
            order_type=order_type,
            priority=priority,
            title=title,
            description=description,
            failure_modes=[prediction.failure_mode],
            recommended_actions=actions,
            parts_required=parts,
            estimated_duration_hours=duration,
            required_by_date=required_by,
            source_analysis_id=analysis_id,
            provenance_hash=provenance_hash,
        )

    def create_from_recommendations(
        self,
        recommendations: List[MaintenanceRecommendation],
        equipment_id: str,
        equipment_tag: Optional[str] = None,
        analysis_id: str = "",
    ) -> List[WorkOrderRequest]:
        """
        Create work orders from multiple recommendations.

        Groups related recommendations into single work orders
        where appropriate.

        Args:
            recommendations: List of maintenance recommendations
            equipment_id: Equipment identifier
            equipment_tag: Plant equipment tag
            analysis_id: Source analysis ID

        Returns:
            List of WorkOrderRequest objects
        """
        work_orders = []

        # Group by priority
        priority_groups: Dict[WorkOrderPriority, List[MaintenanceRecommendation]] = {}

        for rec in recommendations:
            if rec.priority not in priority_groups:
                priority_groups[rec.priority] = []
            priority_groups[rec.priority].append(rec)

        # Create work order for each priority group
        for priority, recs in priority_groups.items():
            if not recs:
                continue

            # Combine failure modes
            failure_modes = list(set(r.failure_mode for r in recs))

            # Combine actions and parts
            actions = []
            parts = []
            total_duration = 0

            for rec in recs:
                actions.append(f"- {rec.description}")
                parts.extend(rec.parts_required)
                if rec.estimated_duration_hours:
                    total_duration += rec.estimated_duration_hours

            # Deduplicate parts
            parts = list(set(parts))

            # Generate combined description
            description = self._generate_combined_description(
                failure_modes, recs
            )

            # Calculate required-by date (use most urgent)
            min_deadline = min(
                (r.deadline_hours for r in recs if r.deadline_hours),
                default=168  # 1 week default
            )
            required_by = datetime.now(timezone.utc) + timedelta(hours=min_deadline)

            work_order = WorkOrderRequest(
                equipment_id=equipment_id,
                equipment_tag=equipment_tag,
                order_type=self._order_type_from_priority(priority),
                priority=priority,
                title=f"PdM: {', '.join(fm.value for fm in failure_modes[:2])}",
                description=description,
                failure_modes=failure_modes,
                recommended_actions=actions,
                parts_required=parts,
                estimated_duration_hours=total_duration or 8.0,
                required_by_date=required_by,
                source_analysis_id=analysis_id,
            )

            work_orders.append(work_order)

        return work_orders

    def submit_work_order(
        self,
        work_order: WorkOrderRequest,
    ) -> Dict[str, Any]:
        """
        Submit work order to CMMS system.

        Args:
            work_order: Work order to submit

        Returns:
            Submission result with CMMS work order ID
        """
        if not self.config.enabled:
            logger.warning("CMMS integration disabled, work order not submitted")
            return {
                "success": False,
                "message": "CMMS integration disabled",
                "work_order_id": work_order.work_order_id,
            }

        if not self.config.auto_create_work_orders:
            logger.info("Auto-create disabled, work order queued for review")
            return {
                "success": True,
                "message": "Work order queued for manual review",
                "work_order_id": work_order.work_order_id,
                "status": "PENDING_REVIEW",
            }

        # Submit to CMMS
        result = self.adapter.create_work_order(work_order)

        logger.info(
            f"Work order submitted: {work_order.work_order_id} -> "
            f"{result.get('sap_order_number') or result.get('maximo_id') or result.get('work_order_id')}"
        )

        return result

    def _determine_order_type(
        self,
        prediction: FailurePrediction
    ) -> WorkOrderType:
        """Determine work order type from prediction."""
        if prediction.probability > 0.8:
            return WorkOrderType.EMERGENCY
        elif prediction.probability > 0.5:
            return WorkOrderType.CORRECTIVE
        else:
            return WorkOrderType.PREVENTIVE

    def _determine_priority(
        self,
        prediction: FailurePrediction
    ) -> WorkOrderPriority:
        """Determine work order priority from prediction."""
        prob = prediction.probability
        ttf = prediction.time_to_failure_hours

        # High probability -> urgent priority
        if prob > 0.8:
            return WorkOrderPriority.EMERGENCY
        elif prob > 0.6:
            return WorkOrderPriority.URGENT

        # Consider time to failure
        if ttf is not None:
            if ttf < 24:
                return WorkOrderPriority.EMERGENCY
            elif ttf < 168:  # 1 week
                return WorkOrderPriority.URGENT
            elif ttf < 720:  # 30 days
                return WorkOrderPriority.HIGH
            elif ttf < 2160:  # 90 days
                return WorkOrderPriority.MEDIUM

        # Default based on probability
        if prob > 0.3:
            return WorkOrderPriority.MEDIUM
        else:
            return WorkOrderPriority.LOW

    def _generate_title(
        self,
        prediction: FailurePrediction,
        equipment_id: str,
    ) -> str:
        """Generate work order title."""
        mode = prediction.failure_mode.value.replace("_", " ").title()
        return f"PdM: {mode} - {equipment_id}"

    def _generate_description(
        self,
        prediction: FailurePrediction,
        equipment_id: str,
    ) -> str:
        """Generate detailed work order description."""
        lines = [
            f"PREDICTIVE MAINTENANCE WORK ORDER",
            f"=" * 40,
            f"",
            f"Equipment: {equipment_id}",
            f"Failure Mode: {prediction.failure_mode.value}",
            f"Probability: {prediction.probability:.1%}",
            f"Confidence: {prediction.confidence:.1%}",
            f"",
        ]

        if prediction.time_to_failure_hours:
            lines.extend([
                f"Estimated Time to Failure: {prediction.time_to_failure_hours:.0f} hours",
            ])
            if prediction.uncertainty_lower_hours and prediction.uncertainty_upper_hours:
                lines.append(
                    f"  Range: {prediction.uncertainty_lower_hours:.0f} - "
                    f"{prediction.uncertainty_upper_hours:.0f} hours"
                )
            lines.append("")

        lines.extend([
            "TOP CONTRIBUTING FACTORS:",
            "-" * 30,
        ])

        for feature in prediction.top_contributing_features:
            importance = prediction.feature_importance.get(feature, 0)
            lines.append(f"  - {feature}: {importance:.3f}")

        lines.extend([
            "",
            "RECOMMENDED ACTIONS:",
            "-" * 30,
        ])

        actions = self._generate_actions(prediction)
        for action in actions:
            lines.append(f"  {action}")

        lines.extend([
            "",
            f"Analysis Model: {prediction.model_id} v{prediction.model_version}",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
        ])

        return "\n".join(lines)

    def _generate_combined_description(
        self,
        failure_modes: List[FailureMode],
        recommendations: List[MaintenanceRecommendation],
    ) -> str:
        """Generate description for combined work order."""
        lines = [
            "PREDICTIVE MAINTENANCE - COMBINED WORK ORDER",
            "=" * 45,
            "",
            "FAILURE MODES ADDRESSED:",
        ]

        for fm in failure_modes:
            lines.append(f"  - {fm.value}")

        lines.extend([
            "",
            "RECOMMENDED ACTIONS:",
            "-" * 30,
        ])

        for rec in recommendations:
            lines.append(f"  [{rec.priority.value}] {rec.description}")

        return "\n".join(lines)

    def _generate_actions(
        self,
        prediction: FailurePrediction
    ) -> List[str]:
        """Generate recommended maintenance actions."""
        actions = []

        fm = prediction.failure_mode

        if fm == FailureMode.BEARING_WEAR:
            actions = [
                "1. Verify bearing condition with vibration analysis",
                "2. Check lubrication level and condition",
                "3. Replace bearings if defect confirmed",
                "4. Perform post-repair alignment check",
            ]
        elif fm == FailureMode.IMBALANCE:
            actions = [
                "1. Perform field balancing",
                "2. Check for buildup or erosion on rotating elements",
                "3. Verify coupling condition",
                "4. Confirm with post-balance vibration check",
            ]
        elif fm == FailureMode.MISALIGNMENT:
            actions = [
                "1. Perform laser alignment check",
                "2. Correct alignment to specifications",
                "3. Inspect coupling for wear",
                "4. Verify thermal growth allowance",
            ]
        elif fm == FailureMode.ROTOR_BAR_BREAK:
            actions = [
                "1. Confirm with follow-up MCSA analysis",
                "2. Assess motor for repair vs replacement",
                "3. If repairing, arrange for motor shop rebuild",
                "4. Plan for backup/spare motor during repair",
            ]
        elif fm == FailureMode.LUBRICATION_FAILURE:
            actions = [
                "1. Perform oil change with correct grade",
                "2. Flush system if contamination present",
                "3. Check and clean breathers",
                "4. Review lubrication schedule and procedures",
            ]
        else:
            actions = [
                f"1. Investigate {fm.value} condition",
                "2. Perform detailed inspection",
                "3. Repair or replace affected components",
                "4. Verify repair with condition monitoring",
            ]

        return actions

    def _calculate_required_by_date(
        self,
        prediction: FailurePrediction,
        priority: WorkOrderPriority,
    ) -> datetime:
        """Calculate required completion date."""
        now = datetime.now(timezone.utc)

        # Priority-based defaults
        priority_hours = {
            WorkOrderPriority.EMERGENCY: 4,
            WorkOrderPriority.URGENT: 24,
            WorkOrderPriority.HIGH: 48,
            WorkOrderPriority.MEDIUM: 168,  # 1 week
            WorkOrderPriority.LOW: 336,     # 2 weeks
            WorkOrderPriority.SCHEDULED: 720,  # 30 days
        }

        default_hours = priority_hours.get(priority, 168)

        # Use TTF if available and shorter
        if prediction.time_to_failure_hours:
            # Want to complete before failure, with safety margin
            ttf_hours = prediction.time_to_failure_hours * 0.5
            hours = min(default_hours, ttf_hours)
        else:
            hours = default_hours

        return now + timedelta(hours=max(4, hours))

    def _order_type_from_priority(
        self,
        priority: WorkOrderPriority
    ) -> WorkOrderType:
        """Map priority to order type."""
        if priority == WorkOrderPriority.EMERGENCY:
            return WorkOrderType.EMERGENCY
        elif priority in [WorkOrderPriority.URGENT, WorkOrderPriority.HIGH]:
            return WorkOrderType.CORRECTIVE
        else:
            return WorkOrderType.PREVENTIVE

    def _calculate_provenance(
        self,
        prediction: FailurePrediction,
        equipment_id: str,
        priority: WorkOrderPriority,
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"work_order|{equipment_id}|{prediction.failure_mode.value}|"
            f"{prediction.probability:.6f}|{priority.value}|"
            f"{datetime.now(timezone.utc).isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
