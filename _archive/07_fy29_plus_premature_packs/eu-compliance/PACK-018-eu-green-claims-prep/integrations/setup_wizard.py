# -*- coding: utf-8 -*-
"""
GreenClaimsSetupWizard - Configuration Wizard for PACK-018
=============================================================

Interactive setup wizard that guides users through initial configuration
of the EU Green Claims Prep Pack. Collects entity profile, claim inventory,
evidence source connections, label registry, sector configuration,
verification setup, notification preferences, and validates the complete
configuration before activation.

Wizard Steps (8 total):
    1. ENTITY_PROFILE      -- Organization name, jurisdiction, NACE codes
    2. CLAIM_INVENTORY     -- Register existing environmental marketing claims
    3. EVIDENCE_SOURCES    -- Configure data sources for claim substantiation
    4. LABEL_REGISTRY      -- Register sustainability labels in use
    5. SECTOR_CONFIG       -- Sector-specific compliance requirements
    6. VERIFICATION_SETUP  -- Configure third-party verification preferences
    7. NOTIFICATION_CONFIG -- Set up alert and notification channels
    8. VALIDATION          -- Validate and finalize configuration

Sector Presets:
    - manufacturing: Product-centric claims, PEF emphasis
    - financial: Investment product claims, SFDR overlap
    - energy: Renewable energy claims, emission intensity focus
    - retail: Consumer-facing claims, ECGT focus
    - food_beverage: Organic, natural, sustainable sourcing claims
    - technology: Digital sustainability, energy efficiency claims
    - construction: Building material, circularity claims

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-018 EU Green Claims Prep Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

__all__ = [
    "WizardStep",
    "StepStatus",
    "SectorPreset",
    "WizardConfig",
    "StepResult",
    "WizardState",
    "GreenClaimsSetupWizard",
]

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class WizardStep(str, Enum):
    """The 8 steps of the setup wizard."""

    ENTITY_PROFILE = "entity_profile"
    CLAIM_INVENTORY = "claim_inventory"
    EVIDENCE_SOURCES = "evidence_sources"
    LABEL_REGISTRY = "label_registry"
    SECTOR_CONFIG = "sector_config"
    VERIFICATION_SETUP = "verification_setup"
    NOTIFICATION_CONFIG = "notification_config"
    VALIDATION = "validation"

class StepStatus(str, Enum):
    """Status of a single wizard step."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class SectorPreset(str, Enum):
    """Available sector presets for configuration."""

    MANUFACTURING = "manufacturing"
    FINANCIAL = "financial"
    ENERGY = "energy"
    RETAIL = "retail"
    FOOD_BEVERAGE = "food_beverage"
    TECHNOLOGY = "technology"
    CONSTRUCTION = "construction"
    MULTI_SECTOR = "multi_sector"

# ---------------------------------------------------------------------------
# Wizard Step Order
# ---------------------------------------------------------------------------

WIZARD_STEP_ORDER: List[WizardStep] = [
    WizardStep.ENTITY_PROFILE,
    WizardStep.CLAIM_INVENTORY,
    WizardStep.EVIDENCE_SOURCES,
    WizardStep.LABEL_REGISTRY,
    WizardStep.SECTOR_CONFIG,
    WizardStep.VERIFICATION_SETUP,
    WizardStep.NOTIFICATION_CONFIG,
    WizardStep.VALIDATION,
]

STEP_DESCRIPTIONS: Dict[WizardStep, str] = {
    WizardStep.ENTITY_PROFILE: "Set up your organization profile, jurisdiction, and NACE codes",
    WizardStep.CLAIM_INVENTORY: "Register your existing environmental marketing claims",
    WizardStep.EVIDENCE_SOURCES: "Configure data sources for claim substantiation",
    WizardStep.LABEL_REGISTRY: "Register sustainability labels currently in use",
    WizardStep.SECTOR_CONFIG: "Apply sector-specific compliance requirements",
    WizardStep.VERIFICATION_SETUP: "Configure third-party verification and auditor preferences",
    WizardStep.NOTIFICATION_CONFIG: "Set up alerts for compliance deadlines and status changes",
    WizardStep.VALIDATION: "Validate and finalize the complete configuration",
}

SECTOR_NACE_MAP: Dict[str, SectorPreset] = {
    "A": SectorPreset.FOOD_BEVERAGE,
    "B": SectorPreset.MANUFACTURING,
    "C": SectorPreset.MANUFACTURING,
    "D": SectorPreset.ENERGY,
    "E": SectorPreset.MANUFACTURING,
    "F": SectorPreset.CONSTRUCTION,
    "G": SectorPreset.RETAIL,
    "H": SectorPreset.MANUFACTURING,
    "I": SectorPreset.FOOD_BEVERAGE,
    "J": SectorPreset.TECHNOLOGY,
    "K": SectorPreset.FINANCIAL,
    "L": SectorPreset.CONSTRUCTION,
    "M": SectorPreset.TECHNOLOGY,
    "N": SectorPreset.MULTI_SECTOR,
}

SECTOR_CLAIM_FOCUS: Dict[SectorPreset, List[str]] = {
    SectorPreset.MANUFACTURING: [
        "recyclable", "low_carbon", "circular", "clean_manufacturing",
        "zero_waste", "recycled_content", "durable",
    ],
    SectorPreset.FINANCIAL: [
        "sustainable", "green_investment", "esg_integrated",
        "climate_positive", "impact_investing",
    ],
    SectorPreset.ENERGY: [
        "renewable_energy", "clean_energy", "carbon_neutral",
        "net_zero", "green_power",
    ],
    SectorPreset.RETAIL: [
        "eco_friendly", "sustainable", "recyclable",
        "ethically_sourced", "organic",
    ],
    SectorPreset.FOOD_BEVERAGE: [
        "organic", "sustainable_farming", "non_toxic",
        "biodiversity_positive", "deforestation_free",
    ],
    SectorPreset.TECHNOLOGY: [
        "energy_efficient", "carbon_neutral", "e_waste_free",
        "repairable", "durable",
    ],
    SectorPreset.CONSTRUCTION: [
        "green_building", "energy_efficient", "circular",
        "low_carbon", "recycled_content",
    ],
    SectorPreset.MULTI_SECTOR: [
        "sustainable", "carbon_neutral", "recyclable",
        "circular", "net_zero",
    ],
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class WizardConfig(BaseModel):
    """Configuration for the setup wizard."""

    pack_id: str = Field(default="PACK-018")
    enable_sector_detection: bool = Field(default=True)
    allow_skip_steps: bool = Field(default=False)
    enable_provenance: bool = Field(default=True)

class StepResult(BaseModel):
    """Result of a single wizard step execution."""

    step: WizardStep = Field(...)
    status: StepStatus = Field(default=StepStatus.NOT_STARTED)
    data_collected: Dict[str, Any] = Field(default_factory=dict)
    validation_errors: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)
    completed_at: Optional[datetime] = Field(None)
    provenance_hash: str = Field(default="")

class WizardState(BaseModel):
    """Tracks overall wizard progress."""

    wizard_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-018")
    current_step: WizardStep = Field(default=WizardStep.ENTITY_PROFILE)
    current_step_index: int = Field(default=0)
    total_steps: int = Field(default=8)
    steps_completed: List[str] = Field(default_factory=list)
    steps_skipped: List[str] = Field(default_factory=list)
    step_results: Dict[str, StepResult] = Field(default_factory=dict)
    is_complete: bool = Field(default=False)
    entity_name: str = Field(default="")
    sector: str = Field(default="")
    started_at: datetime = Field(default_factory=utcnow)
    completed_at: Optional[datetime] = Field(None)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# GreenClaimsSetupWizard
# ---------------------------------------------------------------------------

class GreenClaimsSetupWizard:
    """8-step configuration wizard for PACK-018.

    Guides users through the initial setup of the EU Green Claims Prep
    Pack, collecting organization profile, claim inventory, evidence
    sources, labels, sector configuration, verification preferences,
    notifications, and validating the complete configuration.

    Attributes:
        config: Wizard configuration.
        state: Current wizard state.

    Example:
        >>> wizard = GreenClaimsSetupWizard()
        >>> result = wizard.run_step(WizardStep.ENTITY_PROFILE, {"name": "Acme"})
        >>> assert result["status"] == "completed"
        >>> next_step = wizard.advance()
        >>> assert next_step["next_step"] == "claim_inventory"
    """

    def __init__(self, config: Optional[WizardConfig] = None) -> None:
        """Initialize GreenClaimsSetupWizard.

        Args:
            config: Wizard configuration. Defaults used if None.
        """
        self.config = config or WizardConfig()
        self.state = WizardState(pack_id=self.config.pack_id)
        logger.info(
            "GreenClaimsSetupWizard initialized (pack=%s, steps=%d)",
            self.config.pack_id,
            self.state.total_steps,
        )

    def run_step(
        self,
        step: WizardStep,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a single wizard step with input data.

        Args:
            step: The wizard step to execute.
            input_data: User-provided data for this step.

        Returns:
            Dict with step status, collected data, validation results,
            and provenance hash.
        """
        data = input_data or {}
        handler = self._get_step_handler(step)
        step_result = handler(data)

        self.state.step_results[step.value] = step_result
        if step_result.status == StepStatus.COMPLETED:
            if step.value not in self.state.steps_completed:
                self.state.steps_completed.append(step.value)

        if self.config.enable_provenance:
            step_result.provenance_hash = _compute_hash(step_result)

        logger.info(
            "Wizard step '%s': %s (errors=%d, warnings=%d)",
            step.value,
            step_result.status.value,
            len(step_result.validation_errors),
            len(step_result.validation_warnings),
        )

        return step_result.model_dump(mode="json")

    def advance(self) -> Dict[str, Any]:
        """Advance to the next wizard step.

        Returns:
            Dict with next_step, step_index, progress_pct, and
            description of the next step.
        """
        current_idx = self.state.current_step_index
        next_idx = current_idx + 1

        if next_idx >= len(WIZARD_STEP_ORDER):
            self.state.is_complete = True
            self.state.completed_at = utcnow()
            if self.config.enable_provenance:
                self.state.provenance_hash = _compute_hash(self.state)
            return {
                "wizard_complete": True,
                "steps_completed": len(self.state.steps_completed),
                "total_steps": self.state.total_steps,
                "provenance_hash": self.state.provenance_hash,
            }

        next_step = WIZARD_STEP_ORDER[next_idx]
        self.state.current_step = next_step
        self.state.current_step_index = next_idx

        return {
            "wizard_complete": False,
            "next_step": next_step.value,
            "step_index": next_idx,
            "progress_pct": round(next_idx / self.state.total_steps * 100, 1),
            "description": STEP_DESCRIPTIONS.get(next_step, ""),
            "steps_completed": len(self.state.steps_completed),
            "total_steps": self.state.total_steps,
        }

    def get_state(self) -> Dict[str, Any]:
        """Get current wizard state.

        Returns:
            Dict with complete wizard state.
        """
        return self.state.model_dump(mode="json")

    def get_step_info(self, step: WizardStep) -> Dict[str, Any]:
        """Get information about a specific wizard step.

        Args:
            step: The wizard step to get info for.

        Returns:
            Dict with step name, description, index, and required fields.
        """
        idx = WIZARD_STEP_ORDER.index(step) if step in WIZARD_STEP_ORDER else -1
        return {
            "step": step.value,
            "description": STEP_DESCRIPTIONS.get(step, ""),
            "index": idx,
            "required_fields": self._get_required_fields(step),
        }

    def run_setup(
        self,
        entity_name: str = "",
        nace_code: str = "",
        jurisdiction: str = "EU",
    ) -> Dict[str, Any]:
        """Execute the full 7-step setup wizard in non-interactive mode.

        Runs all steps sequentially with the provided parameters and
        returns the complete configuration result.

        Args:
            entity_name: Organization name.
            nace_code: NACE activity code for sector detection.
            jurisdiction: Regulatory jurisdiction.

        Returns:
            Dict with wizard state, step results, and provenance hash.
        """
        start = utcnow()

        # Step 1: Entity Profile
        self.run_step(WizardStep.ENTITY_PROFILE, {
            "name": entity_name,
            "jurisdiction": jurisdiction,
            "nace_codes": [nace_code] if nace_code else [],
        })
        self.advance()

        # Step 2: Claim Inventory
        self.run_step(WizardStep.CLAIM_INVENTORY, {"claims": []})
        self.advance()

        # Step 3: Evidence Sources
        self.run_step(WizardStep.EVIDENCE_SOURCES, {"sources": []})
        self.advance()

        # Step 4: Label Registry
        self.run_step(WizardStep.LABEL_REGISTRY, {"labels": []})
        self.advance()

        # Step 5: Sector Config
        self.run_step(WizardStep.SECTOR_CONFIG, {"sector": self.state.sector})
        self.advance()

        # Step 6: Verification Setup
        self.run_step(WizardStep.VERIFICATION_SETUP, {})
        self.advance()

        # Step 7: Notification Config
        self.run_step(WizardStep.NOTIFICATION_CONFIG, {})
        self.advance()

        # Step 8: Validation
        self.run_step(WizardStep.VALIDATION, {})

        self.state.completed_at = utcnow()
        if self.config.enable_provenance:
            self.state.provenance_hash = _compute_hash(self.state)

        elapsed = (self.state.completed_at - start).total_seconds() * 1000
        logger.info(
            "SetupWizard run_setup complete: %d/%d steps in %.1fms",
            len(self.state.steps_completed),
            self.state.total_steps,
            elapsed,
        )

        return self.state.model_dump(mode="json")

    def select_sector(self, nace_code: str) -> Dict[str, Any]:
        """Select business sector from NACE code.

        Detects the appropriate sector preset from the NACE
        classification code and returns sector-specific metadata.

        Args:
            nace_code: NACE activity code (e.g., "C20.11").

        Returns:
            Dict with detected sector, claim focus areas, and hash.
        """
        return self.detect_sector(nace_code)

    def define_claim_scope(self, scope: str = "both") -> Dict[str, Any]:
        """Define the scope of environmental claims to process.

        Per Green Claims Directive Art. 2, claims can be product-level
        or organisation-level.

        Args:
            scope: Claim scope (product/corporate/both).

        Returns:
            Dict with scope configuration.
        """
        valid_scopes = {"product", "corporate", "both"}
        resolved = scope.lower() if scope.lower() in valid_scopes else "both"
        result = {
            "scope": resolved,
            "product_claims": resolved in ("product", "both"),
            "corporate_claims": resolved in ("corporate", "both"),
            "provenance_hash": _compute_hash({"scope": resolved}),
        }
        logger.info("SetupWizard claim scope defined: %s", resolved)
        return result

    def register_channels(self, channels: List[str]) -> Dict[str, Any]:
        """Register communication channels where claims are published.

        Per ECGD Art. 2(1), environmental claims include messages
        across any communication channel.

        Args:
            channels: List of channel names.

        Returns:
            Dict with registered channels and count.
        """
        result = {
            "channels": channels,
            "total_channels": len(channels),
            "provenance_hash": _compute_hash({"channels": channels}),
        }
        logger.info("SetupWizard channels registered: %d", len(channels))
        return result

    def inventory_labels(self, labels: List[str]) -> Dict[str, Any]:
        """Inventory environmental labels currently in use.

        Per ECGT Art. 10a/10b, sustainability labels must be based
        on recognized certification schemes.

        Args:
            labels: List of label identifiers.

        Returns:
            Dict with label inventory and verification status.
        """
        result = {
            "labels": labels,
            "total_labels": len(labels),
            "provenance_hash": _compute_hash({"labels": labels}),
        }
        logger.info("SetupWizard labels inventoried: %d", len(labels))
        return result

    def configure_pef_database(self, database: str = "ef_3.1") -> Dict[str, Any]:
        """Configure PEF background database for lifecycle assessment.

        Args:
            database: PEF database identifier.

        Returns:
            Dict with PEF database configuration.
        """
        result = {
            "database": database,
            "full_lca_required": True,
            "provenance_hash": _compute_hash({"database": database}),
        }
        logger.info("SetupWizard PEF database configured: %s", database)
        return result

    def setup_evidence_repository(self, retention_years: int = 5) -> Dict[str, Any]:
        """Configure evidence repository storage and retention.

        Per Green Claims Directive Art. 3(6), substantiation evidence
        must be kept up to date and available for inspection.

        Args:
            retention_years: Number of years to retain evidence.

        Returns:
            Dict with evidence repository configuration.
        """
        result = {
            "storage_type": "s3",
            "retention_years": retention_years,
            "sha256_integrity": True,
            "auto_lineage": True,
            "provenance_hash": _compute_hash({"retention": retention_years}),
        }
        logger.info("SetupWizard evidence repo configured: %d years", retention_years)
        return result

    def select_cab_preferences(self, format_str: str = "PDF") -> Dict[str, Any]:
        """Select Conformity Assessment Body submission preferences.

        Per Green Claims Directive Art. 10, claims must be verified
        by an accredited CAB before being made public.

        Args:
            format_str: CAB submission format (PDF/JSON/XBRL/HTML).

        Returns:
            Dict with CAB submission preferences.
        """
        valid_formats = {"PDF", "JSON", "XBRL", "HTML"}
        fmt = format_str.upper() if format_str.upper() in valid_formats else "PDF"
        result = {
            "submission_format": fmt,
            "auto_generate_dossier": True,
            "language": "en",
            "provenance_hash": _compute_hash({"format": fmt}),
        }
        logger.info("SetupWizard CAB preferences set: %s", fmt)
        return result

    def generate_config(self) -> Dict[str, Any]:
        """Generate a complete pack configuration from wizard state.

        Assembles all step outputs into a single configuration
        dictionary suitable for serialization to pack_config.yaml.

        Returns:
            Dict with complete pack configuration and hash.
        """
        config: Dict[str, Any] = {
            "pack_id": self.config.pack_id,
            "version": _MODULE_VERSION,
            "entity_name": self.state.entity_name,
            "sector": self.state.sector,
            "steps_completed": self.state.steps_completed,
            "step_data": {
                step_name: step_result.data_collected
                for step_name, step_result in self.state.step_results.items()
            },
            "generated_at": str(utcnow()),
        }
        config["config_hash"] = _compute_hash(config)
        logger.info("SetupWizard configuration generated (hash=%s)", config["config_hash"][:12])
        return config

    def detect_sector(self, nace_code: str) -> Dict[str, Any]:
        """Detect sector from NACE code.

        Args:
            nace_code: NACE classification code (e.g., "C20.11").

        Returns:
            Dict with detected sector, preset, and claim focus areas.
        """
        section = nace_code[0].upper() if nace_code else ""
        preset = SECTOR_NACE_MAP.get(section, SectorPreset.MULTI_SECTOR)
        claim_focus = SECTOR_CLAIM_FOCUS.get(preset, [])

        return {
            "nace_code": nace_code,
            "nace_section": section,
            "detected_sector": preset.value,
            "claim_focus_areas": claim_focus,
            "claim_focus_count": len(claim_focus),
        }

    # ------------------------------------------------------------------
    # Step handlers
    # ------------------------------------------------------------------

    def _get_step_handler(self, step: WizardStep):
        """Get handler function for a wizard step."""
        handlers = {
            WizardStep.ENTITY_PROFILE: self._step_entity_profile,
            WizardStep.CLAIM_INVENTORY: self._step_claim_inventory,
            WizardStep.EVIDENCE_SOURCES: self._step_evidence_sources,
            WizardStep.LABEL_REGISTRY: self._step_label_registry,
            WizardStep.SECTOR_CONFIG: self._step_sector_config,
            WizardStep.VERIFICATION_SETUP: self._step_verification_setup,
            WizardStep.NOTIFICATION_CONFIG: self._step_notification_config,
            WizardStep.VALIDATION: self._step_validation,
        }
        handler = handlers.get(step)
        if handler is None:
            raise ValueError(f"No handler for step: {step.value}")
        return handler

    def _step_entity_profile(self, data: Dict[str, Any]) -> StepResult:
        """Step 1: Collect organization profile."""
        result = StepResult(step=WizardStep.ENTITY_PROFILE)
        errors: List[str] = []
        warnings: List[str] = []

        name = data.get("name", "")
        if not name:
            errors.append("Organization name is required")

        jurisdiction = data.get("jurisdiction", "EU")
        nace_codes = data.get("nace_codes", [])

        if not nace_codes:
            warnings.append("No NACE codes provided; sector detection unavailable")

        if errors:
            result.status = StepStatus.FAILED
        else:
            result.status = StepStatus.COMPLETED
            self.state.entity_name = name
            if nace_codes and self.config.enable_sector_detection:
                sector_info = self.detect_sector(nace_codes[0] if nace_codes else "")
                self.state.sector = sector_info.get("detected_sector", "")

        result.data_collected = {
            "name": name,
            "jurisdiction": jurisdiction,
            "nace_codes": nace_codes,
            "sector": self.state.sector,
        }
        result.validation_errors = errors
        result.validation_warnings = warnings
        result.completed_at = utcnow()
        return result

    def _step_claim_inventory(self, data: Dict[str, Any]) -> StepResult:
        """Step 2: Register existing environmental claims."""
        result = StepResult(step=WizardStep.CLAIM_INVENTORY)
        claims = data.get("claims", [])
        warnings: List[str] = []

        if not claims:
            warnings.append("No claims registered; you can add claims later")

        registered = []
        for claim in claims:
            registered.append({
                "claim_id": _new_uuid(),
                "text": claim.get("text", "") if isinstance(claim, dict) else str(claim),
                "type": claim.get("type", "unclassified") if isinstance(claim, dict) else "unclassified",
                "registered_at": str(utcnow()),
            })

        result.status = StepStatus.COMPLETED
        result.data_collected = {
            "claims_registered": len(registered),
            "claims": registered,
        }
        result.validation_warnings = warnings
        result.completed_at = utcnow()
        return result

    def _step_evidence_sources(self, data: Dict[str, Any]) -> StepResult:
        """Step 3: Configure evidence data sources."""
        result = StepResult(step=WizardStep.EVIDENCE_SOURCES)
        sources = data.get("sources", [])
        warnings: List[str] = []

        if not sources:
            warnings.append("No evidence sources configured; substantiation will be limited")

        configured = []
        for source in sources:
            configured.append({
                "source_id": _new_uuid(),
                "type": source.get("type", "manual") if isinstance(source, dict) else "manual",
                "name": source.get("name", "") if isinstance(source, dict) else str(source),
                "connected": False,
            })

        result.status = StepStatus.COMPLETED
        result.data_collected = {
            "sources_configured": len(configured),
            "sources": configured,
        }
        result.validation_warnings = warnings
        result.completed_at = utcnow()
        return result

    def _step_label_registry(self, data: Dict[str, Any]) -> StepResult:
        """Step 4: Register sustainability labels in use."""
        result = StepResult(step=WizardStep.LABEL_REGISTRY)
        labels = data.get("labels", [])
        warnings: List[str] = []

        if not labels:
            warnings.append("No labels registered; label audit will have no data")

        registered = []
        for label in labels:
            label_name = label.get("name", "") if isinstance(label, dict) else str(label)
            registered.append({
                "label_id": _new_uuid(),
                "name": label_name,
                "certification_body": label.get("certification_body", "") if isinstance(label, dict) else "",
                "expiry_date": label.get("expiry_date", "") if isinstance(label, dict) else "",
            })

        result.status = StepStatus.COMPLETED
        result.data_collected = {
            "labels_registered": len(registered),
            "labels": registered,
        }
        result.validation_warnings = warnings
        result.completed_at = utcnow()
        return result

    def _step_sector_config(self, data: Dict[str, Any]) -> StepResult:
        """Step 5: Apply sector-specific configuration."""
        result = StepResult(step=WizardStep.SECTOR_CONFIG)
        sector = data.get("sector", self.state.sector)

        try:
            preset = SectorPreset(sector) if sector else SectorPreset.MULTI_SECTOR
        except ValueError:
            preset = SectorPreset.MULTI_SECTOR

        claim_focus = SECTOR_CLAIM_FOCUS.get(preset, [])

        result.status = StepStatus.COMPLETED
        result.data_collected = {
            "sector": preset.value,
            "claim_focus_areas": claim_focus,
            "custom_rules": data.get("custom_rules", []),
        }
        result.completed_at = utcnow()
        return result

    def _step_verification_setup(self, data: Dict[str, Any]) -> StepResult:
        """Step 6: Configure verification preferences."""
        result = StepResult(step=WizardStep.VERIFICATION_SETUP)

        result.status = StepStatus.COMPLETED
        result.data_collected = {
            "verification_body": data.get("verification_body", ""),
            "verification_standard": data.get("verification_standard", "ISO_14024"),
            "auto_verify": data.get("auto_verify", False),
            "verification_frequency": data.get("verification_frequency", "annual"),
        }
        result.completed_at = utcnow()
        return result

    def _step_notification_config(self, data: Dict[str, Any]) -> StepResult:
        """Step 7: Configure notification channels."""
        result = StepResult(step=WizardStep.NOTIFICATION_CONFIG)

        result.status = StepStatus.COMPLETED
        result.data_collected = {
            "email_notifications": data.get("email_notifications", True),
            "email_recipients": data.get("email_recipients", []),
            "slack_webhook": data.get("slack_webhook", ""),
            "alert_on_violation": data.get("alert_on_violation", True),
            "alert_on_deadline": data.get("alert_on_deadline", True),
            "digest_frequency": data.get("digest_frequency", "weekly"),
        }
        result.completed_at = utcnow()
        return result

    def _step_validation(self, data: Dict[str, Any]) -> StepResult:
        """Step 8: Validate and finalize configuration."""
        result = StepResult(step=WizardStep.VALIDATION)
        errors: List[str] = []
        warnings: List[str] = []

        if WizardStep.ENTITY_PROFILE.value not in self.state.steps_completed:
            errors.append("Entity profile step must be completed before validation")

        if WizardStep.CLAIM_INVENTORY.value not in self.state.steps_completed:
            warnings.append("No claims inventoried; pack will start with empty claim list")

        if WizardStep.EVIDENCE_SOURCES.value not in self.state.steps_completed:
            warnings.append("No evidence sources configured; substantiation capabilities limited")

        completed_count = len(self.state.steps_completed)
        required_count = self.state.total_steps - 1  # Validation itself excluded

        if errors:
            result.status = StepStatus.FAILED
        else:
            result.status = StepStatus.COMPLETED
            self.state.is_complete = True
            self.state.completed_at = utcnow()

        result.data_collected = {
            "steps_completed": completed_count,
            "steps_required": required_count,
            "completion_pct": round(completed_count / required_count * 100, 1) if required_count > 0 else 0.0,
            "entity_name": self.state.entity_name,
            "sector": self.state.sector,
            "configuration_valid": len(errors) == 0,
        }
        result.validation_errors = errors
        result.validation_warnings = warnings
        result.completed_at = utcnow()
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_required_fields(self, step: WizardStep) -> List[str]:
        """Get required input fields for a wizard step."""
        field_map: Dict[WizardStep, List[str]] = {
            WizardStep.ENTITY_PROFILE: ["name", "jurisdiction"],
            WizardStep.CLAIM_INVENTORY: ["claims"],
            WizardStep.EVIDENCE_SOURCES: ["sources"],
            WizardStep.LABEL_REGISTRY: ["labels"],
            WizardStep.SECTOR_CONFIG: ["sector"],
            WizardStep.VERIFICATION_SETUP: ["verification_body"],
            WizardStep.NOTIFICATION_CONFIG: ["email_recipients"],
            WizardStep.VALIDATION: [],
        }
        return field_map.get(step, [])
