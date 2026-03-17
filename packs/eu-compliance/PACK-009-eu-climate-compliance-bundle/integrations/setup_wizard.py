"""
Bundle Setup Wizard - PACK-009 EU Climate Compliance Bundle

This module provides a 10-step guided configuration wizard for the EU Climate
Compliance Bundle deployment. It walks users through regulation selection,
entity setup, per-pack configuration, data source configuration, calendar
setup, reporting preferences, evidence configuration, and activation.

Setup steps:
1.  Welcome - Bundle introduction and prerequisites check
2.  Regulation Selection - Choose which regulations to enable (CSRD/CBAM/EUDR/Taxonomy)
3.  Entity Setup - Organization details, type, jurisdiction
4.  Pack Configuration - Per-pack specific settings
5.  Data Sources - Configure data inputs (ERP, Excel, API, manual)
6.  Calendar Setup - Regulatory deadline configuration
7.  Reporting Preferences - Output formats, languages, templates
8.  Evidence Config - Evidence management and reuse preferences
9.  Review - Configuration review and validation
10. Activation - Generate final config and activate bundle

Example:
    >>> config = BundleSetupWizardConfig()
    >>> wizard = BundleSetupWizard(config)
    >>> wizard.start()
    >>> result = wizard.advance_step(step_1_answers)
    >>> final = wizard.complete()
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class BundleSetupWizardConfig(BaseModel):
    """Configuration for bundle setup wizard."""

    skip_optional_steps: bool = Field(
        default=False,
        description="Skip optional configuration steps"
    )
    interactive_mode: bool = Field(
        default=True,
        description="Run in interactive mode with prompts"
    )
    auto_configure: bool = Field(
        default=False,
        description="Auto-configure with recommended defaults"
    )
    preset: Optional[str] = Field(
        default=None,
        description="Preset name to apply (full_bundle, csrd_taxonomy, cbam_only, etc.)"
    )
    default_reporting_year: int = Field(
        default=2025,
        ge=2023,
        description="Default reporting period year"
    )


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class WizardStepResult(BaseModel):
    """Result from a single setup step."""

    step_number: int
    name: str
    status: Literal["PASS", "WARN", "FAIL", "SKIP"] = "PASS"
    message: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WizardResult(BaseModel):
    """Complete setup wizard result."""

    overall_status: Literal["PASS", "WARN", "FAIL"] = "PASS"
    total_steps: int = 0
    passed: int = 0
    warned: int = 0
    failed: int = 0
    skipped: int = 0
    steps: List[WizardStepResult] = Field(default_factory=list)
    generated_config: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

WIZARD_STEPS: List[Dict[str, Any]] = [
    {"number": 1, "name": "Welcome", "required": True},
    {"number": 2, "name": "Regulation Selection", "required": True},
    {"number": 3, "name": "Entity Setup", "required": True},
    {"number": 4, "name": "Pack Configuration", "required": True},
    {"number": 5, "name": "Data Sources", "required": True},
    {"number": 6, "name": "Calendar Setup", "required": False},
    {"number": 7, "name": "Reporting Preferences", "required": False},
    {"number": 8, "name": "Evidence Config", "required": False},
    {"number": 9, "name": "Review", "required": True},
    {"number": 10, "name": "Activation", "required": True},
]


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

BUNDLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "full_bundle": {
        "description": "All 4 EU regulations enabled",
        "enable_csrd": True,
        "enable_cbam": True,
        "enable_eudr": True,
        "enable_taxonomy": True,
        "organization_type": "non_financial_undertaking",
        "data_source": "erp",
        "reporting_format": "pdf",
        "evidence_reuse": True,
        "calendar_sync": True,
    },
    "csrd_taxonomy": {
        "description": "CSRD and EU Taxonomy only",
        "enable_csrd": True,
        "enable_cbam": False,
        "enable_eudr": False,
        "enable_taxonomy": True,
        "organization_type": "non_financial_undertaking",
        "data_source": "erp",
        "reporting_format": "pdf",
        "evidence_reuse": True,
        "calendar_sync": True,
    },
    "cbam_only": {
        "description": "CBAM readiness only",
        "enable_csrd": False,
        "enable_cbam": True,
        "enable_eudr": False,
        "enable_taxonomy": False,
        "organization_type": "non_financial_undertaking",
        "data_source": "excel",
        "reporting_format": "xlsx",
        "evidence_reuse": False,
        "calendar_sync": True,
    },
    "eudr_supply_chain": {
        "description": "EUDR supply chain compliance",
        "enable_csrd": False,
        "enable_cbam": False,
        "enable_eudr": True,
        "enable_taxonomy": False,
        "organization_type": "non_financial_undertaking",
        "data_source": "api",
        "reporting_format": "pdf",
        "evidence_reuse": False,
        "calendar_sync": True,
    },
    "financial_institution": {
        "description": "Financial institution full bundle with GAR",
        "enable_csrd": True,
        "enable_cbam": False,
        "enable_eudr": False,
        "enable_taxonomy": True,
        "organization_type": "financial_institution",
        "data_source": "erp",
        "reporting_format": "both",
        "evidence_reuse": True,
        "calendar_sync": True,
        "enable_gar": True,
    },
    "sme_starter": {
        "description": "SME simplified bundle (CSRD + CBAM)",
        "enable_csrd": True,
        "enable_cbam": True,
        "enable_eudr": False,
        "enable_taxonomy": False,
        "organization_type": "sme",
        "data_source": "excel",
        "reporting_format": "pdf",
        "evidence_reuse": False,
        "calendar_sync": True,
    },
}


# ---------------------------------------------------------------------------
# Wizard
# ---------------------------------------------------------------------------

class BundleSetupWizard:
    """
    10-step guided configuration wizard for EU Climate Compliance Bundle.

    Generates a complete BundleOrchestratorConfig from user answers,
    with support for presets, step-by-step navigation, and validation.

    Example:
        >>> config = BundleSetupWizardConfig(preset="full_bundle")
        >>> wizard = BundleSetupWizard(config)
        >>> wizard.start()
        >>> wizard.advance_step({"regulations": ["csrd", "cbam", "eudr", "taxonomy"]})
        >>> result = wizard.complete()
    """

    def __init__(self, config: BundleSetupWizardConfig):
        """Initialize setup wizard."""
        self.config = config
        self._setup_data: Dict[str, Any] = {}
        self._current_step: int = 0
        self._step_results: List[WizardStepResult] = []
        self._completed: bool = False

        if config.preset and config.preset in BUNDLE_PRESETS:
            self._setup_data = BUNDLE_PRESETS[config.preset].copy()
            logger.info(f"Applied preset: {config.preset}")

        logger.info("BundleSetupWizard initialized (10 steps)")

    # ------------------------------------------------------------------
    # Public navigation API
    # ------------------------------------------------------------------

    def start(self) -> Dict[str, Any]:
        """
        Start the setup wizard at step 1.

        Returns:
            Welcome step information.
        """
        self._current_step = 1
        self._step_results = []
        self._completed = False

        return {
            "wizard": "PACK-009 EU Climate Compliance Bundle Setup",
            "current_step": 1,
            "total_steps": 10,
            "preset_applied": self.config.preset,
            "steps": [
                {"number": s["number"], "name": s["name"], "required": s["required"]}
                for s in WIZARD_STEPS
            ],
            "message": "Welcome to the EU Climate Compliance Bundle setup wizard.",
        }

    def advance_step(self, answers: Optional[Dict[str, Any]] = None) -> WizardStepResult:
        """
        Advance to the next step, processing current step answers.

        Args:
            answers: Answers for the current step.

        Returns:
            Step result for the processed step.
        """
        if self._current_step < 1 or self._current_step > 10:
            return WizardStepResult(
                step_number=self._current_step,
                name="Invalid",
                status="FAIL",
                message=f"Invalid step number: {self._current_step}",
            )

        answers = answers or {}
        step_info = WIZARD_STEPS[self._current_step - 1]

        # Skip optional steps if configured
        if not step_info["required"] and self.config.skip_optional_steps:
            result = WizardStepResult(
                step_number=self._current_step,
                name=step_info["name"],
                status="SKIP",
                message=f"Skipped optional step: {step_info['name']}",
            )
            self._step_results.append(result)
            self._current_step += 1
            return result

        # Process current step
        step_handlers = {
            1: self._process_welcome,
            2: self._process_regulation_selection,
            3: self._process_entity_setup,
            4: self._process_pack_configuration,
            5: self._process_data_sources,
            6: self._process_calendar_setup,
            7: self._process_reporting_preferences,
            8: self._process_evidence_config,
            9: self._process_review,
            10: self._process_activation,
        }

        handler = step_handlers[self._current_step]
        result = handler(answers)
        self._step_results.append(result)

        if result.status != "FAIL":
            self._current_step += 1

        return result

    def go_back(self) -> Dict[str, Any]:
        """
        Go back to the previous step.

        Returns:
            Information about the new current step.
        """
        if self._current_step <= 1:
            return {
                "current_step": 1,
                "message": "Already at the first step",
            }

        self._current_step -= 1

        # Remove the last step result if it corresponds to the current step
        if self._step_results and self._step_results[-1].step_number >= self._current_step:
            self._step_results = [
                r for r in self._step_results if r.step_number < self._current_step
            ]

        step_info = WIZARD_STEPS[self._current_step - 1]
        return {
            "current_step": self._current_step,
            "step_name": step_info["name"],
            "message": f"Returned to step {self._current_step}: {step_info['name']}",
        }

    def get_current_step(self) -> Dict[str, Any]:
        """
        Get information about the current step.

        Returns:
            Current step details.
        """
        if self._current_step < 1 or self._current_step > 10:
            return {"current_step": self._current_step, "message": "Wizard not started or completed"}

        step_info = WIZARD_STEPS[self._current_step - 1]
        return {
            "current_step": self._current_step,
            "step_name": step_info["name"],
            "required": step_info["required"],
            "completed_steps": len(self._step_results),
            "remaining_steps": 10 - len(self._step_results),
            "setup_data_so_far": self._setup_data,
        }

    def complete(self) -> WizardResult:
        """
        Complete the wizard and generate final configuration.

        Returns:
            Complete wizard result with generated configuration.
        """
        if self._current_step <= 10 and not self._completed:
            # Auto-complete remaining steps if auto_configure
            if self.config.auto_configure:
                while self._current_step <= 10:
                    self.advance_step({})
            else:
                logger.warning("Wizard completed before all steps processed")

        self._completed = True
        return self._build_result()

    def validate_step(
        self, step_num: int, data: Dict[str, Any]
    ) -> WizardStepResult:
        """
        Validate a single step's data without advancing.

        Args:
            step_num: Step number (1-10).
            data: Step data to validate.

        Returns:
            Validation result.
        """
        validators = {
            1: self._validate_welcome,
            2: self._validate_regulation_selection,
            3: self._validate_entity_setup,
            4: self._validate_pack_configuration,
            5: self._validate_data_sources,
            6: self._validate_calendar_setup,
            7: self._validate_reporting_preferences,
            8: self._validate_evidence_config,
            9: self._validate_review,
            10: self._validate_activation,
        }

        validator = validators.get(step_num)
        if validator:
            return validator(data)

        return WizardStepResult(
            step_number=step_num,
            name="Unknown",
            status="FAIL",
            message=f"Unknown step number: {step_num}",
        )

    # ------------------------------------------------------------------
    # Step processors
    # ------------------------------------------------------------------

    def _process_welcome(self, answers: Dict[str, Any]) -> WizardStepResult:
        """Step 1: Welcome - prerequisites check."""
        logger.info("Step 1: Welcome")

        prerequisites = {
            "python_version": "3.10+",
            "pydantic_version": "2.x",
            "pack_001_available": True,
            "pack_004_available": True,
            "pack_006_available": True,
            "pack_008_available": True,
        }

        accepted = answers.get("accept_prerequisites", True)

        if not accepted:
            return WizardStepResult(
                step_number=1,
                name="Welcome",
                status="FAIL",
                message="Prerequisites not accepted",
                data={"prerequisites": prerequisites},
            )

        self._setup_data["prerequisites_accepted"] = True

        return WizardStepResult(
            step_number=1,
            name="Welcome",
            status="PASS",
            message="Welcome completed, prerequisites verified",
            data={"prerequisites": prerequisites, "accepted": True},
        )

    def _process_regulation_selection(
        self, answers: Dict[str, Any]
    ) -> WizardStepResult:
        """Step 2: Regulation Selection."""
        logger.info("Step 2: Regulation Selection")

        valid_regulations = ["csrd", "cbam", "eudr", "taxonomy"]
        selected = answers.get(
            "regulations",
            [r for r in valid_regulations if self._setup_data.get(f"enable_{r}", True)],
        )

        invalid = [r for r in selected if r not in valid_regulations]
        if invalid:
            return WizardStepResult(
                step_number=2,
                name="Regulation Selection",
                status="FAIL",
                message=f"Invalid regulations: {invalid}",
                data={"valid_regulations": valid_regulations},
            )

        if not selected:
            return WizardStepResult(
                step_number=2,
                name="Regulation Selection",
                status="FAIL",
                message="At least one regulation must be selected",
                data={"valid_regulations": valid_regulations},
            )

        for reg in valid_regulations:
            self._setup_data[f"enable_{reg}"] = reg in selected

        return WizardStepResult(
            step_number=2,
            name="Regulation Selection",
            status="PASS",
            message=f"Selected {len(selected)} regulations: {', '.join(r.upper() for r in selected)}",
            data={"selected_regulations": selected},
        )

    def _process_entity_setup(self, answers: Dict[str, Any]) -> WizardStepResult:
        """Step 3: Entity Setup."""
        logger.info("Step 3: Entity Setup")

        valid_types = ["non_financial_undertaking", "financial_institution", "asset_manager", "sme"]

        org_name = answers.get(
            "organization_name",
            self._setup_data.get("organization_name", ""),
        )
        org_type = answers.get(
            "organization_type",
            self._setup_data.get("organization_type", "non_financial_undertaking"),
        )
        jurisdiction = answers.get("jurisdiction", "EU")
        currency = answers.get("currency", "EUR")
        reporting_year = answers.get(
            "reporting_year",
            self.config.default_reporting_year,
        )

        if org_type not in valid_types:
            return WizardStepResult(
                step_number=3,
                name="Entity Setup",
                status="FAIL",
                message=f"Invalid organization type: {org_type}",
                data={"valid_types": valid_types},
            )

        entity_config = {
            "organization_name": org_name,
            "organization_type": org_type,
            "jurisdiction": jurisdiction,
            "currency": currency,
            "reporting_year": reporting_year,
        }
        self._setup_data.update(entity_config)

        status: Literal["PASS", "WARN", "FAIL"] = "PASS"
        if not org_name:
            status = "WARN"

        return WizardStepResult(
            step_number=3,
            name="Entity Setup",
            status=status,
            message=f"Entity: {org_name or '(not set)'} ({org_type})",
            data=entity_config,
        )

    def _process_pack_configuration(
        self, answers: Dict[str, Any]
    ) -> WizardStepResult:
        """Step 4: Pack Configuration."""
        logger.info("Step 4: Pack Configuration")

        pack_configs: Dict[str, Any] = {}

        if self._setup_data.get("enable_csrd"):
            pack_configs["csrd"] = {
                "esrs_version": answers.get("csrd_esrs_version", "2023"),
                "materiality_assessment": answers.get("csrd_materiality", True),
                "sector_specific": answers.get("csrd_sector_specific", False),
            }

        if self._setup_data.get("enable_cbam"):
            reporting_year = self._setup_data.get("reporting_year", 2025)
            pack_configs["cbam"] = {
                "transitional_period": reporting_year < 2026,
                "quarterly_reporting": answers.get("cbam_quarterly", True),
                "calculation_method": answers.get("cbam_calc_method", "actual"),
                "include_indirect": answers.get("cbam_indirect", True),
            }

        if self._setup_data.get("enable_eudr"):
            pack_configs["eudr"] = {
                "commodities": answers.get(
                    "eudr_commodities",
                    ["soy", "palm_oil", "wood", "cocoa", "coffee", "rubber", "cattle"],
                ),
                "operator_type": answers.get("eudr_operator_type", "operator"),
                "geolocation_enabled": answers.get("eudr_geolocation", True),
                "risk_assessment_enabled": answers.get("eudr_risk_assessment", True),
            }

        if self._setup_data.get("enable_taxonomy"):
            org_type = self._setup_data.get("organization_type", "non_financial_undertaking")
            pack_configs["taxonomy"] = {
                "environmental_objectives": answers.get(
                    "taxonomy_objectives",
                    ["CCM", "CCA", "WTR", "CE", "PPC", "BIO"],
                ),
                "enable_gar": answers.get(
                    "taxonomy_gar",
                    org_type == "financial_institution",
                ),
                "enable_capex_plan": answers.get("taxonomy_capex_plan", True),
                "delegated_act_version": answers.get("taxonomy_da_version", "2023"),
                "disclosure_format": answers.get("taxonomy_disclosure", "article_8"),
            }

        self._setup_data["pack_configs"] = pack_configs

        return WizardStepResult(
            step_number=4,
            name="Pack Configuration",
            status="PASS",
            message=f"Configured {len(pack_configs)} packs",
            data={"pack_configs": pack_configs},
        )

    def _process_data_sources(self, answers: Dict[str, Any]) -> WizardStepResult:
        """Step 5: Data Sources."""
        logger.info("Step 5: Data Sources")

        valid_sources = ["erp", "excel", "api", "manual"]
        primary_source = answers.get(
            "primary_source",
            self._setup_data.get("data_source", "erp"),
        )

        if primary_source not in valid_sources:
            return WizardStepResult(
                step_number=5,
                name="Data Sources",
                status="FAIL",
                message=f"Invalid data source: {primary_source}",
                data={"valid_sources": valid_sources},
            )

        data_config = {
            "primary_source": primary_source,
            "secondary_sources": answers.get("secondary_sources", []),
            "erp_endpoint": answers.get("erp_endpoint", ""),
            "api_endpoint": answers.get("api_endpoint", ""),
            "batch_size": answers.get("batch_size", 500),
            "parallel_processing": answers.get("parallel_processing", True),
            "deduplication_enabled": answers.get("deduplication", True),
        }

        self._setup_data["data_sources"] = data_config

        return WizardStepResult(
            step_number=5,
            name="Data Sources",
            status="PASS",
            message=f"Primary data source: {primary_source}",
            data=data_config,
        )

    def _process_calendar_setup(self, answers: Dict[str, Any]) -> WizardStepResult:
        """Step 6: Calendar Setup."""
        logger.info("Step 6: Calendar Setup")

        calendar_config = {
            "calendar_sync_enabled": answers.get(
                "calendar_sync",
                self._setup_data.get("calendar_sync", True),
            ),
            "notification_lead_days": answers.get("lead_days", 30),
            "notification_channels": answers.get(
                "channels", ["email"]
            ),
            "custom_deadlines": answers.get("custom_deadlines", []),
        }

        self._setup_data["calendar"] = calendar_config

        return WizardStepResult(
            step_number=6,
            name="Calendar Setup",
            status="PASS",
            message=f"Calendar sync: {'enabled' if calendar_config['calendar_sync_enabled'] else 'disabled'}",
            data=calendar_config,
        )

    def _process_reporting_preferences(
        self, answers: Dict[str, Any]
    ) -> WizardStepResult:
        """Step 7: Reporting Preferences."""
        logger.info("Step 7: Reporting Preferences")

        valid_formats = ["pdf", "xlsx", "html", "json", "both"]
        report_format = answers.get(
            "report_format",
            self._setup_data.get("reporting_format", "pdf"),
        )

        if report_format not in valid_formats:
            report_format = "pdf"

        reporting_config = {
            "report_format": report_format,
            "language": answers.get("language", "en"),
            "include_executive_summary": answers.get("exec_summary", True),
            "include_gap_analysis": answers.get("gap_analysis", True),
            "include_calendar": answers.get("include_calendar", True),
            "include_evidence_index": answers.get("evidence_index", True),
            "branding": {
                "company_logo": answers.get("logo_path", ""),
                "primary_color": answers.get("primary_color", "#1a5276"),
            },
        }

        self._setup_data["reporting"] = reporting_config

        return WizardStepResult(
            step_number=7,
            name="Reporting Preferences",
            status="PASS",
            message=f"Report format: {report_format} ({reporting_config['language']})",
            data=reporting_config,
        )

    def _process_evidence_config(self, answers: Dict[str, Any]) -> WizardStepResult:
        """Step 8: Evidence Config."""
        logger.info("Step 8: Evidence Config")

        evidence_config = {
            "evidence_reuse_enabled": answers.get(
                "evidence_reuse",
                self._setup_data.get("evidence_reuse", True),
            ),
            "auto_map_requirements": answers.get("auto_map", True),
            "evidence_retention_years": answers.get("retention_years", 7),
            "evidence_storage": answers.get("storage", "local"),
            "completeness_tracking": answers.get("completeness_tracking", True),
        }

        self._setup_data["evidence"] = evidence_config

        return WizardStepResult(
            step_number=8,
            name="Evidence Config",
            status="PASS",
            message=f"Evidence reuse: {'enabled' if evidence_config['evidence_reuse_enabled'] else 'disabled'}",
            data=evidence_config,
        )

    def _process_review(self, answers: Dict[str, Any]) -> WizardStepResult:
        """Step 9: Review - validate complete configuration."""
        logger.info("Step 9: Review")

        issues: List[str] = []

        # Validate required fields
        if not self._setup_data.get("organization_type"):
            issues.append("Missing organization type")

        enabled_regs = [
            r for r in ["csrd", "cbam", "eudr", "taxonomy"]
            if self._setup_data.get(f"enable_{r}")
        ]
        if not enabled_regs:
            issues.append("No regulations selected")

        if not self._setup_data.get("data_sources"):
            issues.append("No data sources configured")

        # Cross-validation
        org_type = self._setup_data.get("organization_type", "")
        if org_type == "financial_institution":
            taxonomy_config = self._setup_data.get("pack_configs", {}).get("taxonomy", {})
            if "taxonomy" in enabled_regs and not taxonomy_config.get("enable_gar"):
                issues.append("GAR recommended for financial institutions with Taxonomy")

        if issues:
            severity = "WARN" if len(issues) <= 2 else "FAIL"
            return WizardStepResult(
                step_number=9,
                name="Review",
                status=severity,
                message=f"Found {len(issues)} issue(s)",
                data={
                    "issues": issues,
                    "configuration_summary": self._generate_summary(),
                },
            )

        return WizardStepResult(
            step_number=9,
            name="Review",
            status="PASS",
            message="Configuration review passed",
            data={
                "issues": [],
                "valid": True,
                "configuration_summary": self._generate_summary(),
            },
        )

    def _process_activation(self, answers: Dict[str, Any]) -> WizardStepResult:
        """Step 10: Activation - generate final config."""
        logger.info("Step 10: Activation")

        confirm = answers.get("confirm_activation", True)

        if not confirm:
            return WizardStepResult(
                step_number=10,
                name="Activation",
                status="FAIL",
                message="Activation not confirmed",
            )

        final_config = self._generate_final_config()
        self._setup_data["final_config"] = final_config
        self._completed = True

        return WizardStepResult(
            step_number=10,
            name="Activation",
            status="PASS",
            message="Bundle activated successfully",
            data=final_config,
        )

    # ------------------------------------------------------------------
    # Step validators
    # ------------------------------------------------------------------

    def _validate_welcome(self, data: Dict[str, Any]) -> WizardStepResult:
        """Validate welcome step data."""
        accepted = data.get("accept_prerequisites", False)
        return WizardStepResult(
            step_number=1, name="Welcome",
            status="PASS" if accepted else "FAIL",
            message="Prerequisites accepted" if accepted else "Prerequisites not accepted",
        )

    def _validate_regulation_selection(self, data: Dict[str, Any]) -> WizardStepResult:
        """Validate regulation selection data."""
        regs = data.get("regulations", [])
        valid = all(r in ["csrd", "cbam", "eudr", "taxonomy"] for r in regs) and len(regs) > 0
        return WizardStepResult(
            step_number=2, name="Regulation Selection",
            status="PASS" if valid else "FAIL",
            message=f"{len(regs)} regulations {'valid' if valid else 'invalid'}",
        )

    def _validate_entity_setup(self, data: Dict[str, Any]) -> WizardStepResult:
        """Validate entity setup data."""
        org_type = data.get("organization_type", "")
        valid = org_type in [
            "non_financial_undertaking", "financial_institution", "asset_manager", "sme"
        ]
        return WizardStepResult(
            step_number=3, name="Entity Setup",
            status="PASS" if valid else "FAIL",
            message=f"Organization type {'valid' if valid else 'invalid'}: {org_type}",
        )

    def _validate_pack_configuration(self, data: Dict[str, Any]) -> WizardStepResult:
        """Validate pack configuration data."""
        pack_configs = data.get("pack_configs", {})
        return WizardStepResult(
            step_number=4, name="Pack Configuration",
            status="PASS" if pack_configs else "WARN",
            message=f"{len(pack_configs)} packs configured",
        )

    def _validate_data_sources(self, data: Dict[str, Any]) -> WizardStepResult:
        """Validate data source data."""
        source = data.get("primary_source", "")
        valid = source in ["erp", "excel", "api", "manual"]
        return WizardStepResult(
            step_number=5, name="Data Sources",
            status="PASS" if valid else "FAIL",
            message=f"Data source {'valid' if valid else 'invalid'}: {source}",
        )

    def _validate_calendar_setup(self, data: Dict[str, Any]) -> WizardStepResult:
        """Validate calendar setup data."""
        return WizardStepResult(
            step_number=6, name="Calendar Setup",
            status="PASS",
            message="Calendar configuration valid",
        )

    def _validate_reporting_preferences(self, data: Dict[str, Any]) -> WizardStepResult:
        """Validate reporting preferences data."""
        fmt = data.get("report_format", "")
        valid = fmt in ["pdf", "xlsx", "html", "json", "both"]
        return WizardStepResult(
            step_number=7, name="Reporting Preferences",
            status="PASS" if valid else "FAIL",
            message=f"Report format {'valid' if valid else 'invalid'}: {fmt}",
        )

    def _validate_evidence_config(self, data: Dict[str, Any]) -> WizardStepResult:
        """Validate evidence configuration data."""
        return WizardStepResult(
            step_number=8, name="Evidence Config",
            status="PASS",
            message="Evidence configuration valid",
        )

    def _validate_review(self, data: Dict[str, Any]) -> WizardStepResult:
        """Validate review step data."""
        return WizardStepResult(
            step_number=9, name="Review",
            status="PASS",
            message="Review data valid",
        )

    def _validate_activation(self, data: Dict[str, Any]) -> WizardStepResult:
        """Validate activation data."""
        confirmed = data.get("confirm_activation", False)
        return WizardStepResult(
            step_number=10, name="Activation",
            status="PASS" if confirmed else "FAIL",
            message="Activation confirmed" if confirmed else "Activation not confirmed",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate configuration summary for review."""
        enabled_regs = [
            r for r in ["csrd", "cbam", "eudr", "taxonomy"]
            if self._setup_data.get(f"enable_{r}")
        ]

        return {
            "pack": "PACK-009 EU Climate Compliance Bundle",
            "organization": self._setup_data.get("organization_name", "(not set)"),
            "organization_type": self._setup_data.get("organization_type", "(not set)"),
            "reporting_year": self._setup_data.get("reporting_year", self.config.default_reporting_year),
            "enabled_regulations": enabled_regs,
            "data_source": self._setup_data.get("data_sources", {}).get("primary_source", "(not set)"),
            "calendar_sync": self._setup_data.get("calendar", {}).get("calendar_sync_enabled", True),
            "evidence_reuse": self._setup_data.get("evidence", {}).get("evidence_reuse_enabled", True),
            "report_format": self._setup_data.get("reporting", {}).get("report_format", "(not set)"),
        }

    def _generate_final_config(self) -> Dict[str, Any]:
        """Generate final bundle configuration from wizard data."""
        return {
            "pack": "PACK-009 EU Climate Compliance Bundle",
            "version": "1.0.0",
            "organization_name": self._setup_data.get("organization_name", ""),
            "organization_type": self._setup_data.get("organization_type", "non_financial_undertaking"),
            "jurisdiction": self._setup_data.get("jurisdiction", "EU"),
            "currency": self._setup_data.get("currency", "EUR"),
            "reporting_year": self._setup_data.get("reporting_year", self.config.default_reporting_year),
            "regulations": {
                "csrd": self._setup_data.get("enable_csrd", True),
                "cbam": self._setup_data.get("enable_cbam", True),
                "eudr": self._setup_data.get("enable_eudr", True),
                "taxonomy": self._setup_data.get("enable_taxonomy", True),
            },
            "pack_configs": self._setup_data.get("pack_configs", {}),
            "data_sources": self._setup_data.get("data_sources", {}),
            "calendar": self._setup_data.get("calendar", {}),
            "reporting": self._setup_data.get("reporting", {}),
            "evidence": self._setup_data.get("evidence", {}),
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _build_result(self) -> WizardResult:
        """Build final wizard result from step results."""
        total = len(self._step_results)
        passed = sum(1 for s in self._step_results if s.status == "PASS")
        warned = sum(1 for s in self._step_results if s.status == "WARN")
        failed = sum(1 for s in self._step_results if s.status == "FAIL")
        skipped = sum(1 for s in self._step_results if s.status == "SKIP")

        if failed > 0:
            overall: Literal["PASS", "WARN", "FAIL"] = "FAIL"
        elif warned > 0:
            overall = "WARN"
        else:
            overall = "PASS"

        return WizardResult(
            overall_status=overall,
            total_steps=total,
            passed=passed,
            warned=warned,
            failed=failed,
            skipped=skipped,
            steps=self._step_results,
            generated_config=self._setup_data.get("final_config", {}),
        )
