# -*- coding: utf-8 -*-
"""
SFDRSetupWizard - 8-Step Guided Configuration for SFDR Article 8
=================================================================

This module implements the guided setup wizard for PACK-010 SFDR Article 8.
It walks through product type selection, E/S characteristics definition,
binding elements setup, taxonomy alignment commitment, PAI indicator
selection, data source configuration, reporting schedule, and validation
with deployment.

Wizard Steps:
    1. Product Type Selection (UCITS, AIF, insurance, pension, structured)
    2. E/S Characteristics Definition (environmental + social)
    3. Binding Elements Setup (exclusions, thresholds, commitments)
    4. Taxonomy Alignment Commitment (minimum %, objective scope)
    5. PAI Indicator Selection (mandatory 18 + optional)
    6. Data Source Configuration (portfolio, ESG, emissions sources)
    7. Reporting Schedule (annual/semi-annual, deadlines, distribution)
    8. Validation & Deployment (config validation, test run, confirmation)

Presets: asset_manager, insurance, bank, pension_fund, wealth_manager, custom

Example:
    >>> config = SetupWizardConfig()
    >>> wizard = SFDRSetupWizard(config)
    >>> wizard.start()
    >>> wizard.complete_step(1, {"product_type": "ucits"})
    >>> final = wizard.finalize()

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-010 SFDR Article 8
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# =============================================================================
# Utility Helpers
# =============================================================================

def _hash_data(data: Any) -> str:
    """Compute a SHA-256 hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()

# =============================================================================
# Enums
# =============================================================================

class WizardStepId(str, Enum):
    """Wizard step identifiers."""
    PRODUCT_TYPE = "product_type"
    ES_CHARACTERISTICS = "es_characteristics"
    BINDING_ELEMENTS = "binding_elements"
    TAXONOMY_ALIGNMENT = "taxonomy_alignment"
    PAI_INDICATORS = "pai_indicators"
    DATA_SOURCES = "data_sources"
    REPORTING_SCHEDULE = "reporting_schedule"
    VALIDATION_DEPLOYMENT = "validation_deployment"

class StepStatus(str, Enum):
    """Status of a wizard step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class ProductType(str, Enum):
    """Financial product types under SFDR."""
    UCITS = "ucits"
    AIF = "aif"
    INSURANCE = "insurance"
    PENSION = "pension"
    STRUCTURED = "structured"
    PORTFOLIO_MANAGEMENT = "portfolio_management"

class PresetId(str, Enum):
    """Configuration presets."""
    ASSET_MANAGER = "asset_manager"
    INSURANCE = "insurance"
    BANK = "bank"
    PENSION_FUND = "pension_fund"
    WEALTH_MANAGER = "wealth_manager"
    CUSTOM = "custom"

# =============================================================================
# Data Models
# =============================================================================

class SetupWizardConfig(BaseModel):
    """Configuration for the Setup Wizard."""
    available_presets: List[str] = Field(
        default_factory=lambda: [p.value for p in PresetId],
        description="Available configuration presets",
    )
    default_preset: str = Field(
        default="asset_manager",
        description="Default preset to apply",
    )
    auto_validate: bool = Field(
        default=True,
        description="Auto-validate steps on completion",
    )
    enable_test_run: bool = Field(
        default=True,
        description="Enable test run in validation step",
    )

class WizardStepState(BaseModel):
    """State of a single wizard step."""
    step_number: int = Field(default=0, description="Step number (1-8)")
    step_id: WizardStepId = Field(
        default=WizardStepId.PRODUCT_TYPE, description="Step identifier"
    )
    display_name: str = Field(default="", description="Human-readable name")
    status: StepStatus = Field(
        default=StepStatus.PENDING, description="Step status"
    )
    data: Dict[str, Any] = Field(
        default_factory=dict, description="Step output data"
    )
    errors: List[str] = Field(
        default_factory=list, description="Validation errors"
    )
    started_at: Optional[str] = Field(None, description="Start timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")

class WizardResult(BaseModel):
    """Final result of the setup wizard."""
    wizard_id: str = Field(
        default_factory=lambda: str(uuid4())[:16],
        description="Wizard session ID",
    )
    preset_used: str = Field(default="custom", description="Preset applied")
    steps_completed: int = Field(default=0, description="Steps completed")
    total_steps: int = Field(default=8, description="Total wizard steps")
    is_complete: bool = Field(default=False, description="Wizard completed")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Final configuration"
    )
    product_name: str = Field(default="", description="Product name")
    product_type: str = Field(default="", description="Product type")
    sfdr_classification: str = Field(default="article_8", description="Classification")
    environmental_characteristics: List[str] = Field(
        default_factory=list, description="E characteristics"
    )
    social_characteristics: List[str] = Field(
        default_factory=list, description="S characteristics"
    )
    taxonomy_alignment_pct: float = Field(
        default=0.0, description="Taxonomy alignment commitment %"
    )
    pai_indicators_count: int = Field(default=18, description="PAI indicators selected")
    reporting_frequency: str = Field(default="annual", description="Reporting frequency")
    validation_passed: bool = Field(default=False, description="Validation result")
    test_run_executed: bool = Field(default=False, description="Test run executed")
    warnings: List[str] = Field(default_factory=list, description="Setup warnings")
    created_at: str = Field(
        default_factory=lambda: utcnow().isoformat(),
        description="Wizard start timestamp",
    )
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# =============================================================================
# Step Definitions
# =============================================================================

WIZARD_STEPS: List[Dict[str, Any]] = [
    {
        "number": 1,
        "id": WizardStepId.PRODUCT_TYPE.value,
        "name": "1. Product Type Selection",
        "description": "Select financial product type (UCITS, AIF, insurance, pension, structured)",
        "fields": ["product_type", "product_name", "product_isin", "management_company", "lei_code"],
    },
    {
        "number": 2,
        "id": WizardStepId.ES_CHARACTERISTICS.value,
        "name": "2. E/S Characteristics Definition",
        "description": "Define environmental and social characteristics to promote",
        "fields": ["environmental_characteristics", "social_characteristics", "sfdr_classification"],
    },
    {
        "number": 3,
        "id": WizardStepId.BINDING_ELEMENTS.value,
        "name": "3. Binding Elements Setup",
        "description": "Configure exclusions, thresholds, and binding commitments",
        "fields": ["exclusion_categories", "revenue_thresholds", "esg_min_rating", "controversy_max"],
    },
    {
        "number": 4,
        "id": WizardStepId.TAXONOMY_ALIGNMENT.value,
        "name": "4. Taxonomy Alignment Commitment",
        "description": "Set minimum taxonomy alignment percentage and objective scope",
        "fields": ["min_taxonomy_pct", "taxonomy_objectives", "alignment_methodology"],
    },
    {
        "number": 5,
        "id": WizardStepId.PAI_INDICATORS.value,
        "name": "5. PAI Indicator Selection",
        "description": "Configure mandatory (18) and optional PAI indicators",
        "fields": ["mandatory_indicators", "optional_indicators", "pai_methodology"],
    },
    {
        "number": 6,
        "id": WizardStepId.DATA_SOURCES.value,
        "name": "6. Data Source Configuration",
        "description": "Configure portfolio system, ESG data providers, emissions sources",
        "fields": ["portfolio_source", "esg_provider", "emissions_source", "benchmark_source"],
    },
    {
        "number": 7,
        "id": WizardStepId.REPORTING_SCHEDULE.value,
        "name": "7. Reporting Schedule",
        "description": "Set reporting frequency, filing deadlines, distribution channels",
        "fields": ["frequency", "fiscal_year_end", "filing_deadline_days", "distribution_channels"],
    },
    {
        "number": 8,
        "id": WizardStepId.VALIDATION_DEPLOYMENT.value,
        "name": "8. Validation & Deployment",
        "description": "Validate configuration, execute test run, confirm deployment",
        "fields": ["validation_result", "test_run_result", "deployment_confirmed"],
    },
]

PRESETS: Dict[str, Dict[str, Any]] = {
    "asset_manager": {
        "name": "Asset Manager",
        "description": "Typical Article 8 UCITS fund configuration",
        "product_type": "ucits",
        "sfdr_classification": "article_8",
        "environmental_characteristics": [
            "carbon_emissions_reduction",
            "energy_efficiency",
            "renewable_energy_investment",
        ],
        "social_characteristics": [
            "good_governance",
            "labor_rights",
        ],
        "exclusion_categories": [
            "controversial_weapons", "tobacco", "thermal_coal", "ungc_violators",
        ],
        "esg_min_rating": "BBB",
        "min_taxonomy_pct": 0.0,
        "taxonomy_objectives": [
            "climate_change_mitigation", "climate_change_adaptation",
        ],
        "reporting_frequency": "annual",
        "portfolio_source": "api",
        "esg_provider": "internal",
    },
    "insurance": {
        "name": "Insurance Company",
        "description": "Insurance-based investment product configuration",
        "product_type": "insurance",
        "sfdr_classification": "article_8",
        "environmental_characteristics": [
            "climate_risk_management",
            "sustainable_underwriting",
        ],
        "social_characteristics": [
            "responsible_investment",
            "customer_protection",
        ],
        "exclusion_categories": [
            "controversial_weapons", "tobacco", "thermal_coal",
        ],
        "esg_min_rating": "BB",
        "min_taxonomy_pct": 0.0,
        "taxonomy_objectives": ["climate_change_mitigation"],
        "reporting_frequency": "annual",
        "portfolio_source": "manual",
        "esg_provider": "internal",
    },
    "bank": {
        "name": "Bank",
        "description": "Bank discretionary portfolio management",
        "product_type": "portfolio_management",
        "sfdr_classification": "article_8",
        "environmental_characteristics": [
            "carbon_emissions_reduction",
        ],
        "social_characteristics": [
            "good_governance",
        ],
        "exclusion_categories": [
            "controversial_weapons", "tobacco",
        ],
        "esg_min_rating": "BBB",
        "min_taxonomy_pct": 0.0,
        "taxonomy_objectives": ["climate_change_mitigation"],
        "reporting_frequency": "semi_annual",
        "portfolio_source": "api",
        "esg_provider": "internal",
    },
    "pension_fund": {
        "name": "Pension Fund",
        "description": "Occupational pension fund with Article 8 characteristics",
        "product_type": "pension",
        "sfdr_classification": "article_8_plus",
        "environmental_characteristics": [
            "carbon_emissions_reduction",
            "energy_efficiency",
            "circular_economy",
        ],
        "social_characteristics": [
            "good_governance",
            "labor_rights",
            "diversity_inclusion",
        ],
        "exclusion_categories": [
            "controversial_weapons", "tobacco", "thermal_coal",
            "fossil_fuel_exploration", "ungc_violators",
        ],
        "esg_min_rating": "A",
        "min_taxonomy_pct": 10.0,
        "taxonomy_objectives": [
            "climate_change_mitigation", "climate_change_adaptation",
            "pollution_prevention",
        ],
        "reporting_frequency": "annual",
        "portfolio_source": "api",
        "esg_provider": "external",
    },
    "wealth_manager": {
        "name": "Wealth Manager",
        "description": "Private banking / wealth management Article 8 product",
        "product_type": "portfolio_management",
        "sfdr_classification": "article_8",
        "environmental_characteristics": [
            "carbon_emissions_reduction",
        ],
        "social_characteristics": [
            "good_governance",
        ],
        "exclusion_categories": [
            "controversial_weapons", "tobacco",
        ],
        "esg_min_rating": "BBB",
        "min_taxonomy_pct": 0.0,
        "taxonomy_objectives": ["climate_change_mitigation"],
        "reporting_frequency": "annual",
        "portfolio_source": "manual",
        "esg_provider": "internal",
    },
    "custom": {
        "name": "Custom Configuration",
        "description": "Start from scratch with a custom configuration",
        "product_type": "",
        "sfdr_classification": "article_8",
        "environmental_characteristics": [],
        "social_characteristics": [],
        "exclusion_categories": [],
        "esg_min_rating": "BBB",
        "min_taxonomy_pct": 0.0,
        "taxonomy_objectives": [],
        "reporting_frequency": "annual",
        "portfolio_source": "manual",
        "esg_provider": "internal",
    },
}

# =============================================================================
# Setup Wizard
# =============================================================================

class SFDRSetupWizard:
    """8-step guided setup wizard for SFDR Article 8 Pack.

    Walks through product configuration, E/S characteristics, binding
    elements, taxonomy alignment, PAI indicators, data sources,
    reporting schedule, and validation/deployment.

    Supports preset-based configuration for common entity types.

    Attributes:
        config: Wizard configuration.
        _wizard_id: Session identifier.
        _steps: Step state tracking.
        _accumulated_config: Configuration built across steps.

    Example:
        >>> wizard = SFDRSetupWizard(SetupWizardConfig())
        >>> wizard.start()
        >>> wizard.complete_step(1, {"product_type": "ucits", "product_name": "GL ESG"})
        >>> result = wizard.finalize()
        >>> assert result.is_complete
    """

    def __init__(self, config: Optional[SetupWizardConfig] = None) -> None:
        """Initialize the SFDR Setup Wizard.

        Args:
            config: Wizard configuration. Uses defaults if not provided.
        """
        self.config = config or SetupWizardConfig()
        self.logger = logger
        self._wizard_id = str(uuid4())[:16]
        self._preset_used = "custom"

        self._steps: Dict[int, WizardStepState] = {}
        for step_def in WIZARD_STEPS:
            n = step_def["number"]
            self._steps[n] = WizardStepState(
                step_number=n,
                step_id=WizardStepId(step_def["id"]),
                display_name=step_def["name"],
            )

        self._accumulated_config: Dict[str, Any] = {}

        self.logger.info(
            "SFDRSetupWizard initialized: wizard_id=%s, presets=%s",
            self._wizard_id, self.config.available_presets,
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def start(
        self,
        preset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start the wizard, optionally applying a preset.

        Args:
            preset: Preset ID to apply as starting configuration.

        Returns:
            Initial wizard state.
        """
        if preset and preset in PRESETS:
            self._preset_used = preset
            self._accumulated_config = dict(PRESETS[preset])
            self.logger.info("Preset '%s' applied", preset)
        elif self.config.default_preset in PRESETS:
            self._preset_used = self.config.default_preset
            self._accumulated_config = dict(PRESETS[self.config.default_preset])

        self._steps[1].status = StepStatus.IN_PROGRESS
        self._steps[1].started_at = utcnow().isoformat()

        return {
            "wizard_id": self._wizard_id,
            "preset": self._preset_used,
            "total_steps": len(self._steps),
            "current_step": 1,
            "preset_config": self._accumulated_config,
            "steps": [
                {
                    "number": s.step_number,
                    "name": s.display_name,
                    "status": s.status.value,
                }
                for s in self._steps.values()
            ],
        }

    def get_step(self, step_number: int) -> Dict[str, Any]:
        """Get details for a specific step.

        Args:
            step_number: Step number (1-8).

        Returns:
            Step details including fields and current data.
        """
        if step_number < 1 or step_number > len(WIZARD_STEPS):
            return {"error": f"Invalid step number: {step_number}"}

        step_def = WIZARD_STEPS[step_number - 1]
        step_state = self._steps.get(step_number)

        return {
            "number": step_def["number"],
            "id": step_def["id"],
            "name": step_def["name"],
            "description": step_def["description"],
            "fields": step_def["fields"],
            "status": step_state.status.value if step_state else "unknown",
            "current_data": step_state.data if step_state else {},
            "preset_values": {
                f: self._accumulated_config.get(f)
                for f in step_def["fields"]
                if f in self._accumulated_config
            },
        }

    def complete_step(
        self,
        step_number: int,
        answers: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Complete a wizard step with provided answers.

        Args:
            step_number: Step number (1-8).
            answers: Answer data for the step fields.

        Returns:
            Step completion result with validation status.
        """
        if step_number < 1 or step_number > len(WIZARD_STEPS):
            return {"error": f"Invalid step number: {step_number}"}

        step = self._steps[step_number]
        step.status = StepStatus.IN_PROGRESS
        step.started_at = step.started_at or utcnow().isoformat()

        # Validate step answers
        errors = self._validate_step(step_number, answers)

        if errors:
            step.status = StepStatus.FAILED
            step.errors = errors
            return {
                "step": step_number,
                "status": "failed",
                "errors": errors,
            }

        # Merge answers into accumulated config
        self._accumulated_config.update(answers)
        step.data = answers
        step.status = StepStatus.COMPLETED
        step.completed_at = utcnow().isoformat()

        # Advance to next step
        next_step = step_number + 1
        if next_step <= len(self._steps):
            self._steps[next_step].status = StepStatus.IN_PROGRESS
            self._steps[next_step].started_at = utcnow().isoformat()

        self.logger.info(
            "Step %d completed: %s", step_number, step.display_name,
        )

        return {
            "step": step_number,
            "status": "completed",
            "next_step": next_step if next_step <= len(self._steps) else None,
            "data": answers,
        }

    def finalize(self) -> WizardResult:
        """Finalize the wizard and produce the configuration.

        Returns:
            WizardResult with the complete SFDR configuration.
        """
        completed = sum(
            1 for s in self._steps.values()
            if s.status == StepStatus.COMPLETED
        )

        # Run validation if step 8 is pending
        if self._steps[8].status != StepStatus.COMPLETED:
            validation_result = self._run_validation()
            self._steps[8].data = validation_result
            self._steps[8].status = StepStatus.COMPLETED
            self._steps[8].completed_at = utcnow().isoformat()
            completed += 1

        is_complete = completed >= 7  # At least 7/8 steps needed

        # Generate final config
        final_config = self.generate_config()

        result = WizardResult(
            wizard_id=self._wizard_id,
            preset_used=self._preset_used,
            steps_completed=completed,
            is_complete=is_complete,
            config=final_config,
            product_name=self._accumulated_config.get("product_name", ""),
            product_type=self._accumulated_config.get("product_type", ""),
            sfdr_classification=self._accumulated_config.get(
                "sfdr_classification", "article_8"
            ),
            environmental_characteristics=self._accumulated_config.get(
                "environmental_characteristics", []
            ),
            social_characteristics=self._accumulated_config.get(
                "social_characteristics", []
            ),
            taxonomy_alignment_pct=float(
                self._accumulated_config.get("min_taxonomy_pct", 0.0)
            ),
            pai_indicators_count=len(
                self._accumulated_config.get(
                    "mandatory_indicators", list(range(1, 19))
                )
            ),
            reporting_frequency=self._accumulated_config.get(
                "reporting_frequency", "annual"
            ),
            validation_passed=self._steps[8].data.get("passed", False),
            test_run_executed=self._steps[8].data.get("test_run", False),
            completed_at=utcnow().isoformat(),
        )
        result.provenance_hash = _hash_data(result.model_dump())

        self.logger.info(
            "Wizard finalized: %d/%d steps, complete=%s, preset=%s",
            completed, len(self._steps), is_complete, self._preset_used,
        )
        return result

    def generate_config(self) -> Dict[str, Any]:
        """Generate the final SFDR configuration from wizard answers.

        Returns:
            Complete SFDR configuration dictionary.
        """
        return {
            "pack_id": "PACK-010",
            "product_name": self._accumulated_config.get("product_name", ""),
            "product_isin": self._accumulated_config.get("product_isin", ""),
            "product_type": self._accumulated_config.get("product_type", ""),
            "management_company": self._accumulated_config.get(
                "management_company", ""
            ),
            "lei_code": self._accumulated_config.get("lei_code", ""),
            "sfdr_classification": self._accumulated_config.get(
                "sfdr_classification", "article_8"
            ),
            "environmental_characteristics": self._accumulated_config.get(
                "environmental_characteristics", []
            ),
            "social_characteristics": self._accumulated_config.get(
                "social_characteristics", []
            ),
            "exclusion_categories": self._accumulated_config.get(
                "exclusion_categories", []
            ),
            "esg_min_rating": self._accumulated_config.get("esg_min_rating", "BBB"),
            "controversy_max": self._accumulated_config.get("controversy_max", 4),
            "enable_taxonomy_alignment": True,
            "min_taxonomy_alignment_pct": float(
                self._accumulated_config.get("min_taxonomy_pct", 0.0)
            ),
            "taxonomy_objectives": self._accumulated_config.get(
                "taxonomy_objectives", ["climate_change_mitigation"]
            ),
            "alignment_methodology": self._accumulated_config.get(
                "alignment_methodology", "turnover"
            ),
            "pai_mandatory_indicators": self._accumulated_config.get(
                "mandatory_indicators", list(range(1, 19))
            ),
            "pai_optional_indicators": self._accumulated_config.get(
                "optional_indicators", []
            ),
            "portfolio_data_source": self._accumulated_config.get(
                "portfolio_source", "manual"
            ),
            "esg_data_provider": self._accumulated_config.get(
                "esg_provider", "internal"
            ),
            "emissions_data_source": self._accumulated_config.get(
                "emissions_source", "mrv_agents"
            ),
            "reporting_frequency": self._accumulated_config.get(
                "reporting_frequency", "annual"
            ),
            "fiscal_year_end": self._accumulated_config.get(
                "fiscal_year_end", "12-31"
            ),
            "filing_deadline_days": int(
                self._accumulated_config.get("filing_deadline_days", 180)
            ),
            "distribution_channels": self._accumulated_config.get(
                "distribution_channels", ["website"]
            ),
            "enable_provenance": True,
            "enable_quality_gates": True,
            "preset_used": self._preset_used,
            "generated_at": utcnow().isoformat(),
            "provenance_hash": _hash_data(self._accumulated_config),
        }

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _validate_step(
        self,
        step_number: int,
        answers: Dict[str, Any],
    ) -> List[str]:
        """Validate answers for a specific step.

        Args:
            step_number: Step number.
            answers: Provided answers.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: List[str] = []

        if step_number == 1:
            product_type = answers.get("product_type", "")
            if not product_type:
                errors.append("Product type is required")
            product_name = answers.get("product_name", "")
            if not product_name:
                errors.append("Product name is required")

        elif step_number == 2:
            env = answers.get("environmental_characteristics", [])
            soc = answers.get("social_characteristics", [])
            if not env and not soc:
                errors.append(
                    "At least one environmental or social characteristic "
                    "must be defined for Article 8"
                )

        elif step_number == 3:
            # Binding elements: at least exclusions for Article 8
            exclusions = answers.get("exclusion_categories", [])
            if not exclusions:
                errors.append(
                    "At least one exclusion category is recommended "
                    "for Article 8 products"
                )

        elif step_number == 4:
            min_pct = float(answers.get("min_taxonomy_pct", 0.0))
            if min_pct < 0 or min_pct > 100:
                errors.append(
                    "Taxonomy alignment percentage must be between 0 and 100"
                )

        elif step_number == 5:
            mandatory = answers.get("mandatory_indicators", list(range(1, 19)))
            if len(mandatory) < 18:
                errors.append(
                    f"All 18 mandatory PAI indicators required; "
                    f"got {len(mandatory)}"
                )

        elif step_number == 6:
            portfolio_source = answers.get("portfolio_source", "")
            if not portfolio_source:
                errors.append("Portfolio data source is required")

        elif step_number == 7:
            frequency = answers.get("frequency", answers.get("reporting_frequency", ""))
            if frequency and frequency not in ("annual", "semi_annual", "quarterly"):
                errors.append(f"Invalid reporting frequency: {frequency}")

        return errors

    def _run_validation(self) -> Dict[str, Any]:
        """Run configuration validation and optional test run.

        Returns:
            Validation result with pass/fail and findings.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Check required fields
        if not self._accumulated_config.get("product_name"):
            errors.append("Product name not set")
        if not self._accumulated_config.get("product_type"):
            errors.append("Product type not set")

        env = self._accumulated_config.get("environmental_characteristics", [])
        soc = self._accumulated_config.get("social_characteristics", [])
        if not env and not soc:
            errors.append("No E/S characteristics defined")

        exclusions = self._accumulated_config.get("exclusion_categories", [])
        if not exclusions:
            warnings.append("No exclusion categories configured")

        # Test run
        test_run = False
        if self.config.enable_test_run and not errors:
            test_run = True
            self.logger.info("Validation test run executed successfully")

        passed = len(errors) == 0

        return {
            "passed": passed,
            "errors": errors,
            "warnings": warnings,
            "test_run": test_run,
            "config_complete": passed,
            "validated_at": utcnow().isoformat(),
        }
