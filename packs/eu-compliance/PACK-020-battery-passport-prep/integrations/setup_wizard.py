# -*- coding: utf-8 -*-
"""
BatteryPassportSetupWizard - Configuration Setup for PACK-020
=================================================================

Interactive setup wizard that guides users through initial battery passport
pack configuration. Detects battery category, suggests appropriate presets,
validates configuration inputs, estimates resource requirements, and
generates custom pack_config.yaml files optimized for the user's battery
type and manufacturing profile.

Methods:
    - create_configuration()       -- Generate full configuration programmatically
    - validate_configuration()     -- Validate configuration inputs
    - get_category_defaults()      -- Get default settings for a battery category
    - estimate_requirements()      -- Estimate data and compliance requirements
    - get_preset()                 -- Load a category-specific preset

Battery Category Presets:
    - ev                          -- Electric Vehicle batteries (Art 7, 8, 10, 39, 77)
    - lmt                         -- Light Means of Transport (e-bikes, scooters)
    - industrial                  -- Industrial batteries (>2kWh stationary)
    - sli                         -- Starting, Lighting, Ignition (automotive 12V)
    - portable                    -- Portable batteries (consumer electronics)
    - stationary_storage          -- Stationary Energy Storage Systems

Legal References:
    - Regulation (EU) 2023/1542, Art 2 (Battery categories)
    - Art 7: Carbon footprint (EV, LMT, industrial from 2025/2028)
    - Art 8: Recycled content (EV, LMT, industrial, SLI from 2031)
    - Art 77: Digital battery passport (EV, LMT, industrial from 2027)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-020 Battery Passport Prep Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
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


class SetupStatus(str, Enum):
    """Setup wizard execution status."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class BatteryCategory(str, Enum):
    """EU Battery Regulation battery categories (Art 2)."""

    EV = "ev"
    LMT = "lmt"
    INDUSTRIAL = "industrial"
    SLI = "sli"
    PORTABLE = "portable"
    STATIONARY_STORAGE = "stationary_storage"


class ChemistryType(str, Enum):
    """Common battery chemistry types."""

    NMC = "NMC"
    NCA = "NCA"
    LFP = "LFP"
    LCO = "LCO"
    LTO = "LTO"
    NI_MH = "NiMH"
    LEAD_ACID = "PbA"
    SODIUM_ION = "Na-ion"
    SOLID_STATE = "Solid-State"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class WizardConfig(BaseModel):
    """Configuration for the setup wizard."""

    interactive_mode: bool = Field(default=False)
    generate_output_file: bool = Field(default=True)
    output_dir: Optional[Path] = Field(None)


class ManufacturerProfile(BaseModel):
    """Manufacturer profile for setup wizard."""

    manufacturer_name: str = Field(default="")
    battery_category: BatteryCategory = Field(default=BatteryCategory.EV)
    chemistry_type: ChemistryType = Field(default=ChemistryType.NMC)
    battery_model: str = Field(default="")
    nominal_capacity_kwh: float = Field(default=0.0, ge=0.0)
    production_country: str = Field(default="")
    production_volume_annual: int = Field(default=0, ge=0)
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    has_existing_passport: bool = Field(default=False)

    @field_validator("manufacturer_name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate manufacturer name is not empty for production use."""
        return v.strip()


class RequirementsEstimate(BaseModel):
    """Estimated data and compliance requirements."""

    battery_category: str = Field(default="")
    passport_required: bool = Field(default=False)
    passport_deadline: str = Field(default="")
    carbon_footprint_required: bool = Field(default=False)
    carbon_footprint_deadline: str = Field(default="")
    recycled_content_required: bool = Field(default=False)
    recycled_content_deadline: str = Field(default="")
    due_diligence_required: bool = Field(default=True)
    performance_requirements: bool = Field(default=True)
    labelling_required: bool = Field(default=True)
    end_of_life_obligations: bool = Field(default=True)
    estimated_passport_fields: int = Field(default=0)
    estimated_data_sources: int = Field(default=0)
    estimated_supplier_assessments: int = Field(default=0)
    complexity_level: str = Field(default="medium")


class SetupResult(BaseModel):
    """Result of setup wizard execution."""

    wizard_id: str = Field(default_factory=_new_uuid)
    status: SetupStatus = Field(default=SetupStatus.NOT_STARTED)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    profile: Optional[ManufacturerProfile] = Field(None)
    selected_preset: Optional[str] = Field(None)
    requirements: Optional[RequirementsEstimate] = Field(None)
    config_data: Dict[str, Any] = Field(default_factory=dict)
    config_file_path: Optional[str] = Field(None)
    validation_errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Category Preset Defaults
# ---------------------------------------------------------------------------

CATEGORY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    BatteryCategory.EV.value: {
        "preset_name": "ev_battery",
        "passport_required": True,
        "passport_deadline": "2027-02-18",
        "carbon_footprint_required": True,
        "carbon_footprint_deadline": "2025-02-18",
        "recycled_content_required": True,
        "recycled_content_deadline": "2031-08-18",
        "due_diligence_required": True,
        "performance_requirements": True,
        "labelling_required": True,
        "end_of_life_obligations": True,
        "estimated_passport_fields": 90,
        "estimated_data_sources": 12,
        "estimated_supplier_assessments": 50,
        "complexity_level": "high",
        "applicable_articles": [
            "Art 6", "Art 7", "Art 8", "Art 10", "Art 11", "Art 13",
            "Art 39", "Art 57", "Art 77",
        ],
        "recycled_content_targets_2031": {
            "cobalt": 16, "lithium": 6, "nickel": 6, "lead": 85,
        },
        "recycled_content_targets_2036": {
            "cobalt": 26, "lithium": 12, "nickel": 15, "lead": 85,
        },
    },
    BatteryCategory.LMT.value: {
        "preset_name": "lmt_battery",
        "passport_required": True,
        "passport_deadline": "2027-02-18",
        "carbon_footprint_required": True,
        "carbon_footprint_deadline": "2028-08-18",
        "recycled_content_required": True,
        "recycled_content_deadline": "2031-08-18",
        "due_diligence_required": True,
        "performance_requirements": True,
        "labelling_required": True,
        "end_of_life_obligations": True,
        "estimated_passport_fields": 75,
        "estimated_data_sources": 8,
        "estimated_supplier_assessments": 25,
        "complexity_level": "medium",
        "applicable_articles": [
            "Art 6", "Art 7", "Art 8", "Art 10", "Art 13",
            "Art 39", "Art 57", "Art 77",
        ],
        "recycled_content_targets_2031": {
            "cobalt": 16, "lithium": 6, "nickel": 6, "lead": 85,
        },
        "recycled_content_targets_2036": {
            "cobalt": 26, "lithium": 12, "nickel": 15, "lead": 85,
        },
    },
    BatteryCategory.INDUSTRIAL.value: {
        "preset_name": "industrial_battery",
        "passport_required": True,
        "passport_deadline": "2027-02-18",
        "carbon_footprint_required": True,
        "carbon_footprint_deadline": "2025-02-18",
        "recycled_content_required": True,
        "recycled_content_deadline": "2031-08-18",
        "due_diligence_required": True,
        "performance_requirements": True,
        "labelling_required": True,
        "end_of_life_obligations": True,
        "estimated_passport_fields": 80,
        "estimated_data_sources": 10,
        "estimated_supplier_assessments": 35,
        "complexity_level": "high",
        "applicable_articles": [
            "Art 6", "Art 7", "Art 8", "Art 10", "Art 13",
            "Art 39", "Art 57", "Art 77",
        ],
        "recycled_content_targets_2031": {
            "cobalt": 16, "lithium": 6, "nickel": 6, "lead": 85,
        },
        "recycled_content_targets_2036": {
            "cobalt": 26, "lithium": 12, "nickel": 15, "lead": 85,
        },
    },
    BatteryCategory.SLI.value: {
        "preset_name": "sli_battery",
        "passport_required": False,
        "passport_deadline": "",
        "carbon_footprint_required": False,
        "carbon_footprint_deadline": "",
        "recycled_content_required": True,
        "recycled_content_deadline": "2031-08-18",
        "due_diligence_required": True,
        "performance_requirements": True,
        "labelling_required": True,
        "end_of_life_obligations": True,
        "estimated_passport_fields": 30,
        "estimated_data_sources": 5,
        "estimated_supplier_assessments": 15,
        "complexity_level": "low",
        "applicable_articles": [
            "Art 6", "Art 8", "Art 10", "Art 13", "Art 39", "Art 57",
        ],
        "recycled_content_targets_2031": {"lead": 85},
        "recycled_content_targets_2036": {"lead": 85},
    },
    BatteryCategory.PORTABLE.value: {
        "preset_name": "portable_battery",
        "passport_required": False,
        "passport_deadline": "",
        "carbon_footprint_required": False,
        "carbon_footprint_deadline": "",
        "recycled_content_required": False,
        "recycled_content_deadline": "",
        "due_diligence_required": True,
        "performance_requirements": True,
        "labelling_required": True,
        "end_of_life_obligations": True,
        "estimated_passport_fields": 20,
        "estimated_data_sources": 4,
        "estimated_supplier_assessments": 10,
        "complexity_level": "low",
        "applicable_articles": [
            "Art 6", "Art 10", "Art 13", "Art 39", "Art 57",
        ],
        "recycled_content_targets_2031": {},
        "recycled_content_targets_2036": {},
    },
    BatteryCategory.STATIONARY_STORAGE.value: {
        "preset_name": "stationary_storage_battery",
        "passport_required": True,
        "passport_deadline": "2027-02-18",
        "carbon_footprint_required": True,
        "carbon_footprint_deadline": "2025-02-18",
        "recycled_content_required": True,
        "recycled_content_deadline": "2031-08-18",
        "due_diligence_required": True,
        "performance_requirements": True,
        "labelling_required": True,
        "end_of_life_obligations": True,
        "estimated_passport_fields": 85,
        "estimated_data_sources": 10,
        "estimated_supplier_assessments": 40,
        "complexity_level": "high",
        "applicable_articles": [
            "Art 6", "Art 7", "Art 8", "Art 10", "Art 13",
            "Art 39", "Art 57", "Art 77",
        ],
        "recycled_content_targets_2031": {
            "cobalt": 16, "lithium": 6, "nickel": 6, "lead": 85,
        },
        "recycled_content_targets_2036": {
            "cobalt": 26, "lithium": 12, "nickel": 15, "lead": 85,
        },
    },
}

AVAILABLE_PRESETS: List[str] = list(CATEGORY_DEFAULTS.keys())


# ---------------------------------------------------------------------------
# BatteryPassportSetupWizard
# ---------------------------------------------------------------------------


class BatteryPassportSetupWizard:
    """Configuration setup wizard for PACK-020 Battery Passport Prep.

    Guides users through battery category selection, estimates regulatory
    requirements, validates configuration inputs, and generates pack
    configuration files.

    Attributes:
        config: Wizard configuration.
        base_path: Pack base directory path.

    Example:
        >>> wizard = BatteryPassportSetupWizard()
        >>> result = wizard.create_configuration(profile)
        >>> assert result.status == SetupStatus.COMPLETED
    """

    def __init__(
        self,
        config: Optional[WizardConfig] = None,
        base_path: Optional[Path] = None,
    ) -> None:
        """Initialize BatteryPassportSetupWizard."""
        self.config = config or WizardConfig()
        if base_path is None:
            self.base_path = Path(__file__).parent.parent
        else:
            self.base_path = base_path

        self.presets_dir = self.base_path / "config" / "presets"
        self.output_dir = self.config.output_dir or (self.base_path / "config")

        logger.info(
            "BatteryPassportSetupWizard initialized (base_path=%s)",
            self.base_path,
        )

    def create_configuration(
        self, profile: ManufacturerProfile
    ) -> SetupResult:
        """Generate a full configuration from a manufacturer profile.

        Args:
            profile: Manufacturer profile with battery details.

        Returns:
            SetupResult with generated configuration and validation status.
        """
        result = SetupResult(
            started_at=_utcnow(),
            status=SetupStatus.IN_PROGRESS,
            profile=profile,
        )

        try:
            # Step 1: Get category defaults
            defaults = self.get_category_defaults(profile.battery_category)
            result.selected_preset = defaults.get("preset_name", "")

            # Step 2: Build configuration data
            config_data = self._build_config(profile, defaults)
            result.config_data = config_data

            # Step 3: Validate
            errors = self.validate_configuration(config_data)
            result.validation_errors = errors

            if errors:
                result.warnings.append(
                    f"Configuration has {len(errors)} validation issues"
                )

            # Step 4: Estimate requirements
            result.requirements = self.estimate_requirements(
                profile.battery_category
            )

            # Step 5: Generate config file
            if self.config.generate_output_file:
                config_path = self._write_config_file(config_data, profile)
                result.config_file_path = str(config_path)

            result.status = SetupStatus.COMPLETED
            result.completed_at = _utcnow()

        except Exception as exc:
            result.status = SetupStatus.FAILED
            result.validation_errors.append(str(exc))
            logger.error("Setup wizard failed: %s", str(exc), exc_info=True)

        if result.started_at:
            result.duration_ms = (
                _utcnow() - result.started_at
            ).total_seconds() * 1000

        result.provenance_hash = _compute_hash(result)
        return result

    def validate_configuration(
        self, config_data: Dict[str, Any]
    ) -> List[str]:
        """Validate configuration inputs against Battery Regulation requirements.

        Args:
            config_data: Configuration dictionary to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        required_keys = ["pack_id", "battery_category", "manufacturer_name"]
        for key in required_keys:
            if key not in config_data or not config_data[key]:
                errors.append(f"Missing required key: {key}")

        category = config_data.get("battery_category", "")
        if category and category not in [c.value for c in BatteryCategory]:
            errors.append(f"Invalid battery category: {category}")

        if config_data.get("passport_required"):
            passport_fields = config_data.get("passport_fields", {})
            if not passport_fields:
                errors.append("passport_required=True but no passport_fields defined")

        if config_data.get("carbon_footprint_required"):
            if not config_data.get("battery_capacity_kwh"):
                errors.append(
                    "carbon_footprint_required=True but battery_capacity_kwh not set"
                )

        if config_data.get("recycled_content_required"):
            targets = config_data.get("recycled_content_targets", {})
            if not targets:
                errors.append(
                    "recycled_content_required=True but no targets defined"
                )

        logger.info("Validation completed: %d errors", len(errors))
        return errors

    def get_category_defaults(
        self, category: BatteryCategory
    ) -> Dict[str, Any]:
        """Get default settings for a battery category.

        Args:
            category: Battery category to get defaults for.

        Returns:
            Dict with default settings for the category.
        """
        defaults = CATEGORY_DEFAULTS.get(category.value, {})
        if not defaults:
            logger.warning("No defaults for category: %s", category.value)
            return CATEGORY_DEFAULTS[BatteryCategory.EV.value]

        logger.info(
            "Category defaults: %s (%s complexity)",
            category.value, defaults.get("complexity_level", "unknown"),
        )
        return dict(defaults)

    def estimate_requirements(
        self, category: BatteryCategory
    ) -> RequirementsEstimate:
        """Estimate data and compliance requirements for a battery category.

        Args:
            category: Battery category to estimate for.

        Returns:
            RequirementsEstimate with estimated workload and deadlines.
        """
        defaults = self.get_category_defaults(category)

        estimate = RequirementsEstimate(
            battery_category=category.value,
            passport_required=defaults.get("passport_required", False),
            passport_deadline=defaults.get("passport_deadline", ""),
            carbon_footprint_required=defaults.get(
                "carbon_footprint_required", False
            ),
            carbon_footprint_deadline=defaults.get(
                "carbon_footprint_deadline", ""
            ),
            recycled_content_required=defaults.get(
                "recycled_content_required", False
            ),
            recycled_content_deadline=defaults.get(
                "recycled_content_deadline", ""
            ),
            due_diligence_required=defaults.get("due_diligence_required", True),
            performance_requirements=defaults.get(
                "performance_requirements", True
            ),
            labelling_required=defaults.get("labelling_required", True),
            end_of_life_obligations=defaults.get("end_of_life_obligations", True),
            estimated_passport_fields=defaults.get(
                "estimated_passport_fields", 0
            ),
            estimated_data_sources=defaults.get("estimated_data_sources", 0),
            estimated_supplier_assessments=defaults.get(
                "estimated_supplier_assessments", 0
            ),
            complexity_level=defaults.get("complexity_level", "medium"),
        )

        logger.info(
            "Requirements estimate: %s category, %s complexity, %d fields",
            category.value,
            estimate.complexity_level,
            estimate.estimated_passport_fields,
        )
        return estimate

    def get_preset(self, preset_name: str) -> Dict[str, Any]:
        """Load a category-specific preset by name.

        Args:
            preset_name: Preset name (battery category value).

        Returns:
            Dict with preset configuration data.

        Raises:
            ValueError: If preset not found.
        """
        if preset_name not in CATEGORY_DEFAULTS:
            available = ", ".join(AVAILABLE_PRESETS)
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {available}"
            )

        preset = dict(CATEGORY_DEFAULTS[preset_name])
        logger.info("Loaded preset: %s", preset_name)
        return preset

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_config(
        self,
        profile: ManufacturerProfile,
        defaults: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build complete configuration from profile and defaults."""
        return {
            "pack_id": "PACK-020",
            "pack_version": _MODULE_VERSION,
            "preset_name": defaults.get("preset_name", ""),
            "manufacturer_name": profile.manufacturer_name,
            "battery_category": profile.battery_category.value,
            "chemistry_type": profile.chemistry_type.value,
            "battery_model": profile.battery_model,
            "battery_capacity_kwh": profile.nominal_capacity_kwh,
            "production_country": profile.production_country,
            "production_volume_annual": profile.production_volume_annual,
            "reporting_year": profile.reporting_year,
            "passport_required": defaults.get("passport_required", False),
            "passport_deadline": defaults.get("passport_deadline", ""),
            "carbon_footprint_required": defaults.get(
                "carbon_footprint_required", False
            ),
            "carbon_footprint_deadline": defaults.get(
                "carbon_footprint_deadline", ""
            ),
            "recycled_content_required": defaults.get(
                "recycled_content_required", False
            ),
            "recycled_content_deadline": defaults.get(
                "recycled_content_deadline", ""
            ),
            "recycled_content_targets": defaults.get(
                "recycled_content_targets_2031", {}
            ),
            "due_diligence_required": defaults.get("due_diligence_required", True),
            "performance_requirements": defaults.get(
                "performance_requirements", True
            ),
            "labelling_required": defaults.get("labelling_required", True),
            "end_of_life_obligations": defaults.get(
                "end_of_life_obligations", True
            ),
            "applicable_articles": defaults.get("applicable_articles", []),
            "passport_fields": {},
            "engines_enabled": {
                "carbon_footprint": defaults.get("carbon_footprint_required", False),
                "recycled_content": defaults.get("recycled_content_required", False),
                "passport_compiler": defaults.get("passport_required", False),
                "performance": defaults.get("performance_requirements", True),
                "due_diligence": defaults.get("due_diligence_required", True),
                "labelling": defaults.get("labelling_required", True),
                "end_of_life": defaults.get("end_of_life_obligations", True),
                "conformity": True,
            },
        }

    def _write_config_file(
        self,
        config_data: Dict[str, Any],
        profile: ManufacturerProfile,
    ) -> Path:
        """Write configuration to a YAML file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        org_slug = profile.manufacturer_name.lower().replace(" ", "_")[:20]
        timestamp = _utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"pack_config_{org_slug}_{timestamp}.json"
        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, default=str)

        logger.info("Generated config file: %s", output_path)
        return output_path


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for setup wizard."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    profile = ManufacturerProfile(
        manufacturer_name="Example Battery Corp",
        battery_category=BatteryCategory.EV,
        chemistry_type=ChemistryType.NMC,
        nominal_capacity_kwh=60.0,
        production_country="DEU",
        reporting_year=2025,
    )

    wizard = BatteryPassportSetupWizard()
    result = wizard.create_configuration(profile)

    print(f"\nSetup Status: {result.status.value}")
    if result.config_file_path:
        print(f"Config file: {result.config_file_path}")
    if result.requirements:
        print(f"Complexity: {result.requirements.complexity_level}")
        print(f"Passport fields: {result.requirements.estimated_passport_fields}")
    if result.validation_errors:
        print(f"Validation errors: {len(result.validation_errors)}")

    sys.exit(0 if result.status == SetupStatus.COMPLETED else 1)


if __name__ == "__main__":
    main()
