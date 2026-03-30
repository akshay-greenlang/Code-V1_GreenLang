# -*- coding: utf-8 -*-
"""
PackSetupWizard - Interactive Setup for PACK-017 ESRS Full Coverage Pack
==========================================================================

Interactive command-line wizard that guides users through initial pack
configuration. Detects sector from NACE codes, suggests appropriate presets,
validates configuration inputs, and generates custom pack_config.yaml files
optimized for the user's organization profile.

Methods:
    - run_setup()           -- Execute full interactive setup wizard
    - validate_config()     -- Validate configuration inputs
    - detect_sector()       -- Detect sector from NACE code
    - suggest_preset()      -- Suggest preset based on sector detection
    - generate_config()     -- Generate pack_config.yaml from inputs
    - load_preset()         -- Load a sector-specific preset
    - customize_preset()    -- Customize preset with user inputs

Sector Presets Available:
    - manufacturing         -- Industrial manufacturing (NACE C)
    - financial_services    -- Banks, insurance, asset management (NACE K)
    - energy                -- Power generation, utilities (NACE D)
    - retail                -- Retail trade (NACE G)
    - technology            -- IT services, software (NACE J)
    - multi_sector          -- Conglomerate or diversified businesses

NACE Code Detection:
    - A: Agriculture, forestry and fishing
    - B: Mining and quarrying
    - C: Manufacturing
    - D: Electricity, gas, steam and air conditioning supply
    - E: Water supply; sewerage, waste management
    - F: Construction
    - G: Wholesale and retail trade
    - H: Transportation and storage
    - I: Accommodation and food service activities
    - J: Information and communication
    - K: Financial and insurance activities
    - L: Real estate activities
    - M: Professional, scientific and technical activities
    - N: Administrative and support service activities

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-017 ESRS Full Coverage Pack
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

import yaml
from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

class SectorType(str, Enum):
    """Business sector types."""

    MANUFACTURING = "manufacturing"
    FINANCIAL_SERVICES = "financial_services"
    ENERGY = "energy"
    RETAIL = "retail"
    TECHNOLOGY = "technology"
    MULTI_SECTOR = "multi_sector"
    UNKNOWN = "unknown"

class MaterialityLevel(str, Enum):
    """ESRS topic materiality levels."""

    HIGHLY_MATERIAL = "HIGHLY_MATERIAL"
    MATERIAL = "MATERIAL"
    MAY_BE_MATERIAL = "MAY_BE_MATERIAL"
    NOT_MATERIAL = "NOT_MATERIAL"
    CONTEXT_DEPENDENT = "CONTEXT_DEPENDENT"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class WizardConfig(BaseModel):
    """Configuration for the setup wizard."""

    interactive_mode: bool = Field(default=True)
    auto_detect_sector: bool = Field(default=True)
    validate_nace_code: bool = Field(default=True)
    suggest_presets: bool = Field(default=True)
    generate_output_file: bool = Field(default=True)
    output_dir: Optional[Path] = Field(None)

class OrganizationProfile(BaseModel):
    """Organization profile for setup wizard."""

    organization_name: str = Field(default="")
    nace_code: str = Field(default="")
    nace_division_description: str = Field(default="")
    sector: SectorType = Field(default=SectorType.UNKNOWN)
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    has_subsidiaries: bool = Field(default=False)
    employee_count: Optional[int] = Field(None, ge=0)
    revenue_eur_million: Optional[float] = Field(None, ge=0.0)
    countries_of_operation: List[str] = Field(default_factory=list)

    @field_validator("nace_code")
    @classmethod
    def validate_nace_code(cls, v: str) -> str:
        """Validate NACE code format."""
        if not v:
            return v
        if len(v) < 1:
            raise ValueError("NACE code must be at least 1 character")
        if v[0] not in "ABCDEFGHIJKLMN":
            raise ValueError(f"Invalid NACE division: {v[0]}")
        return v.upper()

class MaterialityPreferences(BaseModel):
    """User preferences for materiality assessment."""

    e1_climate_change: MaterialityLevel = Field(default=MaterialityLevel.MATERIAL)
    e2_pollution: MaterialityLevel = Field(default=MaterialityLevel.MAY_BE_MATERIAL)
    e3_water_marine_resources: MaterialityLevel = Field(default=MaterialityLevel.MAY_BE_MATERIAL)
    e4_biodiversity_ecosystems: MaterialityLevel = Field(default=MaterialityLevel.MAY_BE_MATERIAL)
    e5_resource_circular_economy: MaterialityLevel = Field(default=MaterialityLevel.MATERIAL)
    s1_own_workforce: MaterialityLevel = Field(default=MaterialityLevel.MATERIAL)
    s2_workers_value_chain: MaterialityLevel = Field(default=MaterialityLevel.MATERIAL)
    s3_affected_communities: MaterialityLevel = Field(default=MaterialityLevel.MAY_BE_MATERIAL)
    s4_consumers_end_users: MaterialityLevel = Field(default=MaterialityLevel.CONTEXT_DEPENDENT)
    g1_business_conduct: MaterialityLevel = Field(default=MaterialityLevel.MATERIAL)

class SetupResult(BaseModel):
    """Result of setup wizard execution."""

    wizard_id: str = Field(default_factory=_new_uuid)
    status: SetupStatus = Field(default=SetupStatus.NOT_STARTED)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    organization: Optional[OrganizationProfile] = Field(None)
    suggested_preset: Optional[str] = Field(None)
    selected_preset: Optional[str] = Field(None)
    config_file_path: Optional[str] = Field(None)
    validation_errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# NACE Code Mapping
# ---------------------------------------------------------------------------

NACE_DIVISIONS: Dict[str, Dict[str, Any]] = {
    "A": {
        "description": "Agriculture, forestry and fishing",
        "sector": SectorType.MULTI_SECTOR,
        "suggested_preset": "multi_sector",
    },
    "B": {
        "description": "Mining and quarrying",
        "sector": SectorType.MANUFACTURING,
        "suggested_preset": "manufacturing",
    },
    "C": {
        "description": "Manufacturing",
        "sector": SectorType.MANUFACTURING,
        "suggested_preset": "manufacturing",
    },
    "D": {
        "description": "Electricity, gas, steam and air conditioning supply",
        "sector": SectorType.ENERGY,
        "suggested_preset": "energy",
    },
    "E": {
        "description": "Water supply; sewerage, waste management",
        "sector": SectorType.MULTI_SECTOR,
        "suggested_preset": "multi_sector",
    },
    "F": {
        "description": "Construction",
        "sector": SectorType.MULTI_SECTOR,
        "suggested_preset": "multi_sector",
    },
    "G": {
        "description": "Wholesale and retail trade",
        "sector": SectorType.RETAIL,
        "suggested_preset": "retail",
    },
    "H": {
        "description": "Transportation and storage",
        "sector": SectorType.MULTI_SECTOR,
        "suggested_preset": "multi_sector",
    },
    "I": {
        "description": "Accommodation and food service activities",
        "sector": SectorType.RETAIL,
        "suggested_preset": "retail",
    },
    "J": {
        "description": "Information and communication",
        "sector": SectorType.TECHNOLOGY,
        "suggested_preset": "technology",
    },
    "K": {
        "description": "Financial and insurance activities",
        "sector": SectorType.FINANCIAL_SERVICES,
        "suggested_preset": "financial_services",
    },
    "L": {
        "description": "Real estate activities",
        "sector": SectorType.MULTI_SECTOR,
        "suggested_preset": "multi_sector",
    },
    "M": {
        "description": "Professional, scientific and technical activities",
        "sector": SectorType.TECHNOLOGY,
        "suggested_preset": "technology",
    },
    "N": {
        "description": "Administrative and support service activities",
        "sector": SectorType.MULTI_SECTOR,
        "suggested_preset": "multi_sector",
    },
}

AVAILABLE_PRESETS: List[str] = [
    "manufacturing",
    "financial_services",
    "energy",
    "retail",
    "technology",
    "multi_sector",
]

# ---------------------------------------------------------------------------
# PackSetupWizard
# ---------------------------------------------------------------------------

class PackSetupWizard:
    """Interactive setup wizard for PACK-017 ESRS Full Coverage Pack.

    Guides users through configuration, detects sector from NACE codes,
    suggests appropriate presets, and generates custom pack configuration.

    Attributes:
        config: Wizard configuration.
        base_path: Pack base directory path.

    Example:
        >>> wizard = PackSetupWizard()
        >>> result = wizard.run_setup()
        >>> assert result.status == SetupStatus.COMPLETED
    """

    def __init__(
        self,
        config: Optional[WizardConfig] = None,
        base_path: Optional[Path] = None,
    ) -> None:
        """Initialize PackSetupWizard."""
        self.config = config or WizardConfig()
        if base_path is None:
            self.base_path = Path(__file__).parent.parent
        else:
            self.base_path = base_path

        self.presets_dir = self.base_path / "config" / "presets"
        self.output_dir = self.config.output_dir or (self.base_path / "config")

        logger.info(
            "PackSetupWizard initialized (base_path=%s)", self.base_path
        )

    def run_setup(self) -> SetupResult:
        """Execute full interactive setup wizard.

        Returns:
            SetupResult with generated configuration and status.
        """
        result = SetupResult(
            started_at=utcnow(),
            status=SetupStatus.IN_PROGRESS,
        )

        try:
            print("\n" + "=" * 70)
            print("  PACK-017 ESRS Full Coverage Pack - Setup Wizard")
            print("=" * 70)
            print("\nWelcome! This wizard will help you configure the ESRS pack")
            print("for your organization.\n")

            # Step 1: Collect organization profile
            profile = self._collect_organization_profile()
            result.organization = profile

            # Step 2: Detect sector
            if self.config.auto_detect_sector and profile.nace_code:
                detected_sector = self.detect_sector(profile.nace_code)
                profile.sector = detected_sector
                print(f"\nDetected sector: {detected_sector.value}")

            # Step 3: Suggest preset
            if self.config.suggest_presets:
                preset = self.suggest_preset(profile)
                result.suggested_preset = preset
                print(f"Suggested preset: {preset}")

                if self.config.interactive_mode:
                    use_suggested = self._prompt_yes_no(
                        f"\nUse suggested preset '{preset}'?", default=True
                    )
                    if use_suggested:
                        result.selected_preset = preset
                    else:
                        result.selected_preset = self._prompt_preset_selection()
                else:
                    result.selected_preset = preset
            else:
                result.selected_preset = self._prompt_preset_selection()

            # Step 4: Load and customize preset
            if result.selected_preset:
                config_data = self.load_preset(result.selected_preset)

                if self.config.interactive_mode:
                    customize = self._prompt_yes_no(
                        "\nCustomize materiality settings?", default=False
                    )
                    if customize:
                        materiality = self._collect_materiality_preferences()
                        config_data = self._apply_materiality_customization(
                            config_data, materiality
                        )

                # Update with organization details
                config_data = self._apply_organization_details(
                    config_data, profile
                )

                # Step 5: Validate configuration
                validation_errors = self.validate_config(config_data)
                result.validation_errors = validation_errors

                if validation_errors:
                    print("\n⚠️  Configuration validation found issues:")
                    for error in validation_errors:
                        print(f"  - {error}")
                    result.warnings.append(
                        f"Configuration has {len(validation_errors)} validation issues"
                    )

                # Step 6: Generate config file
                if self.config.generate_output_file:
                    config_path = self.generate_config(config_data, profile)
                    result.config_file_path = str(config_path)
                    print(f"\n✓ Configuration saved to: {config_path}")

            result.status = SetupStatus.COMPLETED
            result.completed_at = utcnow()

            print("\n" + "=" * 70)
            print("  Setup Complete!")
            print("=" * 70)
            print(f"\nConfiguration ID: {result.wizard_id}")
            if result.config_file_path:
                print(f"Config file: {result.config_file_path}")
            print("\nNext steps:")
            print("  1. Review the generated configuration file")
            print("  2. Run health check: python -m integrations.health_check")
            print("  3. Start ESRS data collection and reporting\n")

        except KeyboardInterrupt:
            result.status = SetupStatus.FAILED
            result.validation_errors.append("Setup interrupted by user")
            print("\n\nSetup cancelled by user.")

        except Exception as exc:
            result.status = SetupStatus.FAILED
            result.validation_errors.append(str(exc))
            logger.error("Setup wizard failed: %s", str(exc), exc_info=True)
            print(f"\n❌ Setup failed: {str(exc)}")

        if result.started_at:
            result.duration_ms = (
                utcnow() - result.started_at
            ).total_seconds() * 1000

        result.provenance_hash = _compute_hash(result)
        return result

    def detect_sector(self, nace_code: str) -> SectorType:
        """Detect sector from NACE code.

        Args:
            nace_code: NACE code (e.g., "C", "C10", "C10.1").

        Returns:
            Detected SectorType.
        """
        if not nace_code:
            return SectorType.UNKNOWN

        # Get first character (division)
        division = nace_code[0].upper()
        nace_info = NACE_DIVISIONS.get(division)

        if nace_info:
            logger.info(
                "Detected sector %s from NACE code %s",
                nace_info["sector"].value,
                nace_code,
            )
            return nace_info["sector"]

        logger.warning("Unknown NACE code: %s", nace_code)
        return SectorType.UNKNOWN

    def suggest_preset(self, profile: OrganizationProfile) -> str:
        """Suggest preset based on sector detection.

        Args:
            profile: Organization profile with sector information.

        Returns:
            Suggested preset name.
        """
        if profile.nace_code:
            division = profile.nace_code[0].upper()
            nace_info = NACE_DIVISIONS.get(division)
            if nace_info:
                preset = nace_info["suggested_preset"]
                logger.info("Suggested preset: %s", preset)
                return preset

        # Default to multi_sector if no sector detected
        return "multi_sector"

    def load_preset(self, preset_name: str) -> Dict[str, Any]:
        """Load a sector-specific preset.

        Args:
            preset_name: Name of preset to load.

        Returns:
            Dict with preset configuration data.

        Raises:
            FileNotFoundError: If preset file not found.
        """
        preset_path = self.presets_dir / f"{preset_name}.yaml"

        if not preset_path.exists():
            raise FileNotFoundError(f"Preset not found: {preset_path}")

        with open(preset_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        logger.info("Loaded preset: %s", preset_name)
        return data

    def validate_config(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate configuration inputs.

        Args:
            config_data: Configuration dictionary to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        # Check required top-level keys
        required_keys = ["preset_name", "pack_id", "materiality"]
        for key in required_keys:
            if key not in config_data:
                errors.append(f"Missing required key: {key}")

        # Validate materiality section
        if "materiality" in config_data:
            materiality = config_data["materiality"]
            expected_topics = [
                "e1_climate_change",
                "e2_pollution",
                "e3_water_marine_resources",
                "e4_biodiversity_ecosystems",
                "e5_resource_use_circular_economy",
                "s1_own_workforce",
                "s2_workers_in_value_chain",
                "s3_affected_communities",
                "s4_consumers_end_users",
                "g1_business_conduct",
            ]
            for topic in expected_topics:
                if topic not in materiality:
                    errors.append(f"Missing materiality assessment for: {topic}")

        # Validate enabled standards have disclosure_requirements
        for standard_key in config_data:
            if standard_key.startswith(("e", "s", "g")) and "_" in standard_key:
                standard_config = config_data.get(standard_key, {})
                if isinstance(standard_config, dict):
                    if standard_config.get("enabled") is True:
                        if not standard_config.get("disclosure_requirements"):
                            errors.append(
                                f"Enabled standard {standard_key} has no disclosure_requirements"
                            )

        logger.info("Validation completed: %d errors", len(errors))
        return errors

    def generate_config(
        self,
        config_data: Dict[str, Any],
        profile: OrganizationProfile,
    ) -> Path:
        """Generate pack_config.yaml from inputs.

        Args:
            config_data: Configuration data dictionary.
            profile: Organization profile.

        Returns:
            Path to generated configuration file.
        """
        # Update organization-specific fields
        config_data["organization_name"] = profile.organization_name
        config_data["reporting_year"] = profile.reporting_year
        config_data["nace_codes"] = [profile.nace_code] if profile.nace_code else []

        # Generate output filename
        org_slug = profile.organization_name.lower().replace(" ", "_")[:20]
        timestamp = utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"pack_config_{org_slug}_{timestamp}.yaml"
        output_path = self.output_dir / filename

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write YAML file
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(
                config_data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        logger.info("Generated config file: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Internal helper methods
    # ------------------------------------------------------------------

    def _collect_organization_profile(self) -> OrganizationProfile:
        """Collect organization profile from user input."""
        print("\n--- Organization Profile ---\n")

        profile = OrganizationProfile()

        profile.organization_name = self._prompt_text(
            "Organization name", required=True
        )
        profile.nace_code = self._prompt_text(
            "NACE code (e.g., C, C10, K64.19)", required=False
        ).upper()

        if profile.nace_code:
            division = profile.nace_code[0]
            nace_info = NACE_DIVISIONS.get(division)
            if nace_info:
                profile.nace_division_description = nace_info["description"]
                print(f"  → {nace_info['description']}")

        profile.reporting_year = self._prompt_int(
            "Reporting year", default=2025, min_val=2020, max_val=2030
        )

        profile.has_subsidiaries = self._prompt_yes_no(
            "Does the organization have subsidiaries?", default=False
        )

        profile.employee_count = self._prompt_int(
            "Number of employees (optional)", required=False, min_val=0
        )

        profile.revenue_eur_million = self._prompt_float(
            "Revenue in EUR million (optional)", required=False, min_val=0.0
        )

        return profile

    def _collect_materiality_preferences(self) -> MaterialityPreferences:
        """Collect materiality preferences from user."""
        print("\n--- Materiality Assessment ---\n")
        print("Rate materiality: 1=HIGHLY_MATERIAL, 2=MATERIAL, 3=MAY_BE_MATERIAL, 4=NOT_MATERIAL\n")

        levels = [
            MaterialityLevel.HIGHLY_MATERIAL,
            MaterialityLevel.MATERIAL,
            MaterialityLevel.MAY_BE_MATERIAL,
            MaterialityLevel.NOT_MATERIAL,
        ]

        prefs = MaterialityPreferences()

        prefs.e1_climate_change = levels[
            self._prompt_int("E1 Climate Change", default=1, min_val=1, max_val=4) - 1
        ]
        prefs.e2_pollution = levels[
            self._prompt_int("E2 Pollution", default=2, min_val=1, max_val=4) - 1
        ]
        prefs.e3_water_marine_resources = levels[
            self._prompt_int("E3 Water & Marine", default=3, min_val=1, max_val=4) - 1
        ]
        prefs.e4_biodiversity_ecosystems = levels[
            self._prompt_int("E4 Biodiversity", default=3, min_val=1, max_val=4) - 1
        ]
        prefs.e5_resource_circular_economy = levels[
            self._prompt_int("E5 Circular Economy", default=2, min_val=1, max_val=4) - 1
        ]
        prefs.s1_own_workforce = levels[
            self._prompt_int("S1 Own Workforce", default=2, min_val=1, max_val=4) - 1
        ]
        prefs.s2_workers_value_chain = levels[
            self._prompt_int("S2 Value Chain Workers", default=2, min_val=1, max_val=4) - 1
        ]
        prefs.s3_affected_communities = levels[
            self._prompt_int("S3 Affected Communities", default=3, min_val=1, max_val=4) - 1
        ]
        prefs.s4_consumers_end_users = levels[
            self._prompt_int("S4 Consumers", default=3, min_val=1, max_val=4) - 1
        ]
        prefs.g1_business_conduct = levels[
            self._prompt_int("G1 Business Conduct", default=2, min_val=1, max_val=4) - 1
        ]

        return prefs

    def _prompt_preset_selection(self) -> str:
        """Prompt user to select a preset."""
        print("\n--- Available Presets ---\n")
        for i, preset in enumerate(AVAILABLE_PRESETS, 1):
            print(f"  {i}. {preset}")

        choice = self._prompt_int(
            "\nSelect preset (1-6)",
            default=6,
            min_val=1,
            max_val=len(AVAILABLE_PRESETS),
        )
        return AVAILABLE_PRESETS[choice - 1]

    def _apply_materiality_customization(
        self,
        config_data: Dict[str, Any],
        materiality: MaterialityPreferences,
    ) -> Dict[str, Any]:
        """Apply materiality customization to config."""
        config_data["materiality"]["e1_climate_change"] = materiality.e1_climate_change.value
        config_data["materiality"]["e2_pollution"] = materiality.e2_pollution.value
        config_data["materiality"]["e3_water_marine_resources"] = materiality.e3_water_marine_resources.value
        config_data["materiality"]["e4_biodiversity_ecosystems"] = materiality.e4_biodiversity_ecosystems.value
        config_data["materiality"]["e5_resource_use_circular_economy"] = materiality.e5_resource_circular_economy.value
        config_data["materiality"]["s1_own_workforce"] = materiality.s1_own_workforce.value
        config_data["materiality"]["s2_workers_in_value_chain"] = materiality.s2_workers_value_chain.value
        config_data["materiality"]["s3_affected_communities"] = materiality.s3_affected_communities.value
        config_data["materiality"]["s4_consumers_end_users"] = materiality.s4_consumers_end_users.value
        config_data["materiality"]["g1_business_conduct"] = materiality.g1_business_conduct.value
        return config_data

    def _apply_organization_details(
        self,
        config_data: Dict[str, Any],
        profile: OrganizationProfile,
    ) -> Dict[str, Any]:
        """Apply organization details to config."""
        config_data["organization_name"] = profile.organization_name
        config_data["reporting_year"] = profile.reporting_year
        config_data["nace_codes"] = [profile.nace_code] if profile.nace_code else []
        return config_data

    def _prompt_text(
        self,
        prompt: str,
        required: bool = True,
        default: Optional[str] = None,
    ) -> str:
        """Prompt user for text input."""
        prompt_str = f"{prompt}"
        if default:
            prompt_str += f" [{default}]"
        prompt_str += ": "

        while True:
            value = input(prompt_str).strip()
            if not value and default:
                return default
            if not value and not required:
                return ""
            if value:
                return value
            print("  ⚠️  This field is required.")

    def _prompt_int(
        self,
        prompt: str,
        default: Optional[int] = None,
        required: bool = True,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
    ) -> Optional[int]:
        """Prompt user for integer input."""
        prompt_str = f"{prompt}"
        if default is not None:
            prompt_str += f" [{default}]"
        prompt_str += ": "

        while True:
            value_str = input(prompt_str).strip()
            if not value_str and default is not None:
                return default
            if not value_str and not required:
                return None

            try:
                value = int(value_str)
                if min_val is not None and value < min_val:
                    print(f"  ⚠️  Value must be >= {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"  ⚠️  Value must be <= {max_val}")
                    continue
                return value
            except ValueError:
                print("  ⚠️  Please enter a valid integer.")

    def _prompt_float(
        self,
        prompt: str,
        default: Optional[float] = None,
        required: bool = True,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> Optional[float]:
        """Prompt user for float input."""
        prompt_str = f"{prompt}"
        if default is not None:
            prompt_str += f" [{default}]"
        prompt_str += ": "

        while True:
            value_str = input(prompt_str).strip()
            if not value_str and default is not None:
                return default
            if not value_str and not required:
                return None

            try:
                value = float(value_str)
                if min_val is not None and value < min_val:
                    print(f"  ⚠️  Value must be >= {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"  ⚠️  Value must be <= {max_val}")
                    continue
                return value
            except ValueError:
                print("  ⚠️  Please enter a valid number.")

    def _prompt_yes_no(self, prompt: str, default: bool = True) -> bool:
        """Prompt user for yes/no input."""
        default_str = "Y/n" if default else "y/N"
        prompt_str = f"{prompt} [{default_str}]: "

        while True:
            value = input(prompt_str).strip().lower()
            if not value:
                return default
            if value in ("y", "yes"):
                return True
            if value in ("n", "no"):
                return False
            print("  ⚠️  Please enter 'y' or 'n'.")

# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for setup wizard."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    wizard = PackSetupWizard()
    result = wizard.run_setup()

    sys.exit(0 if result.status == SetupStatus.COMPLETED else 1)

if __name__ == "__main__":
    main()
