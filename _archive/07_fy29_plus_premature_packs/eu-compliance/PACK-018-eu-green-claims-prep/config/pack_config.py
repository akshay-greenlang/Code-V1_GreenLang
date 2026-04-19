"""
PACK-018 EU Green Claims Prep Pack - Pack Configuration

Implements PackConfig (Pydantic BaseModel) with all tuneable parameters for the
EU Green Claims Prep Pack: claim scope, engine/workflow selection, evidence
retention, PEF database, eco-label registry refresh, greenwashing risk
thresholds, and CAB submission formatting.

Enums:
    ClaimScope: PRODUCT | CORPORATE | BOTH
    CommunicationChannel: PACKAGING | WEBSITE | ADVERTISING | SOCIAL_MEDIA |
                          INVESTOR_RELATIONS | PRESS_RELEASE | POINT_OF_SALE

Regulatory Context:
    - EU Green Claims Directive (COM/2023/166)
    - ECGT Directive (EU) 2024/825
    - Product Environmental Footprint (PEF) - Commission Rec. 2013/179/EU
    - ISO 14021/14024/14025 environmental label standards

Example:
    >>> config = PackConfig()
    >>> warnings = config.validate()
    >>> engine_cfg = config.get_engine_config("claim_substantiation")
    >>> print(config.config_hash)
"""

import hashlib
import json
import logging
import os
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent
PRESETS_DIR = CONFIG_DIR / "presets"
DEMO_DIR = CONFIG_DIR / "demo"

ALL_ENGINES: List[str] = [
    "claim_substantiation", "evidence_chain", "lifecycle_assessment",
    "label_certification", "greenwashing_detection", "comparative_claims",
    "trader_obligation", "green_claims_benchmark",
]

ALL_WORKFLOWS: List[str] = [
    "claim_intake", "substantiation_assessment", "evidence_collection",
    "lifecycle_scoring", "label_validation", "greenwashing_scan",
    "comparison_review", "disclosure_generation", "cab_submission",
    "periodic_review",
]

AVAILABLE_PRESETS: Dict[str, str] = {
    "manufacturing": "Manufacturing - product claims, LCA/PEF, PEFCR, DPP",
    "retail": "Retail - high-volume product claims, ECGT, label compliance",
    "financial_services": "Financial services - ESG fund claims, SFDR, offsets",
    "energy": "Energy - renewable energy, carbon claims, transition plans",
    "technology": "Technology - device/cloud claims, DPP electronics, WEEE",
    "sme": "SME simplified - reduced engines, screening LCA, secondary data",
}


# ============================================================================
# Enums
# ============================================================================

class ClaimScope(str, Enum):
    """Scope of environmental claims: PRODUCT, CORPORATE, or BOTH."""
    PRODUCT = "PRODUCT"
    CORPORATE = "CORPORATE"
    BOTH = "BOTH"


class CommunicationChannel(str, Enum):
    """Channels through which green claims are communicated (ECGD Art. 2(1))."""
    PACKAGING = "PACKAGING"
    WEBSITE = "WEBSITE"
    ADVERTISING = "ADVERTISING"
    SOCIAL_MEDIA = "SOCIAL_MEDIA"
    INVESTOR_RELATIONS = "INVESTOR_RELATIONS"
    PRESS_RELEASE = "PRESS_RELEASE"
    POINT_OF_SALE = "POINT_OF_SALE"


# ============================================================================
# PackConfig
# ============================================================================

class PackConfig(BaseModel):
    """Top-level configuration for PACK-018 EU Green Claims Prep Pack.

    Attributes:
        pack_name: Immutable pack identifier.
        version: Configuration schema version.
        sector: Target business sector.
        enabled_engines: Active engine list (subset of ALL_ENGINES).
        enabled_workflows: Active workflow list (subset of ALL_WORKFLOWS).
        claim_scope: PRODUCT, CORPORATE, or BOTH.
        communication_channels: Where claims are published.
        evidence_retention_years: Years to retain evidence (default 5).
        pef_database: PEF characterisation factor DB version (default ef_3.1).
        eco_label_registry_refresh_days: Registry re-sync interval (default 30).
        max_claims: Upper bound on claims per run (default 10000).
        max_evidence_items: Upper bound on evidence records (default 50000).
        greenwashing_risk_threshold: Score 0-100 flagging threshold (default 50).
        cab_submission_format: CAB dossier output format.
        reporting_language: ISO 639-1 code (default en).
    """

    pack_name: str = Field("PACK-018-eu-green-claims-prep", description="Pack identifier")
    version: str = Field("1.0.0", description="Config schema version")
    sector: str = Field("MANUFACTURING", description="Target business sector")
    enabled_engines: List[str] = Field(
        default_factory=lambda: list(ALL_ENGINES),
        description="Engines to activate",
    )
    enabled_workflows: List[str] = Field(
        default_factory=lambda: list(ALL_WORKFLOWS),
        description="Workflows to activate",
    )
    claim_scope: ClaimScope = Field(
        ClaimScope.BOTH, description="Product-level, corporate-level, or both",
    )
    communication_channels: List[CommunicationChannel] = Field(
        default_factory=lambda: [
            CommunicationChannel.PACKAGING,
            CommunicationChannel.WEBSITE,
            CommunicationChannel.ADVERTISING,
        ],
        description="Channels through which green claims are communicated",
    )
    evidence_retention_years: int = Field(
        5, ge=1, le=20,
        description="Years to retain substantiation evidence (ECGD Art. 3(6))",
    )
    pef_database: str = Field(
        "ef_3.1", description="PEF characterisation factor database version",
    )
    eco_label_registry_refresh_days: int = Field(
        30, ge=1, le=365,
        description="Days between eco-label registry refreshes",
    )
    max_claims: int = Field(
        10000, ge=1, le=1000000, description="Max claims per assessment run",
    )
    max_evidence_items: int = Field(
        50000, ge=1, le=5000000, description="Max evidence records per run",
    )
    greenwashing_risk_threshold: Decimal = Field(
        Decimal("50"), ge=Decimal("0"), le=Decimal("100"),
        description="Score (0-100) above which a claim is flagged as high risk",
    )
    cab_submission_format: str = Field(
        "PDF", description="CAB verification dossier output format (PDF/JSON/XBRL/HTML)",
    )
    reporting_language: str = Field(
        "en", description="ISO 639-1 language code for reports",
    )

    # --- Computed property ---------------------------------------------------
    @property
    def config_hash(self) -> str:
        """SHA-256 hash of the serialised configuration for provenance tracking."""
        serialised = json.dumps(self.model_dump(mode="json"), sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

    # --- Validators ----------------------------------------------------------
    @field_validator("enabled_engines")
    @classmethod
    def validate_engines(cls, v: List[str]) -> List[str]:
        """Ensure every enabled engine is a recognised engine name."""
        invalid = set(v) - set(ALL_ENGINES)
        if invalid:
            raise ValueError(f"Unknown engine(s): {sorted(invalid)}. Valid: {ALL_ENGINES}")
        return v

    @field_validator("enabled_workflows")
    @classmethod
    def validate_workflows(cls, v: List[str]) -> List[str]:
        """Ensure every enabled workflow is a recognised workflow name."""
        invalid = set(v) - set(ALL_WORKFLOWS)
        if invalid:
            raise ValueError(f"Unknown workflow(s): {sorted(invalid)}. Valid: {ALL_WORKFLOWS}")
        return v

    @field_validator("cab_submission_format")
    @classmethod
    def validate_cab_format(cls, v: str) -> str:
        """Ensure CAB submission format is one of PDF, JSON, XBRL, HTML."""
        allowed = {"PDF", "JSON", "XBRL", "HTML"}
        if v.upper() not in allowed:
            raise ValueError(f"Unsupported CAB format: {v}. Allowed: {sorted(allowed)}")
        return v.upper()

    @field_validator("reporting_language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Ensure reporting language is a 2-letter ISO 639-1 code."""
        if len(v) != 2 or not v.isalpha():
            raise ValueError(f"reporting_language must be 2-letter ISO 639-1 code, got: {v}")
        return v.lower()

    # --- Public methods ------------------------------------------------------
    def validate(self) -> List[str]:
        """Run comprehensive validation; return list of warning strings.

        Returns:
            List of warnings. Empty means fully valid.
        """
        warnings: List[str] = []
        if not self.enabled_engines:
            warnings.append("No engines enabled. claim_substantiation required (ECGD Art. 3).")
        if "claim_substantiation" not in self.enabled_engines:
            warnings.append("claim_substantiation disabled. ECGD Art. 3 requires substantiation.")
        if "greenwashing_detection" not in self.enabled_engines:
            warnings.append("greenwashing_detection disabled. ECGT/UCPD risk.")
        if (
            self.claim_scope in (ClaimScope.PRODUCT, ClaimScope.BOTH)
            and "lifecycle_assessment" not in self.enabled_engines
        ):
            warnings.append("Product claims in scope but lifecycle_assessment disabled (ECGD Art. 3(2)(d)).")
        if self.greenwashing_risk_threshold > Decimal("80"):
            warnings.append(f"Greenwashing threshold {self.greenwashing_risk_threshold} may be too permissive.")
        if self.evidence_retention_years < 3:
            warnings.append(f"Evidence retention {self.evidence_retention_years}y below 3y minimum recommended.")
        if not self.communication_channels:
            warnings.append("No communication channels configured.")
        return warnings

    def get_engine_config(self, engine_name: str) -> Dict[str, Any]:
        """Return configuration scoped to a specific engine.

        Args:
            engine_name: Must be in ALL_ENGINES.

        Returns:
            Dict with engine name, enabled status, and relevant pack parameters.

        Raises:
            ValueError: If engine_name is not recognised.
        """
        if engine_name not in ALL_ENGINES:
            raise ValueError(f"Unknown engine: {engine_name}. Valid: {ALL_ENGINES}")
        return {
            "engine": engine_name,
            "enabled": engine_name in self.enabled_engines,
            "sector": self.sector,
            "claim_scope": self.claim_scope.value,
            "pef_database": self.pef_database,
            "greenwashing_risk_threshold": str(self.greenwashing_risk_threshold),
            "evidence_retention_years": self.evidence_retention_years,
            "reporting_language": self.reporting_language,
            "config_hash": self.config_hash,
        }

    def get_workflow_config(self) -> Dict[str, Any]:
        """Return workflow orchestration configuration.

        Returns:
            Dict mapping each workflow to enabled status plus shared parameters.
        """
        return {
            "workflows": {wf: (wf in self.enabled_workflows) for wf in ALL_WORKFLOWS},
            "max_claims": self.max_claims,
            "max_evidence_items": self.max_evidence_items,
            "cab_submission_format": self.cab_submission_format,
            "reporting_language": self.reporting_language,
            "config_hash": self.config_hash,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialise full configuration to a plain dict with config_hash."""
        data = self.model_dump(mode="json")
        data["config_hash"] = self.config_hash
        return data

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PackConfig":
        """Load PackConfig from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            PackConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        env_overrides = cls._load_env_overrides()
        if env_overrides:
            raw = _deep_merge(raw, env_overrides)
        return cls(**raw)

    @classmethod
    def from_preset(cls, preset_name: str) -> "PackConfig":
        """Load PackConfig from a bundled sector preset.

        Args:
            preset_name: One of AVAILABLE_PRESETS keys.

        Returns:
            PackConfig instance.

        Raises:
            ValueError: If preset_name is not recognised.
            FileNotFoundError: If preset YAML is missing.
        """
        if preset_name not in AVAILABLE_PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {sorted(AVAILABLE_PRESETS.keys())}")
        preset_path = PRESETS_DIR / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset file not found: {preset_path}")
        return cls.from_yaml(preset_path)

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load overrides from GREEN_CLAIMS_PACK_* environment variables."""
        overrides: Dict[str, Any] = {}
        prefix = "GREEN_CLAIMS_PACK_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            config_key = key[len(prefix):].lower()
            if value.lower() in ("true", "false"):
                overrides[config_key] = value.lower() == "true"
            elif value.isdigit():
                overrides[config_key] = int(value)
            else:
                overrides[config_key] = value
        return overrides


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
