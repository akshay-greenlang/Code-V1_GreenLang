"""
PACK-001 CSRD Starter Pack - Configuration Manager

This module implements the PackConfig class that loads, merges, and validates
all configuration for the CSRD Starter Pack. It supports layered configuration
with base manifest, size presets, sector presets, and environment overrides.

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Size preset (large_enterprise / mid_market / sme / first_time_reporter)
    3. Sector preset (manufacturing / financial_services / technology / retail / energy)
    4. Environment overrides (CSRD_PACK_* environment variables)
    5. Explicit runtime overrides

Example:
    >>> config = PackConfig.load(
    ...     size_preset="mid_market",
    ...     sector_preset="manufacturing",
    ... )
    >>> print(config.pack.metadata.display_name)
    'CSRD Starter Pack'
    >>> print(config.active_agents)
    ['AGENT-MRV-001', 'AGENT-MRV-002', ...]
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent


# =============================================================================
# Pydantic Models - Typed configuration data structures
# =============================================================================


class ComplianceReference(BaseModel):
    """Reference to a regulatory compliance standard."""

    id: str = Field(..., description="Short identifier for the regulation")
    name: str = Field(..., description="Full name of the regulation")
    regulation: str = Field(..., description="Official regulation number")
    effective_date: str = Field(..., description="Date regulation became effective")
    description: str = Field("", description="Brief description of the regulation")


class PackMetadata(BaseModel):
    """Pack manifest metadata."""

    name: str = Field(..., description="Pack identifier slug")
    version: str = Field(..., description="Semantic version string")
    display_name: str = Field(..., description="Human-readable pack name")
    description: str = Field("", description="Pack description")
    category: str = Field(..., description="Pack category (e.g., eu-compliance)")
    author: str = Field("", description="Pack author or team")
    license: str = Field("Proprietary", description="License type")
    min_platform_version: str = Field("2.0.0", description="Minimum GreenLang platform version")
    release_date: str = Field("", description="Release date ISO string")
    support_tier: str = Field("enterprise", description="Support tier level")
    documentation_url: str = Field("", description="URL to pack documentation")
    changelog_url: str = Field("", description="URL to changelog")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    compliance_references: List[ComplianceReference] = Field(
        default_factory=list, description="Regulatory compliance references"
    )


class AgentComponentConfig(BaseModel):
    """Configuration for a single agent component."""

    id: str = Field(..., description="Agent identifier (e.g., AGENT-MRV-001)")
    name: str = Field("", description="Human-readable agent name")
    description: str = Field("", description="Agent description")
    required: bool = Field(True, description="Whether this agent is required")
    enabled: bool = Field(True, description="Whether this agent is enabled")
    version: str = Field("", description="Agent version override")
    config_overrides: Dict[str, Any] = Field(
        default_factory=dict, description="Agent-specific configuration overrides"
    )


class ComponentsConfig(BaseModel):
    """All components included in the pack."""

    apps: List[AgentComponentConfig] = Field(default_factory=list)
    data_agents: List[AgentComponentConfig] = Field(default_factory=list)
    quality_agents: List[AgentComponentConfig] = Field(default_factory=list)
    mrv_scope1: List[AgentComponentConfig] = Field(default_factory=list)
    mrv_scope2: List[AgentComponentConfig] = Field(default_factory=list)
    mrv_scope3: List[AgentComponentConfig] = Field(default_factory=list)
    foundation: List[AgentComponentConfig] = Field(default_factory=list)

    def get_all_agent_ids(self) -> List[str]:
        """Return all agent IDs across all component groups."""
        ids: List[str] = []
        for group in [
            self.apps,
            self.data_agents,
            self.quality_agents,
            self.mrv_scope1,
            self.mrv_scope2,
            self.mrv_scope3,
            self.foundation,
        ]:
            ids.extend(agent.id for agent in group)
        return ids

    def get_enabled_agent_ids(self) -> List[str]:
        """Return only enabled agent IDs across all component groups."""
        ids: List[str] = []
        for group in [
            self.apps,
            self.data_agents,
            self.quality_agents,
            self.mrv_scope1,
            self.mrv_scope2,
            self.mrv_scope3,
            self.foundation,
        ]:
            ids.extend(agent.id for agent in group if agent.enabled)
        return ids

    def get_required_agent_ids(self) -> List[str]:
        """Return only required agent IDs across all component groups."""
        ids: List[str] = []
        for group in [
            self.apps,
            self.data_agents,
            self.quality_agents,
            self.mrv_scope1,
            self.mrv_scope2,
            self.mrv_scope3,
            self.foundation,
        ]:
            ids.extend(agent.id for agent in group if agent.required)
        return ids

    def find_agent(self, agent_id: str) -> Optional[AgentComponentConfig]:
        """Find an agent by its ID across all groups."""
        for group in [
            self.apps,
            self.data_agents,
            self.quality_agents,
            self.mrv_scope1,
            self.mrv_scope2,
            self.mrv_scope3,
            self.foundation,
        ]:
            for agent in group:
                if agent.id == agent_id:
                    return agent
        return None


class WorkflowPhaseConfig(BaseModel):
    """Configuration for a single phase within a workflow."""

    name: str = Field(..., description="Phase identifier")
    description: str = Field("", description="Phase description")
    agents: List[str] = Field(default_factory=list, description="Agent IDs in this phase")
    duration_days: int = Field(1, ge=1, description="Estimated duration in days")


class WorkflowConfig(BaseModel):
    """Configuration for a workflow orchestration."""

    display_name: str = Field(..., description="Human-readable workflow name")
    description: str = Field("", description="Workflow description")
    schedule: str = Field("on_demand", description="Schedule type: annual, quarterly, on_demand")
    estimated_duration_days: int = Field(1, ge=1, description="Total estimated duration")
    phases: List[WorkflowPhaseConfig] = Field(
        default_factory=list, description="Ordered list of workflow phases"
    )
    enabled: bool = Field(True, description="Whether this workflow is enabled")

    def get_all_agent_ids(self) -> List[str]:
        """Return all unique agent IDs used across all phases."""
        ids: set[str] = set()
        for phase in self.phases:
            ids.update(phase.agents)
        return sorted(ids)


class TemplateConfig(BaseModel):
    """Configuration for a report template."""

    id: str = Field(..., description="Template identifier")
    display_name: str = Field(..., description="Human-readable template name")
    description: str = Field("", description="Template description")
    format: str = Field("pdf", description="Output format (pdf, xhtml, html, zip)")
    template_file: str = Field(..., description="Path to template file")
    xbrl_taxonomy: str = Field("", description="XBRL taxonomy version if applicable")
    enabled: bool = Field(True, description="Whether this template is enabled")


class PerformanceTargets(BaseModel):
    """Performance targets for the pack."""

    data_ingestion_rps: int = Field(5000, description="Records per second for data ingestion")
    data_ingestion_max_latency_ms: int = Field(200, description="Max latency for ingestion")
    ghg_single_scope_max_seconds: int = Field(30, description="Max seconds for single scope calc")
    ghg_full_inventory_max_seconds: int = Field(300, description="Max seconds for full inventory")
    report_esrs_max_seconds: int = Field(120, description="Max seconds for ESRS report")
    report_summary_max_seconds: int = Field(60, description="Max seconds for executive summary")
    report_xbrl_max_seconds: int = Field(180, description="Max seconds for XBRL tagging")
    api_p50_ms: int = Field(100, description="API p50 latency target")
    api_p95_ms: int = Field(500, description="API p95 latency target")
    api_p99_ms: int = Field(2000, description="API p99 latency target")
    availability_percent: float = Field(99.9, description="Target uptime percentage")


class RequirementsConfig(BaseModel):
    """System requirements for the pack."""

    python_version: str = Field(">=3.11", description="Minimum Python version")
    postgresql_version: str = Field(">=14", description="Minimum PostgreSQL version")
    redis_version: str = Field(">=7", description="Minimum Redis version")
    min_cpu_cores: int = Field(4, description="Minimum CPU cores")
    min_memory_gb: int = Field(16, description="Minimum memory in GB")
    min_storage_gb: int = Field(100, description="Minimum storage in GB")
    recommended_cpu_cores: int = Field(8, description="Recommended CPU cores")
    recommended_memory_gb: int = Field(32, description="Recommended memory in GB")
    recommended_storage_gb: int = Field(500, description="Recommended storage in GB")
    database_extensions: List[str] = Field(
        default_factory=lambda: ["pgvector", "timescaledb"],
        description="Required database extensions",
    )
    min_db_connections: int = Field(20, description="Minimum database connections")


class ESRSStandardConfig(BaseModel):
    """Configuration for an individual ESRS standard."""

    id: str = Field(..., description="Standard ID (e.g., E1, S1, G1)")
    name: str = Field(..., description="Standard name")
    enabled: bool = Field(True, description="Whether this standard is active")
    mandatory: bool = Field(False, description="Whether this standard is always required")
    material: Optional[bool] = Field(None, description="Materiality assessment result")
    disclosure_requirements: List[str] = Field(
        default_factory=list, description="Active disclosure requirement IDs"
    )


class Scope3Config(BaseModel):
    """Configuration for Scope 3 category enablement."""

    enabled_categories: List[int] = Field(
        default_factory=lambda: list(range(1, 16)),
        description="Enabled Scope 3 category numbers (1-15)",
    )
    priority_categories: List[int] = Field(
        default_factory=list,
        description="Priority categories for enhanced calculation",
    )
    screening_required: bool = Field(
        True, description="Whether Scope 3 screening is required"
    )
    materiality_threshold_percent: float = Field(
        5.0, description="Threshold below which a category may be excluded"
    )


class XBRLConfig(BaseModel):
    """XBRL tagging configuration."""

    enabled: bool = Field(True, description="Whether XBRL tagging is active")
    taxonomy_version: str = Field("ESRS_2023", description="XBRL taxonomy version")
    output_format: str = Field("inline_xbrl", description="XBRL output format")
    validation_level: str = Field("strict", description="Validation strictness")
    languages: List[str] = Field(
        default_factory=lambda: ["en"], description="Report languages"
    )


class ConsolidationConfig(BaseModel):
    """Multi-entity consolidation configuration."""

    enabled: bool = Field(False, description="Whether multi-entity consolidation is active")
    method: str = Field("operational_control", description="Consolidation approach")
    subsidiaries: List[str] = Field(
        default_factory=list, description="Subsidiary entity IDs"
    )
    intercompany_elimination: bool = Field(
        True, description="Eliminate intercompany transactions"
    )


class GuidedModeConfig(BaseModel):
    """Configuration for guided/tutorial mode."""

    enabled: bool = Field(False, description="Whether guided mode is active")
    tutorial_mode: bool = Field(False, description="Show tutorial overlays")
    pre_populated_examples: bool = Field(False, description="Load example data")
    ai_assistance_level: str = Field(
        "standard", description="AI assistance level: minimal, standard, enhanced"
    )
    step_by_step: bool = Field(False, description="Enable step-by-step guidance")
    onboarding_extended: bool = Field(False, description="Extended onboarding flow")


class DemoConfig(BaseModel):
    """Demo mode configuration."""

    enabled: bool = Field(False, description="Whether demo mode is active")
    use_sample_data: bool = Field(False, description="Use bundled sample data")
    skip_external_apis: bool = Field(False, description="Skip calls to external APIs")
    mock_ai_responses: bool = Field(False, description="Use mocked AI responses")
    fast_execution: bool = Field(False, description="Skip delays and long operations")
    sample_company_profile: str = Field("", description="Path to sample company profile")
    sample_esg_data: str = Field("", description="Path to sample ESG dataset")


class PresetConfig(BaseModel):
    """Merged configuration from size and sector presets."""

    size_preset_id: str = Field("", description="Active size preset identifier")
    sector_preset_id: str = Field("", description="Active sector preset identifier")
    esrs_standards: List[ESRSStandardConfig] = Field(
        default_factory=list, description="ESRS standard configurations"
    )
    scope3: Scope3Config = Field(
        default_factory=Scope3Config, description="Scope 3 configuration"
    )
    xbrl: XBRLConfig = Field(
        default_factory=XBRLConfig, description="XBRL configuration"
    )
    consolidation: ConsolidationConfig = Field(
        default_factory=ConsolidationConfig, description="Consolidation configuration"
    )
    guided_mode: GuidedModeConfig = Field(
        default_factory=GuidedModeConfig, description="Guided mode configuration"
    )
    demo: DemoConfig = Field(
        default_factory=DemoConfig, description="Demo mode configuration"
    )
    emission_factor_priorities: List[str] = Field(
        default_factory=list,
        description="Ordered list of emission factor sources to prioritize",
    )
    sector_specific: Dict[str, Any] = Field(
        default_factory=dict, description="Sector-specific configuration values"
    )


class CSRDPackConfig(BaseModel):
    """
    Top-level CSRD Pack configuration model.

    This model represents the fully merged and validated configuration
    for a CSRD Starter Pack deployment. It combines the base manifest,
    size preset, sector preset, and any runtime overrides.
    """

    metadata: PackMetadata = Field(..., description="Pack metadata")
    components: ComponentsConfig = Field(
        default_factory=ComponentsConfig, description="Component configurations"
    )
    workflows: Dict[str, WorkflowConfig] = Field(
        default_factory=dict, description="Workflow configurations"
    )
    templates: List[TemplateConfig] = Field(
        default_factory=list, description="Template configurations"
    )
    performance: PerformanceTargets = Field(
        default_factory=PerformanceTargets, description="Performance targets"
    )
    requirements: RequirementsConfig = Field(
        default_factory=RequirementsConfig, description="System requirements"
    )
    presets: PresetConfig = Field(
        default_factory=PresetConfig, description="Active preset configuration"
    )

    @field_validator("workflows", mode="before")
    @classmethod
    def parse_workflows(cls, v: Any) -> Dict[str, WorkflowConfig]:
        """Parse workflow definitions from YAML structure."""
        if isinstance(v, dict):
            parsed: Dict[str, WorkflowConfig] = {}
            for key, val in v.items():
                if isinstance(val, WorkflowConfig):
                    parsed[key] = val
                elif isinstance(val, dict):
                    parsed[key] = WorkflowConfig(**val)
                else:
                    parsed[key] = val
            return parsed
        return v

    def get_enabled_workflows(self) -> Dict[str, WorkflowConfig]:
        """Return only enabled workflows."""
        return {k: v for k, v in self.workflows.items() if v.enabled}

    def get_active_agent_ids(self) -> List[str]:
        """Return all enabled agent IDs from components."""
        return self.components.get_enabled_agent_ids()

    def get_material_esrs_standards(self) -> List[ESRSStandardConfig]:
        """Return ESRS standards marked as material or mandatory."""
        return [
            s
            for s in self.presets.esrs_standards
            if s.mandatory or s.material is True or s.enabled
        ]


# =============================================================================
# PackConfig - Main configuration manager
# =============================================================================


class PackConfig:
    """
    Configuration manager for PACK-001 CSRD Starter Pack.

    Loads and merges configuration from multiple sources in the following
    priority order (later sources override earlier):

        1. Base pack.yaml manifest
        2. Size preset (large_enterprise, mid_market, sme, first_time_reporter)
        3. Sector preset (manufacturing, financial_services, technology, retail, energy)
        4. Environment variables (CSRD_PACK_* prefix)
        5. Runtime overrides

    Attributes:
        pack: The fully resolved CSRDPackConfig instance.
        config_hash: SHA-256 hash of the resolved configuration for provenance.
        loaded_at: Timestamp when configuration was loaded.

    Example:
        >>> config = PackConfig.load(
        ...     size_preset="mid_market",
        ...     sector_preset="manufacturing",
        ... )
        >>> print(config.pack.metadata.version)
        '1.0.0'
        >>> print(config.active_agents)
        ['AGENT-MRV-001', ...]
    """

    VALID_SIZE_PRESETS = {"large_enterprise", "mid_market", "sme", "first_time_reporter"}
    VALID_SECTOR_PRESETS = {
        "manufacturing",
        "financial_services",
        "technology",
        "retail",
        "energy",
    }

    def __init__(
        self,
        pack: CSRDPackConfig,
        config_hash: str,
        loaded_at: datetime,
        source_files: List[str],
    ) -> None:
        """
        Initialize PackConfig with resolved configuration.

        Args:
            pack: Fully resolved pack configuration.
            config_hash: SHA-256 hash of the configuration.
            loaded_at: Timestamp of configuration loading.
            source_files: List of source files that were loaded.
        """
        self.pack = pack
        self.config_hash = config_hash
        self.loaded_at = loaded_at
        self.source_files = source_files

    @classmethod
    def load(
        cls,
        pack_dir: Optional[Union[str, Path]] = None,
        size_preset: Optional[str] = None,
        sector_preset: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        demo_mode: bool = False,
    ) -> "PackConfig":
        """
        Load and merge pack configuration from all sources.

        Args:
            pack_dir: Path to the pack root directory. Defaults to PACK_BASE_DIR.
            size_preset: Size preset to apply (large_enterprise, mid_market, sme, first_time_reporter).
            sector_preset: Sector preset to apply (manufacturing, financial_services, technology, retail, energy).
            overrides: Dictionary of runtime configuration overrides.
            demo_mode: If True, loads demo configuration.

        Returns:
            Fully resolved PackConfig instance.

        Raises:
            FileNotFoundError: If pack.yaml or preset files are missing.
            ValueError: If preset names are invalid or configuration is malformed.
        """
        start_time = datetime.now()
        pack_dir = Path(pack_dir) if pack_dir else PACK_BASE_DIR
        source_files: List[str] = []

        logger.info(
            "Loading CSRD Starter Pack configuration from %s",
            pack_dir,
        )

        # Step 1: Load base manifest
        manifest_path = pack_dir / "pack.yaml"
        base_config = cls._load_yaml_file(manifest_path)
        source_files.append(str(manifest_path))
        logger.info("Loaded base manifest: %s", manifest_path)

        # Step 2: Parse base config into structured form
        pack_config_dict = cls._parse_manifest(base_config)

        # Step 3: Apply size preset
        if size_preset:
            cls._validate_preset_name(size_preset, cls.VALID_SIZE_PRESETS, "size")
            preset_path = pack_dir / "config" / "presets" / f"{size_preset}.yaml"
            preset_data = cls._load_yaml_file(preset_path)
            source_files.append(str(preset_path))
            pack_config_dict = cls._merge_preset(
                pack_config_dict, preset_data, "size", size_preset
            )
            logger.info("Applied size preset: %s", size_preset)

        # Step 4: Apply sector preset
        if sector_preset:
            cls._validate_preset_name(sector_preset, cls.VALID_SECTOR_PRESETS, "sector")
            sector_path = pack_dir / "config" / "sectors" / f"{sector_preset}.yaml"
            sector_data = cls._load_yaml_file(sector_path)
            source_files.append(str(sector_path))
            pack_config_dict = cls._merge_preset(
                pack_config_dict, sector_data, "sector", sector_preset
            )
            logger.info("Applied sector preset: %s", sector_preset)

        # Step 5: Apply demo mode
        if demo_mode:
            demo_config_path = pack_dir / "config" / "demo" / "demo_config.yaml"
            demo_data = cls._load_yaml_file(demo_config_path)
            source_files.append(str(demo_config_path))
            pack_config_dict = cls._merge_demo(pack_config_dict, demo_data)
            logger.info("Applied demo mode configuration")

        # Step 6: Apply environment variable overrides
        env_overrides = cls._load_env_overrides()
        if env_overrides:
            pack_config_dict = cls._deep_merge(pack_config_dict, env_overrides)
            logger.info(
                "Applied %d environment variable overrides",
                len(env_overrides),
            )

        # Step 7: Apply runtime overrides
        if overrides:
            pack_config_dict = cls._deep_merge(pack_config_dict, overrides)
            logger.info("Applied runtime overrides")

        # Step 8: Validate and create typed config
        pack = CSRDPackConfig(**pack_config_dict)

        # Step 9: Calculate configuration hash
        config_json = json.dumps(pack_config_dict, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()

        load_duration = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(
            "Pack configuration loaded in %.1fms (hash: %s)",
            load_duration,
            config_hash[:12],
        )

        return cls(
            pack=pack,
            config_hash=config_hash,
            loaded_at=start_time,
            source_files=source_files,
        )

    @property
    def active_agents(self) -> List[str]:
        """Return list of all enabled agent IDs."""
        return self.pack.get_active_agent_ids()

    @property
    def active_workflows(self) -> Dict[str, WorkflowConfig]:
        """Return dictionary of enabled workflows."""
        return self.pack.get_enabled_workflows()

    @property
    def material_standards(self) -> List[ESRSStandardConfig]:
        """Return ESRS standards that are material or mandatory."""
        return self.pack.get_material_esrs_standards()

    def get_agent_config(self, agent_id: str) -> Optional[AgentComponentConfig]:
        """
        Retrieve configuration for a specific agent.

        Args:
            agent_id: The agent identifier (e.g., AGENT-MRV-001).

        Returns:
            AgentComponentConfig if found, None otherwise.
        """
        return self.pack.components.find_agent(agent_id)

    def get_workflow_config(self, workflow_id: str) -> Optional[WorkflowConfig]:
        """
        Retrieve configuration for a specific workflow.

        Args:
            workflow_id: The workflow identifier (e.g., annual_reporting).

        Returns:
            WorkflowConfig if found, None otherwise.
        """
        return self.pack.workflows.get(workflow_id)

    def is_scope3_category_enabled(self, category: int) -> bool:
        """
        Check if a Scope 3 category is enabled.

        Args:
            category: Scope 3 category number (1-15).

        Returns:
            True if the category is enabled, False otherwise.
        """
        return category in self.pack.presets.scope3.enabled_categories

    def is_esrs_standard_enabled(self, standard_id: str) -> bool:
        """
        Check if an ESRS standard is enabled.

        Args:
            standard_id: ESRS standard identifier (e.g., E1, S1, G1).

        Returns:
            True if the standard is enabled, False otherwise.
        """
        for standard in self.pack.presets.esrs_standards:
            if standard.id == standard_id:
                return standard.enabled
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full configuration to a dictionary."""
        return {
            "pack": self.pack.model_dump(),
            "config_hash": self.config_hash,
            "loaded_at": self.loaded_at.isoformat(),
            "source_files": self.source_files,
        }

    def get_provenance_hash(self) -> str:
        """
        Get SHA-256 hash of the configuration for audit provenance.

        Returns:
            SHA-256 hex digest of the configuration.
        """
        return self.config_hash

    # =========================================================================
    # Internal Methods
    # =========================================================================

    @staticmethod
    def _load_yaml_file(path: Path) -> Dict[str, Any]:
        """
        Load and parse a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Parsed YAML content as a dictionary.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not valid YAML.
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}") from e

        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML dictionary in {path}, got {type(data).__name__}")

        return data

    @staticmethod
    def _validate_preset_name(name: str, valid_names: set, preset_type: str) -> None:
        """
        Validate that a preset name is recognized.

        Args:
            name: The preset name to validate.
            valid_names: Set of valid preset names.
            preset_type: Type of preset (size, sector) for error messaging.

        Raises:
            ValueError: If the preset name is not valid.
        """
        if name not in valid_names:
            raise ValueError(
                f"Invalid {preset_type} preset '{name}'. "
                f"Valid options: {sorted(valid_names)}"
            )

    @classmethod
    def _parse_manifest(cls, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse raw YAML manifest into the structure expected by CSRDPackConfig.

        Args:
            raw: Raw parsed YAML dictionary from pack.yaml.

        Returns:
            Restructured dictionary suitable for CSRDPackConfig initialization.
        """
        result: Dict[str, Any] = {}

        # Metadata
        result["metadata"] = raw.get("metadata", {})

        # Components - restructure from YAML lists to Pydantic model format
        raw_components = raw.get("components", {})
        result["components"] = {}
        for group_key in [
            "apps",
            "data_agents",
            "quality_agents",
            "mrv_scope1",
            "mrv_scope2",
            "mrv_scope3",
            "foundation",
        ]:
            agents = raw_components.get(group_key, [])
            result["components"][group_key] = [
                {
                    "id": a.get("id", ""),
                    "name": a.get("name", ""),
                    "description": a.get("description", ""),
                    "required": a.get("required", True),
                    "enabled": a.get("enabled", True),
                    "version": a.get("version", ""),
                    "config_overrides": a.get("config_overrides", {}),
                }
                for a in agents
            ]

        # Workflows
        result["workflows"] = raw.get("workflows", {})

        # Templates
        result["templates"] = raw.get("templates", [])

        # Performance targets - flatten from nested YAML
        raw_perf = raw.get("performance", {})
        result["performance"] = {
            "data_ingestion_rps": raw_perf.get("data_ingestion", {}).get(
                "throughput_records_per_second", 5000
            ),
            "data_ingestion_max_latency_ms": raw_perf.get("data_ingestion", {}).get(
                "max_latency_ms", 200
            ),
            "ghg_single_scope_max_seconds": raw_perf.get("ghg_calculation", {}).get(
                "single_scope_max_seconds", 30
            ),
            "ghg_full_inventory_max_seconds": raw_perf.get("ghg_calculation", {}).get(
                "full_inventory_max_seconds", 300
            ),
            "report_esrs_max_seconds": raw_perf.get("report_generation", {}).get(
                "esrs_disclosure_max_seconds", 120
            ),
            "report_summary_max_seconds": raw_perf.get("report_generation", {}).get(
                "executive_summary_max_seconds", 60
            ),
            "report_xbrl_max_seconds": raw_perf.get("report_generation", {}).get(
                "xbrl_tagging_max_seconds", 180
            ),
            "api_p50_ms": raw_perf.get("api_response", {}).get("p50_ms", 100),
            "api_p95_ms": raw_perf.get("api_response", {}).get("p95_ms", 500),
            "api_p99_ms": raw_perf.get("api_response", {}).get("p99_ms", 2000),
            "availability_percent": raw_perf.get("availability", {}).get(
                "target_uptime_percent", 99.9
            ),
        }

        # Requirements - flatten from nested YAML
        raw_req = raw.get("requirements", {})
        raw_runtime = raw_req.get("runtime", {})
        raw_infra = raw_req.get("infrastructure", {})
        raw_db = raw_req.get("database", {})
        result["requirements"] = {
            "python_version": raw_runtime.get("python", ">=3.11"),
            "postgresql_version": raw_runtime.get("postgresql", ">=14"),
            "redis_version": raw_runtime.get("redis", ">=7"),
            "min_cpu_cores": raw_infra.get("min_cpu_cores", 4),
            "min_memory_gb": raw_infra.get("min_memory_gb", 16),
            "min_storage_gb": raw_infra.get("min_storage_gb", 100),
            "recommended_cpu_cores": raw_infra.get("recommended_cpu_cores", 8),
            "recommended_memory_gb": raw_infra.get("recommended_memory_gb", 32),
            "recommended_storage_gb": raw_infra.get("recommended_storage_gb", 500),
            "database_extensions": raw_db.get("extensions", ["pgvector", "timescaledb"]),
            "min_db_connections": raw_db.get("min_connections", 20),
        }

        # Presets - initialize empty, filled by preset loading
        result["presets"] = {}

        return result

    @classmethod
    def _merge_preset(
        cls,
        base: Dict[str, Any],
        preset_data: Dict[str, Any],
        preset_type: str,
        preset_id: str,
    ) -> Dict[str, Any]:
        """
        Merge a preset configuration into the base configuration.

        Preset data can contain:
          - esrs_standards: Override ESRS standard enablement
          - scope3: Override Scope 3 configuration
          - xbrl: Override XBRL configuration
          - consolidation: Override consolidation settings
          - guided_mode: Override guided mode settings
          - agent_overrides: Enable/disable specific agents
          - sector_specific: Sector-specific configuration values
          - emission_factor_priorities: Ordered emission factor sources

        Args:
            base: Current configuration dictionary.
            preset_data: Preset configuration to merge.
            preset_type: Type of preset (size, sector).
            preset_id: Identifier of the preset.

        Returns:
            Merged configuration dictionary.
        """
        result = cls._deep_merge(base, {})

        # Initialize presets if missing
        if "presets" not in result:
            result["presets"] = {}

        # Set preset identifier
        if preset_type == "size":
            result["presets"]["size_preset_id"] = preset_id
        elif preset_type == "sector":
            result["presets"]["sector_preset_id"] = preset_id

        # Merge ESRS standards
        if "esrs_standards" in preset_data:
            result["presets"]["esrs_standards"] = preset_data["esrs_standards"]

        # Merge Scope 3 configuration
        if "scope3" in preset_data:
            result["presets"]["scope3"] = preset_data["scope3"]

        # Merge XBRL configuration
        if "xbrl" in preset_data:
            result["presets"]["xbrl"] = preset_data["xbrl"]

        # Merge consolidation configuration
        if "consolidation" in preset_data:
            result["presets"]["consolidation"] = preset_data["consolidation"]

        # Merge guided mode configuration
        if "guided_mode" in preset_data:
            result["presets"]["guided_mode"] = preset_data["guided_mode"]

        # Merge demo configuration
        if "demo" in preset_data:
            result["presets"]["demo"] = preset_data["demo"]

        # Merge emission factor priorities
        if "emission_factor_priorities" in preset_data:
            result["presets"]["emission_factor_priorities"] = preset_data[
                "emission_factor_priorities"
            ]

        # Merge sector-specific configuration
        if "sector_specific" in preset_data:
            existing_sector = result["presets"].get("sector_specific", {})
            result["presets"]["sector_specific"] = cls._deep_merge(
                existing_sector, preset_data["sector_specific"]
            )

        # Apply agent overrides (enable/disable specific agents)
        if "agent_overrides" in preset_data:
            result = cls._apply_agent_overrides(result, preset_data["agent_overrides"])

        # Merge performance overrides
        if "performance" in preset_data:
            result["performance"] = cls._deep_merge(
                result.get("performance", {}), preset_data["performance"]
            )

        return result

    @classmethod
    def _merge_demo(
        cls, base: Dict[str, Any], demo_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge demo configuration into base configuration.

        Args:
            base: Current configuration dictionary.
            demo_data: Demo configuration to merge.

        Returns:
            Merged configuration dictionary.
        """
        result = cls._deep_merge(base, {})

        if "presets" not in result:
            result["presets"] = {}

        result["presets"]["demo"] = {
            "enabled": demo_data.get("enabled", True),
            "use_sample_data": demo_data.get("use_sample_data", True),
            "skip_external_apis": demo_data.get("skip_external_apis", True),
            "mock_ai_responses": demo_data.get("mock_ai_responses", True),
            "fast_execution": demo_data.get("fast_execution", True),
            "sample_company_profile": demo_data.get("sample_company_profile", ""),
            "sample_esg_data": demo_data.get("sample_esg_data", ""),
        }

        # Apply demo-specific agent overrides if present
        if "agent_overrides" in demo_data:
            result = cls._apply_agent_overrides(result, demo_data["agent_overrides"])

        return result

    @classmethod
    def _apply_agent_overrides(
        cls,
        config: Dict[str, Any],
        overrides: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Apply agent-level overrides to component configuration.

        Args:
            config: Current configuration dictionary.
            overrides: Agent override specifications keyed by agent ID.

        Returns:
            Updated configuration dictionary.
        """
        components = config.get("components", {})

        for group_key in [
            "apps",
            "data_agents",
            "quality_agents",
            "mrv_scope1",
            "mrv_scope2",
            "mrv_scope3",
            "foundation",
        ]:
            agents = components.get(group_key, [])
            for agent in agents:
                agent_id = agent.get("id", "")
                if agent_id in overrides:
                    agent_override = overrides[agent_id]
                    if "enabled" in agent_override:
                        agent["enabled"] = agent_override["enabled"]
                    if "config_overrides" in agent_override:
                        existing = agent.get("config_overrides", {})
                        agent["config_overrides"] = cls._deep_merge(
                            existing, agent_override["config_overrides"]
                        )

        return config

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """
        Load configuration overrides from environment variables.

        Environment variables with the prefix CSRD_PACK_ are parsed as
        configuration overrides. The variable name is converted to a
        nested dictionary path using double underscores as separators.

        Examples:
            CSRD_PACK_PERFORMANCE__API_P50_MS=50
            CSRD_PACK_PRESETS__SCOPE3__SCREENING_REQUIRED=false

        Returns:
            Dictionary of environment-based overrides.
        """
        overrides: Dict[str, Any] = {}
        prefix = "CSRD_PACK_"

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_path = key[len(prefix) :].lower().split("__")
                parsed_value = PackConfig._parse_env_value(value)
                current = overrides
                for i, part in enumerate(config_path):
                    if i == len(config_path) - 1:
                        current[part] = parsed_value
                    else:
                        current = current.setdefault(part, {})

        return overrides

    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """
        Parse an environment variable value to its appropriate Python type.

        Args:
            value: Raw string value from environment.

        Returns:
            Parsed value (bool, int, float, or str).
        """
        lower_val = value.lower()
        if lower_val in ("true", "yes", "1"):
            return True
        if lower_val in ("false", "no", "0"):
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    @classmethod
    def _deep_merge(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries. Override values take precedence.

        Lists are replaced (not appended). Nested dicts are recursively merged.

        Args:
            base: Base dictionary.
            override: Override dictionary (values take precedence).

        Returns:
            New merged dictionary.
        """
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = cls._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        """Return string representation of PackConfig."""
        return (
            f"PackConfig("
            f"version={self.pack.metadata.version!r}, "
            f"size={self.pack.presets.size_preset_id!r}, "
            f"sector={self.pack.presets.sector_preset_id!r}, "
            f"agents={len(self.active_agents)}, "
            f"hash={self.config_hash[:12]})"
        )
