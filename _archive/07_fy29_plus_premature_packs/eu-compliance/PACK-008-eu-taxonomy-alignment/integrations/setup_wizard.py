"""
Setup Wizard - PACK-008 EU Taxonomy Alignment

This module provides a 10-step guided configuration wizard for the EU Taxonomy
Alignment Pack. It walks users through organization setup, environmental objective
selection, activity mapping, and disclosure configuration.

Setup steps:
1. Organization Type (NFU / Financial Institution / Asset Manager)
2. Environmental Objectives (select from 6 objectives)
3. NACE Activities (map company activities to NACE codes)
4. Financial Data Sources (ERP, Excel, API, manual)
5. Reporting Period (fiscal year, dates)
6. Disclosure Requirements (Article 8, EBA Pillar 3, both)
7. Agent Configuration (enable/disable optional agents)
8. Data Connections (database, cache, external APIs)
9. Validation (run configuration validation)
10. Summary (generate final TaxonomyAlignmentConfig)

Example:
    >>> config = SetupWizardConfig()
    >>> wizard = TaxonomySetupWizard(config)
    >>> result = await wizard.run_setup(answers)
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SetupWizardConfig(BaseModel):
    """Configuration for setup wizard."""

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
        description="Preset name to apply (non_financial_undertaking, financial_institution, etc.)"
    )


class StepResult(BaseModel):
    """Result from a single setup step."""

    step_number: int
    name: str
    status: Literal["PASS", "WARN", "FAIL", "SKIP"] = "PASS"
    message: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SetupResult(BaseModel):
    """Complete setup wizard result."""

    overall_status: Literal["PASS", "WARN", "FAIL"] = "PASS"
    total_steps: int = 0
    passed: int = 0
    warned: int = 0
    failed: int = 0
    skipped: int = 0
    steps: List[StepResult] = Field(default_factory=list)
    generated_config: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TaxonomySetupWizard:
    """
    10-step guided configuration wizard for EU Taxonomy Alignment Pack.

    Generates a complete TaxonomyAlignmentConfig from user answers.

    Example:
        >>> config = SetupWizardConfig()
        >>> wizard = TaxonomySetupWizard(config)
        >>> result = await wizard.run_setup({"organization_type": "non_financial_undertaking"})
    """

    PRESETS: Dict[str, Dict[str, Any]] = {
        "non_financial_undertaking": {
            "organization_type": "non_financial_undertaking",
            "environmental_objectives": ["CCM", "CCA", "WTR", "CE", "PPC", "BIO"],
            "enable_gar": False,
            "disclosure_format": "article_8",
            "enable_capex_plan": True
        },
        "financial_institution": {
            "organization_type": "financial_institution",
            "environmental_objectives": ["CCM", "CCA", "WTR", "CE", "PPC", "BIO"],
            "enable_gar": True,
            "disclosure_format": "both",
            "enable_capex_plan": False
        },
        "asset_manager": {
            "organization_type": "asset_manager",
            "environmental_objectives": ["CCM", "CCA"],
            "enable_gar": True,
            "disclosure_format": "eba_pillar_3",
            "enable_capex_plan": False
        },
        "large_enterprise": {
            "organization_type": "non_financial_undertaking",
            "environmental_objectives": ["CCM", "CCA", "WTR", "CE", "PPC", "BIO"],
            "enable_gar": False,
            "disclosure_format": "article_8",
            "enable_capex_plan": True,
            "enable_cross_framework": True
        },
        "sme_simplified": {
            "organization_type": "non_financial_undertaking",
            "environmental_objectives": ["CCM"],
            "enable_gar": False,
            "disclosure_format": "article_8",
            "enable_capex_plan": False,
            "simplified_mode": True
        }
    }

    def __init__(self, config: SetupWizardConfig):
        """Initialize setup wizard."""
        self.config = config
        self._setup_data: Dict[str, Any] = {}

        # Apply preset if specified
        if config.preset and config.preset in self.PRESETS:
            self._setup_data = self.PRESETS[config.preset].copy()
            logger.info(f"Applied preset: {config.preset}")

        logger.info("TaxonomySetupWizard initialized (10 steps)")

    async def run_setup(
        self,
        answers: Optional[Dict[str, Any]] = None
    ) -> SetupResult:
        """
        Execute complete 10-step setup wizard.

        Args:
            answers: Pre-provided answers (for non-interactive mode)

        Returns:
            Setup result with generated configuration
        """
        logger.info("Starting PACK-008 Taxonomy Alignment setup wizard")

        answers = answers or {}
        steps: List[StepResult] = []

        # Step 1: Organization Type
        steps.append(await self._step_1_organization_type(answers))

        # Step 2: Environmental Objectives
        steps.append(await self._step_2_environmental_objectives(answers))

        # Step 3: NACE Activities
        steps.append(await self._step_3_nace_activities(answers))

        # Step 4: Financial Data Sources
        steps.append(await self._step_4_financial_data_sources(answers))

        # Step 5: Reporting Period
        steps.append(await self._step_5_reporting_period(answers))

        # Step 6: Disclosure Requirements
        steps.append(await self._step_6_disclosure_requirements(answers))

        # Step 7: Agent Configuration
        if not self.config.skip_optional_steps:
            steps.append(await self._step_7_agent_configuration(answers))

        # Step 8: Data Connections
        if not self.config.skip_optional_steps:
            steps.append(await self._step_8_data_connections(answers))

        # Step 9: Validation
        steps.append(await self._step_9_validation())

        # Step 10: Summary
        steps.append(await self._step_10_summary())

        # Build result
        result = self._build_result(steps)

        logger.info(f"Setup wizard complete: {result.overall_status}")

        return result

    async def validate_step(
        self,
        step_num: int,
        data: Dict[str, Any]
    ) -> StepResult:
        """
        Validate a single step's data without running the full wizard.

        Args:
            step_num: Step number (1-10)
            data: Step data to validate

        Returns:
            Validation result for the step
        """
        validators = {
            1: self._validate_organization_type,
            2: self._validate_objectives,
            3: self._validate_nace_activities,
            4: self._validate_financial_sources,
            5: self._validate_reporting_period,
            6: self._validate_disclosure_format,
            7: self._validate_agent_config,
            8: self._validate_data_connections,
            9: self._validate_overall,
            10: self._validate_summary
        }

        validator = validators.get(step_num)
        if validator:
            return await validator(data)

        return StepResult(
            step_number=step_num,
            name="Unknown",
            status="FAIL",
            message=f"Unknown step number: {step_num}"
        )

    async def _step_1_organization_type(self, answers: Dict[str, Any]) -> StepResult:
        """Step 1: Select organization type."""
        logger.info("Step 1: Organization Type")

        try:
            org_type = answers.get(
                "organization_type",
                self._setup_data.get("organization_type", "non_financial_undertaking")
            )

            valid_types = ["non_financial_undertaking", "financial_institution", "asset_manager"]
            if org_type not in valid_types:
                return StepResult(
                    step_number=1,
                    name="Organization Type",
                    status="FAIL",
                    message=f"Invalid organization type: {org_type}",
                    data={"valid_types": valid_types}
                )

            self._setup_data["organization_type"] = org_type

            return StepResult(
                step_number=1,
                name="Organization Type",
                status="PASS",
                message=f"Organization type: {org_type}",
                data={"organization_type": org_type}
            )

        except Exception as e:
            return StepResult(
                step_number=1, name="Organization Type",
                status="FAIL", message=f"Error: {str(e)}"
            )

    async def _step_2_environmental_objectives(self, answers: Dict[str, Any]) -> StepResult:
        """Step 2: Select environmental objectives."""
        logger.info("Step 2: Environmental Objectives")

        try:
            all_objectives = ["CCM", "CCA", "WTR", "CE", "PPC", "BIO"]
            selected = answers.get(
                "environmental_objectives",
                self._setup_data.get("environmental_objectives", all_objectives)
            )

            # Validate selections
            invalid = [o for o in selected if o not in all_objectives]
            if invalid:
                return StepResult(
                    step_number=2,
                    name="Environmental Objectives",
                    status="FAIL",
                    message=f"Invalid objectives: {invalid}",
                    data={"valid_objectives": all_objectives}
                )

            self._setup_data["environmental_objectives"] = selected

            return StepResult(
                step_number=2,
                name="Environmental Objectives",
                status="PASS",
                message=f"Selected {len(selected)} objectives: {', '.join(selected)}",
                data={"objectives": selected}
            )

        except Exception as e:
            return StepResult(
                step_number=2, name="Environmental Objectives",
                status="FAIL", message=f"Error: {str(e)}"
            )

    async def _step_3_nace_activities(self, answers: Dict[str, Any]) -> StepResult:
        """Step 3: Map company activities to NACE codes."""
        logger.info("Step 3: NACE Activities")

        try:
            nace_codes = answers.get(
                "nace_codes",
                self._setup_data.get("nace_codes", [])
            )

            self._setup_data["nace_codes"] = nace_codes

            if not nace_codes and not self.config.auto_configure:
                return StepResult(
                    step_number=3,
                    name="NACE Activities",
                    status="WARN",
                    message="No NACE codes provided (activities will be mapped later)",
                    data={"nace_codes": []}
                )

            return StepResult(
                step_number=3,
                name="NACE Activities",
                status="PASS",
                message=f"Mapped {len(nace_codes)} NACE codes",
                data={"nace_codes": nace_codes, "count": len(nace_codes)}
            )

        except Exception as e:
            return StepResult(
                step_number=3, name="NACE Activities",
                status="FAIL", message=f"Error: {str(e)}"
            )

    async def _step_4_financial_data_sources(self, answers: Dict[str, Any]) -> StepResult:
        """Step 4: Configure financial data sources."""
        logger.info("Step 4: Financial Data Sources")

        try:
            valid_sources = ["erp", "excel", "api", "manual"]
            data_source = answers.get("data_source", "erp")
            currency = answers.get("currency", "EUR")
            fiscal_year_end = answers.get("fiscal_year_end", "12-31")

            if data_source not in valid_sources:
                return StepResult(
                    step_number=4,
                    name="Financial Data Sources",
                    status="FAIL",
                    message=f"Invalid data source: {data_source}",
                    data={"valid_sources": valid_sources}
                )

            financial_config = {
                "data_source": data_source,
                "currency": currency,
                "fiscal_year_end": fiscal_year_end
            }

            self._setup_data["financial"] = financial_config

            return StepResult(
                step_number=4,
                name="Financial Data Sources",
                status="PASS",
                message=f"Financial data: {data_source} ({currency})",
                data=financial_config
            )

        except Exception as e:
            return StepResult(
                step_number=4, name="Financial Data Sources",
                status="FAIL", message=f"Error: {str(e)}"
            )

    async def _step_5_reporting_period(self, answers: Dict[str, Any]) -> StepResult:
        """Step 5: Configure reporting period."""
        logger.info("Step 5: Reporting Period")

        try:
            reporting_year = answers.get("reporting_year", 2025)
            period_type = answers.get("period_type", "annual")

            period_config = {
                "reporting_year": reporting_year,
                "period_type": period_type,
                "period_start": f"{reporting_year}-01-01",
                "period_end": f"{reporting_year}-12-31"
            }

            self._setup_data["reporting_period"] = period_config

            return StepResult(
                step_number=5,
                name="Reporting Period",
                status="PASS",
                message=f"Reporting period: FY{reporting_year} ({period_type})",
                data=period_config
            )

        except Exception as e:
            return StepResult(
                step_number=5, name="Reporting Period",
                status="FAIL", message=f"Error: {str(e)}"
            )

    async def _step_6_disclosure_requirements(self, answers: Dict[str, Any]) -> StepResult:
        """Step 6: Configure disclosure requirements."""
        logger.info("Step 6: Disclosure Requirements")

        try:
            valid_formats = ["article_8", "eba_pillar_3", "both"]
            disclosure_format = answers.get(
                "disclosure_format",
                self._setup_data.get("disclosure_format", "article_8")
            )

            if disclosure_format not in valid_formats:
                return StepResult(
                    step_number=6,
                    name="Disclosure Requirements",
                    status="FAIL",
                    message=f"Invalid format: {disclosure_format}",
                    data={"valid_formats": valid_formats}
                )

            org_type = self._setup_data.get("organization_type", "")
            if org_type == "financial_institution" and disclosure_format == "article_8":
                disclosure_format = "both"
                logger.info("Auto-upgraded to 'both' for financial institution")

            disclosure_config = {
                "format": disclosure_format,
                "include_nuclear_gas": answers.get("include_nuclear_gas", False),
                "include_yoy_comparison": answers.get("include_yoy_comparison", True),
                "enable_xbrl": answers.get("enable_xbrl", False)
            }

            self._setup_data["disclosure"] = disclosure_config

            return StepResult(
                step_number=6,
                name="Disclosure Requirements",
                status="PASS",
                message=f"Disclosure format: {disclosure_format}",
                data=disclosure_config
            )

        except Exception as e:
            return StepResult(
                step_number=6, name="Disclosure Requirements",
                status="FAIL", message=f"Error: {str(e)}"
            )

    async def _step_7_agent_configuration(self, answers: Dict[str, Any]) -> StepResult:
        """Step 7: Configure optional agent settings."""
        logger.info("Step 7: Agent Configuration")

        try:
            agent_config = {
                "mrv_agents_enabled": answers.get("mrv_agents_enabled", True),
                "data_agents_enabled": answers.get("data_agents_enabled", True),
                "foundation_agents_enabled": answers.get("foundation_agents_enabled", True),
                "cross_framework_enabled": answers.get("cross_framework_enabled", True),
                "parallel_processing": answers.get("parallel_processing", True),
                "batch_size": answers.get("batch_size", 500)
            }

            self._setup_data["agents"] = agent_config

            return StepResult(
                step_number=7,
                name="Agent Configuration",
                status="PASS",
                message="Agent configuration set",
                data=agent_config
            )

        except Exception as e:
            return StepResult(
                step_number=7, name="Agent Configuration",
                status="FAIL", message=f"Error: {str(e)}"
            )

    async def _step_8_data_connections(self, answers: Dict[str, Any]) -> StepResult:
        """Step 8: Configure data connections."""
        logger.info("Step 8: Data Connections")

        try:
            connection_config = {
                "database_url": answers.get("database_url", ""),
                "cache_url": answers.get("cache_url", ""),
                "erp_endpoint": answers.get("erp_endpoint", ""),
                "taxonomy_app_url": answers.get(
                    "taxonomy_app_url",
                    "https://api.greenlang.com/taxonomy/v1"
                )
            }

            self._setup_data["connections"] = connection_config

            # Warn if critical connections missing
            if not connection_config["database_url"]:
                return StepResult(
                    step_number=8,
                    name="Data Connections",
                    status="WARN",
                    message="No database URL configured (will use defaults)",
                    data=connection_config
                )

            return StepResult(
                step_number=8,
                name="Data Connections",
                status="PASS",
                message="Data connections configured",
                data=connection_config
            )

        except Exception as e:
            return StepResult(
                step_number=8, name="Data Connections",
                status="FAIL", message=f"Error: {str(e)}"
            )

    async def _step_9_validation(self) -> StepResult:
        """Step 9: Run configuration validation."""
        logger.info("Step 9: Validation")

        try:
            issues = []

            # Validate required config
            if not self._setup_data.get("organization_type"):
                issues.append("Missing organization type")

            if not self._setup_data.get("environmental_objectives"):
                issues.append("No environmental objectives selected")

            # Validate GAR requirement for financial institutions
            org_type = self._setup_data.get("organization_type", "")
            if org_type == "financial_institution":
                enable_gar = self._setup_data.get("enable_gar", False)
                if not enable_gar:
                    issues.append("GAR calculation recommended for financial institutions")

            if issues:
                return StepResult(
                    step_number=9,
                    name="Validation",
                    status="WARN" if len(issues) <= 2 else "FAIL",
                    message=f"Found {len(issues)} validation issues",
                    data={"issues": issues}
                )

            return StepResult(
                step_number=9,
                name="Validation",
                status="PASS",
                message="Configuration validation passed",
                data={"issues": [], "valid": True}
            )

        except Exception as e:
            return StepResult(
                step_number=9, name="Validation",
                status="FAIL", message=f"Validation error: {str(e)}"
            )

    async def _step_10_summary(self) -> StepResult:
        """Step 10: Generate final configuration summary."""
        logger.info("Step 10: Summary")

        try:
            final_config = {
                "pack": "PACK-008 EU Taxonomy Alignment",
                "version": "1.0.0",
                "organization_type": self._setup_data.get("organization_type"),
                "environmental_objectives": self._setup_data.get("environmental_objectives"),
                "nace_codes": self._setup_data.get("nace_codes", []),
                "financial": self._setup_data.get("financial", {}),
                "reporting_period": self._setup_data.get("reporting_period", {}),
                "disclosure": self._setup_data.get("disclosure", {}),
                "agents": self._setup_data.get("agents", {}),
                "connections": self._setup_data.get("connections", {}),
                "generated_at": datetime.utcnow().isoformat()
            }

            self._setup_data["final_config"] = final_config

            return StepResult(
                step_number=10,
                name="Summary",
                status="PASS",
                message="Configuration generated successfully",
                data=final_config
            )

        except Exception as e:
            return StepResult(
                step_number=10, name="Summary",
                status="FAIL", message=f"Summary error: {str(e)}"
            )

    async def _validate_organization_type(self, data: Dict[str, Any]) -> StepResult:
        """Validate organization type data."""
        org_type = data.get("organization_type", "")
        valid = org_type in ["non_financial_undertaking", "financial_institution", "asset_manager"]
        return StepResult(
            step_number=1, name="Organization Type",
            status="PASS" if valid else "FAIL",
            message=f"Organization type {'valid' if valid else 'invalid'}: {org_type}"
        )

    async def _validate_objectives(self, data: Dict[str, Any]) -> StepResult:
        """Validate environmental objectives data."""
        objectives = data.get("environmental_objectives", [])
        valid = all(o in ["CCM", "CCA", "WTR", "CE", "PPC", "BIO"] for o in objectives)
        return StepResult(
            step_number=2, name="Environmental Objectives",
            status="PASS" if valid and objectives else "FAIL",
            message=f"{len(objectives)} objectives {'valid' if valid else 'invalid'}"
        )

    async def _validate_nace_activities(self, data: Dict[str, Any]) -> StepResult:
        """Validate NACE activities data."""
        nace_codes = data.get("nace_codes", [])
        return StepResult(
            step_number=3, name="NACE Activities",
            status="PASS" if nace_codes else "WARN",
            message=f"{len(nace_codes)} NACE codes provided"
        )

    async def _validate_financial_sources(self, data: Dict[str, Any]) -> StepResult:
        """Validate financial source data."""
        source = data.get("data_source", "")
        valid = source in ["erp", "excel", "api", "manual"]
        return StepResult(
            step_number=4, name="Financial Data Sources",
            status="PASS" if valid else "FAIL",
            message=f"Data source {'valid' if valid else 'invalid'}: {source}"
        )

    async def _validate_reporting_period(self, data: Dict[str, Any]) -> StepResult:
        """Validate reporting period data."""
        year = data.get("reporting_year", 0)
        valid = 2020 <= year <= 2030
        return StepResult(
            step_number=5, name="Reporting Period",
            status="PASS" if valid else "FAIL",
            message=f"Reporting year {'valid' if valid else 'invalid'}: {year}"
        )

    async def _validate_disclosure_format(self, data: Dict[str, Any]) -> StepResult:
        """Validate disclosure format data."""
        fmt = data.get("disclosure_format", "")
        valid = fmt in ["article_8", "eba_pillar_3", "both"]
        return StepResult(
            step_number=6, name="Disclosure Requirements",
            status="PASS" if valid else "FAIL",
            message=f"Disclosure format {'valid' if valid else 'invalid'}: {fmt}"
        )

    async def _validate_agent_config(self, data: Dict[str, Any]) -> StepResult:
        """Validate agent configuration data."""
        return StepResult(
            step_number=7, name="Agent Configuration",
            status="PASS", message="Agent configuration valid"
        )

    async def _validate_data_connections(self, data: Dict[str, Any]) -> StepResult:
        """Validate data connections data."""
        return StepResult(
            step_number=8, name="Data Connections",
            status="PASS", message="Data connections valid"
        )

    async def _validate_overall(self, data: Dict[str, Any]) -> StepResult:
        """Validate overall configuration."""
        return StepResult(
            step_number=9, name="Validation",
            status="PASS", message="Overall configuration valid"
        )

    async def _validate_summary(self, data: Dict[str, Any]) -> StepResult:
        """Validate summary data."""
        return StepResult(
            step_number=10, name="Summary",
            status="PASS", message="Summary valid"
        )

    def _build_result(self, steps: List[StepResult]) -> SetupResult:
        """Build final setup result from steps."""
        total = len(steps)
        passed = sum(1 for s in steps if s.status == "PASS")
        warned = sum(1 for s in steps if s.status == "WARN")
        failed = sum(1 for s in steps if s.status == "FAIL")
        skipped = sum(1 for s in steps if s.status == "SKIP")

        if failed > 0:
            overall = "FAIL"
        elif warned > 0:
            overall = "WARN"
        else:
            overall = "PASS"

        return SetupResult(
            overall_status=overall,
            total_steps=total,
            passed=passed,
            warned=warned,
            failed=failed,
            skipped=skipped,
            steps=steps,
            generated_config=self._setup_data.get("final_config", {})
        )

    def get_setup_summary(self) -> Dict[str, Any]:
        """Get summary of current setup configuration."""
        return {
            "pack": "PACK-008 EU Taxonomy Alignment",
            "configuration": self._setup_data,
            "features": {
                "organization_type": self._setup_data.get("organization_type", "not_set"),
                "objectives_count": len(
                    self._setup_data.get("environmental_objectives", [])
                ),
                "nace_codes_count": len(self._setup_data.get("nace_codes", [])),
                "disclosure_format": self._setup_data.get("disclosure", {}).get("format", "not_set"),
                "gar_enabled": self._setup_data.get("enable_gar", False)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
