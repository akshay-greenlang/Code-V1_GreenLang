# -*- coding: utf-8 -*-
"""
CSRD Reporting Platform - Python SDK

Simple, Pythonic API for CSRD reporting per European Sustainability Reporting Standards (ESRS).

Design Philosophy:
- One main function: csrd_build_report()
- Accepts files OR DataFrames OR dictionaries
- Returns structured Python objects
- Composable (can use individual agents)
- Zero hallucination for calculations (deterministic)
- AI-assisted for materiality (requires human review)

Version: 1.0.0
Author: GreenLang CSRD Team
"""

import json
import os
import sys
import tempfile
import yaml
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import pandas as pd
from greenlang.determinism import DeterministicClock

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import agents
from agents.intake_agent import IntakeAgent
from agents.materiality_agent import MaterialityAgent, LLMConfig
from agents.calculator_agent import CalculatorAgent
from agents.aggregator_agent import AggregatorAgent
from agents.reporting_agent import ReportingAgent
from agents.audit_agent import AuditAgent


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CSRDConfig:
    """
    CSRD configuration for repeated use.

    Stores company information and default paths so you don't have to
    repeat them for every report.

    Example:
        config = CSRDConfig(
            company_name="Acme Corp",
            company_lei="529900ABCDEFGHIJKLMN",
            reporting_year=2024,
            sector="Manufacturing"
        )

        report = csrd_build_report(
            esg_data="data.csv",
            company_profile="profile.json",
            config=config
        )
    """
    # Company info
    company_name: str
    company_lei: str  # Legal Entity Identifier
    reporting_year: int
    sector: str

    # Optional company details
    country: str = "DE"
    employee_count: Optional[int] = None
    revenue: Optional[float] = None
    total_assets: Optional[float] = None

    # Optional paths
    esrs_data_points_path: str = "data/esrs_data_points.json"
    data_quality_rules_path: str = "rules/data_quality_rules.yaml"
    esrs_formulas_path: str = "rules/esrs_formulas.yaml"
    emission_factors_path: str = "data/emission_factors.json"
    compliance_rules_path: str = "rules/compliance_rules.yaml"
    framework_mappings_path: str = "data/framework_mappings.json"

    # LLM config for materiality assessment
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    llm_api_key: Optional[str] = None

    # Thresholds
    quality_threshold: float = 0.80
    impact_materiality_threshold: float = 5.0
    financial_materiality_threshold: float = 5.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CSRDConfig':
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'CSRDConfig':
        """Load from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        # Flatten structure if needed
        if "company" in data:
            return cls(
                company_name=data["company"]["name"],
                company_lei=data["company"]["lei"],
                reporting_year=data["company"].get("reporting_year", DeterministicClock.now().year),
                sector=data["company"].get("sector", ""),
                country=data["company"].get("country", "DE"),
                employee_count=data["company"].get("employee_count"),
                revenue=data["company"].get("revenue"),
                total_assets=data["company"].get("total_assets"),
                esrs_data_points_path=data.get("paths", {}).get("esrs_data_points", "data/esrs_data_points.json"),
                data_quality_rules_path=data.get("paths", {}).get("data_quality_rules", "rules/data_quality_rules.yaml"),
                esrs_formulas_path=data.get("paths", {}).get("esrs_formulas", "rules/esrs_formulas.yaml"),
                emission_factors_path=data.get("paths", {}).get("emission_factors", "data/emission_factors.json"),
                compliance_rules_path=data.get("paths", {}).get("compliance_rules", "rules/compliance_rules.yaml"),
                framework_mappings_path=data.get("paths", {}).get("framework_mappings", "data/framework_mappings.json"),
                llm_provider=data.get("llm", {}).get("provider", "openai"),
                llm_model=data.get("llm", {}).get("model", "gpt-4o"),
                llm_api_key=data.get("llm", {}).get("api_key"),
                quality_threshold=data.get("thresholds", {}).get("quality", 0.80),
                impact_materiality_threshold=data.get("thresholds", {}).get("impact_materiality", 5.0),
                financial_materiality_threshold=data.get("thresholds", {}).get("financial_materiality", 5.0)
            )
        else:
            return cls(**data)

    @classmethod
    def from_env(cls) -> 'CSRDConfig':
        """Load from environment variables."""
        return cls(
            company_name=os.getenv("CSRD_COMPANY_NAME", ""),
            company_lei=os.getenv("CSRD_COMPANY_LEI", ""),
            reporting_year=int(os.getenv("CSRD_REPORTING_YEAR", DeterministicClock.now().year)),
            sector=os.getenv("CSRD_SECTOR", ""),
            country=os.getenv("CSRD_COUNTRY", "DE"),
            llm_api_key=os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        )


@dataclass
class ESRSMetrics:
    """
    Calculated ESRS metrics with complete provenance.

    All calculations are deterministic (zero hallucination guarantee).
    """
    total_metrics_calculated: int
    metrics_by_standard: Dict[str, int]

    # Climate metrics (E1)
    scope_1_emissions_tco2e: Optional[float] = None
    scope_2_emissions_tco2e: Optional[float] = None
    scope_3_emissions_tco2e: Optional[float] = None
    total_ghg_emissions_tco2e: Optional[float] = None
    ghg_intensity: Optional[float] = None
    total_energy_consumption_mwh: Optional[float] = None
    renewable_energy_percentage: Optional[float] = None

    # Social metrics (S1)
    total_workforce: Optional[int] = None
    employee_turnover_rate: Optional[float] = None
    gender_pay_gap: Optional[float] = None
    work_related_accidents: Optional[int] = None

    # Governance metrics (G1)
    board_gender_diversity: Optional[float] = None
    ethics_violations: Optional[int] = None

    # Metadata
    calculation_method: str = "deterministic"
    zero_hallucination_guarantee: bool = True
    processing_time_seconds: Optional[float] = None

    # Raw data access
    raw_metrics: Dict[str, Any] = None


@dataclass
class MaterialityAssessment:
    """
    Double materiality assessment results.

    AI-generated assessments REQUIRE human review.
    """
    total_topics_assessed: int
    material_topics_count: int
    material_from_impact: int
    material_from_financial: int
    double_material_count: int

    # Material ESRS standards triggered
    esrs_standards_triggered: List[str] = None

    # Material topics details
    material_topics: List[Dict[str, Any]] = None

    # AI metadata
    ai_powered: bool = True
    requires_human_review: bool = True
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    average_confidence: Optional[float] = None
    review_flags_count: int = 0

    # Processing
    processing_time_minutes: Optional[float] = None


@dataclass
class ComplianceStatus:
    """
    ESRS compliance validation results.

    Deterministic rule-based validation (zero hallucination).
    """
    compliance_status: str  # "PASS", "FAIL", "WARNING"
    total_rules_checked: int
    rules_passed: int
    rules_failed: int
    rules_warning: int

    # Failure breakdown
    critical_failures: int = 0
    major_failures: int = 0
    minor_failures: int = 0

    # Details
    failed_rules: List[Dict[str, Any]] = None
    warning_rules: List[Dict[str, Any]] = None

    # Audit readiness
    audit_ready: bool = False
    audit_package_generated: bool = False

    # Metadata
    validation_timestamp: Optional[str] = None
    validation_duration_seconds: Optional[float] = None


@dataclass
class CSRDReport:
    """
    Complete CSRD report result with convenient access methods.

    Wraps the full report data with Pythonic attribute access
    and utility methods.

    Example:
        report = csrd_build_report(...)

        print(f"Compliance: {report.compliance_status.compliance_status}")
        print(f"Material topics: {report.materiality.material_topics_count}")
        print(f"GHG emissions: {report.metrics.total_ghg_emissions_tco2e:.2f} tCO2e")

        # Save outputs
        report.save_json("output/report.json")
        report.save_summary("output/summary.md")

        # Export to DataFrame
        df = report.to_dataframe()
    """
    # Core components
    company_info: Dict[str, Any]
    reporting_period: Dict[str, Any]

    # Results
    data_validation: Dict[str, Any]
    materiality: MaterialityAssessment
    metrics: ESRSMetrics
    compliance_status: ComplianceStatus

    # Optional components
    aggregated_frameworks: Optional[Dict[str, Any]] = None
    xbrl_report: Optional[Dict[str, Any]] = None
    audit_package: Optional[Dict[str, Any]] = None

    # Metadata
    report_id: Optional[str] = None
    generated_at: Optional[str] = None
    processing_time_total_minutes: Optional[float] = None

    # Warnings/info
    warnings: List[str] = None
    info_messages: List[str] = None

    # Raw data (complete report package)
    raw_report: Dict[str, Any] = None

    @property
    def is_compliant(self) -> bool:
        """Whether report passes all compliance checks."""
        return self.compliance_status.compliance_status == "PASS"

    @property
    def is_audit_ready(self) -> bool:
        """Whether report is ready for external audit."""
        return self.compliance_status.audit_ready

    @property
    def material_standards(self) -> List[str]:
        """List of material ESRS standards."""
        return self.materiality.esrs_standards_triggered or []

    def to_dict(self) -> Dict[str, Any]:
        """Get raw report dictionary."""
        return self.raw_report or asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save_json(self, path: str):
        """Save complete report to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def save_summary(self, path: str):
        """Save human-readable summary to Markdown file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.summary())

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert validated ESG data points to pandas DataFrame.

        Returns:
            DataFrame with one row per data point
        """
        if self.raw_report and "validated_data" in self.raw_report:
            data_points = self.raw_report["validated_data"].get("data_points", [])
            return pd.DataFrame(data_points)
        return pd.DataFrame()

    def summary(self) -> str:
        """
        Get a quick text summary of the report.

        Returns:
            Multi-line string with key metrics
        """
        warnings_section = ""
        if self.materiality.requires_human_review:
            warnings_section = "\nWARNING: AI-generated materiality assessment requires human review"

        if self.compliance_status.compliance_status == "FAIL":
            warnings_section += f"\nWARNING: Compliance validation FAILED ({self.compliance_status.rules_failed} rules)"

        return f"""
CSRD Report Summary
===================
Report ID: {self.report_id}
Company: {self.company_info.get('legal_name', 'Unknown')}
Reporting Period: {self.reporting_period.get('year', 'Unknown')}
Generated: {self.generated_at}

Materiality Assessment:
- Total Topics Assessed: {self.materiality.total_topics_assessed}
- Material Topics: {self.materiality.material_topics_count}
- Material ESRS Standards: {', '.join(self.material_standards)}
- AI Confidence: {self.materiality.average_confidence:.0%}

ESRS Metrics Calculated:
- Total Metrics: {self.metrics.total_metrics_calculated}
- GHG Emissions: {self.metrics.total_ghg_emissions_tco2e:.2f} tCO2e (Scope 1+2+3)
- Workforce: {self.metrics.total_workforce} employees
- Processing Time: {self.metrics.processing_time_seconds:.2f}s

Compliance Status: {self.compliance_status.compliance_status}
- Rules Checked: {self.compliance_status.total_rules_checked}
- Rules Passed: {self.compliance_status.rules_passed}
- Rules Failed: {self.compliance_status.rules_failed}
- Audit Ready: {'Yes' if self.compliance_status.audit_ready else 'No'}
{warnings_section}

Total Processing Time: {self.processing_time_total_minutes:.1f} minutes
        """.strip()

    def __repr__(self) -> str:
        return f"CSRDReport(report_id='{self.report_id}', company='{self.company_info.get('legal_name')}', " \
               f"compliance='{self.compliance_status.compliance_status}', material_topics={self.materiality.material_topics_count})"


# ============================================================================
# HELPER FUNCTIONS - INPUT HANDLING
# ============================================================================

def _load_input_data(
    input_data: Union[str, Path, pd.DataFrame, Dict, List],
    data_type: str = "esg_data"
) -> Tuple[Optional[str], Optional[pd.DataFrame], Optional[Dict]]:
    """
    Load and normalize input data from various formats.

    Args:
        input_data: Input in various formats
        data_type: Type of data ("esg_data", "company_profile", etc.)

    Returns:
        Tuple of (file_path, dataframe, dict)
    """
    # File path (string or Path)
    if isinstance(input_data, (str, Path)):
        input_path = Path(input_data)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # For company_profile, return dict
        if data_type == "company_profile":
            if input_path.suffix.lower() == '.json':
                with open(input_path, 'r') as f:
                    return None, None, json.load(f)
            elif input_path.suffix.lower() in ['.yaml', '.yml']:
                with open(input_path, 'r') as f:
                    return None, None, yaml.safe_load(f)

        # For ESG data, return file path
        return str(input_path), None, None

    # DataFrame
    elif isinstance(input_data, pd.DataFrame):
        return None, input_data, None

    # Dictionary
    elif isinstance(input_data, dict):
        return None, None, input_data

    # List (convert to DataFrame)
    elif isinstance(input_data, list):
        return None, pd.DataFrame(input_data), None

    else:
        raise ValueError(f"Unsupported input type: {type(input_data)}")


def _save_dataframe_to_temp(df: pd.DataFrame) -> str:
    """Save DataFrame to temporary CSV file."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    return temp_file.name


def _build_company_context(
    company_profile: Dict[str, Any],
    config: Optional[CSRDConfig] = None
) -> Dict[str, Any]:
    """Build company context from profile and config."""
    context = {
        "company_info": {},
        "business_profile": {},
        "company_size": {},
        "reporting_scope": {}
    }

    # Merge from profile
    if "company_info" in company_profile:
        context["company_info"] = company_profile["company_info"]
    elif "legal_name" in company_profile:
        context["company_info"] = {
            "legal_name": company_profile.get("legal_name"),
            "lei": company_profile.get("lei"),
            "country": company_profile.get("country")
        }

    # Merge from config
    if config:
        if not context["company_info"].get("legal_name"):
            context["company_info"]["legal_name"] = config.company_name
        if not context["company_info"].get("lei"):
            context["company_info"]["lei"] = config.company_lei
        if not context["company_info"].get("country"):
            context["company_info"]["country"] = config.country

        context["business_profile"]["sector"] = config.sector
        context["company_size"]["revenue"] = {"total_revenue": config.revenue} if config.revenue else {}
        context["company_size"]["total_assets"] = config.total_assets
        context["reporting_scope"]["reporting_year"] = config.reporting_year

    # Fill from company_profile
    if "business_profile" in company_profile:
        context["business_profile"].update(company_profile["business_profile"])
    if "company_size" in company_profile:
        context["company_size"].update(company_profile["company_size"])
    if "reporting_scope" in company_profile:
        context["reporting_scope"].update(company_profile["reporting_scope"])

    return context


# ============================================================================
# MAIN SDK FUNCTION
# ============================================================================

def csrd_build_report(
    esg_data: Union[str, Path, pd.DataFrame],
    company_profile: Union[str, Path, Dict],
    config: Optional[CSRDConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
    skip_materiality: bool = False,
    skip_audit: bool = False,
    verbose: bool = False,

    # Override config paths
    esrs_data_points_path: Optional[str] = None,
    data_quality_rules_path: Optional[str] = None,
    esrs_formulas_path: Optional[str] = None,
    emission_factors_path: Optional[str] = None,
    compliance_rules_path: Optional[str] = None,

    # Override thresholds
    quality_threshold: Optional[float] = None,
    impact_materiality_threshold: Optional[float] = None,
    financial_materiality_threshold: Optional[float] = None,

    # LLM config
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None
) -> CSRDReport:
    """
    Build complete CSRD report in one function call.

    This is the main SDK function. Accepts ESG data and company profile in various
    formats, and returns a complete CSRD report with:
    - Data validation and quality assessment
    - Double materiality assessment (AI-assisted)
    - ESRS metrics calculation (deterministic)
    - Compliance validation
    - Audit package generation

    Args:
        esg_data: Path to ESG data file (CSV/JSON/Excel/Parquet) OR DataFrame
        company_profile: Path to company profile file (JSON/YAML) OR Dict
        config: CSRDConfig object (alternative to individual params)
        output_dir: Optional directory to save all outputs
        skip_materiality: Skip materiality assessment (use for testing)
        skip_audit: Skip audit compliance checks
        verbose: Enable verbose logging

        # Path overrides (optional)
        esrs_data_points_path: Path to ESRS data points catalog
        data_quality_rules_path: Path to data quality rules
        esrs_formulas_path: Path to ESRS formulas
        emission_factors_path: Path to emission factors
        compliance_rules_path: Path to compliance rules

        # Threshold overrides (optional)
        quality_threshold: Minimum data quality score (0.0-1.0)
        impact_materiality_threshold: Impact materiality threshold (0-10)
        financial_materiality_threshold: Financial materiality threshold (0-10)

        # LLM config overrides (optional)
        llm_provider: LLM provider ("openai" or "anthropic")
        llm_model: LLM model name
        llm_api_key: LLM API key

    Returns:
        CSRDReport object with complete results and convenience methods

    Examples:
        # From files
        report = csrd_build_report(
            esg_data="data/esg_data.csv",
            company_profile="data/company.json"
        )

        # From DataFrame with config
        import pandas as pd
        df = pd.read_csv("esg_data.csv")
        config = CSRDConfig.from_yaml(".csrd.yaml")
        report = csrd_build_report(
            esg_data=df,
            company_profile="company.json",
            config=config
        )

        # With output directory
        report = csrd_build_report(
            esg_data="data.csv",
            company_profile="company.json",
            output_dir="output/csrd_2024"
        )

        # Access results
        print(f"Compliance: {report.compliance_status.compliance_status}")
        print(f"Material topics: {report.materiality.material_topics_count}")
        print(f"GHG emissions: {report.metrics.total_ghg_emissions_tco2e:.2f} tCO2e")

        # Save outputs
        report.save_json("output/report.json")
        report.save_summary("output/summary.md")
    """
    start_time = DeterministicClock.now()

    # Configure logging
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    # Load/merge configuration
    if config is None:
        # Try to load from environment or create minimal config
        config = CSRDConfig(
            company_name="",
            company_lei="",
            reporting_year=DeterministicClock.now().year,
            sector=""
        )

    # Apply overrides
    if esrs_data_points_path:
        config.esrs_data_points_path = esrs_data_points_path
    if data_quality_rules_path:
        config.data_quality_rules_path = data_quality_rules_path
    if esrs_formulas_path:
        config.esrs_formulas_path = esrs_formulas_path
    if emission_factors_path:
        config.emission_factors_path = emission_factors_path
    if compliance_rules_path:
        config.compliance_rules_path = compliance_rules_path
    if quality_threshold is not None:
        config.quality_threshold = quality_threshold
    if impact_materiality_threshold is not None:
        config.impact_materiality_threshold = impact_materiality_threshold
    if financial_materiality_threshold is not None:
        config.financial_materiality_threshold = financial_materiality_threshold
    if llm_provider:
        config.llm_provider = llm_provider
    if llm_model:
        config.llm_model = llm_model
    if llm_api_key:
        config.llm_api_key = llm_api_key

    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # Load inputs
    esg_file, esg_df, _ = _load_input_data(esg_data, "esg_data")
    _, _, company_dict = _load_input_data(company_profile, "company_profile")

    # If DataFrame provided, save to temp file for agent processing
    if esg_df is not None and esg_file is None:
        esg_file = _save_dataframe_to_temp(esg_df)

    if not esg_file:
        raise ValueError("Could not load ESG data")

    if not company_dict:
        raise ValueError("Could not load company profile")

    # Build company context
    company_context = _build_company_context(company_dict, config)

    # Initialize warnings/info
    warnings = []
    info_messages = []

    # ========================================================================
    # STEP 1: DATA VALIDATION & INTAKE
    # ========================================================================
    print("Step 1/6: Validating ESG data...")

    intake_agent = IntakeAgent(
        esrs_data_points_path=config.esrs_data_points_path,
        data_quality_rules_path=config.data_quality_rules_path,
        quality_threshold=config.quality_threshold
    )

    intake_output_file = str(Path(output_dir) / "01_validated_data.json") if output_dir else None

    validated_data = intake_agent.process(
        input_file=esg_file,
        company_profile=company_context,
        output_file=intake_output_file
    )

    # Check data quality
    if not validated_data["metadata"]["quality_threshold_met"]:
        warnings.append(f"Data quality score ({validated_data['metadata']['data_quality_score']:.1f}) below threshold")

    print(f"  - Validated {validated_data['metadata']['valid_records']}/{validated_data['metadata']['total_records']} records")
    print(f"  - Data quality score: {validated_data['metadata']['data_quality_score']:.1f}/100")

    # ========================================================================
    # STEP 2: MATERIALITY ASSESSMENT (AI-ASSISTED)
    # ========================================================================
    materiality_result = None

    if not skip_materiality:
        print("Step 2/6: Conducting double materiality assessment (AI-assisted)...")

        llm_config = LLMConfig(
            provider=config.llm_provider,
            model=config.llm_model,
            api_key=config.llm_api_key
        )

        materiality_agent = MaterialityAgent(
            esrs_data_points_path=config.esrs_data_points_path,
            llm_config=llm_config,
            impact_threshold=config.impact_materiality_threshold,
            financial_threshold=config.financial_materiality_threshold
        )

        materiality_output_file = str(Path(output_dir) / "02_materiality_assessment.json") if output_dir else None

        materiality_result = materiality_agent.process(
            company_context=company_context,
            esg_data=validated_data,
            output_file=materiality_output_file
        )

        warnings.append("AI-generated materiality assessment requires human review")

        print(f"  - Material topics: {materiality_result['summary_statistics']['material_topics_count']}/{materiality_result['summary_statistics']['total_topics_assessed']}")
        print(f"  - Average confidence: {materiality_result['ai_metadata']['average_confidence']:.0%}")
        print(f"  - WARNING: Human review required ({len(materiality_result['review_flags'])} items flagged)")
    else:
        print("Step 2/6: Skipping materiality assessment")
        # Create dummy materiality result
        materiality_result = {
            "material_topics": [],
            "summary_statistics": {
                "total_topics_assessed": 0,
                "material_topics_count": 0,
                "material_from_impact": 0,
                "material_from_financial": 0,
                "double_material_count": 0,
                "esrs_standards_triggered": []
            },
            "ai_metadata": {
                "llm_provider": "none",
                "llm_model": "none",
                "average_confidence": 0.0,
                "processing_time_minutes": 0.0
            },
            "review_flags": []
        }

    # ========================================================================
    # STEP 3: METRICS CALCULATION (ZERO HALLUCINATION)
    # ========================================================================
    print("Step 3/6: Calculating ESRS metrics (deterministic, zero hallucination)...")

    calculator_agent = CalculatorAgent(
        esrs_formulas_path=config.esrs_formulas_path,
        emission_factors_path=config.emission_factors_path
    )

    # Extract material standards to calculate
    material_standards = materiality_result["summary_statistics"].get("esrs_standards_triggered", [])

    # For testing/demo, calculate some core metrics even without materiality
    if not material_standards:
        material_standards = ["E1", "S1", "G1"]

    # Build list of metrics to calculate
    metrics_to_calculate = []
    for standard in material_standards:
        # Add core metrics for each standard (simplified for demo)
        if standard == "E1":
            metrics_to_calculate.extend(["E1-1", "E1-2", "E1-3", "E1-4", "E1-5", "E1-6"])
        elif standard == "S1":
            metrics_to_calculate.extend(["S1-1", "S1-2", "S1-3", "S1-4"])
        elif standard == "G1":
            metrics_to_calculate.extend(["G1-1", "G1-2"])

    # Prepare input data for calculations (from validated data)
    calculation_input = {}
    for dp in validated_data.get("data_points", []):
        metric_code = dp.get("metric_code")
        if metric_code:
            calculation_input[metric_code] = dp.get("value")

    calculation_output_file = str(Path(output_dir) / "03_calculated_metrics.json") if output_dir else None

    calculated_metrics = calculator_agent.calculate_batch(
        metric_codes=metrics_to_calculate,
        input_data=calculation_input
    )

    if calculation_output_file:
        calculator_agent.write_output(calculated_metrics, calculation_output_file)

    print(f"  - Calculated {calculated_metrics['metadata']['metrics_calculated']}/{calculated_metrics['metadata']['total_metrics_requested']} metrics")
    print(f"  - Processing: {calculated_metrics['metadata']['ms_per_metric']:.2f} ms/metric")
    print(f"  - Zero hallucination: {calculated_metrics['metadata']['zero_hallucination_guarantee']}")

    # ========================================================================
    # STEP 4: FRAMEWORK AGGREGATION (Optional - skipped for now)
    # ========================================================================
    print("Step 4/6: Aggregating frameworks (skipped)")
    aggregated_frameworks = None

    # ========================================================================
    # STEP 5: REPORT GENERATION (Simplified for SDK)
    # ========================================================================
    print("Step 5/6: Generating report package...")
    xbrl_report = None  # XBRL generation can be added later

    # ========================================================================
    # STEP 6: COMPLIANCE AUDIT
    # ========================================================================
    compliance_result = None

    if not skip_audit:
        print("Step 6/6: Validating ESRS compliance...")

        audit_agent = AuditAgent(
            compliance_rules_path=config.compliance_rules_path
        )

        # Build report data for audit
        audit_input = {
            "materiality_assessment": materiality_result,
            "metrics": calculated_metrics,
            "validated_data": validated_data,
            "material_standards": material_standards
        }

        audit_output_file = str(Path(output_dir) / "06_audit_compliance.json") if output_dir else None

        compliance_result = audit_agent.audit(
            report_data=audit_input,
            output_file=audit_output_file
        )

        print(f"  - Compliance status: {compliance_result['compliance_report']['compliance_status']}")
        print(f"  - Rules passed: {compliance_result['compliance_report']['rules_passed']}/{compliance_result['compliance_report']['total_rules_checked']}")

        if compliance_result['compliance_report']['rules_failed'] > 0:
            warnings.append(f"Compliance validation FAILED ({compliance_result['compliance_report']['rules_failed']} rules)")
    else:
        print("Step 6/6: Skipping compliance audit")
        # Create dummy compliance result
        compliance_result = {
            "compliance_report": {
                "compliance_status": "SKIPPED",
                "total_rules_checked": 0,
                "rules_passed": 0,
                "rules_failed": 0,
                "rules_warning": 0,
                "critical_failures": 0,
                "major_failures": 0,
                "minor_failures": 0,
                "validation_timestamp": DeterministicClock.now().isoformat(),
                "validation_duration_seconds": 0.0
            },
            "rule_results": [],
            "audit_package": None
        }

    # ========================================================================
    # BUILD CSRD REPORT OBJECT
    # ========================================================================

    end_time = DeterministicClock.now()
    processing_time = (end_time - start_time).total_seconds() / 60.0

    # Extract key metrics from calculated results
    metrics_dict = {m["metric_code"]: m["value"] for m in calculated_metrics.get("calculated_metrics", [])}

    metrics = ESRSMetrics(
        total_metrics_calculated=calculated_metrics["metadata"]["metrics_calculated"],
        metrics_by_standard={std: len([m for m in metrics_to_calculate if m.startswith(std)]) for std in material_standards},
        scope_1_emissions_tco2e=metrics_dict.get("E1-1"),
        scope_2_emissions_tco2e=metrics_dict.get("E1-2"),
        scope_3_emissions_tco2e=metrics_dict.get("E1-3"),
        total_ghg_emissions_tco2e=metrics_dict.get("E1-4"),
        ghg_intensity=metrics_dict.get("E1-5"),
        total_energy_consumption_mwh=metrics_dict.get("E1-6"),
        total_workforce=int(metrics_dict.get("S1-1", 0)) if metrics_dict.get("S1-1") else None,
        employee_turnover_rate=metrics_dict.get("S1-2"),
        gender_pay_gap=metrics_dict.get("S1-3"),
        work_related_accidents=int(metrics_dict.get("S1-4", 0)) if metrics_dict.get("S1-4") else None,
        board_gender_diversity=metrics_dict.get("G1-1"),
        ethics_violations=int(metrics_dict.get("G1-2", 0)) if metrics_dict.get("G1-2") else None,
        processing_time_seconds=calculated_metrics["metadata"]["processing_time_seconds"],
        raw_metrics=calculated_metrics
    )

    materiality_assessment = MaterialityAssessment(
        total_topics_assessed=materiality_result["summary_statistics"]["total_topics_assessed"],
        material_topics_count=materiality_result["summary_statistics"]["material_topics_count"],
        material_from_impact=materiality_result["summary_statistics"]["material_from_impact"],
        material_from_financial=materiality_result["summary_statistics"]["material_from_financial"],
        double_material_count=materiality_result["summary_statistics"]["double_material_count"],
        esrs_standards_triggered=materiality_result["summary_statistics"].get("esrs_standards_triggered", []),
        material_topics=materiality_result.get("material_topics", []),
        llm_provider=materiality_result["ai_metadata"]["llm_provider"],
        llm_model=materiality_result["ai_metadata"]["llm_model"],
        average_confidence=materiality_result["ai_metadata"]["average_confidence"],
        review_flags_count=len(materiality_result.get("review_flags", [])),
        processing_time_minutes=materiality_result["ai_metadata"]["processing_time_minutes"]
    )

    compliance_status = ComplianceStatus(
        compliance_status=compliance_result["compliance_report"]["compliance_status"],
        total_rules_checked=compliance_result["compliance_report"]["total_rules_checked"],
        rules_passed=compliance_result["compliance_report"]["rules_passed"],
        rules_failed=compliance_result["compliance_report"]["rules_failed"],
        rules_warning=compliance_result["compliance_report"]["rules_warning"],
        critical_failures=compliance_result["compliance_report"]["critical_failures"],
        major_failures=compliance_result["compliance_report"]["major_failures"],
        minor_failures=compliance_result["compliance_report"]["minor_failures"],
        failed_rules=[r for r in compliance_result.get("rule_results", []) if r.get("status") == "fail"],
        warning_rules=[r for r in compliance_result.get("rule_results", []) if r.get("status") == "warning"],
        audit_ready=(compliance_result["compliance_report"]["compliance_status"] == "PASS"),
        audit_package_generated=(compliance_result.get("audit_package") is not None),
        validation_timestamp=compliance_result["compliance_report"]["validation_timestamp"],
        validation_duration_seconds=compliance_result["compliance_report"]["validation_duration_seconds"]
    )

    # Build raw report
    raw_report = {
        "report_metadata": {
            "report_id": f"CSRD-{company_context['company_info'].get('lei', 'UNKNOWN')}-{config.reporting_year}",
            "generated_at": end_time.isoformat(),
            "sdk_version": "1.0.0"
        },
        "company_info": company_context["company_info"],
        "reporting_period": {"year": config.reporting_year},
        "validated_data": validated_data,
        "materiality_assessment": materiality_result,
        "calculated_metrics": calculated_metrics,
        "compliance_validation": compliance_result,
        "aggregated_frameworks": aggregated_frameworks,
        "xbrl_report": xbrl_report,
        "warnings": warnings,
        "info_messages": info_messages
    }

    report = CSRDReport(
        company_info=company_context["company_info"],
        reporting_period={"year": config.reporting_year},
        data_validation=validated_data,
        materiality=materiality_assessment,
        metrics=metrics,
        compliance_status=compliance_status,
        aggregated_frameworks=aggregated_frameworks,
        xbrl_report=xbrl_report,
        audit_package=compliance_result.get("audit_package"),
        report_id=f"CSRD-{company_context['company_info'].get('lei', 'UNKNOWN')}-{config.reporting_year}",
        generated_at=end_time.isoformat(),
        processing_time_total_minutes=processing_time,
        warnings=warnings,
        info_messages=info_messages,
        raw_report=raw_report
    )

    # Save complete report if output_dir specified
    if output_dir:
        report.save_json(str(Path(output_dir) / "00_complete_report.json"))
        report.save_summary(str(Path(output_dir) / "00_summary.md"))

    print(f"\nReport generation complete in {processing_time:.1f} minutes")
    print(f"Compliance: {report.compliance_status.compliance_status}")

    return report


# ============================================================================
# INDIVIDUAL AGENT ACCESS FUNCTIONS
# ============================================================================

def csrd_validate_data(
    esg_data: Union[str, Path, pd.DataFrame],
    company_profile: Optional[Union[str, Path, Dict]] = None,
    config: Optional[CSRDConfig] = None,
    esrs_data_points_path: Optional[str] = None,
    data_quality_rules_path: Optional[str] = None,
    quality_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Validate ESG data without generating a full report.

    Useful for quick validation checks or pre-flight validation.

    Args:
        esg_data: Path to ESG data file OR DataFrame
        company_profile: Path to company profile OR Dict (optional)
        config: CSRDConfig object (optional)
        esrs_data_points_path: Path to ESRS data points catalog
        data_quality_rules_path: Path to data quality rules
        quality_threshold: Minimum data quality score (0.0-1.0)

    Returns:
        Validation result dictionary with metadata, data points, validation issues

    Example:
        result = csrd_validate_data(esg_data="data.csv")
        print(f"Valid: {result['metadata']['valid_records']}")
        print(f"Invalid: {result['metadata']['invalid_records']}")
        print(f"Quality score: {result['metadata']['data_quality_score']:.1f}/100")
    """
    # Setup config
    if config is None:
        config = CSRDConfig(company_name="", company_lei="", reporting_year=2024, sector="")

    if esrs_data_points_path:
        config.esrs_data_points_path = esrs_data_points_path
    if data_quality_rules_path:
        config.data_quality_rules_path = data_quality_rules_path
    if quality_threshold is not None:
        config.quality_threshold = quality_threshold

    # Load ESG data
    esg_file, esg_df, _ = _load_input_data(esg_data, "esg_data")
    if esg_df is not None and esg_file is None:
        esg_file = _save_dataframe_to_temp(esg_df)

    # Load company profile if provided
    company_context = None
    if company_profile:
        _, _, company_dict = _load_input_data(company_profile, "company_profile")
        company_context = _build_company_context(company_dict, config)

    # Initialize agent
    agent = IntakeAgent(
        esrs_data_points_path=config.esrs_data_points_path,
        data_quality_rules_path=config.data_quality_rules_path,
        quality_threshold=config.quality_threshold
    )

    # Process
    result = agent.process(input_file=esg_file, company_profile=company_context)

    return result


def csrd_assess_materiality(
    esg_data: Optional[Union[str, Path, pd.DataFrame, Dict]] = None,
    company_context: Optional[Union[str, Path, Dict]] = None,
    config: Optional[CSRDConfig] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    impact_threshold: Optional[float] = None,
    financial_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Conduct double materiality assessment using AI.

    WARNING: AI-generated assessments require human review.

    Args:
        esg_data: ESG data (optional, provides context)
        company_context: Company context OR path to profile
        config: CSRDConfig object (optional)
        llm_provider: LLM provider ("openai" or "anthropic")
        llm_model: LLM model name
        llm_api_key: LLM API key
        impact_threshold: Impact materiality threshold (0-10)
        financial_threshold: Financial materiality threshold (0-10)

    Returns:
        Materiality assessment result with material topics, scores, review flags

    Example:
        result = csrd_assess_materiality(
            company_context="company.json",
            llm_provider="openai",
            llm_model="gpt-4o"
        )
        print(f"Material topics: {result['summary_statistics']['material_topics_count']}")
        print(f"WARNING: Human review required!")
    """
    # Setup config
    if config is None:
        config = CSRDConfig(company_name="", company_lei="", reporting_year=2024, sector="")

    if llm_provider:
        config.llm_provider = llm_provider
    if llm_model:
        config.llm_model = llm_model
    if llm_api_key:
        config.llm_api_key = llm_api_key
    if impact_threshold is not None:
        config.impact_materiality_threshold = impact_threshold
    if financial_threshold is not None:
        config.financial_materiality_threshold = financial_threshold

    # Load company context
    if isinstance(company_context, (str, Path)):
        _, _, company_dict = _load_input_data(company_context, "company_profile")
        company_context = _build_company_context(company_dict, config)
    elif isinstance(company_context, dict):
        company_context = _build_company_context(company_context, config)
    else:
        raise ValueError("company_context must be a file path or dictionary")

    # Load ESG data if provided
    esg_data_dict = None
    if esg_data:
        esg_file, esg_df, esg_dict = _load_input_data(esg_data, "esg_data")
        if esg_dict:
            esg_data_dict = esg_dict
        elif esg_df is not None:
            esg_data_dict = {"data_points": esg_df.to_dict('records')}

    # Initialize agent
    llm_config = LLMConfig(
        provider=config.llm_provider,
        model=config.llm_model,
        api_key=config.llm_api_key
    )

    agent = MaterialityAgent(
        esrs_data_points_path=config.esrs_data_points_path,
        llm_config=llm_config,
        impact_threshold=config.impact_materiality_threshold,
        financial_threshold=config.financial_materiality_threshold
    )

    # Process
    result = agent.process(
        company_context=company_context,
        esg_data=esg_data_dict
    )

    return result


def csrd_calculate_metrics(
    validated_data: Union[Dict, str, Path],
    materiality: Optional[Dict] = None,
    config: Optional[CSRDConfig] = None,
    esrs_formulas_path: Optional[str] = None,
    emission_factors_path: Optional[str] = None,
    metrics_to_calculate: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate ESRS metrics with zero hallucination guarantee.

    All calculations are deterministic (NO AI/LLM).

    Args:
        validated_data: Validated data from intake agent OR path to file
        materiality: Materiality assessment result (optional, determines which metrics to calculate)
        config: CSRDConfig object (optional)
        esrs_formulas_path: Path to ESRS formulas
        emission_factors_path: Path to emission factors
        metrics_to_calculate: Explicit list of metric codes to calculate

    Returns:
        Calculation result with calculated metrics, provenance, errors

    Example:
        # After validation
        validated = csrd_validate_data(esg_data="data.csv")

        # Calculate metrics
        result = csrd_calculate_metrics(
            validated_data=validated,
            metrics_to_calculate=["E1-1", "E1-2", "E1-3", "E1-4"]
        )
        print(f"Calculated: {result['metadata']['metrics_calculated']} metrics")
        print(f"Zero hallucination: {result['metadata']['zero_hallucination_guarantee']}")
    """
    # Setup config
    if config is None:
        config = CSRDConfig(company_name="", company_lei="", reporting_year=2024, sector="")

    if esrs_formulas_path:
        config.esrs_formulas_path = esrs_formulas_path
    if emission_factors_path:
        config.emission_factors_path = emission_factors_path

    # Load validated data if path
    if isinstance(validated_data, (str, Path)):
        with open(validated_data, 'r') as f:
            validated_data = json.load(f)

    # Determine metrics to calculate
    if metrics_to_calculate is None:
        if materiality:
            # Extract from materiality
            material_standards = materiality.get("summary_statistics", {}).get("esrs_standards_triggered", [])
        else:
            # Default core metrics
            material_standards = ["E1", "S1", "G1"]

        metrics_to_calculate = []
        for standard in material_standards:
            if standard == "E1":
                metrics_to_calculate.extend(["E1-1", "E1-2", "E1-3", "E1-4", "E1-5", "E1-6"])
            elif standard == "S1":
                metrics_to_calculate.extend(["S1-1", "S1-2", "S1-3", "S1-4"])
            elif standard == "G1":
                metrics_to_calculate.extend(["G1-1", "G1-2"])

    # Prepare input data
    calculation_input = {}
    for dp in validated_data.get("data_points", []):
        metric_code = dp.get("metric_code")
        if metric_code:
            calculation_input[metric_code] = dp.get("value")

    # Initialize agent
    agent = CalculatorAgent(
        esrs_formulas_path=config.esrs_formulas_path,
        emission_factors_path=config.emission_factors_path
    )

    # Calculate
    result = agent.calculate_batch(
        metric_codes=metrics_to_calculate,
        input_data=calculation_input
    )

    return result


def csrd_aggregate_frameworks(
    metrics: Dict[str, Any],
    tcfd: Optional[Dict] = None,
    gri: Optional[Dict] = None,
    sasb: Optional[Dict] = None,
    config: Optional[CSRDConfig] = None
) -> Dict[str, Any]:
    """
    Aggregate data across multiple reporting frameworks.

    Maps TCFD, GRI, SASB data to ESRS standards.

    Args:
        metrics: ESRS metrics (from calculator)
        tcfd: TCFD data (optional)
        gri: GRI data (optional)
        sasb: SASB data (optional)
        config: CSRDConfig object (optional)

    Returns:
        Aggregated framework data with cross-mappings

    Example:
        result = csrd_aggregate_frameworks(
            metrics=calculated_metrics,
            tcfd=tcfd_data,
            gri=gri_data
        )
    """
    # Placeholder - aggregator implementation
    return {
        "aggregated_metrics": [],
        "framework_coverage": {},
        "gap_analysis": {}
    }


def csrd_generate_report(
    aggregated_data: Dict[str, Any],
    materiality: Dict[str, Any],
    company_profile: Union[str, Path, Dict],
    config: Optional[CSRDConfig] = None,
    output_format: str = "xbrl"
) -> Dict[str, Any]:
    """
    Generate ESEF-compliant CSRD report with XBRL tagging.

    Args:
        aggregated_data: Aggregated metrics and data
        materiality: Materiality assessment
        company_profile: Company profile
        config: CSRDConfig object (optional)
        output_format: Output format ("xbrl", "pdf", "both")

    Returns:
        Report package with XBRL, ESEF, PDF outputs

    Example:
        report = csrd_generate_report(
            aggregated_data=aggregated,
            materiality=materiality,
            company_profile="company.json",
            output_format="xbrl"
        )
    """
    # Placeholder - reporting implementation
    return {
        "xbrl_report": {},
        "esef_package": {},
        "pdf_report": None
    }


def csrd_audit_compliance(
    report_package: Dict[str, Any],
    audit_trail: Optional[Dict[str, Any]] = None,
    config: Optional[CSRDConfig] = None
) -> Dict[str, Any]:
    """
    Validate CSRD report against ESRS compliance rules.

    All checks are deterministic (NO AI/LLM).

    Args:
        report_package: Complete report package to audit
        audit_trail: Audit trail data (optional)
        config: CSRDConfig object (optional)

    Returns:
        Compliance validation result with pass/fail status, rule results

    Example:
        result = csrd_audit_compliance(report_package=complete_report)
        print(f"Compliance: {result['compliance_report']['compliance_status']}")
        print(f"Rules passed: {result['compliance_report']['rules_passed']}")
    """
    # Setup config
    if config is None:
        config = CSRDConfig(company_name="", company_lei="", reporting_year=2024, sector="")

    # Initialize agent
    agent = AuditAgent(compliance_rules_path=config.compliance_rules_path)

    # Audit
    result = agent.audit(report_data=report_package)

    return result


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Example usage of the SDK."""

    print("CSRD Reporting Platform - Python SDK")
    print("=" * 80)
    print()

    # Example 1: Basic usage
    print("Example 1: Basic usage")
    print("-" * 80)

    # Note: This is example code - files may not exist
    try:
        # Create config
        config = CSRDConfig(
            company_name="Acme Manufacturing GmbH",
            company_lei="529900ABCDEFGHIJKLMN",
            reporting_year=2024,
            sector="Manufacturing",
            country="DE",
            employee_count=5000,
            revenue=500000000.0
        )

        # Generate report (if files exist)
        # report = csrd_build_report(
        #     esg_data="../examples/demo_esg_data.csv",
        #     company_profile="../examples/demo_company.json",
        #     config=config,
        #     output_dir="../output/demo_report"
        # )

        # print(report.summary())

        print("Config created successfully")
        print(f"Company: {config.company_name}")
        print(f"Reporting Year: {config.reporting_year}")

    except Exception as e:
        print(f"Note: Demo files not found (this is OK when running from sdk/ directory)")
        print(f"Error: {e}")
        print()

    # Example 2: Individual agent usage
    print("\nExample 2: Individual agent usage (validation only)")
    print("-" * 80)

    # try:
    #     result = csrd_validate_data(
    #         esg_data="../examples/demo_esg_data.csv",
    #         config=config
    #     )
    #
    #     print(f"Total records: {result['metadata']['total_records']}")
    #     print(f"Valid: {result['metadata']['valid_records']}")
    #     print(f"Invalid: {result['metadata']['invalid_records']}")
    #     print(f"Data quality: {result['metadata']['data_quality_score']:.1f}/100")
    #
    # except FileNotFoundError:
    #     print("Note: Demo files not found")

    print("Example code complete!")
    print()
    print("=" * 80)
    print("SDK ready for use!")
    print()
    print("Quick start:")
    print("  from sdk.csrd_sdk import csrd_build_report, CSRDConfig")
    print()
    print("  config = CSRDConfig(")
    print("      company_name='Your Company',")
    print("      company_lei='...',")
    print("      reporting_year=2024,")
    print("      sector='...'")
    print("  )")
    print()
    print("  report = csrd_build_report(")
    print("      esg_data='your_data.csv',")
    print("      company_profile='company.json',")
    print("      config=config")
    print("  )")
    print()
    print("  print(report.summary())")
