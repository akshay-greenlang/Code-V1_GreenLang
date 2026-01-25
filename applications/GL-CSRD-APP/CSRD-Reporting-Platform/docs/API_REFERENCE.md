# CSRD Platform - API Reference

**Complete API Documentation for the CSRD/ESRS Digital Reporting Platform**

Version 1.0.0 | Last Updated: 2025-10-18

---

## Table of Contents

1. [SDK Functions](#sdk-functions)
2. [Configuration Classes](#configuration-classes)
3. [Data Model Classes](#data-model-classes)
4. [Agent Classes](#agent-classes)
5. [Pipeline Classes](#pipeline-classes)
6. [Provenance Models](#provenance-models)
7. [Utility Functions](#utility-functions)
8. [Error Handling](#error-handling)

---

## SDK Functions

The SDK provides high-level functions for common CSRD reporting tasks.

### csrd_build_report()

Main function to generate a complete CSRD report.

**Import:**
```python
from sdk.csrd_sdk import csrd_build_report
```

**Signature:**
```python
def csrd_build_report(
    esg_data: Union[str, Path, pd.DataFrame],
    company_profile: Union[str, Path, Dict],
    config: Optional[CSRDConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
    skip_materiality: bool = False,
    skip_audit: bool = False,
    verbose: bool = False,

    # Path overrides
    esrs_data_points_path: Optional[str] = None,
    data_quality_rules_path: Optional[str] = None,
    esrs_formulas_path: Optional[str] = None,
    emission_factors_path: Optional[str] = None,
    compliance_rules_path: Optional[str] = None,

    # Threshold overrides
    quality_threshold: Optional[float] = None,
    impact_materiality_threshold: Optional[float] = None,
    financial_materiality_threshold: Optional[float] = None,

    # LLM config overrides
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None
) -> CSRDReport
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `esg_data` | str \| Path \| DataFrame | Yes | - | ESG data as file path or DataFrame |
| `company_profile` | str \| Path \| Dict | Yes | - | Company profile as file path or dict |
| `config` | CSRDConfig | No | None | Configuration object |
| `output_dir` | str \| Path | No | None | Directory to save all outputs |
| `skip_materiality` | bool | No | False | Skip materiality assessment |
| `skip_audit` | bool | No | False | Skip compliance audit |
| `verbose` | bool | No | False | Enable verbose logging |
| `quality_threshold` | float | No | 0.80 | Minimum data quality score (0-1) |
| `impact_materiality_threshold` | float | No | 5.0 | Impact materiality threshold (0-10) |
| `financial_materiality_threshold` | float | No | 5.0 | Financial materiality threshold (0-10) |
| `llm_provider` | str | No | "openai" | LLM provider: "openai" or "anthropic" |
| `llm_model` | str | No | "gpt-4o" | LLM model name |
| `llm_api_key` | str | No | None | LLM API key |

**Returns:**
- `CSRDReport`: Complete report object with all results

**Raises:**
- `FileNotFoundError`: If input files don't exist
- `ValueError`: If data is invalid or incompatible
- `RuntimeError`: If pipeline execution fails

**Example:**
```python
from sdk.csrd_sdk import csrd_build_report, CSRDConfig

# Basic usage
report = csrd_build_report(
    esg_data="data/esg_data.csv",
    company_profile="data/company.json"
)

# With configuration
config = CSRDConfig(
    company_name="Acme Corp",
    company_lei="549300ABC123DEF456GH",
    reporting_year=2024,
    sector="Manufacturing",
    llm_api_key="sk-..."
)

report = csrd_build_report(
    esg_data="data/esg_data.csv",
    company_profile="data/company.json",
    config=config,
    output_dir="output/csrd_2024",
    verbose=True
)

# Access results
print(f"Compliance: {report.compliance_status.compliance_status}")
print(f"Material topics: {report.materiality.material_topics_count}")
print(f"GHG emissions: {report.metrics.total_ghg_emissions_tco2e:.2f} tCO2e")

# Save outputs
report.save_json("report.json")
report.save_summary("summary.md")
```

**Notes:**
- Executes all 6 agents in sequence
- Returns structured Python object with complete results
- Automatically handles data format conversion
- Saves intermediate outputs if `output_dir` is provided
- Zero hallucination guarantee for all calculations

---

### csrd_validate_data()

Validate ESG data without generating a full report.

**Import:**
```python
from sdk.csrd_sdk import csrd_validate_data
```

**Signature:**
```python
def csrd_validate_data(
    esg_data: Union[str, Path, pd.DataFrame],
    company_profile: Optional[Union[str, Path, Dict]] = None,
    config: Optional[CSRDConfig] = None,
    esrs_data_points_path: Optional[str] = None,
    data_quality_rules_path: Optional[str] = None,
    quality_threshold: Optional[float] = None
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `esg_data` | str \| Path \| DataFrame | Yes | ESG data to validate |
| `company_profile` | str \| Path \| Dict | No | Company profile (optional) |
| `config` | CSRDConfig | No | Configuration object |
| `esrs_data_points_path` | str | No | Path to ESRS catalog |
| `data_quality_rules_path` | str | No | Path to quality rules |
| `quality_threshold` | float | No | Quality threshold (0-1) |

**Returns:**
```python
{
    "metadata": {
        "total_records": int,
        "valid_records": int,
        "invalid_records": int,
        "warnings": int,
        "data_quality_score": float,  # 0-100
        "quality_threshold_met": bool,
        "processing_time_seconds": float
    },
    "data_points": List[Dict],
    "validation_issues": List[Dict],
    "quality_metrics": Dict[str, float]
}
```

**Example:**
```python
from sdk.csrd_sdk import csrd_validate_data

result = csrd_validate_data(
    esg_data="data.csv",
    quality_threshold=0.85
)

print(f"Total: {result['metadata']['total_records']}")
print(f"Valid: {result['metadata']['valid_records']}")
print(f"Quality: {result['metadata']['data_quality_score']:.1f}/100")

# Check validation issues
for issue in result['validation_issues']:
    print(f"{issue['severity']}: {issue['message']}")
```

---

### csrd_assess_materiality()

Conduct double materiality assessment using AI.

**Import:**
```python
from sdk.csrd_sdk import csrd_assess_materiality
```

**Signature:**
```python
def csrd_assess_materiality(
    esg_data: Optional[Union[str, Path, pd.DataFrame, Dict]] = None,
    company_context: Optional[Union[str, Path, Dict]] = None,
    config: Optional[CSRDConfig] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    impact_threshold: Optional[float] = None,
    financial_threshold: Optional[float] = None
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `esg_data` | str \| Path \| DataFrame \| Dict | No | ESG data (optional context) |
| `company_context` | str \| Path \| Dict | Yes | Company context/profile |
| `config` | CSRDConfig | No | Configuration object |
| `llm_provider` | str | No | LLM provider |
| `llm_model` | str | No | LLM model name |
| `llm_api_key` | str | No | LLM API key |
| `impact_threshold` | float | No | Impact threshold (0-10) |
| `financial_threshold` | float | No | Financial threshold (0-10) |

**Returns:**
```python
{
    "material_topics": List[Dict],
    "summary_statistics": {
        "total_topics_assessed": int,
        "material_topics_count": int,
        "material_from_impact": int,
        "material_from_financial": int,
        "double_material_count": int,
        "esrs_standards_triggered": List[str]
    },
    "ai_metadata": {
        "llm_provider": str,
        "llm_model": str,
        "average_confidence": float,
        "processing_time_minutes": float
    },
    "review_flags": List[Dict]
}
```

**Example:**
```python
from sdk.csrd_sdk import csrd_assess_materiality

result = csrd_assess_materiality(
    company_context="company.json",
    llm_provider="openai",
    llm_model="gpt-4o",
    llm_api_key="sk-...",
    impact_threshold=6.0,
    financial_threshold=6.0
)

print(f"Material topics: {result['summary_statistics']['material_topics_count']}")
print(f"Material ESRS: {result['summary_statistics']['esrs_standards_triggered']}")
print(f"AI confidence: {result['ai_metadata']['average_confidence']:.0%}")

# WARNING: Requires human review!
for flag in result['review_flags']:
    print(f"Review: {flag['topic']} - {flag['reason']}")
```

**Warning:**
- AI-generated assessments REQUIRE human review
- Not suitable for final reporting without validation
- Review all flagged items carefully

---

### csrd_calculate_metrics()

Calculate ESRS metrics with zero hallucination guarantee.

**Import:**
```python
from sdk.csrd_sdk import csrd_calculate_metrics
```

**Signature:**
```python
def csrd_calculate_metrics(
    validated_data: Union[Dict, str, Path],
    materiality: Optional[Dict] = None,
    config: Optional[CSRDConfig] = None,
    esrs_formulas_path: Optional[str] = None,
    emission_factors_path: Optional[str] = None,
    metrics_to_calculate: Optional[List[str]] = None
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `validated_data` | Dict \| str \| Path | Yes | Validated data from intake |
| `materiality` | Dict | No | Materiality assessment |
| `config` | CSRDConfig | No | Configuration |
| `esrs_formulas_path` | str | No | Path to formulas |
| `emission_factors_path` | str | No | Path to factors |
| `metrics_to_calculate` | List[str] | No | Specific metrics to calculate |

**Returns:**
```python
{
    "calculated_metrics": List[Dict],
    "metadata": {
        "total_metrics_requested": int,
        "metrics_calculated": int,
        "calculation_errors": int,
        "processing_time_seconds": float,
        "ms_per_metric": float,
        "zero_hallucination_guarantee": bool  # Always True
    },
    "calculation_provenance": List[Dict],
    "errors": List[Dict]
}
```

**Example:**
```python
from sdk.csrd_sdk import csrd_calculate_metrics

# After validation
validated = csrd_validate_data(esg_data="data.csv")

# Calculate specific metrics
result = csrd_calculate_metrics(
    validated_data=validated,
    metrics_to_calculate=["E1-1", "E1-2", "E1-3", "E1-4", "S1-1", "G1-1"]
)

print(f"Calculated: {result['metadata']['metrics_calculated']} metrics")
print(f"Time: {result['metadata']['ms_per_metric']:.2f} ms per metric")
print(f"Zero hallucination: {result['metadata']['zero_hallucination_guarantee']}")

# Access calculated values
for metric in result['calculated_metrics']:
    print(f"{metric['metric_code']}: {metric['value']} {metric['unit']}")
    print(f"  Formula: {metric['formula_used']}")
    print(f"  Provenance: {metric['calculation_provenance']['method']}")
```

**Notes:**
- 100% deterministic (no LLM involvement)
- Same inputs always produce same outputs
- Complete provenance for every calculation
- Average processing: <5ms per metric

---

## Configuration Classes

### CSRDConfig

Main configuration class for the CSRD pipeline.

**Import:**
```python
from sdk.csrd_sdk import CSRDConfig
```

**Constructor:**
```python
CSRDConfig(
    # Company info (required)
    company_name: str,
    company_lei: str,
    reporting_year: int,
    sector: str,

    # Company details (optional)
    country: str = "DE",
    employee_count: Optional[int] = None,
    revenue: Optional[float] = None,
    total_assets: Optional[float] = None,

    # Paths (optional)
    esrs_data_points_path: str = "data/esrs_data_points.json",
    data_quality_rules_path: str = "rules/data_quality_rules.yaml",
    esrs_formulas_path: str = "rules/esrs_formulas.yaml",
    emission_factors_path: str = "data/emission_factors.json",
    compliance_rules_path: str = "rules/compliance_rules.yaml",
    framework_mappings_path: str = "data/framework_mappings.json",

    # LLM config
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o",
    llm_api_key: Optional[str] = None,

    # Thresholds
    quality_threshold: float = 0.80,
    impact_materiality_threshold: float = 5.0,
    financial_materiality_threshold: float = 5.0
)
```

**Class Methods:**

```python
# Load from YAML file
config = CSRDConfig.from_yaml("csrd_config.yaml")

# Load from dictionary
config = CSRDConfig.from_dict(config_dict)

# Load from environment variables
config = CSRDConfig.from_env()

# Convert to dictionary
config_dict = config.to_dict()
```

**Example:**
```python
from sdk.csrd_sdk import CSRDConfig

# Create configuration
config = CSRDConfig(
    company_name="Acme Manufacturing EU B.V.",
    company_lei="549300ABC123DEF456GH",
    reporting_year=2024,
    sector="Manufacturing",
    country="NL",
    employee_count=1250,
    revenue=450000000.0,

    # LLM settings
    llm_provider="openai",
    llm_model="gpt-4o",
    llm_api_key="sk-...",

    # Quality settings
    quality_threshold=0.85,
    impact_materiality_threshold=6.0,
    financial_materiality_threshold=6.0
)

# Load from file
config = CSRDConfig.from_yaml("csrd_config.yaml")

# Save to dict
config_dict = config.to_dict()
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `company_name` | str | Legal company name |
| `company_lei` | str | Legal Entity Identifier (20 chars) |
| `reporting_year` | int | Reporting fiscal year |
| `sector` | str | Industry sector |
| `country` | str | ISO 3166-1 alpha-2 country code |
| `employee_count` | int | Total employee count |
| `revenue` | float | Annual revenue (in EUR) |
| `total_assets` | float | Total assets (in EUR) |
| `llm_provider` | str | LLM provider: "openai" or "anthropic" |
| `llm_model` | str | LLM model name |
| `llm_api_key` | str | LLM API key |
| `quality_threshold` | float | Min data quality score (0-1) |
| `impact_materiality_threshold` | float | Impact threshold (0-10) |
| `financial_materiality_threshold` | float | Financial threshold (0-10) |

---

## Data Model Classes

### CSRDReport

Complete CSRD report object returned by `csrd_build_report()`.

**Import:**
```python
from sdk.csrd_sdk import CSRDReport
```

**Attributes:**

```python
# Core components
report.company_info: Dict[str, Any]
report.reporting_period: Dict[str, Any]
report.data_validation: Dict[str, Any]
report.materiality: MaterialityAssessment
report.metrics: ESRSMetrics
report.compliance_status: ComplianceStatus

# Optional components
report.aggregated_frameworks: Optional[Dict[str, Any]]
report.xbrl_report: Optional[Dict[str, Any]]
report.audit_package: Optional[Dict[str, Any]]

# Metadata
report.report_id: str
report.generated_at: str
report.processing_time_total_minutes: float
report.warnings: List[str]
report.info_messages: List[str]
report.raw_report: Dict[str, Any]
```

**Properties:**

```python
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
```

**Methods:**

```python
def to_dict(self) -> Dict[str, Any]:
    """Get raw report dictionary."""

def to_json(self, indent: int = 2) -> str:
    """Convert to JSON string."""

def save_json(self, path: str):
    """Save complete report to JSON file."""

def save_summary(self, path: str):
    """Save human-readable summary to Markdown file."""

def to_dataframe(self) -> pd.DataFrame:
    """Convert validated ESG data to pandas DataFrame."""

def summary(self) -> str:
    """Get a quick text summary of the report."""
```

**Example:**
```python
report = csrd_build_report(...)

# Access properties
print(f"Compliant: {report.is_compliant}")
print(f"Audit ready: {report.is_audit_ready}")
print(f"Material standards: {', '.join(report.material_standards)}")

# Access components
print(f"Company: {report.company_info['legal_name']}")
print(f"Material topics: {report.materiality.material_topics_count}")
print(f"GHG emissions: {report.metrics.total_ghg_emissions_tco2e} tCO2e")
print(f"Compliance: {report.compliance_status.compliance_status}")

# Save outputs
report.save_json("complete_report.json")
report.save_summary("summary.md")

# Convert to DataFrame
df = report.to_dataframe()
df.to_csv("esg_data.csv", index=False)

# Get summary
print(report.summary())
```

---

### MaterialityAssessment

Double materiality assessment results.

**Attributes:**

```python
total_topics_assessed: int
material_topics_count: int
material_from_impact: int
material_from_financial: int
double_material_count: int

# Material ESRS standards triggered
esrs_standards_triggered: List[str]

# Material topics details
material_topics: List[Dict[str, Any]]

# AI metadata
ai_powered: bool = True
requires_human_review: bool = True
llm_provider: Optional[str]
llm_model: Optional[str]
average_confidence: Optional[float]
review_flags_count: int

# Processing
processing_time_minutes: Optional[float]
```

**Example:**
```python
mat = report.materiality

print(f"Total topics: {mat.total_topics_assessed}")
print(f"Material: {mat.material_topics_count}")
print(f"Double material: {mat.double_material_count}")
print(f"Standards: {', '.join(mat.esrs_standards_triggered)}")
print(f"AI confidence: {mat.average_confidence:.0%}")
print(f"Review flags: {mat.review_flags_count}")

# Access material topics
for topic in mat.material_topics:
    print(f"{topic['topic']}: {topic['esrs_standard']}")
    print(f"  Impact: {topic['impact_materiality']['score']}/10")
    print(f"  Financial: {topic['financial_materiality']['score']}/10")
```

---

### ESRSMetrics

Calculated ESRS metrics with complete provenance.

**Attributes:**

```python
total_metrics_calculated: int
metrics_by_standard: Dict[str, int]

# Climate metrics (E1)
scope_1_emissions_tco2e: Optional[float]
scope_2_emissions_tco2e: Optional[float]
scope_3_emissions_tco2e: Optional[float]
total_ghg_emissions_tco2e: Optional[float]
ghg_intensity: Optional[float]
total_energy_consumption_mwh: Optional[float]
renewable_energy_percentage: Optional[float]

# Social metrics (S1)
total_workforce: Optional[int]
employee_turnover_rate: Optional[float]
gender_pay_gap: Optional[float]
work_related_accidents: Optional[int]

# Governance metrics (G1)
board_gender_diversity: Optional[float]
ethics_violations: Optional[int]

# Metadata
calculation_method: str = "deterministic"
zero_hallucination_guarantee: bool = True
processing_time_seconds: Optional[float]

# Raw data access
raw_metrics: Dict[str, Any]
```

**Example:**
```python
metrics = report.metrics

print(f"Total metrics: {metrics.total_metrics_calculated}")
print(f"Processing time: {metrics.processing_time_seconds:.2f}s")
print(f"Zero hallucination: {metrics.zero_hallucination_guarantee}")

# Climate metrics
if metrics.total_ghg_emissions_tco2e:
    print(f"\nGHG Emissions:")
    print(f"  Scope 1: {metrics.scope_1_emissions_tco2e:,.2f} tCO2e")
    print(f"  Scope 2: {metrics.scope_2_emissions_tco2e:,.2f} tCO2e")
    print(f"  Scope 3: {metrics.scope_3_emissions_tco2e:,.2f} tCO2e")
    print(f"  Total: {metrics.total_ghg_emissions_tco2e:,.2f} tCO2e")

# Social metrics
if metrics.total_workforce:
    print(f"\nWorkforce:")
    print(f"  Total: {metrics.total_workforce:,} employees")
    print(f"  Turnover: {metrics.employee_turnover_rate:.1f}%")
    print(f"  Gender pay gap: {metrics.gender_pay_gap:.1f}%")

# Access raw metrics
for metric in metrics.raw_metrics.get('calculated_metrics', []):
    print(f"{metric['metric_code']}: {metric['value']} {metric['unit']}")
```

---

### ComplianceStatus

ESRS compliance validation results.

**Attributes:**

```python
compliance_status: str  # "PASS", "FAIL", "WARNING"
total_rules_checked: int
rules_passed: int
rules_failed: int
rules_warning: int

# Failure breakdown
critical_failures: int
major_failures: int
minor_failures: int

# Details
failed_rules: List[Dict[str, Any]]
warning_rules: List[Dict[str, Any]]

# Audit readiness
audit_ready: bool
audit_package_generated: bool

# Metadata
validation_timestamp: Optional[str]
validation_duration_seconds: Optional[float]
```

**Example:**
```python
compliance = report.compliance_status

print(f"Status: {compliance.compliance_status}")
print(f"Audit ready: {compliance.audit_ready}")
print(f"\nRule Execution:")
print(f"  Total: {compliance.total_rules_checked}")
print(f"  Passed: {compliance.rules_passed}")
print(f"  Failed: {compliance.rules_failed}")
print(f"  Warnings: {compliance.rules_warning}")

print(f"\nFailure Breakdown:")
print(f"  Critical: {compliance.critical_failures}")
print(f"  Major: {compliance.major_failures}")
print(f"  Minor: {compliance.minor_failures}")

# Review failed rules
if compliance.failed_rules:
    print("\nFailed Rules:")
    for rule in compliance.failed_rules:
        print(f"  [{rule['severity']}] {rule['rule_id']}: {rule['message']}")
```

---

## Agent Classes

All agent classes are located in the `agents/` directory.

### IntakeAgent

Data validation and enrichment agent.

**Import:**
```python
from agents.intake_agent import IntakeAgent
```

**Constructor:**
```python
IntakeAgent(
    esrs_data_points_path: str = "data/esrs_data_points.json",
    data_quality_rules_path: str = "rules/data_quality_rules.yaml",
    quality_threshold: float = 0.80
)
```

**Methods:**

```python
def process(
    self,
    input_file: str,
    company_profile: Optional[Dict[str, Any]] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate and enrich ESG data.

    Args:
        input_file: Path to ESG data file
        company_profile: Company profile dict
        output_file: Optional output path

    Returns:
        Validation result with metadata, data points, issues
    """
```

**Example:**
```python
from agents.intake_agent import IntakeAgent

agent = IntakeAgent(
    quality_threshold=0.85
)

result = agent.process(
    input_file="data/esg_data.csv",
    company_profile=company_dict,
    output_file="output/validated_data.json"
)

print(f"Valid: {result['metadata']['valid_records']}")
print(f"Invalid: {result['metadata']['invalid_records']}")
print(f"Quality: {result['metadata']['data_quality_score']:.1f}/100")
```

---

### MaterialityAgent

AI-powered double materiality assessment agent.

**Import:**
```python
from agents.materiality_agent import MaterialityAgent, LLMConfig
```

**Constructor:**
```python
LLMConfig(
    provider: str = "openai",
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 4000
)

MaterialityAgent(
    esrs_data_points_path: str = "data/esrs_data_points.json",
    llm_config: LLMConfig = None,
    impact_threshold: float = 5.0,
    financial_threshold: float = 5.0
)
```

**Methods:**

```python
def process(
    self,
    company_context: Dict[str, Any],
    esg_data: Optional[Dict[str, Any]] = None,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Conduct double materiality assessment.

    Args:
        company_context: Company profile and context
        esg_data: ESG data (optional)
        output_file: Optional output path

    Returns:
        Materiality assessment result
    """
```

**Example:**
```python
from agents.materiality_agent import MaterialityAgent, LLMConfig

llm_config = LLMConfig(
    provider="openai",
    model="gpt-4o",
    api_key="sk-...",
    temperature=0.3
)

agent = MaterialityAgent(
    llm_config=llm_config,
    impact_threshold=6.0,
    financial_threshold=6.0
)

result = agent.process(
    company_context=company_dict,
    esg_data=validated_data,
    output_file="output/materiality.json"
)

print(f"Material topics: {result['summary_statistics']['material_topics_count']}")
```

**Warning:** Requires human review!

---

### CalculatorAgent

Zero-hallucination ESRS metrics calculation agent.

**Import:**
```python
from agents.calculator_agent import CalculatorAgent
```

**Constructor:**
```python
CalculatorAgent(
    esrs_formulas_path: str = "rules/esrs_formulas.yaml",
    emission_factors_path: str = "data/emission_factors.json"
)
```

**Methods:**

```python
def calculate_batch(
    self,
    metric_codes: List[str],
    input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate multiple metrics in batch.

    Args:
        metric_codes: List of metric codes to calculate
        input_data: Input data dictionary

    Returns:
        Calculation results with provenance
    """

def write_output(
    self,
    result: Dict[str, Any],
    output_file: str
):
    """Write calculation results to file."""
```

**Example:**
```python
from agents.calculator_agent import CalculatorAgent

agent = CalculatorAgent()

result = agent.calculate_batch(
    metric_codes=["E1-1", "E1-2", "E1-3", "E1-4"],
    input_data=input_dict
)

agent.write_output(result, "output/calculated_metrics.json")

print(f"Calculated: {result['metadata']['metrics_calculated']}")
print(f"Zero hallucination: {result['metadata']['zero_hallucination_guarantee']}")
```

---

### AuditAgent

Compliance validation and audit trail generation agent.

**Import:**
```python
from agents.audit_agent import AuditAgent
```

**Constructor:**
```python
AuditAgent(
    compliance_rules_path: str = "rules/compliance_rules.yaml"
)
```

**Methods:**

```python
def audit(
    self,
    report_data: Dict[str, Any],
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate report against ESRS compliance rules.

    Args:
        report_data: Complete report data
        output_file: Optional output path

    Returns:
        Compliance validation result
    """
```

**Example:**
```python
from agents.audit_agent import AuditAgent

agent = AuditAgent()

result = agent.audit(
    report_data=complete_report,
    output_file="output/compliance.json"
)

print(f"Status: {result['compliance_report']['compliance_status']}")
print(f"Passed: {result['compliance_report']['rules_passed']}")
print(f"Failed: {result['compliance_report']['rules_failed']}")
```

---

## Pipeline Classes

### CSRDPipeline

Complete pipeline orchestrator (internal use).

**Import:**
```python
from csrd_pipeline import CSRDPipeline
```

**Constructor:**
```python
CSRDPipeline(config_path: str)
```

**Methods:**
```python
def run(
    self,
    esg_data_file: str,
    company_profile_file: str,
    output_dir: str,
    **kwargs
) -> PipelineResult:
    """Execute complete pipeline."""
```

**Note:** For most use cases, use the SDK functions instead.

---

## Error Handling

### Exception Classes

```python
class CSRDException(Exception):
    """Base exception for CSRD platform."""

class DataValidationError(CSRDException):
    """Data validation failed."""

class CalculationError(CSRDException):
    """Metric calculation failed."""

class ComplianceError(CSRDException):
    """Compliance validation failed."""

class ConfigurationError(CSRDException):
    """Invalid configuration."""
```

### Error Handling Example

```python
from sdk.csrd_sdk import csrd_build_report, DataValidationError

try:
    report = csrd_build_report(
        esg_data="data.csv",
        company_profile="company.json"
    )
except FileNotFoundError as e:
    print(f"Input file not found: {e}")
except DataValidationError as e:
    print(f"Data validation failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Return Type Specifications

### Validation Result

```typescript
{
  metadata: {
    total_records: number,
    valid_records: number,
    invalid_records: number,
    warnings: number,
    data_quality_score: number,  // 0-100
    quality_threshold_met: boolean,
    processing_time_seconds: number
  },
  data_points: Array<{
    metric_code: string,
    metric_name: string,
    value: any,
    unit: string,
    period_start: string,
    period_end: string,
    data_quality: string,
    validation_status: string
  }>,
  validation_issues: Array<{
    severity: "error" | "warning" | "info",
    error_code: string,
    message: string,
    field: string,
    row: number
  }>,
  quality_metrics: {
    completeness: number,
    accuracy: number,
    consistency: number,
    timeliness: number
  }
}
```

### Calculation Result

```typescript
{
  calculated_metrics: Array<{
    metric_code: string,
    metric_name: string,
    value: number,
    unit: string,
    formula_used: string,
    calculation_provenance: {
      method: "deterministic",
      inputs: Array<string>,
      formula: string,
      timestamp: string
    }
  }>,
  metadata: {
    total_metrics_requested: number,
    metrics_calculated: number,
    calculation_errors: number,
    processing_time_seconds: number,
    ms_per_metric: number,
    zero_hallucination_guarantee: boolean
  },
  errors: Array<{
    metric_code: string,
    error_message: string,
    error_type: string
  }>
}
```

---

## Version Information

**Current Version:** 1.0.0

**Breaking Changes:** None (initial release)

**Deprecations:** None

**Changelog:**
- v1.0.0 (2025-10-18): Initial release

---

## Additional Resources

**Related Documentation:**
- [User Guide](USER_GUIDE.md) - Complete user manual
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Production deployment
- [Architecture Guide](ARCHITECTURE.md) - System architecture

**Support:**
- Email: csrd@greenlang.io
- GitHub: https://github.com/akshay-greenlang/Code-V1_GreenLang
- Issues: https://github.com/akshay-greenlang/Code-V1_GreenLang/issues

---

**Last Updated:** 2025-10-18
**Version:** 1.0.0
**License:** MIT
