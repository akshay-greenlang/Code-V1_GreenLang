"""
CBAM Importer Copilot - Python SDK

Simple, Pythonic API for CBAM reporting.

Design Philosophy:
- One main function: cbam_build_report()
- Accepts files OR DataFrames
- Returns structured Python objects
- Composable (can use individual agents)

Version: 1.0.0
Author: GreenLang CBAM Team
"""

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import pandas as pd

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from cbam_pipeline import CBAMPipeline
from agents import ShipmentIntakeAgent, EmissionsCalculatorAgent, ReportingPackagerAgent


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CBAMConfig:
    """
    CBAM configuration for repeated use.

    Stores importer information and default paths so you don't have to
    repeat them for every report.

    Example:
        config = CBAMConfig(
            importer_name="Acme Steel EU BV",
            importer_country="NL",
            importer_eori="NL123456789012",
            declarant_name="John Smith",
            declarant_position="Compliance Officer"
        )

        report = cbam_build_report(input_file="shipments.csv", config=config)
    """
    importer_name: str
    importer_country: str
    importer_eori: str
    declarant_name: str
    declarant_position: str

    # Optional paths
    cn_codes_path: str = "data/cn_codes.json"
    cbam_rules_path: str = "rules/cbam_rules.yaml"
    suppliers_path: str = "examples/demo_suppliers.yaml"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CBAMConfig':
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'CBAMConfig':
        """Load from YAML file."""
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        # Flatten structure if needed
        if "importer" in data:
            return cls(
                importer_name=data["importer"]["name"],
                importer_country=data["importer"]["country"],
                importer_eori=data["importer"]["eori"],
                declarant_name=data["declarant"]["name"],
                declarant_position=data["declarant"]["position"],
                cn_codes_path=data.get("paths", {}).get("cn_codes", "data/cn_codes.json"),
                cbam_rules_path=data.get("paths", {}).get("rules", "rules/cbam_rules.yaml"),
                suppliers_path=data.get("paths", {}).get("suppliers", "examples/demo_suppliers.yaml")
            )
        else:
            return cls(**data)


@dataclass
class CBAMReport:
    """
    CBAM report result with convenient access methods.

    Wraps the raw report dictionary with Pythonic attribute access
    and utility methods.

    Example:
        report = cbam_build_report(...)

        print(f"Report ID: {report.report_id}")
        print(f"Total emissions: {report.total_emissions_tco2:.2f} tCO2")
        print(f"Valid: {report.is_valid}")

        # Save to file
        report.save_json("output/report.json")
        report.save_summary("output/summary.md")

        # Export to DataFrame
        df = report.to_dataframe()
    """
    raw_report: Dict[str, Any]

    @property
    def report_id(self) -> str:
        """Report ID."""
        return self.raw_report.get("report_metadata", {}).get("report_id", "")

    @property
    def quarter(self) -> str:
        """Reporting quarter."""
        return self.raw_report.get("report_metadata", {}).get("quarter", "")

    @property
    def generated_at(self) -> str:
        """Generation timestamp."""
        return self.raw_report.get("report_metadata", {}).get("generated_at", "")

    @property
    def total_shipments(self) -> int:
        """Total number of shipments."""
        return self.raw_report.get("goods_summary", {}).get("total_shipments", 0)

    @property
    def total_mass_tonnes(self) -> float:
        """Total mass in tonnes."""
        return self.raw_report.get("goods_summary", {}).get("total_mass_tonnes", 0.0)

    @property
    def total_emissions_tco2(self) -> float:
        """Total embedded emissions in tCO2."""
        return self.raw_report.get("emissions_summary", {}).get("total_embedded_emissions_tco2", 0.0)

    @property
    def is_valid(self) -> bool:
        """Whether report passes all validation rules."""
        return self.raw_report.get("validation_results", {}).get("is_valid", False)

    @property
    def errors(self) -> List[Dict[str, Any]]:
        """Validation errors (if any)."""
        return self.raw_report.get("validation_results", {}).get("errors", [])

    @property
    def warnings(self) -> List[Dict[str, Any]]:
        """Validation warnings (if any)."""
        return self.raw_report.get("validation_results", {}).get("warnings", [])

    def to_dict(self) -> Dict[str, Any]:
        """Get raw report dictionary."""
        return self.raw_report

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.raw_report, indent=indent, default=str)

    def save_json(self, path: str):
        """Save report to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.raw_report, f, indent=2, default=str)

    def save_summary(self, path: str):
        """Save human-readable summary to Markdown file."""
        # Use ReportingPackagerAgent to generate summary
        agent = ReportingPackagerAgent()
        agent.write_summary(self.raw_report, path)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert detailed goods to pandas DataFrame.

        Returns:
            DataFrame with one row per shipment
        """
        detailed_goods = self.raw_report.get("detailed_goods", [])
        return pd.DataFrame(detailed_goods)

    def summary(self) -> str:
        """
        Get a quick text summary of the report.

        Returns:
            Multi-line string with key metrics
        """
        return f"""
CBAM Report Summary
===================
Report ID: {self.report_id}
Quarter: {self.quarter}
Generated: {self.generated_at}

Goods Summary:
- Total Shipments: {self.total_shipments}
- Total Mass: {self.total_mass_tonnes:.2f} tonnes
- Total Emissions: {self.total_emissions_tco2:.2f} tCO2

Validation: {'✓ PASS' if self.is_valid else '✗ FAIL'}
{f'Errors: {len(self.errors)}' if self.errors else ''}
{f'Warnings: {len(self.warnings)}' if self.warnings else ''}
        """.strip()

    def __repr__(self) -> str:
        return f"CBAMReport(report_id='{self.report_id}', total_emissions={self.total_emissions_tco2:.2f} tCO2, valid={self.is_valid})"


# ============================================================================
# MAIN SDK FUNCTION
# ============================================================================

def cbam_build_report(
    input_file: Optional[str] = None,
    input_dataframe: Optional[pd.DataFrame] = None,
    importer_name: Optional[str] = None,
    importer_country: Optional[str] = None,
    importer_eori: Optional[str] = None,
    declarant_name: Optional[str] = None,
    declarant_position: Optional[str] = None,
    config: Optional[CBAMConfig] = None,
    output_json: Optional[str] = None,
    output_summary: Optional[str] = None,
    cn_codes_path: str = "data/cn_codes.json",
    cbam_rules_path: str = "rules/cbam_rules.yaml",
    suppliers_path: str = "examples/demo_suppliers.yaml",
    intermediate_dir: Optional[str] = None
) -> CBAMReport:
    """
    Generate a CBAM Transitional Registry report.

    This is the main SDK function. Accepts either a file path or a pandas
    DataFrame as input, and returns a structured CBAMReport object.

    Args:
        input_file: Path to shipments file (CSV/JSON/Excel)
        input_dataframe: Pandas DataFrame with shipment data
        importer_name: EU importer legal name
        importer_country: EU country code (e.g., NL, DE, FR)
        importer_eori: EORI number
        declarant_name: Person making declaration
        declarant_position: Declarant position/title
        config: CBAMConfig object (alternative to individual params)
        output_json: Optional path to save JSON report
        output_summary: Optional path to save Markdown summary
        cn_codes_path: Path to CN codes JSON
        cbam_rules_path: Path to CBAM rules YAML
        suppliers_path: Path to suppliers YAML
        intermediate_dir: Optional directory for intermediate outputs

    Returns:
        CBAMReport object with results and convenience methods

    Examples:
        # From file
        report = cbam_build_report(
            input_file="shipments.csv",
            importer_name="Acme Steel EU BV",
            importer_country="NL",
            importer_eori="NL123456789012",
            declarant_name="John Smith",
            declarant_position="Compliance Officer"
        )

        # From DataFrame
        import pandas as pd
        df = pd.read_csv("shipments.csv")
        report = cbam_build_report(
            input_dataframe=df,
            importer_name="Acme Steel EU BV",
            ...
        )

        # With config
        config = CBAMConfig.from_yaml(".cbam.yaml")
        report = cbam_build_report(input_file="shipments.csv", config=config)

        # Access results
        print(f"Total emissions: {report.total_emissions_tco2:.2f} tCO2")
        print(f"Valid: {report.is_valid}")

        # Save outputs
        report.save_json("output/report.json")
        report.save_summary("output/summary.md")

        # Export to DataFrame
        df = report.to_dataframe()
    """

    # Use config if provided
    if config:
        importer_name = importer_name or config.importer_name
        importer_country = importer_country or config.importer_country
        importer_eori = importer_eori or config.importer_eori
        declarant_name = declarant_name or config.declarant_name
        declarant_position = declarant_position or config.declarant_position
        cn_codes_path = config.cn_codes_path
        cbam_rules_path = config.cbam_rules_path
        suppliers_path = config.suppliers_path

    # Validate required parameters
    required = {
        "importer_name": importer_name,
        "importer_country": importer_country,
        "importer_eori": importer_eori,
        "declarant_name": declarant_name,
        "declarant_position": declarant_position
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Missing required parameters: {', '.join(missing)}")

    # Validate input
    if not input_file and input_dataframe is None:
        raise ValueError("Either input_file or input_dataframe must be provided")

    if input_file and input_dataframe is not None:
        raise ValueError("Provide either input_file OR input_dataframe, not both")

    # If DataFrame provided, save to temporary CSV
    if input_dataframe is not None:
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        input_dataframe.to_csv(temp_file.name, index=False)
        input_file = temp_file.name

    # Build importer info dict
    importer_info = {
        "importer_name": importer_name,
        "importer_country": importer_country,
        "importer_eori": importer_eori,
        "declarant_name": declarant_name,
        "declarant_position": declarant_position
    }

    # Initialize pipeline
    pipeline = CBAMPipeline(
        cn_codes_path=cn_codes_path,
        cbam_rules_path=cbam_rules_path,
        suppliers_path=suppliers_path
    )

    # Run pipeline
    raw_report = pipeline.run(
        input_file=input_file,
        importer_info=importer_info,
        output_report_path=output_json,
        output_summary_path=output_summary,
        intermediate_output_dir=intermediate_dir
    )

    # Wrap in CBAMReport object
    report = CBAMReport(raw_report=raw_report)

    return report


# ============================================================================
# HELPER FUNCTIONS (For Composability)
# ============================================================================

def cbam_validate_shipments(
    input_file: Optional[str] = None,
    input_dataframe: Optional[pd.DataFrame] = None,
    cn_codes_path: str = "data/cn_codes.json",
    cbam_rules_path: str = "rules/cbam_rules.yaml",
    suppliers_path: str = "examples/demo_suppliers.yaml"
) -> Dict[str, Any]:
    """
    Validate shipments without generating a full report.

    Useful for quick validation checks or pre-flight validation.

    Args:
        input_file: Path to shipments file
        input_dataframe: Pandas DataFrame with shipments
        cn_codes_path: Path to CN codes JSON
        cbam_rules_path: Path to CBAM rules YAML
        suppliers_path: Path to suppliers YAML

    Returns:
        Validation result dictionary with metadata, errors, warnings

    Example:
        result = cbam_validate_shipments(input_file="shipments.csv")
        print(f"Valid: {result['metadata']['valid_records']}")
        print(f"Invalid: {result['metadata']['invalid_records']}")
    """
    # Handle DataFrame input
    if input_dataframe is not None:
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        input_dataframe.to_csv(temp_file.name, index=False)
        input_file = temp_file.name

    if not input_file:
        raise ValueError("Either input_file or input_dataframe must be provided")

    # Initialize agent
    agent = ShipmentIntakeAgent(
        cn_codes_path=cn_codes_path,
        cbam_rules_path=cbam_rules_path,
        suppliers_path=suppliers_path
    )

    # Process
    result = agent.process(input_file)

    return result


def cbam_calculate_emissions(
    shipments: Union[List[Dict], pd.DataFrame],
    suppliers_path: str = "examples/demo_suppliers.yaml",
    cbam_rules_path: str = "rules/cbam_rules.yaml"
) -> Dict[str, Any]:
    """
    Calculate emissions for validated shipments.

    This is a lower-level function for users who want to use individual
    agents in their own pipelines.

    Args:
        shipments: List of shipment dicts or DataFrame
        suppliers_path: Path to suppliers YAML
        cbam_rules_path: Path to CBAM rules YAML

    Returns:
        Calculation result dictionary with shipments and emissions

    Example:
        # After validation
        validated = cbam_validate_shipments(input_file="shipments.csv")
        shipments = validated['shipments']

        # Calculate emissions
        result = cbam_calculate_emissions(shipments)
        print(f"Total emissions: {result['metadata']['total_emissions_tco2']:.2f} tCO2")
    """
    # Convert DataFrame to list of dicts
    if isinstance(shipments, pd.DataFrame):
        shipments = shipments.to_dict('records')

    # Initialize agent
    agent = EmissionsCalculatorAgent(
        suppliers_path=suppliers_path,
        cbam_rules_path=cbam_rules_path
    )

    # Calculate
    result = agent.calculate_batch(shipments)

    return result


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Example usage of the SDK."""

    print("CBAM Importer Copilot - Python SDK")
    print("=" * 60)
    print()

    # Example 1: Basic usage
    print("Example 1: Basic usage")
    print("-" * 60)

    try:
        report = cbam_build_report(
            input_file="../examples/demo_shipments.csv",
            importer_name="Acme Steel EU BV",
            importer_country="NL",
            importer_eori="NL123456789012",
            declarant_name="John Smith",
            declarant_position="Compliance Officer",
            cn_codes_path="../data/cn_codes.json",
            cbam_rules_path="../rules/cbam_rules.yaml",
            suppliers_path="../examples/demo_suppliers.yaml"
        )

        print(report.summary())

    except FileNotFoundError as e:
        print(f"Note: Demo files not found (this is OK when running from sdk/ directory)")
        print(f"Error: {e}")
        print()

    # Example 2: Using configuration
    print("\nExample 2: Using configuration")
    print("-" * 60)

    config = CBAMConfig(
        importer_name="Acme Steel EU BV",
        importer_country="NL",
        importer_eori="NL123456789012",
        declarant_name="John Smith",
        declarant_position="Compliance Officer"
    )

    print(f"Config created: {config.importer_name}")
    print()

    # Example 3: Validation only
    print("Example 3: Validation only")
    print("-" * 60)

    try:
        result = cbam_validate_shipments(
            input_file="../examples/demo_shipments.csv",
            cn_codes_path="../data/cn_codes.json",
            cbam_rules_path="../rules/cbam_rules.yaml",
            suppliers_path="../examples/demo_suppliers.yaml"
        )

        print(f"Total records: {result['metadata']['total_records']}")
        print(f"Valid: {result['metadata']['valid_records']}")
        print(f"Invalid: {result['metadata']['invalid_records']}")

    except FileNotFoundError:
        print("Note: Demo files not found (this is OK when running from sdk/ directory)")

    print()
    print("=" * 60)
    print("SDK examples complete!")
