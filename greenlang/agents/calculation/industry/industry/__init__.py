"""
GreenLang Industry Emission Calculators - Zero Hallucination Guarantee

This package provides deterministic, bit-perfect, reproducible emission
calculators for major industrial sectors. All calculators guarantee:

    1. ZERO HALLUCINATION: No LLM/AI in calculation path
    2. DETERMINISTIC: Same input always produces identical output
    3. REPRODUCIBLE: Complete provenance tracking with SHA-256 hashes
    4. AUDITABLE: Every calculation step is documented
    5. COMPLIANT: CBAM, IPCC, and regulatory methodology alignment

Supported Industries:
    - Steel (BF-BOF, EAF, DRI-EAF, H2-DRI)
    - Cement (CEM I-V, clinker ratios, SCM substitution)
    - Aluminum (Primary smelting, alumina refining, secondary recycling)
    - Chemicals (Ammonia SMR/coal/green, Hydrogen gray/blue/green)
    - Glass (Container, flat, cullet credits)
    - Paper (Virgin kraft, recycled, biogenic carbon tracking)

Usage:
    from greenlang.calculators.industry import (
        SteelEmissionCalculator,
        CementEmissionCalculator,
        AluminumEmissionCalculator,
        ChemicalEmissionCalculator,
        GlassEmissionCalculator,
        PaperEmissionCalculator
    )

    # Example: Steel calculation
    calculator = SteelEmissionCalculator()
    result = calculator.calculate(SteelCalculationInput(...))

    # Result includes:
    # - total_emissions_tco2
    # - emission_intensity_tco2_per_tonne
    # - calculation_steps (complete provenance)
    # - provenance_hash (SHA-256 for reproducibility verification)

Sources:
    All emission factors are from authoritative, peer-reviewed sources:
    - IPCC 2006 Guidelines for National Greenhouse Gas Inventories
    - IEA Technology Roadmaps (Steel, Hydrogen, Ammonia)
    - World Steel Association CO2 Data Collection
    - GCCA Getting the Numbers Right (Cement)
    - International Aluminium Institute LCA Dataset
    - CEPI Key Statistics (Paper)
    - Glass Alliance Europe Decarbonisation Roadmap

CBAM Compliance:
    All calculators support CBAM (Carbon Border Adjustment Mechanism)
    output formatting per EU Regulation 2023/956 Annex III.

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "GreenLang Team"
__license__ = "Proprietary"

# =============================================================================
# STEEL CALCULATOR
# =============================================================================

from .steel_calculator import (
    SteelEmissionCalculator,
    SteelCalculationInput,
    SteelCalculationResult,
    SteelProductionRoute,
    HydrogenSource,
    SteelEmissionFactors,
    format_cbam_output as format_steel_cbam_output,
)

# =============================================================================
# CEMENT CALCULATOR
# =============================================================================

from .cement_calculator import (
    CementEmissionCalculator,
    CementCalculationInput,
    CementCalculationResult,
    CementType,
    KilnFuelType,
    SCMType,
    SCMInput,
    CementEmissionFactors,
    format_cbam_output as format_cement_cbam_output,
)

# =============================================================================
# ALUMINUM CALCULATOR
# =============================================================================

from .aluminum_calculator import (
    AluminumEmissionCalculator,
    AluminumCalculationInput,
    AluminumCalculationResult,
    AluminumProductionRoute,
    AnodeType,
    PFCPerformanceLevel,
    AluminumEmissionFactors,
    format_cbam_output as format_aluminum_cbam_output,
)

# =============================================================================
# CHEMICALS CALCULATOR (Ammonia & Hydrogen)
# =============================================================================

from .chemicals_calculator import (
    ChemicalEmissionCalculator,
    AmmoniaCalculationInput,
    AmmoniaCalculationResult,
    HydrogenCalculationInput,
    HydrogenCalculationResult,
    AmmoniaProductionRoute,
    HydrogenProductionRoute,
    ChemicalEmissionFactors,
    format_cbam_output_ammonia,
    format_cbam_output_hydrogen,
)

# =============================================================================
# GLASS CALCULATOR
# =============================================================================

from .glass_calculator import (
    GlassEmissionCalculator,
    GlassCalculationInput,
    GlassCalculationResult,
    GlassType,
    FuelType as GlassFuelType,
    GlassEmissionFactors,
    format_cbam_output as format_glass_cbam_output,
)

# =============================================================================
# PAPER CALCULATOR
# =============================================================================

from .paper_calculator import (
    PaperEmissionCalculator,
    PaperCalculationInput,
    PaperCalculationResult,
    PaperProductType,
    EnergySource as PaperEnergySource,
    PaperEmissionFactors,
    format_cbam_output as format_paper_cbam_output,
)

# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

# All calculators
ALL_CALCULATORS = [
    SteelEmissionCalculator,
    CementEmissionCalculator,
    AluminumEmissionCalculator,
    ChemicalEmissionCalculator,
    GlassEmissionCalculator,
    PaperEmissionCalculator,
]

# All emission factor classes
ALL_EMISSION_FACTORS = [
    SteelEmissionFactors,
    CementEmissionFactors,
    AluminumEmissionFactors,
    ChemicalEmissionFactors,
    GlassEmissionFactors,
    PaperEmissionFactors,
]

# CBAM-relevant industries (currently in scope)
CBAM_INDUSTRIES = [
    "Steel",
    "Cement",
    "Aluminum",
    "Chemicals",  # Hydrogen, Ammonia
]

# Industry to calculator mapping
INDUSTRY_CALCULATORS = {
    "steel": SteelEmissionCalculator,
    "cement": CementEmissionCalculator,
    "aluminum": AluminumEmissionCalculator,
    "aluminium": AluminumEmissionCalculator,  # British spelling
    "chemicals": ChemicalEmissionCalculator,
    "ammonia": ChemicalEmissionCalculator,
    "hydrogen": ChemicalEmissionCalculator,
    "glass": GlassEmissionCalculator,
    "paper": PaperEmissionCalculator,
    "pulp": PaperEmissionCalculator,
}


def get_calculator(industry: str):
    """
    Get the appropriate calculator for an industry.

    Args:
        industry: Industry name (case-insensitive)

    Returns:
        Calculator instance

    Raises:
        ValueError: If industry is not supported

    Example:
        calculator = get_calculator("steel")
        result = calculator.calculate(input_data)
    """
    industry_lower = industry.lower().strip()

    if industry_lower not in INDUSTRY_CALCULATORS:
        supported = ", ".join(sorted(set(INDUSTRY_CALCULATORS.keys())))
        raise ValueError(
            f"Unsupported industry: '{industry}'. "
            f"Supported industries: {supported}"
        )

    calculator_class = INDUSTRY_CALCULATORS[industry_lower]
    return calculator_class()


def list_supported_industries() -> list:
    """
    List all supported industries.

    Returns:
        List of industry names
    """
    return sorted(set(INDUSTRY_CALCULATORS.keys()))


def get_emission_factors(industry: str) -> type:
    """
    Get emission factor constants for an industry.

    Args:
        industry: Industry name

    Returns:
        Emission factor class with all constants
    """
    factor_map = {
        "steel": SteelEmissionFactors,
        "cement": CementEmissionFactors,
        "aluminum": AluminumEmissionFactors,
        "aluminium": AluminumEmissionFactors,
        "chemicals": ChemicalEmissionFactors,
        "ammonia": ChemicalEmissionFactors,
        "hydrogen": ChemicalEmissionFactors,
        "glass": GlassEmissionFactors,
        "paper": PaperEmissionFactors,
        "pulp": PaperEmissionFactors,
    }

    industry_lower = industry.lower().strip()
    if industry_lower not in factor_map:
        raise ValueError(f"Unsupported industry: '{industry}'")

    return factor_map[industry_lower]


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Version
    "__version__",

    # Steel
    "SteelEmissionCalculator",
    "SteelCalculationInput",
    "SteelCalculationResult",
    "SteelProductionRoute",
    "HydrogenSource",
    "SteelEmissionFactors",
    "format_steel_cbam_output",

    # Cement
    "CementEmissionCalculator",
    "CementCalculationInput",
    "CementCalculationResult",
    "CementType",
    "KilnFuelType",
    "SCMType",
    "SCMInput",
    "CementEmissionFactors",
    "format_cement_cbam_output",

    # Aluminum
    "AluminumEmissionCalculator",
    "AluminumCalculationInput",
    "AluminumCalculationResult",
    "AluminumProductionRoute",
    "AnodeType",
    "PFCPerformanceLevel",
    "AluminumEmissionFactors",
    "format_aluminum_cbam_output",

    # Chemicals
    "ChemicalEmissionCalculator",
    "AmmoniaCalculationInput",
    "AmmoniaCalculationResult",
    "HydrogenCalculationInput",
    "HydrogenCalculationResult",
    "AmmoniaProductionRoute",
    "HydrogenProductionRoute",
    "ChemicalEmissionFactors",
    "format_cbam_output_ammonia",
    "format_cbam_output_hydrogen",

    # Glass
    "GlassEmissionCalculator",
    "GlassCalculationInput",
    "GlassCalculationResult",
    "GlassType",
    "GlassFuelType",
    "GlassEmissionFactors",
    "format_glass_cbam_output",

    # Paper
    "PaperEmissionCalculator",
    "PaperCalculationInput",
    "PaperCalculationResult",
    "PaperProductType",
    "PaperEnergySource",
    "PaperEmissionFactors",
    "format_paper_cbam_output",

    # Utilities
    "get_calculator",
    "list_supported_industries",
    "get_emission_factors",
    "ALL_CALCULATORS",
    "ALL_EMISSION_FACTORS",
    "CBAM_INDUSTRIES",
    "INDUSTRY_CALCULATORS",
]
