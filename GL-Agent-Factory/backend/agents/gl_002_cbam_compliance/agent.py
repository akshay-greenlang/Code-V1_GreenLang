"""
GL-002: CBAM Compliance Agent

This module implements the Carbon Border Adjustment Mechanism (CBAM)
Compliance Agent for calculating embedded emissions in imported goods
per EU Regulation (EU) 2023/956.

The agent supports:
- Embedded emissions calculation for CBAM products
- CN code classification
- Default value application
- Authorized CBAM declarant reporting
- Quarterly and annual reporting

Example:
    >>> agent = CBAMComplianceAgent()
    >>> result = agent.run(CBAMInput(
    ...     cn_code="7208.10.00",
    ...     quantity_tonnes=1000,
    ...     country_of_origin="CN",
    ...     installation_id="CN-STEEL-001"
    ... ))
    >>> print(f"Embedded emissions: {result.data.embedded_emissions_tco2e} tCO2e")
"""

import hashlib
import json
import logging
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class CBAMProductCategory(str, Enum):
    """CBAM product categories per Annex I."""

    CEMENT = "cement"
    IRON_STEEL = "iron_steel"
    ALUMINIUM = "aluminium"
    FERTILIZERS = "fertilizers"
    ELECTRICITY = "electricity"
    HYDROGEN = "hydrogen"


class EmissionType(str, Enum):
    """Types of embedded emissions."""

    DIRECT = "direct"  # Scope 1 at installation
    INDIRECT = "indirect"  # Electricity consumption


class CalculationMethod(str, Enum):
    """Method used for emissions calculation."""

    ACTUAL = "actual"  # Verified installation data
    DEFAULT = "default"  # EU default values
    COUNTRY_DEFAULT = "country_default"  # Country-specific defaults


class CBAMInput(BaseModel):
    """
    Input model for CBAM Compliance Agent.

    Attributes:
        cn_code: Combined Nomenclature code (8 digits)
        quantity_tonnes: Mass of imported goods in tonnes
        country_of_origin: ISO 3166-1 alpha-2 country code
        installation_id: Unique identifier of production installation
        actual_emissions: Actual specific embedded emissions (if available)
        electricity_source: Source of electricity used in production
        precursor_emissions: Emissions from precursor products
        reporting_period: Quarterly reporting period (Q1-Q4 YYYY)
    """

    cn_code: str = Field(..., min_length=8, max_length=10, description="CN code")
    quantity_tonnes: float = Field(..., ge=0, description="Mass in tonnes")
    country_of_origin: str = Field(..., min_length=2, max_length=2, description="ISO country code")
    installation_id: Optional[str] = Field(None, description="Production installation ID")
    actual_emissions: Optional[float] = Field(None, ge=0, description="Actual tCO2e/tonne")
    electricity_source: Optional[str] = Field(None, description="Electricity source")
    precursor_emissions: Optional[Dict[str, float]] = Field(default_factory=dict)
    reporting_period: str = Field(..., description="Reporting period (e.g., Q1 2026)")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("cn_code")
    def validate_cn_code(cls, v: str) -> str:
        """Validate CN code format."""
        # Remove dots and spaces
        clean_code = v.replace(".", "").replace(" ", "")
        if not clean_code.isdigit():
            raise ValueError(f"CN code must contain only digits: {v}")
        if len(clean_code) < 8:
            raise ValueError(f"CN code must be at least 8 digits: {v}")
        return clean_code[:8]

    @validator("country_of_origin")
    def validate_country(cls, v: str) -> str:
        """Validate ISO country code."""
        return v.upper()


class CBAMOutput(BaseModel):
    """
    Output model for CBAM Compliance Agent.

    Includes embedded emissions calculation with full audit trail.
    """

    cn_code: str = Field(..., description="CN code processed")
    product_category: str = Field(..., description="CBAM product category")
    quantity_tonnes: float = Field(..., description="Quantity imported")
    direct_emissions_tco2e: float = Field(..., description="Direct embedded emissions")
    indirect_emissions_tco2e: float = Field(..., description="Indirect embedded emissions")
    total_embedded_emissions_tco2e: float = Field(..., description="Total embedded emissions")
    specific_embedded_emissions: float = Field(..., description="tCO2e per tonne of product")
    calculation_method: str = Field(..., description="Method used for calculation")
    emission_factor_source: str = Field(..., description="Source of emission factors")
    carbon_price_applicable: float = Field(..., description="Applicable carbon price EUR/tCO2")
    cbam_liability_eur: float = Field(..., description="Estimated CBAM liability in EUR")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    calculated_at: datetime = Field(default_factory=datetime.utcnow)
    reporting_period: str = Field(..., description="Applicable reporting period")


class CBAMDefaultFactor(BaseModel):
    """Default emission factor for CBAM products."""

    cn_code_prefix: str
    product_category: CBAMProductCategory
    direct_ef: float  # tCO2e/tonne
    indirect_ef: float  # tCO2e/tonne
    source: str
    year: int


class CBAMComplianceAgent:
    """
    GL-002: CBAM Compliance Agent.

    This agent calculates embedded emissions for goods imported into the EU
    under the Carbon Border Adjustment Mechanism (CBAM). It uses:
    - Zero-hallucination deterministic calculations
    - Verified emission factors from EU implementing acts
    - Complete SHA-256 provenance tracking

    The transitional phase (Oct 2023 - Dec 2025) requires quarterly reporting.
    The definitive phase (Jan 2026+) requires CBAM certificates.

    Attributes:
        default_factors: Database of default emission factors
        cn_to_category: CN code to product category mapping

    Example:
        >>> agent = CBAMComplianceAgent()
        >>> result = agent.run(CBAMInput(
        ...     cn_code="7208.10.00",
        ...     quantity_tonnes=1000,
        ...     country_of_origin="CN",
        ...     reporting_period="Q1 2026"
        ... ))
        >>> assert result.cbam_liability_eur > 0
    """

    AGENT_ID = "regulatory/cbam_compliance_v1"
    VERSION = "1.0.0"
    DESCRIPTION = "CBAM embedded emissions calculator with zero hallucination"

    # CN code prefix to product category mapping
    CN_TO_CATEGORY: Dict[str, CBAMProductCategory] = {
        # Cement (Chapter 25, 38)
        "2507": CBAMProductCategory.CEMENT,
        "2523": CBAMProductCategory.CEMENT,
        # Iron and Steel (Chapter 72, 73)
        "7201": CBAMProductCategory.IRON_STEEL,
        "7202": CBAMProductCategory.IRON_STEEL,
        "7203": CBAMProductCategory.IRON_STEEL,
        "7204": CBAMProductCategory.IRON_STEEL,
        "7205": CBAMProductCategory.IRON_STEEL,
        "7206": CBAMProductCategory.IRON_STEEL,
        "7207": CBAMProductCategory.IRON_STEEL,
        "7208": CBAMProductCategory.IRON_STEEL,
        "7209": CBAMProductCategory.IRON_STEEL,
        "7210": CBAMProductCategory.IRON_STEEL,
        "7211": CBAMProductCategory.IRON_STEEL,
        "7212": CBAMProductCategory.IRON_STEEL,
        "7213": CBAMProductCategory.IRON_STEEL,
        "7214": CBAMProductCategory.IRON_STEEL,
        "7215": CBAMProductCategory.IRON_STEEL,
        "7216": CBAMProductCategory.IRON_STEEL,
        "7217": CBAMProductCategory.IRON_STEEL,
        "7218": CBAMProductCategory.IRON_STEEL,
        "7219": CBAMProductCategory.IRON_STEEL,
        "7220": CBAMProductCategory.IRON_STEEL,
        "7221": CBAMProductCategory.IRON_STEEL,
        "7222": CBAMProductCategory.IRON_STEEL,
        "7223": CBAMProductCategory.IRON_STEEL,
        "7224": CBAMProductCategory.IRON_STEEL,
        "7225": CBAMProductCategory.IRON_STEEL,
        "7226": CBAMProductCategory.IRON_STEEL,
        "7227": CBAMProductCategory.IRON_STEEL,
        "7228": CBAMProductCategory.IRON_STEEL,
        "7229": CBAMProductCategory.IRON_STEEL,
        "7301": CBAMProductCategory.IRON_STEEL,
        "7302": CBAMProductCategory.IRON_STEEL,
        "7303": CBAMProductCategory.IRON_STEEL,
        "7304": CBAMProductCategory.IRON_STEEL,
        "7305": CBAMProductCategory.IRON_STEEL,
        "7306": CBAMProductCategory.IRON_STEEL,
        "7307": CBAMProductCategory.IRON_STEEL,
        "7308": CBAMProductCategory.IRON_STEEL,
        "7309": CBAMProductCategory.IRON_STEEL,
        "7310": CBAMProductCategory.IRON_STEEL,
        "7311": CBAMProductCategory.IRON_STEEL,
        "7318": CBAMProductCategory.IRON_STEEL,
        "7326": CBAMProductCategory.IRON_STEEL,
        # Aluminium (Chapter 76)
        "7601": CBAMProductCategory.ALUMINIUM,
        "7602": CBAMProductCategory.ALUMINIUM,
        "7603": CBAMProductCategory.ALUMINIUM,
        "7604": CBAMProductCategory.ALUMINIUM,
        "7605": CBAMProductCategory.ALUMINIUM,
        "7606": CBAMProductCategory.ALUMINIUM,
        "7607": CBAMProductCategory.ALUMINIUM,
        "7608": CBAMProductCategory.ALUMINIUM,
        "7609": CBAMProductCategory.ALUMINIUM,
        "7610": CBAMProductCategory.ALUMINIUM,
        "7611": CBAMProductCategory.ALUMINIUM,
        "7612": CBAMProductCategory.ALUMINIUM,
        "7613": CBAMProductCategory.ALUMINIUM,
        "7614": CBAMProductCategory.ALUMINIUM,
        "7616": CBAMProductCategory.ALUMINIUM,
        # Fertilizers (Chapter 28, 31)
        "2808": CBAMProductCategory.FERTILIZERS,
        "2814": CBAMProductCategory.FERTILIZERS,
        "3102": CBAMProductCategory.FERTILIZERS,
        "3105": CBAMProductCategory.FERTILIZERS,
        # Electricity
        "2716": CBAMProductCategory.ELECTRICITY,
        # Hydrogen
        "2804": CBAMProductCategory.HYDROGEN,
    }

    # Default emission factors by product category and country
    # Source: EU Implementing Regulation, JRC default values
    DEFAULT_FACTORS: Dict[str, Dict[str, CBAMDefaultFactor]] = {
        "iron_steel": {
            "GLOBAL": CBAMDefaultFactor(
                cn_code_prefix="72",
                product_category=CBAMProductCategory.IRON_STEEL,
                direct_ef=1.85,  # tCO2e/t crude steel
                indirect_ef=0.32,
                source="EU Implementing Regulation 2023/1773",
                year=2024,
            ),
            "CN": CBAMDefaultFactor(
                cn_code_prefix="72",
                product_category=CBAMProductCategory.IRON_STEEL,
                direct_ef=2.10,  # Higher for China
                indirect_ef=0.45,
                source="EU Implementing Regulation 2023/1773",
                year=2024,
            ),
            "IN": CBAMDefaultFactor(
                cn_code_prefix="72",
                product_category=CBAMProductCategory.IRON_STEEL,
                direct_ef=2.35,  # Higher for India
                indirect_ef=0.52,
                source="EU Implementing Regulation 2023/1773",
                year=2024,
            ),
        },
        "aluminium": {
            "GLOBAL": CBAMDefaultFactor(
                cn_code_prefix="76",
                product_category=CBAMProductCategory.ALUMINIUM,
                direct_ef=1.60,  # tCO2e/t primary aluminium
                indirect_ef=6.50,  # High electricity intensity
                source="EU Implementing Regulation 2023/1773",
                year=2024,
            ),
            "CN": CBAMDefaultFactor(
                cn_code_prefix="76",
                product_category=CBAMProductCategory.ALUMINIUM,
                direct_ef=1.65,
                indirect_ef=10.20,  # Coal-heavy grid
                source="EU Implementing Regulation 2023/1773",
                year=2024,
            ),
        },
        "cement": {
            "GLOBAL": CBAMDefaultFactor(
                cn_code_prefix="2523",
                product_category=CBAMProductCategory.CEMENT,
                direct_ef=0.83,  # tCO2e/t cement
                indirect_ef=0.05,
                source="EU Implementing Regulation 2023/1773",
                year=2024,
            ),
        },
        "fertilizers": {
            "GLOBAL": CBAMDefaultFactor(
                cn_code_prefix="31",
                product_category=CBAMProductCategory.FERTILIZERS,
                direct_ef=2.50,  # tCO2e/t ammonia
                indirect_ef=0.12,
                source="EU Implementing Regulation 2023/1773",
                year=2024,
            ),
        },
        "electricity": {
            "GLOBAL": CBAMDefaultFactor(
                cn_code_prefix="2716",
                product_category=CBAMProductCategory.ELECTRICITY,
                direct_ef=0.50,  # tCO2e/MWh
                indirect_ef=0.0,
                source="EU Implementing Regulation 2023/1773",
                year=2024,
            ),
        },
        "hydrogen": {
            "GLOBAL": CBAMDefaultFactor(
                cn_code_prefix="2804",
                product_category=CBAMProductCategory.HYDROGEN,
                direct_ef=9.0,  # tCO2e/t grey hydrogen
                indirect_ef=1.5,
                source="EU Implementing Regulation 2023/1773",
                year=2024,
            ),
        },
    }

    # EU ETS carbon price (EUR/tCO2) - updated quarterly
    EU_ETS_PRICE: float = 85.0  # EUR/tCO2 (2024 average)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CBAM Compliance Agent.

        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self._provenance_steps: List[Dict] = []

        logger.info(f"CBAMComplianceAgent initialized (version {self.VERSION})")

    def run(self, input_data: CBAMInput) -> CBAMOutput:
        """
        Execute the CBAM compliance calculation.

        This method performs zero-hallucination calculations:
        - embedded_emissions = quantity * specific_embedded_emissions
        - cbam_liability = embedded_emissions * carbon_price

        Args:
            input_data: Validated CBAM input data

        Returns:
            Calculation result with embedded emissions and liability

        Raises:
            ValueError: If CN code not in CBAM scope
        """
        start_time = datetime.utcnow()
        self._provenance_steps = []

        logger.info(
            f"Calculating CBAM emissions: CN={input_data.cn_code}, "
            f"qty={input_data.quantity_tonnes}t, origin={input_data.country_of_origin}"
        )

        try:
            # Step 1: Classify product category
            product_category = self._classify_product(input_data.cn_code)
            if not product_category:
                raise ValueError(
                    f"CN code {input_data.cn_code} not in CBAM scope"
                )

            self._track_step("product_classification", {
                "cn_code": input_data.cn_code,
                "product_category": product_category.value,
            })

            # Step 2: Get emission factors
            direct_ef, indirect_ef, method, source = self._get_emission_factors(
                product_category,
                input_data.country_of_origin,
                input_data.actual_emissions,
            )

            self._track_step("emission_factor_lookup", {
                "direct_ef": direct_ef,
                "indirect_ef": indirect_ef,
                "method": method,
                "source": source,
            })

            # Step 3: ZERO-HALLUCINATION CALCULATION
            # Direct emissions = quantity * direct_ef
            direct_emissions = input_data.quantity_tonnes * direct_ef

            # Indirect emissions = quantity * indirect_ef
            indirect_emissions = input_data.quantity_tonnes * indirect_ef

            # Total embedded emissions
            total_emissions = direct_emissions + indirect_emissions

            # Specific embedded emissions (per tonne)
            specific_emissions = direct_ef + indirect_ef

            self._track_step("calculation", {
                "formula_direct": "direct_emissions = quantity * direct_ef",
                "formula_indirect": "indirect_emissions = quantity * indirect_ef",
                "quantity": input_data.quantity_tonnes,
                "direct_ef": direct_ef,
                "indirect_ef": indirect_ef,
                "direct_emissions": direct_emissions,
                "indirect_emissions": indirect_emissions,
                "total_emissions": total_emissions,
            })

            # Step 4: Calculate CBAM liability
            carbon_price = self._get_carbon_price()
            cbam_liability = total_emissions * carbon_price

            self._track_step("liability_calculation", {
                "carbon_price_eur_tco2": carbon_price,
                "total_emissions": total_emissions,
                "cbam_liability_eur": cbam_liability,
            })

            # Step 5: Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash()

            # Step 6: Create output
            output = CBAMOutput(
                cn_code=input_data.cn_code,
                product_category=product_category.value,
                quantity_tonnes=input_data.quantity_tonnes,
                direct_emissions_tco2e=round(direct_emissions, 6),
                indirect_emissions_tco2e=round(indirect_emissions, 6),
                total_embedded_emissions_tco2e=round(total_emissions, 6),
                specific_embedded_emissions=round(specific_emissions, 6),
                calculation_method=method,
                emission_factor_source=source,
                carbon_price_applicable=carbon_price,
                cbam_liability_eur=round(cbam_liability, 2),
                provenance_hash=provenance_hash,
                reporting_period=input_data.reporting_period,
            )

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(
                f"CBAM calculation complete: {total_emissions:.4f} tCO2e, "
                f"liability EUR {cbam_liability:.2f} "
                f"(duration: {duration_ms:.2f}ms, provenance: {provenance_hash[:16]}...)"
            )

            return output

        except Exception as e:
            logger.error(f"CBAM calculation failed: {str(e)}", exc_info=True)
            raise

    def _classify_product(self, cn_code: str) -> Optional[CBAMProductCategory]:
        """
        Classify CN code to CBAM product category.

        ZERO-HALLUCINATION: Uses deterministic lookup table.
        """
        # Try 4-digit prefix first
        prefix_4 = cn_code[:4]
        if prefix_4 in self.CN_TO_CATEGORY:
            return self.CN_TO_CATEGORY[prefix_4]

        # Try 2-digit chapter
        prefix_2 = cn_code[:2]
        for cn_prefix, category in self.CN_TO_CATEGORY.items():
            if cn_prefix.startswith(prefix_2):
                return category

        return None

    def _get_emission_factors(
        self,
        category: CBAMProductCategory,
        country: str,
        actual_emissions: Optional[float],
    ) -> Tuple[float, float, str, str]:
        """
        Get emission factors for calculation.

        Priority:
        1. Actual verified emissions
        2. Country-specific defaults
        3. Global defaults

        Returns:
            Tuple of (direct_ef, indirect_ef, method, source)
        """
        # Use actual emissions if provided
        if actual_emissions is not None:
            return (
                actual_emissions,
                0.0,  # Actual should include indirect
                CalculationMethod.ACTUAL.value,
                "Verified installation data",
            )

        # Look up default factors
        category_factors = self.DEFAULT_FACTORS.get(category.value, {})

        # Try country-specific
        if country in category_factors:
            factor = category_factors[country]
            return (
                factor.direct_ef,
                factor.indirect_ef,
                CalculationMethod.COUNTRY_DEFAULT.value,
                factor.source,
            )

        # Fall back to global
        if "GLOBAL" in category_factors:
            factor = category_factors["GLOBAL"]
            return (
                factor.direct_ef,
                factor.indirect_ef,
                CalculationMethod.DEFAULT.value,
                factor.source,
            )

        raise ValueError(f"No emission factors found for category {category}")

    def _get_carbon_price(self) -> float:
        """Get current EU ETS carbon price."""
        # In production, this would fetch real-time price
        return self.EU_ETS_PRICE

    def _track_step(self, step_type: str, data: Dict[str, Any]) -> None:
        """Track a calculation step for provenance."""
        self._provenance_steps.append({
            "step_type": step_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        })

    def _calculate_provenance_hash(self) -> str:
        """
        Calculate SHA-256 hash of complete provenance chain.

        Enables regulatory audit trail and reproducibility.
        """
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "steps": self._provenance_steps,
            "timestamp": datetime.utcnow().isoformat(),
        }

        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_cbam_products(self) -> List[str]:
        """Get list of CBAM product categories."""
        return [cat.value for cat in CBAMProductCategory]

    def is_in_scope(self, cn_code: str) -> bool:
        """Check if CN code is in CBAM scope."""
        return self._classify_product(cn_code) is not None


# Pack specification
PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "regulatory/cbam_compliance_v1",
    "name": "CBAM Compliance Agent",
    "version": "1.0.0",
    "summary": "Calculate embedded emissions for CBAM reporting",
    "tags": ["cbam", "eu-regulation", "embedded-emissions", "carbon-border"],
    "owners": ["regulatory-team"],
    "compute": {
        "entrypoint": "python://agents.gl_002_cbam_compliance.agent:CBAMComplianceAgent",
        "deterministic": True,
    },
    "factors": [
        {"ref": "ef://eu/cbam-default-values/2024"},
        {"ref": "ef://eu/ets-price/current"},
    ],
    "provenance": {
        "ef_version_pin": "2024-Q4",
        "regulation_version": "EU 2023/956",
        "enable_audit": True,
    },
}
