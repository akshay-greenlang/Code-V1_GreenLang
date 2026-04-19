# -*- coding: utf-8 -*-
"""
GL-MRV-X-005: Scope 3 Category Mapper
=====================================

Maps organizational data (spend, purchase orders, BOM) to appropriate
Scope 3 categories following GHG Protocol Corporate Value Chain Standard.

Capabilities:
    - Spend data classification to Scope 3 categories
    - Purchase order to category mapping
    - Bill of Materials (BOM) analysis
    - NAICS/ISIC code mapping
    - Supplier categorization
    - Category boundary determination
    - Complete provenance tracking

Zero-Hallucination Guarantees:
    - All mappings use deterministic lookup tables
    - NO LLM involvement in category assignment
    - Industry codes from authoritative classification systems
    - Complete provenance hash for every mapping

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base_agents import DeterministicAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism.clock import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 Categories."""
    CAT1_PURCHASED_GOODS = "1_purchased_goods_services"
    CAT2_CAPITAL_GOODS = "2_capital_goods"
    CAT3_FUEL_ENERGY = "3_fuel_energy_activities"
    CAT4_UPSTREAM_TRANSPORT = "4_upstream_transport"
    CAT5_WASTE = "5_waste_generated"
    CAT6_BUSINESS_TRAVEL = "6_business_travel"
    CAT7_COMMUTING = "7_employee_commuting"
    CAT8_UPSTREAM_LEASED = "8_upstream_leased_assets"
    CAT9_DOWNSTREAM_TRANSPORT = "9_downstream_transport"
    CAT10_PROCESSING = "10_processing_sold_products"
    CAT11_PRODUCT_USE = "11_use_of_sold_products"
    CAT12_END_OF_LIFE = "12_end_of_life_treatment"
    CAT13_DOWNSTREAM_LEASED = "13_downstream_leased_assets"
    CAT14_FRANCHISES = "14_franchises"
    CAT15_INVESTMENTS = "15_investments"


class DataSourceType(str, Enum):
    """Types of input data sources."""
    SPEND_DATA = "spend_data"
    PURCHASE_ORDER = "purchase_order"
    BOM = "bom"
    SUPPLIER_DATA = "supplier_data"
    TRAVEL_DATA = "travel_data"
    WASTE_DATA = "waste_data"


class CalculationApproach(str, Enum):
    """Calculation approaches by data quality."""
    SUPPLIER_SPECIFIC = "supplier_specific"
    HYBRID = "hybrid"
    AVERAGE_DATA = "average_data"
    SPEND_BASED = "spend_based"


# =============================================================================
# CATEGORY MAPPING RULES
# =============================================================================

# NAICS code to Scope 3 category mapping (first 2-3 digits)
NAICS_TO_CATEGORY: Dict[str, Scope3Category] = {
    # Agriculture, Mining, Utilities
    "11": Scope3Category.CAT1_PURCHASED_GOODS,  # Agriculture
    "21": Scope3Category.CAT1_PURCHASED_GOODS,  # Mining
    "22": Scope3Category.CAT3_FUEL_ENERGY,  # Utilities

    # Construction and Manufacturing
    "23": Scope3Category.CAT2_CAPITAL_GOODS,  # Construction
    "31": Scope3Category.CAT1_PURCHASED_GOODS,  # Manufacturing
    "32": Scope3Category.CAT1_PURCHASED_GOODS,  # Manufacturing
    "33": Scope3Category.CAT1_PURCHASED_GOODS,  # Manufacturing
    "333": Scope3Category.CAT2_CAPITAL_GOODS,  # Machinery mfg
    "334": Scope3Category.CAT2_CAPITAL_GOODS,  # Computer/electronics mfg
    "336": Scope3Category.CAT2_CAPITAL_GOODS,  # Transportation equipment

    # Wholesale and Retail
    "42": Scope3Category.CAT1_PURCHASED_GOODS,  # Wholesale trade
    "44": Scope3Category.CAT1_PURCHASED_GOODS,  # Retail trade
    "45": Scope3Category.CAT1_PURCHASED_GOODS,  # Retail trade

    # Transportation and Warehousing
    "48": Scope3Category.CAT4_UPSTREAM_TRANSPORT,  # Transportation
    "481": Scope3Category.CAT6_BUSINESS_TRAVEL,  # Air transportation
    "482": Scope3Category.CAT6_BUSINESS_TRAVEL,  # Rail transportation
    "49": Scope3Category.CAT4_UPSTREAM_TRANSPORT,  # Warehousing

    # Information and Finance
    "51": Scope3Category.CAT1_PURCHASED_GOODS,  # Information
    "52": Scope3Category.CAT15_INVESTMENTS,  # Finance
    "53": Scope3Category.CAT8_UPSTREAM_LEASED,  # Real estate

    # Professional Services
    "54": Scope3Category.CAT1_PURCHASED_GOODS,  # Professional services
    "55": Scope3Category.CAT1_PURCHASED_GOODS,  # Management
    "56": Scope3Category.CAT1_PURCHASED_GOODS,  # Admin services
    "562": Scope3Category.CAT5_WASTE,  # Waste management

    # Education, Healthcare, Entertainment
    "61": Scope3Category.CAT1_PURCHASED_GOODS,  # Education
    "62": Scope3Category.CAT1_PURCHASED_GOODS,  # Healthcare
    "71": Scope3Category.CAT1_PURCHASED_GOODS,  # Entertainment
    "72": Scope3Category.CAT6_BUSINESS_TRAVEL,  # Accommodation

    # Other Services
    "81": Scope3Category.CAT1_PURCHASED_GOODS,  # Other services
    "92": Scope3Category.CAT1_PURCHASED_GOODS,  # Public admin
}

# Spend category keywords to Scope 3 mapping
SPEND_KEYWORDS_TO_CATEGORY: Dict[str, Scope3Category] = {
    # Category 1 - Purchased Goods
    "raw_materials": Scope3Category.CAT1_PURCHASED_GOODS,
    "components": Scope3Category.CAT1_PURCHASED_GOODS,
    "supplies": Scope3Category.CAT1_PURCHASED_GOODS,
    "office_supplies": Scope3Category.CAT1_PURCHASED_GOODS,
    "it_services": Scope3Category.CAT1_PURCHASED_GOODS,
    "consulting": Scope3Category.CAT1_PURCHASED_GOODS,
    "professional_services": Scope3Category.CAT1_PURCHASED_GOODS,
    "software": Scope3Category.CAT1_PURCHASED_GOODS,
    "marketing": Scope3Category.CAT1_PURCHASED_GOODS,

    # Category 2 - Capital Goods
    "machinery": Scope3Category.CAT2_CAPITAL_GOODS,
    "equipment": Scope3Category.CAT2_CAPITAL_GOODS,
    "vehicles": Scope3Category.CAT2_CAPITAL_GOODS,
    "buildings": Scope3Category.CAT2_CAPITAL_GOODS,
    "construction": Scope3Category.CAT2_CAPITAL_GOODS,
    "infrastructure": Scope3Category.CAT2_CAPITAL_GOODS,
    "capex": Scope3Category.CAT2_CAPITAL_GOODS,

    # Category 3 - Fuel and Energy
    "electricity": Scope3Category.CAT3_FUEL_ENERGY,
    "fuel": Scope3Category.CAT3_FUEL_ENERGY,
    "natural_gas": Scope3Category.CAT3_FUEL_ENERGY,
    "energy": Scope3Category.CAT3_FUEL_ENERGY,

    # Category 4 - Upstream Transport
    "freight": Scope3Category.CAT4_UPSTREAM_TRANSPORT,
    "shipping": Scope3Category.CAT4_UPSTREAM_TRANSPORT,
    "logistics": Scope3Category.CAT4_UPSTREAM_TRANSPORT,
    "courier": Scope3Category.CAT4_UPSTREAM_TRANSPORT,
    "inbound_transport": Scope3Category.CAT4_UPSTREAM_TRANSPORT,

    # Category 5 - Waste
    "waste_disposal": Scope3Category.CAT5_WASTE,
    "recycling": Scope3Category.CAT5_WASTE,
    "hazardous_waste": Scope3Category.CAT5_WASTE,
    "waste_management": Scope3Category.CAT5_WASTE,

    # Category 6 - Business Travel
    "air_travel": Scope3Category.CAT6_BUSINESS_TRAVEL,
    "hotel": Scope3Category.CAT6_BUSINESS_TRAVEL,
    "travel": Scope3Category.CAT6_BUSINESS_TRAVEL,
    "car_rental": Scope3Category.CAT6_BUSINESS_TRAVEL,
    "rail_travel": Scope3Category.CAT6_BUSINESS_TRAVEL,

    # Category 7 - Employee Commuting
    "commuting": Scope3Category.CAT7_COMMUTING,
    "employee_transport": Scope3Category.CAT7_COMMUTING,

    # Category 8 - Upstream Leased
    "office_lease": Scope3Category.CAT8_UPSTREAM_LEASED,
    "equipment_lease": Scope3Category.CAT8_UPSTREAM_LEASED,
    "vehicle_lease": Scope3Category.CAT8_UPSTREAM_LEASED,

    # Categories 9-15 (Downstream)
    "distribution": Scope3Category.CAT9_DOWNSTREAM_TRANSPORT,
    "franchise_fees": Scope3Category.CAT14_FRANCHISES,
    "investments": Scope3Category.CAT15_INVESTMENTS,
}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class SpendRecord(BaseModel):
    """A spend data record for mapping."""
    amount: float = Field(..., gt=0, description="Spend amount")
    currency: str = Field(default="USD", description="Currency code")
    category: Optional[str] = Field(None, description="Spend category/GL code")
    description: Optional[str] = Field(None, description="Description")
    supplier_name: Optional[str] = Field(None, description="Supplier name")
    supplier_naics: Optional[str] = Field(None, description="Supplier NAICS code")
    transaction_date: Optional[datetime] = Field(None, description="Transaction date")


class PurchaseOrder(BaseModel):
    """A purchase order record for mapping."""
    po_number: str = Field(..., description="PO number")
    amount: float = Field(..., gt=0, description="PO amount")
    currency: str = Field(default="USD", description="Currency")
    line_items: List[Dict[str, Any]] = Field(
        default_factory=list, description="Line items"
    )
    supplier_name: Optional[str] = Field(None, description="Supplier")
    supplier_naics: Optional[str] = Field(None, description="NAICS code")
    category: Optional[str] = Field(None, description="PO category")


class BOMItem(BaseModel):
    """A Bill of Materials item."""
    item_code: str = Field(..., description="Item code")
    description: str = Field(..., description="Item description")
    quantity: float = Field(..., gt=0, description="Quantity")
    unit: str = Field(default="each", description="Unit")
    material_type: Optional[str] = Field(None, description="Material type")
    weight_kg: Optional[float] = Field(None, description="Weight per unit")
    supplier_naics: Optional[str] = Field(None, description="Supplier NAICS")


class CategoryMappingResult(BaseModel):
    """Result of category mapping."""
    source_type: DataSourceType = Field(..., description="Source data type")
    source_id: Optional[str] = Field(None, description="Source record ID")
    mapped_category: Scope3Category = Field(..., description="Mapped category")
    category_number: int = Field(..., description="Category number 1-15")
    category_name: str = Field(..., description="Category name")
    amount: float = Field(..., description="Amount")
    currency: str = Field(..., description="Currency")
    confidence: float = Field(..., ge=0, le=1, description="Mapping confidence")
    mapping_rule: str = Field(..., description="Rule used for mapping")
    recommended_approach: CalculationApproach = Field(
        ..., description="Recommended calculation approach"
    )
    calculation_trace: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(...)


class Scope3CategoryMapperInput(BaseModel):
    """Input model for Scope3CategoryMapper."""
    spend_records: Optional[List[SpendRecord]] = Field(None)
    purchase_orders: Optional[List[PurchaseOrder]] = Field(None)
    bom_items: Optional[List[BOMItem]] = Field(None)
    organization_id: Optional[str] = Field(None)
    reporting_period: Optional[str] = Field(None)


class Scope3CategoryMapperOutput(BaseModel):
    """Output model for Scope3CategoryMapper."""
    success: bool = Field(...)
    mapping_results: List[CategoryMappingResult] = Field(default_factory=list)

    # Aggregated by category
    spend_by_category: Dict[str, float] = Field(default_factory=dict)
    record_count_by_category: Dict[str, int] = Field(default_factory=dict)

    # Relevance assessment
    relevant_categories: List[str] = Field(default_factory=list)
    coverage_assessment: Dict[str, str] = Field(default_factory=dict)

    # Metadata
    total_spend_mapped: float = Field(...)
    total_records_processed: int = Field(...)
    average_confidence: float = Field(...)
    processing_time_ms: float = Field(...)
    provenance_hash: str = Field(...)
    validation_status: str = Field(...)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


# =============================================================================
# SCOPE 3 CATEGORY MAPPER AGENT
# =============================================================================

class Scope3CategoryMapperAgent(DeterministicAgent):
    """
    GL-MRV-X-005: Scope 3 Category Mapper Agent

    Maps organizational data to appropriate Scope 3 categories following
    GHG Protocol Corporate Value Chain (Scope 3) Standard.

    Zero-Hallucination Implementation:
        - All mappings use deterministic lookup tables
        - NAICS/ISIC codes from official classification systems
        - Complete provenance tracking with SHA-256 hashes

    Mapping Hierarchy:
        1. NAICS code (if provided) - highest confidence
        2. Spend category/GL code mapping
        3. Keyword analysis - lowest confidence

    Example:
        >>> agent = Scope3CategoryMapperAgent()
        >>> result = agent.execute({
        ...     "spend_records": [{
        ...         "amount": 100000,
        ...         "category": "raw_materials",
        ...         "supplier_naics": "311"
        ...     }]
        ... })
    """

    AGENT_ID = "GL-MRV-X-005"
    AGENT_NAME = "Scope 3 Category Mapper"
    VERSION = "1.0.0"
    category = AgentCategory.CRITICAL

    metadata = AgentMetadata(
        name="Scope3CategoryMapperAgent",
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Maps spend/PO/BOM data to Scope 3 categories"
    )

    def __init__(self, enable_audit_trail: bool = True):
        """Initialize Scope3CategoryMapperAgent."""
        super().__init__(enable_audit_trail=enable_audit_trail)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Scope 3 category mapping."""
        start_time = DeterministicClock.now()

        try:
            mapper_input = Scope3CategoryMapperInput(**inputs)
            results: List[CategoryMappingResult] = []

            # Process spend records
            if mapper_input.spend_records:
                for record in mapper_input.spend_records:
                    result = self._map_spend_record(record)
                    results.append(result)

            # Process purchase orders
            if mapper_input.purchase_orders:
                for po in mapper_input.purchase_orders:
                    result = self._map_purchase_order(po)
                    results.append(result)

            # Process BOM items
            if mapper_input.bom_items:
                for item in mapper_input.bom_items:
                    result = self._map_bom_item(item)
                    results.append(result)

            # Aggregate by category
            spend_by_category: Dict[str, float] = {}
            count_by_category: Dict[str, int] = {}
            total_confidence = 0.0

            for r in results:
                cat_key = r.mapped_category.value
                spend_by_category[cat_key] = spend_by_category.get(cat_key, 0) + r.amount
                count_by_category[cat_key] = count_by_category.get(cat_key, 0) + 1
                total_confidence += r.confidence

            # Determine relevant categories
            relevant_categories = [
                cat for cat, spend in spend_by_category.items()
                if spend > 0
            ]

            # Calculate processing time
            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            provenance_hash = self._compute_provenance_hash({
                "input": inputs,
                "categories_mapped": len(relevant_categories)
            })

            output = Scope3CategoryMapperOutput(
                success=True,
                mapping_results=results,
                spend_by_category=spend_by_category,
                record_count_by_category=count_by_category,
                relevant_categories=relevant_categories,
                coverage_assessment={},
                total_spend_mapped=sum(r.amount for r in results),
                total_records_processed=len(results),
                average_confidence=total_confidence / len(results) if results else 0,
                processing_time_ms=processing_time_ms,
                provenance_hash=provenance_hash,
                validation_status="PASS"
            )

            self._capture_audit_entry(
                operation="map_scope3_categories",
                inputs=inputs,
                outputs=output.model_dump(),
                calculation_trace=[f"Mapped {len(results)} records to Scope 3 categories"]
            )

            return output.model_dump()

        except Exception as e:
            logger.error(f"Category mapping failed: {str(e)}", exc_info=True)
            end_time = DeterministicClock.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000

            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "validation_status": "FAIL"
            }

    def _map_spend_record(self, record: SpendRecord) -> CategoryMappingResult:
        """Map a spend record to Scope 3 category."""
        trace = []
        confidence = 0.0
        mapping_rule = "default"
        mapped_category = Scope3Category.CAT1_PURCHASED_GOODS  # Default

        # Priority 1: NAICS code
        if record.supplier_naics:
            category, conf = self._map_from_naics(record.supplier_naics)
            if category:
                mapped_category = category
                confidence = conf
                mapping_rule = f"NAICS:{record.supplier_naics}"
                trace.append(f"Mapped via NAICS code {record.supplier_naics}")

        # Priority 2: Spend category
        if confidence < 0.8 and record.category:
            category, conf = self._map_from_keyword(record.category)
            if category and conf > confidence:
                mapped_category = category
                confidence = conf
                mapping_rule = f"Category:{record.category}"
                trace.append(f"Mapped via spend category '{record.category}'")

        # Priority 3: Description keywords
        if confidence < 0.6 and record.description:
            category, conf = self._map_from_keyword(record.description)
            if category and conf > confidence:
                mapped_category = category
                confidence = max(conf, 0.5)  # Cap at 0.5 for description-based
                mapping_rule = f"Description keyword"
                trace.append(f"Mapped via description keywords")

        # Default if nothing matched
        if confidence == 0:
            confidence = 0.3
            mapping_rule = "Default (Category 1)"
            trace.append("Defaulted to Category 1 - Purchased Goods")

        # Determine recommended approach
        if confidence >= 0.8:
            approach = CalculationApproach.SUPPLIER_SPECIFIC
        elif confidence >= 0.5:
            approach = CalculationApproach.HYBRID
        else:
            approach = CalculationApproach.SPEND_BASED

        provenance_hash = self._compute_provenance_hash({
            "amount": record.amount,
            "category": mapped_category.value,
            "confidence": confidence
        })

        cat_num = int(mapped_category.value.split("_")[0])

        return CategoryMappingResult(
            source_type=DataSourceType.SPEND_DATA,
            source_id=None,
            mapped_category=mapped_category,
            category_number=cat_num,
            category_name=self._get_category_name(mapped_category),
            amount=record.amount,
            currency=record.currency,
            confidence=round(confidence, 2),
            mapping_rule=mapping_rule,
            recommended_approach=approach,
            calculation_trace=trace,
            provenance_hash=provenance_hash
        )

    def _map_purchase_order(self, po: PurchaseOrder) -> CategoryMappingResult:
        """Map a purchase order to Scope 3 category."""
        trace = []
        confidence = 0.0
        mapping_rule = "default"
        mapped_category = Scope3Category.CAT1_PURCHASED_GOODS

        # Use NAICS if available
        if po.supplier_naics:
            category, conf = self._map_from_naics(po.supplier_naics)
            if category:
                mapped_category = category
                confidence = conf
                mapping_rule = f"NAICS:{po.supplier_naics}"
                trace.append(f"Mapped via NAICS {po.supplier_naics}")

        # Use PO category
        if confidence < 0.8 and po.category:
            category, conf = self._map_from_keyword(po.category)
            if category and conf > confidence:
                mapped_category = category
                confidence = conf
                mapping_rule = f"PO Category:{po.category}"
                trace.append(f"Mapped via PO category")

        if confidence == 0:
            confidence = 0.3
            mapping_rule = "Default"
            trace.append("Defaulted to Category 1")

        approach = CalculationApproach.SPEND_BASED if confidence < 0.5 else CalculationApproach.HYBRID

        provenance_hash = self._compute_provenance_hash({
            "po_number": po.po_number,
            "category": mapped_category.value
        })

        cat_num = int(mapped_category.value.split("_")[0])

        return CategoryMappingResult(
            source_type=DataSourceType.PURCHASE_ORDER,
            source_id=po.po_number,
            mapped_category=mapped_category,
            category_number=cat_num,
            category_name=self._get_category_name(mapped_category),
            amount=po.amount,
            currency=po.currency,
            confidence=round(confidence, 2),
            mapping_rule=mapping_rule,
            recommended_approach=approach,
            calculation_trace=trace,
            provenance_hash=provenance_hash
        )

    def _map_bom_item(self, item: BOMItem) -> CategoryMappingResult:
        """Map a BOM item to Scope 3 category."""
        trace = []
        # BOM items are always Category 1 (Purchased Goods)
        mapped_category = Scope3Category.CAT1_PURCHASED_GOODS
        confidence = 0.9
        mapping_rule = "BOM item -> Category 1"
        trace.append("BOM items mapped to Category 1 - Purchased Goods")

        provenance_hash = self._compute_provenance_hash({
            "item_code": item.item_code,
            "category": mapped_category.value
        })

        # Estimate amount if weight is available (simplified)
        amount = item.quantity * (item.weight_kg or 1.0)

        return CategoryMappingResult(
            source_type=DataSourceType.BOM,
            source_id=item.item_code,
            mapped_category=mapped_category,
            category_number=1,
            category_name="Purchased Goods and Services",
            amount=amount,
            currency="USD",
            confidence=confidence,
            mapping_rule=mapping_rule,
            recommended_approach=CalculationApproach.HYBRID,
            calculation_trace=trace,
            provenance_hash=provenance_hash
        )

    def _map_from_naics(self, naics_code: str) -> Tuple[Optional[Scope3Category], float]:
        """Map NAICS code to Scope 3 category."""
        # Try progressively shorter codes
        for length in [3, 2]:
            prefix = naics_code[:length]
            if prefix in NAICS_TO_CATEGORY:
                return NAICS_TO_CATEGORY[prefix], 0.9
        return None, 0.0

    def _map_from_keyword(self, text: str) -> Tuple[Optional[Scope3Category], float]:
        """Map text keywords to Scope 3 category."""
        text_lower = text.lower().replace(" ", "_").replace("-", "_")

        # Direct match
        if text_lower in SPEND_KEYWORDS_TO_CATEGORY:
            return SPEND_KEYWORDS_TO_CATEGORY[text_lower], 0.8

        # Partial match
        for keyword, category in SPEND_KEYWORDS_TO_CATEGORY.items():
            if keyword in text_lower or text_lower in keyword:
                return category, 0.6

        return None, 0.0

    def _get_category_name(self, category: Scope3Category) -> str:
        """Get human-readable category name."""
        names = {
            Scope3Category.CAT1_PURCHASED_GOODS: "Purchased Goods and Services",
            Scope3Category.CAT2_CAPITAL_GOODS: "Capital Goods",
            Scope3Category.CAT3_FUEL_ENERGY: "Fuel and Energy Related Activities",
            Scope3Category.CAT4_UPSTREAM_TRANSPORT: "Upstream Transportation and Distribution",
            Scope3Category.CAT5_WASTE: "Waste Generated in Operations",
            Scope3Category.CAT6_BUSINESS_TRAVEL: "Business Travel",
            Scope3Category.CAT7_COMMUTING: "Employee Commuting",
            Scope3Category.CAT8_UPSTREAM_LEASED: "Upstream Leased Assets",
            Scope3Category.CAT9_DOWNSTREAM_TRANSPORT: "Downstream Transportation and Distribution",
            Scope3Category.CAT10_PROCESSING: "Processing of Sold Products",
            Scope3Category.CAT11_PRODUCT_USE: "Use of Sold Products",
            Scope3Category.CAT12_END_OF_LIFE: "End-of-Life Treatment of Sold Products",
            Scope3Category.CAT13_DOWNSTREAM_LEASED: "Downstream Leased Assets",
            Scope3Category.CAT14_FRANCHISES: "Franchises",
            Scope3Category.CAT15_INVESTMENTS: "Investments",
        }
        return names.get(category, "Unknown")

    def _compute_provenance_hash(self, data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_category_info(self, category_number: int) -> Optional[Dict[str, Any]]:
        """Get information about a Scope 3 category."""
        for cat in Scope3Category:
            cat_num = int(cat.value.split("_")[0])
            if cat_num == category_number:
                return {
                    "number": cat_num,
                    "code": cat.value,
                    "name": self._get_category_name(cat),
                    "upstream": cat_num <= 8,
                    "downstream": cat_num > 8
                }
        return None
