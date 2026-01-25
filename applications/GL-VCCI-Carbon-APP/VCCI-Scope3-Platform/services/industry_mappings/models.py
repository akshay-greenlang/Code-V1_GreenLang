# -*- coding: utf-8 -*-
"""
Data Models for Industry Mappings

Comprehensive Pydantic models for industry classification codes, product mappings,
and taxonomy entries with full validation.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field, validator, root_validator
import re


class IndustryCategory(str, Enum):
    """Industry category types"""
    MANUFACTURING = "manufacturing"
    SERVICES = "services"
    AGRICULTURE = "agriculture"
    MINING = "mining"
    CONSTRUCTION = "construction"
    UTILITIES = "utilities"
    TRANSPORTATION = "transportation"
    WHOLESALE = "wholesale"
    RETAIL = "retail"
    INFORMATION = "information"
    FINANCE = "finance"
    REAL_ESTATE = "real_estate"
    PROFESSIONAL = "professional"
    EDUCATION = "education"
    HEALTHCARE = "healthcare"
    HOSPITALITY = "hospitality"
    OTHER = "other"


class MappingStrategy(str, Enum):
    """Mapping strategy types"""
    EXACT_CODE = "exact_code"
    KEYWORD_SEARCH = "keyword_search"
    FUZZY_MATCH = "fuzzy_match"
    ML_CLASSIFICATION = "ml_classification"
    HIERARCHICAL = "hierarchical"
    CROSSWALK = "crosswalk"
    MANUAL = "manual"


class ConfidenceLevel(str, Enum):
    """Confidence level for mappings"""
    HIGH = "high"  # 90-100%
    MEDIUM = "medium"  # 70-89%
    LOW = "low"  # 50-69%
    VERY_LOW = "very_low"  # <50%


class CodeHierarchy(BaseModel):
    """Hierarchical code structure"""
    level_1: Optional[str] = Field(None, description="Top level (sector)")
    level_2: Optional[str] = Field(None, description="Second level (subsector)")
    level_3: Optional[str] = Field(None, description="Third level (group)")
    level_4: Optional[str] = Field(None, description="Fourth level (class)")
    level_5: Optional[str] = Field(None, description="Fifth level (subclass)")
    level_6: Optional[str] = Field(None, description="Sixth level (detail)")

    class Config:
        json_schema_extra = {
            "example": {
                "level_1": "31",
                "level_2": "311",
                "level_3": "3111",
                "level_4": "31111",
                "level_5": "311111"
            }
        }


class NAICSCode(BaseModel):
    """NAICS 2022 Code Model"""
    code: str = Field(..., description="NAICS code (2-6 digits)")
    title: str = Field(..., description="Official NAICS title")
    description: str = Field(..., description="Detailed description")
    level: int = Field(..., ge=1, le=6, description="Hierarchy level (2-6 digits)")
    parent_code: Optional[str] = Field(None, description="Parent code in hierarchy")
    category: IndustryCategory = Field(..., description="Industry category")
    keywords: List[str] = Field(default_factory=list, description="Search keywords")
    examples: List[str] = Field(default_factory=list, description="Example activities")
    exclusions: List[str] = Field(default_factory=list, description="Excluded activities")
    cross_references: List[str] = Field(default_factory=list, description="Related codes")
    year: int = Field(default=2022, description="NAICS version year")
    active: bool = Field(default=True, description="Code is active")

    @validator("code")
    def validate_code(cls, v):
        """Validate NAICS code format"""
        if not re.match(r"^\d{2,6}$", v):
            raise ValueError("NAICS code must be 2-6 digits")
        return v

    @validator("level")
    def validate_level(cls, v, values):
        """Validate level matches code length"""
        if "code" in values:
            code_length = len(values["code"])
            if v != code_length:
                raise ValueError(f"Level {v} doesn't match code length {code_length}")
        return v

    def get_hierarchy(self) -> CodeHierarchy:
        """Get hierarchical breakdown"""
        hierarchy = CodeHierarchy()
        if len(self.code) >= 2:
            hierarchy.level_1 = self.code[:2]
        if len(self.code) >= 3:
            hierarchy.level_2 = self.code[:3]
        if len(self.code) >= 4:
            hierarchy.level_3 = self.code[:4]
        if len(self.code) >= 5:
            hierarchy.level_4 = self.code[:5]
        if len(self.code) >= 6:
            hierarchy.level_5 = self.code[:6]
        return hierarchy

    class Config:
        json_schema_extra = {
            "example": {
                "code": "311111",
                "title": "Dog and Cat Food Manufacturing",
                "description": "Manufacturing pet food from ingredients",
                "level": 6,
                "parent_code": "31111",
                "category": "manufacturing",
                "keywords": ["pet food", "dog food", "cat food"],
                "examples": ["Dry dog food", "Wet cat food"],
                "year": 2022,
                "active": True
            }
        }


class ISICCode(BaseModel):
    """ISIC Rev 4 Code Model"""
    code: str = Field(..., description="ISIC code (1-4 characters)")
    title: str = Field(..., description="Official ISIC title")
    description: str = Field(..., description="Detailed description")
    section: str = Field(..., description="Section (A-U)")
    division: str = Field(..., description="Division (2 digits)")
    group: Optional[str] = Field(None, description="Group (3 digits)")
    class_code: Optional[str] = Field(None, description="Class (4 digits)")
    level: int = Field(..., ge=1, le=4, description="Hierarchy level")
    parent_code: Optional[str] = Field(None, description="Parent code")
    category: IndustryCategory = Field(..., description="Industry category")
    keywords: List[str] = Field(default_factory=list, description="Search keywords")
    examples: List[str] = Field(default_factory=list, description="Example activities")
    naics_equivalents: List[str] = Field(default_factory=list, description="NAICS equivalents")
    regional_notes: Dict[str, str] = Field(default_factory=dict, description="Regional variations")
    revision: str = Field(default="Rev 4", description="ISIC revision")
    active: bool = Field(default=True, description="Code is active")

    @validator("code")
    def validate_code(cls, v):
        """Validate ISIC code format"""
        if not re.match(r"^[A-U]\d{0,3}$", v):
            raise ValueError("ISIC code must be section letter + 0-3 digits")
        return v

    @validator("section")
    def validate_section(cls, v):
        """Validate section letter"""
        if not re.match(r"^[A-U]$", v):
            raise ValueError("Section must be A-U")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "code": "C1080",
                "title": "Manufacture of prepared animal feeds",
                "description": "Manufacture of prepared feeds for pets and farm animals",
                "section": "C",
                "division": "10",
                "group": "108",
                "class_code": "1080",
                "level": 4,
                "category": "manufacturing",
                "naics_equivalents": ["311111", "311119"],
                "revision": "Rev 4"
            }
        }


class TaxonomyEntry(BaseModel):
    """Custom Product Taxonomy Entry"""
    id: str = Field(..., description="Unique taxonomy ID")
    name: str = Field(..., description="Product/service name")
    category: str = Field(..., description="Top-level category")
    subcategory: Optional[str] = Field(None, description="Subcategory")
    material_type: Optional[str] = Field(None, description="Material type")
    unit: str = Field(..., description="Standard unit (kg, m3, kWh, etc.)")
    naics_codes: List[str] = Field(default_factory=list, description="Related NAICS codes")
    isic_codes: List[str] = Field(default_factory=list, description="Related ISIC codes")
    emission_factor_id: Optional[str] = Field(None, description="Linked emission factor ID")
    keywords: List[str] = Field(default_factory=list, description="Search keywords")
    synonyms: List[str] = Field(default_factory=list, description="Alternative names")
    regional_variations: Dict[str, str] = Field(default_factory=dict, description="Regional names")
    specifications: Dict[str, str] = Field(default_factory=dict, description="Product specs")
    typical_uses: List[str] = Field(default_factory=list, description="Common uses")
    data_quality: str = Field(default="medium", description="Data quality rating")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    active: bool = Field(default=True, description="Entry is active")

    @validator("unit")
    def validate_unit(cls, v):
        """Validate unit format"""
        valid_units = [
            "kg", "g", "t", "lb", "oz",  # Mass
            "m3", "l", "ml", "gal",  # Volume
            "kWh", "MWh", "GJ", "MJ",  # Energy
            "km", "mi", "m",  # Distance
            "item", "unit", "piece",  # Count
            "hour", "day", "month"  # Time
        ]
        if v not in valid_units:
            raise ValueError(f"Unit must be one of {valid_units}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "id": "STEEL_REBAR_001",
                "name": "Steel Reinforcement Bar",
                "category": "Construction Materials",
                "subcategory": "Steel Products",
                "material_type": "Carbon Steel",
                "unit": "kg",
                "naics_codes": ["331110"],
                "isic_codes": ["C2410"],
                "keywords": ["rebar", "reinforcement", "steel bar"],
                "synonyms": ["reinforcing bar", "rebar steel"]
            }
        }


class MappingResult(BaseModel):
    """Result of an industry mapping operation"""
    input_text: str = Field(..., description="Original input text")
    matched: bool = Field(..., description="Whether a match was found")
    naics_code: Optional[str] = Field(None, description="Matched NAICS code")
    isic_code: Optional[str] = Field(None, description="Matched ISIC code")
    taxonomy_id: Optional[str] = Field(None, description="Matched taxonomy ID")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence (0-1)")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence level")
    strategy_used: MappingStrategy = Field(..., description="Strategy that produced match")
    alternative_matches: List[Dict[str, Union[str, float]]] = Field(
        default_factory=list,
        description="Alternative matches with scores"
    )
    matched_title: Optional[str] = Field(None, description="Title of matched code")
    category: Optional[IndustryCategory] = Field(None, description="Industry category")
    keywords_matched: List[str] = Field(default_factory=list, description="Matched keywords")
    processing_time_ms: float = Field(..., description="Processing time in ms")
    warnings: List[str] = Field(default_factory=list, description="Mapping warnings")
    metadata: Dict[str, any] = Field(default_factory=dict, description="Additional metadata")

    @validator("confidence_level", always=True)
    def set_confidence_level(cls, v, values):
        """Auto-set confidence level from score"""
        if "confidence_score" in values:
            score = values["confidence_score"]
            if score >= 0.9:
                return ConfidenceLevel.HIGH
            elif score >= 0.7:
                return ConfidenceLevel.MEDIUM
            elif score >= 0.5:
                return ConfidenceLevel.LOW
            else:
                return ConfidenceLevel.VERY_LOW
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "input_text": "steel rebar manufacturing",
                "matched": True,
                "naics_code": "331110",
                "confidence_score": 0.95,
                "confidence_level": "high",
                "strategy_used": "keyword_search",
                "matched_title": "Iron and Steel Mills",
                "processing_time_ms": 8.5
            }
        }


class ProductMapping(BaseModel):
    """Complete product to industry code mapping"""
    product_name: str = Field(..., description="Product or service name")
    product_description: Optional[str] = Field(None, description="Detailed description")
    taxonomy_entry: Optional[TaxonomyEntry] = Field(None, description="Custom taxonomy match")
    naics_mapping: Optional[NAICSCode] = Field(None, description="NAICS code mapping")
    isic_mapping: Optional[ISICCode] = Field(None, description="ISIC code mapping")
    mapping_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    validation_status: str = Field(..., description="Validation status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(default="system", description="Creator")
    notes: Optional[str] = Field(None, description="Additional notes")

    class Config:
        json_schema_extra = {
            "example": {
                "product_name": "Steel Rebar",
                "product_description": "Construction reinforcement bars",
                "mapping_confidence": 0.95,
                "validation_status": "validated",
                "created_by": "mapper_engine"
            }
        }


class ValidationResult(BaseModel):
    """Result of mapping validation"""
    valid: bool = Field(..., description="Whether mapping is valid")
    code: Optional[str] = Field(None, description="Code being validated")
    code_type: str = Field(..., description="NAICS, ISIC, or CUSTOM")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    coverage_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Coverage score")
    quality_metrics: Dict[str, float] = Field(default_factory=dict, description="Quality metrics")
    validated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "valid": True,
                "code": "331110",
                "code_type": "NAICS",
                "errors": [],
                "warnings": ["Consider adding more keywords"],
                "coverage_score": 0.92
            }
        }


class CoverageAnalysis(BaseModel):
    """Coverage analysis for industry mappings"""
    total_products: int = Field(..., description="Total products analyzed")
    mapped_products: int = Field(..., description="Successfully mapped products")
    coverage_percentage: float = Field(..., ge=0.0, le=100.0, description="Coverage %")
    high_confidence_count: int = Field(..., description="High confidence mappings")
    medium_confidence_count: int = Field(..., description="Medium confidence mappings")
    low_confidence_count: int = Field(..., description="Low confidence mappings")
    unmapped_products: List[str] = Field(default_factory=list, description="Unmapped products")
    category_coverage: Dict[str, float] = Field(default_factory=dict, description="By category")
    strategy_distribution: Dict[str, int] = Field(default_factory=dict, description="By strategy")
    average_confidence: float = Field(..., ge=0.0, le=1.0, description="Average confidence")
    analysis_date: datetime = Field(default_factory=datetime.utcnow)

    @validator("coverage_percentage", always=True)
    def calculate_coverage(cls, v, values):
        """Calculate coverage percentage"""
        if "total_products" in values and "mapped_products" in values:
            total = values["total_products"]
            mapped = values["mapped_products"]
            if total > 0:
                return (mapped / total) * 100
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "total_products": 1000,
                "mapped_products": 950,
                "coverage_percentage": 95.0,
                "high_confidence_count": 850,
                "medium_confidence_count": 80,
                "low_confidence_count": 20,
                "average_confidence": 0.91
            }
        }
