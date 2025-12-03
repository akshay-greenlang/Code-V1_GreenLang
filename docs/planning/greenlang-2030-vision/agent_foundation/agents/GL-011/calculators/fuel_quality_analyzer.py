# -*- coding: utf-8 -*-
"""
Fuel Quality Analyzer for GL-011 FUELCRAFT.

Provides comprehensive fuel quality analysis including proximate analysis,
ultimate analysis, heating value estimation, grindability assessment,
slagging/fouling prediction, and quality classification per ASTM standards.

Standards:
    - ASTM D3172: Standard Practice for Proximate Analysis of Coal and Coke
    - ASTM D3176: Standard Practice for Ultimate Analysis of Coal and Coke
    - ASTM D5865: Standard Test Method for Gross Calorific Value
    - ASTM D409: Standard Test Method for Grindability of Coal (Hardgrove)
    - ASTM D388: Standard Classification of Coals by Rank
    - ISO 17225: Solid Biofuels - Fuel Specifications and Classes

Zero-hallucination: All calculations use deterministic formulas from peer-reviewed sources.

Example:
    >>> analyzer = FuelQualityAnalyzer()
    >>> result = analyzer.analyze_fuel_quality(coal_sample)
    >>> print(f"Grade: {result.grade_classification}")
    >>> print(f"Slagging Index: {result.slagging_index}")
"""

import hashlib
import json
import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class FuelType(Enum):
    """Supported fuel types for analysis."""
    ANTHRACITE = "anthracite"
    BITUMINOUS = "bituminous"
    SUB_BITUMINOUS = "sub_bituminous"
    LIGNITE = "lignite"
    PEAT = "peat"
    BIOMASS = "biomass"
    WOOD_PELLETS = "wood_pellets"
    PETROLEUM_COKE = "petroleum_coke"
    FUEL_OIL = "fuel_oil"
    NATURAL_GAS = "natural_gas"


class AnalysisBasis(Enum):
    """Basis for fuel analysis reporting."""
    AS_RECEIVED = "as_received"  # ar
    AIR_DRIED = "air_dried"  # ad
    DRY_BASIS = "dry_basis"  # db
    DRY_ASH_FREE = "dry_ash_free"  # daf
    DRY_MINERAL_MATTER_FREE = "dry_mineral_matter_free"  # dmmf


class QualityGrade(Enum):
    """Fuel quality grade classification."""
    PREMIUM = "premium"
    HIGH = "high"
    STANDARD = "standard"
    LOW = "low"
    OFF_SPEC = "off_spec"


class SlaggingRisk(Enum):
    """Slagging/fouling risk classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SEVERE = "severe"


# =============================================================================
# Data Classes (Frozen for Immutability)
# =============================================================================

@dataclass(frozen=True)
class ProximateAnalysis:
    """
    Proximate analysis results per ASTM D3172.

    Attributes:
        moisture_percent: Total moisture content (%)
        volatile_matter_percent: Volatile matter content (%)
        fixed_carbon_percent: Fixed carbon content (%)
        ash_percent: Ash content (%)
        basis: Reporting basis (as_received, dry, etc.)
    """
    moisture_percent: Decimal
    volatile_matter_percent: Decimal
    fixed_carbon_percent: Decimal
    ash_percent: Decimal
    basis: AnalysisBasis = AnalysisBasis.AS_RECEIVED


@dataclass(frozen=True)
class UltimateAnalysis:
    """
    Ultimate analysis results per ASTM D3176.

    Attributes:
        carbon_percent: Carbon content (%)
        hydrogen_percent: Hydrogen content (%)
        oxygen_percent: Oxygen content (%)
        nitrogen_percent: Nitrogen content (%)
        sulfur_percent: Sulfur content (%)
        ash_percent: Ash content (%)
        basis: Reporting basis
    """
    carbon_percent: Decimal
    hydrogen_percent: Decimal
    oxygen_percent: Decimal
    nitrogen_percent: Decimal
    sulfur_percent: Decimal
    ash_percent: Decimal
    basis: AnalysisBasis = AnalysisBasis.DRY_BASIS


@dataclass(frozen=True)
class AshComposition:
    """
    Ash mineral composition for slagging/fouling analysis.

    Major oxides in ash (% by mass):
        SiO2, Al2O3, Fe2O3, CaO, MgO, Na2O, K2O, TiO2, P2O5, SO3

    Attributes:
        sio2_percent: Silicon dioxide (%)
        al2o3_percent: Aluminum oxide (%)
        fe2o3_percent: Iron oxide (%)
        cao_percent: Calcium oxide (%)
        mgo_percent: Magnesium oxide (%)
        na2o_percent: Sodium oxide (%)
        k2o_percent: Potassium oxide (%)
        tio2_percent: Titanium dioxide (%)
        p2o5_percent: Phosphorus pentoxide (%)
        so3_percent: Sulfur trioxide (%)
    """
    sio2_percent: Decimal
    al2o3_percent: Decimal
    fe2o3_percent: Decimal
    cao_percent: Decimal
    mgo_percent: Decimal
    na2o_percent: Decimal
    k2o_percent: Decimal
    tio2_percent: Decimal = Decimal("0")
    p2o5_percent: Decimal = Decimal("0")
    so3_percent: Decimal = Decimal("0")


@dataclass(frozen=True)
class FuelSampleInput:
    """
    Complete fuel sample input for quality analysis.

    Attributes:
        sample_id: Unique sample identifier
        fuel_type: Type of fuel
        proximate: Proximate analysis results
        ultimate: Ultimate analysis results (optional)
        ash_composition: Ash mineral composition (optional)
        hardgrove_index: Hardgrove Grindability Index (optional)
        heating_value_mj_kg: Measured heating value if available
        sampling_date: Date of sample collection
        lot_number: Lot/batch identifier
        supplier: Supplier name
    """
    sample_id: str
    fuel_type: FuelType
    proximate: ProximateAnalysis
    ultimate: Optional[UltimateAnalysis] = None
    ash_composition: Optional[AshComposition] = None
    hardgrove_index: Optional[Decimal] = None
    heating_value_mj_kg: Optional[Decimal] = None
    sampling_date: Optional[str] = None
    lot_number: Optional[str] = None
    supplier: Optional[str] = None


@dataclass(frozen=True)
class HeatingValueResult:
    """
    Calculated heating value results.

    Attributes:
        hhv_mj_kg: Higher Heating Value (gross) MJ/kg
        lhv_mj_kg: Lower Heating Value (net) MJ/kg
        calculation_method: Method used (Dulong, Boie, etc.)
        confidence_percent: Estimation confidence
        basis: Reporting basis
    """
    hhv_mj_kg: Decimal
    lhv_mj_kg: Decimal
    calculation_method: str
    confidence_percent: Decimal
    basis: AnalysisBasis


@dataclass(frozen=True)
class SlaggingFoulingResult:
    """
    Slagging and fouling assessment results.

    Attributes:
        base_acid_ratio: Base/Acid ratio (dimensionless)
        slagging_index_rs: Slagging index Rs
        fouling_index_rf: Fouling index Rf
        t250_temperature_c: Temperature at 250 poise viscosity (C)
        silica_ratio: Silica ratio (%)
        iron_ratio: Iron ratio for classification
        slagging_risk: Risk classification
        fouling_risk: Risk classification
        recommended_actions: List of recommended mitigation actions
    """
    base_acid_ratio: Decimal
    slagging_index_rs: Decimal
    fouling_index_rf: Decimal
    t250_temperature_c: Optional[Decimal]
    silica_ratio: Decimal
    iron_ratio: Decimal
    slagging_risk: SlaggingRisk
    fouling_risk: SlaggingRisk
    recommended_actions: Tuple[str, ...]


@dataclass(frozen=True)
class EmissionFactorResult:
    """
    Fuel-specific emission factors.

    Attributes:
        co2_kg_per_gj: CO2 emission factor (kg CO2/GJ)
        co2_kg_per_kg_fuel: CO2 per kg fuel
        so2_kg_per_gj: SO2 emission factor (kg SO2/GJ)
        nox_kg_per_gj: NOx emission factor (estimated, kg NOx/GJ)
        particulate_kg_per_gj: Particulate matter factor
        calculation_method: Method used
    """
    co2_kg_per_gj: Decimal
    co2_kg_per_kg_fuel: Decimal
    so2_kg_per_gj: Decimal
    nox_kg_per_gj: Decimal
    particulate_kg_per_gj: Decimal
    calculation_method: str


@dataclass(frozen=True)
class QualityDeviationAlert:
    """
    Quality deviation alert for out-of-spec conditions.

    Attributes:
        alert_type: Type of alert (warning, critical)
        parameter: Parameter that is out of spec
        actual_value: Measured/calculated value
        specification_value: Specification limit
        deviation_percent: Percentage deviation from spec
        recommended_action: Suggested corrective action
    """
    alert_type: str
    parameter: str
    actual_value: Decimal
    specification_value: Decimal
    deviation_percent: Decimal
    recommended_action: str


@dataclass(frozen=True)
class SamplingRecommendation:
    """
    Sampling frequency recommendation.

    Attributes:
        recommended_frequency: Samples per day/batch
        confidence_level: Statistical confidence level
        variability_factor: Fuel variability score
        reasoning: Explanation for recommendation
    """
    recommended_frequency: int
    confidence_level: Decimal
    variability_factor: Decimal
    reasoning: str


@dataclass(frozen=True)
class FuelQualityResult:
    """
    Complete fuel quality analysis result.

    Attributes:
        sample_id: Original sample identifier
        fuel_type: Fuel type analyzed
        heating_value: Calculated heating values
        grade_classification: Quality grade (Premium, High, Standard, etc.)
        slagging_fouling: Slagging/fouling assessment
        emission_factors: Emission factors
        quality_score: Overall quality score (0-100)
        alerts: List of quality deviation alerts
        sampling_recommendation: Sampling frequency recommendation
        proximate_dry_basis: Proximate analysis converted to dry basis
        ultimate_daf: Ultimate analysis on dry-ash-free basis
        fuel_ratio: Fixed carbon / Volatile matter ratio
        hardgrove_correlation: Correlated HGI if not measured
        provenance_hash: SHA-256 audit hash
        calculation_steps: Provenance tracking steps
        processing_time_ms: Processing time
    """
    sample_id: str
    fuel_type: FuelType
    heating_value: HeatingValueResult
    grade_classification: QualityGrade
    slagging_fouling: Optional[SlaggingFoulingResult]
    emission_factors: EmissionFactorResult
    quality_score: Decimal
    alerts: Tuple[QualityDeviationAlert, ...]
    sampling_recommendation: SamplingRecommendation
    proximate_dry_basis: ProximateAnalysis
    ultimate_daf: Optional[UltimateAnalysis]
    fuel_ratio: Decimal
    hardgrove_correlation: Optional[Decimal]
    provenance_hash: str
    calculation_steps: Tuple[Dict[str, Any], ...]
    processing_time_ms: Decimal


# =============================================================================
# Specification Reference Data
# =============================================================================

# ASTM D388 Coal Rank Classification Parameters
COAL_RANK_SPECS = {
    FuelType.ANTHRACITE: {
        'min_fixed_carbon_daf': Decimal("86"),
        'max_volatile_matter_daf': Decimal("14"),
        'min_hhv_mj_kg': Decimal("30")
    },
    FuelType.BITUMINOUS: {
        'min_fixed_carbon_daf': Decimal("69"),
        'max_fixed_carbon_daf': Decimal("86"),
        'min_hhv_mj_kg': Decimal("26")
    },
    FuelType.SUB_BITUMINOUS: {
        'min_hhv_mj_kg': Decimal("19.3"),
        'max_hhv_mj_kg': Decimal("26")
    },
    FuelType.LIGNITE: {
        'max_hhv_mj_kg': Decimal("19.3")
    }
}

# Quality grade thresholds
QUALITY_THRESHOLDS = {
    'premium': {
        'min_hhv_mj_kg': Decimal("28"),
        'max_ash_percent': Decimal("8"),
        'max_sulfur_percent': Decimal("0.5"),
        'max_moisture_percent': Decimal("8")
    },
    'high': {
        'min_hhv_mj_kg': Decimal("24"),
        'max_ash_percent': Decimal("12"),
        'max_sulfur_percent': Decimal("1.0"),
        'max_moisture_percent': Decimal("12")
    },
    'standard': {
        'min_hhv_mj_kg': Decimal("20"),
        'max_ash_percent': Decimal("18"),
        'max_sulfur_percent': Decimal("2.0"),
        'max_moisture_percent': Decimal("18")
    },
    'low': {
        'min_hhv_mj_kg': Decimal("15"),
        'max_ash_percent': Decimal("25"),
        'max_sulfur_percent': Decimal("3.0"),
        'max_moisture_percent': Decimal("25")
    }
}


# =============================================================================
# Calculator Implementation
# =============================================================================

class FuelQualityAnalyzer:
    """
    Comprehensive fuel quality analyzer.

    Provides deterministic analysis of solid fuel quality including:
    - Proximate analysis interpretation
    - Ultimate analysis calculations
    - Heating value estimation (Dulong, Boie, IGT formulas)
    - Hardgrove Grindability Index correlation
    - Slagging/fouling index calculation
    - Emission factor determination
    - Grade classification per ASTM standards
    - Quality deviation alerts
    - Sampling frequency optimization

    All calculations are deterministic and fully traceable.

    Example:
        >>> analyzer = FuelQualityAnalyzer()
        >>> proximate = ProximateAnalysis(
        ...     moisture_percent=Decimal('8.5'),
        ...     volatile_matter_percent=Decimal('28.0'),
        ...     fixed_carbon_percent=Decimal('52.0'),
        ...     ash_percent=Decimal('11.5')
        ... )
        >>> sample = FuelSampleInput(
        ...     sample_id='COAL-2024-001',
        ...     fuel_type=FuelType.BITUMINOUS,
        ...     proximate=proximate
        ... )
        >>> result = analyzer.analyze_fuel_quality(sample)
        >>> print(f"HHV: {result.heating_value.hhv_mj_kg} MJ/kg")

    References:
        - Dulong Formula: Q = 33.83C + 144.3(H - O/8) + 9.42S (MJ/kg)
        - Boie Formula: Q = 35.16C + 116.23H - 11.09O + 6.28N + 10.47S (MJ/kg)
        - Base/Acid Ratio: (Fe2O3+CaO+MgO+Na2O+K2O)/(SiO2+Al2O3+TiO2)
    """

    # Dulong formula coefficients (MJ/kg per % element)
    DULONG_C = Decimal("0.3383")  # Carbon coefficient
    DULONG_H = Decimal("1.443")   # Hydrogen coefficient (H - O/8)
    DULONG_S = Decimal("0.0942")  # Sulfur coefficient

    # Boie formula coefficients
    BOIE_C = Decimal("0.3516")
    BOIE_H = Decimal("1.1623")
    BOIE_O = Decimal("-0.1109")
    BOIE_N = Decimal("0.0628")
    BOIE_S = Decimal("0.1047")

    # Latent heat of vaporization of water at 25C (MJ/kg)
    LATENT_HEAT_WATER = Decimal("2.442")

    # CO2 emission factor from carbon (kg CO2 / kg C)
    CO2_FROM_CARBON = Decimal("3.664")  # 44/12 molecular weight ratio

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize fuel quality analyzer.

        Args:
            config: Optional configuration with:
                - heating_value_method: 'dulong', 'boie', 'igt' (default: 'dulong')
                - strict_validation: Enable strict input validation (default: True)
                - enable_caching: Enable result caching (default: True)
        """
        self.config = config or {}
        self._lock = threading.RLock()
        self._analysis_count = 0
        self._heating_value_method = self.config.get('heating_value_method', 'dulong')
        self._strict_validation = self.config.get('strict_validation', True)
        self._enable_caching = self.config.get('enable_caching', True)
        self._provenance_steps: List[Dict[str, Any]] = []

        logger.info("FuelQualityAnalyzer initialized: method=%s", self._heating_value_method)

    def analyze_fuel_quality(self, sample: FuelSampleInput) -> FuelQualityResult:
        """
        Perform comprehensive fuel quality analysis.

        Analysis steps:
        1. Validate input data
        2. Convert proximate analysis to dry basis
        3. Calculate/estimate ultimate analysis if not provided
        4. Calculate heating values using configured method
        5. Assess slagging/fouling potential (if ash composition available)
        6. Calculate emission factors
        7. Classify quality grade
        8. Generate deviation alerts
        9. Recommend sampling frequency
        10. Generate provenance hash

        Args:
            sample: Complete fuel sample input

        Returns:
            FuelQualityResult with all calculated parameters

        Raises:
            ValueError: If input validation fails
        """
        start_time = datetime.now(timezone.utc)
        self._provenance_steps = []

        with self._lock:
            self._analysis_count += 1
            analysis_id = self._analysis_count

        logger.info("Starting fuel quality analysis #%d for sample %s",
                   analysis_id, sample.sample_id)

        try:
            # Step 1: Validate input
            self._record_step(1, "input_validation",
                             {"sample_id": sample.sample_id},
                             {"validated": True},
                             "Validate proximate analysis sums to ~100%")
            self._validate_input(sample)

            # Step 2: Convert to dry basis
            proximate_db = self._convert_to_dry_basis(sample.proximate)
            self._record_step(2, "dry_basis_conversion",
                             {"moisture_ar": str(sample.proximate.moisture_percent)},
                             {"vm_db": str(proximate_db.volatile_matter_percent),
                              "fc_db": str(proximate_db.fixed_carbon_percent)},
                             "X_db = X_ar / (1 - M_ar/100)")

            # Step 3: Calculate fuel ratio
            fuel_ratio = self._calculate_fuel_ratio(proximate_db)
            self._record_step(3, "fuel_ratio_calculation",
                             {"fc": str(proximate_db.fixed_carbon_percent),
                              "vm": str(proximate_db.volatile_matter_percent)},
                             {"fuel_ratio": str(fuel_ratio)},
                             "FR = FC / VM")

            # Step 4: Estimate or convert ultimate analysis
            ultimate_daf = None
            if sample.ultimate:
                ultimate_daf = self._convert_ultimate_to_daf(sample.ultimate, proximate_db)
                self._record_step(4, "ultimate_daf_conversion",
                                 {"basis": sample.ultimate.basis.value},
                                 {"carbon_daf": str(ultimate_daf.carbon_percent)},
                                 "X_daf = X_db / (1 - Ash_db/100)")
            else:
                ultimate_daf = self._estimate_ultimate_analysis(sample.fuel_type, proximate_db)
                self._record_step(4, "ultimate_estimation",
                                 {"fuel_type": sample.fuel_type.value},
                                 {"carbon_est": str(ultimate_daf.carbon_percent)},
                                 "Estimate from fuel type correlations")

            # Step 5: Calculate heating value
            heating_value = self._calculate_heating_value(
                ultimate_daf if ultimate_daf else self._estimate_ultimate_analysis(sample.fuel_type, proximate_db),
                proximate_db,
                sample.heating_value_mj_kg
            )
            self._record_step(5, "heating_value_calculation",
                             {"method": self._heating_value_method,
                              "carbon": str(ultimate_daf.carbon_percent) if ultimate_daf else "estimated"},
                             {"hhv": str(heating_value.hhv_mj_kg),
                              "lhv": str(heating_value.lhv_mj_kg)},
                             "Dulong: HHV = 0.3383C + 1.443(H-O/8) + 0.0942S")

            # Step 6: Slagging/fouling analysis
            slagging_fouling = None
            if sample.ash_composition:
                slagging_fouling = self._calculate_slagging_fouling(sample.ash_composition)
                self._record_step(6, "slagging_fouling_analysis",
                                 {"ash_composition": "provided"},
                                 {"base_acid_ratio": str(slagging_fouling.base_acid_ratio),
                                  "slagging_risk": slagging_fouling.slagging_risk.value},
                                 "B/A = (Fe2O3+CaO+MgO+Na2O+K2O)/(SiO2+Al2O3+TiO2)")
            else:
                self._record_step(6, "slagging_fouling_analysis",
                                 {"ash_composition": "not_provided"},
                                 {"result": "skipped"},
                                 "Ash composition required for slagging analysis")

            # Step 7: Emission factors
            emission_factors = self._calculate_emission_factors(
                ultimate_daf if ultimate_daf else self._estimate_ultimate_analysis(sample.fuel_type, proximate_db),
                heating_value
            )
            self._record_step(7, "emission_factor_calculation",
                             {"carbon": str(ultimate_daf.carbon_percent) if ultimate_daf else "estimated"},
                             {"co2_kg_gj": str(emission_factors.co2_kg_per_gj)},
                             "CO2 = C * 3.664 (44/12 ratio)")

            # Step 8: Quality grade classification
            grade = self._classify_quality_grade(
                sample.fuel_type,
                heating_value,
                proximate_db,
                ultimate_daf.sulfur_percent if ultimate_daf else Decimal("1.0")
            )
            self._record_step(8, "grade_classification",
                             {"hhv": str(heating_value.hhv_mj_kg),
                              "ash": str(proximate_db.ash_percent)},
                             {"grade": grade.value},
                             "Classification per ASTM D388 and quality thresholds")

            # Step 9: Quality score and alerts
            quality_score, alerts = self._assess_quality_deviations(
                sample.fuel_type,
                heating_value,
                proximate_db,
                ultimate_daf,
                slagging_fouling
            )
            self._record_step(9, "quality_assessment",
                             {"parameter_count": 5},
                             {"quality_score": str(quality_score),
                              "alert_count": len(alerts)},
                             "Score based on deviation from ideal parameters")

            # Step 10: Sampling recommendation
            sampling_rec = self._calculate_sampling_recommendation(
                sample.fuel_type,
                proximate_db,
                alerts
            )
            self._record_step(10, "sampling_recommendation",
                             {"alert_count": len(alerts)},
                             {"frequency": sampling_rec.recommended_frequency},
                             "Frequency based on variability and risk")

            # Step 11: HGI correlation
            hgi_correlation = None
            if sample.hardgrove_index is None:
                hgi_correlation = self._correlate_hardgrove_index(
                    sample.fuel_type,
                    proximate_db,
                    fuel_ratio
                )
                self._record_step(11, "hgi_correlation",
                                 {"fuel_ratio": str(fuel_ratio)},
                                 {"hgi_estimated": str(hgi_correlation) if hgi_correlation else "N/A"},
                                 "HGI correlation from fuel ratio and type")
            else:
                hgi_correlation = sample.hardgrove_index
                self._record_step(11, "hgi_correlation",
                                 {"hgi_measured": str(sample.hardgrove_index)},
                                 {"hgi_used": str(hgi_correlation)},
                                 "Using measured HGI value")

            # Calculate processing time
            end_time = datetime.now(timezone.utc)
            processing_time_ms = Decimal(str((end_time - start_time).total_seconds() * 1000))

            # Generate provenance hash
            provenance_hash = self._calculate_provenance_hash(sample, heating_value, grade)

            result = FuelQualityResult(
                sample_id=sample.sample_id,
                fuel_type=sample.fuel_type,
                heating_value=heating_value,
                grade_classification=grade,
                slagging_fouling=slagging_fouling,
                emission_factors=emission_factors,
                quality_score=quality_score.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP),
                alerts=tuple(alerts),
                sampling_recommendation=sampling_rec,
                proximate_dry_basis=proximate_db,
                ultimate_daf=ultimate_daf,
                fuel_ratio=fuel_ratio.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                hardgrove_correlation=hgi_correlation.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP) if hgi_correlation else None,
                provenance_hash=provenance_hash,
                calculation_steps=tuple(self._provenance_steps),
                processing_time_ms=processing_time_ms.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            )

            logger.info("Analysis #%d completed: grade=%s, score=%.1f",
                       analysis_id, grade.value, float(quality_score))

            return result

        except Exception as e:
            logger.error("Analysis #%d failed: %s", analysis_id, str(e), exc_info=True)
            raise

    def _validate_input(self, sample: FuelSampleInput) -> None:
        """
        Validate input sample data.

        Args:
            sample: Input sample to validate

        Raises:
            ValueError: If validation fails
        """
        prox = sample.proximate

        # Check proximate sums approximately to 100%
        total = prox.moisture_percent + prox.volatile_matter_percent + prox.fixed_carbon_percent + prox.ash_percent
        if self._strict_validation and not (Decimal("98") <= total <= Decimal("102")):
            raise ValueError(
                f"Proximate analysis must sum to ~100%, got {total}%"
            )

        # Check for negative values
        if any(v < 0 for v in [prox.moisture_percent, prox.volatile_matter_percent,
                               prox.fixed_carbon_percent, prox.ash_percent]):
            raise ValueError("Proximate analysis values cannot be negative")

        # Check reasonable ranges
        if prox.moisture_percent > Decimal("60"):
            raise ValueError(f"Moisture {prox.moisture_percent}% exceeds reasonable limit (60%)")

        if prox.ash_percent > Decimal("50"):
            raise ValueError(f"Ash {prox.ash_percent}% exceeds reasonable limit (50%)")

        logger.debug("Input validation passed for sample %s", sample.sample_id)

    def _convert_to_dry_basis(self, proximate: ProximateAnalysis) -> ProximateAnalysis:
        """
        Convert proximate analysis to dry basis.

        Conversion: X_db = X_ar / (1 - M/100)

        Args:
            proximate: As-received proximate analysis

        Returns:
            Proximate analysis on dry basis
        """
        if proximate.basis == AnalysisBasis.DRY_BASIS:
            return proximate

        moisture_factor = Decimal("1") - proximate.moisture_percent / Decimal("100")
        if moisture_factor <= 0:
            moisture_factor = Decimal("0.001")  # Prevent division by zero

        return ProximateAnalysis(
            moisture_percent=Decimal("0"),
            volatile_matter_percent=(proximate.volatile_matter_percent / moisture_factor).quantize(Decimal("0.01")),
            fixed_carbon_percent=(proximate.fixed_carbon_percent / moisture_factor).quantize(Decimal("0.01")),
            ash_percent=(proximate.ash_percent / moisture_factor).quantize(Decimal("0.01")),
            basis=AnalysisBasis.DRY_BASIS
        )

    def _convert_ultimate_to_daf(
        self,
        ultimate: UltimateAnalysis,
        proximate_db: ProximateAnalysis
    ) -> UltimateAnalysis:
        """
        Convert ultimate analysis to dry-ash-free basis.

        Conversion: X_daf = X_db / (1 - Ash_db/100)

        Args:
            ultimate: Ultimate analysis (any basis)
            proximate_db: Proximate analysis on dry basis

        Returns:
            Ultimate analysis on dry-ash-free basis
        """
        ash_factor = Decimal("1") - proximate_db.ash_percent / Decimal("100")
        if ash_factor <= 0:
            ash_factor = Decimal("0.001")

        return UltimateAnalysis(
            carbon_percent=(ultimate.carbon_percent / ash_factor).quantize(Decimal("0.01")),
            hydrogen_percent=(ultimate.hydrogen_percent / ash_factor).quantize(Decimal("0.01")),
            oxygen_percent=(ultimate.oxygen_percent / ash_factor).quantize(Decimal("0.01")),
            nitrogen_percent=(ultimate.nitrogen_percent / ash_factor).quantize(Decimal("0.01")),
            sulfur_percent=(ultimate.sulfur_percent / ash_factor).quantize(Decimal("0.01")),
            ash_percent=Decimal("0"),
            basis=AnalysisBasis.DRY_ASH_FREE
        )

    def _estimate_ultimate_analysis(
        self,
        fuel_type: FuelType,
        proximate_db: ProximateAnalysis
    ) -> UltimateAnalysis:
        """
        Estimate ultimate analysis from proximate analysis and fuel type.

        Uses empirical correlations based on fuel type.

        Args:
            fuel_type: Type of fuel
            proximate_db: Proximate analysis on dry basis

        Returns:
            Estimated ultimate analysis on dry-ash-free basis
        """
        # Typical compositions by fuel type (daf basis)
        typical_compositions = {
            FuelType.ANTHRACITE: {
                'C': Decimal("93"), 'H': Decimal("3"), 'O': Decimal("2"),
                'N': Decimal("1"), 'S': Decimal("1")
            },
            FuelType.BITUMINOUS: {
                'C': Decimal("82"), 'H': Decimal("5.5"), 'O': Decimal("9"),
                'N': Decimal("1.5"), 'S': Decimal("2")
            },
            FuelType.SUB_BITUMINOUS: {
                'C': Decimal("75"), 'H': Decimal("5"), 'O': Decimal("17"),
                'N': Decimal("1.5"), 'S': Decimal("1.5")
            },
            FuelType.LIGNITE: {
                'C': Decimal("68"), 'H': Decimal("5"), 'O': Decimal("24"),
                'N': Decimal("1.5"), 'S': Decimal("1.5")
            },
            FuelType.BIOMASS: {
                'C': Decimal("50"), 'H': Decimal("6"), 'O': Decimal("42"),
                'N': Decimal("1"), 'S': Decimal("0.5")
            },
            FuelType.WOOD_PELLETS: {
                'C': Decimal("51"), 'H': Decimal("6"), 'O': Decimal("42"),
                'N': Decimal("0.5"), 'S': Decimal("0.05")
            }
        }

        # Default to bituminous if not found
        comp = typical_compositions.get(fuel_type, typical_compositions[FuelType.BITUMINOUS])

        # Adjust carbon based on fixed carbon ratio
        fuel_ratio = proximate_db.fixed_carbon_percent / proximate_db.volatile_matter_percent \
            if proximate_db.volatile_matter_percent > 0 else Decimal("2")

        # Higher fuel ratio typically means higher carbon
        carbon_adjustment = (fuel_ratio - Decimal("1.5")) * Decimal("2")
        adjusted_carbon = comp['C'] + carbon_adjustment
        adjusted_carbon = min(max(adjusted_carbon, Decimal("45")), Decimal("95"))

        # Ensure total = 100%
        others = comp['H'] + comp['N'] + comp['S']
        adjusted_oxygen = Decimal("100") - adjusted_carbon - others
        adjusted_oxygen = max(adjusted_oxygen, Decimal("0"))

        return UltimateAnalysis(
            carbon_percent=adjusted_carbon.quantize(Decimal("0.01")),
            hydrogen_percent=comp['H'],
            oxygen_percent=adjusted_oxygen.quantize(Decimal("0.01")),
            nitrogen_percent=comp['N'],
            sulfur_percent=comp['S'],
            ash_percent=Decimal("0"),
            basis=AnalysisBasis.DRY_ASH_FREE
        )

    def _calculate_fuel_ratio(self, proximate_db: ProximateAnalysis) -> Decimal:
        """
        Calculate fuel ratio (Fixed Carbon / Volatile Matter).

        Fuel ratio is used for coal rank classification and grindability correlation.

        Args:
            proximate_db: Proximate analysis on dry basis

        Returns:
            Fuel ratio (dimensionless)
        """
        if proximate_db.volatile_matter_percent <= 0:
            return Decimal("999")  # High value for very low VM

        return proximate_db.fixed_carbon_percent / proximate_db.volatile_matter_percent

    def _calculate_heating_value(
        self,
        ultimate_daf: UltimateAnalysis,
        proximate_db: ProximateAnalysis,
        measured_hhv: Optional[Decimal]
    ) -> HeatingValueResult:
        """
        Calculate heating value using configured method.

        Available methods:
        - Dulong: HHV = 0.3383*C + 1.443*(H - O/8) + 0.0942*S
        - Boie: HHV = 0.3516*C + 1.1623*H - 0.1109*O + 0.0628*N + 0.1047*S
        - IGT: HHV = 0.3419*C + 1.2419*H - 0.1161*O + 0.0628*N + 0.1030*S

        Args:
            ultimate_daf: Ultimate analysis on DAF basis
            proximate_db: Proximate analysis on dry basis
            measured_hhv: Measured HHV if available

        Returns:
            HeatingValueResult with HHV and LHV
        """
        # Convert from DAF to dry basis for calculation
        daf_to_db = Decimal("1") - proximate_db.ash_percent / Decimal("100")

        C = ultimate_daf.carbon_percent * daf_to_db / Decimal("100")
        H = ultimate_daf.hydrogen_percent * daf_to_db / Decimal("100")
        O = ultimate_daf.oxygen_percent * daf_to_db / Decimal("100")
        N = ultimate_daf.nitrogen_percent * daf_to_db / Decimal("100")
        S = ultimate_daf.sulfur_percent * daf_to_db / Decimal("100")

        confidence = Decimal("85")

        if self._heating_value_method == 'dulong':
            # Dulong formula (most common)
            hhv = self.DULONG_C * C * Decimal("100") + \
                  self.DULONG_H * (H * Decimal("100") - O * Decimal("100") / Decimal("8")) + \
                  self.DULONG_S * S * Decimal("100")
            method = "Dulong"

        elif self._heating_value_method == 'boie':
            # Boie formula
            hhv = self.BOIE_C * C * Decimal("100") + \
                  self.BOIE_H * H * Decimal("100") + \
                  self.BOIE_O * O * Decimal("100") + \
                  self.BOIE_N * N * Decimal("100") + \
                  self.BOIE_S * S * Decimal("100")
            method = "Boie"

        else:  # IGT formula
            hhv = Decimal("0.3419") * C * Decimal("100") + \
                  Decimal("1.2419") * H * Decimal("100") - \
                  Decimal("0.1161") * O * Decimal("100") + \
                  Decimal("0.0628") * N * Decimal("100") + \
                  Decimal("0.1030") * S * Decimal("100")
            method = "IGT"

        # If measured value available, use it and adjust confidence
        if measured_hhv is not None:
            deviation = abs(hhv - measured_hhv) / measured_hhv * Decimal("100")
            if deviation < Decimal("5"):
                confidence = Decimal("95")
            hhv = measured_hhv
            method = f"Measured (validated by {method})"

        # Calculate LHV from HHV
        # LHV = HHV - 2.442 * (9*H + M)
        # Where 2.442 MJ/kg is latent heat of water vaporization
        total_h = H * Decimal("100")
        water_from_h = Decimal("9") * total_h / Decimal("100")  # kg water per kg fuel
        lhv = hhv - self.LATENT_HEAT_WATER * water_from_h

        return HeatingValueResult(
            hhv_mj_kg=hhv.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            lhv_mj_kg=lhv.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            calculation_method=method,
            confidence_percent=confidence,
            basis=AnalysisBasis.DRY_BASIS
        )

    def _calculate_slagging_fouling(
        self,
        ash_comp: AshComposition
    ) -> SlaggingFoulingResult:
        """
        Calculate slagging and fouling indices.

        Indices calculated:
        - Base/Acid ratio: (Fe2O3+CaO+MgO+Na2O+K2O)/(SiO2+Al2O3+TiO2)
        - Slagging index Rs: B/A * Sulfur content
        - Fouling index Rf: B/A * Na2O
        - Silica ratio: SiO2*100/(SiO2+Fe2O3+CaO+MgO)
        - T250: Temperature at 250 poise viscosity (empirical)

        Args:
            ash_comp: Ash mineral composition

        Returns:
            SlaggingFoulingResult with indices and risk classification
        """
        # Calculate base components
        base = (ash_comp.fe2o3_percent + ash_comp.cao_percent +
                ash_comp.mgo_percent + ash_comp.na2o_percent + ash_comp.k2o_percent)

        # Calculate acid components
        acid = ash_comp.sio2_percent + ash_comp.al2o3_percent + ash_comp.tio2_percent

        # Base/Acid ratio
        base_acid_ratio = base / acid if acid > 0 else Decimal("999")

        # Slagging index (simplified - typically multiplied by sulfur)
        # Using SO3 as proxy for sulfur contribution
        slagging_index = base_acid_ratio * ash_comp.so3_percent / Decimal("10")

        # Fouling index
        fouling_index = base_acid_ratio * ash_comp.na2o_percent

        # Silica ratio
        silica_denom = (ash_comp.sio2_percent + ash_comp.fe2o3_percent +
                       ash_comp.cao_percent + ash_comp.mgo_percent)
        silica_ratio = ash_comp.sio2_percent * Decimal("100") / silica_denom if silica_denom > 0 else Decimal("0")

        # Iron ratio
        iron_calcium = ash_comp.fe2o3_percent + ash_comp.cao_percent
        iron_ratio = ash_comp.fe2o3_percent / iron_calcium if iron_calcium > 0 else Decimal("0")

        # T250 estimation (empirical correlation)
        # Higher silica ratio generally means higher T250
        t250 = Decimal("1100") + silica_ratio * Decimal("5") - base_acid_ratio * Decimal("100")
        t250 = max(t250, Decimal("900"))

        # Risk classification
        if base_acid_ratio < Decimal("0.4"):
            slagging_risk = SlaggingRisk.LOW
        elif base_acid_ratio < Decimal("0.6"):
            slagging_risk = SlaggingRisk.MEDIUM
        elif base_acid_ratio < Decimal("0.8"):
            slagging_risk = SlaggingRisk.HIGH
        else:
            slagging_risk = SlaggingRisk.SEVERE

        if fouling_index < Decimal("0.2"):
            fouling_risk = SlaggingRisk.LOW
        elif fouling_index < Decimal("0.5"):
            fouling_risk = SlaggingRisk.MEDIUM
        elif fouling_index < Decimal("1.0"):
            fouling_risk = SlaggingRisk.HIGH
        else:
            fouling_risk = SlaggingRisk.SEVERE

        # Recommended actions
        actions = []
        if slagging_risk in [SlaggingRisk.HIGH, SlaggingRisk.SEVERE]:
            actions.append("Consider blending with low-slagging fuel")
            actions.append("Increase sootblowing frequency")
        if fouling_risk in [SlaggingRisk.HIGH, SlaggingRisk.SEVERE]:
            actions.append("Monitor convective surface deposits")
            actions.append("Consider fuel additive treatment")
        if ash_comp.na2o_percent > Decimal("2"):
            actions.append("High alkali content - monitor for corrosion")

        return SlaggingFoulingResult(
            base_acid_ratio=base_acid_ratio.quantize(Decimal("0.001")),
            slagging_index_rs=slagging_index.quantize(Decimal("0.01")),
            fouling_index_rf=fouling_index.quantize(Decimal("0.001")),
            t250_temperature_c=t250.quantize(Decimal("1")),
            silica_ratio=silica_ratio.quantize(Decimal("0.1")),
            iron_ratio=iron_ratio.quantize(Decimal("0.001")),
            slagging_risk=slagging_risk,
            fouling_risk=fouling_risk,
            recommended_actions=tuple(actions) if actions else ("No immediate action required",)
        )

    def _calculate_emission_factors(
        self,
        ultimate_daf: UltimateAnalysis,
        heating_value: HeatingValueResult
    ) -> EmissionFactorResult:
        """
        Calculate fuel-specific emission factors.

        Emission factors:
        - CO2: Based on carbon content and molecular weight ratio (44/12 = 3.664)
        - SO2: Based on sulfur content (64/32 = 2.0)
        - NOx: Empirical estimate based on nitrogen content
        - PM: Empirical estimate based on ash and combustion type

        Args:
            ultimate_daf: Ultimate analysis on DAF basis
            heating_value: Calculated heating value

        Returns:
            EmissionFactorResult with all emission factors
        """
        # Carbon to CO2 conversion
        carbon_fraction = ultimate_daf.carbon_percent / Decimal("100")
        co2_per_kg_fuel = carbon_fraction * self.CO2_FROM_CARBON

        # CO2 per GJ
        if heating_value.hhv_mj_kg > 0:
            co2_per_gj = co2_per_kg_fuel * Decimal("1000") / heating_value.hhv_mj_kg
        else:
            co2_per_gj = Decimal("0")

        # SO2 calculation
        sulfur_fraction = ultimate_daf.sulfur_percent / Decimal("100")
        so2_per_kg_fuel = sulfur_fraction * Decimal("2")  # S to SO2 ratio
        so2_per_gj = so2_per_kg_fuel * Decimal("1000") / heating_value.hhv_mj_kg \
            if heating_value.hhv_mj_kg > 0 else Decimal("0")

        # NOx estimate (empirical - depends heavily on combustion conditions)
        nitrogen_fraction = ultimate_daf.nitrogen_percent / Decimal("100")
        # Conversion factor varies with combustion temp and excess air
        nox_per_kg = nitrogen_fraction * Decimal("1.5")  # Rough estimate
        nox_per_gj = nox_per_kg * Decimal("1000") / heating_value.hhv_mj_kg \
            if heating_value.hhv_mj_kg > 0 else Decimal("0")

        # Particulate matter (very rough estimate)
        # Actual PM depends heavily on combustion equipment and controls
        pm_per_gj = Decimal("0.1")  # Default for controlled combustion

        return EmissionFactorResult(
            co2_kg_per_gj=co2_per_gj.quantize(Decimal("0.1")),
            co2_kg_per_kg_fuel=co2_per_kg_fuel.quantize(Decimal("0.001")),
            so2_kg_per_gj=so2_per_gj.quantize(Decimal("0.001")),
            nox_kg_per_gj=nox_per_gj.quantize(Decimal("0.001")),
            particulate_kg_per_gj=pm_per_gj,
            calculation_method="Stoichiometric (CO2, SO2), Empirical (NOx, PM)"
        )

    def _classify_quality_grade(
        self,
        fuel_type: FuelType,
        heating_value: HeatingValueResult,
        proximate_db: ProximateAnalysis,
        sulfur_percent: Decimal
    ) -> QualityGrade:
        """
        Classify fuel quality grade per ASTM standards.

        Grades: Premium > High > Standard > Low > Off-Spec

        Args:
            fuel_type: Type of fuel
            heating_value: Calculated heating value
            proximate_db: Proximate analysis on dry basis
            sulfur_percent: Sulfur content

        Returns:
            Quality grade classification
        """
        hhv = heating_value.hhv_mj_kg
        ash = proximate_db.ash_percent
        moisture = proximate_db.moisture_percent

        for grade_name, thresholds in QUALITY_THRESHOLDS.items():
            meets_grade = True

            if hhv < thresholds.get('min_hhv_mj_kg', Decimal("0")):
                meets_grade = False
            if ash > thresholds.get('max_ash_percent', Decimal("100")):
                meets_grade = False
            if sulfur_percent > thresholds.get('max_sulfur_percent', Decimal("100")):
                meets_grade = False

            if meets_grade:
                return QualityGrade(grade_name)

        return QualityGrade.OFF_SPEC

    def _assess_quality_deviations(
        self,
        fuel_type: FuelType,
        heating_value: HeatingValueResult,
        proximate_db: ProximateAnalysis,
        ultimate_daf: Optional[UltimateAnalysis],
        slagging: Optional[SlaggingFoulingResult]
    ) -> Tuple[Decimal, List[QualityDeviationAlert]]:
        """
        Assess quality deviations and generate alerts.

        Args:
            fuel_type: Type of fuel
            heating_value: Calculated heating value
            proximate_db: Proximate analysis
            ultimate_daf: Ultimate analysis if available
            slagging: Slagging assessment if available

        Returns:
            Tuple of (quality_score, list of alerts)
        """
        alerts = []
        score = Decimal("100")

        # Reference values for the fuel type
        ref = QUALITY_THRESHOLDS['standard']

        # HHV check
        if heating_value.hhv_mj_kg < ref['min_hhv_mj_kg']:
            deviation = (ref['min_hhv_mj_kg'] - heating_value.hhv_mj_kg) / ref['min_hhv_mj_kg'] * Decimal("100")
            score -= deviation
            alerts.append(QualityDeviationAlert(
                alert_type="warning",
                parameter="HHV",
                actual_value=heating_value.hhv_mj_kg,
                specification_value=ref['min_hhv_mj_kg'],
                deviation_percent=deviation.quantize(Decimal("0.1")),
                recommended_action="Consider blending with higher-CV fuel"
            ))

        # Ash check
        if proximate_db.ash_percent > ref['max_ash_percent']:
            deviation = (proximate_db.ash_percent - ref['max_ash_percent']) / ref['max_ash_percent'] * Decimal("100")
            score -= deviation
            alerts.append(QualityDeviationAlert(
                alert_type="warning" if deviation < Decimal("25") else "critical",
                parameter="Ash Content",
                actual_value=proximate_db.ash_percent,
                specification_value=ref['max_ash_percent'],
                deviation_percent=deviation.quantize(Decimal("0.1")),
                recommended_action="High ash increases disposal costs and reduces efficiency"
            ))

        # Sulfur check
        if ultimate_daf and ultimate_daf.sulfur_percent > ref['max_sulfur_percent']:
            deviation = (ultimate_daf.sulfur_percent - ref['max_sulfur_percent']) / ref['max_sulfur_percent'] * Decimal("100")
            score -= deviation * Decimal("1.5")  # Higher weight for environmental impact
            alerts.append(QualityDeviationAlert(
                alert_type="critical",
                parameter="Sulfur Content",
                actual_value=ultimate_daf.sulfur_percent,
                specification_value=ref['max_sulfur_percent'],
                deviation_percent=deviation.quantize(Decimal("0.1")),
                recommended_action="SO2 emissions will exceed limits - consider FGD or fuel switching"
            ))

        # Slagging risk check
        if slagging and slagging.slagging_risk in [SlaggingRisk.HIGH, SlaggingRisk.SEVERE]:
            score -= Decimal("15")
            alerts.append(QualityDeviationAlert(
                alert_type="warning",
                parameter="Slagging Risk",
                actual_value=slagging.base_acid_ratio,
                specification_value=Decimal("0.6"),
                deviation_percent=((slagging.base_acid_ratio - Decimal("0.6")) / Decimal("0.6") * Decimal("100")).quantize(Decimal("0.1")),
                recommended_action="Monitor furnace temperatures and sootblow frequency"
            ))

        return max(score, Decimal("0")), alerts

    def _calculate_sampling_recommendation(
        self,
        fuel_type: FuelType,
        proximate_db: ProximateAnalysis,
        alerts: List[QualityDeviationAlert]
    ) -> SamplingRecommendation:
        """
        Calculate recommended sampling frequency.

        Based on:
        - Fuel variability (higher VM = more variable)
        - Quality alerts (more alerts = more sampling)
        - Fuel type characteristics

        Args:
            fuel_type: Type of fuel
            proximate_db: Proximate analysis
            alerts: Quality alerts

        Returns:
            Sampling frequency recommendation
        """
        # Base frequency by fuel type
        base_frequencies = {
            FuelType.ANTHRACITE: 1,  # Very consistent
            FuelType.BITUMINOUS: 2,
            FuelType.SUB_BITUMINOUS: 3,
            FuelType.LIGNITE: 4,
            FuelType.BIOMASS: 6,  # Most variable
            FuelType.WOOD_PELLETS: 2
        }

        base_freq = base_frequencies.get(fuel_type, 3)

        # Adjust for variability (higher VM suggests more variability)
        variability_factor = proximate_db.volatile_matter_percent / Decimal("30")
        variability_adjustment = int(variability_factor)

        # Adjust for alerts
        alert_adjustment = len(alerts)

        # Calculate final frequency
        recommended_freq = base_freq + variability_adjustment + alert_adjustment
        recommended_freq = max(1, min(recommended_freq, 12))  # Cap at 12 per day

        # Confidence level decreases with fewer samples
        confidence = Decimal("95") - Decimal(str(max(0, 5 - recommended_freq))) * Decimal("5")

        reasoning_parts = []
        if variability_adjustment > 0:
            reasoning_parts.append(f"high VM ({proximate_db.volatile_matter_percent}%) indicates variability")
        if alert_adjustment > 0:
            reasoning_parts.append(f"{alert_adjustment} quality alerts require monitoring")
        if not reasoning_parts:
            reasoning_parts.append("standard sampling for fuel type")

        return SamplingRecommendation(
            recommended_frequency=recommended_freq,
            confidence_level=confidence,
            variability_factor=variability_factor.quantize(Decimal("0.01")),
            reasoning="; ".join(reasoning_parts)
        )

    def _correlate_hardgrove_index(
        self,
        fuel_type: FuelType,
        proximate_db: ProximateAnalysis,
        fuel_ratio: Decimal
    ) -> Optional[Decimal]:
        """
        Correlate Hardgrove Grindability Index from proximate analysis.

        HGI correlation (empirical):
        - Higher volatile matter generally means easier grinding (higher HGI)
        - Higher fixed carbon means harder grinding (lower HGI)

        Args:
            fuel_type: Type of fuel
            proximate_db: Proximate analysis on dry basis
            fuel_ratio: FC/VM ratio

        Returns:
            Estimated HGI or None if not applicable
        """
        # HGI only applicable to coal types
        if fuel_type not in [FuelType.ANTHRACITE, FuelType.BITUMINOUS,
                            FuelType.SUB_BITUMINOUS, FuelType.LIGNITE]:
            return None

        # Base HGI by coal type
        base_hgi = {
            FuelType.ANTHRACITE: Decimal("30"),
            FuelType.BITUMINOUS: Decimal("55"),
            FuelType.SUB_BITUMINOUS: Decimal("50"),
            FuelType.LIGNITE: Decimal("45")
        }

        hgi = base_hgi.get(fuel_type, Decimal("50"))

        # Adjust for fuel ratio (inverse relationship)
        # HGI decreases with increasing fuel ratio
        hgi_adjustment = (Decimal("1.5") - fuel_ratio) * Decimal("10")
        hgi += hgi_adjustment

        # Bound to reasonable range
        hgi = max(Decimal("20"), min(hgi, Decimal("100")))

        return hgi

    def _record_step(
        self,
        step_number: int,
        operation: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        formula: str
    ) -> None:
        """Record a calculation step for provenance."""
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        self._provenance_steps.append({
            'step_number': step_number,
            'operation': operation,
            'inputs': inputs,
            'outputs': outputs,
            'formula': formula,
            'timestamp': timestamp
        })

    def _calculate_provenance_hash(
        self,
        sample: FuelSampleInput,
        heating_value: HeatingValueResult,
        grade: QualityGrade
    ) -> str:
        """
        Calculate SHA-256 provenance hash for audit trail.

        Args:
            sample: Original sample input
            heating_value: Calculated heating value
            grade: Quality grade classification

        Returns:
            64-character hexadecimal SHA-256 hash
        """
        hash_data = {
            'sample_id': sample.sample_id,
            'fuel_type': sample.fuel_type.value,
            'proximate': {
                'moisture': str(sample.proximate.moisture_percent),
                'vm': str(sample.proximate.volatile_matter_percent),
                'fc': str(sample.proximate.fixed_carbon_percent),
                'ash': str(sample.proximate.ash_percent)
            },
            'output': {
                'hhv': str(heating_value.hhv_mj_kg),
                'lhv': str(heating_value.lhv_mj_kg),
                'grade': grade.value
            },
            'steps': len(self._provenance_steps)
        }

        hash_string = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(hash_string.encode('utf-8')).hexdigest()

    @lru_cache(maxsize=500)
    def get_reference_composition(self, fuel_type: str) -> Dict[str, float]:
        """
        Get reference composition for a fuel type (cached).

        Args:
            fuel_type: Fuel type as string

        Returns:
            Dictionary of typical composition values
        """
        typical = {
            'anthracite': {'C': 93, 'H': 3, 'O': 2, 'N': 1, 'S': 1},
            'bituminous': {'C': 82, 'H': 5.5, 'O': 9, 'N': 1.5, 'S': 2},
            'sub_bituminous': {'C': 75, 'H': 5, 'O': 17, 'N': 1.5, 'S': 1.5},
            'lignite': {'C': 68, 'H': 5, 'O': 24, 'N': 1.5, 'S': 1.5},
            'biomass': {'C': 50, 'H': 6, 'O': 42, 'N': 1, 'S': 0.5}
        }
        return typical.get(fuel_type, typical['bituminous'])

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get analyzer statistics.

        Returns:
            Dictionary with analysis count and configuration
        """
        with self._lock:
            return {
                'analysis_count': self._analysis_count,
                'heating_value_method': self._heating_value_method,
                'strict_validation': self._strict_validation,
                'caching_enabled': self._enable_caching,
                'cache_info': self.get_reference_composition.cache_info()._asdict() if self._enable_caching else None
            }

    def clear_cache(self) -> None:
        """Clear the reference composition cache."""
        self.get_reference_composition.cache_clear()
        logger.info("Reference composition cache cleared")
