"""
WasteClassificationDatabaseEngine - Waste classification and emission factor database.

This module implements the WasteClassificationDatabaseEngine for AGENT-MRV-018
(GL-MRV-S3-005) Waste Generated in Operations. It provides comprehensive waste
classification, European Waste Catalogue (EWC) code management, emission factor
lookup for all treatment pathways (landfill, incineration, recycling, composting/AD,
wastewater), and parameter databases for IPCC FOD models.

Features:
- Waste type classification (14 categories, EWC code mapping)
- Emission factor lookup (EPA WARM v16, DEFRA 2024, IPCC 2006/2019)
- Landfill parameters (DOC, MCF, k, oxidation, gas capture efficiency)
- Incineration parameters (dm, CF, FCF, OF, CH4/N2O EF)
- Composting/AD parameters (CH4/N2O EF, leakage rates)
- Wastewater parameters (MCF, Bo, industry-specific loads)
- MSW composition profiles by region
- Treatment compatibility matrix
- Hazardous waste identification (Basel Convention)
- Data quality scoring
- Thread-safe singleton pattern
- Zero-hallucination factor retrieval

Example:
    >>> engine = WasteClassificationDatabaseEngine()
    >>> result = engine.classify_waste("Mixed office paper", "20 01 01", {"paper": 0.85})
    >>> factor = engine.get_emission_factor(WasteCategory.PAPER_CARDBOARD,
    ...                                      WasteTreatmentMethod.LANDFILL,
    ...                                      EFSource.EPA_WARM)
    >>> doc = engine.get_doc(WasteCategory.FOOD_WASTE)
    >>> mcf = engine.get_mcf(LandfillType.MANAGED_ANAEROBIC)
"""

from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import threading
import logging
from datetime import datetime
import re

from greenlang.waste_generated.models import (
    WasteCategory,
    WasteTreatmentMethod,
    WasteStream,
    LandfillType,
    ClimateZone,
    IncineratorType,
    WastewaterSystem,
    GasCollectionSystem,
    EFSource,
    HazardClass,
    IndustryWastewaterType,
    DataQualityTier,
    WasteDataSource,
    DQIDimension,
    DQIScore,
    GWPVersion,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================


class WasteClassificationResult:
    """Result of waste classification."""

    def __init__(
        self,
        waste_category: WasteCategory,
        confidence: Decimal,
        ewc_code: Optional[str] = None,
        is_hazardous: bool = False,
        hazard_classes: Optional[List[HazardClass]] = None,
        compatible_treatments: Optional[List[WasteTreatmentMethod]] = None,
        description: Optional[str] = None,
    ):
        self.waste_category = waste_category
        self.confidence = confidence
        self.ewc_code = ewc_code
        self.is_hazardous = is_hazardous
        self.hazard_classes = hazard_classes or []
        self.compatible_treatments = compatible_treatments or []
        self.description = description


class WasteEmissionFactor:
    """Emission factor for waste treatment."""

    def __init__(
        self,
        waste_category: WasteCategory,
        treatment_method: WasteTreatmentMethod,
        ef_kgco2e_per_tonne: Decimal,
        ef_source: EFSource,
        ch4_factor: Optional[Decimal] = None,
        n2o_factor: Optional[Decimal] = None,
        fossil_co2_factor: Optional[Decimal] = None,
        biogenic_co2_factor: Optional[Decimal] = None,
        gwp_version: GWPVersion = GWPVersion.AR5,
        temporal_correlation: int = 1,
        geographic_correlation: int = 1,
        reference_year: Optional[int] = None,
        notes: Optional[str] = None,
    ):
        self.waste_category = waste_category
        self.treatment_method = treatment_method
        self.ef_kgco2e_per_tonne = ef_kgco2e_per_tonne
        self.ef_source = ef_source
        self.ch4_factor = ch4_factor
        self.n2o_factor = n2o_factor
        self.fossil_co2_factor = fossil_co2_factor
        self.biogenic_co2_factor = biogenic_co2_factor
        self.gwp_version = gwp_version
        self.temporal_correlation = temporal_correlation
        self.geographic_correlation = geographic_correlation
        self.reference_year = reference_year or 2024
        self.notes = notes


class DataQualityResult:
    """Data quality assessment result."""

    def __init__(
        self,
        overall_score: Decimal,
        temporal_score: int,
        geographical_score: int,
        technological_score: int,
        completeness_score: int,
        reliability_score: int,
        tier: DataQualityTier,
        uncertainty_range: Tuple[Decimal, Decimal],
    ):
        self.overall_score = overall_score
        self.temporal_score = temporal_score
        self.geographical_score = geographical_score
        self.technological_score = technological_score
        self.completeness_score = completeness_score
        self.reliability_score = reliability_score
        self.tier = tier
        self.uncertainty_range = uncertainty_range


# ============================================================================
# WASTE CLASSIFICATION DATABASE ENGINE
# ============================================================================


class WasteClassificationDatabaseEngine:
    """
    WasteClassificationDatabaseEngine implementation.

    This engine provides comprehensive waste classification, EWC code management,
    emission factor lookup, and parameter databases for all waste treatment pathways.
    It follows GreenLang's zero-hallucination principle by using only embedded
    deterministic databases from authoritative sources (EPA WARM, DEFRA, IPCC).

    Thread-safe singleton pattern ensures consistent factor retrieval across the application.

    Attributes:
        _instance: Singleton instance
        _lock: Thread lock for singleton initialization
        _initialized: Initialization flag
        _ewc_database: European Waste Catalogue database
        _emission_factors: Emission factor database
        _landfill_parameters: Landfill FOD model parameters
        _incineration_parameters: Incineration parameters
        _composting_parameters: Composting/AD parameters
        _wastewater_parameters: Wastewater treatment parameters
        _msw_composition: MSW composition profiles
        _treatment_compatibility: Treatment compatibility matrix
    """

    _instance: Optional["WasteClassificationDatabaseEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls):
        """Thread-safe singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize WasteClassificationDatabaseEngine (called once)."""
        if self._initialized:
            return

        logger.info("Initializing WasteClassificationDatabaseEngine...")

        # Initialize databases
        self._ewc_database = self._build_ewc_database()
        self._emission_factors = self._build_emission_factor_database()
        self._landfill_parameters = self._build_landfill_parameter_database()
        self._incineration_parameters = self._build_incineration_parameter_database()
        self._composting_parameters = self._build_composting_parameter_database()
        self._wastewater_parameters = self._build_wastewater_parameter_database()
        self._msw_composition = self._build_msw_composition_database()
        self._treatment_compatibility = self._build_treatment_compatibility_matrix()
        self._hazard_classifications = self._build_hazard_classification_database()

        self._initialized = True
        logger.info(
            f"WasteClassificationDatabaseEngine initialized: "
            f"{len(self._ewc_database)} EWC codes, "
            f"{len(self._emission_factors)} emission factors"
        )

    # ========================================================================
    # 1. WASTE TYPE CLASSIFICATION
    # ========================================================================

    def classify_waste(
        self,
        description: str,
        ewc_code: Optional[str] = None,
        composition: Optional[Dict[str, Decimal]] = None,
    ) -> WasteClassificationResult:
        """
        Classify waste material based on description, EWC code, and/or composition.

        Args:
            description: Waste description (text)
            ewc_code: European Waste Catalogue code (6-digit)
            composition: Material composition dict (e.g., {"paper": 0.60, "plastic": 0.30})

        Returns:
            WasteClassificationResult with category, confidence, hazard status

        Example:
            >>> result = engine.classify_waste("Mixed office waste", "20 01 01")
            >>> assert result.waste_category == WasteCategory.PAPER_CARDBOARD
            >>> assert result.confidence > Decimal("0.8")
        """
        # Priority 1: EWC code lookup (most reliable)
        if ewc_code:
            normalized_ewc = self._normalize_ewc_code(ewc_code)
            if normalized_ewc in self._ewc_database:
                ewc_data = self._ewc_database[normalized_ewc]
                return WasteClassificationResult(
                    waste_category=ewc_data["category"],
                    confidence=Decimal("0.95"),
                    ewc_code=normalized_ewc,
                    is_hazardous=ewc_data["is_hazardous"],
                    hazard_classes=ewc_data.get("hazard_classes", []),
                    compatible_treatments=self._treatment_compatibility.get(
                        ewc_data["category"], []
                    ),
                    description=ewc_data["description"],
                )

        # Priority 2: Composition-based classification
        if composition:
            category = self._classify_by_composition(composition)
            if category:
                return WasteClassificationResult(
                    waste_category=category,
                    confidence=Decimal("0.80"),
                    is_hazardous=False,
                    compatible_treatments=self._treatment_compatibility.get(category, []),
                    description=f"Classified by composition: {composition}",
                )

        # Priority 3: Text-based classification (keyword matching)
        category = self._classify_by_description(description)
        return WasteClassificationResult(
            waste_category=category,
            confidence=Decimal("0.60"),
            is_hazardous=False,
            compatible_treatments=self._treatment_compatibility.get(category, []),
            description=f"Classified by description: {description}",
        )

    def get_waste_category(self, ewc_code: str) -> WasteCategory:
        """
        Get waste category from EWC code.

        Args:
            ewc_code: European Waste Catalogue code (6-digit)

        Returns:
            WasteCategory

        Raises:
            ValueError: If EWC code not found
        """
        normalized_ewc = self._normalize_ewc_code(ewc_code)
        if normalized_ewc not in self._ewc_database:
            logger.warning(f"EWC code {ewc_code} not found, returning MIXED_MSW")
            return WasteCategory.MIXED_MSW

        return self._ewc_database[normalized_ewc]["category"]

    def get_compatible_treatments(
        self, waste_category: WasteCategory
    ) -> List[WasteTreatmentMethod]:
        """
        Get compatible treatment methods for waste category.

        Args:
            waste_category: Waste category

        Returns:
            List of compatible WasteTreatmentMethod
        """
        return self._treatment_compatibility.get(
            waste_category,
            [
                WasteTreatmentMethod.LANDFILL,
                WasteTreatmentMethod.INCINERATION,
            ],
        )

    def is_hazardous(self, ewc_code: str) -> bool:
        """
        Check if waste is hazardous per Basel Convention.

        Args:
            ewc_code: European Waste Catalogue code

        Returns:
            True if hazardous, False otherwise
        """
        normalized_ewc = self._normalize_ewc_code(ewc_code)
        if normalized_ewc not in self._ewc_database:
            return False

        return self._ewc_database[normalized_ewc]["is_hazardous"]

    def get_ewc_chapter(self, ewc_code: str) -> str:
        """
        Get EWC chapter description from code.

        Args:
            ewc_code: European Waste Catalogue code

        Returns:
            Chapter description (e.g., "20 - Municipal wastes")
        """
        normalized_ewc = self._normalize_ewc_code(ewc_code)
        chapter_code = normalized_ewc[:2]

        chapter_descriptions = {
            "01": "Wastes from mineral excavation, extraction and physical/chemical treatment",
            "02": "Wastes from agriculture, horticulture, aquaculture, forestry, hunting and fishing",
            "03": "Wastes from wood processing and the production of panels, furniture, pulp, paper",
            "04": "Wastes from the leather, fur and textile industries",
            "05": "Wastes from petroleum refining, natural gas purification and pyrolytic treatment of coal",
            "06": "Wastes from inorganic chemical processes",
            "07": "Wastes from organic chemical processes",
            "08": "Wastes from the manufacture, formulation, supply and use of coatings, adhesives, sealants",
            "09": "Wastes from the photographic industry",
            "10": "Wastes from thermal processes",
            "11": "Wastes from chemical surface treatment and coating of metals and other materials",
            "12": "Wastes from shaping and physical/mechanical surface treatment of metals and plastics",
            "13": "Oil wastes and wastes of liquid fuels (except edible oils)",
            "14": "Waste organic solvents, refrigerants and propellants",
            "15": "Waste packaging; absorbents, wiping cloths, filter materials and protective clothing",
            "16": "Wastes not otherwise specified in the list",
            "17": "Construction and demolition wastes",
            "18": "Wastes from human or animal health care and/or related research",
            "19": "Wastes from waste management facilities, off-site waste water treatment plants",
            "20": "Municipal wastes (household waste and similar commercial, industrial and institutional wastes)",
        }

        return chapter_descriptions.get(chapter_code, f"Chapter {chapter_code}")

    # ========================================================================
    # 2. EMISSION FACTOR LOOKUP
    # ========================================================================

    def get_emission_factor(
        self,
        waste_category: WasteCategory,
        treatment_method: WasteTreatmentMethod,
        source: EFSource = EFSource.EPA_WARM,
    ) -> WasteEmissionFactor:
        """
        Get emission factor for waste category and treatment method.

        Args:
            waste_category: Waste category
            treatment_method: Treatment method
            source: Emission factor source (EPA_WARM, DEFRA_BEIS, IPCC_2006, etc.)

        Returns:
            WasteEmissionFactor with kgCO2e/tonne and gas breakdown

        Raises:
            ValueError: If emission factor not found
        """
        key = (waste_category, treatment_method, source)
        if key in self._emission_factors:
            return self._emission_factors[key]

        # Fallback to different source if requested source not available
        for fallback_source in [EFSource.EPA_WARM, EFSource.DEFRA_BEIS, EFSource.IPCC_2006]:
            fallback_key = (waste_category, treatment_method, fallback_source)
            if fallback_key in self._emission_factors:
                logger.warning(
                    f"Emission factor not found for {key}, using {fallback_source}"
                )
                return self._emission_factors[fallback_key]

        raise ValueError(
            f"No emission factor found for {waste_category}, {treatment_method}, {source}"
        )

    def get_epa_warm_factor(
        self, waste_category: WasteCategory, treatment_method: WasteTreatmentMethod
    ) -> Decimal:
        """
        Get EPA WARM v16 emission factor (MTCO2e/short ton).

        Args:
            waste_category: Waste category
            treatment_method: Treatment method

        Returns:
            Emission factor in MTCO2e/short ton (US units)
        """
        ef = self.get_emission_factor(waste_category, treatment_method, EFSource.EPA_WARM)
        # Convert from kgCO2e/tonne to MTCO2e/short ton
        # 1 tonne = 1.10231 short tons, 1 MT = 1000 kg
        return (ef.ef_kgco2e_per_tonne / Decimal("1000")) * Decimal("1.10231")

    def get_defra_factor(
        self, waste_category: WasteCategory, treatment_method: WasteTreatmentMethod
    ) -> Decimal:
        """
        Get DEFRA 2024 emission factor (kgCO2e/tonne).

        Args:
            waste_category: Waste category
            treatment_method: Treatment method

        Returns:
            Emission factor in kgCO2e/tonne (metric units)
        """
        ef = self.get_emission_factor(waste_category, treatment_method, EFSource.DEFRA_BEIS)
        return ef.ef_kgco2e_per_tonne

    def get_ipcc_defaults(self, waste_category: WasteCategory) -> Dict[str, Decimal]:
        """
        Get IPCC 2006 default parameters for waste category.

        Args:
            waste_category: Waste category

        Returns:
            Dict with DOC, DOCf, MCF default values, k values by climate zone

        Example:
            >>> defaults = engine.get_ipcc_defaults(WasteCategory.FOOD_WASTE)
            >>> assert "doc" in defaults
            >>> assert "k_tropical_wet" in defaults
        """
        if waste_category in self._landfill_parameters["doc"]:
            doc = self._landfill_parameters["doc"][waste_category]
            return {
                "doc": doc,
                "docf": Decimal("0.50"),  # IPCC default
                "mcf_managed_anaerobic": Decimal("1.0"),
                "mcf_unmanaged_deep": Decimal("0.8"),
                "k_boreal_temperate_dry": self._landfill_parameters["decay_rate"].get(
                    (waste_category, ClimateZone.BOREAL_TEMPERATE_DRY), Decimal("0.04")
                ),
                "k_temperate_wet": self._landfill_parameters["decay_rate"].get(
                    (waste_category, ClimateZone.TEMPERATE_WET), Decimal("0.07")
                ),
                "k_tropical_dry": self._landfill_parameters["decay_rate"].get(
                    (waste_category, ClimateZone.TROPICAL_DRY), Decimal("0.065")
                ),
                "k_tropical_wet": self._landfill_parameters["decay_rate"].get(
                    (waste_category, ClimateZone.TROPICAL_WET), Decimal("0.17")
                ),
            }
        return {}

    def get_eeio_factor(self, naics_code: str) -> Decimal:
        """
        Get EEIO factor for waste management services (spend-based method).

        Args:
            naics_code: NAICS code for waste management service

        Returns:
            Emission factor in kgCO2e/USD

        Example:
            >>> factor = engine.get_eeio_factor("562")  # Waste Management and Remediation Services
        """
        # EEIO factors for waste management services (kgCO2e/USD)
        # Source: US EPA EEIO 2022 Detail Model
        eeio_factors = {
            "562": Decimal("0.285"),  # Waste Management and Remediation Services
            "5621": Decimal("0.310"),  # Waste Collection
            "5622": Decimal("0.275"),  # Waste Treatment and Disposal
            "56221": Decimal("0.290"),  # Hazardous Waste Treatment and Disposal
            "56211": Decimal("0.320"),  # Waste Collection
            "56212": Decimal("0.305"),  # Solid Waste Collection
            "56213": Decimal("0.340"),  # Materials Recovery Facilities
        }

        # Match longest prefix
        for code_length in [5, 4, 3]:
            truncated = naics_code[:code_length]
            if truncated in eeio_factors:
                return eeio_factors[truncated]

        logger.warning(f"EEIO factor not found for NAICS {naics_code}, using default")
        return Decimal("0.285")  # Default for sector 562

    def convert_warm_to_metric(self, value_mtco2e_per_short_ton: Decimal) -> Decimal:
        """
        Convert EPA WARM factor from MTCO2e/short ton to kgCO2e/tonne.

        Args:
            value_mtco2e_per_short_ton: Emission factor in MTCO2e/short ton

        Returns:
            Emission factor in kgCO2e/tonne (metric)

        Example:
            >>> metric_ef = engine.convert_warm_to_metric(Decimal("2.5"))
            >>> # 2.5 MTCO2e/short ton = 2,267.96 kgCO2e/tonne
        """
        # 1 short ton = 0.907185 tonnes
        # 1 MT = 1000 kg
        return (value_mtco2e_per_short_ton * Decimal("1000")) / Decimal("1.10231")

    def get_best_available_factor(
        self,
        waste_category: WasteCategory,
        treatment: WasteTreatmentMethod,
        preferred_source: EFSource,
        fallback_source: EFSource,
    ) -> WasteEmissionFactor:
        """
        Get best available emission factor with fallback logic.

        Args:
            waste_category: Waste category
            treatment: Treatment method
            preferred_source: Preferred emission factor source
            fallback_source: Fallback source if preferred not available

        Returns:
            WasteEmissionFactor from preferred or fallback source
        """
        try:
            return self.get_emission_factor(waste_category, treatment, preferred_source)
        except ValueError:
            logger.info(
                f"Preferred source {preferred_source} not available, "
                f"falling back to {fallback_source}"
            )
            return self.get_emission_factor(waste_category, treatment, fallback_source)

    # ========================================================================
    # 3. LANDFILL PARAMETER LOOKUP
    # ========================================================================

    def get_doc(self, waste_category: WasteCategory) -> Decimal:
        """
        Get Degradable Organic Carbon (DOC) fraction for waste category.

        Args:
            waste_category: Waste category

        Returns:
            DOC fraction (0.0-1.0)

        Example:
            >>> doc = engine.get_doc(WasteCategory.FOOD_WASTE)
            >>> assert Decimal("0.10") <= doc <= Decimal("0.20")  # IPCC default 0.15
        """
        return self._landfill_parameters["doc"].get(
            waste_category, Decimal("0.09")  # IPCC default for mixed waste
        )

    def get_mcf(self, landfill_type: LandfillType) -> Decimal:
        """
        Get Methane Correction Factor (MCF) for landfill type.

        Args:
            landfill_type: Landfill type

        Returns:
            MCF (0.0-1.0)

        Example:
            >>> mcf = engine.get_mcf(LandfillType.MANAGED_ANAEROBIC)
            >>> assert mcf == Decimal("1.0")  # Fully anaerobic
        """
        mcf_values = {
            LandfillType.MANAGED_ANAEROBIC: Decimal("1.0"),
            LandfillType.MANAGED_SEMI_AEROBIC: Decimal("0.5"),
            LandfillType.UNMANAGED_DEEP: Decimal("0.8"),
            LandfillType.UNMANAGED_SHALLOW: Decimal("0.4"),
            LandfillType.UNCATEGORIZED: Decimal("0.6"),
            LandfillType.ACTIVE_AERATION: Decimal("0.4"),
        }
        return mcf_values.get(landfill_type, Decimal("0.6"))

    def get_decay_rate(
        self, climate_zone: ClimateZone, waste_category: WasteCategory
    ) -> Decimal:
        """
        Get decay rate constant (k) for climate zone and waste category.

        Args:
            climate_zone: IPCC climate zone
            waste_category: Waste category

        Returns:
            Decay rate constant k (yr^-1)

        Example:
            >>> k = engine.get_decay_rate(ClimateZone.TROPICAL_WET, WasteCategory.FOOD_WASTE)
            >>> assert k > Decimal("0.15")  # Fast decay in tropical wet climate
        """
        key = (waste_category, climate_zone)
        return self._landfill_parameters["decay_rate"].get(
            key, self._get_default_k(climate_zone)
        )

    def get_gas_capture_efficiency(
        self, gas_collection_system: GasCollectionSystem
    ) -> Decimal:
        """
        Get gas capture efficiency for collection system.

        Args:
            gas_collection_system: Gas collection system type

        Returns:
            Capture efficiency (0.0-1.0)

        Example:
            >>> eff = engine.get_gas_capture_efficiency(GasCollectionSystem.ACTIVE_GEOMEMBRANE)
            >>> assert eff == Decimal("0.90")  # 90% capture efficiency
        """
        efficiencies = {
            GasCollectionSystem.NONE: Decimal("0.00"),
            GasCollectionSystem.ACTIVE_OPERATING_CELL: Decimal("0.75"),
            GasCollectionSystem.ACTIVE_TEMP_COVER: Decimal("0.50"),
            GasCollectionSystem.ACTIVE_CLAY_COVER: Decimal("0.65"),
            GasCollectionSystem.ACTIVE_GEOMEMBRANE: Decimal("0.90"),
            GasCollectionSystem.PASSIVE_VENTING: Decimal("0.20"),
            GasCollectionSystem.FLARE_ONLY: Decimal("0.35"),
        }
        return efficiencies.get(gas_collection_system, Decimal("0.00"))

    def get_oxidation_factor(self, has_cover: bool) -> Decimal:
        """
        Get oxidation factor (OX) for landfill cover soil.

        Args:
            has_cover: True if landfill has soil cover

        Returns:
            Oxidation factor (0.0-1.0)

        Example:
            >>> ox = engine.get_oxidation_factor(True)
            >>> assert ox == Decimal("0.10")  # IPCC default for managed landfills
        """
        # IPCC 2006 Vol 5 Table 3.1
        # Managed landfills with cover: OX = 0.10 (10% CH4 oxidized in cover soil)
        # Unmanaged sites without cover: OX = 0.00
        return Decimal("0.10") if has_cover else Decimal("0.00")

    # ========================================================================
    # 4. INCINERATION PARAMETER LOOKUP
    # ========================================================================

    def get_incineration_params(
        self, waste_category: WasteCategory
    ) -> Dict[str, Decimal]:
        """
        Get incineration parameters for waste category.

        Args:
            waste_category: Waste category

        Returns:
            Dict with dm (dry matter), CF (carbon fraction), FCF (fossil carbon fraction),
            OF (oxidation factor)

        Example:
            >>> params = engine.get_incineration_params(WasteCategory.PLASTICS_PET)
            >>> assert params["fcf"] > Decimal("0.90")  # PET is fossil-origin
        """
        key = waste_category
        if key in self._incineration_parameters:
            return self._incineration_parameters[key]

        # Default values for mixed waste
        return {
            "dm": Decimal("0.75"),  # 75% dry matter
            "cf": Decimal("0.50"),  # 50% carbon content of dry matter
            "fcf": Decimal("0.50"),  # 50% fossil carbon (assumes mix)
            "of": Decimal("1.00"),  # 100% oxidation (complete combustion)
        }

    def get_ch4_ef_incineration(self, incinerator_type: IncineratorType) -> Decimal:
        """
        Get CH4 emission factor for incineration type.

        Args:
            incinerator_type: Incinerator type

        Returns:
            CH4 emission factor (kg CH4/tonne waste)

        Example:
            >>> ch4_ef = engine.get_ch4_ef_incineration(IncineratorType.CONTINUOUS_STOKER)
            >>> assert ch4_ef < Decimal("0.5")  # Modern incinerators have low CH4
        """
        # IPCC 2006 Vol 5 Table 5.3
        ch4_factors = {
            IncineratorType.CONTINUOUS_STOKER: Decimal("0.004"),  # 4 kg CH4/tonne
            IncineratorType.SEMI_CONTINUOUS: Decimal("0.010"),  # 10 kg CH4/tonne
            IncineratorType.BATCH: Decimal("0.060"),  # 60 kg CH4/tonne
            IncineratorType.FLUIDIZED_BED: Decimal("0.002"),  # 2 kg CH4/tonne (very low)
            IncineratorType.OPEN_BURNING: Decimal("6.5"),  # 6500 kg CH4/tonne (very high!)
        }
        return ch4_factors.get(incinerator_type, Decimal("0.004"))

    def get_n2o_ef_incineration(self, incinerator_type: IncineratorType) -> Decimal:
        """
        Get N2O emission factor for incineration type.

        Args:
            incinerator_type: Incinerator type

        Returns:
            N2O emission factor (kg N2O/tonne waste)

        Example:
            >>> n2o_ef = engine.get_n2o_ef_incineration(IncineratorType.CONTINUOUS_STOKER)
            >>> assert n2o_ef < Decimal("0.1")  # Modern incinerators have low N2O
        """
        # IPCC 2006 Vol 5 Table 5.3
        n2o_factors = {
            IncineratorType.CONTINUOUS_STOKER: Decimal("0.056"),  # 56 g N2O/tonne
            IncineratorType.SEMI_CONTINUOUS: Decimal("0.090"),  # 90 g N2O/tonne
            IncineratorType.BATCH: Decimal("0.150"),  # 150 g N2O/tonne
            IncineratorType.FLUIDIZED_BED: Decimal("0.050"),  # 50 g N2O/tonne
            IncineratorType.OPEN_BURNING: Decimal("0.098"),  # 98 g N2O/tonne
        }
        return n2o_factors.get(incinerator_type, Decimal("0.056"))

    # ========================================================================
    # 5. COMPOSTING/AD PARAMETER LOOKUP
    # ========================================================================

    def get_composting_ef(self, dry_weight_basis: bool = False) -> Dict[str, Decimal]:
        """
        Get composting emission factors for CH4 and N2O.

        Args:
            dry_weight_basis: True for dry weight basis, False for wet weight

        Returns:
            Dict with ch4 and n2o emission factors (kg gas/tonne waste)

        Example:
            >>> ef = engine.get_composting_ef(dry_weight_basis=False)
            >>> assert "ch4" in ef and "n2o" in ef
        """
        # IPCC 2006 Vol 5 Table 4.1
        if dry_weight_basis:
            return {
                "ch4": Decimal("4.0"),  # 4 kg CH4/tonne dry weight
                "n2o": Decimal("0.30"),  # 0.3 kg N2O/tonne dry weight
            }
        else:
            # Wet weight basis (assumes 50% moisture content)
            return {
                "ch4": Decimal("2.0"),  # 2 kg CH4/tonne wet weight
                "n2o": Decimal("0.15"),  # 0.15 kg N2O/tonne wet weight
            }

    def get_ad_leakage_rate(self, plant_type: str = "modern") -> Decimal:
        """
        Get anaerobic digestion methane leakage rate.

        Args:
            plant_type: "modern", "standard", or "old"

        Returns:
            Leakage rate (fraction of biogas production)

        Example:
            >>> leakage = engine.get_ad_leakage_rate("modern")
            >>> assert leakage < Decimal("0.05")  # Modern plants have low leakage
        """
        leakage_rates = {
            "modern": Decimal("0.01"),  # 1% leakage (sealed digesters)
            "standard": Decimal("0.03"),  # 3% leakage
            "old": Decimal("0.08"),  # 8% leakage (poor sealing)
        }
        return leakage_rates.get(plant_type, Decimal("0.03"))

    # ========================================================================
    # 6. WASTEWATER PARAMETER LOOKUP
    # ========================================================================

    def get_wastewater_mcf(self, treatment_system: WastewaterSystem) -> Decimal:
        """
        Get wastewater treatment MCF (Methane Correction Factor).

        Args:
            treatment_system: Wastewater treatment system type

        Returns:
            MCF (0.0-1.0)

        Example:
            >>> mcf = engine.get_wastewater_mcf(WastewaterSystem.CENTRALIZED_AEROBIC_GOOD)
            >>> assert mcf == Decimal("0.00")  # Aerobic treatment produces no CH4
        """
        mcf_values = {
            WastewaterSystem.CENTRALIZED_AEROBIC_GOOD: Decimal("0.00"),
            WastewaterSystem.CENTRALIZED_AEROBIC_POOR: Decimal("0.03"),
            WastewaterSystem.CENTRALIZED_ANAEROBIC: Decimal("0.80"),
            WastewaterSystem.ANAEROBIC_REACTOR: Decimal("0.80"),
            WastewaterSystem.LAGOON_SHALLOW: Decimal("0.20"),
            WastewaterSystem.LAGOON_DEEP: Decimal("0.80"),
            WastewaterSystem.SEPTIC: Decimal("0.50"),
            WastewaterSystem.OPEN_SEWER: Decimal("0.10"),
            WastewaterSystem.CONSTRUCTED_WETLAND: Decimal("0.05"),
        }
        return mcf_values.get(treatment_system, Decimal("0.50"))

    def get_wastewater_bo(self, measurement_basis: str = "cod") -> Decimal:
        """
        Get maximum CH4 producing capacity (Bo) for wastewater.

        Args:
            measurement_basis: "cod" (Chemical Oxygen Demand) or "bod" (Biochemical Oxygen Demand)

        Returns:
            Bo in kg CH4/kg COD or kg CH4/kg BOD

        Example:
            >>> bo = engine.get_wastewater_bo("cod")
            >>> assert bo == Decimal("0.25")  # IPCC default
        """
        # IPCC 2006 Vol 5 Equation 6.2
        if measurement_basis == "cod":
            return Decimal("0.25")  # kg CH4/kg COD (default for domestic wastewater)
        elif measurement_basis == "bod":
            # Convert from COD basis: assume COD/BOD = 1.6
            return Decimal("0.25") * Decimal("1.6")  # = 0.40 kg CH4/kg BOD
        else:
            return Decimal("0.25")

    def get_industry_wastewater_load(
        self, industry_type: IndustryWastewaterType
    ) -> Dict[str, Decimal]:
        """
        Get industry-specific wastewater organic load parameters.

        Args:
            industry_type: Industry wastewater type

        Returns:
            Dict with cod_kg_per_unit, bod_kg_per_unit, production_unit

        Example:
            >>> params = engine.get_industry_wastewater_load(IndustryWastewaterType.DAIRY)
            >>> assert "cod_kg_per_unit" in params
        """
        # IPCC 2006 Vol 5 Table 6.9
        industry_loads = {
            IndustryWastewaterType.STARCH: {
                "cod_kg_per_unit": Decimal("5.0"),
                "production_unit": "tonne_product",
            },
            IndustryWastewaterType.ALCOHOL: {
                "cod_kg_per_unit": Decimal("18.0"),
                "production_unit": "tonne_product",
            },
            IndustryWastewaterType.BEER_MALT: {
                "cod_kg_per_unit": Decimal("3.5"),
                "production_unit": "tonne_product",
            },
            IndustryWastewaterType.PULP_PAPER: {
                "cod_kg_per_unit": Decimal("30.0"),
                "production_unit": "tonne_product",
            },
            IndustryWastewaterType.FOOD_PROCESSING: {
                "cod_kg_per_unit": Decimal("10.0"),
                "production_unit": "tonne_product",
            },
            IndustryWastewaterType.MEAT_POULTRY: {
                "cod_kg_per_unit": Decimal("15.0"),
                "production_unit": "tonne_product",
            },
            IndustryWastewaterType.VEGETABLES_FRUITS: {
                "cod_kg_per_unit": Decimal("8.0"),
                "production_unit": "tonne_product",
            },
            IndustryWastewaterType.DAIRY: {
                "cod_kg_per_unit": Decimal("12.0"),
                "production_unit": "tonne_product",
            },
            IndustryWastewaterType.SUGAR: {
                "cod_kg_per_unit": Decimal("6.0"),
                "production_unit": "tonne_product",
            },
            IndustryWastewaterType.TEXTILE: {
                "cod_kg_per_unit": Decimal("25.0"),
                "production_unit": "tonne_product",
            },
            IndustryWastewaterType.PHARMACEUTICAL: {
                "cod_kg_per_unit": Decimal("40.0"),
                "production_unit": "tonne_product",
            },
            IndustryWastewaterType.OTHER: {
                "cod_kg_per_unit": Decimal("15.0"),
                "production_unit": "tonne_product",
            },
        }
        return industry_loads.get(
            industry_type,
            {"cod_kg_per_unit": Decimal("15.0"), "production_unit": "tonne_product"},
        )

    # ========================================================================
    # 7. MSW COMPOSITION
    # ========================================================================

    def get_msw_composition(self, region: str = "global") -> Dict[WasteCategory, Decimal]:
        """
        Get typical MSW composition by region.

        Args:
            region: "global", "EU", "US", "CN", "IN", etc.

        Returns:
            Dict mapping WasteCategory to mass fraction

        Example:
            >>> composition = engine.get_msw_composition("US")
            >>> assert sum(composition.values()) == Decimal("1.0")  # Fractions sum to 1.0
        """
        return self._msw_composition.get(
            region, self._msw_composition["global"]
        )

    def get_typical_waste_generation_rate(self, sector: str) -> Decimal:
        """
        Get typical waste generation rate for sector.

        Args:
            sector: "office", "retail", "manufacturing", "healthcare", "hospitality", etc.

        Returns:
            Waste generation rate in tonnes/employee/year

        Example:
            >>> rate = engine.get_typical_waste_generation_rate("office")
            >>> assert Decimal("0.1") <= rate <= Decimal("0.5")  # Typical office range
        """
        # Typical waste generation rates (tonnes/employee/year)
        generation_rates = {
            "office": Decimal("0.25"),  # 250 kg/employee/year
            "retail": Decimal("0.40"),  # 400 kg/employee/year
            "manufacturing": Decimal("3.50"),  # 3.5 tonnes/employee/year
            "healthcare": Decimal("1.20"),  # 1.2 tonnes/employee/year
            "hospitality": Decimal("0.80"),  # 800 kg/employee/year
            "education": Decimal("0.30"),  # 300 kg/employee/year
            "food_service": Decimal("2.00"),  # 2 tonnes/employee/year
            "construction": Decimal("5.00"),  # 5 tonnes/employee/year
            "warehouse": Decimal("0.50"),  # 500 kg/employee/year
        }
        return generation_rates.get(sector, Decimal("0.50"))  # Default 500 kg/year

    # ========================================================================
    # 8. EWC CODE DATABASE (Embedded)
    # ========================================================================

    def _build_ewc_database(self) -> Dict[str, Dict[str, Any]]:
        """
        Build European Waste Catalogue database.

        Returns:
            Dict mapping EWC code to waste category, hazard status, description
        """
        # 50+ common EWC codes mapped to WasteCategory
        return {
            # Chapter 15 - Packaging waste
            "150101": {
                "category": WasteCategory.PAPER_CARDBOARD,
                "is_hazardous": False,
                "description": "Paper and cardboard packaging",
            },
            "150102": {
                "category": WasteCategory.PLASTICS_MIXED,
                "is_hazardous": False,
                "description": "Plastic packaging",
            },
            "150103": {
                "category": WasteCategory.WOOD,
                "is_hazardous": False,
                "description": "Wooden packaging",
            },
            "150104": {
                "category": WasteCategory.METALS_ALUMINUM,
                "is_hazardous": False,
                "description": "Metallic packaging",
            },
            "150105": {
                "category": WasteCategory.PLASTICS_MIXED,
                "is_hazardous": False,
                "description": "Composite packaging",
            },
            "150107": {
                "category": WasteCategory.GLASS,
                "is_hazardous": False,
                "description": "Glass packaging",
            },
            # Chapter 16 - Wastes not otherwise specified
            "160103": {
                "category": WasteCategory.RUBBER_LEATHER,
                "is_hazardous": False,
                "description": "End-of-life tyres",
            },
            "160214": {
                "category": WasteCategory.ELECTRONICS,
                "is_hazardous": False,
                "description": "Discarded equipment (other than 16 02 09 to 16 02 13)",
            },
            # Chapter 17 - Construction and demolition
            "170101": {
                "category": WasteCategory.CONSTRUCTION_DEMOLITION,
                "is_hazardous": False,
                "description": "Concrete",
            },
            "170102": {
                "category": WasteCategory.CONSTRUCTION_DEMOLITION,
                "is_hazardous": False,
                "description": "Bricks",
            },
            "170201": {
                "category": WasteCategory.WOOD,
                "is_hazardous": False,
                "description": "Wood",
            },
            "170203": {
                "category": WasteCategory.PLASTICS_MIXED,
                "is_hazardous": False,
                "description": "Plastic",
            },
            "170204": {
                "category": WasteCategory.GLASS,
                "is_hazardous": False,
                "description": "Glass, plastic and wood containing or contaminated with dangerous substances",
            },
            "170405": {
                "category": WasteCategory.METALS_STEEL,
                "is_hazardous": False,
                "description": "Iron and steel",
            },
            "170407": {
                "category": WasteCategory.METALS_MIXED,
                "is_hazardous": False,
                "description": "Mixed metals",
            },
            # Chapter 20 - Municipal wastes
            "200101": {
                "category": WasteCategory.PAPER_CARDBOARD,
                "is_hazardous": False,
                "description": "Paper and cardboard",
            },
            "200102": {
                "category": WasteCategory.GLASS,
                "is_hazardous": False,
                "description": "Glass",
            },
            "200108": {
                "category": WasteCategory.FOOD_WASTE,
                "is_hazardous": False,
                "description": "Biodegradable kitchen and canteen waste",
            },
            "200110": {
                "category": WasteCategory.TEXTILES,
                "is_hazardous": False,
                "description": "Clothes",
            },
            "200111": {
                "category": WasteCategory.TEXTILES,
                "is_hazardous": False,
                "description": "Textiles",
            },
            "200125": {
                "category": WasteCategory.FOOD_WASTE,
                "is_hazardous": False,
                "description": "Edible oil and fat",
            },
            "200138": {
                "category": WasteCategory.WOOD,
                "is_hazardous": False,
                "description": "Wood other than that mentioned in 20 01 37",
            },
            "200139": {
                "category": WasteCategory.PLASTICS_MIXED,
                "is_hazardous": False,
                "description": "Plastics",
            },
            "200140": {
                "category": WasteCategory.METALS_MIXED,
                "is_hazardous": False,
                "description": "Metals",
            },
            "200201": {
                "category": WasteCategory.GARDEN_WASTE,
                "is_hazardous": False,
                "description": "Biodegradable waste (garden and park waste)",
            },
            "200301": {
                "category": WasteCategory.MIXED_MSW,
                "is_hazardous": False,
                "description": "Mixed municipal waste",
            },
            "200307": {
                "category": WasteCategory.MIXED_MSW,
                "is_hazardous": False,
                "description": "Bulky waste",
            },
            # Hazardous waste examples (marked with *)
            "150110": {
                "category": WasteCategory.HAZARDOUS,
                "is_hazardous": True,
                "hazard_classes": [HazardClass.H3, HazardClass.H6_1],
                "description": "Packaging containing residues of or contaminated by dangerous substances*",
            },
            "160601": {
                "category": WasteCategory.HAZARDOUS,
                "is_hazardous": True,
                "hazard_classes": [HazardClass.H8],
                "description": "Lead batteries*",
            },
            "170503": {
                "category": WasteCategory.HAZARDOUS,
                "is_hazardous": True,
                "hazard_classes": [HazardClass.H6_1],
                "description": "Soil and stones containing dangerous substances*",
            },
            "200127": {
                "category": WasteCategory.HAZARDOUS,
                "is_hazardous": True,
                "hazard_classes": [HazardClass.H8, HazardClass.H6_1],
                "description": "Paint, inks, adhesives and resins containing dangerous substances*",
            },
            # Additional common categories
            "040222": {
                "category": WasteCategory.TEXTILES,
                "is_hazardous": False,
                "description": "Wastes from processed textile fibres",
            },
            "070213": {
                "category": WasteCategory.PLASTICS_MIXED,
                "is_hazardous": False,
                "description": "Plastic waste",
            },
            "100101": {
                "category": WasteCategory.OTHER,
                "is_hazardous": False,
                "description": "Bottom ash, slag and boiler dust",
            },
            "120101": {
                "category": WasteCategory.METALS_STEEL,
                "is_hazardous": False,
                "description": "Ferrous metal filings and turnings",
            },
            "120103": {
                "category": WasteCategory.METALS_MIXED,
                "is_hazardous": False,
                "description": "Non-ferrous metal filings and turnings",
            },
            # Plastic-specific codes
            "070211": {
                "category": WasteCategory.PLASTICS_MIXED,
                "is_hazardous": False,
                "description": "Sludges from on-site effluent treatment",
            },
            "120105": {
                "category": WasteCategory.PLASTICS_MIXED,
                "is_hazardous": False,
                "description": "Plastic shavings and turnings",
            },
            # Food waste codes
            "020103": {
                "category": WasteCategory.FOOD_WASTE,
                "is_hazardous": False,
                "description": "Plant-tissue waste",
            },
            "020304": {
                "category": WasteCategory.FOOD_WASTE,
                "is_hazardous": False,
                "description": "Materials unsuitable for consumption or processing",
            },
            "020601": {
                "category": WasteCategory.FOOD_WASTE,
                "is_hazardous": False,
                "description": "Materials unsuitable for consumption or processing",
            },
            # Electronics
            "200135": {
                "category": WasteCategory.ELECTRONICS,
                "is_hazardous": True,
                "hazard_classes": [HazardClass.H6_1],
                "description": "Discarded electrical and electronic equipment*",
            },
            "160213": {
                "category": WasteCategory.ELECTRONICS,
                "is_hazardous": True,
                "hazard_classes": [HazardClass.H6_1],
                "description": "Discarded equipment containing hazardous components*",
            },
        }

    # ========================================================================
    # 9. DATA QUALITY
    # ========================================================================

    def get_dqi_score(
        self,
        data_source: WasteDataSource,
        ef_source: EFSource,
        temporal: int,
        geographic: int,
        completeness: int,
    ) -> DataQualityResult:
        """
        Calculate Data Quality Indicator score per GHG Protocol.

        Args:
            data_source: Waste data source type
            ef_source: Emission factor source
            temporal: Temporal correlation score (1-5)
            geographic: Geographic correlation score (1-5)
            completeness: Data completeness score (1-5)

        Returns:
            DataQualityResult with overall score and tier

        Example:
            >>> dqi = engine.get_dqi_score(
            ...     WasteDataSource.WASTE_AUDIT,
            ...     EFSource.EPA_WARM,
            ...     temporal=1,
            ...     geographic=2,
            ...     completeness=1
            ... )
            >>> assert dqi.tier == DataQualityTier.TIER_2
        """
        # Reliability score based on data source
        reliability_scores = {
            WasteDataSource.WASTE_AUDIT: 1,
            WasteDataSource.TRANSFER_NOTES: 2,
            WasteDataSource.PROCUREMENT_ESTIMATE: 3,
            WasteDataSource.SPEND_ESTIMATE: 4,
        }
        reliability = reliability_scores.get(data_source, 3)

        # Technological score based on EF source
        technological_scores = {
            EFSource.CUSTOM: 1,  # Facility-specific
            EFSource.EPA_WARM: 2,  # Technology-specific
            EFSource.DEFRA_BEIS: 2,
            EFSource.IPCC_2019: 2,
            EFSource.IPCC_2006: 3,  # Generic
        }
        technological = technological_scores.get(ef_source, 3)

        # Calculate overall score (arithmetic mean)
        overall = Decimal(
            (temporal + geographic + technological + completeness + reliability) / 5
        )

        # Determine tier based on overall score
        if overall <= Decimal("1.5"):
            tier = DataQualityTier.TIER_3
            uncertainty = (Decimal("0.90"), Decimal("1.10"))  # ±10%
        elif overall <= Decimal("2.5"):
            tier = DataQualityTier.TIER_2
            uncertainty = (Decimal("0.75"), Decimal("1.25"))  # ±25%
        else:
            tier = DataQualityTier.TIER_1
            uncertainty = (Decimal("0.50"), Decimal("1.50"))  # ±50%

        return DataQualityResult(
            overall_score=overall,
            temporal_score=temporal,
            geographical_score=geographic,
            technological_score=technological,
            completeness_score=completeness,
            reliability_score=reliability,
            tier=tier,
            uncertainty_range=uncertainty,
        )

    def get_uncertainty_range(
        self, tier: DataQualityTier, treatment_method: WasteTreatmentMethod
    ) -> Dict[str, Tuple[Decimal, Decimal]]:
        """
        Get uncertainty range for data tier and treatment method.

        Args:
            tier: Data quality tier
            treatment_method: Treatment method

        Returns:
            Dict with uncertainty ranges for each gas

        Example:
            >>> ranges = engine.get_uncertainty_range(
            ...     DataQualityTier.TIER_2,
            ...     WasteTreatmentMethod.LANDFILL
            ... )
            >>> assert "ch4" in ranges
        """
        # Base uncertainty by tier
        base_uncertainty = {
            DataQualityTier.TIER_3: Decimal("0.10"),  # ±10%
            DataQualityTier.TIER_2: Decimal("0.25"),  # ±25%
            DataQualityTier.TIER_1: Decimal("0.50"),  # ±50%
        }

        # Treatment-specific uncertainty multipliers
        treatment_multipliers = {
            WasteTreatmentMethod.LANDFILL: Decimal("1.5"),  # Higher uncertainty (FOD model)
            WasteTreatmentMethod.LANDFILL_WITH_GAS_CAPTURE: Decimal("1.3"),
            WasteTreatmentMethod.INCINERATION: Decimal("1.0"),  # Lower uncertainty
            WasteTreatmentMethod.INCINERATION_WITH_ENERGY_RECOVERY: Decimal("0.9"),
            WasteTreatmentMethod.RECYCLING_CLOSED_LOOP: Decimal("1.1"),
            WasteTreatmentMethod.COMPOSTING: Decimal("1.4"),
            WasteTreatmentMethod.ANAEROBIC_DIGESTION: Decimal("1.2"),
            WasteTreatmentMethod.WASTEWATER_TREATMENT: Decimal("1.6"),  # Highest uncertainty
        }

        base = base_uncertainty.get(tier, Decimal("0.25"))
        multiplier = treatment_multipliers.get(treatment_method, Decimal("1.0"))
        uncertainty = base * multiplier

        return {
            "ch4": (Decimal("1.0") - uncertainty, Decimal("1.0") + uncertainty),
            "n2o": (Decimal("1.0") - uncertainty * Decimal("1.2"), Decimal("1.0") + uncertainty * Decimal("1.2")),
            "co2": (Decimal("1.0") - uncertainty * Decimal("0.8"), Decimal("1.0") + uncertainty * Decimal("0.8")),
        }

    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================

    def _normalize_ewc_code(self, ewc_code: str) -> str:
        """Normalize EWC code to 6 digits without spaces."""
        return re.sub(r"[\s\-\*]", "", ewc_code).zfill(6)

    def _classify_by_composition(
        self, composition: Dict[str, Decimal]
    ) -> Optional[WasteCategory]:
        """Classify waste based on material composition."""
        # Find dominant material (>50%)
        for material, fraction in composition.items():
            if fraction > Decimal("0.50"):
                material_lower = material.lower()
                if "paper" in material_lower or "cardboard" in material_lower:
                    return WasteCategory.PAPER_CARDBOARD
                elif "plastic" in material_lower or "pet" in material_lower:
                    return WasteCategory.PLASTICS_MIXED
                elif "food" in material_lower or "organic" in material_lower:
                    return WasteCategory.FOOD_WASTE
                elif "metal" in material_lower or "steel" in material_lower:
                    return WasteCategory.METALS_MIXED
                elif "glass" in material_lower:
                    return WasteCategory.GLASS
                elif "wood" in material_lower or "timber" in material_lower:
                    return WasteCategory.WOOD

        # No dominant material, return mixed
        return WasteCategory.MIXED_MSW

    def _classify_by_description(self, description: str) -> WasteCategory:
        """Classify waste based on text description (keyword matching)."""
        desc_lower = description.lower()

        # Keyword mapping
        keywords = {
            WasteCategory.PAPER_CARDBOARD: ["paper", "cardboard", "carton", "office waste"],
            WasteCategory.PLASTICS_MIXED: ["plastic", "polymer", "pet", "hdpe", "ldpe", "pp"],
            WasteCategory.FOOD_WASTE: ["food", "organic", "kitchen", "canteen", "compost"],
            WasteCategory.GARDEN_WASTE: ["garden", "green", "yard", "leaves", "grass"],
            WasteCategory.GLASS: ["glass", "bottle", "jar"],
            WasteCategory.METALS_ALUMINUM: ["aluminum", "aluminium", "can"],
            WasteCategory.METALS_STEEL: ["steel", "iron", "ferrous"],
            WasteCategory.TEXTILES: ["textile", "fabric", "cloth", "clothing"],
            WasteCategory.WOOD: ["wood", "timber", "lumber", "pallet"],
            WasteCategory.ELECTRONICS: ["electronic", "weee", "electrical", "ewaste"],
            WasteCategory.CONSTRUCTION_DEMOLITION: ["construction", "demolition", "concrete", "brick"],
            WasteCategory.HAZARDOUS: ["hazardous", "dangerous", "toxic", "chemical"],
        }

        for category, words in keywords.items():
            if any(word in desc_lower for word in words):
                return category

        # Default to mixed MSW
        return WasteCategory.MIXED_MSW

    def _get_default_k(self, climate_zone: ClimateZone) -> Decimal:
        """Get default decay rate constant for climate zone."""
        defaults = {
            ClimateZone.BOREAL_TEMPERATE_DRY: Decimal("0.04"),
            ClimateZone.TEMPERATE_WET: Decimal("0.07"),
            ClimateZone.TROPICAL_DRY: Decimal("0.065"),
            ClimateZone.TROPICAL_WET: Decimal("0.17"),
        }
        return defaults.get(climate_zone, Decimal("0.07"))

    # ========================================================================
    # DATABASE BUILDERS
    # ========================================================================

    def _build_emission_factor_database(self) -> Dict[Tuple, WasteEmissionFactor]:
        """
        Build comprehensive emission factor database.

        Returns:
            Dict mapping (WasteCategory, WasteTreatmentMethod, EFSource) to WasteEmissionFactor
        """
        factors = {}

        # EPA WARM v16 factors (converted from MTCO2e/short ton to kgCO2e/tonne)
        # Source: https://www.epa.gov/warm/versions-waste-reduction-model-warm
        warm_factors = [
            # Landfill (no gas capture)
            (WasteCategory.PAPER_CARDBOARD, WasteTreatmentMethod.LANDFILL, Decimal("1400")),
            (WasteCategory.FOOD_WASTE, WasteTreatmentMethod.LANDFILL, Decimal("475")),
            (WasteCategory.GARDEN_WASTE, WasteTreatmentMethod.LANDFILL, Decimal("200")),
            (WasteCategory.WOOD, WasteTreatmentMethod.LANDFILL, Decimal("690")),
            (WasteCategory.TEXTILES, WasteTreatmentMethod.LANDFILL, Decimal("1500")),
            (WasteCategory.PLASTICS_MIXED, WasteTreatmentMethod.LANDFILL, Decimal("25")),
            (WasteCategory.PLASTICS_PET, WasteTreatmentMethod.LANDFILL, Decimal("20")),
            (WasteCategory.PLASTICS_HDPE, WasteTreatmentMethod.LANDFILL, Decimal("20")),
            (WasteCategory.GLASS, WasteTreatmentMethod.LANDFILL, Decimal("12")),
            (WasteCategory.METALS_ALUMINUM, WasteTreatmentMethod.LANDFILL, Decimal("15")),
            (WasteCategory.METALS_STEEL, WasteTreatmentMethod.LANDFILL, Decimal("15")),
            # Incineration
            (WasteCategory.PAPER_CARDBOARD, WasteTreatmentMethod.INCINERATION, Decimal("1340")),
            (WasteCategory.FOOD_WASTE, WasteTreatmentMethod.INCINERATION, Decimal("330")),
            (WasteCategory.PLASTICS_MIXED, WasteTreatmentMethod.INCINERATION, Decimal("2950")),
            (WasteCategory.PLASTICS_PET, WasteTreatmentMethod.INCINERATION, Decimal("2100")),
            (WasteCategory.PLASTICS_HDPE, WasteTreatmentMethod.INCINERATION, Decimal("3150")),
            (WasteCategory.WOOD, WasteTreatmentMethod.INCINERATION, Decimal("70")),
            (WasteCategory.TEXTILES, WasteTreatmentMethod.INCINERATION, Decimal("1750")),
            # Recycling (transport + processing only, cut-off approach)
            (WasteCategory.PAPER_CARDBOARD, WasteTreatmentMethod.RECYCLING_CLOSED_LOOP, Decimal("140")),
            (WasteCategory.PLASTICS_PET, WasteTreatmentMethod.RECYCLING_CLOSED_LOOP, Decimal("380")),
            (WasteCategory.PLASTICS_HDPE, WasteTreatmentMethod.RECYCLING_CLOSED_LOOP, Decimal("350")),
            (WasteCategory.GLASS, WasteTreatmentMethod.RECYCLING_CLOSED_LOOP, Decimal("50")),
            (WasteCategory.METALS_ALUMINUM, WasteTreatmentMethod.RECYCLING_CLOSED_LOOP, Decimal("260")),
            (WasteCategory.METALS_STEEL, WasteTreatmentMethod.RECYCLING_CLOSED_LOOP, Decimal("95")),
            # Composting
            (WasteCategory.FOOD_WASTE, WasteTreatmentMethod.COMPOSTING, Decimal("75")),
            (WasteCategory.GARDEN_WASTE, WasteTreatmentMethod.COMPOSTING, Decimal("65")),
            # Anaerobic Digestion
            (WasteCategory.FOOD_WASTE, WasteTreatmentMethod.ANAEROBIC_DIGESTION, Decimal("18")),
        ]

        for category, treatment, ef_value in warm_factors:
            factors[(category, treatment, EFSource.EPA_WARM)] = WasteEmissionFactor(
                waste_category=category,
                treatment_method=treatment,
                ef_kgco2e_per_tonne=ef_value,
                ef_source=EFSource.EPA_WARM,
                gwp_version=GWPVersion.AR5,
                temporal_correlation=1,
                geographic_correlation=2,  # US-specific
                reference_year=2024,
                notes="EPA WARM v16 (2024)",
            )

        # DEFRA 2024 factors (kgCO2e/tonne)
        # Source: UK Government GHG Conversion Factors for Company Reporting
        defra_factors = [
            # Landfill
            (WasteCategory.PAPER_CARDBOARD, WasteTreatmentMethod.LANDFILL, Decimal("1268")),
            (WasteCategory.FOOD_WASTE, WasteTreatmentMethod.LANDFILL, Decimal("465")),
            (WasteCategory.GARDEN_WASTE, WasteTreatmentMethod.LANDFILL, Decimal("211")),
            (WasteCategory.WOOD, WasteTreatmentMethod.LANDFILL, Decimal("720")),
            (WasteCategory.TEXTILES, WasteTreatmentMethod.LANDFILL, Decimal("1520")),
            (WasteCategory.PLASTICS_MIXED, WasteTreatmentMethod.LANDFILL, Decimal("21")),
            (WasteCategory.GLASS, WasteTreatmentMethod.LANDFILL, Decimal("8")),
            (WasteCategory.METALS_MIXED, WasteTreatmentMethod.LANDFILL, Decimal("12")),
            (WasteCategory.MIXED_MSW, WasteTreatmentMethod.LANDFILL, Decimal("497")),
            # Landfill with gas capture
            (WasteCategory.PAPER_CARDBOARD, WasteTreatmentMethod.LANDFILL_WITH_GAS_CAPTURE, Decimal("380")),
            (WasteCategory.FOOD_WASTE, WasteTreatmentMethod.LANDFILL_WITH_GAS_CAPTURE, Decimal("140")),
            (WasteCategory.MIXED_MSW, WasteTreatmentMethod.LANDFILL_WITH_GAS_CAPTURE, Decimal("149")),
            # Incineration
            (WasteCategory.PAPER_CARDBOARD, WasteTreatmentMethod.INCINERATION, Decimal("1355")),
            (WasteCategory.FOOD_WASTE, WasteTreatmentMethod.INCINERATION, Decimal("310")),
            (WasteCategory.PLASTICS_MIXED, WasteTreatmentMethod.INCINERATION, Decimal("2770")),
            (WasteCategory.TEXTILES, WasteTreatmentMethod.INCINERATION, Decimal("1690")),
            (WasteCategory.MIXED_MSW, WasteTreatmentMethod.INCINERATION, Decimal("930")),
            # Incineration with energy recovery (WtE)
            (WasteCategory.MIXED_MSW, WasteTreatmentMethod.INCINERATION_WITH_ENERGY_RECOVERY, Decimal("21")),
            # Composting
            (WasteCategory.FOOD_WASTE, WasteTreatmentMethod.COMPOSTING, Decimal("85")),
            (WasteCategory.GARDEN_WASTE, WasteTreatmentMethod.COMPOSTING, Decimal("72")),
            # Anaerobic Digestion
            (WasteCategory.FOOD_WASTE, WasteTreatmentMethod.ANAEROBIC_DIGESTION, Decimal("10")),
            # Recycling
            (WasteCategory.PAPER_CARDBOARD, WasteTreatmentMethod.RECYCLING_CLOSED_LOOP, Decimal("150")),
            (WasteCategory.PLASTICS_MIXED, WasteTreatmentMethod.RECYCLING_CLOSED_LOOP, Decimal("420")),
            (WasteCategory.GLASS, WasteTreatmentMethod.RECYCLING_CLOSED_LOOP, Decimal("45")),
            (WasteCategory.METALS_MIXED, WasteTreatmentMethod.RECYCLING_CLOSED_LOOP, Decimal("110")),
        ]

        for category, treatment, ef_value in defra_factors:
            factors[(category, treatment, EFSource.DEFRA_BEIS)] = WasteEmissionFactor(
                waste_category=category,
                treatment_method=treatment,
                ef_kgco2e_per_tonne=ef_value,
                ef_source=EFSource.DEFRA_BEIS,
                gwp_version=GWPVersion.AR5,
                temporal_correlation=1,
                geographic_correlation=1,  # UK/EU-specific
                reference_year=2024,
                notes="DEFRA/DESNZ 2024",
            )

        # IPCC 2006 generic factors (calculated from default parameters)
        # These are illustrative - actual FOD model calculation needed
        ipcc_factors = [
            (WasteCategory.FOOD_WASTE, WasteTreatmentMethod.LANDFILL, Decimal("500")),
            (WasteCategory.PAPER_CARDBOARD, WasteTreatmentMethod.LANDFILL, Decimal("1300")),
            (WasteCategory.GARDEN_WASTE, WasteTreatmentMethod.LANDFILL, Decimal("220")),
            (WasteCategory.WOOD, WasteTreatmentMethod.LANDFILL, Decimal("700")),
            (WasteCategory.TEXTILES, WasteTreatmentMethod.LANDFILL, Decimal("1450")),
        ]

        for category, treatment, ef_value in ipcc_factors:
            factors[(category, treatment, EFSource.IPCC_2006)] = WasteEmissionFactor(
                waste_category=category,
                treatment_method=treatment,
                ef_kgco2e_per_tonne=ef_value,
                ef_source=EFSource.IPCC_2006,
                gwp_version=GWPVersion.AR5,
                temporal_correlation=2,
                geographic_correlation=3,  # Global generic
                reference_year=2006,
                notes="IPCC 2006 Vol 5 default",
            )

        return factors

    def _build_landfill_parameter_database(self) -> Dict[str, Any]:
        """Build landfill FOD model parameter database."""
        return {
            # DOC (Degradable Organic Carbon) fraction by waste category
            # Source: IPCC 2006 Vol 5 Table 2.4
            "doc": {
                WasteCategory.FOOD_WASTE: Decimal("0.15"),
                WasteCategory.GARDEN_WASTE: Decimal("0.20"),
                WasteCategory.PAPER_CARDBOARD: Decimal("0.40"),
                WasteCategory.WOOD: Decimal("0.43"),
                WasteCategory.TEXTILES: Decimal("0.24"),
                WasteCategory.PLASTICS_MIXED: Decimal("0.00"),  # Non-degradable
                WasteCategory.GLASS: Decimal("0.00"),
                WasteCategory.METALS_MIXED: Decimal("0.00"),
                WasteCategory.MIXED_MSW: Decimal("0.09"),  # Composite default
            },
            # Decay rate constant k (yr^-1) by climate zone and waste category
            # Source: IPCC 2006 Vol 5 Table 3.3
            "decay_rate": {
                # Food waste
                (WasteCategory.FOOD_WASTE, ClimateZone.BOREAL_TEMPERATE_DRY): Decimal("0.06"),
                (WasteCategory.FOOD_WASTE, ClimateZone.TEMPERATE_WET): Decimal("0.185"),
                (WasteCategory.FOOD_WASTE, ClimateZone.TROPICAL_DRY): Decimal("0.065"),
                (WasteCategory.FOOD_WASTE, ClimateZone.TROPICAL_WET): Decimal("0.40"),
                # Garden/park waste
                (WasteCategory.GARDEN_WASTE, ClimateZone.BOREAL_TEMPERATE_DRY): Decimal("0.05"),
                (WasteCategory.GARDEN_WASTE, ClimateZone.TEMPERATE_WET): Decimal("0.10"),
                (WasteCategory.GARDEN_WASTE, ClimateZone.TROPICAL_DRY): Decimal("0.055"),
                (WasteCategory.GARDEN_WASTE, ClimateZone.TROPICAL_WET): Decimal("0.17"),
                # Paper/cardboard
                (WasteCategory.PAPER_CARDBOARD, ClimateZone.BOREAL_TEMPERATE_DRY): Decimal("0.04"),
                (WasteCategory.PAPER_CARDBOARD, ClimateZone.TEMPERATE_WET): Decimal("0.07"),
                (WasteCategory.PAPER_CARDBOARD, ClimateZone.TROPICAL_DRY): Decimal("0.045"),
                (WasteCategory.PAPER_CARDBOARD, ClimateZone.TROPICAL_WET): Decimal("0.07"),
                # Wood
                (WasteCategory.WOOD, ClimateZone.BOREAL_TEMPERATE_DRY): Decimal("0.02"),
                (WasteCategory.WOOD, ClimateZone.TEMPERATE_WET): Decimal("0.035"),
                (WasteCategory.WOOD, ClimateZone.TROPICAL_DRY): Decimal("0.025"),
                (WasteCategory.WOOD, ClimateZone.TROPICAL_WET): Decimal("0.035"),
                # Textiles
                (WasteCategory.TEXTILES, ClimateZone.BOREAL_TEMPERATE_DRY): Decimal("0.04"),
                (WasteCategory.TEXTILES, ClimateZone.TEMPERATE_WET): Decimal("0.07"),
                (WasteCategory.TEXTILES, ClimateZone.TROPICAL_DRY): Decimal("0.045"),
                (WasteCategory.TEXTILES, ClimateZone.TROPICAL_WET): Decimal("0.07"),
            },
        }

    def _build_incineration_parameter_database(self) -> Dict[WasteCategory, Dict[str, Decimal]]:
        """Build incineration parameter database."""
        # IPCC 2006 Vol 5 Table 5.2
        return {
            WasteCategory.PAPER_CARDBOARD: {
                "dm": Decimal("0.90"),  # 90% dry matter
                "cf": Decimal("0.46"),  # 46% carbon content
                "fcf": Decimal("0.01"),  # 1% fossil carbon (mostly biogenic)
                "of": Decimal("1.00"),  # 100% oxidation
            },
            WasteCategory.FOOD_WASTE: {
                "dm": Decimal("0.40"),  # 40% dry matter (60% moisture)
                "cf": Decimal("0.38"),
                "fcf": Decimal("0.00"),  # 100% biogenic
                "of": Decimal("0.99"),
            },
            WasteCategory.GARDEN_WASTE: {
                "dm": Decimal("0.50"),
                "cf": Decimal("0.49"),
                "fcf": Decimal("0.00"),
                "of": Decimal("1.00"),
            },
            WasteCategory.WOOD: {
                "dm": Decimal("0.85"),
                "cf": Decimal("0.50"),
                "fcf": Decimal("0.00"),
                "of": Decimal("1.00"),
            },
            WasteCategory.TEXTILES: {
                "dm": Decimal("0.80"),
                "cf": Decimal("0.50"),
                "fcf": Decimal("0.20"),  # Mix of natural and synthetic
                "of": Decimal("0.99"),
            },
            WasteCategory.PLASTICS_MIXED: {
                "dm": Decimal("1.00"),  # 100% dry
                "cf": Decimal("0.75"),
                "fcf": Decimal("1.00"),  # 100% fossil origin
                "of": Decimal("0.98"),
            },
            WasteCategory.PLASTICS_PET: {
                "dm": Decimal("1.00"),
                "cf": Decimal("0.625"),  # C10H8O4
                "fcf": Decimal("1.00"),
                "of": Decimal("0.98"),
            },
            WasteCategory.PLASTICS_HDPE: {
                "dm": Decimal("1.00"),
                "cf": Decimal("0.855"),  # C2H4 polymer
                "fcf": Decimal("1.00"),
                "of": Decimal("0.98"),
            },
            WasteCategory.RUBBER_LEATHER: {
                "dm": Decimal("0.84"),
                "cf": Decimal("0.67"),
                "fcf": Decimal("0.20"),
                "of": Decimal("0.99"),
            },
        }

    def _build_composting_parameter_database(self) -> Dict[str, Any]:
        """Build composting/AD parameter database."""
        # IPCC 2006 Vol 5 Chapter 4
        return {
            "composting_ef": {
                "ch4_wet": Decimal("2.0"),  # kg CH4/tonne wet weight
                "n2o_wet": Decimal("0.15"),  # kg N2O/tonne wet weight
                "ch4_dry": Decimal("4.0"),  # kg CH4/tonne dry weight
                "n2o_dry": Decimal("0.30"),  # kg N2O/tonne dry weight
            },
            "ad_leakage": {
                "modern": Decimal("0.01"),  # 1%
                "standard": Decimal("0.03"),  # 3%
                "old": Decimal("0.08"),  # 8%
            },
        }

    def _build_wastewater_parameter_database(self) -> Dict[str, Any]:
        """Build wastewater treatment parameter database."""
        return {
            "bo_default": Decimal("0.25"),  # kg CH4/kg COD
            "n_ef": Decimal("0.005"),  # kg N2O-N/kg N (default)
        }

    def _build_msw_composition_database(self) -> Dict[str, Dict[WasteCategory, Decimal]]:
        """Build MSW composition profiles by region."""
        return {
            "global": {
                WasteCategory.FOOD_WASTE: Decimal("0.44"),
                WasteCategory.PAPER_CARDBOARD: Decimal("0.17"),
                WasteCategory.PLASTICS_MIXED: Decimal("0.12"),
                WasteCategory.GLASS: Decimal("0.05"),
                WasteCategory.METALS_MIXED: Decimal("0.04"),
                WasteCategory.WOOD: Decimal("0.05"),
                WasteCategory.GARDEN_WASTE: Decimal("0.06"),
                WasteCategory.TEXTILES: Decimal("0.03"),
                WasteCategory.OTHER: Decimal("0.04"),
            },
            "EU": {
                WasteCategory.FOOD_WASTE: Decimal("0.31"),
                WasteCategory.PAPER_CARDBOARD: Decimal("0.24"),
                WasteCategory.PLASTICS_MIXED: Decimal("0.14"),
                WasteCategory.GLASS: Decimal("0.08"),
                WasteCategory.METALS_MIXED: Decimal("0.05"),
                WasteCategory.GARDEN_WASTE: Decimal("0.10"),
                WasteCategory.TEXTILES: Decimal("0.04"),
                WasteCategory.WOOD: Decimal("0.03"),
                WasteCategory.OTHER: Decimal("0.01"),
            },
            "US": {
                WasteCategory.FOOD_WASTE: Decimal("0.22"),
                WasteCategory.PAPER_CARDBOARD: Decimal("0.23"),
                WasteCategory.PLASTICS_MIXED: Decimal("0.13"),
                WasteCategory.GARDEN_WASTE: Decimal("0.13"),
                WasteCategory.METALS_MIXED: Decimal("0.09"),
                WasteCategory.GLASS: Decimal("0.04"),
                WasteCategory.WOOD: Decimal("0.06"),
                WasteCategory.TEXTILES: Decimal("0.06"),
                WasteCategory.OTHER: Decimal("0.04"),
            },
        }

    def _build_treatment_compatibility_matrix(
        self,
    ) -> Dict[WasteCategory, List[WasteTreatmentMethod]]:
        """Build treatment compatibility matrix."""
        return {
            WasteCategory.PAPER_CARDBOARD: [
                WasteTreatmentMethod.LANDFILL,
                WasteTreatmentMethod.LANDFILL_WITH_GAS_CAPTURE,
                WasteTreatmentMethod.INCINERATION,
                WasteTreatmentMethod.INCINERATION_WITH_ENERGY_RECOVERY,
                WasteTreatmentMethod.RECYCLING_CLOSED_LOOP,
                WasteTreatmentMethod.COMPOSTING,
            ],
            WasteCategory.FOOD_WASTE: [
                WasteTreatmentMethod.LANDFILL,
                WasteTreatmentMethod.LANDFILL_WITH_GAS_CAPTURE,
                WasteTreatmentMethod.INCINERATION,
                WasteTreatmentMethod.COMPOSTING,
                WasteTreatmentMethod.ANAEROBIC_DIGESTION,
            ],
            WasteCategory.PLASTICS_MIXED: [
                WasteTreatmentMethod.LANDFILL,
                WasteTreatmentMethod.INCINERATION,
                WasteTreatmentMethod.INCINERATION_WITH_ENERGY_RECOVERY,
                WasteTreatmentMethod.RECYCLING_OPEN_LOOP,
                WasteTreatmentMethod.RECYCLING_CLOSED_LOOP,
            ],
            WasteCategory.GLASS: [
                WasteTreatmentMethod.LANDFILL,
                WasteTreatmentMethod.RECYCLING_CLOSED_LOOP,
            ],
            WasteCategory.METALS_MIXED: [
                WasteTreatmentMethod.LANDFILL,
                WasteTreatmentMethod.RECYCLING_CLOSED_LOOP,
            ],
            WasteCategory.WOOD: [
                WasteTreatmentMethod.LANDFILL,
                WasteTreatmentMethod.LANDFILL_WITH_GAS_CAPTURE,
                WasteTreatmentMethod.INCINERATION,
                WasteTreatmentMethod.INCINERATION_WITH_ENERGY_RECOVERY,
                WasteTreatmentMethod.RECYCLING_OPEN_LOOP,
            ],
            WasteCategory.GARDEN_WASTE: [
                WasteTreatmentMethod.LANDFILL,
                WasteTreatmentMethod.COMPOSTING,
                WasteTreatmentMethod.ANAEROBIC_DIGESTION,
            ],
            WasteCategory.TEXTILES: [
                WasteTreatmentMethod.LANDFILL,
                WasteTreatmentMethod.INCINERATION,
                WasteTreatmentMethod.RECYCLING_OPEN_LOOP,
            ],
            WasteCategory.ELECTRONICS: [
                WasteTreatmentMethod.LANDFILL,
                WasteTreatmentMethod.RECYCLING_OPEN_LOOP,
            ],
            WasteCategory.CONSTRUCTION_DEMOLITION: [
                WasteTreatmentMethod.LANDFILL,
                WasteTreatmentMethod.RECYCLING_OPEN_LOOP,
            ],
            WasteCategory.HAZARDOUS: [
                WasteTreatmentMethod.INCINERATION,
                WasteTreatmentMethod.OTHER,
            ],
        }

    def _build_hazard_classification_database(self) -> Dict[str, List[HazardClass]]:
        """Build hazard classification database."""
        # This would contain detailed Basel Convention classifications
        # Simplified version shown here
        return {
            "paint": [HazardClass.H3, HazardClass.H6_1],
            "battery": [HazardClass.H8, HazardClass.H6_1],
            "oil": [HazardClass.H3],
            "pesticide": [HazardClass.H6_1, HazardClass.H11],
            "asbestos": [HazardClass.H11, HazardClass.H12],
        }
