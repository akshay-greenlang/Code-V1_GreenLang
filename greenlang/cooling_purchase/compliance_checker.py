"""
ComplianceCheckerEngine - AGENT-MRV-012 Cooling Purchase Agent

7 frameworks × 12 requirements = 84 total compliance checks.
Thread-safe singleton pattern.
"""

import threading
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime
from enum import Enum

try:
    from greenlang.cooling_purchase.models import (
        ComplianceCheckResult,
        ComplianceStatus,
        CalculationResult,
        CoolingTechnology,
        Refrigerant,
        DataQualityTier,
        GWPSource
    )
except ImportError:
    from models import (
        ComplianceCheckResult,
        ComplianceStatus,
        CalculationResult,
        CoolingTechnology,
        Refrigerant,
        DataQualityTier,
        GWPSource
    )

try:
    from greenlang.cooling_purchase.config import config
except ImportError:
    config = None

try:
    from greenlang.cooling_purchase.metrics import metrics
except ImportError:
    metrics = None

try:
    from greenlang.cooling_purchase.provenance import ProvenanceTracker
except ImportError:
    ProvenanceTracker = None


class ComplianceCheckerEngine:
    """Thread-safe singleton compliance checker for cooling purchase emissions."""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        with self._lock:
            if self._initialized:
                return
            self._frameworks = self._build_frameworks()
            self._ashrae_cop_minimums = self._build_ashrae_minimums()
            self._fgas_phase_down = self._build_fgas_phase_down()
            self._initialized = True

    @classmethod
    def reset(cls):
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def _build_frameworks(self) -> Dict[str, List[str]]:
        """Build 7 frameworks with 12 requirements each."""
        return {
            "ghg_protocol": [
                "cooling_output_documented",
                "cop_source_identified",
                "emission_factor_sourced",
                "uncertainty_assessed",
                "scope2_classification",
                "provenance_chain",
                "gas_breakdown",
                "calculation_tier_stated",
                "technology_documented",
                "temporal_representativeness",
                "geographic_representativeness",
                "completeness"
            ],
            "iso_14064": [
                "organizational_boundary",
                "scope2_quantification",
                "emission_factors_traceable",
                "uncertainty_reported",
                "base_year_comparison",
                "data_quality_assessment",
                "methodology_documented",
                "significant_sources",
                "exclusions_justified",
                "verification_ready",
                "gwp_source_specified",
                "reporting_period_defined"
            ],
            "csrd_esrs": [
                "scope2_disclosed",
                "location_based_reported",
                "methodology_transparency",
                "ef_data_sources",
                "cooling_energy_breakdown",
                "technology_disclosure",
                "district_cooling_identified",
                "year_over_year_comparison",
                "assurance_ready",
                "double_materiality",
                "transition_plan_alignment",
                "value_chain_consideration"
            ],
            "cdp": [
                "scope2_reported",
                "calculation_methodology",
                "ef_sources_listed",
                "verification_status",
                "emission_trends",
                "boundary_description",
                "cooling_specific_disclosure",
                "exclusion_rationale",
                "data_quality_description",
                "energy_consumption_reported",
                "target_coverage",
                "renewable_cooling"
            ],
            "sbti": [
                "scope2_included",
                "location_based_minimum",
                "market_based_if_available",
                "base_year_emissions",
                "target_year_pathway",
                "annual_reduction",
                "fossil_vs_renewable",
                "cooling_efficiency_trend",
                "absolute_vs_intensity",
                "cop_improvement_tracking",
                "supplier_engagement",
                "progress_reporting"
            ],
            "ashrae_90_1": [
                "minimum_cop_met",
                "iplv_calculated",
                "part_load_documented",
                "condenser_type_specified",
                "capacity_documented",
                "auxiliary_energy_accounted",
                "economizer_hours",
                "refrigerant_type_specified",
                "leak_detection",
                "maintenance_schedule",
                "efficiency_degradation",
                "commissioning_verified"
            ],
            "eu_fgas": [
                "refrigerant_identified",
                "gwp_recorded",
                "charge_quantity",
                "leak_rate_monitored",
                "leak_check_frequency",
                "recovery_plan",
                "phase_down_compliance",
                "alternative_assessment",
                "logbook_maintained",
                "certified_personnel",
                "reporting_threshold",
                "containment_measures"
            ]
        }

    def _build_ashrae_minimums(self) -> Dict[str, Dict[str, Decimal]]:
        """ASHRAE 90.1 minimum COP by technology and capacity."""
        return {
            "chiller_water_cooled": {
                "under_150": Decimal("5.5"),
                "150_to_300": Decimal("6.0"),
                "over_300": Decimal("6.4")
            },
            "chiller_air_cooled": {
                "under_150": Decimal("2.8"),
                "150_to_300": Decimal("3.0"),
                "over_300": Decimal("3.1")
            },
            "absorption_chiller_single": {
                "all": Decimal("0.7")
            },
            "absorption_chiller_double": {
                "all": Decimal("1.0")
            },
            "district_cooling": {
                "all": Decimal("4.0")
            },
            "geothermal_cooling": {
                "all": Decimal("3.5")
            },
            "evaporative_cooling": {
                "all": Decimal("2.5")
            }
        }

    def _build_fgas_phase_down(self) -> Dict[int, Decimal]:
        """EU F-Gas phase-down schedule (% of 2015 baseline)."""
        return {
            2015: Decimal("100"),
            2018: Decimal("63"),
            2021: Decimal("45"),
            2024: Decimal("31"),
            2027: Decimal("24"),
            2030: Decimal("21"),
            2033: Decimal("15"),
            2036: Decimal("10")
        }

    # ==================== PUBLIC API ====================

    def check_compliance(
        self,
        calculation_result: CalculationResult,
        frameworks: Optional[List[str]] = None
    ) -> List[ComplianceCheckResult]:
        """Check compliance against multiple frameworks."""
        if frameworks is None:
            frameworks = list(self._frameworks.keys())

        results = []
        for framework in frameworks:
            if framework not in self._frameworks:
                continue
            result = self.check_single_framework(calculation_result, framework)
            results.append(result)

        if metrics:
            for result in results:
                metrics.record_compliance_check(
                    framework=result.framework,
                    status=result.status.value,
                    met_count=result.requirements_met,
                    total_count=result.total_requirements
                )

        return results

    def check_single_framework(
        self,
        calculation_result: CalculationResult,
        framework: str
    ) -> ComplianceCheckResult:
        """Check compliance for a single framework."""
        method_map = {
            "ghg_protocol": self._check_ghg_protocol,
            "iso_14064": self._check_iso_14064,
            "csrd_esrs": self._check_csrd_esrs,
            "cdp": self._check_cdp,
            "sbti": self._check_sbti,
            "ashrae_90_1": self._check_ashrae_90_1,
            "eu_fgas": self._check_eu_fgas
        }

        if framework not in method_map:
            raise ValueError(f"Unknown framework: {framework}")

        return method_map[framework](calculation_result)

    def get_all_frameworks(self) -> List[str]:
        """Get list of all framework identifiers."""
        return list(self._frameworks.keys())

    def get_framework_requirements(self, framework: str) -> List[Dict[str, str]]:
        """Get requirements for a framework."""
        if framework not in self._frameworks:
            raise ValueError(f"Unknown framework: {framework}")

        return [
            {"requirement": req, "framework": framework}
            for req in self._frameworks[framework]
        ]

    def get_total_requirements_count(self) -> int:
        """Get total number of requirements across all frameworks."""
        return sum(len(reqs) for reqs in self._frameworks.values())

    def get_ashrae_minimum_cop(
        self,
        technology: CoolingTechnology,
        capacity_tons: Decimal
    ) -> Decimal:
        """Get ASHRAE 90.1 minimum COP for technology/capacity."""
        tech_key = technology.value if isinstance(technology, CoolingTechnology) else technology

        if tech_key not in self._ashrae_cop_minimums:
            return Decimal("0")

        cop_table = self._ashrae_cop_minimums[tech_key]

        if "all" in cop_table:
            return cop_table["all"]

        if capacity_tons < 150:
            return cop_table.get("under_150", Decimal("0"))
        elif capacity_tons <= 300:
            return cop_table.get("150_to_300", Decimal("0"))
        else:
            return cop_table.get("over_300", Decimal("0"))

    def check_ashrae_minimum_cop(
        self,
        technology: CoolingTechnology,
        cop_used: Decimal,
        capacity_tons: Decimal
    ) -> bool:
        """Check if COP meets ASHRAE 90.1 minimum."""
        minimum = self.get_ashrae_minimum_cop(technology, capacity_tons)
        return cop_used >= minimum

    def check_phase_down_compliance(
        self,
        refrigerant: Refrigerant,
        year: int
    ) -> bool:
        """Check EU F-Gas phase-down compliance."""
        if year not in self._fgas_phase_down:
            latest_year = max(y for y in self._fgas_phase_down.keys() if y <= year)
            allowed_percent = self._fgas_phase_down.get(latest_year, Decimal("0"))
        else:
            allowed_percent = self._fgas_phase_down[year]

        gwp = self._get_refrigerant_gwp(refrigerant)
        if gwp < 150:
            return True
        if gwp < 2500:
            return allowed_percent >= Decimal("31")

        return False

    def calculate_co2e_charge(
        self,
        refrigerant: Refrigerant,
        charge_kg: Decimal,
        gwp_source: GWPSource
    ) -> Decimal:
        """Calculate CO2e from refrigerant charge."""
        gwp = self._get_refrigerant_gwp(refrigerant)
        return charge_kg * gwp / Decimal("1000")

    def check_reporting_threshold(self, co2e_charge: Decimal) -> bool:
        """Check if charge exceeds 500 tCO2e reporting threshold."""
        return co2e_charge >= Decimal("500")

    def calculate_compliance_score(self, met: int, total: int) -> Decimal:
        """Calculate compliance score (0-100%)."""
        if total == 0:
            return Decimal("0")
        return (Decimal(met) / Decimal(total)) * Decimal("100")

    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine metadata."""
        return {
            "engine": "ComplianceCheckerEngine",
            "version": "1.0.0",
            "total_frameworks": len(self._frameworks),
            "total_requirements": self.get_total_requirements_count(),
            "frameworks": self.get_all_frameworks()
        }

    # ==================== FRAMEWORK CHECKS ====================

    def _check_ghg_protocol(self, result: CalculationResult) -> ComplianceCheckResult:
        """Check GHG Protocol Scope 2 Guidance compliance."""
        requirements = self._frameworks["ghg_protocol"]
        met_requirements = []
        unmet_requirements = []

        cooling_output_documented = self._has_value(result, "cooling_output_kwh_th")
        cop_source_identified = self._has_value(result, "cop_used")
        emission_factor_sourced = self._has_value(result, "emission_factor_kg_co2e_kwh")
        uncertainty_assessed = self._has_value(result, "uncertainty_percent")
        scope2_classification = self._has_metadata(result, "scope") or self._has_value(result, "scope")
        provenance_chain = self._has_value(result, "provenance_hash")
        gas_breakdown = self._has_gas_breakdown(result)
        calculation_tier_stated = self._has_value(result, "calculation_type")
        technology_documented = self._has_value(result, "calculation_type") or self._has_metadata(result, "technology")
        temporal_representativeness = self._has_metadata(result, "year") or self._has_metadata(result, "period")
        geographic_representativeness = self._has_metadata(result, "location") or self._has_metadata(result, "country")
        completeness = self._has_value(result, "total_emissions_kg_co2e")

        checks = [
            ("cooling_output_documented", cooling_output_documented),
            ("cop_source_identified", cop_source_identified),
            ("emission_factor_sourced", emission_factor_sourced),
            ("uncertainty_assessed", uncertainty_assessed),
            ("scope2_classification", scope2_classification),
            ("provenance_chain", provenance_chain),
            ("gas_breakdown", gas_breakdown),
            ("calculation_tier_stated", calculation_tier_stated),
            ("technology_documented", technology_documented),
            ("temporal_representativeness", temporal_representativeness),
            ("geographic_representativeness", geographic_representativeness),
            ("completeness", completeness)
        ]

        for req, is_met in checks:
            if is_met:
                met_requirements.append(req)
            else:
                unmet_requirements.append(req)

        status = self._determine_status(len(met_requirements), len(requirements))

        return ComplianceCheckResult(
            framework="ghg_protocol",
            status=status,
            requirements_met=len(met_requirements),
            total_requirements=len(requirements),
            met_requirements=met_requirements,
            unmet_requirements=unmet_requirements,
            compliance_score=self.calculate_compliance_score(len(met_requirements), len(requirements)),
            checked_at=datetime.utcnow()
        )

    def _check_iso_14064(self, result: CalculationResult) -> ComplianceCheckResult:
        """Check ISO 14064-1 compliance."""
        requirements = self._frameworks["iso_14064"]
        met_requirements = []
        unmet_requirements = []

        organizational_boundary = self._has_metadata(result, "boundary") or self._has_metadata(result, "organization")
        scope2_quantification = self._has_value(result, "total_emissions_kg_co2e")
        emission_factors_traceable = self._has_value(result, "emission_factor_kg_co2e_kwh")
        uncertainty_reported = self._has_value(result, "uncertainty_percent")
        base_year_comparison = self._has_metadata(result, "base_year")
        data_quality_assessment = self._has_value(result, "data_quality_score")
        methodology_documented = self._has_value(result, "calculation_type")
        significant_sources = self._has_value(result, "electricity_input_kwh")
        exclusions_justified = self._has_metadata(result, "exclusions")
        verification_ready = provenance_chain = self._has_value(result, "provenance_hash")
        gwp_source_specified = self._has_metadata(result, "gwp_source")
        reporting_period_defined = self._has_metadata(result, "period") or self._has_metadata(result, "year")

        checks = [
            ("organizational_boundary", organizational_boundary),
            ("scope2_quantification", scope2_quantification),
            ("emission_factors_traceable", emission_factors_traceable),
            ("uncertainty_reported", uncertainty_reported),
            ("base_year_comparison", base_year_comparison),
            ("data_quality_assessment", data_quality_assessment),
            ("methodology_documented", methodology_documented),
            ("significant_sources", significant_sources),
            ("exclusions_justified", exclusions_justified),
            ("verification_ready", verification_ready),
            ("gwp_source_specified", gwp_source_specified),
            ("reporting_period_defined", reporting_period_defined)
        ]

        for req, is_met in checks:
            if is_met:
                met_requirements.append(req)
            else:
                unmet_requirements.append(req)

        status = self._determine_status(len(met_requirements), len(requirements))

        return ComplianceCheckResult(
            framework="iso_14064",
            status=status,
            requirements_met=len(met_requirements),
            total_requirements=len(requirements),
            met_requirements=met_requirements,
            unmet_requirements=unmet_requirements,
            compliance_score=self.calculate_compliance_score(len(met_requirements), len(requirements)),
            checked_at=datetime.utcnow()
        )

    def _check_csrd_esrs(self, result: CalculationResult) -> ComplianceCheckResult:
        """Check CSRD ESRS E1 compliance."""
        requirements = self._frameworks["csrd_esrs"]
        met_requirements = []
        unmet_requirements = []

        scope2_disclosed = self._has_value(result, "total_emissions_kg_co2e")
        location_based_reported = self._has_value(result, "emission_factor_kg_co2e_kwh")
        methodology_transparency = self._has_value(result, "calculation_type")
        ef_data_sources = self._has_metadata(result, "ef_source")
        cooling_energy_breakdown = self._has_value(result, "electricity_input_kwh")
        technology_disclosure = self._has_metadata(result, "technology")
        district_cooling_identified = self._has_metadata(result, "district_cooling")
        year_over_year_comparison = self._has_metadata(result, "trend")
        assurance_ready = self._has_value(result, "provenance_hash")
        double_materiality = self._has_metadata(result, "materiality")
        transition_plan_alignment = self._has_metadata(result, "transition_plan")
        value_chain_consideration = self._has_metadata(result, "value_chain")

        checks = [
            ("scope2_disclosed", scope2_disclosed),
            ("location_based_reported", location_based_reported),
            ("methodology_transparency", methodology_transparency),
            ("ef_data_sources", ef_data_sources),
            ("cooling_energy_breakdown", cooling_energy_breakdown),
            ("technology_disclosure", technology_disclosure),
            ("district_cooling_identified", district_cooling_identified),
            ("year_over_year_comparison", year_over_year_comparison),
            ("assurance_ready", assurance_ready),
            ("double_materiality", double_materiality),
            ("transition_plan_alignment", transition_plan_alignment),
            ("value_chain_consideration", value_chain_consideration)
        ]

        for req, is_met in checks:
            if is_met:
                met_requirements.append(req)
            else:
                unmet_requirements.append(req)

        status = self._determine_status(len(met_requirements), len(requirements))

        return ComplianceCheckResult(
            framework="csrd_esrs",
            status=status,
            requirements_met=len(met_requirements),
            total_requirements=len(requirements),
            met_requirements=met_requirements,
            unmet_requirements=unmet_requirements,
            compliance_score=self.calculate_compliance_score(len(met_requirements), len(requirements)),
            checked_at=datetime.utcnow()
        )

    def _check_cdp(self, result: CalculationResult) -> ComplianceCheckResult:
        """Check CDP Climate Change questionnaire compliance."""
        requirements = self._frameworks["cdp"]
        met_requirements = []
        unmet_requirements = []

        scope2_reported = self._has_value(result, "total_emissions_kg_co2e")
        calculation_methodology = self._has_value(result, "calculation_type")
        ef_sources_listed = self._has_metadata(result, "ef_source")
        verification_status = self._has_metadata(result, "verification")
        emission_trends = self._has_metadata(result, "trend")
        boundary_description = self._has_metadata(result, "boundary")
        cooling_specific_disclosure = self._has_value(result, "cooling_output_kwh_th")
        exclusion_rationale = self._has_metadata(result, "exclusions")
        data_quality_description = self._has_value(result, "data_quality_score")
        energy_consumption_reported = self._has_value(result, "electricity_input_kwh")
        target_coverage = self._has_metadata(result, "target")
        renewable_cooling = self._has_metadata(result, "renewable")

        checks = [
            ("scope2_reported", scope2_reported),
            ("calculation_methodology", calculation_methodology),
            ("ef_sources_listed", ef_sources_listed),
            ("verification_status", verification_status),
            ("emission_trends", emission_trends),
            ("boundary_description", boundary_description),
            ("cooling_specific_disclosure", cooling_specific_disclosure),
            ("exclusion_rationale", exclusion_rationale),
            ("data_quality_description", data_quality_description),
            ("energy_consumption_reported", energy_consumption_reported),
            ("target_coverage", target_coverage),
            ("renewable_cooling", renewable_cooling)
        ]

        for req, is_met in checks:
            if is_met:
                met_requirements.append(req)
            else:
                unmet_requirements.append(req)

        status = self._determine_status(len(met_requirements), len(requirements))

        return ComplianceCheckResult(
            framework="cdp",
            status=status,
            requirements_met=len(met_requirements),
            total_requirements=len(requirements),
            met_requirements=met_requirements,
            unmet_requirements=unmet_requirements,
            compliance_score=self.calculate_compliance_score(len(met_requirements), len(requirements)),
            checked_at=datetime.utcnow()
        )

    def _check_sbti(self, result: CalculationResult) -> ComplianceCheckResult:
        """Check Science Based Targets initiative compliance."""
        requirements = self._frameworks["sbti"]
        met_requirements = []
        unmet_requirements = []

        scope2_included = self._has_value(result, "total_emissions_kg_co2e")
        location_based_minimum = self._has_value(result, "emission_factor_kg_co2e_kwh")
        market_based_if_available = self._has_metadata(result, "market_based")
        base_year_emissions = self._has_metadata(result, "base_year")
        target_year_pathway = self._has_metadata(result, "target_year")
        annual_reduction = self._has_metadata(result, "reduction_rate")
        fossil_vs_renewable = self._has_metadata(result, "renewable")
        cooling_efficiency_trend = self._has_value(result, "cop_used")
        absolute_vs_intensity = self._has_metadata(result, "intensity")
        cop_improvement_tracking = self._has_metadata(result, "cop_trend")
        supplier_engagement = self._has_metadata(result, "supplier")
        progress_reporting = self._has_metadata(result, "progress")

        checks = [
            ("scope2_included", scope2_included),
            ("location_based_minimum", location_based_minimum),
            ("market_based_if_available", market_based_if_available),
            ("base_year_emissions", base_year_emissions),
            ("target_year_pathway", target_year_pathway),
            ("annual_reduction", annual_reduction),
            ("fossil_vs_renewable", fossil_vs_renewable),
            ("cooling_efficiency_trend", cooling_efficiency_trend),
            ("absolute_vs_intensity", absolute_vs_intensity),
            ("cop_improvement_tracking", cop_improvement_tracking),
            ("supplier_engagement", supplier_engagement),
            ("progress_reporting", progress_reporting)
        ]

        for req, is_met in checks:
            if is_met:
                met_requirements.append(req)
            else:
                unmet_requirements.append(req)

        status = self._determine_status(len(met_requirements), len(requirements))

        return ComplianceCheckResult(
            framework="sbti",
            status=status,
            requirements_met=len(met_requirements),
            total_requirements=len(requirements),
            met_requirements=met_requirements,
            unmet_requirements=unmet_requirements,
            compliance_score=self.calculate_compliance_score(len(met_requirements), len(requirements)),
            checked_at=datetime.utcnow()
        )

    def _check_ashrae_90_1(self, result: CalculationResult) -> ComplianceCheckResult:
        """Check ASHRAE 90.1 equipment efficiency compliance."""
        requirements = self._frameworks["ashrae_90_1"]
        met_requirements = []
        unmet_requirements = []

        minimum_cop_met = self._check_ashrae_cop_requirement(result)
        iplv_calculated = self._has_metadata(result, "iplv")
        part_load_documented = self._has_metadata(result, "part_load")
        condenser_type_specified = self._has_metadata(result, "condenser_type")
        capacity_documented = self._has_metadata(result, "capacity_tons")
        auxiliary_energy_accounted = self._has_metadata(result, "auxiliary_energy")
        economizer_hours = self._has_metadata(result, "economizer")
        refrigerant_type_specified = self._has_metadata(result, "refrigerant")
        leak_detection = self._has_metadata(result, "leak_detection")
        maintenance_schedule = self._has_metadata(result, "maintenance")
        efficiency_degradation = self._has_metadata(result, "degradation")
        commissioning_verified = self._has_metadata(result, "commissioning")

        checks = [
            ("minimum_cop_met", minimum_cop_met),
            ("iplv_calculated", iplv_calculated),
            ("part_load_documented", part_load_documented),
            ("condenser_type_specified", condenser_type_specified),
            ("capacity_documented", capacity_documented),
            ("auxiliary_energy_accounted", auxiliary_energy_accounted),
            ("economizer_hours", economizer_hours),
            ("refrigerant_type_specified", refrigerant_type_specified),
            ("leak_detection", leak_detection),
            ("maintenance_schedule", maintenance_schedule),
            ("efficiency_degradation", efficiency_degradation),
            ("commissioning_verified", commissioning_verified)
        ]

        for req, is_met in checks:
            if is_met:
                met_requirements.append(req)
            else:
                unmet_requirements.append(req)

        status = self._determine_status(len(met_requirements), len(requirements))

        return ComplianceCheckResult(
            framework="ashrae_90_1",
            status=status,
            requirements_met=len(met_requirements),
            total_requirements=len(requirements),
            met_requirements=met_requirements,
            unmet_requirements=unmet_requirements,
            compliance_score=self.calculate_compliance_score(len(met_requirements), len(requirements)),
            checked_at=datetime.utcnow()
        )

    def _check_eu_fgas(self, result: CalculationResult) -> ComplianceCheckResult:
        """Check EU F-Gas Regulation compliance."""
        requirements = self._frameworks["eu_fgas"]
        met_requirements = []
        unmet_requirements = []

        refrigerant_identified = self._has_metadata(result, "refrigerant")
        gwp_recorded = self._has_metadata(result, "gwp")
        charge_quantity = self._has_metadata(result, "charge_kg")
        leak_rate_monitored = self._has_metadata(result, "leak_rate")
        leak_check_frequency = self._has_metadata(result, "leak_check_freq")
        recovery_plan = self._has_metadata(result, "recovery_plan")
        phase_down_compliance = self._check_fgas_phase_down_requirement(result)
        alternative_assessment = self._has_metadata(result, "alternatives")
        logbook_maintained = self._has_metadata(result, "logbook")
        certified_personnel = self._has_metadata(result, "certified")
        reporting_threshold = self._check_fgas_reporting_threshold(result)
        containment_measures = self._has_metadata(result, "containment")

        checks = [
            ("refrigerant_identified", refrigerant_identified),
            ("gwp_recorded", gwp_recorded),
            ("charge_quantity", charge_quantity),
            ("leak_rate_monitored", leak_rate_monitored),
            ("leak_check_frequency", leak_check_frequency),
            ("recovery_plan", recovery_plan),
            ("phase_down_compliance", phase_down_compliance),
            ("alternative_assessment", alternative_assessment),
            ("logbook_maintained", logbook_maintained),
            ("certified_personnel", certified_personnel),
            ("reporting_threshold", reporting_threshold),
            ("containment_measures", containment_measures)
        ]

        for req, is_met in checks:
            if is_met:
                met_requirements.append(req)
            else:
                unmet_requirements.append(req)

        status = self._determine_status(len(met_requirements), len(requirements))

        return ComplianceCheckResult(
            framework="eu_fgas",
            status=status,
            requirements_met=len(met_requirements),
            total_requirements=len(requirements),
            met_requirements=met_requirements,
            unmet_requirements=unmet_requirements,
            compliance_score=self.calculate_compliance_score(len(met_requirements), len(requirements)),
            checked_at=datetime.utcnow()
        )

    # ==================== HELPER METHODS ====================

    def _has_value(self, result: CalculationResult, field: str) -> bool:
        """Check if result has a non-zero/non-None value for field."""
        try:
            if isinstance(result, dict):
                val = result.get(field)
            else:
                val = getattr(result, field, None)

            if val is None:
                return False
            if isinstance(val, (int, float, Decimal)):
                return val > 0
            if isinstance(val, str):
                return len(val) > 0
            if isinstance(val, list):
                return len(val) > 0
            return True
        except:
            return False

    def _has_metadata(self, result: CalculationResult, key: str) -> bool:
        """Check if result has metadata key."""
        try:
            if isinstance(result, dict):
                metadata = result.get("metadata", {})
            else:
                metadata = getattr(result, "metadata", {})

            if isinstance(metadata, dict):
                return key in metadata and metadata[key] is not None
            return False
        except:
            return False

    def _has_gas_breakdown(self, result: CalculationResult) -> bool:
        """Check if result has gas breakdown."""
        try:
            if isinstance(result, dict):
                breakdown = result.get("gas_breakdown", {})
            else:
                breakdown = getattr(result, "gas_breakdown", {})

            if isinstance(breakdown, dict):
                return len(breakdown) > 0
            return False
        except:
            return False

    def _check_ashrae_cop_requirement(self, result: CalculationResult) -> bool:
        """Check if COP meets ASHRAE minimum."""
        try:
            if isinstance(result, dict):
                metadata = result.get("metadata", {})
            else:
                metadata = getattr(result, "metadata", {})

            if not isinstance(metadata, dict):
                return False

            technology = metadata.get("technology")
            capacity_tons = metadata.get("capacity_tons")
            cop_used = self._get_result_value(result, "cop_used")

            if not all([technology, capacity_tons, cop_used]):
                return False

            if isinstance(technology, str):
                try:
                    technology = CoolingTechnology(technology)
                except:
                    return False

            return self.check_ashrae_minimum_cop(
                technology,
                Decimal(str(cop_used)),
                Decimal(str(capacity_tons))
            )
        except:
            return False

    def _check_fgas_phase_down_requirement(self, result: CalculationResult) -> bool:
        """Check F-Gas phase-down compliance."""
        try:
            if isinstance(result, dict):
                metadata = result.get("metadata", {})
            else:
                metadata = getattr(result, "metadata", {})

            if not isinstance(metadata, dict):
                return False

            refrigerant = metadata.get("refrigerant")
            year = metadata.get("year")

            if not all([refrigerant, year]):
                return False

            if isinstance(refrigerant, str):
                try:
                    refrigerant = Refrigerant(refrigerant)
                except:
                    return False

            return self.check_phase_down_compliance(refrigerant, int(year))
        except:
            return False

    def _check_fgas_reporting_threshold(self, result: CalculationResult) -> bool:
        """Check F-Gas reporting threshold."""
        try:
            if isinstance(result, dict):
                metadata = result.get("metadata", {})
            else:
                metadata = getattr(result, "metadata", {})

            if not isinstance(metadata, dict):
                return True

            refrigerant = metadata.get("refrigerant")
            charge_kg = metadata.get("charge_kg")
            gwp_source = metadata.get("gwp_source", GWPSource.AR6)

            if not all([refrigerant, charge_kg]):
                return True

            if isinstance(refrigerant, str):
                try:
                    refrigerant = Refrigerant(refrigerant)
                except:
                    return True

            if isinstance(gwp_source, str):
                try:
                    gwp_source = GWPSource(gwp_source)
                except:
                    gwp_source = GWPSource.AR6

            co2e_charge = self.calculate_co2e_charge(
                refrigerant,
                Decimal(str(charge_kg)),
                gwp_source
            )

            return self.check_reporting_threshold(co2e_charge)
        except:
            return True

    def _get_result_value(self, result: CalculationResult, field: str) -> Any:
        """Get value from result (dict or object)."""
        if isinstance(result, dict):
            return result.get(field)
        return getattr(result, field, None)

    def _get_refrigerant_gwp(self, refrigerant: Refrigerant) -> Decimal:
        """Get GWP for refrigerant (simplified lookup)."""
        gwp_map = {
            Refrigerant.R22: Decimal("1810"),
            Refrigerant.R134A: Decimal("1430"),
            Refrigerant.R404A: Decimal("3922"),
            Refrigerant.R407C: Decimal("1774"),
            Refrigerant.R410A: Decimal("2088"),
            Refrigerant.R32: Decimal("675"),
            Refrigerant.R1234YF: Decimal("4"),
            Refrigerant.R1234ZE: Decimal("7"),
            Refrigerant.R290: Decimal("3"),
            Refrigerant.R600A: Decimal("3"),
            Refrigerant.R717: Decimal("0"),
            Refrigerant.R744: Decimal("1"),
            Refrigerant.R718: Decimal("0")
        }
        return gwp_map.get(refrigerant, Decimal("0"))

    def _determine_status(self, met: int, total: int) -> ComplianceStatus:
        """Determine compliance status from met/total."""
        if total == 0:
            return ComplianceStatus.NON_COMPLIANT

        if met == total:
            return ComplianceStatus.COMPLIANT
        elif met >= total * 0.67:
            return ComplianceStatus.PARTIAL
        else:
            return ComplianceStatus.NON_COMPLIANT


# ==================== MODULE-LEVEL FUNCTIONS ====================

_engine_instance: Optional[ComplianceCheckerEngine] = None
_engine_lock = threading.RLock()


def get_compliance_checker() -> ComplianceCheckerEngine:
    """Get or create singleton ComplianceCheckerEngine instance."""
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = ComplianceCheckerEngine()
        return _engine_instance


def reset_compliance_checker():
    """Reset singleton instance (for testing)."""
    global _engine_instance
    with _engine_lock:
        _engine_instance = None
    ComplianceCheckerEngine.reset()
