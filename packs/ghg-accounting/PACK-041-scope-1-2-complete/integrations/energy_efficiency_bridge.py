# -*- coding: utf-8 -*-
"""
EnergyEfficiencyBridge - Integration with Energy Efficiency Packs (PACK-041)
==============================================================================

This module provides integration with Energy Efficiency Packs (PACK-031
through PACK-040) to import audit findings, building assessments, quick
wins, ISO 50001 data, benchmarks, and link efficiency measures to GHG
emission reductions in the Scope 1-2 inventory.

Pack Integrations:
    PACK-031: Industrial Energy Audit findings and baselines
    PACK-032: Building Energy Assessment data and retrofits
    PACK-033: Quick Wins Identifier measures and savings
    PACK-034: ISO 50001 EnMS data and energy performance
    PACK-035: Energy Benchmark results and comparisons

Key Capability:
    Links efficiency measures to quantified emission reductions,
    enabling accurate tracking of GHG abatement from energy efficiency
    investments.

Zero-Hallucination:
    Avoided emissions calculated deterministically:
    avoided_tco2e = energy_saved_kwh * grid_ef / 1000

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-041 Scope 1-2 Complete
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EfficiencyPackSource(str, Enum):
    """Source energy efficiency packs."""

    PACK_031 = "PACK-031"
    PACK_032 = "PACK-032"
    PACK_033 = "PACK-033"
    PACK_034 = "PACK-034"
    PACK_035 = "PACK-035"

class MeasureCategory(str, Enum):
    """Energy efficiency measure categories."""

    LIGHTING = "lighting"
    HVAC = "hvac"
    BUILDING_ENVELOPE = "building_envelope"
    MOTORS_DRIVES = "motors_drives"
    COMPRESSED_AIR = "compressed_air"
    PROCESS_HEAT = "process_heat"
    CONTROLS_AUTOMATION = "controls_automation"
    RENEWABLE = "renewable"
    BEHAVIORAL = "behavioral"
    OTHER = "other"

class EmissionScope(str, Enum):
    """Emission scope affected by efficiency measure."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    BOTH = "both"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class AuditFinding(BaseModel):
    """Energy audit finding from PACK-031."""

    finding_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    category: MeasureCategory = Field(default=MeasureCategory.HVAC)
    description: str = Field(default="")
    estimated_savings_kwh: float = Field(default=0.0, ge=0.0)
    estimated_savings_therms: float = Field(default=0.0, ge=0.0)
    estimated_cost_savings_usd: float = Field(default=0.0, ge=0.0)
    implementation_cost_usd: float = Field(default=0.0, ge=0.0)
    payback_years: float = Field(default=0.0, ge=0.0)
    scope_affected: EmissionScope = Field(default=EmissionScope.SCOPE_2)

class BuildingAssessmentData(BaseModel):
    """Building assessment data from PACK-032."""

    assessment_id: str = Field(default_factory=_new_uuid)
    building_id: str = Field(default="")
    building_type: str = Field(default="office")
    floor_area_sqft: float = Field(default=0.0, ge=0.0)
    annual_energy_kwh: float = Field(default=0.0, ge=0.0)
    annual_gas_therms: float = Field(default=0.0, ge=0.0)
    eui_kbtu_per_sqft: float = Field(default=0.0, ge=0.0)
    energy_star_score: Optional[int] = Field(None, ge=1, le=100)
    retrofit_opportunities: List[Dict[str, Any]] = Field(default_factory=list)

class QuickWinMeasure(BaseModel):
    """Quick win measure from PACK-033."""

    measure_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    category: MeasureCategory = Field(default=MeasureCategory.BEHAVIORAL)
    description: str = Field(default="")
    estimated_savings_kwh: float = Field(default=0.0, ge=0.0)
    estimated_savings_therms: float = Field(default=0.0, ge=0.0)
    implementation_cost_usd: float = Field(default=0.0, ge=0.0)
    payback_months: float = Field(default=0.0, ge=0.0)
    complexity: str = Field(default="low")

class ISO50001Data(BaseModel):
    """ISO 50001 EnMS data from PACK-034."""

    enms_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    baseline_energy_kwh: float = Field(default=0.0, ge=0.0)
    current_energy_kwh: float = Field(default=0.0, ge=0.0)
    enpi_value: float = Field(default=0.0)
    enpi_unit: str = Field(default="kWh/unit")
    improvement_pct: float = Field(default=0.0)
    significant_energy_uses: List[Dict[str, Any]] = Field(default_factory=list)

class BenchmarkResult(BaseModel):
    """Benchmark result from PACK-035."""

    benchmark_id: str = Field(default_factory=_new_uuid)
    facility_id: str = Field(default="")
    benchmark_metric: str = Field(default="eui_kbtu_per_sqft")
    facility_value: float = Field(default=0.0)
    peer_median: float = Field(default=0.0)
    peer_top_quartile: float = Field(default=0.0)
    percentile_rank: int = Field(default=50, ge=1, le=100)

class EmissionReductionLink(BaseModel):
    """Link between efficiency measure and emission reduction."""

    link_id: str = Field(default_factory=_new_uuid)
    measure_id: str = Field(default="")
    measure_description: str = Field(default="")
    energy_saved_kwh: float = Field(default=0.0)
    energy_saved_therms: float = Field(default=0.0)
    scope1_reduction_tco2e: float = Field(default=0.0)
    scope2_reduction_tco2e: float = Field(default=0.0)
    total_reduction_tco2e: float = Field(default=0.0)
    grid_ef_kgco2_per_kwh: float = Field(default=0.417)
    gas_ef_kgco2_per_therm: float = Field(default=5.302)
    provenance_hash: str = Field(default="")

class ImportResult(BaseModel):
    """Result of importing data from an efficiency pack."""

    import_id: str = Field(default_factory=_new_uuid)
    source_pack: str = Field(default="")
    records_imported: int = Field(default=0)
    status: str = Field(default="success")
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=utcnow)

# ---------------------------------------------------------------------------
# EnergyEfficiencyBridge
# ---------------------------------------------------------------------------

class EnergyEfficiencyBridge:
    """Integration with Energy Efficiency Packs (PACK-031 to PACK-040).

    Imports energy audit findings, building assessments, quick wins,
    ISO 50001 data, and benchmarks, then links efficiency measures to
    quantified emission reductions in the Scope 1-2 inventory.

    Attributes:
        _audit_findings: Imported audit findings.
        _assessments: Imported building assessments.
        _quick_wins: Imported quick win measures.
        _enms_data: Imported ISO 50001 data.
        _benchmarks: Imported benchmark results.
        _emission_links: Linked emission reductions.

    Example:
        >>> bridge = EnergyEfficiencyBridge()
        >>> findings = bridge.import_audit_findings(pack_031_data)
        >>> link = bridge.link_efficiency_to_emissions(measures, inventory)
    """

    def __init__(self) -> None:
        """Initialize EnergyEfficiencyBridge."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._audit_findings: List[AuditFinding] = []
        self._assessments: List[BuildingAssessmentData] = []
        self._quick_wins: List[QuickWinMeasure] = []
        self._enms_data: List[ISO50001Data] = []
        self._benchmarks: List[BenchmarkResult] = []
        self._emission_links: List[EmissionReductionLink] = []

        self.logger.info("EnergyEfficiencyBridge initialized")

    # -------------------------------------------------------------------------
    # Import Methods
    # -------------------------------------------------------------------------

    def import_audit_findings(
        self,
        pack_031_data: Dict[str, Any],
    ) -> ImportResult:
        """Import industrial energy audit findings from PACK-031.

        Args:
            pack_031_data: Dict with findings list from PACK-031.

        Returns:
            ImportResult with import status.
        """
        start_time = time.monotonic()
        findings = pack_031_data.get("findings", [])

        for f in findings:
            finding = AuditFinding(
                facility_id=f.get("facility_id", ""),
                category=MeasureCategory(f.get("category", "hvac")),
                description=f.get("description", ""),
                estimated_savings_kwh=f.get("estimated_savings_kwh", 0.0),
                estimated_savings_therms=f.get("estimated_savings_therms", 0.0),
                estimated_cost_savings_usd=f.get("estimated_cost_savings_usd", 0.0),
                implementation_cost_usd=f.get("implementation_cost_usd", 0.0),
                payback_years=f.get("payback_years", 0.0),
                scope_affected=EmissionScope(f.get("scope_affected", "scope_2")),
            )
            self._audit_findings.append(finding)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        result = ImportResult(
            source_pack=EfficiencyPackSource.PACK_031.value,
            records_imported=len(findings),
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info("Imported %d audit findings from PACK-031", len(findings))
        return result

    def import_building_assessment(
        self,
        pack_032_data: Dict[str, Any],
    ) -> ImportResult:
        """Import building assessment data from PACK-032.

        Args:
            pack_032_data: Dict with assessments list from PACK-032.

        Returns:
            ImportResult with import status.
        """
        start_time = time.monotonic()
        assessments = pack_032_data.get("assessments", [])

        for a in assessments:
            assessment = BuildingAssessmentData(
                building_id=a.get("building_id", ""),
                building_type=a.get("building_type", "office"),
                floor_area_sqft=a.get("floor_area_sqft", 0.0),
                annual_energy_kwh=a.get("annual_energy_kwh", 0.0),
                annual_gas_therms=a.get("annual_gas_therms", 0.0),
                eui_kbtu_per_sqft=a.get("eui_kbtu_per_sqft", 0.0),
                energy_star_score=a.get("energy_star_score"),
                retrofit_opportunities=a.get("retrofit_opportunities", []),
            )
            self._assessments.append(assessment)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        result = ImportResult(
            source_pack=EfficiencyPackSource.PACK_032.value,
            records_imported=len(assessments),
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info("Imported %d building assessments from PACK-032", len(assessments))
        return result

    def import_quick_wins(
        self,
        pack_033_data: Dict[str, Any],
    ) -> ImportResult:
        """Import quick win measures from PACK-033.

        Args:
            pack_033_data: Dict with measures list from PACK-033.

        Returns:
            ImportResult with import status.
        """
        start_time = time.monotonic()
        measures = pack_033_data.get("measures", [])

        for m in measures:
            measure = QuickWinMeasure(
                facility_id=m.get("facility_id", ""),
                category=MeasureCategory(m.get("category", "behavioral")),
                description=m.get("description", ""),
                estimated_savings_kwh=m.get("estimated_savings_kwh", 0.0),
                estimated_savings_therms=m.get("estimated_savings_therms", 0.0),
                implementation_cost_usd=m.get("implementation_cost_usd", 0.0),
                payback_months=m.get("payback_months", 0.0),
                complexity=m.get("complexity", "low"),
            )
            self._quick_wins.append(measure)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        result = ImportResult(
            source_pack=EfficiencyPackSource.PACK_033.value,
            records_imported=len(measures),
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info("Imported %d quick wins from PACK-033", len(measures))
        return result

    def import_iso50001_data(
        self,
        pack_034_data: Dict[str, Any],
    ) -> ImportResult:
        """Import ISO 50001 EnMS data from PACK-034.

        Args:
            pack_034_data: Dict with enms_records list from PACK-034.

        Returns:
            ImportResult with import status.
        """
        start_time = time.monotonic()
        records = pack_034_data.get("enms_records", [])

        for r in records:
            enms = ISO50001Data(
                facility_id=r.get("facility_id", ""),
                baseline_energy_kwh=r.get("baseline_energy_kwh", 0.0),
                current_energy_kwh=r.get("current_energy_kwh", 0.0),
                enpi_value=r.get("enpi_value", 0.0),
                enpi_unit=r.get("enpi_unit", "kWh/unit"),
                improvement_pct=r.get("improvement_pct", 0.0),
                significant_energy_uses=r.get("significant_energy_uses", []),
            )
            self._enms_data.append(enms)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        result = ImportResult(
            source_pack=EfficiencyPackSource.PACK_034.value,
            records_imported=len(records),
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info("Imported %d ISO 50001 records from PACK-034", len(records))
        return result

    def import_benchmark_results(
        self,
        pack_035_data: Dict[str, Any],
    ) -> ImportResult:
        """Import energy benchmark results from PACK-035.

        Args:
            pack_035_data: Dict with benchmarks list from PACK-035.

        Returns:
            ImportResult with import status.
        """
        start_time = time.monotonic()
        benchmarks = pack_035_data.get("benchmarks", [])

        for b in benchmarks:
            benchmark = BenchmarkResult(
                facility_id=b.get("facility_id", ""),
                benchmark_metric=b.get("benchmark_metric", "eui_kbtu_per_sqft"),
                facility_value=b.get("facility_value", 0.0),
                peer_median=b.get("peer_median", 0.0),
                peer_top_quartile=b.get("peer_top_quartile", 0.0),
                percentile_rank=b.get("percentile_rank", 50),
            )
            self._benchmarks.append(benchmark)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        result = ImportResult(
            source_pack=EfficiencyPackSource.PACK_035.value,
            records_imported=len(benchmarks),
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info("Imported %d benchmarks from PACK-035", len(benchmarks))
        return result

    # -------------------------------------------------------------------------
    # Emission Reduction Linkage
    # -------------------------------------------------------------------------

    def link_efficiency_to_emissions(
        self,
        measures: List[Dict[str, Any]],
        inventory: Dict[str, Any],
    ) -> List[EmissionReductionLink]:
        """Link efficiency measures to quantified emission reductions.

        Deterministic calculation:
            scope2_reduction = kwh_saved * grid_ef / 1000
            scope1_reduction = therms_saved * gas_ef / 1000

        Args:
            measures: List of measures with savings data.
            inventory: Current GHG inventory context with grid_ef, gas_ef.

        Returns:
            List of EmissionReductionLink records.
        """
        grid_ef = Decimal(str(inventory.get("grid_ef_kgco2_per_kwh", 0.417)))
        gas_ef = Decimal(str(inventory.get("gas_ef_kgco2_per_therm", 5.302)))
        links: List[EmissionReductionLink] = []

        for measure in measures:
            kwh_saved = Decimal(str(measure.get("energy_saved_kwh", 0.0)))
            therms_saved = Decimal(str(measure.get("energy_saved_therms", 0.0)))

            scope2_reduction = (kwh_saved * grid_ef / Decimal("1000")).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            scope1_reduction = (therms_saved * gas_ef / Decimal("1000")).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            total = scope2_reduction + scope1_reduction

            link = EmissionReductionLink(
                measure_id=measure.get("measure_id", _new_uuid()),
                measure_description=measure.get("description", ""),
                energy_saved_kwh=float(kwh_saved),
                energy_saved_therms=float(therms_saved),
                scope1_reduction_tco2e=float(scope1_reduction),
                scope2_reduction_tco2e=float(scope2_reduction),
                total_reduction_tco2e=float(total),
                grid_ef_kgco2_per_kwh=float(grid_ef),
                gas_ef_kgco2_per_therm=float(gas_ef),
            )
            link.provenance_hash = _compute_hash(link)
            links.append(link)
            self._emission_links.append(link)

        total_reduction = sum(l.total_reduction_tco2e for l in links)
        self.logger.info(
            "Linked %d measures to %.3f tCO2e total reduction",
            len(links), total_reduction,
        )
        return links

    def calculate_avoided_emissions(
        self,
        measures: List[Dict[str, Any]],
        grid_ef_kgco2_per_kwh: float = 0.417,
        gas_ef_kgco2_per_therm: float = 5.302,
    ) -> Decimal:
        """Calculate total avoided emissions from efficiency measures.

        Args:
            measures: List of measures with energy_saved_kwh, energy_saved_therms.
            grid_ef_kgco2_per_kwh: Grid emission factor.
            gas_ef_kgco2_per_therm: Natural gas emission factor.

        Returns:
            Total avoided emissions in tCO2e (Decimal).
        """
        grid_ef = Decimal(str(grid_ef_kgco2_per_kwh))
        gas_ef = Decimal(str(gas_ef_kgco2_per_therm))
        total = Decimal("0")

        for measure in measures:
            kwh = Decimal(str(measure.get("energy_saved_kwh", 0)))
            therms = Decimal(str(measure.get("energy_saved_therms", 0)))
            avoided = (kwh * grid_ef + therms * gas_ef) / Decimal("1000")
            total += avoided

        result = total.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        self.logger.info("Total avoided emissions: %.3f tCO2e from %d measures", float(result), len(measures))
        return result
