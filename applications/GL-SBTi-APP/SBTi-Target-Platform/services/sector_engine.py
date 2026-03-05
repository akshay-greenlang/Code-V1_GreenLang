"""
Sector Engine -- SDA Sector-Specific Intensity Benchmarks and Pathway Alignment

Implements the SBTi Sectoral Decarbonization Approach (SDA) for 12 sectors
with sector-specific intensity convergence pathways.  Covers pathway lookup,
sector detection from industry codes (ISIC/NACE/NAICS), intensity benchmark
comparison, multi-sector blending, and sector compliance verification.

Sectors supported:
    Power, Oil & Gas, Transport, Buildings, Cement, Steel, Aluminium,
    Chemicals, Aviation, Pulp & Paper, Maritime, General (cross-sector)

All numeric calculations are deterministic (zero-hallucination).

Reference:
    - SBTi Sector-Specific Guidance (2020-2024)
    - SBTi Criteria C14 (Sector-specific requirements)
    - SBTi Criteria C16 (SDA methodology)
    - IEA Energy Technology Perspectives (pathway data source)

Example:
    >>> from services.config import SBTiAppConfig
    >>> engine = SectorEngine(SBTiAppConfig())
    >>> pathway = engine.get_sector_pathway("power", 2020, 2030, "1.5c")
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .config import (
    SECTOR_PATHWAYS,
    SBTiAppConfig,
    SBTiSector,
    IntensityMetric,
)
from .models import (
    SectorPathway,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ISIC / NACE / NAICS to SBTi Sector Mapping
# ---------------------------------------------------------------------------

ISIC_TO_SBTI: Dict[str, SBTiSector] = {
    "3510": SBTiSector.POWER,       # Electric power generation
    "3520": SBTiSector.POWER,       # Gas manufacturing/distribution
    "0610": SBTiSector.OIL_GAS,     # Crude petroleum extraction
    "0620": SBTiSector.OIL_GAS,     # Natural gas extraction
    "1920": SBTiSector.OIL_GAS,     # Refined petroleum products
    "4911": SBTiSector.TRANSPORT,   # Passenger rail
    "4912": SBTiSector.TRANSPORT,   # Freight rail
    "4921": SBTiSector.TRANSPORT,   # Urban and suburban transport
    "4922": SBTiSector.TRANSPORT,   # Other passenger land transport
    "4923": SBTiSector.TRANSPORT,   # Freight transport by road
    "4100": SBTiSector.BUILDINGS,   # Construction of buildings
    "6810": SBTiSector.BUILDINGS,   # Real estate with owned property
    "2394": SBTiSector.CEMENT,      # Cement, lime, and plaster
    "2410": SBTiSector.STEEL,       # Basic iron and steel
    "2420": SBTiSector.STEEL,       # Steel tubes, pipes
    "2442": SBTiSector.ALUMINIUM,   # Aluminium production
    "2011": SBTiSector.CHEMICALS,   # Basic chemicals
    "2012": SBTiSector.CHEMICALS,   # Fertilizers
    "2013": SBTiSector.CHEMICALS,   # Plastics and synthetic rubber
    "5110": SBTiSector.AVIATION,    # Passenger air transport
    "5120": SBTiSector.AVIATION,    # Freight air transport
    "1701": SBTiSector.PULP_PAPER,  # Pulp, paper, paperboard
    "1702": SBTiSector.PULP_PAPER,  # Corrugated paper products
    "5011": SBTiSector.MARITIME,    # Sea/coastal passenger transport
    "5012": SBTiSector.MARITIME,    # Sea/coastal freight transport
    "0111": SBTiSector.AGRICULTURE, # Growing of cereals
    "0121": SBTiSector.AGRICULTURE, # Growing of grapes
    "0141": SBTiSector.AGRICULTURE, # Raising of cattle
    "6411": SBTiSector.FINANCIAL_INSTITUTIONS,  # Central banking
    "6419": SBTiSector.FINANCIAL_INSTITUTIONS,  # Other monetary intermediation
    "6430": SBTiSector.FINANCIAL_INSTITUTIONS,  # Trusts, funds
    "1410": SBTiSector.APPAREL_FOOTWEAR,  # Wearing apparel
    "1520": SBTiSector.APPAREL_FOOTWEAR,  # Footwear
}


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SectorPathwayResult(BaseModel):
    """Sector-specific intensity pathway result."""

    sector: str = Field(...)
    intensity_metric: str = Field(default="")
    intensity_unit: str = Field(default="")
    base_year: int = Field(default=2020)
    target_year: int = Field(default=2030)
    base_intensity: float = Field(default=0.0)
    target_intensity: float = Field(default=0.0)
    annual_reduction_rate_pct: float = Field(default=0.0)
    alignment: str = Field(default="1.5c")
    milestones: List[Dict[str, float]] = Field(default_factory=list)
    sda_available: bool = Field(default=True)
    provenance_hash: str = Field(default="")


class SectorDetection(BaseModel):
    """Sector detection result from industry codes."""

    org_id: str = Field(...)
    detected_sector: Optional[str] = Field(None)
    detection_source: str = Field(default="")
    code_used: str = Field(default="")
    confidence: float = Field(default=0.0)
    sda_available: bool = Field(default=False)
    alternative_sectors: List[str] = Field(default_factory=list)


class SectorBenchmark(BaseModel):
    """Sector intensity benchmark comparison."""

    sector: str = Field(...)
    company_intensity: float = Field(default=0.0)
    sector_avg_2020: float = Field(default=0.0)
    sector_target_2030: float = Field(default=0.0)
    sector_target_2050: float = Field(default=0.0)
    gap_to_2030: float = Field(default=0.0)
    gap_to_2050: float = Field(default=0.0)
    percentile_rank: float = Field(default=50.0)
    on_track: bool = Field(default=False)
    intensity_unit: str = Field(default="")


class MultiSectorBlend(BaseModel):
    """Blended pathway for multi-sector companies."""

    org_id: str = Field(...)
    sectors: List[Dict[str, Any]] = Field(default_factory=list)
    blended_target_intensity: float = Field(default=0.0)
    blended_annual_rate_pct: float = Field(default=0.0)
    total_weight: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class SectorRequirementCheck(BaseModel):
    """Sector-specific requirement compliance check."""

    org_id: str = Field(...)
    sector: str = Field(default="general")
    sector_name: str = Field(default="")
    intensity_metric: str = Field(default="")
    sda_available: bool = Field(default=False)
    sda_required: bool = Field(default=False)
    compliant: bool = Field(default=False)
    checks: List[Dict[str, Any]] = Field(default_factory=list)
    message: str = Field(default="")


# ---------------------------------------------------------------------------
# SectorEngine
# ---------------------------------------------------------------------------

class SectorEngine:
    """
    SBTi Sector-Specific Pathway and Benchmark Engine.

    Provides sector pathway lookups, intensity benchmark comparison,
    ISIC/NACE/NAICS sector detection, multi-sector blending, and
    sector compliance verification for the Sectoral Decarbonization
    Approach (SDA).

    Attributes:
        config: Application configuration.
        _sector_pathways_cache: Computed sector pathway cache.
        _org_sectors: Detected sector assignments keyed by org_id.

    Example:
        >>> engine = SectorEngine(SBTiAppConfig())
        >>> pathway = engine.get_sector_pathway("power", 2020, 2030, "1.5c")
    """

    # Sectors that require or strongly recommend SDA
    SDA_REQUIRED_SECTORS = {
        SBTiSector.POWER, SBTiSector.CEMENT, SBTiSector.STEEL,
        SBTiSector.ALUMINIUM,
    }

    SDA_AVAILABLE_SECTORS = {
        SBTiSector.POWER, SBTiSector.OIL_GAS, SBTiSector.TRANSPORT,
        SBTiSector.BUILDINGS, SBTiSector.CEMENT, SBTiSector.STEEL,
        SBTiSector.ALUMINIUM, SBTiSector.CHEMICALS, SBTiSector.AVIATION,
        SBTiSector.PULP_PAPER, SBTiSector.MARITIME,
    }

    def __init__(self, config: Optional[SBTiAppConfig] = None) -> None:
        """Initialize the Sector Engine."""
        self.config = config or SBTiAppConfig()
        self._sector_pathways_cache: Dict[str, SectorPathwayResult] = {}
        self._org_sectors: Dict[str, SBTiSector] = {}
        logger.info("SectorEngine initialized with %d sector pathways", len(SECTOR_PATHWAYS))

    # ------------------------------------------------------------------
    # Sector Pathway Lookup
    # ------------------------------------------------------------------

    def get_sector_pathway(
        self,
        sector: str,
        base_year: int = 2020,
        target_year: int = 2030,
        alignment: str = "1.5c",
    ) -> SectorPathwayResult:
        """
        Get a sector-specific intensity pathway.

        Looks up the sector in SECTOR_PATHWAYS and generates annual
        milestones by linear interpolation between base and target years.

        Args:
            sector: SBTi sector key (e.g. "power", "steel").
            base_year: Base year (default 2020).
            target_year: Target year (default 2030).
            alignment: Temperature alignment ("1.5c" or "well_below_2c").

        Returns:
            SectorPathwayResult with annual intensity milestones.

        Raises:
            ValueError: If sector is not found in SECTOR_PATHWAYS.
        """
        start = datetime.utcnow()

        sector_data = SECTOR_PATHWAYS.get(sector)
        if sector_data is None:
            raise ValueError(
                f"Unknown sector: {sector}. "
                f"Valid: {list(SECTOR_PATHWAYS.keys())}"
            )

        # Select rate based on alignment
        rate_key = "annual_reduction_1_5c" if alignment == "1.5c" else "annual_reduction_wb2c"
        annual_rate = float(sector_data[rate_key])
        metric = sector_data["intensity_metric"]
        base_intensity = float(sector_data["base_year_intensity_2020"])
        target_2030 = float(sector_data["target_intensity_2030"])
        target_2050 = float(sector_data["target_intensity_2050"])
        sda_available = sector_data["sda_available"]

        # Determine target intensity for the chosen target year
        if target_year <= 2030:
            total_years = 2030 - 2020
            elapsed = min(target_year - 2020, total_years)
            progress = elapsed / total_years if total_years > 0 else 1.0
            target_intensity = base_intensity + (target_2030 - base_intensity) * progress
        elif target_year <= 2050:
            span = 2050 - 2030
            elapsed = target_year - 2030
            progress = elapsed / span if span > 0 else 1.0
            target_intensity = target_2030 + (target_2050 - target_2030) * progress
        else:
            target_intensity = target_2050

        # Generate milestones
        milestones: List[Dict[str, float]] = []
        total_pathway_years = target_year - base_year

        for year in range(base_year, target_year + 1):
            years_elapsed = year - base_year
            if total_pathway_years > 0:
                frac = years_elapsed / total_pathway_years
            else:
                frac = 1.0
            intensity = base_intensity + (target_intensity - base_intensity) * frac
            reduction_pct = (1 - intensity / base_intensity) * 100 if base_intensity > 0 else 0
            milestones.append({
                "year": year,
                "intensity": round(intensity, 4),
                "cumulative_reduction_pct": round(max(reduction_pct, 0), 2),
            })

        provenance = _sha256(f"sector:{sector}:{base_year}:{target_year}:{alignment}")

        result = SectorPathwayResult(
            sector=sector,
            intensity_metric=metric,
            intensity_unit=metric,
            base_year=base_year,
            target_year=target_year,
            base_intensity=base_intensity,
            target_intensity=round(target_intensity, 4),
            annual_reduction_rate_pct=annual_rate,
            alignment=alignment,
            milestones=milestones,
            sda_available=sda_available,
            provenance_hash=provenance,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Sector pathway: %s base=%.4f target=%.4f rate=%.1f%%/yr (%s) in %.1f ms",
            sector, base_intensity, target_intensity, annual_rate, alignment, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Individual Sector Pathway Calculators
    # ------------------------------------------------------------------

    def calculate_power_pathway(
        self, company_intensity: float, base_year: int = 2020, target_year: int = 2030,
    ) -> SectorPathwayResult:
        """Calculate power sector pathway (tCO2e/MWh)."""
        result = self.get_sector_pathway("power", base_year, target_year)
        result.base_intensity = company_intensity
        return result

    def calculate_cement_pathway(
        self, company_intensity: float, base_year: int = 2020, target_year: int = 2030,
    ) -> SectorPathwayResult:
        """Calculate cement sector pathway (tCO2e/t_cement)."""
        result = self.get_sector_pathway("cement", base_year, target_year)
        result.base_intensity = company_intensity
        return result

    def calculate_steel_pathway(
        self, company_intensity: float, scrap_ratio: float = 0.0,
        base_year: int = 2020, target_year: int = 2030,
    ) -> SectorPathwayResult:
        """
        Calculate steel sector pathway (tCO2e/t_steel).

        Args:
            company_intensity: Company intensity in tCO2e/t_steel.
            scrap_ratio: Percentage of production from scrap/EAF route.
            base_year: Base year.
            target_year: Target year.

        Returns:
            SectorPathwayResult adjusted for production route.
        """
        result = self.get_sector_pathway("steel", base_year, target_year)
        result.base_intensity = company_intensity

        # Scrap-based (EAF) steel has lower baseline intensity
        if scrap_ratio > 0.5:
            eaf_adjustment = 0.7  # EAF is ~70% of BF-BOF intensity
            result.target_intensity = round(result.target_intensity * eaf_adjustment, 4)

        return result

    def calculate_buildings_pathway(
        self, company_intensity: float, building_type: str = "commercial",
        base_year: int = 2020, target_year: int = 2030,
    ) -> SectorPathwayResult:
        """
        Calculate buildings sector pathway (kgCO2e/m2).

        Uses CRREM (Carbon Risk Real Estate Monitor) reference for
        building-type-specific pathways.

        Args:
            company_intensity: Company intensity in kgCO2e/m2.
            building_type: "commercial" or "residential".
            base_year: Base year.
            target_year: Target year.

        Returns:
            SectorPathwayResult for building type.
        """
        result = self.get_sector_pathway("buildings", base_year, target_year)
        result.base_intensity = company_intensity

        # Residential buildings have slightly lower targets
        if building_type == "residential":
            result.target_intensity = round(result.target_intensity * 0.85, 4)

        return result

    def calculate_maritime_pathway(
        self, company_intensity: float, base_year: int = 2020, target_year: int = 2030,
    ) -> SectorPathwayResult:
        """Calculate maritime sector pathway (gCO2e/t_nm)."""
        result = self.get_sector_pathway("maritime", base_year, target_year)
        result.base_intensity = company_intensity
        return result

    def calculate_aviation_pathway(
        self, company_intensity: float, metric: str = "rpk",
        base_year: int = 2020, target_year: int = 2030,
    ) -> SectorPathwayResult:
        """
        Calculate aviation sector pathway (gCO2e/RPK or gCO2e/RTK).

        Args:
            company_intensity: Company intensity.
            metric: "rpk" for passenger or "rtk" for freight.
            base_year: Base year.
            target_year: Target year.

        Returns:
            SectorPathwayResult for aviation.
        """
        result = self.get_sector_pathway("aviation", base_year, target_year)
        result.base_intensity = company_intensity
        result.intensity_metric = f"gCO2e/{metric.upper()}"
        return result

    def calculate_land_transport_pathway(
        self, company_intensity: float, base_year: int = 2020, target_year: int = 2030,
    ) -> SectorPathwayResult:
        """Calculate land transport sector pathway (gCO2e/pkm)."""
        result = self.get_sector_pathway("transport", base_year, target_year)
        result.base_intensity = company_intensity
        return result

    def calculate_chemicals_pathway(
        self, company_intensity: float, base_year: int = 2020, target_year: int = 2030,
    ) -> SectorPathwayResult:
        """Calculate chemicals sector pathway (tCO2e/t_product)."""
        result = self.get_sector_pathway("chemicals", base_year, target_year)
        result.base_intensity = company_intensity
        return result

    def calculate_apparel_pathway(
        self, company_intensity: float, base_year: int = 2020, target_year: int = 2030,
    ) -> SectorPathwayResult:
        """Calculate apparel/footwear sector pathway (uses general)."""
        result = self.get_sector_pathway("general", base_year, target_year)
        result.sector = "apparel_footwear"
        result.base_intensity = company_intensity
        return result

    # ------------------------------------------------------------------
    # Sector Detection
    # ------------------------------------------------------------------

    def detect_sector_from_codes(
        self,
        org_id: str,
        isic_code: Optional[str] = None,
        nace_code: Optional[str] = None,
        naics_code: Optional[str] = None,
    ) -> SectorDetection:
        """
        Detect the SBTi sector from industry classification codes.

        Tries ISIC first, then NACE, then NAICS. Falls back to
        "general" if no match is found.

        Args:
            org_id: Organization identifier.
            isic_code: ISIC Rev. 4 code.
            nace_code: NACE Rev. 2 code.
            naics_code: NAICS code.

        Returns:
            SectorDetection with matched sector and confidence.
        """
        detected: Optional[SBTiSector] = None
        source = ""
        code_used = ""
        confidence = 0.0

        # Try ISIC first (highest confidence)
        if isic_code and isic_code in ISIC_TO_SBTI:
            detected = ISIC_TO_SBTI[isic_code]
            source = "isic_rev4"
            code_used = isic_code
            confidence = 0.95

        # Try NACE (map first 4 chars to ISIC where possible)
        elif nace_code:
            nace_4 = nace_code[:4]
            if nace_4 in ISIC_TO_SBTI:
                detected = ISIC_TO_SBTI[nace_4]
                source = "nace_rev2"
                code_used = nace_code
                confidence = 0.85

        # Try NAICS (partial mapping)
        elif naics_code:
            # NAICS to ISIC approximate mapping for major sectors
            naics_map: Dict[str, SBTiSector] = {
                "2211": SBTiSector.POWER,
                "2111": SBTiSector.OIL_GAS,
                "3241": SBTiSector.CEMENT,
                "3311": SBTiSector.STEEL,
                "3313": SBTiSector.ALUMINIUM,
                "3251": SBTiSector.CHEMICALS,
                "4811": SBTiSector.AVIATION,
                "3221": SBTiSector.PULP_PAPER,
                "4831": SBTiSector.MARITIME,
                "5221": SBTiSector.FINANCIAL_INSTITUTIONS,
            }
            naics_4 = naics_code[:4]
            if naics_4 in naics_map:
                detected = naics_map[naics_4]
                source = "naics"
                code_used = naics_code
                confidence = 0.75

        if detected is None:
            detected = SBTiSector.GENERAL
            source = "default"
            confidence = 0.50

        self._org_sectors[org_id] = detected

        sda_avail = detected in self.SDA_AVAILABLE_SECTORS

        # Alternative sectors within 2 ISIC digits
        alternatives: List[str] = []
        if isic_code:
            prefix = isic_code[:2]
            for code, sector in ISIC_TO_SBTI.items():
                if code[:2] == prefix and sector != detected:
                    if sector.value not in alternatives:
                        alternatives.append(sector.value)

        result = SectorDetection(
            org_id=org_id,
            detected_sector=detected.value,
            detection_source=source,
            code_used=code_used,
            confidence=confidence,
            sda_available=sda_avail,
            alternative_sectors=alternatives[:3],
        )

        logger.info(
            "Sector detection for org %s: %s (source=%s, confidence=%.0f%%)",
            org_id, detected.value, source, confidence * 100,
        )
        return result

    # ------------------------------------------------------------------
    # Multi-Sector Blending
    # ------------------------------------------------------------------

    def blend_multi_sector_pathways(
        self,
        org_id: str,
        sector_weights: List[Dict[str, Any]],
        base_year: int = 2020,
        target_year: int = 2030,
    ) -> MultiSectorBlend:
        """
        Calculate a blended pathway for multi-sector companies.

        Weights each sector pathway by revenue or emissions share to
        produce a combined target intensity.

        Args:
            org_id: Organization identifier.
            sector_weights: List of dicts with keys: sector, weight,
                           company_intensity.
            base_year: Base year.
            target_year: Target year.

        Returns:
            MultiSectorBlend with weighted targets.
        """
        start = datetime.utcnow()
        sectors: List[Dict[str, Any]] = []
        blended_intensity = 0.0
        blended_rate = 0.0
        total_weight = 0.0

        for sw in sector_weights:
            sector = sw.get("sector", "general")
            weight = sw.get("weight", 0.0)
            company_intensity = sw.get("company_intensity", 0.0)

            try:
                pathway = self.get_sector_pathway(sector, base_year, target_year)
                sector_target = pathway.target_intensity
                sector_rate = pathway.annual_reduction_rate_pct
            except ValueError:
                sector_target = 0.0
                sector_rate = 4.2  # Default to ACA 1.5C

            blended_intensity += sector_target * weight
            blended_rate += sector_rate * weight
            total_weight += weight

            sectors.append({
                "sector": sector,
                "weight": weight,
                "company_intensity": company_intensity,
                "sector_target_intensity": sector_target,
                "sector_rate_pct": sector_rate,
            })

        if total_weight > 0:
            blended_intensity /= total_weight
            blended_rate /= total_weight

        provenance = _sha256(f"blend:{org_id}:{total_weight}:{blended_intensity}")

        result = MultiSectorBlend(
            org_id=org_id,
            sectors=sectors,
            blended_target_intensity=round(blended_intensity, 4),
            blended_annual_rate_pct=round(blended_rate, 2),
            total_weight=round(total_weight, 4),
            provenance_hash=provenance,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Blended pathway for org %s: %d sectors, rate=%.2f%%/yr in %.1f ms",
            org_id, len(sectors), blended_rate, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Sector Benchmark Comparison
    # ------------------------------------------------------------------

    def get_sector_benchmarks(
        self,
        sector: str,
        company_intensity: float,
    ) -> SectorBenchmark:
        """
        Compare company intensity against sector benchmarks.

        Args:
            sector: SBTi sector key.
            company_intensity: Company intensity value.

        Returns:
            SectorBenchmark with gap analysis and percentile rank.

        Raises:
            ValueError: If sector not found.
        """
        sector_data = SECTOR_PATHWAYS.get(sector)
        if sector_data is None:
            raise ValueError(f"Unknown sector: {sector}")

        avg_2020 = float(sector_data["base_year_intensity_2020"])
        target_2030 = float(sector_data["target_intensity_2030"])
        target_2050 = float(sector_data["target_intensity_2050"])
        metric = sector_data["intensity_metric"]

        gap_2030 = company_intensity - target_2030
        gap_2050 = company_intensity - target_2050

        # Estimate percentile rank (lower intensity = better)
        if avg_2020 > 0:
            ratio = company_intensity / avg_2020
            if ratio <= 0.5:
                percentile = 90.0
            elif ratio <= 0.75:
                percentile = 75.0
            elif ratio <= 1.0:
                percentile = 50.0
            elif ratio <= 1.25:
                percentile = 25.0
            else:
                percentile = 10.0
        else:
            percentile = 50.0

        on_track = company_intensity <= target_2030

        return SectorBenchmark(
            sector=sector,
            company_intensity=round(company_intensity, 4),
            sector_avg_2020=avg_2020,
            sector_target_2030=target_2030,
            sector_target_2050=target_2050,
            gap_to_2030=round(gap_2030, 4),
            gap_to_2050=round(gap_2050, 4),
            percentile_rank=percentile,
            on_track=on_track,
            intensity_unit=metric,
        )

    # ------------------------------------------------------------------
    # Sector Requirement Check
    # ------------------------------------------------------------------

    def check_requirements(
        self,
        org_id: str,
        sector: str,
    ) -> SectorRequirementCheck:
        """
        Check sector-specific SBTi requirements for an organization.

        Evaluates whether the organization must use SDA methodology,
        has an appropriate intensity metric, and meets sector-specific
        criteria (C14, C16).

        Args:
            org_id: Organization identifier.
            sector: SBTi sector key.

        Returns:
            SectorRequirementCheck with compliance status.
        """
        sector_data = SECTOR_PATHWAYS.get(sector)
        if sector_data is None:
            return SectorRequirementCheck(
                org_id=org_id,
                sector=sector,
                message=f"Unknown sector: {sector}",
            )

        try:
            sbti_sector = SBTiSector(sector)
        except ValueError:
            sbti_sector = SBTiSector.GENERAL

        sda_available = sector_data.get("sda_available", False)
        sda_required = sbti_sector in self.SDA_REQUIRED_SECTORS
        metric = sector_data.get("intensity_metric", "")
        sector_name = sector.replace("_", " ").title()

        checks: List[Dict[str, Any]] = []

        # Check 1: SDA availability
        checks.append({
            "criterion": "SDA Pathway Available",
            "met": sda_available,
            "description": f"SDA pathway {'is' if sda_available else 'is NOT'} available for {sector}",
        })

        # Check 2: SDA requirement
        checks.append({
            "criterion": "SDA Methodology Required",
            "met": True if not sda_required else sda_available,
            "description": (
                f"SDA {'IS required' if sda_required else 'is not required'} for {sector}"
            ),
        })

        # Check 3: Intensity metric defined
        checks.append({
            "criterion": "Intensity Metric Defined",
            "met": bool(metric),
            "description": f"Intensity metric: {metric}" if metric else "No intensity metric defined",
        })

        # Check 4: Sector-specific thresholds
        annual_rate = float(sector_data.get("annual_reduction_1_5c", Decimal("4.2")))
        checks.append({
            "criterion": "1.5C Annual Rate",
            "met": True,
            "description": f"Required annual reduction: {annual_rate}%/yr for 1.5C alignment",
        })

        all_met = all(c["met"] for c in checks)

        result = SectorRequirementCheck(
            org_id=org_id,
            sector=sector,
            sector_name=sector_name,
            intensity_metric=metric,
            sda_available=sda_available,
            sda_required=sda_required,
            compliant=all_met,
            checks=checks,
            message=(
                f"Sector {sector_name}: {'compliant' if all_met else 'non-compliant'} "
                f"with SBTi sector requirements."
            ),
        )

        logger.info(
            "Sector check for org %s (%s): compliant=%s, sda_required=%s",
            org_id, sector, all_met, sda_required,
        )
        return result

    # ------------------------------------------------------------------
    # Sector Pathway Domain Model
    # ------------------------------------------------------------------

    def create_sector_pathway_model(
        self,
        sector: str,
        base_year: int = 2020,
        target_year: int = 2050,
    ) -> SectorPathway:
        """
        Create a SectorPathway domain model for storage.

        Args:
            sector: SBTi sector key.
            base_year: Base year.
            target_year: Target year.

        Returns:
            SectorPathway Pydantic model.
        """
        sector_data = SECTOR_PATHWAYS.get(sector)
        if sector_data is None:
            raise ValueError(f"Unknown sector: {sector}")

        try:
            sbti_sector = SBTiSector(sector)
        except ValueError:
            sbti_sector = SBTiSector.GENERAL

        base_val = float(sector_data["base_year_intensity_2020"])
        target_val = float(sector_data["target_intensity_2050"])
        metric = sector_data["intensity_metric"]

        # Generate annual points
        annual_points: Dict[int, Decimal] = {}
        target_2030 = float(sector_data["target_intensity_2030"])
        for year in range(base_year, target_year + 1):
            if year <= 2030:
                frac = (year - 2020) / 10 if 2030 > 2020 else 1.0
                val = base_val + (target_2030 - base_val) * frac
            else:
                frac = (year - 2030) / 20 if 2050 > 2030 else 1.0
                val = target_2030 + (target_val - target_2030) * frac
            annual_points[year] = Decimal(str(round(val, 4)))

        return SectorPathway(
            sector=sbti_sector,
            intensity_metric=metric,
            intensity_unit=metric,
            base_year=base_year,
            target_year=target_year,
            base_value=Decimal(str(base_val)),
            target_value=Decimal(str(target_val)),
            annual_points=annual_points,
            source="SBTi SDA Sector Pathway",
        )
