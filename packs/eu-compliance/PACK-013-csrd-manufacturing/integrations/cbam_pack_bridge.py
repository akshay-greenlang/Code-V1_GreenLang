"""
PACK-013 CSRD Manufacturing Pack - CBAM Pack Bridge.

Bridge to PACK-004 (CBAM Readiness) and PACK-005 (CBAM Complete) for
manufacturers producing CBAM-affected goods (cement, iron/steel,
aluminium, fertilisers, electricity, hydrogen).  Maps facility-level
embedded-emissions data into the CBAM quarterly/annual reporting format.
"""

import hashlib
import importlib
import logging
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CBAMPhase(str, Enum):
    """CBAM regulation phase."""
    TRANSITIONAL = "transitional"
    DEFINITIVE = "definitive"


class CBAMGoodCategory(str, Enum):
    """CBAM goods categories (Annex I)."""
    CEMENT = "cement"
    IRON_STEEL = "iron_steel"
    ALUMINIUM = "aluminium"
    FERTILISERS = "fertilisers"
    ELECTRICITY = "electricity"
    HYDROGEN = "hydrogen"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class CBAMBridgeConfig(BaseModel):
    """Configuration for the CBAM pack bridge."""
    affected_goods: List[CBAMGoodCategory] = Field(default_factory=list)
    cbam_phase: CBAMPhase = Field(default=CBAMPhase.TRANSITIONAL)
    quarterly_reporting: bool = Field(default=True)
    pack_module_prefix: str = Field(
        default="packs.eu_compliance",
        description="Python module prefix for CBAM packs",
    )
    use_default_values: bool = Field(
        default=False,
        description="Allow EU default values where actual data is missing",
    )
    installation_id: Optional[str] = Field(default=None)
    deminimis_threshold_eur: float = Field(
        default=150.0,
        description="De minimis threshold in EUR per consignment",
    )
    carbon_price_eur_per_tco2e: float = Field(
        default=90.0,
        description="Current EUA price assumption for cost estimation",
    )


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class EmbeddedEmissionsRecord(BaseModel):
    """Embedded emissions for a single good."""
    good_category: str
    cn_code: str = Field(default="")
    production_volume_tonnes: float = Field(default=0.0, ge=0.0)
    direct_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    indirect_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    total_embedded_tco2e: float = Field(default=0.0, ge=0.0)
    specific_embedded_emissions: float = Field(
        default=0.0, ge=0.0,
        description="tCO2e per tonne of product",
    )
    data_source: str = Field(default="actual")
    methodology: str = Field(default="direct_measurement")


class CBAMBridgeResult(BaseModel):
    """Result of CBAM bridge operations."""
    embedded_emissions: List[EmbeddedEmissionsRecord] = Field(
        default_factory=list
    )
    total_embedded_tco2e: float = Field(default=0.0, ge=0.0)
    certificate_requirement: Dict[str, Any] = Field(
        default_factory=dict
    )
    quarterly_data: Dict[str, Any] = Field(default_factory=dict)
    deminimis_applicable: bool = Field(default=False)
    provenance_hash: str = Field(default="")
    cbam_phase: CBAMPhase = Field(default=CBAMPhase.TRANSITIONAL)
    submission_status: str = Field(default="pending")


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class CBAMPackBridge:
    """
    Bridge between PACK-013 manufacturing data and PACK-004/005 CBAM
    reporting packs.

    Transforms facility-level process-emissions and production data into
    CBAM embedded-emissions records suitable for quarterly reporting
    during the transitional phase or annual compliance during the
    definitive phase.
    """

    def __init__(
        self, config: Optional[CBAMBridgeConfig] = None
    ) -> None:
        self.config = config or CBAMBridgeConfig()
        self._pack_module: Any = None
        self._load_pack()

    # -- pack loading --------------------------------------------------------

    def _load_pack(self) -> None:
        """Import the appropriate CBAM pack module."""
        pack_name = (
            "PACK_005_cbam_complete"
            if self.config.cbam_phase == CBAMPhase.DEFINITIVE
            else "PACK_004_cbam_readiness"
        )
        module_name = f"{self.config.pack_module_prefix}.{pack_name}"
        try:
            self._pack_module = importlib.import_module(module_name)
            logger.info("Loaded CBAM pack: %s", module_name)
        except ImportError:
            logger.warning(
                "CBAM pack %s not available; bridge uses built-in logic",
                module_name,
            )
            self._pack_module = None

    @staticmethod
    def _compute_hash(data: Any) -> str:
        raw = str(data).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    # -- CN code mapping -----------------------------------------------------

    CN_CODE_MAP: Dict[str, List[str]] = {
        CBAMGoodCategory.CEMENT.value: [
            "2523 10", "2523 21", "2523 29", "2523 30", "2523 90",
        ],
        CBAMGoodCategory.IRON_STEEL.value: [
            "7201", "7202", "7203", "7204", "7205", "7206", "7207",
            "7208", "7209", "7210", "7211", "7212", "7213",
        ],
        CBAMGoodCategory.ALUMINIUM.value: [
            "7601", "7603", "7604", "7605", "7606", "7607",
        ],
        CBAMGoodCategory.FERTILISERS.value: [
            "2808 00", "3102", "3105",
        ],
        CBAMGoodCategory.ELECTRICITY.value: [
            "2716 00",
        ],
        CBAMGoodCategory.HYDROGEN.value: [
            "2804 10",
        ],
    }

    # -- EU default emission factors -----------------------------------------

    EU_DEFAULT_FACTORS: Dict[str, float] = {
        CBAMGoodCategory.CEMENT.value: 0.766,
        CBAMGoodCategory.IRON_STEEL.value: 1.852,
        CBAMGoodCategory.ALUMINIUM.value: 6.700,
        CBAMGoodCategory.FERTILISERS.value: 3.600,
        CBAMGoodCategory.ELECTRICITY.value: 0.450,
        CBAMGoodCategory.HYDROGEN.value: 9.000,
    }

    # -- public API ----------------------------------------------------------

    def submit_embedded_emissions(
        self, facility_data: Dict[str, Any]
    ) -> CBAMBridgeResult:
        """
        Calculate and submit embedded emissions for CBAM-affected goods.

        Args:
            facility_data: Dict containing production volumes, process
                emissions, and energy data keyed by good category.

        Returns:
            CBAMBridgeResult with per-good embedded emissions.
        """
        records: List[EmbeddedEmissionsRecord] = []

        for good in self.config.affected_goods:
            good_data = facility_data.get(good.value, {})
            record = self._calculate_embedded(good.value, good_data)
            records.append(record)

        total_embedded = sum(r.total_embedded_tco2e for r in records)
        cert = self.get_certificate_requirement(total_embedded)
        quarterly = (
            self.generate_quarterly_report(
                facility_data.get("period", "Q1-2025")
            )
            if self.config.quarterly_reporting else {}
        )
        deminimis = self.check_deminimis(
            [r.model_dump() for r in records]
        )

        combined = {
            "records": [r.model_dump() for r in records],
            "total": total_embedded,
        }

        return CBAMBridgeResult(
            embedded_emissions=records,
            total_embedded_tco2e=total_embedded,
            certificate_requirement=cert,
            quarterly_data=quarterly,
            deminimis_applicable=deminimis,
            provenance_hash=self._compute_hash(combined),
            cbam_phase=self.config.cbam_phase,
            submission_status="submitted",
        )

    def get_certificate_requirement(
        self, embedded_emissions: float
    ) -> Dict[str, Any]:
        """
        Calculate CBAM certificate requirement for the definitive phase.

        During the transitional phase, no certificates are required but
        the estimated requirement is still calculated for planning.
        """
        carbon_price = self.config.carbon_price_eur_per_tco2e
        estimated_cost = embedded_emissions * carbon_price

        # In the definitive phase, certificates must be surrendered
        certificates_needed = int(embedded_emissions) + (
            1 if embedded_emissions % 1 > 0 else 0
        )

        return {
            "total_embedded_tco2e": embedded_emissions,
            "certificates_needed": certificates_needed,
            "estimated_cost_eur": round(estimated_cost, 2),
            "carbon_price_eur": carbon_price,
            "phase": self.config.cbam_phase.value,
            "binding": self.config.cbam_phase == CBAMPhase.DEFINITIVE,
            "foreign_carbon_price_deduction": 0.0,
        }

    def generate_quarterly_report(
        self, period: str
    ) -> Dict[str, Any]:
        """
        Generate CBAM quarterly report data.

        Required during the transitional phase (Oct 2023 - Dec 2025).
        """
        return {
            "report_type": "quarterly",
            "period": period,
            "affected_goods": [
                g.value for g in self.config.affected_goods
            ],
            "installation_id": self.config.installation_id,
            "cbam_phase": self.config.cbam_phase.value,
            "status": "draft",
            "sections": {
                "importer_details": {},
                "goods_imported": {},
                "embedded_emissions": {},
                "carbon_price_paid": {},
            },
        }

    def check_deminimis(
        self, goods: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if the consignment falls below the de minimis threshold.

        Returns True if de minimis applies (no CBAM reporting required).
        """
        total_value = sum(
            g.get("consignment_value_eur", 0.0) for g in goods
        )
        if total_value <= self.config.deminimis_threshold_eur:
            logger.info(
                "De minimis applies: total value %.2f EUR <= %.2f EUR",
                total_value, self.config.deminimis_threshold_eur,
            )
            return True
        return False

    # -- internal helpers ----------------------------------------------------

    def _calculate_embedded(
        self, good_category: str, good_data: Dict[str, Any]
    ) -> EmbeddedEmissionsRecord:
        """Calculate embedded emissions for a single good category."""
        production_vol = good_data.get("production_volume_tonnes", 0.0)
        direct = good_data.get("direct_emissions_tco2e", 0.0)
        indirect = good_data.get("indirect_emissions_tco2e", 0.0)

        # Fall back to EU default factors when data is missing
        data_source = "actual"
        if direct == 0.0 and self.config.use_default_values:
            factor = self.EU_DEFAULT_FACTORS.get(good_category, 1.0)
            direct = production_vol * factor
            data_source = "eu_default"
            logger.info(
                "Using EU default factor %.3f for %s",
                factor, good_category,
            )

        total = direct + indirect
        specific = total / production_vol if production_vol > 0 else 0.0
        cn_codes = self.CN_CODE_MAP.get(good_category, [])

        return EmbeddedEmissionsRecord(
            good_category=good_category,
            cn_code=cn_codes[0] if cn_codes else "",
            production_volume_tonnes=production_vol,
            direct_emissions_tco2e=round(direct, 4),
            indirect_emissions_tco2e=round(indirect, 4),
            total_embedded_tco2e=round(total, 4),
            specific_embedded_emissions=round(specific, 6),
            data_source=data_source,
            methodology=(
                "eu_default_values" if data_source == "eu_default"
                else "direct_measurement"
            ),
        )
