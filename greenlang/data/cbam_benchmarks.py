"""
CBAM Benchmark Database - EU Default Values for Carbon Border Adjustment Mechanism

This module provides CBAM default benchmark values from EU Implementing Regulation
for carbon intensity of CBAM-regulated goods.

Sources:
- EU Regulation 2023/956 (CBAM Regulation)
- Commission Implementing Regulation (EU) 2023/1773
- Annex II: Default values for embedded emissions
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ProductType(str, Enum):
    """CBAM-regulated product types."""
    STEEL_HOT_ROLLED = "steel_hot_rolled_coil"
    STEEL_REBAR = "steel_rebar"
    STEEL_WIRE_ROD = "steel_wire_rod"
    CEMENT_PORTLAND = "cement_portland"
    CEMENT_CLINKER = "cement_clinker"
    ALUMINUM_UNWROUGHT = "aluminum_unwrought"
    ALUMINUM_PRODUCTS = "aluminum_products"
    FERTILIZER_AMMONIA = "fertilizer_ammonia"
    FERTILIZER_UREA = "fertilizer_urea"
    FERTILIZER_NITRIC_ACID = "fertilizer_nitric_acid"
    ELECTRICITY = "electricity"
    HYDROGEN = "hydrogen"


class ProductionMethod(str, Enum):
    """Production methods affecting carbon intensity."""
    BASIC_OXYGEN_FURNACE = "basic_oxygen_furnace"
    ELECTRIC_ARC_FURNACE = "electric_arc_furnace"
    BLAST_FURNACE = "blast_furnace"
    DRY_PROCESS = "dry_process"
    WET_PROCESS = "wet_process"
    ELECTROLYSIS = "electrolysis"
    HABER_BOSCH = "haber_bosch"
    STEAM_REFORMING = "steam_reforming"


@dataclass
class CBAMBenchmark:
    """CBAM default benchmark value."""
    product_type: str
    production_method: Optional[str]
    benchmark_value: float  # tCO2e per tonne
    unit: str  # Always tCO2e/tonne for CBAM
    cn_codes: list  # Combined Nomenclature codes
    effective_date: str
    source: str
    notes: str


class CBAMBenchmarkDatabase:
    """
    Database of CBAM default benchmark values.

    These are the EU default values used when actual embedded emissions
    cannot be verified. Importers exceeding these benchmarks must purchase
    CBAM certificates.
    """

    def __init__(self):
        self.benchmarks: Dict[str, CBAMBenchmark] = {}
        self._load_benchmarks()

    def _load_benchmarks(self):
        """Load CBAM benchmarks from EU Implementing Regulation."""

        # Steel products (CN codes 7208-7229)
        self.benchmarks["steel_hot_rolled_coil"] = CBAMBenchmark(
            product_type="steel_hot_rolled_coil",
            production_method="basic_oxygen_furnace",
            benchmark_value=1.85,
            unit="tCO2e/tonne",
            cn_codes=["7208", "7209", "7210", "7211"],
            effective_date="2026-01-01",
            source="EU Implementing Regulation 2023/1773 Annex II",
            notes="Hot rolled coil steel, basic oxygen furnace production"
        )

        self.benchmarks["steel_rebar"] = CBAMBenchmark(
            product_type="steel_rebar",
            production_method="electric_arc_furnace",
            benchmark_value=1.35,
            unit="tCO2e/tonne",
            cn_codes=["7213", "7214"],
            effective_date="2026-01-01",
            source="EU Implementing Regulation 2023/1773 Annex II",
            notes="Reinforcing bars, electric arc furnace (typically uses scrap)"
        )

        self.benchmarks["steel_wire_rod"] = CBAMBenchmark(
            product_type="steel_wire_rod",
            production_method="basic_oxygen_furnace",
            benchmark_value=1.75,
            unit="tCO2e/tonne",
            cn_codes=["7213", "7227"],
            effective_date="2026-01-01",
            source="EU Implementing Regulation 2023/1773 Annex II",
            notes="Wire rod steel products"
        )

        # Cement products (CN codes 2523)
        self.benchmarks["cement_clinker"] = CBAMBenchmark(
            product_type="cement_clinker",
            production_method="dry_process",
            benchmark_value=0.766,
            unit="tCO2e/tonne",
            cn_codes=["2523 10"],
            effective_date="2026-01-01",
            source="EU Implementing Regulation 2023/1773 Annex II",
            notes="Cement clinker, dry process (most common method)"
        )

        self.benchmarks["cement_portland"] = CBAMBenchmark(
            product_type="cement_portland",
            production_method="dry_process",
            benchmark_value=0.670,
            unit="tCO2e/tonne",
            cn_codes=["2523 21", "2523 29"],
            effective_date="2026-01-01",
            source="EU Implementing Regulation 2023/1773 Annex II",
            notes="Portland cement (includes clinker + additions)"
        )

        # Aluminum products (CN codes 7601-7616)
        self.benchmarks["aluminum_unwrought"] = CBAMBenchmark(
            product_type="aluminum_unwrought",
            production_method="electrolysis",
            benchmark_value=8.6,
            unit="tCO2e/tonne",
            cn_codes=["7601"],
            effective_date="2026-01-01",
            source="EU Implementing Regulation 2023/1773 Annex II",
            notes="Primary aluminum, electrolysis (very energy-intensive)"
        )

        self.benchmarks["aluminum_products"] = CBAMBenchmark(
            product_type="aluminum_products",
            production_method="electrolysis",
            benchmark_value=1.5,
            unit="tCO2e/tonne",
            cn_codes=["7604", "7605", "7606", "7607", "7608"],
            effective_date="2026-01-01",
            source="EU Implementing Regulation 2023/1773 Annex II",
            notes="Aluminum products (bars, wire, plates, foil, tubes)"
        )

        # Fertilizers (CN codes 2814, 3102)
        self.benchmarks["fertilizer_ammonia"] = CBAMBenchmark(
            product_type="fertilizer_ammonia",
            production_method="haber_bosch",
            benchmark_value=2.4,
            unit="tCO2e/tonne",
            cn_codes=["2814"],
            effective_date="2026-01-01",
            source="EU Implementing Regulation 2023/1773 Annex II",
            notes="Ammonia production via Haber-Bosch process"
        )

        self.benchmarks["fertilizer_urea"] = CBAMBenchmark(
            product_type="fertilizer_urea",
            production_method="haber_bosch",
            benchmark_value=1.6,
            unit="tCO2e/tonne",
            cn_codes=["3102 10"],
            effective_date="2026-01-01",
            source="EU Implementing Regulation 2023/1773 Annex II",
            notes="Urea fertilizer"
        )

        self.benchmarks["fertilizer_nitric_acid"] = CBAMBenchmark(
            product_type="fertilizer_nitric_acid",
            production_method="oxidation",
            benchmark_value=0.5,
            unit="tCO2e/tonne",
            cn_codes=["2808 00"],
            effective_date="2026-01-01",
            source="EU Implementing Regulation 2023/1773 Annex II",
            notes="Nitric acid for fertilizer production"
        )

        # Electricity (CN code 2716)
        self.benchmarks["electricity"] = CBAMBenchmark(
            product_type="electricity",
            production_method=None,
            benchmark_value=0.429,
            unit="tCO2e/MWh",
            cn_codes=["2716 00 00"],
            effective_date="2026-01-01",
            source="EU Implementing Regulation 2023/1773 Annex II",
            notes="Imported electricity, EU average grid intensity used as default"
        )

        # Hydrogen (CN code 2804)
        self.benchmarks["hydrogen"] = CBAMBenchmark(
            product_type="hydrogen",
            production_method="steam_reforming",
            benchmark_value=10.5,
            unit="tCO2e/tonne",
            cn_codes=["2804 10 00"],
            effective_date="2026-01-01",
            source="EU Implementing Regulation 2023/1773 Annex II",
            notes="Hydrogen via steam methane reforming (grey hydrogen)"
        )

    def lookup(self, product_type: str, production_method: Optional[str] = None) -> Optional[CBAMBenchmark]:
        """
        Look up CBAM benchmark for product type.

        Args:
            product_type: Product type (e.g., "steel_hot_rolled_coil")
            production_method: Production method (optional, may affect benchmark)

        Returns:
            CBAMBenchmark if found, None otherwise
        """
        benchmark = self.benchmarks.get(product_type)

        if benchmark and production_method:
            # Check if production method matches
            if benchmark.production_method and benchmark.production_method != production_method:
                # Try to find alternative benchmark for this production method
                for key, bm in self.benchmarks.items():
                    if product_type in key and bm.production_method == production_method:
                        return bm

        return benchmark

    def list_products(self) -> list:
        """List all CBAM-regulated product types."""
        return list(self.benchmarks.keys())

    def get_by_cn_code(self, cn_code: str) -> Optional[CBAMBenchmark]:
        """Look up benchmark by CN (Combined Nomenclature) code."""
        for benchmark in self.benchmarks.values():
            for code in benchmark.cn_codes:
                if cn_code.startswith(code):
                    return benchmark
        return None


# Global instance
_cbam_db: Optional[CBAMBenchmarkDatabase] = None


def get_cbam_database() -> CBAMBenchmarkDatabase:
    """Get global CBAM benchmark database instance."""
    global _cbam_db
    if _cbam_db is None:
        _cbam_db = CBAMBenchmarkDatabase()
    return _cbam_db
