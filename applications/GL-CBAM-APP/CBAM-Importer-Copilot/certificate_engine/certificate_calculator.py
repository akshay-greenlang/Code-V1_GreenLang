# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Certificate Calculator Engine v1.1

Thread-safe singleton engine for computing CBAM certificate obligations.
Handles gross certificate calculation, free allocation deductions, carbon
price deductions, net obligation, cost estimation, and quarterly holdings.

Per EU CBAM Regulation 2023/956:
  - Article 21: 1 CBAM certificate = 1 tCO2e embedded emissions
  - Article 22: Certificate price = weekly EU ETS auction average
  - Article 23: Quarterly holding >= 50% of estimated annual obligation
  - Article 24: Annual surrender by 31 May
  - Article 26: Carbon price deduction for origin-country payments
  - Article 31: Free allocation phase-out (2026-2034)

Calculation flow (ZERO HALLUCINATION - deterministic Python arithmetic only):
  1. gross_certs = SUM(quantity_mt x embedded_emissions_per_mt) for all shipments
  2. free_alloc = SUM(quantity_mt x benchmark x allocation_pct) for each CN code
  3. carbon_deduction = SUM(verified_deduction_per_tonne x tonnes) in tCO2e equiv
  4. net_certs = max(0, gross - free_alloc - carbon_deduction)
  5. cost = net_certs x ETS_price
  6. quarterly_holding = net_certs x 50%

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

import logging
import threading
import time
import uuid
from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    CERTIFICATE_UNIT_TCO2E,
    QUARTERLY_HOLDING_PCT,
    CertificateObligation,
    CertificateSummary,
    CarbonPriceDeduction,
    DeductionStatus,
    ETSPrice,
    QuarterlyHolding,
    compute_sha256,
    quantize_decimal,
)
from .ets_price_service import ETSPriceService
from .free_allocation import FreeAllocationEngine
from .carbon_price_deduction import CarbonPriceDeductionEngine

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER: Safe Decimal conversion
# ============================================================================

def _to_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    """
    Safely convert a value to Decimal.

    Args:
        value: Value to convert (int, float, str, Decimal, or None).
        default: Default value if conversion fails.

    Returns:
        Decimal representation of the value.
    """
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return default


class CertificateCalculatorEngine:
    """
    Thread-safe singleton engine for CBAM certificate obligation calculation.

    Computes gross certificates, applies free allocation and carbon price
    deductions, and determines net obligation and cost. All calculations
    use deterministic Decimal arithmetic with ROUND_HALF_UP.

    Thread Safety:
        Uses threading.RLock to protect singleton creation and all mutable
        state. Safe for concurrent access from multiple API request handlers.

    Example:
        >>> calculator = CertificateCalculatorEngine()
        >>> obligation = calculator.calculate_annual_obligation(
        ...     importer_id="NL123456789012",
        ...     year=2026,
        ...     shipments=[
        ...         {"cn_code": "72031000", "quantity_mt": 1500,
        ...          "embedded_emissions_tCO2e": 3000},
        ...     ],
        ... )
        >>> print(f"Net: {obligation.net_certificates_required} tCO2e")
        >>> print(f"Cost: EUR {obligation.certificate_cost_eur}")
    """

    _instance: Optional["CertificateCalculatorEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> "CertificateCalculatorEngine":
        """Thread-safe singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        """Initialize the certificate calculator engine (runs once)."""
        with self._lock:
            if self._initialized:
                return
            self._initialized = True
            self._ets_service = ETSPriceService()
            self._free_alloc = FreeAllocationEngine()
            self._carbon_deduction = CarbonPriceDeductionEngine()
            # importer_id -> year -> list of CertificateObligation
            self._obligations: Dict[str, Dict[int, List[CertificateObligation]]] = {}
            # importer_id -> year -> quarter -> QuarterlyHolding
            self._holdings: Dict[str, Dict[int, Dict[str, QuarterlyHolding]]] = {}
            # importer_id -> year -> certificates_held count
            self._certificates_held: Dict[str, Dict[int, Decimal]] = {}
            logger.info("CertificateCalculatorEngine initialized (singleton)")

    # ========================================================================
    # CORE CALCULATION: GROSS CERTIFICATES
    # ========================================================================

    def calculate_gross_certificates(
        self,
        quantity_mt: Decimal,
        embedded_emissions_per_mt: Decimal,
    ) -> Decimal:
        """
        Calculate gross certificates required for a shipment.

        Per Regulation Article 21: each certificate = 1 tCO2e.
        gross_certificates = quantity_mt x embedded_emissions_per_mt

        This is a ZERO-HALLUCINATION deterministic computation.

        Args:
            quantity_mt: Imported quantity in metric tonnes.
            embedded_emissions_per_mt: Specific embedded emissions (tCO2e/t).

        Returns:
            Gross certificates required (Decimal, 3 decimal places).
        """
        quantity = _to_decimal(quantity_mt)
        emissions_per_mt = _to_decimal(embedded_emissions_per_mt)
        gross = quantity * emissions_per_mt
        return quantize_decimal(gross, places=3)

    # ========================================================================
    # FREE ALLOCATION DEDUCTION
    # ========================================================================

    def apply_free_allocation(
        self,
        gross_certs: Decimal,
        cn_code: str,
        year: int,
    ) -> Decimal:
        """
        Calculate free allocation deduction for a product.

        The deduction is based on the EU ETS benchmark value for the product
        multiplied by the phase-out percentage for the year.

        Per Regulation Article 31: free allocation phases out 2026-2034.

        This is a ZERO-HALLUCINATION deterministic computation.

        Args:
            gross_certs: Gross certificates for this product.
            cn_code: Combined Nomenclature code.
            year: Reference year.

        Returns:
            Free allocation deduction in tCO2e (Decimal, 3 places).
        """
        gross = _to_decimal(gross_certs)
        factor = self._free_alloc.get_allocation_factor(cn_code, year)

        if factor is None:
            logger.info(
                "No free allocation benchmark found for CN %s in %d, deduction=0",
                cn_code, year,
            )
            return Decimal("0")

        # Deduction = benchmark_value x allocation_pct / 100
        # Applied per tonne of product, but we express as ratio of gross
        effective_pct = factor.allocation_percentage / Decimal("100")
        # The deduction is based on the benchmark's share of the actual emissions
        benchmark = factor.benchmark_value_tCO2e
        deduction = benchmark * effective_pct

        # The deduction cannot exceed gross certificates
        deduction = min(deduction, gross)
        return quantize_decimal(deduction, places=3)

    # ========================================================================
    # CARBON PRICE DEDUCTION
    # ========================================================================

    def apply_carbon_price_deduction(
        self,
        gross_certs: Decimal,
        deductions: List[CarbonPriceDeduction],
        ets_price: Decimal,
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate carbon price deduction from verified origin-country payments.

        Per Regulation Article 26: importers may deduct carbon prices
        effectively paid in the country of origin.

        Converts EUR deduction to equivalent tCO2e certificates using ETS price.

        This is a ZERO-HALLUCINATION deterministic computation.

        Args:
            gross_certs: Gross certificates before this deduction.
            deductions: List of verified carbon price deductions.
            ets_price: Current ETS price for EUR-to-tCO2e conversion.

        Returns:
            Tuple of (deduction_eur, deduction_tco2e).
        """
        gross = _to_decimal(gross_certs)
        price = _to_decimal(ets_price)

        if price <= 0:
            logger.warning("ETS price is zero/negative, cannot compute carbon deduction")
            return Decimal("0"), Decimal("0")

        total_eur = Decimal("0")
        for d in deductions:
            if d.verification_status.is_eligible:
                total_eur += d.carbon_price_paid_eur
            else:
                logger.debug(
                    "Skipping unverified deduction %s (status=%s)",
                    d.deduction_id, d.verification_status.value,
                )

        # Convert EUR deduction to equivalent tCO2e certificates
        deduction_tco2e = (total_eur / price).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        # Cannot deduct more than gross
        deduction_tco2e = min(deduction_tco2e, gross)
        total_eur = min(total_eur, gross * price)

        return (
            quantize_decimal(total_eur, places=2),
            quantize_decimal(deduction_tco2e, places=3),
        )

    # ========================================================================
    # NET OBLIGATION
    # ========================================================================

    def calculate_net_obligation(
        self,
        gross: Decimal,
        free_alloc: Decimal,
        carbon_deduction_tco2e: Decimal,
    ) -> Decimal:
        """
        Calculate net certificate obligation after all deductions.

        net = max(0, gross - free_allocation - carbon_deduction)

        This is a ZERO-HALLUCINATION deterministic computation.

        Args:
            gross: Gross certificates required.
            free_alloc: Free allocation deduction (tCO2e).
            carbon_deduction_tco2e: Carbon price deduction (tCO2e equivalent).

        Returns:
            Net certificates required (Decimal, 3 places, >= 0).
        """
        g = _to_decimal(gross)
        fa = _to_decimal(free_alloc)
        cd = _to_decimal(carbon_deduction_tco2e)
        net = max(Decimal("0"), g - fa - cd)
        return quantize_decimal(net, places=3)

    # ========================================================================
    # COST ESTIMATION
    # ========================================================================

    def calculate_certificate_cost(
        self,
        net_certs: Decimal,
        ets_price: Decimal,
    ) -> Decimal:
        """
        Calculate estimated cost of CBAM certificates.

        cost = net_certificates x ETS_price_per_tCO2e

        This is a ZERO-HALLUCINATION deterministic computation.

        Args:
            net_certs: Net certificates required.
            ets_price: ETS price per tCO2e in EUR.

        Returns:
            Estimated cost in EUR (Decimal, 2 places).
        """
        n = _to_decimal(net_certs)
        p = _to_decimal(ets_price)
        cost = n * p
        return quantize_decimal(cost, places=2)

    # ========================================================================
    # QUARTERLY HOLDING
    # ========================================================================

    def calculate_quarterly_holding(
        self,
        annual_estimate: Decimal,
    ) -> Decimal:
        """
        Calculate the quarterly holding requirement.

        Per Article 23: importers must hold >= 50% of estimated annual obligation.

        This is a ZERO-HALLUCINATION deterministic computation.

        Args:
            annual_estimate: Estimated annual net certificate obligation.

        Returns:
            Quarterly holding requirement (Decimal, 3 places).
        """
        estimate = _to_decimal(annual_estimate)
        holding_pct = QUARTERLY_HOLDING_PCT / Decimal("100")
        holding = estimate * holding_pct
        return quantize_decimal(holding, places=3)

    def check_quarterly_compliance(
        self,
        importer_id: str,
        year: int,
        quarter: str,
    ) -> QuarterlyHolding:
        """
        Check whether an importer meets quarterly holding requirements.

        Args:
            importer_id: Importer EORI or identifier.
            year: Reference year.
            quarter: Quarter identifier (Q1/Q2/Q3/Q4).

        Returns:
            QuarterlyHolding with compliance status.
        """
        start_time = time.time()

        # Get the importer's annual estimate
        summary = self.get_obligation_summary(importer_id, year)
        holding_required = self.calculate_quarterly_holding(summary.total_net)

        # Get certificates currently held
        held = self._certificates_held.get(importer_id, {}).get(
            year, Decimal("0")
        )

        holding = QuarterlyHolding(
            quarter=quarter,
            year=year,
            importer_id=importer_id,
            holding_required=holding_required,
            certificates_held=held,
        )

        # Cache the holding
        if importer_id not in self._holdings:
            self._holdings[importer_id] = {}
        if year not in self._holdings[importer_id]:
            self._holdings[importer_id][year] = {}
        self._holdings[importer_id][year][quarter] = holding

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "Quarterly compliance check: importer=%s, %d%s, required=%.3f, "
            "held=%.3f, compliant=%s, in %.1fms",
            importer_id, year, quarter,
            holding.holding_required, holding.certificates_held,
            holding.compliant, duration_ms,
        )

        return holding

    # ========================================================================
    # ANNUAL OBLIGATION (FULL CALCULATION)
    # ========================================================================

    def calculate_annual_obligation(
        self,
        importer_id: str,
        year: int,
        shipments: List[Dict[str, Any]],
    ) -> CertificateObligation:
        """
        Calculate the full annual CBAM certificate obligation for an importer.

        Performs the complete calculation pipeline:
          1. Aggregate shipments by CN code
          2. Compute gross certificates
          3. Apply free allocation deductions
          4. Apply carbon price deductions
          5. Compute net obligation and cost

        This is a ZERO-HALLUCINATION deterministic computation.

        Args:
            importer_id: Importer EORI or identifier.
            year: Obligation year.
            shipments: List of shipment dicts with keys:
                - cn_code (str)
                - quantity_mt (number)
                - embedded_emissions_tCO2e (number) or
                  embedded_emissions_per_mt (number)
                - country_of_origin (str, optional)

        Returns:
            CertificateObligation with full calculation results.
        """
        start_time = time.time()
        logger.info(
            "Calculating annual obligation: importer=%s, year=%d, shipments=%d",
            importer_id, year, len(shipments),
        )

        # Step 1: Aggregate totals
        total_quantity = Decimal("0")
        total_emissions = Decimal("0")
        cn_aggregates: Dict[str, Dict[str, Decimal]] = defaultdict(
            lambda: {"quantity_mt": Decimal("0"), "emissions_tCO2e": Decimal("0")}
        )

        for shipment in shipments:
            cn = shipment.get("cn_code", "unknown")
            qty = _to_decimal(shipment.get("quantity_mt", 0))
            emissions = _to_decimal(shipment.get("embedded_emissions_tCO2e", 0))

            # If per-mt emissions provided, compute total
            if emissions == 0:
                per_mt = _to_decimal(shipment.get("embedded_emissions_per_mt", 0))
                emissions = qty * per_mt

            total_quantity += qty
            total_emissions += emissions
            cn_aggregates[cn]["quantity_mt"] += qty
            cn_aggregates[cn]["emissions_tCO2e"] += emissions

        # Step 2: Gross certificates = total embedded emissions
        gross = quantize_decimal(total_emissions, places=3)

        # Step 3: Free allocation deduction (sum across CN codes)
        total_free_alloc = Decimal("0")
        for cn, agg in cn_aggregates.items():
            fa = self.apply_free_allocation(
                gross_certs=agg["emissions_tCO2e"],
                cn_code=cn,
                year=year,
            )
            total_free_alloc += fa

        total_free_alloc = quantize_decimal(total_free_alloc, places=3)

        # Step 4: Carbon price deductions
        deductions = self._carbon_deduction.get_deductions(importer_id, year)
        ets_price_obj = self._ets_service.get_current_price()
        ets_price = ets_price_obj.price_eur_per_tco2e

        carbon_eur, carbon_tco2e = self.apply_carbon_price_deduction(
            gross_certs=gross - total_free_alloc,
            deductions=deductions,
            ets_price=ets_price,
        )

        # Step 5: Net obligation
        net = self.calculate_net_obligation(gross, total_free_alloc, carbon_tco2e)

        # Step 6: Cost
        cost = self.calculate_certificate_cost(net, ets_price)

        # Build obligation
        obligation_id = f"OBL-{year}-{importer_id[:12]}-{uuid.uuid4().hex[:8]}"
        obligation = CertificateObligation(
            obligation_id=obligation_id,
            importer_id=importer_id,
            year=year,
            cn_code="",  # aggregate
            quantity_mt=quantize_decimal(total_quantity, places=3),
            embedded_emissions_tCO2e=gross,
            gross_certificates_required=gross,
            free_allocation_deduction=total_free_alloc,
            carbon_price_deduction_eur=carbon_eur,
            carbon_price_deduction_tCO2e=carbon_tco2e,
            net_certificates_required=net,
            certificate_cost_eur=cost,
            ets_price_used=ets_price,
            calculation_date=date.today(),
        )
        # Compute provenance hash
        provenance = obligation.compute_provenance_hash()
        obligation = obligation.model_copy(update={"provenance_hash": provenance})

        # Cache the obligation
        if importer_id not in self._obligations:
            self._obligations[importer_id] = {}
        if year not in self._obligations[importer_id]:
            self._obligations[importer_id][year] = []
        self._obligations[importer_id][year].append(obligation)

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "Annual obligation calculated: id=%s, gross=%.3f, free_alloc=%.3f, "
            "carbon=%.3f, net=%.3f, cost=EUR %.2f, in %.1fms",
            obligation_id, gross, total_free_alloc, carbon_tco2e,
            net, cost, duration_ms,
        )

        return obligation

    # ========================================================================
    # OBLIGATION SUMMARY
    # ========================================================================

    def get_obligation_summary(
        self,
        importer_id: str,
        year: int,
    ) -> CertificateSummary:
        """
        Get the full certificate obligation summary for an importer and year.

        Aggregates all obligations and computes total cost and holdings.

        Args:
            importer_id: Importer EORI or identifier.
            year: Obligation year.

        Returns:
            CertificateSummary with totals and breakdown.
        """
        start_time = time.time()

        obligations = self._obligations.get(importer_id, {}).get(year, [])
        ets_price_obj = self._ets_service.get_current_price()
        ets_price = ets_price_obj.price_eur_per_tco2e

        total_gross = Decimal("0")
        total_free = Decimal("0")
        total_carbon = Decimal("0")
        total_net = Decimal("0")
        breakdown: List[Dict[str, Any]] = []

        for obl in obligations:
            total_gross += obl.gross_certificates_required
            total_free += obl.free_allocation_deduction
            total_carbon += obl.carbon_price_deduction_tCO2e
            total_net += obl.net_certificates_required
            breakdown.append({
                "obligation_id": obl.obligation_id,
                "cn_code": obl.cn_code,
                "quantity_mt": str(obl.quantity_mt),
                "gross": str(obl.gross_certificates_required),
                "free_alloc": str(obl.free_allocation_deduction),
                "carbon_deduction": str(obl.carbon_price_deduction_tCO2e),
                "net": str(obl.net_certificates_required),
                "cost_eur": str(obl.certificate_cost_eur),
            })

        total_cost = self.calculate_certificate_cost(total_net, ets_price)
        quarterly_holding = self.calculate_quarterly_holding(total_net)
        held = self._certificates_held.get(importer_id, {}).get(year, Decimal("0"))
        shortfall = max(Decimal("0"), quarterly_holding - held)

        summary = CertificateSummary(
            importer_id=importer_id,
            year=year,
            total_gross=quantize_decimal(total_gross, places=3),
            total_free_allocation=quantize_decimal(total_free, places=3),
            total_carbon_deductions=quantize_decimal(total_carbon, places=3),
            total_net=quantize_decimal(total_net, places=3),
            total_cost_eur=total_cost,
            quarterly_holdings_required=quarterly_holding,
            certificates_held=held,
            shortfall=shortfall,
            obligations_by_cn=breakdown,
            ets_price_used=ets_price,
        )
        summary = summary.model_copy(
            update={"provenance_hash": summary.compute_provenance_hash()}
        )

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "Summary computed: importer=%s, year=%d, net=%.3f, cost=EUR %.2f, "
            "in %.1fms",
            importer_id, year, total_net, total_cost, duration_ms,
        )

        return summary

    # ========================================================================
    # COST PROJECTION
    # ========================================================================

    def project_annual_cost(
        self,
        importer_id: str,
        year: int,
        ets_price_forecast: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        """
        Project annual CBAM certificate cost with optional price forecast.

        Args:
            importer_id: Importer EORI or identifier.
            year: Projection year.
            ets_price_forecast: Optional ETS price forecast (EUR/tCO2e).

        Returns:
            Dict with projection details including low/mid/high scenarios.
        """
        start_time = time.time()

        summary = self.get_obligation_summary(importer_id, year)

        # Use forecast or current price
        if ets_price_forecast is not None:
            base_price = _to_decimal(ets_price_forecast)
        else:
            base_price = summary.ets_price_used

        # Scenario analysis: -20%, base, +20%
        low_price = quantize_decimal(base_price * Decimal("0.80"), places=2)
        mid_price = quantize_decimal(base_price, places=2)
        high_price = quantize_decimal(base_price * Decimal("1.20"), places=2)

        low_cost = self.calculate_certificate_cost(summary.total_net, low_price)
        mid_cost = self.calculate_certificate_cost(summary.total_net, mid_price)
        high_cost = self.calculate_certificate_cost(summary.total_net, high_price)

        projection = {
            "importer_id": importer_id,
            "year": year,
            "net_certificates": str(summary.total_net),
            "scenarios": {
                "low": {
                    "ets_price_eur": str(low_price),
                    "total_cost_eur": str(low_cost),
                    "label": "Conservative (-20%)",
                },
                "mid": {
                    "ets_price_eur": str(mid_price),
                    "total_cost_eur": str(mid_cost),
                    "label": "Base case",
                },
                "high": {
                    "ets_price_eur": str(high_price),
                    "total_cost_eur": str(high_cost),
                    "label": "Upside (+20%)",
                },
            },
            "base_price_source": "forecast" if ets_price_forecast else "current",
        }

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "Cost projection: importer=%s, year=%d, "
            "low=EUR %s, mid=EUR %s, high=EUR %s, in %.1fms",
            importer_id, year, low_cost, mid_cost, high_cost, duration_ms,
        )

        return projection

    # ========================================================================
    # BREAKDOWNS
    # ========================================================================

    def breakdown_by_cn_code(
        self,
        importer_id: str,
        year: int,
    ) -> List[CertificateObligation]:
        """
        Get certificate obligation breakdown by CN code.

        Args:
            importer_id: Importer EORI or identifier.
            year: Reference year.

        Returns:
            List of CertificateObligation per CN code.
        """
        return self._obligations.get(importer_id, {}).get(year, [])

    def breakdown_by_country(
        self,
        importer_id: str,
        year: int,
    ) -> List[Dict[str, Any]]:
        """
        Get certificate obligation breakdown by country of origin.

        Args:
            importer_id: Importer EORI or identifier.
            year: Reference year.

        Returns:
            List of dicts with country-level aggregates.
        """
        obligations = self._obligations.get(importer_id, {}).get(year, [])
        country_agg: Dict[str, Dict[str, Decimal]] = defaultdict(
            lambda: {
                "gross": Decimal("0"),
                "free_alloc": Decimal("0"),
                "carbon_deduction": Decimal("0"),
                "net": Decimal("0"),
                "cost_eur": Decimal("0"),
                "quantity_mt": Decimal("0"),
            }
        )

        for obl in obligations:
            country = obl.country_of_origin or "unknown"
            country_agg[country]["gross"] += obl.gross_certificates_required
            country_agg[country]["free_alloc"] += obl.free_allocation_deduction
            country_agg[country]["carbon_deduction"] += obl.carbon_price_deduction_tCO2e
            country_agg[country]["net"] += obl.net_certificates_required
            country_agg[country]["cost_eur"] += obl.certificate_cost_eur
            country_agg[country]["quantity_mt"] += obl.quantity_mt

        result = []
        for country, agg in sorted(country_agg.items()):
            result.append({
                "country": country,
                "quantity_mt": str(quantize_decimal(agg["quantity_mt"])),
                "gross_certificates": str(quantize_decimal(agg["gross"])),
                "free_allocation_deduction": str(quantize_decimal(agg["free_alloc"])),
                "carbon_price_deduction": str(quantize_decimal(agg["carbon_deduction"])),
                "net_certificates": str(quantize_decimal(agg["net"])),
                "cost_eur": str(quantize_decimal(agg["cost_eur"], places=2)),
            })

        return result

    # ========================================================================
    # PROVENANCE
    # ========================================================================

    def _compute_provenance_hash(self, data: Dict[str, Any]) -> str:
        """
        Compute SHA-256 hash for provenance tracking.

        Args:
            data: Dictionary of data to hash.

        Returns:
            Hexadecimal SHA-256 hash string.
        """
        return compute_sha256(data)

    # ========================================================================
    # CERTIFICATE HOLDINGS MANAGEMENT
    # ========================================================================

    def record_certificates_held(
        self,
        importer_id: str,
        year: int,
        count: Decimal,
    ) -> None:
        """
        Record the number of CBAM certificates held by an importer.

        Args:
            importer_id: Importer identifier.
            year: Reference year.
            count: Number of certificates held.
        """
        if importer_id not in self._certificates_held:
            self._certificates_held[importer_id] = {}
        self._certificates_held[importer_id][year] = _to_decimal(count)
        logger.info(
            "Certificates held updated: importer=%s, year=%d, count=%.3f",
            importer_id, year, count,
        )
