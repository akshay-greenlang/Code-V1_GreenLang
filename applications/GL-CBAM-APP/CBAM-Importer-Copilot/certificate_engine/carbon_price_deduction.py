# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Carbon Price Deduction Engine v1.1

Thread-safe singleton engine for managing carbon price deductions under
EU CBAM Regulation 2023/956 Article 26.  Importers may deduct carbon
prices effectively paid in the country of origin for embedded emissions
in CBAM goods.

Key responsibilities:
  - Register new carbon price deduction claims with ECB FX conversion
  - Verification workflow (PENDING -> VERIFIED -> APPROVED | REJECTED)
  - Evidence document management
  - Country carbon pricing reference data (50+ jurisdictions)
  - Deduction aggregation and summary reporting
  - SHA-256 provenance hashing for full audit trail

Regulatory references:
  - Article 26(1): Deduction for carbon price paid in country of origin
  - Article 26(2): Verification of actual payment and evidence
  - Article 26(3): Conditions (no free allocation, no rebate, not offset)
  - Article 26(4): Calculation based on average carbon price paid
  - Implementing Regulation 2023/1773 Articles 7-8: Evidence requirements

All financial calculations use Decimal with ROUND_HALF_UP for regulatory
precision.  This is a ZERO-HALLUCINATION module -- no LLM calls for any
numeric computation.

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

import logging
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .models import (
    CarbonPriceDeduction,
    CarbonPricingScheme,
    DeductionStatus,
    ECB_EXCHANGE_RATES,
    compute_sha256,
    quantize_decimal,
)

logger = logging.getLogger(__name__)


# ============================================================================
# COUNTRY CARBON PRICING REFERENCE DATA
# ============================================================================
# Source: World Bank Carbon Pricing Dashboard, ICAP ETS Map,
#         IMF Carbon Price Tracker (as of Jan 2026)
#
# Each entry:
#   scheme   - CarbonPricingScheme enum value
#   price    - Effective carbon price per tCO2e in local currency
#   currency - ISO 4217 currency code
#   name     - Scheme name
#   notes    - Additional regulatory context

COUNTRY_CARBON_PRICING: Dict[str, Dict[str, Any]] = {
    # ---- EMISSIONS TRADING SYSTEMS ----
    "CN": {
        "scheme": CarbonPricingScheme.ETS,
        "price": Decimal("72.00"),
        "currency": "CNY",
        "name": "China National ETS",
        "sector": "Power generation",
        "start_year": 2021,
        "notes": "Covers ~4.5 GtCO2e; power sector only",
    },
    "KR": {
        "scheme": CarbonPricingScheme.ETS,
        "price": Decimal("12000.00"),
        "currency": "KRW",
        "name": "Korea ETS (K-ETS)",
        "sector": "Industry, power, buildings, transport, waste",
        "start_year": 2015,
        "notes": "Covers ~73% of national GHG emissions",
    },
    "NZ": {
        "scheme": CarbonPricingScheme.ETS,
        "price": Decimal("70.00"),
        "currency": "NZD",
        "name": "New Zealand ETS (NZ ETS)",
        "sector": "All sectors except agriculture",
        "start_year": 2008,
        "notes": "Broad coverage with cost containment reserve",
    },
    "CH": {
        "scheme": CarbonPricingScheme.HYBRID,
        "price": Decimal("130.00"),
        "currency": "CHF",
        "name": "Swiss ETS + CO2 Levy",
        "sector": "Industry, heating fuels",
        "start_year": 2008,
        "notes": "Linked to EU ETS since 2020",
    },
    "GB": {
        "scheme": CarbonPricingScheme.ETS,
        "price": Decimal("48.00"),
        "currency": "GBP",
        "name": "UK ETS",
        "sector": "Energy-intensive industry, power, aviation",
        "start_year": 2021,
        "notes": "Post-Brexit replacement for EU ETS participation",
    },
    "CA": {
        "scheme": CarbonPricingScheme.HYBRID,
        "price": Decimal("80.00"),
        "currency": "CAD",
        "name": "Canadian Federal Carbon Pricing",
        "sector": "All sectors (backstop + OBPS)",
        "start_year": 2019,
        "notes": "Federal backstop + provincial systems; rising to CAD 170 by 2030",
    },
    "JP": {
        "scheme": CarbonPricingScheme.CARBON_TAX,
        "price": Decimal("289.00"),
        "currency": "JPY",
        "name": "Japan Tax for Climate Change Mitigation",
        "sector": "Fossil fuels (upstream)",
        "start_year": 2012,
        "notes": "JPY 289/tCO2e; GX-ETS voluntary phase started 2023",
    },
    "MX": {
        "scheme": CarbonPricingScheme.CARBON_TAX,
        "price": Decimal("58.87"),
        "currency": "MXN",
        "name": "Mexico Carbon Tax",
        "sector": "Fossil fuels (excl. natural gas)",
        "start_year": 2014,
        "notes": "Pilot ETS launched 2020; tax coexists",
    },
    "ZA": {
        "scheme": CarbonPricingScheme.CARBON_TAX,
        "price": Decimal("190.00"),
        "currency": "ZAR",
        "name": "South Africa Carbon Tax",
        "sector": "Industry, power, buildings, transport",
        "start_year": 2019,
        "notes": "Phase 1 rate with allowances; rising schedule",
    },
    "CO": {
        "scheme": CarbonPricingScheme.CARBON_TAX,
        "price": Decimal("26200.00"),
        "currency": "COP",
        "name": "Colombia Carbon Tax",
        "sector": "Fossil fuels",
        "start_year": 2017,
        "notes": "Offset mechanism available; covers ~24% of emissions",
    },
    "CL": {
        "scheme": CarbonPricingScheme.CARBON_TAX,
        "price": Decimal("5.00"),
        "currency": "USD",
        "name": "Chile Carbon Tax (Green Tax)",
        "sector": "Boilers and turbines >= 50 MWt",
        "start_year": 2017,
        "notes": "USD 5/tCO2e; reform planned to increase rate",
    },
    "SG": {
        "scheme": CarbonPricingScheme.CARBON_TAX,
        "price": Decimal("25.00"),
        "currency": "SGD",
        "name": "Singapore Carbon Tax",
        "sector": "All facilities >= 25 ktCO2e/yr",
        "start_year": 2019,
        "notes": "Rising to SGD 50-80 by 2030; international credits allowed",
    },
    "UA": {
        "scheme": CarbonPricingScheme.CARBON_TAX,
        "price": Decimal("30.00"),
        "currency": "UAH",
        "name": "Ukraine Environmental Tax (CO2 component)",
        "sector": "Stationary combustion",
        "start_year": 2011,
        "notes": "Nominal rate; EU ETS alignment in progress",
    },
    "TR": {
        "scheme": CarbonPricingScheme.ETS,
        "price": Decimal("120.00"),
        "currency": "TRY",
        "name": "Turkey ETS (pilot)",
        "sector": "Energy, industry",
        "start_year": 2024,
        "notes": "Pilot ETS; aligned with EU CBAM preparation",
    },
    "BR": {
        "scheme": CarbonPricingScheme.ETS,
        "price": Decimal("45.00"),
        "currency": "BRL",
        "name": "Brazil ETS (SBCE)",
        "sector": "Industry, power",
        "start_year": 2025,
        "notes": "Regulated carbon market law approved Dec 2024",
    },
    "IN": {
        "scheme": CarbonPricingScheme.CARBON_TAX,
        "price": Decimal("400.00"),
        "currency": "INR",
        "name": "India Carbon Credit Trading Scheme",
        "sector": "Energy-intensive sectors",
        "start_year": 2023,
        "notes": "Replaces PAT scheme; CCTS framework",
    },
    "ID": {
        "scheme": CarbonPricingScheme.CARBON_TAX,
        "price": Decimal("30.00"),
        "currency": "USD",
        "name": "Indonesia Carbon Tax (planned)",
        "sector": "Coal-fired power plants (initial)",
        "start_year": 2025,
        "notes": "Linked to domestic ETS pilot; phased approach",
    },
    "TH": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "THB",
        "name": "Thailand (voluntary market only)",
        "sector": "Voluntary",
        "start_year": 0,
        "notes": "T-VER voluntary scheme; no mandatory carbon price",
    },
    "VN": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "VND",
        "name": "Vietnam (ETS planned for 2028)",
        "sector": "Planned",
        "start_year": 0,
        "notes": "Domestic carbon credit exchange launched 2025; ETS by 2028",
    },
    "EG": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "EGP",
        "name": "Egypt (no carbon pricing)",
        "sector": "None",
        "start_year": 0,
        "notes": "No mandatory carbon pricing mechanism",
    },
    "SA": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "SAR",
        "name": "Saudi Arabia (no carbon pricing)",
        "sector": "None",
        "start_year": 0,
        "notes": "No mandatory carbon pricing mechanism",
    },
    "AE": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "AED",
        "name": "UAE (no carbon pricing)",
        "sector": "None",
        "start_year": 0,
        "notes": "Voluntary carbon credit scheme only",
    },
    "RU": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "RUB",
        "name": "Russia (Sakhalin pilot only)",
        "sector": "Sakhalin region pilot",
        "start_year": 2025,
        "notes": "Sakhalin regional ETS pilot; no national scheme",
    },
    "BY": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "BYN",
        "name": "Belarus (no carbon pricing)",
        "sector": "None",
        "start_year": 0,
        "notes": "No mandatory carbon pricing mechanism",
    },
    "RS": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "RSD",
        "name": "Serbia (EU accession alignment planned)",
        "sector": "Planned",
        "start_year": 0,
        "notes": "Working toward EU ETS compatibility for accession",
    },
    "BA": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "BAM",
        "name": "Bosnia and Herzegovina (no carbon pricing)",
        "sector": "None",
        "start_year": 0,
        "notes": "No mandatory carbon pricing mechanism",
    },
    "AL": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "ALL",
        "name": "Albania (EU accession alignment planned)",
        "sector": "Planned",
        "start_year": 0,
        "notes": "Energy Community member; working toward alignment",
    },
    "MK": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "MKD",
        "name": "North Macedonia (no carbon pricing)",
        "sector": "None",
        "start_year": 0,
        "notes": "No mandatory carbon pricing mechanism",
    },
    "ME": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "EUR",
        "name": "Montenegro (EU accession alignment planned)",
        "sector": "Planned",
        "start_year": 0,
        "notes": "Energy Community member; working toward alignment",
    },
    "NO": {
        "scheme": CarbonPricingScheme.HYBRID,
        "price": Decimal("952.00"),
        "currency": "NOK",
        "name": "Norway Carbon Tax + EU ETS",
        "sector": "Petroleum, industry, aviation",
        "start_year": 1991,
        "notes": "EU ETS participant; additional CO2 tax on petroleum",
    },
    "SE": {
        "scheme": CarbonPricingScheme.HYBRID,
        "price": Decimal("1330.00"),
        "currency": "SEK",
        "name": "Sweden Carbon Tax + EU ETS",
        "sector": "Heating, transport fuels",
        "start_year": 1991,
        "notes": "Highest effective carbon tax globally; EU ETS participant",
    },
    "DK": {
        "scheme": CarbonPricingScheme.HYBRID,
        "price": Decimal("180.00"),
        "currency": "DKK",
        "name": "Denmark CO2 Tax + EU ETS",
        "sector": "Non-ETS sectors + additional levy",
        "start_year": 1992,
        "notes": "New high-rate CO2 tax from 2025 (DKK 750/t by 2030)",
    },
    "FI": {
        "scheme": CarbonPricingScheme.HYBRID,
        "price": Decimal("77.00"),
        "currency": "EUR",
        "name": "Finland Carbon Tax + EU ETS",
        "sector": "Transport, heating fuels",
        "start_year": 1990,
        "notes": "First country with carbon tax; EU ETS participant",
    },
    "PL": {
        "scheme": CarbonPricingScheme.ETS,
        "price": Decimal("0.00"),
        "currency": "PLN",
        "name": "Poland (EU ETS only)",
        "sector": "EU ETS sectors",
        "start_year": 2005,
        "notes": "EU ETS participant; no additional carbon tax",
    },
    "CZ": {
        "scheme": CarbonPricingScheme.ETS,
        "price": Decimal("0.00"),
        "currency": "CZK",
        "name": "Czech Republic (EU ETS only)",
        "sector": "EU ETS sectors",
        "start_year": 2005,
        "notes": "EU ETS participant; no additional carbon tax",
    },
    "AU": {
        "scheme": CarbonPricingScheme.ETS,
        "price": Decimal("38.00"),
        "currency": "AUD",
        "name": "Australia Safeguard Mechanism",
        "sector": "Facilities >= 100 ktCO2e/yr",
        "start_year": 2023,
        "notes": "Reformed Safeguard Mechanism with declining baselines",
    },
    "TW": {
        "scheme": CarbonPricingScheme.CARBON_TAX,
        "price": Decimal("300.00"),
        "currency": "TWD",
        "name": "Taiwan Carbon Fee",
        "sector": "Large emitters >= 25 ktCO2e/yr",
        "start_year": 2025,
        "notes": "Initial rate TWD 300/tCO2e; phased introduction",
    },
    "KZ": {
        "scheme": CarbonPricingScheme.ETS,
        "price": Decimal("1500.00"),
        "currency": "KZT",
        "name": "Kazakhstan ETS",
        "sector": "Energy, industry, mining",
        "start_year": 2013,
        "notes": "First ETS in Central Asia; relaunched 2018",
    },
    "MA": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "MAD",
        "name": "Morocco (no mandatory carbon pricing)",
        "sector": "None",
        "start_year": 0,
        "notes": "Voluntary carbon credit market only",
    },
    "TN": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "TND",
        "name": "Tunisia (no carbon pricing)",
        "sector": "None",
        "start_year": 0,
        "notes": "No mandatory carbon pricing mechanism",
    },
    "DZ": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "DZD",
        "name": "Algeria (no carbon pricing)",
        "sector": "None",
        "start_year": 0,
        "notes": "No mandatory carbon pricing mechanism",
    },
    "PK": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "PKR",
        "name": "Pakistan (no carbon pricing)",
        "sector": "None",
        "start_year": 0,
        "notes": "No mandatory carbon pricing mechanism",
    },
    "BD": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "BDT",
        "name": "Bangladesh (no carbon pricing)",
        "sector": "None",
        "start_year": 0,
        "notes": "No mandatory carbon pricing mechanism",
    },
    "NG": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "NGN",
        "name": "Nigeria (no carbon pricing)",
        "sector": "None",
        "start_year": 0,
        "notes": "No mandatory carbon pricing mechanism",
    },
    "GH": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "GHS",
        "name": "Ghana (no carbon pricing)",
        "sector": "None",
        "start_year": 0,
        "notes": "No mandatory carbon pricing mechanism",
    },
    "KE": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "KES",
        "name": "Kenya (no carbon pricing)",
        "sector": "None",
        "start_year": 0,
        "notes": "Climate Change Act 2016; no carbon price",
    },
    "AR": {
        "scheme": CarbonPricingScheme.CARBON_TAX,
        "price": Decimal("5.00"),
        "currency": "USD",
        "name": "Argentina Carbon Tax",
        "sector": "Liquid and solid fuels",
        "start_year": 2018,
        "notes": "USD ~5/tCO2e equivalent; covers ~20% of GHG",
    },
    "UY": {
        "scheme": CarbonPricingScheme.CARBON_TAX,
        "price": Decimal("137.00"),
        "currency": "USD",
        "name": "Uruguay IMESI Carbon Tax",
        "sector": "Gasoline",
        "start_year": 2022,
        "notes": "Implicit carbon tax via fuel excise",
    },
    "IS": {
        "scheme": CarbonPricingScheme.HYBRID,
        "price": Decimal("35.00"),
        "currency": "EUR",
        "name": "Iceland Carbon Tax + EU ETS",
        "sector": "Fossil fuels, industry",
        "start_year": 2010,
        "notes": "EEA/EU ETS participant + domestic carbon tax",
    },
    "LI": {
        "scheme": CarbonPricingScheme.HYBRID,
        "price": Decimal("130.00"),
        "currency": "CHF",
        "name": "Liechtenstein (Swiss ETS linked)",
        "sector": "Industry, heating",
        "start_year": 2008,
        "notes": "Linked to Swiss/EU ETS via EEA",
    },
    "XK": {
        "scheme": CarbonPricingScheme.NONE,
        "price": Decimal("0.00"),
        "currency": "EUR",
        "name": "Kosovo (no carbon pricing)",
        "sector": "None",
        "start_year": 0,
        "notes": "Energy Community member; no mandatory pricing",
    },
}

# Countries where carbon price has been verified eligible for Article 26
# deduction (effective carbon price actually paid, not rebated or offset)
ELIGIBLE_SCHEMES: Dict[str, bool] = {
    "CN": True,    # China National ETS
    "KR": True,    # Korea ETS
    "NZ": True,    # New Zealand ETS
    "CH": True,    # Swiss ETS (linked to EU ETS)
    "GB": True,    # UK ETS
    "CA": True,    # Canadian Federal Carbon Pricing
    "JP": True,    # Japan Climate Tax
    "MX": True,    # Mexico Carbon Tax
    "ZA": True,    # South Africa Carbon Tax
    "CO": True,    # Colombia Carbon Tax
    "CL": True,    # Chile Green Tax
    "SG": True,    # Singapore Carbon Tax
    "TR": True,    # Turkey ETS (pilot)
    "BR": True,    # Brazil SBCE
    "IN": True,    # India CCTS
    "AU": True,    # Australia Safeguard Mechanism
    "NO": True,    # Norway (EU ETS + tax, but outside EU)
    "IS": True,    # Iceland (EEA/EU ETS + tax)
    "AR": True,    # Argentina Carbon Tax
    "TW": True,    # Taiwan Carbon Fee
    "KZ": True,    # Kazakhstan ETS
    "UA": True,    # Ukraine Environmental Tax
    "UY": True,    # Uruguay IMESI
}


class CarbonPriceDeductionEngine:
    """
    Thread-safe singleton engine for carbon price deduction management.

    Handles the complete lifecycle of Article 26 carbon price deductions:
    registration with ECB FX conversion, verification workflow, evidence
    management, and aggregation for certificate obligation calculation.

    Thread Safety:
        Uses threading.RLock to protect singleton creation and all mutable
        state.  Safe for concurrent access from multiple API request handlers.

    Example:
        >>> engine = CarbonPriceDeductionEngine()
        >>> deduction = engine.register_deduction(
        ...     deduction_id="CPD-2026-NL123-001",
        ...     importer_id="NL123456789012",
        ...     installation_id="TR-INSTALL-001",
        ...     country="TR",
        ...     pricing_scheme=CarbonPricingScheme.ETS,
        ...     carbon_price_paid_local=Decimal("500000"),
        ...     currency="TRY",
        ...     tonnes_covered=Decimal("1000"),
        ...     evidence_docs=["tax-receipt-2026-001.pdf"],
        ...     year=2026,
        ... )
        >>> print(f"Deduction: EUR {deduction.deduction_per_tonne_eur}/tCO2e")
    """

    _instance: Optional["CarbonPriceDeductionEngine"] = None
    _lock: threading.RLock = threading.RLock()

    def __new__(cls) -> "CarbonPriceDeductionEngine":
        """Thread-safe singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        """Initialize the carbon price deduction engine (runs once)."""
        with self._lock:
            if self._initialized:
                return
            self._initialized = True
            # deduction_id -> CarbonPriceDeduction
            self._deductions: Dict[str, CarbonPriceDeduction] = {}
            # importer_id -> year -> list of deduction_ids
            self._importer_index: Dict[str, Dict[int, List[str]]] = {}
            logger.info("CarbonPriceDeductionEngine initialized (singleton)")

    # ========================================================================
    # ECB EXCHANGE RATE CONVERSION
    # ========================================================================

    def _convert_to_eur(
        self,
        amount_local: Decimal,
        currency: str,
    ) -> tuple[Decimal, Decimal]:
        """
        Convert a local-currency amount to EUR using ECB exchange rates.

        This is a ZERO-HALLUCINATION deterministic computation.

        Args:
            amount_local: Amount in local currency.
            currency: ISO 4217 currency code.

        Returns:
            Tuple of (amount_eur, exchange_rate_used).

        Raises:
            ValueError: If the currency is not in the ECB rate table.
        """
        currency_upper = currency.upper()

        if currency_upper == "EUR":
            return quantize_decimal(amount_local, places=2), Decimal("1.000000")

        rate = ECB_EXCHANGE_RATES.get(currency_upper)
        if rate is None:
            raise ValueError(
                f"Unsupported currency '{currency_upper}'. "
                f"Available: {sorted(ECB_EXCHANGE_RATES.keys())}"
            )

        # ECB rates in our table are already EUR per 1 unit of foreign currency
        amount_eur = (amount_local * rate).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        logger.debug(
            "FX conversion: %s %s x %s = EUR %s",
            amount_local, currency_upper, rate, amount_eur,
        )

        return amount_eur, rate

    # ========================================================================
    # REGISTER DEDUCTION
    # ========================================================================

    def register_deduction(
        self,
        deduction_id: str,
        importer_id: str,
        installation_id: str,
        country: str,
        pricing_scheme: CarbonPricingScheme,
        carbon_price_paid_local: Decimal,
        currency: str,
        tonnes_covered: Decimal,
        evidence_docs: List[str],
        year: int,
    ) -> CarbonPriceDeduction:
        """
        Register a new carbon price deduction claim.

        Performs ECB FX conversion, computes per-tonne deduction rate,
        generates SHA-256 provenance hash, and stores the deduction
        in PENDING status for subsequent verification.

        Per Regulation 2023/956 Article 26(1): importers may claim a
        deduction for carbon prices effectively paid in the country of
        origin for the embedded emissions of CBAM goods.

        This is a ZERO-HALLUCINATION deterministic computation for all
        financial calculations.

        Args:
            deduction_id: Unique deduction identifier (e.g. "CPD-2026-NL123-001").
            importer_id: Importer EORI number or internal identifier.
            installation_id: Installation ID in the country of origin.
            country: ISO 3166-1 alpha-2 country code of the installation.
            pricing_scheme: Type of carbon pricing mechanism.
            carbon_price_paid_local: Amount paid in local currency.
            currency: ISO 4217 currency code of payment.
            tonnes_covered: Tonnes of CO2e covered by this carbon price payment.
            evidence_docs: List of evidence document references.
            year: Reference year for the deduction.

        Returns:
            Created CarbonPriceDeduction with PENDING status.

        Raises:
            ValueError: If deduction_id already exists or FX conversion fails.
        """
        start_time = time.time()

        with self._lock:
            if deduction_id in self._deductions:
                raise ValueError(
                    f"Deduction '{deduction_id}' already exists. "
                    f"Use a unique deduction_id."
                )

        # Validate country
        country_upper = country.upper()
        if len(country_upper) != 2:
            raise ValueError(
                f"Country code must be ISO 3166-1 alpha-2 (2 chars), "
                f"got '{country_upper}'"
            )

        # Validate tonnes
        tonnes = Decimal(str(tonnes_covered))
        if tonnes <= 0:
            raise ValueError("tonnes_covered must be positive")

        # Validate amount
        amount_local = Decimal(str(carbon_price_paid_local))
        if amount_local < 0:
            raise ValueError("carbon_price_paid_local cannot be negative")

        # ECB FX conversion
        amount_eur, exchange_rate = self._convert_to_eur(amount_local, currency)

        # Compute per-tonne deduction
        if tonnes > 0 and amount_eur > 0:
            deduction_per_tonne = (amount_eur / tonnes).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            deduction_per_tonne = Decimal("0.00")

        # Build the deduction record
        deduction = CarbonPriceDeduction(
            deduction_id=deduction_id,
            importer_id=importer_id,
            installation_id=installation_id,
            country=country_upper,
            pricing_scheme=pricing_scheme,
            carbon_price_paid_eur=amount_eur,
            carbon_price_paid_local=quantize_decimal(amount_local, places=2),
            exchange_rate=exchange_rate,
            currency=currency.upper(),
            tonnes_covered=quantize_decimal(tonnes, places=3),
            deduction_per_tonne_eur=deduction_per_tonne,
            evidence_docs=list(evidence_docs),
            verification_status=DeductionStatus.PENDING,
            year=year,
        )

        # Compute provenance hash
        provenance = deduction.compute_provenance_hash()
        deduction = deduction.model_copy(update={"provenance_hash": provenance})

        # Store
        with self._lock:
            self._deductions[deduction_id] = deduction
            if importer_id not in self._importer_index:
                self._importer_index[importer_id] = {}
            if year not in self._importer_index[importer_id]:
                self._importer_index[importer_id][year] = []
            self._importer_index[importer_id][year].append(deduction_id)

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "Deduction registered: id=%s, importer=%s, country=%s, "
            "local=%s %s, eur=%s, per_tonne=%s, in %.1fms",
            deduction_id, importer_id, country_upper,
            amount_local, currency.upper(), amount_eur,
            deduction_per_tonne, duration_ms,
        )

        return deduction

    # ========================================================================
    # RETRIEVAL
    # ========================================================================

    def get_deductions(
        self,
        importer_id: str,
        year: int,
    ) -> List[CarbonPriceDeduction]:
        """
        Get all carbon price deductions for an importer and year.

        This method is called by CertificateCalculatorEngine to compute
        the Article 26 deduction from the gross certificate obligation.

        Args:
            importer_id: Importer EORI or identifier.
            year: Reference year.

        Returns:
            List of CarbonPriceDeduction (all statuses, caller filters).
        """
        with self._lock:
            deduction_ids = (
                self._importer_index
                .get(importer_id, {})
                .get(year, [])
            )
            return [
                self._deductions[did]
                for did in deduction_ids
                if did in self._deductions
            ]

    def get_deduction(
        self,
        deduction_id: str,
    ) -> Optional[CarbonPriceDeduction]:
        """
        Get a specific deduction by ID.

        Args:
            deduction_id: Unique deduction identifier.

        Returns:
            CarbonPriceDeduction if found, None otherwise.
        """
        with self._lock:
            return self._deductions.get(deduction_id)

    # ========================================================================
    # VERIFICATION WORKFLOW
    # ========================================================================

    def verify_deduction(
        self,
        deduction_id: str,
        verified_by: str,
    ) -> CarbonPriceDeduction:
        """
        Mark a deduction as verified.

        Transition: PENDING -> VERIFIED.
        Per Article 26(2), the competent authority must verify that the
        carbon price was effectively paid and not subject to rebate or
        compensation.

        Args:
            deduction_id: Deduction to verify.
            verified_by: Identifier of the verifier.

        Returns:
            Updated CarbonPriceDeduction with VERIFIED status.

        Raises:
            ValueError: If deduction not found or invalid status transition.
        """
        return self._transition_status(
            deduction_id=deduction_id,
            target_status=DeductionStatus.VERIFIED,
            actor=verified_by,
            allowed_from={DeductionStatus.PENDING},
        )

    def approve_deduction(
        self,
        deduction_id: str,
        approved_by: str,
    ) -> CarbonPriceDeduction:
        """
        Mark a deduction as approved.

        Transition: VERIFIED -> APPROVED.
        Final approval confirming the deduction is eligible for
        certificate obligation reduction.

        Args:
            deduction_id: Deduction to approve.
            approved_by: Identifier of the approver.

        Returns:
            Updated CarbonPriceDeduction with APPROVED status.

        Raises:
            ValueError: If deduction not found or invalid status transition.
        """
        return self._transition_status(
            deduction_id=deduction_id,
            target_status=DeductionStatus.APPROVED,
            actor=approved_by,
            allowed_from={DeductionStatus.VERIFIED},
        )

    def reject_deduction(
        self,
        deduction_id: str,
        rejected_by: str,
        reason: str = "",
    ) -> CarbonPriceDeduction:
        """
        Reject a deduction.

        Transition: PENDING | VERIFIED -> REJECTED.
        The deduction will not be eligible for certificate obligation
        reduction.

        Args:
            deduction_id: Deduction to reject.
            rejected_by: Identifier of the rejector.
            reason: Reason for rejection (logged, not stored on model).

        Returns:
            Updated CarbonPriceDeduction with REJECTED status.

        Raises:
            ValueError: If deduction not found or invalid status transition.
        """
        if reason:
            logger.info(
                "Deduction %s rejection reason: %s", deduction_id, reason
            )

        return self._transition_status(
            deduction_id=deduction_id,
            target_status=DeductionStatus.REJECTED,
            actor=rejected_by,
            allowed_from={DeductionStatus.PENDING, DeductionStatus.VERIFIED},
        )

    def _transition_status(
        self,
        deduction_id: str,
        target_status: DeductionStatus,
        actor: str,
        allowed_from: set,
    ) -> CarbonPriceDeduction:
        """
        Execute a status transition on a deduction.

        Args:
            deduction_id: Target deduction.
            target_status: Desired new status.
            actor: Who is performing the transition.
            allowed_from: Set of statuses from which this transition is valid.

        Returns:
            Updated CarbonPriceDeduction.

        Raises:
            ValueError: If deduction not found or transition invalid.
        """
        with self._lock:
            deduction = self._deductions.get(deduction_id)
            if deduction is None:
                raise ValueError(
                    f"Deduction '{deduction_id}' not found"
                )

            current = deduction.verification_status
            if current not in allowed_from:
                raise ValueError(
                    f"Cannot transition deduction '{deduction_id}' from "
                    f"'{current.value}' to '{target_status.value}'. "
                    f"Allowed from: {[s.value for s in allowed_from]}"
                )

            now = datetime.now(timezone.utc)
            updates: Dict[str, Any] = {
                "verification_status": target_status,
                "verified_by": actor,
                "verified_at": now,
            }

            updated = deduction.model_copy(update=updates)
            # Recompute provenance hash after status change
            new_hash = updated.compute_provenance_hash()
            updated = updated.model_copy(update={"provenance_hash": new_hash})
            self._deductions[deduction_id] = updated

        logger.info(
            "Deduction status transition: id=%s, %s -> %s, by=%s",
            deduction_id, current.value, target_status.value, actor,
        )

        return updated

    # ========================================================================
    # EVIDENCE MANAGEMENT
    # ========================================================================

    def add_evidence(
        self,
        deduction_id: str,
        document_ref: str,
    ) -> CarbonPriceDeduction:
        """
        Add an evidence document reference to a deduction.

        Per Implementing Regulation 2023/1773 Article 7, importers must
        provide evidence of the carbon price effectively paid:
          - Tax receipts or payment confirmations
          - ETS allowance surrender confirmations
          - Third-party verification statements
          - Installation emissions reports

        Args:
            deduction_id: Target deduction.
            document_ref: Document reference string (filename, URL, or ID).

        Returns:
            Updated CarbonPriceDeduction with the new evidence doc appended.

        Raises:
            ValueError: If deduction not found.
        """
        with self._lock:
            deduction = self._deductions.get(deduction_id)
            if deduction is None:
                raise ValueError(
                    f"Deduction '{deduction_id}' not found"
                )

            current_docs = list(deduction.evidence_docs)
            if document_ref in current_docs:
                logger.warning(
                    "Evidence '%s' already attached to deduction %s",
                    document_ref, deduction_id,
                )
                return deduction

            current_docs.append(document_ref)
            updated = deduction.model_copy(
                update={"evidence_docs": current_docs}
            )
            self._deductions[deduction_id] = updated

        logger.info(
            "Evidence added: deduction=%s, doc=%s, total_docs=%d",
            deduction_id, document_ref, len(current_docs),
        )

        return updated

    # ========================================================================
    # COUNTRY CARBON PRICING LOOKUP
    # ========================================================================

    def get_country_carbon_pricing(
        self,
        country: str,
    ) -> Dict[str, Any]:
        """
        Get carbon pricing information for a country.

        Returns the pricing scheme type, effective price, currency,
        and eligibility for CBAM Article 26 deduction.

        Args:
            country: ISO 3166-1 alpha-2 country code.

        Returns:
            Dict with scheme info, price, currency, eligibility, and notes.
        """
        country_upper = country.upper()
        pricing = COUNTRY_CARBON_PRICING.get(country_upper)

        if pricing is None:
            return {
                "country": country_upper,
                "scheme": CarbonPricingScheme.NONE.value,
                "scheme_description": CarbonPricingScheme.NONE.description,
                "price_per_tco2e": "0.00",
                "currency": "N/A",
                "name": "Unknown country or no data available",
                "eligible_for_deduction": False,
                "notes": "Country not found in carbon pricing reference data",
            }

        # Compute EUR equivalent for reference
        price_local = pricing["price"]
        currency = pricing["currency"]
        try:
            price_eur, _ = self._convert_to_eur(price_local, currency)
        except ValueError:
            price_eur = Decimal("0.00")

        eligible = ELIGIBLE_SCHEMES.get(country_upper, False)
        scheme: CarbonPricingScheme = pricing["scheme"]

        return {
            "country": country_upper,
            "scheme": scheme.value,
            "scheme_description": scheme.description,
            "name": pricing["name"],
            "price_per_tco2e_local": str(price_local),
            "price_per_tco2e_eur": str(price_eur),
            "currency": currency,
            "sector": pricing.get("sector", ""),
            "start_year": pricing.get("start_year", 0),
            "eligible_for_deduction": eligible,
            "notes": pricing.get("notes", ""),
        }

    # ========================================================================
    # AGGREGATION: TOTAL DEDUCTION
    # ========================================================================

    def get_total_deduction_eur(
        self,
        importer_id: str,
        year: int,
    ) -> Decimal:
        """
        Get total eligible deduction amount in EUR for an importer and year.

        Only deductions with VERIFIED or APPROVED status are included,
        per Article 26 eligibility requirements.

        This is a ZERO-HALLUCINATION deterministic computation.

        Args:
            importer_id: Importer EORI or identifier.
            year: Reference year.

        Returns:
            Total eligible deduction in EUR (Decimal, 2 places).
        """
        deductions = self.get_deductions(importer_id, year)
        total = Decimal("0")

        for d in deductions:
            if d.verification_status.is_eligible:
                total += d.carbon_price_paid_eur

        result = quantize_decimal(total, places=2)

        logger.debug(
            "Total deduction: importer=%s, year=%d, EUR %s "
            "(%d eligible of %d total)",
            importer_id, year, result,
            sum(1 for d in deductions if d.verification_status.is_eligible),
            len(deductions),
        )

        return result

    # ========================================================================
    # DEDUCTION SUMMARY
    # ========================================================================

    def get_deduction_summary(
        self,
        importer_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Get a detailed summary of all deductions for an importer and year.

        Provides breakdowns by country and pricing scheme, with totals
        for eligible, pending, and rejected deductions.

        Args:
            importer_id: Importer EORI or identifier.
            year: Reference year.

        Returns:
            Dict with total, country breakdown, scheme breakdown,
            and per-status counts.
        """
        start_time = time.time()
        deductions = self.get_deductions(importer_id, year)

        if not deductions:
            return {
                "importer_id": importer_id,
                "year": year,
                "total_deductions": 0,
                "total_eligible_eur": "0.00",
                "total_pending_eur": "0.00",
                "total_rejected_eur": "0.00",
                "total_tonnes_covered": "0.000",
                "by_country": [],
                "by_scheme": [],
                "deductions": [],
                "provenance_hash": compute_sha256({
                    "importer_id": importer_id,
                    "year": year,
                    "total": "0",
                }),
            }

        # Aggregate by status
        eligible_eur = Decimal("0")
        pending_eur = Decimal("0")
        rejected_eur = Decimal("0")
        total_tonnes = Decimal("0")

        # By-country aggregation
        country_agg: Dict[str, Dict[str, Decimal]] = defaultdict(
            lambda: {
                "eur": Decimal("0"),
                "tonnes": Decimal("0"),
                "count": Decimal("0"),
            }
        )

        # By-scheme aggregation
        scheme_agg: Dict[str, Dict[str, Decimal]] = defaultdict(
            lambda: {
                "eur": Decimal("0"),
                "tonnes": Decimal("0"),
                "count": Decimal("0"),
            }
        )

        deduction_records: List[Dict[str, Any]] = []

        for d in deductions:
            total_tonnes += d.tonnes_covered

            if d.verification_status.is_eligible:
                eligible_eur += d.carbon_price_paid_eur
            elif d.verification_status == DeductionStatus.PENDING:
                pending_eur += d.carbon_price_paid_eur
            elif d.verification_status == DeductionStatus.REJECTED:
                rejected_eur += d.carbon_price_paid_eur

            country_agg[d.country]["eur"] += d.carbon_price_paid_eur
            country_agg[d.country]["tonnes"] += d.tonnes_covered
            country_agg[d.country]["count"] += 1

            scheme_agg[d.pricing_scheme.value]["eur"] += d.carbon_price_paid_eur
            scheme_agg[d.pricing_scheme.value]["tonnes"] += d.tonnes_covered
            scheme_agg[d.pricing_scheme.value]["count"] += 1

            deduction_records.append({
                "deduction_id": d.deduction_id,
                "country": d.country,
                "pricing_scheme": d.pricing_scheme.value,
                "carbon_price_paid_eur": str(d.carbon_price_paid_eur),
                "deduction_per_tonne_eur": str(d.deduction_per_tonne_eur),
                "tonnes_covered": str(d.tonnes_covered),
                "verification_status": d.verification_status.value,
                "evidence_count": len(d.evidence_docs),
            })

        # Build country breakdown
        by_country = []
        for c, agg in sorted(country_agg.items()):
            by_country.append({
                "country": c,
                "total_eur": str(quantize_decimal(agg["eur"], places=2)),
                "tonnes_covered": str(quantize_decimal(agg["tonnes"], places=3)),
                "deduction_count": int(agg["count"]),
            })

        # Build scheme breakdown
        by_scheme = []
        for s, agg in sorted(scheme_agg.items()):
            by_scheme.append({
                "scheme": s,
                "total_eur": str(quantize_decimal(agg["eur"], places=2)),
                "tonnes_covered": str(quantize_decimal(agg["tonnes"], places=3)),
                "deduction_count": int(agg["count"]),
            })

        # Provenance hash for the summary
        provenance_data = {
            "importer_id": importer_id,
            "year": year,
            "total_eligible_eur": str(eligible_eur),
            "total_pending_eur": str(pending_eur),
            "total_rejected_eur": str(rejected_eur),
            "total_tonnes": str(total_tonnes),
            "deduction_count": len(deductions),
        }

        summary = {
            "importer_id": importer_id,
            "year": year,
            "total_deductions": len(deductions),
            "total_eligible_eur": str(quantize_decimal(eligible_eur, places=2)),
            "total_pending_eur": str(quantize_decimal(pending_eur, places=2)),
            "total_rejected_eur": str(quantize_decimal(rejected_eur, places=2)),
            "total_tonnes_covered": str(quantize_decimal(total_tonnes, places=3)),
            "by_country": by_country,
            "by_scheme": by_scheme,
            "deductions": deduction_records,
            "provenance_hash": compute_sha256(provenance_data),
        }

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "Deduction summary: importer=%s, year=%d, count=%d, "
            "eligible=EUR %s, pending=EUR %s, rejected=EUR %s, in %.1fms",
            importer_id, year, len(deductions),
            quantize_decimal(eligible_eur, places=2),
            quantize_decimal(pending_eur, places=2),
            quantize_decimal(rejected_eur, places=2),
            duration_ms,
        )

        return summary

    # ========================================================================
    # INTROSPECTION
    # ========================================================================

    def get_supported_countries(self) -> List[str]:
        """
        Get list of country codes with carbon pricing reference data.

        Returns:
            Sorted list of ISO 3166-1 alpha-2 country codes.
        """
        return sorted(COUNTRY_CARBON_PRICING.keys())

    def get_eligible_countries(self) -> List[str]:
        """
        Get list of countries eligible for Article 26 deductions.

        Returns:
            Sorted list of ISO 3166-1 alpha-2 country codes.
        """
        return sorted(
            code for code, eligible in ELIGIBLE_SCHEMES.items() if eligible
        )

    def get_deduction_count(
        self,
        importer_id: Optional[str] = None,
        year: Optional[int] = None,
    ) -> int:
        """
        Get total deduction count, optionally filtered.

        Args:
            importer_id: Filter by importer (optional).
            year: Filter by year (optional).

        Returns:
            Number of deductions matching the filter.
        """
        with self._lock:
            if importer_id is not None and year is not None:
                return len(
                    self._importer_index
                    .get(importer_id, {})
                    .get(year, [])
                )
            if importer_id is not None:
                return sum(
                    len(ids)
                    for ids in self._importer_index
                    .get(importer_id, {})
                    .values()
                )
            return len(self._deductions)
