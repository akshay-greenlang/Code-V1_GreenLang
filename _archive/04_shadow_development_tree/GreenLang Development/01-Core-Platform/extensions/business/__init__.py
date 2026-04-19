"""
GreenLang Business Module

This module provides business logic for pricing, licensing, ROI/TCO calculations,
and usage metering for the GreenLang Climate Intelligence Platform.

Components:
    - PricingCalculator: Calculates pricing, discounts, and quotes
    - ROICalculator: Calculates return on investment
    - TCOCalculator: Calculates total cost of ownership
    - LicenseManager: Manages license validation and feature gating
    - UsageMetering: Tracks and reports usage metrics

Example:
    >>> from greenlang.business import PricingCalculator, LicenseManager
    >>>
    >>> # Calculate pricing for a customer
    >>> calculator = PricingCalculator()
    >>> quote = calculator.generate_quote(
    ...     edition="professional",
    ...     add_ons=["GL-CBAM-APP", "GL-ProcessHeat-APP"],
    ...     term_years=3
    ... )
    >>> print(f"Annual price: ${quote.annual_price:,.2f}")
    >>>
    >>> # Check license entitlements
    >>> license_mgr = LicenseManager(license_key="GL-LIC-XXXXX")
    >>> if license_mgr.has_feature("ml_optimization"):
    ...     enable_ml_features()
"""

from greenlang.business.pricing_calculator import (
    PricingCalculator,
    ROICalculator,
    TCOCalculator,
    Quote,
    ROIResult,
    TCOResult,
    PricingTier,
    DiscountType,
)

from greenlang.business.licensing import (
    LicenseManager,
    License,
    LicenseType,
    LicenseTier,
    LicenseValidationError,
    LicenseExpiredError,
    FeatureNotLicensedError,
)

__all__ = [
    # Pricing
    "PricingCalculator",
    "ROICalculator",
    "TCOCalculator",
    "Quote",
    "ROIResult",
    "TCOResult",
    "PricingTier",
    "DiscountType",
    # Licensing
    "LicenseManager",
    "License",
    "LicenseType",
    "LicenseTier",
    "LicenseValidationError",
    "LicenseExpiredError",
    "FeatureNotLicensedError",
]

__version__ = "1.0.0"
