# GreenLang Business Documentation

This directory contains comprehensive business documentation for the GreenLang Climate Intelligence Platform, including pricing models, licensing frameworks, and packaging guidelines.

## Document Index

| Document | Description | Audience |
|----------|-------------|----------|
| [comparison_charts.md](comparison_charts.md) | Product and competitor comparison matrices | Sales, Marketing |
| [edition_tiers.md](edition_tiers.md) | Good/Better/Best edition definitions | Sales, Product |
| [packaging_guidelines.md](packaging_guidelines.md) | Product bundling and packaging rules | Sales, Product |
| [licensing_framework.md](licensing_framework.md) | License types, terms, and compliance | Legal, Sales |
| [pricing_model.md](pricing_model.md) | Comprehensive pricing and business model | Sales, Finance |

## Quick Reference

### Edition Pricing (Annual)

| Edition | Price | Users | Agents | Key Features |
|---------|-------|-------|--------|--------------|
| Essentials | $100,000 | 10 | 25 | Core monitoring, basic compliance |
| Professional | $250,000 | 50 | 100 | ML optimization, advanced compliance |
| Enterprise | $500,000+ | Unlimited | 500+ | Full suite, Agent Factory, custom |

### Bundle Discounts

| Discount Type | Range |
|---------------|-------|
| Volume | 5-25% |
| Multi-Year (2-5 yr) | 10-20% |
| Multi-Product (2-5) | 5-15% |
| **Maximum Combined** | **40%** |

### License Types

- **Subscription**: Annual/monthly recurring access
- **Perpetual**: One-time purchase + annual maintenance
- **Usage-Based**: Pay per consumption
- **Enterprise Agreement**: Custom terms for large deployments

## Code Modules

The business logic is implemented in:

```
greenlang/business/
    __init__.py           # Module exports
    pricing_calculator.py  # Quote generation, ROI/TCO calculators
    licensing.py          # License management and feature gating
```

### Example Usage

```python
from greenlang.business import PricingCalculator, LicenseManager

# Generate a pricing quote
calculator = PricingCalculator()
quote = calculator.generate_quote(
    customer_name="Acme Corp",
    edition="professional",
    add_ons=["GL-CBAM-APP", "GL-ProcessHeat-APP"],
    term_years=3,
)
print(quote.summary())

# Check license entitlements
license_mgr = LicenseManager(license_key="GL-LIC-XXXX-XXXX-XXXX-XXXX")
if license_mgr.has_feature("ml_optimization"):
    run_optimization()
```

## Related Documents

- [Product Roadmap](../planning/greenlang-2030-vision/GL_PRODUCT_ROADMAP_2025_2030.md)
- [Application Specifications](../planning/greenlang-2030-vision/GL_APPLICATION_SPECIFICATIONS.md)
- [Solution Packs Catalog](../planning/greenlang-2030-vision/GL_SOLUTION_PACKS_CATALOG.md)

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-07 | GL-ProductManager | Initial release (TASK-217-230) |

---

*GreenLang Climate Intelligence Platform - Business Documentation*
