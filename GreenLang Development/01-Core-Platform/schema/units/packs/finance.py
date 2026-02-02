"""
Finance Unit Pack for GL-FOUND-X-002.

Financial and monetary units.
"""

from greenlang.schema.units.catalog import UnitCatalog, UnitDefinition


def load_finance_units(catalog: UnitCatalog) -> None:
    """
    Load financial units into the catalog.

    Note: Currency conversion factors are not static and would
    typically come from an external service. These are placeholders.

    Args:
        catalog: Unit catalog to populate
    """
    # Major currencies (factors are placeholders - should use live rates)
    catalog.register(UnitDefinition("USD", "US Dollar", "currency", 1.0,
                                     aliases=["$", "US$", "dollar", "dollars"]))
    catalog.register(UnitDefinition("EUR", "Euro", "currency", 1.0,  # Placeholder
                                     aliases=["euro", "euros"]))
    catalog.register(UnitDefinition("GBP", "British Pound", "currency", 1.0,  # Placeholder
                                     aliases=["pound", "pounds sterling"]))
    catalog.register(UnitDefinition("JPY", "Japanese Yen", "currency", 1.0,  # Placeholder
                                     aliases=["yen"]))
    catalog.register(UnitDefinition("CNY", "Chinese Yuan", "currency", 1.0,  # Placeholder
                                     aliases=["yuan", "RMB"]))
    catalog.register(UnitDefinition("INR", "Indian Rupee", "currency", 1.0,  # Placeholder
                                     aliases=["rupee", "rupees"]))
    catalog.set_canonical("currency", "USD")

    # Percentage (dimensionless)
    catalog.register(UnitDefinition("%", "Percent", "dimensionless", 0.01,
                                     aliases=["percent", "pct"]))
    catalog.register(UnitDefinition("ppm", "Parts per million", "dimensionless", 1e-6))
    catalog.register(UnitDefinition("ppb", "Parts per billion", "dimensionless", 1e-9))
    catalog.set_canonical("dimensionless", "1")

    # Dimensionless unit
    catalog.register(UnitDefinition("1", "Dimensionless", "dimensionless", 1.0,
                                     aliases=["dimensionless", "unitless"]))
