"""
Climate Unit Pack for GL-FOUND-X-002.

Climate and emissions-specific units.
"""

from greenlang.schema.units.catalog import UnitCatalog, UnitDefinition


def load_climate_units(catalog: UnitCatalog) -> None:
    """
    Load climate and emissions units into the catalog.

    Args:
        catalog: Unit catalog to populate
    """
    # Emissions units (CO2 equivalent)
    catalog.register(UnitDefinition("kgCO2e", "Kilogram CO2 equivalent", "emissions", 1.0,
                                     aliases=["kg CO2e", "kgCO2eq", "kg CO2 equivalent"]))
    catalog.register(UnitDefinition("tCO2e", "Tonne CO2 equivalent", "emissions", 1000.0,
                                     aliases=["t CO2e", "tCO2eq", "metric ton CO2e"]))
    catalog.register(UnitDefinition("gCO2e", "Gram CO2 equivalent", "emissions", 0.001,
                                     aliases=["g CO2e"]))
    catalog.register(UnitDefinition("MtCO2e", "Megatonne CO2 equivalent", "emissions", 1e9,
                                     aliases=["Mt CO2e"]))
    catalog.set_canonical("emissions", "kgCO2e")

    # Emission factors are often per unit of activity
    # These are compound units that may need special handling

    # US customary energy units
    catalog.register(UnitDefinition("BTU", "British Thermal Unit", "energy", 1055.06,
                                     aliases=["Btu", "btu"]))
    catalog.register(UnitDefinition("therm", "Therm", "energy", 105506000.0))
    catalog.register(UnitDefinition("MMBTU", "Million BTU", "energy", 1.055e9,
                                     aliases=["MMBtu", "mmBtu"]))

    # Fuel volume units
    catalog.register(UnitDefinition("gallon", "US gallon", "volume", 0.00378541,
                                     aliases=["gal", "US gallon"]))
    catalog.register(UnitDefinition("barrel", "Oil barrel", "volume", 0.158987,
                                     aliases=["bbl", "oil barrel"]))

    # Distance units (US customary)
    catalog.register(UnitDefinition("mile", "Mile", "length", 1609.344,
                                     aliases=["mi", "miles"]))
    catalog.register(UnitDefinition("ft", "Foot", "length", 0.3048,
                                     aliases=["feet", "foot"]))
    catalog.register(UnitDefinition("in", "Inch", "length", 0.0254,
                                     aliases=["inch", "inches"]))

    # Mass units (US customary)
    catalog.register(UnitDefinition("lb", "Pound", "mass", 0.453592,
                                     aliases=["lbs", "pound", "pounds"]))
    catalog.register(UnitDefinition("oz", "Ounce", "mass", 0.0283495,
                                     aliases=["ounce", "ounces"]))
    catalog.register(UnitDefinition("ton", "Short ton", "mass", 907.185,
                                     aliases=["short ton", "US ton"]))

    # Area units (US customary)
    catalog.register(UnitDefinition("acre", "Acre", "area", 4046.86,
                                     aliases=["acres"]))
    catalog.register(UnitDefinition("sqft", "Square foot", "area", 0.092903,
                                     aliases=["sq ft", "ft2", "ft^2"]))

    # Temperature (Fahrenheit)
    # Note: Fahrenheit needs special handling due to offset
    # F = C * 9/5 + 32, so C = (F - 32) * 5/9
    # K = C + 273.15 = (F - 32) * 5/9 + 273.15
    catalog.register(UnitDefinition("F", "Fahrenheit", "temperature", 5/9, (32 * 5/9) + 273.15,
                                     aliases=["fahrenheit", "degF"]))
