"""
SI Unit Pack for GL-FOUND-X-002.

Core SI (International System of Units) definitions.
"""

from greenlang.schema.units.catalog import UnitCatalog, UnitDefinition


def load_si_units(catalog: UnitCatalog) -> None:
    """
    Load SI base and derived units into the catalog.

    Args:
        catalog: Unit catalog to populate
    """
    # Energy units
    catalog.register(UnitDefinition("J", "Joule", "energy", 1.0))
    catalog.register(UnitDefinition("kJ", "Kilojoule", "energy", 1000.0))
    catalog.register(UnitDefinition("MJ", "Megajoule", "energy", 1e6))
    catalog.register(UnitDefinition("Wh", "Watt-hour", "energy", 3600.0))
    catalog.register(UnitDefinition("kWh", "Kilowatt-hour", "energy", 3.6e6, aliases=["KWH", "kwh"]))
    catalog.register(UnitDefinition("MWh", "Megawatt-hour", "energy", 3.6e9))
    catalog.set_canonical("energy", "kWh")

    # Mass units
    catalog.register(UnitDefinition("kg", "Kilogram", "mass", 1.0, aliases=["KG"]))
    catalog.register(UnitDefinition("g", "Gram", "mass", 0.001))
    catalog.register(UnitDefinition("mg", "Milligram", "mass", 1e-6))
    catalog.register(UnitDefinition("t", "Tonne", "mass", 1000.0, aliases=["tonne", "metric ton"]))
    catalog.set_canonical("mass", "kg")

    # Length units
    catalog.register(UnitDefinition("m", "Metre", "length", 1.0, aliases=["meter"]))
    catalog.register(UnitDefinition("km", "Kilometre", "length", 1000.0, aliases=["kilometer"]))
    catalog.register(UnitDefinition("cm", "Centimetre", "length", 0.01))
    catalog.register(UnitDefinition("mm", "Millimetre", "length", 0.001))
    catalog.set_canonical("length", "m")

    # Area units
    catalog.register(UnitDefinition("m2", "Square metre", "area", 1.0, aliases=["m^2", "sq m"]))
    catalog.register(UnitDefinition("km2", "Square kilometre", "area", 1e6, aliases=["km^2"]))
    catalog.register(UnitDefinition("ha", "Hectare", "area", 10000.0))
    catalog.set_canonical("area", "m2")

    # Volume units
    catalog.register(UnitDefinition("m3", "Cubic metre", "volume", 1.0, aliases=["m^3"]))
    catalog.register(UnitDefinition("L", "Litre", "volume", 0.001, aliases=["l", "liter"]))
    catalog.register(UnitDefinition("mL", "Millilitre", "volume", 1e-6, aliases=["ml"]))
    catalog.set_canonical("volume", "m3")

    # Temperature units (with offset)
    catalog.register(UnitDefinition("K", "Kelvin", "temperature", 1.0, 0.0))
    catalog.register(UnitDefinition("C", "Celsius", "temperature", 1.0, 273.15, aliases=["celsius", "degC"]))
    catalog.set_canonical("temperature", "K")

    # Time units
    catalog.register(UnitDefinition("s", "Second", "time", 1.0, aliases=["sec"]))
    catalog.register(UnitDefinition("min", "Minute", "time", 60.0))
    catalog.register(UnitDefinition("h", "Hour", "time", 3600.0, aliases=["hr", "hour"]))
    catalog.register(UnitDefinition("d", "Day", "time", 86400.0, aliases=["day"]))
    catalog.set_canonical("time", "s")

    # Power units
    catalog.register(UnitDefinition("W", "Watt", "power", 1.0))
    catalog.register(UnitDefinition("kW", "Kilowatt", "power", 1000.0))
    catalog.register(UnitDefinition("MW", "Megawatt", "power", 1e6))
    catalog.set_canonical("power", "W")

    # Pressure units
    catalog.register(UnitDefinition("Pa", "Pascal", "pressure", 1.0))
    catalog.register(UnitDefinition("kPa", "Kilopascal", "pressure", 1000.0))
    catalog.register(UnitDefinition("bar", "Bar", "pressure", 100000.0))
    catalog.set_canonical("pressure", "Pa")

    # Force units
    catalog.register(UnitDefinition("N", "Newton", "force", 1.0))
    catalog.register(UnitDefinition("kN", "Kilonewton", "force", 1000.0))
    catalog.set_canonical("force", "N")
