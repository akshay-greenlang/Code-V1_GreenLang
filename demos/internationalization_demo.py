# -*- coding: utf-8 -*-
"""
demos/internationalization_demo.py

Internationalization (i18n) Demo for FuelAgentAI v2

Demonstrates:
- Multi-language support (8 languages)
- Unit conversions (imperial ↔ metric)
- Regional defaults
- Number formatting
- Real-world scenarios across regions

Author: GreenLang Framework Team
Date: October 2025
"""

from greenlang.utils.unit_conversion import (
    UnitConverter,
    get_regional_defaults,
    format_number,
)
from greenlang.i18n.messages import (
    I18n,
    get_supported_languages,
)
import json


def demo_supported_languages():
    """Demo: List supported languages."""
    print("=" * 80)
    print("DEMO 1: Supported Languages")
    print("=" * 80)

    languages = get_supported_languages()

    print("\nFuelAgentAI v2 supports 8 languages:")
    for code, name in languages.items():
        print(f"  [{code}] {name}")


def demo_message_translation():
    """Demo: Message translation across languages."""
    print("\n" + "=" * 80)
    print("DEMO 2: Message Translation")
    print("=" * 80)

    languages = ["en", "es", "fr", "de", "zh", "ja", "pt", "hi"]
    message_key = "calculation_complete"

    print(f"\nTranslation of '{message_key}':")
    for lang in languages:
        i18n = I18n(lang)
        msg = i18n.get(message_key)
        print(f"  [{lang}] {msg}")


def demo_unit_conversions():
    """Demo: Unit conversions across regions."""
    print("\n" + "=" * 80)
    print("DEMO 3: Unit Conversions")
    print("=" * 80)

    converter = UnitConverter()

    # Volume conversions
    print("\n1. Volume Conversions:")
    print(f"   100 US gallons = {converter.convert_volume(100, 'gallons', 'liters'):.2f} liters")
    print(f"   100 liters = {converter.convert_volume(100, 'liters', 'gallons'):.2f} US gallons")
    print(f"   1 m³ = {converter.convert_volume(1, 'm3', 'gallons'):.2f} US gallons")

    # Energy conversions
    print("\n2. Energy Conversions:")
    print(f"   1000 therms = {converter.convert_energy(1000, 'therms', 'kWh'):.2f} kWh")
    print(f"   10,000 kWh = {converter.convert_energy(10000, 'kWh', 'therms'):.2f} therms")
    print(f"   100 GJ = {converter.convert_energy(100, 'GJ', 'kWh'):.2f} kWh")

    # Mass conversions
    print("\n3. Mass Conversions:")
    print(f"   100 US tons = {converter.convert_mass(100, 'tons', 'tonnes'):.2f} metric tonnes")
    print(f"   50 tonnes = {converter.convert_mass(50, 'tonnes', 'kg'):.2f} kg")
    print(f"   2000 lbs = {converter.convert_mass(2000, 'lbs', 'kg'):.2f} kg")

    # Temperature conversions
    print("\n4. Temperature Conversions:")
    print(f"   32°F = {converter.convert_temperature(32, 'F', 'C'):.2f}°C")
    print(f"   100°C = {converter.convert_temperature(100, 'C', 'F'):.2f}°F")
    print(f"   0°C = {converter.convert_temperature(0, 'C', 'K'):.2f} K")


def demo_regional_defaults():
    """Demo: Regional defaults for different countries."""
    print("\n" + "=" * 80)
    print("DEMO 4: Regional Defaults")
    print("=" * 80)

    regions = ["US", "UK", "EU", "CA", "IN", "CN", "JP", "BR"]

    print("\nRegional unit preferences:")
    print(f"{'Region':<10} {'Volume':<15} {'Energy':<10} {'Mass':<10} {'Currency':<10}")
    print("-" * 60)

    for region in regions:
        defaults = get_regional_defaults(region)
        print(f"{region:<10} "
              f"{defaults['volume_unit'].value:<15} "
              f"{defaults['energy_unit'].value:<10} "
              f"{defaults['mass_unit'].value:<10} "
              f"{defaults['currency']:<10}")


def demo_number_formatting():
    """Demo: Number formatting by region."""
    print("\n" + "=" * 80)
    print("DEMO 5: Number Formatting")
    print("=" * 80)

    value = 1234567.89

    regions = {
        "US": "United States",
        "UK": "United Kingdom",
        "EU": "European Union",
        "CA": "Canada",
        "BR": "Brazil",
    }

    print(f"\nFormatting {value}:")
    for code, name in regions.items():
        formatted = format_number(value, code, decimals=2)
        print(f"  {name:<20} {formatted:>15}")


def demo_real_world_scenario_us():
    """Demo: Real-world scenario - US company."""
    print("\n" + "=" * 80)
    print("DEMO 6: Real-World Scenario - US Company")
    print("=" * 80)

    # US company: 1000 gallons diesel
    i18n = I18n("en")
    converter = UnitConverter()

    print("\nCompany: Acme Corp (United States)")
    print("Fuel consumption: 1,000 gallons diesel")

    # Emission calculation would happen here
    emissions_kg = 10210.0  # Hypothetical result

    # Format for US audience
    print(f"\n{i18n.get('total_emissions')}: {format_number(emissions_kg, 'US', 0)} kgCO2e")
    print(f"{i18n.get('fuel_consumption')}: {format_number(1000, 'US', 0)} gallons")
    print(f"{i18n.get('emission_factor')}: 10.21 kgCO2e/gallon")


def demo_real_world_scenario_eu():
    """Demo: Real-world scenario - EU company."""
    print("\n" + "=" * 80)
    print("DEMO 7: Real-World Scenario - EU Company")
    print("=" * 80)

    # EU company: Same emissions, but in liters
    i18n = I18n("de")  # German
    converter = UnitConverter()

    # Convert 1000 gallons to liters
    liters = converter.convert_volume(1000, "gallons", "liters")

    print("\nUnternehmen: Acme GmbH (Deutschland)")
    print(f"Kraftstoffverbrauch: {format_number(liters, 'EU', 0)} Liter Diesel")

    emissions_kg = 10210.0

    # Format for EU audience (German)
    print(f"\n{i18n.get('total_emissions')}: {format_number(emissions_kg, 'EU', 0)} kgCO2e")
    print(f"{i18n.get('fuel_consumption')}: {format_number(liters, 'EU', 0)} Liter")
    print(f"{i18n.get('emission_factor')}: {format_number(2.70, 'EU', 2)} kgCO2e/Liter")


def demo_real_world_scenario_china():
    """Demo: Real-world scenario - China company."""
    print("\n" + "=" * 80)
    print("DEMO 8: Real-World Scenario - China Company")
    print("=" * 80)

    # China company: Natural gas in kWh
    i18n = I18n("zh")  # Chinese
    converter = UnitConverter()

    # 10,000 therms = 293,001 kWh
    kwh = converter.convert_energy(10000, "therms", "kWh")

    print("\n公司: 绿色能源有限公司 (中国)")
    print(f"天然气消耗: {format_number(kwh, 'CN', 0)} kWh")

    emissions_kg = 53010.0

    # Format for China audience
    print(f"\n{i18n.get('total_emissions')}: {format_number(emissions_kg, 'CN', 0)} kgCO2e")
    print(f"{i18n.get('fuel_consumption')}: {format_number(kwh, 'CN', 0)} kWh")
    print(f"{i18n.get('emission_factor')}: {format_number(0.181, 'CN', 3)} kgCO2e/kWh")


def demo_multilingual_recommendations():
    """Demo: Multi-language recommendations."""
    print("\n" + "=" * 80)
    print("DEMO 9: Multi-Language Recommendations")
    print("=" * 80)

    recommendations = [
        {
            "action": "Switch to biodiesel (B20)",
            "potential_reduction_pct": 15,
            "feasibility": "high",
        },
        {
            "action": "Upgrade to electric vehicle fleet",
            "potential_reduction_pct": 65,
            "feasibility": "medium",
        },
    ]

    languages = ["en", "es", "fr"]

    print("\nRecommendations for diesel fleet:")
    for lang in languages:
        i18n = I18n(lang)
        print(f"\n[{lang.upper()}] {i18n.get('recommendations')}:")

        # In production, this would use i18n.translate_recommendations()
        # For demo, just show the header translation
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec['action']} (-{rec['potential_reduction_pct']}%)")


def demo_cross_region_comparison():
    """Demo: Cross-region comparison."""
    print("\n" + "=" * 80)
    print("DEMO 10: Cross-Region Comparison")
    print("=" * 80)

    print("\nSame consumption (1000 gallons diesel), different regions:")
    print(f"{'Region':<10} {'Local Units':<25} {'Emissions (kgCO2e)':<20} {'Formatted':<15}")
    print("-" * 75)

    converter = UnitConverter()
    emissions = 10210.0

    # US
    print(f"{'US':<10} {'1,000 gallons':<25} {emissions:<20.2f} "
          f"{format_number(emissions, 'US', 0):<15}")

    # UK
    liters_uk = converter.convert_volume(1000, "gallons", "liters")
    print(f"{'UK':<10} {f'{liters_uk:.0f} liters':<25} {emissions:<20.2f} "
          f"{format_number(emissions, 'UK', 0):<15}")

    # EU
    liters_eu = converter.convert_volume(1000, "gallons", "liters")
    print(f"{'EU':<10} {f'{liters_eu:.0f} liters':<25} {emissions:<20.2f} "
          f"{format_number(emissions, 'EU', 0):<15}")

    # China
    print(f"{'China':<10} {f'{liters_eu:.0f} liters':<25} {emissions:<20.2f} "
          f"{format_number(emissions, 'CN', 0):<15}")


def demo_temperature_scenarios():
    """Demo: Temperature conversion scenarios."""
    print("\n" + "=" * 80)
    print("DEMO 11: Temperature Conversion Scenarios")
    print("=" * 80)

    converter = UnitConverter()

    print("\nCommon temperature conversions:")

    scenarios = [
        ("Freezing point of water", 32, "F"),
        ("Room temperature", 20, "C"),
        ("Boiling point of water", 100, "C"),
        ("Absolute zero", 0, "K"),
    ]

    for name, value, unit in scenarios:
        if unit == "F":
            celsius = converter.convert_temperature(value, "F", "C")
            kelvin = converter.convert_temperature(value, "F", "K")
            print(f"\n{name}: {value}°F = {celsius:.1f}°C = {kelvin:.1f} K")
        elif unit == "C":
            fahrenheit = converter.convert_temperature(value, "C", "F")
            kelvin = converter.convert_temperature(value, "C", "K")
            print(f"\n{name}: {value}°C = {fahrenheit:.1f}°F = {kelvin:.1f} K")
        elif unit == "K":
            celsius = converter.convert_temperature(value, "K", "C")
            fahrenheit = converter.convert_temperature(value, "K", "F")
            print(f"\n{name}: {value} K = {celsius:.1f}°C = {fahrenheit:.1f}°F")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("INTERNATIONALIZATION DEMO")
    print("FuelAgentAI v2 - Multi-Language & Multi-Region Support")
    print("=" * 80)

    demo_supported_languages()
    demo_message_translation()
    demo_unit_conversions()
    demo_regional_defaults()
    demo_number_formatting()
    demo_real_world_scenario_us()
    demo_real_world_scenario_eu()
    demo_real_world_scenario_china()
    demo_multilingual_recommendations()
    demo_cross_region_comparison()
    demo_temperature_scenarios()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nKey Features:")
    print("  • 8 languages supported (EN, ES, FR, DE, ZH, JA, PT, HI)")
    print("  • 10+ regions with specific defaults")
    print("  • Comprehensive unit conversions (volume, energy, mass, temperature, pressure)")
    print("  • Locale-aware number formatting")
    print("  • Seamless cross-region comparisons")
    print()
