"""
Demo Response Library

Pre-recorded responses for FakeProvider when no API keys are available.
All responses are realistic but clearly marked as demo mode.

Categories:
- Grid intensity calculations
- Fuel emissions calculations
- Generic climate Q&A
- Tool call demonstrations

Demo responses cite "demo mode" and encourage users to add API keys for production use.
"""

from typing import Dict, List, Any


# Demo responses for common climate queries
DEMO_RESPONSES: Dict[str, str] = {
    # Grid intensity queries
    "grid_intensity_ca": """Based on California's energy mix (demo mode):

The carbon intensity of California's electricity grid is approximately 200-250 gCO2e/kWh (grams of CO2 equivalent per kilowatt-hour). This is significantly lower than the US national average of ~400 gCO2e/kWh due to California's high renewable energy penetration.

Key factors:
- 33% renewable energy (solar, wind, geothermal)
- 9% nuclear (zero emissions)
- 34% natural gas
- Remainder: hydroelectric and imports

NOTE: This is a demo response. For production use with real-time grid data, please set your OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.""",

    "grid_intensity_texas": """Based on Texas (ERCOT) grid data (demo mode):

The carbon intensity of the Texas electricity grid (ERCOT) is approximately 400-450 gCO2e/kWh. This is near the US national average.

Energy mix:
- 52% natural gas
- 20% wind (growing rapidly)
- 13% coal
- 11% nuclear
- Remainder: solar and other sources

NOTE: This is a demo response. For production use with real-time grid data, please set your OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.""",

    "grid_intensity_generic": """Carbon intensity of electricity grids (demo mode):

Grid carbon intensity varies significantly by region:
- Clean grids: 50-200 gCO2e/kWh (high renewables/hydro/nuclear)
- Average grids: 300-500 gCO2e/kWh (mixed fossil and renewables)
- Coal-heavy grids: 600-1000+ gCO2e/kWh

US average: ~400 gCO2e/kWh
European average: ~300 gCO2e/kWh
Global average: ~475 gCO2e/kWh

NOTE: This is a demo response. For production use with real-time grid data, please set your OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.""",

    # Fuel emissions queries
    "fuel_emissions_diesel": """Diesel combustion emissions (demo mode):

When burned, diesel fuel produces approximately:
- 10.16 kg CO2e per gallon (US gallon)
- 2.68 kg CO2e per liter

This includes:
- Direct CO2 emissions: 10.15 kg/gallon
- CH4 and N2O emissions: ~0.01 kg CO2e/gallon

For example, 100 gallons of diesel would produce:
100 gallons × 10.16 kg CO2e/gallon = 1,016 kg CO2e (1.016 metric tons)

NOTE: This is a demo response using EPA emission factors. For production calculations, please set your OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.""",

    "fuel_emissions_gasoline": """Gasoline combustion emissions (demo mode):

When burned, gasoline produces approximately:
- 8.89 kg CO2e per gallon (US gallon)
- 2.35 kg CO2e per liter

This includes:
- Direct CO2 emissions: 8.87 kg/gallon
- CH4 and N2O emissions: ~0.02 kg CO2e/gallon

For example, 50 gallons of gasoline would produce:
50 gallons × 8.89 kg CO2e/gallon = 444.5 kg CO2e

NOTE: This is a demo response using EPA emission factors. For production calculations, please set your OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.""",

    "fuel_emissions_natural_gas": """Natural gas combustion emissions (demo mode):

When burned, natural gas produces approximately:
- 53.06 kg CO2e per thousand cubic feet (Mcf)
- 0.0531 kg CO2e per cubic foot
- 5.31 kg CO2e per therm (100,000 BTU)

This includes:
- Direct CO2 emissions: ~53.0 kg/Mcf
- CH4 and N2O emissions: ~0.06 kg CO2e/Mcf

Natural gas is the cleanest fossil fuel, producing ~40% less CO2 than coal and ~25% less than oil per unit of energy.

NOTE: This is a demo response using EPA emission factors. For production calculations, please set your OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.""",

    # Generic climate queries
    "climate_what_is_carbon_footprint": """What is a carbon footprint? (demo mode)

A carbon footprint is the total amount of greenhouse gas emissions (primarily CO2) caused directly and indirectly by an individual, organization, event, or product. It's measured in tons of CO2 equivalent (CO2e) per year.

Components include:
1. Direct emissions: Transportation, heating, etc.
2. Indirect emissions: Electricity use, purchased goods/services
3. Lifecycle emissions: Manufacturing, disposal

Average carbon footprints:
- US person: ~16 tons CO2e/year
- EU person: ~7 tons CO2e/year
- Global average: ~4 tons CO2e/year
- Paris Agreement target: ~2-3 tons CO2e/year by 2050

NOTE: This is a demo response. For production AI assistance, please set your OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.""",

    "climate_renewable_energy": """Renewable energy overview (demo mode)

Renewable energy sources include:
1. Solar (photovoltaic and thermal)
2. Wind (onshore and offshore)
3. Hydroelectric
4. Geothermal
5. Biomass/Bioenergy

Benefits:
- Zero or low greenhouse gas emissions
- Abundant and sustainable
- Decreasing costs (solar/wind now cost-competitive with fossil fuels)
- Energy independence

Challenges:
- Intermittency (solar/wind depend on weather)
- Grid integration and storage needs
- Initial capital costs
- Land use considerations

Global renewable capacity is growing rapidly, with solar and wind leading the expansion.

NOTE: This is a demo response. For production AI assistance, please set your OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.""",

    "climate_carbon_offsets": """Carbon offsets explained (demo mode)

Carbon offsets are reductions in greenhouse gas emissions made to compensate for emissions elsewhere. One carbon offset represents one metric ton of CO2e reduced or removed from the atmosphere.

Types:
1. Avoidance: Renewable energy, energy efficiency
2. Removal: Reforestation, direct air capture, soil sequestration

Quality criteria:
- Additionality: Would not have happened without offset funding
- Permanence: Long-term storage (especially for removal projects)
- Verification: Third-party auditing
- No double-counting: Each offset claimed only once

Standards: Gold Standard, Verified Carbon Standard (Verra), American Carbon Registry

NOTE: This is a demo response. For production AI assistance, please set your OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.""",

    # Generic fallback
    "default": """I'm running in demo mode without an API key.

This is a simulated response from GreenLang's Intelligence Layer. To get real AI-powered responses for your climate and sustainability questions:

1. Add an API key to your environment:
   - OpenAI: export OPENAI_API_KEY=sk-...
   - Anthropic: export ANTHROPIC_API_KEY=sk-...

2. Restart your application

For common climate calculations, try asking about:
- Grid carbon intensity in your region
- Fuel combustion emissions
- Carbon footprints
- Renewable energy
- Carbon offsets

Demo mode provides pre-recorded responses for these topics.""",
}


# Tool call demonstrations for FakeProvider
DEMO_TOOL_CALLS: Dict[str, List[Dict[str, Any]]] = {
    # When asked to calculate fuel emissions
    "calculate_fuel_emissions": [
        {
            "id": "demo_call_001",
            "name": "calculate_fuel_emissions",
            "arguments": {
                "fuel_type": "diesel",
                "amount": 100.0,
                "unit": "gallons"
            }
        }
    ],

    "get_grid_intensity": [
        {
            "id": "demo_call_002",
            "name": "get_grid_intensity",
            "arguments": {
                "region": "CA",
                "datetime": "2025-10-01T12:00:00"
            }
        }
    ],

    "calculate_emissions": [
        {
            "id": "demo_call_003",
            "name": "calculate_emissions",
            "arguments": {
                "activity_type": "electricity",
                "amount": 1000.0,
                "unit": "kWh"
            }
        }
    ],
}


def get_demo_response(query: str, tools: List[Any] = None) -> Dict[str, Any]:
    """
    Get demo response for a query

    Returns appropriate pre-recorded response based on query content.
    If tools are provided, may return tool call instead of text.

    Args:
        query: User query text
        tools: Available tool definitions

    Returns:
        Dict with "type" ("text" or "tool_call"), "content", and optional "tool_calls"
    """
    query_lower = query.lower()

    # Check if tools are available and query suggests tool use
    if tools:
        # Fuel emissions
        if any(word in query_lower for word in ["diesel", "gasoline", "fuel", "gallon"]):
            if any(word in query_lower for word in ["calculate", "compute", "emissions", "co2"]):
                return {
                    "type": "tool_call",
                    "content": None,
                    "tool_calls": DEMO_TOOL_CALLS["calculate_fuel_emissions"]
                }

        # Grid intensity
        if any(word in query_lower for word in ["grid", "electricity", "intensity", "carbon intensity"]):
            if any(word in query_lower for word in ["get", "fetch", "what is", "calculate"]):
                return {
                    "type": "tool_call",
                    "content": None,
                    "tool_calls": DEMO_TOOL_CALLS["get_grid_intensity"]
                }

        # Generic emissions calculation
        if "calculate" in query_lower and "emission" in query_lower:
            return {
                "type": "tool_call",
                "content": None,
                "tool_calls": DEMO_TOOL_CALLS["calculate_emissions"]
            }

    # Text responses based on query content

    # Grid intensity
    if "grid" in query_lower or "electricity" in query_lower:
        if "california" in query_lower or "ca" in query_lower:
            response = DEMO_RESPONSES["grid_intensity_ca"]
        elif "texas" in query_lower or "ercot" in query_lower:
            response = DEMO_RESPONSES["grid_intensity_texas"]
        else:
            response = DEMO_RESPONSES["grid_intensity_generic"]
        return {"type": "text", "content": response}

    # Fuel emissions
    if "diesel" in query_lower:
        return {"type": "text", "content": DEMO_RESPONSES["fuel_emissions_diesel"]}

    if "gasoline" in query_lower or "petrol" in query_lower:
        return {"type": "text", "content": DEMO_RESPONSES["fuel_emissions_gasoline"]}

    if "natural gas" in query_lower or "lng" in query_lower:
        return {"type": "text", "content": DEMO_RESPONSES["fuel_emissions_natural_gas"]}

    # Climate topics
    if "carbon footprint" in query_lower or "footprint" in query_lower:
        return {"type": "text", "content": DEMO_RESPONSES["climate_what_is_carbon_footprint"]}

    if "renewable" in query_lower or "solar" in query_lower or "wind" in query_lower:
        return {"type": "text", "content": DEMO_RESPONSES["climate_renewable_energy"]}

    if "offset" in query_lower or "carbon credit" in query_lower:
        return {"type": "text", "content": DEMO_RESPONSES["climate_carbon_offsets"]}

    # Default fallback
    return {"type": "text", "content": DEMO_RESPONSES["default"]}


def get_demo_tool_result(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Get demo result for a tool call

    Returns simulated tool execution result.

    Args:
        tool_name: Name of tool being called
        arguments: Tool arguments

    Returns:
        Tool result as string (would be real calculation in production)
    """
    if tool_name == "calculate_fuel_emissions":
        fuel_type = arguments.get("fuel_type", "diesel")
        amount = arguments.get("amount", 100.0)

        # Use demo emission factors
        factors = {
            "diesel": 10.16,  # kg CO2e per gallon
            "gasoline": 8.89,
            "natural_gas": 0.0531,  # per cubic foot
        }

        factor = factors.get(fuel_type, 10.0)
        emissions = amount * factor

        return f"""{{
    "fuel_type": "{fuel_type}",
    "amount": {amount},
    "emissions_kg_co2e": {emissions:.2f},
    "emissions_tons_co2e": {emissions/1000:.3f},
    "source": "EPA emission factors (demo mode)",
    "note": "This is a demo calculation. Add API key for production use."
}}"""

    elif tool_name == "get_grid_intensity":
        region = arguments.get("region", "US")

        # Demo intensities
        intensities = {
            "CA": 220,
            "TX": 425,
            "US": 400,
        }

        intensity = intensities.get(region.upper(), 400)

        return f"""{{
    "region": "{region}",
    "intensity_g_co2e_per_kwh": {intensity},
    "timestamp": "2025-10-01T12:00:00Z",
    "source": "Simulated grid data (demo mode)",
    "note": "This is demo data. Add API key for real-time grid intensity."
}}"""

    elif tool_name == "calculate_emissions":
        activity = arguments.get("activity_type", "electricity")
        amount = arguments.get("amount", 1000.0)

        # Demo emission factors
        factors = {
            "electricity": 0.4,  # kg CO2e per kWh (US average)
            "natural_gas": 5.31,  # kg CO2e per therm
            "vehicle": 0.411,  # kg CO2e per mile (average car)
        }

        factor = factors.get(activity, 0.4)
        emissions = amount * factor

        return f"""{{
    "activity_type": "{activity}",
    "amount": {amount},
    "emissions_kg_co2e": {emissions:.2f},
    "source": "Demo emission factors",
    "note": "This is a demo calculation. Add API key for production use."
}}"""

    else:
        return f"""{{
    "error": "Unknown tool: {tool_name}",
    "note": "This is demo mode. Add API key for full tool support."
}}"""
