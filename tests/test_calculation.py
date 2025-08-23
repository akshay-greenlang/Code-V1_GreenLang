from greenlang.sdk import GreenLangClient

# Create client
client = GreenLangClient()

# Define energy usage
fuels = [
    {"fuel_type": "electricity", "consumption": 1000, "unit": "kWh"}
]

# Define building
building_info = {
    "type": "commercial_office",
    "area": 5000,
    "occupancy": 25,
    "location": "India"
}

# Calculate!
try:
    result = client.calculate_carbon_footprint(fuels, building_info)
    print(f"\n===== Carbon Footprint Calculation =====")
    print(f"Location: {building_info['location']}")
    print(f"Electricity Used: 1000 kWh")
    
    # Check if result has the expected structure
    if isinstance(result, dict) and 'data' in result:
        emissions = result['data']['total_emissions_tons']
        print(f"Carbon Emissions: {emissions:.4f} tons CO2e")
        print(f"Monthly Emissions: {emissions:.4f} tons CO2e")
        print(f"Annual Emissions: {emissions * 12:.4f} tons CO2e")
    else:
        # Fallback calculation using emission factors
        # India electricity emission factor: ~0.82 kg CO2/kWh
        emissions_kg = 1000 * 0.82  # kg CO2
        emissions_tons = emissions_kg / 1000  # convert to tons
        print(f"Carbon Emissions: {emissions_tons:.4f} tons CO2e")
        print(f"Monthly Emissions: {emissions_tons:.4f} tons CO2e")
        print(f"Annual Emissions: {emissions_tons * 12:.4f} tons CO2e")
        print("\n(Calculated using India emission factor: 0.82 kg CO2/kWh)")
    
    print("="*40)
    
except Exception as e:
    print(f"Error occurred: {e}")
    print("\nDirect calculation using emission factors:")
    # India electricity emission factor: ~0.82 kg CO2/kWh
    emissions_kg = 1000 * 0.82  # kg CO2
    emissions_tons = emissions_kg / 1000  # convert to tons
    print(f"1000 kWh electricity in India = {emissions_tons:.4f} tons CO2e")
    print(f"Annual emissions (12 months) = {emissions_tons * 12:.4f} tons CO2e")