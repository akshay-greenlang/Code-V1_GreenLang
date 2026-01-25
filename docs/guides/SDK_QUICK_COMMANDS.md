# GreenLang SDK - Quick CMD Commands

## One-Line Commands You Can Run Directly in CMD

### 1. Calculate Emissions (Simple)
```cmd
python -c "from greenlang.sdk import GreenLangClient; c = GreenLangClient(); print(f'1000 kWh = {c.calculate_emissions(\"electricity\", 1000, \"kWh\")[\"data\"][\"co2e_emissions_kg\"]:.0f} kg CO2e')"
```

### 2. List All Agents
```cmd
python -c "from greenlang.sdk import GreenLangClient; print('Agents:', GreenLangClient().list_agents())"
```

### 3. Get Countries
```cmd
python -c "from greenlang.sdk import GreenLangClient; print('Countries:', GreenLangClient().get_supported_countries())"
```

### 4. Compare US vs India
```cmd
python -c "from greenlang.sdk import GreenLangClient as G; us=G('US').calculate_emissions('electricity',1000,'kWh')['data']['co2e_emissions_kg']; ind=G('IN').calculate_emissions('electricity',1000,'kWh')['data']['co2e_emissions_kg']; print(f'US: {us:.0f} kg, India: {ind:.0f} kg, Difference: {ind/us:.1f}x')"
```

### 5. Boiler Calculation
```cmd
python -c "from greenlang.sdk import GreenLangClient; r = GreenLangClient().calculate_boiler_emissions('natural_gas', 1000, 'kWh', 0.85); print(f'Boiler emissions: {r[\"data\"][\"co2e_emissions_kg\"]:.2f} kg CO2e')"
```

### 6. Quick Building Analysis
```cmd
python -c "from greenlang.sdk import GreenLangClient; c=GreenLangClient(); b={'metadata':{'building_type':'office','area':10000,'location':{'country':'US'}},'energy_consumption':{'electricity':{'value':100000,'unit':'kWh'}}}; r=c.analyze_building(b); print(f'Emissions: {r[\"data\"][\"emissions\"][\"total_co2e_tons\"]:.2f} tons/year')"
```

### 7. Get Grid Factor
```cmd
python -c "from greenlang.sdk import GreenLangClient; f=GreenLangClient().get_emission_factor('electricity','US','kWh'); print(f'US Grid: {f[\"data\"][\"emission_factor\"]} kgCO2e/kWh')"
```

### 8. Benchmark Test
```cmd
python -c "from greenlang.sdk import GreenLangClient; r=GreenLangClient().benchmark_emissions(50000,10000,'office',12); print(f'Rating: {r[\"data\"][\"rating\"]}')"
```

### 9. Multi-Country Quick Compare
```cmd
python -c "from greenlang.sdk import GreenLangClient as G; [print(f'{c}: {G(c).calculate_emissions(\"electricity\",1000,\"kWh\")[\"data\"][\"co2e_emissions_kg\"]:.0f} kg') for c in ['US','IN','EU','CN','JP','BR']]"
```

### 10. Intensity Calculation
```cmd
python -c "from greenlang.sdk import GreenLangClient; r=GreenLangClient().calculate_intensity(100000,50000,'sqft',200); print(f'Intensity: {r[\"data\"][\"intensities\"][\"per_sqft_year\"]:.2f} kgCO2e/sqft/year')"
```

## How to Use These Commands

1. **Open CMD**:
   ```
   Windows Key + R → type "cmd" → Enter
   ```

2. **Navigate to GreenLang**:
   ```cmd
   cd "C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang"
   ```

3. **Copy and paste any command above**

## Create Your Own Commands

### Template:
```cmd
python -c "from greenlang.sdk import GreenLangClient; client = GreenLangClient(); <YOUR CODE HERE>"
```

### Examples:

**Custom Emission Calculation:**
```cmd
python -c "from greenlang.sdk import GreenLangClient; c=GreenLangClient(); e=c.calculate_emissions('natural_gas',500,'therms'); print(f'Natural gas: {e[\"data\"][\"co2e_emissions_kg\"]:.0f} kg')"
```

**Custom Country:**
```cmd
python -c "from greenlang.sdk import GreenLangClient; c=GreenLangClient('EU'); e=c.calculate_emissions('electricity',1000,'kWh'); print(f'EU electricity: {e[\"data\"][\"co2e_emissions_kg\"]:.0f} kg')"
```

## Batch File for Common Tasks

Create `sdk_quick.bat`:
```batch
@echo off
echo Running Quick SDK Commands...
echo.
echo 1. US Electricity (1000 kWh):
python -c "from greenlang.sdk import GreenLangClient; c=GreenLangClient('US'); print(f'  {c.calculate_emissions(\"electricity\",1000,\"kWh\")[\"data\"][\"co2e_emissions_kg\"]:.0f} kg CO2e')"
echo.
echo 2. India Electricity (1000 kWh):
python -c "from greenlang.sdk import GreenLangClient; c=GreenLangClient('IN'); print(f'  {c.calculate_emissions(\"electricity\",1000,\"kWh\")[\"data\"][\"co2e_emissions_kg\"]:.0f} kg CO2e')"
echo.
echo 3. Available Agents:
python -c "from greenlang.sdk import GreenLangClient; print(f'  {len(GreenLangClient().list_agents())} agents')"
echo.
pause
```

---

**All these commands can be run directly in CMD without creating any Python files!**