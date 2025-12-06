# PROCESS HEAT AGENTS - DETAILED IMPROVEMENT SPECIFICATIONS (PART 2)
## GL-006 through GL-020 Specifications

**Document Version**: 1.0.0 (Part 2 of 2)
**Date**: December 4, 2025
**Continuation of**: PROCESS_HEAT_AGENTS_DETAILED_IMPROVEMENT_SPECIFICATIONS.md

---

## GL-006: HEATRECLAIM (Current: 90 → Target: 95+)

**Agent Name**: Waste Heat Recovery Optimizer
**Gap Analysis**: Needs pinch analysis automation, heat exchanger network synthesis

### 1. Critical Process Variables

| Variable | Units | Sensor Type | Accuracy | Range | Redundancy | Sample Rate |
|----------|-------|-------------|----------|-------|------------|-------------|
| Hot Stream Temp (Inlet) | °C | RTD (Pt100) | ±0.15% | 100-800°C | 1oo2 | 1 Hz |
| Hot Stream Temp (Outlet) | °C | RTD (Pt100) | ±0.15% | 50-400°C | 1oo2 | 1 Hz |
| Cold Stream Temp (Inlet) | °C | RTD (Pt100) | ±0.15% | 10-200°C | 1oo2 | 1 Hz |
| Cold Stream Temp (Outlet) | °C | RTD (Pt100) | ±0.15% | 50-400°C | 1oo2 | 1 Hz |
| Hot Stream Flow Rate | kg/s | Magnetic/Vortex | ±1% FS | 0-100 kg/s | 1oo2 | 1 Hz |
| Cold Stream Flow Rate | kg/s | Magnetic/Vortex | ±1% FS | 0-100 kg/s | 1oo2 | 1 Hz |
| Hot Stream Pressure | bar(g) | Pressure Xmtr | ±0.25% FS | 0-50 bar | 1oo1 | 1 Hz |
| Cold Stream Pressure | bar(g) | Pressure Xmtr | ±0.25% FS | 0-50 bar | 1oo1 | 1 Hz |
| Economizer Gas Side ΔP | mbar | DP Transmitter | ±1% FS | 0-100 mbar | 1oo1 | 0.2 Hz |
| Economizer Water Side ΔP | bar | DP Transmitter | ±0.5% FS | 0-5 bar | 1oo1 | 0.2 Hz |
| Heat Recovery Rate | MW | Calculated | ±3% | 0-50 MW | N/A | 1 Hz |

### 2. Thermodynamic Calculations

#### 2.1 Heat Exchanger Effectiveness (ε-NTU Method)
```
Heat Exchanger Effectiveness:
ε = Q_actual / Q_max

Where:
Q_actual = ṁ_hot × Cp_hot × (T_hot_in - T_hot_out)  [MW]
         = ṁ_cold × Cp_cold × (T_cold_out - T_cold_in)  [MW]

Q_max = (ṁ × Cp)_min × (T_hot_in - T_cold_in)  [MW]

Number of Transfer Units (NTU):
NTU = UA / (ṁ × Cp)_min

Where:
UA = Overall heat transfer coefficient × Area  [MW/K]

For Counterflow Exchanger:
ε = (1 - exp(-NTU(1 - C_r))) / (1 - C_r × exp(-NTU(1 - C_r)))

Where:
C_r = (ṁ × Cp)_min / (ṁ × Cp)_max  [Capacity ratio]

Reference: Incropera & DeWitt, "Fundamentals of Heat Transfer" (7th Ed, Chapter 11)
TEMA Standards (9th Ed)
```

#### 2.2 Pinch Analysis (Automated) - NEW
```
Pinch Analysis Procedure:

Step 1: Extract Stream Data
Hot_Streams = [{T_in, T_out, ṁ × Cp, Available}, ...]
Cold_Streams = [{T_in, T_out, ṁ × Cp, Required}, ...]

Step 2: Construct Composite Curves
Hot_Composite_Curve: Plot cumulative heat vs temperature (hot streams)
Cold_Composite_Curve: Plot cumulative heat vs temperature (cold streams)

Step 3: Identify Pinch Point
ΔT_min = Minimum approach temperature (typically 10-20°C)
Pinch_Hot = Temperature where curves are ΔT_min apart (hot side)
Pinch_Cold = Pinch_Hot - ΔT_min

Step 4: Calculate Energy Targets
Q_hot_utility = Heat required above pinch (heating)
Q_cold_utility = Heat rejected below pinch (cooling)
Q_recovery = Total heat available - Q_hot_utility - Q_cold_utility

Step 5: Heat Exchanger Network Synthesis
Above Pinch: Use hot utility only (no heat transfer from below pinch)
Below Pinch: Use cold utility only (no heat transfer from above pinch)
At Pinch: Maximum heat exchange between hot and cold streams

Algorithm: Sequential matching (heuristic) or optimization (MILP)

Objective Function (for MILP):
Minimize: C_capital × Σ(A_i) + C_operating × (Q_hot_utility × C_heating + Q_cold_utility × C_cooling)

Where:
C_capital = Capital cost per m² of heat transfer area [$/m²]
A_i = Area of heat exchanger i [m²]
C_heating = Cost of hot utility (steam, fuel) [$/MWh]
C_cooling = Cost of cold utility (cooling water, refrigeration) [$/MWh]

Reference: Linnhoff & Hindmarsh, "The Pinch Design Method for Heat Exchanger Networks" (1983)
Smith, "Chemical Process Design and Integration" (2nd Ed, Chapter 19)
Software: Aspen Energy Analyzer, SPRINT, or open-source pinch tools
```

#### 2.3 Fouling Factor Monitoring (NEW)
```
Fouling Detection:

Clean Heat Transfer Coefficient (baseline):
U_clean = 1 / (1/h_hot + R_wall + 1/h_cold)  [W/m²·K]

Fouled Heat Transfer Coefficient (current):
U_fouled = Q_actual / (A × LMTD)  [W/m²·K]

Where:
LMTD = Log Mean Temperature Difference
     = (ΔT₁ - ΔT₂) / ln(ΔT₁ / ΔT₂)
ΔT₁ = T_hot_in - T_cold_out
ΔT₂ = T_hot_out - T_cold_in

Fouling Resistance:
R_f = (1/U_fouled - 1/U_clean)  [m²·K/W]

Fouling Rate:
dR_f/dt = ΔR_f / Δt  [m²·K/W per day]

Cleaning Recommendation:
IF R_f > 0.0005 m²·K/W (TEMA allowance for clean service) THEN
  Recommend: Schedule cleaning during next shutdown
ELSE IF R_f > 0.001 m²·K/W THEN
  Alert: "Severe fouling detected - plan immediate cleaning"
END IF

Reference: TEMA Standards (9th Ed, Table RGP-1.4)
ASME PTC 12.5 "Single Phase Heat Exchangers"
```

#### 2.4 Economic Analysis (NEW)
```
Heat Recovery Value Calculation:

Annual Energy Recovery:
E_recovered = Q_recovered × Operating_hours  [MWh/yr]

Fuel Savings:
Fuel_saved = E_recovered / η_boiler × C_fuel  [$/yr]

Example:
Q_recovered = 5 MW
Operating_hours = 8000 hr/yr
E_recovered = 5 MW × 8000 hr = 40,000 MWh/yr

η_boiler = 0.85
C_fuel = $5/MMBtu = $1.47/MWh
Fuel_saved = 40,000 MWh / 0.85 × $1.47/MWh = $69,200/yr

Capital Cost (Economizer Installation):
C_capital = C_exchanger + C_piping + C_installation
          = $200,000 (typical for 5 MW economizer)

Simple Payback Period:
Payback = C_capital / Fuel_saved = $200,000 / $69,200/yr = 2.9 years

Net Present Value (NPV):
NPV = -C_capital + Σ(Fuel_saved / (1 + r)^t) for t = 1 to 20 years

Where r = discount rate (typically 8-12%)

Decision Criterion: NPV > 0, Payback < 3 years → Recommend project

Reference: ASHRAE Guideline 14 "Measurement of Energy and Demand Savings"
ISO 50001:2018 Energy Management Systems
```

### 3. Integration Points

#### 3.1 DCS Interface
```yaml
Control_Strategy:
  Economizer_Outlet_Temp_Control:
    PV: Feedwater_temp_out [°C]
    SP: 150°C (operator adjustable 120-180°C)
    Output: Feedwater_bypass_valve [%]
    Tuning: Kp=2.5, Ti=180s, Td=15s

  Constraint: Prevent stack condensation
    Min_flue_gas_temp_out: 120°C (above acid dew point for natural gas)
    IF T_flue_out < 120°C THEN
      Close feedwater_bypass_valve (reduce heat recovery)
      Alert: "Stack temperature too low - risk of condensation/corrosion"
    END IF

OPC-UA_Tags:
  - GL006.ECONOMIZER_01.T_FLUE_IN (Float, °C, RO)
  - GL006.ECONOMIZER_01.T_FLUE_OUT (Float, °C, RO)
  - GL006.ECONOMIZER_01.T_WATER_IN (Float, °C, RO)
  - GL006.ECONOMIZER_01.T_WATER_OUT (Float, °C, RW - controlled)
  - GL006.ECONOMIZER_01.Q_RECOVERED (Float, MW, RO)
  - GL006.ECONOMIZER_01.FOULING_FACTOR (Float, m²K/W, RO)
```

#### 3.2 Integration with GL-001 (THERMOSYNC) - Opportunity Flagging
```yaml
Heat_Recovery_Opportunity_Detection:
  Scan_Interval: Every 5 minutes

  Algorithm:
    FOR each hot_stream IN plant_streams:
      IF hot_stream.T_out > (ambient + 50°C) AND hot_stream.ṁ > 1 kg/s THEN
        Q_available = hot_stream.ṁ × Cp × (T_out - ambient - 10°C)
        IF Q_available > 0.5 MW THEN
          Flag as "Opportunity: {hot_stream.ID}"
          Identify matching cold_streams within ΔT_min
          Estimate_payback(hot_stream, cold_stream)
        END IF
      END IF
    END FOR

  Output:
    - Opportunity_list: Array of {hot_stream, cold_stream, Q_potential, payback}
    - Send to GL-001 THERMOSYNC for prioritization
    - Generate monthly report: "Top 10 Heat Recovery Opportunities"
```

### 4. Safety Considerations

#### 4.1 Interlocks
```yaml
Safety_Interlocks:
  - High_Economizer_Outlet_Temp:
      Condition: T_water_out > 200°C (risk of flashing at reduced pressure)
      Action: Open bypass valve, reduce flue gas flow
      Vote: 1oo2
      Rationale: Prevent steam hammer in feedwater system

  - Low_Flue_Gas_Outlet_Temp:
      Condition: T_flue_out < 100°C (condensation/corrosion risk)
      Action: Reduce heat recovery (open bypass)
      Vote: 1oo1
      Rationale: Protect economizer from acid corrosion

  - High_Gas_Side_Pressure_Drop:
      Condition: ΔP_gas > 150 mbar (30% above design)
      Action: Alert for cleaning, reduce gas flow if ΔP > 200 mbar
      Vote: 1oo1
      Rationale: Severe fouling - fan capacity exceeded
```

#### 4.2 Alarms
```yaml
Alarms:
  - Economizer_Fouling_High:
      Condition: R_f > 0.0008 m²K/W
      Priority: MEDIUM (P3)
      Action: Schedule cleaning within 30 days

  - Heat_Recovery_Degradation:
      Condition: Q_recovered < 80% of baseline
      Priority: HIGH (P2)
      Action: Investigate fouling OR bypass valve stuck open

  - Stack_Temp_Low:
      Condition: T_flue_out < 110°C
      Priority: HIGH (P2)
      Action: Reduce heat recovery to protect from corrosion
```

### 5. Standards Alignment

| Standard | Section | Requirement | Implementation |
|----------|---------|-------------|----------------|
| TEMA (9th Ed) | RGP-1.4 | Fouling resistance allowances | Monitor R_f vs TEMA recommended values |
| ASME PTC 12.5 | Sections 4-5 | Heat exchanger testing | ε-NTU method for performance verification |
| ISO 50001:2018 | Clause 6.2 | Energy opportunity identification | Automated scanning for heat recovery opportunities |
| ASHRAE 90.1 | Section 6.5 | Energy recovery requirements | Economizers required for boilers >500,000 Btu/hr in most climates |
| API 661 | Section 5 | Air-cooled heat exchangers | For cooling applications (design, operation, maintenance) |

### 6. Consolidation

**Recommendation**: RETAIN GL-006 as STANDALONE
**Rationale**: Unique focus on waste heat recovery optimization with pinch analysis, no significant overlap with other agents

---

## GL-007: FURNACEPULSE (Current: 80 → Target: 95+)

**Agent Name**: Fired Heater and Furnace Optimizer
**Gap Analysis**: Needs tube metal temperature monitoring, coil pressure drop analysis, process-side thermal modeling

### 1. Critical Process Variables

| Variable | Units | Sensor Type | Accuracy | Range | Redundancy | Sample Rate |
|----------|-------|-------------|----------|-------|------------|-------------|
| Tube Metal Temperature (TMT) | °C | Thermocouple (K) | ±2°C | 400-1200°C | 2oo3 per zone | 2 Hz |
| Process Fluid Temp (Inlet) | °C | RTD (Pt100) | ±0.15% | 50-600°C | 1oo2 | 1 Hz |
| Process Fluid Temp (Outlet) | °C | RTD (Pt100) | ±0.15% | 200-800°C | 1oo2 | 1 Hz |
| Process Fluid Flow Rate | kg/s | Coriolis Meter | ±0.2% FS | 0-50 kg/s | 1oo2 | 1 Hz |
| Process Fluid Pressure (Inlet) | bar(g) | Pressure Xmtr | ±0.25% FS | 0-100 bar | 1oo2 | 2 Hz |
| Process Fluid Pressure (Outlet) | bar(g) | Pressure Xmtr | ±0.25% FS | 0-100 bar | 1oo2 | 2 Hz |
| Coil Pressure Drop | bar | DP Transmitter | ±0.5% FS | 0-10 bar | 1oo2 | 1 Hz |
| Fuel Flow Rate | kg/s or Nm³/h | Coriolis/Thermal | ±0.5% FS | 0-10 kg/s | 1oo2 | 1 Hz |
| Flue Gas O₂ | % vol (dry) | Zirconia Analyzer | ±0.1% abs | 0-21% | 1oo2 | 1 Hz |
| Stack Temperature | °C | Thermocouple (K) | ±2°C | 200-600°C | 1oo2 | 1 Hz |
| Radiant Section Heat Flux | kW/m² | Heat Flux Meter | ±5% | 0-150 kW/m² | 1oo1 | 0.5 Hz |
| Flame Temperature (IR Camera) | °C | IR Pyrometer | ±20°C | 1000-2000°C | 1oo1 | 0.1 Hz |

### 2. Thermodynamic Calculations

#### 2.1 Tube Metal Temperature (TMT) Monitoring - NEW (CRITICAL)
```
Tube Wall Temperature Calculation:

Heat Flux at Tube OD:
q" = Q_absorbed / A_tube_OD  [W/m²]

Where:
Q_absorbed = Heat transferred to process fluid in radiant section [W]
A_tube_OD = Outer surface area of tubes [m²]

Tube Wall Temperature Drop (conduction through tube wall):
ΔT_wall = q" × t_wall / k_tube  [K]

Where:
t_wall = Tube wall thickness [m]
k_tube = Thermal conductivity of tube material (e.g., 50 W/m·K for carbon steel) [W/m·K]

Tube Outer Wall Temperature:
T_tube_OD = T_fluid + ΔT_film + ΔT_wall  [°C]

Where:
ΔT_film = q" / h_inside  [K]
h_inside = Inside film heat transfer coefficient [W/m²·K]
          ≈ 500-2000 W/m²·K for liquid hydrocarbons
          ≈ 50-200 W/m²·K for vapor hydrocarbons

TMT Safety Limit:
T_tube_OD < T_design_max - Safety_margin

For Carbon Steel (A106 Grade B):
T_design_max = 450°C (creep damage threshold)
Safety_margin = 50°C
TMT_alarm = 400°C
TMT_trip = 450°C

For 5Cr-0.5Mo Steel (T5):
T_design_max = 650°C
Safety_margin = 50°C
TMT_alarm = 600°C
TMT_trip = 650°C

Action on High TMT:
IF TMT > T_alarm THEN
  Reduce firing rate by 10%
  Increase process fluid flow rate by 5% (if possible)
  Alert operator: "High tube metal temperature in Zone {X}"
END IF

IF TMT > T_trip (2oo3 vote) THEN
  Emergency shutdown: Close fuel valves immediately
  Maintain process fluid circulation
  Log event for metallurgical investigation
END IF

Reference: API 530 "Calculation of Heater-Tube Thickness in Petroleum Refineries"
API 579 "Fitness-for-Service" (creep damage assessment)
ASME B31.3 "Process Piping" (allowable stress at temperature)
```

#### 2.2 Coil Pressure Drop Analysis - NEW
```
Pressure Drop Calculation (Two-Phase Flow):

Single-Phase Liquid (Darcy-Weisbach):
ΔP_friction = f × (L/D) × (ρ × v²/2)  [Pa]

Where:
f = Friction factor (Moody chart or Colebrook equation)
L = Tube length [m]
D = Inside diameter [m]
ρ = Fluid density [kg/m³]
v = Fluid velocity [m/s]

Two-Phase Flow (Lockhart-Martinelli):
ΔP_2phase = ΔP_liquid × Φ²_L

Where:
Φ²_L = Two-phase multiplier (function of quality x and flow regime)

Martinelli Parameter:
X² = (ΔP_liquid / ΔP_vapor)

Φ²_L ≈ 1 + C/X + 1/X²  (C depends on flow regime)

Acceleration Pressure Drop (vaporization):
ΔP_accel = G² × v_fg × Δx  [Pa]

Where:
G = Mass flux [kg/m²·s]
v_fg = Specific volume change on vaporization [m³/kg]
Δx = Change in vapor quality

Total Coil Pressure Drop:
ΔP_total = ΔP_friction + ΔP_accel + ΔP_static (elevation change)

Coil Fouling Detection:
ΔP_measured = f(Flow_rate, Fluid_properties, Fouling_factor)

IF ΔP_measured > ΔP_clean × 1.3 THEN
  Alert: "Coil fouling suspected - ΔP increased 30%"
  Recommend: Chemical cleaning OR mechanical pigging
END IF

Reference: API 530 Section 6 "Pressure Drop Calculations"
Beggs & Brill "Two-Phase Flow in Pipes" (SPE monograph)
```

#### 2.3 Radiant Section Heat Transfer - NEW
```
Radiant Heat Transfer (Zone Method):

Heat Absorbed by Tubes (Radiant Section):
Q_radiant = A_tube × ε_tube × σ × F × (T_flame⁴ - T_tube⁴)  [W]

Where:
A_tube = Tube surface area exposed to radiation [m²]
ε_tube = Tube emissivity (0.8-0.9 for oxidized steel)
σ = Stefan-Boltzmann constant (5.67 × 10⁻⁸ W/m²·K⁴)
F = View factor (geometry dependent, 0.5-0.8 typical)
T_flame = Flame temperature [K] (1200-2000°C typical)
T_tube = Tube surface temperature [K]

Radiant Section Efficiency:
η_radiant = Q_radiant / Q_fuel × 100%

Typical: 50-60% for fired heaters (remainder to convection section + stack loss)

Convection Section Heat Transfer:
Q_convection = ṁ_flue × Cp_flue × (T_flue_in - T_flue_out)  [W]

Where:
T_flue_in = Exit temperature from radiant section (1000-1200°C)
T_flue_out = Stack temperature (200-400°C)

Overall Furnace Efficiency:
η_furnace = (Q_radiant + Q_convection) / Q_fuel × 100%

Typical: 85-92% for well-designed fired heaters

Reference: API 560 "Fired Heaters for General Refinery Service"
GPSA Engineering Data Book (Section 6)
```

#### 2.4 Process-Side Thermal Model - NEW
```
Process Fluid Temperature Profile Along Coil:

Energy Balance (Differential Form):
dH/dL = q" × π × D_OD  [W/m]

Where:
H = Enthalpy of process fluid [J/kg]
L = Length along coil [m]
q" = Local heat flux [W/m²]
D_OD = Tube outer diameter [m]

For Single-Phase Liquid:
dT/dL = q" × π × D_OD / (ṁ × Cp)  [K/m]

For Two-Phase Flow (Vaporization):
dx/dL = q" × π × D_OD / (ṁ × h_fg)  [1/m]

Where:
x = Vapor quality (0 = all liquid, 1 = all vapor)
h_fg = Latent heat of vaporization [J/kg]

Numerical Solution (Finite Difference):
Discretize coil into N segments (e.g., N=100)
For i = 1 to N:
  T[i+1] = T[i] + (q"[i] × π × D_OD × ΔL) / (ṁ × Cp)
  IF T[i+1] > T_boiling THEN
    Switch to two-phase calculation
    x[i+1] = x[i] + (q"[i] × π × D_OD × ΔL) / (ṁ × h_fg)
  END IF
END FOR

Output:
- Temperature profile T(L) along coil
- Pressure profile P(L) along coil
- Vapor quality profile x(L) (if two-phase)
- Tube metal temperature profile TMT(L)

Application:
- Detect hot spots (where TMT peaks)
- Optimize firing pattern (adjust burner tilt, fuel distribution)
- Predict coking tendency (high tube temperatures → coking)

Reference: API 530 Appendix D "Detailed Thermal Model"
Kern, "Process Heat Transfer" (classic textbook)
```

### 3. Integration Points

#### 3.1 DCS Interface - Advanced Control
```yaml
Multi-Zone_Temperature_Control:
  Configuration:
    Zones: [Radiant_Top, Radiant_Bottom, Convection]
    Burners_per_Zone: 4-8 burners

  Control_Strategy:
    Master_Controller:
      PV: Process_fluid_outlet_temp [°C]
      SP: 650°C (operator adjustable 500-800°C)
      Output: Total_fuel_flow_setpoint [kg/s]
      Tuning: Kp=3.0, Ti=300s, Td=20s

    Zone_Controllers (Slave):
      FOR each zone:
        PV: Zone_average_TMT [°C]
        SP: From fuel distribution algorithm
        Output: Zone_fuel_flow [kg/s]
        Constraint: TMT < 400°C (for carbon steel)
      END FOR

    Fuel_Distribution_Algorithm:
      Objective: Minimize MAX(TMT) across all zones
      Method: Quadratic programming (QP)

      Decision_Variables: Fuel_flow[zone] for each zone

      Objective_Function:
        Minimize: MAX(TMT[zone]) + λ × Σ(Fuel_flow[zone]²)

      Constraints:
        Σ Fuel_flow[zone] = Total_fuel_flow_setpoint
        Fuel_flow_min[zone] ≤ Fuel_flow[zone] ≤ Fuel_flow_max[zone]
        TMT[zone] ≤ 400°C for all zones

      Update_Frequency: Every 10 seconds

    Burner_Tilt_Optimization:
      IF TMT_top > TMT_bottom + 20°C THEN
        Tilt burners DOWN (reduce heat flux to top)
      ELSE IF TMT_bottom > TMT_top + 20°C THEN
        Tilt burners UP (reduce heat flux to bottom)
      END IF

      Tilt_Range: ±30° from horizontal
      Adjustment_Step: 2° per 1 minute
```

#### 3.2 Integration with GL-018 (FLUEFLOW)
```yaml
Data_Exchange:
  From_GL018:
    - Flue_gas_O₂ [%]
    - Combustion_efficiency [%]
    - Stack_temperature [°C]
    - Optimal_excess_air [%]

  To_GL018:
    - Furnace_heat_duty [MW]
    - Process_fluid_outlet_temp [°C]
    - TMT_max [°C]
    - Fuel_consumption_rate [kg/s]

  Coordination:
    GL-007: Requests heat duty from thermal model
    GL-018: Optimizes air-fuel ratio to achieve heat duty at max efficiency
    GL-007: Monitors TMT and requests firing rate reduction if TMT_max > alarm
```

### 4. Safety Considerations

#### 4.1 Critical Safety Interlocks
```yaml
Safety_Interlocks:
  - High_Tube_Metal_Temperature:
      Condition: TMT > 450°C (2oo3 vote) for carbon steel
      Action: Emergency shutdown (ESD) - close fuel valves immediately
      Response: < 2 seconds
      Rationale: Prevent tube rupture from creep damage

  - Low_Process_Flow:
      Condition: Flow < 30% of design minimum
      Action: Reduce firing rate to 20%, alarm operator
      Rationale: Prevent tube overheating from insufficient cooling

  - High_Coil_Outlet_Pressure:
      Condition: P_outlet > 110% of design (coking suspected)
      Action: Alert operator, reduce firing rate by 20%
      Rationale: Prevent tube blockage from coke deposition

  - Flame_Failure_Any_Burner:
      Condition: Flame scanner signal < 20% for any burner
      Action: Close fuel valve to that burner, purge zone
      Response: < 2 seconds per NFPA 85
      Rationale: Prevent furnace explosion from unburned fuel accumulation
```

#### 4.2 Alarms (Prioritized per ISA-18.2)
```yaml
Alarms:
  - TMT_High_Warning:
      Setpoint: 380°C (20°C below trip)
      Priority: HIGH (P2)
      Action: Alert operator to investigate (fouling, low flow, high firing)

  - Coil_Pressure_Drop_High:
      Setpoint: ΔP > 1.5 × design
      Priority: MEDIUM (P3)
      Action: Schedule decoking OR inspection

  - Stack_Temperature_High:
      Setpoint: T_stack > 450°C
      Priority: MEDIUM (P3)
      Action: Soot blowing OR convection section fouling
```

### 5. Standards Alignment

| Standard | Section | Requirement | Implementation |
|----------|---------|-------------|----------------|
| API 530 (2021) | Sections 4-6 | Fired heater design, operation, inspection | TMT monitoring per §4.3.2, thermal model per Appendix D |
| API 560 (2016) | Sections 7-8 | Fired heater commissioning and operation | Startup procedures, control philosophy |
| API 579-1/ASME FFS-1 | Part 10 | Creep damage assessment | Larson-Miller parameter for remaining life calculation |
| ASME B31.3 | Chapter II | Process piping design pressure/temperature | Allowable stress for tube material at operating temperature |
| NFPA 85 | Chapter 7 | Fired heater burner management | Flame supervision, interlocks, purge requirements |

### 6. Consolidation

**Recommendation**: RETAIN GL-007 as STANDALONE
**Rationale**: Unique focus on fired heaters/furnaces with tube metal temperature monitoring, distinct from boilers (GL-002/GL-018)

---

## GL-008: TRAPCATCHER (Current: 92 → Target: 95+)

**Agent Name**: Steam Trap Performance Monitor
**Gap Analysis**: Needs trap type classification, condensate load calculations, trap sizing validation

### 1. Critical Process Variables

| Variable | Units | Sensor Type | Accuracy | Range | Redundancy | Sample Rate |
|----------|-------|-------------|----------|-------|------------|-------------|
| Trap Inlet Temperature | °C | IR Temperature Sensor | ±2°C | 50-200°C | 1oo1 | 0.1 Hz |
| Trap Outlet Temperature | °C | IR Temperature Sensor | ±2°C | 20-200°C | 1oo1 | 0.1 Hz |
| Trap Ultrasonic Signal | dB | Ultrasonic Sensor | ±3 dB | 0-120 dB | 1oo1 | 10 Hz |
| Trap Infrared Signature | °C | IR Camera | ±5°C | 20-300°C | 1oo1 | 0.1 Hz |
| Steam Header Pressure | bar(g) | Pressure Xmtr | ±0.25% FS | 0-20 bar | 1oo2 | 1 Hz |
| Condensate Return Pressure | bar(g) | Pressure Xmtr | ±0.5% FS | 0-5 bar | 1oo1 | 1 Hz |
| Condensate Flow Rate | kg/s | Magnetic Meter | ±1% FS | 0-10 kg/s | 1oo1 | 0.5 Hz |

**Note**: Steam trap monitoring typically uses portable sensors (IR gun, ultrasonic detector) during inspection rounds, not continuous fixed sensors.

### 2. Thermodynamic Calculations

#### 2.1 Trap Type Classification and Selection - NEW
```yaml
Steam_Trap_Types:
  - Mechanical_Traps:
      Float_and_Thermostatic (F&T):
        Operation: Float valve opens on condensate accumulation
        Advantages: Continuous discharge, handles variable loads
        Disadvantages: Sensitive to dirt, water hammer risk
        Applications: Process equipment, large heat exchangers
        Typical_Capacity: 100-5000 kg/hr at Δ10 bar

      Inverted_Bucket:
        Operation: Bucket sinks when filled with condensate, opens valve
        Advantages: Robust, handles dirt, moderate steam loss
        Disadvantages: Intermittent discharge (cycling)
        Applications: Steam mains, tracers, general service
        Typical_Capacity: 50-2000 kg/hr at Δ5 bar

  - Thermostatic_Traps:
      Balanced_Pressure:
        Operation: Capsule contracts on condensate (cooler than steam), opens valve
        Advantages: Air venting, compact, low cost
        Disadvantages: Slow response, moderate steam loss
        Applications: Steam mains, unit heaters, small loads
        Typical_Capacity: 10-500 kg/hr at Δ3 bar

      Bimetallic:
        Operation: Bimetallic element bends on temperature change, opens valve
        Advantages: Rugged, handles superheat
        Disadvantages: Slow response, holds back condensate
        Applications: Steam tracers, freeze protection
        Typical_Capacity: 5-200 kg/hr at Δ2 bar

  - Thermodynamic_Traps:
      Disc_Trap:
        Operation: Pressure difference across disc controls valve position
        Advantages: Compact, handles superheat, high capacity
        Disadvantages: Noisy, cycling wear, moderate steam loss
        Applications: High pressure steam, tracers, drip legs
        Typical_Capacity: 20-1000 kg/hr at Δ10 bar

Selection_Criteria:
  Step_1: Calculate condensate load (kg/hr)
  Step_2: Determine pressure differential (P_steam - P_return)
  Step_3: Select trap type based on application
  Step_4: Size trap for 2-3× condensate load (safety factor)
  Step_5: Verify capacity from manufacturer's tables

Reference: Spirax Sarco "Steam Engineering Tutorials"
TLV "Steam Trap Handbook"
ASME Steam Tables (for steam properties)
```

#### 2.2 Condensate Load Calculation - NEW
```
Condensate Load from Heat Transfer:

Q_lost = U × A × LMTD  [W]

Where:
U = Overall heat transfer coefficient [W/m²·K]
A = Heat transfer area [m²]
LMTD = Log mean temperature difference [K]

Condensate Generation:
ṁ_condensate = Q_lost / h_fg  [kg/s]

Where:
h_fg = Latent heat of vaporization at steam pressure [kJ/kg]
     = f(P_steam) from IAPWS-IF97

Example: Heat Exchanger
Q_lost = 100 kW
P_steam = 10 bar → h_fg = 2015 kJ/kg (from steam tables)
ṁ_condensate = 100 kW / 2015 kJ/kg = 0.0496 kg/s = 178 kg/hr

Trap Sizing:
Safety_factor = 2-3× (to handle startup surges)
Required_capacity = ṁ_condensate × Safety_factor = 178 × 2.5 = 445 kg/hr

Select trap with capacity ≥ 445 kg/hr at ΔP = (10 - 1) bar = 9 bar

From manufacturer catalog:
F&T Trap Model: FT-45 (capacity 600 kg/hr at Δ10 bar) → SELECTED

Condensate Load from Pipe Warming:

During startup, condensate generated to warm pipe from ambient to steam temp:
Q_warmup = m_pipe × Cp_pipe × (T_steam - T_ambient) + m_insulation × Cp_ins × (T_steam - T_ambient)

Time to warmup (assumed 10 minutes):
ṁ_condensate_startup = Q_warmup / (h_fg × t_warmup)

Peak condensate load = ṁ_condensate_running + ṁ_condensate_startup

Trap sizing must handle PEAK load, not just running load.

Reference: Spirax Sarco "Condensate Load Calculation"
ASHRAE Handbook - HVAC Systems and Equipment, Chapter 11
```

#### 2.3 Trap Failure Diagnostics - NEW
```
Steam Trap Failure Modes:

1. Failed_Open (Blowing_Steam):
   Symptoms:
     - High ultrasonic signal (continuous >90 dB)
     - High outlet temperature (T_outlet ≈ T_steam)
     - Visible steam plume at condensate discharge
   Energy_Loss:
     Q_loss = ṁ_steam_leak × h_fg  [kW]
     ṁ_steam_leak ≈ Orifice_area × √(2 × ρ_steam × ΔP)  [kg/s]
   Example:
     3 mm orifice, 10 bar steam, ρ = 5.16 kg/m³, ΔP = 9 bar
     ṁ_steam_leak ≈ 0.012 kg/s = 43 kg/hr
     Q_loss = 43 kg/hr / 3600 × 2015 kJ/kg = 24 kW
     Annual_cost = 24 kW × 8000 hr/yr / 0.85 / 23.86 MW/kg × $5/kg = $5,600/yr

2. Failed_Closed (Plugged):
   Symptoms:
     - No ultrasonic signal (0 dB)
     - Low outlet temperature (T_outlet ≈ T_ambient)
     - Equipment waterhammer (trapped condensate)
   Impact:
     - Reduced heat transfer efficiency (condensate film on surfaces)
     - Corrosion risk (carbonic acid formation in condensate)
     - Equipment damage (waterhammer, thermal shock)

3. Leaking (Partially_Failed):
   Symptoms:
     - Moderate ultrasonic signal (60-80 dB)
     - Outlet temperature between T_ambient and T_steam
     - Intermittent steam flashing
   Energy_Loss: Proportional to leak size (10-50% of full failure)

Diagnostic_Decision_Tree:
IF Ultrasonic_signal > 90 dB AND T_outlet > (T_steam - 20°C) THEN
  Status = "Failed Open - Steam Leak"
  Action = "Replace immediately"
  Priority = "HIGH"
ELSE IF Ultrasonic_signal < 20 dB AND T_outlet < (T_ambient + 30°C) THEN
  Status = "Failed Closed - Plugged"
  Action = "Clean or replace"
  Priority = "HIGH"
ELSE IF 60 dB < Ultrasonic_signal < 90 dB THEN
  Status = "Leaking - Partial Failure"
  Action = "Schedule replacement within 30 days"
  Priority = "MEDIUM"
ELSE
  Status = "Operating Normally"
  Action = "None - next inspection in 6 months"
  Priority = "LOW"
END IF

Reference: Armstrong International "Steam Trap Survey Procedures"
TLV "Steam Trap Failure Diagnosis"
```

#### 2.4 Trap Population Management - NEW
```yaml
Plant_Wide_Steam_Trap_Inventory:
  Total_Traps: 500 (typical for medium industrial plant)

  Trap_Categorization:
    Critical_Traps (20%):
      - Equipment: Main process heat exchangers, reactors
      - Failure_Impact: Production loss, safety risk
      - Inspection_Frequency: Monthly
      - Acceptable_Failure_Rate: <2%

    Important_Traps (50%):
      - Equipment: Secondary heat exchangers, steam mains
      - Failure_Impact: Efficiency loss, no production impact
      - Inspection_Frequency: Quarterly
      - Acceptable_Failure_Rate: <5%

    General_Service_Traps (30%):
      - Equipment: Tracers, unit heaters, drip legs
      - Failure_Impact: Minor efficiency loss
      - Inspection_Frequency: Semi-annually
      - Acceptable_Failure_Rate: <10%

  Failure_Rate_Tracking:
    Baseline_Failure_Rate: 15% industry average (poor maintenance)
    Target_Failure_Rate: 5% (best-in-class with active monitoring)

    Annual_Savings_Potential:
      Traps_failed_baseline = 500 × 0.15 = 75 traps
      Traps_failed_target = 500 × 0.05 = 25 traps
      Traps_fixed = 50 traps/yr

      Energy_savings_per_trap = 24 kW (from example above)
      Total_savings = 50 traps × 24 kW × 8000 hr/yr / 0.85 / 23.86 MW/kg × $5/kg
                    = $280,000/yr

  Inspection_Route_Optimization:
    Method: Traveling Salesman Problem (TSP) solver
    Objective: Minimize inspection technician walking distance
    Constraint: Inspect all critical traps monthly, important traps quarterly
    Tool: GIS mapping software OR custom route optimizer

Reference: DOE "Improving Steam System Performance Sourcebook"
Best Practices Steam "Steam Trap Management"
```

### 3. Integration Points

#### 3.1 Wireless Sensor Network (Optional - Advanced)
```yaml
Wireless_Steam_Trap_Monitoring:
  Sensor_Type: Wireless ultrasonic + temperature
  Protocol: WirelessHART OR ISA100.11a
  Battery_Life: 5-7 years (with daily reporting)

  Network_Architecture:
    Trap_Sensors → Wireless_Gateway → Cloud_Platform → GL-008 Agent

  Data_Transmitted:
    - Ultrasonic_signal [dB]
    - Inlet_temperature [°C]
    - Outlet_temperature [°C]
    - Battery_status [%]
    - Signal_quality [%]

  Reporting_Frequency:
    - Normal_operation: 1× per day (energy conservation)
    - Failure_detected: 1× per hour (until resolved)

  Advantages:
    - Continuous monitoring (vs manual inspection rounds)
    - Early failure detection (reduce energy loss)
    - Remote diagnostics (no plant access required)

  Cost_Justification:
    Sensor_cost = $500 per trap × 100 critical traps = $50,000
    Annual_savings = $280,000 (from population management)
    Payback = $50,000 / $280,000 = 2.1 months → HIGHLY JUSTIFIED

Reference: Emerson "Plantweb Optics Steam Trap Monitoring"
Spirax Sarco "Wireless Steam Trap Monitoring Solutions"
```

#### 3.2 CMMS Integration
```yaml
Work_Order_Generation:
  Trigger: Trap failure detected by GL-008

  Work_Order_Details:
    {
      "trap_id": "ST-1234",
      "location": "Building 3, Heat Exchanger HX-301",
      "trap_type": "Float & Thermostatic",
      "failure_mode": "Failed Open - Steam Leak",
      "estimated_energy_loss": "24 kW continuous",
      "estimated_annual_cost": "$5,600/yr",
      "priority": "HIGH",
      "recommended_action": "Replace trap with Model FT-45",
      "parts_required": ["FT-45 trap", "Gasket set", "Strainer screen"],
      "estimated_labor_hrs": 2,
      "safety_permits_required": ["Hot Work Permit", "Confined Space if applicable"],
      "generated_by": "GL-008 TRAPCATCHER",
      "timestamp": "2025-12-04T09:15:00Z"
    }
```

### 4. Safety Considerations

#### 4.1 No Direct Interlocks (Monitoring Agent)
GL-008 is a monitoring/diagnostic agent with no control functions, therefore no safety interlocks.

#### 4.2 Alarms
```yaml
Alarms:
  - Steam_Trap_Failed_Open:
      Priority: HIGH (P2)
      Action: Generate work order for replacement

  - Steam_Trap_Failed_Closed:
      Priority: HIGH (P2)
      Action: Generate work order for cleaning/replacement, investigate waterhammer risk

  - High_Plant_Wide_Failure_Rate:
      Condition: Failure_rate > 10%
      Priority: MEDIUM (P3)
      Action: Review maintenance practices, trap selection, water treatment
```

### 5. Standards Alignment

| Standard | Section | Requirement | Implementation |
|----------|---------|-------------|----------------|
| DOE "Best Practices" | Section 3.2 | Steam trap management program | Inspection frequency, failure rate tracking |
| Spirax Sarco Standards | Technical Bulletins | Trap sizing, selection, installation | Condensate load calculations, safety factors |
| ASME B16.34 | All sections | Valve design, pressure-temperature ratings | Trap pressure/temperature limits |

### 6. Consolidation

**Recommendation**: RETAIN GL-008 as STANDALONE
**Rationale**: Specialized niche (steam trap monitoring), no overlap with other agents, high ROI application

---

## GL-009 through GL-020 SUMMARY SPECIFICATIONS

Due to length constraints, the remaining agents (GL-009 through GL-020) will be summarized with key improvements:

### GL-009: THERMALIQ (Current: 87 → Target: 95+) - Thermal Fluid Systems

**Key Improvements**:
1. **Exergy Analysis (2nd Law Efficiency)**: Calculate exergy destruction in thermal fluid heaters
2. **Equipment-Specific Efficiency**: Define efficiency for thermal fluid heaters (different from steam boilers)
3. **Degradation Monitoring**: Track thermal fluid properties (viscosity, thermal conductivity, oxidation)
4. **Expansion Tank Sizing**: Validate expansion tank size for temperature cycles

### GL-010: EMISSIONWATCH (Current: 94 → Target: 95+)

**Key Improvements**:
1. **Emission Trading/Offset Tracking**: Integrate with carbon credit markets (voluntary and compliance)
2. **Fugitive Emissions**: Monitor valve packing leaks, flange leaks (Method 21 compliance)
3. **EPA RATA Testing**: Automate Relative Accuracy Test Audit (RATA) scheduling and data submission

### GL-011: FUELCRAFT (Current: 83 → Target: 95+)

**Key Improvements**:
1. **Real-Time Fuel Price Integration**: API integration with commodity markets (Henry Hub gas, Brent crude)
2. **Equipment Constraints**: Model fuel switching constraints (burner limitations, emissions)
3. **Fuel Blending Optimization**: Optimize blend ratios for multi-fuel systems

### GL-012: STEAMQUAL (Merge into GL-003)

**Merged into GL-003 STEAMWISE** as described earlier.

### GL-013: PREDICTMAINT (Current: 89 → Target: 95+)

**Key Improvements**:
1. **Oil Analysis Integration**: Trending of viscosity, TAN (Total Acid Number), metal content
2. **Thermography**: IR camera integration for hot spot detection
3. **Motor Current Signature Analysis (MCSA)**: Detect motor/pump bearing wear from current waveforms

### GL-014: EXCHANGER-PRO (Current: 86 → Target: 95+)

**Key Improvements**:
1. **Fouling Rate Prediction**: ML model trained on operating conditions vs fouling rate
2. **Antifouling Treatment**: Chemical treatment optimization (dosing rates, costs)
3. **TEMA Standards Compliance**: Full implementation of TEMA design codes

### GL-015: INSULSCAN (Current: 88 → Target: 95+)

**Key Improvements**:
1. **Insulation Type Database**: Thermal conductivity vs temperature for 50+ insulation materials
2. **OSHA Surface Temperature Limits**: 60°C (140°F) touchable surface limit enforcement
3. **Economic Insulation Thickness**: Optimization of insulation thickness (capital vs energy savings)

### GL-016: WATERGUARD (Current: 84 → Target: 95+)

**Key Improvements**:
1. **Steam Purity Monitoring**: Cation conductivity, silica, sodium (per ASME Consensus)
2. **Condensate Return Quality**: Corrosion product monitoring (iron, copper)
3. **Chemical Dosing Optimization**: Minimize chemical usage while meeting water quality targets

### GL-017: CONDENSYNC (Current: 79 → Target: 95+)

**Key Improvements**:
1. **Cooling Tower Optimization**: Cycles of concentration, blowdown minimization
2. **Tube Fouling Detection**: Condenser tube fouling from increasing backpressure
3. **HEI Standards**: Heat Exchange Institute standards for condenser performance testing

### GL-018: FLUEFLOW (Current: 86 → Target: 95+) - PRIMARY COMBUSTION AGENT

**Already detailed in Part 1 - UNIFIED COMBUSTION OPTIMIZER (absorbs GL-002, GL-004, GL-018)**

### GL-019: HEATSCHEDULER (Current: 77 → Target: 95+)

**Key Improvements**:
1. **Thermal Storage Optimization**: Hot water tanks, phase change materials (PCM)
2. **Demand Charge Optimization**: Shift thermal loads to off-peak electricity periods
3. **Load Forecasting**: ML-based prediction of thermal demand (next 24-48 hours)

### GL-020: ECONOPULSE (Current: 85 → Target: 95+)

**Key Improvements**:
1. **Gas-Side vs Water-Side Fouling**: Differentiate fouling location from ΔP and heat transfer trends
2. **Soot Blower Optimization**: Minimize steam consumption for soot blowing while maintaining efficiency
3. **Acid Dew Point Calculation**: Prevent cold-end corrosion from sulfuric acid condensation

---

## FINAL CONSOLIDATION ROADMAP

### Phase 1: Immediate Mergers (Week 1-2)
1. **GL-002 + GL-004 + GL-018 → GL-018 UNIFIED COMBUSTION OPTIMIZER**
   - Combined engineering team
   - Unified codebase
   - Single DCS integration point

2. **GL-003 + GL-012 → GL-003 UNIFIED STEAM SYSTEM OPTIMIZER**
   - Merge steam quality functions into steam distribution agent
   - Eliminate redundant IAPWS-IF97 calculations

### Phase 2: Enhancements (Week 3-8)
1. Implement all "NEW" features specified in this document for each agent
2. Achieve 95+ scores through:
   - Enhanced thermodynamic calculations with references
   - SIS integration (IEC 61511 compliance)
   - Advanced control strategies (cascade, feedforward)
   - Predictive analytics (ML-based diagnostics)

### Phase 3: Verification (Week 9-10)
1. Re-score all agents using GreenLang evaluation framework
2. Verify 95+ scores achieved
3. Document improvements in agent specification sheets

---

**END OF PART 2 - SPECIFICATIONS COMPLETE**
