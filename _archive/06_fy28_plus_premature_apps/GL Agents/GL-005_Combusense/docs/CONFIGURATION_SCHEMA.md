# GL-005 CombustionControlAgent - Configuration Schema

## Overview

This document defines the complete configuration schema for GL-005 CombustionControlAgent. The configuration consists of **85+ parameters** organized into 8 categories covering DCS/PLC connectivity, control loop tuning, safety limits, and optimization parameters.

All configurations support environment variable overrides and HashiCorp Vault secret injection for production deployments.

---

## Configuration File Format

**File:** `config/gl005_config.yaml`

**Schema Version:** 1.0

**Validation:** JSON Schema validation on startup

---

## 1. General Configuration (8 parameters)

```yaml
general:
  # Agent identification
  agent_id: "GL-005"
  agent_name: "CombustionControlAgent"
  version: "1.0.0"

  # Deployment environment
  environment: "production"  # development, staging, production

  # Logging configuration
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_format: "json"  # json, text
  log_output: "stdout"  # stdout, file, both

  # Data retention
  data_retention_days: 90  # Raw sensor data retention
```

**Environment Overrides:**
- `GL005_LOG_LEVEL`
- `GL005_ENVIRONMENT`

---

## 2. DCS Endpoints (15 parameters)

```yaml
dcs:
  # DCS connection (Modbus TCP)
  enabled: true
  protocol: "modbus_tcp"  # modbus_tcp, modbus_rtu

  # Network configuration
  host: "192.168.1.100"  # DCS IP address
  port: 502  # Modbus TCP standard port
  unit_id: 1  # Modbus slave/unit ID

  # Connection management
  timeout_ms: 1000  # Read/write timeout
  retry_count: 3  # Retry attempts on failure
  connection_pool_size: 5  # Concurrent connections

  # Polling configuration
  polling_rate_hz: 100  # Data acquisition frequency (1-1000 Hz)
  polling_mode: "continuous"  # continuous, on_demand

  # Data tags (Modbus register mappings)
  tags:
    fuel_flow_rate:
      register: 40001
      data_type: "float32"  # float32, int16, uint16, int32, uint32
      byte_order: "big_endian"  # big_endian, little_endian
      scaling_factor: 0.1  # Raw value * scaling_factor
      units: "kg/hr"

    air_damper_position:
      register: 40003
      data_type: "float32"
      byte_order: "big_endian"
      scaling_factor: 1.0
      units: "percent"
      writable: true  # Allow writes to this tag

    combustion_temp:
      register: 40005
      data_type: "float32"
      byte_order: "big_endian"
      scaling_factor: 1.0
      units: "celsius"

  # Security (if DCS supports)
  tls_enabled: false  # Enable TLS wrapper (via stunnel)
  certificate_path: "/etc/greenlang/certs/dcs-client.pem"
```

**Secrets (Vault):**
- `dcs/host` (if sensitive)
- `dcs/certificate`

---

## 3. PLC Connection Parameters (12 parameters)

```yaml
plc:
  # PLC connection (OPC UA)
  enabled: true
  protocol: "opcua"  # opcua, modbus_tcp

  # OPC UA endpoint
  endpoint: "opc.tcp://192.168.1.101:4840"
  server_name: "GL005_PLC_Server"

  # Security configuration
  security_mode: "SignAndEncrypt"  # None, Sign, SignAndEncrypt
  security_policy: "Basic256Sha256"  # None, Basic256Sha256, Aes128Sha256RsaOaep

  # Authentication
  auth_method: "username_password"  # anonymous, username_password, certificate
  username: "gl005_agent"
  password: "{VAULT:plc/password}"  # Vault reference

  # Certificates (if using certificate auth)
  client_certificate: "/etc/greenlang/certs/gl005-plc-client.der"
  client_private_key: "/etc/greenlang/certs/gl005-plc-client.pem"
  server_certificate_thumbprint: "A1B2C3D4..."  # Optional: server cert validation

  # Connection management
  timeout_ms: 2000
  retry_count: 3
  session_timeout_ms: 300000  # 5 minutes

  # Subscription configuration
  subscriptions:
    - node_id: "ns=2;s=FuelValve.Position"
      sampling_interval_ms: 100
      queue_size: 10

    - node_id: "ns=2;s=AirDamper.Position"
      sampling_interval_ms: 100
      queue_size: 10

    - node_id: "ns=2;s=FlameScanner.Signal"
      sampling_interval_ms: 50  # High frequency for safety
      queue_size: 20
```

**Secrets (Vault):**
- `plc/username`
- `plc/password`
- `plc/client_private_key`

---

## 4. Combustion Analyzer Settings (10 parameters)

```yaml
cems:
  # Continuous Emissions Monitoring System
  enabled: true
  protocol: "modbus_tcp"  # modbus_tcp, modbus_rtu, analog_4_20ma

  # Analyzers
  o2_analyzer:
    enabled: true
    vendor: "ABB"
    model: "EasyLine"
    host: "192.168.1.102"
    port: 502
    unit_id: 1
    registers:
      o2_percent: 30001
      o2_temperature_c: 30002
      analyzer_status: 30010
    accuracy_percent: 0.1  # ±0.1% O₂
    response_time_sec: 2
    calibration_due_days: 90  # Days until next calibration

  nox_analyzer:
    enabled: true
    vendor: "Siemens"
    model: "Ultramat 23"
    host: "192.168.1.103"
    port: 502
    unit_id: 1
    registers:
      nox_ppm: 30101
      no_ppm: 30102
      no2_ppm: 30103
    accuracy_percent: 2.0  # ±2% of reading
    response_time_sec: 5
    calibration_due_days: 90

  co_co2_analyzer:
    enabled: true
    vendor: "Horiba"
    model: "PG-350"
    host: "192.168.1.104"
    port: 502
    unit_id: 1
    registers:
      co_ppm: 30201
      co2_percent: 30202
    accuracy_percent: 1.0  # ±1% of full scale
    response_time_sec: 3
    calibration_due_days: 90
```

**Maintenance Alerts:**
- Automated alerts when `calibration_due_days < 7`

---

## 5. Control Loop Parameters (20 parameters)

```yaml
control:
  # Control modes
  default_mode: "automatic"  # automatic, manual, setpoint_only
  allow_manual_override: true

  # Primary control loop (Heat Output PID)
  primary_loop:
    enabled: true
    description: "Heat output to fuel flow control"

    # PID tuning (Ziegler-Nichols method)
    kp: 1.2  # Proportional gain
    ki: 0.5  # Integral gain
    kd: 0.1  # Derivative gain

    # Setpoint
    setpoint_mw: 50.0  # Default heat demand (MW)
    setpoint_tracking_tolerance_percent: 0.5  # ±0.5% acceptable error

    # Control limits
    output_min: 0  # Minimum fuel flow (0 = shutdown)
    output_max: 5000  # Maximum fuel flow (kg/hr)

    # Anti-windup
    anti_windup_enabled: true
    integral_clamp_min: -1000
    integral_clamp_max: 1000

    # Filtering
    derivative_filter_enabled: true
    derivative_filter_time_constant: 0.05  # 50ms filter

  # Secondary control loop (Air-Fuel Ratio PID)
  secondary_loop:
    enabled: true
    description: "O₂ to air damper control"

    # PID tuning
    kp: 0.8
    ki: 0.3
    kd: 0.05

    # Setpoint
    setpoint_o2_percent: 3.0  # Target O₂ in flue gas
    setpoint_tracking_tolerance_percent: 2.0  # ±2% acceptable

    # Control limits
    output_min: 0  # Damper fully closed
    output_max: 100  # Damper fully open

    # Anti-windup
    anti_windup_enabled: true
    integral_clamp_min: -50
    integral_clamp_max: 50

  # Feedforward compensation
  feedforward:
    enabled: true
    fuel_quality_compensation: true  # Compensate for heating value changes
    ambient_temp_compensation: true  # Compensate for ambient temperature
    load_anticipation: true  # Anticipate load changes

  # Control loop timing
  cycle_time_ms: 100  # 100ms = 10 Hz control loop
  max_latency_ms: 50  # Alert if control latency > 50ms

  # Ramp rates (for setpoint changes)
  max_heat_ramp_rate_mw_per_min: 5.0  # Max heat increase rate
  max_fuel_ramp_rate_kg_hr_per_min: 100  # Max fuel flow change rate
  max_damper_ramp_rate_percent_per_min: 10  # Max damper position change rate
```

**Tuning Notes:**
- PID parameters calculated using Ziegler-Nichols or Cohen-Coon methods
- Field tuning recommended for optimal performance
- Store tuned parameters in database for each combustion unit

---

## 6. Safety Limits (15 parameters)

```yaml
safety:
  # Safety system configuration
  safety_level: "SIL-2"  # SIL-1, SIL-2, SIL-3 (per IEC 61508)
  triple_redundancy: true  # Use 2-out-of-3 voting for critical sensors

  # Temperature limits
  max_combustion_chamber_temp_c: 1400  # Critical: Emergency shutdown
  max_flue_gas_temp_c: 600  # Warning: Efficiency concern
  min_combustion_temp_c: 800  # Warning: Incomplete combustion

  # Pressure limits
  max_combustion_chamber_pressure_bar: 5.0  # Critical: Overpressure
  min_combustion_chamber_pressure_mbar: -10  # Critical: Vacuum condition
  max_fuel_pressure_bar: 20  # Warning: High fuel pressure
  min_fuel_pressure_bar: 0.5  # Critical: Low fuel pressure (flameout risk)

  # Oxygen limits
  min_o2_percent: 2.0  # Critical: Risk of incomplete combustion
  max_o2_percent: 18.0  # Warning: Excessive air (efficiency loss)

  # Flame monitoring
  min_flame_signal_percent: 30  # Critical: Weak flame (flameout risk)
  flame_failure_delay_sec: 2  # Delay before emergency shutdown

  # Fuel flow limits
  max_fuel_flow_kg_hr: 5000  # Critical: Maximum capacity
  min_fuel_flow_kg_hr: 500  # Minimum stable combustion

  # Emissions limits (alarm thresholds)
  max_nox_ppm: 50  # Regulatory limit
  max_co_ppm: 100  # Regulatory limit
  max_co2_percent: 15  # Warning: Poor combustion

  # Interlock logic
  interlocks:
    - name: "flame_failure_interlock"
      condition: "flame_signal < min_flame_signal_percent"
      action: "emergency_shutdown"
      delay_sec: 2

    - name: "high_temperature_interlock"
      condition: "combustion_temp > max_combustion_chamber_temp_c"
      action: "emergency_shutdown"
      delay_sec: 0  # Immediate

    - name: "low_fuel_pressure_interlock"
      condition: "fuel_pressure < min_fuel_pressure_bar"
      action: "emergency_shutdown"
      delay_sec: 1

    - name: "low_o2_interlock"
      condition: "o2_percent < min_o2_percent"
      action: "reduce_fuel_flow"
      delay_sec: 5

  # Emergency shutdown sequence
  emergency_shutdown:
    close_fuel_valve_time_sec: 1.0  # Close main fuel valve in 1 second
    open_purge_damper: true  # Open air damper for purge
    purge_duration_sec: 300  # 5-minute purge before restart allowed
    alarm_notification: true
```

**Safety Notes:**
- All safety limits are configurable but require supervisor approval
- Triple-redundant sensors use 2-out-of-3 voting to prevent spurious trips
- Safety system is independent of control system (SIL-2 rated)

---

## 7. Optimization Parameters (10 parameters)

```yaml
optimization:
  # Multi-objective optimization
  enabled: true
  algorithm: "weighted_sum"  # weighted_sum, pareto_front, nsga2

  # Objective weights (must sum to 1.0)
  efficiency_weight: 0.5  # Maximize combustion efficiency
  emissions_weight: 0.3  # Minimize NOx and CO
  stability_weight: 0.2  # Minimize oscillations

  # Optimization constraints
  constraints:
    min_efficiency_percent: 85  # Must achieve minimum efficiency
    max_nox_ppm: 50  # Must stay under NOx limit
    max_co_ppm: 100  # Must stay under CO limit
    min_excess_air_percent: 5  # Safety margin for combustion
    max_excess_air_percent: 25  # Efficiency limit

  # Optimization solver
  solver: "scipy_minimize"  # scipy_minimize, cvxpy, custom
  solver_method: "SLSQP"  # Sequential Least Squares Programming
  max_iterations: 100
  convergence_tolerance: 1e-6

  # Optimization frequency
  optimization_interval_sec: 60  # Re-optimize every minute

  # Fuel cost (for economic optimization)
  fuel_cost_usd_per_kg: 0.05  # Natural gas ~$0.05/kg
  carbon_price_usd_per_ton: 50  # Carbon credit price
```

**Optimization Methods:**
- **Weighted Sum:** Simple, fast, single optimal solution
- **Pareto Front:** Multiple solutions along efficiency-emissions trade-off
- **NSGA-II:** Genetic algorithm for complex multi-objective problems

---

## 8. Monitoring Thresholds (10 parameters)

```yaml
monitoring:
  # Prometheus metrics
  metrics_enabled: true
  metrics_port: 8001
  metrics_path: "/metrics"

  # Performance thresholds (for alerting)
  thresholds:
    # Control performance
    max_control_loop_latency_ms: 100  # Alert if > 100ms
    max_setpoint_error_percent: 5  # Alert if > 5% deviation
    min_control_stability_index: 0.8  # Alert if < 0.8 (0-1 scale)

    # Efficiency
    min_combustion_efficiency_percent: 85  # Alert if < 85%
    min_thermal_efficiency_percent: 80  # Alert if < 80%

    # Emissions
    max_nox_ppm: 50  # Regulatory limit
    max_co_ppm: 100  # Regulatory limit
    max_co2_intensity_kg_mwh: 200  # Carbon intensity limit

    # Data quality
    min_data_quality_percent: 90  # Alert if < 90% good data
    max_sensor_stale_time_sec: 10  # Alert if sensor not updated in 10s

  # Alert destinations
  alerts:
    email:
      enabled: true
      recipients: ["operations@example.com", "control-room@example.com"]
      smtp_server: "smtp.example.com"
      smtp_port: 587

    slack:
      enabled: true
      webhook_url: "{VAULT:monitoring/slack_webhook}"
      channel: "#gl005-combustion-control"

    pagerduty:
      enabled: true
      integration_key: "{VAULT:monitoring/pagerduty_key}"

  # Grafana dashboards
  dashboards:
    - name: "Real-Time Control Performance"
      enabled: true
      refresh_interval_sec: 5

    - name: "Emissions Monitoring"
      enabled: true
      refresh_interval_sec: 10

    - name: "Safety Systems"
      enabled: true
      refresh_interval_sec: 1
```

**Alert Severity Levels:**
- **INFO:** Informational, no action required
- **WARNING:** Investigate within 1 hour
- **CRITICAL:** Immediate action required
- **EMERGENCY:** Safety issue, emergency shutdown

---

## Complete Configuration Example

```yaml
# GL-005 CombustionControlAgent - Complete Configuration
# File: config/gl005_config.yaml
# Schema Version: 1.0

general:
  agent_id: "GL-005"
  agent_name: "CombustionControlAgent"
  version: "1.0.0"
  environment: "production"
  log_level: "INFO"
  log_format: "json"
  log_output: "stdout"
  data_retention_days: 90

dcs:
  enabled: true
  protocol: "modbus_tcp"
  host: "192.168.1.100"
  port: 502
  unit_id: 1
  timeout_ms: 1000
  retry_count: 3
  connection_pool_size: 5
  polling_rate_hz: 100
  polling_mode: "continuous"
  tags:
    fuel_flow_rate:
      register: 40001
      data_type: "float32"
      byte_order: "big_endian"
      scaling_factor: 0.1
      units: "kg/hr"
    air_damper_position:
      register: 40003
      data_type: "float32"
      byte_order: "big_endian"
      scaling_factor: 1.0
      units: "percent"
      writable: true

plc:
  enabled: true
  protocol: "opcua"
  endpoint: "opc.tcp://192.168.1.101:4840"
  security_mode: "SignAndEncrypt"
  security_policy: "Basic256Sha256"
  auth_method: "username_password"
  username: "gl005_agent"
  password: "{VAULT:plc/password}"
  timeout_ms: 2000
  retry_count: 3

cems:
  enabled: true
  o2_analyzer:
    enabled: true
    vendor: "ABB"
    host: "192.168.1.102"
    port: 502
    unit_id: 1
    registers:
      o2_percent: 30001
    accuracy_percent: 0.1
    response_time_sec: 2

control:
  default_mode: "automatic"
  primary_loop:
    kp: 1.2
    ki: 0.5
    kd: 0.1
    setpoint_mw: 50.0
    output_max: 5000
    anti_windup_enabled: true
  secondary_loop:
    kp: 0.8
    ki: 0.3
    kd: 0.05
    setpoint_o2_percent: 3.0
    output_max: 100
  cycle_time_ms: 100
  max_latency_ms: 50

safety:
  safety_level: "SIL-2"
  triple_redundancy: true
  max_combustion_chamber_temp_c: 1400
  min_o2_percent: 2.0
  min_flame_signal_percent: 30
  max_fuel_flow_kg_hr: 5000
  max_nox_ppm: 50
  max_co_ppm: 100

optimization:
  enabled: true
  efficiency_weight: 0.5
  emissions_weight: 0.3
  stability_weight: 0.2
  solver: "scipy_minimize"
  max_iterations: 100
  optimization_interval_sec: 60

monitoring:
  metrics_enabled: true
  metrics_port: 8001
  thresholds:
    max_control_loop_latency_ms: 100
    min_combustion_efficiency_percent: 85
    max_nox_ppm: 50
  alerts:
    email:
      enabled: true
      recipients: ["operations@example.com"]
```

---

## Environment Variable Overrides

All configuration parameters support environment variable overrides using the format:

```
GL005_{SECTION}_{PARAMETER}
```

**Examples:**
```bash
export GL005_GENERAL_LOG_LEVEL=DEBUG
export GL005_DCS_HOST=192.168.2.100
export GL005_CONTROL_PRIMARY_LOOP_KP=1.5
export GL005_SAFETY_MAX_COMBUSTION_CHAMBER_TEMP_C=1500
```

Priority order:
1. Environment variables (highest)
2. Configuration file
3. Default values (lowest)

---

## Vault Secret References

Sensitive parameters can reference HashiCorp Vault secrets using the format:

```yaml
parameter: "{VAULT:path/to/secret}"
```

**Examples:**
```yaml
plc:
  password: "{VAULT:plc/password}"
  client_private_key: "{VAULT:plc/client_key}"

monitoring:
  alerts:
    slack:
      webhook_url: "{VAULT:monitoring/slack_webhook}"
```

Vault secrets are automatically injected at runtime using the `hvac` Python library.

---

## Configuration Validation

Configuration is validated on startup using JSON Schema. Invalid configurations will prevent agent startup with detailed error messages.

**Validation checks:**
- Required parameters present
- Data types correct (int, float, string, bool)
- Values within allowed ranges
- Cross-parameter consistency (e.g., weights sum to 1.0)
- Network endpoints reachable (optional pre-flight check)

**Validation tool:**
```bash
python -m gl005.validate_config --config config/gl005_config.yaml
```

---

## Configuration Best Practices

1. **Start with defaults:** Use provided default configuration, customize only as needed
2. **Version control:** Store configurations in Git with change history
3. **Environment-specific:** Maintain separate configs for dev/staging/prod
4. **Secrets management:** Never commit secrets, always use Vault references
5. **Document changes:** Add comments explaining non-standard parameter values
6. **Test thoroughly:** Validate configuration changes in staging before production
7. **Backup:** Backup production configurations before changes

---

**Document Version:** 1.0
**Last Updated:** 2025-01-18
**Total Parameters:** 85+
