# Demo Environment Architecture Specification

**Document:** Demo Environment Technical Architecture
**Version:** 1.0
**Date:** December 5, 2025
**Owner:** Product Management / Solution Architecture
**Status:** Specification

---

## Executive Summary

This document specifies the technical architecture for GreenLang Process Heat product demonstration environments. The demo environment enables sales teams, partners, and prospective customers to experience the full capabilities of ThermalCommand, BoilerOptimizer, WasteHeatRecovery, and EmissionsGuardian products with realistic, industry-specific data scenarios.

### Business Requirements

1. **Sales Enablement:** Enable sales team to deliver compelling product demonstrations
2. **Self-Service Trials:** Allow qualified prospects to explore products independently
3. **Partner Training:** Provide partners with hands-on experience for certification
4. **Proof of Concept:** Support rapid POC deployments with customer data

### Key Metrics

| Metric | Target |
|--------|--------|
| Demo provisioning time | < 5 minutes |
| Demo availability | 99.9% |
| Concurrent demo sessions | 50+ |
| Demo reset/refresh time | < 1 minute |
| Data freshness | Real-time simulation |

---

## 1. Architecture Overview

### 1.1 High-Level Architecture

```
                                    DEMO ENVIRONMENT ARCHITECTURE
    +-----------------------------------------------------------------------------------------+
    |                                                                                          |
    |  +------------------+    +------------------+    +------------------+                    |
    |  |  DEMO PORTAL     |    |  DEMO PLATFORM   |    |  DEMO DATA       |                    |
    |  +------------------+    +------------------+    +------------------+                    |
    |  |                  |    |                  |    |                  |                    |
    |  | - Self-service   |    | - Multi-tenant   |    | - Synthetic data |                    |
    |  |   provisioning   |--->|   sandboxes      |<---|   generator      |                    |
    |  | - Guided demos   |    | - Full product   |    | - 5 industry     |                    |
    |  | - Lead capture   |    |   functionality  |    |   scenarios      |                    |
    |  | - Usage tracking |    | - Isolated data  |    | - Live simulation|                    |
    |  |                  |    |                  |    |                  |                    |
    |  +------------------+    +------------------+    +------------------+                    |
    |           |                      |                      |                               |
    |           v                      v                      v                               |
    |  +------------------+    +------------------+    +------------------+                    |
    |  |  DEMO SCRIPTS    |    |  DEMO SCENARIOS  |    |  DEMO ANALYTICS  |                    |
    |  +------------------+    +------------------+    +------------------+                    |
    |  |                  |    |                  |    |                  |                    |
    |  | - Product tours  |    | - Oil & Gas      |    | - Usage metrics  |                    |
    |  | - Feature demos  |    | - Chemicals      |    | - Lead scoring   |                    |
    |  | - Use case walks |    | - Steel/Metals   |    | - Conversion     |                    |
    |  | - Persona-based  |    | - Food & Bev     |    |   tracking       |                    |
    |  |                  |    | - Cement         |    |                  |                    |
    |  +------------------+    +------------------+    +------------------+                    |
    |                                                                                          |
    +-----------------------------------------------------------------------------------------+
```

### 1.2 Component Diagram

```
+-------------------+     +-------------------+     +-------------------+
|   Demo Portal     |     |   Demo Platform   |     |   Data Services   |
+-------------------+     +-------------------+     +-------------------+
|                   |     |                   |     |                   |
| React Frontend    |     | GreenLang SaaS    |     | Data Generator    |
| - Landing page    |<--->| (Multi-tenant)    |<--->| - Synthetic data  |
| - Registration    |     | - ThermalCommand  |     | - Real-time sim   |
| - Sandbox access  |     | - BoilerOptimizer |     | - Event injection |
| - Guided tours    |     | - WasteHeatRec    |     |                   |
|                   |     | - EmissionsGuard  |     | Data Store        |
| Auth Service      |     |                   |     | - PostgreSQL      |
| - Demo tokens     |     | Sandbox Manager   |     | - TimescaleDB     |
| - SSO bypass      |     | - Provisioning    |     | - Redis cache     |
| - Session mgmt    |     | - Isolation       |     |                   |
|                   |     | - Reset/refresh   |     | Simulation Engine |
+-------------------+     +-------------------+     | - Process models  |
                                                    | - Anomaly inject  |
                                                    | - Time compression|
                                                    +-------------------+
```

### 1.3 Deployment Architecture

```yaml
demo_deployment:
  environment: demo-production
  cloud_provider: AWS
  region: us-east-1  # Primary
  dr_region: eu-west-1  # Secondary for EU demos

  kubernetes:
    cluster: demo-eks-cluster
    node_groups:
      - name: demo-platform
        instance_type: m6i.xlarge
        min_size: 3
        max_size: 10
        labels:
          workload: demo-platform
      - name: demo-data
        instance_type: r6i.large
        min_size: 2
        max_size: 5
        labels:
          workload: demo-data

  namespaces:
    - demo-portal
    - demo-platform
    - demo-data
    - demo-monitoring

  ingress:
    type: ALB
    domains:
      - demo.greenlang.com
      - try.greenlang.com
    ssl: ACM managed
```

---

## 2. Demo Portal

### 2.1 Portal Features

| Feature | Description | Priority |
|---------|-------------|----------|
| Self-Service Registration | Prospect signs up with email/LinkedIn | P0 |
| Demo Selection | Choose product and industry scenario | P0 |
| Instant Provisioning | Sandbox ready in < 5 minutes | P0 |
| Guided Tours | Interactive product walkthroughs | P0 |
| Feature Exploration | Free-form product exploration | P0 |
| Lead Capture | CRM integration (Salesforce/HubSpot) | P0 |
| Session Recording | Optional screen recording | P1 |
| Demo Scheduling | Book live demo with sales | P1 |
| Partner Portal | Partner-specific demo access | P1 |

### 2.2 User Flows

```
SELF-SERVICE DEMO FLOW:

[Landing Page] --> [Registration Form] --> [Email Verification]
                           |
                           v
                   [Demo Selection]
                   - Product: ThermalCommand / BoilerOptimizer / etc.
                   - Industry: Oil&Gas / Chemicals / Steel / etc.
                   - Role: Executive / Engineer / Operator
                           |
                           v
                   [Sandbox Provisioning]
                   - Create demo tenant
                   - Load industry data
                   - Configure dashboards
                   - Initialize ML models
                           |
                           v
                   [Demo Experience]
                   - Guided tour (optional)
                   - Feature exploration
                   - Sample workflows
                   - Outcome preview
                           |
                           v
                   [Call to Action]
                   - Request pricing
                   - Schedule sales call
                   - Download materials
                   - Start POC
```

### 2.3 Portal Technical Stack

```yaml
portal_tech_stack:
  frontend:
    framework: React 18
    styling: Tailwind CSS
    state: React Query + Zustand
    analytics: Amplitude
    tours: Shepherd.js

  backend:
    framework: FastAPI (Python)
    database: PostgreSQL
    cache: Redis
    queue: Celery + Redis

  integrations:
    crm: Salesforce / HubSpot API
    email: SendGrid
    analytics: Segment
    recording: Loom API (optional)
```

---

## 3. Demo Platform

### 3.1 Multi-Tenant Sandbox Architecture

```
SANDBOX ISOLATION MODEL:

+------------------------------------------------------------------+
|                        DEMO PLATFORM                              |
+------------------------------------------------------------------+
|                                                                   |
|   +-------------------+  +-------------------+  +-------------------+
|   |   SANDBOX A       |  |   SANDBOX B       |  |   SANDBOX C       |
|   |   (Oil & Gas)     |  |   (Chemicals)     |  |   (Steel)         |
|   +-------------------+  +-------------------+  +-------------------+
|   |                   |  |                   |  |                   |
|   | - Tenant: demo_001|  | - Tenant: demo_002|  | - Tenant: demo_003|
|   | - Products: TC,EG |  | - Products: BO,WHR|  | - Products: TC,EG |
|   | - Assets: 150     |  | - Assets: 50      |  | - Assets: 200     |
|   | - Users: 3        |  | - Users: 2        |  | - Users: 5        |
|   | - Data: Isolated  |  | - Data: Isolated  |  | - Data: Isolated  |
|   | - ML: Pre-trained |  | - ML: Pre-trained |  | - ML: Pre-trained |
|   |                   |  |                   |  |                   |
|   | [RESET POLICY]    |  | [RESET POLICY]    |  | [RESET POLICY]    |
|   | - Auto: 24 hours  |  | - Auto: 24 hours  |  | - Auto: 24 hours  |
|   | - Manual: Allowed |  | - Manual: Allowed |  | - Manual: Allowed |
|   +-------------------+  +-------------------+  +-------------------+
|                                                                   |
+------------------------------------------------------------------+
```

### 3.2 Sandbox Specifications

| Specification | Value | Notes |
|---------------|-------|-------|
| Max concurrent sandboxes | 100 | Auto-scale as needed |
| Sandbox lifetime | 24 hours | Extendable for qualified leads |
| Data isolation | Complete | Tenant-level database schemas |
| Product features | 100% | Full functionality enabled |
| ML models | Pre-trained | Industry-specific models |
| Performance | Production-like | Same response times |
| Storage limit | 10 GB | Per sandbox |
| API rate limit | 1000/min | Per sandbox |

### 3.3 Sandbox Provisioning API

```yaml
openapi: 3.0.0
info:
  title: Demo Sandbox API
  version: 1.0.0

paths:
  /api/v1/sandboxes:
    post:
      summary: Create new demo sandbox
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required:
                - email
                - product
                - industry
              properties:
                email:
                  type: string
                  format: email
                product:
                  type: string
                  enum: [thermalcommand, boileroptimizer, wasteheatrecovery, emissionsguardian]
                industry:
                  type: string
                  enum: [oil_gas, chemicals, steel, food_beverage, cement]
                company:
                  type: string
                role:
                  type: string
                  enum: [executive, engineer, operator, consultant]
      responses:
        201:
          content:
            application/json:
              schema:
                type: object
                properties:
                  sandbox_id:
                    type: string
                  access_url:
                    type: string
                  access_token:
                    type: string
                  expires_at:
                    type: string
                    format: date-time
                  credentials:
                    type: object
                    properties:
                      username:
                        type: string
                      temporary_password:
                        type: string

  /api/v1/sandboxes/{sandbox_id}/reset:
    post:
      summary: Reset sandbox to initial state
      responses:
        200:
          description: Sandbox reset successful

  /api/v1/sandboxes/{sandbox_id}/extend:
    post:
      summary: Extend sandbox lifetime
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                hours:
                  type: integer
                  minimum: 1
                  maximum: 168  # Max 1 week
                reason:
                  type: string
```

---

## 4. Demo Data

### 4.1 Synthetic Data Generator

The data generator creates realistic, industry-specific operational data for demo environments.

```python
# Demo Data Generator Architecture

class DemoDataGenerator:
    """
    Generates synthetic operational data for demo environments.
    """

    def __init__(self, industry: str, duration_months: int = 12):
        self.industry = industry
        self.duration = duration_months
        self.config = INDUSTRY_CONFIGS[industry]

    def generate_facility(self) -> Facility:
        """Generate complete facility with assets and data."""
        facility = Facility(
            name=self.config.facility_name,
            industry=self.industry,
            assets=self._generate_assets(),
            sensors=self._generate_sensors(),
            historical_data=self._generate_historical_data(),
            anomalies=self._inject_anomalies(),
            events=self._generate_events()
        )
        return facility

    def _generate_assets(self) -> List[Asset]:
        """Generate assets based on industry template."""
        pass

    def _generate_sensors(self) -> List[Sensor]:
        """Generate sensor data points."""
        pass

    def _generate_historical_data(self) -> DataFrame:
        """Generate 12+ months of historical data."""
        pass

    def _inject_anomalies(self) -> List[Anomaly]:
        """Inject realistic anomalies for demo scenarios."""
        pass

    def _generate_events(self) -> List[Event]:
        """Generate maintenance events, alarms, etc."""
        pass
```

### 4.2 Industry Scenario Configurations

#### Scenario 1: Oil & Gas Refinery

```yaml
oil_gas_refinery:
  name: "Demo Refinery Complex"
  annual_energy_cost: $25,000,000
  co2_emissions: 150,000 tons/year

  assets:
    process_heaters:
      count: 12
      capacity_range: [20, 100]  # MMBtu/hr
      fuel: natural_gas

    boilers:
      count: 5
      capacity_range: [80, 200]  # MMBtu/hr
      steam_pressure: [150, 600]  # psig

    heat_exchangers:
      count: 45
      types: [shell_tube, plate, air_cooled]

    steam_system:
      headers: 3
      traps: 150
      turbines: 4

  data_characteristics:
    sampling_rate: 1_minute
    historical_depth: 12_months
    anomaly_rate: 2_percent

  demo_scenarios:
    - name: "Combustion Optimization"
      problem: "Excess air running 25% above optimal"
      opportunity: "$1.2M annual savings"
      features: [o2_trim, efficiency_monitoring, combustion_tuning]

    - name: "Predictive Maintenance"
      problem: "Unexpected heater tube failures"
      opportunity: "40% reduction in unplanned downtime"
      features: [tmt_monitoring, failure_prediction, work_order_gen]

    - name: "Emissions Compliance"
      problem: "NOx approaching permit limits"
      opportunity: "100% compliance, avoid $2M penalties"
      features: [real_time_emissions, predictive_alerts, auto_reports]
```

#### Scenario 2: Chemical Plant

```yaml
chemical_plant:
  name: "Demo Chemical Works"
  annual_energy_cost: $12,000,000
  co2_emissions: 80,000 tons/year

  assets:
    reactors:
      count: 8
      types: [batch, continuous]
      heating: steam_jacketed

    distillation_columns:
      count: 6
      reboilers: shell_tube
      condensers: air_cooled

    boilers:
      count: 3
      capacity_range: [50, 100]  # MMBtu/hr

    heat_recovery:
      economizers: 3
      waste_heat_boilers: 2
      air_preheaters: 2

  demo_scenarios:
    - name: "Steam System Optimization"
      problem: "Steam losses exceeding 15%"
      opportunity: "$800K annual savings"
      features: [steam_balance, trap_monitoring, header_optimization]

    - name: "Waste Heat Recovery"
      problem: "30% of heat rejected to cooling"
      opportunity: "$1.5M in new recovery opportunities"
      features: [pinch_analysis, opportunity_finder, project_economics]
```

#### Scenario 3: Steel Mill

```yaml
steel_mill:
  name: "Demo Steel Works"
  annual_energy_cost: $20,000,000
  co2_emissions: 200,000 tons/year

  assets:
    reheat_furnaces:
      count: 4
      capacity: 150  # tons/hr
      fuel: mixed_gas

    ladle_heaters:
      count: 6
      type: regenerative

    annealing_furnaces:
      count: 3
      atmosphere: controlled

    waste_heat_recovery:
      recuperators: 4
      waste_heat_boilers: 2

  demo_scenarios:
    - name: "Furnace Optimization"
      problem: "Specific fuel consumption 15% above benchmark"
      opportunity: "$2.5M fuel cost reduction"
      features: [thermal_profiling, tmt_monitoring, combustion_control]

    - name: "EU ETS Compliance"
      problem: "CO2 allowance costs rising"
      opportunity: "15% emissions reduction, $1.8M cost avoidance"
      features: [carbon_tracking, reduction_planning, allowance_management]
```

#### Scenario 4: Food & Beverage

```yaml
food_beverage:
  name: "Demo Food Processing Plant"
  annual_energy_cost: $5,000,000
  co2_emissions: 30,000 tons/year

  assets:
    cooking_systems:
      count: 8
      types: [batch_cookers, continuous_cookers]

    drying_systems:
      count: 4
      types: [spray_dryer, drum_dryer]

    sterilization:
      count: 6
      types: [retort, pasteurizer]

    boilers:
      count: 3
      capacity_range: [20, 60]  # MMBtu/hr

    refrigeration:
      count: 10
      types: [ammonia, co2]

  demo_scenarios:
    - name: "Energy Efficiency"
      problem: "Energy intensity 20% above best-in-class"
      opportunity: "$600K annual savings"
      features: [monitoring, efficiency_tracking, load_scheduling]

    - name: "Heat Recovery from Refrigeration"
      problem: "Refrigeration waste heat not captured"
      opportunity: "$200K in heat reuse"
      features: [waste_heat_analysis, heat_pump_integration]
```

#### Scenario 5: Cement Plant

```yaml
cement_plant:
  name: "Demo Cement Works"
  annual_energy_cost: $15,000,000
  co2_emissions: 250,000 tons/year

  assets:
    rotary_kiln:
      count: 1
      capacity: 5000  # tons/day
      fuel: coal_petcoke

    preheater_tower:
      stages: 5
      calciner: in_line

    clinker_cooler:
      type: grate
      heat_recovery: yes

    coal_mill:
      count: 2
      type: vertical_roller

  demo_scenarios:
    - name: "Kiln Optimization"
      problem: "Specific heat consumption above 750 kcal/kg"
      opportunity: "5% fuel reduction, $1.2M savings"
      features: [kiln_monitoring, flame_optimization, coating_analysis]

    - name: "Carbon Reduction"
      problem: "CO2 intensity above EU benchmark"
      opportunity: "Meet EU ETS Phase 4 requirements"
      features: [carbon_accounting, alternative_fuels, reduction_roadmap]
```

### 4.3 Real-Time Simulation Engine

```python
class RealTimeSimulator:
    """
    Simulates real-time operational data for demo environments.
    Provides live data updates without requiring actual industrial systems.
    """

    def __init__(self, facility: Facility, time_compression: float = 1.0):
        self.facility = facility
        self.time_compression = time_compression  # 1.0 = real-time, 60.0 = 1 hour per minute
        self.running = False

    async def start(self):
        """Start real-time simulation."""
        self.running = True
        while self.running:
            # Update all sensor values
            for sensor in self.facility.sensors:
                new_value = self._simulate_sensor(sensor)
                await self._publish_value(sensor.id, new_value)

            # Check for triggered events
            events = self._check_events()
            for event in events:
                await self._publish_event(event)

            # Sleep based on time compression
            await asyncio.sleep(1.0 / self.time_compression)

    def _simulate_sensor(self, sensor: Sensor) -> float:
        """Generate next sensor value based on process model."""
        base_value = sensor.current_value

        # Apply process dynamics
        process_change = self._apply_process_model(sensor)

        # Apply noise
        noise = random.gauss(0, sensor.noise_sigma)

        # Apply any active anomalies
        anomaly_effect = self._apply_anomalies(sensor)

        return base_value + process_change + noise + anomaly_effect

    def inject_anomaly(self, anomaly_type: str, asset_id: str, duration: int):
        """Inject a specific anomaly for demo purposes."""
        # Examples: efficiency_drop, sensor_drift, equipment_failure
        pass

    def trigger_event(self, event_type: str, asset_id: str):
        """Trigger a specific event for demo purposes."""
        # Examples: alarm, maintenance_due, permit_exceedance
        pass
```

---

## 5. Demo Analytics

### 5.1 Usage Tracking

Track all demo interactions for lead scoring and product improvement:

```yaml
demo_analytics:
  events_tracked:
    # Session events
    - demo_started
    - demo_completed
    - demo_abandoned
    - demo_extended

    # Feature engagement
    - feature_viewed
    - feature_clicked
    - dashboard_accessed
    - report_generated
    - alert_acknowledged

    # Value moments
    - savings_calculated
    - optimization_applied
    - anomaly_detected
    - recommendation_viewed

    # Conversion signals
    - pricing_requested
    - sales_contact_requested
    - trial_extended
    - poc_started

  lead_scoring:
    high_intent:
      - pricing_requested: +50 points
      - sales_contact_requested: +40 points
      - demo_extended: +30 points
      - feature_depth > 5: +25 points

    medium_intent:
      - demo_completed: +20 points
      - report_generated: +15 points
      - multiple_sessions: +10 points

    low_intent:
      - demo_started: +5 points
      - quick_abandonment: -10 points

  dashboards:
    - name: Demo Performance
      metrics: [demos_started, completion_rate, avg_duration, conversion_rate]

    - name: Lead Quality
      metrics: [leads_generated, mql_rate, sql_rate, opportunity_value]

    - name: Product Engagement
      metrics: [feature_usage, top_features, drop_off_points]
```

### 5.2 CRM Integration

```yaml
crm_integration:
  platforms:
    - salesforce
    - hubspot

  sync_triggers:
    - demo_completed
    - high_intent_behavior
    - pricing_requested
    - sales_contact_requested

  data_synced:
    contact:
      - email
      - company
      - role
      - phone (if provided)

    demo_activity:
      - product_demoed
      - industry_scenario
      - features_explored
      - duration
      - engagement_score

    lead_score:
      - total_score
      - score_breakdown
      - recommended_action
```

---

## 6. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)

| Task | Owner | Deliverable |
|------|-------|-------------|
| Deploy demo K8s cluster | DevOps | Running cluster |
| Create sandbox provisioning service | Backend | API v1 |
| Build demo portal MVP | Frontend | Landing + registration |
| Create 2 industry data sets | Data Eng | Oil&Gas, Steel |
| Implement basic guided tour | Product | ThermalCommand tour |

### Phase 2: Core Features (Weeks 4-6)

| Task | Owner | Deliverable |
|------|-------|-------------|
| Complete all 5 industry scenarios | Data Eng | Full data sets |
| Implement real-time simulation | Backend | Live simulation |
| Add all product tours | Product | 4 product tours |
| Integrate CRM | Backend | Salesforce sync |
| Launch beta to sales team | All | Internal demo |

### Phase 3: Scale & Polish (Weeks 7-9)

| Task | Owner | Deliverable |
|------|-------|-------------|
| Performance optimization | DevOps | < 5 min provisioning |
| Analytics dashboard | Data | Usage analytics |
| Partner portal | Frontend | Partner access |
| Mobile optimization | Frontend | Responsive design |
| Public launch | Marketing | demo.greenlang.com |

---

## 7. Success Metrics

### Launch Criteria

- [ ] All 4 products demonstrable
- [ ] All 5 industry scenarios available
- [ ] Provisioning < 5 minutes
- [ ] 99.9% availability achieved
- [ ] CRM integration operational
- [ ] Analytics tracking complete
- [ ] Sales team trained

### Post-Launch KPIs

| Metric | Target (30 days) | Target (90 days) |
|--------|------------------|------------------|
| Demos started | 500 | 2,000 |
| Completion rate | 60% | 70% |
| Leads generated | 200 | 800 |
| MQL rate | 25% | 30% |
| Sales meetings booked | 50 | 200 |
| Pipeline generated | $5M | $20M |

---

## Appendix A: Demo Environment URLs

| Environment | URL | Purpose |
|-------------|-----|---------|
| Production | demo.greenlang.com | Customer-facing demos |
| Partner | partners.demo.greenlang.com | Partner certification |
| Sales | sales-demo.greenlang.com | Internal sales training |
| Staging | demo-staging.greenlang.com | Pre-release testing |

---

## Appendix B: Demo Script Library

Demo scripts are stored in `/docs/products/demo/scripts/`:

| Script | Product | Duration | Audience |
|--------|---------|----------|----------|
| thermal_command_executive.md | ThermalCommand | 15 min | C-level |
| thermal_command_engineer.md | ThermalCommand | 45 min | Engineers |
| boiler_optimizer_overview.md | BoilerOptimizer | 20 min | All |
| waste_heat_discovery.md | WasteHeatRecovery | 30 min | Energy managers |
| emissions_compliance.md | EmissionsGuardian | 25 min | Compliance officers |

---

**Document Control:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-05 | Product/Solutions | Initial specification |

---

**END OF DOCUMENT**
