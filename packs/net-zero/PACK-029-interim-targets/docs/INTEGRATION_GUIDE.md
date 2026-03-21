# PACK-029 Interim Targets Pack -- Integration Guide

**Pack ID:** PACK-029-interim-targets
**Version:** 1.0.0
**Last Updated:** 2026-03-19

---

## Table of Contents

1. [Integration Overview](#integration-overview)
2. [PACK-021 Integration](#pack-021-integration)
3. [PACK-028 Integration](#pack-028-integration)
4. [MRV Integration](#mrv-integration)
5. [SBTi Portal Integration](#sbti-portal-integration)
6. [CDP Integration](#cdp-integration)
7. [TCFD Integration](#tcfd-integration)
8. [Initiative Tracker Integration](#initiative-tracker-integration)
9. [Budget System Integration](#budget-system-integration)
10. [Alerting Integration](#alerting-integration)
11. [Assurance Portal Integration](#assurance-portal-integration)
12. [API Authentication and Authorization](#api-authentication-and-authorization)
13. [Rate Limiting and Retry Strategies](#rate-limiting-and-retry-strategies)

---

## Integration Overview

PACK-029 integrates with 10 external systems and platform components:

```
                    +---------------------------+
                    |     PACK-029 Interim       |
                    |     Targets Pack           |
                    +---------------------------+
                    /    |    |    |    |    \
                   /     |    |    |    |     \
            +-----+ +-----+ +---+ +---+ +-----+ +-------+
            |P-021| |P-028| |MRV| |CDP| |TCFD | |SBTi   |
            +-----+ +-----+ +---+ +---+ +-----+ +-------+
               |       |      |     |      |        |
            Baseline  Sector  30   C4.1  Metrics  Annual
            Targets   Paths  Agents C4.2  Targets  Discl.
                                    |
                              +-----+-----+
                              |           |
                           +------+  +-------+
                           |Slack |  |Teams  |
                           +------+  +-------+
                              Alerting
```

### Integration Priority

| Priority | Integration | Reason |
|----------|-------------|--------|
| HIGH | PACK-021 Bridge | Baseline emissions feed -- required for target calculation |
| HIGH | MRV Bridge | Actual emissions for monitoring -- required for quarterly/annual |
| MEDIUM | PACK-028 Bridge | Sector pathways -- enriches variance analysis |
| MEDIUM | CDP Bridge | CDP disclosure -- required for CDP respondents |
| MEDIUM | SBTi Portal Bridge | SBTi submission -- required for SBTi participants |
| MEDIUM | TCFD Bridge | TCFD disclosure -- required for TCFD supporters |
| MEDIUM | Alerting Bridge | Off-track alerts -- strongly recommended |
| LOW | Initiative Tracker | Corrective action tracking -- optional |
| LOW | Budget System | Carbon pricing integration -- optional |
| LOW | Assurance Portal | Third-party verification -- optional |

---

## PACK-021 Integration

### Overview

PACK-021 (Net Zero Starter Pack) provides the baseline emissions data and long-term targets that PACK-029 uses as the foundation for interim target calculation.

### Data Flow

```
PACK-021                          PACK-029
+-------------------+             +-------------------+
| Baseline Engine   | --------->  | Interim Target    |
| - Scope 1 tCO2e  |  baseline   | Engine            |
| - Scope 2 tCO2e  |  data       | - 5-year targets  |
| - Scope 3 tCO2e  |             | - 10-year targets |
| - Base year       |             | - Annual pathway  |
+-------------------+             +-------------------+
| Long-Term Target  | --------->  |                   |
| - Target year     |  target     |                   |
| - Reduction %     |  params     |                   |
| - Net-zero year   |             |                   |
+-------------------+             +-------------------+
```

### Configuration

```bash
# Environment variables
export INTERIM_TARGETS_PACK021_ENABLED="true"
export INTERIM_TARGETS_PACK021_BASE_URL="http://localhost:8021"
export INTERIM_TARGETS_PACK021_API_KEY="pack021-api-key"
```

### Usage

```python
from integrations.pack021_bridge import Pack021Bridge

bridge = Pack021Bridge()

# Import baseline from PACK-021
baseline = await bridge.get_baseline(entity_id="gm-001")
print(f"Baseline Year: {baseline.base_year}")
print(f"Scope 1: {baseline.scope_1_tco2e:,.0f} tCO2e")
print(f"Scope 2: {baseline.scope_2_tco2e:,.0f} tCO2e")
print(f"Scope 3: {baseline.scope_3_tco2e:,.0f} tCO2e")

# Import long-term target from PACK-021
target = await bridge.get_long_term_target(entity_id="gm-001")
print(f"Target Year: {target.target_year}")
print(f"Reduction: {target.reduction_pct}%")

# Full import (baseline + target in one call)
full = await bridge.import_all(entity_id="gm-001")
```

### Fallback (Standalone Mode)

If PACK-021 is not available, provide baseline and target data directly:

```python
from engines.interim_target_engine import InterimTargetInput, BaselineData, LongTermTarget

input_data = InterimTargetInput(
    entity_name="Acme Corp",
    baseline=BaselineData(
        base_year=2021,
        scope_1_tco2e=50_000,
        scope_2_tco2e=30_000,
        scope_3_tco2e=120_000,
    ),
    long_term_target=LongTermTarget(
        target_year=2050,
        reduction_pct=90,
    ),
)
```

---

## PACK-028 Integration

### Overview

PACK-028 (Sector Pathway Pack) provides sector-specific decarbonization pathways and abatement levers that PACK-029 uses to enrich variance analysis and corrective action planning.

### Data Flow

```
PACK-028                          PACK-029
+-------------------+             +-------------------+
| Sector Pathway    | --------->  | Variance Analysis |
| - Sector ID       |  pathway    | - Sector context  |
| - Intensity path  |  data       | - Benchmark gap   |
| - IEA milestones  |             +-------------------+
+-------------------+             +-------------------+
| Abatement Levers  | --------->  | Corrective Action |
| - Lever list      |  levers     | - Initiative scan |
| - Cost curves     |  data       | - MACC optimize   |
| - Timelines       |             +-------------------+
+-------------------+
```

### Configuration

```bash
export INTERIM_TARGETS_PACK028_ENABLED="true"
export INTERIM_TARGETS_PACK028_BASE_URL="http://localhost:8028"
export INTERIM_TARGETS_PACK028_API_KEY="pack028-api-key"
```

### Usage

```python
from integrations.pack028_bridge import Pack028Bridge

bridge = Pack028Bridge()

# Import sector pathway
pathway = await bridge.get_sector_pathway(
    entity_id="gm-001",
    sector="steel",
    scenario="nze_15c",
)

# Import abatement levers for corrective action
levers = await bridge.get_abatement_levers(
    entity_id="gm-001",
    sector="steel",
)
```

---

## MRV Integration

### Overview

The MRV Bridge connects PACK-029 to all 30 MRV agents for actual emissions calculation. This is critical for quarterly monitoring and annual review.

### Agent Routing

| Scope | MRV Agents | Usage |
|-------|-----------|-------|
| Scope 1 | MRV-001 through MRV-008 | Stationary, mobile, process, fugitive, refrigerant, land use, waste, agricultural |
| Scope 2 | MRV-009 through MRV-013 | Location-based, market-based, steam/heat, cooling, dual reporting |
| Scope 3 | MRV-014 through MRV-030 | Categories 1-15 + cross-cutting mapper + audit trail |

### Configuration

```bash
export INTERIM_TARGETS_MRV_ENABLED="true"
export INTERIM_TARGETS_MRV_BASE_URL="http://localhost:8030"
```

### Usage

```python
from integrations.mrv_bridge import MRVBridge

bridge = MRVBridge()

# Get latest emissions for an entity
emissions = await bridge.get_emissions(
    entity_id="gm-001",
    period="2025-Q3",
    scopes=["scope_1", "scope_2", "scope_3"],
)

print(f"Scope 1: {emissions.scope_1_tco2e:,.0f} tCO2e")
print(f"Scope 2: {emissions.scope_2_tco2e:,.0f} tCO2e")
print(f"Scope 3: {emissions.scope_3_tco2e:,.0f} tCO2e")

# Route to specific MRV agent
stationary = await bridge.route_to_agent(
    agent="MRV-001",
    entity_id="gm-001",
    period="2025-Q3",
)
```

---

## SBTi Portal Integration

### Overview

The SBTi Portal Bridge formats interim target data for SBTi submission and annual disclosure.

### Submission Package

The bridge generates a complete SBTi submission package including:

1. **Target Information Form**: Company details, target parameters, scope coverage
2. **Supporting Documentation**: Calculation methodology, data sources, assumptions
3. **Validation Results**: 21-criteria validation report
4. **Annual Disclosure**: Year-over-year progress data

### Configuration

```bash
export INTERIM_TARGETS_SBTI_PORTAL_ENABLED="true"
export INTERIM_TARGETS_SBTI_API_VERSION="v2"
```

### Usage

```python
from integrations.sbti_portal_bridge import SBTiPortalBridge

bridge = SBTiPortalBridge()

# Generate submission package
package = await bridge.generate_submission(
    entity_id="gm-001",
    submission_type="near_term",
    include_validation=True,
)

# Save package
package.save("sbti_submission_gm001.zip")

# Generate annual disclosure
disclosure = await bridge.generate_annual_disclosure(
    entity_id="gm-001",
    reporting_year=2025,
)
```

### SBTi Submission Checklist

| Item | Source | Status |
|------|--------|--------|
| Company profile | Manual input | Required |
| Base year emissions | PACK-021 Bridge | Auto-populated |
| Near-term target | PACK-029 Interim Target Engine | Auto-calculated |
| Long-term target | PACK-021 or manual | Required |
| 21-criteria validation | PACK-029 SBTi Validation Engine | Auto-validated |
| Calculation methodology | PACK-029 provenance | Auto-generated |
| Annual progress data | PACK-029 Annual Review | Auto-populated |

---

## CDP Integration

### Overview

The CDP Bridge generates CDP Climate Change questionnaire responses for sections C4.1 (interim target description) and C4.2 (interim target details).

### CDP Sections Covered

| CDP Section | Content | PACK-029 Source |
|-------------|---------|----------------|
| C4.1 | Text description of interim targets | Interim Target Engine |
| C4.1a | Details of interim targets | Interim Target Engine + SBTi Validation |
| C4.2 | Summary of progress against targets | Annual Review Engine |
| C5.1 | Base year emissions (cross-ref) | PACK-021 Bridge |
| C6.1 | Current year emissions (cross-ref) | MRV Bridge |

### Configuration

```bash
export INTERIM_TARGETS_CDP_ENABLED="true"
export INTERIM_TARGETS_CDP_API_VERSION="2025"
```

### Usage

```python
from integrations.cdp_bridge import CDPBridge

bridge = CDPBridge()

# Generate C4.1 text
c4_1 = await bridge.generate_c4_1(
    entity_id="gm-001",
    reporting_year=2025,
)
print(f"C4.1 Text ({len(c4_1.text)} chars):")
print(c4_1.text[:500])

# Generate C4.2 table rows
c4_2 = await bridge.generate_c4_2(
    entity_id="gm-001",
    reporting_year=2025,
)
for row in c4_2.rows:
    print(f"  Scope: {row.scope}, Target: {row.target_year}, "
          f"Reduction: {row.reduction_pct}%, Status: {row.status}")

# Export full CDP package
export = await bridge.export_full(
    entity_id="gm-001",
    reporting_year=2025,
    format="xlsx",
)
export.save("cdp_c4_export_2025.xlsx")
```

### C4.2 Output Fields

| Field | Description | Source |
|-------|-------------|--------|
| `target_reference_number` | Unique target ID | Auto-generated |
| `year_target_was_set` | Year target was established | InterimTargetResult |
| `target_coverage` | Scopes and categories covered | InterimTargetInput |
| `scope` | GHG Protocol scope | ScopeTimeline |
| `base_year` | Baseline year | BaselineData |
| `base_year_emissions_tco2e` | Baseline emissions | BaselineData |
| `target_year` | Target milestone year | InterimMilestone |
| `target_reduction_pct` | Reduction percentage | InterimMilestone |
| `is_sbti_validated` | SBTi validation status | SBTiValidationResult |
| `progress_against_target` | Annual progress | AnnualReviewResult |

---

## TCFD Integration

### Overview

The TCFD Bridge generates content for the TCFD Metrics and Targets recommendation pillar.

### TCFD Content Generated

| TCFD Disclosure | Content | PACK-029 Source |
|----------------|---------|----------------|
| Scope 1 GHG emissions | Annual Scope 1 emissions | MRV Bridge |
| Scope 2 GHG emissions | Annual Scope 2 emissions (market + location) | MRV Bridge |
| Scope 3 GHG emissions | Annual Scope 3 emissions by category | MRV Bridge |
| Targets against GHG | Interim targets and progress | Interim Target + Annual Review |
| Forward-looking metrics | Trend extrapolation and projections | Trend Extrapolation Engine |
| Transition risks | Gap-to-target as transition risk indicator | Variance Analysis Engine |

### Usage

```python
from integrations.tcfd_bridge import TCFDBridge

bridge = TCFDBridge()

# Generate full TCFD Metrics and Targets section
tcfd = await bridge.generate_metrics_and_targets(
    entity_id="gm-001",
    reporting_year=2025,
)

for section in tcfd.sections:
    print(f"\n{section.title}:")
    print(f"  {section.content[:300]}...")

# Export as formatted document
tcfd.export("tcfd_metrics_targets_2025.html", format="html")
```

---

## Initiative Tracker Integration

### Overview

The Initiative Tracker integration connects corrective action plans to operational project management systems.

### Supported Systems

| System | Integration Type | Status |
|--------|-----------------|--------|
| Internal tracker | REST API | Supported |
| Jira | Webhook | Supported |
| Asana | REST API | Planned v1.1 |
| Microsoft Planner | Graph API | Planned v1.1 |

### Usage

```python
from integrations.initiative_tracker import InitiativeTracker

tracker = InitiativeTracker(system="internal")

# Push corrective action initiatives
await tracker.push_initiatives(
    corrective_action_result=corrective_result,
    assign_to="sustainability-team",
    project="net-zero-2030",
)

# Pull initiative progress
progress = await tracker.pull_progress(
    entity_id="gm-001",
    initiative_ids=["init-001", "init-002"],
)
```

---

## Budget System Integration

### Overview

The Budget System integration connects interim targets to internal carbon pricing and financial planning.

### Internal Carbon Price

PACK-029 can apply an internal carbon price to quantify the financial impact of gap-to-target:

```python
from integrations.budget_bridge import BudgetBridge

bridge = BudgetBridge()

# Calculate financial impact of gap
financial = await bridge.calculate_carbon_cost(
    gap_tco2e=22_000,
    carbon_price_usd_per_tco2e=85,  # Internal carbon price
)

print(f"Gap: {financial.gap_tco2e:,.0f} tCO2e")
print(f"Carbon Price: ${financial.carbon_price:,.0f}/tCO2e")
print(f"Financial Impact: ${financial.total_cost_usd:,.0f}")
```

---

## Alerting Integration

### Overview

The Alerting Bridge sends real-time notifications when monitoring detects off-track performance.

### Supported Channels

| Channel | Configuration | Message Format |
|---------|--------------|----------------|
| Email | SMTP or SendGrid | HTML template |
| Slack | Webhook URL | Slack Block Kit |
| Microsoft Teams | Webhook URL | Adaptive Card |
| PagerDuty | Integration key | PagerDuty event |
| Custom webhook | HTTP POST URL | JSON payload |

### Configuration

```bash
# Email
export INTERIM_TARGETS_ALERT_EMAIL_ENABLED="true"
export INTERIM_TARGETS_ALERT_EMAIL_SMTP_HOST="smtp.company.com"
export INTERIM_TARGETS_ALERT_EMAIL_FROM="alerts@greenlang.io"
export INTERIM_TARGETS_ALERT_EMAIL_TO="sustainability@company.com"

# Slack
export INTERIM_TARGETS_ALERT_SLACK_WEBHOOK="https://hooks.slack.com/services/..."
export INTERIM_TARGETS_ALERT_SLACK_CHANNEL="#net-zero-alerts"

# Microsoft Teams
export INTERIM_TARGETS_ALERT_TEAMS_WEBHOOK="https://outlook.office.com/webhook/..."
```

### Usage

```python
from integrations.alerting_bridge import AlertingBridge

bridge = AlertingBridge()

# Send alert manually
await bridge.send_alert(
    entity_id="gm-001",
    alert_type="off_track_red",
    severity="critical",
    message="Q3 2025 Scope 1+2 emissions 18% above target",
    channels=["email", "slack"],
)

# Configure escalation rules
bridge.set_escalation(
    entity_id="gm-001",
    rules=[
        {"trigger": "off_track_amber", "channels": ["email"], "recipients": ["target_manager"]},
        {"trigger": "off_track_red", "channels": ["email", "slack"], "recipients": ["sustainability_director", "cfo"]},
        {"trigger": "budget_exhaustion", "channels": ["email", "slack", "teams"], "recipients": ["board"]},
    ],
)
```

### Slack Message Format

```json
{
  "blocks": [
    {
      "type": "header",
      "text": {"type": "plain_text", "text": "Net Zero Alert: RED Status"}
    },
    {
      "type": "section",
      "fields": [
        {"type": "mrkdwn", "text": "*Entity:* GlobalManufacturing Inc."},
        {"type": "mrkdwn", "text": "*Quarter:* Q3 2025"},
        {"type": "mrkdwn", "text": "*Status:* :red_circle: RED"},
        {"type": "mrkdwn", "text": "*Variance:* +18.2% above target"},
        {"type": "mrkdwn", "text": "*Scope:* Scope 1+2"},
        {"type": "mrkdwn", "text": "*Action:* Corrective action planning required"}
      ]
    }
  ]
}
```

---

## Assurance Portal Integration

### Overview

The Assurance Portal integration prepares workpapers and evidence packages for third-party verification engagements.

### Supported Standards

| Standard | Coverage | Use Case |
|----------|----------|----------|
| ISO 14064-3 | Full | GHG statement verification |
| ISAE 3410 | Full | Assurance engagement on GHG statements |
| ISAE 3000 | Partial | General assurance on sustainability information |
| AA1000AS | Partial | AccountAbility assurance standard |

### Evidence Package Contents

| Item | Description | Source |
|------|-------------|--------|
| Calculation workpapers | Detailed calculation trace for each target | InterimTargetEngine (provenance hash) |
| Input data summary | Baseline data, assumptions, parameters | InterimTargetInput |
| SBTi validation report | 21-criteria validation results | SBTiValidationEngine |
| Variance analysis | LMDI decomposition with data sources | VarianceAnalysisEngine |
| Progress tracking | Quarterly and annual monitoring data | MonitoringEngine + ReviewEngine |
| Audit trail | Immutable log of all calculations | SEC-005 Audit Logging |
| Provenance hashes | SHA-256 hashes of all outputs | All engines |

### Usage

```python
from integrations.assurance_bridge import AssuranceBridge

bridge = AssuranceBridge()

# Generate assurance package
package = await bridge.generate_evidence_package(
    entity_id="gm-001",
    reporting_year=2025,
    standard="iso_14064_3",
    assurance_level="limited",
)

package.save("assurance_package_2025.zip")
print(f"Package contains {len(package.files)} files")
print(f"Total provenance hashes: {len(package.hashes)}")
```

---

## API Authentication and Authorization

### JWT Authentication

All integration endpoints require JWT Bearer tokens obtained from SEC-001.

```python
import httpx

# Obtain token
response = httpx.post(
    "https://api.greenlang.io/v1/auth/token",
    json={"username": "user@company.com", "password": "..."},
)
token = response.json()["access_token"]

# Use token in requests
headers = {"Authorization": f"Bearer {token}"}
response = httpx.get(
    "https://api.greenlang.io/v1/packs/029/health",
    headers=headers,
)
```

### Service-to-Service Authentication

For pack-to-pack integration (PACK-021, PACK-028), use service accounts:

```python
from integrations.auth import ServiceAuth

auth = ServiceAuth(
    service_id="pack-029",
    service_secret="...",
)
token = await auth.get_service_token()
```

### RBAC Permissions

| Permission | Description | Roles |
|------------|-------------|-------|
| `pack029:engine:execute` | Run engines | target_manager, admin |
| `pack029:workflow:execute` | Run workflows | target_manager, admin |
| `pack029:template:render` | Generate reports | progress_analyst, target_manager, admin |
| `pack029:config:write` | Modify configuration | admin |
| `pack029:read` | Read all data | progress_analyst, target_manager, auditor, admin |
| `pack029:alerts:manage` | Manage alerts | target_manager, admin |
| `pack029:health:read` | Health check | all authenticated |

---

## Rate Limiting and Retry Strategies

### Rate Limits

| Endpoint Type | Limit | Window |
|--------------|-------|--------|
| Engine execution | 60/min | Per API key |
| Workflow execution | 20/min | Per API key |
| Report generation | 30/min | Per API key |
| Health check | 120/min | Per API key |
| Integration bridge | 100/min | Per integration |

### Retry Strategy

```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
async def call_engine(client, endpoint, payload):
    response = await client.post(endpoint, json=payload)
    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 30))
        raise Exception(f"Rate limited, retry after {retry_after}s")
    response.raise_for_status()
    return response.json()
```

### Circuit Breaker

For external integrations (SBTi, CDP), PACK-029 implements a circuit breaker:

| State | Condition | Behavior |
|-------|-----------|----------|
| CLOSED | Normal operation | Requests pass through |
| OPEN | 5 consecutive failures | Requests fail fast for 60s |
| HALF-OPEN | After 60s cooldown | Single test request |

---

**End of Integration Guide**
