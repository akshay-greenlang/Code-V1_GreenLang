# PRD-OBS-005: SLO/SLI Definitions & Error Budget Management Platform

**Component**: OBS-005 - SLO/SLI Definitions & Error Budget Management
**Priority**: P1 - High
**Status**: Approved
**Version**: 1.0
**Date**: 2026-02-07
**Author**: GreenLang Platform Team
**Depends On**: OBS-001 (Prometheus), OBS-002 (Grafana), OBS-003 (Tracing), OBS-004 (Alerting), INFRA-002 (PostgreSQL+TimescaleDB), INFRA-003 (Redis)

---

## 1. Executive Summary

Build a comprehensive SLO/SLI Definitions & Error Budget Management Platform that transforms GreenLang's static SLO YAML definitions into a production-grade, programmatic service. The platform implements **Google SRE Book multi-window burn rate alerting**, real-time error budget tracking, automated Prometheus recording rule generation, Grafana SLO dashboard provisioning, SLO compliance reporting, and a REST API for SLO lifecycle management. It integrates with all existing observability components (OBS-001 through OBS-004) to provide a unified reliability layer for the carbon accounting platform.

## 2. Current State Assessment

### 2.1 Existing Infrastructure (Do NOT Duplicate)

| Component | Location | Lines | Status |
|-----------|----------|-------|--------|
| Static SLO Definitions | `deployment/infrastructure/monitoring/slos/slo_definitions.yaml` | 473 | 16 SLOs defined, no processing engine |
| Prometheus HA Cluster | `deployment/helm/prometheus-stack/` | Config | Fully deployed (OBS-001) |
| Grafana HA Platform | `deployment/helm/grafana/` | Config | 7-folder hierarchy (OBS-002) |
| OpenTelemetry Tracing | `greenlang/infrastructure/tracing/` | ~20K | Full distributed tracing (OBS-003) |
| Unified Alerting Service | `greenlang/infrastructure/alerting_service/` | ~19.4K | 6-channel alerting (OBS-004) |
| Alertmanager HA | `deployment/monitoring/alerts/` | ~8K | 200+ alert rules, 32 files |
| Recording Rules (basic) | `deployment/infrastructure/monitoring/prometheus/recording_rules.yml` | ~100 | Basic aggregation rules only |

### 2.2 Key Gaps

1. **No SLO Processing Engine**: Static YAML with no runtime evaluation
2. **No Error Budget Tracking**: No real-time error budget consumption tracking
3. **No Burn Rate Alerting**: No multi-window burn rate alerts (Google SRE standard)
4. **No Recording Rule Generation**: SLI metrics not computed via Prometheus recording rules
5. **No SLO Dashboard**: No dedicated Grafana SLO overview dashboard
6. **No Error Budget Policies**: No automated actions when budgets are exhausted
7. **No SLO Compliance Reporting**: No weekly/monthly/quarterly SLO compliance reports
8. **No SLO API**: No REST API for SLO CRUD, budget queries, or compliance checks
9. **No SLO-to-Alert Integration**: No bridge between SLO violations and OBS-004 alerting
10. **No SLO Versioning**: No history of SLO target changes over time

## 3. Architecture

### 3.1 High-Level Architecture

```
+-----------------------------------------------------------------------+
|                    Data Sources (SLI Inputs)                            |
|  +----------------+ +----------------+ +----------------+              |
|  | Prometheus     | | Application    | | Database       |              |
|  | Metrics        | | Metrics        | | Metrics        |              |
|  +-------+--------+ +-------+--------+ +-------+--------+              |
+---------|--------------------|-------------------|----------------------+
          |                    |                   |
          v                    v                   v
+-----------------------------------------------------------------------+
|              SLO/SLI Management Service (OBS-005)                      |
|                                                                        |
|  +----------------------------------------------------------------+   |
|  |                   SLO Definition Manager                        |   |
|  |  YAML Parser | DB Persistence | Version History | CRUD API     |   |
|  +-----------------------------+----------------------------------+   |
|                                |                                       |
|  +-----------------------------v----------------------------------+   |
|  |                   SLI Calculator Engine                         |   |
|  |  Availability | Latency | Correctness | Throughput | Freshness |   |
|  |  PromQL Builder | Recording Rule Generator                     |   |
|  +-----------------------------+----------------------------------+   |
|                                |                                       |
|  +-----------------------------v----------------------------------+   |
|  |                   Error Budget Engine                           |   |
|  |  Real-time Budget | Consumption Rate | Forecast | Policies     |   |
|  +-----------------------------+----------------------------------+   |
|                                |                                       |
|  +-----------------------------v----------------------------------+   |
|  |                   Burn Rate Alert Engine                        |   |
|  |  Multi-window (1h/6h/3d) | Fast/Medium/Slow burn detection    |   |
|  |  Google SRE Book methodology | Alert Rule Generator            |   |
|  +-----------------------------+----------------------------------+   |
|                                |                                       |
|  +-----------------------------v----------------------------------+   |
|  |                   Integration Layer                             |   |
|  |  +------------+ +------------+ +------------+ +------------+   |   |
|  |  | Prometheus | | Grafana    | | OBS-004    | | Compliance |   |   |
|  |  | Recording  | | Dashboard  | | Alerting   | | Reporter   |   |   |
|  |  | Rules Gen  | | Generator  | | Bridge     | | Engine     |   |   |
|  |  +------------+ +------------+ +------------+ +------------+   |   |
|  +----------------------------------------------------------------+   |
|                                                                        |
|  +------------------+ +------------------+ +--------------------+      |
|  | REST API         | | Prometheus       | | TimescaleDB        |      |
|  | 20+ endpoints    | | Metrics          | | SLO History        |      |
|  +------------------+ +------------------+ +--------------------+      |
+-----------------------------------------------------------------------+
```

### 3.2 Component Selection

| Component | Technology | Version | Justification |
|-----------|-----------|---------|---------------|
| SLO Storage | PostgreSQL + TimescaleDB | Existing | Already deployed (INFRA-002); time-series native |
| SLI Cache | Redis | Existing | Already deployed (INFRA-003); fast lookups |
| Recording Rules | Prometheus | Existing | Native PromQL recording rules (OBS-001) |
| Dashboards | Grafana | Existing | Dashboard provisioning API (OBS-002) |
| Alerting Bridge | OBS-004 SDK | Existing | Unified alerting (OBS-004) |
| PromQL Engine | prometheus-client | 0.20+ | SLI metric computation |
| YAML Parser | PyYAML + pydantic | Existing | Configuration and validation |
| API Framework | FastAPI | Existing | REST API endpoints |

### 3.3 Integration Strategy

- **Reads** existing `slo_definitions.yaml` — backward-compatible YAML ingestion
- **Generates** Prometheus recording rules — SLI computation at Prometheus level
- **Generates** Prometheus alert rules — multi-window burn rate alerts
- **Generates** Grafana dashboard JSON — SLO overview + per-service dashboards
- **Bridges** to OBS-004 — fires alerts through unified alerting service
- **Stores** history in TimescaleDB — SLO target changes, budget consumption over time
- **Caches** current state in Redis — real-time error budget lookups

## 4. Detailed Component Specifications

### 4.1 SLO/SLI Service SDK

**Location**: `greenlang/infrastructure/slo_service/`

**Files**:

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `__init__.py` | Public API exports | ~120 |
| `config.py` | SLOServiceConfig with env-var overrides | ~220 |
| `models.py` | SLO, SLI, ErrorBudget, BurnRate, SLOReport, SLOWindow models | ~450 |
| `sli_calculator.py` | SLI calculation engine (5 types: availability, latency, correctness, throughput, freshness) | ~380 |
| `slo_manager.py` | SLO CRUD manager with YAML/DB persistence and version history | ~420 |
| `error_budget.py` | Error budget calculator, consumption tracker, forecasting, exhaustion policies | ~350 |
| `burn_rate.py` | Multi-window burn rate engine (Google SRE Book): fast (1h/5m), medium (6h/30m), slow (3d/6h) | ~380 |
| `recording_rules.py` | Prometheus recording rule generator for SLI ratios and error budget metrics | ~320 |
| `alert_rules.py` | Prometheus alert rule generator for multi-window burn rate alerting | ~300 |
| `dashboard_generator.py` | Grafana dashboard JSON generator for SLO overview and per-service views | ~450 |
| `compliance_reporter.py` | SLO compliance reporting engine (weekly/monthly/quarterly) with trend analysis | ~350 |
| `alerting_bridge.py` | Bridge to OBS-004 alerting service for SLO violation notifications | ~200 |
| `metrics.py` | Prometheus metrics for the SLO service itself | ~150 |
| `api/__init__.py` | API init | ~10 |
| `api/router.py` | REST API for SLO management (20+ endpoints) | ~500 |
| `setup.py` | `configure_slo_service(app)` one-liner setup + SLOService facade | ~250 |

**Total**: ~17 files, ~4,850 lines estimated

### 4.2 Configuration

```python
@dataclass
class SLOServiceConfig:
    # Service identification
    service_name: str = "greenlang-slo-service"
    environment: str = "dev"  # dev, staging, prod
    enabled: bool = True

    # SLO definitions source
    slo_definitions_path: str = "deployment/infrastructure/monitoring/slos/slo_definitions.yaml"
    slo_definitions_reload_interval_seconds: int = 300  # Reload YAML every 5 min

    # Database (TimescaleDB)
    database_url: str = ""
    database_pool_min: int = 2
    database_pool_max: int = 10

    # Redis cache
    redis_url: str = ""
    redis_key_prefix: str = "gl:slo:"
    cache_ttl_seconds: int = 60  # Error budget cache TTL

    # Prometheus integration
    prometheus_url: str = "http://prometheus:9090"
    recording_rules_output_path: str = "deployment/monitoring/recording-rules/slo-recording-rules.yaml"
    alert_rules_output_path: str = "deployment/monitoring/alerts/slo-burn-rate-alerts.yaml"

    # Grafana integration
    grafana_url: str = "http://grafana:3000"
    grafana_api_key: str = ""
    dashboard_output_path: str = "deployment/monitoring/dashboards/slo-overview.json"
    dashboard_folder: str = "SLO"

    # Burn rate windows (Google SRE Book defaults)
    burn_rate_fast_long_window: str = "1h"
    burn_rate_fast_short_window: str = "5m"
    burn_rate_fast_threshold: float = 14.4  # Exhausts 30d budget in 2 hours
    burn_rate_medium_long_window: str = "6h"
    burn_rate_medium_short_window: str = "30m"
    burn_rate_medium_threshold: float = 6.0  # Exhausts 30d budget in 1 day
    burn_rate_slow_long_window: str = "3d"
    burn_rate_slow_short_window: str = "6h"
    burn_rate_slow_threshold: float = 1.0  # Exhausts 30d budget in 30 days

    # Error budget policies
    budget_warning_threshold_percent: float = 20.0  # Warn at 20% consumed
    budget_critical_threshold_percent: float = 50.0  # Critical at 50% consumed
    budget_exhausted_threshold_percent: float = 100.0
    budget_exhausted_action: str = "freeze_deployments"  # freeze_deployments, alert_only, none

    # Compliance reporting
    reporting_enabled: bool = True
    reporting_weekly_day: str = "Monday"
    reporting_weekly_time: str = "09:00"
    reporting_monthly_day: int = 1
    reporting_retention_days: int = 365

    # Alerting bridge (OBS-004)
    alerting_bridge_enabled: bool = True

    # Rate limiting
    api_rate_limit_per_minute: int = 120
```

### 4.3 SLO Model

```python
class SLIType(str, Enum):
    AVAILABILITY = "availability"   # Good requests / Total requests
    LATENCY = "latency"             # Requests under threshold / Total requests
    CORRECTNESS = "correctness"     # Correct results / Total results
    THROUGHPUT = "throughput"        # Time above minimum RPS / Total time
    FRESHNESS = "freshness"         # Data within freshness window / Total data

class SLOWindow(str, Enum):
    ROLLING_7D = "7d"
    ROLLING_28D = "28d"
    ROLLING_30D = "30d"
    ROLLING_90D = "90d"
    CALENDAR_MONTH = "calendar_month"
    CALENDAR_QUARTER = "calendar_quarter"

class BurnRateWindow(str, Enum):
    FAST = "fast"      # 1h long, 5m short — 2h budget exhaustion
    MEDIUM = "medium"  # 6h long, 30m short — 1d budget exhaustion
    SLOW = "slow"      # 3d long, 6h short — 30d budget exhaustion

@dataclass
class SLI:
    sli_id: str                         # Unique identifier
    name: str                           # Human-readable name
    type: SLIType                       # SLI type
    good_events_query: str              # PromQL for good events
    total_events_query: str             # PromQL for total events
    threshold: Optional[float] = None   # For latency/throughput types
    unit: str = ""                      # ms, requests/sec, etc.

@dataclass
class ErrorBudget:
    total_budget_minutes: float         # Total budget in the window
    consumed_minutes: float             # Consumed budget
    remaining_minutes: float            # Remaining budget
    consumption_percent: float          # % consumed (0-100)
    remaining_percent: float            # % remaining (0-100)
    burn_rate_1h: float                 # Current 1h burn rate
    burn_rate_6h: float                 # Current 6h burn rate
    burn_rate_3d: float                 # Current 3d burn rate
    forecast_exhaustion_date: Optional[datetime]  # Predicted exhaustion
    status: str                         # "healthy", "warning", "critical", "exhausted"

@dataclass
class SLO:
    slo_id: str                         # UUID
    name: str                           # e.g., "api-availability"
    service: str                        # e.g., "greenlang-api-gateway"
    description: str
    category: str                       # availability, latency, correctness, throughput
    sli: SLI                            # Associated SLI definition
    target: float                       # Target percentage (e.g., 99.95)
    window: SLOWindow                   # Measurement window
    labels: Dict[str, str]              # Prometheus-style labels
    annotations: Dict[str, str]         # Runbook URLs, dashboard links
    owner_team: str                     # Owning team
    tier: str                           # "critical", "standard", "best-effort"
    error_budget: Optional[ErrorBudget] # Current error budget state
    burn_rate_alerts: List[BurnRateAlert]
    created_at: datetime
    updated_at: datetime
    version: int                        # SLO version for history tracking

@dataclass
class BurnRateAlert:
    name: str
    description: str
    window: BurnRateWindow              # fast, medium, slow
    long_window: str                    # e.g., "1h"
    short_window: str                   # e.g., "5m"
    burn_rate_threshold: float          # e.g., 14.4
    severity: str                       # critical, warning, info
    notify_channels: List[str]          # OBS-004 channels to notify

@dataclass
class SLOReport:
    report_id: str
    report_type: str                    # weekly, monthly, quarterly
    period_start: datetime
    period_end: datetime
    slos: List[SLOReportEntry]
    overall_compliance_percent: float
    total_slos: int
    meeting_target: int
    breached: int
    generated_at: datetime

@dataclass
class SLOReportEntry:
    slo_name: str
    service: str
    target: float
    achieved: float
    met_target: bool
    error_budget_consumed_percent: float
    worst_day: Optional[datetime]
    worst_day_value: Optional[float]
    trend: str                          # "improving", "stable", "degrading"
```

### 4.4 SLI Calculator Engine

The SLI Calculator translates SLO definitions into PromQL queries and recording rules:

```python
class SLICalculator:
    """Calculates Service Level Indicators from Prometheus metrics.

    Supports 5 SLI types:
    - Availability: success_rate = good_events / total_events
    - Latency: fast_rate = requests_under_threshold / total_requests
    - Correctness: accuracy_rate = correct_results / total_results
    - Throughput: capacity_rate = time_above_min_rps / total_time
    - Freshness: fresh_rate = fresh_data / total_data

    Generates Prometheus recording rules for efficient SLI computation.
    """
    def calculate_sli(self, slo: SLO, window: str = "5m") -> float
    def calculate_sli_over_window(self, slo: SLO) -> float
    def generate_recording_rule(self, slo: SLO) -> Dict
    def query_prometheus(self, query: str) -> float
    def build_sli_ratio_query(self, slo: SLO, window: str) -> str
    def build_error_rate_query(self, slo: SLO, window: str) -> str
```

### 4.5 Error Budget Engine

```python
class ErrorBudgetEngine:
    """Real-time error budget tracking with forecasting.

    Error Budget = (1 - SLO_target) * window_duration
    Consumed = window_duration * (1 - actual_SLI_ratio)

    Features:
    - Real-time consumption tracking via Prometheus queries
    - Budget forecasting based on current burn rate
    - Budget exhaustion policies (freeze deployments, alert-only)
    - Redis caching for fast budget lookups
    - TimescaleDB history for trend analysis
    """
    async def get_error_budget(self, slo: SLO) -> ErrorBudget
    async def get_budget_consumption_rate(self, slo: SLO, window: str) -> float
    async def forecast_budget_exhaustion(self, slo: SLO) -> Optional[datetime]
    async def check_budget_policy(self, slo: SLO) -> BudgetPolicyAction
    async def record_budget_snapshot(self, slo: SLO, budget: ErrorBudget) -> None
    async def get_budget_history(self, slo_id: str, days: int) -> List[ErrorBudget]
```

### 4.6 Multi-Window Burn Rate Engine

Implements the **Google SRE Book** multi-window, multi-burn-rate alerting methodology:

```python
class BurnRateEngine:
    """Multi-window burn rate alert engine.

    Burn Rate = actual_error_rate / allowed_error_rate

    Windows (Google SRE Book):
    - Fast:   long=1h,  short=5m  → Detects 2-hour budget exhaustion
    - Medium: long=6h,  short=30m → Detects 1-day budget exhaustion
    - Slow:   long=3d,  short=6h  → Detects 30-day budget exhaustion

    A burn rate alert fires when BOTH the long and short windows
    exceed the threshold simultaneously. This reduces false positives
    while maintaining fast detection.

    Thresholds:
    - Fast:   14.4x (= 720/50, exhausts 30d budget in 2h) → Critical
    - Medium: 6.0x  (= 720/120, exhausts in 1d)            → Warning
    - Slow:   1.0x  (= 720/720, exhausts in 30d)           → Info
    """
    def calculate_burn_rate(self, slo: SLO, window: str) -> float
    def check_burn_rate_alert(self, slo: SLO, window: BurnRateWindow) -> bool
    def generate_burn_rate_alert_rules(self, slo: SLO) -> List[Dict]
    def generate_all_burn_rate_rules(self, slos: List[SLO]) -> str
```

### 4.7 Prometheus Recording Rule Generator

```python
class RecordingRuleGenerator:
    """Generates Prometheus recording rules for SLI computation.

    Generated rules follow the naming convention:
    - greenlang:slo:{service}_{sli_type}_{window}:ratio
    - greenlang:slo:{service}_{sli_type}_{window}:error_rate
    - greenlang:slo:{service}_error_budget_{window}:remaining
    - greenlang:slo:{service}_burn_rate_{burn_window}:ratio

    Recording rules are evaluated at Prometheus level for efficiency,
    avoiding expensive range queries at dashboard/alert evaluation time.
    """
    def generate_sli_recording_rules(self, slos: List[SLO]) -> str
    def generate_error_budget_rules(self, slos: List[SLO]) -> str
    def generate_burn_rate_rules(self, slos: List[SLO]) -> str
    def generate_all_rules(self, slos: List[SLO]) -> str
    def write_rules_file(self, slos: List[SLO], output_path: str) -> None
```

### 4.8 Prometheus Alert Rule Generator

```python
class AlertRuleGenerator:
    """Generates multi-window burn rate alert rules.

    Each SLO produces 3 alert rules:
    - {slo_name}_BurnRateFast (critical): 1h/5m windows
    - {slo_name}_BurnRateMedium (warning): 6h/30m windows
    - {slo_name}_BurnRateSlow (info): 3d/6h windows

    Additionally generates:
    - {slo_name}_ErrorBudgetExhausted: Budget at 0%
    - {slo_name}_ErrorBudgetCritical: Budget below 50%
    - {slo_name}_ErrorBudgetWarning: Budget below 80%
    """
    def generate_burn_rate_alerts(self, slo: SLO) -> List[Dict]
    def generate_budget_alerts(self, slo: SLO) -> List[Dict]
    def generate_all_alerts(self, slos: List[SLO]) -> str
    def write_alerts_file(self, slos: List[SLO], output_path: str) -> None
```

### 4.9 Grafana Dashboard Generator

```python
class DashboardGenerator:
    """Generates Grafana dashboard JSON for SLO visibility.

    Dashboards generated:
    1. SLO Overview Dashboard — all SLOs at a glance
    2. Per-Service SLO Dashboard — deep dive per service
    3. Error Budget Dashboard — budget consumption and forecasting
    4. Burn Rate Dashboard — real-time burn rate visualization

    Uses Grafana provisioning API or file-based provisioning.
    """
    def generate_overview_dashboard(self, slos: List[SLO]) -> Dict
    def generate_service_dashboard(self, service: str, slos: List[SLO]) -> Dict
    def generate_error_budget_dashboard(self, slos: List[SLO]) -> Dict
    def generate_burn_rate_dashboard(self, slos: List[SLO]) -> Dict
    def write_dashboards(self, slos: List[SLO], output_dir: str) -> None
```

### 4.10 SLO Compliance Reporter

```python
class ComplianceReporter:
    """SLO compliance reporting engine.

    Report types:
    - Weekly: MTTA/MTTR summary + SLO compliance snapshot
    - Monthly: Full compliance report with trends
    - Quarterly: Executive summary with recommendations

    Reports include:
    - SLO achievement vs target
    - Error budget consumption trend
    - Worst performing SLOs
    - Improving/degrading trend analysis
    - Recommendations for SLO adjustments
    """
    async def generate_weekly_report(self, period_end: datetime) -> SLOReport
    async def generate_monthly_report(self, month: int, year: int) -> SLOReport
    async def generate_quarterly_report(self, quarter: int, year: int) -> SLOReport
    async def get_slo_trend(self, slo_id: str, days: int) -> str
```

### 4.11 Alerting Bridge (OBS-004 Integration)

```python
class AlertingBridge:
    """Bridge between SLO violations and OBS-004 Unified Alerting Service.

    When an SLO burn rate threshold is breached:
    1. Creates an Alert via OBS-004 AlertingService
    2. Routes through OBS-004 escalation policies
    3. Sends notifications via configured channels

    Maps SLO severity to alert severity:
    - Fast burn  → Critical alert
    - Medium burn → Warning alert
    - Slow burn  → Info alert
    - Budget exhausted → Critical alert with deployment freeze
    """
    async def fire_slo_alert(self, slo: SLO, burn_window: BurnRateWindow) -> None
    async def resolve_slo_alert(self, slo: SLO, burn_window: BurnRateWindow) -> None
    async def fire_budget_alert(self, slo: SLO, budget: ErrorBudget) -> None
```

### 4.12 API Endpoints

**Location**: `greenlang/infrastructure/slo_service/api/router.py`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/slos` | List all SLOs (filter by service, team, tier) |
| POST | `/api/v1/slos` | Create a new SLO definition |
| GET | `/api/v1/slos/{id}` | Get SLO details with current error budget |
| PUT | `/api/v1/slos/{id}` | Update SLO definition (creates new version) |
| DELETE | `/api/v1/slos/{id}` | Soft-delete an SLO |
| GET | `/api/v1/slos/{id}/history` | Get SLO version history |
| GET | `/api/v1/slos/{id}/budget` | Get current error budget |
| GET | `/api/v1/slos/{id}/budget/history` | Get error budget history |
| GET | `/api/v1/slos/{id}/burn-rate` | Get current burn rates (all windows) |
| GET | `/api/v1/slos/{id}/compliance` | Get SLO compliance for period |
| POST | `/api/v1/slos/import` | Import SLOs from YAML file |
| POST | `/api/v1/slos/export` | Export SLOs to YAML format |
| GET | `/api/v1/slos/overview` | SLO health overview (all services) |
| GET | `/api/v1/slos/budgets` | Error budget summary (all SLOs) |
| GET | `/api/v1/slos/compliance/report` | Generate compliance report |
| POST | `/api/v1/slos/recording-rules/generate` | Generate Prometheus recording rules |
| POST | `/api/v1/slos/alert-rules/generate` | Generate burn rate alert rules |
| POST | `/api/v1/slos/dashboards/generate` | Generate Grafana dashboards |
| GET | `/api/v1/slos/health` | SLO service health check |
| POST | `/api/v1/slos/evaluate` | Trigger SLO evaluation cycle |

### 4.13 Kubernetes Manifests

**Location**: `deployment/kubernetes/slo-service/`

| File | Description |
|------|-------------|
| `deployment.yaml` | 2-replica deployment with health checks |
| `service.yaml` | ClusterIP service on port 8080 |
| `configmap.yaml` | SLO definitions, recording rules, dashboard templates |
| `hpa.yaml` | HPA (min 2, max 4, CPU 60%) |
| `networkpolicy.yaml` | Ingress from Grafana, Prometheus; egress to Prometheus, PostgreSQL, Redis |
| `servicemonitor.yaml` | Prometheus scrape config |
| `cronjob-reports.yaml` | CronJob for weekly/monthly compliance reports |
| `kustomization.yaml` | Kustomize base |

### 4.14 Database Migration

**Location**: `deployment/database/migrations/sql/V020__slo_service.sql`

```sql
-- SLO definitions table
CREATE TABLE IF NOT EXISTS slo.definitions (
    slo_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    service VARCHAR(100) NOT NULL,
    description TEXT,
    category VARCHAR(50) NOT NULL,
    sli_type VARCHAR(50) NOT NULL,
    sli_good_events_query TEXT NOT NULL,
    sli_total_events_query TEXT NOT NULL,
    sli_threshold DOUBLE PRECISION,
    target DOUBLE PRECISION NOT NULL,
    window VARCHAR(20) NOT NULL DEFAULT '30d',
    labels JSONB NOT NULL DEFAULT '{}',
    annotations JSONB NOT NULL DEFAULT '{}',
    owner_team VARCHAR(100),
    tier VARCHAR(20) NOT NULL DEFAULT 'standard',
    burn_rate_alerts JSONB NOT NULL DEFAULT '[]',
    is_active BOOLEAN NOT NULL DEFAULT true,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- SLO version history
CREATE TABLE IF NOT EXISTS slo.definition_history (
    id BIGSERIAL PRIMARY KEY,
    slo_id UUID NOT NULL REFERENCES slo.definitions(slo_id),
    version INTEGER NOT NULL,
    target DOUBLE PRECISION NOT NULL,
    window VARCHAR(20) NOT NULL,
    change_description TEXT,
    changed_by VARCHAR(100),
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Error budget snapshots (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS slo.error_budget_snapshots (
    snapshot_time TIMESTAMPTZ NOT NULL,
    slo_id UUID NOT NULL REFERENCES slo.definitions(slo_id),
    total_budget_minutes DOUBLE PRECISION NOT NULL,
    consumed_minutes DOUBLE PRECISION NOT NULL,
    remaining_minutes DOUBLE PRECISION NOT NULL,
    consumption_percent DOUBLE PRECISION NOT NULL,
    burn_rate_1h DOUBLE PRECISION,
    burn_rate_6h DOUBLE PRECISION,
    burn_rate_3d DOUBLE PRECISION,
    sli_value DOUBLE PRECISION,
    status VARCHAR(20) NOT NULL DEFAULT 'healthy'
);
SELECT create_hypertable('slo.error_budget_snapshots', 'snapshot_time');

-- SLO compliance reports
CREATE TABLE IF NOT EXISTS slo.compliance_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_type VARCHAR(20) NOT NULL,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    overall_compliance_percent DOUBLE PRECISION,
    total_slos INTEGER,
    meeting_target INTEGER,
    breached INTEGER,
    report_data JSONB NOT NULL DEFAULT '{}',
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- SLO evaluation events
CREATE TABLE IF NOT EXISTS slo.evaluation_log (
    eval_time TIMESTAMPTZ NOT NULL,
    slo_id UUID NOT NULL,
    sli_value DOUBLE PRECISION,
    target DOUBLE PRECISION,
    met_target BOOLEAN,
    burn_rate_fast DOUBLE PRECISION,
    burn_rate_medium DOUBLE PRECISION,
    burn_rate_slow DOUBLE PRECISION,
    error_budget_remaining_percent DOUBLE PRECISION
);
SELECT create_hypertable('slo.evaluation_log', 'eval_time');

-- Continuous aggregate for hourly SLO summaries
CREATE MATERIALIZED VIEW slo.hourly_summaries
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', eval_time) AS bucket,
    slo_id,
    AVG(sli_value) AS avg_sli,
    MIN(sli_value) AS min_sli,
    MAX(sli_value) AS max_sli,
    AVG(error_budget_remaining_percent) AS avg_budget_remaining,
    COUNT(*) FILTER (WHERE met_target) AS met_count,
    COUNT(*) AS total_count
FROM slo.evaluation_log
GROUP BY bucket, slo_id
WITH NO DATA;

-- Indexes
CREATE INDEX idx_slo_definitions_service ON slo.definitions(service);
CREATE INDEX idx_slo_definitions_team ON slo.definitions(owner_team);
CREATE INDEX idx_slo_definitions_active ON slo.definitions(is_active);
CREATE INDEX idx_slo_history_slo_id ON slo.definition_history(slo_id, version);
CREATE INDEX idx_slo_budget_slo_time ON slo.error_budget_snapshots(slo_id, snapshot_time DESC);
CREATE INDEX idx_slo_eval_slo_time ON slo.evaluation_log(slo_id, eval_time DESC);
CREATE INDEX idx_slo_reports_type_period ON slo.compliance_reports(report_type, period_start);

-- Permissions
GRANT USAGE ON SCHEMA slo TO greenlang_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA slo TO greenlang_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA slo TO greenlang_app;
```

### 4.15 Monitoring & Alerting

#### Prometheus Metrics (SLO Service Self-Monitoring)

| Metric | Type | Description |
|--------|------|-------------|
| `gl_slo_evaluations_total` | Counter | Total SLO evaluations by slo_name, result |
| `gl_slo_evaluation_duration_seconds` | Histogram | SLO evaluation latency |
| `gl_slo_error_budget_remaining_percent` | Gauge | Error budget remaining by slo_name |
| `gl_slo_burn_rate` | Gauge | Current burn rate by slo_name, window |
| `gl_slo_definitions_total` | Gauge | Total active SLO definitions |
| `gl_slo_compliance_percent` | Gauge | Overall SLO compliance percentage |
| `gl_slo_alerts_fired_total` | Counter | SLO burn rate alerts fired |
| `gl_slo_recording_rules_generated_total` | Counter | Recording rules generated |
| `gl_slo_budget_snapshots_total` | Counter | Budget snapshots stored |
| `gl_slo_reports_generated_total` | Counter | Compliance reports generated |

#### Grafana Dashboards

**Location**: `deployment/monitoring/dashboards/`

**Dashboard 1: SLO Overview** (`slo-overview.json`) - 24 panels:
1. SLO Compliance Summary (stat)
2. SLOs Meeting Target / Total (stat)
3. Error Budget Overview (gauge grid)
4. SLO Status Table (table - all SLOs with traffic lights)
5. Error Budget Remaining % by SLO (bar chart)
6. Error Budget Consumption Trend (time series)
7. Burn Rate - Fast Window (time series)
8. Burn Rate - Medium Window (time series)
9. Burn Rate - Slow Window (time series)
10. SLI Value Trend by Service (time series)
11. SLOs by Tier (pie chart)
12. SLOs by Category (pie chart)
13. Worst Performing SLOs (table)
14. Best Performing SLOs (table)
15. Budget Exhaustion Forecast (table)
16. SLO Evaluation Rate (time series)
17. SLO History / Version Changes (annotation overlay)
18. Weekly Compliance Trend (time series)
19. Monthly Compliance Trend (bar chart)
20. Error Budget Burn Down Chart (time series)
21. SLO by Team (bar chart)
22. Alert Volume from SLO Violations (time series)
23. Service Reliability Score (gauge)
24. SLO Service Health (stat)

**Dashboard 2: Per-Service SLO** (`slo-service-detail.json`) - 16 panels
**Dashboard 3: Error Budget Deep Dive** (`slo-error-budget.json`) - 12 panels

#### Alert Rules

**Location**: `deployment/monitoring/alerts/slo-service-alerts.yaml`

| Alert | Severity | Condition |
|-------|----------|-----------|
| `SLOServiceDown` | critical | No healthy instances for 5m |
| `SLOEvaluationFailing` | critical | Evaluation errors > 10% for 10m |
| `ErrorBudgetExhausted` | critical | Any SLO budget at 0% |
| `ErrorBudgetCritical` | warning | Any SLO budget below 50% for 1h |
| `ErrorBudgetWarning` | info | Any SLO budget below 80% for 6h |
| `HighBurnRateFast` | critical | Fast burn rate exceeded for 5m |
| `HighBurnRateMedium` | warning | Medium burn rate exceeded for 30m |
| `SLOComplianceBelow95` | warning | Overall compliance below 95% |
| `RecordingRuleGenerationFailed` | warning | Rule generation error |
| `BudgetSnapshotStale` | warning | No snapshots for 15m |

#### Runbooks

**Location**: `docs/runbooks/`

| Runbook | Description |
|---------|-------------|
| `slo-service-down.md` | Diagnose SLO service failures |
| `error-budget-exhausted.md` | Response when error budget hits 0% |
| `high-burn-rate.md` | Investigating and resolving high burn rates |
| `slo-compliance-degraded.md` | Actions when compliance drops below target |

### 4.16 CI/CD Pipeline

**Location**: `.github/workflows/slo-service-ci.yml`

**Jobs**:
1. **lint**: Python linting (ruff/flake8) + YAML validation
2. **test-unit**: Run unit tests with coverage (target: 85%+)
3. **test-integration**: Integration tests with mock Prometheus/Redis
4. **validate-recording-rules**: Validate generated Prometheus rules with promtool
5. **validate-dashboards**: Validate Grafana dashboard JSON schema
6. **validate-alerts**: Validate alert rules with promtool

## 5. Testing Requirements

### 5.1 Unit Tests

**Location**: `tests/unit/slo_service/`

| Test File | Tests (est.) | Coverage |
|-----------|-------------|----------|
| `test_config.py` | 15 | Config defaults, env parsing, validation |
| `test_models.py` | 25 | SLO/SLI/ErrorBudget models, serialization, validation |
| `test_sli_calculator.py` | 20 | SLI calculation for all 5 types, PromQL generation |
| `test_slo_manager.py` | 20 | CRUD operations, versioning, YAML import/export |
| `test_error_budget.py` | 18 | Budget calculation, consumption, forecasting |
| `test_burn_rate.py` | 20 | Multi-window burn rate calculation, alert evaluation |
| `test_recording_rules.py` | 15 | Recording rule generation, YAML output validation |
| `test_alert_rules.py` | 15 | Alert rule generation, PromQL correctness |
| `test_dashboard_generator.py` | 12 | Dashboard JSON generation, panel validation |
| `test_compliance_reporter.py` | 15 | Report generation, trend analysis |
| `test_alerting_bridge.py` | 10 | OBS-004 integration, alert mapping |
| `test_metrics.py` | 8 | Prometheus metric recording |
| `test_api.py` | 25 | All 20 API endpoints |

**Total**: ~218 unit tests

### 5.2 Integration Tests

**Location**: `tests/integration/slo_service/`

| Test File | Tests (est.) | Coverage |
|-----------|-------------|----------|
| `test_end_to_end.py` | 12 | Full SLO lifecycle: define→evaluate→alert→report |
| `test_prometheus_integration.py` | 8 | Recording rule evaluation, query validation |
| `test_database_persistence.py` | 10 | TimescaleDB storage, continuous aggregates |
| `test_redis_caching.py` | 8 | Error budget caching, invalidation |
| `conftest.py` | N/A | Fixtures, mock servers |

**Total**: ~38 integration tests

### 5.3 Load Tests

**Location**: `tests/load/slo_service/`

| Test File | Tests (est.) | Coverage |
|-----------|-------------|----------|
| `test_evaluation_throughput.py` | 5 | 100 SLOs evaluated in < 30s |
| `test_api_throughput.py` | 5 | 500 API requests/sec sustained |

**Total**: ~10 load tests

## 6. Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| SLO definitions supported | 50+ | Configuration count |
| SLI types supported | 5 (availability, latency, correctness, throughput, freshness) | Code review |
| Burn rate windows | 3 (fast/medium/slow) per SLO | Code review |
| Error budget latency | < 500ms for budget query | Prometheus histogram |
| Recording rules accuracy | 100% PromQL correctness | promtool validation |
| Dashboard panels | 52+ across 3 dashboards | JSON count |
| API endpoint count | 20 | Code review |
| Test count | 266+ (218 unit + 38 integration + 10 load) | pytest count |
| Code coverage | 85%+ | pytest-cov report |
| Compliance report accuracy | 100% vs manual calculation | Integration test |

## 7. Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| `pydantic` | ~=2.6 | SLO model validation |
| `httpx` | ~=0.27 | Prometheus/Grafana API client |
| `PyYAML` | ~=6.0 | SLO definition parsing |
| `jinja2` | ~=3.1 | Recording rule/dashboard templates |
| `redis` | ~=5.0 | Error budget caching |

**No new dependencies required** — all packages already in the project.

## 8. Rollout Plan

### Phase 1: Core Engine + Models
- SLO/SLI models and configuration
- SLI calculator for 5 types
- SLO manager with YAML import
- Error budget engine
- Burn rate engine
- Unit tests for core components

### Phase 2: Rule Generation + Dashboards
- Prometheus recording rule generator
- Prometheus alert rule generator (multi-window burn rate)
- Grafana dashboard generator (3 dashboards)
- promtool validation

### Phase 3: API + Integration
- REST API (20 endpoints)
- OBS-004 alerting bridge
- Compliance reporter
- Database migration V020
- Redis caching

### Phase 4: Deployment + Operations
- K8s manifests
- CI/CD pipeline
- Alert rules for SLO service self-monitoring
- Runbook documentation
- Auth integration

---

**Approved By**: Platform Engineering Team
**Review Date**: 2026-02-07
