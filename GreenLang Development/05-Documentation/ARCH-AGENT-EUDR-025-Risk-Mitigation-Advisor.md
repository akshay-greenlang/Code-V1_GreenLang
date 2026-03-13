# AGENT-EUDR-025: Risk Mitigation Advisor -- Technical Architecture Specification

## Document Info

| Field | Value |
|-------|-------|
| **Document ID** | ARCH-AGENT-EUDR-025 |
| **Agent ID** | GL-EUDR-RMA-025 |
| **Component** | Risk Mitigation Advisor Agent |
| **Category** | EUDR Regulatory Agent -- Risk Mitigation & Remediation Intelligence |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Architecture Specification |
| **Author** | GL-AppArchitect |
| **Date** | 2026-03-11 |
| **Regulation** | Regulation (EU) 2023/1115 -- EUDR, Articles 8, 10, 11, 29, 31; ISO 31000:2018 Risk Management; ISO 14001:2015 Environmental Management |
| **Enforcement** | December 30, 2025 (large operators -- ACTIVE); June 30, 2026 (SMEs) |
| **DB Migration** | V113 |
| **Metric Prefix** | `gl_eudr_rma_` |
| **Config Prefix** | `GL_EUDR_RMA_` |
| **API Prefix** | `/v1/eudr-rma` |
| **RBAC Prefix** | `eudr-rma:` |

---

## 1. Executive Summary

### 1.1 Purpose

AGENT-EUDR-025 Risk Mitigation Advisor is a specialized compliance agent providing ML-powered mitigation strategy recommendation, structured remediation plan management, supplier capacity building, a 500+ mitigation measure knowledge base, effectiveness tracking with ROI analysis, continuous monitoring with adaptive management, cost-benefit budget optimization via linear programming, multi-stakeholder collaboration, and audit-ready mitigation documentation for EUDR compliance. It is the 25th agent in the EUDR agent family and establishes the Risk Mitigation and Remediation Intelligence sub-category.

The agent is the operational bridge between the risk assessment layer (EUDR-016 through EUDR-024) and the compliance reporting layer (GL-EUDR-APP DDS generation). It consumes risk signals from all nine upstream risk assessment agents, applies gradient-boosted decision tree recommendation algorithms (XGBoost/LightGBM) with deterministic rule-engine fallback, designs structured remediation plans per ISO 31000:2018, tracks implementation and effectiveness, optimizes mitigation budgets, and generates the mitigation evidence required for Due Diligence Statements under EUDR Articles 10-12.

### 1.2 Regulatory Driver

EUDR Article 11(1) mandates that when risk assessment identifies non-negligible risk, operators shall adopt risk mitigation measures that are "adequate and proportionate" to reduce risk to a negligible level. Article 8(3) requires annual review and update of the due diligence system. Article 12(2)(d) requires DDS to include mitigation measures adopted. Articles 14-16 empower competent authorities to inspect mitigation adequacy. Non-compliance exposes operators to penalties of up to 4% of annual EU turnover under Articles 22-23.

### 1.3 Key Differentiators

- **Nine-agent risk integration** -- consumes risk signals from EUDR-016 through EUDR-024 for holistic mitigation
- **ML-powered with deterministic fallback** -- XGBoost/LightGBM recommendation with SHAP explainability; rule-based fallback when confidence < 0.7
- **500+ proven mitigation measures** -- largest structured, evidence-based library in the EUDR compliance market
- **Closed-loop effectiveness tracking** -- before/after risk scoring with ROI feeds back into recommendation engine
- **Cost-benefit optimization** -- linear programming (PuLP/OR-Tools) for budget allocation, unique in EUDR market
- **ISO 31000:2018 alignment** -- full alignment with international risk management standard
- **Zero-hallucination guarantee** -- all numeric calculations use deterministic Decimal arithmetic; LLM excluded from calculation path

### 1.4 Performance Targets

| Metric | Target |
|--------|--------|
| Strategy recommendation (single supplier) | < 2s (p99) |
| Remediation plan generation | < 5s (p99) |
| Mitigation measure search (500+ measures) | < 500ms (p99) |
| Effectiveness calculation (single plan) | < 3s (p99) |
| Adaptive management event processing | < 5s (p99 event-to-detection) |
| Cost-benefit optimization (500 suppliers) | < 30s (p99) |
| Report generation (single supplier) | < 10s (p99) |
| Batch report generation (100 suppliers) | < 5 minutes |
| API p95 latency (standard queries) | < 200ms |
| Collaboration message delivery | < 5s (p99 real-time) |
| ML model inference throughput | 100 inferences/second |
| Redis cache hit rate target | > 65% |

### 1.5 Development Estimates

| Phase | Scope | Duration | Engineers |
|-------|-------|----------|-----------|
| Phase 1 | Core engines (E1-E4): Strategy Selector, Plan Designer, Capacity Builder, Measure Library | 3 weeks | 2 |
| Phase 2 | Optimization engines (E5-E7): Effectiveness Tracker, Adaptive Management, Cost-Benefit Optimizer | 2 weeks | 2 |
| Phase 3 | Collaboration engine (E8), API routes, auth integration, reporting | 2 weeks | 2 |
| Phase 4 | DB migration V113, Grafana dashboard, integration testing | 1 week | 2 |
| Phase 5 | Testing (unit + integration + performance + ML), security audit | 2 weeks | 2 |
| **Total** | **Complete agent** | **10 weeks** | **2 engineers** |

### 1.6 Estimated Output

- ~45 files (agent code + API + reference data + ML)
- ~42K lines of code
- ~920+ tests
- V113 database migration (10 tables + 4 hypertables)
- 1 Grafana dashboard (18 panels)

---

## 2. System Architecture Overview

### 2.1 Component Diagram

```
+-----------------------------------------------------------------------------------+
|                              GL-EUDR-APP v1.0 (Frontend)                          |
+--------------------------------------+--------------------------------------------+
                                       |
+--------------------------------------v--------------------------------------------+
|                           Unified API Layer (FastAPI)                              |
|                       /v1/eudr-rma  (~35 endpoints)                               |
+---+------+------+------+------+------+------+------+------+-----------------------+
    |      |      |      |      |      |      |      |      |
    v      v      v      v      v      v      v      v      v
+------+ +------+ +------+ +------+ +------+ +------+ +------+ +------+
|Strat | |Plan  | |Capac | |Meas  | |Effect| |Adapt | |CostBn| |Stakeh|
|Select| |Design| |Build | |Libry | |Track | |Mgmt  | |Optim | |Collab|
|(E1)  | |(E2)  | |(E3)  | |(E4)  | |(E5)  | |(E6)  | |(E7)  | |(E8)  |
+--+---+ +--+---+ +--+---+ +--+---+ +--+---+ +--+---+ +--+---+ +--+---+
   |        |        |        |        |        |        |        |
   +--------+--------+--------+--------+--------+--------+--------+
                              |
              +---------------+----------------+
              |               |                |
    +---------v--+   +--------v---+   +--------v-------+
    | PostgreSQL |   |   Redis    |   |  S3 / MinIO    |
    | TimescaleDB|   |  Cache     |   | Evidence Store |
    | (14 tables)|   | ML Models  |   | Report Store   |
    +------------+   +------------+   +----------------+

+-----------------------------------------------------------------------------------+
|                   Upstream Risk Assessment Agent Integrations                      |
|                                                                                   |
| EUDR-016  EUDR-017  EUDR-018  EUDR-019  EUDR-020  EUDR-021  EUDR-022  EUDR-023  |
| Country   Supplier  Commodity Corruption Deforest  Indigen.  Protected Legal      |
| Risk      Risk      Risk      Index      Alert     Rights    Area      Compliance |
|                                                                                   |
|                              EUDR-024                                             |
|                         Third-Party Audit                                         |
+-----------------------------------------------------------------------------------+
```

### 2.2 Engine Interaction Flow

```
                Risk Signals from 9 Agents
              (EUDR-016 through EUDR-024)
                         |
                         v
               +-------------------+
               | E1: Strategy      |-------> Ranked Strategies + SHAP
               |    Selector       |         (mitigation_strategies table)
               +--------+----------+
                        |
            +-----------+-----------+
            v                       v
   +-------------------+   +-------------------+
   | E2: Remediation   |   | E4: Measure       |
   |    Plan Designer  |   |    Library        |
   |                   |   | (500+ measures)   |
   +--------+----------+   +-------------------+
            |
            v
   +-------------------+
   | E3: Capacity      |-------> Enrollment Records
   |    Building Mgr   |         (capacity_building_enrollments)
   +--------+----------+
            |
            v
   +-------------------+
   | E5: Effectiveness |-------> Before/After Scores, ROI
   |    Tracker        |         (effectiveness_records hypertable)
   |                   |-----> Feedback to E1 (model retraining)
   +--------+----------+
            |
            v
   +-------------------+
   | E6: Adaptive      |<------ Event Streams (EUDR-016-024)
   |    Management     |-------> Trigger Events, Adjustments
   |                   |         (trigger_events hypertable)
   +--------+----------+
            |
            v
   +-------------------+
   | E7: Cost-Benefit  |-------> Pareto Frontier, Budget Allocation
   |    Optimizer      |         (optimization_results table)
   +--------+----------+
            |
            v
   +-------------------+
   | E8: Stakeholder   |-------> Messages, Tasks, Supplier Portal
   |    Collaboration  |         (collaboration_messages hypertable)
   +-------------------+
            |
            v
   +-------------------+
   | Reporting Module  |-------> PDF/JSON/HTML/XLSX/XML
   | (mitigation_      |         (mitigation_reports hypertable)
   |  reporter.py)     |         (S3 report store)
   +-------------------+
```

---

## 3. Module Structure

### 3.1 Directory Layout

```
greenlang/agents/eudr/risk_mitigation_advisor/
    __init__.py                          # Public API exports
    config.py                            # GL_EUDR_RMA_ env prefix configuration
    models.py                            # Pydantic v2 models (~40 models)
    metrics.py                           # 18 Prometheus metrics (gl_eudr_rma_ prefix)
    provenance.py                        # SHA-256 provenance tracking
    setup.py                             # RiskMitigationAdvisorService facade
    #
    # === 8 Processing Engines ===
    #
    strategy_selector.py                 # Engine 1: ML-powered recommendation
    remediation_plan_designer.py         # Engine 2: Structured plan generation
    supplier_capacity_builder.py         # Engine 3: Capacity building management
    mitigation_measure_library.py        # Engine 4: 500+ measure catalog
    effectiveness_tracker.py             # Engine 5: Before/after scoring, ROI
    adaptive_management.py               # Engine 6: Continuous monitoring, triggers
    cost_benefit_optimizer.py            # Engine 7: Budget allocation optimization
    stakeholder_collaboration.py         # Engine 8: Multi-party coordination
    #
    # === Reporting Module ===
    #
    mitigation_reporter.py               # Report generation (7 report types, 5 formats)
    #
    # === API Layer ===
    #
    api/
        __init__.py
        router.py                        # Main router (/v1/eudr-rma), sub-router aggregation
        dependencies.py                  # FastAPI dependencies (service injection, auth)
        schemas.py                       # API-specific request/response schemas
        strategy_routes.py               # Strategy recommendation endpoints (5)
        plan_routes.py                   # Remediation plan CRUD endpoints (7)
        capacity_routes.py               # Capacity building endpoints (5)
        library_routes.py                # Mitigation measure library endpoints (4)
        effectiveness_routes.py          # Effectiveness tracking endpoints (4)
        monitoring_routes.py             # Adaptive management endpoints (4)
        optimization_routes.py           # Cost-benefit optimization endpoints (4)
        collaboration_routes.py          # Stakeholder collaboration endpoints (4)
        reporting_routes.py              # Report generation endpoints (4)
        admin_routes.py                  # Admin and health endpoints (2)
    #
    # === Reference Data ===
    #
    reference_data/
        __init__.py
        mitigation_measures.py           # 500+ mitigation measure definitions
        plan_templates.py                # 8 remediation plan templates
        capacity_building_curricula.py   # Training content metadata (7 commodities x 4 tiers)
        risk_category_strategies.py      # Category-specific strategy rules (8 categories)
        trigger_response_matrix.py       # Event-to-adjustment mapping rules
    #
    # === ML Module ===
    #
    ml/
        __init__.py
        model_trainer.py                 # XGBoost/LightGBM training pipeline
        feature_engineering.py           # Feature extraction from 9-agent risk inputs
        model_registry.py               # Model versioning and deployment
        shap_explainer.py               # SHAP-based explainability
        deterministic_fallback.py        # Rule-based decision tree fallback
```

### 3.2 Test Directory Layout

```
tests/agents/eudr/risk_mitigation_advisor/
    __init__.py
    conftest.py                          # Shared fixtures, mock services, test data
    test_strategy_selector.py            # Engine 1: 120+ tests
    test_remediation_plan_designer.py    # Engine 2: 80+ tests
    test_supplier_capacity_builder.py    # Engine 3: 70+ tests
    test_mitigation_measure_library.py   # Engine 4: 60+ tests
    test_effectiveness_tracker.py        # Engine 5: 90+ tests
    test_adaptive_management.py          # Engine 6: 80+ tests
    test_cost_benefit_optimizer.py       # Engine 7: 70+ tests
    test_stakeholder_collaboration.py    # Engine 8: 60+ tests
    test_mitigation_reporter.py          # Reporting: 70+ tests
    test_api_routes.py                   # API layer: 90+ tests
    test_models.py                       # Pydantic models: 40+ tests
    test_config.py                       # Configuration: 15+ tests
    test_provenance.py                   # Provenance chain: 20+ tests
    test_ml_pipeline.py                  # ML model: 30+ tests
    test_golden_scenarios.py             # 35 golden test scenarios
    test_determinism.py                  # 15+ bit-perfect reproducibility tests
    test_performance.py                  # 25+ performance benchmark tests
    test_security.py                     # 10+ security and RBAC tests
```

### 3.3 Deployment Artifacts

```
deployment/
    database/migrations/sql/
        V113__agent_eudr_risk_mitigation_advisor.sql
    monitoring/dashboards/
        eudr-risk-mitigation-advisor.json
```

---

## 4. Engine Specifications

### 4.1 Engine 1: Strategy Selection Engine

**File:** `strategy_selector.py`
**Purpose:** ML-powered recommendation engine that consumes risk inputs from all 9 upstream EUDR agents (016-024) and recommends context-appropriate mitigation strategies ranked by predicted effectiveness.

**Responsibilities:**
- Consume risk signals from 9 upstream agents via REST API calls
- Extract feature vector (45+ features) from multi-dimensional risk profile
- Execute gradient-boosted decision tree inference (XGBoost primary, LightGBM secondary)
- Generate ranked list of top-5 recommended strategies with predicted effectiveness (0-100), cost range (EUR), complexity, and time-to-effect
- Produce SHAP explanations showing which risk factors drove each recommendation
- Implement deterministic fallback mode when ML model confidence < 0.7 threshold
- Support 8 risk categories for strategy recommendation
- Generate composite strategies addressing multiple risk dimensions simultaneously
- Validate strategies against ISO 31000 risk treatment taxonomy (avoid, reduce, share, retain)
- Record all recommendations with full provenance trail (input data hash, model version, parameters, timestamp)
- Cache recent recommendations in Redis (TTL: 1 hour) for identical risk profiles

**Feature Engineering (from 9 agents):**

```python
def extract_features(risk_context: RiskContext) -> Dict[str, Decimal]:
    """
    Extract 45+ features from 9-agent risk inputs.
    All values normalized to 0-100 Decimal scale.
    """
    features = {}
    # EUDR-016: Country Risk (5 features)
    features["country_risk_score"] = risk_context.country.risk_score
    features["country_due_diligence_level"] = DILIGENCE_MAP[risk_context.country.due_diligence_level]
    features["country_governance_index"] = risk_context.country.governance_index
    features["country_enforcement_score"] = risk_context.country.enforcement_score
    features["country_forest_cover_pct"] = risk_context.country.forest_cover_pct

    # EUDR-017: Supplier Risk (6 features)
    features["supplier_risk_score"] = risk_context.supplier.composite_risk
    features["supplier_compliance_history"] = risk_context.supplier.compliance_history_score
    features["supplier_audit_count"] = Decimal(str(risk_context.supplier.audit_count))
    features["supplier_open_cars"] = Decimal(str(risk_context.supplier.open_cars))
    features["supplier_tier_depth"] = Decimal(str(risk_context.supplier.tier_depth))
    features["supplier_volume_pct"] = risk_context.supplier.volume_pct

    # EUDR-018: Commodity Risk (5 features)
    features["commodity_risk_score"] = risk_context.commodity.risk_score
    features["commodity_deforestation_correlation"] = risk_context.commodity.deforestation_correlation
    features["commodity_certification_coverage"] = risk_context.commodity.certification_coverage
    features["commodity_traceability_score"] = risk_context.commodity.traceability_score
    features["commodity_substitution_feasibility"] = risk_context.commodity.substitution_feasibility

    # EUDR-019: Corruption Risk (4 features)
    features["corruption_cpi_score"] = risk_context.corruption.cpi_score
    features["corruption_governance_score"] = risk_context.corruption.governance_score
    features["corruption_transparency_index"] = risk_context.corruption.transparency_index
    features["corruption_alert_count"] = Decimal(str(risk_context.corruption.alert_count))

    # EUDR-020: Deforestation Risk (5 features)
    features["deforestation_alert_severity"] = risk_context.deforestation.max_alert_severity
    features["deforestation_alert_count"] = Decimal(str(risk_context.deforestation.alert_count))
    features["deforestation_proximity_km"] = risk_context.deforestation.min_proximity_km
    features["deforestation_area_ha"] = risk_context.deforestation.total_area_ha
    features["deforestation_post_cutoff"] = Decimal("100") if risk_context.deforestation.post_cutoff else Decimal("0")

    # EUDR-021: Indigenous Rights (4 features)
    features["indigenous_territory_overlap"] = risk_context.indigenous.overlap_score
    features["indigenous_fpic_status"] = FPIC_MAP[risk_context.indigenous.fpic_status]
    features["indigenous_violation_count"] = Decimal(str(risk_context.indigenous.violation_count))
    features["indigenous_community_engagement"] = risk_context.indigenous.engagement_score

    # EUDR-022: Protected Areas (4 features)
    features["protected_area_proximity_km"] = risk_context.protected_area.proximity_km
    features["protected_area_iucn_category"] = IUCN_MAP[risk_context.protected_area.iucn_category]
    features["protected_area_overlap_pct"] = risk_context.protected_area.overlap_pct
    features["protected_area_buffer_violation"] = Decimal("100") if risk_context.protected_area.buffer_violation else Decimal("0")

    # EUDR-023: Legal Compliance (4 features)
    features["legal_compliance_gap_count"] = Decimal(str(risk_context.legal.gap_count))
    features["legal_permit_status"] = PERMIT_MAP[risk_context.legal.permit_status]
    features["legal_certification_validity"] = risk_context.legal.certification_validity_score
    features["legal_regulatory_alignment"] = risk_context.legal.regulatory_alignment_score

    # EUDR-024: Audit (4 features)
    features["audit_nc_critical_count"] = Decimal(str(risk_context.audit.critical_ncs))
    features["audit_nc_major_count"] = Decimal(str(risk_context.audit.major_ncs))
    features["audit_car_overdue_count"] = Decimal(str(risk_context.audit.overdue_cars))
    features["audit_compliance_rate"] = risk_context.audit.compliance_rate

    # Derived composite features (4 features)
    features["composite_risk_score"] = _calculate_composite(features)
    features["risk_dimension_count"] = _count_high_risk_dimensions(features)
    features["max_single_dimension_risk"] = _max_dimension_risk(features)
    features["risk_urgency_score"] = _calculate_urgency(features)

    return features  # 45+ features total
```

**ML Recommendation Pipeline:**

```python
from decimal import Decimal, ROUND_HALF_UP
import xgboost as xgb
import shap

async def recommend_strategies(
    risk_context: RiskContext,
    operator_id: str,
    mode: str = "ml",        # "ml" or "deterministic"
    top_k: int = 5,
) -> List[MitigationStrategy]:
    """
    Returns ranked mitigation strategies. Deterministic mode guaranteed
    bit-perfect across runs. ML mode uses XGBoost with SHAP.
    """
    features = extract_features(risk_context)
    feature_hash = _sha256_features(features)

    # Check cache
    cached = await redis.get(f"rma:strategy:{feature_hash}")
    if cached:
        return _deserialize_strategies(cached)

    if mode == "deterministic" or not _model_available():
        strategies = _deterministic_recommend(features, top_k)
    else:
        # ML inference
        feature_array = _features_to_array(features)
        model = _get_model()  # Loaded from registry
        predictions = model.predict(feature_array)
        confidence = float(predictions[0][1])  # confidence score

        if confidence < 0.7:
            # Fallback to deterministic
            strategies = _deterministic_recommend(features, top_k)
            for s in strategies:
                s.confidence_score = Decimal(str(confidence))
                s.model_version = "deterministic_fallback"
        else:
            # Use ML predictions
            strategy_scores = model.predict_proba(feature_array)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(feature_array)
            strategies = _build_strategies_from_ml(
                strategy_scores, shap_values, features, top_k
            )

    # Provenance
    for s in strategies:
        s.provenance_hash = _compute_provenance(
            feature_hash, s.strategy_id, s.model_version
        )

    # Cache for 1 hour
    await redis.setex(f"rma:strategy:{feature_hash}", 3600, _serialize(strategies))

    return strategies
```

**Deterministic Fallback (Rule-Based Decision Tree):**

```python
def _deterministic_recommend(
    features: Dict[str, Decimal],
    top_k: int,
) -> List[MitigationStrategy]:
    """
    Pure deterministic strategy selection using weighted scoring.
    No ML model involved. Bit-perfect reproducible.
    """
    candidate_strategies = []

    # Rule 1: Critical deforestation -> Emergency response
    if features["deforestation_post_cutoff"] == Decimal("100"):
        candidate_strategies.append(("emergency_deforestation_response", Decimal("95")))

    # Rule 2: High country risk -> Enhanced monitoring + landscape intervention
    if features["country_risk_score"] >= Decimal("70"):
        candidate_strategies.append(("enhanced_country_monitoring", Decimal("80")))
        candidate_strategies.append(("landscape_level_intervention", Decimal("75")))

    # Rule 3: High supplier risk -> Capacity building + corrective action
    if features["supplier_risk_score"] >= Decimal("70"):
        candidate_strategies.append(("supplier_capacity_building", Decimal("80")))
        candidate_strategies.append(("corrective_action_plan", Decimal("78")))

    # Rule 4: Indigenous rights violation -> FPIC remediation
    if features["indigenous_territory_overlap"] >= Decimal("50"):
        candidate_strategies.append(("fpic_remediation", Decimal("85")))

    # Rule 5: Protected area encroachment -> Buffer restoration
    if features["protected_area_buffer_violation"] == Decimal("100"):
        candidate_strategies.append(("buffer_zone_restoration", Decimal("82")))

    # Rule 6: Legal compliance gaps -> Legal gap closure
    if features["legal_compliance_gap_count"] >= Decimal("3"):
        candidate_strategies.append(("legal_gap_closure", Decimal("78")))

    # Rule 7: High corruption -> Anti-corruption measures
    if features["corruption_cpi_score"] >= Decimal("70"):
        candidate_strategies.append(("anti_corruption_measures", Decimal("75")))

    # Rule 8: Audit non-conformances -> CAR-linked remediation
    if features["audit_nc_critical_count"] >= Decimal("1"):
        candidate_strategies.append(("car_linked_remediation", Decimal("85")))

    # Rule 9: Commodity risk -> Certification enrollment
    if features["commodity_certification_coverage"] < Decimal("50"):
        candidate_strategies.append(("certification_enrollment", Decimal("70")))

    # Rule 10: Multi-dimensional risk -> Composite strategy
    if features["risk_dimension_count"] >= Decimal("3"):
        candidate_strategies.append(("composite_multi_risk_strategy", Decimal("88")))

    # Sort by score descending, take top_k
    candidate_strategies.sort(key=lambda x: x[1], reverse=True)
    return [_build_strategy(name, score) for name, score in candidate_strategies[:top_k]]
```

**Upstream Dependencies:**
- EUDR-016: `get_country_risk(country_code)` returns risk_score, due_diligence_level, governance_index
- EUDR-017: `get_supplier_risk(supplier_id)` returns composite_risk, compliance_history, open_cars
- EUDR-018: `get_commodity_risk(commodity, country_code)` returns risk_score, deforestation_correlation
- EUDR-019: `get_corruption_index(country_code)` returns cpi_score, governance_score
- EUDR-020: `get_deforestation_alerts(supplier_id)` returns alert_severity, proximity, post_cutoff
- EUDR-021: `get_indigenous_rights_status(supplier_id)` returns overlap_score, fpic_status
- EUDR-022: `get_protected_area_status(supplier_id)` returns proximity, overlap_pct, buffer_violation
- EUDR-023: `get_legal_compliance(supplier_id)` returns gap_count, permit_status
- EUDR-024: `get_audit_status(supplier_id)` returns critical_ncs, major_ncs, overdue_cars

**Zero-Hallucination Guarantee:** ML model is used ONLY for ranking and scoring strategy candidates that are pre-defined in the Measure Library (Engine 4). The ML model never generates strategy text or numeric values. Deterministic fallback uses fixed Decimal weights and thresholds. All provenance recorded with SHA-256 hashes.

**Estimated Lines of Code:** 4,500-5,000

---

### 4.2 Engine 2: Remediation Plan Design Engine

**File:** `remediation_plan_designer.py`
**Purpose:** Generates structured, multi-phase remediation plans with SMART milestones, Gantt chart data, dependency tracking, and plan versioning based on Strategy Selector recommendations.

**Responsibilities:**
- Generate multi-phase remediation plans from selected strategies
- Create SMART milestones: Specific (linked to risk factor), Measurable (quantified KPI), Achievable (validated against resources), Relevant (mapped to EUDR article), Time-bound (deadline date)
- Support 4 plan phases: Preparation (weeks 1-2), Implementation (weeks 3-8), Verification (weeks 9-10), Monitoring (ongoing)
- Provide 8 plan templates for common remediation scenarios
- Generate Gantt chart data with dependency tracking and critical path analysis
- Track milestone completion with evidence upload requirements
- Support plan versioning with change history and approval workflows
- Manage plan status lifecycle: Draft -> Active -> On Track / At Risk / Delayed -> Completed / Suspended / Abandoned
- Link each plan element to specific EUDR article requirements
- Support plan cloning for applying patterns to similar suppliers

**Plan Template Library (8 Templates, Pre-Coded):**

```python
PLAN_TEMPLATES = {
    "supplier_capacity_building": PlanTemplate(
        name="Supplier Capacity Building",
        phases=["Assessment", "Training", "Practice Change", "Verification"],
        duration_weeks_range=(12, 24),
        risk_categories=["supplier", "commodity"],
        milestone_count=8,
        default_kpis=["competency_score", "module_completion_rate", "risk_score_delta"],
    ),
    "emergency_deforestation_response": PlanTemplate(
        name="Emergency Deforestation Response",
        phases=["Suspend", "Investigate", "Remediate", "Resume"],
        duration_weeks_range=(2, 8),
        risk_categories=["deforestation"],
        milestone_count=6,
        default_kpis=["days_to_suspension", "investigation_completeness", "restoration_area_ha"],
    ),
    "certification_enrollment": PlanTemplate(
        name="Certification Enrollment",
        phases=["Gap Assessment", "Preparation", "Audit", "Certification"],
        duration_weeks_range=(24, 52),
        risk_categories=["legal", "commodity"],
        milestone_count=10,
        default_kpis=["gap_closure_rate", "audit_readiness_score", "certification_status"],
    ),
    "enhanced_monitoring_deployment": PlanTemplate(
        name="Enhanced Monitoring Deployment",
        phases=["Baseline", "Deploy", "Calibrate", "Operate"],
        duration_weeks_range=(4, 8),
        risk_categories=["country", "deforestation"],
        milestone_count=6,
        default_kpis=["monitoring_coverage_pct", "alert_response_time", "detection_accuracy"],
    ),
    "fpic_remediation": PlanTemplate(
        name="FPIC Remediation",
        phases=["Identify", "Engage", "Consult", "Agree", "Monitor"],
        duration_weeks_range=(16, 36),
        risk_categories=["indigenous_rights"],
        milestone_count=10,
        default_kpis=["community_meetings_held", "fpic_agreement_status", "grievance_resolution_rate"],
    ),
    "legal_gap_closure": PlanTemplate(
        name="Legal Gap Closure",
        phases=["Assessment", "Legal Support", "Permit Acquisition", "Verification"],
        duration_weeks_range=(8, 24),
        risk_categories=["legal_compliance"],
        milestone_count=8,
        default_kpis=["permits_acquired", "compliance_gap_count", "legal_readiness_score"],
    ),
    "anti_corruption_measures": PlanTemplate(
        name="Anti-Corruption Measures",
        phases=["Assessment", "Controls", "Training", "Monitoring"],
        duration_weeks_range=(8, 16),
        risk_categories=["corruption"],
        milestone_count=8,
        default_kpis=["control_implementation_rate", "training_completion_pct", "transparency_score"],
    ),
    "buffer_zone_restoration": PlanTemplate(
        name="Protected Area Buffer Restoration",
        phases=["Assessment", "Planning", "Restoration", "Monitoring"],
        duration_weeks_range=(24, 52),
        risk_categories=["protected_areas"],
        milestone_count=8,
        default_kpis=["restoration_area_ha", "encroachment_incidents", "buffer_compliance_pct"],
    ),
}
```

**Plan Generation Algorithm (Deterministic):**

```python
from decimal import Decimal, ROUND_HALF_UP
from datetime import date, timedelta

def generate_remediation_plan(
    strategies: List[MitigationStrategy],
    supplier_context: SupplierContext,
    start_date: date,
    budget_allocated: Decimal,
) -> RemediationPlan:
    """Deterministic plan generation from selected strategies."""
    template = _select_template(strategies)
    phases = []
    milestones = []
    current_date = start_date

    for i, phase_name in enumerate(template.phases):
        phase_duration = _calculate_phase_duration(
            template.duration_weeks_range, len(template.phases), i
        )
        phase_end = current_date + timedelta(weeks=phase_duration)
        phases.append(PlanPhase(
            phase_id=f"P{i+1}",
            name=phase_name,
            start_date=current_date,
            end_date=phase_end,
            status="pending",
        ))

        # Generate SMART milestones for this phase
        phase_milestones = _generate_smart_milestones(
            phase_name, current_date, phase_end,
            supplier_context, strategies, template.default_kpis
        )
        milestones.extend(phase_milestones)
        current_date = phase_end

    plan = RemediationPlan(
        operator_id=supplier_context.operator_id,
        supplier_id=supplier_context.supplier_id,
        plan_name=f"{template.name} - {supplier_context.supplier_name}",
        risk_finding_ids=[s.strategy_id for s in strategies],
        strategy_ids=[s.strategy_id for s in strategies],
        status="draft",
        phases=[p.model_dump() for p in phases],
        budget_allocated=budget_allocated,
        start_date=start_date,
        target_end_date=current_date,
        plan_template=template.name,
        version=1,
    )
    return plan
```

**Zero-Hallucination Guarantee:** Plan generation uses pre-coded templates and deterministic date arithmetic. Milestone names and KPI definitions come from template reference data. No LLM used in plan structure generation. Duration calculations use integer week arithmetic.

**Estimated Lines of Code:** 3,500-4,000

---

### 4.3 Engine 3: Capacity Building Manager Engine

**File:** `supplier_capacity_builder.py`
**Purpose:** Designs and manages supplier development programs with 4 capacity building tiers, commodity-specific training modules, and progress tracking.

**Responsibilities:**
- Manage 4 capacity building tiers: Tier 1 (Awareness), Tier 2 (Basic Compliance), Tier 3 (Advanced Practices), Tier 4 (Leadership)
- Provide commodity-specific training modules for 7 EUDR commodities (22 modules each)
- Track individual supplier progress through tiers with competency assessments at each gate
- Allocate technical assistance resources (field trainers, agronomists, GIS specialists, legal advisors)
- Manage resource scheduling and availability across programs
- Generate capacity building scorecards per supplier
- Support group training sessions with attendance tracking
- Integrate with EUDR-017 to correlate capacity building progress with risk score improvement
- Calculate tier advancement eligibility using deterministic scoring

**Tier Advancement Scoring (Deterministic):**

```python
def calculate_tier_advancement_eligibility(
    enrollment: CapacityBuildingEnrollment,
    competency_scores: Dict[str, Decimal],
    modules_completed: int,
    modules_total: int,
) -> TierAdvancementResult:
    """Deterministic tier gate assessment."""
    GATE_THRESHOLDS = {
        1: {"min_competency": Decimal("60"), "min_completion_pct": Decimal("80")},
        2: {"min_competency": Decimal("70"), "min_completion_pct": Decimal("85")},
        3: {"min_competency": Decimal("80"), "min_completion_pct": Decimal("90")},
        4: {"min_competency": Decimal("90"), "min_completion_pct": Decimal("95")},
    }

    current_tier = enrollment.current_tier
    if current_tier >= 4:
        return TierAdvancementResult(eligible=False, reason="Already at maximum tier")

    next_tier = current_tier + 1
    threshold = GATE_THRESHOLDS[current_tier]

    # Calculate average competency
    if competency_scores:
        avg_competency = sum(competency_scores.values()) / Decimal(str(len(competency_scores)))
        avg_competency = avg_competency.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    else:
        avg_competency = Decimal("0")

    # Calculate completion percentage
    completion_pct = (Decimal(str(modules_completed)) / Decimal(str(modules_total)) * Decimal("100")
                      ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    eligible = (
        avg_competency >= threshold["min_competency"]
        and completion_pct >= threshold["min_completion_pct"]
    )

    return TierAdvancementResult(
        eligible=eligible,
        current_tier=current_tier,
        next_tier=next_tier if eligible else current_tier,
        avg_competency=avg_competency,
        completion_pct=completion_pct,
        threshold=threshold,
    )
```

**Upstream Dependencies:**
- EUDR-017 Supplier Risk Scorer: supplier risk profiles and risk score changes post-enrollment

**Zero-Hallucination Guarantee:** Tier advancement uses deterministic Decimal thresholds. Module completion is integer counting. Competency averaging uses Decimal arithmetic with explicit rounding. No LLM in assessment path.

**Estimated Lines of Code:** 3,200-3,800

---

### 4.4 Engine 4: Measure Library Engine

**File:** `mitigation_measure_library.py`
**Purpose:** Searchable knowledge base of 500+ proven mitigation measures organized across 8 risk categories with full-text search, faceted filtering, and measure comparison.

**Responsibilities:**
- Store and serve 500+ mitigation measures across 8 risk categories
- Implement full-text search with relevance ranking via PostgreSQL GIN index
- Support faceted filtering by risk category, commodity, country, cost range, complexity, effectiveness, ISO 31000 type
- Provide measure detail view with effectiveness evidence from case studies and research
- Support measure comparison (side-by-side for 2-5 measures)
- Maintain version-controlled measure data with update history
- Tag measures with EUDR articles and certification scheme requirements
- Generate recommended measure packages grouped by risk scenario
- Support community-contributed measures with review workflow

**Measure Category Distribution:**

| Risk Category | Count | Coverage |
|---------------|-------|----------|
| Country Risk | 65+ | Landscape programs, multi-stakeholder initiatives, government partnerships |
| Supplier Risk | 80+ | Training, corrective actions, scorecards, site visits, alternative sourcing |
| Commodity Risk | 75+ | Certification, sustainable production, yield improvement, intercropping |
| Corruption Risk | 55+ | Third-party verification, enhanced due diligence, anti-bribery training |
| Deforestation Risk | 70+ | Satellite monitoring, zero-deforestation commitments, restoration, fire prevention |
| Indigenous Rights | 50+ | FPIC implementation, benefit-sharing, grievance mechanisms, land rights support |
| Protected Areas | 55+ | Buffer zone restoration, community conservation, alternative livelihoods |
| Legal Compliance | 60+ | Permit acquisition, environmental impact assessment, labour compliance |

**Full-Text Search Implementation:**

```python
async def search_measures(
    query: Optional[str] = None,
    risk_category: Optional[str] = None,
    commodity: Optional[str] = None,
    country_code: Optional[str] = None,
    cost_max: Optional[Decimal] = None,
    complexity: Optional[str] = None,
    iso_31000_type: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> MeasureSearchResult:
    """
    Full-text search with faceted filtering.
    Uses PostgreSQL GIN index for sub-500ms response.
    """
    base_query = """
        SELECT *, ts_rank(
            to_tsvector('english', name || ' ' || description),
            plainto_tsquery('english', %(query)s)
        ) AS relevance
        FROM eudr_risk_mitigation_advisor.mitigation_measures
        WHERE is_active = TRUE
    """
    params = {"query": query or ""}
    conditions = []

    if query:
        conditions.append(
            "to_tsvector('english', name || ' ' || description) "
            "@@ plainto_tsquery('english', %(query)s)"
        )
    if risk_category:
        conditions.append("risk_category = %(risk_category)s")
        params["risk_category"] = risk_category
    if commodity:
        conditions.append("applicability->'commodities' ? %(commodity)s")
        params["commodity"] = commodity
    if country_code:
        conditions.append("applicability->'countries' ? %(country_code)s")
        params["country_code"] = country_code
    if cost_max is not None:
        conditions.append("cost_estimate_min <= %(cost_max)s")
        params["cost_max"] = float(cost_max)
    if complexity:
        conditions.append("implementation_complexity = %(complexity)s")
        params["complexity"] = complexity
    if iso_31000_type:
        conditions.append("iso_31000_type = %(iso_31000_type)s")
        params["iso_31000_type"] = iso_31000_type

    if conditions:
        base_query += " AND " + " AND ".join(conditions)

    order = "relevance DESC" if query else "effectiveness_rating DESC"
    base_query += f" ORDER BY {order} LIMIT %(limit)s OFFSET %(offset)s"
    params["limit"] = limit
    params["offset"] = offset

    rows = await db.fetch(base_query, params)
    return MeasureSearchResult(measures=rows, total=len(rows))
```

**Zero-Hallucination Guarantee:** All 500+ measures are pre-authored reference data with effectiveness evidence from published sources. No LLM generates measure content. Search uses PostgreSQL full-text search, not semantic/LLM search. Relevance ranking is deterministic (ts_rank).

**Estimated Lines of Code:** 3,000-3,500

---

### 4.5 Engine 5: Effectiveness Tracking Engine

**File:** `effectiveness_tracker.py`
**Purpose:** Measures mitigation impact through before/after risk scoring, ROI analysis, trend detection, statistical significance testing, and feedback to the Strategy Selector for continuous improvement.

**Responsibilities:**
- Capture baseline risk scores from all 9 upstream agents at plan activation (T0 snapshot)
- Capture periodic risk score updates at configurable intervals (default: monthly)
- Calculate risk reduction delta per dimension using Decimal arithmetic
- Calculate composite risk reduction using weighted aggregation
- Perform ROI analysis: ROI = (Risk Reduction Value - Mitigation Cost) / Mitigation Cost * 100
- Track predicted vs. actual risk reduction with deviation analysis
- Implement statistical significance testing (paired t-test via SciPy) with configurable confidence (default: 95%)
- Generate effectiveness trend charts data
- Identify underperforming measures (actual < 50% of predicted) and flag for adjustment
- Feed effectiveness outcomes back to Strategy Selector ML model

**Risk Reduction Calculation (Deterministic):**

```python
from decimal import Decimal, ROUND_HALF_UP

def calculate_risk_reduction(
    baseline_scores: Dict[str, Decimal],
    current_scores: Dict[str, Decimal],
    weights: Dict[str, Decimal],
) -> EffectivenessResult:
    """
    All arithmetic uses Decimal. Bit-perfect reproducible.
    """
    dimension_reductions = {}
    weighted_sum = Decimal("0")
    weight_total = Decimal("0")

    for dimension, baseline in baseline_scores.items():
        current = current_scores.get(dimension, baseline)
        if baseline > Decimal("0"):
            reduction = ((baseline - current) / baseline * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            reduction = Decimal("0.00")
        dimension_reductions[dimension] = reduction

        w = weights.get(dimension, Decimal("1"))
        weighted_sum += reduction * w
        weight_total += w

    composite = (weighted_sum / weight_total).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ) if weight_total > Decimal("0") else Decimal("0.00")

    return EffectivenessResult(
        dimension_reductions=dimension_reductions,
        composite_reduction=composite,
    )


def calculate_roi(
    risk_reduction_value: Decimal,
    mitigation_cost: Decimal,
) -> Decimal:
    """
    ROI = (Risk Reduction Value - Mitigation Cost) / Mitigation Cost * 100
    Risk Reduction Value = Risk Score Reduction * Penalty Exposure * Probability Factor
    """
    if mitigation_cost <= Decimal("0"):
        return Decimal("0.00")
    roi = ((risk_reduction_value - mitigation_cost) / mitigation_cost * Decimal("100")).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    return roi
```

**Zero-Hallucination Guarantee:** All calculations use Python Decimal with explicit ROUND_HALF_UP. No floating-point arithmetic. Statistical significance testing uses SciPy (deterministic given same inputs). ROI formula is fixed and documented.

**Estimated Lines of Code:** 3,500-4,000

---

### 4.6 Engine 6: Continuous Monitoring Engine

**File:** `adaptive_management.py`
**Purpose:** Monitors active mitigation plans against real-time risk signals from all 9 upstream agents, detects trigger events, and recommends adaptive adjustments.

**Responsibilities:**
- Subscribe to event streams from 9 upstream risk agents via Redis Streams
- Detect 6 trigger event types requiring plan adjustment
- Generate adaptive adjustment recommendations within defined SLAs
- Support 5 adjustment types: Plan Acceleration, Scope Expansion, Strategy Replacement, Emergency Response, Plan De-escalation
- Implement alert fatigue prevention (consolidation, configurable quiet periods)
- Track plan drift metric (planned vs. actual risk trajectory)
- Automate annual due diligence system review per Article 8(3)
- Support configurable escalation chains for unacknowledged recommendations
- Record all events and decisions with provenance trail

**Trigger Event Response Matrix (Pre-Coded Reference Data):**

```python
TRIGGER_RESPONSE_MATRIX = {
    "critical_deforestation_alert": TriggerConfig(
        source_agent="EUDR-020",
        severity="critical",
        response_sla_hours=4,
        default_adjustment="emergency_response",
        actions=["suspend_sourcing", "notify_supplier", "deploy_monitoring", "commission_investigation"],
    ),
    "country_reclassification_high": TriggerConfig(
        source_agent="EUDR-016",
        severity="high",
        response_sla_hours=48,
        default_adjustment="scope_expansion",
        actions=["activate_enhanced_due_diligence", "review_all_supplier_plans"],
    ),
    "supplier_risk_spike_50pct": TriggerConfig(
        source_agent="EUDR-017",
        severity="high",
        response_sla_hours=24,
        default_adjustment="plan_acceleration",
        actions=["accelerate_timeline", "add_measures", "increase_monitoring"],
    ),
    "supplier_risk_spike_20pct": TriggerConfig(
        source_agent="EUDR-017",
        severity="medium",
        response_sla_hours=48,
        default_adjustment="scope_expansion",
        actions=["add_measures", "schedule_review"],
    ),
    "audit_critical_nc": TriggerConfig(
        source_agent="EUDR-024",
        severity="high",
        response_sla_hours=24,
        default_adjustment="plan_acceleration",
        actions=["link_car_to_plan", "accelerate_remediation", "increase_verification"],
    ),
    "indigenous_rights_violation": TriggerConfig(
        source_agent="EUDR-021",
        severity="high",
        response_sla_hours=24,
        default_adjustment="scope_expansion",
        actions=["activate_fpic_remediation", "engage_community", "suspend_if_severe"],
    ),
    "protected_area_encroachment": TriggerConfig(
        source_agent="EUDR-022",
        severity="high",
        response_sla_hours=24,
        default_adjustment="scope_expansion",
        actions=["activate_buffer_intervention", "deploy_monitoring", "engage_authorities"],
    ),
    "risk_improvement_30pct": TriggerConfig(
        source_agent="any",
        severity="low",
        response_sla_hours=168,  # 1 week
        default_adjustment="plan_de_escalation",
        actions=["assess_de_escalation", "reduce_monitoring_frequency"],
    ),
}
```

**Event Processing Pipeline:**

```python
async def process_risk_event(event: RiskEvent) -> Optional[TriggerEvent]:
    """
    Process incoming risk event from upstream agent.
    Deterministic trigger detection based on pre-coded rules.
    """
    # Match event to trigger rules
    trigger_config = _match_trigger(event)
    if trigger_config is None:
        return None  # Event does not match any trigger rule

    # Check alert fatigue: suppress if similar trigger within quiet period
    if await _is_within_quiet_period(event.plan_id, trigger_config.source_agent):
        return None

    # Find affected plans
    affected_plans = await _find_affected_plans(event)

    for plan in affected_plans:
        trigger = TriggerEvent(
            plan_id=plan.plan_id,
            trigger_type=event.event_type,
            source_agent=event.source_agent,
            severity=trigger_config.severity,
            description=_build_trigger_description(event, trigger_config),
            risk_data=event.risk_data,
            recommended_adjustment={
                "type": trigger_config.default_adjustment,
                "actions": trigger_config.actions,
                "sla_hours": trigger_config.response_sla_hours,
            },
            adjustment_type=trigger_config.default_adjustment,
        )
        await _store_trigger(trigger)
        await _notify_escalation_chain(trigger, plan)

    return trigger
```

**Zero-Hallucination Guarantee:** Trigger detection uses deterministic rule matching against pre-coded response matrix. SLA calculations use integer hour arithmetic. No LLM in event processing or adjustment recommendation.

**Estimated Lines of Code:** 3,800-4,200

---

### 4.7 Engine 7: Cost-Benefit Optimizer Engine

**File:** `cost_benefit_optimizer.py`
**Purpose:** Budget allocation engine that maximizes aggregate risk reduction subject to budget constraints using linear programming (PuLP/OR-Tools) and portfolio optimization.

**Responsibilities:**
- Accept budget constraints (total, per-supplier caps, per-category limits)
- Calculate cost-effectiveness ratio for each candidate mitigation measure
- Implement linear programming optimization via PuLP/OR-Tools
- Generate Pareto-optimal frontier showing budget vs. risk reduction trade-offs
- Support multi-scenario "what-if" analysis
- Provide sensitivity analysis identifying highest-impact budget decisions
- Prioritize investments using RICE framework (Reach, Impact, Confidence, Effort)
- Generate quarterly budget allocation recommendations
- Track actual spend vs. planned allocation with variance reporting
- Support multi-year budget planning with projections

**Linear Programming Model (Deterministic):**

```python
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD

def optimize_budget_allocation(
    suppliers: List[SupplierMitigationContext],
    candidate_measures: Dict[str, List[MeasureCost]],
    total_budget: Decimal,
    per_supplier_cap: Optional[Decimal] = None,
    category_budgets: Optional[Dict[str, Decimal]] = None,
) -> OptimizationResult:
    """
    Deterministic LP optimization. PuLP CBC solver is deterministic
    given identical inputs. All monetary values in EUR as Decimal.
    """
    prob = LpProblem("mitigation_budget_optimization", LpMaximize)

    # Decision variables: x[supplier][measure] = 1 if measure selected
    x = {}
    for supplier in suppliers:
        x[supplier.supplier_id] = {}
        for measure in candidate_measures.get(supplier.supplier_id, []):
            var_name = f"x_{supplier.supplier_id}_{measure.measure_id}"
            x[supplier.supplier_id][measure.measure_id] = LpVariable(
                var_name, cat="Binary"
            )

    # Objective: maximize weighted risk reduction
    objective_terms = []
    for supplier in suppliers:
        weight = float(supplier.risk_level_weight * supplier.volume_weight)
        for measure in candidate_measures.get(supplier.supplier_id, []):
            var = x[supplier.supplier_id][measure.measure_id]
            risk_reduction = float(measure.expected_risk_reduction)
            objective_terms.append(var * risk_reduction * weight)
    prob += lpSum(objective_terms)

    # Constraint 1: Total budget
    cost_terms = []
    for supplier in suppliers:
        for measure in candidate_measures.get(supplier.supplier_id, []):
            var = x[supplier.supplier_id][measure.measure_id]
            cost_terms.append(var * float(measure.cost_eur))
    prob += lpSum(cost_terms) <= float(total_budget), "total_budget"

    # Constraint 2: Per-supplier cap
    if per_supplier_cap:
        for supplier in suppliers:
            supplier_costs = []
            for measure in candidate_measures.get(supplier.supplier_id, []):
                var = x[supplier.supplier_id][measure.measure_id]
                supplier_costs.append(var * float(measure.cost_eur))
            prob += lpSum(supplier_costs) <= float(per_supplier_cap), f"cap_{supplier.supplier_id}"

    # Constraint 3: Category budgets
    if category_budgets:
        for category, cat_budget in category_budgets.items():
            cat_costs = []
            for supplier in suppliers:
                for measure in candidate_measures.get(supplier.supplier_id, []):
                    if measure.risk_category == category:
                        var = x[supplier.supplier_id][measure.measure_id]
                        cat_costs.append(var * float(measure.cost_eur))
            if cat_costs:
                prob += lpSum(cat_costs) <= float(cat_budget), f"cat_{category}"

    # Solve (deterministic CBC solver, no randomness)
    solver = PULP_CBC_CMD(msg=0, timeLimit=30)
    prob.solve(solver)

    # Extract results
    allocations = []
    total_cost = Decimal("0")
    total_reduction = Decimal("0")
    for supplier in suppliers:
        for measure in candidate_measures.get(supplier.supplier_id, []):
            var = x[supplier.supplier_id][measure.measure_id]
            if var.varValue and var.varValue > 0.5:
                allocations.append(BudgetAllocation(
                    supplier_id=supplier.supplier_id,
                    measure_id=measure.measure_id,
                    cost_eur=measure.cost_eur,
                    expected_risk_reduction=measure.expected_risk_reduction,
                ))
                total_cost += measure.cost_eur
                total_reduction += measure.expected_risk_reduction

    return OptimizationResult(
        allocations=allocations,
        total_cost=total_cost,
        total_predicted_risk_reduction=total_reduction,
        solver_status=prob.status,
        budget_utilization=(total_cost / total_budget * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        ),
    )
```

**Pareto Frontier Generation:**

```python
def generate_pareto_frontier(
    suppliers: List[SupplierMitigationContext],
    candidate_measures: Dict[str, List[MeasureCost]],
    budget_range: Tuple[Decimal, Decimal],
    steps: int = 10,
) -> List[ParetoPoint]:
    """Generate Pareto-optimal frontier by varying budget constraint."""
    min_budget, max_budget = budget_range
    step_size = ((max_budget - min_budget) / Decimal(str(steps))).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    points = []
    for i in range(steps + 1):
        budget = min_budget + step_size * Decimal(str(i))
        result = optimize_budget_allocation(suppliers, candidate_measures, budget)
        points.append(ParetoPoint(
            budget=budget,
            risk_reduction=result.total_predicted_risk_reduction,
            allocation_count=len(result.allocations),
        ))
    return points
```

**Zero-Hallucination Guarantee:** LP optimization is deterministic (PuLP CBC solver with fixed seed). Budget calculations use Decimal arithmetic. Pareto frontier uses discrete budget steps. No LLM in optimization path.

**Estimated Lines of Code:** 3,500-4,000

---

### 4.8 Engine 8: Stakeholder Collaboration Engine

**File:** `stakeholder_collaboration.py`
**Purpose:** Multi-party coordination platform connecting internal compliance teams, procurement, suppliers, NGO partners, certification bodies, and competent authorities around shared mitigation objectives.

**Responsibilities:**
- Support 6 stakeholder roles with role-based access control
- Implement threaded communication channels per mitigation plan
- Support task assignment to any stakeholder with due dates and tracking
- Provide supplier self-service portal: view plan, report progress, upload evidence
- Support NGO partnership workspace with shared goals and progress tracking
- Implement document sharing with version control and access logging
- Generate stakeholder-specific progress dashboards
- Support bulk communication to supplier groups
- Implement activity audit trail for regulatory evidence

**Stakeholder Access Matrix (Deterministic RBAC):**

```python
STAKEHOLDER_ACCESS_MATRIX = {
    "internal_compliance": {
        "view_all_plans": True, "create_edit_plans": True,
        "report_progress": True, "upload_evidence": True,
        "view_risk_scores": "full", "communication": "full",
        "analytics": "full", "export_reports": "full",
    },
    "procurement": {
        "view_all_plans": True, "create_edit_plans": "limited",
        "report_progress": True, "upload_evidence": True,
        "view_risk_scores": "full", "communication": "full",
        "analytics": "full", "export_reports": "full",
    },
    "supplier": {
        "view_all_plans": "own_only", "create_edit_plans": False,
        "report_progress": "own_plan", "upload_evidence": "own_plan",
        "view_risk_scores": "own_risk", "communication": "plan_scoped",
        "analytics": "own_trends", "export_reports": "own",
    },
    "ngo_partner": {
        "view_all_plans": "landscape", "create_edit_plans": False,
        "report_progress": "joint_plans", "upload_evidence": "joint_plans",
        "view_risk_scores": "aggregate", "communication": "landscape",
        "analytics": "landscape", "export_reports": "landscape",
    },
    "certification_body": {
        "view_all_plans": "scheme_related", "create_edit_plans": False,
        "report_progress": "audit_results", "upload_evidence": "audit_reports",
        "view_risk_scores": "scheme_related", "communication": "scheme",
        "analytics": "scheme", "export_reports": "scheme",
    },
    "competent_authority": {
        "view_all_plans": "requested", "create_edit_plans": False,
        "report_progress": False, "upload_evidence": False,
        "view_risk_scores": "full", "communication": "request_response",
        "analytics": "compliance", "export_reports": "compliance",
    },
}
```

**Zero-Hallucination Guarantee:** Access control is deterministic RBAC. Message storage is append-only with timestamps. No LLM in access decisions or collaboration logic.

**Estimated Lines of Code:** 3,000-3,500

---

## 5. Data Flow Architecture

### 5.1 P0 Feature Data Flows

**F1: Strategy Recommendation Flow**
```
EUDR-016 (country_risk) -------+
EUDR-017 (supplier_risk) ------+
EUDR-018 (commodity_risk) -----+
EUDR-019 (corruption_index) ---+---> E1: extract_features() -> recommend_strategies()
EUDR-020 (deforestation) ------+         |
EUDR-021 (indigenous_rights) --+         v
EUDR-022 (protected_areas) ----+    ML Model or Deterministic Fallback
EUDR-023 (legal_compliance) ---+         |
EUDR-024 (audit_status) -------+         v
                                   mitigation_strategies table
                                         |
                                         v
                              API: POST /v1/eudr-rma/strategies/recommend
```

**F2: Remediation Plan Flow**
```
E1: selected_strategies ----> E2: generate_remediation_plan()
E4: measure_library -------+         |
                                     v
                              remediation_plans + plan_milestones tables
                                     |
                                     v
                              API: POST /v1/eudr-rma/plans
                              API: GET  /v1/eudr-rma/plans/{id}/gantt
```

**F3: Capacity Building Flow**
```
EUDR-017 (supplier_risk) ----> E3: enroll_supplier()
E4: training_modules ------+         |
                                     v
                              capacity_building_enrollments table
                                     |
                                     v
                              E3: calculate_tier_advancement()
                                     |
                                     v
                              API: GET /v1/eudr-rma/capacity-building/scorecard/{id}
```

**F4: Measure Library Search Flow**
```
API: GET /v1/eudr-rma/measures ----> E4: search_measures()
     ?query=...&category=...              |
     &commodity=...&cost_max=...          v
                                   PostgreSQL GIN full-text search
                                         |
                                         v
                                   Ranked measure results (< 500ms)
```

**F5: Effectiveness Tracking Flow**
```
Plan activation -------> E5: capture_baseline() -> T0 snapshot (9 agents)
                              |
Monthly schedule -------> E5: capture_periodic() -> Tn scores
                              |
                              v
                        E5: calculate_risk_reduction() (Decimal)
                        E5: calculate_roi() (Decimal)
                              |
                              v
                        effectiveness_records hypertable
                              |
                              v
                        E1: feedback loop (model retraining data)
```

**F6: Adaptive Management Flow**
```
EUDR-016 events ----+
EUDR-017 events ----+
EUDR-020 events ----+---> Redis Streams ----> E6: process_risk_event()
EUDR-021 events ----+                              |
EUDR-022 events ----+                              v
EUDR-024 events ----+                   _match_trigger() (deterministic)
                                              |
                                              v
                                   trigger_events hypertable
                                              |
                                              v
                                   _notify_escalation_chain()
                                              |
                                              v
                              API: GET /v1/eudr-rma/monitoring/triggers
```

**F7: Cost-Benefit Optimization Flow**
```
API: POST /v1/eudr-rma/optimization/run
     {budget, constraints} ----> E7: optimize_budget_allocation()
E4: measure_costs -----------+         |
E5: historical_effectiveness +         v
                                  PuLP CBC solver (deterministic)
                                       |
                                       v
                                  optimization_results table
                                       |
                                       v
                              E7: generate_pareto_frontier()
                                       |
                                       v
                              API: GET /v1/eudr-rma/optimization/{id}/pareto
```

**F8: Stakeholder Collaboration Flow**
```
API: POST /v1/eudr-rma/collaboration/{plan_id}/messages
                              |
                              v
                        E8: validate_access() (RBAC matrix)
                              |
                              v
                        collaboration_messages hypertable
                              |
                              v
                        Real-time notification (WebSocket/SSE)
                              |
                              v
                        Supplier portal: GET /v1/eudr-rma/collaboration/supplier-portal/{id}
```

**F9: Report Generation Flow**
```
E1 (strategies) -----+
E2 (plans) ----------+
E3 (enrollments) ----+---> mitigation_reporter: generate_report()
E5 (effectiveness) --+           |
E7 (optimization) ---+           v
                          Template rendering (Jinja2)
                                 |
                                 v
                          SHA-256 hash + S3 upload
                                 |
                                 v
                          mitigation_reports hypertable
                                 |
                                 v
                          API: GET /v1/eudr-rma/reports/{id}/download
```

---

## 6. Database Schema (V113)

### 6.1 Overview

- **Migration:** `V113__agent_eudr_risk_mitigation_advisor.sql`
- **Schema:** `eudr_risk_mitigation_advisor`
- **Tables:** 14 (10 regular + 4 TimescaleDB hypertables)
- **Estimated indexes:** ~30
- **Retention:** 5 years (EUDR Article 31)

### 6.2 Regular Tables (10)

**Table 1: `eudr_risk_mitigation_advisor.mitigation_strategies`** -- Recommended and selected strategies

| Column | Type | Constraints |
|--------|------|-------------|
| strategy_id | UUID | PK DEFAULT gen_random_uuid() |
| operator_id | UUID | NOT NULL |
| supplier_id | UUID | |
| name | VARCHAR(500) | NOT NULL |
| description | TEXT | |
| risk_categories | JSONB | NOT NULL DEFAULT '[]' |
| iso_31000_type | VARCHAR(50) | NOT NULL |
| target_risk_factors | JSONB | DEFAULT '[]' |
| predicted_effectiveness | NUMERIC(5,2) | DEFAULT 0.0 |
| confidence_score | NUMERIC(4,3) | DEFAULT 0.0 |
| cost_estimate | JSONB | DEFAULT '{}' |
| implementation_complexity | VARCHAR(20) | DEFAULT 'medium' CHECK IN ('low','medium','high','very_high') |
| time_to_effect_weeks | INTEGER | DEFAULT 8 |
| prerequisite_conditions | JSONB | DEFAULT '[]' |
| eudr_articles | JSONB | DEFAULT '[]' |
| shap_explanation | JSONB | DEFAULT '{}' |
| measure_ids | JSONB | DEFAULT '[]' |
| model_version | VARCHAR(50) | |
| provenance_hash | VARCHAR(64) | NOT NULL |
| status | VARCHAR(30) | DEFAULT 'recommended' CHECK IN ('recommended','selected','active','completed','rejected') |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** operator_id, supplier_id, status, (operator_id, status)

---

**Table 2: `eudr_risk_mitigation_advisor.remediation_plans`** -- Remediation plan records

| Column | Type | Constraints |
|--------|------|-------------|
| plan_id | UUID | PK DEFAULT gen_random_uuid() |
| operator_id | UUID | NOT NULL |
| supplier_id | UUID | |
| plan_name | VARCHAR(500) | NOT NULL |
| risk_finding_ids | JSONB | DEFAULT '[]' |
| strategy_ids | JSONB | DEFAULT '[]' |
| status | VARCHAR(30) | DEFAULT 'draft' CHECK IN ('draft','active','on_track','at_risk','delayed','completed','suspended','abandoned') |
| phases | JSONB | DEFAULT '[]' |
| budget_allocated | NUMERIC(18,2) | DEFAULT 0.0 |
| budget_spent | NUMERIC(18,2) | DEFAULT 0.0 |
| start_date | DATE | |
| target_end_date | DATE | |
| actual_end_date | DATE | |
| responsible_parties | JSONB | DEFAULT '[]' |
| escalation_triggers | JSONB | DEFAULT '[]' |
| plan_template | VARCHAR(100) | |
| version | INTEGER | DEFAULT 1 |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** operator_id, supplier_id, status, (operator_id, status), (supplier_id, status)

---

**Table 3: `eudr_risk_mitigation_advisor.plan_milestones`** -- SMART milestones

| Column | Type | Constraints |
|--------|------|-------------|
| milestone_id | UUID | PK DEFAULT gen_random_uuid() |
| plan_id | UUID | NOT NULL FK -> remediation_plans |
| name | VARCHAR(500) | NOT NULL |
| description | TEXT | |
| phase | VARCHAR(100) | |
| due_date | DATE | NOT NULL |
| completed_date | DATE | |
| status | VARCHAR(30) | DEFAULT 'pending' CHECK IN ('pending','in_progress','completed','overdue','skipped') |
| kpi_target | VARCHAR(200) | |
| evidence_required | JSONB | DEFAULT '[]' |
| evidence_uploaded | JSONB | DEFAULT '[]' |
| eudr_article | VARCHAR(20) | |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** plan_id, status, due_date, (plan_id, status)

---

**Table 4: `eudr_risk_mitigation_advisor.mitigation_measures`** -- 500+ measure library

| Column | Type | Constraints |
|--------|------|-------------|
| measure_id | UUID | PK DEFAULT gen_random_uuid() |
| name | VARCHAR(500) | NOT NULL |
| description | TEXT | NOT NULL |
| risk_category | VARCHAR(50) | NOT NULL |
| sub_category | VARCHAR(100) | |
| target_risk_factors | JSONB | DEFAULT '[]' |
| applicability | JSONB | DEFAULT '{}' |
| effectiveness_evidence | JSONB | DEFAULT '[]' |
| effectiveness_rating | NUMERIC(5,2) | DEFAULT 0.0 |
| cost_estimate_min | NUMERIC(18,2) | |
| cost_estimate_max | NUMERIC(18,2) | |
| implementation_complexity | VARCHAR(20) | DEFAULT 'medium' |
| time_to_effect_weeks | INTEGER | DEFAULT 8 |
| prerequisite_conditions | JSONB | DEFAULT '[]' |
| expected_risk_reduction_min | NUMERIC(5,2) | |
| expected_risk_reduction_max | NUMERIC(5,2) | |
| iso_31000_type | VARCHAR(50) | |
| eudr_articles | JSONB | DEFAULT '[]' |
| certification_schemes | JSONB | DEFAULT '[]' |
| tags | JSONB | DEFAULT '[]' |
| version | VARCHAR(20) | DEFAULT '1.0.0' |
| is_active | BOOLEAN | DEFAULT TRUE |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** risk_category, implementation_complexity, is_active, GIN(to_tsvector('english', name || ' ' || description))

---

**Table 5: `eudr_risk_mitigation_advisor.capacity_building_enrollments`** -- Supplier enrollments

| Column | Type | Constraints |
|--------|------|-------------|
| enrollment_id | UUID | PK DEFAULT gen_random_uuid() |
| supplier_id | UUID | NOT NULL |
| program_id | VARCHAR(100) | NOT NULL |
| commodity | VARCHAR(50) | NOT NULL |
| current_tier | INTEGER | DEFAULT 1 CHECK (current_tier BETWEEN 1 AND 4) |
| modules_completed | INTEGER | DEFAULT 0 |
| modules_total | INTEGER | DEFAULT 22 |
| competency_scores | JSONB | DEFAULT '{}' |
| enrolled_date | DATE | NOT NULL |
| target_completion_date | DATE | |
| status | VARCHAR(30) | DEFAULT 'active' CHECK IN ('active','paused','completed','withdrawn') |
| risk_score_at_enrollment | NUMERIC(5,2) | |
| current_risk_score | NUMERIC(5,2) | |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** supplier_id, status, commodity, (supplier_id, status)

---

**Table 6: `eudr_risk_mitigation_advisor.optimization_results`** -- Budget optimization

| Column | Type | Constraints |
|--------|------|-------------|
| optimization_id | UUID | PK DEFAULT gen_random_uuid() |
| operator_id | UUID | NOT NULL |
| budget_total | NUMERIC(18,2) | NOT NULL |
| budget_constraints | JSONB | DEFAULT '{}' |
| optimization_result | JSONB | NOT NULL |
| pareto_frontier | JSONB | DEFAULT '[]' |
| sensitivity_analysis | JSONB | DEFAULT '{}' |
| total_predicted_risk_reduction | NUMERIC(5,2) | |
| solver_status | VARCHAR(30) | |
| computation_time_ms | INTEGER | |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** operator_id, created_at

---

**Table 7: `eudr_risk_mitigation_advisor.collaboration_tasks`** -- Stakeholder tasks

| Column | Type | Constraints |
|--------|------|-------------|
| task_id | UUID | PK DEFAULT gen_random_uuid() |
| plan_id | UUID | NOT NULL FK -> remediation_plans |
| assigned_to | VARCHAR(100) | NOT NULL |
| assigned_role | VARCHAR(50) | NOT NULL |
| title | VARCHAR(500) | NOT NULL |
| description | TEXT | |
| due_date | DATE | |
| priority | VARCHAR(20) | DEFAULT 'medium' CHECK IN ('low','medium','high','urgent') |
| status | VARCHAR(30) | DEFAULT 'pending' CHECK IN ('pending','in_progress','completed','cancelled') |
| evidence_ids | JSONB | DEFAULT '[]' |
| created_by | VARCHAR(100) | NOT NULL |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| updated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** plan_id, assigned_to, status, (plan_id, status)

---

**Table 8: `eudr_risk_mitigation_advisor.ml_model_registry`** -- ML model versioning

| Column | Type | Constraints |
|--------|------|-------------|
| model_id | UUID | PK DEFAULT gen_random_uuid() |
| model_name | VARCHAR(200) | NOT NULL |
| model_version | VARCHAR(50) | NOT NULL |
| model_type | VARCHAR(50) | NOT NULL CHECK IN ('xgboost','lightgbm') |
| s3_path | VARCHAR(1000) | NOT NULL |
| training_data_hash | VARCHAR(64) | NOT NULL |
| training_metrics | JSONB | DEFAULT '{}' |
| feature_importance | JSONB | DEFAULT '{}' |
| hyperparameters | JSONB | DEFAULT '{}' |
| is_active | BOOLEAN | DEFAULT FALSE |
| deployed_at | TIMESTAMPTZ | |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| UNIQUE(model_name, model_version) | | |

**Indexes:** model_name, is_active, (model_name, is_active)

---

**Table 9: `eudr_risk_mitigation_advisor.evidence_documents`** -- Evidence files

| Column | Type | Constraints |
|--------|------|-------------|
| evidence_id | UUID | PK DEFAULT gen_random_uuid() |
| plan_id | UUID | FK -> remediation_plans |
| milestone_id | UUID | FK -> plan_milestones |
| file_name | VARCHAR(500) | NOT NULL |
| file_path | VARCHAR(1000) | NOT NULL |
| file_size_bytes | BIGINT | |
| mime_type | VARCHAR(100) | |
| description | TEXT | |
| sha256_hash | VARCHAR(64) | NOT NULL |
| uploaded_by | VARCHAR(100) | NOT NULL |
| metadata | JSONB | DEFAULT '{}' |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** plan_id, milestone_id, sha256_hash

---

**Table 10: `eudr_risk_mitigation_advisor.plan_versions`** -- Plan version history

| Column | Type | Constraints |
|--------|------|-------------|
| version_id | UUID | PK DEFAULT gen_random_uuid() |
| plan_id | UUID | NOT NULL FK -> remediation_plans |
| version_number | INTEGER | NOT NULL |
| plan_snapshot | JSONB | NOT NULL |
| change_summary | TEXT | |
| changed_by | VARCHAR(100) | NOT NULL |
| approved_by | VARCHAR(100) | |
| approved_at | TIMESTAMPTZ | |
| created_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |

**Indexes:** plan_id, (plan_id, version_number)

---

### 6.3 TimescaleDB Hypertables (4)

**Hypertable 11: `eudr_risk_mitigation_advisor.effectiveness_records`**

| Column | Type | Constraints |
|--------|------|-------------|
| record_id | UUID | DEFAULT gen_random_uuid() |
| plan_id | UUID | NOT NULL |
| supplier_id | UUID | NOT NULL |
| baseline_risk_scores | JSONB | NOT NULL |
| current_risk_scores | JSONB | NOT NULL |
| risk_reduction_pct | JSONB | NOT NULL |
| composite_reduction_pct | NUMERIC(5,2) | |
| predicted_reduction_pct | NUMERIC(5,2) | |
| deviation_pct | NUMERIC(5,2) | |
| roi | NUMERIC(10,2) | |
| cost_to_date | NUMERIC(18,2) | |
| statistical_significance | BOOLEAN | DEFAULT FALSE |
| p_value | NUMERIC(6,4) | |
| provenance_hash | VARCHAR(64) | NOT NULL |
| measured_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| PRIMARY KEY (record_id, measured_at) | | |

- Chunk interval: 30 days; Retention: 5 years
- **Indexes:** plan_id, supplier_id

---

**Hypertable 12: `eudr_risk_mitigation_advisor.trigger_events`**

| Column | Type | Constraints |
|--------|------|-------------|
| event_id | UUID | DEFAULT gen_random_uuid() |
| plan_id | UUID | |
| trigger_type | VARCHAR(50) | NOT NULL |
| source_agent | VARCHAR(50) | NOT NULL |
| severity | VARCHAR(20) | NOT NULL CHECK IN ('critical','high','medium','low') |
| description | TEXT | NOT NULL |
| risk_data | JSONB | DEFAULT '{}' |
| recommended_adjustment | JSONB | DEFAULT '{}' |
| adjustment_type | VARCHAR(50) | |
| acknowledged | BOOLEAN | DEFAULT FALSE |
| acknowledged_by | VARCHAR(100) | |
| acknowledged_at | TIMESTAMPTZ | |
| resolved | BOOLEAN | DEFAULT FALSE |
| resolved_at | TIMESTAMPTZ | |
| detected_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| PRIMARY KEY (event_id, detected_at) | | |

- Chunk interval: 30 days; Retention: 5 years
- **Indexes:** plan_id, severity, resolved

---

**Hypertable 13: `eudr_risk_mitigation_advisor.collaboration_messages`**

| Column | Type | Constraints |
|--------|------|-------------|
| message_id | UUID | DEFAULT gen_random_uuid() |
| plan_id | UUID | NOT NULL |
| sender_id | VARCHAR(100) | NOT NULL |
| sender_role | VARCHAR(50) | NOT NULL |
| message_type | VARCHAR(30) | DEFAULT 'text' CHECK IN ('text','task_update','evidence_upload','system_notification') |
| content | TEXT | NOT NULL |
| attachments | JSONB | DEFAULT '[]' |
| mentions | JSONB | DEFAULT '[]' |
| read_by | JSONB | DEFAULT '[]' |
| sent_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| PRIMARY KEY (message_id, sent_at) | | |

- Chunk interval: 30 days; Retention: 5 years
- **Indexes:** plan_id, sender_role

---

**Hypertable 14: `eudr_risk_mitigation_advisor.mitigation_reports`**

| Column | Type | Constraints |
|--------|------|-------------|
| report_id | UUID | DEFAULT gen_random_uuid() |
| operator_id | UUID | NOT NULL |
| report_type | VARCHAR(50) | NOT NULL CHECK IN ('dds_mitigation','authority_package','annual_review','supplier_scorecard','portfolio_summary','risk_mitigation_mapping','effectiveness_analysis') |
| report_scope | JSONB | DEFAULT '{}' |
| report_data | JSONB | NOT NULL |
| format | VARCHAR(10) | DEFAULT 'pdf' CHECK IN ('pdf','json','html','xlsx','xml') |
| language | VARCHAR(5) | DEFAULT 'en' CHECK IN ('en','fr','de','es','pt') |
| s3_key | VARCHAR(500) | |
| provenance_hash | VARCHAR(64) | NOT NULL |
| generated_at | TIMESTAMPTZ | NOT NULL DEFAULT NOW() |
| PRIMARY KEY (report_id, generated_at) | | |

- Chunk interval: 90 days; Retention: 5 years (EUDR Article 31)
- **Indexes:** operator_id, report_type

### 6.4 Continuous Aggregates (2)

**1. `gl_eudr_rma_monthly_effectiveness_summary`**
```sql
SELECT
    time_bucket('1 month', measured_at) AS month,
    supplier_id,
    AVG(composite_reduction_pct) AS avg_reduction,
    AVG(roi) AS avg_roi,
    COUNT(*) AS measurement_count
FROM eudr_risk_mitigation_advisor.effectiveness_records
GROUP BY 1, 2;
```

**2. `gl_eudr_rma_monthly_trigger_summary`**
```sql
SELECT
    time_bucket('1 month', detected_at) AS month,
    source_agent,
    severity,
    COUNT(*) AS trigger_count,
    SUM(CASE WHEN resolved THEN 1 ELSE 0 END) AS resolved_count
FROM eudr_risk_mitigation_advisor.trigger_events
GROUP BY 1, 2, 3;
```

---

## 7. API Architecture (~35 Endpoints)

**API Prefix:** `/v1/eudr-rma`

### 7.1 Strategy Recommendation Routes (`strategy_routes.py`) -- 5 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/strategies/recommend` | `eudr-rma:strategies:execute` | Generate mitigation strategy recommendations for a risk context |
| GET | `/strategies` | `eudr-rma:strategies:read` | List recommended strategies (filters: status, supplier, category) |
| GET | `/strategies/{strategy_id}` | `eudr-rma:strategies:read` | Get strategy details with SHAP explanation |
| POST | `/strategies/{strategy_id}/select` | `eudr-rma:plans:write` | Select strategy for implementation |
| GET | `/strategies/{strategy_id}/explain` | `eudr-rma:strategies:read` | Get detailed SHAP explainability report |

### 7.2 Remediation Plan Routes (`plan_routes.py`) -- 7 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/plans` | `eudr-rma:plans:write` | Create a new remediation plan |
| GET | `/plans` | `eudr-rma:plans:read` | List plans (filters: status, supplier, commodity) |
| GET | `/plans/{plan_id}` | `eudr-rma:plans:read` | Get plan details with milestones and progress |
| PUT | `/plans/{plan_id}` | `eudr-rma:plans:write` | Update plan details |
| PUT | `/plans/{plan_id}/status` | `eudr-rma:plans:approve` | Update plan status (activate, complete, suspend) |
| POST | `/plans/{plan_id}/clone` | `eudr-rma:plans:write` | Clone plan as template for another supplier |
| GET | `/plans/{plan_id}/gantt` | `eudr-rma:plans:read` | Get Gantt chart data with dependencies |

### 7.3 Plan Milestone Routes (in `plan_routes.py`) -- 3 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/plans/{plan_id}/milestones` | `eudr-rma:plans:write` | Add milestone to plan |
| PUT | `/plans/{plan_id}/milestones/{milestone_id}` | `eudr-rma:plans:write` | Update milestone status/evidence |
| POST | `/plans/{plan_id}/milestones/{milestone_id}/evidence` | `eudr-rma:plans:write` | Upload evidence for milestone |

### 7.4 Capacity Building Routes (`capacity_routes.py`) -- 5 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/capacity-building/enroll` | `eudr-rma:capacity:manage` | Enroll supplier in capacity building program |
| GET | `/capacity-building/enrollments` | `eudr-rma:capacity:read` | List enrollments (filters: status, commodity, tier) |
| GET | `/capacity-building/enrollments/{enrollment_id}` | `eudr-rma:capacity:read` | Get enrollment progress |
| PUT | `/capacity-building/enrollments/{enrollment_id}/progress` | `eudr-rma:capacity:manage` | Update module completion |
| GET | `/capacity-building/scorecard/{supplier_id}` | `eudr-rma:capacity:read` | Get supplier capacity scorecard |

### 7.5 Mitigation Measure Library Routes (`library_routes.py`) -- 4 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| GET | `/measures` | `eudr-rma:library:read` | Search/list measures (full-text + faceted filters) |
| GET | `/measures/{measure_id}` | `eudr-rma:library:read` | Get measure details with evidence |
| GET | `/measures/compare` | `eudr-rma:library:read` | Compare measures side-by-side (query: ids) |
| GET | `/measures/packages/{risk_scenario}` | `eudr-rma:library:read` | Get recommended measure package |

### 7.6 Effectiveness Tracking Routes (`effectiveness_routes.py`) -- 4 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| GET | `/effectiveness/{plan_id}` | `eudr-rma:effectiveness:read` | Get effectiveness metrics for plan |
| GET | `/effectiveness/supplier/{supplier_id}` | `eudr-rma:effectiveness:read` | Get supplier effectiveness history |
| GET | `/effectiveness/portfolio` | `eudr-rma:effectiveness:read` | Get portfolio-level effectiveness summary |
| GET | `/effectiveness/roi` | `eudr-rma:effectiveness:read` | Get ROI analysis across portfolio |

### 7.7 Adaptive Management Routes (`monitoring_routes.py`) -- 4 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| GET | `/monitoring/triggers` | `eudr-rma:monitoring:read` | List active trigger events |
| PUT | `/monitoring/triggers/{event_id}/acknowledge` | `eudr-rma:monitoring:acknowledge` | Acknowledge a trigger event |
| GET | `/monitoring/dashboard` | `eudr-rma:monitoring:read` | Get real-time monitoring dashboard data |
| GET | `/monitoring/drift/{plan_id}` | `eudr-rma:monitoring:read` | Get plan drift analysis |

### 7.8 Cost-Benefit Optimization Routes (`optimization_routes.py`) -- 4 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/optimization/run` | `eudr-rma:optimization:execute` | Run budget optimization |
| GET | `/optimization/{optimization_id}` | `eudr-rma:optimization:read` | Get optimization results |
| GET | `/optimization/{optimization_id}/pareto` | `eudr-rma:optimization:read` | Get Pareto frontier data |
| GET | `/optimization/{optimization_id}/sensitivity` | `eudr-rma:optimization:read` | Get sensitivity analysis |

### 7.9 Stakeholder Collaboration Routes (`collaboration_routes.py`) -- 4 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/collaboration/{plan_id}/messages` | `eudr-rma:collaboration:participate` | Post message to plan thread |
| GET | `/collaboration/{plan_id}/messages` | `eudr-rma:collaboration:participate` | Get plan conversation thread |
| POST | `/collaboration/{plan_id}/tasks` | `eudr-rma:collaboration:manage` | Assign task to stakeholder |
| GET | `/collaboration/supplier-portal/{supplier_id}` | `eudr-rma:supplier-portal:access` | Supplier self-service portal data |

### 7.10 Reporting Routes (`reporting_routes.py`) -- 4 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| POST | `/reports/generate` | `eudr-rma:reports:generate` | Generate mitigation report |
| GET | `/reports` | `eudr-rma:reports:read` | List generated reports |
| GET | `/reports/{report_id}/download` | `eudr-rma:reports:read` | Download report file |
| GET | `/reports/dds-section/{operator_id}` | `eudr-rma:reports:dds` | Get DDS mitigation section data |

### 7.11 Admin Routes (`admin_routes.py`) -- 2 endpoints

| Method | Path | Permission | Description |
|--------|------|------------|-------------|
| GET | `/health` | (public) | Health check |
| GET | `/stats` | `eudr-rma:audit:read` | Service statistics |

**Total: ~43 endpoints**

---

## 8. Integration Architecture

### 8.1 Upstream Dependencies (9 EUDR Risk Assessment Agents)

| Agent | Integration Method | Data Consumed |
|-------|-------------------|---------------|
| EUDR-016 Country Risk Evaluator | REST API + Event Stream | Country risk scores, due diligence levels, governance indices, hotspot data |
| EUDR-017 Supplier Risk Scorer | REST API + Event Stream (bidirectional) | Supplier composite risk, compliance history, risk factor breakdown |
| EUDR-018 Commodity Risk Analyzer | REST API | Commodity risk profiles, deforestation correlation, certification coverage |
| EUDR-019 Corruption Index Monitor | REST API + Event Stream | CPI scores, governance scores, corruption alerts, transparency indices |
| EUDR-020 Deforestation Alert System | Event Stream (priority) | Alert severity, proximity, affected plots, area, cutoff verification |
| EUDR-021 Indigenous Rights Checker | REST API + Event Stream | Territory overlap scores, FPIC status, violation alerts |
| EUDR-022 Protected Area Validator | REST API + Event Stream | Protected area proximity, buffer zone violations, IUCN categories |
| EUDR-023 Legal Compliance Verifier | REST API | Legal compliance gaps, permit status, certification validity |
| EUDR-024 Third-Party Audit Manager | REST API + Event Stream | Audit findings, NC severity, CAR status, audit compliance rate |

### 8.2 Additional Upstream Dependencies

| Agent | Integration Method | Data Consumed |
|-------|-------------------|---------------|
| AGENT-DATA-005 EUDR Traceability Connector | REST API | Supply chain data for context enrichment |
| AGENT-FOUND-005 Citations and Evidence Agent | REST API | Regulatory references for compliance mapping |

### 8.3 Downstream Consumers

| Consumer | Integration Method | Data Provided |
|----------|-------------------|---------------|
| GL-EUDR-APP v1.0 | API integration | Mitigation dashboard data, plan status, effectiveness metrics |
| GL-EUDR-APP DDS Generator | API | DDS mitigation section (Article 12(2)(d)) |
| EUDR-017 Supplier Risk Scorer | Event Stream | Mitigation status updates influencing supplier risk scoring |
| EUDR-024 Third-Party Audit Manager | API | CAR-linked remediation plans, mitigation evidence |
| External Auditors | Read-only API + Report Downloads | Audit-ready mitigation evidence packages |
| Competent Authorities | Report Downloads | Regulatory compliance documentation packages |

### 8.4 API Contracts

**Provided to EUDR-017 (Supplier Risk Scorer):**
```python
async def get_supplier_mitigation_status(supplier_id: str) -> Dict:
    """Returns mitigation status data for supplier risk scoring."""
    return {
        "supplier_id": str,
        "active_plans": int,
        "plans_on_track": int,
        "plans_delayed": int,
        "composite_risk_reduction_pct": Decimal,    # 0-100
        "capacity_building_tier": int,               # 1-4
        "capacity_building_completion_pct": Decimal, # 0-100
        "mitigation_investment_eur": Decimal,
        "effectiveness_roi": Decimal,
        "last_effectiveness_measurement": str,       # ISO 8601
        "provenance_hash": str,
    }
```

**Consumed from EUDR-016 (Country Risk Evaluator):**
```python
async def get_country_risk_for_mitigation(country_code: str) -> Dict:
    return {
        "country_code": str,
        "risk_level": str,                   # "low" | "standard" | "high"
        "risk_score": Decimal,               # 0-100
        "due_diligence_level": str,
        "governance_index": Decimal,         # 0-100
        "enforcement_score": Decimal,        # 0-100
        "forest_cover_pct": Decimal,
        "provenance_hash": str,
    }
```

**Consumed from EUDR-020 (Deforestation Alert System):**
```python
async def get_deforestation_alerts_for_mitigation(supplier_id: str) -> Dict:
    return {
        "supplier_id": str,
        "active_alerts": List[Dict],
        "max_alert_severity": Decimal,       # 0-100
        "alert_count": int,
        "min_proximity_km": Decimal,
        "total_area_ha": Decimal,
        "post_cutoff": bool,
        "provenance_hash": str,
    }
```

### 8.5 Infrastructure Integrations

| Component | Usage |
|-----------|-------|
| PostgreSQL 14+ (TimescaleDB) | Primary data store (14 tables, 4 hypertables) |
| Redis 7+ | Strategy recommendation cache, ML model cache, session state, event streams |
| S3 (AWS/MinIO) | Evidence documents, ML model artifacts, generated reports |
| SEC-001 JWT Auth | JWT RS256 token validation on all protected endpoints |
| SEC-002 RBAC | 21 permissions with `eudr-rma:` prefix |
| SEC-003 Encryption | AES-256-GCM for sensitive data at rest |
| SEC-005 Audit Logging | All CRUD operations logged |
| OBS-001 Prometheus | 18 metrics with `gl_eudr_rma_` prefix |
| OBS-003 OpenTelemetry | Distributed tracing across engine and agent calls |

---

## 9. ML/AI Architecture

### 9.1 Model Architecture

**Primary Model: XGBoost Gradient-Boosted Decision Tree**

| Parameter | Value |
|-----------|-------|
| Algorithm | XGBoost (xgb.XGBClassifier) |
| Task | Multi-label classification (strategy ranking) |
| Input features | 45+ numeric features from 9-agent risk profile |
| Output | Probability distribution over 50+ strategy templates |
| Training data | 10,000+ historical mitigation outcomes (synthetic + real) |
| Validation | 5-fold cross-validation, stratified by risk category |
| Target metric | Weighted F1-score >= 0.80 |
| Hyperparameters | max_depth=6, learning_rate=0.1, n_estimators=200, min_child_weight=5 |
| Regularization | L1 (alpha=0.1), L2 (lambda=1.0) |
| Feature selection | Mutual information + SHAP importance ranking |

**Secondary Model: LightGBM (Fallback/Ensemble)**

| Parameter | Value |
|-----------|-------|
| Algorithm | LightGBM (lgb.LGBMClassifier) |
| Usage | Ensemble member for improved confidence; fallback if XGBoost unavailable |
| Hyperparameters | num_leaves=31, learning_rate=0.1, n_estimators=150, min_data_in_leaf=20 |

### 9.2 SHAP Explainability

```python
import shap

def generate_shap_explanation(
    model: xgb.XGBClassifier,
    feature_array: np.ndarray,
    feature_names: List[str],
    strategy_index: int,
) -> Dict[str, float]:
    """
    Generate SHAP values for a specific strategy recommendation.
    Returns feature -> SHAP value mapping for audit trail.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_array)

    # Extract SHAP values for the recommended strategy
    strategy_shap = shap_values[strategy_index][0]
    explanation = {}
    for i, feature_name in enumerate(feature_names):
        explanation[feature_name] = round(float(strategy_shap[i]), 4)

    # Sort by absolute SHAP value (most influential first)
    sorted_explanation = dict(
        sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)
    )
    return sorted_explanation
```

### 9.3 Model Training Pipeline

```
Historical Data Collection
    -> Feature Engineering (extract_features from 9-agent snapshots)
    -> Train/Test Split (80/20, stratified)
    -> Hyperparameter Optimization (Optuna, 100 trials)
    -> Model Training (XGBoost + LightGBM)
    -> Validation (5-fold CV, F1 >= 0.80)
    -> SHAP Analysis (feature importance ranking)
    -> Model Registration (model_registry table + S3 artifact)
    -> A/B Deployment (shadow mode -> primary)
```

### 9.4 Deterministic Fallback Guarantee

The ML model is NEVER in the critical calculation path. It is used only for strategy RANKING. The actual strategies are pre-defined reference data (Measure Library, Engine 4). If ML confidence < 0.7, the system falls back to the deterministic rule-based decision tree (Section 4.1) which uses fixed Decimal weights and pre-coded thresholds.

| Component | ML Allowed | Deterministic Required |
|-----------|-----------|----------------------|
| Strategy ranking/scoring | Yes (with fallback) | Fallback when confidence < 0.7 |
| Risk reduction calculation | No | Decimal arithmetic only |
| ROI calculation | No | Decimal arithmetic only |
| Tier advancement scoring | No | Decimal thresholds only |
| LP budget optimization | No | PuLP CBC solver only |
| Trigger event detection | No | Rule-based matching only |
| SLA/deadline calculation | No | Integer day arithmetic only |
| Report data generation | No | Engine outputs only |
| Narrative text in reports | Yes (clearly marked) | Template text as default |

---

## 10. Security Architecture

### 10.1 Authentication

- All endpoints (except `/health`) require JWT authentication via SEC-001
- JWT RS256 token validation with configurable JWKS endpoint
- API key support for machine-to-machine integrations (inter-agent calls, supplier portal)
- Token expiry: 1 hour (configurable via `GL_EUDR_RMA_TOKEN_EXPIRY_S`)

### 10.2 Authorization (RBAC -- 21 Permissions)

| # | Permission | Description | Roles |
|---|------------|-------------|-------|
| 1 | `eudr-rma:strategies:read` | View mitigation strategy recommendations | Viewer, Analyst, Compliance Officer, Admin |
| 2 | `eudr-rma:strategies:execute` | Generate new strategy recommendations | Analyst, Compliance Officer, Admin |
| 3 | `eudr-rma:plans:read` | View remediation plans and milestones | Viewer, Analyst, Compliance Officer, Admin |
| 4 | `eudr-rma:plans:write` | Create, update, clone remediation plans | Analyst, Compliance Officer, Admin |
| 5 | `eudr-rma:plans:approve` | Approve plan activation and completion | Compliance Officer, Admin |
| 6 | `eudr-rma:capacity:read` | View capacity building enrollments | Viewer, Analyst, Compliance Officer, Admin |
| 7 | `eudr-rma:capacity:manage` | Manage capacity building programs | Procurement Manager, Compliance Officer, Admin |
| 8 | `eudr-rma:library:read` | Browse mitigation measure library | Viewer, Analyst, Compliance Officer, Admin |
| 9 | `eudr-rma:library:manage` | Add/update mitigation measures | Compliance Officer, Admin |
| 10 | `eudr-rma:effectiveness:read` | View effectiveness tracking metrics | Viewer, Analyst, Compliance Officer, Admin |
| 11 | `eudr-rma:monitoring:read` | View adaptive management dashboard | Viewer, Analyst, Compliance Officer, Admin |
| 12 | `eudr-rma:monitoring:acknowledge` | Acknowledge trigger events | Analyst, Compliance Officer, Admin |
| 13 | `eudr-rma:optimization:execute` | Run cost-benefit optimization | Sustainability Director, Compliance Officer, Admin |
| 14 | `eudr-rma:optimization:read` | View optimization results | Analyst, Compliance Officer, Admin |
| 15 | `eudr-rma:collaboration:participate` | Post messages, upload documents | All authenticated roles |
| 16 | `eudr-rma:collaboration:manage` | Manage collaboration settings, assign tasks | Compliance Officer, Admin |
| 17 | `eudr-rma:reports:read` | View and download reports | Viewer, Analyst, Compliance Officer, Auditor, Admin |
| 18 | `eudr-rma:reports:generate` | Generate new reports | Analyst, Compliance Officer, Admin |
| 19 | `eudr-rma:reports:dds` | Export DDS mitigation section | Compliance Officer, Admin |
| 20 | `eudr-rma:audit:read` | View audit trail and provenance data | Auditor (read-only), Compliance Officer, Admin |
| 21 | `eudr-rma:supplier-portal:access` | Supplier self-service portal access | Supplier (external role) |

### 10.3 Data Privacy and GDPR

| Data Type | Classification | Protection |
|-----------|---------------|------------|
| Supplier contact information | PII | AES-256 encryption via SEC-003; GDPR consent; right to erasure |
| Mitigation plan details | Confidential | RBAC access control; operator-level tenant isolation |
| Risk scores and assessments | Confidential | RBAC access control; encryption at rest |
| ML model artifacts | Internal | S3 encryption; restricted to Admin role |
| Collaboration messages | Confidential | RBAC per stakeholder role; 5-year retention |
| Evidence documents | Confidential | AES-256 at rest; SHA-256 integrity; S3 encryption |
| Authority correspondence | Highly Confidential | AES-256; restricted to Compliance Officer + Admin |
| Analytics (aggregated) | Internal | Role-based access; anonymized where possible |

### 10.4 Rate Limiting

- Default: 100 requests/minute per tenant
- Strategy recommendation: 30 requests/minute per tenant (ML inference cost)
- Optimization execution: 10 requests/minute per tenant (solver cost)
- Evidence upload: 20 requests/minute per tenant
- Report generation: 10 requests/minute per tenant
- Configurable via `GL_EUDR_RMA_RATE_LIMIT_*` environment variables

### 10.5 Multi-Tenant Isolation

- All queries filtered by `operator_id` at the repository layer
- Supplier portal access scoped to supplier's own data only
- NGO partners see aggregate/landscape data only, never individual supplier details
- Competent authority access limited to requested compliance packages

---

## 11. Observability Architecture

### 11.1 Prometheus Metrics (18)

**Prefix:** `gl_eudr_rma_`

#### Counters (8)

| # | Metric | Labels | Description |
|---|--------|--------|-------------|
| 1 | `gl_eudr_rma_strategies_recommended_total` | risk_category, mode | Strategy recommendations generated by category and mode (ml/deterministic) |
| 2 | `gl_eudr_rma_plans_created_total` | template, status | Remediation plans created by template and initial status |
| 3 | `gl_eudr_rma_milestones_completed_total` | phase | Milestones completed by plan phase |
| 4 | `gl_eudr_rma_enrollments_total` | commodity, tier | Capacity building enrollments by commodity and tier |
| 5 | `gl_eudr_rma_trigger_events_total` | source_agent, severity | Trigger events detected by source and severity |
| 6 | `gl_eudr_rma_optimizations_run_total` | solver_status | Budget optimizations executed by solver outcome |
| 7 | `gl_eudr_rma_reports_generated_total` | report_type, format | Reports generated by type and format |
| 8 | `gl_eudr_rma_ml_fallback_total` | reason | Times ML model fell back to deterministic mode |

#### Histograms (5)

| # | Metric | Buckets | Description |
|---|--------|---------|-------------|
| 9 | `gl_eudr_rma_strategy_recommendation_seconds` | 0.1, 0.5, 1, 2, 5 | Strategy recommendation latency |
| 10 | `gl_eudr_rma_plan_generation_seconds` | 0.5, 1, 2, 5, 10 | Remediation plan generation latency |
| 11 | `gl_eudr_rma_optimization_seconds` | 1, 5, 10, 20, 30, 60 | Budget optimization solver latency |
| 12 | `gl_eudr_rma_report_generation_seconds` | 1, 5, 10, 20, 30 | Report generation latency |
| 13 | `gl_eudr_rma_api_request_seconds` | 0.01, 0.05, 0.1, 0.2, 0.5, 1 | API request latency by endpoint |

#### Gauges (5)

| # | Metric | Description |
|---|--------|-------------|
| 14 | `gl_eudr_rma_active_plans` | Currently active remediation plans |
| 15 | `gl_eudr_rma_active_enrollments` | Currently active capacity building enrollments |
| 16 | `gl_eudr_rma_unresolved_triggers` | Unresolved trigger events by severity |
| 17 | `gl_eudr_rma_ml_model_confidence` | Current ML model average confidence score |
| 18 | `gl_eudr_rma_cache_hit_ratio` | Redis cache hit ratio (0.0-1.0) |

### 11.2 Grafana Dashboard

**Dashboard:** `eudr-risk-mitigation-advisor.json`
**Panels:** 18

| # | Panel | Type | Data Source |
|---|-------|------|-------------|
| 1 | Strategy Recommendations Rate | Time series | Counter #1 |
| 2 | ML vs Deterministic Mode Split | Pie chart | Counter #1 by mode label |
| 3 | Strategy Recommendation Latency (p50/p95/p99) | Time series | Histogram #9 |
| 4 | Active Remediation Plans | Stat | Gauge #14 |
| 5 | Plan Status Distribution | Pie chart | PostgreSQL query |
| 6 | Milestones Completed Over Time | Time series | Counter #3 |
| 7 | Capacity Building Enrollments by Commodity | Bar chart | Counter #4 |
| 8 | Active Enrollments by Tier | Stacked bar | Gauge #15 + PostgreSQL |
| 9 | Trigger Events by Severity | Time series | Counter #5 |
| 10 | Unresolved Triggers | Stat (alert threshold) | Gauge #16 |
| 11 | Budget Optimization Solver Time | Histogram | Histogram #11 |
| 12 | Reports Generated by Type | Bar chart | Counter #7 |
| 13 | Report Generation Latency | Time series | Histogram #12 |
| 14 | Portfolio Risk Reduction Trend | Time series | Continuous aggregate #1 |
| 15 | Portfolio ROI Trend | Time series | Continuous aggregate #1 |
| 16 | API Latency by Endpoint | Heatmap | Histogram #13 |
| 17 | ML Model Confidence | Gauge | Gauge #17 |
| 18 | Cache Hit Ratio | Gauge | Gauge #18 |

### 11.3 Alerting Rules

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| High ML Fallback Rate | `gl_eudr_rma_ml_fallback_total` > 50% in 1h | Warning | Investigate model health |
| Strategy Latency SLA Breach | `gl_eudr_rma_strategy_recommendation_seconds` p99 > 2s | Warning | Scale or optimize |
| Optimization Timeout | `gl_eudr_rma_optimization_seconds` p99 > 30s | Critical | Review solver constraints |
| Unresolved Critical Triggers | `gl_eudr_rma_unresolved_triggers{severity="critical"}` > 0 for 4h | Critical | Page on-call |
| Cache Hit Ratio Drop | `gl_eudr_rma_cache_hit_ratio` < 0.5 for 15m | Warning | Investigate Redis |
| Active Plans Spike | `gl_eudr_rma_active_plans` increase > 200% in 1h | Info | Capacity planning |

### 11.4 OpenTelemetry Tracing

- Span per engine invocation (E1 through E8)
- Span per upstream agent API call (9 agents)
- Span per database query
- Span per Redis cache operation
- Span per S3 storage operation
- Span per ML model inference
- Trace context propagated to upstream agent calls via W3C TraceContext headers

---

## 12. Zero-Hallucination Guarantees

### 12.1 Design Principles

1. **No LLM in calculation path** -- All numeric calculations (risk reduction, ROI, tier advancement, LP optimization, SLA deadlines) use deterministic algorithms with Python Decimal arithmetic
2. **ML for ranking only** -- XGBoost/LightGBM rank pre-defined strategy templates; they never generate strategies, numeric values, or compliance determinations
3. **Deterministic fallback** -- When ML confidence < 0.7, system falls back to rule-based decision tree with fixed weights and thresholds
4. **Bit-perfect reproducibility** -- Deterministic mode produces identical outputs for identical inputs across runs, platforms, and deployments
5. **Complete provenance** -- Every recommendation, calculation, and report includes SHA-256 hash chain linking inputs to outputs

### 12.2 Provenance Chain

```python
import hashlib
import json
from datetime import datetime

def compute_provenance_hash(
    input_data: Dict,
    computation_type: str,
    model_version: str,
    timestamp: datetime,
) -> str:
    """
    SHA-256 provenance hash for audit trail.
    Deterministic: same inputs always produce same hash.
    """
    payload = json.dumps({
        "input_hash": hashlib.sha256(
            json.dumps(input_data, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "computation_type": computation_type,
        "model_version": model_version,
        "timestamp": timestamp.isoformat(),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
```

### 12.3 Validation Rules

| Rule | Enforcement |
|------|-------------|
| Risk scores must be Decimal(5,2), range 0.00-100.00 | Pydantic validator |
| ROI must use Decimal(10,2) | Pydantic validator |
| Budget amounts must use Decimal(18,2) | Pydantic validator |
| Percentage reductions must be Decimal(5,2), range 0.00-100.00 | Pydantic validator |
| All provenance hashes must be 64-character hex strings | Pydantic regex validator |
| Plan dates must be valid ISO 8601 | Pydantic date validator |
| LP solver must return OPTIMAL status or raise error | Solver status check |
| SHAP values must be finite floats (no NaN/Inf) | Post-inference validation |

---

## 13. Testing Strategy

### 13.1 Test Categories

| Category | Test Count | Description |
|----------|-----------|-------------|
| Strategy Selector Unit Tests | 120+ | ML recommendation, deterministic fallback, SHAP, feature engineering, edge cases |
| Remediation Plan Tests | 80+ | Plan generation, milestone creation, Gantt data, template cloning, versioning |
| Capacity Building Tests | 70+ | Enrollment, tier progression, competency scoring, scorecard generation |
| Mitigation Measure Library Tests | 60+ | Full-text search, faceted filtering, comparison, packaging |
| Effectiveness Tracking Tests | 90+ | Baseline capture, delta calculation, ROI, statistical significance, feedback |
| Adaptive Management Tests | 80+ | Trigger detection, adjustment recommendation, escalation, alert fatigue |
| Cost-Benefit Optimizer Tests | 70+ | LP optimization, Pareto frontier, sensitivity analysis, constraint handling |
| Stakeholder Collaboration Tests | 60+ | Role-based access, messaging, task assignment, supplier portal |
| Reporting Tests | 70+ | All 7 report types, 5 formats, multi-language, provenance hashes |
| API Route Tests | 90+ | All ~43 endpoints, auth, error handling, pagination, rate limiting |
| Integration Tests | 40+ | Cross-agent integration with EUDR-016 through 024 |
| Performance Tests | 25+ | Latency targets, batch processing, optimization solver timing |
| ML Pipeline Tests | 30+ | Model accuracy, training pipeline, feature engineering, model versioning |
| Golden Scenario Tests | 35+ | End-to-end mitigation scenarios for all 7 commodities and 8 risk categories |
| Determinism Tests | 15+ | Bit-perfect reproducibility for all deterministic paths |
| Security Tests | 10+ | RBAC enforcement, tenant isolation, rate limiting |
| **Total** | **945+** | |

### 13.2 Golden Test Scenarios

| # | Scenario | Risk Source | Expected Outcome |
|---|----------|-------------|-----------------|
| 1 | High-risk country cocoa sourcing | EUDR-016 (high country risk) | Enhanced monitoring + landscape intervention recommended |
| 2 | Supplier with critical audit findings | EUDR-024 (critical NC) | CAR-linked remediation plan with 2-week SLA |
| 3 | Deforestation alert on palm oil plot | EUDR-020 (critical alert) | Emergency response: suspend + investigate + remediate |
| 4 | Indigenous territory overlap | EUDR-021 (direct overlap) | FPIC remediation workflow with community engagement |
| 5 | Protected area buffer violation | EUDR-022 (buffer encroachment) | Buffer restoration plan + encroachment prevention |
| 6 | Legal compliance gap (missing permits) | EUDR-023 (permit gap) | Legal gap closure plan with permit acquisition |
| 7 | High corruption risk origin | EUDR-019 (high CPI risk) | Anti-corruption measures + third-party verification |
| 8 | Multi-dimensional risk (3+ dimensions) | EUDR-016/017/018 | Composite strategy addressing all dimensions |
| 9 | Budget-constrained optimization | All agents | Optimal allocation across 50 suppliers |
| 10 | Adaptive management trigger cascade | EUDR-020 event | Plan acceleration + scope expansion |
| 11 | Supplier capacity building Tier 1->2 | EUDR-017 + E3 | Tier advancement after module completion |
| 12 | Effectiveness ROI positive | E5 | ROI > 3:1 after 6-month mitigation |
| 13 | Deterministic fallback (low confidence) | E1 | Rule-based recommendation matches expected output |
| 14 | DDS report generation | E1-E8 | Complete Article 12(2)(d) mitigation section |
| 15 | Supplier portal self-service | E8 | Supplier views own plan, uploads evidence |

### 13.3 Test Coverage Targets

| Coverage Type | Target |
|---------------|--------|
| Line coverage | >= 85% |
| Branch coverage | >= 80% |
| Integration test coverage | >= 70% |
| API endpoint coverage | 100% |
| Report format coverage | 100% (all 5 formats) |
| Risk category coverage | 100% (all 8 categories) |
| Deterministic path coverage | 100% |

---

## 14. Deployment Architecture

### 14.1 Containerization

```yaml
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY greenlang/agents/eudr/risk_mitigation_advisor/ ./greenlang/agents/eudr/risk_mitigation_advisor/
COPY greenlang/infrastructure/ ./greenlang/infrastructure/
CMD ["uvicorn", "greenlang.agents.eudr.risk_mitigation_advisor.api.router:app", "--host", "0.0.0.0", "--port", "8025"]
EXPOSE 8025
HEALTHCHECK CMD curl -f http://localhost:8025/health || exit 1
```

### 14.2 Kubernetes Resources

| Resource | Configuration |
|----------|--------------|
| Deployment | 2-4 replicas (HPA: CPU 70%, Memory 80%) |
| Service | ClusterIP, port 8025 |
| Ingress | `/v1/eudr-rma` path routing via Kong API Gateway |
| ConfigMap | Non-sensitive configuration (`GL_EUDR_RMA_*` env vars) |
| Secret | Database credentials, Redis URL, S3 keys, ML model paths |
| PVC | ML model cache (1 GB) |
| Resource Limits | CPU: 2 (request) / 8 (limit), Memory: 4Gi (request) / 16Gi (limit) |
| Readiness Probe | HTTP GET `/health`, period 10s, timeout 5s |
| Liveness Probe | HTTP GET `/health`, period 30s, timeout 10s |

### 14.3 Resource Requirements

| Resource | Development | Staging | Production |
|----------|-------------|---------|------------|
| CPU | 2 vCPU | 4 vCPU | 8 vCPU |
| Memory | 4 GB | 8 GB | 16 GB |
| ML Model Memory | 512 MB | 1 GB | 2 GB |
| Database Storage | 10 GB | 50 GB | 500 GB |
| S3 Storage | 5 GB | 25 GB | 250 GB |

### 14.4 CI/CD Pipeline

```
Push to branch -> GitHub Actions:
    1. Lint (ruff, mypy)
    2. Unit tests (pytest, 945+ tests)
    3. Integration tests (Docker Compose with PostgreSQL + Redis)
    4. Performance tests (locust benchmarks)
    5. Security scan (Bandit, safety)
    6. Build Docker image
    7. Push to ECR
    8. Deploy to staging (K8s rolling update)
    9. Smoke tests on staging
    10. Manual approval gate
    11. Deploy to production (K8s rolling update)
```

---

## 15. Risks and Mitigations

| # | Risk | Probability | Impact | Mitigation |
|---|------|------------|--------|------------|
| 1 | ML model accuracy below 80% target | Medium | High | Deterministic fallback guarantees functional operation; expand training data |
| 2 | LP solver timeout for large portfolios (>500 suppliers) | Low | Medium | Timeout at 30s; decompose into sub-problems; heuristic pre-filtering |
| 3 | Upstream agent unavailability | Medium | High | Cached last-known risk scores with staleness indicator; graceful degradation |
| 4 | Database performance degradation with 10M+ effectiveness records | Low | Medium | TimescaleDB hypertable with retention policy; continuous aggregates for queries |
| 5 | Supplier portal adoption below 70% target | Medium | Medium | Mobile-responsive design; multi-language support; simplified UX |
| 6 | EUDR regulatory changes invalidate measure library | Low | High | Quarterly measure library review cycle; version-controlled measures |
| 7 | Alert fatigue from excessive trigger events | Medium | Medium | Configurable quiet periods; trigger consolidation; priority filtering |
| 8 | SHAP computation overhead for real-time recommendations | Low | Medium | Pre-compute SHAP for common patterns; cache explanations |
| 9 | Cross-agent integration complexity (9 agents) | Medium | High | Adapter pattern with circuit breakers; comprehensive integration tests |
| 10 | Multi-tenant data leakage in collaboration hub | Low | Critical | operator_id filter at repository layer; penetration testing; RBAC enforcement |

---

## 16. Appendices

### Appendix A: Technology Stack Summary

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.11+ |
| Web Framework | FastAPI | 0.104.0+ |
| Server | Uvicorn | 0.24.0+ |
| Data Validation | Pydantic | 2.5.0+ |
| ML (Primary) | XGBoost | 2.0.0+ |
| ML (Secondary) | LightGBM | 4.1.0+ |
| Explainability | SHAP | 0.43.0+ |
| Optimization | PuLP | 2.7.0+ |
| Statistics | SciPy | 1.11.0+ |
| Numeric | NumPy | 1.24.0+ |
| Database | PostgreSQL + TimescaleDB | 14+ / 2.12+ |
| ORM | SQLAlchemy | 2.0.0+ |
| Async DB | psycopg + psycopg_pool | 3.1+ |
| Cache | Redis | 7.0+ |
| Object Storage | S3 (boto3) | Latest |
| PDF Generation | WeasyPrint + Jinja2 | 60.0+ |
| Spreadsheet | openpyxl | 3.1.0+ |
| XML | lxml | 4.9.0+ |
| Auth | python-jose (RS256) | 3.3.0+ |
| Encryption | cryptography | 41.0.0+ |
| Monitoring | prometheus-client | 0.18.0+ |
| Tracing | opentelemetry-sdk | 1.21.0+ |
| Testing | pytest + pytest-asyncio | 7.4+ / 0.23+ |
| Containers | Docker | 24+ |
| Orchestration | Kubernetes (EKS) | 1.28+ |
| CI/CD | GitHub Actions | Latest |

### Appendix B: Configuration Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GL_EUDR_RMA_DB_URL` | (required) | PostgreSQL connection string |
| `GL_EUDR_RMA_REDIS_URL` | (required) | Redis connection string |
| `GL_EUDR_RMA_S3_BUCKET` | `gl-eudr-rma` | S3 bucket for evidence and reports |
| `GL_EUDR_RMA_ML_MODEL_PATH` | `s3://gl-models/eudr-rma/` | ML model artifact path |
| `GL_EUDR_RMA_ML_CONFIDENCE_THRESHOLD` | `0.7` | Minimum ML confidence for recommendation |
| `GL_EUDR_RMA_EFFECTIVENESS_INTERVAL_DAYS` | `30` | Effectiveness measurement interval |
| `GL_EUDR_RMA_QUIET_PERIOD_HOURS` | `24` | Alert fatigue quiet period |
| `GL_EUDR_RMA_OPTIMIZATION_TIMEOUT_S` | `30` | LP solver timeout |
| `GL_EUDR_RMA_CACHE_TTL_STRATEGY_S` | `3600` | Strategy recommendation cache TTL |
| `GL_EUDR_RMA_CACHE_TTL_MEASURE_S` | `7200` | Measure library cache TTL |
| `GL_EUDR_RMA_RATE_LIMIT_DEFAULT` | `100` | Default rate limit (requests/minute) |
| `GL_EUDR_RMA_RATE_LIMIT_ML` | `30` | ML inference rate limit |
| `GL_EUDR_RMA_RATE_LIMIT_OPTIMIZE` | `10` | Optimization rate limit |
| `GL_EUDR_RMA_TOKEN_EXPIRY_S` | `3600` | JWT token expiry |
| `GL_EUDR_RMA_LOG_LEVEL` | `INFO` | Logging level |

### Appendix C: EUDR Article Coverage Matrix

| EUDR Article | Feature | Engine |
|-------------|---------|--------|
| Art. 8(1) Due diligence system | Full pipeline integration | E1-E8 |
| Art. 8(3) Annual review | Annual review automation | E6 |
| Art. 10(1) Risk assessment consumption | 9-agent risk input | E1 |
| Art. 10(2)(a) Supply chain complexity | Complexity mitigation | E1, E4 |
| Art. 10(2)(b) Circumvention risk | Anti-circumvention measures | E4 |
| Art. 10(2)(c-d) Country risk | Country-specific strategies | E1, E4, E6 |
| Art. 10(2)(e) Corruption risk | Anti-corruption measures | E1, E4 |
| Art. 10(2)(f) Supplier risk | Supplier remediation | E1, E2, E3 |
| Art. 11(1) Risk mitigation adequacy | Strategy selection + tracking | E1, E5 |
| Art. 11(2)(a) Additional information | Supplier information workflows | E3, E8 |
| Art. 11(2)(b) Independent audits | EUDR-024 integration | E1, E6 |
| Art. 11(2)(c) Other measures | 500+ measure library | E4 |
| Art. 12(2)(d) DDS mitigation section | DDS report generation | Reporter |
| Art. 29(1) Country benchmarking | Country-specific templates | E1, E4 |
| Art. 31(1) Record keeping 5 years | All data retained 5 years | All tables |

---

**End of Architecture Specification**

*Document version: 1.0.0 | Author: GL-AppArchitect | Date: 2026-03-11*
