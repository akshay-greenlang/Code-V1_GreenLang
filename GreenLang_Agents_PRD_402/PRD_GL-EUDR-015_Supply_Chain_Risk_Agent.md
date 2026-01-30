# PRD: Supply Chain Risk Agent (GL-EUDR-015)

**Agent family:** EUDRTraceabilityFamily
**Layer:** Supply Chain Traceability
**Primary domains:** Risk assessment, supply chain vulnerability, mitigation planning
**Priority:** P0 (highest)
**Doc version:** 1.0
**Last updated:** 2026-01-30 (Asia/Kolkata)

---

## 1. Executive Summary

**Supply Chain Risk Agent (GL-EUDR-015)** assesses and monitors risks throughout the EUDR supply chain. It evaluates supplier risks, geographic risks, traceability gaps, and compliance vulnerabilities to enable proactive risk mitigation and maintain continuous EUDR compliance.

---

## 2. EUDR Risk Framework

### 2.1 Regulatory Context

Per EUDR Article 10, operators must conduct risk assessment considering:
- **Country risk:** Based on EU country benchmarking
- **Product complexity:** Number of supply chain tiers
- **Supplier reliability:** Track record and certifications
- **Information quality:** Completeness of traceability data

### 2.2 Risk Categories

| Category | Description | EUDR Relevance |
|---|---|---|
| **Deforestation Risk** | Risk commodity linked to forest loss | Core EUDR requirement |
| **Legality Risk** | Risk of illegal production/trade | Article 3 compliance |
| **Traceability Risk** | Risk of broken supply chain visibility | Article 9 compliance |
| **Supplier Risk** | Risk from supplier non-compliance | Due diligence obligation |
| **Geographic Risk** | Country/region-specific risks | Article 29 benchmarking |
| **Documentation Risk** | Incomplete/invalid documentation | Evidence requirements |

### 2.3 EU Country Benchmarking

Per EUDR Article 29, countries are classified as:
- **Low Risk:** Simplified due diligence permitted
- **Standard Risk:** Full due diligence required
- **High Risk:** Enhanced due diligence required

---

## 3. Data Model

```sql
-- Risk Profiles
CREATE TABLE risk_profiles (
    profile_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(100) NOT NULL,
    entity_id UUID NOT NULL,

    -- Overall Risk
    overall_risk_score DECIMAL(5,2) NOT NULL,
    risk_level VARCHAR(50) NOT NULL,

    -- Category Scores
    deforestation_risk_score DECIMAL(5,2),
    legality_risk_score DECIMAL(5,2),
    traceability_risk_score DECIMAL(5,2),
    supplier_risk_score DECIMAL(5,2),
    geographic_risk_score DECIMAL(5,2),
    documentation_risk_score DECIMAL(5,2),

    -- Risk Factors
    risk_factors JSONB DEFAULT '[]',
    mitigating_factors JSONB DEFAULT '[]',

    -- Assessment
    assessment_date TIMESTAMP NOT NULL,
    assessment_method VARCHAR(100) NOT NULL,
    assessed_by VARCHAR(255),

    -- Validity
    valid_until DATE,
    requires_reassessment BOOLEAN DEFAULT FALSE,
    reassessment_triggers JSONB DEFAULT '[]',

    -- Status
    is_current BOOLEAN DEFAULT TRUE,
    superseded_by UUID,

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_entity_type CHECK (
        entity_type IN (
            'SUPPLIER', 'FACILITY', 'ORIGIN_COUNTRY', 'REGION',
            'SUPPLY_CHAIN', 'BATCH', 'COMMODITY', 'OPERATOR'
        )
    ),
    CONSTRAINT valid_risk_level CHECK (
        risk_level IN ('LOW', 'STANDARD', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT valid_score_range CHECK (
        overall_risk_score >= 0 AND overall_risk_score <= 100
    )
);

-- Risk Factors
CREATE TABLE risk_factors (
    factor_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile_id UUID REFERENCES risk_profiles(profile_id),

    -- Factor Details
    factor_category VARCHAR(100) NOT NULL,
    factor_type VARCHAR(100) NOT NULL,
    factor_name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Scoring
    severity VARCHAR(50) NOT NULL,
    likelihood VARCHAR(50) NOT NULL,
    impact_score DECIMAL(5,2) NOT NULL,
    weight DECIMAL(3,2) DEFAULT 1.0,

    -- Evidence
    evidence_source VARCHAR(255),
    evidence_date DATE,
    evidence_reference VARCHAR(255),

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    mitigated BOOLEAN DEFAULT FALSE,
    mitigation_id UUID,

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_severity CHECK (
        severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT valid_likelihood CHECK (
        likelihood IN ('RARE', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'ALMOST_CERTAIN')
    )
);

-- Country Risk Classifications
CREATE TABLE country_risk_classifications (
    classification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country_code CHAR(2) NOT NULL,
    country_name VARCHAR(255) NOT NULL,

    -- EU Benchmarking
    eu_benchmark_status VARCHAR(50),  -- LOW, STANDARD, HIGH (per Article 29)
    benchmark_effective_date DATE,

    -- Risk Scores by Category
    deforestation_risk VARCHAR(50) NOT NULL,
    governance_risk VARCHAR(50) NOT NULL,
    corruption_risk VARCHAR(50) NOT NULL,
    enforcement_risk VARCHAR(50) NOT NULL,

    -- Overall
    overall_risk VARCHAR(50) NOT NULL,

    -- Data Sources
    data_sources JSONB DEFAULT '[]',
    last_updated DATE NOT NULL,

    -- Commodity-Specific
    commodity_risks JSONB DEFAULT '{}',  -- {PALM_OIL: HIGH, COCOA: MEDIUM, ...}

    -- Notes
    notes TEXT,

    UNIQUE(country_code),

    CONSTRAINT valid_risk CHECK (
        overall_risk IN ('LOW', 'STANDARD', 'HIGH')
    )
);

-- Risk Assessments
CREATE TABLE risk_assessments (
    assessment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_reference VARCHAR(100) UNIQUE NOT NULL,

    -- Scope
    assessment_type VARCHAR(100) NOT NULL,
    scope_description TEXT,

    -- Target
    target_entity_type VARCHAR(100) NOT NULL,
    target_entity_ids UUID[],
    supplier_ids UUID[],
    country_codes CHAR(2)[],
    commodity_categories VARCHAR(50)[],

    -- Timeline
    initiated_date TIMESTAMP NOT NULL,
    completed_date TIMESTAMP,
    valid_until DATE,

    -- Methodology
    assessment_methodology VARCHAR(100) NOT NULL,
    data_sources JSONB DEFAULT '[]',
    automated_checks JSONB DEFAULT '[]',
    manual_reviews JSONB DEFAULT '[]',

    -- Results
    status VARCHAR(50) DEFAULT 'IN_PROGRESS',
    overall_risk_level VARCHAR(50),
    risk_summary JSONB,
    findings JSONB DEFAULT '[]',

    -- Recommendations
    recommendations JSONB DEFAULT '[]',
    required_mitigations JSONB DEFAULT '[]',

    -- Sign-off
    reviewed_by VARCHAR(255),
    approved_by VARCHAR(255),
    approval_date TIMESTAMP,

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_assessment_type CHECK (
        assessment_type IN (
            'SUPPLIER_ONBOARDING', 'PERIODIC_REVIEW', 'TRIGGERED_REVIEW',
            'COUNTRY_ASSESSMENT', 'SUPPLY_CHAIN_MAPPING', 'DDS_PREPARATION'
        )
    ),
    CONSTRAINT valid_status CHECK (
        status IN ('IN_PROGRESS', 'REVIEW', 'COMPLETED', 'EXPIRED', 'CANCELLED')
    )
);

-- Risk Mitigation Actions
CREATE TABLE risk_mitigations (
    mitigation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id UUID REFERENCES risk_assessments(assessment_id),
    profile_id UUID REFERENCES risk_profiles(profile_id),

    -- Target
    risk_factor_ids UUID[],
    target_risk_level VARCHAR(50),

    -- Action
    mitigation_type VARCHAR(100) NOT NULL,
    action_description TEXT NOT NULL,
    priority VARCHAR(50) NOT NULL,

    -- Timeline
    due_date DATE,
    started_date TIMESTAMP,
    completed_date TIMESTAMP,

    -- Assignment
    assigned_to VARCHAR(255),
    responsible_party VARCHAR(255),

    -- Progress
    status VARCHAR(50) DEFAULT 'PENDING',
    progress_percentage INTEGER DEFAULT 0,
    progress_notes TEXT,

    -- Effectiveness
    effectiveness_verified BOOLEAN DEFAULT FALSE,
    verification_date TIMESTAMP,
    residual_risk_level VARCHAR(50),

    -- Evidence
    evidence_documents UUID[],

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_mitigation_type CHECK (
        mitigation_type IN (
            'SUPPLIER_AUDIT', 'DOCUMENTATION_REQUEST', 'CERTIFICATION_REQUIREMENT',
            'ENHANCED_MONITORING', 'SUPPLIER_TERMINATION', 'ALTERNATIVE_SOURCING',
            'PROCESS_IMPROVEMENT', 'TRAINING', 'TECHNOLOGY_IMPLEMENTATION'
        )
    ),
    CONSTRAINT valid_priority CHECK (
        priority IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT valid_status CHECK (
        status IN ('PENDING', 'IN_PROGRESS', 'COMPLETED', 'OVERDUE', 'CANCELLED')
    )
);

-- Risk Alerts
CREATE TABLE risk_alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Alert Details
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    title VARCHAR(500) NOT NULL,
    description TEXT,

    -- Target
    entity_type VARCHAR(100) NOT NULL,
    entity_id UUID NOT NULL,
    related_entities JSONB DEFAULT '[]',

    -- Trigger
    trigger_source VARCHAR(100) NOT NULL,
    trigger_details JSONB,
    triggered_at TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Status
    status VARCHAR(50) DEFAULT 'OPEN',
    acknowledged_at TIMESTAMP,
    acknowledged_by VARCHAR(255),
    resolved_at TIMESTAMP,
    resolved_by VARCHAR(255),
    resolution_notes TEXT,

    -- Actions
    recommended_actions JSONB DEFAULT '[]',
    actions_taken JSONB DEFAULT '[]',

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_alert_type CHECK (
        alert_type IN (
            'DEFORESTATION_DETECTED', 'SUPPLIER_SANCTIONED', 'CERTIFICATION_EXPIRED',
            'TRACEABILITY_GAP', 'COUNTRY_RISK_CHANGE', 'COMPLIANCE_VIOLATION',
            'DOCUMENTATION_MISSING', 'THRESHOLD_EXCEEDED', 'ANOMALY_DETECTED'
        )
    ),
    CONSTRAINT valid_alert_severity CHECK (
        severity IN ('INFO', 'WARNING', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT valid_alert_status CHECK (
        status IN ('OPEN', 'ACKNOWLEDGED', 'IN_PROGRESS', 'RESOLVED', 'ESCALATED')
    )
);

-- Indexes
CREATE INDEX idx_risk_profiles_entity ON risk_profiles(entity_type, entity_id);
CREATE INDEX idx_risk_profiles_level ON risk_profiles(risk_level);
CREATE INDEX idx_risk_profiles_current ON risk_profiles(is_current) WHERE is_current = TRUE;
CREATE INDEX idx_risk_factors_profile ON risk_factors(profile_id);
CREATE INDEX idx_risk_factors_category ON risk_factors(factor_category);
CREATE INDEX idx_country_risk_code ON country_risk_classifications(country_code);
CREATE INDEX idx_country_risk_level ON country_risk_classifications(overall_risk);
CREATE INDEX idx_assessments_status ON risk_assessments(status);
CREATE INDEX idx_assessments_type ON risk_assessments(assessment_type);
CREATE INDEX idx_mitigations_status ON risk_mitigations(status);
CREATE INDEX idx_mitigations_due ON risk_mitigations(due_date) WHERE status NOT IN ('COMPLETED', 'CANCELLED');
CREATE INDEX idx_alerts_status ON risk_alerts(status);
CREATE INDEX idx_alerts_severity ON risk_alerts(severity);
CREATE INDEX idx_alerts_entity ON risk_alerts(entity_type, entity_id);
```

---

## 4. Functional Requirements

### 4.1 Risk Assessment
- **FR-001 (P0):** Assess supplier risk based on multiple factors
- **FR-002 (P0):** Calculate geographic/country risk per EU benchmarking
- **FR-003 (P0):** Evaluate traceability completeness risk
- **FR-004 (P0):** Assess deforestation risk using satellite data
- **FR-005 (P0):** Generate overall risk scores and levels

### 4.2 Country Risk Management
- **FR-010 (P0):** Maintain country risk classifications
- **FR-011 (P0):** Apply EU benchmarking status when published
- **FR-012 (P0):** Track commodity-specific country risks
- **FR-013 (P1):** Monitor country risk changes

### 4.3 Risk Monitoring
- **FR-020 (P0):** Continuous monitoring of risk indicators
- **FR-021 (P0):** Generate alerts for risk threshold breaches
- **FR-022 (P0):** Detect anomalies in supply chain patterns
- **FR-023 (P0):** Track certification expirations
- **FR-024 (P1):** Monitor sanctions lists

### 4.4 Mitigation Management
- **FR-030 (P0):** Create mitigation action plans
- **FR-031 (P0):** Track mitigation progress
- **FR-032 (P0):** Verify mitigation effectiveness
- **FR-033 (P0):** Escalate overdue mitigations

### 4.5 Reporting
- **FR-040 (P0):** Generate risk assessment reports
- **FR-041 (P0):** Provide risk dashboards
- **FR-042 (P0):** Export risk data for DDS
- **FR-043 (P1):** Trend analysis and forecasting

---

## 5. Risk Assessment Engine

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
import math


class RiskLevel(Enum):
    LOW = "LOW"
    STANDARD = "STANDARD"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskCategory(Enum):
    DEFORESTATION = "DEFORESTATION"
    LEGALITY = "LEGALITY"
    TRACEABILITY = "TRACEABILITY"
    SUPPLIER = "SUPPLIER"
    GEOGRAPHIC = "GEOGRAPHIC"
    DOCUMENTATION = "DOCUMENTATION"


class Likelihood(Enum):
    RARE = 1
    UNLIKELY = 2
    POSSIBLE = 3
    LIKELY = 4
    ALMOST_CERTAIN = 5


class Severity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class RiskFactor:
    factor_id: UUID
    category: RiskCategory
    factor_type: str
    name: str
    description: str
    severity: Severity
    likelihood: Likelihood
    weight: Decimal = Decimal("1.0")
    evidence_source: Optional[str] = None
    mitigated: bool = False

    @property
    def impact_score(self) -> Decimal:
        """Calculate impact score from severity and likelihood."""
        base_score = self.severity.value * self.likelihood.value
        # Normalize to 0-100 scale (max is 4 * 5 = 20)
        return Decimal(str(base_score / 20 * 100)) * self.weight


@dataclass
class RiskProfile:
    profile_id: UUID
    entity_type: str
    entity_id: UUID
    overall_risk_score: Decimal
    risk_level: RiskLevel
    category_scores: Dict[RiskCategory, Decimal]
    risk_factors: List[RiskFactor]
    mitigating_factors: List[Dict]
    assessment_date: datetime
    valid_until: date


@dataclass
class RiskAssessmentResult:
    assessment_id: UUID
    target_entity_type: str
    target_entity_id: UUID
    overall_risk_score: Decimal
    risk_level: RiskLevel
    category_scores: Dict[RiskCategory, Decimal]
    risk_factors: List[RiskFactor]
    recommendations: List[Dict]
    required_mitigations: List[Dict]


class SupplyChainRiskEngine:
    """
    Comprehensive risk assessment engine for EUDR supply chains.
    """

    # Category weights for overall score calculation
    CATEGORY_WEIGHTS = {
        RiskCategory.DEFORESTATION: Decimal("0.30"),
        RiskCategory.LEGALITY: Decimal("0.20"),
        RiskCategory.TRACEABILITY: Decimal("0.15"),
        RiskCategory.SUPPLIER: Decimal("0.15"),
        RiskCategory.GEOGRAPHIC: Decimal("0.10"),
        RiskCategory.DOCUMENTATION: Decimal("0.10")
    }

    # Risk level thresholds
    RISK_THRESHOLDS = {
        RiskLevel.LOW: (Decimal("0"), Decimal("25")),
        RiskLevel.STANDARD: (Decimal("25"), Decimal("50")),
        RiskLevel.HIGH: (Decimal("50"), Decimal("75")),
        RiskLevel.CRITICAL: (Decimal("75"), Decimal("100"))
    }

    def __init__(
        self,
        db_session,
        deforestation_service,
        supplier_service,
        country_risk_service,
        traceability_service
    ):
        self.db = db_session
        self.deforestation_service = deforestation_service
        self.supplier_service = supplier_service
        self.country_risk_service = country_risk_service
        self.traceability_service = traceability_service

    def assess_supplier_risk(
        self,
        supplier_id: UUID,
        commodity_category: str,
        assessment_depth: str = "STANDARD"
    ) -> RiskAssessmentResult:
        """
        Perform comprehensive risk assessment for a supplier.

        Args:
            supplier_id: The supplier to assess
            commodity_category: Primary commodity category
            assessment_depth: QUICK, STANDARD, or ENHANCED

        Returns:
            RiskAssessmentResult with scores and recommendations
        """
        supplier = self.supplier_service.get_supplier(supplier_id)
        risk_factors = []

        # 1. Geographic Risk Assessment
        geo_factors = self._assess_geographic_risk(
            supplier.country_code,
            supplier.region,
            commodity_category
        )
        risk_factors.extend(geo_factors)

        # 2. Deforestation Risk Assessment
        deforestation_factors = self._assess_deforestation_risk(
            supplier_id,
            commodity_category
        )
        risk_factors.extend(deforestation_factors)

        # 3. Supplier Profile Risk
        supplier_factors = self._assess_supplier_profile_risk(supplier)
        risk_factors.extend(supplier_factors)

        # 4. Traceability Risk
        traceability_factors = self._assess_traceability_risk(supplier_id)
        risk_factors.extend(traceability_factors)

        # 5. Documentation Risk
        doc_factors = self._assess_documentation_risk(supplier_id)
        risk_factors.extend(doc_factors)

        # 6. Legality Risk
        legality_factors = self._assess_legality_risk(supplier_id, supplier.country_code)
        risk_factors.extend(legality_factors)

        # Calculate category scores
        category_scores = self._calculate_category_scores(risk_factors)

        # Calculate overall score
        overall_score = self._calculate_overall_score(category_scores)

        # Determine risk level
        risk_level = self._determine_risk_level(overall_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            risk_level,
            risk_factors,
            category_scores
        )

        # Identify required mitigations
        required_mitigations = self._identify_required_mitigations(
            risk_factors,
            risk_level
        )

        return RiskAssessmentResult(
            assessment_id=self._generate_uuid(),
            target_entity_type="SUPPLIER",
            target_entity_id=supplier_id,
            overall_risk_score=overall_score,
            risk_level=risk_level,
            category_scores=category_scores,
            risk_factors=risk_factors,
            recommendations=recommendations,
            required_mitigations=required_mitigations
        )

    def _assess_geographic_risk(
        self,
        country_code: str,
        region: Optional[str],
        commodity_category: str
    ) -> List[RiskFactor]:
        """
        Assess geographic and country-level risks.
        """
        factors = []

        # Get country risk classification
        country_risk = self.country_risk_service.get_classification(country_code)

        if country_risk:
            # Overall country risk
            if country_risk.overall_risk == "HIGH":
                factors.append(RiskFactor(
                    factor_id=self._generate_uuid(),
                    category=RiskCategory.GEOGRAPHIC,
                    factor_type="COUNTRY_CLASSIFICATION",
                    name="High-Risk Country",
                    description=f"{country_risk.country_name} classified as high-risk for EUDR",
                    severity=Severity.HIGH,
                    likelihood=Likelihood.LIKELY,
                    weight=Decimal("1.5"),
                    evidence_source="EU_COUNTRY_BENCHMARKING"
                ))
            elif country_risk.overall_risk == "STANDARD":
                factors.append(RiskFactor(
                    factor_id=self._generate_uuid(),
                    category=RiskCategory.GEOGRAPHIC,
                    factor_type="COUNTRY_CLASSIFICATION",
                    name="Standard-Risk Country",
                    description=f"{country_risk.country_name} requires standard due diligence",
                    severity=Severity.MEDIUM,
                    likelihood=Likelihood.POSSIBLE,
                    evidence_source="EU_COUNTRY_BENCHMARKING"
                ))

            # Commodity-specific country risk
            commodity_risk = country_risk.commodity_risks.get(commodity_category)
            if commodity_risk == "HIGH":
                factors.append(RiskFactor(
                    factor_id=self._generate_uuid(),
                    category=RiskCategory.GEOGRAPHIC,
                    factor_type="COMMODITY_COUNTRY_RISK",
                    name=f"High {commodity_category} Risk in Country",
                    description=f"{commodity_category} production in {country_code} has elevated risk",
                    severity=Severity.HIGH,
                    likelihood=Likelihood.LIKELY,
                    evidence_source="COMMODITY_ANALYSIS"
                ))

            # Governance risk
            if country_risk.governance_risk == "HIGH":
                factors.append(RiskFactor(
                    factor_id=self._generate_uuid(),
                    category=RiskCategory.LEGALITY,
                    factor_type="GOVERNANCE_RISK",
                    name="Weak Governance",
                    description="Country has weak forest governance indicators",
                    severity=Severity.MEDIUM,
                    likelihood=Likelihood.LIKELY,
                    evidence_source="FOREST_GOVERNANCE_INDEX"
                ))

        return factors

    def _assess_deforestation_risk(
        self,
        supplier_id: UUID,
        commodity_category: str
    ) -> List[RiskFactor]:
        """
        Assess deforestation-related risks.
        """
        factors = []

        # Get supplier's plots
        plots = self.supplier_service.get_supplier_plots(supplier_id)

        if not plots:
            factors.append(RiskFactor(
                factor_id=self._generate_uuid(),
                category=RiskCategory.DEFORESTATION,
                factor_type="NO_PLOTS_REGISTERED",
                name="No Production Plots Registered",
                description="Supplier has no registered production plots",
                severity=Severity.CRITICAL,
                likelihood=Likelihood.ALMOST_CERTAIN,
                evidence_source="PLOT_REGISTRY"
            ))
            return factors

        for plot in plots:
            # Check deforestation verification
            if not plot.deforestation_status_verified:
                factors.append(RiskFactor(
                    factor_id=self._generate_uuid(),
                    category=RiskCategory.DEFORESTATION,
                    factor_type="UNVERIFIED_DEFORESTATION",
                    name="Unverified Deforestation Status",
                    description=f"Plot {plot.plot_id} has not been verified for deforestation",
                    severity=Severity.HIGH,
                    likelihood=Likelihood.POSSIBLE,
                    evidence_source="PLOT_REGISTRY"
                ))

            # Check forest proximity
            proximity = self.deforestation_service.check_forest_proximity(plot.coordinates)
            if proximity and proximity.distance_km < 5:
                factors.append(RiskFactor(
                    factor_id=self._generate_uuid(),
                    category=RiskCategory.DEFORESTATION,
                    factor_type="FOREST_PROXIMITY",
                    name="Near Forest Area",
                    description=f"Plot within {proximity.distance_km}km of forest area",
                    severity=Severity.MEDIUM,
                    likelihood=Likelihood.POSSIBLE,
                    evidence_source="FOREST_MONITORING"
                ))

            # Check recent deforestation alerts
            alerts = self.deforestation_service.get_alerts_near_plot(
                plot.coordinates,
                radius_km=10,
                since_date=date(2020, 12, 31)  # EUDR cutoff
            )
            if alerts:
                factors.append(RiskFactor(
                    factor_id=self._generate_uuid(),
                    category=RiskCategory.DEFORESTATION,
                    factor_type="DEFORESTATION_ALERTS",
                    name="Deforestation Alerts Nearby",
                    description=f"{len(alerts)} deforestation alerts within 10km since Dec 2020",
                    severity=Severity.HIGH if len(alerts) > 5 else Severity.MEDIUM,
                    likelihood=Likelihood.LIKELY if len(alerts) > 5 else Likelihood.POSSIBLE,
                    evidence_source="GFW_ALERTS"
                ))

        return factors

    def _assess_supplier_profile_risk(self, supplier) -> List[RiskFactor]:
        """
        Assess risks from supplier profile and history.
        """
        factors = []

        # Check verification status
        if supplier.verification_status != "VERIFIED":
            factors.append(RiskFactor(
                factor_id=self._generate_uuid(),
                category=RiskCategory.SUPPLIER,
                factor_type="UNVERIFIED_SUPPLIER",
                name="Unverified Supplier",
                description="Supplier identity/business has not been verified",
                severity=Severity.HIGH,
                likelihood=Likelihood.POSSIBLE,
                evidence_source="SUPPLIER_REGISTRY"
            ))

        # Check certifications
        valid_certs = [c for c in supplier.certifications if c.is_valid]
        if not valid_certs:
            factors.append(RiskFactor(
                factor_id=self._generate_uuid(),
                category=RiskCategory.SUPPLIER,
                factor_type="NO_CERTIFICATIONS",
                name="No Valid Certifications",
                description="Supplier holds no valid sustainability certifications",
                severity=Severity.MEDIUM,
                likelihood=Likelihood.POSSIBLE,
                evidence_source="CERTIFICATION_CHECK"
            ))

        # Check expiring certifications
        expiring_soon = [c for c in valid_certs
                        if c.expiry_date and (c.expiry_date - date.today()).days < 90]
        if expiring_soon:
            factors.append(RiskFactor(
                factor_id=self._generate_uuid(),
                category=RiskCategory.SUPPLIER,
                factor_type="EXPIRING_CERTIFICATIONS",
                name="Certifications Expiring Soon",
                description=f"{len(expiring_soon)} certifications expiring within 90 days",
                severity=Severity.LOW,
                likelihood=Likelihood.ALMOST_CERTAIN,
                evidence_source="CERTIFICATION_CHECK"
            ))

        # Check sanctions
        if supplier.sanctions_status == "SANCTIONED":
            factors.append(RiskFactor(
                factor_id=self._generate_uuid(),
                category=RiskCategory.LEGALITY,
                factor_type="SANCTIONED_ENTITY",
                name="Sanctioned Supplier",
                description="Supplier appears on sanctions list",
                severity=Severity.CRITICAL,
                likelihood=Likelihood.ALMOST_CERTAIN,
                evidence_source="SANCTIONS_SCREENING"
            ))

        # Check business age
        if supplier.registration_date:
            years_in_business = (date.today() - supplier.registration_date).days / 365
            if years_in_business < 2:
                factors.append(RiskFactor(
                    factor_id=self._generate_uuid(),
                    category=RiskCategory.SUPPLIER,
                    factor_type="NEW_SUPPLIER",
                    name="New Business",
                    description=f"Supplier operating for less than 2 years",
                    severity=Severity.LOW,
                    likelihood=Likelihood.POSSIBLE,
                    evidence_source="BUSINESS_REGISTRY"
                ))

        return factors

    def _assess_traceability_risk(self, supplier_id: UUID) -> List[RiskFactor]:
        """
        Assess traceability chain completeness risks.
        """
        factors = []

        # Get traceability metrics
        metrics = self.traceability_service.get_supplier_metrics(supplier_id)

        if metrics:
            # Plot coverage
            if metrics.plot_coverage < 100:
                factors.append(RiskFactor(
                    factor_id=self._generate_uuid(),
                    category=RiskCategory.TRACEABILITY,
                    factor_type="INCOMPLETE_PLOT_COVERAGE",
                    name="Incomplete Plot Coverage",
                    description=f"Only {metrics.plot_coverage}% of volume traced to plots",
                    severity=Severity.HIGH if metrics.plot_coverage < 80 else Severity.MEDIUM,
                    likelihood=Likelihood.ALMOST_CERTAIN,
                    evidence_source="TRACEABILITY_ANALYSIS"
                ))

            # Custody chain gaps
            if metrics.custody_gaps > 0:
                factors.append(RiskFactor(
                    factor_id=self._generate_uuid(),
                    category=RiskCategory.TRACEABILITY,
                    factor_type="CUSTODY_GAPS",
                    name="Chain of Custody Gaps",
                    description=f"{metrics.custody_gaps} gaps detected in custody chain",
                    severity=Severity.HIGH,
                    likelihood=Likelihood.ALMOST_CERTAIN,
                    evidence_source="CHAIN_ANALYSIS"
                ))

            # Multi-tier visibility
            if metrics.tier_visibility < 3:
                factors.append(RiskFactor(
                    factor_id=self._generate_uuid(),
                    category=RiskCategory.TRACEABILITY,
                    factor_type="LIMITED_VISIBILITY",
                    name="Limited Supply Chain Visibility",
                    description=f"Visibility limited to {metrics.tier_visibility} tiers",
                    severity=Severity.MEDIUM,
                    likelihood=Likelihood.LIKELY,
                    evidence_source="SUPPLY_CHAIN_MAP"
                ))

        return factors

    def _assess_documentation_risk(self, supplier_id: UUID) -> List[RiskFactor]:
        """
        Assess documentation completeness risks.
        """
        factors = []

        # Get documentation status
        docs = self.supplier_service.get_supplier_documents(supplier_id)

        required_docs = [
            "BUSINESS_LICENSE",
            "LAND_TITLE",
            "ORIGIN_DECLARATION",
            "TRADE_LICENSE"
        ]

        missing_docs = [d for d in required_docs if d not in [doc.doc_type for doc in docs]]

        if missing_docs:
            factors.append(RiskFactor(
                factor_id=self._generate_uuid(),
                category=RiskCategory.DOCUMENTATION,
                factor_type="MISSING_DOCUMENTS",
                name="Missing Required Documents",
                description=f"Missing: {', '.join(missing_docs)}",
                severity=Severity.HIGH if len(missing_docs) > 2 else Severity.MEDIUM,
                likelihood=Likelihood.ALMOST_CERTAIN,
                evidence_source="DOCUMENT_CHECK"
            ))

        # Check document validity
        expired_docs = [d for d in docs if d.expiry_date and d.expiry_date < date.today()]
        if expired_docs:
            factors.append(RiskFactor(
                factor_id=self._generate_uuid(),
                category=RiskCategory.DOCUMENTATION,
                factor_type="EXPIRED_DOCUMENTS",
                name="Expired Documents",
                description=f"{len(expired_docs)} documents have expired",
                severity=Severity.MEDIUM,
                likelihood=Likelihood.ALMOST_CERTAIN,
                evidence_source="DOCUMENT_CHECK"
            ))

        return factors

    def _assess_legality_risk(
        self,
        supplier_id: UUID,
        country_code: str
    ) -> List[RiskFactor]:
        """
        Assess legal compliance risks.
        """
        factors = []

        # Get country's legal framework risk
        country_risk = self.country_risk_service.get_classification(country_code)

        if country_risk and country_risk.enforcement_risk == "HIGH":
            factors.append(RiskFactor(
                factor_id=self._generate_uuid(),
                category=RiskCategory.LEGALITY,
                factor_type="WEAK_ENFORCEMENT",
                name="Weak Law Enforcement",
                description="Country has weak forest law enforcement",
                severity=Severity.MEDIUM,
                likelihood=Likelihood.LIKELY,
                evidence_source="ENFORCEMENT_INDEX"
            ))

        return factors

    def _calculate_category_scores(
        self,
        factors: List[RiskFactor]
    ) -> Dict[RiskCategory, Decimal]:
        """
        Calculate risk scores per category.
        """
        category_scores = {}

        for category in RiskCategory:
            category_factors = [f for f in factors if f.category == category and not f.mitigated]

            if not category_factors:
                category_scores[category] = Decimal("0")
            else:
                # Sum impact scores, capped at 100
                total_impact = sum(f.impact_score for f in category_factors)
                category_scores[category] = min(total_impact, Decimal("100"))

        return category_scores

    def _calculate_overall_score(
        self,
        category_scores: Dict[RiskCategory, Decimal]
    ) -> Decimal:
        """
        Calculate weighted overall risk score.
        """
        overall = Decimal("0")

        for category, score in category_scores.items():
            weight = self.CATEGORY_WEIGHTS.get(category, Decimal("0.1"))
            overall += score * weight

        return min(overall, Decimal("100"))

    def _determine_risk_level(self, score: Decimal) -> RiskLevel:
        """
        Determine risk level from score.
        """
        for level, (lower, upper) in self.RISK_THRESHOLDS.items():
            if lower <= score < upper:
                return level
        return RiskLevel.CRITICAL

    def _generate_recommendations(
        self,
        risk_level: RiskLevel,
        factors: List[RiskFactor],
        category_scores: Dict[RiskCategory, Decimal]
    ) -> List[Dict]:
        """
        Generate actionable recommendations based on risk assessment.
        """
        recommendations = []

        # Level-based recommendations
        if risk_level == RiskLevel.CRITICAL:
            recommendations.append({
                "priority": "CRITICAL",
                "recommendation": "Suspend purchasing until critical risks are mitigated",
                "category": "GENERAL"
            })
        elif risk_level == RiskLevel.HIGH:
            recommendations.append({
                "priority": "HIGH",
                "recommendation": "Implement enhanced due diligence measures",
                "category": "GENERAL"
            })

        # Category-specific recommendations
        highest_risk_category = max(category_scores.items(), key=lambda x: x[1])

        if highest_risk_category[0] == RiskCategory.DEFORESTATION:
            recommendations.append({
                "priority": "HIGH",
                "recommendation": "Request satellite imagery verification for all plots",
                "category": "DEFORESTATION"
            })
        elif highest_risk_category[0] == RiskCategory.TRACEABILITY:
            recommendations.append({
                "priority": "HIGH",
                "recommendation": "Conduct supply chain mapping audit",
                "category": "TRACEABILITY"
            })

        # Factor-specific recommendations
        critical_factors = [f for f in factors if f.severity == Severity.CRITICAL]
        for factor in critical_factors:
            recommendations.append({
                "priority": "CRITICAL",
                "recommendation": f"Address: {factor.name}",
                "category": factor.category.value,
                "factor_id": str(factor.factor_id)
            })

        return recommendations

    def _identify_required_mitigations(
        self,
        factors: List[RiskFactor],
        risk_level: RiskLevel
    ) -> List[Dict]:
        """
        Identify required mitigation actions.
        """
        mitigations = []

        for factor in factors:
            if factor.severity in [Severity.CRITICAL, Severity.HIGH] and not factor.mitigated:
                mitigation = {
                    "risk_factor_id": str(factor.factor_id),
                    "risk_factor_name": factor.name,
                    "category": factor.category.value,
                    "priority": "CRITICAL" if factor.severity == Severity.CRITICAL else "HIGH",
                    "suggested_action": self._suggest_mitigation_action(factor)
                }
                mitigations.append(mitigation)

        return mitigations

    def _suggest_mitigation_action(self, factor: RiskFactor) -> str:
        """
        Suggest mitigation action based on factor type.
        """
        action_map = {
            "NO_PLOTS_REGISTERED": "Request supplier to register all production plots with geolocation",
            "UNVERIFIED_DEFORESTATION": "Commission satellite verification of plot deforestation status",
            "DEFORESTATION_ALERTS": "Investigate deforestation alerts and verify plot boundaries",
            "UNVERIFIED_SUPPLIER": "Conduct supplier verification and site visit",
            "NO_CERTIFICATIONS": "Require sustainability certification or equivalent documentation",
            "SANCTIONED_ENTITY": "Terminate supplier relationship immediately",
            "INCOMPLETE_PLOT_COVERAGE": "Request complete traceability to origin for all volumes",
            "CUSTODY_GAPS": "Investigate and document missing custody transfers",
            "MISSING_DOCUMENTS": "Request missing documentation from supplier"
        }

        return action_map.get(
            factor.factor_type,
            f"Investigate and mitigate: {factor.name}"
        )

    def _generate_uuid(self) -> UUID:
        """Generate a new UUID."""
        import uuid
        return uuid.uuid4()
```

---

## 6. API Specification

```yaml
openapi: 3.0.3
info:
  title: Supply Chain Risk Agent API
  version: 1.0.0

paths:
  /api/v1/risk/assessments:
    post:
      summary: Create risk assessment
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required:
                - assessment_type
                - target_entity_type
                - target_entity_ids
              properties:
                assessment_type:
                  type: string
                  enum: [SUPPLIER_ONBOARDING, PERIODIC_REVIEW, TRIGGERED_REVIEW, DDS_PREPARATION]
                target_entity_type:
                  type: string
                  enum: [SUPPLIER, FACILITY, SUPPLY_CHAIN]
                target_entity_ids:
                  type: array
                  items:
                    type: string
                    format: uuid
                assessment_depth:
                  type: string
                  enum: [QUICK, STANDARD, ENHANCED]
                  default: STANDARD
    get:
      summary: List risk assessments
      parameters:
        - name: status
          in: query
          schema:
            type: string
        - name: target_entity_type
          in: query
          schema:
            type: string

  /api/v1/risk/assessments/{assessment_id}:
    get:
      summary: Get assessment details
    patch:
      summary: Update assessment

  /api/v1/risk/profiles:
    get:
      summary: List risk profiles
      parameters:
        - name: entity_type
          in: query
          schema:
            type: string
        - name: risk_level
          in: query
          schema:
            type: string
            enum: [LOW, STANDARD, HIGH, CRITICAL]

  /api/v1/risk/profiles/{profile_id}:
    get:
      summary: Get risk profile

  /api/v1/risk/suppliers/{supplier_id}/risk:
    get:
      summary: Get supplier risk profile
    post:
      summary: Trigger supplier risk assessment

  /api/v1/risk/countries:
    get:
      summary: List country risk classifications
    post:
      summary: Update country classification (admin)

  /api/v1/risk/countries/{country_code}:
    get:
      summary: Get country risk details

  /api/v1/risk/mitigations:
    post:
      summary: Create mitigation action
    get:
      summary: List mitigations

  /api/v1/risk/mitigations/{mitigation_id}:
    get:
      summary: Get mitigation details
    patch:
      summary: Update mitigation status

  /api/v1/risk/alerts:
    get:
      summary: List risk alerts
      parameters:
        - name: severity
          in: query
          schema:
            type: string
            enum: [INFO, WARNING, HIGH, CRITICAL]
        - name: status
          in: query
          schema:
            type: string
            enum: [OPEN, ACKNOWLEDGED, IN_PROGRESS, RESOLVED]

  /api/v1/risk/alerts/{alert_id}:
    get:
      summary: Get alert details
    patch:
      summary: Update alert status (acknowledge/resolve)

  /api/v1/risk/dashboard:
    get:
      summary: Get risk dashboard metrics
      parameters:
        - name: scope
          in: query
          schema:
            type: string
            enum: [ALL, SUPPLIER, COUNTRY, COMMODITY]

  /api/v1/risk/trends:
    get:
      summary: Get risk trend analysis
      parameters:
        - name: period
          in: query
          schema:
            type: string
            enum: [7D, 30D, 90D, 1Y]
```

---

## 7. Non-Functional Requirements

### 7.1 Performance
- Risk assessment: <60 seconds for standard depth
- Profile retrieval: <200ms
- Alert processing: <5 seconds from trigger
- Dashboard refresh: <3 seconds

### 7.2 Scalability
- Support 100,000+ supplier profiles
- Process 10,000+ assessments per day
- Handle 1,000+ concurrent alert checks

### 7.3 Accuracy
- Risk scoring consistent across assessments
- Country classifications updated within 24h of EU publication
- False positive rate <5% for deforestation alerts

### 7.4 Availability
- 99.9% uptime for risk queries
- Alert processing: 99.99% message delivery
- Graceful degradation for external data sources

---

## 8. Integration Points

| System | Integration Type | Purpose |
|---|---|---|
| GL-EUDR-002 Geolocation | Query | Get plot coordinates for analysis |
| GL-EUDR-004 Supplier Verification | Query | Get supplier verification status |
| GL-EUDR-005 Plot Registry | Query | Get registered plots |
| GL-EUDR-014 Traceability Audit | Query | Get traceability metrics |
| Global Forest Watch | API | Deforestation alerts |
| Sanctions Lists | API | Entity screening |
| EU EUDR Portal | Webhook | Country benchmark updates |

---

## 9. Data Sources for Risk Assessment

| Source | Data Type | Update Frequency |
|---|---|---|
| Global Forest Watch | Deforestation alerts | Daily |
| Hansen/UMD | Forest cover change | Annual |
| Transparency International | Corruption index | Annual |
| World Bank | Governance indicators | Annual |
| EU Commission | Country benchmarks | As published |
| OFAC/EU Sanctions | Sanctioned entities | Daily |
| Certification Bodies | Certification status | Real-time |

---

## 10. Success Metrics

- **Assessment Coverage:** 100% of active suppliers assessed
- **Refresh Rate:** All profiles refreshed within validity period
- **Alert Response:** <24 hours to acknowledge critical alerts
- **Mitigation Completion:** >90% mitigations completed on time
- **Risk Prediction Accuracy:** >85% alignment with actual compliance issues

---

*Document Version: 1.0*
*Created: 2026-01-30*
*Status: APPROVED FOR IMPLEMENTATION*
