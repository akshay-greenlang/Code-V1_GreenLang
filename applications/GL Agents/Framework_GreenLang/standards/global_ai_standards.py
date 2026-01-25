"""
GreenLang Framework - Global AI Standards v2.0

Comprehensive quality standards aligned with:
- Anthropic (Claude) AI Safety Standards
- OpenAI Safety & Alignment Framework
- Google DeepMind Responsible AI Principles
- ISO/IEC 42001:2023 (AI Management System)
- ISO/IEC 23894:2023 (AI Risk Management)
- NIST AI RMF 1.0
- EU AI Act (High-Risk Systems)
- IEEE 7000 Series (Ethical AI)

This is the authoritative scoring framework for GreenLang agent evaluation.
Target: 95+/100 for production deployment.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json


class ScoreTier(Enum):
    """Scoring tier classification."""
    TIER_1_EXCELLENT = "tier_1"      # 95-100: Production Ready, Industry Leading
    TIER_2_GOOD = "tier_2"           # 85-94: Production Ready with Observations
    TIER_3_ACCEPTABLE = "tier_3"     # 75-84: Beta Deployment, Improvements Needed
    TIER_4_DEVELOPING = "tier_4"     # 60-74: Alpha/Development Only
    TIER_5_INADEQUATE = "tier_5"     # <60: Not Certified


class ComplianceLevel(Enum):
    """Requirement compliance level."""
    MANDATORY = "mandatory"          # Blocking - must pass
    REQUIRED = "required"            # Strong requirement
    RECOMMENDED = "recommended"      # Should have
    OPTIONAL = "optional"            # Nice to have


class DomainStandard(Enum):
    """Industry domain standards."""
    # Safety Standards
    IEC_61511 = "IEC 61511"          # Functional Safety (SIL)
    IEC_61508 = "IEC 61508"          # Functional Safety
    NFPA_85 = "NFPA 85"              # Boiler Combustion Safety
    NFPA_86 = "NFPA 86"              # Furnace Safety

    # Emissions & Environmental
    EPA_40_CFR_75 = "EPA 40 CFR Part 75"    # CEMS
    EPA_40_CFR_98 = "EPA 40 CFR Part 98"    # GHG Reporting
    EPA_METHOD_19 = "EPA Method 19"          # Emissions
    GHG_PROTOCOL = "GHG Protocol"
    TCFD = "TCFD"

    # Thermodynamic Standards
    IAPWS_IF97 = "IAPWS-IF97"        # Steam Properties
    ASME_PTC_4 = "ASME PTC 4"        # Boiler Efficiency
    ASME_PTC_4_1 = "ASME PTC 4.1"    # Combustion
    ASME_PTC_4_3 = "ASME PTC 4.3"    # Air Heaters
    ASME_PTC_4_4 = "ASME PTC 4.4"    # Heat Recovery
    ASME_PTC_12_5 = "ASME PTC 12.5"  # Heat Exchangers
    ASME_PTC_39 = "ASME PTC 39"      # Steam Traps

    # Quality & Uncertainty
    ISO_14414 = "ISO 14414"          # Pump Energy Assessment
    ISO_50001 = "ISO 50001"          # Energy Management
    ISO_98 = "ISO 98"                # Uncertainty


@dataclass
class EvaluationCriterion:
    """Single evaluation criterion within a category."""
    name: str
    description: str
    points: float
    compliance: ComplianceLevel = ComplianceLevel.REQUIRED
    verification_method: str = ""
    evidence_required: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "points": self.points,
            "compliance": self.compliance.value,
            "verification_method": self.verification_method,
            "evidence_required": self.evidence_required
        }


@dataclass
class ScoringCategory:
    """
    Major scoring category in the Global AI Standards.

    Each category has a maximum point allocation and contains
    multiple evaluation criteria.
    """
    name: str
    description: str
    max_points: float
    weight: float = 1.0
    compliance: ComplianceLevel = ComplianceLevel.REQUIRED
    criteria: List[EvaluationCriterion] = field(default_factory=list)
    applicable_standards: List[DomainStandard] = field(default_factory=list)

    def get_weighted_max(self) -> float:
        """Get weighted maximum points."""
        return self.max_points * self.weight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "max_points": self.max_points,
            "weight": self.weight,
            "compliance": self.compliance.value,
            "criteria": [c.to_dict() for c in self.criteria],
            "applicable_standards": [s.value for s in self.applicable_standards]
        }


@dataclass
class GlobalAIStandard:
    """
    Complete Global AI Standards Framework for GreenLang Agents.

    This is the authoritative scoring framework based on leading AI
    company standards (Anthropic, OpenAI, Google DeepMind) and
    international regulations.

    Total: 100 points across 8 categories
    Target: 95+ for production deployment
    """
    name: str = "GreenLang Global AI Standards v2.0"
    version: str = "2.0.0"
    effective_date: str = "2024-12-24"

    # Tier thresholds
    THRESHOLD_TIER_1: float = 95.0   # Industry Leading
    THRESHOLD_TIER_2: float = 85.0   # Production Ready
    THRESHOLD_TIER_3: float = 75.0   # Beta
    THRESHOLD_TIER_4: float = 60.0   # Alpha

    # Categories (initialized in __post_init__)
    categories: List[ScoringCategory] = field(default_factory=list)

    def __post_init__(self):
        """Initialize the 8 scoring categories."""
        if not self.categories:
            self.categories = self._create_categories()

    def _create_categories(self) -> List[ScoringCategory]:
        """Create the 8 Global AI Standards categories."""
        return [
            # 1. SAFETY & ALIGNMENT (15 points)
            ScoringCategory(
                name="Safety & Alignment",
                description="Safety mechanisms, circuit breakers, read-only enforcement, and alignment with human values",
                max_points=15.0,
                compliance=ComplianceLevel.MANDATORY,
                applicable_standards=[
                    DomainStandard.IEC_61511,
                    DomainStandard.IEC_61508,
                    DomainStandard.NFPA_85,
                    DomainStandard.NFPA_86
                ],
                criteria=[
                    EvaluationCriterion(
                        name="Circuit Breaker Pattern",
                        description="Implementation of circuit breaker for all external integrations",
                        points=4.0,
                        compliance=ComplianceLevel.MANDATORY,
                        verification_method="Code review for CircuitBreaker class",
                        evidence_required=["circuit_breaker.py", "State machine implementation"]
                    ),
                    EvaluationCriterion(
                        name="Read-Only Enforcement",
                        description="Advisory-only mode with no write access to control systems",
                        points=3.0,
                        compliance=ComplianceLevel.MANDATORY,
                        verification_method="Check for write protections on SCADA/DCS connectors",
                        evidence_required=["Read-only connector configuration", "Write access denial logs"]
                    ),
                    EvaluationCriterion(
                        name="Safety Interlocks",
                        description="IEC 61511 SIL-rated safety interlocks and validators",
                        points=3.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Review safety interlock logic",
                        evidence_required=["interlock_manager.py", "SIS validator"]
                    ),
                    EvaluationCriterion(
                        name="Action Gate / Velocity Limits",
                        description="Rate limiting and action gating for all control recommendations",
                        points=3.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Check action gate implementation",
                        evidence_required=["action_gate.py", "Velocity limit configuration"]
                    ),
                    EvaluationCriterion(
                        name="Emergency Shutdown",
                        description="Graceful degradation and emergency stop capabilities",
                        points=2.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Test emergency shutdown sequence",
                        evidence_required=["emergency_response.py", "Fail-safe defaults"]
                    ),
                ]
            ),

            # 2. EXPLAINABILITY (15 points)
            ScoringCategory(
                name="Explainability",
                description="SHAP/LIME integration, engineering rationale, and decision transparency",
                max_points=15.0,
                compliance=ComplianceLevel.MANDATORY,
                criteria=[
                    EvaluationCriterion(
                        name="SHAP TreeExplainer Integration",
                        description="Actual SHAP TreeExplainer implementation (not just schemas)",
                        points=5.0,
                        compliance=ComplianceLevel.MANDATORY,
                        verification_method="Verify SHAP calculation code, not just data classes",
                        evidence_required=["shap_explainer.py with TreeExplainer", "Feature importance outputs"]
                    ),
                    EvaluationCriterion(
                        name="LIME Explainer",
                        description="Local Interpretable Model-agnostic Explanations",
                        points=3.0,
                        compliance=ComplianceLevel.RECOMMENDED,
                        verification_method="Check LIME implementation",
                        evidence_required=["lime_explainer.py", "Local explanations"]
                    ),
                    EvaluationCriterion(
                        name="Engineering Rationale with Citations",
                        description="Rule-based explanations citing thermodynamic principles and standards",
                        points=4.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Review rationale generator for citations",
                        evidence_required=["engineering_rationale.py", "Citations to ASME/EPA/NIST"]
                    ),
                    EvaluationCriterion(
                        name="Causal Analysis",
                        description="Root cause analysis and counterfactual reasoning",
                        points=2.0,
                        compliance=ComplianceLevel.RECOMMENDED,
                        verification_method="Check causal graph and counterfactual engine",
                        evidence_required=["causal_graph.py", "counterfactual_engine.py"]
                    ),
                    EvaluationCriterion(
                        name="Decision Audit Trail",
                        description="Complete logging of all decisions with reasoning",
                        points=1.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Review audit logging for decision capture",
                        evidence_required=["Audit logs with decision reasoning"]
                    ),
                ]
            ),

            # 3. DETERMINISM & REPRODUCIBILITY (15 points)
            ScoringCategory(
                name="Determinism",
                description="Zero-hallucination architecture, SHA-256 provenance, reproducible calculations",
                max_points=15.0,
                compliance=ComplianceLevel.MANDATORY,
                criteria=[
                    EvaluationCriterion(
                        name="Zero-Hallucination Architecture",
                        description="No LLM in critical calculation path; ML for detection support only",
                        points=5.0,
                        compliance=ComplianceLevel.MANDATORY,
                        verification_method="Code review for LLM usage patterns",
                        evidence_required=["Deterministic calculator implementations", "ML-free compliance path"]
                    ),
                    EvaluationCriterion(
                        name="SHA-256 Provenance Tracking",
                        description="Cryptographic hashing of all calculation inputs/outputs",
                        points=4.0,
                        compliance=ComplianceLevel.MANDATORY,
                        verification_method="Check for SHA-256 hash generation",
                        evidence_required=["provenance.py", "computation_hash in all result schemas"]
                    ),
                    EvaluationCriterion(
                        name="Decimal Precision",
                        description="Fixed-point arithmetic with explicit rounding (Decimal ROUND_HALF_UP)",
                        points=3.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Check for Decimal usage in critical calculations",
                        evidence_required=["Decimal imports", "Explicit rounding mode"]
                    ),
                    EvaluationCriterion(
                        name="Seed Management",
                        description="Explicit RNG seeding for all stochastic components",
                        points=2.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Check random seed configuration",
                        evidence_required=["random_seed in config", "Seed storage for reproduction"]
                    ),
                    EvaluationCriterion(
                        name="State Isolation",
                        description="No shared mutable global state; functional patterns",
                        points=1.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Code review for global state",
                        evidence_required=["Thread-safe operations", "No global mutations"]
                    ),
                ]
            ),

            # 4. TESTING (15 points)
            ScoringCategory(
                name="Testing",
                description="Unit/integration tests, golden value validation, 85%+ coverage with CI enforcement",
                max_points=15.0,
                compliance=ComplianceLevel.MANDATORY,
                criteria=[
                    EvaluationCriterion(
                        name="85%+ Code Coverage",
                        description="pytest-cov with 85% minimum coverage enforced in CI",
                        points=5.0,
                        compliance=ComplianceLevel.MANDATORY,
                        verification_method="Run pytest --cov and verify threshold",
                        evidence_required=["pytest.ini with cov-fail-under=85", "CI workflow with coverage check"]
                    ),
                    EvaluationCriterion(
                        name="Golden Value Tests",
                        description="Validation against NIST/IAPWS reference data",
                        points=4.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Check for golden value test files",
                        evidence_required=["tests/golden/ or golden_values.py", "NIST/IAPWS reference comparisons"]
                    ),
                    EvaluationCriterion(
                        name="Property-Based Testing",
                        description="Hypothesis or similar for edge case discovery",
                        points=2.0,
                        compliance=ComplianceLevel.RECOMMENDED,
                        verification_method="Check for hypothesis tests",
                        evidence_required=["tests/property/", "@given decorators"]
                    ),
                    EvaluationCriterion(
                        name="Integration Tests",
                        description="End-to-end workflow testing",
                        points=2.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Check for integration test files",
                        evidence_required=["tests/integration/", "Orchestrator tests"]
                    ),
                    EvaluationCriterion(
                        name="Chaos/Resilience Testing",
                        description="Fault injection and failure scenario tests",
                        points=2.0,
                        compliance=ComplianceLevel.RECOMMENDED,
                        verification_method="Check for chaos test scenarios",
                        evidence_required=["tests/chaos/", "Failure injection tests"]
                    ),
                ]
            ),

            # 5. REGULATORY COMPLIANCE (10 points)
            ScoringCategory(
                name="Regulatory Compliance",
                description="EPA, NIST, IAPWS-IF97, ASME PTC adherence with documentation",
                max_points=10.0,
                compliance=ComplianceLevel.MANDATORY,
                applicable_standards=[
                    DomainStandard.EPA_40_CFR_75,
                    DomainStandard.EPA_40_CFR_98,
                    DomainStandard.IAPWS_IF97,
                    DomainStandard.ASME_PTC_4,
                    DomainStandard.GHG_PROTOCOL
                ],
                criteria=[
                    EvaluationCriterion(
                        name="Domain Standard Implementation",
                        description="Primary domain standard fully implemented (EPA/ASME/NIST)",
                        points=4.0,
                        compliance=ComplianceLevel.MANDATORY,
                        verification_method="Review standard implementation",
                        evidence_required=["Standard-specific calculator", "Compliance documentation"]
                    ),
                    EvaluationCriterion(
                        name="Compliance Documentation",
                        description="Mapping of code to regulatory requirements",
                        points=2.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Check compliance documentation",
                        evidence_required=["compliance/ directory", "Standard mapping document"]
                    ),
                    EvaluationCriterion(
                        name="Evidence Package Generation",
                        description="Automated evidence generation for audits",
                        points=2.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Check evidence pack generation",
                        evidence_required=["evidence_pack.py", "Retention policy (7+ years)"]
                    ),
                    EvaluationCriterion(
                        name="Multi-Standard Support",
                        description="Support for multiple applicable standards",
                        points=2.0,
                        compliance=ComplianceLevel.RECOMMENDED,
                        verification_method="Count supported standards",
                        evidence_required=["Multiple standard implementations"]
                    ),
                ]
            ),

            # 6. CODE QUALITY (10 points)
            ScoringCategory(
                name="Code Quality",
                description="Type hints, Pydantic validation, error handling, linting",
                max_points=10.0,
                compliance=ComplianceLevel.REQUIRED,
                criteria=[
                    EvaluationCriterion(
                        name="Pydantic V2 Models",
                        description="All inputs/outputs use Pydantic BaseModel with validators",
                        points=3.0,
                        compliance=ComplianceLevel.MANDATORY,
                        verification_method="Check for Pydantic usage",
                        evidence_required=["schemas.py with BaseModel", "Field validators"]
                    ),
                    EvaluationCriterion(
                        name="Complete Type Hints",
                        description="All public functions have type annotations",
                        points=2.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Run mypy",
                        evidence_required=["mypy.ini or pyproject.toml config", "No type errors"]
                    ),
                    EvaluationCriterion(
                        name="Error Handling",
                        description="Comprehensive error handling with custom exceptions",
                        points=2.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Review exception handling",
                        evidence_required=["Custom exception classes", "Try/except blocks"]
                    ),
                    EvaluationCriterion(
                        name="Linting (Ruff/Black)",
                        description="Code formatted and linted with Ruff or Black",
                        points=2.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Run ruff check and black --check",
                        evidence_required=["ruff.toml or pyproject.toml", "Clean lint output"]
                    ),
                    EvaluationCriterion(
                        name="Security Scanning",
                        description="Bandit security scanning with no high-severity issues",
                        points=1.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Run bandit",
                        evidence_required=["bandit.yaml config", "Clean security scan"]
                    ),
                ]
            ),

            # 7. AUDITABILITY (10 points)
            ScoringCategory(
                name="Auditability",
                description="Provenance tracking, logging, traceability, evidence generation",
                max_points=10.0,
                compliance=ComplianceLevel.MANDATORY,
                criteria=[
                    EvaluationCriterion(
                        name="Calculation Event Logging",
                        description="All calculations logged with inputs, outputs, and hashes",
                        points=3.0,
                        compliance=ComplianceLevel.MANDATORY,
                        verification_method="Review audit logging",
                        evidence_required=["audit_logger.py", "Calculation event schemas"]
                    ),
                    EvaluationCriterion(
                        name="Immutable Evidence Packs",
                        description="Cryptographically sealed evidence packages",
                        points=3.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Check evidence sealing",
                        evidence_required=["evidence_pack.py", "Digital signatures or Merkle trees"]
                    ),
                    EvaluationCriterion(
                        name="Retention Policy",
                        description="Configurable retention with regulatory minimum (7 years)",
                        points=2.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Check retention configuration",
                        evidence_required=["Retention policy configuration", "Archive strategy"]
                    ),
                    EvaluationCriterion(
                        name="Chain of Custody",
                        description="Track all modifications with user/system attribution",
                        points=2.0,
                        compliance=ComplianceLevel.RECOMMENDED,
                        verification_method="Review modification tracking",
                        evidence_required=["Modification logs", "User attribution"]
                    ),
                ]
            ),

            # 8. PRODUCTION READINESS (10 points)
            ScoringCategory(
                name="Production Readiness",
                description="CI/CD, containerization, Kubernetes, monitoring, observability",
                max_points=10.0,
                compliance=ComplianceLevel.REQUIRED,
                criteria=[
                    EvaluationCriterion(
                        name="CI/CD Pipeline",
                        description="GitHub Actions with lint, test, security, and coverage gates",
                        points=3.0,
                        compliance=ComplianceLevel.MANDATORY,
                        verification_method="Check .github/workflows/",
                        evidence_required=["quality.yml or ci.yml", "Coverage enforcement"]
                    ),
                    EvaluationCriterion(
                        name="Docker Containerization",
                        description="Multi-stage Dockerfile with health checks",
                        points=2.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Review Dockerfile",
                        evidence_required=["Dockerfile", "docker-compose.yaml"]
                    ),
                    EvaluationCriterion(
                        name="Kubernetes Manifests",
                        description="Deployment, Service, HPA, PDB, NetworkPolicy",
                        points=2.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Check deploy/kubernetes/",
                        evidence_required=["K8s manifests", "Helm chart (optional)"]
                    ),
                    EvaluationCriterion(
                        name="Observability",
                        description="Prometheus metrics, structured logging, health endpoints",
                        points=2.0,
                        compliance=ComplianceLevel.REQUIRED,
                        verification_method="Check metrics and health endpoints",
                        evidence_required=["metrics.py", "health endpoint", "Grafana dashboards"]
                    ),
                    EvaluationCriterion(
                        name="Runbooks & Documentation",
                        description="Operational runbooks for incident response",
                        points=1.0,
                        compliance=ComplianceLevel.RECOMMENDED,
                        verification_method="Check docs/runbooks/",
                        evidence_required=["Runbook files", "Incident response procedures"]
                    ),
                ]
            ),
        ]

    def get_total_points(self) -> float:
        """Get total possible points."""
        return sum(c.max_points for c in self.categories)

    def get_category_by_name(self, name: str) -> Optional[ScoringCategory]:
        """Get category by name."""
        for cat in self.categories:
            if cat.name.lower() == name.lower():
                return cat
        return None

    def get_tier(self, score: float) -> ScoreTier:
        """Get tier for a score."""
        if score >= self.THRESHOLD_TIER_1:
            return ScoreTier.TIER_1_EXCELLENT
        elif score >= self.THRESHOLD_TIER_2:
            return ScoreTier.TIER_2_GOOD
        elif score >= self.THRESHOLD_TIER_3:
            return ScoreTier.TIER_3_ACCEPTABLE
        elif score >= self.THRESHOLD_TIER_4:
            return ScoreTier.TIER_4_DEVELOPING
        else:
            return ScoreTier.TIER_5_INADEQUATE

    def get_grade(self, score: float) -> str:
        """Get letter grade for score."""
        if score >= 97:
            return "A+"
        elif score >= 93:
            return "A"
        elif score >= 90:
            return "A-"
        elif score >= 87:
            return "B+"
        elif score >= 83:
            return "B"
        elif score >= 80:
            return "B-"
        elif score >= 77:
            return "C+"
        elif score >= 73:
            return "C"
        elif score >= 70:
            return "C-"
        elif score >= 67:
            return "D+"
        elif score >= 63:
            return "D"
        elif score >= 60:
            return "D-"
        else:
            return "F"

    def get_certification(self, score: float) -> str:
        """Get certification level for score."""
        tier = self.get_tier(score)
        certifications = {
            ScoreTier.TIER_1_EXCELLENT: "Production Ready - Industry Leading",
            ScoreTier.TIER_2_GOOD: "Production Ready",
            ScoreTier.TIER_3_ACCEPTABLE: "Beta Deployment",
            ScoreTier.TIER_4_DEVELOPING: "Alpha/Development Only",
            ScoreTier.TIER_5_INADEQUATE: "Not Certified"
        }
        return certifications[tier]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "effective_date": self.effective_date,
            "total_points": self.get_total_points(),
            "thresholds": {
                "tier_1": self.THRESHOLD_TIER_1,
                "tier_2": self.THRESHOLD_TIER_2,
                "tier_3": self.THRESHOLD_TIER_3,
                "tier_4": self.THRESHOLD_TIER_4,
            },
            "categories": [c.to_dict() for c in self.categories]
        }


@dataclass
class AgentScore:
    """Complete score for a GreenLang agent."""
    agent_id: str
    agent_name: str
    total_score: float
    max_score: float = 100.0
    tier: ScoreTier = ScoreTier.TIER_5_INADEQUATE
    grade: str = "F"
    certification: str = "Not Certified"
    category_scores: Dict[str, float] = field(default_factory=dict)
    gaps: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    improvement_actions: List[str] = field(default_factory=list)
    scored_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""

    def __post_init__(self):
        """Compute derived fields."""
        # Defer standard reference to avoid circular dependency during module load
        try:
            standard = GLOBAL_AI_STANDARD
            self.tier = standard.get_tier(self.total_score)
            self.grade = standard.get_grade(self.total_score)
            self.certification = standard.get_certification(self.total_score)
        except NameError:
            # GLOBAL_AI_STANDARD not yet defined, will be initialized later
            pass
        if not self.provenance_hash:
            self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 provenance hash."""
        data = {
            "agent_id": self.agent_id,
            "total_score": self.total_score,
            "category_scores": self.category_scores,
            "scored_at": self.scored_at.isoformat()
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "total_score": self.total_score,
            "max_score": self.max_score,
            "tier": self.tier.value,
            "grade": self.grade,
            "certification": self.certification,
            "category_scores": self.category_scores,
            "gaps": self.gaps,
            "strengths": self.strengths,
            "improvement_actions": self.improvement_actions,
            "scored_at": self.scored_at.isoformat(),
            "provenance_hash": self.provenance_hash
        }


# Current Agent Scores (as evaluated 2024-12-24)
CURRENT_AGENT_SCORES: Dict[str, AgentScore] = {
    "GL-001": AgentScore(
        agent_id="GL-001",
        agent_name="ThermalCommand",
        total_score=83.0,
        category_scores={
            "Safety & Alignment": 14,
            "Explainability": 14,
            "Determinism": 14,
            "Testing": 13,
            "Regulatory Compliance": 9,
            "Code Quality": 9,
            "Auditability": 14,
            "Production Readiness": 13
        },
        strengths=[
            "Layered safety (action gate + SIS validator)",
            "Comprehensive audit trail",
            "Full CI/CD pipeline"
        ],
        gaps=[
            "EPA mapping incomplete",
            "Floating-point precision without guards"
        ]
    ),
    "GL-002": AgentScore(
        agent_id="GL-002",
        agent_name="Flameguard",
        total_score=77.0,
        category_scores={
            "Safety & Alignment": 13,
            "Explainability": 12,
            "Determinism": 12,
            "Testing": 11,
            "Regulatory Compliance": 11,
            "Code Quality": 8,
            "Auditability": 12,
            "Production Readiness": 11
        },
        strengths=[
            "ASME PTC 4.1 + NFPA 85 compliance",
            "Circuit breaker implementation"
        ],
        gaps=[
            "Weak test coverage",
            "No CI/CD pipeline"
        ]
    ),
    "GL-003": AgentScore(
        agent_id="GL-003",
        agent_name="UnifiedSteam",
        total_score=87.0,
        category_scores={
            "Safety & Alignment": 14,
            "Explainability": 14,
            "Determinism": 15,
            "Testing": 13,
            "Regulatory Compliance": 12,
            "Code Quality": 9,
            "Auditability": 13,
            "Production Readiness": 14
        },
        strengths=[
            "IAPWS-IF97 with Decimal precision",
            "Causal analysis framework",
            "Full Helm deployment"
        ],
        gaps=[
            "CI/CD not visible in repo"
        ]
    ),
    "GL-004": AgentScore(
        agent_id="GL-004",
        agent_name="Burnmaster",
        total_score=84.0,
        category_scores={
            "Safety & Alignment": 13,
            "Explainability": 13,
            "Determinism": 13,
            "Testing": 12,
            "Regulatory Compliance": 12,
            "Code Quality": 9,
            "Auditability": 13,
            "Production Readiness": 12
        },
        strengths=[
            "ASME PTC 4 + EPA Method 19 + TCFD",
            "Multi-fuel support"
        ],
        gaps=[
            "No circuit breaker",
            "Missing CI/CD"
        ]
    ),
    "GL-005": AgentScore(
        agent_id="GL-005",
        agent_name="Combusense",
        total_score=76.0,
        category_scores={
            "Safety & Alignment": 12,
            "Explainability": 11,
            "Determinism": 12,
            "Testing": 12,
            "Regulatory Compliance": 8,
            "Code Quality": 9,
            "Auditability": 13,
            "Production Readiness": 14
        },
        strengths=[
            "Event sourcing architecture",
            "Grafana dashboards",
            "Kustomize overlays"
        ],
        gaps=[
            "Missing circuit breaker",
            "No ASME/EPA compliance",
            "Limited SHAP integration"
        ]
    ),
    "GL-006": AgentScore(
        agent_id="GL-006",
        agent_name="HEATRECLAIM",
        total_score=76.0,
        category_scores={
            "Safety & Alignment": 12,
            "Explainability": 13,
            "Determinism": 14,
            "Testing": 10,
            "Regulatory Compliance": 9,
            "Code Quality": 9,
            "Auditability": 9,
            "Production Readiness": 0
        },
        strengths=[
            "Pinch analysis with academic citations",
            "Strong thermodynamic rigor"
        ],
        gaps=[
            "No deployment infrastructure",
            "SHAP not implemented",
            "Minimal test coverage"
        ]
    ),
    "GL-007": AgentScore(
        agent_id="GL-007",
        agent_name="FurnacePulse",
        total_score=81.0,
        category_scores={
            "Safety & Alignment": 13,
            "Explainability": 13,
            "Determinism": 13,
            "Testing": 11,
            "Regulatory Compliance": 12,
            "Code Quality": 9,
            "Auditability": 10,
            "Production Readiness": 10
        },
        strengths=[
            "NFPA 86 compliance framework",
            "Advisory-only safety posture",
            "Health monitoring"
        ],
        gaps=[
            "SHAP not implemented",
            "No CI/CD pipeline"
        ]
    ),
    "GL-008": AgentScore(
        agent_id="GL-008",
        agent_name="Trapcatcher",
        total_score=72.0,
        category_scores={
            "Safety & Alignment": 13,
            "Explainability": 12,
            "Determinism": 14,
            "Testing": 9,
            "Regulatory Compliance": 10,
            "Code Quality": 8,
            "Auditability": 9,
            "Production Readiness": 5
        },
        strengths=[
            "ASME PTC 39 compliance",
            "Zero-hallucination guarantee"
        ],
        gaps=[
            "Test suite incomplete",
            "No deployment strategy",
            "Explainer not implemented"
        ]
    ),
    "GL-009": AgentScore(
        agent_id="GL-009",
        agent_name="ThermalIQ",
        total_score=74.0,
        category_scores={
            "Safety & Alignment": 11,
            "Explainability": 12,
            "Determinism": 14,
            "Testing": 8,
            "Regulatory Compliance": 7,
            "Code Quality": 9,
            "Auditability": 8,
            "Production Readiness": 5
        },
        strengths=[
            "Thermodynamic calculations",
            "Sankey visualization"
        ],
        gaps=[
            "SHAP integration incomplete",
            "No production infrastructure",
            "Unknown test coverage"
        ]
    ),
    "GL-010": AgentScore(
        agent_id="GL-010",
        agent_name="EmissionGuardian",
        total_score=78.0,
        category_scores={
            "Safety & Alignment": 13,
            "Explainability": 12,
            "Determinism": 13,
            "Testing": 12,
            "Regulatory Compliance": 14,
            "Code Quality": 9,
            "Auditability": 10,
            "Production Readiness": 9
        },
        strengths=[
            "EPA 40 CFR Part 75 complete",
            "Golden value testing",
            "Comprehensive audit trail"
        ],
        gaps=[
            "No CI/CD pipeline",
            "ML components not explained"
        ]
    ),
}


# Singleton instance
GLOBAL_AI_STANDARD = GlobalAIStandard()


def get_improvement_roadmap(agent_id: str) -> List[str]:
    """
    Get specific improvement actions for an agent to reach 95+.

    Args:
        agent_id: Agent identifier (e.g., "GL-001")

    Returns:
        List of improvement actions
    """
    if agent_id not in CURRENT_AGENT_SCORES:
        return ["Agent not found in scoring database"]

    score = CURRENT_AGENT_SCORES[agent_id]
    actions = []

    # Check each category for gaps
    for cat in GLOBAL_AI_STANDARD.categories:
        cat_score = score.category_scores.get(cat.name, 0)
        gap = cat.max_points - cat_score

        if gap >= 3:
            # Significant gap - needs attention
            for criterion in cat.criteria:
                if criterion.compliance in [ComplianceLevel.MANDATORY, ComplianceLevel.REQUIRED]:
                    actions.append(f"[{cat.name}] {criterion.name}: {criterion.description}")
        elif gap >= 1:
            # Minor gap
            for criterion in cat.criteria:
                if criterion.compliance == ComplianceLevel.MANDATORY:
                    actions.append(f"[{cat.name}] {criterion.name}: {criterion.description}")

    return actions[:15]  # Return top 15 actions
