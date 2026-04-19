"""
GreenLang Framework - Quality Standards Definition

Comprehensive quality standards for GreenLang AI agents based on:
- ISO/IEC 42001:2023 (AI Management System)
- ISO/IEC 23894:2023 (AI Risk Management)
- NIST AI RMF 1.0
- EU AI Act (high-risk systems)
- IEEE 7000 series (Ethical AI)

This module defines the scoring framework for agent quality assessment.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ComplianceLevel(Enum):
    """Compliance level for requirements."""
    MANDATORY = "mandatory"      # Must have - failure is blocking
    RECOMMENDED = "recommended"  # Should have - affects score significantly
    OPTIONAL = "optional"        # Nice to have - minor score impact
    ASPIRATIONAL = "aspirational"  # Future target


class AgentClass(Enum):
    """Classification of agent types."""
    CALCULATOR = "calculator"    # Pure calculation agents
    OPTIMIZER = "optimizer"      # Optimization agents
    ANALYZER = "analyzer"        # Analysis and reporting agents
    PREDICTOR = "predictor"      # ML-based prediction agents
    ORCHESTRATOR = "orchestrator"  # Multi-agent coordination
    HYBRID = "hybrid"            # Combination of above


class DomainCategory(Enum):
    """Industrial domain categories."""
    EMISSIONS = "emissions"          # GHG/carbon emissions
    ENERGY = "energy"                # Energy efficiency
    PROCESS_HEAT = "process_heat"    # Heat recovery/steam
    WATER = "water"                  # Water management
    WASTE = "waste"                  # Waste management
    SUPPLY_CHAIN = "supply_chain"    # Supply chain sustainability
    COMPLIANCE = "compliance"        # Regulatory compliance
    GENERAL = "general"              # General purpose


class RequirementLevel(Enum):
    """Requirement satisfaction level."""
    NOT_APPLICABLE = "N/A"
    NOT_MET = "not_met"
    PARTIALLY_MET = "partially_met"
    FULLY_MET = "fully_met"
    EXCEEDED = "exceeded"


@dataclass
class QualityDimension:
    """
    Single dimension of agent quality.

    Each dimension has a maximum score and weighting factor.
    """
    name: str
    description: str
    max_score: float = 100.0
    weight: float = 1.0
    compliance_level: ComplianceLevel = ComplianceLevel.RECOMMENDED
    sub_dimensions: List["QualityDimension"] = field(default_factory=list)
    evaluation_criteria: List[str] = field(default_factory=list)

    def weighted_max(self) -> float:
        """Get weighted maximum score."""
        return self.max_score * self.weight


@dataclass
class QualityStandard:
    """
    Complete quality standard for GreenLang agents.

    Defines all dimensions, weights, and thresholds for scoring.
    """
    name: str = "GreenLang Agent Quality Standard v1.0"
    version: str = "1.0.0"

    # Score thresholds
    THRESHOLD_PRODUCTION: float = 90.0    # Ready for production
    THRESHOLD_BETA: float = 80.0          # Beta deployment
    THRESHOLD_ALPHA: float = 70.0         # Alpha/testing
    THRESHOLD_MINIMUM: float = 60.0       # Minimum viable

    # Quality dimensions with weights (total weight = 100)
    dimensions: List[QualityDimension] = field(default_factory=list)

    def __post_init__(self):
        """Initialize default quality dimensions."""
        if not self.dimensions:
            self.dimensions = self._default_dimensions()

    def _default_dimensions(self) -> List[QualityDimension]:
        """Create default GreenLang quality dimensions."""
        return [
            # 1. MATHEMATICAL RIGOR (20%)
            QualityDimension(
                name="Mathematical Rigor",
                description="Accuracy and correctness of mathematical formulations",
                weight=20.0,
                compliance_level=ComplianceLevel.MANDATORY,
                sub_dimensions=[
                    QualityDimension(
                        name="Formula Correctness",
                        description="Equations match authoritative sources",
                        weight=8.0,
                        evaluation_criteria=[
                            "All formulas cite peer-reviewed references",
                            "Unit consistency in all equations",
                            "Boundary conditions handled correctly",
                            "Numerical stability verified",
                        ]
                    ),
                    QualityDimension(
                        name="Physical Laws Compliance",
                        description="Adherence to conservation laws and physics",
                        weight=6.0,
                        evaluation_criteria=[
                            "Energy conservation verified",
                            "Mass balance maintained",
                            "Second law compliance (exergy)",
                            "Thermodynamic feasibility",
                        ]
                    ),
                    QualityDimension(
                        name="Numerical Methods",
                        description="Quality of numerical algorithms",
                        weight=6.0,
                        evaluation_criteria=[
                            "Convergence criteria defined",
                            "Tolerance levels specified",
                            "Iteration limits set",
                            "Error handling for edge cases",
                        ]
                    ),
                ],
            ),

            # 2. DETERMINISM & REPRODUCIBILITY (15%)
            QualityDimension(
                name="Determinism & Reproducibility",
                description="Ability to reproduce identical results",
                weight=15.0,
                compliance_level=ComplianceLevel.MANDATORY,
                sub_dimensions=[
                    QualityDimension(
                        name="Computation Provenance",
                        description="SHA-256 tracking of all calculations",
                        weight=7.0,
                        evaluation_criteria=[
                            "Every calculation has SHA-256 hash",
                            "Input parameters included in hash",
                            "Version information tracked",
                            "Timestamp recorded (UTC)",
                        ]
                    ),
                    QualityDimension(
                        name="Seed Management",
                        description="Control of random number generation",
                        weight=4.0,
                        evaluation_criteria=[
                            "All RNG seeded explicitly",
                            "Seeds stored for reproduction",
                            "No hidden randomness",
                        ]
                    ),
                    QualityDimension(
                        name="State Isolation",
                        description="No shared mutable state",
                        weight=4.0,
                        evaluation_criteria=[
                            "Functional calculation patterns",
                            "No global state mutations",
                            "Thread-safe operations",
                        ]
                    ),
                ],
            ),

            # 3. DATA MODELS & VALIDATION (15%)
            QualityDimension(
                name="Data Models & Validation",
                description="Quality of data structures and validation",
                weight=15.0,
                compliance_level=ComplianceLevel.MANDATORY,
                sub_dimensions=[
                    QualityDimension(
                        name="Schema Quality",
                        description="Pydantic/dataclass definitions",
                        weight=6.0,
                        evaluation_criteria=[
                            "All inputs have Pydantic models",
                            "Comprehensive field validators",
                            "Type hints complete",
                            "Default values appropriate",
                        ]
                    ),
                    QualityDimension(
                        name="Input Validation",
                        description="Validation of all inputs",
                        weight=5.0,
                        evaluation_criteria=[
                            "Range checks for physical values",
                            "Unit validation",
                            "Null/empty handling",
                            "Cross-field validation",
                        ]
                    ),
                    QualityDimension(
                        name="Output Contracts",
                        description="Well-defined output schemas",
                        weight=4.0,
                        evaluation_criteria=[
                            "All outputs have schemas",
                            "Error responses defined",
                            "Metadata included",
                        ]
                    ),
                ],
            ),

            # 4. EXPLAINABILITY & TRANSPARENCY (15%)
            QualityDimension(
                name="Explainability & Transparency",
                description="Ability to explain decisions",
                weight=15.0,
                compliance_level=ComplianceLevel.MANDATORY,
                sub_dimensions=[
                    QualityDimension(
                        name="Engineering Rationale",
                        description="Rule-based explanations",
                        weight=6.0,
                        evaluation_criteria=[
                            "Constraint satisfaction explained",
                            "Design rules documented",
                            "Trade-offs identified",
                            "Feasibility reasoning",
                        ]
                    ),
                    QualityDimension(
                        name="Statistical Attribution",
                        description="SHAP/LIME explanations",
                        weight=5.0,
                        evaluation_criteria=[
                            "Feature importance available",
                            "Local explanations (LIME)",
                            "Global explanations (SHAP)",
                        ]
                    ),
                    QualityDimension(
                        name="Audit Trail",
                        description="Complete decision history",
                        weight=4.0,
                        evaluation_criteria=[
                            "All decisions logged",
                            "Timestamps included",
                            "User/system attribution",
                        ]
                    ),
                ],
            ),

            # 5. TESTING & VALIDATION (12%)
            QualityDimension(
                name="Testing & Validation",
                description="Test coverage and validation",
                weight=12.0,
                compliance_level=ComplianceLevel.MANDATORY,
                sub_dimensions=[
                    QualityDimension(
                        name="Unit Test Coverage",
                        description="Code coverage metrics",
                        weight=4.0,
                        evaluation_criteria=[
                            ">=80% line coverage",
                            ">=70% branch coverage",
                            "All calculators tested",
                            "Edge cases covered",
                        ]
                    ),
                    QualityDimension(
                        name="Integration Tests",
                        description="End-to-end testing",
                        weight=4.0,
                        evaluation_criteria=[
                            "Orchestrator tests",
                            "API endpoint tests",
                            "Workflow tests",
                        ]
                    ),
                    QualityDimension(
                        name="Validation Against References",
                        description="Benchmark testing",
                        weight=4.0,
                        evaluation_criteria=[
                            "Textbook examples validated",
                            "Industry benchmarks passed",
                            "Golden test cases",
                        ]
                    ),
                ],
            ),

            # 6. API & INTEGRATION (8%)
            QualityDimension(
                name="API & Integration",
                description="API quality and integrations",
                weight=8.0,
                compliance_level=ComplianceLevel.RECOMMENDED,
                sub_dimensions=[
                    QualityDimension(
                        name="REST API",
                        description="RESTful API quality",
                        weight=3.0,
                        evaluation_criteria=[
                            "OpenAPI documentation",
                            "Proper HTTP methods",
                            "Error handling",
                            "Rate limiting",
                        ]
                    ),
                    QualityDimension(
                        name="GraphQL API",
                        description="GraphQL schema quality",
                        weight=2.0,
                        evaluation_criteria=[
                            "Schema defined",
                            "Queries and mutations",
                            "Type safety",
                        ]
                    ),
                    QualityDimension(
                        name="Event Streaming",
                        description="Kafka/event integration",
                        weight=3.0,
                        evaluation_criteria=[
                            "Kafka topics defined",
                            "Event schemas",
                            "Async processing",
                        ]
                    ),
                ],
            ),

            # 7. DEPLOYMENT & OPERATIONS (8%)
            QualityDimension(
                name="Deployment & Operations",
                description="Production readiness",
                weight=8.0,
                compliance_level=ComplianceLevel.RECOMMENDED,
                sub_dimensions=[
                    QualityDimension(
                        name="Containerization",
                        description="Docker configuration",
                        weight=3.0,
                        evaluation_criteria=[
                            "Dockerfile present",
                            "Multi-stage build",
                            "Non-root user",
                            "Health checks",
                        ]
                    ),
                    QualityDimension(
                        name="Kubernetes",
                        description="K8s manifests",
                        weight=3.0,
                        evaluation_criteria=[
                            "Deployment manifest",
                            "Service defined",
                            "HPA configured",
                            "PDB defined",
                        ]
                    ),
                    QualityDimension(
                        name="Observability",
                        description="Monitoring and logging",
                        weight=2.0,
                        evaluation_criteria=[
                            "Structured logging",
                            "Metrics exposed",
                            "Health endpoints",
                        ]
                    ),
                ],
            ),

            # 8. DOCUMENTATION (5%)
            QualityDimension(
                name="Documentation",
                description="Documentation quality",
                weight=5.0,
                compliance_level=ComplianceLevel.RECOMMENDED,
                sub_dimensions=[
                    QualityDimension(
                        name="README",
                        description="Project documentation",
                        weight=2.0,
                        evaluation_criteria=[
                            "Installation instructions",
                            "Quick start guide",
                            "API documentation",
                        ]
                    ),
                    QualityDimension(
                        name="Code Documentation",
                        description="Inline documentation",
                        weight=2.0,
                        evaluation_criteria=[
                            "Docstrings on public methods",
                            "Type hints",
                            "Example usage",
                        ]
                    ),
                    QualityDimension(
                        name="Architecture Docs",
                        description="Design documentation",
                        weight=1.0,
                        evaluation_criteria=[
                            "Architecture diagram",
                            "Data flow documentation",
                        ]
                    ),
                ],
            ),

            # 9. SAFETY & CONSTRAINTS (2%)
            QualityDimension(
                name="Safety & Constraints",
                description="Safety mechanisms",
                weight=2.0,
                compliance_level=ComplianceLevel.MANDATORY,
                sub_dimensions=[
                    QualityDimension(
                        name="Physical Constraints",
                        description="Safety limits enforced",
                        weight=1.0,
                        evaluation_criteria=[
                            "Temperature limits",
                            "Pressure limits",
                            "Flow rate limits",
                        ]
                    ),
                    QualityDimension(
                        name="Operational Guards",
                        description="Runtime safety checks",
                        weight=1.0,
                        evaluation_criteria=[
                            "Timeout handling",
                            "Resource limits",
                            "Graceful degradation",
                        ]
                    ),
                ],
            ),
        ]

    def get_total_weight(self) -> float:
        """Get total weight of all dimensions."""
        return sum(d.weight for d in self.dimensions)

    def get_grade(self, score: float) -> str:
        """Get letter grade for score."""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "A-"
        elif score >= 80:
            return "B+"
        elif score >= 75:
            return "B"
        elif score >= 70:
            return "B-"
        elif score >= 65:
            return "C+"
        elif score >= 60:
            return "C"
        else:
            return "F"

    def get_certification_level(self, score: float) -> str:
        """Get certification level for score."""
        if score >= self.THRESHOLD_PRODUCTION:
            return "Production Ready"
        elif score >= self.THRESHOLD_BETA:
            return "Beta"
        elif score >= self.THRESHOLD_ALPHA:
            return "Alpha"
        elif score >= self.THRESHOLD_MINIMUM:
            return "Development"
        else:
            return "Not Certified"


# Singleton instance
GREENLANG_STANDARD = QualityStandard()
