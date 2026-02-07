# -*- coding: utf-8 -*-
"""
STRIDE Threat Identification Engine - SEC-010 Phase 2

Implements the STRIDE methodology for systematic threat identification.
Analyzes system components, data flows, and trust boundaries to generate
comprehensive threat models.

STRIDE Categories:
    - S: Spoofing - Impersonation attacks
    - T: Tampering - Data modification attacks
    - R: Repudiation - Denying actions
    - I: Information Disclosure - Data leakage
    - D: Denial of Service - Availability attacks
    - E: Elevation of Privilege - Authorization bypass

Example:
    >>> from greenlang.infrastructure.threat_modeling import (
    ...     STRIDEEngine, Component, ComponentType,
    ... )
    >>> engine = STRIDEEngine()
    >>> component = Component(
    ...     name="API Gateway",
    ...     component_type=ComponentType.API,
    ...     trust_level=2,
    ... )
    >>> threats = engine.analyze_component(component)
    >>> len(threats)
    8

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from greenlang.infrastructure.threat_modeling.models import (
    Component,
    ComponentType,
    DataClassification,
    DataFlow,
    Threat,
    ThreatCategory,
    ThreatModel,
    ThreatStatus,
    TrustBoundary,
)
from greenlang.infrastructure.threat_modeling.config import (
    get_config,
    get_default_threat_patterns,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# STRIDE Threat Category Descriptions
# ---------------------------------------------------------------------------

THREAT_CATEGORIES: Dict[str, Dict[str, str]] = {
    "S": {
        "name": "Spoofing",
        "description": "Attacks that involve impersonating a user, component, or system to gain unauthorized access. Spoofing threats violate authentication.",
        "question": "Can an attacker pretend to be someone or something else?",
        "examples": [
            "Session hijacking",
            "Credential theft",
            "IP spoofing",
            "DNS spoofing",
            "Certificate forgery",
        ],
    },
    "T": {
        "name": "Tampering",
        "description": "Attacks that modify data, code, or configurations without proper authorization. Tampering threats violate integrity.",
        "question": "Can an attacker modify data without being detected?",
        "examples": [
            "SQL injection",
            "Cross-site scripting (XSS)",
            "Man-in-the-middle attacks",
            "Database manipulation",
            "Configuration tampering",
        ],
    },
    "R": {
        "name": "Repudiation",
        "description": "Ability to deny having performed an action when there is no proof otherwise. Repudiation threats violate non-repudiation.",
        "question": "Can an attacker deny performing an action?",
        "examples": [
            "Insufficient logging",
            "Log tampering",
            "Missing audit trails",
            "Unsigned transactions",
            "Anonymous actions",
        ],
    },
    "I": {
        "name": "Information Disclosure",
        "description": "Unauthorized access to or disclosure of sensitive information. Information disclosure threats violate confidentiality.",
        "question": "Can sensitive information be exposed to unauthorized parties?",
        "examples": [
            "Data breaches",
            "Exposure of PII",
            "Credential leakage",
            "Error message information",
            "Side-channel attacks",
        ],
    },
    "D": {
        "name": "Denial of Service",
        "description": "Attacks that prevent legitimate users from accessing the system or its resources. DoS threats violate availability.",
        "question": "Can an attacker prevent legitimate access to the system?",
        "examples": [
            "Resource exhaustion",
            "DDoS attacks",
            "Slowloris attacks",
            "Database lock attacks",
            "Algorithmic complexity attacks",
        ],
    },
    "E": {
        "name": "Elevation of Privilege",
        "description": "Attacks that allow an attacker to gain elevated access rights or capabilities. EoP threats violate authorization.",
        "question": "Can an attacker gain access beyond their authorization level?",
        "examples": [
            "Privilege escalation",
            "IDOR vulnerabilities",
            "Broken access control",
            "JWT manipulation",
            "Role bypass",
        ],
    },
}


# ---------------------------------------------------------------------------
# Component Type to Threat Mapping
# ---------------------------------------------------------------------------


def _get_component_threat_patterns(component_type: ComponentType) -> List[Dict[str, Any]]:
    """Get threat patterns for a specific component type.

    Args:
        component_type: The type of component.

    Returns:
        List of threat pattern dictionaries.
    """
    patterns = get_default_threat_patterns()

    type_mapping = {
        ComponentType.WEB_APP: patterns.WEB_APP,
        ComponentType.API: patterns.API,
        ComponentType.DATABASE: patterns.DATABASE,
        ComponentType.MESSAGE_QUEUE: patterns.MESSAGE_QUEUE,
        ComponentType.EXTERNAL_SERVICE: patterns.EXTERNAL_SERVICE,
        ComponentType.CACHE: patterns.CACHE,
        ComponentType.FILE_STORAGE: patterns.FILE_STORAGE,
        ComponentType.LOAD_BALANCER: patterns.LOAD_BALANCER,
        ComponentType.CONTAINER: patterns.CONTAINER,
    }

    return type_mapping.get(component_type, [])


# ---------------------------------------------------------------------------
# STRIDE Engine Implementation
# ---------------------------------------------------------------------------


class STRIDEEngine:
    """STRIDE threat identification engine.

    Provides systematic threat analysis using the STRIDE methodology.
    Analyzes components, data flows, and trust boundaries to identify
    potential security threats.

    Attributes:
        config: Threat modeling configuration.
        threat_patterns: Default threat patterns for component types.

    Example:
        >>> engine = STRIDEEngine()
        >>> threats = engine.analyze_component(api_component)
        >>> model = engine.generate_threat_model(system_definition)
    """

    def __init__(self) -> None:
        """Initialize the STRIDE engine."""
        self.config = get_config()
        self.threat_patterns = get_default_threat_patterns()
        logger.debug("STRIDE engine initialized")

    def analyze_component(
        self,
        component: Component,
        include_generic: bool = True,
    ) -> List[Threat]:
        """Generate STRIDE threats for a component.

        Analyzes a single component and generates threats based on its
        type, trust level, and data classification.

        Args:
            component: The component to analyze.
            include_generic: Whether to include generic STRIDE threats.

        Returns:
            List of identified threats.
        """
        threats: List[Threat] = []
        start_time = datetime.now(timezone.utc)

        logger.info(
            "Analyzing component: name=%s, type=%s, trust_level=%d",
            component.name,
            component.component_type.value,
            component.trust_level,
        )

        # Get component-specific threat patterns
        patterns = _get_component_threat_patterns(component.component_type)

        for pattern in patterns:
            category = ThreatCategory(pattern["category"])
            weight = self.config.stride_weights.get(pattern["category"], 1.0)

            # Calculate base likelihood and impact based on component properties
            base_likelihood = self._calculate_base_likelihood(component, category)
            base_impact = self._calculate_base_impact(component, category)

            # Apply category weight
            risk_score = (base_likelihood * base_impact / 5.0) * weight

            threat = Threat(
                id=str(uuid4()),
                category=category,
                title=pattern["title"],
                description=pattern["description"],
                component_id=component.id,
                likelihood=base_likelihood,
                impact=base_impact,
                risk_score=min(10.0, risk_score * 2),  # Scale to 0-10
                severity=self.config.get_severity_for_score(risk_score * 2),
                status=ThreatStatus.IDENTIFIED,
                threat_source="stride_analysis",
                identified_by="stride_engine",
                identified_at=datetime.now(timezone.utc),
            )
            threats.append(threat)

        # Add generic STRIDE threats if enabled
        if include_generic:
            generic_threats = self._generate_generic_threats(component)
            threats.extend(generic_threats)

        # Limit threats per component
        if len(threats) > self.config.max_threats_per_component:
            # Prioritize by risk score
            threats.sort(key=lambda t: t.risk_score, reverse=True)
            threats = threats[: self.config.max_threats_per_component]

        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Component analysis complete: component=%s, threats=%d, duration_ms=%.2f",
            component.name,
            len(threats),
            duration_ms,
        )

        return threats

    def analyze_data_flow(
        self,
        flow: DataFlow,
        source_component: Optional[Component] = None,
        destination_component: Optional[Component] = None,
    ) -> List[Threat]:
        """Generate threats for a data flow.

        Analyzes a data flow for potential threats, especially those
        related to data in transit and trust boundary crossings.

        Args:
            flow: The data flow to analyze.
            source_component: Optional source component for context.
            destination_component: Optional destination component for context.

        Returns:
            List of identified threats.
        """
        threats: List[Threat] = []

        logger.debug(
            "Analyzing data flow: %s -> %s, encrypted=%s",
            flow.source_component_id,
            flow.destination_component_id,
            flow.encryption,
        )

        # Check for unencrypted sensitive data
        if not flow.encryption and flow.data_classification in (
            DataClassification.CONFIDENTIAL,
            DataClassification.RESTRICTED,
            DataClassification.SECRET,
        ):
            threats.append(
                Threat(
                    category=ThreatCategory.INFORMATION_DISCLOSURE,
                    title="Unencrypted Sensitive Data in Transit",
                    description=f"Data flow contains {flow.data_classification.value} data without encryption, exposing it to eavesdropping.",
                    data_flow_id=flow.id,
                    likelihood=4,
                    impact=5,
                    risk_score=8.0,
                    severity="critical",
                    countermeasures=["Implement TLS 1.3", "Use mTLS for service-to-service"],
                    cwe_ids=["CWE-319"],
                )
            )

        # Check for missing authentication
        if not flow.authentication_required:
            threats.append(
                Threat(
                    category=ThreatCategory.SPOOFING,
                    title="Unauthenticated Data Flow",
                    description="Data flow does not require authentication, allowing unauthorized access.",
                    data_flow_id=flow.id,
                    likelihood=4,
                    impact=4,
                    risk_score=6.4,
                    severity="high",
                    countermeasures=["Implement JWT authentication", "Use API keys"],
                    cwe_ids=["CWE-287"],
                )
            )

        # Check for missing authorization
        if not flow.authorization_required and flow.data_classification != DataClassification.PUBLIC:
            threats.append(
                Threat(
                    category=ThreatCategory.ELEVATION_OF_PRIVILEGE,
                    title="Missing Authorization on Data Flow",
                    description="Data flow does not enforce authorization, allowing unauthorized data access.",
                    data_flow_id=flow.id,
                    likelihood=3,
                    impact=4,
                    risk_score=5.6,
                    severity="medium",
                    countermeasures=["Implement RBAC", "Add resource-level authorization"],
                    cwe_ids=["CWE-862"],
                )
            )

        # Trust boundary crossing threats
        if flow.crosses_trust_boundary:
            threats.extend(self._analyze_boundary_crossing(flow))

        # Man-in-the-middle threat for all flows
        threats.append(
            Threat(
                category=ThreatCategory.TAMPERING,
                title="Man-in-the-Middle Attack",
                description=f"Data flow using {flow.protocol} may be vulnerable to interception and modification.",
                data_flow_id=flow.id,
                likelihood=2 if flow.encryption else 4,
                impact=4,
                risk_score=3.2 if flow.encryption else 6.4,
                severity="low" if flow.encryption else "high",
                countermeasures=["Use TLS 1.3", "Implement certificate pinning", "Use mTLS"],
                cwe_ids=["CWE-300"],
            )
        )

        # Repudiation threat if no logging
        threats.append(
            Threat(
                category=ThreatCategory.REPUDIATION,
                title="Missing Data Flow Audit Trail",
                description="Data transfers may not be logged, preventing audit and forensic analysis.",
                data_flow_id=flow.id,
                likelihood=3,
                impact=3,
                risk_score=4.5,
                severity="medium",
                countermeasures=["Enable access logging", "Implement audit trail for sensitive data"],
                cwe_ids=["CWE-778"],
            )
        )

        return threats

    def analyze_trust_boundary(
        self,
        boundary: TrustBoundary,
        flows: List[DataFlow],
        components: List[Component],
    ) -> List[Threat]:
        """Generate threats for a trust boundary.

        Analyzes flows crossing a trust boundary and generates
        appropriate threats for boundary crossing scenarios.

        Args:
            boundary: The trust boundary to analyze.
            flows: All data flows in the system.
            components: All components in the system.

        Returns:
            List of identified threats.
        """
        threats: List[Threat] = []

        logger.debug(
            "Analyzing trust boundary: name=%s, components=%d",
            boundary.name,
            len(boundary.component_ids),
        )

        # Find flows crossing this boundary
        boundary_component_set = set(boundary.component_ids)
        crossing_flows = [
            f for f in flows
            if (
                (f.source_component_id in boundary_component_set and f.destination_component_id not in boundary_component_set)
                or (f.source_component_id not in boundary_component_set and f.destination_component_id in boundary_component_set)
            )
        ]

        for flow in crossing_flows:
            # Input validation threat
            if flow.destination_component_id in boundary_component_set:
                threats.append(
                    Threat(
                        category=ThreatCategory.TAMPERING,
                        title=f"Input Validation at Boundary: {boundary.name}",
                        description=f"Data entering trust boundary '{boundary.name}' must be validated to prevent injection attacks.",
                        data_flow_id=flow.id,
                        likelihood=4,
                        impact=4,
                        risk_score=6.4,
                        severity="high",
                        countermeasures=["Implement input validation", "Use parameterized queries", "Sanitize all inputs"],
                        cwe_ids=["CWE-20"],
                    )
                )

            # Output encoding threat
            if flow.source_component_id in boundary_component_set:
                threats.append(
                    Threat(
                        category=ThreatCategory.INFORMATION_DISCLOSURE,
                        title=f"Output Encoding at Boundary: {boundary.name}",
                        description=f"Data leaving trust boundary '{boundary.name}' should be properly encoded to prevent information leakage.",
                        data_flow_id=flow.id,
                        likelihood=3,
                        impact=3,
                        risk_score=4.5,
                        severity="medium",
                        countermeasures=["Implement output encoding", "Remove internal data from responses"],
                        cwe_ids=["CWE-200"],
                    )
                )

        # Generic boundary crossing threat
        if crossing_flows:
            threats.append(
                Threat(
                    category=ThreatCategory.ELEVATION_OF_PRIVILEGE,
                    title=f"Trust Boundary Bypass: {boundary.name}",
                    description=f"Attacker may attempt to bypass trust boundary '{boundary.name}' to access higher-trust components.",
                    likelihood=3,
                    impact=5,
                    risk_score=6.0,
                    severity="high",
                    countermeasures=["Enforce authentication at boundary", "Implement network segmentation"],
                    cwe_ids=["CWE-269"],
                )
            )

        return threats

    def generate_threat_model(
        self,
        service_name: str,
        components: List[Component],
        data_flows: List[DataFlow],
        trust_boundaries: Optional[List[TrustBoundary]] = None,
        description: str = "",
        owner: str = "",
    ) -> ThreatModel:
        """Generate a complete threat model for a system.

        Analyzes all components, data flows, and trust boundaries
        to create a comprehensive threat model.

        Args:
            service_name: Name of the service being modeled.
            components: List of system components.
            data_flows: List of data flows.
            trust_boundaries: Optional list of trust boundaries.
            description: Description of the system.
            owner: Team responsible for the system.

        Returns:
            Complete ThreatModel instance.
        """
        start_time = datetime.now(timezone.utc)
        all_threats: List[Threat] = []

        logger.info(
            "Generating threat model: service=%s, components=%d, flows=%d",
            service_name,
            len(components),
            len(data_flows),
        )

        # Analyze each component
        for component in components:
            component_threats = self.analyze_component(component)
            all_threats.extend(component_threats)

        # Analyze each data flow
        for flow in data_flows:
            source = next((c for c in components if c.id == flow.source_component_id), None)
            dest = next((c for c in components if c.id == flow.destination_component_id), None)
            flow_threats = self.analyze_data_flow(flow, source, dest)
            all_threats.extend(flow_threats)

        # Analyze trust boundaries
        if trust_boundaries:
            for boundary in trust_boundaries:
                boundary_threats = self.analyze_trust_boundary(boundary, data_flows, components)
                all_threats.extend(boundary_threats)

        # Calculate category scores
        category_scores = self._calculate_category_scores(all_threats)

        # Calculate overall risk score
        overall_risk_score = self._calculate_overall_risk_score(all_threats)

        # Create threat model
        threat_model = ThreatModel(
            service_name=service_name,
            version="1.0.0",
            description=description,
            owner=owner,
            components=components,
            data_flows=data_flows,
            trust_boundaries=trust_boundaries or [],
            threats=all_threats,
            overall_risk_score=overall_risk_score,
            category_scores=category_scores,
            created_by="stride_engine",
            created_at=start_time,
            updated_at=datetime.now(timezone.utc),
        )

        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Threat model generated: service=%s, threats=%d, overall_risk=%.2f, duration_ms=%.2f",
            service_name,
            len(all_threats),
            overall_risk_score,
            duration_ms,
        )

        return threat_model

    def _calculate_base_likelihood(
        self,
        component: Component,
        category: ThreatCategory,
    ) -> int:
        """Calculate base likelihood for a threat.

        Args:
            component: The component being analyzed.
            category: The STRIDE threat category.

        Returns:
            Likelihood score (1-5).
        """
        # Start with base likelihood of 3
        likelihood = 3

        # Adjust based on trust level (lower trust = higher likelihood)
        if component.trust_level <= 1:
            likelihood += 1
        elif component.trust_level >= 4:
            likelihood -= 1

        # Adjust based on exposure
        if component.is_external:
            likelihood += 1

        # Adjust based on network zone
        if component.network_zone in ("dmz", "external"):
            likelihood += 1

        # Category-specific adjustments
        if category == ThreatCategory.DENIAL_OF_SERVICE and component.component_type == ComponentType.API:
            likelihood += 1
        elif category == ThreatCategory.SPOOFING and component.component_type == ComponentType.IDENTITY_PROVIDER:
            likelihood += 1

        return max(1, min(5, likelihood))

    def _calculate_base_impact(
        self,
        component: Component,
        category: ThreatCategory,
    ) -> int:
        """Calculate base impact for a threat.

        Args:
            component: The component being analyzed.
            category: The STRIDE threat category.

        Returns:
            Impact score (1-5).
        """
        # Start with base impact of 3
        impact = 3

        # Adjust based on data classification
        classification_impact = {
            DataClassification.PUBLIC: -1,
            DataClassification.INTERNAL: 0,
            DataClassification.CONFIDENTIAL: 1,
            DataClassification.RESTRICTED: 2,
            DataClassification.SECRET: 2,
        }
        impact += classification_impact.get(component.data_classification, 0)

        # Category-specific adjustments
        if category == ThreatCategory.INFORMATION_DISCLOSURE:
            if component.data_classification in (DataClassification.RESTRICTED, DataClassification.SECRET):
                impact += 1
        elif category == ThreatCategory.TAMPERING:
            if component.component_type == ComponentType.DATABASE:
                impact += 1

        return max(1, min(5, impact))

    def _generate_generic_threats(self, component: Component) -> List[Threat]:
        """Generate generic STRIDE threats for any component.

        Args:
            component: The component to analyze.

        Returns:
            List of generic threats.
        """
        generic_threats = []

        # Generate one threat per STRIDE category
        for category_code, category_info in THREAT_CATEGORIES.items():
            category = ThreatCategory(category_code)
            weight = self.config.stride_weights.get(category_code, 1.0)

            likelihood = self._calculate_base_likelihood(component, category)
            impact = self._calculate_base_impact(component, category)
            risk_score = (likelihood * impact / 5.0) * weight * 2

            threat = Threat(
                category=category,
                title=f"{category_info['name']} threat for {component.name}",
                description=f"{category_info['description']} Question: {category_info['question']}",
                component_id=component.id,
                likelihood=likelihood,
                impact=impact,
                risk_score=min(10.0, risk_score),
                severity=self.config.get_severity_for_score(risk_score),
                threat_source="stride_generic",
                countermeasures=[f"Implement {category_info['name']} controls"],
            )
            generic_threats.append(threat)

        return generic_threats

    def _analyze_boundary_crossing(self, flow: DataFlow) -> List[Threat]:
        """Analyze threats for flows crossing trust boundaries.

        Args:
            flow: The data flow crossing a boundary.

        Returns:
            List of boundary-related threats.
        """
        threats = []

        # Enhanced authentication required at boundary
        threats.append(
            Threat(
                category=ThreatCategory.SPOOFING,
                title="Trust Boundary Authentication",
                description="Data flow crosses trust boundary; enhanced authentication is required.",
                data_flow_id=flow.id,
                likelihood=3,
                impact=4,
                risk_score=5.6,
                severity="medium",
                countermeasures=["Implement mTLS", "Use strong authentication at boundary"],
                cwe_ids=["CWE-287"],
            )
        )

        # Data validation at boundary
        threats.append(
            Threat(
                category=ThreatCategory.TAMPERING,
                title="Trust Boundary Data Validation",
                description="All data crossing trust boundary must be validated and sanitized.",
                data_flow_id=flow.id,
                likelihood=4,
                impact=4,
                risk_score=6.4,
                severity="high",
                countermeasures=["Implement strict input validation", "Use schema validation"],
                cwe_ids=["CWE-20"],
            )
        )

        return threats

    def _calculate_category_scores(self, threats: List[Threat]) -> Dict[str, float]:
        """Calculate average risk score per STRIDE category.

        Args:
            threats: List of all threats.

        Returns:
            Dictionary mapping category code to average risk score.
        """
        category_scores: Dict[str, List[float]] = {cat.value: [] for cat in ThreatCategory}

        for threat in threats:
            category_scores[threat.category.value].append(threat.risk_score)

        return {
            cat: (sum(scores) / len(scores) if scores else 0.0)
            for cat, scores in category_scores.items()
        }

    def _calculate_overall_risk_score(self, threats: List[Threat]) -> float:
        """Calculate overall risk score from all threats.

        Uses a weighted average that emphasizes high-severity threats.

        Args:
            threats: List of all threats.

        Returns:
            Overall risk score (0-10).
        """
        if not threats:
            return 0.0

        # Weight threats by severity
        severity_weights = {"critical": 4.0, "high": 2.0, "medium": 1.0, "low": 0.5}
        weighted_sum = sum(
            t.risk_score * severity_weights.get(t.severity, 1.0)
            for t in threats
        )
        total_weight = sum(
            severity_weights.get(t.severity, 1.0)
            for t in threats
        )

        return min(10.0, weighted_sum / total_weight) if total_weight > 0 else 0.0

    def get_category_info(self, category: ThreatCategory) -> Dict[str, Any]:
        """Get detailed information about a STRIDE category.

        Args:
            category: The threat category.

        Returns:
            Dictionary with category information.
        """
        return THREAT_CATEGORIES.get(category.value, {})

    def get_all_categories(self) -> Dict[str, Dict[str, str]]:
        """Get information about all STRIDE categories.

        Returns:
            Dictionary of all category information.
        """
        return THREAT_CATEGORIES.copy()


__all__ = [
    "STRIDEEngine",
    "THREAT_CATEGORIES",
]
