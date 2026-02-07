# -*- coding: utf-8 -*-
"""
Attack Surface Mapper - SEC-010 Phase 2

Maps and analyzes the attack surface of a system by identifying
exposed endpoints, data stores, authentication points, and network
exposure. Calculates an exposure score to quantify the attack surface.

Example:
    >>> from greenlang.infrastructure.threat_modeling import AttackSurfaceMapper
    >>> mapper = AttackSurfaceMapper()
    >>> endpoints = mapper.map_endpoints(threat_model)
    >>> exposure_score = mapper.calculate_exposure_score(threat_model)
    >>> report = mapper.generate_attack_surface_report(threat_model)

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.threat_modeling.models import (
    Component,
    ComponentType,
    DataClassification,
    DataFlow,
    ThreatModel,
)
from greenlang.infrastructure.threat_modeling.config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes for Attack Surface Elements
# ---------------------------------------------------------------------------


class ExposureLevel(str, Enum):
    """Exposure level for attack surface elements."""

    INTERNET = "internet"
    """Exposed to the public internet."""

    PARTNER = "partner"
    """Exposed to partner networks."""

    INTERNAL = "internal"
    """Exposed only internally."""

    PRIVATE = "private"
    """Private, not exposed."""


@dataclass
class Endpoint:
    """Represents an exposed API endpoint."""

    path: str
    method: str
    component_id: str
    component_name: str
    authentication_required: bool
    authorization_level: str
    data_classification: DataClassification
    exposure_level: ExposureLevel
    protocol: str
    port: Optional[int]
    rate_limited: bool = False
    input_validation: bool = True
    description: str = ""


@dataclass
class DataStore:
    """Represents a data storage system in the attack surface."""

    name: str
    component_id: str
    store_type: str  # database, cache, file_storage, etc.
    data_classification: DataClassification
    encryption_at_rest: bool
    access_control: str  # none, basic, rbac, abac
    backup_enabled: bool
    network_exposure: ExposureLevel
    contains_pii: bool = False
    contains_secrets: bool = False


@dataclass
class AuthenticationPoint:
    """Represents an authentication entry point."""

    name: str
    component_id: str
    auth_method: str  # password, jwt, oauth, saml, mtls, api_key
    mfa_enabled: bool
    session_timeout_minutes: int
    brute_force_protection: bool
    exposure_level: ExposureLevel
    supports_sso: bool = False
    password_policy_enforced: bool = True


@dataclass
class NetworkExposure:
    """Represents network-level exposure."""

    component_id: str
    component_name: str
    network_zone: str  # dmz, internal, external, private
    exposed_ports: List[int]
    protocols: List[str]
    firewall_protected: bool
    ddos_protection: bool
    tls_required: bool
    ip_whitelist_enabled: bool = False


@dataclass
class AttackSurfaceReport:
    """Complete attack surface report."""

    service_name: str
    generated_at: datetime
    total_endpoints: int
    total_data_stores: int
    total_auth_points: int
    exposure_score: float
    risk_level: str
    endpoints: List[Endpoint]
    data_stores: List[DataStore]
    authentication_points: List[AuthenticationPoint]
    network_exposures: List[NetworkExposure]
    recommendations: List[str]
    high_risk_areas: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Attack Surface Mapper Implementation
# ---------------------------------------------------------------------------


class AttackSurfaceMapper:
    """Maps and analyzes the attack surface of a system.

    Identifies exposed endpoints, data stores, authentication points,
    and network exposure to calculate a comprehensive exposure score.

    Attributes:
        config: Threat modeling configuration.

    Example:
        >>> mapper = AttackSurfaceMapper()
        >>> report = mapper.generate_attack_surface_report(threat_model)
        >>> print(f"Exposure Score: {report.exposure_score}")
    """

    def __init__(self) -> None:
        """Initialize the attack surface mapper."""
        self.config = get_config()
        logger.debug("Attack surface mapper initialized")

    def map_endpoints(self, threat_model: ThreatModel) -> List[Endpoint]:
        """List all external API endpoints from the threat model.

        Identifies API and web application components and maps their
        exposed endpoints.

        Args:
            threat_model: The threat model to analyze.

        Returns:
            List of identified endpoints.
        """
        endpoints: List[Endpoint] = []

        logger.info("Mapping endpoints for service: %s", threat_model.service_name)

        # Find API and web app components
        api_components = [
            c for c in threat_model.components
            if c.component_type in (ComponentType.API, ComponentType.WEB_APP, ComponentType.LOAD_BALANCER)
        ]

        for component in api_components:
            # Determine exposure level based on component properties
            exposure_level = self._determine_exposure_level(component)

            # Find inbound data flows to identify endpoint patterns
            inbound_flows = [
                f for f in threat_model.data_flows
                if f.destination_component_id == component.id
            ]

            # Create endpoint entries
            for flow in inbound_flows:
                endpoint = Endpoint(
                    path=f"/{component.name.lower().replace(' ', '-')}/{flow.data_type}",
                    method="POST" if flow.data_type not in ("query", "read", "get") else "GET",
                    component_id=component.id,
                    component_name=component.name,
                    authentication_required=flow.authentication_required,
                    authorization_level="rbac" if flow.authorization_required else "none",
                    data_classification=flow.data_classification,
                    exposure_level=exposure_level,
                    protocol=flow.protocol,
                    port=flow.port or (443 if flow.encryption else 80),
                    rate_limited=True,  # Assume rate limiting is implemented
                    input_validation=True,  # Assume validation is implemented
                    description=flow.description or f"Endpoint for {flow.data_type}",
                )
                endpoints.append(endpoint)

            # If no inbound flows, create a generic endpoint
            if not inbound_flows:
                endpoint = Endpoint(
                    path=f"/{component.name.lower().replace(' ', '-')}",
                    method="GET",
                    component_id=component.id,
                    component_name=component.name,
                    authentication_required=True,
                    authorization_level="rbac",
                    data_classification=component.data_classification,
                    exposure_level=exposure_level,
                    protocol="https",
                    port=443,
                    rate_limited=True,
                    input_validation=True,
                    description=f"API endpoint for {component.name}",
                )
                endpoints.append(endpoint)

        logger.info("Mapped %d endpoints", len(endpoints))
        return endpoints

    def map_data_stores(self, threat_model: ThreatModel) -> List[DataStore]:
        """Map all data stores in the system.

        Identifies databases, caches, file storage, and other data
        persistence components.

        Args:
            threat_model: The threat model to analyze.

        Returns:
            List of identified data stores.
        """
        data_stores: List[DataStore] = []

        logger.info("Mapping data stores for service: %s", threat_model.service_name)

        # Find data store components
        store_types = (
            ComponentType.DATABASE,
            ComponentType.CACHE,
            ComponentType.FILE_STORAGE,
        )
        store_components = [
            c for c in threat_model.components
            if c.component_type in store_types
        ]

        for component in store_components:
            # Determine store type
            store_type_map = {
                ComponentType.DATABASE: "database",
                ComponentType.CACHE: "cache",
                ComponentType.FILE_STORAGE: "file_storage",
            }
            store_type = store_type_map.get(component.component_type, "unknown")

            # Determine exposure level
            exposure_level = self._determine_exposure_level(component)

            # Check for sensitive data
            contains_pii = component.data_classification in (
                DataClassification.RESTRICTED,
                DataClassification.SECRET,
            )
            contains_secrets = "secret" in component.name.lower() or component.component_type == ComponentType.SECRET_MANAGER

            data_store = DataStore(
                name=component.name,
                component_id=component.id,
                store_type=store_type,
                data_classification=component.data_classification,
                encryption_at_rest=component.data_classification != DataClassification.PUBLIC,
                access_control="rbac",  # Assume RBAC is implemented
                backup_enabled=True,  # Assume backups are enabled
                network_exposure=exposure_level,
                contains_pii=contains_pii,
                contains_secrets=contains_secrets,
            )
            data_stores.append(data_store)

        logger.info("Mapped %d data stores", len(data_stores))
        return data_stores

    def map_authentication_points(self, threat_model: ThreatModel) -> List[AuthenticationPoint]:
        """Map all authentication entry points.

        Identifies identity providers, authentication services, and
        login endpoints.

        Args:
            threat_model: The threat model to analyze.

        Returns:
            List of identified authentication points.
        """
        auth_points: List[AuthenticationPoint] = []

        logger.info("Mapping authentication points for service: %s", threat_model.service_name)

        # Find identity provider components
        idp_components = [
            c for c in threat_model.components
            if c.component_type == ComponentType.IDENTITY_PROVIDER
            or "auth" in c.name.lower()
            or "identity" in c.name.lower()
            or "login" in c.name.lower()
        ]

        for component in idp_components:
            exposure_level = self._determine_exposure_level(component)

            auth_point = AuthenticationPoint(
                name=component.name,
                component_id=component.id,
                auth_method="jwt",  # Default assumption
                mfa_enabled=True,  # Assume MFA is available
                session_timeout_minutes=30,
                brute_force_protection=True,
                exposure_level=exposure_level,
                supports_sso="sso" in component.name.lower(),
                password_policy_enforced=True,
            )
            auth_points.append(auth_point)

        # Also check data flows for authentication methods
        for flow in threat_model.data_flows:
            if flow.authentication_required and flow.authentication_method:
                # Find destination component
                dest = next(
                    (c for c in threat_model.components if c.id == flow.destination_component_id),
                    None,
                )
                if dest and not any(ap.component_id == dest.id for ap in auth_points):
                    auth_point = AuthenticationPoint(
                        name=f"{dest.name} Authentication",
                        component_id=dest.id,
                        auth_method=flow.authentication_method,
                        mfa_enabled=False,
                        session_timeout_minutes=60,
                        brute_force_protection=True,
                        exposure_level=self._determine_exposure_level(dest),
                        supports_sso=False,
                        password_policy_enforced=True,
                    )
                    auth_points.append(auth_point)

        logger.info("Mapped %d authentication points", len(auth_points))
        return auth_points

    def map_network_exposure(self, threat_model: ThreatModel) -> List[NetworkExposure]:
        """Map network-level exposure for all components.

        Identifies network zones, exposed ports, and protocols for
        each component in the system.

        Args:
            threat_model: The threat model to analyze.

        Returns:
            List of network exposure information.
        """
        network_exposures: List[NetworkExposure] = []

        logger.info("Mapping network exposure for service: %s", threat_model.service_name)

        for component in threat_model.components:
            # Find exposed ports from data flows
            exposed_ports: List[int] = []
            protocols: List[str] = []

            for flow in threat_model.data_flows:
                if flow.destination_component_id == component.id:
                    if flow.port:
                        exposed_ports.append(flow.port)
                    protocols.append(flow.protocol)

            # Remove duplicates
            exposed_ports = list(set(exposed_ports))
            protocols = list(set(protocols))

            # Determine protection status
            firewall_protected = component.network_zone != "external"
            ddos_protection = component.component_type in (ComponentType.LOAD_BALANCER, ComponentType.CDN)
            tls_required = any(f.encryption for f in threat_model.data_flows if f.destination_component_id == component.id)

            network_exposure = NetworkExposure(
                component_id=component.id,
                component_name=component.name,
                network_zone=component.network_zone,
                exposed_ports=exposed_ports or [443],  # Default to HTTPS
                protocols=protocols or ["https"],
                firewall_protected=firewall_protected,
                ddos_protection=ddos_protection,
                tls_required=tls_required,
                ip_whitelist_enabled=component.network_zone == "internal",
            )
            network_exposures.append(network_exposure)

        logger.info("Mapped network exposure for %d components", len(network_exposures))
        return network_exposures

    def calculate_exposure_score(self, threat_model: ThreatModel) -> float:
        """Calculate composite exposure rating for the system.

        Combines endpoint, data store, authentication, and network
        exposure factors into a single score (0-10).

        Args:
            threat_model: The threat model to analyze.

        Returns:
            Exposure score from 0.0 (minimal) to 10.0 (maximum).
        """
        logger.info("Calculating exposure score for service: %s", threat_model.service_name)

        # Map attack surface elements
        endpoints = self.map_endpoints(threat_model)
        data_stores = self.map_data_stores(threat_model)
        auth_points = self.map_authentication_points(threat_model)
        network_exposures = self.map_network_exposure(threat_model)

        # Calculate component scores
        endpoint_score = self._calculate_endpoint_score(endpoints)
        data_store_score = self._calculate_data_store_score(data_stores)
        auth_score = self._calculate_auth_score(auth_points)
        network_score = self._calculate_network_score(network_exposures)

        # Weighted combination
        weights = {
            "endpoints": 0.30,
            "data_stores": 0.25,
            "authentication": 0.25,
            "network": 0.20,
        }

        exposure_score = (
            endpoint_score * weights["endpoints"]
            + data_store_score * weights["data_stores"]
            + auth_score * weights["authentication"]
            + network_score * weights["network"]
        )

        logger.info(
            "Exposure score calculated: %.2f (endpoints=%.2f, data=%.2f, auth=%.2f, network=%.2f)",
            exposure_score,
            endpoint_score,
            data_store_score,
            auth_score,
            network_score,
        )

        return min(10.0, max(0.0, exposure_score))

    def generate_attack_surface_report(self, threat_model: ThreatModel) -> AttackSurfaceReport:
        """Generate a comprehensive attack surface report.

        Creates a detailed report including all attack surface elements,
        exposure score, and recommendations for reducing the attack surface.

        Args:
            threat_model: The threat model to analyze.

        Returns:
            Complete AttackSurfaceReport.
        """
        logger.info("Generating attack surface report for service: %s", threat_model.service_name)
        start_time = datetime.now(timezone.utc)

        # Map all attack surface elements
        endpoints = self.map_endpoints(threat_model)
        data_stores = self.map_data_stores(threat_model)
        auth_points = self.map_authentication_points(threat_model)
        network_exposures = self.map_network_exposure(threat_model)

        # Calculate exposure score
        exposure_score = self.calculate_exposure_score(threat_model)

        # Determine risk level
        if exposure_score >= 8.0:
            risk_level = "critical"
        elif exposure_score >= 6.0:
            risk_level = "high"
        elif exposure_score >= 4.0:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Generate recommendations
        recommendations = self._generate_recommendations(
            endpoints, data_stores, auth_points, network_exposures
        )

        # Identify high-risk areas
        high_risk_areas = self._identify_high_risk_areas(
            endpoints, data_stores, auth_points, network_exposures
        )

        report = AttackSurfaceReport(
            service_name=threat_model.service_name,
            generated_at=start_time,
            total_endpoints=len(endpoints),
            total_data_stores=len(data_stores),
            total_auth_points=len(auth_points),
            exposure_score=exposure_score,
            risk_level=risk_level,
            endpoints=endpoints,
            data_stores=data_stores,
            authentication_points=auth_points,
            network_exposures=network_exposures,
            recommendations=recommendations,
            high_risk_areas=high_risk_areas,
        )

        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Attack surface report generated: exposure=%.2f, risk=%s, duration_ms=%.2f",
            exposure_score,
            risk_level,
            duration_ms,
        )

        return report

    def _determine_exposure_level(self, component: Component) -> ExposureLevel:
        """Determine exposure level for a component.

        Args:
            component: The component to analyze.

        Returns:
            ExposureLevel for the component.
        """
        if component.is_external:
            return ExposureLevel.INTERNET
        if component.network_zone == "dmz":
            return ExposureLevel.INTERNET
        if component.network_zone == "external":
            return ExposureLevel.INTERNET
        if component.network_zone == "partner":
            return ExposureLevel.PARTNER
        if component.network_zone == "internal":
            return ExposureLevel.INTERNAL
        return ExposureLevel.PRIVATE

    def _calculate_endpoint_score(self, endpoints: List[Endpoint]) -> float:
        """Calculate exposure score for endpoints.

        Args:
            endpoints: List of endpoints.

        Returns:
            Endpoint exposure score (0-10).
        """
        if not endpoints:
            return 0.0

        score = 0.0
        for endpoint in endpoints:
            # Base score per endpoint
            ep_score = 1.0

            # Increase for internet exposure
            if endpoint.exposure_level == ExposureLevel.INTERNET:
                ep_score += 2.0

            # Increase for sensitive data
            if endpoint.data_classification in (DataClassification.RESTRICTED, DataClassification.SECRET):
                ep_score += 1.5

            # Decrease for authentication
            if endpoint.authentication_required:
                ep_score -= 0.5

            # Decrease for rate limiting
            if endpoint.rate_limited:
                ep_score -= 0.3

            score += max(0, ep_score)

        # Normalize to 0-10
        return min(10.0, score / len(endpoints) * 2)

    def _calculate_data_store_score(self, data_stores: List[DataStore]) -> float:
        """Calculate exposure score for data stores.

        Args:
            data_stores: List of data stores.

        Returns:
            Data store exposure score (0-10).
        """
        if not data_stores:
            return 0.0

        score = 0.0
        for store in data_stores:
            # Base score per store
            ds_score = 1.0

            # Increase for sensitive data
            if store.contains_pii:
                ds_score += 2.0
            if store.contains_secrets:
                ds_score += 2.0

            # Increase for network exposure
            if store.network_exposure == ExposureLevel.INTERNET:
                ds_score += 3.0

            # Decrease for encryption
            if store.encryption_at_rest:
                ds_score -= 1.0

            # Decrease for access control
            if store.access_control in ("rbac", "abac"):
                ds_score -= 0.5

            score += max(0, ds_score)

        return min(10.0, score / len(data_stores) * 2)

    def _calculate_auth_score(self, auth_points: List[AuthenticationPoint]) -> float:
        """Calculate exposure score for authentication points.

        Args:
            auth_points: List of authentication points.

        Returns:
            Authentication exposure score (0-10).
        """
        if not auth_points:
            return 2.0  # No auth points is concerning

        score = 0.0
        for auth in auth_points:
            # Base score per auth point
            auth_score = 1.0

            # Increase for internet exposure
            if auth.exposure_level == ExposureLevel.INTERNET:
                auth_score += 2.0

            # Decrease for MFA
            if auth.mfa_enabled:
                auth_score -= 1.0

            # Decrease for brute force protection
            if auth.brute_force_protection:
                auth_score -= 0.5

            # Increase for weak auth methods
            if auth.auth_method in ("api_key", "basic"):
                auth_score += 1.0

            score += max(0, auth_score)

        return min(10.0, score / len(auth_points) * 2)

    def _calculate_network_score(self, network_exposures: List[NetworkExposure]) -> float:
        """Calculate exposure score for network exposure.

        Args:
            network_exposures: List of network exposures.

        Returns:
            Network exposure score (0-10).
        """
        if not network_exposures:
            return 0.0

        score = 0.0
        for exposure in network_exposures:
            # Base score
            net_score = 1.0

            # Increase for external network zone
            if exposure.network_zone in ("external", "dmz"):
                net_score += 2.0

            # Increase for number of exposed ports
            net_score += len(exposure.exposed_ports) * 0.2

            # Decrease for protections
            if exposure.firewall_protected:
                net_score -= 0.5
            if exposure.ddos_protection:
                net_score -= 0.5
            if exposure.tls_required:
                net_score -= 0.3
            if exposure.ip_whitelist_enabled:
                net_score -= 0.3

            score += max(0, net_score)

        return min(10.0, score / len(network_exposures) * 2)

    def _generate_recommendations(
        self,
        endpoints: List[Endpoint],
        data_stores: List[DataStore],
        auth_points: List[AuthenticationPoint],
        network_exposures: List[NetworkExposure],
    ) -> List[str]:
        """Generate recommendations for reducing attack surface.

        Args:
            endpoints: List of endpoints.
            data_stores: List of data stores.
            auth_points: List of authentication points.
            network_exposures: List of network exposures.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        # Endpoint recommendations
        unauthenticated = [e for e in endpoints if not e.authentication_required]
        if unauthenticated:
            recommendations.append(
                f"Implement authentication for {len(unauthenticated)} unauthenticated endpoints"
            )

        unrated = [e for e in endpoints if not e.rate_limited]
        if unrated:
            recommendations.append(
                f"Add rate limiting to {len(unrated)} endpoints"
            )

        # Data store recommendations
        unencrypted = [d for d in data_stores if not d.encryption_at_rest and d.contains_pii]
        if unencrypted:
            recommendations.append(
                f"Enable encryption at rest for {len(unencrypted)} data stores containing PII"
            )

        # Auth recommendations
        no_mfa = [a for a in auth_points if not a.mfa_enabled and a.exposure_level == ExposureLevel.INTERNET]
        if no_mfa:
            recommendations.append(
                f"Enable MFA for {len(no_mfa)} internet-exposed authentication points"
            )

        # Network recommendations
        no_firewall = [n for n in network_exposures if not n.firewall_protected and n.network_zone != "private"]
        if no_firewall:
            recommendations.append(
                f"Add firewall protection for {len(no_firewall)} exposed components"
            )

        # General recommendations
        if not recommendations:
            recommendations.append("Attack surface is well managed. Continue regular reviews.")

        return recommendations

    def _identify_high_risk_areas(
        self,
        endpoints: List[Endpoint],
        data_stores: List[DataStore],
        auth_points: List[AuthenticationPoint],
        network_exposures: List[NetworkExposure],
    ) -> List[Dict[str, Any]]:
        """Identify high-risk areas in the attack surface.

        Args:
            endpoints: List of endpoints.
            data_stores: List of data stores.
            auth_points: List of authentication points.
            network_exposures: List of network exposures.

        Returns:
            List of high-risk area dictionaries.
        """
        high_risk: List[Dict[str, Any]] = []

        # High-risk endpoints
        for endpoint in endpoints:
            if (
                endpoint.exposure_level == ExposureLevel.INTERNET
                and not endpoint.authentication_required
                and endpoint.data_classification in (DataClassification.RESTRICTED, DataClassification.SECRET)
            ):
                high_risk.append({
                    "type": "endpoint",
                    "name": endpoint.path,
                    "component": endpoint.component_name,
                    "risk": "Unauthenticated endpoint with sensitive data exposed to internet",
                    "severity": "critical",
                })

        # High-risk data stores
        for store in data_stores:
            if store.network_exposure == ExposureLevel.INTERNET and store.contains_pii:
                high_risk.append({
                    "type": "data_store",
                    "name": store.name,
                    "risk": "Data store with PII exposed to internet",
                    "severity": "critical",
                })

        # High-risk auth points
        for auth in auth_points:
            if auth.exposure_level == ExposureLevel.INTERNET and not auth.brute_force_protection:
                high_risk.append({
                    "type": "authentication",
                    "name": auth.name,
                    "risk": "Internet-exposed auth without brute force protection",
                    "severity": "high",
                })

        return high_risk


__all__ = [
    "AttackSurfaceMapper",
    "AttackSurfaceReport",
    "Endpoint",
    "DataStore",
    "AuthenticationPoint",
    "NetworkExposure",
    "ExposureLevel",
]
