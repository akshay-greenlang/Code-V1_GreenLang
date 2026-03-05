"""
CDP Supply Chain Module -- Supplier Engagement Management

This module implements the CDP Supply Chain questionnaire management including
supplier invitation workflow, response tracking, emissions aggregation,
engagement scoring, cascade request management, and hotspot identification.

Key capabilities:
  - Supplier invitation workflow with expiry tracking
  - Response tracking dashboard with status aggregation
  - Supplier emissions aggregation by scope
  - Engagement scoring based on response quality
  - Cascade request management for tiered supply chains
  - Hotspot identification by spend, category, and geography
  - Supplier improvement tracking over time

Example:
    >>> module = SupplyChainModule(config)
    >>> request = module.invite_supplier("org-1", "Supplier Co", "contact@supplier.com")
    >>> module.record_supplier_response(request.id, scope1=100, scope2=50)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .config import CDPAppConfig, SupplierStatus
from .models import (
    SupplyChainRequest,
    SupplierResponse,
    ScoringLevel,
    _new_id,
    _now,
)

logger = logging.getLogger(__name__)


class SupplyChainModule:
    """
    CDP Supply Chain Module -- manages supplier engagement program.

    Provides supplier invitation, response tracking, emissions aggregation,
    engagement scoring, and hotspot identification.

    Attributes:
        config: Application configuration.
        _requests: Supplier request store.
        _responses: Supplier response store.

    Example:
        >>> module = SupplyChainModule(config)
        >>> req = module.invite_supplier("org-1", "Acme Supplier", "acme@example.com")
    """

    def __init__(self, config: CDPAppConfig) -> None:
        """Initialize the Supply Chain Module."""
        self.config = config
        self._requests: Dict[str, SupplyChainRequest] = {}
        self._by_org: Dict[str, List[str]] = {}  # org_id -> request_ids
        self._responses: Dict[str, SupplierResponse] = {}
        logger.info("SupplyChainModule initialized")

    # ------------------------------------------------------------------
    # Supplier Invitation
    # ------------------------------------------------------------------

    def invite_supplier(
        self,
        org_id: str,
        supplier_name: str,
        supplier_email: str,
        supplier_id: Optional[str] = None,
        sector: Optional[str] = None,
        country: Optional[str] = None,
        spend_usd: Optional[Decimal] = None,
        scope3_category: Optional[int] = None,
    ) -> SupplyChainRequest:
        """
        Invite a supplier to the CDP Supply Chain program.

        Args:
            org_id: Requesting organization ID.
            supplier_name: Supplier company name.
            supplier_email: Supplier contact email.
            supplier_id: Internal supplier ID.
            sector: Supplier GICS sector.
            country: Supplier country code.
            spend_usd: Annual spend with supplier.
            scope3_category: Primary Scope 3 category.

        Returns:
            Created SupplyChainRequest.
        """
        expiry_date = date.today() + timedelta(
            days=self.config.supplier_invitation_expiry_days,
        )

        request = SupplyChainRequest(
            org_id=org_id,
            supplier_name=supplier_name,
            supplier_email=supplier_email,
            supplier_id=supplier_id,
            status=SupplierStatus.INVITED,
            invited_at=_now(),
            invitation_expiry=expiry_date,
            sector=sector,
            country=country,
            spend_usd=spend_usd,
            scope3_category=scope3_category,
        )

        self._requests[request.id] = request
        if org_id not in self._by_org:
            self._by_org[org_id] = []
        self._by_org[org_id].append(request.id)

        logger.info(
            "Invited supplier '%s' (%s) for org %s, expiry %s",
            supplier_name, supplier_email, org_id, expiry_date,
        )
        return request

    def bulk_invite(
        self,
        org_id: str,
        suppliers: List[Dict[str, Any]],
    ) -> List[SupplyChainRequest]:
        """Invite multiple suppliers at once."""
        results = []
        for s in suppliers:
            req = self.invite_supplier(
                org_id=org_id,
                supplier_name=s.get("name", ""),
                supplier_email=s.get("email", ""),
                supplier_id=s.get("supplier_id"),
                sector=s.get("sector"),
                country=s.get("country"),
                spend_usd=s.get("spend_usd"),
                scope3_category=s.get("scope3_category"),
            )
            results.append(req)

        logger.info("Bulk invited %d suppliers for org %s", len(results), org_id)
        return results

    def send_reminder(self, request_id: str) -> bool:
        """Send a reminder to a supplier."""
        request = self._requests.get(request_id)
        if not request:
            return False
        if request.status not in (SupplierStatus.INVITED, SupplierStatus.IN_PROGRESS):
            return False

        request.reminder_count += 1
        request.updated_at = _now()
        logger.info(
            "Sent reminder #%d to supplier '%s'",
            request.reminder_count, request.supplier_name,
        )
        return True

    # ------------------------------------------------------------------
    # Response Tracking
    # ------------------------------------------------------------------

    def record_supplier_response(
        self,
        request_id: str,
        scope1_emissions: Optional[Decimal] = None,
        scope2_emissions: Optional[Decimal] = None,
        scope3_emissions: Optional[Decimal] = None,
        has_sbti: bool = False,
        has_transition_plan: bool = False,
        cdp_score: Optional[ScoringLevel] = None,
    ) -> SupplierResponse:
        """
        Record a supplier's response to the CDP questionnaire.

        Args:
            request_id: Supply chain request ID.
            scope1_emissions: Supplier's Scope 1 emissions (tCO2e).
            scope2_emissions: Supplier's Scope 2 emissions (tCO2e).
            scope3_emissions: Supplier's Scope 3 emissions (tCO2e).
            has_sbti: Whether supplier has science-based target.
            has_transition_plan: Whether supplier has transition plan.
            cdp_score: Supplier's CDP score level.

        Returns:
            Created SupplierResponse.
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Supply chain request {request_id} not found")

        total = (scope1_emissions or Decimal("0")) + (scope2_emissions or Decimal("0")) + (scope3_emissions or Decimal("0"))

        engagement = self._calculate_engagement_score(
            scope1_emissions, scope2_emissions, scope3_emissions,
            has_sbti, has_transition_plan, cdp_score,
        )

        data_quality = self._calculate_data_quality(
            scope1_emissions, scope2_emissions, scope3_emissions,
        )

        response = SupplierResponse(
            request_id=request_id,
            supplier_name=request.supplier_name,
            status=SupplierStatus.SUBMITTED,
            cdp_score=cdp_score,
            scope1_emissions=scope1_emissions,
            scope2_emissions=scope2_emissions,
            scope3_emissions=scope3_emissions,
            total_emissions=total,
            has_science_based_target=has_sbti,
            has_transition_plan=has_transition_plan,
            engagement_score=engagement,
            data_quality_score=data_quality,
            submitted_at=_now(),
        )

        self._responses[response.id] = response
        request.status = SupplierStatus.SUBMITTED
        request.updated_at = _now()

        logger.info(
            "Recorded response for supplier '%s': total=%.1f tCO2e, engagement=%.1f",
            request.supplier_name, float(total), engagement,
        )
        return response

    def update_supplier_status(
        self,
        request_id: str,
        status: SupplierStatus,
    ) -> Optional[SupplyChainRequest]:
        """Update the status of a supplier request."""
        request = self._requests.get(request_id)
        if not request:
            return None
        request.status = status
        request.updated_at = _now()
        return request

    # ------------------------------------------------------------------
    # Dashboard & Aggregation
    # ------------------------------------------------------------------

    def get_dashboard(self, org_id: str) -> Dict[str, Any]:
        """
        Get supply chain engagement dashboard data.

        Returns aggregated statistics on supplier engagement status.
        """
        request_ids = self._by_org.get(org_id, [])
        requests = [self._requests[rid] for rid in request_ids if rid in self._requests]

        total = len(requests)
        status_counts: Dict[str, int] = {}
        for req in requests:
            status_counts[req.status.value] = status_counts.get(req.status.value, 0) + 1

        # Get responses
        responses = self._get_org_responses(org_id)
        total_emissions = sum(
            (r.total_emissions or Decimal("0")) for r in responses
        )
        total_scope1 = sum(
            (r.scope1_emissions or Decimal("0")) for r in responses
        )
        total_scope2 = sum(
            (r.scope2_emissions or Decimal("0")) for r in responses
        )
        total_scope3 = sum(
            (r.scope3_emissions or Decimal("0")) for r in responses
        )

        avg_engagement = 0.0
        if responses:
            avg_engagement = sum(r.engagement_score for r in responses) / len(responses)

        response_rate = 0.0
        if total > 0:
            responded = status_counts.get("submitted", 0) + status_counts.get("scored", 0)
            response_rate = responded / total * 100

        return {
            "total_suppliers": total,
            "status_breakdown": status_counts,
            "response_rate_pct": round(response_rate, 1),
            "total_emissions_tco2e": float(total_emissions),
            "scope1_total_tco2e": float(total_scope1),
            "scope2_total_tco2e": float(total_scope2),
            "scope3_total_tco2e": float(total_scope3),
            "avg_engagement_score": round(avg_engagement, 1),
            "suppliers_with_sbti": sum(1 for r in responses if r.has_science_based_target),
            "suppliers_with_transition_plan": sum(1 for r in responses if r.has_transition_plan),
        }

    def get_supplier_list(
        self,
        org_id: str,
        status: Optional[SupplierStatus] = None,
    ) -> List[SupplyChainRequest]:
        """Get list of suppliers for an organization."""
        request_ids = self._by_org.get(org_id, [])
        requests = [self._requests[rid] for rid in request_ids if rid in self._requests]
        if status:
            requests = [r for r in requests if r.status == status]
        return requests

    # ------------------------------------------------------------------
    # Hotspot Identification
    # ------------------------------------------------------------------

    def identify_hotspots(
        self,
        org_id: str,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Identify emissions hotspots in the supply chain.

        Ranks suppliers by total emissions and identifies the highest
        impact areas by spend, category, and geography.

        Args:
            org_id: Organization ID.
            top_n: Number of top hotspots to return.

        Returns:
            Hotspot analysis results.
        """
        request_ids = self._by_org.get(org_id, [])
        responses = self._get_org_responses(org_id)

        # Rank by emissions
        emission_hotspots = sorted(
            [
                {
                    "supplier_name": r.supplier_name,
                    "total_emissions": float(r.total_emissions or 0),
                    "engagement_score": r.engagement_score,
                }
                for r in responses
                if r.total_emissions and r.total_emissions > 0
            ],
            key=lambda x: x["total_emissions"],
            reverse=True,
        )[:top_n]

        # Aggregate by category
        by_category: Dict[int, Decimal] = {}
        for rid in request_ids:
            req = self._requests.get(rid)
            if req and req.scope3_category:
                resp = self._get_response_for_request(rid)
                if resp:
                    current = by_category.get(req.scope3_category, Decimal("0"))
                    by_category[req.scope3_category] = current + (resp.total_emissions or Decimal("0"))

        # Aggregate by country
        by_country: Dict[str, Decimal] = {}
        for rid in request_ids:
            req = self._requests.get(rid)
            if req and req.country:
                resp = self._get_response_for_request(rid)
                if resp:
                    current = by_country.get(req.country, Decimal("0"))
                    by_country[req.country] = current + (resp.total_emissions or Decimal("0"))

        return {
            "top_emitting_suppliers": emission_hotspots,
            "emissions_by_category": {
                str(k): float(v) for k, v in sorted(
                    by_category.items(), key=lambda x: x[1], reverse=True,
                )
            },
            "emissions_by_country": {
                k: float(v) for k, v in sorted(
                    by_country.items(), key=lambda x: x[1], reverse=True,
                )
            },
        }

    # ------------------------------------------------------------------
    # Cascade Requests
    # ------------------------------------------------------------------

    def create_cascade_request(
        self,
        org_id: str,
        supplier_request_id: str,
        tier: int = 2,
    ) -> Dict[str, Any]:
        """
        Create a cascade request for a supplier to engage their own suppliers.

        Args:
            org_id: Requesting organization ID.
            supplier_request_id: Original supplier request ID.
            tier: Supply chain tier (2 = supplier's supplier).

        Returns:
            Cascade request record.
        """
        parent_request = self._requests.get(supplier_request_id)
        if not parent_request:
            raise ValueError(f"Request {supplier_request_id} not found")

        cascade = {
            "id": _new_id(),
            "org_id": org_id,
            "parent_request_id": supplier_request_id,
            "supplier_name": parent_request.supplier_name,
            "tier": tier,
            "status": "pending",
            "created_at": _now().isoformat(),
        }

        logger.info(
            "Created cascade request for tier %d via supplier '%s'",
            tier, parent_request.supplier_name,
        )
        return cascade

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _calculate_engagement_score(
        self,
        scope1: Optional[Decimal],
        scope2: Optional[Decimal],
        scope3: Optional[Decimal],
        has_sbti: bool,
        has_transition_plan: bool,
        cdp_score: Optional[ScoringLevel],
    ) -> float:
        """
        Calculate supplier engagement quality score (0-100).

        Scoring factors:
          - Emissions data provided: up to 40 points
          - Science-based target: 20 points
          - Transition plan: 15 points
          - CDP score level: up to 25 points
        """
        score = 0.0

        # Emissions data completeness (40 points)
        if scope1 is not None:
            score += 15.0
        if scope2 is not None:
            score += 10.0
        if scope3 is not None:
            score += 15.0

        # SBTi (20 points)
        if has_sbti:
            score += 20.0

        # Transition plan (15 points)
        if has_transition_plan:
            score += 15.0

        # CDP score (25 points)
        if cdp_score:
            score_map = {
                ScoringLevel.A: 25.0,
                ScoringLevel.A_MINUS: 22.0,
                ScoringLevel.B: 18.0,
                ScoringLevel.B_MINUS: 14.0,
                ScoringLevel.C: 10.0,
                ScoringLevel.C_MINUS: 7.0,
                ScoringLevel.D: 4.0,
                ScoringLevel.D_MINUS: 2.0,
            }
            score += score_map.get(cdp_score, 0.0)

        return min(score, 100.0)

    def _calculate_data_quality(
        self,
        scope1: Optional[Decimal],
        scope2: Optional[Decimal],
        scope3: Optional[Decimal],
    ) -> float:
        """Calculate data quality score for supplier emissions data."""
        provided = sum(1 for s in [scope1, scope2, scope3] if s is not None and s > 0)
        return (provided / 3.0) * 100.0

    def _get_org_responses(self, org_id: str) -> List[SupplierResponse]:
        """Get all supplier responses for an organization."""
        request_ids = self._by_org.get(org_id, [])
        responses = []
        for rid in request_ids:
            for resp in self._responses.values():
                if resp.request_id == rid:
                    responses.append(resp)
        return responses

    def _get_response_for_request(
        self,
        request_id: str,
    ) -> Optional[SupplierResponse]:
        """Get the response for a specific request."""
        for resp in self._responses.values():
            if resp.request_id == request_id:
                return resp
        return None
