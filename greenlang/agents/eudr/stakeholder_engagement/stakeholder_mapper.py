# -*- coding: utf-8 -*-
"""
StakeholderMapper Engine - AGENT-EUDR-031

Centralized registry of all stakeholders across the EUDR supply chain.
Provides stakeholder registration, auto-discovery from supply chain
graphs, categorization, rights classification, and relationship
lifecycle management.

Zero-Hallucination: All categorization and rights classification use
deterministic rule-based logic. No LLM involvement.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-031 (GL-EUDR-SET-031)
Regulation: EU 2023/1115 (EUDR), ILO Convention 169, UNDRIP
Status: Production Ready
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from greenlang.agents.eudr.stakeholder_engagement.config import (
    StakeholderEngagementConfig,
)
from greenlang.agents.eudr.stakeholder_engagement.models import (
    EUDRCommodity,
    RightsClassification,
    StakeholderCategory,
    StakeholderRecord,
    StakeholderStatus,
)
from greenlang.agents.eudr.stakeholder_engagement.provenance import (
    ProvenanceTracker,
)

logger = logging.getLogger(__name__)

# Legal framework references by country
COUNTRY_LEGAL_FRAMEWORKS = {
    "CO": "Colombian Constitution Art. 330, Law 21/1991",
    "BR": "Brazilian Constitution Art. 231, Statute of the Indian (Law 6001/1973)",
    "ID": "Indonesian Law 41/1999 on Forestry, Constitutional Court Decision 35/2012",
    "GH": "Ghanaian Customary Land Law, Land Act 2020",
    "PE": "Peru ILO 169 Ratification (Law 29785)",
    "MY": "Malaysian Aboriginal Peoples Act 1954",
    "CG": "Republic of Congo Forest Code 2020",
    "CM": "Cameroon Law 94/01 on Forestry",
}


class StakeholderMapper:
    """Stakeholder mapping and registration engine.

    Registers, categorizes, and manages stakeholder records
    for EUDR due diligence compliance.

    Attributes:
        _config: Engine configuration.
        _provenance: Provenance hash chain tracker.
        _stakeholders: In-memory stakeholder store.
    """

    def __init__(self, config: StakeholderEngagementConfig) -> None:
        """Initialize StakeholderMapper.

        Args:
            config: Stakeholder engagement configuration.
        """
        self._config = config
        self._provenance = ProvenanceTracker()
        self._stakeholders: Dict[str, StakeholderRecord] = {}
        logger.info("StakeholderMapper initialized")

    async def map_stakeholder(
        self,
        operator_id: str,
        name: str,
        category: StakeholderCategory,
        country_code: str,
        region: str,
        commodity: EUDRCommodity,
        contact_info: Dict[str, Any],
        rights_classification: RightsClassification,
        population_estimate: Optional[int] = None,
        affected_area_hectares: Optional[Decimal] = None,
    ) -> StakeholderRecord:
        """Register a new stakeholder.

        Args:
            operator_id: Operator registering the stakeholder.
            name: Stakeholder name.
            category: Stakeholder category.
            country_code: ISO 3166-1 country code.
            region: Geographic region.
            commodity: Associated EUDR commodity.
            contact_info: Contact information dictionary.
            rights_classification: Rights classification.
            population_estimate: Estimated population if community.
            affected_area_hectares: Affected area in hectares.

        Returns:
            Newly created StakeholderRecord.

        Raises:
            ValueError: If required fields are empty.
        """
        if not operator_id or not operator_id.strip():
            raise ValueError("operator_id is required")
        if not name or not name.strip():
            raise ValueError("name is required")

        stakeholder_id = f"STK-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now(tz=timezone.utc)

        record = StakeholderRecord(
            stakeholder_id=stakeholder_id,
            operator_id=operator_id,
            name=name,
            category=category,
            status=StakeholderStatus.ACTIVE,
            country_code=country_code,
            region=region,
            commodity=commodity,
            contact_info=contact_info,
            rights_classification=rights_classification,
            population_estimate=population_estimate,
            affected_area_hectares=affected_area_hectares,
            engagement_history=[],
            notes="",
            created_at=now,
            updated_at=now,
        )

        self._stakeholders[stakeholder_id] = record
        self._provenance.record(
            "stakeholder", "register", stakeholder_id, "AGENT-EUDR-031",
            metadata={"category": category.value},
        )
        logger.info("Stakeholder %s registered: %s", stakeholder_id, name)
        return record

    async def discover_from_supply_chain(
        self,
        supply_chain_data: Dict[str, Any],
    ) -> List[StakeholderRecord]:
        """Discover stakeholders from supply chain data.

        Args:
            supply_chain_data: Supply chain information including
                operator_id, commodity, country, suppliers, etc.

        Returns:
            List of discovered StakeholderRecord instances.

        Raises:
            ValueError: If operator_id is missing.
        """
        if not supply_chain_data:
            return []

        operator_id = supply_chain_data.get("operator_id", "")
        if not operator_id:
            raise ValueError("operator_id is required in supply_chain_data")

        commodity_str = supply_chain_data.get("commodity", "coffee")
        country = supply_chain_data.get("country", "")
        suppliers = supply_chain_data.get("suppliers", [])
        indigenous_territories = supply_chain_data.get("indigenous_territories", [])

        results: List[StakeholderRecord] = []

        # Discover stakeholders from suppliers
        for supplier in suppliers:
            community = supplier.get("community", supplier.get("name", ""))
            if not community:
                continue
            region = supplier.get("region", "")

            try:
                commodity_enum = EUDRCommodity(commodity_str)
            except ValueError:
                commodity_enum = EUDRCommodity.COFFEE

            record = StakeholderRecord(
                stakeholder_id=f"STK-{uuid.uuid4().hex[:8].upper()}",
                operator_id=operator_id,
                name=f"Community near {community}",
                category=StakeholderCategory.LOCAL_COMMUNITY,
                status=StakeholderStatus.ACTIVE,
                country_code=country,
                region=region,
                commodity=commodity_enum,
                contact_info={},
                rights_classification=RightsClassification(
                    has_land_rights=False,
                    has_customary_rights=False,
                    has_indigenous_status=False,
                    fpic_required=False,
                    applicable_conventions=[],
                    legal_framework="",
                ),
            )
            self._stakeholders[record.stakeholder_id] = record
            results.append(record)

        # Discover stakeholders from indigenous territories
        for territory in indigenous_territories:
            territory_name = territory.get("name", "Unknown Community")
            try:
                commodity_enum = EUDRCommodity(commodity_str)
            except ValueError:
                commodity_enum = EUDRCommodity.COFFEE

            rights = RightsClassification(
                has_land_rights=True,
                has_customary_rights=True,
                has_indigenous_status=True,
                fpic_required=True,
                applicable_conventions=["ILO 169", "UNDRIP"],
                legal_framework=COUNTRY_LEGAL_FRAMEWORKS.get(country, ""),
            )

            record = StakeholderRecord(
                stakeholder_id=f"STK-{uuid.uuid4().hex[:8].upper()}",
                operator_id=operator_id,
                name=territory_name,
                category=StakeholderCategory.INDIGENOUS_COMMUNITY,
                status=StakeholderStatus.ACTIVE,
                country_code=country,
                region="",
                commodity=commodity_enum,
                contact_info={},
                rights_classification=rights,
                affected_area_hectares=Decimal(str(territory.get("overlap_hectares", 0))),
            )
            self._stakeholders[record.stakeholder_id] = record
            results.append(record)

        return results

    def categorize_stakeholder(self, profile: Dict[str, Any]) -> StakeholderCategory:
        """Categorize a stakeholder based on profile attributes.

        Args:
            profile: Stakeholder profile dictionary with keys like
                indigenous_status, organization_type, farm_size_hectares.

        Returns:
            Appropriate StakeholderCategory.
        """
        if profile.get("indigenous_status", False):
            return StakeholderCategory.INDIGENOUS_COMMUNITY

        org_type = profile.get("organization_type", "").lower()
        if org_type == "cooperative":
            return StakeholderCategory.COOPERATIVE
        if org_type == "ngo":
            return StakeholderCategory.NGO
        if org_type == "government":
            return StakeholderCategory.GOVERNMENT_AGENCY
        if org_type == "certification_body":
            return StakeholderCategory.CERTIFICATION_BODY
        if org_type == "worker_union":
            return StakeholderCategory.WORKER_UNION

        farm_size = profile.get("farm_size_hectares", None)
        if org_type == "individual" and farm_size is not None:
            if float(farm_size) < 10:
                return StakeholderCategory.SMALLHOLDER

        if profile.get("community_type") or profile.get("population"):
            return StakeholderCategory.LOCAL_COMMUNITY

        return StakeholderCategory.OTHER

    def classify_rights(self, profile: Dict[str, Any]) -> RightsClassification:
        """Classify rights for a stakeholder based on profile.

        Args:
            profile: Stakeholder profile dictionary.

        Returns:
            RightsClassification with applicable rights.
        """
        is_indigenous = profile.get("indigenous_status", False)
        has_land = profile.get("land_rights", False)
        has_customary = profile.get("customary_rights", False)
        country_code = profile.get("country_code", "")

        fpic_required = is_indigenous
        conventions: List[str] = []
        legal_framework = ""

        if is_indigenous:
            conventions = ["ILO 169", "UNDRIP"]
            legal_framework = COUNTRY_LEGAL_FRAMEWORKS.get(country_code, "")
            if not legal_framework and country_code:
                legal_framework = f"National legislation of {country_code}"

        if has_land and not legal_framework and country_code:
            legal_framework = COUNTRY_LEGAL_FRAMEWORKS.get(
                country_code, f"Land rights law of {country_code}"
            )

        return RightsClassification(
            has_land_rights=has_land or is_indigenous,
            has_customary_rights=has_customary,
            has_indigenous_status=is_indigenous,
            fpic_required=fpic_required,
            applicable_conventions=conventions,
            legal_framework=legal_framework,
        )

    async def get_stakeholder(
        self,
        stakeholder_id: str,
    ) -> Optional[StakeholderRecord]:
        """Retrieve a stakeholder by ID.

        Args:
            stakeholder_id: Stakeholder identifier.

        Returns:
            StakeholderRecord or None if not found.

        Raises:
            ValueError: If stakeholder_id is empty.
        """
        if not stakeholder_id or not stakeholder_id.strip():
            raise ValueError("stakeholder_id is required")

        return self._stakeholders.get(stakeholder_id)

    async def list_stakeholders(
        self,
        operator_id: str,
        category: Optional[StakeholderCategory] = None,
    ) -> List[StakeholderRecord]:
        """List stakeholders filtered by operator and optional category.

        Args:
            operator_id: Operator identifier to filter by.
            category: Optional category filter.

        Returns:
            List of matching StakeholderRecord instances.

        Raises:
            ValueError: If operator_id is empty.
        """
        if not operator_id or not operator_id.strip():
            raise ValueError("operator_id is required")

        results = [
            s for s in self._stakeholders.values()
            if s.operator_id == operator_id
        ]

        if category is not None:
            results = [s for s in results if s.category == category]

        return results
