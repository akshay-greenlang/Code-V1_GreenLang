# -*- coding: utf-8 -*-
"""
IndigenousRegistryEngine - Feature 6: Indigenous Community Registry

Manages a structured database of indigenous communities with legal
protections tracking, commodity relevance mapping, engagement history,
and privacy-controlled access. Supports CRUD operations with full
provenance and audit trail.

Per PRD F6.1-F6.4: community profiles, legal protections, ILO 169
coverage flags, representative organizations, and EUDR commodity
relevance tagging.

Example:
    >>> engine = IndigenousRegistryEngine(config, provenance)
    >>> community = await engine.register_community(
    ...     community_name="Yanomami",
    ...     people_name="Yanomami",
    ...     country_code="BR",
    ...     territory_ids=["t-001"],
    ... )
    >>> assert community.community_id is not None

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 (Feature 6)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.agents.eudr.indigenous_rights_checker.config import (
    IndigenousRightsCheckerConfig,
)
from greenlang.agents.eudr.indigenous_rights_checker.models import (
    CommunityRecognitionStatus,
    IndigenousCommunity,
)
from greenlang.agents.eudr.indigenous_rights_checker.provenance import (
    ProvenanceTracker,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL templates
# ---------------------------------------------------------------------------

_SQL_INSERT_COMMUNITY = """
    INSERT INTO eudr_indigenous_rights_checker.gl_eudr_irc_communities (
        community_id, community_name, indigenous_name, people_name,
        language, estimated_population, country_code, region,
        territory_ids, legal_recognition_status,
        applicable_legal_protections, ilo_169_coverage,
        fpic_legal_requirement, representative_organizations,
        contact_channels, commodity_relevance,
        engagement_history_summary, data_source, provenance_hash
    ) VALUES (
        %(community_id)s, %(community_name)s, %(indigenous_name)s,
        %(people_name)s, %(language)s, %(estimated_population)s,
        %(country_code)s, %(region)s, %(territory_ids)s,
        %(legal_recognition_status)s, %(applicable_legal_protections)s,
        %(ilo_169_coverage)s, %(fpic_legal_requirement)s,
        %(representative_organizations)s, %(contact_channels)s,
        %(commodity_relevance)s, %(engagement_history_summary)s,
        %(data_source)s, %(provenance_hash)s
    )
"""

_SQL_GET_COMMUNITY = """
    SELECT community_id, community_name, indigenous_name, people_name,
           language, estimated_population, country_code, region,
           territory_ids, legal_recognition_status,
           applicable_legal_protections, ilo_169_coverage,
           fpic_legal_requirement, representative_organizations,
           contact_channels, commodity_relevance,
           engagement_history_summary, data_source, provenance_hash,
           created_at, updated_at
    FROM eudr_indigenous_rights_checker.gl_eudr_irc_communities
    WHERE community_id = %(community_id)s
"""

_SQL_SEARCH_COMMUNITIES = """
    SELECT community_id, community_name, indigenous_name, people_name,
           language, estimated_population, country_code, region,
           territory_ids, legal_recognition_status,
           applicable_legal_protections, ilo_169_coverage,
           fpic_legal_requirement, commodity_relevance,
           provenance_hash, created_at
    FROM eudr_indigenous_rights_checker.gl_eudr_irc_communities
    WHERE 1=1
"""

_SQL_UPDATE_COMMUNITY = """
    UPDATE eudr_indigenous_rights_checker.gl_eudr_irc_communities
    SET community_name = %(community_name)s,
        indigenous_name = %(indigenous_name)s,
        people_name = %(people_name)s,
        language = %(language)s,
        estimated_population = %(estimated_population)s,
        region = %(region)s,
        territory_ids = %(territory_ids)s,
        legal_recognition_status = %(legal_recognition_status)s,
        applicable_legal_protections = %(applicable_legal_protections)s,
        ilo_169_coverage = %(ilo_169_coverage)s,
        fpic_legal_requirement = %(fpic_legal_requirement)s,
        representative_organizations = %(representative_organizations)s,
        contact_channels = %(contact_channels)s,
        commodity_relevance = %(commodity_relevance)s,
        engagement_history_summary = %(engagement_history_summary)s,
        provenance_hash = %(provenance_hash)s,
        updated_at = NOW()
    WHERE community_id = %(community_id)s
"""

_SQL_COUNT_COMMUNITIES = """
    SELECT COUNT(*)
    FROM eudr_indigenous_rights_checker.gl_eudr_irc_communities
"""

_SQL_COMMUNITIES_BY_TERRITORY = """
    SELECT community_id, community_name, people_name, country_code,
           ilo_169_coverage, fpic_legal_requirement, commodity_relevance,
           provenance_hash
    FROM eudr_indigenous_rights_checker.gl_eudr_irc_communities
    WHERE territory_ids @> %(territory_id_json)s
"""


class IndigenousRegistryEngine:
    """Engine for managing indigenous community registry.

    Provides CRUD operations for indigenous community profiles with
    legal protections tracking, ILO 169 coverage, commodity relevance,
    privacy controls, and full provenance audit trail.

    Attributes:
        _config: Agent configuration.
        _provenance: Provenance tracker for audit trail.
        _pool: Async database connection pool.
    """

    def __init__(
        self,
        config: IndigenousRightsCheckerConfig,
        provenance: ProvenanceTracker,
    ) -> None:
        """Initialize IndigenousRegistryEngine."""
        self._config = config
        self._provenance = provenance
        self._pool: Any = None
        logger.info("IndigenousRegistryEngine initialized")

    async def startup(self, pool: Any) -> None:
        """Set the database connection pool."""
        self._pool = pool

    async def shutdown(self) -> None:
        """Clean up engine resources."""
        self._pool = None

    # -------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------

    async def register_community(
        self,
        community_name: str,
        people_name: str,
        country_code: str,
        indigenous_name: Optional[str] = None,
        language: Optional[str] = None,
        estimated_population: Optional[int] = None,
        region: Optional[str] = None,
        territory_ids: Optional[List[str]] = None,
        legal_recognition_status: Optional[str] = None,
        applicable_legal_protections: Optional[List[str]] = None,
        representative_organizations: Optional[List[Dict[str, Any]]] = None,
        contact_channels: Optional[List[Dict[str, Any]]] = None,
        commodity_relevance: Optional[List[str]] = None,
        data_source: Optional[str] = None,
    ) -> IndigenousCommunity:
        """Register a new indigenous community in the registry.

        Automatically determines ILO 169 coverage and FPIC legal
        requirements based on country code reference data.

        Args:
            community_name: Official community name.
            people_name: Indigenous people/ethnic group name.
            country_code: ISO 3166-1 alpha-2 code.
            indigenous_name: Optional name in indigenous language.
            language: Primary language.
            estimated_population: Estimated population.
            region: Administrative region.
            territory_ids: Linked territory identifiers.
            legal_recognition_status: CommunityRecognitionStatus value.
            applicable_legal_protections: List of legal protections.
            representative_organizations: Representative org details.
            contact_channels: Contact information.
            commodity_relevance: EUDR commodity codes.
            data_source: Data source identifier.

        Returns:
            IndigenousCommunity with auto-populated legal fields.
        """
        community_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Auto-determine ILO 169 coverage and FPIC requirement
        ilo_coverage = self._determine_ilo_169_coverage(country_code)
        fpic_required = self._determine_fpic_requirement(country_code)

        # Auto-populate legal protections if not provided
        legal_protections = (
            applicable_legal_protections
            or self._get_default_legal_protections(country_code)
        )

        # Determine recognition status
        recognition = None
        if legal_recognition_status:
            recognition = CommunityRecognitionStatus(legal_recognition_status)

        provenance_hash = self._provenance.compute_data_hash({
            "community_id": community_id,
            "community_name": community_name,
            "country_code": country_code,
            "created_at": now.isoformat(),
        })

        community = IndigenousCommunity(
            community_id=community_id,
            community_name=community_name,
            indigenous_name=indigenous_name,
            people_name=people_name,
            language=language,
            estimated_population=estimated_population,
            country_code=country_code,
            region=region,
            territory_ids=territory_ids or [],
            legal_recognition_status=recognition,
            applicable_legal_protections=legal_protections,
            ilo_169_coverage=ilo_coverage,
            fpic_legal_requirement=fpic_required,
            representative_organizations=representative_organizations or [],
            contact_channels=contact_channels or [],
            commodity_relevance=commodity_relevance or [],
            engagement_history_summary={},
            data_source=data_source,
            provenance_hash=provenance_hash,
            created_at=now,
            updated_at=now,
        )

        self._provenance.record(
            "community", "create", community_id,
            metadata={
                "community_name": community_name,
                "country_code": country_code,
                "ilo_169_coverage": ilo_coverage,
            },
        )

        await self._persist_community(community)

        logger.info(
            f"Community registered: {community_id} "
            f"name={community_name} country={country_code}"
        )

        return community

    async def get_community(
        self, community_id: str
    ) -> Optional[IndigenousCommunity]:
        """Get a community record by identifier.

        Args:
            community_id: Community UUID.

        Returns:
            IndigenousCommunity or None if not found.
        """
        if self._pool is None:
            return None

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    _SQL_GET_COMMUNITY,
                    {"community_id": community_id},
                )
                row = await cur.fetchone()

        if row is None:
            return None

        return self._row_to_community(row)

    async def search_communities(
        self,
        country_code: Optional[str] = None,
        people_name: Optional[str] = None,
        commodity: Optional[str] = None,
        ilo_169_only: bool = False,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Search communities with optional filters.

        Args:
            country_code: Filter by ISO 3166-1 alpha-2 code.
            people_name: Filter by people/ethnic group name (partial).
            commodity: Filter by EUDR commodity relevance.
            ilo_169_only: Only return ILO 169 covered communities.
            limit: Maximum records to return.

        Returns:
            List of community summary dictionaries.
        """
        if self._pool is None:
            return []

        # Build dynamic query with filters
        query = _SQL_SEARCH_COMMUNITIES
        params: Dict[str, Any] = {}

        if country_code:
            query += " AND country_code = %(country_code)s"
            params["country_code"] = country_code

        if people_name:
            query += " AND people_name ILIKE %(people_name)s"
            params["people_name"] = f"%{people_name}%"

        if commodity:
            query += " AND commodity_relevance @> %(commodity_json)s"
            params["commodity_json"] = json.dumps([commodity])

        if ilo_169_only:
            query += " AND ilo_169_coverage = TRUE"

        query += " ORDER BY created_at DESC LIMIT %(limit)s"
        params["limit"] = limit

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                rows = await cur.fetchall()

        return [
            {
                "community_id": str(row[0]),
                "community_name": row[1],
                "indigenous_name": row[2],
                "people_name": row[3],
                "language": row[4],
                "estimated_population": row[5],
                "country_code": row[6],
                "region": row[7],
                "ilo_169_coverage": row[11],
                "fpic_legal_requirement": row[12],
                "commodity_relevance": row[13],
                "provenance_hash": row[14],
                "created_at": row[15].isoformat() if row[15] else None,
            }
            for row in rows
        ]

    async def update_community(
        self,
        community_id: str,
        updates: Dict[str, Any],
    ) -> Optional[IndigenousCommunity]:
        """Update an existing community record.

        Args:
            community_id: Community UUID.
            updates: Dictionary of fields to update.

        Returns:
            Updated IndigenousCommunity or None if not found.
        """
        existing = await self.get_community(community_id)
        if existing is None:
            logger.warning(f"Community not found for update: {community_id}")
            return None

        # Apply updates
        updated_data = existing.model_dump()
        for key, value in updates.items():
            if key in updated_data and key != "community_id":
                updated_data[key] = value

        # Recompute provenance hash
        now = datetime.now(timezone.utc)
        provenance_hash = self._provenance.compute_data_hash({
            "community_id": community_id,
            "updated_fields": list(updates.keys()),
            "updated_at": now.isoformat(),
        })
        updated_data["provenance_hash"] = provenance_hash
        updated_data["updated_at"] = now

        updated_community = IndigenousCommunity(**updated_data)

        await self._persist_community_update(updated_community)

        self._provenance.record(
            "community", "update", community_id,
            metadata={"updated_fields": list(updates.keys())},
        )

        logger.info(
            f"Community updated: {community_id} "
            f"fields={list(updates.keys())}"
        )

        return updated_community

    async def get_communities_for_territory(
        self, territory_id: str
    ) -> List[Dict[str, Any]]:
        """Get communities linked to a specific territory.

        Args:
            territory_id: Territory UUID.

        Returns:
            List of community summary dictionaries.
        """
        if self._pool is None:
            return []

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    _SQL_COMMUNITIES_BY_TERRITORY,
                    {"territory_id_json": json.dumps([territory_id])},
                )
                rows = await cur.fetchall()

        return [
            {
                "community_id": str(row[0]),
                "community_name": row[1],
                "people_name": row[2],
                "country_code": row[3],
                "ilo_169_coverage": row[4],
                "fpic_legal_requirement": row[5],
                "commodity_relevance": row[6],
                "provenance_hash": row[7],
            }
            for row in rows
        ]

    async def get_community_count(self) -> int:
        """Get total count of registered communities.

        Returns:
            Total community count.
        """
        if self._pool is None:
            return 0

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_COUNT_COMMUNITIES)
                row = await cur.fetchone()

        return row[0] if row else 0

    # -------------------------------------------------------------------
    # Reference data helpers
    # -------------------------------------------------------------------

    def _determine_ilo_169_coverage(self, country_code: str) -> bool:
        """Determine ILO 169 coverage from reference data.

        Args:
            country_code: ISO 3166-1 alpha-2 code.

        Returns:
            True if country has ratified ILO 169.
        """
        from greenlang.agents.eudr.indigenous_rights_checker.reference_data.ilo_169_countries import (
            is_ilo_169_ratified,
        )
        return is_ilo_169_ratified(country_code)

    def _determine_fpic_requirement(self, country_code: str) -> bool:
        """Determine FPIC legal requirement from reference data.

        Args:
            country_code: ISO 3166-1 alpha-2 code.

        Returns:
            True if FPIC is legally required.
        """
        from greenlang.agents.eudr.indigenous_rights_checker.reference_data.fpic_legal_frameworks import (
            is_fpic_legally_required,
        )
        return is_fpic_legally_required(country_code)

    def _get_default_legal_protections(
        self, country_code: str
    ) -> List[str]:
        """Get default legal protections for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 code.

        Returns:
            List of applicable legal protection strings.
        """
        from greenlang.agents.eudr.indigenous_rights_checker.reference_data.fpic_legal_frameworks import (
            get_fpic_requirements,
        )
        framework = get_fpic_requirements(country_code)
        if not framework:
            return []
        return framework.get("legal_basis", [])

    # -------------------------------------------------------------------
    # Row mapping
    # -------------------------------------------------------------------

    def _row_to_community(self, row: Any) -> IndigenousCommunity:
        """Convert a database row to IndigenousCommunity model.

        Args:
            row: Database row tuple.

        Returns:
            IndigenousCommunity instance.
        """
        recognition = None
        if row[9]:
            try:
                recognition = CommunityRecognitionStatus(row[9])
            except ValueError:
                recognition = None

        territory_ids = row[8] if isinstance(row[8], list) else []
        legal_prots = row[10] if isinstance(row[10], list) else []
        rep_orgs = row[13] if isinstance(row[13], list) else []
        contacts = row[14] if isinstance(row[14], list) else []
        commodities = row[15] if isinstance(row[15], list) else []
        engagement = row[16] if isinstance(row[16], dict) else {}

        return IndigenousCommunity(
            community_id=str(row[0]),
            community_name=row[1],
            indigenous_name=row[2],
            people_name=row[3],
            language=row[4],
            estimated_population=row[5],
            country_code=row[6],
            region=row[7],
            territory_ids=territory_ids,
            legal_recognition_status=recognition,
            applicable_legal_protections=legal_prots,
            ilo_169_coverage=row[11] if row[11] is not None else False,
            fpic_legal_requirement=row[12] if row[12] is not None else False,
            representative_organizations=rep_orgs,
            contact_channels=contacts,
            commodity_relevance=commodities,
            engagement_history_summary=engagement,
            data_source=row[17],
            provenance_hash=row[18],
            created_at=row[19],
            updated_at=row[20],
        )

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    async def _persist_community(
        self, community: IndigenousCommunity
    ) -> None:
        """Persist community record to database."""
        if self._pool is None:
            return

        params = {
            "community_id": community.community_id,
            "community_name": community.community_name,
            "indigenous_name": community.indigenous_name,
            "people_name": community.people_name,
            "language": community.language,
            "estimated_population": community.estimated_population,
            "country_code": community.country_code,
            "region": community.region,
            "territory_ids": json.dumps(community.territory_ids),
            "legal_recognition_status": (
                community.legal_recognition_status.value
                if community.legal_recognition_status
                else None
            ),
            "applicable_legal_protections": json.dumps(
                community.applicable_legal_protections
            ),
            "ilo_169_coverage": community.ilo_169_coverage,
            "fpic_legal_requirement": community.fpic_legal_requirement,
            "representative_organizations": json.dumps(
                community.representative_organizations
            ),
            "contact_channels": json.dumps(community.contact_channels),
            "commodity_relevance": json.dumps(community.commodity_relevance),
            "engagement_history_summary": json.dumps(
                community.engagement_history_summary
            ),
            "data_source": community.data_source,
            "provenance_hash": community.provenance_hash,
        }

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_INSERT_COMMUNITY, params)
            await conn.commit()

    async def _persist_community_update(
        self, community: IndigenousCommunity
    ) -> None:
        """Persist updated community record to database."""
        if self._pool is None:
            return

        params = {
            "community_id": community.community_id,
            "community_name": community.community_name,
            "indigenous_name": community.indigenous_name,
            "people_name": community.people_name,
            "language": community.language,
            "estimated_population": community.estimated_population,
            "region": community.region,
            "territory_ids": json.dumps(community.territory_ids),
            "legal_recognition_status": (
                community.legal_recognition_status.value
                if community.legal_recognition_status
                else None
            ),
            "applicable_legal_protections": json.dumps(
                community.applicable_legal_protections
            ),
            "ilo_169_coverage": community.ilo_169_coverage,
            "fpic_legal_requirement": community.fpic_legal_requirement,
            "representative_organizations": json.dumps(
                community.representative_organizations
            ),
            "contact_channels": json.dumps(community.contact_channels),
            "commodity_relevance": json.dumps(community.commodity_relevance),
            "engagement_history_summary": json.dumps(
                community.engagement_history_summary
            ),
            "provenance_hash": community.provenance_hash,
        }

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_UPDATE_COMMUNITY, params)
            await conn.commit()
