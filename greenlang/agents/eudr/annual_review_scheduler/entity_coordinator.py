# -*- coding: utf-8 -*-
"""
Entity Coordinator Engine - AGENT-EUDR-034

Coordinates annual reviews across multiple organizational entities,
managing review dependencies, cascading review requirements down
the entity hierarchy, and aggregating completion status.

Zero-Hallucination:
    - All completion percentages are deterministic Decimal arithmetic
    - Dependency resolution uses topological traversal (no ML/LLM)
    - Cascade depth is bounded by configuration

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-034 (GL-EUDR-ARS-034)
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Set

from .config import AnnualReviewSchedulerConfig, get_config
from .models import (
    AGENT_ID,
    EntityCoordination,
    EntityCoordinationRecord,
    EntityDependency,
    EntityReviewInfo,
    EntityReviewStatus,
    EntityRole,
    EntityStatus,
    EntityType,
    ReviewPhase,
)
from .provenance import ProvenanceTracker
from . import metrics as m

logger = logging.getLogger(__name__)


class EntityCoordinator:
    """Multi-entity review coordination engine.

    Identifies entities requiring review, cascades requirements down
    the organizational hierarchy, tracks dependencies between entity
    reviews, and aggregates completion across all entities.

    Example:
        >>> coordinator = EntityCoordinator()
        >>> record = await coordinator.identify_review_entities(
        ...     operator_id="OP-001",
        ...     entities=[{"entity_id": "SUB-001", "entity_type": "subsidiary"}],
        ... )
        >>> assert record.total_entities > 0
    """

    def __init__(
        self,
        config: Optional[AnnualReviewSchedulerConfig] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize EntityCoordinator engine."""
        self.config = config or get_config()
        self._provenance = provenance or ProvenanceTracker()
        self._coordination_records: Dict[str, EntityCoordinationRecord] = {}
        self._entities: Dict[str, EntityCoordination] = {}
        self._dependencies: Dict[str, EntityDependency] = {}
        logger.info("EntityCoordinator engine initialized")

    async def identify_review_entities(
        self,
        operator_id: str,
        entities: List[Dict[str, Any]],
        cycle_id: str = "",
    ) -> EntityCoordinationRecord:
        """Identify all entities that require review.

        Args:
            operator_id: Root operator identifier.
            entities: List of entity data dictionaries.
            cycle_id: Associated review cycle ID.

        Returns:
            EntityCoordinationRecord with identified entities.
        """
        start_time = time.monotonic()
        now = datetime.now(timezone.utc).replace(microsecond=0)
        coordination_id = str(uuid.uuid4())

        entity_infos: List[EntityReviewInfo] = []

        # Process entity list (cap to parallel limit)
        capped = entities[: self.config.entity_parallel_review_limit]
        for entity_data in capped:
            entity_info = EntityReviewInfo(
                entity_id=entity_data.get("entity_id", str(uuid.uuid4())),
                entity_type=EntityType(
                    entity_data.get("entity_type", "operator")
                ),
                entity_name=entity_data.get("entity_name", ""),
                review_status=EntityReviewStatus.NOT_STARTED,
                completion_percent=Decimal("0"),
                assigned_reviewer=entity_data.get("assigned_reviewer"),
                parent_entity_id=entity_data.get("parent_entity_id"),
                dependencies=entity_data.get("dependencies", []),
            )
            entity_infos.append(entity_info)
            m.record_entity_coordinated(entity_info.entity_type.value)

        record = EntityCoordinationRecord(
            coordination_id=coordination_id,
            operator_id=operator_id,
            cycle_id=cycle_id,
            entities=entity_infos,
            total_entities=len(entity_infos),
            completed_entities=0,
            cascade_depth=0,
            overall_completion=Decimal("0"),
            blocked_entities=0,
            escalated_entities=0,
            coordinated_at=now,
        )

        # Provenance
        prov_data = {
            "coordination_id": coordination_id,
            "operator_id": operator_id,
            "total_entities": len(entity_infos),
            "coordinated_at": now.isoformat(),
        }
        record.provenance_hash = self._provenance.compute_hash(prov_data)
        self._provenance.record(
            "entity_coordination", "identify", coordination_id, AGENT_ID,
            metadata={"operator_id": operator_id, "entities": len(entity_infos)},
        )

        self._coordination_records[coordination_id] = record

        elapsed = time.monotonic() - start_time
        m.observe_entity_coordination_duration(elapsed)

        logger.info(
            "Identified %d entities for review (operator=%s, cycle=%s)",
            len(entity_infos), operator_id, cycle_id,
        )
        return record

    async def cascade_reviews(
        self,
        coordination_id: str,
        child_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> EntityCoordinationRecord:
        """Cascade review requirements to child entities.

        Propagates review requirements down the organizational hierarchy
        up to the configured maximum cascade depth.

        Args:
            coordination_id: Coordination record identifier.
            child_entities: Child entities to cascade to.

        Returns:
            Updated EntityCoordinationRecord.

        Raises:
            ValueError: If coordination record not found.
        """
        start_time = time.monotonic()
        record = self._get_record(coordination_id)
        now = datetime.now(timezone.utc).replace(microsecond=0)

        current_depth = record.cascade_depth
        if current_depth >= self.config.entity_cascade_max_depth:
            logger.warning(
                "Cascade depth limit reached (%d) for coordination %s",
                current_depth, coordination_id,
            )
            return record

        if child_entities:
            for child_data in child_entities:
                child_info = EntityReviewInfo(
                    entity_id=child_data.get("entity_id", str(uuid.uuid4())),
                    entity_type=EntityType(
                        child_data.get("entity_type", "supplier")
                    ),
                    entity_name=child_data.get("entity_name", ""),
                    review_status=EntityReviewStatus.NOT_STARTED,
                    completion_percent=Decimal("0"),
                    parent_entity_id=child_data.get("parent_entity_id"),
                    dependencies=child_data.get("dependencies", []),
                )
                record.entities.append(child_info)
                m.record_entity_coordinated(child_info.entity_type.value)

        record.cascade_depth = current_depth + 1
        record.total_entities = len(record.entities)
        record.coordinated_at = now
        m.record_entity_cascade()

        elapsed = time.monotonic() - start_time
        m.observe_cascade_resolution_duration(elapsed)

        logger.info(
            "Cascaded to depth %d for coordination %s (%d total entities)",
            record.cascade_depth, coordination_id, record.total_entities,
        )
        return record

    async def track_dependencies(
        self,
        coordination_id: str,
    ) -> EntityCoordinationRecord:
        """Evaluate dependency status and identify blocked entities.

        Performs a breadth-first traversal of entity dependencies
        to identify entities that are blocked or ready for review.

        Args:
            coordination_id: Coordination record identifier.

        Returns:
            Updated EntityCoordinationRecord with dependency status.
        """
        record = self._get_record(coordination_id)

        # Build entity lookup
        entity_map: Dict[str, EntityReviewInfo] = {
            e.entity_id: e for e in record.entities
        }

        completed_ids: Set[str] = {
            e.entity_id for e in record.entities
            if e.review_status == EntityReviewStatus.APPROVED
        }

        blocked_count = 0
        for entity in record.entities:
            if entity.review_status in (
                EntityReviewStatus.APPROVED,
                EntityReviewStatus.REJECTED,
            ):
                continue

            # Check if all dependencies are met
            deps = entity.dependencies
            if deps:
                unmet = [d for d in deps if d not in completed_ids]
                if unmet:
                    blocked_count += 1
                    if entity.review_status != EntityReviewStatus.AWAITING_INPUT:
                        entity.review_status = EntityReviewStatus.AWAITING_INPUT

        record.blocked_entities = blocked_count

        logger.info(
            "Dependency check for coordination %s: %d blocked of %d",
            coordination_id, blocked_count, record.total_entities,
        )
        return record

    async def aggregate_completion(
        self,
        coordination_id: str,
    ) -> EntityCoordinationRecord:
        """Aggregate completion status across all entities.

        Args:
            coordination_id: Coordination record identifier.

        Returns:
            Updated EntityCoordinationRecord with aggregated stats.
        """
        record = self._get_record(coordination_id)

        completed = sum(
            1 for e in record.entities
            if e.review_status == EntityReviewStatus.APPROVED
        )
        escalated = sum(
            1 for e in record.entities
            if e.review_status == EntityReviewStatus.ESCALATED
        )

        record.completed_entities = completed
        record.escalated_entities = escalated

        if record.total_entities > 0:
            # Weighted completion: individual entity percentages
            total_pct = sum(e.completion_percent for e in record.entities)
            record.overall_completion = (
                total_pct / Decimal(str(record.total_entities))
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        m.set_entity_completion(float(record.overall_completion))

        logger.info(
            "Coordination %s: %d/%d completed (%.1f%% overall)",
            coordination_id, completed, record.total_entities,
            record.overall_completion,
        )
        return record

    async def update_entity_status(
        self,
        coordination_id: str,
        entity_id: str,
        new_status: EntityReviewStatus,
        completion_percent: Optional[Decimal] = None,
    ) -> EntityCoordinationRecord:
        """Update the review status of a specific entity.

        Args:
            coordination_id: Coordination record identifier.
            entity_id: Entity identifier.
            new_status: New review status.
            completion_percent: Optional completion percentage.

        Returns:
            Updated EntityCoordinationRecord.

        Raises:
            ValueError: If entity not found.
        """
        record = self._get_record(coordination_id)

        entity_found = False
        for entity in record.entities:
            if entity.entity_id == entity_id:
                entity.review_status = new_status
                if completion_percent is not None:
                    entity.completion_percent = completion_percent
                if new_status == EntityReviewStatus.APPROVED:
                    entity.completion_percent = Decimal("100")
                entity_found = True
                break

        if not entity_found:
            raise ValueError(
                f"Entity {entity_id} not found in coordination {coordination_id}"
            )

        # Recalculate aggregation
        return await self.aggregate_completion(coordination_id)

    async def get_record(
        self, coordination_id: str,
    ) -> Optional[EntityCoordinationRecord]:
        """Get a specific coordination record by ID."""
        return self._coordination_records.get(coordination_id)

    async def list_records(
        self,
        operator_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[EntityCoordinationRecord]:
        """List coordination records with optional filters."""
        results = list(self._coordination_records.values())
        if operator_id:
            results = [r for r in results if r.operator_id == operator_id]
        results.sort(key=lambda r: r.coordinated_at, reverse=True)
        return results[offset: offset + limit]

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        return {
            "engine": "EntityCoordinator",
            "status": "healthy",
            "total_coordinations": len(self._coordination_records),
            "max_cascade_depth": self.config.entity_cascade_max_depth,
        }

    # -- Engine-test API (EntityCoordination-based) --

    async def assign_entity(
        self,
        cycle_id: str,
        name: str,
        role: EntityRole,
        email: str,
        phases: Optional[List[ReviewPhase]] = None,
    ) -> EntityCoordination:
        """Assign an entity to a review cycle."""
        entity_id = f"entity-{uuid.uuid4()}"
        # External auditors start as INVITED
        status = EntityStatus.INVITED if role == EntityRole.EXTERNAL_AUDITOR else EntityStatus.ACTIVE
        entity = EntityCoordination(
            entity_id=entity_id,
            cycle_id=cycle_id,
            name=name,
            role=role,
            email=email,
            status=status,
            assigned_phases=phases or [],
            dependencies=[],
        )
        self._entities[entity_id] = entity
        m.record_entity_coordinated("engine")
        return entity

    async def get_entity(self, entity_id: str) -> EntityCoordination:
        """Get an entity by ID."""
        entity = self._entities.get(entity_id)
        if entity is None:
            raise ValueError(f"Entity {entity_id} not found")
        return entity

    async def list_entities(
        self,
        cycle_id: Optional[str] = None,
        role: Optional[EntityRole] = None,
        status: Optional[EntityStatus] = None,
        phase: Optional[ReviewPhase] = None,
    ) -> List[EntityCoordination]:
        """List entities with optional filters."""
        results = list(self._entities.values())
        if cycle_id:
            results = [e for e in results if e.cycle_id == cycle_id]
        if role is not None:
            results = [e for e in results if e.role == role]
        if status is not None:
            results = [e for e in results if e.status == status]
        if phase is not None:
            results = [e for e in results if phase in e.assigned_phases]
        return results

    async def update_entity(
        self,
        entity_id: str,
        name: Optional[str] = None,
        email: Optional[str] = None,
        phases: Optional[List[ReviewPhase]] = None,
    ) -> EntityCoordination:
        """Update an entity."""
        entity = await self.get_entity(entity_id)
        if entity.status == EntityStatus.REMOVED:
            raise ValueError(f"Cannot update removed entity {entity_id}")
        if name is not None:
            entity.name = name
        if email is not None:
            entity.email = email
        if phases is not None:
            entity.assigned_phases = phases
        return entity

    async def deactivate_entity(self, entity_id: str) -> EntityCoordination:
        """Deactivate an entity."""
        entity = await self.get_entity(entity_id)
        entity.status = EntityStatus.INACTIVE
        return entity

    async def remove_entity(self, entity_id: str) -> EntityCoordination:
        """Remove an entity."""
        entity = await self.get_entity(entity_id)
        entity.status = EntityStatus.REMOVED
        return entity

    async def accept_invitation(self, entity_id: str) -> EntityCoordination:
        """Accept an invitation."""
        entity = await self.get_entity(entity_id)
        entity.status = EntityStatus.ACTIVE
        return entity

    async def decline_invitation(self, entity_id: str) -> EntityCoordination:
        """Decline an invitation."""
        entity = await self.get_entity(entity_id)
        entity.status = EntityStatus.DECLINED
        return entity

    async def create_dependency(
        self,
        source_entity_id: str,
        target_entity_id: str,
        dependency_type: str,
        phase: ReviewPhase,
        description: str = "",
    ) -> EntityDependency:
        """Create a dependency between two entities."""
        # Check for circular dependency
        for dep in self._dependencies.values():
            if dep.source_entity_id == target_entity_id and dep.target_entity_id == source_entity_id:
                raise ValueError(f"Creating this dependency would be circular")
        dep_id = f"dep-{uuid.uuid4()}"
        dep = EntityDependency(
            dependency_id=dep_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            dependency_type=dependency_type,
            phase=phase,
            description=description,
            resolved=False,
        )
        self._dependencies[dep_id] = dep
        return dep

    async def list_dependencies(
        self,
        cycle_id: Optional[str] = None,
        resolved: Optional[bool] = None,
    ) -> List[EntityDependency]:
        """List dependencies."""
        results = list(self._dependencies.values())
        if resolved is not None:
            results = [d for d in results if d.resolved == resolved]
        # Filter by cycle if entities in cycle
        if cycle_id:
            cycle_entity_ids = {e.entity_id for e in self._entities.values() if e.cycle_id == cycle_id}
            results = [d for d in results if d.source_entity_id in cycle_entity_ids or d.target_entity_id in cycle_entity_ids]
        return results

    async def resolve_dependency(self, dependency_id: str) -> EntityDependency:
        """Resolve a dependency."""
        dep = self._dependencies.get(dependency_id)
        if dep is None:
            raise ValueError(f"Dependency {dependency_id} not found")
        dep.resolved = True
        return dep

    async def generate_raci_matrix(self, cycle_id: str) -> Dict[str, Any]:
        """Generate a RACI matrix for a cycle."""
        entities = await self.list_entities(cycle_id=cycle_id)
        if not entities:
            return {}
        matrix: Dict[str, Any] = {}
        for phase in ReviewPhase:
            phase_entities = [e for e in entities if phase in e.assigned_phases]
            if phase_entities:
                matrix[phase.value] = {
                    e.name: e.role.value for e in phase_entities
                }
        return matrix

    async def get_workload_summary(self, cycle_id: str) -> Dict[str, Any]:
        """Get workload summary for a cycle."""
        entities = await self.list_entities(cycle_id=cycle_id)
        summary: Dict[str, Any] = {}
        for entity in entities:
            summary[entity.entity_id] = {
                "name": entity.name,
                "role": entity.role.value,
                "phase_count": len(entity.assigned_phases),
                "status": entity.status.value,
            }
        return summary

    async def bulk_assign(
        self,
        cycle_id: str,
        entities_data: List[Dict[str, Any]],
    ) -> List[EntityCoordination]:
        """Bulk assign entities to a cycle."""
        results: List[EntityCoordination] = []
        for data in entities_data:
            entity = await self.assign_entity(
                cycle_id=cycle_id,
                name=data.get("name", ""),
                role=data.get("role", EntityRole.CONTRIBUTOR),
                email=data.get("email", ""),
                phases=data.get("phases", []),
            )
            results.append(entity)
        return results

    async def count_by_role(self, cycle_id: str) -> Dict[str, int]:
        """Count entities by role in a cycle."""
        entities = await self.list_entities(cycle_id=cycle_id)
        counts: Dict[str, int] = {}
        for entity in entities:
            role_val = entity.role.value
            counts[role_val] = counts.get(role_val, 0) + 1
        return counts

    # -- Private helpers --

    def _get_record(self, coordination_id: str) -> EntityCoordinationRecord:
        """Retrieve a coordination record or raise ValueError."""
        record = self._coordination_records.get(coordination_id)
        if record is None:
            raise ValueError(f"Coordination record {coordination_id} not found")
        return record


# Alias for backward compatibility with tests
EntityCoordinatorEngine = EntityCoordinator
