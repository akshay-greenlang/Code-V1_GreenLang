# -*- coding: utf-8 -*-
"""
DataLoaders for GreenLang Factors GraphQL (GAP-12).

Prevents N+1 queries when a single GraphQL response references many
factors / sources / editions / method packs.  Each loader batches the
underlying repository / registry lookup into a single call per request.

Wire-up:
    * The factors loaders are attached to the GraphQLContext by
      :func:`attach_factor_loaders` during context construction.
    * Each loader instance is per-request (short-lived) so caches are
      flushed automatically between requests.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DataLoader base class.
#
# The existing ``greenlang.integration.api.graphql.dataloaders`` module
# transitively imports ``greenlang.api.graphql.*`` (legacy path that no
# longer exists).  To keep the Factors loaders usable standalone — and
# to avoid the fragile alias — we inline a minimal batching DataLoader
# here.  The public contract (``load``, ``load_many``, ``clear``,
# ``prime``) is identical to the DataLoader spec used across the
# codebase.
# ---------------------------------------------------------------------------


class DataLoader:
    """Minimal async DataLoader with batching + per-key caching."""

    def __init__(self, batch_load_fn, max_batch_size: int = 100):
        self.batch_load_fn = batch_load_fn
        self.max_batch_size = max_batch_size
        self.cache: Dict[str, Any] = {}
        self.queue: List[Any] = []
        self.dispatched = False

    async def load(self, key: str) -> Optional[Any]:
        if key in self.cache:
            return self.cache[key]
        future: asyncio.Future = asyncio.Future()
        self.queue.append((key, future))
        if not self.dispatched:
            self.dispatched = True
            asyncio.create_task(self._dispatch_queue())
        return await future

    async def load_many(self, keys: List[str]) -> List[Optional[Any]]:
        tasks = [self.load(k) for k in keys]
        return await asyncio.gather(*tasks)

    async def _dispatch_queue(self) -> None:
        await asyncio.sleep(0.001)
        queue = self.queue
        self.queue = []
        self.dispatched = False
        if not queue:
            return

        keys = [item[0] for item in queue]
        futures = [item[1] for item in queue]

        try:
            values = await self.batch_load_fn(keys)
            if len(values) != len(keys):
                raise ValueError(
                    f"Batch load returned {len(values)} items for {len(keys)} keys"
                )
            for key, value, future in zip(keys, values, futures):
                self.cache[key] = value
                future.set_result(value)
        except Exception as exc:
            logger.error("Batch load error: %s", exc)
            for future in futures:
                if not future.done():
                    future.set_exception(exc)

    def clear(self) -> None:
        self.cache.clear()

    def prime(self, key: str, value: Any) -> None:
        self.cache[key] = value


# ==============================================================================
# Helpers
# ==============================================================================


def _factor_to_graphql(record: Any, edition_id: Optional[str] = None) -> Any:
    """Convert an EmissionFactorRecord (or a SDK Factor dict) to the
    GraphQL :class:`Factor` type.

    This is a late import to avoid a circular dependency with the types
    module at module load time.
    """
    from greenlang.integration.api.graphql.types_factors import (
        Factor,
        QualityScore,
        Source,
    )

    # Allow either a record object from the catalog repo, a dict, or an
    # already-built Strawberry Factor.
    if isinstance(record, Factor):
        return record

    if isinstance(record, dict):
        data = record
    else:
        # EmissionFactorRecord.to_dict() is the canonical path.
        if hasattr(record, "to_dict"):
            data = record.to_dict()
        else:
            data = {}

    # Pull nested structures carefully — the record's internal shape is
    # different from the flat SDK shape we expose via GraphQL.
    dqs = data.get("dqs") or {}
    provenance = data.get("provenance") or {}
    gwp = data.get("gwp_100yr") or {}
    license_info = data.get("license_info") or {}

    quality = None
    if dqs:
        quality = QualityScore(
            overall_score=float(dqs.get("overall_score") or 0.0),
            rating=str(dqs.get("rating")) if dqs.get("rating") is not None else None,
            temporal=dqs.get("temporal"),
            geographical=dqs.get("geographical"),
            technological=dqs.get("technological"),
            representativeness=dqs.get("representativeness"),
            methodological=dqs.get("methodological"),
        )

    source = None
    if provenance:
        source = Source(
            source_id=str(
                data.get("source_id")
                or provenance.get("source_org")
                or "unknown"
            ),
            organization=provenance.get("source_org"),
            publication=provenance.get("source_publication"),
            year=provenance.get("source_year"),
            url=provenance.get("source_url"),
            methodology=(
                provenance.get("methodology").value
                if hasattr(provenance.get("methodology"), "value")
                else provenance.get("methodology")
            ),
            version=provenance.get("version"),
        )

    scope_val = data.get("scope")
    if hasattr(scope_val, "value"):
        scope_val = scope_val.value
    boundary_val = data.get("boundary")
    if hasattr(boundary_val, "value"):
        boundary_val = boundary_val.value

    return Factor(
        factor_id=str(data.get("factor_id", "")),
        fuel_type=data.get("fuel_type"),
        unit=data.get("unit"),
        geography=data.get("geography"),
        geography_level=data.get("geography_level"),
        scope=str(scope_val) if scope_val is not None else None,
        boundary=str(boundary_val) if boundary_val is not None else None,
        co2_per_unit=gwp.get("CO2") if gwp else None,
        ch4_per_unit=gwp.get("CH4") if gwp else None,
        n2o_per_unit=gwp.get("N2O") if gwp else None,
        co2e_per_unit=(
            gwp.get("co2e_total")
            if gwp
            else data.get("co2e_per_unit")
        ),
        data_quality=quality,
        source=source,
        uncertainty_95ci=data.get("uncertainty_95ci"),
        valid_from=str(data.get("valid_from")) if data.get("valid_from") else None,
        valid_to=str(data.get("valid_to")) if data.get("valid_to") else None,
        factor_status=data.get("factor_status") or "certified",
        license=license_info.get("license") if license_info else data.get("license"),
        license_class=data.get("license_class"),
        compliance_frameworks=list(data.get("compliance_frameworks") or []),
        tags=list(data.get("tags") or []),
        activity_tags=list(data.get("activity_tags") or []),
        sector_tags=list(data.get("sector_tags") or []),
        notes=data.get("notes"),
        edition_id=edition_id or data.get("edition_id"),
        source_id=data.get("source_id"),
        source_release=data.get("source_release"),
        release_version=data.get("release_version"),
        replacement_factor_id=data.get("replacement_factor_id"),
        content_hash=data.get("content_hash"),
    )


# ==============================================================================
# FactorByIdLoader
# ==============================================================================


class FactorByIdLoader(DataLoader):
    """Batch-load Factor objects by (optionally edition-scoped) factor_id."""

    def __init__(self, service: Any, edition_id: Optional[str] = None):
        self.service = service
        self.edition_id = edition_id
        super().__init__(self._batch_load_factors)

    def _resolve_edition(self) -> str:
        if self.edition_id:
            return self.edition_id
        # Fall back to the service's default edition.
        return self.service.repo.get_default_edition_id()

    async def _batch_load_factors(self, factor_ids: List[str]) -> List[Optional[Any]]:
        """Load many factors in one go.  Runs synchronously inside the
        service because the repo APIs are sync — we wrap them in
        ``asyncio.to_thread`` so the event loop stays responsive.
        """
        logger.debug("Batch loading %d factors", len(factor_ids))

        def _load() -> List[Optional[Any]]:
            edition = self._resolve_edition()
            out: List[Optional[Any]] = []
            for fid in factor_ids:
                rec = self.service.repo.get_factor(edition, fid)
                if rec is None:
                    out.append(None)
                else:
                    out.append(_factor_to_graphql(rec, edition_id=edition))
            return out

        return await asyncio.to_thread(_load)


# ==============================================================================
# SourceByIdLoader
# ==============================================================================


class SourceByIdLoader(DataLoader):
    """Batch-load Source objects from the CTO source registry."""

    def __init__(self, source_registry: Optional[Dict[str, Any]] = None):
        self._registry = source_registry
        super().__init__(self._batch_load_sources)

    def _ensure_registry(self) -> Dict[str, Any]:
        if self._registry is not None:
            return self._registry
        try:
            from greenlang.factors.source_registry import registry_by_id

            self._registry = registry_by_id()
        except Exception as exc:
            logger.warning("Failed to load source registry: %s", exc)
            self._registry = {}
        return self._registry

    async def _batch_load_sources(self, source_ids: List[str]) -> List[Optional[Any]]:
        from greenlang.integration.api.graphql.types_factors import Source

        def _load() -> List[Optional[Any]]:
            reg = self._ensure_registry()
            out: List[Optional[Any]] = []
            for sid in source_ids:
                entry = reg.get(sid)
                if entry is None:
                    out.append(None)
                    continue
                out.append(
                    Source(
                        source_id=entry.source_id,
                        organization=getattr(entry, "publisher", None)
                        or entry.display_name,
                        publication=entry.display_name,
                        year=None,
                        url=entry.watch_url,
                        methodology=None,
                        license=entry.license_class,
                        version=getattr(entry, "dataset_version", None),
                    )
                )
            return out

        return await asyncio.to_thread(_load)


# ==============================================================================
# EditionByIdLoader
# ==============================================================================


class EditionByIdLoader(DataLoader):
    """Batch-load Edition objects from the catalog."""

    def __init__(self, service: Any):
        self.service = service
        super().__init__(self._batch_load_editions)

    async def _batch_load_editions(self, edition_ids: List[str]) -> List[Optional[Any]]:
        from greenlang.integration.api.graphql.types_factors import Edition

        def _load() -> List[Optional[Any]]:
            all_editions = self.service.repo.list_editions(include_pending=True)
            by_id = {row.edition_id: row for row in all_editions}
            out: List[Optional[Any]] = []
            for eid in edition_ids:
                row = by_id.get(eid)
                if row is None:
                    out.append(None)
                else:
                    out.append(
                        Edition(
                            edition_id=row.edition_id,
                            status=row.status,
                            label=row.label,
                            manifest_hash=row.manifest_hash,
                            published_at=None,
                        )
                    )
            return out

        return await asyncio.to_thread(_load)


# ==============================================================================
# MethodPackByIdLoader
# ==============================================================================


class MethodPackByIdLoader(DataLoader):
    """Batch-load MethodPack objects from the method-pack registry."""

    def __init__(self):
        super().__init__(self._batch_load_method_packs)

    async def _batch_load_method_packs(
        self, pack_ids: List[str]
    ) -> List[Optional[Any]]:
        from greenlang.integration.api.graphql.types_factors import MethodPack

        def _load() -> List[Optional[Any]]:
            try:
                from greenlang.factors.method_packs import get_pack
            except ImportError:
                return [None for _ in pack_ids]

            out: List[Optional[Any]] = []
            for pid in pack_ids:
                try:
                    pack = get_pack(pid)
                except Exception:
                    out.append(None)
                    continue

                out.append(
                    MethodPack(
                        method_pack_id=pid,
                        name=getattr(pack, "name", None) or pid,
                        version=getattr(pack, "version", None),
                        scope=str(getattr(pack, "scope", None) or ""),
                        description=getattr(pack, "description", None),
                        jurisdictions=list(getattr(pack, "jurisdictions", []) or []),
                    )
                )
            return out

        return await asyncio.to_thread(_load)


# ==============================================================================
# Factory
# ==============================================================================


@dataclass
class FactorDataLoaderFactory:
    """Build all factor-related loaders for a single request."""

    service: Any

    def create_factor_loader(
        self, edition_id: Optional[str] = None
    ) -> FactorByIdLoader:
        return FactorByIdLoader(self.service, edition_id=edition_id)

    def create_source_loader(self) -> SourceByIdLoader:
        return SourceByIdLoader()

    def create_edition_loader(self) -> EditionByIdLoader:
        return EditionByIdLoader(self.service)

    def create_method_pack_loader(self) -> MethodPackByIdLoader:
        return MethodPackByIdLoader()

    def create_all_loaders(self) -> Dict[str, DataLoader]:
        return {
            "factor": self.create_factor_loader(),
            "source": self.create_source_loader(),
            "edition": self.create_edition_loader(),
            "method_pack": self.create_method_pack_loader(),
        }


def attach_factor_loaders(context: Any, service: Any) -> None:
    """Attach factor loaders onto a GraphQLContext.

    Used by the factors resolvers when the context wasn't constructed
    with loaders pre-populated (tests, alternative context classes).
    The loaders are stashed on ``context.metadata`` so we never stomp
    on attributes the base GraphQLContext owns.
    """
    if context is None:
        return
    factory = FactorDataLoaderFactory(service=service)
    loaders = factory.create_all_loaders()

    # Prefer direct attributes for ergonomics; fall back to metadata.
    try:
        context.factor_loader = loaders["factor"]
        context.source_loader = loaders["source"]
        context.edition_loader = loaders["edition"]
        context.method_pack_loader = loaders["method_pack"]
    except (AttributeError, TypeError):
        # Frozen / restricted context — stash on metadata.
        if not hasattr(context, "metadata") or context.metadata is None:
            context.metadata = {}
        context.metadata["factor_loaders"] = loaders


__all__ = [
    "FactorByIdLoader",
    "SourceByIdLoader",
    "EditionByIdLoader",
    "MethodPackByIdLoader",
    "FactorDataLoaderFactory",
    "attach_factor_loaders",
    "_factor_to_graphql",
]
