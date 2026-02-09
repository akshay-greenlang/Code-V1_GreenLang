# -*- coding: utf-8 -*-
"""
Data Gateway Service Facade - AGENT-DATA-004: API Gateway Agent (GL-DATA-GW-001)

Provides the main service class and FastAPI integration functions:
- DataGatewayService: Composes all 7 engines into a single facade
- configure_data_gateway(app): Register service on FastAPI app
- get_data_gateway(app): Retrieve service from app state
- get_router(): Return FastAPI router for mounting

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 API Gateway Agent
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DataGatewayService:
    """Facade composing all Data Gateway engines.

    Provides a single entry point for all data gateway operations,
    delegating to the appropriate engine for each operation type.

    Attributes:
        config: DataGatewayConfig instance.
        provenance: ProvenanceTracker instance.
        query_parser: QueryParserEngine instance.
        query_router: QueryRouterEngine instance.
        connection_manager: ConnectionManagerEngine instance.
        response_aggregator: ResponseAggregatorEngine instance.
        schema_translator: SchemaTranslatorEngine instance.
        cache_manager: CacheManagerEngine instance.
        data_catalog: DataCatalogEngine instance.
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialize the Data Gateway Service with all engines.

        Args:
            config: DataGatewayConfig instance. If None, loads from env.
        """
        if config is None:
            from greenlang.data_gateway.config import get_config
            config = get_config()

        self.config = config

        # Initialize provenance
        from greenlang.data_gateway.provenance import ProvenanceTracker
        self.provenance = ProvenanceTracker()

        # Initialize engines
        from greenlang.data_gateway.query_parser import QueryParserEngine
        from greenlang.data_gateway.query_router import QueryRouterEngine
        from greenlang.data_gateway.connection_manager import (
            ConnectionManagerEngine,
        )
        from greenlang.data_gateway.response_aggregator import (
            ResponseAggregatorEngine,
        )
        from greenlang.data_gateway.schema_translator import (
            SchemaTranslatorEngine,
        )
        from greenlang.data_gateway.cache_manager import CacheManagerEngine
        from greenlang.data_gateway.data_catalog import DataCatalogEngine

        self.connection_manager = ConnectionManagerEngine(
            config=config,
            provenance=self.provenance,
        )
        self.query_parser = QueryParserEngine(
            config=config,
            provenance=self.provenance,
        )
        self.query_router = QueryRouterEngine(
            config=config,
            provenance=self.provenance,
            connection_manager=self.connection_manager,
        )
        self.response_aggregator = ResponseAggregatorEngine(
            config=config,
            provenance=self.provenance,
        )
        self.schema_translator = SchemaTranslatorEngine(
            config=config,
            provenance=self.provenance,
        )
        self.cache_manager = CacheManagerEngine(
            config=config,
            provenance=self.provenance,
        )
        self.data_catalog = DataCatalogEngine(
            config=config,
            provenance=self.provenance,
        )

        # In-memory query templates
        self._templates: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "DataGatewayService initialized with all 7 engines + provenance"
        )

    # =========================================================================
    # Query Execution Delegation
    # =========================================================================

    def execute_query(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a unified data query.

        Parses the request, checks cache, routes to sources, and returns
        the result.

        Args:
            request: Query request dictionary.

        Returns:
            QueryResult dictionary.
        """
        start_time = time.monotonic()

        try:
            # Update active queries gauge
            try:
                from greenlang.data_gateway.metrics import update_active_queries
                update_active_queries(1)
            except ImportError:
                pass

            # Check cache first
            query_hash = self.cache_manager.compute_query_hash(request)
            cached = self.cache_manager.get(query_hash)
            if cached is not None:
                logger.info("Query served from cache: %s", query_hash[:16])
                return cached

            # Parse query
            plan = self.query_parser.parse(request)

            # Validate
            errors = self.query_parser.validate_query(plan)
            if errors:
                return {
                    "query_id": plan.get("query_id", ""),
                    "source_id": "",
                    "data": [],
                    "total_count": 0,
                    "row_count": 0,
                    "errors": errors,
                    "execution_time_ms": 0.0,
                    "created_at": plan.get("created_at", ""),
                }

            # Execute
            result = self.query_router.execute(plan)

            # Apply aggregations if any
            aggregations = plan.get("aggregations", [])
            if aggregations and result.get("data"):
                agg_results = self.response_aggregator.apply_aggregations(
                    result["data"], aggregations,
                )
                result["aggregations"] = agg_results

            # Cache result
            source_ids = plan.get("sources", [])
            self.cache_manager.put(
                query_hash,
                result,
                source_id=",".join(source_ids) if source_ids else "none",
            )

            return result

        finally:
            try:
                from greenlang.data_gateway.metrics import update_active_queries
                update_active_queries(-1)
            except ImportError:
                pass

    def execute_batch(
        self,
        requests: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute a batch of queries.

        Args:
            requests: List of query request dictionaries.

        Returns:
            List of QueryResult dictionaries.
        """
        results: List[Dict[str, Any]] = []
        for request in requests:
            try:
                result = self.execute_query(request)
                results.append(result)
            except Exception as e:
                logger.error("Batch query error: %s", e)
                results.append({
                    "query_id": "",
                    "source_id": "",
                    "data": [],
                    "total_count": 0,
                    "row_count": 0,
                    "errors": [str(e)],
                    "execution_time_ms": 0.0,
                    "created_at": "",
                })
        return results

    # =========================================================================
    # Source Registration Delegation
    # =========================================================================

    def register_source(self, request: Dict[str, Any]) -> str:
        """Register a data source. Delegates to ConnectionManagerEngine.

        Args:
            request: Source registration data.

        Returns:
            Generated source_id.
        """
        return self.connection_manager.register_source(request)

    def test_source(self, source_id: str) -> Dict[str, Any]:
        """Test source connectivity. Delegates to ConnectionManagerEngine.

        Args:
            source_id: Source identifier.

        Returns:
            Test result dictionary.
        """
        return self.connection_manager.test_connection(source_id)

    # =========================================================================
    # Schema Translation Delegation
    # =========================================================================

    def translate_schema(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Translate data between schemas. Delegates to SchemaTranslatorEngine.

        Args:
            request: Translation request with data, source_type, target_type.

        Returns:
            Translated data dictionary.
        """
        data = request.get("data", {})
        source_type = request.get("source_type", "")
        target_type = request.get("target_type", "")
        return self.schema_translator.translate(
            data, source_type, target_type,
        )

    # =========================================================================
    # Cache Delegation
    # =========================================================================

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics. Delegates to CacheManagerEngine.

        Returns:
            Cache statistics dictionary.
        """
        return self.cache_manager.get_stats()

    def invalidate_cache(self, request: Dict[str, Any]) -> int:
        """Invalidate cache entries. Delegates to CacheManagerEngine.

        Args:
            request: Invalidation request with optional source_id,
                     query_hash, or invalidate_all.

        Returns:
            Number of entries invalidated.
        """
        return self.cache_manager.invalidate(
            source_id=request.get("source_id"),
            query_hash=request.get("query_hash"),
            invalidate_all=request.get("invalidate_all", False),
        )

    # =========================================================================
    # Catalog Delegation
    # =========================================================================

    def search_catalog(self, query: str) -> List[Dict[str, Any]]:
        """Search data catalog. Delegates to DataCatalogEngine.

        Args:
            query: Search query string.

        Returns:
            List of matching catalog entries.
        """
        return self.data_catalog.search(query)

    # =========================================================================
    # Template Management
    # =========================================================================

    def create_template(self, template: Dict[str, Any]) -> str:
        """Create a query template.

        Args:
            template: Template definition with name and query structure.

        Returns:
            Generated template_id.
        """
        import uuid
        template_id = f"TPL-{uuid.uuid4().hex[:12]}"
        template["template_id"] = template_id

        from datetime import datetime, timezone
        template["created_at"] = (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        )

        self._templates[template_id] = template

        # Record provenance
        from greenlang.data_gateway.provenance import ProvenanceTracker
        import hashlib
        import json
        data_hash = hashlib.sha256(
            json.dumps(template, sort_keys=True, default=str).encode()
        ).hexdigest()
        self.provenance.record(
            entity_type="template",
            entity_id=template_id,
            action="template_creation",
            data_hash=data_hash,
        )

        logger.info(
            "Created template %s: name=%s",
            template_id, template.get("name", ""),
        )
        return template_id

    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get a query template by ID.

        Args:
            template_id: Template identifier.

        Returns:
            Template dictionary or None.
        """
        return self._templates.get(template_id)

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all query templates.

        Returns:
            List of template dictionaries.
        """
        return list(self._templates.values())

    def execute_template(
        self,
        template_id: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a query from a template.

        Args:
            template_id: Template identifier.
            parameters: Optional parameter overrides.

        Returns:
            QueryResult dictionary.

        Raises:
            ValueError: If template not found.
        """
        template = self._templates.get(template_id)
        if template is None:
            raise ValueError(f"Template not found: {template_id}")

        # Build request from template
        request = dict(template.get("query", {}))

        # Apply parameter overrides
        if parameters:
            for key, value in parameters.items():
                request[key] = value

        return self.execute_query(request)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics.

        Returns:
            Dictionary with statistics from all engines.
        """
        return {
            "agent_id": "GL-DATA-GW-001",
            "agent_name": "API Gateway Agent",
            "version": "1.0.0",
            "query_parser": self.query_parser.get_statistics(),
            "query_router": self.query_router.get_statistics(),
            "connection_manager": self.connection_manager.get_statistics(),
            "response_aggregator": self.response_aggregator.get_statistics(),
            "schema_translator": self.schema_translator.get_statistics(),
            "cache_manager": self.cache_manager.get_statistics(),
            "data_catalog": self.data_catalog.get_statistics(),
            "provenance": {
                "total_entries": self.provenance.entry_count,
                "total_entities": self.provenance.entity_count,
            },
            "templates": {
                "total_templates": len(self._templates),
            },
        }


# =============================================================================
# FastAPI Integration
# =============================================================================

_SERVICE_KEY = "data_gateway_service"


def configure_data_gateway(app: Any) -> DataGatewayService:
    """Register the Data Gateway Service on a FastAPI application.

    Creates the service, attaches it to app.state, and includes the
    API router.

    Args:
        app: FastAPI application instance.

    Returns:
        Configured DataGatewayService instance.
    """
    service = DataGatewayService()
    app.state.data_gateway_service = service

    # Include router
    from greenlang.data_gateway.api.router import router
    app.include_router(router)

    logger.info("Data Gateway Service configured on FastAPI app")
    return service


def get_data_gateway(app: Any) -> DataGatewayService:
    """Retrieve the Data Gateway Service from a FastAPI application.

    Args:
        app: FastAPI application instance.

    Returns:
        DataGatewayService instance.

    Raises:
        RuntimeError: If service not configured.
    """
    service = getattr(app.state, _SERVICE_KEY, None)
    if service is None:
        raise RuntimeError(
            "Data Gateway Service not configured. "
            "Call configure_data_gateway(app) first."
        )
    return service


def get_router():
    """Return the FastAPI router for the Data Gateway Service.

    Returns:
        FastAPI APIRouter instance.
    """
    from greenlang.data_gateway.api.router import router
    return router
