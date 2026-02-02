# -*- coding: utf-8 -*-
# GL-013 GraphQL API Service

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class WorkOrderMode(str, Enum):
    DRAFT = "draft"
    AUTO = "auto"
    HUMAN_IN_LOOP = "human_in_loop"

class AssetStatus(str, Enum):
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"

class GraphQLConfig(BaseModel):
    service_name: str = Field(default="gl-013-graphql")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080)
    enable_playground: bool = Field(default=True)
    jwt_secret: Optional[str] = None
    enable_audit_logging: bool = Field(default=True)

class AssetType(BaseModel):
    id: str
    name: str
    asset_type: str
    location: str
    status: AssetStatus = AssetStatus.OPERATIONAL
    tags: List[str] = Field(default_factory=list)

class ExplanationType(BaseModel):
    prediction_id: str
    feature_importances: Dict[str, float] = Field(default_factory=dict)
    model_confidence: float = 0.0

class PredictionType(BaseModel):
    id: str
    asset_id: str
    model_id: str
    model_version: str
    prediction_value: float
    confidence: float
    remaining_useful_life_hours: Optional[float] = None
    failure_probability: Optional[float] = None
    explanation: Optional[ExplanationType] = None

class WorkOrderMutation(BaseModel):
    work_order_id: str
    prediction_id: str
    asset_id: str
    status: str
    mode: WorkOrderMode
    priority: str
    work_type: str
    description: str
    approval_required: bool = False

@dataclass
class AuthContext:
    user_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    authenticated: bool = False

class AuditLogger:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._entries: List[Dict] = []

    async def log(self, operation: str, status: str = "success") -> None:
        if self.enabled:
            self._entries.append({"operation": operation, "status": status, "ts": datetime.now(timezone.utc).isoformat()})

class AuthorizationMiddleware:
    def __init__(self, config: GraphQLConfig):
        self.config = config

    async def validate_token(self, token: str) -> AuthContext:
        if not token: return AuthContext(authenticated=False)
        return AuthContext(user_id="user", authenticated=True)

    async def check_permission(self, context: AuthContext, op: str) -> bool:
        return context.authenticated

class QueryResolver:
    def __init__(self):
        self._assets: Dict[str, AssetType] = {}
        self._predictions: Dict[str, PredictionType] = {}
        self._lock = asyncio.Lock()

    async def resolve_asset(self, id: str) -> Optional[AssetType]:
        async with self._lock:
            return self._assets.get(id)

    async def resolve_predictions(self, asset_id: str, limit: int = 100) -> List[PredictionType]:
        async with self._lock:
            return [p for p in self._predictions.values() if p.asset_id == asset_id][:limit]

    async def add_asset(self, asset: AssetType) -> None:
        async with self._lock:
            self._assets[asset.id] = asset

    async def add_prediction(self, prediction: PredictionType) -> None:
        async with self._lock:
            self._predictions[prediction.id] = prediction

class MutationResolver:
    def __init__(self, query_resolver: QueryResolver):
        self.query_resolver = query_resolver
        self._work_orders: Dict[str, WorkOrderMutation] = {}

    async def resolve_request_work_order(self, prediction_id: str, mode: WorkOrderMode = WorkOrderMode.DRAFT) -> WorkOrderMutation:
        prediction = self.query_resolver._predictions.get(prediction_id)
        if not prediction: raise ValueError(f"Prediction not found: {prediction_id}")
        wo = WorkOrderMutation(work_order_id=f"WO-{uuid4().hex[:8].upper()}", prediction_id=prediction_id, asset_id=prediction.asset_id, status="draft" if mode == WorkOrderMode.DRAFT else "pending", mode=mode, priority="medium", work_type="predictive", description=f"Maint for {prediction_id}", approval_required=mode == WorkOrderMode.HUMAN_IN_LOOP)
        self._work_orders[wo.work_order_id] = wo
        return wo

class GraphQLService:
    def __init__(self, config: Optional[GraphQLConfig] = None):
        self.config = config or GraphQLConfig()
        self.audit_logger = AuditLogger(self.config.enable_audit_logging)
        self.auth_middleware = AuthorizationMiddleware(self.config)
        self.query_resolver = QueryResolver()
        self.mutation_resolver = MutationResolver(self.query_resolver)
        self._running = False
        self._requests = 0

    async def start(self) -> None:
        logger.info(f"Starting GraphQL on {self.config.host}:{self.config.port}")
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def execute_query(self, query: str, variables: Optional[Dict] = None) -> Dict:
        self._requests += 1
        try:
            result = await self._execute(query, variables)
            await self.audit_logger.log("query", "success")
            return {"data": result, "errors": None}
        except Exception as e:
            await self.audit_logger.log("query", "error")
            return {"data": None, "errors": [{"message": str(e)}]}

    async def _execute(self, query: str, variables: Optional[Dict]) -> Dict:
        q = query.lower()
        if "asset(" in q and variables and "id" in variables:
            asset = await self.query_resolver.resolve_asset(variables["id"])
            return {"asset": asset.dict() if asset else None}
        if "predictions(" in q and variables and "assetId" in variables:
            preds = await self.query_resolver.resolve_predictions(variables["assetId"])
            return {"predictions": [p.dict() for p in preds]}
        return {}

    @property
    def stats(self) -> Dict:
        return {"running": self._running, "requests": self._requests}

def create_graphql_service(config_dict: Optional[Dict] = None) -> GraphQLService:
    config = GraphQLConfig(**config_dict) if config_dict else GraphQLConfig()
    return GraphQLService(config)

async def start_graphql_service(config_dict: Optional[Dict] = None) -> GraphQLService:
    svc = create_graphql_service(config_dict)
    await svc.start()
    return svc
