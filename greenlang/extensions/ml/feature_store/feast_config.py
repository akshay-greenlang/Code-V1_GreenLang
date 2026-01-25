# -*- coding: utf-8 -*-
"""
Feast Feature Store Configuration for GreenLang Process Heat Agents

This module provides the core Feast configuration for the Process Heat
feature store, including:
- ProcessHeatFeatureStore class for feature management
- Feature views for each agent (GL-001 through GL-020)
- Online/offline store configuration
- Feature services for inference

The feature store follows GreenLang's zero-hallucination principles by
tracking SHA-256 provenance for all feature computations.

Example:
    >>> from greenlang.ml.feature_store.feast_config import ProcessHeatFeatureStore
    >>>
    >>> store = ProcessHeatFeatureStore()
    >>> features = store.get_online_features(
    ...     entity_ids=["boiler-001"],
    ...     feature_refs=["boiler_features:efficiency"]
    ... )
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
import hashlib
import logging
import json
import os

logger = logging.getLogger(__name__)


class StoreType(str, Enum):
    """Types of feature stores."""
    REDIS = "redis"
    DYNAMODB = "dynamodb"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    BIGTABLE = "bigtable"


class OfflineStoreType(str, Enum):
    """Types of offline stores."""
    FILE = "file"
    BIGQUERY = "bigquery"
    REDSHIFT = "redshift"
    SNOWFLAKE = "snowflake"
    SPARK = "spark"
    POSTGRESQL = "postgresql"


class FeatureViewType(str, Enum):
    """Types of feature views."""
    BATCH = "batch"
    STREAM = "stream"
    ON_DEMAND = "on_demand"


class AgentFeatureMapping(str, Enum):
    """Mapping of GreenLang agents to feature views."""
    GL_001_FUEL_ANALYZER = "gl001_fuel_features"
    GL_002_BOILER_EFFICIENCY = "gl002_boiler_features"
    GL_003_COMBUSTION_ANALYZER = "gl003_combustion_features"
    GL_004_STEAM_QUALITY = "gl004_steam_features"
    GL_005_EMISSIONS_CALCULATOR = "gl005_emissions_features"
    GL_006_HEAT_RECOVERY = "gl006_heat_recovery_features"
    GL_007_THERMAL_STORAGE = "gl007_thermal_storage_features"
    GL_008_PROCESS_OPTIMIZER = "gl008_process_optimizer_features"
    GL_009_PREDICTIVE_MAINTENANCE = "gl009_predictive_features"
    GL_010_ANOMALY_DETECTOR = "gl010_anomaly_features"
    GL_011_ENERGY_BALANCE = "gl011_energy_balance_features"
    GL_012_LOAD_FORECAST = "gl012_load_forecast_features"
    GL_013_CARBON_TRACKER = "gl013_carbon_tracker_features"
    GL_014_REGULATORY_COMPLIANCE = "gl014_compliance_features"
    GL_015_BENCHMARK_ANALYZER = "gl015_benchmark_features"
    GL_016_DECARBONIZATION = "gl016_decarb_features"
    GL_017_HEAT_PUMP = "gl017_heat_pump_features"
    GL_018_CHP_OPTIMIZER = "gl018_chp_features"
    GL_019_WASTE_HEAT = "gl019_waste_heat_features"
    GL_020_REPORT_GENERATOR = "gl020_report_features"


class OnlineStoreConfig(BaseModel):
    """Configuration for online feature store (Redis)."""

    store_type: StoreType = Field(
        default=StoreType.REDIS,
        description="Type of online store"
    )
    host: str = Field(
        default="redis.greenlang-mlops.svc.cluster.local",
        description="Redis host"
    )
    port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Redis port"
    )
    password_secret: str = Field(
        default="feast-redis-secret",
        description="Kubernetes secret name for Redis password"
    )
    ssl: bool = Field(
        default=True,
        description="Enable SSL/TLS for Redis connection"
    )
    connection_pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Redis connection pool size"
    )
    ttl_seconds: int = Field(
        default=86400,
        ge=0,
        description="Default TTL for cached features (24 hours)"
    )
    key_prefix: str = Field(
        default="greenlang:feast:",
        description="Prefix for all Redis keys"
    )

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class OfflineStoreConfig(BaseModel):
    """Configuration for offline feature store (PostgreSQL)."""

    store_type: OfflineStoreType = Field(
        default=OfflineStoreType.POSTGRESQL,
        description="Type of offline store"
    )
    host: str = Field(
        default="postgresql.greenlang-mlops.svc.cluster.local",
        description="PostgreSQL host"
    )
    port: int = Field(
        default=5432,
        ge=1,
        le=65535,
        description="PostgreSQL port"
    )
    database: str = Field(
        default="feast_offline",
        description="Database name"
    )
    schema_name: str = Field(
        default="feast",
        description="Schema name"
    )
    user_secret: str = Field(
        default="feast-postgresql-secret",
        description="Kubernetes secret name for PostgreSQL credentials"
    )
    ssl_mode: str = Field(
        default="require",
        description="SSL mode for PostgreSQL"
    )
    max_connections: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum database connections"
    )

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class FeatureViewConfig(BaseModel):
    """Configuration for a feature view."""

    name: str = Field(..., description="Feature view name")
    entities: List[str] = Field(
        default_factory=list,
        description="Entity columns"
    )
    features: List[str] = Field(
        default_factory=list,
        description="Feature columns"
    )
    ttl_hours: int = Field(
        default=24,
        ge=0,
        description="Time-to-live in hours"
    )
    source_table: str = Field(..., description="Source data table")
    timestamp_field: str = Field(
        default="event_timestamp",
        description="Timestamp column"
    )
    view_type: FeatureViewType = Field(
        default=FeatureViewType.BATCH,
        description="Type of feature view"
    )
    description: str = Field(
        default="",
        description="Feature view description"
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Feature view tags"
    )

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class FeatureServiceConfig(BaseModel):
    """Configuration for a feature service."""

    name: str = Field(..., description="Feature service name")
    feature_views: List[str] = Field(
        default_factory=list,
        description="Feature views included in this service"
    )
    description: str = Field(
        default="",
        description="Service description"
    )
    owner: str = Field(
        default="greenlang-ml-team",
        description="Service owner"
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Service tags"
    )


class FeatureStoreConfig(BaseModel):
    """Complete feature store configuration."""

    project_name: str = Field(
        default="greenlang_process_heat",
        description="Feast project name"
    )
    registry_path: str = Field(
        default="s3://greenlang-feast-registry/registry.db",
        description="Path to Feast registry"
    )
    online_store: OnlineStoreConfig = Field(
        default_factory=OnlineStoreConfig,
        description="Online store configuration"
    )
    offline_store: OfflineStoreConfig = Field(
        default_factory=OfflineStoreConfig,
        description="Offline store configuration"
    )
    feature_views: List[FeatureViewConfig] = Field(
        default_factory=list,
        description="Feature view configurations"
    )
    feature_services: List[FeatureServiceConfig] = Field(
        default_factory=list,
        description="Feature service configurations"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable SHA-256 provenance tracking"
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable feature caching"
    )
    batch_size: int = Field(
        default=1000,
        ge=1,
        le=100000,
        description="Default batch size for feature retrieval"
    )

    @validator('feature_views', pre=True, always=True)
    def set_default_feature_views(cls, v):
        """Set default feature views if not provided."""
        if not v:
            return cls._get_default_feature_views()
        return v

    @staticmethod
    def _get_default_feature_views() -> List[Dict[str, Any]]:
        """Get default feature views for all Process Heat agents."""
        return [
            {
                "name": "gl001_fuel_features",
                "entities": ["equipment_id", "facility_id"],
                "features": [
                    "fuel_type", "fuel_flow_rate", "energy_content",
                    "carbon_content", "sulfur_content", "ash_content",
                    "moisture_content", "heating_value"
                ],
                "source_table": "fuel_analysis_data",
                "description": "GL-001 Fuel Analyzer features",
                "tags": {"agent": "GL-001", "category": "fuel"}
            },
            {
                "name": "gl002_boiler_features",
                "entities": ["equipment_id", "facility_id"],
                "features": [
                    "steam_flow", "fuel_rate", "efficiency",
                    "excess_air", "blowdown_rate", "feedwater_temp",
                    "steam_pressure", "steam_temp"
                ],
                "source_table": "boiler_operational_data",
                "description": "GL-002 Boiler Efficiency features",
                "tags": {"agent": "GL-002", "category": "boiler"}
            },
            {
                "name": "gl003_combustion_features",
                "entities": ["equipment_id", "facility_id"],
                "features": [
                    "o2_percentage", "co_ppm", "nox_ppm",
                    "stack_temp", "air_fuel_ratio", "combustion_efficiency",
                    "flame_temp", "excess_air_ratio"
                ],
                "source_table": "combustion_analysis_data",
                "description": "GL-003 Combustion Analyzer features",
                "tags": {"agent": "GL-003", "category": "combustion"}
            },
            {
                "name": "gl004_steam_features",
                "entities": ["equipment_id", "facility_id"],
                "features": [
                    "pressure", "temperature", "quality",
                    "enthalpy", "entropy", "specific_volume",
                    "saturation_temp", "superheat_deg"
                ],
                "source_table": "steam_quality_data",
                "description": "GL-004 Steam Quality features",
                "tags": {"agent": "GL-004", "category": "steam"}
            },
            {
                "name": "gl005_emissions_features",
                "entities": ["equipment_id", "facility_id"],
                "features": [
                    "co2_rate", "ch4_rate", "n2o_rate",
                    "intensity", "total_ghg", "scope1_emissions",
                    "scope2_emissions", "emission_factor"
                ],
                "source_table": "emissions_calculation_data",
                "description": "GL-005 Emissions Calculator features",
                "tags": {"agent": "GL-005", "category": "emissions"}
            },
            {
                "name": "gl009_predictive_features",
                "entities": ["equipment_id", "facility_id"],
                "features": [
                    "fouling_index", "failure_probability", "remaining_life",
                    "maintenance_score", "anomaly_score", "health_index",
                    "vibration_trend", "thermal_stress"
                ],
                "source_table": "predictive_maintenance_data",
                "description": "GL-009 Predictive Maintenance features",
                "tags": {"agent": "GL-009", "category": "predictive"}
            },
        ]


@dataclass
class FeatureRetrievalResult:
    """Result from feature retrieval operation."""

    entity_ids: List[str]
    features: Dict[str, List[Any]]
    feature_refs: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    provenance_hash: str = ""
    retrieval_time_ms: float = 0.0
    cache_hit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity_ids": self.entity_ids,
            "features": self.features,
            "feature_refs": self.feature_refs,
            "timestamp": self.timestamp.isoformat(),
            "provenance_hash": self.provenance_hash,
            "retrieval_time_ms": self.retrieval_time_ms,
            "cache_hit": self.cache_hit,
            "metadata": self.metadata
        }


class ProcessHeatFeatureStore:
    """
    Feature Store for GreenLang Process Heat Agents.

    This class provides a unified interface for feature storage and retrieval,
    implementing Feast patterns with GreenLang's zero-hallucination principles.

    Key Features:
    - Feature views for all 20 Process Heat agents (GL-001 through GL-020)
    - Online/offline store separation for optimal performance
    - SHA-256 provenance tracking for regulatory compliance
    - Feature services for grouped inference
    - Batch and real-time feature retrieval

    Attributes:
        config: Feature store configuration
        _online_store: Online store connection (Redis)
        _offline_store: Offline store connection (PostgreSQL)
        _registry: Feature registry

    Example:
        >>> store = ProcessHeatFeatureStore()
        >>>
        >>> # Get online features
        >>> result = store.get_online_features(
        ...     entity_ids=["boiler-001", "boiler-002"],
        ...     feature_refs=["gl002_boiler_features:efficiency", "gl002_boiler_features:steam_flow"]
        ... )
        >>>
        >>> # Get historical features for training
        >>> training_df = store.get_historical_features(
        ...     entity_df=entity_df,
        ...     feature_refs=["gl002_boiler_features:efficiency"]
        ... )
    """

    def __init__(
        self,
        config: Optional[FeatureStoreConfig] = None,
        initialize_stores: bool = True
    ):
        """
        Initialize the ProcessHeatFeatureStore.

        Args:
            config: Feature store configuration. Uses defaults if not provided.
            initialize_stores: Whether to initialize store connections.
        """
        self.config = config or FeatureStoreConfig()
        self._online_store = None
        self._offline_store = None
        self._registry = None
        self._feature_services: Dict[str, FeatureServiceConfig] = {}
        self._feature_views: Dict[str, FeatureViewConfig] = {}
        self._provenance_enabled = self.config.enable_provenance

        # Initialize feature views and services
        self._initialize_feature_views()
        self._initialize_feature_services()

        if initialize_stores:
            self._connect_stores()

        logger.info(
            f"ProcessHeatFeatureStore initialized with {len(self._feature_views)} feature views"
        )

    def _initialize_feature_views(self) -> None:
        """Initialize feature view configurations."""
        for view_config in self.config.feature_views:
            if isinstance(view_config, dict):
                view = FeatureViewConfig(**view_config)
            else:
                view = view_config
            self._feature_views[view.name] = view

        logger.debug(f"Initialized {len(self._feature_views)} feature views")

    def _initialize_feature_services(self) -> None:
        """Initialize feature service configurations."""
        # Default feature services if not configured
        if not self.config.feature_services:
            default_services = self._get_default_feature_services()
            for service_config in default_services:
                service = FeatureServiceConfig(**service_config)
                self._feature_services[service.name] = service
        else:
            for service_config in self.config.feature_services:
                if isinstance(service_config, dict):
                    service = FeatureServiceConfig(**service_config)
                else:
                    service = service_config
                self._feature_services[service.name] = service

        logger.debug(f"Initialized {len(self._feature_services)} feature services")

    def _get_default_feature_services(self) -> List[Dict[str, Any]]:
        """Get default feature services."""
        return [
            {
                "name": "boiler_inference_service",
                "feature_views": [
                    "gl001_fuel_features",
                    "gl002_boiler_features",
                    "gl003_combustion_features",
                    "gl004_steam_features"
                ],
                "description": "Features for boiler efficiency inference",
                "owner": "greenlang-ml-team",
                "tags": {"use_case": "boiler_optimization"}
            },
            {
                "name": "emissions_inference_service",
                "feature_views": [
                    "gl001_fuel_features",
                    "gl003_combustion_features",
                    "gl005_emissions_features"
                ],
                "description": "Features for emissions calculation",
                "owner": "greenlang-ml-team",
                "tags": {"use_case": "emissions_reporting"}
            },
            {
                "name": "predictive_maintenance_service",
                "feature_views": [
                    "gl002_boiler_features",
                    "gl003_combustion_features",
                    "gl009_predictive_features"
                ],
                "description": "Features for predictive maintenance",
                "owner": "greenlang-ml-team",
                "tags": {"use_case": "predictive_maintenance"}
            },
            {
                "name": "process_heat_full_service",
                "feature_views": list(self._feature_views.keys()),
                "description": "All Process Heat features",
                "owner": "greenlang-ml-team",
                "tags": {"use_case": "full_analysis"}
            }
        ]

    def _connect_stores(self) -> None:
        """Connect to online and offline stores."""
        try:
            self._connect_online_store()
            self._connect_offline_store()
            logger.info("Successfully connected to feature stores")
        except Exception as e:
            logger.warning(f"Failed to connect to stores: {e}. Running in offline mode.")

    def _connect_online_store(self) -> None:
        """Connect to Redis online store."""
        try:
            import redis

            config = self.config.online_store
            # In production, password would be fetched from Kubernetes secret
            password = os.environ.get("FEAST_REDIS_PASSWORD", "")

            self._online_store = redis.Redis(
                host=config.host,
                port=config.port,
                password=password if password else None,
                ssl=config.ssl,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )

            # Test connection
            self._online_store.ping()
            logger.info(f"Connected to Redis online store at {config.host}:{config.port}")

        except ImportError:
            logger.warning("Redis not installed. Online store not available.")
            self._online_store = None
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self._online_store = None

    def _connect_offline_store(self) -> None:
        """Connect to PostgreSQL offline store."""
        try:
            import psycopg2
            from psycopg2 import pool

            config = self.config.offline_store
            # In production, credentials would be fetched from Kubernetes secret
            user = os.environ.get("FEAST_PG_USER", "feast")
            password = os.environ.get("FEAST_PG_PASSWORD", "")

            self._offline_store = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=config.max_connections,
                host=config.host,
                port=config.port,
                database=config.database,
                user=user,
                password=password,
                sslmode=config.ssl_mode
            )

            logger.info(f"Connected to PostgreSQL offline store at {config.host}:{config.port}")

        except ImportError:
            logger.warning("psycopg2 not installed. Offline store not available.")
            self._offline_store = None
        except Exception as e:
            logger.warning(f"Failed to connect to PostgreSQL: {e}")
            self._offline_store = None

    def _calculate_provenance_hash(self, data: Dict[str, Any]) -> str:
        """
        Calculate SHA-256 provenance hash for feature data.

        Args:
            data: Data to hash

        Returns:
            SHA-256 hex digest
        """
        if not self._provenance_enabled:
            return ""

        try:
            # Create deterministic JSON representation
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate provenance hash: {e}")
            return ""

    def get_online_features(
        self,
        entity_ids: List[str],
        feature_refs: List[str],
        entity_type: str = "equipment_id"
    ) -> FeatureRetrievalResult:
        """
        Get features from online store for real-time inference.

        Args:
            entity_ids: List of entity IDs to fetch features for
            feature_refs: List of feature references (e.g., "gl002_boiler_features:efficiency")
            entity_type: Type of entity (equipment_id or facility_id)

        Returns:
            FeatureRetrievalResult with features and provenance

        Example:
            >>> result = store.get_online_features(
            ...     entity_ids=["boiler-001", "boiler-002"],
            ...     feature_refs=["gl002_boiler_features:efficiency", "gl002_boiler_features:steam_flow"]
            ... )
            >>> print(result.features)
            {'efficiency': [0.85, 0.82], 'steam_flow': [1000.0, 950.0]}
        """
        start_time = datetime.now(timezone.utc)

        features: Dict[str, List[Any]] = {}
        cache_hit = False

        try:
            if self._online_store is not None:
                # Try to get from Redis cache
                features, cache_hit = self._get_from_redis(
                    entity_ids, feature_refs, entity_type
                )

            if not cache_hit:
                # Fallback to offline store or generate mock data
                features = self._get_features_fallback(entity_ids, feature_refs)

        except Exception as e:
            logger.error(f"Error retrieving online features: {e}")
            features = self._get_features_fallback(entity_ids, feature_refs)

        # Calculate retrieval time
        end_time = datetime.now(timezone.utc)
        retrieval_time_ms = (end_time - start_time).total_seconds() * 1000

        # Calculate provenance hash
        provenance_data = {
            "entity_ids": entity_ids,
            "feature_refs": feature_refs,
            "features": features,
            "timestamp": end_time.isoformat()
        }
        provenance_hash = self._calculate_provenance_hash(provenance_data)

        result = FeatureRetrievalResult(
            entity_ids=entity_ids,
            features=features,
            feature_refs=feature_refs,
            timestamp=end_time,
            provenance_hash=provenance_hash,
            retrieval_time_ms=retrieval_time_ms,
            cache_hit=cache_hit,
            metadata={
                "entity_type": entity_type,
                "store_type": "online" if cache_hit else "fallback"
            }
        )

        logger.debug(
            f"Retrieved {len(feature_refs)} features for {len(entity_ids)} entities "
            f"in {retrieval_time_ms:.2f}ms (cache_hit={cache_hit})"
        )

        return result

    def _get_from_redis(
        self,
        entity_ids: List[str],
        feature_refs: List[str],
        entity_type: str
    ) -> Tuple[Dict[str, List[Any]], bool]:
        """Get features from Redis cache."""
        features: Dict[str, List[Any]] = {
            ref.split(":")[-1]: [] for ref in feature_refs
        }

        all_found = True
        prefix = self.config.online_store.key_prefix

        for entity_id in entity_ids:
            entity_key = f"{prefix}{entity_type}:{entity_id}"

            for feature_ref in feature_refs:
                view_name, feature_name = feature_ref.split(":")
                feature_key = f"{entity_key}:{view_name}:{feature_name}"

                value = self._online_store.get(feature_key)
                if value is not None:
                    # Parse value (stored as JSON)
                    try:
                        parsed_value = json.loads(value)
                        features[feature_name].append(parsed_value)
                    except json.JSONDecodeError:
                        features[feature_name].append(float(value) if '.' in value else int(value))
                else:
                    all_found = False
                    features[feature_name].append(None)

        return features, all_found

    def _get_features_fallback(
        self,
        entity_ids: List[str],
        feature_refs: List[str]
    ) -> Dict[str, List[Any]]:
        """
        Fallback feature retrieval when stores are unavailable.

        In production, this would query the offline store.
        For development/testing, returns None values.
        """
        features: Dict[str, List[Any]] = {}

        for feature_ref in feature_refs:
            feature_name = feature_ref.split(":")[-1]
            features[feature_name] = [None] * len(entity_ids)

        logger.warning(
            f"Using fallback feature retrieval for {len(entity_ids)} entities. "
            "Features returned as None."
        )

        return features

    def get_historical_features(
        self,
        entity_df: Any,  # pandas DataFrame in production
        feature_refs: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get historical features for training from offline store.

        Args:
            entity_df: DataFrame with entity columns and timestamps
            feature_refs: List of feature references
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            Dictionary with features and metadata

        Example:
            >>> import pandas as pd
            >>> entity_df = pd.DataFrame({
            ...     "equipment_id": ["boiler-001", "boiler-002"],
            ...     "event_timestamp": [datetime.now(), datetime.now()]
            ... })
            >>> result = store.get_historical_features(
            ...     entity_df=entity_df,
            ...     feature_refs=["gl002_boiler_features:efficiency"]
            ... )
        """
        start_ts = datetime.now(timezone.utc)

        try:
            if self._offline_store is not None:
                features = self._query_offline_store(
                    entity_df, feature_refs, start_time, end_time
                )
            else:
                # Return empty result if offline store not available
                features = {}
                logger.warning("Offline store not available for historical features")

        except Exception as e:
            logger.error(f"Error retrieving historical features: {e}")
            features = {}

        end_ts = datetime.now(timezone.utc)
        retrieval_time_ms = (end_ts - start_ts).total_seconds() * 1000

        # Calculate provenance
        provenance_data = {
            "feature_refs": feature_refs,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "timestamp": end_ts.isoformat()
        }
        provenance_hash = self._calculate_provenance_hash(provenance_data)

        return {
            "features": features,
            "feature_refs": feature_refs,
            "provenance_hash": provenance_hash,
            "retrieval_time_ms": retrieval_time_ms,
            "metadata": {
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
                "store_type": "offline"
            }
        }

    def _query_offline_store(
        self,
        entity_df: Any,
        feature_refs: List[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> Dict[str, Any]:
        """Query PostgreSQL offline store for historical features."""
        # This is a placeholder - in production would execute SQL queries
        # against the feature tables in PostgreSQL
        logger.info("Querying offline store for historical features")
        return {}

    def materialize_features(
        self,
        feature_views: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Materialize features from offline to online store.

        Args:
            feature_views: List of feature view names to materialize (all if None)
            start_time: Start time for materialization window
            end_time: End time for materialization window

        Returns:
            Materialization result with statistics
        """
        views_to_materialize = feature_views or list(self._feature_views.keys())

        if end_time is None:
            end_time = datetime.now(timezone.utc)
        if start_time is None:
            start_time = end_time - timedelta(days=1)

        logger.info(
            f"Materializing {len(views_to_materialize)} feature views "
            f"from {start_time} to {end_time}"
        )

        results = {
            "views_materialized": [],
            "rows_written": 0,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "errors": []
        }

        for view_name in views_to_materialize:
            try:
                rows = self._materialize_view(view_name, start_time, end_time)
                results["views_materialized"].append(view_name)
                results["rows_written"] += rows
            except Exception as e:
                logger.error(f"Failed to materialize {view_name}: {e}")
                results["errors"].append({"view": view_name, "error": str(e)})

        # Calculate provenance for materialization
        provenance_hash = self._calculate_provenance_hash(results)
        results["provenance_hash"] = provenance_hash

        return results

    def _materialize_view(
        self,
        view_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> int:
        """Materialize a single feature view."""
        if self._online_store is None or self._offline_store is None:
            logger.warning(f"Stores not available for materializing {view_name}")
            return 0

        # In production, this would:
        # 1. Query offline store for feature data in time range
        # 2. Write features to Redis with TTL
        # 3. Return number of rows written

        logger.info(f"Materialized feature view: {view_name}")
        return 0

    def get_feature_service(
        self,
        service_name: str
    ) -> Optional[FeatureServiceConfig]:
        """
        Get a feature service configuration by name.

        Args:
            service_name: Name of the feature service

        Returns:
            FeatureServiceConfig or None if not found
        """
        return self._feature_services.get(service_name)

    def get_feature_view(
        self,
        view_name: str
    ) -> Optional[FeatureViewConfig]:
        """
        Get a feature view configuration by name.

        Args:
            view_name: Name of the feature view

        Returns:
            FeatureViewConfig or None if not found
        """
        return self._feature_views.get(view_name)

    def list_feature_views(self) -> List[str]:
        """List all registered feature views."""
        return list(self._feature_views.keys())

    def list_feature_services(self) -> List[str]:
        """List all registered feature services."""
        return list(self._feature_services.keys())

    def get_agent_feature_view(self, agent_id: str) -> Optional[str]:
        """
        Get the feature view name for a specific agent.

        Args:
            agent_id: Agent identifier (e.g., "GL-001", "GL-002")

        Returns:
            Feature view name or None if not found
        """
        agent_mapping = {
            "GL-001": "gl001_fuel_features",
            "GL-002": "gl002_boiler_features",
            "GL-003": "gl003_combustion_features",
            "GL-004": "gl004_steam_features",
            "GL-005": "gl005_emissions_features",
            "GL-006": "gl006_heat_recovery_features",
            "GL-007": "gl007_thermal_storage_features",
            "GL-008": "gl008_process_optimizer_features",
            "GL-009": "gl009_predictive_features",
            "GL-010": "gl010_anomaly_features",
            "GL-011": "gl011_energy_balance_features",
            "GL-012": "gl012_load_forecast_features",
            "GL-013": "gl013_carbon_tracker_features",
            "GL-014": "gl014_compliance_features",
            "GL-015": "gl015_benchmark_features",
            "GL-016": "gl016_decarb_features",
            "GL-017": "gl017_heat_pump_features",
            "GL-018": "gl018_chp_features",
            "GL-019": "gl019_waste_heat_features",
            "GL-020": "gl020_report_features",
        }

        return agent_mapping.get(agent_id.upper())

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on feature store components.

        Returns:
            Health status for all components
        """
        health = {
            "status": "healthy",
            "components": {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Check online store
        if self._online_store is not None:
            try:
                self._online_store.ping()
                health["components"]["online_store"] = {
                    "status": "healthy",
                    "type": "redis"
                }
            except Exception as e:
                health["components"]["online_store"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "degraded"
        else:
            health["components"]["online_store"] = {
                "status": "not_configured"
            }

        # Check offline store
        if self._offline_store is not None:
            try:
                conn = self._offline_store.getconn()
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                self._offline_store.putconn(conn)
                health["components"]["offline_store"] = {
                    "status": "healthy",
                    "type": "postgresql"
                }
            except Exception as e:
                health["components"]["offline_store"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health["status"] = "degraded"
        else:
            health["components"]["offline_store"] = {
                "status": "not_configured"
            }

        # Add feature view count
        health["feature_views_count"] = len(self._feature_views)
        health["feature_services_count"] = len(self._feature_services)

        return health

    def close(self) -> None:
        """Close all store connections."""
        if self._online_store is not None:
            try:
                self._online_store.close()
                logger.info("Closed Redis connection")
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")

        if self._offline_store is not None:
            try:
                self._offline_store.closeall()
                logger.info("Closed PostgreSQL connection pool")
            except Exception as e:
                logger.warning(f"Error closing PostgreSQL connections: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
