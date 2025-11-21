# -*- coding: utf-8 -*-
"""
GreenLang Redis Cache Architecture
Version: 1.0.0
High-performance caching layer for 100K+ requests/second
"""

import os
import redis
import json
import hashlib
import pickle
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import aioredis
from redis.sentinel import Sentinel
from rediscluster import RedisCluster
from greenlang.determinism import DeterministicClock

# ==============================================
# REDIS CONFIGURATION
# ==============================================

class RedisConfig:
    """Redis cluster configuration for production deployment."""

    # Cluster nodes
    CLUSTER_NODES = [
        {"host": "redis-node-1.greenlang.internal", "port": 7000},
        {"host": "redis-node-2.greenlang.internal", "port": 7000},
        {"host": "redis-node-3.greenlang.internal", "port": 7000},
        {"host": "redis-node-4.greenlang.internal", "port": 7000},
        {"host": "redis-node-5.greenlang.internal", "port": 7000},
        {"host": "redis-node-6.greenlang.internal", "port": 7000},
    ]

    # Sentinel configuration for HA
    SENTINEL_NODES = [
        ("sentinel-1.greenlang.internal", 26379),
        ("sentinel-2.greenlang.internal", 26379),
        ("sentinel-3.greenlang.internal", 26379),
    ]

    MASTER_NAME = "greenlang-master"

    # Connection pool settings
    MAX_CONNECTIONS = 500
    CONNECTION_TIMEOUT = 5
    SOCKET_TIMEOUT = 5
    SOCKET_CONNECT_TIMEOUT = 5
    SOCKET_KEEPALIVE = True
    SOCKET_KEEPALIVE_OPTIONS = {
        1: 1,  # TCP_KEEPIDLE
        2: 1,  # TCP_KEEPINTVL
        3: 3,  # TCP_KEEPCNT
    }

    # Performance settings
    DECODE_RESPONSES = False  # Keep False for binary data support
    ENCODING = "utf-8"
    MAX_RETRIES = 3
    RETRY_ON_TIMEOUT = True

    # Persistence configuration
    PERSISTENCE = {
        "aof_enabled": True,
        "aof_fsync": "everysec",  # Balance between performance and durability
        "save_rules": [
            (900, 1),    # Save after 900 sec if at least 1 key changed
            (300, 10),   # Save after 300 sec if at least 10 keys changed
            (60, 10000), # Save after 60 sec if at least 10000 keys changed
        ],
        "maxmemory": "8gb",
        "maxmemory_policy": "allkeys-lru",
    }

# ==============================================
# CACHE KEY PATTERNS
# ==============================================

class CacheKeyPattern:
    """Standardized cache key patterns for different data types."""

    # User session cache
    USER_SESSION = "session:{user_id}:{session_id}"
    USER_PROFILE = "user:profile:{user_id}"
    USER_PERMISSIONS = "user:permissions:{user_id}"

    # Organization cache
    ORG_SETTINGS = "org:{org_id}:settings"
    ORG_METADATA = "org:{org_id}:metadata"
    ORG_USERS = "org:{org_id}:users"

    # Emissions data cache
    EMISSIONS_DAILY = "emissions:{org_id}:{date}:daily"
    EMISSIONS_MONTHLY = "emissions:{org_id}:{year}:{month}"
    EMISSIONS_AGGREGATE = "emissions:{org_id}:aggregate:{period}"

    # Supply chain cache
    SUPPLIER_DATA = "supplier:{supplier_id}:data"
    SUPPLIER_METRICS = "supplier:{supplier_id}:metrics:{period}"
    PROCUREMENT_ORDERS = "procurement:{org_id}:orders:{date}"

    # CSRD reporting cache
    CSRD_REPORT = "csrd:{org_id}:report:{report_id}"
    CSRD_DRAFT = "csrd:{org_id}:draft:{report_id}"
    CSRD_CALCULATIONS = "csrd:{org_id}:calc:{metric}:{period}"

    # API response cache
    API_RESPONSE = "api:{endpoint}:{params_hash}"
    API_RATE_LIMIT = "ratelimit:{api_key}:{window}"

    # Analytics cache
    DASHBOARD_DATA = "dashboard:{org_id}:{dashboard_id}:{widget_id}"
    KPI_VALUE = "kpi:{org_id}:{kpi_id}:{period}"
    BENCHMARK_DATA = "benchmark:{industry}:{metric}:{year}"

    # Search cache
    SEARCH_RESULTS = "search:{index}:{query_hash}"
    SEARCH_SUGGEST = "search:suggest:{prefix}"

    # ML model cache
    ML_PREDICTION = "ml:prediction:{model_id}:{input_hash}"
    ML_FEATURE = "ml:feature:{feature_set}:{entity_id}"

# ==============================================
# CACHE MANAGER
# ==============================================

class CacheManager:
    """Main cache management class with all caching strategies."""

    def __init__(self, config: RedisConfig):
        """Initialize Redis connections for different purposes."""

        # Main cluster connection
        self.cluster = RedisCluster(
            startup_nodes=config.CLUSTER_NODES,
            decode_responses=False,
            skip_full_coverage_check=True,
            max_connections=config.MAX_CONNECTIONS,
            socket_timeout=config.SOCKET_TIMEOUT,
            socket_connect_timeout=config.SOCKET_CONNECT_TIMEOUT,
            retry_on_timeout=config.RETRY_ON_TIMEOUT,
            max_retries=config.MAX_RETRIES
        )

        # Sentinel connection for HA
        self.sentinel = Sentinel(
            config.SENTINEL_NODES,
            socket_timeout=config.SOCKET_TIMEOUT
        )
        self.master = self.sentinel.master_for(
            config.MASTER_NAME,
            socket_timeout=config.SOCKET_TIMEOUT
        )

        # Async connection pool
        self.async_pool = None
        self.config = config

    async def init_async(self):
        """Initialize async Redis connection."""
        self.async_pool = await aioredis.create_redis_pool(
            'redis://redis-cluster.greenlang.internal',
            minsize=10,
            maxsize=50
        )

    # ==============================================
    # SESSION MANAGEMENT
    # ==============================================

    def set_user_session(
        self,
        user_id: str,
        session_id: str,
        session_data: Dict[str, Any],
        ttl: int = 3600
    ) -> bool:
        """Store user session with automatic expiration."""
        key = f"session:{user_id}:{session_id}"
        pipeline = self.cluster.pipeline()

        # Store session data
        pipeline.hset(key, mapping={
            'user_id': user_id,
            'session_id': session_id,
            'created_at': DeterministicClock.now().isoformat(),
            'data': json.dumps(session_data)
        })

        # Set expiration
        pipeline.expire(key, ttl)

        # Add to active sessions set
        pipeline.sadd(f"active_sessions:{user_id}", session_id)

        # Track session in sorted set by timestamp
        pipeline.zadd(
            "sessions:by_time",
            {f"{user_id}:{session_id}": DeterministicClock.now().timestamp()}
        )

        results = pipeline.execute()
        return all(results)

    def get_user_session(self, user_id: str, session_id: str) -> Optional[Dict]:
        """Retrieve user session and refresh TTL."""
        key = f"session:{user_id}:{session_id}"

        pipeline = self.cluster.pipeline()
        pipeline.hgetall(key)
        pipeline.expire(key, 3600)  # Refresh TTL on access

        results = pipeline.execute()
        session_data = results[0]

        if session_data:
            session_data['data'] = json.loads(session_data.get('data', '{}'))
            return session_data
        return None

    # ==============================================
    # EMISSIONS DATA CACHING
    # ==============================================

    def cache_emissions_data(
        self,
        org_id: str,
        date: str,
        emissions_data: Dict[str, Any],
        ttl: int = 86400
    ) -> bool:
        """Cache daily emissions data with compression."""
        key = f"emissions:{org_id}:{date}:daily"

        # Compress large data
        compressed_data = self._compress_data(emissions_data)

        # Store with metadata
        pipeline = self.cluster.pipeline()
        pipeline.set(
            key,
            compressed_data,
            ex=ttl,
            nx=False  # Overwrite if exists
        )

        # Update index
        pipeline.zadd(
            f"emissions:index:{org_id}",
            {date: datetime.strptime(date, "%Y-%m-%d").timestamp()}
        )

        # Store aggregates for quick access
        if 'total_co2e' in emissions_data:
            pipeline.zadd(
                f"emissions:totals:{org_id}",
                {date: emissions_data['total_co2e']}
            )

        results = pipeline.execute()
        return results[0]

    def get_emissions_range(
        self,
        org_id: str,
        start_date: str,
        end_date: str
    ) -> List[Dict]:
        """Get emissions data for date range."""
        # Get dates in range from index
        start_ts = datetime.strptime(start_date, "%Y-%m-%d").timestamp()
        end_ts = datetime.strptime(end_date, "%Y-%m-%d").timestamp()

        dates = self.cluster.zrangebyscore(
            f"emissions:index:{org_id}",
            start_ts,
            end_ts
        )

        # Batch fetch data
        pipeline = self.cluster.pipeline()
        for date_bytes in dates:
            date = date_bytes.decode('utf-8')
            pipeline.get(f"emissions:{org_id}:{date}:daily")

        results = pipeline.execute()

        # Decompress and return
        emissions_list = []
        for data in results:
            if data:
                emissions_list.append(self._decompress_data(data))

        return emissions_list

    # ==============================================
    # API RESPONSE CACHING
    # ==============================================

    def cache_api_response(
        self,
        endpoint: str,
        params: Dict[str, Any],
        response: Any,
        ttl: int = 300
    ) -> bool:
        """Cache API response with intelligent TTL."""
        # Generate cache key
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        key = f"api:{endpoint}:{params_hash}"

        # Determine TTL based on endpoint
        ttl = self._get_dynamic_ttl(endpoint, params)

        # Cache response
        pipeline = self.cluster.pipeline()
        pipeline.setex(
            key,
            ttl,
            pickle.dumps(response)
        )

        # Track cache hit statistics
        stats_key = f"api:stats:{endpoint}"
        pipeline.hincrby(stats_key, "cached", 1)
        pipeline.expire(stats_key, 86400)

        results = pipeline.execute()
        return results[0]

    def get_cached_api_response(
        self,
        endpoint: str,
        params: Dict[str, Any]
    ) -> Optional[Any]:
        """Get cached API response if available."""
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        key = f"api:{endpoint}:{params_hash}"

        # Get from cache
        data = self.cluster.get(key)

        if data:
            # Track hit
            self.cluster.hincrby(f"api:stats:{endpoint}", "hits", 1)
            return pickle.loads(data)
        else:
            # Track miss
            self.cluster.hincrby(f"api:stats:{endpoint}", "misses", 1)
            return None

    # ==============================================
    # RATE LIMITING
    # ==============================================

    def check_rate_limit(
        self,
        api_key: str,
        limit: int = 100,
        window: int = 60
    ) -> tuple[bool, int]:
        """Check and update rate limit using sliding window."""
        now = DeterministicClock.now()
        window_start = now - timedelta(seconds=window)

        key = f"ratelimit:{api_key}:{window}"

        pipeline = self.cluster.pipeline()

        # Remove old entries
        pipeline.zremrangebyscore(
            key,
            0,
            window_start.timestamp()
        )

        # Count requests in window
        pipeline.zcard(key)

        # Add current request
        pipeline.zadd(key, {str(now.timestamp()): now.timestamp()})

        # Set expiry
        pipeline.expire(key, window + 1)

        results = pipeline.execute()
        request_count = results[1]

        return request_count < limit, limit - request_count

    # ==============================================
    # DISTRIBUTED LOCKING
    # ==============================================

    def acquire_lock(
        self,
        resource: str,
        ttl: int = 10,
        blocking: bool = True,
        timeout: float = 5
    ) -> bool:
        """Acquire distributed lock using Redlock algorithm."""
        lock_key = f"lock:{resource}"
        identifier = str(DeterministicClock.now().timestamp())

        if blocking:
            end = DeterministicClock.now() + timedelta(seconds=timeout)
            while DeterministicClock.now() < end:
                if self.cluster.set(
                    lock_key,
                    identifier,
                    nx=True,
                    ex=ttl
                ):
                    return True
                asyncio.sleep(0.001)
            return False
        else:
            return self.cluster.set(
                lock_key,
                identifier,
                nx=True,
                ex=ttl
            )

    def release_lock(self, resource: str) -> bool:
        """Release distributed lock."""
        lock_key = f"lock:{resource}"
        return self.cluster.delete(lock_key) == 1

    # ==============================================
    # REAL-TIME LEADERBOARDS
    # ==============================================

    def update_sustainability_leaderboard(
        self,
        org_id: str,
        score: float,
        category: str = "overall"
    ) -> int:
        """Update organization's position in sustainability leaderboard."""
        key = f"leaderboard:{category}:current"

        # Update score
        self.cluster.zadd(key, {org_id: score})

        # Get rank (0-indexed, so add 1)
        rank = self.cluster.zrevrank(key, org_id)

        if rank is not None:
            # Cache rank for quick access
            self.cluster.setex(
                f"rank:{category}:{org_id}",
                3600,
                str(rank + 1)
            )
            return rank + 1
        return -1

    def get_leaderboard_top(
        self,
        category: str = "overall",
        limit: int = 10
    ) -> List[tuple]:
        """Get top organizations from leaderboard."""
        key = f"leaderboard:{category}:current"

        # Get top scores with organizations
        results = self.cluster.zrevrange(
            key,
            0,
            limit - 1,
            withscores=True
        )

        return [(org.decode('utf-8'), score) for org, score in results]

    # ==============================================
    # PUB/SUB FOR REAL-TIME UPDATES
    # ==============================================

    def publish_event(
        self,
        channel: str,
        event_type: str,
        data: Dict[str, Any]
    ) -> int:
        """Publish event to Redis pub/sub channel."""
        message = {
            'event_type': event_type,
            'timestamp': DeterministicClock.now().isoformat(),
            'data': data
        }

        return self.cluster.publish(
            f"events:{channel}",
            json.dumps(message)
        )

    def subscribe_to_events(
        self,
        channels: List[str],
        callback: callable
    ):
        """Subscribe to event channels."""
        pubsub = self.cluster.pubsub()

        for channel in channels:
            pubsub.subscribe(f"events:{channel}")

        # Listen for messages
        for message in pubsub.listen():
            if message['type'] == 'message':
                event_data = json.loads(message['data'])
                callback(event_data)

    # ==============================================
    # GEOSPATIAL OPERATIONS
    # ==============================================

    def add_facility_location(
        self,
        facility_id: str,
        longitude: float,
        latitude: float,
        metadata: Dict[str, Any]
    ) -> bool:
        """Add facility location for geospatial queries."""
        key = "facilities:locations"

        # Add to geospatial index
        result = self.cluster.geoadd(
            key,
            longitude,
            latitude,
            facility_id
        )

        # Store metadata
        self.cluster.hset(
            f"facility:{facility_id}",
            mapping=metadata
        )

        return result == 1

    def find_nearby_facilities(
        self,
        longitude: float,
        latitude: float,
        radius: float,
        unit: str = "km"
    ) -> List[Dict]:
        """Find facilities within radius."""
        key = "facilities:locations"

        # Search within radius
        results = self.cluster.georadius(
            key,
            longitude,
            latitude,
            radius,
            unit=unit,
            withdist=True,
            withcoord=True
        )

        # Fetch metadata for each facility
        facilities = []
        for facility_id, distance, (lon, lat) in results:
            metadata = self.cluster.hgetall(f"facility:{facility_id}")
            facilities.append({
                'id': facility_id,
                'distance': distance,
                'coordinates': {'lon': lon, 'lat': lat},
                'metadata': metadata
            })

        return facilities

    # ==============================================
    # TIME SERIES DATA
    # ==============================================

    def add_time_series_data(
        self,
        metric: str,
        timestamp: int,
        value: float,
        labels: Dict[str, str] = None
    ) -> bool:
        """Add time series data point."""
        key = f"ts:{metric}"

        # Create time series if not exists
        try:
            self.cluster.execute_command(
                'TS.CREATE',
                key,
                'RETENTION', 86400000,  # 24 hours in milliseconds
                'ENCODING', 'COMPRESSED',
                'CHUNK_SIZE', 4096
            )
        except:
            pass  # Already exists

        # Add data point
        result = self.cluster.execute_command(
            'TS.ADD',
            key,
            timestamp,
            value
        )

        return result is not None

    def get_time_series_range(
        self,
        metric: str,
        start_time: int,
        end_time: int,
        aggregation: str = None,
        bucket_size: int = None
    ) -> List[tuple]:
        """Get time series data in range."""
        key = f"ts:{metric}"

        if aggregation and bucket_size:
            # Get aggregated data
            results = self.cluster.execute_command(
                'TS.RANGE',
                key,
                start_time,
                end_time,
                'AGGREGATION',
                aggregation,
                bucket_size
            )
        else:
            # Get raw data
            results = self.cluster.execute_command(
                'TS.RANGE',
                key,
                start_time,
                end_time
            )

        return results

    # ==============================================
    # HELPER METHODS
    # ==============================================

    def _compress_data(self, data: Any) -> bytes:
        """Compress data for storage."""
        import zlib
        json_str = json.dumps(data)
        return zlib.compress(json_str.encode())

    def _decompress_data(self, data: bytes) -> Any:
        """Decompress data from storage."""
        import zlib
        json_str = zlib.decompress(data).decode()
        return json.loads(json_str)

    def _get_dynamic_ttl(self, endpoint: str, params: Dict) -> int:
        """Calculate dynamic TTL based on endpoint and params."""
        # Real-time data: 30 seconds
        if 'real-time' in endpoint:
            return 30

        # Daily aggregates: 1 hour
        if 'daily' in endpoint:
            return 3600

        # Monthly aggregates: 24 hours
        if 'monthly' in endpoint:
            return 86400

        # Historical data: 1 week
        if 'historical' in endpoint:
            return 604800

        # Default: 5 minutes
        return 300

    # ==============================================
    # CACHE WARMING
    # ==============================================

    async def warm_cache(self, org_id: str):
        """Pre-populate cache with frequently accessed data."""
        tasks = []

        # Warm organization settings
        tasks.append(self._warm_org_settings(org_id))

        # Warm recent emissions data
        tasks.append(self._warm_emissions_data(org_id))

        # Warm dashboard data
        tasks.append(self._warm_dashboard_data(org_id))

        # Warm KPI values
        tasks.append(self._warm_kpi_data(org_id))

        await asyncio.gather(*tasks)

    async def _warm_org_settings(self, org_id: str):
        """Warm organization settings cache."""
        # Fetch from database
        settings = await self._fetch_org_settings_from_db(org_id)

        # Cache for 24 hours
        key = f"org:{org_id}:settings"
        await self.async_pool.setex(
            key,
            86400,
            json.dumps(settings)
        )

    async def _warm_emissions_data(self, org_id: str):
        """Warm recent emissions data cache."""
        # Get last 30 days of emissions
        end_date = DeterministicClock.now()
        start_date = end_date - timedelta(days=30)

        # Fetch from database
        emissions_data = await self._fetch_emissions_from_db(
            org_id,
            start_date,
            end_date
        )

        # Cache each day
        for date, data in emissions_data.items():
            await self.cache_emissions_data(org_id, date, data)

    # ==============================================
    # MONITORING AND METRICS
    # ==============================================

    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        info = self.cluster.info()

        metrics = {
            'memory': {
                'used_memory': info.get('used_memory_human'),
                'used_memory_rss': info.get('used_memory_rss_human'),
                'used_memory_peak': info.get('used_memory_peak_human'),
                'memory_fragmentation_ratio': info.get('mem_fragmentation_ratio')
            },
            'stats': {
                'total_connections': info.get('total_connections_received'),
                'total_commands': info.get('total_commands_processed'),
                'instantaneous_ops': info.get('instantaneous_ops_per_sec'),
                'keyspace_hits': info.get('keyspace_hits'),
                'keyspace_misses': info.get('keyspace_misses'),
                'hit_rate': self._calculate_hit_rate(info)
            },
            'persistence': {
                'rdb_last_save': datetime.fromtimestamp(
                    info.get('rdb_last_save_time', 0)
                ).isoformat(),
                'aof_enabled': info.get('aof_enabled'),
                'aof_rewrite_in_progress': info.get('aof_rewrite_in_progress')
            },
            'replication': {
                'role': info.get('role'),
                'connected_slaves': info.get('connected_slaves'),
                'master_repl_offset': info.get('master_repl_offset')
            }
        }

        return metrics

    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calculate cache hit rate."""
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)

        if hits + misses == 0:
            return 0.0

        return round((hits / (hits + misses)) * 100, 2)

# ==============================================
# CACHE DECORATORS
# ==============================================

def cached(ttl: int = 300, key_prefix: str = None):
    """Decorator for caching function results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_prefix:
                cache_key = f"{key_prefix}:{func.__name__}"
            else:
                cache_key = func.__name__

            # Add arguments to key
            args_str = ':'.join(str(arg) for arg in args)
            kwargs_str = ':'.join(f"{k}={v}" for k, v in kwargs.items())
            cache_key = f"{cache_key}:{args_str}:{kwargs_str}"

            # Check cache
            cache_manager = CacheManager(RedisConfig())
            cached_result = cache_manager.cluster.get(cache_key)

            if cached_result:
                return pickle.loads(cached_result)

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            cache_manager.cluster.setex(
                cache_key,
                ttl,
                pickle.dumps(result)
            )

            return result

        return wrapper
    return decorator

# ==============================================
# USAGE EXAMPLES
# ==============================================

if __name__ == "__main__":
    # Initialize cache manager
    cache = CacheManager(RedisConfig())

    # Example: Cache user session
    cache.set_user_session(
        user_id="user123",
        session_id="session456",
        session_data={
            "permissions": ["read", "write"],
            "org_id": "org789",
            "preferences": {"theme": "dark"}
        }
    )

    # Example: Cache emissions data
    cache.cache_emissions_data(
        org_id="org789",
        date="2024-01-01",
        emissions_data={
            "total_co2e": 1234.56,
            "scope1": 500.0,
            "scope2": 400.0,
            "scope3": 334.56,
            "data_quality_score": 85
        }
    )

    # Example: Rate limiting
    # Note: In production, api_key should come from request headers or environment
    example_api_key = os.getenv("EXAMPLE_API_KEY", "example_key_placeholder")
    allowed, remaining = cache.check_rate_limit(
        api_key=example_api_key,
        limit=100,
        window=60
    )

    # Example: Update leaderboard
    rank = cache.update_sustainability_leaderboard(
        org_id="org789",
        score=92.5,
        category="emissions_reduction"
    )

    # Example: Get cache metrics
    metrics = cache.get_cache_metrics()
    print(f"Cache hit rate: {metrics['stats']['hit_rate']}%")