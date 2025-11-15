"""
LLMCapableAgent - Enhanced agent with LLM, caching, messaging, and RAG capabilities.

This module extends BaseAgent with production infrastructure including:
- Multi-provider LLM routing (Anthropic, OpenAI)
- 4-tier caching system
- Redis Streams message broker
- Vector database integration for RAG
- PostgreSQL state persistence

Example:
    >>> from llm_capable_agent import LLMCapableAgent, LLMAgentConfig
    >>> config = LLMAgentConfig(
    ...     name="ESGAnalyzer",
    ...     version="1.0.0",
    ...     llm_enabled=True,
    ...     cache_enabled=True,
    ...     rag_enabled=True
    ... )
    >>> class MyLLMAgent(LLMCapableAgent):
    ...     async def _execute_core(self, input_data, context):
    ...         # Use self.llm_router for LLM calls
    ...         # Use self.cache for caching
    ...         # Use self.message_broker for messaging
    ...         # Use self.vector_store for RAG
    ...         return result
"""

import asyncio
import logging
from typing import Any, Dict, Optional
from pathlib import Path

from pydantic import BaseModel, Field

from base_agent import BaseAgent, AgentConfig, ExecutionContext
from llm.llm_router import LLMRouter, RoutingStrategy
from llm.providers.anthropic_provider import AnthropicProvider
from llm.providers.openai_provider import OpenAIProvider
from llm.circuit_breaker import CircuitBreaker
from llm.rate_limiter import RateLimiter
from llm.cost_tracker import CostTracker
from cache.cache_manager import CacheManager
from messaging.redis_streams_broker import RedisStreamsBroker
from rag.vector_stores.factory import VectorStoreFactory, VectorStoreType
from database.postgres_manager import PostgresManager
from cache.redis_manager import RedisManager

logger = logging.getLogger(__name__)


class LLMAgentConfig(AgentConfig):
    """Extended configuration for LLM-capable agents."""

    # LLM configuration
    llm_enabled: bool = Field(True, description="Enable LLM capabilities")
    llm_routing_strategy: RoutingStrategy = Field(
        RoutingStrategy.PRIORITY,
        description="LLM provider routing strategy"
    )
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    default_model: str = Field("claude-3-5-sonnet-20241022", description="Default LLM model")
    max_tokens: int = Field(4096, ge=1, le=100000, description="Max tokens per request")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="LLM temperature")

    # Caching configuration
    cache_enabled: bool = Field(True, description="Enable response caching")
    cache_ttl_seconds: int = Field(3600, ge=60, description="Cache TTL in seconds")

    # Messaging configuration
    messaging_enabled: bool = Field(False, description="Enable message broker")
    redis_url: Optional[str] = Field(None, description="Redis URL for messaging")

    # RAG configuration
    rag_enabled: bool = Field(False, description="Enable RAG capabilities")
    vector_store_type: VectorStoreType = Field(
        VectorStoreType.CHROMADB,
        description="Vector store type"
    )
    collection_name: Optional[str] = Field(None, description="Vector store collection name")

    # Database configuration
    postgres_url: Optional[str] = Field(None, description="PostgreSQL connection URL")

    # Cost tracking
    cost_tracking_enabled: bool = Field(True, description="Enable cost tracking")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for cost allocation")


class LLMCapableAgent(BaseAgent):
    """
    Enhanced agent base class with LLM and infrastructure capabilities.

    This class extends BaseAgent with production-ready infrastructure:
    - Multi-provider LLM routing with failover
    - 4-tier caching (in-memory → local Redis → cluster → PostgreSQL)
    - Redis Streams message broker for agent coordination
    - Vector database for RAG (ChromaDB/Pinecone)
    - PostgreSQL for state persistence
    - Cost tracking per tenant/agent

    Subclasses can use:
        self.llm_router - for LLM API calls
        self.cache - for caching responses
        self.message_broker - for agent messaging
        self.vector_store - for RAG operations
        self.db_manager - for database operations

    Example:
        >>> class ESGAnalyzerAgent(LLMCapableAgent):
        ...     async def _execute_core(self, input_data, context):
        ...         # Use LLM to analyze text
        ...         response = await self.llm_router.generate(
        ...             prompt="Analyze ESG report: " + input_data,
        ...             max_tokens=2000
        ...         )
        ...         return response.content
    """

    def __init__(self, config: LLMAgentConfig):
        """
        Initialize LLM-capable agent.

        Args:
            config: LLM agent configuration
        """
        super().__init__(config)
        self.llm_config = config

        # Infrastructure components (initialized in _initialize_core)
        self.llm_router: Optional[LLMRouter] = None
        self.cache: Optional[CacheManager] = None
        self.message_broker: Optional[RedisStreamsBroker] = None
        self.vector_store = None
        self.db_manager: Optional[PostgresManager] = None
        self.redis_manager: Optional[RedisManager] = None
        self.cost_tracker: Optional[CostTracker] = None

        self._logger.info(
            f"LLMCapableAgent created: llm={config.llm_enabled}, "
            f"cache={config.cache_enabled}, rag={config.rag_enabled}, "
            f"messaging={config.messaging_enabled}"
        )

    async def _initialize_core(self) -> None:
        """
        Initialize infrastructure components.

        Sets up:
        1. LLM providers and router (if enabled)
        2. Cache manager (if enabled)
        3. Message broker (if enabled)
        4. Vector store for RAG (if enabled)
        5. Database connections (if configured)
        """
        self._logger.info("Initializing LLM-capable agent infrastructure")

        # Initialize Redis manager (used by cache and messaging)
        if self.llm_config.cache_enabled or self.llm_config.messaging_enabled:
            await self._initialize_redis()

        # Initialize PostgreSQL (if configured)
        if self.llm_config.postgres_url:
            await self._initialize_postgres()

        # Initialize LLM infrastructure
        if self.llm_config.llm_enabled:
            await self._initialize_llm()

        # Initialize cache
        if self.llm_config.cache_enabled:
            await self._initialize_cache()

        # Initialize message broker
        if self.llm_config.messaging_enabled:
            await self._initialize_messaging()

        # Initialize RAG
        if self.llm_config.rag_enabled:
            await self._initialize_rag()

        self._logger.info("LLM-capable agent infrastructure initialized successfully")

    async def _initialize_redis(self) -> None:
        """Initialize Redis manager."""
        self._logger.debug("Initializing Redis manager")

        redis_url = self.llm_config.redis_url or "redis://localhost:6379"

        self.redis_manager = RedisManager(
            cluster_mode=False,  # Use sentinel for production
            sentinel_hosts=[
                ("localhost", 26379),
                ("localhost", 26380),
                ("localhost", 26381)
            ],
            master_name="mymaster"
        )

        await self.redis_manager.initialize()
        self._logger.info("Redis manager initialized")

    async def _initialize_postgres(self) -> None:
        """Initialize PostgreSQL manager."""
        self._logger.debug("Initializing PostgreSQL manager")

        self.db_manager = PostgresManager(
            primary_url=self.llm_config.postgres_url,
            replica_urls=[],  # Add replica URLs in production
            pool_size=10
        )

        await self.db_manager.initialize()
        self._logger.info("PostgreSQL manager initialized")

    async def _initialize_llm(self) -> None:
        """Initialize LLM providers and router."""
        self._logger.debug("Initializing LLM infrastructure")

        # Create cost tracker
        self.cost_tracker = CostTracker(
            db_path=Path("./cost_tracking.db")
        )
        await self.cost_tracker.initialize()

        # Create providers
        providers = []

        # Anthropic provider
        if self.llm_config.anthropic_api_key:
            anthropic_breaker = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60.0,
                name="anthropic"
            )
            anthropic_limiter = RateLimiter(
                requests_per_minute=1000,
                tokens_per_minute=100000,
                name="anthropic"
            )

            anthropic_provider = AnthropicProvider(
                api_key=self.llm_config.anthropic_api_key,
                model_id=self.llm_config.default_model,
                circuit_breaker=anthropic_breaker,
                rate_limiter=anthropic_limiter,
                cost_tracker=self.cost_tracker
            )
            await anthropic_provider.initialize()
            providers.append(anthropic_provider)

        # OpenAI provider
        if self.llm_config.openai_api_key:
            openai_breaker = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60.0,
                name="openai"
            )
            openai_limiter = RateLimiter(
                requests_per_minute=10000,
                tokens_per_minute=2000000,
                name="openai"
            )

            openai_provider = OpenAIProvider(
                api_key=self.llm_config.openai_api_key,
                model_id="gpt-4",
                circuit_breaker=openai_breaker,
                rate_limiter=openai_limiter,
                cost_tracker=self.cost_tracker
            )
            await openai_provider.initialize()
            providers.append(openai_provider)

        if not providers:
            raise RuntimeError(
                "No LLM providers configured. Set anthropic_api_key or openai_api_key."
            )

        # Create router
        self.llm_router = LLMRouter(
            providers=providers,
            default_strategy=self.llm_config.llm_routing_strategy
        )

        self._logger.info(
            f"LLM router initialized with {len(providers)} providers, "
            f"strategy={self.llm_config.llm_routing_strategy.value}"
        )

    async def _initialize_cache(self) -> None:
        """Initialize cache manager."""
        self._logger.debug("Initializing cache manager")

        if not self.redis_manager:
            raise RuntimeError("Redis manager required for caching")

        self.cache = CacheManager(
            redis_local=self.redis_manager,
            redis_cluster=self.redis_manager,
            postgres=self.db_manager,
            default_ttl=self.llm_config.cache_ttl_seconds
        )

        await self.cache.initialize()
        self._logger.info("Cache manager initialized")

    async def _initialize_messaging(self) -> None:
        """Initialize message broker."""
        self._logger.debug("Initializing message broker")

        if not self.redis_manager:
            raise RuntimeError("Redis manager required for messaging")

        redis_client = await self.redis_manager.get_client()

        self.message_broker = RedisStreamsBroker(
            redis_client=redis_client,
            consumer_group=f"agent_{self.config.name}",
            consumer_name=self.config.agent_id
        )

        await self.message_broker.initialize()
        self._logger.info("Message broker initialized")

    async def _initialize_rag(self) -> None:
        """Initialize RAG vector store."""
        self._logger.debug("Initializing RAG vector store")

        collection_name = self.llm_config.collection_name or f"{self.config.name}_knowledge"

        self.vector_store = await VectorStoreFactory.create(
            store_type=self.llm_config.vector_store_type,
            collection_name=collection_name,
            embedding_dimension=1536  # OpenAI ada-002 dimension
        )

        await self.vector_store.initialize()
        self._logger.info(
            f"Vector store initialized: type={self.llm_config.vector_store_type.value}, "
            f"collection={collection_name}"
        )

    async def _terminate_core(self) -> None:
        """
        Cleanup infrastructure components.

        Gracefully shuts down all infrastructure connections.
        """
        self._logger.info("Terminating LLM-capable agent infrastructure")

        # Close vector store
        if self.vector_store:
            await self.vector_store.close()
            self._logger.debug("Vector store closed")

        # Close message broker
        if self.message_broker:
            await self.message_broker.close()
            self._logger.debug("Message broker closed")

        # Close cache
        if self.cache:
            await self.cache.close()
            self._logger.debug("Cache manager closed")

        # Close LLM providers
        if self.llm_router:
            for provider in self.llm_router.providers:
                await provider.close()
            self._logger.debug("LLM providers closed")

        # Close cost tracker
        if self.cost_tracker:
            await self.cost_tracker.close()
            self._logger.debug("Cost tracker closed")

        # Close database
        if self.db_manager:
            await self.db_manager.close()
            self._logger.debug("PostgreSQL manager closed")

        # Close Redis
        if self.redis_manager:
            await self.redis_manager.close()
            self._logger.debug("Redis manager closed")

        self._logger.info("LLM-capable agent infrastructure terminated successfully")

    async def _collect_custom_metrics(self) -> Dict[str, Any]:
        """Collect infrastructure metrics."""
        metrics = {}

        # LLM metrics
        if self.llm_router:
            metrics['llm_providers'] = len(self.llm_router.providers)
            metrics['llm_strategy'] = self.llm_config.llm_routing_strategy.value

        # Cache metrics
        if self.cache:
            cache_stats = await self.cache.get_stats()
            metrics['cache_stats'] = cache_stats

        # Cost metrics
        if self.cost_tracker and self.llm_config.tenant_id:
            total_cost = await self.cost_tracker.get_total_cost(
                tenant_id=self.llm_config.tenant_id
            )
            metrics['total_cost_usd'] = total_cost

        # Message broker metrics
        if self.message_broker:
            metrics['message_broker_active'] = True

        # RAG metrics
        if self.vector_store:
            metrics['rag_enabled'] = True
            metrics['vector_store_type'] = self.llm_config.vector_store_type.value

        return metrics

    async def _get_custom_state(self) -> Dict[str, Any]:
        """Get infrastructure state for checkpointing."""
        state = {}

        # Save cost tracker state
        if self.cost_tracker and self.llm_config.tenant_id:
            total_cost = await self.cost_tracker.get_total_cost(
                tenant_id=self.llm_config.tenant_id
            )
            state['total_cost'] = total_cost

        # Save cache stats
        if self.cache:
            cache_stats = await self.cache.get_stats()
            state['cache_stats'] = cache_stats

        return state

    # Helper methods for common operations

    async def llm_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        cache_key: Optional[str] = None
    ) -> str:
        """
        Generate LLM response with automatic caching.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Max tokens (defaults to config)
            temperature: Temperature (defaults to config)
            cache_key: Optional cache key (auto-generated if None)

        Returns:
            Generated text response
        """
        if not self.llm_router:
            raise RuntimeError("LLM not enabled for this agent")

        # Check cache if enabled
        if self.cache and cache_key:
            cached = await self.cache.get(cache_key)
            if cached:
                self._logger.debug(f"Cache hit for key: {cache_key}")
                return cached

        # Generate with LLM
        from llm.models import GenerationRequest

        request = GenerationRequest(
            messages=[{"role": "user", "content": prompt}],
            system=system_prompt,
            max_tokens=max_tokens or self.llm_config.max_tokens,
            temperature=temperature if temperature is not None else self.llm_config.temperature
        )

        response = await self.llm_router.generate(request)

        # Cache if enabled
        if self.cache and cache_key:
            await self.cache.set(
                cache_key,
                response.content,
                ttl=self.llm_config.cache_ttl_seconds
            )

        return response.content

    async def rag_query(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.7
    ) -> list:
        """
        Query RAG vector store for relevant documents.

        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum similarity score

        Returns:
            List of relevant documents
        """
        if not self.vector_store:
            raise RuntimeError("RAG not enabled for this agent")

        results = await self.vector_store.search(
            query_text=query,
            top_k=top_k
        )

        # Filter by score
        filtered = [r for r in results if r.get('score', 0) >= min_score]

        return filtered

    async def send_message(
        self,
        stream_name: str,
        message: Dict[str, Any],
        priority: int = 5
    ) -> str:
        """
        Send message to another agent via message broker.

        Args:
            stream_name: Target stream name
            message: Message payload
            priority: Message priority (1=highest, 10=lowest)

        Returns:
            Message ID
        """
        if not self.message_broker:
            raise RuntimeError("Messaging not enabled for this agent")

        from messaging.message import Message, MessagePriority

        msg = Message(
            type="agent_message",
            payload=message,
            priority=MessagePriority(priority),
            source=self.config.name
        )

        msg_id = await self.message_broker.publish(stream_name, msg)
        self._logger.debug(f"Sent message {msg_id} to stream {stream_name}")

        return msg_id


# Example: LLM-powered ESG analyzer
class ESGAnalyzerAgent(LLMCapableAgent):
    """Example LLM-capable agent for ESG report analysis."""

    async def _execute_core(self, input_data: Any, context: ExecutionContext) -> Any:
        """Analyze ESG report using LLM."""
        self._logger.info("Analyzing ESG report with LLM")

        if not isinstance(input_data, dict) or 'report_text' not in input_data:
            raise ValueError("Input must be dict with 'report_text' key")

        report_text = input_data['report_text']

        # Generate cache key
        import hashlib
        cache_key = f"esg_analysis_{hashlib.sha256(report_text.encode()).hexdigest()}"

        # Analyze with LLM (with caching)
        analysis = await self.llm_generate(
            prompt=f"Analyze this ESG report and extract key metrics:\n\n{report_text}",
            system_prompt="You are an ESG analysis expert. Extract factual data only.",
            cache_key=cache_key
        )

        return {
            'analysis': analysis,
            'report_length': len(report_text),
            'cached': await self.cache.exists(cache_key) if self.cache else False
        }


if __name__ == "__main__":
    # Example usage
    async def main():
        """Test LLMCapableAgent."""
        config = LLMAgentConfig(
            name="test_llm_agent",
            version="1.0.0",
            llm_enabled=True,
            cache_enabled=True,
            rag_enabled=False,
            messaging_enabled=False,
            anthropic_api_key="your-api-key-here"  # Replace with real key
        )

        agent = ESGAnalyzerAgent(config)

        try:
            # Initialize
            await agent.initialize()
            print(f"Agent initialized: {agent}")

            # Execute
            test_input = {
                'report_text': "Sample ESG report with sustainability metrics..."
            }
            result = await agent.execute(test_input)
            print(f"Execution result: {result.dict()}")

        finally:
            # Cleanup
            await agent.terminate()
            print(f"Agent terminated: {agent}")

    asyncio.run(main())
