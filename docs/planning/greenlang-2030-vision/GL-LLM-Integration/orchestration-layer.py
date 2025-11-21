# -*- coding: utf-8 -*-
"""
GreenLang LLM Orchestration Layer
Intelligent routing, caching, and optimization for LLM requests
"""

import asyncio
import hashlib
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
import heapq
import redis
from functools import lru_cache
from greenlang.determinism import DeterministicClock
from greenlang.determinism import FinancialDecimal

class Priority(Enum):
    """Request priority levels"""
    CRITICAL = 1  # Regulatory/compliance
    HIGH = 2      # Production features
    MEDIUM = 3    # Standard operations
    LOW = 4       # Background tasks

@dataclass
class LLMRequest:
    """Enhanced LLM request with routing metadata"""
    id: str
    prompt: str
    task_type: str
    priority: Priority
    customer_id: str
    max_tokens: int
    temperature: float
    deadline: Optional[datetime] = None
    retry_count: int = 0
    preferred_provider: Optional[str] = None
    fallback_providers: List[str] = field(default_factory=list)
    cache_key: Optional[str] = None
    parent_request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMResponse:
    """Enhanced response with performance metrics"""
    request_id: str
    content: str
    provider: str
    model: str
    tokens_used: int
    latency_ms: int
    cost: float
    cached: bool
    confidence: float
    timestamp: datetime

class TokenBucket:
    """Token bucket for rate limiting"""

    def __init__(self, capacity: int, refill_rate: int):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()

    def consume(self, tokens: int) -> bool:
        """Try to consume tokens"""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self):
        """Refill bucket based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now


class LLMOrchestrator:
    """
    Central orchestration layer for all LLM operations
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = self._initialize_providers()
        self.cache = SmartCache(config.get('cache_config', {}))
        self.router = IntelligentRouter(self.providers)
        self.rate_limiter = RateLimiter(config.get('rate_limits', {}))
        self.cost_tracker = CostTracker()
        self.batch_processor = BatchProcessor()
        self.metrics_collector = MetricsCollector()

    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """Main entry point for LLM request processing"""

        start_time = time.time()

        # Step 1: Check cache
        cached_response = await self.cache.get(request)
        if cached_response:
            self.metrics_collector.record_cache_hit(request)
            return cached_response

        # Step 2: Route to optimal provider
        provider, model = await self.router.route(request)

        # Step 3: Check rate limits
        await self.rate_limiter.acquire(provider, request.max_tokens)

        # Step 4: Check if can batch
        if request.priority == Priority.LOW:
            return await self.batch_processor.add(request)

        # Step 5: Execute request
        try:
            response = await self._execute_request(request, provider, model)

            # Step 6: Cache response
            await self.cache.store(request, response)

            # Step 7: Track costs
            self.cost_tracker.record(request, response)

            # Step 8: Collect metrics
            self.metrics_collector.record_request(request, response, time.time() - start_time)

            return response

        except Exception as e:
            # Handle failures with fallback
            return await self._handle_failure(request, provider, e)

    async def _execute_request(self, request: LLMRequest, provider: str, model: str) -> LLMResponse:
        """Execute request with specific provider"""

        provider_client = self.providers[provider]

        start = time.time()
        result = await provider_client.complete(
            prompt=request.prompt,
            model=model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        latency = int((time.time() - start) * 1000)

        return LLMResponse(
            request_id=request.id,
            content=result['content'],
            provider=provider,
            model=model,
            tokens_used=result['tokens'],
            latency_ms=latency,
            cost=self._calculate_cost(provider, model, result['tokens']),
            cached=False,
            confidence=result.get('confidence', 0.8),
            timestamp=DeterministicClock.utcnow()
        )

    async def _handle_failure(self, request: LLMRequest, failed_provider: str, error: Exception) -> LLMResponse:
        """Handle provider failure with fallback logic"""

        self.metrics_collector.record_failure(failed_provider, error)

        # Try fallback providers
        for fallback in request.fallback_providers:
            if fallback != failed_provider:
                try:
                    return await self._execute_request(request, fallback, self.router.get_model(fallback))
                except:
                    continue

        # All providers failed
        raise Exception(f"All providers failed for request {request.id}")

    def _calculate_cost(self, provider: str, model: str, tokens: int) -> float:
        """Calculate cost based on provider pricing"""
        pricing = {
            'anthropic': {'claude-3-sonnet': 0.003},
            'openai': {'gpt-4-turbo': 0.01},
            'google': {'gemini-pro': 0.0035}
        }
        rate = pricing.get(provider, {}).get(model, 0.01)
        return (tokens / 1000) * rate


class SmartCache:
    """
    Multi-tier caching system with semantic similarity
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = redis.Redis(**config.get('redis', {}))
        self.memory_cache = {}  # In-memory L1 cache
        self.semantic_cache = SemanticCache()  # Similarity-based cache
        self.ttl_default = config.get('ttl', 3600)

    async def get(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Retrieve from cache with multiple strategies"""

        # Level 1: Exact match in memory
        cache_key = self._generate_key(request)
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        # Level 2: Exact match in Redis
        redis_result = self.redis_client.get(cache_key)
        if redis_result:
            response = self._deserialize(redis_result)
            self.memory_cache[cache_key] = response  # Promote to L1
            return response

        # Level 3: Semantic similarity
        if request.task_type in ['classification', 'entity_resolution']:
            similar = await self.semantic_cache.find_similar(request.prompt)
            if similar and similar['confidence'] > 0.95:
                return similar['response']

        return None

    async def store(self, request: LLMRequest, response: LLMResponse):
        """Store in multiple cache layers"""

        cache_key = self._generate_key(request)

        # Store in memory (L1)
        self.memory_cache[cache_key] = response

        # Store in Redis (L2)
        self.redis_client.setex(
            cache_key,
            self.ttl_default,
            self._serialize(response)
        )

        # Store in semantic cache if applicable
        if request.task_type in ['classification', 'entity_resolution']:
            await self.semantic_cache.store(request.prompt, response)

    def _generate_key(self, request: LLMRequest) -> str:
        """Generate cache key from request"""
        key_parts = [
            request.task_type,
            request.prompt[:100],  # First 100 chars
            str(request.max_tokens),
            str(request.temperature)
        ]
        key_string = '|'.join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _serialize(self, response: LLMResponse) -> bytes:
        """Serialize response for storage"""
        data = {
            'request_id': response.request_id,
            'content': response.content,
            'provider': response.provider,
            'model': response.model,
            'tokens_used': response.tokens_used,
            'latency_ms': response.latency_ms,
            'cost': response.cost,
            'cached': True,
            'confidence': response.confidence,
            'timestamp': response.timestamp.isoformat()
        }
        return json.dumps(data).encode()

    def _deserialize(self, data: bytes) -> LLMResponse:
        """Deserialize response from storage"""
        obj = json.loads(data.decode())
        obj['timestamp'] = datetime.fromisoformat(obj['timestamp'])
        return LLMResponse(**obj)


class SemanticCache:
    """Semantic similarity-based caching"""

    def __init__(self):
        self.embeddings = {}
        self.responses = {}

    async def find_similar(self, prompt: str, threshold: float = 0.95) -> Optional[Dict]:
        """Find semantically similar cached prompt"""

        prompt_embedding = await self._get_embedding(prompt)

        best_match = None
        best_score = 0

        for cached_prompt, cached_embedding in self.embeddings.items():
            similarity = self._cosine_similarity(prompt_embedding, cached_embedding)
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = cached_prompt

        if best_match:
            return {
                'response': self.responses[best_match],
                'confidence': best_score
            }
        return None

    async def store(self, prompt: str, response: LLMResponse):
        """Store prompt and response with embedding"""
        embedding = await self._get_embedding(prompt)
        self.embeddings[prompt] = embedding
        self.responses[prompt] = response

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text (simplified)"""
        # In production, use actual embedding model
        return [hash(text) % 100 / 100 for _ in range(384)]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5
        return dot_product / (magnitude_a * magnitude_b)


class IntelligentRouter:
    """
    Routes requests to optimal providers based on multiple factors
    """

    def __init__(self, providers: Dict):
        self.providers = providers
        self.performance_history = defaultdict(lambda: {'latency': [], 'success_rate': 1.0})
        self.cost_models = self._initialize_cost_models()

    async def route(self, request: LLMRequest) -> Tuple[str, str]:
        """Select optimal provider and model for request"""

        # Get candidate providers
        candidates = self._get_candidates(request)

        # Score each candidate
        scores = {}
        for provider, model in candidates:
            score = self._score_provider(provider, model, request)
            scores[(provider, model)] = score

        # Select best
        best = max(scores, key=scores.get)
        return best

    def _get_candidates(self, request: LLMRequest) -> List[Tuple[str, str]]:
        """Get list of candidate provider/model pairs"""

        task_mapping = {
            'classification': [('openai', 'gpt-4-turbo'), ('anthropic', 'claude-3-haiku')],
            'entity_resolution': [('openai', 'gpt-4-turbo'), ('google', 'gemini-pro')],
            'materiality_assessment': [('anthropic', 'claude-3-sonnet'), ('openai', 'gpt-4')],
            'narrative_generation': [('anthropic', 'claude-3-sonnet'), ('openai', 'gpt-4-turbo')],
            'document_extraction': [('google', 'gemini-pro'), ('openai', 'gpt-4-vision')],
            'code_generation': [('anthropic', 'claude-3-sonnet'), ('openai', 'gpt-4-turbo')]
        }

        return task_mapping.get(request.task_type, [('openai', 'gpt-4-turbo')])

    def _score_provider(self, provider: str, model: str, request: LLMRequest) -> float:
        """Score provider based on multiple factors"""

        score = 0.0

        # Factor 1: Historical performance (40%)
        perf = self.performance_history[provider]
        avg_latency = sum(perf['latency'][-10:]) / max(len(perf['latency'][-10:]), 1)
        latency_score = max(0, 1 - (avg_latency / 5000))  # Normalize to 0-1
        score += latency_score * 0.4

        # Factor 2: Success rate (30%)
        score += perf['success_rate'] * 0.3

        # Factor 3: Cost (20%)
        cost = self.cost_models[provider][model]
        cost_score = max(0, 1 - (cost / 0.05))  # Normalize assuming $0.05/1k tokens is max
        score += cost_score * 0.2

        # Factor 4: Priority alignment (10%)
        if request.priority == Priority.CRITICAL and provider == 'anthropic':
            score += 0.1
        elif request.priority == Priority.LOW and 'llama' in model:
            score += 0.1

        return score


class RateLimiter:
    """
    Provider-specific rate limiting
    """

    def __init__(self, limits: Dict[str, Dict[str, int]]):
        self.limits = limits
        self.buckets = {}
        self._initialize_buckets()

    def _initialize_buckets(self):
        """Initialize token buckets for each provider"""
        for provider, limits in self.limits.items():
            self.buckets[provider] = {
                'rpm': TokenBucket(limits['rpm'], limits['rpm'] / 60),
                'tpm': TokenBucket(limits['tpm'], limits['tpm'] / 60)
            }

    async def acquire(self, provider: str, tokens: int):
        """Acquire permission to make request"""

        if provider not in self.buckets:
            return  # No limits configured

        # Wait for both request and token allowance
        while True:
            if (self.buckets[provider]['rpm'].consume(1) and
                self.buckets[provider]['tpm'].consume(tokens)):
                return
            await asyncio.sleep(0.1)  # Wait 100ms and retry


class BatchProcessor:
    """
    Batch similar requests for efficiency
    """

    def __init__(self, batch_size: int = 10, wait_time: int = 5):
        self.batch_size = batch_size
        self.wait_time = wait_time
        self.queues = defaultdict(list)
        self.processing = False

    async def add(self, request: LLMRequest) -> LLMResponse:
        """Add request to batch queue"""

        queue_key = f"{request.task_type}_{request.provider}_{request.model}"
        queue = self.queues[queue_key]

        # Add to queue
        future = asyncio.Future()
        queue.append((request, future))

        # Start processing if needed
        if not self.processing:
            asyncio.create_task(self._process_batches())

        # Wait for result
        return await future

    async def _process_batches(self):
        """Process batched requests"""
        self.processing = True

        while any(self.queues.values()):
            for queue_key, queue in list(self.queues.items()):
                if len(queue) >= self.batch_size or time.time() - queue[0][0].timestamp > self.wait_time:
                    batch = queue[:self.batch_size]
                    self.queues[queue_key] = queue[self.batch_size:]

                    # Process batch
                    results = await self._execute_batch(batch)

                    # Resolve futures
                    for (request, future), result in zip(batch, results):
                        future.set_result(result)

            await asyncio.sleep(0.1)

        self.processing = False

    async def _execute_batch(self, batch: List[Tuple[LLMRequest, asyncio.Future]]) -> List[LLMResponse]:
        """Execute batch of requests together"""
        # Implementation would combine prompts and execute together
        # This is a simplified version
        results = []
        for request, _ in batch:
            # In reality, would combine into single API call
            result = LLMResponse(
                request_id=request.id,
                content=f"Batched response for {request.id}",
                provider='batch',
                model='batch',
                tokens_used=100,
                latency_ms=100,
                cost=0.001,
                cached=False,
                confidence=0.85,
                timestamp=DeterministicClock.utcnow()
            )
            results.append(result)
        return results


class CostTracker:
    """
    Track and allocate costs by customer
    """

    def __init__(self):
        self.costs = defaultdict(lambda: defaultdict(float))
        self.budgets = {}
        self.alerts = []

    def record(self, request: LLMRequest, response: LLMResponse):
        """Record cost for request"""

        customer = request.customer_id
        self.costs[customer]['total'] += response.cost
        self.costs[customer][request.task_type] += response.cost
        self.costs[customer][response.provider] += response.cost

        # Check budget
        if customer in self.budgets:
            if self.costs[customer]['total'] > self.budgets[customer]:
                self.alerts.append({
                    'customer': customer,
                    'spent': self.costs[customer]['total'],
                    'budget': self.budgets[customer],
                    'timestamp': DeterministicClock.utcnow()
                })

    def set_budget(self, customer: str, budget: float):
        """Set customer budget"""
        self.budgets[customer] = budget

    def get_report(self, customer: str, period: str = 'daily') -> Dict:
        """Get cost report for customer"""
        return {
            'customer': customer,
            'period': period,
            'total': self.costs[customer]['total'],
            'by_task': dict(self.costs[customer]),
            'budget_remaining': self.budgets.get(customer, FinancialDecimal.from_string('inf')) - self.costs[customer]['total']
        }


class MetricsCollector:
    """
    Collect and aggregate performance metrics
    """

    def __init__(self):
        self.metrics = defaultdict(list)
        self.aggregates = {}

    def record_request(self, request: LLMRequest, response: LLMResponse, total_time: float):
        """Record metrics for request"""

        self.metrics['latency'].append({
            'provider': response.provider,
            'model': response.model,
            'task_type': request.task_type,
            'latency_ms': response.latency_ms,
            'total_time_ms': total_time * 1000,
            'timestamp': response.timestamp
        })

        self.metrics['tokens'].append({
            'provider': response.provider,
            'model': response.model,
            'tokens': response.tokens_used,
            'timestamp': response.timestamp
        })

        self.metrics['cost'].append({
            'customer': request.customer_id,
            'cost': response.cost,
            'timestamp': response.timestamp
        })

    def record_cache_hit(self, request: LLMRequest):
        """Record cache hit"""
        self.metrics['cache_hits'].append({
            'task_type': request.task_type,
            'timestamp': DeterministicClock.utcnow()
        })

    def record_failure(self, provider: str, error: Exception):
        """Record provider failure"""
        self.metrics['failures'].append({
            'provider': provider,
            'error': str(error),
            'timestamp': DeterministicClock.utcnow()
        })

    def get_summary(self, window: timedelta = timedelta(hours=1)) -> Dict:
        """Get metrics summary for time window"""

        cutoff = DeterministicClock.utcnow() - window

        # Filter metrics within window
        recent_latency = [m for m in self.metrics['latency'] if m['timestamp'] > cutoff]
        recent_cache = [m for m in self.metrics['cache_hits'] if m['timestamp'] > cutoff]

        return {
            'avg_latency_ms': sum(m['latency_ms'] for m in recent_latency) / max(len(recent_latency), 1),
            'cache_hit_rate': len(recent_cache) / max(len(recent_latency) + len(recent_cache), 1),
            'total_requests': len(recent_latency),
            'total_cost': sum(m['cost'] for m in self.metrics['cost'] if m['timestamp'] > cutoff),
            'failures': len([f for f in self.metrics['failures'] if f['timestamp'] > cutoff])
        }


# Export main components
__all__ = [
    'LLMOrchestrator',
    'LLMRequest',
    'LLMResponse',
    'Priority',
    'SmartCache',
    'IntelligentRouter',
    'RateLimiter',
    'CostTracker',
    'MetricsCollector'
]