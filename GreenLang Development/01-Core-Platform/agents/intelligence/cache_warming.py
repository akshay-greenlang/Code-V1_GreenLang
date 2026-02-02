# -*- coding: utf-8 -*-
"""
Cache Warming for Semantic Cache

Pre-populates semantic cache with common queries to improve hit rates:
- Common climate/carbon queries
- High-frequency agent interactions
- Background refresh jobs
- Query frequency tracking

Benefits:
- Immediate cache hits for common queries
- Reduced cold-start latency
- Better user experience
- Optimized cost savings

Architecture:
    Application Startup -> Load Common Queries -> Generate Responses -> Cache
                              |
                        Background Job (hourly/daily)
                              |
                         Refresh Cache
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Coroutine, Dict, List, Optional

from greenlang.agents.intelligence.semantic_cache import SemanticCache, get_global_cache
from greenlang.utilities.determinism import DeterministicClock


logger = logging.getLogger(__name__)


@dataclass
class WarmingQuery:
    """
    Query for cache warming

    Attributes:
        prompt: Query prompt
        expected_response: Expected/template response
        metadata: Query metadata (model, temperature, etc.)
        agent_id: Agent ID
        priority: Priority (0-10, higher = more important)
        frequency: Expected query frequency (queries per day)
    """
    prompt: str
    expected_response: str
    metadata: Dict[str, Any]
    agent_id: str
    priority: int = 5
    frequency: float = 1.0  # queries per day


# Pre-defined common queries for climate/carbon calculations
COMMON_QUERIES = [
    WarmingQuery(
        prompt="What is the carbon footprint of natural gas?",
        expected_response=(
            "Natural gas has a carbon footprint of approximately 0.185 kg CO2 per kWh. "
            "This includes both combustion emissions and upstream methane leakage. "
            "For a typical residential heating system, this translates to about 5.3 tonnes "
            "CO2 per year for average consumption."
        ),
        metadata={"model": "gpt-4", "temperature": 0.0},
        agent_id="carbon_calc",
        priority=9,
        frequency=10.0,
    ),
    WarmingQuery(
        prompt="Calculate emissions for 1000 kWh electricity",
        expected_response=(
            "For 1000 kWh of electricity, emissions depend on the grid mix:\n"
            "- US Average Grid: ~400 kg CO2\n"
            "- Coal-heavy Grid: ~900 kg CO2\n"
            "- Renewable Grid: ~50 kg CO2\n"
            "Using the US average grid intensity of 0.4 kg CO2/kWh, "
            "1000 kWh produces approximately 400 kg CO2."
        ),
        metadata={"model": "gpt-4", "temperature": 0.0},
        agent_id="emission_calc",
        priority=8,
        frequency=8.0,
    ),
    WarmingQuery(
        prompt="Recommend boiler replacement options",
        expected_response=(
            "Based on efficiency and emissions, here are boiler replacement options:\n\n"
            "1. High-Efficiency Condensing Boiler (Gas)\n"
            "   - Efficiency: 92-95%\n"
            "   - Cost: $3,000-$5,000\n"
            "   - Emissions: 0.185 kg CO2/kWh\n\n"
            "2. Air-Source Heat Pump\n"
            "   - COP: 3.0-4.0\n"
            "   - Cost: $8,000-$12,000\n"
            "   - Emissions: 0.13 kg CO2/kWh (grid average)\n\n"
            "3. Ground-Source Heat Pump\n"
            "   - COP: 4.0-5.0\n"
            "   - Cost: $15,000-$25,000\n"
            "   - Emissions: 0.10 kg CO2/kWh\n\n"
            "Recommendation: For most homes, an air-source heat pump provides the best "
            "balance of cost, efficiency, and emissions reduction."
        ),
        metadata={"model": "gpt-4", "temperature": 0.3},
        agent_id="recommendation",
        priority=7,
        frequency=5.0,
    ),
    WarmingQuery(
        prompt="What is grid carbon intensity in California?",
        expected_response=(
            "California grid carbon intensity:\n"
            "- Average: 0.240 kg CO2/kWh (2023)\n"
            "- Daytime (solar peak): 0.150 kg CO2/kWh\n"
            "- Evening (gas peak): 0.350 kg CO2/kWh\n"
            "- Renewable percentage: ~60%\n\n"
            "California has one of the cleanest grids in the US due to high renewable "
            "penetration (solar, wind, hydro) and aggressive climate policies."
        ),
        metadata={"model": "gpt-4", "temperature": 0.0},
        agent_id="grid_data",
        priority=7,
        frequency=6.0,
    ),
    WarmingQuery(
        prompt="How to reduce home energy consumption?",
        expected_response=(
            "Top strategies to reduce home energy consumption:\n\n"
            "1. Insulation & Air Sealing (30-40% savings)\n"
            "   - Attic insulation (R-38 to R-60)\n"
            "   - Air seal windows, doors, penetrations\n"
            "   - Estimated savings: $400-$600/year\n\n"
            "2. HVAC Upgrade (20-30% savings)\n"
            "   - High-efficiency heat pump (SEER 16+)\n"
            "   - Smart thermostat\n"
            "   - Estimated savings: $300-$500/year\n\n"
            "3. LED Lighting (5-10% savings)\n"
            "   - Replace all incandescent/CFL bulbs\n"
            "   - Estimated savings: $75-$150/year\n\n"
            "4. Energy-Efficient Appliances (10-15% savings)\n"
            "   - ENERGY STAR refrigerator, dishwasher, washer\n"
            "   - Estimated savings: $150-$250/year\n\n"
            "Total potential savings: $925-$1,500/year"
        ),
        metadata={"model": "gpt-4", "temperature": 0.3},
        agent_id="efficiency",
        priority=6,
        frequency=4.0,
    ),
    WarmingQuery(
        prompt="Compare emissions: electric vs gas vehicle",
        expected_response=(
            "Emissions comparison (annual, 12,000 miles):\n\n"
            "Electric Vehicle (EV):\n"
            "- Electricity consumption: ~3,600 kWh/year\n"
            "- Emissions (US grid): 1,440 kg CO2/year\n"
            "- Emissions (clean grid): 180 kg CO2/year\n"
            "- Operating cost: $450/year (@ $0.125/kWh)\n\n"
            "Gas Vehicle (30 MPG):\n"
            "- Gas consumption: 400 gallons/year\n"
            "- Emissions: 3,600 kg CO2/year\n"
            "- Operating cost: $1,400/year (@ $3.50/gallon)\n\n"
            "Result: EV produces 60% less emissions on average grid, "
            "95% less on clean grid. Savings: 2,160 kg CO2/year."
        ),
        metadata={"model": "gpt-4", "temperature": 0.0},
        agent_id="transport",
        priority=6,
        frequency=3.0,
    ),
    WarmingQuery(
        prompt="What is embodied carbon in buildings?",
        expected_response=(
            "Embodied carbon is the total greenhouse gas emissions from manufacturing, "
            "transporting, and installing building materials, plus end-of-life disposal.\n\n"
            "Typical embodied carbon by material:\n"
            "- Concrete: 150-200 kg CO2/m³\n"
            "- Steel: 1,800-2,000 kg CO2/tonne\n"
            "- Timber: 100-200 kg CO2/m³ (carbon stored)\n"
            "- Aluminum: 8,000-10,000 kg CO2/tonne\n\n"
            "For a typical residential building:\n"
            "- Embodied carbon: 300-500 kg CO2/m²\n"
            "- Operational carbon: 20-40 kg CO2/m²/year\n\n"
            "Embodied carbon represents ~20% of total lifecycle emissions."
        ),
        metadata={"model": "gpt-4", "temperature": 0.0},
        agent_id="building_analysis",
        priority=5,
        frequency=2.0,
    ),
    WarmingQuery(
        prompt="Calculate solar panel ROI and emissions savings",
        expected_response=(
            "Solar panel ROI analysis (6 kW residential system):\n\n"
            "System Cost: $18,000 (before incentives)\n"
            "Federal Tax Credit (30%): -$5,400\n"
            "Net Cost: $12,600\n\n"
            "Annual Production: 8,000 kWh\n"
            "Annual Savings: $1,000 (@ $0.125/kWh)\n"
            "Payback Period: 12.6 years\n"
            "25-year ROI: 98% ($24,900 total savings)\n\n"
            "Emissions Savings:\n"
            "- Annual: 3,200 kg CO2 (@ 0.4 kg/kWh grid)\n"
            "- 25-year: 80,000 kg CO2 (80 tonnes)\n"
            "- Equivalent: Taking 17 gas cars off the road for 1 year"
        ),
        metadata={"model": "gpt-4", "temperature": 0.0},
        agent_id="solar_analysis",
        priority=5,
        frequency=2.5,
    ),
    WarmingQuery(
        prompt="What are Scope 1, 2, and 3 emissions?",
        expected_response=(
            "Corporate emissions scopes:\n\n"
            "Scope 1 - Direct Emissions:\n"
            "- Company-owned vehicles\n"
            "- On-site fuel combustion\n"
            "- Industrial processes\n"
            "- Example: Factory natural gas boiler\n\n"
            "Scope 2 - Indirect Emissions (Energy):\n"
            "- Purchased electricity\n"
            "- Purchased heating/cooling\n"
            "- Example: Office building electricity\n\n"
            "Scope 3 - Indirect Emissions (Value Chain):\n"
            "- Supplier emissions\n"
            "- Employee commuting\n"
            "- Product use & disposal\n"
            "- Business travel\n"
            "- Example: Manufacturing supply chain\n\n"
            "Typical distribution: Scope 3 > 70% of total emissions"
        ),
        metadata={"model": "gpt-4", "temperature": 0.0},
        agent_id="corporate_reporting",
        priority=4,
        frequency=1.5,
    ),
    WarmingQuery(
        prompt="How does carbon offsetting work?",
        expected_response=(
            "Carbon offsetting allows organizations to compensate for emissions by funding "
            "projects that reduce or remove CO2 from the atmosphere.\n\n"
            "Common offset types:\n"
            "1. Renewable Energy Projects\n"
            "   - Wind, solar farms in developing countries\n"
            "   - Cost: $5-15 per tonne CO2\n\n"
            "2. Forestry & Reforestation\n"
            "   - Tree planting, forest protection\n"
            "   - Cost: $10-30 per tonne CO2\n\n"
            "3. Direct Air Capture\n"
            "   - Technology removes CO2 from air\n"
            "   - Cost: $100-600 per tonne CO2\n\n"
            "4. Methane Capture\n"
            "   - Landfill gas, agricultural methane\n"
            "   - Cost: $5-20 per tonne CO2e\n\n"
            "Best practices: Choose verified projects (Gold Standard, Verra), "
            "prioritize permanent removal, focus on reduction first."
        ),
        metadata={"model": "gpt-4", "temperature": 0.3},
        agent_id="offset_advisor",
        priority=4,
        frequency=1.0,
    ),
]


@dataclass
class QueryStats:
    """
    Track query frequency for cache warming

    Attributes:
        prompt: Query prompt
        count: Number of times queried
        last_seen: Last query timestamp
        agent_id: Agent ID
    """
    prompt: str
    count: int = 0
    last_seen: Optional[datetime] = None
    agent_id: Optional[str] = None


class CacheWarmer:
    """
    Cache warming manager

    Handles:
    - Pre-population with common queries
    - Background refresh jobs
    - Query frequency tracking
    - Smart warming based on usage patterns
    """

    def __init__(
        self,
        cache: Optional[SemanticCache] = None,
        llm_callback: Optional[Callable[[str, Dict[str, Any]], Coroutine[Any, Any, str]]] = None,
    ):
        """
        Initialize cache warmer

        Args:
            cache: Semantic cache instance (default: global cache)
            llm_callback: Async function to generate responses: (prompt, metadata) -> response
        """
        self.cache = cache or get_global_cache()
        self.llm_callback = llm_callback

        # Query frequency tracking
        self.query_stats: Dict[str, QueryStats] = {}

        logger.info("CacheWarmer initialized")

    def warm_with_queries(
        self,
        queries: List[WarmingQuery],
        use_llm: bool = False,
    ):
        """
        Warm cache with pre-defined queries

        Args:
            queries: List of warming queries
            use_llm: If True, use LLM callback to generate responses (costs $)
                     If False, use expected_response from query (free)
        """
        logger.info(f"Warming cache with {len(queries)} queries...")

        for query in queries:
            # Use expected response (free) or generate with LLM (costs $)
            if use_llm and self.llm_callback:
                # Skip LLM generation in sync context
                logger.warning("LLM callback requires async context. Using expected response.")
                response = query.expected_response
            else:
                response = query.expected_response

            # Add to cache
            self.cache.set(
                prompt=query.prompt,
                response=response,
                metadata=query.metadata,
                agent_id=query.agent_id,
            )

            logger.debug(f"Cached: {query.prompt[:60]}...")

        logger.info(f"Cache warming complete. Cache size: {self.cache.faiss_index.size}")

    async def warm_with_queries_async(
        self,
        queries: List[WarmingQuery],
        use_llm: bool = False,
    ):
        """
        Warm cache with pre-defined queries (async)

        Args:
            queries: List of warming queries
            use_llm: If True, use LLM callback to generate responses
        """
        logger.info(f"Warming cache with {len(queries)} queries (async)...")

        for query in queries:
            # Use LLM or expected response
            if use_llm and self.llm_callback:
                try:
                    response = await self.llm_callback(query.prompt, query.metadata)
                except Exception as e:
                    logger.error(f"LLM callback failed: {e}. Using expected response.")
                    response = query.expected_response
            else:
                response = query.expected_response

            # Add to cache
            self.cache.set(
                prompt=query.prompt,
                response=response,
                metadata=query.metadata,
                agent_id=query.agent_id,
            )

            logger.debug(f"Cached: {query.prompt[:60]}...")

        logger.info(f"Cache warming complete. Cache size: {self.cache.faiss_index.size}")

    def warm_common_queries(self, use_llm: bool = False):
        """
        Warm cache with common climate/carbon queries

        Args:
            use_llm: If True, use LLM to generate responses (costs $)
        """
        self.warm_with_queries(COMMON_QUERIES, use_llm=use_llm)

    async def warm_common_queries_async(self, use_llm: bool = False):
        """
        Warm cache with common climate/carbon queries (async)

        Args:
            use_llm: If True, use LLM to generate responses
        """
        await self.warm_with_queries_async(COMMON_QUERIES, use_llm=use_llm)

    def track_query(self, prompt: str, agent_id: Optional[str] = None):
        """
        Track query frequency for warming candidates

        Args:
            prompt: Query prompt
            agent_id: Agent ID
        """
        if prompt not in self.query_stats:
            self.query_stats[prompt] = QueryStats(
                prompt=prompt,
                agent_id=agent_id,
            )

        stats = self.query_stats[prompt]
        stats.count += 1
        stats.last_seen = DeterministicClock.now()

    def get_top_queries(self, limit: int = 10) -> List[QueryStats]:
        """
        Get top queries by frequency

        Args:
            limit: Maximum number to return

        Returns:
            List of top query statistics
        """
        sorted_queries = sorted(
            self.query_stats.values(),
            key=lambda q: q.count,
            reverse=True,
        )
        return sorted_queries[:limit]

    def get_warming_candidates(
        self,
        min_frequency: int = 3,
        time_window_hours: int = 24,
    ) -> List[str]:
        """
        Get queries that should be added to warming set

        Args:
            min_frequency: Minimum query count to be candidate
            time_window_hours: Time window for frequency calculation

        Returns:
            List of query prompts
        """
        cutoff = DeterministicClock.now() - timedelta(hours=time_window_hours)

        candidates = []
        for stats in self.query_stats.values():
            if stats.count >= min_frequency and stats.last_seen and stats.last_seen >= cutoff:
                candidates.append(stats.prompt)

        return candidates

    async def background_refresh(self, interval_seconds: int = 3600):
        """
        Background job to refresh cache periodically

        Args:
            interval_seconds: Refresh interval (default: 1 hour)
        """
        logger.info(f"Starting background refresh (interval: {interval_seconds}s)")

        while True:
            try:
                await asyncio.sleep(interval_seconds)

                logger.info("Refreshing cache...")

                # Refresh common queries
                await self.warm_common_queries_async(use_llm=False)

                logger.info("Cache refresh complete")

            except Exception as e:
                logger.error(f"Background refresh failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache warming statistics"""
        return {
            "total_tracked_queries": len(self.query_stats),
            "top_queries": [
                {"prompt": q.prompt[:60], "count": q.count, "agent_id": q.agent_id}
                for q in self.get_top_queries(limit=5)
            ],
            "warming_candidates": len(self.get_warming_candidates()),
        }


def warm_cache_on_startup(cache: Optional[SemanticCache] = None):
    """
    Warm cache on application startup

    This function should be called during application initialization
    to pre-populate the cache with common queries.

    Args:
        cache: Semantic cache instance (default: global cache)
    """
    logger.info("Warming cache on startup...")

    warmer = CacheWarmer(cache=cache)
    warmer.warm_common_queries(use_llm=False)

    logger.info("Startup cache warming complete")


if __name__ == "__main__":
    """
    Demo and testing
    """
    print("=" * 80)
    print("GreenLang Cache Warming Demo")
    print("=" * 80)

    # Initialize cache warmer
    cache = SemanticCache()
    warmer = CacheWarmer(cache=cache)

    # Warm cache with common queries
    print("\n1. Warming cache with common queries...")
    warmer.warm_common_queries(use_llm=False)

    # Show cache stats
    print("\n2. Cache statistics:")
    cache_stats = cache.get_stats()
    for key, value in cache_stats.items():
        print(f"   {key}: {value}")

    # Test cache hits
    print("\n3. Testing cache hits...")
    test_queries = [
        "What is the carbon footprint of natural gas?",
        "What's the CO2 emissions from natural gas?",
        "Tell me about natural gas emissions",
    ]

    for query in test_queries:
        result = cache.get(query, similarity_threshold=0.85)
        if result:
            response, similarity, entry = result
            print(f"\n   Query: {query}")
            print(f"   Similarity: {similarity:.3f}")
            print(f"   Hit count: {entry.hit_count}")
        else:
            print(f"\n   Query: {query}")
            print(f"   Result: Cache miss")

    # Track queries
    print("\n4. Tracking query frequency...")
    warmer.track_query("What is grid carbon intensity in California?")
    warmer.track_query("What is grid carbon intensity in California?")
    warmer.track_query("What is grid carbon intensity in California?")
    warmer.track_query("How to reduce home energy?")

    # Show warming stats
    print("\n5. Warming statistics:")
    warming_stats = warmer.get_stats()
    for key, value in warming_stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 80)
