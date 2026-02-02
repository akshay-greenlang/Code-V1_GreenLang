# -*- coding: utf-8 -*-
"""
RAG Retrieval Quality Benchmarking
GL Intelligence Infrastructure

Measures retrieval quality using standard IR metrics:
- NDCG@K (Normalized Discounted Cumulative Gain)
- Precision@K
- Recall@K
- MRR (Mean Reciprocal Rank)

Version: 1.0.0
Date: 2025-11-06
"""

import pytest
import asyncio
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import math
from collections import defaultdict


class RetrievalMetrics:
    """Calculate retrieval quality metrics."""

    @staticmethod
    def dcg_at_k(relevances: List[float], k: int) -> float:
        """
        Calculate Discounted Cumulative Gain at K.

        DCG@K = Σ(rel_i / log2(i+1)) for i=1 to k

        Args:
            relevances: List of relevance scores (0-1) in retrieval order
            k: Cut-off rank

        Returns:
            DCG score
        """
        relevances = relevances[:k]
        return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))

    @staticmethod
    def ndcg_at_k(relevances: List[float], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.

        NDCG@K = DCG@K / IDCG@K
        where IDCG = DCG of perfect ranking

        Args:
            relevances: List of relevance scores in retrieval order
            k: Cut-off rank

        Returns:
            NDCG score (0-1)
        """
        dcg = RetrievalMetrics.dcg_at_k(relevances, k)

        # Ideal DCG: Sort relevances descending
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = RetrievalMetrics.dcg_at_k(ideal_relevances, k)

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def precision_at_k(relevances: List[float], k: int, threshold: float = 0.5) -> float:
        """
        Calculate Precision at K.

        Precision@K = (# relevant docs in top K) / K

        Args:
            relevances: List of relevance scores in retrieval order
            k: Cut-off rank
            threshold: Relevance threshold (default: 0.5)

        Returns:
            Precision score (0-1)
        """
        relevances = relevances[:k]
        relevant_count = sum(1 for rel in relevances if rel >= threshold)
        return relevant_count / k if k > 0 else 0.0

    @staticmethod
    def recall_at_k(
        relevances: List[float],
        total_relevant: int,
        k: int,
        threshold: float = 0.5
    ) -> float:
        """
        Calculate Recall at K.

        Recall@K = (# relevant docs in top K) / (total # relevant docs)

        Args:
            relevances: List of relevance scores in retrieval order
            total_relevant: Total number of relevant documents
            k: Cut-off rank
            threshold: Relevance threshold

        Returns:
            Recall score (0-1)
        """
        relevances = relevances[:k]
        relevant_count = sum(1 for rel in relevances if rel >= threshold)
        return relevant_count / total_relevant if total_relevant > 0 else 0.0

    @staticmethod
    def mean_reciprocal_rank(relevances: List[float], threshold: float = 0.5) -> float:
        """
        Calculate Mean Reciprocal Rank.

        MRR = 1 / rank of first relevant document

        Args:
            relevances: List of relevance scores in retrieval order
            threshold: Relevance threshold

        Returns:
            MRR score (0-1)
        """
        for idx, rel in enumerate(relevances):
            if rel >= threshold:
                return 1.0 / (idx + 1)
        return 0.0


# Benchmark dataset
BENCHMARK_QUERIES = [
    {
        "id": "q1",
        "query": "What is the emission factor for natural gas combustion?",
        "expected_answers": [
            "0.0531 kg CO2e/kWh",
            "53.06 kg CO2e/MMBtu",
            "natural gas combustion"
        ],
        "collection": "ghg_protocol_corp",
        "difficulty": "easy"
    },
    {
        "id": "q2",
        "query": "Explain the difference between Scope 1, Scope 2, and Scope 3 emissions",
        "expected_answers": [
            "Scope 1: Direct emissions",
            "Scope 2: Indirect electricity",
            "Scope 3: Value chain"
        ],
        "collection": "ghg_protocol_corp",
        "difficulty": "medium"
    },
    {
        "id": "q3",
        "query": "What is the COP range for industrial heat pumps at different temperatures?",
        "expected_answers": [
            "COP 3.5-4.5 low temperature",
            "COP 2.5-3.5 medium temperature",
            "COP 2.0-3.0 high temperature"
        ],
        "collection": "technology_database",
        "difficulty": "medium"
    },
    {
        "id": "q4",
        "query": "What is the typical payback period for solar thermal systems in industrial applications?",
        "expected_answers": [
            "5-10 years",
            "payback period",
            "solar thermal"
        ],
        "collection": "technology_database",
        "difficulty": "easy"
    },
    {
        "id": "q5",
        "query": "How much CO2 reduction was achieved in the food processing plant heat pump case study?",
        "expected_answers": [
            "520 tons CO2/year",
            "65% reduction",
            "food processing"
        ],
        "collection": "case_studies",
        "difficulty": "medium"
    },
    {
        "id": "q6",
        "query": "What were the key success factors for the steel mill waste heat recovery project?",
        "expected_answers": [
            "consistent high-grade waste heat",
            "high electricity prices",
            "ORC system"
        ],
        "collection": "case_studies",
        "difficulty": "hard"
    },
    {
        "id": "q7",
        "query": "Compare emission factors for coal vs natural gas combustion",
        "expected_answers": [
            "coal: 0.341 kg CO2e/kWh",
            "natural gas: 0.0531 kg CO2e/kWh",
            "coal is 6x higher"
        ],
        "collection": "ghg_protocol_corp",
        "difficulty": "medium"
    },
    {
        "id": "q8",
        "query": "What are the best applications for industrial heat pumps?",
        "expected_answers": [
            "food and beverage",
            "chemical manufacturing",
            "pharmaceutical"
        ],
        "collection": "technology_database",
        "difficulty": "easy"
    },
    {
        "id": "q9",
        "query": "What is the overall efficiency of Combined Heat and Power systems?",
        "expected_answers": [
            "65-85% overall",
            "CHP",
            "cogeneration"
        ],
        "collection": "technology_database",
        "difficulty": "easy"
    },
    {
        "id": "q10",
        "query": "What location considerations are important for solar thermal installations?",
        "expected_answers": [
            "1,800 kWh/m²/year solar irradiance",
            "sun belt regions",
            "shading analysis"
        ],
        "collection": "technology_database",
        "difficulty": "medium"
    }
]


class RAGBenchmark:
    """RAG retrieval quality benchmark."""

    def __init__(self, rag_engine):
        """
        Initialize benchmark.

        Args:
            rag_engine: RAGEngine instance to benchmark
        """
        self.engine = rag_engine
        self.results = []

    async def run_benchmark(
        self,
        queries: List[Dict[str, Any]] = BENCHMARK_QUERIES,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Run full benchmark suite.

        Args:
            queries: List of benchmark queries
            top_k: Number of results to retrieve

        Returns:
            Benchmark results with metrics
        """
        print(f"\n{'='*80}")
        print(f"RUNNING RAG BENCHMARK ({len(queries)} queries)")
        print(f"{'='*80}\n")

        results = []

        for query_data in queries:
            result = await self._benchmark_query(query_data, top_k)
            results.append(result)

            # Print per-query results
            print(f"\nQuery {result['query_id']}: {result['query'][:60]}...")
            print(f"  NDCG@{top_k}: {result['ndcg']:.3f}")
            print(f"  Precision@{top_k}: {result['precision']:.3f}")
            print(f"  MRR: {result['mrr']:.3f}")
            print(f"  Difficulty: {result['difficulty']}")

        # Aggregate metrics
        aggregate = self._aggregate_results(results, top_k)

        print(f"\n{'='*80}")
        print("AGGREGATE METRICS")
        print(f"{'='*80}")
        print(f"\nNDCG@{top_k}: {aggregate['ndcg_mean']:.3f} (±{aggregate['ndcg_std']:.3f})")
        print(f"Precision@{top_k}: {aggregate['precision_mean']:.3f} (±{aggregate['precision_std']:.3f})")
        print(f"Recall@{top_k}: {aggregate['recall_mean']:.3f} (±{aggregate['recall_std']:.3f})")
        print(f"MRR: {aggregate['mrr_mean']:.3f} (±{aggregate['mrr_std']:.3f})")

        print(f"\nBy Difficulty:")
        for difficulty, metrics in aggregate['by_difficulty'].items():
            print(f"  {difficulty.capitalize()}: NDCG={metrics['ndcg']:.3f}, Precision={metrics['precision']:.3f}")

        # Quality assessment
        self._print_quality_assessment(aggregate, top_k)

        return {
            "per_query_results": results,
            "aggregate_metrics": aggregate,
            "timestamp": asyncio.get_event_loop().time()
        }

    async def _benchmark_query(
        self,
        query_data: Dict[str, Any],
        top_k: int
    ) -> Dict[str, Any]:
        """
        Benchmark a single query.

        Args:
            query_data: Query metadata and expected answers
            top_k: Number of results to retrieve

        Returns:
            Query results with metrics
        """
        query = query_data["query"]

        # Retrieve documents
        result = await self.engine.query(
            query=query,
            top_k=top_k,
            collections=self.engine.config.allowlist
        )

        # Calculate relevance scores by keyword matching
        relevances = self._calculate_relevances(
            result.chunks,
            query_data["expected_answers"]
        )

        # Calculate metrics
        total_relevant = len(query_data["expected_answers"])

        return {
            "query_id": query_data["id"],
            "query": query,
            "collection": query_data["collection"],
            "difficulty": query_data["difficulty"],
            "chunks_retrieved": len(result.chunks),
            "search_time_ms": result.search_time_ms,
            "relevances": relevances,
            "ndcg": RetrievalMetrics.ndcg_at_k(relevances, top_k),
            "precision": RetrievalMetrics.precision_at_k(relevances, top_k),
            "recall": RetrievalMetrics.recall_at_k(relevances, total_relevant, top_k),
            "mrr": RetrievalMetrics.mean_reciprocal_rank(relevances)
        }

    def _calculate_relevances(
        self,
        chunks: List[Any],
        expected_answers: List[str]
    ) -> List[float]:
        """
        Calculate relevance scores for retrieved chunks.

        Uses keyword matching as proxy for relevance.
        In production, this would use human annotation.

        Args:
            chunks: Retrieved chunks
            expected_answers: List of expected answer keywords

        Returns:
            List of relevance scores (0-1)
        """
        relevances = []

        for chunk in chunks:
            chunk_text = chunk.text.lower()

            # Count matching keywords
            matches = sum(
                1 for keyword in expected_answers
                if keyword.lower() in chunk_text
            )

            # Relevance = proportion of keywords found
            relevance = matches / len(expected_answers) if expected_answers else 0.0
            relevances.append(relevance)

        return relevances

    def _aggregate_results(
        self,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> Dict[str, Any]:
        """
        Aggregate results across all queries.

        Args:
            results: List of per-query results
            top_k: K value used

        Returns:
            Aggregated metrics
        """
        import statistics

        # Overall metrics
        ndcgs = [r["ndcg"] for r in results]
        precisions = [r["precision"] for r in results]
        recalls = [r["recall"] for r in results]
        mrrs = [r["mrr"] for r in results]

        # By difficulty
        by_difficulty = defaultdict(lambda: {"ndcg": [], "precision": [], "recall": []})
        for r in results:
            difficulty = r["difficulty"]
            by_difficulty[difficulty]["ndcg"].append(r["ndcg"])
            by_difficulty[difficulty]["precision"].append(r["precision"])
            by_difficulty[difficulty]["recall"].append(r["recall"])

        # Aggregate by difficulty
        difficulty_metrics = {}
        for difficulty, metrics in by_difficulty.items():
            difficulty_metrics[difficulty] = {
                "ndcg": statistics.mean(metrics["ndcg"]),
                "precision": statistics.mean(metrics["precision"]),
                "recall": statistics.mean(metrics["recall"])
            }

        return {
            "top_k": top_k,
            "num_queries": len(results),
            "ndcg_mean": statistics.mean(ndcgs),
            "ndcg_std": statistics.stdev(ndcgs) if len(ndcgs) > 1 else 0.0,
            "precision_mean": statistics.mean(precisions),
            "precision_std": statistics.stdev(precisions) if len(precisions) > 1 else 0.0,
            "recall_mean": statistics.mean(recalls),
            "recall_std": statistics.stdev(recalls) if len(recalls) > 1 else 0.0,
            "mrr_mean": statistics.mean(mrrs),
            "mrr_std": statistics.stdev(mrrs) if len(mrrs) > 1 else 0.0,
            "by_difficulty": difficulty_metrics
        }

    def _print_quality_assessment(self, aggregate: Dict[str, Any], top_k: int):
        """
        Print quality assessment based on metrics.

        Args:
            aggregate: Aggregated metrics
            top_k: K value
        """
        print(f"\n{'='*80}")
        print("QUALITY ASSESSMENT")
        print(f"{'='*80}")

        ndcg = aggregate["ndcg_mean"]
        precision = aggregate["precision_mean"]

        assessments = []

        # NDCG assessment
        if ndcg >= 0.7:
            assessments.append(f"✅ EXCELLENT: NDCG@{top_k} ≥ 0.7 (production-ready)")
        elif ndcg >= 0.5:
            assessments.append(f"✓ GOOD: NDCG@{top_k} ≥ 0.5 (acceptable quality)")
        else:
            assessments.append(f"⚠️  NEEDS IMPROVEMENT: NDCG@{top_k} < 0.5")

        # Precision assessment
        if precision >= 0.8:
            assessments.append(f"✅ EXCELLENT: Precision@{top_k} ≥ 0.8 (high precision)")
        elif precision >= 0.6:
            assessments.append(f"✓ GOOD: Precision@{top_k} ≥ 0.6 (acceptable precision)")
        else:
            assessments.append(f"⚠️  NEEDS IMPROVEMENT: Precision@{top_k} < 0.6")

        for assessment in assessments:
            print(f"\n{assessment}")

        # Overall verdict
        if ndcg >= 0.7 and precision >= 0.8:
            print(f"\n🎯 VERDICT: RAG system is PRODUCTION-READY")
        elif ndcg >= 0.5 and precision >= 0.6:
            print(f"\n👍 VERDICT: RAG system is ACCEPTABLE for initial deployment")
        else:
            print(f"\n⚠️  VERDICT: RAG system needs improvement before production")


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_rag_benchmark():
    """
    Run RAG benchmark test.

    This test requires:
    - Knowledge base to be ingested
    - RAGEngine to be operational
    """
    pytest.skip("Run manually after ingestion: pytest tests/intelligence/test_rag_benchmarking.py -v -s")


if __name__ == "__main__":
    print("RAG Benchmarking Script")
    print("Run this after knowledge base ingestion")
    print("\nTo run benchmark:")
    print("  pytest tests/intelligence/test_rag_benchmarking.py::test_rag_benchmark -v -s")
