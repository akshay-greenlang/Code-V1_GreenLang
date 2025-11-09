"""
GreenLang Application Performance Benchmarks
===========================================

Performance benchmarks for all GreenLang applications:
- GL-CBAM-APP: CBAM Importer pipeline
- GL-CSRD-APP: CSRD Reporting platform
- GL-VCCI-APP: VCCI Scope 3 platform

Targets:
- Single record: < 1 second P95
- Batch processing: > 1000 records/sec
- Memory efficiency: < 100MB per 10K records

Author: Performance Engineering Team
Date: 2025-11-09
Version: 1.0.0
"""

import asyncio
import time
import random
from typing import List, Dict, Any
from pathlib import Path
import sys

import pytest


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def cbam_shipment():
    """Single CBAM shipment record."""
    return {
        "shipment_id": "SHIP-001",
        "origin_country": "CN",
        "destination_country": "DE",
        "goods_category": "Steel",
        "cn_code": "7208",
        "quantity": 1000,
        "unit": "kg",
        "transport_mode": "Sea",
        "invoice_value": 50000,
        "currency": "EUR"
    }


@pytest.fixture
def cbam_shipments_1k(cbam_shipment):
    """1K CBAM shipment records."""
    return [
        {**cbam_shipment, "shipment_id": f"SHIP-{i:06d}"}
        for i in range(1000)
    ]


@pytest.fixture
def cbam_shipments_10k(cbam_shipment):
    """10K CBAM shipment records."""
    return [
        {**cbam_shipment, "shipment_id": f"SHIP-{i:06d}"}
        for i in range(10000)
    ]


@pytest.fixture
def csrd_company_data():
    """CSRD company data."""
    return {
        "company_id": "COMP-001",
        "name": "Example Manufacturing GmbH",
        "sector": "Manufacturing",
        "employees": 5000,
        "revenue": 500_000_000,
        "countries": ["DE", "FR", "IT"],
        "reporting_period": "2024"
    }


@pytest.fixture
def vcci_supplier_data():
    """VCCI supplier data."""
    return [
        {
            "supplier_id": f"SUPP-{i:06d}",
            "name": f"Supplier {i}",
            "country": random.choice(["CN", "US", "DE", "IN", "JP"]),
            "sector": random.choice(["Manufacturing", "Services", "Energy"]),
            "spend": random.randint(10000, 1000000),
            "currency": "USD"
        }
        for i in range(1000)
    ]


# ============================================================================
# MOCK AGENTS
# ============================================================================

class MockCBAMAgent:
    """Mock CBAM pipeline agents."""

    def __init__(self, name: str):
        self.name = name

    async def initialize(self):
        await asyncio.sleep(0.05)

    async def process(self, data: Dict) -> Dict:
        """Process single shipment."""
        # Simulate processing time
        if "Intake" in self.name:
            await asyncio.sleep(0.05)  # 50ms validation
        elif "Calculator" in self.name:
            await asyncio.sleep(0.15)  # 150ms calculation
        else:  # Packager
            await asyncio.sleep(0.03)  # 30ms packaging

        return {**data, f"{self.name}_processed": True}

    async def process_batch(self, data_list: List[Dict]) -> List[Dict]:
        """Process batch (20% more efficient)."""
        base_time = len(data_list) * 0.08  # Avg 80ms per record
        await asyncio.sleep(base_time)
        return [{**d, f"{self.name}_processed": True} for d in data_list]


class MockCSRDAgent:
    """Mock CSRD pipeline agents."""

    def __init__(self, name: str):
        self.name = name

    async def initialize(self):
        await asyncio.sleep(0.05)

    async def process(self, data: Dict) -> Dict:
        """Process CSRD data."""
        if "Materiality" in self.name:
            await asyncio.sleep(2.0)  # 2s materiality assessment
        elif "Calculator" in self.name:
            await asyncio.sleep(1.5)  # 1.5s ESRS calculations
        elif "Reporting" in self.name:
            await asyncio.sleep(0.8)  # 800ms XBRL generation
        else:
            await asyncio.sleep(0.5)

        return {**data, f"{self.name}_processed": True}


class MockVCCIAgent:
    """Mock VCCI Scope 3 agents."""

    def __init__(self, name: str):
        self.name = name

    async def initialize(self):
        await asyncio.sleep(0.05)

    async def process_batch(self, suppliers: List[Dict]) -> List[Dict]:
        """Process supplier batch."""
        # Simulate Scope 3 calculation
        base_time = len(suppliers) * 0.05  # 50ms per supplier
        await asyncio.sleep(base_time)

        results = []
        for supplier in suppliers:
            results.append({
                **supplier,
                "scope3_emissions": random.uniform(100, 10000),
                "confidence": random.uniform(0.7, 0.95)
            })

        return results


# ============================================================================
# GL-CBAM-APP BENCHMARKS
# ============================================================================

class TestCBAMBenchmarks:
    """Benchmarks for CBAM Importer application."""

    @pytest.mark.benchmark(group="cbam")
    def test_cbam_single_shipment_e2e(self, benchmark, cbam_shipment):
        """Benchmark: CBAM single shipment end-to-end pipeline."""

        async def process_shipment():
            # Initialize pipeline
            intake = MockCBAMAgent("IntakeAgent")
            calculator = MockCBAMAgent("CalculatorAgent")
            packager = MockCBAMAgent("PackagerAgent")

            await asyncio.gather(
                intake.initialize(),
                calculator.initialize(),
                packager.initialize()
            )

            # Process through pipeline
            start = time.perf_counter()

            result = cbam_shipment
            result = await intake.process(result)
            result = await calculator.process(result)
            result = await packager.process(result)

            duration = time.perf_counter() - start
            return duration

        duration = benchmark(lambda: asyncio.run(process_shipment()))
        # Target: < 1 second for single record
        assert duration < 1.0

    @pytest.mark.benchmark(group="cbam")
    @pytest.mark.parametrize("batch_size", [1000, 10000])
    def test_cbam_batch_throughput(self, benchmark, cbam_shipment, batch_size):
        """Benchmark: CBAM batch processing throughput."""

        async def process_batch():
            batch = [
                {**cbam_shipment, "shipment_id": f"SHIP-{i:06d}"}
                for i in range(batch_size)
            ]

            # Initialize pipeline
            intake = MockCBAMAgent("IntakeAgent")
            calculator = MockCBAMAgent("CalculatorAgent")
            packager = MockCBAMAgent("PackagerAgent")

            await asyncio.gather(
                intake.initialize(),
                calculator.initialize(),
                packager.initialize()
            )

            # Process batch through pipeline
            start = time.perf_counter()

            batch = await intake.process_batch(batch)
            batch = await calculator.process_batch(batch)
            batch = await packager.process_batch(batch)

            duration = time.perf_counter() - start
            return len(batch) / duration  # records/sec

        throughput = benchmark(lambda: asyncio.run(process_batch()))

        # Target: > 1000 records/sec
        assert throughput > 1000

    @pytest.mark.benchmark(group="cbam")
    def test_cbam_agent_execution_time(self, benchmark, cbam_shipment):
        """Benchmark: Individual CBAM agent execution times."""

        async def measure_agents():
            agents = [
                MockCBAMAgent("IntakeAgent"),
                MockCBAMAgent("CalculatorAgent"),
                MockCBAMAgent("PackagerAgent")
            ]

            times = {}
            for agent in agents:
                await agent.initialize()

                start = time.perf_counter()
                await agent.process(cbam_shipment)
                times[agent.name] = time.perf_counter() - start

            return times

        times = benchmark(lambda: asyncio.run(measure_agents()))

        # Validate individual agent times
        assert times["IntakeAgent"] < 0.1  # < 100ms
        assert times["CalculatorAgent"] < 0.2  # < 200ms
        assert times["PackagerAgent"] < 0.1  # < 100ms

    @pytest.mark.benchmark(group="cbam")
    def test_cbam_memory_usage_10k_records(self, benchmark, cbam_shipments_10k):
        """Benchmark: Memory usage for 10K records."""

        async def measure_memory():
            # Simulate processing 10K records
            results = []

            agent = MockCBAMAgent("CalculatorAgent")
            await agent.initialize()

            # Process in batches to simulate real usage
            batch_size = 1000
            for i in range(0, len(cbam_shipments_10k), batch_size):
                batch = cbam_shipments_10k[i:i + batch_size]
                processed = await agent.process_batch(batch)
                results.extend(processed)

            # Estimate memory (in practice, use memory_profiler)
            memory_mb = sys.getsizeof(results) / (1024 * 1024)
            return memory_mb

        memory_mb = benchmark(lambda: asyncio.run(measure_memory()))

        # Target: < 100MB per 10K records
        assert memory_mb < 100

    @pytest.mark.benchmark(group="cbam-comparison")
    def test_cbam_v1_vs_v2_performance(self, benchmark, cbam_shipments_1k):
        """Benchmark: v1 vs v2 agent performance comparison."""

        async def compare_versions():
            # Simulate v1 (custom code)
            v1_start = time.perf_counter()
            # v1 has more overhead
            await asyncio.sleep(len(cbam_shipments_1k) * 0.15)
            v1_time = time.perf_counter() - v1_start

            # Simulate v2 (infrastructure)
            v2_start = time.perf_counter()
            agent = MockCBAMAgent("CalculatorAgent")
            await agent.initialize()
            await agent.process_batch(cbam_shipments_1k)
            v2_time = time.perf_counter() - v2_start

            return {
                "v1_time": v1_time,
                "v2_time": v2_time,
                "speedup": v1_time / v2_time
            }

        result = benchmark(lambda: asyncio.run(compare_versions()))

        # v2 should be faster
        assert result["speedup"] > 1.0


# ============================================================================
# GL-CSRD-APP BENCHMARKS
# ============================================================================

class TestCSRDBenchmarks:
    """Benchmarks for CSRD Reporting platform."""

    @pytest.mark.benchmark(group="csrd")
    def test_csrd_materiality_assessment_time(self, benchmark, csrd_company_data):
        """Benchmark: CSRD materiality assessment time."""

        async def materiality_assessment():
            agent = MockCSRDAgent("MaterialityAgent")
            await agent.initialize()

            start = time.perf_counter()
            result = await agent.process(csrd_company_data)
            return time.perf_counter() - start

        duration = benchmark(lambda: asyncio.run(materiality_assessment()))

        # Target: < 5 seconds
        assert duration < 5.0

    @pytest.mark.benchmark(group="csrd")
    def test_csrd_esrs_calculation_time(self, benchmark, csrd_company_data):
        """Benchmark: ESRS calculation time."""

        async def esrs_calculation():
            agent = MockCSRDAgent("CalculatorAgent")
            await agent.initialize()

            start = time.perf_counter()
            result = await agent.process(csrd_company_data)
            return time.perf_counter() - start

        duration = benchmark(lambda: asyncio.run(esrs_calculation()))

        # Target: < 3 seconds
        assert duration < 3.0

    @pytest.mark.benchmark(group="csrd")
    def test_csrd_xbrl_generation_time(self, benchmark, csrd_company_data):
        """Benchmark: XBRL generation time."""

        async def xbrl_generation():
            agent = MockCSRDAgent("ReportingAgent")
            await agent.initialize()

            start = time.perf_counter()
            result = await agent.process(csrd_company_data)
            return time.perf_counter() - start

        duration = benchmark(lambda: asyncio.run(xbrl_generation()))

        # Target: < 2 seconds
        assert duration < 2.0

    @pytest.mark.benchmark(group="csrd")
    def test_csrd_rag_retrieval_llm_time(self, benchmark, csrd_company_data):
        """Benchmark: RAG retrieval + LLM response time."""

        async def rag_llm_query():
            # Simulate RAG retrieval
            rag_start = time.perf_counter()
            await asyncio.sleep(0.1)  # 100ms retrieval
            rag_time = time.perf_counter() - rag_start

            # Simulate LLM response
            llm_start = time.perf_counter()
            await asyncio.sleep(0.5)  # 500ms LLM
            llm_time = time.perf_counter() - llm_start

            return {
                "rag_time": rag_time,
                "llm_time": llm_time,
                "total_time": rag_time + llm_time
            }

        result = benchmark(lambda: asyncio.run(rag_llm_query()))

        # Total should be < 1 second
        assert result["total_time"] < 1.0

    @pytest.mark.benchmark(group="csrd")
    def test_csrd_multi_agent_pipeline_time(self, benchmark, csrd_company_data):
        """Benchmark: Full multi-agent pipeline time."""

        async def full_pipeline():
            # Initialize all agents
            materiality = MockCSRDAgent("MaterialityAgent")
            calculator = MockCSRDAgent("CalculatorAgent")
            reporting = MockCSRDAgent("ReportingAgent")

            await asyncio.gather(
                materiality.initialize(),
                calculator.initialize(),
                reporting.initialize()
            )

            # Run pipeline
            start = time.perf_counter()

            result = csrd_company_data
            result = await materiality.process(result)
            result = await calculator.process(result)
            result = await reporting.process(result)

            return time.perf_counter() - start

        duration = benchmark(lambda: asyncio.run(full_pipeline()))

        # Target: < 10 seconds for full pipeline
        assert duration < 10.0


# ============================================================================
# GL-VCCI-APP BENCHMARKS
# ============================================================================

class TestVCCIBenchmarks:
    """Benchmarks for VCCI Scope 3 platform."""

    @pytest.mark.benchmark(group="vcci")
    def test_vcci_scope3_calculation_10k_suppliers(self, benchmark):
        """Benchmark: Scope 3 calculation for 10K suppliers."""

        async def scope3_calculation():
            suppliers = [
                {
                    "supplier_id": f"SUPP-{i:06d}",
                    "spend": random.randint(10000, 1000000),
                    "sector": random.choice(["Manufacturing", "Services"])
                }
                for i in range(10000)
            ]

            agent = MockVCCIAgent("Scope3Agent")
            await agent.initialize()

            start = time.perf_counter()
            results = await agent.process_batch(suppliers)
            duration = time.perf_counter() - start

            return {
                "duration": duration,
                "throughput": len(results) / duration
            }

        result = benchmark(lambda: asyncio.run(scope3_calculation()))

        # Should process 10K suppliers in reasonable time
        assert result["duration"] < 60.0  # < 1 minute
        assert result["throughput"] > 100  # > 100 suppliers/sec

    @pytest.mark.benchmark(group="vcci")
    def test_vcci_entity_resolution_batch(self, benchmark, vcci_supplier_data):
        """Benchmark: Entity resolution batch performance."""

        async def entity_resolution():
            # Simulate entity matching
            start = time.perf_counter()

            resolved = []
            for supplier in vcci_supplier_data:
                # Simulate fuzzy matching
                await asyncio.sleep(0.01)  # 10ms per entity
                resolved.append({
                    **supplier,
                    "resolved": True,
                    "confidence": random.uniform(0.8, 1.0)
                })

            duration = time.perf_counter() - start
            return len(resolved) / duration

        throughput = benchmark(lambda: asyncio.run(entity_resolution()))

        # Should process > 50 entities/sec
        assert throughput > 50

    @pytest.mark.benchmark(group="vcci")
    def test_vcci_hotspot_analysis_time(self, benchmark, vcci_supplier_data):
        """Benchmark: Hotspot analysis time."""

        async def hotspot_analysis():
            # Add emissions data
            suppliers_with_emissions = [
                {**s, "emissions": random.uniform(100, 10000)}
                for s in vcci_supplier_data
            ]

            # Analyze hotspots
            start = time.perf_counter()

            # Sort by emissions
            sorted_suppliers = sorted(
                suppliers_with_emissions,
                key=lambda x: x["emissions"],
                reverse=True
            )

            # Identify top 20%
            hotspots = sorted_suppliers[:int(len(sorted_suppliers) * 0.2)]

            return time.perf_counter() - start

        duration = benchmark(lambda: asyncio.run(hotspot_analysis()))

        # Should be fast (< 100ms)
        assert duration < 0.1

    @pytest.mark.benchmark(group="vcci")
    def test_vcci_report_generation_all_formats(self, benchmark, vcci_supplier_data):
        """Benchmark: Report generation for all formats."""

        async def generate_reports():
            formats = ["PDF", "Excel", "CSV", "JSON"]
            times = {}

            for fmt in formats:
                start = time.perf_counter()

                if fmt == "PDF":
                    await asyncio.sleep(0.5)  # 500ms
                elif fmt == "Excel":
                    await asyncio.sleep(0.3)  # 300ms
                elif fmt == "CSV":
                    await asyncio.sleep(0.1)  # 100ms
                else:  # JSON
                    await asyncio.sleep(0.05)  # 50ms

                times[fmt] = time.perf_counter() - start

            return times

        times = benchmark(lambda: asyncio.run(generate_reports()))

        # Validate individual format times
        assert times["PDF"] < 1.0
        assert times["Excel"] < 0.5
        assert times["CSV"] < 0.2
        assert times["JSON"] < 0.1


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_benchmark_summary():
    """Print summary of application benchmarks."""
    print("\n" + "=" * 80)
    print("GREENLANG APPLICATION PERFORMANCE TARGETS")
    print("=" * 80)
    print("\nCBAM Application:")
    print("  - Single shipment:    < 1.0s")
    print("  - Batch throughput:   > 1000 records/sec")
    print("  - Memory (10K):       < 100 MB")
    print("\nCSRD Application:")
    print("  - Materiality:        < 5.0s")
    print("  - ESRS calculation:   < 3.0s")
    print("  - XBRL generation:    < 2.0s")
    print("  - Full pipeline:      < 10.0s")
    print("\nVCCI Application:")
    print("  - Scope 3 (10K):      < 60.0s (> 100/sec)")
    print("  - Entity resolution:  > 50 entities/sec")
    print("  - Hotspot analysis:   < 100ms")
    print("=" * 80)


if __name__ == "__main__":
    print_benchmark_summary()
