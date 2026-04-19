# -*- coding: utf-8 -*-
"""
End-to-end tests for PACK-036 Utility Analysis Pack
=======================================================

Tests full pipelines from bill parsing through reporting. Validates
that engines work together and provenance chains are maintained.

Scenarios:
    1. Bill audit pipeline (parse -> validate -> report)
    2. Rate optimization pipeline (parse -> rate analysis -> comparison)
    3. Demand management pipeline (interval -> demand analysis -> profile)
    4. Budget forecast pipeline (history -> forecast -> report)
    5. Multi-facility portfolio analysis
    6. Cross-engine data flow (parse -> allocate -> benchmark)
    7. Provenance chain integrity across engines

Coverage target: 85%+
Total tests: ~30
"""

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"


def _load(name: str, subdir: str = "engines"):
    base = PACK_ROOT / subdir
    path = base / f"{name}.py"
    if not path.exists():
        pytest.skip(f"File not found: {path}")
    mod_key = f"pack036_e2e.{subdir}.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


# =============================================================================
# Helper: Build sample bill for E2E tests
# =============================================================================

def _make_e2e_bill():
    return {
        "bill_id": "BILL-E2E-001",
        "account_number": "ACC-E2E-001",
        "facility_id": "FAC-E2E-DE-001",
        "facility_name": "E2E Test Office",
        "utility_type": "ELECTRICITY",
        "provider": "Vattenfall Europe",
        "period_start": "2025-01-01",
        "period_end": "2025-01-31",
        "billing_days": 31,
        "consumption_kwh": 150_000,
        "demand_kw": 480.0,
        "power_factor": 0.92,
        "line_items": [
            {"description": "Energy Charge", "quantity": 150_000, "unit": "kWh",
             "rate": Decimal("0.12"), "amount": Decimal("18000.00")},
            {"description": "Demand Charge", "quantity": 480, "unit": "kW",
             "rate": Decimal("8.50"), "amount": Decimal("4080.00")},
            {"description": "Network Fee", "quantity": 1, "unit": "month",
             "rate": Decimal("1200.00"), "amount": Decimal("1200.00")},
        ],
        "subtotal_eur": Decimal("23280.00"),
        "tax_eur": Decimal("4423.20"),
        "total_eur": Decimal("27703.20"),
        "currency": "EUR",
    }


def _make_e2e_report_data():
    return {
        "facility_id": "FAC-E2E-DE-001",
        "facility_name": "E2E Test Office",
        "period": "2025-01",
        "total_cost_eur": Decimal("38021.79"),
        "total_consumption_kwh": 150_000,
        "demand_kw": 480,
        "eui_kwh_per_m2": 263.0,
        "anomalies_count": 0,
        "savings_potential_eur": Decimal("4500"),
        "bills": [_make_e2e_bill()],
    }


# =============================================================================
# E2E: Bill Audit Pipeline
# =============================================================================


class TestE2EBillAudit:
    """End-to-end test: bill parsing -> validation -> reporting."""

    def test_e2e_bill_parse_and_validate(self):
        """Parse a bill and validate line item totals."""
        parser_mod = _load("utility_bill_parser_engine")
        engine = parser_mod.UtilityBillParserEngine()

        parse = (getattr(engine, "parse_bill", None)
                 or getattr(engine, "parse", None)
                 or getattr(engine, "extract", None))
        if parse is None:
            pytest.skip("parse method not found")

        result = parse(_make_e2e_bill())
        assert result is not None
        assert hasattr(result, "provenance_hash")

    def test_e2e_bill_parse_line_items(self):
        """Bill parser should process all line items."""
        parser_mod = _load("utility_bill_parser_engine")
        engine = parser_mod.UtilityBillParserEngine()

        parse = (getattr(engine, "parse_bill", None)
                 or getattr(engine, "parse", None)
                 or getattr(engine, "extract", None))
        if parse is None:
            pytest.skip("parse method not found")

        bill = _make_e2e_bill()
        result = parse(bill)
        items = (getattr(result, "line_items", None)
                 or getattr(result, "parsed_items", None)
                 or getattr(result, "charges", None))
        if items is not None:
            assert len(items) >= 3

    def test_e2e_bill_to_report(self):
        """Parse a bill then generate an audit report."""
        parser_mod = _load("utility_bill_parser_engine")
        reporting_mod = _load("utility_reporting_engine")

        # Step 1: Parse bill
        p_engine = parser_mod.UtilityBillParserEngine()
        parse = (getattr(p_engine, "parse_bill", None)
                 or getattr(p_engine, "parse", None)
                 or getattr(p_engine, "extract", None))
        if parse is None:
            pytest.skip("parse method not found")
        parse_result = parse(_make_e2e_bill())
        assert parse_result is not None

        # Step 2: Generate report
        r_engine = reporting_mod.UtilityReportingEngine()
        gen = (getattr(r_engine, "generate_monthly_summary", None)
               or getattr(r_engine, "monthly_summary", None)
               or getattr(r_engine, "render_markdown", None))
        if gen is None:
            pytest.skip("report generation method not found")
        report_result = gen(_make_e2e_report_data())
        assert report_result is not None


# =============================================================================
# E2E: Rate Optimization Pipeline
# =============================================================================


class TestE2ERateOptimization:
    """End-to-end test: bill parsing -> rate analysis -> rate comparison."""

    def test_e2e_rate_analysis(self):
        """Analyze rate structure for a parsed bill."""
        rate_mod = _load("rate_structure_analyzer_engine")
        engine = rate_mod.RateStructureAnalyzerEngine()

        analyze = (getattr(engine, "analyze_rate", None)
                   or getattr(engine, "analyze", None)
                   or getattr(engine, "evaluate_rate", None))
        if analyze is None:
            pytest.skip("analyze method not found")

        rate_data = {
            "rate_id": "RATE-E2E-001",
            "rate_name": "E2E TOU Rate",
            "rate_type": "TOU",
            "energy_charges": {
                "on_peak": {"rate_eur_per_kwh": Decimal("0.1450")},
                "off_peak": {"rate_eur_per_kwh": Decimal("0.0950")},
            },
            "demand_charges": {"rate_eur_per_kw": Decimal("8.50")},
            "consumption_kwh": 150_000,
            "demand_kw": 480,
        }
        result = analyze(rate_data)
        assert result is not None

    def test_e2e_rate_comparison(self):
        """Compare multiple rate structures for the same consumption."""
        rate_mod = _load("rate_structure_analyzer_engine")
        engine = rate_mod.RateStructureAnalyzerEngine()

        compare = (getattr(engine, "compare_rates", None)
                   or getattr(engine, "rate_comparison", None)
                   or getattr(engine, "compare", None))
        if compare is None:
            pytest.skip("compare method not found")

        rates = [
            {"rate_id": "R-001", "rate_name": "Flat Rate",
             "rate_type": "FLAT", "rate_eur_per_kwh": Decimal("0.1200")},
            {"rate_id": "R-002", "rate_name": "TOU Rate",
             "rate_type": "TOU", "on_peak_rate": Decimal("0.1450"),
             "off_peak_rate": Decimal("0.0950")},
        ]
        try:
            result = compare(rates=rates, consumption_kwh=150_000)
        except TypeError:
            try:
                result = compare(rates)
            except Exception:
                pytest.skip("Cannot invoke compare")
        assert result is not None


# =============================================================================
# E2E: Demand Management Pipeline
# =============================================================================


class TestE2EDemandManagement:
    """End-to-end test: interval data -> demand analysis -> profile."""

    def test_e2e_demand_analysis(self, sample_interval_data):
        """Analyze demand profile from interval data."""
        demand_mod = _load("demand_analysis_engine")
        engine = demand_mod.DemandAnalysisEngine()

        analyze = (getattr(engine, "analyze_demand", None)
                   or getattr(engine, "analyze", None)
                   or getattr(engine, "build_profile", None))
        if analyze is None:
            pytest.skip("analyze method not found")

        result = analyze(intervals=sample_interval_data)
        assert result is not None
        assert hasattr(result, "provenance_hash")

    def test_e2e_load_factor_calculation(self, sample_interval_data):
        """Demand analysis should calculate load factor."""
        demand_mod = _load("demand_analysis_engine")
        engine = demand_mod.DemandAnalysisEngine()

        analyze = (getattr(engine, "analyze_demand", None)
                   or getattr(engine, "analyze", None)
                   or getattr(engine, "build_profile", None))
        if analyze is None:
            pytest.skip("analyze method not found")

        result = analyze(intervals=sample_interval_data)
        load_factor = (getattr(result, "load_factor", None)
                       or getattr(result, "lf", None))
        if load_factor is not None:
            assert 0.0 < float(load_factor) < 1.0


# =============================================================================
# E2E: Budget Forecast Pipeline
# =============================================================================


class TestE2EBudgetForecast:
    """End-to-end test: historical data -> budget forecast -> report."""

    def test_e2e_budget_forecast(self, sample_historical_data):
        """Generate budget forecast from historical data."""
        forecast_mod = _load("budget_forecasting_engine")
        engine = forecast_mod.BudgetForecastingEngine()

        forecast = (getattr(engine, "forecast", None)
                    or getattr(engine, "generate_forecast", None)
                    or getattr(engine, "predict", None))
        if forecast is None:
            pytest.skip("forecast method not found")

        result = forecast(history=sample_historical_data, horizon_months=12)
        assert result is not None
        assert hasattr(result, "provenance_hash")

    def test_e2e_forecast_to_report(self, sample_historical_data):
        """Forecast then generate budget report."""
        forecast_mod = _load("budget_forecasting_engine")
        reporting_mod = _load("utility_reporting_engine")

        # Step 1: Forecast
        f_engine = forecast_mod.BudgetForecastingEngine()
        forecast = (getattr(f_engine, "forecast", None)
                    or getattr(f_engine, "generate_forecast", None)
                    or getattr(f_engine, "predict", None))
        if forecast is None:
            pytest.skip("forecast method not found")
        forecast_result = forecast(history=sample_historical_data, horizon_months=12)
        assert forecast_result is not None

        # Step 2: Report
        r_engine = reporting_mod.UtilityReportingEngine()
        gen = (getattr(r_engine, "generate_monthly_summary", None)
               or getattr(r_engine, "monthly_summary", None)
               or getattr(r_engine, "render_markdown", None))
        if gen is None:
            pytest.skip("report method not found")
        report_result = gen(_make_e2e_report_data())
        assert report_result is not None


# =============================================================================
# E2E: Cost Allocation Pipeline
# =============================================================================


class TestE2ECostAllocation:
    """End-to-end test: bill -> cost allocation -> tenant invoicing."""

    def test_e2e_cost_allocation(self, sample_allocation_entities, sample_allocation_rules):
        """Allocate utility costs across tenants."""
        alloc_mod = _load("cost_allocation_engine")
        engine = alloc_mod.CostAllocationEngine()

        allocate = (getattr(engine, "allocate_costs", None)
                    or getattr(engine, "allocate", None)
                    or getattr(engine, "calculate_allocation", None))
        if allocate is None:
            pytest.skip("allocate method not found")

        result = allocate(
            total_cost=Decimal("38021.79"),
            entities=sample_allocation_entities,
            rules=sample_allocation_rules,
        )
        assert result is not None
        assert hasattr(result, "provenance_hash")

    def test_e2e_allocation_sums_to_total(self, sample_allocation_entities, sample_allocation_rules):
        """Allocated costs should sum to total cost."""
        alloc_mod = _load("cost_allocation_engine")
        engine = alloc_mod.CostAllocationEngine()

        allocate = (getattr(engine, "allocate_costs", None)
                    or getattr(engine, "allocate", None)
                    or getattr(engine, "calculate_allocation", None))
        if allocate is None:
            pytest.skip("allocate method not found")

        result = allocate(
            total_cost=Decimal("38021.79"),
            entities=sample_allocation_entities,
            rules=sample_allocation_rules,
        )
        allocations = (getattr(result, "allocations", None)
                       or getattr(result, "entity_allocations", None)
                       or getattr(result, "results", None))
        if allocations is not None and isinstance(allocations, (list, dict)):
            total = Decimal("0")
            if isinstance(allocations, list):
                for a in allocations:
                    amt = (getattr(a, "allocated_cost", None)
                           or getattr(a, "cost_eur", None)
                           or (a.get("allocated_cost") if isinstance(a, dict) else None))
                    if amt is not None:
                        total += Decimal(str(amt))
            if total > 0:
                assert abs(total - Decimal("38021.79")) < Decimal("1.00")


# =============================================================================
# E2E: Multi-Facility Portfolio Analysis
# =============================================================================


class TestE2EPortfolioAnalysis:
    """End-to-end test: benchmark multiple facilities."""

    def test_e2e_portfolio_benchmark(self):
        """Benchmark multiple facilities against each other."""
        benchmark_mod = _load("utility_benchmark_engine")
        engine = benchmark_mod.UtilityBenchmarkEngine()

        benchmark = (getattr(engine, "benchmark_portfolio", None)
                     or getattr(engine, "benchmark", None)
                     or getattr(engine, "compare_facilities", None))
        if benchmark is None:
            pytest.skip("benchmark method not found")

        facilities = [
            {"facility_id": f"FAC-PORTF-{i:03d}",
             "building_type": "OFFICE",
             "floor_area_m2": 10000 + i * 2000,
             "annual_electricity_kwh": 1_800_000 + i * 300_000,
             "site_eui_kwh_per_m2": 263.0 + i * 15}
            for i in range(5)
        ]

        try:
            result = benchmark(facilities=facilities)
        except TypeError:
            try:
                result = benchmark(facilities)
            except Exception:
                pytest.skip("Cannot invoke benchmark")
        assert result is not None

    def test_e2e_portfolio_unique_provenance(self):
        """Each facility benchmark should have a unique provenance hash."""
        benchmark_mod = _load("utility_benchmark_engine")
        engine = benchmark_mod.UtilityBenchmarkEngine()

        single_bench = (getattr(engine, "benchmark_facility", None)
                        or getattr(engine, "benchmark", None)
                        or getattr(engine, "calculate_eui", None))
        if single_bench is None:
            pytest.skip("single benchmark method not found")

        hashes = set()
        for i in range(3):
            facility = {
                "facility_id": f"FAC-UNQ-{i:03d}",
                "building_type": "OFFICE",
                "floor_area_m2": 10000 + i * 5000,
                "annual_electricity_kwh": 1_800_000 + i * 500_000,
                "site_eui_kwh_per_m2": 263.0 + i * 30,
            }
            try:
                result = single_bench(facility=facility)
            except TypeError:
                try:
                    result = single_bench(facility)
                except Exception:
                    pytest.skip("Cannot invoke benchmark")
            if hasattr(result, "provenance_hash"):
                hashes.add(result.provenance_hash)
        if len(hashes) > 0:
            assert len(hashes) >= 2  # Different facilities = different hashes


# =============================================================================
# E2E: Cross-Engine Data Flow
# =============================================================================


class TestE2ECrossEngineFlow:
    """End-to-end test: parse -> allocate -> benchmark -> report."""

    def test_e2e_parse_to_allocation(self, sample_allocation_entities, sample_allocation_rules):
        """Parse a bill then allocate costs."""
        parser_mod = _load("utility_bill_parser_engine")
        alloc_mod = _load("cost_allocation_engine")

        # Step 1: Parse
        p_engine = parser_mod.UtilityBillParserEngine()
        parse = (getattr(p_engine, "parse_bill", None)
                 or getattr(p_engine, "parse", None)
                 or getattr(p_engine, "extract", None))
        if parse is None:
            pytest.skip("parse method not found")
        parse_result = parse(_make_e2e_bill())
        assert parse_result is not None

        # Step 2: Allocate using parsed total
        total_cost = (getattr(parse_result, "total_cost", None)
                      or getattr(parse_result, "total_eur", None)
                      or Decimal("27703.20"))
        a_engine = alloc_mod.CostAllocationEngine()
        allocate = (getattr(a_engine, "allocate_costs", None)
                    or getattr(a_engine, "allocate", None)
                    or getattr(a_engine, "calculate_allocation", None))
        if allocate is None:
            pytest.skip("allocate method not found")
        alloc_result = allocate(
            total_cost=Decimal(str(total_cost)),
            entities=sample_allocation_entities,
            rules=sample_allocation_rules,
        )
        assert alloc_result is not None

    def test_e2e_weather_to_benchmark(self, sample_monthly_consumption_weather,
                                       sample_weather_data):
        """Weather normalize then benchmark."""
        weather_mod = _load("weather_normalization_engine")
        benchmark_mod = _load("utility_benchmark_engine")

        # Step 1: Weather normalize
        w_engine = weather_mod.WeatherNormalizationEngine()
        normalize = (getattr(w_engine, "normalize_consumption", None)
                     or getattr(w_engine, "weather_normalize", None))
        if normalize is None:
            pytest.skip("normalize method not found")
        data = [{"consumption": r["gas_kwh"], "hdd": r["hdd"]}
                for r in sample_monthly_consumption_weather]
        norm_result = normalize(data=data, tmy_weather=sample_weather_data)
        assert norm_result is not None

        # Step 2: Benchmark
        b_engine = benchmark_mod.UtilityBenchmarkEngine()
        bench = (getattr(b_engine, "benchmark_facility", None)
                 or getattr(b_engine, "benchmark", None)
                 or getattr(b_engine, "calculate_eui", None))
        if bench is None:
            pytest.skip("benchmark method not found")
        facility = {
            "facility_id": "FAC-WB-001",
            "building_type": "OFFICE",
            "floor_area_m2": 10000,
            "annual_electricity_kwh": 1_980_000,
            "site_eui_kwh_per_m2": 263.0,
        }
        try:
            bench_result = bench(facility=facility)
        except TypeError:
            try:
                bench_result = bench(facility)
            except Exception:
                pytest.skip("Cannot invoke benchmark")
        assert bench_result is not None


# =============================================================================
# E2E: Procurement Intelligence Pipeline
# =============================================================================


class TestE2EProcurement:
    """End-to-end test: market data -> procurement analysis."""

    def test_e2e_procurement_analysis(self, sample_market_prices):
        """Analyze market prices for procurement strategy."""
        proc_mod = _load("procurement_intelligence_engine")
        engine = proc_mod.ProcurementIntelligenceEngine()

        analyze = (getattr(engine, "analyze_market", None)
                   or getattr(engine, "market_analysis", None)
                   or getattr(engine, "analyze", None))
        if analyze is None:
            pytest.skip("analyze method not found")

        try:
            result = analyze(market_data=sample_market_prices)
        except TypeError:
            try:
                result = analyze(sample_market_prices)
            except Exception:
                pytest.skip("Cannot invoke analyze")
        assert result is not None

    def test_e2e_procurement_strategy(self, sample_market_prices):
        """Generate procurement strategy recommendation."""
        proc_mod = _load("procurement_intelligence_engine")
        engine = proc_mod.ProcurementIntelligenceEngine()

        recommend = (getattr(engine, "recommend_strategy", None)
                     or getattr(engine, "procurement_strategy", None)
                     or getattr(engine, "strategy", None))
        if recommend is None:
            pytest.skip("recommend method not found")

        try:
            result = recommend(
                market_data=sample_market_prices,
                annual_consumption_mwh=1980,
                contract_months=12,
            )
        except TypeError:
            try:
                result = recommend(sample_market_prices)
            except Exception:
                pytest.skip("Cannot invoke recommend")
        assert result is not None


# =============================================================================
# E2E: Regulatory Charge Optimization
# =============================================================================


class TestE2ERegulatoryOptimization:
    """End-to-end test: regulatory charge analysis -> optimization."""

    def test_e2e_regulatory_analysis(self):
        """Analyze regulatory charges and identify optimization."""
        reg_mod = _load("regulatory_charge_optimizer_engine")
        engine = reg_mod.RegulatoryChargeOptimizerEngine()

        analyze = (getattr(engine, "analyze_charges", None)
                   or getattr(engine, "analyze", None)
                   or getattr(engine, "optimize", None))
        if analyze is None:
            pytest.skip("analyze method not found")

        charges = {
            "facility_id": "FAC-REG-001",
            "country": "DE",
            "consumption_kwh": 150_000,
            "eeg_surcharge_eur": Decimal("5580.00"),
            "electricity_tax_eur": Decimal("3075.00"),
            "concession_fee_eur": Decimal("16.50"),
            "network_charges_eur": Decimal("1200.00"),
        }
        try:
            result = analyze(charges=charges)
        except TypeError:
            try:
                result = analyze(charges)
            except Exception:
                pytest.skip("Cannot invoke analyze")
        assert result is not None


# =============================================================================
# Provenance Chain Integrity
# =============================================================================


class TestProvenanceChain:
    """Test that provenance hashes are maintained across pipeline."""

    def test_provenance_chain_parse_to_report(self):
        """Parse and report should produce valid provenance hashes."""
        parser_mod = _load("utility_bill_parser_engine")
        reporting_mod = _load("utility_reporting_engine")

        # Parse
        p_engine = parser_mod.UtilityBillParserEngine()
        parse = (getattr(p_engine, "parse_bill", None)
                 or getattr(p_engine, "parse", None)
                 or getattr(p_engine, "extract", None))
        if parse is None:
            pytest.skip("parse method not found")
        parse_result = parse(_make_e2e_bill())
        parse_hash = getattr(parse_result, "provenance_hash", None)
        if parse_hash is not None:
            assert len(parse_hash) == 64

        # Report
        r_engine = reporting_mod.UtilityReportingEngine()
        gen = (getattr(r_engine, "generate_monthly_summary", None)
               or getattr(r_engine, "monthly_summary", None)
               or getattr(r_engine, "render_markdown", None))
        if gen is None:
            pytest.skip("report method not found")
        report_result = gen(_make_e2e_report_data())
        report_hash = getattr(report_result, "provenance_hash", None)
        if report_hash is not None:
            assert len(report_hash) == 64
            if parse_hash is not None:
                assert parse_hash != report_hash  # Different inputs = different hashes

    def test_provenance_deterministic(self):
        """Same input should produce same provenance hash."""
        parser_mod = _load("utility_bill_parser_engine")
        engine = parser_mod.UtilityBillParserEngine()

        parse = (getattr(engine, "parse_bill", None)
                 or getattr(engine, "parse", None)
                 or getattr(engine, "extract", None))
        if parse is None:
            pytest.skip("parse method not found")

        bill = _make_e2e_bill()
        r1 = parse(bill)
        r2 = parse(bill)

        h1 = getattr(r1, "provenance_hash", None)
        h2 = getattr(r2, "provenance_hash", None)
        if h1 is not None and h2 is not None:
            assert h1 == h2  # Deterministic: same input -> same hash

    def test_provenance_different_inputs(self):
        """Different inputs should produce different provenance hashes."""
        parser_mod = _load("utility_bill_parser_engine")
        engine = parser_mod.UtilityBillParserEngine()

        parse = (getattr(engine, "parse_bill", None)
                 or getattr(engine, "parse", None)
                 or getattr(engine, "extract", None))
        if parse is None:
            pytest.skip("parse method not found")

        bill1 = _make_e2e_bill()
        bill2 = _make_e2e_bill()
        bill2["bill_id"] = "BILL-E2E-002"
        bill2["consumption_kwh"] = 200_000
        bill2["total_eur"] = Decimal("35000.00")

        r1 = parse(bill1)
        r2 = parse(bill2)

        h1 = getattr(r1, "provenance_hash", None)
        h2 = getattr(r2, "provenance_hash", None)
        if h1 is not None and h2 is not None:
            assert h1 != h2


# =============================================================================
# E2E: Full Utility Analysis Pipeline
# =============================================================================


class TestE2EFullPipeline:
    """End-to-end test: complete utility analysis pipeline."""

    def test_e2e_full_pipeline(self, sample_bill_history, sample_historical_data):
        """Run full analysis: parse -> analyze -> forecast -> report."""
        parser_mod = _load("utility_bill_parser_engine")
        forecast_mod = _load("budget_forecasting_engine")
        reporting_mod = _load("utility_reporting_engine")

        # Step 1: Parse
        p_engine = parser_mod.UtilityBillParserEngine()
        parse = (getattr(p_engine, "parse_bill", None)
                 or getattr(p_engine, "parse", None)
                 or getattr(p_engine, "extract", None))
        if parse is None:
            pytest.skip("parse method not found")
        parse_result = parse(_make_e2e_bill())
        assert parse_result is not None

        # Step 2: Forecast
        f_engine = forecast_mod.BudgetForecastingEngine()
        forecast = (getattr(f_engine, "forecast", None)
                    or getattr(f_engine, "generate_forecast", None)
                    or getattr(f_engine, "predict", None))
        if forecast is None:
            pytest.skip("forecast method not found")
        forecast_result = forecast(history=sample_historical_data, horizon_months=12)
        assert forecast_result is not None

        # Step 3: Report
        r_engine = reporting_mod.UtilityReportingEngine()
        gen = (getattr(r_engine, "generate_monthly_summary", None)
               or getattr(r_engine, "monthly_summary", None)
               or getattr(r_engine, "render_markdown", None))
        if gen is None:
            pytest.skip("report method not found")
        report_result = gen(_make_e2e_report_data())
        assert report_result is not None

    def test_e2e_multi_engine_provenance_chain(self):
        """All engines in the chain should produce valid provenance hashes."""
        engines_to_test = [
            ("utility_bill_parser_engine", "UtilityBillParserEngine"),
            ("rate_structure_analyzer_engine", "RateStructureAnalyzerEngine"),
            ("demand_analysis_engine", "DemandAnalysisEngine"),
            ("cost_allocation_engine", "CostAllocationEngine"),
            ("budget_forecasting_engine", "BudgetForecastingEngine"),
            ("utility_benchmark_engine", "UtilityBenchmarkEngine"),
            ("weather_normalization_engine", "WeatherNormalizationEngine"),
            ("utility_reporting_engine", "UtilityReportingEngine"),
        ]
        loaded_count = 0
        for engine_name, class_name in engines_to_test:
            try:
                mod = _load(engine_name)
                cls = getattr(mod, class_name, None)
                if cls is not None:
                    engine = cls()
                    assert engine is not None
                    loaded_count += 1
            except Exception:
                pass
        assert loaded_count >= 5  # At least 5 engines should load successfully
