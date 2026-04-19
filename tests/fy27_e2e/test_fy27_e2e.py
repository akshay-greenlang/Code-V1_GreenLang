# -*- coding: utf-8 -*-
"""FY27 end-to-end integration test.

Runs the full FY27 stack (GL-Comply-APP -> 10 adapters -> Scope Engine ->
Factor catalog -> GWP -> framework views -> unified report) in a subprocess
so the path-based imports inside `applications/GL-Comply-APP/Compliance-Hub/`
(which uses simple `agents`/`schemas`/`services` names that collide with
`greenlang.agents` / `greenlang.schemas`) run with a clean sys.modules.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


_HUB_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "applications"
    / "GL-Comply-APP"
    / "Compliance-Hub"
)

_E2E_SCRIPT = r"""
import asyncio
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import tempfile

HUB = Path(r'{HUB}')
sys.path.insert(0, str(HUB))

import schemas.models as hub_schemas
import services  # registers 10 adapters
import agents.orchestrator_agent as hub_orch
import agents.unified_report_agent as hub_report
from greenlang.schemas.enums import ComplianceStatus, ReportFormat

def build():
    return hub_schemas.ComplianceRequest(
        entity=hub_schemas.EntitySnapshot(
            entity_id='e2e-acme-de', legal_name='ACME Europe GmbH', jurisdiction='DE',
            revenue_eur=120_000_000, employees=750,
            imports_cbam_goods=True, handles_eudr_commodities=True,
        ),
        reporting_period_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        reporting_period_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
        frameworks=list(hub_schemas.FrameworkEnum),
        data_sources={{
            'activities': [
                {{'activity_id': 'd1', 'activity_type': 'stationary_combustion',
                  'fuel_type': 'diesel', 'quantity': '1000', 'unit': 'gallons', 'year': 2024}},
                {{'activity_id': 'g1', 'activity_type': 'stationary_combustion',
                  'fuel_type': 'natural_gas', 'quantity': '500', 'unit': 'therms', 'year': 2024}},
            ],
            'eudr_commodities': [{{'cn_code': '1801', 'weight_kg': 50000, 'origin': 'CI'}}],
            'taxonomy_activities': [
                {{'turnover_eur': 80_000_000, 'eligible': True, 'aligned': True}},
                {{'turnover_eur': 40_000_000, 'eligible': False, 'aligned': False}},
            ],
        }},
    )

req = build()
orchestrator = hub_orch.ComplianceOrchestrator()
r1 = asyncio.run(orchestrator.run(req))
r2 = asyncio.run(orchestrator.run(req))

ghg = r1.results[hub_schemas.FrameworkEnum.GHG_PROTOCOL].metrics
csrd = r1.results[hub_schemas.FrameworkEnum.CSRD].metrics
eudr = r1.results[hub_schemas.FrameworkEnum.EUDR]
tax = r1.results[hub_schemas.FrameworkEnum.EU_TAXONOMY]

with tempfile.TemporaryDirectory() as tmp:
    agent = hub_report.UnifiedReportAgent()
    reports = {{
        fmt.value: agent.generate(r1, fmt, Path(tmp))
        for fmt in [ReportFormat.JSON, ReportFormat.XML, ReportFormat.PDF]
    }}

result = {{
    'adapters_registered': len(services.registry.available()),
    'frameworks_in_report': len(r1.results),
    'aggregate_hash_1': r1.aggregate_provenance_hash,
    'aggregate_hash_2': r2.aggregate_provenance_hash,
    'hashes_match': r1.aggregate_provenance_hash == r2.aggregate_provenance_hash,
    'hash_len': len(r1.aggregate_provenance_hash),
    'overall_status': r1.overall_status.value,
    'ghg_scope_1_kg': ghg['scope_1_co2e_kg'],
    'csrd_scope_1_kg': csrd['scope_1_co2e_kg'],
    'eudr_status': eudr.compliance_status.value,
    'eudr_commodities': eudr.metrics.get('commodities_count', 0),
    'taxonomy_aligned_share': tax.metrics.get('aligned_share'),
    'reports': {{k: {{'size': v['size_bytes'], 'hash_len': len(v['content_hash'])}} for k, v in reports.items()}},
    'eudr_in_gap_analysis': any(g.get('framework') == 'eudr' for g in r1.gap_analysis),
}}
print(json.dumps(result))
"""


def _run_e2e():
    import os

    script = _E2E_SCRIPT.format(HUB=_HUB_ROOT).replace("{{", "{").replace("}}", "}")
    env = os.environ.copy()
    # pytest may have injected PYTHONPATH pointing at the project root; drop it
    # so our explicit sys.path.insert controls resolution.
    env.pop("PYTHONPATH", None)
    out = subprocess.run(
        [sys.executable, "-I", "-c", script],  # -I: isolated mode
        cwd=str(_HUB_ROOT.parent.parent.parent),
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    if out.returncode != 0:
        pytest.fail(
            f"E2E subprocess failed (rc={out.returncode}):\n"
            f"STDOUT (tail):\n{out.stdout[-2000:]}\n"
            f"STDERR (tail):\n{out.stderr[-2000:]}"
        )
    last_line = out.stdout.strip().splitlines()[-1]
    try:
        return json.loads(last_line)
    except json.JSONDecodeError as e:
        pytest.fail(
            f"Could not parse E2E JSON output: {e}\n"
            f"STDOUT (tail):\n{out.stdout[-2000:]}"
        )


@pytest.fixture(scope="module")
def e2e_result():
    return _run_e2e()


def test_fy27_all_ten_adapters_registered(e2e_result):
    assert e2e_result["adapters_registered"] == 10
    assert e2e_result["frameworks_in_report"] == 10


def test_fy27_deterministic_aggregate_hash(e2e_result):
    assert e2e_result["hashes_match"] is True
    assert e2e_result["hash_len"] == 64


def test_fy27_scope_engine_ghg_csrd_consistent(e2e_result):
    from decimal import Decimal

    ghg = Decimal(e2e_result["ghg_scope_1_kg"])
    csrd = Decimal(e2e_result["csrd_scope_1_kg"])
    assert ghg == csrd
    # Diesel 1000 gal + NG 500 therm -> ~12,922 kg Scope 1 (EPA 2024 factors)
    assert Decimal(12000) < ghg < Decimal(14000)


def test_fy27_eudr_under_review_and_in_gap_analysis(e2e_result):
    assert e2e_result["eudr_status"] == "under_review"
    assert e2e_result["eudr_commodities"] == 1
    assert e2e_result["eudr_in_gap_analysis"] is True


def test_fy27_taxonomy_aligned_share(e2e_result):
    aligned = float(e2e_result["taxonomy_aligned_share"])
    # 80M / 120M = 0.666...
    assert 0.65 < aligned < 0.67


def test_fy27_overall_status_partially_compliant(e2e_result):
    assert e2e_result["overall_status"] == "partially_compliant"


def test_fy27_unified_reports_generated_in_all_formats(e2e_result):
    reports = e2e_result["reports"]
    assert set(reports.keys()) >= {"json", "xml", "pdf"}
    for fmt, stat in reports.items():
        assert stat["size"] > 0, f"{fmt} report is empty"
        assert stat["hash_len"] == 64, f"{fmt} hash length wrong"
