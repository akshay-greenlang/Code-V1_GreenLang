# PACK-030: Quick Start Tutorial

**Pack:** PACK-030 Net Zero Reporting Pack
**Version:** 1.0.0
**Last Updated:** 2026-03-20

---

## Overview

This guide walks you through generating your first multi-framework climate disclosure report using PACK-030. By the end of this tutorial, you will have generated reports for all 7 supported frameworks from a single data aggregation.

**Time Required:** ~15 minutes
**Prerequisites:** PACK-030 installed and configured (see `installation.md`)

---

## Step 1: Initialize Your Organization

```python
from packs.net_zero.pack030.config import PACK030Config, BrandingConfig

# Create your organization configuration
config = PACK030Config(
    pack_id="PACK-030-net-zero-reporting",
    pack_version="1.0.0",
    frameworks={
        "SBTi": {"name": "SBTi", "version": "v1.1", "output_formats": ["PDF", "JSON"]},
        "CDP": {"name": "CDP", "version": "2025", "output_formats": ["Excel", "JSON"]},
        "TCFD": {"name": "TCFD", "version": "2023", "output_formats": ["PDF", "HTML"]},
        "GRI": {"name": "GRI 305", "version": "2016", "output_formats": ["PDF", "HTML"]},
        "ISSB": {"name": "IFRS S2", "version": "2023", "output_formats": ["PDF", "XBRL"]},
        "SEC": {"name": "SEC Climate", "version": "2024", "output_formats": ["PDF", "iXBRL"]},
        "CSRD": {"name": "ESRS E1", "version": "2024", "output_formats": ["PDF"]},
    },
    branding=BrandingConfig(
        logo_path="/path/to/company-logo.png",
        primary_color="#1E3A8A",
        secondary_color="#3B82F6",
        font_family="Arial, sans-serif",
        style="corporate",
    ),
    languages=["en"],
    assurance_enabled=True,
)
```

## Step 2: Aggregate Data from Source Systems

```python
from packs.net_zero.pack030.engines import DataAggregationEngine

# Initialize the aggregation engine
engine = DataAggregationEngine(config=config)

# Aggregate data from all source packs and applications
aggregated = await engine.aggregate_all(
    organization_id="your-org-uuid",
    reporting_period=("2025-01-01", "2025-12-31"),
)

# Check aggregation results
print(f"Data sources connected: {aggregated.source_count}")
print(f"Metrics collected: {aggregated.metric_count}")
print(f"Data completeness: {aggregated.completeness_score}%")
print(f"Reconciliation issues: {len(aggregated.reconciliation_issues)}")

# Review any data gaps
for gap in aggregated.gaps:
    print(f"  Gap: {gap.metric_name} - {gap.framework} - {gap.description}")
```

## Step 3: Generate a Single Framework Report (SBTi)

```python
from packs.net_zero.pack030.workflows import SBTiProgressWorkflow

# Initialize SBTi workflow
sbti_workflow = SBTiProgressWorkflow(config=config)

# Generate SBTi progress report
sbti_result = await sbti_workflow.execute(
    organization_id="your-org-uuid",
    reporting_year=2025,
    aggregated_data=aggregated,
)

# Access the report
print(f"Report ID: {sbti_result.report_id}")
print(f"Status: {sbti_result.status}")
print(f"Provenance Hash: {sbti_result.provenance_hash}")

# Save as PDF
await sbti_result.save_pdf("output/sbti_progress_2025.pdf")
print("SBTi progress report saved to output/sbti_progress_2025.pdf")
```

## Step 4: Generate All 7 Framework Reports

```python
from packs.net_zero.pack030.workflows import MultiFrameworkWorkflow

# Initialize multi-framework workflow
multi_workflow = MultiFrameworkWorkflow(config=config)

# Generate all reports in parallel (<10 seconds)
result = await multi_workflow.execute(
    organization_id="your-org-uuid",
    reporting_period=("2025-01-01", "2025-12-31"),
)

# Access individual reports
for framework, report in result.reports.items():
    print(f"{framework}: {report.status} ({report.generation_time:.1f}s)")

# Output:
# SBTi: completed (2.1s)
# CDP: completed (3.5s)
# TCFD: completed (2.8s)
# GRI: completed (1.9s)
# ISSB: completed (2.3s)
# SEC: completed (2.7s)
# CSRD: completed (3.1s)

# Check cross-framework consistency
print(f"Consistency score: {result.consistency_score}%")
print(f"Total generation time: {result.total_time:.1f}s")
```

## Step 5: Validate Reports

```python
from packs.net_zero.pack030.engines import ValidationEngine

validator = ValidationEngine(config=config)

# Validate all reports against framework schemas
for framework, report in result.reports.items():
    validation = await validator.validate(
        report_data=report.data,
        framework=framework,
    )
    print(f"{framework}: {validation.quality_score}% quality")
    print(f"  Errors: {len(validation.errors)}")
    print(f"  Warnings: {len(validation.warnings)}")
    print(f"  Completeness: {validation.completeness_score}%")
```

## Step 6: Generate Narratives

```python
from packs.net_zero.pack030.engines import NarrativeGenerationEngine

narrative_engine = NarrativeGenerationEngine(config=config)

# Generate TCFD governance narrative with citations
narrative = await narrative_engine.generate_narrative(
    section_type="governance",
    framework="TCFD",
    language="en",
    source_data=aggregated,
)

print(f"Narrative ({len(narrative.content)} chars):")
print(narrative.content[:500])
print(f"Citations: {len(narrative.citations)}")
print(f"Consistency score: {narrative.consistency_score}%")
```

## Step 7: Render in Multiple Formats

```python
from packs.net_zero.pack030.engines import FormatRenderingEngine

renderer = FormatRenderingEngine(config=config)

# Render TCFD report as PDF
tcfd_report = result.reports["TCFD"]
pdf_file = await renderer.render_pdf(tcfd_report, branding=config.branding)
await pdf_file.save("output/tcfd_disclosure_2025.pdf")

# Render as interactive HTML
html_file = await renderer.render_html(tcfd_report, branding=config.branding)
await html_file.save("output/tcfd_disclosure_2025.html")

# Render SEC disclosure as iXBRL
sec_report = result.reports["SEC"]
ixbrl_file = await renderer.render_ixbrl(sec_report, taxonomy="SEC-2024")
await ixbrl_file.save("output/sec_climate_disclosure_2025.html")

# Export CDP questionnaire as Excel
cdp_report = result.reports["CDP"]
excel_file = await renderer.render_excel(cdp_report)
await excel_file.save("output/cdp_questionnaire_2025.xlsx")
```

## Step 8: Package Assurance Evidence

```python
from packs.net_zero.pack030.engines import AssurancePackagingEngine

assurance_engine = AssurancePackagingEngine(config=config)

# Create ISAE 3410 evidence bundle
evidence_bundle = await assurance_engine.package_evidence(
    report_ids=[r.report_id for r in result.reports.values()],
    audit_scope="full",
    include_lineage_diagrams=True,
    include_methodology_docs=True,
    include_control_matrix=True,
)

# Save evidence bundle
await evidence_bundle.save_zip("output/assurance_evidence_2025.zip")
print(f"Evidence bundle: {evidence_bundle.file_count} files")
print(f"Provenance hashes: {evidence_bundle.provenance_count}")
print(f"Control matrix items: {evidence_bundle.control_matrix_count}")
```

## Step 9: Launch Executive Dashboard

```python
from packs.net_zero.pack030.engines import DashboardGenerationEngine

dashboard_engine = DashboardGenerationEngine(config=config)

# Generate executive dashboard
dashboard = await dashboard_engine.generate_executive_dashboard(
    organization_id="your-org-uuid",
    report_data=result,
)

# Save as standalone HTML
await dashboard.save("output/executive_dashboard_2025.html")
print("Dashboard saved - open in browser to view")

# Generate stakeholder-specific views
for stakeholder in ["investor", "regulator", "customer", "employee"]:
    view = await dashboard_engine.generate_stakeholder_view(
        stakeholder_type=stakeholder,
        report_data=result,
    )
    await view.save(f"output/{stakeholder}_dashboard_2025.html")
    print(f"{stakeholder.title()} view saved")
```

## Step 10: Review and Approve

```python
from packs.net_zero.pack030.workflows import ApprovalWorkflow

# Submit reports for internal review
approval = ApprovalWorkflow(config=config)

for framework, report in result.reports.items():
    await approval.submit_for_review(
        report_id=report.report_id,
        reviewers=["reviewer1@company.com", "reviewer2@company.com"],
        deadline_days=7,
    )
    print(f"{framework} report submitted for review")
```

---

## Next Steps

1. **Customize branding**: Update `BrandingConfig` with your company logo and colors
2. **Add languages**: Add `"de"`, `"fr"`, `"es"` to the `languages` list for multilingual reports
3. **Configure presets**: Use pre-built presets for specific workflows (see `configuration.md`)
4. **Set up scheduled generation**: Configure cron jobs for periodic report generation
5. **Integrate with CI/CD**: Add report generation to your deployment pipeline
6. **Enable notifications**: Configure Slack/email alerts for deadline reminders

---

## Common Patterns

### Using Configuration Presets

```python
from packs.net_zero.pack030.config import load_preset

# Load a pre-built preset
config = load_preset("cdp_alist")  # Optimized for CDP A-list scoring
config = load_preset("sec_10k")    # Optimized for SEC compliance
config = load_preset("multi_framework")  # All 7 frameworks
```

### Generating Reports via API

```bash
# Generate SBTi report via REST API
curl -X POST http://localhost:8030/api/v1/reports/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "organization_id": "your-org-uuid",
    "framework": "SBTi",
    "reporting_period": {"start": "2025-01-01", "end": "2025-12-31"},
    "output_formats": ["PDF", "JSON"]
  }'
```

### Scheduling Multi-Framework Reports

```python
from packs.net_zero.pack030.scheduler import ReportScheduler

scheduler = ReportScheduler(config=config)

# Schedule quarterly reporting
await scheduler.schedule(
    organization_id="your-org-uuid",
    frequency="quarterly",
    frameworks=["SBTi", "CDP", "TCFD"],
    notify=["sustainability@company.com"],
)
```

---

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| Data aggregation timeout | Increase `DATA_AGGREGATION_TIMEOUT` env var |
| Missing metrics | Check prerequisite pack connections |
| XBRL validation errors | Update XBRL taxonomy cache |
| PDF rendering fails | Verify WeasyPrint system dependencies |
| Consistency score low | Review narrative contradictions in validation report |

See `docs/user/troubleshooting.md` for detailed solutions.

---

*Built with GreenLang Platform - Zero-Hallucination Climate Intelligence*
