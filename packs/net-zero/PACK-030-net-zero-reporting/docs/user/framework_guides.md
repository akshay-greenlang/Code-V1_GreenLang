# PACK-030: Framework-Specific Guides

**Pack:** PACK-030 Net Zero Reporting Pack
**Version:** 1.0.0
**Last Updated:** 2026-03-20

---

## Table of Contents

1. [SBTi Annual Progress Report](#1-sbti-annual-progress-report)
2. [CDP Climate Change Questionnaire](#2-cdp-climate-change-questionnaire)
3. [TCFD Disclosure Report](#3-tcfd-disclosure-report)
4. [GRI 305 Emissions Disclosure](#4-gri-305-emissions-disclosure)
5. [ISSB IFRS S2 Climate Disclosure](#5-issb-ifrs-s2-climate-disclosure)
6. [SEC Climate Disclosure (10-K)](#6-sec-climate-disclosure-10-k)
7. [CSRD ESRS E1 Climate Change](#7-csrd-esrs-e1-climate-change)

---

## 1. SBTi Annual Progress Report

### Overview

The SBTi Progress Report workflow generates the annual progress disclosure required by the Science Based Targets initiative for organizations with validated targets under the SBTi Corporate Net-Zero Standard v1.1.

### What PACK-030 Generates

- Target description with base year recalculation status
- Progress table: base year vs. current year vs. target year
- Scope 1, Scope 2 (location and market-based), and Scope 3 breakdowns
- Variance explanation for any deviation from linear trajectory
- Forward-looking projection to target year
- Evidence of reduction initiatives undertaken

### Data Sources

| Source | Data Points |
|--------|------------|
| GL-SBTi-APP | Validated target details, base year, target year, ambition level |
| PACK-021 | Baseline emissions inventory |
| PACK-029 | Interim target progress, variance analysis |
| GL-GHG-APP | Current year GHG inventory |

### Usage

```python
from packs.net_zero.pack030.workflows import SBTiProgressWorkflow

workflow = SBTiProgressWorkflow(config=config)
result = await workflow.execute(
    organization_id="your-org-uuid",
    reporting_year=2025,
)

# Save outputs
await result.save_pdf("output/sbti_progress_2025.pdf")
await result.save_json("output/sbti_progress_2025.json")
```

### Output Structure

```
SBTi Progress Report
+-- 1. Organization Overview
+-- 2. Target Description
|   +-- Near-term target (Scope 1+2)
|   +-- Near-term target (Scope 3)
|   +-- Long-term net-zero target
+-- 3. Base Year Emissions
+-- 4. Current Year Emissions
+-- 5. Progress Summary Table
+-- 6. Variance Explanation
+-- 7. Reduction Initiatives
+-- 8. Forward Projection
+-- 9. Methodology Notes
```

### SBTi-Specific Validation

PACK-030 validates the report against SBTi requirements:
- Base year recalculation trigger check (>5% change)
- Scope coverage completeness (Scope 1+2 minimum 95%)
- Scope 3 coverage (minimum 67% of total Scope 3)
- Target ambition alignment (1.5C or WB2C pathway)
- Annual linear reduction rate verification

---

## 2. CDP Climate Change Questionnaire

### Overview

The CDP Questionnaire workflow generates responses for the CDP Climate Change questionnaire modules C0 through C12, supporting A-list scoring ambitions.

### What PACK-030 Generates

| Module | Title | Content Type |
|--------|-------|-------------|
| C0 | Introduction | Organization details |
| C1 | Governance | Board oversight, management responsibility |
| C2 | Risks and Opportunities | Climate risk identification, financial impact |
| C3 | Business Strategy | Strategic integration of climate issues |
| C4 | Targets and Performance | Emission reduction targets and progress |
| C5 | Emissions Methodology | Reporting methodology, boundary |
| C6 | Emissions Data (Scope 1/2) | Scope 1 and 2 emissions breakdown |
| C7 | Emissions Data (Scope 3) | Scope 3 category breakdown |
| C8 | Energy | Energy consumption, renewable energy |
| C9 | Additional Metrics | Intensity metrics |
| C10 | Verification | Third-party verification status |
| C11 | Carbon Pricing | Internal carbon pricing mechanisms |
| C12 | Engagement | Value chain engagement activities |

### Data Sources

| Source | Modules Covered |
|--------|----------------|
| GL-CDP-APP | Historical responses, scoring data |
| GL-GHG-APP | C5, C6, C7 emissions data |
| PACK-029 | C4 targets and progress |
| GL-TCFD-APP | C2, C3 risks and strategy |

### Usage

```python
from packs.net_zero.pack030.workflows import CDPQuestionnaireWorkflow

workflow = CDPQuestionnaireWorkflow(config=config)
result = await workflow.execute(
    organization_id="your-org-uuid",
    cdp_year=2025,
)

# Completeness scoring
print(f"Overall completeness: {result.completeness_score}%")
for module in result.modules:
    print(f"  {module.id}: {module.completeness}% ({module.questions_answered}/{module.questions_total})")

# Export to Excel for review
await result.save_excel("output/cdp_questionnaire_2025.xlsx")
```

### CDP Scoring Optimization

PACK-030 provides scoring guidance:
- **A-list requirements**: Identifies gaps that would prevent A-list scoring
- **Response quality**: Suggests improvements to text responses
- **Data completeness**: Highlights missing data points
- **Narrative consistency**: Ensures C1-C4 narratives align with C6-C7 data
- **Verification status**: Recommends verification level for scoring benefit

---

## 3. TCFD Disclosure Report

### Overview

The TCFD Disclosure workflow generates a 4-pillar report aligned with the Task Force on Climate-related Financial Disclosures Recommendations (2023).

### What PACK-030 Generates

| Pillar | Recommended Disclosures |
|--------|------------------------|
| **Governance** | a) Board oversight of climate risks and opportunities; b) Management's role in assessing and managing |
| **Strategy** | a) Climate risks and opportunities identified; b) Impact on business, strategy, financial planning; c) Resilience under different scenarios |
| **Risk Management** | a) Processes for identifying and assessing; b) Processes for managing; c) Integration with overall risk management |
| **Metrics & Targets** | a) Metrics used to assess climate risks; b) Scope 1, 2, 3 emissions; c) Targets used and performance against targets |

### Data Sources

| Source | Pillar |
|--------|--------|
| GL-TCFD-APP | Strategy (scenario analysis), Risk Management |
| GL-GHG-APP | Metrics & Targets (Scope 1/2/3 emissions) |
| PACK-029 | Metrics & Targets (targets and progress) |
| PACK-022 | Strategy (reduction initiatives) |

### Usage

```python
from packs.net_zero.pack030.workflows import TCFDDisclosureWorkflow

workflow = TCFDDisclosureWorkflow(config=config)
result = await workflow.execute(
    organization_id="your-org-uuid",
    reporting_period=("2025-01-01", "2025-12-31"),
    include_scenario_analysis=True,
)

# Save as executive PDF with charts
await result.save_pdf("output/tcfd_disclosure_2025.pdf")

# Save as interactive HTML with drill-down
await result.save_html("output/tcfd_disclosure_2025.html")
```

### TCFD Best Practices

PACK-030 follows TCFD best practices:
- **Scenario analysis**: 1.5C, 2C, and 4C scenarios with physical and transition risks
- **Forward-looking statements**: Clearly labeled with assumptions documented
- **Financial quantification**: Climate risk financial impacts where available
- **Cross-referencing**: Links to CDP, GRI, and ISSB disclosures for consistency

---

## 4. GRI 305 Emissions Disclosure

### Overview

The GRI 305 Disclosure workflow generates emissions disclosures aligned with GRI 305 (2016) covering all seven sub-standards.

### What PACK-030 Generates

| Standard | Title | Required Metrics |
|----------|-------|-----------------|
| 305-1 | Direct (Scope 1) GHG emissions | Gross Scope 1, gases included, biogenic CO2 |
| 305-2 | Energy indirect (Scope 2) | Location-based, market-based |
| 305-3 | Other indirect (Scope 3) | Categories 1-15 breakdown |
| 305-4 | GHG emissions intensity | Organization-specific ratio |
| 305-5 | Reduction of GHG emissions | Reduction initiatives and impact |
| 305-6 | Emissions of ODS | Ozone-depleting substances |
| 305-7 | NOx, SOx, and other | Significant air emissions |

### Usage

```python
from packs.net_zero.pack030.workflows import GRI305Workflow

workflow = GRI305Workflow(config=config)
result = await workflow.execute(
    organization_id="your-org-uuid",
    reporting_year=2025,
)

# Generate GRI Content Index
content_index = result.generate_content_index()
await content_index.save("output/gri_content_index_2025.pdf")

# Save full 305 disclosure
await result.save_pdf("output/gri_305_disclosure_2025.pdf")
```

### GRI Content Index

PACK-030 automatically generates the GRI Content Index table showing:
- Disclosure number and title
- Page reference in report
- Omission reason (if applicable)
- External assurance status

---

## 5. ISSB IFRS S2 Climate Disclosure

### Overview

The ISSB IFRS S2 workflow generates climate-related disclosures aligned with IFRS S2 Climate-related Disclosures (2023), including industry-specific metrics from SASB standards.

### What PACK-030 Generates

| Section | IFRS S2 Reference | Content |
|---------|-------------------|---------|
| Governance | Paragraphs 5-7 | Board oversight, management roles |
| Strategy | Paragraphs 8-22 | Climate risks, opportunities, transition plans, resilience |
| Risk Management | Paragraphs 23-27 | Risk identification, assessment, management, integration |
| Metrics & Targets | Paragraphs 28-37 | Cross-industry metrics, industry-specific (SASB), targets |

### XBRL Tagging

PACK-030 automatically applies XBRL taxonomy tags to all quantitative disclosures:

```python
from packs.net_zero.pack030.workflows import ISSBWorkflow

workflow = ISSBWorkflow(config=config)
result = await workflow.execute(
    organization_id="your-org-uuid",
    fiscal_period=("2025-01-01", "2025-12-31"),
    industry="manufacturing",  # For SASB industry metrics
)

# Save with XBRL tagging
await result.save_pdf("output/issb_s2_2025.pdf")
await result.save_xbrl("output/issb_s2_2025.xbrl")
```

---

## 6. SEC Climate Disclosure (10-K)

### Overview

The SEC Climate Disclosure workflow generates the climate-related sections for 10-K filings, including XBRL/iXBRL tagging required by the SEC climate disclosure rules.

### What PACK-030 Generates

| Section | Regulation | Content |
|---------|-----------|---------|
| Item 1 | Business Description | Climate risks in business description |
| Item 1A | Risk Factors | Climate-related risk factors |
| Item 7 | MD&A | Climate impacts in Management's Discussion and Analysis |
| Reg S-K 1502 | Governance | Climate risk governance |
| Reg S-K 1503 | Strategy | Climate risk strategy and financial impacts |
| Reg S-K 1504 | Risk Management | Climate risk management processes |
| Reg S-K 1505 | Metrics | Scope 1 and 2 emissions |
| Reg S-K 1506 | Targets | Climate targets and progress |

### XBRL/iXBRL Requirements

PACK-030 generates both XBRL and iXBRL (inline XBRL) outputs:
- **XBRL**: Machine-readable XML file for automated processing
- **iXBRL**: Human-readable HTML with embedded XBRL tags for dual-purpose filing

```python
from packs.net_zero.pack030.workflows import SECClimateWorkflow

workflow = SECClimateWorkflow(config=config)
result = await workflow.execute(
    organization_id="your-org-uuid",
    fiscal_year=2025,
    filer_type="large_accelerated",
)

# Generate all SEC outputs
await result.save_pdf("output/sec_climate_2025.pdf")
await result.save_xbrl("output/sec_climate_2025.xbrl")
await result.save_ixbrl("output/sec_climate_2025_ixbrl.html")

# Generate attestation report template
await result.save_attestation("output/sec_attestation_2025.pdf")
```

### SEC Taxonomy Validation

PACK-030 validates all XBRL tags against the official SEC taxonomy:
- Element name verification
- Context reference validation
- Unit reference validation
- Decimal precision checks
- Taxonomy version compatibility

---

## 7. CSRD ESRS E1 Climate Change

### Overview

The CSRD ESRS E1 workflow generates the Climate Change disclosure required by the Corporate Sustainability Reporting Directive (CSRD), covering all 9 disclosure requirements of ESRS E1.

### What PACK-030 Generates

| Disclosure | Title | Content |
|-----------|-------|---------|
| E1-1 | Transition plan for climate change mitigation | Transition plan aligned with 1.5C, implementation status |
| E1-2 | Policies related to climate change | Climate policies, integration with business strategy |
| E1-3 | Actions and resources | Actions taken, resources committed, expected outcomes |
| E1-4 | Targets related to climate change | GHG reduction targets, progress tracking |
| E1-5 | Energy consumption and mix | Total energy, renewable share, energy intensity |
| E1-6 | Gross Scope 1, 2, 3 emissions | Disaggregated emissions with methodology |
| E1-7 | GHG removals and carbon credits | Removal activities, credit quality, retirement |
| E1-8 | Internal carbon pricing | Carbon pricing mechanisms, coverage, price levels |
| E1-9 | Anticipated financial effects | Financial effects of physical and transition risks |

### Digital Taxonomy

PACK-030 applies CSRD digital taxonomy tags to all data points for machine-readable submission:

```python
from packs.net_zero.pack030.workflows import CSRDWorkflow

workflow = CSRDWorkflow(config=config)
result = await workflow.execute(
    organization_id="your-org-uuid",
    reporting_period=("2025-01-01", "2025-12-31"),
)

# Save with digital taxonomy
await result.save_pdf("output/csrd_e1_2025.pdf")
await result.save_digital_taxonomy("output/csrd_e1_2025_taxonomy.xml")
```

### CSRD Double Materiality

PACK-030 supports the double materiality assessment required by CSRD:
- **Impact materiality**: Organization's impact on climate
- **Financial materiality**: Climate's impact on the organization
- **Cross-reference**: Links E1 disclosures to double materiality assessment results

### Multi-Language Support

CSRD reports can be generated in the official languages of EU member states:

```python
# Generate reports in multiple languages
for lang in ["en", "de", "fr", "es"]:
    result = await workflow.execute(
        organization_id="your-org-uuid",
        reporting_period=("2025-01-01", "2025-12-31"),
        language=lang,
    )
    await result.save_pdf(f"output/csrd_e1_2025_{lang}.pdf")
```

---

## Cross-Framework Consistency

### Consistency Validation

When generating reports for multiple frameworks, PACK-030 automatically validates consistency:

```python
from packs.net_zero.pack030.engines import ValidationEngine

validator = ValidationEngine(config=config)

# Run cross-framework consistency check
consistency = await validator.validate_consistency(
    reports=multi_framework_result.reports,
)

print(f"Overall consistency: {consistency.score}%")
for issue in consistency.issues:
    print(f"  {issue.severity}: {issue.description}")
    print(f"    Framework A ({issue.framework_a}): {issue.value_a}")
    print(f"    Framework B ({issue.framework_b}): {issue.value_b}")
```

### Common Consistency Checks

| Check | Frameworks | Description |
|-------|-----------|-------------|
| Scope 1 total | All 7 | Same Scope 1 total reported across all frameworks |
| Scope 2 approach | TCFD, GRI, ISSB, SEC | Location vs. market-based consistency |
| Scope 3 categories | CDP, GRI, ISSB | Same categories reported with same values |
| Target description | SBTi, CDP, TCFD | Consistent target wording and metrics |
| Base year | SBTi, TCFD, CSRD | Same base year emissions reported |
| Methodology | All 7 | Same GHG accounting methodology referenced |
| Organizational boundary | All 7 | Same consolidation approach |

---

## Framework Comparison Matrix

| Feature | SBTi | CDP | TCFD | GRI | ISSB | SEC | CSRD |
|---------|------|-----|------|-----|------|-----|------|
| Scope 1 required | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Scope 2 required | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Scope 3 required | Yes* | Yes | Yes | Recommended | Yes | No** | Yes |
| Scenario analysis | No | Recommended | Yes | No | Yes | No | Yes |
| XBRL tagging | No | No | No | No | Yes | Yes | Yes |
| Narrative content | Limited | Extensive | Moderate | Limited | Moderate | Moderate | Extensive |
| Financial quantification | No | Optional | Recommended | No | Yes | Yes | Yes |
| Assurance required | No | Recommended | Varies | Optional | Varies | Yes*** | Yes |

\* 67% of Scope 3 emissions
\** Scope 3 disclosure voluntary under current rule
\*** Large accelerated filers, limited assurance initially

---

*Built with GreenLang Platform - Zero-Hallucination Climate Intelligence*
