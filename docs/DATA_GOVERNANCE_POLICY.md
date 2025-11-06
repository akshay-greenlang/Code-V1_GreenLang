# FuelAgentAI Data Governance Policy

**Version:** 1.0
**Effective Date:** 2025-10-24
**Owner:** Data Governance Board
**Review Frequency:** Quarterly
**Status:** Active

---

## Executive Summary

This policy establishes governance rules for emission factor data in FuelAgentAI, ensuring **accuracy, auditability, consistency, and regulatory compliance**. It defines decision-making authority, update procedures, conflict resolution, and quality standards.

**Core Principles:**
1. **Accuracy First:** Data quality over convenience
2. **Transparency:** All decisions documented and traceable
3. **Regulatory Alignment:** Official sources preferred
4. **Historical Integrity:** Past calculations remain reproducible
5. **Customer Trust:** No surprise changes to reported emissions

---

## 1. Governance Structure

### 1.1 Data Governance Board

**Members:**
- **Chair:** CTO (final decision authority)
- **Data Lead:** Manages factor database, proposes updates
- **Compliance Officer:** Ensures regulatory alignment
- **Technical Lead:** Assesses implementation feasibility
- **Customer Success Lead:** Represents customer impact
- **External Advisor:** Subject matter expert (carbon accounting) - quarterly reviews

**Meeting Frequency:** Monthly (or ad-hoc for urgent changes)

**Decision Quorum:** 3 of 5 members (including CTO or Compliance Officer)

---

### 1.2 Roles & Responsibilities

| Role | Responsibilities | Authority Level |
|------|------------------|-----------------|
| **CTO** | Final approval on major changes (>5% emission delta) | High |
| **Data Lead** | Factor ingestion, validation, routine updates | Medium |
| **Compliance Officer** | Regulatory review, licensing compliance | Medium |
| **Technical Lead** | Schema changes, database migrations | Medium |
| **Customer Success** | Customer impact assessment, communication | Low |
| **QA Engineer** | Validation testing, golden set parity | Low |

---

## 2. Emission Factor Lifecycle

```
┌─────────────┐
│   PROPOSAL  │ ← Data Lead identifies new source or update
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  VALIDATION │ ← QA validates accuracy, DQS, licensing
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   REVIEW    │ ← Governance Board reviews (if material change)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  APPROVAL   │ ← CTO approves (if > 5% delta)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ DEPLOYMENT  │ ← Technical Lead deploys with version increment
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ MONITORING  │ ← Monitor customer impact, support tickets
└─────────────┘
```

---

## 3. Source Precedence Rules

### 3.1 Official Precedence Hierarchy

When multiple sources provide factors for the **same fuel, unit, geography, and time period**, use this precedence:

**Tier 1: National Regulatory (Highest)**
1. EPA (United States)
2. UK DESNZ/BEIS (United Kingdom)
3. EEA (European Union)
4. National authorities (Australia NGA, India BEE, etc.)

**Tier 2: International Standards**
5. IEA (International Energy Agency)
6. IPCC (country-specific from EFDB)

**Tier 3: Research & Industry**
7. GREET (Argonne National Lab)
8. Ecoinvent (if licensed)
9. Peer-reviewed academic publications

**Tier 4: Default/Fallback**
10. IPCC Tier 1 defaults (global)

**Rationale:**
- Regulatory sources have legal weight and are auditor-accepted
- National > International > Global for geographic specificity
- Official > Commercial > Academic for trust

---

### 3.2 Tie-Breaking Rules

If two sources at the **same tier** provide conflicting factors:

1. **Most recent publication year** (2024 > 2023)
2. **Higher data quality score** (DQS 4.5 > DQS 4.0)
3. **More transparent methodology** (Tier 3 > Tier 1 IPCC)
4. **Official government > NGO > Academic**
5. **Smaller uncertainty range** (±5% > ±10%)

**Example:**
```
Conflict:
- EPA 2024: diesel = 10.21 kgCO2e/gallon (DQS: 4.4, ±5%)
- IEA 2024: diesel = 10.18 kgCO2e/gallon (DQS: 4.2, ±8%)

Decision: Use EPA (same year, higher DQS, smaller uncertainty)
```

---

### 3.3 Source Override Exceptions

Customers may request **alternative factors** in these cases:

| Scenario | Override Allowed? | Approval Required | Notes |
|----------|-------------------|-------------------|-------|
| Facility-specific measurement | ✅ Yes | Customer Success | Must provide measurement report |
| Supplier-specific factor | ✅ Yes | Customer Success | Must provide EPD/PCR |
| Regional factor (state/grid zone) | ✅ Yes | Data Lead | If more specific than default |
| Historical factor (past year) | ✅ Yes | Data Lead | For restating reports |
| Proprietary factor (paid DB) | ⚠️ Case-by-case | CTO | Licensing & cost review |
| Customer preference (no basis) | ❌ No | - | Must use official source |

**Custom Factor Requirements:**
- Source documentation (report, certificate, etc.)
- Methodology disclosure
- Uncertainty quantification
- DQS self-assessment
- Signed attestation (for audited reports)

---

## 4. Update Procedures

### 4.1 Routine Updates (Quarterly)

**Trigger:** EPA, IEA, BEIS publish annual updates

**Process:**
1. **Week 1:** Data Lead downloads new factors
2. **Week 2:** QA validates against golden set
3. **Week 3:** Calculate emission deltas (% change vs. previous)
4. **Week 4:** Deploy if **all deltas < 5%** (automatic approval)

**Automatic Approval Criteria:**
- ✅ All emission factor changes < 5%
- ✅ Same source organization
- ✅ Same methodology
- ✅ No licensing changes
- ✅ Golden set tests pass (100%)

**Example:**
```
EPA 2024 Update:
- Diesel: 10.21 → 10.23 kgCO2e/gallon (+0.2%) ✅ Auto-approve
- Natural gas: 5.30 → 5.28 kgCO2e/therm (-0.4%) ✅ Auto-approve
- Electricity (US avg): 0.385 → 0.378 (-1.8%) ✅ Auto-approve
```

---

### 4.2 Material Changes (> 5% Delta)

**Trigger:** Factor change > 5% or methodology change

**Process:**
1. **Proposal:** Data Lead prepares change proposal
2. **Impact Analysis:** Estimate affected customers and emission deltas
3. **Review:** Governance Board meeting (within 5 business days)
4. **Approval:** CTO approves with customer communication plan
5. **Notification:** Email customers 30 days before deployment
6. **Deployment:** Staged rollout with monitoring

**Change Proposal Template:**
```markdown
## Emission Factor Change Proposal

**Factor ID:** EF:US:coal:2024:v2
**Current Value:** 2086 kgCO2e/ton
**Proposed Value:** 1950 kgCO2e/ton
**Change:** -6.5% (material change)

**Rationale:**
EPA updated coal emission factor using improved measurement methodology (Tier 3 vs. Tier 1).

**Source:**
EPA (2024). Emission Factors for GHG Inventories, v2.
https://www.epa.gov/...

**Impact Analysis:**
- Affected customers: 127 (15% of active users)
- Avg emission reduction: -8.2% for coal users
- Max customer impact: -12% (heavy coal user)

**DQS:**
- Previous: 4.2 (Tier 1 estimate)
- New: 4.8 (Tier 3 measurement)
- Improvement: +0.6

**Recommendation:**
Approve and deploy with 30-day customer notice.

**Communication Plan:**
1. Email affected customers (127) - Day 0
2. Post changelog - Day 0
3. Office hours for Q&A - Day 7
4. Deploy - Day 30
```

**Approval Decision Matrix:**

| Delta | DQS Change | Regulatory Source | Approval |
|-------|------------|-------------------|----------|
| < 5% | Any | Any | Auto-approve |
| 5-10% | +0.3 or better | Yes | Board approval |
| 5-10% | < +0.3 | Yes | Board + CTO |
| > 10% | +0.5 or better | Yes | Board + CTO + customer vote |
| > 10% | < +0.5 | Any | Reject (unless regulatory mandate) |

**Customer Vote:**
For changes > 10%, survey affected customers:
- "We're updating coal emission factors based on improved EPA data (Tier 3 measurement). This will reduce your reported emissions by ~8%. Approve update?"
- If > 75% approve → Deploy
- If < 75% → Offer opt-in flag for early adopters

---

### 4.3 Emergency Corrections

**Trigger:** Error discovered in deployed factor (e.g., wrong unit, data entry mistake)

**Process:**
1. **Incident Report:** Data Lead files incident
2. **Root Cause Analysis:** Determine how error occurred
3. **Immediate Fix:** Deploy correction within 24 hours
4. **Customer Notification:** Email all affected users
5. **Restatement Option:** Offer to re-run past calculations
6. **Process Improvement:** Update validation checks

**Example Incident:**
```
INCIDENT-2024-11: Propane factor unit error

**Error:** Propane factor listed as 5.76 kgCO2e/gallon (correct)
         but system applied as kgCO2e/liter (4× overstatement)

**Impact:** 23 customers overstated propane emissions by ~400%
           for 2 weeks (2024-10-15 to 2024-10-29)

**Root Cause:** Unit conversion bug in ingestion script

**Fix:**
- Corrected factor metadata (deployed 2024-10-29 18:00 UTC)
- Emailed 23 customers with corrected values
- Offered free restatement service

**Prevention:**
- Added unit validation test (kg vs. liter vs. gallon)
- Require peer review for all factor changes
- Automated sanity checks (flag if emission delta > 50%)
```

---

## 5. Version Control & Historical Integrity

### 5.1 Immutable Past Principle

**Rule:** Historical emission factors are **NEVER retroactively changed**.

**Rationale:**
- Audited reports must remain reproducible
- Regulatory filings cannot be changed post-submission
- Customer trust requires consistency

**Implementation:**
- Each factor has `valid_from` and `valid_to` dates
- New factors create **new versions**, not overwrites
- API allows querying factors "as of" a specific date

**Example:**
```python
# Factor versions for US diesel
EF:US:diesel:2023:v1  valid_from: 2023-01-01, valid_to: 2023-12-31
EF:US:diesel:2024:v1  valid_from: 2024-01-01, valid_to: None (current)

# If 2024 factor is corrected:
EF:US:diesel:2024:v2  valid_from: 2024-01-01, valid_to: None (replaces v1)
                      supersedes: "EF:US:diesel:2024:v1"

# Historical query (2023 report):
factor = db.get_factor("diesel", "gallons", "US", as_of_date="2023-06-15")
# Returns: EF:US:diesel:2023:v1 (not current factor)
```

---

### 5.2 Versioning Scheme

**Format:** `EF:{geography}:{fuel_type}:{year}:v{version}`

**Examples:**
- `EF:US:diesel:2024:v1` - First version for 2024
- `EF:US:diesel:2024:v2` - Correction or methodology update for 2024
- `EF:UK:natural_gas:WTT:2024:v1` - UK natural gas, WTT boundary

**Version Increment Triggers:**
- **Patch (v1 → v2):** Correction of error, same time period
- **Minor (2024 → 2025):** Annual update, same methodology
- **Major (combustion → WTW):** Boundary change, different value

---

### 5.3 Deprecation & Archival

**Deprecated Factors:**
Factors older than 5 years are marked deprecated but **remain queryable**.

```python
factor = db.get_factor("diesel", "gallons", "US", as_of_date="2018-06-15")
# Returns:
{
    "factor_id": "EF:US:diesel:2018:v1",
    "co2e_kg_per_unit": 10.15,
    "deprecated": True,
    "deprecation_reason": "Factor > 5 years old. Use latest for new calculations.",
    "replacement_factor_id": "EF:US:diesel:2024:v1"
}
```

**Archival Policy:**
- Factors > 10 years: Moved to archive database (slower query)
- Factors > 20 years: Require special request for access
- Never deleted (regulatory retention requirement)

---

## 6. Conflict Resolution

### 6.1 Source Conflicts

**Scenario:** EPA and IEA publish different factors for the same fuel/year.

**Resolution Process:**
1. **Check Tier:** Use higher-tier source (EPA > IEA for US)
2. **If Same Tier:** Apply tie-breakers (Section 3.2)
3. **Document Decision:** Record in factor metadata

**Example:**
```json
{
  "factor_id": "EF:US:diesel:2024:v1",
  "co2e_kg_per_unit": 10.21,
  "source_org": "EPA",
  "notes": "Conflict: IEA reported 10.18 kgCO2e/gallon. Selected EPA per precedence rule (national > international)."
}
```

---

### 6.2 Customer Disputes

**Scenario:** Customer claims our factor is wrong.

**Resolution Process:**
1. **Acknowledge:** Within 24 hours
2. **Investigate:** Data Lead reviews source and calculation
3. **Response Options:**
   - **Correct:** If we're wrong, issue correction (Section 4.3)
   - **Explain:** If we're right, provide citation and methodology
   - **Alternative:** Offer custom factor override (Section 3.3)
4. **Escalate:** If unresolved, Governance Board review

**SLA:**
- Tier 1 (critical): 24-hour response
- Tier 2 (important): 3-business-day response
- Tier 3 (question): 5-business-day response

---

### 6.3 Regulatory Conflicts

**Scenario:** CSRD requires different factor than GHG Protocol.

**Resolution:**
- **Maintain Multiple Factors:** Store both in database
- **Tag with Compliance Framework:** `compliance_frameworks: ["CSRD"]`
- **API Parameter:** Allow client to specify framework preference

```python
# Example: Different factors for location-based vs. market-based Scope 2
EF:US:electricity:location:2024:v1  # Grid average (GHG Protocol location-based)
EF:US:electricity:market:2024:v1    # Residual mix (GHG Protocol market-based)
```

---

## 7. Data Quality Standards

### 7.1 Minimum Quality Thresholds

All factors must meet these minimums:

| Criterion | Minimum | Preferred | Rationale |
|-----------|---------|-----------|-----------|
| **DQS Overall** | ≥ 3.0 | ≥ 4.0 | GHG Protocol "good quality" |
| **Uncertainty** | ≤ 25% | ≤ 10% | Usable for reporting |
| **Source Age** | ≤ 5 years | ≤ 2 years | Reasonably current |
| **Methodology** | Documented | Peer-reviewed | Auditable |
| **Geographic Match** | Regional | Country/state | Representative |

**Rejection Criteria:**
- ❌ DQS < 3.0 (unless only available source)
- ❌ Uncertainty > 50% (too unreliable)
- ❌ Source > 10 years old (outdated)
- ❌ Methodology unknown (not auditable)

---

### 7.2 Validation Checks (Automated)

Before deploying any factor:

```python
def validate_emission_factor(factor: EmissionFactorRecord) -> List[str]:
    """
    Automated validation checks.
    Returns list of warnings (empty if all pass).
    """
    warnings = []

    # 1. Reasonable value range
    if factor.gwp_100yr.co2e_total < 0.001:
        warnings.append("Factor suspiciously low (< 0.001 kgCO2e/unit)")
    if factor.gwp_100yr.co2e_total > 10000:
        warnings.append("Factor suspiciously high (> 10,000 kgCO2e/unit)")

    # 2. GWP calculation
    expected_co2e = (
        factor.vectors.CO2 +
        factor.vectors.CH4 * factor.gwp_100yr.CH4_gwp +
        factor.vectors.N2O * factor.gwp_100yr.N2O_gwp
    )
    if abs(factor.gwp_100yr.co2e_total - expected_co2e) > 0.01:
        warnings.append("CO2e mismatch: calculated != declared")

    # 3. DQS threshold
    if factor.dqs.overall_score < 3.0:
        warnings.append("DQS below minimum threshold (< 3.0)")

    # 4. Source recency
    years_old = 2024 - factor.provenance.source_year
    if years_old > 5:
        warnings.append(f"Source is {years_old} years old (> 5 year threshold)")

    # 5. Required fields
    if not factor.provenance.citation:
        warnings.append("Missing citation")

    # 6. Licensing
    if not factor.license_info.redistribution_allowed:
        warnings.append("Redistribution not allowed - cannot use in customer-facing reports")

    # 7. Comparison to previous version (if exists)
    previous = db.get_factor(factor.fuel_type, factor.unit, factor.geography,
                             as_of_date=factor.valid_from - timedelta(days=1))
    if previous:
        delta_pct = abs(factor.gwp_100yr.co2e_total - previous.gwp_100yr.co2e_total) / previous.gwp_100yr.co2e_total * 100
        if delta_pct > 20:
            warnings.append(f"Large change from previous version: {delta_pct:.1f}% (flag for review)")

    return warnings
```

**Deployment Decision:**
- **0 warnings:** Auto-deploy (routine update)
- **1-2 warnings:** Data Lead review required
- **3+ warnings:** Governance Board review required
- **Critical warning** (missing citation, DQS < 2.0): Reject

---

## 8. Customer Communication

### 8.1 Transparency Requirements

For any emission factor change affecting > 10 customers:

**Required Communications:**
1. **Pre-deployment notice** (30 days before) - Email to affected customers
2. **Changelog entry** - Public, version-controlled
3. **API changelog** - Machine-readable
4. **Release notes** - Summary of changes

**Email Template:**
```
Subject: [Advance Notice] Emission Factor Update - {Fuel Type} ({Country})

Dear Customer,

We're updating the emission factor for {fuel_type} in {country} based on the latest {source_org} data.

CHANGE SUMMARY:
- Current Factor: {old_value} kgCO2e/{unit}
- New Factor: {new_value} kgCO2e/{unit}
- Change: {delta_pct}% ({increase/decrease})
- Effective Date: {deployment_date}

SOURCE:
{source_org} ({source_year}). {source_publication}.
{citation_url}

REASON FOR CHANGE:
{rationale}

IMPACT ON YOUR REPORTS:
Based on your historical usage, this will {increase/decrease} your reported emissions by approximately {customer_impact_pct}%.

ACTION REQUIRED:
None. The update will be applied automatically on {deployment_date}.

If you need to restate past reports with the new factor, please contact support@greenlang.ai.

QUESTIONS?
Reply to this email or join our office hours on {date}.

Best regards,
GreenLang Data Team
```

---

### 8.2 Changelog Format

**Public Changelog** (`CHANGELOG.md`):
```markdown
# Emission Factor Changelog

## [2024-Q4] - 2024-10-24

### Updated
- **US Diesel** (EF:US:diesel:2024:v1)
  - New: 10.23 kgCO2e/gallon (was: 10.21)
  - Change: +0.2%
  - Source: EPA (2024). Emission Factors for GHG Inventories 2024.
  - Reason: Annual EPA update with improved measurement methodology.

- **UK Electricity** (EF:UK:electricity:2024:v1)
  - New: 0.198 kgCO2e/kWh (was: 0.212)
  - Change: -6.6%
  - Source: UK DESNZ (2024). GHG Conversion Factors 2024.
  - Reason: Increased renewable generation in 2024 grid mix.

### Deprecated
- **US Coal (2019)** (EF:US:coal:2019:v1)
  - Now > 5 years old. Use EF:US:coal:2024:v1 for new calculations.
```

**Machine-Readable API** (`/api/factors/changelog?since=2024-10-01`):
```json
{
  "changes": [
    {
      "factor_id": "EF:US:diesel:2024:v1",
      "change_type": "update",
      "previous_value": 10.21,
      "new_value": 10.23,
      "change_pct": 0.2,
      "effective_date": "2024-10-24",
      "source": "EPA (2024)",
      "rationale": "Annual EPA update"
    }
  ]
}
```

---

## 9. Audit & Compliance

### 9.1 Audit Trail Requirements

For every emission factor change:

**Required Documentation:**
1. Change proposal (Section 4.2)
2. Governance Board approval (meeting minutes)
3. QA validation report (test results)
4. Customer communication records (emails sent)
5. Deployment logs (who, when, what)

**Retention Period:** 10 years (regulatory requirement)

---

### 9.2 Annual Audit

**Frequency:** Annually (Q1 of each year)

**Scope:**
- Review all factor changes in past year
- Validate precedence rules were followed
- Check customer communications were sent
- Verify test coverage (100% golden set pass)
- Assess data quality trends (DQS improving?)

**Auditor:** External carbon accounting expert

**Output:** Audit report with recommendations

---

## 10. Continuous Improvement

### 10.1 Metrics

Track these KPIs monthly:

| Metric | Target | Current | Trend |
|--------|--------|---------|-------|
| **Coverage** | > 95% of customer requests | 92% | ↗️ |
| **Average DQS** | ≥ 4.0 | 4.2 | → |
| **Average Uncertainty** | ≤ 10% | 8.5% | ↘️ |
| **Source Conflicts** | < 5/month | 3 | → |
| **Customer Disputes** | < 2/month | 1 | ↘️ |
| **Emergency Corrections** | 0/quarter | 0 | → |
| **Update Frequency** | Quarterly | Quarterly | → |

---

### 10.2 Feedback Loop

**Sources of Feedback:**
1. Customer disputes (Section 6.2)
2. Auditor recommendations (Section 9.2)
3. Regulatory updates (EPA, IEA, BEIS changes)
4. Academic research (new methodologies)
5. Internal quality checks (automated validation)

**Quarterly Review:**
- Data Governance Board reviews metrics
- Identifies gaps in coverage
- Proposes process improvements
- Updates this policy if needed

---

## 11. Glossary

- **DQS:** Data Quality Score (5-dimension GHG Protocol scoring)
- **Material Change:** Emission factor change > 5%
- **Precedence:** Ranking of sources to resolve conflicts
- **Golden Set:** 20 reference calculations for regression testing
- **Immutable Past:** Historical factors never changed retroactively

---

## 12. Document History

| Version | Date | Changes | Approver |
|---------|------|---------|----------|
| 1.0 | 2024-10-24 | Initial policy | CTO |

---

## 13. Approval

**Approved by:**
- [ ] CTO
- [ ] Data Lead
- [ ] Compliance Officer
- [ ] Legal Counsel

**Next Review:** 2025-01-24 (Quarterly)

---

**Document Owner:** Data Lead
**Contact:** data-governance@greenlang.ai
