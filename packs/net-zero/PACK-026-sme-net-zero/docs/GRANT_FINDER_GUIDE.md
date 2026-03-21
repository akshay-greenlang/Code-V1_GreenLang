# PACK-026 Grant Finder Guide

Discover grants, subsidies, tax incentives, and green loans that can fund your net-zero journey. PACK-026 automatically matches your business profile to 50+ programs across the UK, EU, and US.

---

## Table of Contents

1. [How Grant Matching Works](#how-grant-matching-works)
2. [Grant Database Coverage](#grant-database-coverage)
3. [Understanding Your Match Score](#understanding-your-match-score)
4. [Eligibility Criteria Explained](#eligibility-criteria-explained)
5. [Application Timeline Planning](#application-timeline-planning)
6. [Required Documentation Checklist](#required-documentation-checklist)
7. [Success Tips](#success-tips)
8. [Grant Program Directory](#grant-program-directory)
9. [Example Grant Matches](#example-grant-matches)

---

## How Grant Matching Works

PACK-026 uses a 6-factor matching algorithm to score your eligibility for each grant program.

### The Matching Process

```
YOUR PROFILE                    GRANT DATABASE                 RESULTS
+---------------+               +---------------+              +------------+
| Country: UK   |               | 50+ programs  |              | Match 1    |
| Region: W.Mid |               | UK, EU, US    |              | Score: 92  |
| Sector: Mfg   |--matching---->| Updated       |--ranked----->| Match 2    |
| Employees: 45 |  algorithm    | monthly       |  by score    | Score: 78  |
| Actions: LED, |               | Eligibility   |              | Match 3    |
|   heat pump   |               | criteria      |              | Score: 65  |
+---------------+               +---------------+              +------------+
```

### Matching Factors

| Factor | Weight | Description |
|--------|--------|-------------|
| Country | 25% (mandatory) | Grant must be available in your country. If no match, score = 0. |
| Sector | 20% | Your NACE sector code matches the grant's eligible sectors. Partial match scores 10%. |
| Company size | 15% | Your employee count is within the grant's size threshold. |
| Planned actions | 25% | Your planned decarbonization actions match the grant's supported activities. 5 points per matching action, up to 25 points. |
| Region | 10% (bonus) | Your region matches the grant's geographic target area. |
| Timing | 5% (bonus) | Application deadline is more than 30 days away (enough time to apply). |

### How to Get the Best Matches

1. **Complete your profile** -- include your NACE sector code, postcode, and employee count
2. **Plan your actions first** -- run Quick Wins before Grant Finder so the matcher knows what you plan to do
3. **Be specific about location** -- regional grants often have the highest match scores
4. **Check regularly** -- new programs are added monthly and deadlines change

---

## Grant Database Coverage

### Summary

| Region | Programs | Updated | Coverage |
|--------|----------|---------|----------|
| UK | 20+ | Monthly | National + regional (England, Scotland, Wales, NI) |
| EU | 15+ | Monthly | EU-wide + major member states (DE, FR, IT, ES, NL) |
| US | 15+ | Monthly | Federal + major states (CA, NY, TX, IL, WA) |
| **Total** | **50+** | **Monthly** | **3 regions, 20+ countries** |

### Database Update Schedule

- **Grant programs:** Synced monthly from government sources
- **Deadlines:** Checked weekly for updates
- **New programs:** Added within 30 days of announcement
- **Expired programs:** Marked as inactive after deadline passes
- **Verification:** URLs and eligibility criteria manually verified quarterly

---

## Understanding Your Match Score

### Score Ranges

| Score | Rating | Meaning |
|-------|--------|---------|
| 90-100 | Excellent | Strong match -- you are very likely eligible. Apply immediately. |
| 75-89 | Good | Good match -- review eligibility criteria and apply if you meet all requirements. |
| 60-74 | Fair | Possible match -- some criteria may not be met. Review carefully before investing time in application. |
| 40-59 | Weak | Partial match -- significant criteria gaps. Consider whether it is worth applying. |
| 0-39 | Poor | Not recommended -- fundamental eligibility issues. |

### What the Score Does NOT Tell You

- **Competitiveness:** A high match score means you are eligible, not that you will win. Many grants are competitive.
- **Application quality:** Your score is based on eligibility, not the quality of your application narrative.
- **Likelihood of award:** Some programs fund 80% of applicants; others fund 5%. Match score does not reflect this.

---

## Eligibility Criteria Explained

Most grant programs assess eligibility based on these criteria:

### Company Size

| Definition | Employees | Revenue | Balance Sheet |
|-----------|-----------|---------|---------------|
| Micro enterprise | < 10 | < EUR 2M | < EUR 2M |
| Small enterprise | < 50 | < EUR 10M | < EUR 10M |
| Medium enterprise | < 250 | < EUR 50M | < EUR 43M |

These are the EU definitions used by most UK and EU programs. US programs may use different thresholds (e.g., SBA size standards by NAICS code).

### Sector Eligibility

Most programs target specific sectors:
- **Energy efficiency grants:** Available to all sectors
- **Manufacturing innovation grants:** Manufacturing (NACE C) only
- **Agricultural grants:** Agriculture (NACE A) only
- **Transport grants:** Transport/logistics (NACE H) or any sector with fleet vehicles

### Geographic Eligibility

- **National programs:** Available to all businesses in the country
- **Regional programs:** Specific to a region (e.g., West Midlands Growth Hub, Scottish Enterprise)
- **EU structural funds:** Targeted at less-developed or transition regions
- **US state programs:** State-specific with different rules per state

### Activity Eligibility

Grants fund specific activities:
- **Energy audits:** Assessment of energy use and efficiency opportunities
- **Equipment upgrades:** LED, HVAC, insulation, heat pumps, solar PV
- **Fleet electrification:** EV purchase or charging infrastructure
- **Innovation/R&D:** Development of new clean technologies
- **Training:** Sustainability skills development for staff
- **Certification:** ISO 14001 or similar environmental certifications

---

## Application Timeline Planning

### Typical Grant Application Timeline

```
Week 1-2:     DISCOVERY & ELIGIBILITY CHECK
               - Run PACK-026 Grant Finder
               - Review top 3-5 matches
               - Verify eligibility criteria in detail
               - Note application deadlines

Week 3-4:     PREPARATION
               - Gather required documentation (see checklist below)
               - Generate PACK-026 baseline report
               - Generate PACK-026 cost-benefit analysis for planned actions
               - Draft project description and expected outcomes

Week 5-6:     APPLICATION
               - Complete application form
               - Attach supporting documents
               - Get internal sign-off (if needed)
               - Submit before deadline (ideally 1+ weeks early)

Week 7-12:    REVIEW & DECISION
               - Application reviewed by funding body
               - May receive clarification questions
               - Decision communicated (timeline varies by program)

Week 13+:     IMPLEMENTATION
               - Sign grant agreement
               - Implement funded actions
               - Report progress as required
               - Claim grant payments (staged or on completion)
```

### Key Dates to Track

PACK-026 tracks these dates for each matched grant:

```python
grant = grants.matches[0]

print(f"Application opens: {grant.application_open_date}")
print(f"Application deadline: {grant.application_deadline}")
print(f"Decision expected: {grant.expected_decision_date}")
print(f"Implementation period: {grant.implementation_months} months")
print(f"Reporting deadlines: {grant.reporting_dates}")
```

---

## Required Documentation Checklist

Most grant applications require some or all of the following. PACK-026 can generate items marked with a checkmark.

### Company Information

- [ ] Company registration number
- [ ] Company accounts (last 2 years)
- [ ] Tax returns (last 2 years)
- [ ] Proof of SME status (employee count, revenue)
- [ ] Bank details for payment

### Environmental Data (PACK-026 Generated)

- [x] GHG baseline report (Bronze/Silver/Gold)
- [x] Scope 1/2/3 emissions breakdown
- [x] Data quality assessment and tier
- [x] Industry peer comparison
- [x] Emission reduction targets (SBTi SME aligned)

### Project Plan (PACK-026 Assisted)

- [x] Planned decarbonization actions (from Quick Wins engine)
- [x] Cost-benefit analysis for each action (NPV, IRR, payback)
- [x] Expected emission reductions (tCO2e per action)
- [x] Implementation timeline (from action prioritization)
- [ ] Project management plan (who, when, how)
- [ ] Risk assessment (what could go wrong)

### Financial Information (PACK-026 Assisted)

- [x] Total project cost breakdown
- [x] Expected annual savings (GBP)
- [x] Payback period for each action
- [x] Net-of-grant cost calculation
- [ ] Match funding evidence (if co-funding required)
- [ ] Supplier quotes for equipment/services

### Supporting Evidence

- [ ] Energy bills (last 12 months)
- [ ] Photos of existing equipment (if upgrading)
- [ ] Supplier quotes (2-3 quotes per item)
- [ ] Planning permission (if required for solar PV, etc.)
- [ ] Landlord consent (if leased premises)

---

## Success Tips

### 1. Apply Early

Most grants are first-come, first-served or assessed in rounds. Applying early gives you:
- More time for the funding body to request clarifications
- Better chance before the budget is allocated
- Less competition from late applicants

### 2. Focus on Impact, Not Just Cost

Grant reviewers want to see:
- **Environmental impact:** How many tCO2e will you reduce? (Use PACK-026 baseline report)
- **Economic impact:** How many jobs will you protect/create? What cost savings will you achieve?
- **Replicability:** Can your approach be adopted by other SMEs in your sector?

### 3. Show a Clear Plan

Do not just say "we will reduce emissions." Show:
- Specific actions (from PACK-026 Quick Wins)
- Quantified savings (from PACK-026 Cost-Benefit engine)
- Implementation timeline (from PACK-026 Action Prioritization)
- Measurement plan (from PACK-026 Progress Tracker)

### 4. Get Your Numbers Right

Grant reviewers check your numbers. PACK-026 provides:
- Auditable calculations with SHA-256 provenance hashing
- Published emission factors (DEFRA 2024, IEA 2024)
- Industry benchmark comparisons for credibility
- Confidence intervals showing data quality

### 5. Match Your Actions to the Grant's Priorities

If a grant prioritizes energy efficiency, lead with your energy efficiency actions. If it prioritizes innovation, emphasize the novel aspects of your approach. Use the grant program's own language in your application.

### 6. Prepare Strong Supporting Data

The best applications include:
- 12 months of utility bills (not just totals)
- 2-3 supplier quotes for equipment
- Board minutes showing commitment to net zero
- Photos of current equipment (for upgrade grants)

### 7. Do Not Forget the Reporting Requirements

Many SMEs win grants but fail to claim them because they do not meet reporting requirements. Before applying, check:
- How often must you report? (Monthly, quarterly, annually?)
- What data must you provide? (PACK-026 can automate most of this)
- Is there a final evaluation/audit?

---

## Grant Program Directory

### UK Programs

#### Industrial Energy Transformation Fund (IETF)

| Attribute | Details |
|-----------|---------|
| **Provider** | BEIS / DESNZ |
| **Award range** | GBP 100,000 - 30,000,000 |
| **Eligibility** | Manufacturing and industrial SMEs |
| **Funded activities** | Energy efficiency, fuel switching, CCS feasibility studies |
| **Application** | Competitive rounds (2-3 per year) |
| **Website** | [gov.uk/ietf](https://www.gov.uk/government/collections/industrial-energy-transformation-fund) |

#### Energy Efficiency Grant Scheme

| Attribute | Details |
|-----------|---------|
| **Provider** | British Business Bank |
| **Award range** | GBP 1,000 - 25,000 |
| **Eligibility** | All SMEs with fewer than 250 employees |
| **Funded activities** | LED lighting, insulation, HVAC upgrades, smart controls |
| **Application** | Rolling (until budget exhausted) |
| **Website** | [british-business-bank.co.uk](https://www.british-business-bank.co.uk) |

#### Boiler Upgrade Scheme (BUS)

| Attribute | Details |
|-----------|---------|
| **Provider** | Ofgem |
| **Award range** | GBP 5,000 (air source heat pump) - 7,500 (ground source) |
| **Eligibility** | SMEs replacing fossil fuel heating with heat pumps |
| **Funded activities** | Air source heat pumps, ground source heat pumps, biomass boilers |
| **Application** | Via MCS-certified installer |
| **Website** | [ofgem.gov.uk/bus](https://www.ofgem.gov.uk/environmental-and-social-schemes/boiler-upgrade-scheme-bus) |

#### Enhanced Capital Allowances (ECA)

| Attribute | Details |
|-----------|---------|
| **Provider** | HMRC |
| **Award range** | 100% first-year tax deduction on qualifying expenditure |
| **Eligibility** | All businesses purchasing qualifying energy-efficient equipment |
| **Funded activities** | Motors, lighting, HVAC, refrigeration, CHP (from the ETL) |
| **Application** | Claimed through tax return |
| **Website** | [gov.uk/eca](https://www.gov.uk/guidance/enhanced-capital-allowances) |

#### Salix Finance (Public Sector + Some SMEs)

| Attribute | Details |
|-----------|---------|
| **Provider** | Salix Finance |
| **Award range** | Interest-free loans, variable amount |
| **Eligibility** | Public sector and qualifying SMEs |
| **Funded activities** | LED, heating, insulation, controls, renewables |
| **Application** | Rolling applications |
| **Website** | [salixfinance.co.uk](https://www.salixfinance.co.uk) |

#### Green Finance Institute Programs

| Attribute | Details |
|-----------|---------|
| **Provider** | Green Finance Institute |
| **Award range** | Variable (advisory + financing) |
| **Eligibility** | SMEs seeking green finance |
| **Funded activities** | Green building retrofit, renewable energy, sustainable transport |
| **Application** | Through partner financial institutions |
| **Website** | [greenfinanceinstitute.co.uk](https://www.greenfinanceinstitute.co.uk) |

#### Innovate UK Smart Grants

| Attribute | Details |
|-----------|---------|
| **Provider** | UKRI / Innovate UK |
| **Award range** | GBP 25,000 - 500,000 |
| **Eligibility** | Innovative SMEs developing clean technology |
| **Funded activities** | R&D, prototype development, market testing |
| **Application** | Competitive rounds (quarterly) |
| **Website** | [apply-for-innovation-funding.service.gov.uk](https://apply-for-innovation-funding.service.gov.uk) |

#### Local Authority Energy Schemes

| Attribute | Details |
|-----------|---------|
| **Provider** | Various local councils |
| **Award range** | GBP 500 - 10,000 |
| **Eligibility** | SMEs in the council area |
| **Funded activities** | Energy audits, LED, insulation, renewable energy |
| **Application** | Via local council or Growth Hub |
| **Note** | Varies significantly by area. PACK-026 matches your postcode to local programs. |

---

### EU Programs

#### EU Cohesion Fund

| Attribute | Details |
|-----------|---------|
| **Provider** | European Commission |
| **Award range** | Variable (managed by member states) |
| **Eligibility** | SMEs in eligible regions (GDP < 90% EU average) |
| **Funded activities** | Energy efficiency, renewable energy, sustainable transport |
| **Application** | Through national managing authorities |
| **Website** | [ec.europa.eu/regional_policy](https://ec.europa.eu/regional_policy/funding/cohesion-fund_en) |

#### Just Transition Mechanism

| Attribute | Details |
|-----------|---------|
| **Provider** | European Commission |
| **Award range** | Variable |
| **Eligibility** | SMEs in regions heavily dependent on fossil fuels or carbon-intensive industry |
| **Funded activities** | Workers reskilling, SME diversification, clean energy transition |
| **Application** | Through national territorial just transition plans |
| **Website** | [ec.europa.eu/just-transition](https://ec.europa.eu/info/strategy/priorities-2019-2024/european-green-deal/just-transition-mechanism_en) |

#### LIFE Programme

| Attribute | Details |
|-----------|---------|
| **Provider** | European Commission (CINEA) |
| **Award range** | EUR 60,000 - 10,000,000 |
| **Eligibility** | Any EU-based organization including SMEs |
| **Funded activities** | Climate mitigation and adaptation projects, environmental innovation |
| **Application** | Annual calls for proposals |
| **Website** | [cinea.ec.europa.eu/life](https://cinea.ec.europa.eu/programmes/life_en) |

#### Horizon Europe -- EIC Accelerator

| Attribute | Details |
|-----------|---------|
| **Provider** | European Innovation Council |
| **Award range** | EUR 50,000 (Pathfinder) - 2,500,000 (Accelerator) |
| **Eligibility** | Innovative SMEs with breakthrough clean technologies |
| **Funded activities** | Deep tech R&D, scaling up, market deployment |
| **Application** | Rolling (Accelerator) or annual calls (Pathfinder) |
| **Website** | [eic.ec.europa.eu](https://eic.ec.europa.eu) |

#### InvestEU SME Window

| Attribute | Details |
|-----------|---------|
| **Provider** | European Investment Fund (EIF) |
| **Award range** | Loan guarantees (via financial intermediaries) |
| **Eligibility** | All EU SMEs |
| **Funded activities** | Green investments, energy efficiency, clean transport |
| **Application** | Through local banks partnered with EIF |
| **Website** | [investeu.europa.eu](https://investeu.europa.eu) |

---

### US Programs

#### DOE Small Business Innovation Research (SBIR)

| Attribute | Details |
|-----------|---------|
| **Provider** | US Department of Energy |
| **Award range** | USD 200,000 (Phase I) - 1,100,000 (Phase II) |
| **Eligibility** | US small businesses with < 500 employees |
| **Funded activities** | Clean energy R&D, energy efficiency innovation |
| **Application** | Quarterly solicitations |
| **Website** | [science.osti.gov/sbir](https://science.osti.gov/sbir) |

#### EPA Environmental Justice Grants

| Attribute | Details |
|-----------|---------|
| **Provider** | US Environmental Protection Agency |
| **Award range** | USD 50,000 - 500,000 |
| **Eligibility** | Community-based organizations and small businesses |
| **Funded activities** | Environmental education, pollution reduction, community resilience |
| **Application** | Annual solicitations |
| **Website** | [epa.gov/environmentaljustice/grants](https://www.epa.gov/environmentaljustice/environmental-justice-grants-funding-and-technical-assistance) |

#### USDA Rural Energy for America Program (REAP)

| Attribute | Details |
|-----------|---------|
| **Provider** | US Department of Agriculture |
| **Award range** | USD 2,500 - 1,000,000 (grants); USD 5,000 - 25,000,000 (loans) |
| **Eligibility** | Rural businesses and agricultural producers |
| **Funded activities** | Renewable energy systems, energy efficiency improvements |
| **Application** | Annual application windows |
| **Website** | [rd.usda.gov/reap](https://www.rd.usda.gov/programs-services/energy-programs/rural-energy-america-program-renewable-energy-systems-energy-efficiency-improvement-guaranteed-loans) |

#### IRA Clean Energy Tax Credits

| Attribute | Details |
|-----------|---------|
| **Provider** | IRS (Inflation Reduction Act) |
| **Award range** | 30% Investment Tax Credit (ITC) or Production Tax Credit (PTC) |
| **Eligibility** | Any US business investing in clean energy |
| **Funded activities** | Solar PV, wind, battery storage, EV charging, heat pumps, clean vehicles |
| **Application** | Claimed through tax filing |
| **Website** | [irs.gov/credits-deductions/clean-energy-tax-credits](https://www.irs.gov/credits-deductions/clean-energy-tax-credits) |

#### SBA Green Loans (504/7a)

| Attribute | Details |
|-----------|---------|
| **Provider** | US Small Business Administration |
| **Award range** | USD 50,000 - 5,000,000 |
| **Eligibility** | US small businesses |
| **Funded activities** | Energy-efficient buildings, equipment, renewable energy |
| **Application** | Through SBA-approved lenders |
| **Website** | [sba.gov/funding-programs](https://www.sba.gov/funding-programs/loans) |

---

## Example Grant Matches

### Example 1: Micro Office Business (UK, 8 employees)

**Profile:** IT consultancy, London, 8 employees, GBP 800K revenue. Planned actions: LED lighting, green energy tariff, cycle-to-work scheme.

| Rank | Grant | Match Score | Award | Reason |
|------|-------|:-----------:|-------|--------|
| 1 | Energy Efficiency Grant Scheme | 88 | GBP 1K-5K | LED lighting matches funded activities |
| 2 | Enhanced Capital Allowances | 82 | Tax deduction | LED equipment qualifies for ECA |
| 3 | Local Authority Energy Scheme (GLA) | 75 | GBP 500-2K | London-based, energy efficiency |
| 4 | SME Climate Commitment Fund | 68 | GBP 1K-3K | Climate commitment matches criteria |

### Example 2: Small Manufacturer (UK, 45 employees)

**Profile:** Light manufacturing, West Midlands, 45 employees, GBP 3.5M revenue. Planned actions: LED, heat pump, EV fleet, compressed air fix.

| Rank | Grant | Match Score | Award | Reason |
|------|-------|:-----------:|-------|--------|
| 1 | IETF (Round 4) | 92 | GBP 100K-1M | Manufacturing + energy efficiency + fuel switching |
| 2 | Boiler Upgrade Scheme | 85 | GBP 5K-7.5K | Heat pump installation |
| 3 | Energy Efficiency Grant | 82 | GBP 5K-25K | LED + compressed air |
| 4 | Enhanced Capital Allowances | 80 | Tax deduction | All equipment qualifies |
| 5 | West Midlands Growth Hub | 78 | GBP 1K-10K | Regional match + manufacturing |

### Example 3: Medium Tech Company (US, 120 employees)

**Profile:** Software company, California, 120 employees, USD 15M revenue. Planned actions: solar PV, EV charging, smart HVAC, remote work policy.

| Rank | Grant | Match Score | Award | Reason |
|------|-------|:-----------:|-------|--------|
| 1 | IRA Clean Energy Tax Credits | 90 | 30% of solar + EV costs | Solar PV and EV charging qualify |
| 2 | California SGIP (Self-Gen) | 85 | USD 10K-50K | Solar + storage incentive |
| 3 | DOE SBIR (Clean Energy) | 72 | USD 200K-1.1M | If developing clean tech IP |
| 4 | SBA 504 Green Loan | 68 | USD 50K-5M | Solar PV and building upgrades |
| 5 | PG&E On-Bill Financing | 65 | Interest-free loan | HVAC and lighting upgrades |

---

## Keeping Your Grants Up to Date

### Automatic Updates

PACK-026 syncs the grant database monthly. To force an immediate sync:

```bash
greenlang pack-026 sync-grants --force
```

### Notifications

Set up notifications for grant-related events:

```python
from packs.net_zero.PACK_026_sme_net_zero.engines.grant_finder_engine import GrantFinderEngine

finder = GrantFinderEngine()

# Enable notifications
await finder.configure_notifications(
    entity_id="your-entity-id",
    notify_on=["new_high_match", "deadline_approaching", "program_update"],
    notification_channel="email",
    email="owner@your-company.com",
    deadline_warning_days=30,  # Alert 30 days before deadline
)
```

### Submitting New Grant Programs

Know of a grant program not in our database? Submit it:

```bash
greenlang pack-026 suggest-grant \
    --name "My Local Grant" \
    --provider "Local Council" \
    --url "https://council.gov.uk/grants" \
    --country UK \
    --region "Yorkshire"
```

Or email grants@greenlang.io with program details.

---

*Grant Finder Guide -- PACK-026 SME Net Zero Pack v1.0.0*
*Grant database updated monthly. Last sync: 2026-03-18.*
*For support, contact support@greenlang.io*
