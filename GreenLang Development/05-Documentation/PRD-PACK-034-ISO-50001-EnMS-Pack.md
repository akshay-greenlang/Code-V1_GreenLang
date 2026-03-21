# PRD-PACK-034: ISO 50001 Energy Management System Pack

**Pack ID:** PACK-034-iso-50001-enms
**Category:** Energy Efficiency Packs
**Tier:** Professional
**Version:** 1.0.0
**Status:** Approved
**Author:** GreenLang Product Team
**Date:** 2026-03-21
**Prerequisite:** None (standalone; enhanced with PACK-031 Industrial Energy Audit Pack, PACK-032 Building Energy Assessment Pack, and PACK-033 Quick Wins Identifier Pack if present; complemented by PACK-021/022/023 Net Zero Packs)

---

## 1. Executive Summary

### 1.1 Problem Statement

ISO 50001:2018 is the internationally recognized standard for Energy Management Systems (EnMS), providing organizations with a structured framework to achieve continual improvement in energy performance. Over 30,000 organizations worldwide hold ISO 50001 certification, and the EU Energy Efficiency Directive (EED) 2023/1791 mandates energy management systems for enterprises consuming more than 85 TJ per year (Article 12). Despite strong regulatory drivers and proven energy savings of 10-30% within the first three years of implementation, organizations face persistent challenges in establishing and maintaining an effective EnMS:

1. **Significant Energy Use (SEU) identification complexity**: ISO 50001:2018 Clause 6.3 requires organizations to identify SEUs through systematic energy review, yet most facilities lack the analytical tools to perform Pareto analysis across hundreds of energy-consuming systems. Without rigorous SEU identification, organizations focus on easily visible energy consumers while overlooking systems that collectively represent 30-50% of total consumption. The standard requires criteria-based determination of SEUs accounting for substantial energy consumption, considerable potential for improvement, and planned changes -- a multi-dimensional analysis that spreadsheets cannot reliably perform.

2. **Energy Baseline (EnB) establishment difficulty**: ISO 50006:2014 defines the methodology for establishing energy baselines using regression analysis against relevant variables (production volume, degree-days, occupancy, working days). Most organizations lack the statistical capabilities to build multivariate regression models, validate them against ASHRAE Guideline 14-2014 criteria (R-squared >= 0.75, CV(RMSE) <= 25%, NMBE within +/- 10%), and maintain baselines through structural changes and variable recalibration. Poor baselines undermine the entire EnMS measurement framework.

3. **Energy Performance Indicator (EnPI) management gaps**: ISO 50006 defines multiple EnPI types (simple ratio, regression-based, CUSUM, energy intensity index), each appropriate for different monitoring contexts. Organizations typically default to a single simple ratio EnPI (kWh/unit) that fails to account for weather, production mix, or operational variations. Without statistical validation of EnPI adequacy and sensitivity analysis, organizations report misleading energy performance data that does not reflect true underlying efficiency.

4. **CUSUM monitoring absence**: Cumulative Sum (CUSUM) charts are the recommended tool under ISO 50006 for ongoing energy performance monitoring, providing early detection of performance drift before it becomes significant. Most EnMS implementations lack CUSUM monitoring entirely, relying instead on monthly comparisons that miss gradual degradation trends. Without CUSUM control limits and automatic alerting, organizations discover performance problems months after they begin.

5. **Degree-day normalization challenges**: Weather-sensitive facilities require Heating Degree Day (HDD) and Cooling Degree Day (CDD) normalization to separate weather effects from operational efficiency changes. Determining facility-specific base temperatures through change-point regression, sourcing reliable weather station data, and applying normalization consistently across reporting periods requires specialized analytical tools that most energy managers lack.

6. **Energy balance and sub-metering gaps**: ISO 50001 Clause 6.6 requires organizations to monitor energy performance at appropriate levels. Most facilities have fiscal metering only (whole-building electricity and gas meters) without sub-metering of major systems. Constructing an accurate energy balance -- accounting for all inputs, useful outputs, and losses -- is essential for identifying improvement opportunities but requires systematic Sankey diagram analysis and metering gap assessment.

7. **Action plan management weakness**: ISO 50001 Clause 6.2 requires objectives, energy targets, and action plans for achieving them, following SMART criteria (Specific, Measurable, Achievable, Relevant, Time-bound). Organizations struggle to link identified energy conservation measures to quantified targets, track implementation progress, allocate resources, and verify outcomes. Without structured action plan management, EnMS objectives remain aspirational rather than operational.

8. **Clause-by-clause compliance gaps**: ISO 50001:2018 contains requirements across 10 clauses (Context, Leadership, Planning, Support, Operation, Performance Evaluation, Improvement), with 39 mandatory "shall" requirements. Organizations implementing the standard for the first time, or transitioning from the 2011 edition, struggle to achieve complete compliance. Gap analysis against all clauses requires systematic assessment that manual checklists handle poorly, especially when tracking remediation progress across multiple nonconformities.

9. **Performance trending and ISO 50015 verification**: ISO 50015:2014 defines the methodology for measurement and verification (M&V) of energy performance improvements within an EnMS. Year-over-year performance comparison requires normalized baselines, validated regression models, and statistical confidence intervals. Most organizations cannot demonstrate that claimed energy savings are statistically significant, undermining credibility with certification body auditors, senior management, and external stakeholders.

10. **Management review preparation burden**: ISO 50001 Clause 9.3 requires management review at planned intervals, with defined inputs (audit results, energy performance and EnPI trends, status of corrective actions, projected energy performance, objectives and targets status) and outputs (decisions on continual improvement, resource allocation, policy changes). Assembling a comprehensive management review package from disparate data sources is time-consuming and error-prone, often resulting in incomplete reviews that fail to drive meaningful management decisions.

### 1.2 Solution Overview

PACK-034 is the **ISO 50001 Energy Management System Pack** -- the fourth pack in the "Energy Efficiency Packs" category, complementing PACK-031 (Industrial Energy Audit), PACK-032 (Building Energy Assessment), and PACK-033 (Quick Wins Identifier). While PACK-031 through PACK-033 focus on energy auditing, assessment, and quick win identification, PACK-034 provides the ongoing energy management system framework required by ISO 50001:2018 for continual improvement in energy performance.

The pack delivers automated SEU identification with Pareto analysis and criteria-based ranking (Clause 6.3), ISO 50006-compliant baseline establishment with multivariate regression and statistical validation, EnPI calculation and tracking across all ISO 50006 EnPI types, CUSUM monitoring with control limits and automatic alert generation, HDD/CDD normalization with facility-specific change-point models, energy balance construction with Sankey diagram data and sub-metering gap identification, SMART action plan management with resource tracking and outcome verification, clause-by-clause gap analysis with remediation tracking and certification readiness scoring, ISO 50015-compliant performance trending with year-over-year normalized comparison, and management review package generation with all Clause 9.3 required inputs and outputs.

The pack includes 10 engines, 8 workflows, 10 templates, 12 integrations, and 8 presets covering the complete ISO 50001 EnMS lifecycle from initial energy review through certification and ongoing continual improvement.

Every calculation is **zero-hallucination** (deterministic lookups and arithmetic only, no LLM in any calculation path), **bit-perfect reproducible**, and **SHA-256 hashed** for audit assurance.

### 1.3 Key Differentiators

| Dimension | Manual / Consultant Approach | PACK-034 ISO 50001 EnMS Pack |
|-----------|------------------------------|-------------------------------|
| SEU identification | Manual spreadsheet Pareto (covers 60-70% of energy) | Automated multi-criteria SEU analysis (covers 95%+ of energy) |
| Baseline establishment | Consultant-built Excel model (2-4 weeks, EUR 10,000-20,000) | Automated regression with statistical validation (<1 hour) |
| EnPI management | Single simple-ratio EnPI per facility | Multi-type EnPIs per ISO 50006 with statistical validation |
| CUSUM monitoring | Rarely implemented | Automated CUSUM with control limits and real-time alerts |
| Weather normalization | Manual HDD/CDD lookup and calculation | Automated change-point regression with station-sourced data |
| Energy balance | Sankey by specialist consultant (EUR 5,000-15,000) | Automated balance with metering gap identification |
| Action plan tracking | Spreadsheet or project management tool | Integrated SMART action plans linked to EnPIs and targets |
| Compliance gap analysis | Manual checklist (2-4 weeks consultant time) | Automated clause-by-clause assessment with remediation tracking |
| Performance trending | Annual report by consultant | Continuous ISO 50015 M&V with statistical confidence intervals |
| Management review | Manual package assembly (1-2 weeks per review) | Automated Clause 9.3 package generation (<1 hour) |
| Certification readiness | Subjective consultant assessment | Quantified readiness score with clause-level gap tracking |
| Audit trail | Paper-based EnMS documentation | SHA-256 provenance, full calculation lineage, digital audit trail |

### 1.4 ISO 50001:2018 Clause Coverage

| Clause | Title | Pack Coverage |
|--------|-------|---------------|
| 4.1 | Understanding the organization and its context | Facility profiling, scope definition via SetupWizard |
| 4.2 | Understanding the needs and expectations of interested parties | Stakeholder registry, regulatory obligation tracking |
| 4.3 | Determining the scope of the EnMS | Scope boundary definition, energy carrier registry |
| 4.4 | Energy management system | Full EnMS framework via orchestrator and workflows |
| 5.1 | Leadership and commitment | Management review inputs/outputs (ManagementReviewEngine) |
| 5.2 | Energy policy | EnergyPolicyTemplate with mandatory elements |
| 5.3 | Organizational roles, responsibilities and authorities | RBAC integration, energy team registry |
| 6.1 | Actions to address risks and opportunities | Risk/opportunity register linked to action plans |
| 6.2 | Objectives, energy targets, and planning to achieve them | ActionPlanEngine with SMART validation |
| 6.3 | Energy review | SEUAnalyzerEngine + EnergyBaselineEngine + EnPICalculatorEngine |
| 6.4 | Energy performance indicators | EnPICalculatorEngine (all ISO 50006 types) |
| 6.5 | Energy baseline | EnergyBaselineEngine (static, dynamic, adjusted, segmented) |
| 6.6 | Planning for collection of energy data | MeteringBridge + data collection planning |
| 7.1-7.5 | Support (resources, competence, awareness, communication, documentation) | EnMSDocumentationTemplate, training tracking |
| 8.1 | Operational planning and control | OperationalControlTemplate, operational criteria |
| 8.2 | Design | Design criteria for energy-significant equipment |
| 8.3 | Procurement | Energy procurement criteria documentation |
| 9.1 | Monitoring, measurement, analysis and evaluation | CUSUMMonitorEngine + PerformanceTrendEngine |
| 9.2 | Internal audit | InternalAuditTemplate + ComplianceCheckerEngine |
| 9.3 | Management review | ManagementReviewEngine with full input/output coverage |
| 10.1 | Nonconformity and corrective action | CorrectiveActionTemplate, CA tracking |
| 10.2 | Continual improvement | Performance trending, EnPI improvement tracking |

### 1.5 Target Users

**Primary:**
- Energy managers responsible for ISO 50001 implementation and certification
- ISO 50001 management representatives (EnMS team leaders)
- Facility engineers maintaining energy baselines and EnPIs
- EnMS internal auditors conducting Clause 9.2 audits

**Secondary:**
- ISO 50001 implementation consultants working with multiple client sites
- Certification body (CB) auditors reviewing EnMS documentation and performance data
- Corporate sustainability teams rolling out ISO 50001 across multi-site portfolios
- EHS managers meeting EED Article 12 mandatory EnMS requirements (>85 TJ/year)
- CFOs evaluating energy performance improvement ROI
- Operations managers responsible for operational control procedures
- Data center operators seeking ISO 50001 certification for PUE optimization

### 1.6 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| ISO 50001:2018 clause coverage | 100% of clauses 4.1 through 10.2 | Automated clause mapping verification |
| EnPI calculation accuracy per ISO 50006 | 100% match with manual engineering calculations | Cross-validated against 100 known-value EnPI test cases |
| Baseline regression fit | R-squared >= 0.85 for regression models (0.75 minimum) | Statistical fit per ASHRAE Guideline 14-2014 |
| CUSUM detection sensitivity | Detect 5% performance drift within 2 reporting periods | Validated against synthetic drift injection scenarios |
| Savings verification accuracy per ISO 50015 | Within 10% of actual verified savings at 12 months | Cross-validated against IPMVP M&V results |
| Certification readiness score accuracy | 95% correlation with actual CB audit findings | Validated against 20+ CB audit reports |
| Management review package generation time | <1 hour (vs. 1-2 weeks manual) | Time from trigger to complete Clause 9.3 package |
| Time to establish EnMS (from greenfield) | <3 months (vs. 9-18 months typical) | Time from initial setup to Stage 1 audit readiness |
| Customer NPS | >50 | Net Promoter Score survey |
| Energy performance improvement | 5-15% in first year post-implementation | EnPI improvement tracked via CUSUM |

---

## 2. Regulatory & Framework Basis

### 2.1 Primary Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| ISO 50001:2018 | Energy management systems -- Requirements with guidance for use | Core standard; all 10 clauses implemented across engines, workflows, and templates |
| ISO 50006:2014 | Measuring energy performance using energy baselines and energy performance indicators | EnB methodology (EnergyBaselineEngine), EnPI types (EnPICalculatorEngine), CUSUM (CUSUMMonitorEngine) |
| ISO 50015:2014 | Measurement and verification of energy performance of organizations | M&V methodology for savings verification (PerformanceTrendEngine) |
| ISO 50004:2020 | Guidance for implementation, maintenance and improvement of an EnMS | Implementation guidance informing workflows and setup wizard |
| ISO 50002:2014 | Energy audits -- Requirements with guidance for use | Energy audit integration complementing PACK-031 |
| EU Energy Efficiency Directive (EED) | Directive 2023/1791 (recast) | Article 12 mandatory EnMS for enterprises >85 TJ/year; Article 11 audit exemption for certified EnMS |

### 2.2 Statistical and M&V Standards

| Standard | Reference | Pack Relevance |
|----------|-----------|----------------|
| ASHRAE Guideline 14-2014 | Measurement of Energy, Demand, and Water Savings | Statistical requirements for baseline regression (R-squared, CV(RMSE), NMBE) |
| IPMVP Core Concepts 2022 | International Performance Measurement and Verification Protocol | M&V Options A-D for savings verification integrated in PerformanceTrendEngine |
| EN 16247-1:2022 | Energy audits -- Part 1: General requirements | Energy audit methodology complementing Clause 6.3 energy review |
| ISO 50049:2020 | Calculation methods for energy efficiency and energy consumption variations | Normalization methodology for production-adjusted baselines |

### 2.3 EU Regulatory Framework

| Regulation | Reference | Pack Relevance |
|------------|-----------|----------------|
| EU Emissions Trading System (EU ETS) | Directive 2003/87/EC (as amended by 2023/959) | Energy efficiency requirements for ETS installations; free allocation conditioned on efficiency |
| Industrial Emissions Directive (IED) | Directive 2010/75/EU | BAT and BAT-AEL compliance for energy efficiency via BREF documents |
| Energy Performance of Buildings Directive (EPBD) | Directive 2024/1275 (recast) | Building energy performance for facilities with conditioned spaces |
| Ecodesign for Sustainable Products Regulation | Regulation 2024/1781 | Motor, pump, fan, compressor efficiency requirements (IE classes) |
| EU Taxonomy Regulation | Regulation 2020/852 | Climate mitigation substantial contribution criteria for energy efficiency investments |

### 2.4 Supporting Standards and Frameworks

| Standard / Framework | Reference | Pack Relevance |
|---------------------|-----------|----------------|
| GHG Protocol Corporate Standard | WRI/WBCSD (2015) | Scope 1+2 emissions linked to energy consumption tracked via MRV bridge |
| ISO 14064-1:2018 | Organization GHG quantification | GHG emissions linked to energy use for combined reporting |
| ESRS E1 Climate Change | EU CSRD (2023) | E1-5 energy consumption and mix disclosure; EnPI data feeds CSRD reporting |
| CDP Climate Change | CDP (2024) | C7 Energy breakdown, C8 Energy-related emissions |
| TCFD Recommendations | FSB/TCFD (2017) | Metrics and targets for energy efficiency |
| IEC 60034-30-1 | Rotating electrical machines -- Efficiency classes (IE code) | Motor efficiency classes IE1 through IE5 for SEU equipment analysis |
| EN 12953/EN 12952 | Shell/Water-tube boilers | Boiler efficiency for thermal SEU analysis |
| ISO 11011:2013 | Compressed air -- Energy efficiency -- Assessment | Compressed air SEU assessment methodology |

---

## 3. Technical Architecture

### 3.1 Components Overview

| Component Type | Count | Description |
|----------------|-------|-------------|
| Engines | 10 | ISO 50001 EnMS calculation and analysis engines |
| Workflows | 8 | Multi-phase orchestration workflows |
| Templates | 10 | Report, dashboard, documentation, and compliance templates |
| Integrations | 12 | Agent, app, data, and system bridges |
| Presets | 8 | Facility-type-specific configurations |
| Config | 1 | Runtime configuration (Pydantic v2) |
| Demo | 1 | Demo configuration |

### 3.2 Engines

| # | Engine | Purpose |
|---|--------|---------|
| 1 | `seu_analyzer_engine.py` | ISO 50001 Clause 6.3 Significant Energy Use (SEU) identification. Performs Pareto analysis across all energy-consuming systems to identify the vital few that constitute 80%+ of total consumption. Multi-criteria SEU determination using three ISO 50001 criteria: (a) substantial energy consumption, (b) considerable potential for improvement in energy performance, and (c) planned changes or new developments likely to impact energy performance. Outputs ranked SEU list with consumption magnitude (kWh/yr, % of total), improvement potential (estimated savings range), current efficiency vs. best-practice benchmark, relevant variables affecting each SEU, and recommended EnPIs per SEU. Supports hierarchical SEU decomposition (facility -> system -> subsystem -> equipment). |
| 2 | `energy_baseline_engine.py` | ISO 50006 Energy Baseline (EnB) establishment using multivariate regression analysis. Supports four baseline types: static (fixed 12-month reference period), dynamic (rolling 12-month recalculated monthly), adjusted (static with normalization for structural changes), and segmented (separate baselines for distinct operating modes). Performs automatic variable selection using stepwise regression with AIC/BIC criteria. Statistical validation per ASHRAE 14-2014: R-squared >= 0.75 required (0.85 target), CV(RMSE) <= 25%, NMBE within +/- 10%, Durbin-Watson autocorrelation check, and outlier detection via studentized residuals (> 2.5 sigma). Handles multi-fuel facilities with separate baselines per energy carrier and primary energy conversion. |
| 3 | `enpi_calculator_engine.py` | ISO 50006 Energy Performance Indicator (EnPI) calculation across all defined types: simple ratio (energy/activity), regression-based (E_predicted = a + bX1 + cX2), CUSUM (cumulative sum of actual vs. baseline), energy intensity index (E_actual/E_baseline_adjusted * 100), and Specific Energy Consumption (energy/production_normalized). Provides statistical validation of EnPI adequacy including sensitivity analysis (does the EnPI respond to known efficiency changes?), specificity analysis (does the EnPI avoid response to non-efficiency factors?), and reproducibility testing. Outputs EnPI values with confidence intervals, trend direction, and improvement percentage vs. baseline. |
| 4 | `cusum_monitor_engine.py` | Cumulative Sum (CUSUM) control chart calculation and monitoring per ISO 50006. Computes CUSUM as the running total of (E_actual - E_expected) for each reporting period, where E_expected comes from the validated baseline regression model. Establishes upper and lower control limits using configurable sigma multipliers (default 2-sigma for warning, 3-sigma for action). Detects performance drift direction (improving vs. degrading), drift onset period, drift magnitude (kWh and percentage), and statistical significance. Generates automatic alerts when CUSUM exceeds control limits or shows sustained unidirectional trend (8+ consecutive points above/below zero). Supports both tabular CUSUM and V-mask approaches. |
| 5 | `degree_day_engine.py` | Heating Degree Day (HDD) and Cooling Degree Day (CDD) calculation with facility-specific change-point models. Computes HDD as sum of max(0, T_base - T_avg) and CDD as sum of max(0, T_avg - T_base) for each day, with configurable base temperatures. Performs change-point regression to determine facility-specific balance points: 3-parameter model (heating only), 4-parameter model (heating + cooling), and 5-parameter model (full HVAC with deadband). Sources weather data via integration bridge (NOAA ISD, Meteostat, Open-Meteo). Provides TMY (Typical Meteorological Year) degree-day data for forward projection and long-term normalization. Supports multiple weather stations with distance-weighted averaging. |
| 6 | `energy_balance_engine.py` | Facility energy balance construction with Sankey diagram data structures. Maps all energy inputs (electricity, natural gas, LPG, fuel oil, district heat/cooling, renewables) through conversion systems (boilers, chillers, CHP, transformers), distribution systems (steam, chilled water, compressed air, electrical), and end uses (motors, lighting, HVAC, process heat, refrigeration) to useful output and losses. Calculates balance residual (unaccounted energy) as a data quality indicator -- residual >5% triggers metering gap investigation. Identifies sub-metering needs by flagging unmetered SEUs. Supports hierarchical balance (site -> building -> floor -> zone) and temporal balance (annual, monthly, daily). |
| 7 | `action_plan_engine.py` | ISO 50001 Clause 6.2 objectives, energy targets, and action plan management. Validates action plans against SMART criteria: Specific (linked to named SEU and EnPI), Measurable (quantified energy savings target in kWh/yr), Achievable (validated against engineering calculations and benchmarks), Relevant (linked to energy policy commitments), Time-bound (start date, end date, milestones). Tracks implementation status (planned, in-progress, completed, verified), resource allocation (budget, personnel, contractors), and outcome verification (actual vs. target savings, variance analysis). Calculates portfolio-level metrics: total investment, total projected savings, aggregate payback, cumulative EnPI improvement. |
| 8 | `compliance_checker_engine.py` | ISO 50001:2018 clause-by-clause gap analysis and certification readiness assessment. Evaluates compliance across all 39 mandatory "shall" requirements in clauses 4 through 10. For each requirement: determines compliance status (conforming, minor nonconformity, major nonconformity, not assessed), identifies objective evidence of conformity, flags gaps with specific remediation recommendations, and estimates remediation effort (hours). Calculates overall certification readiness score (0-100) with clause-level breakdown. Tracks remediation progress over time with milestone dates. Differentiates between Stage 1 (documentation review) and Stage 2 (implementation audit) readiness. |
| 9 | `performance_trend_engine.py` | ISO 50015-compliant energy performance trending with year-over-year normalized comparison and savings verification. Computes adjusted baseline energy for each reporting period using validated regression model with current-period values of relevant variables. Calculates energy savings as (E_baseline_adjusted - E_actual) with 90% confidence intervals per ASHRAE 14-2014. Performs statistical significance testing (t-test) on claimed savings. Supports IPMVP Options A-D: Option A (key parameter measurement with stipulated values), Option B (all parameter measurement), Option C (whole facility regression), Option D (calibrated simulation). Generates cumulative savings over EnMS lifetime with inflation-adjusted financial value. |
| 10 | `management_review_engine.py` | ISO 50001 Clause 9.3 management review package generation. Assembles all required review inputs: (a) status of actions from previous management reviews, (b) changes in issues and requirements, (c) energy performance and EnPI trends, (d) status of corrective actions, (e) projected energy performance for coming period, (f) status of objectives and energy targets. Generates structured review outputs: (a) conclusions on adequacy, suitability, effectiveness and continual improvement, (b) decisions on resource allocation, (c) decisions on policy revision needs. Includes executive KPI dashboard with traffic-light status indicators and trend sparklines. |

### 3.3 Workflows

| # | Workflow | Phases | Purpose |
|---|----------|--------|---------|
| 1 | `energy_review_workflow.py` | 4: DataCollection -> SEUIdentification -> BaselineEnPIEstablishment -> OpportunityRegister | Full ISO 50001 Clause 6.3 energy review from data gathering through SEU identification, baseline/EnPI establishment, and energy saving opportunity identification |
| 2 | `baseline_establishment_workflow.py` | 3: DataValidation -> RegressionModeling -> BaselineApproval | ISO 50006 baseline establishment from data validation through multivariate regression analysis and formal baseline approval with statistical validation report |
| 3 | `action_plan_workflow.py` | 4: ObjectiveSetting -> ActionDefinition -> ResourceAllocation -> TimelineGeneration | Clause 6.2 action plan development from strategic objectives through specific SMART actions, resource allocation, and implementation timeline |
| 4 | `operational_control_workflow.py` | 3: CriteriaDefinition -> MonitoringSetup -> DeviationResponse | Clause 8.1 operational control establishment from defining operating criteria for SEUs through monitoring configuration and deviation response procedures |
| 5 | `monitoring_workflow.py` | 4: MeteringValidation -> DataCollection -> EnPICalculation -> AlertReporting | Clause 9.1 monitoring, measurement, analysis and evaluation from metering validation through data collection, EnPI calculation, and CUSUM alerting |
| 6 | `performance_analysis_workflow.py` | 3: EnPICalculation -> CUSUMAnalysis -> TrendReporting | Performance analysis combining EnPI calculation, CUSUM chart generation with control limit evaluation, and normalized performance trend reporting |
| 7 | `mv_verification_workflow.py` | 3: BaselineComparison -> SavingsQuantification -> UncertaintyAssessment | ISO 50015 M&V verification from normalized baseline comparison through savings quantification and statistical uncertainty assessment |
| 8 | `audit_certification_workflow.py` | 4: GapAnalysis -> InternalAudit -> CorrectiveActions -> CertificationReadiness | Certification preparation from clause-by-clause gap analysis through internal audit execution, corrective action management, and Stage 1/Stage 2 readiness assessment |

### 3.4 Templates

| # | Template | Formats | Purpose |
|---|----------|---------|---------|
| 1 | `energy_policy_template.py` | MD, HTML, PDF, JSON | ISO 50001 Clause 5.2 energy policy document with all mandatory elements: commitment to continual improvement, commitment to information and resources, commitment to legal compliance, support for procurement of energy-efficient products/services, support for design activities |
| 2 | `energy_review_report_template.py` | MD, HTML, PDF, JSON | Clause 6.3 energy review report including past/present energy use analysis, SEU list with Pareto chart data, relevant variables per SEU, current energy performance, estimated future energy use and consumption, and identified improvement opportunities |
| 3 | `enpi_methodology_template.py` | MD, HTML, PDF, JSON | ISO 50006-compliant EnPI methodology document: EnPI definitions, baseline specification, regression model parameters, statistical validation results, normalization methodology, EnPI review/update schedule, and data collection plan |
| 4 | `action_plan_template.py` | MD, HTML, PDF, JSON | Clause 6.2 action plan document: objectives and energy targets linked to policy commitments, specific actions with SMART attributes, responsible persons, resource requirements, timelines, and methods for evaluating results |
| 5 | `operational_control_template.py` | MD, HTML, PDF, JSON | Clause 8.1 operational control documentation: operating criteria for SEUs, maintenance procedures affecting energy performance, criteria for energy-significant procurement, and criteria for energy-significant design |
| 6 | `performance_report_template.py` | MD, HTML, PDF, JSON | Clause 9.1 performance evaluation report: EnPI values vs. targets, CUSUM charts with control limits, normalized savings vs. baseline, statistical validation of improvements, and recommended actions for off-target performance |
| 7 | `internal_audit_template.py` | MD, HTML, PDF, JSON | Clause 9.2 internal audit program and report: audit scope, criteria, frequency, methods, auditor competence requirements, findings by clause with objective evidence, nonconformity classification, and corrective action requirements |
| 8 | `management_review_template.py` | MD, HTML, PDF, JSON | Clause 9.3 management review minutes: all required inputs documented, discussion summary, decisions and actions with responsibilities and deadlines, resource allocation decisions, and continual improvement commitments |
| 9 | `corrective_action_template.py` | MD, HTML, PDF, JSON | Clause 10.2 corrective action record: nonconformity description, root cause analysis (5-why, fishbone), corrective action defined, implementation evidence, effectiveness verification, closure sign-off, and link to EnMS clause |
| 10 | `enms_documentation_template.py` | MD, HTML, PDF, JSON | Complete EnMS documentation package: scope statement, energy policy, energy review summary, SEU register, EnPI register with baselines, action plan summary, operational control summaries, monitoring plan, internal audit program, management review schedule, and document control register |

### 3.5 Integrations

| # | Integration | Purpose |
|---|-------------|---------|
| 1 | `pack_orchestrator.py` | 10-phase DAG pipeline: FacilitySetup -> DataIngestion -> EnergyReview -> SEUAnalysis -> BaselineEstablishment -> EnPIConfiguration -> ActionPlanning -> ComplianceAssessment -> PerformanceMonitoring -> ManagementReview. Conditional phases for CUSUM alerting (triggered by monitoring data), M&V verification (triggered by action plan completion), and audit preparation (triggered by certification timeline). Retry with exponential backoff, SHA-256 provenance chain, phase-level caching. |
| 2 | `mrv_enms_bridge.py` | Routes to AGENT-MRV agents for GHG emissions linked to energy consumption: MRV-001 (Stationary Combustion for fuel-based SEUs), MRV-002 (Refrigerants for cooling system SEUs), MRV-009/010 (Scope 2 Location/Market-Based for electricity SEUs), MRV-011/012 (Steam/Cooling for district energy), MRV-016 (Fuel & Energy Activities Category 3). Bi-directional: MRV provides emission factors; EnMS provides energy performance data for emission reduction tracking. |
| 3 | `data_enms_bridge.py` | Routes to AGENT-DATA agents: DATA-002 (Excel/CSV for utility bills, meter exports, equipment inventories), DATA-003 (ERP/Finance for energy procurement costs and budget tracking), DATA-010 (Data Quality Profiler for metering data assessment), DATA-019 (Validation Rule Engine for EnMS data quality rules). |
| 4 | `pack031_bridge.py` | PACK-031 Industrial Energy Audit integration: imports energy audit findings as input to energy review (Clause 6.3), equipment efficiency data for SEU analysis, baseline models from audit baselines, and savings opportunities for action plan seeding. Avoids duplicate SEU identification where PACK-031 audit data is available. |
| 5 | `pack032_bridge.py` | PACK-032 Building Energy Assessment integration: imports building energy performance data (EPC ratings, HVAC system assessments, envelope data) for facilities with significant building energy use. Building assessment results feed SEU identification for building-type facilities (offices, data centers, retail). |
| 6 | `pack033_bridge.py` | PACK-033 Quick Wins Identifier integration: imports prioritized quick win measures as candidate actions for Clause 6.2 action plans. Quick win payback calculations feed action plan financial analysis. Implementation status from PACK-033 feeds EnMS action plan tracking. |
| 7 | `eed_compliance_bridge.py` | EU Energy Efficiency Directive 2023/1791 compliance integration: Article 12 mandatory EnMS tracking for enterprises >85 TJ/year, Article 11 audit exemption status for ISO 50001 certified organizations, reporting obligation management, and regulatory deadline tracking. |
| 8 | `bms_scada_bridge.py` | Building Management System (BMS) and SCADA data integration for real-time operational data: energy metering data, equipment runtimes, temperature setpoints, HVAC schedules, alarm data. Supports BACnet/IP, Modbus TCP/RTU, and OPC-UA protocols. Critical for operational control monitoring (Clause 8.1) and CUSUM data feeds. |
| 9 | `metering_bridge.py` | Sub-metering hierarchy management: meter registry (fiscal, sub, virtual), meter data validation (missing data detection, range checking, summation checking), automatic gap filling for short outages, and metering plan alignment with SEU monitoring requirements (Clause 6.6). Supports 15-minute, hourly, daily, and monthly interval data. |
| 10 | `health_check.py` | 15-category system verification covering all 10 engines, 8 workflows, database connectivity, cache status, MRV bridge connectivity, DATA bridge connectivity, PACK-031/032/033 bridge connectivity, BMS/SCADA connectivity, metering data freshness, weather data availability, baseline model validity, and authentication/authorization status. |
| 11 | `setup_wizard.py` | 8-step guided EnMS configuration: (1) organization context and scope boundary, (2) energy carrier registry and utility accounts, (3) metering infrastructure and sub-metering hierarchy, (4) production and activity data sources, (5) weather station selection and degree-day configuration, (6) SEU identification criteria and thresholds, (7) EnPI type selection and baseline period, (8) management review schedule and internal audit program. |
| 12 | `certification_body_bridge.py` | Certification body (CB) audit interface: generates document packages for Stage 1 (documentation review) and Stage 2 (implementation audit), tracks audit findings and nonconformities, manages corrective action timelines, tracks certificate validity dates, and provides surveillance audit preparation (annual) and recertification audit preparation (3-year cycle). |

### 3.6 Presets

| # | Preset | Facility Type | Key Characteristics |
|---|--------|--------------|---------------------|
| 1 | `manufacturing.yaml` | Discrete/Batch Manufacturing | Motor-driven systems 60-75% of electricity, production-normalized baselines, shift-based energy patterns, compressed air as SEU, IE motor class tracking, production volume as primary relevant variable. Typical SEUs: compressed air (20-35%), motors/drives (30-50%), process heat (10-25%). |
| 2 | `commercial_office.yaml` | Commercial Office Building | HVAC 40-55% of energy, lighting 20-30%, plug loads 10-20%; HDD/CDD-dominated baseline, occupancy as relevant variable, BMS integration critical, building envelope performance. Typical SEUs: HVAC system, lighting system, IT infrastructure. |
| 3 | `data_center.yaml` | Data Center | Cooling 35-45% of energy, IT load 45-55%; PUE as primary EnPI, IT load as relevant variable, precision cooling optimization, UPS efficiency tracking, free cooling potential. Typical SEUs: CRAC/CRAH units, chillers, UPS systems. |
| 4 | `healthcare.yaml` | Hospital / Healthcare Facility | 24/7 operation, HVAC 45-60%, strict ventilation requirements (ACH rates), steam for sterilization, medical gas systems, emergency power systems, compliance constraints on temperature/humidity ranges. Typical SEUs: HVAC, steam, DHW, lighting. |
| 5 | `retail_chain.yaml` | Retail Chain (Multi-Site) | Lighting 30-40%, HVAC 30-40%, refrigeration 15-30% (food retail); multi-site portfolio baselines, sales-normalized EnPIs, extended operating hours, seasonal occupancy variation. Corporate rollout configuration for 10-500+ sites. |
| 6 | `logistics_warehouse.yaml` | Logistics & Warehouse | Lighting 35-50% (high-bay), HVAC for temperature-controlled zones, dock door losses, MHE (material handling equipment) charging, dispatch volume as relevant variable. Large floor area with zoned energy management. |
| 7 | `food_processing.yaml` | Food & Beverage Processing | Refrigeration 20-40%, steam/hot water 25-40%, compressed air 10-15%; seasonal production variation, product mix as relevant variable, CIP (clean-in-place) energy, hygiene constraints on heat recovery, production batch normalization. |
| 8 | `sme_multi_site.yaml` | SME Multi-Site (<250 employees) | Simplified 6-engine flow (SEU analyzer, baseline, EnPI, compliance checker, action plan, management review); reduced data requirements; aggregate-level analysis; pre-populated typical EnPI ranges by sector; guided walkthrough for first-time ISO 50001 implementers; EED SME exemption check. |

---

## 4. Engine Specifications

### 4.1 Engine 1: SEU Analyzer Engine

**Purpose:** Identify Significant Energy Uses per ISO 50001 Clause 6.3 using Pareto analysis and multi-criteria determination.

**SEU Identification Methodology:**

| Step | Method | Output |
|------|--------|--------|
| 1. Energy disaggregation | Break total consumption into systems/subsystems using meter data and engineering estimates | Per-system energy consumption (kWh/yr, % of total) |
| 2. Pareto analysis | Rank systems by consumption magnitude, identify 80/20 threshold | Ranked list with cumulative percentage |
| 3. Improvement potential assessment | Compare current efficiency against benchmark/BAT for each system | Estimated savings range (kWh/yr, %) |
| 4. Planned change evaluation | Flag systems with upcoming modifications, expansions, or replacements | Change impact assessment |
| 5. Multi-criteria scoring | Weighted score across consumption (40%), improvement potential (35%), planned changes (25%) | SEU classification (Significant / Not Significant) |

**SEU Hierarchical Decomposition:**

```
Facility (total energy)
  -> System (e.g., Compressed Air System = 25% of total)
    -> Subsystem (e.g., Compressor Room = 80% of CA system)
      -> Equipment (e.g., Compressor #1 = 45% of compressor room)
```

**Key Models:**
- `SEUInput` - Facility energy data, meter hierarchy, equipment inventory, production data, benchmark database
- `SEUResult` - Ranked SEU list with consumption, percentage, improvement potential, relevant variables, recommended EnPIs, criteria scores
- `ParetoAnalysis` - Cumulative percentage, 80/20 threshold, A/B/C classification
- `SEUProfile` - SEU name, boundary, energy carriers, relevant variables, current EnPI, benchmark EnPI, gap percentage

**Non-Functional Requirements:**
- Full facility SEU analysis (500+ equipment items, 12 months data): <5 minutes
- Pareto ranking and multi-criteria scoring: <30 seconds
- Reproducibility: bit-perfect (SHA-256 verified)

### 4.2 Engine 2: Energy Baseline Engine

**Purpose:** Establish ISO 50006-compliant energy baselines using multivariate regression.

**Baseline Types:**

| Type | Description | Use Case |
|------|-------------|----------|
| Static | Fixed 12-month reference period | Initial EnMS implementation, M&V baseline |
| Dynamic | Rolling 12-month recalculated monthly | Continuous performance monitoring |
| Adjusted | Static with normalization for structural changes | Post-expansion or equipment replacement |
| Segmented | Separate baselines for distinct operating modes | Shift-based or seasonal facilities |

**Regression Validation per ASHRAE 14-2014:**

| Statistic | Threshold | Description |
|-----------|-----------|-------------|
| R-squared | >= 0.75 (required), >= 0.85 (target) | Proportion of variance explained by model |
| CV(RMSE) | <= 25% (monthly), <= 30% (daily) | Coefficient of variation of root mean square error |
| NMBE | +/- 10% | Normalized mean bias error |
| t-test (coefficients) | p < 0.05 | Statistical significance of regression coefficients |
| F-test (model) | p < 0.05 | Overall model significance |
| Durbin-Watson | 1.5 - 2.5 | Autocorrelation check |

**Key Models:**
- `BaselineInput` - Facility ID, energy time series (monthly/weekly/daily/15-min), production data, weather data, operating schedule, meter hierarchy
- `BaselineResult` - Regression coefficients, R-squared, CV(RMSE), NMBE, EnPI values, normalization factors, statistical validation pass/fail, residual analysis
- `RegressionModel` - Coefficients, p-values, confidence intervals, variable importance, equation string
- `BaselineAdjustment` - Adjustment type (weather, production, structural), adjustment factor, justification, approval status

### 4.3 Engine 3: EnPI Calculator Engine

**Purpose:** Calculate and validate EnPIs across all ISO 50006 types.

**EnPI Types:**

| Type | Formula | Example | Best For |
|------|---------|---------|----------|
| Simple ratio | Energy / Activity | kWh / tonne product | Simple, stable processes |
| Regression-based | E = a + b*X1 + c*X2 | kWh = f(tonnes, HDD) | Multi-variable facilities |
| CUSUM | Sum(E_actual - E_expected) | Cumulative kWh deviation | Ongoing monitoring |
| Intensity index | (E_actual / E_baseline_adj) * 100 | 95 = 5% improvement | Year-over-year tracking |
| SEC | E / Production_normalized | GJ/tonne_norm | Benchmarking |

**EnPI Validation Tests:**

| Test | Method | Pass Criteria |
|------|--------|---------------|
| Sensitivity | Inject known efficiency change, verify EnPI response | EnPI changes by >= 80% of injected change |
| Specificity | Inject non-efficiency change (weather, production), verify minimal EnPI response | EnPI changes by <= 20% of injected change |
| Reproducibility | Repeat calculation with identical inputs | Bit-perfect match (SHA-256) |
| Stability | Calculate EnPI for 12 consecutive periods of stable operation | CV <= 10% |

**Key Models:**
- `EnPIInput` - SEU identifier, energy data, activity data, baseline model, reporting period
- `EnPIResult` - EnPI value, confidence interval, trend (improving/stable/degrading), improvement vs. baseline (%), statistical validity
- `EnPIDefinition` - Name, formula type, parameters, baseline value, target value, frequency, responsible person

### 4.4 Engine 4: CUSUM Monitor Engine

**Purpose:** Continuous CUSUM monitoring for early performance drift detection.

**CUSUM Calculation:**

```
CUSUM_t = CUSUM_(t-1) + (E_actual_t - E_expected_t)

Where:
  E_expected_t = baseline regression model evaluated with period-t relevant variables
  CUSUM_0 = 0 (start of monitoring period)

Downward CUSUM (improving): cumulative savings accumulating
Upward CUSUM (degrading): cumulative excess energy accumulating
```

**Control Limits:**

| Limit | Calculation | Action |
|-------|-------------|--------|
| Warning (2-sigma) | +/- 2 * sigma * sqrt(n) | Investigate, increase monitoring frequency |
| Action (3-sigma) | +/- 3 * sigma * sqrt(n) | Root cause analysis, corrective action required |
| Run rule | 8+ consecutive points above/below zero | Sustained trend, likely structural change |

**Alert Types:**

| Alert | Trigger | Priority |
|-------|---------|----------|
| Performance degradation | CUSUM crosses upper action limit | High |
| Sustained negative trend | 8+ periods of increasing CUSUM | Medium |
| Performance improvement | CUSUM crosses lower action limit (favorable) | Informational |
| Baseline validity | Residual pattern indicates model drift | Medium |
| Data gap | Missing metering data for >2 consecutive periods | High |

**Key Models:**
- `CUSUMInput` - Baseline model, actual energy data, relevant variable data, control limit parameters
- `CUSUMResult` - CUSUM value per period, control limit status, trend direction, drift magnitude, alert list
- `CUSUMAlert` - Alert type, severity, trigger period, CUSUM value, recommended action

### 4.5 Engine 5: Degree Day Engine

**Purpose:** HDD/CDD calculation with change-point models for weather normalization.

**Degree Day Calculation:**

```
HDD_daily = max(0, T_base_heating - T_avg_daily)
CDD_daily = max(0, T_avg_daily - T_base_cooling)

HDD_monthly = Sum(HDD_daily) for all days in month
CDD_monthly = Sum(CDD_daily) for all days in month
```

**Change-Point Models:**

| Model | Parameters | Application |
|-------|-----------|-------------|
| 3-parameter (3P) | Baseload + heating slope + balance point | Heating-only facilities |
| 4-parameter (4P) | Baseload + heating slope + cooling slope + single balance point | Facilities with heating and cooling, narrow deadband |
| 5-parameter (5P) | Baseload + heating slope + cooling slope + heating balance point + cooling balance point | Full HVAC with distinct heating/cooling balance points |

**Weather Data Sources:**

| Source | Coverage | Resolution | Latency |
|--------|----------|------------|---------|
| NOAA ISD | Global (30,000+ stations) | Hourly | 24-48 hours |
| Meteostat | Global | Hourly/Daily | 24 hours |
| Open-Meteo | Global (reanalysis) | Hourly | Real-time |
| TMY3 | USA (1,020 stations) | Hourly (typical year) | Static dataset |
| CIBSE TRY | UK (14 locations) | Hourly (test reference year) | Static dataset |

**Key Models:**
- `DegreeDayInput` - Facility location (lat/lon), weather station selection, base temperatures, date range
- `DegreeDayResult` - HDD/CDD per period, balance points (determined or configured), change-point model parameters, long-term normals, TMY comparison
- `ChangePointModel` - Model type (3P/4P/5P), balance point(s), slopes, baseload, R-squared, equation

### 4.6 Engine 6: Energy Balance Engine

**Purpose:** Construct facility energy balance with Sankey data and identify metering gaps.

**Balance Structure:**

```
Total Energy Input = Useful Energy Output + Conversion Losses + Distribution Losses + End-Use Losses + Unaccounted

Balance Residual = |Total Input - (Sum of all identified outputs and losses)| / Total Input * 100%

Target: Residual < 5% (good), < 2% (excellent)
Residual > 10%: metering infrastructure inadequate for EnMS purposes
```

**Sankey Data Elements:**

| Element | Description | Data Source |
|---------|-------------|-------------|
| Source nodes | Energy carriers (electricity, gas, fuel oil, district heat, solar) | Fiscal meters, utility bills |
| Conversion nodes | Boilers, chillers, CHP, transformers, compressors | Sub-meters, nameplate data |
| Distribution nodes | Steam headers, chilled water loops, electrical distribution, compressed air mains | Sub-meters or engineering estimates |
| End-use nodes | Motors, lighting, HVAC terminal units, process equipment | Sub-meters or disaggregation estimates |
| Loss nodes | Stack losses, distribution losses, transformer losses, standby losses | Engineering calculations |

**Key Models:**
- `EnergyBalanceInput` - Meter hierarchy, utility data, equipment inventory, engineering estimates
- `EnergyBalanceResult` - Sankey node/link data, balance by energy carrier, residual analysis, metering gap list, sub-metering recommendations
- `MeteringGap` - Unmetered SEU, estimated consumption, recommended meter type, estimated cost, priority

### 4.7 Engine 7: Action Plan Engine

**Purpose:** SMART action plan management per ISO 50001 Clause 6.2.

**SMART Validation:**

| Criterion | Validation Rule | Example |
|-----------|----------------|---------|
| Specific | Linked to named SEU, named EnPI, specific measure description | "Install VSD on AHU-1 supply fan motor" |
| Measurable | Quantified energy savings target (kWh/yr or %) with measurement method | "Reduce AHU-1 electricity by 35,000 kWh/yr (25%)" |
| Achievable | Savings within engineering calculation range, confirmed by PACK-031/033 analysis | Engineering calc confirms 30-40% fan energy reduction |
| Relevant | Linked to energy policy commitment and SEU improvement | Links to Policy Item 3: "Improve HVAC efficiency" |
| Time-bound | Start date, end date, milestones with calendar dates | "Start: 2026-Q2, Complete: 2026-Q3, Verify: 2027-Q1" |

**Action Plan Status Lifecycle:**

```
Proposed -> Approved -> In Progress -> Completed -> Verification -> Verified (or Failed)
                                          |
                                          v
                                      On Hold / Cancelled
```

**Key Models:**
- `ActionPlanInput` - Objective, linked SEU/EnPI, proposed measures (from PACK-031/033 or manual), resources, timeline
- `ActionPlanResult` - SMART validation results, implementation schedule, resource allocation, projected savings, portfolio metrics
- `ActionItem` - ID, description, SEU link, EnPI link, target savings, status, responsible person, start/end dates, actual savings, variance
- `PortfolioMetrics` - Total measures, total investment (EUR), total projected savings (kWh/yr, EUR/yr), aggregate payback, portfolio IRR

### 4.8 Engine 8: Compliance Checker Engine

**Purpose:** Clause-by-clause gap analysis with certification readiness scoring.

**Clause Assessment Matrix:**

| Clause | Requirement Count ("shall" statements) | Evidence Types |
|--------|----------------------------------------|----------------|
| 4 Context | 4 | Scope document, interested party register |
| 5 Leadership | 6 | Energy policy, roles/responsibilities, management commitment evidence |
| 6 Planning | 12 | Energy review, SEU register, EnPI definitions, baselines, objectives/targets, action plans, data collection plan |
| 7 Support | 5 | Competence records, awareness evidence, communication records, documented information |
| 8 Operation | 4 | Operational control procedures, design criteria, procurement criteria |
| 9 Performance Evaluation | 8 | Monitoring records, EnPI data, internal audit program/reports, management review records |
| 10 Improvement | 4 | Nonconformity records, corrective actions, continual improvement evidence |
| **Total** | **39** (mandatory "shall" requirements mapped) | |

**Readiness Scoring:**

```
Clause_score = (conforming_requirements / total_requirements) * 100

Overall_readiness = weighted_average(clause_scores)

Weights: Planning (6) = 25%, Performance Evaluation (9) = 20%, Operation (8) = 15%,
         Leadership (5) = 15%, Improvement (10) = 10%, Support (7) = 10%, Context (4) = 5%
```

**Key Models:**
- `ComplianceInput` - EnMS documentation inventory, evidence references, previous audit findings
- `ComplianceResult` - Per-clause compliance status, overall readiness score (0-100), gap list with remediation recommendations, Stage 1/Stage 2 readiness assessment
- `GapItem` - Clause reference, requirement text, compliance status, evidence gap, remediation action, estimated effort (hours), priority, target date
- `ReadinessScore` - Overall score, clause-level scores, trend (improving/stable/declining), estimated time to certification readiness

### 4.9 Engine 9: Performance Trend Engine

**Purpose:** ISO 50015-compliant performance trending and savings verification.

**Savings Calculation per IPMVP:**

```
Savings = E_baseline_adjusted - E_actual +/- Non-Routine_Adjustments

E_baseline_adjusted = baseline regression model with current-period relevant variables

90% Confidence Interval:
Savings +/- 1.645 * sqrt((CV(RMSE) * E_baseline_adjusted)^2 + measurement_uncertainty^2)
```

**IPMVP Options:**

| Option | Name | Application | Data Requirement |
|--------|------|-------------|------------------|
| A | Retrofit Isolation: Key Parameter | Single equipment/system; measure key parameter, stipulate others | Spot or short-term measurement of key parameter |
| B | Retrofit Isolation: All Parameters | Single equipment/system; measure all parameters | Continuous measurement of all energy parameters |
| C | Whole Facility | Whole-building savings; regression-based | Whole-building meter data (12+ months pre and post) |
| D | Calibrated Simulation | New construction or major renovation | Calibrated energy model (EnergyPlus, eQUEST) |

**Year-over-Year Comparison:**

| Metric | Calculation | Purpose |
|--------|-------------|---------|
| Absolute savings | E_baseline_adj - E_actual (kWh) | Total energy saved |
| Percentage improvement | (E_baseline_adj - E_actual) / E_baseline_adj * 100 | Efficiency improvement rate |
| Cost savings | Savings_kWh * blended_rate (EUR) | Financial value of improvement |
| Carbon reduction | Savings_kWh * emission_factor (tCO2e) | GHG impact |
| Cumulative savings | Sum(annual savings) since EnMS inception | Lifetime EnMS value |

**Key Models:**
- `TrendInput` - Baseline model, actual energy data by period, relevant variables, IPMVP option selection
- `TrendResult` - Savings per period with confidence intervals, cumulative savings, t-test significance, YoY comparison, cost savings, carbon reduction
- `StatisticalTest` - Test type (t-test), test statistic, p-value, degrees of freedom, conclusion (significant/not significant)

### 4.10 Engine 10: Management Review Engine

**Purpose:** Generate Clause 9.3 management review packages with all required inputs and outputs.

**Required Review Inputs (Clause 9.3):**

| Input | Source | Content |
|-------|--------|---------|
| (a) Status of actions from previous reviews | Previous management review records | Action item completion status, carry-forward items |
| (b) Changes in external/internal issues | Organization context register, regulatory tracker | New regulations, organizational changes, market conditions |
| (c) Energy performance and EnPI trends | EnPICalculatorEngine, CUSUMMonitorEngine | EnPI values, CUSUM status, normalized trends |
| (d) Results of evaluation of compliance | ComplianceCheckerEngine, EED bridge | Legal compliance status, regulatory obligation status |
| (e) Status of corrective actions | Corrective action register | Open/closed CAs, overdue CAs, effectiveness verification |
| (f) Projected energy performance | PerformanceTrendEngine | Forward projection based on current trends and planned actions |
| (g) Recommendations for improvement | ActionPlanEngine, PACK-031/033 bridges | New improvement opportunities, technology updates |

**Review Output Structure:**

| Output | Content | Format |
|--------|---------|--------|
| Overall EnMS adequacy | Assessment of whether EnMS meets organizational needs | Traffic light (Green/Amber/Red) with narrative |
| EnPI performance summary | Dashboard with sparklines, targets, actuals, variances | Table + charts data |
| Resource allocation decisions | Budget approvals, staffing decisions, equipment purchases | Decision log with responsibilities |
| Policy revision needs | Whether energy policy requires updates | Yes/No with recommended changes |
| Continual improvement commitments | Specific improvement actions for next period | Action list with SMART attributes |

**Key Models:**
- `ManagementReviewInput` - Review date, previous review actions, EnPI data, compliance status, corrective actions, projections, improvement recommendations
- `ManagementReviewResult` - Formatted review package, executive dashboard, decision log, action items, policy revision recommendations
- `KPIDashboard` - Per-EnPI: value, target, variance, trend, status (green/amber/red), sparkline data (12-month)

---

## 5. Workflow Specifications

### 5.1 Workflow 1: Energy Review Workflow

**Purpose:** Complete ISO 50001 Clause 6.3 energy review.

| Phase | Steps | Duration |
|-------|-------|----------|
| 1. Data Collection | Import utility bills (12+ months), meter data, production data, weather data; run data quality assessment | <30 minutes |
| 2. SEU Identification | Disaggregate consumption by system, perform Pareto analysis, apply multi-criteria SEU determination | <15 minutes |
| 3. Baseline/EnPI Establishment | Build regression models per SEU, validate statistics, define EnPIs, set improvement targets | <30 minutes |
| 4. Opportunity Register | Identify improvement opportunities per SEU, estimate savings potential, populate action plan candidates | <15 minutes |

### 5.2 Workflow 2: Baseline Establishment Workflow

**Purpose:** ISO 50006-compliant baseline establishment with formal approval.

| Phase | Steps | Duration |
|-------|-------|----------|
| 1. Data Validation | Verify data completeness (12+ months), check for gaps, outliers, meter errors; assess data quality score | <15 minutes |
| 2. Regression Modeling | Automatic variable selection, multivariate regression, change-point modeling, statistical validation (R-sq, CV(RMSE), NMBE) | <10 minutes |
| 3. Baseline Approval | Generate baseline report with model parameters, statistical validation results, EnPI definitions; route for management approval | <5 minutes |

### 5.3 Workflow 3: Action Plan Workflow

**Purpose:** Clause 6.2 objectives and action plan development.

| Phase | Steps | Duration |
|-------|-------|----------|
| 1. Objective Setting | Define strategic objectives linked to energy policy, set energy targets per SEU/EnPI | <30 minutes |
| 2. Action Definition | Define specific actions per objective with SMART validation, link to SEUs and EnPIs | <1 hour |
| 3. Resource Allocation | Assign budgets, personnel, contractors; verify resource availability | <30 minutes |
| 4. Timeline Generation | Set milestones, dependencies, and completion dates; generate Gantt chart data | <15 minutes |

### 5.4 Workflow 4: Operational Control Workflow

**Purpose:** Clause 8.1 operational control for SEUs.

| Phase | Steps | Duration |
|-------|-------|----------|
| 1. Criteria Definition | Define operating criteria for each SEU (setpoints, schedules, maintenance intervals, efficiency thresholds) | <1 hour |
| 2. Monitoring Setup | Configure monitoring parameters, alarm thresholds, and data collection intervals per SEU | <30 minutes |
| 3. Deviation Response | Define response procedures for deviations from operational criteria, escalation paths, and corrective action triggers | <30 minutes |

### 5.5 Workflow 5: Monitoring Workflow

**Purpose:** Clause 9.1 ongoing monitoring, measurement, and analysis.

| Phase | Steps | Duration |
|-------|-------|----------|
| 1. Metering Validation | Verify meter accuracy, check calibration status, validate summation consistency | <10 minutes |
| 2. Data Collection | Ingest meter data (15-min/hourly/daily/monthly), production data, weather data; fill gaps | <5 minutes |
| 3. EnPI Calculation | Calculate all defined EnPIs for current period, compare to targets and baselines | <5 minutes |
| 4. Alert/Reporting | Evaluate CUSUM control limits, generate alerts for deviations, produce monitoring report | <5 minutes |

### 5.6 Workflow 6: Performance Analysis Workflow

**Purpose:** Periodic performance analysis combining EnPI, CUSUM, and trends.

| Phase | Steps | Duration |
|-------|-------|----------|
| 1. EnPI Calculation | Compute all EnPI values for analysis period with confidence intervals | <5 minutes |
| 2. CUSUM Analysis | Update CUSUM charts, evaluate control limits, assess trend direction and magnitude | <5 minutes |
| 3. Trend Reporting | Generate normalized YoY comparison, savings quantification, statistical significance testing | <10 minutes |

### 5.7 Workflow 7: M&V Verification Workflow

**Purpose:** ISO 50015 measurement and verification of savings.

| Phase | Steps | Duration |
|-------|-------|----------|
| 1. Baseline Comparison | Adjust baseline for current-period relevant variables, compute expected energy consumption | <5 minutes |
| 2. Savings Quantification | Calculate savings with 90% confidence intervals, apply non-routine adjustments | <10 minutes |
| 3. Uncertainty Assessment | Compute combined uncertainty from baseline model, measurement, and non-routine sources; verify statistical significance | <5 minutes |

### 5.8 Workflow 8: Audit/Certification Workflow

**Purpose:** Certification preparation from gap analysis to readiness.

| Phase | Steps | Duration |
|-------|-------|----------|
| 1. Gap Analysis | Evaluate all 39 "shall" requirements, identify gaps, assess current evidence | <1 hour |
| 2. Internal Audit | Execute Clause 9.2 internal audit per audit program, document findings and nonconformities | <2 hours |
| 3. Corrective Actions | Define and track corrective actions for all nonconformities, verify effectiveness | Ongoing |
| 4. Certification Readiness | Calculate readiness score, prepare Stage 1/Stage 2 document packages, generate audit schedule | <1 hour |

---

## 6. Configuration

### 6.1 Pack Configuration (Pydantic v2)

```python
class Pack034Config(BaseModel):
    """PACK-034 ISO 50001 EnMS Pack runtime configuration."""

    # Facility
    facility_id: str
    facility_name: str
    facility_type: FacilityType  # Enum: manufacturing, commercial_office, data_center, etc.

    # Energy carriers
    energy_carriers: list[EnergyCarrier]  # electricity, natural_gas, lpg, fuel_oil, etc.

    # Baseline configuration
    baseline_type: BaselineType  # static, dynamic, adjusted, segmented
    baseline_period_months: int = 12
    regression_r_squared_minimum: Decimal = Decimal("0.75")
    regression_r_squared_target: Decimal = Decimal("0.85")
    cv_rmse_maximum: Decimal = Decimal("0.25")
    nmbe_maximum: Decimal = Decimal("0.10")

    # Degree-day configuration
    hdd_base_temperature_c: Decimal = Decimal("15.5")
    cdd_base_temperature_c: Decimal = Decimal("22.0")
    weather_station_id: str | None = None

    # CUSUM configuration
    cusum_warning_sigma: Decimal = Decimal("2.0")
    cusum_action_sigma: Decimal = Decimal("3.0")
    cusum_run_rule_length: int = 8

    # SEU configuration
    seu_pareto_threshold: Decimal = Decimal("0.80")  # 80% of total consumption
    seu_improvement_weight: Decimal = Decimal("0.35")
    seu_consumption_weight: Decimal = Decimal("0.40")
    seu_planned_change_weight: Decimal = Decimal("0.25")

    # Compliance
    target_certification_date: date | None = None
    certification_body: str | None = None
    eed_applicable: bool = False
    eed_consumption_tj: Decimal | None = None  # For Article 12 threshold check

    # Management review
    review_frequency_months: int = 6  # Typically every 6 or 12 months

    # M&V
    ipmvp_default_option: str = "C"  # A, B, C, or D
    confidence_level: Decimal = Decimal("0.90")
```

---

## 7. Security Requirements

- JWT RS256 authentication
- RBAC with 6 roles: `energy_manager`, `enms_representative`, `internal_auditor`, `facility_engineer`, `external_auditor`, `admin`
- Facility-level access control (users see only facilities assigned to them)
- AES-256-GCM encryption at rest for all energy data, EnPI calculations, baseline models, and compliance assessments
- TLS 1.3 for data in transit
- SHA-256 provenance hashing on all calculation outputs (SEU analysis, baselines, EnPIs, CUSUM, compliance scores, savings verification)
- Full audit trail per SEC-005 (who changed what, when, with provenance chain)
- BMS/SCADA credentials encrypted via Vault (SEC-006)
- Metering and utility credentials encrypted via Vault
- Read-only mode for external auditors (certification body auditors see documentation and performance data, cannot modify)
- Data retention: minimum 5 years for EnMS operational records, 8 years for compliance and financial records, indefinite for baselines and EnPI definitions

**RBAC Permission Matrix:**

| Permission | energy_manager | enms_representative | internal_auditor | facility_engineer | external_auditor | admin |
|------------|---------------|--------------------|--------------------|-------------------|------------------|-------|
| Create/edit facility | Yes | Yes | No | No | No | Yes |
| Upload energy data | Yes | Yes | No | Yes | No | Yes |
| Define/modify SEUs | Yes | Yes | No | No | No | Yes |
| Establish/modify baselines | Yes | Yes | No | Yes | No | Yes |
| Define/modify EnPIs | Yes | Yes | No | No | No | Yes |
| Run CUSUM monitoring | Yes | Yes | No | Yes | No | Yes |
| Create/edit action plans | Yes | Yes | No | No | No | Yes |
| Run compliance assessment | Yes | Yes | Yes | No | No | Yes |
| Conduct internal audit | No | Yes | Yes | No | No | Yes |
| View performance reports | Yes | Yes | Yes | Yes | Yes (assigned) | Yes |
| Generate management review | Yes | Yes | No | No | No | Yes |
| Generate CB audit package | Yes | Yes | No | No | Yes (assigned) | Yes |
| Export data | Yes | Yes | Yes | Yes | Yes (assigned) | Yes |
| Manage users | No | No | No | No | No | Yes |
| View all facilities | No | Yes | Yes | No | No | Yes |
| Delete records | No | No | No | No | No | Yes |

---

## 8. Database Migrations

Inherits platform migrations V001-V255. Pack-specific migrations:

| Migration | Tables | Purpose |
|-----------|--------|---------|
| V256__pack034_enms_core_001 | `enms_facilities`, `enms_facility_profiles`, `enms_energy_carriers`, `enms_scope_definitions`, `enms_stakeholder_register` | Core EnMS facility registry with scope boundary definitions (Clause 4.3), energy carrier registry, facility type profiles, and interested party/stakeholder register (Clause 4.2) |
| V257__pack034_seu_002 | `enms_seu_register`, `enms_seu_criteria`, `enms_pareto_analysis`, `enms_seu_variables`, `enms_seu_history` | SEU identification and tracking: SEU register with multi-criteria scores, Pareto analysis results, relevant variable definitions per SEU, and SEU classification history for change tracking |
| V258__pack034_baseline_003 | `enms_baselines`, `enms_regression_models`, `enms_baseline_adjustments`, `enms_variable_selection`, `enms_baseline_approvals` | Energy baseline establishment: baseline definitions with regression model parameters, statistical validation results (R-squared, CV(RMSE), NMBE), baseline adjustment records for structural changes, variable selection history, and formal baseline approval workflow |
| V259__pack034_enpi_004 | `enms_enpi_definitions`, `enms_enpi_values`, `enms_enpi_targets`, `enms_enpi_validation`, `enms_enpi_history` | EnPI management: EnPI definitions per ISO 50006 (type, formula, parameters), calculated EnPI values per reporting period, target values linked to objectives, statistical validation test results, and EnPI change history |
| V260__pack034_cusum_005 | `enms_cusum_series`, `enms_cusum_alerts`, `enms_control_limits`, `enms_drift_detections` | CUSUM monitoring: CUSUM time series data per EnPI/SEU, alert records with severity and response status, control limit definitions (warning/action sigma), and drift detection results with onset period and magnitude |
| V261__pack034_action_plans_006 | `enms_objectives`, `enms_energy_targets`, `enms_action_plans`, `enms_action_items`, `enms_resource_allocations`, `enms_milestones` | Action plan management: strategic objectives linked to energy policy, quantified energy targets per SEU/EnPI, action plans with SMART validation, individual action items with status lifecycle, resource allocation records, and milestone tracking |
| V262__pack034_compliance_007 | `enms_clause_assessments`, `enms_gap_items`, `enms_readiness_scores`, `enms_remediation_plans`, `enms_audit_findings` | Compliance and certification: per-clause compliance assessments against all 39 requirements, gap items with remediation recommendations, readiness score history, remediation plan tracking, and internal audit finding records |
| V263__pack034_monitoring_008 | `enms_meter_registry`, `enms_meter_data`, `enms_energy_balances`, `enms_sankey_data`, `enms_metering_gaps` | Monitoring infrastructure: meter hierarchy registry (fiscal/sub/virtual), validated meter data storage, energy balance calculations with Sankey node/link data, and metering gap identification with sub-metering recommendations |
| V264__pack034_performance_009 | `enms_performance_periods`, `enms_savings_verification`, `enms_statistical_tests`, `enms_trend_analysis`, `enms_management_reviews` | Performance evaluation: reporting period definitions, ISO 50015 savings verification results with confidence intervals, statistical significance test records, trend analysis data, and management review records with inputs/outputs/decisions |
| V265__pack034_views_indexes_010 | Views, indexes, RLS policies, functions | Performance views (SEU dashboard, EnPI trends, CUSUM charts, compliance status, action plan progress), composite indexes on (facility_id, created_at) for time-series queries, GIN indexes on JSONB columns, partial indexes on status columns, RLS policies for facility-level access control, and utility functions for EnPI calculation and CUSUM computation |

**Table Prefix:** `enms_` (Energy Management System)

**Row-Level Security (RLS):**
- All tables have `facility_id` column for facility-level access control
- RLS policies enforce that users can only see data for facilities assigned to their role
- External auditors (CB auditors) have read-only access to specifically assigned facility records
- Internal auditors can view all facilities within their organizational scope
- Admin role bypasses RLS for cross-facility reporting

**Indexes:**
- Composite indexes on `(facility_id, created_at)` for time-series queries
- Composite indexes on `(facility_id, seu_id)` for SEU-specific queries
- GIN indexes on JSONB columns for flexible metadata storage (regression parameters, Sankey data, EnPI definitions)
- Partial indexes on `status` columns for active-record filtering (active baselines, open action items, unresolved gaps)
- B-tree indexes on `enpi_id`, `seu_id`, `baseline_id`, `action_plan_id` for foreign key joins
- Full-text search index on `enms_gap_items.description` for gap searching

---

## 9. File Structure

```
packs/energy-efficiency/PACK-034-iso-50001-enms/
  __init__.py
  pack.yaml
  config/
    __init__.py
    pack_config.py
    demo/
      __init__.py
      demo_config.yaml
    presets/
      __init__.py
      manufacturing.yaml
      commercial_office.yaml
      data_center.yaml
      healthcare.yaml
      retail_chain.yaml
      logistics_warehouse.yaml
      food_processing.yaml
      sme_multi_site.yaml
  engines/
    __init__.py
    seu_analyzer_engine.py
    energy_baseline_engine.py
    enpi_calculator_engine.py
    cusum_monitor_engine.py
    degree_day_engine.py
    energy_balance_engine.py
    action_plan_engine.py
    compliance_checker_engine.py
    performance_trend_engine.py
    management_review_engine.py
  workflows/
    __init__.py
    energy_review_workflow.py
    baseline_establishment_workflow.py
    action_plan_workflow.py
    operational_control_workflow.py
    monitoring_workflow.py
    performance_analysis_workflow.py
    mv_verification_workflow.py
    audit_certification_workflow.py
  templates/
    __init__.py
    energy_policy_template.py
    energy_review_report_template.py
    enpi_methodology_template.py
    action_plan_template.py
    operational_control_template.py
    performance_report_template.py
    internal_audit_template.py
    management_review_template.py
    corrective_action_template.py
    enms_documentation_template.py
  integrations/
    __init__.py
    pack_orchestrator.py
    mrv_enms_bridge.py
    data_enms_bridge.py
    pack031_bridge.py
    pack032_bridge.py
    pack033_bridge.py
    eed_compliance_bridge.py
    bms_scada_bridge.py
    metering_bridge.py
    health_check.py
    setup_wizard.py
    certification_body_bridge.py
  tests/
    __init__.py
    conftest.py
    test_manifest.py
    test_config.py
    test_seu_analyzer_engine.py
    test_energy_baseline_engine.py
    test_enpi_calculator_engine.py
    test_cusum_monitor_engine.py
    test_degree_day_engine.py
    test_energy_balance_engine.py
    test_action_plan_engine.py
    test_compliance_checker_engine.py
    test_performance_trend_engine.py
    test_management_review_engine.py
    test_workflows.py
    test_templates.py
    test_integrations.py
    test_presets.py
    test_e2e.py
```

**Migration Files:**

```
deployment/database/migrations/sql/
  V256__pack034_enms_core_001.sql
  V256__pack034_enms_core_001.down.sql
  V257__pack034_seu_002.sql
  V257__pack034_seu_002.down.sql
  V258__pack034_baseline_003.sql
  V258__pack034_baseline_003.down.sql
  V259__pack034_enpi_004.sql
  V259__pack034_enpi_004.down.sql
  V260__pack034_cusum_005.sql
  V260__pack034_cusum_005.down.sql
  V261__pack034_action_plans_006.sql
  V261__pack034_action_plans_006.down.sql
  V262__pack034_compliance_007.sql
  V262__pack034_compliance_007.down.sql
  V263__pack034_monitoring_008.sql
  V263__pack034_monitoring_008.down.sql
  V264__pack034_performance_009.sql
  V264__pack034_performance_009.down.sql
  V265__pack034_views_indexes_010.sql
  V265__pack034_views_indexes_010.down.sql
```

**Total File Count:** ~112 files
- 10 engines + engines/__init__.py = 11
- 8 workflows + workflows/__init__.py = 9
- 10 templates + templates/__init__.py = 11
- 12 integrations + integrations/__init__.py = 13
- Config: pack_config.py + 8 presets + demo config + 3 __init__.py = 14
- Root: __init__.py + pack.yaml = 2
- Tests: 15 test files + conftest.py + __init__.py = 17
- Migrations: 20 SQL files (10 up + 10 down) = 20
- Subtotal directory __init__.py files: ~5

---

## 10. Testing Requirements

| Test Type | Coverage Target | Scope |
|-----------|-----------------|-------|
| Unit Tests | >90% line coverage | All 10 engines, all config models, all presets |
| Workflow Tests | >85% | All 8 workflows with synthetic facility data |
| Template Tests | 100% | All 10 templates in 3+ formats (MD, HTML, JSON, PDF where applicable) |
| Integration Tests | >80% | All 12 integrations with mock agents, BMS simulators, and metering data |
| E2E Tests | Core happy path | Full pipeline from facility setup to management review package |
| SEU Tests | 100% | Pareto analysis, multi-criteria scoring, hierarchical decomposition with 200+ equipment scenarios |
| Baseline Tests | 100% | Regression validation (R-squared, CV(RMSE), NMBE), all 4 baseline types, change-point models |
| EnPI Tests | 100% | All 5 EnPI types, sensitivity/specificity validation, ISO 50006 worked examples |
| CUSUM Tests | 100% | Control limit calculation, drift detection, alert generation, run rule evaluation |
| Compliance Tests | 100% | All 39 "shall" requirements mapped, readiness scoring, gap identification |
| M&V Tests | 100% | IPMVP Options A-D, savings calculation with confidence intervals, statistical significance |
| Management Review Tests | 100% | All Clause 9.3 inputs/outputs, KPI dashboard generation, decision log |
| Preset Tests | 100% | All 8 facility-type presets with representative scenarios |
| Manifest Tests | 100% | pack.yaml validation, component counts, version |

**Test Count Target:** 800+ tests (60-80 per engine, 40-50 integration, 20-30 E2E)

**Known-Value Validation Sets:**
- 100 EnPI calculations validated against ISO 50006 worked examples and manual engineering calculations
- 50 regression model validations against ASHRAE 14-2014 statistical criteria
- 30 CUSUM scenarios with known drift injection validated against manual computation
- 25 degree-day calculations validated against published HDD/CDD databases (NOAA, CIBSE)
- 20 energy balance scenarios validated against metered facility data
- 20 savings verification calculations validated against IPMVP worked examples
- 20 SMART validation scenarios validated against ISO 50001 auditor expectations
- 10 full certification readiness assessments validated against actual CB audit reports

---

## 11. Dependencies

### 11.1 Pack Dependencies

| Dependency | Type | Purpose |
|------------|------|---------|
| PACK-031 (Industrial Energy Audit) | Optional (enhanced) | Import audit findings, equipment efficiency data, baseline models for energy review |
| PACK-032 (Building Energy Assessment) | Optional (enhanced) | Import building energy data, EPC ratings, HVAC assessments for building-type facilities |
| PACK-033 (Quick Wins Identifier) | Optional (enhanced) | Import prioritized quick wins as action plan candidates, implementation status tracking |

### 11.2 Agent Dependencies

| Agent Group | Agents | Purpose |
|-------------|--------|---------|
| AGENT-MRV | MRV-001, MRV-002, MRV-009, MRV-010, MRV-011, MRV-012, MRV-016 | Scope 1/2/3 emission factors and calculations linked to energy consumption |
| AGENT-DATA | DATA-002, DATA-003, DATA-010, DATA-019 | Data intake (Excel/CSV, ERP), quality profiling, validation rules |

### 11.3 Infrastructure Dependencies

| Component | Dependency |
|-----------|------------|
| PostgreSQL + TimescaleDB | Time-series meter data storage, EnPI value history |
| Redis | CUSUM cache, session state, alert queue |
| Vault | BMS/SCADA credential encryption, API key management |
| Kong API Gateway | Rate limiting, authentication proxy |

---

## 12. Release Plan

| Phase | Deliverable | Timeline |
|-------|-------------|----------|
| Phase 1 | PRD Approval | 2026-03-21 |
| Phase 2 | Engine implementation (10 engines) | 2026-03-22 to 2026-03-25 |
| Phase 3 | Workflow implementation (8 workflows) | 2026-03-25 to 2026-03-26 |
| Phase 4 | Template implementation (10 templates) | 2026-03-26 to 2026-03-27 |
| Phase 5 | Integration implementation (12 integrations) | 2026-03-27 to 2026-03-28 |
| Phase 6 | Test suite (800+ tests) | 2026-03-28 to 2026-03-30 |
| Phase 7 | Database migrations (V256-V265) | 2026-03-30 |
| Phase 8 | Documentation & Release | 2026-03-31 |

---

## 13. Appendix: ISO 50001:2018 Clause Reference

### Mandatory "Shall" Requirements by Clause

| Clause | Requirement | PACK-034 Implementation |
|--------|-------------|------------------------|
| 4.1 | Determine external and internal issues relevant to EnMS | SetupWizard step 1, stakeholder register |
| 4.2 | Determine interested parties and their requirements | Stakeholder register, regulatory obligation tracker |
| 4.3 | Determine the scope and boundaries of the EnMS | Scope definition in facility profile |
| 4.4 | Establish, implement, maintain and continually improve the EnMS | Full pack orchestration pipeline |
| 5.1 | Top management shall demonstrate leadership and commitment | ManagementReviewEngine, policy template |
| 5.2 | Top management shall establish an energy policy | EnergyPolicyTemplate with mandatory elements |
| 5.3 | Top management shall ensure responsibilities and authorities are assigned | RBAC integration, energy team registry |
| 6.1 | Determine risks and opportunities, plan actions to address them | Risk register linked to action plans |
| 6.2 | Establish objectives, targets, and action plans | ActionPlanEngine with SMART validation |
| 6.3 | Conduct and document the energy review | SEUAnalyzerEngine + EnergyReviewWorkflow |
| 6.4 | Determine EnPIs | EnPICalculatorEngine (all ISO 50006 types) |
| 6.5 | Establish the EnB | EnergyBaselineEngine with regression |
| 6.6 | Plan for energy data collection | MeteringBridge, data collection plan |
| 7.1 | Determine and provide resources | Action plan resource allocation |
| 7.2 | Determine necessary competence | Training/competence tracking |
| 7.3 | Ensure awareness of energy policy and EnMS roles | Awareness documentation templates |
| 7.4 | Determine internal and external communications | Communication plan template |
| 7.5 | Create and control documented information | EnMSDocumentationTemplate, document control |
| 8.1 | Plan and control operations related to SEUs | OperationalControlWorkflow, criteria definition |
| 8.2 | Consider energy performance in design activities | Design criteria documentation |
| 8.3 | Consider energy performance in procurement | Procurement criteria documentation |
| 9.1 | Monitor, measure, analyze and evaluate energy performance | MonitoringWorkflow, CUSUMMonitorEngine |
| 9.2 | Conduct internal audits at planned intervals | InternalAuditTemplate, AuditCertificationWorkflow |
| 9.3 | Review the EnMS at planned intervals | ManagementReviewEngine, ManagementReviewTemplate |
| 10.1 | React to nonconformity and take corrective action | CorrectiveActionTemplate, CA tracking |
| 10.2 | Continually improve the EnMS | PerformanceTrendEngine, improvement tracking |

---

## 14. Appendix: ISO 50006 EnPI Quick Reference

### EnPI Type Selection Guide

| Facility Characteristic | Recommended EnPI Type | Rationale |
|------------------------|----------------------|-----------|
| Single product, stable operations | Simple ratio (kWh/unit) | Easy to understand, adequate for simple processes |
| Multiple products or weather-dependent | Regression-based | Accounts for multiple relevant variables |
| Ongoing performance monitoring | CUSUM | Detects gradual performance drift early |
| Year-over-year executive reporting | Energy Intensity Index | Easy to communicate (100 = baseline, <100 = improvement) |
| Industry benchmarking | SEC (kWh/tonne or GJ/tonne) | Comparable across organizations in same sector |

### Statistical Validation Quick Reference

| Statistic | Formula | Acceptable Range | Reference |
|-----------|---------|------------------|-----------|
| R-squared | 1 - (SS_res / SS_total) | >= 0.75 (monthly) | ASHRAE 14-2014 |
| CV(RMSE) | sqrt(sum((y_i - y_hat_i)^2) / n) / y_bar * 100% | <= 25% (monthly), <= 30% (daily) | ASHRAE 14-2014 |
| NMBE | sum(y_i - y_hat_i) / (n * y_bar) * 100% | +/- 10% (monthly) | ASHRAE 14-2014 |
| t-statistic | coefficient / standard_error | |t| > 2.0 (p < 0.05) | Standard statistics |
| F-statistic | (SS_regression / df_reg) / (SS_residual / df_res) | p < 0.05 | Standard statistics |
| Durbin-Watson | d = sum((e_t - e_{t-1})^2) / sum(e_t^2) | 1.5 - 2.5 (no autocorrelation) | Standard statistics |

---

## 15. Appendix: CUSUM Interpretation Guide

### CUSUM Chart Patterns

| Pattern | Interpretation | Action |
|---------|---------------|--------|
| Horizontal (fluctuating around zero) | Energy performance matches baseline; no significant change | Continue monitoring; no action needed |
| Downward slope (negative cumulative) | Energy consumption consistently below baseline; improvement | Document savings, verify with ISO 50015, report in management review |
| Upward slope (positive cumulative) | Energy consumption consistently above baseline; degradation | Investigate root cause, initiate corrective action per Clause 10.1 |
| Step change (sudden shift) | Abrupt performance change (equipment failure, operational change) | Identify cause; if structural, consider baseline adjustment |
| Gradual slope change | Slow performance drift (degradation, fouling, calibration drift) | Schedule maintenance investigation, increase monitoring frequency |
| Return to zero after deviation | Temporary performance anomaly that self-corrected | Document cause (weather event, production anomaly, temporary fault) |

---

## 16. Appendix: Degree-Day Quick Reference

### Default Base Temperatures by Region

| Region | HDD Base (C) | CDD Base (C) | Reference |
|--------|-------------|-------------|-----------|
| EU (default) | 15.5 | 22.0 | EN ISO 15927-6 |
| UK | 15.5 | 22.0 | CIBSE TM41 |
| USA | 18.3 (65F) | 18.3 (65F) | ASHRAE |
| Germany | 15.0 | 18.3 | VDI 3807 |
| France | 18.0 | 21.0 | RT 2012 |

### Change-Point Model Selection

| Model | Parameters | When to Use |
|-------|-----------|-------------|
| 3P Heating | Baseload, heating slope, heating balance point | Heating-only buildings/processes (warehouses, northern climate) |
| 3P Cooling | Baseload, cooling slope, cooling balance point | Cooling-only facilities (data centers, tropical climate) |
| 4P | Baseload, heating slope, cooling slope, single balance point | Buildings with narrow HVAC deadband |
| 5P | Baseload, heating slope, cooling slope, heating BP, cooling BP | Full HVAC buildings with distinct heating and cooling regimes |

---

*Document version: 1.0.0 | Status: Approved | Last updated: 2026-03-21*
