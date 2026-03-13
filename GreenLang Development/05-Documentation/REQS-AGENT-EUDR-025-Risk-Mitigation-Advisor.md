# REQS: AGENT-EUDR-025 -- Risk Mitigation Advisor Agent
# Regulatory Requirements Document

## Document Info

| Field | Value |
|-------|-------|
| **REQS ID** | REQS-AGENT-EUDR-025 |
| **Agent ID** | GL-EUDR-RMA-025 |
| **Component** | Risk Mitigation Advisor Agent |
| **Category** | EUDR Regulatory Agent -- Risk Mitigation & Remediation Management |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Author** | GL-RegulatoryIntelligence |
| **Date** | 2026-03-11 |
| **Regulation** | Regulation (EU) 2023/1115 (EUDR), Articles 8, 9, 10, 11, 12, 13, 18-23, 29, 31; ISO 31000:2018 (Risk Management Guidelines); ISO 31010:2019 (Risk Assessment Techniques); OECD Due Diligence Guidance for Responsible Supply Chains; UNGP on Business and Human Rights |
| **Enforcement** | December 30, 2026 (large/medium operators); June 30, 2027 (SMEs) |

---

## 1. Regulatory Basis and Legal Framework

### 1.1 Primary Regulation: EU Deforestation Regulation (EUDR)

Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023, concerning the making available on the Union market and the export from the Union of certain commodities and products associated with deforestation and forest degradation, establishes a mandatory three-step due diligence framework for operators and traders dealing in seven regulated commodities (cattle, cocoa, coffee, oil palm, rubber, soya, wood) and their derived products. The three steps, defined in Article 8, are: (1) information collection (Article 9), (2) risk assessment (Article 10), and (3) risk mitigation (Article 10(2) and as further elaborated through the regulation's enforcement provisions).

Risk mitigation is the third and final step of the EUDR due diligence process. It is the step that determines whether a product may legally be placed on the EU market or exported from the Union. Where risk assessment under Article 10(1) identifies a non-negligible risk that relevant products are non-compliant with Article 3 (deforestation-free, legally produced, covered by a due diligence statement), the operator or trader shall not place those products on the market or export them unless and until the risk has been mitigated to a negligible level through adequate risk mitigation measures.

The Risk Mitigation Advisor Agent (EUDR-025) is the capstone agent of the Risk Assessment sub-category (EUDR-016 through EUDR-025). While EUDR-016 through EUDR-024 assess, quantify, and verify various dimensions of risk (country risk, supplier risk, commodity risk, corruption risk, deforestation risk, indigenous rights risk, protected area risk, legal compliance risk, and third-party audit findings), EUDR-025 synthesizes all risk assessment outputs and prescribes, tracks, and verifies the risk mitigation measures necessary to bring residual risk to a negligible level acceptable to competent authorities.

**Enforcement Timeline (as amended by Regulation (EU) 2024/3234):**
- December 30, 2026: Large and medium operators/traders -- full obligations apply
- June 30, 2027: Micro and small operators/traders -- full obligations apply
- Competent authorities enforcement duties active from June 30, 2026

### 1.2 Article 8 -- Due Diligence System and the Mitigation Obligation

Article 8 of the EUDR defines the due diligence system that operators and non-SME traders must establish, implement, and maintain. The system comprises three interlocking elements:

**Article 8(1):** Prior to placing relevant products on the Union market or exporting them, operators shall exercise due diligence in relation to all relevant products. The due diligence shall include the information collection referred to in Article 9, the risk assessment measures referred to in Article 10, and the risk mitigation measures referred to in Article 10.

**Article 8(2):** Operators shall establish, implement, maintain, regularly evaluate, and update a due diligence system. The system shall include risk management policies and procedures, a designated compliance officer, and adequate human and technical resources.

**Article 8(3):** Operators shall review and update their due diligence system at least once per year and whenever there is a significant change in the nature of the risk or the operator's supply chain.

**Implications for Risk Mitigation:**
- Risk mitigation is not an optional step; it is a mandatory component of the due diligence system that must be exercised before any product is placed on the market.
- The due diligence system must include documented risk management policies and procedures, which necessarily encompass risk mitigation strategy selection, implementation, and verification.
- Annual review requirements mean that mitigation measures must be periodically re-evaluated for effectiveness, not merely implemented once.
- A designated compliance officer must oversee the risk mitigation process, creating accountability for mitigation decisions.

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-001: The agent SHALL support the establishment and maintenance of documented risk mitigation policies and procedures as part of the operator's Article 8(2) due diligence system.
- REQ-025-002: The agent SHALL enforce annual review cycles for all active risk mitigation measures per Article 8(3), triggering re-assessment when due.
- REQ-025-003: The agent SHALL designate and track compliance officer accountability for each risk mitigation decision and remediation plan.
- REQ-025-004: The agent SHALL prevent placement of products on the market (by flagging DDS submissions as blocked) when Article 10 risk assessment identifies non-negligible risk and no adequate mitigation measures are in place.

### 1.3 Article 9 -- Information Requirements That Drive Mitigation Triggers

Article 9 defines the information that operators must collect prior to risk assessment and mitigation. Gaps in Article 9 information are themselves risk factors that may trigger mitigation requirements.

**Article 9(1) Required Information:**

| Element | Article 9(1) Sub-paragraph | Mitigation Trigger |
|---------|----------------------------|-------------------|
| Product description and quantity | (a) | Missing or inconsistent product information triggers enhanced verification |
| Country of production | (b) | High-risk country classification triggers mandatory mitigation |
| Geolocation of all plots of land | (d) | Missing, incomplete, or unverifiable geolocation triggers immediate mitigation |
| Date or time range of production | (e) | Temporal gaps overlapping with EUDR cutoff date (31 Dec 2020) trigger verification |
| Compliance with relevant legislation | (f) | Inability to confirm legality triggers legal compliance mitigation |
| Supplier information | (g) | Unknown or unverified suppliers trigger enhanced due diligence |
| Supply chain information | (h) | Gaps in supply chain traceability trigger chain-of-custody verification |

**Article 9(2):** For products sourced from low-risk countries or parts thereof (Article 29 benchmarking), simplified information requirements apply per Article 13. However, even simplified due diligence requires information sufficient to verify that the product is deforestation-free and legally produced.

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-005: The agent SHALL evaluate Article 9(1) information completeness and identify specific information gaps that constitute mitigation triggers.
- REQ-025-006: The agent SHALL categorize information gaps by severity (critical gap, major gap, minor gap) and recommend appropriate mitigation measures for each gap category.
- REQ-025-007: The agent SHALL track information gap closure as a mitigation measure, verifying that missing information is obtained and validated before the product is cleared for market placement.

### 1.4 Article 10 -- Risk Assessment and the Mitigation Decision Point

Article 10 is the central article governing the transition from risk identification to risk mitigation. It establishes both the criteria for risk assessment and the obligation to mitigate identified risks.

**Article 10(1):** Operators shall assess and identify the risk that relevant products intended to be placed on the market or exported are non-compliant with Article 3. The risk assessment shall be adequate and proportionate to the risk of non-compliance, taking into account the criteria set out in paragraph 2.

**Article 10(2) -- Risk Assessment Criteria (Mitigation Trigger Factors):**

Each criterion in Article 10(2) constitutes a potential trigger for risk mitigation. The agent must evaluate all criteria and determine which trigger mitigation:

| Criterion | Art. 10(2) | Risk Factor | Mitigation Trigger Threshold |
|-----------|-----------|-------------|-------------------------------|
| Assigned risk category of country of production | (a) | Country benchmarked as high or standard risk | High risk: mandatory enhanced mitigation; Standard risk: proportionate mitigation |
| Presence of forests including deforestation or degradation | (b) | Active deforestation in sourcing region | Any post-2020 deforestation detection triggers immediate mitigation |
| Source of commodities (wild, plantation, mixed) | (c) | Commodity production patterns | High-risk commodities in high-risk geographies trigger enhanced mitigation |
| Complexity of the supply chain | (d) | Multi-tier, multi-country supply chains | Supply chains with 4+ tiers or transshipment countries trigger enhanced traceability mitigation |
| Risk of circumvention or mixing | (e) | Possible contamination with unknown-origin products | Mass balance discrepancies or segregation failures trigger chain-of-custody mitigation |
| Prevalence of deforestation in the country | (f) | Historical and current deforestation rates | Countries with deforestation rates above global average trigger enhanced mitigation |
| Concerns about the country of production | (g) | Armed conflicts, sanctions, corruption, weak governance | Elevated corruption or governance concerns trigger enhanced documentation and verification |
| Risk of money laundering or tax evasion | (h) | Financial crime indicators | FATF high-risk jurisdictions trigger financial due diligence mitigation |
| Findings from previous risk assessments | (i) | Historical non-compliance | Previous non-negligible risk findings require escalated mitigation |
| Complementary information | (j) | Third-party reports, NGO alerts, media reports | Substantiated concern submissions trigger investigation and mitigation |
| Concerns raised by relevant stakeholders | (k) | Indigenous peoples, local communities, civil society | Unresolved stakeholder concerns require consultation and mitigation |
| Certification or third-party verified schemes | (l) | Certification status and coverage | Certification gaps relative to EUDR requirements trigger supplementary mitigation |
| Information from third parties | (m) | Intelligence indicating non-compliance risk | Third-party risk intelligence triggers verification mitigation |
| Consultation with stakeholders | (n) | Stakeholder engagement outcomes | Negative consultation outcomes trigger enhanced engagement mitigation |

**Article 10(3) -- The Mitigation Obligation:**

Where the risk assessment under paragraph 1 identifies a non-negligible risk that the relevant products are not compliant with Article 3, the operator or trader shall not place the relevant products on the market or export them unless and until that risk has been adequately mitigated. Risk mitigation measures may include:

- Requiring additional information, data, or documents from suppliers
- Carrying out independent surveys, audits, or other assessments
- Supporting suppliers through capacity building, training, and investment
- Any other appropriate risk mitigation measure

**Article 10(4):** The decisions on risk mitigation procedures and measures shall be documented, reviewed at least on an annual basis, and made available by the operators to the competent authorities upon request. Operators shall be able to demonstrate how decisions on risk mitigation procedures and measures were taken.

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-008: The agent SHALL evaluate all 14 Article 10(2) risk assessment criteria and determine which criteria trigger risk mitigation for each product-supply chain combination.
- REQ-025-009: The agent SHALL implement a non-negligible risk determination engine that synthesizes inputs from EUDR-016 through EUDR-024 into a composite risk assessment that identifies whether Article 10(3) mitigation is required.
- REQ-025-010: The agent SHALL recommend specific risk mitigation measures appropriate to each identified risk factor, drawing from the Article 10(3) enumerated measures and ISO 31000:2018 risk treatment options.
- REQ-025-011: The agent SHALL document all risk mitigation decisions per Article 10(4), including the rationale for measure selection, expected effectiveness, implementation timeline, and verification method.
- REQ-025-012: The agent SHALL enforce the prohibition on market placement when non-negligible risk has been identified and adequate mitigation has not been demonstrated, blocking DDS submission until mitigation is verified.
- REQ-025-013: The agent SHALL conduct annual review of all active mitigation measures per Article 10(4) and generate re-assessment reports for compliance officer review.

### 1.5 Article 11 -- Simplified Due Diligence and Reduced Mitigation

Article 11, read in conjunction with Article 13, provides for simplified due diligence obligations where the European Commission has classified a country or part thereof as "low risk" under Article 29. Simplified due diligence affects the mitigation obligation in the following ways:

**Article 11(1) / Article 13:** Products sourced exclusively from countries or parts of countries classified as low risk benefit from simplified due diligence. The operator is not required to conduct the full Article 10 risk assessment or implement Article 10(3) risk mitigation measures, provided:
- All information required under Article 9 (as adapted for simplified due diligence) has been collected
- There is no substantive reason to believe the products are non-compliant
- No new information has come to the operator's attention that would indicate non-compliance risk

**Implications for Risk Mitigation Reduction:**
- For 100% low-risk sourced products: risk mitigation measures under Article 10(3) are not required, but the operator must still collect Article 9 information and submit a DDS
- Simplified due diligence applies only where all components of the product originate from low-risk areas; any mixture with standard or high-risk origins disqualifies simplified treatment
- If an operator receives "new information" suggesting non-compliance risk for a low-risk sourced product, full risk assessment and mitigation must be applied
- The Commission periodically reviews country classifications; a reclassification from low to standard or high risk triggers full risk assessment and mitigation for affected supply chains

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-014: The agent SHALL evaluate eligibility for simplified due diligence based on the Article 29 country benchmarking classification of all origins in the product's supply chain.
- REQ-025-015: The agent SHALL downgrade mitigation requirements for products sourced exclusively from low-risk countries, while maintaining Article 9 information collection and DDS submission obligations.
- REQ-025-016: The agent SHALL monitor for "new information" triggers that revoke simplified due diligence eligibility and escalate to full risk assessment and mitigation.
- REQ-025-017: The agent SHALL automatically escalate mitigation requirements when the Commission reclassifies a sourcing country from low risk to standard or high risk.

### 1.6 Article 12 -- Due Diligence Statement and Mitigation Documentation

Article 12 requires operators to submit a due diligence statement (DDS) to the EU Information System before placing relevant products on the market. The DDS must confirm that due diligence has been exercised and that the risk of non-compliance is negligible or has been adequately mitigated.

**Article 12(2):** The DDS shall contain a declaration by the operator that due diligence was exercised in accordance with Articles 8 and 10, and that no risk or only a negligible risk was found, or that the risk identified has been mitigated.

**Implications for Risk Mitigation:**
- The DDS is the formal attestation that risk mitigation has been completed
- The operator must be able to defend the DDS content upon competent authority request, demonstrating the mitigation measures taken
- A false or misleading DDS constitutes a separate compliance offense under the penalty framework

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-018: The agent SHALL generate the risk mitigation section of the DDS, documenting the risk factors identified, mitigation measures applied, and residual risk assessment.
- REQ-025-019: The agent SHALL validate that all identified non-negligible risks have been mitigated before allowing the DDS to be marked as ready for submission.
- REQ-025-020: The agent SHALL maintain a complete audit trail linking each DDS to the underlying risk assessment, mitigation measures, and verification evidence.

### 1.7 Articles 18-21 -- Remedial Action and Corrective Measures

Articles 18 through 21 establish the framework for competent authority enforcement actions and operator remediation obligations when non-compliance is identified.

**Article 18 -- Requests for Remedial Action:**
Where competent authorities establish that an operator or trader has not complied with the regulation, or that a relevant product is non-compliant, they shall without delay require the operator or trader to take appropriate and proportionate corrective action to bring the non-compliance to an end within a specified and reasonable period.

Corrective actions may include:
- Bringing the due diligence system into compliance
- Completing missing risk assessments or mitigation measures
- Providing additional documentation or evidence
- Implementing enhanced monitoring or verification

**Article 19 -- Interim Measures:**
Where the competent authority identifies a serious risk of non-compliance, it may take interim measures including:
- Suspension of the placing on the market or export of relevant products
- Seizure of relevant products
- Suspension of the validity of DDS

**Article 20 -- Definitive Measures:**
Where the competent authority establishes non-compliance, it may order:
- Withdrawal or recall of non-compliant products from the market
- Prohibition on placing or making available the non-compliant products
- Donation of confiscated products to charitable or public interest purposes, or disposal

**Article 21 -- Obligation to Inform:**
Operators who become aware that a relevant product which they have placed on the market or exported is not compliant shall immediately inform the competent authorities and take the necessary corrective measures.

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-021: The agent SHALL manage remediation plans issued in response to competent authority requests for corrective action under Article 18, tracking required actions, deadlines, and completion status.
- REQ-025-022: The agent SHALL track interim measures (Article 19) including product suspensions, seizures, and DDS validity suspensions, linking them to required remediation activities.
- REQ-025-023: The agent SHALL manage definitive measure responses (Article 20) including product withdrawal, recall, and disposal tracking with evidence documentation.
- REQ-025-024: The agent SHALL implement a self-disclosure workflow per Article 21, enabling operators to report identified non-compliance and document corrective measures taken.

### 1.8 Articles 22-23 -- Penalty Framework and Mitigation as Defense

Articles 22 and 23 establish the penalty framework that creates the economic incentive for robust risk mitigation.

**Article 22(1):** Member States shall lay down rules on penalties applicable to infringements and shall take all measures necessary to ensure that they are implemented. The penalties provided for shall be effective, proportionate, and dissuasive.

**Article 23 -- Specific Penalties:**

| Penalty | Description | Mitigation Relevance |
|---------|-------------|---------------------|
| Fines | Proportionate to environmental damage, product value, and losses; maximum not less than 4% of annual EU-wide turnover | Documented mitigation efforts may be considered as mitigating factors |
| Confiscation | Confiscation of non-compliant products and of revenue gained from transactions involving the products | Effective mitigation prevents non-compliant products from reaching the market |
| Market Exclusion | Temporary prohibition from placing relevant products on the market | Robust mitigation system reduces risk of market exclusion |
| Public Procurement Exclusion | Temporary exclusion from participation in public procurement procedures | Demonstrated mitigation program supports reinstatement |
| Public Naming | Public disclosure of the identity of the operator/trader responsible | Proactive mitigation reduces reputational penalty risk |

**Mitigation as a Defense Factor:**
While the EUDR does not create an explicit safe harbor for operators who implement risk mitigation, the requirement that penalties be "proportionate" implies that the quality and comprehensiveness of an operator's risk mitigation system will be relevant to penalty determination. Competent authorities exercising their discretion in penalty assessment can consider:
- Whether the operator had a documented, systematic risk mitigation process
- Whether identified risks were addressed with proportionate measures
- Whether mitigation measures were verified for effectiveness
- Whether the operator self-disclosed non-compliance upon discovery (Article 21)
- Whether the operator cooperated with competent authority investigations

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-025: The agent SHALL calculate penalty exposure scores based on unmitigated risk factors, using the Article 23 penalty framework with estimated financial impact.
- REQ-025-026: The agent SHALL generate compliance defense packages demonstrating the operator's systematic risk mitigation efforts, suitable for submission to competent authorities as mitigating evidence in penalty proceedings.

### 1.9 Article 29 -- Country Benchmarking and Risk-Differentiated Mitigation

Article 29 mandates the European Commission to classify countries and parts of countries into three risk categories: low, standard, and high. This classification directly determines the intensity and scope of required risk mitigation.

**Risk-Differentiated Mitigation Requirements:**

| Country Risk | Mitigation Intensity | Minimum Measures | CA Check Rate |
|-------------|---------------------|-----------------|---------------|
| Low | Simplified (Article 13) | Information collection; DDS submission; monitoring for new information | 1% of operators; 1% of quantity |
| Standard | Standard (Article 10) | Full risk assessment; proportionate mitigation for identified risks; annual review | 3% of operators; 3% of quantity |
| High | Enhanced (Article 10 + supplementary) | Comprehensive risk assessment; enhanced mitigation including independent verification, supplier audits, satellite monitoring; quarterly review | 9% of operators; 9% of quantity |

**Implications for Mitigation Strategy:**
- High-risk country sourcing requires the most intensive mitigation, typically including independent third-party audits, satellite monitoring verification, enhanced documentation, and supplier capacity building
- Standard-risk country sourcing requires proportionate mitigation based on specific risk factors identified in the Article 10(2) assessment
- The 9% inspection rate for high-risk country products means that operators sourcing from high-risk countries must maintain continuous inspection readiness with comprehensive mitigation documentation

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-027: The agent SHALL apply risk-differentiated mitigation intensity levels (simplified, standard, enhanced) based on the Article 29 country benchmarking classification.
- REQ-025-028: The agent SHALL escalate mitigation intensity automatically when country benchmarking classifications change (e.g., standard to high).
- REQ-025-029: The agent SHALL maintain enhanced mitigation documentation for high-risk country sourcing, supporting the 9% competent authority inspection rate.

---

## 2. ISO 31000:2018 Risk Management Framework

### 2.1 ISO 31000:2018 -- Risk Management Guidelines

ISO 31000:2018 provides the internationally recognized framework for risk management, including risk treatment (mitigation) strategies. While the EUDR does not explicitly reference ISO 31000, the standard provides the methodological foundation for systematic risk mitigation that aligns with the EUDR's due diligence requirements and is recognized by competent authorities and auditors as best practice.

**Risk Management Principles (Clause 4):**

ISO 31000:2018 establishes eight principles that an effective risk management framework should exhibit. The Risk Mitigation Advisor Agent must implement these principles:

| # | Principle | Description | Agent Implementation |
|---|-----------|-------------|---------------------|
| 1 | Integrated | Risk management is an integral part of all organizational activities | Mitigation embedded in DDS workflow; not a standalone process |
| 2 | Structured and Comprehensive | A structured and comprehensive approach contributes to consistent and comparable results | Systematic mitigation strategy selection using defined methodology |
| 3 | Customized | The framework and process are customized and proportionate | Mitigation proportionate to country risk, commodity risk, and supply chain complexity |
| 4 | Inclusive | Appropriate and timely involvement of stakeholders | Supplier engagement, community consultation, stakeholder feedback integration |
| 5 | Dynamic | Risks can emerge, change, or disappear as the organization's context changes | Continuous monitoring with event-driven mitigation adjustments |
| 6 | Best Available Information | Based on historical and current information, as well as future expectations | Integration of EUDR-016 through EUDR-024 risk intelligence |
| 7 | Human and Cultural Factors | Human behavior and culture significantly influence all aspects of risk management | Supplier capacity building; cultural context in mitigation design |
| 8 | Continual Improvement | Risk management is continually improved through learning and experience | Mitigation effectiveness tracking; lessons learned; adaptive management |

### 2.2 ISO 31000:2018 Clause 6.5 -- Risk Treatment

Clause 6.5 of ISO 31000:2018 defines the risk treatment process, which is the systematic selection and implementation of options for addressing risk. This clause provides the methodological foundation for the agent's mitigation strategy engine.

**Clause 6.5.1 -- General:**
Risk treatment involves selecting one or more options for modifying risks and implementing those options. Once implemented, treatments provide or modify controls. Risk treatment involves an iterative process of:
- Formulating and selecting risk treatment options
- Planning and implementing risk treatment
- Assessing the effectiveness of that treatment
- Deciding whether the remaining (residual) risk is acceptable
- If not acceptable, taking further treatment

**Clause 6.5.2 -- Selection of Risk Treatment Options:**
Options for treating risk may involve one or more of the following:
- Avoiding the risk by deciding not to start or continue with the activity that gives rise to the risk
- Taking or increasing risk in order to pursue an opportunity
- Removing the risk source
- Changing the likelihood
- Changing the consequences
- Sharing the risk (e.g., through contracts, buying insurance)
- Retaining the risk by informed decision

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-030: The agent SHALL implement the ISO 31000:2018 risk treatment process including formulation, selection, implementation, effectiveness assessment, and residual risk evaluation.
- REQ-025-031: The agent SHALL support all ISO 31000:2018 risk treatment option categories (avoid, reduce likelihood, reduce consequences, share/transfer, retain) mapped to EUDR-specific mitigation measures.

### 2.3 Risk Mitigation Hierarchy

The Risk Mitigation Advisor Agent implements a four-tier risk mitigation hierarchy, derived from ISO 31000:2018 risk treatment options and adapted to the EUDR regulatory context:

**Tier 1 -- Avoid (Highest Priority):**

| Strategy | EUDR Application | When Applied |
|----------|-----------------|-------------|
| Supplier Substitution | Replace non-compliant supplier with verified compliant alternative | Supplier risk score exceeds critical threshold; irremediable non-compliance |
| Sourcing Origin Change | Shift sourcing from high-risk to low/standard-risk countries or regions | Country risk makes mitigation impractical or cost-prohibitive |
| Product Withdrawal | Remove non-compliant product from market placement pipeline | Post-cutoff deforestation confirmed; legal compliance impossible |
| Activity Cessation | Discontinue trade in specific commodity-origin combination | Systemic non-compliance in specific supply chain segment |

**Tier 2 -- Reduce (Primary Mitigation):**

| Strategy | EUDR Application | When Applied |
|----------|-----------------|-------------|
| Enhanced Verification | Independent surveys, audits, satellite monitoring verification | Standard or high-risk country sourcing with specific identified risks |
| Documentation Enhancement | Obtain additional permits, certificates, geolocation data | Information gaps identified in Article 9 data collection |
| Supply Chain Simplification | Reduce supply chain tiers, eliminate unnecessary intermediaries | Supply chain complexity identified as risk factor under Article 10(2)(d) |
| Segregation Improvement | Enhance chain-of-custody controls to prevent mixing | Circumvention or mixing risk identified under Article 10(2)(e) |
| Supplier Capacity Building | Training, technical assistance, investment in supplier compliance | Supplier lacks capability but shows willingness to comply |
| Monitoring Enhancement | Increase satellite monitoring frequency, expand buffer zones | Deforestation alerts in proximity to supply chain plots |
| Legal Compliance Support | Assist suppliers in obtaining required permits and certifications | Legal compliance gaps identified by EUDR-023 |

**Tier 3 -- Transfer (Risk Sharing):**

| Strategy | EUDR Application | When Applied |
|----------|-----------------|-------------|
| Contractual Obligations | Supplier contracts with compliance warranties and indemnification | Standard risk sourcing with contractual risk allocation |
| Insurance Coverage | EUDR compliance insurance for financial penalty risk | Supplementary financial risk management |
| Certification Reliance | Leverage third-party certification as partial risk mitigation | Active, valid certification covering EUDR-relevant scope |
| Third-Party Verification | Commission independent verification by accredited bodies | Enhanced due diligence for high-risk supply chains |

**Tier 4 -- Accept (Informed Retention):**

| Strategy | EUDR Application | When Applied |
|----------|-----------------|-------------|
| Negligible Risk Acceptance | Accept residual risk after mitigation when risk is negligible | All material risks have been mitigated; residual risk below threshold |
| Monitored Acceptance | Accept residual risk with enhanced monitoring and contingency plan | Low residual risk with monitoring controls in place |

**Critical Constraint:** Under the EUDR, risk acceptance (Tier 4) is only permissible when the risk has been reduced to "negligible" (or was negligible from the outset). The EUDR does not permit operators to accept non-negligible risk and proceed with market placement. This distinguishes EUDR risk mitigation from general ISO 31000 risk management, where organizations may accept higher levels of residual risk based on risk appetite.

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-032: The agent SHALL implement the four-tier risk mitigation hierarchy (Avoid, Reduce, Transfer, Accept) with EUDR-specific strategy options at each tier.
- REQ-025-033: The agent SHALL recommend mitigation strategies in hierarchical order, preferring higher-tier strategies (Avoid) for higher-severity risks and lower-tier strategies (Accept) only for negligible residual risks.
- REQ-025-034: The agent SHALL enforce the EUDR constraint that risk acceptance (Tier 4) is only permissible when residual risk is negligible, preventing acceptance of non-negligible risk.

### 2.4 ISO 31000:2018 Clause 6.5.3 -- Preparing and Implementing Risk Treatment Plans

Clause 6.5.3 specifies that risk treatment plans must document:

| Plan Element | ISO 31000 Reference | EUDR Application |
|-------------|---------------------|------------------|
| Rationale for selection | The expected benefits to be gained | Why this mitigation measure addresses the identified EUDR risk |
| Accountability | Those accountable and responsible for approving and implementing | Compliance officer, supplier contact, audit manager |
| Proposed actions | The actions to be taken | Specific mitigation steps (audit, document collection, supplier training) |
| Resources required | Including contingencies | Budget, personnel, technical tools, timeline buffer |
| Performance measures | How effectiveness will be evaluated | KPIs for risk reduction, verification criteria, residual risk target |
| Constraints | Limitations on implementation | Supplier cooperation, geographic access, regulatory timelines |
| Reporting and monitoring | Required reporting and monitoring activities | Progress reporting frequency, escalation triggers, status dashboards |
| Timeline | When actions are expected to be undertaken and completed | Milestones, deadlines, DDS submission target date |

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-035: The agent SHALL generate risk mitigation plans containing all eight ISO 31000:2018 Clause 6.5.3 plan elements.
- REQ-025-036: The agent SHALL integrate mitigation plans into the organization's management processes, linking them to DDS submission workflows, supplier management, and audit programs.
- REQ-025-037: The agent SHALL track mitigation plan implementation status against milestones and deadlines, with automated escalation for overdue actions.

---

## 3. Risk Mitigation Triggers and Thresholds

### 3.1 Composite Risk Score and Mitigation Threshold

The Risk Mitigation Advisor Agent consumes risk assessment outputs from nine upstream EUDR risk agents (EUDR-016 through EUDR-024) and synthesizes them into a composite risk score that determines whether mitigation is required and at what intensity.

**Risk Agent Input Matrix:**

| Agent | Agent ID | Risk Dimension | Score Range | Weight |
|-------|----------|---------------|-------------|--------|
| Country Risk Evaluator | EUDR-016 | Country-level deforestation and governance risk | 0-100 | 20% |
| Supplier Risk Scorer | EUDR-017 | Supplier-level compliance and performance risk | 0-100 | 15% |
| Commodity Risk Analyzer | EUDR-018 | Commodity-specific deforestation correlation risk | 0-100 | 10% |
| Corruption Index Monitor | EUDR-019 | Governance integrity and corruption risk | 0-100 | 10% |
| Deforestation Alert System | EUDR-020 | Active deforestation detection near supply plots | 0-100 | 15% |
| Indigenous Rights Checker | EUDR-021 | Indigenous peoples' rights compliance risk | 0-100 | 8% |
| Protected Area Validator | EUDR-022 | Protected area encroachment risk | 0-100 | 7% |
| Legal Compliance Verifier | EUDR-023 | Legal framework compliance risk (8 categories) | 0-100 | 10% |
| Third-Party Audit Manager | EUDR-024 | Audit findings and non-conformance status | 0-100 | 5% |

**Composite Risk Score Calculation:**

The composite risk score is a weighted average of all nine agent scores, subject to critical override rules:

```
Composite Score = SUM(Agent_Score_i * Weight_i) for i = 1 to 9
```

**Critical Override Rules:**
- If any single agent score >= 90 (Critical), the composite score is elevated to at least 80 regardless of weighted average
- If EUDR-020 (Deforestation Alert) detects confirmed post-cutoff deforestation, the composite score is set to 100 (maximum)
- If EUDR-023 (Legal Compliance) identifies a fundamental legality failure, the composite score is set to at least 90
- If EUDR-021 (Indigenous Rights) identifies confirmed FPIC violation, the composite score is elevated to at least 85

### 3.2 Mitigation Threshold Classification

| Composite Score | Risk Level | Mitigation Requirement | Mitigation Intensity |
|-----------------|-----------|------------------------|---------------------|
| 0-10 | Negligible | No mitigation required; DDS may proceed | None (simplified due diligence eligible) |
| 11-25 | Low | Monitoring-level mitigation; document and monitor | Light (enhanced monitoring, periodic review) |
| 26-50 | Medium | Proportionate mitigation required before DDS | Standard (verification, documentation, supplier engagement) |
| 51-75 | High | Enhanced mitigation required before DDS | Enhanced (independent audit, satellite verification, capacity building) |
| 76-90 | Critical | Comprehensive mitigation package required | Intensive (multiple concurrent measures, third-party verification, continuous monitoring) |
| 91-100 | Prohibitive | Risk avoidance recommended; market placement blocked pending resolution | Maximum (supplier substitution or origin change recommended; product cannot be placed on market) |

### 3.3 Individual Risk Factor Triggers

Beyond the composite score, specific risk factors may independently trigger mitigation requirements:

| Trigger Category | Specific Trigger | Mitigation Action | Priority |
|-----------------|-----------------|-------------------|----------|
| Deforestation Detection | Post-cutoff deforestation confirmed on supply plot | Immediate product block; supplier investigation; origin verification | P0 -- Immediate |
| Deforestation Detection | Deforestation alert within 10km buffer zone | Enhanced monitoring; supplier verification; satellite review | P1 -- Urgent |
| Legal Non-Compliance | Fundamental legality failure (missing key permits) | Product block; legal compliance remediation plan | P0 -- Immediate |
| Legal Non-Compliance | Partial legality gaps (minor permit issues) | Documentation enhancement; legal support for supplier | P2 -- Standard |
| Indigenous Rights | Confirmed FPIC violation on supply plot | Product block; community consultation and remediation | P0 -- Immediate |
| Indigenous Rights | Overlap with indigenous territory (no verified FPIC) | FPIC verification; community engagement | P1 -- Urgent |
| Protected Areas | Production within protected area boundary | Product block; origin verification; alternative sourcing | P0 -- Immediate |
| Protected Areas | Production within protected area buffer zone | Enhanced monitoring; boundary verification | P2 -- Standard |
| Country Risk | Country reclassified to high risk | Enhanced mitigation for all supply chains from that country | P1 -- Urgent |
| Supplier Risk | Supplier risk score exceeds critical threshold (>= 80) | Supplier engagement; capacity building or substitution | P1 -- Urgent |
| Audit Findings | Critical non-conformance from third-party audit | Corrective action; enhanced verification | P0 -- Immediate |
| Audit Findings | Major non-conformance pattern (3+ recurring) | Root cause analysis; systemic remediation | P1 -- Urgent |
| Substantiated Concerns | Article 14 concern submitted to competent authority | Investigation; evidence preparation; mitigation review | P1 -- Urgent |
| Information Gaps | Critical Article 9 information missing | Information collection mitigation; product block if unresolvable | P0 -- Immediate |

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-038: The agent SHALL calculate composite risk scores using the weighted multi-agent input matrix with critical override rules.
- REQ-025-039: The agent SHALL classify risk into six levels (Negligible, Low, Medium, High, Critical, Prohibitive) with corresponding mitigation intensity requirements.
- REQ-025-040: The agent SHALL evaluate individual risk factor triggers independently of the composite score and apply immediate mitigation actions for P0 triggers.
- REQ-025-041: The agent SHALL recalculate composite risk scores and re-evaluate mitigation triggers whenever any upstream agent publishes a risk score update.

---

## 4. Mitigation Strategy Selection Methodology

### 4.1 Strategy Selection Framework

The Risk Mitigation Advisor Agent employs a systematic methodology for selecting appropriate mitigation strategies for each identified risk factor. The methodology follows ISO 31000:2018 Clause 6.5.2 guidance on selection criteria.

**Selection Criteria:**

| Criterion | Description | Weight |
|-----------|-------------|--------|
| Effectiveness | Degree to which the measure reduces the identified risk to a negligible level | 30% |
| Feasibility | Practical implementability considering supplier cooperation, geographic access, technical capability | 20% |
| Time-to-Effect | Time required for the measure to produce measurable risk reduction | 15% |
| Cost-Proportionality | Cost of the measure relative to the value at risk and the severity of the risk | 10% |
| Regulatory Acceptability | Likelihood that competent authorities will accept the measure as adequate mitigation | 15% |
| Sustainability | Long-term durability of the risk reduction achieved by the measure | 10% |

### 4.2 Strategy Selection Decision Matrix

For each identified risk factor, the agent evaluates applicable mitigation strategies against the selection criteria and recommends the optimal strategy or combination of strategies.

**Decision Rules:**

1. **Single Critical Risk Factor (Score >= 90):** Apply the highest-tier available strategy (Avoid if feasible; otherwise multi-layered Reduce + Transfer).
2. **Multiple Moderate Risk Factors (3+ factors scoring 40-70):** Apply a portfolio of Reduce strategies addressing each factor, with Transfer strategies for residual risk.
3. **Single Moderate Risk Factor (Score 40-70):** Apply the most effective single Reduce strategy with verification.
4. **Low-Level Risk Factors (Score 11-39):** Apply proportionate monitoring-level measures (Monitored Acceptance or Light Reduce).
5. **Negligible Risk (Score 0-10):** No active mitigation required; Accept with standard monitoring.

### 4.3 Strategy Combination Rules

Multiple mitigation strategies may be combined for comprehensive risk reduction:

**Additive Strategies (can be layered):**
- Enhanced documentation + independent audit
- Supplier capacity building + satellite monitoring + periodic verification
- Contractual obligations + third-party certification + enhanced monitoring

**Mutually Exclusive Strategies (choose one):**
- Supplier substitution OR supplier capacity building (cannot do both for same supplier)
- Product withdrawal OR enhanced verification for same product batch
- Origin change OR enhanced in-origin monitoring for same supply chain

**Sequenced Strategies (apply in order):**
1. Immediate containment (product hold, enhanced monitoring)
2. Root cause analysis (investigation, evidence collection)
3. Corrective action (supplier remediation, documentation, verification)
4. Verification (independent audit, effectiveness review)
5. Ongoing monitoring (continuous controls, periodic review)

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-042: The agent SHALL apply the six-criterion strategy selection framework to evaluate and rank applicable mitigation strategies for each risk factor.
- REQ-025-043: The agent SHALL recommend strategy combinations using additive, mutually exclusive, and sequenced strategy rules.
- REQ-025-044: The agent SHALL generate a scored recommendation report for each mitigation strategy, including effectiveness estimate, implementation cost, timeline, and regulatory acceptability assessment.

---

## 5. Remediation Plan Development

### 5.1 Remediation Plan Structure

When risk mitigation requires active intervention (Tier 2 -- Reduce strategies), the agent generates structured remediation plans following ISO 31000:2018 Clause 6.5.3 requirements.

**Remediation Plan Template:**

| Section | Content | Mandatory |
|---------|---------|-----------|
| 1. Plan Identification | Plan ID, creation date, version, owner, product reference, supply chain reference | Yes |
| 2. Risk Summary | Identified risk factors with scores, triggering Article 10(2) criteria, composite risk score | Yes |
| 3. Root Cause Analysis | Underlying causes of identified risks using structured methodology (5-Why, Fishbone, Fault Tree) | Yes (for High/Critical risks) |
| 4. Mitigation Objectives | Specific, measurable risk reduction targets with target residual risk score | Yes |
| 5. Selected Measures | Detailed description of each mitigation measure with rationale for selection | Yes |
| 6. Implementation Schedule | Milestones, deadlines, dependencies, critical path analysis | Yes |
| 7. Resource Allocation | Budget, personnel, technical tools, external services required | Yes |
| 8. Responsibility Matrix | Named individuals accountable for each action item with RACI assignments | Yes |
| 9. Supplier Engagement Plan | Actions required of suppliers, capacity building activities, contractual amendments | Yes (if supplier-facing measures) |
| 10. Verification Criteria | How each measure's effectiveness will be verified (evidence type, acceptance threshold) | Yes |
| 11. Monitoring Protocol | Ongoing monitoring activities during and after implementation | Yes |
| 12. Escalation Procedures | Triggers and procedures for escalating to higher-tier mitigation if current measures are insufficient | Yes |
| 13. Timeline Constraints | DDS submission deadline, competent authority response deadlines, certification renewal dates | Yes |
| 14. Residual Risk Assessment | Expected residual risk score after mitigation; gap analysis if target is not met | Yes |
| 15. Approval Record | Compliance officer approval, management sign-off, approval date | Yes |

### 5.2 Remediation Plan Categories

Different risk scenarios require different remediation plan types:

| Plan Type | Trigger | Scope | Typical Duration | Review Cycle |
|-----------|---------|-------|------------------|-------------|
| Standard Remediation | Medium risk (26-50) | Single risk factor or supplier | 30-90 days | Monthly |
| Enhanced Remediation | High risk (51-75) | Multiple risk factors or multi-supplier | 60-180 days | Bi-weekly |
| Critical Remediation | Critical risk (76-90) | Comprehensive supply chain segment | 90-365 days | Weekly |
| Emergency Remediation | Prohibitive risk (91-100) | Immediate containment + long-term restructuring | Containment: 72 hours; Full: 180-365 days | Daily (containment); Weekly (ongoing) |
| Competent Authority Response | Article 18 request | As specified by competent authority | As specified | As specified |
| Self-Disclosure Response | Article 21 self-report | Comprehensive corrective action | 30-180 days | Weekly |

### 5.3 Root Cause Analysis Requirements

For High and Critical risk remediation plans, root cause analysis is mandatory. The agent supports three structured methodologies:

**5-Why Analysis:**
- Iterative questioning (minimum 5 levels) to identify the fundamental cause
- Each "why" must be supported by evidence or reasonable inference
- The root cause must be actionable (an organizational, process, or system failure, not an external condition)

**Ishikawa (Fishbone) Diagram:**
- Categories: People, Process, Technology, Documentation, Suppliers, Environment, Regulation
- Each category is analyzed for potential contributing factors
- Contributing factors are ranked by significance and addressability

**Fault Tree Analysis:**
- Top-level event: Non-compliance with EUDR Article 3
- Intermediate events: Specific risk factors (deforestation, legality, traceability)
- Basic events: Root causes that can be directly addressed through mitigation

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-045: The agent SHALL generate structured remediation plans containing all 15 plan sections for Medium, High, and Critical risk levels.
- REQ-025-046: The agent SHALL categorize remediation plans by type (Standard, Enhanced, Critical, Emergency, CA Response, Self-Disclosure) with appropriate scope and review cycles.
- REQ-025-047: The agent SHALL require root cause analysis using at least one structured methodology (5-Why, Fishbone, Fault Tree) for all High and Critical risk remediation plans.
- REQ-025-048: The agent SHALL track remediation plan implementation progress against milestones with automated status reporting and escalation for overdue actions.

---

## 6. Supplier Capacity Building Programs

### 6.1 EUDR Basis for Supplier Capacity Building

Article 10(3) explicitly includes "supporting suppliers through capacity building, training, and investment" as a risk mitigation measure. This reflects the EUDR's recognition that many non-compliance risks originate from supplier-side capability gaps rather than willful non-compliance, particularly for smallholder producers in developing countries.

**Recital 34:** The regulation acknowledges the importance of supporting smallholder farmers in complying with EUDR requirements, recognizing that they may lack the technical and financial resources to implement deforestation-free and legally compliant production practices independently.

**OECD Due Diligence Guidance (referenced in Recital 44):** The OECD Due Diligence Guidance for Responsible Supply Chains of Minerals from Conflict-Affected and High-Risk Areas, and the OECD-FAO Guidance for Responsible Agricultural Supply Chains, both emphasize that upstream due diligence should include supplier development and capacity building as alternatives to supply chain disengagement.

### 6.2 Capacity Building Program Structure

The agent supports structured supplier capacity building programs with the following components:

**Program Levels:**

| Level | Target | Content | Duration | Investment |
|-------|--------|---------|----------|-----------|
| Level 1: Awareness | All suppliers | EUDR requirements overview; compliance timeline; basic obligations | 1-2 sessions (4-8 hours) | Low (training materials, translation) |
| Level 2: Technical Training | Medium-risk suppliers | Geolocation data collection; documentation requirements; chain-of-custody procedures | 2-4 sessions (16-32 hours) | Medium (trainer, equipment, materials) |
| Level 3: Implementation Support | High-risk suppliers | On-site technical assistance; system implementation; process re-engineering | 3-12 months of periodic engagement | High (dedicated support, technology, infrastructure) |
| Level 4: Financial Support | Smallholder groups | Compliance investment co-funding; group certification support; cooperative formation | 12-36 months | High (co-investment, credit, insurance) |

### 6.3 Capacity Building Content Areas

| Content Area | EUDR Relevance | Delivery Method |
|-------------|---------------|-----------------|
| GPS/Geolocation Data Collection | Article 9(1)(d) geolocation requirement | Mobile device training, GPS tool provision |
| Record-Keeping and Documentation | Article 9(1) information requirements | Template provision, digital tool training |
| Chain-of-Custody Management | Article 10(2)(e) circumvention prevention | Process training, system setup assistance |
| Sustainable Agricultural Practices | Deforestation-free production | Agronomic training, agroforestry guidance |
| Legal Compliance Requirements | Article 2(40) relevant legislation | Legal awareness, permit application support |
| Environmental Protection | Forest conservation, biodiversity | Best management practices, set-aside guidance |
| Indigenous Rights and FPIC | Article 10(2)(d) indigenous rights | FPIC awareness, consultation protocol training |
| Grievance Mechanisms | Stakeholder engagement | Complaint handling system setup |
| Digital Literacy | Platform usage for compliance data entry | System training, connectivity support |

### 6.4 Capacity Building Effectiveness Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Supplier compliance rate improvement | >= 25% improvement within 12 months | Pre/post risk score comparison |
| Geolocation data completeness | >= 95% of plots geolocated after Level 2 training | Article 9(1)(d) data completeness check |
| Documentation compliance | >= 90% of required documents provided within 30 days | Document completeness tracking |
| Non-conformance reduction | >= 50% reduction in supplier-level NCRs within 18 months | NCR trend analysis |
| Training completion rate | >= 90% of targeted suppliers complete assigned program level | Attendance and assessment tracking |
| Supplier retention rate | >= 80% of capacity-built suppliers retained in supply chain | Supplier continuity tracking |

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-049: The agent SHALL support structured supplier capacity building programs with four levels (Awareness, Technical Training, Implementation Support, Financial Support).
- REQ-025-050: The agent SHALL track capacity building program enrollment, progress, completion, and effectiveness for each supplier.
- REQ-025-051: The agent SHALL measure capacity building effectiveness through pre/post risk score comparison and NCR trend analysis.
- REQ-025-052: The agent SHALL recommend appropriate capacity building program levels based on supplier risk scores and identified capability gaps.

---

## 7. Mitigation Measure Effectiveness Tracking

### 7.1 Effectiveness Assessment Framework

Each implemented mitigation measure must be assessed for effectiveness to determine whether the identified risk has been reduced to a negligible level. The agent implements a systematic effectiveness assessment framework aligned with ISO 31000:2018 Clause 6.5 (monitoring and review of risk treatment).

**Effectiveness Assessment Dimensions:**

| Dimension | Description | Measurement |
|-----------|-------------|-------------|
| Risk Reduction | Quantifiable reduction in the identified risk score | Pre-mitigation score vs. post-mitigation score |
| Residual Risk | Remaining risk after mitigation | Composite score recalculation with mitigation applied |
| Timeliness | Whether mitigation was implemented within the planned timeline | Actual completion date vs. planned deadline |
| Sustainability | Whether risk reduction is maintained over time | Periodic reassessment at 30/60/90/180/365 day intervals |
| Completeness | Whether all planned mitigation actions were fully implemented | Implementation checklist completion rate |
| Verification | Whether independent verification confirms effectiveness | Third-party audit, satellite verification, document review |

### 7.2 Effectiveness Scoring Model

| Score | Rating | Description | Action Required |
|-------|--------|-------------|-----------------|
| 90-100 | Highly Effective | Risk reduced to negligible; mitigation fully implemented and verified | Close mitigation; transition to monitoring |
| 70-89 | Effective | Risk significantly reduced; minor residual risk remains within acceptable bounds | Monitor; schedule periodic review |
| 50-69 | Partially Effective | Risk reduced but residual risk remains non-negligible | Implement supplementary measures; investigate gaps |
| 30-49 | Marginally Effective | Risk reduction insufficient; mitigation addressing symptoms not root cause | Redesign mitigation approach; conduct root cause re-analysis |
| 0-29 | Ineffective | No meaningful risk reduction; mitigation failed or was not implemented | Escalate to higher-tier mitigation; consider risk avoidance |

### 7.3 Continuous Effectiveness Monitoring

The agent implements ongoing monitoring to detect mitigation degradation:

**Monitoring Schedule:**

| Risk Level | Initial Review | Ongoing Review | Trigger-Based Review |
|-----------|---------------|----------------|---------------------|
| Low | 90 days after implementation | Annually | Country risk change; new information |
| Medium | 60 days after implementation | Semi-annually | Supplier risk change; audit finding |
| High | 30 days after implementation | Quarterly | Any related risk score change |
| Critical | 14 days after implementation | Monthly | Any risk event |

**Degradation Indicators:**
- Risk score increase after initial reduction
- Supplier non-conformance recurrence
- New deforestation alerts in previously cleared areas
- Certification status change (suspension/withdrawal)
- Competent authority inquiry or inspection
- Stakeholder complaint or substantiated concern

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-053: The agent SHALL assess mitigation effectiveness across six dimensions (risk reduction, residual risk, timeliness, sustainability, completeness, verification).
- REQ-025-054: The agent SHALL score mitigation effectiveness on a 0-100 scale and classify as Highly Effective, Effective, Partially Effective, Marginally Effective, or Ineffective.
- REQ-025-055: The agent SHALL implement continuous effectiveness monitoring with risk-level-appropriate review schedules and trigger-based reassessment.
- REQ-025-056: The agent SHALL detect mitigation degradation through automated monitoring of degradation indicators and trigger escalation when effectiveness drops below the Effective threshold.

---

## 8. Continuous Monitoring and Adaptive Management

### 8.1 Adaptive Risk Management Cycle

The EUDR's annual review requirement (Article 8(3) and Article 10(4)) mandates an adaptive management approach where risk mitigation is continuously refined based on new information, changing conditions, and effectiveness feedback. The agent implements a Plan-Do-Check-Act (PDCA) cycle adapted for EUDR risk mitigation:

**Plan:**
- Identify and assess risks using EUDR-016 through EUDR-024 outputs
- Select mitigation strategies using the hierarchy and selection methodology
- Develop remediation plans with milestones, resources, and verification criteria
- Obtain compliance officer approval

**Do:**
- Implement mitigation measures per approved plans
- Execute supplier capacity building programs
- Collect verification evidence
- Document implementation activities

**Check:**
- Assess mitigation effectiveness using the scoring model
- Measure residual risk against targets
- Review monitoring data for degradation indicators
- Analyze trends in risk scores and mitigation outcomes across the portfolio

**Act:**
- Adjust mitigation strategies for underperforming measures
- Escalate to higher-tier mitigation where needed
- Update risk management policies and procedures
- Share lessons learned across the organization

### 8.2 Event-Driven Mitigation Adjustments

The agent subscribes to events from upstream risk agents and adjusts mitigation in real-time:

| Event | Source Agent | Mitigation Response |
|-------|-------------|-------------------|
| Country risk score increase | EUDR-016 | Escalate mitigation for all supply chains from affected country |
| Supplier risk score increase | EUDR-017 | Trigger supplier-specific mitigation review |
| Commodity risk alert | EUDR-018 | Escalate mitigation for affected commodity-origin combinations |
| Corruption index change | EUDR-019 | Adjust documentation and verification requirements |
| Deforestation alert issued | EUDR-020 | Immediate containment; investigation; product hold if on supply plot |
| Indigenous rights violation detected | EUDR-021 | FPIC verification; community engagement; potential product hold |
| Protected area encroachment detected | EUDR-022 | Boundary verification; potential product block |
| Legal compliance failure identified | EUDR-023 | Legal remediation; permit acquisition; potential product block |
| Audit non-conformance issued | EUDR-024 | CAR management; corrective action tracking |
| Certificate status change | EUDR-024 | Mitigation review; enhanced verification if suspended/withdrawn |

### 8.3 Portfolio-Level Risk Management

Beyond individual product-supply chain mitigation, the agent provides portfolio-level risk management:

**Portfolio Analytics:**
- Aggregate mitigation status across all active supply chains
- Identify systemic risk patterns requiring enterprise-level intervention
- Track total risk exposure (products at risk, revenue at risk, penalty exposure)
- Benchmark mitigation performance across business units, commodities, and origins

**Resource Optimization:**
- Prioritize mitigation resources toward highest-impact risk reductions
- Identify opportunities to consolidate mitigation activities across related supply chains
- Optimize capacity building investments for maximum supplier coverage
- Balance mitigation intensity against cost-proportionality requirements

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-057: The agent SHALL implement the PDCA adaptive management cycle with quarterly plan reviews and annual comprehensive reassessments.
- REQ-025-058: The agent SHALL respond to event-driven risk changes from upstream agents within defined SLA timelines (P0: < 1 hour; P1: < 24 hours; P2: < 72 hours).
- REQ-025-059: The agent SHALL provide portfolio-level risk management analytics including aggregate mitigation status, systemic risk patterns, and total risk exposure.
- REQ-025-060: The agent SHALL optimize mitigation resource allocation across the portfolio using cost-effectiveness analysis.

---

## 9. Integration with Third-Party Audits and Verification

### 9.1 Audit-Mitigation Integration Model

Risk mitigation and third-party auditing are deeply interconnected under the EUDR. Article 10(3) explicitly lists "independent surveys, audits, or other assessments" as a risk mitigation measure. The Risk Mitigation Advisor Agent integrates bidirectionally with EUDR-024 (Third-Party Audit Manager) to coordinate audit-driven mitigation activities.

**Integration Flows:**

| Flow Direction | Data Type | Purpose |
|---------------|-----------|---------|
| EUDR-025 to EUDR-024 | Mitigation-triggered audit requests | When mitigation strategy includes third-party verification, the agent requests EUDR-024 to schedule and manage the audit |
| EUDR-024 to EUDR-025 | Audit findings and NCR status | Audit findings feed into the risk assessment and may trigger new or modified mitigation measures |
| EUDR-025 to EUDR-024 | CAR-linked mitigation plans | When audit CARs require mitigation, the agent generates remediation plans linked to the CAR lifecycle |
| EUDR-024 to EUDR-025 | CAR closure verification | Successful CAR closure and effectiveness verification feeds into mitigation effectiveness scoring |
| EUDR-025 to EUDR-024 | Audit scope recommendations | Risk mitigation analysis informs audit scope and focus areas for risk-based audit scheduling |

### 9.2 Verification Requirements by Mitigation Type

| Mitigation Measure | Verification Method | Verification Frequency | Verification Agent |
|--------------------|-------------------|----------------------|-------------------|
| Enhanced documentation | Document review by EUDR-012 | Per document submission | EUDR-012 (Document Authentication) |
| Independent audit | On-site or remote audit per ISO 19011 | As per audit schedule | EUDR-024 (Third-Party Audit Manager) |
| Satellite monitoring verification | Multi-source satellite analysis | Continuous (5-day Sentinel-2 revisit) | EUDR-020 (Deforestation Alert System) |
| Supplier capacity building | Pre/post assessment; training completion verification | Per program milestone | EUDR-025 (self) |
| Legal compliance remediation | Permit verification; legal opinion | Per remediation milestone | EUDR-023 (Legal Compliance Verifier) |
| Chain-of-custody improvement | Mass balance reconciliation; segregation verification | Monthly or per batch | EUDR-010, EUDR-011 |
| FPIC completion | FPIC documentation review; community verification | Per FPIC process milestone | EUDR-021 (Indigenous Rights Checker) |
| Protected area compliance | Spatial analysis verification | Quarterly | EUDR-022 (Protected Area Validator) |

### 9.3 Certification Scheme Mitigation Credit

Recital 52 of the EUDR states that voluntary certification or third-party verified schemes "could be used in the risk assessment procedure" but "should not substitute the operator's responsibility" for due diligence. The agent implements a structured approach to crediting certification as partial risk mitigation:

**Certification Mitigation Credit Rules:**

| Condition | Credit Allowed | Residual Risk Treatment |
|-----------|---------------|------------------------|
| Active, valid certification with scope covering EUDR requirements | Up to 50% risk score reduction for covered criteria | Operator must independently verify uncovered criteria |
| Certification with partial EUDR coverage | Proportional credit based on coverage mapping | Full mitigation required for uncovered criteria |
| Suspended or under-investigation certification | No credit | Full operator-level mitigation required |
| Expired certification | No credit | Full operator-level mitigation required |
| Certification from non-accredited body | No credit | Full operator-level mitigation required |

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-061: The agent SHALL integrate bidirectionally with EUDR-024 (Third-Party Audit Manager) per the defined integration flow model.
- REQ-025-062: The agent SHALL request audit scheduling through EUDR-024 when mitigation strategies include independent third-party verification.
- REQ-025-063: The agent SHALL ingest audit findings and NCR status from EUDR-024 and incorporate them into mitigation effectiveness assessments.
- REQ-025-064: The agent SHALL apply certification mitigation credit rules per the defined credit table, limiting credit to a maximum of 50% risk score reduction and requiring operator-level verification for uncovered criteria.

---

## 10. Documentation Requirements for Due Diligence Statements

### 10.1 DDS Risk Mitigation Section

The due diligence statement (DDS) submitted to the EU Information System under Article 12 must include documentation of risk mitigation measures. The agent generates the risk mitigation section of the DDS with the following content:

**Mandatory DDS Mitigation Content:**

| Field | Content | Data Source |
|-------|---------|-------------|
| Risk Assessment Summary | Composite risk score, individual risk factor scores, risk level classification | EUDR-016 through EUDR-024 outputs |
| Mitigation Decision | Whether mitigation was required (negligible risk or non-negligible risk identified) | Composite risk threshold evaluation |
| Mitigation Measures Applied | Description of each mitigation measure implemented | Remediation plan records |
| Mitigation Effectiveness | Effectiveness score and residual risk assessment for each measure | Effectiveness assessment framework |
| Residual Risk Determination | Final residual risk score and determination (negligible or not) | Post-mitigation risk recalculation |
| Verification Evidence | References to verification documents (audit reports, satellite analyses, certifications) | Evidence registry |
| Compliance Officer Attestation | Name, title, and attestation of the compliance officer who approved the mitigation | Approval record |
| Review Date | Date of most recent annual review of mitigation measures | Review tracking |

### 10.2 Mitigation Evidence Package

For each DDS, the agent maintains a mitigation evidence package that can be produced for competent authority inspection:

**Evidence Package Contents:**

| Evidence Category | Document Types | Retention |
|-------------------|---------------|-----------|
| Risk Assessment Records | Risk agent output reports, composite score calculations, risk factor analyses | 5 years per Art. 4(7) |
| Mitigation Plans | Remediation plans, resource allocation records, approval records | 5 years per Art. 4(7) |
| Implementation Evidence | Activity logs, supplier communications, training records, document submissions | 5 years per Art. 4(7) |
| Verification Reports | Audit reports, satellite analysis reports, certification verification, document authentication | 5 years per Art. 4(7) |
| Effectiveness Assessments | Effectiveness scores, residual risk calculations, review records | 5 years per Art. 4(7) |
| Supplier Capacity Building | Program enrollment, completion certificates, pre/post assessments | 5 years per Art. 4(7) |
| Stakeholder Engagement | Community consultation records, grievance responses, FPIC documentation | 5 years per Art. 4(7) |
| Compliance Officer Records | Decision logs, approval records, annual review reports | 5 years per Art. 4(7) |

### 10.3 Documentation Integrity

All mitigation documentation must meet the GreenLang platform immutability and provenance standards:

- SHA-256 content hashes on all generated documents
- Append-only audit trail for all mitigation records
- Tamper detection through hash chain verification
- Digital signature on compliance officer attestations
- Integration with SEC-005 (Centralized Audit Logging) for all mitigation activities
- Cross-reference linkage between DDS records, risk assessments, mitigation plans, and verification evidence

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-065: The agent SHALL generate the risk mitigation section of the DDS containing all mandatory fields per the defined content table.
- REQ-025-066: The agent SHALL maintain mitigation evidence packages for each DDS with all eight evidence categories, retained for a minimum of 5 years per Article 4(7).
- REQ-025-067: The agent SHALL implement SHA-256 hash chains, digital signatures, and tamper detection on all mitigation documentation.
- REQ-025-068: The agent SHALL integrate with SEC-005 (Centralized Audit Logging) for audit trail of all mitigation activities.

---

## 11. Competent Authority Acceptance Criteria

### 11.1 Adequacy Standards for Risk Mitigation

Competent authorities assessing operator compliance under Article 29 will evaluate the adequacy of risk mitigation measures. While the EUDR does not prescribe specific acceptance criteria, the European Commission Guidance Document (2nd Edition, April 2025) and Member State enforcement practices establish the following adequacy standards:

**Proportionality Test:**
- Mitigation measures must be proportionate to the severity and nature of the identified risk
- Low-risk scenarios require lighter mitigation; high-risk scenarios require more intensive measures
- Cost and feasibility constraints are recognized, but cannot justify inadequate mitigation for serious risks

**Effectiveness Test:**
- Mitigation measures must demonstrably reduce the identified risk
- Operators must be able to show evidence of risk reduction, not merely the intent to mitigate
- Residual risk after mitigation must be negligible or, at minimum, significantly reduced from the pre-mitigation level

**Completeness Test:**
- All material risk factors identified in the Article 10(2) assessment must be addressed by at least one mitigation measure
- No significant risk factor may be left unmitigated without documented justification
- Mitigation coverage must span all dimensions of the product's supply chain

**Documentation Test:**
- Mitigation decisions, rationale, implementation, and verification must be documented
- Documentation must be retrievable and presentable to competent authorities within the timeframes specified by national implementing legislation
- Documentation must be internally consistent and supported by verifiable evidence

**Timeliness Test:**
- Mitigation measures must be implemented before the product is placed on the market
- There must be no temporal gap between risk identification and mitigation implementation
- Annual reviews must be conducted and documented as required by Article 10(4)

### 11.2 Competent Authority Rejection Criteria

Competent authorities may reject risk mitigation as inadequate in the following circumstances:

| Rejection Criterion | Description | Agent Safeguard |
|---------------------|-------------|-----------------|
| Disproportionate to risk | Mitigation intensity too low for the severity of identified risk | Automated proportionality check via risk level-to-mitigation intensity mapping |
| No evidence of effectiveness | Mitigation measures documented but no verification of risk reduction | Mandatory effectiveness assessment with scored verification |
| Incomplete risk coverage | Material risk factors identified but not addressed by mitigation | Risk factor-to-mitigation linkage matrix with gap detection |
| Insufficient documentation | Mitigation records incomplete, inconsistent, or unverifiable | 15-section remediation plan template with mandatory fields |
| Untimely implementation | Mitigation implemented after market placement | DDS submission gate requiring verified mitigation before clearance |
| Stale mitigation | Mitigation based on outdated risk assessment (no annual review) | Automated annual review enforcement with stale mitigation alerts |
| Self-serving assessment | Operator's mitigation assessment lacks independence or objectivity | Third-party verification integration; certification credit limits |
| Recurring non-compliance | Same risk factors recur despite previous mitigation, suggesting ineffective measures | Recurrence tracking; escalation to higher-tier mitigation |

### 11.3 Member State Enforcement Variation

Each of the 27 EU Member States implements the EUDR through national legislation, which may add specific requirements for risk mitigation adequacy. The agent supports configurable Member State profiles:

| Configuration Area | Variable by Member State |
|-------------------|------------------------|
| Response timeframes | Timeframes for responding to competent authority requests for mitigation documentation |
| Documentation language | Language requirements for mitigation documentation (national language, English, or both) |
| Electronic submission formats | Technical formats for electronic submission of mitigation records |
| Penalty calculation | Specific penalty calculation methodologies within the Article 23 framework |
| Inspection protocols | Specific inspection procedures and advance notification requirements |
| Appeal procedures | Procedures for appealing competent authority mitigation adequacy determinations |

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-069: The agent SHALL validate mitigation adequacy against the five acceptance tests (proportionality, effectiveness, completeness, documentation, timeliness) before clearing products for DDS submission.
- REQ-025-070: The agent SHALL detect and prevent the eight competent authority rejection scenarios through automated safeguards.
- REQ-025-071: The agent SHALL support configurable Member State enforcement profiles for the 27 EU Member States, accommodating variations in response timeframes, language requirements, submission formats, and inspection protocols.

---

## 12. Integration with Risk Assessment Agents (EUDR-016 through EUDR-024)

### 12.1 Agent Dependency Matrix

The Risk Mitigation Advisor Agent (EUDR-025) is the downstream consumer of all risk assessment agent outputs and the upstream provider of mitigation status information back to risk assessment agents.

**Inbound Data Flows (Risk Assessment to Mitigation):**

| Source Agent | Data Type | Update Frequency | Use in Mitigation |
|-------------|-----------|-----------------|-------------------|
| EUDR-016 (Country Risk Evaluator) | Country risk scores (0-100), due diligence level classification, Article 29 benchmarking | On country score change; daily sync | Determine mitigation intensity level; drive country-specific mitigation strategies |
| EUDR-017 (Supplier Risk Scorer) | Supplier risk scores (0-100), supplier compliance history | On supplier score change; daily sync | Prioritize supplier-specific mitigation; inform capacity building recommendations |
| EUDR-018 (Commodity Risk Analyzer) | Commodity-specific risk profiles, deforestation correlation | On commodity risk change; monthly update | Select commodity-appropriate mitigation measures |
| EUDR-019 (Corruption Index Monitor) | Corruption risk scores, governance indicators | On index change; quarterly update | Adjust documentation and verification requirements; enhanced due diligence triggers |
| EUDR-020 (Deforestation Alert System) | Deforestation alerts with severity, location, cutoff date verification | Real-time (event-driven) | Immediate containment mitigation; product hold triggers; investigation initiation |
| EUDR-021 (Indigenous Rights Checker) | Territory overlap analysis, FPIC status, rights violation alerts | On status change; event-driven | FPIC remediation; community engagement mitigation; product hold triggers |
| EUDR-022 (Protected Area Validator) | Protected area overlap, buffer zone status, encroachment alerts | On status change; event-driven | Boundary verification mitigation; origin change recommendations |
| EUDR-023 (Legal Compliance Verifier) | Legal compliance scores (8 categories), document verification, red flag alerts | On compliance change; event-driven | Legal remediation plans; permit acquisition support; documentation mitigation |
| EUDR-024 (Third-Party Audit Manager) | Audit findings, NCR status, CAR lifecycle, certification status | On audit completion; event-driven | Audit-driven mitigation; CAR-linked remediation; certification credit |

**Outbound Data Flows (Mitigation to Risk Assessment):**

| Target Agent | Data Type | Update Frequency | Purpose |
|-------------|-----------|-----------------|---------|
| EUDR-017 (Supplier Risk Scorer) | Mitigation status per supplier, capacity building progress | On mitigation status change | Adjust supplier risk score based on active mitigation measures |
| EUDR-024 (Third-Party Audit Manager) | Mitigation-triggered audit requests, remediation plan scope | On mitigation plan creation | Schedule verification audits aligned with mitigation plans |
| GL-EUDR-APP | Mitigation dashboards, DDS mitigation section, portfolio risk status | Real-time | User-facing mitigation management and reporting |

### 12.2 Event Bus Integration

The agent publishes and subscribes to events via the GreenLang event bus:

**Published Events:**

| Event | Description | Trigger |
|-------|-------------|---------|
| `eudr.mitigation.plan_created` | New remediation plan created for a product-supply chain | Risk assessment identifies non-negligible risk |
| `eudr.mitigation.plan_updated` | Remediation plan modified (scope, timeline, measures) | Plan amendment by compliance officer |
| `eudr.mitigation.measure_implemented` | Specific mitigation measure completed | Implementation confirmation |
| `eudr.mitigation.effectiveness_assessed` | Effectiveness assessment completed for a measure | Scheduled or triggered assessment |
| `eudr.mitigation.product_cleared` | Product cleared for DDS submission after mitigation | Residual risk reduced to negligible |
| `eudr.mitigation.product_blocked` | Product blocked from DDS submission due to unmitigated risk | Non-negligible risk without adequate mitigation |
| `eudr.mitigation.escalation_triggered` | Mitigation escalated to higher tier | Ineffective mitigation; degradation detected |
| `eudr.mitigation.capacity_building_completed` | Supplier capacity building program completed | Program milestone reached |
| `eudr.mitigation.annual_review_completed` | Annual mitigation review completed per Article 10(4) | Annual review cycle |

**Subscribed Events:**

| Event | Source | Response |
|-------|--------|----------|
| `eudr.risk.country_updated` | EUDR-016 | Re-evaluate all mitigation plans for affected country |
| `eudr.risk.supplier_updated` | EUDR-017 | Re-evaluate supplier-specific mitigation plans |
| `eudr.risk.commodity_updated` | EUDR-018 | Re-evaluate commodity-specific mitigation strategies |
| `eudr.risk.corruption_updated` | EUDR-019 | Adjust verification and documentation requirements |
| `eudr.alert.deforestation_detected` | EUDR-020 | Immediate containment; product hold assessment |
| `eudr.rights.violation_detected` | EUDR-021 | FPIC remediation; community engagement |
| `eudr.area.encroachment_detected` | EUDR-022 | Protected area remediation; boundary verification |
| `eudr.legal.compliance_failed` | EUDR-023 | Legal remediation plan initiation |
| `eudr.audit.completed` | EUDR-024 | Incorporate findings into mitigation assessment |
| `eudr.car.issued` | EUDR-024 | Link CAR to mitigation plan; track remediation |
| `eudr.car.closed` | EUDR-024 | Update mitigation effectiveness; recalculate residual risk |
| `eudr.certification.status_changed` | EUDR-024 | Recalculate certification credit; adjust mitigation |

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-072: The agent SHALL consume risk assessment outputs from all nine upstream agents (EUDR-016 through EUDR-024) per the defined inbound data flow matrix.
- REQ-025-073: The agent SHALL publish mitigation status updates to downstream consumers (EUDR-017, EUDR-024, GL-EUDR-APP) per the defined outbound data flow matrix.
- REQ-025-074: The agent SHALL publish and subscribe to all defined event types via the GreenLang event bus with guaranteed message delivery (99.9% SLA).
- REQ-025-075: The agent SHALL respond to critical events (deforestation detection, legal failure, rights violation, protected area encroachment) within 1 hour of event receipt.

---

## 13. Data Retention and Immutability Requirements

### 13.1 Retention Schedule

| Record Type | Minimum Retention | Regulatory Basis | Storage |
|-------------|-------------------|------------------|---------|
| Risk Mitigation Plans | 5 years from associated DDS date | EUDR Art. 4(7) | Encrypted at rest (AES-256-GCM) |
| Mitigation Implementation Evidence | 5 years from implementation date | EUDR Art. 4(7) | Encrypted at rest (AES-256-GCM) |
| Effectiveness Assessment Records | 5 years from assessment date | EUDR Art. 4(7), Art. 10(4) | Encrypted at rest (AES-256-GCM) |
| Composite Risk Score History | 5 years from score date | EUDR Art. 4(7) | Encrypted at rest (AES-256-GCM) |
| DDS Mitigation Sections | 5 years from DDS submission | EUDR Art. 4(7), Art. 12 | Encrypted at rest (AES-256-GCM) |
| Capacity Building Program Records | 5 years from program completion | EUDR Art. 4(7) | Encrypted at rest (AES-256-GCM) |
| Competent Authority Interaction Records | 5 years from interaction date | EUDR Art. 29 | Encrypted at rest (AES-256-GCM) |
| Annual Review Records | 5 years from review date | EUDR Art. 10(4) | Encrypted at rest (AES-256-GCM) |

### 13.2 Immutability and Provenance

All mitigation records must be immutable once finalized:

- SHA-256 content hashes on all mitigation plans, assessment records, and evidence files
- Append-only audit trail for all record modifications (no deletion, no overwrite)
- Tamper detection through hash chain verification
- Digital signature on compliance officer attestations and DDS mitigation sections
- Integration with SEC-005 (Centralized Audit Logging) for all agent activities
- Provenance tracking linking every mitigation decision to its regulatory basis, risk assessment inputs, and verification evidence

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-076: The agent SHALL enforce minimum 5-year retention for all mitigation-related records per EUDR Article 4(7).
- REQ-025-077: The agent SHALL implement immutable record storage with SHA-256 hash chains, digital signatures, and tamper detection.
- REQ-025-078: The agent SHALL integrate with SEC-005 (Centralized Audit Logging) for audit trail of all mitigation management activities.

---

## 14. Performance and Quality Requirements

### 14.1 System Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| Composite risk score calculation | < 2 seconds for single product-supply chain | p99 latency |
| Mitigation strategy recommendation | < 5 seconds per product-supply chain | p99 latency |
| Remediation plan generation | < 10 seconds per plan | p99 latency |
| DDS mitigation section generation | < 5 seconds per DDS | p99 latency |
| Effectiveness assessment calculation | < 3 seconds per measure | p99 latency |
| Portfolio risk dashboard refresh | < 5 seconds for full portfolio | p99 latency |
| Event response (P0 critical) | < 1 hour from event receipt to mitigation action | SLA compliance |
| Event response (P1 urgent) | < 24 hours from event receipt to mitigation action | SLA compliance |
| Event response (P2 standard) | < 72 hours from event receipt to mitigation action | SLA compliance |
| Concurrent mitigation plans | 5,000+ active plans | Capacity |
| API availability | 99.9% uptime | Monthly SLA |
| Event bus message delivery | 99.9% guaranteed delivery | Message delivery SLA |

### 14.2 Zero-Hallucination Guarantee

Consistent with the GreenLang platform standard:

- All risk scores, mitigation recommendations, and effectiveness assessments are deterministic
- No LLM in the critical path for compliance determinations
- SHA-256 provenance hashes on all generated outputs
- Bit-perfect reproducibility: Same inputs produce identical outputs
- All regulatory references are traceable to specific articles, clauses, and provisions
- Mitigation strategy selection is rule-based and auditable, not probabilistic

**Regulatory Requirement for Agent EUDR-025:**
- REQ-025-079: The agent SHALL meet all performance targets specified in Section 14.1.
- REQ-025-080: The agent SHALL implement zero-hallucination deterministic processing for all compliance-critical operations.

---

## 15. Requirements Traceability Matrix

| Requirement ID | Description | Regulatory Source | Priority |
|----------------|-------------|-------------------|----------|
| REQ-025-001 | Documented risk mitigation policies and procedures | EUDR Art. 8(2) | P0 |
| REQ-025-002 | Annual review cycle enforcement | EUDR Art. 8(3) | P0 |
| REQ-025-003 | Compliance officer accountability tracking | EUDR Art. 8(2) | P0 |
| REQ-025-004 | DDS submission blocking for unmitigated non-negligible risk | EUDR Art. 10(3) | P0 |
| REQ-025-005 | Article 9(1) information completeness evaluation | EUDR Art. 9(1) | P0 |
| REQ-025-006 | Information gap categorization and mitigation recommendation | EUDR Art. 9(1), Art. 10 | P0 |
| REQ-025-007 | Information gap closure tracking | EUDR Art. 9(1) | P0 |
| REQ-025-008 | Evaluate all 14 Article 10(2) risk criteria | EUDR Art. 10(2) | P0 |
| REQ-025-009 | Non-negligible risk determination engine | EUDR Art. 10(1), (3) | P0 |
| REQ-025-010 | Mitigation measure recommendation | EUDR Art. 10(3), ISO 31000 | P0 |
| REQ-025-011 | Document all mitigation decisions per Art. 10(4) | EUDR Art. 10(4) | P0 |
| REQ-025-012 | Market placement prohibition enforcement | EUDR Art. 10(3) | P0 |
| REQ-025-013 | Annual mitigation review and re-assessment | EUDR Art. 10(4) | P0 |
| REQ-025-014 | Simplified due diligence eligibility evaluation | EUDR Art. 11, 13, 29 | P0 |
| REQ-025-015 | Mitigation downgrade for low-risk country products | EUDR Art. 13 | P0 |
| REQ-025-016 | New information trigger monitoring for simplified DD | EUDR Art. 13 | P1 |
| REQ-025-017 | Country reclassification escalation | EUDR Art. 29 | P0 |
| REQ-025-018 | DDS risk mitigation section generation | EUDR Art. 12(2) | P0 |
| REQ-025-019 | DDS readiness validation | EUDR Art. 12(2) | P0 |
| REQ-025-020 | DDS-to-mitigation audit trail | EUDR Art. 4(7), Art. 12 | P0 |
| REQ-025-021 | CA remedial action plan management | EUDR Art. 18 | P0 |
| REQ-025-022 | Interim measure tracking | EUDR Art. 19 | P0 |
| REQ-025-023 | Definitive measure response management | EUDR Art. 20 | P0 |
| REQ-025-024 | Self-disclosure workflow per Art. 21 | EUDR Art. 21 | P0 |
| REQ-025-025 | Penalty exposure calculation | EUDR Art. 22-23 | P1 |
| REQ-025-026 | Compliance defense package generation | EUDR Art. 22-23 | P1 |
| REQ-025-027 | Risk-differentiated mitigation intensity | EUDR Art. 29 | P0 |
| REQ-025-028 | Automatic mitigation escalation on country reclassification | EUDR Art. 29 | P0 |
| REQ-025-029 | Enhanced documentation for high-risk country sourcing | EUDR Art. 29 | P0 |
| REQ-025-030 | ISO 31000 risk treatment process implementation | ISO 31000:2018 Cl. 6.5 | P0 |
| REQ-025-031 | ISO 31000 risk treatment option support | ISO 31000:2018 Cl. 6.5.2 | P0 |
| REQ-025-032 | Four-tier mitigation hierarchy implementation | ISO 31000:2018, EUDR Art. 10(3) | P0 |
| REQ-025-033 | Hierarchical strategy recommendation | ISO 31000:2018, EUDR Art. 10(3) | P0 |
| REQ-025-034 | EUDR constraint on risk acceptance (negligible only) | EUDR Art. 10(3) | P0 |
| REQ-025-035 | ISO 31000 Cl. 6.5.3 compliant mitigation plans | ISO 31000:2018 Cl. 6.5.3 | P0 |
| REQ-025-036 | Mitigation plan integration into management processes | ISO 31000:2018 Cl. 6.5.3 | P1 |
| REQ-025-037 | Mitigation plan implementation tracking with escalation | ISO 31000:2018, EUDR Art. 10(4) | P0 |
| REQ-025-038 | Weighted multi-agent composite risk score calculation | EUDR Art. 10(2) | P0 |
| REQ-025-039 | Six-level risk classification | EUDR Art. 10, ISO 31000 | P0 |
| REQ-025-040 | Individual risk factor trigger evaluation | EUDR Art. 10(2) | P0 |
| REQ-025-041 | Dynamic risk score recalculation on agent updates | EUDR Art. 8(3) | P0 |
| REQ-025-042 | Six-criterion strategy selection framework | ISO 31000:2018 Cl. 6.5.2 | P0 |
| REQ-025-043 | Strategy combination rules (additive, exclusive, sequenced) | ISO 31000:2018 Cl. 6.5.2 | P1 |
| REQ-025-044 | Scored recommendation reports per strategy | ISO 31000:2018, EUDR Art. 10(4) | P1 |
| REQ-025-045 | 15-section remediation plan generation | ISO 31000:2018 Cl. 6.5.3 | P0 |
| REQ-025-046 | Remediation plan categorization | EUDR Art. 10, 18, 21 | P0 |
| REQ-025-047 | Root cause analysis for High/Critical risks | ISO 31000:2018, ISO 31010 | P0 |
| REQ-025-048 | Remediation plan implementation tracking | ISO 31000:2018 Cl. 6.5.3 | P0 |
| REQ-025-049 | Four-level supplier capacity building programs | EUDR Art. 10(3), Recital 34 | P0 |
| REQ-025-050 | Capacity building enrollment and progress tracking | EUDR Art. 10(3) | P0 |
| REQ-025-051 | Capacity building effectiveness measurement | EUDR Art. 10(3), Art. 10(4) | P1 |
| REQ-025-052 | Risk-based capacity building level recommendation | EUDR Art. 10(3) | P1 |
| REQ-025-053 | Six-dimension effectiveness assessment | ISO 31000:2018 Cl. 6.5 | P0 |
| REQ-025-054 | Effectiveness scoring model (0-100) | ISO 31000:2018 Cl. 6.5 | P0 |
| REQ-025-055 | Continuous effectiveness monitoring | EUDR Art. 10(4), ISO 31000 | P0 |
| REQ-025-056 | Mitigation degradation detection and escalation | EUDR Art. 8(3), ISO 31000 | P0 |
| REQ-025-057 | PDCA adaptive management cycle | ISO 31000:2018, EUDR Art. 8(3) | P0 |
| REQ-025-058 | Event-driven risk response SLAs | GreenLang Architecture | P0 |
| REQ-025-059 | Portfolio-level risk management analytics | ISO 31000:2018 Cl. 4 | P1 |
| REQ-025-060 | Resource allocation optimization | ISO 31000:2018 Cl. 6.5.3 | P1 |
| REQ-025-061 | Bidirectional EUDR-024 integration | GreenLang Architecture | P0 |
| REQ-025-062 | Mitigation-triggered audit scheduling | EUDR Art. 10(3), ISO 19011 | P0 |
| REQ-025-063 | Audit finding ingestion for effectiveness assessment | EUDR Art. 10(2)(l)-(n) | P0 |
| REQ-025-064 | Certification mitigation credit rules | EUDR Recital 52 | P0 |
| REQ-025-065 | DDS mitigation section generation | EUDR Art. 12(2) | P0 |
| REQ-025-066 | Mitigation evidence package maintenance | EUDR Art. 4(7) | P0 |
| REQ-025-067 | SHA-256 hash chains and digital signatures | EUDR Art. 4(7), SEC-003/005 | P0 |
| REQ-025-068 | SEC-005 audit logging integration | SEC-005 | P0 |
| REQ-025-069 | Five-test mitigation adequacy validation | EUDR Art. 10(3), 29 | P0 |
| REQ-025-070 | Eight rejection scenario prevention | EUDR Art. 10(3), 29 | P0 |
| REQ-025-071 | 27 Member State enforcement profile support | EUDR Art. 29 | P1 |
| REQ-025-072 | Nine-agent inbound data flow consumption | GreenLang Architecture | P0 |
| REQ-025-073 | Outbound mitigation status publication | GreenLang Architecture | P0 |
| REQ-025-074 | Event bus integration (99.9% SLA) | GreenLang Architecture | P0 |
| REQ-025-075 | Critical event response within 1 hour | GreenLang Architecture, EUDR | P0 |
| REQ-025-076 | 5-year record retention | EUDR Art. 4(7) | P0 |
| REQ-025-077 | Immutable storage with hash chains | EUDR Art. 4(7), SEC-003/005 | P0 |
| REQ-025-078 | SEC-005 audit logging for mitigation activities | SEC-005 | P0 |
| REQ-025-079 | Performance targets per Section 14.1 | System requirement | P0 |
| REQ-025-080 | Zero-hallucination deterministic processing | GreenLang standard | P0 |

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **Adaptive Management** | A structured, iterative approach to risk management that adjusts mitigation strategies based on monitoring results, effectiveness feedback, and changing conditions |
| **Capacity Building** | Structured programs to develop supplier capabilities for EUDR compliance, including training, technical assistance, and investment support |
| **Composite Risk Score** | A weighted aggregate score (0-100) synthesizing outputs from nine EUDR risk assessment agents into a single risk determination |
| **Competent Authority** | National authority designated by an EU Member State to enforce the EUDR per Article 29 |
| **Corrective Action** | Measures taken to eliminate the cause of a detected non-conformity or other undesirable situation and prevent its recurrence |
| **DDS** | Due Diligence Statement -- the formal statement submitted to the EU Information System per EUDR Article 4(2) confirming that due diligence has been exercised |
| **Degradation Indicator** | A measurable signal that a previously effective mitigation measure is losing its effectiveness over time |
| **Effectiveness Assessment** | A systematic evaluation of whether an implemented mitigation measure has achieved its intended risk reduction objective |
| **EUDR** | EU Deforestation Regulation -- Regulation (EU) 2023/1115 |
| **FPIC** | Free, Prior and Informed Consent -- the right of indigenous peoples to give or withhold consent to activities affecting their lands, territories, and resources |
| **Mitigation Hierarchy** | A four-tier priority ordering of risk treatment strategies: Avoid, Reduce, Transfer, Accept |
| **Negligible Risk** | A level of risk so low that it does not require active mitigation measures; the threshold below which products may be placed on the EU market |
| **Non-Negligible Risk** | A level of risk that requires active mitigation measures before the product may be placed on the EU market under EUDR Article 10(3) |
| **PDCA** | Plan-Do-Check-Act cycle -- a continuous improvement methodology applied to risk mitigation management |
| **Portfolio Risk** | The aggregate risk exposure across all of an operator's active supply chains, products, and sourcing origins |
| **Remediation Plan** | A structured document defining the mitigation measures, timeline, resources, responsibilities, and verification criteria for addressing identified risks |
| **Residual Risk** | The risk remaining after mitigation measures have been implemented; must be negligible for EUDR compliance |
| **Risk Avoidance** | Tier 1 mitigation strategy involving the elimination of the risk source (e.g., supplier substitution, origin change) |
| **Risk Reduction** | Tier 2 mitigation strategy involving measures to decrease the likelihood or consequence of the identified risk |
| **Risk Transfer** | Tier 3 mitigation strategy involving the sharing of risk with another party through contracts, insurance, or third-party verification |
| **Risk Acceptance** | Tier 4 mitigation strategy involving the informed decision to retain a risk; under EUDR, only permissible for negligible residual risk |
| **Root Cause Analysis** | A systematic investigation to identify the underlying causes of a risk or non-conformity (5-Why, Fishbone/Ishikawa, Fault Tree) |
| **Substantiated Concern** | An evidence-based notification submitted to a competent authority per EUDR Article 14 alleging non-compliance |

---

## 17. References

### 17.1 Primary Regulatory Sources

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 on deforestation-free products
2. Regulation (EU) 2024/3234 amending Regulation (EU) 2023/1115 as regards the date of application
3. European Commission EUDR Guidance Document (2nd Edition, April 2025)
4. European Commission FAQ on EUDR Implementation (May 2025, 40+ new answers)
5. European Commission Implementing Regulation on the EU Information System (TRACES)

### 17.2 International Standards

6. ISO 31000:2018 -- Risk management -- Guidelines
7. ISO 31010:2019 -- Risk management -- Risk assessment techniques
8. ISO 19011:2018 -- Guidelines for auditing management systems
9. ISO/IEC 17065:2012 -- Requirements for bodies certifying products, processes, and services
10. ISO 14001:2015 -- Environmental management systems

### 17.3 Due Diligence Guidance

11. OECD Due Diligence Guidance for Responsible Supply Chains of Minerals from Conflict-Affected and High-Risk Areas (3rd Edition)
12. OECD-FAO Guidance for Responsible Agricultural Supply Chains (2016)
13. UN Guiding Principles on Business and Human Rights (Ruggie Principles, 2011)
14. European Commission Communication on Due Diligence (COM(2022) 71 final)

### 17.4 GreenLang Platform References

15. PRD-AGENT-EUDR-016 -- Country Risk Evaluator Agent
16. PRD-AGENT-EUDR-017 -- Supplier Risk Scorer Agent
17. PRD-AGENT-EUDR-018 -- Commodity Risk Analyzer Agent
18. PRD-AGENT-EUDR-019 -- Corruption Index Monitor Agent
19. PRD-AGENT-EUDR-020 -- Deforestation Alert System Agent
20. PRD-AGENT-EUDR-021 -- Indigenous Rights Checker Agent
21. PRD-AGENT-EUDR-022 -- Protected Area Validator Agent
22. PRD-AGENT-EUDR-023 -- Legal Compliance Verifier Agent
23. PRD-AGENT-EUDR-024 -- Third-Party Audit Manager Agent
24. REQS-AGENT-EUDR-024 -- Third-Party Audit Manager Regulatory Requirements
25. SEC-003 -- Encryption at Rest (AES-256-GCM)
26. SEC-005 -- Centralized Audit Logging

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-11 | GL-RegulatoryIntelligence | Initial regulatory requirements document |

---

*This document was prepared by GL-RegulatoryIntelligence based on analysis of Regulation (EU) 2023/1115 (including amendments by Regulation (EU) 2024/3234), ISO 31000:2018, ISO 31010:2019, OECD Due Diligence Guidance, and the European Commission EUDR Guidance Document. All regulatory references are traceable to specific articles, clauses, and provisions. This document does not constitute legal advice.*
