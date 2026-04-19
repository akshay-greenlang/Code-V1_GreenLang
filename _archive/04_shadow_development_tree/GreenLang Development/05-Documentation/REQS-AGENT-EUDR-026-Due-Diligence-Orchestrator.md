# REQS: AGENT-EUDR-026 -- Due Diligence Orchestrator Agent
# Regulatory Requirements Document

## Document Info

| Field | Value |
|-------|-------|
| **REQS ID** | REQS-AGENT-EUDR-026 |
| **Agent ID** | GL-EUDR-DDO-026 |
| **Component** | Due Diligence Orchestrator Agent |
| **Category** | EUDR Regulatory Agent -- Due Diligence Workflow Orchestration |
| **Priority** | P0 -- Critical (EUDR Enforcement Active) |
| **Version** | 1.0.0 |
| **Status** | Draft |
| **Author** | GL-RegulatoryIntelligence |
| **Date** | 2026-03-11 |
| **Regulation** | Regulation (EU) 2023/1115 (EUDR), Articles 3, 4, 5, 8, 9, 10, 11, 12, 13, 14-16, 18-23, 29; Regulation (EU) 2024/3234 (First Postponement); Regulation (EU) 2025/2650 (Second Postponement and Simplification); ISO 19011:2018 (Auditing Management Systems); ISO 31000:2018 (Risk Management); OECD Due Diligence Guidance for Responsible Agricultural Supply Chains |
| **Enforcement** | December 30, 2026 (large/medium operators); June 30, 2027 (SMEs) |

---

## 1. Regulatory Basis and Legal Framework

### 1.1 Primary Regulation: EU Deforestation Regulation (EUDR)

Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023, concerning the making available on the Union market and the export from the Union of certain commodities and products associated with deforestation and forest degradation, establishes a mandatory three-step due diligence framework for operators and traders dealing in seven regulated commodities (cattle, cocoa, coffee, oil palm, rubber, soya, wood) and their derived products. The regulation requires every operator to exercise due diligence prior to placing relevant products on the Union market or exporting them, and to submit a Due Diligence Statement (DDS) confirming that the risk of non-compliance has been assessed and found to be negligible, or that identified risks have been adequately mitigated.

The Due Diligence Orchestrator Agent (EUDR-026) is the capstone agent of the entire EUDR agent family (EUDR-001 through EUDR-025). It is not a risk assessment agent, a supply chain traceability agent, or a risk mitigation agent. It is the workflow coordination engine that orchestrates all 25 upstream agents into a coherent, auditable, three-phase due diligence process that satisfies Articles 8, 9, 10, 11, 12, and 13 of the EUDR. Without an orchestration layer, the 25 individual agent capabilities remain disconnected tools rather than an integrated due diligence system as the regulation requires.

The EUDR mandates a system, not a collection of point solutions. Article 8(2) requires operators to "establish, implement, maintain, regularly evaluate, and update a due diligence system" -- the word "system" implies coordinated, repeatable, documented processes that produce consistent outcomes. The Due Diligence Orchestrator Agent is the technical embodiment of this system requirement.

**Enforcement Timeline (as amended by Regulation (EU) 2025/2650):**
- December 30, 2026: Large and medium operators/traders -- full obligations apply
- June 30, 2027: Micro and small operators/traders -- full obligations apply
- Competent authorities enforcement duties active from enforcement dates
- EU Information System (TRACES NT) operational for DDS submission

### 1.2 Article 8 -- Due Diligence System Requirements

Article 8 of the EUDR defines the due diligence system that operators and non-SME traders must establish, implement, and maintain. This article is the primary regulatory basis for the orchestrator agent because it mandates a systematic, maintained, and documented approach to due diligence.

**Article 8(1):** Prior to placing relevant products on the Union market or exporting them, operators shall exercise due diligence in relation to all relevant products. The due diligence shall include:
- (a) the information collection referred to in Article 9
- (b) the risk assessment measures referred to in Article 10
- (c) the risk mitigation measures referred to in Article 10

**Article 8(2):** Operators shall establish, implement, maintain, regularly evaluate, and update a due diligence system. The system shall include:
- Risk management policies and procedures
- A designated compliance officer
- Adequate human and technical resources
- Documented internal controls

**Article 8(3):** Operators shall review and update their due diligence system at least once per year and whenever there is a significant change in the nature of the risk or the operator's supply chain.

**Article 8(4):** Operators shall keep records of the due diligence system and the due diligence statements for at least five years from the date the statements were submitted.

**Implications for Orchestration:**

The Article 8 system requirement creates five mandatory orchestration capabilities:

1. **Sequential Phase Enforcement**: The three due diligence steps (information collection, risk assessment, risk mitigation) must be executed in order. The orchestrator must enforce that Phase 1 (information) is complete before Phase 2 (risk assessment) begins, and that Phase 2 is complete before Phase 3 (risk mitigation) begins. A risk assessment conducted on incomplete information, or a DDS submitted without completed risk assessment, is non-compliant.

2. **System Maintenance and Versioning**: The due diligence system itself must be maintained and updated. The orchestrator must track workflow definitions, agent configurations, quality gate thresholds, and decision rules as versioned artifacts subject to annual review.

3. **Compliance Officer Governance**: All material decisions within the orchestrated workflow must be traceable to a designated compliance officer. The orchestrator must enforce approval gates where human judgment is required by the regulation.

4. **Resource Management**: The orchestrator must manage the allocation of computational and human resources across concurrent due diligence workflows, ensuring that all workflows receive adequate resources to complete within regulatory timelines.

5. **Auditability**: Every orchestration decision, state transition, agent invocation, quality gate evaluation, and checkpoint must be logged in an immutable audit trail that can be presented to competent authorities upon request.

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-001: The agent SHALL orchestrate the three-phase due diligence process (information collection, risk assessment, risk mitigation) in strict sequential order as defined in Article 8(1), preventing out-of-order phase execution.
- REQ-026-002: The agent SHALL implement the due diligence system as a versioned, maintainable workflow definition per Article 8(2), with documented risk management policies, internal controls, and resource allocation.
- REQ-026-003: The agent SHALL enforce annual system review cycles per Article 8(3), triggering workflow definition re-evaluation and update when due or when significant supply chain changes are detected.
- REQ-026-004: The agent SHALL maintain five-year retention for all workflow execution records, agent outputs, quality gate evaluations, and decision logs per Article 8(4).
- REQ-026-005: The agent SHALL enforce compliance officer approval gates at phase transitions and at DDS submission, ensuring human governance of material due diligence decisions.

### 1.3 Article 9 -- Information Requirements (Phase 1 Orchestration)

Article 9 defines the mandatory information that operators must collect as the first step of due diligence. The orchestrator must coordinate all information-gathering agents to collect, validate, and consolidate Article 9 data before permitting the workflow to advance to risk assessment.

**Article 9(1) Required Information Elements:**

| Element | Art. 9(1) Reference | Responsible Upstream Agent(s) | Orchestration Role |
|---------|---------------------|-------------------------------|-------------------|
| Product description, trade name, common name, and (if applicable) full scientific name | (a) | EUDR-009 (Chain of Custody) | Validate product identification completeness |
| Quantity (net mass in kg, volume, or number of items) | (a) | EUDR-011 (Mass Balance Calculator) | Validate quantity reconciliation across supply chain |
| Country of production | (b) | EUDR-001 (Supply Chain Mapping Master) | Validate origin country identification for all supply chain paths |
| Geolocation of all plots of land where relevant commodities were produced | (d) | EUDR-002 (Geolocation Verification), EUDR-006 (Plot Boundary Manager), EUDR-007 (GPS Coordinate Validator) | Validate geolocation completeness and accuracy |
| Date or time range of production | (e) | EUDR-015 (Mobile Data Collector), EUDR-009 (Chain of Custody) | Validate production date relative to EUDR cutoff date (31 December 2020) |
| Verification that relevant products are deforestation-free | (f) | EUDR-003 (Satellite Monitoring), EUDR-004 (Forest Cover Analysis), EUDR-005 (Land Use Change Detector) | Coordinate satellite and land use verification |
| Compliance with relevant legislation of the country of production | (f) | EUDR-023 (Legal Compliance Verifier) | Validate legality determination completeness |
| Supply chain information (names, addresses of suppliers and recipients) | (g), (h) | EUDR-001 (Supply Chain Mapping Master), EUDR-008 (Multi-Tier Supplier Tracker) | Validate supply chain completeness and actor identification |

**Article 9(2) Geolocation Specifics:**

The geolocation requirement in Article 9(1)(d) is subject to specific format rules that the orchestrator must enforce:
- Plots of land exceeding four hectares used for production of relevant commodities other than cattle: geolocation provided as polygons using latitude and longitude coordinates with sufficient points to describe the perimeter
- Plots of land of four hectares or less: geolocation may be provided as a single latitude/longitude point
- For cattle: geolocation of all establishments where the animals were kept

The orchestrator must verify that geolocation data from EUDR-002 (Geolocation Verification), EUDR-006 (Plot Boundary Manager), and EUDR-007 (GPS Coordinate Validator) meets these format requirements before passing the data to the risk assessment phase.

**Article 9(3) Information Sufficiency Test:**

The orchestrator must apply an information sufficiency test before advancing from Phase 1 to Phase 2. Information is sufficient when:
- All mandatory Article 9(1) elements have been collected for every product in the due diligence scope
- Geolocation data meets the format requirements of Article 9(2) for every plot of land
- Production dates have been verified relative to the EUDR cutoff date
- Supply chain information covers all tiers from origin to the operator

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-006: The agent SHALL orchestrate all information-gathering agents (EUDR-001 through EUDR-015) to collect the complete set of Article 9(1) mandatory information elements.
- REQ-026-007: The agent SHALL validate geolocation data format compliance per Article 9(2), enforcing polygon format for plots exceeding four hectares and accepting point format for plots of four hectares or less.
- REQ-026-008: The agent SHALL implement an information sufficiency quality gate that evaluates completeness of all Article 9(1) elements before permitting transition from Phase 1 (information collection) to Phase 2 (risk assessment).
- REQ-026-009: The agent SHALL generate an information gap report when the sufficiency quality gate fails, identifying specific missing or incomplete data elements and the upstream agent(s) responsible for providing them.
- REQ-026-010: The agent SHALL support iterative information collection, allowing the workflow to return to Phase 1 from Phase 2 if the risk assessment identifies additional information needs per Article 10(2).

### 1.4 Article 10 -- Risk Assessment Framework (Phase 2 Orchestration)

Article 10 establishes the risk assessment framework that constitutes the second phase of the due diligence process. The orchestrator must coordinate all risk assessment agents (EUDR-016 through EUDR-025) and ensure that all Article 10(2) criteria are evaluated before a risk determination is made.

**Article 10(1):** Operators shall assess and identify the risk that relevant products intended to be placed on the market or exported are non-compliant with Article 3. The risk assessment shall be adequate and proportionate to the risk of non-compliance, taking into account the criteria set out in paragraph 2.

**Article 10(2) Risk Assessment Criteria -- Agent Mapping:**

The EUDR specifies 14 criteria that must be considered in the risk assessment. The orchestrator maps each criterion to the responsible upstream risk assessment agent(s):

| Criterion | Art. 10(2) Ref | Description | Primary Agent | Supporting Agent(s) |
|-----------|----------------|-------------|---------------|---------------------|
| Country risk classification | (a) | Assigned risk category of the country of production per Article 29 benchmarking | EUDR-016 (Country Risk Evaluator) | -- |
| Forest presence and deforestation | (b) | Presence of forests in the country of production, including prevalence of deforestation or forest degradation | EUDR-020 (Deforestation Alert System) | EUDR-003, EUDR-004, EUDR-005 |
| Commodity source type | (c) | Whether the relevant commodity is sourced from wild forest, plantation, mixed, or other production systems | EUDR-018 (Commodity Risk Analyzer) | EUDR-001 |
| Supply chain complexity | (d) | Complexity of the supply chain, including risk of mixing or substitution; engagement with indigenous peoples | EUDR-008 (Multi-Tier Supplier Tracker) | EUDR-021 (Indigenous Rights Checker) |
| Circumvention or mixing risk | (e) | Risk of circumvention of the regulation, including mixing with products of unknown or non-compliant origin | EUDR-010 (Segregation Verifier) | EUDR-011 (Mass Balance Calculator) |
| Deforestation prevalence | (f) | Prevalence of deforestation and forest degradation in the country, region, or area of production | EUDR-020 (Deforestation Alert System) | EUDR-016 |
| Country concerns | (g) | Concerns related to the country of production or origin, including governance, corruption, armed conflict, sanctions | EUDR-019 (Corruption Index Monitor) | EUDR-016 |
| Financial crime risk | (h) | Risk of money laundering, tax evasion, or other financial crimes linked to commodity trade | EUDR-019 (Corruption Index Monitor) | EUDR-023 (Legal Compliance Verifier) |
| Previous assessment findings | (i) | Findings from previous due diligence assessments on the same products, suppliers, or origins | EUDR-017 (Supplier Risk Scorer) | EUDR-025 (Risk Mitigation Advisor) |
| Complementary information | (j) | Third-party reports, NGO alerts, media reports, substantiated concerns | EUDR-020 (Deforestation Alert System) | EUDR-024 (Third-Party Audit Manager) |
| Stakeholder concerns | (k) | Concerns raised by indigenous peoples, local communities, civil society organizations | EUDR-021 (Indigenous Rights Checker) | EUDR-022 (Protected Area Validator) |
| Certification schemes | (l) | Information from certification or third-party verified schemes relevant to compliance | EUDR-024 (Third-Party Audit Manager) | -- |
| Third-party intelligence | (m) | Information from third parties indicating risks of non-compliance | EUDR-024 (Third-Party Audit Manager) | EUDR-020 |
| Stakeholder consultation | (n) | Consultation with and cooperation with relevant stakeholders | EUDR-021 (Indigenous Rights Checker) | EUDR-022 |

**Orchestration of Parallel Risk Assessment:**

The 14 Article 10(2) criteria are evaluated by 10 risk assessment agents (EUDR-016 through EUDR-025). Many of these agents can operate in parallel because they evaluate independent risk dimensions. However, certain agents have dependencies:

- EUDR-025 (Risk Mitigation Advisor) depends on outputs from EUDR-016 through EUDR-024, and therefore must execute last in the risk assessment phase
- EUDR-020 (Deforestation Alert System) depends on geolocation data validated by EUDR-002, EUDR-006, and EUDR-007 in Phase 1
- EUDR-021 (Indigenous Rights Checker) depends on plot boundary data from EUDR-006

The orchestrator must model these dependencies as a directed acyclic graph (DAG) and execute agents in topological order, maximizing parallelism while respecting data dependencies.

**Article 10(3) -- Risk Determination:**

After all 14 criteria have been evaluated, the orchestrator must synthesize agent outputs into a risk determination:
- **Negligible risk**: The workflow proceeds directly to DDS generation (Phase 3 is simplified to DDS preparation only)
- **Non-negligible risk**: The workflow must proceed to Phase 3 (risk mitigation) before DDS generation is permitted

The orchestrator must document the risk determination decision, including the inputs from all risk assessment agents, the composite risk score, and the regulatory basis for the negligible/non-negligible classification.

**Article 10(4) -- Documentation Requirement:**

The decisions on risk assessment procedures and measures shall be documented, reviewed at least on an annual basis, and made available by the operators to the competent authorities upon request. The orchestrator must ensure that every risk assessment workflow execution produces a complete, timestamped, immutable documentation package.

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-011: The agent SHALL orchestrate all 10 risk assessment agents (EUDR-016 through EUDR-025) to evaluate all 14 Article 10(2) risk assessment criteria.
- REQ-026-012: The agent SHALL model risk assessment agent dependencies as a DAG and execute agents in topological order, maximizing parallel execution while respecting data dependencies.
- REQ-026-013: The agent SHALL implement a risk determination quality gate that synthesizes outputs from all risk assessment agents into a negligible/non-negligible risk classification per Article 10(1).
- REQ-026-014: The agent SHALL route the workflow to Phase 3 (risk mitigation) when non-negligible risk is identified, or directly to DDS generation when risk is negligible.
- REQ-026-015: The agent SHALL generate complete risk assessment documentation packages per Article 10(4), including all agent outputs, composite risk scores, and decision rationale, stored immutably for five years.

### 1.5 Article 10(3) -- Risk Mitigation (Phase 3 Orchestration)

When the risk assessment identifies non-negligible risk, Article 10(3) requires operators to adopt adequate risk mitigation measures before placing the product on the market. The orchestrator manages the risk mitigation phase as follows:

**Article 10(3):** Where the risk assessment under paragraph 1 identifies a non-negligible risk that the relevant products are not compliant with Article 3, the operator or trader shall not place those products on the market or export them unless and until that risk has been adequately mitigated. Risk mitigation measures may include:
- Requiring additional information, data, or documents from suppliers
- Carrying out independent surveys, audits, or other assessments
- Supporting suppliers through capacity building, training, and investment
- Any other appropriate risk mitigation measure

**Orchestration of Risk Mitigation:**

The orchestrator coordinates risk mitigation through EUDR-025 (Risk Mitigation Advisor), which serves as the primary mitigation intelligence agent. The orchestrator's role is to:

1. Pass the complete risk assessment output (all 10 agents) to EUDR-025 for mitigation strategy recommendation
2. Manage the lifecycle of remediation plans generated by EUDR-025
3. Track mitigation measure implementation across upstream agents (e.g., requesting additional information from EUDR-001 Supply Chain Mapping, commissioning satellite verification through EUDR-003, triggering audits through EUDR-024)
4. Re-invoke risk assessment agents after mitigation measures are implemented to verify that residual risk has been reduced to negligible
5. Enforce the market placement prohibition until mitigation is verified

**Iterative Mitigation Loop:**

The orchestrator implements an iterative mitigation-reassessment loop:
1. EUDR-025 recommends mitigation measures based on risk assessment
2. Mitigation measures are implemented (may involve re-invocation of upstream agents)
3. Risk assessment agents are re-invoked to evaluate residual risk
4. If residual risk is negligible: proceed to DDS generation
5. If residual risk remains non-negligible: return to step 1 with updated risk profile
6. Maximum iteration limit prevents infinite loops (configurable, default 5 iterations)

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-016: The agent SHALL coordinate risk mitigation through EUDR-025 (Risk Mitigation Advisor), passing complete risk assessment outputs and managing remediation plan lifecycle.
- REQ-026-017: The agent SHALL implement an iterative mitigation-reassessment loop that re-invokes risk assessment agents after mitigation implementation to verify residual risk reduction.
- REQ-026-018: The agent SHALL enforce the Article 10(3) market placement prohibition by blocking DDS generation until the risk determination quality gate confirms negligible residual risk.
- REQ-026-019: The agent SHALL implement a configurable maximum iteration limit for the mitigation-reassessment loop (default: 5 iterations) to prevent infinite loops, escalating to compliance officer review when the limit is reached.

### 1.6 Article 11 -- Simplified Due Diligence

Article 11, read in conjunction with Article 13, provides for simplified due diligence obligations where the European Commission has classified a country or part thereof as "low risk" under Article 29. The orchestrator must support a simplified workflow variant.

**Article 11(1) / Article 13:** Products sourced exclusively from countries or parts of countries classified as low risk benefit from simplified due diligence. In simplified due diligence:
- Information collection per Article 9 is still required (Phase 1 is not eliminated)
- Risk assessment under Article 10 may be reduced to verifying the low-risk classification and confirming no new information suggests non-compliance
- Risk mitigation under Article 10(3) is not required, provided no non-negligible risk is identified through the simplified assessment
- A DDS must still be submitted

**Orchestration Implications:**

The orchestrator must support two workflow variants:

1. **Standard Due Diligence Workflow**: All three phases (information collection, full risk assessment, risk mitigation if needed), engaging all 25 upstream agents as defined in Sections 1.3, 1.4, and 1.5.

2. **Simplified Due Diligence Workflow**: Phase 1 (information collection) proceeds normally but with potentially reduced agent invocation. Phase 2 invokes EUDR-016 (Country Risk Evaluator) to confirm low-risk classification, and performs a lightweight check against the other 14 Article 10(2) criteria to verify no new information contradicts the simplified pathway. Phase 3 is skipped unless the simplified assessment identifies non-negligible risk.

**Disqualification from Simplified Due Diligence:**

The orchestrator must automatically disqualify products from the simplified pathway when:
- Any component of the product originates from a standard-risk or high-risk country or region
- New information has come to the operator's attention suggesting non-compliance risk
- The Commission has reclassified the sourcing country from low risk to standard or high risk
- A substantiated concern has been submitted regarding the product, supplier, or origin

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-020: The agent SHALL support a simplified due diligence workflow variant for products sourced exclusively from Article 29 low-risk countries, reducing Phase 2 agent invocation while maintaining Phase 1 information collection and DDS submission.
- REQ-026-021: The agent SHALL evaluate simplified due diligence eligibility before workflow initiation, checking country risk classification for all origins in the product's supply chain.
- REQ-026-022: The agent SHALL automatically disqualify products from simplified due diligence and escalate to standard workflow when mixed-origin sourcing, new information, country reclassification, or substantiated concerns are detected.

### 1.7 Article 12 -- Due Diligence Statement (DDS) Generation

Article 12 requires operators to submit a Due Diligence Statement to the EU Information System before placing relevant products on the market. The orchestrator manages DDS generation as the final output of the due diligence workflow.

**Article 12(1):** Operators who, on the basis of the due diligence exercised in accordance with Article 8, conclude that the relevant products comply with Article 3 shall make available a due diligence statement to the competent authorities through the information system.

**Article 12(2) DDS Content (Annex II):**

The DDS must contain the following information, which the orchestrator aggregates from upstream agent outputs:

| DDS Field | Source Agent(s) | Orchestration Validation |
|-----------|----------------|-------------------------|
| Operator name, address, EORI number | System configuration | Verify operator identity and registration |
| Product description, HS/CN codes, common/scientific names | EUDR-009 (Chain of Custody) | Validate product identification against Article 9(1)(a) |
| Product quantity (net mass, volume, items) | EUDR-011 (Mass Balance Calculator) | Validate quantity reconciliation |
| Country of production | EUDR-001 (Supply Chain Mapping Master) | Validate country identification for all origins |
| Geolocation of plots of land | EUDR-002, EUDR-006, EUDR-007 | Validate format compliance per Article 9(2) |
| Production date or time range | EUDR-009, EUDR-015 | Validate date relative to EUDR cutoff |
| Reference number(s) of preceding DDS (if applicable) | System tracking | Validate DDS chain-of-custody |
| Declaration that due diligence was exercised per Article 8 | Orchestrator attestation | Verify all three phases completed |
| Declaration that negligible risk was found or risk was mitigated | Risk determination output | Verify risk quality gate passed |
| Compliance officer name and signature | Approval gate record | Verify compliance officer approval |

**DDS Submission Workflow:**

1. Orchestrator aggregates all required DDS fields from upstream agent outputs
2. DDS content is validated against Annex II requirements
3. Compliance officer is notified for review and digital signature
4. Upon approval, DDS is formatted for EU Information System (TRACES NT) submission
5. DDS reference number is recorded and linked to all underlying due diligence records
6. Five-year retention timer is initiated

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-023: The agent SHALL aggregate DDS content from all upstream agent outputs per Annex II requirements and validate completeness before submission.
- REQ-026-024: The agent SHALL enforce compliance officer review and digital signature as a mandatory gate before DDS submission.
- REQ-026-025: The agent SHALL format the DDS for submission to the EU Information System (TRACES NT) and record the DDS reference number upon successful submission.
- REQ-026-026: The agent SHALL link each DDS to the complete underlying due diligence record (all agent outputs, quality gate evaluations, decision logs, approval records) for five-year retention.

### 1.8 Articles 14-16 -- Monitoring, Substantiated Concerns, and Operator Obligations

Articles 14 through 16 establish the framework for external monitoring and substantiated concerns that may trigger due diligence re-execution.

**Article 14(1):** Any natural or legal person may submit substantiated concerns to competent authorities where they consider that one or more operators or traders are not complying with the regulation. A substantiated concern is a duly reasoned claim based on objective and verifiable information.

**Article 15:** Competent authorities shall examine substantiated concerns within a reasonable period and take appropriate measures, including inspections and audits.

**Article 16:** Monitoring organizations recognized by Member States may submit substantiated concerns and participate in compliance monitoring.

**Orchestration Implications for Substantiated Concerns:**

When a substantiated concern is submitted regarding a product, supplier, or origin that falls within the scope of a completed due diligence workflow, the orchestrator must:

1. Identify all affected due diligence workflows and DDS submissions
2. Re-evaluate whether the substantiated concern constitutes "new information" that invalidates the existing risk assessment
3. If new information is confirmed: initiate a re-execution of the due diligence workflow from Phase 2 (risk assessment) or Phase 1 (information collection) as appropriate
4. If the concern relates to a product already on the market: trigger the Article 21 self-disclosure workflow
5. Preserve the original due diligence record alongside the re-executed workflow for audit trail purposes

**Article 4(7) -- Operator Obligation to Inform:**

When an operator becomes aware or has reason to believe that a relevant product they have placed on the market is non-compliant, they must immediately inform the competent authorities and take necessary corrective measures. The orchestrator must support this self-disclosure workflow as an event-triggered re-execution pathway.

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-027: The agent SHALL support event-triggered re-execution of due diligence workflows when substantiated concerns (Article 14), competent authority requests (Article 15), or new information invalidates an existing risk assessment.
- REQ-026-028: The agent SHALL implement a self-disclosure workflow per Article 4(7) that enables operators to report suspected non-compliance and initiates corrective due diligence re-execution.
- REQ-026-029: The agent SHALL preserve the original due diligence record alongside re-executed workflows, maintaining an immutable audit trail of both the original and corrected assessments.

### 1.9 Articles 22-23 -- Penalty Framework

Articles 22 and 23 establish the penalty framework that creates the economic basis for robust due diligence orchestration.

**Article 22(1):** Member States shall lay down rules on penalties applicable to infringements and shall take all measures necessary to ensure that they are implemented. Penalties shall be effective, proportionate, and dissuasive.

**Article 23 -- Specific Penalties:**

| Penalty | Maximum | Orchestration Relevance |
|---------|---------|------------------------|
| Financial penalties (fines) | Not less than 4% of annual EU-wide turnover | Documented, systematic due diligence reduces penalty exposure |
| Confiscation of products | Full confiscation of non-compliant products and revenue | Complete due diligence prevents non-compliant products from reaching market |
| Temporary market exclusion | Duration determined by Member State | Maintained due diligence system supports reinstatement |
| Public procurement exclusion | Temporary ban from public tenders | Demonstrated due diligence system reduces exclusion risk |
| Public naming | Disclosure of non-compliant operator identity | Transparent, auditable system reduces reputational damage |

**Orchestration as Penalty Defense:**

A complete, auditable, systematically executed due diligence workflow orchestrated by EUDR-026 serves as the primary defense against EUDR penalties. When an operator can demonstrate to competent authorities that:
- A documented due diligence system was in place (Article 8(2))
- All Article 9 information was collected and validated
- All Article 10(2) risk criteria were systematically evaluated
- Risk mitigation measures were implemented when non-negligible risk was identified
- A valid DDS was submitted before market placement
- The system was reviewed annually (Article 8(3))

then the operator has a strong basis for demonstrating compliance even if a particular product is later found to have originated from a deforested area. The "proportionate" penalty standard in Article 22(1) requires competent authorities to consider the quality of the operator's due diligence system when determining penalties.

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-030: The agent SHALL generate compliance defense packages demonstrating systematic due diligence execution, suitable for submission to competent authorities in penalty proceedings.
- REQ-026-031: The agent SHALL calculate penalty exposure estimates based on unmitigated risk factors and incomplete due diligence execution, providing operators with a financial incentive metric for due diligence completion.

### 1.10 Article 29 -- Country Benchmarking and Workflow Differentiation

Article 29 mandates the European Commission to classify countries and parts of countries into three risk categories: low, standard, and high. This classification directly determines the due diligence workflow variant and intensity that the orchestrator must apply.

**Workflow Differentiation by Country Risk:**

| Country Risk | Workflow Variant | Phase 1 Intensity | Phase 2 Intensity | Phase 3 Intensity | CA Check Rate |
|-------------|-----------------|-------------------|-------------------|-------------------|---------------|
| Low | Simplified (Art. 13) | Standard information collection | Lightweight risk check | Skipped unless risk identified | 1% operators; 1% quantity |
| Standard | Standard (Art. 10) | Full information collection | Full 14-criteria assessment | Required if non-negligible risk | 3% operators; 3% quantity |
| High | Enhanced | Full + supplementary information | Full + enhanced verification | Mandatory enhanced mitigation | 9% operators; 9% quantity |

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-032: The agent SHALL select the appropriate workflow variant (simplified, standard, enhanced) based on the Article 29 country benchmarking classification of all origins in the product's supply chain.
- REQ-026-033: The agent SHALL automatically escalate workflow intensity when country classifications change, re-executing affected due diligence workflows under the new classification.

---

## 2. ISO 19011:2018 -- Systematic Audit Approach to Due Diligence

### 2.1 Applicability of ISO 19011 to Due Diligence Orchestration

ISO 19011:2018 provides guidelines for auditing management systems. While the EUDR does not explicitly reference ISO 19011, the standard provides a methodologically rigorous framework for organizing the systematic, evidence-based, repeatable due diligence process that the regulation requires. The Due Diligence Orchestrator Agent adopts ISO 19011 principles to ensure that the due diligence workflow meets the standards expected by competent authority auditors and third-party verifiers.

The EUDR due diligence process is functionally equivalent to an internal compliance audit: it is a systematic, independent, documented process for obtaining evidence and evaluating it objectively to determine compliance with defined criteria (Article 3 requirements). The orchestrator applies ISO 19011 principles to structure this process.

### 2.2 Seven Principles of Auditing Applied to Due Diligence

ISO 19011:2018 Clause 4 defines seven principles of auditing. Each principle maps to a specific orchestration capability:

| # | Principle | ISO 19011 Definition | Due Diligence Orchestration Application |
|---|-----------|---------------------|----------------------------------------|
| 1 | Integrity | The foundation of professionalism; auditors should perform their work ethically, honestly, and responsibly | All due diligence determinations are evidence-based and documented; no undisclosed overrides or manual adjustments to risk scores |
| 2 | Fair Presentation | The obligation to report truthfully and accurately | DDS content accurately reflects the due diligence findings; no selective reporting of favorable results |
| 3 | Due Professional Care | The application of diligence and judgment in auditing | Quality gates enforce thoroughness; no shortcuts that skip required assessments |
| 4 | Confidentiality | Security of information | Supplier data, geolocation, and risk assessments protected per SEC-003 (AES-256-GCM) and SEC-011 (PII Redaction) |
| 5 | Independence | The basis for impartiality and objectivity of conclusions | Agent-based assessment is algorithmically objective; compliance officer review provides independent human judgment |
| 6 | Evidence-Based Approach | The rational method for reaching reliable and reproducible conclusions | Every risk score, quality gate evaluation, and mitigation determination traceable to specific evidence artifacts |
| 7 | Risk-Based Approach | Audit approach that considers risks and opportunities | Workflow intensity proportionate to risk level; enhanced orchestration for high-risk supply chains |

### 2.3 ISO 19011 Audit Process Mapped to Due Diligence Workflow

ISO 19011:2018 Clause 6 defines the audit process. The orchestrator maps each audit process step to the due diligence workflow:

| ISO 19011 Audit Step | Clause | Due Diligence Workflow Equivalent | Orchestrator Function |
|----------------------|--------|----------------------------------|----------------------|
| Initiating the audit | 6.2 | Due diligence workflow initiation for a product/shipment | Workflow trigger, scope definition, agent selection |
| Preparing audit activities | 6.3 | Phase 1 preparation -- identifying information sources and agent configuration | Agent configuration, dependency resolution, resource allocation |
| Conducting audit activities | 6.4 | Phases 1-3 execution -- information collection, risk assessment, risk mitigation | DAG execution, quality gate enforcement, checkpoint management |
| Preparing the audit report | 6.5 | DDS generation and compliance documentation | DDS aggregation, documentation packaging, audit trail compilation |
| Completing the audit | 6.6 | DDS submission and workflow closure | Submission to EU Information System, five-year retention initiation |
| Conducting audit follow-up | 6.7 | Post-DDS monitoring, annual review, substantiated concern response | Ongoing monitoring, re-execution triggers, annual review enforcement |

### 2.4 Competence Requirements (Clause 7)

ISO 19011:2018 Clause 7 defines competence requirements for auditors. In the context of automated due diligence orchestration, competence requirements apply to:

1. **Agent Competence**: Each upstream agent must be validated for its specific assessment domain. The orchestrator must verify agent readiness, data source currency, and model validity before invocation.
2. **System Competence**: The orchestrator must self-validate its workflow definitions, quality gate configurations, and decision rules against regulatory requirements.
3. **Human Competence**: The compliance officer exercising approval gates must have documented competence in EUDR requirements. The orchestrator must track compliance officer qualifications and training.

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-034: The agent SHALL implement ISO 19011:2018 auditing principles (integrity, fair presentation, due professional care, confidentiality, independence, evidence-based approach, risk-based approach) as design constraints on the orchestrated due diligence workflow.
- REQ-026-035: The agent SHALL structure the due diligence workflow to align with the ISO 19011 audit process (initiation, preparation, execution, reporting, completion, follow-up).
- REQ-026-036: The agent SHALL verify agent readiness (data source currency, model validity, configuration correctness) before invocation, per ISO 19011 Clause 7 competence requirements.

---

## 3. Three-Phase Due Diligence Framework

### 3.1 Phase Architecture Overview

The Due Diligence Orchestrator implements a three-phase framework that directly maps to the Article 8(1) requirements. Each phase has defined entry criteria, execution steps, quality gates, and exit criteria.

**Phase Diagram:**

```
[INITIATION] --> [PHASE 1: Information Collection] --> [QG-1: Sufficiency Gate]
                                                              |
                            +------ FAIL (info gaps) <--------+--------> PASS
                            |                                             |
                            v                                             v
                    [Remediation Loop]                  [PHASE 2: Risk Assessment]
                    [Collect missing data]                        |
                            |                                     v
                            +----> [QG-1 retry]         [QG-2: Risk Determination Gate]
                                                              |
                                +---- NON-NEGLIGIBLE <--------+--------> NEGLIGIBLE
                                |                                          |
                                v                                          v
                    [PHASE 3: Risk Mitigation]                   [DDS Generation]
                            |                                          |
                            v                                          v
                    [QG-3: Residual Risk Gate]               [QG-4: DDS Validation]
                            |                                          |
                    +--- PASS ---+--- FAIL                             v
                    |            |   (iterate)               [Compliance Officer Approval]
                    v            v                                     |
              [DDS Generation]  [Mitigation Loop]                      v
                    |                                         [DDS Submission]
                    v                                                  |
              [QG-4: DDS Validation]                                   v
                    |                                         [WORKFLOW COMPLETE]
                    v
              [Compliance Officer Approval]
                    |
                    v
              [DDS Submission]
                    |
                    v
              [WORKFLOW COMPLETE]
```

### 3.2 Phase 1: Information Collection

**Entry Criteria:**
- Due diligence workflow has been initiated with a defined scope (product, supply chain, commodity)
- Operator identity and registration verified
- Compliance officer assigned to the workflow

**Execution Steps:**

| Step | Agent(s) Invoked | Purpose | Parallel/Sequential |
|------|-----------------|---------|-------------------|
| 1.1 | EUDR-001 (Supply Chain Mapping Master) | Map complete multi-tier supply chain graph | Sequential (foundational) |
| 1.2 | EUDR-008 (Multi-Tier Supplier Tracker) | Discover and verify all supply chain tiers | Sequential (depends on 1.1) |
| 1.3 | EUDR-002 (Geolocation Verification) | Validate geolocation data for all plots | Parallel with 1.4-1.7 (after 1.1) |
| 1.4 | EUDR-006 (Plot Boundary Manager) | Manage plot boundary polygons/points | Parallel with 1.3 |
| 1.5 | EUDR-007 (GPS Coordinate Validator) | Validate GPS coordinate accuracy and format | Parallel with 1.3 (depends on 1.4) |
| 1.6 | EUDR-003 (Satellite Monitoring) | Obtain satellite imagery for deforestation verification | Parallel with 1.3-1.5 |
| 1.7 | EUDR-004 (Forest Cover Analysis) | Analyze forest cover at production plots | Parallel (depends on 1.6) |
| 1.8 | EUDR-005 (Land Use Change Detector) | Detect land use changes since cutoff date | Sequential (depends on 1.6, 1.7) |
| 1.9 | EUDR-009 (Chain of Custody) | Establish chain of custody documentation | Parallel with 1.3-1.8 |
| 1.10 | EUDR-010 (Segregation Verifier) | Verify commodity segregation integrity | Sequential (depends on 1.9) |
| 1.11 | EUDR-011 (Mass Balance Calculator) | Calculate and verify mass balance | Sequential (depends on 1.9, 1.10) |
| 1.12 | EUDR-012 (Document Authentication) | Authenticate supply chain documents | Parallel with 1.9-1.11 |
| 1.13 | EUDR-013 (Blockchain Integration) | Record chain-of-custody on blockchain | Sequential (depends on 1.9) |
| 1.14 | EUDR-014 (QR Code Generator) | Generate QR codes for traceability | Sequential (depends on 1.13) |
| 1.15 | EUDR-015 (Mobile Data Collector) | Collect field-level data from producers | Parallel with 1.3-1.8 |

**Quality Gate QG-1: Information Sufficiency**

The information sufficiency quality gate evaluates whether all Article 9(1) mandatory information elements have been collected:

| Check | Criteria | Pass Threshold | Failure Action |
|-------|----------|---------------|----------------|
| Product identification | Description, HS code, scientific name present and validated | 100% completeness | Return to Step 1.9 |
| Quantity verification | Net mass/volume/items reconciled across supply chain | Mass balance within 2% tolerance | Return to Step 1.11 |
| Country identification | Production country identified for all supply chain paths | 100% coverage | Return to Step 1.1 |
| Geolocation completeness | GPS coordinates or polygons for all production plots | 100% of plots geolocated | Return to Steps 1.3-1.5 |
| Geolocation format | Polygons for plots > 4ha; points or polygons for plots <= 4ha | 100% format compliance | Return to Step 1.4 |
| Production date | Date or date range for all production events | 100% coverage; all dates verifiable relative to cutoff | Return to Step 1.15 |
| Deforestation-free evidence | Satellite/field evidence for all plots | 100% of plots covered | Return to Steps 1.6-1.8 |
| Legal compliance data | Relevant legislation compliance documentation | All 8 legal categories covered | Return to Step 1.12 |
| Supply chain completeness | All tiers mapped from origin to operator | 100% path completeness | Return to Steps 1.1-1.2 |

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-037: The agent SHALL execute Phase 1 information collection by invoking EUDR-001 through EUDR-015 in the defined dependency order, maximizing parallel execution where dependencies permit.
- REQ-026-038: The agent SHALL implement Quality Gate QG-1 (Information Sufficiency) evaluating all nine Article 9(1) completeness checks before permitting transition to Phase 2.
- REQ-026-039: The agent SHALL support iterative Phase 1 re-execution when QG-1 fails, routing the workflow back to the specific agent(s) responsible for the missing information.

### 3.3 Phase 2: Risk Assessment

**Entry Criteria:**
- QG-1 (Information Sufficiency) has passed
- All Article 9(1) mandatory information elements are available
- Phase 1 execution record is complete and checksummed

**Execution Steps:**

| Step | Agent(s) Invoked | Art. 10(2) Criteria Covered | Parallel/Sequential |
|------|-----------------|---------------------------|-------------------|
| 2.1 | EUDR-016 (Country Risk Evaluator) | (a) Country risk, (f) Deforestation prevalence | Parallel (independent) |
| 2.2 | EUDR-017 (Supplier Risk Scorer) | (i) Previous findings | Parallel (independent) |
| 2.3 | EUDR-018 (Commodity Risk Analyzer) | (c) Commodity source type | Parallel (independent) |
| 2.4 | EUDR-019 (Corruption Index Monitor) | (g) Country concerns, (h) Financial crime | Parallel (independent) |
| 2.5 | EUDR-020 (Deforestation Alert System) | (b) Forest presence, (f) Deforestation, (j) Complementary info | Parallel (depends on Phase 1 geolocation) |
| 2.6 | EUDR-021 (Indigenous Rights Checker) | (d) Supply chain complexity (indigenous), (k) Stakeholder concerns, (n) Stakeholder consultation | Parallel (depends on Phase 1 plot data) |
| 2.7 | EUDR-022 (Protected Area Validator) | (k) Stakeholder concerns (protected areas) | Parallel (depends on Phase 1 plot data) |
| 2.8 | EUDR-023 (Legal Compliance Verifier) | (h) Financial crime (legal), (d) Complexity | Parallel (independent) |
| 2.9 | EUDR-024 (Third-Party Audit Manager) | (l) Certification, (m) Third-party intelligence, (n) Consultation | Parallel (independent) |
| 2.10 | EUDR-025 (Risk Mitigation Advisor) | Composite risk synthesis | Sequential (depends on 2.1-2.9) |

**Quality Gate QG-2: Risk Determination**

| Check | Criteria | Pass Condition | Failure Condition |
|-------|----------|---------------|-------------------|
| All criteria evaluated | All 14 Article 10(2) criteria have an assessment result | 14/14 criteria assessed | Any criterion missing assessment |
| Composite risk score calculated | Weighted composite score computed from all 10 agents | Score is a valid number 0-100 | Calculation error or missing input |
| Risk classification assigned | Negligible/Non-negligible determination made | Clear classification with documented rationale | Ambiguous or undocumented determination |
| Critical overrides applied | P0 risk triggers evaluated and applied | All override rules checked | Override rule evaluation incomplete |
| Simplified DD eligibility | Simplified pathway eligibility confirmed or denied | Eligibility determination documented | Undetermined eligibility |

**Risk Determination Outcomes:**

| Outcome | Composite Score | Workflow Routing | DDS Content |
|---------|----------------|-----------------|-------------|
| Negligible Risk | 0-10 | Proceed to DDS Generation (skip Phase 3) | "No risk or only a negligible risk was found" |
| Non-Negligible Risk (Low) | 11-25 | Proceed to Phase 3 with monitoring-level mitigation | "Risk was identified and has been adequately mitigated" |
| Non-Negligible Risk (Medium) | 26-50 | Proceed to Phase 3 with standard mitigation | "Risk was identified and has been adequately mitigated" |
| Non-Negligible Risk (High) | 51-75 | Proceed to Phase 3 with enhanced mitigation | "Risk was identified and has been adequately mitigated" |
| Non-Negligible Risk (Critical) | 76-90 | Proceed to Phase 3 with intensive mitigation | "Risk was identified and has been adequately mitigated" |
| Prohibitive Risk | 91-100 | Phase 3 with avoidance recommendation; product blocked | DDS cannot be submitted until risk resolved |

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-040: The agent SHALL execute Phase 2 risk assessment by invoking EUDR-016 through EUDR-025 in the defined dependency order, with Steps 2.1-2.9 executing in parallel and Step 2.10 executing after all prior steps complete.
- REQ-026-041: The agent SHALL implement Quality Gate QG-2 (Risk Determination) verifying that all 14 Article 10(2) criteria have been assessed, the composite risk score is calculated, and a negligible/non-negligible classification is assigned.
- REQ-026-042: The agent SHALL route the workflow based on QG-2 outcome: to DDS generation for negligible risk, or to Phase 3 for non-negligible risk.

### 3.4 Phase 3: Risk Mitigation

**Entry Criteria:**
- QG-2 (Risk Determination) has identified non-negligible risk
- Complete risk assessment output available from all 10 risk agents
- Risk classification and composite score documented

**Execution Steps:**

| Step | Action | Agent(s) | Details |
|------|--------|----------|---------|
| 3.1 | Mitigation strategy recommendation | EUDR-025 | Generate context-specific mitigation recommendations based on risk profile |
| 3.2 | Remediation plan generation | EUDR-025 | Create structured remediation plan with milestones, responsibilities, timelines |
| 3.3 | Compliance officer review | Human gate | Compliance officer reviews and approves remediation plan |
| 3.4 | Mitigation execution | Various (EUDR-001 through EUDR-024 as needed) | Execute mitigation measures (re-invoke upstream agents as needed) |
| 3.5 | Mitigation verification | EUDR-024, EUDR-025 | Verify mitigation implementation and collect evidence |
| 3.6 | Risk re-assessment | EUDR-016 through EUDR-025 | Re-invoke risk assessment agents with post-mitigation data |
| 3.7 | Residual risk evaluation | EUDR-025 | Evaluate whether residual risk is now negligible |

**Quality Gate QG-3: Residual Risk**

| Check | Criteria | Pass Condition |
|-------|----------|---------------|
| Residual risk score | Post-mitigation composite risk score | Score <= 10 (negligible) |
| All P0 triggers resolved | No unresolved P0 (immediate) risk triggers | Zero active P0 triggers |
| Mitigation measures verified | All planned measures confirmed implemented | 100% implementation verified |
| Evidence package complete | Mitigation evidence collected and validated | All evidence artifacts present |
| Compliance officer attestation | CO confirms mitigation adequacy | Signed attestation on record |

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-043: The agent SHALL execute Phase 3 risk mitigation through EUDR-025 with compliance officer approval gates for remediation plans.
- REQ-026-044: The agent SHALL re-invoke risk assessment agents (Steps 2.1-2.10) after mitigation execution to verify residual risk reduction.
- REQ-026-045: The agent SHALL implement Quality Gate QG-3 (Residual Risk) requiring post-mitigation composite risk score <= 10, zero active P0 triggers, 100% mitigation implementation, and compliance officer attestation.

### 3.5 DDS Generation and Submission

**Entry Criteria:**
- Either QG-2 confirmed negligible risk (Phase 3 skipped), or QG-3 confirmed residual risk is negligible after mitigation

**Quality Gate QG-4: DDS Validation**

| Check | Criteria | Pass Condition |
|-------|----------|---------------|
| Annex II completeness | All DDS fields populated per Annex II | 100% field completion |
| Data consistency | DDS data consistent with underlying due diligence records | Zero discrepancies |
| Geolocation format | All geolocation data meets Article 9(2) format requirements | 100% format compliance |
| Risk declaration accuracy | DDS risk declaration matches actual risk determination | Declaration matches workflow outcome |
| Preceding DDS linkage | If applicable, preceding DDS reference numbers valid | All references resolved |
| Operator identification | EORI, name, address verified | Identity confirmed |

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-046: The agent SHALL implement Quality Gate QG-4 (DDS Validation) verifying Annex II completeness, data consistency, geolocation format compliance, risk declaration accuracy, and operator identification before DDS submission.
- REQ-026-047: The agent SHALL generate the complete DDS content by aggregating validated outputs from all upstream agents and formatting for EU Information System submission.

---

## 4. Workflow State Management and Checkpoint Recovery

### 4.1 Workflow State Machine

The orchestrator manages each due diligence workflow as a state machine with the following states:

| State | Description | Transitions |
|-------|-------------|-------------|
| `INITIATED` | Workflow created with defined scope | -> `PHASE1_RUNNING` |
| `PHASE1_RUNNING` | Phase 1 information collection in progress | -> `PHASE1_QG1_EVALUATING`, `PHASE1_FAILED`, `CANCELLED` |
| `PHASE1_QG1_EVALUATING` | Quality Gate QG-1 being evaluated | -> `PHASE2_RUNNING`, `PHASE1_REMEDIATION`, `PHASE1_FAILED` |
| `PHASE1_REMEDIATION` | Information gaps being addressed | -> `PHASE1_QG1_EVALUATING`, `PHASE1_FAILED`, `CANCELLED` |
| `PHASE1_FAILED` | Phase 1 failed after maximum remediation attempts | -> `CANCELLED`, `PHASE1_RUNNING` (manual restart) |
| `PHASE2_RUNNING` | Phase 2 risk assessment in progress | -> `PHASE2_QG2_EVALUATING`, `PHASE2_FAILED`, `CANCELLED` |
| `PHASE2_QG2_EVALUATING` | Quality Gate QG-2 being evaluated | -> `PHASE3_RUNNING`, `DDS_GENERATING`, `PHASE2_FAILED` |
| `PHASE2_FAILED` | Phase 2 failed (agent errors) | -> `CANCELLED`, `PHASE2_RUNNING` (manual restart) |
| `PHASE3_RUNNING` | Phase 3 risk mitigation in progress | -> `PHASE3_QG3_EVALUATING`, `PHASE3_FAILED`, `CANCELLED` |
| `PHASE3_QG3_EVALUATING` | Quality Gate QG-3 being evaluated | -> `DDS_GENERATING`, `PHASE3_ITERATION`, `PHASE3_ESCALATED` |
| `PHASE3_ITERATION` | Mitigation-reassessment loop iteration | -> `PHASE3_RUNNING`, `PHASE3_ESCALATED` |
| `PHASE3_ESCALATED` | Max mitigation iterations reached; compliance officer review | -> `PHASE3_RUNNING`, `DDS_GENERATING`, `CANCELLED` |
| `PHASE3_FAILED` | Phase 3 failed (unrecoverable) | -> `CANCELLED`, `PHASE3_RUNNING` (manual restart) |
| `DDS_GENERATING` | DDS content being aggregated and validated | -> `DDS_QG4_EVALUATING`, `DDS_FAILED` |
| `DDS_QG4_EVALUATING` | Quality Gate QG-4 being evaluated | -> `DDS_AWAITING_APPROVAL`, `DDS_REMEDIATION` |
| `DDS_REMEDIATION` | DDS validation failures being addressed | -> `DDS_QG4_EVALUATING` |
| `DDS_AWAITING_APPROVAL` | Waiting for compliance officer digital signature | -> `DDS_SUBMITTING`, `DDS_REJECTED` |
| `DDS_REJECTED` | Compliance officer rejected DDS | -> `PHASE2_RUNNING`, `PHASE3_RUNNING`, `CANCELLED` |
| `DDS_SUBMITTING` | DDS being submitted to EU Information System | -> `COMPLETED`, `DDS_SUBMISSION_FAILED` |
| `DDS_SUBMISSION_FAILED` | EU Information System submission failed | -> `DDS_SUBMITTING` (retry) |
| `COMPLETED` | Workflow successfully completed; DDS submitted | -> `RE_EXECUTING` (on trigger) |
| `CANCELLED` | Workflow cancelled by operator or system | Terminal state |
| `RE_EXECUTING` | Re-execution triggered by new information or substantiated concern | -> `PHASE1_RUNNING`, `PHASE2_RUNNING` |

### 4.2 Checkpoint and Recovery

The orchestrator implements checkpoint-based recovery for long-running workflows:

**Checkpoint Strategy:**
- A checkpoint is saved after every state transition
- A checkpoint is saved after every agent completes execution
- A checkpoint is saved after every quality gate evaluation
- Checkpoints are stored in PostgreSQL with SHA-256 content hashes

**Checkpoint Content:**

| Field | Description |
|-------|-------------|
| `workflow_id` | Unique workflow identifier |
| `checkpoint_id` | Unique checkpoint identifier (UUID) |
| `checkpoint_timestamp` | ISO 8601 timestamp of checkpoint creation |
| `workflow_state` | Current state machine state |
| `phase` | Current phase (1, 2, 3, DDS) |
| `completed_agents` | List of agents that have completed with their outputs |
| `pending_agents` | List of agents awaiting execution |
| `running_agents` | List of agents currently executing |
| `quality_gate_results` | Results of all evaluated quality gates |
| `agent_outputs` | Serialized outputs from completed agents |
| `iteration_count` | Number of mitigation-reassessment iterations completed |
| `content_hash` | SHA-256 hash of checkpoint content |
| `previous_checkpoint_hash` | Hash chain linking to previous checkpoint |

**Recovery Process:**
1. On workflow restart, load the most recent valid checkpoint
2. Verify checkpoint integrity through hash chain validation
3. Restore workflow state, completed agent outputs, and pending agent list
4. Resume execution from the checkpoint state, re-invoking only agents that were running or pending at checkpoint time
5. Log recovery event in audit trail

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-048: The agent SHALL implement a state machine with all defined states and transitions for due diligence workflow lifecycle management.
- REQ-026-049: The agent SHALL save checkpoints after every state transition, agent completion, and quality gate evaluation, with SHA-256 hash chain integrity verification.
- REQ-026-050: The agent SHALL support workflow recovery from the most recent valid checkpoint, restoring state and resuming execution without re-invoking completed agents.
- REQ-026-051: The agent SHALL log all state transitions, checkpoint creations, and recovery events in the immutable audit trail.

### 4.3 Concurrent Workflow Management

The orchestrator must manage multiple concurrent due diligence workflows for the same operator:

**Concurrency Requirements:**

| Metric | Minimum Capacity |
|--------|-----------------|
| Concurrent active workflows | 500 per operator instance |
| Concurrent agent invocations across workflows | 2,000 |
| Workflow queue depth | 10,000 pending workflows |
| Agent execution pool size | 100 concurrent agent executions |

**Resource Isolation:**
- Each workflow has an isolated execution context preventing cross-contamination
- Agent outputs from one workflow cannot leak into another workflow
- Shared agent resources (e.g., country risk database, satellite imagery cache) use read-only access patterns with per-workflow result isolation

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-052: The agent SHALL support at least 500 concurrent active due diligence workflows per operator instance with full execution context isolation.
- REQ-026-053: The agent SHALL prevent cross-contamination between concurrent workflows, ensuring that agent outputs are isolated per workflow execution context.

---

## 5. Data Collection Orchestration from 25 Upstream Agents

### 5.1 Agent Dependency Graph

The orchestrator manages a dependency graph across all 25 upstream EUDR agents. The graph is organized by phase and dependency order:

**Phase 1 Dependency Graph (Supply Chain Traceability -- EUDR-001 through EUDR-015):**

```
Layer 0 (Foundation):
  EUDR-001 (Supply Chain Mapping Master) -- foundational graph
  EUDR-015 (Mobile Data Collector) -- field data input

Layer 1 (Depends on Layer 0):
  EUDR-008 (Multi-Tier Supplier Tracker) -- depends on EUDR-001
  EUDR-002 (Geolocation Verification) -- depends on EUDR-001 for plot locations
  EUDR-003 (Satellite Monitoring) -- depends on EUDR-001 for plot locations

Layer 2 (Depends on Layer 1):
  EUDR-006 (Plot Boundary Manager) -- depends on EUDR-002
  EUDR-004 (Forest Cover Analysis) -- depends on EUDR-003
  EUDR-009 (Chain of Custody) -- depends on EUDR-008

Layer 3 (Depends on Layer 2):
  EUDR-007 (GPS Coordinate Validator) -- depends on EUDR-006
  EUDR-005 (Land Use Change Detector) -- depends on EUDR-003, EUDR-004
  EUDR-010 (Segregation Verifier) -- depends on EUDR-009
  EUDR-012 (Document Authentication) -- depends on EUDR-009

Layer 4 (Depends on Layer 3):
  EUDR-011 (Mass Balance Calculator) -- depends on EUDR-009, EUDR-010
  EUDR-013 (Blockchain Integration) -- depends on EUDR-009

Layer 5 (Depends on Layer 4):
  EUDR-014 (QR Code Generator) -- depends on EUDR-013
```

**Phase 2 Dependency Graph (Risk Assessment -- EUDR-016 through EUDR-025):**

```
Layer 0 (Independent assessment -- parallel):
  EUDR-016 (Country Risk Evaluator)
  EUDR-017 (Supplier Risk Scorer)
  EUDR-018 (Commodity Risk Analyzer)
  EUDR-019 (Corruption Index Monitor)
  EUDR-020 (Deforestation Alert System) -- depends on Phase 1 geolocation
  EUDR-021 (Indigenous Rights Checker) -- depends on Phase 1 plot data
  EUDR-022 (Protected Area Validator) -- depends on Phase 1 plot data
  EUDR-023 (Legal Compliance Verifier)
  EUDR-024 (Third-Party Audit Manager)

Layer 1 (Synthesis -- sequential after Layer 0):
  EUDR-025 (Risk Mitigation Advisor) -- depends on all Layer 0 agents
```

### 5.2 Agent Invocation Protocol

Each upstream agent is invoked through a standardized protocol:

**Invocation Request:**

| Field | Type | Description |
|-------|------|-------------|
| `workflow_id` | UUID | Parent workflow identifier |
| `agent_id` | string | Target agent identifier (e.g., "GL-EUDR-SCM-001") |
| `invocation_id` | UUID | Unique invocation identifier |
| `phase` | integer | Current workflow phase (1, 2, 3) |
| `input_data` | object | Agent-specific input payload |
| `dependency_outputs` | object | Outputs from upstream dependency agents |
| `timeout_seconds` | integer | Maximum execution time |
| `retry_policy` | object | Retry configuration (max retries, backoff) |
| `checkpoint_enabled` | boolean | Whether to checkpoint agent output |

**Invocation Response:**

| Field | Type | Description |
|-------|------|-------------|
| `invocation_id` | UUID | Matching invocation identifier |
| `agent_id` | string | Agent identifier |
| `status` | enum | SUCCESS, FAILED, TIMEOUT, PARTIAL |
| `output_data` | object | Agent-specific output payload |
| `execution_time_ms` | integer | Actual execution duration |
| `evidence_artifacts` | list | List of evidence document references |
| `content_hash` | string | SHA-256 hash of output_data |
| `provenance` | object | Data lineage and source references |

### 5.3 Agent Output Consolidation

After all agents in a phase complete, the orchestrator consolidates outputs into a phase result:

**Phase 1 Consolidated Output:**
- Complete supply chain graph (from EUDR-001, EUDR-008)
- Validated geolocation dataset (from EUDR-002, EUDR-006, EUDR-007)
- Satellite and land use analysis results (from EUDR-003, EUDR-004, EUDR-005)
- Chain of custody records (from EUDR-009, EUDR-010, EUDR-011)
- Authenticated documents (from EUDR-012)
- Blockchain records (from EUDR-013, EUDR-014)
- Field data collection results (from EUDR-015)
- Information completeness assessment (QG-1 input)

**Phase 2 Consolidated Output:**
- Individual risk scores from all 10 risk agents (0-100 each)
- Composite weighted risk score
- Risk classification (negligible / non-negligible with level)
- Article 10(2) criteria coverage matrix (14 criteria, all assessed)
- Risk mitigation recommendation (from EUDR-025)
- Risk determination documentation package

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-054: The agent SHALL manage the complete agent dependency graph for all 25 upstream agents across Phase 1 and Phase 2, executing agents in topological order with maximum parallelism.
- REQ-026-055: The agent SHALL invoke each upstream agent through the standardized invocation protocol with workflow ID linkage, timeout enforcement, retry policy, and checkpoint support.
- REQ-026-056: The agent SHALL consolidate agent outputs into phase-level results with SHA-256 content hashes and provenance tracking.

---

## 6. Quality Gates and Validation

### 6.1 Quality Gate Framework

The orchestrator implements four quality gates (QG-1 through QG-4) as defined in Section 3. Each quality gate follows a standardized evaluation framework:

**Quality Gate Evaluation Protocol:**

| Step | Action | Description |
|------|--------|-------------|
| 1 | Collect inputs | Gather all required inputs from completed agents |
| 2 | Validate completeness | Verify all required inputs are present |
| 3 | Evaluate checks | Execute each check in the quality gate definition |
| 4 | Calculate gate score | Compute overall pass/fail based on individual check results |
| 5 | Generate report | Create quality gate evaluation report with findings |
| 6 | Record decision | Immutably log the gate decision with rationale |
| 7 | Route workflow | Direct workflow to appropriate next state based on outcome |

### 6.2 Cross-Agent Data Quality Checks

Beyond the phase-specific quality gates, the orchestrator performs cross-agent data quality checks to ensure consistency across agent outputs:

| Check | Agents Involved | Validation |
|-------|----------------|-----------|
| Geolocation consistency | EUDR-001, EUDR-002, EUDR-006, EUDR-007 | Plot coordinates referenced by all agents match within tolerance |
| Supply chain graph integrity | EUDR-001, EUDR-008, EUDR-009 | All supply chain actors referenced consistently across agents |
| Quantity reconciliation | EUDR-009, EUDR-010, EUDR-011 | Mass balance across chain-of-custody, segregation, and calculator |
| Temporal consistency | EUDR-003, EUDR-005, EUDR-009, EUDR-015 | Production dates, satellite imagery dates, and custody transfer dates are chronologically consistent |
| Document cross-reference | EUDR-012, EUDR-009, EUDR-023 | Documents authenticated by EUDR-012 are the same referenced by custody and legal agents |
| Risk score reconciliation | EUDR-016 through EUDR-025 | Individual risk scores sum consistently with composite; no contradictions in risk factors |
| Commodity classification | EUDR-001, EUDR-018 | Commodity type consistent between supply chain mapping and commodity risk analysis |

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-057: The agent SHALL execute cross-agent data quality checks at each phase boundary, verifying consistency of geolocation, supply chain graph, quantity, temporal data, documents, risk scores, and commodity classification across all contributing agents.
- REQ-026-058: The agent SHALL block phase transitions when critical cross-agent inconsistencies are detected, routing the workflow to remediation until inconsistencies are resolved.

---

## 7. Parallel vs. Sequential Execution Patterns

### 7.1 Execution Strategy

The orchestrator employs a hybrid execution strategy that maximizes throughput while respecting data dependencies:

**Parallel Execution Rules:**
1. Agents with no mutual dependencies execute in parallel within the same layer
2. Maximum parallel agent count is configurable (default: 10 per workflow)
3. Resource-intensive agents (satellite analysis, mass balance) may be throttled to prevent resource exhaustion
4. Parallel agent results are collected with a configurable timeout (default: agent-specific)

**Sequential Execution Rules:**
1. Agents with explicit dependencies execute sequentially after dependencies complete
2. Quality gates execute sequentially as workflow checkpoints
3. Compliance officer approval gates are sequential blocking points
4. Phase transitions are sequential (Phase 1 must complete before Phase 2 begins)

### 7.2 Execution Timing Estimates

| Phase | Agent Count | Layers | Estimated Duration (Standard) | Estimated Duration (Enhanced) |
|-------|-------------|--------|------------------------------|-------------------------------|
| Phase 1 | 15 agents | 6 layers | 60-120 seconds | 120-300 seconds |
| QG-1 | Quality gate | 1 | 5-10 seconds | 5-10 seconds |
| Phase 2 | 10 agents | 2 layers | 30-60 seconds | 60-120 seconds |
| QG-2 | Quality gate | 1 | 5-10 seconds | 5-10 seconds |
| Phase 3 | Variable | Variable | 0-120 seconds (per iteration) | 0-300 seconds (per iteration) |
| QG-3 | Quality gate | 1 | 5-10 seconds | 5-10 seconds |
| DDS Generation | Aggregation | 1 | 10-20 seconds | 10-20 seconds |
| QG-4 | Quality gate | 1 | 5-10 seconds | 5-10 seconds |
| **Total (automated)** | -- | -- | **120-250 seconds** | **210-480 seconds** |

Note: Compliance officer approval gates are human-mediated and not included in automated duration estimates. The < 5 minute performance target refers to the automated execution time excluding human approval gates.

### 7.3 Timeout and Cancellation

| Timeout Type | Default | Configurable | Action on Timeout |
|-------------|---------|-------------|-------------------|
| Individual agent timeout | Agent-specific (30-300s) | Yes | Retry per retry policy; then fail agent |
| Phase timeout | Phase 1: 600s; Phase 2: 300s; Phase 3: 600s | Yes | Fail phase; checkpoint state |
| Workflow timeout | 3600s (1 hour automated; 7 days including human gates) | Yes | Cancel workflow; preserve checkpoint |
| Quality gate timeout | 30s | Yes | Fail gate; log timeout |

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-059: The agent SHALL execute the complete automated due diligence workflow (excluding human approval gates) within 5 minutes for standard-complexity workflows (single commodity, single origin, Tier 3 supply chain).
- REQ-026-060: The agent SHALL support configurable timeouts at agent, phase, quality gate, and workflow levels with appropriate failure handling and checkpoint preservation on timeout.

---

## 8. Error Handling and Remediation Workflows

### 8.1 Error Classification

The orchestrator classifies errors into four severity categories:

| Severity | Description | Examples | Response |
|----------|-------------|----------|----------|
| **Transient** | Temporary failures that may resolve on retry | Network timeout, database connection pool exhaustion, rate limiting | Automatic retry with exponential backoff |
| **Degraded** | Agent returned partial results or operated in degraded mode | Satellite imagery partial coverage, stale reference data, cached risk scores | Accept with degradation flag; schedule re-assessment |
| **Agent Failure** | Agent returned an error or could not complete | Agent code error, invalid input data, unrecoverable dependency failure | Skip agent with failure record; evaluate impact on quality gate |
| **Critical** | Fundamental failure preventing workflow progression | Database corruption, authentication failure, regulatory data source unavailable | Halt workflow; alert operations; preserve checkpoint |

### 8.2 Retry Strategy

The orchestrator implements a tiered retry strategy:

**Retry Configuration:**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `max_retries` | 3 | 0-10 | Maximum retry attempts per agent invocation |
| `initial_delay_ms` | 1000 | 100-30000 | Delay before first retry |
| `backoff_multiplier` | 2.0 | 1.0-5.0 | Exponential backoff multiplier |
| `max_delay_ms` | 30000 | 1000-300000 | Maximum delay between retries |
| `retry_on_timeout` | true | boolean | Whether to retry on agent timeout |
| `retry_on_error` | true | boolean | Whether to retry on agent error |

**Retry Decision Tree:**
1. Is the error transient? -> Retry with backoff
2. Is the error degraded? -> Accept partial result with flag
3. Has max_retries been reached? -> Fail agent; evaluate workflow impact
4. Is the agent critical for the current quality gate? -> Block progression; alert compliance officer
5. Is the agent optional for the current quality gate? -> Continue without agent; record gap

### 8.3 Degraded Mode Operation

When one or more agents fail or return degraded results, the orchestrator can continue in degraded mode under specific conditions:

**Degraded Mode Rules:**

| Condition | Degraded Mode Permitted | Quality Gate Impact |
|-----------|------------------------|-------------------|
| Non-critical Phase 1 agent failure (e.g., EUDR-014 QR Code) | Yes | QG-1 evaluates without QR code data; no impact on information sufficiency |
| Critical Phase 1 agent failure (e.g., EUDR-002 Geolocation) | No | Workflow blocked; geolocation is mandatory under Article 9(1)(d) |
| Non-critical Phase 2 agent failure (e.g., EUDR-019 Corruption) | Yes, with elevated risk | QG-2 applies precautionary principle: missing agent score treated as elevated risk |
| Critical Phase 2 agent failure (e.g., EUDR-020 Deforestation) | No | Workflow blocked; deforestation assessment is mandatory |
| Phase 3 EUDR-025 failure | No | Mitigation cannot proceed without advisor; workflow blocked |

**Precautionary Principle in Degraded Mode:**

When a risk assessment agent fails in Phase 2, the orchestrator applies the precautionary principle: the missing risk dimension is treated as elevated risk (default score: 75/100) rather than zero risk. This ensures that agent failures cannot inadvertently reduce the composite risk score and allow non-compliant products to reach the market.

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-061: The agent SHALL classify errors into four severity categories (transient, degraded, agent failure, critical) and apply appropriate response strategies.
- REQ-026-062: The agent SHALL implement exponential backoff retry with configurable parameters for transient errors.
- REQ-026-063: The agent SHALL apply the precautionary principle when risk assessment agents fail, treating missing risk dimensions as elevated risk (default: 75/100) rather than zero risk.
- REQ-026-064: The agent SHALL block workflow progression when critical agents (geolocation, deforestation, legal compliance) fail, preventing DDS generation without mandatory assessments.

---

## 9. Audit Trail and Provenance Tracking

### 9.1 Audit Trail Requirements

The orchestrator maintains a comprehensive, immutable audit trail for every due diligence workflow execution. This audit trail serves two regulatory purposes:

1. **Article 4(7) Record Retention**: Operators must keep records of due diligence for five years from DDS submission
2. **Articles 15-16 Competent Authority Inspection**: Competent authorities may request complete due diligence records at any time during the retention period

**Audit Trail Granularity:**

Every significant event in the workflow execution is recorded as an audit entry:

| Event Category | Event Types | Data Captured |
|----------------|------------|---------------|
| Workflow lifecycle | Created, started, paused, resumed, completed, cancelled | Workflow ID, timestamp, initiator, reason |
| State transitions | Every state change in the state machine | Source state, target state, trigger, timestamp |
| Agent invocations | Agent invoked, completed, failed, retried | Agent ID, invocation ID, input hash, output hash, duration |
| Quality gate evaluations | Gate evaluated, passed, failed | Gate ID, all check results, overall decision, rationale |
| Checkpoint operations | Created, restored, verified | Checkpoint ID, content hash, hash chain |
| Data access | Agent input data assembled from dependencies | Data source, access timestamp, data hash |
| Human decisions | Compliance officer approvals, rejections, overrides | Decision maker identity, decision, rationale, timestamp |
| Error events | Errors, retries, degraded mode entries | Error type, severity, retry count, resolution |
| DDS operations | Generated, validated, approved, submitted | DDS content hash, reference number, submission timestamp |

### 9.2 Provenance Chain

Each agent output carries a provenance chain that traces the data back to its original sources:

```
DDS Field (e.g., "Country of Production: Brazil")
  <- Orchestrator aggregation (Phase 1 consolidated output)
    <- EUDR-001 (Supply Chain Mapping Master) output
      <- Supplier declaration (Document ID: DOC-2026-001234)
        <- EUDR-012 (Document Authentication) verification
      <- ERP import (Source: SAP S/4HANA, Record ID: PO-2026-5678)
        <- AGENT-DATA-003 (ERP/Finance Connector) extraction
```

Each link in the provenance chain includes:
- Source identifier (agent ID, document ID, system ID)
- Timestamp of data creation or extraction
- Content hash (SHA-256) for integrity verification
- Confidence level (verified, declared, inferred)
- Regulatory basis (which Article 9(1) requirement the data satisfies)

### 9.3 Immutability and Tamper Detection

**Immutability Controls:**
- All audit trail entries are append-only (no deletion, no modification)
- Each entry includes a SHA-256 content hash
- Entries are linked in a hash chain (each entry references the previous entry's hash)
- Hash chain integrity is verifiable at any point
- Digital signatures on compliance officer decisions
- Integration with SEC-005 (Centralized Audit Logging) for platform-wide audit trail

**Tamper Detection:**
- Periodic hash chain verification (every 24 hours)
- Random spot-check verification on audit trail access
- Alert on any hash chain break or integrity failure
- Backup copies stored in separate storage with independent hash verification

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-065: The agent SHALL maintain an immutable, append-only audit trail for every workflow execution event, capturing all event categories defined in Section 9.1.
- REQ-026-066: The agent SHALL implement SHA-256 hash chain provenance for all agent outputs and DDS fields, traceable from the DDS back to original source documents.
- REQ-026-067: The agent SHALL integrate with SEC-005 (Centralized Audit Logging) for platform-wide audit trail consistency.
- REQ-026-068: The agent SHALL implement tamper detection through periodic hash chain verification with alerts on integrity failures.

---

## 10. Competent Authority Inspection Readiness

### 10.1 Inspection Scenarios

Competent authorities in each EU Member State conduct risk-based inspections to verify operator compliance. The orchestrator must ensure that operators can respond to inspections with complete, organized, verifiable evidence packages.

**Inspection Rate Requirements (Article 29):**

| Country Risk | Minimum Operator Check Rate | Minimum Quantity Check Rate |
|-------------|---------------------------|---------------------------|
| Low risk | 1% of operators annually | 1% of quantity annually |
| Standard risk | 3% of operators annually | 3% of quantity annually |
| High risk | 9% of operators annually | 9% of quantity annually |

**Inspection Types:**

| Type | Trigger | Scope | Response Time |
|------|---------|-------|---------------|
| Routine inspection | Random risk-based selection | Full due diligence system review | 10-30 business days |
| Targeted inspection | Substantiated concern (Article 14) | Specific product/supplier/origin review | 5-15 business days |
| Customs verification | Product at EU border | Specific shipment DDS verification | 24-72 hours |
| Emergency inspection | Confirmed non-compliance alert | Immediate system access and evidence | Same day |

### 10.2 Evidence Package Generation

The orchestrator can generate inspection-ready evidence packages on demand:

**Standard Evidence Package:**

| Section | Content | Source |
|---------|---------|--------|
| 1. DDS copies | All DDS submissions within inspection scope | DDS repository |
| 2. Due diligence system description | Workflow definitions, quality gate configurations, agent descriptions | System configuration |
| 3. Information collection records | Phase 1 consolidated outputs with provenance | Workflow execution records |
| 4. Risk assessment records | Phase 2 consolidated outputs with all agent scores and criteria mapping | Workflow execution records |
| 5. Risk mitigation records | Phase 3 remediation plans, implementation evidence, effectiveness assessments | Workflow execution records |
| 6. Quality gate evaluation logs | All QG-1 through QG-4 evaluation records | Quality gate records |
| 7. Compliance officer decisions | Approval records, override justifications, attestations | Human decision records |
| 8. Audit trail extract | Complete event log for workflows within scope | Audit trail repository |
| 9. Supply chain documentation | Supply chain maps, supplier registrations, custody chain | EUDR-001, EUDR-008, EUDR-009 outputs |
| 10. Geolocation evidence | Plot coordinates, boundary polygons, satellite imagery references | EUDR-002, EUDR-006, EUDR-007 outputs |
| 11. Annual review records | System review outcomes, configuration changes, policy updates | Annual review records |
| 12. Training and competence records | Compliance officer training, system user training | HR/training system |

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-069: The agent SHALL generate competent authority evidence packages containing all 12 sections within 24 hours of request for routine inspections and within 4 hours for emergency inspections.
- REQ-026-070: The agent SHALL maintain continuous inspection readiness by ensuring all workflow execution records are indexed, searchable, and exportable at all times.
- REQ-026-071: The agent SHALL support evidence package export in formats acceptable to competent authorities, including PDF, JSON, XML, and structured archive formats.

---

## 11. Integration Patterns for Upstream Agents

### 11.1 Supply Chain Traceability Agents (EUDR-001 through EUDR-015)

The 15 Supply Chain Traceability agents provide the data foundation for Phase 1 (information collection). The orchestrator integrates with these agents through the following patterns:

**Data Flow Pattern: Fan-Out with Dependency Resolution**

The orchestrator fans out execution across all 15 agents according to the dependency graph (Section 5.1), collecting results as each agent completes. Dependencies are resolved by passing upstream agent outputs as inputs to downstream agents.

| Agent | Key Output for Orchestrator | Quality Gate Contribution |
|-------|---------------------------|--------------------------|
| EUDR-001 | Supply chain graph (nodes, edges, actor metadata) | QG-1: Supply chain completeness |
| EUDR-002 | Verified geolocation data (coordinates, accuracy scores) | QG-1: Geolocation completeness |
| EUDR-003 | Satellite imagery analysis results (deforestation indicators) | QG-1: Deforestation-free evidence |
| EUDR-004 | Forest cover statistics (before/after cutoff date) | QG-1: Deforestation-free evidence |
| EUDR-005 | Land use change classification (forest, agriculture, urban) | QG-1: Deforestation-free evidence |
| EUDR-006 | Plot boundary polygons (GeoJSON format) | QG-1: Geolocation format compliance |
| EUDR-007 | GPS coordinate validation results (accuracy, consistency) | QG-1: Geolocation completeness |
| EUDR-008 | Multi-tier supplier registry (all tiers mapped) | QG-1: Supply chain completeness |
| EUDR-009 | Chain of custody records (custody transfers documented) | QG-1: Supply chain completeness |
| EUDR-010 | Segregation verification results (mixing risk assessment) | QG-1: Quantity verification |
| EUDR-011 | Mass balance calculation (input/output reconciliation) | QG-1: Quantity verification |
| EUDR-012 | Document authentication results (verified, suspect, rejected) | QG-1: Legal compliance data |
| EUDR-013 | Blockchain transaction records (immutable custody chain) | QG-1: Supply chain completeness |
| EUDR-014 | QR code assignment records (product-level traceability) | Non-critical for QG-1 |
| EUDR-015 | Field data collection results (producer-level data) | QG-1: Production date, geolocation |

### 11.2 Risk Assessment Agents (EUDR-016 through EUDR-025)

The 10 Risk Assessment agents provide the analytical foundation for Phase 2 (risk assessment). The orchestrator integrates with these agents through the following pattern:

**Data Flow Pattern: Parallel Assessment with Fan-In Synthesis**

All 9 independent risk agents (EUDR-016 through EUDR-024) execute in parallel, each consuming relevant Phase 1 outputs. After all 9 complete, EUDR-025 (Risk Mitigation Advisor) executes with all 9 outputs as input.

| Agent | Risk Dimension | Score Range | Art. 10(2) Criteria |
|-------|---------------|-------------|---------------------|
| EUDR-016 | Country risk (deforestation, governance) | 0-100 | (a), (f), (g) |
| EUDR-017 | Supplier risk (compliance history, capacity) | 0-100 | (i) |
| EUDR-018 | Commodity risk (deforestation correlation, source type) | 0-100 | (c) |
| EUDR-019 | Corruption risk (governance, financial crime) | 0-100 | (g), (h) |
| EUDR-020 | Deforestation risk (active alerts, forest presence) | 0-100 | (b), (f), (j) |
| EUDR-021 | Indigenous rights risk (territory overlap, FPIC status) | 0-100 | (d), (k), (n) |
| EUDR-022 | Protected area risk (boundary overlap, encroachment) | 0-100 | (k) |
| EUDR-023 | Legal compliance risk (8 legal categories) | 0-100 | (d), (h) |
| EUDR-024 | Audit risk (findings, certifications, third-party intel) | 0-100 | (l), (m), (n) |
| EUDR-025 | Composite synthesis and mitigation recommendation | 0-100 | All 14 criteria |

**Composite Risk Score Calculation:**

The orchestrator delegates composite risk calculation to EUDR-025 (Risk Mitigation Advisor), which applies the weighted scoring model defined in REQS-AGENT-EUDR-025 Section 3.1 (20% country + 15% supplier + 10% commodity + 10% corruption + 15% deforestation + 8% indigenous + 7% protected area + 10% legal + 5% audit) with critical override rules.

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-072: The agent SHALL integrate with all 15 Supply Chain Traceability agents (EUDR-001 through EUDR-015) using the fan-out dependency resolution pattern for Phase 1 data collection.
- REQ-026-073: The agent SHALL integrate with all 10 Risk Assessment agents (EUDR-016 through EUDR-025) using the parallel assessment with fan-in synthesis pattern for Phase 2 risk evaluation.
- REQ-026-074: The agent SHALL pass complete Phase 1 consolidated output as input to Phase 2 risk assessment agents, ensuring all agents have access to the full information base.

### 11.3 Event Bus Integration

The orchestrator publishes and subscribes to events via the GreenLang event bus for real-time coordination:

**Published Events:**

| Event | Description | Trigger |
|-------|-------------|---------|
| `eudr.dd.workflow_initiated` | New due diligence workflow started | Operator initiates DD for a product |
| `eudr.dd.phase1_completed` | Phase 1 information collection completed | QG-1 passed |
| `eudr.dd.phase2_completed` | Phase 2 risk assessment completed | QG-2 evaluated |
| `eudr.dd.phase3_completed` | Phase 3 risk mitigation completed | QG-3 passed |
| `eudr.dd.dds_generated` | DDS content generated and validated | QG-4 passed |
| `eudr.dd.dds_submitted` | DDS submitted to EU Information System | Submission confirmed |
| `eudr.dd.workflow_completed` | Full workflow completed successfully | DDS submission confirmed |
| `eudr.dd.workflow_failed` | Workflow failed (unrecoverable) | Critical error or cancellation |
| `eudr.dd.quality_gate_failed` | Quality gate evaluation failed | QG-1, QG-2, QG-3, or QG-4 failed |
| `eudr.dd.agent_failed` | Upstream agent execution failed | Agent error after retry exhaustion |
| `eudr.dd.re_execution_triggered` | Workflow re-execution initiated | New information, substantiated concern |

**Subscribed Events:**

| Event | Source | Response |
|-------|--------|----------|
| `eudr.risk.country_updated` | EUDR-016 | Re-evaluate affected workflows; escalate if needed |
| `eudr.alert.deforestation_detected` | EUDR-020 | Trigger re-assessment for affected products |
| `eudr.rights.violation_detected` | EUDR-021 | Trigger re-assessment for affected products |
| `eudr.area.encroachment_detected` | EUDR-022 | Trigger re-assessment for affected products |
| `eudr.legal.compliance_failed` | EUDR-023 | Trigger re-assessment for affected products |
| `eudr.audit.completed` | EUDR-024 | Incorporate findings into active workflows |
| `eudr.mitigation.product_cleared` | EUDR-025 | Advance affected workflows to DDS generation |
| `eudr.mitigation.product_blocked` | EUDR-025 | Hold affected workflows in Phase 3 |

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-075: The agent SHALL publish all defined workflow lifecycle events via the GreenLang event bus with guaranteed message delivery (99.9% SLA).
- REQ-026-076: The agent SHALL subscribe to all defined external events and trigger appropriate workflow actions (re-assessment, re-execution, hold) within the SLA defined for each event priority.

---

## 12. Data Retention and Immutability Requirements

### 12.1 Retention Schedule

| Record Type | Minimum Retention | Regulatory Basis | Storage |
|-------------|-------------------|------------------|---------|
| Workflow execution records | 5 years from DDS submission date | EUDR Art. 4(7) | Encrypted at rest (AES-256-GCM) |
| Agent invocation records | 5 years from invocation date | EUDR Art. 4(7) | Encrypted at rest (AES-256-GCM) |
| Quality gate evaluation records | 5 years from evaluation date | EUDR Art. 4(7), Art. 10(4) | Encrypted at rest (AES-256-GCM) |
| Checkpoint data | 5 years from workflow completion | EUDR Art. 4(7) | Encrypted at rest (AES-256-GCM) |
| Audit trail entries | 5 years from entry date | EUDR Art. 4(7), Art. 10(4) | Encrypted at rest (AES-256-GCM) |
| DDS content and submission records | 5 years from DDS submission date | EUDR Art. 4(7), Art. 12 | Encrypted at rest (AES-256-GCM) |
| Compliance officer decision records | 5 years from decision date | EUDR Art. 8(2), Art. 10(4) | Encrypted at rest (AES-256-GCM) |
| Evidence packages | 5 years from package generation date | EUDR Art. 4(7) | Encrypted at rest (AES-256-GCM) |
| Workflow definition versions | 5 years from version retirement | EUDR Art. 8(3) | Encrypted at rest (AES-256-GCM) |

### 12.2 Immutability Controls

All orchestration records must be immutable once finalized:
- SHA-256 content hashes on all workflow records, agent outputs, and quality gate evaluations
- Append-only audit trail for all record modifications (no deletion, no overwrite)
- Hash chain linking consecutive records for tamper detection
- Digital signatures on compliance officer decisions
- Integration with SEC-005 (Centralized Audit Logging) for all orchestrator activities
- Integration with SEC-003 (Encryption at Rest) for data protection
- Provenance tracking linking every DDS field to its source data through the complete agent chain

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-077: The agent SHALL enforce minimum 5-year retention for all orchestration records per EUDR Article 4(7).
- REQ-026-078: The agent SHALL implement immutable record storage with SHA-256 hash chains, digital signatures, and tamper detection for all workflow records.

---

## 13. Performance and Scalability Requirements

### 13.1 System Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| Standard DD workflow (automated, single commodity, single origin) | < 5 minutes end-to-end | p99 latency (excluding human approval gates) |
| Complex DD workflow (multi-commodity, multi-origin, Tier 5+ supply chain) | < 15 minutes end-to-end | p99 latency (excluding human approval gates) |
| Simplified DD workflow (low-risk country, single origin) | < 2 minutes end-to-end | p99 latency (excluding human approval gates) |
| Quality gate evaluation | < 10 seconds per gate | p99 latency |
| Agent invocation overhead | < 500ms per invocation (orchestrator overhead only) | p99 latency |
| Checkpoint save/restore | < 2 seconds per checkpoint | p99 latency |
| Evidence package generation (routine) | < 24 hours | SLA compliance |
| Evidence package generation (emergency) | < 4 hours | SLA compliance |
| DDS generation | < 30 seconds | p99 latency |
| Audit trail query | < 5 seconds for single workflow | p99 latency |
| Concurrent active workflows | 500+ per operator instance | Capacity test |
| API availability | 99.9% uptime | Monthly SLA |

### 13.2 Zero-Hallucination Guarantee

Consistent with the GreenLang platform standard:

- All workflow routing decisions are deterministic and rule-based
- Quality gate evaluations are deterministic (same inputs always produce same pass/fail)
- No LLM in the critical path for workflow control, quality gate evaluation, or DDS generation
- SHA-256 provenance hashes on all workflow records and agent outputs
- Bit-perfect reproducibility: replaying a workflow with the same inputs produces identical outputs
- All regulatory references are traceable to specific EUDR articles, clauses, and provisions
- Risk determination (negligible vs. non-negligible) is rule-based with documented thresholds

### 13.3 Scalability

| Dimension | Minimum | Target | Maximum |
|-----------|---------|--------|---------|
| Concurrent workflows per instance | 100 | 500 | 2,000 |
| Total workflow executions per day | 1,000 | 5,000 | 50,000 |
| Agent invocations per second | 50 | 200 | 1,000 |
| Checkpoint storage per workflow | 1 MB | 10 MB | 100 MB |
| Audit trail entries per workflow | 100 | 500 | 5,000 |
| Upstream agents managed | 25 | 25 | 50 (future expansion) |

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-079: The agent SHALL meet all performance targets specified in Section 13.1, with the standard due diligence workflow completing within 5 minutes excluding human approval gates.
- REQ-026-080: The agent SHALL implement zero-hallucination deterministic processing for all workflow control, quality gate evaluation, DDS generation, and risk determination operations.
- REQ-026-081: The agent SHALL scale to support 500+ concurrent workflows per operator instance with linear performance degradation under load.

---

## 14. Security and Access Control

### 14.1 Authentication and Authorization

The orchestrator integrates with the GreenLang security infrastructure:

| Security Component | Integration | Purpose |
|-------------------|-------------|---------|
| SEC-001 (JWT Authentication) | JWT RS256 token validation | Authenticate all API requests to the orchestrator |
| SEC-002 (RBAC Authorization) | Role-based permission checks | Enforce role-based access to workflow operations |
| SEC-003 (Encryption at Rest) | AES-256-GCM encryption | Protect all stored workflow records |
| SEC-004 (TLS 1.3) | Transport encryption | Protect all data in transit |
| SEC-005 (Centralized Audit Logging) | Audit event publishing | Log all orchestrator activities |
| SEC-006 (Secrets Management) | Vault integration | Manage agent credentials and API keys |
| SEC-011 (PII Detection/Redaction) | PII scanning | Detect and redact PII in supplier data |

### 14.2 RBAC Permissions

| Permission | Description | Roles |
|-----------|-------------|-------|
| `eudr.dd.workflow.create` | Initiate a new due diligence workflow | compliance_officer, compliance_manager, admin |
| `eudr.dd.workflow.read` | View workflow status and records | compliance_officer, compliance_manager, auditor, admin |
| `eudr.dd.workflow.cancel` | Cancel an active workflow | compliance_manager, admin |
| `eudr.dd.dds.approve` | Approve DDS for submission (compliance officer gate) | compliance_officer, compliance_manager |
| `eudr.dd.dds.submit` | Submit DDS to EU Information System | compliance_manager, admin |
| `eudr.dd.evidence.export` | Export evidence packages for competent authorities | compliance_officer, compliance_manager, auditor, admin |
| `eudr.dd.config.manage` | Manage workflow definitions, quality gate thresholds | compliance_manager, admin |
| `eudr.dd.audit.read` | Read audit trail records | auditor, compliance_manager, admin |
| `eudr.dd.reexecute` | Trigger workflow re-execution | compliance_officer, compliance_manager, admin |

**Regulatory Requirement for Agent EUDR-026:**
- REQ-026-082: The agent SHALL integrate with SEC-001 through SEC-006 and SEC-011 for authentication, authorization, encryption, audit logging, secrets management, and PII protection.
- REQ-026-083: The agent SHALL enforce RBAC permissions for all workflow operations, ensuring that only authorized roles can initiate, approve, submit, and manage due diligence workflows.

---

## 15. Requirements Traceability Matrix

| Requirement ID | Description | Regulatory Source | Priority |
|----------------|-------------|-------------------|----------|
| REQ-026-001 | Three-phase sequential orchestration | EUDR Art. 8(1) | P0 |
| REQ-026-002 | Versioned, maintainable workflow definition | EUDR Art. 8(2) | P0 |
| REQ-026-003 | Annual system review enforcement | EUDR Art. 8(3) | P0 |
| REQ-026-004 | Five-year retention for workflow records | EUDR Art. 8(4) | P0 |
| REQ-026-005 | Compliance officer approval gates | EUDR Art. 8(2) | P0 |
| REQ-026-006 | Phase 1 information collection orchestration | EUDR Art. 9(1) | P0 |
| REQ-026-007 | Geolocation format validation | EUDR Art. 9(2) | P0 |
| REQ-026-008 | QG-1 information sufficiency gate | EUDR Art. 9(1) | P0 |
| REQ-026-009 | Information gap reporting | EUDR Art. 9(1) | P0 |
| REQ-026-010 | Iterative information collection | EUDR Art. 10(2) | P1 |
| REQ-026-011 | Phase 2 risk assessment orchestration (all 14 criteria) | EUDR Art. 10(2) | P0 |
| REQ-026-012 | DAG-based parallel risk assessment execution | EUDR Art. 10(1) | P0 |
| REQ-026-013 | QG-2 risk determination gate | EUDR Art. 10(1) | P0 |
| REQ-026-014 | Workflow routing based on risk determination | EUDR Art. 10(3) | P0 |
| REQ-026-015 | Risk assessment documentation packages | EUDR Art. 10(4) | P0 |
| REQ-026-016 | Phase 3 risk mitigation coordination | EUDR Art. 10(3) | P0 |
| REQ-026-017 | Iterative mitigation-reassessment loop | EUDR Art. 10(3) | P0 |
| REQ-026-018 | Market placement prohibition enforcement | EUDR Art. 10(3) | P0 |
| REQ-026-019 | Maximum iteration limit with escalation | ISO 31000:2018 | P1 |
| REQ-026-020 | Simplified due diligence workflow variant | EUDR Art. 11, 13 | P0 |
| REQ-026-021 | Simplified DD eligibility evaluation | EUDR Art. 11, 29 | P0 |
| REQ-026-022 | Automatic escalation from simplified to standard | EUDR Art. 13 | P0 |
| REQ-026-023 | DDS content aggregation and validation | EUDR Art. 12, Annex II | P0 |
| REQ-026-024 | Compliance officer DDS approval gate | EUDR Art. 12(2) | P0 |
| REQ-026-025 | DDS submission to EU Information System | EUDR Art. 12(1) | P0 |
| REQ-026-026 | DDS-to-record linkage for five-year retention | EUDR Art. 4(7), 12 | P0 |
| REQ-026-027 | Event-triggered workflow re-execution | EUDR Art. 14, 15 | P0 |
| REQ-026-028 | Self-disclosure workflow | EUDR Art. 4(7) | P0 |
| REQ-026-029 | Original and re-executed record preservation | EUDR Art. 4(7) | P0 |
| REQ-026-030 | Compliance defense package generation | EUDR Art. 22-23 | P1 |
| REQ-026-031 | Penalty exposure estimation | EUDR Art. 22-23 | P2 |
| REQ-026-032 | Workflow variant selection by country risk | EUDR Art. 29 | P0 |
| REQ-026-033 | Automatic workflow escalation on country reclassification | EUDR Art. 29 | P0 |
| REQ-026-034 | ISO 19011 auditing principles as design constraints | ISO 19011:2018 Cl. 4 | P1 |
| REQ-026-035 | ISO 19011 audit process alignment | ISO 19011:2018 Cl. 6 | P1 |
| REQ-026-036 | Agent readiness verification before invocation | ISO 19011:2018 Cl. 7 | P0 |
| REQ-026-037 | Phase 1 agent execution in dependency order | EUDR Art. 9 | P0 |
| REQ-026-038 | QG-1 with nine completeness checks | EUDR Art. 9(1) | P0 |
| REQ-026-039 | Iterative Phase 1 re-execution on QG-1 failure | EUDR Art. 9(1) | P0 |
| REQ-026-040 | Phase 2 agent execution with parallel + sequential | EUDR Art. 10(2) | P0 |
| REQ-026-041 | QG-2 with criteria coverage and composite score verification | EUDR Art. 10(1), (2) | P0 |
| REQ-026-042 | Workflow routing based on QG-2 outcome | EUDR Art. 10(3) | P0 |
| REQ-026-043 | Phase 3 execution with compliance officer gates | EUDR Art. 10(3) | P0 |
| REQ-026-044 | Post-mitigation risk re-assessment | EUDR Art. 10(3) | P0 |
| REQ-026-045 | QG-3 residual risk gate with five checks | EUDR Art. 10(3) | P0 |
| REQ-026-046 | QG-4 DDS validation with six checks | EUDR Art. 12, Annex II | P0 |
| REQ-026-047 | DDS content generation from agent outputs | EUDR Art. 12, Annex II | P0 |
| REQ-026-048 | State machine with all defined states and transitions | EUDR Art. 8(2) | P0 |
| REQ-026-049 | Checkpoint saving with SHA-256 hash chain | EUDR Art. 4(7) | P0 |
| REQ-026-050 | Workflow recovery from checkpoint | EUDR Art. 8(2) | P0 |
| REQ-026-051 | State transition and recovery audit logging | EUDR Art. 4(7) | P0 |
| REQ-026-052 | 500+ concurrent workflows per instance | EUDR Art. 8(2) | P1 |
| REQ-026-053 | Cross-workflow execution context isolation | EUDR Art. 8(2) | P0 |
| REQ-026-054 | 25-agent dependency graph management | EUDR Art. 8(1) | P0 |
| REQ-026-055 | Standardized agent invocation protocol | EUDR Art. 8(2) | P0 |
| REQ-026-056 | Phase-level output consolidation with hashes | EUDR Art. 4(7) | P0 |
| REQ-026-057 | Cross-agent data quality checks | EUDR Art. 8(2), 10(4) | P0 |
| REQ-026-058 | Phase transition blocking on inconsistencies | EUDR Art. 8(2) | P0 |
| REQ-026-059 | < 5 minute standard workflow execution | Performance | P0 |
| REQ-026-060 | Configurable timeouts at all levels | Performance | P1 |
| REQ-026-061 | Four-category error classification | EUDR Art. 8(2) | P0 |
| REQ-026-062 | Exponential backoff retry for transient errors | Performance | P1 |
| REQ-026-063 | Precautionary principle for missing risk scores | EUDR Art. 10(1) | P0 |
| REQ-026-064 | Critical agent failure blocks workflow | EUDR Art. 9, 10 | P0 |
| REQ-026-065 | Immutable append-only audit trail | EUDR Art. 4(7), 10(4) | P0 |
| REQ-026-066 | SHA-256 hash chain provenance | EUDR Art. 4(7) | P0 |
| REQ-026-067 | SEC-005 audit logging integration | SEC Integration | P0 |
| REQ-026-068 | Tamper detection with periodic verification | EUDR Art. 4(7) | P0 |
| REQ-026-069 | Evidence package generation (24h routine, 4h emergency) | EUDR Art. 15, 29 | P0 |
| REQ-026-070 | Continuous inspection readiness | EUDR Art. 29 | P0 |
| REQ-026-071 | Multi-format evidence export (PDF, JSON, XML) | EUDR Art. 15, 29 | P1 |
| REQ-026-072 | Phase 1 integration with EUDR-001 through EUDR-015 | EUDR Art. 9 | P0 |
| REQ-026-073 | Phase 2 integration with EUDR-016 through EUDR-025 | EUDR Art. 10 | P0 |
| REQ-026-074 | Phase 1 output passthrough to Phase 2 agents | EUDR Art. 8(1) | P0 |
| REQ-026-075 | Event bus publishing (99.9% delivery SLA) | EUDR Art. 8(2) | P0 |
| REQ-026-076 | Event bus subscription and response | EUDR Art. 14, 15, 21 | P0 |
| REQ-026-077 | Five-year record retention | EUDR Art. 4(7) | P0 |
| REQ-026-078 | Immutable records with hash chains and signatures | EUDR Art. 4(7) | P0 |
| REQ-026-079 | Performance targets met (Section 13.1) | Performance | P0 |
| REQ-026-080 | Zero-hallucination deterministic processing | Platform Standard | P0 |
| REQ-026-081 | 500+ concurrent workflow scalability | Performance | P1 |
| REQ-026-082 | Security infrastructure integration (SEC-001 to SEC-011) | Security | P0 |
| REQ-026-083 | RBAC permission enforcement for all operations | Security | P0 |

---

## 16. Glossary

| Term | Definition |
|------|-----------|
| **CA** | Competent Authority -- Member State authority designated to enforce the EUDR |
| **DAG** | Directed Acyclic Graph -- graph structure used for dependency-aware execution ordering |
| **DDS** | Due Diligence Statement -- the formal declaration submitted to the EU Information System per Article 12 |
| **DD** | Due Diligence -- the three-step process of information collection, risk assessment, and risk mitigation |
| **EORI** | Economic Operators Registration and Identification -- EU customs registration number |
| **EUDR** | EU Deforestation Regulation -- Regulation (EU) 2023/1115 |
| **FPIC** | Free, Prior and Informed Consent -- requirement for activities affecting indigenous territories |
| **QG** | Quality Gate -- a mandatory checkpoint that evaluates defined criteria before permitting workflow progression |
| **TRACES NT** | Trade Control and Expert System New Technology -- the EU Information System for DDS submission |

---

## 17. References

1. Regulation (EU) 2023/1115 of the European Parliament and of the Council of 31 May 2023 (EUDR)
2. Regulation (EU) 2024/3234 amending Regulation (EU) 2023/1115 (First Postponement)
3. Regulation (EU) 2025/2650 amending Regulation (EU) 2023/1115 (Second Postponement and Simplification)
4. European Commission Guidance Document C(2025) 2485 for Regulation (EU) 2023/1115
5. ISO 19011:2018 -- Guidelines for auditing management systems
6. ISO 31000:2018 -- Risk management -- Guidelines
7. OECD-FAO Guidance for Responsible Agricultural Supply Chains (2016)
8. REQS-AGENT-EUDR-024 -- Third-Party Audit Manager Agent (GreenLang internal)
9. REQS-AGENT-EUDR-025 -- Risk Mitigation Advisor Agent (GreenLang internal)
10. AGENT-FOUND-001 -- GreenLang Orchestrator DAG Execution Engine (GreenLang internal)

---

*Document generated by GL-RegulatoryIntelligence on 2026-03-11.*
*This document will be reviewed and updated upon publication of additional European Commission guidance, delegated acts, or implementing acts related to Regulation (EU) 2023/1115.*
