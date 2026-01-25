# Reporting Guide - User Guide

**Audience**: Sustainability managers, ESG analysts, compliance officers, executive reporting teams
**Prerequisites**: Completed data uploads, basic understanding of Scope 3 categories, reporting requirements
**Time**: 20-60 minutes per report generation (varies by complexity)

---

## Overview

The GL-VCCI Platform enables automated generation of multi-standard Scope 3 carbon emission reports. This guide covers all aspects of report creation, customization, and distribution to meet your compliance and stakeholder communication needs.

### What You'll Learn
- Overview of supported reporting standards
- Step-by-step report generation for each standard
- Customization options and parameters
- Export formats and distribution
- Scheduling automated reports
- Interpreting and validating results
- Best practices for compliance reporting

### Supported Reporting Standards

The platform supports the following major standards:

1. **ESRS E1** (European Sustainability Reporting Standards - Climate Change)
2. **CDP Climate Change** (Carbon Disclosure Project)
3. **GHG Protocol Corporate Standard** (Scope 3 Supplement)
4. **ISO 14083:2023** (Quantification and reporting of GHG emissions)
5. **IFRS S2** (Climate-related Disclosures)

**💡 Tip**: Each standard has different requirements and disclosure formats. Choose based on your regulatory obligations and stakeholder needs.

---

## Section 1: Report Types Overview

### ESRS E1 - European Sustainability Reporting Standards

**Purpose**: EU Corporate Sustainability Reporting Directive (CSRD) compliance

**Who Needs This**:
- EU-based companies >500 employees
- Listed companies in EU
- EU subsidiaries of non-EU parent companies
- By 2028, extended to most large and listed companies

**Key Requirements**:
- Disclose all 15 Scope 3 categories (if material)
- Value chain emissions (upstream and downstream)
- Transition plan disclosure
- Scenario analysis
- Forward-looking targets
- Double materiality assessment

**Report Structure**:
- ESRS 2: General Disclosures
- E1-1: Transition plan for climate change mitigation
- E1-2: Policies related to climate change mitigation
- E1-3: Actions and resources in relation to climate change policies
- E1-4: Targets related to climate change mitigation
- E1-5: Energy consumption and mix
- E1-6: Gross Scopes 1, 2, 3 and Total GHG emissions
- E1-7: GHG removals and carbon credits
- E1-8: Internal carbon pricing
- E1-9: Anticipated financial effects from material climate risks and opportunities

[Screenshot: ESRS E1 report template preview]

**Platform Support**:
- ✅ Automated data collection for E1-6
- ✅ Category-by-category breakdowns
- ✅ Value chain mapping
- ✅ Data quality disclosures
- ⚠️ Manual input required for policies, targets, and scenario analysis (E1-1 through E1-5)

**💡 Tip**: ESRS requires assurance. Plan for third-party verification during report preparation.

### CDP Climate Change Questionnaire

**Purpose**: Voluntary disclosure to investors and supply chain partners

**Who Needs This**:
- Companies responding to investor requests
- Companies in major supply chains (CDP Supply Chain Program)
- Organizations seeking climate leadership recognition
- 18,700+ companies disclose to CDP globally

**Key Requirements**:
- Governance and strategy
- Risks and opportunities
- Business strategy and financial planning
- Targets and performance
- Emissions methodology (Scopes 1, 2, 3)
- Energy usage
- Verification status

**CDP Sections Relevant to Scope 3**:
- C6.5: Scope 3 emissions data
- C6.5a: Category-by-category breakdown
- C6.10: Scope 3 Category 1 emissions (Purchased goods and services)
- Plus category-specific questions (C6.11-C6.15)

**Scoring Considerations**:
- Data quality: Primary data scores higher than spend-based
- Coverage: Reporting more categories improves score
- Targets: Science-based targets earn more points
- Verification: Third-party assurance adds credibility
- Engagement: Supplier engagement activities increase score

[Screenshot: CDP scoring levels - Disclosure, Awareness, Management, Leadership]

**Platform Support**:
- ✅ Complete Scope 3 emission calculations
- ✅ Category breakdowns with methodology
- ✅ Data quality indicators
- ✅ Supplier engagement metrics
- ✅ Export to CDP format
- ⚠️ Manual input for governance, strategy sections

**💡 Tip**: CDP scoring algorithm rewards completeness and data quality. Use supplier-specific data wherever possible.

### GHG Protocol Corporate Standard - Scope 3

**Purpose**: Foundation standard for corporate emission inventories

**Who Needs This**:
- Most organizations measuring emissions
- Basis for many other standards (CDP, SBTi, etc.)
- Required for Science-Based Targets Initiative (SBTi)
- De facto global standard

**Key Requirements**:
- Report all 15 Scope 3 categories (if applicable)
- Document organizational and operational boundaries
- Describe calculation methodologies
- Disclose data quality
- Explain any exclusions
- Present base year and comparison

**15 Scope 3 Categories**:

**Upstream (Categories 1-8)**:
1. Purchased Goods and Services
2. Capital Goods
3. Fuel- and Energy-Related Activities
4. Upstream Transportation and Distribution
5. Waste Generated in Operations
6. Business Travel
7. Employee Commuting
8. Upstream Leased Assets

**Downstream (Categories 9-15)**:
9. Downstream Transportation and Distribution
10. Processing of Sold Products
11. Use of Sold Products
12. End-of-Life Treatment of Sold Products
13. Downstream Leased Assets
14. Franchises
15. Investments

[Screenshot: Scope 3 category wheel diagram]

**Materiality Assessment**:
Platform helps identify which categories apply:
- ✅ Automatically identifies relevant categories from data
- ✅ Estimates emission magnitudes for prioritization
- ✅ Flags categories below materiality threshold (<5% typically)

**Platform Support**:
- ✅ All 15 categories supported
- ✅ Multiple calculation methodologies per category
- ✅ Hierarchical methodology application
- ✅ Completeness tracking
- ✅ Base year comparison
- ✅ Full methodology disclosure

**💡 Tip**: GHG Protocol requires reporting all relevant categories. Document why any categories are excluded.

### ISO 14083:2023 - Quantification and Reporting of GHG Emissions

**Purpose**: Standardized methodology for transport and logistics emissions

**Who Needs This**:
- Logistics and transport companies
- Companies with significant Categories 4 and 9 (transport)
- Organizations seeking standardized transport emission calculations
- Required for EN 16258 and GLEC Framework compliance

**Key Requirements**:
- Well-to-wheel (WTW) emission factors
- Hub-based and distance-based calculations
- Modal split accounting (road, rail, air, sea)
- Allocation methodologies for shared transport
- Empty running considerations
- Infrastructure emissions

**ISO 14083 vs. GHG Protocol**:
- More granular transport-specific guidance
- Standardized allocation methods
- Harmonizes with GLEC Framework
- Compatible with GHG Protocol (complementary, not competing)

**Transport Modes Covered**:
- Road freight (truck, van)
- Rail freight
- Air freight
- Sea freight
- Inland waterway
- Multimodal transport

[Screenshot: ISO 14083 calculation framework]

**Platform Support**:
- ✅ Modal split analysis
- ✅ Distance-based and spend-based calculations
- ✅ Load factor adjustments
- ✅ Well-to-wheel emission factors
- ✅ Allocation for less-than-truckload (LTL)
- ✅ Integration with logistics data

**💡 Tip**: Use ISO 14083 for detailed transport reporting, then roll up to GHG Protocol Category 4 and 9.

### IFRS S2 - Climate-Related Disclosures

**Purpose**: Financial disclosure of climate-related risks and opportunities

**Who Needs This**:
- Publicly listed companies (especially international)
- Organizations preparing for mandatory climate financial disclosures
- Companies seeking investor confidence
- Aligned with TCFD recommendations

**Key Requirements**:
- Governance around climate-related risks
- Strategy for managing climate risks and opportunities
- Risk management processes
- Metrics and targets (including Scope 3)
- Financial impact disclosures
- Scenario analysis

**IFRS S2 vs. Other Standards**:
- Focus on financial materiality (not just environmental)
- Investor-focused (not general stakeholder)
- Integrated with financial reporting
- Cross-industry metric: Scope 3 emissions

[Screenshot: IFRS S2 four-pillar structure]

**Cross-Industry Metrics** (Required):
- Absolute Scope 3 emissions (tCO2e)
- Scope 3 emission intensity (tCO2e per revenue, per output unit)
- Disaggregated by Scope 3 category
- Financed emissions (for financial institutions)

**Platform Support**:
- ✅ Absolute Scope 3 emission calculations
- ✅ Intensity metrics (per revenue, per FTE, per product)
- ✅ Category disaggregation
- ✅ Trend analysis over time
- ⚠️ Manual input for governance, strategy, risk management sections

**💡 Tip**: IFRS S2 requires connection between emissions and financial impacts. Document how Scope 3 affects enterprise value.

---

## Section 2: Step-by-Step Report Generation

### Before You Begin: Data Readiness Checklist

Ensure data is ready before generating reports:

- ✅ **All data uploaded** for reporting period
- ✅ **Validation complete** - no critical errors
- ✅ **Supplier PCF data** collected where available
- ✅ **Organizational boundaries** defined
- ✅ **Base year** established (if first report, this becomes base year)
- ✅ **Methodology decisions** documented
- ✅ **Review data quality scores** - aim for average >3.0

**💡 Tip**: Run a data quality check before report generation. Navigate to Data > Quality Dashboard to review.

### Generating an ESRS E1 Report

**Step 1: Navigate to Report Generation**

1. Click **"Reports"** in main navigation
2. Select **"Generate New Report"**
3. Choose **"ESRS E1 - Climate Change"** from standard dropdown

[Screenshot: Report type selection screen]

**Step 2: Configure Report Parameters**

**Reporting Period**:
- **Financial Year**: Select your fiscal year (e.g., 2024)
- **Custom Period**: Or choose specific dates
  - Start Date: YYYY-MM-DD
  - End Date: YYYY-MM-DD
- **Comparative Period**: Previous year for comparison (optional but recommended)

**Organizational Boundary**:
- **Consolidation Approach**:
  - ○ Operational Control (most common)
  - ○ Financial Control
  - ○ Equity Share
- **Entities Included**: Select business units
  - ☑️ All entities (default)
  - ☐ Custom selection (advanced)

**Scope 3 Category Selection**:
- ☑️ **Category 1**: Purchased Goods and Services
- ☑️ **Category 2**: Capital Goods
- ☑️ **Category 3**: Fuel and Energy-Related Activities
- ☑️ **Category 4**: Upstream Transportation
- ☐ **Category 5**: Waste (not material - excluded)
- ☑️ **Category 6**: Business Travel
- ☑️ **Category 7**: Employee Commuting
- ☐ **Category 8**: Upstream Leased Assets (not applicable)
- ... (categories 9-15)

**💡 Tip**: Only include material categories (>5% of total Scope 3 or otherwise significant). Document exclusions.

**Data Quality Settings**:
- **Minimum Quality Threshold**: 2.0 (adjust based on requirements)
- **Exclude Low-Quality Data**: ☐ No (include but flag)
- **Verification Status**: Include all (verified and unverified)

**Report Sections** (ESRS E1 specific):
- ☑️ **E1-6**: GHG Emissions (automatically populated from data)
- ☑️ **E1-7**: GHG Removals and Carbon Credits
- ☐ **E1-1 to E1-5**: Policies and targets (manual input)
- ☐ **E1-8**: Internal carbon pricing (manual input)
- ☐ **E1-9**: Financial effects (manual input)

[Screenshot: ESRS E1 configuration form]

**Step 3: Configure Advanced Options**

Click **"Advanced Options"** to expand:

**Emission Factor Databases**:
- Priority 1: Supplier-specific PCF data (PACT)
- Priority 2: Ecoinvent 3.9
- Priority 3: US EPA EEIO
- Priority 4: DEFRA UK factors

**Allocation Methods** (for shared facilities/products):
- ○ Physical allocation (mass, volume)
- ● Economic allocation (revenue-based)
- ○ Other (specify)

**Uncertainty Analysis**:
- ☑️ Include uncertainty ranges
- Method: Monte Carlo (10,000 iterations)

**Assurance Level**:
- Limited assurance planned
- Provider: [Enter assurance provider]

**Step 4: Add Report Metadata**

**Report Information**:
- **Report Title**: "2024 ESRS E1 Climate-Related Disclosures"
- **Prepared By**: Your name
- **Reviewed By**: Reviewer name (optional)
- **Report Purpose**: "CSRD Compliance - Annual Sustainability Report"
- **Confidentiality**:
  - ○ Public
  - ● Internal
  - ○ Confidential

**Company Information** (pre-filled from profile):
- Legal Name
- Reporting Year
- Contact Information
- Logo and branding

**Step 5: Generate Report**

1. Click **"Preview Settings"** to review all parameters
2. Verify all selections are correct
3. Click **"Generate Report"**
4. **Processing time**: 30-90 seconds
   - Progress bar shows status
   - "Calculating emissions by category..."
   - "Aggregating results..."
   - "Generating visualizations..."
   - "Creating document..."

[Screenshot: Report generation progress bar]

**Step 6: Review Generated Report**

Report opens in preview mode:

**Executive Summary Page**:
- Total Scope 3 emissions: 45,230 tCO2e
- Year-over-year change: +5.2% (vs. 2023)
- Number of categories reported: 10 of 15
- Data quality score: 3.4/5.0
- Key drivers: Category 1 (62%), Category 4 (18%)

**Detailed Sections**:
1. **Introduction and Methodology**
   - Organizational boundaries
   - Reporting standards followed
   - Calculation approaches
   - Data sources and quality

2. **E1-6: Scope 3 GHG Emissions**
   - Total emissions by category
   - Upstream vs. downstream
   - Emission intensity metrics
   - Year-over-year trends

3. **Category-by-Category Analysis**
   - Detailed breakdown for each category
   - Calculation methodology
   - Data sources
   - Quality indicators
   - Key emission drivers

4. **Data Quality Statement**
   - Quality by category
   - Primary vs. secondary data
   - Improvement initiatives
   - Limitations and exclusions

5. **Verification Statement** (if applicable)
   - Assurance provider details
   - Assurance level
   - Assurance opinion

[Screenshot: Sample report page showing category breakdown]

**Step 7: Review and Edit**

Before finalizing, review carefully:

1. **Check calculations**:
   - Spot-check major categories
   - Verify totals add up correctly
   - Compare to previous reports for reasonableness

2. **Review narrative**:
   - Edit executive summary
   - Add context to significant changes
   - Customize methodology descriptions

3. **Verify data quality**:
   - Review quality flags
   - Check that low-quality data is disclosed
   - Document improvement plans

4. **Add manual sections** (for ESRS E1):
   - E1-1: Transition plan (narrative)
   - E1-2: Policies (text and references)
   - E1-3: Actions and resources (initiatives)
   - E1-4: Targets (quantitative targets)
   - E1-5: Energy consumption (if not auto-populated)

5. **Edit visualizations**:
   - Customize chart colors
   - Add annotations
   - Select chart types

**💡 Tip**: Save as draft frequently. Changes are auto-saved every 2 minutes, but manual save ensures nothing is lost.

**Step 8: Finalize and Export**

Once satisfied with report:

1. **Click "Finalize Report"**
   - Warning: Finalized reports cannot be edited (only superseded)
   - Creates permanent audit trail
2. **Add finalization notes**: Document any last-minute changes
3. **Confirm finalization**
4. **Report status** changes from "Draft" to "Final"

**Export Options**:
- **PDF**: For distribution and submission
  - Standard layout
  - Custom branded layout
- **Word (.docx)**: For further editing
  - Editable format
  - Useful for adding qualitative sections
- **Excel (.xlsx)**: Raw data and calculations
  - All data tables
  - Calculation worksheets
  - Charts as separate sheets
- **JSON**: Machine-readable format
  - API integration
  - Data warehouse import
- **XBRL**: For ESRS digital filing
  - EU ESEF format
  - Tagged data points

5. **Click "Download"** to get your preferred format
6. **Or click "Share"** to email or publish

[Screenshot: Export format options]

**💡 Tip**: Export to Word first if you need to add substantial narrative content, then convert to PDF for final distribution.

### Generating a CDP Climate Change Report

**Step 1-2: Navigate and Configure**

Similar to ESRS E1, but with CDP-specific options:

1. Navigate to **Reports > Generate New Report**
2. Select **"CDP Climate Change Questionnaire"**
3. **CDP-Specific Settings**:
   - **Responding To**:
     - ○ Investor Request
     - ○ Supply Chain Member Request
     - ● Both
   - **CDP Questionnaire Year**: 2024 (auto-selects current year)
   - **Reporting Boundary**: Same as financial accounts (Yes/No)

[Screenshot: CDP report configuration]

**Step 3: CDP-Specific Configuration**

**C6.5: Scope 3 Emissions Methodology**:
- **Calculation Approach**: For each category, select:
  - Supplier-specific method
  - Hybrid method
  - Average data method
  - Spend-based method
  - Distance-based method
  - Fuel-based method
  - Investment-specific method

**C6.5a: Data Quality Indicators**:
Platform auto-assigns:
- **Percentage of Scope 3 emissions calculated using primary data**: 32%
- **Percentage using secondary data**: 41%
- **Percentage using estimates**: 27%

**C6.10: Scope 3 Category 1 Detail** (Purchased Goods):
- Total emissions: 28,100 tCO2e
- Calculation methodology: Spend-based and supplier-specific hybrid
- Percentage of emissions calculated using supplier-provided data: 25%
- Explanation of changes: "15% increase due to volume growth, partially offset by renewable energy use by top 3 suppliers"

**CDP Scoring Optimization**:
Platform provides scoring hints:
- ✅ "Include supplier engagement metrics to improve Management score"
- ✅ "Add emissions reduction targets aligned with SBTi for Leadership"
- ⚠️ "Data quality below 50% primary data - consider supplier engagement to improve score"

[Screenshot: CDP scoring optimization suggestions]

**Step 4: Generate and Review**

1. Click **"Generate Report"**
2. **CDP Output Format**:
   - Pre-filled CDP ORS (Online Response System) format
   - Question-by-question responses
   - Data tables formatted for CDP upload
   - Supporting documentation appendix

3. **Review CDP-Specific Sections**:
   - C6.5: Scope 3 emissions data table
   - C6.5a: Category detail
   - C6.10-C6.15: Category-specific questions
   - Verification statements

**Step 5: Export for CDP Submission**

**Export Options**:
- **CDP ORS Upload File**: Excel format ready to import to CDP portal
- **PDF Report**: For internal records and verification
- **Supporting Documentation**: Zip file with methodology docs

**Submission Checklist**:
- ✅ All 15 categories addressed (reported or explained why excluded)
- ✅ Methodology disclosed for each category
- ✅ Data quality indicators provided
- ✅ Verification status stated
- ✅ Changes from prior year explained
- ✅ Supporting documents attached

**💡 Tip**: CDP submission deadline is typically July 31. Start report preparation in May to allow time for data collection and review.

### Generating a GHG Protocol Report

**Step 1-2: Navigate and Configure**

1. Navigate to **Reports > Generate New Report**
2. Select **"GHG Protocol Corporate Standard - Scope 3"**

**GHG Protocol-Specific Settings**:

**Organizational Boundaries**:
- **Consolidation Approach**:
  - ● Operational control
  - ○ Financial control
  - ○ Equity share
- **Describe boundary**: "All entities where we have operational control, covering 100% of operations"

**Operational Boundaries** (Scope 3 Categories):
- Document which categories apply to your business
- Explain exclusions for non-applicable categories
- Platform auto-suggests based on uploaded data

**Base Year**:
- **Base Year**: 2023 (or earlier if recalculating)
- **Base Year Emissions**: 43,000 tCO2e (if known)
- **Base Year Recalculation Policy**:
  - Recalculate if structural changes >5%
  - Document methodology changes

[Screenshot: GHG Protocol organizational boundary setup]

**Step 3: Methodology Selection per Category**

For each Scope 3 category, document calculation approach:

**Category 1: Purchased Goods and Services**
- **Primary Method**: Spend-based (EEIO factors)
- **Secondary Method**: Supplier-specific data (where available)
- **Activity Data Source**: Procurement system (SAP)
- **Emission Factor Source**: US EPA EEIO 2022
- **Percentage Coverage**: 95% of spend

**Category 4: Upstream Transportation**
- **Primary Method**: Distance-based
- **Activity Data**: Shipment records (weight, distance, mode)
- **Emission Factor Source**: DEFRA 2024, ISO 14083
- **Percentage Coverage**: 80% of shipments

**Category 6: Business Travel**
- **Primary Method**: Distance-based (air), fuel-based (rental cars)
- **Activity Data**: Travel booking system
- **Emission Factor Source**: DEFRA 2024
- **Percentage Coverage**: 100% of booked travel

[Screenshot: Methodology documentation by category]

**💡 Tip**: GHG Protocol requires transparency on methodology. More detail is better than less.

**Step 4: Data Quality and Uncertainty**

**Data Quality by Category**:
Platform calculates and displays:
- Category 1: Quality 2.5 (mostly spend-based)
- Category 4: Quality 3.8 (good activity data)
- Category 6: Quality 4.2 (primary travel data)

**Uncertainty Assessment**:
- **Method**: Monte Carlo simulation or default ranges
- **Uncertainty by Category**:
  - Category 1: ±50% (spend-based)
  - Category 4: ±20% (distance-based)
  - Category 6: ±15% (fuel-based)
- **Overall Scope 3 Uncertainty**: ±35%

**Step 5: Generate Report**

1. Click **"Generate Report"**
2. **GHG Protocol Report Structure**:

   **Section 1: Introduction**
   - Company overview
   - Reporting period
   - Organizational and operational boundaries

   **Section 2: Scope 3 Emission Inventory**
   - Total Scope 3 emissions
   - Emissions by category (upstream and downstream)
   - Comparison to base year
   - Emission trends

   **Section 3: Methodology**
   - Calculation approaches by category
   - Data sources and quality
   - Emission factor sources
   - Allocation methodologies

   **Section 4: Exclusions and Uncertainties**
   - Categories excluded and why
   - Data gaps and limitations
   - Uncertainty analysis
   - Improvement plans

   **Section 5: Assurance**
   - Verification scope and approach
   - Assurance statement

   **Appendices**:
   - Detailed calculations
   - Emission factor tables
   - Organizational boundary details

[Screenshot: GHG Protocol report table of contents]

**Step 6: Review and Finalize**

1. **Verify completeness**: All 15 categories addressed
2. **Check base year comparison**: Explains changes accurately
3. **Review exclusions**: Properly documented
4. **Validate uncertainty**: Reasonable ranges
5. **Finalize and export**

**Export Formats**:
- PDF (most common)
- Excel (detailed calculations)
- Word (editable)

**💡 Tip**: GHG Protocol reports are often used as the foundation for other standards (CDP, SBTi, etc.). Make this comprehensive.

### Generating an ISO 14083 Report

**Best for**: Organizations with significant transport and logistics emissions (Categories 4 and 9)

**Step 1-2: Navigate and Configure**

1. Navigate to **Reports > Generate New Report**
2. Select **"ISO 14083 - Transport GHG Emissions"**

**ISO 14083-Specific Settings**:

**Transport Chain Covered**:
- ☑️ Upstream transport (Category 4)
- ☐ Downstream transport (Category 9)
- ☑️ Business travel transport (Category 6 air/rail/road)

**Calculation Approach**:
- ● Distance-based (primary)
- ○ Fuel-based
- ○ Spend-based (not recommended for ISO 14083)

**Emission Scope**:
- ☑️ Well-to-wheel (WTW) - Total lifecycle
- ☐ Tank-to-wheel (TTW) - Direct combustion only

[Screenshot: ISO 14083 configuration]

**Step 3: Transport Mode Configuration**

For each transport mode, configure:

**Road Freight**:
- **Vehicle Types**: Rigid truck, Articulated truck, Light commercial vehicle
- **Load Factors**: Average 65% (or vehicle-specific)
- **Empty Running**: 25% (return trips)
- **Fuel Type**: Diesel, HVO, Electric
- **Emission Factor Source**: EN 16258, GLEC Framework

**Rail Freight**:
- **Rail Types**: Electric, Diesel
- **Load Factors**: Average 70%
- **Emission Factor Source**: Country-specific grid factors

**Air Freight**:
- **Flight Types**: Domestic, Short-haul (<3000km), Long-haul (>3000km)
- **Freight Configuration**: Belly cargo, Dedicated freighter
- **Radiative Forcing**: Multiplier 2.0 (accounts for non-CO2 effects)

**Sea Freight**:
- **Vessel Types**: Container, Bulk, Tanker, Ro-Ro
- **Vessel Size**: TEU capacity or DWT
- **Fuel Type**: HFO, MGO, LNG

[Screenshot: Transport mode configuration matrix]

**Step 4: Allocation Methodology**

Critical for shared transport:

**Allocation Basis**:
- ● Mass-based (ton-km)
- ○ Volume-based (m3-km)
- ○ Revenue-based
- ○ Other (specify)

**Shared Transport Factors**:
- **LTL (Less-than-Truckload)**: Allocate by weight and distance
- **Multimodal**: Track each leg separately
- **Intermodal**: Include transshipment emissions

**Step 5: Generate Report**

**ISO 14083 Report Structure**:

1. **Scope and Boundaries**
   - Transport chain description
   - System boundaries
   - Functional unit (e.g., ton-km)

2. **Transport Emission Inventory**
   - Total emissions by mode
   - Modal split (% of ton-km)
   - Well-to-wheel breakdown (WTT vs. TTW)
   - Emission intensity (gCO2e per ton-km)

3. **Methodology**
   - Calculation formulas for each mode
   - Emission factors used
   - Allocation methodology
   - Load factor assumptions
   - Empty running factors

4. **Data Quality**
   - Primary vs. modeled data
   - Data uncertainty
   - Temporal and geographical applicability

5. **Modal Comparison**
   - Emission intensity benchmarking
   - Mode shifting scenarios
   - Optimization opportunities

[Screenshot: ISO 14083 modal split visualization]

**Step 6: Review Transport Calculations**

**Validation Checks**:
- ✅ **Load factors realistic**: 50-80% typical range
- ✅ **Empty running included**: 20-30% typical
- ✅ **Modal emissions align with benchmarks**:
  - Sea freight: 10-40 gCO2e/ton-km
  - Rail freight: 20-60 gCO2e/ton-km
  - Road freight: 60-150 gCO2e/ton-km
  - Air freight: 500-1,500 gCO2e/ton-km
- ✅ **Well-to-tank factors included**: ~15-25% of tank-to-wheel

**💡 Tip**: ISO 14083 calculations can be complex. Use the platform's built-in validation to catch errors.

**Step 7: Export and Integration**

1. **Finalize report**
2. **Export formats**:
   - PDF: Standard ISO 14083 report
   - Excel: Detailed calculations by shipment
   - JSON: Integration with TMS (Transport Management System)

3. **Integration with GHG Protocol**:
   - Platform automatically rolls up ISO 14083 results to GHG Protocol Categories 4, 6, and 9
   - Maintains detailed ISO 14083 documentation as supporting evidence

**💡 Tip**: If you have detailed transport data, generate ISO 14083 report first, then use results for GHG Protocol/CDP reporting.

### Generating an IFRS S2 Report

**Step 1-2: Navigate and Configure**

1. Navigate to **Reports > Generate New Report**
2. Select **"IFRS S2 - Climate-Related Disclosures"**

**IFRS S2-Specific Settings**:

**Disclosure Scope**:
- ☑️ Governance
- ☑️ Strategy
- ☑️ Risk Management
- ☑️ Metrics and Targets (includes Scope 3)

**Financial Integration**:
- **Reporting Currency**: USD
- **Financial Year**: 2024
- **Revenue for Intensity**: $500M (from financial statements)

[Screenshot: IFRS S2 configuration with financial data]

**Step 3: Metrics Configuration**

**Cross-Industry Metrics** (Required):

**Absolute Emissions**:
- Scope 3 total: [Auto-calculated]
- By category: [Auto-generated table]

**Emission Intensity**:
- **Primary Intensity Metric**:
  - ● tCO2e per $M revenue
  - ○ tCO2e per unit produced
  - ○ tCO2e per FTE
- **Secondary Metrics** (optional): Select additional

**Disaggregation**:
- ☑️ By Scope 3 category
- ☑️ By business unit
- ☑️ By geography
- ☑️ By product line

**Financed Emissions** (if applicable to financial institutions):
- Not applicable for most organizations
- Required for banks, investors, insurers

**Step 4: Financial Materiality Assessment**

**Climate-Related Risks Affecting Scope 3**:
Document financial impacts:

**Transition Risks**:
- **Carbon Pricing**: Impact of potential Scope 3 carbon tax
  - Scenario: $50/tCO2e applied to purchased goods
  - Financial Impact: $2.25M annually
  - Mitigation: Supplier engagement program

- **Supplier Transition Costs**: Suppliers pass through decarbonization costs
  - Scenario: 5% price increase from top suppliers
  - Financial Impact: $1.8M annually
  - Mitigation: Long-term contracts, support supplier efficiency

**Physical Risks**:
- **Supply Chain Disruption**: Climate events affect supplier operations
  - Scenario: 10% of suppliers face climate-related disruptions
  - Financial Impact: $3.5M (lost production, alternative sourcing)
  - Mitigation: Diversify supplier base, build resilience

[Screenshot: Financial impact assessment template]

**Opportunities**:
- **Low-Carbon Products**: Market demand for low-carbon offerings
  - Opportunity: Premium pricing for verified low-carbon products
  - Financial Impact: +$5M revenue potential
  - Strategy: PCF labeling, marketing differentiation

**💡 Tip**: IFRS S2 requires quantification of financial impacts. Work with finance team to develop scenarios.

**Step 5: Generate Report**

**IFRS S2 Report Structure**:

1. **Governance**
   - Board oversight of climate risks
   - Management's role in assessing and managing climate risks
   - (Manual input required - platform provides template)

2. **Strategy**
   - Climate-related risks and opportunities (including Scope 3)
   - Effect on business model and value chain
   - Financial position, performance, and cash flows
   - Scenario analysis (2°C and 4°C scenarios)
   - (Manual input with auto-populated Scope 3 data)

3. **Risk Management**
   - Processes for identifying and assessing climate risks (including value chain)
   - Processes for managing climate risks
   - Integration into overall risk management
   - (Manual input required)

4. **Metrics and Targets**
   - **Cross-Industry Metrics** (Auto-populated):
     - Scope 3 emissions (absolute)
     - Scope 3 emission intensity
     - Disaggregated by category
   - **Industry-Specific Metrics**: Varies by industry
   - **Targets**:
     - Scope 3 reduction target: 30% by 2030 (vs. 2023)
     - Science-based: Aligned with SBTi
     - Progress tracking: [Auto-updated]

[Screenshot: IFRS S2 metrics and targets section]

**Step 6: Scenario Analysis**

IFRS S2 requires scenario analysis. Platform provides templates:

**2°C Scenario (Paris Agreement Aligned)**:
- **Carbon Price**: $100/tCO2e by 2030
- **Scope 3 Financial Impact**: $4.5M annually
- **Adaptation Strategy**:
  - Accelerate supplier engagement
  - Shift to low-carbon materials
  - Increase use of recycled content
- **Net Financial Impact**: -$2.1M (after adaptation)

**4°C Scenario (High Physical Risk)**:
- **Supply Chain Disruptions**: 25% of suppliers affected
- **Scope 3 Financial Impact**: $8.2M annually
- **Adaptation Strategy**:
  - Geographic diversification
  - Supplier resilience programs
  - Inventory buffers
- **Net Financial Impact**: -$4.5M (after adaptation)

[Screenshot: Scenario analysis visualization]

**Step 7: Review and Finalize**

1. **Review metrics**: Verify calculations
2. **Complete narrative sections**: Governance, strategy, risk management
3. **Validate financial impacts**: Cross-check with finance team
4. **Board review**: IFRS S2 requires board-level oversight
5. **Finalize and export**

**Export Formats**:
- PDF: For annual report integration
- XBRL: Digital filing format
- HTML: For website disclosure

**💡 Tip**: Integrate IFRS S2 disclosure into annual financial report for maximum investor visibility.

---

## Section 3: Customization Options

### Date Range and Period Selection

**Flexible Period Options**:

**Standard Periods**:
- Calendar Year (Jan 1 - Dec 31)
- Fiscal Year (configure in Settings)
- Quarterly (Q1, Q2, Q3, Q4)
- Half-Year (H1, H2)

**Custom Periods**:
- Any start and end date
- Useful for project-specific reporting
- Ad-hoc analysis

**Multi-Period Comparison**:
- ☑️ Compare to previous period
- ☑️ Compare to base year
- ☑️ Show 5-year trend

[Screenshot: Period selection interface with comparison options]

**💡 Tip**: Align reporting period with financial year for easier integration with annual reports.

### Scopes and Categories

**Category Selection**:

**Include/Exclude Categories**:
- Select only relevant categories
- Document exclusion rationale
- Platform checks completeness

**Category Groupings**:
- **Upstream**: Categories 1-8
- **Downstream**: Categories 9-15
- **Custom Groups**: Define your own groupings (e.g., "Supply Chain" = Cat 1+2+4)

**Materiality Thresholds**:
- **Significance Level**: 5% (categories below threshold flagged)
- **Minimum Reporting**: Report if >1,000 tCO2e (regardless of %)

[Screenshot: Category selection with materiality thresholds]

### Business Unit and Geographic Segmentation

**Organizational Segmentation**:

**By Business Unit**:
- ☑️ Manufacturing Division
- ☑️ Services Division
- ☑️ Retail Division
- View emissions by unit
- Compare performance across units

**By Geography**:
- ☑️ North America
- ☑️ Europe
- ☑️ Asia-Pacific
- Regional breakdowns
- Country-level detail (if data available)

**By Product Line**:
- ☑️ Product A
- ☑️ Product B
- ☑️ Product C
- Emission intensity per product
- Useful for product carbon labels

[Screenshot: Segmentation configuration]

**💡 Tip**: Segmentation requires data tagging at upload time. Configure tags in Data > Upload Settings.

### Methodology Customization

**Calculation Method Preferences**:

**Hierarchy of Methods**:
1. Supplier-specific data (highest quality)
2. Product-specific data (secondary)
3. Industry average data (tertiary)
4. Spend-based estimates (lowest quality)

**Override Hierarchy**:
- Allow manual method selection per category
- Document justification for deviations

**Emission Factor Selection**:
- **Primary Database**: Ecoinvent 3.9
- **Secondary Database**: US EPA EEIO
- **Tertiary Database**: DEFRA
- **Custom Factors**: Upload your own

[Screenshot: Methodology hierarchy configuration]

### Branding and Formatting

**Report Customization**:

**Branding**:
- Company logo (top-left, top-right, or center)
- Brand colors (primary, secondary)
- Font selection
- Header/footer customization

**Layout**:
- Portrait or landscape
- Single-column or two-column
- Chart size and placement
- Table formatting

**Content Sections**:
- ☑️ Executive summary
- ☑️ Table of contents
- ☑️ Methodology section
- ☑️ Category detail pages
- ☑️ Appendices
- ☐ Assurance statement page

[Screenshot: Branding and formatting options]

**Templates**:
- **Default**: Standard platform layout
- **Executive Brief**: Condensed 2-page summary
- **Detailed Technical**: Full methodology and calculations
- **Investor**: Financial materiality focus
- **Custom**: Save your own templates

**💡 Tip**: Create templates for recurring reports to ensure consistency and save time.

---

## Section 4: Export Formats and Distribution

### PDF Export

**PDF Options**:

**Quality**:
- Print Quality (300 DPI) - Larger file size
- Screen Quality (150 DPI) - Smaller file size

**Security**:
- ☑️ Password protect
- ☑️ Restrict editing
- ☑️ Restrict printing
- ☑️ Add watermark ("Draft", "Confidential", etc.)

**Accessibility**:
- ☑️ PDF/A compliance (archival)
- ☑️ Tagged for screen readers
- ☑️ Accessible tables and charts

[Screenshot: PDF export options]

**💡 Tip**: Use Print Quality for submission copies, Screen Quality for internal review to reduce email size.

### Excel Export

**Excel Workbook Structure**:

**Worksheets Included**:
1. **Summary**: Key metrics and totals
2. **Category Breakdown**: Detailed emissions by category
3. **Data Sources**: Activity data used in calculations
4. **Emission Factors**: Factors applied
5. **Calculations**: Formulas and intermediate steps
6. **Charts**: All visualizations as separate sheets
7. **Metadata**: Report parameters and settings

**Features**:
- Formulas preserved (can recalculate)
- Filterable tables
- Pivot tables for analysis
- Charts are editable

**💡 Tip**: Excel export is perfect for auditors who want to verify calculations or for further analysis in BI tools.

### Word Export

**Word Document Features**:

**Editable Format**:
- All text fully editable
- Charts inserted as images (can be replaced)
- Tables in native Word format
- Styles applied for consistent formatting

**Use Cases**:
- Add qualitative sections (governance, strategy)
- Customize narrative for different audiences
- Translate to other languages
- Combine with other reports

**💡 Tip**: Export to Word when you need to add substantial qualitative content not supported by automated generation.

### JSON/XML Export

**Machine-Readable Formats**:

**JSON Structure**:
```json
{
  "report": {
    "id": "rpt_2024_001",
    "standard": "GHG_Protocol_Scope3",
    "period": {
      "start": "2024-01-01",
      "end": "2024-12-31"
    },
    "emissions": {
      "total": 45230,
      "unit": "tCO2e",
      "categories": [
        {
          "category": 1,
          "name": "Purchased Goods and Services",
          "emissions": 28100,
          "methodology": "hybrid",
          "quality": 3.2
        }
        // ... other categories
      ]
    }
  }
}
```

**XML Options**:
- XBRL (for ESRS digital filing)
- Custom XML schema
- CDP ORS format

**Use Cases**:
- API integration
- Data warehouse import
- ESG reporting platforms
- Regulatory submission portals

### Email Distribution

**Share Reports Directly**:

1. **Click "Share" button** on finalized report
2. **Enter recipient emails**: Comma-separated
3. **Add message**: Optional cover note
4. **Select format**: PDF, Excel, or both
5. **Set permissions**:
   - ○ View only
   - ○ Download allowed
   - ○ Comment access
6. **Expiration**: Set link expiration (7, 30, 90 days, or never)
7. **Click "Send"**

[Screenshot: Email share dialog]

**💡 Tip**: Use view-only links for external stakeholders to prevent unauthorized redistribution.

### Public Publishing

**Publish to Web**:

**Create Public Link**:
- Generates unique URL
- Embeddable in website
- Responsive design for mobile
- No login required

**Customization**:
- Add introduction text
- Show/hide sections
- Branding applied
- SEO-friendly

**Use Cases**:
- Sustainability webpage
- Annual report integration
- Stakeholder transparency
- ESG rating agencies

[Screenshot: Published web report example]

**💡 Tip**: Public links are indexed by search engines. Great for ESG ratings and investor relations.

---

## Section 5: Schedule Automated Reports

### Setting Up Recurring Reports

**Automate Regular Reporting**:

1. **Navigate to Reports** > **Scheduled Reports**
2. **Click "New Scheduled Report"**
3. **Configure schedule**:

**Report Configuration**:
- **Report Type**: GHG Protocol Scope 3 (or any standard)
- **Report Template**: Select saved template
- **Parameters**: Define once, reuse each period

**Schedule Settings**:
- **Frequency**:
  - ○ Monthly (first day of month)
  - ● Quarterly (first day of quarter)
  - ○ Annually (specify date)
  - ○ Custom (cron expression)
- **Time**: 2:00 AM (off-peak)
- **Timezone**: [Your timezone]

**Reporting Period**:
- ● Rolling previous period (e.g., "previous quarter")
- ○ Fixed period (e.g., "Q1 2024")
- ○ Year-to-date

[Screenshot: Scheduled report configuration]

**Distribution**:
- **Recipients**: Email list
- **Subject**: "Quarterly Scope 3 Report - [Period]"
- **Format**: PDF
- **Include Link**: Also provide web link

**Notification**:
- ☑️ Notify when report generated
- ☑️ Alert if generation fails
- ☑️ Send preview 1 day before

**💡 Tip**: Schedule reports to run just after month/quarter close when data uploads are complete.

### Managing Scheduled Reports

**View Schedule Dashboard**:
- All scheduled reports
- Next run date
- Last run status
- Success/failure history

**Actions**:
- **Pause**: Temporarily disable
- **Edit**: Change parameters
- **Run Now**: Manual trigger
- **Delete**: Remove schedule

[Screenshot: Scheduled reports dashboard]

### Conditional Alerts

**Set Data Quality Thresholds**:

Alert if:
- Data quality score < 3.0
- Missing data for any category
- Emissions increase >20% vs. prior period
- Calculation errors detected

**Alert Actions**:
- Email notification to admin
- Pause scheduled report
- Require manual review before distribution

**💡 Tip**: Use alerts to catch data issues before reports go to stakeholders.

---

## Section 6: Interpret Report Results

### Understanding Emission Calculations

**Reading Category Results**:

**Category 1 Example**: Purchased Goods and Services

```
Category 1: Purchased Goods and Services
Total Emissions: 28,100 tCO2e

Calculation Breakdown:
- Supplier-Specific PCF Data:    7,200 tCO2e (25% of emissions)
  → Quality Rating: 4.5/5.0
  → Covers: Top 5 suppliers

- Industry Average Factors:      10,500 tCO2e (37% of emissions)
  → Quality Rating: 3.0/5.0
  → Covers: Next 20 suppliers

- Spend-Based Estimates:         10,400 tCO2e (38% of emissions)
  → Quality Rating: 2.0/5.0
  → Covers: Remaining 150+ suppliers

Overall Data Quality: 3.2/5.0
```

**Key Insights**:
- 25% of emissions calculated with high-quality supplier data
- 75% still using estimates - improvement opportunity
- Top 5 suppliers represent 25% of emissions but <5% of supplier count
- Pareto principle: Focus supplier engagement on top emitters

[Screenshot: Category breakdown with methodology mix]

**💡 Tip**: Focus data quality improvement efforts on categories with low scores and high emission magnitudes.

### Emission Intensity Metrics

**Understanding Intensity**:

**Absolute Emissions**: Total tCO2e
- Useful for: Targets, compliance reporting
- Limitation: Doesn't account for business growth

**Emission Intensity**: tCO2e per business metric
- tCO2e per $M revenue
- tCO2e per unit produced
- tCO2e per employee

**Example**:
```
Year 2023:
- Total Scope 3: 43,000 tCO2e
- Revenue: $450M
- Intensity: 95.6 tCO2e/$M revenue

Year 2024:
- Total Scope 3: 45,230 tCO2e (+5.2% absolute)
- Revenue: $500M
- Intensity: 90.5 tCO2e/$M revenue (-5.3% intensity)
```

**Interpretation**:
- Absolute emissions increased (more production)
- But intensity improved (more efficient)
- Business grew faster than emissions
- Decoupling growth from emissions ✅

[Screenshot: Absolute vs. intensity trend chart]

**💡 Tip**: Report both absolute and intensity metrics. Absolute for compliance, intensity for efficiency tracking.

### Year-Over-Year Comparison

**Analyzing Changes**:

**Scope 3 Change Analysis**:
```
Total Scope 3 Emissions:
2023: 43,000 tCO2e
2024: 45,230 tCO2e
Change: +2,230 tCO2e (+5.2%)

Drivers of Change:

Increases:
+ Category 1 (Purchased Goods): +3,100 tCO2e
  → 15% volume increase (business growth)
  → 5% methodology improvement (better data)

+ Category 4 (Transport): +400 tCO2e
  → Increased shipping distances (new supplier locations)

Decreases:
- Category 6 (Business Travel): -1,000 tCO2e
  → Continued remote work adoption
  → Shift from air to rail in Europe

- Category 1 (Supplier Improvements): -270 tCO2e
  → Top 3 suppliers switched to renewable energy

Net Change: +2,230 tCO2e
```

[Screenshot: Waterfall chart showing emission changes]

**💡 Tip**: Break down changes into structural (business growth), operational (efficiency), and methodological (data quality) components.

### Benchmark Comparisons

**Industry Benchmarking**:

Platform provides anonymized benchmarks:

**Your Organization**:
- Scope 3 Intensity: 90.5 tCO2e/$M revenue
- Data Quality Score: 3.2/5.0

**Industry Average** (Manufacturing - Industrial Equipment):
- Scope 3 Intensity: 105 tCO2e/$M revenue
- Data Quality Score: 2.8/5.0

**Top Quartile**:
- Scope 3 Intensity: 75 tCO2e/$M revenue
- Data Quality Score: 3.8/5.0

**Interpretation**:
- ✅ You're better than industry average (14% lower intensity)
- ⚠️ But 17% above top quartile - room for improvement
- ✅ Data quality above average

[Screenshot: Benchmark comparison chart]

**💡 Tip**: Use benchmarks to set ambitious but realistic targets. Top quartile performance is often achievable.

### Data Quality Indicators

**Quality Score Breakdown**:

**Overall Score: 3.2/5.0** - Fair quality

**By Category**:
- Category 1: 3.2 (Fair - mix of methods)
- Category 2: 2.5 (Poor - mostly spend-based)
- Category 4: 3.8 (Good - detailed shipment data)
- Category 6: 4.2 (Excellent - primary travel data)

**By Methodology**:
- Supplier-Specific Data (Quality 5): 15% of emissions
- Secondary Data (Quality 3-4): 40% of emissions
- Spend-Based Estimates (Quality 1-2): 45% of emissions

**Improvement Roadmap**:
1. Engage top 10 Category 1 suppliers for PCF data (would improve to 3.5)
2. Implement capital goods tracking system for Category 2 (would improve to 3.0)
3. Continue Category 6 best practices

**Target**: Overall score 4.0 by end of 2025

[Screenshot: Data quality improvement roadmap]

**💡 Tip**: Set quality score targets as part of your emissions strategy. Better data leads to better decisions.

---

## Section 7: Best Practices for Compliance Reporting

### Pre-Reporting Checklist

**30 Days Before Deadline**:
- ✅ Verify all data uploaded for reporting period
- ✅ Run data quality check
- ✅ Identify and resolve validation errors
- ✅ Contact suppliers for missing PCF data
- ✅ Review methodology documentation
- ✅ Check for significant changes requiring disclosure

**14 Days Before Deadline**:
- ✅ Generate draft report
- ✅ Review all calculations
- ✅ Compare to prior year - investigate significant changes
- ✅ Verify base year recalculation (if applicable)
- ✅ Complete qualitative sections (governance, targets)
- ✅ Internal stakeholder review

**7 Days Before Deadline**:
- ✅ Finalize report
- ✅ Executive review and approval
- ✅ Third-party assurance (if required)
- ✅ Legal review (if public disclosure)
- ✅ Prepare submission materials

**💡 Tip**: Start early. Last-minute data issues can derail submission deadlines.

### Documentation Requirements

**Maintain Audit Trail**:

**Required Documentation**:
1. **Activity Data Sources**
   - Procurement data export from ERP
   - Travel booking data
   - Logistics shipment records
   - Timestamps and version control

2. **Emission Factor Sources**
   - Database versions (Ecoinvent 3.9, EPA EEIO 2022, etc.)
   - Supplier-provided PCF certificates
   - Custom factor calculations
   - References and citations

3. **Methodology Documents**
   - Calculation procedures by category
   - Allocation methodologies
   - Organizational boundary definitions
   - Exclusion justifications

4. **Review and Approval**
   - Internal review sign-offs
   - Management approval
   - Board review (for IFRS S2, ESRS)
   - Assurance provider reports

**Storage**:
Platform automatically maintains audit trail:
- All data uploads archived
- Report versions saved
- Methodology changes tracked
- User actions logged

**💡 Tip**: Store documentation for 7+ years for audit and regulatory purposes.

### Third-Party Assurance

**Assurance Levels**:

**Limited Assurance**:
- Most common for GHG reports
- "Nothing has come to our attention that indicates the emissions are materially misstated"
- Lower cost, faster process
- Sufficient for most reporting requirements

**Reasonable Assurance**:
- Highest level
- "In our opinion, the emissions are fairly stated"
- Required by some regulations (e.g., EU CSRD phases)
- More intensive, higher cost

**Assurance Scope**:
- Scope 1, 2, 3 (all)
- Scope 1, 2 (partial)
- Selected Scope 3 categories (partial)

**Assurance Process**:

1. **Select Provider**:
   - Accredited firms (Big 4 accounting, specialized firms)
   - Check qualifications (ISO 14064-3)
   - Obtain quote

2. **Pre-Assurance Preparation**:
   - Organize documentation
   - Prepare calculation workbooks
   - Document data sources
   - Review methodology

3. **Assurance Engagement**:
   - Provider reviews documentation
   - Tests calculations
   - Samples transactions
   - Interviews personnel
   - Identifies issues

4. **Resolution**:
   - Address findings
   - Correct errors
   - Document justifications
   - Update report

5. **Assurance Statement**:
   - Provider issues opinion
   - Include in report
   - Publish with disclosures

**💡 Tip**: Engage assurance provider early (during data collection) to avoid surprises and rework.

### Common Reporting Pitfalls

**Pitfall 1: Incomplete Category Coverage**
- **Issue**: Excluding applicable categories without justification
- **Solution**: Document why each of 15 categories is included or excluded
- **Platform Help**: Auto-suggests applicable categories from data

**Pitfall 2: Inconsistent Boundaries**
- **Issue**: Different boundaries for Scope 3 vs. financial reporting
- **Solution**: Align organizational boundaries, document any differences
- **Platform Help**: Boundary settings applied consistently across all calculations

**Pitfall 3: Lack of Base Year Recalculation**
- **Issue**: Not recalculating base year after structural changes (mergers, acquisitions, divestitures)
- **Solution**: Follow base year recalculation policy (typically >5% change triggers recalculation)
- **Platform Help**: Tracks organizational changes and flags when recalculation needed

**Pitfall 4: Insufficient Data Quality Disclosure**
- **Issue**: Presenting estimates as facts without quality indicators
- **Solution**: Always disclose methodology and data quality level
- **Platform Help**: Auto-calculates quality scores, generates disclosure text

**Pitfall 5: Ignoring Uncertainty**
- **Issue**: Reporting precise numbers when underlying data has high uncertainty
- **Solution**: Provide uncertainty ranges, acknowledge limitations
- **Platform Help**: Monte Carlo uncertainty analysis built-in

**Pitfall 6: Double Counting**
- **Issue**: Counting same emissions in multiple categories (e.g., transport in Cat 1 and Cat 4)
- **Solution**: Clear allocation rules, avoid overlap
- **Platform Help**: Double-count detection alerts

**Pitfall 7: Cherry-Picking Positive Results**
- **Issue**: Highlighting reductions while hiding increases
- **Solution**: Balanced reporting, explain all material changes
- **Platform Help**: Objective reporting of all changes

**💡 Tip**: Review common reporting errors checklist before finalizing. Platform includes automated checks for many common issues.

---

## Troubleshooting

### Common Issues and Solutions

**Issue**: Report shows "Insufficient Data" error
**Solution**:
1. Check that data is uploaded for the selected reporting period
2. Navigate to Data > Upload History to verify
3. Ensure data passed validation
4. Check date formats match (YYYY-MM-DD)
5. Verify organizational unit selection matches uploaded data

**Issue**: Emissions seem unrealistically high or low
**Solution**:
1. Check units are correct (kg vs. tonnes, km vs. miles)
2. Verify emission factors are appropriate for region
3. Review for duplicate data uploads
4. Check calculation methodology is correct for category
5. Compare to prior periods and benchmarks
6. Use "Calculation Details" button to drill into specific transactions

**Issue**: Cannot generate ESRS E1 report - missing sections
**Solution**:
1. ESRS E1 requires manual input for qualitative sections (E1-1 to E1-5)
2. Navigate to Reports > ESRS E1 Template
3. Complete required narrative sections:
   - Transition plan (E1-1)
   - Policies (E1-2)
   - Actions (E1-3)
   - Targets (E1-4)
4. Save sections
5. Retry report generation

**Issue**: CDP export file format error
**Solution**:
1. Ensure you're using current CDP questionnaire year
2. Check all required questions are answered
3. Download fresh CDP ORS template
4. Use platform's "Export to CDP" function (not manual Excel)
5. Validate file using CDP's online tool before submission

**Issue**: Base year comparison shows "Not Available"
**Solution**:
1. Verify base year data is in system
2. Check base year is set in Settings > Reporting Configuration
3. Ensure base year data covers same categories as current year
4. If first year of reporting, base year comparison not possible (will be available next year)

**Issue**: Third-party assurance provider found calculation errors
**Solution**:
1. Note the specific errors identified
2. Check if issue is with source data or calculation
3. If source data error:
   - Correct in original system
   - Re-upload to platform
   - Regenerate report
4. If calculation error:
   - Review methodology settings
   - Verify emission factors
   - Check allocation methods
   - Contact support if platform calculation issue
5. Document corrections for audit trail

**Issue**: Report generation timeout - "Report taking too long"
**Solution**:
1. Platform limits report generation to 5 minutes
2. Usually indicates very large dataset (>100,000 transactions)
3. Solutions:
   - Reduce date range and generate multiple reports
   - Use summary-level data instead of transaction-level
   - Schedule report for off-peak hours
   - Contact support to increase timeout for large reports
4. Check system status page for performance issues

**Issue**: Charts not displaying correctly in exported PDF
**Solution**:
1. Try regenerating report
2. Clear browser cache
3. Use "Print Quality" PDF option instead of "Screen Quality"
4. Check browser is updated (Chrome recommended)
5. Try exporting to Word, then save as PDF from Word
6. Contact support with screenshot of issue

---

## FAQ

**Q: How long does report generation take?**
**A**: Most reports generate in 30-90 seconds. Large reports with extensive data (>50,000 transactions) may take 2-3 minutes. ISO 14083 reports with complex transport calculations may take slightly longer.

**Q: Can I edit a finalized report?**
**A**: No. Finalized reports are locked for audit trail purposes. If changes are needed:
- Withdraw and regenerate (if not yet submitted)
- Create new version with change notes (supersedes previous)
- Document reason for changes

**Q: Which report format should I use for regulatory submission?**
**A**: Depends on regulator:
- ESRS E1 (EU CSRD): XBRL (digital filing format)
- CDP: Excel (ORS format) or direct portal submission
- GHG Protocol: PDF (most common)
- IFRS S2: Integrated into financial report (typically PDF or XBRL)
Check specific regulator requirements.

**Q: How do I know if my data quality is sufficient?**
**A**: General guidelines:
- **Excellent (4.0+)**: Exceeds most requirements, suitable for any standard
- **Good (3.0-3.9)**: Acceptable for compliance, meets most standards
- **Fair (2.0-2.9)**: Minimum acceptable, may need improvement for some standards
- **Poor (<2.0)**: Insufficient for compliance, should not finalize
Check specific standard requirements. CDP and SBTi favor higher quality.

**Q: Can I use the same report for multiple standards?**
**A**: Partially. The underlying emission calculations are the same, but each standard has different:
- Disclosure requirements
- Report structure
- Narrative sections
- Terminology
Best practice: Generate standard-specific report for each. Platform reuses calculations across standards.

**Q: How often should I generate reports?**
**A**: Depends on use case:
- **Regulatory Compliance**: Annually (typically calendar year or fiscal year)
- **Internal Management**: Quarterly recommended
- **Investor Relations**: Semi-annually or annually
- **Supplier Reporting**: Annually or when requested
Set up scheduled reports for regular cadence.

**Q: What if I don't have data for all Scope 3 categories?**
**A**: You can:
- Report categories where data is available
- Document why other categories are excluded:
  - Not applicable to business model
  - Not material (below threshold, typically <5%)
  - Data not available (document collection plan)
- Most standards allow phased approach - improve coverage over time
- Platform helps identify applicable categories

**Q: Can I benchmark my emissions against industry peers?**
**A**: Yes. Platform provides anonymized benchmarking:
- Navigate to Analytics > Benchmarking
- Select your industry
- Compare emission intensity metrics
- All peer data is aggregated and anonymized
- Opt-in to contribute your data to benchmarks

**Q: How do I explain an emissions increase in my report?**
**A**: Be transparent and provide context:
1. Quantify the increase (absolute and %)
2. Break down drivers:
   - Business growth (more production)
   - Acquisitions (structural change)
   - Methodology changes (better data)
   - Operational changes
3. Note any offsetting reductions
4. Explain actions to address
5. Platform auto-generates change analysis - customize narrative

**Q: What's the difference between market-based and location-based for Scope 3?**
**A**: This distinction primarily applies to Scope 2 (purchased electricity). For Scope 3:
- Most categories use location-based (average grid factors)
- Exception: If supplier provides market-based PCF data (e.g., using renewable energy certificates), you can use that
- Document which approach used
- Some standards require reporting both

**Q: Do I need third-party assurance?**
**A**: Depends on standard and company size:
- **ESRS E1 (EU CSRD)**: Assurance required (limited initially, reasonable later)
- **CDP**: Not required, but improves score
- **GHG Protocol**: Recommended, not required
- **IFRS S2**: Increasingly expected by investors
- **SBTi**: Not required for target setting, but adds credibility
Many organizations start with limited assurance for Scopes 1 and 2, adding Scope 3 over time.

---

## Related Resources

### Platform Documentation
- [Getting Started Guide](./GETTING_STARTED.md) - Platform overview and setup
- [Data Upload Guide](./DATA_UPLOAD_GUIDE.md) - Prepare and upload emission activity data
- [Dashboard Usage Guide](./DASHBOARD_USAGE_GUIDE.md) - Analyze emissions before reporting
- [Supplier Portal Guide](./SUPPLIER_PORTAL_GUIDE.md) - Collect supplier PCF data

### Reporting Standards
- **ESRS E1**: `https://www.efrag.org/lab6` - Official ESRS documentation
- **CDP**: `https://www.cdp.net/en/guidance` - CDP guidance and questionnaires
- **GHG Protocol**: `https://ghgprotocol.org/scope-3-standard` - Scope 3 Standard and calculation tools
- **ISO 14083**: Purchase from ISO or national standards body
- **IFRS S2**: `https://www.ifrs.org/issued-standards/ifrs-sustainability-standards-navigator/` - IFRS sustainability standards

### Training and Support
- **Webinars**: Monthly training on each reporting standard
  - ESRS E1 Deep Dive: First Wednesday monthly
  - CDP Reporting Best Practices: Second Wednesday monthly
  - GHG Protocol Scope 3: Third Wednesday monthly
- **Office Hours**: Fridays 2-4pm for reporting questions
- **Support Email**: reporting-support@greenlang.io
- **Community**: `https://community.greenlang.io/reporting` - Share experiences with other users

### External Resources
- **GHG Protocol Calculation Tools**: Free Excel tools for each Scope 3 category
- **CDP Technical Notes**: Sector-specific guidance
- **WBCSD PACT**: Guidance on supplier data exchange
- **SBTi Resources**: Setting science-based Scope 3 targets
- **Assurance Providers**: Directory of qualified GHG assurance firms

---

**Congratulations!** You now have comprehensive knowledge of report generation on the GL-VCCI Platform. Remember:
- Start with data quality
- Choose the right standard for your needs
- Document methodology thoroughly
- Review before finalizing
- Leverage automation for recurring reports

For questions: reporting-support@greenlang.io

---

*Last Updated: 2025-11-07*
*Document Version: 1.0*
*Platform Version: 2.5.0*
