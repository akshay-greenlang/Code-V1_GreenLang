# Dashboard Usage Guide - User Guide

**Audience**: All platform users - executives, sustainability managers, analysts, procurement teams
**Prerequisites**: Data uploaded to platform, basic understanding of Scope 3 emissions
**Time**: 20-30 minutes to learn dashboard basics, ongoing for analysis

---

## Overview

Dashboards are your primary interface for understanding and analyzing your organization's Scope 3 emissions. This guide covers all aspects of dashboard navigation, customization, and analysis to help you make data-driven decarbonization decisions.

### What You'll Learn
- Navigate the five core dashboards
- Use filters and drill-downs for detailed analysis
- Create custom dashboard views
- Save and share dashboards with stakeholders
- Export dashboard data for external analysis
- Set up alerts and notifications
- Best practices for dashboard-driven insights

### Dashboard Philosophy

**Real-Time Insights**:
Dashboards update automatically as new data is uploaded, providing current view of emissions without manual report generation.

**Layered Detail**:
Start with high-level overview, drill down into specific categories, suppliers, or transactions for root cause analysis.

**Action-Oriented**:
Every dashboard is designed to answer specific questions and drive actions, not just display data.

---

## Section 1: Dashboard Overview

### The Five Core Dashboards

**1. Overview Dashboard** - Executive Summary
- **Purpose**: High-level view of entire Scope 3 footprint
- **Audience**: Executives, leadership team
- **Key Question**: "How are we performing overall?"
- **Update Frequency**: Real-time
- **Typical Use**: Monthly leadership meetings, board reports

**2. Emissions Dashboard** - Category Analysis
- **Purpose**: Detailed breakdown by Scope 3 category
- **Audience**: Sustainability managers, analysts
- **Key Question**: "Where are our emissions coming from?"
- **Update Frequency**: Real-time
- **Typical Use**: Deep-dive analysis, target setting

**3. Supplier Dashboard** - Value Chain View
- **Purpose**: Emissions by supplier and engagement status
- **Audience**: Procurement teams, supplier relationship managers
- **Key Question**: "Which suppliers should we engage?"
- **Update Frequency**: Real-time
- **Typical Use**: Supplier engagement planning, performance tracking

**4. Hotspot Dashboard** - Opportunity Identification
- **Purpose**: AI-identified reduction opportunities
- **Audience**: Sustainability managers, operations teams
- **Key Question**: "Where can we reduce emissions most effectively?"
- **Update Frequency**: Weekly (AI re-runs analysis)
- **Typical Use**: Reduction strategy development, project prioritization

**5. Data Quality Dashboard** - Data Governance
- **Purpose**: Track data completeness and quality
- **Audience**: Data managers, sustainability analysts
- **Key Question**: "How reliable are our emission calculations?"
- **Update Frequency**: Real-time
- **Typical Use**: Data improvement initiatives, assurance preparation

[Screenshot: Dashboard selector navigation bar]

**💡 Tip**: Bookmark your most-used dashboard. Platform remembers your last-viewed dashboard and opens it by default.

### Dashboard Layout Structure

**Standard Layout Components**:

**Top Bar**:
- Dashboard title and description
- Date range selector
- Refresh button
- Share button
- Customize button
- Export button

**Key Metrics Row** (Top):
- 3-5 large metric cards
- Primary KPIs for at-a-glance view
- Color-coded (green = good, yellow = warning, red = alert)
- Click to drill down

**Visualization Section** (Middle):
- Charts and graphs
- Tables with sortable columns
- Maps (if geographic data available)
- Drag to rearrange (in customize mode)

**Detail Panel** (Bottom):
- Supporting data tables
- Transaction-level detail
- Filters and selections

**Side Panel** (Right, collapsible):
- Filters
- Date range
- Segment selection
- Saved views

[Screenshot: Dashboard layout with all components labeled]

---

## Section 2: Overview Dashboard

### Key Metrics Cards

**Total Scope 3 Emissions**:
- **Value**: 45,230 tCO2e
- **Change**: +5.2% vs. prior period (red indicator)
- **Trend**: Small sparkline showing last 12 months
- **Click Action**: Opens Emissions Dashboard

**Emission Intensity**:
- **Value**: 90.5 tCO2e per $M revenue
- **Change**: -5.3% vs. prior period (green indicator)
- **Interpretation**: Emissions growing slower than revenue (decoupling)
- **Click Action**: Opens intensity trend chart

**Active Suppliers**:
- **Value**: 215 suppliers
- **With PCF Data**: 15 (7%)
- **Engagement Status**: 45 pending responses
- **Click Action**: Opens Supplier Dashboard

**Data Quality Score**:
- **Value**: 3.2 out of 5.0
- **Rating**: Fair
- **Coverage**: 68% of emissions
- **Click Action**: Opens Data Quality Dashboard

**Reduction Target Progress**:
- **Target**: 30% reduction by 2030 (vs. 2023 baseline)
- **Progress**: 2.1% achieved
- **Status**: On track (green)
- **Click Action**: Opens target tracking detail

[Screenshot: Key metrics cards with values and trends]

**💡 Tip**: Hover over any metric for tooltip with definition and calculation method.

### Emissions Trend Chart

**12-Month Trend Visualization**:
- **X-Axis**: Months (or quarters, years)
- **Y-Axis**: Total emissions (tCO2e)
- **Lines**:
  - Blue line: Actual emissions
  - Dotted blue line: Prior year (comparison)
  - Orange line: Emission intensity (tCO2e/$M revenue)
  - Green line: Target trajectory
- **Bands**: Shaded area showing uncertainty range

**Interactions**:
- **Hover**: See exact values for any month
- **Click Point**: Drill into that month's details
- **Drag to Zoom**: Select date range to zoom in
- **Legend Toggle**: Click legend items to show/hide lines

[Screenshot: Emissions trend chart with all elements]

**Insights to Look For**:
- ✅ Declining trend (good - emissions reducing)
- ⚠️ Increasing trend (requires investigation)
- ✅ Intensity declining while absolute increasing (business growing, efficiency improving)
- ⚠️ Seasonal patterns (plan for high-emission periods)
- ⚠️ Sudden spikes (investigate data quality or business changes)

### Category Breakdown

**Donut Chart Visualization**:
- **Segments**: Each Scope 3 category (1-15)
- **Size**: Proportional to emission magnitude
- **Color**: Gradient from red (highest) to green (lowest)
- **Labels**: Category name and percentage

**Categories Displayed**:
1. Category 1 (Purchased Goods): 62% - 28,100 tCO2e
2. Category 4 (Upstream Transport): 18% - 8,100 tCO2e
3. Category 6 (Business Travel): 12% - 5,400 tCO2e
4. Category 11 (Use of Sold Products): 5% - 2,300 tCO2e
5. Category 7 (Employee Commuting): 3% - 1,330 tCO2e
6. Others: <1% each

**Interactions**:
- **Hover**: See exact emissions and data quality for category
- **Click Segment**: Drill into category detail
- **Right-Click**: Export category data

[Screenshot: Category donut chart with percentages]

**💡 Tip**: Focus on top 3 categories - typically represent 80-90% of total emissions (Pareto principle).

### Top Emitting Suppliers

**Table View**:
| Rank | Supplier Name | Emissions (tCO2e) | % of Total | Data Quality | Engagement Status |
|------|---------------|-------------------|------------|--------------|-------------------|
| 1 | Acme Manufacturing | 8,450 | 18.7% | 2.5 (Poor) | PCF Requested |
| 2 | Global Logistics Co | 6,200 | 13.7% | 3.8 (Good) | PCF Received |
| 3 | Steel Supplies Inc | 4,100 | 9.1% | 2.0 (Poor) | Not Contacted |
| 4 | Tech Components Ltd | 3,800 | 8.4% | 4.5 (Excellent) | PCF Received |
| 5 | Energy Services Corp | 2,900 | 6.4% | 3.2 (Fair) | In Discussion |
| 6-10 | Others | 8,300 | 18.3% | - | - |

**Sortable Columns**:
- Click column header to sort
- Ascending/descending toggle

**Actions**:
- **Click Supplier**: View supplier detail page
- **Engagement Button**: Launch supplier engagement workflow
- **Export**: Download full supplier list

[Screenshot: Top suppliers table]

**Insights to Look For**:
- ⚠️ High emissions + Poor data quality = Priority for engagement
- ✅ High emissions + Excellent quality = Good supplier, maintain relationship
- 💡 Top 10 suppliers typically represent 50-70% of emissions

### Recent Activity Feed

**Activity Timeline**:
- Data uploads
- Report generations
- Supplier responses
- Alert notifications
- Significant changes

**Example Entries**:
- 📊 "Q4 2024 procurement data uploaded" - 2 hours ago
- ✅ "Acme Manufacturing submitted PCF data" - Yesterday
- 📈 "Category 1 emissions increased 15%" - 2 days ago
- 📄 "CDP Climate Change report generated" - Last week
- 🔔 "Data quality dropped below 3.0 threshold" - Last week

**Click Activity**: Navigate to related detail or take action

[Screenshot: Activity feed with icons and timestamps]

**💡 Tip**: Check activity feed daily to stay current on changes and required actions.

### Quick Actions Panel

**Common Actions**:
- 📤 Upload New Data
- 📧 Invite Supplier to Portal
- 📊 Generate Report
- 🎯 Create Reduction Target
- ⚙️ Customize Dashboard
- 📥 Export Dashboard Data

**One-Click Access**: Launch workflows without navigation

---

## Section 3: Emissions Dashboard

### Category-by-Category Analysis

**Category Cards Grid**:

Each category displays:
- **Category Number and Name**
- **Total Emissions**: Value in tCO2e
- **Change**: vs. prior period (% and absolute)
- **Data Quality**: Score and visual indicator
- **Materiality**: Percentage of total Scope 3
- **Status Indicator**:
  - 🔴 High priority (high emissions, low quality)
  - 🟡 Medium priority
  - 🟢 Good state (high quality data)

[Screenshot: Grid of category cards]

**Example Card - Category 1**:
```
Category 1: Purchased Goods and Services
28,100 tCO2e (+8.2% vs. prior period)
────────────────────────────────
Materiality: 62% of total
Data Quality: ⭐⭐⭐☆☆ (3.2/5.0)
Status: 🟡 Medium Priority
────────────────────────────────
Actions: [Drill Down] [Engage Suppliers]
```

**Interactions**:
- **Click Card**: Opens category detail page
- **Hover**: Quick stats tooltip
- **Filter**: Show only material categories (>5%)

### Category Deep Dive

**When You Click a Category**:

Opens detailed view with:

**1. Emission Calculation Breakdown**:
```
Category 1: Purchased Goods and Services
Total: 28,100 tCO2e

By Calculation Method:
├─ Supplier-Specific PCF:    7,200 tCO2e (26%) - Quality: 4.5/5.0
├─ Product-Level Factors:    6,500 tCO2e (23%) - Quality: 3.5/5.0
├─ Industry Averages:        8,900 tCO2e (32%) - Quality: 2.8/5.0
└─ Spend-Based (EEIO):       5,500 tCO2e (19%) - Quality: 2.0/5.0
```

**Visualization**: Stacked bar chart showing methodology mix

[Screenshot: Methodology breakdown stacked bar]

**2. Emission by Supplier** (Top 20):
- Table or bar chart
- Supplier name, emissions, % of category
- Data quality indicator per supplier
- Engagement status

**3. Trend Over Time**:
- Line chart of category emissions
- Last 12 months or custom range
- Overlay prior year for comparison

**4. Drivers of Change**:
AI-generated explanation:
```
Category 1 increased 8.2% (+2,140 tCO2e) due to:
• Volume increase: +15% more purchases (+4,200 tCO2e)
• Supplier improvements: 3 suppliers adopted renewable energy (-1,200 tCO2e)
• Methodology refinement: Better data quality adjusted baseline (-860 tCO2e)
```

**5. Improvement Opportunities**:
- Top 10 specific opportunities
- Estimated emission reduction potential
- Implementation difficulty
- Cost-effectiveness

**💡 Tip**: Start with highest-emission categories. Even small percentage reductions have big absolute impact.

### Upstream vs. Downstream View

**Toggle View**:
- **Upstream** (Categories 1-8): Emissions from purchased goods, services, logistics
- **Downstream** (Categories 9-15): Emissions from use of your products, end-of-life

**Pie Chart Split**:
- Upstream: 88% (39,800 tCO2e)
- Downstream: 12% (5,430 tCO2e)

**Interpretation**:
- **Upstream-heavy**: Focus on supply chain engagement, procurement policies
- **Downstream-heavy**: Focus on product design, customer guidance
- **Balanced**: Comprehensive strategy needed

[Screenshot: Upstream vs. downstream pie chart]

### Emission Intensity Metrics

**Multiple Intensity Views**:

**Per Revenue**:
- 90.5 tCO2e per $M revenue
- Common for all industries
- Useful for investor reporting (IFRS S2)

**Per Employee**:
- 10.2 tCO2e per FTE
- Useful for services companies
- Benchmark against industry

**Per Product Unit** (if applicable):
- 2.3 tCO2e per widget produced
- Product-level carbon labeling
- Customer communication

**Per Square Foot** (real estate):
- 0.05 tCO2e per sq ft
- Facility-level benchmarking

**Custom Intensity**:
- Define your own denominator
- Examples: per transaction, per delivery, per patient day

[Screenshot: Intensity metrics cards with trend arrows]

**💡 Tip**: Track both absolute and intensity. Intensity allows growth while reducing impact per unit.

### Scenario Modeling

**"What-If" Analysis**:

**Create Scenario**:
1. Click "New Scenario" button
2. Name scenario: "Renewable Energy Transition"
3. Define changes:
   - Category 1: Top 5 suppliers adopt 100% renewable energy
   - Expected reduction: 15% for those suppliers
4. Platform calculates impact
5. View results

**Scenario Results**:
```
Scenario: Renewable Energy Transition

Current State:
Category 1: 28,100 tCO2e

Projected State:
Category 1: 26,800 tCO2e

Reduction: 1,300 tCO2e (-4.6%)
Total Scope 3 Reduction: -2.9%
```

**Compare Multiple Scenarios**:
- Side-by-side comparison
- Cost-benefit analysis
- Implementation timeline
- Cumulative effect

[Screenshot: Scenario comparison table]

**💡 Tip**: Use scenarios for target-setting and board presentations. Shows emission reduction potential before committing resources.

---

## Section 4: Supplier Dashboard

### Supplier Portfolio Overview

**Key Metrics**:

**Total Suppliers**: 215
- **With PCF Data**: 15 (7%)
- **Engagement in Progress**: 45 (21%)
- **Not Contacted**: 155 (72%)

**Emission Coverage**:
- **Suppliers with PCF Data**: 15 suppliers = 18% of emissions
- **Goal**: 50% coverage by end of year

**Average Response Time**: 21 days (from invitation to PCF submission)

**Data Quality by Supplier Type**:
- Tier 1 (top 20): Average 3.8/5.0
- Tier 2 (next 50): Average 2.9/5.0
- Tier 3 (remaining): Average 2.1/5.0

[Screenshot: Supplier portfolio metrics]

### Supplier Ranking and Prioritization

**Supplier Table** (Sortable and Filterable):

Columns:
- **Rank**: By emission magnitude
- **Supplier Name**: Company name
- **Estimated Emissions**: tCO2e
- **% of Total Scope 3**: Materiality
- **Data Quality Score**: 1-5 stars
- **Engagement Status**: Badge with color
- **Last Contact**: Date
- **Actions**: Quick action buttons

**Engagement Status Options**:
- 🟢 **PCF Received**: Supplier submitted data
- 🔵 **In Progress**: Engagement started, awaiting response
- 🟡 **Pending**: Invitation sent, no response yet
- ⚪ **Not Contacted**: No engagement initiated
- 🔴 **Declined**: Supplier declined to participate
- ⚫ **Unresponsive**: Multiple follow-ups, no response

**Example Rows**:
| Rank | Supplier | Emissions | % Total | Quality | Status | Actions |
|------|----------|-----------|---------|---------|--------|---------|
| 1 | Acme Mfg | 8,450 | 18.7% | ⭐⭐☆☆☆ | 🟡 Pending | [Follow Up] [View] |
| 2 | Global Logistics | 6,200 | 13.7% | ⭐⭐⭐⭐☆ | 🟢 Received | [View Data] |
| 3 | Steel Supplies | 4,100 | 9.1% | ⭐⭐☆☆☆ | ⚪ Not Contacted | [Invite] |

[Screenshot: Supplier ranking table]

**Sorting Options**:
- By emission magnitude (default)
- By data quality (prioritize low quality)
- By engagement status
- By last contact date
- Alphabetically

**Filtering Options**:
- Engagement status
- Data quality threshold
- Emission threshold
- Geography
- Industry sector

**💡 Tip**: Focus on top 20 suppliers first. Use filter "Top 20 by Emissions + Not Contacted" to prioritize engagement.

### Supplier Engagement Workflow

**Quick Actions from Dashboard**:

**1. Invite Supplier**:
- Click "Invite" button
- Pre-filled invitation form:
  - Supplier name and contact
  - Products to request PCF for
  - Due date (default: 30 days)
  - Message template (customizable)
- Send invitation
- Status changes to "Pending"

**2. Follow Up**:
- Click "Follow Up" for pending invitations
- View engagement timeline
- Send reminder (automatic template)
- Schedule call
- Escalate to account manager

**3. Track Responses**:
- Response rate by supplier tier
- Average time to respond
- Completion rate
- Identify best practices (what messaging works)

[Screenshot: Supplier engagement workflow]

### Supplier Detail Page

**Click Supplier Name** to open detail page:

**Overview Tab**:
- Company information
- Contact details
- Relationship summary (years, spend, categories)
- Emission estimate and quality
- Engagement history timeline

**Emissions Tab**:
- Emissions by product
- Emissions by category
- Trend over time
- Data quality breakdown

**Products Tab**:
- List of products purchased
- Emissions per product
- PCF data availability
- Request specific product PCFs

**Engagement Tab**:
- Communication history
- Invitations sent and responses
- PCF submissions
- Data quality feedback
- Notes and follow-ups

**Documents Tab**:
- Uploaded PCF certificates
- LCA reports
- Sustainability reports
- Contracts and agreements

[Screenshot: Supplier detail page tabs]

**💡 Tip**: Use supplier detail page to prepare for engagement calls. Review history and current data quality.

### Geographic Supplier Map

**Interactive World Map**:
- Suppliers plotted by headquarters location
- Bubble size = emission magnitude
- Color = data quality (red = poor, green = excellent)
- Click bubble = supplier detail
- Zoom and pan

**Insights**:
- Identify geographic concentration
- Regional emission hotspots
- Target regional engagement campaigns
- Assess supply chain resilience

[Screenshot: Geographic supplier map with bubbles]

**💡 Tip**: Use map view for executive presentations. Visual impact helps communicate supply chain footprint.

---

## Section 5: Hotspot Dashboard

### AI-Powered Hotspot Identification

**What is a Hotspot?**:
Specific emission sources with high reduction potential based on:
- Emission magnitude
- Data quality (certainty of calculation)
- Reduction feasibility (technical, economic)
- Organizational influence (ability to change)

**Hotspot Analysis Process**:
1. Platform analyzes all transactions
2. AI identifies patterns and outliers
3. Calculates reduction potential
4. Scores feasibility
5. Ranks opportunities
6. Updates weekly

### Top 20 Hotspots

**Hotspot Table**:

Columns:
- **Rank**: By overall opportunity score
- **Description**: What and where
- **Current Emissions**: tCO2e/year
- **Reduction Potential**: tCO2e/year and %
- **Feasibility**: Low/Medium/High
- **Cost**: $ (High/Medium/Low/Negative*)
- **Payback**: Years
- **Status**: Not Started/In Progress/Completed

*Negative cost = cost savings (e.g., energy efficiency)

**Example Hotspots**:

**Rank 1: Switch Top 5 Suppliers to Renewable Energy**
- Current: 12,300 tCO2e/year
- Potential: -1,850 tCO2e/year (-15%)
- Feasibility: High (suppliers have green tariff options)
- Cost: Low (suppliers pass through cost, minimal premium)
- Payback: 2 years
- Status: 🟡 In Progress (2 of 5 committed)

**Rank 2: Modal Shift - Air to Sea Freight for Non-Urgent**
- Current: 3,200 tCO2e/year
- Potential: -2,400 tCO2e/year (-75%)
- Feasibility: Medium (requires planning, longer lead times)
- Cost: Negative (sea freight cheaper than air)
- Payback: Immediate
- Status: ⚪ Not Started

**Rank 3: Consolidate Shipments - Reduce Frequency**
- Current: 2,100 tCO2e/year
- Potential: -630 tCO2e/year (-30%)
- Feasibility: Medium (requires inventory management changes)
- Cost: Low
- Payback: 1 year
- Status: ⚪ Not Started

[Screenshot: Hotspots table with opportunity scores]

**💡 Tip**: Focus on "quick wins" - high potential, high feasibility, low cost. Build momentum with early successes.

### Hotspot Detail View

**Click Hotspot** to open detail:

**1. Detailed Description**:
- What: Specific emission source
- Where: Category, supplier, product, location
- Why: Root cause analysis
- Current State: Baseline emissions and methodology

**2. Reduction Strategy**:
AI-generated recommendation:
```
Strategy: Switch to Sea Freight for Non-Urgent Shipments

Current Practice:
- All shipments from Asia to US via air freight
- 250 shipments/year averaging 500 kg each
- Emission Factor: 1.2 kgCO2e per ton-km
- Distance: 10,000 km
- Total: 3,200 tCO2e/year

Proposed Practice:
- 80% of shipments switch to sea freight
- 20% remain air (urgent deliveries)
- Sea Emission Factor: 0.012 kgCO2e per ton-km
- Total: 800 tCO2e/year (air) + 240 tCO2e/year (sea) = 1,040 tCO2e

Reduction: 2,160 tCO2e/year (-68%)
```

**3. Implementation Plan**:
- Step-by-step actions
- Responsible teams
- Timeline (weeks or months)
- Dependencies
- Risks and mitigation

**4. Cost-Benefit Analysis**:
- Implementation costs
- Operational cost changes
- Emission reduction value (at internal carbon price)
- Net financial impact
- Payback period
- NPV over 5 years

**5. Similar Opportunities**:
- Related hotspots (e.g., other modal shifts)
- Apply same strategy elsewhere
- Scalability

[Screenshot: Hotspot detail with strategy and cost-benefit]

### Hotspot Categories

**Filter by Opportunity Type**:

**Supplier-Related**:
- Supplier renewable energy adoption
- Supplier efficiency improvements
- Alternative suppliers (lower carbon)
- Supplier consolidation

**Logistics-Related**:
- Modal shift (air→sea, truck→rail)
- Shipment consolidation
- Optimize routing
- Nearshoring (reduce distance)

**Procurement-Related**:
- Material substitution (high carbon → low carbon)
- Circular economy (recycled content)
- Design for environment (lightweight, efficient)
- Volume optimization (reduce waste)

**Process-Related**:
- Energy efficiency
- Process optimization
- Waste reduction
- Technology upgrades

**Behavioral**:
- Travel policy changes (reduce flights)
- Remote work policies (reduce commuting)
- Procurement guidelines (favor low-carbon)

[Screenshot: Hotspot category filter]

**💡 Tip**: Address multiple hotspots simultaneously for synergistic benefits. Example: Supplier engagement + Material substitution.

### Progress Tracking

**Track Initiative Implementation**:

1. **Mark hotspot as "In Progress"**
2. **Add project details**:
   - Project owner
   - Start date
   - Target completion date
   - Milestones
3. **Update progress**:
   - % complete
   - Actual emission reduction achieved
   - Actual costs incurred
   - Lessons learned
4. **Mark as "Completed"**
5. **Platform calculates**:
   - Total reduction achieved
   - Progress toward targets
   - ROI realized

**Progress Dashboard**:
- Number of hotspots addressed
- Total emission reduction achieved
- Total cost or savings
- Average payback period
- Success rate

[Screenshot: Hotspot progress tracking dashboard]

### Export Hotspot Analysis

**Create Business Case**:
1. Select hotspots for initiative
2. Click "Export Business Case"
3. Generates PowerPoint or PDF:
   - Executive summary
   - Detailed opportunity analysis
   - Implementation plan
   - Financial analysis
   - Risk assessment
4. Customizable template

**Use for**:
- Executive approval
- Budget requests
- Sustainability committee meetings
- Board presentations

---

## Section 6: Data Quality Dashboard

### Overall Quality Score

**Quality Score Calculation**:
```
Weighted Average Quality Score: 3.2 / 5.0

By Emission Magnitude:
Quality 5 (Excellent):      18% of emissions (8,140 tCO2e)
Quality 4 (Good):           22% of emissions (9,950 tCO2e)
Quality 3 (Fair):           31% of emissions (14,020 tCO2e)
Quality 2 (Poor):           24% of emissions (10,855 tCO2e)
Quality 1 (Very Poor):       5% of emissions (2,265 tCO2e)
```

**Visualization**: Horizontal stacked bar showing quality distribution

[Screenshot: Quality score distribution bar]

**Target**: 4.0 by end of 2025

### Quality by Category

**Category Quality Matrix**:

| Category | Emissions (tCO2e) | Quality Score | Primary Data % | Target Quality |
|----------|-------------------|---------------|----------------|----------------|
| Cat 1: Purchased Goods | 28,100 | 3.2 | 26% | 4.0 |
| Cat 2: Capital Goods | 3,200 | 2.5 | 10% | 3.5 |
| Cat 4: Transport | 8,100 | 3.8 | 65% | 4.0 |
| Cat 6: Business Travel | 5,400 | 4.2 | 95% | 4.5 |
| Cat 7: Commuting | 1,330 | 2.8 | 5% | 3.5 |

**Color Coding**:
- 🟢 Green: Quality ≥ 4.0 (excellent)
- 🟡 Yellow: Quality 3.0-3.9 (good, can improve)
- 🔴 Red: Quality < 3.0 (poor, priority for improvement)

**Click Category**: View quality improvement plan

[Screenshot: Category quality matrix]

### Methodology Mix

**Calculation Method Breakdown**:

**Donut Chart**:
- **Supplier-Specific PCF** (Quality 5): 18% - 8,140 tCO2e
- **Product-Specific Factors** (Quality 4): 22% - 9,950 tCO2e
- **Industry Averages** (Quality 3): 31% - 14,020 tCO2e
- **Spend-Based EEIO** (Quality 2): 24% - 10,855 tCO2e
- **Estimates/Proxies** (Quality 1): 5% - 2,265 tCO2e

**Goal**: Shift from bottom (spend-based) to top (supplier-specific)

**Strategy**:
1. Engage suppliers for PCF data (move from Quality 2→5)
2. Collect activity data (move from Quality 2→3)
3. Use product-specific factors (move from Quality 3→4)

[Screenshot: Methodology mix donut chart]

**💡 Tip**: Every supplier engaged is a quality improvement. Track methodology shifts as success metric.

### Data Completeness

**Field Completeness**:

| Field | Completeness | Impact on Quality |
|-------|--------------|-------------------|
| Supplier Name | 100% | Required |
| Category | 100% | Required |
| Spend Amount | 100% | Required |
| Transaction Date | 100% | Required |
| Quantity | 45% | High - enables activity-based |
| Unit | 45% | High - enables activity-based |
| Product Description | 78% | Medium - helps factor matching |
| Geography | 62% | Medium - enables regional factors |
| Business Unit | 85% | Low - for segmentation only |

**Improvement Priority**: Focus on Quantity and Unit (high impact, currently 45%)

**Action Plan**:
1. Work with ERP team to include quantity in exports
2. Provide guidance to procurement on capturing units
3. Backfill quantity for top emitting transactions

[Screenshot: Field completeness bar chart]

### Data Age and Currency

**Data Recency**:

**By Data Age**:
- 0-6 months: 62% of emissions (current)
- 6-12 months: 28% of emissions (acceptable)
- 12-24 months: 8% of emissions (outdated, should update)
- >24 months: 2% of emissions (very outdated, must update)

**Oldest Data**:
- Category 2 (Capital Goods): Average age 18 months
- Category 7 (Commuting): Average age 14 months
- Category 1 (Purchased Goods): Average age 4 months

**Action**: Update Category 2 and 7 data sources

[Screenshot: Data age distribution]

### Uncertainty Analysis

**Emission Uncertainty Ranges**:

Platform calculates uncertainty for each emission source based on data quality.

**Overall Scope 3 Uncertainty**: ±35%
- With 95% confidence: 29,400 tCO2e to 61,060 tCO2e
- Expected value: 45,230 tCO2e

**By Category**:
| Category | Emissions | Uncertainty Range | Confidence |
|----------|-----------|-------------------|------------|
| Cat 1 | 28,100 tCO2e | ±45% | Low-Medium |
| Cat 4 | 8,100 tCO2e | ±20% | High |
| Cat 6 | 5,400 tCO2e | ±10% | Very High |

**Visualization**: Error bars on emission chart showing ranges

[Screenshot: Uncertainty ranges visualization]

**Why It Matters**:
- Transparency for stakeholders
- Setting realistic targets (don't over-rely on uncertain data)
- Prioritizing data quality improvements
- Assurance preparation

**💡 Tip**: Reducing uncertainty is as important as reducing emissions. Better data = better decisions.

### Quality Improvement Roadmap

**Automated Improvement Plan**:

Platform generates personalized roadmap:

**Q1 2025**:
1. ✅ Engage top 5 suppliers for PCF data (Cat 1)
   - Expected quality improvement: 3.2 → 3.5
   - Target: 15% of Cat 1 emissions covered by PCF
2. ✅ Implement travel data integration (Cat 6)
   - Expected quality improvement: 4.2 → 4.5
   - Already excellent, maintain and refine

**Q2 2025**:
3. Collect product quantities from procurement (Cat 1)
   - Expected quality improvement: 3.5 → 3.8
   - Move from spend-based to activity-based
4. Update commuting survey (Cat 7)
   - Expected quality improvement: 2.8 → 3.2
   - Current data 14 months old

**Q3 2025**:
5. Engage next 10 suppliers for PCF data (Cat 1)
   - Expected quality improvement: 3.8 → 4.0
   - Target: 35% of Cat 1 emissions covered by PCF

**Q4 2025**:
6. Implement capital goods tracking system (Cat 2)
   - Expected quality improvement: 2.5 → 3.5
   - Major improvement in lowest-quality category

**Target**: Overall quality 4.0 by end of 2025

[Screenshot: Quality improvement roadmap Gantt chart]

**💡 Tip**: Share roadmap with leadership to secure resources for data collection initiatives.

---

## Section 7: Filters and Drill-Downs

### Global Filters (Apply to All Dashboards)

**Filter Panel** (Right side, collapsible):

**Date Range**:
- **Presets**: This Month, This Quarter, This Year, Last 12 Months
- **Custom**: Select any start and end date
- **Comparison**: Toggle to show prior period comparison

**Business Units** (if segmented):
- ☑️ Manufacturing Division
- ☑️ Services Division
- ☐ Retail Division (unchecked = excluded)
- Select All / Deselect All

**Geographic Region**:
- ☑️ North America
- ☑️ Europe
- ☑️ Asia-Pacific
- ☐ Other

**Scope 3 Categories**:
- Select individual categories or use presets:
  - All Categories
  - Upstream Only (1-8)
  - Downstream Only (9-15)
  - Material Categories Only (>5%)

**Data Quality Threshold**:
- Slider: 1.0 to 5.0
- Include only emissions with quality ≥ selected
- Example: Set to 3.0 to view only fair-quality or better

**Suppliers**:
- Search and select specific suppliers
- Or filter by tier:
  - Tier 1 (Top 20 by emissions)
  - Tier 2 (Next 50)
  - Tier 3 (All others)

[Screenshot: Filter panel with all options]

**💡 Tip**: Filters persist across dashboards. Set once, apply everywhere.

### Drill-Down Navigation

**Click-Through Hierarchy**:

**Level 1: Overview Dashboard**
- Total Scope 3: 45,230 tCO2e
- Click →

**Level 2: Category Level**
- Category 1: 28,100 tCO2e
- Click →

**Level 3: Supplier Level**
- Acme Manufacturing: 8,450 tCO2e
- Click →

**Level 4: Product Level**
- Industrial Motors: 3,200 tCO2e
- Click →

**Level 5: Transaction Level**
- PO-2024-001: 125 tCO2e
- View full transaction detail

**Navigation**: Breadcrumb trail at top shows path, click any level to jump back

[Screenshot: Breadcrumb navigation showing drill-down path]

### Advanced Filtering

**Combine Multiple Filters**:

Example: "Show me high-emission, low-quality suppliers in Asia"
- Filter 1: Suppliers with >1,000 tCO2e
- Filter 2: Data quality < 3.0
- Filter 3: Geography = Asia-Pacific
- Result: 8 suppliers meeting all criteria

**Save Filter Combination**: Click "Save Filter Set" for reuse

**Filter Logic**:
- Default: AND (all conditions must be true)
- Advanced: Switch to OR logic if needed

[Screenshot: Advanced filter builder]

### Chart Interactions

**All Charts Support**:

**Hover**: Tooltip with exact values
**Click**: Drill down to next level
**Right-Click**: Context menu
- Export chart as image (PNG, SVG)
- Export data (CSV, Excel)
- Add to custom dashboard
- Share via email

**Zoom**:
- Click and drag on chart to zoom into selection
- Double-click to reset zoom

**Legend Toggle**:
- Click legend items to show/hide series
- Useful for complex charts with many categories

**Chart Type Switcher**:
- Bar chart ↔ Line chart ↔ Pie chart
- Choose visualization that best tells your story

[Screenshot: Chart interaction menu]

**💡 Tip**: Right-click any chart to export for inclusion in presentations or reports.

---

## Section 8: Creating Custom Dashboards

### Dashboard Customization Mode

**Enter Customization**:
1. Click "Customize Dashboard" button (top-right)
2. Dashboard enters edit mode
3. All widgets show resize handles and move cursors

**Edit Mode Features**:
- **Add Widget**: Library of available widgets
- **Remove Widget**: X button on each widget
- **Resize**: Drag corners to resize
- **Move**: Drag to reposition
- **Configure**: Gear icon to change widget settings

[Screenshot: Dashboard in edit mode with handles visible]

### Widget Library

**Available Widgets**:

**Metric Cards**:
- Single KPI with trend
- Multiple KPIs grid
- Comparison cards (current vs. target)

**Charts**:
- Line chart (trends)
- Bar chart (comparisons)
- Pie/donut chart (proportions)
- Area chart (stacked trends)
- Scatter plot (correlations)
- Waterfall chart (changes)

**Tables**:
- Data table (sortable, filterable)
- Pivot table (cross-tabulation)
- Ranking table (top N items)

**Text**:
- Text box (annotations, instructions)
- Markdown support (headers, lists, links)
- Executive summary

**Images**:
- Company logo
- Charts from external sources
- Screenshots

**Maps**:
- Geographic supplier map
- Regional emission heatmap

**AI Widgets**:
- Insight summary (auto-generated)
- Recommendation list
- Anomaly detection

[Screenshot: Widget library panel]

### Building a Custom Dashboard

**Example: Executive Sustainability Dashboard**

**Step 1: Create New Dashboard**
1. Navigate to Dashboards
2. Click "New Custom Dashboard"
3. Name: "Executive Sustainability Dashboard"
4. Description: "Monthly board report"
5. Visibility: Private / Team / Organization

**Step 2: Add Widgets**

**Top Row - Key Metrics** (4 metric cards):
1. Total Scope 3 Emissions
2. vs. Target (% progress)
3. Emission Intensity (per $M revenue)
4. Data Quality Score

**Second Row - Trends** (2 large charts):
1. 12-Month Emission Trend (line chart)
   - Configure: Last 12 months, monthly granularity
   - Compare to prior year
2. Category Breakdown (donut chart)
   - Top 5 categories
   - Others grouped

**Third Row - Priorities** (2 medium widgets):
1. Top 10 Hotspots (table)
   - Rank, Description, Potential, Status
2. Supplier Engagement Progress (stacked bar)
   - PCF Received vs. In Progress vs. Not Contacted

**Fourth Row - Insights** (1 full-width widget):
1. AI-Generated Insights (text box)
   - Auto-updates with latest analysis
   - Key trends and recommendations

[Screenshot: Custom executive dashboard layout]

**Step 3: Configure Widgets**

**For Each Widget**:
1. Click gear icon
2. Set data source (auto-detect or manual select)
3. Choose visualization options:
   - Colors (match brand)
   - Labels and titles
   - Axes and scales
   - Legend position
4. Set refresh rate:
   - Real-time
   - Hourly
   - Daily
   - Manual only
5. Save configuration

**Step 4: Save Dashboard**
1. Click "Save Dashboard"
2. Set as default (optional)
3. Share with team (optional)

**💡 Tip**: Create role-specific dashboards. Executives need different views than analysts.

### Dashboard Templates

**Pre-Built Templates**:

**Executive Dashboard**:
- High-level KPIs
- Trends and targets
- Minimal detail, maximum insight

**Analyst Dashboard**:
- Detailed data tables
- Multiple drill-down options
- Data quality focus

**Procurement Dashboard**:
- Supplier-centric views
- Category analysis
- Spend correlation

**Operations Dashboard**:
- Hotspot focus
- Reduction initiatives tracking
- Operational metrics

**Investor Relations Dashboard**:
- IFRS S2 metrics
- Financial materiality
- Scenario analysis

**Use Template**:
1. Click "New Dashboard from Template"
2. Select template
3. Customize as needed
4. Save as your own

[Screenshot: Dashboard template gallery]

---

## Section 9: Sharing and Collaboration

### Share Dashboard

**Share Options**:

**1. Share Link** (View-Only):
- Click "Share" button
- Copy link
- Set permissions:
  - Anyone with link (public)
  - Organization only
  - Specific users
- Set expiration (optional)
- Recipient views dashboard in browser (no login required)

**2. Scheduled Email**:
- Click "Share" > "Schedule Email"
- Recipients: email list
- Frequency: Daily, Weekly, Monthly
- Time: Specify time and timezone
- Format: PDF attachment or link
- Include: Dashboard snapshot + optional commentary

**3. Embed in Website/Portal**:
- Click "Share" > "Embed Code"
- Copy iframe code
- Paste into website or internal portal
- Responsive design auto-adjusts to container
- Use for: Intranet, sustainability webpage, investor portal

**4. Export and Share**:
- Export as PDF or PowerPoint
- Send via email
- Static snapshot (doesn't update)

[Screenshot: Share dialog with options]

**💡 Tip**: Use scheduled email for regular updates (e.g., monthly executive report). Recipients always get latest data.

### Collaborative Features

**Comments and Annotations**:
1. Click "Add Comment" button
2. Click any widget to comment on
3. Type comment
4. @ mention team members
5. Comments appear as icons on dashboard
6. Click to view conversation thread

**Use Cases**:
- Ask questions about data
- Provide context for anomalies
- Request clarifications
- Document decisions

[Screenshot: Comment thread on dashboard]

**Dashboard Versioning**:
- Platform auto-saves versions
- View version history
- Restore previous version
- Compare versions side-by-side

**Real-Time Collaboration**:
- See who's viewing dashboard (presence indicators)
- Synchronized view (everyone sees same filters)
- Useful for meetings and presentations

### Access Control

**Permission Levels**:

**Viewer**:
- View dashboards
- Apply filters
- Export data
- Cannot edit or share

**Editor**:
- All Viewer permissions
- Customize dashboards
- Create new dashboards
- Share with others

**Admin**:
- All Editor permissions
- Manage access permissions
- Delete dashboards
- Set organization defaults

**Set Permissions**:
1. Click "Share" > "Manage Access"
2. Add users or groups
3. Assign permission level
4. Save

[Screenshot: Access control panel]

**💡 Tip**: Create team dashboards with Editor access for collaboration, but limit Admin access to prevent accidental deletion.

---

## Section 10: Exporting Dashboard Data

### Export Formats

**1. PDF Export**:
- **Use Case**: Presentations, executive reports, print
- **Options**:
  - Portrait or landscape
  - Include all widgets or selected
  - Add cover page and notes
  - Apply branding
- **Output**: Static snapshot, high-quality graphics

**2. PowerPoint Export**:
- **Use Case**: Editable presentations, combine with other slides
- **Options**:
  - One slide per widget
  - Or full dashboard on one slide
  - Editable text and charts
- **Output**: .pptx file

**3. Excel Export**:
- **Use Case**: Further analysis, pivot tables, custom charts
- **Options**:
  - Raw data tables
  - One sheet per widget
  - Include formulas
- **Output**: .xlsx file with data and metadata

**4. CSV Export**:
- **Use Case**: Import to other systems, BI tools, databases
- **Options**:
  - Selected dataset
  - Apply current filters
  - Choose delimiter
- **Output**: .csv file

**5. Image Export** (Individual Widgets):
- **Use Case**: Insert into documents, web pages
- **Formats**: PNG (raster), SVG (vector)
- **Resolution**: Screen (150 DPI) or Print (300 DPI)
- **Transparency**: Optional transparent background

[Screenshot: Export format selection dialog]

### Export Workflow

**Step-by-Step**:

1. **Navigate to desired dashboard**
2. **Apply filters** to show exactly what you want to export
3. **Click "Export" button** (top-right)
4. **Select format**
5. **Configure options**:
   - **Date Range**: Confirm or adjust
   - **Filters**: Confirm active filters included
   - **Widgets**: Select all or specific widgets
   - **Branding**: Include logo and colors
   - **Notes**: Add commentary or context
6. **Preview** (optional)
7. **Click "Generate Export"**
8. **Download file** (typically ready in 10-30 seconds)

**💡 Tip**: Export filtered views for targeted reports (e.g., "Category 1 only" or "Top 10 Suppliers").

### Scheduled Exports

**Automate Regular Exports**:

1. **Click "Export" > "Schedule Export"**
2. **Configure export**:
   - Format: PDF, Excel, or both
   - Frequency: Daily, Weekly, Monthly, Quarterly
   - Day and Time: When to generate
   - Filters: Fixed or rolling (e.g., "Last 30 Days")
3. **Set delivery**:
   - Email to recipients
   - Or save to cloud storage (Google Drive, OneDrive, SharePoint)
4. **Save schedule**

**Use Cases**:
- Monthly executive reports
- Weekly team updates
- Quarterly board materials
- Annual compliance documentation

[Screenshot: Scheduled export configuration]

**💡 Tip**: Schedule exports to run just after month-end data uploads complete. Ensure freshest data in reports.

### Data API Access

**For Advanced Users**:

Platform provides RESTful API for programmatic data access:

**API Endpoints**:
- `/api/v1/emissions` - Emission data
- `/api/v1/suppliers` - Supplier data
- `/api/v1/categories` - Category breakdowns
- `/api/v1/hotspots` - Hotspot analysis

**Authentication**: Bearer token (generate in Settings > API Access)

**Example Request**:
```bash
curl -X GET \
  'https://api.vcci.greenlang.io/v1/emissions?period=2024-Q4' \
  -H 'Authorization: Bearer YOUR_API_KEY'
```

**Example Response**:
```json
{
  "period": "2024-Q4",
  "total_emissions": 45230,
  "unit": "tCO2e",
  "categories": [
    {
      "category": 1,
      "emissions": 28100,
      "quality": 3.2
    }
  ]
}
```

**Use Cases**:
- Integration with BI tools (Tableau, Power BI)
- Custom applications
- Data warehouse loading
- Advanced analytics

**Documentation**: `https://api.greenlang.io/docs`

---

## Section 11: Alerts and Notifications

### Setting Up Alerts

**Create Alert**:
1. **From Dashboard**: Click "Create Alert" button
2. **Select Metric**: What to monitor
   - Total Scope 3 emissions
   - Specific category
   - Data quality score
   - Supplier engagement status
   - Any dashboard KPI
3. **Define Condition**:
   - Increases by >X%
   - Decreases by >X%
   - Exceeds threshold (absolute value)
   - Falls below threshold
   - Changes from previous period
4. **Set Notification**:
   - Email
   - In-platform notification
   - Slack/Teams integration
   - SMS (for critical alerts)
5. **Choose Recipients**:
   - Yourself
   - Team distribution list
   - Specific stakeholders
6. **Set Frequency**:
   - Immediate (when condition met)
   - Daily digest (if any alerts triggered)
   - Weekly summary
7. **Save Alert**

[Screenshot: Alert creation wizard]

### Alert Examples

**Alert 1: Emission Spike**
- **Metric**: Total Scope 3 Emissions
- **Condition**: Increases by >10% month-over-month
- **Why**: Identify unexpected increases early
- **Action**: Investigate cause (data error, business change, etc.)

**Alert 2: Data Quality Drop**
- **Metric**: Overall Data Quality Score
- **Condition**: Falls below 3.0
- **Why**: Maintain minimum quality for reporting
- **Action**: Review data uploads, engage suppliers

**Alert 3: Target Risk**
- **Metric**: Progress toward annual target
- **Condition**: Behind pace (e.g., need 8% reduction by now, only 3% achieved)
- **Why**: Early warning to intensify reduction efforts
- **Action**: Accelerate reduction initiatives

**Alert 4: Supplier Non-Response**
- **Metric**: Supplier engagement status
- **Condition**: No response after 14 days
- **Why**: Timely follow-up improves response rate
- **Action**: Send reminder, escalate if needed

**Alert 5: Hotspot Opportunity**
- **Metric**: New hotspots identified
- **Condition**: Hotspot with >1,000 tCO2e reduction potential
- **Why**: Don't miss high-impact opportunities
- **Action**: Evaluate and prioritize

[Screenshot: Alert notification examples]

### Managing Alerts

**View All Alerts**:
- Navigate to Settings > Alerts
- See all active alerts
- View alert history (triggered vs. not triggered)

**Actions**:
- **Pause**: Temporarily disable
- **Edit**: Change conditions or recipients
- **Delete**: Remove alert
- **Test**: Manually trigger to verify notification works

**Alert Performance**:
- True positive rate (alerts correctly identifying issues)
- False positive rate (alerts triggering unnecessarily)
- Adjust thresholds to minimize false positives

**💡 Tip**: Start with conservative thresholds, then tighten as you understand normal variability.

### Notification Center

**In-Platform Notifications**:
- Bell icon (top-right) shows unread count
- Click to view notification feed
- Categories:
  - 🔴 Alerts (conditions triggered)
  - 📊 Data uploads complete
  - ✅ Reports generated
  - 💬 Comments and mentions
  - 📥 Supplier responses
  - 🔄 System updates

**Notification Actions**:
- Click to view related dashboard/report
- Mark as read
- Dismiss
- Snooze (remind later)

[Screenshot: Notification center dropdown]

---

## Troubleshooting

### Common Issues and Solutions

**Issue**: Dashboard shows "No Data Available"
**Solution**:
1. Check date range filter - may be set to period with no data
2. Verify data upload completed (Data > Upload History)
3. Check business unit filter - may have excluded all units
4. Verify category filter - may have deselected all categories
5. Click "Reset Filters" to clear all filters
6. Refresh dashboard (F5 or refresh button)

**Issue**: Dashboard loading slowly or timing out
**Solution**:
1. Reduce date range (e.g., 1 quarter instead of 5 years)
2. Limit number of widgets (fewer widgets = faster load)
3. Check internet connection speed
4. Clear browser cache
5. Try different browser (Chrome recommended)
6. Check platform status page: status.greenlang.io
7. Contact support if persistent

**Issue**: Charts not displaying correctly
**Solution**:
1. Refresh page
2. Try different browser
3. Disable browser extensions (especially ad blockers)
4. Enable JavaScript
5. Update browser to latest version
6. Check if WebGL is enabled (for advanced visualizations)
7. Clear browser cache and cookies

**Issue**: Export fails or downloads corrupt file
**Solution**:
1. Reduce export scope (fewer widgets or shorter date range)
2. Try different format (PDF vs. Excel)
3. Disable VPN (sometimes interferes with large downloads)
4. Check available disk space
5. Try incognito/private browsing mode
6. Contact support with error message

**Issue**: Dashboard changes not saving
**Solution**:
1. Verify you have Editor permissions (not just Viewer)
2. Check if dashboard is locked (finalized dashboards can't be edited)
3. Ensure you clicked "Save" not just "Close"
4. Try refreshing and making changes again
5. Check browser console for errors (F12 > Console tab)

**Issue**: Filters not working correctly
**Solution**:
1. Clear all filters and reapply one at a time
2. Check for conflicting filters (e.g., Category 1 AND Category 2)
3. Verify filter logic (AND vs. OR)
4. Some filters require data refresh - click "Apply Filters"
5. Reset to default filters and try again

**Issue**: Drill-down not working (click has no effect)
**Solution**:
1. Verify widget supports drill-down (some summary widgets don't)
2. Check if maximum drill-down level reached
3. Ensure you're clicking chart element, not white space
4. Try right-click > "Drill Down" from context menu
5. Check browser console for JavaScript errors

**Issue**: Shared dashboard link not working for recipient
**Solution**:
1. Verify link hasn't expired (check expiration settings)
2. Check recipient has necessary permissions
3. Ensure organization settings allow external sharing (if external recipient)
4. Try generating new share link
5. Check recipient's browser compatibility
6. Send PDF export as alternative

---

## FAQ

**Q: How often do dashboards update?**
**A**: Real-time. As soon as new data is uploaded or changes are made, dashboards reflect the updates immediately. No manual refresh needed (though you can force refresh with F5 or refresh button).

**Q: Can I access dashboards on mobile?**
**A**: Yes. Dashboards are responsive and work on mobile browsers. For best experience:
- Use landscape orientation
- Tap-and-hold for drill-downs
- Pinch-to-zoom on charts
- Some complex interactions easier on desktop

**Q: How many custom dashboards can I create?**
**A**: Unlimited. Create as many as you need for different audiences, use cases, or analyses. Organize with folders and tags.

**Q: Can I schedule dashboard to refresh automatically?**
**A**: Dashboards auto-refresh when underlying data changes. For presentations, use "Auto-Refresh" mode to update every N seconds during meetings.

**Q: Why don't my emissions match between dashboards?**
**A**: Check filters. Different dashboards may have different default filters (date range, categories, business units). Use "Sync Filters Across Dashboards" setting to keep consistent.

**Q: Can I copy widgets between dashboards?**
**A**: Yes. In customize mode:
1. Right-click widget
2. Select "Copy to Another Dashboard"
3. Choose target dashboard
4. Widget is duplicated with same configuration

**Q: How do I print a dashboard?**
**A**: Best practice: Export to PDF first, then print. Direct browser print works but may have formatting issues. PDF export optimizes for print layout.

**Q: Can I embed a dashboard in PowerPoint for live data?**
**A**: Sort of. PowerPoint doesn't support live embeds, but you can:
- Take screenshot and insert as image (static)
- Insert hyperlink to dashboard (requires internet)
- Use PowerPoint export for editable charts (static)

**Q: Why is my custom dashboard missing after logging out?**
**A**: Check visibility setting. If set to "Private", only you can see it. If dashboard was shared by someone else and they deleted it, it's gone. Save important dashboards to your account.

**Q: Can I benchmark my dashboard metrics against industry?**
**A**: Yes. Enable benchmarking in Settings > Benchmarking. Dashboards will show anonymized peer comparison where applicable (e.g., emission intensity vs. industry average).

---

## Related Resources

### Platform Documentation
- [Getting Started Guide](./GETTING_STARTED.md) - Platform overview
- [Data Upload Guide](./DATA_UPLOAD_GUIDE.md) - Prepare data for dashboards
- [Reporting Guide](./REPORTING_GUIDE.md) - Generate reports from dashboard insights
- [Supplier Portal Guide](./SUPPLIER_PORTAL_GUIDE.md) - Improve data quality via supplier engagement

### Video Tutorials
- "Dashboard Navigation" (8 min) - Basic navigation and filtering
- "Creating Custom Dashboards" (15 min) - Build personalized views
- "Advanced Analytics" (20 min) - Drill-downs and correlations
- "Sharing and Collaboration" (10 min) - Share insights with stakeholders

### Dashboard Templates
- Executive Dashboard Template
- Analyst Dashboard Template
- Procurement Dashboard Template
- Operations Dashboard Template
- Download from: Dashboards > New Dashboard > From Template

### Support
- **Help Center**: Click "?" icon > Dashboard Help
- **Interactive Tutorial**: First-time users guided through dashboard
- **Live Chat**: Mon-Fri 9am-5pm EST
- **Email Support**: dashboard-support@greenlang.io
- **Community Forum**: `https://community.greenlang.io/dashboards`

### External Resources
- **Data Visualization Best Practices**: Principles for effective dashboards
- **Carbon Accounting Fundamentals**: Understanding emission metrics
- **Dashboard Design Patterns**: Industry standards for KPI dashboards

---

**Dashboard Success Tips**:
1. ✅ Start with Overview Dashboard - get big picture first
2. ✅ Use drill-downs - don't settle for summary, investigate root causes
3. ✅ Apply filters liberally - focus on specific questions
4. ✅ Create custom dashboards for recurring analyses
5. ✅ Share insights - dashboards are for communication, not just analysis
6. ✅ Set up alerts - proactive monitoring beats reactive firefighting
7. ✅ Review weekly - regular dashboard review drives continuous improvement

**Remember**: Dashboards are tools for insight, not just data display. Ask questions, investigate anomalies, and drive action.

For questions: dashboard-support@greenlang.io

---

*Last Updated: 2025-11-07*
*Document Version: 1.0*
*Platform Version: 2.5.0*
