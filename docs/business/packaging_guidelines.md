# GreenLang Packaging Guidelines

**Version:** 1.0
**Date:** 2025-12-07
**Document Owner:** GL-ProductManager
**Status:** APPROVED

---

## Executive Summary

This document defines the packaging strategies, bundling rules, and guidelines for assembling GreenLang products into market-ready offerings. Proper packaging ensures customers receive coherent solutions that address their complete needs while maximizing value and simplifying purchasing decisions.

---

## 1. Packaging Philosophy

### 1.1 Core Principles

1. **Customer-Centric**: Package products around customer problems, not technical features
2. **Clear Value**: Each package must have a clear, quantifiable value proposition
3. **Logical Progression**: Packages should form natural upgrade paths
4. **Simplicity**: Minimize complexity in package structures
5. **Flexibility**: Allow customization without fragmenting the product line

### 1.2 Packaging Hierarchy

```
Platform (GreenLang Climate Intelligence Platform)
    |
    +-- Editions (Essentials, Professional, Enterprise)
    |       |
    |       +-- Applications (CSRD, CBAM, ProcessHeat, etc.)
    |
    +-- Solution Bundles (Industry-specific, Use-case specific)
    |       |
    |       +-- Applications + Agents + Services
    |
    +-- Add-On Modules (Feature expansions)
    |
    +-- Professional Services (Implementation, Training, Custom)
```

---

## 2. Product Bundling Strategies

### 2.1 Core Platform Bundle

The foundation every customer receives, regardless of edition.

| Component | Description | Included In |
|-----------|-------------|-------------|
| GreenLang Platform Core | Base infrastructure, authentication, API | All editions |
| Data Management | Import, export, storage, basic analytics | All editions |
| User Management | Users, roles, basic permissions | All editions |
| Standard Dashboards | Pre-built visualization templates | All editions |
| Basic Integrations | CSV import, standard API access | All editions |
| Documentation & Training | Online docs, video tutorials | All editions |
| Community Support | Forum access, knowledge base | All editions |

### 2.2 Edition Bundles

#### 2.2.1 Essentials Edition Bundle

**Target:** Organizations starting their sustainability journey

| Bundle Component | Quantity | Value |
|-----------------|----------|-------|
| Core Platform | Full | $40,000 |
| Monitoring Agents | 10 | $20,000 |
| Optimization Agents (Basic) | 5 | $15,000 |
| Compliance Agents (Basic) | 5 | $15,000 |
| Analytics Agents (Basic) | 5 | $10,000 |
| Standard Integrations | 5 | $10,000 |
| Business Hours Support | 1 year | $10,000 |
| **Bundle Total Value** | - | **$120,000** |
| **Bundle Price** | - | **$100,000** |
| **Customer Savings** | - | **17%** |

#### 2.2.2 Professional Edition Bundle

**Target:** Growing organizations needing advanced capabilities

| Bundle Component | Quantity | Value |
|-----------------|----------|-------|
| Core Platform | Full | $40,000 |
| All Essentials Components | Full | $60,000 |
| Monitoring Agents | 40 | $60,000 |
| Optimization Agents (Advanced) | 30 | $60,000 |
| Compliance Agents (Full) | 20 | $40,000 |
| Analytics Agents (Advanced) | 10 | $30,000 |
| ML Optimization Engine | Full | $50,000 |
| Advanced Integrations | 15 | $30,000 |
| 24x5 Support | 1 year | $25,000 |
| Dedicated CSM (Shared) | 1 year | $15,000 |
| **Bundle Total Value** | - | **$410,000** |
| **Bundle Price** | - | **$250,000** |
| **Customer Savings** | - | **39%** |

#### 2.2.3 Enterprise Edition Bundle

**Target:** Large organizations with complex requirements

| Bundle Component | Quantity | Value |
|-----------------|----------|-------|
| Core Platform | Full + Enhanced | $60,000 |
| All Professional Components | Full | $250,000 |
| Complete Agent Library | 500+ | $200,000 |
| Agent Factory Access | Full | $100,000 |
| Custom Integration Hours | 40/year | $20,000 |
| Enterprise Integrations | Unlimited | $50,000 |
| 24x7 Priority Support | 1 year | $50,000 |
| Dedicated TAM | 1 year | $40,000 |
| Quarterly Business Reviews | 4/year | $20,000 |
| **Bundle Total Value** | - | **$790,000** |
| **Bundle Price** | - | **$500,000** |
| **Customer Savings** | - | **37%** |

### 2.3 Application Bundles

#### 2.3.1 Regulatory Compliance Bundle

**Includes:** CSRD + CBAM + EUDR + EU Taxonomy

| Component | Standalone Price | Bundle Price |
|-----------|-----------------|--------------|
| GL-CSRD-APP | $250,000 | - |
| GL-CBAM-APP | $200,000 | - |
| GL-EUDR-APP | $150,000 | - |
| GL-Taxonomy-APP | $100,000 | - |
| **Total Standalone** | **$700,000** | - |
| **Bundle Price** | - | **$490,000** |
| **Customer Savings** | - | **30%** |

#### 2.3.2 Industrial Decarbonization Bundle

**Includes:** ProcessHeat + HeatRecovery + EnergyManagement

| Component | Standalone Price | Bundle Price |
|-----------|-----------------|--------------|
| GL-ProcessHeat-APP | $350,000 | - |
| GL-HeatRecovery-APP | $200,000 | - |
| GL-EnergyManagement-APP | $250,000 | - |
| **Total Standalone** | **$800,000** | - |
| **Bundle Price** | - | **$600,000** |
| **Customer Savings** | - | **25%** |

#### 2.3.3 Smart Buildings Bundle

**Includes:** SmartBuilding + HVAC + Lighting + Analytics

| Component | Standalone Price | Bundle Price |
|-----------|-----------------|--------------|
| GL-SmartBuilding-APP | $400,000 | - |
| GL-HVAC-Optimizer | $200,000 | - |
| GL-Lighting-Control | $100,000 | - |
| GL-Building-Analytics | $150,000 | - |
| **Total Standalone** | **$850,000** | - |
| **Bundle Price** | - | **$600,000** |
| **Customer Savings** | - | **29%** |

---

## 3. Module Packaging Rules

### 3.1 Module Classification

| Module Type | Description | Pricing Model | Can Be Standalone |
|-------------|-------------|---------------|-------------------|
| **Core Module** | Essential platform functionality | Included in edition | No |
| **Feature Module** | Extends core capabilities | Add-on pricing | No |
| **Application Module** | Complete application | Standalone or bundle | Yes |
| **Integration Module** | Connects to external systems | Per-system pricing | No |
| **Agent Module** | Additional AI agents | Per-agent or pack | Depends |

### 3.2 Module Dependency Rules

```
Rules:
1. Core modules cannot be removed or disabled
2. Feature modules require Professional or Enterprise edition
3. Application modules require minimum edition level
4. Integration modules require compatible application
5. Agent modules require minimum agent capacity

Dependency Graph:
Core Platform (Required)
    |
    +-- Data Management (Required)
    |       |
    |       +-- Advanced Analytics (Pro+)
    |               |
    |               +-- ML Engine (Pro+)
    |                       |
    |                       +-- Custom ML (Ent)
    |
    +-- User Management (Required)
    |       |
    |       +-- SSO/SAML (Pro+)
    |               |
    |               +-- Custom Roles (Ent)
    |
    +-- Compliance Framework (Required)
            |
            +-- Multi-Regulation (Pro+)
                    |
                    +-- Custom Frameworks (Ent)
```

### 3.3 Module Compatibility Matrix

| Module | Essentials | Professional | Enterprise | Prerequisites |
|--------|:----------:|:------------:|:----------:|---------------|
| Advanced Analytics | - | Yes | Yes | None |
| ML Optimization | - | Yes | Yes | Advanced Analytics |
| Custom ML Models | - | - | Yes | ML Optimization |
| Agent Factory | - | - | Yes | None |
| Edge Computing | - | - | Yes | None |
| Real-Time Streaming | - | Yes | Yes | None |
| Custom Integrations | - | - | Yes | None |
| White-Label | - | - | Yes | None |
| On-Premise | - | - | Yes | None |
| Multi-Tenant | - | Yes | Yes | None |

---

## 4. Add-On Module Guidelines

### 4.1 Add-On Categories

#### 4.1.1 Capacity Add-Ons

Expand usage limits beyond edition defaults.

| Add-On | Essentials | Professional | Enterprise | Price |
|--------|------------|--------------|------------|-------|
| +10 Users | Yes | Yes | N/A (unlimited) | $1,000/user/year |
| +100K Data Points | Yes | Yes | N/A (unlimited) | $500/month |
| +5 Dashboards | Yes | Yes | N/A (unlimited) | $200/dashboard/year |
| +1 Facility | Yes | Yes | N/A (unlimited) | $5,000/facility/year |

#### 4.1.2 Feature Add-Ons

Unlock capabilities beyond edition defaults.

| Add-On | Essentials | Professional | Enterprise | Price |
|--------|------------|--------------|------------|-------|
| ML Optimization Engine | +$50,000 | Included | Included | $50,000/year |
| Advanced Reporting | +$25,000 | Included | Included | $25,000/year |
| API Write Access | +$15,000 | Included | Included | $15,000/year |
| Custom Dashboards | +$10,000 | Included | Included | $10,000/year |
| Multi-Language | +$5,000/lang | 5 included | All included | $5,000/lang/year |

#### 4.1.3 Agent Add-Ons

Purchase additional agents beyond edition allocation.

| Add-On | Unit Price | Pack of 10 | Pack of 50 | Pack of 100 |
|--------|------------|------------|------------|-------------|
| Monitoring Agents | $2,000 | $18,000 | $80,000 | $150,000 |
| Optimization Agents | $3,000 | $27,000 | $120,000 | $225,000 |
| Compliance Agents | $2,500 | $22,500 | $100,000 | $187,500 |
| Analytics Agents | $2,000 | $18,000 | $80,000 | $150,000 |

#### 4.1.4 Support Add-Ons

Enhance support beyond edition defaults.

| Add-On | Essentials | Professional | Price |
|--------|------------|--------------|-------|
| Upgrade to 24x5 | Yes | N/A | $15,000/year |
| Upgrade to 24x7 | Yes | Yes | $25,000/year |
| Dedicated CSM | Yes | Upgrade | $25,000/year |
| Technical Account Manager | N/A | N/A | Enterprise only |
| Extended Training | Yes | Yes | $5,000/day |

### 4.2 Add-On Packaging Rules

1. **Minimum Order**: Some add-ons have minimum quantities
2. **Edition Requirements**: Certain add-ons require specific editions
3. **Bundle Restrictions**: Some add-ons cannot be combined
4. **Term Alignment**: Add-on terms must align with subscription terms
5. **Renewal Pricing**: Add-ons renew at current list price

### 4.3 Add-On Discount Rules

| Add-On Value | Discount |
|--------------|----------|
| $10,000 - $24,999 | 0% |
| $25,000 - $49,999 | 5% |
| $50,000 - $99,999 | 10% |
| $100,000 - $199,999 | 15% |
| $200,000+ | Negotiate |

---

## 5. Industry Vertical Solution Bundles

### 5.1 Manufacturing Solutions

#### 5.1.1 Steel Manufacturing Bundle

**Target:** Steel producers and processors

| Component | Description | Value |
|-----------|-------------|-------|
| GL-ProcessHeat-APP | Core process optimization | $350,000 |
| Steel-Specific Agents (20) | EAF, blast furnace, rolling | $100,000 |
| CBAM Compliance Module | Carbon border reporting | $100,000 |
| Energy Management | Real-time optimization | $150,000 |
| Implementation Services | 12-week deployment | $75,000 |
| **Total Value** | - | **$775,000** |
| **Bundle Price** | - | **$550,000** |
| **Savings** | - | **29%** |

#### 5.1.2 Chemical Manufacturing Bundle

**Target:** Chemical producers and processors

| Component | Description | Value |
|-----------|-------------|-------|
| GL-ProcessHeat-APP | Core process optimization | $350,000 |
| Chemical-Specific Agents (25) | Reactors, distillation | $125,000 |
| Safety Compliance (PSM/OSHA) | Process safety management | $100,000 |
| Emissions Monitoring | Real-time tracking | $100,000 |
| Implementation Services | 16-week deployment | $100,000 |
| **Total Value** | - | **$775,000** |
| **Bundle Price** | - | **$575,000** |
| **Savings** | - | **26%** |

#### 5.1.3 Food & Beverage Bundle

**Target:** Food processors and beverage manufacturers

| Component | Description | Value |
|-----------|-------------|-------|
| GL-FoodBev-APP | Industry-specific optimization | $300,000 |
| Refrigeration Optimization | Cold chain management | $100,000 |
| Water Management | Usage and treatment | $75,000 |
| CSRD Compliance | Sustainability reporting | $150,000 |
| Implementation Services | 10-week deployment | $50,000 |
| **Total Value** | - | **$675,000** |
| **Bundle Price** | - | **$475,000** |
| **Savings** | - | **30%** |

### 5.2 Buildings Solutions

#### 5.2.1 Commercial Office Bundle

**Target:** Class A/B office buildings and portfolios

| Component | Description | Value |
|-----------|-------------|-------|
| GL-SmartBuilding-APP | Complete building platform | $400,000 |
| HVAC Optimization Suite | ML-powered HVAC control | $150,000 |
| Occupancy Analytics | Space utilization | $75,000 |
| LEED/WELL Support | Certification tracking | $50,000 |
| Implementation Services | 8-week deployment | $40,000 |
| **Total Value** | - | **$715,000** |
| **Bundle Price** | - | **$500,000** |
| **Savings** | - | **30%** |

#### 5.2.2 Healthcare Facilities Bundle

**Target:** Hospitals, medical centers, long-term care

| Component | Description | Value |
|-----------|-------------|-------|
| GL-SmartBuilding-APP | Complete building platform | $400,000 |
| Healthcare-Specific Agents | OR, patient rooms, labs | $150,000 |
| Air Quality Management | Infection control | $100,000 |
| Regulatory Compliance | JCAHO, CMS requirements | $100,000 |
| Implementation Services | 12-week deployment | $75,000 |
| **Total Value** | - | **$825,000** |
| **Bundle Price** | - | **$600,000** |
| **Savings** | - | **27%** |

#### 5.2.3 Retail & Hospitality Bundle

**Target:** Retail chains, hotels, restaurants

| Component | Description | Value |
|-----------|-------------|-------|
| GL-SmartBuilding-APP | Multi-site management | $350,000 |
| Retail-Specific Agents | Stores, refrigeration | $100,000 |
| Guest Comfort Module | Hospitality optimization | $75,000 |
| Portfolio Management | Multi-site analytics | $100,000 |
| Implementation Services | 8-week deployment | $40,000 |
| **Total Value** | - | **$665,000** |
| **Bundle Price** | - | **$450,000** |
| **Savings** | - | **32%** |

### 5.3 Financial Services Solutions

#### 5.3.1 Bank/Insurance Climate Risk Bundle

**Target:** Banks, insurers, asset managers

| Component | Description | Value |
|-----------|-------------|-------|
| GL-ClimateRisk-APP | Physical and transition risk | $500,000 |
| Portfolio Carbon Analytics | Investment footprinting | $200,000 |
| TCFD Reporting Suite | Disclosure automation | $150,000 |
| Regulatory Compliance | SEC, SFDR, MiFID | $150,000 |
| Implementation Services | 12-week deployment | $100,000 |
| **Total Value** | - | **$1,100,000** |
| **Bundle Price** | - | **$800,000** |
| **Savings** | - | **27%** |

### 5.4 Supply Chain Solutions

#### 5.4.1 Scope 3 Management Bundle

**Target:** Large enterprises with complex supply chains

| Component | Description | Value |
|-----------|-------------|-------|
| GL-Scope3-APP | Complete Scope 3 platform | $400,000 |
| Supplier Engagement Portal | Data collection | $150,000 |
| Product Carbon Footprint | LCA calculations | $150,000 |
| SBTi/CDP Integration | Target setting, reporting | $100,000 |
| Implementation Services | 16-week deployment | $100,000 |
| **Total Value** | - | **$900,000** |
| **Bundle Price** | - | **$650,000** |
| **Savings** | - | **28%** |

---

## 6. Bundle Configuration Rules

### 6.1 Valid Bundle Combinations

| Base Edition | Can Add Applications | Can Add Industry Bundles | Can Add Add-Ons |
|--------------|---------------------|-------------------------|-----------------|
| Essentials | Limited (2) | Limited (1) | Yes (with limits) |
| Professional | Yes (5) | Yes (2) | Yes |
| Enterprise | Unlimited | Unlimited | Unlimited |

### 6.2 Bundle Stacking Rules

1. **Edition + Applications**: Applications stack on edition
2. **Edition + Industry Bundle**: Industry bundles include edition requirements
3. **Multiple Industry Bundles**: Requires Enterprise edition
4. **Add-Ons**: Stack on any base configuration

### 6.3 Discount Stacking Rules

| Discount Type | Can Stack With |
|---------------|----------------|
| Edition Discount | Multi-year only |
| Bundle Discount | Volume discount only |
| Volume Discount | All except bundle |
| Multi-Year Discount | All |
| Partner Discount | Volume, multi-year |

### 6.4 Maximum Discount Caps

| Customer Type | Maximum Discount |
|---------------|------------------|
| Commercial | 35% |
| Government | 40% |
| Education/Non-Profit | 50% |
| Strategic Partner | Negotiated (approval required) |

---

## 7. Packaging Best Practices

### 7.1 Sales Team Guidelines

1. **Start with customer needs**: Identify pain points before proposing packages
2. **Right-size the solution**: Don't over-sell or under-sell
3. **Lead with value**: Focus on outcomes, not features
4. **Build for growth**: Position upgrade paths early
5. **Simplify proposals**: Limit to 2-3 options maximum

### 7.2 Common Packaging Mistakes

| Mistake | Impact | Correct Approach |
|---------|--------|------------------|
| Over-bundling | Customer confusion, longer sales cycles | Focus on core needs first |
| Under-bundling | Revenue leakage, poor adoption | Include implementation services |
| Wrong edition | Churn risk, support burden | Use assessment questionnaire |
| Missing add-ons | Underserved customer | Complete needs assessment |
| Excessive discounting | Margin erosion | Use value-based selling |

### 7.3 Proposal Structure

```
Standard Proposal Structure:

1. Executive Summary
   - Customer challenges
   - Proposed solution
   - Expected outcomes

2. Recommended Package
   - Edition selection rationale
   - Applications included
   - Add-ons included

3. Investment Summary
   - Base pricing
   - Discounts applied
   - Total investment

4. Implementation Timeline
   - Phases
   - Milestones
   - Resources required

5. ROI Analysis
   - Expected savings
   - Payback period
   - 5-year value
```

---

## 8. Package Lifecycle Management

### 8.1 Package Versioning

| Version | When Used | Duration |
|---------|-----------|----------|
| Current | Active sales | Until replaced |
| Previous | Existing customers | 12-month grace |
| Legacy | Long-term customers | Grandfathered |
| Preview | Early adopters | 90-day trial |

### 8.2 Package Retirement Process

1. **Notice Period**: 6 months minimum
2. **Migration Path**: Clear upgrade options
3. **Pricing Protection**: Existing customers maintain pricing
4. **Feature Transition**: Map retired features to new packages
5. **Communication**: Direct outreach to affected customers

### 8.3 New Package Introduction

1. **Market Research**: Validate need and pricing
2. **Internal Training**: Sales and support enablement
3. **Pilot Period**: 90-day controlled availability
4. **General Availability**: Full launch
5. **Performance Review**: 6-month assessment

---

## Appendix A: Package Quick Reference

### Package Naming Convention

```
[GL]-[Domain]-[Type]-[Tier/Version]

Examples:
- GL-CSRD-APP-PRO (CSRD Application, Professional tier)
- GL-MFG-BUNDLE-ENT (Manufacturing Bundle, Enterprise)
- GL-ML-MODULE-V2 (ML Module, Version 2)
```

### Package Codes

| Code | Package | Price |
|------|---------|-------|
| GL-ESS-001 | Essentials Edition | $100,000 |
| GL-PRO-001 | Professional Edition | $250,000 |
| GL-ENT-001 | Enterprise Edition | $500,000 |
| GL-REG-BDL-001 | Regulatory Compliance Bundle | $490,000 |
| GL-IND-BDL-001 | Industrial Decarbonization Bundle | $600,000 |
| GL-BLD-BDL-001 | Smart Buildings Bundle | $600,000 |
| GL-STL-BDL-001 | Steel Manufacturing Bundle | $550,000 |
| GL-CHM-BDL-001 | Chemical Manufacturing Bundle | $575,000 |
| GL-FNB-BDL-001 | Food & Beverage Bundle | $475,000 |
| GL-OFF-BDL-001 | Commercial Office Bundle | $500,000 |
| GL-HCF-BDL-001 | Healthcare Facilities Bundle | $600,000 |
| GL-RTL-BDL-001 | Retail & Hospitality Bundle | $450,000 |
| GL-FIN-BDL-001 | Financial Services Bundle | $800,000 |
| GL-SC3-BDL-001 | Scope 3 Management Bundle | $650,000 |

---

*Document End*

*Last Updated: 2025-12-07*
*Next Review: 2026-03-07*
