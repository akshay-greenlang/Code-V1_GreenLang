# GL-VCCI Scope 3 Transaction Upload Template Guide

**Version**: 1.0
**Last Updated**: 2025-11-07
**Platform**: GL-VCCI Carbon Intelligence Platform

---

## Table of Contents

1. [Introduction](#introduction)
2. [Template Overview](#template-overview)
3. [Field Definitions](#field-definitions)
4. [Data Validation Rules](#data-validation-rules)
5. [GHG Protocol Categories](#ghg-protocol-categories)
6. [Product Categories (UNSPSC)](#product-categories-unspsc)
7. [Country and Currency Codes](#country-and-currency-codes)
8. [Data Quality Best Practices](#data-quality-best-practices)
9. [Common Data Quality Issues](#common-data-quality-issues)
10. [Bulk Upload Tips](#bulk-upload-tips)
11. [Field-by-Field Examples](#field-by-field-examples)
12. [Industry-Specific Guidance](#industry-specific-guidance)
13. [Troubleshooting](#troubleshooting)
14. [API Integration](#api-integration)
15. [FAQ](#faq)

---

## Introduction

This guide provides comprehensive instructions for preparing and uploading Scope 3 carbon accounting transaction data to the GL-VCCI Carbon Intelligence Platform. The platform supports both CSV and JSON formats for maximum flexibility.

### Purpose

The transaction upload templates enable organizations to:
- Import procurement and supply chain data for carbon accounting
- Ensure data consistency and quality across uploads
- Automate Scope 3 emissions calculations
- Maintain audit trails for regulatory compliance
- Enable data-driven carbon reduction strategies

### Supported Formats

- **CSV**: Best for manual data entry and Excel exports
- **JSON**: Ideal for API integration and automated data feeds

### Prerequisites

Before uploading data, ensure you have:
- Valid organizational credentials and API access
- Supplier master data configured in the platform
- Understanding of GHG Protocol Scope 3 categories
- Product categorization system (UNSPSC or custom)
- Appropriate data access permissions

---

## Template Overview

### CSV Template Structure

The CSV template contains 13 core fields organized as follows:

```
transaction_id,date,supplier_name,supplier_id,product_name,product_category,
quantity,unit,spend_usd,currency,ghg_category,country,description
```

**File Requirements**:
- **Encoding**: UTF-8 with BOM
- **Delimiter**: Comma (,)
- **Quote Character**: Double quote (")
- **Escape Character**: Backslash (\)
- **Line Endings**: CRLF (Windows) or LF (Unix)
- **Maximum File Size**: 50 MB
- **Maximum Rows**: 50,000 per file

### JSON Template Structure

The JSON template provides additional capabilities:

```json
{
  "metadata": { ... },
  "transactions": [ ... ],
  "validation_config": { ... }
}
```

**File Requirements**:
- **Encoding**: UTF-8
- **Maximum File Size**: 100 MB
- **Maximum Transactions**: 10,000 per upload
- **Schema Version**: 1.0.0

---

## Field Definitions

### 1. transaction_id

**Type**: String (Alphanumeric)
**Required**: Yes
**Maximum Length**: 50 characters
**Uniqueness**: Must be unique within your organization

**Description**: A unique identifier for each transaction record. This ID is used to prevent duplicate uploads and enable transaction tracking throughout the platform.

**Naming Convention Recommendations**:
- Include year and sequential number: `TXN-2024-00001`
- Include department or category: `TXN-2024-MFG-00001`
- Use your ERP system's reference: `SAP-PO-12345678`

**Examples**:
```
TXN-2024-MFG-00001
PO-2024-AUG-1234
INV-STEEL-2024-Q3-001
SAP-4500123456
ORACLE-REQ-2024-9876
```

**Validation Rules**:
- Cannot contain commas, quotes, or newlines
- Cannot be empty or whitespace only
- Must be unique across all active transactions
- Case-sensitive comparison

**Common Errors**:
- ❌ Duplicate transaction IDs
- ❌ Special characters: `TXN-2024/001` (contains /)
- ❌ Too long: exceeds 50 characters
- ❌ Leading/trailing spaces

---

### 2. date

**Type**: Date
**Required**: Yes
**Format**: ISO 8601 (YYYY-MM-DD)

**Description**: The date when the transaction occurred. This is typically the invoice date, purchase order date, or goods receipt date.

**Valid Range**:
- **Minimum**: 5 years before current date
- **Maximum**: Current date (no future dates)

**Examples**:
```
2024-08-15
2024-01-01
2023-12-31
2020-06-15
```

**Validation Rules**:
- Must be valid calendar date
- Year must be 4 digits
- Month must be 01-12
- Day must be valid for the month
- Cannot be future dated
- Must be within reporting period window

**Common Errors**:
- ❌ Wrong format: `08/15/2024` or `15-08-2024`
- ❌ Invalid dates: `2024-02-30` (Feb 30th doesn't exist)
- ❌ Future dates: `2025-12-31`
- ❌ Two-digit years: `24-08-15`

**Best Practices**:
- Use invoice date for purchased goods
- Use goods receipt date for delivered items
- Use travel date for business travel
- Maintain consistency within reporting periods

---

### 3. supplier_name

**Type**: String
**Required**: Yes
**Maximum Length**: 200 characters

**Description**: The full legal name of the supplier or service provider. This should match the name in your supplier master data.

**Examples**:
```
Precision Steel Manufacturing Co.
ChemTech Solutions GmbH
Pacific Freight Logistics
GreenPower Energy Solutions
CloudCompute Services Inc.
Acme Corporation Ltd.
Global Industrial Supplies LLC
```

**Validation Rules**:
- Cannot be empty or whitespace only
- Should match supplier master data (if validation enabled)
- Special characters allowed: &, ., -, ', spaces
- No leading/trailing whitespace

**Best Practices**:
- Use official registered business names
- Include legal entity designation (LLC, Inc., GmbH, etc.)
- Maintain consistent naming across uploads
- Avoid abbreviations unless official
- Use title case or sentence case consistently

**Common Errors**:
- ❌ Inconsistent naming: "Acme Corp" vs "ACME CORPORATION"
- ❌ Abbreviations: "ABC Co." vs "ABC Company"
- ❌ Missing legal designations
- ❌ Typos and misspellings

---

### 4. supplier_id

**Type**: String (Alphanumeric)
**Required**: Yes
**Maximum Length**: 50 characters

**Description**: A unique identifier for the supplier in your organization's system. This enables supplier linkage and aggregation.

**Naming Convention Recommendations**:
- Use your ERP vendor/supplier IDs
- Maintain consistent format: `SUP-1001`
- Include supplier type prefix: `VENDOR-1001`

**Examples**:
```
SUP-1001
VENDOR-2024-001
SAP-V-123456
DUNS-123456789
GLN-1234567890123
```

**Validation Rules**:
- Must exist in supplier master (if validation enabled)
- Cannot be empty or whitespace only
- Case-sensitive

**Best Practices**:
- Use existing ERP supplier codes
- Register suppliers before uploading transactions
- Use DUNS or GLN numbers for standardization
- Maintain supplier master data quality

**Common Errors**:
- ❌ Supplier not registered in master data
- ❌ Inconsistent supplier IDs for same supplier
- ❌ Using supplier name instead of ID

---

### 5. product_name

**Type**: String
**Required**: Yes
**Maximum Length**: 500 characters

**Description**: Descriptive name or specification of the product or service purchased. Should be detailed enough to enable accurate carbon factor matching.

**Examples**:
```
Cold-rolled steel sheets - Grade 304
Industrial adhesive polymer - Type A200
Ocean freight container shipping - 40ft
Electricity - Renewable Energy Mix
Cloud infrastructure services - Compute
Business travel - International flights
Recycled cardboard packaging
Waste disposal and recycling services
HVAC maintenance and R-22 refrigerant
```

**Validation Rules**:
- Cannot be empty or whitespace only
- Should be descriptive for carbon factor matching
- Avoid generic names like "Materials" or "Services"

**Best Practices**:
- Include product specifications and grades
- Include service types and descriptions
- Use consistent naming conventions
- Include relevant details for emission factors
- Avoid vague descriptions

**Common Errors**:
- ❌ Too generic: "Materials"
- ❌ Missing specifications: "Steel" (what type?)
- ❌ Unclear services: "Consulting" (what kind?)
- ❌ Abbreviations without explanation

---

### 6. product_category

**Type**: String (UNSPSC Code)
**Required**: Yes
**Format**: UNSPSC classification code

**Description**: The United Nations Standard Products and Services Code (UNSPSC) for categorizing products and services. Used for emission factor matching and spend analysis.

**UNSPSC Hierarchy**:
- **Segment** (2 digits): Highest level (e.g., 33 = Metals)
- **Family** (2 digits): Second level (e.g., 10 = Ferrous metals)
- **Class** (2 digits): Third level (e.g., 15 = Steel)
- **Commodity** (2 digits): Lowest level (specific product)

**Format Examples**:
```
3310.15.10.00 (Steel sheets)
11121507 (Industrial adhesives)
78101803 (Ocean freight)
26111701 (Electricity)
81111800 (IT services)
```

**Common Categories by Industry**:

**Manufacturing**:
- `3310` - Ferrous metals and alloys
- `3010` - Non-ferrous metals
- `1112` - Adhesives and sealants
- `1210` - Plastic materials

**Logistics**:
- `7810` - Transportation services
- `7811` - Freight services
- `8611` - Warehousing and storage

**Energy**:
- `2611` - Electrical power
- `1511` - Fuels and additives
- `4010` - HVAC equipment

**Technology**:
- `8111` - Computer services
- `4321` - Computer equipment
- `4311` - Telecommunications equipment

**Best Practices**:
- Use most specific commodity code available
- Maintain code mapping documentation
- Validate codes against UNSPSC standard
- Use consistent codes for similar products

**Common Errors**:
- ❌ Invalid code format
- ❌ Using segment only (too broad)
- ❌ Mixing coding systems (NAICS, SIC with UNSPSC)

---

### 7. quantity

**Type**: Numeric (Decimal)
**Required**: Yes
**Precision**: Up to 6 decimal places

**Description**: The numeric quantity of the product or service purchased. Must be a positive number greater than zero.

**Examples**:
```
5000 (kg of steel)
500 (liters of chemicals)
12 (shipping containers)
150000 (kWh of electricity)
5000 (compute hours)
250 (kg freight weight)
15 (business trips)
```

**Validation Rules**:
- Must be positive (> 0)
- No negative values
- Maximum value: 999,999,999.999999
- Decimal separator: period (.)

**Best Practices**:
- Use appropriate precision for unit type
- Convert to standard units before upload
- Document conversion factors used
- Include fractional quantities where relevant

**Common Errors**:
- ❌ Negative quantities: `-100`
- ❌ Zero values: `0`
- ❌ Non-numeric: `approx 100`
- ❌ Comma decimal separator: `1,500.5` (use 1500.5)

---

### 8. unit

**Type**: String (Enumerated)
**Required**: Yes

**Description**: The unit of measurement for the quantity field. Must match the product type and enable accurate emissions calculations.

**Supported Units**:

**Mass**:
- `kg` - Kilograms
- `tonnes` - Metric tonnes (1000 kg)
- `lbs` - Pounds
- `mt` - Metric tons

**Volume**:
- `liters` - Liters
- `m3` - Cubic meters
- `gallons` - Gallons

**Energy**:
- `kWh` - Kilowatt-hours
- `MWh` - Megawatt-hours
- `GJ` - Gigajoules
- `BTU` - British Thermal Units

**Distance**:
- `km` - Kilometers
- `miles` - Miles
- `tkm` - Tonne-kilometers (freight)
- `passenger-km` - Passenger-kilometers

**Count**:
- `pieces` - Individual items
- `units` - Generic units
- `containers` - Shipping containers
- `trips` - Travel trips
- `hours` - Service hours
- `compute-hours` - Cloud computing hours

**Examples by Category**:
```
Raw Materials: kg, tonnes, liters, m3
Energy: kWh, MWh, GJ
Transportation: km, tkm, containers, trips
Services: hours, units, pieces
Cloud/IT: compute-hours, GB-hours
```

**Validation Rules**:
- Must be from approved unit list
- Case-sensitive (use lowercase)
- Must match quantity type

**Best Practices**:
- Use metric units where possible
- Maintain consistency within product categories
- Document unit conversion factors
- Use standard abbreviations

**Common Errors**:
- ❌ Non-standard units: "bags", "boxes"
- ❌ Wrong case: "KG" instead of "kg"
- ❌ Mismatch with quantity: services measured in "kg"

---

### 9. spend_usd

**Type**: Numeric (Currency)
**Required**: Yes
**Precision**: 2 decimal places

**Description**: The total spend amount for the transaction in US Dollars. If the original transaction was in another currency, convert to USD using appropriate exchange rates.

**Format**:
- Decimal separator: period (.)
- No currency symbols
- No thousands separators
- Positive values only
- Two decimal places: `12500.00`

**Examples**:
```
12500.00
8750.00
24000.00
18000.00
15000.00
5500.00
9600.00
```

**Validation Rules**:
- Must be positive (> 0)
- Maximum: 999,999,999.99
- Exactly 2 decimal places
- No currency symbols or formatting

**Currency Conversion**:
- Use exchange rate from transaction date
- Document exchange rate source
- Common sources: OANDA, XE.com, Central Bank rates
- Store original amount and currency in `currency` field

**Best Practices**:
- Apply consistent exchange rate methodology
- Use mid-market rates for historical transactions
- Document conversion factors
- Include original currency information

**Common Errors**:
- ❌ Currency symbols: `$12,500.00`
- ❌ Thousands separators: `12,500.00`
- ❌ Wrong precision: `12500.5` (need 12500.50)
- ❌ Negative values: `-1000.00`

---

### 10. currency

**Type**: String (ISO 4217)
**Required**: Yes
**Format**: 3-letter uppercase currency code

**Description**: The original transaction currency before USD conversion. Uses ISO 4217 standard currency codes.

**Common Currency Codes**:
```
USD - United States Dollar
EUR - Euro
GBP - British Pound Sterling
CNY - Chinese Yuan
JPY - Japanese Yen
CAD - Canadian Dollar
AUD - Australian Dollar
CHF - Swiss Franc
INR - Indian Rupee
BRL - Brazilian Real
MXN - Mexican Peso
SGD - Singapore Dollar
HKD - Hong Kong Dollar
KRW - South Korean Won
SEK - Swedish Krona
NOK - Norwegian Krone
DKK - Danish Krone
```

**Validation Rules**:
- Must be valid ISO 4217 code
- Exactly 3 uppercase letters
- No lowercase or special characters

**Best Practices**:
- Store original currency even if transaction is in USD
- Use for exchange rate audit trails
- Enable multi-currency reporting
- Document exchange rate sources

**Common Errors**:
- ❌ Lowercase: `usd`
- ❌ Full names: `US Dollar`
- ❌ Symbols: `$`
- ❌ Wrong length: `US` or `EURO`

---

### 11. ghg_category

**Type**: Integer
**Required**: Yes
**Range**: 1-15

**Description**: The Scope 3 GHG Protocol category number (1-15) that best represents this transaction. See [GHG Protocol Categories](#ghg-protocol-categories) section for detailed definitions.

**Category Quick Reference**:
```
1  - Purchased Goods and Services
2  - Capital Goods
3  - Fuel- and Energy-Related Activities
4  - Upstream Transportation and Distribution
5  - Waste Generated in Operations
6  - Business Travel
7  - Employee Commuting
8  - Upstream Leased Assets
9  - Downstream Transportation and Distribution
10 - Processing of Sold Products
11 - Use of Sold Products
12 - End-of-Life Treatment of Sold Products
13 - Downstream Leased Assets
14 - Franchises
15 - Investments
```

**Examples by Product Type**:
```
Raw materials (steel, chemicals) → 1
Manufacturing equipment → 2
Electricity, natural gas → 3
Freight, logistics → 4
Waste management → 5
Employee flights → 6
Cloud computing → 1
Packaging materials → 1
```

**Validation Rules**:
- Must be integer 1-15
- No decimals or fractions
- Must align with transaction type

**Best Practices**:
- Review GHG Protocol guidance for complex transactions
- Document categorization decisions
- Maintain consistency across similar transactions
- Consider upstream vs. downstream distinction

**Common Errors**:
- ❌ Out of range: `0`, `16`, `20`
- ❌ Wrong category for transaction type
- ❌ Mixing Scope 1/2 with Scope 3

---

### 12. country

**Type**: String (ISO 3166-1 alpha-2)
**Required**: Yes
**Format**: 2-letter uppercase country code

**Description**: The country where the supplier is located or where the service is provided. Uses ISO 3166-1 alpha-2 standard country codes.

**Common Country Codes**:
```
US - United States
GB - United Kingdom
CN - China
DE - Germany
FR - France
JP - Japan
IN - India
CA - Canada
AU - Australia
BR - Brazil
MX - Mexico
IT - Italy
ES - Spain
NL - Netherlands
SE - Sweden
CH - Switzerland
SG - Singapore
KR - South Korea
```

**Validation Rules**:
- Must be valid ISO 3166-1 alpha-2 code
- Exactly 2 uppercase letters
- No lowercase or special characters

**Usage Guidelines**:
- Use supplier's primary operating country
- For transportation, use origin or destination based on context
- For global services, use invoicing entity country
- Use for regional emission factor application

**Best Practices**:
- Validate against ISO 3166-1 standard
- Consider regional emission factor variations
- Document country determination methodology
- Use for supply chain mapping

**Common Errors**:
- ❌ Three-letter codes: `USA` (use `US`)
- ❌ Full names: `United States`
- ❌ Lowercase: `us`
- ❌ Invalid codes: `UK` (should be `GB`)

---

### 13. description

**Type**: String
**Required**: No (Optional)
**Maximum Length**: 1000 characters

**Description**: Additional context, notes, or details about the transaction. This field is optional but highly recommended for providing context that aids in data quality review and audit trails.

**Use Cases**:
- Product specifications and technical details
- Service scope and deliverables
- Project or cost center information
- Special handling or processing notes
- Data quality flags or assumptions

**Examples**:
```
"High-grade stainless steel for automotive parts manufacturing"
"Shanghai to Los Angeles route - consumer electronics shipment"
"AWS EC2 instances for data processing and ML training"
"Employee travel for client meetings and conferences Q3 2024"
"Monthly electricity consumption for manufacturing facility"
"Recycled content: 80%. FSC certified packaging."
"Emergency procurement - expedited delivery"
"Part of sustainability initiative - renewable energy sourcing"
```

**Best Practices**:
- Include relevant technical specifications
- Note any special circumstances
- Reference projects or cost centers
- Flag data quality issues or assumptions
- Keep descriptions concise but informative

**Common Pitfalls**:
- Exceeding 1000 character limit
- Including sensitive or confidential information
- Duplicating information from other fields
- Using inconsistent terminology

---

## Data Validation Rules

### Upload-Level Validations

**File Format Validation**:
- CSV files must be valid UTF-8 encoded text
- JSON files must be valid JSON with proper schema
- File size limits: CSV (50 MB), JSON (100 MB)
- Row limits: CSV (50,000), JSON (10,000 transactions)

**Data Integrity Checks**:
1. **Uniqueness**: No duplicate `transaction_id` values
2. **Completeness**: All required fields present
3. **Format Compliance**: All fields match specified formats
4. **Reference Integrity**: Supplier IDs exist in master data
5. **Temporal Consistency**: Dates within valid range

### Field-Level Validations

**String Fields**:
- No leading or trailing whitespace
- Maximum length enforcement
- Character encoding validation
- Special character restrictions

**Numeric Fields**:
- Type validation (integer vs. decimal)
- Range validation (min/max)
- Precision validation (decimal places)
- Sign validation (positive/negative)

**Date Fields**:
- Format validation (ISO 8601)
- Range validation (past 5 years to today)
- Calendar validity (no Feb 30th)
- Future date prevention

**Code Fields (ISO standards)**:
- Format validation (length, case, pattern)
- Code existence validation against standards
- Version compatibility

### Business Rule Validations

**Logical Consistency**:
- Quantity and unit compatibility
- GHG category and product type alignment
- Spend amount reasonableness
- Date within reporting period

**Data Quality Thresholds**:
- Minimum spend thresholds
- Maximum quantity limits
- Statistical outlier detection
- Duplicate transaction detection

**Supplier Validation**:
- Supplier ID exists in master
- Supplier name matches master (fuzzy)
- Supplier active status check
- Supplier country consistency

### Validation Error Levels

**Critical Errors** (Upload rejected):
- Invalid file format
- Missing required fields
- Invalid data types
- Duplicate transaction IDs
- Invalid reference data (supplier not found)

**Warnings** (Upload succeeds with flags):
- Unusual spend amounts (outliers)
- Generic product descriptions
- Missing optional fields
- Data quality score below threshold

**Info Messages** (No action required):
- Successful validation
- Number of records processed
- Auto-categorization applied
- Emission factors matched

---

## GHG Protocol Categories

### Overview

The GHG Protocol Corporate Value Chain (Scope 3) Accounting and Reporting Standard defines 15 categories of Scope 3 emissions. Each transaction must be assigned to exactly one category.

### Category 1: Purchased Goods and Services

**Definition**: Emissions from the production of products and services purchased or acquired by the reporting company.

**Examples**:
- Raw materials and commodities
- Packaging materials
- Office supplies and equipment
- Professional services (consulting, legal, accounting)
- IT services and cloud computing
- Marketing and advertising services
- Facilities management services

**Transaction Characteristics**:
- Product or service consumed in operations
- Not a capital asset
- Not fuel or energy (see Category 3)
- Not transportation (see Category 4)

**Best Practices**:
- Use spend-based or supplier-specific emission factors
- Prioritize high-spend categories for accuracy
- Engage suppliers for primary data
- Track product-level carbon footprints

### Category 2: Capital Goods

**Definition**: Emissions from the production of capital goods purchased or acquired by the reporting company.

**Examples**:
- Manufacturing equipment and machinery
- Building construction and renovations
- Company vehicles and fleet
- IT infrastructure and servers
- Office furniture and fixtures
- Production tooling and molds

**Transaction Characteristics**:
- Asset with useful life > 1 year
- Capitalized in financial accounting
- Depreciated over time
- Not consumed immediately

**Best Practices**:
- Track capital expenditures separately
- Use supplier-specific data where available
- Consider full lifecycle impacts
- Allocate emissions over asset lifetime

### Category 3: Fuel- and Energy-Related Activities

**Definition**: Emissions from the production of fuels and energy purchased by the reporting company, not already accounted in Scope 1 or 2.

**Subcategories**:
a) Upstream emissions of purchased fuels
b) Upstream emissions of purchased electricity
c) Transmission and distribution (T&D) losses

**Examples**:
- Extraction, refining, and transportation of fuels
- Generation and transmission of purchased electricity
- Grid transmission and distribution losses
- Steam and heating/cooling energy production

**Transaction Characteristics**:
- Related to energy consumption
- Not the direct fuel/electricity cost (Scope 1/2)
- Upstream supply chain emissions
- T&D system losses

**Best Practices**:
- Use regional emission factors for electricity
- Track fuel supply chain separately
- Calculate T&D losses based on grid data
- Consider renewable energy certificates

### Category 4: Upstream Transportation and Distribution

**Definition**: Emissions from transportation and distribution of products purchased by the reporting company between supplier and company facilities.

**Included**:
- Inbound logistics and freight
- Third-party warehousing and distribution
- Retail and storage in vehicles and facilities
- Transportation of purchased goods

**Examples**:
- Ocean freight (container shipping)
- Air freight
- Ground transportation (trucking, rail)
- Courier and express delivery
- Warehousing and distribution centers

**Transaction Characteristics**:
- Transportation services purchased
- Movement of goods to reporting company
- Third-party logistics providers
- Storage in transit

**Best Practices**:
- Collect distance and mode data
- Use weight-distance methodology
- Distinguish between transport modes
- Track empty return trips if applicable

### Category 5: Waste Generated in Operations

**Definition**: Emissions from third-party disposal and treatment of waste generated in the reporting company's operations.

**Waste Types**:
- Solid waste (landfill, incineration)
- Wastewater treatment
- Hazardous waste disposal
- Recycling and composting

**Examples**:
- Municipal solid waste disposal
- Industrial waste management
- Construction and demolition waste
- E-waste recycling
- Wastewater treatment services

**Transaction Characteristics**:
- Waste disposal services purchased
- Weight or volume of waste
- Treatment method specified
- Third-party managed

**Best Practices**:
- Track waste by type and treatment method
- Measure waste generation rates
- Calculate avoided emissions from recycling
- Implement waste reduction programs

### Category 6: Business Travel

**Definition**: Emissions from transportation of employees for business-related activities during the reporting year.

**Included**:
- Air travel (commercial flights)
- Ground transportation (rental cars, taxis, trains)
- Hotel accommodations
- Ferries and other transportation

**Examples**:
- Employee flights for meetings and conferences
- Rental cars during business trips
- Train travel for business purposes
- Hotel stays during business travel
- Taxis and rideshare services

**Transaction Characteristics**:
- Employee business travel only
- Third-party transportation services
- Distance-based calculations
- Passenger-kilometers methodology

**Best Practices**:
- Track travel by mode and distance
- Distinguish travel class (economy, business)
- Use corporate travel data
- Implement virtual meeting alternatives

### Category 7: Employee Commuting

**Definition**: Emissions from transportation of employees between their homes and worksites during the reporting year.

**Included**:
- Daily commuting by employees
- Remote work emissions (if significant)

**Examples**:
- Personal vehicle commuting
- Public transportation (bus, metro, rail)
- Carpooling and vanpools
- Bicycle and walking (zero emissions)
- Remote work energy consumption

**Transaction Characteristics**:
- Regular daily commuting
- Employee-controlled transportation
- Survey-based data collection
- Average commute distances

**Best Practices**:
- Conduct employee commute surveys
- Track remote work percentages
- Calculate average commute distance
- Promote sustainable commuting options

### Category 8: Upstream Leased Assets

**Definition**: Emissions from operation of assets leased by the reporting company (lessee) and not already included in Scope 1 or 2.

**Included**:
- Leased buildings and facilities
- Leased vehicles and equipment
- Leased data centers

**Examples**:
- Office building leases (energy not controlled)
- Vehicle leasing programs
- Equipment leases
- Co-location data centers

**Transaction Characteristics**:
- Operating leases (not capital leases)
- Assets not controlled by reporting company
- Energy consumption not in Scope 1/2
- Lease payments traceable

**Best Practices**:
- Collect utility data from lessors
- Estimate based on square footage
- Track lease agreements
- Allocate based on usage

### Category 9: Downstream Transportation and Distribution

**Definition**: Emissions from transportation and distribution of products sold by the reporting company between company operations and end consumer.

**Included**:
- Outbound logistics
- Retail and storage by third parties
- Transportation to customers

**Examples**:
- Product distribution to retailers
- E-commerce last-mile delivery
- Third-party warehousing
- Customer direct shipping

**Transaction Characteristics**:
- Products sold by reporting company
- Third-party transportation
- Between company and end customer
- Post-sale logistics

**Best Practices**:
- Track shipment weights and distances
- Use logistics provider data
- Calculate based on sales volumes
- Optimize distribution networks

### Category 10: Processing of Sold Products

**Definition**: Emissions from processing of intermediate products sold by the reporting company to downstream companies.

**Included**:
- Energy and materials for processing
- Manufacturing steps by customers

**Examples**:
- Steel processing by automotive manufacturers
- Chemical conversion by manufacturers
- Food processing by packagers
- Semiconductor manufacturing by OEMs

**Transaction Characteristics**:
- Intermediate products (not finished goods)
- Sold to downstream processors
- Processing emissions quantifiable
- Material-specific factors

**Best Practices**:
- Identify intermediate product sales
- Estimate processing energy requirements
- Use industry-average data
- Engage with downstream customers

### Category 11: Use of Sold Products

**Definition**: Emissions from use of products sold by the reporting company during their lifetime.

**Product Types**:
- Energy-consuming products (appliances, vehicles)
- Fuels and feedstocks sold
- Products releasing emissions during use

**Examples**:
- Electricity from sold generators
- Fuel combustion from gasoline sales
- Emissions from sold vehicles during use
- Energy consumption by sold appliances

**Transaction Characteristics**:
- Direct use-phase emissions
- Product lifetime considerations
- Usage patterns and intensity
- Energy efficiency ratings

**Best Practices**:
- Estimate product lifetimes
- Calculate energy consumption rates
- Use usage pattern surveys
- Track product energy efficiency

### Category 12: End-of-Life Treatment of Sold Products

**Definition**: Emissions from waste disposal and treatment of products sold by the reporting company at the end of their life.

**Treatment Methods**:
- Landfilling
- Incineration
- Recycling
- Composting

**Examples**:
- Consumer product disposal
- Packaging waste treatment
- Electronic waste recycling
- Product material recovery

**Transaction Characteristics**:
- Sold product waste treatment
- Material composition known
- Treatment methods estimated
- Recycling rates applied

**Best Practices**:
- Analyze product material composition
- Estimate treatment method distribution
- Calculate avoided emissions from recycling
- Design for end-of-life recyclability

### Category 13: Downstream Leased Assets

**Definition**: Emissions from operation of assets owned by the reporting company and leased to other entities, not already included in Scope 1 or 2.

**Included**:
- Real estate leased to others
- Equipment leased to customers
- Vehicle leasing programs

**Examples**:
- Commercial real estate leasing
- Equipment leasing business
- Fleet leasing to customers
- Data center leasing

**Transaction Characteristics**:
- Assets owned by reporting company
- Leased to other entities (lessor)
- Operating emissions by lessee
- Lease revenue traceable

**Best Practices**:
- Collect utility data from tenants
- Estimate based on asset characteristics
- Track lease agreements and terms
- Allocate based on leased area or usage

### Category 14: Franchises

**Definition**: Emissions from operation of franchises not included in Scope 1 or 2.

**Included**:
- Franchise location energy use
- Franchise operations emissions
- Franchise transportation

**Examples**:
- Restaurant franchise operations
- Retail franchise stores
- Hotel franchise properties
- Service franchise locations

**Transaction Characteristics**:
- Franchise business model
- Franchise-owned operations
- Energy and operational data
- Franchise agreement terms

**Best Practices**:
- Collect data from franchisees
- Estimate based on sales or locations
- Use per-location emission factors
- Engage franchisees in reporting

### Category 15: Investments

**Definition**: Emissions associated with the reporting company's investments (for financial institutions and investors).

**Investment Types**:
- Equity investments
- Debt investments
- Project finance
- Managed investments

**Examples**:
- Corporate equity holdings
- Government and corporate bonds
- Real estate investments
- Private equity and venture capital

**Transaction Characteristics**:
- Financial institution reporting
- Investment portfolio analysis
- Financed emissions calculations
- Attribution methodologies

**Best Practices**:
- Calculate financed emissions
- Use PCAF methodology
- Collect portfolio company data
- Report by asset class

---

## Product Categories (UNSPSC)

### Overview

The United Nations Standard Products and Services Code (UNSPSC) is a hierarchical classification system for products and services. It's used in the GL-VCCI platform for:
- Emission factor matching
- Spend category analysis
- Supply chain mapping
- Compliance reporting

### UNSPSC Hierarchy

**Structure**: Segment-Family-Class-Commodity (8 digits)

Example: `33.10.15.10` = Ferrous Metals (33) > Iron and Steel (10) > Steel (15) > Steel Sheets (10)

### Key UNSPSC Segments for Carbon Accounting

#### Segment 10-17: Live Plant and Animal Material and Accessories

**Common Families**:
- `1010` - Live animals
- `1011` - Livestock
- `1110` - Crops and agricultural products
- `1410` - Biological materials

**Carbon Accounting Use Cases**:
- Agricultural inputs
- Food products
- Biofuels and biomass

#### Segment 11-18: Chemicals and Allied Products

**Common Families**:
- `1112` - Adhesives and sealants
- `1115` - Industrial chemicals
- `1121` - Acids and bases
- `1201` - Plastic materials

**Carbon Accounting Use Cases**:
- Process chemicals
- Industrial materials
- Packaging materials

**Examples**:
```
11121507 - Industrial adhesive polymers
11150000 - Industrial gases
12010000 - Plastic resins and materials
```

#### Segment 26-30: Energy and Environmental

**Common Families**:
- `2611` - Electrical power
- `2612` - Natural gas
- `2613` - Fuel oil and petroleum
- `7611` - Waste management

**Carbon Accounting Use Cases**:
- Energy procurement
- Fuel purchases
- Waste services

**Examples**:
```
26111701 - Electrical power
26121600 - Natural gas
26131700 - Diesel fuel
76111803 - Waste disposal services
```

#### Segment 30-31: Metals and Materials

**Common Families**:
- `3010` - Non-ferrous metals (aluminum, copper)
- `3020` - Precious metals
- `3310` - Ferrous metals (steel, iron)
- `3410` - Building materials

**Carbon Accounting Use Cases**:
- Raw material procurement
- Manufacturing inputs
- Construction materials

**Examples**:
```
3010.30.00.00 - Aluminum products
3310.15.10.00 - Steel sheets
3410.00.00.00 - Concrete and masonry
```

#### Segment 43: IT and Telecommunications

**Common Families**:
- `4311` - Computer hardware
- `4321` - Computer equipment
- `4611` - Telecommunications equipment

**Carbon Accounting Use Cases**:
- IT equipment purchases
- Hardware procurement
- Electronics manufacturing

**Examples**:
```
43211500 - Desktop computers
43212100 - Servers
46111600 - Networking equipment
```

#### Segment 78: Transportation and Logistics

**Common Families**:
- `7810` - Transportation services
- `7811` - Freight and cargo services
- `7812` - Warehousing and storage

**Carbon Accounting Use Cases**:
- Freight and logistics (Category 4)
- Business travel (Category 6)
- Distribution services

**Examples**:
```
78101803 - Ocean freight
78102202 - Air freight
78111801 - Passenger air transportation
78121500 - Warehousing services
```

#### Segment 81: Computer Services

**Common Families**:
- `8111` - IT services
- `8112` - Software services
- `8113` - Cloud computing

**Carbon Accounting Use Cases**:
- Cloud services (Category 1)
- IT outsourcing
- SaaS platforms

**Examples**:
```
81111800 - Cloud infrastructure services
81111803 - Data processing services
81112000 - Software development
```

### UNSPSC Code Assignment Guide

**Step 1: Identify Segment** (First 2 digits)
- What is the high-level category?
- Manufacturing, Services, Energy, etc.

**Step 2: Identify Family** (Next 2 digits)
- What is the product/service family?
- Specific material type or service category

**Step 3: Identify Class** (Next 2 digits)
- What is the specific class of product?
- More detailed classification

**Step 4: Identify Commodity** (Final 2 digits)
- What is the exact product/service?
- Most granular level

**Best Practices**:
- Use most specific code available (8 digits preferred)
- Maintain code mapping documentation
- Validate codes against UNSPSC database
- Update codes when UNSPSC standard changes
- Use consistent codes for similar products

**Resources**:
- UNSPSC Official Database: https://www.unspsc.org/
- Code search and validation tools
- Industry-specific code lists
- Supplier product catalogs with UNSPSC codes

---

## Country and Currency Codes

### ISO 3166-1 Country Codes (Alpha-2)

**Major Trading Countries**:
```
US - United States
CA - Canada
MX - Mexico

GB - United Kingdom
DE - Germany
FR - France
IT - Italy
ES - Spain
NL - Netherlands
BE - Belgium
CH - Switzerland
AT - Austria
SE - Sweden
NO - Norway
DK - Denmark
FI - Finland
IE - Ireland
PL - Poland

CN - China
JP - Japan
KR - South Korea
IN - India
SG - Singapore
HK - Hong Kong
TW - Taiwan
MY - Malaysia
TH - Thailand
VN - Vietnam
ID - Indonesia
PH - Philippines

AU - Australia
NZ - New Zealand

BR - Brazil
AR - Argentina
CL - Chile
CO - Colombia

ZA - South Africa
EG - Egypt
NG - Nigeria

AE - United Arab Emirates
SA - Saudi Arabia
IL - Israel
TR - Turkey
```

**Regional Groupings**:
- **North America**: US, CA, MX
- **European Union**: 27 member states (varies)
- **APAC**: CN, JP, KR, IN, SG, AU, NZ
- **LATAM**: BR, AR, CL, CO, MX
- **MENA**: AE, SA, EG, IL, TR

**Usage Guidelines**:
- Use for supplier location
- Apply regional emission factors
- Enable supply chain mapping
- Support compliance reporting (CBAM, etc.)

### ISO 4217 Currency Codes

**Major Global Currencies**:
```
USD - United States Dollar
EUR - Euro (19 Eurozone countries)
GBP - British Pound Sterling
JPY - Japanese Yen
CNY - Chinese Yuan (Renminbi)
CHF - Swiss Franc
CAD - Canadian Dollar
AUD - Australian Dollar
NZD - New Zealand Dollar

HKD - Hong Kong Dollar
SGD - Singapore Dollar
KRW - South Korean Won
INR - Indian Rupee
MYR - Malaysian Ringgit
THB - Thai Baht
IDR - Indonesian Rupiah

MXN - Mexican Peso
BRL - Brazilian Real
ARS - Argentine Peso
CLP - Chilean Peso
COP - Colombian Peso

SEK - Swedish Krona
NOK - Norwegian Krone
DKK - Danish Krone
PLN - Polish Zloty
CZK - Czech Koruna
HUF - Hungarian Forint

AED - UAE Dirham
SAR - Saudi Riyal
ZAR - South African Rand
TRY - Turkish Lira
ILS - Israeli Shekel
```

**Exchange Rate Guidelines**:
- Use transaction date exchange rate
- Source from reliable providers (OANDA, XE, Central Banks)
- Document exchange rate methodology
- Store original currency and amount
- Apply consistent conversion approach

**Common Exchange Rate Sources**:
1. **OANDA** - Historical rates, API available
2. **XE.com** - Mid-market rates
3. **Central Banks** - Official rates
4. **Bloomberg/Reuters** - Financial institution rates
5. **ERP Systems** - Configured corporate rates

---

## Data Quality Best Practices

### Data Collection

**1. Source Data Integrity**
- Extract from authoritative systems (ERP, procurement)
- Validate data at source before export
- Maintain data lineage documentation
- Implement automated extraction where possible

**2. Data Completeness**
- Capture all required fields
- Minimize null or missing values
- Document data gaps and assumptions
- Set completeness targets (>95%)

**3. Data Accuracy**
- Validate against business rules
- Cross-reference with financial data
- Implement data quality checks
- Monitor accuracy metrics over time

**4. Data Consistency**
- Use standardized naming conventions
- Apply consistent units and formats
- Maintain master data quality
- Implement data governance policies

### Data Preparation

**1. Data Cleansing**
- Remove duplicates
- Correct formatting errors
- Standardize field values
- Handle missing data appropriately

**2. Data Transformation**
- Convert units to standard formats
- Apply currency conversions
- Normalize date formats
- Categorize transactions correctly

**3. Data Enrichment**
- Add missing product categories
- Link to supplier master data
- Apply emission factors
- Calculate derived fields

**4. Data Validation**
- Run pre-upload validation checks
- Review validation reports
- Correct errors before upload
- Document validation results

### Master Data Management

**1. Supplier Master Data**
- Maintain accurate supplier registry
- Keep contact information current
- Track supplier characteristics (size, industry)
- Update emission factor data

**2. Product Master Data**
- Maintain product catalog
- Assign UNSPSC codes
- Link to emission factors
- Track product specifications

**3. Organizational Data**
- Define organizational hierarchy
- Maintain facility/location data
- Track business unit structure
- Update reporting boundaries

### Ongoing Data Quality

**1. Monitoring**
- Track data quality metrics
- Monitor upload success rates
- Review error patterns
- Identify improvement opportunities

**2. Continuous Improvement**
- Implement corrective actions
- Update validation rules
- Enhance data collection processes
- Train data stewards

**3. Auditing**
- Conduct periodic data audits
- Review data quality reports
- Verify calculation accuracy
- Maintain audit trails

---

## Common Data Quality Issues

### Issue 1: Duplicate Transactions

**Symptoms**:
- Same transaction_id uploaded multiple times
- Similar transactions with different IDs
- Duplicate spend amounts and dates

**Causes**:
- Multiple data sources without deduplication
- Reprocessing of corrected data
- System integration errors

**Solutions**:
- Implement unique ID generation
- Check for duplicates before upload
- Use idempotent upload APIs
- Maintain upload history

**Prevention**:
- Single source of truth for transaction data
- Automated deduplication processes
- Upload tracking and reconciliation

### Issue 2: Missing or Invalid Supplier Data

**Symptoms**:
- Supplier ID not found errors
- Inconsistent supplier names
- Missing supplier information

**Causes**:
- Supplier not registered in master data
- Typos in supplier names or IDs
- Supplier master data out of sync

**Solutions**:
- Register suppliers before uploading transactions
- Use fuzzy matching for supplier names
- Implement supplier data validation
- Maintain supplier master data quality

**Prevention**:
- Supplier onboarding workflow
- Regular supplier master updates
- Automated supplier matching
- Supplier data governance

### Issue 3: Incorrect GHG Categorization

**Symptoms**:
- Transactions in wrong Scope 3 category
- Inconsistent categorization over time
- Missing category assignments

**Causes**:
- Misunderstanding of GHG Protocol
- Complex transactions spanning multiple categories
- Lack of categorization guidance

**Solutions**:
- Review GHG Protocol guidance
- Implement categorization logic
- Use AI-powered auto-categorization
- Document categorization rules

**Prevention**:
- Training on GHG Protocol
- Categorization decision trees
- Regular category audits
- Expert review of complex cases

### Issue 4: Generic Product Descriptions

**Symptoms**:
- Product names like "Materials" or "Services"
- Insufficient detail for emission factor matching
- Low data quality scores

**Causes**:
- Poor source data quality
- Lack of product master data
- Generic GL account descriptions

**Solutions**:
- Enrich product descriptions from source
- Link to product master data
- Use AI to enhance descriptions
- Engage procurement for details

**Prevention**:
- Require detailed descriptions at purchase
- Maintain product catalog
- Train procurement teams
- Implement data quality gates

### Issue 5: Outlier Spend Amounts

**Symptoms**:
- Unusually high or low spend values
- Orders of magnitude errors
- Inconsistent with historical data

**Causes**:
- Decimal point errors
- Currency conversion mistakes
- Wrong unit multipliers
- Data entry errors

**Solutions**:
- Implement statistical outlier detection
- Review flagged transactions
- Validate against source systems
- Correct errors before finalization

**Prevention**:
- Automated validation at data entry
- Range checks on spend fields
- Cross-validation with financials
- Regular data quality reviews

### Issue 6: Inconsistent Units

**Symptoms**:
- Wrong unit for product type
- Non-standard unit abbreviations
- Unit conversion errors

**Causes**:
- Multiple source systems with different units
- Lack of unit standardization
- Manual data entry errors

**Solutions**:
- Convert to standard units before upload
- Use approved unit list
- Implement unit validation
- Document conversion factors

**Prevention**:
- Standardize units across systems
- Automated unit conversion
- Unit mapping tables
- Data quality training

### Issue 7: Date Range Issues

**Symptoms**:
- Future-dated transactions
- Transactions outside reporting period
- Invalid date formats

**Causes**:
- System date errors
- Fiscal year vs. calendar year confusion
- Date format inconsistencies

**Solutions**:
- Validate date ranges before upload
- Convert to ISO 8601 format
- Filter to reporting period
- Review and correct errors

**Prevention**:
- Automated date validation
- Consistent date formats
- Clear reporting period definitions
- System date accuracy checks

### Issue 8: Missing UNSPSC Codes

**Symptoms**:
- Empty product_category field
- Invalid UNSPSC codes
- Inconsistent code formats

**Causes**:
- Product master data incomplete
- Legacy data without codes
- Manual data entry without validation

**Solutions**:
- Assign UNSPSC codes from product descriptions
- Use AI-powered code suggestion
- Maintain product-to-code mapping
- Manual code assignment for complex cases

**Prevention**:
- Require UNSPSC codes at product creation
- Maintain product master data
- Automated code assignment
- Regular product catalog updates

### Issue 9: Currency Conversion Errors

**Symptoms**:
- Unrealistic spend_usd values
- Inconsistent with original amounts
- Wrong exchange rates applied

**Causes**:
- Using wrong exchange rate date
- Incorrect exchange rate source
- Manual calculation errors

**Solutions**:
- Use automated exchange rate lookup
- Apply transaction date rates
- Validate converted amounts
- Document rate sources

**Prevention**:
- Automated currency conversion
- Reliable exchange rate APIs
- Conversion audit trails
- Regular rate update processes

### Issue 10: Incomplete Transaction Context

**Symptoms**:
- Missing description field
- Insufficient detail for audit
- Unable to validate emissions calculations

**Causes**:
- Optional field not populated
- Source data lacks context
- Time constraints on data preparation

**Solutions**:
- Encourage description field usage
- Extract context from source systems
- Add notes during data review
- Use AI to generate descriptions

**Prevention**:
- Make description field required for high-value transactions
- Capture context at transaction creation
- Implement data enrichment processes
- Train data stewards on importance

---

## Bulk Upload Tips

### Preparing Large Uploads

**1. File Size Management**
- Split files exceeding size limits
- Target 10,000-20,000 rows per file
- Use compression for large files
- Monitor upload performance

**2. Batch Processing Strategy**
- Upload in chronological batches
- Process by reporting period
- Group by supplier or category
- Implement retry logic for failures

**3. Performance Optimization**
- Remove unnecessary columns
- Minimize file formatting
- Use CSV for large volumes
- Upload during off-peak hours

### Upload Workflow

**Step 1: Data Extraction**
- Export from source system
- Include all required fields
- Apply initial filters
- Document extraction parameters

**Step 2: Data Validation**
- Run pre-upload validation scripts
- Review validation reports
- Correct identified errors
- Document validation results

**Step 3: Data Staging**
- Stage files in upload directory
- Verify file integrity
- Check file encoding
- Backup original files

**Step 4: Upload Execution**
- Use API or web interface
- Monitor upload progress
- Review upload logs
- Verify record counts

**Step 5: Post-Upload Verification**
- Check upload status
- Review error reports
- Validate data in platform
- Reconcile uploaded records

### Error Handling

**1. Upload Failures**
- Review error messages
- Identify root causes
- Correct data issues
- Retry failed uploads

**2. Partial Uploads**
- Determine successfully uploaded records
- Extract failed records
- Correct and re-upload
- Maintain upload history

**3. Data Corrections**
- Identify incorrect data post-upload
- Use correction/reversal process
- Re-upload corrected data
- Document corrections

### Automation Strategies

**1. Scheduled Uploads**
- Automate regular uploads (weekly, monthly)
- Implement upload orchestration
- Monitor automated processes
- Alert on failures

**2. API Integration**
- Use platform APIs for programmatic upload
- Implement error handling
- Use retry logic with exponential backoff
- Log all API interactions

**3. ETL Pipelines**
- Build extract-transform-load workflows
- Implement data quality checks
- Automate data transformations
- Schedule regular execution

**Sample Python Upload Script**:
```python
import requests
import pandas as pd
import json
from datetime import datetime

def upload_transactions(file_path, api_key, api_endpoint):
    """
    Upload transactions from CSV to GL-VCCI platform
    """
    # Read and validate data
    df = pd.read_csv(file_path)

    # Data validation
    required_fields = [
        'transaction_id', 'date', 'supplier_name', 'supplier_id',
        'product_name', 'product_category', 'quantity', 'unit',
        'spend_usd', 'currency', 'ghg_category', 'country'
    ]

    missing_fields = set(required_fields) - set(df.columns)
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    # Convert to JSON format
    transactions = df.to_dict('records')

    payload = {
        "metadata": {
            "upload_id": f"UPL-{datetime.now().strftime('%Y-%m%d%H%M%S')}",
            "upload_date": datetime.now().isoformat(),
            "organization_id": "ORG-1001",
            "uploaded_by": "api@company.com"
        },
        "transactions": transactions,
        "validation_config": {
            "strict_mode": True,
            "validate_supplier_exists": True
        }
    }

    # Upload via API
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        api_endpoint,
        headers=headers,
        json=payload,
        timeout=300
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Upload successful: {result['records_processed']} records")
        return result
    else:
        print(f"Upload failed: {response.status_code}")
        print(response.text)
        raise Exception("Upload failed")

# Usage
upload_transactions(
    file_path="transactions_q3_2024.csv",
    api_key="your_api_key_here",
    api_endpoint="https://api.gl-vcci.com/v1/transactions/upload"
)
```

### Performance Tips

**1. Optimize Data Preparation**
- Pre-validate data offline
- Remove unnecessary columns
- Sort by transaction date
- Use efficient data types

**2. Network Optimization**
- Use compression (gzip)
- Upload during off-peak hours
- Consider geographic proximity to servers
- Use wired connections for large uploads

**3. Parallel Processing**
- Split large files into chunks
- Upload chunks in parallel
- Implement thread pooling
- Monitor resource usage

**4. Monitoring and Logging**
- Log all upload attempts
- Track upload duration
- Monitor success rates
- Alert on anomalies

---

## Field-by-Field Examples

### Example 1: Manufacturing - Raw Materials

**Scenario**: Purchase of cold-rolled steel sheets for automotive manufacturing

```csv
transaction_id: TXN-2024-MFG-00001
date: 2024-08-15
supplier_name: Precision Steel Manufacturing Co.
supplier_id: SUP-1001
product_name: Cold-rolled steel sheets - Grade 304
product_category: 3310.15.10.00
quantity: 5000
unit: kg
spend_usd: 12500.00
currency: USD
ghg_category: 1
country: US
description: High-grade stainless steel for automotive parts manufacturing
```

**Key Points**:
- Category 1 (Purchased Goods)
- Specific product grade included
- Weight-based unit (kg)
- Detailed description for audit

### Example 2: Logistics - Ocean Freight

**Scenario**: Container shipping from Shanghai to Los Angeles

```csv
transaction_id: TXN-2024-RTL-00003
date: 2024-09-01
supplier_name: Pacific Freight Logistics
supplier_id: SUP-2001
product_name: Ocean freight container shipping - 40ft
product_category: 78101803
quantity: 12
unit: containers
spend_usd: 24000.00
currency: USD
ghg_category: 4
country: CN
description: Shanghai to Los Angeles route - consumer electronics shipment
```

**Key Points**:
- Category 4 (Upstream Transportation)
- Container count as quantity
- Origin country (CN = China)
- Route information in description

### Example 3: Energy - Electricity Purchase

**Scenario**: Monthly electricity consumption for manufacturing facility

```csv
transaction_id: TXN-2024-MFG-00004
date: 2024-09-05
supplier_name: GreenPower Energy Solutions
supplier_id: SUP-3001
product_name: Electricity - Renewable Energy Mix
product_category: 26111701
quantity: 150000
unit: kWh
spend_usd: 18000.00
currency: USD
ghg_category: 3
country: US
description: Monthly electricity consumption for manufacturing facility
```

**Key Points**:
- Category 3 (Fuel and Energy-Related Activities)
- Energy unit (kWh)
- Renewable energy specified
- Facility context provided

### Example 4: Technology - Cloud Services

**Scenario**: AWS cloud computing services for data processing

```csv
transaction_id: TXN-2024-TECH-00005
date: 2024-09-10
supplier_name: CloudCompute Services Inc.
supplier_id: SUP-4001
product_name: Cloud infrastructure services - Compute
product_category: 81111800
quantity: 5000
unit: compute-hours
spend_usd: 15000.00
currency: USD
ghg_category: 1
country: US
description: AWS EC2 instances for data processing and ML training
```

**Key Points**:
- Category 1 (Purchased Services)
- Compute-hours as custom unit
- Cloud provider specified
- Use case described

### Example 5: Travel - Business Flights

**Scenario**: International business travel for Q3 2024

```csv
transaction_id: TXN-2024-TECH-00008
date: 2024-09-25
supplier_name: TravelCorp Business Services
supplier_id: SUP-5001
product_name: Business travel - International flights
product_category: 78111801
quantity: 15
unit: trips
spend_usd: 22500.00
currency: USD
ghg_category: 6
country: US
description: Employee travel for client meetings and conferences Q3 2024
```

**Key Points**:
- Category 6 (Business Travel)
- Trips as quantity unit
- Travel purpose stated
- Reporting period indicated

### Example 6: Packaging - Recycled Materials

**Scenario**: Purchase of sustainable packaging materials

```csv
transaction_id: TXN-2024-RTL-00009
date: 2024-10-01
supplier_name: EcoPackaging Solutions Ltd.
supplier_id: SUP-6001
product_name: Recycled cardboard packaging
product_category: 14111506
quantity: 10000
unit: units
spend_usd: 6800.00
currency: EUR
description: Sustainable packaging materials for product distribution
```

**Key Points**:
- Category 1 (Purchased Goods)
- Recycled content specified
- Original currency (EUR)
- Sustainability context

### Example 7: Waste Management

**Scenario**: Monthly hazardous waste disposal service

```csv
transaction_id: TXN-2024-MFG-00010
date: 2024-10-05
supplier_name: Industrial Waste Management Co.
supplier_id: SUP-7001
product_name: Waste disposal and recycling services
product_category: 76111803
quantity: 50
unit: tonnes
spend_usd: 3500.00
currency: USD
ghg_category: 5
country: US
description: Monthly hazardous and non-hazardous waste processing
```

**Key Points**:
- Category 5 (Waste Generated)
- Weight in tonnes
- Waste type specified
- Regular service frequency

### Example 8: HVAC - Refrigerant

**Scenario**: Data center cooling system maintenance

```csv
transaction_id: TXN-2024-TECH-00011
date: 2024-10-10
supplier_name: DataCenter Cooling Systems
supplier_id: SUP-8001
product_name: HVAC maintenance and R-22 refrigerant
product_category: 40101500
quantity: 100
unit: kg
spend_usd: 8500.00
currency: USD
ghg_category: 3
country: US
description: Quarterly maintenance and refrigerant top-up for cooling systems
```

**Key Points**:
- Category 3 (Energy-Related)
- Refrigerant type specified (R-22)
- Maintenance frequency noted
- Facility type indicated

---

## Industry-Specific Guidance

### Manufacturing Industry

**Key Scope 3 Categories**:
- Category 1: Raw materials, components, packaging
- Category 2: Production equipment, machinery
- Category 3: Industrial electricity and fuel
- Category 4: Inbound logistics
- Category 5: Manufacturing waste

**Data Collection Priorities**:
1. **High-volume materials**: Steel, aluminum, plastics, chemicals
2. **Energy-intensive processes**: Forging, casting, heat treatment
3. **Supplier-specific data**: Tier 1 supplier emissions
4. **Transportation**: Inbound freight modes and distances

**Common Product Categories**:
- `3310` - Ferrous metals
- `3010` - Non-ferrous metals
- `1112` - Adhesives and chemicals
- `1210` - Plastics
- `4010` - HVAC and industrial equipment

**Best Practices**:
- Track material mass and composition
- Calculate material-specific emission factors
- Engage suppliers for product carbon footprints
- Monitor supply chain changes
- Implement supplier scorecards

### Retail Industry

**Key Scope 3 Categories**:
- Category 1: Purchased goods for resale
- Category 4: Inbound logistics from suppliers
- Category 9: Outbound logistics to customers
- Category 11: Customer use of products (if applicable)
- Category 12: Product end-of-life

**Data Collection Priorities**:
1. **Product sourcing**: SKU-level data where possible
2. **Transportation**: Multi-modal freight data
3. **Packaging**: Materials and weights
4. **Store operations**: Leased store energy (Category 8)

**Common Product Categories**:
- `5000` - Consumer goods
- `4600` - Electronics
- `5300` - Clothing and accessories
- `7810` - Transportation services
- `1411` - Packaging materials

**Best Practices**:
- Leverage supplier collaboration platforms
- Use product category defaults for long tail
- Track high-volume SKUs specifically
- Monitor transportation modes and distances
- Implement sustainable packaging programs

### Technology Industry

**Key Scope 3 Categories**:
- Category 1: Cloud services, software, IT equipment
- Category 2: Data center equipment, servers
- Category 6: Business travel (high for tech companies)
- Category 11: Product use-phase energy (devices)

**Data Collection Priorities**:
1. **Cloud services**: Compute hours, storage, data transfer
2. **Hardware**: Laptops, servers, networking equipment
3. **Business travel**: Flights, hotels, ground transport
4. **Data centers**: PUE, energy sources, cooling

**Common Product Categories**:
- `8111` - IT services and cloud computing
- `4321` - Computer equipment and servers
- `4311` - Telecommunications
- `4611` - Networking equipment
- `7811` - Business travel services

**Best Practices**:
- Use cloud provider carbon data (AWS, Azure, GCP)
- Track compute efficiency metrics
- Implement virtual meeting policies
- Choose renewable energy data centers
- Design for energy-efficient devices

### Logistics and Transportation

**Key Scope 3 Categories**:
- Category 1: Fuel purchases (if not Scope 1)
- Category 3: Upstream fuel emissions
- Category 4: Subcontracted transportation
- Category 8: Leased vehicles and equipment

**Data Collection Priorities**:
1. **Fuel consumption**: Liters/gallons by fuel type
2. **Distance data**: Kilometers/miles by route
3. **Vehicle types**: Trucks, trains, ships, planes
4. **Load factors**: Cargo weight and capacity utilization

**Common Product Categories**:
- `1511` - Fuels (diesel, gasoline, jet fuel)
- `7810` - Transportation services
- `7812` - Warehousing
- `2510` - Vehicles and fleet

**Best Practices**:
- Calculate tonne-kilometers or passenger-kilometers
- Track by transport mode for accuracy
- Monitor load factors and empty miles
- Use GPS and telematics data
- Optimize routes for efficiency

### Food and Agriculture

**Key Scope 3 Categories**:
- Category 1: Agricultural inputs, ingredients
- Category 4: Food transportation (cold chain)
- Category 5: Food waste
- Category 11: Consumer food preparation

**Data Collection Priorities**:
1. **Ingredient sourcing**: Type, quantity, origin
2. **Agricultural practices**: Conventional vs. organic, regenerative
3. **Refrigerated transport**: Cold chain logistics
4. **Food waste**: Pre-consumer and post-consumer

**Common Product Categories**:
- `1010` - Livestock and meat products
- `1011` - Agricultural products
- `5010` - Food products
- `1511` - Agricultural chemicals
- `7810` - Refrigerated transport

**Best Practices**:
- Use agricultural emission factors by commodity
- Track land use change for key ingredients
- Monitor cold chain energy consumption
- Measure and reduce food waste
- Source sustainably produced ingredients

### Construction and Real Estate

**Key Scope 3 Categories**:
- Category 1: Building materials (concrete, steel)
- Category 2: Construction equipment
- Category 4: Material transportation
- Category 13: Leased building operations (for lessors)

**Data Collection Priorities**:
1. **Material quantities**: Concrete (m³), steel (tonnes), etc.
2. **Equipment use**: Heavy machinery fuel consumption
3. **Transportation**: Material delivery distances
4. **Building operations**: Tenant energy use (for lessors)

**Common Product Categories**:
- `3010` - Structural metals
- `3410` - Concrete and masonry
- `3020` - Lumber and wood products
- `2311` - Building equipment
- `7810` - Heavy haul transportation

**Best Practices**:
- Use material-specific emission factors
- Track embodied carbon in materials
- Monitor construction waste
- Design for operational efficiency
- Use sustainable and recycled materials

---

## Troubleshooting

### Upload Errors

#### Error: "Invalid file format"

**Possible Causes**:
- File is not valid CSV or JSON
- Wrong encoding (not UTF-8)
- Corrupted file

**Solutions**:
1. Verify file extension (.csv or .json)
2. Open file in text editor to check format
3. Re-export from source system
4. Use UTF-8 encoding explicitly

#### Error: "Missing required field: [field_name]"

**Possible Causes**:
- CSV missing column header
- JSON missing required property
- Column name misspelled

**Solutions**:
1. Verify all required fields present
2. Check column name spelling (case-sensitive)
3. Compare to template headers
4. Add missing column to source data

#### Error: "Duplicate transaction_id: [id]"

**Possible Causes**:
- Same transaction uploaded twice
- ID generation not unique
- Reprocessing without deduplication

**Solutions**:
1. Check for duplicate rows in file
2. Remove duplicates before upload
3. Use `skip_duplicates` configuration
4. Verify ID generation logic

#### Error: "Supplier not found: [supplier_id]"

**Possible Causes**:
- Supplier not registered in master data
- Typo in supplier_id
- Case sensitivity mismatch

**Solutions**:
1. Register supplier in master data first
2. Verify supplier_id spelling
3. Check for leading/trailing spaces
4. Use supplier lookup tool to verify

#### Error: "Invalid date format"

**Possible Causes**:
- Date not in YYYY-MM-DD format
- Invalid calendar date
- Wrong date type in Excel

**Solutions**:
1. Convert to ISO 8601 format (YYYY-MM-DD)
2. Verify date is valid calendar date
3. Check for date formatting issues in Excel
4. Use text format for date columns in CSV

### Data Quality Warnings

#### Warning: "Unusually high spend amount"

**Possible Causes**:
- Decimal point error
- Currency conversion error
- Legitimate high-value transaction

**Actions**:
1. Verify spend amount against source
2. Check currency conversion calculation
3. Review for decimal point errors
4. Document if legitimate outlier

#### Warning: "Generic product description"

**Possible Causes**:
- Source data has minimal detail
- Product master data incomplete
- GL account descriptions used

**Actions**:
1. Enrich from product master data
2. Add details to description field
3. Request more detail from procurement
4. Use AI-powered description enhancement

#### Warning: "Low data quality score"

**Possible Causes**:
- Multiple validation warnings
- Missing optional fields
- Low confidence emission factor match

**Actions**:
1. Review individual data quality issues
2. Complete optional fields where possible
3. Improve product descriptions
4. Verify product categories

### Performance Issues

#### Issue: Upload taking too long

**Possible Causes**:
- File too large
- Network issues
- Server load

**Solutions**:
1. Split into smaller files
2. Upload during off-peak hours
3. Check network connection
4. Use API instead of web interface

#### Issue: Timeout errors

**Possible Causes**:
- File exceeds processing time limits
- Complex validation taking too long
- Network timeout

**Solutions**:
1. Reduce file size
2. Disable strict validation temporarily
3. Increase timeout in API client
4. Contact support for assistance

### Data Correction

#### Issue: Need to correct uploaded data

**Options**:
1. **Delete and re-upload**: For major errors
2. **Upload corrections**: Use correction transaction type
3. **Manual adjustment**: For individual records
4. **Bulk update API**: For systematic corrections

**Best Practice Workflow**:
1. Identify incorrect records
2. Document corrections needed
3. Prepare corrected data file
4. Upload with appropriate correction method
5. Verify corrections applied
6. Update source systems to prevent recurrence

---

## API Integration

### Authentication

**API Key Authentication**:
```bash
curl -X POST https://api.gl-vcci.com/v1/transactions/upload \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d @transaction_data.json
```

**OAuth 2.0 Authentication**:
```python
import requests
from requests_oauthlib import OAuth2Session

client_id = 'your_client_id'
client_secret = 'your_client_secret'
token_url = 'https://auth.gl-vcci.com/oauth/token'

oauth = OAuth2Session(client_id)
token = oauth.fetch_token(
    token_url,
    client_id=client_id,
    client_secret=client_secret
)

# Use token for API calls
headers = {"Authorization": f"Bearer {token['access_token']}"}
```

### Upload Endpoints

**CSV Upload**:
```
POST /v1/transactions/upload/csv
Content-Type: multipart/form-data

Parameters:
- file: CSV file
- organization_id: string
- reporting_period: string (optional)
```

**JSON Upload**:
```
POST /v1/transactions/upload/json
Content-Type: application/json

Body: JSON object with metadata and transactions array
```

**Bulk Upload (Multiple Files)**:
```
POST /v1/transactions/upload/batch
Content-Type: multipart/form-data

Parameters:
- files[]: Array of CSV or JSON files
- organization_id: string
```

### Response Format

**Success Response**:
```json
{
  "status": "success",
  "upload_id": "UPL-2024-10152024-0001",
  "records_processed": 1500,
  "records_accepted": 1485,
  "records_rejected": 15,
  "warnings": 23,
  "processing_time_ms": 5432,
  "validation_report_url": "https://api.gl-vcci.com/v1/uploads/UPL-2024-10152024-0001/report"
}
```

**Error Response**:
```json
{
  "status": "error",
  "error_code": "INVALID_FILE_FORMAT",
  "error_message": "File must be valid CSV or JSON",
  "details": {
    "line_number": 42,
    "field": "date",
    "value": "2024-13-45"
  }
}
```

### Validation Endpoint

**Pre-Upload Validation**:
```
POST /v1/transactions/validate
Content-Type: application/json

Body: Same format as upload, but no data is stored
```

**Response**:
```json
{
  "valid": true,
  "errors": [],
  "warnings": [
    {
      "transaction_id": "TXN-2024-001",
      "field": "product_name",
      "message": "Generic product description may reduce accuracy",
      "severity": "warning"
    }
  ]
}
```

### Status and Monitoring

**Check Upload Status**:
```
GET /v1/uploads/{upload_id}/status
```

**Download Validation Report**:
```
GET /v1/uploads/{upload_id}/report
Accept: application/pdf or text/csv
```

**List Recent Uploads**:
```
GET /v1/uploads?organization_id={org_id}&limit=50
```

### Rate Limiting

**Limits**:
- Standard tier: 100 requests/hour
- Premium tier: 1000 requests/hour
- Enterprise tier: Custom limits

**Rate Limit Headers**:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1609459200
```

**Handling Rate Limits**:
```python
import time
from requests.exceptions import HTTPError

def upload_with_retry(url, data, headers, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response.json()
        except HTTPError as e:
            if e.response.status_code == 429:  # Rate limited
                retry_after = int(e.response.headers.get('Retry-After', 60))
                print(f"Rate limited. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
            else:
                raise
    raise Exception("Max retries exceeded")
```

---

## FAQ

### General Questions

**Q: What file formats are supported?**
A: CSV and JSON formats are supported. CSV is recommended for manual uploads and Excel exports. JSON is ideal for API integration and automated uploads.

**Q: How many transactions can I upload at once?**
A: CSV uploads support up to 50,000 rows. JSON uploads support up to 10,000 transactions. For larger volumes, split into multiple files.

**Q: Can I upload data for multiple reporting periods?**
A: Yes, but it's recommended to upload data by reporting period for better organization and easier reconciliation.

**Q: How often should I upload data?**
A: For best results, upload data monthly or quarterly to maintain data freshness and enable timely emissions tracking.

### Data Preparation

**Q: What should I use for transaction_id?**
A: Use your existing ERP transaction ID, purchase order number, or invoice number. Ensure it's unique within your organization.

**Q: How do I handle multi-currency transactions?**
A: Store the original currency in the `currency` field and convert the amount to USD for the `spend_usd` field using the transaction date exchange rate.

**Q: What if I don't know the UNSPSC code?**
A: The platform can suggest UNSPSC codes based on product descriptions. However, manual verification is recommended for accuracy.

**Q: Can I include custom fields?**
A: Yes, in the JSON format you can add custom fields in the `custom_fields` object. These fields are stored but not used in emissions calculations.

### Emissions Calculations

**Q: How are emissions calculated from my data?**
A: The platform matches your transactions to emission factors based on product category, supplier data, and spend amounts. Multiple methodologies are used (spend-based, activity-based, supplier-specific).

**Q: What if no emission factor exists for my product?**
A: The platform uses hierarchical matching, falling back to broader categories if specific factors aren't available. You can also request custom emission factors.

**Q: Can I provide supplier-specific emission factors?**
A: Yes, supplier-specific emission factors (PCFs) can be registered in the platform and will be prioritized over generic factors.

**Q: How accurate are the emissions calculations?**
A: Accuracy depends on data quality and emission factor granularity. Supplier-specific data provides highest accuracy, followed by activity-based, then spend-based methods.

### Data Quality

**Q: What happens if my data has errors?**
A: Critical errors will prevent upload. Warnings allow upload but flag data quality issues. You can review validation reports and correct errors.

**Q: Can I edit transactions after upload?**
A: Yes, you can upload correction transactions or use the bulk update API. Direct editing in the UI is also available for individual records.

**Q: How do I handle duplicate transactions?**
A: The platform detects duplicate transaction_ids and rejects them. Use unique IDs and implement deduplication in your data preparation process.

**Q: What is a data quality score?**
A: A metric (0-100) indicating transaction data completeness, accuracy, and emission factor match confidence. Higher scores indicate better data quality.

### Compliance and Reporting

**Q: Is this compliant with GHG Protocol?**
A: Yes, the platform follows GHG Protocol Corporate Value Chain (Scope 3) Accounting and Reporting Standard guidance.

**Q: Can I use this data for CDP reporting?**
A: Yes, the platform provides CDP-formatted reports and meets CDP data requirements.

**Q: What about other standards (TCFD, CSRD, etc.)?**
A: The platform supports multiple reporting frameworks and can generate reports for TCFD, CSRD, SEC Climate Rule, and others.

**Q: How long is data retained?**
A: Transaction data is retained for a minimum of 7 years to support compliance and auditing requirements.

### Technical Questions

**Q: What is the API rate limit?**
A: Standard tier: 100 requests/hour. Premium: 1000 requests/hour. Enterprise: custom limits. Contact support for rate limit increases.

**Q: Can I automate uploads?**
A: Yes, use the API with scheduled jobs or ETL tools. Sample scripts are provided in the API documentation.

**Q: What authentication methods are supported?**
A: API Key authentication and OAuth 2.0 are supported. API keys are recommended for server-to-server integration.

**Q: Is there a test environment?**
A: Yes, a sandbox environment is available for testing uploads and integration before production deployment.

### Support

**Q: Where can I get help?**
A: Contact support at support@gl-vcci.com or visit the help center at https://help.gl-vcci.com

**Q: Are training resources available?**
A: Yes, training videos, webinars, and documentation are available in the platform's learning center.

**Q: Can I request custom features?**
A: Yes, submit feature requests through the platform or contact your account manager.

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-11-07 | Initial release | GL-VCCI Platform Team |

---

## Contact and Support

**Technical Support**:
- Email: support@gl-vcci.com
- Phone: +1 (555) 123-4567
- Hours: Monday-Friday, 9 AM - 5 PM EST

**Documentation**:
- Help Center: https://help.gl-vcci.com
- API Documentation: https://api-docs.gl-vcci.com
- Video Tutorials: https://learn.gl-vcci.com

**Account Management**:
- Email: accounts@gl-vcci.com
- Phone: +1 (555) 123-4568

---

© 2024-2025 GL-VCCI Carbon Intelligence Platform. All rights reserved.
