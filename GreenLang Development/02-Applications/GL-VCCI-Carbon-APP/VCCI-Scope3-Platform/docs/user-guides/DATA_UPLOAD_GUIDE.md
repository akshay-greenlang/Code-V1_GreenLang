# Data Upload Guide - User Guide

**Audience**: Sustainability analysts, data managers, procurement teams, IT administrators
**Prerequisites**: Procurement data access, understanding of Scope 3 categories, basic Excel/data skills
**Time**: 30-60 minutes for first upload, 15-30 minutes for subsequent uploads

---

## Overview

High-quality emission reporting starts with high-quality data. This guide covers everything you need to know about uploading procurement and activity data to the GL-VCCI Platform, from preparing your data files to handling validation errors and tracking upload history.

### What You'll Learn
- Supported file formats and their use cases
- Download and complete data templates
- Data mapping and field requirements
- Validation rules and quality checks
- Bulk upload procedures
- Error handling and correction workflows
- Upload history tracking and management

### Why Data Quality Matters

**Impact of Data Quality**:
- **Accuracy**: Better data = more accurate emission calculations
- **Credibility**: High-quality data improves stakeholder confidence
- **Compliance**: Many standards require minimum data quality levels
- **Decision-Making**: Reliable data enables informed reduction strategies
- **Efficiency**: Clean data reduces time spent on corrections and re-uploads

**Data Quality Hierarchy**:
1. **Supplier-Specific PCF Data** (Best) - Direct from supplier, product-specific
2. **Activity-Based Data** - Detailed transaction data (quantities, distances)
3. **Spend-Based Data** - Financial data with emission intensity factors
4. **Estimates** (Worst) - Rough approximations

**💡 Tip**: Aim to move up the quality hierarchy over time. Start with spend-based if necessary, then systematically collect activity-based and supplier-specific data.

---

## Section 1: Supported File Formats

### CSV (Comma-Separated Values)

**When to Use**:
- ✅ Large datasets (>10,000 rows)
- ✅ Automated exports from ERP systems
- ✅ Fastest processing speed
- ✅ Cross-platform compatibility

**Advantages**:
- Lightweight file size
- Fast upload and processing
- No formatting complexity
- Works with any data tool

**Limitations**:
- No data validation until upload
- Manual column mapping may be needed
- Single worksheet only

**Best Practices**:
- Use UTF-8 encoding to handle special characters
- Ensure consistent delimiter (comma, semicolon, or tab)
- Quote text fields containing commas
- Remove blank rows at end of file

[Screenshot: Sample CSV file in text editor]

**💡 Tip**: Export CSV from Excel using "CSV UTF-8" format to preserve international characters and avoid encoding issues.

### Excel (.xlsx)

**When to Use**:
- ✅ Small to medium datasets (<10,000 rows)
- ✅ Manual data entry or curation
- ✅ Need data validation before upload
- ✅ Multiple worksheets (e.g., transactions + metadata)

**Advantages**:
- Familiar interface
- Built-in data validation
- Formulas for calculations
- Multiple worksheets supported
- Formatted templates available

**Limitations**:
- Slower processing for large files
- Larger file sizes
- Version compatibility issues (use .xlsx, not .xls)

**Best Practices**:
- Use platform-provided templates
- Enable data validation on columns
- Remove formulas (convert to values before upload)
- Delete unused worksheets
- Keep file size under 50MB

[Screenshot: Excel template with data validation dropdowns]

**💡 Tip**: Platform templates have built-in validation. Use them to catch errors before upload.

### JSON (JavaScript Object Notation)

**When to Use**:
- ✅ API integrations
- ✅ Automated data pipelines
- ✅ Complex nested data structures
- ✅ System-to-system transfers

**Advantages**:
- Machine-readable
- Structured data with validation
- Supports hierarchical data
- Industry standard for APIs

**Limitations**:
- Not human-readable for large files
- Requires technical knowledge
- Limited tools for manual editing

**Example Structure**:
```json
{
  "upload_metadata": {
    "source": "SAP_ERP",
    "period": "2024-Q4",
    "uploaded_by": "user@company.com"
  },
  "transactions": [
    {
      "transaction_id": "PO-2024-001",
      "date": "2024-10-15",
      "supplier": "Acme Manufacturing",
      "category": "Purchased Goods",
      "spend_amount": 15000,
      "currency": "USD",
      "quantity": 500,
      "unit": "kg"
    }
  ]
}
```

**💡 Tip**: Use JSON for automated integrations. For manual uploads, stick with Excel or CSV.

### XML (Extensible Markup Language)

**When to Use**:
- ✅ Legacy system integrations
- ✅ Specific industry standards (e.g., cXML for procurement)
- ✅ Regulatory submission formats
- ✅ Complex data with strict schema

**Advantages**:
- Structured with schema validation
- Supports complex hierarchies
- Industry standards support (cXML, UBL)

**Limitations**:
- Verbose (larger file sizes)
- Requires technical knowledge
- Slower processing than JSON

**Example Structure**:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<procurement_data>
  <transaction>
    <transaction_id>PO-2024-001</transaction_id>
    <date>2024-10-15</date>
    <supplier>Acme Manufacturing</supplier>
    <category>Purchased Goods</category>
    <spend_amount currency="USD">15000</spend_amount>
    <quantity unit="kg">500</quantity>
  </transaction>
</procurement_data>
```

**💡 Tip**: XML is best for system integrations. Contact support for schema documentation.

### PDF (Portable Document Format)

**When to Use**:
- ✅ Invoice data extraction
- ✅ Shipping documents (bills of lading)
- ✅ Travel receipts and itineraries
- ✅ Supplier certificates

**Advantages**:
- Universal format
- Preserves original formatting
- Often the only format available (invoices, receipts)

**Limitations**:
- Requires OCR (Optical Character Recognition)
- Data extraction accuracy varies
- Manual review often needed
- Slower processing

**Platform Capabilities**:
- AI-powered data extraction
- Recognizes common invoice formats
- Extracts: dates, amounts, supplier names, line items
- Confidence scores for extracted data
- Manual review queue for low-confidence items

[Screenshot: PDF upload with extracted data review interface]

**💡 Tip**: PDF extraction works best for structured documents (invoices, receipts). For unstructured PDFs, manually enter data instead.

**Supported PDF Types**:
- ✅ Invoices and purchase orders
- ✅ Shipping documents
- ✅ Travel itineraries and receipts
- ✅ Utility bills
- ❌ Scanned images without OCR (convert with OCR first)
- ❌ Password-protected PDFs (remove password first)

---

## Section 2: Template Download and Completion

### Downloading Templates

**Step 1: Access Template Library**

1. Navigate to **Data** > **Upload Data**
2. Click **"Download Template"** button
3. **Select template type**:

**Available Templates**:
- **Procurement Data (General)** - All purchased goods and services
- **Transportation Data** - Shipments and logistics
- **Travel Data** - Business travel details
- **Energy & Utilities** - Electricity, gas, fuel
- **Capital Goods** - Asset purchases
- **Waste Data** - Waste generation and disposal
- **Employee Commuting** - Commute patterns

[Screenshot: Template selection dropdown]

4. **Choose format**: Excel (.xlsx) or CSV
5. **Download** - File saves to your computer

**💡 Tip**: Start with the general procurement template. You can always switch to category-specific templates later.

### Understanding Template Structure

**Procurement Data Template Layout**:

**Tab 1: Instructions**
- Field definitions
- Required vs. optional fields
- Data format examples
- Common errors to avoid
- Support contact information

**Tab 2: Data Entry**
- Pre-formatted columns
- Data validation dropdowns
- Example rows (delete before upload)
- Field tooltips

**Tab 3: Category Mapping**
- GHG Protocol Scope 3 categories
- Category descriptions
- Applicability guidelines

**Tab 4: Unit Conversions**
- Common unit conversions
- Reference for consistency

[Screenshot: Excel template with all tabs visible]

### Required Fields vs. Optional Fields

**Always Required**:
- ✅ **Transaction ID**: Unique identifier for each transaction
- ✅ **Date**: Transaction date (YYYY-MM-DD format)
- ✅ **Supplier Name**: Company providing goods/services
- ✅ **Category**: GHG Protocol Scope 3 category
- ✅ **Spend Amount**: Total cost of transaction
- ✅ **Currency**: Three-letter currency code (USD, EUR, GBP, etc.)

**Required for Activity-Based Calculations** (higher quality):
- ⭐ **Quantity**: Amount purchased
- ⭐ **Unit**: Unit of measurement (kg, liters, kWh, etc.)
- ⭐ **Product/Service Description**: What was purchased

**Optional but Valuable**:
- 💡 Business Unit: Internal department or division
- 💡 Geography: Location of purchase or supplier
- 💡 Product Category: Internal classification
- 💡 Cost Center: Accounting code
- 💡 Notes: Any additional context

[Screenshot: Required fields highlighted in template]

**💡 Tip**: More fields = better analysis. Aim to include optional fields, especially business unit and geography for segmented reporting.

### Field-by-Field Completion Guide

**Transaction ID**:
- **Format**: Alphanumeric, no special characters except dash and underscore
- **Examples**:
  - ✅ PO-2024-001
  - ✅ INV_12345
  - ✅ TR20241015_001
  - ❌ PO#2024/001 (no # or /)
- **Must be unique**: No duplicates in same upload
- **Best Practice**: Use your ERP's transaction ID

**Date**:
- **Format**: YYYY-MM-DD (ISO 8601)
- **Examples**:
  - ✅ 2024-10-15
  - ✅ 2024-01-01
  - ❌ 10/15/2024 (wrong format)
  - ❌ 15-10-2024 (wrong format)
  - ❌ 2024/10/15 (wrong delimiter)
- **Range**: Must be within 10 years (past or future)
- **Best Practice**: Export dates from ERP in ISO format

**Supplier Name**:
- **Format**: Text, up to 255 characters
- **Examples**:
  - ✅ Acme Manufacturing Inc.
  - ✅ ABC Corp
  - ✅ Local Supplier Ltd.
- **Best Practice**: Use consistent naming
  - Don't mix "Acme Inc." and "Acme Incorporated"
  - Platform will suggest matches during upload
  - Clean up supplier names before upload

**Category** (GHG Protocol Scope 3):
- **Format**: Category number or name
- **Valid Values**:
  - Category 1 or "Purchased Goods and Services"
  - Category 2 or "Capital Goods"
  - Category 3 or "Fuel and Energy Related Activities"
  - Category 4 or "Upstream Transportation and Distribution"
  - Category 5 or "Waste Generated in Operations"
  - Category 6 or "Business Travel"
  - Category 7 or "Employee Commuting"
  - Category 8 or "Upstream Leased Assets"
  - Category 9 or "Downstream Transportation and Distribution"
  - Category 10 or "Processing of Sold Products"
  - Category 11 or "Use of Sold Products"
  - Category 12 or "End-of-Life Treatment of Sold Products"
  - Category 13 or "Downstream Leased Assets"
  - Category 14 or "Franchises"
  - Category 15 or "Investments"
- **Use Excel dropdown** in template for validation
- **Best Practice**: When in doubt, use Category 1 (most general)

[Screenshot: Category dropdown in Excel template]

**Spend Amount**:
- **Format**: Numeric, no currency symbols or commas
- **Examples**:
  - ✅ 15000
  - ✅ 1500.50
  - ✅ 0.99
  - ❌ $15,000 (no symbols or commas)
  - ❌ 15K (no abbreviations)
- **Decimal**: Use period (.) as decimal separator
- **Must be positive**: Negative values for returns/credits (use separate flag)

**Currency**:
- **Format**: Three-letter ISO 4217 code
- **Common Codes**:
  - USD (US Dollar)
  - EUR (Euro)
  - GBP (British Pound)
  - JPY (Japanese Yen)
  - CNY (Chinese Yuan)
  - INR (Indian Rupee)
  - CAD (Canadian Dollar)
  - AUD (Australian Dollar)
- **Best Practice**: Platform converts to reporting currency automatically

**Quantity**:
- **Format**: Numeric, positive
- **Examples**:
  - ✅ 500
  - ✅ 1.5
  - ❌ Five hundred (no text)
- **Required for**: Activity-based calculations
- **Leave blank if**: Only spend-based calculation possible

**Unit**:
- **Format**: Text, standardized units preferred
- **Examples**:
  - ✅ kg (kilograms)
  - ✅ liters
  - ✅ kWh (kilowatt-hours)
  - ✅ pieces
  - ✅ hours
  - ✅ ton-km (for transport)
- **Best Practice**: Use metric units (kg, km, liters, kWh)
- **If using imperial**: Platform converts (specify: lbs, miles, gallons)
- **Must match quantity**: If quantity is 500, unit might be "kg" or "pieces"

**Product/Service Description**:
- **Format**: Text, up to 500 characters
- **Examples**:
  - ✅ "Industrial motor, 50 HP, electric"
  - ✅ "Office cleaning services, monthly"
  - ✅ "Steel sheet, galvanized, 2mm thickness"
- **Best Practice**: Include key details
  - Material composition
  - Key specifications
  - Intended use
- **AI Tip**: Platform uses description for emission factor matching

[Screenshot: Completed template rows with all fields filled]

**💡 Tip**: Fill out as many optional fields as possible. They enable better categorization, analysis, and reporting.

### Data Validation Rules

**Built-in Template Validation**:

Excel templates include:
- ✅ Dropdown lists for categories
- ✅ Date format validation
- ✅ Numeric field validation
- ✅ Required field highlighting (red cells)
- ✅ Currency code validation
- ✅ Unit standardization

**How to Use**:
1. Open template in Excel
2. Enter data in each cell
3. Invalid entries turn red with error message
4. Hover over red triangle for details
5. Correct and re-check

[Screenshot: Excel validation error message]

**Common Validation Errors**:
- "Date must be in YYYY-MM-DD format"
- "Category must be 1-15 or category name"
- "Spend amount must be a number"
- "Currency must be 3-letter code"
- "Transaction ID must be unique"

**💡 Tip**: Fix all validation errors in Excel before uploading to platform for faster processing.

---

## Section 3: Data Mapping and Validation Rules

### Automatic Data Mapping

**When You Upload Non-Template Files**:

If you upload your own format (not platform template):

1. **Platform analyzes columns** and suggests mappings
2. **Review suggested mappings**:
   - "Transaction_Number" → Transaction ID ✅
   - "PO_Date" → Date ✅
   - "Vendor_Name" → Supplier Name ✅
   - "Amount" → Spend Amount ✅
3. **Adjust if needed**:
   - Drag and drop to remap
   - Mark columns as "Do Not Import"
4. **Preview mapped data**
5. **Confirm and continue**

[Screenshot: Data mapping interface with drag-and-drop columns]

**💡 Tip**: Use consistent column names in your exports to speed up mapping for future uploads.

### Supplier Name Matching

**Intelligent Supplier Matching**:

Platform recognizes supplier variations:
- "Acme Inc." = "Acme Incorporated" = "ACME INC"
- "ABC Corp" = "ABC Corporation"

**Matching Process**:

1. **Upload file** with supplier names
2. **Platform matches to existing suppliers**:
   - ✅ Exact match: Auto-linked
   - ⚠️ Close match: Review suggestion
   - ❌ No match: Create new supplier entry

3. **Review close matches**:
   - "Acme Manufacturing" in file
   - Existing: "Acme Mfg Inc."
   - **Suggestion**: Same company? (85% confidence)
   - **Action**: Accept or Reject

4. **Bulk actions**:
   - Accept all high-confidence matches (>90%)
   - Review medium-confidence matches (70-90%)
   - Reject low-confidence matches (<70%)

[Screenshot: Supplier matching review screen]

**💡 Tip**: Build a supplier master list with standardized names. Use this for all uploads to avoid matching issues.

### Category Auto-Assignment

**AI-Powered Category Suggestion**:

If category field is blank or uncertain:

1. **Platform analyzes**:
   - Product/service description
   - Supplier name and industry
   - Historical categorization patterns
   - Keyword matching

2. **Suggests category**:
   - "Industrial motor" → Category 1 (Purchased Goods) - 95% confidence
   - "Freight shipment" → Category 4 (Upstream Transportation) - 98% confidence
   - "Flight ticket" → Category 6 (Business Travel) - 99% confidence

3. **Review and confirm**:
   - Accept suggestion
   - Change to different category
   - Apply to similar transactions

[Screenshot: AI category suggestions with confidence scores]

**💡 Tip**: Platform learns from your corrections. Accept/reject suggestions to improve future accuracy.

### Unit Standardization

**Automatic Unit Conversion**:

Platform converts to standard units:

**Weight**:
- lbs → kg (multiply by 0.453592)
- tons → tonnes (multiply by 0.907185)
- oz → kg (multiply by 0.0283495)

**Distance**:
- miles → km (multiply by 1.60934)
- feet → meters (multiply by 0.3048)

**Volume**:
- gallons (US) → liters (multiply by 3.78541)
- gallons (UK) → liters (multiply by 4.54609)

**Energy**:
- BTU → kWh (multiply by 0.000293071)
- therms → kWh (multiply by 29.3071)

**Example**:
- Uploaded: 500 lbs
- Stored: 226.8 kg
- Display: Shows both for verification

**💡 Tip**: Platform handles conversions, but using metric units (kg, km, liters, kWh) from the start avoids potential errors.

### Validation Rules and Quality Checks

**Tier 1: Critical Errors (Must Fix)**:
- ❌ Missing required fields
- ❌ Invalid date format
- ❌ Non-numeric spend amount
- ❌ Duplicate transaction IDs
- ❌ Invalid currency code
- ❌ Invalid category

**Result**: Upload blocked until fixed

**Tier 2: Warnings (Should Review)**:
- ⚠️ Unusual spend amount (>$1M or <$1)
- ⚠️ Date outside typical range (>1 year old or future)
- ⚠️ Missing optional fields (quantity, unit)
- ⚠️ Unrecognized supplier name
- ⚠️ Mismatched quantity-unit pairs

**Result**: Upload proceeds, but data flagged for review

**Tier 3: Informational (Good to Know)**:
- ℹ️ Potential duplicate (similar transaction already exists)
- ℹ️ Supplier in multiple categories (unusual pattern)
- ℹ️ Lower data quality (spend-based only, no quantity)

**Result**: Upload proceeds, no action required

[Screenshot: Validation results showing all three tiers]

**💡 Tip**: Aim for zero errors and warnings. Information notices are fine.

---

## Section 4: Bulk Upload Procedures

### Step-by-Step Upload Process

**Step 1: Prepare Your File**

1. **Export data from source system** (ERP, travel system, etc.)
2. **Use platform template** or map your columns
3. **Complete all required fields**
4. **Run local validation** (if using Excel template)
5. **Remove test/sample rows**
6. **Save file**:
   - Excel: Save as .xlsx (not .xls)
   - CSV: Save as UTF-8 CSV
7. **Check file size**: Must be under 50MB (if larger, split into multiple files)

**Step 2: Navigate to Upload**

1. Go to **Data** > **Upload Data**
2. Click **"New Upload"** button
3. **Select upload method**:
   - ○ Single File Upload
   - ○ Multiple Files (batch)
   - ○ Compressed Archive (.zip)

[Screenshot: Upload method selection]

**Step 3: Select File and Configure**

1. **Click "Choose File"** or drag and drop
2. **File uploads** - progress bar shows status
3. **Configure upload settings**:

   **Upload Metadata**:
   - **Upload Name**: "Q4 2024 Procurement Data"
   - **Description**: "Oct-Dec procurement from SAP"
   - **Data Source**: Select from dropdown (SAP, Oracle, Workday, etc.) or "Other"
   - **Reporting Period**: Select or enter custom dates

   **Data Quality Settings**:
   - **Duplicate Handling**:
     - ○ Skip duplicates
     - ● Update existing
     - ○ Import as new
   - **Validation Level**:
     - ○ Strict (no warnings allowed)
     - ● Standard (warnings allowed)
     - ○ Lenient (errors allowed, manual review)

   **Processing Options**:
   - ☑️ Auto-assign categories (AI suggestions)
   - ☑️ Auto-match suppliers
   - ☑️ Convert units to standard
   - ☑️ Currency conversion to reporting currency
   - ☐ Skip validation (not recommended)

[Screenshot: Upload configuration form]

4. **Click "Next"**

**Step 4: Data Mapping** (if non-template file)

1. **Review column mappings**:
   - Platform suggests mappings based on column names
   - Green checkmark = confident match
   - Yellow warning = low confidence, review
   - Red X = no match, requires manual mapping

2. **Adjust mappings**:
   - Click column to change mapping
   - Drag and drop to remap
   - Select "Ignore Column" for unused columns

3. **Preview data**:
   - Shows first 10 rows with mappings applied
   - Verify data looks correct

4. **Click "Next"**

[Screenshot: Column mapping with preview]

**Step 5: Validation**

1. **Platform validates data**:
   - Progress bar shows validation status
   - Typically 5-15 seconds per 1,000 rows

2. **Review validation results**:

   **Summary**:
   - Total rows: 5,000
   - ✅ Valid rows: 4,850 (97%)
   - ⚠️ Warnings: 120 (2.4%)
   - ❌ Errors: 30 (0.6%)

   **Detailed Results**:
   - **Errors** (must fix):
     - Row 45: Missing required field "Date"
     - Row 127: Invalid category "Cat 1" (use "Category 1")
     - Row 312: Duplicate Transaction ID "PO-2024-100"
   - **Warnings** (review recommended):
     - Row 88: Spend amount very high ($950,000) - please verify
     - Row 234: Missing quantity - will use spend-based calculation
   - **Info**:
     - 200 suppliers auto-matched
     - 15 new suppliers created
     - 300 categories auto-assigned

[Screenshot: Validation results dashboard]

3. **Options**:
   - **Fix and Re-upload**: Download error report, fix issues, re-upload
   - **Proceed with Warnings**: Import valid rows, skip errors
   - **Cancel**: Start over

**💡 Tip**: For first upload, aim for 100% validation success. Fix all errors and warnings. For recurring uploads, proceed with warnings is acceptable.

**Step 6: Error Correction** (if needed)

1. **Download error report**:
   - Click "Download Error Report"
   - Excel file with only error rows
   - Error descriptions in separate column

2. **Fix errors in Excel**:
   - Correct each error based on description
   - Delete error description column
   - Save file

3. **Re-upload corrected file**:
   - Upload flows faster (platform remembers mappings)
   - Only failed rows need correction

**Alternative**: Click "Edit in Browser" to fix errors directly in platform interface (for small number of errors)

[Screenshot: In-browser error editing interface]

**Step 7: Review and Confirm**

1. **Final review**:
   - **Data Summary**:
     - Rows to import: 4,850
     - Date range: 2024-10-01 to 2024-12-31
     - Total spend: $2,450,000
     - Suppliers: 215
     - Categories: 6

   - **Data Quality Preview**:
     - 45% have quantity data (activity-based)
     - 55% spend-based only
     - Estimated quality score: 2.8/5.0

2. **Set status**:
   - ○ Draft (can edit later)
   - ● Final (locks data for audit trail)

3. **Add notes** (optional):
   - "Q4 procurement data from SAP. Excludes capital goods (separate upload)."

4. **Click "Import Data"**

[Screenshot: Final confirmation screen]

**Step 8: Processing and Completion**

1. **Import processing**:
   - Progress bar shows status
   - "Importing rows..."
   - "Matching suppliers..."
   - "Calculating emissions..."
   - "Updating dashboards..."

2. **Processing time**:
   - Small uploads (<1,000 rows): 10-30 seconds
   - Medium uploads (1,000-10,000 rows): 1-3 minutes
   - Large uploads (>10,000 rows): 5-10 minutes

3. **Completion notification**:
   - Success message appears
   - Email notification sent
   - Summary statistics displayed:
     - 4,850 transactions imported
     - Total emissions: 1,245 tCO2e (estimated)
     - Average quality score: 2.8/5.0

4. **Next steps**:
   - **View Data**: See imported transactions
   - **View Dashboard**: Updated emission dashboard
   - **Generate Report**: Create emission report
   - **Upload More**: Add additional data

[Screenshot: Upload success notification]

**💡 Tip**: For large uploads, platform processes in background. You can continue working and receive notification when complete.

### Batch Upload Multiple Files

**When to Use**:
- Multiple months/quarters to upload
- Different data sources (procurement + travel + logistics)
- Split large dataset across files

**Process**:

1. **Navigate to Data** > **Upload Data**
2. **Select "Batch Upload"**
3. **Upload multiple files**:
   - Drag and drop up to 10 files
   - Or click to select multiple
   - Can mix formats (CSV, Excel, etc.)

4. **Configure each file**:
   - Assign to reporting period
   - Set upload name
   - Apply validation settings

5. **Process all**:
   - Platform processes sequentially
   - Shows progress for each file
   - Continues even if one file has errors

6. **Review results**:
   - Summary for each file
   - Combined totals
   - Any errors flagged per file

[Screenshot: Batch upload interface with multiple files]

**💡 Tip**: Use batch upload for monthly procurement data. Upload 3 months at once to save time.

### Compressed Archive Upload

**For Very Large Uploads**:

1. **Compress files** into .zip archive:
   - Include multiple CSV/Excel files
   - Can include supporting documentation (PDFs)
   - Keep under 200MB compressed

2. **Upload .zip file**:
   - Platform extracts automatically
   - Processes all data files
   - Stores supporting docs

3. **Review extracted files**:
   - Confirm all files detected
   - Configure each file
   - Process

**💡 Tip**: Ideal for annual data uploads or initial platform setup with historical data.

---

## Section 5: Error Handling and Corrections

### Common Upload Errors

**Error 1: Date Format Invalid**
- **Message**: "Date must be in YYYY-MM-DD format"
- **Cause**: Date in wrong format (MM/DD/YYYY, DD-MM-YYYY, etc.)
- **Solution**:
  1. Open file in Excel
  2. Select date column
  3. Format > Cells > Custom > "yyyy-mm-dd"
  4. If Excel auto-converts, type as text with apostrophe: '2024-10-15
  5. Save and re-upload

**Error 2: Duplicate Transaction ID**
- **Message**: "Transaction ID 'PO-2024-100' already exists"
- **Cause**: Same transaction uploaded twice OR genuinely duplicate ID
- **Solution**:
  - If duplicate upload: Skip this row
  - If same transaction, different data: Use "Update Existing" setting
  - If genuinely duplicate ID in source: Append suffix (PO-2024-100_A, PO-2024-100_B)

**Error 3: Missing Required Field**
- **Message**: "Required field 'Supplier Name' is empty in row 45"
- **Cause**: Required field left blank
- **Solution**:
  1. Download error report
  2. Fill in missing data
  3. If data truly unavailable, use placeholder: "Unknown Supplier" (will flag for follow-up)
  4. Re-upload

**Error 4: Invalid Category**
- **Message**: "Category 'Cat 1' not recognized"
- **Cause**: Category not in standard list (1-15 or category names)
- **Solution**:
  1. Review GHG Protocol category definitions
  2. Update to: "Category 1" or "Purchased Goods and Services"
  3. Use dropdown in Excel template to avoid errors

**Error 5: Non-Numeric Spend Amount**
- **Message**: "Spend amount '$15,000' must be a number"
- **Cause**: Currency symbols, commas, or text in numeric field
- **Solution**:
  1. Remove currency symbols ($, €, £, etc.)
  2. Remove thousands separators (commas)
  3. Keep decimal point (.)
  4. Result: 15000 or 15000.00

[Screenshot: Error report Excel file with descriptions]

**Error 6: Currency Code Invalid**
- **Message**: "Currency 'US Dollar' not recognized"
- **Cause**: Full currency name instead of 3-letter code
- **Solution**:
  - Use ISO 4217 codes: USD, EUR, GBP, JPY, etc.
  - See template for full list

**Error 7: File Too Large**
- **Message**: "File size exceeds 50MB limit"
- **Cause**: Too many rows or embedded objects
- **Solution**:
  1. Split into multiple files (by month, category, etc.)
  2. Remove formatting, formulas, images
  3. Save as CSV (smaller than Excel)
  4. Use batch upload for multiple files

**Error 8: Encoding Issues (Special Characters)**
- **Message**: "Invalid character in row 123"
- **Cause**: Non-UTF-8 encoding
- **Solution**:
  1. Open file in Excel
  2. Save As > CSV UTF-8
  3. Re-upload
  4. Avoid copying from Word/PDF (hidden characters)

**💡 Tip**: Most errors are format-related. Using platform templates prevents 90% of errors.

### Validation Warning Resolution

**Warning 1: Missing Quantity**
- **Message**: "Row 88: Missing quantity - will use spend-based calculation"
- **Impact**: Lower data quality (spend-based instead of activity-based)
- **Action**:
  - Optional: Add quantity if available
  - Or accept warning - spend-based calculation still works
  - Consider collecting quantity for future uploads

**Warning 2: Unusual Spend Amount**
- **Message**: "Row 127: Spend amount $950,000 seems high - please verify"
- **Impact**: Potential data entry error (extra zero?)
- **Action**:
  1. Verify amount is correct
  2. If correct, add note: "Large capital equipment purchase"
  3. If error, correct amount
  - Platform flags as outlier for review

**Warning 3: Supplier Not Recognized**
- **Message**: "Row 45: Supplier 'XYZ Corp' not in database - will create new"
- **Impact**: May create duplicate if similar supplier exists
- **Action**:
  1. Check existing suppliers: Search for similar names
  2. If match exists, update supplier name in file
  3. If genuinely new, accept warning

**Warning 4: Category Auto-Assigned (Low Confidence)**
- **Message**: "Row 234: Category auto-assigned to 'Category 1' (65% confidence) - please verify"
- **Impact**: May be wrong category
- **Action**:
  1. Review product description
  2. Verify category is correct
  3. Change if needed
  4. Accept if correct - adds to training data

[Screenshot: Warning resolution interface]

**💡 Tip**: Warnings don't block upload. Review and fix high-impact warnings (unusual amounts, low-confidence categories). Low-impact warnings can be addressed later.

### Correcting Data After Upload

**Option 1: Edit Individual Transactions**

1. **Navigate to Data** > **View Data**
2. **Find transaction** (search by ID, supplier, date)
3. **Click "Edit"**
4. **Modify fields**
5. **Save changes**
6. **Platform recalculates** emissions automatically

**Use Case**: Fix small number of errors (1-10 transactions)

**Option 2: Bulk Update**

1. **Navigate to Data** > **View Data**
2. **Filter** to transactions needing updates
3. **Select all** (or select specific rows)
4. **Click "Bulk Actions"** > **"Update Fields"**
5. **Choose field** to update
6. **Enter new value** (applies to all selected)
7. **Confirm**

**Use Case**: Fix common error across many transactions (e.g., wrong category applied to all transactions from one supplier)

[Screenshot: Bulk update interface]

**Option 3: Delete and Re-Upload**

1. **Navigate to Data** > **Upload History**
2. **Find problematic upload**
3. **Click "Delete Upload"**
4. **Confirm deletion** (cannot be undone)
5. **Fix errors** in original file
6. **Re-upload** corrected file

**Use Case**: Major errors affecting most rows, easier to fix in source file

**⚠️ Warning**: Deleting upload also deletes all associated emissions calculations. Only delete if truly necessary.

**Option 3: Supersede Upload**

1. **Navigate to Data** > **Upload History**
2. **Find upload to replace**
3. **Click "Supersede"**
4. **Upload corrected file**
5. **Platform archives** old data (for audit trail)
6. **New data** used for reporting

**Use Case**: Update data while maintaining history (e.g., revised spend amounts from finance)

**💡 Tip**: Use Edit for small fixes, Bulk Update for systematic errors, Supersede for maintaining audit trail.

---

## Section 6: Upload History and Tracking

### Viewing Upload History

**Access Upload History**:
1. Navigate to **Data** > **Upload History**
2. **View all uploads**:
   - Upload name and description
   - Date uploaded
   - Uploaded by (user)
   - Status (Processing, Complete, Failed, Deleted)
   - Rows imported
   - Date range of data

[Screenshot: Upload history table]

**Filter and Search**:
- **Date Range**: Filter by upload date or data date range
- **Status**: Show only complete, failed, or processing
- **Uploaded By**: Filter by user (if multiple users upload data)
- **Data Source**: Filter by ERP system or source

**Sort**:
- By upload date (most recent first)
- By data date range
- By number of rows
- By user

### Upload Details

**Click any upload** to view details:

**Overview Tab**:
- Upload metadata (name, description, dates)
- Source information
- Status and progress
- Row count and statistics
- Data quality summary
- Associated reports (which reports use this data)

**Data Tab**:
- View all imported transactions
- Searchable and filterable
- Export to Excel for analysis

**Validation Tab**:
- Original validation results
- Errors encountered (if any)
- Warnings logged
- Actions taken

**Activity Log Tab**:
- Upload timestamp
- Validation timestamp
- Import completion timestamp
- Any edits or updates
- Who performed each action

[Screenshot: Upload details view with all tabs]

### Managing Uploads

**Actions Available**:

**View Data**:
- See all transactions from this upload
- Analyze and filter
- Export subset

**Edit Upload Metadata**:
- Change upload name or description
- Update reporting period assignment
- Add notes

**Download Source File**:
- Retrieve original upload file
- Useful for audit or correction

**Export Processed Data**:
- Download data as imported (after mapping and validation)
- Includes any auto-corrections
- Excel or CSV format

**Supersede**:
- Replace with corrected version
- Maintains audit trail

**Delete**:
- Remove upload and all associated data
- Cannot be undone
- Only for Draft status uploads

**Lock/Unlock**:
- Lock: Prevents editing (for finalized data)
- Unlock: Allow corrections (with permissions)

[Screenshot: Upload actions menu]

**💡 Tip**: Lock uploads once associated reports are finalized to prevent accidental changes.

### Audit Trail and Compliance

**Complete Audit Trail**:

Platform maintains comprehensive audit logs:
- ✅ Who uploaded data (user ID and email)
- ✅ When uploaded (timestamp)
- ✅ Source file (archived)
- ✅ Original data (before validation/corrections)
- ✅ Processed data (after mapping)
- ✅ All changes (edits, updates, deletions)
- ✅ Who made changes (user tracking)
- ✅ Which reports use the data (linkage)

**Compliance Support**:
- SOC 2 Type II audit trail requirements
- ISO 14064-3 verification support
- CDP assurance evidence
- Internal audit support

**Export Audit Trail**:
1. Navigate to **Data** > **Upload History**
2. Select upload(s)
3. Click **"Export Audit Trail"**
4. Download comprehensive audit package:
   - Source files
   - Validation reports
   - Change logs
   - User activity
   - Timestamp evidence

[Screenshot: Audit trail export package]

**💡 Tip**: Regular audit trail exports (quarterly) provide backup and facilitate assurance processes.

### Data Retention and Deletion

**Retention Policy**:
- Active data: Indefinite retention
- Deleted uploads: Archived for 7 years (recoverable)
- Source files: Retained for 10 years
- Audit logs: Retained for 10 years

**Data Deletion**:
Users cannot permanently delete data that has been:
- Included in finalized reports
- Submitted for assurance
- Used in regulatory submissions

**Administrators can**:
- Archive old data (remove from active use but preserve)
- Export and delete (with proper justification and approval)

**GDPR Compliance**:
- Right to deletion supported (with restrictions for compliance data)
- Data minimization principles followed
- Only collect necessary data

**💡 Tip**: Don't delete data unless absolutely necessary. Archived data doesn't count against storage limits.

---

## Troubleshooting

### Common Issues and Solutions

**Issue**: Upload fails with "Unknown Error"
**Solution**:
1. Check internet connection
2. Verify file isn't corrupted (open locally to confirm)
3. Try smaller batch size (split file)
4. Clear browser cache and retry
5. Try different browser (Chrome recommended)
6. Check system status page: status.greenlang.io
7. Contact support with error code

**Issue**: Upload stuck at "Processing" for >10 minutes
**Solution**:
1. Refresh page (processing continues in background)
2. Check notification center for completion alert
3. Large files (>10,000 rows) can take 5-10 minutes
4. If still stuck after 20 minutes, contact support
5. Don't re-upload (creates duplicates)

**Issue**: Emission calculations seem wrong after upload
**Solution**:
1. Check data quality score - low score indicates estimates
2. Verify units are correct (kg vs. lbs can cause 2x error)
3. Review category assignments - wrong category affects factors
4. Check for duplicate uploads (would double emissions)
5. Navigate to dashboard > click emission category > drill into calculations
6. Use "Recalculate" button to refresh
7. Contact support if still incorrect

**Issue**: Cannot find uploaded data in dashboard
**Solution**:
1. Verify upload status is "Complete" not "Processing"
2. Check date range filter on dashboard matches upload dates
3. Verify business unit filter (if segmented)
4. Navigate to Data > View Data to confirm import
5. Check if upload was set to "Draft" (drafts excluded from dashboards)
6. Refresh dashboard (use refresh button)

**Issue**: Supplier names not matching correctly
**Solution**:
1. Review supplier master in Settings > Suppliers
2. Add aliases for common variations
3. During upload, review matching suggestions carefully
4. Accept high-confidence matches (>90%)
5. Manually map medium-confidence matches (70-90%)
6. Reject low-confidence matches (<70%)
7. Clean supplier names in source system for future uploads

**Issue**: UTF-8 encoding errors (special characters garbled)
**Solution**:
1. Open file in Excel
2. File > Save As
3. Select "CSV UTF-8" (not regular CSV)
4. Confirm special characters display correctly
5. Re-upload
6. Alternative: Use Excel format (.xlsx) which handles encoding better

**Issue**: File size limit exceeded
**Solution**:
1. Check file size: Must be <50MB
2. If Excel file:
   - Remove formatting (colors, fonts)
   - Remove formulas (convert to values)
   - Remove images or embedded objects
   - Save as CSV (typically 70% smaller)
3. Split into multiple files by:
   - Month
   - Category
   - Supplier
4. Use batch upload for multiple files

**Issue**: Uploaded wrong file by mistake
**Solution**:
1. If upload still processing: Click "Cancel"
2. If already imported:
   - Navigate to Data > Upload History
   - Find incorrect upload
   - Click "Delete" (if draft status)
   - Or click "Supersede" to replace (if already used in reports)
3. Upload correct file
4. Verify new data in dashboard

---

## FAQ

**Q: How often should I upload data?**
**A**: Best practice is monthly. This provides timely emission tracking and allows early identification of issues. For compliance reporting, ensure all data for the reporting period is uploaded before generating reports. Automated integrations can upload daily.

**Q: What's the maximum file size?**
**A**: 50MB per file. For larger datasets:
- Split into multiple files
- Use CSV format (smaller than Excel)
- Use batch upload
- Contact support for API integration (no size limit)

**Q: Can I upload data retroactively?**
**A**: Yes. Upload historical data for any period within last 10 years. Useful for:
- Establishing base year
- Tracking multi-year trends
- Initial platform setup
Ensure date field reflects actual transaction date, not upload date.

**Q: What happens if I upload the same data twice?**
**A**: Depends on duplicate handling setting:
- **Skip duplicates**: Second upload ignored (based on Transaction ID)
- **Update existing**: Second upload overwrites first (useful for corrections)
- **Import as new**: Creates duplicate records (use carefully)
Platform detects duplicates by Transaction ID.

**Q: How do I handle multi-currency data?**
**A**: Platform handles automatically:
- Upload data in original currencies (USD, EUR, GBP, etc.)
- Platform converts to your reporting currency using daily exchange rates
- Conversion rates stored for audit trail
- Can view in original or converted currency

**Q: Can I upload negative amounts (returns, credits)?**
**A**: Yes. Use negative values for:
- Purchase returns
- Credits or rebates
- Adjustments
Emissions calculated on net amounts (positive - negative).

**Q: What if I don't know the GHG Protocol category?**
**A**: Options:
1. Use platform's AI auto-categorization (suggested based on description)
2. Leave blank and review suggestions during upload
3. Default to Category 1 (Purchased Goods and Services) - most general
4. Review category mapping guide in template
5. Platform learns from your corrections

**Q: How detailed should product descriptions be?**
**A**: More detail = better emission factor matching:
- **Minimum**: "Office supplies"
- **Better**: "Paper, printing supplies, stationery"
- **Best**: "Recycled copy paper, 500 sheets per ream"
AI uses descriptions to match specific emission factors.

**Q: Can multiple users upload data?**
**A**: Yes. User permissions control who can:
- Upload data (Contributor role and above)
- Edit uploaded data (Contributor role)
- Delete uploads (Admin role only)
Audit trail tracks which user performed each action.

**Q: How do I upload travel data from booking tool?**
**A**: Most travel tools export to Excel or CSV:
1. Export travel bookings (flights, hotels, rental cars)
2. Ensure export includes: dates, destinations, travel class, distance
3. Use platform's Travel Data template
4. Map travel tool columns to template
5. Upload
Platform calculates emissions based on DEFRA/EPA factors.

**Q: What if my ERP uses different category names?**
**A**: During upload mapping:
1. Platform suggests GHG Protocol category based on your categories
2. Review and confirm mappings
3. Save mapping template for future uploads
Example: Your "Raw Materials" → Platform "Category 1"

---

## Related Resources

### Within Platform
- [Getting Started Guide](./GETTING_STARTED.md) - Platform overview
- [Supplier Portal Guide](./SUPPLIER_PORTAL_GUIDE.md) - Collect supplier PCF data for higher quality
- [Reporting Guide](./REPORTING_GUIDE.md) - Generate reports from uploaded data
- [Dashboard Usage Guide](./DASHBOARD_USAGE_GUIDE.md) - Analyze uploaded data

### Video Tutorials
- "Data Upload Walkthrough" (15 min) - Step-by-step guide
- "Handling Common Upload Errors" (10 min) - Troubleshooting
- "Improving Data Quality" (12 min) - Best practices
- "Setting Up ERP Integration" (20 min) - Automated uploads

### Templates and Tools
- **Excel Templates**: Data > Upload Data > Download Template
- **CSV Templates**: Available for each category
- **Mapping Templates**: Save and reuse custom mappings
- **Validation Checklist**: Pre-upload quality check

### Support
- **Help Center**: Click "?" icon > Data Upload Help
- **Live Chat**: Mon-Fri 9am-5pm EST
- **Email Support**: data-support@greenlang.io
- **Response Time**: Within 24 hours (usually faster)

### External Resources
- **GHG Protocol Scope 3 Category Guidance**: Detailed descriptions of all 15 categories
- **Procurement Data Standards**: Recommendations for ERP data export
- **ISO 14064-1**: GHG accounting standard
- **Data Quality Framework**: WBCSD guidance on emission data quality

---

**Success Tips for Data Uploads**:
1. ✅ Use platform templates - prevents 90% of errors
2. ✅ Start small - test with 100 rows before full upload
3. ✅ Clean data at source - fix ERP data quality issues
4. ✅ Document process - create internal upload procedures
5. ✅ Upload regularly - monthly uploads easier than annual
6. ✅ Engage IT team - set up automated integrations
7. ✅ Focus on quality - complete data beats fast data

For questions: data-support@greenlang.io

---

*Last Updated: 2025-11-07*
*Document Version: 1.0*
*Platform Version: 2.5.0*
