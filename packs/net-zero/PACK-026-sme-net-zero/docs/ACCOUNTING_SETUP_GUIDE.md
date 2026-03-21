# PACK-026 Accounting Software Setup Guide

Connect your accounting software to automatically import financial data for emissions calculations. PACK-026 supports Xero, QuickBooks Online, and Sage.

**Estimated setup time:** 15-30 minutes per provider.

---

## Table of Contents

1. [Overview](#overview)
2. [Xero Setup](#xero-setup)
3. [QuickBooks Online Setup](#quickbooks-online-setup)
4. [Sage Setup](#sage-setup)
5. [GL Code to Scope 3 Category Mapping](#gl-code-to-scope-3-category-mapping)
6. [Customizing Spend Classification](#customizing-spend-classification)
7. [Troubleshooting Common Issues](#troubleshooting-common-issues)
8. [Data Privacy and Security](#data-privacy-and-security)
9. [Sync Options](#sync-options)

---

## Overview

### Why Connect Your Accounting Software?

Connecting your accounting software to PACK-026 provides three benefits:

1. **Automatic data import** -- No manual data entry; spend categories are pulled directly from your chart of accounts
2. **Better accuracy** -- Silver-tier baseline (+/- 15%) instead of Bronze (+/- 40%)
3. **Ongoing tracking** -- Automatic progress updates as new transactions flow in

### What Data Does PACK-026 Access?

PACK-026 requests **read-only** access to:

| Data Type | What We Read | What We Do Not Read |
|-----------|-------------|-------------------|
| Profit & Loss | Revenue, expense categories, account totals | Individual transaction details (optional) |
| Chart of Accounts | Account names, codes, and types | Bank account numbers |
| Transactions (optional) | Amounts, categories, dates | Payee bank details, personal data |

We **never** access:
- Bank account numbers or sort codes
- Payment card details
- Employee salary details
- Customer personal information
- Your accounting login credentials (OAuth2 means we never see your password)

### Supported Providers

| Provider | Authentication | API Version | Free Plan Support |
|----------|---------------|-------------|-------------------|
| Xero | OAuth2 | Xero API v2 | Yes |
| QuickBooks Online | OAuth2 | QBO API v3 | Yes |
| Sage | OAuth2 / API Key | Sage Business Cloud | Yes |

---

## Xero Setup

### Prerequisites

- A Xero account (any plan: Starter, Standard, or Premium)
- Admin or Standard user access to Xero
- A GreenLang instance with PACK-026 installed

### Step 1: Create a Xero App (5 minutes)

1. Go to [developer.xero.com/app/manage](https://developer.xero.com/app/manage)
2. Click **New app**
3. Fill in the form:
   - **App name:** `GreenLang Carbon Tracker` (or any name you prefer)
   - **Integration type:** Web app
   - **Company or application URL:** Your GreenLang instance URL (e.g., `https://your-company.greenlang.io`)
   - **Redirect URI:** `https://your-company.greenlang.io/callback/xero`
4. Click **Create app**
5. Note your **Client ID** and generate a **Client Secret**

### Step 2: Configure PACK-026 (5 minutes)

```python
from packs.net_zero.PACK_026_sme_net_zero.integrations.xero_integration import XeroIntegration

xero = XeroIntegration(
    client_id="YOUR_XERO_CLIENT_ID",
    client_secret="YOUR_XERO_CLIENT_SECRET",
    redirect_uri="https://your-company.greenlang.io/callback/xero",
    scopes=[
        "openid",
        "profile",
        "accounting.transactions.read",
        "accounting.reports.read",
        "accounting.settings.read",
    ],
)
```

Or via CLI:

```bash
greenlang pack-026 setup-accounting \
    --provider xero \
    --client-id YOUR_XERO_CLIENT_ID \
    --client-secret YOUR_XERO_CLIENT_SECRET \
    --redirect-uri https://your-company.greenlang.io/callback/xero
```

### Step 3: Authenticate (2 minutes)

```python
# This opens a browser window for Xero OAuth2 consent
auth_url = await xero.get_authorization_url()
print(f"Open this URL in your browser: {auth_url}")

# After granting consent, Xero redirects with an authorization code
# The integration handles the token exchange automatically
await xero.handle_callback(authorization_code="CODE_FROM_REDIRECT")
```

### Step 4: Pull Financial Data (3 minutes)

```python
# Pull Profit & Loss for the last financial year
pl_data = await xero.pull_profit_and_loss(
    from_date="2025-01-01",
    to_date="2025-12-31",
    periods=1,  # Annual summary
)

print(f"Revenue: GBP {pl_data.revenue:,.0f}")
print(f"Total expenses: GBP {pl_data.total_expenses:,.0f}")
print(f"Expense categories: {len(pl_data.categories)}")

# View category breakdown
for cat in pl_data.categories:
    print(f"  {cat.code} - {cat.name}: GBP {cat.amount:,.0f}")
```

### Step 5: Map Chart of Accounts (5 minutes)

PACK-026 auto-maps your Xero accounts to emission categories. Review and customize:

```python
# Auto-classify spend into emission categories
classified = await xero.classify_spend(
    pl_data,
    mapping="default_xero_mapping",
)

# Review the mapping
for mapping in classified.mappings:
    print(f"  {mapping.account_code} ({mapping.account_name})")
    print(f"    -> Scope 3 Category: {mapping.scope3_category}")
    print(f"    -> Amount: GBP {mapping.amount:,.0f}")
    print(f"    -> Confidence: {mapping.confidence}")
```

### Xero Permissions Required

| Scope | Description | Required |
|-------|-------------|----------|
| `accounting.transactions.read` | Read bank transactions and invoices | Yes |
| `accounting.reports.read` | Read Profit & Loss and Balance Sheet | Yes |
| `accounting.settings.read` | Read Chart of Accounts | Yes |
| `openid` | Basic authentication | Yes |
| `profile` | Organization name | Yes |

---

## QuickBooks Online Setup

### Prerequisites

- A QuickBooks Online account (Simple Start, Essentials, or Plus)
- Admin access to QuickBooks
- A GreenLang instance with PACK-026 installed

### Step 1: Create a QuickBooks App (5 minutes)

1. Go to [developer.intuit.com/app/developer/dashboard](https://developer.intuit.com/app/developer/dashboard)
2. Click **Create an app**
3. Select **QuickBooks Online and Payments**
4. Fill in the form:
   - **App name:** `GreenLang Carbon Tracker`
   - **Scope:** Select **Accounting** (read-only is sufficient)
5. In the **Keys & OAuth** section, note your **Client ID** and **Client Secret**
6. Add your redirect URI: `https://your-company.greenlang.io/callback/quickbooks`

### Step 2: Configure PACK-026 (5 minutes)

```python
from packs.net_zero.PACK_026_sme_net_zero.integrations.quickbooks_integration import QuickBooksIntegration

qb = QuickBooksIntegration(
    client_id="YOUR_QB_CLIENT_ID",
    client_secret="YOUR_QB_CLIENT_SECRET",
    redirect_uri="https://your-company.greenlang.io/callback/quickbooks",
    environment="production",  # Use "sandbox" for testing
)
```

Or via CLI:

```bash
greenlang pack-026 setup-accounting \
    --provider quickbooks \
    --client-id YOUR_QB_CLIENT_ID \
    --client-secret YOUR_QB_CLIENT_SECRET \
    --redirect-uri https://your-company.greenlang.io/callback/quickbooks \
    --environment production
```

### Step 3: Authenticate (2 minutes)

```python
auth_url = await qb.get_authorization_url()
print(f"Open this URL in your browser: {auth_url}")

# After granting consent in QuickBooks
await qb.handle_callback(
    authorization_code="CODE_FROM_REDIRECT",
    realm_id="YOUR_COMPANY_REALM_ID",
)
```

### Step 4: Pull P&L Data (3 minutes)

```python
# Pull Profit & Loss report
pl_data = await qb.pull_profit_and_loss(
    from_date="2025-01-01",
    to_date="2025-12-31",
    accounting_method="accrual",  # or "cash"
)

# Auto-classify spend
classified = await qb.classify_spend(
    pl_data,
    mapping="default_quickbooks_mapping",
)
```

### Step 5: Map Categories (5 minutes)

QuickBooks uses a different account structure from Xero. PACK-026 handles the mapping:

```python
# QuickBooks account types -> Emission categories
# "Expense" accounts are auto-mapped by name and category
# "Cost of Goods Sold" -> Scope 3 Category 1

for mapping in classified.mappings:
    print(f"  {mapping.account_type}: {mapping.account_name}")
    print(f"    -> {mapping.scope3_category}: GBP {mapping.amount:,.0f}")
```

### QuickBooks Permissions Required

| Scope | Description | Required |
|-------|-------------|----------|
| `com.intuit.quickbooks.accounting` | Read accounting data | Yes |

---

## Sage Setup

### Prerequisites

- A Sage Business Cloud account (Sage Accounting or Sage 50)
- Admin access to Sage
- A GreenLang instance with PACK-026 installed

### Step 1: Register with Sage Developer (5 minutes)

1. Go to [developer.sage.com](https://developer.sage.com)
2. Create a developer account (if you do not have one)
3. Click **Create app**
4. Fill in the form:
   - **App name:** `GreenLang Carbon Tracker`
   - **Callback URL:** `https://your-company.greenlang.io/callback/sage`
5. Note your **Client ID**, **Client Secret**, and **Signing Secret**

### Step 2: Configure PACK-026 (5 minutes)

```python
from packs.net_zero.PACK_026_sme_net_zero.integrations.sage_integration import SageIntegration

sage = SageIntegration(
    client_id="YOUR_SAGE_CLIENT_ID",
    client_secret="YOUR_SAGE_CLIENT_SECRET",
    redirect_uri="https://your-company.greenlang.io/callback/sage",
    country="GB",  # or "US", "DE", "FR", etc.
)
```

Or via CLI:

```bash
greenlang pack-026 setup-accounting \
    --provider sage \
    --client-id YOUR_SAGE_CLIENT_ID \
    --client-secret YOUR_SAGE_CLIENT_SECRET \
    --redirect-uri https://your-company.greenlang.io/callback/sage
```

### Step 3: Authenticate (2 minutes)

```python
auth_url = await sage.get_authorization_url()
print(f"Open this URL in your browser: {auth_url}")

await sage.handle_callback(authorization_code="CODE_FROM_REDIRECT")
```

### Step 4: Pull Nominal Ledger Data (3 minutes)

```python
# Pull nominal ledger (Sage equivalent of Chart of Accounts + P&L)
nominal_data = await sage.pull_nominal_ledger(
    from_date="2025-01-01",
    to_date="2025-12-31",
)

# Auto-classify by nominal code
classified = await sage.classify_spend(
    nominal_data,
    mapping="default_sage_mapping",
)
```

### Step 5: Map Nominal Codes (5 minutes)

Sage uses nominal codes (similar to GL codes). PACK-026 provides default mappings:

```python
# Sage nominal codes -> Emission categories
# 5000-5999 (Purchases) -> Scope 3 Cat 1
# 6200-6299 (Travel) -> Scope 3 Cat 6
# 7000-7099 (Overheads) -> Scope 3 Cat 1

for mapping in classified.mappings:
    print(f"  {mapping.nominal_code}: {mapping.nominal_name}")
    print(f"    -> {mapping.scope3_category}: GBP {mapping.amount:,.0f}")
```

### Sage Permissions Required

| Scope | Description | Required |
|-------|-------------|----------|
| `full_access` | Read-only access to accounts and transactions | Yes |

---

## GL Code to Scope 3 Category Mapping

### Default Mapping Table

PACK-026 provides a default mapping from GL/nominal codes to GHG Protocol Scope 3 categories. This mapping works across Xero, QuickBooks, and Sage.

| GL Code Range | Typical Account Name | GHG Scope | GHG Category | Emission Factor Source |
|:------------:|---------------------|-----------|-------------|----------------------|
| 4000-4999 | Sales / Revenue | N/A | Not an emission source | -- |
| 5000-5099 | Raw Materials / Inventory | Scope 3 | Cat 1: Purchased Goods | DEFRA EEIO by SIC |
| 5100-5199 | Direct Labour (Production) | N/A | Not an emission source | -- |
| 5200-5299 | Subcontractor Costs | Scope 3 | Cat 1: Purchased Services | DEFRA EEIO by SIC |
| 5300-5399 | Packaging Materials | Scope 3 | Cat 1: Purchased Goods | DEFRA EEIO |
| 5400-5499 | Freight / Carriage Inward | Scope 3 | Cat 4: Upstream Transport | DEFRA transport EF |
| 5500-5599 | Import Duties / Customs | N/A | Not an emission source | -- |
| 5900-5999 | Other Direct Costs | Scope 3 | Cat 1: Purchased Services | DEFRA EEIO |
| 6000-6049 | Rent | N/A | Not an emission source (Scope 1/2 if energy included) | -- |
| 6050-6099 | Business Rates / Property Tax | N/A | Not an emission source | -- |
| 6100-6149 | Electricity | Scope 2 | Purchased Electricity | DEFRA grid EF |
| 6150-6199 | Gas | Scope 1 | Natural Gas Combustion | DEFRA gas EF |
| 6200-6249 | Domestic Travel (Rail/Car) | Scope 3 | Cat 6: Business Travel | DEFRA transport EF |
| 6250-6299 | International Travel (Air) | Scope 3 | Cat 6: Business Travel | DEFRA air EF |
| 6300-6349 | Company Vehicle Fuel | Scope 1 | Mobile Combustion | DEFRA fuel EF |
| 6350-6399 | Vehicle Lease / Insurance | N/A | Not an emission source | -- |
| 6400-6449 | Delivery / Distribution | Scope 3 | Cat 9: Downstream Transport | DEFRA transport EF |
| 6450-6499 | Courier / Postal | Scope 3 | Cat 4: Upstream Transport | DEFRA EEIO |
| 6500-6549 | Staff Entertaining | Scope 3 | Cat 6: Business Travel | DEFRA EEIO |
| 6550-6599 | Subsistence / Meals | Scope 3 | Cat 6: Business Travel | DEFRA EEIO |
| 6600-6699 | Advertising / Marketing | Scope 3 | Cat 1: Purchased Services | DEFRA EEIO |
| 6700-6799 | IT / Software / Hosting | Scope 3 | Cat 1: Purchased Services | DEFRA EEIO |
| 6800-6849 | Telephone / Mobile | Scope 3 | Cat 1: Purchased Services | DEFRA EEIO |
| 6850-6899 | Internet / Data | Scope 3 | Cat 1: Purchased Services | DEFRA EEIO |
| 6900-6999 | Bank Charges / Finance Costs | N/A | Not an emission source | -- |
| 7000-7049 | Office Supplies / Stationery | Scope 3 | Cat 1: Purchased Goods | DEFRA EEIO |
| 7050-7099 | Printing / Copying | Scope 3 | Cat 1: Purchased Goods | DEFRA EEIO |
| 7100-7149 | Legal Fees | Scope 3 | Cat 1: Purchased Services | DEFRA EEIO |
| 7150-7199 | Accountancy Fees | Scope 3 | Cat 1: Purchased Services | DEFRA EEIO |
| 7200-7249 | Consultancy Fees | Scope 3 | Cat 1: Purchased Services | DEFRA EEIO |
| 7250-7299 | Training / Conferences | Scope 3 | Cat 6: Business Travel | DEFRA EEIO |
| 7300-7399 | Insurance | N/A | Not an emission source | -- |
| 7400-7499 | Cleaning / Maintenance | Scope 3 | Cat 1: Purchased Services | DEFRA EEIO |
| 7500-7599 | Waste Disposal / Recycling | Scope 3 | Cat 5: Waste | DEFRA waste EF |
| 7600-7699 | Repairs / Maintenance | Scope 3 | Cat 1: Purchased Services | DEFRA EEIO |
| 7700-7799 | Equipment Hire | Scope 3 | Cat 1: Purchased Services | DEFRA EEIO |
| 7800-7899 | Subscriptions / Memberships | Scope 3 | Cat 1: Purchased Services | DEFRA EEIO |
| 7900-7999 | Depreciation / Amortization | N/A | Not an emission source | -- |
| 8000-8999 | Capital Expenditure | Scope 3 | Cat 2: Capital Goods | DEFRA EEIO |
| 9000-9999 | Tax / Dividends | N/A | Not an emission source | -- |

### Example: Mapping a Typical Office SME

```
Acme Consulting Ltd -- Xero P&L Year 2025

Account Code  Account Name              Amount (GBP)   Scope 3 Category
-----------  -------------------------  ------------   ----------------
5200         Subcontractors              45,000         Cat 1 (Services)
6100         Electricity                 12,000         Scope 2
6150         Gas                          4,000         Scope 1
6200         UK Rail Travel               3,500         Cat 6 (Travel)
6250         International Flights        4,500         Cat 6 (Travel)
6300         Company Car Fuel             3,000         Scope 1
6700         Software Subscriptions       8,000         Cat 1 (Services)
6800         Mobile Phone Contracts       2,400         Cat 1 (Services)
7000         Office Supplies              1,200         Cat 1 (Goods)
7150         Accountancy Fees             6,000         Cat 1 (Services)
7400         Office Cleaning              3,600         Cat 1 (Services)
7500         Waste Collection             1,200         Cat 5 (Waste)
7800         Professional Memberships       800         Cat 1 (Services)

SUMMARY:
  Scope 1 (Gas + Fuel):    GBP 7,000   ->  ~5.2 tCO2e
  Scope 2 (Electricity):   GBP 12,000  ->  ~8.9 tCO2e
  Scope 3 Cat 1 (Purch):   GBP 67,000  ->  ~4.1 tCO2e (EEIO factor: 0.062 tCO2e/GBP 1000)
  Scope 3 Cat 5 (Waste):   GBP 1,200   ->  ~0.7 tCO2e
  Scope 3 Cat 6 (Travel):  GBP 8,000   ->  ~2.1 tCO2e
  TOTAL:                                    ~21.0 tCO2e
```

---

## Customizing Spend Classification

### Override the Default Mapping

If your chart of accounts uses non-standard codes, customize the mapping:

```python
# Define custom mappings for specific accounts
custom_mappings = {
    "4100": {"scope": None, "category": None, "reason": "Revenue, not an emission"},
    "5500": {"scope": "scope3", "category": "cat1", "reason": "Custom: raw materials"},
    "6666": {"scope": "scope1", "category": "gas", "reason": "Custom: heating gas account"},
    "7777": {"scope": "scope3", "category": "cat6", "reason": "Custom: team travel budget"},
}

classified = await xero.classify_spend(
    pl_data,
    mapping="default_xero_mapping",
    custom_overrides=custom_mappings,
)
```

### Add New Account Mappings

```python
# Add a mapping for a new account code
await xero.add_mapping(
    account_code="6175",
    account_name="LPG / Heating Oil",
    scope="scope1",
    category="heating_oil",
    emission_factor_source="DEFRA",
    emission_factor_value=0.24687,  # kgCO2e per kWh
)
```

### Exclude Accounts from Classification

```python
# Exclude specific accounts (e.g., intercompany, tax, dividends)
await xero.exclude_accounts(
    account_codes=["9000", "9100", "9200", "9999"],
    reason="Tax and dividend accounts -- no emissions",
)
```

---

## Troubleshooting Common Issues

### "OAuth2 consent failed" / "Redirect URI mismatch"

**Cause:** The redirect URI in your PACK-026 config does not match the one registered with your accounting provider.

**Fix:**
1. Check the exact redirect URI in your Xero/QuickBooks/Sage developer dashboard
2. Ensure it matches character-for-character in your PACK-026 config (including `https://` and trailing slash)
3. Common mistake: `http://` vs `https://` -- accounting providers require HTTPS

### "Token expired" / "401 Unauthorized"

**Cause:** The OAuth2 access token has expired (typically after 30-60 minutes).

**Fix:** PACK-026 auto-refreshes tokens. If auto-refresh fails:
```python
# Force re-authentication
await xero.reauthenticate()
```

### "Rate limit exceeded" / "429 Too Many Requests"

**Cause:** You have exceeded the API rate limit for your accounting provider.

**Fix:**
- Xero: 60 calls per minute (PACK-026 uses 5-10 calls per sync)
- QuickBooks: 500 calls per minute (PACK-026 uses 3-5 calls per sync)
- Sage: 300 calls per minute (PACK-026 uses 3-5 calls per sync)

If you see 429 errors, PACK-026 automatically backs off and retries. Wait 1 minute and try again.

### "No data returned" for P&L report

**Cause:** The date range does not match your accounting period, or the P&L has not been finalized.

**Fix:**
1. Check your financial year end date in your accounting software
2. Ensure the date range covers a complete period
3. Try pulling data for a prior period that has been finalized

### "Classification confidence: LOW" for many accounts

**Cause:** Your chart of accounts uses non-standard naming or codes that do not match default mappings.

**Fix:**
1. Review the low-confidence mappings: `classified.get_low_confidence_mappings()`
2. Add custom overrides for your specific account structure
3. Contact support@greenlang.io for help with custom mapping

### "Multi-currency transactions not supported"

**Cause:** Your accounting software has transactions in multiple currencies.

**Fix:** PACK-026 converts all amounts to your base currency (set in your accounting software) before classification. If you see this error:
1. Ensure your base currency is set correctly in your accounting software
2. Set the currency in PACK-026: `config.currency = "GBP"`

---

## Data Privacy and Security

### What Data Is Stored

| Data | Stored In | Encrypted | Retention |
|------|-----------|-----------|-----------|
| OAuth2 access token | HashiCorp Vault | AES-256 | Until revoked |
| OAuth2 refresh token | HashiCorp Vault | AES-256 | Until revoked |
| P&L category totals | PostgreSQL | AES-256-GCM | Until deleted by user |
| Classified spend | PostgreSQL | AES-256-GCM | Until deleted by user |
| Individual transactions | NOT stored | N/A | Processed in memory only |
| Account names/codes | PostgreSQL | At rest encryption | Until deleted by user |

### What Data Is NOT Stored

- Login credentials (passwords) -- OAuth2 only
- Bank account numbers
- Payment card details
- Individual invoice line items (only category totals)
- Customer or supplier names
- Employee personal data

### Revoking Access

You can revoke PACK-026's access to your accounting software at any time:

1. **In PACK-026:**
   ```python
   await xero.revoke_access()
   ```

2. **In your accounting software:**
   - Xero: Settings > Connected Apps > Remove GreenLang
   - QuickBooks: Gear > Account Settings > Connected Apps > Disconnect
   - Sage: Settings > Connected Apps > Revoke

3. **Delete stored data:**
   ```python
   await xero.delete_all_financial_data(entity_id="your-entity-id")
   ```

### GDPR Compliance

- PACK-026 processes financial data under the **legitimate interest** legal basis for emission calculation
- Data Subject Access Requests (DSARs) supported: export all stored data for an entity
- Right to erasure: delete all stored financial data with a single API call
- No data is transferred to third parties
- All data processing occurs within your GreenLang instance

---

## Sync Options

### Monthly Batch Sync (Free Tier)

```python
# Sync once per month (recommended for annual reporting)
await xero.configure_sync(
    frequency="monthly",
    sync_day=1,              # 1st of each month
    lookback_months=1,       # Pull last month's data
    auto_classify=True,      # Auto-map to emission categories
)
```

- **Frequency:** Once per month, on the 1st
- **Data pulled:** Previous calendar month
- **Best for:** Annual reporting, SME Climate Hub, free tier

### Weekly Batch Sync (Premium)

```python
# Sync weekly (recommended for quarterly reporting)
await xero.configure_sync(
    frequency="weekly",
    sync_day="monday",       # Every Monday
    lookback_days=7,         # Pull last 7 days
    auto_classify=True,
)
```

- **Frequency:** Every Monday
- **Data pulled:** Previous 7 days
- **Best for:** Quarterly progress reviews, Silver tier

### Real-Time Sync (Premium)

```python
# Sync in near-real-time via webhooks
await xero.configure_sync(
    frequency="realtime",
    webhook_url="https://your-company.greenlang.io/webhooks/xero",
    event_types=["invoice.created", "expense.created"],
    auto_classify=True,
)
```

- **Frequency:** Within 1 hour of transaction
- **Trigger:** Xero webhooks for new invoices and expenses
- **Best for:** Continuous monitoring, Gold tier, large SMEs

### Manual Sync

```python
# Trigger a manual sync at any time
result = await xero.sync_now(
    from_date="2025-01-01",
    to_date="2025-12-31",
)

print(f"Synced {result.transaction_count} transactions")
print(f"Total spend classified: GBP {result.total_spend:,.0f}")
print(f"Classification confidence: {result.avg_confidence:.0f}%")
```

---

*Accounting Setup Guide -- PACK-026 SME Net Zero Pack v1.0.0*
*For support, contact support@greenlang.io*
