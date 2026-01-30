# PRD: Mass Balance Agent (GL-EUDR-012)

**Agent family:** EUDRTraceabilityFamily
**Layer:** Supply Chain Traceability
**Primary domains:** Mass balance accounting, volume reconciliation, flow tracking
**Priority:** P0 (highest)
**Doc version:** 1.0
**Last updated:** 2026-01-30 (Asia/Kolkata)

---

## 1. Executive Summary

**Mass Balance Agent (GL-EUDR-012)** implements mass balance accounting to track volumes of EUDR-compliant commodities through the supply chain. It ensures that compliant volumes are accurately tracked even when physical segregation is not maintained.

---

## 2. Mass Balance Concept

Mass balance is an accounting system that:
- Tracks **credits** (compliant inputs)
- Tracks **debits** (compliant outputs sold/shipped)
- Maintains **balance** (current compliant inventory)
- Ensures outputs never exceed inputs
- Operates at facility/account level

### EUDR Mass Balance Requirements

Per EUDR Article 10, mass balance can be used when:
- Physical segregation is impractical
- Commodities are fungible
- Proper accounting controls exist
- Audit trail is maintained

---

## 3. Data Model

```sql
-- Mass Balance Accounts
CREATE TABLE mass_balance_accounts (
    account_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id UUID NOT NULL,
    commodity_category VARCHAR(50) NOT NULL,
    commodity_type VARCHAR(100) NOT NULL,

    -- Balances
    current_balance DECIMAL(15,3) NOT NULL DEFAULT 0,
    compliant_balance DECIMAL(15,3) NOT NULL DEFAULT 0,
    non_compliant_balance DECIMAL(15,3) NOT NULL DEFAULT 0,
    quantity_unit VARCHAR(20) NOT NULL,

    -- Period
    period_start DATE,
    period_end DATE,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    last_reconciled TIMESTAMP,
    reconciliation_status VARCHAR(50) DEFAULT 'PENDING',

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(facility_id, commodity_category, commodity_type)
);

-- Mass Balance Ledger Entries
CREATE TABLE mass_balance_entries (
    entry_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID REFERENCES mass_balance_accounts(account_id),

    -- Transaction
    entry_type VARCHAR(20) NOT NULL,  -- CREDIT or DEBIT
    entry_date TIMESTAMP NOT NULL,
    quantity DECIMAL(15,3) NOT NULL,
    quantity_unit VARCHAR(20) NOT NULL,

    -- Classification
    compliance_type VARCHAR(50) NOT NULL,  -- COMPLIANT, NON_COMPLIANT

    -- Reference
    reference_type VARCHAR(50) NOT NULL,  -- RECEIPT, SHIPMENT, PROCESSING, ADJUSTMENT
    reference_id UUID,
    batch_id UUID,
    origin_plots UUID[],

    -- Balance after entry
    running_balance DECIMAL(15,3) NOT NULL,

    -- Audit
    created_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(100),
    notes TEXT,

    CONSTRAINT valid_entry_type CHECK (entry_type IN ('CREDIT', 'DEBIT')),
    CONSTRAINT positive_quantity CHECK (quantity > 0)
);

-- Mass Balance Reconciliations
CREATE TABLE mass_balance_reconciliations (
    reconciliation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID REFERENCES mass_balance_accounts(account_id),
    reconciliation_date TIMESTAMP NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,

    -- Calculated balances
    opening_balance DECIMAL(15,3) NOT NULL,
    total_credits DECIMAL(15,3) NOT NULL,
    total_debits DECIMAL(15,3) NOT NULL,
    calculated_closing DECIMAL(15,3) NOT NULL,

    -- Actual balance
    actual_balance DECIMAL(15,3) NOT NULL,
    variance DECIMAL(15,3) NOT NULL,
    variance_percentage DECIMAL(5,2),

    -- Result
    status VARCHAR(50) NOT NULL,
    issues JSONB DEFAULT '[]',

    reconciled_by VARCHAR(100),

    CONSTRAINT valid_status CHECK (
        status IN ('BALANCED', 'VARIANCE', 'INVESTIGATION_REQUIRED')
    )
);

-- Indexes
CREATE INDEX idx_mb_accounts_facility ON mass_balance_accounts(facility_id);
CREATE INDEX idx_mb_entries_account ON mass_balance_entries(account_id);
CREATE INDEX idx_mb_entries_date ON mass_balance_entries(entry_date);
CREATE INDEX idx_mb_entries_batch ON mass_balance_entries(batch_id);
CREATE INDEX idx_mb_recon_account ON mass_balance_reconciliations(account_id);
```

---

## 4. Functional Requirements

### 4.1 Account Management
- **FR-001 (P0):** Create mass balance accounts per facility/commodity
- **FR-002 (P0):** Track compliant and non-compliant balances separately
- **FR-003 (P0):** Enforce balance cannot go negative
- **FR-004 (P0):** Support multi-period accounting

### 4.2 Transactions
- **FR-010 (P0):** Record credits (compliant receipts)
- **FR-011 (P0):** Record debits (compliant shipments)
- **FR-012 (P0):** Prevent over-claiming (debit > balance)
- **FR-013 (P0):** Track origin plots for credits
- **FR-014 (P1):** Support adjustments with audit trail

### 4.3 Reconciliation
- **FR-020 (P0):** Reconcile book vs physical balance
- **FR-021 (P0):** Calculate and flag variances
- **FR-022 (P0):** Generate reconciliation reports
- **FR-023 (P1):** Investigate variance root causes

### 4.4 Reporting
- **FR-030 (P0):** Generate mass balance statements
- **FR-031 (P0):** Calculate compliance percentage
- **FR-032 (P0):** Export for DDS reporting

---

## 5. Mass Balance Engine

```python
class MassBalanceEngine:
    """
    Mass balance accounting engine for EUDR compliance.
    """

    def credit(
        self,
        account_id: UUID,
        quantity: Decimal,
        batch_id: UUID,
        origin_plots: List[UUID],
        compliance_type: str = "COMPLIANT"
    ) -> MassBalanceEntry:
        """
        Add credit (receipt) to mass balance account.
        """
        account = self.get_account(account_id)

        new_balance = account.current_balance + quantity
        if compliance_type == "COMPLIANT":
            account.compliant_balance += quantity
        else:
            account.non_compliant_balance += quantity

        entry = MassBalanceEntry(
            account_id=account_id,
            entry_type="CREDIT",
            quantity=quantity,
            compliance_type=compliance_type,
            reference_type="RECEIPT",
            batch_id=batch_id,
            origin_plots=origin_plots,
            running_balance=new_balance
        )

        self.save_entry(entry)
        self.update_account(account, new_balance)

        return entry

    def debit(
        self,
        account_id: UUID,
        quantity: Decimal,
        batch_id: UUID,
        compliance_type: str = "COMPLIANT"
    ) -> MassBalanceEntry:
        """
        Deduct from mass balance account.
        Validates sufficient balance exists.
        """
        account = self.get_account(account_id)

        # Validate sufficient balance
        if compliance_type == "COMPLIANT":
            if quantity > account.compliant_balance:
                raise InsufficientBalanceError(
                    f"Cannot debit {quantity}. Compliant balance is {account.compliant_balance}"
                )
            account.compliant_balance -= quantity
        else:
            if quantity > account.non_compliant_balance:
                raise InsufficientBalanceError(
                    f"Cannot debit {quantity}. Non-compliant balance is {account.non_compliant_balance}"
                )
            account.non_compliant_balance -= quantity

        new_balance = account.current_balance - quantity

        entry = MassBalanceEntry(
            account_id=account_id,
            entry_type="DEBIT",
            quantity=quantity,
            compliance_type=compliance_type,
            reference_type="SHIPMENT",
            batch_id=batch_id,
            running_balance=new_balance
        )

        self.save_entry(entry)
        self.update_account(account, new_balance)

        return entry

    def reconcile(
        self,
        account_id: UUID,
        period_start: date,
        period_end: date,
        actual_balance: Decimal
    ) -> MassBalanceReconciliation:
        """
        Reconcile book balance with physical inventory.
        """
        account = self.get_account(account_id)

        # Get opening balance
        opening_entry = self.get_entry_at_date(account_id, period_start)
        opening_balance = opening_entry.running_balance if opening_entry else Decimal(0)

        # Calculate period movements
        entries = self.get_entries_for_period(account_id, period_start, period_end)
        total_credits = sum(e.quantity for e in entries if e.entry_type == "CREDIT")
        total_debits = sum(e.quantity for e in entries if e.entry_type == "DEBIT")

        # Calculate expected closing
        calculated_closing = opening_balance + total_credits - total_debits

        # Calculate variance
        variance = actual_balance - calculated_closing
        variance_pct = (variance / calculated_closing * 100) if calculated_closing else 0

        # Determine status
        if abs(variance_pct) <= 1:
            status = "BALANCED"
        elif abs(variance_pct) <= 5:
            status = "VARIANCE"
        else:
            status = "INVESTIGATION_REQUIRED"

        reconciliation = MassBalanceReconciliation(
            account_id=account_id,
            period_start=period_start,
            period_end=period_end,
            opening_balance=opening_balance,
            total_credits=total_credits,
            total_debits=total_debits,
            calculated_closing=calculated_closing,
            actual_balance=actual_balance,
            variance=variance,
            variance_percentage=variance_pct,
            status=status
        )

        self.save_reconciliation(reconciliation)
        return reconciliation

    def get_compliance_percentage(self, account_id: UUID) -> Decimal:
        """
        Calculate percentage of compliant inventory.
        """
        account = self.get_account(account_id)
        total = account.compliant_balance + account.non_compliant_balance
        if total == 0:
            return Decimal(100)
        return (account.compliant_balance / total) * 100
```

---

## 6. API Specification

```yaml
paths:
  /api/v1/mass-balance/accounts:
    post:
      summary: Create mass balance account
    get:
      summary: List accounts

  /api/v1/mass-balance/accounts/{account_id}:
    get:
      summary: Get account balance

  /api/v1/mass-balance/accounts/{account_id}/credit:
    post:
      summary: Record credit (receipt)

  /api/v1/mass-balance/accounts/{account_id}/debit:
    post:
      summary: Record debit (shipment)

  /api/v1/mass-balance/accounts/{account_id}/reconcile:
    post:
      summary: Reconcile balance

  /api/v1/mass-balance/accounts/{account_id}/statement:
    get:
      summary: Get balance statement

  /api/v1/mass-balance/accounts/{account_id}/entries:
    get:
      summary: Get ledger entries
```

---

## 7. Success Metrics

- **Balance Accuracy:** <2% variance in reconciliations
- **Compliance Tracking:** 100% entries classified
- **No Over-Claims:** 0% debits exceeding balance
- **Reconciliation Frequency:** Monthly minimum

---

*Document Version: 1.0*
*Created: 2026-01-30*
*Status: APPROVED FOR IMPLEMENTATION*
