# GREENLANG MIGRATION STRATEGY

**From 5% Configuration to 70% Framework: The Complete Migration Path**

**Version:** 1.0
**Date:** 2025-10-15
**Status:** Migration Playbook
**Target:** Existing GreenLang Agents

---

## 📋 EXECUTIVE SUMMARY

### **The Challenge**

Existing GreenLang agents are built with 95% custom code and only 5% GreenLang configuration. To leverage the new framework (50-70% GreenLang), existing agents need a migration path.

### **The Solution**

A **phased, incremental migration strategy** that:
- ✅ Maintains backward compatibility
- ✅ Allows gradual adoption (opt-in)
- ✅ Provides automated migration tools
- ✅ Shows immediate ROI at each step
- ✅ Zero downtime for production agents

### **Migration Timeline**

| Agent Size | Manual Effort | Automated Tools | Total Time |
|------------|---------------|-----------------|------------|
| **Small** (500-1000 LOC) | 2-4 hours | 1 hour | **3-5 hours** |
| **Medium** (1000-2000 LOC) | 1-2 days | 2 hours | **1.5-2.5 days** |
| **Large** (2000+ LOC) | 3-5 days | 4 hours | **4-6 days** |

**ROI:** 60-80% LOC reduction, better quality, production-ready features

---

## 🎯 MIGRATION PHILOSOPHY

### **Guiding Principles**

1. **Incremental, Not All-at-Once**
   - Migrate one component at a time
   - Each step provides immediate value
   - Can pause/resume migration

2. **Opt-In, Not Forced**
   - Agents can choose when to migrate
   - Old approach remains supported (deprecated)
   - No breaking changes to pack.yaml

3. **Automated Where Possible**
   - Code scanners identify migration opportunities
   - Code generators create framework integration
   - Testing tools verify correctness

4. **Show ROI Early**
   - First migration step saves 200+ lines
   - Quality improvements visible immediately
   - Performance gains measurable

---

## 🗺️ MIGRATION ROADMAP

### **The 5-Step Migration Path**

```
Current State (5%)
     ↓
Step 1: Base Classes (→ 30%)     [2-4 hours]
     ↓
Step 2: Provenance (→ 40%)       [1-2 hours]
     ↓
Step 3: Validation (→ 50%)       [2-4 hours]
     ↓
Step 4: I/O & Batch (→ 60%)      [2-3 hours]
     ↓
Step 5: Reporting (→ 70%)        [1-2 hours]
     ↓
Target State (70%)
```

**Total Migration Time:** 8-15 hours for typical agent

---

## 📖 STEP-BY-STEP MIGRATION GUIDE

### **Step 1: Migrate to Base Agent Classes**

**Goal:** Replace custom initialization, logging, metrics with framework base classes

**Before (Custom Code - 95%):**

```python
# agents/my_agent.py (BEFORE)
import logging
from datetime import datetime
from typing import Dict, Any, List

class MyAgent:
    """Custom agent with all boilerplate."""

    def __init__(self, config_path: str, data_path: str):
        # 50 lines of initialization boilerplate
        self.config = self._load_config(config_path)
        self.data = self._load_data(data_path)

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

        # Stats tracking
        self.stats = {
            'total_processed': 0,
            'errors': 0,
            'warnings': 0,
            'start_time': None,
            'end_time': None
        }

        self.logger.info("MyAgent initialized")

    def _load_config(self, path: str) -> Dict:
        # 15 lines of file loading
        import json
        with open(path) as f:
            return json.load(f)

    def _load_data(self, path: str) -> Dict:
        # 20 lines of data loading
        import json
        with open(path) as f:
            return json.load(f)

    def process(self, input_data: Any) -> Dict:
        # 30 lines of execution wrapper
        self.stats['start_time'] = datetime.now()

        try:
            result = self._do_process(input_data)
            self.stats['total_processed'] += 1
            return {'success': True, 'data': result}

        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            self.stats['errors'] += 1
            return {'success': False, 'error': str(e)}

        finally:
            self.stats['end_time'] = datetime.now()

    def _do_process(self, input_data: Any) -> Any:
        # Your actual business logic (100 lines)
        ...
```

**After (Framework - 30%):**

```python
# agents/my_agent.py (AFTER)
from greenlang.agents import BaseDataProcessor, AgentResult
from typing import Any

class MyAgent(BaseDataProcessor):
    """Agent using framework base class."""

    agent_id = "my-agent"
    version = "1.0.0"

    def __init__(self, **kwargs):
        # 5 lines - framework handles rest
        super().__init__(
            resources={
                'config': 'config/my_config.yaml',
                'data': 'data/my_data.json'
            },
            **kwargs
        )

    def execute(self, input_data: Any, **kwargs) -> AgentResult:
        # Your actual business logic (100 lines - unchanged!)
        result = self._do_process(input_data)

        return AgentResult(
            success=True,
            data=result,
            metadata=self.stats.get_summary()
        )

    def _do_process(self, input_data: Any) -> Any:
        # Same business logic as before (100 lines)
        ...
```

**Migration Steps:**

1. ✅ Add framework import: `from greenlang.agents import BaseDataProcessor`
2. ✅ Change inheritance: `class MyAgent(BaseDataProcessor)`
3. ✅ Replace `__init__` with super() call
4. ✅ Replace `process()` with `execute()` returning `AgentResult`
5. ✅ Delete boilerplate: logging setup, stats tracking, error handling
6. ✅ Run tests to verify behavior unchanged

**Automated Tool:**

```bash
# Migration assistant
gl migrate --step base-classes --agent agents/my_agent.py --dry-run
gl migrate --step base-classes --agent agents/my_agent.py --apply
```

**LOC Impact:**
- **Before:** 200 lines (100 boilerplate + 100 business logic)
- **After:** 120 lines (20 framework integration + 100 business logic)
- **Savings:** 80 lines (40% reduction)

**Time:** 2-4 hours

**Validation:**
- [ ] All tests pass
- [ ] Logging still works
- [ ] Metrics collected
- [ ] Provenance captured (if enabled)

---

### **Step 2: Add Automatic Provenance**

**Goal:** Replace custom provenance tracking with framework provenance

**Before:**

```python
# Custom provenance (60 lines)
import hashlib

def hash_file(path: str) -> str:
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def create_provenance(input_file: str) -> Dict:
    return {
        'input_file': input_file,
        'file_hash': hash_file(input_file),
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }

# In agent:
def process(self, input_file: str) -> Dict:
    provenance = create_provenance(input_file)
    result = self._do_process(input_file)
    result['provenance'] = provenance
    return result
```

**After:**

```python
# Automatic framework provenance (0 lines custom code!)
from greenlang.agents import BaseDataProcessor

class MyAgent(BaseDataProcessor):
    def __init__(self, **kwargs):
        super().__init__(enable_provenance=True, **kwargs)
        # That's it! Provenance automatic

    def execute(self, input_file: str, **kwargs) -> AgentResult:
        # Framework automatically captures:
        # - File hash (SHA256)
        # - Environment info
        # - Execution metadata
        # - Complete audit trail
        result = self._do_process(input_file)

        return AgentResult(
            success=True,
            data=result
            # provenance added automatically by framework!
        )
```

**Migration Steps:**

1. ✅ Enable provenance in `__init__`: `enable_provenance=True`
2. ✅ Delete custom hash_file() function
3. ✅ Delete custom create_provenance() function
4. ✅ Remove manual provenance assignment
5. ✅ Verify provenance in output

**Automated Tool:**

```bash
gl migrate --step provenance --agent agents/my_agent.py --apply
```

**LOC Impact:**
- **Before:** 120 lines (60 provenance + 60 other)
- **After:** 60 lines (0 provenance + 60 other)
- **Savings:** 60 lines (50% additional reduction)

**Time:** 1-2 hours

**Validation:**
- [ ] Provenance in output
- [ ] SHA256 hash correct
- [ ] Environment captured

---

### **Step 3: Replace Validation with Framework**

**Goal:** Replace custom validation logic with declarative framework validation

**Before:**

```python
# Custom validation (150 lines)
def validate_item(item: Dict) -> Tuple[bool, List[str]]:
    errors = []

    # Required fields
    if 'id' not in item:
        errors.append("Missing required field: id")
    if 'amount' not in item:
        errors.append("Missing required field: amount")

    # Type validation
    if not isinstance(item.get('amount'), (int, float)):
        errors.append("Amount must be numeric")

    # Range validation
    if item.get('amount', 0) < 0:
        errors.append("Amount must be positive")

    # Format validation
    if 'email' in item:
        if '@' not in item['email']:
            errors.append("Invalid email format")

    # Business rules
    if item.get('type') == 'premium' and item.get('amount', 0) < 100:
        errors.append("Premium type requires amount >= 100")

    return len(errors) == 0, errors
```

**After:**

```python
# Declarative validation (5 lines + YAML file)
from greenlang.validation import ValidationFramework

class MyAgent(BaseDataProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Declarative validation
        self.validator = ValidationFramework(
            schema='schemas/item.schema.json',
            rules='rules/business_rules.yaml'
        )

    def execute(self, items: List[Dict], **kwargs) -> AgentResult:
        # Framework validates with one line
        result = self.process_batch(
            items=items,
            validate_fn=self.validator.validate  # Framework validates!
        )

        return AgentResult(success=True, data=result)
```

**Create validation files:**

```yaml
# rules/business_rules.yaml
rules:
  - id: premium_amount_check
    condition: "type == 'premium' AND amount < 100"
    error_code: E001
    message: "Premium type requires amount >= 100"
    severity: error

  - id: email_format
    condition: "email AND '@' not in email"
    error_code: E002
    message: "Invalid email format"
    severity: error
```

```json
// schemas/item.schema.json
{
  "type": "object",
  "required": ["id", "amount"],
  "properties": {
    "id": {"type": "string"},
    "amount": {"type": "number", "minimum": 0},
    "email": {"type": "string", "format": "email"},
    "type": {"type": "string", "enum": ["standard", "premium"]}
  }
}
```

**Migration Steps:**

1. ✅ Create JSON schema file
2. ✅ Create business rules YAML file
3. ✅ Add ValidationFramework to __init__
4. ✅ Replace custom validation with framework
5. ✅ Delete custom validation function
6. ✅ Run tests

**Automated Tool:**

```bash
gl migrate --step validation --agent agents/my_agent.py --generate-schema
gl migrate --step validation --agent agents/my_agent.py --apply
```

**LOC Impact:**
- **Before:** 210 lines (150 validation + 60 other)
- **After:** 65 lines (5 validation + 60 other + schemas)
- **Savings:** 145 lines (70% additional reduction)

**Time:** 2-4 hours (including schema creation)

**Validation:**
- [ ] All validation tests pass
- [ ] Error messages correct
- [ ] Schema covers all cases

---

### **Step 4: Use Framework I/O and Batch Processing**

**Goal:** Replace custom file reading/writing and batch processing

**Before:**

```python
# Custom I/O (80 lines)
import pandas as pd
from pathlib import Path

def read_file(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == '.csv':
        return pd.read_csv(path)
    elif p.suffix == '.json':
        return pd.read_json(path)
    elif p.suffix == '.xlsx':
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")

def write_file(data: pd.DataFrame, path: str):
    p = Path(path)
    if p.suffix == '.csv':
        data.to_csv(path, index=False)
    elif p.suffix == '.json':
        data.to_json(path, orient='records')
    elif p.suffix == '.xlsx':
        data.to_excel(path, index=False)

# Custom batch processing (50 lines)
def process_batch(items: List[Dict]) -> List[Dict]:
    results = []
    for item in items:
        try:
            result = process_item(item)
            results.append(result)
        except Exception as e:
            logger.error(f"Item failed: {e}")
            results.append({'error': str(e)})
    return results
```

**After:**

```python
# Framework I/O and batch processing (0 lines!)
from greenlang.agents import BaseDataProcessor

class MyAgent(BaseDataProcessor):
    def execute(self, input_path: str, output_path: str, **kwargs) -> AgentResult:
        # Framework reads any format automatically
        df = self.read_input(input_path)

        # Framework handles batch processing
        result = self.process_batch(
            items=df.to_dict('records'),
            transform_fn=self.process_item  # Your business logic
        )

        # Framework writes any format automatically
        self.write_output(result['items'], output_path)

        return AgentResult(success=True, data=result)

    def process_item(self, item: Dict) -> Dict:
        # Your business logic unchanged (30 lines)
        ...
```

**Migration Steps:**

1. ✅ Replace custom read_file with self.read_input
2. ✅ Replace custom write_file with self.write_output
3. ✅ Replace custom batch loop with self.process_batch
4. ✅ Delete custom I/O functions
5. ✅ Run tests

**Automated Tool:**

```bash
gl migrate --step io-batch --agent agents/my_agent.py --apply
```

**LOC Impact:**
- **Before:** 195 lines (130 I/O/batch + 65 other)
- **After:** 65 lines (0 I/O/batch + 65 other)
- **Savings:** 130 lines (67% additional reduction)

**Time:** 2-3 hours

**Validation:**
- [ ] Reads all formats correctly
- [ ] Writes all formats correctly
- [ ] Batch processing works
- [ ] Progress tracking visible

---

### **Step 5: Use Framework Reporting (If Applicable)**

**Goal:** Replace custom aggregation and reporting with framework utilities

**Before:**

```python
# Custom aggregation (100 lines)
def aggregate_by_category(items: List[Dict]) -> Dict:
    from collections import defaultdict

    by_category = defaultdict(lambda: {'count': 0, 'total': 0})

    for item in items:
        cat = item.get('category', 'unknown')
        by_category[cat]['count'] += 1
        by_category[cat]['total'] += item.get('amount', 0)

    return dict(by_category)

def generate_report(data: Dict) -> str:
    # 50 lines of Markdown generation
    report = "# Report\n\n"
    report += "## Summary\n\n"
    # ... more report generation
    return report
```

**After:**

```python
# Framework aggregation and reporting (5 lines)
from greenlang.agents import BaseReporter
from greenlang.reporting import MultiDimensionalAggregator

class MyAgent(BaseReporter):
    def execute(self, items: List[Dict], **kwargs) -> AgentResult:
        # Framework aggregates
        summary = self.aggregate(
            data=items,
            dimensions=['category'],
            metrics={'count': 'count()', 'total': 'sum(amount)'}
        )

        # Framework formats
        report_md = self.format_report(summary, format='markdown')

        return AgentResult(success=True, data={'report': report_md})
```

**Migration Steps:**

1. ✅ Change base class to BaseReporter (if reporting agent)
2. ✅ Replace custom aggregation with self.aggregate
3. ✅ Replace custom report generation with self.format_report
4. ✅ Delete custom aggregation functions
5. ✅ Run tests

**Automated Tool:**

```bash
gl migrate --step reporting --agent agents/my_agent.py --apply
```

**LOC Impact:**
- **Before:** 165 lines (100 reporting + 65 other)
- **After:** 70 lines (5 reporting + 65 other)
- **Savings:** 95 lines (58% additional reduction)

**Time:** 1-2 hours

**Validation:**
- [ ] Aggregations correct
- [ ] Report format correct
- [ ] Multi-dimensional grouping works

---

## 📊 CUMULATIVE MIGRATION IMPACT

### **Complete Migration Journey**

| Step | Before | After | Savings | % Framework |
|------|--------|-------|---------|-------------|
| **Start** | 500 | 500 | 0 | 5% |
| **+Step 1 (Base)** | 500 | 420 | 80 | 30% |
| **+Step 2 (Provenance)** | 420 | 360 | 140 | 40% |
| **+Step 3 (Validation)** | 360 | 215 | 285 | 50% |
| **+Step 4 (I/O/Batch)** | 215 | 85 | 415 | 60% |
| **+Step 5 (Reporting)** | 165 | 70 | 430 | 70% |

**Total Savings:** 430 lines (86% reduction)
**Total Time:** 8-15 hours
**Framework Contribution:** 70%

---

## 🛠️ MIGRATION TOOLS

### **Automated Migration Assistant**

```bash
# Install migration tools
pip install greenlang-migrate

# Scan agent for migration opportunities
gl migrate scan agents/my_agent.py

# Output:
# ✓ Can migrate to base classes (saves 80 lines)
# ✓ Can add automatic provenance (saves 60 lines)
# ✓ Can use validation framework (saves 145 lines)
# ✓ Can use I/O utilities (saves 130 lines)
# ✗ Not a reporting agent (skip)
#
# Total potential savings: 415 lines (83%)

# Migrate step-by-step
gl migrate --step base-classes --agent agents/my_agent.py --dry-run
gl migrate --step base-classes --agent agents/my_agent.py --apply

gl migrate --step provenance --agent agents/my_agent.py --apply
gl migrate --step validation --agent agents/my_agent.py --generate-schema --apply
gl migrate --step io-batch --agent agents/my_agent.py --apply

# Or migrate all at once
gl migrate --all --agent agents/my_agent.py --apply

# Verify migration
gl migrate verify agents/my_agent.py
pytest tests/test_my_agent.py -v
```

---

## ✅ POST-MIGRATION CHECKLIST

### **Verification Steps**

After each migration step:

1. **Tests**
   - [ ] All unit tests pass
   - [ ] All integration tests pass
   - [ ] Performance tests pass

2. **Functionality**
   - [ ] Input/output behavior unchanged
   - [ ] Error handling works
   - [ ] Logging captured correctly

3. **Quality**
   - [ ] Provenance data complete
   - [ ] Metrics collected
   - [ ] Code coverage maintained

4. **Documentation**
   - [ ] Update agent README
   - [ ] Update pack.yaml if needed
   - [ ] Add migration notes

---

## 🎯 MIGRATION BEST PRACTICES

### **Do's**

✅ **Migrate incrementally** - One step at a time
✅ **Test after each step** - Verify before next step
✅ **Use automation tools** - Don't migrate manually if tool exists
✅ **Keep business logic unchanged** - Only replace boilerplate
✅ **Document migration** - Add notes for future reference

### **Don'ts**

❌ **Don't migrate everything at once** - Too risky
❌ **Don't skip testing** - Each step must be verified
❌ **Don't change business logic** - Migration ≠ refactoring
❌ **Don't break backward compat** - Old agents must still work
❌ **Don't rush** - Quality over speed

---

## 📞 SUPPORT & RESOURCES

### **Migration Help**

- **Documentation:** https://docs.greenlang.io/migration
- **Video Tutorials:** https://youtube.com/greenlang/migration-series
- **Community Forum:** https://community.greenlang.io
- **Migration Office Hours:** Tuesdays 2-4pm PST

### **Example Migrations**

See complete migration examples:
- **CBAM Importer:** `examples/migrations/cbam/`
- **Data Processor:** `examples/migrations/data-processor/`
- **Calculator:** `examples/migrations/calculator/`
- **Reporter:** `examples/migrations/reporter/`

---

## 🎉 SUCCESS STORIES

### **CBAM Importer Migration**

**Before:**
- 3,172 lines custom code
- 2-3 weeks development time
- Custom provenance (605 lines)
- Custom validation (750 lines)

**After:**
- 1,200 lines custom code (62% reduction)
- 1-2 weeks development time (50% faster)
- Automatic provenance (0 lines)
- Declarative validation (5 lines + YAML)

**Migration Time:** 3 days
**ROI:** 1,972 lines saved

---

## 📝 MIGRATION PLAN TEMPLATE

Use this template for your agent:

```markdown
# Migration Plan: [Agent Name]

**Agent:** agents/[name].py
**Current LOC:** [number]
**Target LOC:** [estimated]
**Estimated Time:** [hours/days]

## Pre-Migration
- [ ] Backup original code
- [ ] Ensure all tests pass
- [ ] Review migration guide
- [ ] Install migration tools

## Step 1: Base Classes
- [ ] Scan for opportunities: `gl migrate scan`
- [ ] Apply migration: `gl migrate --step base-classes --apply`
- [ ] Run tests: `pytest tests/test_[name].py`
- [ ] Commit: `git commit -m "Migrate to BaseDataProcessor"`

## Step 2: Provenance
- [ ] Apply: `gl migrate --step provenance --apply`
- [ ] Verify provenance in output
- [ ] Run tests
- [ ] Commit

## Step 3: Validation
- [ ] Generate schemas: `gl migrate --step validation --generate-schema`
- [ ] Review generated schemas
- [ ] Apply migration
- [ ] Run tests
- [ ] Commit

## Step 4: I/O & Batch
- [ ] Apply: `gl migrate --step io-batch --apply`
- [ ] Test all input formats
- [ ] Test all output formats
- [ ] Run tests
- [ ] Commit

## Step 5: Reporting (if applicable)
- [ ] Apply: `gl migrate --step reporting --apply`
- [ ] Verify aggregations
- [ ] Verify report format
- [ ] Run tests
- [ ] Commit

## Post-Migration
- [ ] Full test suite passes
- [ ] Documentation updated
- [ ] Performance benchmarked
- [ ] Deploy to staging
- [ ] Monitor for issues
- [ ] Deploy to production

## Results
- **Original LOC:** [number]
- **Final LOC:** [number]
- **Savings:** [number] lines ([percent]%)
- **Migration Time:** [actual time]
- **Issues Found:** [list any issues]
```

---

## 🚀 CONCLUSION

Migration to the GreenLang framework:

✅ **Reduces code by 60-80%** (less to maintain)
✅ **Improves quality** (production-ready features)
✅ **Takes 8-15 hours** (reasonable investment)
✅ **Provides immediate ROI** (value at each step)
✅ **Is fully supported** (tools + documentation)

**Recommendation:** Begin migration with Tier 1 agents (highest ROI).

---

**Status:** 🚀 Ready for Migration
**Next:** Choose first agent to migrate

---

*"The best time to plant a tree was 20 years ago. The second best time is now."* - Migration Wisdom
