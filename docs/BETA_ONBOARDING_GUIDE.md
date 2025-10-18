# GreenLang Framework v0.5 - Beta Onboarding Guide

**Welcome, Beta Tester!** 🎉

Thank you for joining the GreenLang Framework beta program. This guide will help you get started quickly and provide you with everything you need to be successful.

---

## 📋 TABLE OF CONTENTS

1. [Welcome & Overview](#welcome--overview)
2. [Beta Program Goals](#beta-program-goals)
3. [Getting Started in 15 Minutes](#getting-started-in-15-minutes)
4. [Your First Agent](#your-first-agent)
5. [Core Concepts](#core-concepts)
6. [Common Use Cases](#common-use-cases)
7. [Providing Feedback](#providing-feedback)
8. [Troubleshooting](#troubleshooting)
9. [Beta User Resources](#beta-user-resources)
10. [Next Steps](#next-steps)

---

## 🎯 WELCOME & OVERVIEW

### What is GreenLang Framework?

GreenLang is a **Python framework for building climate-aware data processing agents**. It provides:

✅ **50-70% code reduction** - Framework handles infrastructure, you write business logic
✅ **Built-in provenance tracking** - Automatic audit trails for regulatory compliance
✅ **Zero-hallucination calculations** - Deterministic, cached, traceable computations
✅ **Multi-format I/O** - CSV, JSON, Excel, Parquet out of the box
✅ **Validation framework** - Schema + business rules validation
✅ **Production-ready** - Logging, metrics, error handling included

### Who Should Use This Framework?

- **Data Engineers** processing climate/energy data
- **Climate Analysts** building carbon accounting tools
- **Compliance Officers** needing audit trails (EU CBAM, etc.)
- **Python Developers** wanting to build agents faster

### Beta Version: v0.5.0

**Status:** Feature Complete, Production Testing
**Release Date:** 2025-10-17
**Stability:** Beta (expect minor API changes)

---

## 🎯 BETA PROGRAM GOALS

### What We're Testing

1. **API Usability** - Is the framework easy to learn and use?
2. **Documentation Quality** - Are guides clear and comprehensive?
3. **Performance** - Does it meet <5% overhead target?
4. **Real-World Usage** - Does it work for your specific use cases?
5. **Edge Cases** - What breaks? What's missing?

### What We Need From You

📝 **Feedback on:**
- Installation experience
- Learning curve
- Documentation gaps
- Feature requests
- Bugs and issues
- Performance in your environment

🎯 **Your Commitment:**
- Build at least 1 agent using the framework
- Provide feedback within 2 weeks
- Report any bugs you encounter
- Answer a brief survey at the end

🎁 **What You Get:**
- Early access to framework features
- Direct line to core development team
- Influence on roadmap priorities
- Recognition in v1.0 release notes
- Potential speaking opportunities at launch event

---

## ⚡ GETTING STARTED IN 15 MINUTES

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment tool (recommended)

### Step 1: Install (2 minutes)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install GreenLang (beta version)
pip install greenlang-cli==0.3.0

# Verify installation
gl --version
# Expected output: greenlang 0.3.0
```

### Step 2: Run Quick Start Example (5 minutes)

```bash
# Clone examples repository (or download)
git clone https://github.com/greenlang/greenlang-examples.git
cd greenlang-examples

# Run the simplest example
python examples/01_simple_agent.py
```

**Expected Output:**
```
Agent: HelloWorldAgent
Status: Success
Message: Hello, Alice! Welcome to GreenLang.
Execution Time: 2.34ms
```

### Step 3: Create Your First Agent (8 minutes)

Create `my_first_agent.py`:

```python
from greenlang.agents import BaseAgent, AgentResult
from typing import Dict, Any

class MyFirstAgent(BaseAgent):
    """My first GreenLang agent!"""

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Process the input and return a result."""
        name = input_data.get("name", "World")
        message = f"Hello from GreenLang, {name}!"

        return AgentResult(
            success=True,
            data={"greeting": message}
        )

# Run it
if __name__ == "__main__":
    agent = MyFirstAgent()
    result = agent.run({"name": "Beta Tester"})

    print(f"Success: {result.success}")
    print(f"Greeting: {result.data['greeting']}")
    print(f"Time: {result.metrics.execution_time_ms:.2f}ms")
```

**Run it:**
```bash
python my_first_agent.py
```

**🎉 Congratulations!** You've created your first GreenLang agent in under 15 minutes.

---

## 🚀 YOUR FIRST AGENT

### The Minimum Viable Agent

```python
from greenlang.agents import BaseAgent, AgentResult

class MinimalAgent(BaseAgent):
    def execute(self, input_data):
        # Your business logic here
        result = input_data.get("value", 0) * 2

        return AgentResult(
            success=True,
            data={"result": result}
        )
```

**That's it!** The framework handles:
- ✅ Input validation
- ✅ Error handling
- ✅ Metrics collection
- ✅ Logging
- ✅ Statistics tracking

### Data Processing Agent (Batch Processing)

```python
from greenlang.agents import BaseDataProcessor

class CSVProcessor(BaseDataProcessor):
    def process_record(self, record):
        """Process a single record - framework handles batching."""
        # Transform one record
        record["processed"] = True
        record["value"] = record["value"] * 2
        return record

# Usage
processor = CSVProcessor()
result = processor.run({
    "records": [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20}
    ]
})

# Framework automatically:
# - Batches records (default 1000 per batch)
# - Shows progress bar
# - Collects errors
# - Tracks statistics
```

### Calculator Agent (With Caching)

```python
from greenlang.agents import BaseCalculator
from greenlang.agents.decorators import cached, deterministic

class EmissionsCalculator(BaseCalculator):
    @cached(ttl_seconds=3600)  # Cache results for 1 hour
    @deterministic(seed=42)     # Reproducible results
    def calculate(self, inputs):
        """Calculate emissions - framework caches automatically."""
        energy = inputs["energy_kwh"]
        factor = inputs["emission_factor"]

        emissions = energy * factor

        # Track calculation steps (auto-included in result)
        self.add_calculation_step(
            step_name="emissions",
            formula="energy_kwh * emission_factor",
            inputs=inputs,
            result=emissions,
            units="kg_co2e"
        )

        return emissions

# First call - calculates
result1 = calc.run({"inputs": {"energy_kwh": 1000, "emission_factor": 0.5}})

# Second call - returns cached result instantly!
result2 = calc.run({"inputs": {"energy_kwh": 1000, "emission_factor": 0.5}})
assert result2.cached == True
```

### Reporter Agent (Multi-Format Output)

```python
from greenlang.agents import BaseReporter, ReportSection

class MonthlyReporter(BaseReporter):
    def aggregate_data(self, input_data):
        """Aggregate your data."""
        return {
            "total_energy": sum(input_data["energy_values"]),
            "average": sum(input_data["energy_values"]) / len(input_data["energy_values"])
        }

    def build_sections(self, aggregated_data):
        """Build report sections."""
        return [
            ReportSection(
                title="Summary",
                content=f"Total Energy: {aggregated_data['total_energy']} kWh",
                section_type="text"
            ),
            ReportSection(
                title="Details",
                content=aggregated_data,
                section_type="table"
            )
        ]

# Generate reports in multiple formats
reporter = MonthlyReporter(config=ReporterConfig(output_format="markdown"))
result = reporter.run({"energy_values": [100, 200, 300]})
print(result.data["report"])  # Markdown formatted

# Or HTML, JSON, Excel - just change config!
```

---

## 📚 CORE CONCEPTS

### 1. Agent Lifecycle

Every agent goes through this lifecycle:

```
run() → validate() → preprocess() → EXECUTE() → postprocess() → cleanup()
         ↓              ↓              ↓            ↓
     (optional)     (optional)    (required)   (optional)
```

**You only need to override `execute()`** - everything else is optional.

### 2. AgentResult

Every agent returns an `AgentResult`:

```python
AgentResult(
    success=True,              # Did it work?
    data={"key": "value"},     # Your results
    error=None,                # Error message if failed
    metadata={},               # Extra info
    metrics=AgentMetrics(...), # Performance data
    timestamp="2025-10-17...", # When it ran
    provenance_id="uuid..."    # Audit trail ID
)
```

### 3. Automatic Features

The framework automatically provides:

```
✅ Logging      → self.logger.info("message")
✅ Metrics      → result.metrics.execution_time_ms
✅ Statistics   → agent.get_stats()
✅ Caching      → resources cached automatically
✅ Validation   → validate_input() called automatically
✅ Error Handling → try/except wrapped
```

### 4. Provenance Tracking

Every execution creates an audit trail:

```python
from greenlang.agents.decorators import traced

class MyAgent(BaseAgent):
    @traced(save_path="provenance.json")
    def execute(self, input_data):
        # Execution automatically tracked
        return AgentResult(...)

# Creates provenance.json with:
# - Input hash
# - Output hash
# - Environment snapshot
# - Dependencies
# - Execution time
# - System info
```

### 5. Validation Framework

Validate data with schemas and business rules:

```python
from greenlang.validation import ValidationFramework, JSONSchemaValidator

# Define schema
schema = {
    "type": "object",
    "properties": {
        "energy": {"type": "number", "minimum": 0},
        "date": {"type": "string", "format": "date"}
    },
    "required": ["energy", "date"]
}

# Create validator
framework = ValidationFramework()
framework.add_validator("schema", JSONSchemaValidator(schema))

# Validate
result = framework.validate(your_data)
if not result.valid:
    print(result.get_summary())  # "Validation FAILED: 2 errors"
```

---

## 💡 COMMON USE CASES

### Use Case 1: Process CSV Files

```python
from greenlang.agents import BaseDataProcessor
from greenlang.io import DataReader, DataWriter

class CSVTransformer(BaseDataProcessor):
    def process_record(self, record):
        # Transform each row
        record["transformed"] = True
        return record

# Process CSV
processor = CSVTransformer()
data = DataReader.read_csv("input.csv")
result = processor.run({"records": data})
DataWriter.write_csv(result.data["records"], "output.csv")
```

### Use Case 2: Calculate Carbon Emissions

```python
from greenlang.agents import BaseCalculator

class CarbonCalculator(BaseCalculator):
    def calculate(self, inputs):
        energy_kwh = inputs["energy_kwh"]
        grid_factor = 0.385  # kg CO2/kWh

        emissions = energy_kwh * grid_factor
        return emissions

calc = CarbonCalculator()
result = calc.run({"inputs": {"energy_kwh": 1000}})
print(f"Emissions: {result.result_value} kg CO2")
```

### Use Case 3: Generate Reports

```python
from greenlang.agents import BaseReporter

class SustainabilityReporter(BaseReporter):
    def aggregate_data(self, input_data):
        return {
            "total_emissions": sum(input_data["emissions"]),
            "reduction_target": input_data["target"]
        }

    def build_sections(self, aggregated):
        return [
            ReportSection(
                title="Emissions Summary",
                content=f"Total: {aggregated['total_emissions']} tons CO2",
                section_type="text"
            )
        ]

reporter = SustainabilityReporter()
result = reporter.run({
    "emissions": [10, 20, 30],
    "target": 100
})
# Get Markdown, HTML, or Excel report!
```

### Use Case 4: Build Multi-Agent Pipeline

```python
# Step 1: Data Intake
intake = CSVProcessor()
step1_result = intake.run({"file": "data.csv"})

# Step 2: Calculate
calculator = CarbonCalculator()
step2_result = calculator.run({
    "inputs": step1_result.data
})

# Step 3: Report
reporter = SustainabilityReporter()
final_result = reporter.run({
    "emissions": step2_result.data["emissions"]
})

# Full audit trail available for entire pipeline!
```

---

## 📝 PROVIDING FEEDBACK

### How to Report Bugs

**Option 1: GitHub Issues** (Preferred)
```
Go to: https://github.com/greenlang/greenlang/issues
Click: "New Issue"
Select: "Bug Report"
Fill in: Description, steps to reproduce, environment
```

**Option 2: Email**
```
Send to: beta@greenlang.org
Subject: [BUG] Brief description
Include: Code snippet, error message, Python version
```

**What Makes a Good Bug Report:**
✅ Minimal code to reproduce the issue
✅ Expected vs actual behavior
✅ Error messages (full stacktrace)
✅ Python version and OS
✅ GreenLang version (`gl --version`)

### How to Request Features

**Feature Request Template:**
```markdown
**What feature would you like?**
Brief description

**Why is this useful?**
Your use case

**Example usage:**
```python
# How you'd like to use it
```

**Alternatives considered:**
What workarounds exist?
```

**Submit to:** https://github.com/greenlang/greenlang/discussions

### Weekly Check-ins

We'll send you a quick survey every week:
- What did you build this week?
- What worked well?
- What was frustrating?
- What features are you missing?

**Takes 2 minutes, helps us immensely!**

---

## 🔧 TROUBLESHOOTING

### Common Issues

#### Issue: Import Error

```python
ImportError: No module named 'greenlang'
```

**Solution:**
```bash
# Verify installation
pip list | grep greenlang

# Reinstall if needed
pip install --force-reinstall greenlang-cli==0.3.0

# Check Python version
python --version  # Must be 3.8+
```

#### Issue: Agent Not Executing

```python
result = agent.run({"data": "value"})
# Returns: AgentResult(success=False, error="Agent is disabled")
```

**Solution:**
```python
# Check agent is enabled
config = AgentConfig(name="MyAgent", enabled=True)  # ← Make sure this is True
agent = MyAgent(config=config)
```

#### Issue: Validation Always Fails

```python
result.valid == False
# But data looks correct
```

**Solution:**
```python
# Print detailed errors
for error in result.errors:
    print(f"Field: {error.field}")
    print(f"Message: {error.message}")
    print(f"Value: {error.value}")

# Check validator configuration
framework = ValidationFramework()
print(framework.get_validator_names())  # See which validators are registered
```

#### Issue: Slow Performance

```python
# Agent takes too long
result.metrics.execution_time_ms > 10000  # 10+ seconds
```

**Solutions:**
```python
# 1. Enable caching
@cached(ttl_seconds=3600)
def calculate(self, inputs):
    ...

# 2. Use parallel processing
config = DataProcessorConfig(parallel_workers=4)

# 3. Increase batch size
config = DataProcessorConfig(batch_size=5000)  # Default is 1000

# 4. Disable metrics if not needed
config = AgentConfig(enable_metrics=False)
```

### Getting Help

**Documentation:**
- Quick Start: `docs/QUICK_START.md`
- API Reference: `docs/API_REFERENCE.md`
- Examples: `examples/` directory
- Architecture: `docs/ARCHITECTURE_DIAGRAMS.md`

**Community:**
- GitHub Discussions: https://github.com/greenlang/greenlang/discussions
- Email Support: beta@greenlang.org
- Office Hours: Every Friday 2-3 PM UTC (Zoom link in welcome email)

**Emergency Contact:**
- Critical bugs: beta-urgent@greenlang.org
- Response time: <24 hours

---

## 🎓 BETA USER RESOURCES

### Exclusive Resources

1. **Beta Slack Channel**
   - Join: Use link in welcome email
   - Get help from core team and other beta users
   - Share your work
   - Vote on features

2. **Weekly Office Hours**
   - **When:** Every Friday 2-3 PM UTC
   - **What:** Q&A, demos, feedback sessions
   - **Zoom:** Link sent weekly

3. **Beta Documentation**
   - Access to unreleased features
   - Early API docs
   - Migration guides

4. **Direct Communication**
   - Email: beta@greenlang.org
   - Response: <48 hours
   - For urgent issues: beta-urgent@greenlang.org

### Beta Testing Timeline

```
Week 1-2: Onboarding
- Install framework
- Run examples
- Build first agent
- Report installation issues

Week 3-4: Development
- Build your use case
- Test advanced features
- Provide feedback
- Report bugs

Week 5-6: Refinement
- Test bug fixes
- Validate improvements
- Final survey
- Case study (optional)

Week 7: Launch Prep
- v1.0 release candidate
- Final validation
- Recognition in release notes
```

### Beta User Benefits

🎁 **During Beta:**
- Early access to features
- Direct line to dev team
- Influence roadmap
- Exclusive training

🏆 **After Beta:**
- Listed in v1.0 credits
- Speaking opportunity at launch event
- Free Pro license (when available)
- Beta tester badge on GitHub
- Priority support for 1 year

---

## 🚀 NEXT STEPS

### Your First Week

**Day 1:** Setup and Hello World
- ✅ Install GreenLang
- ✅ Run 01_simple_agent.py example
- ✅ Create your first agent
- ✅ Join beta Slack channel

**Day 2-3:** Explore Examples
- 📖 Read through all 10 examples
- 🔬 Modify examples to fit your data
- 💡 Identify your use case

**Day 4-5:** Build Your Agent
- 🏗️ Create your first real agent
- 🧪 Write tests for it
- 📝 Document what you learned

**Week 2+:** Provide Feedback
- 🐛 Report any bugs
- 💬 Share in Slack
- 📊 Complete weekly survey
- 🎯 Request features you need

### Suggested Learning Path

1. **Start Simple**
   ```
   examples/01_simple_agent.py → Understand basics
   examples/02_data_processor.py → Batch processing
   examples/03_calculator_cached.py → Caching
   ```

2. **Add Complexity**
   ```
   examples/05_provenance_tracking.py → Audit trails
   examples/06_validation_framework.py → Validation
   examples/08_parallel_processing.py → Performance
   ```

3. **Build Real Systems**
   ```
   examples/10_complete_pipeline.py → Multi-agent systems
   Your own agents → Production use
   ```

### Resources to Bookmark

📚 **Essential Docs:**
- Quick Start: `/docs/QUICK_START.md`
- API Reference: `/docs/API_REFERENCE.md`
- Architecture: `/docs/ARCHITECTURE_DIAGRAMS.md`

💻 **Code:**
- Examples: `/examples/`
- Tests: `/tests/` (see how we test)
- Benchmarks: `/benchmarks/`

🔗 **Links:**
- GitHub: https://github.com/greenlang/greenlang
- Docs Site: https://docs.greenlang.org
- Beta Portal: https://beta.greenlang.org

---

## 📞 CONTACT & SUPPORT

### Core Team

**Project Lead:** [Name]
- Email: lead@greenlang.org
- Available: Fridays during office hours

**Technical Lead:** [Name]
- Email: tech@greenlang.org
- Available: For technical questions

**Community Manager:** [Name]
- Email: community@greenlang.org
- Available: Daily in Slack

### Support Channels

| Type | Channel | Response Time |
|------|---------|---------------|
| 🐛 Bugs | GitHub Issues | <48h |
| 💡 Features | GitHub Discussions | <72h |
| ❓ Questions | Slack | <24h |
| 📧 General | beta@greenlang.org | <48h |
| 🚨 Urgent | beta-urgent@greenlang.org | <24h |

### Office Hours Schedule

**Every Friday 2-3 PM UTC**
- Week 1: Installation & Setup Help
- Week 2: Advanced Features Deep Dive
- Week 3: Performance Optimization
- Week 4: Production Deployment
- Week 5: Custom Use Cases
- Week 6: Final Q&A

**Zoom Link:** Sent in weekly email

---

## 🎉 THANK YOU!

**You're helping shape the future of climate data processing!**

Your feedback during this beta will directly influence:
- ✅ API design decisions
- ✅ Feature prioritization
- ✅ Documentation improvements
- ✅ Performance optimizations
- ✅ Production readiness

**We're excited to see what you build!**

---

## 📋 BETA CHECKLIST

Use this checklist to track your onboarding:

**Week 1:**
- [ ] Install GreenLang v0.5.0
- [ ] Run `gl --version` successfully
- [ ] Execute 01_simple_agent.py example
- [ ] Create and run your first custom agent
- [ ] Join beta Slack channel
- [ ] Attend first office hours (optional)
- [ ] Complete Week 1 survey

**Week 2:**
- [ ] Build agent for your actual use case
- [ ] Test at least 3 different agent types (DataProcessor, Calculator, Reporter)
- [ ] Try adding provenance tracking
- [ ] Report at least 1 bug or feature request
- [ ] Complete Week 2 survey

**Week 3-4:**
- [ ] Build complete pipeline with 2+ agents
- [ ] Add validation to your agents
- [ ] Write tests for your agents
- [ ] Share your work in Slack
- [ ] Complete mid-beta survey

**Week 5-6:**
- [ ] Test bug fixes from your reports
- [ ] Try v1.0 release candidate
- [ ] Write case study (optional)
- [ ] Complete final beta survey
- [ ] Provide testimonial (optional)

---

**Welcome aboard! Let's build the future of climate data processing together!** 🌍🚀

**Questions?** Email beta@greenlang.org or ask in Slack!

---

**Document Version:** 1.0.0 (Beta)
**Last Updated:** 2025-10-17
**For:** GreenLang v0.5.0 Beta Testers
