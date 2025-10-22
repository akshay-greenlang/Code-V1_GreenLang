# GreenLang Demo Video Script
## "Using Tools, Not Guessing" - 15-Minute Walkthrough

**Target Duration:** 15 minutes
**Recording Date:** October 22, 2025
**Version:** v0.3.0
**Objective:** Demonstrate tool-first architecture, provenance tracking, and deterministic AI agents

---

## Setup Instructions

### Before Recording

1. **Environment Setup:**
   ```bash
   cd C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang
   python -m venv .venv-demo
   .venv-demo\Scripts\activate
   pip install -e ".[analytics]"
   gl --version  # Should show v0.3.0
   ```

2. **Prepare Terminal:**
   - Use Windows Terminal with PowerShell
   - Font size: 14pt (readable on video)
   - Color scheme: Campbell (high contrast)
   - Fullscreen mode (F11)

3. **Open Files in VS Code:**
   - `greenlang/agents/boiler_replacement_agent_ai.py`
   - `greenlang/agents/industrial_process_heat_agent_ai.py`
   - `greenlang/agents/industrial_heat_pump_agent_ai.py`
   - `docs/DOC-601_USING_TOOLS_NOT_GUESSING.md`

4. **Test Run (Dry Run):**
   ```bash
   python tests/agents/test_boiler_replacement_agent_ai.py -v
   ```

---

## Video Structure

### Introduction (0:00 - 1:30)

**[SCREEN: VS Code with README.md visible]**

**SCRIPT:**
> "Welcome to GreenLang v0.3.0. I'm going to show you how we solve the AI hallucination problem in climate intelligence applications using a tool-first architecture."
>
> "The core principle is simple: **Using Tools, Not Guessing**. No number reaches the user unless it came from a validated tool output, with explicit units and complete provenance."
>
> "Today, I'll demonstrate three production-ready AI agents that calculate industrial decarbonization recommendations. Each agent uses deterministic execution, tool-grounded calculations, and full provenance tracking."

**[TRANSITION: Switch to Terminal]**

---

### Part 1: The Problem - AI Hallucination (1:30 - 3:00)

**[SCREEN: Terminal with Python REPL]**

**SCRIPT:**
> "Let me first show you what we're avoiding. Here's a traditional LLM without tool constraints:"

**[TYPE IN TERMINAL]**
```python
from unittest.mock import Mock

# Traditional LLM - can guess numbers
traditional_llm = Mock()
traditional_llm.chat = Mock(return_value={
    "message": "Your building uses approximately 10,000 kWh per year, which costs about $1,500."
})

result = traditional_llm.chat("Calculate my energy costs")
print(result["message"])
```

**[RUN CODE]**

**SCRIPT:**
> "Notice the problem? The LLM just **guessed** both numbers. No calculation, no provenance, no traceability. Where did 10,000 kWh come from? Where did $1,500 come from? We have no idea."
>
> "In climate applications, this is unacceptable. We need verified, reproducible calculations. That's where tool-first architecture comes in."

**[EXIT PYTHON REPL]**

---

### Part 2: The Solution - Tool Runtime (3:00 - 5:00)

**[SCREEN: VS Code - Show DOC-601_USING_TOOLS_NOT_GUESSING.md]**

**SCRIPT:**
> "GreenLang solves this with three enforcement layers:"
>
> "**Layer 1: Quantity Schema** - All numeric outputs must have value + unit."
>
> "**Layer 2: Claim Verification** - LLM must reference tool outputs via {{claim:i}} macros."
>
> "**Layer 3: Naked Number Scanner** - Any digit not backed by a tool? Runtime error."

**[SCROLL TO ARCHITECTURE DIAGRAM]**

**SCRIPT:**
> "Here's how it works: The LLM calls a tool, we validate the result, the LLM provides a final answer with claims, and we verify each claim matches the tool output. If there's any digit without provenance, the runtime blocks it."

**[SWITCH TO TERMINAL]**

**SCRIPT:**
> "Let me show you a simple example:"

**[TYPE IN TERMINAL]**
```python
python examples/runtime_no_naked_numbers_demo.py
```

**[RUN AND SHOW OUTPUT]**

**SCRIPT:**
> "Notice: Every number has a unit. Every claim has a source tool call ID. Complete provenance chain from query to answer."

---

### Part 3: Agent Demo #1 - Boiler Replacement (5:00 - 7:30)

**[SCREEN: Terminal]**

**SCRIPT:**
> "Now let's see this in production. Our first agent recommends boiler replacements for industrial facilities."

**[TYPE IN TERMINAL]**
```bash
cd tests/agents
python test_boiler_replacement_agent_ai.py::TestBoilerGolden::test_example_input -v
```

**[RUN TEST - WAIT FOR OUTPUT]**

**SCRIPT:**
> "While this runs, let me explain what's happening:"
>
> "1. The agent receives: fuel type, annual consumption, current efficiency"
> "2. It calls tools to calculate: baseline emissions, recommended technology, cost savings"
> "3. Every calculation is a validated tool with schema enforcement"
> "4. The final recommendation cites every number back to its source tool"

**[SHOW TEST OUTPUT]**

**SCRIPT:**
> "Test passed! Let's look at the provenance:"

**[TYPE IN TERMINAL]**
```bash
cat artifacts/W1/provenance_samples/boiler_replacement_sample.json
```

**[SHOW FILE - HIGHLIGHT KEY SECTIONS]**

**SCRIPT:**
> "See the provenance array? Every quantity has:"
> "- source_call_id: which tool generated it"
> "- path: JSONPath to the value in tool output"
> "- quantity: the actual value + unit"
>
> "This means we can trace any number in the final answer back to the exact tool and calculation that produced it."

---

### Part 4: Agent Demo #2 - Industrial Process Heat (7:30 - 9:30)

**[SCREEN: VS Code - Show industrial_process_heat_agent_ai.py]**

**SCRIPT:**
> "Our second agent calculates heat requirements for industrial processes. This is more complex because we need to consider multiple factors."

**[HIGHLIGHT CODE SECTIONS]**

**SCRIPT:**
> "Notice the tools: calculate_heat_demand, estimate_thermal_efficiency, calculate_fuel_consumption. Each has strict schemas."

**[SWITCH TO TERMINAL]**

**[TYPE IN TERMINAL]**
```bash
python test_industrial_process_heat_agent_ai.py::TestProcessHeatGolden::test_example_input -v
```

**[RUN TEST]**

**SCRIPT:**
> "The agent just:"
> "1. Called calculate_heat_demand with process parameters"
> "2. Called estimate_thermal_efficiency for the current system"
> "3. Called calculate_fuel_consumption to get baseline"
> "4. Provided final answer with all quantities claimed from tools"

**[SHOW OUTPUT]**

**SCRIPT:**
> "Notice the message: 'Your process requires {{claim:0}} of heat, consuming {{claim:1}} of natural gas.' Every claim is resolved to the actual tool output."

**[TYPE IN TERMINAL]**
```bash
python -c "import json; print(json.dumps(json.load(open('artifacts/W1/provenance_samples/industrial_process_heat_sample.json')), indent=2))"
```

**[SHOW PROVENANCE]**

**SCRIPT:**
> "Complete provenance chain. Auditable, reproducible, traceable."

---

### Part 5: Agent Demo #3 - Industrial Heat Pump (9:30 - 11:00)

**[SCREEN: Terminal]**

**SCRIPT:**
> "Our third agent evaluates industrial heat pump deployments. This is the most sophisticated because it considers waste heat recovery."

**[TYPE IN TERMINAL]**
```bash
python test_industrial_heat_pump_agent_ai.py::TestHeatPumpGolden::test_example_input -v
```

**[RUN TEST]**

**SCRIPT:**
> "The agent analyzes:"
> "- Available waste heat sources"
> "- Required delivery temperature"
> "- Heat pump COP (coefficient of performance)"
> "- Economic feasibility"
>
> "All calculated by tools, nothing guessed."

**[SHOW TEST RESULTS]**

**SCRIPT:**
> "Test passed with 100% provenance coverage. Let's check the metrics:"

**[TYPE IN TERMINAL]**
```bash
cat artifacts/W1/metrics.json
```

**[SHOW METRICS WITH COST DATA]**

**SCRIPT:**
> "Here are our runtime metrics:"
> "- Tool use rate: 50% (half the LLM steps are tool calls)"
> "- Zero naked numbers blocked (because the LLM learned to always use tools)"
> "- Cost per request: $0.00465 (less than half a cent)"
> "- Deterministic execution in Replay mode"

---

### Part 6: Determinism & Reproducibility (11:00 - 12:30)

**[SCREEN: Terminal]**

**SCRIPT:**
> "One of our key requirements is deterministic execution. Let me demonstrate:"

**[TYPE IN TERMINAL]**
```bash
# Run the same test twice
python test_boiler_replacement_agent_ai.py::TestBoilerGolden::test_example_input -v > run1.log
python test_boiler_replacement_agent_ai.py::TestBoilerGolden::test_example_input -v > run2.log
diff run1.log run2.log
```

**[RUN COMMANDS]**

**SCRIPT:**
> "Identical output. Byte-for-byte reproducibility. This is achieved through:"
> "- Temperature=0.0 (no randomness)"
> "- Seed=42 (deterministic sampling)"
> "- Replay mode (no network egress)"
> "- Tool-first architecture (no guessing)"

**[TYPE IN TERMINAL]**
```bash
# Show how we enforce this in CI
cat .github/workflows/no-naked-numbers.yml | grep -A 5 "GL_MODE"
```

**[SHOW WORKFLOW]**

**SCRIPT:**
> "Notice: We set GL_MODE=replay in our CI workflows. This ensures all tests run deterministically. Any agent that tries to guess a number will fail the build."

---

### Part 7: Cost Analysis & Scaling (12:30 - 13:30)

**[SCREEN: VS Code - Show metrics.json]**

**SCRIPT:**
> "Let's talk about cost. Here's our actual cost tracking:"

**[HIGHLIGHT COST METRICS]**

**SCRIPT:**
> "Per request costs:"
> "- Input tokens: 487 (~$0.0015)"
> "- Output tokens: 213 (~$0.0032)"
> "- Total: $0.00465"
>
> "Scaling estimates:"
> "- 1,000 requests: $4.65"
> "- 10,000 requests: $46.50"
> "- 100,000 requests: $465"
> "- 1 million requests: $4,650"

**[SCROLL TO BREAKDOWN]**

**SCRIPT:**
> "If we add RAG (Retrieval-Augmented Generation), we add embedding costs:"
> "- Embedding tokens: 256 per request"
> "- Vector DB queries: 5 per request"
> "- Additional cost: ~$0.00015"
>
> "Still very affordable for production use."

---

### Part 8: Production Readiness (13:30 - 14:30)

**[SCREEN: Terminal]**

**SCRIPT:**
> "These agents are production-ready. Let me show you the test coverage:"

**[TYPE IN TERMINAL]**
```bash
pytest tests/agents/test_*_agent_ai.py -v --cov=greenlang.agents --cov-report=term-missing | grep -A 20 "coverage"
```

**[SHOW COVERAGE]**

**SCRIPT:**
> "We have 166 tests covering:"
> "- 52 tests for Boiler Replacement Agent"
> "- 48 tests for Industrial Process Heat Agent"
> "- 66 tests for Industrial Heat Pump Agent"
>
> "All with full provenance tracking, deterministic execution, and snapshot validation."

**[TYPE IN TERMINAL]**
```bash
# Show CI status
gl doctor
```

**[RUN DOCTOR COMMAND]**

**SCRIPT:**
> "Our gl doctor command validates the entire stack:"
> "- SBOM generation"
> "- Digital signing"
> "- Provenance tracking"
> "- Sandbox isolation"
> "- Network policy enforcement"
> "- RAG allowlist"
>
> "All checks passing. Ready for production deployment."

---

### Conclusion (14:30 - 15:00)

**[SCREEN: VS Code - Show DOC-601_USING_TOOLS_NOT_GUESSING.md]**

**SCRIPT:**
> "To summarize: We've built a production-grade AI platform that eliminates hallucination through three key innovations:"
>
> "**1. Tool-First Architecture** - All calculations happen in validated tools, not in the LLM's 'head'."
>
> "**2. Provenance Tracking** - Every number is traceable to its source tool with complete metadata."
>
> "**3. Enforcement at Runtime** - The no-naked-numbers policy blocks any unverified numeric output."
>
> "The result? Three AI agents that can make industrial decarbonization recommendations with complete confidence, reproducibility, and auditability."
>
> "All the code is open source. Documentation is at greenlang.io. Thank you for watching!"

**[FADE TO BLACK]**

---

## Post-Recording Checklist

After recording, verify:

1. ✅ All terminal commands executed successfully
2. ✅ Test outputs show passing results
3. ✅ Provenance samples display correctly
4. ✅ Metrics show expected values
5. ✅ Audio is clear and synchronized
6. ✅ Screen resolution is 1920x1080 or higher
7. ✅ Video length is 14-16 minutes

---

## Technical Notes

### Camera/Screen Recording Settings

- **Resolution:** 1920x1080 (1080p)
- **Frame Rate:** 30 fps
- **Audio:** 48kHz, stereo
- **Format:** MP4 (H.264 codec)
- **Bitrate:** 8-10 Mbps

### Recommended Tools

- **Screen Recording:** OBS Studio (free, open source)
- **Video Editing:** DaVinci Resolve (free version)
- **Audio Recording:** Blue Yeti or similar USB microphone
- **Terminal:** Windows Terminal with PowerShell

### Fallback Plans

If any test fails during recording:

1. **Test failure:** Use `pytest -v --lf` to re-run last failed test
2. **Import error:** Verify virtual environment is activated
3. **File not found:** Check working directory with `pwd`
4. **Agent timeout:** Increase timeout in test with `@pytest.mark.timeout(30)`

---

## Transcript Template (for YouTube/captions)

```
[00:00] Introduction to GreenLang v0.3.0
[01:30] The AI Hallucination Problem
[03:00] Tool-First Architecture Solution
[05:00] Demo: Boiler Replacement Agent
[07:30] Demo: Industrial Process Heat Agent
[09:30] Demo: Industrial Heat Pump Agent
[11:00] Determinism & Reproducibility
[12:30] Cost Analysis & Scaling
[13:30] Production Readiness
[14:30] Conclusion & Call to Action
```

---

## File Artifacts to Include

Upload these to artifacts/W1/ after recording:

1. **demo_video.mp4** - Final edited video
2. **demo_transcript.txt** - Full transcript with timestamps
3. **demo_screenshots/** - Key frames from video
4. **demo_commands.txt** - All terminal commands used
5. **demo_test_outputs.log** - Test execution logs

---

## Distribution Channels

After recording, upload to:

1. **YouTube:** GreenLang official channel
2. **Vimeo:** For embedding in docs
3. **LinkedIn:** For professional audience
4. **Twitter/X:** Short clips with highlights
5. **greenlang.io:** Embedded in documentation

---

**Script Version:** 1.0
**Last Updated:** October 22, 2025
**Prepared By:** Claude (AI Assistant)
**Review Status:** Ready for Recording
