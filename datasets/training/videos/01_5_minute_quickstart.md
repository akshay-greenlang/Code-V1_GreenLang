# Video 1: GreenLang 5-Minute Quick Start

**Duration:** 5 minutes
**Target Audience:** All developers
**Goal:** Get developers productive in 5 minutes

---

## Video Script

### Opening (0:00 - 0:30)

**[VISUAL: GreenLang logo animation]**

**NARRATION:**
"Welcome to GreenLang! In just 5 minutes, you'll learn the one rule that will transform how you build applications at our company: Never write custom code when infrastructure exists. Let's dive in."

**[TRANSITION: Fade to developer at computer]**

---

### The Problem (0:30 - 1:30)

**[VISUAL: Split screen showing duplicate code across multiple files]**

**NARRATION:**
"Before GreenLang, our codebase looked like this. Forty-seven different implementations of LLM chat. Twenty-three different caching systems. Every developer reinventing the wheel."

**[VISUAL: Highlight code with red circles around duplicated patterns]**

**NARRATION:**
"The result? Hundreds of wasted hours. Thousands in unnecessary costs. Countless bugs. Code reviews that asked the same question over and over: 'Why didn't you use infrastructure?'"

**[VISUAL: Chart showing time/cost waste]**

**NARRATION:**
"We needed a solution. Enter GreenLang-First."

**[TRANSITION: Smooth wipe to clean code]**

---

### The Solution (1:30 - 2:30)

**[VISUAL: Side-by-side comparison]**

**LEFT SIDE (RED X):**
```python
import openai
client = openai.Client(api_key="...")
response = client.chat.completions.create(...)
```

**RIGHT SIDE (GREEN CHECK):**
```python
from GL_COMMONS.infrastructure.llm import ChatSession
session = ChatSession(provider="openai")
response = session.send_message("Hello")
```

**NARRATION:**
"GreenLang-First is simple. Instead of writing custom code, use our infrastructure. One interface for all LLM providers. Automatic caching, retry logic, cost tracking, and monitoring. All built-in."

**[VISUAL: Icons appearing showing features: cache, retry, cost, monitoring]**

**NARRATION:**
"The same pattern applies everywhere. Agents, database access, validation, caching. One infrastructure. One way. Zero exceptions."

---

### Enforcement (2:30 - 3:30)

**[VISUAL: Terminal showing pre-commit hook]**

**NARRATION:**
"Here's the magic. We enforce GreenLang-First automatically. Watch what happens when someone tries to use custom code."

**[VISUAL: Developer typing custom code]**

```python
import openai  # This will be caught!
```

**[VISUAL: Git commit attempt]**

```bash
git commit -m "Add feature"
```

**[VISUAL: Red error message appears]**

```
GreenLang-First Policy Violation Detected!

VIOLATION: Direct OpenAI import
Required: Use ChatSession instead

See: docs/GREENLANG_FIRST_POLICY.md
```

**NARRATION:**
"Pre-commit hooks catch violations before they reach the repository. Fix the violation, commit succeeds. It's that simple."

**[VISUAL: Developer fixing code, commit succeeds with green checkmarks]**

---

### Get Started (3:30 - 4:30)

**[VISUAL: Terminal commands with clean typography]**

**NARRATION:**
"Ready to get started? Three commands. That's it."

**[VISUAL: Commands appear one by one]**

```bash
# Install pre-commit hooks
pre-commit install

# Check your code
pre-commit run --all-files

# Start coding with infrastructure
```

**[VISUAL: Quick code example montage]**

```python
# LLM
from GL_COMMONS.infrastructure.llm import ChatSession

# Caching
from GL_COMMONS.infrastructure.cache import CacheManager

# Agents
from GL_COMMONS.infrastructure.agents import Agent

# Validation
from GL_COMMONS.infrastructure.validation import ValidationFramework
```

**NARRATION:**
"Four core components. ChatSession for LLMs. CacheManager for caching. Agent for business logic. ValidationFramework for data quality. Everything else builds on these."

---

### Call to Action (4:30 - 5:00)

**[VISUAL: Resources screen with links]**

**NARRATION:**
"That's it! You now understand GreenLang-First. Next steps: Take Workshop 1 for hands-on practice. Join us on Slack at #greenlang-help. Read the full policy in the docs."

**[VISUAL: Three boxes appearing with animations]**

📚 **Workshop 1** → Hands-on training
💬 **#greenlang-help** → Get support
📖 **Docs** → Deep dive

**NARRATION:**
"Remember: before writing any code, ask yourself: does infrastructure already do this? If yes, use it. If no, write an ADR. Keep it simple. Keep it GreenLang."

**[VISUAL: GreenLang logo]**

**NARRATION:**
"Happy coding!"

---

## Slide Deck Outline

### Slide 1: Title
- GreenLang 5-Minute Quick Start
- One Rule to Rule Them All

### Slide 2: The Problem
- 47 different LLM implementations
- 23 different cache systems
- $12K+ wasted on duplicate LLM calls
- 200+ hours in redundant code reviews

### Slide 3: The Rule
**NEVER WRITE CUSTOM CODE WHEN INFRASTRUCTURE EXISTS**

### Slide 4: Before & After
| Before (❌) | After (✅) |
|------------|----------|
| import openai | from GL_COMMONS.infrastructure.llm import ChatSession |
| 50 lines of custom code | 3 lines using infrastructure |
| No caching, retry, monitoring | All included automatically |

### Slide 5: Enforcement
- Pre-commit hooks (local)
- CI/CD pipeline (GitHub)
- Code review (human)

### Slide 6: Get Started
```bash
pre-commit install
pre-commit run --all-files
```

### Slide 7: Core Components
- ChatSession → LLMs
- CacheManager → Caching
- Agent → Business logic
- ValidationFramework → Data quality

### Slide 8: Next Steps
- Workshop 1
- #greenlang-help
- docs/GREENLANG_FIRST_POLICY.md

---

## Demo Scenario

### Setup
- Clean repository
- VSCode editor
- Terminal side-by-side

### Demo Steps

1. **Show violation**
   ```python
   # bad_code.py
   import openai
   client = openai.Client()
   ```

2. **Try to commit**
   ```bash
   git add bad_code.py
   git commit -m "Add feature"
   # FAILS with violation
   ```

3. **Fix with infrastructure**
   ```python
   # good_code.py
   from GL_COMMONS.infrastructure.llm import ChatSession
   session = ChatSession(provider="openai")
   ```

4. **Commit succeeds**
   ```bash
   git add good_code.py
   git commit -m "Add feature using infrastructure"
   # SUCCESS
   ```

5. **Show features**
   ```python
   # All built-in:
   print(session.get_cost())  # Cost tracking
   print(session.get_token_count())  # Token counting
   # Caching, retry, monitoring - automatic!
   ```

---

## Production Notes

### Visuals Needed
- GreenLang logo animation
- Code editor screenshots (VSCode)
- Terminal recordings
- Diagram: Before/After architecture
- Icons: cache, retry, cost, monitoring
- Charts: cost savings, time savings

### Music
- Upbeat, modern tech music
- Medium energy (not too aggressive)
- Fade under narration
- Swell at key moments

### Pacing
- Fast but clear
- Pause after key points
- Emphasize "Never write custom code"
- Enthusiasm without hype

### Captions
- Full captions for accessibility
- Code syntax highlighted
- Key terms in bold

---

## Follow-Up Videos

After watching, viewers should see:
- Video 2: Building Your First Agent (15 min)
- Workshop 1: Deep dive (2 hours)
- Quick Reference Card (PDF download)
