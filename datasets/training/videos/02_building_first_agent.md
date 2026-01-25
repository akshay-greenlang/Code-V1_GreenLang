# Video 2: Building Your First Agent

**Duration:** 15 minutes
**Prerequisites:** Video 1 or Workshop 1
**Goal:** Build a complete working agent

---

## Script Overview

### Part 1: Agent Basics (0:00 - 3:00)
- What is an agent?
- Agent lifecycle: setup → execute → teardown
- Why inherit from Agent base class?

### Part 2: Live Coding (3:00 - 11:00)
**Build: Emission Calculator Agent**

```python
from GL_COMMONS.infrastructure.agents import Agent
from GL_COMMONS.infrastructure.llm import ChatSession
from GL_COMMONS.infrastructure.cache import CacheManager

class EmissionAgent(Agent):
    def setup(self):
        self.llm = ChatSession(provider="openai", model="gpt-4")
        self.cache = CacheManager()

    def execute(self):
        activity = self.input_data["activity"]
        amount = self.input_data["amount"]

        # Check cache
        cache_key = f"emission:{activity}:{amount}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # Calculate with LLM
        result = self.llm.send_message(
            f"Calculate CO2 for {amount} {activity}"
        )

        # Cache result
        self.cache.set(cache_key, result, ttl=3600)
        return result

    def teardown(self):
        self.llm.close()
```

### Part 3: Testing (11:00 - 14:00)
- Run the agent
- Check metrics
- Verify caching works
- View logs

### Part 4: Wrap-up (14:00 - 15:00)
- Key takeaways
- Next steps
- Resources

---

## Demo Script

**[Show completed agent running]**

"By the end of this video, you'll have built this: a production-ready emission calculator that uses LLM infrastructure, caching, and monitoring. Let's build it step by step."

**[Split screen: code + explanation]**

"Every agent starts with three methods..."

---

See full implementation in Workshop 3 materials.
