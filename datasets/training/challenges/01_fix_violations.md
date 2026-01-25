# Challenge 1: Fix the Violations

**Difficulty:** Beginner
**Time:** 30 minutes
**Goal:** Find and fix 10 GreenLang-First policy violations

---

## Instructions

You've inherited a codebase with multiple infrastructure violations. Your task: fix all violations using GreenLang infrastructure.

## The Code (BROKEN)

```python
# bad_agent.py - FULL OF VIOLATIONS!
import openai  # VIOLATION 1
import anthropic  # VIOLATION 2
import redis  # VIOLATION 3
import psycopg2  # VIOLATION 4
import logging

logger = logging.getLogger(__name__)

class EmissionCalculator:  # VIOLATION 5: Doesn't inherit from Agent
    def __init__(self):
        # VIOLATION 6: Direct OpenAI client
        self.openai_client = openai.Client(api_key="sk-...")

        # VIOLATION 7: Direct Redis client
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379
        )

        # VIOLATION 8: Direct database connection
        self.db_conn = psycopg2.connect(
            host="localhost",
            database="emissions",
            user="admin",
            password="admin123"
        )

    def calculate(self, activity, amount):
        # VIOLATION 9: Manual token counting
        tokens = len(activity.split()) + len(str(amount).split())

        # VIOLATION 10: Custom retry logic (instead of infrastructure)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": f"Calculate {amount} {activity}"}]
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

        return response.choices[0].message.content
```

---

## Your Task

Fix all 10 violations by using GreenLang infrastructure.

### Violation Checklist

- [ ] VIOLATION 1: Direct openai import
- [ ] VIOLATION 2: Direct anthropic import
- [ ] VIOLATION 3: Direct redis import
- [ ] VIOLATION 4: Direct psycopg2 import
- [ ] VIOLATION 5: Doesn't inherit from Agent
- [ ] VIOLATION 6: Direct OpenAI client
- [ ] VIOLATION 7: Direct Redis client
- [ ] VIOLATION 8: Direct database connection
- [ ] VIOLATION 9: Manual token counting
- [ ] VIOLATION 10: Custom retry logic

---

## Solution

```python
# good_agent.py - FIXED!
from GL_COMMONS.infrastructure.agents import Agent  # FIX 5
from GL_COMMONS.infrastructure.llm import ChatSession  # FIX 1, 6
from GL_COMMONS.infrastructure.cache import CacheManager  # FIX 3, 7
from GL_COMMONS.infrastructure.database import DatabaseManager  # FIX 4, 8
import logging

logger = logging.getLogger(__name__)

class EmissionCalculator(Agent):  # FIX 5: Inherit from Agent
    def setup(self):
        # FIX 6: Use ChatSession (includes automatic retry - FIX 10)
        self.llm = ChatSession(
            provider="openai",
            model="gpt-4",
            max_retries=3  # FIX 10: Automatic retry
        )

        # FIX 7: Use CacheManager
        self.cache = CacheManager()

        # FIX 8: Use DatabaseManager
        self.db = DatabaseManager()

    def execute(self):
        activity = self.input_data["activity"]
        amount = self.input_data["amount"]

        # Call LLM (automatic token counting - FIX 9)
        response = self.llm.send_message(
            f"Calculate {amount} {activity}"
        )

        # FIX 9: Automatic token counting
        tokens = self.llm.get_token_count()
        logger.info(f"Used {tokens} tokens")

        return response

    def teardown(self):
        self.llm.close()
```

---

## Grading

Run the auto-grader:

```bash
python grade_challenge_1.py bad_agent.py
```

**Expected output:**
```
Challenge 1: Fix the Violations
================================

✓ All violations fixed!
✓ Code passes GreenLang-First checks
✓ All tests pass

Score: 100/100

Completion time: 28 minutes
Grade: A+
```

---

## Bonus Challenge

Add these enhancements:
1. Add semantic caching
2. Add telemetry/metrics
3. Add input validation
4. Add error handling with logging

**Bonus points:** +20
