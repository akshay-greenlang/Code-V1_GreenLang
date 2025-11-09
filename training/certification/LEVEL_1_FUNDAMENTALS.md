# Level 1: GreenLang-First Fundamentals Certification

**Duration:** 90 minutes
**Passing Score:** 80%
**Prerequisites:** Workshop 1 completed

---

## Part 1: Multiple Choice (40 points)

### Question 1 (2 points)
What is the core principle of GreenLang-First?

A) Write the best code possible
B) Never write custom code when infrastructure exists
C) Always use the latest technologies
D) Optimize for performance first

**Answer:** B

---

### Question 2 (2 points)
Which import is ALLOWED under GreenLang-First?

A) `import openai`
B) `import anthropic`
C) `from GL_COMMONS.infrastructure.llm import ChatSession`
D) `import redis`

**Answer:** C

---

### Question 3 (2 points)
What are the three layers of enforcement?

A) Tests, Reviews, Production
B) Pre-commit hooks, CI/CD, Code review
C) Local, Staging, Production
D) Developer, Manager, Executive

**Answer:** B

---

### Question 4 (2 points)
When is custom code allowed?

A) Never
B) When you're in a hurry
C) When you have an approved ADR
D) When infrastructure is too complex

**Answer:** C

---

### Question 5 (2 points)
What does ADR stand for?

A) Automated Decision Record
B) Architecture Decision Record
C) Agent Development Record
D) Application Design Reference

**Answer:** B

---

### Question 6 (2 points)
Which is the correct way to use LLM?

A) `openai.Client().chat.completions.create()`
B) `anthropic.Anthropic().messages.create()`
C) `ChatSession(provider="openai").send_message()`
D) `llm_call(prompt)`

**Answer:** C

---

### Question 7 (2 points)
What must all agents inherit from?

A) `class Agent`
B) `BaseAgent`
C) `AgentInterface`
D) Nothing - standalone classes are fine

**Answer:** A

---

### Question 8 (2 points)
What are the three agent lifecycle methods?

A) start, run, stop
B) init, execute, cleanup
C) setup, execute, teardown
D) begin, process, end

**Answer:** C

---

### Question 9 (2 points)
Which caching solution should you use?

A) `import redis` directly
B) Custom cache implementation
C) `CacheManager` from infrastructure
D) No caching needed

**Answer:** C

---

### Question 10 (2 points)
What happens if you try to commit code with violations?

A) Commit succeeds with warning
B) Pre-commit hook blocks the commit
C) CI/CD catches it later
D) Nothing - it's just a guideline

**Answer:** B

---

### Questions 11-20 (20 points)
[Additional questions covering: validation, database access, monitoring, error handling, best practices, etc.]

---

## Part 2: Code Review (30 points)

### Exercise 1 (15 points)
Identify all violations in this code:

```python
import openai
import redis

class DataProcessor:
    def __init__(self):
        self.client = openai.Client(api_key="sk-...")
        self.redis = redis.Redis(host='localhost')

    def process(self, data):
        # Manual validation
        if not data:
            raise ValueError("Empty data")

        # Call OpenAI
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": str(data)}]
        )

        return response.choices[0].message.content
```

**Violations to identify:**
1. Direct openai import
2. Direct redis import
3. Doesn't inherit from Agent
4. Direct OpenAI client usage
5. Direct Redis client usage
6. Manual validation instead of ValidationFramework
7. No caching
8. No error handling
9. No logging
10. No metrics/telemetry

**Points:** 1.5 points per violation found (max 15)

---

### Exercise 2 (15 points)
Fix the code using GreenLang infrastructure:

**Correct Answer:**
```python
from GL_COMMONS.infrastructure.agents import Agent
from GL_COMMONS.infrastructure.llm import ChatSession
from GL_COMMONS.infrastructure.cache import CacheManager
from GL_COMMONS.infrastructure.validation import ValidationFramework
import logging

logger = logging.getLogger(__name__)

class DataProcessor(Agent):
    def setup(self):
        self.llm = ChatSession(provider="openai", model="gpt-4")
        self.cache = CacheManager()
        self.validator = ValidationFramework()

    def execute(self):
        data = self.input_data

        # Validate
        schema = {"data": {"type": "string", "required": True}}
        self.validator.validate({"data": data}, schema)

        # Check cache
        cache_key = f"processed:{hash(data)}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # Process with LLM
        try:
            response = self.llm.send_message(str(data))

            # Cache result
            self.cache.set(cache_key, response, ttl=3600)

            return response

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

    def teardown(self):
        self.llm.close()
```

**Grading:**
- Inherits from Agent: 3 points
- Uses ChatSession: 3 points
- Uses CacheManager: 3 points
- Uses ValidationFramework: 2 points
- Proper error handling: 2 points
- Logging: 2 points

---

## Part 3: Practical Coding (30 points)

### Task: Build a Simple Agent

**Requirements:**
1. Create an agent that calculates simple math expressions
2. Use ChatSession for complex calculations
3. Implement caching
4. Add validation
5. Include error handling

**Starter Code:**
```python
from GL_COMMONS.infrastructure.agents import Agent

class MathAgent(Agent):
    def setup(self):
        # TODO: Initialize infrastructure
        pass

    def execute(self):
        # TODO: Implement calculation
        pass

    def teardown(self):
        # TODO: Cleanup
        pass
```

**Test Cases:**
```python
agent = MathAgent()
agent.setup()

# Test 1: Simple calculation
result = agent.execute_with_input({"expression": "2 + 2"})
assert result == "4"

# Test 2: Complex calculation
result = agent.execute_with_input({"expression": "What is 15% of 200?"})
assert "30" in result

# Test 3: Caching works
# Second call should be cached
```

**Grading:**
- Agent inherits correctly: 5 points
- Uses ChatSession: 8 points
- Implements caching: 7 points
- Validates input: 5 points
- Error handling: 5 points

---

## Answer Key

### Part 1: Multiple Choice
1. B
2. C
3. B
4. C
5. B
6. C
7. A
8. C
9. C
10. B

### Part 2: Code Review
See solutions above

### Part 3: Practical Coding
See reference implementation:

```python
from GL_COMMONS.infrastructure.agents import Agent
from GL_COMMONS.infrastructure.llm import ChatSession
from GL_COMMONS.infrastructure.cache import CacheManager
from GL_COMMONS.infrastructure.validation import ValidationFramework
import logging

logger = logging.getLogger(__name__)

class MathAgent(Agent):
    def setup(self):
        self.llm = ChatSession(
            provider="openai",
            model="gpt-4",
            system_message="You are a math calculator. Return only the numeric answer."
        )
        self.cache = CacheManager()
        self.validator = ValidationFramework()

    def execute(self):
        expression = self.input_data.get("expression")

        # Validate
        schema = {"expression": {"type": "string", "required": True}}
        self.validator.validate({"expression": expression}, schema)

        # Check cache
        cache_key = f"math:{expression}"
        cached = self.cache.get(cache_key)
        if cached:
            logger.info("Cache hit")
            return cached

        # Calculate with LLM
        try:
            result = self.llm.send_message(f"Calculate: {expression}")

            # Cache result
            self.cache.set(cache_key, result, ttl=3600)

            return result

        except Exception as e:
            logger.error(f"Calculation failed: {e}")
            raise

    def teardown(self):
        if self.llm:
            self.llm.close()
```

---

## Scoring

| Section | Points | Weight |
|---------|--------|--------|
| Multiple Choice | 40 | 40% |
| Code Review | 30 | 30% |
| Practical Coding | 30 | 30% |
| **Total** | **100** | **100%** |

**Passing Score:** 80/100 (80%)

---

## Certificate

Upon passing, you receive:
- **GreenLang-First Fundamentals Certificate**
- Digital badge for email signature
- Access to Level 2 certification
- Listed in company certification registry

---

## Next Steps

After certification:
1. Take Level 2: Infrastructure Practitioner
2. Build production agents
3. Contribute to infrastructure
4. Mentor new developers

---

**Good luck!**
