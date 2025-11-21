#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Pair Programming Assistant

Conversational AI helper for infrastructure questions.
Context-aware assistant that knows your codebase.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any


class KnowledgeBase:
    """Infrastructure knowledge base."""

    def __init__(self):
        self.infrastructure_docs = self._build_knowledge_base()

    def _build_knowledge_base(self) -> Dict[str, str]:
        """Build knowledge base of infrastructure components."""
        return {
            "ChatSession": """
ChatSession is the unified LLM interface for GreenLang.

Usage:
```python
from shared.infrastructure.llm import ChatSession

session = ChatSession(provider='openai', model='gpt-4')
response = session.chat("Your prompt here")
print(response.content)
```

Supports: OpenAI, Anthropic, Google AI
Features: Token counting, streaming, conversation history
""",

            "BaseAgent": """
BaseAgent is the base class for all agents in GreenLang.

Usage:
```python
from shared.infrastructure.agents import BaseAgent

class MyAgent(BaseAgent):
    def execute(self, input_data: dict) -> dict:
        # Your logic here
        return {"status": "success", "result": data}

    def validate_input(self, input_data: dict) -> bool:
        return True
```

Features: Batch processing, validation, logging, monitoring
""",

            "CacheManager": """
CacheManager provides distributed caching with Redis.

Usage:
```python
from shared.infrastructure.cache import CacheManager

cache = CacheManager(ttl=3600)

# Decorator
@cache.cached(key_prefix="my_func")
def expensive_function(args):
    return result

# Manual
cache.set("key", value)
result = cache.get("key")
```

Features: TTL, invalidation, cache warming, fallback
""",

            "ValidationFramework": """
ValidationFramework provides schema-based validation.

Usage:
```python
from shared.infrastructure.validation import ValidationFramework, Field

class MyValidator(ValidationFramework):
    schema = {
        "field1": Field(type=str, required=True),
        "field2": Field(type=int, min_value=0)
    }

validator = MyValidator()
result = validator.validate(data)
```

Features: Schema validation, business rules, detailed errors
""",

            "caching": """
For caching in GreenLang, use CacheManager:

1. Import: from shared.infrastructure.cache import CacheManager
2. Initialize: cache = CacheManager(ttl=3600)
3. Use decorator or manual methods

Benefits: Distributed caching, automatic TTL, metrics
""",

            "validation": """
For validation in GreenLang, use ValidationFramework:

1. Define schema with Field objects
2. Implement validate_business_rules() for custom logic
3. Call validator.validate(data)

Features: Type checking, required fields, custom rules
""",

            "logging": """
For logging in GreenLang, use Logger:

```python
from shared.infrastructure.logging import Logger

logger = Logger(name=__name__)
logger.info("Message", extra={"context": "value"})
```

Features: Structured logging, correlation IDs, JSON output
""",

            "agents": """
For creating agents in GreenLang:

1. Inherit from BaseAgent
2. Implement execute() method
3. Add validate_input() and validate_output()

Example:
```python
class MyAgent(BaseAgent):
    def execute(self, input_data):
        # Process data
        return {"result": data}
```
"""
        }

    def search(self, query: str) -> str:
        """Search knowledge base."""
        query_lower = query.lower()

        # Direct match
        for key, value in self.infrastructure_docs.items():
            if key.lower() in query_lower:
                return value

        # Keyword match
        if any(word in query_lower for word in ['cache', 'caching', 'cached']):
            return self.infrastructure_docs.get('caching', '')

        if any(word in query_lower for word in ['validate', 'validation', 'check']):
            return self.infrastructure_docs.get('validation', '')

        if any(word in query_lower for word in ['log', 'logging', 'print']):
            return self.infrastructure_docs.get('logging', '')

        if any(word in query_lower for word in ['agent', 'agents', 'workflow']):
            return self.infrastructure_docs.get('agents', '')

        if any(word in query_lower for word in ['llm', 'gpt', 'openai', 'chat']):
            return self.infrastructure_docs.get('ChatSession', '')

        return ""


class AIAssistant:
    """AI pair programming assistant."""

    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.conversation_history = []

    def chat(self, user_message: str) -> str:
        """Process user message and generate response."""
        self.conversation_history.append({"role": "user", "content": user_message})

        # Search knowledge base
        kb_result = self.knowledge_base.search(user_message)

        # Generate response
        response = self._generate_response(user_message, kb_result)

        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def _generate_response(self, user_message: str, kb_result: str) -> str:
        """Generate response to user message."""
        message_lower = user_message.lower()

        # Handle common questions
        if any(word in message_lower for word in ['how', 'what', 'when', 'where', 'why']):
            if kb_result:
                return f"Here's what I found:\n\n{kb_result}\n\nLet me know if you need more details!"

        # Code questions
        if 'code' in message_lower or 'example' in message_lower:
            if kb_result:
                return f"Here's an example:\n\n{kb_result}"

        # Migration questions
        if 'migrate' in message_lower or 'convert' in message_lower:
            return """To migrate to GreenLang infrastructure:

1. Run: greenlang migrate --app your-app --dry-run
2. Review the recommendations
3. Apply changes: greenlang migrate --app your-app --auto-fix
4. Test your application

Need help with a specific migration? Ask me!"""

        # Performance questions
        if 'performance' in message_lower or 'slow' in message_lower or 'optimize' in message_lower:
            return """To improve performance:

1. Add caching with CacheManager
2. Use batch processing in agents
3. Profile with: greenlang profile your-file.py
4. Check cache hit rates

What specific performance issue are you facing?"""

        # Testing questions
        if 'test' in message_lower:
            return """For testing with GreenLang infrastructure:

1. Use pytest for unit tests
2. Test agents with sample data
3. Mock infrastructure components
4. Check validation logic

Example:
```python
def test_agent():
    agent = MyAgent()
    result = agent.execute({"data": "test"})
    assert result["status"] == "success"
```

Need help with specific tests?"""

        # Default response
        if kb_result:
            return kb_result

        return """I'm here to help with GreenLang infrastructure!

Ask me about:
- Using infrastructure components (ChatSession, BaseAgent, etc.)
- Migrating existing code
- Performance optimization
- Testing and debugging
- Best practices

What would you like to know?"""

    def start_interactive(self):
        """Start interactive chat session."""
        print("=" * 80)
        print("GREENLANG AI ASSISTANT")
        print("=" * 80)
        print("\nHi! I'm your GreenLang AI assistant. Ask me anything about infrastructure!")
        print("Type 'exit' or 'quit' to end the conversation.\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\nAssistant: Goodbye! Happy coding! 🌿")
                    break

                if not user_input:
                    continue

                response = self.chat(user_input)
                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                print("\n\nAssistant: Goodbye! Happy coding! 🌿")
                break
            except EOFError:
                break


class QuickHelp:
    """Quick help system."""

    TOPICS = {
        "llm": """
Using ChatSession for LLM interactions:

1. Import: from shared.infrastructure.llm import ChatSession
2. Create: session = ChatSession(provider='openai', model='gpt-4')
3. Use: response = session.chat("Your prompt")

Supports: OpenAI, Anthropic, Google AI
""",

        "agents": """
Creating agents with BaseAgent:

1. Import: from shared.infrastructure.agents import BaseAgent
2. Inherit: class MyAgent(BaseAgent)
3. Implement: execute(), validate_input(), validate_output()

Features: Batch processing, validation, logging
""",

        "caching": """
Using CacheManager:

1. Import: from shared.infrastructure.cache import CacheManager
2. Create: cache = CacheManager(ttl=3600)
3. Use: @cache.cached(key_prefix="func") or cache.get/set

Features: Redis backend, TTL, invalidation
""",

        "migration": """
Migrating to GreenLang infrastructure:

1. Scan: greenlang migrate --app your-app --dry-run
2. Review: Check recommendations and impact
3. Apply: greenlang migrate --app your-app --auto-fix
4. Test: Run your test suite

Tools: migrate, recommend, health-check
"""
    }

    @staticmethod
    def show_help(topic: str = None):
        """Show quick help."""
        if topic and topic in QuickHelp.TOPICS:
            print(QuickHelp.TOPICS[topic])
        else:
            print("Available topics:")
            for t in QuickHelp.TOPICS.keys():
                print(f"  - {t}")
            print("\nUsage: greenlang chat --help <topic>")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='AI pair programming assistant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start interactive chat
  greenlang chat

  # Ask a specific question
  greenlang chat "How do I add caching to my agent?"

  # Get quick help
  greenlang chat --help llm
        """
    )

    parser.add_argument('question', nargs='?', help='Ask a specific question')
    parser.add_argument('--help-topic', help='Get quick help on a topic')

    args = parser.parse_args()

    assistant = AIAssistant()

    if args.help_topic:
        QuickHelp.show_help(args.help_topic)
    elif args.question:
        response = assistant.chat(args.question)
        print(f"\n{response}\n")
    else:
        assistant.start_interactive()


if __name__ == '__main__':
    main()
