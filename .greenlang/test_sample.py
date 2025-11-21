# -*- coding: utf-8 -*-
"""
Sample code for testing migration tools.
This file contains patterns that should be detected and migrated.
"""

import os
import openai
import redis
import jsonschema
from typing import List, Dict


class DataProcessorAgent:
    """Custom agent that should be converted to inherit from Agent."""

    def __init__(self, config):
        self.config = config
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = openai.OpenAI(api_key=api_key)
        self.cache = redis.Redis(host='localhost', port=6379)

    def process(self, data):
        """Process data."""
        # Validate data
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        jsonschema.validate(data, schema)

        # Cache result
        key = f"result:{data.get('id')}"
        cached = self.cache.get(key)

        if cached:
            return cached

        # Process with LLM
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"Process: {data}"}
            ]
        )

        result = response.choices[0].message.content

        # Cache
        self.cache.set(key, result, ex=3600)

        return result

    def process_batch(self, items: List[Dict]):
        """Process multiple items."""
        results = []
        for item in items:
            result = self.process(item)
            results.append(result)
        return results
