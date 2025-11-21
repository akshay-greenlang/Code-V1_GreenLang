#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

logger = logging.getLogger(__name__)
GreenLang Infrastructure Code Generator

Generates boilerplate code for agents, pipelines, chat sessions, cache, and validation.
"""

import logging
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from greenlang.determinism import DeterministicClock


class CodeGenerator:
    """Generates GreenLang infrastructure code."""

    def __init__(self):
        self.templates = {
            'agent': self._agent_template,
            'pipeline': self._pipeline_template,
            'llm-session': self._llm_session_template,
            'cache': self._cache_template,
            'validation': self._validation_template,
            'config': self._config_template,
        }

    def _agent_template(self, name: str, **kwargs) -> str:
        """Generate agent template."""
        description = kwargs.get('description', f'{name} agent implementation')
        has_batch = kwargs.get('batch', False)

        template = f'''"""
{name}

{description}

Created: {DeterministicClock.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

from typing import Any, Dict, List, Optional
from greenlang.sdk.base import Agent
from greenlang.utils.logging import StructuredLogger


class {name}(Agent):
    """
    {description}
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the {name}.

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config or {{}}
        self.logger = StructuredLogger(__name__)

    def execute(self, input_data: Any) -> Any:
        """
        Execute the agent logic.

        Args:
            input_data: Input data to process

        Returns:
            Processed result
        """
        self.logger.info(f"Executing {{self.__class__.__name__}}", input_data=input_data)

        try:
            # TODO: Implement your agent logic here
            result = self._process(input_data)

            self.logger.info(f"{{self.__class__.__name__}} completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error in {{self.__class__.__name__}}", error=str(e))
            raise

    def _process(self, input_data: Any) -> Any:
        """
        Process the input data.

        Args:
            input_data: Input data

        Returns:
            Processed result
        """
        # TODO: Implement processing logic
        return input_data
'''

        if has_batch:
            template += '''

    def process_batch(self, items: List[Any], max_workers: int = 10) -> List[Any]:
        """
        Process a batch of items using the built-in batch processor.

        Args:
            items: List of items to process
            max_workers: Maximum number of parallel workers

        Returns:
            List of processed results
        """
        return self.batch_process(items, max_workers=max_workers)
'''

        template += '''


# Example usage
if __name__ == "__main__":
    # Create agent instance
    agent = {name}()

    # Execute agent
    result = agent.execute({{"example": "data"}})
    print(f"Result: {{result}}")
'''

        if has_batch:
            template += '''
    # Process batch
    items = [{"example": f"data_{i}"} for i in range(10)]
    results = agent.process_batch(items)
    print(f"Batch results: {results}")
'''

        return template

    def _pipeline_template(self, name: str, **kwargs) -> str:
        """Generate pipeline template."""
        agents = kwargs.get('agents', ['Agent1', 'Agent2', 'Agent3'])
        description = kwargs.get('description', f'{name} pipeline implementation')

        agents_str = ', '.join(agents)

        template = f'''"""
{name}

{description}

Created: {DeterministicClock.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

from typing import Any, Dict, List, Optional
from greenlang.sdk.base import Pipeline, Agent
from greenlang.utils.logging import StructuredLogger


# TODO: Import your agent classes
# from .agents import {agents_str}


class {name}(Pipeline):
    """
    {description}
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the {name}.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {{}}
        self.logger = StructuredLogger(__name__)

        # TODO: Initialize your agents
        agents = [
            # {agents_str}
        ]

        super().__init__(agents=agents)

    def preprocess(self, input_data: Any) -> Any:
        """
        Preprocess input before pipeline execution.

        Args:
            input_data: Input data

        Returns:
            Preprocessed data
        """
        self.logger.info("Preprocessing input")
        # TODO: Implement preprocessing logic
        return input_data

    def postprocess(self, result: Any) -> Any:
        """
        Postprocess result after pipeline execution.

        Args:
            result: Pipeline result

        Returns:
            Postprocessed result
        """
        self.logger.info("Postprocessing result")
        # TODO: Implement postprocessing logic
        return result


# Example usage
if __name__ == "__main__":
    # Create pipeline
    pipeline = {name}()

    # Execute pipeline
    input_data = {{"example": "data"}}
    result = pipeline.execute(input_data)
    print(f"Result: {{result}}")
'''

        return template

    def _llm_session_template(self, name: str, **kwargs) -> str:
        """Generate LLM session template."""
        provider = kwargs.get('provider', 'openai')
        model = kwargs.get('model', 'gpt-4')

        template = f'''"""
LLM Session Configuration

Provider: {provider}
Model: {model}

Created: {DeterministicClock.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

from typing import List, Dict, Any, Optional
from greenlang.intelligence import ChatSession
from greenlang.utils.logging import StructuredLogger


class LLMSessionManager:
    """
    Manages LLM chat sessions using GreenLang infrastructure.
    """

    def __init__(self, provider: str = "{provider}", model: str = "{model}"):
        """
        Initialize the LLM session manager.

        Args:
            provider: LLM provider (openai, anthropic, etc.)
            model: Model name
        """
        self.provider = provider
        self.model = model
        self.logger = StructuredLogger(__name__)
        self.session = None

    def create_session(self, system_prompt: Optional[str] = None) -> ChatSession:
        """
        Create a new chat session.

        Args:
            system_prompt: System prompt for the session

        Returns:
            ChatSession instance
        """
        self.logger.info(f"Creating {{self.provider}} session with model {{self.model}}")

        self.session = ChatSession(
            provider=self.provider,
            model=self.model,
            system_prompt=system_prompt
        )

        return self.session

    def chat(self, message: str, **kwargs) -> str:
        """
        Send a message and get response.

        Args:
            message: User message
            **kwargs: Additional parameters

        Returns:
            LLM response
        """
        if not self.session:
            self.create_session()

        self.logger.info("Sending message to LLM", message=message)

        response = self.session.send_message(message, **kwargs)

        self.logger.info("Received response from LLM")

        return response

    def batch_chat(self, messages: List[str], **kwargs) -> List[str]:
        """
        Process multiple messages in batch.

        Args:
            messages: List of messages
            **kwargs: Additional parameters

        Returns:
            List of responses
        """
        if not self.session:
            self.create_session()

        self.logger.info(f"Processing batch of {{len(messages)}} messages")

        responses = []
        for message in messages:
            response = self.chat(message, **kwargs)
            responses.append(response)

        return responses


# Example usage
if __name__ == "__main__":
    # Create session manager
    manager = LLMSessionManager()

    # Single message
    response = manager.chat("What is the capital of France?")
    print(f"Response: {{response}}")

    # Batch messages
    messages = [
        "What is 2+2?",
        "What is the weather like today?",
        "Tell me a joke"
    ]
    responses = manager.batch_chat(messages)
    for msg, resp in zip(messages, responses):
        print(f"Q: {{msg}}\\nA: {{resp}}\\n")
'''

        return template

    def _cache_template(self, name: str, **kwargs) -> str:
        """Generate cache manager template."""

        template = f'''"""
Cache Manager Configuration

Created: {DeterministicClock.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

from typing import Any, Optional
from greenlang.cache import CacheManager
from greenlang.utils.logging import StructuredLogger
import json


class {name}CacheManager:
    """
    Manages caching using GreenLang infrastructure.
    """

    def __init__(self, prefix: str = "{name.lower()}"):
        """
        Initialize cache manager.

        Args:
            prefix: Key prefix for namespacing
        """
        self.cache = CacheManager()
        self.prefix = prefix
        self.logger = StructuredLogger(__name__)

    def _make_key(self, key: str) -> str:
        """
        Create prefixed cache key.

        Args:
            key: Base key

        Returns:
            Prefixed key
        """
        return f"{{self.prefix}}:{{key}}"

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        full_key = self._make_key(key)
        value = self.cache.get(full_key)

        if value is None:
            self.logger.debug(f"Cache miss: {{full_key}}")
            return default

        self.logger.debug(f"Cache hit: {{full_key}}")
        return json.loads(value) if isinstance(value, str) else value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds

        Returns:
            Success status
        """
        full_key = self._make_key(key)
        serialized = json.dumps(value) if not isinstance(value, str) else value

        success = self.cache.set(full_key, serialized, ttl=ttl)

        if success:
            self.logger.debug(f"Cached: {{full_key}} (TTL: {{ttl}})")
        else:
            self.logger.warning(f"Failed to cache: {{full_key}}")

        return success

    def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            Success status
        """
        full_key = self._make_key(key)
        success = self.cache.delete(full_key)

        if success:
            self.logger.debug(f"Deleted from cache: {{full_key}}")

        return success

    def cached(self, ttl: Optional[int] = None):
        """
        Decorator for caching function results.

        Args:
            ttl: Time to live in seconds

        Returns:
            Decorator function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Create cache key from function name and arguments
                cache_key = f"{{func.__name__}}:{{args}}:{{kwargs}}"

                # Try to get from cache
                result = self.get(cache_key)

                if result is not None:
                    return result

                # Execute function
                result = func(*args, **kwargs)

                # Cache result
                self.set(cache_key, result, ttl=ttl)

                return result

            return wrapper
        return decorator


# Example usage
if __name__ == "__main__":
    # Create cache manager
    cache = {name}CacheManager()

    # Set value
    cache.set("user:123", {{"name": "John", "age": 30}}, ttl=3600)

    # Get value
    user = cache.get("user:123")
    print(f"User: {{user}}")

    # Use decorator
    @cache.cached(ttl=300)
    def expensive_computation(x):
        print(f"Computing {{x}}...")
        return x ** 2

    print(expensive_computation(5))  # Computes
    print(expensive_computation(5))  # From cache
'''

        return template

    def _validation_template(self, name: str, **kwargs) -> str:
        """Generate validation schema template."""

        template = f'''"""
Validation Schema Configuration

Created: {DeterministicClock.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

from typing import Any, Dict, List
from greenlang.validation import ValidationFramework
from greenlang.utils.logging import StructuredLogger


class {name}Validator:
    """
    Validation using GreenLang infrastructure.
    """

    def __init__(self):
        """Initialize validator."""
        self.validator = ValidationFramework()
        self.logger = StructuredLogger(__name__)

        # Define schemas
        self.schemas = {{
            "user": {{
                "type": "object",
                "properties": {{
                    "name": {{"type": "string", "minLength": 1}},
                    "email": {{"type": "string", "format": "email"}},
                    "age": {{"type": "integer", "minimum": 0, "maximum": 150}}
                }},
                "required": ["name", "email"]
            }},
            "product": {{
                "type": "object",
                "properties": {{
                    "id": {{"type": "string"}},
                    "name": {{"type": "string"}},
                    "price": {{"type": "number", "minimum": 0}},
                    "category": {{"type": "string"}}
                }},
                "required": ["id", "name", "price"]
            }}
        }}

    def validate(self, data: Any, schema_name: str) -> bool:
        """
        Validate data against schema.

        Args:
            data: Data to validate
            schema_name: Name of schema to use

        Returns:
            True if valid

        Raises:
            ValidationError if invalid
        """
        if schema_name not in self.schemas:
            raise ValueError(f"Unknown schema: {{schema_name}}")

        schema = self.schemas[schema_name]

        self.logger.info(f"Validating against {{schema_name}} schema")

        try:
            self.validator.validate_schema(data, schema)
            self.logger.info("Validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Validation failed: {{e}}")
            raise

    def validate_batch(self, items: List[Any], schema_name: str) -> List[bool]:
        """
        Validate multiple items.

        Args:
            items: List of items to validate
            schema_name: Schema name

        Returns:
            List of validation results
        """
        results = []
        for item in items:
            try:
                self.validate(item, schema_name)
                results.append(True)
            except Exception:
                results.append(False)

        return results

    def add_schema(self, name: str, schema: Dict[str, Any]):
        """
        Add a new validation schema.

        Args:
            name: Schema name
            schema: JSON schema definition
        """
        self.schemas[name] = schema
        self.logger.info(f"Added schema: {{name}}")


# Example usage
if __name__ == "__main__":
    # Create validator
    validator = {name}Validator()

    # Validate user data
    user_data = {{
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30
    }}

    try:
        validator.validate(user_data, "user")
        print("User data is valid!")
    except Exception as e:
        print(f"Validation error: {{e}}")

    # Validate product data
    product_data = {{
        "id": "PROD-123",
        "name": "Widget",
        "price": 29.99,
        "category": "Tools"
    }}

    try:
        validator.validate(product_data, "product")
        print("Product data is valid!")
    except Exception as e:
        print(f"Validation error: {{e}}")
'''

        return template

    def _config_template(self, name: str, **kwargs) -> str:
        """Generate config manager template."""

        template = f'''"""
Configuration Management

Created: {DeterministicClock.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

from typing import Any, Optional
from greenlang.config import Config
from greenlang.utils.logging import StructuredLogger
import os


class {name}Config:
    """
    Configuration management using GreenLang infrastructure.
    """

    def __init__(self, env: str = "development"):
        """
        Initialize configuration.

        Args:
            env: Environment name (development, staging, production)
        """
        self.config = Config()
        self.env = env
        self.logger = StructuredLogger(__name__)

        # Load environment-specific config
        self._load_config()

    def _load_config(self):
        """Load configuration based on environment."""
        self.logger.info(f"Loading configuration for {{self.env}}")

        # Define configuration
        self.settings = {{
            "development": {{
                "debug": True,
                "log_level": "DEBUG",
                "database_url": self.config.get("DEV_DATABASE_URL"),
                "api_timeout": 30
            }},
            "staging": {{
                "debug": False,
                "log_level": "INFO",
                "database_url": self.config.get("STAGING_DATABASE_URL"),
                "api_timeout": 60
            }},
            "production": {{
                "debug": False,
                "log_level": "WARNING",
                "database_url": self.config.get("PROD_DATABASE_URL"),
                "api_timeout": 120
            }}
        }}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value
        """
        env_settings = self.settings.get(self.env, {{}})
        return env_settings.get(key, default)

    def get_secret(self, key: str) -> Optional[str]:
        """
        Get secret value from environment.

        Args:
            key: Secret key

        Returns:
            Secret value or None
        """
        return self.config.get_secret(key)

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.env == "production"

    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.get("debug", False)


# Example usage
if __name__ == "__main__":
    # Create config
    config = {name}Config(env=os.getenv("ENV", "development"))

    # Get settings
    print(f"Debug mode: {{config.is_debug()}}")
    print(f"Log level: {{config.get('log_level')}}")
    print(f"API timeout: {{config.get('api_timeout')}}")

    # Get secrets
    api_key = config.get_secret("API_KEY")
    print(f"API key loaded: {{bool(api_key)}}")
'''

        return template

    def generate(self, type_name: str, name: str, output_path: Optional[str] = None, **kwargs) -> str:
        """
        Generate code.

        Args:
            type_name: Type of code to generate
            name: Name for the generated code
            output_path: Optional output file path
            **kwargs: Additional parameters

        Returns:
            Generated code
        """
        if type_name not in self.templates:
            raise ValueError(f"Unknown type: {type_name}. Available: {list(self.templates.keys())}")

        # Generate code
        template_func = self.templates[type_name]
        code = template_func(name, **kwargs)

        # Write to file if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)

            print(f"✓ Generated {type_name}: {output_path}")

        return code


def main():
    parser = argparse.ArgumentParser(
        description="GreenLang Code Generator - Generate infrastructure boilerplate code"
    )

    parser.add_argument(
        '--type',
        required=True,
        choices=['agent', 'pipeline', 'llm-session', 'cache', 'validation', 'config'],
        help='Type of code to generate'
    )

    parser.add_argument(
        '--name',
        required=True,
        help='Name for the generated code'
    )

    parser.add_argument(
        '--output',
        help='Output file path'
    )

    parser.add_argument(
        '--agents',
        help='Comma-separated list of agent names (for pipeline)'
    )

    parser.add_argument(
        '--provider',
        default='openai',
        help='LLM provider (for llm-session)'
    )

    parser.add_argument(
        '--model',
        default='gpt-4',
        help='LLM model (for llm-session)'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Include batch processing (for agent)'
    )

    parser.add_argument(
        '--description',
        help='Description for the generated code'
    )

    args = parser.parse_args()

    # Prepare kwargs
    kwargs = {}

    if args.agents:
        kwargs['agents'] = [a.strip() for a in args.agents.split(',')]

    if args.provider:
        kwargs['provider'] = args.provider

    if args.model:
        kwargs['model'] = args.model

    if args.batch:
        kwargs['batch'] = True

    if args.description:
        kwargs['description'] = args.description

    # Generate code
    generator = CodeGenerator()

    try:
        code = generator.generate(
            args.type,
            args.name,
            output_path=args.output,
            **kwargs
        )

        if not args.output:
            print("\n" + "=" * 80)
            print(f"Generated {args.type}: {args.name}")
            print("=" * 80)
            print(code)

    except Exception as e:
        logger.error(f"{e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
