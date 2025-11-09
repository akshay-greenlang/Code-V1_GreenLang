"""
Mock Objects
============

Mock implementations of GreenLang infrastructure components.

This module provides mock objects for testing without actual infrastructure:
- MockChatSession: Mock LLM chat sessions
- MockCacheManager: Mock cache operations
- MockDatabaseManager: Mock database operations
- MockValidationFramework: Mock validation
- MockTelemetryManager: Mock telemetry
"""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
import time
import json


class MockChatSession:
    """
    Mock implementation of ChatSession for testing LLM interactions.

    Example:
    --------
    ```python
    mock_chat = MockChatSession()
    mock_chat.add_response("Hello, how can I help?")
    response = mock_chat.send_message("Hi")
    # response == "Hello, how can I help?"
    ```
    """

    def __init__(self):
        self.messages = []
        self.responses = []
        self.response_index = 0
        self.call_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0

    def send_message(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Send a message and get a response.

        Args:
            message: User message
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Mock response
        """
        self.call_count += 1

        # Record the message
        self.messages.append({
            'message': message,
            'system_prompt': system_prompt,
            'kwargs': kwargs,
            'timestamp': time.time(),
        })

        # Get response
        if self.response_index < len(self.responses):
            response = self.responses[self.response_index]
            self.response_index += 1

            # Track tokens and cost
            tokens = response.get('tokens', len(response['text'].split()) * 1.3)
            cost = response.get('cost', tokens * 0.00001)

            self.total_tokens += tokens
            self.total_cost += cost

            return response['text']
        else:
            # Default response
            return "Mock response"

    def stream_message(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Stream a message response.

        Args:
            message: User message
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Yields:
            Response chunks
        """
        response = self.send_message(message, system_prompt, **kwargs)

        # Split into chunks
        words = response.split()
        for word in words:
            yield word + " "
            time.sleep(0.01)  # Simulate streaming delay

    def add_response(
        self,
        text: str,
        tokens: Optional[int] = None,
        cost: Optional[float] = None
    ):
        """Add a mock response to the queue."""
        self.responses.append({
            'text': text,
            'tokens': tokens,
            'cost': cost,
        })

    def reset(self):
        """Reset the mock session."""
        self.messages.clear()
        self.responses.clear()
        self.response_index = 0
        self.call_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0


class MockCacheManager:
    """
    Mock implementation of CacheManager for testing cache operations.

    Example:
    --------
    ```python
    cache = MockCacheManager()
    cache.set("key", "value", ttl=60)
    value = cache.get("key")  # "value"
    ```
    """

    def __init__(self):
        self._cache = {}
        self._ttls = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        # Check if key exists and not expired
        if key in self._cache:
            if key in self._ttls:
                if time.time() > self._ttls[key]:
                    # Expired
                    del self._cache[key]
                    del self._ttls[key]
                    self.misses += 1
                    return None

            self.hits += 1
            return self._cache[key]
        else:
            self.misses += 1
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        self._cache[key] = value

        if ttl is not None:
            self._ttls[key] = time.time() + ttl

    def delete(self, key: str):
        """Delete a key from cache."""
        self._cache.pop(key, None)
        self._ttls.pop(key, None)

    def clear(self):
        """Clear all cache."""
        self._cache.clear()
        self._ttls.clear()

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return self.get(key) is not None

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / max(total, 1),
            'size': len(self._cache),
        }


class MockDatabaseManager:
    """
    Mock implementation of DatabaseManager for testing database operations.

    Example:
    --------
    ```python
    db = MockDatabaseManager()
    db.insert("users", {"name": "John", "email": "john@example.com"})
    users = db.query("SELECT * FROM users")
    ```
    """

    def __init__(self):
        self._tables = {}
        self._transactions = []
        self._in_transaction = False
        self._transaction_data = None

    def query(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a query.

        Args:
            sql: SQL query
            params: Query parameters

        Returns:
            List of result rows
        """
        # Very basic mock implementation
        # In a real implementation, this would parse SQL
        sql_lower = sql.lower()

        if 'select' in sql_lower:
            # Extract table name (very basic)
            if 'from' in sql_lower:
                parts = sql_lower.split('from')[1].strip().split()
                table_name = parts[0]

                if table_name in self._tables:
                    return list(self._tables[table_name])

        return []

    def insert(
        self,
        table: str,
        data: Dict[str, Any]
    ) -> int:
        """
        Insert a record.

        Args:
            table: Table name
            data: Data to insert

        Returns:
            Inserted record ID
        """
        if table not in self._tables:
            self._tables[table] = []

        # Add auto-incrementing ID
        record_id = len(self._tables[table]) + 1
        record = {'id': record_id, **data}

        if self._in_transaction:
            # Add to transaction buffer
            if self._transaction_data is None:
                self._transaction_data = {}
            if table not in self._transaction_data:
                self._transaction_data[table] = []
            self._transaction_data[table].append(record)
        else:
            self._tables[table].append(record)

        return record_id

    def update(
        self,
        table: str,
        data: Dict[str, Any],
        where: Dict[str, Any]
    ) -> int:
        """
        Update records.

        Args:
            table: Table name
            data: Data to update
            where: WHERE clause conditions

        Returns:
            Number of updated records
        """
        if table not in self._tables:
            return 0

        updated_count = 0

        for record in self._tables[table]:
            # Check if record matches where clause
            matches = all(
                record.get(k) == v
                for k, v in where.items()
            )

            if matches:
                record.update(data)
                updated_count += 1

        return updated_count

    def delete(
        self,
        table: str,
        where: Dict[str, Any]
    ) -> int:
        """
        Delete records.

        Args:
            table: Table name
            where: WHERE clause conditions

        Returns:
            Number of deleted records
        """
        if table not in self._tables:
            return 0

        original_count = len(self._tables[table])

        self._tables[table] = [
            record for record in self._tables[table]
            if not all(record.get(k) == v for k, v in where.items())
        ]

        return original_count - len(self._tables[table])

    def begin_transaction(self):
        """Begin a transaction."""
        self._in_transaction = True
        self._transaction_data = {}

    def commit(self):
        """Commit the transaction."""
        if self._in_transaction and self._transaction_data:
            # Apply transaction data
            for table, records in self._transaction_data.items():
                if table not in self._tables:
                    self._tables[table] = []
                self._tables[table].extend(records)

        self._in_transaction = False
        self._transaction_data = None

    def rollback(self):
        """Rollback the transaction."""
        self._in_transaction = False
        self._transaction_data = None

    def reset(self):
        """Reset the database."""
        self._tables.clear()
        self._transactions.clear()
        self._in_transaction = False
        self._transaction_data = None


class MockValidationFramework:
    """
    Mock implementation of ValidationFramework.

    Example:
    --------
    ```python
    validator = MockValidationFramework()
    result = validator.validate(data, schema)
    ```
    """

    def __init__(self):
        self.validations = []

    def validate(
        self,
        data: Any,
        schema: Any
    ) -> Dict[str, Any]:
        """
        Validate data against schema.

        Args:
            data: Data to validate
            schema: Validation schema

        Returns:
            Validation result
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
        }

        self.validations.append({
            'data': data,
            'schema': schema,
            'result': result,
            'timestamp': time.time(),
        })

        return result

    def reset(self):
        """Reset validation history."""
        self.validations.clear()


class MockTelemetryManager:
    """
    Mock implementation of TelemetryManager.

    Example:
    --------
    ```python
    telemetry = MockTelemetryManager()
    telemetry.track_event("user_login", {"user_id": "123"})
    ```
    """

    def __init__(self):
        self.events = []
        self.metrics = []
        self.logs = []

    def track_event(
        self,
        event_name: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """Track an event."""
        self.events.append({
            'name': event_name,
            'properties': properties or {},
            'timestamp': time.time(),
        })

    def track_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """Track a metric."""
        self.metrics.append({
            'name': metric_name,
            'value': value,
            'tags': tags or {},
            'timestamp': time.time(),
        })

    def log(
        self,
        level: str,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log a message."""
        self.logs.append({
            'level': level,
            'message': message,
            'context': context or {},
            'timestamp': time.time(),
        })

    def reset(self):
        """Reset telemetry data."""
        self.events.clear()
        self.metrics.clear()
        self.logs.clear()

    def get_event_count(self, event_name: str) -> int:
        """Get count of specific event."""
        return len([e for e in self.events if e['name'] == event_name])

    def get_metric_values(self, metric_name: str) -> List[float]:
        """Get all values for a specific metric."""
        return [m['value'] for m in self.metrics if m['name'] == metric_name]
