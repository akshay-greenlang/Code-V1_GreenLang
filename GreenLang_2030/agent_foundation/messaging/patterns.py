"""
Messaging Coordination Patterns

Implements common agent coordination patterns:
- Request-Reply: Synchronous request-response
- Pub-Sub: Broadcast to multiple agents
- Work Queue: Distribute tasks among workers
- Event Sourcing: Log all agent actions
- Saga Pattern: Distributed transactions

Example:
    >>> pattern = RequestReplyPattern(broker)
    >>> response = await pattern.send_request("agent.llm", {"prompt": "..."})
"""

from typing import Dict, List, Optional, Any, Callable
import asyncio
import logging
from datetime import datetime
from enum import Enum

from .broker_interface import MessageBrokerInterface
from .message import Message, MessagePriority

logger = logging.getLogger(__name__)


class PatternType(str, Enum):
    """Messaging pattern types."""
    REQUEST_REPLY = "request_reply"
    PUB_SUB = "pub_sub"
    WORK_QUEUE = "work_queue"
    EVENT_SOURCING = "event_sourcing"
    SAGA = "saga"


class RequestReplyPattern:
    """
    Request-Reply pattern for synchronous communication.

    Agent A sends request → Agent B processes → sends reply.
    Used for LLM calls, database queries, calculations.

    Example:
        >>> pattern = RequestReplyPattern(broker)
        >>> response = await pattern.send_request(
        ...     "agent.llm",
        ...     {"prompt": "Analyze this ESG report"}
        ... )
        >>> print(response.payload["result"])
    """

    def __init__(self, broker: MessageBrokerInterface):
        """Initialize pattern with broker."""
        self.broker = broker

    async def send_request(
        self,
        topic: str,
        payload: Dict[str, Any],
        timeout: float = 30.0,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> Optional[Message]:
        """
        Send request and wait for reply.

        Args:
            topic: Request topic
            payload: Request data
            timeout: Response timeout in seconds
            priority: Message priority

        Returns:
            Response message or None if timeout
        """
        logger.debug(f"Sending request to {topic}")
        response = await self.broker.request(topic, payload, timeout)

        if response is None:
            logger.warning(f"Request to {topic} timed out after {timeout}s")

        return response

    async def handle_request(
        self,
        topic: str,
        handler: Callable[[Dict[str, Any]], Dict[str, Any]],
        consumer_group: str = "request_handlers",
    ) -> None:
        """
        Handle incoming requests and send replies.

        Args:
            topic: Request topic to handle
            handler: Request handler function (sync or async)
            consumer_group: Consumer group name
        """
        logger.info(f"Starting request handler for {topic}")

        async for message in self.broker.consume(topic, consumer_group):
            try:
                # Process request
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(message.payload)
                else:
                    result = handler(message.payload)

                # Send reply
                await self.broker.reply(message, result)
                await self.broker.acknowledge(message)

                logger.debug(f"Handled request {message.id}")

            except Exception as e:
                logger.error(f"Request handler error: {e}", exc_info=True)
                await self.broker.nack(message, str(e), requeue=False)


class PubSubPattern:
    """
    Publish-Subscribe pattern for broadcasting.

    One agent publishes → Multiple agents receive.
    Used for event notifications, status updates.

    Example:
        >>> pattern = PubSubPattern(broker)
        >>> await pattern.publish("agent.events", {"event": "calculation_complete"})
        >>> await pattern.subscribe("agent.events.*", event_handler)
    """

    def __init__(self, broker: MessageBrokerInterface):
        """Initialize pattern with broker."""
        self.broker = broker

    async def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> str:
        """
        Publish message to all subscribers.

        Args:
            topic: Publication topic
            payload: Event data
            priority: Message priority

        Returns:
            Message ID
        """
        return await self.broker.publish(topic, payload, priority)

    async def subscribe(
        self,
        pattern: str,
        handler: Callable[[Message], None],
    ) -> None:
        """
        Subscribe to topic pattern.

        Args:
            pattern: Topic pattern (supports wildcards like "agent.events.*")
            handler: Event handler function
        """
        await self.broker.subscribe(pattern, handler)


class WorkQueuePattern:
    """
    Work Queue pattern for load distribution.

    Multiple workers compete for tasks from queue.
    Used for parallel processing, batch jobs.

    Example:
        >>> pattern = WorkQueuePattern(broker)
        >>> await pattern.submit_task("agent.tasks", {"task": "analyze_esg"})
        >>> await pattern.process_tasks("agent.tasks", task_handler, workers=10)
    """

    def __init__(self, broker: MessageBrokerInterface):
        """Initialize pattern with broker."""
        self.broker = broker

    async def submit_task(
        self,
        queue: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> str:
        """
        Submit task to work queue.

        Args:
            queue: Queue name
            payload: Task data
            priority: Task priority

        Returns:
            Task ID
        """
        return await self.broker.publish(queue, payload, priority)

    async def submit_batch(
        self,
        queue: str,
        payloads: List[Dict[str, Any]],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> List[str]:
        """
        Submit batch of tasks (80% overhead reduction).

        Args:
            queue: Queue name
            payloads: List of task data
            priority: Task priority

        Returns:
            List of task IDs
        """
        return await self.broker.publish_batch(queue, payloads, priority)

    async def process_tasks(
        self,
        queue: str,
        handler: Callable[[Dict[str, Any]], Any],
        consumer_group: str = "workers",
        num_workers: int = 1,
    ) -> None:
        """
        Process tasks with multiple workers.

        Args:
            queue: Queue name
            handler: Task handler function
            consumer_group: Consumer group name
            num_workers: Number of parallel workers
        """
        async def worker(worker_id: int):
            """Worker coroutine."""
            logger.info(f"Worker {worker_id} started on {queue}")

            async for message in self.broker.consume(queue, consumer_group):
                try:
                    # Process task
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(message.payload)
                    else:
                        result = handler(message.payload)

                    await self.broker.acknowledge(message)
                    logger.debug(f"Worker {worker_id} completed task {message.id}")

                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
                    await self.broker.nack(message, str(e))

        # Start worker tasks
        tasks = [asyncio.create_task(worker(i)) for i in range(num_workers)]
        await asyncio.gather(*tasks)


class EventSourcingPattern:
    """
    Event Sourcing pattern for audit logging.

    All agent actions are logged as immutable events.
    Used for compliance, audit trails, replay.

    Example:
        >>> pattern = EventSourcingPattern(broker)
        >>> await pattern.log_event("calculation", {"input": {...}, "output": {...}})
        >>> events = await pattern.get_event_history("agent_123", limit=100)
    """

    def __init__(self, broker: MessageBrokerInterface):
        """Initialize pattern with broker."""
        self.broker = broker
        self.event_stream = "agent.events"

    async def log_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        agent_id: Optional[str] = None,
    ) -> str:
        """
        Log immutable event.

        Args:
            event_type: Event type (e.g., "calculation", "validation")
            event_data: Event payload
            agent_id: Agent identifier

        Returns:
            Event ID
        """
        payload = {
            "event_type": event_type,
            "event_data": event_data,
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return await self.broker.publish(
            self.event_stream,
            payload,
            priority=MessagePriority.NORMAL,
        )

    async def replay_events(
        self,
        handler: Callable[[Dict[str, Any]], None],
        from_timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Replay events for reconstruction.

        Args:
            handler: Event handler function
            from_timestamp: Replay from this timestamp
        """
        # Implementation depends on broker's ability to read from specific point
        # For Redis Streams, use XREAD with timestamp
        logger.info("Replaying events...")
        # TODO: Implement event replay logic


class SagaPattern:
    """
    Saga pattern for distributed transactions.

    Coordinates multi-step workflows with compensation.
    Each step can be rolled back if later steps fail.

    Example:
        >>> saga = SagaPattern(broker)
        >>> saga.add_step("validate_data", validate_handler, compensate_validate)
        >>> saga.add_step("calculate_emissions", calc_handler, compensate_calc)
        >>> await saga.execute({"data": {...}})
    """

    def __init__(self, broker: MessageBrokerInterface):
        """Initialize saga pattern."""
        self.broker = broker
        self.steps: List[Dict[str, Any]] = []

    def add_step(
        self,
        name: str,
        handler: Callable[[Dict[str, Any]], Dict[str, Any]],
        compensate: Callable[[Dict[str, Any]], None],
    ) -> None:
        """
        Add saga step with compensation.

        Args:
            name: Step name
            handler: Forward handler (execute step)
            compensate: Compensation handler (rollback step)
        """
        self.steps.append({
            "name": name,
            "handler": handler,
            "compensate": compensate,
        })

    async def execute(
        self,
        initial_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute saga with automatic compensation on failure.

        Args:
            initial_data: Initial saga data

        Returns:
            Final result if successful

        Raises:
            SagaError: If saga fails after compensation
        """
        completed_steps = []
        current_data = initial_data.copy()

        try:
            # Execute forward steps
            for step in self.steps:
                logger.info(f"Executing saga step: {step['name']}")

                if asyncio.iscoroutinefunction(step["handler"]):
                    result = await step["handler"](current_data)
                else:
                    result = step["handler"](current_data)

                current_data.update(result)
                completed_steps.append(step)

            logger.info("Saga completed successfully")
            return current_data

        except Exception as e:
            logger.error(f"Saga failed at step {step['name']}: {e}", exc_info=True)

            # Compensate in reverse order
            for step in reversed(completed_steps):
                try:
                    logger.info(f"Compensating step: {step['name']}")
                    if asyncio.iscoroutinefunction(step["compensate"]):
                        await step["compensate"](current_data)
                    else:
                        step["compensate"](current_data)
                except Exception as comp_error:
                    logger.error(f"Compensation failed: {comp_error}", exc_info=True)

            raise SagaError(f"Saga failed: {e}") from e


class SagaError(Exception):
    """Exception raised when saga execution fails."""
    pass


class CircuitBreakerPattern:
    """
    Circuit Breaker pattern for fault tolerance.

    Prevents cascading failures by temporarily blocking requests
    to failing services.

    States:
        - CLOSED: Normal operation (requests pass through)
        - OPEN: Service failing (requests blocked)
        - HALF_OPEN: Testing if service recovered

    Example:
        >>> breaker = CircuitBreakerPattern(
        ...     failure_threshold=5,
        ...     timeout_seconds=60
        ... )
        >>> result = await breaker.call(risky_operation, arg1, arg2)
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_max_calls: int = 3,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            timeout_seconds: Seconds before attempting recovery
            half_open_max_calls: Max calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.half_open_calls = 0

    async def call(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function through circuit breaker.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        # Check circuit state
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.half_open_calls = 0
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerError("Circuit breaker is OPEN")

        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Success - handle state transitions
            if self.state == "HALF_OPEN":
                self.half_open_calls += 1
                if self.half_open_calls >= self.half_open_max_calls:
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker CLOSED (recovered)")

            return result

        except Exception as e:
            # Failure - record and potentially open circuit
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")

            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True

        elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout_seconds

    def reset(self) -> None:
        """Manually reset circuit breaker."""
        self.state = "CLOSED"
        self.failure_count = 0
        self.half_open_calls = 0
        logger.info("Circuit breaker manually reset")


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass
