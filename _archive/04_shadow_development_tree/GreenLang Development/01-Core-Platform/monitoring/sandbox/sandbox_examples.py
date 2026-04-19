#!/usr/bin/env python3
"""
Sandbox Usage Examples for GreenLang

Demonstrates various sandbox configurations and usage patterns.
"""

import os
import sys
import time
import logging
from typing import Any, Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from greenlang.sandbox.os_sandbox import (
    OSSandbox,
    OSSandboxConfig,
    SecurityLevel,
    IsolationType,
    SandboxMode,
    ResourceLimits,
    NetworkConfig,
    FilesystemConfig,
    WhitelistConfig,
    execute_sandboxed,
    execute_strict,
    execute_moderate,
    execute_lenient,
    create_config_by_level,
    SandboxTimeoutError,
    SandboxExecutionError,
    SandboxViolationError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_basic_execution():
    """Example 1: Basic sandboxed execution with default settings"""
    print("\n" + "=" * 60)
    print("Example 1: Basic Sandboxed Execution")
    print("=" * 60)

    def calculate_fibonacci(n: int) -> int:
        """Calculate nth Fibonacci number"""
        if n <= 1:
            return n
        return calculate_fibonacci(n - 1) + calculate_fibonacci(n - 2)

    # Execute with moderate security (default)
    try:
        result = execute_moderate(calculate_fibonacci, 10)
        print(f"Fibonacci(10) = {result}")
        print("✓ Execution successful with moderate security")
    except Exception as e:
        print(f"✗ Execution failed: {e}")


def example_2_strict_security():
    """Example 2: Strict security with minimal permissions"""
    print("\n" + "=" * 60)
    print("Example 2: Strict Security Sandbox")
    print("=" * 60)

    def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data with calculations only"""
        result = {
            "sum": sum(data.get("numbers", [])),
            "count": len(data.get("numbers", [])),
            "processed_at": time.time()
        }
        return result

    # Execute with strict security
    test_data = {"numbers": [1, 2, 3, 4, 5]}

    try:
        result = execute_strict(process_data, test_data)
        print(f"Input: {test_data}")
        print(f"Result: {result}")
        print("✓ Strict execution successful")
    except Exception as e:
        print(f"✗ Strict execution failed: {e}")


def example_3_resource_limits():
    """Example 3: Custom resource limits"""
    print("\n" + "=" * 60)
    print("Example 3: Resource-Limited Sandbox")
    print("=" * 60)

    def memory_intensive_task():
        """Task that uses significant memory"""
        # Try to allocate large list
        data = []
        for i in range(1000000):
            data.append(i * 2)
        return len(data)

    # Create config with strict memory limits
    config = OSSandboxConfig(
        security_level=SecurityLevel.MODERATE,
        isolation_type=IsolationType.BASIC,  # Use basic for cross-platform
        limits=ResourceLimits(
            memory_limit_bytes=100 * 1024 * 1024,  # 100MB limit
            cpu_time_limit_seconds=5,
            max_processes=4
        ),
        execution_timeout=10
    )

    try:
        with OSSandbox(config) as sandbox:
            result = sandbox.execute(memory_intensive_task)
            print(f"Processed {result} items within memory limits")
            print("✓ Resource-limited execution successful")
    except Exception as e:
        print(f"✗ Resource-limited execution failed: {e}")


def example_4_timeout_enforcement():
    """Example 4: Timeout enforcement"""
    print("\n" + "=" * 60)
    print("Example 4: Timeout Enforcement")
    print("=" * 60)

    def long_running_task():
        """Task that takes too long"""
        time.sleep(10)  # Sleep for 10 seconds
        return "Completed"

    # Create config with 2-second timeout
    config = OSSandboxConfig(
        security_level=SecurityLevel.MODERATE,
        isolation_type=IsolationType.BASIC,
        execution_timeout=2  # 2-second timeout
    )

    try:
        with OSSandbox(config) as sandbox:
            result = sandbox.execute(long_running_task)
            print(f"Result: {result}")
    except SandboxTimeoutError as e:
        print(f"✓ Timeout correctly enforced: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


def example_5_network_isolation():
    """Example 5: Network isolation"""
    print("\n" + "=" * 60)
    print("Example 5: Network Isolation")
    print("=" * 60)

    def network_task():
        """Task that attempts network access"""
        import socket
        try:
            # Try to connect to Google DNS
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect(("8.8.8.8", 53))
            s.close()
            return "Network access successful"
        except Exception as e:
            return f"Network blocked: {e}"

    # Config with no network access
    config_no_network = OSSandboxConfig(
        security_level=SecurityLevel.MODERATE,
        isolation_type=IsolationType.BASIC,
        network=NetworkConfig(
            allow_network=False,
            allow_loopback=False
        )
    )

    # Config with network access
    config_with_network = OSSandboxConfig(
        security_level=SecurityLevel.LENIENT,
        isolation_type=IsolationType.BASIC,
        network=NetworkConfig(
            allow_network=True,
            allow_loopback=True
        )
    )

    print("Testing without network access:")
    try:
        with OSSandbox(config_no_network) as sandbox:
            result = sandbox.execute(network_task)
            print(f"  Result: {result}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\nTesting with network access:")
    try:
        with OSSandbox(config_with_network) as sandbox:
            result = sandbox.execute(network_task)
            print(f"  Result: {result}")
    except Exception as e:
        print(f"  Error: {e}")


def example_6_filesystem_isolation():
    """Example 6: Filesystem isolation"""
    print("\n" + "=" * 60)
    print("Example 6: Filesystem Isolation")
    print("=" * 60)

    def file_task():
        """Task that attempts file access"""
        results = []

        # Try to read /etc/passwd (should be blocked)
        try:
            with open("/etc/passwd", "r") as f:
                results.append("Read /etc/passwd: SUCCESS (shouldn't happen!)")
        except Exception as e:
            results.append(f"Read /etc/passwd: BLOCKED ({type(e).__name__})")

        # Try to write to /tmp (might be allowed)
        try:
            temp_file = "/tmp/sandbox_test.txt"
            with open(temp_file, "w") as f:
                f.write("test")
            results.append("Write to /tmp: SUCCESS")
            os.remove(temp_file)
        except Exception as e:
            results.append(f"Write to /tmp: BLOCKED ({type(e).__name__})")

        return results

    config = OSSandboxConfig(
        security_level=SecurityLevel.MODERATE,
        isolation_type=IsolationType.BASIC,
        filesystem=FilesystemConfig(
            blocked_paths=["/etc", "/root", "/home"],
            read_write_paths=["/tmp"]
        )
    )

    try:
        with OSSandbox(config) as sandbox:
            results = sandbox.execute(file_task)
            for result in results:
                print(f"  {result}")
    except Exception as e:
        print(f"✗ Filesystem test failed: {e}")


def example_7_docker_sandbox():
    """Example 7: Docker-based sandbox (if available)"""
    print("\n" + "=" * 60)
    print("Example 7: Docker-Based Sandbox")
    print("=" * 60)

    def docker_task():
        """Task to run in Docker container"""
        import platform
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "message": "Hello from Docker sandbox!"
        }

    config = OSSandboxConfig(
        security_level=SecurityLevel.STRICT,
        isolation_type=IsolationType.CONTAINER,
        container_image="python:3.9-alpine",
        container_runtime="docker",
        network=NetworkConfig(allow_network=False),
        limits=ResourceLimits(
            memory_limit_bytes=256 * 1024 * 1024,  # 256MB
            cpu_percent_limit=50  # 50% of one CPU
        ),
        fallback_to_basic=True  # Fall back if Docker not available
    )

    try:
        with OSSandbox(config) as sandbox:
            result = sandbox.execute(docker_task)
            print(f"Docker execution result: {result}")

            # Check which isolation was actually used
            if config.isolation_type == IsolationType.CONTAINER:
                print("✓ Successfully used Docker isolation")
            else:
                print(f"✓ Fell back to {config.isolation_type.value} isolation")
    except Exception as e:
        print(f"✗ Docker sandbox failed: {e}")


def example_8_audit_logging():
    """Example 8: Audit logging and metrics"""
    print("\n" + "=" * 60)
    print("Example 8: Audit Logging and Metrics")
    print("=" * 60)

    def audited_task(x: int, y: int) -> int:
        """Task that will be audited"""
        result = x * y
        # Simulate some work
        time.sleep(0.1)
        return result

    # Create config with audit logging
    audit_file = "/tmp/sandbox_audit.log"
    config = OSSandboxConfig(
        security_level=SecurityLevel.MODERATE,
        isolation_type=IsolationType.BASIC,
        enable_audit=True,
        audit_log_path=audit_file,
        enable_metrics=True,
        detect_escape_attempts=True
    )

    try:
        with OSSandbox(config) as sandbox:
            result = sandbox.execute(audited_task, 5, 7)
            print(f"Task result: {result}")

            # Display metrics
            metrics = sandbox.metrics
            print("\nExecution Metrics:")
            print(f"  Setup time: {metrics.setup_time:.3f}s")
            print(f"  Execution time: {metrics.execution_time:.3f}s")
            print(f"  Cleanup time: {metrics.cleanup_time:.3f}s")
            print(f"  Violations detected: {metrics.violations_detected}")
            print(f"  Escape attempts: {metrics.escape_attempts}")

            # Display audit log
            print("\nAudit Log Events:")
            for event in sandbox.audit_log:
                print(f"  [{event['timestamp']}] {event['event_type']}: "
                      f"{event.get('execution_time', 'N/A')}s")

            print("✓ Audit logging successful")
    except Exception as e:
        print(f"✗ Audit logging failed: {e}")


def example_9_whitelist_configuration():
    """Example 9: Whitelist configuration"""
    print("\n" + "=" * 60)
    print("Example 9: Whitelist Configuration")
    print("=" * 60)

    def restricted_task():
        """Task that uses various modules"""
        results = []

        # Try allowed modules
        try:
            import json
            results.append("json module: ALLOWED")
        except ImportError:
            results.append("json module: BLOCKED")

        # Try potentially dangerous module
        try:
            import subprocess
            results.append("subprocess module: ALLOWED (shouldn't be!)")
        except ImportError:
            results.append("subprocess module: BLOCKED")

        return results

    config = OSSandboxConfig(
        security_level=SecurityLevel.STRICT,
        isolation_type=IsolationType.BASIC,
        whitelist=WhitelistConfig(
            allowed_modules=["json", "math", "datetime"],
            allowed_file_extensions=[".txt", ".json"],
            allowed_network_protocols=["https"]
        )
    )

    try:
        with OSSandbox(config) as sandbox:
            results = sandbox.execute(restricted_task)
            for result in results:
                print(f"  {result}")
    except Exception as e:
        print(f"✗ Whitelist test failed: {e}")


def example_10_agent_sandbox():
    """Example 10: Sandbox for GreenLang agent execution"""
    print("\n" + "=" * 60)
    print("Example 10: GreenLang Agent Sandbox")
    print("=" * 60)

    class MockAgent:
        """Mock GreenLang agent for demonstration"""

        def __init__(self, name: str):
            self.name = name

        def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
            """Process data with zero-hallucination approach"""
            # Simulate agent processing
            result = {
                "agent": self.name,
                "input_count": len(data),
                "processed_at": time.time(),
                "calculations": {
                    "sum": sum(data.get("values", [])),
                    "avg": sum(data.get("values", [])) / len(data.get("values", [1]))
                }
            }
            return result

    # Create agent-specific sandbox config
    config = OSSandboxConfig(
        security_level=SecurityLevel.STRICT,
        isolation_type=IsolationType.BASIC,
        sandbox_mode=SandboxMode.ENFORCING,
        limits=ResourceLimits(
            memory_limit_bytes=512 * 1024 * 1024,  # 512MB for agent
            cpu_time_limit_seconds=30,
            max_processes=8
        ),
        network=NetworkConfig(
            allow_network=False,  # No network for calculations
            allow_loopback=True   # Allow local communication
        ),
        filesystem=FilesystemConfig(
            create_temp_root=True,
            blocked_paths=["/etc", "/root", "/home", "/var"],
            read_write_paths=["/tmp"]
        ),
        execution_timeout=30,
        enable_audit=True,
        detect_escape_attempts=True
    )

    # Execute agent in sandbox
    agent = MockAgent("TestAgent")
    test_data = {"values": [10, 20, 30, 40, 50]}

    try:
        with OSSandbox(config) as sandbox:
            # Execute agent's process method
            result = sandbox.execute(agent.process, test_data)

            print(f"Agent: {result['agent']}")
            print(f"Input count: {result['input_count']}")
            print(f"Calculations: {result['calculations']}")
            print("✓ Agent executed successfully in sandbox")

            # Show security metrics
            print("\nSecurity Metrics:")
            print(f"  Escape attempts: {sandbox.metrics.escape_attempts}")
            print(f"  Violations: {sandbox.metrics.violations_detected}")
            print(f"  Execution time: {sandbox.metrics.execution_time:.3f}s")

    except Exception as e:
        print(f"✗ Agent execution failed: {e}")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("GreenLang Sandbox Examples")
    print("=" * 60)

    examples = [
        example_1_basic_execution,
        example_2_strict_security,
        example_3_resource_limits,
        example_4_timeout_enforcement,
        example_5_network_isolation,
        example_6_filesystem_isolation,
        example_7_docker_sandbox,
        example_8_audit_logging,
        example_9_whitelist_configuration,
        example_10_agent_sandbox
    ]

    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"\nExample {i} encountered an error: {e}")

    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()