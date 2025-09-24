#!/usr/bin/env python3
"""
Example usage of GreenLang OS-level sandbox

This script demonstrates various sandbox configurations and usage patterns.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add the parent directory to sys.path to import greenlang modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from greenlang.sandbox.os_sandbox import (
        execute_sandboxed,
        create_default_config,
        create_secure_config,
        OSSandbox,
        OSSandboxConfig,
        IsolationType,
        SandboxMode,
        ResourceLimits,
        NetworkConfig,
        FilesystemConfig
    )
    from greenlang.runtime.guard import (
        RuntimeGuard,
        get_sandbox_info,
        is_os_sandbox_available
    )

    print("✓ Successfully imported GreenLang sandbox modules")
except ImportError as e:
    print(f"✗ Failed to import GreenLang modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_basic_execution():
    """Test basic function execution in sandbox"""
    print("\n=== Testing Basic Execution ===")

    def simple_computation(x, y):
        """Simple function to test sandbox execution"""
        return x * y + (x - y)

    try:
        config = create_default_config()
        result = execute_sandboxed(simple_computation, config, 10, 5)
        print(f"✓ Basic execution successful: 10 * 5 + (10 - 5) = {result}")
        assert result == 55, f"Expected 55, got {result}"
    except Exception as e:
        print(f"✗ Basic execution failed: {e}")


def test_filesystem_operations():
    """Test filesystem operations in sandbox"""
    print("\n=== Testing Filesystem Operations ===")

    def file_operations():
        """Test file read/write operations"""
        import tempfile
        import os

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, prefix='sandbox_test_') as f:
            f.write("Hello from sandbox!")
            temp_path = f.name

        # Read the file back
        with open(temp_path, 'r') as f:
            content = f.read()

        # Clean up
        os.unlink(temp_path)

        return content

    try:
        config = create_default_config()
        config.filesystem.read_write_paths.append("/tmp")

        result = execute_sandboxed(file_operations, config)
        print(f"✓ File operations successful: '{result}'")
        assert result == "Hello from sandbox!", f"Expected 'Hello from sandbox!', got '{result}'"
    except Exception as e:
        print(f"✗ File operations failed: {e}")


def test_network_restrictions():
    """Test network access restrictions"""
    print("\n=== Testing Network Restrictions ===")

    def network_operation():
        """Test network operation that should be blocked"""
        import socket
        try:
            # Try to connect to a blocked metadata endpoint
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect(("169.254.169.254", 80))
            sock.close()
            return "Network access allowed (unexpected)"
        except Exception as e:
            return f"Network access blocked: {type(e).__name__}"

    try:
        config = create_default_config()
        config.network.allow_network = False

        result = execute_sandboxed(network_operation, config)
        print(f"✓ Network restriction test: {result}")
        assert "blocked" in result.lower(), f"Expected network to be blocked, got: {result}"
    except Exception as e:
        print(f"✗ Network restriction test failed: {e}")


def test_resource_limits():
    """Test resource limit enforcement"""
    print("\n=== Testing Resource Limits ===")

    def memory_intensive_operation():
        """Test memory limit enforcement"""
        try:
            # Try to allocate more memory than limit
            big_list = []
            for i in range(1000000):  # Try to allocate ~100MB+ of memory
                big_list.append("x" * 1000)
            return f"Allocated {len(big_list)} items"
        except MemoryError:
            return "Memory limit enforced"
        except Exception as e:
            return f"Other error: {type(e).__name__}"

    try:
        config = create_default_config()
        config.limits.memory_limit_bytes = 50 * 1024 * 1024  # 50MB limit

        result = execute_sandboxed(memory_intensive_operation, config)
        print(f"✓ Resource limit test: {result}")
    except Exception as e:
        print(f"✗ Resource limit test failed: {e}")


def test_different_isolation_types():
    """Test different isolation types"""
    print("\n=== Testing Different Isolation Types ===")

    def test_function():
        """Simple test function"""
        import os
        return f"PID: {os.getpid()}, Platform: {sys.platform}"

    isolation_types = [
        (IsolationType.BASIC, "Basic"),
        (IsolationType.NAMESPACE, "Namespace"),
        (IsolationType.CONTAINER, "Container"),
        (IsolationType.GVISOR, "gVisor")
    ]

    for isolation_type, name in isolation_types:
        try:
            config = OSSandboxConfig(
                isolation_type=isolation_type,
                sandbox_mode=SandboxMode.PERMISSIVE,
                limits=ResourceLimits(
                    memory_limit_bytes=128 * 1024 * 1024,
                    cpu_time_limit_seconds=30
                ),
                execution_timeout=60,
                fallback_to_basic=True
            )

            result = execute_sandboxed(test_function, config)
            print(f"✓ {name} isolation: {result}")
        except Exception as e:
            print(f"✗ {name} isolation failed: {e}")


def test_runtime_guard_integration():
    """Test integration with runtime guard"""
    print("\n=== Testing Runtime Guard Integration ===")

    try:
        # Create guard with test capabilities
        capabilities = {
            "fs": {
                "allow": True,
                "read": {"allowlist": ["/tmp/**"]},
                "write": {"allowlist": ["/tmp/**"]}
            },
            "net": {
                "allow": False
            }
        }

        guard = RuntimeGuard(capabilities, enable_os_sandbox=True)

        def test_guard_function():
            """Test function that uses guarded resources"""
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w') as f:
                f.write("Guard test")
                f.flush()
                with open(f.name, 'r') as rf:
                    return rf.read()

        result = guard.execute_in_sandbox(test_guard_function)
        print(f"✓ Runtime guard integration: '{result}'")

        # Print sandbox info
        info = get_sandbox_info()
        print(f"✓ Sandbox info: {info}")

    except Exception as e:
        print(f"✗ Runtime guard integration failed: {e}")


def test_error_handling():
    """Test error handling and fallback mechanisms"""
    print("\n=== Testing Error Handling ===")

    def failing_function():
        """Function that raises an exception"""
        raise ValueError("Test exception from sandbox")

    try:
        config = create_default_config()
        result = execute_sandboxed(failing_function, config)
        print(f"✗ Expected exception but got result: {result}")
    except ValueError as e:
        print(f"✓ Exception properly propagated: {e}")
    except Exception as e:
        print(f"✗ Unexpected exception type: {type(e).__name__}: {e}")


def demonstrate_security_features():
    """Demonstrate various security features"""
    print("\n=== Security Feature Demonstration ===")

    def security_test():
        """Test various security restrictions"""
        results = []

        # Test file access restrictions
        try:
            with open("/etc/passwd", "r") as f:
                f.read(100)
            results.append("✗ /etc/passwd access allowed (security issue)")
        except Exception:
            results.append("✓ /etc/passwd access blocked")

        # Test subprocess restrictions
        try:
            import subprocess
            subprocess.run(["id"], capture_output=True, timeout=1)
            results.append("✗ subprocess execution allowed (security issue)")
        except Exception:
            results.append("✓ subprocess execution blocked")

        # Test network metadata access
        try:
            import socket
            sock = socket.socket()
            sock.settimeout(1)
            sock.connect(("169.254.169.254", 80))
            sock.close()
            results.append("✗ Metadata endpoint access allowed (security issue)")
        except Exception:
            results.append("✓ Metadata endpoint access blocked")

        return results

    try:
        config = create_secure_config()
        results = execute_sandboxed(security_test, config)

        for result in results:
            print(f"  {result}")

    except Exception as e:
        print(f"✗ Security test failed: {e}")


def benchmark_performance():
    """Benchmark sandbox performance"""
    print("\n=== Performance Benchmark ===")

    def compute_intensive_task():
        """CPU-intensive task for benchmarking"""
        total = 0
        for i in range(100000):
            total += i * i
        return total

    # Test different configurations
    configs = [
        ("No Sandbox", None),
        ("Basic Sandbox", create_default_config()),
        ("Secure Sandbox", create_secure_config())
    ]

    for name, config in configs:
        times = []
        for _ in range(3):  # Run 3 times for average
            start_time = time.time()

            if config is None:
                # Run without sandbox
                result = compute_intensive_task()
            else:
                result = execute_sandboxed(compute_intensive_task, config)

            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        print(f"  {name}: {avg_time:.3f}s average (result: {result})")


def main():
    """Run all tests and demonstrations"""
    print("GreenLang OS-Level Sandbox Example")
    print("=" * 50)

    # Check sandbox availability
    print(f"OS Sandbox Available: {is_os_sandbox_available()}")
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version}")

    # Run tests
    test_basic_execution()
    test_filesystem_operations()
    test_network_restrictions()
    test_resource_limits()
    test_different_isolation_types()
    test_runtime_guard_integration()
    test_error_handling()
    demonstrate_security_features()
    benchmark_performance()

    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nTo use in your own code:")
    print("  from greenlang.sandbox.os_sandbox import execute_sandboxed, create_default_config")
    print("  result = execute_sandboxed(my_function, create_default_config(), *args)")


if __name__ == "__main__":
    main()