"""
Runtime Guard - Enforces capability-based security at runtime

This module is loaded in a separate worker process to enforce deny-by-default
security policies for network, filesystem, subprocess, and clock access.
"""

import os
import sys
import json
import socket
import subprocess
import time
import datetime
import pathlib
import builtins
import logging
import tempfile
from typing import Any, Dict, List, Optional
from pathlib import Path
from functools import wraps
from urllib.parse import urlparse
import fnmatch

logger = logging.getLogger(__name__)


class CapabilityViolation(Exception):
    """Raised when a capability-controlled operation is attempted without permission"""

    def __init__(
        self, capability: str, operation: str, details: str = "", suggestion: str = ""
    ):
        self.capability = capability
        self.operation = operation
        self.details = details
        self.suggestion = suggestion

        message = f"CapabilityViolation: {capability} access denied\n"
        message += f"  Operation: {operation}\n"
        if details:
            message += f"  Details: {details}\n"
        message += f"\n  Capability required: {capability}\n"

        if suggestion:
            message += f"\n  To fix:\n{suggestion}"
        else:
            message += f"\n  To fix:\n"
            message += f"  1. Add to manifest.yaml:\n"
            message += f"     capabilities:\n"
            message += f"       {capability}:\n"
            message += f"         allow: true\n"
            message += f"  2. Request approval from security team"

        super().__init__(message)


class RuntimeGuard:
    """
    Runtime security guard that enforces capability-based access control
    """

    def __init__(self, capabilities: Optional[Dict[str, Any]] = None):
        """
        Initialize the runtime guard with specified capabilities

        Args:
            capabilities: Dict of capability configurations from manifest
        """
        self.capabilities = capabilities or {}
        self.audit_log = []
        # Capability override removed for production security
        self.override_mode = False

        # Parse environment variables for paths
        temp_dir = Path(tempfile.gettempdir())
        self.input_dir = Path(os.environ.get("GL_INPUT_DIR", temp_dir / "gl_input"))
        self.pack_data_dir = Path(
            os.environ.get("GL_PACK_DATA_DIR", temp_dir / "gl_pack_data")
        )
        self.run_tmp = Path(os.environ.get("GL_RUN_TMP", temp_dir / "gl_run"))

        # Resolve paths to absolute
        self.input_dir = self.input_dir.resolve()
        self.pack_data_dir = self.pack_data_dir.resolve()
        self.run_tmp = self.run_tmp.resolve()

        # Metadata endpoints to block
        self.blocked_metadata_ips = {
            "169.254.169.254",  # AWS
            "169.254.170.2",  # AWS ECS
            "100.100.100.200",  # Alibaba Cloud
            "169.254.169.253",  # OpenStack
        }

        # RFC1918 private ranges
        self.private_ranges = [
            ("10.0.0.0", "10.255.255.255"),
            ("172.16.0.0", "172.31.255.255"),
            ("192.168.0.0", "192.168.255.255"),
        ]

        # Override mode has been removed for production security
        # All capabilities must be explicitly declared in manifests

        # Store original functions before patching
        self._store_originals()

        # Apply security patches
        self._patch_all()

    def _store_originals(self):
        """Store original functions before patching"""
        self.originals = {
            "socket": socket.socket,
            "open": builtins.open,
            "subprocess_popen": subprocess.Popen,
            "subprocess_run": subprocess.run,
            "os_system": os.system,
            "time_time": time.time,
            "datetime_now": datetime.datetime.now,
            "pathlib_open": pathlib.Path.open,
            "os_open": os.open,
            "os_remove": os.remove,
            "os_rename": os.rename,
        }

    def _patch_all(self):
        """Apply all security patches"""
        self._patch_network()
        self._patch_filesystem()
        self._patch_subprocess()
        self._patch_clock()

    def _audit_event(
        self,
        capability: str,
        operation: str,
        target: str,
        result: str,
        details: Optional[Dict] = None,
    ):
        """Record audit event"""
        event = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "capability": capability,
            "operation": operation,
            "target": target,
            "result": result,
            "details": details or {},
            "pack": os.environ.get("GL_PACK_NAME", "unknown"),
            "run_id": os.environ.get("GL_RUN_ID", "unknown"),
        }
        self.audit_log.append(event)
        logger.info(f"Capability audit: {capability}/{operation} -> {result}")

    def _check_capability(self, capability: str) -> bool:
        """Check if a capability is allowed"""
        # No override mode - capabilities must be explicitly declared

        cap_config = self.capabilities.get(capability, {})
        return cap_config.get("allow", False)

    def _patch_network(self):
        """Patch network-related functions"""

        # Check if network is allowed
        net_allowed = self._check_capability("net")
        net_config = self.capabilities.get("net", {})
        allowed_domains = []

        if net_allowed and net_config.get("outbound"):
            allowed_domains = net_config["outbound"].get("allowlist", [])

        # Patch socket.socket
        original_socket = self.originals["socket"]

        def guarded_socket(*args, **kwargs):
            if not net_allowed:
                self._audit_event("net", "socket.create", "socket", "denied")
                raise CapabilityViolation("net", "Creating network socket")

            sock = original_socket(*args, **kwargs)

            # Wrap connect method
            original_connect = sock.connect

            def guarded_connect(address):
                host = address[0] if isinstance(address, tuple) else str(address)

                # Check metadata endpoints
                if host in self.blocked_metadata_ips:
                    self._audit_event(
                        "net",
                        "socket.connect",
                        host,
                        "denied",
                        {"reason": "metadata_endpoint"},
                    )
                    raise CapabilityViolation(
                        "net", f"Connecting to {host}", "Metadata endpoints are blocked"
                    )

                # Check against allowlist
                if allowed_domains and not self._match_domain(host, allowed_domains):
                    self._audit_event(
                        "net",
                        "socket.connect",
                        host,
                        "denied",
                        {"reason": "not_in_allowlist"},
                    )
                    raise CapabilityViolation(
                        "net",
                        f"Connecting to {host}",
                        f"Domain not in allowlist",
                        f"Add '{host}' to capabilities.net.outbound.allowlist",
                    )

                self._audit_event("net", "socket.connect", host, "allowed")
                return original_connect(address)

            sock.connect = guarded_connect
            return sock

        socket.socket = guarded_socket

        # Patch common HTTP libraries
        try:
            import http.client

            original_http_request = http.client.HTTPConnection.request

            @wraps(original_http_request)
            def guarded_http_request(self, *args, **kwargs):
                if not net_allowed:
                    self._audit_event("net", "http.request", str(self.host), "denied")
                    raise CapabilityViolation("net", "HTTP request")
                return original_http_request(self, *args, **kwargs)

            http.client.HTTPConnection.request = guarded_http_request
        except ImportError:
            pass

        try:
            import urllib.request

            original_urlopen = urllib.request.urlopen

            @wraps(original_urlopen)
            def guarded_urlopen(url, *args, **kwargs):
                if not net_allowed:
                    self._audit_event("net", "urllib.urlopen", str(url), "denied")
                    raise CapabilityViolation("net", f"Opening URL: {url}")

                # Check URL against allowlist
                parsed = urlparse(str(url))
                host = parsed.hostname or parsed.netloc

                if allowed_domains and not self._match_domain(host, allowed_domains):
                    self._audit_event("net", "urllib.urlopen", str(url), "denied")
                    raise CapabilityViolation(
                        "net", f"Opening URL: {url}", f"Domain {host} not in allowlist"
                    )

                self._audit_event("net", "urllib.urlopen", str(url), "allowed")
                return original_urlopen(url, *args, **kwargs)

            urllib.request.urlopen = guarded_urlopen
        except ImportError:
            pass

        try:
            import requests

            original_request = requests.Session.request

            @wraps(original_request)
            def guarded_request(self, method, url, *args, **kwargs):
                if not net_allowed:
                    self._audit_event("net", "requests.request", str(url), "denied")
                    raise CapabilityViolation("net", f"{method} request to {url}")

                # Check URL against allowlist
                parsed = urlparse(str(url))
                host = parsed.hostname or parsed.netloc

                if allowed_domains and not self._match_domain(host, allowed_domains):
                    self._audit_event("net", "requests.request", str(url), "denied")
                    raise CapabilityViolation(
                        "net",
                        f"{method} request to {url}",
                        f"Domain {host} not in allowlist",
                    )

                self._audit_event("net", "requests.request", str(url), "allowed")
                return original_request(self, method, url, *args, **kwargs)

            requests.Session.request = guarded_request
        except ImportError:
            pass

    def _patch_filesystem(self):
        """Patch filesystem-related functions"""

        fs_allowed = self._check_capability("fs")
        fs_config = self.capabilities.get("fs", {})

        read_allowlist = []
        write_allowlist = []

        if fs_allowed:
            if fs_config.get("read"):
                read_allowlist = fs_config["read"].get("allowlist", [])
            if fs_config.get("write"):
                write_allowlist = fs_config["write"].get("allowlist", [])

        # Patch builtins.open
        original_open = self.originals["open"]

        @wraps(original_open)
        def guarded_open(file, mode="r", *args, **kwargs):
            path = Path(file).resolve()

            # Determine if read or write
            is_write = any(m in mode for m in ["w", "a", "x", "+"])
            is_read = "r" in mode or (not is_write)

            if is_read:
                if not fs_allowed:
                    self._audit_event("fs", "open.read", str(path), "denied")
                    raise CapabilityViolation("fs", f"Reading file: {path}")

                if not self._check_path_allowed(path, read_allowlist, "read"):
                    self._audit_event("fs", "open.read", str(path), "denied")
                    raise CapabilityViolation(
                        "fs", f"Reading file: {path}", "Path not in read allowlist"
                    )

            if is_write:
                if not fs_allowed:
                    self._audit_event("fs", "open.write", str(path), "denied")
                    raise CapabilityViolation("fs", f"Writing file: {path}")

                if not self._check_path_allowed(path, write_allowlist, "write"):
                    self._audit_event("fs", "open.write", str(path), "denied")
                    raise CapabilityViolation(
                        "fs", f"Writing file: {path}", "Path not in write allowlist"
                    )

            self._audit_event(
                "fs", f'open.{"write" if is_write else "read"}', str(path), "allowed"
            )
            return original_open(file, mode, *args, **kwargs)

        builtins.open = guarded_open

        # Patch pathlib.Path.open
        original_pathlib_open = self.originals["pathlib_open"]

        @wraps(original_pathlib_open)
        def guarded_pathlib_open(self, mode="r", *args, **kwargs):
            return guarded_open(self, mode, *args, **kwargs)

        pathlib.Path.open = guarded_pathlib_open

        # Patch os.open
        original_os_open = self.originals["os_open"]

        @wraps(original_os_open)
        def guarded_os_open(path, flags, *args, **kwargs):
            resolved_path = Path(path).resolve()

            # Check based on flags
            is_write = flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT)

            if not fs_allowed:
                self._audit_event("fs", "os.open", str(resolved_path), "denied")
                raise CapabilityViolation("fs", f"Opening file: {resolved_path}")

            if is_write and not self._check_path_allowed(
                resolved_path, write_allowlist, "write"
            ):
                self._audit_event("fs", "os.open.write", str(resolved_path), "denied")
                raise CapabilityViolation(
                    "fs",
                    f"Writing file: {resolved_path}",
                    "Path not in write allowlist",
                )

            if not is_write and not self._check_path_allowed(
                resolved_path, read_allowlist, "read"
            ):
                self._audit_event("fs", "os.open.read", str(resolved_path), "denied")
                raise CapabilityViolation(
                    "fs", f"Reading file: {resolved_path}", "Path not in read allowlist"
                )

            self._audit_event("fs", "os.open", str(resolved_path), "allowed")
            return original_os_open(path, flags, *args, **kwargs)

        os.open = guarded_os_open

        # Patch os.remove
        original_remove = self.originals["os_remove"]

        @wraps(original_remove)
        def guarded_remove(path):
            resolved_path = Path(path).resolve()

            if not fs_allowed:
                self._audit_event("fs", "os.remove", str(resolved_path), "denied")
                raise CapabilityViolation("fs", f"Removing file: {resolved_path}")

            if not self._check_path_allowed(resolved_path, write_allowlist, "write"):
                self._audit_event("fs", "os.remove", str(resolved_path), "denied")
                raise CapabilityViolation(
                    "fs",
                    f"Removing file: {resolved_path}",
                    "Path not in write allowlist",
                )

            self._audit_event("fs", "os.remove", str(resolved_path), "allowed")
            return original_remove(path)

        os.remove = guarded_remove
        os.unlink = guarded_remove  # unlink is an alias

        # Patch os.rename
        original_rename = self.originals["os_rename"]

        @wraps(original_rename)
        def guarded_rename(src, dst):
            src_path = Path(src).resolve()
            dst_path = Path(dst).resolve()

            if not fs_allowed:
                self._audit_event(
                    "fs", "os.rename", f"{src_path} -> {dst_path}", "denied"
                )
                raise CapabilityViolation("fs", f"Renaming {src_path} to {dst_path}")

            if not self._check_path_allowed(src_path, write_allowlist, "write"):
                self._audit_event("fs", "os.rename.src", str(src_path), "denied")
                raise CapabilityViolation(
                    "fs",
                    f"Renaming source: {src_path}",
                    "Source path not in write allowlist",
                )

            if not self._check_path_allowed(dst_path, write_allowlist, "write"):
                self._audit_event("fs", "os.rename.dst", str(dst_path), "denied")
                raise CapabilityViolation(
                    "fs",
                    f"Renaming destination: {dst_path}",
                    "Destination path not in write allowlist",
                )

            self._audit_event("fs", "os.rename", f"{src_path} -> {dst_path}", "allowed")
            return original_rename(src, dst)

        os.rename = guarded_rename

    def _patch_subprocess(self):
        """Patch subprocess-related functions"""

        subprocess_allowed = self._check_capability("subprocess")
        subprocess_config = self.capabilities.get("subprocess", {})
        allowed_binaries = (
            subprocess_config.get("allowlist", []) if subprocess_allowed else []
        )

        # Patch subprocess.Popen
        original_popen = self.originals["subprocess_popen"]

        @wraps(original_popen)
        def guarded_popen(args, *pargs, **kwargs):
            if not subprocess_allowed:
                self._audit_event("subprocess", "Popen", str(args), "denied")
                raise CapabilityViolation("subprocess", f"Executing: {args}")

            # Extract command
            cmd = args[0] if isinstance(args, list) else args.split()[0]
            cmd_path = Path(cmd).resolve()

            if str(cmd_path) not in allowed_binaries:
                self._audit_event("subprocess", "Popen", str(cmd_path), "denied")
                raise CapabilityViolation(
                    "subprocess",
                    f"Executing: {cmd_path}",
                    f"Binary not in allowlist",
                    f"Add '{cmd_path}' to capabilities.subprocess.allowlist",
                )

            # Sanitize environment
            env = kwargs.get("env", os.environ.copy())
            safe_env = {
                "PATH": env.get("PATH", ""),
                "LANG": env.get("LANG", "C"),
                "LC_ALL": env.get("LC_ALL", "C"),
            }
            # Keep GL_* variables
            for key, value in env.items():
                if key.startswith("GL_"):
                    safe_env[key] = value

            kwargs["env"] = safe_env
            kwargs["close_fds"] = True

            self._audit_event("subprocess", "Popen", str(cmd_path), "allowed")
            return original_popen(args, *pargs, **kwargs)

        subprocess.Popen = guarded_popen

        # Patch subprocess.run
        original_run = self.originals["subprocess_run"]

        @wraps(original_run)
        def guarded_run(args, *pargs, **kwargs):
            if not subprocess_allowed:
                self._audit_event("subprocess", "run", str(args), "denied")
                raise CapabilityViolation("subprocess", f"Running: {args}")

            # Use guarded Popen
            return subprocess.run(args, *pargs, **kwargs)

        subprocess.run = guarded_run

        # Patch os.system
        @wraps(self.originals["os_system"])
        def guarded_system(command):
            if not subprocess_allowed:
                self._audit_event("subprocess", "os.system", command, "denied")
                raise CapabilityViolation("subprocess", f"System command: {command}")

            self._audit_event("subprocess", "os.system", command, "denied")
            raise CapabilityViolation(
                "subprocess",
                "os.system is not allowed",
                "Use subprocess.run with explicit binary path",
            )

        os.system = guarded_system

    def _patch_clock(self):
        """Patch time/clock-related functions"""

        clock_allowed = self._check_capability("clock")

        if not clock_allowed:
            # Frozen time mode
            frozen_time = float(os.environ.get("GL_FROZEN_TIME", time.time()))
            frozen_datetime = datetime.datetime.fromtimestamp(frozen_time)
            monotonic_counter = [0.0]

            # Patch time.time
            def guarded_time():
                self._audit_event("clock", "time.time", "frozen", "allowed")
                return frozen_time

            time.time = guarded_time

            # Patch datetime.datetime.now
            def guarded_now(tz=None):
                self._audit_event("clock", "datetime.now", "frozen", "allowed")
                if tz:
                    return frozen_datetime.replace(tzinfo=tz)
                return frozen_datetime

            datetime.datetime.now = staticmethod(guarded_now)

            # Patch time.perf_counter with monotonic counter
            def guarded_perf_counter():
                monotonic_counter[0] += 0.001  # Increment by 1ms
                self._audit_event("clock", "perf_counter", "monotonic", "allowed")
                return monotonic_counter[0]

            time.perf_counter = guarded_perf_counter
            time.monotonic = guarded_perf_counter

    def _match_domain(self, host: str, patterns: List[str]) -> bool:
        """Check if host matches any of the domain patterns"""
        for pattern in patterns:
            # Handle wildcard patterns
            if "*" in pattern:
                if fnmatch.fnmatch(host, pattern):
                    return True
            else:
                # Exact match or subdomain match
                if host == pattern or host.endswith("." + pattern):
                    return True
        return False

    def _check_path_allowed(self, path: Path, allowlist: List[str], mode: str) -> bool:
        """Check if a path is allowed based on allowlist"""
        if self.override_mode:
            return True

        # Resolve the path
        path = path.resolve()

        # Check against each pattern in allowlist
        for pattern in allowlist:
            # Replace environment variables
            expanded = pattern.replace("${INPUT_DIR}", str(self.input_dir))
            expanded = expanded.replace("${PACK_DATA_DIR}", str(self.pack_data_dir))
            expanded = expanded.replace("${RUN_TMP}", str(self.run_tmp))

            # Check if path matches pattern
            if "**" in expanded:
                # Recursive glob pattern
                base, _ = expanded.split("**", 1)
                base_path = Path(base).resolve()
                if self._is_subpath(path, base_path):
                    return True
            elif "*" in expanded:
                # Simple glob pattern
                if path.match(expanded):
                    return True
            else:
                # Exact path or prefix
                allowed_path = Path(expanded).resolve()
                if path == allowed_path or self._is_subpath(path, allowed_path):
                    return True

        # Block sensitive paths
        sensitive_paths = [
            Path.home(),
            Path("/etc"),
            Path("/proc"),
            Path("/sys"),
            Path("/var/run"),
            Path("/root"),
        ]

        for sensitive in sensitive_paths:
            if sensitive.exists() and self._is_subpath(path, sensitive):
                # Only allow if explicitly in allowlist
                return False

        return False

    def _is_subpath(self, path: Path, parent: Path) -> bool:
        """Check if path is a subpath of parent"""
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    def get_audit_log(self) -> List[Dict]:
        """Get the audit log"""
        return self.audit_log

    def save_audit_log(self, filepath: str):
        """Save audit log to file"""
        with open(filepath, "w") as f:
            json.dump(self.audit_log, f, indent=2, default=str)


def initialize_guard():
    """
    Initialize the runtime guard based on environment configuration

    This function is called when the guard module is imported in a worker process.
    """
    # Get capabilities from environment
    caps_json = os.environ.get("GL_CAPS", "{}")

    try:
        capabilities = json.loads(caps_json)
    except json.JSONDecodeError:
        logger.error("Failed to parse GL_CAPS environment variable")
        capabilities = {}

    # Create and activate guard
    guard = RuntimeGuard(capabilities)

    # Store guard instance for access
    sys.modules[__name__].guard_instance = guard

    logger.info(
        f"Runtime guard initialized with capabilities: {list(capabilities.keys())}"
    )

    return guard


# Auto-initialize when imported as main guard module
if __name__ == "__main__" or os.environ.get("GL_GUARD_INIT") == "1":
    initialize_guard()
