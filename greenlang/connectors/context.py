"""
Connector Context for Execution Control
========================================

Extends BaseContext with connector-specific execution control.

Key Features:
- Mode enforcement (record/replay/golden)
- Policy integration (egress control)
- Cache management
- Security controls
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum

# Import existing infrastructure
from greenlang.core.context_manager import BaseContext, ContextType
from greenlang.intelligence.determinism import CacheMode  # Reuse existing!


# Export CacheMode for convenience
__all__ = ['ConnectorContext', 'CacheMode']


@dataclass
class ConnectorContext(BaseContext):
    """
    Context for connector execution

    Extends BaseContext with connector-specific fields while maintaining
    compatibility with the unified context manager.

    Integrates with:
    - Policy enforcement (greenlang.policy.enforcer)
    - Security guard (greenlang.runtime.guard)
    - Determinism infrastructure (greenlang.intelligence.determinism)
    """

    # Set context type
    context_type: ContextType = field(default=ContextType.EXECUTION)

    # Connector identification
    connector_id: str = ""
    connector_version: str = "0.1.0"

    # Execution mode (reuse existing CacheMode enum!)
    mode: CacheMode = CacheMode.REPLAY  # Default to safe replay mode

    # Cache configuration
    cache_dir: Path = field(default_factory=lambda: Path(".greenlang/replay-cache"))
    cache_backend: str = "json"  # "json" or "sqlite"

    # Security and policy
    allow_egress: bool = field(default=False)  # Default deny
    egress_allowlist: List[str] = field(default_factory=list)  # Allowed domains
    require_tls: bool = field(default=True)

    # Authentication
    auth_config: Dict[str, Any] = field(default_factory=dict)
    api_key_env: Optional[str] = None  # Environment variable name for API key

    # Rate limiting and budgets
    rate_limit_per_hour: Optional[int] = None
    timeout_seconds: float = 30.0
    max_retries: int = 3

    # Query context
    query_params: Dict[str, Any] = field(default_factory=dict)

    # Snapshot management
    snapshot_path: Optional[Path] = None  # Explicit snapshot file
    snapshot_required: bool = False  # Fail if snapshot missing in replay

    def check_egress(self, url: str) -> None:
        """
        Check if network egress is allowed

        Integrates with existing security infrastructure:
        - Replay mode: always deny (deterministic execution)
        - Record/golden mode: check policy

        Args:
            url: Target URL

        Raises:
            ConnectorReplayRequired: If in replay mode
            ConnectorSecurityError: If egress not allowed
        """
        from greenlang.connectors.errors import (
            ConnectorReplayRequired,
            ConnectorSecurityError
        )

        # Replay mode: no network ever
        if self.mode == CacheMode.REPLAY:
            raise ConnectorReplayRequired(
                f"Replay mode prohibits network access to {url}. "
                f"Provide snapshot or switch to record mode.",
                connector=self.connector_id,
                url=url
            )

        # Check if egress allowed at all
        if not self.allow_egress:
            raise ConnectorSecurityError(
                f"Egress blocked by policy for {url}",
                connector=self.connector_id,
                url=url
            )

        # Check domain allowlist
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc

        if self.egress_allowlist:
            allowed = any(
                self._match_domain(domain, pattern)
                for pattern in self.egress_allowlist
            )
            if not allowed:
                raise ConnectorSecurityError(
                    f"Domain {domain} not in allowlist: {self.egress_allowlist}",
                    connector=self.connector_id,
                    url=url,
                    context={"allowlist": self.egress_allowlist}
                )

        # Require TLS
        if self.require_tls and parsed.scheme != "https":
            raise ConnectorSecurityError(
                f"TLS required but got {parsed.scheme}:// for {url}",
                connector=self.connector_id,
                url=url
            )

    def _match_domain(self, domain: str, pattern: str) -> bool:
        """
        Match domain against allowlist pattern

        Supports:
        - Exact match: "api.example.com"
        - Subdomain wildcard: "*.example.com"
        - Port specification: "api.example.com:443"

        Args:
            domain: Domain to check
            pattern: Allowlist pattern

        Returns:
            True if matches
        """
        # Exact match
        if domain == pattern:
            return True

        # Wildcard subdomain
        if pattern.startswith("*."):
            parent = pattern[2:]
            return domain.endswith(f".{parent}") or domain == parent

        # Port handling
        if ":" in pattern and ":" in domain:
            return domain == pattern

        # Check without port
        domain_no_port = domain.split(":")[0]
        pattern_no_port = pattern.split(":")[0]

        return domain_no_port == pattern_no_port

    def get_cache_path(self, query_hash: str) -> Path:
        """
        Get cache file path for a query

        Args:
            query_hash: SHA-256 hash of query

        Returns:
            Path to cache file
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.cache_backend == "sqlite":
            return self.cache_dir / f"{self.connector_id.replace('/', '_')}.db"
        else:
            return self.cache_dir / f"{self.connector_id.replace('/', '_')}_{query_hash[:8]}.json"

    def to_env(self) -> Dict[str, str]:
        """
        Convert to environment variables

        Extends BaseContext.to_env() with connector-specific vars
        """
        env = super().to_env()

        # Add connector-specific vars
        env["GL_CONNECTOR_ID"] = self.connector_id
        env["GL_CONNECTOR_MODE"] = self.mode.value
        env["GL_CONNECTOR_ALLOW_EGRESS"] = str(self.allow_egress)

        if self.api_key_env:
            # Don't copy the actual key, just note the env var name
            env["GL_CONNECTOR_API_KEY_ENV"] = self.api_key_env

        return env

    @classmethod
    def for_replay(
        cls,
        connector_id: str,
        snapshot_path: Optional[Path] = None,
        **kwargs
    ) -> "ConnectorContext":
        """
        Create context for replay mode

        Convenience factory for deterministic replay.

        Args:
            connector_id: Connector ID
            snapshot_path: Path to snapshot file
            **kwargs: Additional context fields

        Returns:
            ConnectorContext configured for replay
        """
        return cls(
            connector_id=connector_id,
            mode=CacheMode.REPLAY,
            snapshot_path=snapshot_path,
            allow_egress=False,  # No network in replay
            **kwargs
        )

    @classmethod
    def for_record(
        cls,
        connector_id: str,
        allow_egress: bool = True,
        egress_allowlist: Optional[List[str]] = None,
        **kwargs
    ) -> "ConnectorContext":
        """
        Create context for record mode

        Convenience factory for recording new snapshots.

        Args:
            connector_id: Connector ID
            allow_egress: Allow network access
            egress_allowlist: Allowed domains
            **kwargs: Additional context fields

        Returns:
            ConnectorContext configured for record
        """
        return cls(
            connector_id=connector_id,
            mode=CacheMode.RECORD,
            allow_egress=allow_egress,
            egress_allowlist=egress_allowlist or [],
            **kwargs
        )

    @classmethod
    def for_golden(
        cls,
        connector_id: str,
        snapshot_path: Path,
        **kwargs
    ) -> "ConnectorContext":
        """
        Create context for golden mode

        Convenience factory for golden/reference testing.

        Args:
            connector_id: Connector ID
            snapshot_path: Path to golden snapshot
            **kwargs: Additional context fields

        Returns:
            ConnectorContext configured for golden
        """
        return cls(
            connector_id=connector_id,
            mode=CacheMode.GOLDEN,
            snapshot_path=snapshot_path,
            snapshot_required=True,  # Golden must have snapshot
            allow_egress=False,  # No network in golden
            **kwargs
        )
