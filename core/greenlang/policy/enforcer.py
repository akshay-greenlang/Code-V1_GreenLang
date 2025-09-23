"""
Policy Enforcer
===============

Enforces OPA (Open Policy Agent) policies at install and runtime.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Tuple
from .opa import evaluate as opa_eval

logger = logging.getLogger(__name__)


# Standalone functions for direct import
def check_install(pm, path: str, stage: str = "publish") -> None:
    """
    Check if pack can be installed or published (raises on failure)

    Args:
        pm: PackManifest object
        path: Path to pack directory
        stage: Either "publish" or "add"

    Raises:
        RuntimeError: If policy denies the operation
    """
    dec = opa_eval(
        "bundles/install.rego",
        {
            "pack": pm.model_dump() if hasattr(pm, "model_dump") else pm.dict(),
            "stage": stage,
        },
    )
    if not dec["allow"]:
        raise RuntimeError(dec.get("reason", "policy denied"))


def check_run(pipeline, ctx) -> None:
    """
    Check if pipeline can be executed (raises on failure)

    Args:
        pipeline: Pipeline object
        ctx: Execution context

    Raises:
        RuntimeError: If policy denies the operation
    """
    dec = opa_eval(
        "bundles/run.rego",
        {
            "pipeline": (
                pipeline.to_policy_doc()
                if hasattr(pipeline, "to_policy_doc")
                else pipeline.__dict__
            ),
            "egress": getattr(ctx, "egress_targets", []),
            "region": getattr(ctx, "region", "unknown"),
        },
    )
    if not dec["allow"]:
        raise RuntimeError(dec.get("reason", "policy denied"))


class PolicyEnforcer:
    """
    Policy enforcement using OPA-style rules

    Policies can be enforced at:
    - Install time: When adding packs
    - Runtime: Before executing pipelines
    - Data access: When accessing sensitive data
    """

    def __init__(self, policy_dir: Optional[Path] = None):
        """
        Initialize policy enforcer

        Args:
            policy_dir: Directory containing policy files
        """
        self.policy_dir = policy_dir or Path.home() / ".greenlang" / "policies"
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        self.policies = {}
        self._load_policies()

    def _load_policies(self):
        """Load all policy files"""
        for policy_file in self.policy_dir.glob("*.rego"):
            try:
                with open(policy_file) as f:
                    policy_content = f.read()
                self.policies[policy_file.stem] = policy_content
                logger.info(f"Loaded policy: {policy_file.stem}")
            except Exception as e:
                logger.error(f"Failed to load policy {policy_file}: {e}")

    def check(self, policy_file: Path, input_data: Dict[str, Any]) -> bool:
        """
        Check if input satisfies policy

        Args:
            policy_file: Path to policy file
            input_data: Data to check against policy

        Returns:
            True if policy check passes
        """
        if not policy_file.exists():
            logger.error(f"Policy file not found: {policy_file}")
            return False

        try:
            # Try to use actual OPA evaluation if available
            try:
                from .opa_evaluator import evaluate_rego

                result = evaluate_rego(policy_file, input_data)
                return result.get("allow", False)
            except ImportError:
                # Fallback to simple evaluation
                return self._simple_rego_eval(policy_file, input_data)

        except Exception as e:
            logger.error(f"Policy check failed: {e}")
            return False

    def _simple_rego_eval(self, policy_file: Path, input_data: Dict[str, Any]) -> bool:
        """
        Simple Rego-like evaluation without OPA

        This is a simplified evaluator for demo purposes.
        In production, use actual OPA.
        """
        with open(policy_file) as f:
            policy_content = f.read()

        # Parse basic Rego patterns
        if "package greenlang.install" in policy_content:
            return self._eval_install_policy(input_data)
        elif "package greenlang.runtime" in policy_content:
            return self._eval_runtime_policy(input_data)
        else:
            # Generic evaluation
            return self._eval_generic_policy(policy_content, input_data)

    def _eval_install_policy(self, input_data: Dict[str, Any]) -> bool:
        """Evaluate install policy"""
        pack = input_data.get("pack", {})

        # Check license
        license = pack.get("license", "")
        if license not in ["MIT", "Apache-2.0", "Commercial"]:
            logger.warning(f"Policy denied: License {license} not in allowlist")
            return False

        # Check network policy for packs
        if pack.get("kind") == "pack":
            network = pack.get("policy", {}).get("network", [])
            if not network:
                logger.warning("Policy denied: Network allowlist is empty")
                return False

        # Check EF vintage
        ef_vintage = pack.get("policy", {}).get("ef_vintage_min")
        if ef_vintage and ef_vintage < 2024:
            logger.warning(f"Policy denied: EF vintage {ef_vintage} too old")
            return False

        # Check SBOM
        if not pack.get("security", {}).get("sbom"):
            logger.warning("Policy denied: SBOM not provided")
            return False

        return True

    def _eval_runtime_policy(self, input_data: Dict[str, Any]) -> bool:
        """Evaluate runtime policy"""
        # Check authentication
        if not input_data.get("user", {}).get("authenticated"):
            logger.warning("Policy denied: User not authenticated")
            return False

        # Check resource limits
        resources = input_data.get("resources", {})
        if resources.get("memory_mb", 0) > 4096:
            logger.warning("Policy denied: Excessive memory requested")
            return False
        if resources.get("cpu_cores", 0) > 4:
            logger.warning("Policy denied: Excessive CPU requested")
            return False

        # Check rate limits
        rpm = input_data.get("user", {}).get("requests_per_minute", 0)
        role = input_data.get("user", {}).get("role", "basic")

        if role == "premium" and rpm > 1000:
            logger.warning("Policy denied: Premium rate limit exceeded")
            return False
        elif role != "premium" and rpm > 100:
            logger.warning("Policy denied: Rate limit exceeded")
            return False

        return True

    def _eval_generic_policy(
        self, policy_content: str, input_data: Dict[str, Any]
    ) -> bool:
        """Generic policy evaluation"""
        # Simple keyword-based evaluation
        if "deny" in policy_content.lower():
            # Conservative: deny by default if deny rules exist
            return False

        if "allow {" in policy_content:
            # Check for basic allow conditions
            if "authenticated" in policy_content:
                if not input_data.get("user", {}).get("authenticated"):
                    return False

        # Default allow
        return True

    def check_install(
        self, pack_manifest: Any, path: str, stage: Literal["publish", "add"]
    ) -> Tuple[bool, List[str]]:
        """
        Check if pack can be installed or published

        Args:
            pack_manifest: Pack manifest object
            path: Path to pack directory
            stage: Either "publish" or "add"

        Returns:
            (allowed, list_of_reasons)
        """
        # Build input document
        input_doc = {
            "stage": stage,
            "pack": (
                pack_manifest.dict()
                if hasattr(pack_manifest, "dict")
                else pack_manifest
            ),
            "files": self._list_files(path),
            "licenses": self._detect_licenses(path),
        }

        # Evaluate policy
        decision = opa_eval("bundles/install.rego", input_doc)

        allowed = decision.get("allow", False)
        reasons = decision.get(
            "reasons", [decision.get("reason", "No reason provided")]
        )

        if not isinstance(reasons, list):
            reasons = [reasons]

        return allowed, reasons

    def check_run(self, pipeline: Any, context: Any) -> Tuple[bool, List[str]]:
        """
        Check if pipeline can be executed

        Args:
            pipeline: Pipeline object
            context: Execution context

        Returns:
            (allowed, list_of_reasons)
        """
        # Build input document
        input_doc = {
            "stage": "run",
            "pipeline": self._pipeline_to_policy_doc(pipeline),
            "profile": getattr(context, "profile", "dev"),
            "region": getattr(context, "region", None),
            "egress": self._get_egress_targets(context),
            "ef_vintage": (
                context.metadata.get("ef_vintage")
                if hasattr(context, "metadata")
                else None
            ),
        }

        # Evaluate policy
        decision = opa_eval("bundles/run.rego", input_doc)

        allowed = decision.get("allow", False)
        reasons = decision.get(
            "reasons", [decision.get("reason", "No reason provided")]
        )

        if not isinstance(reasons, list):
            reasons = [reasons]

        return allowed, reasons

    def add_policy(self, policy_file: Path):
        """Add a new policy"""
        if not policy_file.exists():
            raise ValueError(f"Policy file not found: {policy_file}")

        # Copy to policy directory
        dest = self.policy_dir / policy_file.name

        with open(policy_file) as f:
            content = f.read()

        with open(dest, "w") as f:
            f.write(content)

        self.policies[policy_file.stem] = content
        logger.info(f"Added policy: {policy_file.stem}")

    def remove_policy(self, policy_name: str):
        """Remove a policy"""
        policy_file = self.policy_dir / f"{policy_name}.rego"

        if policy_file.exists():
            policy_file.unlink()
            del self.policies[policy_name]
            logger.info(f"Removed policy: {policy_name}")
        else:
            raise ValueError(f"Policy not found: {policy_name}")

    def list_policies(self) -> List[str]:
        """List all policies"""
        return list(self.policies.keys())

    def get_policy(self, policy_name: str) -> Optional[str]:
        """Get policy content"""
        return self.policies.get(policy_name)

    def _list_files(self, path: str) -> List[str]:
        """List all files in a directory"""
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            return []

        files = []
        for item in p.rglob("*"):
            if item.is_file():
                files.append(str(item.relative_to(p)))
        return files

    def _detect_licenses(self, path: str) -> List[str]:
        """Detect licenses in a directory"""
        from pathlib import Path

        licenses = []
        p = Path(path)

        # Check for license files
        for pattern in ["LICENSE*", "LICENCE*", "COPYING*"]:
            for license_file in p.glob(pattern):
                if license_file.is_file():
                    # Try to detect license type from content
                    content = license_file.read_text(errors="ignore")[:500].lower()
                    if "mit" in content:
                        licenses.append("MIT")
                    elif "apache" in content:
                        licenses.append("Apache-2.0")
                    elif "gpl" in content:
                        licenses.append("GPL")
                    elif "bsd" in content:
                        licenses.append("BSD")
                    else:
                        licenses.append("Unknown")

        return list(set(licenses))

    def _pipeline_to_policy_doc(self, pipeline: Any) -> Dict[str, Any]:
        """Convert pipeline to policy document"""
        if hasattr(pipeline, "to_policy_doc"):
            return pipeline.to_policy_doc()
        elif hasattr(pipeline, "dict"):
            return pipeline.dict()
        elif hasattr(pipeline, "to_dict"):
            return pipeline.to_dict()
        else:
            return {
                "name": getattr(pipeline, "name", "unknown"),
                "version": getattr(pipeline, "version", "1.0"),
                "steps": getattr(pipeline, "steps", []),
            }

    def _get_egress_targets(self, context: Any) -> List[str]:
        """Get allowed egress targets from context"""
        if hasattr(context, "config") and hasattr(context.config, "egress_targets"):
            return context.config.egress_targets
        return []

    def create_default_policies(self):
        """Create default policy templates"""

        # Install policy template
        install_policy = """package greenlang.install

# Deny untrusted sources
deny[msg] {
    input.pack.source == "untrusted"
    msg := "Pack from untrusted source"
}

# Deny packs without signatures
deny[msg] {
    input.pack.provenance.signing == false
    msg := "Pack must be signed"
}

# Limit pack size
deny[msg] {
    input.pack.size > 100000000  # 100MB
    msg := "Pack too large"
}

# Allow verified publishers
allow {
    input.pack.publisher in ["greenlang", "verified"]
}
"""

        # Runtime policy template
        runtime_policy = """package greenlang.runtime

# Deny excessive resource usage
deny[msg] {
    input.resources.memory > 4096  # 4GB
    msg := "Excessive memory requested"
}

deny[msg] {
    input.resources.cpu > 4
    msg := "Excessive CPU requested"
}

# Deny access to sensitive data
deny[msg] {
    input.data.classification == "confidential"
    input.user.clearance != "high"
    msg := "Insufficient clearance for confidential data"
}

# Rate limiting
deny[msg] {
    input.user.requests_per_minute > 100
    msg := "Rate limit exceeded"
}

# Allow authenticated users
allow {
    input.user.authenticated == true
}
"""

        # Data access policy template
        data_policy = """package greenlang.data

# Enforce data residency
deny[msg] {
    input.data.region != input.user.region
    input.data.residency_required == true
    msg := "Data residency violation"
}

# Enforce encryption
deny[msg] {
    input.data.sensitive == true
    input.connection.encrypted == false
    msg := "Sensitive data requires encryption"
}

# Audit logging
audit[msg] {
    input.data.classification in ["confidential", "restricted"]
    msg := sprintf("Data access: user=%s data=%s", [input.user.id, input.data.id])
}
"""

        # Save default policies
        with open(self.policy_dir / "install.rego", "w") as f:
            f.write(install_policy)

        with open(self.policy_dir / "runtime.rego", "w") as f:
            f.write(runtime_policy)

        with open(self.policy_dir / "data.rego", "w") as f:
            f.write(data_policy)

        logger.info("Created default policies")
