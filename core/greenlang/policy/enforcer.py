"""
Policy Enforcer
===============

Enforces OPA (Open Policy Agent) policies at install and runtime.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
        # TODO: Integrate with actual OPA or implement Rego evaluation
        # For now, implement simple rule checking
        
        if not policy_file.exists():
            logger.error(f"Policy file not found: {policy_file}")
            return False
        
        try:
            with open(policy_file) as f:
                policy_content = f.read()
            
            # Simple policy format for demo
            # Real implementation would use OPA
            if "deny" in policy_content:
                # Check deny rules
                if "untrusted_source" in policy_content:
                    source = input_data.get("source", "")
                    if source and "untrusted" in source:
                        logger.warning("Policy denied: untrusted source")
                        return False
                
                if "max_resources" in policy_content:
                    resources = input_data.get("resources", {})
                    if resources.get("memory", 0) > 4096:  # 4GB limit
                        logger.warning("Policy denied: excessive resources")
                        return False
            
            # Default allow if no deny rules match
            return True
            
        except Exception as e:
            logger.error(f"Policy check failed: {e}")
            return False
    
    def check_install(self, pack_manifest: Dict[str, Any]) -> bool:
        """
        Check if pack can be installed
        
        Args:
            pack_manifest: Pack manifest to check
        
        Returns:
            True if pack can be installed
        """
        # Check against install policy
        install_policy = self.policy_dir / "install.rego"
        
        if install_policy.exists():
            return self.check(install_policy, {"pack": pack_manifest})
        
        # Default allow if no policy
        return True
    
    def check_runtime(self, pipeline: str, input_data: Dict[str, Any]) -> bool:
        """
        Check if pipeline can be executed
        
        Args:
            pipeline: Pipeline reference
            input_data: Pipeline input
        
        Returns:
            True if pipeline can be executed
        """
        # Check against runtime policy
        runtime_policy = self.policy_dir / "runtime.rego"
        
        if runtime_policy.exists():
            return self.check(runtime_policy, {
                "pipeline": pipeline,
                "input": input_data
            })
        
        # Default allow if no policy
        return True
    
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