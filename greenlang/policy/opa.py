"""
OPA (Open Policy Agent) integration for policy evaluation
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def evaluate(
    policy_path: str, input_doc: Dict[str, Any], data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate an OPA policy against input document

    Args:
        policy_path: Path to .rego policy file or bundle
        input_doc: Input document for policy evaluation
        data: Optional data document for policy

    Returns:
        Decision document with at least {"allow": bool, "reason": str}
    """
    # Check if OPA is available
    if not _check_opa_installed():
        logger.error("OPA not installed, denying by default")
        return {"allow": False, "reason": "POLICY.DENIED: OPA not available"}

    # Resolve policy path
    policy_file = _resolve_policy_path(policy_path)
    if not policy_file.exists():
        logger.error(f"Policy not found: {policy_path}, denying by default")
        return {
            "allow": False,
            "reason": f"POLICY.DENIED: Policy not found: {policy_path}",
        }

    try:
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(input_doc, f)
            input_file = Path(f.name)

        # Create temporary data file if provided
        data_file = None
        if data:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(data, f)
                data_file = Path(f.name)

        # Build OPA command - use local opa.exe if available
        cwd = Path.cwd()
        opa_exe = cwd / "opa.exe"
        opa_cmd = str(opa_exe) if opa_exe.exists() else "opa"

        cmd = [
            opa_cmd,
            "eval",
            "-d",
            str(policy_file),
            "-i",
            str(input_file),
            "--format",
            "json",
            "data.greenlang.decision",
        ]

        if data_file:
            cmd.extend(["-d", str(data_file)])

        # Execute OPA
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            logger.error(f"OPA evaluation failed: {result.stderr}")
            return {
                "allow": False,
                "reason": f"Policy evaluation error: {result.stderr[:200]}",
            }

        # Parse result
        output = json.loads(result.stdout)

        # Extract decision from OPA output structure
        # OPA returns: {"result": [{"expressions": [{"value": {...}}]}]}
        if output.get("result") and len(output["result"]) > 0:
            expressions = output["result"][0].get("expressions", [])
            if expressions and len(expressions) > 0:
                decision = expressions[0].get("value", {})

                # Ensure required fields
                if "allow" not in decision:
                    decision["allow"] = False
                if "reason" not in decision:
                    decision["reason"] = "No reason provided"

                return decision

        # Fallback if no valid result
        return {"allow": False, "reason": "No policy decision returned"}

    except subprocess.TimeoutExpired:
        logger.error("OPA evaluation timed out")
        return {"allow": False, "reason": "Policy evaluation timeout"}
    except Exception as e:
        logger.error(f"OPA evaluation error: {e}")
        return {"allow": False, "reason": f"Policy evaluation error: {str(e)[:200]}"}
    finally:
        # Cleanup temp files
        if "input_file" in locals():
            input_file.unlink(missing_ok=True)
        if "data_file" in locals() and data_file:
            data_file.unlink(missing_ok=True)


def evaluate_inline(policy_content: str, input_doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate inline policy content

    Args:
        policy_content: Rego policy as string
        input_doc: Input document

    Returns:
        Decision document
    """
    try:
        # Create temporary policy file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".rego", delete=False) as f:
            f.write(policy_content)
            policy_file = f.name

        # Evaluate using the temporary file
        return evaluate(policy_file, input_doc)

    finally:
        # Cleanup
        if "policy_file" in locals():
            Path(policy_file).unlink(missing_ok=True)


def _check_opa_installed() -> bool:
    """Check if OPA is installed and available"""
    # Try local opa.exe first (Windows)
    cwd = Path.cwd()
    opa_exe = cwd / "opa.exe"
    if opa_exe.exists():
        try:
            result = subprocess.run(
                [str(opa_exe), "version"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            pass

    # Try system OPA
    try:
        result = subprocess.run(["opa", "version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _resolve_policy_path(policy_path: str) -> Path:
    """
    Resolve policy path from various locations

    Checks in order:
    1. Absolute path
    2. Relative to current directory
    3. In user config directory
    4. In system config directory
    5. In package bundles directory
    """
    path = Path(policy_path)

    # If absolute or exists relative to cwd
    if path.is_absolute() or path.exists():
        return path

    # Check standard locations
    search_paths = [
        Path.cwd() / "policies" / policy_path,
        Path.home() / ".greenlang" / "policies" / policy_path,
        Path.home() / ".config" / "greenlang" / "policies" / policy_path,
        Path(__file__).parent / "bundles" / policy_path,
    ]

    for search_path in search_paths:
        if search_path.exists():
            return search_path

    # Return original path (will fail exists check)
    return path


def validate_policy(policy_path: str) -> tuple[bool, list[str]]:
    """
    Validate a policy file for syntax errors

    Returns:
        (is_valid, list_of_errors)
    """
    policy_file = _resolve_policy_path(policy_path)

    if not policy_file.exists():
        return False, [f"Policy file not found: {policy_path}"]

    if not _check_opa_installed():
        return False, ["OPA not installed - cannot validate policy"]

    # Use local OPA if available
    cwd = Path.cwd()
    opa_exe = cwd / "opa.exe"
    opa_cmd = str(opa_exe) if opa_exe.exists() else "opa"

    try:
        result = subprocess.run(
            [opa_cmd, "test", str(policy_file), "--explain", "fails"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            return True, []
        else:
            errors = (
                result.stderr.strip().split("\n") if result.stderr else [result.stdout]
            )
            return False, errors

    except Exception as e:
        return False, [f"Validation error: {str(e)}"]
