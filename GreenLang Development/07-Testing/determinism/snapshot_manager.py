# -*- coding: utf-8 -*-
"""Snapshot Testing Infrastructure for GreenLang.

This module provides golden file testing capabilities for agent outputs.
Snapshots are saved as JSON files and can be compared to detect regressions.

Key Features:
- JSON-based snapshot storage
- Pretty-formatted snapshots for readability
- Snapshot diffs with detailed change reports
- Auto-update mode for blessing new snapshots
- Cross-platform path normalization

Author: GreenLang Framework Team
Phase: Phase 3 - Production Hardening
Date: November 2024
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SnapshotDiff:
    """Difference between snapshot and actual output.

    Attributes:
        matches: Whether snapshot matches output
        added: Fields added in new output
        removed: Fields removed from snapshot
        changed: Fields with changed values
        details: Detailed list of changes
    """
    matches: bool
    added: List[str]
    removed: List[str]
    changed: List[Dict[str, Any]]
    details: str

    def __str__(self) -> str:
        """Human-readable representation."""
        if self.matches:
            return "Snapshot matches output (no changes)"

        parts = []
        if self.added:
            parts.append(f"Added fields: {', '.join(self.added)}")
        if self.removed:
            parts.append(f"Removed fields: {', '.join(self.removed)}")
        if self.changed:
            parts.append(f"Changed fields: {len(self.changed)}")

        return "\n".join(parts) + f"\n\nDetails:\n{self.details}"


class SnapshotManager:
    """Manager for snapshot-based testing.

    This class handles saving and comparing agent outputs to golden files
    (snapshots) for regression testing.

    Usage:
        >>> manager = SnapshotManager()
        >>> manager.save_snapshot("test_fuel_agent", agent_output)
        >>> diff = manager.compare_snapshot("test_fuel_agent", new_output)
        >>> assert diff.matches

    Features:
        - Automatic snapshot directory creation
        - Pretty JSON formatting for readability
        - Detailed diff reporting
        - Auto-update mode via environment variable
        - Cross-platform compatibility
    """

    def __init__(
        self,
        snapshot_dir: Optional[Path] = None,
        auto_update: Optional[bool] = None,
        normalize_output: bool = True,
    ):
        """Initialize SnapshotManager.

        Args:
            snapshot_dir: Directory for snapshot files (default: tests/determinism/snapshots)
            auto_update: Auto-update snapshots (default: from UPDATE_SNAPSHOTS env var)
            normalize_output: Normalize outputs before saving
        """
        if snapshot_dir is None:
            # Default to tests/determinism/snapshots
            self.snapshot_dir = Path(__file__).parent / "snapshots"
        else:
            self.snapshot_dir = Path(snapshot_dir)

        # Create directory if it doesn't exist
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Auto-update mode (can be overridden by env var)
        if auto_update is None:
            self.auto_update = os.getenv("UPDATE_SNAPSHOTS", "0") == "1"
        else:
            self.auto_update = auto_update

        self.normalize_output = normalize_output

    def _normalize_data(self, data: Any) -> Any:
        """Normalize data for consistent comparison.

        Args:
            data: Data to normalize

        Returns:
            Normalized data
        """
        if not self.normalize_output:
            return data

        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                # Skip non-deterministic fields
                if key in (
                    'timestamp', 'created_at', 'updated_at', 'run_id',
                    'execution_time', 'duration_ms', 'wall_time',
                    'platform', 'hostname', 'username',
                ):
                    continue
                result[key] = self._normalize_data(value)
            return result

        if isinstance(data, (list, tuple)):
            return [self._normalize_data(item) for item in data]

        if isinstance(data, str):
            # Normalize paths
            return data.replace('\\', '/')

        return data

    def _get_snapshot_path(self, test_name: str) -> Path:
        """Get path to snapshot file.

        Args:
            test_name: Name of the test

        Returns:
            Path to snapshot file
        """
        # Sanitize test name for filesystem
        safe_name = re.sub(r'[^\w\-]', '_', test_name)
        return self.snapshot_dir / f"{safe_name}.snapshot.json"

    def save_snapshot(
        self,
        test_name: str,
        output: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save output as snapshot.

        Args:
            test_name: Name of the test
            output: Agent output to save
            metadata: Optional metadata to include

        Returns:
            Path to saved snapshot file
        """
        # Normalize output
        normalized = self._normalize_data(output)

        # Create snapshot structure
        snapshot = {
            "test_name": test_name,
            "output": normalized,
        }

        if metadata:
            snapshot["metadata"] = metadata

        # Save to file with pretty formatting
        snapshot_path = self._get_snapshot_path(test_name)
        with open(snapshot_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, sort_keys=True, ensure_ascii=False)

        return snapshot_path

    def load_snapshot(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Load snapshot from file.

        Args:
            test_name: Name of the test

        Returns:
            Snapshot data or None if not found
        """
        snapshot_path = self._get_snapshot_path(test_name)

        if not snapshot_path.exists():
            return None

        with open(snapshot_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def compare_snapshot(
        self,
        test_name: str,
        actual_output: Any,
        update_on_mismatch: Optional[bool] = None,
    ) -> SnapshotDiff:
        """Compare actual output to saved snapshot.

        Args:
            test_name: Name of the test
            actual_output: Current agent output
            update_on_mismatch: Update snapshot if different (default: auto_update setting)

        Returns:
            SnapshotDiff with comparison results
        """
        # Load existing snapshot
        snapshot_data = self.load_snapshot(test_name)

        # Normalize actual output
        normalized_output = self._normalize_data(actual_output)

        # If no snapshot exists, save it
        if snapshot_data is None:
            self.save_snapshot(test_name, actual_output)
            return SnapshotDiff(
                matches=True,
                added=[],
                removed=[],
                changed=[],
                details="No snapshot found - created new snapshot",
            )

        # Extract expected output from snapshot
        expected_output = snapshot_data.get("output", {})

        # Compare outputs
        diff = self._compare_outputs(expected_output, normalized_output)

        # Auto-update if enabled
        should_update = update_on_mismatch if update_on_mismatch is not None else self.auto_update
        if not diff.matches and should_update:
            self.save_snapshot(test_name, actual_output)
            diff.details += "\n\n[UPDATED] Snapshot has been updated"

        return diff

    def _compare_outputs(
        self,
        expected: Any,
        actual: Any,
        path: str = "root",
    ) -> SnapshotDiff:
        """Compare two outputs and generate diff.

        Args:
            expected: Expected output from snapshot
            actual: Actual output from test
            path: Current path in nested structure

        Returns:
            SnapshotDiff with comparison results
        """
        added = []
        removed = []
        changed = []
        details_parts = []

        # Type mismatch
        if type(expected) != type(actual):
            changed.append({
                "path": path,
                "type": "type_mismatch",
                "expected_type": type(expected).__name__,
                "actual_type": type(actual).__name__,
            })
            details_parts.append(
                f"[TYPE MISMATCH] {path}: "
                f"expected {type(expected).__name__}, got {type(actual).__name__}"
            )

        # Dict comparison
        elif isinstance(expected, dict):
            expected_keys = set(expected.keys())
            actual_keys = set(actual.keys())

            # Added keys
            for key in actual_keys - expected_keys:
                added.append(f"{path}.{key}")
                details_parts.append(f"[ADDED] {path}.{key} = {actual[key]}")

            # Removed keys
            for key in expected_keys - actual_keys:
                removed.append(f"{path}.{key}")
                details_parts.append(f"[REMOVED] {path}.{key}")

            # Compare common keys recursively
            for key in expected_keys & actual_keys:
                sub_diff = self._compare_outputs(
                    expected[key], actual[key], f"{path}.{key}"
                )
                added.extend(sub_diff.added)
                removed.extend(sub_diff.removed)
                changed.extend(sub_diff.changed)
                if sub_diff.details:
                    details_parts.append(sub_diff.details)

        # List comparison
        elif isinstance(expected, list):
            if len(expected) != len(actual):
                changed.append({
                    "path": path,
                    "type": "length_mismatch",
                    "expected_length": len(expected),
                    "actual_length": len(actual),
                })
                details_parts.append(
                    f"[LENGTH MISMATCH] {path}: "
                    f"expected {len(expected)} items, got {len(actual)}"
                )
            else:
                for i, (exp_item, act_item) in enumerate(zip(expected, actual)):
                    sub_diff = self._compare_outputs(exp_item, act_item, f"{path}[{i}]")
                    added.extend(sub_diff.added)
                    removed.extend(sub_diff.removed)
                    changed.extend(sub_diff.changed)
                    if sub_diff.details:
                        details_parts.append(sub_diff.details)

        # Value comparison
        elif expected != actual:
            changed.append({
                "path": path,
                "type": "value_mismatch",
                "expected": expected,
                "actual": actual,
            })
            details_parts.append(
                f"[CHANGED] {path}:\n"
                f"  Expected: {expected}\n"
                f"  Actual:   {actual}"
            )

        # Build result
        matches = not (added or removed or changed)
        details = "\n".join(details_parts) if details_parts else "No differences"

        return SnapshotDiff(
            matches=matches,
            added=added,
            removed=removed,
            changed=changed,
            details=details,
        )

    def delete_snapshot(self, test_name: str) -> bool:
        """Delete a snapshot file.

        Args:
            test_name: Name of the test

        Returns:
            True if snapshot was deleted, False if not found
        """
        snapshot_path = self._get_snapshot_path(test_name)
        if snapshot_path.exists():
            snapshot_path.unlink()
            return True
        return False

    def list_snapshots(self) -> List[str]:
        """List all snapshot test names.

        Returns:
            List of test names with snapshots
        """
        snapshots = []
        for path in self.snapshot_dir.glob("*.snapshot.json"):
            # Extract test name from filename
            test_name = path.stem.replace('.snapshot', '')
            snapshots.append(test_name)
        return sorted(snapshots)

    def get_snapshot_info(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a snapshot.

        Args:
            test_name: Name of the test

        Returns:
            Snapshot metadata or None if not found
        """
        snapshot_path = self._get_snapshot_path(test_name)
        if not snapshot_path.exists():
            return None

        return {
            "test_name": test_name,
            "path": str(snapshot_path),
            "size_bytes": snapshot_path.stat().st_size,
            "exists": True,
        }


def assert_snapshot_matches(
    diff: SnapshotDiff,
    message: Optional[str] = None
) -> None:
    """Assert that snapshot diff shows a match.

    Args:
        diff: SnapshotDiff to check
        message: Optional custom error message

    Raises:
        AssertionError: If snapshot doesn't match
    """
    if not diff.matches:
        error_msg = message or (
            f"Snapshot does not match actual output!\n\n"
            f"{diff}\n"
        )
        raise AssertionError(error_msg)
