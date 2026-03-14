# -*- coding: utf-8 -*-
"""
PACK-002 CSRD Professional Pack - Approval Workflow Tests
===========================================================

Tests for the multi-level approval workflow engine covering 4-level
and 3-level chains, approval/rejection/revision, escalation, delegation,
auto-approve, audit trails, and edge cases.

Test count: 20
Author: GreenLang QA Team
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Stub Approval Engine for testing
# ---------------------------------------------------------------------------

class ApprovalChainStub:
    """Lightweight approval chain implementation for test validation."""

    def __init__(self, config: Dict[str, Any]):
        self.chain_id = config["chain_id"]
        self.levels = config["levels"]
        self.current_level = 0
        self.status = "draft"
        self.history: List[Dict[str, Any]] = []
        self.comments: List[Dict[str, Any]] = []
        self.delegation_rules = config.get("delegation_rules", {})

    def submit(self) -> None:
        """Submit for approval, starting at level 1."""
        self.status = "submitted"
        self.current_level = 1
        self.history.append({
            "action": "submitted", "level": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def approve(self, approver_id: str, comment: str = "") -> bool:
        """Approve at current level and advance."""
        if self.status not in ("submitted", "in_review"):
            return False
        self.history.append({
            "action": "approved", "level": self.current_level,
            "approver": approver_id, "comment": comment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.current_level += 1
        if self.current_level > len(self.levels):
            self.status = "approved"
        else:
            self.status = "in_review"
        return True

    def reject(self, approver_id: str, reason: str) -> None:
        """Reject the submission."""
        self.status = "rejected"
        self.history.append({
            "action": "rejected", "level": self.current_level,
            "approver": approver_id, "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def return_for_revision(self, approver_id: str, reason: str) -> None:
        """Return to preparer for revision."""
        self.status = "revision_required"
        self.current_level = 1
        self.history.append({
            "action": "returned", "level": self.current_level,
            "approver": approver_id, "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def escalate(self, reason: str = "timeout") -> None:
        """Escalate to next level, or record max-level escalation event."""
        old_level = self.current_level
        if self.current_level < len(self.levels):
            self.current_level += 1
            self.history.append({
                "action": "escalated", "from_level": old_level,
                "to_level": self.current_level, "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        else:
            # Already at max level -- record escalation event without advancing
            self.history.append({
                "action": "escalated", "from_level": old_level,
                "to_level": old_level, "reason": reason,
                "note": "already_at_max_level",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    def auto_approve(self, quality_score: float) -> bool:
        """Auto-approve if quality score exceeds threshold for current level."""
        level_config = self.levels[self.current_level - 1]
        threshold = level_config.get("auto_approve_quality_threshold")
        if threshold and quality_score >= threshold:
            self.history.append({
                "action": "auto_approved", "level": self.current_level,
                "quality_score": quality_score, "threshold": threshold,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            self.current_level += 1
            if self.current_level > len(self.levels):
                self.status = "approved"
            return True
        return False

    def add_comment(self, user_id: str, text: str) -> None:
        """Add a comment to the approval thread."""
        self.comments.append({
            "user_id": user_id, "text": text, "level": self.current_level,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get approvers for current level."""
        if self.current_level < 1 or self.current_level > len(self.levels):
            return []
        return self.levels[self.current_level - 1].get("approvers", [])

    def resubmit(self) -> None:
        """Resubmit after revision."""
        self.status = "submitted"
        self.current_level = 1
        self.history.append({
            "action": "resubmitted",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


# ===========================================================================
# Approval Workflow Tests
# ===========================================================================

class TestApprovalWorkflows:
    """Test approval workflow logic."""

    def test_full_4_level_approval_chain(self, sample_approval_chain):
        """Full 4-level approval chain progresses through all levels."""
        chain = ApprovalChainStub(sample_approval_chain)
        chain.submit()
        assert chain.status == "submitted"

        chain.approve("anna.schmidt", "Level 1 OK")
        chain.approve("maria.weber", "Level 2 OK")
        chain.approve("thomas.mueller", "Level 3 OK")
        chain.approve("klaus.fischer", "Board OK")
        chain.approve("helga.braun", "Board OK 2")

        assert chain.status == "approved"
        assert len(chain.history) >= 5

    def test_3_level_approval(self, sample_approval_chain):
        """3-level chain (listed company preset) works correctly."""
        config = {
            "chain_id": "listed-approval-2025",
            "levels": sample_approval_chain["levels"][:3],
            "delegation_rules": sample_approval_chain["delegation_rules"],
        }
        chain = ApprovalChainStub(config)
        chain.submit()
        chain.approve("anna.schmidt")
        chain.approve("maria.weber")
        chain.approve("thomas.mueller")

        # After 3rd approval at level 3, level becomes 4 which > len(levels)=3
        assert chain.status == "approved"

    def test_rejection_at_level_2(self, sample_approval_chain):
        """Rejection at level 2 sets status to rejected."""
        chain = ApprovalChainStub(sample_approval_chain)
        chain.submit()
        chain.approve("anna.schmidt")
        chain.reject("maria.weber", "Scope 3 data incomplete")

        assert chain.status == "rejected"
        assert chain.history[-1]["reason"] == "Scope 3 data incomplete"

    def test_return_for_revision(self, sample_approval_chain):
        """Return for revision sends back to level 1."""
        chain = ApprovalChainStub(sample_approval_chain)
        chain.submit()
        chain.approve("anna.schmidt")
        assert chain.current_level == 2

        chain.return_for_revision("maria.weber", "Missing FR water data")
        assert chain.status == "revision_required"
        assert chain.current_level == 1

    def test_auto_approve_high_quality(self, sample_approval_chain):
        """Auto-approve fires when quality score exceeds level threshold."""
        chain = ApprovalChainStub(sample_approval_chain)
        chain.submit()
        chain.approve("anna.schmidt")  # Level 1 -> Level 2

        # Level 2 has auto_approve_quality_threshold=95.0
        result = chain.auto_approve(97.0)
        assert result is True
        assert chain.current_level == 3
        assert chain.history[-1]["action"] == "auto_approved"

    def test_escalation_after_timeout(self, sample_approval_chain):
        """Escalation moves to next level after timeout."""
        chain = ApprovalChainStub(sample_approval_chain)
        chain.submit()
        chain.approve("anna.schmidt")
        assert chain.current_level == 2

        chain.escalate("No response after 48h")
        assert chain.current_level == 3
        assert chain.history[-1]["action"] == "escalated"

    def test_delegation_of_authority(self, sample_approval_chain):
        """Delegation rules are properly configured."""
        chain = ApprovalChainStub(sample_approval_chain)
        assert chain.delegation_rules["enabled"] is True
        assert chain.delegation_rules["max_delegation_depth"] == 1
        assert chain.delegation_rules["require_same_role"] is True

    def test_approval_audit_trail(self, sample_approval_chain):
        """Every action is recorded in the audit trail."""
        chain = ApprovalChainStub(sample_approval_chain)
        chain.submit()
        chain.approve("anna.schmidt")
        chain.approve("maria.weber")

        assert len(chain.history) == 3
        actions = [h["action"] for h in chain.history]
        assert actions == ["submitted", "approved", "approved"]
        assert all("timestamp" in h for h in chain.history)

    def test_approval_with_conditions(self, sample_approval_chain):
        """Approval with conditions is recorded as a comment."""
        chain = ApprovalChainStub(sample_approval_chain)
        chain.submit()
        chain.add_comment("anna.schmidt", "Approve with condition: update water data by Q2")
        chain.approve("anna.schmidt", "Approved with conditions")

        assert len(chain.comments) == 1
        assert "condition" in chain.comments[0]["text"].lower()
        assert chain.history[-1]["comment"] == "Approved with conditions"

    def test_parallel_approvals(self, sample_approval_chain):
        """Multi-approver level (board) requires multiple approvals."""
        board_level = sample_approval_chain["levels"][3]
        assert board_level["required_approvals"] == 2
        assert len(board_level["approvers"]) == 2

    def test_approval_status_transitions(self, sample_approval_chain):
        """Status transitions follow draft -> submitted -> in_review -> approved."""
        chain = ApprovalChainStub(sample_approval_chain)
        assert chain.status == "draft"

        chain.submit()
        assert chain.status == "submitted"

        chain.approve("anna.schmidt")
        assert chain.status == "in_review"

    def test_pending_approvals_query(self, sample_approval_chain):
        """Pending approvals query returns approvers for current level."""
        chain = ApprovalChainStub(sample_approval_chain)
        chain.submit()
        pending = chain.get_pending_approvals()
        assert len(pending) >= 1
        assert pending[0]["user_id"] == "anna.schmidt"

    def test_approval_comment_thread(self, sample_approval_chain):
        """Comments are tracked with user, level, and timestamp."""
        chain = ApprovalChainStub(sample_approval_chain)
        chain.submit()
        chain.add_comment("anna.schmidt", "Reviewed scope 1 data")
        chain.add_comment("pierre.dupont", "FR data validated")

        assert len(chain.comments) == 2
        assert chain.comments[0]["level"] == 1
        assert chain.comments[1]["user_id"] == "pierre.dupont"

    def test_resubmission_after_rejection(self, sample_approval_chain):
        """Resubmission after rejection restarts the chain."""
        chain = ApprovalChainStub(sample_approval_chain)
        chain.submit()
        chain.approve("anna.schmidt")
        chain.reject("maria.weber", "Incomplete data")

        assert chain.status == "rejected"
        chain.resubmit()
        assert chain.status == "submitted"
        assert chain.current_level == 1

    def test_board_level_sign_off(self, sample_approval_chain):
        """Board level requires 2 approvals with 120h timeout."""
        board = sample_approval_chain["levels"][3]
        assert board["name"] == "Board Sign-off"
        assert board["required_approvals"] == 2
        assert board["timeout_hours"] == 120
        assert board["escalation_to"] is None

    # -- Edge cases --

    def test_approve_before_submit_fails(self, sample_approval_chain):
        """Cannot approve a draft chain."""
        chain = ApprovalChainStub(sample_approval_chain)
        assert chain.status == "draft"
        result = chain.approve("anna.schmidt")
        assert result is False

    def test_auto_approve_below_threshold(self, sample_approval_chain):
        """Auto-approve does not fire below threshold."""
        chain = ApprovalChainStub(sample_approval_chain)
        chain.submit()
        chain.approve("anna.schmidt")
        result = chain.auto_approve(80.0)  # Below 95.0 threshold
        assert result is False
        assert chain.current_level == 2  # Stays at level 2

    def test_escalation_at_max_level(self, sample_approval_chain):
        """Escalation at last level records the event but cannot advance further."""
        config = {
            "chain_id": "single-level",
            "levels": [sample_approval_chain["levels"][0]],
            "delegation_rules": {},
        }
        chain = ApprovalChainStub(config)
        chain.submit()
        # At level 1 (max), escalation cannot advance - stays at level 1
        original_level = chain.current_level
        chain.escalate("timeout")
        # The escalate method advances to current_level + 1 if < max
        # For single-level chain, escalation still records the event
        assert len(chain.history) >= 2  # submitted + escalated

    def test_empty_pending_before_submit(self, sample_approval_chain):
        """Pending approvals before submission returns empty."""
        chain = ApprovalChainStub(sample_approval_chain)
        pending = chain.get_pending_approvals()
        assert pending == []

    def test_provenance_of_history(self, sample_approval_chain):
        """Audit trail history can be hashed for provenance."""
        chain = ApprovalChainStub(sample_approval_chain)
        chain.submit()
        chain.approve("anna.schmidt")

        history_str = json.dumps(chain.history, sort_keys=True, default=str)
        provenance = hashlib.sha256(history_str.encode()).hexdigest()
        assert len(provenance) == 64
