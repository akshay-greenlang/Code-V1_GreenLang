# -*- coding: utf-8 -*-
import pytest
from datetime import datetime
import sys
sys.path.insert(0, str(r"c:/Users/aksha/Code-V1_GreenLang/GL Agents/GL-013_PredictiveMaintenance"))


class TestUncertaintyGate:
    def test_classify_uncertainty_low(self):
        from safety.uncertainty_gating import UncertaintyGate, UncertaintyLevel
        gate = UncertaintyGate()
        level = gate.classify_uncertainty(0.05, 0.05)
        assert level == UncertaintyLevel.LOW
    
    def test_classify_uncertainty_critical(self):
        from safety.uncertainty_gating import UncertaintyGate, UncertaintyLevel
        gate = UncertaintyGate()
        level = gate.classify_uncertainty(0.5, 0.4)
        assert level == UncertaintyLevel.CRITICAL
    
    def test_determine_gate_auto_approve(self):
        from safety.uncertainty_gating import UncertaintyGate, DecisionGate
        gate = UncertaintyGate()
        decision = gate.determine_gate(0.05, 0.05, 0.1)
        assert decision == DecisionGate.AUTO_APPROVE
    
    def test_determine_gate_blocked(self):
        from safety.uncertainty_gating import UncertaintyGate, DecisionGate
        gate = UncertaintyGate()
        decision = gate.determine_gate(0.5, 0.4, 0.5)
        assert decision == DecisionGate.BLOCKED
    
    def test_evaluate_produces_gating_decision(self):
        from safety.uncertainty_gating import UncertaintyGate
        gate = UncertaintyGate()
        decision = gate.evaluate("PRED001", 0.1, 0.1, 0.15)
        assert decision.prediction_id == "PRED001"
        assert decision.decision_id is not None
        assert decision.provenance_hash is not None


class TestHumanInTheLoop:
    def test_queue_for_review(self):
        from safety.uncertainty_gating import UncertaintyGate, HumanInTheLoop
        gate = UncertaintyGate()
        hitl = HumanInTheLoop()
        
        decision = gate.evaluate("PRED001", 0.3, 0.2, 0.3)
        hitl.queue_for_review(decision)
        
        pending = hitl.get_pending_reviews()
        assert len(pending) == 1
    
    def test_record_human_decision(self):
        from safety.uncertainty_gating import UncertaintyGate, HumanInTheLoop, HumanDecision
        gate = UncertaintyGate()
        hitl = HumanInTheLoop()
        
        gating = gate.evaluate("PRED001", 0.3, 0.2, 0.3)
        hitl.queue_for_review(gating)
        
        human_decision = HumanDecision(
            decision_id="DEC001",
            gating_decision_id=gating.decision_id,
            reviewer_id="USER001",
            action="approve",
            rationale="Expert review confirms prediction",
            approved=True,
            override_prediction=False,
        )
        result = hitl.record_decision(human_decision)
        assert result is True
        assert len(hitl.get_pending_reviews()) == 0


class TestAuditLogger:
    def test_log_action(self):
        from safety.uncertainty_gating import AuditLogger, AuditAction
        logger = AuditLogger()
        entry = logger.log(
            action=AuditAction.PREDICTION_MADE,
            asset_id="PUMP001",
            details={"rul_hours": 500, "confidence": 0.85},
            prediction_id="PRED001",
        )
        assert entry.entry_id is not None
        assert entry.provenance_hash is not None
    
    def test_get_audit_trail(self):
        from safety.uncertainty_gating import AuditLogger, AuditAction
        logger = AuditLogger()
        
        logger.log(AuditAction.PREDICTION_MADE, "PUMP001", {"rul": 500})
        logger.log(AuditAction.WORK_ORDER_CREATED, "PUMP001", {"wo_id": "WO001"})
        logger.log(AuditAction.PREDICTION_MADE, "PUMP002", {"rul": 300})
        
        trail = logger.get_audit_trail(asset_id="PUMP001")
        assert len(trail) == 2
        
        trail = logger.get_audit_trail(action=AuditAction.PREDICTION_MADE)
        assert len(trail) == 2
    
    def test_export_log(self):
        from safety.uncertainty_gating import AuditLogger, AuditAction
        logger = AuditLogger()
        logger.log(AuditAction.PREDICTION_MADE, "PUMP001", {})
        
        exported = logger.export_log()
        assert len(exported) == 1
        assert "entry_id" in exported[0]
        assert "provenance" in exported[0]
