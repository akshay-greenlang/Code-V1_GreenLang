# -*- coding: utf-8 -*-
"""GL-010 EmissionsGuardian - Chain of Custody Audit Module."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class RegulatoryFramework(str, Enum):
    EPA_40CFR75 = "epa_40cfr75"
    EU_ETS_MRV = "eu_ets_mrv"
    CARB = "carb"
    RGGI = "rggi"
    WCI = "wci"
    INTERNAL = "internal"


class CustodyAction(str, Enum):
    CREATED = "created"
    RECEIVED = "received"
    VALIDATED = "validated"
    CALCULATED = "calculated"
    TRANSFORMED = "transformed"
    TRANSFERRED = "transferred"
    AGGREGATED = "aggregated"
    REPORTED = "reported"
    ARCHIVED = "archived"
    CORRECTED = "corrected"
    SUBSTITUTED = "substituted"


class CustodianType(str, Enum):
    CEMS = "cems"
    DAS = "das"
    AGENT = "agent"
    OPERATOR = "operator"
    SYSTEM = "system"
    REGULATOR = "regulator"
    AUDITOR = "auditor"


class EmissionDataType(str, Enum):
    CEMS_HOURLY = "cems_hourly"
    CEMS_DAILY = "cems_daily"
    FUEL_FLOW = "fuel_flow"
    STACK_FLOW = "stack_flow"
    EMISSION_RATE = "emission_rate"
    QUARTERLY_REPORT = "quarterly_report"
    ANNUAL_REPORT = "annual_report"
    VERIFICATION_STATEMENT = "verification"


class ChainIntegrity(str, Enum):
    VALID = "valid"
    BROKEN = "broken"
    TAMPERED = "tampered"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"


def compute_sha256(data: Union[str, bytes, Dict[str, Any]]) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, default=str)
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def generate_entry_id() -> str:
    return f"COC-{uuid.uuid4().hex[:12].upper()}"


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CustodyEntry:
    entry_id: str
    timestamp: datetime
    action: CustodyAction
    custodian_id: str
    custodian_type: CustodianType
    data_type: EmissionDataType
    data_reference: str
    data_hash: str
    previous_hash: str
    entry_hash: str = field(default="")
    metadata: Dict[str, Any] = field(default_factory=dict)
    regulatory_framework: RegulatoryFramework = RegulatoryFramework.EPA_40CFR75
    facility_id: str = ""
    unit_id: Optional[str] = None

    def __post_init__(self):
        if not self.entry_hash:
            self.entry_hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        hash_content = {
            "entry_id": self.entry_id, "timestamp": self.timestamp.isoformat(),
            "action": self.action.value, "custodian_id": self.custodian_id,
            "custodian_type": self.custodian_type.value, "data_type": self.data_type.value,
            "data_reference": self.data_reference, "data_hash": self.data_hash,
            "previous_hash": self.previous_hash, "metadata": self.metadata,
            "regulatory_framework": self.regulatory_framework.value,
            "facility_id": self.facility_id, "unit_id": self.unit_id,
        }
        return compute_sha256(hash_content)

    def verify_hash(self) -> bool:
        return self.entry_hash == self.calculate_hash()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id, "timestamp": self.timestamp.isoformat(),
            "action": self.action.value, "custodian_id": self.custodian_id,
            "custodian_type": self.custodian_type.value, "data_type": self.data_type.value,
            "data_reference": self.data_reference, "data_hash": self.data_hash,
            "previous_hash": self.previous_hash, "entry_hash": self.entry_hash,
            "metadata": self.metadata, "regulatory_framework": self.regulatory_framework.value,
            "facility_id": self.facility_id, "unit_id": self.unit_id,
        }


@dataclass
class CustodyTransfer:
    transfer_id: str
    timestamp: datetime
    from_custodian: str
    from_type: CustodianType
    to_custodian: str
    to_type: CustodianType
    data_references: List[str]
    transfer_hash: str = ""
    acknowledgment_hash: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    def __post_init__(self):
        if not self.transfer_hash:
            content = {
                "transfer_id": self.transfer_id, "timestamp": self.timestamp.isoformat(),
                "from_custodian": self.from_custodian, "from_type": self.from_type.value,
                "to_custodian": self.to_custodian, "to_type": self.to_type.value,
                "data_references": sorted(self.data_references),
            }
            self.transfer_hash = compute_sha256(content)

    def acknowledge(self, acknowledgment_hash: str) -> None:
        self.acknowledgment_hash = acknowledgment_hash
        self.acknowledged_at = get_utc_now()

    def is_acknowledged(self) -> bool:
        return self.acknowledgment_hash is not None


@dataclass
class DataLineageNode:
    node_id: str
    data_reference: str
    data_hash: str
    source_nodes: List[str] = field(default_factory=list)
    transformation: str = ""
    agent_id: str = ""
    timestamp: datetime = field(default_factory=get_utc_now)
    input_hashes: List[str] = field(default_factory=list)
    output_hash: str = ""
    formula_reference: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id, "data_reference": self.data_reference,
            "data_hash": self.data_hash, "source_nodes": self.source_nodes,
            "transformation": self.transformation, "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(), "input_hashes": self.input_hashes,
            "output_hash": self.output_hash, "formula_reference": self.formula_reference,
        }


@dataclass
class ChainVerificationResult:
    is_valid: bool
    integrity: ChainIntegrity
    verified_entries: int
    first_invalid_entry: Optional[str] = None
    verification_timestamp: datetime = field(default_factory=get_utc_now)
    verification_hash: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.verification_hash:
            content = {"is_valid": self.is_valid, "integrity": self.integrity.value,
                       "verified_entries": self.verified_entries,
                       "first_invalid_entry": self.first_invalid_entry,
                       "timestamp": self.verification_timestamp.isoformat()}
            self.verification_hash = compute_sha256(content)


class ImmutableAuditLog:
    GENESIS_HASH = "0" * 64

    def __init__(self, facility_id: str,
                 regulatory_framework: RegulatoryFramework = RegulatoryFramework.EPA_40CFR75):
        self.facility_id = facility_id
        self.regulatory_framework = regulatory_framework
        self._entries: List[CustodyEntry] = []
        self._transfers: List[CustodyTransfer] = []
        self._lineage_graph: Dict[str, DataLineageNode] = {}
        self._entry_index: Dict[str, int] = {}

    @property
    def entries(self) -> List[CustodyEntry]:
        return list(self._entries)

    @property
    def transfers(self) -> List[CustodyTransfer]:
        return list(self._transfers)

    @property
    def lineage_graph(self) -> Dict[str, DataLineageNode]:
        return dict(self._lineage_graph)

    def _get_previous_hash(self) -> str:
        return self._entries[-1].entry_hash if self._entries else self.GENESIS_HASH

    def record_custody(self, action: CustodyAction, custodian_id: str,
                       custodian_type: CustodianType, data_type: EmissionDataType,
                       data_reference: str, data_content: Any,
                       unit_id: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> CustodyEntry:
        entry = CustodyEntry(
            entry_id=generate_entry_id(), timestamp=get_utc_now(), action=action,
            custodian_id=custodian_id, custodian_type=custodian_type,
            data_type=data_type, data_reference=data_reference,
            data_hash=compute_sha256(data_content), previous_hash=self._get_previous_hash(),
            metadata=metadata or {}, regulatory_framework=self.regulatory_framework,
            facility_id=self.facility_id, unit_id=unit_id,
        )
        self._entries.append(entry)
        self._entry_index[entry.entry_id] = len(self._entries) - 1
        return entry

    def record_calculation(self, custodian_id: str, input_references: List[str],
                           input_data: List[Any], output_reference: str, output_data: Any,
                           formula_reference: str, unit_id: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> Tuple[CustodyEntry, DataLineageNode]:
        input_hashes = [compute_sha256(d) for d in input_data]
        output_hash = compute_sha256(output_data)
        source_nodes = [r for r in input_references if r in self._lineage_graph]

        lineage_node = DataLineageNode(
            node_id=f"LIN-{uuid.uuid4().hex[:12].upper()}", data_reference=output_reference,
            data_hash=output_hash, source_nodes=source_nodes,
            transformation=f"Calculation using {formula_reference}", agent_id=custodian_id,
            input_hashes=input_hashes, output_hash=output_hash, formula_reference=formula_reference,
        )
        self._lineage_graph[output_reference] = lineage_node

        calc_metadata = {**(metadata or {}), "formula_reference": formula_reference,
                         "input_references": input_references, "input_hashes": input_hashes,
                         "lineage_node_id": lineage_node.node_id}

        entry = self.record_custody(
            action=CustodyAction.CALCULATED, custodian_id=custodian_id,
            custodian_type=CustodianType.AGENT, data_type=EmissionDataType.EMISSION_RATE,
            data_reference=output_reference, data_content=output_data,
            unit_id=unit_id, metadata=calc_metadata,
        )
        return entry, lineage_node

    def record_transfer(self, from_custodian: str, from_type: CustodianType,
                        to_custodian: str, to_type: CustodianType,
                        data_references: List[str]) -> CustodyTransfer:
        transfer = CustodyTransfer(
            transfer_id=f"XFR-{uuid.uuid4().hex[:12].upper()}", timestamp=get_utc_now(),
            from_custodian=from_custodian, from_type=from_type,
            to_custodian=to_custodian, to_type=to_type, data_references=data_references,
        )
        self._transfers.append(transfer)

        for ref in data_references:
            self.record_custody(
                action=CustodyAction.TRANSFERRED, custodian_id=from_custodian,
                custodian_type=from_type, data_type=EmissionDataType.CEMS_HOURLY,
                data_reference=ref, data_content={"transfer_id": transfer.transfer_id, "direction": "out"},
                metadata={"transfer_to": to_custodian},
            )
            self.record_custody(
                action=CustodyAction.RECEIVED, custodian_id=to_custodian,
                custodian_type=to_type, data_type=EmissionDataType.CEMS_HOURLY,
                data_reference=ref, data_content={"transfer_id": transfer.transfer_id, "direction": "in"},
                metadata={"transfer_from": from_custodian},
            )
        return transfer

    def verify_chain(self) -> ChainVerificationResult:
        if not self._entries:
            return ChainVerificationResult(is_valid=True, integrity=ChainIntegrity.VALID,
                                           verified_entries=0, details={"message": "Empty chain is valid"})
        expected_previous = self.GENESIS_HASH
        for i, entry in enumerate(self._entries):
            if not entry.verify_hash():
                return ChainVerificationResult(is_valid=False, integrity=ChainIntegrity.TAMPERED,
                                               verified_entries=i, first_invalid_entry=entry.entry_id,
                                               details={"error": "Entry hash mismatch"})
            if entry.previous_hash != expected_previous:
                return ChainVerificationResult(is_valid=False, integrity=ChainIntegrity.BROKEN,
                                               verified_entries=i, first_invalid_entry=entry.entry_id,
                                               details={"error": "Chain link broken"})
            expected_previous = entry.entry_hash
        return ChainVerificationResult(is_valid=True, integrity=ChainIntegrity.VALID,
                                       verified_entries=len(self._entries),
                                       details={"message": f"All {len(self._entries)} entries verified"})

    def detect_tampering(self) -> Tuple[bool, Optional[str]]:
        result = self.verify_chain()
        return (False, None) if result.is_valid else (True, result.first_invalid_entry)

    def get_lineage(self, data_reference: str) -> List[DataLineageNode]:
        result, visited = [], set()
        def trace(ref: str) -> None:
            if ref in visited: return
            visited.add(ref)
            if ref in self._lineage_graph:
                node = self._lineage_graph[ref]
                for src in node.source_nodes: trace(src)
                result.append(node)
        trace(data_reference)
        return result

    def export_for_regulator(self, framework: RegulatoryFramework,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> Dict[str, Any]:
        entries = self._entries
        if start_date: entries = [e for e in entries if e.timestamp >= start_date]
        if end_date: entries = [e for e in entries if e.timestamp <= end_date]
        verification = self.verify_chain()
        export_data = {
            "export_timestamp": get_utc_now().isoformat(), "facility_id": self.facility_id,
            "regulatory_framework": framework.value,
            "chain_verification": {"is_valid": verification.is_valid, "integrity": verification.integrity.value,
                                   "verified_entries": verification.verified_entries,
                                   "verification_hash": verification.verification_hash},
            "date_range": {"start": start_date.isoformat() if start_date else None,
                          "end": end_date.isoformat() if end_date else None},
            "entries": [e.to_dict() for e in entries],
            "transfers": [{"transfer_id": t.transfer_id, "timestamp": t.timestamp.isoformat(),
                          "from_custodian": t.from_custodian, "to_custodian": t.to_custodian,
                          "data_references": t.data_references, "acknowledged": t.is_acknowledged()}
                         for t in self._transfers],
            "lineage_graph": {ref: node.to_dict() for ref, node in self._lineage_graph.items()},
        }
        if framework == RegulatoryFramework.EPA_40CFR75:
            export_data["epa_specific"] = {"retention_requirement_years": 3,
                                           "data_validation_status": "PASSED" if verification.is_valid else "FAILED",
                                           "certification_statement": "Data collected per EPA 40 CFR Part 75."}
        elif framework == RegulatoryFramework.EU_ETS_MRV:
            export_data["eu_ets_specific"] = {"mrv_compliance": True, "verification_level": "REASONABLE_ASSURANCE",
                                              "materiality_threshold_percent": 5.0}
        return export_data

    def get_entries_by_action(self, action: CustodyAction) -> List[CustodyEntry]:
        return [e for e in self._entries if e.action == action]

    def get_entries_by_custodian(self, custodian_id: str) -> List[CustodyEntry]:
        return [e for e in self._entries if e.custodian_id == custodian_id]

    def get_entry_by_id(self, entry_id: str) -> Optional[CustodyEntry]:
        idx = self._entry_index.get(entry_id)
        return self._entries[idx] if idx is not None else None


class EPACompliancePackager:
    def __init__(self, audit_log: ImmutableAuditLog):
        self.audit_log = audit_log

    def package_quarterly_report(self, quarter: int, year: int) -> Dict[str, Any]:
        starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
        ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
        sm, sd = starts[quarter]
        em, ed = ends[quarter]
        start = datetime(year, sm, sd, tzinfo=timezone.utc)
        end = datetime(year, em, ed, 23, 59, 59, tzinfo=timezone.utc)
        export = self.audit_log.export_for_regulator(RegulatoryFramework.EPA_40CFR75, start, end)
        export.update({"report_type": "QUARTERLY", "quarter": quarter, "year": year})
        return export

    def package_annual_certification(self, year: int) -> Dict[str, Any]:
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        export = self.audit_log.export_for_regulator(RegulatoryFramework.EPA_40CFR75, start, end)
        export.update({"report_type": "ANNUAL_CERTIFICATION", "year": year,
                      "certification": {"certified_by": None, "certification_date": None,
                                        "statement": "Data recorded per Federal requirements."}})
        return export


class EUETSCompliancePackager:
    def __init__(self, audit_log: ImmutableAuditLog):
        self.audit_log = audit_log

    def package_annual_verification(self, year: int) -> Dict[str, Any]:
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        export = self.audit_log.export_for_regulator(RegulatoryFramework.EU_ETS_MRV, start, end)
        export.update({"report_type": "ANNUAL_VERIFICATION", "year": year,
                      "mrv_requirements": {"monitoring_plan_version": None, "tier_level": None,
                                           "uncertainty_assessment": None, "improvement_recommendations": []},
                      "verifier_info": {"verifier_name": None, "accreditation_number": None,
                                        "verification_date": None, "opinion": None}})
        return export


def create_audit_log(facility_id: str,
                     framework: RegulatoryFramework = RegulatoryFramework.EPA_40CFR75) -> ImmutableAuditLog:
    return ImmutableAuditLog(facility_id, framework)


def create_epa_packager(audit_log: ImmutableAuditLog) -> EPACompliancePackager:
    return EPACompliancePackager(audit_log)


def create_eu_ets_packager(audit_log: ImmutableAuditLog) -> EUETSCompliancePackager:
    return EUETSCompliancePackager(audit_log)


__all__ = [
    "RegulatoryFramework", "CustodyAction", "CustodianType", "EmissionDataType", "ChainIntegrity",
    "CustodyEntry", "CustodyTransfer", "DataLineageNode", "ChainVerificationResult",
    "ImmutableAuditLog", "EPACompliancePackager", "EUETSCompliancePackager",
    "create_audit_log", "create_epa_packager", "create_eu_ets_packager",
    "compute_sha256", "generate_entry_id", "get_utc_now",
]
