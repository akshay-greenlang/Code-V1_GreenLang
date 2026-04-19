"""
Immutable Evidence Pack Generator for GL-010 EMISSIONGUARDIAN
Supports 7-year retention per EPA/FDA regulations.
"""
from __future__ import annotations
import base64, gzip, hashlib, json, logging, uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

class EvidenceType(Enum):
    CALCULATION = "CALCULATION"
    SAFETY_CHECK = "SAFETY_CHECK"
    RECOMMENDATION = "RECOMMENDATION"
    CONFIGURATION_CHANGE = "CONFIGURATION_CHANGE"
    REGULATORY_SUBMISSION = "REGULATORY_SUBMISSION"
    OPTIMIZATION_RESULT = "OPTIMIZATION_RESULT"
    EMISSIONS_MONITORING = "EMISSIONS_MONITORING"
    NOX_ANALYSIS = "NOX_ANALYSIS"
    CO2_TRACKING = "CO2_TRACKING"
    COMPLIANCE_VERIFICATION = "COMPLIANCE_VERIFICATION"

class SealStatus(Enum):
    UNSEALED = "UNSEALED"
    SEALED = "SEALED"
    VERIFIED = "VERIFIED"
    TAMPERED = "TAMPERED"

class ExportFormat(Enum):
    JSON = "JSON"
    XML = "XML"
    PDF = "PDF"
    EPA_XML = "EPA_XML"
    FDA_XML = "FDA_XML"

@dataclass(frozen=True)
class EvidenceRecord:
    record_id: str
    timestamp: datetime
    evidence_type: EvidenceType
    data: Dict[str, Any]
    input_hash: str
    output_hash: str
    provenance_chain: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"record_id": self.record_id, "timestamp": self.timestamp.isoformat(),
                "evidence_type": self.evidence_type.value, "data": self.data,
                "input_hash": self.input_hash, "output_hash": self.output_hash,
                "provenance_chain": list(self.provenance_chain)}

    def compute_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()

class EvidencePackMetadata(BaseModel):
    pack_id: str
    agent_id: str = "GL-010"
    agent_name: str = "EMISSIONGUARDIAN"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    retention_years: int = 7
    retention_expires_at: Optional[datetime] = None
    record_count: int = 0
    seal_status: SealStatus = SealStatus.UNSEALED
    pack_version: str = "1.0.0"
    regulatory_frameworks: List[str] = Field(default_factory=lambda: ["EPA_40_CFR_60", "EPA_40_CFR_75", "EPA_40_CFR_98", "EU_ETS"])

class SealedPackEnvelope(BaseModel):
    pack_id: str
    sealed_at: datetime
    content_base64: str
    content_hash: str
    merkle_root: str
    seal_hash: str
    hash_chain: List[str] = Field(default_factory=list)
    algorithm: str = "SHA-256"
    sealer_id: str = "GL-010-SEALER"

class EvidencePackGenerator:
    """Immutable Evidence Pack Generator for GL-010 EMISSIONGUARDIAN."""

    def __init__(self, retention_years: int = 7):
        self.retention_years = retention_years
        self.agent_id = "GL-010"
        self.agent_name = "EMISSIONGUARDIAN"

    def create_evidence_pack(self, records: List[EvidenceRecord]) -> bytes:
        if not records: raise ValueError("Empty records")
        metadata = EvidencePackMetadata(pack_id=str(uuid.uuid4()), record_count=len(records))
        pack_data = {"metadata": metadata.dict(), "records": [r.to_dict() for r in records],
                     "hash_chain": self._build_hash_chain(records), "merkle_root": self._compute_merkle_root(records)}
        return gzip.compress(json.dumps(pack_data, sort_keys=True, default=str).encode())

    def seal_pack(self, pack: bytes) -> bytes:
        start_time = datetime.now(timezone.utc)
        decompressed = gzip.decompress(pack)
        pack_data = json.loads(decompressed)
        pack_id = pack_data["metadata"]["pack_id"]
        content_hash = hashlib.sha256(decompressed).hexdigest()
        seal_input = json.dumps({"pack_id": pack_id, "content_hash": content_hash, "merkle_root": pack_data.get("merkle_root", ""),
                                 "hash_chain": pack_data.get("hash_chain", []), "sealed_at": start_time.isoformat()}, sort_keys=True)
        seal_hash = hashlib.sha256(seal_input.encode()).hexdigest()
        envelope = SealedPackEnvelope(pack_id=pack_id, sealed_at=start_time, content_base64=base64.b64encode(pack).decode(),
                                      content_hash=content_hash, merkle_root=pack_data.get("merkle_root", ""), seal_hash=seal_hash, hash_chain=pack_data.get("hash_chain", []))
        return envelope.json(indent=2).encode()

    def verify_pack(self, sealed_pack: bytes) -> SealStatus:
        try:
            envelope = SealedPackEnvelope(**json.loads(sealed_pack))
            decompressed = gzip.decompress(base64.b64decode(envelope.content_base64))
            if hashlib.sha256(decompressed).hexdigest() != envelope.content_hash: return SealStatus.TAMPERED
            return SealStatus.VERIFIED
        except: return SealStatus.TAMPERED

    def export_regulatory_format(self, sealed_pack: bytes, format: ExportFormat) -> bytes:
        envelope = SealedPackEnvelope(**json.loads(sealed_pack))
        if format == ExportFormat.JSON: return sealed_pack
        elif format in [ExportFormat.XML, ExportFormat.EPA_XML, ExportFormat.FDA_XML]:
            root = ET.Element("EvidencePack", packId=envelope.pack_id)
            ET.SubElement(root, "SealHash").text = envelope.seal_hash
            return ET.tostring(root, encoding="utf-8", xml_declaration=True)
        return f"Pack ID: {envelope.pack_id}
Seal: {envelope.seal_hash}".encode()

    def _build_hash_chain(self, records):
        chain, prev = [], ""
        for r in records: link = hashlib.sha256(f"{prev}:{r.compute_hash()}".encode()).hexdigest(); chain.append(link); prev = link
        return chain

    def _compute_merkle_root(self, records):
        if not records: return hashlib.sha256(b"").hexdigest()
        leaves = [r.compute_hash() for r in records]
        while len(leaves) > 1:
            if len(leaves) % 2: leaves.append(leaves[-1])
            leaves = [hashlib.sha256((leaves[i] + leaves[i+1]).encode()).hexdigest() for i in range(0, len(leaves), 2)]
        return leaves[0]

    def create_evidence_record(self, evidence_type: EvidenceType, data: Dict, inputs: Dict, outputs: Dict, provenance_chain=None) -> EvidenceRecord:
        return EvidenceRecord(record_id=str(uuid.uuid4()), timestamp=datetime.now(timezone.utc), evidence_type=evidence_type, data=data,
                              input_hash=hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest(),
                              output_hash=hashlib.sha256(json.dumps(outputs, sort_keys=True, default=str).encode()).hexdigest(), provenance_chain=provenance_chain or [])
