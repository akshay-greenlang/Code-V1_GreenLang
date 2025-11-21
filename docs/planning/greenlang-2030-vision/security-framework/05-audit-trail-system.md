# Audit Trail System

## 1. Immutable Logging Architecture

### Logging Infrastructure

```yaml
# audit-logging-config.yaml
audit_system:
  architecture:
    collectors:
      - type: "fluentd"
        deployment: "daemonset"
        config:
          inputs:
            - application_logs
            - system_logs
            - security_events
            - api_calls
          outputs:
            - kafka_topics
            - s3_archive
            - blockchain_anchor

    pipeline:
      ingestion:
        - fluentd_collectors
        - kafka_stream
        - stream_processing

      processing:
        - enrichment
        - normalization
        - correlation
        - anomaly_detection

      storage:
        - hot_storage: "elasticsearch"
        - warm_storage: "s3_infrequent"
        - cold_storage: "glacier"
        - immutable_ledger: "blockchain"

  immutability:
    write_once_storage:
      enabled: true
      compliance_mode: true
      retention_days: 2555  # 7 years

    cryptographic_signing:
      algorithm: "SHA256withRSA"
      key_rotation: "annual"
      timestamp_authority: "RFC3161"

    blockchain_anchoring:
      enabled: true
      chain: "ethereum"
      interval: "hourly"
      merkle_tree: true
```

### Immutable Event Store

```python
# immutable_event_store.py
import hashlib
import json
import time
from typing import Dict, List, Optional
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.x509 import load_pem_x509_certificate

class ImmutableEventStore:
    def __init__(self, storage_backend):
        self.storage = storage_backend
        self.chain_head = None
        self.private_key = self.load_signing_key()
        self.certificate = self.load_certificate()

    def log_event(self, event: Dict) -> str:
        """Log an immutable audit event"""
        # Create event record
        event_record = {
            "event_id": self.generate_event_id(),
            "timestamp": time.time(),
            "timestamp_rfc3339": self.get_rfc3339_timestamp(),
            "event_type": event.get("type"),
            "actor": event.get("actor"),
            "resource": event.get("resource"),
            "action": event.get("action"),
            "result": event.get("result"),
            "metadata": event.get("metadata", {}),
            "previous_hash": self.get_chain_head(),
            "nonce": self.generate_nonce()
        }

        # Calculate event hash
        event_hash = self.calculate_hash(event_record)
        event_record["hash"] = event_hash

        # Sign the event
        signature = self.sign_event(event_record)
        event_record["signature"] = signature

        # Store immutably
        storage_location = self.store_immutable(event_record)

        # Update chain head
        self.chain_head = event_hash

        # Anchor to blockchain periodically
        if self.should_anchor():
            self.anchor_to_blockchain(event_hash)

        return event_hash

    def calculate_hash(self, event: Dict) -> str:
        """Calculate cryptographic hash of event"""
        # Remove signature field if present
        event_copy = {k: v for k, v in event.items() if k != "signature"}

        # Canonical JSON serialization
        canonical = json.dumps(event_copy, sort_keys=True)

        # Calculate SHA-256 hash
        hash_obj = hashlib.sha256(canonical.encode())
        return hash_obj.hexdigest()

    def sign_event(self, event: Dict) -> str:
        """Digitally sign the event"""
        message = json.dumps(event, sort_keys=True).encode()

        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        return signature.hex()

    def verify_event(self, event: Dict) -> bool:
        """Verify event integrity and signature"""
        # Verify hash
        calculated_hash = self.calculate_hash(event)
        if calculated_hash != event.get("hash"):
            return False

        # Verify signature
        signature = bytes.fromhex(event["signature"])
        event_copy = {k: v for k, v in event.items() if k != "signature"}
        message = json.dumps(event_copy, sort_keys=True).encode()

        try:
            public_key = self.certificate.public_key()
            public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except:
            return False

    def store_immutable(self, event: Dict) -> str:
        """Store event in immutable storage"""
        # Write to WORM storage
        location = self.storage.write_once(event)

        # Set compliance lock
        self.storage.set_compliance_lock(location, days=2555)

        return location

    def query_events(self, filters: Dict) -> List[Dict]:
        """Query audit events with integrity verification"""
        events = self.storage.query(filters)

        # Verify each event
        verified_events = []
        for event in events:
            if self.verify_event(event):
                verified_events.append(event)
            else:
                # Log tampering attempt
                self.log_tampering_detected(event)

        return verified_events

    def export_for_compliance(self, start_date: str, end_date: str) -> Dict:
        """Export audit trail for compliance review"""
        events = self.query_events({
            "timestamp": {"$gte": start_date, "$lte": end_date}
        })

        return {
            "export_date": self.get_rfc3339_timestamp(),
            "period": {"start": start_date, "end": end_date},
            "total_events": len(events),
            "events": events,
            "integrity_proof": self.generate_integrity_proof(events),
            "certificate": self.certificate.public_bytes(
                encoding=serialization.Encoding.PEM
            ).decode()
        }

    def generate_integrity_proof(self, events: List[Dict]) -> Dict:
        """Generate cryptographic proof of audit trail integrity"""
        # Build Merkle tree
        merkle_tree = self.build_merkle_tree(events)

        return {
            "merkle_root": merkle_tree["root"],
            "blockchain_anchors": self.get_blockchain_anchors(events),
            "timestamp_tokens": self.get_timestamp_tokens(events),
            "certificate_chain": self.get_certificate_chain()
        }
```

## 2. Event Tracking Configuration

### Comprehensive Event Schema

```yaml
# event-schema.yaml
event_categories:
  authentication:
    events:
      - login_attempt:
          fields:
            - user_id
            - method
            - ip_address
            - user_agent
            - success
            - mfa_used
            - failure_reason

      - logout:
          fields:
            - user_id
            - session_id
            - duration
            - ip_address

      - password_change:
          fields:
            - user_id
            - requester_id
            - method
            - ip_address

      - mfa_update:
          fields:
            - user_id
            - action
            - method
            - ip_address

  authorization:
    events:
      - permission_granted:
          fields:
            - user_id
            - resource
            - permission
            - context

      - permission_denied:
          fields:
            - user_id
            - resource
            - permission
            - reason

      - role_assignment:
          fields:
            - user_id
            - role
            - assigned_by
            - expiration

  data_access:
    events:
      - data_read:
          fields:
            - user_id
            - resource_type
            - resource_id
            - fields_accessed
            - purpose

      - data_write:
          fields:
            - user_id
            - resource_type
            - resource_id
            - fields_modified
            - old_values
            - new_values

      - data_delete:
          fields:
            - user_id
            - resource_type
            - resource_id
            - reason
            - approval_id

      - data_export:
          fields:
            - user_id
            - export_type
            - filters
            - row_count
            - destination

  api_activity:
    events:
      - api_call:
          fields:
            - user_id
            - endpoint
            - method
            - parameters
            - response_code
            - duration_ms
            - ip_address

      - rate_limit_exceeded:
          fields:
            - user_id
            - endpoint
            - limit
            - window

      - api_error:
          fields:
            - user_id
            - endpoint
            - error_code
            - error_message
            - stack_trace

  security_events:
    events:
      - suspicious_activity:
          fields:
            - type
            - actor
            - details
            - risk_score
            - action_taken

      - security_scan:
          fields:
            - scan_type
            - target
            - findings
            - severity

      - incident_created:
          fields:
            - incident_id
            - severity
            - description
            - affected_resources

  compliance_events:
    events:
      - consent_granted:
          fields:
            - user_id
            - purpose
            - scope
            - expiration

      - consent_withdrawn:
          fields:
            - user_id
            - purpose
            - reason

      - data_retention:
          fields:
            - action
            - data_type
            - count
            - reason

      - audit_performed:
          fields:
            - audit_type
            - scope
            - findings
            - auditor
```

### Event Collector Implementation

```python
# event_collector.py
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import structlog

class EventCollector:
    def __init__(self, event_store, enrichment_service):
        self.event_store = event_store
        self.enrichment = enrichment_service
        self.logger = structlog.get_logger()
        self.event_queue = asyncio.Queue()

    async def collect_event(
        self,
        event_type: str,
        actor: str,
        resource: str,
        action: str,
        result: str,
        metadata: Optional[Dict] = None
    ):
        """Collect and process an audit event"""
        event = {
            "type": event_type,
            "actor": actor,
            "resource": resource,
            "action": action,
            "result": result,
            "metadata": metadata or {},
            "correlation_id": self.generate_correlation_id(),
            "session_id": self.get_session_id(),
            "request_id": self.get_request_id()
        }

        # Enrich event
        enriched = await self.enrich_event(event)

        # Apply privacy filters
        filtered = self.apply_privacy_filters(enriched)

        # Queue for processing
        await self.event_queue.put(filtered)

        # Log locally
        self.logger.info(
            "audit_event",
            event_type=event_type,
            actor=actor,
            resource=resource,
            action=action,
            result=result
        )

        return event["correlation_id"]

    async def enrich_event(self, event: Dict) -> Dict:
        """Enrich event with additional context"""
        enriched = event.copy()

        # Add geographic information
        if "ip_address" in event["metadata"]:
            geo = await self.enrichment.get_geo_location(
                event["metadata"]["ip_address"]
            )
            enriched["metadata"]["geo_location"] = geo

        # Add user context
        user_context = await self.enrichment.get_user_context(event["actor"])
        enriched["metadata"]["user_context"] = user_context

        # Add system context
        enriched["metadata"]["system_context"] = {
            "environment": self.get_environment(),
            "version": self.get_application_version(),
            "host": self.get_hostname(),
            "container_id": self.get_container_id()
        }

        # Add threat intelligence
        if self.is_security_event(event):
            threat_data = await self.enrichment.get_threat_intelligence(event)
            enriched["metadata"]["threat_intelligence"] = threat_data

        return enriched

    def apply_privacy_filters(self, event: Dict) -> Dict:
        """Apply privacy filters to protect sensitive data"""
        filtered = event.copy()

        # Mask PII fields
        pii_fields = ["ssn", "credit_card", "bank_account", "medical_record"]
        for field in pii_fields:
            if field in filtered["metadata"]:
                filtered["metadata"][field] = self.mask_sensitive_data(
                    filtered["metadata"][field]
                )

        # Hash user identifiers for privacy
        if self.should_anonymize(event):
            filtered["actor"] = self.hash_identifier(filtered["actor"])

        # Redact sensitive values
        if "old_values" in filtered["metadata"]:
            filtered["metadata"]["old_values"] = self.redact_sensitive_values(
                filtered["metadata"]["old_values"]
            )

        if "new_values" in filtered["metadata"]:
            filtered["metadata"]["new_values"] = self.redact_sensitive_values(
                filtered["metadata"]["new_values"]
            )

        return filtered

    async def process_queue(self):
        """Process queued events"""
        batch = []
        while True:
            try:
                # Collect batch
                while len(batch) < 100:
                    event = await asyncio.wait_for(
                        self.event_queue.get(),
                        timeout=1.0
                    )
                    batch.append(event)

            except asyncio.TimeoutError:
                # Process batch if any events collected
                if batch:
                    await self.process_batch(batch)
                    batch = []

    async def process_batch(self, events: List[Dict]):
        """Process a batch of events"""
        # Store in event store
        for event in events:
            try:
                await self.event_store.log_event(event)
            except Exception as e:
                self.logger.error("Failed to store event", error=str(e))

        # Send to SIEM
        await self.send_to_siem(events)

        # Check for anomalies
        anomalies = await self.detect_anomalies(events)
        if anomalies:
            await self.handle_anomalies(anomalies)

        # Update metrics
        self.update_metrics(events)
```

## 3. Blockchain-based Verification

### Blockchain Anchoring Service

```python
# blockchain_anchor.py
from web3 import Web3
from typing import Dict, List, Optional
import hashlib
import json

class BlockchainAnchor:
    def __init__(self, config: Dict):
        self.w3 = Web3(Web3.HTTPProvider(config["ethereum_node"]))
        self.contract_address = config["contract_address"]
        self.contract = self.load_contract()
        self.account = config["account_address"]
        self.private_key = config["private_key"]

    def load_contract(self):
        """Load audit trail smart contract"""
        abi = """[
            {
                "name": "anchorHash",
                "type": "function",
                "inputs": [
                    {"name": "_hash", "type": "bytes32"},
                    {"name": "_metadata", "type": "string"}
                ],
                "outputs": [{"name": "", "type": "uint256"}]
            },
            {
                "name": "verifyHash",
                "type": "function",
                "inputs": [
                    {"name": "_hash", "type": "bytes32"},
                    {"name": "_blockNumber", "type": "uint256"}
                ],
                "outputs": [{"name": "", "type": "bool"}]
            },
            {
                "name": "getAnchor",
                "type": "function",
                "inputs": [{"name": "_hash", "type": "bytes32"}],
                "outputs": [
                    {"name": "blockNumber", "type": "uint256"},
                    {"name": "timestamp", "type": "uint256"},
                    {"name": "metadata", "type": "string"}
                ]
            }
        ]"""

        return self.w3.eth.contract(
            address=self.contract_address,
            abi=json.loads(abi)
        )

    def anchor_merkle_root(self, merkle_root: str, event_hashes: List[str]) -> str:
        """Anchor Merkle root to blockchain"""
        metadata = {
            "merkle_root": merkle_root,
            "event_count": len(event_hashes),
            "timestamp": datetime.now().isoformat(),
            "first_event": event_hashes[0] if event_hashes else None,
            "last_event": event_hashes[-1] if event_hashes else None
        }

        # Prepare transaction
        nonce = self.w3.eth.get_transaction_count(self.account)

        transaction = self.contract.functions.anchorHash(
            bytes.fromhex(merkle_root),
            json.dumps(metadata)
        ).build_transaction({
            'from': self.account,
            'nonce': nonce,
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })

        # Sign and send transaction
        signed_txn = self.w3.eth.account.sign_transaction(
            transaction,
            private_key=self.private_key
        )

        tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)

        # Wait for confirmation
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

        return {
            "transaction_hash": receipt["transactionHash"].hex(),
            "block_number": receipt["blockNumber"],
            "merkle_root": merkle_root,
            "gas_used": receipt["gasUsed"]
        }

    def verify_anchor(self, merkle_root: str, block_number: int) -> bool:
        """Verify Merkle root anchor on blockchain"""
        try:
            result = self.contract.functions.verifyHash(
                bytes.fromhex(merkle_root),
                block_number
            ).call()
            return result
        except:
            return False

    def build_merkle_tree(self, events: List[Dict]) -> Dict:
        """Build Merkle tree from events"""
        # Get event hashes
        hashes = [event["hash"] for event in events]

        # Build tree
        tree = MerkleTree(hashes)

        return {
            "root": tree.get_root(),
            "leaves": hashes,
            "tree_height": tree.get_height(),
            "proof_for_event": lambda event_hash: tree.get_proof(event_hash)
        }

class MerkleTree:
    def __init__(self, hashes: List[str]):
        self.leaves = hashes
        self.tree = self.build_tree(hashes)

    def build_tree(self, hashes: List[str]) -> List[List[str]]:
        """Build complete Merkle tree"""
        if not hashes:
            return [[]]

        tree = [hashes]

        while len(tree[-1]) > 1:
            level = []
            previous_level = tree[-1]

            for i in range(0, len(previous_level), 2):
                if i + 1 < len(previous_level):
                    combined = previous_level[i] + previous_level[i + 1]
                else:
                    combined = previous_level[i] + previous_level[i]

                hash_obj = hashlib.sha256(combined.encode())
                level.append(hash_obj.hexdigest())

            tree.append(level)

        return tree

    def get_root(self) -> str:
        """Get Merkle root"""
        if self.tree and self.tree[-1]:
            return self.tree[-1][0]
        return ""

    def get_proof(self, leaf: str) -> List[Dict]:
        """Get Merkle proof for a leaf"""
        if leaf not in self.leaves:
            return []

        proof = []
        index = self.leaves.index(leaf)

        for level in self.tree[:-1]:
            if index % 2 == 0:
                # Right sibling
                if index + 1 < len(level):
                    proof.append({
                        "position": "right",
                        "hash": level[index + 1]
                    })
            else:
                # Left sibling
                proof.append({
                    "position": "left",
                    "hash": level[index - 1]
                })

            index = index // 2

        return proof

    def verify_proof(self, leaf: str, proof: List[Dict], root: str) -> bool:
        """Verify Merkle proof"""
        current = leaf

        for step in proof:
            if step["position"] == "left":
                combined = step["hash"] + current
            else:
                combined = current + step["hash"]

            hash_obj = hashlib.sha256(combined.encode())
            current = hash_obj.hexdigest()

        return current == root
```

### Smart Contract for Audit Trail

```solidity
// AuditTrail.sol
pragma solidity ^0.8.0;

contract AuditTrail {
    struct Anchor {
        uint256 blockNumber;
        uint256 timestamp;
        string metadata;
        address creator;
    }

    mapping(bytes32 => Anchor) public anchors;
    mapping(address => bool) public authorizedAnchors;

    event HashAnchored(
        bytes32 indexed hash,
        uint256 blockNumber,
        uint256 timestamp,
        address indexed creator
    );

    modifier onlyAuthorized() {
        require(
            authorizedAnchors[msg.sender],
            "Unauthorized anchor address"
        );
        _;
    }

    constructor() {
        authorizedAnchors[msg.sender] = true;
    }

    function anchorHash(
        bytes32 _hash,
        string memory _metadata
    ) public onlyAuthorized returns (uint256) {
        require(
            anchors[_hash].blockNumber == 0,
            "Hash already anchored"
        );

        anchors[_hash] = Anchor({
            blockNumber: block.number,
            timestamp: block.timestamp,
            metadata: _metadata,
            creator: msg.sender
        });

        emit HashAnchored(_hash, block.number, block.timestamp, msg.sender);

        return block.number;
    }

    function verifyHash(
        bytes32 _hash,
        uint256 _blockNumber
    ) public view returns (bool) {
        return anchors[_hash].blockNumber == _blockNumber;
    }

    function getAnchor(bytes32 _hash) public view returns (
        uint256 blockNumber,
        uint256 timestamp,
        string memory metadata,
        address creator
    ) {
        Anchor memory anchor = anchors[_hash];
        return (
            anchor.blockNumber,
            anchor.timestamp,
            anchor.metadata,
            anchor.creator
        );
    }

    function addAuthorizedAnchor(address _address) public onlyAuthorized {
        authorizedAnchors[_address] = true;
    }

    function removeAuthorizedAnchor(address _address) public onlyAuthorized {
        authorizedAnchors[_address] = false;
    }

    function batchAnchor(
        bytes32[] memory _hashes,
        string[] memory _metadata
    ) public onlyAuthorized {
        require(
            _hashes.length == _metadata.length,
            "Array length mismatch"
        );

        for (uint i = 0; i < _hashes.length; i++) {
            if (anchors[_hashes[i]].blockNumber == 0) {
                anchors[_hashes[i]] = Anchor({
                    blockNumber: block.number,
                    timestamp: block.timestamp,
                    metadata: _metadata[i],
                    creator: msg.sender
                });

                emit HashAnchored(
                    _hashes[i],
                    block.number,
                    block.timestamp,
                    msg.sender
                );
            }
        }
    }
}
```

## 4. Compliance Reporting

### Compliance Report Generator

```python
# compliance_reporting.py
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta

class ComplianceReporter:
    def __init__(self, event_store, report_templates):
        self.event_store = event_store
        self.templates = report_templates

    def generate_soc2_report(self, period: Dict) -> Dict:
        """Generate SOC 2 compliance report"""
        events = self.event_store.query_events({
            "timestamp": {
                "$gte": period["start"],
                "$lte": period["end"]
            }
        })

        return {
            "report_type": "SOC 2 Type II",
            "period": period,
            "generated_at": datetime.now().isoformat(),
            "trust_services_criteria": {
                "security": self.analyze_security_controls(events),
                "availability": self.analyze_availability_controls(events),
                "processing_integrity": self.analyze_processing_integrity(events),
                "confidentiality": self.analyze_confidentiality_controls(events),
                "privacy": self.analyze_privacy_controls(events)
            },
            "control_effectiveness": self.calculate_control_effectiveness(events),
            "exceptions_noted": self.identify_exceptions(events),
            "management_response": self.get_management_responses(),
            "auditor_opinion": "Pending external review"
        }

    def generate_gdpr_report(self, period: Dict) -> Dict:
        """Generate GDPR compliance report"""
        events = self.event_store.query_events({
            "timestamp": {
                "$gte": period["start"],
                "$lte": period["end"]
            },
            "type": {"$in": ["data_access", "consent", "data_deletion"]}
        })

        return {
            "report_type": "GDPR Compliance",
            "period": period,
            "data_subject_requests": {
                "access_requests": self.count_access_requests(events),
                "deletion_requests": self.count_deletion_requests(events),
                "portability_requests": self.count_portability_requests(events),
                "rectification_requests": self.count_rectification_requests(events)
            },
            "consent_management": {
                "consents_granted": self.count_consents_granted(events),
                "consents_withdrawn": self.count_consents_withdrawn(events),
                "consent_audit_trail": self.get_consent_audit_trail(events)
            },
            "data_breaches": self.analyze_data_breaches(events),
            "cross_border_transfers": self.analyze_cross_border_transfers(events),
            "dpia_results": self.get_dpia_results(),
            "compliance_score": self.calculate_gdpr_compliance_score(events)
        }

    def generate_hipaa_report(self, period: Dict) -> Dict:
        """Generate HIPAA compliance report"""
        events = self.event_store.query_events({
            "timestamp": {
                "$gte": period["start"],
                "$lte": period["end"]
            },
            "metadata.phi_involved": True
        })

        return {
            "report_type": "HIPAA Compliance",
            "period": period,
            "administrative_safeguards": {
                "access_controls": self.analyze_phi_access_controls(events),
                "workforce_training": self.get_training_compliance(),
                "incident_response": self.analyze_phi_incidents(events)
            },
            "physical_safeguards": {
                "facility_access": self.analyze_facility_access(events),
                "device_controls": self.analyze_device_controls(events)
            },
            "technical_safeguards": {
                "access_audit": self.analyze_phi_access_audit(events),
                "integrity_controls": self.analyze_integrity_controls(events),
                "transmission_security": self.analyze_transmission_security(events)
            },
            "breach_notifications": self.get_breach_notifications(events),
            "risk_assessment": self.get_risk_assessment_results()
        }

    def generate_executive_dashboard(self) -> Dict:
        """Generate executive compliance dashboard"""
        last_30_days = {
            "start": (datetime.now() - timedelta(days=30)).isoformat(),
            "end": datetime.now().isoformat()
        }

        events = self.event_store.query_events({
            "timestamp": {
                "$gte": last_30_days["start"],
                "$lte": last_30_days["end"]
            }
        })

        return {
            "summary": {
                "total_events": len(events),
                "security_incidents": self.count_security_incidents(events),
                "compliance_violations": self.count_compliance_violations(events),
                "system_availability": self.calculate_availability(events)
            },
            "compliance_status": {
                "soc2": self.get_soc2_status(),
                "iso27001": self.get_iso27001_status(),
                "gdpr": self.get_gdpr_status(),
                "hipaa": self.get_hipaa_status()
            },
            "key_metrics": {
                "mean_time_to_detect": self.calculate_mttd(events),
                "mean_time_to_respond": self.calculate_mttr(events),
                "false_positive_rate": self.calculate_false_positive_rate(events),
                "audit_coverage": self.calculate_audit_coverage(events)
            },
            "trends": {
                "security_events": self.get_security_trend(events),
                "compliance_events": self.get_compliance_trend(events),
                "access_patterns": self.get_access_pattern_trend(events)
            },
            "recommendations": self.generate_recommendations(events)
        }

    def export_for_auditor(self, audit_request: Dict) -> Dict:
        """Export audit trail for external auditor"""
        events = self.event_store.query_events(audit_request["filters"])

        # Verify integrity
        integrity_verified = all(
            self.event_store.verify_event(event) for event in events
        )

        return {
            "audit_export": {
                "request_id": audit_request["id"],
                "auditor": audit_request["auditor"],
                "export_date": datetime.now().isoformat(),
                "period": audit_request["period"],
                "total_events": len(events),
                "events": events,
                "integrity_verified": integrity_verified,
                "blockchain_anchors": self.get_blockchain_proofs(events),
                "signature_verification": self.get_signature_verification(events)
            },
            "attestation": self.generate_attestation(audit_request, events)
        }
```

## 5. Forensic Capabilities

### Forensic Analysis Tools

```python
# forensic_analysis.py
import networkx as nx
from typing import Dict, List, Optional, Tuple
import pandas as pd

class ForensicAnalyzer:
    def __init__(self, event_store):
        self.event_store = event_store

    def investigate_incident(self, incident_id: str) -> Dict:
        """Comprehensive incident investigation"""
        # Get incident details
        incident = self.get_incident_details(incident_id)

        # Build timeline
        timeline = self.build_incident_timeline(incident)

        # Identify actors
        actors = self.identify_actors(timeline)

        # Analyze attack pattern
        attack_pattern = self.analyze_attack_pattern(timeline)

        # Assess impact
        impact = self.assess_impact(timeline)

        # Generate IoCs
        iocs = self.extract_indicators_of_compromise(timeline)

        return {
            "incident_id": incident_id,
            "investigation_date": datetime.now().isoformat(),
            "timeline": timeline,
            "actors": actors,
            "attack_pattern": attack_pattern,
            "impact_assessment": impact,
            "indicators_of_compromise": iocs,
            "recommendations": self.generate_recommendations(incident, timeline)
        }

    def build_incident_timeline(self, incident: Dict) -> List[Dict]:
        """Build detailed timeline of incident"""
        # Get time window
        start_time = incident["detected_at"] - timedelta(hours=24)
        end_time = incident["resolved_at"] if incident.get("resolved_at") else datetime.now()

        # Query related events
        events = self.event_store.query_events({
            "timestamp": {"$gte": start_time, "$lte": end_time},
            "$or": [
                {"actor": {"$in": incident["involved_actors"]}},
                {"resource": {"$in": incident["affected_resources"]}},
                {"metadata.correlation_id": incident["correlation_id"]}
            ]
        })

        # Sort by timestamp
        timeline = sorted(events, key=lambda x: x["timestamp"])

        # Enrich with context
        for event in timeline:
            event["relevance"] = self.calculate_relevance(event, incident)
            event["classification"] = self.classify_event(event, incident)

        return timeline

    def trace_user_activity(self, user_id: str, period: Dict) -> Dict:
        """Trace all activity for a specific user"""
        events = self.event_store.query_events({
            "actor": user_id,
            "timestamp": {"$gte": period["start"], "$lte": period["end"]}
        })

        return {
            "user_id": user_id,
            "period": period,
            "total_events": len(events),
            "activity_summary": {
                "authentication": self.summarize_auth_activity(events),
                "data_access": self.summarize_data_access(events),
                "api_calls": self.summarize_api_activity(events),
                "security_events": self.summarize_security_events(events)
            },
            "behavioral_analysis": self.analyze_user_behavior(events),
            "anomalies_detected": self.detect_user_anomalies(events),
            "risk_score": self.calculate_user_risk_score(events)
        }

    def detect_data_exfiltration(self, period: Dict) -> List[Dict]:
        """Detect potential data exfiltration"""
        events = self.event_store.query_events({
            "timestamp": {"$gte": period["start"], "$lte": period["end"]},
            "type": {"$in": ["data_read", "data_export", "api_call"]}
        })

        suspicious_patterns = []

        # Detect large data transfers
        large_transfers = self.detect_large_transfers(events)
        if large_transfers:
            suspicious_patterns.append({
                "pattern": "large_data_transfer",
                "events": large_transfers,
                "risk_level": "high"
            })

        # Detect unusual access patterns
        unusual_access = self.detect_unusual_access_patterns(events)
        if unusual_access:
            suspicious_patterns.append({
                "pattern": "unusual_access",
                "events": unusual_access,
                "risk_level": "medium"
            })

        # Detect rapid sequential access
        rapid_access = self.detect_rapid_sequential_access(events)
        if rapid_access:
            suspicious_patterns.append({
                "pattern": "rapid_sequential_access",
                "events": rapid_access,
                "risk_level": "high"
            })

        return suspicious_patterns

    def reconstruct_attack_chain(self, ioc: str) -> Dict:
        """Reconstruct complete attack chain from IOC"""
        # Find initial compromise
        initial_events = self.find_initial_compromise(ioc)

        # Build attack graph
        attack_graph = nx.DiGraph()

        for event in initial_events:
            self.build_attack_graph(event, attack_graph)

        # Identify kill chain phases
        kill_chain = self.map_to_kill_chain(attack_graph)

        return {
            "ioc": ioc,
            "initial_compromise": initial_events,
            "attack_graph": self.serialize_graph(attack_graph),
            "kill_chain_phases": kill_chain,
            "ttps": self.extract_ttps(attack_graph),
            "affected_systems": self.identify_affected_systems(attack_graph),
            "data_at_risk": self.identify_data_at_risk(attack_graph)
        }

    def generate_forensic_report(self, case_id: str) -> Dict:
        """Generate comprehensive forensic report"""
        case_data = self.get_case_data(case_id)

        return {
            "case_id": case_id,
            "report_date": datetime.now().isoformat(),
            "executive_summary": self.generate_executive_summary(case_data),
            "technical_findings": {
                "timeline": case_data["timeline"],
                "root_cause": self.identify_root_cause(case_data),
                "attack_vectors": self.identify_attack_vectors(case_data),
                "vulnerabilities_exploited": self.identify_vulnerabilities(case_data)
            },
            "evidence_preservation": {
                "preserved_logs": self.get_preserved_logs(case_id),
                "blockchain_anchors": self.get_blockchain_evidence(case_id),
                "integrity_verification": self.verify_evidence_integrity(case_id)
            },
            "impact_analysis": self.analyze_case_impact(case_data),
            "recommendations": {
                "immediate_actions": self.get_immediate_actions(case_data),
                "long_term_improvements": self.get_long_term_improvements(case_data),
                "preventive_measures": self.get_preventive_measures(case_data)
            },
            "legal_requirements": self.check_legal_requirements(case_data)
        }
```