"""
Provenance tracking for zero-hallucination compliance.

Implements SHA-256 hashing for complete audit trails.
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProvenanceRecord(BaseModel):
    """Complete provenance record for an agent execution."""

    input_hash: str = Field(..., description="SHA-256 hash of input data")
    output_hash: str = Field(..., description="SHA-256 hash of output data")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="All tool invocations")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str = Field(..., description="Agent identifier")
    agent_version: str = Field(..., description="Agent version")
    provenance_chain: str = Field(..., description="Combined SHA-256 hash of entire execution")

    class Config:
        json_schema_extra = {
            "example": {
                "input_hash": "a1b2c3...",
                "output_hash": "d4e5f6...",
                "tool_calls": [{"tool": "calculator", "params": {}, "result_hash": "g7h8i9..."}],
                "timestamp": "2025-12-03T10:00:00Z",
                "agent_id": "gl-fuel-analyzer-v1",
                "agent_version": "1.0.0",
                "provenance_chain": "j1k2l3..."
            }
        }


class ProvenanceTracker:
    """Tracks complete provenance chain for regulatory compliance."""

    def __init__(self, agent_id: str, agent_version: str):
        self.agent_id = agent_id
        self.agent_version = agent_version
        self.tool_calls: List[Dict[str, Any]] = []

    def hash_data(self, data: Any) -> str:
        """Create SHA-256 hash of any data (deterministic)."""
        # Convert to JSON with sorted keys for determinism
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def record_tool_call(self, tool_name: str, params: Dict[str, Any], result: Any) -> str:
        """Record a tool invocation with provenance."""
        result_hash = self.hash_data(result)
        self.tool_calls.append({
            "tool": tool_name,
            "params": params,
            "result_hash": result_hash,
            "timestamp": datetime.utcnow().isoformat()
        })
        return result_hash

    def create_provenance_record(self, input_data: Any, output_data: Any) -> ProvenanceRecord:
        """Create complete provenance record."""
        input_hash = self.hash_data(input_data)
        output_hash = self.hash_data(output_data)

        # Create provenance chain: hash of (input_hash + output_hash + tool_calls)
        chain_data = {
            "input_hash": input_hash,
            "output_hash": output_hash,
            "tool_calls": self.tool_calls
        }
        provenance_chain = self.hash_data(chain_data)

        return ProvenanceRecord(
            input_hash=input_hash,
            output_hash=output_hash,
            tool_calls=self.tool_calls.copy(),
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            provenance_chain=provenance_chain
        )

    def verify_provenance(self, record: ProvenanceRecord, input_data: Any, output_data: Any) -> bool:
        """Verify a provenance record matches the data."""
        expected_input_hash = self.hash_data(input_data)
        expected_output_hash = self.hash_data(output_data)

        return (record.input_hash == expected_input_hash and
                record.output_hash == expected_output_hash)
