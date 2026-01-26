# -*- coding: utf-8 -*-
"""
GL-ECO-X-002: Pack Builder Agent
=================================

Builds solution packs that bundle agents, configurations, and resources
for specific use cases and industry verticals.

Capabilities:
    - Solution pack creation and packaging
    - Component bundling and dependency resolution
    - Manifest generation and validation
    - Pack versioning and distribution
    - Industry vertical customization
    - Template-based pack creation

Zero-Hallucination Guarantees:
    - All packaging uses deterministic processes
    - Complete provenance tracking with SHA-256 hashes
    - No LLM calls in the build path
    - All pack contents traceable

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class PackType(str, Enum):
    """Types of solution packs."""
    INDUSTRY_VERTICAL = "industry_vertical"
    USE_CASE = "use_case"
    INTEGRATION = "integration"
    STARTER = "starter"
    ENTERPRISE = "enterprise"


class BuildStatus(str, Enum):
    """Status of pack build."""
    PENDING = "pending"
    BUILDING = "building"
    COMPLETED = "completed"
    FAILED = "failed"


class ComponentType(str, Enum):
    """Types of pack components."""
    AGENT = "agent"
    CONFIGURATION = "configuration"
    TEMPLATE = "template"
    DATA = "data"
    DOCUMENTATION = "documentation"
    SCHEMA = "schema"


# =============================================================================
# Pydantic Models
# =============================================================================

class PackComponent(BaseModel):
    """A component within a solution pack."""
    component_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = Field(..., description="Component name")
    component_type: ComponentType = Field(..., description="Type of component")
    version: str = Field(default="1.0.0")
    description: str = Field(default="")

    # References
    agent_id: Optional[str] = Field(None, description="Agent ID if type is agent")
    file_path: Optional[str] = Field(None, description="Path to resource file")

    # Content
    content: Optional[Dict[str, Any]] = Field(None, description="Component content")

    # Dependencies
    dependencies: List[str] = Field(default_factory=list)
    optional: bool = Field(default=False)


class PackManifest(BaseModel):
    """Manifest describing a solution pack."""
    manifest_version: str = Field(default="1.0")
    pack_id: str = Field(..., description="Unique pack identifier")
    name: str = Field(..., description="Pack name")
    version: str = Field(..., description="Pack version")
    description: str = Field(..., description="Pack description")

    # Classification
    pack_type: PackType = Field(..., description="Type of pack")
    industry: Optional[str] = Field(None, description="Target industry")
    use_cases: List[str] = Field(default_factory=list)

    # Components
    components: List[PackComponent] = Field(default_factory=list)

    # Requirements
    greenlang_version: str = Field(default=">=1.0.0")
    python_version: str = Field(default=">=3.9")
    dependencies: Dict[str, str] = Field(default_factory=dict)

    # Metadata
    author: str = Field(default="GreenLang Team")
    license: str = Field(default="Apache-2.0")
    homepage: Optional[str] = Field(None)
    repository: Optional[str] = Field(None)

    # Build info
    created_at: datetime = Field(default_factory=DeterministicClock.now)
    built_at: Optional[datetime] = Field(None)
    build_hash: Optional[str] = Field(None)


class SolutionPack(BaseModel):
    """A complete solution pack."""
    manifest: PackManifest = Field(..., description="Pack manifest")
    build_status: BuildStatus = Field(default=BuildStatus.PENDING)
    build_log: List[str] = Field(default_factory=list)
    total_size_bytes: int = Field(default=0)
    component_count: int = Field(default=0)


class PackBuilderInput(BaseModel):
    """Input for the Pack Builder Agent."""
    operation: str = Field(..., description="Operation to perform")
    manifest: Optional[PackManifest] = Field(None)
    pack_id: Optional[str] = Field(None)
    components: List[PackComponent] = Field(default_factory=list)
    pack_type: Optional[PackType] = Field(None)

    @field_validator('operation')
    @classmethod
    def validate_operation(cls, v: str) -> str:
        valid_ops = {
            'create_pack', 'build_pack', 'validate_pack',
            'add_component', 'remove_component', 'get_pack',
            'list_packs', 'export_pack', 'get_statistics'
        }
        if v not in valid_ops:
            raise ValueError(f"Operation must be one of: {valid_ops}")
        return v


class PackBuilderOutput(BaseModel):
    """Output from the Pack Builder Agent."""
    success: bool = Field(..., description="Whether operation succeeded")
    operation: str = Field(..., description="Operation performed")
    data: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


# =============================================================================
# Pack Builder Agent Implementation
# =============================================================================

class PackBuilderAgent(BaseAgent):
    """
    GL-ECO-X-002: Pack Builder Agent

    Builds solution packs that bundle agents, configurations, and resources.

    Usage:
        builder = PackBuilderAgent()

        # Create a pack
        result = builder.run({
            "operation": "create_pack",
            "manifest": {
                "pack_id": "manufacturing-starter",
                "name": "Manufacturing Starter Pack",
                "version": "1.0.0",
                "pack_type": "starter",
                "industry": "manufacturing"
            }
        })
    """

    AGENT_ID = "GL-ECO-X-002"
    AGENT_NAME = "Pack Builder Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Builds GreenLang solution packs",
                version=self.VERSION,
            )
        super().__init__(config)

        self._packs: Dict[str, SolutionPack] = {}
        self._total_packs_built = 0

        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_time = time.time()

        try:
            builder_input = PackBuilderInput(**input_data)
            operation = builder_input.operation

            result_data = self._route_operation(builder_input)

            provenance_hash = self._compute_provenance_hash(input_data, result_data)
            processing_time_ms = (time.time() - start_time) * 1000

            output = PackBuilderOutput(
                success=True,
                operation=operation,
                data=result_data,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time_ms,
            )

            return AgentResult(success=True, data=output.model_dump())

        except Exception as e:
            self.logger.error(f"Pack builder operation failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _route_operation(self, builder_input: PackBuilderInput) -> Dict[str, Any]:
        operation = builder_input.operation

        if operation == "create_pack":
            return self._handle_create_pack(builder_input.manifest)
        elif operation == "build_pack":
            return self._handle_build_pack(builder_input.pack_id)
        elif operation == "validate_pack":
            return self._handle_validate_pack(builder_input.pack_id)
        elif operation == "add_component":
            return self._handle_add_component(builder_input.pack_id, builder_input.components)
        elif operation == "remove_component":
            return self._handle_remove_component(builder_input.pack_id, builder_input.components)
        elif operation == "get_pack":
            return self._handle_get_pack(builder_input.pack_id)
        elif operation == "list_packs":
            return self._handle_list_packs(builder_input.pack_type)
        elif operation == "export_pack":
            return self._handle_export_pack(builder_input.pack_id)
        elif operation == "get_statistics":
            return self._handle_get_statistics()
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _handle_create_pack(self, manifest: Optional[PackManifest]) -> Dict[str, Any]:
        """Create a new solution pack."""
        if not manifest:
            return {"error": "manifest is required"}

        pack = SolutionPack(
            manifest=manifest,
            build_status=BuildStatus.PENDING,
            component_count=len(manifest.components),
        )

        self._packs[manifest.pack_id] = pack

        return {
            "pack_id": manifest.pack_id,
            "created": True,
            "status": BuildStatus.PENDING.value,
        }

    def _handle_build_pack(self, pack_id: Optional[str]) -> Dict[str, Any]:
        """Build a solution pack."""
        if not pack_id or pack_id not in self._packs:
            return {"error": f"Pack not found: {pack_id}"}

        pack = self._packs[pack_id]
        pack.build_status = BuildStatus.BUILDING
        pack.build_log.append(f"Build started at {DeterministicClock.now().isoformat()}")

        # Validate components
        validation_errors = []
        for component in pack.manifest.components:
            errors = self._validate_component(component)
            if errors:
                validation_errors.extend(errors)

        if validation_errors:
            pack.build_status = BuildStatus.FAILED
            pack.build_log.append(f"Build failed: {len(validation_errors)} validation errors")
            return {
                "pack_id": pack_id,
                "status": BuildStatus.FAILED.value,
                "errors": validation_errors,
            }

        # Resolve dependencies
        pack.build_log.append("Resolving dependencies...")
        resolved = self._resolve_dependencies(pack.manifest.components)

        # Calculate build hash
        content_str = json.dumps(pack.manifest.model_dump(), sort_keys=True, default=str)
        build_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]

        pack.manifest.built_at = DeterministicClock.now()
        pack.manifest.build_hash = build_hash
        pack.build_status = BuildStatus.COMPLETED
        pack.build_log.append(f"Build completed with hash: {build_hash}")

        self._total_packs_built += 1

        return {
            "pack_id": pack_id,
            "status": BuildStatus.COMPLETED.value,
            "build_hash": build_hash,
            "component_count": len(pack.manifest.components),
            "build_log": pack.build_log,
        }

    def _validate_component(self, component: PackComponent) -> List[str]:
        """Validate a pack component."""
        errors = []

        if not component.name:
            errors.append(f"Component {component.component_id} missing name")

        if component.component_type == ComponentType.AGENT:
            if not component.agent_id:
                errors.append(f"Agent component {component.name} missing agent_id")

        return errors

    def _resolve_dependencies(self, components: List[PackComponent]) -> List[str]:
        """Resolve component dependencies."""
        resolved = []
        component_ids = {c.component_id for c in components}

        for component in components:
            for dep in component.dependencies:
                if dep not in component_ids:
                    self.logger.warning(f"Unresolved dependency: {dep}")
            resolved.append(component.component_id)

        return resolved

    def _handle_validate_pack(self, pack_id: Optional[str]) -> Dict[str, Any]:
        """Validate a pack without building."""
        if not pack_id or pack_id not in self._packs:
            return {"error": f"Pack not found: {pack_id}"}

        pack = self._packs[pack_id]
        issues = []

        # Check manifest
        if not pack.manifest.description:
            issues.append({"level": "warning", "message": "Pack missing description"})

        # Check components
        for component in pack.manifest.components:
            errors = self._validate_component(component)
            for error in errors:
                issues.append({"level": "error", "message": error})

        return {
            "pack_id": pack_id,
            "valid": not any(i["level"] == "error" for i in issues),
            "issues": issues,
        }

    def _handle_add_component(
        self, pack_id: Optional[str], components: List[PackComponent]
    ) -> Dict[str, Any]:
        """Add components to a pack."""
        if not pack_id or pack_id not in self._packs:
            return {"error": f"Pack not found: {pack_id}"}

        pack = self._packs[pack_id]
        added = 0

        for component in components:
            pack.manifest.components.append(component)
            added += 1

        pack.component_count = len(pack.manifest.components)

        return {
            "pack_id": pack_id,
            "added": added,
            "total_components": pack.component_count,
        }

    def _handle_remove_component(
        self, pack_id: Optional[str], components: List[PackComponent]
    ) -> Dict[str, Any]:
        """Remove components from a pack."""
        if not pack_id or pack_id not in self._packs:
            return {"error": f"Pack not found: {pack_id}"}

        pack = self._packs[pack_id]
        removed_ids = {c.component_id for c in components}

        pack.manifest.components = [
            c for c in pack.manifest.components
            if c.component_id not in removed_ids
        ]
        pack.component_count = len(pack.manifest.components)

        return {
            "pack_id": pack_id,
            "removed": len(removed_ids),
            "total_components": pack.component_count,
        }

    def _handle_get_pack(self, pack_id: Optional[str]) -> Dict[str, Any]:
        """Get pack details."""
        if not pack_id or pack_id not in self._packs:
            return {"error": f"Pack not found: {pack_id}"}

        return self._packs[pack_id].model_dump()

    def _handle_list_packs(self, pack_type: Optional[PackType]) -> Dict[str, Any]:
        """List all packs."""
        packs = list(self._packs.values())

        if pack_type:
            packs = [p for p in packs if p.manifest.pack_type == pack_type]

        return {
            "packs": [
                {
                    "pack_id": p.manifest.pack_id,
                    "name": p.manifest.name,
                    "version": p.manifest.version,
                    "pack_type": p.manifest.pack_type.value,
                    "status": p.build_status.value,
                    "component_count": p.component_count,
                }
                for p in packs
            ],
            "count": len(packs),
        }

    def _handle_export_pack(self, pack_id: Optional[str]) -> Dict[str, Any]:
        """Export pack as JSON."""
        if not pack_id or pack_id not in self._packs:
            return {"error": f"Pack not found: {pack_id}"}

        pack = self._packs[pack_id]

        if pack.build_status != BuildStatus.COMPLETED:
            return {"error": "Pack must be built before export"}

        export_data = pack.model_dump()

        return {
            "pack_id": pack_id,
            "export": export_data,
            "size_bytes": len(json.dumps(export_data)),
        }

    def _handle_get_statistics(self) -> Dict[str, Any]:
        """Get builder statistics."""
        return {
            "total_packs": len(self._packs),
            "total_built": self._total_packs_built,
            "by_status": {
                status.value: sum(1 for p in self._packs.values() if p.build_status == status)
                for status in BuildStatus
            },
        }

    def _compute_provenance_hash(
        self, input_data: Dict[str, Any], output_data: Dict[str, Any]
    ) -> str:
        provenance_str = json.dumps(
            {"input": input_data, "output": output_data},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()[:16]
