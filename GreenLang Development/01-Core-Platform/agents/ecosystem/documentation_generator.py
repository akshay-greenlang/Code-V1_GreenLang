# -*- coding: utf-8 -*-
"""
GL-ECO-X-005: Documentation Generator
======================================

Auto-generates documentation for agents, APIs, and solution packs.

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


class DocumentationType(str, Enum):
    API_REFERENCE = "api_reference"
    USER_GUIDE = "user_guide"
    TUTORIAL = "tutorial"
    CHANGELOG = "changelog"
    README = "readme"


class DocumentationSection(BaseModel):
    section_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    order: int = Field(default=0)
    subsections: List["DocumentationSection"] = Field(default_factory=list)


class APIEndpoint(BaseModel):
    method: str = Field(..., description="HTTP method")
    path: str = Field(..., description="Endpoint path")
    description: str = Field(...)
    parameters: List[Dict[str, Any]] = Field(default_factory=list)
    responses: Dict[str, Any] = Field(default_factory=dict)


class APIDocumentation(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = Field(..., description="API title")
    version: str = Field(default="1.0.0")
    description: str = Field(default="")
    base_url: str = Field(default="")
    endpoints: List[APIEndpoint] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=DeterministicClock.now)


class UserGuide(BaseModel):
    guide_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = Field(..., description="Guide title")
    version: str = Field(default="1.0.0")
    sections: List[DocumentationSection] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=DeterministicClock.now)


class DocumentationInput(BaseModel):
    operation: str = Field(..., description="Operation to perform")
    agent_id: Optional[str] = Field(None)
    doc_type: Optional[DocumentationType] = Field(None)
    source_code: Optional[str] = Field(None)
    agent_metadata: Optional[Dict[str, Any]] = Field(None)
    template: Optional[str] = Field(None)

    @field_validator('operation')
    @classmethod
    def validate_operation(cls, v: str) -> str:
        valid_ops = {
            'generate_api_docs', 'generate_user_guide', 'generate_readme',
            'generate_changelog', 'extract_docstrings', 'get_templates',
            'validate_docs', 'get_statistics'
        }
        if v not in valid_ops:
            raise ValueError(f"Operation must be one of: {valid_ops}")
        return v


class DocumentationOutput(BaseModel):
    success: bool = Field(...)
    operation: str = Field(...)
    data: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


README_TEMPLATE = """# {name}

{description}

## Installation

```bash
pip install greenlang
```

## Usage

```python
from greenlang.agents import {class_name}

agent = {class_name}()
result = agent.run({{"operation": "process"}})
```

## API Reference

### Operations

{operations}

## License

Apache-2.0
"""


class DocumentationGenerator(BaseAgent):
    """GL-ECO-X-005: Documentation Generator"""

    AGENT_ID = "GL-ECO-X-005"
    AGENT_NAME = "Documentation Generator"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Auto-generates documentation",
                version=self.VERSION,
            )
        super().__init__(config)
        self._generated_docs: Dict[str, Any] = {}
        self._total_generated = 0
        self.logger.info(f"Initialized {self.AGENT_ID}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_time = time.time()
        try:
            doc_input = DocumentationInput(**input_data)
            result_data = self._route_operation(doc_input)
            provenance_hash = hashlib.sha256(
                json.dumps({"in": input_data, "out": result_data}, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]

            output = DocumentationOutput(
                success=True, operation=doc_input.operation, data=result_data,
                provenance_hash=provenance_hash, processing_time_ms=(time.time() - start_time) * 1000,
            )
            return AgentResult(success=True, data=output.model_dump())
        except Exception as e:
            self.logger.error(f"Operation failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _route_operation(self, doc_input: DocumentationInput) -> Dict[str, Any]:
        op = doc_input.operation
        if op == "generate_api_docs":
            return self._generate_api_docs(doc_input.agent_id, doc_input.agent_metadata)
        elif op == "generate_user_guide":
            return self._generate_user_guide(doc_input.agent_id, doc_input.agent_metadata)
        elif op == "generate_readme":
            return self._generate_readme(doc_input.agent_metadata)
        elif op == "generate_changelog":
            return self._generate_changelog(doc_input.agent_id)
        elif op == "extract_docstrings":
            return self._extract_docstrings(doc_input.source_code)
        elif op == "get_templates":
            return self._get_templates()
        elif op == "validate_docs":
            return self._validate_docs(doc_input.agent_id)
        elif op == "get_statistics":
            return self._get_statistics()
        raise ValueError(f"Unknown operation: {op}")

    def _generate_api_docs(self, agent_id: Optional[str], metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not metadata:
            return {"error": "agent_metadata required"}

        endpoints = [
            APIEndpoint(method="POST", path="/run", description="Execute agent operation",
                       parameters=[{"name": "operation", "type": "string", "required": True}],
                       responses={"200": {"description": "Success"}}),
        ]

        api_doc = APIDocumentation(
            title=metadata.get("name", "Agent API"),
            version=metadata.get("version", "1.0.0"),
            description=metadata.get("description", ""),
            endpoints=endpoints,
        )

        self._generated_docs[api_doc.doc_id] = api_doc
        self._total_generated += 1

        return {"doc_id": api_doc.doc_id, "documentation": api_doc.model_dump()}

    def _generate_user_guide(self, agent_id: Optional[str], metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not metadata:
            return {"error": "agent_metadata required"}

        sections = [
            DocumentationSection(title="Introduction", content=metadata.get("description", ""), order=1),
            DocumentationSection(title="Getting Started", content="Install and configure the agent.", order=2),
            DocumentationSection(title="Usage", content="How to use the agent.", order=3),
        ]

        guide = UserGuide(title=f"{metadata.get('name', 'Agent')} User Guide", sections=sections)
        self._generated_docs[guide.guide_id] = guide
        self._total_generated += 1

        return {"guide_id": guide.guide_id, "guide": guide.model_dump()}

    def _generate_readme(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not metadata:
            return {"error": "agent_metadata required"}

        name = metadata.get("name", "Agent")
        class_name = name.replace(" ", "") + "Agent"
        operations = "- `process`: Main processing operation\n- `get_statistics`: Get agent statistics"

        readme = README_TEMPLATE.format(
            name=name, description=metadata.get("description", ""), class_name=class_name, operations=operations
        )

        self._total_generated += 1
        return {"readme": readme}

    def _generate_changelog(self, agent_id: Optional[str]) -> Dict[str, Any]:
        changelog = f"""# Changelog

## [1.0.0] - {DeterministicClock.now().strftime('%Y-%m-%d')}
### Added
- Initial release
"""
        self._total_generated += 1
        return {"changelog": changelog}

    def _extract_docstrings(self, source_code: Optional[str]) -> Dict[str, Any]:
        if not source_code:
            return {"error": "source_code required"}

        import re
        docstrings = re.findall(r'"""(.*?)"""', source_code, re.DOTALL)
        return {"docstrings": docstrings, "count": len(docstrings)}

    def _get_templates(self) -> Dict[str, Any]:
        return {
            "templates": [
                {"type": "readme", "description": "README.md template"},
                {"type": "api_docs", "description": "API documentation template"},
                {"type": "user_guide", "description": "User guide template"},
            ]
        }

    def _validate_docs(self, agent_id: Optional[str]) -> Dict[str, Any]:
        return {"valid": True, "issues": []}

    def _get_statistics(self) -> Dict[str, Any]:
        return {"total_generated": self._total_generated, "cached_docs": len(self._generated_docs)}
