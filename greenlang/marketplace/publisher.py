"""
Agent Publishing Workflow

Implements the complete workflow for publishing agents to the marketplace,
including validation, security scanning, performance testing, and metadata extraction.
"""

import ast
import hashlib
import inspect
import json
import os
import tempfile
import zipfile
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from sqlalchemy.orm import Session

from greenlang.marketplace.models import (
    MarketplaceAgent,
    AgentVersion,
    AgentAsset,
    AgentDependency,
    AgentStatus,
)
from greenlang.marketplace.validator import AgentValidator, SecurityScanner, CodeValidator
from greenlang.marketplace.versioning import SemanticVersion, BreakingChangeDetector

logger = logging.getLogger(__name__)


class PublishStage(str, Enum):
    """Publishing workflow stages"""
    UPLOAD = "upload"
    METADATA_EXTRACTION = "metadata_extraction"
    VALIDATION = "validation"
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_TEST = "performance_test"
    DOCUMENTATION_CHECK = "documentation_check"
    ASSET_UPLOAD = "asset_upload"
    PRICING_SETUP = "pricing_setup"
    REVIEW = "review"
    PUBLISH = "publish"


class ValidationSeverity(str, Enum):
    """Validation issue severity"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Single validation issue"""
    severity: ValidationSeverity
    message: str
    code: str
    line: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Complete validation result"""
    passed: bool
    stage: PublishStage
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only errors"""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warnings"""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]


@dataclass
class AgentMetadata:
    """Extracted agent metadata"""
    name: str
    description: str
    version: str
    author: str
    author_email: Optional[str] = None
    homepage: Optional[str] = None
    repository: Optional[str] = None
    license: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    category: Optional[str] = None
    dependencies: Dict[str, str] = field(default_factory=dict)
    python_requires: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None


@dataclass
class PublishingChecklist:
    """Publishing checklist items"""
    code_uploaded: bool = False
    metadata_extracted: bool = False
    validation_passed: bool = False
    security_passed: bool = False
    performance_tested: bool = False
    documentation_complete: bool = False
    assets_uploaded: bool = False
    pricing_configured: bool = False
    ready_to_publish: bool = False

    def is_complete(self) -> bool:
        """Check if all required items are complete"""
        return (
            self.code_uploaded and
            self.metadata_extracted and
            self.validation_passed and
            self.security_passed and
            self.documentation_complete and
            self.pricing_configured
        )


class MetadataExtractor:
    """
    Extract metadata from agent code.

    Parses Python AST to extract docstrings, type hints, and metadata.
    """

    @staticmethod
    def extract_from_file(file_path: str) -> AgentMetadata:
        """
        Extract metadata from a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            Extracted metadata
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source)

        # Find agent class
        agent_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if inherits from BaseAgent
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == 'BaseAgent':
                        agent_class = node
                        break
                if agent_class:
                    break

        if not agent_class:
            raise ValueError("No class inheriting from BaseAgent found")

        # Extract docstring
        docstring = ast.get_docstring(agent_class) or ""
        lines = docstring.split('\n')
        description = lines[0] if lines else "No description"

        # Extract metadata from class attributes or docstring
        metadata = AgentMetadata(
            name=agent_class.name,
            description=description,
            version="1.0.0",  # Default, should be overridden
            author="Unknown"
        )

        # Parse docstring for structured metadata
        for line in lines[1:]:
            line = line.strip()
            if line.startswith("Author:"):
                metadata.author = line.split(":", 1)[1].strip()
            elif line.startswith("Version:"):
                metadata.version = line.split(":", 1)[1].strip()
            elif line.startswith("License:"):
                metadata.license = line.split(":", 1)[1].strip()

        # Extract execute method signature for schema
        for node in agent_class.body:
            if isinstance(node, ast.FunctionDef) and node.name == "execute":
                metadata.input_schema = MetadataExtractor._extract_input_schema(node)
                metadata.output_schema = MetadataExtractor._extract_output_schema(node)

        return metadata

    @staticmethod
    def _extract_input_schema(func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract JSON Schema from function signature"""
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        # Skip 'self' parameter
        for arg in func_node.args.args[1:]:
            param_name = arg.arg
            schema["properties"][param_name] = {"type": "string"}  # Default
            schema["required"].append(param_name)

            # Try to get type annotation
            if arg.annotation:
                type_name = MetadataExtractor._annotation_to_json_type(arg.annotation)
                schema["properties"][param_name]["type"] = type_name

        return schema

    @staticmethod
    def _extract_output_schema(func_node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract output schema from return annotation"""
        schema = {"type": "object"}

        if func_node.returns:
            type_name = MetadataExtractor._annotation_to_json_type(func_node.returns)
            schema["type"] = type_name

        return schema

    @staticmethod
    def _annotation_to_json_type(annotation: ast.expr) -> str:
        """Convert Python type annotation to JSON Schema type"""
        if isinstance(annotation, ast.Name):
            type_map = {
                "str": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
                "dict": "object",
                "list": "array"
            }
            return type_map.get(annotation.id, "string")
        return "string"


class PublishingWorkflow:
    """
    Complete agent publishing workflow.

    Manages the entire process from upload to publication.
    """

    def __init__(self, session: Session, storage_path: str = "/tmp/marketplace"):
        self.session = session
        self.storage_path = storage_path
        self.validator = AgentValidator()
        self.security_scanner = SecurityScanner()
        os.makedirs(storage_path, exist_ok=True)

    def start_publishing(
        self,
        author_id: str,
        author_name: str
    ) -> Tuple[str, PublishingChecklist]:
        """
        Start a new publishing workflow.

        Args:
            author_id: Author user UUID
            author_name: Author name

        Returns:
            Tuple of (draft_id, checklist)
        """
        # Create draft agent
        draft = MarketplaceAgent(
            name="Untitled Agent",
            slug=f"draft-{datetime.utcnow().timestamp()}",
            description="Draft agent",
            author_id=author_id,
            author_name=author_name,
            status=AgentStatus.DRAFT.value
        )

        self.session.add(draft)
        self.session.commit()

        checklist = PublishingChecklist()

        logger.info(f"Started publishing workflow for author {author_id}, draft {draft.id}")

        return str(draft.id), checklist

    def upload_code(
        self,
        draft_id: str,
        code_content: bytes,
        filename: str
    ) -> ValidationResult:
        """
        Upload and validate agent code.

        Args:
            draft_id: Draft agent UUID
            code_content: Code file bytes
            filename: Original filename

        Returns:
            Validation result
        """
        result = ValidationResult(passed=False, stage=PublishStage.UPLOAD)

        # Save to temporary file
        file_path = os.path.join(self.storage_path, f"{draft_id}_{filename}")

        try:
            with open(file_path, 'wb') as f:
                f.write(code_content)

            # Calculate hash
            code_hash = hashlib.sha256(code_content).hexdigest()

            # Validate file
            if not filename.endswith('.py'):
                result.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="Only Python (.py) files are supported",
                    code="INVALID_FILE_TYPE"
                ))
                return result

            # Check size (max 10MB)
            if len(code_content) > 10 * 1024 * 1024:
                result.issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="File size exceeds 10MB limit",
                    code="FILE_TOO_LARGE"
                ))
                return result

            result.metadata = {
                "file_path": file_path,
                "code_hash": code_hash,
                "size": len(code_content)
            }
            result.passed = True

            logger.info(f"Code uploaded for draft {draft_id}: {filename} ({len(code_content)} bytes)")

        except Exception as e:
            logger.error(f"Error uploading code for draft {draft_id}: {e}")
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Upload failed: {str(e)}",
                code="UPLOAD_ERROR"
            ))

        return result

    def extract_metadata(
        self,
        draft_id: str,
        file_path: str
    ) -> Tuple[ValidationResult, Optional[AgentMetadata]]:
        """
        Extract metadata from uploaded code.

        Args:
            draft_id: Draft agent UUID
            file_path: Path to code file

        Returns:
            Tuple of (validation result, metadata)
        """
        result = ValidationResult(passed=False, stage=PublishStage.METADATA_EXTRACTION)

        try:
            metadata = MetadataExtractor.extract_from_file(file_path)

            # Update draft with metadata
            draft = self.session.query(MarketplaceAgent).filter(
                MarketplaceAgent.id == draft_id
            ).first()

            if draft:
                draft.name = metadata.name
                draft.description = metadata.description
                draft.slug = self._generate_slug(metadata.name)
                self.session.commit()

            result.passed = True
            result.metadata = {
                "name": metadata.name,
                "version": metadata.version,
                "author": metadata.author
            }

            logger.info(f"Metadata extracted for draft {draft_id}: {metadata.name} v{metadata.version}")

            return result, metadata

        except Exception as e:
            logger.error(f"Error extracting metadata for draft {draft_id}: {e}")
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Metadata extraction failed: {str(e)}",
                code="METADATA_ERROR"
            ))
            return result, None

    def validate_code(
        self,
        file_path: str
    ) -> ValidationResult:
        """
        Validate agent code structure.

        Args:
            file_path: Path to code file

        Returns:
            Validation result
        """
        result = ValidationResult(passed=False, stage=PublishStage.VALIDATION)

        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # Use validator
        validation_result = self.validator.validate_structure(source)

        # Convert to our format
        for error in validation_result.errors:
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=error,
                code="VALIDATION_ERROR"
            ))

        for warning in validation_result.warnings:
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=warning,
                code="VALIDATION_WARNING"
            ))

        result.passed = len(validation_result.errors) == 0
        result.metadata = validation_result.metadata

        return result

    def scan_security(
        self,
        file_path: str
    ) -> ValidationResult:
        """
        Perform security scan on code.

        Args:
            file_path: Path to code file

        Returns:
            Validation result
        """
        result = ValidationResult(passed=False, stage=PublishStage.SECURITY_SCAN)

        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # Use security scanner
        scan_result = self.security_scanner.scan(source)

        # Convert to our format
        for vuln in scan_result.vulnerabilities:
            severity = (
                ValidationSeverity.ERROR if vuln.get("severity") == "high"
                else ValidationSeverity.WARNING
            )
            result.issues.append(ValidationIssue(
                severity=severity,
                message=vuln.get("message", "Security issue detected"),
                code=vuln.get("code", "SECURITY_ISSUE"),
                line=vuln.get("line")
            ))

        result.passed = not any(
            i.severity == ValidationSeverity.ERROR for i in result.issues
        )

        return result

    def test_performance(
        self,
        file_path: str
    ) -> ValidationResult:
        """
        Test agent performance.

        Args:
            file_path: Path to code file

        Returns:
            Validation result with performance metrics
        """
        result = ValidationResult(passed=False, stage=PublishStage.PERFORMANCE_TEST)

        try:
            # In a real implementation, this would run the agent in a sandbox
            # and measure actual performance. For now, we'll do static analysis.

            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            # Count lines of code
            lines = [l for l in source.split('\n') if l.strip() and not l.strip().startswith('#')]
            loc = len(lines)

            # Estimate complexity
            tree = ast.parse(source)
            complexity = sum(1 for _ in ast.walk(tree))

            result.metadata = {
                "lines_of_code": loc,
                "complexity": complexity,
                "estimated_execution_time_ms": complexity * 0.1,  # Rough estimate
                "estimated_memory_mb": loc * 0.01  # Rough estimate
            }

            # Warn if too complex
            if complexity > 1000:
                result.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="Agent complexity is high, which may affect performance",
                    code="HIGH_COMPLEXITY",
                    details={"complexity": complexity}
                ))

            result.passed = True

            logger.info(f"Performance test completed: {loc} LOC, complexity {complexity}")

        except Exception as e:
            logger.error(f"Error testing performance: {e}")
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Performance test failed: {str(e)}",
                code="PERFORMANCE_ERROR"
            ))

        return result

    def validate_documentation(
        self,
        readme: Optional[str],
        metadata: AgentMetadata
    ) -> ValidationResult:
        """
        Validate documentation completeness.

        Args:
            readme: README content
            metadata: Agent metadata

        Returns:
            Validation result
        """
        result = ValidationResult(passed=False, stage=PublishStage.DOCUMENTATION_CHECK)

        # Check README
        if not readme or len(readme) < 500:
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="README must be at least 500 characters",
                code="README_TOO_SHORT"
            ))
        else:
            # Check for required sections
            required_sections = ["Installation", "Usage", "Example"]
            for section in required_sections:
                if section.lower() not in readme.lower():
                    result.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"README should include a '{section}' section",
                        code="MISSING_SECTION",
                        details={"section": section}
                    ))

        # Check metadata completeness
        if not metadata.description or len(metadata.description) < 20:
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Description must be at least 20 characters",
                code="DESCRIPTION_TOO_SHORT"
            ))

        if not metadata.license:
            result.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="License not specified",
                code="NO_LICENSE"
            ))

        result.passed = len(result.errors) == 0

        return result

    def publish_version(
        self,
        draft_id: str,
        version: str,
        code_hash: str,
        metadata: AgentMetadata,
        changelog: Optional[str] = None
    ) -> Tuple[bool, Optional[str], List[str]]:
        """
        Publish a new version of the agent.

        Args:
            draft_id: Draft agent UUID
            version: Version string (semver)
            code_hash: SHA-256 hash of code
            metadata: Agent metadata
            changelog: Optional changelog

        Returns:
            Tuple of (success, version_id, errors)
        """
        errors = []

        try:
            # Parse version
            sem_ver = SemanticVersion.parse(version)

            # Get draft
            draft = self.session.query(MarketplaceAgent).filter(
                MarketplaceAgent.id == draft_id
            ).first()

            if not draft:
                return False, None, ["Draft not found"]

            # Check for breaking changes if updating existing agent
            breaking_changes = False
            if draft.status == AgentStatus.PUBLISHED.value:
                detector = BreakingChangeDetector(self.session)
                breaking_changes = detector.has_breaking_changes(
                    str(draft.id),
                    metadata.input_schema,
                    metadata.output_schema
                )

            # Create version
            agent_version = AgentVersion(
                agent_id=draft.id,
                version=version,
                version_major=sem_ver.major,
                version_minor=sem_ver.minor,
                version_patch=sem_ver.patch,
                code_hash=code_hash,
                changelog=changelog,
                dependencies_json=metadata.dependencies,
                schema_input=metadata.input_schema,
                schema_output=metadata.output_schema,
                breaking_changes=breaking_changes
            )

            self.session.add(agent_version)

            # Update agent status
            if draft.status == AgentStatus.DRAFT.value:
                draft.status = AgentStatus.PUBLISHED.value
                draft.published_at = datetime.utcnow()

            draft.last_version_at = datetime.utcnow()

            self.session.commit()

            logger.info(f"Published version {version} for agent {draft_id}")

            return True, str(agent_version.id), []

        except Exception as e:
            logger.error(f"Error publishing version: {e}")
            errors.append(f"Publication failed: {str(e)}")
            return False, None, errors

    def _generate_slug(self, name: str) -> str:
        """Generate URL-friendly slug from name"""
        import re
        slug = name.lower()
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        slug = slug.strip('-')
        return slug


class AgentPublisher:
    """
    High-level agent publisher interface.

    Provides simplified API for the publishing workflow.
    """

    def __init__(self, session: Session):
        self.session = session
        self.workflow = PublishingWorkflow(session)

    def create_draft(
        self,
        author_id: str,
        author_name: str
    ) -> Dict[str, Any]:
        """
        Create a new draft agent.

        Args:
            author_id: Author user UUID
            author_name: Author name

        Returns:
            Dictionary with draft_id and checklist
        """
        draft_id, checklist = self.workflow.start_publishing(author_id, author_name)

        return {
            "draft_id": draft_id,
            "checklist": {
                "code_uploaded": checklist.code_uploaded,
                "metadata_extracted": checklist.metadata_extracted,
                "validation_passed": checklist.validation_passed,
                "security_passed": checklist.security_passed,
                "performance_tested": checklist.performance_tested,
                "documentation_complete": checklist.documentation_complete,
                "assets_uploaded": checklist.assets_uploaded,
                "pricing_configured": checklist.pricing_configured,
                "ready_to_publish": checklist.ready_to_publish
            }
        }

    def validate_and_upload(
        self,
        draft_id: str,
        code_content: bytes,
        filename: str,
        readme: str
    ) -> Dict[str, Any]:
        """
        Validate and upload agent code.

        Runs all validation steps and returns comprehensive results.

        Args:
            draft_id: Draft agent UUID
            code_content: Code file bytes
            filename: Original filename
            readme: README content

        Returns:
            Dictionary with validation results
        """
        results = {
            "success": False,
            "stages": {},
            "metadata": None,
            "checklist": None
        }

        # Stage 1: Upload
        upload_result = self.workflow.upload_code(draft_id, code_content, filename)
        results["stages"]["upload"] = self._result_to_dict(upload_result)

        if not upload_result.passed:
            return results

        file_path = upload_result.metadata["file_path"]
        code_hash = upload_result.metadata["code_hash"]

        # Stage 2: Extract metadata
        metadata_result, metadata = self.workflow.extract_metadata(draft_id, file_path)
        results["stages"]["metadata"] = self._result_to_dict(metadata_result)

        if not metadata_result.passed or not metadata:
            return results

        results["metadata"] = {
            "name": metadata.name,
            "version": metadata.version,
            "author": metadata.author,
            "description": metadata.description
        }

        # Stage 3: Validate code
        validation_result = self.workflow.validate_code(file_path)
        results["stages"]["validation"] = self._result_to_dict(validation_result)

        # Stage 4: Security scan
        security_result = self.workflow.scan_security(file_path)
        results["stages"]["security"] = self._result_to_dict(security_result)

        # Stage 5: Performance test
        performance_result = self.workflow.test_performance(file_path)
        results["stages"]["performance"] = self._result_to_dict(performance_result)

        # Stage 6: Documentation
        doc_result = self.workflow.validate_documentation(readme, metadata)
        results["stages"]["documentation"] = self._result_to_dict(doc_result)

        # Determine overall success
        results["success"] = (
            validation_result.passed and
            security_result.passed and
            doc_result.passed
        )

        # Update checklist
        results["checklist"] = {
            "code_uploaded": True,
            "metadata_extracted": metadata_result.passed,
            "validation_passed": validation_result.passed,
            "security_passed": security_result.passed,
            "performance_tested": performance_result.passed,
            "documentation_complete": doc_result.passed,
            "ready_to_publish": results["success"]
        }

        return results

    def _result_to_dict(self, result: ValidationResult) -> Dict[str, Any]:
        """Convert ValidationResult to dictionary"""
        return {
            "passed": result.passed,
            "errors": [
                {"message": i.message, "code": i.code, "line": i.line}
                for i in result.errors
            ],
            "warnings": [
                {"message": i.message, "code": i.code, "line": i.line}
                for i in result.warnings
            ],
            "metadata": result.metadata
        }
