# -*- coding: utf-8 -*-
"""
Zero-Hallucination Architecture for GreenLang
Ensures LLMs never generate false numeric data for regulatory compliance
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from greenlang.determinism import DeterministicClock

# Configure deterministic behavior
DETERMINISTIC_SEED = 42
np.random.seed(DETERMINISTIC_SEED)

class TaskType(Enum):
    """Approved LLM task types"""
    ENTITY_RESOLUTION = "entity_resolution"
    CLASSIFICATION = "classification"
    MATERIALITY_ASSESSMENT = "materiality_assessment"
    DOCUMENT_EXTRACTION = "document_extraction"
    NARRATIVE_GENERATION = "narrative_generation"
    CODE_GENERATION = "code_generation"

class DataTier(Enum):
    """Data quality tiers for transparency"""
    TIER1_ACTUAL = "actual_data"  # Real measurements
    TIER2_ESTIMATED = "ai_estimated"  # AI-based estimation
    TIER3_PROXY = "proxy_data"  # Industry averages

@dataclass
class LLMRequest:
    """Structured LLM request with validation"""
    task_type: TaskType
    prompt: str
    context: Dict[str, Any]
    max_tokens: int = 500
    temperature: float = 0.0  # Deterministic by default
    confidence_threshold: float = 0.80
    require_validation: bool = True
    audit_required: bool = True

@dataclass
class LLMResponse:
    """Validated LLM response with provenance"""
    request_id: str
    task_type: TaskType
    raw_output: str
    structured_output: Dict[str, Any]
    confidence_score: float
    data_tier: DataTier
    validation_passed: bool
    provenance: Dict[str, Any]
    timestamp: datetime
    model_used: str
    tokens_used: int
    cost: float

class ZeroHallucinationOrchestrator:
    """
    Main orchestrator ensuring LLMs never hallucinate numeric values
    """

    def __init__(self):
        self.validators = self._initialize_validators()
        self.audit_logger = AuditLogger()
        self.provenance_tracker = ProvenanceTracker()
        self.calculation_engine = DeterministicCalculationEngine()

    def process_request(self, request: LLMRequest) -> LLMResponse:
        """Process LLM request with full validation pipeline"""

        # Step 1: Validate request type is approved
        if not self._is_approved_task(request.task_type):
            raise ValueError(f"Task type {request.task_type} not approved for LLM processing")

        # Step 2: Pre-process and sanitize
        sanitized_prompt = self._sanitize_prompt(request.prompt)
        request_id = self._generate_request_id(request)

        # Step 3: Check if numeric calculation is attempted
        if self._contains_calculation_request(sanitized_prompt):
            # Redirect to deterministic engine
            return self._handle_calculation_request(request, request_id)

        # Step 4: Execute LLM call with deterministic settings
        raw_output = self._execute_llm_call(
            prompt=sanitized_prompt,
            temperature=0.0,  # Always deterministic for consistency
            max_tokens=request.max_tokens,
            seed=DETERMINISTIC_SEED
        )

        # Step 5: Parse and structure output
        structured_output = self._parse_output(raw_output, request.task_type)

        # Step 6: Validate output
        validation_result = self._validate_output(
            structured_output,
            request.task_type,
            request.confidence_threshold
        )

        # Step 7: Track provenance
        provenance = self.provenance_tracker.track(
            request_id=request_id,
            input_data=request.context,
            output_data=structured_output,
            model_used=self._get_model_info(),
            timestamp=DeterministicClock.utcnow()
        )

        # Step 8: Create response
        response = LLMResponse(
            request_id=request_id,
            task_type=request.task_type,
            raw_output=raw_output,
            structured_output=structured_output,
            confidence_score=validation_result['confidence'],
            data_tier=self._determine_data_tier(structured_output),
            validation_passed=validation_result['passed'],
            provenance=provenance,
            timestamp=DeterministicClock.utcnow(),
            model_used=self._get_model_info(),
            tokens_used=self._count_tokens(sanitized_prompt, raw_output),
            cost=self._calculate_cost()
        )

        # Step 9: Audit log
        if request.audit_required:
            self.audit_logger.log(response)

        return response

    def _is_approved_task(self, task_type: TaskType) -> bool:
        """Check if task type is approved for LLM processing"""
        approved_tasks = {
            TaskType.ENTITY_RESOLUTION,
            TaskType.CLASSIFICATION,
            TaskType.MATERIALITY_ASSESSMENT,
            TaskType.DOCUMENT_EXTRACTION,
            TaskType.NARRATIVE_GENERATION,
            TaskType.CODE_GENERATION
        }
        return task_type in approved_tasks

    def _contains_calculation_request(self, prompt: str) -> bool:
        """Detect if prompt asks for numeric calculations"""
        calculation_indicators = [
            'calculate', 'compute', 'sum', 'total', 'emissions',
            'carbon', 'co2', 'ghg', 'footprint', 'metric',
            'percentage', 'ratio', 'average', 'mean'
        ]
        prompt_lower = prompt.lower()
        return any(indicator in prompt_lower for indicator in calculation_indicators)

    def _handle_calculation_request(self, request: LLMRequest, request_id: str) -> LLMResponse:
        """Redirect calculation requests to deterministic engine"""
        # Use deterministic calculation engine
        result = self.calculation_engine.calculate(request.context)

        return LLMResponse(
            request_id=request_id,
            task_type=request.task_type,
            raw_output="CALCULATION_REDIRECTED",
            structured_output=result,
            confidence_score=1.0,  # Deterministic calculations have 100% confidence
            data_tier=DataTier.TIER1_ACTUAL,
            validation_passed=True,
            provenance={'engine': 'deterministic', 'formula': result.get('formula')},
            timestamp=DeterministicClock.utcnow(),
            model_used='DETERMINISTIC_ENGINE',
            tokens_used=0,
            cost=0.0
        )

    def _sanitize_prompt(self, prompt: str) -> str:
        """Remove potentially harmful content from prompt"""
        # Remove PII patterns
        prompt = self._remove_pii(prompt)
        # Remove injection attempts
        prompt = self._prevent_injection(prompt)
        return prompt

    def _generate_request_id(self, request: LLMRequest) -> str:
        """Generate unique request ID for tracking"""
        content = f"{request.task_type.value}_{request.prompt}_{DeterministicClock.utcnow().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class DeterministicCalculationEngine:
    """
    Handles all numeric calculations with verified formulas
    NEVER uses LLMs for calculations
    """

    def __init__(self):
        self.emission_factors = self._load_emission_factors()
        self.formulas = self._load_verified_formulas()

    def calculate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform deterministic calculation based on context"""

        calculation_type = context.get('calculation_type')

        if calculation_type == 'scope1_emissions':
            return self._calculate_scope1(context)
        elif calculation_type == 'scope2_emissions':
            return self._calculate_scope2(context)
        elif calculation_type == 'scope3_emissions':
            return self._calculate_scope3(context)
        elif calculation_type == 'compliance_metric':
            return self._calculate_compliance_metric(context)
        else:
            raise ValueError(f"Unknown calculation type: {calculation_type}")

    def _calculate_scope1(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Scope 1 emissions using verified formulas"""
        fuel_consumed = context.get('fuel_consumed', 0)
        fuel_type = context.get('fuel_type', 'diesel')
        emission_factor = self.emission_factors['scope1'][fuel_type]

        emissions = fuel_consumed * emission_factor

        return {
            'value': emissions,
            'unit': 'tCO2e',
            'formula': f"{fuel_consumed} * {emission_factor}",
            'source': 'EPA Emission Factors 2024',
            'confidence': 1.0,
            'tier': DataTier.TIER1_ACTUAL.value
        }


class OutputValidator:
    """
    Validates LLM outputs to ensure no hallucinated values
    """

    def __init__(self):
        self.confidence_calculator = ConfidenceCalculator()
        self.fact_checker = FactChecker()

    def validate(self, output: Dict[str, Any], task_type: TaskType, threshold: float) -> Dict[str, Any]:
        """Comprehensive output validation"""

        validation_results = {
            'passed': True,
            'confidence': 1.0,
            'issues': [],
            'warnings': []
        }

        # Check for numeric values in non-calculation tasks
        if task_type != TaskType.CODE_GENERATION:
            numeric_check = self._check_numeric_claims(output)
            if numeric_check['has_unverified_numbers']:
                validation_results['passed'] = False
                validation_results['issues'].append('Contains unverified numeric claims')

        # Calculate confidence score
        confidence = self.confidence_calculator.calculate(output, task_type)
        validation_results['confidence'] = confidence

        if confidence < threshold:
            validation_results['passed'] = False
            validation_results['issues'].append(f'Confidence {confidence:.2f} below threshold {threshold}')

        # Fact-check against knowledge base
        if task_type in [TaskType.ENTITY_RESOLUTION, TaskType.CLASSIFICATION]:
            fact_check = self.fact_checker.verify(output)
            if not fact_check['verified']:
                validation_results['warnings'].append('Could not verify against knowledge base')

        return validation_results

    def _check_numeric_claims(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Detect unverified numeric claims in output"""
        import re

        numeric_pattern = r'\d+\.?\d*\s*(?:kg|t|ton|tonne|CO2|GHG|%|percent)'
        text = json.dumps(output)

        matches = re.findall(numeric_pattern, text, re.IGNORECASE)

        return {
            'has_unverified_numbers': len(matches) > 0,
            'found_numbers': matches
        }


class ProvenanceTracker:
    """
    Tracks the complete lineage of all data and decisions
    """

    def __init__(self):
        self.provenance_store = {}

    def track(self, request_id: str, input_data: Dict, output_data: Dict,
              model_used: str, timestamp: datetime) -> Dict[str, Any]:
        """Create comprehensive provenance record"""

        provenance = {
            'request_id': request_id,
            'timestamp': timestamp.isoformat(),
            'model': {
                'name': model_used,
                'version': self._get_model_version(model_used),
                'parameters': {
                    'temperature': 0.0,
                    'seed': DETERMINISTIC_SEED
                }
            },
            'input': {
                'data': self._hash_sensitive_data(input_data),
                'sources': self._extract_sources(input_data)
            },
            'output': {
                'data': self._hash_sensitive_data(output_data),
                'confidence': output_data.get('confidence', 0.0)
            },
            'lineage': self._build_lineage_graph(request_id)
        }

        self.provenance_store[request_id] = provenance
        return provenance

    def _hash_sensitive_data(self, data: Dict) -> str:
        """Hash sensitive data for privacy"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _extract_sources(self, data: Dict) -> List[str]:
        """Extract data sources from input"""
        sources = []
        if 'sources' in data:
            sources.extend(data['sources'])
        if 'database' in data:
            sources.append(f"Database: {data['database']}")
        return sources

    def _build_lineage_graph(self, request_id: str) -> Dict:
        """Build data lineage graph"""
        return {
            'upstream': self._get_upstream_dependencies(request_id),
            'downstream': self._get_downstream_impacts(request_id)
        }


class AuditLogger:
    """
    Comprehensive audit logging for compliance
    """

    def __init__(self):
        self.logger = logging.getLogger('greenlang.llm.audit')
        self.logger.setLevel(logging.INFO)

    def log(self, response: LLMResponse):
        """Log LLM response for audit trail"""

        audit_entry = {
            'timestamp': response.timestamp.isoformat(),
            'request_id': response.request_id,
            'task_type': response.task_type.value,
            'model_used': response.model_used,
            'confidence_score': response.confidence_score,
            'validation_passed': response.validation_passed,
            'data_tier': response.data_tier.value,
            'tokens_used': response.tokens_used,
            'cost': response.cost,
            'provenance_hash': hashlib.sha256(
                json.dumps(response.provenance, sort_keys=True).encode()
            ).hexdigest()
        }

        self.logger.info(f"LLM_AUDIT: {json.dumps(audit_entry)}")

        # Store in audit database
        self._store_audit_entry(audit_entry)

    def _store_audit_entry(self, entry: Dict):
        """Store audit entry in persistent storage"""
        # Implementation would connect to audit database
        pass


class ConfidenceCalculator:
    """
    Calculate confidence scores for LLM outputs
    """

    def calculate(self, output: Dict[str, Any], task_type: TaskType) -> float:
        """Calculate confidence score based on multiple factors"""

        base_confidence = 0.5
        adjustments = []

        # Task-specific confidence
        task_confidence = {
            TaskType.ENTITY_RESOLUTION: 0.85,
            TaskType.CLASSIFICATION: 0.90,
            TaskType.MATERIALITY_ASSESSMENT: 0.75,
            TaskType.DOCUMENT_EXTRACTION: 0.80,
            TaskType.NARRATIVE_GENERATION: 0.70,
            TaskType.CODE_GENERATION: 0.85
        }
        adjustments.append(task_confidence.get(task_type, 0.70))

        # Output structure validation
        if self._has_required_fields(output, task_type):
            adjustments.append(0.10)

        # Consistency check
        if self._is_consistent(output):
            adjustments.append(0.05)

        # Calculate final confidence
        confidence = base_confidence + sum(adjustments)
        return min(confidence, 1.0)

    def _has_required_fields(self, output: Dict, task_type: TaskType) -> bool:
        """Check if output has required fields for task type"""
        required_fields = {
            TaskType.ENTITY_RESOLUTION: ['entity_id', 'match_score'],
            TaskType.CLASSIFICATION: ['category', 'subcategory'],
            TaskType.MATERIALITY_ASSESSMENT: ['impact_score', 'likelihood']
        }
        fields = required_fields.get(task_type, [])
        return all(field in output for field in fields)

    def _is_consistent(self, output: Dict) -> bool:
        """Check internal consistency of output"""
        # Implementation would check for logical consistency
        return True


class FactChecker:
    """
    Verify LLM outputs against known facts
    """

    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()

    def verify(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Verify output against knowledge base"""

        verification_result = {
            'verified': False,
            'confidence': 0.0,
            'sources': []
        }

        # Check entity resolution
        if 'entity_id' in output:
            entity_verified = self._verify_entity(output['entity_id'])
            if entity_verified:
                verification_result['verified'] = True
                verification_result['sources'].append('entity_database')

        # Check classification
        if 'category' in output:
            category_verified = self._verify_category(output['category'])
            if category_verified:
                verification_result['verified'] = True
                verification_result['sources'].append('category_taxonomy')

        return verification_result

    def _verify_entity(self, entity_id: str) -> bool:
        """Verify entity exists in database"""
        # Would query actual database
        return True

    def _verify_category(self, category: str) -> bool:
        """Verify category in taxonomy"""
        # Would check against official taxonomy
        return True

    def _load_knowledge_base(self) -> Dict:
        """Load verified knowledge base"""
        return {
            'entities': {},
            'categories': {},
            'emission_factors': {}
        }


# Export main orchestrator
__all__ = ['ZeroHallucinationOrchestrator', 'LLMRequest', 'LLMResponse', 'TaskType', 'DataTier']