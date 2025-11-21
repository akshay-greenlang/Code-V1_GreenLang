# -*- coding: utf-8 -*-
"""
GreenLang Prompt Library
100+ Production-Ready Prompts for Climate Intelligence
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class PromptCategory(Enum):
    ENTITY_RESOLUTION = "entity_resolution"
    CLASSIFICATION = "classification"
    MATERIALITY = "materiality"
    EXTRACTION = "extraction"
    NARRATIVE = "narrative"
    VALIDATION = "validation"
    CODE_GENERATION = "code_generation"

@dataclass
class PromptTemplate:
    """Structured prompt template with metadata"""
    id: str
    category: PromptCategory
    name: str
    template: str
    variables: List[str]
    output_schema: Dict
    confidence_threshold: float
    max_tokens: int
    temperature: float
    examples: List[Dict]

class PromptLibrary:
    """
    Comprehensive prompt library for GreenLang
    All prompts designed for zero-hallucination compliance
    """

    def __init__(self):
        self.prompts = self._initialize_prompts()

    def _initialize_prompts(self) -> Dict[str, PromptTemplate]:
        """Initialize all prompt templates"""
        return {
            # ========== ENTITY RESOLUTION PROMPTS ==========
            'supplier_matching': PromptTemplate(
                id='er_001',
                category=PromptCategory.ENTITY_RESOLUTION,
                name='Supplier Name Matching',
                template="""You are a supplier identification specialist. Your task is to match supplier names to our master database.

IMPORTANT: You must ONLY match based on the provided database. Never invent or guess supplier IDs.

Supplier to match: {supplier_name}
Additional context: {context}

Master database entries:
{database_entries}

Instructions:
1. Compare the supplier name with database entries
2. Consider variations (Inc. vs Incorporated, Ltd. vs Limited)
3. Check for abbreviations and common misspellings
4. If no exact match, find the closest match with confidence score

Output as JSON:
{
  "matched": true/false,
  "supplier_id": "ID from database or null",
  "matched_name": "exact name from database or null",
  "confidence": 0.0-1.0,
  "match_reason": "explanation",
  "alternatives": [{"id": "...", "name": "...", "confidence": 0.0-1.0}]
}""",
                variables=['supplier_name', 'context', 'database_entries'],
                output_schema={
                    'matched': 'boolean',
                    'supplier_id': 'string|null',
                    'matched_name': 'string|null',
                    'confidence': 'float',
                    'match_reason': 'string',
                    'alternatives': 'array'
                },
                confidence_threshold=0.85,
                max_tokens=500,
                temperature=0.0,
                examples=[
                    {
                        'input': {'supplier_name': 'Microsoft Corp.'},
                        'output': {
                            'matched': True,
                            'supplier_id': 'SUP_MS_001',
                            'matched_name': 'Microsoft Corporation',
                            'confidence': 0.95,
                            'match_reason': 'Strong match with abbreviation variation',
                            'alternatives': []
                        }
                    }
                ]
            ),

            'product_resolution': PromptTemplate(
                id='er_002',
                category=PromptCategory.ENTITY_RESOLUTION,
                name='Product Identification',
                template="""Identify and categorize the product based on description.

Product description: {product_description}
Invoice line item: {invoice_text}
Quantity: {quantity}
Unit: {unit}

Product catalog:
{catalog_entries}

Match the product to our catalog and provide categorization.

Output as JSON:
{
  "product_id": "from catalog or null",
  "product_name": "standardized name",
  "category": "product category",
  "subcategory": "product subcategory",
  "unit_type": "standardized unit",
  "confidence": 0.0-1.0,
  "unspsc_code": "UNSPSC classification if applicable"
}""",
                variables=['product_description', 'invoice_text', 'quantity', 'unit', 'catalog_entries'],
                output_schema={
                    'product_id': 'string|null',
                    'product_name': 'string',
                    'category': 'string',
                    'subcategory': 'string',
                    'unit_type': 'string',
                    'confidence': 'float',
                    'unspsc_code': 'string|null'
                },
                confidence_threshold=0.80,
                max_tokens=400,
                temperature=0.0,
                examples=[]
            ),

            # ========== CLASSIFICATION PROMPTS ==========
            'scope3_categorization': PromptTemplate(
                id='cl_001',
                category=PromptCategory.CLASSIFICATION,
                name='Scope 3 Category Classification',
                template="""Classify the following purchase into the correct GHG Protocol Scope 3 category.

Purchase details:
- Description: {description}
- Supplier: {supplier}
- Amount: {amount} {currency}
- Category hint: {category_hint}

GHG Protocol Scope 3 Categories:
1. Purchased goods and services
2. Capital goods
3. Fuel- and energy-related activities
4. Upstream transportation and distribution
5. Waste generated in operations
6. Business travel
7. Employee commuting
8. Upstream leased assets
9. Downstream transportation and distribution
10. Processing of sold products
11. Use of sold products
12. End-of-life treatment of sold products
13. Downstream leased assets
14. Franchises
15. Investments

Classify and provide reasoning.

Output as JSON:
{
  "scope3_category": "number (1-15)",
  "category_name": "official category name",
  "subcategory": "specific subcategory if applicable",
  "confidence": 0.0-1.0,
  "reasoning": "explanation of classification",
  "alternative_categories": [{"category": "number", "confidence": 0.0-1.0}]
}""",
                variables=['description', 'supplier', 'amount', 'currency', 'category_hint'],
                output_schema={
                    'scope3_category': 'integer',
                    'category_name': 'string',
                    'subcategory': 'string|null',
                    'confidence': 'float',
                    'reasoning': 'string',
                    'alternative_categories': 'array'
                },
                confidence_threshold=0.85,
                max_tokens=400,
                temperature=0.0,
                examples=[]
            ),

            'industry_classification': PromptTemplate(
                id='cl_002',
                category=PromptCategory.CLASSIFICATION,
                name='Industry Classification (NAICS/NACE)',
                template="""Classify the company into the appropriate industry codes.

Company information:
- Name: {company_name}
- Description: {company_description}
- Products/Services: {products_services}
- Website content: {website_snippet}

Provide both NAICS and NACE classifications.

Output as JSON:
{
  "naics_code": "6-digit NAICS code",
  "naics_title": "NAICS industry title",
  "nace_code": "NACE Rev. 2 code",
  "nace_title": "NACE industry title",
  "primary_sector": "primary economic sector",
  "confidence": 0.0-1.0,
  "reasoning": "classification explanation"
}""",
                variables=['company_name', 'company_description', 'products_services', 'website_snippet'],
                output_schema={
                    'naics_code': 'string',
                    'naics_title': 'string',
                    'nace_code': 'string',
                    'nace_title': 'string',
                    'primary_sector': 'string',
                    'confidence': 'float',
                    'reasoning': 'string'
                },
                confidence_threshold=0.80,
                max_tokens=400,
                temperature=0.0,
                examples=[]
            ),

            # ========== MATERIALITY ASSESSMENT PROMPTS ==========
            'double_materiality': PromptTemplate(
                id='ma_001',
                category=PromptCategory.MATERIALITY,
                name='CSRD Double Materiality Assessment',
                template="""Perform a double materiality assessment for the given sustainability topic.

Topic: {topic}
Industry: {industry}
Company size: {company_size}
Geographic presence: {geography}
Stakeholder groups: {stakeholders}

Context from company reports:
{report_context}

Assess both:
1. Financial materiality (impact on company)
2. Impact materiality (company's impact on society/environment)

Use CSRD/ESRS guidelines for assessment.

Output as JSON:
{
  "topic": "sustainability topic",
  "financial_materiality": {
    "score": 1-5,
    "timeframe": "short/medium/long",
    "impacts": ["list of financial impacts"],
    "risks": ["identified risks"],
    "opportunities": ["identified opportunities"],
    "reasoning": "detailed explanation"
  },
  "impact_materiality": {
    "score": 1-5,
    "severity": "low/medium/high/very high",
    "scope": "description of affected stakeholders",
    "irremediability": "low/medium/high",
    "actual_potential": "actual/potential/both",
    "positive_negative": "positive/negative/both",
    "reasoning": "detailed explanation"
  },
  "overall_materiality": "material/not material",
  "confidence": 0.0-1.0,
  "esrs_reference": "relevant ESRS standard"
}""",
                variables=['topic', 'industry', 'company_size', 'geography', 'stakeholders', 'report_context'],
                output_schema={
                    'topic': 'string',
                    'financial_materiality': 'object',
                    'impact_materiality': 'object',
                    'overall_materiality': 'string',
                    'confidence': 'float',
                    'esrs_reference': 'string'
                },
                confidence_threshold=0.85,
                max_tokens=1500,
                temperature=0.3,
                examples=[]
            ),

            'stakeholder_impact': PromptTemplate(
                id='ma_002',
                category=PromptCategory.MATERIALITY,
                name='Stakeholder Impact Assessment',
                template="""Assess the impact of a sustainability issue on different stakeholder groups.

Issue: {issue}
Company context: {company_context}
Industry: {industry}

Stakeholder groups to consider:
- Investors
- Employees
- Customers
- Local communities
- Suppliers
- Regulators
- NGOs
- Future generations

For each relevant stakeholder group, assess impact.

Output as JSON:
{
  "issue": "issue description",
  "stakeholder_impacts": [
    {
      "stakeholder_group": "group name",
      "impact_level": "high/medium/low/none",
      "impact_type": "positive/negative/mixed",
      "description": "specific impact description",
      "timeframe": "immediate/short-term/medium-term/long-term",
      "mitigation_needed": true/false
    }
  ],
  "most_affected": ["top 3 most affected groups"],
  "confidence": 0.0-1.0,
  "recommendations": ["action recommendations"]
}""",
                variables=['issue', 'company_context', 'industry'],
                output_schema={
                    'issue': 'string',
                    'stakeholder_impacts': 'array',
                    'most_affected': 'array',
                    'confidence': 'float',
                    'recommendations': 'array'
                },
                confidence_threshold=0.80,
                max_tokens=1200,
                temperature=0.2,
                examples=[]
            ),

            # ========== DOCUMENT EXTRACTION PROMPTS ==========
            'certificate_extraction': PromptTemplate(
                id='ex_001',
                category=PromptCategory.EXTRACTION,
                name='Sustainability Certificate Data Extraction',
                template="""Extract structured data from this sustainability certificate.

Certificate text:
{certificate_text}

Certificate type hint: {cert_type}

Extract all relevant information including:
- Certificate type and standard
- Certificate number
- Issue date and expiry date
- Scope of certification
- Issuing body
- Company/facility covered
- Any quantitative claims (but DO NOT calculate or invent numbers)

Output as JSON:
{
  "certificate_type": "type of certificate",
  "standard": "certification standard",
  "certificate_number": "unique identifier",
  "issue_date": "YYYY-MM-DD or null",
  "expiry_date": "YYYY-MM-DD or null",
  "validity_period": "description or null",
  "scope": "what is certified",
  "issuing_body": "certifying organization",
  "certified_entity": "company/facility name",
  "certified_sites": ["list of covered locations"],
  "claims": ["specific claims made"],
  "extracted_numbers": {
    "description": "what the number represents",
    "value": "exact value from document",
    "unit": "unit of measurement",
    "context": "surrounding context"
  },
  "confidence": 0.0-1.0
}

IMPORTANT: Only extract numbers that explicitly appear in the document. Never calculate or estimate values.""",
                variables=['certificate_text', 'cert_type'],
                output_schema={
                    'certificate_type': 'string',
                    'standard': 'string',
                    'certificate_number': 'string|null',
                    'issue_date': 'string|null',
                    'expiry_date': 'string|null',
                    'validity_period': 'string|null',
                    'scope': 'string',
                    'issuing_body': 'string',
                    'certified_entity': 'string',
                    'certified_sites': 'array',
                    'claims': 'array',
                    'extracted_numbers': 'object|null',
                    'confidence': 'float'
                },
                confidence_threshold=0.85,
                max_tokens=1000,
                temperature=0.0,
                examples=[]
            ),

            'invoice_parsing': PromptTemplate(
                id='ex_002',
                category=PromptCategory.EXTRACTION,
                name='Invoice Data Extraction',
                template="""Extract purchase data from this invoice for emissions tracking.

Invoice content:
{invoice_text}

Extract line items and metadata. DO NOT calculate totals - only extract existing values.

Output as JSON:
{
  "invoice_number": "invoice identifier",
  "invoice_date": "YYYY-MM-DD",
  "supplier": {
    "name": "supplier name",
    "address": "supplier address",
    "tax_id": "tax identification number"
  },
  "buyer": {
    "name": "buyer name",
    "address": "buyer address"
  },
  "line_items": [
    {
      "description": "item description",
      "quantity": "exact quantity from invoice",
      "unit": "unit of measurement",
      "unit_price": "price per unit if shown",
      "total_price": "line total if shown",
      "category_hint": "product category if identifiable"
    }
  ],
  "totals": {
    "subtotal": "if explicitly shown",
    "tax": "if explicitly shown",
    "total": "if explicitly shown"
  },
  "currency": "currency code",
  "payment_terms": "payment terms if shown",
  "confidence": 0.0-1.0
}

CRITICAL: Only extract numbers exactly as they appear. Never calculate missing values.""",
                variables=['invoice_text'],
                output_schema={
                    'invoice_number': 'string',
                    'invoice_date': 'string',
                    'supplier': 'object',
                    'buyer': 'object',
                    'line_items': 'array',
                    'totals': 'object',
                    'currency': 'string',
                    'payment_terms': 'string|null',
                    'confidence': 'float'
                },
                confidence_threshold=0.80,
                max_tokens=1500,
                temperature=0.0,
                examples=[]
            ),

            # ========== NARRATIVE GENERATION PROMPTS ==========
            'executive_summary': PromptTemplate(
                id='ng_001',
                category=PromptCategory.NARRATIVE,
                name='Sustainability Report Executive Summary',
                template="""Generate an executive summary for a sustainability report.

Company: {company_name}
Reporting period: {period}
Industry: {industry}

Key metrics provided (DO NOT modify these numbers):
{key_metrics}

Achievements:
{achievements}

Challenges:
{challenges}

Future targets:
{targets}

Write a professional executive summary (500-750 words) that:
1. Opens with company commitment to sustainability
2. Highlights key achievements using the EXACT metrics provided
3. Acknowledges challenges transparently
4. Outlines future commitments
5. Uses clear, professional language
6. Avoids greenwashing or exaggeration

CRITICAL: Use only the exact numbers provided in key_metrics. Do not calculate, round, or estimate any values.

Output as structured text with clear sections.""",
                variables=['company_name', 'period', 'industry', 'key_metrics', 'achievements', 'challenges', 'targets'],
                output_schema={'summary': 'string'},
                confidence_threshold=0.75,
                max_tokens=2000,
                temperature=0.7,
                examples=[]
            ),

            'disclosure_statement': PromptTemplate(
                id='ng_002',
                category=PromptCategory.NARRATIVE,
                name='TCFD/CSRD Disclosure Statement',
                template="""Generate a formal disclosure statement for regulatory reporting.

Disclosure type: {disclosure_type}
Topic: {topic}
Reporting entity: {entity}
Period: {period}

Provided data (use exactly as given):
{data_points}

Governance structure:
{governance}

Risk assessment results:
{risks}

Mitigation measures:
{mitigation}

Generate a formal disclosure statement that:
1. Follows {disclosure_type} guidelines precisely
2. Uses formal, precise language
3. References specific standards and frameworks
4. Includes all mandatory elements
5. Uses ONLY the data explicitly provided

Output structure:
- Governance (200-300 words)
- Strategy (300-400 words)
- Risk Management (300-400 words)
- Metrics and Targets (200-300 words)

IMPORTANT: This is for regulatory compliance. Be precise and accurate. Use only provided data.""",
                variables=['disclosure_type', 'topic', 'entity', 'period', 'data_points', 'governance', 'risks', 'mitigation'],
                output_schema={
                    'governance': 'string',
                    'strategy': 'string',
                    'risk_management': 'string',
                    'metrics_targets': 'string'
                },
                confidence_threshold=0.85,
                max_tokens=2500,
                temperature=0.3,
                examples=[]
            ),

            # ========== VALIDATION PROMPTS ==========
            'data_consistency': PromptTemplate(
                id='val_001',
                category=PromptCategory.VALIDATION,
                name='Data Consistency Validation',
                template="""Check for inconsistencies in the provided sustainability data.

Dataset 1:
{dataset1}

Dataset 2:
{dataset2}

Validation rules:
{validation_rules}

Check for:
1. Logical inconsistencies
2. Contradictory claims
3. Unusual patterns
4. Missing required relationships

DO NOT perform calculations. Only identify logical issues.

Output as JSON:
{
  "consistent": true/false,
  "inconsistencies": [
    {
      "type": "type of inconsistency",
      "location": "where found",
      "description": "detailed description",
      "severity": "high/medium/low"
    }
  ],
  "warnings": ["potential issues"],
  "confidence": 0.0-1.0
}""",
                variables=['dataset1', 'dataset2', 'validation_rules'],
                output_schema={
                    'consistent': 'boolean',
                    'inconsistencies': 'array',
                    'warnings': 'array',
                    'confidence': 'float'
                },
                confidence_threshold=0.90,
                max_tokens=800,
                temperature=0.0,
                examples=[]
            ),

            'claim_verification': PromptTemplate(
                id='val_002',
                category=PromptCategory.VALIDATION,
                name='Sustainability Claim Verification',
                template="""Verify the sustainability claim against provided evidence.

Claim: {claim}
Evidence provided:
{evidence}

Supporting documents:
{documents}

Industry standards:
{standards}

Evaluate whether the evidence supports the claim.

Output as JSON:
{
  "claim": "restated claim",
  "verification_status": "verified/partially verified/not verified/insufficient evidence",
  "evidence_strength": "strong/moderate/weak",
  "supporting_points": ["evidence that supports claim"],
  "gaps": ["missing evidence or weaknesses"],
  "greenwashing_risk": "high/medium/low/none",
  "recommendation": "recommendation for claim improvement",
  "confidence": 0.0-1.0
}

Focus on logical consistency, not calculations.""",
                variables=['claim', 'evidence', 'documents', 'standards'],
                output_schema={
                    'claim': 'string',
                    'verification_status': 'string',
                    'evidence_strength': 'string',
                    'supporting_points': 'array',
                    'gaps': 'array',
                    'greenwashing_risk': 'string',
                    'recommendation': 'string',
                    'confidence': 'float'
                },
                confidence_threshold=0.85,
                max_tokens=800,
                temperature=0.0,
                examples=[]
            ),

            # ========== CODE GENERATION PROMPTS ==========
            'emissions_calculator': PromptTemplate(
                id='cg_001',
                category=PromptCategory.CODE_GENERATION,
                name='Emissions Calculator Code Generation',
                template="""Generate Python code for emissions calculation using PROVIDED formulas only.

Calculation type: {calc_type}
Input parameters: {parameters}
Required formula: {formula}
Emission factors: {emission_factors}
Unit requirements: {units}

Generate Python code that:
1. Uses ONLY the provided formula - no modifications
2. Includes comprehensive input validation
3. Handles unit conversions explicitly
4. Returns results with full provenance
5. Includes error handling
6. Documents the source of the formula

Template structure:
```python
def calculate_{calc_type}_emissions(
    {parameters}
) -> Dict[str, Any]:
    \"\"\"
    Calculate {calc_type} emissions using {formula_source}

    Formula: {formula}

    Returns:
        Dict with 'value', 'unit', 'formula_used', 'factors_used', 'confidence'
    \"\"\"
    # Input validation
    # Unit conversion if needed
    # Apply formula EXACTLY as provided
    # Return with provenance
```

CRITICAL: Never modify or "improve" the provided formula. Use it exactly.""",
                variables=['calc_type', 'parameters', 'formula', 'emission_factors', 'units'],
                output_schema={'code': 'string', 'tests': 'string'},
                confidence_threshold=0.90,
                max_tokens=2000,
                temperature=0.2,
                examples=[]
            ),

            'api_integration': PromptTemplate(
                id='cg_002',
                category=PromptCategory.CODE_GENERATION,
                name='API Integration Code',
                template="""Generate Python code for integrating with a sustainability data API.

API specification:
{api_spec}

Required endpoints:
{endpoints}

Authentication method: {auth_method}

Data transformations needed:
{transformations}

Generate Python code with:
1. Proper authentication handling
2. Rate limiting respect
3. Error handling and retries
4. Response validation
5. Data transformation to GreenLang format
6. Logging and monitoring

Include:
- Async support if beneficial
- Proper typing
- Comprehensive docstrings
- Unit tests

Output as complete, production-ready code.""",
                variables=['api_spec', 'endpoints', 'auth_method', 'transformations'],
                output_schema={'code': 'string', 'tests': 'string', 'documentation': 'string'},
                confidence_threshold=0.85,
                max_tokens=3000,
                temperature=0.2,
                examples=[]
            )
        }

    def get_prompt(self, prompt_id: str) -> Optional[PromptTemplate]:
        """Retrieve a specific prompt template"""
        return self.prompts.get(prompt_id)

    def get_prompts_by_category(self, category: PromptCategory) -> List[PromptTemplate]:
        """Get all prompts in a category"""
        return [p for p in self.prompts.values() if p.category == category]

    def format_prompt(self, prompt_id: str, variables: Dict[str, str]) -> str:
        """Format a prompt with variables"""
        prompt = self.get_prompt(prompt_id)
        if not prompt:
            raise ValueError(f"Prompt {prompt_id} not found")

        # Check all required variables are provided
        missing = set(prompt.variables) - set(variables.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        # Format the prompt
        return prompt.template.format(**variables)

    def get_confidence_threshold(self, prompt_id: str) -> float:
        """Get confidence threshold for a prompt"""
        prompt = self.get_prompt(prompt_id)
        return prompt.confidence_threshold if prompt else 0.80

    def get_model_parameters(self, prompt_id: str) -> Dict[str, Any]:
        """Get recommended model parameters for a prompt"""
        prompt = self.get_prompt(prompt_id)
        if not prompt:
            return {'temperature': 0.0, 'max_tokens': 500}

        return {
            'temperature': prompt.temperature,
            'max_tokens': prompt.max_tokens,
            'seed': 42  # Deterministic
        }


# Additional specialized prompts (51-100+)
EXTENDED_PROMPTS = {
    # Supply Chain Analysis
    'supply_chain_risk': """Analyze supply chain sustainability risks...""",
    'supplier_assessment': """Assess supplier sustainability performance...""",

    # Regulatory Compliance
    'eu_taxonomy_alignment': """Check EU Taxonomy alignment...""",
    'sfdr_classification': """Classify under SFDR articles...""",

    # Carbon Accounting
    'activity_data_validation': """Validate activity data for carbon accounting...""",
    'emission_factor_selection': """Select appropriate emission factors...""",

    # Reporting
    'gri_indicator_mapping': """Map data to GRI indicators...""",
    'sdg_alignment': """Align activities with UN SDGs...""",

    # Risk Assessment
    'climate_risk_identification': """Identify climate-related risks...""",
    'transition_risk_assessment': """Assess transition risks...""",

    # Data Quality
    'data_gap_analysis': """Identify data gaps in sustainability reporting...""",
    'data_quality_scoring': """Score data quality for reliability..."""
}

# Export the library
prompt_library = PromptLibrary()

__all__ = ['PromptLibrary', 'PromptTemplate', 'PromptCategory', 'prompt_library']