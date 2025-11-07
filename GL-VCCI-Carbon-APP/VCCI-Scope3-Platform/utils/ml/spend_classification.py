# GL-VCCI ML Module - Spend Classification
# Spend Classification ML System - Main Classification Logic

"""
Spend Classification
====================

Main spend classification system with LLM-based classification and rule-based fallback.

Classification Strategy:
-----------------------
1. Primary: LLM-based classification (GPT-3.5/Claude)
   - High accuracy (target: 90%+)
   - Confidence-based acceptance (threshold: 0.85)

2. Fallback: Rule-based classification
   - Triggered when LLM confidence < 0.85
   - Keyword/regex/fuzzy matching

3. Confidence-based routing:
   - Confidence ≥ 0.85 → Accept classification
   - 0.5 ≤ Confidence < 0.85 → Use rules fallback or flag for review
   - Confidence < 0.5 → Flag for human review

Features:
--------
- Hybrid LLM + rules approach
- Confidence-based acceptance
- Human review flagging
- Batch processing
- SOC 2 compliant audit logging
- Cost tracking
- Performance metrics

Usage:
------
```python
from utils.ml.spend_classification import SpendClassifier
from utils.ml.config import MLConfig

# Initialize classifier
config = MLConfig()
classifier = SpendClassifier(config)

# Classify single spend
result = await classifier.classify(
    description="Office furniture purchase",
    amount=5000.0,
    supplier="IKEA"
)

print(f"Category: {result.category}")
print(f"Confidence: {result.confidence}")
print(f"Needs review: {result.needs_human_review}")

# Batch classification
results = await classifier.classify_batch([
    {"description": "Flight to NYC", "amount": 450.0},
    {"description": "Electricity bill", "amount": 1200.0}
])

# Close client
await classifier.close()
```
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import MLConfig, Scope3Category
from .exceptions import (
    ClassificationException,
    InvalidCategoryException,
    LowConfidenceException,
)
from .llm_client import ClassificationResult as LLMResult
from .llm_client import LLMClient
from .rules_engine import RuleMatch, RulesEngine

logger = logging.getLogger(__name__)


# ============================================================================
# Models
# ============================================================================

class SpendItem(BaseModel):
    """
    Spend item for classification.

    Attributes:
        description: Procurement description
        amount: Spend amount (optional, can improve classification)
        supplier: Supplier name (optional)
        gl_code: GL code (optional)
        additional_context: Additional context (optional)
    """
    description: str = Field(description="Procurement description")
    amount: Optional[float] = Field(default=None, description="Spend amount")
    supplier: Optional[str] = Field(default=None, description="Supplier name")
    gl_code: Optional[str] = Field(default=None, description="GL code")
    additional_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context"
    )

    def get_enriched_description(self) -> str:
        """
        Get enriched description with context.

        Returns:
            Description with supplier and amount if available
        """
        parts = [self.description]

        if self.supplier:
            parts.append(f"(Supplier: {self.supplier})")

        if self.amount:
            parts.append(f"(Amount: ${self.amount:,.2f})")

        return " ".join(parts)


class ClassificationResult(BaseModel):
    """
    Classification result with metadata.

    Attributes:
        category: Predicted Scope 3 category
        category_name: Human-readable category name
        confidence: Classification confidence (0.0-1.0)
        method: Classification method (llm, rules, hybrid)
        needs_human_review: Whether human review is needed
        reasoning: Classification reasoning
        alternative_categories: Alternative categories with confidence
        llm_result: Original LLM result (if used)
        rules_result: Original rules result (if used)
        metadata: Additional metadata (timing, cost, etc.)
    """
    category: str = Field(description="Predicted Scope 3 category")
    category_name: str = Field(description="Human-readable category name")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence")
    method: str = Field(description="Classification method (llm, rules, hybrid)")
    needs_human_review: bool = Field(description="Needs human review")
    reasoning: Optional[str] = Field(default=None, description="Classification reasoning")
    alternative_categories: List[tuple] = Field(
        default_factory=list,
        description="Alternative categories [(category, confidence), ...]"
    )
    llm_result: Optional[LLMResult] = Field(default=None, description="LLM result")
    rules_result: Optional[RuleMatch] = Field(default=None, description="Rules result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ============================================================================
# Spend Classifier
# ============================================================================

class SpendClassifier:
    """
    Main spend classification system.

    Combines LLM-based classification with rule-based fallback for
    robust Scope 3 category classification.
    """

    def __init__(self, config: MLConfig):
        """
        Initialize spend classifier.

        Args:
            config: ML configuration
        """
        self.config = config

        # Initialize LLM client
        if config.classification.use_llm_primary:
            self.llm_client = LLMClient(config)
        else:
            self.llm_client = None

        # Initialize rules engine
        if config.classification.use_rules_fallback:
            self.rules_engine = RulesEngine(config)
        else:
            self.rules_engine = None

        # Metrics
        self.total_classifications = 0
        self.llm_classifications = 0
        self.rules_classifications = 0
        self.hybrid_classifications = 0
        self.human_review_flagged = 0

        logger.info(
            f"Initialized spend classifier: "
            f"llm_enabled={config.classification.use_llm_primary}, "
            f"rules_enabled={config.classification.use_rules_fallback}, "
            f"confidence_threshold={config.classification.confidence_threshold}"
        )

    async def close(self):
        """Close client connections."""
        if self.llm_client:
            await self.llm_client.close()

    async def classify(
        self,
        description: str,
        amount: Optional[float] = None,
        supplier: Optional[str] = None,
        gl_code: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> ClassificationResult:
        """
        Classify procurement spend.

        Args:
            description: Procurement description
            amount: Spend amount (optional)
            supplier: Supplier name (optional)
            gl_code: GL code (optional)
            additional_context: Additional context (optional)

        Returns:
            Classification result

        Raises:
            ClassificationException: If classification fails
        """
        start_time = time.time()

        # Create spend item
        spend_item = SpendItem(
            description=description,
            amount=amount,
            supplier=supplier,
            gl_code=gl_code,
            additional_context=additional_context
        )

        try:
            # Classify using appropriate strategy
            result = await self._classify_internal(spend_item)

            # Add metadata
            result.metadata.update({
                "classification_time_ms": (time.time() - start_time) * 1000,
                "timestamp": datetime.utcnow().isoformat(),
                "description_length": len(description),
            })

            # Update metrics
            self.total_classifications += 1
            if result.needs_human_review:
                self.human_review_flagged += 1

            # Log classification
            logger.info(
                f"Classified spend: category={result.category}, "
                f"confidence={result.confidence:.2f}, method={result.method}, "
                f"needs_review={result.needs_human_review}"
            )

            return result

        except Exception as e:
            logger.error(f"Classification failed: {e}", exc_info=True)
            raise ClassificationException(
                message=f"Failed to classify spend: {str(e)}",
                details={"description": description[:100]},
                original_error=e
            )

    async def classify_batch(
        self,
        items: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> List[ClassificationResult]:
        """
        Classify batch of spend items.

        Args:
            items: List of spend item dictionaries
            batch_size: Batch size for processing

        Returns:
            List of classification results
        """
        results = []

        # Process in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]

            # Classify concurrently
            tasks = [
                self.classify(**item) if isinstance(item, dict) else self.classify(item)
                for item in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle errors
            for item, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    desc = item.get("description", str(item)) if isinstance(item, dict) else str(item)
                    logger.error(f"Batch classification failed for '{desc[:50]}...': {result}")

                    # Create error result
                    results.append(ClassificationResult(
                        category="unknown",
                        category_name="Unknown",
                        confidence=0.0,
                        method="error",
                        needs_human_review=True,
                        reasoning=f"Error: {str(result)}",
                        metadata={"error": str(result)}
                    ))
                else:
                    results.append(result)

        logger.info(f"Batch classified {len(items)} items: {len(results)} results")
        return results

    async def _classify_internal(self, spend_item: SpendItem) -> ClassificationResult:
        """
        Internal classification logic with fallback strategy.

        Strategy:
        1. Try LLM classification first (if enabled)
        2. If LLM confidence < threshold, try rules fallback
        3. Determine if human review needed
        4. Return best result

        Args:
            spend_item: Spend item to classify

        Returns:
            Classification result
        """
        enriched_desc = spend_item.get_enriched_description()

        llm_result: Optional[LLMResult] = None
        rules_result: Optional[RuleMatch] = None

        # 1. Try LLM classification
        if self.llm_client and self.config.classification.use_llm_primary:
            try:
                llm_result = await self.llm_client.classify_spend(enriched_desc)
                logger.debug(
                    f"LLM classification: category={llm_result.category}, "
                    f"confidence={llm_result.confidence:.2f}"
                )
            except Exception as e:
                logger.warning(f"LLM classification failed, will try rules: {e}")

        # 2. Try rules fallback (if LLM failed or low confidence)
        should_use_rules = (
            self.rules_engine and
            self.config.classification.use_rules_fallback and
            (llm_result is None or
             llm_result.confidence < self.config.classification.confidence_threshold)
        )

        if should_use_rules:
            try:
                rules_result = self.rules_engine.classify(enriched_desc)
                logger.debug(
                    f"Rules classification: category={rules_result.category}, "
                    f"confidence={rules_result.confidence:.2f}"
                )
            except Exception as e:
                logger.warning(f"Rules classification failed: {e}")

        # 3. Determine best result and method
        if llm_result and rules_result:
            # Both available - choose based on confidence
            if llm_result.confidence >= self.config.classification.confidence_threshold:
                # LLM high confidence - use LLM
                final_category = llm_result.category
                final_confidence = llm_result.confidence
                final_reasoning = llm_result.reasoning
                final_alternatives = llm_result.alternative_categories
                method = "llm"
                self.llm_classifications += 1
            elif rules_result.confidence > llm_result.confidence:
                # Rules better than LLM - use rules
                final_category = rules_result.category
                final_confidence = rules_result.confidence
                final_reasoning = f"Rule match: {', '.join(rules_result.matched_keywords or rules_result.matched_patterns)}"
                final_alternatives = []
                method = "rules"
                self.rules_classifications += 1
            else:
                # LLM better but below threshold - use hybrid
                final_category = llm_result.category
                final_confidence = (llm_result.confidence + rules_result.confidence) / 2
                final_reasoning = f"Hybrid: LLM={llm_result.confidence:.2f}, Rules={rules_result.confidence:.2f}"
                final_alternatives = llm_result.alternative_categories
                method = "hybrid"
                self.hybrid_classifications += 1

        elif llm_result:
            # Only LLM available
            final_category = llm_result.category
            final_confidence = llm_result.confidence
            final_reasoning = llm_result.reasoning
            final_alternatives = llm_result.alternative_categories
            method = "llm"
            self.llm_classifications += 1

        elif rules_result:
            # Only rules available
            final_category = rules_result.category
            final_confidence = rules_result.confidence
            final_reasoning = f"Rule match: {', '.join(rules_result.matched_keywords or rules_result.matched_patterns)}"
            final_alternatives = []
            method = "rules"
            self.rules_classifications += 1

        else:
            # No classification available
            final_category = "unknown"
            final_confidence = 0.0
            final_reasoning = "No classification method available"
            final_alternatives = []
            method = "none"

        # 4. Determine if human review needed
        needs_review = (
            final_confidence < self.config.classification.require_human_review_threshold or
            final_category == "unknown"
        )

        # 5. Validate category
        if final_category != "unknown":
            valid_categories = self.config.get_valid_categories()
            if final_category not in valid_categories:
                logger.warning(f"Invalid category: {final_category}, setting to unknown")
                final_category = "unknown"
                needs_review = True

        # 6. Build result
        return ClassificationResult(
            category=final_category,
            category_name=self.config.get_category_name(final_category),
            confidence=final_confidence,
            method=method,
            needs_human_review=needs_review,
            reasoning=final_reasoning,
            alternative_categories=final_alternatives,
            llm_result=llm_result,
            rules_result=rules_result,
            metadata={
                "llm_used": llm_result is not None,
                "rules_used": rules_result is not None,
                "llm_cached": llm_result.cached if llm_result else False,
                "llm_cost_usd": llm_result.cost_usd if llm_result else 0.0,
            }
        )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get classification metrics.

        Returns:
            Metrics dictionary
        """
        return {
            "total_classifications": self.total_classifications,
            "llm_classifications": self.llm_classifications,
            "rules_classifications": self.rules_classifications,
            "hybrid_classifications": self.hybrid_classifications,
            "human_review_flagged": self.human_review_flagged,
            "llm_percentage": (
                (self.llm_classifications / self.total_classifications * 100)
                if self.total_classifications > 0 else 0
            ),
            "rules_percentage": (
                (self.rules_classifications / self.total_classifications * 100)
                if self.total_classifications > 0 else 0
            ),
            "review_rate": (
                (self.human_review_flagged / self.total_classifications * 100)
                if self.total_classifications > 0 else 0
            ),
            "llm_cost_summary": (
                self.llm_client.get_cost_summary()
                if self.llm_client else {}
            )
        }

    def reset_metrics(self):
        """Reset classification metrics."""
        self.total_classifications = 0
        self.llm_classifications = 0
        self.rules_classifications = 0
        self.hybrid_classifications = 0
        self.human_review_flagged = 0

        if self.llm_client:
            self.llm_client.total_tokens_used = 0
            self.llm_client.total_cost_usd = 0.0
            self.llm_client.request_count = 0


# ============================================================================
# Convenience Functions
# ============================================================================

async def classify_spend(
    description: str,
    config: Optional[MLConfig] = None,
    **kwargs
) -> ClassificationResult:
    """
    Classify single spend (convenience function).

    Args:
        description: Procurement description
        config: ML configuration (uses default if not provided)
        **kwargs: Additional spend item fields

    Returns:
        Classification result

    Example:
        >>> result = await classify_spend("Office furniture purchase")
        >>> print(result.category)
        'category_1_purchased_goods_services'
    """
    if config is None:
        from .config import load_config
        config = load_config()

    classifier = SpendClassifier(config)
    try:
        result = await classifier.classify(description, **kwargs)
        return result
    finally:
        await classifier.close()


async def classify_spend_batch(
    descriptions: List[str],
    config: Optional[MLConfig] = None
) -> List[ClassificationResult]:
    """
    Classify batch of spends (convenience function).

    Args:
        descriptions: List of procurement descriptions
        config: ML configuration (uses default if not provided)

    Returns:
        List of classification results

    Example:
        >>> results = await classify_spend_batch([
        ...     "Flight to NYC",
        ...     "Electricity bill"
        ... ])
        >>> for result in results:
        ...     print(f"{result.category}: {result.confidence}")
    """
    if config is None:
        from .config import load_config
        config = load_config()

    items = [{"description": desc} for desc in descriptions]

    classifier = SpendClassifier(config)
    try:
        results = await classifier.classify_batch(items)
        return results
    finally:
        await classifier.close()
