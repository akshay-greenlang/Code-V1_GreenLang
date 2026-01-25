# GreenLang RAG System - Integration Guide

## Overview

This guide explains how to integrate the GreenLang RAG system with the Agent Foundation for safe LLM-powered climate intelligence.

---

## Architecture Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    GreenLang Agent Foundation               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐         ┌──────────────────┐        │
│  │  Agent           │         │  RAG System      │        │
│  │  Intelligence    │◄────────┤                  │        │
│  │                  │         │  • Retrieval     │        │
│  │  • Task Planning │         │  • Confidence    │        │
│  │  • LLM Calls     │         │  • Validation    │        │
│  │  • Validation    │         │                  │        │
│  └──────────────────┘         └──────────────────┘        │
│         │                              │                   │
│         │                              │                   │
│  ┌──────▼──────────┐         ┌────────▼─────────┐        │
│  │  Prompt         │         │  Vector Store    │        │
│  │  Engineering    │         │  (FAISS/Chroma)  │        │
│  └─────────────────┘         └──────────────────┘        │
│         │                              │                   │
│         │                              │                   │
│  ┌──────▼──────────┐         ┌────────▼─────────┐        │
│  │  LLM Provider   │         │  Knowledge Graph │        │
│  │  (OpenAI/etc)   │         │  (Neo4j)         │        │
│  └─────────────────┘         └──────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Integration Pattern 1: Basic RAG Integration

### Step 1: Initialize RAG in Agent Intelligence

```python
# File: agent_foundation/core/agent_intelligence.py

from pathlib import Path
from typing import Optional, Dict, Any, List
from ..rag import RAGSystem, create_vector_store

class AgentIntelligence:
    """Enhanced Agent Intelligence with RAG integration"""

    def __init__(
        self,
        config: Dict[str, Any],
        data_dir: Optional[Path] = None
    ):
        self.config = config
        self.data_dir = data_dir or Path("./data/rag")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize RAG system
        self.rag = self._initialize_rag()

        # Track metrics
        self.metrics = {
            "rag_queries": 0,
            "high_confidence_responses": 0,
            "low_confidence_rejections": 0
        }

    def _initialize_rag(self) -> RAGSystem:
        """Initialize RAG system with production settings"""

        # Create vector store
        vector_store = create_vector_store(
            store_type="faiss",
            dimension=768,
            index_path=str(self.data_dir / "faiss_index.bin"),
            metadata_path=str(self.data_dir / "faiss_metadata.json")
        )

        # Create RAG system
        rag = RAGSystem(
            vector_store=vector_store,
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            chunking_strategy="semantic",
            chunk_size=1000,
            chunk_overlap=200,
            use_reranker=True,
            confidence_threshold=0.8,  # GreenLang requirement
            enable_caching=True  # 66% cost reduction
        )

        return rag

    def ingest_knowledge_base(self, documents: List[Dict]) -> Dict[str, int]:
        """
        Ingest documents into RAG knowledge base

        Args:
            documents: List of documents with content and metadata

        Returns:
            Statistics about ingestion
        """
        chunks_count = self.rag.ingest_documents(documents)

        return {
            "documents_processed": len(documents),
            "chunks_created": chunks_count
        }

    def query_knowledge_base(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True
    ) -> Dict[str, Any]:
        """
        Query RAG system with confidence validation

        Args:
            query: User query
            top_k: Number of results
            use_hybrid: Use hybrid search

        Returns:
            Results with confidence scoring
        """
        self.metrics["rag_queries"] += 1

        # Retrieve with RAG
        result = self.rag.retrieve(
            query=query,
            top_k=top_k,
            use_hybrid=use_hybrid
        )

        # Validate confidence
        if result.confidence >= 0.8:
            self.metrics["high_confidence_responses"] += 1
            return {
                "status": "success",
                "confidence": result.confidence,
                "documents": result.documents,
                "scores": result.scores,
                "can_proceed": True
            }
        else:
            self.metrics["low_confidence_rejections"] += 1
            return {
                "status": "low_confidence",
                "confidence": result.confidence,
                "documents": [],
                "scores": [],
                "can_proceed": False,
                "reason": f"Confidence {result.confidence:.2%} below 80% threshold"
            }
```

### Step 2: Safe LLM Integration

```python
# File: agent_foundation/core/agent_intelligence.py (continued)

class AgentIntelligence:
    # ... previous code ...

    def generate_safe_response(
        self,
        user_query: str,
        llm_client: Any,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Generate LLM response with RAG and safety guarantees

        This is the APPROVED pattern for LLM integration per GreenLang spec:
        - Use LLM ONLY for: classification, entity resolution, narrative
        - NEVER for: numeric calculations, compliance metrics
        - ALWAYS validate with 80%+ confidence threshold
        """

        # Step 1: Retrieve relevant context
        rag_result = self.query_knowledge_base(
            query=user_query,
            top_k=5,
            use_hybrid=True
        )

        # Step 2: Validate confidence (CRITICAL)
        if not rag_result["can_proceed"]:
            return {
                "answer": "I don't have sufficient confidence to answer this question accurately.",
                "confidence": rag_result["confidence"],
                "sources": [],
                "status": "rejected",
                "reason": rag_result["reason"]
            }

        # Step 3: Augment prompt with safety instructions
        augmented_prompt = self.rag.augment_prompt(
            query=user_query,
            context_documents=rag_result["documents"],
            max_context_length=8000
        )

        # Step 4: Call LLM with augmented prompt
        try:
            llm_response = llm_client.generate(
                prompt=augmented_prompt,
                max_tokens=max_tokens,
                temperature=0.1  # Low temperature for factual responses
            )

            # Step 5: Validate response (no calculations)
            if self._contains_calculations(llm_response):
                return {
                    "answer": "Error: LLM attempted to perform calculations. This is prohibited.",
                    "confidence": 0.0,
                    "sources": [],
                    "status": "validation_failed",
                    "reason": "LLM performed prohibited calculations"
                }

            return {
                "answer": llm_response,
                "confidence": rag_result["confidence"],
                "sources": [
                    {
                        "content": doc.content[:200],
                        "score": score,
                        "source": doc.metadata.get("source", "Unknown")
                    }
                    for doc, score in zip(
                        rag_result["documents"][:3],
                        rag_result["scores"][:3]
                    )
                ],
                "status": "success",
                "provenance_hash": self._calculate_provenance(
                    user_query,
                    rag_result["documents"],
                    llm_response
                )
            }

        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "status": "error"
            }

    def _contains_calculations(self, text: str) -> bool:
        """
        Detect if LLM performed numeric calculations (PROHIBITED)

        Returns True if calculations detected, False otherwise
        """
        import re

        # Pattern: number operator number = number
        calc_pattern = r'\d+\.?\d*\s*[+\-*/×÷]\s*\d+\.?\d*\s*=\s*\d+\.?\d*'

        # Pattern: "calculated as", "result is", etc.
        calc_phrases = [
            r'calculated as',
            r'computing',
            r'the result is',
            r'equals',
            r'sum of.*is',
            r'product of.*is'
        ]

        if re.search(calc_pattern, text):
            return True

        for phrase in calc_phrases:
            if re.search(phrase, text, re.IGNORECASE):
                return True

        return False

    def _calculate_provenance(
        self,
        query: str,
        documents: List[Any],
        response: str
    ) -> str:
        """Calculate SHA-256 provenance hash for audit trail"""
        import hashlib

        provenance_str = f"{query}:{''.join([doc.content for doc in documents])}:{response}"
        return hashlib.sha256(provenance_str.encode()).hexdigest()
```

---

## Integration Pattern 2: Multi-Tier Approach

```python
# File: agent_foundation/core/multi_tier_intelligence.py

from enum import Enum
from typing import Dict, Any, Optional

class DataTier(Enum):
    """Data quality tiers per GreenLang spec"""
    TIER_1_ACTUAL = "actual_data"           # Highest quality
    TIER_2_AI_CLASSIFIED = "ai_classified"  # AI classification
    TIER_3_LLM_ESTIMATED = "llm_estimated"  # Lowest quality

class MultiTierIntelligence:
    """
    Multi-tier approach for GreenLang:
    - Tier 1: Actual data from database (no AI)
    - Tier 2: AI classification (approved use case)
    - Tier 3: LLM estimation (with transparency)
    """

    def __init__(self, agent_intelligence: AgentIntelligence):
        self.agent = agent_intelligence
        self.rag = agent_intelligence.rag

    def process_query(
        self,
        query: str,
        database_client: Any,
        llm_client: Any
    ) -> Dict[str, Any]:
        """
        Process query with multi-tier approach

        Priority:
        1. Try database (Tier 1)
        2. Try AI classification (Tier 2)
        3. Try LLM with RAG (Tier 3)
        """

        # Tier 1: Check actual data
        tier1_result = self._try_tier1(query, database_client)
        if tier1_result["found"]:
            return {
                "answer": tier1_result["data"],
                "tier": DataTier.TIER_1_ACTUAL,
                "confidence": 1.0,
                "source": "database",
                "status": "success"
            }

        # Tier 2: Try AI classification
        tier2_result = self._try_tier2(query)
        if tier2_result["found"]:
            return {
                "answer": tier2_result["classification"],
                "tier": DataTier.TIER_2_AI_CLASSIFIED,
                "confidence": tier2_result["confidence"],
                "source": "ai_classifier",
                "status": "success"
            }

        # Tier 3: Use LLM with RAG (with transparency)
        tier3_result = self.agent.generate_safe_response(
            user_query=query,
            llm_client=llm_client
        )

        if tier3_result["status"] == "success":
            return {
                "answer": tier3_result["answer"],
                "tier": DataTier.TIER_3_LLM_ESTIMATED,
                "confidence": tier3_result["confidence"],
                "source": "llm_with_rag",
                "status": "success",
                "warning": "This is an LLM-generated estimate. Verify before using for compliance."
            }

        # All tiers failed
        return {
            "answer": "Unable to answer query with sufficient confidence.",
            "tier": None,
            "confidence": 0.0,
            "status": "failed"
        }

    def _try_tier1(self, query: str, db_client: Any) -> Dict[str, Any]:
        """Tier 1: Query actual database"""
        try:
            # Example: query carbon emissions from database
            result = db_client.query(query)
            if result:
                return {"found": True, "data": result}
        except:
            pass

        return {"found": False}

    def _try_tier2(self, query: str) -> Dict[str, Any]:
        """Tier 2: AI classification (approved use case)"""

        # Example: Classify transaction into Scope 3 category
        if "classify" in query.lower() or "categorize" in query.lower():
            # Use RAG to find similar examples
            rag_result = self.rag.retrieve(query, top_k=5)

            if rag_result.confidence >= 0.8:
                # Extract classification from context
                classification = self._extract_classification(
                    query,
                    rag_result.documents
                )

                return {
                    "found": True,
                    "classification": classification,
                    "confidence": rag_result.confidence
                }

        return {"found": False}

    def _extract_classification(self, query: str, documents: List[Any]) -> str:
        """Extract classification from retrieved documents"""
        # Simple example - in production, use a classifier model
        for doc in documents:
            if "Scope 3 Category" in doc.content:
                # Extract category from content
                import re
                match = re.search(r'Scope 3 Category (\d+)', doc.content)
                if match:
                    return f"Scope 3 Category {match.group(1)}"

        return "Unable to classify"
```

---

## Integration Pattern 3: Entity Resolution

```python
# File: agent_foundation/services/entity_resolution.py

from typing import List, Dict, Any, Optional
from ..rag import RAGSystem

class EntityResolver:
    """
    Entity resolution using RAG
    APPROVED use case: Match supplier names to master data
    """

    def __init__(self, rag_system: RAGSystem):
        self.rag = rag_system

    def resolve_supplier(
        self,
        supplier_name: str,
        confidence_threshold: float = 0.8
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve supplier name to master data entity

        Args:
            supplier_name: Input supplier name (possibly misspelled)
            confidence_threshold: Minimum confidence (default 0.8)

        Returns:
            Resolved entity or None if confidence too low
        """

        # Query RAG with supplier name
        result = self.rag.retrieve(
            query=f"Find supplier: {supplier_name}",
            top_k=3,
            filters={"type": "supplier"}
        )

        # Validate confidence
        if result.confidence < confidence_threshold:
            return None

        # Extract best match
        best_match = result.documents[0]

        return {
            "supplier_id": best_match.metadata.get("supplier_id"),
            "canonical_name": best_match.metadata.get("canonical_name"),
            "confidence": result.scores[0],
            "provenance_hash": result.metadata.get("provenance_hash"),
            "alternatives": [
                {
                    "name": doc.metadata.get("canonical_name"),
                    "score": score
                }
                for doc, score in zip(result.documents[1:], result.scores[1:])
            ]
        }

    def bulk_resolve(
        self,
        supplier_names: List[str],
        confidence_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Bulk entity resolution"""

        results = []
        for name in supplier_names:
            resolved = self.resolve_supplier(name, confidence_threshold)
            results.append({
                "input": name,
                "resolved": resolved,
                "status": "success" if resolved else "low_confidence"
            })

        return results
```

---

## Integration Pattern 4: Materiality Assessment

```python
# File: agent_foundation/assessments/materiality.py

from typing import List, Dict, Any
from ..rag import RAGSystem, KnowledgeGraphStore

class MaterialityAssessor:
    """
    Double materiality assessment using RAG + Knowledge Graph
    APPROVED use case: Determine material topics for CSRD
    """

    def __init__(
        self,
        rag_system: RAGSystem,
        knowledge_graph: Optional[KnowledgeGraphStore] = None
    ):
        self.rag = rag_system
        self.kg = knowledge_graph

    def assess_topic_materiality(
        self,
        topic: str,
        company_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess if topic is material for company

        Args:
            topic: Sustainability topic (e.g., "carbon emissions", "water use")
            company_context: Company information (sector, size, geography)

        Returns:
            Materiality assessment with confidence
        """

        # Build query with context
        query = f"""
        Assess materiality of '{topic}' for:
        - Sector: {company_context.get('sector')}
        - Size: {company_context.get('employee_count')} employees
        - Geography: {company_context.get('geography')}
        - Activities: {', '.join(company_context.get('activities', []))}
        """

        # Retrieve relevant guidance
        result = self.rag.retrieve(
            query=query,
            top_k=10,
            use_hybrid=True,
            filters={"category": "materiality"}
        )

        # Validate confidence
        if result.confidence < 0.8:
            return {
                "topic": topic,
                "is_material": None,
                "confidence": result.confidence,
                "status": "inconclusive",
                "reason": "Insufficient confidence for determination"
            }

        # Extract materiality indicators
        impact_materiality = self._assess_impact_materiality(result.documents)
        financial_materiality = self._assess_financial_materiality(result.documents)

        # Double materiality: material if EITHER impact OR financial
        is_material = impact_materiality["is_material"] or financial_materiality["is_material"]

        return {
            "topic": topic,
            "is_material": is_material,
            "confidence": result.confidence,
            "impact_materiality": impact_materiality,
            "financial_materiality": financial_materiality,
            "supporting_evidence": [
                {
                    "source": doc.metadata.get("source"),
                    "content": doc.content[:200],
                    "score": score
                }
                for doc, score in zip(result.documents[:3], result.scores[:3])
            ],
            "status": "success"
        }

    def _assess_impact_materiality(self, documents: List[Any]) -> Dict[str, Any]:
        """Assess impact on environment/society"""
        # Simple scoring based on keywords in context
        impact_keywords = [
            "significant impact", "material impact", "environmental harm",
            "social consequences", "stakeholder affected"
        ]

        score = 0
        for doc in documents:
            content_lower = doc.content.lower()
            score += sum(1 for kw in impact_keywords if kw in content_lower)

        is_material = score >= 3  # Threshold

        return {
            "is_material": is_material,
            "score": score,
            "threshold": 3
        }

    def _assess_financial_materiality(self, documents: List[Any]) -> Dict[str, Any]:
        """Assess financial impact on company"""
        financial_keywords = [
            "financial risk", "revenue impact", "cost", "regulatory fine",
            "market opportunity", "competitive advantage"
        ]

        score = 0
        for doc in documents:
            content_lower = doc.content.lower()
            score += sum(1 for kw in financial_keywords if kw in content_lower)

        is_material = score >= 3

        return {
            "is_material": is_material,
            "score": score,
            "threshold": 3
        }
```

---

## Testing Integration

```python
# File: tests/integration/test_rag_integration.py

import pytest
from agent_foundation.core import AgentIntelligence
from agent_foundation.rag import RAGSystem, create_vector_store

class TestRAGIntegration:
    """Integration tests for RAG system"""

    @pytest.fixture
    def agent(self):
        """Create agent with RAG"""
        config = {
            "rag_enabled": True,
            "confidence_threshold": 0.8
        }
        return AgentIntelligence(config)

    def test_knowledge_ingestion(self, agent):
        """Test document ingestion"""
        documents = [
            {
                "content": "CSRD requires Scope 1, 2, and 3 emissions reporting.",
                "metadata": {"source": "EU Regulation", "year": 2023}
            }
        ]

        stats = agent.ingest_knowledge_base(documents)

        assert stats["documents_processed"] == 1
        assert stats["chunks_created"] > 0

    def test_high_confidence_query(self, agent):
        """Test query with high confidence"""
        # Ingest knowledge first
        agent.ingest_knowledge_base([
            {"content": "Scope 3 Category 1 covers purchased goods and services."}
        ])

        result = agent.query_knowledge_base("What is Scope 3 Category 1?")

        assert result["status"] == "success"
        assert result["confidence"] >= 0.8
        assert result["can_proceed"] is True

    def test_low_confidence_rejection(self, agent):
        """Test query rejection due to low confidence"""
        result = agent.query_knowledge_base("What is quantum entanglement?")

        assert result["status"] == "low_confidence"
        assert result["can_proceed"] is False

    def test_calculation_detection(self, agent):
        """Test that calculations are detected and rejected"""
        text_with_calc = "The result is 100 * 5 = 500 tonnes CO2"

        assert agent._contains_calculations(text_with_calc) is True

    def test_safe_response_generation(self, agent, mock_llm_client):
        """Test end-to-end safe response generation"""
        # Ingest knowledge
        agent.ingest_knowledge_base([
            {"content": "Carbon emissions are measured in tonnes CO2 equivalent."}
        ])

        # Generate response
        response = agent.generate_safe_response(
            user_query="How are carbon emissions measured?",
            llm_client=mock_llm_client
        )

        assert response["status"] == "success"
        assert response["confidence"] >= 0.8
        assert "provenance_hash" in response
        assert len(response["sources"]) > 0
```

---

## Deployment Checklist

- [ ] Install all dependencies (`pip install -r requirements.txt`)
- [ ] Download spaCy model (`python -m spacy download en_core_web_sm`)
- [ ] Initialize vector store (FAISS/ChromaDB/Pinecone)
- [ ] Ingest initial knowledge base
- [ ] Set up Neo4j (optional, for knowledge graph)
- [ ] Configure confidence thresholds (≥0.8)
- [ ] Enable caching for 66% cost reduction
- [ ] Set up monitoring (Prometheus metrics)
- [ ] Test end-to-end integration
- [ ] Verify zero-hallucination guarantees

---

## Monitoring in Production

```python
# Collect metrics
metrics = agent.rag.get_metrics()

# Log to monitoring system
logger.info(f"RAG Metrics: {metrics}")

# Alert if confidence drops
if metrics["avg_confidence"] < 0.8:
    alert("RAG confidence below threshold!")

# Alert if cache hit rate drops
if metrics["cache_hit_rate"] < 0.5:
    alert("RAG cache efficiency dropped!")
```

---

**Last Updated:** 2025-11-15
**Version:** 1.0
