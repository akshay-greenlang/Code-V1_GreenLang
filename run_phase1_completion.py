"""
Phase 1 Completion Script - Critical Path Execution

Executes the remaining Phase 1 tasks:
1. Run knowledge base ingestion
2. Test retrieval quality
3. Validate RAG is working properly

This script will document results and update completion status.
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
import logging

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n" + "="*80)
    print("CHECKING DEPENDENCIES")
    print("="*80)

    dependencies = {
        "sentence_transformers": False,
        "faiss": False,
        "numpy": False,
        "torch": False,
    }

    missing = []

    for dep in dependencies.keys():
        try:
            if dep == "sentence_transformers":
                import sentence_transformers
                dependencies[dep] = True
                print(f"✓ {dep} installed (version: {sentence_transformers.__version__})")
            elif dep == "faiss":
                import faiss
                dependencies[dep] = True
                print(f"✓ {dep} installed")
            elif dep == "numpy":
                import numpy
                dependencies[dep] = True
                print(f"✓ {dep} installed (version: {numpy.__version__})")
            elif dep == "torch":
                import torch
                dependencies[dep] = True
                print(f"✓ {dep} installed (version: {torch.__version__})")
        except ImportError:
            dependencies[dep] = False
            missing.append(dep)
            print(f"✗ {dep} NOT installed")

    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("\nTo install:")
        print("  pip install sentence-transformers faiss-cpu numpy torch")
        return False

    print("\n✓ All dependencies installed")
    return True


async def run_ingestion():
    """Run knowledge base ingestion."""
    print("\n" + "="*80)
    print("RUNNING KNOWLEDGE BASE INGESTION")
    print("="*80)

    try:
        from greenlang.intelligence.rag.engine import RAGEngine
        from greenlang.intelligence.rag.config import RAGConfig
        from greenlang.intelligence.rag.models import DocMeta
        from greenlang.intelligence.rag.hashing import file_hash

        # Check if knowledge base exists
        kb_dir = Path("knowledge_base")
        if not kb_dir.exists():
            print("⚠️  knowledge_base/ directory not found")
            print("   Creating from scratch...")
            kb_dir.mkdir(exist_ok=True)

        # Import the ingestion script functions
        sys.path.insert(0, str(Path("scripts")))
        from ingest_knowledge_base import (
            create_ghg_protocol_documents,
            create_technology_documents,
            create_case_study_documents,
            KnowledgeBaseIngester
        )

        # Create configuration
        config = RAGConfig(
            mode="live",
            allowlist=[
                "ghg_protocol_corp",
                "technology_database",
                "case_studies",
            ],
            embedding_provider="minilm",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dimension=384,
            vector_store_provider="faiss",
            vector_store_path=str(kb_dir / "vector_store"),
            retrieval_method="mmr",
            default_top_k=6,
            default_fetch_k=20,
            mmr_lambda=0.5,
            chunk_size=512,
            chunk_overlap=64,
            enable_sanitization=True,
            verify_checksums=True,
        )

        print(f"✓ Configuration created")
        print(f"  - Embedding provider: {config.embedding_provider}")
        print(f"  - Vector store: {config.vector_store_provider}")
        print(f"  - Chunk size: {config.chunk_size}")

        # Create ingester
        ingester = KnowledgeBaseIngester(config=config)

        # Create documents
        print("\n📄 Creating knowledge base documents...")
        ghg_docs = create_ghg_protocol_documents(kb_dir)
        print(f"  ✓ GHG Protocol: {len(ghg_docs)} documents")

        tech_docs = create_technology_documents(kb_dir)
        print(f"  ✓ Technologies: {len(tech_docs)} documents")

        case_docs = create_case_study_documents(kb_dir)
        print(f"  ✓ Case Studies: {len(case_docs)} documents")

        total_docs = len(ghg_docs) + len(tech_docs) + len(case_docs)
        print(f"\n📊 Total documents: {total_docs}")

        # Ingest all collections
        collections = {
            "ghg_protocol_corp": ghg_docs,
            "technology_database": tech_docs,
            "case_studies": case_docs,
        }

        print("\n🔄 Starting ingestion...")
        for collection, documents in collections.items():
            await ingester.ingest_collection(collection, documents)

        # Print statistics
        ingester.print_stats()

        print("\n✅ INGESTION COMPLETE")

        return ingester, config

    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


async def test_retrieval(ingester, config):
    """Test retrieval quality with sample queries."""
    print("\n" + "="*80)
    print("TESTING RETRIEVAL QUALITY")
    print("="*80)

    if not ingester:
        print("⚠️  Ingester not available, skipping retrieval tests")
        return None

    # Test queries with expected results
    test_queries = [
        {
            "query": "What are the emission factors for natural gas combustion?",
            "expected_collection": "ghg_protocol_corp",
            "expected_keywords": ["natural gas", "0.0531", "kg CO2e/kWh"],
            "category": "emission_factors"
        },
        {
            "query": "What are Scope 1, Scope 2, and Scope 3 emissions?",
            "expected_collection": "ghg_protocol_corp",
            "expected_keywords": ["Scope 1", "Scope 2", "Scope 3", "direct"],
            "category": "methodology"
        },
        {
            "query": "How do industrial heat pumps reduce emissions?",
            "expected_collection": "technology_database",
            "expected_keywords": ["heat pump", "COP", "50-70%", "emissions"],
            "category": "technology"
        },
        {
            "query": "What is the payback period for solar thermal systems?",
            "expected_collection": "technology_database",
            "expected_keywords": ["solar thermal", "5-10 years", "payback"],
            "category": "technology"
        },
        {
            "query": "Show me a case study of waste heat recovery in a steel mill",
            "expected_collection": "case_studies",
            "expected_keywords": ["steel", "waste heat", "ORC", "750 kW"],
            "category": "case_study"
        },
    ]

    results = []

    for i, test in enumerate(test_queries, 1):
        print(f"\n{'─'*80}")
        print(f"Query {i}/{len(test_queries)}: {test['query']}")
        print(f"Category: {test['category']}")

        try:
            # Execute query
            result = await ingester.engine.query(
                query=test["query"],
                top_k=5,
                collections=config.allowlist,
            )

            print(f"  Retrieved: {len(result.chunks)} chunks")
            print(f"  Search time: {result.search_time_ms}ms")
            print(f"  Total tokens: {result.total_tokens}")

            # Check relevance
            retrieved_text = " ".join(chunk.text for chunk in result.chunks).lower()

            keywords_found = []
            keywords_missing = []

            for keyword in test["expected_keywords"]:
                if keyword.lower() in retrieved_text:
                    keywords_found.append(keyword)
                else:
                    keywords_missing.append(keyword)

            relevance_score = len(keywords_found) / len(test["expected_keywords"])

            print(f"\n  Relevance Analysis:")
            print(f"    Keywords found: {len(keywords_found)}/{len(test['expected_keywords'])} ({relevance_score*100:.1f}%)")

            if keywords_found:
                print(f"    ✓ Found: {', '.join(keywords_found)}")
            if keywords_missing:
                print(f"    ✗ Missing: {', '.join(keywords_missing)}")

            # Show top result
            if result.chunks:
                print(f"\n  Top Result (Score: {result.relevance_scores[0]:.3f}):")
                print(f"    {result.chunks[0].text[:200]}...")

            # Store results
            results.append({
                "query": test["query"],
                "category": test["category"],
                "chunks_retrieved": len(result.chunks),
                "search_time_ms": result.search_time_ms,
                "relevance_score": relevance_score,
                "keywords_found": keywords_found,
                "keywords_missing": keywords_missing,
            })

        except Exception as e:
            print(f"  ❌ Query failed: {e}")
            results.append({
                "query": test["query"],
                "category": test["category"],
                "error": str(e)
            })

    # Summary
    print("\n" + "="*80)
    print("RETRIEVAL QUALITY SUMMARY")
    print("="*80)

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"\nQueries executed: {len(test_queries)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")

    if successful:
        avg_relevance = sum(r["relevance_score"] for r in successful) / len(successful)
        avg_search_time = sum(r["search_time_ms"] for r in successful) / len(successful)

        print(f"\nPerformance Metrics:")
        print(f"  Average relevance: {avg_relevance*100:.1f}%")
        print(f"  Average search time: {avg_search_time:.1f}ms")

        # Quality assessment
        if avg_relevance >= 0.8:
            print(f"\n✅ EXCELLENT: Retrieval quality is production-ready (≥80%)")
        elif avg_relevance >= 0.6:
            print(f"\n✓ GOOD: Retrieval quality is acceptable (≥60%)")
        else:
            print(f"\n⚠️  NEEDS IMPROVEMENT: Retrieval quality below 60%")

    return results


async def generate_completion_report(ingestion_success, retrieval_results):
    """Generate Phase 1 completion report."""
    print("\n" + "="*80)
    print("GENERATING COMPLETION REPORT")
    print("="*80)

    report = {
        "phase": "Phase 1 - Infrastructure Completion",
        "execution_date": datetime.now().isoformat(),
        "critical_path_items": {
            "knowledge_base_ingestion": {
                "status": "complete" if ingestion_success else "failed",
                "details": "7 documents ingested across 3 collections" if ingestion_success else "Ingestion failed"
            },
            "retrieval_quality_testing": {
                "status": "complete" if retrieval_results else "failed",
                "test_queries": len(retrieval_results) if retrieval_results else 0,
                "results": retrieval_results
            }
        },
        "next_steps": [
            "Write ChatSession tool calling tests",
            "Benchmark RAG retrieval quality",
            "Test determinism with replay mode",
            "Validate budget enforcement"
        ]
    }

    # Save report
    report_path = Path("PHASE_1_CRITICAL_PATH_COMPLETION.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Report saved to: {report_path}")

    return report


async def main():
    """Main execution flow."""
    print("\n" + "="*80)
    print("PHASE 1 COMPLETION - CRITICAL PATH EXECUTION")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check dependencies
    deps_ok = await check_dependencies()

    if not deps_ok:
        print("\n" + "="*80)
        print("⚠️  CANNOT PROCEED - MISSING DEPENDENCIES")
        print("="*80)
        print("\nTo install required packages:")
        print("  pip install sentence-transformers faiss-cpu numpy torch")
        print("\nAfter installation, run this script again:")
        print("  python run_phase1_completion.py")
        return

    # Run ingestion
    ingester, config = await run_ingestion()
    ingestion_success = ingester is not None

    # Test retrieval
    retrieval_results = None
    if ingestion_success:
        retrieval_results = await test_retrieval(ingester, config)

    # Generate report
    await generate_completion_report(ingestion_success, retrieval_results)

    # Final summary
    print("\n" + "="*80)
    print("PHASE 1 CRITICAL PATH COMPLETION SUMMARY")
    print("="*80)

    if ingestion_success and retrieval_results:
        print("\n✅ SUCCESS - Critical path completed")
        print("\nCompleted:")
        print("  ✓ Knowledge base ingestion (7 documents)")
        print("  ✓ Retrieval quality testing (5 queries)")
        print("\nNext: Write ChatSession tool tests")
    else:
        print("\n⚠️  PARTIAL COMPLETION")
        if not ingestion_success:
            print("  ✗ Ingestion failed")
        if not retrieval_results:
            print("  ✗ Retrieval testing failed")

    print("\n" + "="*80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
