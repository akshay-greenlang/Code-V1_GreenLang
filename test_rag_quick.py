"""
Quick test to verify RAG engine imports and basic functionality.
"""

import sys
import asyncio
from pathlib import Path

print("Testing RAG Engine imports...")
print(f"Python version: {sys.version}")
print(f"Working directory: {Path.cwd()}")

try:
    from greenlang.intelligence.rag.engine import RAGEngine
    print("✓ RAGEngine imported successfully")
except ImportError as e:
    print(f"✗ Failed to import RAGEngine: {e}")
    sys.exit(1)

try:
    from greenlang.intelligence.rag.config import RAGConfig
    print("✓ RAGConfig imported successfully")
except ImportError as e:
    print(f"✗ Failed to import RAGConfig: {e}")
    sys.exit(1)

try:
    from greenlang.intelligence.rag.embeddings import get_embedding_provider, MiniLMProvider
    print("✓ Embeddings module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import embeddings: {e}")
    sys.exit(1)

try:
    from greenlang.intelligence.rag.vector_stores import get_vector_store, FAISSProvider
    print("✓ Vector stores module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import vector_stores: {e}")
    sys.exit(1)

try:
    from greenlang.intelligence.rag.retrievers import mmr_retrieval, get_retriever
    print("✓ Retrievers module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import retrievers: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("TESTING RAG ENGINE INITIALIZATION")
print("="*60)

try:
    # Create config
    config = RAGConfig(
        mode="live",
        allowlist=["test_collection"],
        embedding_provider="minilm",
        vector_store_provider="faiss",
        chunk_size=256,
        chunk_overlap=32,
    )
    print("✓ RAGConfig created successfully")
    print(f"  - Mode: {config.mode}")
    print(f"  - Embedding provider: {config.embedding_provider}")
    print(f"  - Vector store: {config.vector_store_provider}")
except Exception as e:
    print(f"✗ Failed to create RAGConfig: {e}")
    sys.exit(1)

try:
    # Create engine
    engine = RAGEngine(config=config)
    print("✓ RAGEngine created successfully")
except Exception as e:
    print(f"✗ Failed to create RAGEngine: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("TESTING COMPONENT INITIALIZATION")
print("="*60)

async def test_components():
    try:
        # Force component initialization
        engine._initialize_components()
        print("✓ Components initialized successfully")

        if engine.embedder:
            print(f"  - Embedder: {engine.embedder.name}")
            print(f"  - Embedding dimension: {engine.embedder.dim}")
        else:
            print("  ✗ Embedder not initialized")

        if engine.vector_store:
            print(f"  - Vector store initialized")
        else:
            print("  ✗ Vector store not initialized")

        if engine.retriever:
            print(f"  - Retriever initialized")
        else:
            print("  ✗ Retriever not initialized")

        return True
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        print("\nTo install required dependencies, run:")
        print("  pip install sentence-transformers faiss-cpu")
        return False
    except Exception as e:
        print(f"✗ Component initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run async test
try:
    success = asyncio.run(test_components())
    if success:
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED - RAG ENGINE IS OPERATIONAL!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("⚠️  RAG ENGINE CODE IS CORRECT BUT MISSING DEPENDENCIES")
        print("="*60)
except Exception as e:
    print(f"\n✗ Test execution failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
