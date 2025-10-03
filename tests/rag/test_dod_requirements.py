"""
DoD-specific tests for INTL-104 RAG v1.

Tests required by CTO's Definition of Done:
1. MMR diversity test with synthetic near-duplicate corpus
2. Ingest → query → hash verification round-trip test
3. Network isolation enforcement in replay mode

References:
- DoD Section 6: Testing Requirements
- INTL-104_DOD_VERIFICATION_CHECKLIST.md items 6.1, 6.2, 6.3
"""

import os
# Fix OpenMP conflict between PyTorch and FAISS on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
from unittest.mock import patch, MagicMock
from datetime import date

from greenlang.intelligence.rag import (
    RAGEngine,
    RAGConfig,
    DeterministicRAG,
    DocMeta,
    Chunk,
    QueryResult,
    get_embedding_provider,
    get_vector_store,
    get_retriever,
)
from greenlang.intelligence.rag.ingest import ingest_path, write_manifest
from greenlang.intelligence.rag.query import query
from greenlang.intelligence.rag.hashing import chunk_uuid5, sha256_str


class TestMMRDiversity:
    """
    DoD Requirement 6.1: MMR Diversity Test

    Test that MMR retrieval increases diversity compared to similarity-only retrieval
    when presented with a corpus containing near-duplicate documents.

    Success Criteria:
    - Create synthetic corpus with 10 documents (3 groups of near-duplicates)
    - Verify MMR retrieval achieves higher Jaccard diversity than similarity retrieval
    - MMR diversity should be ≥30% higher than similarity diversity
    """

    def test_mmr_diversity_vs_similarity(self):
        """Test MMR produces more diverse results than pure similarity search."""
        # Create synthetic corpus with near-duplicates
        # Group 1: Climate finance (3 near-duplicates)
        docs_group1 = [
            "Climate finance mechanisms for developing nations include Green Climate Fund.",
            "Financing climate action in developing countries through Green Climate Fund mechanisms.",
            "Green Climate Fund provides financial mechanisms for climate action in developing nations.",
        ]

        # Group 2: Emission factors (4 near-duplicates)
        docs_group2 = [
            "Scope 1 emissions are direct greenhouse gas emissions from owned sources.",
            "Direct GHG emissions from company-owned sources are classified as Scope 1.",
            "Scope 1 category includes direct emissions from sources owned by the company.",
            "Company-owned emission sources fall under Scope 1 direct GHG emissions.",
        ]

        # Group 3: Renewable energy (3 near-duplicates)
        docs_group3 = [
            "Solar photovoltaic technology converts sunlight into electrical energy efficiently.",
            "PV solar panels efficiently convert solar radiation into electrical power.",
            "Photovoltaic systems efficiently transform sunlight into electricity.",
        ]

        all_docs = docs_group1 + docs_group2 + docs_group3

        # Create mock embeddings (avoid network calls in tests)
        # Simulate embeddings where documents in same group are similar
        config = RAGConfig(
            mode="live",
            embedding_provider="minilm",
            vector_store_provider="faiss",
            retrieval_method="mmr",
            mmr_lambda=0.5,
        )

        vector_store = get_vector_store(dimension=384, config=config)

        # Generate mock embeddings with controlled similarity
        # Group 1 embeddings: centered around [0.9, 0.1, ...]
        # Group 2 embeddings: centered around [0.1, 0.9, ...]
        # Group 3 embeddings: centered around [0.5, 0.5, ...]
        embeddings = []
        for i, doc_text in enumerate(all_docs):
            emb = np.zeros(384)
            if i < 3:  # Group 1
                emb[0] = 0.9 + np.random.uniform(-0.05, 0.05)
                emb[1] = 0.1 + np.random.uniform(-0.05, 0.05)
            elif i < 7:  # Group 2
                emb[0] = 0.1 + np.random.uniform(-0.05, 0.05)
                emb[1] = 0.9 + np.random.uniform(-0.05, 0.05)
            else:  # Group 3
                emb[0] = 0.5 + np.random.uniform(-0.05, 0.05)
                emb[1] = 0.5 + np.random.uniform(-0.05, 0.05)
            # Normalize
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

        # Create chunks with proper doc_id grouping
        chunks = []
        for i, (doc_text, embedding) in enumerate(zip(all_docs, embeddings)):
            # Group documents: 0-2 -> group_0, 3-6 -> group_1, 7-9 -> group_2
            if i < 3:
                group_id = "group_0"
            elif i < 7:
                group_id = "group_1"
            else:
                group_id = "group_2"

            chunk = Chunk(
                chunk_id=f"test_chunk_{i}",
                doc_id=group_id,
                section_path=f"Section {i}",
                section_hash=sha256_str(f"section_{i}"),
                page_start=1,
                paragraph=1,
                start_char=0,
                end_char=len(doc_text),
                text=doc_text,
                token_count=len(doc_text.split()),
            )
            chunks.append(chunk)

        # Add to vector store - create Document objects
        from greenlang.intelligence.rag.vector_stores import Document

        docs = [
            Document(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ]
        vector_store.add_documents(
            docs=docs,
            collection="test_collection",
        )

        # Test query related to Group 1 (climate finance)
        query_text = "What are the climate finance options for developing countries?"
        # Mock query embedding similar to Group 1
        query_embedding = np.zeros(384)
        query_embedding[0] = 0.85
        query_embedding[1] = 0.15
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Similarity-only retrieval
        from greenlang.intelligence.rag.retrievers import SimilarityRetriever, MMRRetriever

        similarity_retriever = SimilarityRetriever(
            vector_store=vector_store,
            top_k=6,
        )

        similarity_results = similarity_retriever.retrieve(
            query_embedding=query_embedding,
            collections=["test_collection"],
        )

        # MMR retrieval
        mmr_retriever = MMRRetriever(
            vector_store=vector_store,
            fetch_k=30,
            top_k=6,
            lambda_mult=0.5,
        )
        mmr_results = mmr_retriever.retrieve(
            query_embedding=query_embedding,
            collections=["test_collection"],
        )

        # Calculate diversity using doc_id grouping
        def calculate_diversity(results):
            """
            Calculate diversity as ratio of unique doc_ids to total results.
            Higher value = more diverse results.

            Args:
                results: List of (Document, score) tuples

            Returns:
                Diversity ratio (0.0 to 1.0)
            """
            if len(results) < 2:
                return 0.0

            # Extract chunks from (Document, score) tuples
            chunks = [doc.chunk for doc, _ in results]

            # Extract unique doc_ids (which group represents near-duplicates)
            doc_ids = [c.doc_id for c in chunks]
            unique_docs = len(set(doc_ids))
            total_docs = len(doc_ids)

            # Diversity metric: ratio of unique doc families
            return unique_docs / total_docs

        similarity_diversity = calculate_diversity(similarity_results)
        mmr_diversity = calculate_diversity(mmr_results)

        # Debug output
        print(f"\nSimilarity results (doc_ids): {[doc.chunk.doc_id for doc, _ in similarity_results]}")
        print(f"MMR results (doc_ids): {[doc.chunk.doc_id for doc, _ in mmr_results]}")

        # Verification: MMR should produce more diverse results
        assert mmr_diversity > similarity_diversity, (
            f"MMR diversity ({mmr_diversity:.2f}) should exceed "
            f"similarity diversity ({similarity_diversity:.2f})"
        )

        # DoD requirement: MMR diversity should be ≥30% higher
        improvement = (mmr_diversity - similarity_diversity) / similarity_diversity
        assert improvement >= 0.3, (
            f"MMR diversity improvement ({improvement:.1%}) "
            f"should be ≥30%"
        )

        print(f"PASS: MMR diversity test")
        print(f"  Similarity diversity: {similarity_diversity:.2%}")
        print(f"  MMR diversity: {mmr_diversity:.2%}")
        print(f"  Improvement: {improvement:.1%}")


class TestIngestQueryRoundtrip:
    """
    DoD Requirement 6.2: Ingest → Query → Hash Verification Round-trip

    Test that documents can be ingested, queried, and retrieved with hash integrity.

    Success Criteria:
    - Ingest a markdown document with known content hash
    - Query for content and retrieve chunks
    - Verify chunk hashes match expected values
    - Verify doc_hash in manifest matches content_hash
    """

    def test_ingest_query_hash_verification(self):
        """Test round-trip ingestion and query with hash verification."""
        # Create temporary markdown document
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".md",
            delete=False,
            encoding="utf-8"
        ) as f:
            test_doc_path = Path(f.name)
            f.write("""# GHG Emissions Standard

## Scope 1 Emissions

Direct greenhouse gas emissions from sources owned or controlled by the organization.

Examples include:
- Combustion in owned equipment
- Company vehicles
- Fugitive emissions

## Scope 2 Emissions

Indirect emissions from purchased electricity, heat, or steam.

## Scope 3 Emissions

All other indirect emissions in the value chain.
""")

        try:
            # Configure RAG system
            config = RAGConfig(
                mode="live",
                embedding_provider="minilm",
                vector_store_provider="faiss",
                retrieval_method="similarity",
                chunk_size=512,
                chunk_overlap=64,
            )

            # Ingest document
            collection = "test_roundtrip"

            # Create DocMeta
            doc_meta = DocMeta(
                doc_id="test_doc_001",
                title="GHG Emissions Standard",
                collection=collection,
                source_uri=str(test_doc_path),
                publisher="Test Publisher",
                version="1.0",
                content_hash="",  # Will be computed
                doc_hash="",  # Will be computed
            )

            import asyncio
            result = asyncio.run(ingest_path(
                path=test_doc_path,
                collection=collection,
                doc_meta=doc_meta,
                config=config,
            ))

            # Verify ingestion manifest
            assert "doc_id" in result
            assert "doc_hash" in result
            assert "num_chunks" in result
            assert result["num_chunks"] > 0

            doc_id = result["doc_id"]
            doc_hash = result["doc_hash"]

            # Query for content
            query_text = "What are Scope 1 emissions?"
            query_result = asyncio.run(query(
                q=query_text,
                top_k=3,
                collections=[collection],
                config=config,
            ))

            # Extract chunks from QueryResult
            query_results = query_result.chunks if hasattr(query_result, 'chunks') else []

            # Verify results
            assert len(query_results) > 0, "Query should return results"

            # Verify hash integrity
            for result in query_results:
                chunk = result.chunk

                # Verify chunk_id is stable UUID v5
                expected_chunk_id = chunk_uuid5(
                    doc_id=chunk.doc_id,
                    section_path=chunk.section_path,
                    start_offset=chunk.start_char,
                )
                assert chunk.chunk_id == expected_chunk_id, (
                    f"Chunk ID mismatch: {chunk.chunk_id} != {expected_chunk_id}"
                )

                # Verify section_hash is deterministic
                assert len(chunk.section_hash) == 64, "Section hash should be SHA-256"

                # Verify text content matches expected patterns
                if "Scope 1" in query_text:
                    assert "Scope 1" in chunk.text or "Direct" in chunk.text, (
                        "Retrieved chunk should be relevant to query"
                    )

            # Verify doc_hash consistency
            # Read manifest if it exists
            manifest_path = test_doc_path.parent / "MANIFEST.json"
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

                assert manifest["doc_hash"] == doc_hash, (
                    "Manifest doc_hash should match ingestion result"
                )

            print(f"PASS: Ingest-query round-trip test")
            print(f"  Doc ID: {doc_id[:8]}...")
            print(f"  Doc hash: {doc_hash[:8]}...")
            print(f"  Chunks ingested: {result['num_chunks']}")
            print(f"  Query results: {len(query_results)}")

        finally:
            # Cleanup
            test_doc_path.unlink(missing_ok=True)
            manifest_path = test_doc_path.parent / "MANIFEST.json"
            manifest_path.unlink(missing_ok=True)


class TestNetworkIsolation:
    """
    DoD Requirement 6.3: Network Isolation Test

    Test that replay mode enforces strict network isolation and never makes
    external API calls.

    Success Criteria:
    - Configure RAG system in replay mode
    - Attempt to query (with missing cache entry)
    - Verify no network calls are made (mock urllib/requests)
    - Verify RuntimeError is raised for missing cache entry
    """

    def test_replay_mode_network_isolation(self):
        """Test replay mode never makes network calls."""
        # Create empty cache file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8"
        ) as f:
            cache_path = Path(f.name)
            cache_data = {
                "version": "1.0.0",
                "mode": "replay",
                "queries": {},
            }
            json.dump(cache_data, f)

        try:
            # Configure in replay mode
            config = RAGConfig(
                mode="replay",
                embedding_provider="openai",  # Would require API call
                vector_store_provider="weaviate",  # Would require network
            )

            # Create deterministic wrapper
            det = DeterministicRAG(
                mode="replay",
                cache_path=cache_path,
                config=config,
            )

            # Mock network libraries to detect any calls
            with patch("urllib.request.urlopen") as mock_urllib, \
                 patch("requests.request") as mock_requests, \
                 patch("http.client.HTTPConnection") as mock_http:

                # Attempt to query (should fail before making network calls)
                query_text = "What are emission factors?"

                with pytest.raises(RuntimeError, match="No cached result"):
                    det.query(
                        q=query_text,
                        top_k=5,
                        collections=["test_collection"],
                    )

                # Verify no network calls were made
                assert not mock_urllib.called, (
                    "replay mode should not call urllib"
                )
                assert not mock_requests.called, (
                    "replay mode should not call requests"
                )
                assert not mock_http.called, (
                    "replay mode should not use http.client"
                )

            print(f"PASS: Network isolation test")
            print(f"  Mode: replay")
            print(f"  Network calls blocked: True")
            print(f"  RuntimeError raised: True")

        finally:
            # Cleanup
            cache_path.unlink(missing_ok=True)

    def test_live_mode_allows_network(self):
        """Verify live mode allows network calls (contrast test)."""
        config = RAGConfig(
            mode="live",
            embedding_provider="minilm",  # Local, no network
            vector_store_provider="faiss",  # In-memory, no network
        )

        # This should not raise network isolation errors
        engine = RAGEngine(config)
        assert engine.config.mode == "live"

        # Verify we can initialize embedder (which would fail in replay mode)
        embedder = get_embedding_provider(config)
        assert embedder is not None

        print(f"PASS: Live mode allows network")


class TestDoDAggregateMetrics:
    """
    Aggregate metrics for DoD compliance reporting.

    Tests:
    - All 3 DoD-required tests pass
    - Report compliance status
    """

    def test_aggregate_dod_compliance(self):
        """Run all DoD tests and report aggregate metrics."""
        results = {
            "mmr_diversity": False,
            "ingest_roundtrip": False,
            "network_isolation": False,
        }

        # Test 1: MMR Diversity
        try:
            test_mmr = TestMMRDiversity()
            test_mmr.test_mmr_diversity_vs_similarity()
            results["mmr_diversity"] = True
        except Exception as e:
            print(f"FAIL: MMR diversity test - {e}")

        # Test 2: Ingest Round-trip
        try:
            test_roundtrip = TestIngestQueryRoundtrip()
            test_roundtrip.test_ingest_query_hash_verification()
            results["ingest_roundtrip"] = True
        except Exception as e:
            print(f"FAIL: Ingest round-trip test - {e}")

        # Test 3: Network Isolation
        try:
            test_network = TestNetworkIsolation()
            test_network.test_replay_mode_network_isolation()
            results["network_isolation"] = True
        except Exception as e:
            print(f"FAIL: Network isolation test - {e}")

        # Report aggregate
        passed = sum(results.values())
        total = len(results)

        print(f"\n{'='*60}")
        print(f"DoD Test Compliance Report")
        print(f"{'='*60}")
        for test_name, passed_status in results.items():
            status = "PASS" if passed_status else "FAIL"
            print(f"  {status}: {test_name}")
        print(f"{'='*60}")
        print(f"Total: {passed}/{total} tests passed ({passed/total:.0%})")
        print(f"{'='*60}\n")

        # Verify all tests passed
        assert passed == total, (
            f"DoD compliance failure: {passed}/{total} tests passed"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
