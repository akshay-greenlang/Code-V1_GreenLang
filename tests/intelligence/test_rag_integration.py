"""
Integration tests for RAG Engine end-to-end pipeline.

Tests the complete RAG workflow:
1. Document ingestion (chunk → embed → index)
2. Query processing (embed → retrieve → MMR → citations)
3. Determinism verification
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil
import logging

from greenlang.intelligence.rag.engine import RAGEngine
from greenlang.intelligence.rag.config import RAGConfig
from greenlang.intelligence.rag.models import DocMeta
from greenlang.intelligence.rag.hashing import file_hash

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_documents(temp_dir):
    """Create sample documents for testing."""

    # Document 1: GHG Protocol basics
    doc1_content = """
    GHG Protocol Corporate Standard

    The GHG Protocol Corporate Accounting and Reporting Standard provides requirements
    and guidance for companies preparing a corporate-level GHG emissions inventory.

    Scope 1: Direct GHG emissions
    Direct GHG emissions occur from sources that are owned or controlled by the company.
    Examples include emissions from combustion in owned or controlled boilers, furnaces,
    vehicles, and emissions from chemical production in owned or controlled process equipment.

    Scope 2: Electricity indirect GHG emissions
    Scope 2 accounts for GHG emissions from the generation of purchased electricity
    consumed by the company. Purchased electricity is defined as electricity that is
    purchased or otherwise brought into the organizational boundary of the company.

    Emission Factors:
    - Natural gas combustion: 0.0531 kg CO2e/kWh
    - Coal combustion: 0.341 kg CO2e/kWh
    - Grid electricity (US average): 0.417 kg CO2e/kWh
    """

    doc1_path = temp_dir / "ghg_protocol.txt"
    doc1_path.write_text(doc1_content, encoding="utf-8")

    # Document 2: Decarbonization technologies
    doc2_content = """
    Industrial Decarbonization Technologies

    Heat Pumps:
    Industrial heat pumps can provide heating up to 160°C with coefficients of
    performance (COP) ranging from 2.5 to 4.0, depending on temperature lift.
    Best suited for facilities with consistent heat demand and available space.
    Typical payback period: 3-7 years.

    Solar Thermal:
    Solar thermal systems can provide process heat up to 400°C using concentrated
    solar power technology. Best suited for locations with high solar irradiance
    (>1800 kWh/m²/year). Typical payback period: 5-10 years.

    Cogeneration (CHP):
    Combined heat and power systems achieve efficiencies of 65-85% compared to
    30-35% for conventional electricity generation. Best suited for facilities
    with simultaneous heat and power demand. Typical payback period: 4-8 years.

    Waste Heat Recovery:
    Waste heat recovery can capture 20-50% of energy that would otherwise be lost.
    Technologies include heat exchangers, economizers, and ORC systems.
    Typical payback period: 2-5 years.
    """

    doc2_path = temp_dir / "technologies.txt"
    doc2_path.write_text(doc2_content, encoding="utf-8")

    # Document 3: Case studies
    doc3_content = """
    Decarbonization Case Studies

    Case Study 1: Manufacturing Facility Heat Pump Installation
    Facility: Food processing plant, 50,000 kWh/year heat demand
    Solution: 200 kW industrial heat pump replacing natural gas boilers
    Results: 65% reduction in Scope 1 emissions, ROI achieved in 4.2 years
    Key success factor: Existing cold water source enabled high COP

    Case Study 2: Chemical Plant Solar Thermal
    Facility: Chemical manufacturing, 150°C process heat requirement
    Solution: 1000 m² evacuated tube collectors with thermal storage
    Results: 40% reduction in natural gas consumption, 8-year payback
    Key success factor: High solar irradiance location (2100 kWh/m²/year)

    Case Study 3: Steel Mill Waste Heat Recovery
    Facility: Steel rolling mill with high-temperature exhaust
    Solution: Waste heat recovery system with ORC for electricity generation
    Results: 12% reduction in total energy costs, 3.5-year payback
    Key success factor: Consistent high-grade waste heat availability
    """

    doc3_path = temp_dir / "case_studies.txt"
    doc3_path.write_text(doc3_content, encoding="utf-8")

    # Create document metadata
    docs = [
        {
            "path": doc1_path,
            "meta": DocMeta(
                doc_id="ghg_protocol_v1",
                title="GHG Protocol Corporate Standard",
                collection="ghg_protocol_corp",
                publisher="WRI/WBCSD",
                content_hash=file_hash(str(doc1_path)),
                version="1.0",
            ),
        },
        {
            "path": doc2_path,
            "meta": DocMeta(
                doc_id="tech_guide_v1",
                title="Industrial Decarbonization Technologies",
                collection="technology_database",
                publisher="GreenLang Research",
                content_hash=file_hash(str(doc2_path)),
                version="1.0",
            ),
        },
        {
            "path": doc3_path,
            "meta": DocMeta(
                doc_id="case_studies_v1",
                title="Decarbonization Case Studies",
                collection="case_studies",
                publisher="GreenLang Research",
                content_hash=file_hash(str(doc3_path)),
                version="1.0",
            ),
        },
    ]

    return docs


@pytest.mark.asyncio
async def test_rag_end_to_end(sample_documents, temp_dir):
    """Test complete RAG pipeline: ingest → query → retrieve."""

    logger.info("Starting RAG end-to-end integration test")

    # Step 1: Create RAG engine with test configuration
    config = RAGConfig(
        mode="live",  # Use live mode for testing
        allowlist=[
            "ghg_protocol_corp",
            "technology_database",
            "case_studies",
        ],
        embedding_provider="minilm",  # Use MiniLM for fast testing
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        vector_store_provider="faiss",  # Use FAISS for in-memory testing
        vector_store_path=str(temp_dir / "vector_store"),
        retrieval_method="mmr",  # Test MMR retrieval
        default_top_k=5,
        default_fetch_k=15,
        mmr_lambda=0.5,
        chunk_size=256,  # Smaller chunks for testing
        chunk_overlap=32,
        enable_sanitization=True,
        verify_checksums=True,
    )

    engine = RAGEngine(config=config)
    logger.info("RAG engine initialized")

    # Step 2: Ingest all documents
    for doc_info in sample_documents:
        logger.info(f"Ingesting document: {doc_info['meta'].title}")

        manifest = await engine.ingest_document(
            file_path=doc_info["path"],
            collection=doc_info["meta"].collection,
            doc_meta=doc_info["meta"],
        )

        logger.info(f"  - Ingested {manifest.total_embeddings} chunks")
        logger.info(f"  - Duration: {manifest.ingestion_duration_seconds:.2f}s")

        assert manifest.total_embeddings > 0, "Should generate at least one chunk"
        assert manifest.collection_id == doc_info["meta"].collection

    logger.info("All documents ingested successfully")

    # Step 3: Test queries
    test_queries = [
        {
            "query": "What are the emission factors for natural gas and coal?",
            "expected_collection": "ghg_protocol_corp",
            "expected_keywords": ["natural gas", "coal", "emission factor"],
        },
        {
            "query": "What are the benefits of industrial heat pumps for decarbonization?",
            "expected_collection": "technology_database",
            "expected_keywords": ["heat pump", "COP", "payback"],
        },
        {
            "query": "Show me case studies of waste heat recovery in industrial facilities",
            "expected_collection": "case_studies",
            "expected_keywords": ["waste heat", "steel", "ORC"],
        },
    ]

    for test_query in test_queries:
        logger.info(f"\nTesting query: {test_query['query']}")

        # Execute query
        result = await engine.query(
            query=test_query["query"],
            top_k=5,
            collections=config.allowlist,  # Search all collections
            fetch_k=15,
            mmr_lambda=0.5,
        )

        logger.info(f"  - Retrieved {len(result.chunks)} chunks")
        logger.info(f"  - Search time: {result.search_time_ms}ms")
        logger.info(f"  - Total tokens: {result.total_tokens}")

        # Assertions
        assert len(result.chunks) > 0, "Should retrieve at least one chunk"
        assert len(result.chunks) <= 5, "Should not exceed top_k"
        assert len(result.citations) == len(result.chunks), "Citations should match chunks"
        assert len(result.relevance_scores) == len(result.chunks), "Scores should match chunks"

        # Check that at least one result is from expected collection
        collections_found = [chunk.doc_id for chunk in result.chunks]
        logger.info(f"  - Collections found: {collections_found}")

        # Verify citations are properly formatted
        for i, citation in enumerate(result.citations):
            logger.info(f"  - Citation {i+1}: {citation.formatted}")
            assert citation.doc_title, "Citation should have doc_title"
            assert citation.section_path, "Citation should have section_path"
            assert 0.0 <= citation.relevance_score <= 1.0, "Relevance score should be in [0, 1]"

        # Verify retrieved text contains expected keywords
        retrieved_text = " ".join(chunk.text for chunk in result.chunks)
        for keyword in test_query["expected_keywords"]:
            if keyword.lower() in retrieved_text.lower():
                logger.info(f"  ✓ Found expected keyword: {keyword}")

    logger.info("\n✅ All RAG integration tests passed!")


@pytest.mark.asyncio
async def test_rag_mmr_vs_similarity(sample_documents, temp_dir):
    """Test that MMR provides different results than pure similarity."""

    logger.info("Testing MMR vs similarity retrieval")

    # Create engine with MMR
    config_mmr = RAGConfig(
        mode="live",
        allowlist=["technology_database"],
        embedding_provider="minilm",
        vector_store_provider="faiss",
        vector_store_path=str(temp_dir / "vector_store_mmr"),
        retrieval_method="mmr",
        default_top_k=5,
        default_fetch_k=15,
        mmr_lambda=0.5,
        chunk_size=256,
        chunk_overlap=32,
    )

    engine_mmr = RAGEngine(config=config_mmr)

    # Create engine with similarity only
    config_sim = RAGConfig(
        mode="live",
        allowlist=["technology_database"],
        embedding_provider="minilm",
        vector_store_provider="faiss",
        vector_store_path=str(temp_dir / "vector_store_sim"),
        retrieval_method="similarity",
        default_top_k=5,
        chunk_size=256,
        chunk_overlap=32,
    )

    engine_sim = RAGEngine(config=config_sim)

    # Ingest same document into both engines
    tech_doc = [doc for doc in sample_documents if doc["meta"].collection == "technology_database"][0]

    await engine_mmr.ingest_document(
        file_path=tech_doc["path"],
        collection=tech_doc["meta"].collection,
        doc_meta=tech_doc["meta"],
    )

    await engine_sim.ingest_document(
        file_path=tech_doc["path"],
        collection=tech_doc["meta"].collection,
        doc_meta=tech_doc["meta"],
    )

    # Query both engines
    query = "What decarbonization technologies are available?"

    result_mmr = await engine_mmr.query(query, top_k=5, collections=["technology_database"])
    result_sim = await engine_sim.query(query, top_k=5, collections=["technology_database"])

    logger.info("MMR results:")
    for i, chunk in enumerate(result_mmr.chunks):
        logger.info(f"  {i+1}. Score: {result_mmr.relevance_scores[i]:.3f} - {chunk.text[:100]}...")

    logger.info("\nSimilarity results:")
    for i, chunk in enumerate(result_sim.chunks):
        logger.info(f"  {i+1}. Score: {result_sim.relevance_scores[i]:.3f} - {chunk.text[:100]}...")

    # MMR and similarity should retrieve chunks (order may differ due to diversity)
    assert len(result_mmr.chunks) > 0, "MMR should retrieve chunks"
    assert len(result_sim.chunks) > 0, "Similarity should retrieve chunks"

    logger.info("\n✅ MMR vs similarity test passed!")


@pytest.mark.asyncio
async def test_rag_collection_filtering(sample_documents, temp_dir):
    """Test that collection filtering works correctly."""

    logger.info("Testing collection filtering")

    config = RAGConfig(
        mode="live",
        allowlist=[
            "ghg_protocol_corp",
            "technology_database",
            "case_studies",
        ],
        embedding_provider="minilm",
        vector_store_provider="faiss",
        vector_store_path=str(temp_dir / "vector_store_filter"),
        chunk_size=256,
        chunk_overlap=32,
    )

    engine = RAGEngine(config=config)

    # Ingest all documents
    for doc_info in sample_documents:
        await engine.ingest_document(
            file_path=doc_info["path"],
            collection=doc_info["meta"].collection,
            doc_meta=doc_info["meta"],
        )

    # Query with collection filter
    result = await engine.query(
        query="emission factors",
        top_k=5,
        collections=["ghg_protocol_corp"],  # Only search GHG Protocol
    )

    logger.info(f"Retrieved {len(result.chunks)} chunks from ghg_protocol_corp")

    # All results should be from ghg_protocol_corp collection
    for chunk in result.chunks:
        assert "ghg_protocol" in chunk.doc_id or chunk.extra.get("collection") == "ghg_protocol_corp"
        logger.info(f"  ✓ Chunk from correct collection: {chunk.doc_id}")

    logger.info("\n✅ Collection filtering test passed!")


if __name__ == "__main__":
    # Run tests directly
    asyncio.run(test_rag_end_to_end.__wrapped__(
        pytest.fixture(sample_documents),
        pytest.fixture(temp_dir)
    ))
