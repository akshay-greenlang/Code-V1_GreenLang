#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreenLang RAG Document Ingestion Pipeline
Build and manage vector store from climate knowledge documents
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from greenlang.determinism import DeterministicClock

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from langchain_community.document_loaders import (
        TextLoader,
        PyPDFLoader,
        DirectoryLoader,
        JSONLoader,
        CSVLoader,
        UnstructuredMarkdownLoader
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS, Chroma
    from langchain.schema import Document
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install required packages: pip install langchain langchain-community faiss-cpu sentence-transformers pypdf chromadb")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GreenLangVectorStore:
    """Manage vector store for GreenLang RAG system"""
    
    def __init__(
        self,
        docs_path: str = "knowledge_base/documents",
        vector_store_path: str = "knowledge_base/vector_store",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_chroma: bool = False
    ):
        """
        Initialize the vector store builder
        
        Args:
            docs_path: Path to documents directory
            vector_store_path: Path to save vector store
            embedding_model: HuggingFace model for embeddings
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            use_chroma: Use ChromaDB instead of FAISS
        """
        self.docs_path = Path(docs_path)
        self.vector_store_path = Path(vector_store_path)
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_chroma = use_chroma
        
        # Create directories if they don't exist
        self.docs_path.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.documents = []
        self.vector_store = None
    
    def load_documents(self) -> List[Document]:
        """Load all documents from the documents directory"""
        logger.info(f"Loading documents from {self.docs_path}")
        documents = []
        
        # Define loaders for different file types
        loaders = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".md": UnstructuredMarkdownLoader,
            ".json": lambda path: JSONLoader(
                file_path=path,
                jq_schema='.[]',
                text_content=False
            ),
            ".csv": CSVLoader
        }
        
        # Load all files
        for file_path in self.docs_path.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                
                if ext in loaders:
                    try:
                        loader_class = loaders[ext]
                        loader = loader_class(str(file_path))
                        docs = loader.load()
                        
                        # Add metadata
                        for doc in docs:
                            doc.metadata.update({
                                "source": str(file_path),
                                "file_type": ext,
                                "ingested_at": DeterministicClock.now().isoformat()
                            })
                        
                        documents.extend(docs)
                        logger.info(f"Loaded {len(docs)} documents from {file_path.name}")
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        logger.info("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def create_vector_store(self, documents: List[Document]) -> None:
        """Create vector store from documents"""
        logger.info("Creating vector store...")
        
        if self.use_chroma:
            # Use ChromaDB
            persist_directory = str(self.vector_store_path / "chroma")
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            logger.info(f"ChromaDB vector store created at {persist_directory}")
        else:
            # Use FAISS
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            # Save FAISS index
            faiss_path = str(self.vector_store_path / "faiss")
            self.vector_store.save_local(faiss_path)
            logger.info(f"FAISS vector store saved to {faiss_path}")
    
    def load_vector_store(self) -> Any:
        """Load existing vector store"""
        logger.info("Loading existing vector store...")
        
        if self.use_chroma:
            persist_directory = str(self.vector_store_path / "chroma")
            if Path(persist_directory).exists():
                self.vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info("ChromaDB vector store loaded")
            else:
                logger.warning("No existing ChromaDB found")
        else:
            faiss_path = str(self.vector_store_path / "faiss")
            if Path(faiss_path).exists():
                self.vector_store = FAISS.load_local(
                    faiss_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("FAISS vector store loaded")
            else:
                logger.warning("No existing FAISS index found")
        
        return self.vector_store
    
    def add_documents(self, new_documents: List[Document]) -> None:
        """Add new documents to existing vector store"""
        if self.vector_store is None:
            logger.error("No vector store loaded. Create or load one first.")
            return
        
        logger.info(f"Adding {len(new_documents)} new documents to vector store")
        chunks = self.split_documents(new_documents)
        
        if self.use_chroma:
            self.vector_store.add_documents(chunks)
        else:
            self.vector_store.add_documents(chunks)
            # Save updated FAISS index
            faiss_path = str(self.vector_store_path / "faiss")
            self.vector_store.save_local(faiss_path)
        
        logger.info("Documents added successfully")
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search vector store for relevant documents"""
        if self.vector_store is None:
            logger.error("No vector store loaded")
            return []
        
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def build(self) -> None:
        """Complete pipeline to build vector store"""
        logger.info("Starting vector store build process...")
        
        # Load documents
        documents = self.load_documents()
        
        if not documents:
            logger.warning("No documents found to process")
            return
        
        # Split into chunks
        chunks = self.split_documents(documents)
        
        # Create vector store
        self.create_vector_store(chunks)
        
        # Save metadata
        metadata = {
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "vector_store_type": "ChromaDB" if self.use_chroma else "FAISS",
            "created_at": DeterministicClock.now().isoformat()
        }
        
        metadata_path = self.vector_store_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Vector store build complete. Metadata saved to {metadata_path}")
    
    def add_climate_docs(self) -> None:
        """Add default climate science documents"""
        logger.info("Adding default climate science documents...")
        
        # Create sample climate documents
        climate_docs = [
            Document(
                page_content="""
                Carbon Emissions and Climate Change
                
                Carbon dioxide (CO2) is the primary greenhouse gas emitted through human activities.
                In 2019, CO2 accounted for about 80% of all U.S. greenhouse gas emissions from 
                human activities. The main human activity that emits CO2 is the combustion of 
                fossil fuels (coal, natural gas, and oil) for energy and transportation.
                
                Key emission factors:
                - Electricity: 0.4-0.7 kg CO2/kWh (varies by grid)
                - Natural Gas: 0.185 kg CO2/kWh
                - Coal: 0.34 kg CO2/kWh
                - Gasoline: 2.31 kg CO2/liter
                """,
                metadata={"source": "climate_basics.txt", "category": "emissions"}
            ),
            Document(
                page_content="""
                Building Energy Efficiency
                
                Buildings account for approximately 40% of global energy consumption and 36% of 
                CO2 emissions. Key strategies for reducing building emissions include:
                
                1. Energy-efficient HVAC systems
                2. Improved insulation
                3. LED lighting
                4. Smart building controls
                5. Renewable energy integration
                
                Energy intensity benchmarks:
                - Office buildings: 100-300 kWh/m²/year
                - Retail: 150-400 kWh/m²/year
                - Industrial: 200-500 kWh/m²/year
                """,
                metadata={"source": "building_efficiency.txt", "category": "buildings"}
            ),
            Document(
                page_content="""
                Renewable Energy Sources
                
                Renewable energy sources are crucial for decarbonization:
                
                Solar Energy:
                - Zero operational emissions
                - Lifecycle emissions: 40-50 g CO2/kWh
                - Capacity factors: 15-25%
                
                Wind Energy:
                - Zero operational emissions
                - Lifecycle emissions: 10-15 g CO2/kWh
                - Capacity factors: 25-45%
                
                Comparison with fossil fuels:
                - Coal: 820-1000 g CO2/kWh
                - Natural Gas: 490 g CO2/kWh
                """,
                metadata={"source": "renewable_energy.txt", "category": "energy"}
            ),
            Document(
                page_content="""
                Carbon Footprint Calculation Methods
                
                The GHG Protocol Corporate Standard classifies emissions into three scopes:
                
                Scope 1: Direct emissions from owned or controlled sources
                - On-site fuel combustion
                - Company vehicles
                - Fugitive emissions
                
                Scope 2: Indirect emissions from purchased electricity, steam, heating & cooling
                - Location-based method: Uses average emission factors for the grid
                - Market-based method: Uses emission factors from contractual instruments
                
                Scope 3: All other indirect emissions in the value chain
                - Purchased goods and services
                - Business travel
                - Employee commuting
                - Waste disposal
                """,
                metadata={"source": "ghg_protocol.txt", "category": "methodology"}
            )
        ]
        
        # Save documents to files
        for doc in climate_docs:
            file_name = doc.metadata["source"]
            file_path = self.docs_path / file_name
            with open(file_path, 'w') as f:
                f.write(doc.page_content)
            logger.info(f"Created {file_name}")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GreenLang RAG Vector Store Builder")
    parser.add_argument("--docs-path", default="knowledge_base/documents", help="Path to documents")
    parser.add_argument("--vector-path", default="knowledge_base/vector_store", help="Path to vector store")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")
    parser.add_argument("--use-chroma", action="store_true", help="Use ChromaDB instead of FAISS")
    parser.add_argument("--add-defaults", action="store_true", help="Add default climate documents")
    parser.add_argument("--test-search", help="Test search with a query")
    
    args = parser.parse_args()
    
    # Initialize vector store builder
    builder = GreenLangVectorStore(
        docs_path=args.docs_path,
        vector_store_path=args.vector_path,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_chroma=args.use_chroma
    )
    
    # Add default documents if requested
    if args.add_defaults:
        builder.add_climate_docs()
    
    # Build vector store
    builder.build()
    
    # Test search if requested
    if args.test_search:
        builder.load_vector_store()
        results = builder.search(args.test_search, k=3)
        print(f"\nSearch results for: '{args.test_search}'")
        print("-" * 50)
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"   Content: {doc.page_content[:200]}...")
    
    print("\nVector store build complete!")
    print(f"Documents processed from: {args.docs_path}")
    print(f"Vector store saved to: {args.vector_path}")


if __name__ == "__main__":
    main()