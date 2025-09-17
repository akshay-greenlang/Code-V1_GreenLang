#!/usr/bin/env python3
"""
Test script for GreenLang RAG implementation
"""

import sys
from pathlib import Path

# Removed sys.path manipulation - using installed package

from greenlang.cli.assistant_rag import RAGAssistant


def test_rag_assistant():
    """Test the RAG-enhanced assistant"""
    
    print("=" * 60)
    print("GreenLang RAG Assistant Test")
    print("=" * 60)
    
    # Initialize assistant
    print("\n1. Initializing RAG Assistant...")
    assistant = RAGAssistant()
    
    # Check component status
    print(f"   - LLM Available: {assistant.llm_available}")
    print(f"   - RAG Available: {assistant.rag_available}")
    
    if assistant.rag_available:
        print("   - Vector Store: Loaded successfully")
    else:
        print("   - Vector Store: Not available (run build_vector_store.py first)")
    
    # Test queries
    test_queries = [
        "What is the emission factor for electricity in India?",
        "How do I calculate carbon emissions for buildings?",
        "What are Scope 1, 2, and 3 emissions?",
        "Calculate emissions for 1000 kWh electricity in the US",
        "What are the best practices for reducing building emissions?",
        "Compare renewable energy sources with fossil fuels"
    ]
    
    print("\n2. Testing Queries:")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        
        try:
            result = assistant.process_query(query, use_rag=True)
            
            if result["success"]:
                print(f"Method: {result.get('method', 'Unknown')}")
                
                # Show first 300 chars of response
                response = result["response"]
                if len(response) > 300:
                    print(f"Response: {response[:300]}...")
                else:
                    print(f"Response: {response}")
                
                # Show sources if available
                if "sources" in result:
                    print(f"Sources: {', '.join(result['sources'])}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Exception: {str(e)}")
    
    # Test knowledge base search
    if assistant.rag_available:
        print("\n3. Testing Knowledge Base Search:")
        print("-" * 60)
        
        search_query = "carbon emissions electricity"
        print(f"\nSearching for: '{search_query}'")
        
        docs = assistant.search_knowledge_base(search_query, k=3)
        
        if docs:
            print(f"Found {len(docs)} relevant documents:")
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                content_preview = doc.page_content[:150] + "..."
                print(f"\n{i}. Source: {source}")
                print(f"   Content: {content_preview}")
        else:
            print("No documents found")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_rag_assistant()