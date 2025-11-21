# -*- coding: utf-8 -*-
"""
Example 6: RAG Q&A System
==========================

Demonstrates RAG (Retrieval-Augmented Generation) for knowledge-based Q&A.
"""

import asyncio
from greenlang.intelligence import ChatSession, ChatMessage, MessageRole
from greenlang.intelligence.rag import RAGEngine, Document, EmbeddingModel
from greenlang.intelligence.providers import get_provider, ProviderType


async def main():
    """Run RAG Q&A example."""
    # Initialize provider
    provider = get_provider(ProviderType.OPENAI)

    # Create RAG engine
    rag_engine = RAGEngine(
        embedding_model=EmbeddingModel(
            name="text-embedding-ada-002",
            provider=provider
        ),
        chunk_size=500
    )

    # Add knowledge base documents
    documents = [
        Document(
            id="doc1",
            content="Scope 1 emissions are direct GHG emissions from sources owned or controlled by the company, such as fuel combustion in boilers and vehicles.",
            metadata={"type": "definition"}
        ),
        Document(
            id="doc2",
            content="Scope 2 emissions are indirect GHG emissions from purchased electricity, heat, or steam consumed by the company.",
            metadata={"type": "definition"}
        ),
        Document(
            id="doc3",
            content="Scope 3 emissions are all other indirect emissions in the value chain, including purchased goods, business travel, and product use.",
            metadata={"type": "definition"}
        )
    ]

    for doc in documents:
        rag_engine.add_document(doc)

    print(f"Added {len(documents)} documents to knowledge base")

    # Query with RAG
    query = "What are Scope 2 emissions?"
    print(f"\nQuery: {query}")

    # Retrieve relevant documents
    retrieved_docs = await rag_engine.retrieve(query, top_k=2)

    print(f"\nRetrieved {len(retrieved_docs)} relevant documents:")
    for i, doc in enumerate(retrieved_docs):
        print(f"{i+1}. {doc.content[:100]}...")

    # Generate answer using ChatSession with context
    chat = ChatSession(provider=provider, model="gpt-3.5-turbo")

    context = "\n\n".join([doc.content for doc in retrieved_docs])
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content=f"Context:\n{context}\n\nQuestion: {query}\n\nProvide a clear answer based on the context."
        )
    ]

    response = await chat.chat(messages)

    print(f"\nAnswer: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
