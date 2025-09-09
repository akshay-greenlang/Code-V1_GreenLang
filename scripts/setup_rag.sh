#!/bin/bash
# GreenLang RAG Setup Script for Linux/Mac

echo "========================================"
echo "GreenLang RAG Setup"
echo "========================================"
echo

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed"
    exit 1
fi

echo "Step 1: Installing RAG dependencies..."
echo "----------------------------------------"
pip3 install langchain langchain-community faiss-cpu sentence-transformers pypdf chromadb openai

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo
echo "Step 2: Creating knowledge base directories..."
echo "----------------------------------------"
mkdir -p knowledge_base/documents
mkdir -p knowledge_base/vector_store
mkdir -p scripts

echo "Directories created successfully"

echo
echo "Step 3: Building initial vector store..."
echo "----------------------------------------"
python3 scripts/build_vector_store.py --add-defaults

if [ $? -ne 0 ]; then
    echo "Warning: Could not build vector store"
    echo "Please run manually: python3 scripts/build_vector_store.py --add-defaults"
else
    echo "Vector store built successfully"
fi

echo
echo "Step 4: Testing RAG implementation..."
echo "----------------------------------------"
python3 scripts/test_rag.py

if [ $? -ne 0 ]; then
    echo "Warning: RAG test failed"
    echo "Please check your installation"
else
    echo "RAG test passed successfully"
fi

echo
echo "========================================"
echo "RAG Setup Complete!"
echo "========================================"
echo
echo "Next steps:"
echo "1. Set your OpenAI API key in .env file:"
echo "   OPENAI_API_KEY=your-key-here"
echo
echo "2. Add documents to knowledge_base/documents/"
echo
echo "3. Rebuild vector store:"
echo "   python3 scripts/build_vector_store.py"
echo
echo "4. Test with GreenLang CLI:"
echo "   gl ask 'What are carbon emissions?'"
echo