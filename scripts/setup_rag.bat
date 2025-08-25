@echo off
REM GreenLang RAG Setup Script for Windows

echo ========================================
echo GreenLang RAG Setup
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

echo Step 1: Installing RAG dependencies...
echo ----------------------------------------
pip install langchain langchain-community faiss-cpu sentence-transformers pypdf chromadb openai

if errorlevel 1 (
    echo Error: Failed to install dependencies
    exit /b 1
)

echo.
echo Step 2: Creating knowledge base directories...
echo ----------------------------------------
if not exist "knowledge_base" mkdir knowledge_base
if not exist "knowledge_base\documents" mkdir knowledge_base\documents
if not exist "knowledge_base\vector_store" mkdir knowledge_base\vector_store
if not exist "scripts" mkdir scripts

echo Directories created successfully

echo.
echo Step 3: Building initial vector store...
echo ----------------------------------------
python scripts\build_vector_store.py --add-defaults

if errorlevel 1 (
    echo Warning: Could not build vector store
    echo Please run manually: python scripts\build_vector_store.py --add-defaults
) else (
    echo Vector store built successfully
)

echo.
echo Step 4: Testing RAG implementation...
echo ----------------------------------------
python scripts\test_rag.py

if errorlevel 1 (
    echo Warning: RAG test failed
    echo Please check your installation
) else (
    echo RAG test passed successfully
)

echo.
echo ========================================
echo RAG Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Set your OpenAI API key in .env file:
echo    OPENAI_API_KEY=your-key-here
echo.
echo 2. Add documents to knowledge_base\documents\
echo.
echo 3. Rebuild vector store:
echo    python scripts\build_vector_store.py
echo.
echo 4. Test with GreenLang CLI:
echo    greenlang ask "What are carbon emissions?"
echo.

pause