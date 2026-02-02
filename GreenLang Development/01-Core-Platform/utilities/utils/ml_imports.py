# -*- coding: utf-8 -*-
"""
Runtime import checks for optional ML dependencies.

This module provides helper functions to check for and import optional ML dependencies,
providing clear error messages with installation instructions when dependencies are missing.

Author: GreenLang AI
"""

from typing import Any, Optional, Callable
import importlib
import sys


class MissingDependencyError(ImportError):
    """Raised when an optional ML dependency is missing."""

    def __init__(self, package_name: str, extras: str, feature: str = "this feature"):
        self.package_name = package_name
        self.extras = extras
        self.feature = feature

        message = (
            f"\n{'=' * 80}\n"
            f"Missing Optional Dependency: {package_name}\n"
            f"{'=' * 80}\n\n"
            f"{feature.capitalize()} requires the '{package_name}' package.\n\n"
            f"To install it, run:\n"
            f"  pip install greenlang-cli[{extras}]\n\n"
            f"Or install all AI capabilities:\n"
            f"  pip install greenlang-cli[ai-full]\n"
            f"{'=' * 80}\n"
        )
        super().__init__(message)


def check_ml_dependencies(feature_name: str = "ML features") -> None:
    """
    Check if ML dependencies are installed.

    Args:
        feature_name: Name of the feature requiring ML dependencies

    Raises:
        MissingDependencyError: If ML dependencies are not installed
    """
    try:
        import torch
    except ImportError:
        raise MissingDependencyError(
            package_name="torch",
            extras="ml",
            feature=feature_name
        )

    try:
        import sentence_transformers
    except ImportError:
        raise MissingDependencyError(
            package_name="sentence-transformers",
            extras="ml",
            feature=feature_name
        )


def check_vector_db_dependencies(
    vector_db: str,
    feature_name: str = "vector database features"
) -> None:
    """
    Check if vector database dependencies are installed.

    Args:
        vector_db: Name of the vector database ('weaviate', 'chromadb', 'pinecone', 'qdrant', 'faiss')
        feature_name: Name of the feature requiring vector DB

    Raises:
        MissingDependencyError: If vector DB dependencies are not installed
    """
    package_map = {
        'weaviate': 'weaviate-client',
        'chromadb': 'chromadb',
        'pinecone': 'pinecone-client',
        'qdrant': 'qdrant-client',
        'faiss': 'faiss-cpu',
    }

    package_name = package_map.get(vector_db.lower())
    if not package_name:
        raise ValueError(f"Unknown vector database: {vector_db}")

    # Convert package name to module name
    module_map = {
        'weaviate-client': 'weaviate',
        'chromadb': 'chromadb',
        'pinecone-client': 'pinecone',
        'qdrant-client': 'qdrant_client',
        'faiss-cpu': 'faiss',
    }

    module_name = module_map.get(package_name, package_name.replace('-', '_'))

    try:
        importlib.import_module(module_name)
    except ImportError:
        raise MissingDependencyError(
            package_name=package_name,
            extras="vector-db",
            feature=feature_name
        )


def check_transformers_dependencies(feature_name: str = "transformer model features") -> None:
    """
    Check if transformers library is installed.

    Args:
        feature_name: Name of the feature requiring transformers

    Raises:
        MissingDependencyError: If transformers is not installed
    """
    try:
        import transformers
    except ImportError:
        raise MissingDependencyError(
            package_name="transformers",
            extras="ml",
            feature=feature_name
        )


def lazy_import(
    module_name: str,
    package_name: Optional[str] = None,
    extras: str = "ml",
    feature_name: str = "this feature"
) -> Any:
    """
    Lazily import a module with helpful error message if missing.

    Args:
        module_name: Name of the module to import (e.g., 'torch', 'weaviate')
        package_name: Name of the pip package if different from module (e.g., 'weaviate-client')
        extras: The extras group to install (e.g., 'ml', 'vector-db')
        feature_name: Name of the feature requiring this import

    Returns:
        The imported module

    Raises:
        MissingDependencyError: If the module cannot be imported

    Example:
        torch = lazy_import('torch', extras='ml', feature_name='entity resolution')
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        pkg = package_name or module_name
        raise MissingDependencyError(
            package_name=pkg,
            extras=extras,
            feature=feature_name
        )


def requires_ml(func: Callable) -> Callable:
    """
    Decorator to mark functions that require ML dependencies.

    Example:
        @requires_ml
        def embed_text(text: str):
            import torch
            # ... ML code
    """
    def wrapper(*args, **kwargs):
        check_ml_dependencies(feature_name=f"{func.__name__}")
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def requires_vector_db(vector_db: str) -> Callable:
    """
    Decorator to mark functions that require vector database dependencies.

    Args:
        vector_db: Name of required vector DB ('weaviate', 'chromadb', etc.)

    Example:
        @requires_vector_db('weaviate')
        def store_embeddings(embeddings):
            # ... vector DB code
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            check_vector_db_dependencies(
                vector_db=vector_db,
                feature_name=f"{func.__name__}"
            )
            return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator


def is_ml_available() -> bool:
    """
    Check if ML dependencies are available without raising an error.

    Returns:
        True if ML dependencies are available, False otherwise
    """
    try:
        import torch
        import sentence_transformers
        return True
    except ImportError:
        return False


def is_vector_db_available(vector_db: str) -> bool:
    """
    Check if a vector database is available without raising an error.

    Args:
        vector_db: Name of vector DB ('weaviate', 'chromadb', 'pinecone', 'qdrant', 'faiss')

    Returns:
        True if vector DB is available, False otherwise
    """
    try:
        check_vector_db_dependencies(vector_db)
        return True
    except (ImportError, MissingDependencyError):
        return False


def get_available_ml_features() -> dict:
    """
    Get information about which ML features are available.

    Returns:
        Dictionary with availability status of ML components
    """
    features = {}

    # Check ML dependencies
    try:
        import torch
        features['torch'] = {
            'available': True,
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }
    except ImportError:
        features['torch'] = {'available': False}

    try:
        import sentence_transformers
        features['sentence_transformers'] = {
            'available': True,
            'version': sentence_transformers.__version__
        }
    except ImportError:
        features['sentence_transformers'] = {'available': False}

    try:
        import transformers
        features['transformers'] = {
            'available': True,
            'version': transformers.__version__
        }
    except ImportError:
        features['transformers'] = {'available': False}

    # Check vector databases
    vector_dbs = ['weaviate', 'chromadb', 'pinecone', 'qdrant', 'faiss']
    for vdb in vector_dbs:
        features[vdb] = {'available': is_vector_db_available(vdb)}

    return features


if __name__ == "__main__":
    # Print available ML features
    print("GreenLang ML Feature Availability")
    print("=" * 80)

    features = get_available_ml_features()

    for name, info in features.items():
        status = "✓ Available" if info['available'] else "✗ Not installed"
        version = f" (v{info['version']})" if 'version' in info else ""
        print(f"{name:25} {status}{version}")

    print("\n" + "=" * 80)
    print("\nTo install ML features:")
    print("  pip install greenlang-cli[ml]")
    print("\nTo install vector databases:")
    print("  pip install greenlang-cli[vector-db]")
    print("\nTo install all AI features:")
    print("  pip install greenlang-cli[ai-full]")
