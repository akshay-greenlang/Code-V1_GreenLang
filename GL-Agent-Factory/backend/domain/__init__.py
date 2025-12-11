# -*- coding: utf-8 -*-
"""
GreenLang Domain Module
=======================

Domain-specific knowledge, ontologies, and standards for
industrial process heat decarbonization.

Submodules:
- ontology: OWL/RDF-based domain ontology
- formulas: Thermodynamic and engineering formulas
- schemas: Data schemas and validation
- standards_corpus: Regulatory standards index
- knowledge_graph: Knowledge graph services
- multimodal: Multi-modal data pipeline
- rag: Enhanced retrieval-augmented generation
- templates: Agent templates library
"""

from . import ontology
from . import standards_corpus

__all__ = [
    "ontology",
    "standards_corpus",
]

__version__ = "1.0.0"
