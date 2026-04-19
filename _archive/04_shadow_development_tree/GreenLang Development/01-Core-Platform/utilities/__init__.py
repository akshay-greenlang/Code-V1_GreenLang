"""
GreenLang Utilities Layer

Consolidated module containing common utilities, I/O, serialization, determinism,
lineage, provenance, cache, tools, visualization, and more.

This module provides:
- Common utilities (from utils/)
- I/O operations (from io/, serialization/)
- Deterministic behavior (from determinism/)
- Data lineage tracking (from lineage/)
- Provenance tracking (from provenance/)
- Caching utilities (from cache/)
- Visualization tools (from visualization/, cards/)
- Factory patterns (from factory/)
- Code generation (from generator/)
- Compatibility layers (from compat/)
- Internationalization (from i18n/)
- Exception classes (from exceptions/)
- Miscellaneous tools (from tools/)

Sub-modules:
- utilities.utils: Environment, ML imports, networking
- utilities.io: Readers, formats, resources
- utilities.serialization: Canonical form, serializers
- utilities.determinism: Clock, decimal, files for deterministic behavior
- utilities.lineage: Column tracker, data lineage
- utilities.provenance: Decorators, environment tracking, hashing
- utilities.cache: Cache manager, emission factor cache
- utilities.visualization: Sankey diagrams, charts
- utilities.cards: Card generator, templates, validator
- utilities.factory: Agent factory, templates
- utilities.generator: Code generator, spec parser
- utilities.compat: Compatibility utilities
- utilities.i18n: Internationalization messages
- utilities.exceptions: Custom exception classes
- utilities.tools: Miscellaneous tools

Author: GreenLang Team
Version: 2.0.0
"""

__version__ = "2.0.0"

__all__ = []
