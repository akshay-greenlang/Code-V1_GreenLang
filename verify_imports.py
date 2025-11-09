"""Quick import verification script"""
import sys
from pathlib import Path

# Add greenlang to path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing infrastructure imports...\n")

# Test 1: Intelligence
try:
    from greenlang.intelligence import ChatSession, create_provider
    print("✓ Intelligence: ChatSession, create_provider")
except Exception as e:
    print(f"✗ Intelligence: {e}")

# Test 2: SDK Base
try:
    from greenlang.sdk.base import Agent, Pipeline, Result, Metadata
    print("✓ SDK Base: Agent, Pipeline, Result, Metadata")
except Exception as e:
    print(f"✗ SDK Base: {e}")

# Test 3: Cache
try:
    from greenlang.cache import CacheManager, L1MemoryCache, L2RedisCache, L3DiskCache
    print("✓ Cache: CacheManager, L1MemoryCache, L2RedisCache, L3DiskCache")
except Exception as e:
    print(f"✗ Cache: {e}")

# Test 4: Validation
try:
    from greenlang.validation import ValidationFramework, SchemaValidator, RulesEngine
    print("✓ Validation: ValidationFramework, SchemaValidator, RulesEngine")
except Exception as e:
    print(f"✗ Validation: {e}")

# Test 5: Telemetry
try:
    from greenlang.telemetry import MetricsCollector, TracingManager, StructuredLogger
    print("✓ Telemetry: MetricsCollector, TracingManager, StructuredLogger")
except Exception as e:
    print(f"✗ Telemetry: {e}")

# Test 6: DB
try:
    from greenlang.db import Base, get_engine, get_session, DatabaseConnectionPool
    print("✓ DB: Base, get_engine, get_session, DatabaseConnectionPool")
except Exception as e:
    print(f"✗ DB: {e}")

# Test 7: Auth
try:
    from greenlang.auth import AuthManager, RBACManager, TenantManager
    print("✓ Auth: AuthManager, RBACManager, TenantManager")
except Exception as e:
    print(f"✗ Auth: {e}")

# Test 8: Config
try:
    from greenlang.config import ConfigManager, ServiceContainer, get_config
    print("✓ Config: ConfigManager, ServiceContainer, get_config")
except Exception as e:
    print(f"✗ Config: {e}")

# Test 9: Provenance
try:
    from greenlang.provenance import ProvenanceTracker, ProvenanceRecord
    print("✓ Provenance: ProvenanceTracker, ProvenanceRecord")
except Exception as e:
    print(f"✗ Provenance: {e}")

# Test 10: Services
try:
    from greenlang.services import FactorBroker, EntityResolver
    print("✓ Services: FactorBroker, EntityResolver")
except Exception as e:
    print(f"✗ Services: {e}")

# Test 11: Agent Templates
try:
    from greenlang.agents.templates import IntakeAgent, CalculatorAgent, ReportingAgent
    print("✓ Agent Templates: IntakeAgent, CalculatorAgent, ReportingAgent")
except Exception as e:
    print(f"✗ Agent Templates: {e}")

# Test 12: RAG
try:
    from greenlang.intelligence.rag import RAGEngine, Chunker, EmbeddingProvider
    print("✓ RAG: RAGEngine, Chunker, EmbeddingProvider")
except Exception as e:
    print(f"✗ RAG: {e}")

print("\nAll import tests completed!")
