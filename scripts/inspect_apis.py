"""Inspect EUDR Supply Chain Mapper API signatures."""
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

import inspect

# 1. Graph Engine
from greenlang.agents.eudr.supply_chain_mapper.graph_engine import SupplyChainGraphEngine
print("=== SupplyChainGraphEngine ===")
for name, method in sorted(inspect.getmembers(SupplyChainGraphEngine, predicate=inspect.isfunction)):
    if not name.startswith('_'):
        is_async = inspect.iscoroutinefunction(method)
        sig = inspect.signature(method)
        label = "ASYNC" if is_async else "SYNC "
        print(f"  {label} {name}{sig}")

# 2. ProvenanceTracker
from greenlang.agents.eudr.supply_chain_mapper.provenance import ProvenanceTracker
print("\n=== ProvenanceTracker ===")
t = ProvenanceTracker()
print(f"  Instance attrs: {[a for a in dir(t) if not a.startswith('__')]}")
for name, method in sorted(inspect.getmembers(ProvenanceTracker, predicate=inspect.isfunction)):
    if not name.startswith('__'):
        sig = inspect.signature(method)
        print(f"  {name}{sig}")

# 3. RiskPropagationEngine
from greenlang.agents.eudr.supply_chain_mapper.risk_propagation import RiskPropagationEngine
print("\n=== RiskPropagationEngine ===")
for name, method in sorted(inspect.getmembers(RiskPropagationEngine, predicate=inspect.isfunction)):
    if not name.startswith('_'):
        sig = inspect.signature(method)
        print(f"  {name}{sig}")

# 4. VisualizationEngine
from greenlang.agents.eudr.supply_chain_mapper.visualization_engine import VisualizationEngine
print("\n=== VisualizationEngine ===")
for name, method in sorted(inspect.getmembers(VisualizationEngine, predicate=inspect.isfunction)):
    if not name.startswith('_'):
        sig = inspect.signature(method)
        print(f"  {name}{sig}")

# 5. GapAnalyzer
from greenlang.agents.eudr.supply_chain_mapper.gap_analyzer import GapAnalyzer
print("\n=== GapAnalyzer ===")
for name, method in sorted(inspect.getmembers(GapAnalyzer, predicate=inspect.isfunction)):
    if not name.startswith('_'):
        sig = inspect.signature(method)
        print(f"  {name}{sig}")
