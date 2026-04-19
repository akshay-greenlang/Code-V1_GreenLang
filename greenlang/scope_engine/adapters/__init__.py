# -*- coding: utf-8 -*-
"""Framework adapters — project ScopeComputation into framework-native views.

Pure projection. No recomputation. If you find yourself reaching for factors or
GWP values in an adapter, something is wrong — that belongs upstream.
"""

from greenlang.scope_engine.adapters.base import FrameworkAdapter, register, get, available
from greenlang.scope_engine.adapters.cbam import CBAMAdapter
from greenlang.scope_engine.adapters.csrd_e1 import CSRDE1Adapter
from greenlang.scope_engine.adapters.ghg_protocol import GHGProtocolAdapter
from greenlang.scope_engine.adapters.iso_14064 import ISO14064Adapter
from greenlang.scope_engine.adapters.sbti import SBTiAdapter

# Register built-in adapters on import
register(GHGProtocolAdapter())
register(ISO14064Adapter())
register(SBTiAdapter())
register(CSRDE1Adapter())
register(CBAMAdapter())

__all__ = [
    "FrameworkAdapter",
    "register",
    "get",
    "available",
    "GHGProtocolAdapter",
    "ISO14064Adapter",
    "SBTiAdapter",
    "CSRDE1Adapter",
    "CBAMAdapter",
]
