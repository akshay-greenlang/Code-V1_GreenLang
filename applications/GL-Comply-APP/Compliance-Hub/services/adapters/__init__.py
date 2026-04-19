"""Framework adapters — one per compliance app.

Auto-registers all 10 adapters on import via services.registry.
"""

from services.adapters.cbam_adapter import CBAMAdapter
from services.adapters.cdp_adapter import CDPAdapter
from services.adapters.csrd_adapter import CSRDAdapter
from services.adapters.eudr_adapter import EUDRAdapter
from services.adapters.ghg_adapter import GHGProtocolAdapter
from services.adapters.iso14064_adapter import ISO14064Adapter
from services.adapters.sb253_adapter import SB253Adapter
from services.adapters.sbti_adapter import SBTiAdapter
from services.adapters.taxonomy_adapter import TaxonomyAdapter
from services.adapters.tcfd_adapter import TCFDAdapter
from services import registry


def register_all() -> None:
    for adapter_cls in (
        CSRDAdapter,
        CBAMAdapter,
        EUDRAdapter,
        GHGProtocolAdapter,
        ISO14064Adapter,
        SB253Adapter,
        SBTiAdapter,
        TaxonomyAdapter,
        TCFDAdapter,
        CDPAdapter,
    ):
        registry.register(adapter_cls())


register_all()

__all__ = [
    "CBAMAdapter",
    "CDPAdapter",
    "CSRDAdapter",
    "EUDRAdapter",
    "GHGProtocolAdapter",
    "ISO14064Adapter",
    "SB253Adapter",
    "SBTiAdapter",
    "TaxonomyAdapter",
    "TCFDAdapter",
    "register_all",
]
