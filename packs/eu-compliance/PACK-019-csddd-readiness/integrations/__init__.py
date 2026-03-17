# -*- coding: utf-8 -*-
"""
PACK-019 CSDDD Readiness Pack - Integrations
==============================================

Integration bridges connecting the CSDDD Readiness Pack to other GreenLang
platform components including CSRD/ESRS packs, MRV agents, EUDR agents,
supply chain agents, data agents, EU Taxonomy, and Green Claims Directive.

Modules:
    pack_orchestrator    -- CSDDDOrchestrator: Master 7-phase assessment pipeline
    csrd_pack_bridge     -- CSRDPackBridge: ESRS S1-S4/G1 to CSDDD mapping
    mrv_bridge           -- MRVBridge: MRV emission data for climate transition
    eudr_bridge          -- EUDRBridge: EUDR deforestation impact integration
    supply_chain_bridge  -- SupplyChainBridge: Value chain due diligence
    data_bridge          -- DataBridge: AGENT-DATA intake routing
    green_claims_bridge  -- GreenClaimsBridge: Green Claims cross-validation
    taxonomy_bridge      -- TaxonomyBridge: EU Taxonomy DNSH alignment
    health_check         -- CSDDDHealthCheck: System health verification
    setup_wizard         -- CSDDDSetupWizard: Guided configuration setup
"""
