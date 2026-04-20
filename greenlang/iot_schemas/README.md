# `greenlang.iot_schemas` — FY28 placeholder

**Status:** v0.0.1 placeholder. **Do not build FY27 products on top of this module.**

## Why this exists today

FY27 pilot profiles (CBAM importer, CSRD filer, SB 253 Scope 1+2) don't
ingest live IoT streams — they consume spreadsheets, ERP extracts, and
PDFs. Full IoT handling is an FY28 PlantOS / BuildingOS deliverable.

The placeholder module (this directory) reserves the import path
``greenlang.iot_schemas`` and declares the canonical shapes
(``ProtocolType``, ``CanonicalMeterReading``, ``CanonicalEventEnvelope``)
so FY28 work has a well-known home. That avoids a 3-way merge of IoT
schema locations across PlantOS, BuildingOS, and existing
``greenlang/agents/data/`` modules.

## What ships today

- `ProtocolType` — enum (`opc_ua`, `mqtt`, `modbus`, `bacnet`, `energy_star_portfolio`).
- `QualityFlag` — enum (`good`, `uncertain`, `bad`, `stale`).
- `CanonicalMeterReading` — Pydantic model for one observation.
- `CanonicalEventEnvelope` — Pydantic wrapper tagging the protocol.

## What's deliberately NOT here

- Protocol handlers (OPC-UA client, MQTT subscriber, Modbus poller).
- Equipment taxonomy / asset ontology.
- Stream processing, windowing, backfill.
- Time-series persistence (will use TimescaleDB hypertables already
  provisioned in INFRA-002).

All of the above land with FY28 PlantOS.

## Existing IoT-adjacent agents

These FY27 agents already touch IoT-shaped data and will migrate to
``greenlang.iot_schemas`` types in FY28 without changing their public
behaviour:

- `greenlang/agents/data/bms_connector_agent.py`
- `greenlang/agents/data/iot_meter_management_agent.py`
- `greenlang/agents/data/scada_connector_agent.py`
- `greenlang/agents/mrv/buildings/smart_building_mrv.py`
- `greenlang/agents/process_heat/gl_001_thermal_command/` and siblings

## FY28 acceleration triggers

Per `docs/fy27-scope.md` §2.3:

> If a design partner is explicitly conditioning a CBAM or SB 253 pilot
> on **live OPC-UA / SCADA ingestion** (not periodic CSV exports), flag
> it to the PM and revisit.

Three such requests with willingness-to-pay promotes the module to a
real FY27 line item.

## References

- `docs/fy27-scope.md` §2.3 — the deferral decision.
- `docs/REPO_TOUR.md` §2 — where this module sits in the v3 layer map.
- `GreenLang_Climate_OS_v3_Business_Plan_2026_2031.pdf` — FY28 PlantOS scope.
