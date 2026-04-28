# -*- coding: utf-8 -*-
"""GreenLang Factors v0.1 Alpha — ontology seed data and loaders.

This package holds canonical-reference seed YAML files and the loaders
that idempotently insert their rows into the ``factors_v0_1`` schema:

* ``activity_seed_v0_1.yaml``     - Phase 2 WS5 (this submodule's owner)
* ``geography_seed_v0_1.yaml``    - Phase 2 WS3
* ``unit_seed_v0_1.yaml``         - Phase 2 WS4
* ``methodology_seed_v0_1.yaml``  - Phase 2 WS6

Loaders live under ``loaders/``. Every loader is idempotent
(``ON CONFLICT (urn) DO NOTHING``) and validates each URN through
:func:`greenlang.factors.ontology.urn.parse` before insertion.
"""
