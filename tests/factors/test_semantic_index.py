# -*- coding: utf-8 -*-
"""Tests for :class:`greenlang.factors.matching.semantic_index.SemanticIndex`.

Focus is the in-memory NumPy fallback: pgvector is exercised in
``test_pgvector_index.py`` and skipped here because it needs a live
Postgres instance.
"""

from __future__ import annotations

import pytest

from greenlang.factors.matching.semantic_index import (
    NoopSemanticIndex,
    SemanticIndex,
    get_default_semantic_index,
    get_embedder,
)


# ----------------------------------------------------------------------------
# Synthetic dataset
# ----------------------------------------------------------------------------


# 100 factor-like text snippets across a handful of disjoint topics.
# We score retrieval by checking that the top-k for each topic query is
# dominated by the expected topic.
_TOPICS = {
    "electricity_grid": [
        "purchased electricity grid mix",
        "country average grid intensity electricity",
        "electricity consumption from utility provider",
        "kWh of grid electricity purchased",
        "scope 2 location-based electricity emissions",
        "renewable energy attribute certificates electricity",
        "delivered electricity emission factor kWh",
        "electricity power consumption residential",
        "industrial electricity usage emissions",
        "average grid mix kWh emission factor",
    ],
    "diesel_combustion": [
        "diesel fuel combustion stationary",
        "diesel oil burning emission factor",
        "stationary combustion of diesel",
        "diesel generator fuel combustion",
        "off-road diesel combustion",
        "industrial diesel combustion emissions",
        "diesel fuel oxidation factor",
        "burning diesel for heat",
        "diesel boiler combustion factor",
        "litres diesel combusted",
    ],
    "natural_gas_combustion": [
        "natural gas combustion stationary",
        "natural gas burning emission factor",
        "burning natural gas in boiler",
        "stationary combustion of natural gas",
        "industrial natural gas use",
        "domestic natural gas combustion",
        "natural gas fired furnace emissions",
        "therms natural gas combustion factor",
        "natural gas heating combustion",
        "scope 1 natural gas combustion",
    ],
    "freight_road": [
        "road freight truck transportation",
        "heavy goods vehicle freight road",
        "trucking freight emissions per tkm",
        "road freight 40 tonne articulated",
        "long haul truck freight emissions",
        "diesel truck freight road transport",
        "rigid truck freight road transport",
        "WTW truck freight road emission factor",
        "TTW truck freight road emission factor",
        "ISO 14083 road freight emissions",
    ],
    "refrigerant_leak": [
        "refrigerant leak R-410A HFC",
        "fugitive refrigerant emissions HFC blends",
        "refrigerant top-up leakage",
        "R-134a refrigerant leak emissions",
        "AR6 GWP refrigerant leakage HFC",
        "fugitive HFC refrigerant leak",
        "refrigeration loop leak rate",
        "R-32 refrigerant leak emissions",
        "supermarket refrigerant leak factor",
        "centralised AC refrigerant leak",
    ],
    "purchased_steel": [
        "embodied carbon purchased steel",
        "scope 3 cat1 purchased steel embodied",
        "steel material LCA cradle to gate",
        "structural steel embodied emissions",
        "purchased rolled steel embodied",
        "carbon steel purchased product",
        "steel ingot embodied emission factor",
        "EPD steel product embodied carbon",
        "EAF steel embodied emissions",
        "BF-BOF steel embodied emission factor",
    ],
    "purchased_cement": [
        "embodied carbon purchased cement",
        "Portland cement embodied emissions",
        "cement clinker embodied emission factor",
        "cement product LCA cradle to gate",
        "ready mix concrete cement embodied",
        "purchased cement product embodied",
        "ordinary Portland cement emissions",
        "blended cement embodied carbon",
        "cement supplementary cementitious materials",
        "cement industry embodied emissions",
    ],
    "land_use_change": [
        "land use change deforestation removal",
        "forestland conversion CO2 removal",
        "soil carbon land use change emissions",
        "removals from afforestation projects",
        "deforestation land use change emission",
        "managed forestland carbon stock change",
        "biomass land use change removals",
        "agroforestry land removals",
        "savanna land use change emissions",
        "wetland land use carbon flux",
    ],
    "spend_proxy": [
        "EEIO spend based emission intensity",
        "scope 3 spend proxy procurement category",
        "input output table emissions spend USD",
        "spend based emission factor procurement",
        "monetary unit emission factor spend",
        "supply chain spend proxy intensity",
        "spend based scope 3 procurement",
        "EXIOBASE spend proxy intensity",
        "USEEIO spend proxy emissions",
        "category 1 spend proxy procurement",
    ],
    "wastewater_treatment": [
        "wastewater treatment activated sludge emissions",
        "anaerobic digestion wastewater methane",
        "wastewater treatment facility emissions",
        "industrial wastewater treatment N2O",
        "domestic wastewater treatment emissions",
        "wastewater sludge incineration emissions",
        "BOD load wastewater treatment factor",
        "COD load wastewater treatment factor",
        "MBR wastewater treatment emissions",
        "wastewater nitrification N2O emission",
    ],
}


def _make_index_with_dataset(
    *, edition: str = "test-edition", use_stub: bool = False,
) -> SemanticIndex:
    """Build a SemanticIndex with the 100-factor synthetic dataset.

    By default uses the real sentence-transformers embedder so the
    in-memory fallback's retrieval quality can be exercised.  Pass
    ``use_stub=True`` for fast, deterministic basic tests.
    """
    embedder = get_embedder(use_stub=use_stub)
    index = SemanticIndex(
        embedder=embedder,
        pg_dsn="",  # force in-memory backend
        default_edition_id=edition,
    )
    counter = 0
    for topic, texts in _TOPICS.items():
        for i, text in enumerate(texts):
            counter += 1
            index.upsert(
                factor_id=f"{topic}-{i:02d}",
                text=text,
                metadata={"topic": topic},
            )
    assert counter == 100
    return index


# ----------------------------------------------------------------------------
# Smoke
# ----------------------------------------------------------------------------


class TestSemanticIndexBasics:

    def test_construct_without_pg_uses_memory(self):
        idx = SemanticIndex(pg_dsn="")
        h = idx.health()
        assert h["pg_backend"]["configured"] is False
        assert h["pg_backend"]["connected"] is False
        assert h["memory_records"] == 0

    def test_get_default_singleton_returns_same_instance(self):
        a = get_default_semantic_index()
        b = get_default_semantic_index()
        assert a is b

    def test_upsert_then_search_roundtrip(self):
        idx = SemanticIndex(embedder=get_embedder(use_stub=True), pg_dsn="")
        ok = idx.upsert("ef-1", "diesel combustion stationary")
        assert ok is True
        results = idx.search("diesel combustion", k=5)
        assert results
        assert results[0]["factor_id"] == "ef-1"
        # Cosine similarity is bounded by [-1, 1].
        assert -1.0 <= results[0]["similarity"] <= 1.0
        assert results[0]["distance"] == pytest.approx(1.0 - results[0]["similarity"])

    def test_upsert_with_empty_factor_id_raises(self):
        idx = SemanticIndex(embedder=get_embedder(use_stub=True), pg_dsn="")
        with pytest.raises(ValueError):
            idx.upsert("", "some text")

    def test_search_empty_query_returns_empty(self):
        idx = SemanticIndex(embedder=get_embedder(use_stub=True), pg_dsn="")
        idx.upsert("ef-1", "diesel combustion")
        assert idx.search("", k=5) == []

    def test_delete_removes_record(self):
        idx = SemanticIndex(embedder=get_embedder(use_stub=True), pg_dsn="")
        idx.upsert("ef-1", "diesel combustion stationary")
        assert idx.delete("ef-1") is True
        assert idx.delete("ef-1") is False  # second delete is a no-op


# ----------------------------------------------------------------------------
# Top-k quality on the 100-factor synthetic dataset
# ----------------------------------------------------------------------------


class TestSemanticIndexTopKQuality:
    """The in-memory fallback must return relevant matches above noise.

    Uses the real sentence-transformers embedder to exercise retrieval
    quality.  Skips automatically when the model is unavailable so
    minimal CI environments still pass.
    """

    @pytest.fixture(scope="class")
    def index(self) -> SemanticIndex:
        st = pytest.importorskip("sentence_transformers")
        del st  # silence unused warning
        return _make_index_with_dataset(use_stub=False)

    @pytest.mark.parametrize("query, expected_topic, min_topic_in_top5", [
        ("electricity grid average kWh", "electricity_grid", 2),
        ("diesel combustion factor", "diesel_combustion", 2),
        ("natural gas combustion stationary", "natural_gas_combustion", 2),
        ("road freight truck emissions", "freight_road", 2),
        ("refrigerant leak HFC", "refrigerant_leak", 2),
        ("embodied carbon steel", "purchased_steel", 2),
        ("embodied carbon cement", "purchased_cement", 2),
        ("land use change deforestation", "land_use_change", 2),
        ("spend proxy EEIO procurement", "spend_proxy", 2),
        ("wastewater treatment emissions", "wastewater_treatment", 2),
    ])
    def test_topk_dominated_by_expected_topic(
        self, index, query, expected_topic, min_topic_in_top5,
    ):
        results = index.search(query, k=5)
        assert results, f"no results for query={query!r}"
        topics = [r["metadata"]["topic"] for r in results]
        same_topic = sum(1 for t in topics if t == expected_topic)
        assert same_topic >= min_topic_in_top5, (
            f"query={query!r} expected >= {min_topic_in_top5} {expected_topic!r} "
            f"in top-5, got {same_topic} (topics={topics})"
        )

    def test_index_size_is_100(self, index):
        h = index.health()
        assert h["memory_records"] == 100

    def test_filter_restricts_results_to_one_topic(self, index):
        results = index.search(
            "any string here",
            k=10,
            filters={"topic": "freight_road"},
        )
        assert results
        assert all(r["metadata"]["topic"] == "freight_road" for r in results)


class TestSemanticIndexInMemoryFallbackWithStub:
    """Smoke test: the in-memory fallback can run on the deterministic
    stub embedder too (no sentence-transformers needed), even though
    relevance won't be semantically meaningful."""

    def test_stub_dataset_returns_self_as_top1(self):
        index = _make_index_with_dataset(
            edition="stub-edition", use_stub=True,
        )
        assert index.health()["memory_records"] == 100
        # Querying with the exact text of an indexed factor must return
        # that factor in the top-1 slot (stub is deterministic per text).
        results = index.search("diesel fuel combustion stationary", k=3)
        assert results
        assert results[0]["factor_id"] == "diesel_combustion-00"


# ----------------------------------------------------------------------------
# Backward-compat: NoopSemanticIndex still importable
# ----------------------------------------------------------------------------


class TestBackwardCompat:

    def test_noop_index_returns_empty_for_search(self):
        noop = NoopSemanticIndex()
        assert noop.embed_text("anything") == []
        assert noop.search("ed", [0.0, 0.1], 5) == []
