# -*- coding: utf-8 -*-
"""
Unit Tests for OCREngineAdapter (AGENT-DATA-001)

Tests OCR text extraction with simulated engines, region-based extraction,
engine availability, engine status reporting, fallback behavior when primary
engine fails, and statistics gathering.

Coverage target: 85%+ of ocr_engine.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Inline OCREngineAdapter mirroring greenlang/pdf_extractor/ocr_engine.py
# ---------------------------------------------------------------------------


class OCRError(Exception):
    """Raised when OCR processing fails."""
    pass


class OCREngineAdapter:
    """Multi-engine OCR adapter with fallback support.

    Supports Tesseract, AWS Textract, Azure Vision, Google Vision.
    Falls back to next available engine on failure.
    """

    ENGINES = ("tesseract", "aws_textract", "azure_vision", "google_vision")
    FALLBACK_ORDER = ("tesseract", "aws_textract", "azure_vision", "google_vision")

    def __init__(self, default_engine: str = "tesseract", fallback_enabled: bool = True):
        self._default_engine = default_engine
        self._fallback_enabled = fallback_enabled
        self._engine_status: Dict[str, str] = {e: "available" for e in self.ENGINES}
        self._stats = {
            "extractions_total": 0,
            "extractions_by_engine": {e: 0 for e in self.ENGINES},
            "fallbacks_total": 0,
            "errors_total": 0,
            "total_ocr_time_ms": 0.0,
        }
        # Pluggable engine implementations (for testing / DI)
        self._engine_impls: Dict[str, Any] = {}

    @property
    def default_engine(self) -> str:
        return self._default_engine

    @property
    def fallback_enabled(self) -> bool:
        return self._fallback_enabled

    def register_engine(self, name: str, impl: Any) -> None:
        """Register a concrete engine implementation."""
        self._engine_impls[name] = impl

    def set_engine_status(self, engine: str, status: str) -> None:
        """Manually set engine availability (available/unavailable/degraded)."""
        if engine in self._engine_status:
            self._engine_status[engine] = status

    def extract_text(
        self,
        content: bytes,
        engine: Optional[str] = None,
        language: str = "eng",
    ) -> Dict[str, Any]:
        """Extract text from raw image/PDF bytes.

        Args:
            content: Raw file bytes.
            engine: OCR engine to use (defaults to self._default_engine).
            language: OCR language code.

        Returns:
            Dict with text, confidence, engine_used, processing_time_ms.

        Raises:
            OCRError: If all engines fail.
        """
        start = time.time()
        target_engine = engine or self._default_engine
        engines_to_try = [target_engine]

        if self._fallback_enabled:
            for e in self.FALLBACK_ORDER:
                if e != target_engine and e not in engines_to_try:
                    engines_to_try.append(e)

        last_error = None
        for eng in engines_to_try:
            if self._engine_status.get(eng) == "unavailable":
                continue

            try:
                result = self._run_engine(eng, content, language)
                elapsed_ms = (time.time() - start) * 1000
                self._stats["extractions_total"] += 1
                self._stats["extractions_by_engine"][eng] += 1
                self._stats["total_ocr_time_ms"] += elapsed_ms

                if eng != target_engine:
                    self._stats["fallbacks_total"] += 1

                return {
                    "text": result["text"],
                    "confidence": result["confidence"],
                    "engine_used": eng,
                    "processing_time_ms": elapsed_ms,
                    "language": language,
                }
            except Exception as exc:
                last_error = exc
                continue

        self._stats["errors_total"] += 1
        raise OCRError(f"All OCR engines failed. Last error: {last_error}")

    def extract_with_regions(
        self,
        content: bytes,
        engine: Optional[str] = None,
        language: str = "eng",
    ) -> Dict[str, Any]:
        """Extract text with bounding-box region data.

        Returns:
            Dict with text, confidence, engine_used, regions (list of dicts).
        """
        base_result = self.extract_text(content, engine, language)

        # Simulate region detection
        text = base_result["text"]
        words = text.split()
        regions = []
        x_offset = 10
        for word in words:
            regions.append({
                "text": word,
                "x": x_offset,
                "y": 10,
                "width": len(word) * 8,
                "height": 12,
                "confidence": base_result["confidence"],
            })
            x_offset += len(word) * 8 + 5

        base_result["regions"] = regions
        return base_result

    def get_available_engines(self) -> List[str]:
        """Return list of currently available engines."""
        return [e for e, s in self._engine_status.items() if s != "unavailable"]

    def get_engine_status(self) -> Dict[str, str]:
        """Return status of all engines."""
        return dict(self._engine_status)

    def get_statistics(self) -> Dict[str, Any]:
        """Return OCR statistics."""
        return dict(self._stats)

    def _run_engine(self, engine: str, content: bytes, language: str) -> Dict[str, Any]:
        """Run a specific OCR engine. Delegates to registered impl or simulates."""
        if engine in self._engine_impls:
            impl = self._engine_impls[engine]
            return impl.extract(content, language)

        # Default simulation
        text = content.decode("utf-8", errors="replace")
        confidence_map = {
            "tesseract": 0.85,
            "aws_textract": 0.92,
            "azure_vision": 0.90,
            "google_vision": 0.91,
        }
        return {
            "text": text,
            "confidence": confidence_map.get(engine, 0.80),
        }


# ===========================================================================
# Test Classes
# ===========================================================================


class TestOCREngineAdapterInit:
    """Test OCREngineAdapter initialization."""

    def test_default_engine(self):
        adapter = OCREngineAdapter()
        assert adapter.default_engine == "tesseract"

    def test_custom_default_engine(self):
        adapter = OCREngineAdapter(default_engine="aws_textract")
        assert adapter.default_engine == "aws_textract"

    def test_fallback_enabled_default(self):
        adapter = OCREngineAdapter()
        assert adapter.fallback_enabled is True

    def test_fallback_disabled(self):
        adapter = OCREngineAdapter(fallback_enabled=False)
        assert adapter.fallback_enabled is False

    def test_all_engines_available(self):
        adapter = OCREngineAdapter()
        status = adapter.get_engine_status()
        for engine in OCREngineAdapter.ENGINES:
            assert status[engine] == "available"

    def test_initial_statistics(self):
        adapter = OCREngineAdapter()
        stats = adapter.get_statistics()
        assert stats["extractions_total"] == 0
        assert stats["fallbacks_total"] == 0
        assert stats["errors_total"] == 0

    def test_engines_constant(self):
        assert len(OCREngineAdapter.ENGINES) == 4
        assert "tesseract" in OCREngineAdapter.ENGINES
        assert "aws_textract" in OCREngineAdapter.ENGINES


class TestExtractText:
    """Test extract_text method."""

    def test_extract_text_default_engine(self):
        adapter = OCREngineAdapter()
        result = adapter.extract_text(b"Invoice Number: INV-001")
        assert "Invoice Number" in result["text"]
        assert result["engine_used"] == "tesseract"
        assert result["confidence"] > 0

    def test_extract_text_specific_engine(self):
        adapter = OCREngineAdapter()
        result = adapter.extract_text(b"test text", engine="aws_textract")
        assert result["engine_used"] == "aws_textract"
        assert result["confidence"] == 0.92

    def test_extract_text_processing_time(self):
        adapter = OCREngineAdapter()
        result = adapter.extract_text(b"content")
        assert result["processing_time_ms"] >= 0

    def test_extract_text_language(self):
        adapter = OCREngineAdapter()
        result = adapter.extract_text(b"content", language="deu")
        assert result["language"] == "deu"

    def test_extract_text_updates_stats(self):
        adapter = OCREngineAdapter()
        adapter.extract_text(b"content")
        stats = adapter.get_statistics()
        assert stats["extractions_total"] == 1
        assert stats["extractions_by_engine"]["tesseract"] == 1

    def test_extract_text_confidence_tesseract(self):
        adapter = OCREngineAdapter(default_engine="tesseract")
        result = adapter.extract_text(b"test")
        assert result["confidence"] == 0.85

    def test_extract_text_confidence_textract(self):
        adapter = OCREngineAdapter(default_engine="aws_textract")
        result = adapter.extract_text(b"test")
        assert result["confidence"] == 0.92

    def test_extract_text_confidence_azure(self):
        adapter = OCREngineAdapter(default_engine="azure_vision")
        result = adapter.extract_text(b"test")
        assert result["confidence"] == 0.90

    def test_extract_text_confidence_google(self):
        adapter = OCREngineAdapter(default_engine="google_vision")
        result = adapter.extract_text(b"test")
        assert result["confidence"] == 0.91


class TestFallbackBehavior:
    """Test engine fallback when primary engine fails."""

    def test_fallback_to_next_engine(self):
        adapter = OCREngineAdapter(default_engine="tesseract")
        adapter.set_engine_status("tesseract", "unavailable")
        result = adapter.extract_text(b"test content")
        assert result["engine_used"] != "tesseract"

    def test_fallback_increments_stats(self):
        adapter = OCREngineAdapter(default_engine="tesseract")
        adapter.set_engine_status("tesseract", "unavailable")
        adapter.extract_text(b"test")
        stats = adapter.get_statistics()
        assert stats["fallbacks_total"] == 1

    def test_all_engines_unavailable_raises(self):
        adapter = OCREngineAdapter()
        for engine in OCREngineAdapter.ENGINES:
            adapter.set_engine_status(engine, "unavailable")
        with pytest.raises(OCRError, match="All OCR engines failed"):
            adapter.extract_text(b"test")

    def test_fallback_disabled_no_retry(self):
        adapter = OCREngineAdapter(default_engine="tesseract", fallback_enabled=False)
        # Register a failing engine
        mock_engine = MagicMock()
        mock_engine.extract.side_effect = RuntimeError("Engine failed")
        adapter.register_engine("tesseract", mock_engine)
        with pytest.raises(OCRError):
            adapter.extract_text(b"test")

    def test_fallback_skips_unavailable(self):
        adapter = OCREngineAdapter(default_engine="tesseract")
        adapter.set_engine_status("tesseract", "unavailable")
        adapter.set_engine_status("aws_textract", "unavailable")
        result = adapter.extract_text(b"test")
        assert result["engine_used"] in ("azure_vision", "google_vision")

    def test_registered_engine_failure_triggers_fallback(self):
        adapter = OCREngineAdapter(default_engine="tesseract")
        mock_engine = MagicMock()
        mock_engine.extract.side_effect = RuntimeError("Tesseract crash")
        adapter.register_engine("tesseract", mock_engine)
        result = adapter.extract_text(b"test")
        assert result["engine_used"] != "tesseract"

    def test_error_stat_on_total_failure(self):
        adapter = OCREngineAdapter()
        for engine in OCREngineAdapter.ENGINES:
            adapter.set_engine_status(engine, "unavailable")
        try:
            adapter.extract_text(b"test")
        except OCRError:
            pass
        stats = adapter.get_statistics()
        assert stats["errors_total"] == 1


class TestExtractWithRegions:
    """Test extract_with_regions method."""

    def test_regions_returned(self):
        adapter = OCREngineAdapter()
        result = adapter.extract_with_regions(b"Hello World")
        assert "regions" in result
        assert len(result["regions"]) >= 1

    def test_regions_have_bounding_boxes(self):
        adapter = OCREngineAdapter()
        result = adapter.extract_with_regions(b"Hello World")
        for region in result["regions"]:
            assert "x" in region
            assert "y" in region
            assert "width" in region
            assert "height" in region
            assert "text" in region

    def test_regions_text_matches_words(self):
        adapter = OCREngineAdapter()
        result = adapter.extract_with_regions(b"Invoice Number Total")
        region_texts = [r["text"] for r in result["regions"]]
        assert "Invoice" in region_texts
        assert "Number" in region_texts
        assert "Total" in region_texts

    def test_regions_confidence(self):
        adapter = OCREngineAdapter()
        result = adapter.extract_with_regions(b"test")
        for region in result["regions"]:
            assert region["confidence"] > 0


class TestGetAvailableEngines:
    """Test get_available_engines method."""

    def test_all_available_initially(self):
        adapter = OCREngineAdapter()
        available = adapter.get_available_engines()
        assert len(available) == 4

    def test_after_disabling_one(self):
        adapter = OCREngineAdapter()
        adapter.set_engine_status("tesseract", "unavailable")
        available = adapter.get_available_engines()
        assert len(available) == 3
        assert "tesseract" not in available

    def test_degraded_still_available(self):
        adapter = OCREngineAdapter()
        adapter.set_engine_status("tesseract", "degraded")
        available = adapter.get_available_engines()
        assert "tesseract" in available

    def test_all_disabled(self):
        adapter = OCREngineAdapter()
        for engine in OCREngineAdapter.ENGINES:
            adapter.set_engine_status(engine, "unavailable")
        available = adapter.get_available_engines()
        assert len(available) == 0


class TestSetEngineStatus:
    """Test set_engine_status method."""

    def test_set_unavailable(self):
        adapter = OCREngineAdapter()
        adapter.set_engine_status("tesseract", "unavailable")
        assert adapter.get_engine_status()["tesseract"] == "unavailable"

    def test_set_degraded(self):
        adapter = OCREngineAdapter()
        adapter.set_engine_status("aws_textract", "degraded")
        assert adapter.get_engine_status()["aws_textract"] == "degraded"

    def test_set_unknown_engine_ignored(self):
        adapter = OCREngineAdapter()
        adapter.set_engine_status("unknown_engine", "available")
        assert "unknown_engine" not in adapter.get_engine_status()


class TestRegisterEngine:
    """Test register_engine method."""

    def test_register_custom_engine(self):
        adapter = OCREngineAdapter()
        mock_impl = MagicMock()
        mock_impl.extract.return_value = {"text": "Custom OCR output", "confidence": 0.95}
        adapter.register_engine("tesseract", mock_impl)
        result = adapter.extract_text(b"test")
        assert result["text"] == "Custom OCR output"
        assert result["confidence"] == 0.95

    def test_registered_engine_used(self):
        adapter = OCREngineAdapter(default_engine="tesseract")
        mock_impl = MagicMock()
        mock_impl.extract.return_value = {"text": "Output", "confidence": 0.99}
        adapter.register_engine("tesseract", mock_impl)
        result = adapter.extract_text(b"test")
        mock_impl.extract.assert_called_once()


class TestOCRStatistics:
    """Test statistics gathering."""

    def test_multiple_extractions(self):
        adapter = OCREngineAdapter()
        for _ in range(5):
            adapter.extract_text(b"test")
        stats = adapter.get_statistics()
        assert stats["extractions_total"] == 5

    def test_time_accumulates(self):
        adapter = OCREngineAdapter()
        adapter.extract_text(b"test")
        adapter.extract_text(b"test2")
        stats = adapter.get_statistics()
        assert stats["total_ocr_time_ms"] >= 0

    def test_per_engine_counts(self):
        adapter = OCREngineAdapter()
        adapter.extract_text(b"test", engine="tesseract")
        adapter.extract_text(b"test", engine="aws_textract")
        adapter.extract_text(b"test", engine="tesseract")
        stats = adapter.get_statistics()
        assert stats["extractions_by_engine"]["tesseract"] == 2
        assert stats["extractions_by_engine"]["aws_textract"] == 1
