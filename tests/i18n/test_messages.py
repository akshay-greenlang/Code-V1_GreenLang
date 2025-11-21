# -*- coding: utf-8 -*-
"""
tests/i18n/test_messages.py

Internationalization (i18n) Tests for FuelAgentAI v2

Tests:
- Message translation (8 languages)
- Fallback to English
- Recommendation translation
- Supported languages

Author: GreenLang Framework Team
Date: October 2025
"""

import pytest
from greenlang.i18n.messages import (
    I18n,
    get_translator,
    get_supported_languages,
    MESSAGES,
    SUPPORTED_LANGUAGES,
)


class TestBasicTranslation:
    """Test basic message translation."""

    def test_english_translation(self):
        """Test English translation (default)."""
        i18n = I18n("en")
        msg = i18n.get("calculation_complete")
        assert msg == "Emission calculation completed successfully"

    def test_spanish_translation(self):
        """Test Spanish translation."""
        i18n = I18n("es")
        msg = i18n.get("calculation_complete")
        assert msg == "Cálculo de emissiones completado exitosamente"

    def test_french_translation(self):
        """Test French translation."""
        i18n = I18n("fr")
        msg = i18n.get("calculation_complete")
        assert msg == "Calcul des émissions terminé avec succès"

    def test_german_translation(self):
        """Test German translation."""
        i18n = I18n("de")
        msg = i18n.get("calculation_complete")
        assert msg == "Emissionsberechnung erfolgreich abgeschlossen"

    def test_chinese_translation(self):
        """Test Chinese Simplified translation."""
        i18n = I18n("zh")
        msg = i18n.get("calculation_complete")
        assert msg == "排放计算成功完成"

    def test_japanese_translation(self):
        """Test Japanese translation."""
        i18n = I18n("ja")
        msg = i18n.get("calculation_complete")
        assert msg == "排出量計算が正常に完了しました"

    def test_portuguese_translation(self):
        """Test Portuguese translation."""
        i18n = I18n("pt")
        msg = i18n.get("calculation_complete")
        assert msg == "Cálculo de emissões concluído com sucesso"

    def test_hindi_translation(self):
        """Test Hindi translation."""
        i18n = I18n("hi")
        msg = i18n.get("calculation_complete")
        assert msg == "उत्सर्जन गणना सफलतापूर्वक पूर्ण हुई"


class TestFallbackMechanism:
    """Test fallback to English for unsupported languages."""

    def test_unsupported_language_fallback(self):
        """Test fallback to English for unsupported language."""
        i18n = I18n("unsupported")
        msg = i18n.get("calculation_complete")

        # Should fallback to English
        assert msg == "Emission calculation completed successfully"
        assert i18n.language == "en"

    def test_missing_key_fallback(self):
        """Test fallback to English for missing key."""
        i18n = I18n("es")
        msg = i18n.get("nonexistent_key")

        # Should return the key itself as fallback
        assert msg == "nonexistent_key"


class TestAllMessages:
    """Test all message keys across all languages."""

    def test_all_languages_have_same_keys(self):
        """Test that all languages have the same message keys."""
        en_keys = set(MESSAGES["en"].keys())

        for lang_code, translations in MESSAGES.items():
            lang_keys = set(translations.keys())

            # Check for missing keys
            missing_keys = en_keys - lang_keys
            extra_keys = lang_keys - en_keys

            assert len(missing_keys) == 0, \
                f"Language {lang_code} is missing keys: {missing_keys}"
            assert len(extra_keys) == 0, \
                f"Language {lang_code} has extra keys: {extra_keys}"

    def test_all_messages_non_empty(self):
        """Test that all messages are non-empty strings."""
        for lang_code, translations in MESSAGES.items():
            for key, message in translations.items():
                assert isinstance(message, str), \
                    f"Message {lang_code}.{key} should be a string"
                assert len(message) > 0, \
                    f"Message {lang_code}.{key} should not be empty"


class TestTechnicalTerms:
    """Test translation of technical terms."""

    def test_scope_translation(self):
        """Test translation of 'scope' across languages."""
        translations = {
            "en": "Scope",
            "es": "Alcance",
            "fr": "Portée",
            "de": "Umfang",
            "zh": "范围",
            "ja": "スコープ",
            "pt": "Escopo",
            "hi": "दायरा",
        }

        for lang, expected in translations.items():
            i18n = I18n(lang)
            msg = i18n.get("scope")
            assert msg == expected, f"Scope translation for {lang} incorrect"

    def test_boundary_translation(self):
        """Test translation of 'boundary' across languages."""
        translations = {
            "en": "Boundary",
            "es": "Límite",
            "fr": "Limite",
            "de": "Grenze",
            "zh": "边界",
            "ja": "境界",
            "pt": "Limite",
            "hi": "सीमा",
        }

        for lang, expected in translations.items():
            i18n = I18n(lang)
            msg = i18n.get("boundary")
            assert msg == expected, f"Boundary translation for {lang} incorrect"

    def test_methodology_translation(self):
        """Test translation of 'methodology' across languages."""
        translations = {
            "en": "Methodology",
            "es": "Metodología",
            "fr": "Méthodologie",
            "de": "Methodik",
            "zh": "方法论",
            "ja": "方法論",
            "pt": "Metodologia",
            "hi": "कार्यप्रणाली",
        }

        for lang, expected in translations.items():
            i18n = I18n(lang)
            msg = i18n.get("methodology")
            assert msg == expected, f"Methodology translation for {lang} incorrect"


class TestGetTranslator:
    """Test get_translator helper function."""

    def test_get_translator_english(self):
        """Test getting English translator."""
        translator = get_translator("en")
        assert isinstance(translator, I18n)
        assert translator.language == "en"

    def test_get_translator_spanish(self):
        """Test getting Spanish translator."""
        translator = get_translator("es")
        assert translator.language == "es"

    def test_get_translator_default(self):
        """Test getting translator with default language."""
        translator = get_translator()
        assert translator.language == "en"


class TestSupportedLanguages:
    """Test supported languages listing."""

    def test_get_supported_languages(self):
        """Test getting list of supported languages."""
        languages = get_supported_languages()

        assert isinstance(languages, dict)
        assert len(languages) == 8
        assert "en" in languages
        assert "es" in languages
        assert "fr" in languages
        assert "de" in languages
        assert "zh" in languages
        assert "ja" in languages
        assert "pt" in languages
        assert "hi" in languages

    def test_language_names(self):
        """Test language names are correct."""
        languages = get_supported_languages()

        assert languages["en"] == "English"
        assert languages["es"] == "Spanish (Español)"
        assert languages["fr"] == "French (Français)"
        assert languages["de"] == "German (Deutsch)"
        assert languages["zh"] == "Chinese Simplified (简体中文)"
        assert languages["ja"] == "Japanese (日本語)"
        assert languages["pt"] == "Portuguese (Português)"
        assert languages["hi"] == "Hindi (हिन्दी)"


class TestErrorMessages:
    """Test error message translation."""

    def test_error_occurred_translation(self):
        """Test 'error occurred' message across languages."""
        error_messages = {
            "en": "An error occurred",
            "es": "Ocurrió un error",
            "fr": "Une erreur s'est produite",
            "de": "Ein Fehler ist aufgetreten",
            "zh": "发生错误",
            "ja": "エラーが発生しました",
            "pt": "Ocorreu um erro",
            "hi": "एक त्रुटि हुई",
        }

        for lang, expected in error_messages.items():
            i18n = I18n(lang)
            msg = i18n.get("error_occurred")
            assert msg == expected

    def test_invalid_input_translation(self):
        """Test 'invalid input' message across languages."""
        messages = {
            "en": "Invalid input",
            "es": "Entrada inválida",
            "fr": "Entrée invalide",
            "de": "Ungültige Eingabe",
            "zh": "输入无效",
            "ja": "無効な入力",
            "pt": "Entrada inválida",
            "hi": "अमान्य इनपुट",
        }

        for lang, expected in messages.items():
            i18n = I18n(lang)
            msg = i18n.get("invalid_input")
            assert msg == expected


class TestContextualMessages:
    """Test contextual messages."""

    def test_fast_path_message(self):
        """Test fast path message translation."""
        messages = {
            "en": "Fast path (deterministic)",
            "es": "Ruta rápida (determinística)",
            "fr": "Voie rapide (déterministe)",
            "de": "Schnellpfad (deterministisch)",
            "zh": "快速路径（确定性）",
            "ja": "高速パス（決定論的）",
            "pt": "Caminho rápido (determinístico)",
            "hi": "तेज़ मार्ग (निर्धारक)",
        }

        for lang, expected in messages.items():
            i18n = I18n(lang)
            msg = i18n.get("fast_path")
            assert msg == expected

    def test_multigas_breakdown_message(self):
        """Test multi-gas breakdown message translation."""
        messages = {
            "en": "Multi-gas breakdown",
            "es": "Desglose de múltiples gases",
            "fr": "Répartition multi-gaz",
            "de": "Mehrgas-Aufschlüsselung",
            "zh": "多气体分解",
            "ja": "複数ガス内訳",
            "pt": "Detalhamento de múltiplos gases",
            "hi": "बहु-गैस विवरण",
        }

        for lang, expected in messages.items():
            i18n = I18n(lang)
            msg = i18n.get("multigas_breakdown")
            assert msg == expected


class TestComplianceMessages:
    """Test compliance-related messages."""

    def test_compliance_ready_message(self):
        """Test compliance ready message translation."""
        messages = {
            "en": "Compliance ready for",
            "es": "Listo para cumplimiento de",
            "fr": "Conforme pour",
            "de": "Konform für",
            "zh": "符合",
            "ja": "準拠準備完了",
            "pt": "Pronto para conformidade com",
            "hi": "अनुपालन के लिए तैयार",
        }

        for lang, expected in messages.items():
            i18n = I18n(lang)
            msg = i18n.get("compliance_ready")
            assert msg == expected


class TestCaseSensitivity:
    """Test language code case sensitivity."""

    def test_uppercase_language_code(self):
        """Test that uppercase language codes work."""
        i18n = I18n("ES")  # Uppercase
        assert i18n.language == "es"
        msg = i18n.get("calculation_complete")
        assert "Cálculo" in msg

    def test_mixed_case_language_code(self):
        """Test that mixed case language codes work."""
        i18n = I18n("Fr")  # Mixed case
        assert i18n.language == "fr"
        msg = i18n.get("calculation_complete")
        assert "Calcul" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
