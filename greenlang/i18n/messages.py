"""
greenlang/i18n/messages.py

Multi-Language Support for FuelAgentAI v2

Supported Languages:
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Chinese Simplified (zh)
- Japanese (ja)
- Portuguese (pt)
- Hindi (hi)

Author: GreenLang Framework Team
Date: October 2025
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ==================== MESSAGE CATALOG ====================

MESSAGES = {
    # English
    "en": {
        "calculation_complete": "Emission calculation completed successfully",
        "total_emissions": "Total emissions",
        "fuel_consumption": "Fuel consumption",
        "emission_factor": "Emission factor",
        "data_quality": "Data quality",
        "uncertainty": "Uncertainty",
        "source": "Source",
        "methodology": "Methodology",
        "scope": "Scope",
        "boundary": "Boundary",
        "recommendations": "Recommendations",
        "renewable_offset": "Renewable offset applied",
        "efficiency_adjusted": "Efficiency adjustment applied",
        "multigas_breakdown": "Multi-gas breakdown",
        "provenance": "Provenance information",
        "compliance_ready": "Compliance ready for",
        "error_occurred": "An error occurred",
        "invalid_input": "Invalid input",
        "fuel_not_found": "Fuel type not found",
        "unit_not_supported": "Unit not supported",
        "region_not_supported": "Region not supported",
        "calculation_method": "Calculation method",
        "fast_path": "Fast path (deterministic)",
        "ai_path": "AI-assisted path",
        "cache_hit": "Retrieved from cache",
        "new_calculation": "New calculation",
    },

    # Spanish
    "es": {
        "calculation_complete": "Cálculo de emisiones completado exitosamente",
        "total_emissions": "Emisiones totales",
        "fuel_consumption": "Consumo de combustible",
        "emission_factor": "Factor de emisión",
        "data_quality": "Calidad de datos",
        "uncertainty": "Incertidumbre",
        "source": "Fuente",
        "methodology": "Metodología",
        "scope": "Alcance",
        "boundary": "Límite",
        "recommendations": "Recomendaciones",
        "renewable_offset": "Compensación renovable aplicada",
        "efficiency_adjusted": "Ajuste de eficiencia aplicado",
        "multigas_breakdown": "Desglose de múltiples gases",
        "provenance": "Información de procedencia",
        "compliance_ready": "Listo para cumplimiento de",
        "error_occurred": "Ocurrió un error",
        "invalid_input": "Entrada inválida",
        "fuel_not_found": "Tipo de combustible no encontrado",
        "unit_not_supported": "Unidad no compatible",
        "region_not_supported": "Región no compatible",
        "calculation_method": "Método de cálculo",
        "fast_path": "Ruta rápida (determinística)",
        "ai_path": "Ruta asistida por IA",
        "cache_hit": "Recuperado de caché",
        "new_calculation": "Nuevo cálculo",
    },

    # French
    "fr": {
        "calculation_complete": "Calcul des émissions terminé avec succès",
        "total_emissions": "Émissions totales",
        "fuel_consumption": "Consommation de carburant",
        "emission_factor": "Facteur d'émission",
        "data_quality": "Qualité des données",
        "uncertainty": "Incertitude",
        "source": "Source",
        "methodology": "Méthodologie",
        "scope": "Portée",
        "boundary": "Limite",
        "recommendations": "Recommandations",
        "renewable_offset": "Compensation renouvelable appliquée",
        "efficiency_adjusted": "Ajustement d'efficacité appliqué",
        "multigas_breakdown": "Répartition multi-gaz",
        "provenance": "Information de provenance",
        "compliance_ready": "Conforme pour",
        "error_occurred": "Une erreur s'est produite",
        "invalid_input": "Entrée invalide",
        "fuel_not_found": "Type de carburant non trouvé",
        "unit_not_supported": "Unité non supportée",
        "region_not_supported": "Région non supportée",
        "calculation_method": "Méthode de calcul",
        "fast_path": "Voie rapide (déterministe)",
        "ai_path": "Voie assistée par IA",
        "cache_hit": "Récupéré du cache",
        "new_calculation": "Nouveau calcul",
    },

    # German
    "de": {
        "calculation_complete": "Emissionsberechnung erfolgreich abgeschlossen",
        "total_emissions": "Gesamtemissionen",
        "fuel_consumption": "Kraftstoffverbrauch",
        "emission_factor": "Emissionsfaktor",
        "data_quality": "Datenqualität",
        "uncertainty": "Unsicherheit",
        "source": "Quelle",
        "methodology": "Methodik",
        "scope": "Umfang",
        "boundary": "Grenze",
        "recommendations": "Empfehlungen",
        "renewable_offset": "Erneuerbare Kompensation angewendet",
        "efficiency_adjusted": "Effizienzanpassung angewendet",
        "multigas_breakdown": "Mehrgas-Aufschlüsselung",
        "provenance": "Herkunftsinformationen",
        "compliance_ready": "Konform für",
        "error_occurred": "Ein Fehler ist aufgetreten",
        "invalid_input": "Ungültige Eingabe",
        "fuel_not_found": "Kraftstofftyp nicht gefunden",
        "unit_not_supported": "Einheit nicht unterstützt",
        "region_not_supported": "Region nicht unterstützt",
        "calculation_method": "Berechnungsmethode",
        "fast_path": "Schnellpfad (deterministisch)",
        "ai_path": "KI-unterstützter Pfad",
        "cache_hit": "Aus Cache abgerufen",
        "new_calculation": "Neue Berechnung",
    },

    # Chinese Simplified
    "zh": {
        "calculation_complete": "排放计算成功完成",
        "total_emissions": "总排放量",
        "fuel_consumption": "燃料消耗",
        "emission_factor": "排放因子",
        "data_quality": "数据质量",
        "uncertainty": "不确定性",
        "source": "来源",
        "methodology": "方法论",
        "scope": "范围",
        "boundary": "边界",
        "recommendations": "建议",
        "renewable_offset": "已应用可再生能源抵消",
        "efficiency_adjusted": "已应用效率调整",
        "multigas_breakdown": "多气体分解",
        "provenance": "来源信息",
        "compliance_ready": "符合",
        "error_occurred": "发生错误",
        "invalid_input": "输入无效",
        "fuel_not_found": "未找到燃料类型",
        "unit_not_supported": "不支持的单位",
        "region_not_supported": "不支持的地区",
        "calculation_method": "计算方法",
        "fast_path": "快速路径（确定性）",
        "ai_path": "AI辅助路径",
        "cache_hit": "从缓存检索",
        "new_calculation": "新计算",
    },

    # Japanese
    "ja": {
        "calculation_complete": "排出量計算が正常に完了しました",
        "total_emissions": "総排出量",
        "fuel_consumption": "燃料消費量",
        "emission_factor": "排出係数",
        "data_quality": "データ品質",
        "uncertainty": "不確実性",
        "source": "出典",
        "methodology": "方法論",
        "scope": "スコープ",
        "boundary": "境界",
        "recommendations": "推奨事項",
        "renewable_offset": "再生可能エネルギーオフセット適用済み",
        "efficiency_adjusted": "効率調整適用済み",
        "multigas_breakdown": "複数ガス内訳",
        "provenance": "出所情報",
        "compliance_ready": "準拠準備完了",
        "error_occurred": "エラーが発生しました",
        "invalid_input": "無効な入力",
        "fuel_not_found": "燃料タイプが見つかりません",
        "unit_not_supported": "サポートされていない単位",
        "region_not_supported": "サポートされていない地域",
        "calculation_method": "計算方法",
        "fast_path": "高速パス（決定論的）",
        "ai_path": "AI支援パス",
        "cache_hit": "キャッシュから取得",
        "new_calculation": "新しい計算",
    },

    # Portuguese
    "pt": {
        "calculation_complete": "Cálculo de emissões concluído com sucesso",
        "total_emissions": "Emissões totais",
        "fuel_consumption": "Consumo de combustível",
        "emission_factor": "Fator de emissão",
        "data_quality": "Qualidade dos dados",
        "uncertainty": "Incerteza",
        "source": "Fonte",
        "methodology": "Metodologia",
        "scope": "Escopo",
        "boundary": "Limite",
        "recommendations": "Recomendações",
        "renewable_offset": "Compensação renovável aplicada",
        "efficiency_adjusted": "Ajuste de eficiência aplicado",
        "multigas_breakdown": "Detalhamento de múltiplos gases",
        "provenance": "Informações de proveniência",
        "compliance_ready": "Pronto para conformidade com",
        "error_occurred": "Ocorreu um erro",
        "invalid_input": "Entrada inválida",
        "fuel_not_found": "Tipo de combustível não encontrado",
        "unit_not_supported": "Unidade não suportada",
        "region_not_supported": "Região não suportada",
        "calculation_method": "Método de cálculo",
        "fast_path": "Caminho rápido (determinístico)",
        "ai_path": "Caminho assistido por IA",
        "cache_hit": "Recuperado do cache",
        "new_calculation": "Novo cálculo",
    },

    # Hindi
    "hi": {
        "calculation_complete": "उत्सर्जन गणना सफलतापूर्वक पूर्ण हुई",
        "total_emissions": "कुल उत्सर्जन",
        "fuel_consumption": "ईंधन की खपत",
        "emission_factor": "उत्सर्जन कारक",
        "data_quality": "डेटा गुणवत्ता",
        "uncertainty": "अनिश्चितता",
        "source": "स्रोत",
        "methodology": "कार्यप्रणाली",
        "scope": "दायरा",
        "boundary": "सीमा",
        "recommendations": "सिफारिशें",
        "renewable_offset": "नवीकरणीय ऑफसेट लागू",
        "efficiency_adjusted": "दक्षता समायोजन लागू",
        "multigas_breakdown": "बहु-गैस विवरण",
        "provenance": "उत्पत्ति जानकारी",
        "compliance_ready": "अनुपालन के लिए तैयार",
        "error_occurred": "एक त्रुटि हुई",
        "invalid_input": "अमान्य इनपुट",
        "fuel_not_found": "ईंधन का प्रकार नहीं मिला",
        "unit_not_supported": "इकाई समर्थित नहीं",
        "region_not_supported": "क्षेत्र समर्थित नहीं",
        "calculation_method": "गणना विधि",
        "fast_path": "तेज़ मार्ग (निर्धारक)",
        "ai_path": "AI-सहायक मार्ग",
        "cache_hit": "कैश से पुनर्प्राप्त",
        "new_calculation": "नई गणना",
    },
}


class I18n:
    """
    Internationalization (i18n) support for FuelAgentAI v2.

    Provides multi-language message translation and
    locale-aware formatting.
    """

    def __init__(self, language: str = "en"):
        """
        Initialize i18n with language.

        Args:
            language: ISO 639-1 language code (en, es, fr, de, zh, ja, pt, hi)
        """
        self.language = language.lower()

        if self.language not in MESSAGES:
            logger.warning(f"Language {language} not supported, falling back to English")
            self.language = "en"

    def get(self, key: str, **kwargs) -> str:
        """
        Get translated message.

        Args:
            key: Message key
            **kwargs: Format parameters

        Returns:
            Translated and formatted message
        """
        # Get message for current language
        message = MESSAGES[self.language].get(key)

        # Fallback to English if not found
        if message is None:
            message = MESSAGES["en"].get(key, key)
            logger.warning(f"Message key '{key}' not found for language '{self.language}'")

        # Format with parameters if provided
        if kwargs:
            try:
                message = message.format(**kwargs)
            except KeyError as e:
                logger.error(f"Missing format parameter {e} for message '{key}'")

        return message

    def translate_recommendations(
        self,
        recommendations: list,
        fuel_type: str
    ) -> list:
        """
        Translate fuel switching recommendations.

        Args:
            recommendations: List of recommendation dicts
            fuel_type: Current fuel type

        Returns:
            Translated recommendations list
        """
        # Recommendation templates by language
        templates = {
            "en": {
                "diesel": [
                    {
                        "action": "Switch to biodiesel (B20)",
                        "description": "Blend 20% biodiesel to reduce lifecycle emissions by ~15%",
                    },
                    {
                        "action": "Upgrade to electric vehicle fleet",
                        "description": "Transition to EVs for 50-70% emission reduction (grid-dependent)",
                    },
                ],
                "natural_gas": [
                    {
                        "action": "Improve boiler efficiency",
                        "description": "Upgrade to high-efficiency condensing boiler (90%+ efficiency)",
                    },
                    {
                        "action": "Switch to renewable natural gas (RNG)",
                        "description": "Source RNG from anaerobic digestion or biogas",
                    },
                ],
                "electricity": [
                    {
                        "action": "Purchase renewable energy certificates (RECs)",
                        "description": "Offset Scope 2 emissions via market-based accounting",
                    },
                    {
                        "action": "Install on-site solar PV",
                        "description": "Generate 20-40% of electricity demand on-site",
                    },
                ],
            },
            "es": {
                "diesel": [
                    {
                        "action": "Cambiar a biodiesel (B20)",
                        "description": "Mezclar 20% de biodiesel para reducir las emisiones del ciclo de vida en ~15%",
                    },
                    {
                        "action": "Actualizar a flota de vehículos eléctricos",
                        "description": "Transición a vehículos eléctricos para una reducción de emisiones del 50-70% (dependiente de la red)",
                    },
                ],
                "natural_gas": [
                    {
                        "action": "Mejorar la eficiencia de la caldera",
                        "description": "Actualizar a caldera de condensación de alta eficiencia (eficiencia >90%)",
                    },
                    {
                        "action": "Cambiar a gas natural renovable (GNR)",
                        "description": "Fuente de GNR de digestión anaeróbica o biogás",
                    },
                ],
                "electricity": [
                    {
                        "action": "Comprar certificados de energía renovable (CER)",
                        "description": "Compensar las emisiones de Alcance 2 mediante contabilidad basada en el mercado",
                    },
                    {
                        "action": "Instalar energía solar fotovoltaica in situ",
                        "description": "Generar 20-40% de la demanda de electricidad in situ",
                    },
                ],
            },
            # Add more languages as needed
        }

        # Get templates for current language (fallback to English)
        lang_templates = templates.get(self.language, templates["en"])

        # Get recommendations for fuel type
        fuel_recommendations = lang_templates.get(fuel_type, [])

        # Merge with original recommendations (keep percentages/costs)
        translated = []
        for i, rec in enumerate(recommendations):
            if i < len(fuel_recommendations):
                translated_rec = {**rec, **fuel_recommendations[i]}
                translated.append(translated_rec)
            else:
                translated.append(rec)

        return translated


def get_translator(language: str = "en") -> I18n:
    """
    Get translator instance for language.

    Args:
        language: ISO 639-1 language code

    Returns:
        I18n translator instance
    """
    return I18n(language)


# ==================== SUPPORTED LANGUAGES ====================

SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "zh": "Chinese Simplified (简体中文)",
    "ja": "Japanese (日本語)",
    "pt": "Portuguese (Português)",
    "hi": "Hindi (हिन्दी)",
}


def get_supported_languages() -> Dict[str, str]:
    """
    Get list of supported languages.

    Returns:
        Dict of language code -> language name
    """
    return SUPPORTED_LANGUAGES.copy()
