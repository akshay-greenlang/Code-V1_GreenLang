"""
Localization support for email templates.

Provides i18n for subject lines and key content in EN, DE, FR, ES, CN.
"""
from typing import Dict, Any
from enum import Enum


class Language(str, Enum):
    """Supported languages."""
    ENGLISH = "en"
    GERMAN = "de"
    FRENCH = "fr"
    SPANISH = "es"
    CHINESE = "zh"


# Subject line translations
SUBJECT_TRANSLATIONS = {
    "touch_1_introduction": {
        "en": "Partner with ${company_name} on carbon transparency",
        "de": "Partnerschaft mit ${company_name} für CO2-Transparenz",
        "fr": "Partenariat avec ${company_name} sur la transparence carbone",
        "es": "Asóciese con ${company_name} en transparencia de carbono",
        "zh": "与${company_name}合作实现碳透明度"
    },
    "touch_2_reminder": {
        "en": "Action needed: Carbon data request from ${company_name}",
        "de": "Aktion erforderlich: CO2-Datenanfrage von ${company_name}",
        "fr": "Action requise: Demande de données carbone de ${company_name}",
        "es": "Acción requerida: Solicitud de datos de carbono de ${company_name}",
        "zh": "需要采取行动：${company_name}的碳数据请求"
    },
    "touch_3_final_reminder": {
        "en": "Final reminder: Carbon transparency program deadline approaching",
        "de": "Letzte Erinnerung: Frist für CO2-Transparenzprogramm nähert sich",
        "fr": "Dernier rappel: Date limite du programme de transparence carbone approche",
        "es": "Recordatorio final: Se acerca la fecha límite del programa de transparencia de carbono",
        "zh": "最后提醒：碳透明度计划截止日期临近"
    },
    "touch_4_thank_you": {
        "en": "Thank you from ${company_name} sustainability team",
        "de": "Vielen Dank vom ${company_name} Nachhaltigkeitsteam",
        "fr": "Merci de l'équipe de développement durable de ${company_name}",
        "es": "Gracias del equipo de sostenibilidad de ${company_name}",
        "zh": "${company_name}可持续发展团队的感谢"
    }
}


# Key phrase translations
KEY_PHRASES = {
    "dear": {
        "en": "Dear",
        "de": "Sehr geehrte/r",
        "fr": "Cher/Chère",
        "es": "Estimado/a",
        "zh": "尊敬的"
    },
    "best_regards": {
        "en": "Best regards",
        "de": "Mit freundlichen Grüßen",
        "fr": "Cordialement",
        "es": "Atentamente",
        "zh": "此致敬礼"
    },
    "privacy_policy": {
        "en": "Privacy Policy",
        "de": "Datenschutzrichtlinie",
        "fr": "Politique de confidentialité",
        "es": "Política de privacidad",
        "zh": "隐私政策"
    },
    "unsubscribe": {
        "en": "Unsubscribe",
        "de": "Abmelden",
        "fr": "Se désabonner",
        "es": "Darse de baja",
        "zh": "取消订阅"
    },
    "access_portal": {
        "en": "Access Supplier Portal",
        "de": "Zugang zum Lieferantenportal",
        "fr": "Accéder au portail fournisseur",
        "es": "Acceder al portal de proveedores",
        "zh": "访问供应商门户"
    },
    "upload_data": {
        "en": "Upload Your Data Now",
        "de": "Laden Sie Ihre Daten jetzt hoch",
        "fr": "Téléchargez vos données maintenant",
        "es": "Suba sus datos ahora",
        "zh": "立即上传您的数据"
    },
    "thank_you": {
        "en": "Thank you for your participation!",
        "de": "Vielen Dank für Ihre Teilnahme!",
        "fr": "Merci de votre participation!",
        "es": "¡Gracias por su participación!",
        "zh": "感谢您的参与！"
    }
}


# Content block translations
CONTENT_BLOCKS = {
    "value_proposition": {
        "en": "What's in it for you?",
        "de": "Was bringt es Ihnen?",
        "fr": "Qu'est-ce que cela vous apporte?",
        "es": "¿Qué hay para ti?",
        "zh": "对您有什么好处？"
    },
    "next_steps": {
        "en": "Next Steps:",
        "de": "Nächste Schritte:",
        "fr": "Prochaines étapes:",
        "es": "Próximos pasos:",
        "zh": "下一步："
    },
    "need_help": {
        "en": "Need help?",
        "de": "Brauchen Sie Hilfe?",
        "fr": "Besoin d'aide?",
        "es": "¿Necesita ayuda?",
        "zh": "需要帮助？"
    },
    "deadline": {
        "en": "Deadline",
        "de": "Frist",
        "fr": "Date limite",
        "es": "Fecha límite",
        "zh": "截止日期"
    }
}


class Localizer:
    """
    Handles template localization and translation.
    """

    def __init__(self, default_language: str = "en"):
        """
        Initialize localizer.

        Args:
            default_language: Default language code
        """
        self.default_language = default_language

    def get_subject(
        self,
        template_id: str,
        language: str = "en"
    ) -> str:
        """
        Get localized subject line.

        Args:
            template_id: Template identifier
            language: Language code

        Returns:
            Localized subject line
        """
        subjects = SUBJECT_TRANSLATIONS.get(template_id, {})
        return subjects.get(language, subjects.get(self.default_language, ""))

    def get_phrase(
        self,
        phrase_key: str,
        language: str = "en"
    ) -> str:
        """
        Get localized phrase.

        Args:
            phrase_key: Phrase key
            language: Language code

        Returns:
            Localized phrase
        """
        phrases = KEY_PHRASES.get(phrase_key, {})
        return phrases.get(language, phrases.get(self.default_language, ""))

    def get_content_block(
        self,
        block_key: str,
        language: str = "en"
    ) -> str:
        """
        Get localized content block.

        Args:
            block_key: Content block key
            language: Language code

        Returns:
            Localized content block
        """
        blocks = CONTENT_BLOCKS.get(block_key, {})
        return blocks.get(language, blocks.get(self.default_language, ""))

    def localize_template_data(
        self,
        template_id: str,
        personalization_data: Dict[str, Any],
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Add localized elements to personalization data.

        Args:
            template_id: Template identifier
            personalization_data: Existing personalization data
            language: Language code

        Returns:
            Enhanced personalization data with localized elements
        """
        localized_data = personalization_data.copy()

        # Add localized phrases
        localized_data["l10n_dear"] = self.get_phrase("dear", language)
        localized_data["l10n_best_regards"] = self.get_phrase("best_regards", language)
        localized_data["l10n_privacy_policy"] = self.get_phrase("privacy_policy", language)
        localized_data["l10n_unsubscribe"] = self.get_phrase("unsubscribe", language)

        # Add localized content blocks
        localized_data["l10n_value_proposition"] = self.get_content_block("value_proposition", language)
        localized_data["l10n_next_steps"] = self.get_content_block("next_steps", language)
        localized_data["l10n_need_help"] = self.get_content_block("need_help", language)

        return localized_data

    def get_supported_languages(self) -> list:
        """
        Get list of supported language codes.

        Returns:
            List of language codes
        """
        return [lang.value for lang in Language]

    def get_language_name(self, language_code: str) -> str:
        """
        Get human-readable language name.

        Args:
            language_code: Language code

        Returns:
            Language name
        """
        names = {
            "en": "English",
            "de": "Deutsch",
            "fr": "Français",
            "es": "Español",
            "zh": "中文"
        }
        return names.get(language_code, "English")
