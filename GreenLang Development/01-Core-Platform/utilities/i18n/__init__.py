# -*- coding: utf-8 -*-
"""
GreenLang i18n (Internationalization) Package

This module provides multi-language support for GreenLang messages and labels.

Supported Languages:
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Chinese Simplified (zh)
- Japanese (ja)
- Portuguese (pt)
- Hindi (hi)
"""

from .messages import MESSAGES, I18n, get_translator, get_supported_languages

__all__ = [
    "MESSAGES",
    "I18n",
    "get_translator",
    "get_supported_languages",
]
