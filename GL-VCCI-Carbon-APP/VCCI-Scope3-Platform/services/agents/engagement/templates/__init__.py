"""
Email templates module.
"""
from .email_templates import (
    EMAIL_TEMPLATES,
    TOUCH_1_TEMPLATE,
    TOUCH_2_TEMPLATE,
    TOUCH_3_TEMPLATE,
    TOUCH_4_TEMPLATE,
    get_template,
    render_template,
    list_templates,
    get_template_metadata
)
from .localization import Localizer, Language


__all__ = [
    "EMAIL_TEMPLATES",
    "TOUCH_1_TEMPLATE",
    "TOUCH_2_TEMPLATE",
    "TOUCH_3_TEMPLATE",
    "TOUCH_4_TEMPLATE",
    "get_template",
    "render_template",
    "list_templates",
    "get_template_metadata",
    "Localizer",
    "Language",
]
