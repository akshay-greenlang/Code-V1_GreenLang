"""
Model and Dataset Cards System
===============================

Provides HuggingFace-style cards for documentation and transparency.
"""

from .generator import (
    generate_pack_card,
    generate_dataset_card,
    generate_model_card,
    generate_pipeline_card,
)

from .templates import (
    PACK_CARD_TEMPLATE,
    DATASET_CARD_TEMPLATE,
    MODEL_CARD_TEMPLATE,
    PIPELINE_CARD_TEMPLATE,
)

from .validator import (
    validate_card,
    CardValidator,
    CardValidationResult,
    generate_validation_report,
)

__all__ = [
    "generate_pack_card",
    "generate_dataset_card",
    "generate_model_card",
    "generate_pipeline_card",
    "PACK_CARD_TEMPLATE",
    "DATASET_CARD_TEMPLATE",
    "MODEL_CARD_TEMPLATE",
    "PIPELINE_CARD_TEMPLATE",
    "validate_card",
    "CardValidator",
    "CardValidationResult",
    "generate_validation_report",
]
