# -*- coding: utf-8 -*-
"""Certified vs preview promotion state machine (Q4)."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Tuple

logger = logging.getLogger(__name__)


class PromotionState(str, Enum):
    PREVIEW = "preview"
    CERTIFIED = "certified"
    REJECTED = "rejected"


def transition(
    current: PromotionState,
    *,
    qa_pass: bool,
    methodology_signed: bool,
    legal_ok: bool,
    approval_gate_ok: bool,
) -> Tuple[PromotionState, str]:
    """
    Deterministic next state from gates. Does not persist; callers update DB rows.
    """
    if current == PromotionState.REJECTED:
        return PromotionState.REJECTED, "terminal"
    if not qa_pass:
        return PromotionState.PREVIEW, "awaiting_qa"
    if not methodology_signed:
        return PromotionState.PREVIEW, "awaiting_methodology"
    if not legal_ok:
        return PromotionState.PREVIEW, "awaiting_legal"
    if not approval_gate_ok:
        return PromotionState.PREVIEW, "awaiting_approval_gate"
    logger.info("Promotion %s -> CERTIFIED (all gates passed)", current.value)
    return PromotionState.CERTIFIED, "promoted"
