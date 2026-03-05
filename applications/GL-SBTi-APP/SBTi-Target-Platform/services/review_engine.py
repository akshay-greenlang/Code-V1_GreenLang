"""
Review Engine -- Facade for FiveYearReviewEngine

Re-exports FiveYearReviewEngine as ReviewEngine for the unified
__init__.py and setup module naming convention.

The underlying engine lives in five_year_review_engine.py and
implements five-year review scheduling, deadline tracking, notification
scheduling, readiness assessment, progress summaries, and outcome
recording per SBTi criterion C23.

Example:
    >>> engine = ReviewEngine(config)
    >>> review = engine.create_review("org-1", "tgt-1", date(2021, 6, 15))
"""

from .five_year_review_engine import FiveYearReviewEngine as ReviewEngine

__all__ = ["ReviewEngine"]
