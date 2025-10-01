"""
Budget Enforcement System

Cost and token caps for LLM usage:
- Per-call budgets (prevent runaway costs)
- Per-agent budgets (aggregate tracking)
- Per-workflow budgets (pipeline-level caps)

Enables:
- Cost control in production
- Chargeback to business units
- Early termination when budget exhausted
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class BudgetExceeded(Exception):
    """
    Raised when budget cap is exceeded

    Attributes:
        message: Description of budget violation
        spent_usd: Amount spent so far
        max_usd: Budget cap
        spent_tokens: Tokens consumed so far
        max_tokens: Token cap (if set)
    """

    def __init__(
        self,
        message: str,
        spent_usd: float,
        max_usd: float,
        spent_tokens: int = 0,
        max_tokens: Optional[int] = None,
    ):
        super().__init__(message)
        self.message = message
        self.spent_usd = spent_usd
        self.max_usd = max_usd
        self.spent_tokens = spent_tokens
        self.max_tokens = max_tokens

    def __str__(self) -> str:
        parts = [
            self.message,
            f"Spent: ${self.spent_usd:.4f} / ${self.max_usd:.4f}",
        ]
        if self.max_tokens:
            parts.append(
                f"Tokens: {self.spent_tokens:,} / {self.max_tokens:,}"
            )
        return " | ".join(parts)


class Budget(BaseModel):
    """
    Budget tracker with dollar and token caps

    Enforces:
    - Maximum USD spend per call/agent/workflow
    - Optional token limit (for models with context limits)

    Usage:
        # Create budget with $0.50 cap
        budget = Budget(max_usd=0.50)

        # Check if additional spend would exceed cap
        budget.check(add_usd=0.02, add_tokens=1000)  # OK

        # Add usage (raises BudgetExceeded if over cap)
        budget.add(add_usd=0.02, add_tokens=1000)

        # Check remaining budget
        print(f"Remaining: ${budget.remaining_usd:.4f}")

    Example with token cap:
        # Limit both cost and tokens (e.g., for 8K context models)
        budget = Budget(max_usd=1.00, max_tokens=8000)

        # This would exceed token cap (even if cost OK)
        budget.add(add_usd=0.05, add_tokens=9000)  # raises BudgetExceeded
    """

    max_usd: float = Field(
        default=0.50, description="Maximum USD spend (per call/agent/workflow)"
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Optional token cap (for context limits)"
    )
    spent_usd: float = Field(default=0.0, description="USD spent so far")
    spent_tokens: int = Field(default=0, description="Tokens consumed so far")

    @property
    def remaining_usd(self) -> float:
        """Remaining budget in USD"""
        return max(0.0, self.max_usd - self.spent_usd)

    @property
    def remaining_tokens(self) -> Optional[int]:
        """Remaining token budget (None if no token cap)"""
        if self.max_tokens is None:
            return None
        return max(0, self.max_tokens - self.spent_tokens)

    @property
    def is_exhausted(self) -> bool:
        """Check if budget is exhausted"""
        if self.spent_usd >= self.max_usd:
            return True
        if self.max_tokens and self.spent_tokens >= self.max_tokens:
            return True
        return False

    def check(self, add_usd: float, add_tokens: int) -> None:
        """
        Check if adding usage would exceed budget

        Args:
            add_usd: Additional USD cost
            add_tokens: Additional token count

        Raises:
            BudgetExceeded: If adding usage would exceed cap

        Example:
            budget = Budget(max_usd=0.50)
            budget.add(add_usd=0.48, add_tokens=5000)

            # This check will fail (would exceed $0.50 cap)
            budget.check(add_usd=0.05, add_tokens=500)  # raises BudgetExceeded
        """
        # Check token cap
        if self.max_tokens is not None:
            total_tokens = self.spent_tokens + add_tokens
            if total_tokens > self.max_tokens:
                raise BudgetExceeded(
                    message="Token cap exceeded",
                    spent_usd=self.spent_usd,
                    max_usd=self.max_usd,
                    spent_tokens=total_tokens,
                    max_tokens=self.max_tokens,
                )

        # Check dollar cap
        total_usd = self.spent_usd + add_usd
        if total_usd > self.max_usd:
            raise BudgetExceeded(
                message="Dollar cap exceeded",
                spent_usd=total_usd,
                max_usd=self.max_usd,
                spent_tokens=self.spent_tokens + add_tokens,
                max_tokens=self.max_tokens,
            )

    def add(self, add_usd: float, add_tokens: int) -> None:
        """
        Add usage to budget (with cap enforcement)

        Args:
            add_usd: USD cost to add
            add_tokens: Token count to add

        Raises:
            BudgetExceeded: If adding usage would exceed cap

        Example:
            budget = Budget(max_usd=0.50)

            # Add usage (OK)
            budget.add(add_usd=0.02, add_tokens=1000)

            # Add more (still OK)
            budget.add(add_usd=0.03, add_tokens=1500)

            # This would exceed cap
            budget.add(add_usd=0.50, add_tokens=20000)  # raises BudgetExceeded
        """
        # Check first (raises if would exceed)
        self.check(add_usd, add_tokens)

        # Add to spent amounts
        self.spent_usd += add_usd
        self.spent_tokens += add_tokens

    def reset(self) -> None:
        """
        Reset budget counters to zero

        Useful for reusing Budget object across multiple calls.

        Example:
            budget = Budget(max_usd=0.50)
            budget.add(add_usd=0.20, add_tokens=2000)

            # Reset for next call
            budget.reset()
            assert budget.spent_usd == 0
            assert budget.spent_tokens == 0
        """
        self.spent_usd = 0.0
        self.spent_tokens = 0

    def merge(self, other: Budget) -> None:
        """
        Merge another budget's spending into this one

        Useful for aggregating sub-budgets (e.g., agent budgets -> workflow budget).

        Args:
            other: Budget to merge from

        Raises:
            BudgetExceeded: If merged total would exceed cap

        Example:
            workflow_budget = Budget(max_usd=2.00)
            agent1_budget = Budget(max_usd=0.50, spent_usd=0.30)
            agent2_budget = Budget(max_usd=0.50, spent_usd=0.25)

            # Aggregate agent spending into workflow budget
            workflow_budget.merge(agent1_budget)
            workflow_budget.merge(agent2_budget)

            assert workflow_budget.spent_usd == 0.55
        """
        self.add(add_usd=other.spent_usd, add_tokens=other.spent_tokens)

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "max_usd": 0.50,
                    "max_tokens": 4000,
                    "spent_usd": 0.12,
                    "spent_tokens": 1500,
                }
            ]
        }
