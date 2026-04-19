# -*- coding: utf-8 -*-
"""
Monetization System

Implements payment processing, subscription management, license generation,
and revenue analytics using Stripe integration.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid
import hmac
import hashlib
import logging

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc

from greenlang.marketplace.models import (
    MarketplaceAgent,
    AgentPurchase,
    PricingType,
)
from greenlang.utilities.determinism import FinancialDecimal
from greenlang.utilities.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)


# Stripe configuration (in production, use environment variables)
STRIPE_SECRET_KEY = "sk_test_..."
STRIPE_PUBLISHABLE_KEY = "pk_test_..."
PLATFORM_FEE_PERCENT = 20  # Platform takes 20%


class PaymentStatus(str, Enum):
    """Payment status"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    DISPUTED = "disputed"


class PricingModel(str, Enum):
    """Pricing models"""
    FREE = "free"
    ONE_TIME = "one_time"
    MONTHLY = "subscription_monthly"
    ANNUAL = "subscription_annual"
    USAGE_BASED = "usage_based"
    FREEMIUM = "freemium"


@dataclass
class PaymentIntent:
    """Payment intent data"""
    amount: Decimal
    currency: str
    agent_id: str
    user_id: str
    pricing_type: str
    metadata: Dict[str, Any]


@dataclass
class RefundRequest:
    """Refund request"""
    purchase_id: str
    amount: Optional[Decimal]
    reason: str


@dataclass
class RevenueStats:
    """Revenue statistics"""
    total_revenue: Decimal
    period_revenue: Decimal
    total_purchases: int
    period_purchases: int
    average_transaction: Decimal
    top_agents: List[Dict[str, Any]]


class PaymentProcessor:
    """
    Payment processing with Stripe integration.

    Handles one-time payments, subscriptions, and refunds.
    """

    def __init__(self, session: Session):
        self.session = session
        # In production: import stripe; stripe.api_key = STRIPE_SECRET_KEY

    def create_payment_intent(
        self,
        intent: PaymentIntent
    ) -> Tuple[bool, Optional[str], List[str]]:
        """
        Create Stripe payment intent.

        Args:
            intent: Payment intent data

        Returns:
            Tuple of (success, payment_intent_id, errors)
        """
        errors = []

        try:
            # Get agent
            agent = self.session.query(MarketplaceAgent).filter(
                MarketplaceAgent.id == intent.agent_id
            ).first()

            if not agent:
                return False, None, ["Agent not found"]

            # Validate amount
            if agent.price != intent.amount:
                return False, None, ["Amount mismatch"]

            # In production, create actual Stripe payment intent:
            # stripe_intent = stripe.PaymentIntent.create(
            #     amount=int(intent.amount * 100),  # Convert to cents
            #     currency=intent.currency,
            #     metadata={
            #         "agent_id": intent.agent_id,
            #         "user_id": intent.user_id,
            #     }
            # )

            # For now, simulate payment intent ID
            payment_intent_id = f"pi_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:24]}"

            logger.info(
                f"Created payment intent {payment_intent_id} for "
                f"agent {intent.agent_id}, user {intent.user_id}"
            )

            return True, payment_intent_id, []

        except Exception as e:
            logger.error(f"Error creating payment intent: {e}", exc_info=True)
            errors.append(f"Payment failed: {str(e)}")
            return False, None, errors

    def confirm_payment(
        self,
        payment_intent_id: str,
        agent_id: str,
        user_id: str,
        amount: Decimal,
        currency: str
    ) -> Tuple[bool, Optional[AgentPurchase], List[str]]:
        """
        Confirm payment and create purchase record.

        Args:
            payment_intent_id: Stripe payment intent ID
            agent_id: Agent UUID
            user_id: User UUID
            amount: Payment amount
            currency: Currency code

        Returns:
            Tuple of (success, purchase, errors)
        """
        errors = []

        try:
            # Generate unique transaction ID
            transaction_id = f"txn_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex}"

            # Generate license key
            license_key = self._generate_license_key(agent_id, user_id)

            # Create purchase record
            purchase = AgentPurchase(
                user_id=user_id,
                agent_id=agent_id,
                amount=amount,
                currency=currency,
                transaction_id=transaction_id,
                stripe_payment_intent_id=payment_intent_id,
                pricing_type=PricingType.ONE_TIME.value,
                status=PaymentStatus.COMPLETED.value,
                license_key=license_key
            )

            self.session.add(purchase)

            # Update agent revenue
            agent = self.session.query(MarketplaceAgent).filter(
                MarketplaceAgent.id == agent_id
            ).first()

            if agent:
                agent.total_revenue = (agent.total_revenue or 0) + amount

            self.session.commit()

            logger.info(
                f"Confirmed payment {transaction_id} for agent {agent_id}, "
                f"user {user_id}, amount {amount} {currency}"
            )

            return True, purchase, []

        except Exception as e:
            logger.error(f"Error confirming payment: {e}", exc_info=True)
            self.session.rollback()
            errors.append(f"Payment confirmation failed: {str(e)}")
            return False, None, errors

    def create_subscription(
        self,
        agent_id: str,
        user_id: str,
        pricing_type: PricingModel
    ) -> Tuple[bool, Optional[str], List[str]]:
        """
        Create subscription for monthly/annual pricing.

        Args:
            agent_id: Agent UUID
            user_id: User UUID
            pricing_type: Subscription type (monthly/annual)

        Returns:
            Tuple of (success, subscription_id, errors)
        """
        errors = []

        try:
            agent = self.session.query(MarketplaceAgent).filter(
                MarketplaceAgent.id == agent_id
            ).first()

            if not agent:
                return False, None, ["Agent not found"]

            # In production, create Stripe subscription:
            # subscription = stripe.Subscription.create(
            #     customer=customer_id,
            #     items=[{"price": price_id}],
            #     metadata={"agent_id": agent_id, "user_id": user_id}
            # )

            # Simulate subscription ID
            subscription_id = f"sub_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:24]}"

            # Calculate subscription period
            if pricing_type == PricingModel.MONTHLY:
                period_end = DeterministicClock.utcnow() + timedelta(days=30)
            else:  # Annual
                period_end = DeterministicClock.utcnow() + timedelta(days=365)

            # Create purchase record
            purchase = AgentPurchase(
                user_id=user_id,
                agent_id=agent_id,
                amount=agent.price,
                currency=agent.currency,
                transaction_id=f"txn_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex}",
                pricing_type=pricing_type.value,
                subscription_id=subscription_id,
                subscription_period_start=DeterministicClock.utcnow(),
                subscription_period_end=period_end,
                status=PaymentStatus.COMPLETED.value,
                license_key=self._generate_license_key(agent_id, user_id)
            )

            self.session.add(purchase)
            self.session.commit()

            logger.info(
                f"Created subscription {subscription_id} for agent {agent_id}, "
                f"user {user_id}"
            )

            return True, subscription_id, []

        except Exception as e:
            logger.error(f"Error creating subscription: {e}", exc_info=True)
            errors.append(f"Subscription creation failed: {str(e)}")
            return False, None, errors

    def process_refund(
        self,
        request: RefundRequest
    ) -> Tuple[bool, List[str]]:
        """
        Process refund request.

        Args:
            request: Refund request

        Returns:
            Tuple of (success, errors)
        """
        errors = []

        try:
            purchase = self.session.query(AgentPurchase).filter(
                AgentPurchase.id == request.purchase_id
            ).first()

            if not purchase:
                return False, ["Purchase not found"]

            # Check refund policy (14 days)
            if DeterministicClock.utcnow() - purchase.purchased_at > timedelta(days=14):
                return False, ["Refund period expired (14 days)"]

            # Calculate refund amount
            refund_amount = request.amount or purchase.amount

            if refund_amount > purchase.amount:
                return False, ["Refund amount exceeds purchase amount"]

            # In production, process Stripe refund:
            # stripe.Refund.create(
            #     payment_intent=purchase.stripe_payment_intent_id,
            #     amount=int(refund_amount * 100)
            # )

            # Update purchase
            purchase.status = PaymentStatus.REFUNDED.value
            purchase.refunded_at = DeterministicClock.utcnow()
            purchase.refund_amount = refund_amount
            purchase.refund_reason = request.reason

            # Update agent revenue
            agent = self.session.query(MarketplaceAgent).filter(
                MarketplaceAgent.id == purchase.agent_id
            ).first()

            if agent:
                agent.total_revenue = (agent.total_revenue or 0) - refund_amount

            self.session.commit()

            logger.info(
                f"Processed refund for purchase {request.purchase_id}, "
                f"amount {refund_amount}"
            )

            return True, []

        except Exception as e:
            logger.error(f"Error processing refund: {e}", exc_info=True)
            self.session.rollback()
            errors.append(f"Refund failed: {str(e)}")
            return False, errors

    def _generate_license_key(self, agent_id: str, user_id: str) -> str:
        """Generate unique license key"""
        data = f"{agent_id}{user_id}{deterministic_uuid(__name__, str(DeterministicClock.now())).hex}"
        signature = hmac.new(
            b"secret_key",  # In production, use actual secret
            data.encode(),
            hashlib.sha256
        ).hexdigest()[:16]

        # Format as XXXX-XXXX-XXXX-XXXX
        key = f"{signature[:4]}-{signature[4:8]}-{signature[8:12]}-{signature[12:16]}"
        return key.upper()


class MonetizationManager:
    """
    Main monetization manager.

    Handles payments, subscriptions, and revenue analytics.
    """

    def __init__(self, session: Session):
        self.session = session
        self.payment_processor = PaymentProcessor(session)

    def purchase_agent(
        self,
        agent_id: str,
        user_id: str
    ) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
        """
        Purchase an agent (one-time or subscription).

        Args:
            agent_id: Agent UUID
            user_id: User UUID

        Returns:
            Tuple of (success, purchase_data, errors)
        """
        agent = self.session.query(MarketplaceAgent).filter(
            MarketplaceAgent.id == agent_id
        ).first()

        if not agent:
            return False, None, ["Agent not found"]

        if agent.pricing_type == PricingType.FREE.value:
            # Free agent, just track installation
            return True, {"type": "free", "license_key": None}, []

        # Create payment intent
        intent = PaymentIntent(
            amount=agent.price,
            currency=agent.currency,
            agent_id=agent_id,
            user_id=user_id,
            pricing_type=agent.pricing_type,
            metadata={}
        )

        success, payment_intent_id, errors = self.payment_processor.create_payment_intent(intent)

        if not success:
            return False, None, errors

        return True, {
            "payment_intent_id": payment_intent_id,
            "amount": FinancialDecimal.from_string(agent.price),
            "currency": agent.currency
        }, []

    def get_revenue_stats(
        self,
        agent_id: Optional[str] = None,
        author_id: Optional[str] = None,
        period_days: int = 30
    ) -> RevenueStats:
        """
        Get revenue statistics.

        Args:
            agent_id: Optional agent filter
            author_id: Optional author filter
            period_days: Period for recent stats

        Returns:
            Revenue statistics
        """
        query = self.session.query(AgentPurchase).filter(
            AgentPurchase.status == PaymentStatus.COMPLETED.value
        )

        if agent_id:
            query = query.filter(AgentPurchase.agent_id == agent_id)

        if author_id:
            query = query.join(MarketplaceAgent).filter(
                MarketplaceAgent.author_id == author_id
            )

        # Total revenue
        total_revenue = query.with_entities(
            func.sum(AgentPurchase.amount)
        ).scalar() or Decimal(0)

        # Period revenue
        period_start = DeterministicClock.utcnow() - timedelta(days=period_days)
        period_revenue = query.filter(
            AgentPurchase.purchased_at >= period_start
        ).with_entities(
            func.sum(AgentPurchase.amount)
        ).scalar() or Decimal(0)

        # Counts
        total_purchases = query.count()
        period_purchases = query.filter(
            AgentPurchase.purchased_at >= period_start
        ).count()

        # Average transaction
        avg_transaction = (
            total_revenue / total_purchases
            if total_purchases > 0
            else Decimal(0)
        )

        # Top agents by revenue
        top_agents_query = self.session.query(
            MarketplaceAgent.id,
            MarketplaceAgent.name,
            func.sum(AgentPurchase.amount).label('revenue')
        ).join(AgentPurchase).filter(
            AgentPurchase.status == PaymentStatus.COMPLETED.value
        )

        if author_id:
            top_agents_query = top_agents_query.filter(
                MarketplaceAgent.author_id == author_id
            )

        top_agents = top_agents_query.group_by(
            MarketplaceAgent.id,
            MarketplaceAgent.name
        ).order_by(desc('revenue')).limit(10).all()

        top_agents_list = [
            {
                "agent_id": str(agent_id),
                "agent_name": name,
                "revenue": float(revenue)
            }
            for agent_id, name, revenue in top_agents
        ]

        return RevenueStats(
            total_revenue=total_revenue,
            period_revenue=period_revenue,
            total_purchases=total_purchases,
            period_purchases=period_purchases,
            average_transaction=avg_transaction,
            top_agents=top_agents_list
        )
