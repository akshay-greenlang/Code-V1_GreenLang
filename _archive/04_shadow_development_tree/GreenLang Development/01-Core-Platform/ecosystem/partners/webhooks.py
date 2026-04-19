# -*- coding: utf-8 -*-
"""
Webhook System for GreenLang Partners

This module provides comprehensive webhook functionality for delivering
real-time events to partner applications.

Features:
- Multiple webhook event types
- HMAC signature verification
- Automatic retry with exponential backoff
- Webhook delivery logging
- Webhook testing and validation
- Rate limiting and security
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import asyncio
import hashlib
import hmac
import json
import logging
import time
from uuid import uuid4

import aiohttp
from sqlalchemy import Column, String, Integer, DateTime, Boolean, JSON, ForeignKey, Index, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship
from pydantic import BaseModel, HttpUrl, validator
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from greenlang.utilities.determinism import deterministic_uuid, DeterministicClock

# Setup
logger = logging.getLogger(__name__)
Base = declarative_base()


class WebhookEventType(str, Enum):
    """Webhook event types"""
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    AGENT_RESULT = "agent.result"
    USAGE_LIMIT_REACHED = "usage.limit_reached"
    BILLING_INVOICE_CREATED = "billing.invoice_created"
    PARTNER_TIER_CHANGED = "partner.tier_changed"
    API_KEY_CREATED = "api_key.created"
    API_KEY_REVOKED = "api_key.revoked"


class WebhookStatus(str, Enum):
    """Webhook status"""
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    FAILED = "FAILED"


class DeliveryStatus(str, Enum):
    """Webhook delivery status"""
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RETRYING = "RETRYING"


# Database Models
class WebhookModel(Base):
    """Webhook configuration model"""
    __tablename__ = "webhooks"

    id = Column(String, primary_key=True)
    partner_id = Column(String, ForeignKey("partners.id"), nullable=False)
    url = Column(String, nullable=False)
    secret = Column(String, nullable=False)  # For HMAC signing
    status = Column(String, default=WebhookStatus.ACTIVE)
    event_types = Column(JSON, default=list)  # List of subscribed events
    description = Column(String, nullable=True)

    # Retry configuration
    max_retries = Column(Integer, default=3)
    retry_delay_seconds = Column(Integer, default=60)  # Initial delay
    timeout_seconds = Column(Integer, default=10)

    # Statistics
    total_deliveries = Column(Integer, default=0)
    successful_deliveries = Column(Integer, default=0)
    failed_deliveries = Column(Integer, default=0)
    last_delivery_at = Column(DateTime, nullable=True)
    last_success_at = Column(DateTime, nullable=True)
    last_failure_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    deliveries = relationship("WebhookDeliveryModel", back_populates="webhook", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_webhook_partner', 'partner_id'),
        Index('idx_webhook_status', 'status'),
    )


class WebhookDeliveryModel(Base):
    """Webhook delivery log model"""
    __tablename__ = "webhook_deliveries"

    id = Column(String, primary_key=True)
    webhook_id = Column(String, ForeignKey("webhooks.id"), nullable=False)
    event_type = Column(String, nullable=False)
    event_id = Column(String, nullable=False)  # Idempotency

    # Request details
    url = Column(String, nullable=False)
    payload = Column(JSON, nullable=False)
    headers = Column(JSON, nullable=False)

    # Response details
    status_code = Column(Integer, nullable=True)
    response_body = Column(Text, nullable=True)
    response_time_ms = Column(Integer, nullable=True)

    # Delivery tracking
    status = Column(String, default=DeliveryStatus.PENDING)
    attempt_count = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)
    next_retry_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    webhook = relationship("WebhookModel", back_populates="deliveries")

    # Indexes
    __table_args__ = (
        Index('idx_delivery_webhook', 'webhook_id'),
        Index('idx_delivery_status', 'status'),
        Index('idx_delivery_event', 'event_id'),
        Index('idx_delivery_retry', 'status', 'next_retry_at'),
    )


# Pydantic Models
class WebhookCreate(BaseModel):
    """Schema for creating a webhook"""
    url: HttpUrl
    event_types: List[WebhookEventType]
    description: Optional[str] = None
    max_retries: int = 3
    timeout_seconds: int = 10

    @validator('max_retries')
    def validate_retries(cls, v):
        if v < 0 or v > 10:
            raise ValueError('max_retries must be between 0 and 10')
        return v

    @validator('timeout_seconds')
    def validate_timeout(cls, v):
        if v < 1 or v > 30:
            raise ValueError('timeout_seconds must be between 1 and 30')
        return v


class WebhookUpdate(BaseModel):
    """Schema for updating a webhook"""
    url: Optional[HttpUrl] = None
    event_types: Optional[List[WebhookEventType]] = None
    description: Optional[str] = None
    status: Optional[WebhookStatus] = None
    max_retries: Optional[int] = None
    timeout_seconds: Optional[int] = None


class WebhookResponse(BaseModel):
    """Schema for webhook response"""
    id: str
    partner_id: str
    url: str
    status: WebhookStatus
    event_types: List[str]
    description: Optional[str]
    max_retries: int
    timeout_seconds: int
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int
    last_delivery_at: Optional[datetime]
    last_success_at: Optional[datetime]
    last_failure_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class WebhookWithSecret(WebhookResponse):
    """Schema for webhook with secret (only on creation)"""
    secret: str


class WebhookDeliveryResponse(BaseModel):
    """Schema for webhook delivery response"""
    id: str
    webhook_id: str
    event_type: str
    event_id: str
    url: str
    status: DeliveryStatus
    status_code: Optional[int]
    response_time_ms: Optional[int]
    attempt_count: int
    max_attempts: int
    next_retry_at: Optional[datetime]
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class WebhookTestRequest(BaseModel):
    """Schema for testing a webhook"""
    event_type: WebhookEventType = WebhookEventType.WORKFLOW_COMPLETED
    custom_payload: Optional[Dict[str, Any]] = None


# Dataclasses
@dataclass
class WebhookEvent:
    """Webhook event data"""
    event_type: WebhookEventType
    timestamp: datetime
    partner_id: str
    data: Dict[str, Any]
    event_id: str = field(default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))))


@dataclass
class WebhookDeliveryResult:
    """Result of webhook delivery attempt"""
    success: bool
    status_code: Optional[int]
    response_body: Optional[str]
    response_time_ms: int
    error_message: Optional[str] = None


# Webhook Manager
class WebhookManager:
    """
    Manages webhook delivery, retries, and logging
    """

    def __init__(self, db_session: Session):
        self.db = db_session

    def generate_signature(self, payload: bytes, secret: str) -> str:
        """
        Generate HMAC signature for webhook payload

        Args:
            payload: JSON payload as bytes
            secret: Webhook secret key

        Returns:
            Signature in format "sha256=<hex_digest>"
        """
        signature = hmac.new(
            secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"

    def verify_signature(self, payload: bytes, signature: str, secret: str) -> bool:
        """
        Verify webhook signature

        Args:
            payload: JSON payload as bytes
            signature: Signature from header
            secret: Webhook secret key

        Returns:
            True if signature is valid
        """
        expected_signature = self.generate_signature(payload, secret)
        return hmac.compare_digest(signature, expected_signature)

    async def deliver_webhook(
        self,
        webhook: WebhookModel,
        event: WebhookEvent
    ) -> WebhookDeliveryResult:
        """
        Deliver webhook to partner endpoint

        Args:
            webhook: Webhook configuration
            event: Event to deliver

        Returns:
            Delivery result
        """
        # Prepare payload
        payload = {
            "event": event.event_type.value,
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "partner_id": event.partner_id,
            "data": event.data
        }

        payload_bytes = json.dumps(payload).encode('utf-8')
        signature = self.generate_signature(payload_bytes, webhook.secret)

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "X-GreenLang-Signature": signature,
            "X-GreenLang-Event": event.event_type.value,
            "X-GreenLang-Event-ID": event.event_id,
            "X-GreenLang-Timestamp": str(int(event.timestamp.timestamp())),
            "User-Agent": "GreenLang-Webhooks/1.0"
        }

        start_time = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook.url,
                    data=payload_bytes,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=webhook.timeout_seconds)
                ) as response:
                    response_time_ms = int((time.time() - start_time) * 1000)
                    response_body = await response.text()

                    if 200 <= response.status < 300:
                        return WebhookDeliveryResult(
                            success=True,
                            status_code=response.status,
                            response_body=response_body[:1000],  # Limit storage
                            response_time_ms=response_time_ms
                        )
                    else:
                        return WebhookDeliveryResult(
                            success=False,
                            status_code=response.status,
                            response_body=response_body[:1000],
                            response_time_ms=response_time_ms,
                            error_message=f"HTTP {response.status}: {response_body[:200]}"
                        )

        except asyncio.TimeoutError:
            response_time_ms = int((time.time() - start_time) * 1000)
            return WebhookDeliveryResult(
                success=False,
                status_code=None,
                response_body=None,
                response_time_ms=response_time_ms,
                error_message=f"Timeout after {webhook.timeout_seconds}s"
            )
        except aiohttp.ClientError as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            return WebhookDeliveryResult(
                success=False,
                status_code=None,
                response_body=None,
                response_time_ms=response_time_ms,
                error_message=f"Connection error: {str(e)}"
            )
        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Unexpected error delivering webhook: {e}")
            return WebhookDeliveryResult(
                success=False,
                status_code=None,
                response_body=None,
                response_time_ms=response_time_ms,
                error_message=f"Unexpected error: {str(e)}"
            )

    async def send_event(
        self,
        partner_id: str,
        event: WebhookEvent,
        background_tasks: Optional[BackgroundTasks] = None
    ):
        """
        Send event to all webhooks subscribed to this event type

        Args:
            partner_id: Partner ID
            event: Event to send
            background_tasks: FastAPI background tasks for async delivery
        """
        # Find all active webhooks for this partner and event type
        webhooks = self.db.query(WebhookModel).filter(
            WebhookModel.partner_id == partner_id,
            WebhookModel.status == WebhookStatus.ACTIVE
        ).all()

        for webhook in webhooks:
            # Check if webhook subscribes to this event type
            if event.event_type.value not in webhook.event_types:
                continue

            # Create delivery record
            delivery = WebhookDeliveryModel(
                id=f"del_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:16]}",
                webhook_id=webhook.id,
                event_type=event.event_type.value,
                event_id=event.event_id,
                url=webhook.url,
                payload={
                    "event": event.event_type.value,
                    "event_id": event.event_id,
                    "timestamp": event.timestamp.isoformat(),
                    "partner_id": event.partner_id,
                    "data": event.data
                },
                headers={
                    "Content-Type": "application/json",
                    "X-GreenLang-Event": event.event_type.value
                },
                max_attempts=webhook.max_retries + 1  # Initial + retries
            )

            self.db.add(delivery)
            self.db.commit()

            # Deliver webhook (async)
            if background_tasks:
                background_tasks.add_task(self.process_delivery, delivery.id)
            else:
                asyncio.create_task(self.process_delivery(delivery.id))

    async def process_delivery(self, delivery_id: str):
        """
        Process webhook delivery with retries

        Args:
            delivery_id: Delivery record ID
        """
        delivery = self.db.query(WebhookDeliveryModel).filter(
            WebhookDeliveryModel.id == delivery_id
        ).first()

        if not delivery:
            logger.error(f"Delivery {delivery_id} not found")
            return

        webhook = self.db.query(WebhookModel).filter(
            WebhookModel.id == delivery.webhook_id
        ).first()

        if not webhook:
            logger.error(f"Webhook {delivery.webhook_id} not found")
            return

        # Reconstruct event
        event = WebhookEvent(
            event_type=WebhookEventType(delivery.event_type),
            timestamp=datetime.fromisoformat(delivery.payload["timestamp"]),
            partner_id=delivery.payload["partner_id"],
            data=delivery.payload["data"],
            event_id=delivery.event_id
        )

        # Attempt delivery
        delivery.attempt_count += 1
        delivery.status = DeliveryStatus.RETRYING if delivery.attempt_count > 1 else DeliveryStatus.PENDING
        self.db.commit()

        result = await self.deliver_webhook(webhook, event)

        # Update delivery record
        delivery.status_code = result.status_code
        delivery.response_body = result.response_body
        delivery.response_time_ms = result.response_time_ms

        if result.success:
            # Success
            delivery.status = DeliveryStatus.SUCCESS
            delivery.completed_at = DeterministicClock.utcnow()

            webhook.total_deliveries += 1
            webhook.successful_deliveries += 1
            webhook.last_delivery_at = DeterministicClock.utcnow()
            webhook.last_success_at = DeterministicClock.utcnow()

            logger.info(f"Webhook delivered successfully: {delivery_id}")
        else:
            # Failed
            delivery.error_message = result.error_message

            if delivery.attempt_count >= delivery.max_attempts:
                # Max retries reached
                delivery.status = DeliveryStatus.FAILED
                delivery.completed_at = DeterministicClock.utcnow()

                webhook.total_deliveries += 1
                webhook.failed_deliveries += 1
                webhook.last_delivery_at = DeterministicClock.utcnow()
                webhook.last_failure_at = DeterministicClock.utcnow()

                logger.error(f"Webhook delivery failed after {delivery.attempt_count} attempts: {delivery_id}")
            else:
                # Schedule retry with exponential backoff
                retry_delay = webhook.retry_delay_seconds * (2 ** (delivery.attempt_count - 1))
                delivery.next_retry_at = DeterministicClock.utcnow() + timedelta(seconds=retry_delay)
                delivery.status = DeliveryStatus.RETRYING

                logger.warning(
                    f"Webhook delivery failed (attempt {delivery.attempt_count}/{delivery.max_attempts}), "
                    f"retrying in {retry_delay}s: {delivery_id}"
                )

                # Schedule retry
                asyncio.create_task(self.retry_delivery(delivery_id, retry_delay))

        self.db.commit()

    async def retry_delivery(self, delivery_id: str, delay_seconds: int):
        """
        Retry webhook delivery after delay

        Args:
            delivery_id: Delivery record ID
            delay_seconds: Delay before retry
        """
        await asyncio.sleep(delay_seconds)
        await self.process_delivery(delivery_id)

    async def process_pending_retries(self):
        """
        Process all pending retries (for scheduled task)
        """
        now = DeterministicClock.utcnow()
        pending_deliveries = self.db.query(WebhookDeliveryModel).filter(
            WebhookDeliveryModel.status == DeliveryStatus.RETRYING,
            WebhookDeliveryModel.next_retry_at <= now
        ).all()

        for delivery in pending_deliveries:
            asyncio.create_task(self.process_delivery(delivery.id))


# FastAPI Integration
def create_webhook_app():
    """Create FastAPI app with webhook endpoints"""
    from .api import get_db, get_current_partner_from_token, PartnerModel
    import secrets

    app = FastAPI(title="GreenLang Webhooks API", version="1.0.0")

    @app.post("/api/partners/{partner_id}/webhooks", response_model=WebhookWithSecret, status_code=status.HTTP_201_CREATED)
    async def create_webhook(
        partner_id: str,
        webhook_data: WebhookCreate,
        current_partner: PartnerModel = Depends(get_current_partner_from_token),
        db: Session = Depends(get_db)
    ):
        """
        Register a new webhook

        Creates a webhook that will receive events for the specified event types.
        Returns the webhook configuration including the secret for signature verification.
        """
        if partner_id != current_partner.id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Generate webhook secret
        webhook_secret = secrets.token_hex(32)

        # Create webhook
        webhook = WebhookModel(
            id=f"wh_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:16]}",
            partner_id=partner_id,
            url=str(webhook_data.url),
            secret=webhook_secret,
            event_types=[e.value for e in webhook_data.event_types],
            description=webhook_data.description,
            max_retries=webhook_data.max_retries,
            timeout_seconds=webhook_data.timeout_seconds
        )

        db.add(webhook)
        db.commit()
        db.refresh(webhook)

        logger.info(f"Webhook created for partner {partner_id}: {webhook.id}")

        return WebhookWithSecret(
            id=webhook.id,
            partner_id=webhook.partner_id,
            url=webhook.url,
            status=webhook.status,
            event_types=webhook.event_types,
            description=webhook.description,
            max_retries=webhook.max_retries,
            timeout_seconds=webhook.timeout_seconds,
            total_deliveries=webhook.total_deliveries,
            successful_deliveries=webhook.successful_deliveries,
            failed_deliveries=webhook.failed_deliveries,
            last_delivery_at=webhook.last_delivery_at,
            last_success_at=webhook.last_success_at,
            last_failure_at=webhook.last_failure_at,
            created_at=webhook.created_at,
            secret=webhook_secret  # Only shown on creation
        )

    @app.get("/api/partners/{partner_id}/webhooks", response_model=List[WebhookResponse])
    async def list_webhooks(
        partner_id: str,
        current_partner: PartnerModel = Depends(get_current_partner_from_token),
        db: Session = Depends(get_db)
    ):
        """List all webhooks for a partner"""
        if partner_id != current_partner.id:
            raise HTTPException(status_code=403, detail="Access denied")

        webhooks = db.query(WebhookModel).filter(
            WebhookModel.partner_id == partner_id
        ).order_by(WebhookModel.created_at.desc()).all()

        return webhooks

    @app.get("/api/partners/{partner_id}/webhooks/{webhook_id}", response_model=WebhookResponse)
    async def get_webhook(
        partner_id: str,
        webhook_id: str,
        current_partner: PartnerModel = Depends(get_current_partner_from_token),
        db: Session = Depends(get_db)
    ):
        """Get webhook details"""
        if partner_id != current_partner.id:
            raise HTTPException(status_code=403, detail="Access denied")

        webhook = db.query(WebhookModel).filter(
            WebhookModel.id == webhook_id,
            WebhookModel.partner_id == partner_id
        ).first()

        if not webhook:
            raise HTTPException(status_code=404, detail="Webhook not found")

        return webhook

    @app.put("/api/partners/{partner_id}/webhooks/{webhook_id}", response_model=WebhookResponse)
    async def update_webhook(
        partner_id: str,
        webhook_id: str,
        webhook_update: WebhookUpdate,
        current_partner: PartnerModel = Depends(get_current_partner_from_token),
        db: Session = Depends(get_db)
    ):
        """Update webhook configuration"""
        if partner_id != current_partner.id:
            raise HTTPException(status_code=403, detail="Access denied")

        webhook = db.query(WebhookModel).filter(
            WebhookModel.id == webhook_id,
            WebhookModel.partner_id == partner_id
        ).first()

        if not webhook:
            raise HTTPException(status_code=404, detail="Webhook not found")

        # Update fields
        if webhook_update.url:
            webhook.url = str(webhook_update.url)
        if webhook_update.event_types:
            webhook.event_types = [e.value for e in webhook_update.event_types]
        if webhook_update.description is not None:
            webhook.description = webhook_update.description
        if webhook_update.status:
            webhook.status = webhook_update.status
        if webhook_update.max_retries is not None:
            webhook.max_retries = webhook_update.max_retries
        if webhook_update.timeout_seconds is not None:
            webhook.timeout_seconds = webhook_update.timeout_seconds

        webhook.updated_at = DeterministicClock.utcnow()
        db.commit()
        db.refresh(webhook)

        return webhook

    @app.delete("/api/partners/{partner_id}/webhooks/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_webhook(
        partner_id: str,
        webhook_id: str,
        current_partner: PartnerModel = Depends(get_current_partner_from_token),
        db: Session = Depends(get_db)
    ):
        """Delete a webhook"""
        if partner_id != current_partner.id:
            raise HTTPException(status_code=403, detail="Access denied")

        webhook = db.query(WebhookModel).filter(
            WebhookModel.id == webhook_id,
            WebhookModel.partner_id == partner_id
        ).first()

        if not webhook:
            raise HTTPException(status_code=404, detail="Webhook not found")

        db.delete(webhook)
        db.commit()

        logger.info(f"Webhook deleted: {webhook_id}")

        return None

    @app.post("/api/partners/{partner_id}/webhooks/{webhook_id}/test")
    async def test_webhook(
        partner_id: str,
        webhook_id: str,
        test_data: WebhookTestRequest,
        background_tasks: BackgroundTasks,
        current_partner: PartnerModel = Depends(get_current_partner_from_token),
        db: Session = Depends(get_db)
    ):
        """
        Test a webhook

        Sends a test event to the webhook endpoint to verify configuration.
        """
        if partner_id != current_partner.id:
            raise HTTPException(status_code=403, detail="Access denied")

        webhook = db.query(WebhookModel).filter(
            WebhookModel.id == webhook_id,
            WebhookModel.partner_id == partner_id
        ).first()

        if not webhook:
            raise HTTPException(status_code=404, detail="Webhook not found")

        # Create test event
        test_payload = test_data.custom_payload or {
            "test": True,
            "message": "This is a test webhook delivery",
            "webhook_id": webhook_id
        }

        event = WebhookEvent(
            event_type=test_data.event_type,
            timestamp=DeterministicClock.utcnow(),
            partner_id=partner_id,
            data=test_payload
        )

        # Send event
        manager = WebhookManager(db)
        await manager.send_event(partner_id, event, background_tasks)

        return {
            "message": "Test webhook sent",
            "event_id": event.event_id,
            "event_type": event.event_type.value
        }

    @app.get("/api/partners/{partner_id}/webhooks/{webhook_id}/logs", response_model=List[WebhookDeliveryResponse])
    async def get_webhook_logs(
        partner_id: str,
        webhook_id: str,
        limit: int = 50,
        current_partner: PartnerModel = Depends(get_current_partner_from_token),
        db: Session = Depends(get_db)
    ):
        """
        Get webhook delivery logs

        Returns recent delivery attempts with status and error information.
        """
        if partner_id != current_partner.id:
            raise HTTPException(status_code=403, detail="Access denied")

        webhook = db.query(WebhookModel).filter(
            WebhookModel.id == webhook_id,
            WebhookModel.partner_id == partner_id
        ).first()

        if not webhook:
            raise HTTPException(status_code=404, detail="Webhook not found")

        deliveries = db.query(WebhookDeliveryModel).filter(
            WebhookDeliveryModel.webhook_id == webhook_id
        ).order_by(WebhookDeliveryModel.created_at.desc()).limit(limit).all()

        return deliveries

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_webhook_app()
    uvicorn.run(app, host="0.0.0.0", port=8001)
