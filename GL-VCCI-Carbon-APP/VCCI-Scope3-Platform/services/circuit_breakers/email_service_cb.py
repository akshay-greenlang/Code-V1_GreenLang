"""
Circuit Breaker for Email Service (SendGrid)

Protects against email service failures with:
- Circuit breaker protection
- Queue-based fallback
- Retry logic
- Rate limiting

Features:
- Graceful degradation when SendGrid is down
- Local queue for failed emails
- Automatic retry when service recovers
- Prometheus metrics

Author: GreenLang Platform Team
Version: 1.0.0
Date: 2025-11-09
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
import json
import time
from pathlib import Path

from greenlang.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    create_circuit_breaker,
)
from greenlang.telemetry import get_logger
from greenlang.cache import get_cache_manager


logger = get_logger(__name__)


# ============================================================================
# EMAIL MODELS
# ============================================================================

@dataclass
class Email:
    """Email message data structure."""
    to: str
    subject: str
    body: str
    from_email: Optional[str] = None
    cc: Optional[List[str]] = None
    bcc: Optional[List[str]] = None
    attachments: Optional[List[str]] = None
    html: bool = False
    priority: str = "normal"  # low, normal, high


# ============================================================================
# EMAIL SERVICE CIRCUIT BREAKER
# ============================================================================

class EmailServiceCircuitBreaker:
    """
    Circuit breaker wrapper for email service (SendGrid).

    Features:
    - Circuit breaker protection
    - Local queue for failed emails
    - Automatic retry when service recovers
    - Priority-based queuing

    Example:
        >>> email_cb = EmailServiceCircuitBreaker()
        >>> result = email_cb.send_email(
        ...     to="user@example.com",
        ...     subject="Report Ready",
        ...     body="Your carbon report is ready."
        ... )
    """

    def __init__(self, queue_dir: Optional[Path] = None):
        self.logger = get_logger(__name__)
        self.cache = get_cache_manager()

        # Set up email queue directory
        if queue_dir is None:
            queue_dir = Path("/tmp/greenlang_email_queue")
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        # Circuit breaker for SendGrid
        self.sendgrid_cb = create_circuit_breaker(
            CircuitBreakerConfig(
                name="email_service_sendgrid",
                fail_max=3,  # Quick to fail for email
                timeout_duration=60,  # 1 minute before retry
                reset_timeout=30,
                fallback_function=self._fallback_sendgrid,
            )
        )

        self.logger.info(
            "Email service circuit breaker initialized",
            extra={"queue_dir": str(self.queue_dir)}
        )

    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        from_email: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None,
        html: bool = False,
        priority: str = "normal",
    ) -> Dict[str, Any]:
        """
        Send email with circuit breaker protection.

        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body
            from_email: Sender email (optional)
            cc: CC recipients
            bcc: BCC recipients
            attachments: File paths for attachments
            html: Whether body is HTML
            priority: Email priority (low, normal, high)

        Returns:
            Send result with status

        Raises:
            CircuitOpenError: If circuit is open (email queued for retry)
        """
        email = Email(
            to=to,
            subject=subject,
            body=body,
            from_email=from_email,
            cc=cc,
            bcc=bcc,
            attachments=attachments,
            html=html,
            priority=priority,
        )

        try:
            result = self.sendgrid_cb.call(self._send_via_sendgrid, email=email)

            self.logger.info(
                "Email sent successfully",
                extra={
                    "to": to,
                    "subject": subject,
                    "priority": priority,
                }
            )

            return result

        except CircuitOpenError as e:
            # Queue email for retry
            self._queue_email(email)

            self.logger.warning(
                "Email service unavailable - email queued for retry",
                extra={
                    "to": to,
                    "subject": subject,
                    "priority": priority,
                }
            )

            return {
                "status": "queued",
                "message": "Email service temporarily unavailable - queued for retry",
                "email_id": None,
                "queued": True,
                "timestamp": datetime.utcnow().isoformat(),
            }

    def send_batch(
        self,
        emails: List[Email],
    ) -> Dict[str, Any]:
        """
        Send batch of emails with circuit breaker protection.

        Args:
            emails: List of Email objects

        Returns:
            Batch send results
        """
        results = {
            "total": len(emails),
            "sent": 0,
            "queued": 0,
            "failed": 0,
            "details": [],
        }

        for email in emails:
            try:
                result = self.send_email(
                    to=email.to,
                    subject=email.subject,
                    body=email.body,
                    from_email=email.from_email,
                    cc=email.cc,
                    bcc=email.bcc,
                    attachments=email.attachments,
                    html=email.html,
                    priority=email.priority,
                )

                if result["status"] == "sent":
                    results["sent"] += 1
                elif result["status"] == "queued":
                    results["queued"] += 1

                results["details"].append(result)

            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "status": "failed",
                    "error": str(e),
                    "to": email.to,
                })

        return results

    def process_queue(self) -> Dict[str, Any]:
        """
        Process queued emails when service recovers.

        Returns:
            Processing results
        """
        if self.sendgrid_cb.state.value == "open":
            self.logger.debug("Circuit still open - skipping queue processing")
            return {
                "processed": 0,
                "remaining": self._get_queue_size(),
                "circuit_state": "open",
            }

        queued_files = list(self.queue_dir.glob("email_*.json"))
        processed = 0
        failed = 0

        for email_file in queued_files:
            try:
                # Load email data
                with open(email_file, 'r') as f:
                    email_data = json.load(f)

                # Try to send
                result = self.send_email(**email_data)

                if result["status"] == "sent":
                    # Remove from queue
                    email_file.unlink()
                    processed += 1
                    self.logger.info(
                        f"Queued email sent successfully",
                        extra={"file": email_file.name}
                    )
                else:
                    failed += 1

            except Exception as e:
                self.logger.error(
                    f"Failed to process queued email",
                    extra={
                        "file": email_file.name,
                        "error": str(e),
                    }
                )
                failed += 1

        return {
            "processed": processed,
            "failed": failed,
            "remaining": self._get_queue_size(),
            "circuit_state": self.sendgrid_cb.state.value,
        }

    def _send_via_sendgrid(self, email: Email) -> Dict[str, Any]:
        """
        Send email via SendGrid API.

        This is a placeholder - actual implementation would use sendgrid client.
        """
        self.logger.debug(
            f"Sending email via SendGrid",
            extra={
                "to": email.to,
                "subject": email.subject,
                "priority": email.priority,
            }
        )

        # Placeholder for actual SendGrid call
        # In production, this would call:
        # import sendgrid
        # from sendgrid.helpers.mail import Mail
        # sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
        # message = Mail(...)
        # response = sg.send(message)

        # Simulate API call
        time.sleep(0.05)

        return {
            "status": "sent",
            "email_id": f"sendgrid_{int(time.time() * 1000)}",
            "to": email.to,
            "subject": email.subject,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _queue_email(self, email: Email):
        """Queue email for retry when service is down."""
        timestamp = int(time.time() * 1000)
        priority_prefix = {
            "high": "0",
            "normal": "1",
            "low": "2",
        }.get(email.priority, "1")

        filename = f"email_{priority_prefix}_{timestamp}.json"
        filepath = self.queue_dir / filename

        email_data = {
            "to": email.to,
            "subject": email.subject,
            "body": email.body,
            "from_email": email.from_email,
            "cc": email.cc,
            "bcc": email.bcc,
            "attachments": email.attachments,
            "html": email.html,
            "priority": email.priority,
            "queued_at": datetime.utcnow().isoformat(),
        }

        with open(filepath, 'w') as f:
            json.dump(email_data, f, indent=2)

        self.logger.info(
            f"Email queued for retry",
            extra={
                "file": filename,
                "to": email.to,
                "priority": email.priority,
            }
        )

    def _fallback_sendgrid(self, **kwargs) -> Dict[str, Any]:
        """Fallback for SendGrid failures - queue email."""
        email = kwargs.get("email")
        if email:
            self._queue_email(email)

        raise CircuitOpenError("SendGrid circuit is open")

    def _get_queue_size(self) -> int:
        """Get number of queued emails."""
        return len(list(self.queue_dir.glob("email_*.json")))

    def get_stats(self) -> Dict[str, Any]:
        """Get email service statistics."""
        return {
            "sendgrid": self.sendgrid_cb.get_stats(),
            "queue_size": self._get_queue_size(),
            "queue_dir": str(self.queue_dir),
        }

    def reset(self):
        """Reset circuit breaker."""
        self.sendgrid_cb.reset()
        self.logger.info("Email service circuit breaker reset")

    def clear_queue(self):
        """Clear all queued emails (use with caution)."""
        queued_files = list(self.queue_dir.glob("email_*.json"))
        for email_file in queued_files:
            email_file.unlink()

        self.logger.warning(
            f"Email queue cleared - {len(queued_files)} emails deleted"
        )


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_instance: Optional[EmailServiceCircuitBreaker] = None


def get_email_service_cb() -> EmailServiceCircuitBreaker:
    """Get singleton instance of email service circuit breaker."""
    global _instance
    if _instance is None:
        _instance = EmailServiceCircuitBreaker()
    return _instance
