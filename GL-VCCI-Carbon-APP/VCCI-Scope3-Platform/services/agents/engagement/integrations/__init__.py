"""
Email service integrations (production-ready stubs).
"""
from .sendgrid import SendGridService
from .mailgun import MailgunService
from .aws_ses import AWSSESService


__all__ = [
    "SendGridService",
    "MailgunService",
    "AWSSESService",
]
