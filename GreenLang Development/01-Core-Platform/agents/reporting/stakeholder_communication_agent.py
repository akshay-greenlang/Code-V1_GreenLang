# -*- coding: utf-8 -*-
"""
GL-REP-X-005: Stakeholder Communication Agent
=============================================

Generates stakeholder-specific sustainability communications. RECOMMENDATION
PATH agent with AI-enhanced content generation for different audiences.

Capabilities:
    - Audience-specific content adaptation
    - Multi-channel communication support
    - Key message extraction
    - Visual asset generation briefs
    - Executive summary generation
    - FAQ generation

AI Enhancement:
    - Narrative adaptation for different audiences
    - Key message generation
    - Content optimization

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism import DeterministicClock, deterministic_uuid

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class Audience(str, Enum):
    """Target audiences for communications."""
    INVESTORS = "investors"
    EMPLOYEES = "employees"
    CUSTOMERS = "customers"
    REGULATORS = "regulators"
    MEDIA = "media"
    GENERAL_PUBLIC = "general_public"
    SUPPLIERS = "suppliers"
    COMMUNITIES = "communities"


class ContentFormat(str, Enum):
    """Content format types."""
    EXECUTIVE_SUMMARY = "executive_summary"
    PRESS_RELEASE = "press_release"
    INTERNAL_MEMO = "internal_memo"
    INVESTOR_BRIEF = "investor_brief"
    SOCIAL_MEDIA = "social_media"
    FAQ = "faq"
    INFOGRAPHIC_BRIEF = "infographic_brief"
    PRESENTATION = "presentation"


class ContentTone(str, Enum):
    """Communication tone."""
    FORMAL = "formal"
    PROFESSIONAL = "professional"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class KeyMessage(BaseModel):
    """Key message for communication."""

    message_id: str = Field(
        default_factory=lambda: deterministic_uuid("msg"),
        description="Unique identifier"
    )
    topic: str = Field(...)
    headline: str = Field(...)
    supporting_points: List[str] = Field(default_factory=list)
    data_points: Dict[str, Any] = Field(default_factory=dict)
    suitable_audiences: List[Audience] = Field(default_factory=list)


class CommunicationContent(BaseModel):
    """Generated communication content."""

    content_id: str = Field(
        default_factory=lambda: deterministic_uuid("content"),
        description="Unique identifier"
    )
    format: ContentFormat = Field(...)
    audience: Audience = Field(...)
    tone: ContentTone = Field(...)

    # Content
    title: str = Field(...)
    body: str = Field(...)
    key_messages: List[str] = Field(default_factory=list)
    call_to_action: Optional[str] = Field(None)

    # Metadata
    word_count: int = Field(default=0)
    reading_time_minutes: int = Field(default=0)

    # Supporting elements
    suggested_visuals: List[str] = Field(default_factory=list)
    data_highlights: Dict[str, Any] = Field(default_factory=dict)


class StakeholderCommunicationPack(BaseModel):
    """Complete communication pack for stakeholders."""

    pack_id: str = Field(
        default_factory=lambda: deterministic_uuid("comm_pack"),
        description="Unique identifier"
    )
    organization_name: str = Field(...)
    reporting_period: str = Field(...)
    created_date: date = Field(
        default_factory=lambda: DeterministicClock.now().date()
    )

    # Key messages
    key_messages: List[KeyMessage] = Field(default_factory=list)

    # Content by audience
    communications: List[CommunicationContent] = Field(default_factory=list)

    # Summary
    audiences_covered: List[str] = Field(default_factory=list)
    formats_generated: List[str] = Field(default_factory=list)

    # Provenance
    provenance_hash: str = Field(default="")

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash."""
        content = {
            "organization_name": self.organization_name,
            "reporting_period": self.reporting_period,
            "audiences_covered": self.audiences_covered,
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()


class StakeholderCommInput(BaseModel):
    """Input for stakeholder communication operations."""

    action: str = Field(
        ...,
        description="Action: generate_pack, create_content, extract_messages"
    )
    organization_name: Optional[str] = Field(None)
    reporting_period: Optional[str] = Field(None)
    sustainability_data: Optional[Dict[str, Any]] = Field(None)
    target_audiences: Optional[List[Audience]] = Field(None)
    formats: Optional[List[ContentFormat]] = Field(None)


class StakeholderCommOutput(BaseModel):
    """Output from stakeholder communication operations."""

    success: bool = Field(...)
    action: str = Field(...)
    communication_pack: Optional[StakeholderCommunicationPack] = Field(None)
    content: Optional[CommunicationContent] = Field(None)
    key_messages: Optional[List[KeyMessage]] = Field(None)
    error: Optional[str] = Field(None)
    provenance_hash: str = Field(default="")


# =============================================================================
# STAKEHOLDER COMMUNICATION AGENT
# =============================================================================


class StakeholderCommunicationAgent(BaseAgent):
    """
    GL-REP-X-005: Stakeholder Communication Agent

    Generates stakeholder-specific sustainability communications.
    RECOMMENDATION PATH agent with AI-enhanced content generation.

    Capabilities:
    - Audience-specific content adaptation
    - Multi-format content generation
    - Key message extraction

    Usage:
        agent = StakeholderCommunicationAgent()
        result = agent.run({
            'action': 'generate_pack',
            'organization_name': 'Company',
            'sustainability_data': {...}
        })
    """

    AGENT_ID = "GL-REP-X-005"
    AGENT_NAME = "Stakeholder Communication Agent"
    VERSION = "1.0.0"

    category = AgentCategory.RECOMMENDATION
    metadata = AgentMetadata(
        name=AGENT_NAME,
        category=AgentCategory.RECOMMENDATION,
        uses_chat_session=False,
        uses_rag=False,
        uses_tools=False,
        critical_for_compliance=False,
        description="Stakeholder communication generation"
    )

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize Stakeholder Communication Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Stakeholder communication agent",
                version=self.VERSION,
                parameters={}
            )

        super().__init__(config)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute stakeholder communication operation."""
        import time
        start_time = time.time()

        try:
            agent_input = StakeholderCommInput(**input_data)

            action_handlers = {
                "generate_pack": self._handle_generate_pack,
                "create_content": self._handle_create_content,
                "extract_messages": self._handle_extract_messages,
            }

            handler = action_handlers.get(agent_input.action)
            if not handler:
                raise ValueError(f"Unknown action: {agent_input.action}")

            output = handler(agent_input)
            output.provenance_hash = hashlib.sha256(
                json.dumps({"action": agent_input.action}, sort_keys=True).encode()
            ).hexdigest()

            return AgentResult(
                success=output.success,
                data=output.model_dump(),
            )

        except Exception as e:
            logger.error(f"Stakeholder communication failed: {str(e)}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _handle_generate_pack(
        self,
        input_data: StakeholderCommInput
    ) -> StakeholderCommOutput:
        """Generate complete communication pack."""
        if not input_data.organization_name:
            return StakeholderCommOutput(
                success=False,
                action="generate_pack",
                error="organization_name required",
            )

        data = input_data.sustainability_data or {}

        pack = StakeholderCommunicationPack(
            organization_name=input_data.organization_name,
            reporting_period=input_data.reporting_period or "2024",
        )

        # Extract key messages
        pack.key_messages = self._extract_key_messages(data)

        # Generate content for each audience
        audiences = input_data.target_audiences or [Audience.INVESTORS, Audience.EMPLOYEES]
        formats = input_data.formats or [ContentFormat.EXECUTIVE_SUMMARY]

        for audience in audiences:
            for fmt in formats:
                content = self._generate_content(
                    input_data.organization_name,
                    audience,
                    fmt,
                    pack.key_messages,
                    data,
                )
                pack.communications.append(content)

        pack.audiences_covered = [a.value for a in audiences]
        pack.formats_generated = [f.value for f in formats]
        pack.provenance_hash = pack.calculate_provenance_hash()

        return StakeholderCommOutput(
            success=True,
            action="generate_pack",
            communication_pack=pack,
        )

    def _handle_create_content(
        self,
        input_data: StakeholderCommInput
    ) -> StakeholderCommOutput:
        """Create specific content piece."""
        audiences = input_data.target_audiences or [Audience.INVESTORS]
        formats = input_data.formats or [ContentFormat.EXECUTIVE_SUMMARY]
        data = input_data.sustainability_data or {}

        messages = self._extract_key_messages(data)
        content = self._generate_content(
            input_data.organization_name or "Organization",
            audiences[0],
            formats[0],
            messages,
            data,
        )

        return StakeholderCommOutput(
            success=True,
            action="create_content",
            content=content,
        )

    def _handle_extract_messages(
        self,
        input_data: StakeholderCommInput
    ) -> StakeholderCommOutput:
        """Extract key messages from sustainability data."""
        data = input_data.sustainability_data or {}
        messages = self._extract_key_messages(data)

        return StakeholderCommOutput(
            success=True,
            action="extract_messages",
            key_messages=messages,
        )

    def _extract_key_messages(
        self,
        data: Dict[str, Any]
    ) -> List[KeyMessage]:
        """Extract key messages from sustainability data."""
        messages = []

        # Emissions message
        if "scope1_emissions" in data or "emissions_reduction" in data:
            messages.append(KeyMessage(
                topic="Climate Action",
                headline="Committed to reducing our carbon footprint",
                supporting_points=[
                    "Measured and reported GHG emissions across all scopes",
                    "Implementing emission reduction initiatives",
                ],
                data_points={
                    "scope1": data.get("scope1_emissions"),
                    "reduction": data.get("emissions_reduction"),
                },
                suitable_audiences=[Audience.INVESTORS, Audience.REGULATORS],
            ))

        # Targets message
        if "targets" in data:
            messages.append(KeyMessage(
                topic="Sustainability Targets",
                headline="Clear targets driving our sustainability agenda",
                supporting_points=[
                    "Science-based targets aligned with Paris Agreement",
                    "Progress tracked and reported transparently",
                ],
                suitable_audiences=[Audience.INVESTORS, Audience.EMPLOYEES],
            ))

        # Governance message
        messages.append(KeyMessage(
            topic="Governance",
            headline="Strong sustainability governance at board level",
            supporting_points=[
                "Board oversight of sustainability matters",
                "Executive accountability for ESG performance",
            ],
            suitable_audiences=[Audience.INVESTORS, Audience.REGULATORS],
        ))

        return messages

    def _generate_content(
        self,
        org_name: str,
        audience: Audience,
        fmt: ContentFormat,
        messages: List[KeyMessage],
        data: Dict[str, Any]
    ) -> CommunicationContent:
        """Generate content for specific audience and format."""
        # Determine tone based on audience
        tone_map = {
            Audience.INVESTORS: ContentTone.FORMAL,
            Audience.EMPLOYEES: ContentTone.PROFESSIONAL,
            Audience.CUSTOMERS: ContentTone.CONVERSATIONAL,
            Audience.MEDIA: ContentTone.PROFESSIONAL,
        }
        tone = tone_map.get(audience, ContentTone.PROFESSIONAL)

        # Filter relevant messages
        relevant_messages = [
            m for m in messages
            if audience in m.suitable_audiences or not m.suitable_audiences
        ]

        # Generate title
        title_templates = {
            ContentFormat.EXECUTIVE_SUMMARY: f"{org_name} Sustainability Highlights",
            ContentFormat.PRESS_RELEASE: f"{org_name} Reports Progress on Sustainability Goals",
            ContentFormat.INVESTOR_BRIEF: f"{org_name} ESG Performance Summary",
            ContentFormat.INTERNAL_MEMO: f"Our Sustainability Progress Update",
        }
        title = title_templates.get(fmt, f"{org_name} Sustainability Update")

        # Generate body
        body = self._generate_body(org_name, audience, fmt, relevant_messages, data)

        # Calculate metrics
        word_count = len(body.split())
        reading_time = max(1, word_count // 200)

        return CommunicationContent(
            format=fmt,
            audience=audience,
            tone=tone,
            title=title,
            body=body,
            key_messages=[m.headline for m in relevant_messages],
            word_count=word_count,
            reading_time_minutes=reading_time,
            suggested_visuals=["emissions_trend_chart", "targets_progress"],
            data_highlights={
                "scope1_emissions": data.get("scope1_emissions"),
            },
        )

    def _generate_body(
        self,
        org_name: str,
        audience: Audience,
        fmt: ContentFormat,
        messages: List[KeyMessage],
        data: Dict[str, Any]
    ) -> str:
        """Generate body content."""
        # Template-based generation for now
        # In full implementation, this would use AI for narrative generation

        lines = [f"{org_name} is committed to sustainable business practices."]

        for msg in messages[:3]:
            lines.append(f"\n{msg.headline}")
            for point in msg.supporting_points[:2]:
                lines.append(f"- {point}")

        if audience == Audience.INVESTORS:
            lines.append("\nFor detailed ESG metrics, please refer to our full sustainability report.")
        elif audience == Audience.EMPLOYEES:
            lines.append("\nThank you for your continued contribution to our sustainability journey.")

        return "\n".join(lines)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "StakeholderCommunicationAgent",
    "Audience",
    "ContentFormat",
    "ContentTone",
    "KeyMessage",
    "CommunicationContent",
    "StakeholderCommunicationPack",
    "StakeholderCommInput",
    "StakeholderCommOutput",
]
