"""
CDP Questionnaire Engine -- Full CDP Climate Change Questionnaire Management

This module implements the complete CDP Climate Change questionnaire structure
with 13 modules (M0-M13), 200+ questions, conditional logic, skip patterns,
sector-specific routing, question versioning (2024/2025/2026), and module
completion calculation.

The engine serves as the single source of truth for the questionnaire structure,
providing question lookup, dependency resolution, and module management.

Key capabilities:
  - 13 module definitions with question counts
  - Question registry with 200+ questions covering all CDP modules
  - Question type support: text, numeric, table, select, yes/no, etc.
  - Conditional logic and skip patterns
  - Sector-specific question routing (Financial Services -> M12)
  - Question versioning across CDP cycles (2024/2025/2026)
  - Module completion percentage calculation
  - Scoring category mapping per question

Example:
    >>> engine = QuestionnaireEngine(config)
    >>> questionnaire = engine.create_questionnaire("org-123", 2026)
    >>> questions = engine.get_module_questions("M1")
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .config import (
    CDPAppConfig,
    CDPModule,
    MODULE_DEFINITIONS,
    QuestionType,
    ResponseStatus,
    ScoringCategory,
)
from .models import (
    Module,
    Question,
    QuestionDependency,
    QuestionOption,
    Questionnaire,
    _new_id,
    _now,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Question Registry -- CDP Climate Change 2026 Questionnaire
# ---------------------------------------------------------------------------
# Each entry maps a question number to its full metadata.  The registry covers
# all 13 modules with representative questions per module.  Additional
# questions per version are loaded dynamically.
# ---------------------------------------------------------------------------

def _build_question_registry() -> Dict[str, Dict[str, Any]]:
    """
    Build the complete CDP question registry.

    Returns a dictionary keyed by question_number with full metadata
    for each question across all 13 modules.
    """
    registry: Dict[str, Dict[str, Any]] = {}

    # -----------------------------------------------------------------------
    # M0: Introduction (15 questions)
    # -----------------------------------------------------------------------
    m0_questions = [
        ("C0.1", "Introduction", "Provide a general description and introduction to your organization.",
         QuestionType.TEXT, [], ["SC01"], 1.0, False, None, None),
        ("C0.2", "Reporting year", "State the start and end date of the year for which you are reporting.",
         QuestionType.DATE, [], [], 1.0, False, None, None),
        ("C0.3", "Country/region", "Select the country/area/region in which your organization operates.",
         QuestionType.MULTI_SELECT, [], [], 1.0, False, None, None),
        ("C0.4", "Currency", "Select the currency used for all financial information disclosed throughout your response.",
         QuestionType.SINGLE_SELECT, [
             QuestionOption(value="USD", label="United States Dollar", score_points=0),
             QuestionOption(value="EUR", label="Euro", score_points=0),
             QuestionOption(value="GBP", label="British Pound", score_points=0),
             QuestionOption(value="JPY", label="Japanese Yen", score_points=0),
         ], [], 1.0, False, None, None),
        ("C0.5", "Reporting boundary", "Select the option that describes the reporting boundary for which climate-related impacts on your business are being reported.",
         QuestionType.SINGLE_SELECT, [
             QuestionOption(value="operational_control", label="Operational control", score_points=1),
             QuestionOption(value="financial_control", label="Financial control", score_points=1),
             QuestionOption(value="equity_share", label="Equity share", score_points=1),
         ], [], 1.0, False, None, None),
        ("C0.6", "ISIC classification", "Within your supply chain, please select the ISIC classifications that best describe your organization's activities.",
         QuestionType.MULTI_SELECT, [], [], 1.0, False, None, None),
        ("C0.7", "Sector", "Are there any parts of your direct operations or supply chain that are not included in your disclosure?",
         QuestionType.YES_NO, [], [], 1.0, False, None, None),
        ("C0.7a", "Exclusions", "If yes, identify the part(s) of your direct operations or supply chain not included and provide explanation.",
         QuestionType.TABLE, [], [], 1.0, False, "C0.7", "Yes"),
        ("C0.8", "Emissions methodology", "Does your organization have an ISIN code or another unique identifier (e.g., Ticker, CUSIP, etc.)?",
         QuestionType.YES_NO, [], [], 1.0, False, None, None),
        ("C0.9", "Verification", "Does your organization undertake any verification of climate-related information reported in your CDP disclosure?",
         QuestionType.YES_NO, [], ["SC09"], 1.0, False, None, None),
        ("C0.10", "Verification details", "Provide details of the verification/assurance status that applies to your reported emissions.",
         QuestionType.TABLE, [], ["SC09"], 1.0, False, "C0.9", "Yes"),
        ("C0.11", "Base year", "State the base year you are using for your emissions reporting.",
         QuestionType.NUMERIC, [], [], 1.0, False, None, None),
        ("C0.12", "Recalculations", "Have you recalculated your base year emissions during the reporting period?",
         QuestionType.YES_NO, [], [], 1.0, False, None, None),
        ("C0.13", "Previous CDP response", "Is this your organization's first time responding to CDP's climate change questionnaire?",
         QuestionType.YES_NO, [], [], 1.0, False, None, None),
        ("C0.14", "Board-level oversight", "Provide details on the board-level oversight of climate-related issues within your organization.",
         QuestionType.TEXT, [], ["SC01"], 1.0, False, None, None),
    ]

    for q_num, sub, text, qtype, opts, cats, weight, auto, dep_q, dep_v in m0_questions:
        entry = _make_question_entry(
            q_num, CDPModule.M0_INTRODUCTION, sub, text, qtype, opts, cats, weight,
            auto_source=None, dep_question=dep_q, dep_value=dep_v,
        )
        registry[q_num] = entry

    # -----------------------------------------------------------------------
    # M1: Governance (20 questions)
    # -----------------------------------------------------------------------
    m1_questions = [
        ("C1.1", "Board oversight", "Is there board-level oversight of climate-related issues within your organization?",
         QuestionType.YES_NO, [], ["SC01"], 2.0),
        ("C1.1a", "Board position", "Identify the position(s) (do not include any names) of the individual(s) on the board with responsibility for climate-related issues.",
         QuestionType.TABLE, [], ["SC01"], 2.0),
        ("C1.1b", "Board processes", "Provide further details on the board's oversight of climate-related issues.",
         QuestionType.TEXT, [], ["SC01"], 2.0),
        ("C1.1c", "Board competency", "Why is there no board-level oversight of climate-related issues and what are your plans to change this in the future?",
         QuestionType.TEXT, [], ["SC01"], 1.0),
        ("C1.1d", "Board frequency", "Does your organization have at least one board member with competence on climate-related issues?",
         QuestionType.YES_NO, [], ["SC01"], 2.0),
        ("C1.2", "Management responsibility", "Provide the highest management-level position(s) or committee(s) with responsibility for climate-related issues.",
         QuestionType.TABLE, [], ["SC01"], 2.0),
        ("C1.2a", "Management details", "Describe where in the organizational structure this/these position(s) and/or committees lie, what their associated responsibilities are, and how climate-related issues are monitored at these positions.",
         QuestionType.TEXT, [], ["SC01"], 1.5),
        ("C1.3", "Incentives", "Do you provide incentives for the management of climate-related issues, including the attainment of targets?",
         QuestionType.YES_NO, [], ["SC01"], 2.0),
        ("C1.3a", "Incentive details", "Provide further details on the incentives provided for the management of climate-related issues.",
         QuestionType.TABLE, [], ["SC01"], 2.0),
        ("C1.4", "Board strategy", "Did your organization's board of directors approve the climate transition plan?",
         QuestionType.YES_NO, [], ["SC01", "SC15"], 2.0),
        ("C1.4a", "Strategy integration", "Describe how climate-related risks and opportunities have influenced your organization's strategy.",
         QuestionType.TEXT, [], ["SC01", "SC05"], 1.5),
        ("C1.5", "Risk assessment", "Does your board oversee the setting of sustainability/ESG-related targets?",
         QuestionType.YES_NO, [], ["SC01", "SC07"], 1.5),
        ("C1.5a", "Target oversight", "Provide details on the board's oversight of sustainability/ESG-related targets.",
         QuestionType.TABLE, [], ["SC01", "SC07"], 1.5),
        ("C1.6", "Due diligence", "Does your organization have a process for managing and monitoring climate-related lobbying and/or policy engagement activities?",
         QuestionType.YES_NO, [], ["SC01", "SC14"], 1.0),
        ("C1.6a", "Policy engagement", "Provide details of the process for managing and monitoring climate-related lobbying and policy engagement activities.",
         QuestionType.TEXT, [], ["SC14"], 1.0),
        ("C1.7", "Audit committee", "Does the audit committee or equivalent body have oversight of climate-related issues?",
         QuestionType.YES_NO, [], ["SC01"], 1.0),
        ("C1.8", "Remuneration", "Is climate performance factored into executive remuneration?",
         QuestionType.YES_NO, [], ["SC01"], 1.5),
        ("C1.8a", "Remuneration details", "Describe how climate performance is factored into executive remuneration.",
         QuestionType.TEXT, [], ["SC01"], 1.5),
        ("C1.9", "Skills assessment", "Does the board regularly assess its skills and competencies related to climate?",
         QuestionType.YES_NO, [], ["SC01"], 1.0),
        ("C1.10", "Board training", "Describe any board-level training on climate-related issues in the reporting year.",
         QuestionType.TEXT, [], ["SC01"], 1.0),
    ]

    for q_num, sub, text, qtype, opts, cats, weight in m1_questions:
        entry = _make_question_entry(
            q_num, CDPModule.M1_GOVERNANCE, sub, text, qtype, opts, cats, weight,
        )
        registry[q_num] = entry

    # -----------------------------------------------------------------------
    # M2: Policies & Commitments (15 questions)
    # -----------------------------------------------------------------------
    m2_questions = [
        ("C2.1", "Climate policy", "Does your organization have a climate-related policy?", QuestionType.YES_NO, [], ["SC05"], 2.0),
        ("C2.1a", "Policy scope", "Describe the scope and content of your climate-related policy.", QuestionType.TEXT, [], ["SC05"], 1.5),
        ("C2.2", "Net-zero commitment", "Does your organization have a net-zero target?", QuestionType.YES_NO, [], ["SC07", "SC15"], 2.5),
        ("C2.2a", "Net-zero details", "Provide details of your organization's net-zero target.", QuestionType.TABLE, [], ["SC07", "SC15"], 2.0),
        ("C2.3", "RE100", "Has your organization made a commitment to RE100 or other renewable energy targets?", QuestionType.YES_NO, [], ["SC11"], 1.5),
        ("C2.3a", "RE100 details", "Provide details of your renewable energy commitment.", QuestionType.TEXT, [], ["SC11"], 1.0),
        ("C2.4", "Deforestation", "Does your organization have a policy on deforestation/conversion of natural ecosystems?", QuestionType.YES_NO, [], ["SC05"], 1.0),
        ("C2.4a", "Deforestation details", "Describe the scope and content of your deforestation/conversion-free policy.", QuestionType.TEXT, [], ["SC05"], 1.0),
        ("C2.5", "Human rights", "Does your climate transition plan account for a just transition?", QuestionType.YES_NO, [], ["SC15"], 1.5),
        ("C2.6", "Water commitment", "Does your organization have a water-related commitment?", QuestionType.YES_NO, [], [], 0.5),
        ("C2.7", "Biodiversity", "Does your organization have a biodiversity-related commitment?", QuestionType.YES_NO, [], [], 0.5),
        ("C2.8", "Nature positive", "Has your organization committed to nature-positive outcomes?", QuestionType.YES_NO, [], [], 0.5),
        ("C2.9", "Internal carbon price", "Does your organization use an internal carbon price?", QuestionType.YES_NO, [], ["SC12"], 1.5),
        ("C2.9a", "Carbon price details", "Provide details of your internal carbon pricing.", QuestionType.TABLE, [], ["SC12"], 1.5),
        ("C2.10", "Policy review", "When was your climate policy last reviewed or updated?", QuestionType.DATE, [], ["SC05"], 0.5),
    ]

    for q_num, sub, text, qtype, opts, cats, weight in m2_questions:
        entry = _make_question_entry(q_num, CDPModule.M2_POLICIES, sub, text, qtype, opts, cats, weight)
        registry[q_num] = entry

    # -----------------------------------------------------------------------
    # M3: Risks & Opportunities (25 questions)
    # -----------------------------------------------------------------------
    m3_questions = [
        ("C3.1", "Risk process", "Does your organization's strategy include a climate transition plan that aligns with a 1.5C world?", QuestionType.YES_NO, [], ["SC02", "SC15"], 2.0),
        ("C3.1a", "Risk integration", "Describe your organization's process for identifying, assessing and responding to climate-related risks and opportunities.", QuestionType.TEXT, [], ["SC02"], 2.0),
        ("C3.1b", "Risk frequency", "How does your organization define short-, medium- and long-term time horizons?", QuestionType.TABLE, [], ["SC02"], 1.0),
        ("C3.2", "Physical risks", "Does your organization undertake a climate-related scenario analysis?", QuestionType.YES_NO, [], ["SC06"], 2.5),
        ("C3.2a", "Scenario types", "Provide details of your organization's use of climate-related scenario analysis.", QuestionType.TEXT, [], ["SC06"], 2.0),
        ("C3.2b", "Scenario results", "Provide details of the focal questions, time horizon, and outcomes from your climate scenario analysis.", QuestionType.TABLE, [], ["SC06"], 2.0),
        ("C3.3", "Physical risk detail", "Describe where and how climate-related risks and opportunities have influenced your strategy.", QuestionType.TEXT, [], ["SC05", "SC03"], 2.0),
        ("C3.3a", "Strategy influence", "Which of your organization's strategy-related decisions have been informed by climate-related scenarios?", QuestionType.MULTI_SELECT, [], ["SC05", "SC06"], 1.5),
        ("C3.4", "Transition risks", "Describe your organization's process for managing climate-related risks.", QuestionType.TEXT, [], ["SC02"], 2.0),
        ("C3.5", "Risk identification", "Have you identified any inherent climate-related risks with the potential to have a substantive financial or strategic impact on your business?", QuestionType.YES_NO, [], ["SC03"], 2.0),
        ("C3.5a", "Risk details", "Provide details of risks identified with the potential to have a substantive financial or strategic impact on your business.", QuestionType.TABLE, [], ["SC03"], 2.5),
        ("C3.5b", "Risk changes", "Why do you not consider your organization to be exposed to climate-related risks with the potential to have a substantive financial or strategic impact?", QuestionType.TEXT, [], ["SC03"], 1.0),
        ("C3.6", "Opportunities", "Have you identified any inherent climate-related opportunities with the potential to have a substantive financial or strategic impact on your business?", QuestionType.YES_NO, [], ["SC04"], 2.0),
        ("C3.6a", "Opportunity details", "Provide details of opportunities identified with the potential to have a substantive financial or strategic impact on your business.", QuestionType.TABLE, [], ["SC04"], 2.5),
        ("C3.6b", "No opportunities", "Why do you not consider your organization to have climate-related opportunities?", QuestionType.TEXT, [], ["SC04"], 1.0),
        ("C3.7", "Financial impact", "Provide the total amount of your reported spending/revenue from products/services designed to reduce emissions.", QuestionType.TABLE, [], ["SC17"], 2.0),
        ("C3.7a", "Financial quantification", "What is the financial impact of the identified climate-related risks?", QuestionType.TABLE, [], ["SC17"], 2.0),
        ("C3.8", "Material risks", "Have you quantified the financial impact of the identified risks?", QuestionType.YES_NO, [], ["SC17"], 1.5),
        ("C3.9", "Risk prioritization", "Describe your organization's approach to climate risk prioritization.", QuestionType.TEXT, [], ["SC02"], 1.5),
        ("C3.10", "Resilience", "Describe the resilience of your organization's strategy, taking into consideration different climate-related scenarios.", QuestionType.TEXT, [], ["SC05", "SC06"], 2.0),
        ("C3.11", "Assets at risk", "Quantify the proportion of your total assets that could be considered at risk from climate change.", QuestionType.PERCENTAGE, [], ["SC17"], 1.0),
        ("C3.12", "Revenue at risk", "Quantify the proportion of your total revenue that could be at risk from climate change.", QuestionType.PERCENTAGE, [], ["SC17"], 1.0),
        ("C3.13", "Insurance", "Does your organization use insurance to manage climate-related risks?", QuestionType.YES_NO, [], ["SC02"], 0.5),
        ("C3.14", "Supply chain risk", "Have you identified climate-related risks in your supply chain?", QuestionType.YES_NO, [], ["SC03", "SC13"], 1.5),
        ("C3.14a", "Supply chain details", "Describe the identified climate-related risks in your supply chain.", QuestionType.TABLE, [], ["SC03", "SC13"], 1.5),
    ]

    for q_num, sub, text, qtype, opts, cats, weight in m3_questions:
        entry = _make_question_entry(q_num, CDPModule.M3_RISKS, sub, text, qtype, opts, cats, weight)
        registry[q_num] = entry

    # -----------------------------------------------------------------------
    # M4: Strategy (20 questions)
    # -----------------------------------------------------------------------
    m4_questions = [
        ("C4.1", "Strategy alignment", "Has climate change been integrated into your business strategy?", QuestionType.YES_NO, [], ["SC05"], 2.5),
        ("C4.1a", "Strategy details", "Describe how climate change has been integrated into your business strategy.", QuestionType.TEXT, [], ["SC05"], 2.0),
        ("C4.1b", "Strategy milestones", "Provide details of your climate-related strategic milestones.", QuestionType.TABLE, [], ["SC05", "SC15"], 2.0),
        ("C4.2", "Financial planning", "Has your organization identified climate-related impacts on its financial planning?", QuestionType.YES_NO, [], ["SC05", "SC17"], 2.0),
        ("C4.2a", "Financial impact areas", "Provide details of the areas of your financial planning affected by climate-related impacts.", QuestionType.TABLE, [], ["SC05", "SC17"], 2.0),
        ("C4.3", "Revenue impact", "Have you estimated the financial impact of the climate-related strategy on annual revenue?", QuestionType.YES_NO, [], ["SC17"], 1.5),
        ("C4.3a", "Revenue estimates", "Provide revenue-related financial estimates.", QuestionType.TABLE, [], ["SC17"], 1.5),
        ("C4.4", "CapEx alignment", "What percentage of total CapEx is aligned with your climate transition plan?", QuestionType.PERCENTAGE, [], ["SC15", "SC05"], 2.0),
        ("C4.5", "Low-carbon products", "Do you classify any of your existing goods and/or services as low-carbon products?", QuestionType.YES_NO, [], ["SC04"], 1.5),
        ("C4.5a", "Low-carbon revenue", "Provide details of your products and/or services that you classify as low-carbon products.", QuestionType.TABLE, [], ["SC04"], 1.5),
        ("C4.6", "R&D investment", "Describe your organization's investment in climate-related R&D.", QuestionType.TEXT, [], ["SC08"], 1.5),
        ("C4.7", "M&A considerations", "Do you consider climate-related factors in M&A and investment decisions?", QuestionType.YES_NO, [], ["SC05"], 1.0),
        ("C4.8", "Joint ventures", "Describe how climate factors are considered in joint ventures.", QuestionType.TEXT, [], ["SC05"], 1.0),
        ("C4.9", "Advocacy", "Describe your organization's climate-related public policy engagement.", QuestionType.TEXT, [], ["SC14"], 1.5),
        ("C4.9a", "Trade associations", "Describe the positions taken by any trade associations of which you are a member.", QuestionType.TABLE, [], ["SC14"], 1.0),
        ("C4.10", "Stranded assets", "Has your organization assessed the risk of stranded assets?", QuestionType.YES_NO, [], ["SC03", "SC05"], 1.0),
        ("C4.11", "Just transition", "How does your strategy address the just transition?", QuestionType.TEXT, [], ["SC05", "SC15"], 1.0),
        ("C4.12", "Circular economy", "Describe any circular economy strategies employed.", QuestionType.TEXT, [], ["SC08"], 1.0),
        ("C4.13", "Sector pathway", "Is your sector covered by a recognized low-carbon pathway?", QuestionType.YES_NO, [], ["SC05"], 0.5),
        ("C4.14", "Future strategy", "Describe planned changes to your business model in response to climate.", QuestionType.TEXT, [], ["SC05"], 1.0),
    ]

    for q_num, sub, text, qtype, opts, cats, weight in m4_questions:
        entry = _make_question_entry(q_num, CDPModule.M4_STRATEGY, sub, text, qtype, opts, cats, weight)
        registry[q_num] = entry

    # -----------------------------------------------------------------------
    # M5: Transition Plans (20 questions)
    # -----------------------------------------------------------------------
    m5_questions = [
        ("C5.1", "Transition plan", "Does your organization have a climate transition plan?", QuestionType.YES_NO, [], ["SC15"], 3.0),
        ("C5.1a", "Plan year", "In what year was your transition plan published?", QuestionType.NUMERIC, [], ["SC15"], 1.0),
        ("C5.1b", "Plan link", "Provide a link to your transition plan if publicly available.", QuestionType.TEXT, [], ["SC15"], 2.0),
        ("C5.2", "Plan scope", "What is the scope of your transition plan?", QuestionType.MULTI_SELECT, [], ["SC15"], 2.0),
        ("C5.2a", "Pathway alignment", "Is your transition plan aligned to a 1.5C pathway?", QuestionType.YES_NO, [], ["SC15"], 3.0),
        ("C5.3", "SBTi targets", "Does your organization have an approved Science Based Target?", QuestionType.YES_NO, [], ["SC07", "SC15"], 3.0),
        ("C5.3a", "SBTi details", "Provide details of your Science Based Target.", QuestionType.TABLE, [], ["SC07", "SC15"], 2.5),
        ("C5.4", "Reduction targets", "Provide details of your emissions reduction targets.", QuestionType.TABLE, [], ["SC07"], 2.5),
        ("C5.4a", "Target progress", "Describe the progress made against your emissions reduction targets.", QuestionType.TABLE, [], ["SC07"], 2.0),
        ("C5.5", "Milestones", "Provide details of the key milestones in your transition plan.", QuestionType.TABLE, [], ["SC15"], 2.5),
        ("C5.6", "Assumptions", "What key assumptions underpin your transition plan?", QuestionType.TEXT, [], ["SC15"], 1.5),
        ("C5.7", "Decarbonization levers", "What decarbonization levers are included in your transition plan?", QuestionType.MULTI_SELECT, [], ["SC15", "SC08"], 2.0),
        ("C5.8", "Investment plan", "What are the planned investments associated with your transition plan?", QuestionType.TABLE, [], ["SC15"], 2.0),
        ("C5.9", "Revenue alignment", "What percentage of revenue is aligned with your transition plan?", QuestionType.PERCENTAGE, [], ["SC15"], 1.5),
        ("C5.10", "Board approval", "Has your board approved the transition plan?", QuestionType.YES_NO, [], ["SC01", "SC15"], 2.0),
        ("C5.11", "External review", "Has your transition plan been externally reviewed or assured?", QuestionType.YES_NO, [], ["SC15"], 1.5),
        ("C5.12", "Plan updates", "How frequently is your transition plan reviewed and updated?", QuestionType.SINGLE_SELECT, [], ["SC15"], 1.0),
        ("C5.13", "Stakeholder engagement", "How have stakeholders been engaged in development of the transition plan?", QuestionType.TEXT, [], ["SC15"], 1.0),
        ("C5.14", "Plan challenges", "What are the main challenges to implementing your transition plan?", QuestionType.TEXT, [], ["SC15"], 1.0),
        ("C5.15", "Residual emissions", "How does your transition plan address residual emissions?", QuestionType.TEXT, [], ["SC15"], 1.0),
    ]

    for q_num, sub, text, qtype, opts, cats, weight in m5_questions:
        entry = _make_question_entry(q_num, CDPModule.M5_TRANSITION, sub, text, qtype, opts, cats, weight)
        registry[q_num] = entry

    # -----------------------------------------------------------------------
    # M6: Implementation (20 questions)
    # -----------------------------------------------------------------------
    m6_questions = [
        ("C6.1", "Initiatives overview", "Did you have any emissions reduction initiatives that were active within the reporting year?", QuestionType.YES_NO, [], ["SC08"], 2.5),
        ("C6.1a", "Initiative details", "Provide details of your emissions reduction initiatives active within the reporting year.", QuestionType.TABLE, [], ["SC08"], 2.5),
        ("C6.1b", "Initiative results", "Provide the estimated annual CO2e savings of your initiatives.", QuestionType.TABLE, [], ["SC08"], 2.0),
        ("C6.2", "Energy efficiency", "Describe your energy efficiency initiatives.", QuestionType.TEXT, [], ["SC08", "SC11"], 1.5),
        ("C6.3", "Renewable energy", "Describe your renewable energy procurement.", QuestionType.TEXT, [], ["SC08", "SC11"], 1.5),
        ("C6.4", "Carbon capture", "Are you exploring or using carbon capture/removal technologies?", QuestionType.YES_NO, [], ["SC08"], 1.5),
        ("C6.4a", "CCS details", "Provide details of carbon capture/removal technologies.", QuestionType.TABLE, [], ["SC08"], 1.5),
        ("C6.5", "Internal carbon pricing", "If your organization has an internal carbon price, describe how it incentivizes emissions reduction.", QuestionType.TEXT, [], ["SC12"], 1.5),
        ("C6.5a", "Carbon price impact", "Describe the impact of your internal carbon pricing scheme on business decisions.", QuestionType.TEXT, [], ["SC12"], 1.5),
        ("C6.6", "Avoided emissions", "Have you estimated the avoided emissions generated by your products/services?", QuestionType.YES_NO, [], ["SC08"], 1.0),
        ("C6.6a", "Avoided emissions data", "Provide details of the avoided emissions from your products/services.", QuestionType.TABLE, [], ["SC08"], 1.0),
        ("C6.7", "Innovation", "Describe any significant climate-related innovation your organization has undertaken.", QuestionType.TEXT, [], ["SC08"], 1.5),
        ("C6.8", "R&D spending", "What is your total R&D investment in low-carbon technologies?", QuestionType.CURRENCY, [], ["SC08"], 1.0),
        ("C6.9", "Nature-based solutions", "Does your organization invest in nature-based solutions for emission reductions?", QuestionType.YES_NO, [], ["SC08"], 1.0),
        ("C6.10", "Offset strategy", "Describe your organization's approach to carbon offsets/credits.", QuestionType.TEXT, [], ["SC08", "SC15"], 1.5),
        ("C6.10a", "Offset volume", "Provide details of offsets/credits retired in the reporting year.", QuestionType.TABLE, [], ["SC08"], 1.0),
        ("C6.11", "Supply chain reduction", "What actions have you taken to reduce emissions in your supply chain?", QuestionType.TEXT, [], ["SC08", "SC13"], 1.5),
        ("C6.12", "Employee engagement", "Describe your employee engagement on climate initiatives.", QuestionType.TEXT, [], ["SC08"], 1.0),
        ("C6.13", "Divestment", "Have you divested from high-carbon assets during the reporting year?", QuestionType.YES_NO, [], ["SC08"], 0.5),
        ("C6.14", "CapEx allocation", "What percentage of your total CapEx was dedicated to emissions reduction initiatives?", QuestionType.PERCENTAGE, [], ["SC08"], 1.5),
    ]

    for q_num, sub, text, qtype, opts, cats, weight in m6_questions:
        entry = _make_question_entry(q_num, CDPModule.M6_IMPLEMENTATION, sub, text, qtype, opts, cats, weight)
        registry[q_num] = entry

    # -----------------------------------------------------------------------
    # M7: Environmental Performance - Climate Change (35 questions)
    # -----------------------------------------------------------------------
    m7_questions = [
        ("C7.1", "Scope 1 total", "What were your organization's gross global Scope 1 emissions in metric tons CO2e?", QuestionType.NUMERIC, [], ["SC09"], 3.0),
        ("C7.1a", "Scope 1 breakdown", "Break down your total gross global Scope 1 emissions by greenhouse gas type.", QuestionType.TABLE, [], ["SC09"], 2.0),
        ("C7.1b", "Scope 1 by country", "Break down your total gross global Scope 1 emissions by country/region.", QuestionType.TABLE, [], ["SC09"], 1.5),
        ("C7.1c", "Scope 1 by activity", "Break down your total gross global Scope 1 emissions by business division or activity.", QuestionType.TABLE, [], ["SC09"], 1.5),
        ("C7.2", "Biogenic CO2", "Report your organization's biogenic CO2 emissions separately.", QuestionType.NUMERIC, [], ["SC09"], 1.0),
        ("C7.3", "Scope 2 location", "What were your organization's gross global Scope 2 emissions (location-based) in metric tons CO2e?", QuestionType.NUMERIC, [], ["SC09"], 3.0),
        ("C7.3a", "Scope 2 market", "What were your organization's gross global Scope 2 emissions (market-based) in metric tons CO2e?", QuestionType.NUMERIC, [], ["SC09"], 3.0),
        ("C7.3b", "Scope 2 instruments", "If reporting a Scope 2 figure that is different from the location-based figure, explain.", QuestionType.TEXT, [], ["SC09"], 1.0),
        ("C7.4", "Scope 2 by country", "Break down your Scope 2 emissions by country/region.", QuestionType.TABLE, [], ["SC09"], 1.0),
        ("C7.5", "Scope 1+2 total", "What was your total gross global Scope 1 and 2 emissions combined?", QuestionType.NUMERIC, [], ["SC09"], 2.0),
        ("C7.5a", "Scope 1+2 change", "Describe the change in your Scope 1 and 2 emissions compared to the previous year.", QuestionType.TEXT, [], ["SC09"], 1.5),
        ("C7.6", "Scope 3 screening", "Have you screened your Scope 3 emission sources?", QuestionType.YES_NO, [], ["SC10"], 2.5),
        ("C7.6a", "Scope 3 categories", "Provide details of your Scope 3 emissions by category.", QuestionType.TABLE, [], ["SC10"], 3.0),
        ("C7.6b", "Scope 3 methodology", "Describe the methodology used to calculate your Scope 3 emissions.", QuestionType.TEXT, [], ["SC10"], 2.0),
        ("C7.6c", "Scope 3 exclusions", "Are there any Scope 3 categories you have excluded from your reporting?", QuestionType.TABLE, [], ["SC10"], 1.5),
        ("C7.7", "Total emissions", "What was your total gross emissions for the reporting year in metric tons CO2e?", QuestionType.NUMERIC, [], ["SC09", "SC10"], 2.0),
        ("C7.8", "Emission factors", "Describe the emission factors and GWP values used.", QuestionType.TEXT, [], ["SC09"], 1.0),
        ("C7.9", "Methodology", "Describe the standards, methodologies, and assumptions used for your GHG calculations.", QuestionType.TEXT, [], ["SC09", "SC10"], 1.5),
        ("C7.9a", "Calculation tools", "Which calculation tools or software did you use?", QuestionType.MULTI_SELECT, [], ["SC09"], 0.5),
        ("C7.10", "Verification S1", "Has your organization's Scope 1 emissions been verified by a third party?", QuestionType.YES_NO, [], ["SC09"], 3.0),
        ("C7.10a", "Verification S1 details", "Provide details of the verification/assurance of your Scope 1 emissions.", QuestionType.TABLE, [], ["SC09"], 2.0),
        ("C7.11", "Verification S2", "Has your organization's Scope 2 emissions been verified by a third party?", QuestionType.YES_NO, [], ["SC09"], 3.0),
        ("C7.11a", "Verification S2 details", "Provide details of the verification/assurance of your Scope 2 emissions.", QuestionType.TABLE, [], ["SC09"], 2.0),
        ("C7.12", "Verification S3", "Has your organization's Scope 3 emissions been verified by a third party?", QuestionType.YES_NO, [], ["SC10"], 2.5),
        ("C7.12a", "Verification S3 details", "Provide details of the verification/assurance of your Scope 3 emissions.", QuestionType.TABLE, [], ["SC10"], 2.0),
        ("C7.13", "Energy total", "What is your total energy consumption in the reporting year in MWh?", QuestionType.NUMERIC, [], ["SC11"], 2.0),
        ("C7.13a", "Energy breakdown", "Break down your total energy consumption by fuel type.", QuestionType.TABLE, [], ["SC11"], 1.5),
        ("C7.14", "Renewable energy", "What percentage of your total energy consumption is from renewable sources?", QuestionType.PERCENTAGE, [], ["SC11"], 2.0),
        ("C7.15", "Energy intensity", "Provide your organization's energy intensity ratio.", QuestionType.TABLE, [], ["SC11"], 1.5),
        ("C7.16", "Emissions intensity", "Provide your organization's emissions intensity.", QuestionType.TABLE, [], ["SC09"], 1.5),
        ("C7.17", "Data quality", "Describe the data quality of your emissions reporting.", QuestionType.TEXT, [], ["SC09", "SC10"], 1.0),
        ("C7.18", "Recalculations", "Describe any recalculations of previously reported emissions.", QuestionType.TEXT, [], ["SC09"], 0.5),
        ("C7.19", "Carbon removals", "Report any carbon dioxide removals by your organization.", QuestionType.TABLE, [], ["SC09"], 1.0),
        ("C7.20", "Grid electricity", "Provide details of your grid electricity purchases.", QuestionType.TABLE, [], ["SC09", "SC11"], 1.0),
        ("C7.21", "Purchased instruments", "Provide details of energy attribute certificates/renewable energy certificates.", QuestionType.TABLE, [], ["SC09", "SC11"], 1.0),
    ]

    for q_num, sub, text, qtype, opts, cats, weight in m7_questions:
        auto_src = None
        if q_num in ("C7.1", "C7.1a", "C7.1b", "C7.1c"):
            auto_src = "scope_1"
        elif q_num in ("C7.3", "C7.3a", "C7.4"):
            auto_src = "scope_2"
        elif q_num in ("C7.6a",):
            auto_src = "scope_3"
        elif q_num in ("C7.13", "C7.13a", "C7.14"):
            auto_src = "energy"
        entry = _make_question_entry(q_num, CDPModule.M7_CLIMATE_PERFORMANCE, sub, text, qtype, opts, cats, weight, auto_source=auto_src)
        registry[q_num] = entry

    # -----------------------------------------------------------------------
    # M8: Forests (15 questions -- sector-specific)
    # -----------------------------------------------------------------------
    m8_questions = [
        ("C8.1", "Forest risk", "Are there any forest risk commodities relevant to your organization?", QuestionType.YES_NO, [], [], 1.0),
        ("C8.1a", "Commodities", "Select the forest risk commodities relevant to your organization.", QuestionType.MULTI_SELECT, [], [], 1.0),
        ("C8.2", "Traceability", "Describe your organization's approach to traceability.", QuestionType.TEXT, [], [], 1.5),
        ("C8.3", "Certification", "Does your organization use certification schemes for forest risk commodities?", QuestionType.YES_NO, [], [], 1.0),
        ("C8.4", "Supplier engagement forests", "Describe engagement with suppliers on deforestation.", QuestionType.TEXT, [], [], 1.5),
        ("C8.5", "Production volume", "Provide production/consumption volumes for forest risk commodities.", QuestionType.TABLE, [], [], 1.0),
        ("C8.6", "Deforestation assessment", "Have you assessed deforestation risk in your supply chain?", QuestionType.YES_NO, [], [], 1.0),
        ("C8.7", "Risk assessment results", "Provide details of deforestation risk assessment results.", QuestionType.TABLE, [], [], 1.5),
        ("C8.8", "Mitigation actions", "Describe actions taken to mitigate deforestation risks.", QuestionType.TEXT, [], [], 1.5),
        ("C8.9", "Monitoring", "How do you monitor compliance with your no-deforestation policy?", QuestionType.TEXT, [], [], 1.0),
        ("C8.10", "Targets forests", "Do you have targets related to reducing deforestation?", QuestionType.YES_NO, [], [], 1.0),
        ("C8.11", "EUDR alignment", "Describe alignment with the EU Deforestation Regulation.", QuestionType.TEXT, [], [], 1.0),
        ("C8.12", "Grievance mechanism", "Do you have a grievance mechanism for deforestation-related concerns?", QuestionType.YES_NO, [], [], 0.5),
        ("C8.13", "Restoration", "Describe any restoration or reforestation commitments.", QuestionType.TEXT, [], [], 1.0),
        ("C8.14", "Smallholders", "How do you engage with smallholders in your supply chain?", QuestionType.TEXT, [], [], 0.5),
    ]

    for q_num, sub, text, qtype, opts, cats, weight in m8_questions:
        entry = _make_question_entry(q_num, CDPModule.M8_FORESTS, sub, text, qtype, opts, cats, weight)
        entry["sector_specific"] = True
        registry[q_num] = entry

    # -----------------------------------------------------------------------
    # M9: Water Security (15 questions -- sector-specific)
    # -----------------------------------------------------------------------
    m9_questions = [
        ("C9.1", "Water dependency", "Does your organization face water-related risks?", QuestionType.YES_NO, [], [], 1.0),
        ("C9.2", "Water assessment", "Describe your water risk assessment approach.", QuestionType.TEXT, [], [], 1.5),
        ("C9.3", "Water withdrawal", "Report your total water withdrawal.", QuestionType.TABLE, [], [], 1.0),
        ("C9.4", "Water discharge", "Report your total water discharge.", QuestionType.TABLE, [], [], 1.0),
        ("C9.5", "Water consumption", "Report your total water consumption.", QuestionType.NUMERIC, [], [], 1.0),
        ("C9.6", "Water targets", "Do you have water-related targets?", QuestionType.YES_NO, [], [], 1.0),
        ("C9.7", "Water target details", "Provide details of your water-related targets.", QuestionType.TABLE, [], [], 1.5),
        ("C9.8", "Watershed context", "Describe the watershed context for your operations.", QuestionType.TEXT, [], [], 1.0),
        ("C9.9", "Water governance", "Describe board-level oversight of water-related issues.", QuestionType.TEXT, [], [], 1.0),
        ("C9.10", "Supplier water", "Do you engage suppliers on water issues?", QuestionType.YES_NO, [], [], 0.5),
        ("C9.11", "Water stewardship", "Describe any water stewardship activities.", QuestionType.TEXT, [], [], 1.0),
        ("C9.12", "Water pricing", "Do you incorporate water pricing in your operations?", QuestionType.YES_NO, [], [], 0.5),
        ("C9.13", "Pollution prevention", "Describe actions to prevent water pollution.", QuestionType.TEXT, [], [], 1.0),
        ("C9.14", "Flood risk", "Have you assessed flood risk for your operations?", QuestionType.YES_NO, [], [], 0.5),
        ("C9.15", "Water innovations", "Describe any water-related innovations or technologies.", QuestionType.TEXT, [], [], 0.5),
    ]

    for q_num, sub, text, qtype, opts, cats, weight in m9_questions:
        entry = _make_question_entry(q_num, CDPModule.M9_WATER, sub, text, qtype, opts, cats, weight)
        entry["sector_specific"] = True
        registry[q_num] = entry

    # -----------------------------------------------------------------------
    # M10: Supply Chain (15 questions)
    # -----------------------------------------------------------------------
    m10_questions = [
        ("C10.1", "Supplier engagement", "Do you engage with your value chain on climate-related issues?", QuestionType.YES_NO, [], ["SC13"], 2.5),
        ("C10.1a", "Engagement type", "Select the methods used to engage your value chain.", QuestionType.MULTI_SELECT, [], ["SC13"], 2.0),
        ("C10.1b", "Engagement details", "Provide details of your engagement with the value chain on climate-related issues.", QuestionType.TABLE, [], ["SC13"], 2.0),
        ("C10.2", "Scope 3 collaboration", "Do you collaborate with suppliers to reduce Scope 3 emissions?", QuestionType.YES_NO, [], ["SC13", "SC10"], 2.0),
        ("C10.2a", "Collaboration details", "Describe collaboration with suppliers.", QuestionType.TEXT, [], ["SC13"], 1.5),
        ("C10.3", "Supplier data", "How many suppliers have provided emissions data to you?", QuestionType.NUMERIC, [], ["SC13"], 1.5),
        ("C10.4", "CDP Supply Chain", "Do you use CDP Supply Chain to collect supplier data?", QuestionType.YES_NO, [], ["SC13"], 1.5),
        ("C10.5", "Engagement targets", "Describe any supplier engagement targets.", QuestionType.TEXT, [], ["SC13"], 1.5),
        ("C10.6", "Customer engagement", "Do you engage with customers on climate-related issues?", QuestionType.YES_NO, [], ["SC13"], 1.5),
        ("C10.6a", "Customer details", "Describe customer engagement on climate.", QuestionType.TEXT, [], ["SC13"], 1.0),
        ("C10.7", "Cascade requests", "Did you request climate data from your suppliers through CDP?", QuestionType.YES_NO, [], ["SC13"], 1.5),
        ("C10.8", "Supplier response rate", "What was the response rate from your supply chain program?", QuestionType.PERCENTAGE, [], ["SC13"], 1.0),
        ("C10.9", "Hotspot analysis", "Have you identified emission hotspots in your supply chain?", QuestionType.YES_NO, [], ["SC13", "SC10"], 1.5),
        ("C10.9a", "Hotspot details", "Describe supply chain emission hotspots.", QuestionType.TABLE, [], ["SC13", "SC10"], 1.5),
        ("C10.10", "Procurement criteria", "Do you include climate criteria in procurement decisions?", QuestionType.YES_NO, [], ["SC13"], 1.0),
    ]

    for q_num, sub, text, qtype, opts, cats, weight in m10_questions:
        entry = _make_question_entry(q_num, CDPModule.M10_SUPPLY_CHAIN, sub, text, qtype, opts, cats, weight)
        registry[q_num] = entry

    # -----------------------------------------------------------------------
    # M11: Additional Metrics (10 questions)
    # -----------------------------------------------------------------------
    m11_questions = [
        ("C11.1", "Energy mix", "Report your organization's total energy consumption and generation.", QuestionType.TABLE, [], ["SC11"], 2.0),
        ("C11.2", "Renewable share", "What percentage of electricity is from renewable sources?", QuestionType.PERCENTAGE, [], ["SC11"], 2.0),
        ("C11.3", "Sector metrics", "Provide any sector-specific climate metrics.", QuestionType.TABLE, [], ["SC11"], 1.5),
        ("C11.4", "Carbon intensity", "Provide your carbon intensity metrics.", QuestionType.TABLE, [], ["SC09"], 1.5),
        ("C11.5", "Fossil fuels", "Provide the breakdown of fossil fuel consumption.", QuestionType.TABLE, [], ["SC11"], 1.0),
        ("C11.6", "Power generation", "If you generate electricity, provide details.", QuestionType.TABLE, [], ["SC11"], 1.0),
        ("C11.7", "Fleet data", "If applicable, provide fleet emissions data.", QuestionType.TABLE, [], ["SC09"], 1.0),
        ("C11.8", "Process emissions", "Report any sector-specific process emissions.", QuestionType.TABLE, [], ["SC09"], 1.0),
        ("C11.9", "Refrigerants detail", "Report refrigerant emissions details.", QuestionType.TABLE, [], ["SC09"], 1.0),
        ("C11.10", "Other metrics", "Report any additional environmental metrics.", QuestionType.TEXT, [], [], 0.5),
    ]

    for q_num, sub, text, qtype, opts, cats, weight in m11_questions:
        entry = _make_question_entry(q_num, CDPModule.M11_ADDITIONAL, sub, text, qtype, opts, cats, weight)
        registry[q_num] = entry

    # -----------------------------------------------------------------------
    # M12: Financial Services (20 questions -- FS sector only)
    # -----------------------------------------------------------------------
    m12_questions = [
        ("C12.1", "Portfolio emissions", "Does your organization measure financed emissions?", QuestionType.YES_NO, [], ["SC16"], 3.0),
        ("C12.1a", "PCAF alignment", "Describe your methodology for measuring financed emissions.", QuestionType.TEXT, [], ["SC16"], 2.0),
        ("C12.2", "Financed emissions", "Provide your financed emissions by asset class.", QuestionType.TABLE, [], ["SC16"], 3.0),
        ("C12.3", "Portfolio alignment", "Have you assessed your portfolio alignment with climate goals?", QuestionType.YES_NO, [], ["SC16"], 2.5),
        ("C12.3a", "Alignment methodology", "Describe portfolio alignment methodology used.", QuestionType.TEXT, [], ["SC16"], 2.0),
        ("C12.4", "Engagement", "Do you engage with investees/borrowers on climate?", QuestionType.YES_NO, [], ["SC16", "SC13"], 2.0),
        ("C12.4a", "Engagement details FS", "Describe your engagement activities with investees/borrowers.", QuestionType.TEXT, [], ["SC16", "SC13"], 1.5),
        ("C12.5", "Sector exposure", "Report your exposure to high-carbon sectors.", QuestionType.TABLE, [], ["SC16"], 2.0),
        ("C12.6", "Green finance", "Report your green/sustainable finance activities.", QuestionType.TABLE, [], ["SC16"], 2.0),
        ("C12.7", "Exclusions FS", "Do you have exclusion policies for fossil fuels?", QuestionType.YES_NO, [], ["SC16"], 1.5),
        ("C12.7a", "Exclusion details", "Describe your fossil fuel exclusion policies.", QuestionType.TEXT, [], ["SC16"], 1.5),
        ("C12.8", "TCFD reporting", "Describe your TCFD-aligned reporting for financial products.", QuestionType.TEXT, [], ["SC16"], 1.5),
        ("C12.9", "Net-zero alliance", "Are you a member of any net-zero finance alliances?", QuestionType.YES_NO, [], ["SC16"], 1.5),
        ("C12.9a", "Alliance details", "Provide details of net-zero finance alliance membership.", QuestionType.TEXT, [], ["SC16"], 1.0),
        ("C12.10", "Real estate", "Report emissions from your commercial real estate portfolio.", QuestionType.TABLE, [], ["SC16"], 1.0),
        ("C12.11", "Insurance", "If applicable, report climate-related insurance underwriting.", QuestionType.TABLE, [], ["SC16"], 1.0),
        ("C12.12", "Data quality FS", "Describe the data quality of your financed emissions.", QuestionType.TEXT, [], ["SC16"], 1.0),
        ("C12.13", "Targets FS", "Do you have targets to reduce financed emissions?", QuestionType.YES_NO, [], ["SC16", "SC07"], 2.0),
        ("C12.13a", "Target details FS", "Provide details of financed emission reduction targets.", QuestionType.TABLE, [], ["SC16", "SC07"], 2.0),
        ("C12.14", "Stewardship", "Describe your stewardship activities related to climate.", QuestionType.TEXT, [], ["SC16"], 1.0),
    ]

    for q_num, sub, text, qtype, opts, cats, weight in m12_questions:
        entry = _make_question_entry(q_num, CDPModule.M12_FINANCIAL_SERVICES, sub, text, qtype, opts, cats, weight)
        entry["sector_specific"] = True
        entry["applicable_sectors"] = ["40"]
        registry[q_num] = entry

    # -----------------------------------------------------------------------
    # M13: Sign Off (5 questions)
    # -----------------------------------------------------------------------
    m13_questions = [
        ("C13.1", "Sign off", "Provide details of the person who has signed off (approved) your CDP climate change response.", QuestionType.TABLE, [], [], 1.0),
        ("C13.2", "Verification statement", "Attach your verification statement(s) if applicable.", QuestionType.FILE_UPLOAD, [], ["SC09"], 1.0),
        ("C13.3", "Board endorsement", "Has the board reviewed and endorsed this CDP response?", QuestionType.YES_NO, [], ["SC01"], 1.5),
        ("C13.4", "Accuracy statement", "Confirm that the information provided in this response is accurate.", QuestionType.YES_NO, [], [], 1.0),
        ("C13.5", "Permission", "Grant permission for CDP to share this response with requesting parties.", QuestionType.SINGLE_SELECT, [
            QuestionOption(value="public", label="Publicly available", score_points=1),
            QuestionOption(value="investors_only", label="Investors only", score_points=0.5),
            QuestionOption(value="private", label="Private (not shared)", score_points=0),
        ], [], 0.5),
    ]

    for q_num, sub, text, qtype, opts, cats, weight in m13_questions:
        entry = _make_question_entry(q_num, CDPModule.M13_SIGN_OFF, sub, text, qtype, opts, cats, weight)
        registry[q_num] = entry

    return registry


def _make_question_entry(
    q_num: str,
    module: CDPModule,
    sub_section: str,
    text: str,
    qtype: QuestionType,
    options: List[QuestionOption],
    scoring_cats: List[str],
    weight: float,
    auto_source: Optional[str] = None,
    dep_question: Optional[str] = None,
    dep_value: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a question registry entry dictionary."""
    dependencies = []
    if dep_question and dep_value:
        dependencies.append(QuestionDependency(
            parent_question_id=dep_question,
            condition_type="equals",
            condition_value=dep_value,
            action="show",
        ))

    return {
        "question_number": q_num,
        "module_code": module,
        "sub_section": sub_section,
        "question_text": text,
        "question_type": qtype,
        "options": options,
        "scoring_categories": scoring_cats,
        "scoring_weight": weight,
        "dependencies": dependencies,
        "auto_populate_source": auto_source,
        "sector_specific": False,
        "applicable_sectors": [],
        "year_introduced": 2024,
    }


# Build the registry at module load time
QUESTION_REGISTRY: Dict[str, Dict[str, Any]] = _build_question_registry()


class QuestionnaireEngine:
    """
    CDP Questionnaire Engine -- manages questionnaire structure and questions.

    Provides CRUD for questionnaires and modules, question lookup by module,
    dependency resolution, sector-specific routing, and completion tracking.

    Attributes:
        config: Application configuration.
        _questionnaires: In-memory questionnaire store.
        _modules: In-memory module store.

    Example:
        >>> engine = QuestionnaireEngine(config)
        >>> q = engine.create_questionnaire("org-123", 2026)
        >>> modules = engine.get_modules(q.id)
    """

    def __init__(self, config: CDPAppConfig) -> None:
        """Initialize the Questionnaire Engine."""
        self.config = config
        self._questionnaires: Dict[str, Questionnaire] = {}
        self._modules: Dict[str, List[Module]] = {}  # questionnaire_id -> modules
        self._question_registry = QUESTION_REGISTRY
        logger.info(
            "QuestionnaireEngine initialized with %d questions across %d modules",
            len(self._question_registry),
            len(MODULE_DEFINITIONS),
        )

    # ------------------------------------------------------------------
    # Questionnaire CRUD
    # ------------------------------------------------------------------

    def create_questionnaire(
        self,
        org_id: str,
        year: int,
        version: str = "2026",
        sector_code: Optional[str] = None,
    ) -> Questionnaire:
        """
        Create a new CDP questionnaire for an organization-year.

        Initializes all applicable modules based on sector. Financial Services
        organizations (sector 40) get Module 12 enabled.

        Args:
            org_id: Organization ID.
            year: Reporting year.
            version: Questionnaire version (2024/2025/2026).
            sector_code: GICS sector code for sector-specific routing.

        Returns:
            Created Questionnaire instance.
        """
        deadline = date(year, self.config.submission_deadline_month,
                        self.config.submission_deadline_day)

        total_q = self._count_applicable_questions(sector_code, version)

        questionnaire = Questionnaire(
            org_id=org_id,
            year=year,
            version=version,
            total_questions=total_q,
            submission_deadline=deadline,
        )

        # Create modules
        modules = self._create_modules(questionnaire.id, sector_code, version)
        self._questionnaires[questionnaire.id] = questionnaire
        self._modules[questionnaire.id] = modules

        logger.info(
            "Created questionnaire %s for org %s year %d with %d modules, %d questions",
            questionnaire.id, org_id, year, len(modules), total_q,
        )
        return questionnaire

    def get_questionnaire(self, questionnaire_id: str) -> Optional[Questionnaire]:
        """Retrieve a questionnaire by ID."""
        return self._questionnaires.get(questionnaire_id)

    def list_questionnaires(self, org_id: str) -> List[Questionnaire]:
        """List all questionnaires for an organization."""
        return [q for q in self._questionnaires.values() if q.org_id == org_id]

    def delete_questionnaire(self, questionnaire_id: str) -> bool:
        """Delete a questionnaire and its modules."""
        if questionnaire_id in self._questionnaires:
            del self._questionnaires[questionnaire_id]
            self._modules.pop(questionnaire_id, None)
            logger.info("Deleted questionnaire %s", questionnaire_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Module Operations
    # ------------------------------------------------------------------

    def get_modules(self, questionnaire_id: str) -> List[Module]:
        """Get all modules for a questionnaire."""
        return self._modules.get(questionnaire_id, [])

    def get_module(self, questionnaire_id: str, module_code: str) -> Optional[Module]:
        """Get a specific module by code."""
        modules = self._modules.get(questionnaire_id, [])
        for m in modules:
            if m.module_code.value == module_code:
                return m
        return None

    def update_module_completion(
        self,
        questionnaire_id: str,
        module_code: str,
        answered: int,
        approved: int,
    ) -> Optional[Module]:
        """
        Update module completion counts and recalculate percentages.

        Args:
            questionnaire_id: Questionnaire ID.
            module_code: Module code (M0-M13).
            answered: Number of answered questions.
            approved: Number of approved questions.

        Returns:
            Updated Module or None.
        """
        module = self.get_module(questionnaire_id, module_code)
        if not module:
            return None

        module.answered_questions = min(answered, module.total_questions)
        module.approved_questions = min(approved, module.total_questions)

        if module.total_questions > 0:
            module.completion_pct = Decimal(str(
                round(module.answered_questions / module.total_questions * 100, 1)
            ))
        else:
            module.completion_pct = Decimal("100.0")

        # Update status based on completion
        if module.approved_questions >= module.total_questions:
            module.status = ResponseStatus.APPROVED
        elif module.answered_questions > 0:
            module.status = ResponseStatus.DRAFT
        else:
            module.status = ResponseStatus.NOT_STARTED

        module.updated_at = _now()

        # Update questionnaire-level totals
        self._recalculate_questionnaire_totals(questionnaire_id)
        return module

    # ------------------------------------------------------------------
    # Question Operations
    # ------------------------------------------------------------------

    def get_module_questions(
        self,
        module_code: str,
        version: str = "2026",
        sector_code: Optional[str] = None,
    ) -> List[Question]:
        """
        Get all questions for a specific module.

        Filters by version and sector applicability.

        Args:
            module_code: Module code (M0-M13).
            version: Questionnaire version year.
            sector_code: GICS sector for filtering sector-specific questions.

        Returns:
            List of Question objects.
        """
        questions = []
        version_year = int(version)

        for q_num, entry in self._question_registry.items():
            if entry["module_code"].value != module_code:
                continue

            # Version filter
            if entry.get("year_introduced", 2024) > version_year:
                continue
            year_retired = entry.get("year_retired")
            if year_retired and year_retired <= version_year:
                continue

            # Sector filter
            if entry.get("sector_specific", False):
                applicable = entry.get("applicable_sectors", [])
                if applicable and sector_code not in applicable:
                    continue

            question = self._entry_to_question(entry)
            questions.append(question)

        questions.sort(key=lambda q: q.order)
        return questions

    def get_question(self, question_number: str) -> Optional[Question]:
        """Get a single question by its number."""
        entry = self._question_registry.get(question_number)
        if entry:
            return self._entry_to_question(entry)
        return None

    def get_all_questions(
        self,
        version: str = "2026",
        sector_code: Optional[str] = None,
    ) -> List[Question]:
        """Get all questions across all modules."""
        all_questions = []
        for module_code in MODULE_DEFINITIONS:
            module_qs = self.get_module_questions(module_code, version, sector_code)
            all_questions.extend(module_qs)
        return all_questions

    def get_question_dependencies(self, question_number: str) -> List[QuestionDependency]:
        """Get dependency chain for a question."""
        entry = self._question_registry.get(question_number)
        if not entry:
            return []
        return entry.get("dependencies", [])

    def evaluate_skip_logic(
        self,
        question_number: str,
        all_responses: Dict[str, str],
    ) -> bool:
        """
        Evaluate whether a question should be shown based on skip logic.

        Args:
            question_number: Question to evaluate.
            all_responses: Dict of question_number -> response value.

        Returns:
            True if the question should be displayed, False to skip.
        """
        entry = self._question_registry.get(question_number)
        if not entry:
            return False

        dependencies = entry.get("dependencies", [])
        if not dependencies:
            return True  # No dependencies = always show

        for dep in dependencies:
            parent_value = all_responses.get(dep.parent_question_id, "")
            if dep.condition_type == "equals":
                if parent_value != dep.condition_value:
                    return dep.action != "show"
            elif dep.condition_type == "not_equals":
                if parent_value == dep.condition_value:
                    return dep.action != "show"
            elif dep.condition_type == "contains":
                if dep.condition_value not in parent_value:
                    return dep.action != "show"

        return True

    def get_scoring_category_questions(
        self,
        category_id: str,
        version: str = "2026",
    ) -> List[Question]:
        """Get all questions mapped to a specific scoring category."""
        questions = []
        for q_num, entry in self._question_registry.items():
            if category_id in entry.get("scoring_categories", []):
                questions.append(self._entry_to_question(entry))
        return questions

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _create_modules(
        self,
        questionnaire_id: str,
        sector_code: Optional[str],
        version: str,
    ) -> List[Module]:
        """Create all applicable modules for a questionnaire."""
        modules = []
        for code, defn in MODULE_DEFINITIONS.items():
            # Skip sector-specific modules unless applicable
            if defn.get("sector_specific", False):
                applicable = defn.get("applicable_sectors", [])
                if applicable and sector_code not in applicable:
                    continue

            question_count = self._count_module_questions(
                code, version, sector_code,
            )

            module = Module(
                questionnaire_id=questionnaire_id,
                module_code=CDPModule(code),
                name=defn["name"],
                description=defn["description"],
                order=defn["order"],
                required=defn.get("required", True),
                sector_specific=defn.get("sector_specific", False),
                applicable_sectors=defn.get("applicable_sectors", []),
                total_questions=question_count,
            )
            modules.append(module)

        modules.sort(key=lambda m: m.order)
        return modules

    def _count_module_questions(
        self,
        module_code: str,
        version: str,
        sector_code: Optional[str],
    ) -> int:
        """Count applicable questions in a module."""
        count = 0
        version_year = int(version)
        for entry in self._question_registry.values():
            if entry["module_code"].value != module_code:
                continue
            if entry.get("year_introduced", 2024) > version_year:
                continue
            year_retired = entry.get("year_retired")
            if year_retired and year_retired <= version_year:
                continue
            if entry.get("sector_specific", False):
                applicable = entry.get("applicable_sectors", [])
                if applicable and sector_code not in applicable:
                    continue
            count += 1
        return count

    def _count_applicable_questions(
        self,
        sector_code: Optional[str],
        version: str,
    ) -> int:
        """Count total applicable questions across all modules."""
        total = 0
        for module_code in MODULE_DEFINITIONS:
            total += self._count_module_questions(module_code, version, sector_code)
        return total

    def _recalculate_questionnaire_totals(self, questionnaire_id: str) -> None:
        """Recalculate questionnaire-level completion totals."""
        q = self._questionnaires.get(questionnaire_id)
        if not q:
            return

        modules = self._modules.get(questionnaire_id, [])
        total_answered = sum(m.answered_questions for m in modules)
        total_approved = sum(m.approved_questions for m in modules)
        total_questions = sum(m.total_questions for m in modules)

        q.answered_questions = total_answered
        q.approved_questions = total_approved
        q.total_questions = total_questions

        if total_questions > 0:
            q.completion_pct = Decimal(str(
                round(total_answered / total_questions * 100, 1)
            ))
        else:
            q.completion_pct = Decimal("0.0")

        q.updated_at = _now()

    def _entry_to_question(self, entry: Dict[str, Any]) -> Question:
        """Convert a registry entry to a Question model."""
        q_num = entry["question_number"]
        # Determine order from position within module
        module_entries = [
            (k, v) for k, v in self._question_registry.items()
            if v["module_code"] == entry["module_code"]
        ]
        order = 0
        for idx, (k, _) in enumerate(module_entries):
            if k == q_num:
                order = idx
                break

        # Compute scoring points based on weight
        weight = entry.get("scoring_weight", 1.0)
        return Question(
            question_number=q_num,
            module_code=entry["module_code"],
            sub_section=entry.get("sub_section"),
            question_text=entry["question_text"],
            question_type=entry["question_type"],
            options=entry.get("options", []),
            required=True,
            scoring_categories=entry.get("scoring_categories", []),
            scoring_weight=weight,
            disclosure_points=min(weight * 0.25, 1.0),
            awareness_points=min(weight * 0.50, 2.0),
            management_points=min(weight * 0.75, 3.0),
            leadership_points=min(weight * 1.0, 4.0),
            max_points=4.0,
            dependencies=entry.get("dependencies", []),
            auto_populate_source=entry.get("auto_populate_source"),
            order=order,
        )
