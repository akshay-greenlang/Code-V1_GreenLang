"""
GL-CDP-APP Questionnaire API

Manages CDP Climate Change questionnaire structure: questionnaire instances
per organization-year, 13 modules (M0-M13), 200+ questions with conditional
logic, sector-specific routing, and progress tracking.

CDP Climate Change modules:
    M0  - Introduction (org profile, reporting boundary, base year)
    M1  - Governance (board oversight, management responsibility)
    M2  - Policies & Commitments (climate policies, deforestation-free)
    M3  - Risks & Opportunities (climate risk assessment)
    M4  - Strategy (business strategy alignment, scenario analysis)
    M5  - Transition Plans (1.5C pathway, decarbonization roadmap)
    M6  - Implementation (emissions reduction initiatives)
    M7  - Environmental Performance - Climate Change (Scope 1/2/3)
    M8  - Environmental Performance - Forests (if applicable)
    M9  - Environmental Performance - Water Security (if applicable)
    M10 - Supply Chain (supplier engagement, Scope 3 collaboration)
    M11 - Additional Metrics (sector-specific, energy mix)
    M12 - Financial Services (portfolio emissions, if FS sector)
    M13 - Sign Off (authorization, verification statement)
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/cdp/questionnaires", tags=["Questionnaires"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class QuestionnaireStatus(str, Enum):
    """Lifecycle status of a CDP questionnaire instance."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    SUBMITTED = "submitted"


class QuestionnaireYear(str, Enum):
    """Supported CDP questionnaire years."""
    Y2024 = "2024"
    Y2025 = "2025"
    Y2026 = "2026"


class ModuleStatus(str, Enum):
    """Module completion status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    LOCKED = "locked"


class QuestionType(str, Enum):
    """CDP question input types."""
    TEXT = "text"
    NUMERIC = "numeric"
    PERCENTAGE = "percentage"
    TABLE = "table"
    MULTI_SELECT = "multi_select"
    SINGLE_SELECT = "single_select"
    YES_NO = "yes_no"


class ScoringLevel(str, Enum):
    """CDP scoring level for a question."""
    DISCLOSURE = "disclosure"
    AWARENESS = "awareness"
    MANAGEMENT = "management"
    LEADERSHIP = "leadership"


class SectorClassification(str, Enum):
    """GICS sector classification for sector-specific routing."""
    ENERGY = "energy"
    MATERIALS = "materials"
    INDUSTRIALS = "industrials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    HEALTH_CARE = "health_care"
    FINANCIALS = "financials"
    INFORMATION_TECHNOLOGY = "information_technology"
    COMMUNICATION_SERVICES = "communication_services"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreateQuestionnaireRequest(BaseModel):
    """Request to create a new CDP questionnaire instance."""
    org_id: str = Field(..., description="Organization ID")
    questionnaire_year: QuestionnaireYear = Field(
        QuestionnaireYear.Y2026, description="CDP questionnaire version year"
    )
    reporting_year: int = Field(..., ge=2020, le=2100, description="Reporting period year")
    sector: Optional[SectorClassification] = Field(
        None, description="GICS sector for sector-specific question routing"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "org_id": "org_abc123",
                "questionnaire_year": "2026",
                "reporting_year": 2025,
                "sector": "materials",
            }
        }


class SetSectorRequest(BaseModel):
    """Request to set or update the sector classification."""
    sector: SectorClassification = Field(
        ..., description="GICS sector classification"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "sector": "financials",
            }
        }


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class QuestionnaireResponse(BaseModel):
    """CDP questionnaire instance."""
    questionnaire_id: str
    org_id: str
    questionnaire_year: str
    reporting_year: int
    sector: Optional[str]
    status: str
    total_questions: int
    answered_questions: int
    completion_pct: float
    module_count: int
    created_at: datetime
    updated_at: datetime


class QuestionnaireListEntry(BaseModel):
    """Summary entry in a questionnaire list."""
    questionnaire_id: str
    org_id: str
    questionnaire_year: str
    reporting_year: int
    status: str
    completion_pct: float
    created_at: datetime


class ModuleResponse(BaseModel):
    """CDP questionnaire module."""
    module_id: str
    questionnaire_id: str
    module_code: str
    module_name: str
    description: str
    question_count: int
    answered_count: int
    completion_pct: float
    status: str
    is_applicable: bool
    is_sector_specific: bool
    order_index: int


class QuestionResponse(BaseModel):
    """CDP questionnaire question."""
    question_id: str
    module_code: str
    sub_section: str
    question_number: str
    question_text: str
    guidance_text: str
    question_type: str
    scoring_level: str
    scoring_category: str
    scoring_weight: float
    is_mandatory: bool
    is_conditional: bool
    condition_parent_id: Optional[str]
    condition_trigger_value: Optional[str]
    is_sector_specific: bool
    applicable_sectors: List[str]
    has_response: bool
    response_status: Optional[str]
    order_index: int


class ProgressResponse(BaseModel):
    """Overall completion progress for a questionnaire."""
    questionnaire_id: str
    total_questions: int
    answered_questions: int
    completion_pct: float
    draft_count: int
    in_review_count: int
    approved_count: int
    not_started_count: int
    modules: List[Dict[str, Any]]
    estimated_score_band: Optional[str]
    days_until_deadline: Optional[int]


class ConditionalQuestionResponse(BaseModel):
    """Conditional question logic definition."""
    question_id: str
    question_number: str
    question_text: str
    parent_question_id: str
    parent_question_number: str
    trigger_condition: str
    trigger_value: str
    is_active: bool


# ---------------------------------------------------------------------------
# CDP Module Definitions
# ---------------------------------------------------------------------------

CDP_MODULES = [
    {"code": "M0", "name": "Introduction", "description": "Organization profile, reporting boundary, base year", "question_count": 15, "is_sector_specific": False},
    {"code": "M1", "name": "Governance", "description": "Board oversight, management responsibility, incentives", "question_count": 20, "is_sector_specific": False},
    {"code": "M2", "name": "Policies & Commitments", "description": "Climate policies, commitments, deforestation-free pledges", "question_count": 15, "is_sector_specific": False},
    {"code": "M3", "name": "Risks & Opportunities", "description": "Climate risk assessment, physical and transition risks", "question_count": 25, "is_sector_specific": False},
    {"code": "M4", "name": "Strategy", "description": "Business strategy alignment, scenario analysis", "question_count": 20, "is_sector_specific": False},
    {"code": "M5", "name": "Transition Plans", "description": "1.5C pathway, decarbonization roadmap, milestones", "question_count": 20, "is_sector_specific": False},
    {"code": "M6", "name": "Implementation", "description": "Emissions reduction initiatives, investments, R&D", "question_count": 20, "is_sector_specific": False},
    {"code": "M7", "name": "Environmental Performance - Climate Change", "description": "Scope 1/2/3 emissions, methodology, verification status", "question_count": 35, "is_sector_specific": False},
    {"code": "M8", "name": "Environmental Performance - Forests", "description": "Commodity-driven deforestation (if applicable)", "question_count": 15, "is_sector_specific": True},
    {"code": "M9", "name": "Environmental Performance - Water Security", "description": "Water dependencies and risks (if applicable)", "question_count": 15, "is_sector_specific": True},
    {"code": "M10", "name": "Supply Chain", "description": "Supplier engagement, Scope 3 collaboration", "question_count": 15, "is_sector_specific": False},
    {"code": "M11", "name": "Additional Metrics", "description": "Sector-specific metrics, energy mix", "question_count": 10, "is_sector_specific": True},
    {"code": "M12", "name": "Financial Services", "description": "Portfolio emissions, financed emissions (Financial Services only)", "question_count": 20, "is_sector_specific": True},
    {"code": "M13", "name": "Sign Off", "description": "Authorization, verification statement", "question_count": 5, "is_sector_specific": False},
]


# Simulated question bank (condensed representation)
def _generate_questions_for_module(module_code: str, questionnaire_id: str) -> List[Dict[str, Any]]:
    """Generate simulated CDP questions for a module."""
    module_def = next((m for m in CDP_MODULES if m["code"] == module_code), None)
    if not module_def:
        return []

    scoring_categories = {
        "M0": "governance", "M1": "governance", "M2": "business_strategy",
        "M3": "risk_disclosure", "M4": "business_strategy", "M5": "transition_plan",
        "M6": "emissions_reduction_initiatives", "M7": "scope_1_2_emissions",
        "M8": "value_chain_engagement", "M9": "value_chain_engagement",
        "M10": "value_chain_engagement", "M11": "energy",
        "M12": "portfolio_climate_performance", "M13": "governance",
    }

    questions = []
    for i in range(1, module_def["question_count"] + 1):
        q_num = f"{module_code}.{i}"
        is_conditional = i > module_def["question_count"] * 0.7
        questions.append({
            "question_id": f"q_{module_code.lower()}_{i:03d}",
            "module_code": module_code,
            "sub_section": f"{module_code}.{(i - 1) // 5 + 1}",
            "question_number": q_num,
            "question_text": f"[{q_num}] CDP Climate Change question for {module_def['name']}",
            "guidance_text": f"CDP guidance: Provide detailed information about {module_def['name'].lower()} practices.",
            "question_type": QuestionType.TEXT.value if i % 4 != 0 else QuestionType.TABLE.value,
            "scoring_level": ScoringLevel.DISCLOSURE.value if i <= 5 else ScoringLevel.MANAGEMENT.value,
            "scoring_category": scoring_categories.get(module_code, "governance"),
            "scoring_weight": round(1.0 / module_def["question_count"], 4),
            "is_mandatory": i <= module_def["question_count"] * 0.6,
            "is_conditional": is_conditional,
            "condition_parent_id": f"q_{module_code.lower()}_{max(1, i - 3):03d}" if is_conditional else None,
            "condition_trigger_value": "Yes" if is_conditional else None,
            "is_sector_specific": module_def["is_sector_specific"],
            "applicable_sectors": [] if not module_def["is_sector_specific"] else ["financials"] if module_code == "M12" else [],
            "has_response": False,
            "response_status": None,
            "order_index": i,
        })
    return questions


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_questionnaires: Dict[str, Dict[str, Any]] = {}
_modules: Dict[str, Dict[str, Any]] = {}
_questions: Dict[str, Dict[str, Any]] = {}


def _generate_id(prefix: str) -> str:
    """Generate a prefixed unique identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.utcnow()


def _initialize_modules(questionnaire_id: str, sector: Optional[str]) -> List[Dict[str, Any]]:
    """Initialize modules for a new questionnaire instance."""
    modules = []
    for idx, mod_def in enumerate(CDP_MODULES):
        # Determine applicability based on sector
        is_applicable = True
        if mod_def["code"] == "M12" and sector != SectorClassification.FINANCIALS.value:
            is_applicable = False

        module_id = _generate_id("mod")
        module = {
            "module_id": module_id,
            "questionnaire_id": questionnaire_id,
            "module_code": mod_def["code"],
            "module_name": mod_def["name"],
            "description": mod_def["description"],
            "question_count": mod_def["question_count"],
            "answered_count": 0,
            "completion_pct": 0.0,
            "status": ModuleStatus.NOT_STARTED.value,
            "is_applicable": is_applicable,
            "is_sector_specific": mod_def["is_sector_specific"],
            "order_index": idx,
        }
        _modules[module_id] = module
        modules.append(module)

        # Generate questions for this module
        questions = _generate_questions_for_module(mod_def["code"], questionnaire_id)
        for q in questions:
            q["questionnaire_id"] = questionnaire_id
            q["module_id"] = module_id
            _questions[q["question_id"]] = q

    return modules


def _get_total_questions(questionnaire_id: str) -> int:
    """Count total applicable questions for a questionnaire."""
    return sum(
        1 for q in _questions.values()
        if q.get("questionnaire_id") == questionnaire_id
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "",
    response_model=List[QuestionnaireListEntry],
    summary="List questionnaires",
    description=(
        "Retrieve all CDP questionnaire instances for an organization, "
        "optionally filtered by questionnaire year or status."
    ),
)
async def list_questionnaires(
    org_id: str = Query(..., description="Organization ID"),
    questionnaire_year: Optional[str] = Query(None, description="Filter by questionnaire year"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results offset for pagination"),
) -> List[QuestionnaireListEntry]:
    """List CDP questionnaire instances for an organization."""
    questionnaires = [
        q for q in _questionnaires.values() if q["org_id"] == org_id
    ]
    if questionnaire_year is not None:
        questionnaires = [q for q in questionnaires if q["questionnaire_year"] == questionnaire_year]
    if status_filter is not None:
        questionnaires = [q for q in questionnaires if q["status"] == status_filter]
    questionnaires.sort(key=lambda q: q["reporting_year"], reverse=True)
    page = questionnaires[offset: offset + limit]
    return [
        QuestionnaireListEntry(
            questionnaire_id=q["questionnaire_id"],
            org_id=q["org_id"],
            questionnaire_year=q["questionnaire_year"],
            reporting_year=q["reporting_year"],
            status=q["status"],
            completion_pct=q["completion_pct"],
            created_at=q["created_at"],
        )
        for q in page
    ]


@router.get(
    "/{questionnaire_id}",
    response_model=QuestionnaireResponse,
    summary="Get questionnaire details",
    description="Retrieve full details of a CDP questionnaire instance including completion metrics.",
)
async def get_questionnaire(questionnaire_id: str) -> QuestionnaireResponse:
    """Retrieve a questionnaire by ID."""
    questionnaire = _questionnaires.get(questionnaire_id)
    if not questionnaire:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Questionnaire {questionnaire_id} not found",
        )
    return QuestionnaireResponse(**questionnaire)


@router.post(
    "",
    response_model=QuestionnaireResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create questionnaire instance",
    description=(
        "Create a new CDP Climate Change questionnaire instance for an "
        "organization and reporting year. Initializes all 13 modules and "
        "200+ questions with sector-specific routing applied."
    ),
)
async def create_questionnaire(request: CreateQuestionnaireRequest) -> QuestionnaireResponse:
    """Create a new CDP questionnaire instance."""
    existing = [
        q for q in _questionnaires.values()
        if q["org_id"] == request.org_id
        and q["questionnaire_year"] == request.questionnaire_year.value
        and q["reporting_year"] == request.reporting_year
    ]
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Questionnaire for year {request.reporting_year} "
                f"(version {request.questionnaire_year.value}) already exists: "
                f"{existing[0]['questionnaire_id']}"
            ),
        )

    questionnaire_id = _generate_id("cdpq")
    now = _now()
    sector_val = request.sector.value if request.sector else None

    # Initialize modules and questions
    modules = _initialize_modules(questionnaire_id, sector_val)
    total_questions = _get_total_questions(questionnaire_id)

    questionnaire = {
        "questionnaire_id": questionnaire_id,
        "org_id": request.org_id,
        "questionnaire_year": request.questionnaire_year.value,
        "reporting_year": request.reporting_year,
        "sector": sector_val,
        "status": QuestionnaireStatus.NOT_STARTED.value,
        "total_questions": total_questions,
        "answered_questions": 0,
        "completion_pct": 0.0,
        "module_count": len(modules),
        "created_at": now,
        "updated_at": now,
    }
    _questionnaires[questionnaire_id] = questionnaire
    return QuestionnaireResponse(**questionnaire)


@router.get(
    "/{questionnaire_id}/modules",
    response_model=List[ModuleResponse],
    summary="List modules with completion status",
    description=(
        "Retrieve all 13 CDP modules for a questionnaire with completion "
        "status, applicability, and question counts."
    ),
)
async def list_modules(
    questionnaire_id: str,
    applicable_only: bool = Query(False, description="Only return applicable modules"),
) -> List[ModuleResponse]:
    """List modules for a questionnaire."""
    questionnaire = _questionnaires.get(questionnaire_id)
    if not questionnaire:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Questionnaire {questionnaire_id} not found",
        )
    modules = [
        m for m in _modules.values()
        if m["questionnaire_id"] == questionnaire_id
    ]
    if applicable_only:
        modules = [m for m in modules if m["is_applicable"]]
    modules.sort(key=lambda m: m["order_index"])
    return [ModuleResponse(**m) for m in modules]


@router.get(
    "/{questionnaire_id}/modules/{module_id}",
    response_model=Dict[str, Any],
    summary="Get module detail with questions",
    description=(
        "Retrieve a specific module with its full list of questions, "
        "completion status, and scoring metadata."
    ),
)
async def get_module_detail(
    questionnaire_id: str,
    module_id: str,
) -> Dict[str, Any]:
    """Retrieve module detail with questions."""
    questionnaire = _questionnaires.get(questionnaire_id)
    if not questionnaire:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Questionnaire {questionnaire_id} not found",
        )
    module = _modules.get(module_id)
    if not module or module["questionnaire_id"] != questionnaire_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Module {module_id} not found in questionnaire {questionnaire_id}",
        )

    questions = [
        q for q in _questions.values()
        if q.get("module_id") == module_id and q.get("questionnaire_id") == questionnaire_id
    ]
    questions.sort(key=lambda q: q["order_index"])

    return {
        "module": ModuleResponse(**module).model_dump(),
        "questions": [QuestionResponse(**q) for q in questions],
    }


@router.get(
    "/{questionnaire_id}/questions",
    response_model=List[QuestionResponse],
    summary="List all questions",
    description=(
        "Retrieve all questions for a questionnaire with optional filtering "
        "by module, scoring level, question type, or response status."
    ),
)
async def list_questions(
    questionnaire_id: str,
    module_code: Optional[str] = Query(None, description="Filter by module code (e.g., M7)"),
    scoring_level: Optional[str] = Query(None, description="Filter by scoring level"),
    question_type: Optional[str] = Query(None, description="Filter by question type"),
    mandatory_only: bool = Query(False, description="Only return mandatory questions"),
    unanswered_only: bool = Query(False, description="Only return unanswered questions"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results offset for pagination"),
) -> List[QuestionResponse]:
    """List questions for a questionnaire with filters."""
    questionnaire = _questionnaires.get(questionnaire_id)
    if not questionnaire:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Questionnaire {questionnaire_id} not found",
        )

    questions = [
        q for q in _questions.values()
        if q.get("questionnaire_id") == questionnaire_id
    ]
    if module_code is not None:
        questions = [q for q in questions if q["module_code"] == module_code]
    if scoring_level is not None:
        questions = [q for q in questions if q["scoring_level"] == scoring_level]
    if question_type is not None:
        questions = [q for q in questions if q["question_type"] == question_type]
    if mandatory_only:
        questions = [q for q in questions if q["is_mandatory"]]
    if unanswered_only:
        questions = [q for q in questions if not q["has_response"]]

    questions.sort(key=lambda q: (q["module_code"], q["order_index"]))
    page = questions[offset: offset + limit]
    return [QuestionResponse(**q) for q in page]


@router.get(
    "/{questionnaire_id}/progress",
    response_model=ProgressResponse,
    summary="Overall completion progress",
    description=(
        "Retrieve aggregated completion progress for a questionnaire "
        "including per-module breakdown, estimated score band, and deadline."
    ),
)
async def get_progress(questionnaire_id: str) -> ProgressResponse:
    """Retrieve overall completion progress."""
    questionnaire = _questionnaires.get(questionnaire_id)
    if not questionnaire:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Questionnaire {questionnaire_id} not found",
        )

    modules = [
        m for m in _modules.values()
        if m["questionnaire_id"] == questionnaire_id
    ]
    modules.sort(key=lambda m: m["order_index"])

    module_progress = []
    for m in modules:
        module_progress.append({
            "module_code": m["module_code"],
            "module_name": m["module_name"],
            "question_count": m["question_count"],
            "answered_count": m["answered_count"],
            "completion_pct": m["completion_pct"],
            "status": m["status"],
            "is_applicable": m["is_applicable"],
        })

    total = questionnaire["total_questions"]
    answered = questionnaire["answered_questions"]
    completion = round(answered / total * 100, 1) if total > 0 else 0.0

    # Estimate score band based on completion
    band = None
    if completion >= 80:
        band = "A/A-"
    elif completion >= 60:
        band = "B/B-"
    elif completion >= 40:
        band = "C/C-"
    elif completion >= 20:
        band = "D/D-"
    elif completion > 0:
        band = "D-"

    return ProgressResponse(
        questionnaire_id=questionnaire_id,
        total_questions=total,
        answered_questions=answered,
        completion_pct=completion,
        draft_count=0,
        in_review_count=0,
        approved_count=0,
        not_started_count=total - answered,
        modules=module_progress,
        estimated_score_band=band,
        days_until_deadline=None,
    )


@router.put(
    "/{questionnaire_id}/sector",
    response_model=QuestionnaireResponse,
    summary="Set sector classification",
    description=(
        "Set or update the GICS sector classification for the questionnaire. "
        "This triggers sector-specific question routing, enabling or disabling "
        "modules like M12 (Financial Services) based on the selected sector."
    ),
)
async def set_sector(
    questionnaire_id: str,
    request: SetSectorRequest,
) -> QuestionnaireResponse:
    """Set sector for sector-specific routing."""
    questionnaire = _questionnaires.get(questionnaire_id)
    if not questionnaire:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Questionnaire {questionnaire_id} not found",
        )
    if questionnaire["status"] == QuestionnaireStatus.SUBMITTED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change sector on a submitted questionnaire.",
        )

    questionnaire["sector"] = request.sector.value
    questionnaire["updated_at"] = _now()

    # Update module applicability based on sector
    is_financials = request.sector == SectorClassification.FINANCIALS
    for m in _modules.values():
        if m["questionnaire_id"] == questionnaire_id:
            if m["module_code"] == "M12":
                m["is_applicable"] = is_financials

    return QuestionnaireResponse(**questionnaire)


@router.get(
    "/{questionnaire_id}/conditional-questions",
    response_model=List[ConditionalQuestionResponse],
    summary="Get conditional question logic",
    description=(
        "Retrieve the conditional logic for all questions in the questionnaire. "
        "Shows which questions are conditionally displayed based on parent "
        "question responses, enabling skip patterns and dependency chains."
    ),
)
async def get_conditional_questions(
    questionnaire_id: str,
    module_code: Optional[str] = Query(None, description="Filter by module code"),
    active_only: bool = Query(False, description="Only return currently active conditionals"),
) -> List[ConditionalQuestionResponse]:
    """Retrieve conditional question logic."""
    questionnaire = _questionnaires.get(questionnaire_id)
    if not questionnaire:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Questionnaire {questionnaire_id} not found",
        )

    conditional_questions = [
        q for q in _questions.values()
        if q.get("questionnaire_id") == questionnaire_id and q["is_conditional"]
    ]
    if module_code is not None:
        conditional_questions = [q for q in conditional_questions if q["module_code"] == module_code]

    results = []
    for q in conditional_questions:
        parent_id = q["condition_parent_id"]
        parent = _questions.get(parent_id, {})
        parent_response = parent.get("has_response", False)

        is_active = False
        if parent_response and parent.get("response_status") in ("draft", "in_review", "approved"):
            is_active = True

        if active_only and not is_active:
            continue

        results.append(ConditionalQuestionResponse(
            question_id=q["question_id"],
            question_number=q["question_number"],
            question_text=q["question_text"],
            parent_question_id=parent_id or "",
            parent_question_number=parent.get("question_number", ""),
            trigger_condition="equals",
            trigger_value=q["condition_trigger_value"] or "",
            is_active=is_active,
        ))

    results.sort(key=lambda r: r.question_number)
    return results
