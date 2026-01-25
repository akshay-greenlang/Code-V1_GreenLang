# -*- coding: utf-8 -*-
from datetime import datetime, date
from typing import List, Optional
import logging, hashlib, uuid
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from .schemas import *

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/reports", tags=["Reporting"])

def get_current_user(request: Request):
    return {"user_id": "user-001", "email": "user@example.com", "roles": ["operator"]}

def calculate_provenance_hash(data: dict) -> str:
    return hashlib.sha256(str(sorted(data.items())).encode()).hexdigest()[:16]

@router.get("/quarterly", response_model=dict)
async def get_quarterly_report(
    request: Request,
    facility_id: str = Query(...),
    year: int = Query(..., ge=2000, le=2100),
    quarter: int = Query(..., ge=1, le=4),
    include_details: bool = Query(True),
    format: str = Query("json"),
    current_user: dict = Depends(get_current_user)
):
    report_id = f"QTR-{facility_id}-{year}Q{quarter}"
    return {"report_id": report_id, "facility_id": facility_id, "year": year, "quarter": quarter,
            "status": "generated", "emissions_summary": {}, "compliance_summary": {}}

@router.get("/annual", response_model=dict)
async def get_annual_report(
    request: Request,
    facility_id: str = Query(...),
    year: int = Query(..., ge=2000, le=2100),
    include_details: bool = Query(True),
    format: str = Query("json"),
    current_user: dict = Depends(get_current_user)
):
    report_id = f"ANN-{facility_id}-{year}"
    return {"report_id": report_id, "facility_id": facility_id, "year": year,
            "total_ghg_emissions_mtco2e": 0, "emissions_by_pollutant": {}, "status": "generated"}

@router.get("/deviation", response_model=dict)
async def get_deviation_reports(
    request: Request,
    facility_id: str = Query(...),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    status: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    pagination = PaginationMeta.from_pagination(page, page_size, 0)
    return {"data": [], "pagination": pagination.model_dump()}

@router.get("/audit-trail", response_model=dict)
async def get_audit_trail(
    request: Request,
    facility_id: Optional[str] = Query(None),
    entity_type: Optional[str] = Query(None),
    entity_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    current_user: dict = Depends(get_current_user)
):
    if "admin" not in current_user.get("roles", []) and "auditor" not in current_user.get("roles", []):
        raise HTTPException(status_code=403, detail="Requires admin or auditor role")
    pagination = PaginationMeta.from_pagination(page, page_size, 0)
    return {"data": [], "pagination": pagination.model_dump()}
