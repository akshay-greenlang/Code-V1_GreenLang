# -*- coding: utf-8 -*-
"""
Scope3CalculatorAgent API Routes
GL-VCCI Scope 3 Platform

FastAPI routes for the Scope3CalculatorAgent supporting all 15 Scope 3 categories.

Features:
- RESTful API design
- Comprehensive input validation
- Error handling with detailed responses
- OpenAPI documentation
- Performance monitoring
- Batch processing support
- Statistics and health endpoints

Version: 1.0.0
Date: 2025-11-08
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .agent import Scope3CalculatorAgent
from .models import (
    Category1Input,
    Category2Input,
    Category3Input,
    Category4Input,
    Category5Input,
    Category6Input,
    Category7Input,
    Category8Input,
    Category9Input,
    Category10Input,
    Category11Input,
    Category12Input,
    Category13Input,
    Category14Input,
    Category15Input,
    CalculationResult,
    BatchResult,
)
from .config import CalculatorConfig, get_config
from .exceptions import (
    CalculatorError,
    DataValidationError,
    EmissionFactorNotFoundError,
    BatchProcessingError,
)

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/calculate", tags=["Calculator"])

# Global calculator instance (will be initialized via dependency injection)
_calculator_instance: Optional[Scope3CalculatorAgent] = None


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

def get_calculator() -> Scope3CalculatorAgent:
    """
    Get or create calculator agent instance.

    In production, this should be replaced with proper dependency injection
    that includes FactorBroker and IndustryMapper initialization.
    """
    global _calculator_instance

    if _calculator_instance is None:
        # For now, this is a placeholder that will be replaced during integration
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Calculator agent not initialized. Please configure dependencies."
        )

    return _calculator_instance


def set_calculator(calculator: Scope3CalculatorAgent):
    """Set the global calculator instance (called during app startup)."""
    global _calculator_instance
    _calculator_instance = calculator


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class CategoryInfo(BaseModel):
    """Category information model."""
    category: int = Field(description="Category number (1-15)")
    name: str = Field(description="Category name")
    description: str = Field(description="Category description")
    supported: bool = Field(description="Whether category is currently supported")


class CalculatorHealthResponse(BaseModel):
    """Calculator health check response."""
    status: str = Field(description="Health status")
    version: str = Field(description="Calculator version")
    uptime_seconds: float = Field(description="Uptime in seconds")
    total_calculations: int = Field(description="Total calculations performed")
    success_rate: float = Field(description="Success rate (0-1)")


class BatchCalculationRequest(BaseModel):
    """Batch calculation request model."""
    category: int = Field(ge=1, le=15, description="Scope 3 category (1-15)")
    records: List[Dict[str, Any]] = Field(description="List of calculation input records")

    class Config:
        json_schema_extra = {
            "example": {
                "category": 1,
                "records": [
                    {
                        "product_name": "Steel",
                        "quantity": 1000,
                        "quantity_unit": "kg",
                        "region": "US",
                        "supplier_pcf": 1.85
                    }
                ]
            }
        }


class AllCategoriesRequest(BaseModel):
    """Request model for calculating all categories."""
    data: Dict[int, Dict[str, Any]] = Field(
        description="Mapping of category number to input data"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "data": {
                    "1": {
                        "product_name": "Steel",
                        "quantity": 1000,
                        "quantity_unit": "kg",
                        "region": "US"
                    },
                    "4": {
                        "transport_mode": "road_truck_medium",
                        "distance_km": 500,
                        "weight_tonnes": 10
                    }
                }
            }
        }


# ============================================================================
# CATEGORY-SPECIFIC ENDPOINTS
# ============================================================================

@router.post("/category/1", response_model=CalculationResult, status_code=status.HTTP_200_OK)
async def calculate_category_1(
    data: Category1Input,
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> CalculationResult:
    """
    Calculate Category 1 emissions (Purchased Goods & Services).

    Uses 3-tier waterfall approach:
    - Tier 1: Supplier-specific PCF
    - Tier 2: Product emission factors
    - Tier 3: Spend-based calculation

    Returns detailed calculation result with provenance, data quality, and uncertainty.
    """
    try:
        logger.info(f"Processing Category 1 calculation for product: {data.product_name}")
        result = await calculator.calculate_category_1(data)
        return result
    except DataValidationError as e:
        logger.error(f"Category 1 validation error: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except EmissionFactorNotFoundError as e:
        logger.error(f"Category 1 factor not found: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except CalculatorError as e:
        logger.error(f"Category 1 calculation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in Category 1 calculation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post("/category/2", response_model=CalculationResult, status_code=status.HTTP_200_OK)
async def calculate_category_2(
    data: Category2Input,
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> CalculationResult:
    """Calculate Category 2 emissions (Capital Goods)."""
    try:
        logger.info(f"Processing Category 2 calculation for asset: {data.asset_description}")
        result = await calculator.calculate_category_2(data)
        return result
    except CalculatorError as e:
        logger.error(f"Category 2 error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in Category 2: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post("/category/3", response_model=CalculationResult, status_code=status.HTTP_200_OK)
async def calculate_category_3(
    data: Category3Input,
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> CalculationResult:
    """Calculate Category 3 emissions (Fuel & Energy-Related Activities)."""
    try:
        logger.info(f"Processing Category 3 calculation for fuel type: {data.fuel_or_energy_type}")
        result = await calculator.calculate_category_3(data)
        return result
    except CalculatorError as e:
        logger.error(f"Category 3 error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in Category 3: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post("/category/4", response_model=CalculationResult, status_code=status.HTTP_200_OK)
async def calculate_category_4(
    data: Category4Input,
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> CalculationResult:
    """
    Calculate Category 4 emissions (Upstream Transportation & Distribution).

    ISO 14083 compliant calculation:
    emissions = distance × weight × emission_factor

    Returns calculation result with ISO 14083 compliance verification.
    """
    try:
        logger.info(f"Processing Category 4 calculation: {data.transport_mode.value}, {data.distance_km}km, {data.weight_tonnes}t")
        result = await calculator.calculate_category_4(data)
        return result
    except CalculatorError as e:
        logger.error(f"Category 4 error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in Category 4: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post("/category/5", response_model=CalculationResult, status_code=status.HTTP_200_OK)
async def calculate_category_5(
    data: Category5Input,
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> CalculationResult:
    """Calculate Category 5 emissions (Waste Generated in Operations)."""
    try:
        logger.info(f"Processing Category 5 calculation for waste: {data.waste_description}")
        result = await calculator.calculate_category_5(data)
        return result
    except CalculatorError as e:
        logger.error(f"Category 5 error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in Category 5: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post("/category/6", response_model=CalculationResult, status_code=status.HTTP_200_OK)
async def calculate_category_6(
    data: Category6Input,
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> CalculationResult:
    """
    Calculate Category 6 emissions (Business Travel).

    Includes:
    - Flight emissions (with radiative forcing)
    - Hotel stay emissions
    - Ground transport emissions

    Returns comprehensive travel emissions breakdown.
    """
    try:
        logger.info(f"Processing Category 6 calculation: {len(data.flights)} flights, {len(data.hotels)} hotels")
        result = await calculator.calculate_category_6(data)
        return result
    except CalculatorError as e:
        logger.error(f"Category 6 error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in Category 6: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post("/category/7", response_model=CalculationResult, status_code=status.HTTP_200_OK)
async def calculate_category_7(
    data: Category7Input,
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> CalculationResult:
    """Calculate Category 7 emissions (Employee Commuting)."""
    try:
        logger.info(f"Processing Category 7 calculation for {data.num_employees} employees")
        result = await calculator.calculate_category_7(data)
        return result
    except CalculatorError as e:
        logger.error(f"Category 7 error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in Category 7: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post("/category/8", response_model=CalculationResult, status_code=status.HTTP_200_OK)
async def calculate_category_8(
    data: Category8Input,
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> CalculationResult:
    """Calculate Category 8 emissions (Upstream Leased Assets)."""
    try:
        logger.info(f"Processing Category 8 calculation for lease type: {data.lease_type}")
        result = await calculator.calculate_category_8(data)
        return result
    except CalculatorError as e:
        logger.error(f"Category 8 error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in Category 8: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post("/category/9", response_model=CalculationResult, status_code=status.HTTP_200_OK)
async def calculate_category_9(
    data: Category9Input,
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> CalculationResult:
    """Calculate Category 9 emissions (Downstream Transportation & Distribution)."""
    try:
        logger.info(f"Processing Category 9 calculation: {data.transport_mode.value}")
        result = await calculator.calculate_category_9(data)
        return result
    except CalculatorError as e:
        logger.error(f"Category 9 error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in Category 9: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post("/category/10", response_model=CalculationResult, status_code=status.HTTP_200_OK)
async def calculate_category_10(
    data: Category10Input,
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> CalculationResult:
    """Calculate Category 10 emissions (Processing of Sold Products)."""
    try:
        logger.info(f"Processing Category 10 calculation for product: {data.product_description}")
        result = await calculator.calculate_category_10(data)
        return result
    except CalculatorError as e:
        logger.error(f"Category 10 error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in Category 10: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post("/category/11", response_model=CalculationResult, status_code=status.HTTP_200_OK)
async def calculate_category_11(
    data: Category11Input,
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> CalculationResult:
    """Calculate Category 11 emissions (Use of Sold Products)."""
    try:
        logger.info(f"Processing Category 11 calculation: {data.units_sold} units of {data.product_type.value}")
        result = await calculator.calculate_category_11(data)
        return result
    except CalculatorError as e:
        logger.error(f"Category 11 error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in Category 11: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post("/category/12", response_model=CalculationResult, status_code=status.HTTP_200_OK)
async def calculate_category_12(
    data: Category12Input,
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> CalculationResult:
    """Calculate Category 12 emissions (End-of-Life Treatment of Sold Products)."""
    try:
        logger.info(f"Processing Category 12 calculation for product: {data.product_description}")
        result = await calculator.calculate_category_12(data)
        return result
    except CalculatorError as e:
        logger.error(f"Category 12 error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in Category 12: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post("/category/13", response_model=CalculationResult, status_code=status.HTTP_200_OK)
async def calculate_category_13(
    data: Category13Input,
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> CalculationResult:
    """Calculate Category 13 emissions (Downstream Leased Assets)."""
    try:
        logger.info(f"Processing Category 13 calculation: {data.floor_area} sqm")
        result = await calculator.calculate_category_13(data)
        return result
    except CalculatorError as e:
        logger.error(f"Category 13 error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in Category 13: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post("/category/14", response_model=CalculationResult, status_code=status.HTTP_200_OK)
async def calculate_category_14(
    data: Category14Input,
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> CalculationResult:
    """Calculate Category 14 emissions (Franchises)."""
    try:
        logger.info(f"Processing Category 14 calculation: {data.franchise_count} franchises")
        result = await calculator.calculate_category_14(data)
        return result
    except CalculatorError as e:
        logger.error(f"Category 14 error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in Category 14: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


@router.post("/category/15", response_model=CalculationResult, status_code=status.HTTP_200_OK)
async def calculate_category_15(
    data: Category15Input,
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> CalculationResult:
    """Calculate Category 15 emissions (Investments - PCAF Standard)."""
    try:
        logger.info(f"Processing Category 15 calculation for: {data.portfolio_company}")
        result = await calculator.calculate_category_15(data)
        return result
    except CalculatorError as e:
        logger.error(f"Category 15 error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in Category 15: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


# ============================================================================
# DYNAMIC CATEGORY ENDPOINT
# ============================================================================

@router.post("/{category}", response_model=CalculationResult, status_code=status.HTTP_200_OK)
async def calculate_by_category(
    category: int,
    data: Dict[str, Any] = Body(...),
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> CalculationResult:
    """
    Calculate emissions for a specific category by number.

    Dynamic endpoint that routes to the appropriate category calculator.
    Useful for programmatic access when category is determined at runtime.

    Args:
        category: Scope 3 category number (1-15)
        data: Input data dictionary matching the category's input model

    Returns:
        CalculationResult for the specified category
    """
    if category < 1 or category > 15:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid category: {category}. Must be between 1 and 15."
        )

    try:
        logger.info(f"Processing dynamic calculation for category {category}")
        result = await calculator.calculate_by_category(category, data)
        return result
    except ValueError as e:
        logger.error(f"Invalid category or data: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except CalculatorError as e:
        logger.error(f"Category {category} error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in category {category}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


# ============================================================================
# BATCH PROCESSING ENDPOINTS
# ============================================================================

@router.post("/batch", response_model=BatchResult, status_code=status.HTTP_200_OK)
async def calculate_batch(
    request: BatchCalculationRequest,
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> BatchResult:
    """
    Calculate emissions for multiple records in batch.

    Supports parallel processing for improved performance on large datasets.
    Returns aggregated results with error details for failed records.

    Args:
        request: Batch calculation request with category and records

    Returns:
        BatchResult with aggregated emissions and individual results
    """
    if request.category < 1 or request.category > 15:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid category: {request.category}. Must be between 1 and 15."
        )

    if not request.records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Records list cannot be empty"
        )

    try:
        logger.info(f"Processing batch calculation: {len(request.records)} records for category {request.category}")
        result = await calculator.calculate_batch(request.records, request.category)

        logger.info(
            f"Batch calculation completed: {result.successful_records}/{result.total_records} successful, "
            f"{result.total_emissions_tco2e:.3f} tCO2e"
        )

        return result

    except BatchProcessingError as e:
        # Return partial results with error information
        logger.warning(f"Batch processing completed with errors: {e}")
        # Re-raise to return error response but with batch result details
        raise HTTPException(
            status_code=status.HTTP_207_MULTI_STATUS,
            detail={
                "message": str(e),
                "context": e.context
            }
        )
    except CalculatorError as e:
        logger.error(f"Batch calculation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in batch calculation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


# ============================================================================
# ALL CATEGORIES ENDPOINT
# ============================================================================

@router.post("/all", response_model=Dict[int, CalculationResult], status_code=status.HTTP_200_OK)
async def calculate_all_categories(
    request: AllCategoriesRequest,
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> Dict[int, CalculationResult]:
    """
    Calculate emissions for multiple categories in a single request.

    Useful for comprehensive Scope 3 inventory calculations.
    Returns results for each category that has data provided.

    Args:
        request: Mapping of category numbers to input data

    Returns:
        Dictionary mapping category numbers to calculation results
    """
    if not request.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Data dictionary cannot be empty"
        )

    results = {}
    errors = {}

    for category, input_data in request.data.items():
        try:
            if category < 1 or category > 15:
                errors[category] = f"Invalid category number: {category}"
                continue

            logger.info(f"Processing category {category} in all-categories calculation")
            result = await calculator.calculate_by_category(category, input_data)
            results[category] = result

        except CalculatorError as e:
            logger.error(f"Category {category} error: {e}")
            errors[category] = str(e)
        except Exception as e:
            logger.error(f"Unexpected error in category {category}: {e}", exc_info=True)
            errors[category] = "Internal server error"

    if errors:
        logger.warning(f"All-categories calculation completed with {len(errors)} errors")
        # Return partial results with error information
        raise HTTPException(
            status_code=status.HTTP_207_MULTI_STATUS,
            detail={
                "message": "Some categories failed to calculate",
                "results": results,
                "errors": errors
            }
        )

    logger.info(f"All-categories calculation completed successfully for {len(results)} categories")
    return results


# ============================================================================
# METADATA & UTILITY ENDPOINTS
# ============================================================================

@router.get("/categories", response_model=List[CategoryInfo], status_code=status.HTTP_200_OK)
async def list_categories() -> List[CategoryInfo]:
    """
    List all available Scope 3 categories with descriptions.

    Returns metadata about each of the 15 Scope 3 categories including
    their names, descriptions, and support status.
    """
    categories = [
        CategoryInfo(
            category=1,
            name="Purchased Goods & Services",
            description="Emissions from production of purchased goods and services",
            supported=True
        ),
        CategoryInfo(
            category=2,
            name="Capital Goods",
            description="Emissions from production of capital goods",
            supported=True
        ),
        CategoryInfo(
            category=3,
            name="Fuel & Energy-Related Activities",
            description="Upstream emissions from fuel and energy",
            supported=True
        ),
        CategoryInfo(
            category=4,
            name="Upstream Transportation & Distribution",
            description="Transportation of purchased products (ISO 14083)",
            supported=True
        ),
        CategoryInfo(
            category=5,
            name="Waste Generated in Operations",
            description="Disposal and treatment of waste",
            supported=True
        ),
        CategoryInfo(
            category=6,
            name="Business Travel",
            description="Employee business travel emissions",
            supported=True
        ),
        CategoryInfo(
            category=7,
            name="Employee Commuting",
            description="Employee commuting emissions",
            supported=True
        ),
        CategoryInfo(
            category=8,
            name="Upstream Leased Assets",
            description="Emissions from leased assets (upstream)",
            supported=True
        ),
        CategoryInfo(
            category=9,
            name="Downstream Transportation & Distribution",
            description="Transportation of sold products",
            supported=True
        ),
        CategoryInfo(
            category=10,
            name="Processing of Sold Products",
            description="Processing of intermediate products by third parties",
            supported=True
        ),
        CategoryInfo(
            category=11,
            name="Use of Sold Products",
            description="End-use of sold products",
            supported=True
        ),
        CategoryInfo(
            category=12,
            name="End-of-Life Treatment of Sold Products",
            description="Disposal of sold products at end of life",
            supported=True
        ),
        CategoryInfo(
            category=13,
            name="Downstream Leased Assets",
            description="Emissions from leased assets (downstream)",
            supported=True
        ),
        CategoryInfo(
            category=14,
            name="Franchises",
            description="Emissions from franchise operations",
            supported=True
        ),
        CategoryInfo(
            category=15,
            name="Investments",
            description="Emissions from investments (PCAF Standard)",
            supported=True
        ),
    ]

    return categories


@router.get("/stats", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def get_statistics(
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> Dict[str, Any]:
    """
    Get calculator performance statistics.

    Returns detailed statistics including:
    - Total calculations performed
    - Success/failure rates
    - Performance metrics
    - Category breakdown
    """
    try:
        stats = calculator.get_performance_stats()
        return stats
    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error retrieving statistics")


@router.post("/stats/reset", status_code=status.HTTP_204_NO_CONTENT)
async def reset_statistics(
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
):
    """
    Reset calculator performance statistics.

    Clears all accumulated performance metrics and counters.
    """
    try:
        calculator.reset_stats()
        logger.info("Calculator statistics reset")
        return None
    except Exception as e:
        logger.error(f"Error resetting statistics: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error resetting statistics")


@router.get("/health", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def health_check(
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> Dict[str, Any]:
    """
    Calculator health check endpoint.

    Returns calculator status and basic metrics.
    """
    try:
        stats = calculator.get_performance_stats()

        return {
            "status": "healthy",
            "service": "Scope3CalculatorAgent",
            "version": "1.0.0",
            "categories_supported": 15,
            "total_calculations": stats["total_calculations"],
            "success_rate": stats["success_rate"],
            "features": {
                "monte_carlo": calculator.config.enable_monte_carlo,
                "provenance": calculator.config.enable_provenance,
                "parallel_processing": calculator.config.enable_parallel_processing,
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/config", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def get_configuration(
    calculator: Scope3CalculatorAgent = Depends(get_calculator)
) -> Dict[str, Any]:
    """
    Get current calculator configuration.

    Returns the active configuration settings for the calculator.
    """
    try:
        config_dict = calculator.config.model_dump()
        return config_dict
    except Exception as e:
        logger.error(f"Error retrieving configuration: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error retrieving configuration")


# ============================================================================
# EXPORT ALL
# ============================================================================

__all__ = [
    "router",
    "get_calculator",
    "set_calculator",
]
