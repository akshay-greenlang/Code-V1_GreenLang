"""
Proto Python Stubs for GreenLang gRPC Services

This package contains Python message classes and service stubs
generated from protobuf definitions for GreenLang gRPC services.

Modules:
    process_heat_pb2: Message classes for all services
    process_heat_pb2_grpc: Service stubs and servicer base classes

Example:
    >>> from greenlang.infrastructure.api.proto import process_heat_pb2
    >>> from greenlang.infrastructure.api.proto import process_heat_pb2_grpc
    >>>
    >>> request = process_heat_pb2.CalculationRequest(
    ...     calculation_id="calc-001",
    ...     agent_name="FurnaceAgent"
    ... )
"""

from greenlang.infrastructure.api.proto.process_heat_pb2 import (
    # Calculation messages
    CalculationRequest,
    CalculationResponse,
    StatusRequest,
    StatusResponse,
    StreamRequest,
    CalculationResult,
    EquipmentSpecification,
    FuelInput,
    OperatingConditions,
    # Emissions messages
    EmissionsRequest,
    EmissionsResponse,
    ActivityData,
    EmissionDetail,
    EmissionFactorsRequest,
    EmissionFactorsResponse,
    EmissionFactor,
    # Compliance messages
    ReportRequest,
    ReportResponse,
    ComplianceItem,
    ComplianceCheckRequest,
    ComplianceCheckResponse,
    ComplianceDataPoint,
    ComplianceViolation,
    # Enums
    CalculationStatus,
    ValidationStatus,
    Scope,
    ComplianceStatus,
    Severity,
)

from greenlang.infrastructure.api.proto.process_heat_pb2_grpc import (
    # Service stubs
    ProcessHeatServiceStub,
    EmissionsServiceStub,
    ComplianceServiceStub,
    # Servicer base classes
    ProcessHeatServiceServicer,
    EmissionsServiceServicer,
    ComplianceServiceServicer,
    # Registration functions
    add_ProcessHeatServiceServicer_to_server,
    add_EmissionsServiceServicer_to_server,
    add_ComplianceServiceServicer_to_server,
)

__all__ = [
    # Calculation messages
    "CalculationRequest",
    "CalculationResponse",
    "StatusRequest",
    "StatusResponse",
    "StreamRequest",
    "CalculationResult",
    "EquipmentSpecification",
    "FuelInput",
    "OperatingConditions",
    # Emissions messages
    "EmissionsRequest",
    "EmissionsResponse",
    "ActivityData",
    "EmissionDetail",
    "EmissionFactorsRequest",
    "EmissionFactorsResponse",
    "EmissionFactor",
    # Compliance messages
    "ReportRequest",
    "ReportResponse",
    "ComplianceItem",
    "ComplianceCheckRequest",
    "ComplianceCheckResponse",
    "ComplianceDataPoint",
    "ComplianceViolation",
    # Enums
    "CalculationStatus",
    "ValidationStatus",
    "Scope",
    "ComplianceStatus",
    "Severity",
    # Service stubs
    "ProcessHeatServiceStub",
    "EmissionsServiceStub",
    "ComplianceServiceStub",
    # Servicer base classes
    "ProcessHeatServiceServicer",
    "EmissionsServiceServicer",
    "ComplianceServiceServicer",
    # Registration functions
    "add_ProcessHeatServiceServicer_to_server",
    "add_EmissionsServiceServicer_to_server",
    "add_ComplianceServiceServicer_to_server",
]
