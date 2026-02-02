"""
Process Heat gRPC Service Stubs and Servicer Base Classes

This module provides gRPC service stubs for client-side usage and
servicer base classes for server-side implementation. These classes
match the service definitions in process_heat.proto.

Features:
    - Type-safe service stubs with async support
    - Abstract servicer base classes for implementation
    - Server registration functions
    - Full typing for IDE support

Example (Client):
    >>> channel = grpc.aio.insecure_channel('localhost:50051')
    >>> stub = ProcessHeatServiceStub(channel)
    >>> response = await stub.RunCalculation(request)

Example (Server):
    >>> class MyServicer(ProcessHeatServiceServicer):
    ...     async def RunCalculation(self, request, context):
    ...         return CalculationResponse(...)
    >>>
    >>> add_ProcessHeatServiceServicer_to_server(MyServicer(), server)
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Optional, TYPE_CHECKING

import grpc
from grpc import aio as grpc_aio

from greenlang.infrastructure.api.proto.process_heat_pb2 import (
    # Calculation messages
    CalculationRequest,
    CalculationResponse,
    StatusRequest,
    StatusResponse,
    StreamRequest,
    CalculationResult,
    # Emissions messages
    EmissionsRequest,
    EmissionsResponse,
    EmissionFactorsRequest,
    EmissionFactorsResponse,
    # Compliance messages
    ReportRequest,
    ReportResponse,
    ComplianceCheckRequest,
    ComplianceCheckResponse,
)


# ============================================================================
# SERVICE NAMES
# ============================================================================

PROCESS_HEAT_SERVICE_NAME = "greenlang.infrastructure.api.ProcessHeatService"
EMISSIONS_SERVICE_NAME = "greenlang.infrastructure.api.EmissionsService"
COMPLIANCE_SERVICE_NAME = "greenlang.infrastructure.api.ComplianceService"


# ============================================================================
# PROCESS HEAT SERVICE
# ============================================================================

class ProcessHeatServiceStub:
    """
    Client stub for ProcessHeatService.

    Provides async methods for calling ProcessHeatService RPCs.

    Attributes:
        channel: gRPC channel to use for calls

    Example:
        >>> channel = grpc.aio.insecure_channel('localhost:50051')
        >>> stub = ProcessHeatServiceStub(channel)
        >>> response = await stub.RunCalculation(request)
    """

    def __init__(self, channel: grpc_aio.Channel) -> None:
        """
        Initialize stub with gRPC channel.

        Args:
            channel: Async gRPC channel for making calls
        """
        self._channel = channel

        # Unary-Unary RPCs
        self.RunCalculation: grpc_aio.UnaryUnaryMultiCallable = channel.unary_unary(
            f"/{PROCESS_HEAT_SERVICE_NAME}/RunCalculation",
            request_serializer=lambda req: req.SerializeToString(),
            response_deserializer=CalculationResponse.FromString,
        )

        self.GetStatus: grpc_aio.UnaryUnaryMultiCallable = channel.unary_unary(
            f"/{PROCESS_HEAT_SERVICE_NAME}/GetStatus",
            request_serializer=lambda req: req.SerializeToString(),
            response_deserializer=StatusResponse.FromString,
        )

        # Unary-Stream RPC
        self.StreamResults: grpc_aio.UnaryStreamMultiCallable = channel.unary_stream(
            f"/{PROCESS_HEAT_SERVICE_NAME}/StreamResults",
            request_serializer=lambda req: req.SerializeToString(),
            response_deserializer=CalculationResult.FromString,
        )


class ProcessHeatServiceServicer(ABC):
    """
    Abstract base class for ProcessHeatService implementations.

    Subclass this and implement all abstract methods to create
    a ProcessHeatService server.

    Example:
        >>> class MyProcessHeatServicer(ProcessHeatServiceServicer):
        ...     async def RunCalculation(self, request, context):
        ...         # Implementation
        ...         return CalculationResponse(...)
    """

    @abstractmethod
    async def RunCalculation(
        self,
        request: CalculationRequest,
        context: grpc_aio.ServicerContext,
    ) -> CalculationResponse:
        """
        Queue a calculation for processing.

        Args:
            request: Calculation request with equipment and fuel data
            context: gRPC servicer context

        Returns:
            Response with calculation ID and initial status

        Raises:
            grpc.RpcError: On processing failure
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    @abstractmethod
    async def GetStatus(
        self,
        request: StatusRequest,
        context: grpc_aio.ServicerContext,
    ) -> StatusResponse:
        """
        Get current status of a calculation.

        Args:
            request: Status request with calculation ID
            context: gRPC servicer context

        Returns:
            Current status and progress information

        Raises:
            grpc.RpcError: If calculation not found
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    @abstractmethod
    async def StreamResults(
        self,
        request: StreamRequest,
        context: grpc_aio.ServicerContext,
    ) -> AsyncIterator[CalculationResult]:
        """
        Stream calculation results as they complete.

        Args:
            request: Stream request with optional filters
            context: gRPC servicer context

        Yields:
            Calculation results as they become available

        Raises:
            grpc.RpcError: On stream error
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")
        yield  # type: ignore


def add_ProcessHeatServiceServicer_to_server(
    servicer: ProcessHeatServiceServicer,
    server: grpc_aio.Server,
) -> None:
    """
    Register ProcessHeatServiceServicer with a gRPC server.

    Args:
        servicer: Service implementation instance
        server: gRPC server to register with

    Example:
        >>> server = grpc.aio.server()
        >>> add_ProcessHeatServiceServicer_to_server(MyServicer(), server)
    """
    rpc_method_handlers = {
        "RunCalculation": grpc.unary_unary_rpc_method_handler(
            servicer.RunCalculation,
            request_deserializer=CalculationRequest.FromString,
            response_serializer=lambda resp: resp.SerializeToString(),
        ),
        "GetStatus": grpc.unary_unary_rpc_method_handler(
            servicer.GetStatus,
            request_deserializer=StatusRequest.FromString,
            response_serializer=lambda resp: resp.SerializeToString(),
        ),
        "StreamResults": grpc.unary_stream_rpc_method_handler(
            servicer.StreamResults,
            request_deserializer=StreamRequest.FromString,
            response_serializer=lambda resp: resp.SerializeToString(),
        ),
    }

    generic_handler = grpc.method_handlers_generic_handler(
        PROCESS_HEAT_SERVICE_NAME,
        rpc_method_handlers,
    )

    server.add_generic_rpc_handlers((generic_handler,))


# ============================================================================
# EMISSIONS SERVICE
# ============================================================================

class EmissionsServiceStub:
    """
    Client stub for EmissionsService.

    Provides async methods for calling EmissionsService RPCs.

    Attributes:
        channel: gRPC channel to use for calls

    Example:
        >>> channel = grpc.aio.insecure_channel('localhost:50051')
        >>> stub = EmissionsServiceStub(channel)
        >>> response = await stub.CalculateEmissions(request)
    """

    def __init__(self, channel: grpc_aio.Channel) -> None:
        """
        Initialize stub with gRPC channel.

        Args:
            channel: Async gRPC channel for making calls
        """
        self._channel = channel

        self.CalculateEmissions: grpc_aio.UnaryUnaryMultiCallable = channel.unary_unary(
            f"/{EMISSIONS_SERVICE_NAME}/CalculateEmissions",
            request_serializer=lambda req: req.SerializeToString(),
            response_deserializer=EmissionsResponse.FromString,
        )

        self.GetEmissionFactors: grpc_aio.UnaryUnaryMultiCallable = channel.unary_unary(
            f"/{EMISSIONS_SERVICE_NAME}/GetEmissionFactors",
            request_serializer=lambda req: req.SerializeToString(),
            response_deserializer=EmissionFactorsResponse.FromString,
        )


class EmissionsServiceServicer(ABC):
    """
    Abstract base class for EmissionsService implementations.

    Subclass this and implement all abstract methods to create
    an EmissionsService server.

    Example:
        >>> class MyEmissionsServicer(EmissionsServiceServicer):
        ...     async def CalculateEmissions(self, request, context):
        ...         return EmissionsResponse(...)
    """

    @abstractmethod
    async def CalculateEmissions(
        self,
        request: EmissionsRequest,
        context: grpc_aio.ServicerContext,
    ) -> EmissionsResponse:
        """
        Calculate emissions from activity data.

        Args:
            request: Emissions request with activity data
            context: gRPC servicer context

        Returns:
            Calculated emissions with breakdown

        Raises:
            grpc.RpcError: On calculation failure
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    @abstractmethod
    async def GetEmissionFactors(
        self,
        request: EmissionFactorsRequest,
        context: grpc_aio.ServicerContext,
    ) -> EmissionFactorsResponse:
        """
        Retrieve emission factors for a fuel type.

        Args:
            request: Request with fuel type and filters
            context: gRPC servicer context

        Returns:
            Matching emission factors

        Raises:
            grpc.RpcError: On retrieval failure
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_EmissionsServiceServicer_to_server(
    servicer: EmissionsServiceServicer,
    server: grpc_aio.Server,
) -> None:
    """
    Register EmissionsServiceServicer with a gRPC server.

    Args:
        servicer: Service implementation instance
        server: gRPC server to register with

    Example:
        >>> server = grpc.aio.server()
        >>> add_EmissionsServiceServicer_to_server(MyServicer(), server)
    """
    rpc_method_handlers = {
        "CalculateEmissions": grpc.unary_unary_rpc_method_handler(
            servicer.CalculateEmissions,
            request_deserializer=EmissionsRequest.FromString,
            response_serializer=lambda resp: resp.SerializeToString(),
        ),
        "GetEmissionFactors": grpc.unary_unary_rpc_method_handler(
            servicer.GetEmissionFactors,
            request_deserializer=EmissionFactorsRequest.FromString,
            response_serializer=lambda resp: resp.SerializeToString(),
        ),
    }

    generic_handler = grpc.method_handlers_generic_handler(
        EMISSIONS_SERVICE_NAME,
        rpc_method_handlers,
    )

    server.add_generic_rpc_handlers((generic_handler,))


# ============================================================================
# COMPLIANCE SERVICE
# ============================================================================

class ComplianceServiceStub:
    """
    Client stub for ComplianceService.

    Provides async methods for calling ComplianceService RPCs.

    Attributes:
        channel: gRPC channel to use for calls

    Example:
        >>> channel = grpc.aio.insecure_channel('localhost:50051')
        >>> stub = ComplianceServiceStub(channel)
        >>> response = await stub.GenerateReport(request)
    """

    def __init__(self, channel: grpc_aio.Channel) -> None:
        """
        Initialize stub with gRPC channel.

        Args:
            channel: Async gRPC channel for making calls
        """
        self._channel = channel

        self.GenerateReport: grpc_aio.UnaryUnaryMultiCallable = channel.unary_unary(
            f"/{COMPLIANCE_SERVICE_NAME}/GenerateReport",
            request_serializer=lambda req: req.SerializeToString(),
            response_deserializer=ReportResponse.FromString,
        )

        self.CheckCompliance: grpc_aio.UnaryUnaryMultiCallable = channel.unary_unary(
            f"/{COMPLIANCE_SERVICE_NAME}/CheckCompliance",
            request_serializer=lambda req: req.SerializeToString(),
            response_deserializer=ComplianceCheckResponse.FromString,
        )


class ComplianceServiceServicer(ABC):
    """
    Abstract base class for ComplianceService implementations.

    Subclass this and implement all abstract methods to create
    a ComplianceService server.

    Example:
        >>> class MyComplianceServicer(ComplianceServiceServicer):
        ...     async def GenerateReport(self, request, context):
        ...         return ReportResponse(...)
    """

    @abstractmethod
    async def GenerateReport(
        self,
        request: ReportRequest,
        context: grpc_aio.ServicerContext,
    ) -> ReportResponse:
        """
        Generate a compliance report.

        Args:
            request: Report request with framework and facility
            context: gRPC servicer context

        Returns:
            Generated compliance report

        Raises:
            grpc.RpcError: On report generation failure
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    @abstractmethod
    async def CheckCompliance(
        self,
        request: ComplianceCheckRequest,
        context: grpc_aio.ServicerContext,
    ) -> ComplianceCheckResponse:
        """
        Check compliance against a regulation.

        Args:
            request: Check request with regulation and data points
            context: gRPC servicer context

        Returns:
            Compliance check result with violations

        Raises:
            grpc.RpcError: On check failure
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_ComplianceServiceServicer_to_server(
    servicer: ComplianceServiceServicer,
    server: grpc_aio.Server,
) -> None:
    """
    Register ComplianceServiceServicer with a gRPC server.

    Args:
        servicer: Service implementation instance
        server: gRPC server to register with

    Example:
        >>> server = grpc.aio.server()
        >>> add_ComplianceServiceServicer_to_server(MyServicer(), server)
    """
    rpc_method_handlers = {
        "GenerateReport": grpc.unary_unary_rpc_method_handler(
            servicer.GenerateReport,
            request_deserializer=ReportRequest.FromString,
            response_serializer=lambda resp: resp.SerializeToString(),
        ),
        "CheckCompliance": grpc.unary_unary_rpc_method_handler(
            servicer.CheckCompliance,
            request_deserializer=ComplianceCheckRequest.FromString,
            response_serializer=lambda resp: resp.SerializeToString(),
        ),
    }

    generic_handler = grpc.method_handlers_generic_handler(
        COMPLIANCE_SERVICE_NAME,
        rpc_method_handlers,
    )

    server.add_generic_rpc_handlers((generic_handler,))


# ============================================================================
# REFLECTION SUPPORT
# ============================================================================

# Service names for reflection registration
SERVICE_NAMES = (
    PROCESS_HEAT_SERVICE_NAME,
    EMISSIONS_SERVICE_NAME,
    COMPLIANCE_SERVICE_NAME,
)


def get_service_names() -> tuple:
    """
    Get all service names for reflection registration.

    Returns:
        Tuple of fully-qualified service names

    Example:
        >>> from grpc_reflection.v1alpha import reflection
        >>> reflection.enable_server_reflection(get_service_names(), server)
    """
    return SERVICE_NAMES
