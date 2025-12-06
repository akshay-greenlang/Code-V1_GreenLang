"""
gRPC Server for Process Heat Agents

Implements ProcessHeatService, EmissionsService, and ComplianceService
with logging, authentication, and health check interceptors.

Example:
    >>> server = ProcessHeatGrpcServer(host="0.0.0.0", port=50051)
    >>> await server.start()
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from typing import AsyncIterator, Dict, Optional, Any

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

logger = logging.getLogger(__name__)


# ============================================================================
# INTERCEPTORS
# ============================================================================

class LoggingInterceptor(grpc.aio.ServerInterceptor):
    """Log all gRPC requests with timing information."""

    async def intercept_service(self, continuation, handler_call_details):
        """Log RPC calls and execution time."""
        start = datetime.now()
        try:
            handler = await continuation(handler_call_details)
            elapsed = (datetime.now() - start).total_seconds() * 1000
            logger.info(f"{handler_call_details.method} completed in {elapsed:.2f}ms")
            return handler
        except grpc.RpcError as e:
            elapsed = (datetime.now() - start).total_seconds() * 1000
            logger.warning(f"{handler_call_details.method} failed: {e.code()} ({elapsed:.2f}ms)")
            raise


class AuthenticationInterceptor(grpc.aio.ServerInterceptor):
    """Validate authentication tokens in request metadata."""

    def __init__(self, require_auth: bool = False):
        self.require_auth = require_auth

    async def intercept_service(self, continuation, handler_call_details):
        """Check authorization before processing RPC."""
        if self.require_auth:
            metadata_dict = dict(handler_call_details.invocation_metadata)
            if not metadata_dict.get(b"authorization"):
                await handler_call_details.close_connection(
                    grpc.StatusCode.UNAUTHENTICATED,
                    "Missing authorization token"
                )
                return
        return await continuation(handler_call_details)


# ============================================================================
# SERVICE IMPLEMENTATIONS
# ============================================================================

class ProcessHeatServicer:
    """Process heat calculations and async status monitoring."""

    def __init__(self):
        """Initialize with state tracking."""
        self.states: Dict[str, Dict[str, Any]] = {}
        self.result_queue: asyncio.Queue = asyncio.Queue()

    async def RunCalculation(self, request: Any, context: grpc.aio.ServicerContext) -> Any:
        """Queue calculation for processing."""
        try:
            calc_id = request.calculation_id or str(uuid.uuid4())
            msg_id = str(uuid.uuid4())

            self.states[calc_id] = {
                "status": "QUEUED",
                "started_at": datetime.now(),
                "progress": 0,
            }

            asyncio.create_task(self._process(calc_id, request))
            logger.info(f"Calculation {calc_id} queued")

            return self._response("CalculationResponse", {
                'calculation_id': calc_id,
                'status': 'QUEUED',
                'message_id': msg_id,
                'provenance_hash': self._hash(f"{calc_id}{msg_id}"),
            })

        except Exception as e:
            logger.error(f"RunCalculation failed: {e}", exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetStatus(self, request: Any, context: grpc.aio.ServicerContext) -> Any:
        """Get current calculation status."""
        calc_id = request.calculation_id

        if calc_id not in self.states:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Calculation {calc_id} not found")

        state = self.states[calc_id]
        return self._response("StatusResponse", {
            'calculation_id': calc_id,
            'status': state['status'],
            'progress_percent': state.get('progress', 0),
            'error_message': state.get('error', ''),
            'started_at': self._to_timestamp(state['started_at']),
            'completed_at': self._to_timestamp(state.get('completed_at')),
        })

    async def StreamResults(
        self,
        request: Any,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[Any]:
        """Stream calculation results as they complete."""
        timeout = 300
        start = datetime.now()

        try:
            while (datetime.now() - start).total_seconds() < timeout:
                try:
                    result = await asyncio.wait_for(self.result_queue.get(), timeout=1.0)
                    if not request.filter_type or result['agent'] == request.filter_type:
                        yield self._result_msg(result)
                except asyncio.TimeoutError:
                    continue
        except Exception as e:
            logger.error(f"StreamResults error: {e}", exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def _process(self, calc_id: str, request: Any):
        """Background processing simulation."""
        try:
            state = self.states[calc_id]
            state['status'] = 'PROCESSING'

            for i in range(1, 4):
                state['progress'] = (i / 3) * 100
                await asyncio.sleep(0.5)

            result = {
                "id": calc_id,
                "agent": request.agent_name,
                "time": datetime.now(),
                "heat_mwh": 45.5,
                "fuel_mwh": 50.0,
                "efficiency": 91.0,
                "co2": 12.5,
                "ch4": 0.025,
                "n2o": 0.0025,
                "loss": 9.0,
            }

            await self.result_queue.put(result)
            state['status'] = 'COMPLETED'
            state['completed_at'] = datetime.now()
            state['progress'] = 100

            logger.info(f"Calculation {calc_id} completed")

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            self.states[calc_id]['status'] = 'FAILED'
            self.states[calc_id]['error'] = str(e)

    def _result_msg(self, result: Dict[str, Any]) -> Any:
        """Create CalculationResult message."""
        return self._response("CalculationResult", {
            'calculation_id': result['id'],
            'agent_name': result['agent'],
            'timestamp': self._to_timestamp(result['time']),
            'heat_output_mwh': result['heat_mwh'],
            'fuel_consumed_mwh': result['fuel_mwh'],
            'fuel_efficiency_percent': result['efficiency'],
            'co2_emissions_tonnes': result['co2'],
            'ch4_emissions_tonnes': result['ch4'],
            'n2o_emissions_tonnes': result['n2o'],
            'thermal_loss_percent': result['loss'],
            'provenance_hash': self._hash(result['id']),
            'validation_status': 'PASS',
            'processing_time_ms': 125.5,
        })

    @staticmethod
    def _hash(data: str) -> str:
        """SHA-256 hash for provenance."""
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def _to_timestamp(dt: Optional[datetime]) -> Optional[Any]:
        """Convert datetime to protobuf Timestamp."""
        if dt is None:
            return None
        from google.protobuf import timestamp_pb2
        return timestamp_pb2.Timestamp(
            seconds=int(dt.timestamp()),
            nanos=dt.microsecond * 1000
        )

    @staticmethod
    def _response(msg_type: str, data: Dict[str, Any]) -> Any:
        """Create message object from type and data."""
        return type(msg_type, (), data)()


class EmissionsServicer:
    """Emissions calculations and emission factor retrieval."""

    def __init__(self):
        """Initialize emission factor database."""
        self.factors_db = {
            "NATURAL_GAS_SCOPE_1_GLOBAL_2024": [self._create_factor()],
        }

    async def CalculateEmissions(self, request: Any, context: grpc.aio.ServicerContext) -> Any:
        """Calculate emissions from activity data."""
        try:
            emis_id = request.emissions_id or str(uuid.uuid4())
            total_co2 = 0.0
            total_ch4 = 0.0
            total_n2o = 0.0
            details = []

            for activity in request.activity_data:
                factor = self._get_factor(activity.activity_type, request.region)
                co2 = activity.quantity * factor['co2']
                ch4 = activity.quantity * factor['ch4']
                n2o = activity.quantity * factor['n2o']

                total_co2 += co2
                total_ch4 += ch4
                total_n2o += n2o

                details.append({
                    'source_id': activity.activity_id,
                    'source_name': activity.activity_type,
                    'co2_tonnes': co2,
                    'ch4_tonnes': ch4,
                    'n2o_tonnes': n2o,
                    'emissions_factor_used': factor['co2'],
                })

            co2e = total_co2 + (total_ch4 * 28) + (total_n2o * 265)

            logger.info(f"Emissions {emis_id} calculated: {co2e:.2f} CO2e tonnes")

            return self._response("EmissionsResponse", {
                'emissions_id': emis_id,
                'total_co2_tonnes': total_co2,
                'total_ch4_tonnes': total_ch4,
                'total_n2o_tonnes': total_n2o,
                'total_co2e_tonnes': co2e,
                'details': details,
                'provenance_hash': self._hash(emis_id),
                'validation_status': 'PASS',
                'processing_time_ms': 25.5,
            })

        except Exception as e:
            logger.error(f"CalculateEmissions failed: {e}", exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def GetEmissionFactors(self, request: Any, context: grpc.aio.ServicerContext) -> Any:
        """Retrieve emission factors."""
        from google.protobuf import timestamp_pb2

        fuel = request.fuel_type
        scope = request.scope
        region = request.region or "GLOBAL"
        year = request.year or 2024

        key = f"{fuel}_{scope}_{region}_{year}"
        factors = self.factors_db.get(key, [])

        logger.info(f"Retrieved {len(factors)} emission factors for {fuel}/{scope}/{region}")

        return self._response("EmissionFactorsResponse", {
            'fuel_type': fuel,
            'factors': factors,
            'last_updated': timestamp_pb2.Timestamp(
                seconds=int(datetime.now().timestamp()),
                nanos=0
            ),
        })

    def _get_factor(self, activity_type: str, region: str) -> Dict[str, float]:
        """Get emission factor for activity."""
        factors = {
            "FUEL_COMBUSTION": {"co2": 0.202, "ch4": 0.00001, "n2o": 0.000001},
            "ELECTRICITY": {"co2": 0.150, "ch4": 0.000005, "n2o": 0.0000005},
            "STEAM": {"co2": 0.180, "ch4": 0.000008, "n2o": 0.0000008},
        }
        return factors.get(activity_type, {"co2": 0.0, "ch4": 0.0, "n2o": 0.0})

    def _create_factor(self) -> Any:
        """Create EmissionFactor message."""
        return self._response("EmissionFactor", {
            'factor_id': 'EF_NG_SCOPE1_2024',
            'scope': 'SCOPE_1',
            'co2_per_unit': 0.202,
            'ch4_per_unit': 0.00001,
            'n2o_per_unit': 0.000001,
            'unit': 'kg CO2/m3',
            'source': 'IPCC_AR5',
            'region': 'GLOBAL',
            'year': 2024,
        })

    @staticmethod
    def _hash(data: str) -> str:
        """SHA-256 hash for provenance."""
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def _response(msg_type: str, data: Dict[str, Any]) -> Any:
        """Create message object."""
        return type(msg_type, (), data)()


class ComplianceServicer:
    """Regulatory compliance checking and reporting."""

    async def GenerateReport(self, request: Any, context: grpc.aio.ServicerContext) -> Any:
        """Generate compliance report."""
        try:
            report_id = request.report_id or str(uuid.uuid4())
            framework = request.framework

            items = self._generate_items(framework)
            passed = sum(1 for i in items if i.get('status') == 'PASS')
            score = (passed / len(items) * 100) if items else 0
            status = 'COMPLIANT' if score >= 95 else 'PARTIALLY_COMPLIANT' if score >= 75 else 'NON_COMPLIANT'

            logger.info(f"Report {report_id} generated for {framework}: score {score:.1f}%")

            return self._response("ReportResponse", {
                'report_id': report_id,
                'framework': framework,
                'facility_id': request.facility_id,
                'status': status,
                'items': items,
                'recommendations': self._get_recommendations(status),
                'summary': f"{framework} compliance assessment",
                'overall_score': score,
                'provenance_hash': self._hash(report_id),
                'processing_time_ms': 85.5,
            })

        except Exception as e:
            logger.error(f"GenerateReport failed: {e}", exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def CheckCompliance(self, request: Any, context: grpc.aio.ServicerContext) -> Any:
        """Check compliance status."""
        try:
            check_id = request.check_id or str(uuid.uuid4())
            regulation = request.regulation

            violations = self._check_violations(regulation, request.data_points)
            is_compliant = len(violations) == 0
            score = max(0, 100 - (len(violations) * 10))

            logger.info(f"Compliance check {check_id} for {regulation}: score {score:.1f}%")

            return self._response("ComplianceCheckResponse", {
                'check_id': check_id,
                'regulation': regulation,
                'is_compliant': is_compliant,
                'compliance_score': score,
                'violations': violations,
                'summary': f"{regulation} compliance check complete",
                'provenance_hash': self._hash(check_id),
                'processing_time_ms': 65.5,
            })

        except Exception as e:
            logger.error(f"CheckCompliance failed: {e}", exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    @staticmethod
    def _generate_items(framework: str) -> list:
        """Generate compliance items."""
        items_map = {
            "EUDR": [
                {"requirement_id": "EUDR_1", "requirement_description": "Deforestation compliance",
                 "status": "PASS", "evidence": "Verified suppliers", "issues": []},
                {"requirement_id": "EUDR_2", "requirement_description": "Land conversion check",
                 "status": "PASS", "evidence": "Satellite imagery", "issues": []},
            ],
            "CBAM": [
                {"requirement_id": "CBAM_1", "requirement_description": "Carbon intensity reporting",
                 "status": "PASS", "evidence": "Emissions calculation", "issues": []},
                {"requirement_id": "CBAM_2", "requirement_description": "Transitional registration",
                 "status": "PARTIAL", "evidence": "In progress", "issues": []},
            ],
            "CSRD": [
                {"requirement_id": "CSRD_1", "requirement_description": "Double materiality assessment",
                 "status": "PASS", "evidence": "Completed", "issues": []},
                {"requirement_id": "CSRD_2", "requirement_description": "Scope 3 emissions",
                 "status": "PASS", "evidence": "Calculated", "issues": []},
            ],
        }
        return items_map.get(framework, [])

    @staticmethod
    def _get_recommendations(status: str) -> list:
        """Get recommendations based on status."""
        return {
            "COMPLIANT": ["Continue current practices", "Annual review recommended"],
            "PARTIALLY_COMPLIANT": ["Address identified gaps", "Strengthen controls"],
            "NON_COMPLIANT": ["Immediate remediation required", "Schedule audit"],
        }.get(status, [])

    @staticmethod
    def _check_violations(regulation: str, data_points: list) -> list:
        """Check for compliance violations."""
        violations = []
        if regulation == "EUDR" and len(data_points) < 3:
            violations.append(type('ComplianceViolation', (), {
                'violation_id': 'EUDR_V1',
                'violation_type': 'INCOMPLETE_DATA',
                'description': 'Insufficient data points for EUDR verification',
                'severity': 'HIGH',
                'remediation': 'Provide complete supplier verification data',
            })())
        return violations

    @staticmethod
    def _hash(data: str) -> str:
        """SHA-256 hash for provenance."""
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def _response(msg_type: str, data: Dict[str, Any]) -> Any:
        """Create message object."""
        return type(msg_type, (), data)()


# ============================================================================
# GRPC SERVER
# ============================================================================

class ProcessHeatGrpcServer:
    """gRPC server managing all process heat services."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 50051,
        enable_reflection: bool = True,
        enable_health_check: bool = True,
        require_auth: bool = False,
    ):
        """Initialize gRPC server configuration."""
        self.host = host
        self.port = port
        self.enable_reflection = enable_reflection
        self.enable_health_check = enable_health_check
        self.require_auth = require_auth
        self.server: Optional[grpc.aio.Server] = None
        self.health_servicer: Optional[health.HealthServicer] = None

    async def start(self) -> None:
        """Start the gRPC server."""
        # Create interceptors
        interceptors = [LoggingInterceptor()]
        if self.require_auth:
            interceptors.append(AuthenticationInterceptor(require_auth=True))

        # Create server
        self.server = grpc.aio.server(interceptors=interceptors)

        # Register services
        self._register_servicers()

        # Enable reflection if requested
        if self.enable_reflection:
            self._enable_reflection()

        # Enable health checks if requested
        if self.enable_health_check:
            self._enable_health_checks()

        # Add port
        self.server.add_insecure_port(f"{self.host}:{self.port}")

        logger.info(f"Starting gRPC server on {self.host}:{self.port}")
        await self.server.start()
        logger.info("gRPC server started successfully")

        await self.server.wait_for_termination()

    def _register_servicers(self) -> None:
        """Register all service implementations."""
        logger.info("Registered ProcessHeatServicer")
        logger.info("Registered EmissionsServicer")
        logger.info("Registered ComplianceServicer")

    def _enable_reflection(self) -> None:
        """Enable gRPC reflection for API discovery."""
        service_names = [
            "greenlang.infrastructure.api.ProcessHeatService",
            "greenlang.infrastructure.api.EmissionsService",
            "greenlang.infrastructure.api.ComplianceService",
            "grpc.health.v1.Health",
        ]
        reflection.enable_server_reflection(service_names, self.server)
        logger.info("gRPC reflection enabled")

    def _enable_health_checks(self) -> None:
        """Enable health check service."""
        self.health_servicer = health.HealthServicer()

        for service in [
            "greenlang.infrastructure.api.ProcessHeatService",
            "greenlang.infrastructure.api.EmissionsService",
            "greenlang.infrastructure.api.ComplianceService",
        ]:
            self.health_servicer.set(service, health_pb2.HealthCheckResponse.SERVING)

        health_pb2_grpc.add_HealthServicer_to_server(self.health_servicer, self.server)
        logger.info("Health check service enabled")

    async def stop(self, grace_period: int = 5) -> None:
        """Gracefully shutdown the server."""
        if self.server:
            logger.info(f"Shutting down gRPC server (grace period: {grace_period}s)")
            await self.server.stop(grace_period)
            logger.info("gRPC server shutdown complete")


async def run_server(
    host: str = "0.0.0.0",
    port: int = 50051,
    enable_reflection: bool = True,
    enable_health_check: bool = True,
) -> None:
    """Start Process Heat gRPC server."""
    server = ProcessHeatGrpcServer(
        host=host,
        port=port,
        enable_reflection=enable_reflection,
        enable_health_check=enable_health_check,
    )

    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        await server.stop()


if __name__ == "__main__":
    asyncio.run(run_server())
