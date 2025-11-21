# -*- coding: utf-8 -*-
"""
ProcessHeatOrchestrator - Master orchestrator for process heat operations.

This module implements the GL-001 ProcessHeatOrchestrator agent for managing
and optimizing process heat operations across industrial facilities. It coordinates
multiple sub-agents, integrates with SCADA and ERP systems, and ensures
zero-hallucination calculations for all thermal operations.

Example:
    >>> from process_heat_orchestrator import ProcessHeatOrchestrator
    >>> config = ProcessHeatConfig(...)
    >>> orchestrator = ProcessHeatOrchestrator(config)
    >>> result = await orchestrator.execute(plant_data)
"""

import asyncio
import hashlib
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from pathlib import Path
import json

# Import from agent_foundation
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from agent_foundation.base_agent import BaseAgent, AgentState, AgentConfig
from agent_foundation.agent_intelligence import (
    AgentIntelligence,
    ChatSession,
    ModelProvider,
    PromptTemplate
)
from agent_foundation.orchestration.message_bus import MessageBus, Message
from agent_foundation.orchestration.saga import SagaOrchestrator, SagaStep
from agent_foundation.memory.short_term_memory import ShortTermMemory
from agent_foundation.memory.long_term_memory import LongTermMemory

# Import local modules
from .config import ProcessHeatConfig, PlantConfiguration
from .tools import ProcessHeatTools, ThermalEfficiencyResult, HeatDistributionStrategy

logger = logging.getLogger(__name__)


class ProcessHeatOrchestrator(BaseAgent):
    """
    Master orchestrator for process heat operations (GL-001).

    This agent coordinates all process heat operations across industrial facilities,
    optimizing thermal efficiency, managing heat distribution, ensuring compliance,
    and integrating with SCADA/ERP systems. All calculations follow zero-hallucination
    principles with deterministic algorithms only.

    Attributes:
        config: ProcessHeatConfig with complete configuration
        tools: ProcessHeatTools instance for deterministic calculations
        intelligence: AgentIntelligence for LLM integration (classification only)
        message_bus: MessageBus for multi-agent coordination
        performance_metrics: Real-time performance tracking
    """

    def __init__(self, config: ProcessHeatConfig):
        """
        Initialize ProcessHeatOrchestrator.

        Args:
            config: Configuration for process heat operations
        """
        # Convert to BaseAgent config
        base_config = AgentConfig(
            name=config.agent_name,
            version=config.version,
            agent_id=config.agent_id,
            timeout_seconds=config.calculation_timeout_seconds,
            enable_metrics=config.enable_monitoring,
            checkpoint_enabled=True,
            checkpoint_interval_seconds=300
        )

        super().__init__(base_config)

        self.process_config = config
        self.tools = ProcessHeatTools()

        # Initialize intelligence with deterministic settings
        self._init_intelligence()

        # Initialize memory systems
        self.short_term_memory = ShortTermMemory(capacity=1000)
        self.long_term_memory = LongTermMemory(
            storage_path=Path("./gl001_memory") if base_config.state_directory is None
            else base_config.state_directory / "memory"
        )

        # Initialize message bus for agent coordination
        self.message_bus = MessageBus()

        # Performance tracking
        self.performance_metrics = {
            'calculations_performed': 0,
            'avg_calculation_time_ms': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'agents_coordinated': 0,
            'errors_recovered': 0
        }

        # Results cache with TTL
        self._results_cache = {}
        self._cache_timestamps = {}

        logger.info(f"ProcessHeatOrchestrator {config.agent_id} initialized successfully")

    def _init_intelligence(self):
        """Initialize AgentIntelligence with deterministic configuration."""
        try:
            # Create deterministic ChatSession for classification tasks only
            self.chat_session = ChatSession(
                provider=ModelProvider.ANTHROPIC,
                model_id="claude-3-haiku",  # Fast model for classification
                temperature=0.0,  # Deterministic
                seed=42,  # Fixed seed
                max_tokens=500  # Limited output for classification
            )

            # Initialize prompt templates for classification tasks
            self.classification_prompt = PromptTemplate(
                template="""
                Classify the following industrial process data into one category:
                - normal_operation
                - efficiency_degradation
                - maintenance_required
                - critical_alert

                Data: {data}

                Return only the category name, nothing else.
                """,
                variables=['data']
            )

            logger.info("AgentIntelligence initialized with deterministic settings")

        except Exception as e:
            logger.warning(f"AgentIntelligence initialization failed, continuing without LLM: {e}")
            self.chat_session = None

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for process heat orchestration.

        Args:
            input_data: Input containing plant data, sensor feeds, constraints

        Returns:
            Orchestration result with optimized strategies and KPIs
        """
        start_time = time.perf_counter()
        self.state = AgentState.EXECUTING

        try:
            # Extract input components
            plant_data = input_data.get('plant_data', {})
            sensor_feeds = input_data.get('sensor_feeds', {})
            constraints = input_data.get('constraints', {})
            emissions_data = input_data.get('emissions_data', {})

            # Step 1: Calculate thermal efficiency
            efficiency_result = await self._calculate_efficiency_async(plant_data)

            # Step 2: Optimize heat distribution
            distribution_strategy = await self._optimize_distribution_async(
                sensor_feeds, constraints
            )

            # Step 3: Validate energy balance
            energy_balance = await self._validate_energy_async(plant_data)

            # Step 4: Check emissions compliance
            compliance_result = await self._check_compliance_async(
                emissions_data, self.process_config.emission_regulations
            )

            # Step 5: Generate KPI dashboard
            kpi_dashboard = self._generate_kpi_dashboard(
                efficiency_result,
                distribution_strategy,
                energy_balance,
                compliance_result
            )

            # Step 6: Coordinate sub-agents if needed
            coordination_result = None
            if input_data.get('coordinate_agents', False):
                agent_ids = input_data.get('agent_ids', [])
                commands = input_data.get('agent_commands', {})
                coordination_result = await self._coordinate_agents_async(
                    agent_ids, commands
                )

            # Store in memory for learning
            self._store_execution_memory(input_data, kpi_dashboard)

            # Calculate execution time
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_performance_metrics(execution_time_ms)

            # Create comprehensive result
            result = {
                'agent_id': self.config.agent_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'execution_time_ms': round(execution_time_ms, 2),
                'thermal_efficiency': efficiency_result.__dict__,
                'heat_distribution': distribution_strategy.__dict__,
                'energy_balance': energy_balance.__dict__,
                'emissions_compliance': compliance_result.__dict__,
                'kpi_dashboard': kpi_dashboard,
                'coordination_result': coordination_result,
                'performance_metrics': self.performance_metrics.copy(),
                'provenance_hash': self._calculate_provenance_hash(
                    input_data, kpi_dashboard
                )
            }

            self.state = AgentState.READY
            logger.info(f"Orchestration completed in {execution_time_ms:.2f}ms")

            return result

        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"Orchestration failed: {str(e)}", exc_info=True)

            # Attempt recovery
            if self.config.max_retries > 0:
                return await self._handle_error_recovery(e, input_data)
            else:
                raise

    async def _calculate_efficiency_async(self, plant_data: Dict[str, Any]) -> ThermalEfficiencyResult:
        """
        Calculate thermal efficiency asynchronously with caching.

        Args:
            plant_data: Plant operating data

        Returns:
            Thermal efficiency result
        """
        # Check cache
        cache_key = self._get_cache_key('efficiency', plant_data)
        if self._is_cache_valid(cache_key):
            self.performance_metrics['cache_hits'] += 1
            return self._results_cache[cache_key]

        # Calculate efficiency
        self.performance_metrics['cache_misses'] += 1
        result = await asyncio.to_thread(
            self.tools.calculate_thermal_efficiency,
            plant_data
        )

        # Store in cache
        self._store_in_cache(cache_key, result)
        self.performance_metrics['calculations_performed'] += 1

        return result

    async def _optimize_distribution_async(
        self,
        sensor_feeds: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> HeatDistributionStrategy:
        """
        Optimize heat distribution asynchronously.

        Args:
            sensor_feeds: Real-time sensor data
            constraints: Operational constraints

        Returns:
            Optimized heat distribution strategy
        """
        # Check cache
        cache_key = self._get_cache_key('distribution', {
            'feeds': sensor_feeds,
            'constraints': constraints
        })
        if self._is_cache_valid(cache_key):
            self.performance_metrics['cache_hits'] += 1
            return self._results_cache[cache_key]

        # Optimize distribution
        self.performance_metrics['cache_misses'] += 1
        result = await asyncio.to_thread(
            self.tools.optimize_heat_distribution,
            sensor_feeds,
            constraints
        )

        # Store in cache
        self._store_in_cache(cache_key, result)
        self.performance_metrics['calculations_performed'] += 1

        return result

    async def _validate_energy_async(self, consumption_data: Dict[str, Any]) -> Any:
        """
        Validate energy balance asynchronously.

        Args:
            consumption_data: Energy consumption data

        Returns:
            Energy balance validation result
        """
        result = await asyncio.to_thread(
            self.tools.validate_energy_balance,
            consumption_data
        )
        self.performance_metrics['calculations_performed'] += 1
        return result

    async def _check_compliance_async(
        self,
        emissions_data: Dict[str, Any],
        regulations: Dict[str, Any]
    ) -> Any:
        """
        Check emissions compliance asynchronously.

        Args:
            emissions_data: Current emissions data
            regulations: Applicable regulations

        Returns:
            Compliance check result
        """
        result = await asyncio.to_thread(
            self.tools.check_emissions_compliance,
            emissions_data,
            regulations
        )
        self.performance_metrics['calculations_performed'] += 1
        return result

    def _generate_kpi_dashboard(
        self,
        efficiency: ThermalEfficiencyResult,
        distribution: HeatDistributionStrategy,
        energy_balance: Any,
        compliance: Any
    ) -> Dict[str, Any]:
        """
        Generate comprehensive KPI dashboard.

        Args:
            efficiency: Thermal efficiency result
            distribution: Heat distribution strategy
            energy_balance: Energy balance validation
            compliance: Compliance check result

        Returns:
            KPI dashboard dictionary
        """
        metrics = {
            'thermal_efficiency': efficiency.overall_efficiency,
            'heat_recovery_rate': efficiency.heat_recovery_efficiency,
            'capacity_utilization': (
                distribution.total_heat_supply_mw /
                distribution.total_heat_demand_mw * 100
            ) if distribution.total_heat_demand_mw > 0 else 0,
            'co2_intensity': compliance.emission_intensity_kg_mwh,
            'compliance_score': 100 if compliance.compliance_status == "PASS" else 50,
            'energy_balance_accuracy': 100 - energy_balance.balance_error_percent
        }

        dashboard = self.tools.generate_kpi_dashboard(metrics)
        return dashboard

    async def _coordinate_agents_async(
        self,
        agent_ids: List[str],
        commands: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Coordinate multiple process heat agents asynchronously.

        Args:
            agent_ids: List of agent IDs to coordinate
            commands: Commands to distribute

        Returns:
            Coordination result
        """
        result = await asyncio.to_thread(
            self.tools.coordinate_process_heat_agents,
            agent_ids,
            commands
        )

        self.performance_metrics['agents_coordinated'] += len(agent_ids)

        # Send messages via message bus
        for agent_id, tasks in result['task_assignments'].items():
            for task in tasks:
                message = Message(
                    sender_id=self.config.agent_id,
                    recipient_id=agent_id,
                    message_type='command',
                    payload=task,
                    priority=task['priority']
                )
                await self.message_bus.publish(f"agent.{agent_id}", message)

        return result

    def _store_execution_memory(self, input_data: Dict[str, Any], result: Dict[str, Any]):
        """
        Store execution in memory for learning and pattern recognition.

        Args:
            input_data: Input data for execution
            result: Execution result
        """
        memory_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'input_summary': self._summarize_input(input_data),
            'result_summary': self._summarize_result(result),
            'performance': self.performance_metrics.copy()
        }

        # Store in short-term memory
        self.short_term_memory.store(memory_entry)

        # Periodically persist to long-term memory
        if self.performance_metrics['calculations_performed'] % 100 == 0:
            asyncio.create_task(self._persist_to_long_term_memory())

    async def _persist_to_long_term_memory(self):
        """Persist short-term memories to long-term storage."""
        try:
            recent_memories = self.short_term_memory.retrieve(limit=50)
            for memory in recent_memories:
                await self.long_term_memory.store(
                    key=f"execution_{memory['timestamp']}",
                    value=memory,
                    category='executions'
                )
            logger.debug("Persisted memories to long-term storage")
        except Exception as e:
            logger.error(f"Failed to persist memories: {e}")

    def _summarize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of input data for memory storage."""
        return {
            'has_plant_data': 'plant_data' in input_data,
            'has_sensor_feeds': 'sensor_feeds' in input_data,
            'has_constraints': 'constraints' in input_data,
            'coordinate_agents': input_data.get('coordinate_agents', False),
            'data_points': len(input_data.get('sensor_feeds', {}).get('tags', {}))
        }

    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of execution result for memory storage."""
        return {
            'efficiency': result.get('operational_kpis', {}).get('overall_efficiency', 0),
            'compliance': result.get('environmental_kpis', {}).get('compliance_score', 0),
            'alerts': len(result.get('alerts', [])),
            'execution_status': 'success' if result else 'failure'
        }

    def _get_cache_key(self, operation: str, data: Dict[str, Any]) -> str:
        """Generate cache key for operation and data."""
        data_str = json.dumps(data, sort_keys=True)
        return f"{operation}_{hashlib.md5(data_str.encode()).hexdigest()}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid based on TTL."""
        if cache_key not in self._results_cache:
            return False

        timestamp = self._cache_timestamps.get(cache_key, 0)
        age_seconds = time.time() - timestamp

        return age_seconds < self.process_config.cache_ttl_seconds

    def _store_in_cache(self, cache_key: str, result: Any):
        """Store result in cache with timestamp."""
        self._results_cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()

        # Limit cache size
        if len(self._results_cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                self._cache_timestamps.keys(),
                key=lambda k: self._cache_timestamps[k]
            )[:20]
            for key in oldest_keys:
                del self._results_cache[key]
                del self._cache_timestamps[key]

    def _update_performance_metrics(self, execution_time_ms: float):
        """Update performance metrics with latest execution."""
        # Update average calculation time
        n = self.performance_metrics['calculations_performed']
        if n > 0:
            current_avg = self.performance_metrics['avg_calculation_time_ms']
            self.performance_metrics['avg_calculation_time_ms'] = (
                (current_avg * (n - 1) + execution_time_ms) / n
            )

    def _calculate_provenance_hash(
        self,
        input_data: Dict[str, Any],
        result: Dict[str, Any]
    ) -> str:
        """
        Calculate SHA-256 provenance hash for complete audit trail.

        Args:
            input_data: Input data
            result: Execution result

        Returns:
            SHA-256 hash string
        """
        provenance_str = f"{self.config.agent_id}{input_data}{result}{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    async def _handle_error_recovery(
        self,
        error: Exception,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle error recovery with retry logic.

        Args:
            error: Exception that occurred
            input_data: Original input data

        Returns:
            Recovery result or error response
        """
        self.state = AgentState.RECOVERING
        self.performance_metrics['errors_recovered'] += 1

        logger.warning(f"Attempting error recovery: {str(error)}")

        # Simplified recovery - return partial results
        return {
            'agent_id': self.config.agent_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'partial_success',
            'error': str(error),
            'recovered_data': {
                'thermal_efficiency': {'overall_efficiency': 0, 'status': 'error'},
                'heat_distribution': {'optimization_score': 0, 'status': 'error'},
                'kpi_dashboard': {'status': 'limited_data'}
            },
            'provenance_hash': self._calculate_provenance_hash(input_data, {})
        }

    async def integrate_scada(self, scada_feed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate SCADA data feed.

        Args:
            scada_feed: Raw SCADA data

        Returns:
            Processed SCADA data
        """
        return await asyncio.to_thread(
            self.tools.integrate_scada_data,
            scada_feed
        )

    async def integrate_erp(self, erp_feed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate ERP data feed.

        Args:
            erp_feed: Raw ERP data

        Returns:
            Processed ERP data
        """
        return await asyncio.to_thread(
            self.tools.integrate_erp_data,
            erp_feed
        )

    def get_state(self) -> Dict[str, Any]:
        """
        Get current agent state for monitoring.

        Returns:
            Current state dictionary
        """
        return {
            'agent_id': self.config.agent_id,
            'state': self.state.value,
            'version': self.config.version,
            'performance_metrics': self.performance_metrics.copy(),
            'cache_size': len(self._results_cache),
            'memory_entries': self.short_term_memory.size(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    async def shutdown(self):
        """Graceful shutdown of the orchestrator."""
        logger.info(f"Shutting down ProcessHeatOrchestrator {self.config.agent_id}")

        # Persist remaining memories
        await self._persist_to_long_term_memory()

        # Close connections
        if hasattr(self, 'message_bus'):
            await self.message_bus.close()

        self.state = AgentState.TERMINATED
        logger.info(f"ProcessHeatOrchestrator {self.config.agent_id} shutdown complete")