"""
GreenLang Agent Orchestrator

Orchestrates execution of 18 agents across the CSRD ecosystem:
- 6 Core Pipeline Agents
- 14 GreenLang Platform Agents
- 4 CSRD Domain Agents (newly added)

Supports sequential, parallel, and event-driven execution patterns.

Author: GreenLang AI Team
Date: 2025-10-18
"""

from typing import Dict, List, Any, Optional, Callable
import asyncio
from pathlib import Path
import yaml
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class AgentExecutionResult:
    """Result from agent execution"""

    def __init__(
        self,
        agent_name: str,
        status: str,
        duration_seconds: float,
        output: Any = None,
        error: Optional[str] = None
    ):
        self.agent_name = agent_name
        self.status = status  # 'success', 'failed', 'skipped'
        self.duration_seconds = duration_seconds
        self.output = output
        self.error = error
        self.timestamp = datetime.now().isoformat()


class GreenLangAgentOrchestrator:
    """
    Orchestrate GreenLang agents for CSRD pipeline

    Manages execution of:
    - Core Pipeline Agents (6)
    - GreenLang Platform Agents (14)
    - CSRD Domain Agents (4)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Agent Orchestrator

        Args:
            config_path: Path to agent configuration YAML
        """
        self.config = self._load_config(config_path)
        self.agents = {}
        self.execution_history = []

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load agent orchestration configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)

        # Default configuration
        return {
            'core_pipeline': {
                'enabled': True,
                'agents': ['intake', 'materiality', 'calculator', 'aggregator', 'reporting', 'audit']
            },
            'greenlang_platform': {
                'enabled': True,
                'workflows': {
                    'development_quality': ['codesentinel', 'secscan', 'spec_guardian'],
                    'data_pipeline': ['dataflow_guardian', 'determinism_auditor'],
                    'release_readiness': ['packqc', 'exitbar_auditor']
                }
            },
            'csrd_domain': {
                'enabled': True,
                'agents': ['regulatory_intelligence', 'data_collection', 'supply_chain', 'automated_filing']
            }
        }

    def register_agent(
        self,
        agent_name: str,
        agent_callable: Callable,
        agent_type: str = 'custom'
    ):
        """
        Register an agent for orchestration

        Args:
            agent_name: Unique agent identifier
            agent_callable: Async callable that executes the agent
            agent_type: Type of agent ('core', 'platform', 'domain', 'custom')
        """
        self.agents[agent_name] = {
            'callable': agent_callable,
            'type': agent_type,
            'registered_at': datetime.now().isoformat()
        }
        logger.info(f"Registered agent: {agent_name} (type: {agent_type})")

    async def run_sequential(
        self,
        agent_names: List[str],
        context: Dict[str, Any]
    ) -> List[AgentExecutionResult]:
        """
        Run agents sequentially (one after another)

        Args:
            agent_names: List of agent names to execute
            context: Shared context dictionary

        Returns:
            List of execution results
        """
        logger.info(f"Starting sequential execution of {len(agent_names)} agents")
        results = []

        for agent_name in agent_names:
            result = await self._execute_agent(agent_name, context)
            results.append(result)

            # Stop on failure if configured
            if result.status == 'failed' and self.config.get('stop_on_failure', True):
                logger.error(f"Stopping sequential execution due to failure in {agent_name}")
                break

            # Update context with result
            context[f'{agent_name}_result'] = result.output

        logger.info(f"Sequential execution complete: {len(results)} agents executed")
        return results

    async def run_parallel(
        self,
        agent_names: List[str],
        context: Dict[str, Any]
    ) -> List[AgentExecutionResult]:
        """
        Run agents in parallel (all at once)

        Args:
            agent_names: List of agent names to execute
            context: Shared context dictionary

        Returns:
            List of execution results
        """
        logger.info(f"Starting parallel execution of {len(agent_names)} agents")

        # Create tasks for all agents
        tasks = [
            self._execute_agent(agent_name, context)
            for agent_name in agent_names
        ]

        # Execute all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(AgentExecutionResult(
                    agent_name=agent_names[i],
                    status='failed',
                    duration_seconds=0,
                    error=str(result)
                ))
            else:
                processed_results.append(result)

        logger.info(f"Parallel execution complete: {len(processed_results)} agents executed")
        return processed_results

    async def run_workflow(
        self,
        workflow_name: str,
        context: Dict[str, Any],
        execution_mode: str = 'sequential'
    ) -> Dict[str, Any]:
        """
        Run a predefined workflow

        Args:
            workflow_name: Name of workflow to execute
            context: Shared context
            execution_mode: 'sequential' or 'parallel'

        Returns:
            Workflow execution summary
        """
        logger.info(f"Starting workflow: {workflow_name} (mode: {execution_mode})")

        # Get workflow agents
        workflow_agents = self._get_workflow_agents(workflow_name)

        if not workflow_agents:
            logger.error(f"Workflow {workflow_name} not found or has no agents")
            return {
                'workflow': workflow_name,
                'status': 'failed',
                'error': 'Workflow not found'
            }

        # Execute workflow
        if execution_mode == 'sequential':
            results = await self.run_sequential(workflow_agents, context)
        else:
            results = await self.run_parallel(workflow_agents, context)

        # Calculate summary
        successful = sum(1 for r in results if r.status == 'success')
        failed = sum(1 for r in results if r.status == 'failed')

        summary = {
            'workflow': workflow_name,
            'status': 'success' if failed == 0 else 'failed',
            'total_agents': len(results),
            'successful': successful,
            'failed': failed,
            'execution_mode': execution_mode,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(
            f"Workflow {workflow_name} complete: "
            f"{successful}/{len(results)} agents succeeded"
        )

        return summary

    def _get_workflow_agents(self, workflow_name: str) -> List[str]:
        """Get list of agents for a workflow"""
        # Check core pipeline
        if workflow_name == 'core_pipeline':
            return self.config.get('core_pipeline', {}).get('agents', [])

        # Check GreenLang platform workflows
        gl_workflows = self.config.get('greenlang_platform', {}).get('workflows', {})
        if workflow_name in gl_workflows:
            return gl_workflows[workflow_name]

        # Check CSRD domain
        if workflow_name == 'csrd_domain':
            return self.config.get('csrd_domain', {}).get('agents', [])

        return []

    async def _execute_agent(
        self,
        agent_name: str,
        context: Dict[str, Any]
    ) -> AgentExecutionResult:
        """
        Execute a single agent

        Args:
            agent_name: Name of agent to execute
            context: Execution context

        Returns:
            Execution result
        """
        import time

        start_time = time.time()

        # Check if agent is registered
        if agent_name not in self.agents:
            logger.warning(f"Agent {agent_name} not registered, skipping")
            return AgentExecutionResult(
                agent_name=agent_name,
                status='skipped',
                duration_seconds=0,
                error='Agent not registered'
            )

        try:
            logger.info(f"Executing agent: {agent_name}")

            # Get agent callable
            agent_info = self.agents[agent_name]
            agent_callable = agent_info['callable']

            # Execute agent
            output = await agent_callable(context)

            duration = time.time() - start_time

            result = AgentExecutionResult(
                agent_name=agent_name,
                status='success',
                duration_seconds=duration,
                output=output
            )

            logger.info(f"Agent {agent_name} completed successfully in {duration:.2f}s")

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Agent {agent_name} failed: {str(e)}")

            result = AgentExecutionResult(
                agent_name=agent_name,
                status='failed',
                duration_seconds=duration,
                error=str(e)
            )

        # Save to history
        self.execution_history.append(result)

        return result

    async def run_csrd_full_pipeline(
        self,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run complete CSRD pipeline with all agents

        This orchestrates:
        1. Domain agents (data collection, regulatory check)
        2. Core pipeline (intake → calculate → audit → report)
        3. Platform agents (quality gates, validation)
        4. Filing (automated submission)

        Args:
            input_data: Input data and configuration

        Returns:
            Complete pipeline results
        """
        logger.info("="*60)
        logger.info("STARTING FULL CSRD PIPELINE")
        logger.info("="*60)

        context = {
            'input_data': input_data,
            'timestamp': datetime.now().isoformat()
        }

        pipeline_results = {}

        # Stage 1: Domain Agents (Parallel)
        logger.info("\n--- Stage 1: CSRD Domain Agents ---")
        domain_results = await self.run_workflow(
            'csrd_domain',
            context,
            execution_mode='parallel'
        )
        pipeline_results['domain'] = domain_results

        # Stage 2: Core Pipeline (Sequential)
        logger.info("\n--- Stage 2: Core Pipeline ---")
        core_results = await self.run_workflow(
            'core_pipeline',
            context,
            execution_mode='sequential'
        )
        pipeline_results['core'] = core_results

        # Stage 3: Quality Gates (Parallel)
        logger.info("\n--- Stage 3: Quality Gates ---")
        quality_results = await self.run_workflow(
            'development_quality',
            context,
            execution_mode='parallel'
        )
        pipeline_results['quality'] = quality_results

        # Stage 4: Release Readiness (Sequential)
        logger.info("\n--- Stage 4: Release Readiness ---")
        release_results = await self.run_workflow(
            'release_readiness',
            context,
            execution_mode='sequential'
        )
        pipeline_results['release'] = release_results

        # Calculate overall status
        all_results = [
            domain_results,
            core_results,
            quality_results,
            release_results
        ]

        overall_status = 'success' if all(
            r['status'] == 'success' for r in all_results
        ) else 'failed'

        logger.info("\n" + "="*60)
        logger.info(f"PIPELINE COMPLETE - Status: {overall_status.upper()}")
        logger.info("="*60)

        return {
            'status': overall_status,
            'stages': pipeline_results,
            'timestamp': datetime.now().isoformat()
        }

    def save_execution_history(self, output_path: str):
        """
        Save execution history to JSON file

        Args:
            output_path: Path to save history
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        history_data = [
            {
                'agent_name': r.agent_name,
                'status': r.status,
                'duration_seconds': r.duration_seconds,
                'timestamp': r.timestamp,
                'error': r.error
            }
            for r in self.execution_history
        ]

        with open(output_file, 'w') as f:
            json.dump(history_data, f, indent=2)

        logger.info(f"Execution history saved to {output_file}")


# Example agent callables (mocks for demonstration)

async def mock_intake_agent(context):
    """Mock intake agent"""
    await asyncio.sleep(0.1)
    return {'status': 'success', 'records_processed': 1000}

async def mock_calculator_agent(context):
    """Mock calculator agent"""
    await asyncio.sleep(0.2)
    return {'status': 'success', 'metrics_calculated': 547}

async def mock_codesentinel_agent(context):
    """Mock code quality agent"""
    await asyncio.sleep(0.05)
    return {'status': 'success', 'quality_score': 95}


# Example usage
async def main():
    """Example orchestration"""
    # Initialize orchestrator
    orchestrator = GreenLangAgentOrchestrator()

    # Register mock agents
    orchestrator.register_agent('intake', mock_intake_agent, 'core')
    orchestrator.register_agent('calculator', mock_calculator_agent, 'core')
    orchestrator.register_agent('codesentinel', mock_codesentinel_agent, 'platform')

    # Test sequential execution
    print("\n=== Sequential Execution ===")
    seq_results = await orchestrator.run_sequential(
        ['intake', 'calculator'],
        context={'test': True}
    )

    for result in seq_results:
        print(f"{result.agent_name}: {result.status} ({result.duration_seconds:.2f}s)")

    # Test parallel execution
    print("\n=== Parallel Execution ===")
    par_results = await orchestrator.run_parallel(
        ['intake', 'calculator', 'codesentinel'],
        context={'test': True}
    )

    for result in par_results:
        print(f"{result.agent_name}: {result.status} ({result.duration_seconds:.2f}s)")

    # Save history
    orchestrator.save_execution_history('output/agent_execution_history.json')

    print("\n✅ Orchestration complete")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run example
    asyncio.run(main())
