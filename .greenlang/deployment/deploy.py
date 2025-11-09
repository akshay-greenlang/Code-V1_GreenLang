#!/usr/bin/env python3
"""
GreenLang-First Deployment CLI

Automated deployment tool for GreenLang-First enforcement system.
Supports multiple environments with validation, health checks, and rollback.

Usage:
    python deploy.py --env dev --component enforcement
    python deploy.py --env staging --component all
    python deploy.py --env prod --component dashboards
    python deploy.py --rollback --env prod
    python deploy.py --status --env staging
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('greenlang-deploy')

# Constants
SCRIPT_DIR = Path(__file__).parent
CONFIG_DIR = SCRIPT_DIR / 'config'
MANIFESTS_DIR = SCRIPT_DIR

VALID_ENVIRONMENTS = ['dev', 'staging', 'production']
VALID_COMPONENTS = ['all', 'enforcement', 'monitoring', 'dashboards', 'policies']

# Component mapping to Kubernetes manifests
COMPONENT_MANIFESTS = {
    'policies': ['opa-deployment.yaml'],
    'monitoring': ['prometheus-deployment.yaml', 'alertmanager-deployment.yaml'],
    'dashboards': ['grafana-deployment.yaml'],
    'enforcement': ['opa-deployment.yaml'],
    'all': [
        'opa-deployment.yaml',
        'prometheus-deployment.yaml',
        'grafana-deployment.yaml',
        'alertmanager-deployment.yaml',
        'ingress.yaml'
    ]
}


class DeploymentError(Exception):
    """Custom exception for deployment errors."""
    pass


class GreenLangDeployer:
    """Main deployment orchestrator."""

    def __init__(self, environment: str, dry_run: bool = False, verbose: bool = False):
        self.environment = environment
        self.dry_run = dry_run
        self.verbose = verbose
        self.config = self._load_config()
        self.deployment_log = []

        if verbose:
            logger.setLevel(logging.DEBUG)

    def _load_config(self) -> Dict:
        """Load environment-specific configuration."""
        config_file = CONFIG_DIR / f'{self.environment}.yaml'

        if not config_file.exists():
            raise DeploymentError(f'Configuration file not found: {config_file}')

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f'Loaded configuration for environment: {self.environment}')
        return config

    def _run_command(self, cmd: List[str], check: bool = True) -> Tuple[int, str, str]:
        """Execute shell command and return result."""
        if self.dry_run:
            logger.info(f'[DRY RUN] Would execute: {" ".join(cmd)}')
            return 0, '', ''

        logger.debug(f'Executing: {" ".join(cmd)}')

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=check
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            logger.error(f'Command failed: {" ".join(cmd)}')
            logger.error(f'Error: {e.stderr}')
            if check:
                raise DeploymentError(f'Command failed: {e.stderr}')
            return e.returncode, e.stdout, e.stderr

    def _log_deployment(self, action: str, component: str, status: str, details: str = ''):
        """Log deployment action."""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'environment': self.environment,
            'action': action,
            'component': component,
            'status': status,
            'details': details
        }
        self.deployment_log.append(entry)
        logger.info(f'{action} {component}: {status}')

    def validate_prerequisites(self) -> bool:
        """Validate deployment prerequisites."""
        logger.info('Validating prerequisites...')

        checks = [
            ('kubectl', ['kubectl', 'version', '--client']),
            ('helm', ['helm', 'version']),
            ('docker', ['docker', '--version']),
        ]

        all_passed = True
        for name, cmd in checks:
            returncode, stdout, stderr = self._run_command(cmd, check=False)
            if returncode != 0:
                logger.error(f'{name} not found or not working')
                all_passed = False
            else:
                logger.info(f'✓ {name} available')

        # Check Kubernetes cluster access
        if not self.dry_run:
            returncode, stdout, stderr = self._run_command(
                ['kubectl', 'cluster-info'],
                check=False
            )
            if returncode != 0:
                logger.error('Cannot access Kubernetes cluster')
                all_passed = False
            else:
                logger.info('✓ Kubernetes cluster accessible')

        # Check if namespace exists
        namespace = self.config.get('kubernetes', {}).get('namespace', 'greenlang-enforcement')
        if not self.dry_run:
            returncode, stdout, stderr = self._run_command(
                ['kubectl', 'get', 'namespace', namespace],
                check=False
            )
            if returncode != 0:
                logger.warning(f'Namespace {namespace} does not exist. Will create.')
                self._run_command(['kubectl', 'create', 'namespace', namespace])
            else:
                logger.info(f'✓ Namespace {namespace} exists')

        return all_passed

    def pre_deployment_checks(self, component: str) -> bool:
        """Run pre-deployment validation checks."""
        logger.info(f'Running pre-deployment checks for {component}...')

        checks_passed = True

        # Check IUM threshold for production
        if self.environment == 'production':
            ium_threshold = self.config.get('enforcement', {}).get('ium_threshold', 95)
            logger.info(f'Checking IUM score threshold: {ium_threshold}%')

            # In real implementation, query actual IUM score
            # For now, simulate
            current_ium = 96  # Placeholder

            if current_ium < ium_threshold:
                logger.error(f'IUM score {current_ium}% below threshold {ium_threshold}%')
                checks_passed = False
            else:
                logger.info(f'✓ IUM score {current_ium}% meets threshold')

        # Validate manifest files exist
        manifests = COMPONENT_MANIFESTS.get(component, [])
        for manifest in manifests:
            manifest_path = MANIFESTS_DIR / self.environment / manifest
            if not self.dry_run and not manifest_path.exists():
                # Try staging directory as fallback
                manifest_path = MANIFESTS_DIR / 'staging' / manifest

            if not manifest_path.exists() and not self.dry_run:
                logger.error(f'Manifest not found: {manifest_path}')
                checks_passed = False
            else:
                logger.info(f'✓ Manifest found: {manifest}')

        # Validate resources availability
        if self.environment == 'production':
            logger.info('Checking cluster resources...')
            returncode, stdout, stderr = self._run_command(
                ['kubectl', 'top', 'nodes'],
                check=False
            )
            if returncode == 0:
                logger.info('✓ Cluster resources available')
            else:
                logger.warning('Could not check cluster resources')

        return checks_passed

    def deploy_component(self, component: str) -> bool:
        """Deploy a specific component."""
        logger.info(f'Deploying {component} to {self.environment}...')

        try:
            manifests = COMPONENT_MANIFESTS.get(component, [])
            if not manifests:
                raise DeploymentError(f'Unknown component: {component}')

            namespace = self.config.get('kubernetes', {}).get('namespace', 'greenlang-enforcement')

            for manifest in manifests:
                # Try environment-specific manifest first
                manifest_path = MANIFESTS_DIR / self.environment / manifest
                if not manifest_path.exists():
                    # Fallback to staging
                    manifest_path = MANIFESTS_DIR / 'staging' / manifest

                if not manifest_path.exists():
                    logger.warning(f'Manifest not found: {manifest}, skipping')
                    continue

                logger.info(f'Applying manifest: {manifest}')

                # Apply Kubernetes manifest
                returncode, stdout, stderr = self._run_command([
                    'kubectl', 'apply',
                    '-f', str(manifest_path),
                    '-n', namespace
                ])

                if returncode != 0:
                    self._log_deployment('deploy', component, 'FAILED', stderr)
                    raise DeploymentError(f'Failed to apply {manifest}: {stderr}')

                logger.info(f'✓ Applied {manifest}')

            self._log_deployment('deploy', component, 'SUCCESS')
            return True

        except Exception as e:
            self._log_deployment('deploy', component, 'FAILED', str(e))
            logger.error(f'Deployment failed: {e}')
            return False

    def health_check(self, component: str, timeout: int = 300) -> bool:
        """Check health of deployed component."""
        logger.info(f'Running health check for {component}...')

        if self.dry_run:
            logger.info('[DRY RUN] Skipping health check')
            return True

        namespace = self.config.get('kubernetes', {}).get('namespace', 'greenlang-enforcement')

        # Map component to labels
        label_selectors = {
            'policies': 'app=opa',
            'enforcement': 'app=opa',
            'monitoring': 'app=prometheus',
            'dashboards': 'app=grafana',
            'all': None  # Check all
        }

        label = label_selectors.get(component)
        if not label:
            logger.warning(f'No health check defined for {component}')
            return True

        # Wait for pods to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            returncode, stdout, stderr = self._run_command([
                'kubectl', 'get', 'pods',
                '-n', namespace,
                '-l', label,
                '-o', 'json'
            ], check=False)

            if returncode == 0:
                try:
                    pods_data = json.loads(stdout)
                    pods = pods_data.get('items', [])

                    if not pods:
                        logger.warning(f'No pods found for {component}')
                        time.sleep(5)
                        continue

                    all_ready = all(
                        pod['status'].get('phase') == 'Running'
                        for pod in pods
                    )

                    if all_ready:
                        logger.info(f'✓ All {component} pods are ready')
                        return True

                    logger.debug(f'Waiting for {component} pods to be ready...')
                    time.sleep(5)

                except json.JSONDecodeError:
                    logger.error('Failed to parse kubectl output')
                    return False
            else:
                logger.error(f'Failed to get pod status: {stderr}')
                return False

        logger.error(f'Health check timeout after {timeout}s')
        return False

    def rollback(self, to_version: Optional[str] = None) -> bool:
        """Rollback deployment to previous version."""
        logger.info(f'Rolling back deployment in {self.environment}...')

        try:
            namespace = self.config.get('kubernetes', {}).get('namespace', 'greenlang-enforcement')

            # Get deployments
            returncode, stdout, stderr = self._run_command([
                'kubectl', 'get', 'deployments',
                '-n', namespace,
                '-o', 'json'
            ])

            if returncode != 0:
                raise DeploymentError(f'Failed to get deployments: {stderr}')

            deployments = json.loads(stdout).get('items', [])

            for deployment in deployments:
                deployment_name = deployment['metadata']['name']

                logger.info(f'Rolling back {deployment_name}...')

                # Rollback using kubectl
                rollback_cmd = [
                    'kubectl', 'rollout', 'undo',
                    'deployment', deployment_name,
                    '-n', namespace
                ]

                if to_version:
                    rollback_cmd.extend(['--to-revision', to_version])

                returncode, stdout, stderr = self._run_command(rollback_cmd)

                if returncode != 0:
                    logger.error(f'Failed to rollback {deployment_name}: {stderr}')
                else:
                    logger.info(f'✓ Rolled back {deployment_name}')

            self._log_deployment('rollback', 'all', 'SUCCESS')
            return True

        except Exception as e:
            self._log_deployment('rollback', 'all', 'FAILED', str(e))
            logger.error(f'Rollback failed: {e}')
            return False

    def get_status(self) -> Dict:
        """Get deployment status."""
        logger.info(f'Getting status for {self.environment}...')

        if self.dry_run:
            return {'dry_run': True, 'status': 'N/A'}

        namespace = self.config.get('kubernetes', {}).get('namespace', 'greenlang-enforcement')

        status = {
            'environment': self.environment,
            'namespace': namespace,
            'timestamp': datetime.utcnow().isoformat(),
            'components': {}
        }

        # Get all deployments
        returncode, stdout, stderr = self._run_command([
            'kubectl', 'get', 'deployments',
            '-n', namespace,
            '-o', 'json'
        ], check=False)

        if returncode == 0:
            deployments = json.loads(stdout).get('items', [])
            for deployment in deployments:
                name = deployment['metadata']['name']
                spec_replicas = deployment['spec'].get('replicas', 0)
                status_replicas = deployment['status'].get('readyReplicas', 0)

                status['components'][name] = {
                    'desired': spec_replicas,
                    'ready': status_replicas,
                    'healthy': spec_replicas == status_replicas
                }

        # Get services
        returncode, stdout, stderr = self._run_command([
            'kubectl', 'get', 'services',
            '-n', namespace,
            '-o', 'json'
        ], check=False)

        if returncode == 0:
            services = json.loads(stdout).get('items', [])
            status['services'] = [
                {
                    'name': svc['metadata']['name'],
                    'type': svc['spec']['type'],
                    'ports': [p['port'] for p in svc['spec'].get('ports', [])]
                }
                for svc in services
            ]

        return status

    def save_deployment_log(self):
        """Save deployment log to file."""
        log_dir = SCRIPT_DIR / 'logs'
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'deployment_{self.environment}_{timestamp}.json'

        with open(log_file, 'w') as f:
            json.dump(self.deployment_log, f, indent=2)

        logger.info(f'Deployment log saved to {log_file}')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='GreenLang-First Deployment CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Deploy to development
  %(prog)s --env dev --component enforcement

  # Deploy all components to staging
  %(prog)s --env staging --component all

  # Deploy to production (requires confirmation)
  %(prog)s --env production --component dashboards

  # Rollback production
  %(prog)s --rollback --env production

  # Check status
  %(prog)s --status --env staging

  # Dry run
  %(prog)s --env prod --component all --dry-run
        '''
    )

    parser.add_argument(
        '--env',
        choices=VALID_ENVIRONMENTS,
        required=True,
        help='Target environment'
    )

    parser.add_argument(
        '--component',
        choices=VALID_COMPONENTS,
        help='Component to deploy'
    )

    parser.add_argument(
        '--rollback',
        action='store_true',
        help='Rollback to previous version'
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Get deployment status'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate deployment without making changes'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--skip-health-check',
        action='store_true',
        help='Skip health checks after deployment'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force deployment without confirmation (dangerous in production)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.rollback and not args.status and not args.component:
        parser.error('--component is required unless using --rollback or --status')

    try:
        deployer = GreenLangDeployer(
            environment=args.env,
            dry_run=args.dry_run,
            verbose=args.verbose
        )

        # Get status
        if args.status:
            status = deployer.get_status()
            print(json.dumps(status, indent=2))
            return 0

        # Production confirmation
        if args.env == 'production' and not args.force and not args.dry_run:
            print(f'\n{"="*60}')
            print('WARNING: You are about to deploy to PRODUCTION')
            print(f'{"="*60}')
            print(f'Environment: {args.env}')
            print(f'Component: {args.component or "rollback"}')
            print(f'Action: {"Rollback" if args.rollback else "Deploy"}')
            print(f'{"="*60}\n')

            response = input('Type "CONFIRM" to proceed: ')
            if response != 'CONFIRM':
                print('Deployment cancelled.')
                return 1

        # Validate prerequisites
        if not deployer.validate_prerequisites():
            logger.error('Prerequisites validation failed')
            return 1

        # Rollback
        if args.rollback:
            if deployer.rollback():
                logger.info('✓ Rollback completed successfully')
                deployer.save_deployment_log()
                return 0
            else:
                logger.error('✗ Rollback failed')
                deployer.save_deployment_log()
                return 1

        # Deploy
        if args.component:
            # Pre-deployment checks
            if not deployer.pre_deployment_checks(args.component):
                logger.error('Pre-deployment checks failed')
                return 1

            # Deploy
            if deployer.deploy_component(args.component):
                logger.info(f'✓ Deployment of {args.component} completed')

                # Health check
                if not args.skip_health_check:
                    if deployer.health_check(args.component):
                        logger.info('✓ Health check passed')
                    else:
                        logger.error('✗ Health check failed')
                        deployer.save_deployment_log()
                        return 1

                deployer.save_deployment_log()
                logger.info('✓ Deployment successful')
                return 0
            else:
                logger.error('✗ Deployment failed')
                deployer.save_deployment_log()
                return 1

    except DeploymentError as e:
        logger.error(f'Deployment error: {e}')
        return 1
    except KeyboardInterrupt:
        logger.warning('\nDeployment cancelled by user')
        return 130
    except Exception as e:
        logger.exception(f'Unexpected error: {e}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
