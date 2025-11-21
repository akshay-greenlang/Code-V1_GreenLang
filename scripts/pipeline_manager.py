#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GreenLang Data Pipeline Manager

Command-line interface for managing the emission factors data pipeline.

Usage:
    python pipeline_manager.py import --validate --backup
    python pipeline_manager.py validate
    python pipeline_manager.py dashboard
    python pipeline_manager.py coverage
    python pipeline_manager.py export-report --output report.json
    python pipeline_manager.py schedule --daily --time 02:00
    python pipeline_manager.py workflow submit --factor-id fuel_diesel --change-type updated

Commands:
    import              Run import pipeline
    validate            Validate YAML files
    dashboard           Show quality dashboard
    coverage            Show coverage analysis
    sources             Show source diversity
    freshness           Show data freshness
    export-report       Export monitoring report
    schedule            Configure scheduled imports
    workflow            Manage update workflow
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from greenlang.determinism import FinancialDecimal

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from greenlang.data.pipeline.dashboard import PipelineCLI, DataQualityDashboard
from greenlang.data.pipeline.pipeline import AutomatedImportPipeline, ScheduledImporter
from greenlang.data.pipeline.workflow import UpdateWorkflow, ApprovalManager
from greenlang.data.pipeline.models import ChangeType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='GreenLang Data Pipeline Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Common arguments
    parser.add_argument(
        '--db-path',
        default='C:/Users/aksha/Code-V1_GreenLang/greenlang/data/emission_factors.db',
        help='Path to database'
    )
    parser.add_argument(
        '--data-dir',
        default='C:/Users/aksha/Code-V1_GreenLang/data',
        help='Path to data directory'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Import command
    import_parser = subparsers.add_parser('import', help='Run import pipeline')
    import_parser.add_argument(
        '--validate',
        action='store_true',
        default=True,
        help='Run pre-import validation (default: True)'
    )
    import_parser.add_argument(
        '--no-validate',
        action='store_false',
        dest='validate',
        help='Skip pre-import validation'
    )
    import_parser.add_argument(
        '--backup',
        action='store_true',
        default=True,
        help='Create backup before import (default: True)'
    )
    import_parser.add_argument(
        '--no-backup',
        action='store_false',
        dest='backup',
        help='Skip backup creation'
    )

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate YAML files')

    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Show quality dashboard')

    # Coverage command
    coverage_parser = subparsers.add_parser('coverage', help='Show coverage analysis')

    # Sources command
    sources_parser = subparsers.add_parser('sources', help='Show source diversity')

    # Freshness command
    freshness_parser = subparsers.add_parser('freshness', help='Show data freshness')

    # Export command
    export_parser = subparsers.add_parser('export-report', help='Export monitoring report')
    export_parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output file path'
    )
    export_parser.add_argument(
        '--format', '-f',
        choices=['json', 'html'],
        default='json',
        help='Export format (default: json)'
    )

    # Schedule command
    schedule_parser = subparsers.add_parser('schedule', help='Configure scheduled imports')
    schedule_parser.add_argument(
        '--daily',
        action='store_true',
        help='Schedule daily import'
    )
    schedule_parser.add_argument(
        '--weekly',
        action='store_true',
        help='Schedule weekly import'
    )
    schedule_parser.add_argument(
        '--time',
        default='02:00',
        help='Time to run (HH:MM format, default: 02:00)'
    )
    schedule_parser.add_argument(
        '--day',
        default='sunday',
        help='Day for weekly schedule (default: sunday)'
    )

    # Workflow command
    workflow_parser = subparsers.add_parser('workflow', help='Manage update workflow')
    workflow_subparsers = workflow_parser.add_subparsers(dest='workflow_command')

    # Workflow submit
    submit_parser = workflow_subparsers.add_parser('submit', help='Submit change request')
    submit_parser.add_argument('--factor-id', required=True, help='Factor ID to modify')
    submit_parser.add_argument(
        '--change-type',
        required=True,
        choices=['added', 'updated', 'deprecated', 'deleted'],
        help='Type of change'
    )
    submit_parser.add_argument('--reason', required=True, help='Reason for change')
    submit_parser.add_argument('--value', help='New emission factor value')
    submit_parser.add_argument('--source', help='New source organization')
    submit_parser.add_argument('--uri', help='New source URI')

    # Workflow approve
    approve_parser = workflow_subparsers.add_parser('approve', help='Approve change request')
    approve_parser.add_argument('--request-id', required=True, help='Request ID')
    approve_parser.add_argument('--reviewer', required=True, help='Reviewer name')
    approve_parser.add_argument('--notes', help='Review notes')

    # Workflow reject
    reject_parser = workflow_subparsers.add_parser('reject', help='Reject change request')
    reject_parser.add_argument('--request-id', required=True, help='Request ID')
    reject_parser.add_argument('--reviewer', required=True, help='Reviewer name')
    reject_parser.add_argument('--notes', required=True, help='Rejection reason')

    # Workflow pending
    pending_parser = workflow_subparsers.add_parser('pending', help='List pending reviews')
    pending_parser.add_argument('--reviewer', required=True, help='Reviewer name')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize CLI
    cli = PipelineCLI(args.db_path, args.data_dir)

    # Execute command
    try:
        if args.command == 'import':
            cli.run_import(validate=args.validate, rollback_on_failure=args.backup)

        elif args.command == 'validate':
            cli.validate_data()

        elif args.command == 'dashboard':
            cli.show_dashboard()

        elif args.command == 'coverage':
            cli.show_coverage()

        elif args.command == 'sources':
            cli.show_sources()

        elif args.command == 'freshness':
            cli.show_freshness()

        elif args.command == 'export-report':
            cli.export_report(args.output, args.format)

        elif args.command == 'schedule':
            setup_schedule(args, cli)

        elif args.command == 'workflow':
            handle_workflow(args, cli)

        else:
            parser.print_help()

    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=args.verbose)
        sys.exit(1)


def setup_schedule(args, cli):
    """Setup scheduled imports."""
    yaml_files = [
        str(Path(args.data_dir) / 'emission_factors_registry.yaml'),
        str(Path(args.data_dir) / 'emission_factors_expansion_phase1.yaml'),
        str(Path(args.data_dir) / 'emission_factors_expansion_phase2.yaml')
    ]

    pipeline = AutomatedImportPipeline(args.db_path)
    scheduler = ScheduledImporter(pipeline, yaml_files, args.db_path)

    if args.daily:
        scheduler.schedule_daily(args.time)
        logger.info(f"Scheduled daily import at {args.time}")
    elif args.weekly:
        scheduler.schedule_weekly(args.day, args.time)
        logger.info(f"Scheduled weekly import on {args.day} at {args.time}")
    else:
        logger.error("Specify --daily or --weekly")
        return

    logger.info("Starting scheduler... (Press Ctrl+C to stop)")
    scheduler.run_scheduler()


def handle_workflow(args, cli):
    """Handle workflow commands."""
    approval_manager = ApprovalManager(f"{args.db_path}.versions.db")
    workflow = UpdateWorkflow(args.db_path, approval_manager)

    if args.workflow_command == 'submit':
        # Build proposed changes
        proposed_changes = {}
        if args.value:
            proposed_changes['emission_factor_value'] = FinancialDecimal.from_string(args.value)
        if args.source:
            proposed_changes['source_org'] = args.source
        if args.uri:
            proposed_changes['source_uri'] = args.uri

        if not proposed_changes:
            logger.error("No changes specified (use --value, --source, or --uri)")
            return

        # Submit request
        request = workflow.submit_change_request(
            factor_id=args.factor_id,
            change_type=ChangeType(args.change_type),
            proposed_changes=proposed_changes,
            change_reason=args.reason,
            requested_by="cli_user"
        )

        logger.info(f"Change request submitted: {request.request_id}")
        logger.info(f"Review status: {request.review_status}")

    elif args.workflow_command == 'approve':
        approval_manager.approve_request(
            args.request_id,
            args.reviewer,
            args.notes
        )
        logger.info(f"Change request approved: {args.request_id}")

    elif args.workflow_command == 'reject':
        approval_manager.reject_request(
            args.request_id,
            args.reviewer,
            args.notes
        )
        logger.info(f"Change request rejected: {args.request_id}")

    elif args.workflow_command == 'pending':
        pending = approval_manager.get_pending_reviews(args.reviewer)

        if not pending:
            logger.info(f"No pending reviews for {args.reviewer}")
            return

        logger.info(f"\nPending reviews for {args.reviewer}:")
        for req in pending:
            logger.info(f"  {req['request_id']} - {req['factor_id']} ({req['change_type']})")
            logger.info(f"    Reason: {req['change_reason']}")
            logger.info(f"    Submitted: {req['timestamp']}")

    else:
        logger.error(f"Unknown workflow command: {args.workflow_command}")


if __name__ == '__main__':
    main()
