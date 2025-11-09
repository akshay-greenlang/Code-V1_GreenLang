#!/usr/bin/env python3
"""
GreenLang CLI Tool

Unified command-line interface for all GreenLang migration utilities.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


# Add scripts and tools directories to path
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts')
TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tools')
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, TOOLS_DIR)


class GreenLangCLI:
    """Main CLI class."""

    def __init__(self):
        self.scripts_dir = SCRIPTS_DIR
        self.tools_dir = TOOLS_DIR

    def run_script(self, script_name: str, args: list):
        """Run a Python script with arguments."""
        script_path = os.path.join(self.scripts_dir, script_name)

        if not os.path.exists(script_path):
            print(f"Error: Script not found: {script_path}")
            sys.exit(1)

        # Run script
        cmd = [sys.executable, script_path] + args
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    def run_tool(self, tool_name: str, args: list):
        """Run a tool with arguments."""
        tool_path = os.path.join(self.tools_dir, tool_name)

        if not os.path.exists(tool_path):
            print(f"Error: Tool not found: {tool_path}")
            sys.exit(1)

        # Run tool
        cmd = [sys.executable, tool_path] + args
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    def migrate(self, args):
        """Run migration tool."""
        script_args = [args.app or '.']

        if args.dry_run:
            script_args.append('--dry-run')

        if args.auto_fix:
            script_args.append('--auto-fix')

        if args.format:
            script_args.extend(['--format', args.format])

        if args.output:
            script_args.extend(['--output', args.output])

        if args.category:
            script_args.extend(['--category', args.category])

        self.run_script('migrate_to_infrastructure.py', script_args)

    def rewrite_imports(self, args):
        """Run import rewriter."""
        script_args = [args.path or '.']

        if args.dry_run:
            script_args.append('--dry-run')

        if args.show_diff:
            script_args.append('--show-diff')

        if args.git_diff:
            script_args.append('--git-diff')

        self.run_script('rewrite_imports.py', script_args)

    def convert_agents(self, args):
        """Run agent converter."""
        script_args = [args.path or '.']

        if args.dry_run:
            script_args.append('--dry-run')

        if args.show_diff:
            script_args.append('--show-diff')

        self.run_script('convert_to_base_agent.py', script_args)

    def update_deps(self, args):
        """Run dependency updater."""
        script_args = []

        if args.file:
            script_args.extend(['--file', args.file])

        if args.dry_run:
            script_args.append('--dry-run')

        if args.remove_redundant:
            script_args.append('--remove-redundant')

        if args.scan:
            script_args.extend(['--scan', args.scan])

        if args.install:
            script_args.append('--install')

        if args.show_diff:
            script_args.append('--show-diff')

        self.run_script('update_dependencies.py', script_args)

    def generate_code(self, args):
        """Run code generator."""
        if not args.type or not args.name:
            print("Error: --type and --name are required")
            sys.exit(1)

        script_args = ['--type', args.type, '--name', args.name]

        if args.output:
            script_args.extend(['--output', args.output])

        if args.agents:
            script_args.extend(['--agents', args.agents])

        if args.provider:
            script_args.extend(['--provider', args.provider])

        if args.model:
            script_args.extend(['--model', args.model])

        if args.batch:
            script_args.append('--batch')

        if args.description:
            script_args.extend(['--description', args.description])

        self.run_script('generate_infrastructure_code.py', script_args)

    def generate_report(self, args):
        """Run usage report generator."""
        script_args = [args.directory or '.']

        if args.format:
            script_args.extend(['--format', args.format])

        if args.output:
            script_args.extend(['--output', args.output])

        self.run_script('generate_usage_report.py', script_args)

    def create_adr(self, args):
        """Run ADR generator."""
        script_args = []

        if args.list:
            script_args.append('--list')

        if args.validate:
            script_args.extend(['--validate', args.validate])

        if args.adr_dir:
            script_args.extend(['--adr-dir', args.adr_dir])

        self.run_script('create_adr.py', script_args)

    def serve_dashboard(self, args):
        """Run dashboard server."""
        script_args = []

        if args.directory:
            script_args.extend(['--directory', args.directory])

        if args.port:
            script_args.extend(['--port', str(args.port)])

        self.run_script('serve_dashboard.py', script_args)

    def check_file(self, args):
        """Check a file for migration opportunities."""
        if not args.file:
            print("Error: --file is required")
            sys.exit(1)

        print(f"Checking {args.file} for migration opportunities...\n")

        # Run migration tool on single file
        self.run_script('migrate_to_infrastructure.py', [
            args.file,
            '--dry-run',
            '--format', 'text'
        ])

    def status(self, args):
        """Show migration status."""
        directory = args.directory or '.'

        print(f"\nGreenLang Migration Status")
        print("=" * 80)
        print(f"Directory: {os.path.abspath(directory)}\n")

        # Generate quick report
        self.run_script('generate_usage_report.py', [
            directory,
            '--format', 'text'
        ])

    # AI-Powered Tools
    def search_infrastructure(self, args):
        """Search infrastructure components."""
        tool_args = []

        if args.query:
            tool_args.append(args.query)

        if args.category:
            tool_args.extend(['--category', args.category])

        if args.tag:
            tool_args.extend(['--tag', args.tag])

        if args.top_k:
            tool_args.extend(['--top-k', str(args.top_k)])

        if args.format:
            tool_args.extend(['--format', args.format])

        self.run_tool('infra_search.py', tool_args)

    def recommend_code(self, args):
        """Run code recommender."""
        tool_args = [args.path]

        if args.format:
            tool_args.extend(['--format', args.format])

        if args.output:
            tool_args.extend(['--output', args.output])

        if args.auto_fixable_only:
            tool_args.append('--auto-fixable-only')

        if args.category:
            tool_args.extend(['--category', args.category])

        if args.severity:
            tool_args.extend(['--severity', args.severity])

        self.run_tool('code_recommender.py', tool_args)

    def smart_generate(self, args):
        """Run smart code generator."""
        tool_args = []

        if args.description:
            tool_args.append(args.description)

        if args.output:
            tool_args.extend(['--output', args.output])

        if args.interactive:
            tool_args.append('--interactive')

        if args.preview:
            tool_args.append('--preview')

        self.run_tool('smart_generate.py', tool_args)

    def health_check(self, args):
        """Run infrastructure health check."""
        tool_args = []

        if args.directory:
            tool_args.extend(['--directory', args.directory])

        if args.format:
            tool_args.extend(['--format', args.format])

        if args.output:
            tool_args.extend(['--output', args.output])

        self.run_tool('health_check.py', tool_args)

    def explore_infrastructure(self, args):
        """Launch infrastructure explorer."""
        tool_args = []

        if args.port:
            tool_args.extend(['--port', str(args.port)])

        self.run_tool('explorer.py', tool_args)

    def generate_docs(self, args):
        """Generate documentation."""
        tool_args = []

        if args.source:
            tool_args.extend(['--source', args.source])

        if args.output:
            tool_args.extend(['--output', args.output])

        if args.format:
            tool_args.extend(['--format', args.format])

        self.run_tool('auto_docs.py', tool_args)

    def dependency_graph(self, args):
        """Generate dependency graph."""
        tool_args = []

        if args.directory:
            tool_args.extend(['--directory', args.directory])

        if args.apps:
            tool_args.extend(['--apps'] + args.apps)

        if args.format:
            tool_args.extend(['--format', args.format])

        if args.output:
            tool_args.extend(['--output', args.output])

        if args.interactive:
            tool_args.append('--interactive')

        self.run_tool('dep_graph.py', tool_args)

    def migration_wizard(self, args):
        """Launch migration wizard."""
        self.run_tool('migration_assistant.py', [])

    def profile_code(self, args):
        """Profile code performance."""
        tool_args = [args.file]

        if args.detailed:
            tool_args.append('--detailed')

        if args.format:
            tool_args.extend(['--format', args.format])

        if args.output:
            tool_args.extend(['--output', args.output])

        self.run_tool('profiler.py', tool_args)

    def ai_chat(self, args):
        """Start AI assistant."""
        tool_args = []

        if args.question:
            tool_args.append(args.question)

        if args.help_topic:
            tool_args.extend(['--help-topic', args.help_topic])

        self.run_tool('ai_assistant.py', tool_args)

    # Generator Tools
    def create_app(self, args):
        """Create new application."""
        tool_args = []

        if args.name:
            tool_args.append(args.name)

        if args.template:
            tool_args.extend(['--template', args.template])

        if args.llm:
            tool_args.extend(['--llm', args.llm])

        if args.cache:
            tool_args.extend(['--cache', args.cache])

        if args.database:
            tool_args.extend(['--database', args.database])

        if args.no_tests:
            tool_args.append('--no-tests')

        if args.no_cicd:
            tool_args.append('--no-cicd')

        if args.no_monitoring:
            tool_args.append('--no-monitoring')

        if args.output:
            tool_args.extend(['--output', args.output])

        self.run_tool('create_app.py', tool_args)

    def add_component(self, args):
        """Add component to application."""
        tool_args = [args.component]

        if args.component == 'agent':
            tool_args.append(args.name)
            if args.template:
                tool_args.extend(['--template', args.template])
        elif args.component == 'llm':
            if args.provider:
                tool_args.extend(['--provider', args.provider])
            if args.caching:
                tool_args.append('--caching')
        elif args.component == 'cache':
            if args.type:
                tool_args.extend(['--type', args.type])
        elif args.component == 'database':
            if args.type:
                tool_args.extend(['--type', args.type])
        elif args.component == 'monitoring':
            if args.dashboard:
                tool_args.extend(['--dashboard', args.dashboard])

        if args.app_dir:
            tool_args.extend(['--app-dir', args.app_dir])

        self.run_tool('add_component.py', tool_args)

    def generate_config_tool(self, args):
        """Generate configuration files."""
        tool_args = []

        if args.app_name:
            tool_args.extend(['--app-name', args.app_name])

        if args.environment:
            tool_args.extend(['--environment', args.environment])

        if args.all_environments:
            tool_args.append('--all-environments')

        if args.validate:
            tool_args.extend(['--validate', args.validate])

        if args.output:
            tool_args.extend(['--output', args.output])

        if args.interactive:
            tool_args.append('--interactive')

        if args.app_dir:
            tool_args.extend(['--app-dir', args.app_dir])

        self.run_tool('generate_config.py', tool_args)

    def generate_tests_tool(self, args):
        """Generate tests for agent."""
        tool_args = [args.agent_file]

        if args.output:
            tool_args.extend(['--output', args.output])

        if args.conftest:
            tool_args.append('--conftest')

        self.run_tool('generate_tests.py', tool_args)

    def generate_cicd_tool(self, args):
        """Generate CI/CD configuration."""
        tool_args = []

        if args.platform:
            tool_args.extend(['--platform', args.platform])

        if args.all:
            tool_args.append('--all')

        if args.app_name:
            tool_args.extend(['--app-name', args.app_name])

        if args.with_deploy:
            tool_args.append('--with-deploy')

        if args.output_dir:
            tool_args.extend(['--output-dir', args.output_dir])

        self.run_tool('generate_cicd.py', tool_args)

    def generate_deployment_tool(self, args):
        """Generate deployment configuration."""
        tool_args = []

        if args.platform:
            tool_args.extend(['--platform', args.platform])

        if args.all:
            tool_args.append('--all')

        if args.app_name:
            tool_args.extend(['--app-name', args.app_name])

        if args.output_dir:
            tool_args.extend(['--output-dir', args.output_dir])

        self.run_tool('generate_deployment.py', tool_args)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='GreenLang CLI - Migration and infrastructure utilities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Application Generators
  greenlang create-app my-app --template llm-analysis --llm openai
  greenlang add agent DataProcessor --template calculator
  greenlang generate-config --interactive
  greenlang generate-tests app/agents/my_agent.py
  greenlang generate-cicd --platform github --with-deploy
  greenlang generate-deployment --platform kubernetes

  # Migration Tools
  greenlang migrate --app GL-CBAM-APP --dry-run
  greenlang migrate-wizard

  # Code Generation
  greenlang generate --type agent --name MyAgent --batch
  greenlang smart-generate "Create an agent that validates CSV files"

  # AI-Powered Discovery
  greenlang search "how to cache API responses"
  greenlang recommend my_agent.py
  greenlang chat "How do I add LLM to my agent?"

  # Health & Analysis
  greenlang health-check --directory GL-CBAM-APP --format html
  greenlang dep-graph --interactive --output graph.html
  greenlang profile my_agent.py

  # Documentation & Exploration
  greenlang explore
  greenlang docs --source shared/infrastructure --output docs/

  # Reports & Status
  greenlang report --format html --output report.html
  greenlang status --directory GL-CBAM-APP

For more information, visit: https://github.com/greenlang/greenlang
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Scan and migrate code')
    migrate_parser.add_argument('--app', help='Application directory to scan')
    migrate_parser.add_argument('--dry-run', action='store_true', help='Preview changes without applying')
    migrate_parser.add_argument('--auto-fix', action='store_true', help='Automatically apply changes')
    migrate_parser.add_argument('--format', choices=['text', 'json', 'html'], help='Report format')
    migrate_parser.add_argument('--output', help='Output file for report')
    migrate_parser.add_argument('--category', help='Filter by category')

    # Rewrite imports command
    imports_parser = subparsers.add_parser('imports', help='Rewrite imports')
    imports_parser.add_argument('--path', help='File or directory to rewrite')
    imports_parser.add_argument('--dry-run', action='store_true', help='Preview changes')
    imports_parser.add_argument('--show-diff', action='store_true', help='Show diff')
    imports_parser.add_argument('--git-diff', action='store_true', help='Generate git diff')

    # Convert agents command
    agents_parser = subparsers.add_parser('agents', help='Convert agent classes')
    agents_parser.add_argument('--path', help='File or directory to convert')
    agents_parser.add_argument('--dry-run', action='store_true', help='Preview changes')
    agents_parser.add_argument('--show-diff', action='store_true', help='Show diff')

    # Update dependencies command
    deps_parser = subparsers.add_parser('deps', help='Update dependencies')
    deps_parser.add_argument('--file', default='requirements.txt', help='Requirements file')
    deps_parser.add_argument('--dry-run', action='store_true', help='Preview changes')
    deps_parser.add_argument('--remove-redundant', action='store_true', help='Remove redundant packages')
    deps_parser.add_argument('--scan', help='Scan directory for used packages')
    deps_parser.add_argument('--install', action='store_true', help='Install packages after update')
    deps_parser.add_argument('--show-diff', action='store_true', help='Show diff')

    # Generate code command
    generate_parser = subparsers.add_parser('generate', help='Generate infrastructure code')
    generate_parser.add_argument('--type', required=True, choices=['agent', 'pipeline', 'llm-session', 'cache', 'validation', 'config'])
    generate_parser.add_argument('--name', required=True, help='Name for generated code')
    generate_parser.add_argument('--output', help='Output file path')
    generate_parser.add_argument('--agents', help='Agent names (for pipeline)')
    generate_parser.add_argument('--provider', help='LLM provider (for llm-session)')
    generate_parser.add_argument('--model', help='LLM model (for llm-session)')
    generate_parser.add_argument('--batch', action='store_true', help='Include batch processing (for agent)')
    generate_parser.add_argument('--description', help='Description')

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate usage report')
    report_parser.add_argument('--directory', help='Directory to analyze')
    report_parser.add_argument('--format', choices=['text', 'json', 'html'], default='html', help='Report format')
    report_parser.add_argument('--output', help='Output file')

    # ADR command
    adr_parser = subparsers.add_parser('adr', help='Manage ADRs')
    adr_parser.add_argument('--list', action='store_true', help='List all ADRs')
    adr_parser.add_argument('--validate', help='Validate an ADR file')
    adr_parser.add_argument('--adr-dir', help='ADR directory')

    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Start migration dashboard')
    dashboard_parser.add_argument('--directory', help='Directory to monitor')
    dashboard_parser.add_argument('--port', type=int, default=8080, help='Server port')

    # Check command
    check_parser = subparsers.add_parser('check', help='Check file for migration opportunities')
    check_parser.add_argument('--file', required=True, help='File to check')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show migration status')
    status_parser.add_argument('--directory', help='Directory to check')

    # AI-Powered Tools
    # Search command
    search_parser = subparsers.add_parser('search', help='Search infrastructure components')
    search_parser.add_argument('query', nargs='?', help='Search query')
    search_parser.add_argument('--category', help='Filter by category')
    search_parser.add_argument('--tag', help='Filter by tag')
    search_parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    search_parser.add_argument('--format', choices=['text', 'json'], default='text')

    # Recommend command
    recommend_parser = subparsers.add_parser('recommend', help='Get code recommendations')
    recommend_parser.add_argument('path', help='File or directory to analyze')
    recommend_parser.add_argument('--format', choices=['text', 'json', 'html'], default='text')
    recommend_parser.add_argument('--output', help='Output file')
    recommend_parser.add_argument('--auto-fixable-only', action='store_true')
    recommend_parser.add_argument('--category', help='Filter by category')
    recommend_parser.add_argument('--severity', choices=['error', 'warning', 'info'])

    # Smart-generate command
    smart_gen_parser = subparsers.add_parser('smart-generate', help='Generate code from description')
    smart_gen_parser.add_argument('description', nargs='?', help='Natural language description')
    smart_gen_parser.add_argument('--output', help='Output directory')
    smart_gen_parser.add_argument('--interactive', action='store_true')
    smart_gen_parser.add_argument('--preview', action='store_true')

    # Health-check command
    health_parser = subparsers.add_parser('health-check', help='Run infrastructure health check')
    health_parser.add_argument('--directory', default='.', help='Directory to check')
    health_parser.add_argument('--format', choices=['text', 'json', 'html'], default='text')
    health_parser.add_argument('--output', help='Output file')

    # Explore command
    explore_parser = subparsers.add_parser('explore', help='Launch infrastructure explorer')
    explore_parser.add_argument('--port', type=int, default=8501, help='Port to run on')

    # Docs command
    docs_parser = subparsers.add_parser('docs', help='Generate documentation')
    docs_parser.add_argument('--source', default='shared/infrastructure', help='Source directory')
    docs_parser.add_argument('--output', default='docs', help='Output directory')
    docs_parser.add_argument('--format', choices=['markdown', 'html', 'both'], default='both')

    # Dep-graph command
    dep_graph_parser = subparsers.add_parser('dep-graph', help='Generate dependency graph')
    dep_graph_parser.add_argument('--directory', default='.', help='Root directory')
    dep_graph_parser.add_argument('--apps', nargs='+', help='Specific apps to analyze')
    dep_graph_parser.add_argument('--format', choices=['text', 'dot', 'html', 'json'], default='text')
    dep_graph_parser.add_argument('--output', help='Output file')
    dep_graph_parser.add_argument('--interactive', action='store_true')

    # Migrate-wizard command
    wizard_parser = subparsers.add_parser('migrate-wizard', help='Launch migration wizard')

    # Profile command
    profile_parser = subparsers.add_parser('profile', help='Profile code performance')
    profile_parser.add_argument('file', help='Python file to profile')
    profile_parser.add_argument('--detailed', action='store_true')
    profile_parser.add_argument('--format', choices=['text', 'json'], default='text')
    profile_parser.add_argument('--output', help='Output file')

    # Chat command
    chat_parser = subparsers.add_parser('chat', help='AI pair programming assistant')
    chat_parser.add_argument('question', nargs='?', help='Ask a question')
    chat_parser.add_argument('--help-topic', help='Get help on a topic')

    # Generator Commands
    # Create-app command
    create_app_parser = subparsers.add_parser('create-app', help='Create new GreenLang application')
    create_app_parser.add_argument('name', nargs='?', help='Application name')
    create_app_parser.add_argument('--template', choices=['data-intake', 'calculation', 'llm-analysis', 'pipeline', 'reporting', 'api-service'],
                                    help='Application template')
    create_app_parser.add_argument('--llm', choices=['openai', 'anthropic', 'all'], help='LLM provider')
    create_app_parser.add_argument('--cache', choices=['memory', 'redis', 'both'], help='Cache type')
    create_app_parser.add_argument('--database', choices=['postgresql', 'mongodb', 'both'], help='Database type')
    create_app_parser.add_argument('--no-tests', action='store_true', help='Skip test generation')
    create_app_parser.add_argument('--no-cicd', action='store_true', help='Skip CI/CD generation')
    create_app_parser.add_argument('--no-monitoring', action='store_true', help='Skip monitoring')
    create_app_parser.add_argument('--output', help='Output directory')

    # Add command
    add_parser = subparsers.add_parser('add', help='Add component to existing application')
    add_subparsers = add_parser.add_subparsers(dest='component', help='Component type')

    # Add agent
    add_agent_parser = add_subparsers.add_parser('agent', help='Add an agent')
    add_agent_parser.add_argument('name', help='Agent name')
    add_agent_parser.add_argument('--template', choices=['basic', 'calculator', 'llm-analyzer', 'validator'],
                                   default='basic', help='Agent template')
    add_agent_parser.add_argument('--app-dir', default='.', help='Application directory')

    # Add LLM
    add_llm_parser = add_subparsers.add_parser('llm', help='Add LLM integration')
    add_llm_parser.add_argument('--provider', choices=['openai', 'anthropic'], default='openai', help='LLM provider')
    add_llm_parser.add_argument('--caching', action='store_true', help='Enable caching')
    add_llm_parser.add_argument('--app-dir', default='.', help='Application directory')

    # Add cache
    add_cache_parser = add_subparsers.add_parser('cache', help='Add caching')
    add_cache_parser.add_argument('--type', choices=['memory', 'redis'], default='redis', help='Cache type')
    add_cache_parser.add_argument('--app-dir', default='.', help='Application directory')

    # Add database
    add_db_parser = add_subparsers.add_parser('database', help='Add database')
    add_db_parser.add_argument('--type', choices=['postgresql', 'mongodb'], default='postgresql', help='Database type')
    add_db_parser.add_argument('--app-dir', default='.', help='Application directory')

    # Add monitoring
    add_monitoring_parser = add_subparsers.add_parser('monitoring', help='Add monitoring')
    add_monitoring_parser.add_argument('--dashboard', choices=['grafana', 'prometheus'], default='grafana',
                                        help='Dashboard type')
    add_monitoring_parser.add_argument('--app-dir', default='.', help='Application directory')

    # Generate-config command
    gen_config_parser = subparsers.add_parser('generate-config', help='Generate configuration files')
    gen_config_parser.add_argument('--app-name', help='Application name')
    gen_config_parser.add_argument('--environment', choices=['development', 'staging', 'production'],
                                    help='Generate for specific environment')
    gen_config_parser.add_argument('--all-environments', action='store_true', help='Generate all environment configs')
    gen_config_parser.add_argument('--validate', help='Validate configuration file')
    gen_config_parser.add_argument('--output', help='Output file path')
    gen_config_parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    gen_config_parser.add_argument('--app-dir', default='.', help='Application directory')

    # Generate-tests command
    gen_tests_parser = subparsers.add_parser('generate-tests', help='Generate tests for agent')
    gen_tests_parser.add_argument('agent_file', help='Agent file to generate tests for')
    gen_tests_parser.add_argument('--output', help='Output test file')
    gen_tests_parser.add_argument('--conftest', action='store_true', help='Also generate conftest.py')

    # Generate-cicd command
    gen_cicd_parser = subparsers.add_parser('generate-cicd', help='Generate CI/CD configuration')
    gen_cicd_parser.add_argument('--platform', choices=['github', 'gitlab', 'jenkins'],
                                  help='CI/CD platform')
    gen_cicd_parser.add_argument('--all', action='store_true', help='Generate for all platforms')
    gen_cicd_parser.add_argument('--app-name', default='greenlang-app', help='Application name')
    gen_cicd_parser.add_argument('--with-deploy', action='store_true', help='Include deployment stage')
    gen_cicd_parser.add_argument('--output-dir', default='.', help='Output directory')

    # Generate-deployment command
    gen_deploy_parser = subparsers.add_parser('generate-deployment', help='Generate deployment configuration')
    gen_deploy_parser.add_argument('--platform', choices=['kubernetes', 'terraform', 'docker', 'helm'],
                                     help='Deployment platform')
    gen_deploy_parser.add_argument('--all', action='store_true', help='Generate for all platforms')
    gen_deploy_parser.add_argument('--app-name', default='greenlang-app', help='Application name')
    gen_deploy_parser.add_argument('--output-dir', default='.', help='Output directory')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize CLI
    cli = GreenLangCLI()

    # Route to appropriate command
    if args.command == 'migrate':
        cli.migrate(args)
    elif args.command == 'imports':
        cli.rewrite_imports(args)
    elif args.command == 'agents':
        cli.convert_agents(args)
    elif args.command == 'deps':
        cli.update_deps(args)
    elif args.command == 'generate':
        cli.generate_code(args)
    elif args.command == 'report':
        cli.generate_report(args)
    elif args.command == 'adr':
        cli.create_adr(args)
    elif args.command == 'dashboard':
        cli.serve_dashboard(args)
    elif args.command == 'check':
        cli.check_file(args)
    elif args.command == 'status':
        cli.status(args)
    # AI-Powered Tools
    elif args.command == 'search':
        cli.search_infrastructure(args)
    elif args.command == 'recommend':
        cli.recommend_code(args)
    elif args.command == 'smart-generate':
        cli.smart_generate(args)
    elif args.command == 'health-check':
        cli.health_check(args)
    elif args.command == 'explore':
        cli.explore_infrastructure(args)
    elif args.command == 'docs':
        cli.generate_docs(args)
    elif args.command == 'dep-graph':
        cli.dependency_graph(args)
    elif args.command == 'migrate-wizard':
        cli.migration_wizard(args)
    elif args.command == 'profile':
        cli.profile_code(args)
    elif args.command == 'chat':
        cli.ai_chat(args)
    # Generator Commands
    elif args.command == 'create-app':
        cli.create_app(args)
    elif args.command == 'add':
        cli.add_component(args)
    elif args.command == 'generate-config':
        cli.generate_config_tool(args)
    elif args.command == 'generate-tests':
        cli.generate_tests_tool(args)
    elif args.command == 'generate-cicd':
        cli.generate_cicd_tool(args)
    elif args.command == 'generate-deployment':
        cli.generate_deployment_tool(args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
