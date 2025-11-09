#!/usr/bin/env python3
"""
Configuration Generator

Generate configuration files for GreenLang applications.
Supports multiple environments (dev, staging, prod) and secure defaults.
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List


class ConfigGenerator:
    """Generate configuration files."""

    @staticmethod
    def generate_config_yaml(app_name: str, environments: List[str] = None) -> str:
        """Generate config.yaml with multiple environments."""

        if environments is None:
            environments = ['development', 'staging', 'production']

        config = {
            'app': {
                'name': app_name,
                'version': '1.0.0',
                'description': f'{app_name} - GreenLang Application'
            },
            'environments': {}
        }

        # Development environment
        config['environments']['development'] = {
            'debug': True,
            'log_level': 'DEBUG',
            'llm': {
                'provider': 'openai',
                'model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 2000,
                'timeout': 30
            },
            'cache': {
                'type': 'memory',
                'ttl': 3600,
                'max_size': 1000
            },
            'database': {
                'pool_size': 5,
                'max_overflow': 10,
                'echo': True
            },
            'monitoring': {
                'enabled': True,
                'metrics_port': 9090
            },
            'rate_limiting': {
                'enabled': False
            }
        }

        # Staging environment
        config['environments']['staging'] = {
            'debug': False,
            'log_level': 'INFO',
            'llm': {
                'provider': 'openai',
                'model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 2000,
                'timeout': 60
            },
            'cache': {
                'type': 'redis',
                'ttl': 3600,
                'max_size': 10000
            },
            'database': {
                'pool_size': 10,
                'max_overflow': 20,
                'echo': False
            },
            'monitoring': {
                'enabled': True,
                'metrics_port': 9090
            },
            'rate_limiting': {
                'enabled': True,
                'requests_per_minute': 100
            }
        }

        # Production environment
        config['environments']['production'] = {
            'debug': False,
            'log_level': 'WARNING',
            'llm': {
                'provider': 'openai',
                'model': 'gpt-4',
                'temperature': 0.5,
                'max_tokens': 2000,
                'timeout': 120,
                'retry_attempts': 3,
                'retry_delay': 1
            },
            'cache': {
                'type': 'redis',
                'ttl': 7200,
                'max_size': 100000
            },
            'database': {
                'pool_size': 20,
                'max_overflow': 40,
                'echo': False,
                'pool_pre_ping': True
            },
            'monitoring': {
                'enabled': True,
                'metrics_port': 9090,
                'alerting': True
            },
            'rate_limiting': {
                'enabled': True,
                'requests_per_minute': 1000
            },
            'security': {
                'enable_cors': True,
                'allowed_origins': ['https://yourdomain.com'],
                'enable_csrf': True
            }
        }

        return yaml.dump(config, default_flow_style=False, sort_keys=False)

    @staticmethod
    def generate_env_file(environment: str, app_name: str, features: Dict[str, bool] = None) -> str:
        """Generate .env file for specific environment."""

        if features is None:
            features = {}

        lines = [
            '# Environment Configuration',
            f'ENVIRONMENT={environment}',
            f'APP_NAME={app_name}',
            'APP_VERSION=1.0.0',
            '',
            '# Logging',
            'LOG_LEVEL=INFO' if environment == 'production' else 'LOG_LEVEL=DEBUG',
            'LOG_FORMAT=json' if environment == 'production' else 'LOG_FORMAT=text',
            ''
        ]

        if features.get('llm', True):
            lines.extend([
                '# LLM Configuration',
                'LLM_PROVIDER=openai',
                'LLM_MODEL=gpt-4',
                'LLM_TEMPERATURE=0.7',
                'LLM_MAX_TOKENS=2000',
                'OPENAI_API_KEY=your-openai-key-here',
                'ANTHROPIC_API_KEY=your-anthropic-key-here',
                ''
            ])

        if features.get('cache', True):
            if environment == 'production':
                lines.extend([
                    '# Cache Configuration (Redis)',
                    'CACHE_TYPE=redis',
                    'REDIS_URL=redis://redis:6379/0',
                    'CACHE_TTL=7200',
                    ''
                ])
            else:
                lines.extend([
                    '# Cache Configuration',
                    'CACHE_TYPE=memory',
                    'CACHE_TTL=3600',
                    ''
                ])

        if features.get('database', True):
            lines.extend([
                '# Database Configuration',
                'DATABASE_URL=postgresql://user:password@localhost:5432/dbname',
                'DB_POOL_SIZE=10' if environment != 'production' else 'DB_POOL_SIZE=20',
                'DB_MAX_OVERFLOW=20' if environment != 'production' else 'DB_MAX_OVERFLOW=40',
                ''
            ])

        if features.get('monitoring', True):
            lines.extend([
                '# Monitoring',
                'ENABLE_METRICS=true',
                'METRICS_PORT=9090',
                ''
            ])

        if environment == 'production':
            lines.extend([
                '# Security',
                'SECRET_KEY=your-secret-key-here-change-in-production',
                'ENABLE_CORS=true',
                'ALLOWED_ORIGINS=https://yourdomain.com',
                '',
                '# Rate Limiting',
                'ENABLE_RATE_LIMITING=true',
                'RATE_LIMIT_REQUESTS=1000',
                'RATE_LIMIT_WINDOW=60',
                ''
            ])

        return '\n'.join(lines)

    @staticmethod
    def generate_secrets_yaml(app_name: str) -> str:
        """Generate secrets.yaml template (for Kubernetes)."""

        secrets = {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': f'{app_name}-secrets'
            },
            'type': 'Opaque',
            'stringData': {
                'OPENAI_API_KEY': 'your-openai-key',
                'ANTHROPIC_API_KEY': 'your-anthropic-key',
                'DATABASE_PASSWORD': 'your-db-password',
                'REDIS_PASSWORD': 'your-redis-password',
                'SECRET_KEY': 'your-secret-key'
            }
        }

        return yaml.dump(secrets, default_flow_style=False, sort_keys=False)

    @staticmethod
    def validate_config(config_data: Dict[str, Any]) -> List[str]:
        """Validate configuration and return list of issues."""

        issues = []

        # Check required fields
        if 'app' not in config_data:
            issues.append("Missing 'app' section")

        if 'environments' not in config_data:
            issues.append("Missing 'environments' section")

        # Check for placeholder values in production
        if 'production' in config_data.get('environments', {}):
            prod_config = config_data['environments']['production']

            # Check security settings
            if not prod_config.get('security', {}).get('enable_cors'):
                issues.append("CORS should be enabled in production")

            if not prod_config.get('rate_limiting', {}).get('enabled'):
                issues.append("Rate limiting should be enabled in production")

        return issues


class ConfigWizard:
    """Interactive configuration wizard."""

    def __init__(self, app_dir: str = '.'):
        self.app_dir = Path(app_dir)
        self.generator = ConfigGenerator()

    def run(self):
        """Run interactive configuration wizard."""

        print("\n" + "="*70)
        print("GreenLang Configuration Generator - Interactive Mode")
        print("="*70 + "\n")

        # Get app name
        app_name = input("Application name: ").strip()
        if not app_name:
            print("Error: Application name is required")
            sys.exit(1)

        # Features
        print("\nSelect features to configure:")
        use_llm = input("  LLM integration? (y/n): ").lower().strip() == 'y'
        use_cache = input("  Caching? (y/n): ").lower().strip() == 'y'
        use_database = input("  Database? (y/n): ").lower().strip() == 'y'
        use_monitoring = input("  Monitoring? (y/n): ").lower().strip() == 'y'

        features = {
            'llm': use_llm,
            'cache': use_cache,
            'database': use_database,
            'monitoring': use_monitoring
        }

        # Environments
        print("\nGenerate configurations for:")
        gen_dev = input("  Development? (y/n): ").lower().strip() == 'y'
        gen_staging = input("  Staging? (y/n): ").lower().strip() == 'y'
        gen_prod = input("  Production? (y/n): ").lower().strip() == 'y'

        print("\nGenerating configuration files...\n")

        # Generate config.yaml
        config_yaml = self.generator.generate_config_yaml(app_name)
        config_file = self.app_dir / 'config.yaml'
        config_file.write_text(config_yaml, encoding='utf-8')
        print(f"Generated: {config_file}")

        # Generate environment files
        if gen_dev:
            dev_env = self.generator.generate_env_file('development', app_name, features)
            dev_file = self.app_dir / '.env.development'
            dev_file.write_text(dev_env, encoding='utf-8')
            print(f"Generated: {dev_file}")

        if gen_staging:
            staging_env = self.generator.generate_env_file('staging', app_name, features)
            staging_file = self.app_dir / '.env.staging'
            staging_file.write_text(staging_env, encoding='utf-8')
            print(f"Generated: {staging_file}")

        if gen_prod:
            prod_env = self.generator.generate_env_file('production', app_name, features)
            prod_file = self.app_dir / '.env.production'
            prod_file.write_text(prod_env, encoding='utf-8')
            print(f"Generated: {prod_file}")

        # Generate .env.example
        example_env = self.generator.generate_env_file('development', app_name, features)
        example_file = self.app_dir / '.env.example'
        example_file.write_text(example_env, encoding='utf-8')
        print(f"Generated: {example_file}")

        # Generate secrets template
        secrets_yaml = self.generator.generate_secrets_yaml(app_name)
        secrets_file = self.app_dir / 'secrets.yaml.template'
        secrets_file.write_text(secrets_yaml, encoding='utf-8')
        print(f"Generated: {secrets_file}")

        print("\n" + "="*70)
        print("Configuration files generated successfully!")
        print("="*70 + "\n")

        print("Next steps:")
        print("  1. Copy .env.example to .env")
        print("  2. Update .env with your actual values")
        print("  3. NEVER commit .env or secrets.yaml to version control")
        print("  4. For production, use environment variables or secret management")
        print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate configuration files for GreenLang applications',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  greenlang generate-config

  # Generate for specific environment
  greenlang generate-config --environment production --output .env.production

  # Generate all environments
  greenlang generate-config --all-environments

  # Validate existing configuration
  greenlang generate-config --validate config.yaml
        """
    )

    parser.add_argument('--app-name', help='Application name')
    parser.add_argument('--environment', choices=['development', 'staging', 'production'],
                        help='Generate for specific environment')
    parser.add_argument('--all-environments', action='store_true', help='Generate all environment configs')
    parser.add_argument('--validate', help='Validate configuration file')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--app-dir', default='.', help='Application directory')

    args = parser.parse_args()

    generator = ConfigGenerator()

    # Validate mode
    if args.validate:
        print(f"Validating configuration: {args.validate}\n")

        try:
            with open(args.validate, 'r') as f:
                config_data = yaml.safe_load(f)

            issues = generator.validate_config(config_data)

            if not issues:
                print("Configuration is valid!")
            else:
                print("Configuration issues found:")
                for issue in issues:
                    print(f"  - {issue}")

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

        sys.exit(0)

    # Interactive mode
    if args.interactive or not args.app_name:
        wizard = ConfigWizard(args.app_dir)
        wizard.run()
        sys.exit(0)

    # Generate specific environment
    if args.environment:
        env_content = generator.generate_env_file(args.environment, args.app_name)
        output_file = args.output or f'.env.{args.environment}'

        with open(output_file, 'w') as f:
            f.write(env_content)

        print(f"Generated: {output_file}")

    # Generate all environments
    elif args.all_environments:
        for env in ['development', 'staging', 'production']:
            env_content = generator.generate_env_file(env, args.app_name)
            output_file = f'.env.{env}'

            with open(output_file, 'w') as f:
                f.write(env_content)

            print(f"Generated: {output_file}")

        # Generate config.yaml
        config_yaml = generator.generate_config_yaml(args.app_name)
        with open('config.yaml', 'w') as f:
            f.write(config_yaml)

        print("Generated: config.yaml")


if __name__ == '__main__':
    main()
