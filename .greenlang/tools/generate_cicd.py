#!/usr/bin/env python3
"""
CI/CD Pipeline Generator

Generate CI/CD configurations for GitHub Actions, GitLab CI, and Jenkins.
Includes linting, testing, security scanning, and deployment.
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any


class CICDGenerator:
    """Generate CI/CD pipeline configurations."""

    @staticmethod
    def generate_github_actions(app_name: str, features: Dict[str, bool] = None) -> str:
        """Generate GitHub Actions workflow."""

        if features is None:
            features = {}

        workflow = {
            'name': 'CI/CD Pipeline',
            'on': {
                'push': {
                    'branches': ['main', 'develop']
                },
                'pull_request': {
                    'branches': ['main', 'develop']
                }
            },
            'env': {
                'PYTHON_VERSION': '3.11'
            },
            'jobs': {
                'lint': {
                    'name': 'Code Quality',
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {'uses': 'actions/checkout@v4'},
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v4',
                            'with': {'python-version': '${{ env.PYTHON_VERSION }}'}
                        },
                        {
                            'name': 'Install dependencies',
                            'run': 'pip install black isort mypy pylint'
                        },
                        {
                            'name': 'Run Black',
                            'run': 'black --check app/ tests/'
                        },
                        {
                            'name': 'Run isort',
                            'run': 'isort --check-only app/ tests/'
                        },
                        {
                            'name': 'Run mypy',
                            'run': 'mypy app/',
                            'continue-on-error': True
                        },
                        {
                            'name': 'Run pylint',
                            'run': 'pylint app/',
                            'continue-on-error': True
                        }
                    ]
                },
                'test': {
                    'name': 'Tests',
                    'runs-on': 'ubuntu-latest',
                    'strategy': {
                        'matrix': {
                            'python-version': ['3.10', '3.11', '3.12']
                        }
                    },
                    'steps': [
                        {'uses': 'actions/checkout@v4'},
                        {
                            'name': 'Set up Python ${{ matrix.python-version }}',
                            'uses': 'actions/setup-python@v4',
                            'with': {'python-version': '${{ matrix.python-version }}'}
                        },
                        {
                            'name': 'Install dependencies',
                            'run': 'pip install -r requirements.txt\npip install pytest pytest-cov'
                        },
                        {
                            'name': 'Run tests',
                            'run': 'pytest tests/ -v --cov=app --cov-report=xml --cov-report=html'
                        },
                        {
                            'name': 'Upload coverage',
                            'uses': 'codecov/codecov-action@v3',
                            'with': {
                                'file': './coverage.xml',
                                'fail_ci_if_error': False
                            }
                        }
                    ]
                },
                'security': {
                    'name': 'Security Scan',
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {'uses': 'actions/checkout@v4'},
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v4',
                            'with': {'python-version': '${{ env.PYTHON_VERSION }}'}
                        },
                        {
                            'name': 'Install bandit',
                            'run': 'pip install bandit[toml]'
                        },
                        {
                            'name': 'Run Bandit',
                            'run': 'bandit -r app/ -f json -o bandit-report.json',
                            'continue-on-error': True
                        },
                        {
                            'name': 'Run Safety',
                            'run': 'pip install safety && safety check',
                            'continue-on-error': True
                        }
                    ]
                },
                'build': {
                    'name': 'Build Docker Image',
                    'runs-on': 'ubuntu-latest',
                    'needs': ['lint', 'test', 'security'],
                    'if': "github.event_name == 'push' && github.ref == 'refs/heads/main'",
                    'steps': [
                        {'uses': 'actions/checkout@v4'},
                        {
                            'name': 'Set up Docker Buildx',
                            'uses': 'docker/setup-buildx-action@v3'
                        },
                        {
                            'name': 'Login to Docker Hub',
                            'uses': 'docker/login-action@v3',
                            'with': {
                                'username': '${{ secrets.DOCKER_USERNAME }}',
                                'password': '${{ secrets.DOCKER_PASSWORD }}'
                            }
                        },
                        {
                            'name': 'Build and push',
                            'uses': 'docker/build-push-action@v5',
                            'with': {
                                'context': '.',
                                'push': True,
                                'tags': f'${{{{ secrets.DOCKER_USERNAME }}}}/{app_name}:latest,${{{{ secrets.DOCKER_USERNAME }}}}/{app_name}:${{{{ github.sha }}}}'
                            }
                        }
                    ]
                }
            }
        }

        if features.get('deploy', False):
            workflow['jobs']['deploy'] = {
                'name': 'Deploy',
                'runs-on': 'ubuntu-latest',
                'needs': ['build'],
                'if': "github.event_name == 'push' && github.ref == 'refs/heads/main'",
                'steps': [
                    {'uses': 'actions/checkout@v4'},
                    {
                        'name': 'Deploy to production',
                        'run': 'echo "Add your deployment script here"'
                    }
                ]
            }

        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)

    @staticmethod
    def generate_gitlab_ci() -> str:
        """Generate GitLab CI configuration."""

        return '''stages:
  - lint
  - test
  - security
  - build
  - deploy

variables:
  PYTHON_VERSION: "3.11"

# Lint stage
lint:
  stage: lint
  image: python:${PYTHON_VERSION}
  before_script:
    - pip install black isort mypy pylint
  script:
    - black --check app/ tests/
    - isort --check-only app/ tests/
    - mypy app/ || true
    - pylint app/ || true

# Test stage
test:
  stage: test
  image: python:${PYTHON_VERSION}
  before_script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov
  script:
    - pytest tests/ -v --cov=app --cov-report=xml --cov-report=html
  coverage: '/(?i)total.*? (100(?:\\.0+)?\\%|[1-9]?\\d(?:\\.\\d+)?\\%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    paths:
      - htmlcov/

# Security scan
security:
  stage: security
  image: python:${PYTHON_VERSION}
  before_script:
    - pip install bandit[toml] safety
  script:
    - bandit -r app/ -f json -o bandit-report.json || true
    - safety check || true
  artifacts:
    paths:
      - bandit-report.json
  allow_failure: true

# Build Docker image
build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker tag $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA $CI_REGISTRY_IMAGE:latest
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - docker push $CI_REGISTRY_IMAGE:latest
  only:
    - main

# Deploy to production
deploy:
  stage: deploy
  image: alpine:latest
  before_script:
    - apk add --no-cache curl
  script:
    - echo "Add your deployment script here"
  only:
    - main
  when: manual
'''

    @staticmethod
    def generate_jenkinsfile() -> str:
        """Generate Jenkinsfile."""

        return '''pipeline {
    agent any

    environment {
        PYTHON_VERSION = '3.11'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup') {
            steps {
                sh """
                    python${PYTHON_VERSION} -m venv venv
                    . venv/bin/activate
                    pip install -r requirements.txt
                    pip install pytest pytest-cov black isort mypy pylint bandit safety
                """
            }
        }

        stage('Lint') {
            parallel {
                stage('Black') {
                    steps {
                        sh """
                            . venv/bin/activate
                            black --check app/ tests/
                        """
                    }
                }
                stage('isort') {
                    steps {
                        sh """
                            . venv/bin/activate
                            isort --check-only app/ tests/
                        """
                    }
                }
                stage('mypy') {
                    steps {
                        sh """
                            . venv/bin/activate
                            mypy app/ || true
                        """
                    }
                }
                stage('pylint') {
                    steps {
                        sh """
                            . venv/bin/activate
                            pylint app/ || true
                        """
                    }
                }
            }
        }

        stage('Test') {
            steps {
                sh """
                    . venv/bin/activate
                    pytest tests/ -v --cov=app --cov-report=xml --cov-report=html
                """
            }
            post {
                always {
                    publishHTML([
                        reportDir: 'htmlcov',
                        reportFiles: 'index.html',
                        reportName: 'Coverage Report'
                    ])
                }
            }
        }

        stage('Security') {
            parallel {
                stage('Bandit') {
                    steps {
                        sh """
                            . venv/bin/activate
                            bandit -r app/ -f json -o bandit-report.json || true
                        """
                    }
                }
                stage('Safety') {
                    steps {
                        sh """
                            . venv/bin/activate
                            safety check || true
                        """
                    }
                }
            }
        }

        stage('Build') {
            when {
                branch 'main'
            }
            steps {
                sh """
                    docker build -t myapp:${BUILD_NUMBER} .
                    docker tag myapp:${BUILD_NUMBER} myapp:latest
                """
            }
        }

        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                echo 'Deploying...'
                // Add deployment commands here
            }
        }
    }

    post {
        always {
            cleanWs()
        }
        failure {
            emailext (
                subject: "Build Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Check console output at ${env.BUILD_URL}",
                recipientProviders: [developers(), requestor()]
            )
        }
    }
}
'''


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate CI/CD pipeline configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate GitHub Actions workflow
  greenlang generate-cicd --platform github

  # Generate GitLab CI configuration
  greenlang generate-cicd --platform gitlab

  # Generate Jenkinsfile
  greenlang generate-cicd --platform jenkins

  # Generate all platforms
  greenlang generate-cicd --all
        """
    )

    parser.add_argument('--platform', choices=['github', 'gitlab', 'jenkins'],
                        help='CI/CD platform')
    parser.add_argument('--all', action='store_true', help='Generate for all platforms')
    parser.add_argument('--app-name', default='greenlang-app', help='Application name')
    parser.add_argument('--with-deploy', action='store_true', help='Include deployment stage')
    parser.add_argument('--output-dir', default='.', help='Output directory')

    args = parser.parse_args()

    if not args.platform and not args.all:
        parser.print_help()
        sys.exit(1)

    generator = CICDGenerator()
    output_dir = Path(args.output_dir)

    features = {
        'deploy': args.with_deploy
    }

    print("\nGenerating CI/CD configurations...\n")

    # Generate based on platform
    if args.platform == 'github' or args.all:
        github_dir = output_dir / '.github' / 'workflows'
        github_dir.mkdir(parents=True, exist_ok=True)

        workflow = generator.generate_github_actions(args.app_name, features)
        output_file = github_dir / 'ci.yml'
        output_file.write_text(workflow, encoding='utf-8')

        print(f"Generated: {output_file}")

    if args.platform == 'gitlab' or args.all:
        gitlab_config = generator.generate_gitlab_ci()
        output_file = output_dir / '.gitlab-ci.yml'
        output_file.write_text(gitlab_config, encoding='utf-8')

        print(f"Generated: {output_file}")

    if args.platform == 'jenkins' or args.all:
        jenkinsfile = generator.generate_jenkinsfile()
        output_file = output_dir / 'Jenkinsfile'
        output_file.write_text(jenkinsfile, encoding='utf-8')

        print(f"Generated: {output_file}")

    print("\nCI/CD configurations generated successfully!")
    print("\nNext steps:")
    print("  1. Review and customize the generated files")
    print("  2. Set up required secrets/credentials in your CI/CD platform")
    print("  3. Push to your repository")
    print()


if __name__ == '__main__':
    main()
