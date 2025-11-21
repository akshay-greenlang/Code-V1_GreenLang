#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to demonstrate enforcement system
Creates example violations and shows how they're caught
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def create_test_files():
    """Create test files with violations"""
    test_dir = Path(__file__).parent.parent.parent / "test_enforcement_examples"
    test_dir.mkdir(exist_ok=True)

    # Example 1: Forbidden import (openai)
    (test_dir / "example1_forbidden_import.py").write_text("""
# Example 1: Forbidden Import Violation
import openai

def get_completion(prompt):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response
""")

    # Example 2: Custom Agent without inheritance
    (test_dir / "example2_custom_agent.py").write_text("""
# Example 2: Custom Agent Violation
class CustomAgent:
    def __init__(self, name):
        self.name = name

    def process(self, data):
        return f"Processed {data}"

agent = CustomAgent("my-agent")
""")

    # Example 3: Compliant code
    (test_dir / "example3_compliant.py").write_text("""
# Example 3: Compliant Code
from greenlang.sdk.base import Agent
from greenlang.intelligence import ChatSession

class MyAgent(Agent):
    def validate(self, input_data):
        return True

    def process(self, input_data):
        session = ChatSession()
        return session.chat(f"Process {input_data}")

agent = MyAgent()
""")

    # Example 4: Direct Redis usage
    (test_dir / "example4_redis.py").write_text("""
# Example 4: Direct Redis Violation
import redis

r = redis.Redis(host='localhost', port=6379)

def cache_result(key, value):
    r.setex(key, 3600, value)

def get_cached(key):
    return r.get(key)
""")

    # Example 5: Custom auth
    (test_dir / "example5_custom_auth.py").write_text("""
# Example 5: Custom Auth Violation
from jose import jwt
from passlib.context import CryptContext

SECRET_KEY = "secret"
pwd_context = CryptContext(schemes=["bcrypt"])

def hash_password(password):
    return pwd_context.hash(password)

def create_token(user_id):
    return jwt.encode({"sub": user_id}, SECRET_KEY)
""")

    return test_dir


def run_linter(test_dir):
    """Run linter on test files"""
    print("=" * 80)
    print("RUNNING STATIC LINTER")
    print("=" * 80)
    print()

    # Import linter
    sys.path.insert(0, str(Path(__file__).parent.parent / "linters"))
    from infrastructure_first import lint_directory, format_violations_text

    violations = lint_directory(test_dir)

    print(format_violations_text(violations))
    print()

    return violations


def run_ium_calculator(test_dir):
    """Run IUM calculator on test files"""
    print("=" * 80)
    print("CALCULATING INFRASTRUCTURE USAGE METRICS")
    print("=" * 80)
    print()

    sys.path.insert(0, str(Path(__file__).parent))
    from calculate_ium import analyze_file, aggregate_metrics

    metrics_list = []
    for py_file in test_dir.glob("*.py"):
        metrics = analyze_file(py_file)
        metrics_list.append(metrics)

    report = aggregate_metrics(metrics_list)

    print(f"Overall IUM Score: {report['percentage']:.1f}%")
    print()
    print("Details:")
    for category, data in report['details'].items():
        print(f"  {category.title():15s}: {data['percentage']:5.1f}% ({data['greenlang']}/{data['total']})")
    print()

    return report


def show_recommendations():
    """Show recommendations for fixing violations"""
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    recommendations = [
        ("❌ example1_forbidden_import.py", "Replace 'import openai' with 'from greenlang.intelligence import ChatSession'"),
        ("❌ example2_custom_agent.py", "Make CustomAgent inherit from greenlang.sdk.base.Agent"),
        ("✅ example3_compliant.py", "No changes needed - fully compliant!"),
        ("❌ example4_redis.py", "Replace 'import redis' with 'from greenlang.cache import CacheManager'"),
        ("❌ example5_custom_auth.py", "Replace custom auth with 'from greenlang.auth import AuthManager'"),
    ]

    for file_name, recommendation in recommendations:
        print(f"{file_name}")
        print(f"  → {recommendation}")
        print()


def main():
    """Main test function"""
    print()
    print("🌱 GreenLang Infrastructure-First Enforcement Test")
    print()

    # Create test files
    print("Creating test files with violations...")
    test_dir = create_test_files()
    print(f"Created test files in: {test_dir}")
    print()

    # Run linter
    violations = run_linter(test_dir)

    # Run IUM calculator
    report = run_ium_calculator(test_dir)

    # Show recommendations
    show_recommendations()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Total Violations: {len(violations)}")
    print(f"Overall IUM Score: {report['percentage']:.1f}%")
    print()

    if report['percentage'] < 95:
        print("❌ IUM score below 95% threshold")
        print("   Action: Fix violations or create ADR")
    else:
        print("✅ IUM score meets 95% threshold")

    print()
    print("Next steps:")
    print("  1. Review violations above")
    print("  2. Fix by using GreenLang infrastructure")
    print("  3. If custom code needed, create ADR in .greenlang/adrs/")
    print("  4. See .greenlang/ENFORCEMENT_GUIDE.md for details")
    print()


if __name__ == '__main__':
    main()
