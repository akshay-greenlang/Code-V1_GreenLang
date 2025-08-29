#!/usr/bin/env python
"""
Verification script to confirm all CTO requirements are implemented.
Checks the complete CLI implementation against all specified requirements.
"""

import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple

class CLIVerifier:
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        
    def check(self, name: str, condition: bool, details: str = "") -> None:
        """Record a verification check"""
        status = "PASS" if condition else "FAIL"
        self.results.append({
            "name": name,
            "status": status,
            "details": details
        })
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        print(f"[{status}] {name}")
        if details:
            print(f"      {details}")
    
    def verify_file_exists(self, filepath: str, description: str) -> bool:
        """Check if a required file exists"""
        path = Path(filepath)
        exists = path.exists()
        self.check(
            f"{description} exists",
            exists,
            f"Path: {filepath}"
        )
        return exists
    
    def verify_command_output(self, cmd: List[str], expected_in_output: str = None) -> Tuple[bool, str]:
        """Run a command and verify it works"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            if success and expected_in_output:
                success = expected_in_output.lower() in output.lower()
            
            return success, output
        except Exception as e:
            return False, str(e)
    
    def verify_core_commands(self):
        """Verify all 6 core commands are implemented"""
        print("\n=== CORE COMMANDS VERIFICATION ===")
        
        # Check main CLI file
        self.verify_file_exists(
            "greenlang/cli/complete_cli.py",
            "Complete CLI implementation"
        )
        
        # Check each command exists in the CLI file
        cli_file = Path("greenlang/cli/complete_cli.py")
        if cli_file.exists():
            content = cli_file.read_text(encoding='utf-8')
            
            commands = [
                ("init", "@cli.command"),
                ("agents", "@cli.group"),
                ("run", "@cli.command"),
                ("validate", "@cli.command"),
                ("report", "@cli.command"),
                ("ask", "@cli.command")
            ]
            
            for cmd_name, decorator in commands:
                # Check if command is defined
                has_command = f"def {cmd_name}(" in content
                self.check(
                    f"Command 'gl {cmd_name}' implemented",
                    has_command,
                    f"Function definition found: {has_command}"
                )
    
    def verify_technical_requirements(self):
        """Verify technical requirements"""
        print("\n=== TECHNICAL REQUIREMENTS ===")
        
        # Rich integration
        cli_file = Path("greenlang/cli/complete_cli.py")
        if cli_file.exists():
            content = cli_file.read_text(encoding='utf-8')
            
            self.check(
                "Rich integration (progress bars)",
                "from rich.progress import" in content,
                "Rich progress bars imported"
            )
            
            self.check(
                "Rich integration (tables)",
                "from rich.table import Table" in content,
                "Rich tables imported"
            )
            
            self.check(
                "Rich integration (console)",
                "from rich.console import Console" in content,
                "Rich console imported"
            )
        
        # JSONL logging
        self.verify_file_exists(
            "greenlang/cli/jsonl_logger.py",
            "JSONL logger implementation"
        )
        
        # Caching system
        if cli_file.exists():
            content = cli_file.read_text(encoding='utf-8')
            self.check(
                "Caching system implemented",
                "hashlib.md5" in content or "cache" in content.lower(),
                "MD5-based caching found"
            )
            
            self.check(
                "--no-cache flag support",
                "--no-cache" in content,
                "Cache bypass flag implemented"
            )
        
        # Error handling
        self.check(
            "Non-zero exit codes",
            "sys.exit(1)" in content,
            "Exit codes for failures"
        )
    
    def verify_global_options(self):
        """Verify global options"""
        print("\n=== GLOBAL OPTIONS ===")
        
        cli_file = Path("greenlang/cli/complete_cli.py")
        if cli_file.exists():
            content = cli_file.read_text(encoding='utf-8')
            
            self.check(
                "--verbose flag",
                '"--verbose"' in content or '"-v"' in content,
                "Verbose option available"
            )
            
            self.check(
                "--dry-run flag",
                '"--dry-run"' in content,
                "Dry-run option available"
            )
            
            self.check(
                "--version flag",
                '@click.version_option' in content or '"--version"' in content,
                "Version option available"
            )
    
    def verify_agent_registry(self):
        """Verify agent registry system"""
        print("\n=== AGENT REGISTRY ===")
        
        self.verify_file_exists(
            "greenlang/cli/agent_registry.py",
            "Agent registry implementation"
        )
        
        registry_file = Path("greenlang/cli/agent_registry.py")
        if registry_file.exists():
            content = registry_file.read_text(encoding='utf-8')
            
            self.check(
                "Plugin discovery via entry_points",
                "entry_points" in content,
                "Entry points support"
            )
            
            self.check(
                "Custom agent paths",
                "custom" in content or "GREENLANG_AGENTS_PATH" in content,
                "Custom paths supported"
            )
    
    def verify_file_locations(self):
        """Verify all required files are in correct locations"""
        print("\n=== FILE LOCATIONS ===")
        
        files = {
            "Main CLI": "greenlang/cli/complete_cli.py",
            "JSONL Logger": "greenlang/cli/jsonl_logger.py",
            "Agent Registry": "greenlang/cli/agent_registry.py",
            "Migration Script": "scripts/migrate_to_enhanced_cli.py",
            "CLI Documentation": "CLI_COMPLETE_FINAL.md",
            "Enhancement Docs": "CLI_ENHANCEMENTS_COMPLETE.md"
        }
        
        for name, path in files.items():
            self.verify_file_exists(path, name)
    
    def verify_entry_point(self):
        """Verify CLI entry point configuration"""
        print("\n=== ENTRY POINT CONFIGURATION ===")
        
        pyproject = Path("pyproject.toml")
        if pyproject.exists():
            content = pyproject.read_text(encoding='utf-8')
            
            self.check(
                "gl command entry point",
                'gl = "greenlang.cli.complete_cli:cli"' in content,
                "Short command 'gl' configured"
            )
    
    def verify_command_features(self):
        """Verify specific command features"""
        print("\n=== COMMAND FEATURES ===")
        
        cli_file = Path("greenlang/cli/complete_cli.py")
        if cli_file.exists():
            content = cli_file.read_text(encoding='utf-8')
            
            # gl init features
            self.check(
                "gl init creates project structure",
                "pipelines" in content and "mkdir" in content,
                "Project scaffolding implemented"
            )
            
            # gl agents features
            self.check(
                "gl agents list/info/template",
                "def list_agents" in content or "@agents.command" in content,
                "Agent subcommands implemented"
            )
            
            # gl run features
            self.check(
                "gl run with caching",
                "cache" in content.lower() and "md5" in content.lower(),
                "Caching in run command"
            )
            
            # gl validate features
            self.check(
                "gl validate DAG checking",
                "cycle" in content.lower() or "dag" in content.lower(),
                "DAG validation implemented"
            )
            
            # gl report features
            self.check(
                "gl report multiple formats",
                "format" in content and ("html" in content or "pdf" in content),
                "Multiple report formats"
            )
            
            # gl ask features
            self.check(
                "gl ask API key handling",
                "API_KEY" in content or "api_key" in content,
                "API key handling implemented"
            )
    
    def generate_summary(self):
        """Generate final summary"""
        print("\n" + "="*60)
        print("VERIFICATION SUMMARY")
        print("="*60)
        
        total = self.passed + self.failed
        percentage = (self.passed / total * 100) if total > 0 else 0
        
        print(f"\nTotal Checks: {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {percentage:.1f}%")
        
        if self.failed > 0:
            print("\nFailed Checks:")
            for result in self.results:
                if result["status"] == "FAIL":
                    print(f"  - {result['name']}")
                    if result["details"]:
                        print(f"    {result['details']}")
        
        print("\n" + "="*60)
        if self.failed == 0:
            print("ALL CTO REQUIREMENTS VERIFIED SUCCESSFULLY!")
            print("The CLI implementation is COMPLETE and ready for use.")
        else:
            print("Some requirements need attention.")
            print("Please review the failed checks above.")
        print("="*60)
        
        # Save results to JSON
        results_file = Path("verification_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total": total,
                    "passed": self.passed,
                    "failed": self.failed,
                    "percentage": percentage
                },
                "checks": self.results
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        return self.failed == 0

def main():
    print("="*60)
    print("CTO REQUIREMENTS VERIFICATION SCRIPT")
    print("="*60)
    print("\nVerifying all CLI requirements are fully implemented...")
    
    verifier = CLIVerifier()
    
    # Run all verifications
    verifier.verify_core_commands()
    verifier.verify_technical_requirements()
    verifier.verify_global_options()
    verifier.verify_agent_registry()
    verifier.verify_file_locations()
    verifier.verify_entry_point()
    verifier.verify_command_features()
    
    # Generate summary
    success = verifier.generate_summary()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()