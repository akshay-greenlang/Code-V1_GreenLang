# -*- coding: utf-8 -*-
"""
Integration tests for CLI enhancements
"""

import pytest
import json
import yaml
from pathlib import Path
from click.testing import CliRunner
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import the enhanced CLI
from greenlang.cli.enhanced_main import cli
from greenlang.cli.jsonl_logger import JSONLLogger
from greenlang.cli.agent_registry import AgentRegistry


class TestGlobalOptions:
    """Test global --verbose and --dry-run options"""
    
    def test_verbose_flag(self):
        """Test verbose output flag"""
        runner = CliRunner()
        result = runner.invoke(cli, ['--verbose', '--help'])
        assert result.exit_code == 0
        
    def test_dry_run_flag(self):
        """Test dry-run mode"""
        runner = CliRunner()
        result = runner.invoke(cli, ['--dry-run', 'init'])
        assert result.exit_code == 0
        assert "DRY-RUN" in result.output
        
    def test_verbose_and_dry_run_together(self):
        """Test using both flags together"""
        runner = CliRunner()
        result = runner.invoke(cli, ['--verbose', '--dry-run', 'init'])
        assert result.exit_code == 0
        assert "DRY-RUN" in result.output
    
    def test_verbose_propagation_to_subcommand(self):
        """Test verbose flag propagates to subcommands"""
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['--verbose', 'agents', 'list'])
            assert result.exit_code == 0


class TestAgentDiscovery:
    """Test extensible agent discovery and loading"""
    
    def test_agent_registry_discovery(self):
        """Test agent registry discovers core agents"""
        registry = AgentRegistry()
        agents = registry.discover_agents()
        
        assert len(agents) > 0
        assert any(a['id'] == 'fuel' for a in agents)
        assert any(a['id'] == 'carbon' for a in agents)
    
    def test_agent_list_command(self):
        """Test 'gl agents list' command"""
        runner = CliRunner()
        result = runner.invoke(cli, ['agents', 'list'])
        assert result.exit_code == 0
        assert "Available Agents" in result.output
    
    def test_agent_info_command(self):
        """Test 'gl agents info' command"""
        runner = CliRunner()
        result = runner.invoke(cli, ['agents', 'info', 'fuel'])
        assert result.exit_code == 0
    
    def test_agent_template_command(self):
        """Test 'gl agents template' command"""
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['agents', 'template', 'base', '-o', 'custom.py'])
            assert result.exit_code == 0
            assert Path('custom.py').exists()
            
            content = Path('custom.py').read_text()
            assert 'class CustomAgent' in content
    
    def test_custom_agent_discovery(self):
        """Test discovery of custom agents from filesystem"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a custom agent file
            custom_agent_path = Path(tmpdir) / "my_agent.py"
            custom_agent_path.write_text('''
from greenlang.agents.base import BaseAgent

class MyCustomAgent(BaseAgent):
    """Custom test agent"""
    version = "1.0.0"
    
    def execute(self, input_data):
        return {"success": True}
''')
            
            registry = AgentRegistry(custom_paths=[tmpdir])
            agents = registry.discover_agents()
            
            custom_agents = [a for a in agents if a['type'] == 'custom']
            assert len(custom_agents) > 0


class TestJSONLLogging:
    """Test structured JSONL logging for runs"""
    
    def test_jsonl_logger_basic(self):
        """Test basic JSONL logger functionality"""
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as tmp:
            logger = JSONLLogger(Path(tmp.name))
            
            # Log events
            logger.log_start("run_001", "test_workflow")
            logger.log_step_start("step1", "agent1")
            logger.log_step_complete("step1", True, 1.5)
            logger.log_complete("run_001", True)
            
            logger.close()
            
            # Read and verify
            events = JSONLLogger.read_jsonl(Path(tmp.name))
            assert len(events) == 4
            assert events[0]['event_type'] == 'start'
            assert events[-1]['event_type'] == 'complete'
            
            os.unlink(tmp.name)
    
    def test_run_command_with_jsonl_logging(self):
        """Test 'gl run' command creates JSONL logs"""
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create a simple workflow
            workflow = {
                "name": "test_workflow",
                "description": "Test",
                "steps": [
                    {"name": "step1", "agent_id": "validator"}
                ]
            }
            Path("workflow.yaml").write_text(yaml.dump(workflow))
            
            # Run with logging
            result = runner.invoke(cli, ['run', 'workflow.yaml', '--log-dir', 'logs'])
            
            # Check log file created
            log_files = list(Path('logs').glob('*.jsonl'))
            assert len(log_files) > 0
    
    def test_jsonl_event_filtering(self):
        """Test JSONL event filtering"""
        events = [
            {"event_type": "start", "level": "INFO"},
            {"event_type": "error", "level": "ERROR"},
            {"event_type": "complete", "level": "INFO"}
        ]
        
        errors = JSONLLogger.filter_events(events, level="ERROR")
        assert len(errors) == 1
        assert errors[0]['event_type'] == 'error'


class TestReportFormatting:
    """Test flexible report output handling"""
    
    def test_report_markdown_format(self):
        """Test report generation in Markdown format"""
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create input data
            data = {"emissions": 100, "source": "test"}
            Path("input.json").write_text(json.dumps(data))
            
            result = runner.invoke(cli, ['report', 'input.json', '--format', 'md'])
            assert result.exit_code == 0
            
            # Check output file created
            report_files = list(Path('reports').glob('*.md'))
            assert len(report_files) > 0
    
    def test_report_html_format(self):
        """Test report generation in HTML format"""
        runner = CliRunner()
        with runner.isolated_filesystem():
            data = {"emissions": 100, "source": "test"}
            Path("input.json").write_text(json.dumps(data))
            
            result = runner.invoke(cli, ['report', 'input.json', '--format', 'html'])
            assert result.exit_code == 0
    
    def test_report_json_format(self):
        """Test report generation in JSON format"""
        runner = CliRunner()
        with runner.isolated_filesystem():
            data = {"emissions": 100, "source": "test"}
            Path("input.json").write_text(json.dumps(data))
            
            result = runner.invoke(cli, ['report', 'input.json', '--format', 'json'])
            assert result.exit_code == 0
    
    def test_report_custom_output_directory(self):
        """Test report with custom output directory"""
        runner = CliRunner()
        with runner.isolated_filesystem():
            data = {"emissions": 100}
            Path("input.json").write_text(json.dumps(data))
            
            result = runner.invoke(cli, ['report', 'input.json', '--out', 'custom_reports'])
            assert result.exit_code == 0
            assert Path('custom_reports').exists()
    
    def test_report_pdf_format_warning(self):
        """Test PDF format shows appropriate message when pdfkit not available"""
        runner = CliRunner()
        with runner.isolated_filesystem():
            data = {"emissions": 100}
            Path("input.json").write_text(json.dumps(data))
            
            with patch('importlib.import_module', side_effect=ImportError):
                result = runner.invoke(cli, ['report', 'input.json', '--format', 'pdf'])
                assert "pdfkit" in result.output


class TestAPIKeyHandling:
    """Test graceful handling of missing API keys"""
    
    def test_ask_without_api_key(self):
        """Test 'gl ask' command without API key shows helpful message"""
        runner = CliRunner()
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(cli, ['ask', 'test question'])
            assert result.exit_code == 0
            assert "API Key Required" in result.output
            assert "OPENAI_API_KEY" in result.output
            assert "ANTHROPIC_API_KEY" in result.output
    
    def test_ask_with_api_key(self):
        """Test 'gl ask' command with API key present"""
        runner = CliRunner()
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            with patch('greenlang.cli.assistant.AIAssistant') as MockAssistant:
                mock_instance = MagicMock()
                mock_instance.ask.return_value = "Test response"
                MockAssistant.return_value = mock_instance
                
                result = runner.invoke(cli, ['ask', 'test question'])
                assert result.exit_code == 0
                assert "API Key Required" not in result.output
    
    def test_ask_interactive_mode(self):
        """Test 'gl ask' interactive mode"""
        runner = CliRunner()
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            with patch('greenlang.cli.assistant.AIAssistant') as MockAssistant:
                mock_instance = MagicMock()
                mock_instance.ask.return_value = "Test response"
                MockAssistant.return_value = mock_instance
                
                result = runner.invoke(cli, ['ask'], input='test\nexit\n')
                assert result.exit_code == 0


class TestProjectInitialization:
    """Test project initialization command"""
    
    def test_init_command(self):
        """Test 'gl init' creates project structure"""
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['init'])
            assert result.exit_code == 0
            
            # Check directories created
            assert Path('workflows').exists()
            assert Path('data').exists()
            assert Path('reports').exists()
            assert Path('logs').exists()
            assert Path('agents/custom').exists()
            
            # Check files created
            assert Path('greenlang.yaml').exists()
            assert Path('workflows/sample.yaml').exists()
    
    def test_init_with_dry_run(self):
        """Test 'gl init' in dry-run mode"""
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ['--dry-run', 'init'])
            assert result.exit_code == 0
            assert "DRY-RUN" in result.output
            
            # Check nothing was created
            assert not Path('workflows').exists()
            assert not Path('greenlang.yaml').exists()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])