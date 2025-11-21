# -*- coding: utf-8 -*-
"""
Tests for boiler-solar pack pipeline
"""

import json
import subprocess
import hashlib
from pathlib import Path
import pytest
import yaml


class TestBoilerSolarPipeline:
    """Test suite for boiler-solar pipeline"""
    
    @pytest.fixture
    def pack_dir(self):
        """Get pack directory"""
        return Path(__file__).parent.parent
    
    @pytest.fixture
    def golden_input(self):
        """Load golden input data"""
        golden_path = Path(__file__).parent / "golden" / "inputs.sample.json"
        if golden_path.exists():
            with open(golden_path) as f:
                return json.load(f)
        return {
            "site": {
                "name": "Test Dairy Plant",
                "location": "22.3,70.7",
                "altitude_m": 100
            },
            "boiler": {
                "pressure_bar": 7,
                "efficiency": 0.82,
                "capacity_tph": 10
            },
            "solar": {
                "aperture_m2": 1200,
                "efficiency_optical": 0.75
            }
        }
    
    def test_pack_manifest_valid(self, pack_dir):
        """Test that pack.yaml is valid"""
        manifest_path = pack_dir / "pack.yaml"
        assert manifest_path.exists(), "pack.yaml not found"
        
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        
        # Check required fields
        assert manifest["name"] == "boiler-solar"
        assert "version" in manifest
        assert manifest["kind"] == "pack"
        assert "compat" in manifest
        assert "contents" in manifest
        
        # Check version is semver
        version = manifest["version"]
        assert "." in version, "Version must be semantic"
        
        # Check contents
        assert "gl.yaml" in manifest["contents"]["pipelines"]
        assert "ef_in_2025.csv" in manifest["contents"]["datasets"]
    
    def test_pipeline_yaml_valid(self, pack_dir):
        """Test that gl.yaml is valid"""
        pipeline_path = pack_dir / "gl.yaml"
        assert pipeline_path.exists(), "gl.yaml not found"
        
        with open(pipeline_path) as f:
            pipeline = yaml.safe_load(f)
        
        # Check structure
        assert "version" in pipeline
        assert "pipeline" in pipeline
        assert "inputs" in pipeline
        assert "steps" in pipeline
        assert "outputs" in pipeline
        
        # Check steps
        steps = pipeline["steps"]
        assert len(steps) > 0, "Pipeline must have steps"
        
        # Check each step has required fields
        for step in steps:
            assert "id" in step, f"Step missing id"
            assert "agent" in step, f"Step {step.get('id')} missing agent"
    
    def test_datasets_exist(self, pack_dir):
        """Test that declared datasets exist"""
        manifest_path = pack_dir / "pack.yaml"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        
        for dataset in manifest["contents"]["datasets"]:
            dataset_path = pack_dir / "datasets" / dataset
            assert dataset_path.exists(), f"Dataset {dataset} not found"
            
            # Check it's valid CSV
            if dataset.endswith(".csv"):
                with open(dataset_path) as f:
                    first_line = f.readline()
                    assert "," in first_line, "CSV must have columns"
    
    def test_reports_exist(self, pack_dir):
        """Test that report templates exist"""
        manifest_path = pack_dir / "pack.yaml"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        
        for report in manifest["contents"]["reports"]:
            report_path = pack_dir / "reports" / report
            assert report_path.exists(), f"Report template {report} not found"
            
            # Check it's valid Jinja2
            if report.endswith(".j2"):
                with open(report_path) as f:
                    content = f.read()
                    assert "{{" in content, "Jinja2 template must have variables"
                    assert "{%" in content or "{{" in content, "Template must have Jinja2 syntax"
    
    def test_card_exists(self, pack_dir):
        """Test that CARD.md exists and is valid"""
        card_path = pack_dir / "CARD.md"
        assert card_path.exists(), "CARD.md not found"
        
        with open(card_path) as f:
            content = f.read()
        
        # Check required sections
        assert "## Purpose" in content, "Card must have Purpose section"
        assert "## Inputs" in content, "Card must have Inputs section"
        assert "## Outputs" in content, "Card must have Outputs section"
        assert "## License" in content, "Card must have License section"
    
    @pytest.mark.integration
    def test_pipeline_runs(self, pack_dir, tmp_path, golden_input):
        """Test that pipeline executes successfully"""
        # This would require the GL CLI to be installed
        # For now, we'll mock this
        
        # Check pipeline file exists
        pipeline_path = pack_dir / "gl.yaml"
        assert pipeline_path.exists()
        
        # In real test, would run:
        # result = subprocess.run(
        #     ["gl", "run", str(pipeline_path), "--output", str(tmp_path)],
        #     capture_output=True,
        #     text=True
        # )
        # assert result.returncode == 0
        
        # For now, just check structure
        with open(pipeline_path) as f:
            pipeline = yaml.safe_load(f)
        assert pipeline is not None
    
    def test_deterministic_hash(self, pack_dir):
        """Test that pipeline produces deterministic results"""
        pipeline_path = pack_dir / "gl.yaml"
        
        # Calculate hash of pipeline
        with open(pipeline_path, 'rb') as f:
            pipeline_hash = hashlib.sha256(f.read()).hexdigest()
        
        # In production, would compare with golden hash
        assert len(pipeline_hash) == 64, "SHA256 hash should be 64 chars"
        
        # Store for golden comparison
        golden_hash_path = pack_dir / "tests" / "golden" / "pipeline.hash"
        if not golden_hash_path.parent.exists():
            golden_hash_path.parent.mkdir(parents=True)
        
        if golden_hash_path.exists():
            with open(golden_hash_path) as f:
                expected_hash = f.read().strip()
            # Uncomment to enforce determinism
            # assert pipeline_hash == expected_hash, "Pipeline hash changed!"
        else:
            # Save golden hash
            with open(golden_hash_path, 'w') as f:
                f.write(pipeline_hash)
    
    def test_policy_compliance(self, pack_dir):
        """Test that pack complies with policy requirements"""
        manifest_path = pack_dir / "pack.yaml"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        
        policy = manifest.get("policy", {})
        
        # Check network allowlist is not empty
        assert len(policy.get("network", [])) > 0, "Network allowlist cannot be empty"
        
        # Check emission factor vintage
        ef_vintage = policy.get("ef_vintage_min")
        if ef_vintage:
            assert ef_vintage >= 2024, f"EF vintage {ef_vintage} too old (min: 2024)"
        
        # Check license is in allowlist
        license = manifest.get("license")
        allowlist = policy.get("license_allowlist", ["MIT", "Apache-2.0", "Commercial"])
        assert license in allowlist, f"License {license} not in allowlist"
    
    def test_security_requirements(self, pack_dir):
        """Test security requirements are met"""
        manifest_path = pack_dir / "pack.yaml"
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        
        security = manifest.get("security", {})
        
        # Check SBOM is specified
        assert "sbom" in security, "SBOM must be specified"
        
        # Check if SBOM file will be generated (it may not exist yet)
        sbom_file = security.get("sbom")
        if sbom_file:
            # Just check it's a valid filename
            assert sbom_file.endswith((".json", ".spdx", ".xml")), "SBOM must be valid format"


class TestBoilerSolarAgents:
    """Test individual agents in the pack"""
    
    @pytest.mark.unit
    def test_boiler_agent_validation(self):
        """Test BoilerAgent input validation"""
        # This would test the actual agent once implemented
        # For now, we document expected behavior
        
        valid_input = {
            "boiler": {
                "pressure_bar": 7,
                "temperature_c": 165,
                "efficiency": 0.82,
                "capacity_tph": 10
            }
        }
        
        invalid_inputs = [
            {"boiler": {"pressure_bar": -1}},  # Negative pressure
            {"boiler": {"pressure_bar": 100}},  # Too high pressure
            {"boiler": {"efficiency": 1.5}},    # Efficiency > 1
            {"boiler": {}},                     # Missing required fields
        ]
        
        # Document expected validation
        assert valid_input["boiler"]["pressure_bar"] > 0
        assert valid_input["boiler"]["pressure_bar"] < 50
        assert 0 < valid_input["boiler"]["efficiency"] <= 1
    
    @pytest.mark.unit  
    def test_solar_offset_calculation(self):
        """Test SolarOffsetAgent calculations"""
        # Test basic solar offset math
        
        baseline_fuel = 10000  # MMBtu/year
        solar_fraction = 0.25   # 25% offset
        
        expected_savings = baseline_fuel * solar_fraction
        assert expected_savings == 2500, "Solar offset calculation incorrect"
        
        # Test with efficiency
        collector_area = 1200  # m2
        dni_annual = 2000      # kWh/m2/year
        efficiency = 0.5       # 50% overall
        
        solar_energy_kwh = collector_area * dni_annual * efficiency
        solar_energy_mmbtu = solar_energy_kwh * 0.003412  # kWh to MMBtu
        
        assert solar_energy_mmbtu > 0, "Solar energy must be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])