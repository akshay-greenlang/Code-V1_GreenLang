from pathlib import Path
from types import SimpleNamespace

import yaml

from cbam_pack.models import CBAMConfig, MethodType
from cbam_pack.policy.engine import PolicyConfig, PolicyEngine, PolicyStatus
from cbam_pack.validators.input_validator import InputValidator


def _sample_config() -> CBAMConfig:
    config_path = Path(__file__).resolve().parents[1] / "examples" / "sample_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return CBAMConfig.model_validate(data)


def test_validator_rejects_missing_required_column(tmp_path: Path) -> None:
    imports_path = tmp_path / "imports_missing_cn_code.csv"
    imports_path.write_text(
        "line_id,quarter,year,product_description,country_of_origin,quantity,unit\n"
        "L1,Q1,2025,Steel,CN,100,tonnes\n",
        encoding="utf-8",
    )

    result = InputValidator(fail_fast=False).validate_imports(imports_path)
    assert result.is_valid is False
    assert any("cn_code" in err.message for err in result.errors)


def test_policy_engine_can_block_export_on_fail() -> None:
    config = _sample_config()
    line_results = [
        SimpleNamespace(line_id=f"L{i}", method_direct=MethodType.DEFAULT, method_indirect=MethodType.DEFAULT)
        for i in range(1, 8)
    ] + [
        SimpleNamespace(line_id=f"L{i}", method_direct=MethodType.SUPPLIER_SPECIFIC, method_indirect=MethodType.SUPPLIER_SPECIFIC)
        for i in range(8, 11)
    ]
    calc_result = SimpleNamespace(
        statistics={
            "default_usage_percent": 70.0,
            "total_lines": 10,
            "lines_with_supplier_direct_data": 3,
            "lines_with_supplier_indirect_data": 3,
        },
        line_results=line_results,
    )

    allow_export = PolicyEngine(PolicyConfig(block_export_on_fail=False)).evaluate(calc_result, config)
    block_export = PolicyEngine(PolicyConfig(block_export_on_fail=True)).evaluate(calc_result, config)

    assert allow_export.status == PolicyStatus.FAIL
    assert allow_export.can_export is True
    assert block_export.status == PolicyStatus.FAIL
    assert block_export.can_export is False


def test_policy_engine_reads_policy_from_yaml_config() -> None:
    config = _sample_config().model_dump()
    config["policy"] = {
        "default_usage_cap": 15.0,
        "default_usage_warn_threshold": 10.0,
        "authorization_threshold_tonnes": 42.0,
        "block_export_on_fail": True,
    }
    validated = CBAMConfig.model_validate(config)
    engine = PolicyEngine.from_yaml_config(validated)

    assert engine.config.default_usage_cap_q3_2024_plus == 15.0
    assert engine.config.default_usage_warn_threshold == 10.0
    assert engine.config.authorization_threshold_tonnes == 42.0
    assert engine.config.block_export_on_fail is True
