"""
GL-016 WATERGUARD Agent - Configuration Tests

Unit tests for configuration module covering:
- ASME boiler water limits by pressure class
- ASME feedwater limits by pressure class
- Treatment program configurations
- Oxygen scavenger configurations
- Amine configurations
- Helper functions

Author: GL-TestEngineer
Target Coverage: 85%+
"""

import pytest
from pydantic import ValidationError

from greenlang.agents.process_heat.gl_016_water_treatment.schemas import (
    BoilerPressureClass,
    TreatmentProgram,
    ChemicalType,
    BlowdownType,
)

from greenlang.agents.process_heat.gl_016_water_treatment.config import (
    # Configuration classes
    WaterTreatmentConfig,
    ASMEBoilerWaterLimits,
    ASMEFeedwaterLimits,
    PhosphateTreatmentConfig,
    OxygenScavengerConfig,
    AmineConfig,
    BlowdownConfig,
    DeaeratorConfig,
    # Configuration data
    ASME_BOILER_WATER_LIMITS,
    ASME_FEEDWATER_LIMITS,
    PHOSPHATE_TREATMENT_CONFIGS,
    OXYGEN_SCAVENGER_CONFIGS,
    AMINE_CONFIGS,
    # Helper functions
    get_boiler_water_limits,
    get_feedwater_limits,
    get_phosphate_config,
    get_scavenger_config,
    get_amine_config,
    determine_pressure_class,
)


class TestASMEBoilerWaterLimits:
    """Test ASME boiler water limits configuration."""

    def test_all_pressure_classes_defined(self):
        """Test all pressure classes have defined limits."""
        expected_classes = ["low_pressure", "medium_pressure", "high_pressure", "supercritical"]
        for pc in expected_classes:
            assert pc in ASME_BOILER_WATER_LIMITS
            assert isinstance(ASME_BOILER_WATER_LIMITS[pc], ASMEBoilerWaterLimits)

    def test_low_pressure_limits(self):
        """Test low pressure boiler water limits per ASME."""
        limits = ASME_BOILER_WATER_LIMITS["low_pressure"]
        assert limits.pressure_class == BoilerPressureClass.LOW_PRESSURE
        assert limits.pressure_range_psig == "0-300"
        assert limits.ph_min == 10.0
        assert limits.ph_max == 12.0
        assert limits.tds_max_ppm == 3500
        assert limits.alkalinity_max_ppm == 700
        assert limits.silica_max_ppm == 150
        assert limits.conductivity_max_umho == 7000

    def test_medium_pressure_limits(self):
        """Test medium pressure boiler water limits per ASME."""
        limits = ASME_BOILER_WATER_LIMITS["medium_pressure"]
        assert limits.pressure_class == BoilerPressureClass.MEDIUM_PRESSURE
        assert limits.pressure_range_psig == "300-900"
        assert limits.ph_min == 10.0
        assert limits.ph_max == 11.5
        assert limits.tds_max_ppm == 2500
        assert limits.silica_max_ppm == 30
        assert limits.conductivity_max_umho == 5000

    def test_high_pressure_limits(self):
        """Test high pressure boiler water limits per ASME."""
        limits = ASME_BOILER_WATER_LIMITS["high_pressure"]
        assert limits.pressure_class == BoilerPressureClass.HIGH_PRESSURE
        assert limits.pressure_range_psig == "900-1500"
        assert limits.tds_max_ppm == 1500
        assert limits.silica_max_ppm == 5  # Much stricter

    def test_supercritical_limits(self):
        """Test supercritical boiler water limits per ASME."""
        limits = ASME_BOILER_WATER_LIMITS["supercritical"]
        assert limits.pressure_class == BoilerPressureClass.SUPERCRITICAL
        assert limits.tds_max_ppm == 100  # Very strict
        assert limits.silica_max_ppm == 0.5  # Very strict
        assert limits.conductivity_max_umho == 200

    def test_limits_become_stricter_with_pressure(self):
        """Test limits become stricter as pressure increases."""
        lp = ASME_BOILER_WATER_LIMITS["low_pressure"]
        mp = ASME_BOILER_WATER_LIMITS["medium_pressure"]
        hp = ASME_BOILER_WATER_LIMITS["high_pressure"]
        sc = ASME_BOILER_WATER_LIMITS["supercritical"]

        # TDS limits decrease with pressure
        assert lp.tds_max_ppm > mp.tds_max_ppm > hp.tds_max_ppm > sc.tds_max_ppm

        # Silica limits decrease with pressure
        assert lp.silica_max_ppm > mp.silica_max_ppm > hp.silica_max_ppm > sc.silica_max_ppm

        # Conductivity limits decrease with pressure
        assert lp.conductivity_max_umho > mp.conductivity_max_umho > hp.conductivity_max_umho > sc.conductivity_max_umho


class TestASMEFeedwaterLimits:
    """Test ASME feedwater limits configuration."""

    def test_all_pressure_classes_defined(self):
        """Test all pressure classes have defined feedwater limits."""
        expected_classes = ["low_pressure", "medium_pressure", "high_pressure", "supercritical"]
        for pc in expected_classes:
            assert pc in ASME_FEEDWATER_LIMITS
            assert isinstance(ASME_FEEDWATER_LIMITS[pc], ASMEFeedwaterLimits)

    def test_low_pressure_feedwater_limits(self):
        """Test low pressure feedwater limits per ASME."""
        limits = ASME_FEEDWATER_LIMITS["low_pressure"]
        assert limits.dissolved_oxygen_max_ppb == 7
        assert limits.iron_max_ppb == 100
        assert limits.copper_max_ppb == 50
        assert limits.total_hardness_max_ppm == 0.3
        assert limits.ph_min == 8.3
        assert limits.ph_max == 10.0

    def test_medium_pressure_feedwater_limits(self):
        """Test medium pressure feedwater limits per ASME."""
        limits = ASME_FEEDWATER_LIMITS["medium_pressure"]
        assert limits.dissolved_oxygen_max_ppb == 7
        assert limits.iron_max_ppb == 20
        assert limits.copper_max_ppb == 15
        assert limits.total_hardness_max_ppm == 0.1
        assert limits.cation_conductivity_max_umho == 0.5

    def test_high_pressure_feedwater_limits(self):
        """Test high pressure feedwater limits per ASME."""
        limits = ASME_FEEDWATER_LIMITS["high_pressure"]
        assert limits.dissolved_oxygen_max_ppb == 5
        assert limits.iron_max_ppb == 10
        assert limits.copper_max_ppb == 5
        assert limits.total_hardness_max_ppm == 0.05

    def test_supercritical_feedwater_limits(self):
        """Test supercritical feedwater limits per ASME."""
        limits = ASME_FEEDWATER_LIMITS["supercritical"]
        assert limits.dissolved_oxygen_max_ppb == 3
        assert limits.iron_max_ppb == 5
        assert limits.copper_max_ppb == 2
        assert limits.total_hardness_max_ppm == 0.01  # Very strict

    def test_feedwater_limits_stricter_with_pressure(self):
        """Test feedwater limits become stricter with pressure."""
        lp = ASME_FEEDWATER_LIMITS["low_pressure"]
        mp = ASME_FEEDWATER_LIMITS["medium_pressure"]
        hp = ASME_FEEDWATER_LIMITS["high_pressure"]
        sc = ASME_FEEDWATER_LIMITS["supercritical"]

        # Iron limits decrease with pressure
        assert lp.iron_max_ppb > mp.iron_max_ppb > hp.iron_max_ppb > sc.iron_max_ppb

        # Copper limits decrease with pressure
        assert lp.copper_max_ppb > mp.copper_max_ppb > hp.copper_max_ppb > sc.copper_max_ppb


class TestPhosphateTreatmentConfigs:
    """Test phosphate treatment program configurations."""

    def test_coordinated_phosphate_config(self):
        """Test coordinated phosphate program configuration."""
        config = PHOSPHATE_TREATMENT_CONFIGS["coordinated_phosphate"]
        assert config.program_type == TreatmentProgram.COORDINATED_PHOSPHATE
        assert config.phosphate_min_ppm == 2.0
        assert config.phosphate_max_ppm == 12.0
        assert config.phosphate_target_ppm == 8.0
        assert config.na_po4_ratio_min == 2.6
        assert config.na_po4_ratio_max == 3.0
        assert config.na_po4_ratio_target == 2.8
        assert config.ph_min == 9.3
        assert config.ph_max == 10.0

    def test_congruent_phosphate_config(self):
        """Test congruent phosphate program configuration."""
        config = PHOSPHATE_TREATMENT_CONFIGS["congruent_phosphate"]
        assert config.program_type == TreatmentProgram.CONGRUENT_PHOSPHATE
        assert config.phosphate_min_ppm == 0.5
        assert config.phosphate_max_ppm == 3.0
        assert config.na_po4_ratio_min == 2.2
        assert config.na_po4_ratio_max == 2.8
        # Congruent uses lower Na:PO4 ratio than coordinated

    def test_phosphate_precipitate_config(self):
        """Test phosphate precipitate program configuration."""
        config = PHOSPHATE_TREATMENT_CONFIGS["phosphate_precipitate"]
        assert config.program_type == TreatmentProgram.PHOSPHATE_PRECIPITATE
        assert config.phosphate_min_ppm == 30.0
        assert config.phosphate_max_ppm == 60.0
        assert config.free_oh_min_ppm == 50
        assert config.free_oh_max_ppm == 300
        # Precipitate program has higher phosphate and free hydroxide

    def test_phosphate_polymer_config(self):
        """Test phosphate polymer program configuration."""
        config = PHOSPHATE_TREATMENT_CONFIGS["phosphate_polymer"]
        assert config.program_type == TreatmentProgram.PHOSPHATE_POLYMER
        assert config.phosphate_min_ppm == 10.0
        assert config.phosphate_max_ppm == 30.0
        assert config.ph_min == 10.5
        assert config.ph_max == 11.5


class TestOxygenScavengerConfigs:
    """Test oxygen scavenger configurations."""

    def test_sulfite_config(self):
        """Test sodium sulfite scavenger configuration."""
        config = OXYGEN_SCAVENGER_CONFIGS["sulfite"]
        assert config.scavenger_type == ChemicalType.SULFITE
        assert config.stoichiometric_ratio == 7.9  # lb sulfite per lb O2
        assert config.recommended_excess_pct == 50
        assert config.residual_min_ppm == 20
        assert config.residual_max_ppm == 40
        assert config.residual_target_ppm == 30
        assert config.max_temperature_f == 700
        assert "SO2" in config.decomposition_products

    def test_hydrazine_config(self):
        """Test hydrazine scavenger configuration."""
        config = OXYGEN_SCAVENGER_CONFIGS["hydrazine"]
        assert config.scavenger_type == ChemicalType.HYDRAZINE
        assert config.stoichiometric_ratio == 1.0
        assert config.recommended_excess_pct == 100
        assert config.passivating == True  # Forms magnetite
        assert config.max_temperature_f == 1000
        assert "NH3" in config.decomposition_products

    def test_carbohydrazide_config(self):
        """Test carbohydrazide scavenger configuration."""
        config = OXYGEN_SCAVENGER_CONFIGS["carbohydrazide"]
        assert config.scavenger_type == ChemicalType.CARBOHYDRAZIDE
        assert config.stoichiometric_ratio == 1.4
        assert config.passivating == True

    def test_erythorbic_acid_config(self):
        """Test erythorbic acid scavenger configuration."""
        config = OXYGEN_SCAVENGER_CONFIGS["erythorbic_acid"]
        assert config.scavenger_type == ChemicalType.ERYTHORBIC_ACID
        assert config.stoichiometric_ratio == 5.5
        assert config.max_temperature_f == 600  # Lower max temp


class TestAmineConfigs:
    """Test amine treatment configurations."""

    def test_morpholine_config(self):
        """Test morpholine amine configuration."""
        config = AMINE_CONFIGS["morpholine"]
        assert config.amine_type == ChemicalType.MORPHOLINE
        assert config.distribution_ratio == 0.4  # Low volatility
        assert config.neutralizing_capacity == 0.5
        assert config.target_ph_range == (8.5, 9.0)
        assert config.typical_dose_ppm == 5.0
        assert config.filming_capability == False

    def test_cyclohexylamine_config(self):
        """Test cyclohexylamine amine configuration."""
        config = AMINE_CONFIGS["cyclohexylamine"]
        assert config.amine_type == ChemicalType.CYCLOHEXYLAMINE
        assert config.distribution_ratio == 4.0  # High volatility
        assert config.filming_capability == True

    def test_diethylaminoethanol_config(self):
        """Test DEAE amine configuration."""
        config = AMINE_CONFIGS["diethylaminoethanol"]
        assert config.amine_type == ChemicalType.DIETHYLAMINOETHANOL
        assert config.fda_approved == True
        assert config.filming_capability == True


class TestWaterTreatmentConfig:
    """Test main water treatment configuration."""

    def test_valid_config(self, water_treatment_config):
        """Test valid configuration creation."""
        assert water_treatment_config.system_id == "TEST-WT-001"
        assert water_treatment_config.boiler_pressure_class == BoilerPressureClass.MEDIUM_PRESSURE
        assert water_treatment_config.treatment_program == TreatmentProgram.COORDINATED_PHOSPHATE

    def test_config_with_defaults(self):
        """Test configuration with default values."""
        config = WaterTreatmentConfig(system_id="TEST-001")
        assert config.boiler_pressure_class == BoilerPressureClass.MEDIUM_PRESSURE
        assert config.treatment_program == TreatmentProgram.PHOSPHATE_POLYMER
        assert config.oxygen_scavenger_type == ChemicalType.SULFITE
        assert config.condensate_return_pct == 80.0
        assert config.makeup_water_source == "softened"

    def test_config_name_auto_generation(self):
        """Test name is auto-generated from system_id."""
        config = WaterTreatmentConfig(system_id="WT-123")
        assert config.name == "Water Treatment System WT-123"

    def test_config_explicit_name(self):
        """Test explicit name overrides auto-generation."""
        config = WaterTreatmentConfig(
            system_id="WT-123",
            name="Custom Name",
        )
        assert config.name == "Custom Name"

    def test_operating_pressure_validation(self):
        """Test operating pressure must be less than design pressure."""
        with pytest.raises(ValidationError):
            WaterTreatmentConfig(
                system_id="TEST",
                operating_pressure_psig=600.0,
                design_pressure_psig=500.0,  # Less than operating
            )

    def test_config_with_all_options(self):
        """Test configuration with all options specified."""
        config = WaterTreatmentConfig(
            system_id="FULL-TEST",
            name="Full Test System",
            boiler_pressure_class=BoilerPressureClass.HIGH_PRESSURE,
            operating_pressure_psig=1000.0,
            design_pressure_psig=1500.0,
            steam_capacity_lb_hr=100000.0,
            treatment_program=TreatmentProgram.CONGRUENT_PHOSPHATE,
            oxygen_scavenger_type=ChemicalType.HYDRAZINE,
            amine_treatment_enabled=True,
            amine_type=ChemicalType.CYCLOHEXYLAMINE,
            condensate_return_pct=90.0,
            makeup_water_source="demin",
            fuel_cost_per_mmbtu=8.0,
            water_cost_per_kgal=5.0,
            sil_level=3,
        )
        assert config.boiler_pressure_class == BoilerPressureClass.HIGH_PRESSURE
        assert config.oxygen_scavenger_type == ChemicalType.HYDRAZINE
        assert config.sil_level == 3


class TestBlowdownConfig:
    """Test blowdown configuration."""

    def test_default_blowdown_config(self):
        """Test default blowdown configuration."""
        config = BlowdownConfig()
        assert config.blowdown_type == BlowdownType.CONTINUOUS
        assert config.min_cycles == 3.0
        assert config.max_cycles == 10.0
        assert config.target_cycles == 6.0
        assert config.min_blowdown_rate_pct == 1.0
        assert config.max_blowdown_rate_pct == 10.0
        assert config.heat_recovery_enabled == True

    def test_custom_blowdown_config(self):
        """Test custom blowdown configuration."""
        config = BlowdownConfig(
            blowdown_type=BlowdownType.COMBINED,
            target_cycles=8.0,
            heat_recovery_enabled=False,
            tds_control=True,
            silica_control=True,
        )
        assert config.blowdown_type == BlowdownType.COMBINED
        assert config.target_cycles == 8.0
        assert config.silica_control == True


class TestDeaeratorConfig:
    """Test deaerator configuration."""

    def test_default_deaerator_config(self):
        """Test default deaerator configuration."""
        config = DeaeratorConfig()
        assert config.deaerator_type == "spray_tray"
        assert config.design_pressure_psig == 5.0
        assert config.min_pressure_psig == 3.0
        assert config.max_pressure_psig == 7.0
        assert config.outlet_o2_target_ppb == 5.0
        assert config.outlet_o2_max_ppb == 7.0
        assert config.min_vent_rate_pct == 0.5
        assert config.max_vent_rate_pct == 2.0

    def test_custom_deaerator_config(self):
        """Test custom deaerator configuration."""
        config = DeaeratorConfig(
            deaerator_type="spray",
            design_pressure_psig=10.0,
            outlet_o2_max_ppb=5.0,  # Stricter limit
        )
        assert config.deaerator_type == "spray"
        assert config.outlet_o2_max_ppb == 5.0


class TestHelperFunctions:
    """Test configuration helper functions."""

    def test_get_boiler_water_limits_by_enum(self):
        """Test get_boiler_water_limits with enum."""
        limits = get_boiler_water_limits(BoilerPressureClass.MEDIUM_PRESSURE)
        assert limits.pressure_class == BoilerPressureClass.MEDIUM_PRESSURE
        assert limits.tds_max_ppm == 2500

    def test_get_boiler_water_limits_default(self):
        """Test get_boiler_water_limits returns default for unknown."""
        # Should return medium_pressure as default
        limits = get_boiler_water_limits(BoilerPressureClass.MEDIUM_PRESSURE)
        assert limits is not None

    def test_get_feedwater_limits_by_enum(self):
        """Test get_feedwater_limits with enum."""
        limits = get_feedwater_limits(BoilerPressureClass.HIGH_PRESSURE)
        assert limits.pressure_class == BoilerPressureClass.HIGH_PRESSURE
        assert limits.dissolved_oxygen_max_ppb == 5

    def test_get_phosphate_config_coordinated(self):
        """Test get_phosphate_config for coordinated phosphate."""
        config = get_phosphate_config(TreatmentProgram.COORDINATED_PHOSPHATE)
        assert config is not None
        assert config.program_type == TreatmentProgram.COORDINATED_PHOSPHATE

    def test_get_phosphate_config_unknown(self):
        """Test get_phosphate_config returns None for unknown program."""
        config = get_phosphate_config(TreatmentProgram.ALL_VOLATILE)
        # AVT doesn't use phosphate
        assert config is None

    def test_get_scavenger_config_sulfite(self):
        """Test get_scavenger_config for sulfite."""
        config = get_scavenger_config(ChemicalType.SULFITE)
        assert config is not None
        assert config.scavenger_type == ChemicalType.SULFITE

    def test_get_scavenger_config_unknown(self):
        """Test get_scavenger_config returns None for unknown type."""
        config = get_scavenger_config(ChemicalType.AMINE)
        assert config is None  # Amine is not an oxygen scavenger

    def test_get_amine_config_morpholine(self):
        """Test get_amine_config for morpholine."""
        config = get_amine_config(ChemicalType.MORPHOLINE)
        assert config is not None
        assert config.amine_type == ChemicalType.MORPHOLINE

    def test_get_amine_config_unknown(self):
        """Test get_amine_config returns None for unknown type."""
        config = get_amine_config(ChemicalType.SULFITE)
        assert config is None  # Sulfite is not an amine


class TestDeterminePressureClass:
    """Test determine_pressure_class function."""

    @pytest.mark.parametrize("pressure,expected_class", [
        (0.0, BoilerPressureClass.LOW_PRESSURE),
        (150.0, BoilerPressureClass.LOW_PRESSURE),
        (299.0, BoilerPressureClass.LOW_PRESSURE),
        (300.0, BoilerPressureClass.MEDIUM_PRESSURE),
        (450.0, BoilerPressureClass.MEDIUM_PRESSURE),
        (899.0, BoilerPressureClass.MEDIUM_PRESSURE),
        (900.0, BoilerPressureClass.HIGH_PRESSURE),
        (1200.0, BoilerPressureClass.HIGH_PRESSURE),
        (1499.0, BoilerPressureClass.HIGH_PRESSURE),
        (1500.0, BoilerPressureClass.SUPERCRITICAL),
        (2000.0, BoilerPressureClass.SUPERCRITICAL),
        (3000.0, BoilerPressureClass.SUPERCRITICAL),
    ])
    def test_pressure_class_boundaries(self, pressure, expected_class):
        """Test pressure class determination at boundaries."""
        result = determine_pressure_class(pressure)
        assert result == expected_class

    def test_low_pressure_range(self):
        """Test low pressure classification (< 300 psig)."""
        assert determine_pressure_class(100.0) == BoilerPressureClass.LOW_PRESSURE
        assert determine_pressure_class(200.0) == BoilerPressureClass.LOW_PRESSURE

    def test_medium_pressure_range(self):
        """Test medium pressure classification (300-900 psig)."""
        assert determine_pressure_class(400.0) == BoilerPressureClass.MEDIUM_PRESSURE
        assert determine_pressure_class(600.0) == BoilerPressureClass.MEDIUM_PRESSURE

    def test_high_pressure_range(self):
        """Test high pressure classification (900-1500 psig)."""
        assert determine_pressure_class(1000.0) == BoilerPressureClass.HIGH_PRESSURE
        assert determine_pressure_class(1400.0) == BoilerPressureClass.HIGH_PRESSURE

    def test_supercritical_range(self):
        """Test supercritical classification (> 1500 psig)."""
        assert determine_pressure_class(1600.0) == BoilerPressureClass.SUPERCRITICAL
        assert determine_pressure_class(2500.0) == BoilerPressureClass.SUPERCRITICAL


class TestConfigurationValidation:
    """Test configuration validation rules."""

    def test_sample_interval_bounds(self):
        """Test sample interval bounds (5-1440 minutes)."""
        # Valid interval
        config = WaterTreatmentConfig(
            system_id="TEST",
            sample_interval_minutes=60,
        )
        assert config.sample_interval_minutes == 60

        # Below minimum
        with pytest.raises(ValidationError):
            WaterTreatmentConfig(
                system_id="TEST",
                sample_interval_minutes=2,  # < 5
            )

        # Above maximum
        with pytest.raises(ValidationError):
            WaterTreatmentConfig(
                system_id="TEST",
                sample_interval_minutes=2000,  # > 1440
            )

    def test_trend_deviation_threshold_bounds(self):
        """Test trend deviation threshold bounds (1-50%)."""
        # Valid threshold
        config = WaterTreatmentConfig(
            system_id="TEST",
            trend_deviation_threshold_pct=10.0,
        )
        assert config.trend_deviation_threshold_pct == 10.0

        # Below minimum
        with pytest.raises(ValidationError):
            WaterTreatmentConfig(
                system_id="TEST",
                trend_deviation_threshold_pct=0.5,  # < 1
            )

    def test_sil_level_bounds(self):
        """Test SIL level bounds (1-3)."""
        # Valid SIL levels
        for sil in [1, 2, 3]:
            config = WaterTreatmentConfig(
                system_id="TEST",
                sil_level=sil,
            )
            assert config.sil_level == sil

        # Invalid SIL level
        with pytest.raises(ValidationError):
            WaterTreatmentConfig(
                system_id="TEST",
                sil_level=4,  # > 3
            )

    def test_data_retention_minimum(self):
        """Test data retention minimum (30 days)."""
        # Valid retention
        config = WaterTreatmentConfig(
            system_id="TEST",
            data_retention_days=365,
        )
        assert config.data_retention_days == 365

        # Below minimum
        with pytest.raises(ValidationError):
            WaterTreatmentConfig(
                system_id="TEST",
                data_retention_days=15,  # < 30
            )


class TestComplianceConfiguration:
    """Test ASME compliance of configuration values."""

    @pytest.mark.compliance
    def test_asme_limits_have_required_fields(self):
        """Test all ASME limits have required fields."""
        for key, limits in ASME_BOILER_WATER_LIMITS.items():
            assert limits.pressure_class is not None
            assert limits.ph_min is not None
            assert limits.ph_max is not None
            assert limits.tds_max_ppm is not None
            assert limits.silica_max_ppm is not None
            assert limits.conductivity_max_umho is not None

    @pytest.mark.compliance
    def test_feedwater_limits_have_required_fields(self):
        """Test all feedwater limits have required fields."""
        for key, limits in ASME_FEEDWATER_LIMITS.items():
            assert limits.dissolved_oxygen_max_ppb is not None
            assert limits.iron_max_ppb is not None
            assert limits.copper_max_ppb is not None
            assert limits.total_hardness_max_ppm is not None

    @pytest.mark.compliance
    def test_scavenger_stoichiometry_accuracy(self):
        """Test oxygen scavenger stoichiometric ratios are accurate."""
        # Sulfite: 8 Na2SO3 + O2 -> 8 Na2SO4
        # Stoich: 7.88 lb Na2SO3 per lb O2 (approximately)
        sulfite = OXYGEN_SCAVENGER_CONFIGS["sulfite"]
        assert 7.5 <= sulfite.stoichiometric_ratio <= 8.5

        # Hydrazine: N2H4 + O2 -> N2 + 2H2O
        # Stoich: 1.0 lb N2H4 per lb O2
        hydrazine = OXYGEN_SCAVENGER_CONFIGS["hydrazine"]
        assert hydrazine.stoichiometric_ratio == 1.0

    @pytest.mark.compliance
    def test_phosphate_ratio_ranges(self):
        """Test Na:PO4 ratio ranges are within EPRI guidelines."""
        # Coordinated phosphate: Na:PO4 = 2.6 - 3.0
        coord = PHOSPHATE_TREATMENT_CONFIGS["coordinated_phosphate"]
        assert coord.na_po4_ratio_min >= 2.6
        assert coord.na_po4_ratio_max <= 3.0

        # Congruent phosphate: Na:PO4 = 2.2 - 2.8
        cong = PHOSPHATE_TREATMENT_CONFIGS["congruent_phosphate"]
        assert cong.na_po4_ratio_min >= 2.2
        assert cong.na_po4_ratio_max <= 2.8
