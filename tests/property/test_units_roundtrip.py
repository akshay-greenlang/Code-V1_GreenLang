"""Property-based tests for unit conversions and round-trips."""

from hypothesis import given, strategies as st, assume
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from test_utils import assert_close, normalize_factor


class TestUnitRoundTrips:
    """Test unit conversion round-trip properties."""
    
    @given(
        kwh_value=st.floats(min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False)
    )
    def test_kwh_mwh_roundtrip(self, kwh_value):
        """Test kWh → MWh → kWh round-trip conversion."""
        # Convert kWh to MWh
        mwh_value = normalize_factor(kwh_value, "kWh", "MWh")
        
        # Convert back to kWh
        kwh_roundtrip = normalize_factor(mwh_value, "MWh", "kWh")
        
        # Should match original within floating point precision
        assert_close(kwh_value, kwh_roundtrip, rel_tol=1e-12)
    
    @given(
        therms_value=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False)
    )
    def test_therms_mmbtu_roundtrip(self, therms_value):
        """Test therms → MMBtu → therms round-trip conversion."""
        # Convert therms to MMBtu
        mmbtu_value = normalize_factor(therms_value, "therms", "MMBtu")
        
        # Convert back to therms
        therms_roundtrip = normalize_factor(mmbtu_value, "MMBtu", "therms")
        
        # Should match original
        assert_close(therms_value, therms_roundtrip, rel_tol=1e-12)
    
    @given(
        sqft_value=st.floats(min_value=1.0, max_value=1e8, allow_nan=False, allow_infinity=False)
    )
    def test_sqft_sqm_roundtrip(self, sqft_value):
        """Test sqft → sqm → sqft round-trip conversion."""
        # Convert sqft to sqm
        sqm_value = normalize_factor(sqft_value, "sqft", "sqm")
        
        # Convert back to sqft
        sqft_roundtrip = normalize_factor(sqm_value, "sqm", "sqft")
        
        # Should match original within tolerance
        assert_close(sqft_value, sqft_roundtrip, rel_tol=1e-10)
    
    @given(
        m3_value=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False)
    )
    def test_m3_ft3_roundtrip(self, m3_value):
        """Test m³ → ft³ → m³ round-trip conversion."""
        # Convert m³ to ft³
        ft3_value = normalize_factor(m3_value, "m3", "ft3")
        
        # Convert back to m³
        m3_roundtrip = normalize_factor(ft3_value, "ft3", "m3")
        
        # Should match original
        assert_close(m3_value, m3_roundtrip, rel_tol=1e-10)
    
    @given(
        value=st.floats(min_value=0.001, max_value=1e9, allow_nan=False, allow_infinity=False),
        unit=st.sampled_from(["kWh", "therms", "sqft", "m3"])
    )
    def test_identity_conversion(self, value, unit):
        """Test that converting a unit to itself returns the same value."""
        converted = normalize_factor(value, unit, unit)
        assert converted == value
    
    @given(
        value1=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        value2=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False)
    )
    def test_conversion_preserves_ratios(self, value1, value2):
        """Test that unit conversions preserve ratios between values."""
        assume(value1 != value2)
        
        # Original ratio
        original_ratio = value1 / value2
        
        # Convert both values
        converted1 = normalize_factor(value1, "kWh", "MWh")
        converted2 = normalize_factor(value2, "kWh", "MWh")
        
        # Ratio should be preserved
        converted_ratio = converted1 / converted2
        
        assert_close(original_ratio, converted_ratio, rel_tol=1e-12)
    
    @given(
        values=st.lists(
            st.floats(min_value=1.0, max_value=1e5, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=10
        )
    )
    def test_conversion_preserves_sum(self, values):
        """Test that sum is preserved through unit conversion."""
        # Sum in original units
        original_sum = sum(values)
        
        # Convert sum
        converted_sum = normalize_factor(original_sum, "kWh", "MWh")
        
        # Convert individual values and sum
        converted_values = [normalize_factor(v, "kWh", "MWh") for v in values]
        sum_of_converted = sum(converted_values)
        
        # Both approaches should yield same result
        assert_close(converted_sum, sum_of_converted, rel_tol=1e-10)
    
    @given(
        base=st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        exponent=st.integers(min_value=0, max_value=3)
    )
    def test_conversion_scaling_laws(self, base, exponent):
        """Test that conversions follow proper scaling laws."""
        value = base ** (exponent + 1)
        
        # Linear conversions
        kwh_to_mwh = normalize_factor(value, "kWh", "MWh")
        expected = value * 0.001
        assert_close(kwh_to_mwh, expected, rel_tol=1e-12)
        
        # Area conversions (if squared units were supported)
        sqft_to_sqm = normalize_factor(value, "sqft", "sqm")
        expected_area = value * 0.092903
        assert_close(sqft_to_sqm, expected_area, rel_tol=1e-10)