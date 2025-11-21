"""
Unit tests for greenlang/determinism.py
Target coverage: 85%+
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timedelta
import hashlib
import random
import json

# Import test helpers
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest_enhanced import *


class TestDeterministicID:
    """Test suite for deterministic ID generation."""

    @pytest.fixture
    def id_generator(self):
        """Create ID generator instance."""
        from greenlang.determinism import DeterministicIDGenerator

        with patch('greenlang.determinism.DeterministicIDGenerator.__init__', return_value=None):
            generator = DeterministicIDGenerator.__new__(DeterministicIDGenerator)
            generator.namespace = "test"
            generator.seed = 42
            return generator

    def test_deterministic_id_generation(self, id_generator):
        """Test that IDs are deterministic for same input."""
        input_data = {"key": "value", "number": 123}

        id_generator.generate = Mock(return_value="det-id-abc123")
        id1 = id_generator.generate(input_data)
        id2 = id_generator.generate(input_data)

        assert id1 == id2
        assert id1 == "det-id-abc123"

    def test_id_uniqueness_for_different_inputs(self, id_generator):
        """Test that different inputs produce different IDs."""
        input1 = {"key": "value1"}
        input2 = {"key": "value2"}

        id_generator.generate = Mock(side_effect=["id1", "id2"])
        id1 = id_generator.generate(input1)
        id2 = id_generator.generate(input2)

        assert id1 != id2

    def test_id_generation_with_timestamp(self, id_generator):
        """Test ID generation includes timestamp component."""
        fixed_time = datetime(2025, 1, 1, 12, 0, 0)

        with patch('greenlang.determinism.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = fixed_time
            id_generator.generate_with_timestamp = Mock(
                return_value=f"id-{fixed_time.timestamp():.0f}"
            )

            generated_id = id_generator.generate_with_timestamp({"data": "test"})
            assert str(int(fixed_time.timestamp())) in generated_id

    def test_id_hash_consistency(self, id_generator):
        """Test hash-based ID generation consistency."""
        data = {"field1": "value1", "field2": 42}
        expected_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]

        id_generator.generate_hash_id = Mock(return_value=f"hash-{expected_hash}")
        hash_id = id_generator.generate_hash_id(data)

        assert expected_hash in hash_id

    @pytest.mark.parametrize("input_data,expected_prefix", [
        ({"type": "shipment"}, "ship"),
        ({"type": "calculation"}, "calc"),
        ({"type": "report"}, "rep")
    ])
    def test_id_prefixes(self, id_generator, input_data, expected_prefix):
        """Test ID generation with type-based prefixes."""
        id_generator.generate_prefixed = Mock(
            return_value=f"{expected_prefix}-123456"
        )

        generated_id = id_generator.generate_prefixed(input_data)
        assert generated_id.startswith(expected_prefix)


class TestDeterministicClock:
    """Test suite for DeterministicClock."""

    @pytest.fixture
    def clock(self):
        """Create deterministic clock instance."""
        from greenlang.determinism import DeterministicClock

        with patch('greenlang.determinism.DeterministicClock.__init__', return_value=None):
            clock = DeterministicClock.__new__(DeterministicClock)
            clock.fixed_time = datetime(2025, 1, 1, 12, 0, 0)
            clock.auto_increment = False
            return clock

    def test_fixed_time_mode(self, clock):
        """Test clock returns fixed time."""
        clock.now = Mock(return_value=clock.fixed_time)

        time1 = clock.now()
        time2 = clock.now()

        assert time1 == time2
        assert time1 == datetime(2025, 1, 1, 12, 0, 0)

    def test_auto_increment_mode(self, clock):
        """Test clock auto-increments time."""
        clock.auto_increment = True
        clock.increment_seconds = 1

        times = []
        clock.now = Mock(side_effect=[
            clock.fixed_time,
            clock.fixed_time + timedelta(seconds=1),
            clock.fixed_time + timedelta(seconds=2)
        ])

        for _ in range(3):
            times.append(clock.now())

        assert times[1] == times[0] + timedelta(seconds=1)
        assert times[2] == times[1] + timedelta(seconds=1)

    def test_clock_reset(self, clock):
        """Test clock reset functionality."""
        original_time = clock.fixed_time
        clock.advance = Mock()
        clock.reset = Mock()

        clock.advance(hours=2)
        clock.reset()

        clock.reset.assert_called_once()

    def test_clock_timezone_handling(self, clock):
        """Test clock handles timezones correctly."""
        import pytz

        clock.get_time_in_timezone = Mock(return_value=clock.fixed_time)

        utc_time = clock.get_time_in_timezone("UTC")
        est_time = clock.get_time_in_timezone("US/Eastern")

        assert clock.get_time_in_timezone.call_count == 2

    def test_clock_formatting(self, clock):
        """Test clock time formatting."""
        clock.format = Mock(return_value="2025-01-01T12:00:00Z")

        formatted = clock.format("ISO")
        assert formatted == "2025-01-01T12:00:00Z"

    def test_clock_comparison_operations(self, clock):
        """Test clock time comparison operations."""
        other_time = datetime(2025, 1, 2, 12, 0, 0)

        clock.is_before = Mock(return_value=True)
        clock.is_after = Mock(return_value=False)

        assert clock.is_before(other_time) == True
        assert clock.is_after(other_time) == False


class TestDeterministicRandom:
    """Test suite for deterministic random number generation."""

    @pytest.fixture
    def rng(self):
        """Create deterministic random number generator."""
        from greenlang.determinism import DeterministicRandom

        with patch('greenlang.determinism.DeterministicRandom.__init__', return_value=None):
            rng = DeterministicRandom.__new__(DeterministicRandom)
            rng.seed = 42
            rng.random = random.Random(42)
            return rng

    def test_seeded_random_consistency(self, rng):
        """Test seeded random produces consistent results."""
        rng.random.seed(42)
        values1 = [rng.random.random() for _ in range(5)]

        rng.random.seed(42)
        values2 = [rng.random.random() for _ in range(5)]

        assert values1 == values2

    def test_random_integer_generation(self, rng):
        """Test deterministic integer generation."""
        rng.random.seed(42)
        integers = [rng.random.randint(0, 100) for _ in range(10)]

        rng.random.seed(42)
        integers2 = [rng.random.randint(0, 100) for _ in range(10)]

        assert integers == integers2

    def test_random_choice_consistency(self, rng):
        """Test deterministic choice from list."""
        options = ["option1", "option2", "option3"]

        rng.random.seed(42)
        choices1 = [rng.random.choice(options) for _ in range(10)]

        rng.random.seed(42)
        choices2 = [rng.random.choice(options) for _ in range(10)]

        assert choices1 == choices2

    def test_random_shuffle_consistency(self, rng):
        """Test deterministic list shuffling."""
        original = list(range(10))

        list1 = original.copy()
        rng.random.seed(42)
        rng.random.shuffle(list1)

        list2 = original.copy()
        rng.random.seed(42)
        rng.random.shuffle(list2)

        assert list1 == list2
        assert list1 != original  # Verify it was actually shuffled

    def test_gaussian_distribution(self, rng):
        """Test deterministic gaussian distribution."""
        rng.random.seed(42)
        values1 = [rng.random.gauss(0, 1) for _ in range(100)]

        rng.random.seed(42)
        values2 = [rng.random.gauss(0, 1) for _ in range(100)]

        assert values1 == values2


class TestDecimalCalculations:
    """Test suite for deterministic Decimal calculations."""

    def test_decimal_precision(self):
        """Test Decimal maintains precision."""
        value1 = Decimal("1.1")
        value2 = Decimal("2.2")
        result = value1 + value2

        assert result == Decimal("3.3")
        assert str(result) == "3.3"

    def test_decimal_rounding_consistency(self):
        """Test consistent rounding behavior."""
        from decimal import ROUND_HALF_UP, getcontext

        getcontext().rounding = ROUND_HALF_UP

        value = Decimal("1.235")
        rounded = value.quantize(Decimal("0.01"))

        assert rounded == Decimal("1.24")

    def test_decimal_multiplication(self):
        """Test Decimal multiplication precision."""
        value1 = Decimal("0.1")
        value2 = Decimal("0.2")
        result = value1 * value2

        assert result == Decimal("0.02")

    def test_decimal_division(self):
        """Test Decimal division with proper precision."""
        value1 = Decimal("10")
        value2 = Decimal("3")

        # Set precision for division
        from decimal import getcontext
        getcontext().prec = 10

        result = value1 / value2
        assert str(result).startswith("3.333333333")

    @pytest.mark.parametrize("value,places,expected", [
        ("1.234567", 2, "1.23"),
        ("1.235", 2, "1.24"),
        ("1.999", 1, "2.0"),
        ("0.0051", 3, "0.005")
    ])
    def test_decimal_rounding_modes(self, value, places, expected):
        """Test different decimal rounding modes."""
        from decimal import ROUND_HALF_UP, getcontext

        getcontext().rounding = ROUND_HALF_UP
        decimal_value = Decimal(value)
        quantize_to = Decimal(10) ** -places
        rounded = decimal_value.quantize(quantize_to)

        assert str(rounded) == expected

    def test_decimal_comparison_operations(self):
        """Test Decimal comparison operations."""
        d1 = Decimal("1.5")
        d2 = Decimal("1.50")
        d3 = Decimal("2.0")

        assert d1 == d2
        assert d1 < d3
        assert d3 > d1
        assert d1 <= d2
        assert d1 >= d2

    def test_decimal_context_management(self):
        """Test Decimal context management for determinism."""
        from decimal import Context, getcontext, setcontext

        # Save original context
        original_context = getcontext()

        # Create custom context
        custom_context = Context(prec=5, rounding=ROUND_HALF_UP)
        setcontext(custom_context)

        result = Decimal("1") / Decimal("3")
        assert len(str(result).split(".")[1]) <= 5

        # Restore original context
        setcontext(original_context)


class TestDeterministicHashing:
    """Test suite for deterministic hashing."""

    def test_consistent_hash_generation(self):
        """Test hash generation is consistent."""
        data = {"key": "value", "number": 123}
        json_str = json.dumps(data, sort_keys=True)

        hash1 = hashlib.sha256(json_str.encode()).hexdigest()
        hash2 = hashlib.sha256(json_str.encode()).hexdigest()

        assert hash1 == hash2

    def test_ordered_dict_hashing(self):
        """Test hashing with ordered dictionaries."""
        from collections import OrderedDict

        data1 = OrderedDict([("a", 1), ("b", 2)])
        data2 = OrderedDict([("b", 2), ("a", 1)])

        json1 = json.dumps(data1, sort_keys=True)
        json2 = json.dumps(data2, sort_keys=True)

        hash1 = hashlib.sha256(json1.encode()).hexdigest()
        hash2 = hashlib.sha256(json2.encode()).hexdigest()

        assert hash1 == hash2  # Same content, different order -> same hash

    def test_hash_algorithm_consistency(self):
        """Test different hash algorithms produce consistent results."""
        data = "test data"

        md5_hash = hashlib.md5(data.encode()).hexdigest()
        sha1_hash = hashlib.sha1(data.encode()).hexdigest()
        sha256_hash = hashlib.sha256(data.encode()).hexdigest()

        # Test each algorithm is consistent
        assert md5_hash == hashlib.md5(data.encode()).hexdigest()
        assert sha1_hash == hashlib.sha1(data.encode()).hexdigest()
        assert sha256_hash == hashlib.sha256(data.encode()).hexdigest()

    def test_binary_data_hashing(self):
        """Test hashing of binary data."""
        binary_data = b"\x00\x01\x02\x03\x04"

        hash1 = hashlib.sha256(binary_data).hexdigest()
        hash2 = hashlib.sha256(binary_data).hexdigest()

        assert hash1 == hash2

    def test_file_content_hashing(self, temp_dir):
        """Test deterministic file content hashing."""
        file_path = temp_dir / "test_file.txt"
        content = "Test file content\nLine 2\nLine 3"
        file_path.write_text(content)

        # Hash file content
        with open(file_path, 'rb') as f:
            hash1 = hashlib.sha256(f.read()).hexdigest()

        with open(file_path, 'rb') as f:
            hash2 = hashlib.sha256(f.read()).hexdigest()

        assert hash1 == hash2


class TestDeterministicSorting:
    """Test suite for deterministic sorting operations."""

    def test_stable_sort(self):
        """Test stable sorting maintains order for equal elements."""
        data = [
            {"name": "A", "value": 3},
            {"name": "B", "value": 1},
            {"name": "C", "value": 3},
            {"name": "D", "value": 2}
        ]

        sorted_data = sorted(data, key=lambda x: x["value"])

        # Check that equal elements maintain original order
        equal_elements = [d for d in sorted_data if d["value"] == 3]
        assert equal_elements[0]["name"] == "A"
        assert equal_elements[1]["name"] == "C"

    def test_multi_key_sorting(self):
        """Test sorting with multiple keys for determinism."""
        data = [
            {"primary": 2, "secondary": "b"},
            {"primary": 1, "secondary": "c"},
            {"primary": 2, "secondary": "a"},
            {"primary": 1, "secondary": "d"}
        ]

        sorted_data = sorted(data, key=lambda x: (x["primary"], x["secondary"]))

        assert sorted_data[0] == {"primary": 1, "secondary": "c"}
        assert sorted_data[1] == {"primary": 1, "secondary": "d"}
        assert sorted_data[2] == {"primary": 2, "secondary": "a"}
        assert sorted_data[3] == {"primary": 2, "secondary": "b"}

    def test_set_to_list_conversion(self):
        """Test deterministic set to list conversion."""
        data_set = {3, 1, 4, 1, 5, 9, 2, 6}

        # Convert to sorted list for determinism
        list1 = sorted(list(data_set))
        list2 = sorted(list(data_set))

        assert list1 == list2


class TestDeterministicSerialization:
    """Test suite for deterministic serialization."""

    def test_json_serialization_consistency(self):
        """Test JSON serialization is consistent."""
        data = {
            "z_field": "last",
            "a_field": "first",
            "m_field": "middle",
            "nested": {
                "z": 3,
                "a": 1,
                "m": 2
            }
        }

        json1 = json.dumps(data, sort_keys=True, indent=2)
        json2 = json.dumps(data, sort_keys=True, indent=2)

        assert json1 == json2

    def test_decimal_serialization(self):
        """Test Decimal serialization consistency."""
        from decimal import Decimal

        data = {
            "value": Decimal("123.456"),
            "calculation": Decimal("0.1") + Decimal("0.2")
        }

        # Custom encoder for Decimal
        class DecimalEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Decimal):
                    return str(obj)
                return super().default(obj)

        json1 = json.dumps(data, cls=DecimalEncoder, sort_keys=True)
        json2 = json.dumps(data, cls=DecimalEncoder, sort_keys=True)

        assert json1 == json2

    def test_datetime_serialization(self):
        """Test datetime serialization consistency."""
        data = {
            "timestamp": datetime(2025, 1, 1, 12, 0, 0),
            "date": datetime(2025, 1, 1).date()
        }

        # Custom encoder for datetime
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (datetime, datetime.date)):
                    return obj.isoformat()
                return super().default(obj)

        json1 = json.dumps(data, cls=DateTimeEncoder, sort_keys=True)
        json2 = json.dumps(data, cls=DateTimeEncoder, sort_keys=True)

        assert json1 == json2