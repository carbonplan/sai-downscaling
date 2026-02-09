import time
from io import StringIO
from unittest.mock import patch

import numpy as np
from validators import DatasetValidator

from srm.utils import Timer, lon_to_180


class TestTimer:
    """Test the Timer context manager."""

    def test_timer_basic_usage(self):
        """Test that Timer records elapsed time correctly."""
        with Timer("test operation", verbose=False) as timer:
            time.sleep(0.1)

        assert timer.elapsed is not None
        assert timer.elapsed >= 0.1
        assert timer.name == "test operation"

    def test_timer_verbose_output(self):
        """Test that Timer prints output when verbose=True."""
        with patch("sys.stdout", new=StringIO()) as fake_output:
            with Timer("test operation", verbose=True):
                time.sleep(0.05)

            output = fake_output.getvalue()
            assert "test operation:" in output
            assert "seconds" in output

    def test_timer_no_verbose_output(self):
        """Test that Timer does not print when verbose=False."""
        with patch("sys.stdout", new=StringIO()) as fake_output:
            with Timer("test operation", verbose=False):
                time.sleep(0.05)

            output = fake_output.getvalue()
            assert output == ""

    def test_timer_with_exception(self):
        """Test that Timer records time even when exception occurs."""
        try:
            with Timer("failing operation", verbose=False) as timer:
                time.sleep(0.05)
                raise ValueError("test error")
        except ValueError:
            pass

        # Timer should still record elapsed time
        assert timer.elapsed is not None
        assert timer.elapsed >= 0.05

    def test_timer_format_precision(self):
        """Test that Timer formats output with 2 decimal places."""
        with patch("sys.stdout", new=StringIO()) as fake_output:
            with Timer("test", verbose=True):
                time.sleep(0.123)

            output = fake_output.getvalue()
            # Should have format like "test: 0.12 seconds"
            assert ".12" in output or ".13" in output  # Allow for timing variance


class TestLonTo180:
    def test_monotonic(self, ds_monotonic):
        result = lon_to_180(ds_monotonic)
        validator = DatasetValidator(result)
        validation_result = validator.validate_lon(check_monotonic=True)
        assert validation_result, f"{validation_result.issues}"

    def test_non_monotonic(self, ds_non_monotonic):
        result = lon_to_180(ds_non_monotonic)
        validator = DatasetValidator(result)
        validation_result = validator.validate_lon(check_monotonic=True)
        assert validation_result, f"{validation_result.issues}"

        expected_sorted = [-180, -90, -60, -10, -5, 0, 5, 10]
        np.testing.assert_array_almost_equal(validator.ds["lon"].values, expected_sorted)

    def test_0_360_range_fails(self, ds_0_360):
        validator = DatasetValidator(ds_0_360)
        validation_result = validator.validate_lon(check_monotonic=True)

        assert not validation_result, "expected validation to fail for 0-360 longitude range"
        assert any("outside" in issue for issue in validation_result.issues)
