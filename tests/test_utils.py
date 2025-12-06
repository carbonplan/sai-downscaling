import numpy as np
from validators import DatasetValidator

from srm.utils import lon_to_180


class TestLonTo180:
    def test_monotonic(self, ds_monotonic):
        result = lon_to_180(ds_monotonic)

        validator = DatasetValidator(result)
        is_valid = validator.validate_lon(check_monotonic=True)

        assert is_valid, f"monotonic longitude failed: {validator.get_issues()}"

    def test_non_monotonic(self, ds_non_monotonic):
        result = lon_to_180(ds_non_monotonic)

        validator = DatasetValidator(result)
        is_valid = validator.validate_lon(check_monotonic=True)

        assert is_valid, f"non-monotonic longitude failed: {validator.get_issues()}"

        expected_sorted = [-180, -90, -60, -10, -5, 0, 5, 10]
        np.testing.assert_array_almost_equal(result.lon.values, expected_sorted)

    def test_0_360_range_fails(self, ds_0_360):
        validator = DatasetValidator(ds_0_360)
        is_valid = validator.validate_lon(check_monotonic=True)

        assert not is_valid, "expected validation to fail for 0-360 longitude range"

        # checks the failure case
        assert any("outside" in issue for issue in validator.get_issues())
