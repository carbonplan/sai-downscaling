import numpy as np

from srm.utils import lon_to_180


class TestLonTo180:
    def test_monotonic(self, ds_monotonic):
        result = lon_to_180(ds_monotonic)

        assert result.lon.min() >= -180
        assert result.lon.max() <= 180
        assert np.all(np.diff(result.lon) > 0)

    def test_non_monotonic(self, ds_non_monotonic):
        result = lon_to_180(ds_non_monotonic)

        assert result.lon.min() >= -180
        assert result.lon.max() <= 180
        assert np.all(np.diff(result.lon) > 0)

        expected_sorted = [-180, -90, -60, -10, -5, 0, 5, 10]
        np.testing.assert_array_almost_equal(result.lon.values, expected_sorted)
