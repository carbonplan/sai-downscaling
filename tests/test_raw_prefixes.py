"""Tests for the raw NetCDF S3 layout convention and its ETL prefix wiring."""

from srm.input_data.etl_utils import RAW_ROOT, raw_netcdf_prefix


class TestRawNetcdfPrefix:
    def test_builds_expected_layout(self):
        assert raw_netcdf_prefix("CESM2-WACCM", "g6-1p5k") == (
            "input/raw/CESM2-WACCM/netcdf/g6-1p5k"
        )

    def test_no_trailing_slash(self):
        assert not raw_netcdf_prefix("UKESM", "ssp245").endswith("/")

    def test_rooted_at_raw_root(self):
        assert RAW_ROOT == "input/raw"
        assert raw_netcdf_prefix("MIROC-ES2H", "baseline").startswith(f"{RAW_ROOT}/")
