"""Tests for the raw NetCDF S3 layout convention and its ETL prefix wiring."""

import obstore as obs
import pytest
from obstore.store import from_url

from saidownscale.input_data import cesm2_waccm, ukesm
from saidownscale.input_data.etl_utils import RAW_ROOT, get_aws_creds, raw_netcdf_prefix


class TestRawNetcdfPrefix:
    def test_builds_expected_layout(self):
        assert raw_netcdf_prefix("CESM2-WACCM", "g6-1p5k") == (
            "input/raw/CESM2-WACCM/netcdf/g6-1p5k"
        )

    def test_no_trailing_slash(self):
        assert not raw_netcdf_prefix("UKESM", "ssp245").endswith("/")

    def test_rooted_at_raw_root(self):
        assert RAW_ROOT == "input/raw"
        assert raw_netcdf_prefix("UKESM", "g6-1p5k").startswith(f"{RAW_ROOT}/")


ALL_PREFIXES = [
    *cesm2_waccm.NETCDF_PREFIX.values(),
    *ukesm.S3_INPUT_PREFIX.values(),
    *ukesm.T_PR_INPUT_PREFIX.values(),
]


def _list_paths(prefix: str) -> list[str]:
    """Return every object path under an S3 prefix in the carbonplan-srm bucket."""
    aws = get_aws_creds()
    region = aws.pop("region")
    store = from_url("s3://carbonplan-srm", region=region, **aws)
    stream = obs.list_with_delimiter(store, prefix=prefix, return_arrow=True)
    return [str(p) for p in stream["objects"]["path"].to_numpy()]


@pytest.mark.input_data
class TestRawPrefixesOnS3:
    @pytest.mark.parametrize("prefix", ALL_PREFIXES)
    def test_prefix_is_under_raw_root_and_non_empty(self, prefix):
        assert prefix.startswith("input/raw/")
        paths = _list_paths(prefix)
        assert any(p.endswith(".nc") for p in paths), f"no .nc objects under {prefix}"

    @pytest.mark.parametrize(
        "module, scenario",
        [
            (cesm2_waccm, "historical"),
            (cesm2_waccm, "SSP245"),
            (cesm2_waccm, "G6-1.5K"),
            (cesm2_waccm, "G6-1.5K-END"),
            (ukesm, "SSP245"),
        ],
    )
    def test_all_configured_members_discovered(self, module, scenario):
        # UKESM G6-1.5K is excluded: its T/PR drop carries raw source member ids
        # (u-dp583) that only become CMIP6 ripf labels after T_PR_MEMBER_RENAME is
        # applied downstream of discovery.
        expected = set(module.ENSEMBLE_MEMBERS[scenario])
        found = {m for m, _ in module._get_netcdf_urls(scenario, "tas")}
        assert expected <= found, f"missing members for {scenario}: {expected - found}"

    @pytest.mark.parametrize(
        "module, scenario, variables, forbidden",
        [
            # Each row lists a SHORT prefix while a longer sibling exists, which is
            # the only direction in which string-prefix matching could over-reach.
            # obstore matches prefixes by path component and does not recurse, so
            # neither the sibling's objects nor its child prefix are reachable today;
            # these rows pin that behavior against a future listing-semantics change.
            (cesm2_waccm, "G6-1.5K", ["tas", "pr"], "/netcdf/g6-1p5k-end/"),
            # SSP245 no longer lists S3 at all — paths are built from a filename
            # template — so this row now pins that the template stays off the old drop.
            (ukesm, "SSP245", ["rsds", "pr"], "/netcdf/ssp245-t-pr/"),
            (ukesm, "G6-1.5K", ["hurs", "rsds"], "/netcdf/g6-1p5k-t-pr/"),
        ],
    )
    def test_no_sibling_prefix_bleed(self, module, scenario, variables, forbidden):
        for variable in variables:
            paths = [p for _, p in module._get_netcdf_urls(scenario, variable)]
            assert paths, f"{scenario}/{variable} discovered nothing"
            bleed = [p for p in paths if forbidden in p]
            assert not bleed, f"{scenario}/{variable} leaked into {forbidden}: {bleed[:3]}"
