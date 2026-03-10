"""Tests for CLI helper functions."""

import itertools
import logging

import pytest
from pydantic import ValidationError

from srm.bcsd_config import BCSDConfig
from srm.cli import configs_from_matrix


class TestConfigsFromMatrix:
    """Tests for the configs_from_matrix helper function."""

    def test_single_combination_returns_one_config(self):
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=["r1i1p1f1"],
            scenarios=[None],
        )
        assert len(configs) == 1

    def test_cartesian_product_count(self):
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM", "MIROC"],
            variables=["tas", "pr"],
            members=["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"],
            scenarios=["ssp245", "G6-1pt5k"],
            predict_period_start=2015,
            predict_period_end=2100,
        )
        assert len(configs) == 2 * 2 * 3 * 2  # 24

    def test_returns_bcsd_config_instances(self):
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=["r1i1p1f1"],
            scenarios=[None],
        )
        assert all(isinstance(c, BCSDConfig) for c in configs)

    def test_historical_only_scenario_is_none(self):
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=["r1i1p1f1", "r2i1p1f1"],
            scenarios=[None],
        )
        assert len(configs) == 2
        assert all(c.scenario is None for c in configs)

    def test_all_combinations_present(self):
        gcms = ["CESM2-WACCM", "MIROC"]
        variables = ["tas", "pr"]
        members = ["r1i1p1f1", "r2i1p1f1"]
        scenarios = ["ssp245"]
        configs = configs_from_matrix(
            gcms=gcms,
            variables=variables,
            members=members,
            scenarios=scenarios,
            predict_period_start=2015,
            predict_period_end=2100,
        )
        actual = {(c.gcm, c.variable, c.ensemble_member, c.scenario) for c in configs}
        expected = set(itertools.product(gcms, variables, members, scenarios))
        assert actual == expected

    def test_shared_params_applied_to_all_configs(self):
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM", "MIROC"],
            variables=["tas"],
            members=["r1i1p1f1"],
            scenarios=[None],
            environment="staging",
            version="v2",
            train_period_start=1979,
            train_period_end=2013,
        )
        assert all(c.environment == "staging" for c in configs)
        assert all(c.version == "v2" for c in configs)
        assert all(c.train_period_start == 1979 for c in configs)
        assert all(c.train_period_end == 2013 for c in configs)

    def test_subset_bounds_propagated(self):
        bounds = (-35.0, -22.0, 16.0, 33.0)
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=["r1i1p1f1"],
            scenarios=[None],
            subset_bounds=bounds,
        )
        assert all(c.subset_bounds == bounds for c in configs)

    def test_scenario_without_predict_period_raises(self):
        """BCSDConfig raises ValidationError when scenario is set but predict periods are missing."""
        with pytest.raises(ValidationError):
            configs_from_matrix(
                gcms=["CESM2-WACCM"],
                variables=["tas"],
                members=["r1i1p1f1"],
                scenarios=["ssp245"],
                # predict_period_start / predict_period_end intentionally omitted
            )

    def test_multiple_scenarios_all_present(self):
        scenarios = ["ssp245", "G6-1pt5k", "G6-termination"]
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=["r1i1p1f1"],
            scenarios=scenarios,
            predict_period_start=2015,
            predict_period_end=2100,
        )
        assert len(configs) == 3
        assert {c.scenario for c in configs} == set(scenarios)

    def test_empty_members_returns_empty_list(self):
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=[],
            scenarios=[None],
        )
        assert configs == []

    def test_fields_assigned_correctly(self):
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["pr"],
            members=["r3i1p1f1"],
            scenarios=["ssp245"],
            predict_period_start=2020,
            predict_period_end=2080,
        )
        cfg = configs[0]
        assert cfg.gcm == "CESM2-WACCM"
        assert cfg.variable == "pr"
        assert cfg.ensemble_member == "r3i1p1f1"
        assert cfg.scenario == "ssp245"
        assert cfg.predict_period_start == 2020
        assert cfg.predict_period_end == 2080


class TestConfigsFromMatrixExcludedMembers:
    """Tests for the excluded_members_by_variable parameter of configs_from_matrix."""

    def test_excluded_variable_member_not_in_output(self):
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tasmax"],
            members=["r1i1p1f1", "r7i1p1f1"],
            scenarios=[None],
            excluded_members_by_variable={"tasmax": ["r1i1p1f1"]},
        )
        members = [c.ensemble_member for c in configs]
        assert "r1i1p1f1" not in members
        assert "r7i1p1f1" in members

    def test_exclusion_only_applies_to_specified_variable(self):
        """Excluding r1 for tasmax must not affect tas configs."""
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas", "tasmax"],
            members=["r1i1p1f1", "r7i1p1f1"],
            scenarios=[None],
            excluded_members_by_variable={"tasmax": ["r1i1p1f1"]},
        )
        tas_members = [c.ensemble_member for c in configs if c.variable == "tas"]
        tasmax_members = [c.ensemble_member for c in configs if c.variable == "tasmax"]
        assert "r1i1p1f1" in tas_members
        assert "r1i1p1f1" not in tasmax_members

    def test_multiple_variables_excluded_independently(self):
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tasmax", "pr", "tas"],
            members=["r1i1p1f1", "r2i1p1f1", "r7i1p1f1"],
            scenarios=[None],
            excluded_members_by_variable={
                "tasmax": ["r1i1p1f1", "r2i1p1f1"],
                "pr": ["r1i1p1f1", "r2i1p1f1"],
            },
        )
        for var in ("tasmax", "pr"):
            members = [c.ensemble_member for c in configs if c.variable == var]
            assert "r1i1p1f1" not in members
            assert "r2i1p1f1" not in members
            assert "r7i1p1f1" in members

        # tas is untouched
        tas_members = [c.ensemble_member for c in configs if c.variable == "tas"]
        assert "r1i1p1f1" in tas_members
        assert "r2i1p1f1" in tas_members

    def test_count_is_correct_after_exclusion(self):
        # 1 GCM x 2 variables x 3 members x 1 scenario = 6 total
        # exclude r1 for tasmax only => 6 - 1 = 5
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas", "tasmax"],
            members=["r1i1p1f1", "r2i1p1f1", "r7i1p1f1"],
            scenarios=[None],
            excluded_members_by_variable={"tasmax": ["r1i1p1f1"]},
        )
        assert len(configs) == 5

    def test_none_excluded_members_behaves_like_no_exclusion(self):
        configs_with_none = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=["r1i1p1f1", "r2i1p1f1"],
            scenarios=[None],
            excluded_members_by_variable=None,
        )
        configs_without = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=["r1i1p1f1", "r2i1p1f1"],
            scenarios=[None],
        )
        assert len(configs_with_none) == len(configs_without)

    def test_excluded_combo_emits_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            configs_from_matrix(
                gcms=["CESM2-WACCM"],
                variables=["tasmax"],
                members=["r1i1p1f1", "r7i1p1f1"],
                scenarios=[None],
                excluded_members_by_variable={"tasmax": ["r1i1p1f1"]},
            )
        assert any("r1i1p1f1" in record.message for record in caplog.records)

    def test_empty_exclusion_dict_no_filtering(self):
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tasmax"],
            members=["r1i1p1f1", "r7i1p1f1"],
            scenarios=[None],
            excluded_members_by_variable={},
        )
        assert len(configs) == 2

    def test_all_members_excluded_returns_empty_for_that_variable(self):
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tasmax"],
            members=["r1i1p1f1", "r2i1p1f1"],
            scenarios=[None],
            excluded_members_by_variable={"tasmax": ["r1i1p1f1", "r2i1p1f1"]},
        )
        tasmax_configs = [c for c in configs if c.variable == "tasmax"]
        assert tasmax_configs == []

    def test_single_combination_returns_one_config(self):
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=["r1i1p1f1"],
            scenarios=[None],
        )
        assert len(configs) == 1

    def test_cartesian_product_count(self):
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM", "MIROC"],
            variables=["tas", "pr"],
            members=["r1i1p1f1", "r2i1p1f1", "r3i1p1f1"],
            scenarios=["ssp245", "G6-1pt5k"],
            predict_period_start=2015,
            predict_period_end=2100,
        )
        assert len(configs) == 2 * 2 * 3 * 2  # 24

    def test_returns_bcsd_config_instances(self):
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=["r1i1p1f1"],
            scenarios=[None],
        )
        assert all(isinstance(c, BCSDConfig) for c in configs)

    def test_historical_only_scenario_is_none(self):
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=["r1i1p1f1", "r2i1p1f1"],
            scenarios=[None],
        )
        assert len(configs) == 2
        assert all(c.scenario is None for c in configs)

    def test_all_combinations_present(self):
        gcms = ["CESM2-WACCM", "MIROC"]
        variables = ["tas", "pr"]
        members = ["r1i1p1f1", "r2i1p1f1"]
        scenarios = ["ssp245"]
        configs = configs_from_matrix(
            gcms=gcms,
            variables=variables,
            members=members,
            scenarios=scenarios,
            predict_period_start=2015,
            predict_period_end=2100,
        )
        actual = {(c.gcm, c.variable, c.ensemble_member, c.scenario) for c in configs}
        expected = set(itertools.product(gcms, variables, members, scenarios))
        assert actual == expected

    def test_shared_params_applied_to_all_configs(self):
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM", "MIROC"],
            variables=["tas"],
            members=["r1i1p1f1"],
            scenarios=[None],
            environment="staging",
            version="v2",
            train_period_start=1979,
            train_period_end=2013,
        )
        assert all(c.environment == "staging" for c in configs)
        assert all(c.version == "v2" for c in configs)
        assert all(c.train_period_start == 1979 for c in configs)
        assert all(c.train_period_end == 2013 for c in configs)

    def test_subset_bounds_propagated(self):
        bounds = (-35.0, -22.0, 16.0, 33.0)
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=["r1i1p1f1"],
            scenarios=[None],
            subset_bounds=bounds,
        )
        assert all(c.subset_bounds == bounds for c in configs)

    def test_scenario_without_predict_period_raises(self):
        """BCSDConfig raises ValidationError when scenario is set but predict periods are missing."""
        with pytest.raises(ValidationError):
            configs_from_matrix(
                gcms=["CESM2-WACCM"],
                variables=["tas"],
                members=["r1i1p1f1"],
                scenarios=["ssp245"],
                # predict_period_start / predict_period_end intentionally omitted
            )

    def test_multiple_scenarios_all_present(self):
        scenarios = ["ssp245", "G6-1pt5k", "G6-termination"]
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=["r1i1p1f1"],
            scenarios=scenarios,
            predict_period_start=2015,
            predict_period_end=2100,
        )
        assert len(configs) == 3
        assert {c.scenario for c in configs} == set(scenarios)

    def test_empty_members_returns_empty_list(self):
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["tas"],
            members=[],
            scenarios=[None],
        )
        assert configs == []

    def test_fields_assigned_correctly(self):
        configs = configs_from_matrix(
            gcms=["CESM2-WACCM"],
            variables=["pr"],
            members=["r3i1p1f1"],
            scenarios=["ssp245"],
            predict_period_start=2020,
            predict_period_end=2080,
        )
        cfg = configs[0]
        assert cfg.gcm == "CESM2-WACCM"
        assert cfg.variable == "pr"
        assert cfg.ensemble_member == "r3i1p1f1"
        assert cfg.scenario == "ssp245"
        assert cfg.predict_period_start == 2020
        assert cfg.predict_period_end == 2080
