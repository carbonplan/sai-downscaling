"""
Input data and output store validation checks for the BCSD pipeline.

Use :class:`DatasetValidator` to run checks for a given (gcm, scenario) pair.
Use :func:`validate_output_store` to run checks against a post-consolidation output datatree.
"""

import enum
import hashlib
import inspect
import traceback
from typing import get_args

import pydantic
import xarray as xr

from srm.bcsd_config import VariableName
from srm.config import SCENARIO_TO_GROUP
from srm.datasets import catalog
from srm.qaqc import DatasetChecker, ValidationResult

BLOCKING_CHECKS = {
    "ensemble_member_dim",
    "g6_not_identical_to_ssp245",
    "lineage_member_availability",
    "temporal_coverage",
    "lat_valid",
    "lon_valid",
    "time_axis",
    "calendar",
    "negative_precip",
    "spatial_range_tas",
    "spatial_range_tasmax",
    "spatial_range_tasmin",
    "spatial_range_pr",
    "spatial_range_rsds",
}


GCM_OPTIONS = ("CESM2-WACCM", "MIROC-ES2H", "UKESM")
SCENARIO_OPTIONS = ("historical", "SSP245", "G6-1.5K")
# On-disk variable group names; canonical (lowercase), so no translation needed.
VARIABLE_OPTIONS = get_args(VariableName)

# Expected inclusive daily time bounds per GCM and scenario (observed from actual data).
# CESM2-WACCM uses a "first-of-next-month" time encoding, so its last time step appears
# as the first day of the month following the final data month.
_SCENARIO_TIME_BOUNDS: dict[str, dict[str, tuple[str, str]]] = {
    "CESM2-WACCM": {
        # historical merges ESGF '001' (1978–2015) + Pangeo r*i1p1f1 (1850–2015);
        # the union time axis starts at 1850. End is first-of-next-month encoded.
        "historical": ("1850-01-01", "2015-01-16"),
        "SSP245": ("2015-01-01", "2099-12-31"),
        "G6-1.5K": ("2035-01-01", "2085-01-01"),
    },
    "MIROC-ES2H": {
        "historical": ("1850-01-01", "2014-12-31"),
        # Unified store stitches 2015–2019 gap-fill data together with the GeoMIP baseline.
        "SSP245": ("2015-01-01", "2084-12-31"),
        "G6-1.5K": ("2035-01-01", "2084-12-31"),
    },
    "UKESM": {
        "historical": ("1850-01-01", "2014-12-31"),
        "SSP245": ("2015-01-01", "2099-12-31"),
        "G6-1.5K": ("2035-01-01", "2084-12-31"),
    },
}

_FAST: dict = {"isel_kwargs": {"time": slice(0, 5)}}

_DS_CHECKER_CHECKS: list[tuple[str, str, dict]] = [
    ("ensemble_member_dim", "validate_ensemble_member_dim", {}),
    ("lat_valid", "validate_lat", {}),
    ("lon_valid", "validate_lon", {}),
    ("time_axis", "validate_time_axis", {}),
    ("calendar", "validate_calendar", {}),
    ("negative_precip", "validate_negative_precip", {"var": "pr", **_FAST}),
    ("spatial_range_tas", "validate_spatial_range", {"var": "tas", **_FAST}),
    ("spatial_range_tasmax", "validate_spatial_range", {"var": "tasmax", **_FAST}),
    ("spatial_range_tasmin", "validate_spatial_range", {"var": "tasmin", **_FAST}),
    ("spatial_range_pr", "validate_spatial_range", {"var": "pr", **_FAST}),
    ("spatial_range_rsds", "validate_spatial_range", {"var": "rsds", **_FAST}),
]


def _parse_catalog_value(label: str, value: str, options: tuple[str, ...]) -> str:
    if value in options:
        return value
    raise ValueError(f"Unknown {label} '{value}'. Valid options: {list(options)}")


def parse_gcm(value: str) -> str:
    """Parse a user-provided GCM string to a canonical catalog value."""
    return _parse_catalog_value("GCM", value, GCM_OPTIONS)


def parse_scenario(value: str) -> str:
    """Parse a user-provided scenario string to a canonical catalog value."""
    return _parse_catalog_value("scenario", value, SCENARIO_OPTIONS)


def parse_variable(value: str) -> str:
    """Parse a user-provided variable string to a canonical on-disk group name."""
    return _parse_catalog_value("variable", value, VARIABLE_OPTIONS)


class CheckStatus(enum.StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"  # not applicable for this (gcm, scenario) pair


class CheckResult(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    check_id: str = ""
    gcm: str
    scenario: str
    status: CheckStatus
    message: str = ""
    detail: dict = pydantic.Field(default_factory=dict)


def _get_ensemble_members(ds: xr.Dataset) -> list[str] | None:
    """
    Return ensemble_member coordinate values as strings from an already-opened dataset.

    Returns None if the dataset has no ensemble_member dimension or coordinate.
    """
    if "ensemble_member" in ds.dims or "ensemble_member" in ds.coords:
        return [str(m) for m in ds["ensemble_member"].values]
    return None


class DatasetValidator(pydantic.BaseModel):
    """
    Validates a (gcm, scenario) pair against the BCSD pipeline dataset catalog.

    Raises ``pydantic.ValidationError`` on construction if the gcm or scenario is
    not a recognized option.

    Parameters
    ----------
    gcm : str
        GCM name, must be one of :data:`GCM_OPTIONS`.
    scenario : str
        Scenario name, must be one of :data:`SCENARIO_OPTIONS`.

    Examples
    --------
    >>> validator = DatasetValidator(gcm="CESM2-WACCM", scenario="SSP245")
    >>> results = validator.run_checks()  # all checks
    >>> result = validator.check_temporal_coverage()  # single check
    """

    model_config = pydantic.ConfigDict(frozen=True)

    gcm: str
    scenario: str

    # Keyed by GCM name. Stores xr.DataTree on success or CheckResult on load failure.
    _datatree_cache: dict = pydantic.PrivateAttr(default_factory=dict)

    @pydantic.field_validator("gcm")
    @classmethod
    def _validate_gcm(cls, v: str) -> str:
        return parse_gcm(v)

    @pydantic.field_validator("scenario")
    @classmethod
    def _validate_scenario(cls, v: str) -> str:
        return parse_scenario(v)

    # ── private helpers ──────────────────────────────────────────────────────────

    def _result(
        self,
        status: CheckStatus,
        message: str = "",
        detail: dict | None = None,
    ) -> CheckResult:
        """Pre-filled CheckResult factory — eliminates repeated gcm=/scenario= kwargs."""
        return CheckResult(
            gcm=self.gcm,
            scenario=self.scenario,
            status=status,
            message=message,
            detail=detail or {},
        )

    def _open_datatree(self) -> tuple[xr.DataTree | None, CheckResult | None]:
        """Open the unified GCM datatree store, cached per instance."""
        if self.gcm in self._datatree_cache:
            cached = self._datatree_cache[self.gcm]
            if isinstance(cached, CheckResult):
                return None, cached
            return cached, None

        catalog_entry = catalog.datasets.get(self.gcm)
        if catalog_entry is None:
            err = self._result(
                CheckStatus.FAIL,
                f"Dataset not found in catalog: {self.gcm}",
                {"available_keys": sorted(catalog.datasets.keys())},
            )
            self._datatree_cache[self.gcm] = err
            return None, err

        try:
            dt: xr.DataTree = catalog_entry.to_xarray()  # type: ignore[assignment]
            self._datatree_cache[self.gcm] = dt
            return dt, None
        except Exception as exc:
            tb = traceback.format_exc()
            err = self._result(
                CheckStatus.FAIL,
                f"Failed to load dataset {self.gcm}: {exc}",
                {"traceback": tb},
            )
            self._datatree_cache[self.gcm] = err
            return None, err

    def _open_scenario_ds(self) -> tuple[xr.Dataset | None, CheckResult | None]:
        """Extract this scenario's group from the GCM datatree as a flat xr.Dataset."""
        dt, err = self._open_datatree()
        if err:
            return None, err
        assert dt is not None
        group = SCENARIO_TO_GROUP.get(self.scenario)
        if group is None:
            return None, self._result(
                CheckStatus.SKIP, f"No group mapping for scenario {self.scenario!r}"
            )
        if group not in dt.children:
            return None, self._result(
                CheckStatus.SKIP,
                f"Group {group!r} not present in datatree for {self.gcm}",
            )
        return dt[group].to_dataset(), None

    def _vr_to_cr(self, check_id: str, vr: ValidationResult) -> CheckResult:
        """Map ValidationResult → CheckResult."""
        return CheckResult(
            check_id=check_id,
            gcm=self.gcm,
            scenario=self.scenario,
            status=CheckStatus.PASS if vr else CheckStatus.FAIL,
            message="; ".join(vr.issues) if not vr else "",
        )

    # ── public check methods ─────────────────────────────────────────────────────

    def check_ensemble_member_dim(self) -> CheckResult:
        """Check that the ensemble_member dimension is present and non-empty."""
        ds, err = self._open_scenario_ds()
        if err is not None:
            return err.model_copy(update={"check_id": "ensemble_member_dim"})
        vr = DatasetChecker(ds).validate_ensemble_member_dim()
        return self._vr_to_cr("ensemble_member_dim", vr)

    def check_lineage_member_availability(self) -> CheckResult:
        """
        D1/D2: All lineage-resolved parent members must exist in their stores.

        For each (member, variable) pair registered in the lineage table, resolves the
        historical_member and (for G6-1.5K) the ssp245_member, then checks those exist
        in the respective groups of the unified GCM datatree.

        Skipped when no lineage is registered for this (gcm, scenario).
        """
        from srm.lineage import get_lineage_entries

        entries = get_lineage_entries(self.gcm, self.scenario)
        if not entries:
            return self._result(
                CheckStatus.SKIP,
                f"No lineage registered for {self.gcm} {self.scenario}; skipping.",
            )

        hist_members_needed: set[str] = set()
        ssp245_members_needed: set[str] = set()
        for hist, ssp245, *_ in entries.values():
            hist_members_needed.add(hist)
            if ssp245 is not None:
                ssp245_members_needed.add(ssp245)

        dt, err = self._open_datatree()
        if err is not None:
            return err
        assert dt is not None

        issues: list[str] = []
        detail: dict = {}

        if "historical" not in dt.children:
            return self._result(
                CheckStatus.FAIL,
                f"'historical' group not present in datatree for {self.gcm}",
            )
        hist_ds = dt["historical"].to_dataset()
        hist_available = set(_get_ensemble_members(hist_ds) or [])
        detail["historical_needed"] = sorted(hist_members_needed)
        detail["historical_available"] = sorted(hist_available)
        missing_hist = sorted(hist_members_needed - hist_available)
        if missing_hist:
            issues.append(f"{len(missing_hist)} resolved historical member(s) missing")
            detail["missing_historical"] = missing_hist

        if ssp245_members_needed:
            if "ssp245" not in dt.children:
                return self._result(
                    CheckStatus.FAIL,
                    f"'ssp245' group not present in datatree for {self.gcm}",
                )
            ssp245_ds = dt["ssp245"].to_dataset()
            ssp245_available = set(_get_ensemble_members(ssp245_ds) or [])
            detail["ssp245_needed"] = sorted(ssp245_members_needed)
            detail["ssp245_available"] = sorted(ssp245_available)
            missing_ssp245 = sorted(ssp245_members_needed - ssp245_available)
            if missing_ssp245:
                issues.append(f"{len(missing_ssp245)} resolved SSP245 bridge member(s) missing")
                detail["missing_ssp245"] = missing_ssp245

        if issues:
            return self._result(CheckStatus.FAIL, "; ".join(issues), detail)

        store_labels = "historical" + (" and SSP245" if ssp245_members_needed else "")
        return self._result(
            CheckStatus.PASS,
            f"All {len(entries)} lineage entries resolve to available members "
            f"in {store_labels} store(s).",
            detail,
        )

    def check_g6_not_identical_to_ssp245(self) -> CheckResult:
        """
        E1: G6-1.5K data must not be identical to SSP245 for the same ensemble member.

        Detects copy-paste errors where G6 data was accidentally duplicated from SSP245.
        Samples a small corner slice (first 5 time steps, 10×10 spatial patch) of the
        first common variable and compares SHA-256 checksums per shared member.
        Skipped when scenario is not ``'G6-1.5K'``.
        """
        if self.scenario != "G6-1.5K":
            return self._result(CheckStatus.SKIP, "Only applicable to G6-1.5K scenario.")

        dt, err = self._open_datatree()
        if err is not None:
            return err
        assert dt is not None

        g6_group = SCENARIO_TO_GROUP["G6-1.5K"]
        ssp245_group = SCENARIO_TO_GROUP["SSP245"]
        if g6_group not in dt.children:
            return self._result(
                CheckStatus.SKIP,
                f"Group {g6_group!r} not present in datatree for {self.gcm}.",
            )
        if ssp245_group not in dt.children:
            return self._result(
                CheckStatus.SKIP,
                f"Group {ssp245_group!r} not present in datatree for {self.gcm}.",
            )

        g6_ds = dt[g6_group].to_dataset()
        ssp245_ds = dt[ssp245_group].to_dataset()

        g6_vars = {str(v) for v in g6_ds.data_vars}
        ssp245_vars = {str(v) for v in ssp245_ds.data_vars}
        common_vars = sorted(g6_vars & ssp245_vars)
        if not common_vars:
            return self._result(
                CheckStatus.SKIP,
                "No common variables between G6-1.5K and SSP245; cannot compare.",
                {"g6_vars": sorted(g6_vars), "ssp245_vars": sorted(ssp245_vars)},
            )

        var = common_vars[0]
        sample_kwargs = {"time": slice(0, 5), "lat": slice(0, 10), "lon": slice(0, 10)}

        def _checksum(arr: xr.DataArray) -> str:
            return hashlib.sha256(arr.isel(**sample_kwargs).values.tobytes()).hexdigest()  # type: ignore[call-overload]

        g6_members = _get_ensemble_members(g6_ds)
        ssp245_members = _get_ensemble_members(ssp245_ds)

        identical_members: list[str] = []
        checked_members: list[str] = []

        if g6_members is not None and ssp245_members is not None:
            shared = sorted(set(g6_members) & set(ssp245_members))
            if not shared:
                return self._result(
                    CheckStatus.SKIP,
                    "No shared ensemble members between G6-1.5K and SSP245.",
                )
            for member in shared:
                g6_hash = _checksum(g6_ds[var].sel(ensemble_member=member))
                ssp245_hash = _checksum(ssp245_ds[var].sel(ensemble_member=member))
                checked_members.append(member)
                if g6_hash == ssp245_hash:
                    identical_members.append(member)
        else:
            g6_hash = _checksum(g6_ds[var])
            ssp245_hash = _checksum(ssp245_ds[var])
            checked_members = ["(all)"]
            if g6_hash == ssp245_hash:
                identical_members = ["(all)"]

        if identical_members:
            return self._result(
                CheckStatus.FAIL,
                (
                    f"G6-1.5K data is identical to SSP245 for {len(identical_members)} member(s) "
                    f"(var={var!r}, first 5 time steps, 10×10 corner patch)."
                ),
                {
                    "identical_members": identical_members,
                    "checked_members": checked_members,
                    "variable_sampled": var,
                },
            )

        return self._result(
            CheckStatus.PASS,
            (
                f"G6-1.5K and SSP245 data differ for all {len(checked_members)} "
                f"checked member(s) (var={var!r})."
            ),
            {"checked_members": checked_members, "variable_sampled": var},
        )

    def _check_store_temporal(
        self, ds: xr.Dataset, expected_start: str, expected_end: str
    ) -> tuple[list[str], dict]:
        """Check one dataset's time axis against expected bounds. Returns (issues, detail)."""
        time_index = ds.indexes["time"]
        calendar = ds.time.dt.calendar
        n_times = len(time_index)
        actual_start_str = time_index[0].strftime("%Y-%m-%d")
        actual_end_str = time_index[-1].strftime("%Y-%m-%d")

        n_expected = len(
            xr.date_range(
                start=expected_start, end=expected_end, freq="D", calendar=calendar, use_cftime=True
            )
        )

        detail: dict = {
            "actual_start": actual_start_str,
            "actual_end": actual_end_str,
            "n_times": n_times,
            "n_expected": n_expected,
            "expected_start": expected_start,
            "expected_end": expected_end,
        }
        issues: list[str] = []

        if actual_start_str != expected_start:
            issues.append(f"start date {actual_start_str} != expected {expected_start}")
        if actual_end_str != expected_end:
            issues.append(f"end date {actual_end_str} != expected {expected_end}")

        if n_times != n_expected:
            delta = n_expected - n_times
            label = "missing" if delta > 0 else "extra"
            detail["n_missing_or_extra"] = delta
            issues.append(f"{abs(delta)} {label} time steps (expected {n_expected}, got {n_times})")
        elif n_times > 1:
            inferred_freq = xr.infer_freq(ds.time)
            if inferred_freq != "D":
                detail["inferred_freq"] = inferred_freq
                issues.append(
                    f"time axis is not uniformly daily (inferred freq: {inferred_freq!r}); "
                    "possible gaps or duplicates within range"
                )

        return issues, detail

    def check_temporal_coverage(self) -> CheckResult:
        """
        E2: Time axis must be gapless with correct first and last dates.

        Bounds are looked up per GCM from ``_SCENARIO_TIME_BOUNDS`` (explicit observed date
        strings). Checks the scenario's group in the unified GCM datatree.
        """
        ds, err = self._open_scenario_ds()
        if err is not None:
            return err.model_copy(update={"check_id": "temporal_coverage"})
        assert ds is not None

        if "time" not in ds.dims:
            return self._result(CheckStatus.SKIP, "Dataset has no time dimension.")

        gcm_bounds = _SCENARIO_TIME_BOUNDS.get(self.gcm, {})
        if self.scenario not in gcm_bounds:
            return self._result(
                CheckStatus.SKIP,
                f"No time bounds defined for {self.gcm} / {self.scenario}.",
            )
        expected_start, expected_end = gcm_bounds[self.scenario]

        issues, detail = self._check_store_temporal(ds, expected_start, expected_end)

        if issues:
            return self._result(CheckStatus.FAIL, "; ".join(issues), detail)

        n_times = len(ds.indexes["time"])
        return self._result(
            CheckStatus.PASS,
            f"Temporal coverage complete: {expected_start} to {expected_end} ({n_times} daily steps, no gaps).",
            detail,
        )

    # ── orchestration ────────────────────────────────────────────────────────────

    def run_checks(self) -> list[CheckResult]:
        """Run all applicable checks and return results stamped with ``check_id``."""
        ds, err = self._open_scenario_ds()
        if err:
            return [err]

        checker = DatasetChecker(ds)
        results: list[CheckResult] = []

        for check_id, method, kwargs in _DS_CHECKER_CHECKS:
            m = getattr(checker, method)
            call_kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(m).parameters}
            vr: ValidationResult = m(**call_kwargs)
            results.append(self._vr_to_cr(check_id, vr))

        for bespoke_check, check_id in (
            (self.check_temporal_coverage, "temporal_coverage"),
            (self.check_lineage_member_availability, "lineage_member_availability"),
            (self.check_g6_not_identical_to_ssp245, "g6_not_identical_to_ssp245"),
        ):
            result = bespoke_check()
            if not result.check_id:
                result = result.model_copy(update={"check_id": check_id})
            results.append(result)

        return results


# ── output store validation ──────────────────────────────────────────────────────

# Output leaves are /scenario/variable/member single-variable datasets with no
# ensemble_member dim, so reuse the input primitives but drop the ensemble check.
OUTPUT_CHECKS: list[tuple[str, str, dict]] = [
    c for c in _DS_CHECKER_CHECKS if c[0] != "ensemble_member_dim"
]


def _open_output_datatree(uri: str, branch: str = "main", tag: str | None = None) -> xr.DataTree:
    """Open an icechunk output store as a DataTree at a given branch or tag."""
    import icechunk
    from cloudpathlib import S3Path

    if not uri.startswith(S3Path.cloud_prefix):
        raise ValueError(f"Output store must be an {S3Path.cloud_prefix} URI, got: {uri}")
    path = S3Path(uri)
    storage = icechunk.s3_storage(bucket=path.bucket, prefix=path.key, from_env=True)
    repo = icechunk.Repository.open(storage)
    if tag is not None:
        session = repo.readonly_session(tag=tag)
    else:
        session = repo.readonly_session(branch=branch)
    return xr.open_datatree(session.store, engine="zarr", chunks="auto", consolidated=False)


def validate_output_store(
    store: str | xr.DataTree,
    branch: str = "main",
    tag: str | None = None,
    scenarios: list[str] | None = None,
    variables: list[str] | None = None,
) -> list[CheckResult]:
    """Run OUTPUT_CHECKS against every populated leaf of an output datatree store.

    ``store`` may be an S3 URI string or an already-open DataTree. ``branch`` or ``tag`` mirror icechunk's ``readonly_session`` parameters and are ignored
    when ``store`` is a DataTree. Each CheckResult reuses the ``gcm`` field for the store
    label and the ``scenario`` field for the leaf path.

    ``scenarios`` and ``variables`` restrict validation to matching leaves of the
    ``/scenario/variable/member`` tree; they take the on-disk group names (e.g. ``"ssp245"``,
    ``"tas"``). ``None`` means no filter.
    """
    if isinstance(store, str):
        from cloudpathlib import S3Path

        tree = _open_output_datatree(store, branch=branch, tag=tag)
        label = S3Path(store).name or store
    else:
        tree = store
        label = "output"

    def _select(node: xr.DataTree, names: list[str] | None) -> list[xr.DataTree]:
        """Child nodes to descend into: all children, or only the named ones present."""
        if names is None:
            return list(node.children.values())
        return [node[name] for name in names if name in node.children]

    # Descend the /scenario/variable/member tree one level at a time; member nodes are
    # the leaves we validate. Filtering uses the tree structure, not path parsing.
    scenario_nodes = _select(tree, scenarios)
    variable_nodes = [vn for sn in scenario_nodes for vn in _select(sn, variables)]

    results: list[CheckResult] = []
    for variable_node in variable_nodes:
        leaf_var = variable_node.name
        for node in variable_node.leaves:
            checker = DatasetChecker(node.to_dataset())
            for check_id, method, kwargs in OUTPUT_CHECKS:
                target_var = kwargs.get("var")
                if target_var is not None and target_var != leaf_var:
                    continue
                sig = inspect.signature(getattr(checker, method))
                call_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
                vr: ValidationResult = getattr(checker, method)(**call_kwargs)
                results.append(
                    CheckResult(
                        check_id=check_id,
                        gcm=label,
                        scenario=node.path,
                        status=CheckStatus.PASS if vr else CheckStatus.FAIL,
                        message="" if vr else "; ".join(vr.issues),
                    )
                )
    return results
