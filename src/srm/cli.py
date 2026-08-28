"""
Command-line interface for the BCSD downscaling pipeline.

Provides a typer-based ``bcsd`` command with subcommands for running, validating, and
inspecting the pipeline. Supports both single-config and matrix-expansion execution
with automatic caching and optional Coiled integration.
"""

import contextlib
import itertools
import json
import logging
import os
import sys
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.tree import Tree

from srm.bcsd_config import BCSDConfig, DownscalingMethod, PipelineOptions, VariableConfig
from srm.cache import ArtifactCache
from srm.cost import estimate_workflow
from srm.orchestration import BCSDOrchestrator
from srm.validation import CheckResult, CheckStatus

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

_STATUS_SYMBOL = {
    CheckStatus.PASS: "[green]✓[/green]",
    CheckStatus.FAIL: "[red]✗[/red]",
    CheckStatus.SKIP: "-",
}

app = typer.Typer(help="BCSD downscaling pipeline with automatic caching")


@contextlib.contextmanager
def _validation_cluster(
    use_coiled: bool,
    *,
    n_workers: int,
    worker_vm_type: str,
    adaptive_max: int | None,
) -> Iterator[None]:
    """Yield with a Coiled Dask client active, or as a no-op when ``use_coiled`` is False.

    Validation checks are lazy dask reductions; the active client makes every ``.compute()``
    run on cluster workers while the driver keeps rendering tables and the exit gate — only
    scalar results return to the driver, so full arrays never touch it. When ``use_coiled``
    is False the reductions run in-process on the default scheduler.

    Reuses :func:`srm.config.setup_cluster` (the canonical SRM Coiled setup, as in
    ``input_data/era5.py``) so region, tags, and spot policy live in one place. ``n_workers``
    with ``adaptive_max`` becomes an ``[min, max]`` range so Coiled scales adaptively, and the
    scheduler VM type is pinned to the worker type so their processor architectures match
    (Coiled rejects mixing x86_64 and aarch64 Graviton VMs).
    """
    if not use_coiled:
        yield
        return

    from srm.config import ClusterConfig, setup_cluster

    config = ClusterConfig(
        worker_vm_types=[worker_vm_type],
        scheduler_vm_types=worker_vm_type,
        n_workers=[n_workers, adaptive_max] if adaptive_max is not None else n_workers,
    )
    with setup_cluster(config):
        yield


def _build_check_matrix_table(
    check_ids: list[str],
    columns: list[str],
    index: dict[tuple[str, str], CheckResult],
) -> Table:
    tbl = Table(show_header=True, header_style="bold", box=box.SIMPLE_HEAD, padding=(0, 1))
    tbl.add_column("check", style="dim", no_wrap=True)
    for col in columns:
        tbl.add_column(col, justify="center")
    for cid in check_ids:
        row = [cid]
        for col in columns:
            r = index.get((cid, col))
            row.append(_STATUS_SYMBOL[r.status] if r else " ")
        tbl.add_row(*row)
    return tbl


# (plural key, singular key, BCSDConfig field). The first four axes select a slice of
# input data. 'downscaling_method' is different in kind: it selects an algorithm, and
# picks which per-variable defaults table VariableConfig is built from, so expanding
# over it re-derives variable_config per combination rather than reusing one.
_MATRIX_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("gcms", "gcm", "gcm"),
    ("variables", "variable", "variable"),
    ("ensemble_members", "ensemble_member", "ensemble_member"),
    ("scenarios", "scenario", "scenario"),
    ("downscaling_methods", "downscaling_method", "downscaling_method"),
)


def _print_lineage_summary(configs: list[BCSDConfig]) -> None:
    """Print a compact table of resolved ensemble member lineage.

    Only rows where lineage differs from ensemble_member are shown, deduplicated
    by (gcm, variable, ensemble_member).
    """
    from srm.lineage import resolve_member_lineage

    table = Table(
        title="Ensemble Member Lineage",
        show_header=True,
        header_style="bold cyan",
        box=box.SIMPLE_HEAD,
    )
    table.add_column("GCM", style="cyan")
    table.add_column("Variable")
    table.add_column("Member", justify="right")
    table.add_column("→ Historical", style="green", justify="right")
    table.add_column("→ SSP245 Bridge", style="yellow", justify="right")

    seen: set[tuple[str, str, str]] = set()
    for cfg in configs:
        if cfg.scenario is None:
            continue
        key = (cfg.gcm, cfg.variable, cfg.ensemble_member)
        if key in seen:
            continue
        seen.add(key)
        try:
            lineage = resolve_member_lineage(
                cfg.gcm, cfg.scenario, cfg.ensemble_member, cfg.variable
            )
        except KeyError:
            continue
        hist, ssp245 = lineage.historical, lineage.ssp245_bridge
        if hist == cfg.ensemble_member and ssp245 == cfg.ensemble_member:
            continue
        ssp = ssp245 if ssp245 != cfg.ensemble_member else "—"
        table.add_row(cfg.gcm, cfg.variable, cfg.ensemble_member, hist, ssp)

    if table.row_count:
        console.print(table)


def _print_validate_lineage_summary(gcm: str, scenarios: list[str]) -> None:
    """Print a per-scenario lineage table for the validate command.

    Deduplicated by (member, hist, ssp245_bridge, ssp245_esgf_bridge, sai_parent);
    variables sharing the same parents are listed together. Skipped when no lineage
    is registered for the GCM/scenario pair.
    """

    from srm.lineage import ScenarioMember, get_lineage_entries

    def _sort_token(field: str | ScenarioMember | None) -> str:
        if field is None:
            return ""
        return str(field)

    for scenario in scenarios:
        if scenario == "historical":
            continue
        entries = get_lineage_entries(gcm, scenario)
        if not entries:
            continue

        has_ssp245 = any(e.ssp245_bridge is not None for e in entries.values())
        has_ssp245_esgf = any(e.ssp245_esgf_bridge is not None for e in entries.values())
        has_sai_parent = any(e.sai_parent is not None for e in entries.values())

        tbl = Table(
            title=f"Lineage — {scenario}",
            show_header=True,
            header_style="bold",
            box=box.SIMPLE_HEAD,
            padding=(0, 1),
        )
        tbl.add_column("member", style="cyan", no_wrap=True)
        tbl.add_column("variables", style="dim", no_wrap=True)
        tbl.add_column("→ historical", style="green", justify="right")
        if has_ssp245:
            tbl.add_column("→ SSP245 bridge", style="yellow", justify="right")
        if has_ssp245_esgf:
            tbl.add_column("→ SSP245 ESGF bridge", style="magenta", justify="right")
        if has_sai_parent:
            tbl.add_column("→ SAI parent", style="blue", justify="right")

        seen: dict[tuple[str, str, str | None, str | None, ScenarioMember | None], list[str]] = {}
        for (member, var), entry in sorted(entries.items()):
            key = (
                member,
                entry.historical,
                entry.ssp245_bridge,
                entry.ssp245_esgf_bridge,
                entry.sai_parent,
            )
            if key not in seen:
                seen[key] = []
            if var not in seen[key]:
                seen[key].append(var)

        # Sort on rendered tokens rather than the raw key: the key mixes str, None, and
        # ScenarioMember, and Python compares element-wise only once the earlier fields
        # tie, so a raw sort raises TypeError the moment two rows differ only in a
        # None-vs-set field or only in their SAI parent.
        for (member, hist, ssp245, ssp245_esgf, sai_parent), variables in sorted(
            seen.items(), key=lambda item: tuple(_sort_token(field) for field in item[0])
        ):
            row = [member, "/".join(sorted(variables)), hist]
            if has_ssp245:
                row.append(ssp245 or "—")
            if has_ssp245_esgf:
                row.append(ssp245_esgf or "—")
            if has_sai_parent:
                row.append(str(sai_parent) if sai_parent else "—")
            tbl.add_row(*row)

        console.print(tbl)


def _validate_predict_periods(configs: list[BCSDConfig]) -> None:
    """Reject configs whose predict_period falls outside a member's valid data extent."""
    from srm.validation import CheckStatus, check_config_time_domain

    failures = [r for c in configs if (r := check_config_time_domain(c)).status == CheckStatus.FAIL]
    if failures:
        raise ValueError(
            "predict_period out of bounds for the following configs:\n"
            + "\n".join(
                f"  {r.gcm}/{r.scenario}/{r.ensemble_member}: {r.message}" for r in failures
            )
        )


def _validate_lineage_members(configs: list[BCSDConfig]) -> None:
    """Cross-scenario validation: check resolved members exist in the unified datatree store.

    Opens each GCM's unified datatree at most once and inspects group children to
    determine which ensemble members are present. Silently skips stores that are
    unreachable or whose groups cannot be navigated.
    """
    from srm.config import SCENARIO_TO_GROUP
    from srm.datasets import catalog
    from srm.lineage import resolve_member_lineage

    errors: list[str] = []
    _dt_cache: dict[str, object] = {}  # gcm → DataTree or None

    def _get_dt(gcm: str):
        if gcm not in _dt_cache:
            try:
                _dt_cache[gcm] = catalog.get(gcm).to_xarray()
            except Exception:
                _dt_cache[gcm] = None
        return _dt_cache[gcm]

    def _members(gcm: str, group: str) -> frozenset[str] | None:
        dt = _get_dt(gcm)
        if dt is None:
            return None
        try:
            return frozenset(dt[group].children.keys())
        except Exception:
            return None

    for config in configs:
        if config.scenario is None:
            continue
        try:
            lineage = resolve_member_lineage(
                config.gcm, config.scenario, config.ensemble_member, config.variable
            )
        except KeyError:
            continue
        hist, ssp245 = lineage.historical, lineage.ssp245_bridge

        hist_group = f"historical/{config.variable}"
        known = _members(config.gcm, hist_group)
        if known is not None and hist not in known:
            errors.append(
                f"  {config.gcm}/{config.variable}: historical:{hist!r} not in {hist_group}"
            )

        if ssp245 is not None:
            try:
                scenario_group = SCENARIO_TO_GROUP[config.scenario]
            except KeyError:
                scenario_group = None
            if scenario_group is not None:
                ssp245_group = f"{scenario_group}/{config.variable}"
                known = _members(config.gcm, ssp245_group)
                if known is not None and ssp245 not in known:
                    errors.append(
                        f"  {config.gcm}/{config.variable}: ssp245:{ssp245!r} not in {ssp245_group}"
                    )

    if errors:
        raise ValueError("Resolved ensemble members not found in stores:\n" + "\n".join(errors))


def _validate_variable_overrides(overrides: dict, variables: list[str]) -> None:
    """
    Validate the shape of a ``variable_overrides`` mapping.

    Parameters
    ----------
    overrides : dict
        Mapping of variable name to a mapping of ``VariableConfig`` field to value.
    variables : list[str]
        Variables the run actually covers.

    Raises
    ------
    ValueError
        If a key names a variable outside the run, a value is not a mapping, or a
        field is not a ``VariableConfig`` field.
    """
    valid_fields = set(VariableConfig.model_fields)
    for var, fields in overrides.items():
        if var not in variables:
            raise ValueError(
                f"variable_overrides key {var!r} is not in variables {sorted(variables)}. "
                "Every override must target a variable included in the run."
            )
        if not isinstance(fields, dict):
            raise ValueError(
                f"variable_overrides[{var!r}] must be a mapping of field -> value, "
                f"got {type(fields).__name__}"
            )
        unknown = sorted(set(fields) - valid_fields)
        if unknown:
            raise ValueError(
                f"variable_overrides[{var!r}] has unknown field(s) {unknown}. "
                f"Valid fields: {sorted(valid_fields)}"
            )


def _resolve_variable_config(
    variable: str,
    downscaling_method: DownscalingMethod,
    run_wide: dict | None = None,
    overrides: dict[str, dict] | None = None,
) -> VariableConfig:
    """
    Resolve a ``VariableConfig`` through the three precedence tiers.

    Tiers, last writer wins: per-variable table defaults, then run-wide values, then
    the per-variable override entry. ``None`` values are skipped so an unset CLI flag
    never clobbers a default. Construction goes through ``VariableConfig(**merged)``
    rather than ``model_copy(update=...)`` because ``model_copy`` does not validate,
    which previously let bad values through to the pipeline.

    ``downscaling_method`` selects which defaults table the first tier comes from,
    since BCSD and QDMSD disagree on more than one field (detrending, window length,
    window step, and debias_approach all differ).

    Parameters
    ----------
    variable : str
        Variable name, used to look up the table defaults.
    downscaling_method : DownscalingMethod
        Which defaults table to start from, ``"BCSD"`` or ``"QDMSD"``.
    run_wide : dict or None
        Values applied to every variable in the run.
    overrides : dict[str, dict] or None
        Mapping of variable name to per-variable values.

    Returns
    -------
    VariableConfig
        Fully resolved and validated config for ``variable``.
    """
    run_wide = run_wide or {}
    var_overrides = (overrides or {}).get(variable, {})
    merged = VariableConfig.for_variable(variable, downscaling_method).model_dump()
    for source in (run_wide, var_overrides):
        merged.update({k: v for k, v in source.items() if v is not None})
    return VariableConfig(**merged)


def _parse_variable_overrides(items: list[str]) -> dict[str, dict[str, str]]:
    """
    Parse repeated ``--variable-override`` values into a nested mapping.

    Each item has the form ``{variable}:{field}={value}``, split on the first ``:``
    and then the first ``=``. Repeating the flag accumulates, both across variables
    and within one variable.

    Parameters
    ----------
    items : list[str]
        Raw flag values, e.g. ``["dtr:debias_approach=nonparametric"]``.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping of variable name to field-value pairs. Values stay strings and are
        coerced by pydantic during ``VariableConfig`` construction.

    Raises
    ------
    ValueError
        If an item does not match ``variable:field=value``.
    """
    parsed: dict[str, dict[str, str]] = defaultdict(dict)
    for item in items:
        variable, sep, rest = item.partition(":")
        field, sep2, value = rest.partition("=")
        if not (sep and sep2 and variable.strip() and field.strip()):
            raise ValueError(f"--variable-override must be 'variable:field=value', got {item!r}")
        parsed[variable.strip()][field.strip()] = value.strip()
    return dict(parsed)


def _is_matrix_config(config_dict: dict) -> bool:
    """Return True if any expandable field contains a list."""
    return any(
        isinstance(config_dict.get(plural, config_dict.get(singular)), list)
        for plural, singular, _ in _MATRIX_FIELDS
    )


def _reject_debias_approach_across_methods(
    methods: list,
    run_wide_debias_approach: str | None,
    overrides: dict[str, dict] | None,
) -> None:
    """
    Reject an explicit ``debias_approach`` when the matrix spans more than one method.

    ``BCSDConfig`` pins ``debias_approach='qdm'`` to ``downscaling_method='QDMSD'`` and
    forbids it under ``'BCSD'``, so any single explicit value contradicts one arm of a
    multi-method product. Raising here names the matrix as the cause. Letting the
    cross-field invariant fire instead produces an error that reads like a typo in one
    config rather than an unsatisfiable combination.

    Parameters
    ----------
    methods : list
        The resolved ``downscaling_method`` axis.
    run_wide_debias_approach : str or None
        A ``debias_approach`` applied to every variable in the run.
    overrides : dict[str, dict] or None
        Per-variable overrides, checked for their own ``debias_approach`` entries.
    """
    if len(methods) <= 1:
        return
    remedy = (
        "'qdm' requires 'QDMSD' and 'QDMSD' requires 'qdm', so one method in the matrix "
        "would always contradict it. Drop the value and let each method's defaults table "
        "supply it, or run one method at a time."
    )
    if run_wide_debias_approach:
        raise ValueError(
            f"Cannot combine a run-wide 'debias_approach' ({run_wide_debias_approach!r}) "
            f"with multiple downscaling methods ({methods}). {remedy}"
        )
    flagged = sorted(v for v, fields in (overrides or {}).items() if fields.get("debias_approach"))
    if flagged:
        raise ValueError(
            f"Cannot set 'debias_approach' in variable_overrides for {flagged} with "
            f"multiple downscaling methods ({methods}). {remedy}"
        )


def _expand_matrix_config(config_dict: dict) -> list[BCSDConfig]:
    """Expand a matrix config dict into one BCSDConfig per cartesian-product combination."""
    d = dict(config_dict)
    axes: dict[str, list] = {}
    for plural, singular, field in _MATRIX_FIELDS:
        # Both spellings name the same axis. Popping only the winner would leave the
        # loser in `d`, where it reaches the constructor as a second value for a keyword
        # the product loop already passes.
        if plural in d and singular in d:
            raise ValueError(
                f"Config sets both {plural!r} and {singular!r}, which are two spellings of "
                f"the same axis. Keep {plural!r} for a list of values or {singular!r} for a "
                "single one, not both."
            )
        if plural in d:
            val = d.pop(plural)
        elif singular in d:
            val = d.pop(singular)
        else:
            val = None
        axes[field] = val if isinstance(val, list) else [val]

    overrides = d.pop("variable_overrides", None) or {}
    if overrides:
        _validate_variable_overrides(overrides, axes["variable"])

    # A top-level 'debias_approach' applies to every variable in the matrix, e.g.
    # switching a whole run between standard quantile mapping and quantile delta
    # mapping. It has to be popped out of `d` here rather than left for BCSDConfig,
    # which rejects 'debias_approach' at the top level since it's a per-variable
    # setting everywhere else.
    run_wide_debias_approach = d.pop("debias_approach", None)
    run_wide = {"debias_approach": run_wide_debias_approach} if run_wide_debias_approach else {}

    # 'downscaling_method' was popped into `axes` with the other expandable fields, so
    # unlike them it has to be handed back to the constructor explicitly below. Each
    # entry also picks which per-variable defaults table _resolve_variable_config starts
    # from, which is why variable_config is resolved inside the product loop.
    methods = axes["downscaling_method"]
    _reject_debias_approach_across_methods(methods, run_wide_debias_approach, overrides)

    if "variable_config" in d:
        if len(methods) > 1:
            raise ValueError(
                "Cannot use 'variable_config' with multiple 'downscaling_methods' "
                f"({methods}). An explicit 'variable_config' is passed through verbatim to "
                "every combination, and its 'debias_approach' can only agree with one "
                "method. Drop 'variable_config' and let each method's defaults table supply "
                "it, or run one method at a time."
            )
        if len(axes["variable"]) > 1:
            raise ValueError(
                "Cannot use 'variable_config' in a matrix config with multiple variables "
                f"({axes['variable']}). Use 'variable_overrides' to set per-variable values, "
                "or split into separate config files."
            )
        if overrides:
            raise ValueError(
                "Cannot combine 'variable_config' with 'variable_overrides'. An explicit "
                "'variable_config' is passed through verbatim, so the overrides would be "
                "silently discarded. Fold the override values into 'variable_config', or "
                "drop 'variable_config' and use 'variable_overrides' alone."
            )
        if run_wide_debias_approach:
            raise ValueError(
                "Cannot combine 'variable_config' with a top-level 'debias_approach'. An "
                "explicit 'variable_config' is passed through verbatim, so the top-level "
                "value would be silently discarded. Fold 'debias_approach' into "
                "'variable_config', or drop 'variable_config' and use the top-level key alone."
            )

    configs = []
    for gcm, variable, member, scenario, method in itertools.product(
        axes["gcm"],
        axes["variable"],
        axes["ensemble_member"],
        axes["scenario"],
        axes["downscaling_method"],
    ):
        kwargs = dict(d)
        # When the key is missing, leave both downscaling_method and variable_config out
        # of the kwargs so BCSDConfig raises the single "downscaling_method is required"
        # error rather than a table lookup failure or a None-is-not-a-valid-method error.
        if method is not None:
            kwargs["downscaling_method"] = method
            if "variable_config" not in kwargs:
                kwargs["variable_config"] = _resolve_variable_config(
                    variable, method, run_wide, overrides
                )
        configs.append(
            BCSDConfig(
                gcm=gcm,
                variable=variable,
                ensemble_member=member,
                scenario=scenario,
                **kwargs,
            )
        )
    return configs


def load_configs(config_path: str) -> tuple[list[BCSDConfig], PipelineOptions]:
    """
    Load configuration(s) from YAML file or directory.

    The same flat YAML is parsed into both BCSDConfig (run identity) and
    PipelineOptions (operational settings). Unknown keys are silently ignored
    by each class via extra="ignore".

    Parameters
    ----------
    config_path : str
        Path to YAML config file or directory containing YAML files

    Returns
    -------
    tuple[list[BCSDConfig], PipelineOptions]
        Loaded run configs and operational options (from the first YAML file).
    """
    path = Path(config_path)
    configs = []
    options: PipelineOptions | None = None

    if path.is_file():
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        options = PipelineOptions(**config_dict)
        if _is_matrix_config(config_dict):
            configs.extend(_expand_matrix_config(config_dict))
        else:
            configs.append(BCSDConfig(**config_dict))

    elif path.is_dir():
        for yaml_file in sorted([*path.rglob("*.yaml"), *path.rglob("*.yml")]):
            with open(yaml_file) as f:
                config_dict = yaml.safe_load(f)
            if options is None:
                options = PipelineOptions(**config_dict)
            if _is_matrix_config(config_dict):
                configs.extend(_expand_matrix_config(config_dict))
            else:
                configs.append(BCSDConfig(**config_dict))

    else:
        raise ValueError(f"Config path does not exist: {config_path}")

    if not configs:
        raise ValueError(f"No valid configs found in: {config_path}")

    return configs, options or PipelineOptions()


def configs_from_matrix(
    gcms: list[str],
    variables: list[str],
    members: list[str],
    scenarios: list[str | None],
    *,
    train_period_start: int = 1978,
    train_period_end: int = 2014,
    predict_period_start: int | None = None,
    predict_period_end: int | None = None,
    scratch_dir: str = "s3://carbonplan-srm/scratch/cache/",
    output_dir: str = "s3://carbonplan-srm/scratch/output/",
    environment: str = "qa",
    branch: str = "main",
    subset_bounds: tuple[float, float, float, float] | None = None,
    save_intermediate: bool = False,
    downscaling_methods: list[DownscalingMethod],
    debias_approach: str | None = None,
    verbose: bool = False,
    # VariableConfig overrides (None = use per-variable default)
    detrend_data: bool | None = None,
    do_windowing: bool | None = None,
    running_window_length: int | None = None,
    running_window_step_length: int | None = None,
    disaggregation_method: str | None = None,
    disaggregation_clim_method: str | None = None,
    detrend_method: str | None = None,
    variable_overrides: dict[str, dict] | None = None,
) -> tuple[list[BCSDConfig], PipelineOptions]:
    """
    Generate BCSDConfig objects for every cartesian-product combination of GCMs,
    variables, ensemble members, scenarios, and downscaling methods.

    Parameters
    ----------
    gcms : list[str]
        GCM names (e.g., ["CESM2-WACCM", "MIROC-ES2H"])
    variables : list[str]
        Variables to downscale (e.g., ["tas", "pr"])
    members : list[str]
        Ensemble member labels (e.g., ["r1i1p1f1", "r2i1p1f1"])
    scenarios : list[str | None]
        Scenario names. Pass [None] for historical-only runs.
    train_period_start : int
        Start year of training period
    train_period_end : int
        End year of training period
    predict_period_start : int | None
        Start year of prediction period. Required when scenarios contains non-None values.
    predict_period_end : int | None
        End year of prediction period. Required when scenarios contains non-None values.
    scratch_dir : str
        Base directory for cached intermediate artifacts
    output_dir : str
        Directory for final downscaled outputs
    environment : str
        Environment name (qa, production)
    branch : str
        icechunk output branch (default: "main")
    subset_bounds : tuple[float, float, float, float] | None
        Spatial bounds as (lat_min, lat_max, lon_min, lon_max)
    save_intermediate : bool
        Save intermediate artifacts (detrended, debiased, etc.) to cache
    downscaling_methods : list[DownscalingMethod]
        Downscaling methods to expand over: "BCSD" (detrend + quantile mapping) and
        "QDMSD" (quantile delta mapping). Each entry selects a per-variable defaults
        table, so the matrix gains a fifth axis. Required, with no default.
    debias_approach : str | None
        Override VariableConfig.debias_approach for every variable (parametric,
        nonparametric, nonparametric_hybrid, nonparametric_hybrid_2sided, qdm). None
        uses each variable's own default. Cannot be combined with more than one entry
        in ``downscaling_methods``.
    verbose : bool
        Enable verbose logging
    detrend_data : bool | None
        Override VariableConfig.detrend_data
    do_windowing : bool | None
        Override VariableConfig.do_windowing
    running_window_length : int | None
        Override VariableConfig.running_window_length
    running_window_step_length : int | None
        Override VariableConfig.running_window_step_length
    disaggregation_method : str | None
        Override VariableConfig.disaggregation_method (additive, multiplicative)
    disaggregation_clim_method : str | None
        Override VariableConfig.disaggregation_clim_method (simple, fft)
    detrend_method : str | None
        Override VariableConfig.detrend_method (additive, multiplicative)
    variable_overrides : dict[str, dict] | None
        Per-variable overrides keyed by variable name, e.g.
        ``{"dtr": {"debias_approach": "nonparametric"}}``. Takes precedence over the
        run-wide parameters above.

    Returns
    -------
    list[BCSDConfig]
        One config per cartesian-product combination.
    """
    options = PipelineOptions(
        scratch_dir=scratch_dir,
        output_dir=output_dir,
        environment=environment,
        branch=branch,
        verbose=verbose,
        save_intermediate=save_intermediate,
    )
    variable_overrides = variable_overrides or {}
    if variable_overrides:
        _validate_variable_overrides(variable_overrides, variables)
    _reject_debias_approach_across_methods(downscaling_methods, debias_approach, variable_overrides)

    run_wide = {
        "debias_approach": debias_approach,
        "detrend_data": detrend_data,
        "do_windowing": do_windowing,
        "running_window_length": running_window_length,
        "running_window_step_length": running_window_step_length,
        "disaggregation_method": disaggregation_method,
        "disaggregation_clim_method": disaggregation_clim_method,
        "detrend_method": detrend_method,
    }

    configs = []
    for gcm, variable, member, scenario, method in itertools.product(
        gcms, variables, members, scenarios, downscaling_methods
    ):
        configs.append(
            BCSDConfig(
                gcm=gcm,
                variable=variable,
                ensemble_member=member,
                scenario=scenario,
                train_period_start=train_period_start,
                train_period_end=train_period_end,
                predict_period_start=predict_period_start,
                predict_period_end=predict_period_end,
                subset_bounds=subset_bounds,
                downscaling_method=method,
                variable_config=_resolve_variable_config(
                    variable, method, run_wide, variable_overrides
                ),
            )
        )
    return configs, options


#: CLI stage aliases mapped to pipeline stage names. "all"/None means every stage.
_STAGE_ALIASES: dict[str, str] = {
    "obs": "prepare_observations",
    "prepare_observations": "prepare_observations",
    "historical": "fit_historical",
    "fit_historical": "fit_historical",
    "scenario": "transform_scenario",
    "transform_scenario": "transform_scenario",
}


def _resolve_stage(stage: str | None) -> str | None:
    """Map a ``--stage`` value to a pipeline stage name, or None for the whole workflow.

    Raises
    ------
    typer.BadParameter
        For an unrecognized value. The dispatch below is an if/elif chain with no else,
        so an unknown stage used to run nothing and exit zero.
    """
    if stage is None or stage == "all":
        return None
    try:
        return _STAGE_ALIASES[stage]
    except KeyError:
        raise typer.BadParameter(
            f"Unknown stage {stage!r}; expected one of {sorted(set(_STAGE_ALIASES) | {'all'})}"
        ) from None


def _confirm_cost(
    orchestrator: BCSDOrchestrator,
    configs: list[BCSDConfig],
    executor: str | None,
    force: bool,
    assume_yes: bool,
    stage: str | None = None,
) -> bool:
    """Show what a run will cost and, when a human is watching, ask before dispatching.

    The prompt is skipped when stdin is not a terminal, because every deploy job runs
    non-interactively and would otherwise block until its timeout. The plan is still
    logged there, so the estimate lands in the job output either way.

    Returns
    -------
    bool
        True to proceed with the run.
    """
    if (executor or orchestrator.options.executor) == "local":
        return True
    if not _render_cost_plan(orchestrator, configs, executor, force, stage=stage):
        return True

    if assume_yes:
        return True
    if not sys.stdin.isatty():
        logger.info("Non-interactive session; proceeding without confirmation.")
        return True
    return typer.confirm("Submit these tasks?", default=False)


def _render_cost_plan(
    orchestrator: BCSDOrchestrator,
    configs: list[BCSDConfig],
    executor: str | None,
    force: bool,
    stage: str | None = None,
) -> bool:
    """Print the per-stage cost table. Returns False when there is nothing to submit.

    ``stage`` restricts the estimate to a single stage, matching ``run --stage``; pricing
    all three would overstate what the invocation is about to spend.
    """
    executor_name = executor or orchestrator.options.executor

    only = _resolve_stage(stage)
    if only is None:
        plans = [p for p in orchestrator.plan(configs, force=force) if p.to_run]
    else:
        # submit_stage receives the caller's list untouched, so price it the same way.
        plans = [p for p in [orchestrator.plan_stage(only, configs, force=force)] if p.to_run]
    if not plans:
        console.print("[green]Everything is already cached; nothing to submit.[/green]")
        return False

    estimate = estimate_workflow(
        [(p.stage, p.to_run, p.vm_type, p.regional) for p in plans], executor_name
    )

    table = Table(title=f"Cost estimate ({executor_name})", box=box.SIMPLE, title_justify="left")
    table.add_column("Stage", style="cyan", no_wrap=True)
    table.add_column("Tasks", justify="right")
    table.add_column("Instance", style="magenta", no_wrap=True)
    table.add_column("$/h", justify="right", style="yellow")
    table.add_column("Hours", justify="right")
    table.add_column("Cost", justify="right", style="yellow")
    for plan, wave in zip(plans, estimate.waves):
        skipped = f" (+{plan.cached} cached)" if plan.cached else ""
        table.add_row(
            plan.stage,
            f"{plan.to_run}{skipped}",
            plan.vm_type,
            f"${wave.burn_rate:,.2f}",
            f"{wave.low_hours:.2g}-{wave.high_hours:.2g}",
            f"${wave.low_cost:,.0f}-${wave.high_cost:,.0f}",
        )
    console.print(table)
    console.print(
        f"  Peak burn [yellow]${estimate.peak_burn_rate:,.2f}/hour[/yellow] while running "
        "[dim](exact)[/dim]"
    )
    console.print(
        f"  Total     [yellow]${estimate.low_cost:,.0f}-${estimate.high_cost:,.0f}[/yellow] "
        "[dim](rough: duration estimated from past runs)[/dim]"
    )
    return True


@app.command()
def run(
    config_path: list[str] = typer.Option(
        ..., help="Path to YAML config or directory of configs (can be specified multiple times)"
    ),
    stage: str = typer.Option(None, help="Run specific stage: obs, historical, scenario, or all"),
    force: bool = typer.Option(False, help="Force recompute even if cached"),
    executor: str = typer.Option(
        None,
        "--executor",
        help="Where tasks run: coiled, aws-batch, or local. Defaults to the config value.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the cost confirmation prompt.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show the cost plan and exit without submitting anything"
    ),
    branch: str | None = typer.Option(
        None, "--branch", help="Override the output icechunk branch (e.g. 'v2')"
    ),
    save_intermediate: bool = typer.Option(
        False, "--save-intermediate", help="Save and display intermediate artifacts"
    ),
):
    """Run BCSD pipeline with automatic caching and resumability"""

    # Load configs
    loaded = [load_configs(path) for path in config_path]
    configs = [cfg for cfgs, _ in loaded for cfg in cfgs]
    options = loaded[0][1] if loaded else PipelineOptions()
    updates: dict = {}
    if branch is not None:
        updates["branch"] = branch
    if save_intermediate:
        updates["save_intermediate"] = True
    options = options.model_copy(update=updates)
    _resolve_stage(stage)
    logger.info("Loaded %d configuration(s)", len(configs))
    _print_lineage_summary(configs)
    _validate_lineage_members(configs)
    _validate_predict_periods(configs)

    orchestrator = BCSDOrchestrator(options)

    if dry_run:
        _render_cost_plan(orchestrator, configs, executor, force, stage=stage)
        return

    if not _confirm_cost(orchestrator, configs, executor, force, yes, stage=stage):
        console.print("[yellow]Aborted; nothing was submitted.[/yellow]")
        raise typer.Exit(1)

    cache = orchestrator._get_cache()

    if stage == "obs" or stage == "prepare_observations":
        paths = orchestrator.submit_stage(
            "prepare_observations", configs, force=force, executor=executor
        )
        _print_paths_summary(paths, configs, "prepare_observations", cache)

    elif stage == "historical" or stage == "fit_historical":
        paths = orchestrator.submit_stage("fit_historical", configs, force=force, executor=executor)
        _print_paths_summary(paths, configs, "fit_historical", cache)
        _print_paths_summary(
            _coarse_hist_paths(configs, cache), configs, "debiased_coarse_historical", cache
        )

    elif stage == "scenario" or stage == "transform_scenario":
        paths = orchestrator.submit_stage(
            "transform_scenario", configs, force=force, executor=executor
        )
        _print_paths_summary(paths, configs, "transform_scenario", cache)
        _print_paths_summary(
            _coarse_scenario_paths(configs, cache), configs, "debiased_coarse_scenario", cache
        )

    elif stage == "all" or stage is None:
        all_paths = orchestrator.run_full_workflow(configs, force=force, executor=executor)
        obs_configs = orchestrator._deduplicate_obs_configs(configs)
        hist_configs = orchestrator._deduplicate_historical_configs(configs)
        _print_paths_summary(
            all_paths["prepare_observations"], obs_configs, "prepare_observations", cache
        )
        _print_paths_summary(all_paths["fit_historical"], hist_configs, "fit_historical", cache)
        _print_paths_summary(
            _coarse_hist_paths(hist_configs, cache),
            hist_configs,
            "debiased_coarse_historical",
            cache,
        )
        _print_paths_summary(all_paths["transform_scenario"], configs, "transform_scenario", cache)
        _print_paths_summary(
            _coarse_scenario_paths(configs, cache), configs, "debiased_coarse_scenario", cache
        )
        if options.save_intermediate:
            _print_intermediate_summary(cache)

    else:
        raise ValueError(f"Unknown stage: {stage}")

    logger.info("✓ Complete!")


def _print_intermediate_summary(cache: ArtifactCache) -> None:
    """Print a tree of intermediate artifacts on the current branch."""
    stores = cache.list_intermediate_groups()
    if not stores:
        return

    n_total = sum(len(groups) for groups in stores.values())
    console.print(f"\n[bold]Intermediates[/bold] ({n_total} artifact(s))")
    for store_path, groups in stores.items():
        tree = Tree(f"[cyan]{store_path}[/cyan] [dim](branch: {cache.branch})[/dim]")
        for group in groups:
            _insert_group_path(tree, group.split("/"))
        console.print(tree)


def _insert_group_path(node: Tree, segments: list[str]) -> None:
    """Recursively insert path segments into a Rich Tree, reusing existing nodes."""
    if not segments:
        return
    label = segments[0]
    for child in node.children:
        if child.label == label:
            _insert_group_path(child, segments[1:])
            return
    _insert_group_path(node.add(label), segments[1:])


def _coarse_hist_paths(configs: list[BCSDConfig], cache: ArtifactCache) -> list[str]:
    """Compute debiased_coarse_historical StoreLocation paths for each config."""
    paths = []
    for config in configs:
        cache.config = config
        hist_member = BCSDOrchestrator._resolve_hist_member(config)
        loc = cache.debiased_coarse_historical_loc(hist_member)
        paths.append(f"{loc.store_path}::{loc.group}")
    return paths


def _coarse_scenario_paths(configs: list[BCSDConfig], cache: ArtifactCache) -> list[str]:
    """Compute debiased_coarse_scenario StoreLocation paths for each config."""
    paths = []
    for config in configs:
        cache.config = config
        loc = cache.debiased_coarse_scenario_loc()
        paths.append(f"{loc.store_path}::{loc.group}")
    return paths


def _print_paths_summary(
    paths: list[str],
    _configs: list[BCSDConfig],
    stage: str,
    cache: ArtifactCache | None = None,
) -> None:
    """Print output paths produced by a stage as a nested tree grouped by store."""
    stage_label = {
        "prepare_observations": "Obs Regridded",
        "fit_historical": "Historical",
        "transform_scenario": "Scenario",
        "debiased_coarse_historical": "Debiased Coarse Historical",
        "debiased_coarse_scenario": "Debiased Coarse Scenario",
    }.get(stage, stage)

    n_artifacts = sum(1 for p in paths if p is not None)
    n_failed = sum(1 for p in paths if p is None)
    label = f"\n[bold]{stage_label}[/bold] ({n_artifacts} artifact(s)"
    if n_failed:
        label += f", [red]{n_failed} FAILED[/red]"
    label += ")"
    console.print(label)

    # Group groups by store path, preserving insertion order.
    stores: dict[str, list[str]] = {}
    for path in paths:
        if path is None:
            stores.setdefault("FAILED", []).append("")
        elif "::" in path:
            store, group = path.split("::", 1)
            stores.setdefault(store, []).append(group)
        else:
            stores.setdefault(path, []).append("")

    for store, groups in stores.items():
        branch = ""
        if cache is not None:
            branch = f" [dim](branch: {cache.branch})[/dim]"
        tree = Tree(f"[cyan]{store}[/cyan]{branch}")
        for group in groups:
            if group:
                _insert_group_path(tree, group.split("/"))
            else:
                tree.add("[red]FAILED[/red]")
        console.print(tree)


@app.command()
def run_matrix(
    gcm: list[str] = typer.Option(
        ..., help="GCM name (repeatable: --gcm CESM2-WACCM --gcm MIROC-ES2H)"
    ),
    variable: list[str] = typer.Option(
        ..., help="Variable to downscale (repeatable: --variable tas --variable pr)"
    ),
    member: list[str] = typer.Option(
        ..., help="Ensemble member label (repeatable: --member r1i1p1f1 --member r2i1p1f1)"
    ),
    scenario: list[str] | None = typer.Option(
        None,
        help=(
            "Scenario (repeatable: --scenario SSP245 --scenario G6-1.5K). "
            "Omit for historical-only runs."
        ),
    ),
    train_period_start: int = typer.Option(1978, help="Start year of training period"),
    train_period_end: int = typer.Option(2014, help="End year of training period"),
    predict_period_start: int | None = typer.Option(
        None, help="Start year of prediction period. Required when --scenario is provided."
    ),
    predict_period_end: int | None = typer.Option(
        None, help="End year of prediction period. Required when --scenario is provided."
    ),
    scratch_dir: str = typer.Option(
        "s3://carbonplan-srm/scratch/cache/", help="Base directory for cached artifacts"
    ),
    output_dir: str = typer.Option(
        "s3://carbonplan-srm/scratch/output/", help="Directory for final outputs"
    ),
    environment: str = typer.Option("qa", help="Environment (qa, production)"),
    branch: str = typer.Option("main", help="icechunk output branch (e.g. 'v2', 'v3')"),
    subset_bounds: str | None = typer.Option(
        None,
        help="Spatial bounds as 'lat_min,lat_max,lon_min,lon_max' (e.g. '-35,-22,16,33')",
    ),
    stage: str | None = typer.Option(
        None, help="Run specific stage: obs, historical, scenario, or all"
    ),
    force: bool = typer.Option(False, help="Force recompute even if cached"),
    executor: str = typer.Option(
        None,
        "--executor",
        help="Where tasks run: coiled, aws-batch, or local. Defaults to the config value.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the cost confirmation prompt.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show configs without executing"),
    save_intermediate: bool = typer.Option(
        False,
        "--save-intermediate",
        help="Save intermediate artifacts (detrended, debiased, etc.) to cache",
    ),
    downscaling_method: list[str] = typer.Option(
        ...,
        "--downscaling-method",
        help=(
            "Downscaling method: BCSD (detrend + quantile mapping) or QDMSD (quantile "
            "delta mapping). Selects the per-variable defaults table. Repeatable to "
            "expand the matrix over both: --downscaling-method BCSD --downscaling-method "
            "QDMSD. Required."
        ),
    ),
    debias_approach: str | None = typer.Option(
        None,
        "--debias-approach",
        help=(
            "Debias approach for every variable: parametric, nonparametric, "
            "nonparametric_hybrid, nonparametric_hybrid_2sided, qdm. "
            "Omit to use each variable's default. Override one variable with "
            "--variable-override."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    # VariableConfig overrides
    detrend_data: bool | None = typer.Option(
        None, "--detrend-data/--no-detrend-data", help="Override detrend_data for all variables"
    ),
    do_windowing: bool | None = typer.Option(
        None, "--do-windowing/--no-do-windowing", help="Override do_windowing for all variables"
    ),
    running_window_length: int | None = typer.Option(
        None, "--running-window-length", help="Override running_window_length for all variables"
    ),
    running_window_step_length: int | None = typer.Option(
        None,
        "--running-window-step-length",
        help="Override running_window_step_length for all variables",
    ),
    disaggregation_method: str | None = typer.Option(
        None,
        "--disaggregation-method",
        help="Override disaggregation_method (additive, multiplicative)",
    ),
    disaggregation_clim_method: str | None = typer.Option(
        None,
        "--disaggregation-clim-method",
        help="Override disaggregation_clim_method (simple, fft)",
    ),
    detrend_method: str | None = typer.Option(
        None, "--detrend-method", help="Override detrend_method (additive, multiplicative)"
    ),
    variable_override: list[str] = typer.Option(
        [],
        "--variable-override",
        help=(
            "Per-variable setting as 'variable:field=value' (repeatable), e.g. "
            "'dtr:debias_approach=nonparametric'. Takes precedence over the run-wide "
            "flags above."
        ),
    ),
):
    """Run the pipeline over the cartesian product of GCMs, variables, members,
    scenarios, and downscaling methods.

    Rather than pre-generating config files, specify each dimension as a repeatable
    option and the CLI will run every combination.

    Example (2 GCMs x 2 variables x 3 members x 2 scenarios x 1 method = 24 runs):

        bcsd run-matrix \\
          --gcm CESM2-WACCM --gcm MIROC-ES2H \\
          --variable tas --variable pr \\
          --member r1i1p1f1 --member r2i1p1f1 --member r3i1p1f1 \\
          --scenario SSP245 --scenario G6-1.5K \\
          --downscaling-method BCSD \\
          --predict-period-start 2015 --predict-period-end 2100

    Omit --scenario for historical-only runs.

    Repeat --downscaling-method to compare methods on identical inputs. The two
    methods share one regridded observation artifact and write to separate group
    prefixes, so nothing collides:

        bcsd run-matrix \\
          --gcm CESM2-WACCM --variable pr --member 003 --scenario SSP245 \\
          --downscaling-method BCSD --downscaling-method QDMSD \\
          --predict-period-start 2015 --predict-period-end 2100

    Give one variable a different setting with --variable-override
    (repeatable, 'variable:field=value'):

        bcsd run-matrix \\
          --gcm CESM2-WACCM \\
          --variable tasmax --variable dtr \\
          --member 007 --scenario ssp245 \\
          --downscaling-method BCSD \\
          --predict-period-start 2015 --predict-period-end 2100 \\
          --variable-override dtr:debias_approach=nonparametric
    """
    # Normalize: no --scenario given -> historical-only (scenario=None)
    scenario_values: list[str | None] = scenario if scenario else [None]

    # Validate predict periods are provided when scenarios are given
    has_scenarios = any(s is not None for s in scenario_values)
    if has_scenarios and (predict_period_start is None or predict_period_end is None):
        logger.error(
            "--predict-period-start and --predict-period-end are required "
            "when --scenario is specified."
        )
        raise typer.Exit(1)

    # Parse subset_bounds from "lat_min,lat_max,lon_min,lon_max" string
    parsed_bounds: tuple[float, float, float, float] | None = None
    if subset_bounds:
        try:
            parts = [float(x.strip()) for x in subset_bounds.split(",")]
            if len(parts) != 4:
                raise ValueError
            parsed_bounds = (parts[0], parts[1], parts[2], parts[3])
        except ValueError:
            logger.error(
                "--subset-bounds must be 'lat_min,lat_max,lon_min,lon_max' (e.g. '-35,-22,16,33')"
            )
            raise typer.Exit(1)

    try:
        parsed_overrides = _parse_variable_overrides(variable_override)
    except ValueError as exc:
        logger.error(str(exc))
        raise typer.Exit(1)

    try:
        configs, options = configs_from_matrix(
            gcms=gcm,
            variables=variable,
            members=member,
            scenarios=scenario_values,
            train_period_start=train_period_start,
            train_period_end=train_period_end,
            predict_period_start=predict_period_start,
            predict_period_end=predict_period_end,
            scratch_dir=scratch_dir,
            output_dir=output_dir,
            environment=environment,
            branch=branch,
            subset_bounds=parsed_bounds,
            save_intermediate=save_intermediate,
            downscaling_methods=downscaling_method,
            debias_approach=debias_approach,
            verbose=verbose,
            detrend_data=detrend_data,
            do_windowing=do_windowing,
            running_window_length=running_window_length,
            running_window_step_length=running_window_step_length,
            disaggregation_method=disaggregation_method,
            disaggregation_clim_method=disaggregation_clim_method,
            detrend_method=detrend_method,
            variable_overrides=parsed_overrides,
        )
    except (ValueError, ValidationError) as exc:
        logger.error(str(exc))
        raise typer.Exit(1)

    _resolve_stage(stage)
    _validate_lineage_members(configs)
    _validate_predict_periods(configs)

    n = len(configs)
    logger.info(
        "Generated %d configuration(s): %d GCM(s) x %d variable(s) x %d member(s) x %d scenario(s)",
        n,
        len(gcm),
        len(variable),
        len(member),
        len(scenario_values),
    )
    _print_lineage_summary(configs)

    if dry_run:
        table = Table(
            title=f"Matrix Configurations ({n})", show_header=True, header_style="bold magenta"
        )
        table.add_column("GCM", style="cyan")
        table.add_column("Variable", style="magenta")
        table.add_column("Member", justify="right")
        table.add_column("Scenario", style="yellow")
        # Without a Method column a two-method matrix prints identical rows, since every
        # other axis is shared between the pair.
        table.add_column("Method", style="green")
        for cfg in configs:
            table.add_row(
                cfg.gcm,
                cfg.variable,
                str(cfg.ensemble_member),
                cfg.scenario or "(historical)",
                cfg.downscaling_method,
            )
        console.print(table)
        return

    orchestrator = BCSDOrchestrator(options)

    if not _confirm_cost(orchestrator, configs, executor, force, yes, stage=stage):
        console.print("[yellow]Aborted; nothing was submitted.[/yellow]")
        raise typer.Exit(1)

    if stage == "obs" or stage == "prepare_observations":
        paths = orchestrator.submit_stage(
            "prepare_observations", configs, force=force, executor=executor
        )
        _print_paths_summary(paths, configs, "prepare_observations")
    elif stage == "historical" or stage == "fit_historical":
        paths = orchestrator.submit_stage("fit_historical", configs, force=force, executor=executor)
        _print_paths_summary(paths, configs, "fit_historical")
    elif stage == "scenario" or stage == "transform_scenario":
        paths = orchestrator.submit_stage(
            "transform_scenario", configs, force=force, executor=executor
        )
        _print_paths_summary(paths, configs, "transform_scenario")
    elif stage == "all" or stage is None:
        all_paths = orchestrator.run_full_workflow(configs, force=force, executor=executor)
        obs_configs = orchestrator._deduplicate_obs_configs(configs)
        hist_configs = orchestrator._deduplicate_historical_configs(configs)
        _print_paths_summary(all_paths["prepare_observations"], obs_configs, "prepare_observations")
        _print_paths_summary(all_paths["fit_historical"], hist_configs, "fit_historical")
        _print_paths_summary(all_paths["transform_scenario"], configs, "transform_scenario")
    else:
        logger.error("Unknown stage: %s", stage)
        raise typer.Exit(1)

    logger.info("✓ Complete!")


@app.command()
def status(
    config_path: list[str] = typer.Option(
        ..., help="Path to config(s) (can be specified multiple times)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed path information"),
    branch: str | None = typer.Option(
        None, "--branch", help="Override the output icechunk branch (e.g. 'v2')"
    ),
):
    """Check status of cached artifacts for given configs"""
    loaded = [load_configs(path) for path in config_path]
    configs = [cfg for cfgs, _ in loaded for cfg in cfgs]
    options = loaded[0][1] if loaded else PipelineOptions()
    if branch is not None:
        options = options.model_copy(update={"branch": branch})
    orchestrator = BCSDOrchestrator(options)

    # Show cache configuration if verbose
    if verbose and configs:
        cache = orchestrator._get_cache()
        config = configs[0]
        cache.config = config
        lines = [
            "Cache Configuration:",
            f"  Cache Path: {cache.scratch_dir}",
            f"  Output Path: {cache.output_dir or '(same as cache)'}",
            f"  Environment: {cache.environment}",
            f"  Branch: {cache.branch}",
            "Example Paths:",
            f"  Obs: {cache.obs_loc.store_path}",
            f"  Historical: {cache.historical_loc(config.ensemble_member).store_path}",
        ]
        if config.scenario:
            lines.append(f"  Scenario: {cache.scenario_loc.store_path}")
        logger.info("\n".join(lines))

    status_info = orchestrator.get_status(configs)

    # Create summary table
    table = Table(title="BCSD Pipeline Status", show_header=True, header_style="bold magenta")
    table.add_column("Stage", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Cached", justify="right", style="green")
    table.add_column("Missing", justify="right", style="red")
    table.add_column("Progress", justify="right")

    for stage_name, stage_info in status_info.items():
        total = stage_info["total"]
        cached = stage_info["cached"]
        missing = total - cached
        progress = f"{cached}/{total} ({100 * cached / total:.0f}%)" if total > 0 else "N/A"

        table.add_row(
            stage_name.replace("_", " ").title(),
            str(total),
            str(cached),
            str(missing),
            progress,
        )

    console.print(table)

    # Show missing items if any
    for stage_name, stage_info in status_info.items():
        if stage_info["missing"]:
            items = stage_info["missing"][:10]
            lines = [f"Missing {stage_name}:"] + [f"  • {run_id}" for run_id in items]
            if len(stage_info["missing"]) > 10:
                lines.append(f"  ... and {len(stage_info['missing']) - 10} more")
            logger.warning("\n".join(lines))


@app.command()
def cache_clear(
    config_path: str = typer.Option(
        "configs/example.yaml", "--config-path", "-c", help="Path to YAML config file"
    ),
    stage: str = typer.Option(None, help="Clear specific stage: obs, historical, scenarios"),
    gcm: str = typer.Option(None, help="Clear only specific GCM"),
    variable: str = typer.Option(None, help="Clear only specific variable"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Clear cached artifacts"""

    # Load config to get storage options
    _, options = load_configs(config_path)

    cache = ArtifactCache(
        scratch_dir=options.scratch_dir,
        environment=options.environment,
        branch=options.branch,
    )

    # Build description
    desc_parts = []
    if stage:
        desc_parts.append(f"stage={stage}")
    if gcm:
        desc_parts.append(f"gcm={gcm}")
    if variable:
        desc_parts.append(f"variable={variable}")

    desc = ", ".join(desc_parts) if desc_parts else "ALL artifacts"

    if not confirm:
        confirm = typer.confirm(f"Really delete {desc}?")
        if not confirm:
            logger.warning("Canceled")
            return

    deleted = cache.clear_cache(stage=stage, gcm=gcm, variable=variable)
    logger.info("✓ Deleted %d artifact(s)", deleted)


@app.command()
def cache_list(
    config_path: str = typer.Option(
        "configs/example.yaml", "--config-path", "-c", help="Path to YAML config file"
    ),
    stage: str = typer.Option(None, help="List specific stage: obs, historical, scenarios"),
    gcm: str = typer.Option(None, help="List only specific GCM"),
    variable: str = typer.Option(None, help="List only specific variable"),
):
    """List cached artifacts"""

    # Load config to get storage options
    _, options = load_configs(config_path)

    cache = ArtifactCache(
        scratch_dir=options.scratch_dir,
        environment=options.environment,
        branch=options.branch,
    )
    artifacts = cache.list_artifacts(stage=stage, gcm=gcm, variable=variable)

    if not artifacts:
        logger.warning("No cached artifacts found")
        return

    table = Table(title=f"Cached Artifacts ({len(artifacts)})", show_header=True)
    table.add_column("Path", style="cyan")

    for artifact in artifacts:
        table.add_row(artifact)

    console.print(table)


@app.command()
def release(
    config_path: str = typer.Option(
        "configs/example.yaml", "--config-path", "-c", help="Path to YAML config or directory"
    ),
    tag: str = typer.Option(..., "--tag", help="icechunk tag to create (e.g. 'v0.13.0')"),
    branch: str = typer.Option(
        None, "--branch", help="Branch to tag. Defaults to the installed package version."
    ),
):
    """Freeze the stores a config set writes to under an immutable icechunk tag.

    A branch stays writable, so a later run can overwrite the state a comparison
    baseline points at. Tagging pins the snapshot id, which is what makes a release
    artifact safe to cite as a baseline.
    """
    configs, options = load_configs(config_path)
    if not configs:
        logger.error("No configs found at %s", config_path)
        raise typer.Exit(1)

    # One config set can span several stores (one per gcm/obs/subset triple). Tag each
    # distinct store once; two configs sharing a store would otherwise collide on the
    # second create_tag call.
    seen: set[str] = set()
    tagged = 0
    for config in configs:
        cache = ArtifactCache.from_config(config, options)
        if branch:
            cache.branch = branch
        if cache._output_store in seen:
            continue
        seen.add(cache._output_store)
        try:
            # Logs one line per icechunk store it tags: the output store, plus the
            # scratch store when the two are configured to different paths.
            cache.release(tag)
        except Exception as exc:  # icechunk raises on an existing tag or a missing branch
            # A partially applied tag is possible here: release() tags output before
            # scratch, so a failure on the second leaves the first tagged. Exiting
            # non-zero is what matters, so the release does not report success.
            logger.error(
                "Could not tag %s@%s as %r: %s", cache._output_store, cache.branch, tag, exc
            )
            raise typer.Exit(1) from exc
        tagged += 1

    logger.info("✓ Released %d config store(s) as %r on branch %r", tagged, tag, options.branch)


@app.command()
def validate(
    config_path: list[str] | None = typer.Option(
        None,
        "--config-path",
        "-c",
        help="Path to YAML config or directory of configs (can be specified multiple times). "
        "Derives GCMs and scenarios to validate from the loaded configs.",
    ),
    gcm: list[str] | None = typer.Option(
        None, "--gcm", help="GCM(s) to validate (repeatable). Defaults to all."
    ),
    scenario: list[str] | None = typer.Option(
        None, "--scenario", help="Scenario(s) to validate (repeatable). Defaults to all."
    ),
    use_coiled: bool = typer.Option(
        True,
        "--coiled/--no-coiled",
        help="Run validation reductions on a Coiled Dask cluster instead of locally.",
    ),
    n_workers: int = typer.Option(
        4, "--n-workers", help="Coiled worker count (adaptive minimum when --adaptive-max is set)."
    ),
    worker_vm_type: str = typer.Option(
        "r8g.2xlarge", "--worker-vm-type", help="Coiled worker VM type."
    ),
    adaptive_max: int | None = typer.Option(
        None, "--adaptive-max", help="Enable adaptive scaling up to this many workers."
    ),
) -> None:
    """Validate input datasets against the validation matrix.

    Exits with code 1 if any blocking check fails, otherwise exits with code 0.

    When --config-path is given, GCMs and scenarios are derived from those configs.
    Otherwise, --gcm and --scenario filter the check matrix (defaulting to all known values).

    By default the datasets' lazy dask reductions (spatial-range and negative-precip checks)
    run on a short-lived Coiled Dask cluster; pass --no-coiled to run everything in-process.
    """
    import pydantic

    from srm.validation import (
        BLOCKING_CHECKS,
        GCM_OPTIONS,
        SCENARIO_OPTIONS,
        DatasetValidator,
        check_config_time_domain,
    )

    if config_path:
        loaded = [load_configs(path) for path in config_path]
        configs = [cfg for cfgs, _ in loaded for cfg in cfgs]
        gcm = list(dict.fromkeys(c.gcm for c in configs))
        scenario = list(
            dict.fromkeys(c.scenario if c.scenario is not None else "historical" for c in configs)
        )
        logger.info("Validating %d GCM(s) x %d scenario(s) from configs", len(gcm), len(scenario))

    pairs = [(g, s) for g in (gcm or GCM_OPTIONS) for s in (scenario or SCENARIO_OPTIONS)]

    # Reductions run on Coiled (see _validation_cluster) so the driver only handles scalars.
    all_results = []
    with _validation_cluster(
        use_coiled, n_workers=n_workers, worker_vm_type=worker_vm_type, adaptive_max=adaptive_max
    ):
        for g, s in pairs:
            try:
                all_results.extend(DatasetValidator(gcm=g, scenario=s).run_checks())
            except pydantic.ValidationError as exc:
                logger.error("Invalid input (gcm=%r, scenario=%r): %s", g, s, exc)

    # Per-member config time-domain checks (only when configs are supplied). Driver-only
    # metadata lookups, so kept outside the cluster block; merged into all_results after
    # rendering so blocking-failure aggregation picks them up.
    config_results = [check_config_time_domain(c) for c in configs] if config_path else []

    def _scenario_order(s: str) -> tuple[int, str]:
        if s == "historical":
            return (0, s)
        if s == "baseline":
            return (3, s)
        if "G6" in s or "SAI" in s:
            return (2, s)
        return (1, s)

    by_gcm: dict[str, list] = {}
    for r in all_results:
        by_gcm.setdefault(r.gcm, []).append(r)

    for gcm_name, gcm_results in by_gcm.items():
        console.rule(f"[bold]{gcm_name}[/bold]")

        scenarios = sorted(dict.fromkeys(r.scenario for r in gcm_results), key=_scenario_order)
        check_ids = list(dict.fromkeys(r.check_id for r in gcm_results))
        index = {(r.check_id, r.scenario): r for r in gcm_results}
        # Split: cross-all checks (no SKIP for any scenario) vs scenario-scoped (SKIP for some).
        table_checks = []
        scoped_checks = []
        for cid in check_ids:
            non_skip = [
                s
                for s in scenarios
                if (r := index.get((cid, s))) is not None and r.status != CheckStatus.SKIP
            ]
            if len(non_skip) == 1:
                scoped_checks.append(cid)
            else:
                table_checks.append(cid)

        if table_checks:
            console.print(_build_check_matrix_table(table_checks, scenarios, index))

        if scoped_checks:
            scoped_tbl = Table(
                title="Scenario-specific checks",
                show_header=True,
                header_style="dim",
                box=box.SIMPLE_HEAD,
                padding=(0, 1),
            )
            scoped_tbl.add_column("check", style="dim", no_wrap=True)
            scoped_tbl.add_column("scenario", style="dim")
            scoped_tbl.add_column("result", justify="center")
            for cid in scoped_checks:
                for s in scenarios:
                    r = index.get((cid, s))
                    if r and r.status != CheckStatus.SKIP:
                        scoped_tbl.add_row(cid, s, _STATUS_SYMBOL[r.status])
            if scoped_tbl.row_count:
                console.print(scoped_tbl)

        _print_validate_lineage_summary(gcm_name, scenarios)

    if config_results:
        cfg_tbl = Table(
            title="Config time-domain checks (per ensemble member)",
            show_header=True,
            header_style="dim",
            box=box.SIMPLE_HEAD,
            padding=(0, 1),
        )
        cfg_tbl.add_column("gcm", style="dim", no_wrap=True)
        cfg_tbl.add_column("scenario", style="dim")
        cfg_tbl.add_column("member", style="dim")
        cfg_tbl.add_column("result", justify="center")
        cfg_tbl.add_column("message", style="dim")
        for r in config_results:
            cfg_tbl.add_row(
                r.gcm,
                r.scenario,
                r.ensemble_member or "",
                _STATUS_SYMBOL[r.status],
                r.message,
            )
        console.print(cfg_tbl)

    # Merge after rendering so the gcm/scenario matrix above is unaffected.
    all_results.extend(config_results)

    blocking_failures = [
        r for r in all_results if r.status == CheckStatus.FAIL and r.check_id in BLOCKING_CHECKS
    ]
    single_pair = len(pairs) == 1

    if blocking_failures:
        logger.error("--- Blocking failures ---")
        for r in blocking_failures:
            logger.error("✗ %s (%s/%s): %s", r.check_id, r.gcm, r.scenario, r.message)
            if r.detail:
                console.print_json(json.dumps(r.detail))

    if single_pair:
        printed = {id(r) for r in blocking_failures}
        for r in all_results:
            if r.detail and id(r) not in printed:
                # console.rule/print_json used here for structured JSON detail display
                console.rule(
                    f"[dim]{r.check_id} detail[/dim] for {r.gcm}/{r.scenario}", style="dim"
                )
                console.print_json(json.dumps(r.detail))

    if blocking_failures:
        raise typer.Exit(1)


@app.command()
def validate_output(
    store_uris: list[str] | None = typer.Argument(
        None, help="One or more output datatree icechunk store URIs."
    ),
    config_path: list[str] | None = typer.Option(
        None,
        "--config-path",
        "-c",
        help="Path to YAML config or directory of configs (can be specified multiple times). "
        "Output store URIs are derived from the configs instead of passing store_uris directly.",
    ),
    branch: str | None = typer.Option(None, "--branch", help="Icechunk branch to read."),
    tag: str | None = typer.Option(None, "--tag", help="Icechunk tag to read."),
    scenario: list[str] | None = typer.Option(
        None, "--scenario", help="Scenario(s) to validate (repeatable). Defaults to all."
    ),
    variable: list[str] | None = typer.Option(
        None, "--variable", help="Variable(s) to validate (repeatable). Defaults to all."
    ),
    use_coiled: bool = typer.Option(
        True,
        "--coiled/--no-coiled",
        help="Run validation reductions on a Coiled Dask cluster instead of locally.",
    ),
    n_workers: int = typer.Option(
        4, "--n-workers", help="Coiled worker count (adaptive minimum when --adaptive-max is set)."
    ),
    worker_vm_type: str = typer.Option(
        "r8g.2xlarge", "--worker-vm-type", help="Coiled worker VM type."
    ),
    adaptive_max: int | None = typer.Option(
        None, "--adaptive-max", help="Enable adaptive scaling up to this many workers."
    ),
) -> None:
    """Validate output datatree store(s), one leaf (scenario/variable/member) at a time.

    Renders a single table per store. Exits with code 1 if any blocking check fails in any
    store, otherwise exits with code 0. --scenario and --variable restrict validation to
    matching subtrees (defaulting to the whole store).

    Store URIs can be given explicitly, or derived from the same config(s) used for
    `bcsd run` via --config-path; in the latter case --branch defaults to the branch
    those configs resolve to (the same branch `run` would write).

    When $GITHUB_STEP_SUMMARY is set, a markdown report is appended there in addition
    to the console tables.

    By default the store's lazy dask reductions run on a short-lived Coiled Dask cluster
    (the driver, tables, exit-code gate, and step summary stay local); only scalar results
    return to the driver. Pass ``--no-coiled`` to run everything in-process instead.
    local: `uv run bcsd validate-output <store_uri> [<store_uri> ...] --no-coiled`
    coiled: `uv run bcsd validate-output <store_uri> [<store_uri> ...]`

    """
    from srm.config import SCENARIO_TO_GROUP
    from srm.validation import (
        BLOCKING_CHECKS,
        parse_scenario,
        parse_variable,
        validate_output_store,
    )

    if bool(store_uris) == bool(config_path):
        raise typer.BadParameter(
            "Exactly one of store_uris or --config-path is required.",
            param_hint="'store_uris' / '--config-path'",
        )

    if config_path:
        loaded = [load_configs(p) for p in config_path]
        configs = [cfg for cfgs, _ in loaded for cfg in cfgs]
        options = loaded[0][1] if loaded else PipelineOptions()
        store_uris = sorted(
            {ArtifactCache.from_config(cfg, options).scenario_loc.store_path for cfg in configs}
        )
        if branch is None and tag is None:
            branch = options.branch

    provided = sum(x is not None for x in [branch, tag])
    if provided != 1:
        raise typer.BadParameter(
            "Exactly one of --branch or --tag is required.",
            param_hint="'--branch' / '--tag'",
        )

    # Translate/validate filters to on-disk group names (SSP245 -> ssp245; variables are
    # already canonical). parse_* raise on unknown values.
    scenarios = [SCENARIO_TO_GROUP[parse_scenario(s)] for s in scenario] if scenario else None
    variables = [parse_variable(v) for v in variable] if variable else None
    filtered = bool(scenarios or variables)

    summary_lines: list[str] = []
    any_blocking = False
    # Reductions run on Coiled (see _validation_cluster) so the driver only handles scalars;
    # the cluster tears down when this block exits, before the summary write / exit gate below.
    with _validation_cluster(
        use_coiled, n_workers=n_workers, worker_vm_type=worker_vm_type, adaptive_max=adaptive_max
    ):
        for store_uri in store_uris:
            results = validate_output_store(
                store_uri,
                branch=branch,
                tag=tag,
                scenarios=scenarios,
                variables=variables,
            )
            if not results:
                # An explicit filter matching nothing is an error, not an empty success.
                log = logger.error if filtered else logger.warning
                log("No populated leaves found in %s", store_uri)
                if filtered:
                    any_blocking = True
                continue

            console.rule(f"[bold]{store_uri}[/bold]")
            md_lines = [f"## {store_uri}\n", "| leaf | status | checks |", "| --- | --- | --- |"]

            # Group results by leaf (scenario path).
            leaf_results: dict[str, list[CheckResult]] = defaultdict(list)
            for r in results:
                leaf_results[r.scenario].append(r)

            tbl = Table(show_header=True, header_style="bold", box=box.SIMPLE_HEAD, padding=(0, 1))
            tbl.add_column("leaf", no_wrap=True)
            tbl.add_column("status", justify="center")
            tbl.add_column("checks", justify="right", style="dim")
            for leaf, leaf_rs in sorted(leaf_results.items()):
                n_total = len(leaf_rs)
                n_pass = sum(1 for r in leaf_rs if r.status != CheckStatus.FAIL)
                any_fail = any(r.status == CheckStatus.FAIL for r in leaf_rs)
                status_sym = "[red]✗[/red]" if any_fail else "[green]✓[/green]"
                checks_str = f"{n_pass}/{n_total}"
                tbl.add_row(leaf, status_sym, checks_str)
                md_lines.append(f"| {leaf} | {'❌' if any_fail else '✅'} | {checks_str} |")
            console.print(tbl)

            failures_by_leaf: dict[str, list[CheckResult]] = {
                leaf: [r for r in leaf_rs if r.status == CheckStatus.FAIL]
                for leaf, leaf_rs in leaf_results.items()
                if any(r.status == CheckStatus.FAIL for r in leaf_rs)
            }
            blocking_failures = [
                r
                for leaf_rs in failures_by_leaf.values()
                for r in leaf_rs
                if r.check_id in BLOCKING_CHECKS
            ]
            if blocking_failures:
                any_blocking = True

            if failures_by_leaf:
                n_fail_leaves = len(failures_by_leaf)
                n_fail_checks = sum(len(v) for v in failures_by_leaf.values())
                console.print(
                    f"\n[bold]Failures[/bold] ({n_fail_leaves} leaf{'s' if n_fail_leaves != 1 else ''}, {n_fail_checks} check{'s' if n_fail_checks != 1 else ''}):\n"
                )
                md_lines.append(
                    f"\n**Failures** ({n_fail_leaves} leaf{'s' if n_fail_leaves != 1 else ''}, "
                    f"{n_fail_checks} check{'s' if n_fail_checks != 1 else ''}):\n"
                )
                for leaf, fail_rs in sorted(failures_by_leaf.items()):
                    console.print(f"  [bold]{leaf}[/bold]")
                    md_lines.append(f"- **{leaf}**")
                    for r in fail_rs:
                        blocking_marker = (
                            " [dim](blocking)[/dim]" if r.check_id in BLOCKING_CHECKS else ""
                        )
                        console.print(
                            f"    [red]✗[/red] [dim]{r.check_id}[/dim]{blocking_marker}    {r.message}"
                        )
                        md_blocking_marker = " (blocking)" if r.check_id in BLOCKING_CHECKS else ""
                        md_lines.append(f"  - `{r.check_id}`{md_blocking_marker}: {r.message}")
                        if r.detail:
                            console.print_json(json.dumps(r.detail))
                    console.print()

            summary_lines.append("\n".join(md_lines))

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path and summary_lines:
        with open(summary_path, "a") as f:
            f.write("\n\n".join(summary_lines) + "\n")

    if any_blocking:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
