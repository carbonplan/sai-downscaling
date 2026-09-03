"""
Configuration classes for BCSD downscaling runs.

Defines :class:`BCSDConfig` (run identity fields that determine the cache key) and
:class:`PipelineOptions` (operational settings that do not affect computation). Both
extend :class:`pydantic_settings.BaseSettings` with a ``BCSD_`` env prefix and load
from the same flat YAML file.
"""

from __future__ import annotations

import hashlib
import os
from importlib.metadata import version as _pkg_version
from typing import ClassVar, Literal, get_args

import pydantic_settings
from packaging.version import Version as _Version
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

_cache_version = f"v{_Version(_pkg_version('srm')).public}"

DebiasApproach = Literal[
    "parametric", "nonparametric", "nonparametric_hybrid", "nonparametric_hybrid_2sided", "qdm"
]
DownscalingMethod = Literal["BCSD", "QDMSD"]
DisaggregationMethod = Literal["additive", "multiplicative"]
DisaggregationClimMethod = Literal["simple", "fft", "simple_rolling"]
DetrendMethod = Literal["additive", "multiplicative"]
VariableName = Literal["tas", "tasmax", "tasmin", "pr", "rsds", "dtr", "hurs"]

# Lowercase leading group-path segments that namespace method-dependent icechunk
# artifacts (see ``ArtifactCache._method_prefix``). Read-side tools use this to tell a
# method segment from a scenario group when descending a store, because stores written
# before the namespacing carry no such segment and must still be readable.
METHOD_SEGMENTS: frozenset[str] = frozenset(m.lower() for m in get_args(DownscalingMethod))


class VariableConfig(BaseModel):
    """Variable-specific BCSD configuration parameters"""

    detrend_data: bool
    do_windowing: bool
    running_window_length: int
    disaggregation_method: DisaggregationMethod
    disaggregation_clim_method: DisaggregationClimMethod
    disaggregation_tiny_threshold: float
    detrend_method: DetrendMethod
    debias_approach: DebiasApproach
    running_window_step_length: int

    # Spatial-disaggregation keys renamed to a 'disaggregation_' prefix when
    # 'downscaling_method' was repurposed as the top-level BCSD/QDMSD selector.
    # extra="ignore" would drop the old spellings silently, and one of them now means
    # something different at the top level, so a stale config has to fail loudly.
    _RENAMED_KEYS: ClassVar[dict[str, str]] = {
        "downscaling_method": "disaggregation_method",
        "downscaling_clim_method": "disaggregation_clim_method",
        "downscaling_tiny_threshold": "disaggregation_tiny_threshold",
    }

    @model_validator(mode="before")
    @classmethod
    def _reject_renamed_keys(cls, data):
        """Fail loudly if a config still uses the pre-rename spatial-disaggregation keys."""
        if isinstance(data, dict):
            for old, new in cls._RENAMED_KEYS.items():
                if old in data:
                    raise ValueError(
                        f"'{old}' was renamed to '{new}' on VariableConfig. "
                        f"Spatial-disaggregation settings now carry a 'disaggregation_' "
                        f"prefix, because 'downscaling_method' is the top-level "
                        f"BCSD/QDMSD selector on BCSDConfig. Leaving '{old}' here would "
                        f"be dropped silently, so rename it to '{new}'."
                    )
        return data

    @classmethod
    def for_variable(cls, variable: str, downscaling_method: DownscalingMethod) -> VariableConfig:
        """Load variable-specific defaults for ``variable`` under ``downscaling_method``.

        ``downscaling_method`` selects which table to read: ``BCSD`` uses
        ``BCSD_CONFIG`` (detrend, then standard quantile mapping), while ``QDMSD``
        uses ``QDMSD_CONFIG`` (quantile delta mapping, then spatial disaggregation).
        QDM carries the climate trend through its own quantile mapping, so it never
        needs the separate detrend/retrend step, and it maps over a wider seasonal
        window. Those are the differences between the two tables.

        Each row lists every field rather than leaning on class defaults, because a
        value that is uniform within one table is generally not uniform across both.

        Parameters
        ----------
        variable : str
            Variable name, one of :data:`VariableName`.
        downscaling_method : DownscalingMethod
            Which method table to read, ``"BCSD"`` or ``"QDMSD"``.

        Returns
        -------
        VariableConfig
            Validated defaults for ``variable`` under ``downscaling_method``.
        """
        BCSD_CONFIG = {
            "pr": {
                "detrend_data": False,
                "detrend_method": "multiplicative",
                "do_windowing": True,
                "disaggregation_method": "multiplicative",
                "disaggregation_clim_method": "fft",
                "disaggregation_tiny_threshold": 1.0e-6,  # kg m-2 s-1, ~0.086 mm/day
                "running_window_length": 31,
                "running_window_step_length": 1,
                "debias_approach": "nonparametric_hybrid_2sided",
            },
            "tas": {
                "detrend_data": True,
                "detrend_method": "additive",
                "do_windowing": True,
                "disaggregation_method": "additive",
                "disaggregation_clim_method": "fft",
                "disaggregation_tiny_threshold": 0.0,
                "running_window_length": 31,
                "running_window_step_length": 1,
                "debias_approach": "nonparametric_hybrid_2sided",
            },
            "tasmax": {
                "detrend_data": True,
                "detrend_method": "additive",
                "do_windowing": True,
                "disaggregation_method": "additive",
                "disaggregation_clim_method": "fft",
                "disaggregation_tiny_threshold": 0.0,
                "running_window_length": 31,
                "running_window_step_length": 1,
                "debias_approach": "nonparametric_hybrid_2sided",
            },
            "tasmin": {
                "detrend_data": True,
                "detrend_method": "additive",
                "do_windowing": True,
                "disaggregation_method": "additive",
                "disaggregation_clim_method": "fft",
                "disaggregation_tiny_threshold": 0.0,
                "running_window_length": 31,
                "running_window_step_length": 1,
                "debias_approach": "nonparametric_hybrid_2sided",
            },
            "rsds": {
                "detrend_data": False,
                "detrend_method": "multiplicative",
                "do_windowing": True,
                "disaggregation_method": "multiplicative",
                "disaggregation_clim_method": "fft",
                "disaggregation_tiny_threshold": 1.0,  # W m-2
                "running_window_length": 31,
                "running_window_step_length": 1,
                "debias_approach": "nonparametric",
            },
            "dtr": {
                "detrend_data": False,
                "detrend_method": "multiplicative",
                "do_windowing": True,
                "disaggregation_method": "multiplicative",
                "disaggregation_clim_method": "fft",
                "disaggregation_tiny_threshold": 0.0,
                "running_window_length": 31,
                "running_window_step_length": 1,
                "debias_approach": "nonparametric_hybrid_2sided",
            },
            "hurs": {
                "detrend_data": False,
                "detrend_method": "additive",
                "do_windowing": True,
                "disaggregation_method": "multiplicative",
                "disaggregation_clim_method": "fft",
                "disaggregation_tiny_threshold": 1.0e-2,  # percent
                "running_window_length": 31,
                "running_window_step_length": 1,
                "debias_approach": "nonparametric_hybrid_2sided",
            },
        }

        QDMSD_CONFIG = {
            "pr": {
                "detrend_data": False,
                "detrend_method": "multiplicative",
                "do_windowing": True,
                "disaggregation_method": "multiplicative",
                "disaggregation_clim_method": "fft",
                "disaggregation_tiny_threshold": 1.0e-6,  # kg m-2 s-1, ~0.086 mm/day
                "running_window_length": 91,
                "running_window_step_length": 31,
                "debias_approach": "qdm",
            },
            "tas": {
                "detrend_data": False,
                "detrend_method": "additive",
                "do_windowing": True,
                "disaggregation_method": "additive",
                "disaggregation_clim_method": "fft",
                "disaggregation_tiny_threshold": 0.0,
                "running_window_length": 91,
                "running_window_step_length": 31,
                "debias_approach": "qdm",
            },
            "tasmax": {
                "detrend_data": False,
                "detrend_method": "additive",
                "do_windowing": True,
                "disaggregation_method": "additive",
                "disaggregation_clim_method": "fft",
                "disaggregation_tiny_threshold": 0.0,
                "running_window_length": 91,
                "running_window_step_length": 31,
                "debias_approach": "qdm",
            },
            "tasmin": {
                "detrend_data": False,
                "detrend_method": "additive",
                "do_windowing": True,
                "disaggregation_method": "additive",
                "disaggregation_clim_method": "fft",
                "disaggregation_tiny_threshold": 0.0,
                "running_window_length": 91,
                "running_window_step_length": 31,
                "debias_approach": "qdm",
            },
            "rsds": {
                "detrend_data": False,
                "detrend_method": "multiplicative",
                "do_windowing": True,
                "disaggregation_method": "multiplicative",
                "disaggregation_clim_method": "fft",
                "disaggregation_tiny_threshold": 1.0,  # W m-2
                "running_window_length": 91,
                "running_window_step_length": 31,
                "debias_approach": "qdm",
            },
            "dtr": {
                "detrend_data": False,
                "detrend_method": "multiplicative",
                "do_windowing": True,
                "disaggregation_method": "multiplicative",
                "disaggregation_clim_method": "fft",
                "disaggregation_tiny_threshold": 0.0,
                "running_window_length": 91,
                "running_window_step_length": 31,
                "debias_approach": "qdm",
            },
            "hurs": {
                "detrend_data": False,
                "detrend_method": "additive",
                "do_windowing": True,
                "disaggregation_method": "multiplicative",
                "disaggregation_clim_method": "fft",
                "disaggregation_tiny_threshold": 1.0e-2,  # percent
                "running_window_length": 91,
                "running_window_step_length": 31,
                "debias_approach": "qdm",
            },
        }

        tables = {"BCSD": BCSD_CONFIG, "QDMSD": QDMSD_CONFIG}
        if downscaling_method not in tables:
            raise ValueError(
                f"Unknown downscaling_method: {downscaling_method}. Must be one of {sorted(tables)}"
            )
        table = tables[downscaling_method]

        if variable not in table:
            raise ValueError(f"Unknown variable: {variable}. Must be one of {list(table.keys())}")

        return cls(**table[variable])


class BCSDConfig(pydantic_settings.BaseSettings):
    """
    Run-identity configuration for BCSD downscaling pipeline.

    Captures the parameters that uniquely identify a BCSD run: model, variable,
    ensemble member, scenario, time periods, spatial subsetting, and bias-correction
    method. Operational settings (paths, flags) live in PipelineOptions.
    """

    # Model and data identifiers
    gcm: str = Field(..., description="GCM name (e.g., 'CESM2-WACCM', 'MIROC-ES2H', 'UKESM')")
    variable: VariableName = Field(..., description="Variable to downscale")
    ensemble_member: str = Field(..., description="Ensemble member label (e.g. 'r1i1p1f1', '01')")
    scenario: str | None = Field(
        None,
        description="Scenario name (e.g., 'SSP245', 'G6-1.5K'). None for historical-only runs.",
    )

    @field_validator("scenario", mode="before")
    @classmethod
    def normalize_scenario(cls, v: str | None) -> str | None:
        """Uppercase scenario so 'ssp245' and 'SSP245' are equivalent."""
        return v.upper() if v is not None else v

    # Time periods.
    # Ensure that the train period end and start fall between 1950 and 2014
    # The predict period can be anywhere from 1950 to 2100 because the
    # gcm simulations we're transforming can exist in that entire range

    train_period_start: int = Field(
        1978, ge=1950, le=2014, description="Start year of training period (historical)"
    )
    train_period_end: int = Field(
        2014, ge=1950, le=2014, description="End year of training period (historical)"
    )
    predict_period_start: int | None = Field(
        None,
        ge=2015,
        le=2100,
        description="Start year of prediction period. Required if scenario is specified.",
    )
    predict_period_end: int | None = Field(
        None,
        ge=2015,
        le=2100,
        description="End year of prediction period. Required if scenario is specified.",
    )

    # Spatial subsetting (optional)
    subset_bounds: tuple[float, float, float, float] | None = Field(
        None, description="Spatial bounds as (lat_min, lat_max, lon_min, lon_max). None for global."
    )

    obs_dataset: str = Field(
        "ERA5",
        description="Catalog key for observation dataset (e.g. 'ERA5', 'GDEX-GMF-icechunk').",
    )

    model_config = {"env_prefix": "BCSD_", "extra": "ignore"}

    # Which downscaling method this run uses. Selects the VariableConfig defaults
    # table in VariableConfig.for_variable, so it has to be a run-identity field rather
    # than a PipelineOptions one: changing it changes the numbers, not just the plumbing.
    downscaling_method: DownscalingMethod = Field(
        ...,
        description=(
            "Downscaling method: 'BCSD' (detrend + quantile mapping) or 'QDMSD' "
            "(quantile delta mapping). Selects the per-variable defaults table. "
            "Required: every config states its method rather than inheriting one."
        ),
    )

    # Variable-specific settings. Always populated: when not supplied explicitly it is
    # derived from ``variable`` in a ``mode="before"`` validator (see below), so it is
    # never ``None`` after construction. This is the single source of truth for
    # variable-specific parameters — there are deliberately no per-field accessors on
    # BCSDConfig, since those silently mapped variables to the wrong method (issue #423).
    variable_config: VariableConfig = Field(
        default=None,  # type: ignore[assignment]  # populated by _populate_variable_config
        description="Variable-specific BCSD parameters. Auto-populated from `variable`.",
    )

    # Keys that look like they belong here but do not: ones that moved off BCSDConfig,
    # and ones that are only meaningful during matrix expansion. extra="ignore" would
    # drop them silently, so each is rejected with a pointer to its replacement.
    _REJECTED_KEYS: ClassVar[dict[str, str]] = {
        "mapping_type": (
            "'mapping_type' was renamed to 'debias_approach' and then moved onto "
            "VariableConfig. Set 'variable_config.debias_approach' for a single-variable "
            "config, or 'variable_overrides' in a matrix config."
        ),
        "debias_approach": (
            "'debias_approach' moved from BCSDConfig onto VariableConfig. Set "
            "'variable_config.debias_approach' for a single-variable config; in a matrix "
            "config, use a top-level 'debias_approach' key to apply it to every variable, "
            "or 'variable_overrides' to set it per variable (or --debias-approach / "
            "--variable-override on the CLI, or BCSD_VARIABLE_CONFIG as JSON in the "
            "environment)."
        ),
        "variable_overrides": (
            "'variable_overrides' is resolved during matrix expansion and is not a "
            "BCSDConfig field. It only takes effect in a matrix config, i.e. one with "
            "at least one list-valued key such as 'variables: [\"tas\"]'. For a "
            "single-variable config set 'variable_config' directly instead."
        ),
    }

    @model_validator(mode="before")
    @classmethod
    def _reject_unsupported_keys(cls, data):
        """Fail loudly if a key that BCSDConfig does not honor is still used.

        Checks the supplied data and the ``BCSD_*`` environment. Both need covering:
        pydantic-settings filters env vars against the model's fields before any
        validator runs, so a rejected key set as ``BCSD_DEBIAS_APPROACH`` never reaches
        ``data`` and would otherwise vanish without a trace. The environment lookup
        is case-folded to match pydantic-settings' default ``case_sensitive=False``.
        """
        supplied = set(data) if isinstance(data, dict) else set()
        prefix = cls.model_config.get("env_prefix", "")
        env_keys = {name.casefold() for name in os.environ}
        for key, message in cls._REJECTED_KEYS.items():
            if key in supplied or f"{prefix}{key}".casefold() in env_keys:
                raise ValueError(message)
        return data

    @model_validator(mode="before")
    @classmethod
    def _populate_variable_config(cls, data):
        """Derive ``variable_config`` from ``variable`` when it is not supplied.

        Runs before field validation so ``variable_config`` can be declared as a
        required (non-optional) field and is never ``None`` after construction. An
        invalid ``variable`` raises here with a single clear "Unknown variable" error.
        """
        if isinstance(data, dict) and data.get("variable_config") is None:
            variable = data.get("variable")
            method = data.get("downscaling_method") or os.environ.get("BCSD_DOWNSCALING_METHOD")
            if variable is not None:
                if method is None:
                    raise ValueError(
                        "'downscaling_method' is required: set it to 'BCSD' or 'QDMSD' in "
                        "the config, or BCSD_DOWNSCALING_METHOD in the environment. There "
                        "is deliberately no default, so every run states which method "
                        "produced it."
                    )
                data["variable_config"] = VariableConfig.for_variable(variable, method)
        return data

    @model_validator(mode="after")
    def _check_method_matches_debias_approach(self):
        """Reject a debias_approach that contradicts the declared downscaling_method.

        The two are set independently: ``downscaling_method`` picks the defaults table,
        but ``debias_approach`` can still be overridden run-wide or per variable. A
        mismatch is silently wrong rather than loud, because the rest of the row still
        comes from the other table. ``BCSD`` with ``qdm`` would detrend and retrend
        around a method that carries the trend itself, and ``QDMSD`` without ``qdm``
        would not be quantile delta mapping at all.
        """
        is_qdm = self.variable_config.debias_approach == "qdm"
        if is_qdm != (self.downscaling_method == "QDMSD"):
            raise ValueError(
                f"downscaling_method={self.downscaling_method!r} is incompatible with "
                f"debias_approach={self.variable_config.debias_approach!r}. "
                "'qdm' requires downscaling_method='QDMSD', and 'QDMSD' requires "
                "debias_approach='qdm'. Set both consistently, or drop the "
                "debias_approach override and let the method's table supply it."
            )
        return self

    @field_validator("predict_period_start", "predict_period_end")
    @classmethod
    def validate_scenario_periods(cls, v, info):
        """Ensure predict periods are specified when scenario is set"""
        scenario = info.data.get("scenario")
        if scenario is not None and v is None:
            raise ValueError(
                "predict_period_start and predict_period_end must be specified when scenario is set"
            )
        return v

    @field_validator("train_period_end")
    @classmethod
    def validate_train_period(cls, v, info):
        """Ensure training period is valid"""
        start = info.data.get("train_period_start")
        if start and v < start:
            raise ValueError(f"train_period_end ({v}) must be >= train_period_start ({start})")
        return v

    @field_validator("predict_period_end")
    @classmethod
    def validate_predict_period(cls, v, info):
        """Ensure prediction period is valid"""
        start = info.data.get("predict_period_start")
        if start and v and v < start:
            raise ValueError(f"predict_period_end ({v}) must be >= predict_period_start ({start})")
        return v

    @field_validator("subset_bounds")
    @classmethod
    def validate_bounds(cls, v):
        """Validate spatial bounds"""
        if v is not None:
            lat_min, lat_max, lon_min, lon_max = v
            if lat_min >= lat_max:
                raise ValueError(f"lat_min ({lat_min}) must be < lat_max ({lat_max})")
            if lon_min >= lon_max:
                raise ValueError(f"lon_min ({lon_min}) must be < lon_max ({lon_max})")
            if not (-90 <= lat_min <= 90) or not (-90 <= lat_max <= 90):
                raise ValueError(f"Latitude must be in [-90, 90], got ({lat_min}, {lat_max})")
            if not (-180 <= lon_min <= 360) or not (-180 <= lon_max <= 360):
                raise ValueError(f"Longitude must be in [-180, 360], got ({lon_min}, {lon_max})")
        return v

    @computed_field
    def run_id(self) -> str:
        """
        Unique identifier for this run configuration.
        Used for logging and organizing outputs.
        """
        parts = [
            self.gcm,
            self.variable,
            self.ensemble_member,
        ]

        if self.scenario:
            parts.append(self.scenario)

        if self.subset_bounds:
            parts.append("subset")

        return "_".join(parts)

    @computed_field
    def config_hash(self) -> str:
        """
        Hash of configuration for cache invalidation.
        Only includes parameters that affect computation results.
        """
        # Create deterministic dict of parameters that affect results
        hash_params = {
            "gcm": self.gcm,
            "variable": self.variable,
            "ensemble_member": self.ensemble_member,
            "scenario": self.scenario,
            "train_period": (self.train_period_start, self.train_period_end),
            "predict_period": (self.predict_period_start, self.predict_period_end),
            "subset_bounds": self.subset_bounds,
            # downscaling_method is deliberately absent: it only selects which table
            # variable_config was read from, and variable_config is hashed below, so
            # BCSD and QDMSD already produce different hashes. Adding it would change
            # every existing config's hash and invalidate the S3 cache for no gain.
            "variable_config": self.variable_config.model_dump() if self.variable_config else None,
        }

        # Create stable string representation and hash
        hash_str = str(sorted(hash_params.items()))
        return hashlib.sha256(hash_str.encode()).hexdigest()[:12]

    @computed_field
    def is_sai_scenario(self) -> bool:
        """Check if this is an SAI intervention scenario"""
        return self.scenario and ("G6" in self.scenario.upper() or "SAI" in self.scenario.upper())


class VariableClipBounds(BaseModel):
    min: float | None = None
    max: float | None = None


class PipelineOptions(pydantic_settings.BaseSettings):
    """
    Operational settings for the BCSD pipeline.

    Covers infrastructure (storage paths, environment, branch) and runtime
    flags (verbosity, rechunking, post-processing). These do not affect
    computation results and are separate from BCSDConfig run identity.

    All fields can be overridden via BCSD_* environment variables.
    """

    scratch_dir: str = Field(
        "s3://carbonplan-srm/scratch/cache/",
        description="Base directory for cached intermediate artifacts",
    )
    output_dir: str = Field(
        "s3://carbonplan-srm/scratch/output/", description="Directory for final downscaled outputs"
    )
    environment: str = Field(
        default="qa",
        description="Environment name (qa, production). Separates cache/outputs by deployment stage.",
    )
    branch: str = Field(
        default=_cache_version,
        description="icechunk branch for output writes. Defaults to the installed package version (e.g. 'v1.2.0'). Each version starts a fresh branch; bump the package to get a clean slate. Override with BCSD_BRANCH env var.",
    )
    verbose: bool = Field(True, description="Enable verbose logging")
    rechunk_workflow: bool = Field(
        True, description="Enable strategic rechunking between pipeline stages"
    )
    apply_ocean_mask: bool = Field(
        False, description="Mask ocean pixels to NaN in the final scenario output"
    )
    save_intermediate: bool = Field(
        False,
        description="Save intermediate artifacts (e.g. detrended data, quantile mapping results) to cache for debugging and analysis",
    )
    clip_values: bool = Field(
        True,
        description="Apply post-bias-correction clipping",
    )
    clip_bounds: dict[str, VariableClipBounds] = Field(
        default={
            "pr": VariableClipBounds(min=0.0),
            "rsds": VariableClipBounds(min=0.0),
            "hurs": VariableClipBounds(min=0.0, max=105.0),
        },
        description="Per-variable clip bounds applied when clip_values=True.",
    )
    executor: Literal["coiled", "aws-batch", "local"] = Field(
        "coiled",
        description=(
            "Where stage tasks run. 'coiled' submits one Coiled Batch task per config; "
            "'aws-batch' submits one AWS Batch array job per wave; "
            "'local' runs every config sequentially in-process."
        ),
    )
    batch_job_queue: str = Field("srm-production", description="AWS Batch job queue name")
    batch_job_definition: str = Field(
        "srm-downscaling", description="AWS Batch job definition name"
    )
    batch_region: str = Field("us-west-2", description="Region for the AWS Batch control plane")
    container_image: str | None = Field(
        None,
        description=(
            "Container image remote tasks run in. Set it and the coiled executor runs this "
            "image instead of package-syncing the caller's environment, so both executors "
            "are byte-identical. Left unset, coiled falls back to package sync."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_use_coiled(cls, values):
        """Reject the removed ``use_coiled`` flag rather than silently ignoring it.

        ``extra = "ignore"`` would otherwise swallow the old key and quietly change which
        executor runs, which is exactly the failure mode a hard break is meant to prevent.
        """
        if isinstance(values, dict) and "use_coiled" in values:
            raise ValueError(
                "'use_coiled' was replaced by 'executor'. Set executor to 'coiled', "
                "'aws-batch', or 'local'."
            )
        return values

    model_config = {"env_prefix": "BCSD_", "extra": "ignore"}


class CacheConfig(BaseModel):
    """Configuration for artifact caching behavior"""

    base_dir: str = Field(
        "s3://carbonplan-srm/scratch/cache/", description="Base directory for cache storage"
    )
    force_recompute: bool = Field(
        False, description="Force recomputation even if cached artifacts exist"
    )
    environment: str = Field("qa", description="Environment for cache namespace (qa, production)")
    branch: str = Field(
        _cache_version,
        description="icechunk branch for output writes. Defaults to the installed package version.",
    )
    check_integrity: bool = Field(
        True, description="Verify cached artifacts are valid before using"
    )


class RuntimeConfig(BaseModel):
    """Runtime execution configuration"""

    use_coiled: bool = Field(True, description="Use Coiled for distributed execution")
    coiled_vm_type: str = Field(
        "r8g.2xlarge", description="VM type for Coiled workers (64GB RAM, 8 vCPUs)"
    )
    coiled_region: str = Field("us-west-2", description="AWS region for Coiled execution")
    coiled_keepalive: str = Field(
        "5 minutes", description="Keep Coiled VMs alive for this duration after task completion"
    )
    max_parallel_tasks: int | None = Field(
        None, description="Maximum number of parallel tasks. None for unlimited."
    )


'''
Usage Examples:

# 1. Minimal config (auto-populates variable config)
config = BCSDConfig(
    gcm="CESM2-WACCM",
    variable="tas",
    ensemble_member=0,
    scenario="ssp245",
    downscaling_method="BCSD",
    predict_period_start=2015,
    predict_period_end=2100
)

print(config.run_id)  # "CESM2-WACCM_tas_e00_ssp245"
print(config.variable_config.detrend_data)  # True (auto-loaded from variable config)
print(config.variable_config.disaggregation_method)  # "additive"

# 2. SAI scenario
sai_config = BCSDConfig(
    gcm="CESM2-WACCM",
    variable="pr",
    ensemble_member=1,
    scenario="G6-1.5K",
    downscaling_method="BCSD",
    predict_period_start=2015,
    predict_period_end=2100,
)

print(sai_config.is_sai_scenario)  # True
print(sai_config.variable_config.detrend_data)  # False (precipitation doesn't detrend)

# 3. Regional subset
subset_config = BCSDConfig(
    gcm="MIROC-ES2H",
    variable="tasmax",
    ensemble_member=0,
    scenario="ssp245",
    downscaling_method="BCSD",
    predict_period_start=2015,
    predict_period_end=2100,
    subset_bounds=(-35, -20, 15, 35)  # South Africa
)

# 4. Override variable config
custom_config = BCSDConfig(
    gcm="UKESM",
    variable="tas",
    ensemble_member=2,
    scenario="ssp245",
    downscaling_method="BCSD",
    predict_period_start=2015,
    predict_period_end=2100,
    variable_config=VariableConfig(
        detrend_data=False,  # Custom: don't detrend
        do_windowing=True,
        running_window_length=31,
        running_window_step_length=1,
        disaggregation_method="additive",
        disaggregation_clim_method="simple",
        disaggregation_tiny_threshold=0.0,
        detrend_method="additive",
        debias_approach="nonparametric_hybrid_2sided",
    )
)

# 5. Load from YAML
# Config file: configs/cesm_tas.yaml
"""
gcm: CESM2-WACCM
variable: tas
ensemble_member: 0
scenario: ssp245
downscaling_method: BCSD
predict_period_start: 2015
predict_period_end: 2100
environment: qa
# branch defaults to installed package version; override here if needed
# branch: v1.0.post3
"""

import yaml
with open("configs/cesm_tas.yaml") as f:
    config_dict = yaml.safe_load(f)
config = BCSDConfig(**config_dict)
'''
