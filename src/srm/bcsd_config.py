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
from typing import ClassVar, Literal

import pydantic_settings
from packaging.version import Version as _Version
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

_cache_version = f"v{_Version(_pkg_version('srm')).public}"

DebiasApproach = Literal[
    "parametric", "nonparametric", "nonparametric_hybrid", "nonparametric_hybrid_2sided"
]
DownscalingMethod = Literal["additive", "multiplicative"]
DownscalingClimMethod = Literal["simple", "fft"]
DetrendMethod = Literal["additive", "multiplicative"]
VariableName = Literal["tas", "tasmax", "tasmin", "pr", "rsds", "dtr", "hurs"]


class VariableConfig(BaseModel):
    """Variable-specific BCSD configuration parameters"""

    detrend_data: bool
    do_windowing: bool
    running_window_length: int = 31
    downscaling_method: DownscalingMethod
    downscaling_clim_method: DownscalingClimMethod
    detrend_method: DetrendMethod = "additive"
    debias_approach: DebiasApproach = "nonparametric_hybrid_2sided"

    @classmethod
    def for_variable(cls, variable: str) -> VariableConfig:
        """Load variable-specific config from BCSD_CONFIG.

        Rows list only the fields that vary by variable. Anything uniform across all
        variables, such as ``debias_approach`` and ``running_window_length``, is left
        to the field default above so there is one place to change it.
        """
        BCSD_CONFIG = {
            "pr": {
                "detrend_data": False,
                "detrend_method": "multiplicative",
                "do_windowing": True,
                "downscaling_method": "multiplicative",
                "downscaling_clim_method": "fft",
            },
            "tas": {
                "detrend_data": True,
                "detrend_method": "additive",
                "do_windowing": True,
                "downscaling_method": "additive",
                "downscaling_clim_method": "fft",
            },
            "tasmax": {
                "detrend_data": True,
                "detrend_method": "additive",
                "do_windowing": True,
                "downscaling_method": "additive",
                "downscaling_clim_method": "fft",
            },
            "tasmin": {
                "detrend_data": True,
                "detrend_method": "additive",
                "do_windowing": True,
                "downscaling_method": "additive",
                "downscaling_clim_method": "fft",
            },
            "rsds": {
                "detrend_data": False,
                "detrend_method": "multiplicative",
                "do_windowing": True,
                "downscaling_method": "multiplicative",
                "downscaling_clim_method": "fft",
                "debias_approach": "nonparametric"
            },
            "dtr": {
                "detrend_data": False,
                "detrend_method": "multiplicative",
                "do_windowing": True,
                "downscaling_method": "multiplicative",
                "downscaling_clim_method": "fft",
            },
            "hurs": {
                "detrend_data": False,
                "detrend_method": "additive",
                "do_windowing": True,
                "downscaling_method": "multiplicative",
                "downscaling_clim_method": "fft",
            },
        }

        if variable not in BCSD_CONFIG:
            raise ValueError(
                f"Unknown variable: {variable}. Must be one of {list(BCSD_CONFIG.keys())}"
            )

        return cls(**BCSD_CONFIG[variable])


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
            "'variable_config.debias_approach' for a single-variable config, or "
            "'variable_overrides' in a matrix config (or --debias-approach / "
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
            if variable is not None:
                data["variable_config"] = VariableConfig.for_variable(variable)
        return data

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
            "variable_config": self.variable_config.model_dump() if self.variable_config else None,
        }

        # Create stable string representation and hash
        hash_str = str(sorted(hash_params.items()))
        return hashlib.sha256(hash_str.encode()).hexdigest()[:12]

    @computed_field
    def is_sai_scenario(self) -> bool:
        """Check if this is an SAI intervention scenario"""
        return self.scenario and ("G6" in self.scenario.upper() or "SAI" in self.scenario.upper())

    def make_config_for_variable(self, variable: str) -> BCSDConfig:
        """
        Return a new BCSDConfig for a different variable, keeping all other parameters the same.

        Useful for grabbing paths to intermediate artifacts for a sibling variable (e.g. dtr or
        tasmax when processing tasmin) without redefining the entire config. Variable-specific
        parameters are auto-populated based on the new variable.

        The sibling inherits this config's ``debias_approach`` while taking its own
        per-variable defaults for every other field.
        """
        return BCSDConfig(
            gcm=self.gcm,
            variable=variable,
            ensemble_member=self.ensemble_member,
            scenario=self.scenario,
            train_period_start=self.train_period_start,
            train_period_end=self.train_period_end,
            predict_period_start=self.predict_period_start,
            predict_period_end=self.predict_period_end,
            subset_bounds=self.subset_bounds,
            # The sibling gets its own per-variable defaults but inherits this run's
            # debias_approach, matching the pre-move behavior. model_copy is safe here
            # because the source value is an already-validated Literal.
            variable_config=VariableConfig.for_variable(variable).model_copy(
                update={"debias_approach": self.variable_config.debias_approach}
            ),
        )


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
        "s3://carbonplan-scratch/srm/cache/",
        description="Base directory for cached intermediate artifacts",
    )
    output_dir: str = Field(
        "s3://carbonplan-scratch/srm/outputs/", description="Directory for final downscaled outputs"
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

    model_config = {"env_prefix": "BCSD_", "extra": "ignore"}


class CacheConfig(BaseModel):
    """Configuration for artifact caching behavior"""

    base_dir: str = Field(
        "s3://carbonplan-scratch/srm/cache/", description="Base directory for cache storage"
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
    predict_period_start=2015,
    predict_period_end=2100
)

print(config.run_id)  # "CESM2-WACCM_tas_e00_ssp245"
print(config.variable_config.detrend_data)  # True (auto-loaded from variable config)
print(config.variable_config.downscaling_method)  # "additive"

# 2. SAI scenario
sai_config = BCSDConfig(
    gcm="CESM2-WACCM",
    variable="pr",
    ensemble_member=1,
    scenario="G6-1.5K",
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
    predict_period_start=2015,
    predict_period_end=2100,
    variable_config=VariableConfig(
        detrend_data=False,  # Custom: don't detrend
        do_windowing=True,
        downscaling_method="additive",
        downscaling_clim_method="simple"
    )
)

# 5. Load from YAML
# Config file: configs/cesm_tas.yaml
"""
gcm: CESM2-WACCM
variable: tas
ensemble_member: 0
scenario: ssp245
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
