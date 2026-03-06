from __future__ import annotations

import hashlib
from typing import Literal

import pydantic_settings
from pydantic import BaseModel, Field, computed_field, field_validator


class VariableConfig(BaseModel):
    """Variable-specific BCSD configuration parameters"""

    detrend_data: bool
    do_windowing: bool
    downscaling_method: Literal["additive", "multiplicative"]
    downscaling_clim_method: Literal["simple", "fft"]
    detrend_method: Literal["additive", "multiplicative"] = "additive"

    @classmethod
    def for_variable(cls, variable: str) -> VariableConfig:
        """Load variable-specific config from BCSD_CONFIG"""
        BCSD_CONFIG = {
            "pr": {
                "detrend_data": False,
                "detrend_method": "multiplicative",
                "do_windowing": True,
                "downscaling_method": "multiplicative",
                "downscaling_clim_method": "simple",
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
            "rsds": {
                "detrend_data": True,
                "detrend_method": "multiplicative",
                "do_windowing": True,
                "downscaling_method": "multiplicative",
                "downscaling_clim_method": "simple",
            },
        }

        if variable not in BCSD_CONFIG:
            raise ValueError(
                f"Unknown variable: {variable}. Must be one of {list(BCSD_CONFIG.keys())}"
            )

        return cls(**BCSD_CONFIG[variable])


class BCSDConfig(pydantic_settings.BaseSettings):
    """
    Main configuration for BCSD downscaling pipeline.

    This config captures all parameters needed to uniquely identify a BCSD run,
    including model, variable, ensemble member, scenario, time periods, and
    optional spatial subsetting.
    """

    # Model and data identifiers
    gcm: str = Field(..., description="GCM name (e.g., 'CESM2-WACCM', 'MIROC-ES2H', 'UKESM')")
    variable: Literal["tas", "tasmax", "pr", "rsds"] = Field(
        ..., description="Variable to downscale"
    )
    ensemble_member: str = Field(..., description="Ensemble member label (e.g. 'r1i1p1f1', '01')")
    scenario: str | None = Field(
        None,
        description="Scenario name (e.g., 'ssp245', 'G6-1.5K'). None for historical-only runs.",
    )

    # Time periods
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

    # Cache and output paths
    cache_dir: str = Field(
        "s3://carbonplan-scratch/srm/cache/",
        description="Base directory for cached intermediate artifacts",
    )
    output_dir: str = Field(
        "s3://carbonplan-scratch/srm/outputs/", description="Directory for final downscaled outputs"
    )
    environment: str = Field(
        default="qa",
        description="Environment name (qa, staging, production). Separates cache/outputs by deployment stage.",
    )
    version: str = Field(
        default="v1",
        description="Version identifier for cache/output path namespacing (e.g. 'v1', 'v2'). Override with BCSD_VERSION env var.",
    )

    model_config = {"env_prefix": "BCSD_"}

    # Variable-specific settings (auto-populated)
    variable_config: VariableConfig | None = Field(
        None, description="Variable-specific BCSD parameters. Auto-populated if None."
    )

    # Runtime options
    verbose: bool = Field(True, description="Enable verbose logging")
    rechunk_workflow: bool = Field(
        True, description="Enable strategic rechunking between pipeline stages"
    )
    mapping_type: Literal["parametric", "nonparametric", "nonparametric_hybrid"] = Field(
        "parametric", description="Quantile mapping method for bias correction"
    )

    def model_post_init(self, __context) -> None:
        """Post-initialization validation and auto-population"""
        # Auto-populate variable_config if not provided
        if self.variable_config is None:
            self.variable_config = VariableConfig.for_variable(self.variable)

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
            "mapping_type": self.mapping_type,
        }

        # Create stable string representation and hash
        hash_str = str(sorted(hash_params.items()))
        return hashlib.sha256(hash_str.encode()).hexdigest()[:12]

    @computed_field
    def detrend_data(self) -> bool:
        """Convenience accessor for variable config"""
        return self.variable_config.detrend_data if self.variable_config else False

    @computed_field
    def detrend_method(self) -> str:
        """Convenience accessor for variable config"""
        return self.variable_config.detrend_method if self.variable_config else "additive"

    @computed_field
    def do_windowing(self) -> bool:
        """Convenience accessor for variable config"""
        return self.variable_config.do_windowing if self.variable_config else False

    @computed_field
    def downscaling_method(self) -> str:
        """Convenience accessor for variable config"""
        return self.variable_config.downscaling_method if self.variable_config else "additive"

    @computed_field
    def downscaling_clim_method(self) -> str:
        """Convenience accessor for variable config"""
        return self.variable_config.downscaling_clim_method if self.variable_config else "fft"

    @computed_field
    def is_sai_scenario(self) -> bool:
        """Check if this is an SAI intervention scenario"""
        return self.scenario and ("G6" in self.scenario.upper() or "SAI" in self.scenario.upper())

    def to_legacy_kwargs(self) -> dict:
        """
        Convert to kwargs dict for legacy run_bcsd function.
        Useful for backward compatibility during transition.
        """
        kwargs = {
            "gcm": self.gcm,
            "var_name": self.variable,
            "train_period_start": self.train_period_start,
            "train_period_end": self.train_period_end,
            "verbose": self.verbose,
            "rechunk_workflow": self.rechunk_workflow,
            "subset_bounds": list(self.subset_bounds) if self.subset_bounds else None,
        }

        if self.scenario:
            kwargs.update(
                {
                    "predict_period_start": self.predict_period_start,
                    "predict_period_end": self.predict_period_end,
                }
            )

        return kwargs


class CacheConfig(BaseModel):
    """Configuration for artifact caching behavior"""

    base_dir: str = Field(
        "s3://carbonplan-scratch/srm/cache/", description="Base directory for cache storage"
    )
    force_recompute: bool = Field(
        False, description="Force recomputation even if cached artifacts exist"
    )
    environment: str = Field(
        "qa", description="Environment for cache namespace (qa, staging, production)"
    )
    version: str = Field(
        "v1", description="Version identifier for cache path namespacing (e.g. 'v1', 'v2')"
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
print(config.detrend_data)  # True (auto-loaded from variable config)
print(config.downscaling_method)  # "additive"

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
print(sai_config.detrend_data)  # False (precipitation doesn't detrend)

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
version: v1
"""

import yaml
with open("configs/cesm_tas.yaml") as f:
    config_dict = yaml.safe_load(f)
config = BCSDConfig(**config_dict)

# 6. Convert to legacy format (backward compatibility)
legacy_kwargs = config.to_legacy_kwargs()
from srm.run_bcsd import run_bcsd
result = run_bcsd(**legacy_kwargs)
'''
