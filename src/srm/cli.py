"""
Command-line interface for BCSD downscaling pipeline.

Provides typer-based CLI for running BCSD downscaling with automatic caching,
resumability, and Coiled integration for distributed execution.
"""

import itertools
import logging
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from srm.bcsd_config import BCSDConfig
from srm.cache import ArtifactCache
from srm.orchestration import BCSDOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = typer.Typer(help="BCSD downscaling pipeline with automatic caching")
console = Console()


def load_configs(config_path: str) -> list[BCSDConfig]:
    """
    Load configuration(s) from YAML file or directory.

    Parameters
    ----------
    config_path : str
        Path to YAML config file or directory containing YAML files

    Returns
    -------
    list[BCSDConfig]
        List of loaded configurations
    """
    path = Path(config_path)
    configs = []

    if path.is_file():
        # Single config file
        with open(path) as f:
            config_dict = yaml.safe_load(f)
            configs.append(BCSDConfig(**config_dict))

    elif path.is_dir():
        # Directory of config files
        for yaml_file in sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml")):
            with open(yaml_file) as f:
                config_dict = yaml.safe_load(f)
                configs.append(BCSDConfig(**config_dict))

    else:
        raise ValueError(f"Config path does not exist: {config_path}")

    if not configs:
        raise ValueError(f"No valid configs found in: {config_path}")

    return configs


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
    cache_dir: str = "s3://carbonplan-scratch/srm/cache/",
    output_dir: str = "s3://carbonplan-scratch/srm/outputs/",
    environment: str = "qa",
    version: str = "v1",
    subset_bounds: tuple[float, float, float, float] | None = None,
) -> list[BCSDConfig]:
    """
    Generate BCSDConfig objects for every cartesian-product combination of GCMs,
    variables, ensemble members, and scenarios.

    Parameters
    ----------
    gcms : list[str]
        GCM names (e.g., ["CESM2-WACCM", "MIROC"])
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
    cache_dir : str
        Base directory for cached intermediate artifacts
    output_dir : str
        Directory for final downscaled outputs
    environment : str
        Environment name (qa, staging, production)
    version : str
        Version identifier
    subset_bounds : tuple[float, float, float, float] | None
        Spatial bounds as (lat_min, lat_max, lon_min, lon_max)

    Returns
    -------
    list[BCSDConfig]
        One config per cartesian-product combination.
    """
    configs = []
    for gcm, variable, member, scenario in itertools.product(gcms, variables, members, scenarios):
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
                cache_dir=cache_dir,
                output_dir=output_dir,
                environment=environment,
                version=version,
                subset_bounds=subset_bounds,
            )
        )
    return configs


@app.command()
def run(
    config_path: list[str] = typer.Option(
        ..., help="Path to YAML config or directory of configs (can be specified multiple times)"
    ),
    stage: str = typer.Option(None, help="Run specific stage: obs, historical, scenario, or all"),
    force: bool = typer.Option(False, help="Force recompute even if cached"),
    coiled: bool = typer.Option(True, help="Use Coiled for execution"),
    version: str | None = typer.Option(
        None, "--version", help="Override the version from config (e.g. 'v2')"
    ),
):
    """Run BCSD pipeline with automatic caching and resumability"""

    # Load configs
    configs = [cfg for path in config_path for cfg in load_configs(path)]
    if version is not None:
        configs = [config.model_copy(update={"version": version}) for config in configs]
    console.print(f"[bold green]Loaded {len(configs)} configuration(s)[/bold green]")

    orchestrator = BCSDOrchestrator()

    if stage == "obs" or stage == "prepare_observations":
        orchestrator.submit_stage("prepare_observations", configs, force=force, use_coiled=coiled)

    elif stage == "historical" or stage == "fit_historical":
        orchestrator.submit_stage("fit_historical", configs, force=force, use_coiled=coiled)

    elif stage == "scenario" or stage == "transform_scenario":
        orchestrator.submit_stage("transform_scenario", configs, force=force, use_coiled=coiled)

    elif stage == "all" or stage is None:
        orchestrator.run_full_workflow(configs, force=force, use_coiled=coiled)

    else:
        raise ValueError(f"Unknown stage: {stage}")

    console.print("[bold green]✓ Complete![/bold green]")


@app.command()
def run_matrix(
    gcm: list[str] = typer.Option(..., help="GCM name (repeatable: --gcm CESM2-WACCM --gcm MIROC)"),
    variable: list[str] = typer.Option(
        ..., help="Variable to downscale (repeatable: --variable tas --variable pr)"
    ),
    member: list[str] = typer.Option(
        ..., help="Ensemble member label (repeatable: --member r1i1p1f1 --member r2i1p1f1)"
    ),
    scenario: list[str] | None = typer.Option(
        None,
        help=(
            "Scenario (repeatable: --scenario ssp245 --scenario G6-1pt5k). "
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
    cache_dir: str = typer.Option(
        "s3://carbonplan-scratch/srm/cache/", help="Base directory for cached artifacts"
    ),
    output_dir: str = typer.Option(
        "s3://carbonplan-scratch/srm/outputs/", help="Directory for final outputs"
    ),
    environment: str = typer.Option("qa", help="Environment (qa, staging, production)"),
    version: str = typer.Option("v1", help="Version identifier (e.g. 'v1', 'v2')"),
    subset_bounds: str | None = typer.Option(
        None,
        help="Spatial bounds as 'lat_min,lat_max,lon_min,lon_max' (e.g. '-35,-22,16,33')",
    ),
    stage: str | None = typer.Option(
        None, help="Run specific stage: obs, historical, scenario, or all"
    ),
    force: bool = typer.Option(False, help="Force recompute even if cached"),
    coiled: bool = typer.Option(True, help="Use Coiled for distributed execution"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show configs without executing"),
):
    """Run BCSD pipeline over cartesian product of GCMs x variables x members x scenarios.

    Rather than pre-generating config files, specify each dimension as a repeatable
    option and the CLI will run every combination.

    Example (2 GCMs x 2 variables x 3 members x 2 scenarios = 24 runs):

        bcsd run-matrix \\
          --gcm CESM2-WACCM --gcm MIROC \\
          --variable tas --variable pr \\
          --member r1i1p1f1 --member r2i1p1f1 --member r3i1p1f1 \\
          --scenario ssp245 --scenario G6-1pt5k \\
          --predict-period-start 2015 --predict-period-end 2100

    Omit --scenario for historical-only runs.
    """
    # Normalize: no --scenario given -> historical-only (scenario=None)
    scenario_values: list[str | None] = scenario if scenario else [None]

    # Validate predict periods are provided when scenarios are given
    has_scenarios = any(s is not None for s in scenario_values)
    if has_scenarios and (predict_period_start is None or predict_period_end is None):
        console.print(
            "[red]Error: --predict-period-start and --predict-period-end are required "
            "when --scenario is specified.[/red]"
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
            console.print(
                "[red]Error: --subset-bounds must be 'lat_min,lat_max,lon_min,lon_max' "
                "(e.g. '-35,-22,16,33')[/red]"
            )
            raise typer.Exit(1)

    configs = configs_from_matrix(
        gcms=gcm,
        variables=variable,
        members=member,
        scenarios=scenario_values,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
        cache_dir=cache_dir,
        output_dir=output_dir,
        environment=environment,
        version=version,
        subset_bounds=parsed_bounds,
    )

    n = len(configs)
    console.print(
        f"[bold green]Generated {n} configuration(s): "
        f"{len(gcm)} GCM(s) x {len(variable)} variable(s) x "
        f"{len(member)} member(s) x {len(scenario_values)} scenario(s)[/bold green]"
    )

    if dry_run:
        table = Table(
            title=f"Matrix Configurations ({n})", show_header=True, header_style="bold magenta"
        )
        table.add_column("GCM", style="cyan")
        table.add_column("Variable", style="magenta")
        table.add_column("Member", justify="right")
        table.add_column("Scenario", style="yellow")
        for cfg in configs:
            table.add_row(
                cfg.gcm, cfg.variable, str(cfg.ensemble_member), cfg.scenario or "(historical)"
            )
        console.print(table)
        return

    orchestrator = BCSDOrchestrator()

    if stage == "obs" or stage == "prepare_observations":
        orchestrator.submit_stage("prepare_observations", configs, force=force, use_coiled=coiled)
    elif stage == "historical" or stage == "fit_historical":
        orchestrator.submit_stage("fit_historical", configs, force=force, use_coiled=coiled)
    elif stage == "scenario" or stage == "transform_scenario":
        orchestrator.submit_stage("transform_scenario", configs, force=force, use_coiled=coiled)
    elif stage == "all" or stage is None:
        orchestrator.run_full_workflow(configs, force=force, use_coiled=coiled)
    else:
        console.print(f"[red]Error: Unknown stage: {stage}[/red]")
        raise typer.Exit(1)

    console.print("[bold green]✓ Complete![/bold green]")


@app.command()
def status(
    config_path: list[str] = typer.Option(
        ..., help="Path to config(s) (can be specified multiple times)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed path information"),
    version: str | None = typer.Option(
        None, "--version", help="Override the version from config (e.g. 'v2')"
    ),
):
    """Check status of cached artifacts for given configs"""
    configs = [cfg for path in config_path for cfg in load_configs(path)]
    if version is not None:
        configs = [config.model_copy(update={"version": version}) for config in configs]
    orchestrator = BCSDOrchestrator()

    # Show cache configuration if verbose
    if verbose and configs:
        cache = orchestrator._get_cache(configs[0])
        console.print("\n[cyan]Cache Configuration:[/cyan]")
        console.print(f"  Cache Path: {cache.base_path}")
        console.print(f"  Output Path: {cache.output_dir or '(same as cache)'}")
        console.print(f"  Environment: {cache.environment}")
        console.print(f"  Version: {cache.version}")
        console.print("\n[cyan]Example Paths:[/cyan]")
        config = configs[0]
        console.print(f"  Obs: {cache.get_obs_path(config)}")
        console.print(f"  Historical: {cache.get_historical_path(config)}")
        if config.scenario:
            console.print(f"  Scenario: {cache.get_scenario_path(config)}\n")

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
            console.print(f"\n[yellow]Missing {stage_name}:[/yellow]")
            for run_id in stage_info["missing"][:10]:  # Show first 10
                console.print(f"  • {run_id}")
            if len(stage_info["missing"]) > 10:
                console.print(f"  ... and {len(stage_info['missing']) - 10} more")


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

    # Load config to get cache_dir
    configs = load_configs(config_path)
    if not configs:
        console.print("[red]Error: No valid configurations found[/red]")
        raise typer.Exit(1)

    # Use cache_dir from first config (all should have same cache_dir)
    cache = ArtifactCache(
        base_path=configs[0].cache_dir,
        environment=configs[0].environment,
        version=configs[0].version,
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
            console.print("[yellow]Cancelled[/yellow]")
            return

    deleted = cache.clear_cache(stage=stage, gcm=gcm, variable=variable)
    console.print(f"[green]✓ Deleted {deleted} artifact(s)[/green]")


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

    # Load config to get cache_dir
    configs = load_configs(config_path)
    if not configs:
        console.print("[red]Error: No valid configurations found[/red]")
        raise typer.Exit(1)

    # Use cache_dir from first config (all should have same cache_dir)
    cache = ArtifactCache(
        base_path=configs[0].cache_dir,
        environment=configs[0].environment,
        version=configs[0].version,
    )
    artifacts = cache.list_artifacts(stage=stage, gcm=gcm, variable=variable)

    if not artifacts:
        console.print("[yellow]No cached artifacts found[/yellow]")
        return

    table = Table(title=f"Cached Artifacts ({len(artifacts)})", show_header=True)
    table.add_column("Path", style="cyan")

    for artifact in artifacts:
        table.add_row(artifact)

    console.print(table)
