"""
Command-line interface for BCSD downscaling pipeline.

Provides typer-based CLI for running BCSD downscaling with automatic caching,
resumability, and Coiled integration for distributed execution.
"""

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


@app.command()
def run(
    config_path: str = typer.Option(..., help="Path to YAML config or directory of configs"),
    stage: str = typer.Option(None, help="Run specific stage: obs, historical, scenario, or all"),
    force: bool = typer.Option(False, help="Force recompute even if cached"),
    coiled: bool = typer.Option(True, help="Use Coiled for execution"),
):
    """Run BCSD pipeline with automatic caching and resumability"""

    # Load configs
    configs = load_configs(config_path)
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
def status(
    config_path: str = typer.Option(..., help="Path to config(s)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed path information"),
):
    """Check status of cached artifacts for given configs"""
    configs = load_configs(config_path)
    orchestrator = BCSDOrchestrator()

    # Show cache configuration if verbose
    if verbose and configs:
        cache = orchestrator._get_cache(configs[0])
        console.print("\n[cyan]Cache Configuration:[/cyan]")
        console.print(f"  Cache Path: {cache.base_path}")
        console.print(f"  Output Path: {cache.output_dir or '(same as cache)'}")
        console.print(f"  Environment: {cache.environment}")
        console.print("\n[cyan]Example Paths:[/cyan]")
        config = configs[0]
        console.print(
            f"  Obs: {cache.get_obs_path(config.gcm, config.variable, config.subset_bounds)}"
        )
        console.print(
            f"  Historical: {cache.get_historical_path(config.gcm, config.variable, config.ensemble_member, config.subset_bounds)}"
        )
        if config.scenario:
            console.print(
                f"  Scenario: {cache.get_scenario_path(config.gcm, config.variable, config.ensemble_member, config.scenario, config.subset_bounds)}\n"
            )

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
    cache = ArtifactCache(base_path=configs[0].cache_dir)

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
    cache = ArtifactCache(base_path=configs[0].cache_dir)
    artifacts = cache.list_artifacts(stage=stage, gcm=gcm, variable=variable)

    if not artifacts:
        console.print("[yellow]No cached artifacts found[/yellow]")
        return

    table = Table(title=f"Cached Artifacts ({len(artifacts)})", show_header=True)
    table.add_column("Path", style="cyan")

    for artifact in artifacts:
        table.add_row(artifact)

    console.print(table)
