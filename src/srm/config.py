from dataclasses import dataclass, field

import icechunk
from distributed import Client


@dataclass(frozen=True)
class VarSpec:
    name: str
    units: str
    long_name: str = ""
    cell_methods: str | None = None


@dataclass(frozen=True)
class VarStandards:
    PR: VarSpec = VarSpec(name="pr", units="kg m-2 s-1")
    TAS: VarSpec = VarSpec(name="tas", units="K")
    TASMIN: VarSpec = VarSpec(name="tasmin", units="K")
    TASMAX: VarSpec = VarSpec(name="tasmax", units="K")
    HURS: VarSpec = VarSpec(name="hurs", units="%")
    RSDS: VarSpec = VarSpec(name="rsds", units="W m-2")
    HUSS: VarSpec = VarSpec(name="huss", units="1")
    RLDS: VarSpec = VarSpec(name="rlds", units="W m-2")
    PS: VarSpec = VarSpec(name="ps", units="Pa")
    PSL: VarSpec = VarSpec(name="psl", units="Pa")
    DTR: VarSpec = VarSpec(name="dtr", units="K")


SCENARIO_TO_GROUP: dict[str, str] = {
    "historical": "historical",
    "pangeo-historical": "historical",  # ETL alias: pangeo CMIP6 members merged into historical group
    "SSP245": "ssp245",
    "G6-1.5K": "g6_1p5k",
    "esgf-SSP245": "esgf_ssp245",
}


@dataclass
class ClusterConfig:
    n_workers: list = field(default_factory=lambda: [1, 50])
    region: str = "us-west-2"
    worker_vm_types: list = field(default_factory=lambda: ["r8g.2xlarge"])
    scheduler_vm_types: str = "c8g.2xlarge"
    spot_policy: str = "spot_with_fallback"
    tags: dict = field(default_factory=lambda: {"Project": "SRM"})


def setup_cluster(config: ClusterConfig = None):
    import coiled

    if config is None:
        config = ClusterConfig()
    cluster = coiled.Cluster(
        region=config.region,
        n_workers=config.n_workers,
        worker_vm_types=config.worker_vm_types,
        scheduler_vm_types=config.scheduler_vm_types,
        spot_policy=config.spot_policy,
        use_best_zone=True,
        tags=config.tags,
    )
    return cluster.get_client()


def setup_local_client(n_workers=4):
    return Client(n_workers=n_workers)


_ROOT_MESSAGES = frozenset(("_root", "initialize root group"))


def _icechunk_storage_for_path(path: str):
    """Return an icechunk Storage object for an S3 or local filesystem path."""
    if path.startswith("s3://"):
        path_no_scheme = path[len("s3://") :]
        bucket, _, prefix = path_no_scheme.partition("/")
        return icechunk.s3_storage(bucket=bucket, prefix=prefix)
    return icechunk.local_filesystem_storage(path=path)


def _ensure_root_group(repo: icechunk.Repository) -> str:
    """Commit an empty root group on a brand-new repo. Returns the _root snapshot ID.

    Without this, concurrent first writers to different zarr groups each create
    the root node and their commits cannot rebase
    (NewNodeConflictsWithExistingNode at "/").
    """
    import zarr

    for snapshot in repo.ancestry(branch="main"):
        if snapshot.message in _ROOT_MESSAGES:
            return snapshot.id
    session = repo.writable_session("main")
    zarr.open_group(session.store, mode="a")
    try:
        session.commit("_root")
        return repo.lookup_branch("main")
    except (icechunk.ConflictError, icechunk.RebaseFailedError, icechunk.NoChangesToCommitError):
        # Another VM committed _root concurrently, or root group already exists
        # (old stores used "initialize root group" as commit message).
        for snapshot in repo.ancestry(branch="main"):
            if snapshot.message in _ROOT_MESSAGES:
                return snapshot.id
        return repo.lookup_branch("main")


def init_repo(bucket, prefix, region="us-west-2", readonly: bool = True):
    storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, region=region)
    repo = icechunk.Repository.open_or_create(storage)
    if readonly:
        session = repo.readonly_session("main")
    else:
        _ensure_root_group(repo)
        session = repo.writable_session("main")
    return repo, session
