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


def init_repo(bucket, prefix, region="us-west-2", readonly: bool = True):
    storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, region=region)
    repo = icechunk.Repository.open_or_create(storage)
    if readonly:
        session = repo.readonly_session("main")
    else:
        session = repo.writable_session("main")
    return repo, session
