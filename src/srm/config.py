from dataclasses import dataclass, field

import coiled
import icechunk
from distributed import Client


@dataclass
class ClusterConfig:
    n_workers: list = field(default_factory=lambda: [10, 100])
    region: str = "us-west-2"
    worker_vm_types: list = field(default_factory=lambda: ["r8g.medium", "r8g.large"])
    scheduler_vm_types: str = "c8g.8xlarge"
    spot_policy: str = "spot_with_fallback"
    tags: dict = field(default_factory=lambda: {"Project": "SRM"})


def setup_cluster(config: ClusterConfig = None):  # TODO: add type
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


def setup_local_client(n_workers=64):
    return Client(n_workers=n_workers)


def init_repo(bucket, prefix, region="us-west-2", readonly: bool = True):
    storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, region=region)
    repo = icechunk.Repository.open_or_create(storage)
    if readonly:
        session = repo.readonly_session("main")
    else:
        session = repo.writable_session("main")
    return repo, session
