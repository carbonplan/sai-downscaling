from dataclasses import dataclass, field

from saidownscale.config import ClusterConfig


@dataclass
class BaseETLConfig:
    scenario: str = ""
    s3_bucket: str = "carbonplan-srm"
    cluster: ClusterConfig = field(default_factory=ClusterConfig)

    @property
    def encoding(self):
        from saidownscale import catalog

        entry = catalog.get(self.materialized_key)  # type: ignore[attr-defined]
        return {
            "chunks": getattr(entry, "expected_chunks", None),
            "shards": getattr(entry, "expected_shards", None),
        }
