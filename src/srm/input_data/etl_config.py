from dataclasses import dataclass, field

from srm import catalog
from srm.config import ClusterConfig


@dataclass
class BaseETLConfig:
    scenario: str = ""
    catalog_key: str = ""
    s3_bucket: str = "carbonplan-srm"
    cluster: ClusterConfig = field(default_factory=ClusterConfig)

    @property
    def dataset_entry(self):
        return catalog.get(self.catalog_key)

    @property
    def bucket(self):
        return self.dataset_entry.bucket

    @property
    def prefix(self):
        return self.dataset_entry.prefix

    @property
    def output_path(self):
        return getattr(self.dataset_entry, "path", None)

    @property
    def all_variables(self):
        return [var.name for var in self.dataset_entry.expected_vars]

    @property
    def encoding(self):
        encoding_entry = (
            catalog.get(self.materialized_key)
            if hasattr(self, "materialized_key")
            else self.dataset_entry
        )
        return {
            "chunks": getattr(encoding_entry, "expected_chunks", None),
            "shards": getattr(encoding_entry, "expected_shards", None),
        }
