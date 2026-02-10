from dataclasses import dataclass, field

from srm import catalog
from srm.config import ClusterConfig


@dataclass
class BaseETLConfig:
    scenario: str
    catalog_key: str
    s3_bucket: str = "carbonplan-srm"
    cluster: ClusterConfig = field(default_factory=ClusterConfig)

    def __post_init__(self):
        self.dataset_entry = catalog.get(self.catalog_key)

        self.bucket = self.dataset_entry.bucket
        self.prefix = self.dataset_entry.prefix

        self.output_path = getattr(self.dataset_entry, "path", None)

        self.all_variables = [var.name for var in self.dataset_entry.expected_vars]

        self.encoding = {
            "chunks": self.dataset_entry.expected_chunks,
            "shards": self.dataset_entry.expected_shards,
        }
