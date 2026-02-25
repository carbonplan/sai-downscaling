# How to Compare Outputs Across Code Versions

A common development workflow is to run the pipeline on a small test region against the current code, then run it again after merging new changes, and compare the two sets of outputs to validate that the changes behave as expected. The `version` field is the key mechanism for this: each run writes to a completely isolated path, so both versions of the data coexist in S3 and can be compared at any time.

## Step 1 — Run with the current code on a test region

Pick a small region (`subset_bounds`) and use `environment: qa` so outputs stay isolated from production data. Use a version string that describes the code state, e.g. the branch name or a short commit hash.

```bash
# Run the pipeline with the current code (e.g. pre-merge main)
uv run bcsd run-matrix \
  --gcm CESM2-WACCM \
  --variable tas \
  --member 0 \
  --scenario ssp245 \
  --predict-period-start 2015 --predict-period-end 2100 \
  --subset-bounds '-35,-22,16,33' \
  --cache-dir "s3://carbonplan-scratch/srm/bcsd-cache" \
  --output-dir "s3://carbonplan-scratch/srm/outputs/" \
  --environment qa --version main-baseline 
```

Outputs will be written under `.../outputs/qa/main-baseline/...`.

## Step 2 — Merge the new code and run again with a new version

After merging (or checking out) the new code, re-run with a different `--version`. Because the version is different, all three stages run from scratch on the same test region, producing a fully independent dataset.

```bash
# Run the pipeline with the new code (e.g. after merging a refactor branch)
uv run bcsd run-matrix \
  --gcm CESM2-WACCM \
  --variable tas \
  --member 0 \
  --scenario ssp245 \
  --predict-period-start 2015 --predict-period-end 2100 \
  --subset-bounds '-35,-22,16,33' \
  --cache-dir "s3://carbonplan-scratch/srm/bcsd-cache" \
  --output-dir "s3://carbonplan-scratch/srm/outputs/" \
  --environment qa --version refactor-icechunk 
```

Outputs will be written under `.../outputs/qa/refactor-icechunk/...`.

## Step 3 — Compare the two outputs

Both datasets are now available at their respective version paths and can be loaded and compared side by side:

```python
import xarray as xr
import icechunk

def open_version(output_dir, environment, version, gcm, variable, member, scenario, subset_id):
    path = (
        f"{output_dir}/{environment}/{version}/{scenario.lower()}/"
        f"{gcm}_{variable}_{member:03d}_{subset_id}_{scenario.lower()}.icechunk"
    )
    bucket, _, prefix = path.removeprefix("s3://").partition("/")
    storage = icechunk.s3_storage(bucket=bucket, prefix=prefix)
    repo = icechunk.Repository.open(storage)
    ds = xr.open_dataset(repo.session, engine='zarr', chunks={})
    return ds


kwargs = dict(
    output_dir="s3://carbonplan-scratch/srm/outputs",
    environment="qa",
    gcm="CESM2-WACCM",
    variable="tas",
    member=0,
    scenario="ssp245",
    subset_id="lat-35.0to-22.0_lon16.0to33.0",
)

ds_baseline = open_version(**kwargs, version="main-baseline")
ds_new      = open_version(**kwargs, version="refactor-icechunk")

diff = ds_new["tas"] - ds_baseline["tas"]
print(diff.max().values, diff.min().values)  # should be ~0 for a pure refactor
```

## Tips

- Keep `environment: qa` for all test runs so they never touch staging or production paths.
- Use descriptive `--version` strings (branch names, commit hashes, date stamps) rather than `v1`/`v2` so it is always clear which code produced which data.
- Use `bcsd status` to confirm both versions completed before comparing:

  ```bash
  uv run bcsd status --config-path configs/example.yaml --version main-baseline
  uv run bcsd status --config-path configs/example.yaml --version refactor-icechunk
  ```

- Once you are done comparing, clean up test artifacts with `bcsd cache-clear`:

  ```bash
  # The cache-clear command uses the version from your config;
  # point it at a config that has the version you want to remove.
  uv run bcsd cache-clear --config-path configs/example.yaml --yes
  ```

## See Also

- [Run the pipeline](run-pipeline.md) — end-to-end walkthrough including quick start
- [Manage the cache](manage-cache.md) — resumability, cache inspection, force recompute
- [Configuration reference](../reference/configuration.md) — `version`, `environment`, and other fields
