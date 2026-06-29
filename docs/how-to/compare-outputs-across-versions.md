# How to Compare Outputs Across Code Versions

A common development workflow is to run the pipeline on a small test region with the current
code, run it again after merging new changes, and compare the two sets of outputs to validate
that the changes behave as expected. The `branch` field is the key mechanism for this: each run
writes to a completely isolated icechunk branch within the same store, so both versions of the
data coexist and can be compared at any time without touching different S3 paths.

## Step 1 — Run with the current code on a test region

Pick a small region (`subset_bounds`) and use `environment: qa` so outputs stay isolated from
production data. Use a branch string that describes the code state, e.g. the branch name or a
short commit hash.

```bash
# Run the pipeline with the current code (e.g. pre-merge main)
uv run bcsd run-matrix \
  --gcm CESM2-WACCM \
  --variable tas \
  --member r1i1p1f1 \
  --scenario ssp245 \
  --predict-period-start 2015 --predict-period-end 2100 \
  --subset-bounds '-35,-22,16,33' \
  --scratch-dir "s3://carbonplan-scratch/srm/bcsd-cache" \
  --output-dir "s3://carbonplan-scratch/srm/outputs/" \
  --environment qa --branch main-baseline
```

Outputs will be written to branch `main-baseline` of:
`s3://carbonplan-scratch/srm/outputs/qa/CESM2-WACCM-ERA5-lat-35.0to-22.0_lon16.0to33.0.icechunk`

## Step 2 — Merge the new code and run again with a new branch

After merging (or checking out) the new code, re-run with a different `--branch`. Because the
branch is different, all three stages run from scratch on the same test region, producing a
fully independent dataset written to a separate icechunk branch of the same store.

```bash
# Run the pipeline with the new code (e.g. after merging a refactor branch)
uv run bcsd run-matrix \
  --gcm CESM2-WACCM \
  --variable tas \
  --member r1i1p1f1 \
  --scenario ssp245 \
  --predict-period-start 2015 --predict-period-end 2100 \
  --subset-bounds '-35,-22,16,33' \
  --scratch-dir "s3://carbonplan-scratch/srm/bcsd-cache" \
  --output-dir "s3://carbonplan-scratch/srm/outputs/" \
  --environment qa --branch refactor-icechunk
```

Outputs will be written to branch `refactor-icechunk` of the same store.

## Step 3 — Compare the two outputs

Both datasets are now available on their respective branches and can be loaded and compared
side by side:

```python
import icechunk
import xarray as xr


def open_branch(output_dir, environment, gcm, obs_dataset, subset_id, branch, scenario_group, variable, member):
    store_path = (
        f"{output_dir}/{environment}/{gcm}-{obs_dataset}-{subset_id}.icechunk"
    )
    bucket, _, prefix = store_path.removeprefix("s3://").partition("/")
    storage = icechunk.s3_storage(bucket=bucket, prefix=prefix, from_env=True)
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session(branch=branch)
    group = f"{scenario_group}/{variable}/{member}"
    return xr.open_zarr(session.store, group=group, consolidated=False, zarr_format=3, chunks="auto")


kwargs = dict(
    output_dir="s3://carbonplan-scratch/srm/outputs",
    environment="qa",
    gcm="CESM2-WACCM",
    obs_dataset="ERA5",
    subset_id="lat-35.0to-22.0_lon16.0to33.0",
    scenario_group="ssp245",
    variable="tas",
    member="r1i1p1f1",
)

ds_baseline = open_branch(**kwargs, branch="main-baseline")
ds_new      = open_branch(**kwargs, branch="refactor-icechunk")

diff = ds_new["tas"] - ds_baseline["tas"]
print(diff.max().values, diff.min().values)  # should be ~0 for a pure refactor
```

## Tips

- Keep `environment: qa` for all test runs so they never touch production paths.
- Use descriptive `--branch` strings (branch names, commit hashes, date stamps) rather than
  `v1`/`v2` so it is always clear which code produced which data.
- Use `bcsd status` to confirm both branches completed before comparing:

  ```bash
  uv run bcsd status --config-path configs/example.yaml --branch main-baseline
  uv run bcsd status --config-path configs/example.yaml --branch refactor-icechunk
  ```

- Once you are done comparing, clean up test artifacts with `bcsd cache-clear`:

  ```bash
  uv run bcsd cache-clear --config-path configs/example.yaml --yes
  ```

## See Also

- [Run the pipeline](run-pipeline.md) — end-to-end walkthrough including quick start
- [Manage the cache](manage-cache.md) — resumability, cache inspection, force recompute
- [Configuration reference](../reference/configuration.md) — `branch`, `environment`, and other fields
