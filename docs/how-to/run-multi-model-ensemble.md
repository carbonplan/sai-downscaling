# How to Run a Multi-Model Ensemble

The easiest way to process multiple GCMs, variables, ensemble members, and scenarios is a single `bcsd run-matrix` invocation. The orchestrator automatically deduplicates shared work so each intermediate artifact is computed only once regardless of how many combinations need it.

```bash
# 3 GCMs × 1 variable × 3 members × 1 scenario = 9 runs
# Stage 1: 3 obs tasks (one per GCM)
# Stage 2: 9 historical tasks (3 per GCM)
# Stage 3: 9 scenario tasks
uv run bcsd run-matrix \
  --gcm CESM2-WACCM --gcm MIROC-ES2H --gcm UKESM \
  --variable tas \
  --member r1i1p1f1 --member r2i1p1f1 --member r3i1p1f1 \
  --scenario SSP245 \
  --predict-period-start 2015 --predict-period-end 2100 \
  --scratch-dir "s3://carbonplan-scratch/srm/bcsd-cache" \
  --output-dir "s3://carbonplan-scratch/srm/outputs/" \
  --environment production --version v1 \
  --coiled
```

## Dry Run First

Always preview the matrix before submitting to Coiled:

```bash
uv run bcsd run-matrix \
  --gcm CESM2-WACCM --gcm MIROC-ES2H --gcm UKESM \
  --variable tas \
  --member r1i1p1f1 --member r2i1p1f1 --member r3i1p1f1 \
  --scenario SSP245 \
  --predict-period-start 2015 --predict-period-end 2100 \
  --dry-run
```

## Running Stage by Stage

For large ensembles, you can run each stage separately to monitor progress:

```bash
# Stage 1: prepare observations (one per GCM)
uv run bcsd run-matrix \
  --gcm CESM2-WACCM --gcm MIROC-ES2H --gcm UKESM \
  --variable tas \
  --member r1i1p1f1 --member r2i1p1f1 --member r3i1p1f1 \
  --scenario SSP245 \
  --predict-period-start 2015 --predict-period-end 2100 \
  --stage obs --coiled

# Stage 2: fit historical (one per GCM/member)
uv run bcsd run-matrix \
  --gcm CESM2-WACCM --gcm MIROC-ES2H --gcm UKESM \
  --variable tas \
  --member r1i1p1f1 --member r2i1p1f1 --member r3i1p1f1 \
  --scenario SSP245 \
  --predict-period-start 2015 --predict-period-end 2100 \
  --stage historical --coiled

# Stage 3: transform scenarios (one per GCM/member/scenario)
uv run bcsd run-matrix \
  --gcm CESM2-WACCM --gcm MIROC-ES2H --gcm UKESM \
  --variable tas \
  --member r1i1p1f1 --member r2i1p1f1 --member r3i1p1f1 \
  --scenario SSP245 \
  --predict-period-start 2015 --predict-period-end 2100 \
  --stage scenario --coiled
```

## Checking Progress

After each stage, verify progress with `bcsd status`:

```bash
uv run bcsd status \
  --config-path configs/example.yaml \
  --verbose
```

## See Also

- [Run the pipeline](run-pipeline.md) — quick start and single-run guide
- [CLI reference — bcsd run-matrix](../reference/cli.md#bcsd-run-matrix--run-pipeline-over-a-matrix-recommended) — full option listing
- [Pipeline architecture](../explanation/pipeline-architecture.md) — how deduplication works across stages
