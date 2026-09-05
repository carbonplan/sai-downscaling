# Regenerate Input Data

Re-process raw source files into the icechunk stores the downscaling pipeline reads from. Run this when:

- New ensemble members or variables were added to an existing scenario
- A raw source file was corrected upstream and needs re-ingestion
- An icechunk store is corrupted or accidentally deleted

There are two workflows: one for GCM datasets (CESM2-WACCM, UKESM, NASA-NEX)
and one for ERA5. Use the appropriate workflow for the dataset you want to regenerate.

---

## GCM datasets — `process input data`

### Triggering the workflow

1. Go to **Actions → process input data → Run workflow**
2. Select a **GCM** from the dropdown
3. Enter one or more **scenarios** as a comma-separated string (see valid values below)
4. Optionally pass **extra flags** to the processing command (e.g. `--subset`)
5. Click **Run workflow**

### Inputs

| Input | Required | Description |
|-------|----------|-------------|
| `gcm` | yes | GCM to process. One of `CESM2-WACCM`, `UKESM`, `NASA-NEX` |
| `scenario` | yes | Comma-separated scenario(s). See valid values below |
| `extra_flags` | no | Additional flags passed to the processing script (e.g. `--subset`) |

### Valid scenarios

| GCM | Valid scenarios |
|-----|----------------|
| `CESM2-WACCM` | `historical`, `ssp245`, `G6-1.5K`, `G6-1.5K-END` |
| `UKESM` | `historical`, `SSP245`, `G6-1.5K` |
| `NASA-NEX` | `historical`, `SSP245` |

Scenario names are case-sensitive and must match the values above exactly. To process multiple
scenarios in one trigger, pass them comma-separated: e.g. `historical,ssp245`.

---

## ERA5 — `process ERA5 input data`

ERA5 has its own dedicated workflow with variable- and time-range controls.

### Triggering the workflow

1. Go to **Actions → process ERA5 input data → Run workflow**
2. Fill in the inputs below and click **Run workflow**

### Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `variables` | yes | `all` | Comma-separated CMIP6 variable names, or `all`. Valid: `tas`, `tasmin`, `tasmax`, `pr`, `rsds`, `rlds`, `ps`, `hurs` |
| `start_year` | no | `1950` | First year to include (inclusive) |
| `end_year` | no | `2014` | Last year to include (inclusive) |
| `dry_run` | no | `false` | Run transforms on a short sample and print results without writing |
| `commit_message` | no | _(variable name)_ | Custom icechunk commit message |

---

## Job summary

Once processing completes, the workflow opens the written icechunk store and appends the
xarray `repr` to the [job summary](https://github.blog/news-insights/product-news/supercharging-github-actions-with-job-summaries/).
It looks something like this:

```text
<xarray.DataTree>
Group: /
├── Group: historical
│   Dimensions: (ensemble_member: 3, time: 23741, lat: 192, lon: 288)
│   Coordinates:
│     * ensemble_member  (ensemble_member) <U10 'r1i1p1f1' 'r2i1p1f1' 'r3i1p1f1'
│     * time             (time) datetime64[ns] 1950-01-01 ... 2014-12-31
│   Data variables:
│       tas, pr, rsds, ...
├── Group: ssp245
│   ...
└── Group: g6_1p5k
    ...
```

This gives a quick sanity check on dimensions, ensemble members, and group structure without
opening the store manually. NASA-NEX does not produce a summary (its output is a virtual store).
