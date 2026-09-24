---
orphan: true
---

# Regenerate input data

Re-process raw source files into the icechunk stores the downscaling pipeline reads from. Run this
when:

- new ensemble members or variables were added to an existing scenario
- a raw source file was corrected upstream and needs re-ingestion
- an icechunk store is corrupted or accidentally deleted

We run 2 workflows, and which one you want depends on whether the dataset is global climate model
(GCM) output or observations:

| If you want to regenerate | Use |
| --- | --- |
| A GCM dataset: CESM2-WACCM, UKESM, or NASA-NEX | The `process input data` workflow |
| ERA5 observations | The `process ERA5 input data` workflow |

---

## GCM datasets: the `process input data` workflow

### Trigger the workflow

1. Go to **Actions → process input data → Run workflow**
2. Select a **GCM** from the dropdown
3. Enter 1 or more **scenarios** as a comma-separated string (see valid values below)
4. Optionally pass **extra flags** to the processing command, such as `--subset`
5. Click **Run workflow**

### Inputs

The workflow takes 3 inputs:

| Input | Required | Description |
| --- | --- | --- |
| `gcm` | yes | GCM to process. One of `CESM2-WACCM`, `UKESM`, `NASA-NEX` |
| `scenario` | yes | Comma-separated scenarios. See valid values below |
| `extra_flags` | no | Additional flags passed to the processing script, such as `--subset` |

### Valid scenarios

Each GCM accepts its own set of scenario names:

| GCM | Valid scenarios |
| --- | --- |
| `CESM2-WACCM` | `historical`, `ssp245`, `G6-1.5K`, `G6-1.5K-END` |
| `UKESM` | `historical`, `SSP245`, `G6-1.5K` |
| `NASA-NEX` | `historical`, `SSP245` |

Scenario names are case-sensitive and must match the values above exactly. To process several
scenarios in one trigger, pass them comma-separated, such as `historical,ssp245`.

---

## ERA5: the `process ERA5 input data` workflow

ERA5 has its own workflow, because it needs controls the GCM workflow does not: a variable list and
a time range. The inputs below cover both.

### Trigger the workflow

1. Go to **Actions → process ERA5 input data → Run workflow**
2. Fill in the inputs below and click **Run workflow**

### Inputs

Only `variables` is required, and the rest default to a full 1950 to 2014 run:

| Input | Required | Default | Description |
| --- | --- | --- | --- |
| `variables` | yes | `all` | Comma-separated CMIP6 variable names, or `all`. Valid: `tas`, `tasmin`, `tasmax`, `pr`, `rsds`, `rlds`, `ps`, `hurs` |
| `start_year` | no | `1950` | First year to include (inclusive) |
| `end_year` | no | `2014` | Last year to include (inclusive) |
| `dry_run` | no | `false` | Run transforms on a short sample and print results without writing |
| `commit_message` | no | _(variable name)_ | Custom icechunk commit message |

---

## Job summary

Once processing completes, the workflow opens the written icechunk store and appends the xarray
`repr` to the
[job summary](https://github.blog/news-insights/product-news/supercharging-github-actions-with-job-summaries/).
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

That gives you a quick sanity check on dimensions, ensemble members, and group structure without
opening the store yourself. NASA-NEX produces no summary, because its output is a virtual store.
