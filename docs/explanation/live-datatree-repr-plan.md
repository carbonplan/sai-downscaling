# Plan: live DataTree repr on docs index (not yet implemented)

Goal: embed the interactive xarray `_repr_html_()` for the production DataTree
on the docs index page, regenerated automatically so it never goes stale.

## Why not now
Deferred — scoped as a future PR, not bundled with the README proofread work.

## Constraints
- GitHub markdown (README.md) sanitizes `<style>`/most classes — the rich
  xarray repr won't survive there. Sphinx-built docs pages are not sanitized,
  so this only targets `docs/index.md`, not the README.
- The icechunk output branch is the release tag (`setuptools_scm` version via
  `saidownscale.__version__`), written by the `production` job in `deploy.yml` on
  `release: published`. Docs must read that same branch/version, not `main`.
- `docs.yml` currently only rebuilds on push/PR touching
  `docs/**`, `src/**`, `pyproject.toml` — a release with no code diff would
  never trigger a rebuild, so the repr would go stale silently.

## Proposed pieces
1. `docs/_scripts/generate_datatree_repr.py` — opens the production store
   read-only (`icechunk.s3_storage(..., anonymous=True)`,
   `repo.readonly_session(branch=saidownscale.__version__)`), metadata-only
   (`consolidated=False`, no `.load()`), writes `dt._repr_html_()` to
   `docs/_templates/datatree-repr.html`.
2. `docs/index.md` — include the generated file via MyST raw directive:
   ```{raw} html
   :file: _templates/datatree-repr.html
   ```
3. `.github/workflows/docs.yml` — add a `workflow_run` trigger on `deploy`
   completion, gated to `conclusion == 'success' && event == 'release'`, so
   the docs rebuild fires only after the production write actually lands
   (the production job can take hours). Checkout the release tag's ref on
   that path.
4. Add the generation script as a build step before `sphinx-build`.

## Open questions for the future PR
- Should the repr cover all production GCM stores (currently just
  CESM2-WACCM is live) or one per model once MIROC-ES2H/UKESM ship?
- Confirm `readonly_session(branch=saidownscale.__version__)` matches what
  `deploy.yml`'s snapshot/production jobs actually name the branch at
  release time (should default to the version when `--branch` isn't passed).
