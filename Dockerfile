# Pinned arm64 image for AWS Batch task VMs. Replaces Coiled's package sync: the image
# is the environment, resolved from uv.lock so a task VM cannot drift from CI.
#
# Two stages because cartopy publishes no linux-aarch64 wheel and must be compiled
# against GEOS and PROJ. The toolchain stays in the builder; the runtime keeps only the
# shared libraries the compiled extension links against.
FROM --platform=linux/arm64 ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgeos-dev libproj-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, without the project, so source edits do not rebuild cartopy.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# setuptools_scm reads the version from git metadata, which is not copied in. Without
# this the build falls back to "999" and every output store is stamped
# srm_downscaling:version = 999 (see pipeline.py). CI passes the real version.
#
# Scoped to _FOR_SRM: the unscoped variable applies to every setuptools_scm build in the
# environment, which pins cartopy's metadata to this value and fails the lock check.
# Declared below the dependency layer so a version bump does not recompile cartopy.
ARG SRM_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SRM=${SRM_VERSION}

COPY src/ ./src/
RUN uv sync --frozen --no-dev

# Configs last, so editing one rebuilds only this layer rather than re-running the sync.
# They ship because the deploy workflow submits `bcsd validate --config-path configs/...`
# as a Batch job, and the image is always built from the commit that submits it, so the
# two cannot drift.
COPY configs/ ./configs/


FROM --platform=linux/arm64 ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgeos-c1v5 libproj25 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# The venv hard-codes /app/.venv, so the path must match the builder's.
COPY --from=builder /app /app

# Deliberately just the interpreter prefix: AWS Batch has no entryPoint field on
# containerProperties or containerOverrides, only command, so anything baked in here is
# unreachable from a job definition. Keeping it at `uv run` lets one image serve both the
# pipeline stages and the validate steps, each naming its own command.
ENTRYPOINT ["uv", "run", "--no-sync"]
