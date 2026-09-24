"""Add license and attribution metadata to an icechunk store without rewriting the data.

Writing group attrs touches only each group's ``zarr.json``, so this commits a new snapshot whose
chunk manifests are unchanged. Nothing is re-encoded and no chunk is rewritten, which is what makes
it safe to run against published stores.

The values come from :mod:`saidownscale.licenses` and the rules from
:mod:`saidownscale.store_metadata`. Addresses issues #697 (input) and #700 (output).

Data published before the provenance namespace was renamed needs 2 runs, because a legacy attr is
only removed once its replacement is present. The first run copies ``srm_downscaling:`` across to
``sai_downscaling:`` and leaves both in place, so every reader keeps working. Update anything that
reads the old names, then run again with ``--prune`` to drop them.

This lives in the package rather than in ``scripts/`` so that the AWS Batch image carries it: the
image copies ``src/`` and ``configs/`` only, and every remote task is invoked as
``python -m saidownscale.<module>``. Writes to the published stores need the Source Cooperative
write principal, which the job definition supplies, so a real run belongs on Batch rather than on a
laptop.

Examples
--------
See what would change, without writing::

    uv run python -m saidownscale.apply_store_metadata --store <uri> --branch v1.0.0 --dry-run

Run 1, add metadata and copy provenance onto the current namespace::

    uv run python -m saidownscale.apply_store_metadata --store <uri> --branch v1.0.0 --repair --yes

Run 2, once nothing reads the old names, drop them::

    uv run python -m saidownscale.apply_store_metadata --store <uri> --branch v1.0.0 --prune --yes

An input store needs only one run, because it carries no namespaced provenance to migrate::

    uv run python -m saidownscale.apply_store_metadata --store <uri> --branch main --repair --prune --yes
"""

from __future__ import annotations

import argparse
import re
import sys

import icechunk
import zarr

from saidownscale.config import _icechunk_storage_for_path
from saidownscale.store_metadata import (
    GCMS,
    GroupPlan,
    gcm_from_store,
    plan_group,
    product_from_path,
)


def leaf_group_paths(root: zarr.Group) -> list[str]:
    """Return the paths of groups that directly hold arrays, where a dataset's attrs live.

    Parameters
    ----------
    root : zarr.Group
        Root group of the store.

    Returns
    -------
    list of str
        Group paths, in traversal order. Derived from a single walk: every array's parent is a
        leaf group, so the tree is listed once rather than re-listed per group.
    """
    paths, seen = [], set()
    for name, node in root.members(max_depth=None):
        if not isinstance(node, zarr.Array) or "/" not in name:
            continue
        parent = name.rsplit("/", 1)[0]
        if parent not in seen:
            seen.add(parent)
            paths.append(parent)
    return paths


def build_plans(
    root: zarr.Group, gcm: str, product: str | None, pattern: str | None
) -> tuple[list[GroupPlan], list[tuple[str, str]]]:
    """Plan every matching leaf group, separating the ones that could not be resolved.

    Parameters
    ----------
    root : zarr.Group
        Root group of the store.
    gcm : str
        GCM the store holds.
    product : {"output", "input"} or None
        Override for the product, or None to infer it from each path.
    pattern : str or None
        Regex limiting which group paths are considered, or None for all of them.

    Returns
    -------
    plans : list of GroupPlan
        One plan per resolvable group.
    unresolved : list of tuple of str
        ``(path, reason)`` for each group that could not be planned. These are failures, not
        skips: a group we cannot attribute is one we should not be publishing.
    """
    plans, unresolved = [], []
    for path in leaf_group_paths(root):
        if pattern and not re.search(pattern, path):
            continue
        group = zarr.open_group(root.store, path=path, mode="r")
        try:
            plans.append(
                plan_group(path, dict(group.attrs), gcm, product or product_from_path(path))
            )
        except (ValueError, KeyError) as err:
            unresolved.append((path, str(err)))
    return plans, unresolved


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Return the parsed command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--store", required=True, help="icechunk store URI or local path")
    parser.add_argument("--branch", required=True, help="branch to read and commit to")
    parser.add_argument("--gcm", choices=GCMS, help="override the GCM inferred from --store")
    parser.add_argument(
        "--product",
        choices=("output", "input"),
        help="override the product inferred from each group path",
    )
    parser.add_argument("--match", help="only consider groups whose path matches this regex")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and write nothing")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace attrs that already hold a different value (off by default)",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help=(
            "correct attrs we wrote that are provably wrong, such as a history entry naming a "
            "method the group did not run (off by default)"
        ),
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="delete deprecated attrs we no longer write (off by default, cannot be undone)",
    )
    parser.add_argument("--yes", "-y", action="store_true", help="skip the confirmation prompt")
    parser.add_argument("--message", help="commit message")
    return parser.parse_args(argv)


def _report(
    plans: list[GroupPlan],
    unresolved: list[tuple[str, str]],
    overwrite: bool,
    repair: bool,
    prune: bool,
) -> None:
    """Print the plan for a human to read before anything is written."""
    for plan in plans:
        if not (plan.writes or plan.conflicts or plan.repairs or plan.removals):
            continue
        print(f"  {plan.path}")
        if plan.to_set:
            print(f"      + {' '.join(sorted(plan.to_set))}")
        for key, (current, corrected) in sorted(plan.repairs.items()):
            verb = "repair" if repair else "SKIP"
            print(f"      ~ {key}: {current!r} -> {corrected!r}  [{verb}]")
        for key, (current, wanted) in sorted(plan.conflicts.items()):
            verb = "overwrite" if overwrite else "KEEP"
            # A None current means the attr is absent and held back only because its pair disagrees.
            shown = "<withheld, pairs with a conflict>" if current is None else repr(current)
            print(f"      ! {key}: {shown} -> {wanted!r}  [{verb}]")
        for key, current in sorted(plan.removals.items()):
            verb = "delete" if prune else "SKIP"
            print(f"      - {key}: {current!r}  [{verb}]")
    for path, why in unresolved:
        print(f"  UNRESOLVED {path}: {why}")


def main(argv: list[str] | None = None) -> int:
    """Plan and optionally apply metadata attrs to one store.

    Parameters
    ----------
    argv : list of str or None
        Command line arguments, or None to read ``sys.argv``.

    Returns
    -------
    int
        0 on success, 1 if the user aborted or any group could not be resolved.
    """
    args = _parse_args(argv)

    try:
        gcm = args.gcm or gcm_from_store(args.store)
    except ValueError as err:
        print(f"error: {err}. Pass --gcm.", file=sys.stderr)
        return 1

    repo = icechunk.Repository.open(_icechunk_storage_for_path(args.store))
    root = zarr.open_group(repo.readonly_session(branch=args.branch).store, mode="r")
    plans, unresolved = build_plans(root, gcm, args.product, args.match)

    print(f"store    {args.store}")
    print(f"branch   {args.branch}")
    print(f"gcm      {gcm}")
    changing = [p for p in plans if p.writes or p.conflicts or p.repairs or p.removals]
    print(f"groups   {len(plans)} planned, {len(changing)} with something to change\n")
    _report(plans, unresolved, args.overwrite, args.repair, args.prune)

    conflicted = [p for p in plans if p.conflicts]
    if conflicted and not args.overwrite:
        print(
            f"\n{len(conflicted)} group(s) already hold a different value for at least one attr. "
            "They are left alone. Re-run with --overwrite to replace them."
        )

    repairable = [p for p in plans if p.repairs]
    if repairable and not args.repair:
        print(
            f"\n{len(repairable)} group(s) carry an attr we wrote that is provably wrong. "
            "They are left alone. Re-run with --repair to correct them."
        )

    prunable = [p for p in plans if p.removals]
    if prunable and not args.prune:
        print(
            f"\n{len(prunable)} group(s) carry a deprecated attr. They are left alone. "
            "Re-run with --prune to delete them."
        )

    writers = [
        p
        for p in plans
        if p.writes
        or (p.conflicts and args.overwrite)
        or (p.repairs and args.repair)
        or (p.removals and args.prune)
    ]
    if not writers:
        print("\nnothing to write")
    elif args.dry_run:
        print(f"\ndry run - nothing written ({len(writers)} group(s) would change)")
    else:
        if not args.yes:
            if not sys.stdin.isatty():
                print(
                    f"error: this would change {len(writers)} group(s) and stdin is not a "
                    "terminal. Re-run with --yes to proceed.",
                    file=sys.stderr,
                )
                return 1
            reply = input(f"\nWrite attrs to {len(writers)} group(s) on {args.branch}? [y/N] ")
            if reply.strip().lower() not in ("y", "yes"):
                print("aborted")
                return 1

        session = repo.writable_session(args.branch)
        for plan in writers:
            updates = dict(plan.to_set)
            if args.repair:
                updates.update({k: corrected for k, (_, corrected) in plan.repairs.items()})
            if args.overwrite:
                updates.update({k: wanted for k, (_, wanted) in plan.conflicts.items()})
            group = zarr.open_group(session.store, path=plan.path, mode="a")
            group.attrs.update(updates)
            if args.prune:
                for key in plan.removals:
                    del group.attrs[key]
        snapshot = session.commit(args.message or f"add license and attribution attrs for {gcm}")
        print(f"\ncommitted {snapshot} to {args.branch}: {len(writers)} group(s)")

    if unresolved:
        print(
            f"\nerror: {len(unresolved)} group(s) could not be resolved and were left "
            "unattributed. Record them in saidownscale.licenses, then re-run.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
