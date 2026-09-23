"""Resolve the license and attribution attrs a published store group should carry.

Pure planning only: nothing here opens a store or writes to one, so the rules can be tested
without S3 or icechunk. ``scripts/add_store_metadata.py`` supplies the I/O around it.

Group paths differ between the two store layouts. Output stores nest as
``{method}/{scenario}/{variable}/{member}``, with an extra ``debiased_coarse`` level for the
coarse product; input stores hold one group per scenario at the top level. Rather than index into
either shape, we look for the segment that names a known scenario group, which works for both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from saidownscale.config import (
    ATTR_PREFIX,
    LEGACY_ATTR_PREFIX,
    SCENARIO_TO_GROUP,
    read_attr,
)
from saidownscale.licenses import metadata_attrs

#: Every scenario group a store may hold, used to find the scenario segment of a group path.
SCENARIO_GROUPS = frozenset(SCENARIO_TO_GROUP.values())

#: First path segment of an output group. Input stores start with the scenario instead.
METHODS = frozenset({"bcsd", "qdmsd"})

#: GCM keys, matched against a store URI.
GCMS = ("CESM2-WACCM6", "UKESM1-1-LL")

#: Provenance fields we used to write and no longer do, named without a namespace so both the
#: current and the legacy spelling are caught. ``config_hash`` is a digest of the same config that
#: ``config_json`` records in full beside it, nothing reads it back off a store, and it appears in
#: no cache path, so it recovers nothing that cannot be recomputed.
DEPRECATED_FIELDS = frozenset({"config_hash"})

#: A ``history`` entry, split so the method can be corrected without touching the timestamp or the
#: producer. Both are the true record of what ran and when, even where the method name is wrong.
HISTORY_ENTRY = re.compile(r"^(?P<stamp>.*?: )(?P<method>\S+)(?P<rest> downscaling by .*)$")


@dataclass
class GroupPlan:
    """What one group would gain, keep, or conflict on.

    Parameters
    ----------
    path : str
        Zarr group path within the store.
    gcm : str
        GCM the store holds.
    scenario_group : str
        Scenario group resolved from ``path``.
    product : {"output", "input"}
        Which product this group belongs to.
    to_set : dict of str to str
        Attrs absent from the group, which is all this plan would write.
    unchanged : dict of str to str
        Attrs already holding the wanted value.
    conflicts : dict of str to tuple of str
        Attrs holding a foreign value, mapped to ``(current, wanted)``. Never written unless the
        caller explicitly opts into overwriting.
    repairs : dict of str to tuple of str
        Attrs we wrote ourselves that are provably wrong, mapped to ``(current, corrected)``.
        Kept apart from ``conflicts`` because the correction is derived from the group's own
        attrs rather than guessed, so it can be applied without the blunt overwrite.
    removals : dict of str to str
        Deprecated attrs present on the group, mapped to their current value. Deleting cannot be
        undone by re-running, so these are kept apart from every other category.
    """

    path: str
    gcm: str
    scenario_group: str
    product: str
    to_set: dict[str, str] = field(default_factory=dict)
    unchanged: dict[str, str] = field(default_factory=dict)
    conflicts: dict[str, tuple[str, str]] = field(default_factory=dict)
    repairs: dict[str, tuple[str, str]] = field(default_factory=dict)
    removals: dict[str, str] = field(default_factory=dict)

    @property
    def writes(self) -> bool:
        """Return True when this plan would add at least one attr."""
        return bool(self.to_set)


def gcm_from_store(uri: str) -> str:
    """Infer the GCM from a store URI.

    Both layouts name the GCM in the store filename: ``CESM2-WACCM6.icechunk`` for input and
    ``CESM2-WACCM6-ERA5-global.icechunk`` for output.

    Parameters
    ----------
    uri : str
        Store URI or path.

    Returns
    -------
    str
        The GCM key.

    Raises
    ------
    ValueError
        If no known GCM appears in the URI, so the caller must say which it is.
    """
    for gcm in GCMS:
        if gcm in uri:
            return gcm
    raise ValueError(f"cannot tell which GCM {uri!r} holds")


def scenario_from_path(path: str) -> str | None:
    """Return the scenario group named in a group path, or None if no segment names one.

    Parameters
    ----------
    path : str
        Zarr group path.

    Returns
    -------
    str or None
        The scenario group, or None when the path names none.
    """
    for segment in path.split("/"):
        if segment in SCENARIO_GROUPS:
            return segment
    return None


def product_from_path(path: str) -> str:
    """Return ``"output"`` for a method-prefixed group path, otherwise ``"input"``.

    Parameters
    ----------
    path : str
        Zarr group path.

    Returns
    -------
    str
        Either ``"output"`` or ``"input"``.
    """
    return "output" if path.split("/")[0] in METHODS else "input"


def describe_source(gcm: str, scenario: str, obs_dataset: str, method: str) -> str:
    """Return the CF ``source`` description for a downscaled group.

    Parameters
    ----------
    gcm : str
        GCM the data came from.
    scenario : str
        Scenario as the provenance records it, e.g. ``"G6-1.5K"``.
    obs_dataset : str
        Observation dataset the output was downscaled onto.
    method : str
        Downscaling method, e.g. ``"BCSD"``.

    Returns
    -------
    str
        A one-line description of how the data was produced.
    """
    return f"{gcm} {scenario}, downscaled to the {obs_dataset} grid by the {method} method"


def source_for(existing: dict[str, str], gcm: str) -> str | None:
    """Compose a CF ``source`` for a downscaled group from the provenance already on it.

    Parameters
    ----------
    existing : dict of str to str
        The group's current attrs.
    gcm : str
        GCM the store holds.

    Returns
    -------
    str or None
        A description of how the data was produced, or None when the group lacks the provenance
        to describe itself. Returning None keeps the caller from inventing a value.
    """
    method = read_attr(existing, "downscaling_method")
    obs = read_attr(existing, "observation_dataset")
    scenario = read_attr(existing, "scenario") or "historical"
    if not method or not obs:
        return None
    return describe_source(gcm, scenario, obs, method)


def repair_history(existing: dict[str, str]) -> str | None:
    """Return a corrected ``history`` when it names a method the group did not run.

    Every group written before this was fixed claims ``BCSD downscaling`` regardless of the method
    used, contradicting ``sai_downscaling:downscaling_method`` in the same group. Only that one
    token is replaced: the timestamp and the producing package are the true record of what ran and
    when, so they stay as written.

    Parameters
    ----------
    existing : dict of str to str
        The group's current attrs.

    Returns
    -------
    str or None
        The corrected history, or None when there is nothing to correct or not enough
        information to correct it safely.
    """
    history = existing.get("history")
    method = read_attr(existing, "downscaling_method")
    if not history or not method:
        return None
    match = HISTORY_ENTRY.match(history)
    if match is None or match.group("method") == method:
        return None
    return f"{match.group('stamp')}{method}{match.group('rest')}"


def _plan_provenance_namespace(plan: GroupPlan, existing: dict[str, str]) -> None:
    """Plan the move of v1.0.0 provenance onto the current namespace.

    v1.0.0 is fixed in place rather than regenerated, so its ``srm_downscaling:`` attrs are moved
    across in 2 passes and a group is never left without its provenance. The first pass copies
    each legacy attr to the current namespace, which is purely additive and leaves both in place.
    Once that copy exists, a later pass marks the legacy attr for deletion. A legacy attr is
    therefore only ever removed when its replacement is already present in the same group.

    Fields we have stopped writing are never copied forward. They go straight to removal under
    whichever namespace they appear. A legacy attr whose copy holds a different value is reported
    as a conflict instead of being deleted, because one of the two was edited by hand and picking
    a winner would destroy the evidence.

    Parameters
    ----------
    plan : GroupPlan
        The plan being built, updated in place.
    existing : dict of str to str
        The group's current attrs.
    """
    for key, value in sorted(existing.items()):
        for prefix in (ATTR_PREFIX, LEGACY_ATTR_PREFIX):
            if not key.startswith(prefix):
                continue
            field_name = key[len(prefix) :]
            if field_name in DEPRECATED_FIELDS:
                plan.removals[key] = value
            elif prefix == LEGACY_ATTR_PREFIX:
                current = f"{ATTR_PREFIX}{field_name}"
                if current not in existing:
                    plan.to_set[current] = value
                elif existing[current] == value:
                    plan.removals[key] = value
                else:
                    # The copy disagrees with what it was copied from, so someone edited one of
                    # them. Report it rather than picking a winner and deleting the evidence.
                    plan.conflicts[key] = (value, existing[current])
            break


def plan_group(path: str, existing: dict[str, str], gcm: str, product: str) -> GroupPlan:
    """Work out the attrs one group needs, without touching it.

    Parameters
    ----------
    path : str
        Zarr group path.
    existing : dict of str to str
        The group's current attrs.
    gcm : str
        GCM the store holds.
    product : {"output", "input"}
        Which product this group belongs to.

    Returns
    -------
    GroupPlan
        The additions, matches, and conflicts for this group.

    Raises
    ------
    ValueError
        If the path names no scenario group, or if the path and the group's own
        ``sai_downscaling:scenario`` attr disagree about which scenario it holds.
    KeyError
        If the GCM and scenario pair has no recorded attribution, raised by
        :func:`saidownscale.licenses.attribution_for`.
    """
    scenario_group = scenario_from_path(path)
    if scenario_group is None:
        raise ValueError(f"no scenario group in path {path!r}")

    declared = read_attr(existing, "scenario")
    if declared and SCENARIO_TO_GROUP.get(declared) not in (None, scenario_group):
        raise ValueError(
            f"{path}: path says {scenario_group!r} but attrs say {declared!r}. Refusing to guess."
        )

    target = metadata_attrs(gcm, scenario_group, product=product)
    if product == "output":
        source = source_for(existing, gcm)
        if source is not None:
            target["source"] = source

    plan = GroupPlan(path=path, gcm=gcm, scenario_group=scenario_group, product=product)
    corrected = repair_history(existing)
    if corrected is not None:
        plan.repairs["history"] = (existing["history"], corrected)
    _plan_provenance_namespace(plan, existing)
    for key, value in target.items():
        current = existing.get(key)
        if current is None:
            plan.to_set[key] = value
        elif current == value:
            plan.unchanged[key] = value
        else:
            plan.conflicts[key] = (current, value)
    return plan
