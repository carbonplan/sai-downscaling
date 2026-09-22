"""Resolve the license and attribution attrs a published store group should carry.

Pure planning only: nothing here opens a store or writes to one, so the rules can be tested
without S3 or icechunk. ``scripts/add_store_metadata.py`` supplies the I/O around it.

Group paths differ between the two store layouts. Output stores nest as
``{method}/{scenario}/{variable}/{member}``, with an extra ``debiased_coarse`` level for the
coarse product; input stores hold one group per scenario at the top level. Rather than index into
either shape, we look for the segment that names a known scenario group, which works for both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from saidownscale.config import SCENARIO_TO_GROUP
from saidownscale.licenses import metadata_attrs

#: Every scenario group a store may hold, used to find the scenario segment of a group path.
SCENARIO_GROUPS = frozenset(SCENARIO_TO_GROUP.values())

#: First path segment of an output group. Input stores start with the scenario instead.
METHODS = frozenset({"bcsd", "qdmsd"})

#: GCM keys, matched against a store URI.
GCMS = ("CESM2-WACCM6", "UKESM1-1-LL")


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
        Attrs holding a different value, mapped to ``(current, wanted)``. Never written unless
        the caller explicitly opts into overwriting.
    """

    path: str
    gcm: str
    scenario_group: str
    product: str
    to_set: dict[str, str] = field(default_factory=dict)
    unchanged: dict[str, str] = field(default_factory=dict)
    conflicts: dict[str, tuple[str, str]] = field(default_factory=dict)

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
    method = existing.get("srm_downscaling:downscaling_method")
    obs = existing.get("srm_downscaling:observation_dataset")
    scenario = existing.get("srm_downscaling:scenario") or "historical"
    if not method or not obs:
        return None
    return f"{gcm} {scenario}, downscaled to the {obs} grid by the {method} method"


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
        ``srm_downscaling:scenario`` attr disagree about which scenario it holds.
    KeyError
        If the GCM and scenario pair has no recorded attribution, raised by
        :func:`saidownscale.licenses.attribution_for`.
    """
    scenario_group = scenario_from_path(path)
    if scenario_group is None:
        raise ValueError(f"no scenario group in path {path!r}")

    declared = existing.get("srm_downscaling:scenario")
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
    for key, value in target.items():
        current = existing.get(key)
        if current is None:
            plan.to_set[key] = value
        elif current == value:
            plan.unchanged[key] = value
        else:
            plan.conflicts[key] = (current, value)
    return plan
