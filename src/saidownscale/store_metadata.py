"""Resolve the license and attribution attrs a published store group should carry.

Pure planning only: nothing here opens a store or writes to one, so the rules can be tested
without S3 or icechunk. :mod:`saidownscale.apply_store_metadata` supplies the I/O around it.

Group paths differ between the 2 store layouts. Output stores nest as
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
    PUBLISHED_SCENARIO_NAMES,
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

#: Provenance fields that describe a scenario run and so do not belong on a historical group.
#: Stage 2 is cached without the scenario in its key, and the historical fit reads no SSP2-4.5 or
#: parent SAI data, yet it used to write the full attr set of whichever scenario run reached it
#: first. A historical group therefore named that run's bridge and parent. Named without a
#: namespace, like :data:`DEPRECATED_FIELDS`.
SCENARIO_ONLY_FIELDS = frozenset(
    {"ssp245_ensemble_member", "sai_parent_scenario", "sai_parent_ensemble_member"}
)

#: Attrs we do not republish, matched whole because they carry no namespace. Kept apart from
#: :data:`DEPRECATED_FIELDS`, which names provenance fields under either namespace, and not all of
#: these were ours to begin with.
#:
#: - ``experiment_lineage`` was ours, derived from ``parent_experiment_id``, which only CMORized
#:   CMIP6 output has. On 3 of the 4 CESM input groups it read "unknown_parent -> " plus the
#:   scenario recorded beside it, and no UKESM group ever had it.
#: - ``logname`` came from the source netCDF and is the personal account name of whoever ran the
#:   simulation. The people behind each run are credited by name in ``attribution``, which is the
#:   right place for it, so republishing a username adds nothing a reader needs.
#: - ``ensemble_derivation_logic`` was rendered at ingest from the suite-to-member lookup tables in
#:   the per-model ETL modules. Those tables are the record that cannot drift from what ran, while
#:   a prose copy in a published store can, so the code is the reference now.
#: Attrs we generated that restate what the code already records, or that name a person.
_DROPPED_OURS = frozenset({"experiment_lineage", "ensemble_derivation_logic", "logname"})

#: Run detail inherited from the source netCDFs, describing how and where a simulation was
#: executed rather than what the data is. ``CESM2-WACCM6/historical`` came through the Pangeo
#: CMIP6 archive and so carries the full CMIP6 global attr set, which is why it held 49 attrs
#: while its siblings held 16. ``status`` is also a privacy fix, since it embeds the email address
#: of whoever created the file. Attrs we also write on our own output, such as ``contact``, are in
#: :data:`_DROPPED_INPUT_ONLY` instead, because this set applies to both products.
_DROPPED_INHERITED = frozenset(
    {
        "activity_id",
        "branch_method",
        "branch_time_in_child",
        "data_specs_version",
        "ensemble_member_source",
        "experiment",
        "experiment_id",
        "external_variables",
        "forcing_index",
        "frequency",
        "grid",
        "grid_label",
        "host",
        "initial_file",
        "initialization_index",
        "mip_era",
        "parent_activity_id",
        "parent_mip_era",
        "parent_source_id",
        "parent_time_units",
        "parent_variant_label",
        "physics_index",
        "product",
        "realm",
        "source_id",
        "source_type",
        "status",
        "sub_experiment",
        "sub_experiment_id",
        "table_id",
        "time_period_freq",
        "topography_file",
    }
)

DROPPED_PLAIN_ATTRS = _DROPPED_OURS | _DROPPED_INHERITED

#: Inherited attrs that add nothing a reader of an input group needs (#708 review). ``source``
#: names only the model component (``CAM``, ``Data from Met Office Unified Model``) that ``model``
#: and ``attribution`` already identify, ``Conventions`` claims a CF version for data we have
#: since reprocessed, and the rest describe the run setup or the modeling center, which the
#: citation in ``attribution`` covers. Output groups keep their own ``Conventions``, ``source``,
#: ``institution`` and ``contact``, which is why these are dropped from input groups only.
_DROPPED_INPUT_ONLY = frozenset(
    {
        "Conventions",
        "case",
        "contact",
        "institution",
        "institution_id",
        "model_doi_url",
        "nominal_resolution",
        "parent_experiment_id",
        "source",
        "um_version",
    }
)

#: ``references`` cited only the upstream simulation for the scenario, although a scenario leaf is
#: also built from the historical run, and from the SSP2-4.5 bridge for a G6 leaf. Rather than
#: list every input, output groups point to our dataset ``doi``, whose record cites them all.
_DROPPED_OUTPUT_ONLY = frozenset({"references"})


def dropped_attrs(product: str) -> frozenset[str]:
    """Return the plain attrs a group of this product must not carry.

    Parameters
    ----------
    product : {"output", "input"}
        Which product the group belongs to.

    Returns
    -------
    frozenset of str
        Attr names to remove wherever they appear.
    """
    only = _DROPPED_INPUT_ONLY if product == "input" else _DROPPED_OUTPUT_ONLY
    return DROPPED_PLAIN_ATTRS | only


#: Coordinate attrs that duplicate a group attr we drop, as ``(coord attr, group attr)``.
#: ``apply_ensemble_provenance`` wrote the same sentence twice, to the group as
#: ``ensemble_derivation_logic`` and to the coordinate as ``derivation_method``, so dropping only
#: the group copy would leave the text published.
#:
#: The copy is removed only where it matches the group value exactly. That matters: one published
#: coordinate carries a different sentence naming which CESM historical members are complete and
#: which hold NaN where a variable is missing. That is a data-quality fact with no other home, and
#: a rule keyed on the attr name rather than its value would have deleted it.
DUPLICATED_COORD_ATTRS: dict[str, tuple[str, str]] = {
    "ensemble_member": ("derivation_method", "ensemble_derivation_logic"),
}

#: A ``history`` entry, split so the method can be corrected without touching the timestamp or the
#: producer. Both are the true record of what ran and when, even where the method name is wrong.
HISTORY_ENTRY = re.compile(r"^(?P<stamp>.*?: )(?P<method>\S+)(?P<rest> downscaling by .*)$")

#: Attrs that are only true together, so neither is written while the other is in dispute. A group
#: whose inherited ``license`` we leave alone must not gain our ``license_url``, because the pair
#: would then name 2 different licenses, which is worse than the incomplete metadata we found.
PAIRED_ATTRS: tuple[frozenset[str], ...] = (frozenset({"license", "license_url"}),)

#: How we built an input store. Every input group records ``processing_steps``, so the sentence can
#: always refer the reader to it.
DATA_SOURCE = (
    "This icechunk store was created by ingesting netCDF files and processing following the steps "
    "outlined in the `processing_steps` attribute. See "
    "https://github.com/carbonplan/sai-downscaling for more information."
)

#: ``processing_steps`` for each UKESM input group, spelled the way the CESM ETL spells them. The
#: UKESM ETL never recorded them, so they are written here, where both the ETL and the metadata
#: pass read them. Historical files arrive without duplicate time steps, so that group skips the
#: deduplication.
UKESM_PROCESSING_STEPS: dict[str, str] = {
    "historical": (
        "lon_to_180, lat_lon_sort, trim_negative_precip, convert_calendar_to_proleptic_gregorian"
    ),
    "ssp245": (
        "time_drop_duplicates, lon_to_180, lat_lon_sort, trim_negative_precip, "
        "convert_calendar_to_proleptic_gregorian"
    ),
    "g6_1p5k": (
        "time_drop_duplicates, lon_to_180, lat_lon_sort, trim_negative_precip, "
        "convert_calendar_to_proleptic_gregorian"
    ),
}


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
    conflicts : dict of str to tuple
        Attrs holding a foreign value, mapped to ``(current, wanted)``. Never written unless the
        caller explicitly opts into overwriting. ``current`` is None where the attr is absent and
        withheld only because the attr it pairs with is itself in conflict.
    repairs : dict of str to tuple of str
        Attrs we wrote ourselves that are provably wrong, mapped to ``(current, corrected)``.
        Kept apart from ``conflicts`` because the correction is derived from the group's own
        attrs rather than guessed, so it can be applied without the blunt overwrite.
    removals : dict of str to str
        Deprecated attrs present on the group, mapped to their current value. Deleting cannot be
        undone by re-running, so these are kept apart from every other category.
    coord_removals : dict of str to dict
        Attrs to remove from a child coordinate, keyed by coordinate name. Held separately because
        they live on an array rather than on the group, so the caller has to open a different node
        to apply them.
    """

    path: str
    gcm: str
    scenario_group: str
    product: str
    to_set: dict[str, str] = field(default_factory=dict)
    unchanged: dict[str, str] = field(default_factory=dict)
    conflicts: dict[str, tuple[str | None, str]] = field(default_factory=dict)
    repairs: dict[str, tuple[str, str]] = field(default_factory=dict)
    removals: dict[str, str] = field(default_factory=dict)
    coord_removals: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def writes(self) -> bool:
        """Return True when this plan would add at least one attr."""
        return bool(self.to_set)


def gcm_from_store(uri: str) -> str:
    """Infer the GCM from a store URI.

    Both layouts name the GCM in the store filename: ``CESM2-WACCM6.icechunk`` for input and
    ``CESM2-WACCM6-ERA5-global.icechunk`` for output. Matching on the URI is therefore enough, and a
    caller only has to name the GCM for a store whose path does not follow either convention.

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


def scenario_attr_key(product: str) -> str:
    """Return the attr that records the scenario for a product.

    Output groups namespace their provenance, so the scenario lives under
    :data:`~saidownscale.config.ATTR_PREFIX` there. Input groups carry a plain ``scenario``
    inherited from the ETL, which predates the namespace and is what readers of those stores
    already use.

    Parameters
    ----------
    product : {"output", "input"}
        Which product the group belongs to.

    Returns
    -------
    str
        The attr name.
    """
    return "scenario" if product == "input" else f"{ATTR_PREFIX}scenario"


def names_this_group(scenario: str, scenario_group: str) -> bool:
    """Return True when a recorded scenario names this group, in either spelling.

    A store we have already patched records the published name, so a re-run has to recognize its
    own output. This is checked by comparison rather than by adding the published names to
    ``SCENARIO_TO_GROUP``: that map decides cache paths, so a published name accepted there would
    give one group path a second cache identity. It also cannot be inverted safely, since
    ``ssp245`` and ``esgf_ssp245`` both publish ``SSP2-4.5``.

    Parameters
    ----------
    scenario : str
        The scenario a group records, in either spelling.
    scenario_group : str
        The group resolved from the path.

    Returns
    -------
    bool
        True when the 2 refer to the same scenario.
    """
    return SCENARIO_TO_GROUP.get(
        scenario
    ) == scenario_group or scenario == PUBLISHED_SCENARIO_NAMES.get(scenario_group)


def published_name(scenario: str) -> str:
    """Return the published spelling of a scenario name.

    A name we do not recognize is handed back unchanged, so an unfamiliar value is reported as a
    disagreement elsewhere rather than silently rewritten here. Guessing at a spelling would invent
    provenance, which is worse than refusing the group and saying which one it was.

    Parameters
    ----------
    scenario : str
        A scenario name in any spelling :data:`~saidownscale.config.SCENARIO_TO_GROUP` accepts.

    Returns
    -------
    str
        The published spelling.
    """
    group = SCENARIO_TO_GROUP.get(scenario)
    return PUBLISHED_SCENARIO_NAMES[group] if group else scenario


def recorded_scenario(existing: dict[str, str], product: str) -> str | None:
    """Return the scenario a group records, or None where it records none.

    Parameters
    ----------
    existing : dict of str to str
        The group's current attrs.
    product : {"output", "input"}
        Which product the group belongs to.

    Returns
    -------
    str or None
        The recorded scenario, read from whichever attr that product uses.
    """
    if product == "input":
        return existing.get("scenario")
    return read_attr(existing, "scenario")


def input_attrs(gcm: str, scenario_group: str) -> dict[str, str]:
    """Return every attr we write on an input group.

    Shared by the ETL and :func:`plan_group`, so a fresh ingest and the metadata pass agree.

    Parameters
    ----------
    gcm : str
        GCM key, as in :data:`GCMS`.
    scenario_group : str
        Scenario group as the stores spell it.

    Returns
    -------
    dict of str to str
        License, citation and terms, plus ``data_source`` and, for UKESM, ``processing_steps``.
    """
    attrs = metadata_attrs(gcm, scenario_group, product="input")
    attrs["data_source"] = DATA_SOURCE
    if gcm == "UKESM1-1-LL":
        attrs["processing_steps"] = UKESM_PROCESSING_STEPS[scenario_group]
    return attrs


def source_for(existing: dict[str, str], gcm: str, scenario: str | None = None) -> str | None:
    """Compose a CF ``source`` for a downscaled group from the provenance already on it.

    Parameters
    ----------
    existing : dict of str to str
        The group's current attrs.
    gcm : str
        GCM the store holds.
    scenario : str, optional
        Scenario to describe the data as, overriding what the group records. Used where the
        recorded scenario is the run's rather than the data's, which is every published historical
        group. Without it the description would call historical data ``G6-1.5K``.

    Returns
    -------
    str or None
        A description of how the data was produced, or None when the group lacks the provenance
        to describe itself. Returning None keeps the caller from inventing a value.
    """
    method = read_attr(existing, "downscaling_method")
    obs = read_attr(existing, "observation_dataset")
    recorded = scenario or read_attr(existing, "scenario") or "historical"
    if not method or not obs:
        return None
    return describe_source(gcm, recorded, obs, method)


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


def _plan_provenance_namespace(
    plan: GroupPlan,
    existing: dict[str, str],
    corrections: dict[str, str],
    unwanted: frozenset[str] = DEPRECATED_FIELDS,
) -> None:
    """Plan the move of v1.0.0 provenance onto the current namespace.

    v1.0.0 is fixed in place rather than regenerated, so its ``srm_downscaling:`` attrs are moved
    across in 2 passes and a group is never left without its provenance. The first pass copies
    each legacy attr to the current namespace, which is purely additive and leaves both in place.
    Once that copy exists, a later pass marks the legacy attr for deletion. A legacy attr is
    therefore only ever removed when its replacement is already present in the same group.

    Fields we have stopped writing are never copied forward. They go straight to removal under
    whichever namespace they appear. A legacy attr whose copy holds a different value is reported
    as a conflict instead of being deleted, because one of the 2 was edited by hand and picking
    a winner would destroy the evidence.

    ``corrections`` is what keeps that guard from firing on our own repairs. A field we correct ends
    up with a copy that deliberately differs from the legacy original, which is indistinguishable
    from a hand edit unless the value we meant to write is named. Where it is named, the stale
    original is superseded rather than in dispute, so it is removable and is never copied forward.

    Parameters
    ----------
    plan : GroupPlan
        The plan being built, updated in place.
    existing : dict of str to str
        The group's current attrs.
    corrections : dict of str to str
        Current-namespace keys mapped to the value we consider authoritative, regardless of what
        the group records.
    unwanted : frozenset of str
        Field names, without a namespace, that this group must not carry. Each goes straight to
        removal under either namespace and is never copied forward.
    """
    for key, value in sorted(existing.items()):
        for prefix in (ATTR_PREFIX, LEGACY_ATTR_PREFIX):
            if not key.startswith(prefix):
                continue
            field_name = key[len(prefix) :]
            if field_name in unwanted:
                plan.removals[key] = value
            elif prefix == LEGACY_ATTR_PREFIX:
                current = f"{ATTR_PREFIX}{field_name}"
                superseded = current in corrections and corrections[current] != value
                if current not in existing:
                    # Copying a value a repair is about to correct would only have to be undone,
                    # and the repair supplies the right one under the current namespace anyway.
                    if not superseded:
                        plan.to_set[current] = value
                elif existing[current] == value or (
                    current in corrections and existing[current] == corrections[current]
                ):
                    plan.removals[key] = value
                else:
                    # The copy disagrees with what it was copied from, so someone edited one of
                    # them. Report it rather than picking a winner and deleting the evidence.
                    plan.conflicts[key] = (value, existing[current])
            break


def _hold_paired_attrs(plan: GroupPlan) -> None:
    """Withhold one half of a paired attr while the other half is in conflict.

    Without this, a group whose ``license`` we are leaving alone still gains our ``license_url``,
    and the 2 then resolve to different licenses. The published CESM historical input group is
    exactly that case: it inherits the CMIP6 CC BY-SA statement from its netCDF while our table
    asserts CC BY 4.0. The withheld half moves into ``conflicts`` so it is written under the same
    explicit opt-in as the half it depends on, never on its own.

    Parameters
    ----------
    plan : GroupPlan
        The plan being built, updated in place.
    """
    for pair in PAIRED_ATTRS:
        if not pair & plan.conflicts.keys():
            continue
        for key in sorted(pair & plan.to_set.keys()):
            plan.conflicts[key] = (None, plan.to_set.pop(key))


def plan_group(
    path: str,
    existing: dict[str, str],
    gcm: str,
    product: str,
    coord_attrs: dict[str, dict[str, str]] | None = None,
) -> GroupPlan:
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
    coord_attrs : dict of str to dict, optional
        Attrs of the group's child coordinates, keyed by coordinate name. Needed because a
        coordinate can hold a duplicate of a group attr we drop, and the group's own attrs do not
        reveal it.

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

    declared = recorded_scenario(existing, product)
    if declared and not names_this_group(declared, scenario_group):
        # Stage 2 is cached per GCM, variable, method, and member with no scenario in the key, so
        # whichever scenario's run wrote the historical leg first stamped its own config onto it.
        # The path is the reliable witness there, and ``config_json`` still holds the run's own
        # scenario, so nothing is lost by correcting the attr. Any other disagreement is one we
        # cannot explain, so it still refuses rather than guessing.
        if scenario_group != "historical":
            raise ValueError(
                f"{path}: path says {scenario_group!r} but attrs say {declared!r}. "
                "Refusing to guess."
            )

    published = PUBLISHED_SCENARIO_NAMES[scenario_group]
    if product == "output":
        target = metadata_attrs(gcm, scenario_group, product=product)
        source = source_for(existing, gcm, published)
        if source is not None:
            target["source"] = source
    else:
        target = input_attrs(gcm, scenario_group)

    plan = GroupPlan(path=path, gcm=gcm, scenario_group=scenario_group, product=product)
    corrected = repair_history(existing)
    if corrected is not None:
        plan.repairs["history"] = (existing["history"], corrected)
    if declared is not None and declared != published:
        # Either the group names a scenario it did not hold, which only happens on a historical
        # group, or it names the right one by an older spelling. Both are corrections rather than
        # additions, so both wait for an explicit flag.
        plan.repairs[scenario_attr_key(product)] = (declared, published)
    # The group's own path decides which scenario it holds, so the published name for that path is
    # authoritative whether or not the group currently agrees with it.
    corrections = {f"{ATTR_PREFIX}scenario": published}
    unwanted = DEPRECATED_FIELDS
    historical_output = product == "output" and scenario_group == "historical"
    if historical_output:
        unwanted = DEPRECATED_FIELDS | SCENARIO_ONLY_FIELDS
        # A historical group is shared by every scenario run of its historical member, so the
        # member it holds is the one its path names, not the member of the run that wrote it.
        member = path.rstrip("/").rsplit("/", 1)[-1]
        member_key = f"{ATTR_PREFIX}ensemble_member"
        corrections[member_key] = member
        declared_member = read_attr(existing, "ensemble_member")
        if declared_member is not None and declared_member != member:
            plan.repairs[member_key] = (declared_member, member)
    # The parent names a different scenario than the group, so the path cannot settle it, but it
    # still has to be spelled the published way or one group carries 2 conventions at once.
    parent = read_attr(existing, "sai_parent_scenario")
    if parent is not None and not historical_output:
        parent_key = f"{ATTR_PREFIX}sai_parent_scenario"
        corrections[parent_key] = published_name(parent)
        if corrections[parent_key] != parent:
            plan.repairs[parent_key] = (parent, corrections[parent_key])
    _plan_provenance_namespace(plan, existing, corrections, unwanted)
    for key, value in target.items():
        current = existing.get(key)
        if current is None:
            plan.to_set[key] = value
        elif current == value:
            plan.unchanged[key] = value
        else:
            plan.conflicts[key] = (current, value)
    for key in sorted(dropped_attrs(product) & existing.keys()):
        plan.removals[key] = existing[key]
    for coord, (coord_key, group_key) in DUPLICATED_COORD_ATTRS.items():
        present = (coord_attrs or {}).get(coord, {})
        if coord_key in present and present[coord_key] == existing.get(group_key):
            plan.coord_removals.setdefault(coord, {})[coord_key] = present[coord_key]
    _hold_paired_attrs(plan)
    return plan
