"""Per-variable tolerance policy for snapshot comparison.

These seed values are deliberately loose enough to absorb floating-point and
dask/regrid nondeterminism across library versions, and tight enough to catch a
real scientific change. Every variable pairs a small ``atol`` floor (which dominates near zero,
where the relative term vanishes) with a nonzero ``rtol`` slope (which absorbs
roundoff on large values, since floating-point noise scales with magnitude);
``pr`` needs a much tighter ``atol`` than the loose seed because its stored
values are small in ``kg m-2 s-1``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerance:
    """Absolute and relative tolerance for one variable.

    Mirrors :func:`xarray.testing.assert_allclose` semantics: a cell passes when
    ``abs(candidate - snapshot) <= atol + rtol * abs(snapshot)``.

    Parameters
    ----------
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance, in the variable's stored units.
    """

    rtol: float
    atol: float


# Units: tas/tasmax/tasmin/dtr in K, pr in kg m-2 s-1, rsds in W m-2, hurs in %.
# rtol is uniform; atol is the per-variable knob, sized to each variable's
# stored magnitude. The 1e-3 temperature group is the baseline; entries that
# deviate carry a trailing note.
TOLERANCES: dict[str, Tolerance] = {
    "tas": Tolerance(rtol=1e-5, atol=1e-3),
    "tasmax": Tolerance(rtol=1e-5, atol=1e-3),
    # tasmin is derived as tasmax - dtr (see srm.downscaling_utils.derive_tasmin), so its
    # per-cell drift vs the snapshot is Δtasmax - Δdtr and can reach atol(tasmax) +
    # atol(dtr). Its atol is the sum of its inputs' (2e-3) so shared temperature noise
    # that tasmax and dtr each absorb never trips tasmin as a false positive; a real
    # change still shows up far above this floor.
    "tasmin": Tolerance(rtol=1e-5, atol=2e-3),  # = atol(tasmax) + atol(dtr): derived field
    "dtr": Tolerance(rtol=1e-5, atol=1e-3),
    "pr": Tolerance(rtol=1e-5, atol=1e-10),  # tighter: pr values are tiny (~1e-5 kg m-2 s-1)
    "rsds": Tolerance(rtol=1e-5, atol=1e-2),  # looser: rsds spans a large range (~0-1000 W m-2)
    "hurs": Tolerance(rtol=1e-5, atol=1e-2),  # looser: hurs spans ~0-100 %
}

DEFAULT_TOLERANCE = Tolerance(rtol=1e-5, atol=1e-3)


def tolerance_for(variable: str) -> Tolerance:
    """Return the tolerance for ``variable``, or :data:`DEFAULT_TOLERANCE`.

    Parameters
    ----------
    variable : str
        Canonical variable name (e.g. ``"tas"``).

    Returns
    -------
    Tolerance
        The configured tolerance, or the default for unknown variables.
    """
    return TOLERANCES.get(variable, DEFAULT_TOLERANCE)
