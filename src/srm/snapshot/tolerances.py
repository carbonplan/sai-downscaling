"""Per-variable tolerance policy for snapshot comparison.

These seed values are deliberately loose enough to absorb floating-point and
dask/regrid nondeterminism across library versions, and tight enough to catch a
real scientific change. They are starting points to be confirmed with Claire and
Ori; ``pr`` is atol-dominant because relative tolerance is meaningless near zero.
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


# Units: tas/tasmax/tasmin/dtr in K, pr in mm/day, rsds in W m-2, hurs in %.
TOLERANCES: dict[str, Tolerance] = {
    "tas": Tolerance(rtol=1e-5, atol=1e-3),
    "tasmax": Tolerance(rtol=1e-5, atol=1e-3),
    "tasmin": Tolerance(rtol=1e-5, atol=1e-3),
    "dtr": Tolerance(rtol=1e-5, atol=1e-3),
    "pr": Tolerance(rtol=0.0, atol=1e-6),
    "rsds": Tolerance(rtol=1e-5, atol=1e-2),
    "hurs": Tolerance(rtol=1e-5, atol=1e-2),
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
