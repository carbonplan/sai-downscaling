import dataclasses

import pytest

from saidownscale.snapshot.tolerances import DEFAULT_TOLERANCE, TOLERANCES, Tolerance, tolerance_for


def test_tolerance_for_looks_up_or_falls_back_and_is_frozen():
    assert tolerance_for("tas") == TOLERANCES["tas"]
    assert tolerance_for("not_a_var") is DEFAULT_TOLERANCE
    with pytest.raises(dataclasses.FrozenInstanceError):
        Tolerance(rtol=1e-5, atol=1e-3).rtol = 2.0


def test_tolerance_table_invariants():
    """pr (kg m-2 s-1) needs a tighter floor; tasmin = tasmax - dtr can drift by both atols."""
    assert set(TOLERANCES) == {"tas", "tasmax", "tasmin", "pr", "rsds", "hurs", "dtr"}
    assert TOLERANCES["pr"].atol < TOLERANCES["tas"].atol
    assert TOLERANCES["tasmin"].atol >= TOLERANCES["tasmax"].atol + TOLERANCES["dtr"].atol
