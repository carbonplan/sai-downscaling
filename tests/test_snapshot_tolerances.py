from saidownscale.snapshot.tolerances import DEFAULT_TOLERANCE, TOLERANCES, Tolerance, tolerance_for


def test_known_variable_returns_its_tolerance():
    assert tolerance_for("tas") == TOLERANCES["tas"]


def test_unknown_variable_falls_back_to_default():
    assert tolerance_for("not_a_var") is DEFAULT_TOLERANCE


def test_pr_has_tightest_atol():
    # pr is stored in kg m-2 s-1, where values are tiny (~1e-5), so it needs a
    # much tighter atol floor than the temperature baseline while keeping the
    # uniform nonzero rtol slope.
    assert TOLERANCES["pr"].atol < TOLERANCES["tas"].atol
    assert TOLERANCES["pr"].rtol == 1e-5


def test_all_seven_variables_present():
    assert set(TOLERANCES) == {"tas", "tasmax", "tasmin", "pr", "rsds", "hurs", "dtr"}


def test_tasmin_atol_covers_its_derived_inputs():
    # tasmin is derived as tasmax - dtr (saidownscale.downscaling_utils.derive_tasmin), so its
    # snapshot drift is the difference of the two inputs' drifts and can reach
    # atol(tasmax) + atol(dtr). Its atol must be at least that sum, or shared
    # temperature noise that tasmax and dtr each absorb would trip tasmin as a false
    # positive.
    assert TOLERANCES["tasmin"].atol >= TOLERANCES["tasmax"].atol + TOLERANCES["dtr"].atol


def test_tolerance_is_frozen():
    t = Tolerance(rtol=1e-5, atol=1e-3)
    with __import__("pytest").raises(Exception):
        t.rtol = 2.0
