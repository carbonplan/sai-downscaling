from srm.snapshot.tolerances import DEFAULT_TOLERANCE, TOLERANCES, Tolerance, tolerance_for


def test_known_variable_returns_its_tolerance():
    assert tolerance_for("tas") == TOLERANCES["tas"]


def test_unknown_variable_falls_back_to_default():
    assert tolerance_for("not_a_var") is DEFAULT_TOLERANCE


def test_pr_is_atol_dominant():
    # Precipitation near zero must not rely on relative tolerance.
    assert TOLERANCES["pr"].rtol == 0.0
    assert TOLERANCES["pr"].atol > 0.0


def test_all_seven_variables_present():
    assert set(TOLERANCES) == {"tas", "tasmax", "tasmin", "pr", "rsds", "hurs", "dtr"}


def test_tolerance_is_frozen():
    t = Tolerance(rtol=1e-5, atol=1e-3)
    with __import__("pytest").raises(Exception):
        t.rtol = 2.0
