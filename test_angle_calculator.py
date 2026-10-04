"""
test_angle_calculator.py

Tests for calculate_angle().

Run either way (put this file next to angle_calculator.py):
    python test_angle_calculator.py
    pytest test_angle_calculator.py     (if you have pytest installed)
"""

import math

from angle_calculator import calculate_angle

# How close counts as "equal" when comparing decimal numbers (in degrees).
TOLERANCE = 1e-6


def point_at(vertex, length, angle_deg):
    """
    Helper: return the point that is `length` away from `vertex`,
    in the direction `angle_deg` measured from the positive x-axis.
    Lets us build test triangles with a known, exact angle.
    """
    rad = math.radians(angle_deg)
    return (vertex[0] + length * math.cos(rad),
            vertex[1] + length * math.sin(rad))


def build_case(angle_deg):
    """Build points a, b, c so that the angle at b is angle_deg."""
    b = (0.0, 0.0)
    a = point_at(b, 1.0, 0)          # points along the +x axis
    c = point_at(b, 1.0, angle_deg)  # rotated by angle_deg from a
    return a, b, c


# ---- The four required angles -------------------------------------------

def test_90_degrees():
    # Simple hand-made case: a on the x-axis, c on the y-axis.
    assert math.isclose(calculate_angle((1, 0), (0, 0), (0, 1)), 90.0,
                        abs_tol=TOLERANCE)


def test_120_degrees():
    a, b, c = build_case(120)
    assert math.isclose(calculate_angle(a, b, c), 120.0, abs_tol=TOLERANCE)


def test_150_degrees():
    a, b, c = build_case(150)
    assert math.isclose(calculate_angle(a, b, c), 150.0, abs_tol=TOLERANCE)


def test_180_degrees():
    # Straight line: a and c on opposite sides of b (a fully straight arm).
    # This is the case where rounding error is most likely to push the
    # cosine just past -1, so it also tests np.clip.
    result = calculate_angle((-1, 0), (0, 0), (1, 0))
    assert result is not None and not math.isnan(result)
    assert math.isclose(result, 180.0, abs_tol=TOLERANCE)


# ---- Extra safety tests -------------------------------------------------

def test_0_degrees():
    # Both arrows point the same way: cos = +1, the other clip boundary.
    result = calculate_angle((1, 0), (0, 0), (2, 0))
    assert math.isclose(result, 0.0, abs_tol=TOLERANCE)


def test_zero_length_vector_returns_none():
    # a is the same point as b, so the vector b->a has no length.
    assert calculate_angle((3, 4), (3, 4), (5, 6)) is None
    # c is the same point as b.
    assert calculate_angle((5, 6), (3, 4), (3, 4)) is None


def test_order_of_a_and_c_does_not_matter():
    a, b, c = build_case(120)
    assert math.isclose(calculate_angle(a, b, c),
                        calculate_angle(c, b, a), abs_tol=TOLERANCE)


def test_works_away_from_the_origin():
    # Same 90 degree shape, shifted to (100, 200): angle must not change.
    assert math.isclose(calculate_angle((101, 200), (100, 200), (100, 201)),
                        90.0, abs_tol=TOLERANCE)


def test_accepts_lists_and_numpy_arrays():
    import numpy as np
    assert math.isclose(
        calculate_angle([1, 0], np.array([0, 0]), (0, 1)), 90.0,
        abs_tol=TOLERANCE)


# ---- Simple runner so `python test_angle_calculator.py` works -----------

if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError:
            failed += 1
            print(f"FAIL  {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
