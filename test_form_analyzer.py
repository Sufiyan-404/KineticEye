"""
test_form_analyzer.py

Tests and a small demo for form_analyzer.py.

Run (put this file next to form_analyzer.py):
    python test_form_analyzer.py
    pytest test_form_analyzer.py        (optional, if pytest is installed)

The sample angles are simulated numbers, similar to pushup_angle_demo.py.
"""

from form_analyzer import (
    analyze_push_up_form,
    PHASE_TOP, PHASE_MIDDLE, PHASE_BOTTOM, PHASE_UNKNOWN,
    STATUS_GOOD, STATUS_NEEDS_CORRECTION, STATUS_INCOMPLETE_DATA,
    MSG_GOOD_FORM, MSG_BACK_STRAIGHTER, MSG_SHOULDER_POSITION, MSG_GO_LOWER,
    MSG_ELBOW_MISSING, MSG_HIP_MISSING, MSG_SHOULDER_MISSING,
)


# ---- The four required scenarios ----------------------------------------

def test_good_top_position():
    # Arms nearly straight, body straight, shoulder in the allowed range.
    result = analyze_push_up_form(elbow_angle=175, hip_angle=178,
                                  shoulder_angle=75)
    assert result["movement_phase"] == PHASE_TOP
    assert result["form_status"] == STATUS_GOOD
    assert result["feedback"] == [MSG_GOOD_FORM]


def test_good_bottom_position():
    # Elbows strongly bent, body still straight.
    result = analyze_push_up_form(elbow_angle=75, hip_angle=172,
                                  shoulder_angle=35)
    assert result["movement_phase"] == PHASE_BOTTOM
    assert result["form_status"] == STATUS_GOOD
    assert result["feedback"] == [MSG_GOOD_FORM]


def test_incorrect_hip_posture():
    # Same as a good top position, but the hips sag (hip angle 149).
    result = analyze_push_up_form(elbow_angle=175, hip_angle=149,
                                  shoulder_angle=75)
    assert result["form_status"] == STATUS_NEEDS_CORRECTION
    assert MSG_BACK_STRAIGHTER in result["feedback"]
    # The phase is still TOP: bad form does not change WHERE we are.
    assert result["movement_phase"] == PHASE_TOP


def test_missing_angle_data():
    # Everything missing: must not crash.
    result = analyze_push_up_form(None, None, None)
    assert result["movement_phase"] == PHASE_UNKNOWN
    assert result["form_status"] == STATUS_INCOMPLETE_DATA
    assert MSG_ELBOW_MISSING in result["feedback"]
    assert MSG_HIP_MISSING in result["feedback"]
    assert MSG_SHOULDER_MISSING in result["feedback"]

    # Calling with no arguments at all must also be safe.
    assert analyze_push_up_form()["form_status"] == STATUS_INCOMPLETE_DATA

    # Only the hip missing: the other checks still run.
    result = analyze_push_up_form(elbow_angle=175, hip_angle=None,
                                  shoulder_angle=75)
    assert result["movement_phase"] == PHASE_TOP
    assert result["form_status"] == STATUS_INCOMPLETE_DATA
    assert result["feedback"] == [MSG_HIP_MISSING]

    # A real problem is still reported even when other data is missing.
    result = analyze_push_up_form(elbow_angle=None, hip_angle=140,
                                  shoulder_angle=None)
    assert result["form_status"] == STATUS_NEEDS_CORRECTION
    assert MSG_BACK_STRAIGHTER in result["feedback"]

    # NaN ("not a number") counts as missing too.
    result = analyze_push_up_form(float("nan"), 175, 75)
    assert result["movement_phase"] == PHASE_UNKNOWN
    assert result["form_status"] == STATUS_INCOMPLETE_DATA


# ---- Extra tests for the remaining rules --------------------------------

def test_phase_boundaries():
    # 160 and above is TOP, 90 and below is BOTTOM, in between is MIDDLE.
    assert analyze_push_up_form(elbow_angle=160)["movement_phase"] == PHASE_TOP
    assert analyze_push_up_form(elbow_angle=159)["movement_phase"] == PHASE_MIDDLE
    assert analyze_push_up_form(elbow_angle=120)["movement_phase"] == PHASE_MIDDLE
    assert analyze_push_up_form(elbow_angle=91)["movement_phase"] == PHASE_MIDDLE
    assert analyze_push_up_form(elbow_angle=90)["movement_phase"] == PHASE_BOTTOM


def test_shoulder_position_out_of_range():
    too_wide = analyze_push_up_form(elbow_angle=175, hip_angle=178,
                                    shoulder_angle=130)
    assert too_wide["form_status"] == STATUS_NEEDS_CORRECTION
    assert MSG_SHOULDER_POSITION in too_wide["feedback"]

    too_small = analyze_push_up_form(elbow_angle=175, hip_angle=178,
                                     shoulder_angle=5)
    assert MSG_SHOULDER_POSITION in too_small["feedback"]


def test_depth_check_uses_lowest_elbow_angle_of_the_rep():
    # The rep only reached 120 degrees: too shallow.
    shallow = analyze_push_up_form(elbow_angle=175, hip_angle=178,
                                   shoulder_angle=75, lowest_elbow_angle=120)
    assert shallow["form_status"] == STATUS_NEEDS_CORRECTION
    assert MSG_GO_LOWER in shallow["feedback"]

    # The rep reached 85 degrees: deep enough.
    deep = analyze_push_up_form(elbow_angle=175, hip_angle=178,
                                shoulder_angle=75, lowest_elbow_angle=85)
    assert deep["form_status"] == STATUS_GOOD

    # Without lowest_elbow_angle, the depth check is skipped.
    no_depth = analyze_push_up_form(elbow_angle=120, hip_angle=178,
                                    shoulder_angle=60)
    assert MSG_GO_LOWER not in no_depth["feedback"]


def test_output_contains_the_input_angles():
    result = analyze_push_up_form(elbow_angle=95, hip_angle=170,
                                  shoulder_angle=50)
    assert result["elbow_angle"] == 95
    assert result["hip_angle"] == 170
    assert result["shoulder_angle"] == 50


# ---- Test runner + small demo -------------------------------------------

def print_demo():
    """Print a few sample analyses so you can see the dictionary output."""
    samples = [
        ("Good top position",  dict(elbow_angle=175, hip_angle=178, shoulder_angle=75)),
        ("Good bottom position", dict(elbow_angle=75, hip_angle=172, shoulder_angle=35)),
        ("Sagging hips",       dict(elbow_angle=175, hip_angle=149, shoulder_angle=75)),
        ("Missing angle data", dict(elbow_angle=None, hip_angle=None, shoulder_angle=None)),
    ]
    print("\n--- Demo output ---")
    for name, angles in samples:
        print(f"\n{name}")
        print(analyze_push_up_form(**angles))


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
    print_demo()
