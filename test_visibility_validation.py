"""
test_visibility_validation.py

Tests for the landmark validation in pushup_angle_demo.py:
    - visibility threshold (0.5)
    - missing landmarks, coordinates and visibility
    - invalid values (text, None, True/False, NaN, infinity)
    - each angle being validated independently

Run either way (put this file next to pushup_angle_demo.py):
    python test_visibility_validation.py
    pytest test_visibility_validation.py      (if you have pytest installed)
"""

import contextlib
import copy
import io
import math

import pushup_angle_demo
from pushup_angle_demo import (
    LANDMARK_EXAMPLE,
    VISIBILITY_THRESHOLD,
    ELBOW_LANDMARKS,
    HIP_LANDMARKS,
    SHOULDER_LANDMARKS,
    find_visibility_problems,
    measure_angle_if_reliable,
    get_elbow_angle,
    print_landmark_report,
    main,
)

NAN = float("nan")
INF = float("inf")


# ---- Helpers (their names do not start with "test_") --------------------

def make_landmarks(**changes):
    """
    Copy of the good example landmarks, with changes applied.

    make_landmarks(left_wrist={"visibility": 0.3}) updates just that field.
    make_landmarks(left_wrist=None) replaces the whole entry.
    """
    landmarks = copy.deepcopy(LANDMARK_EXAMPLE)
    for name, change in changes.items():
        if isinstance(change, dict) and isinstance(landmarks.get(name), dict):
            landmarks[name].update(change)
        else:
            landmarks[name] = change
    return landmarks


def remove_field(landmarks, name, field):
    """Delete one field (like "x") from one landmark."""
    del landmarks[name][field]
    return landmarks


def run_report(landmarks):
    """
    Run print_landmark_report() and return (angles_sent, printed_text).

    angles_sent holds exactly what analyze_push_up_form() received, so the
    tests can check that None really reaches the form analyzer.
    """
    sent = {}
    real_analyzer = pushup_angle_demo.analyze_push_up_form

    def recording_analyzer(**kwargs):
        sent.update(kwargs)
        return real_analyzer(**kwargs)

    pushup_angle_demo.analyze_push_up_form = recording_analyzer
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            print_landmark_report("test", landmarks)
    finally:
        pushup_angle_demo.analyze_push_up_form = real_analyzer
    return sent, output.getvalue()


def is_number(value):
    return isinstance(value, float) and math.isfinite(value)


# ---- 1. Valid landmarks --------------------------------------------------

def test_valid_landmarks_give_all_three_angles():
    sent, text = run_report(make_landmarks())
    assert is_number(sent["elbow_angle"])
    assert is_number(sent["hip_angle"])
    assert is_number(sent["shoulder_angle"])
    assert "All required landmarks are reliable." in text
    assert "Status: GOOD" in text


def test_valid_landmarks_have_no_problems():
    for names in (ELBOW_LANDMARKS, HIP_LANDMARKS, SHOULDER_LANDMARKS):
        assert find_visibility_problems(LANDMARK_EXAMPLE, names) == []


def test_numpy_numbers_are_accepted():
    import numpy as np
    landmarks = make_landmarks(
        left_wrist={"x": np.float32(0.72), "visibility": np.float32(0.9)})
    assert find_visibility_problems(landmarks, ELBOW_LANDMARKS) == []


# ---- 2. The 0.5 threshold ------------------------------------------------

def test_threshold_default_is_half():
    assert VISIBILITY_THRESHOLD == 0.5


def test_visibility_exactly_at_threshold_is_accepted():
    landmarks = make_landmarks(left_wrist={"visibility": 0.5})
    assert find_visibility_problems(landmarks, ELBOW_LANDMARKS) == []


def test_visibility_just_below_threshold_is_rejected():
    landmarks = make_landmarks(left_wrist={"visibility": 0.499})
    problems = find_visibility_problems(landmarks, ELBOW_LANDMARKS)
    assert len(problems) == 1
    assert "left_wrist" in problems[0]
    assert "0.499" in problems[0]


def test_threshold_can_be_changed():
    landmarks = make_landmarks(left_wrist={"visibility": 0.7})
    assert find_visibility_problems(landmarks, ELBOW_LANDMARKS) == []
    assert find_visibility_problems(landmarks, ELBOW_LANDMARKS,
                                    threshold=0.8) != []


# ---- 3. Missing data -----------------------------------------------------

def test_missing_landmark_entry():
    landmarks = make_landmarks()
    del landmarks["left_wrist"]
    problems = find_visibility_problems(landmarks, ELBOW_LANDMARKS)
    assert problems == ["left_wrist is missing"]


def test_landmark_entry_set_to_none_counts_as_missing():
    landmarks = make_landmarks(left_wrist=None)
    assert find_visibility_problems(landmarks, ELBOW_LANDMARKS) == [
        "left_wrist is missing"]


def test_missing_visibility_field():
    landmarks = remove_field(make_landmarks(), "left_wrist", "visibility")
    problems = find_visibility_problems(landmarks, ELBOW_LANDMARKS)
    assert len(problems) == 1
    assert "left_wrist" in problems[0] and "no visibility" in problems[0]


def test_empty_dictionary_does_not_crash():
    sent, text = run_report({})
    assert sent == {"elbow_angle": None, "hip_angle": None,
                    "shoulder_angle": None}
    assert "Status: INCOMPLETE_DATA" in text


def test_input_that_is_not_a_dictionary_does_not_crash():
    for bad_input in (None, [], "landmarks", 42):
        sent, text = run_report(bad_input)
        assert sent == {"elbow_angle": None, "hip_angle": None,
                        "shoulder_angle": None}


# ---- 4. Invalid values ---------------------------------------------------

def test_non_numeric_visibility_does_not_crash():
    for bad in ("high", [0.9], {"v": 1}, None, True, False):
        landmarks = make_landmarks(left_wrist={"visibility": bad})
        problems = find_visibility_problems(landmarks, ELBOW_LANDMARKS)
        assert len(problems) == 1, bad
        assert "left_wrist" in problems[0]


def test_non_finite_visibility_is_rejected():
    for bad in (NAN, INF, -INF):
        landmarks = make_landmarks(left_wrist={"visibility": bad})
        problems = find_visibility_problems(landmarks, ELBOW_LANDMARKS)
        assert len(problems) == 1, bad
        assert "invalid visibility" in problems[0]


def test_huge_integer_does_not_crash():
    landmarks = make_landmarks(left_wrist={"visibility": 10 ** 400})
    assert find_visibility_problems(landmarks, ELBOW_LANDMARKS) != []


def test_landmark_that_is_not_a_dictionary():
    for bad in ("oops", [0.7, 0.45], 3.14):
        landmarks = make_landmarks(left_wrist=bad)
        problems = find_visibility_problems(landmarks, ELBOW_LANDMARKS)
        assert problems == ["left_wrist is not a valid landmark dictionary"]
        sent, _ = run_report(landmarks)       # must not crash either
        assert sent["elbow_angle"] is None


# ---- 5. Invalid coordinates ----------------------------------------------

def test_missing_x_or_y_does_not_crash():
    for field in ("x", "y"):
        landmarks = remove_field(make_landmarks(), "left_wrist", field)
        problems = find_visibility_problems(landmarks, ELBOW_LANDMARKS)
        assert len(problems) == 1
        assert f"no '{field}' coordinate" in problems[0]
        sent, text = run_report(landmarks)
        assert sent["elbow_angle"] is None
        assert "Elbow angle not assessed" in text


def test_invalid_coordinate_values_are_rejected():
    for field in ("x", "y"):
        for bad in ("abc", None, True, NAN, INF, -INF):
            landmarks = make_landmarks(left_wrist={field: bad})
            problems = find_visibility_problems(landmarks, ELBOW_LANDMARKS)
            assert len(problems) == 1, (field, bad)
            assert f"invalid '{field}' coordinate" in problems[0]
            sent, _ = run_report(landmarks)
            assert sent["elbow_angle"] is None


def test_nan_coordinate_never_reaches_the_angle_calculation():
    # Without validation a NaN coordinate silently produces a NaN angle.
    landmarks = make_landmarks(left_hip={"x": NAN})
    sent, _ = run_report(landmarks)
    assert sent["hip_angle"] is None
    assert sent["shoulder_angle"] is None


def test_broken_unused_landmark_does_not_stop_the_report():
    # left_knee, right_wrist and right_hip are not used by any angle.
    landmarks = make_landmarks()
    del landmarks["left_knee"]["x"]
    landmarks["right_wrist"] = "garbage"
    landmarks["right_hip"]["y"] = "abc"
    sent, text = run_report(landmarks)
    assert is_number(sent["elbow_angle"])
    assert is_number(sent["hip_angle"])
    assert is_number(sent["shoulder_angle"])
    assert "Status: GOOD" in text


# ---- 6. Independent angle validation -------------------------------------

def angles_assessed(landmarks):
    """Return which of the three angles were passed on as real numbers."""
    sent, _ = run_report(landmarks)
    return (is_number(sent["elbow_angle"]),
            is_number(sent["hip_angle"]),
            is_number(sent["shoulder_angle"]))


def test_low_wrist_only_affects_the_elbow_angle():
    landmarks = make_landmarks(left_wrist={"visibility": 0.491})
    assert angles_assessed(landmarks) == (False, True, True)


def test_low_ankle_only_affects_the_hip_angle():
    landmarks = make_landmarks(left_ankle={"visibility": 0.137})
    assert angles_assessed(landmarks) == (True, False, True)


def test_low_hip_affects_hip_and_shoulder_but_not_elbow():
    landmarks = make_landmarks(left_hip={"visibility": 0.2})
    assert angles_assessed(landmarks) == (True, False, False)


def test_low_elbow_affects_elbow_and_shoulder_but_not_hip():
    landmarks = make_landmarks(left_elbow={"visibility": 0.2})
    assert angles_assessed(landmarks) == (False, True, False)


def test_low_shoulder_affects_all_three_angles():
    landmarks = make_landmarks(left_shoulder={"visibility": 0.2})
    assert angles_assessed(landmarks) == (False, False, False)


def test_low_right_side_visibility_is_ignored():
    landmarks = make_landmarks(right_shoulder={"visibility": 0.01},
                               right_wrist={"visibility": 0.01})
    assert angles_assessed(landmarks) == (True, True, True)


def test_rejected_angle_gives_incomplete_data_not_good():
    landmarks = make_landmarks(left_ankle={"visibility": 0.137})
    sent, text = run_report(landmarks)
    assert sent["hip_angle"] is None
    assert "Status: INCOMPLETE_DATA" in text
    assert "Status: GOOD" not in text


def test_real_world_example_from_the_webcam():
    landmarks = make_landmarks(
        left_shoulder={"visibility": 0.968}, left_elbow={"visibility": 0.888},
        left_wrist={"visibility": 0.491}, left_hip={"visibility": 0.968},
        left_ankle={"visibility": 0.137})
    assert angles_assessed(landmarks) == (False, False, True)


# ---- 7. Diagnostic messages ----------------------------------------------

def test_message_names_the_angle_and_the_landmark_and_why():
    landmarks = make_landmarks(left_wrist={"visibility": 0.491})
    _, text = run_report(landmarks)
    assert ("Elbow angle not assessed: "
            "left_wrist visibility 0.491 is below 0.5") in text


def test_message_lists_every_problem_for_one_angle():
    landmarks = make_landmarks(left_wrist={"visibility": 0.1},
                               left_ankle=None)
    problems = find_visibility_problems(landmarks, ELBOW_LANDMARKS + ["left_ankle"])
    assert len(problems) == 2


def test_measure_angle_if_reliable_returns_angle_or_note():
    points = {name: (lm["x"], lm["y"]) for name, lm in LANDMARK_EXAMPLE.items()}
    angle, note = measure_angle_if_reliable(
        "Elbow", get_elbow_angle, ELBOW_LANDMARKS, LANDMARK_EXAMPLE, points)
    assert is_number(angle) and note is None

    bad = make_landmarks(left_wrist={"visibility": 0.1})
    angle, note = measure_angle_if_reliable(
        "Elbow", get_elbow_angle, ELBOW_LANDMARKS, bad, points)
    assert angle is None
    assert note.startswith("Elbow angle not assessed: ")


# ---- 8. The simulated demo still works -----------------------------------

def test_simulated_demo_still_runs():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main()
    text = output.getvalue()
    for pose_name in pushup_angle_demo.POSES:
        assert pose_name in text
    assert "Person 1 Landmark Format Example" in text
    assert "Status: GOOD" in text


# ---- Simple runner so `python test_visibility_validation.py` works -------

if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as error:      # AssertionError or an unexpected crash
            failed += 1
            print(f"FAIL  {name}  ({type(error).__name__}: {error})")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
