"""
test_feature_extractor.py

Tests for extract_features() in feature_extractor.py.

Run either way (put this file next to feature_extractor.py):
    python test_feature_extractor.py
    pytest test_feature_extractor.py      (if you have pytest installed)

The "analysis" dictionaries in these tests come from the REAL
analyze_push_up_form(), so the extractor is tested against the true shape of
that dictionary. No webcam or MediaPipe is involved.
"""

import ast
import contextlib
import copy
import io
import itertools
import json
import sys
from pathlib import Path

import numpy as np

import feature_extractor
from feature_extractor import (
    extract_features,
    FEATURE_COLUMNS,
    METADATA_COLUMNS,
    MODEL_INPUT_COLUMNS,
    RULE_OUTPUT_COLUMNS,
)
from form_analyzer import (
    analyze_push_up_form,
    MSG_ELBOW_MISSING,
    MSG_HIP_MISSING,
)

NAN = float("nan")
INF = float("inf")

# Values that mean "this angle is unavailable".
UNAVAILABLE_VALUES = [None, NAN, INF, -INF]

ANGLE_NAMES = ["elbow_angle", "hip_angle", "shoulder_angle"]

# A normal, good top position.
NORMAL_ANGLES = {"elbow_angle": 175.0, "hip_angle": 178.0,
                 "shoulder_angle": 75.0}

# The exact columns we promised, written out here on purpose (not imported)
# so the test fails if the module's column list ever changes by accident.
EXPECTED_COLUMNS = [
    "frame_number",
    "elbow_angle",
    "hip_angle",
    "shoulder_angle",
    "elbow_angle_available",
    "hip_angle_available",
    "shoulder_angle_available",
    "all_angles_available",
    "rule_movement_phase",
    "rule_form_status",
    "rule_feedback",
]


# ---- Helpers (their names do not start with "test_") --------------------

def analysis_for(**angles):
    """The real analyzer's result for the normal angles, with changes."""
    return analyze_push_up_form(**{**NORMAL_ANGLES, **angles})


def extract(analysis=None, frame_number=None, **angles):
    """
    Call extract_features() with normal angles, changed by `angles`.
    If no analysis is given, a normal good-form analysis is used.
    """
    values = {**NORMAL_ANGLES, **angles}
    if analysis is None:
        analysis = analysis_for()
    return extract_features(values["elbow_angle"], values["hip_angle"],
                            values["shoulder_angle"], analysis,
                            frame_number=frame_number)


def expect_error(error_type, function, *args, **kwargs):
    """Check that function(...) raises error_type; return its message."""
    try:
        function(*args, **kwargs)
    except error_type as error:
        return str(error)
    raise AssertionError(
        f"expected {error_type.__name__}, but no error was raised")


# ---- 1. A normal valid row ----------------------------------------------

def test_valid_row_with_all_three_angles():
    row = extract(frame_number=12)
    assert row == {
        "frame_number": 12,
        "elbow_angle": 175.0,
        "hip_angle": 178.0,
        "shoulder_angle": 75.0,
        "elbow_angle_available": True,
        "hip_angle_available": True,
        "shoulder_angle_available": True,
        "all_angles_available": True,
        "rule_movement_phase": "TOP",
        "rule_form_status": "GOOD",
        "rule_feedback": "Good form",
    }


def test_rule_columns_copy_the_analyzers_answer_for_a_form_problem():
    analysis = analysis_for(hip_angle=148.95)       # sagging hips
    row = extract(analysis=analysis, hip_angle=148.95)
    assert row["rule_form_status"] == "NEEDS_CORRECTION"
    assert row["rule_movement_phase"] == "TOP"
    assert row["rule_feedback"] == "Keep your back straighter"
    assert row["all_angles_available"] is True


def test_rule_columns_are_copied_not_recalculated():
    # A made-up analysis that disagrees with the angles: the extractor must
    # copy it unchanged, because it has no rules of its own.
    made_up = {"form_status": "SOME_NEW_STATUS",
               "movement_phase": "BOTTOM",
               "feedback": ["Custom message"]}
    row = extract(analysis=made_up)                 # elbow is 175: not BOTTOM
    assert row["rule_movement_phase"] == "BOTTOM"
    assert row["rule_form_status"] == "SOME_NEW_STATUS"
    assert row["rule_feedback"] == "Custom message"


def test_angle_columns_come_from_the_arguments_not_from_analysis():
    # analysis contains 90.0, the argument says 175.0: no error, and the
    # row uses the argument. (No disagreement check, by design.)
    row = extract(analysis=analysis_for(elbow_angle=90.0))
    assert row["elbow_angle"] == 175.0


# ---- 2. Unavailable angles -----------------------------------------------

def test_unavailable_values_become_none_with_a_false_flag():
    for name in ANGLE_NAMES:
        for bad in UNAVAILABLE_VALUES:
            row = extract(analysis=analysis_for(**{name: bad}), **{name: bad})
            assert row[name] is None, (name, bad)
            assert row[f"{name}_available"] is False, (name, bad)
            assert row["all_angles_available"] is False, (name, bad)
            # The other two angles are still available.
            for other in ANGLE_NAMES:
                if other != name:
                    assert row[other] == NORMAL_ANGLES[other]
                    assert row[f"{other}_available"] is True


def test_a_huge_integer_is_unavailable_instead_of_crashing():
    # 10**400 is a whole number, but too big to become a float.
    row = extract(elbow_angle=10 ** 400)
    assert row["elbow_angle"] is None
    assert row["elbow_angle_available"] is False


def test_availability_flags_for_every_combination():
    for combo in itertools.product([True, False], repeat=3):
        angles = {name: (80.0 if available else None)
                  for name, available in zip(ANGLE_NAMES, combo)}
        row = extract(analysis=analysis_for(**angles), **angles)
        assert (row["elbow_angle_available"],
                row["hip_angle_available"],
                row["shoulder_angle_available"]) == combo
        assert row["all_angles_available"] == all(combo)


def test_only_some_angles_available():
    # The webcam case: wrist and ankle hidden, so elbow and hip are None.
    angles = {"elbow_angle": None, "hip_angle": None,
              "shoulder_angle": 74.64660389844737}
    row = extract(analysis=analysis_for(**angles), **angles)
    assert row["elbow_angle"] is None
    assert row["hip_angle"] is None
    assert row["shoulder_angle"] == 74.64660389844737
    assert row["elbow_angle_available"] is False
    assert row["hip_angle_available"] is False
    assert row["shoulder_angle_available"] is True
    assert row["all_angles_available"] is False
    assert row["rule_movement_phase"] == "UNKNOWN"
    assert row["rule_form_status"] == "INCOMPLETE_DATA"


def test_all_angles_unavailable_still_returns_a_row():
    angles = {name: None for name in ANGLE_NAMES}
    row = extract(analysis=analysis_for(**angles), **angles)
    assert list(row) == EXPECTED_COLUMNS
    assert [row[name] for name in ANGLE_NAMES] == [None, None, None]
    assert row["all_angles_available"] is False


def test_integers_and_numpy_numbers_become_plain_floats():
    row = extract(elbow_angle=90, hip_angle=np.float32(170.5),
                  shoulder_angle=np.float64(60.25))
    assert row["elbow_angle"] == 90.0 and type(row["elbow_angle"]) is float
    assert row["hip_angle"] == 170.5 and type(row["hip_angle"]) is float
    assert row["shoulder_angle"] == 60.25
    assert type(row["shoulder_angle"]) is float


# ---- 3. Invalid angle types ----------------------------------------------

def test_boolean_angles_are_rejected():
    for name in ANGLE_NAMES:
        for bad in (True, False, np.bool_(True)):
            message = expect_error(TypeError, extract, **{name: bad})
            assert name in message, (name, bad)


def test_other_invalid_angle_types_are_rejected():
    for name in ANGLE_NAMES:
        for bad in ("90", b"90", [90.0], (90.0,), {"angle": 90}, 1 + 2j,
                    object()):
            message = expect_error(TypeError, extract, **{name: bad})
            assert name in message, (name, bad)
            assert "None" in message          # tells the user what to use


# ---- 4. The analysis dictionary ------------------------------------------

def test_analysis_must_be_a_dictionary():
    for bad in (None, [], (), "analysis", 42):
        message = expect_error(TypeError, extract_features,
                               175.0, 178.0, 75.0, bad)
        assert "analysis" in message, bad


def test_missing_required_analysis_keys():
    for key in ("form_status", "movement_phase", "feedback"):
        analysis = analysis_for()
        del analysis[key]
        message = expect_error(ValueError, extract_features,
                               175.0, 178.0, 75.0, analysis)
        assert key in message

    # An empty dictionary names all three missing keys.
    message = expect_error(ValueError, extract_features,
                           175.0, 178.0, 75.0, {})
    for key in ("form_status", "movement_phase", "feedback"):
        assert key in message


def test_incorrectly_typed_analysis_values():
    bad_values = [
        ("form_status", 5), ("form_status", None),
        ("movement_phase", None), ("movement_phase", 3.5),
        ("feedback", "Good form"), ("feedback", ("Good form",)),
        ("feedback", None), ("feedback", ["fine", 3]), ("feedback", [None]),
    ]
    for key, bad in bad_values:
        analysis = analysis_for()
        analysis[key] = bad
        message = expect_error(TypeError, extract_features,
                               175.0, 178.0, 75.0, analysis)
        assert key in message, (key, bad)


# ---- 5. Frame number -----------------------------------------------------

def test_missing_frame_number_gives_none_but_keeps_the_column():
    row = extract()
    assert "frame_number" in row
    assert row["frame_number"] is None
    assert extract(frame_number=None)["frame_number"] is None


def test_valid_frame_numbers():
    for number in (0, 1, 27, 10 ** 6):
        row = extract(frame_number=number)
        assert row["frame_number"] == number
        assert type(row["frame_number"]) is int
    # NumPy integers are converted to plain ints.
    row = extract(frame_number=np.int64(5))
    assert row["frame_number"] == 5 and type(row["frame_number"]) is int


def test_negative_frame_number_raises_value_error():
    for bad in (-1, -100, np.int64(-3)):
        message = expect_error(ValueError, extract, frame_number=bad)
        assert "negative" in message


def test_invalid_frame_number_types_raise_type_error():
    for bad in (1.0, 2.5, "5", b"5", [1], True, False):
        message = expect_error(TypeError, extract, frame_number=bad)
        assert "frame_number" in message, bad


# ---- 6. Feedback conversion ----------------------------------------------

def test_feedback_list_becomes_a_semicolon_separated_string():
    # Two real messages from the analyzer (elbow and hip unavailable).
    angles = {"elbow_angle": None, "hip_angle": None}
    row = extract(analysis=analysis_for(**angles), **angles)
    assert row["rule_feedback"] == (
        MSG_ELBOW_MISSING + "; " + MSG_HIP_MISSING)


def test_feedback_with_one_message_empty_list_and_three_messages():
    one = extract()                                  # real analysis: "Good form"
    assert one["rule_feedback"] == "Good form"

    for feedback, expected in ([], ""), (["a", "b", "c"], "a; b; c"):
        analysis = {"form_status": "GOOD", "movement_phase": "TOP",
                    "feedback": feedback}
        row = extract(analysis=analysis)
        assert row["rule_feedback"] == expected
        assert isinstance(row["rule_feedback"], str)


# ---- 7. Output columns and types -----------------------------------------

def test_exact_output_column_names_and_order():
    assert FEATURE_COLUMNS == EXPECTED_COLUMNS
    assert len(FEATURE_COLUMNS) == 11
    assert list(extract()) == EXPECTED_COLUMNS
    assert list(extract(frame_number=3)) == EXPECTED_COLUMNS


def test_column_groups_cover_every_column_exactly_once():
    grouped = METADATA_COLUMNS + MODEL_INPUT_COLUMNS + RULE_OUTPUT_COLUMNS
    assert grouped == EXPECTED_COLUMNS
    assert len(set(grouped)) == len(grouped)


def test_value_types_of_every_column():
    row = extract(frame_number=3)
    assert type(row["frame_number"]) is int
    for name in ANGLE_NAMES:
        assert type(row[name]) is float
        assert type(row[f"{name}_available"]) is bool
    assert type(row["all_angles_available"]) is bool
    for name in RULE_OUTPUT_COLUMNS:
        assert type(row[name]) is str


# ---- 8. Independence and purity ------------------------------------------

def test_inputs_are_not_modified():
    analysis = analysis_for(elbow_angle=None, hip_angle=None)
    before = copy.deepcopy(analysis)
    extract(analysis=analysis, elbow_angle=None, hip_angle=None)
    assert analysis == before


def test_extract_features_prints_nothing():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        extract(frame_number=1)
    assert output.getvalue() == ""


def test_row_can_be_saved_as_json():
    # json.dumps(allow_nan=False) fails on NaN, and on NumPy numbers.
    row = extract(elbow_angle=np.float32(80.5), frame_number=np.int64(2),
                  hip_angle=NAN, analysis=analysis_for(hip_angle=NAN))
    text = json.dumps(row, allow_nan=False)
    assert json.loads(text)["hip_angle"] is None


def test_module_imports_only_the_standard_library():
    # Keeps the extractor independent of cv2, MediaPipe, NumPy and the rest
    # of the project (for example pushup_angle_demo).
    source = Path(feature_extractor.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported, "expected the module to import something"
    assert imported <= set(sys.stdlib_module_names), imported


# ---- Simple runner so `python test_feature_extractor.py` works -----------

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
