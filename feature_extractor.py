"""
feature_extractor.py

Turns the results for ONE video frame into ONE flat "row" of features.
This is the step between the form analysis and, later, a Pandas table.

Where this module fits in the pipeline:

    landmark coordinates
            |
            v
    joint angles  (None when an angle could not be trusted)
            |
            v
    analyze_push_up_form()      -> "analysis" dictionary
            |
            v
    feature_extractor.py        <-- THIS FILE: extract_features(...)
            |
            v
    one row (dictionary) per frame

This module does NOT use the webcam, MediaPipe or OpenCV, does NOT print, and
does NOT calculate angles or apply any form rules. It only checks its inputs
and arranges them into a fixed set of columns. It imports nothing from the
rest of the project, only two modules from Python's standard library.

The 11 output columns (always in this order):

    frame_number               int or None   caller's frame counter
    elbow_angle                float or None degrees; None = unavailable
    hip_angle                  float or None
    shoulder_angle             float or None
    elbow_angle_available      bool          True if elbow_angle is a number
    hip_angle_available        bool
    shoulder_angle_available   bool
    all_angles_available       bool          True if all three are available
    rule_movement_phase        str           copied from the analysis
    rule_form_status           str           copied from the analysis
    rule_feedback              str           analysis feedback joined by "; "

The three "rule_" columns are copied from the analyzer's output. They tell you
what the hand-written rules concluded; they are not independent measurements.

IMPORTANT for the caller: pass the SAME angles you gave to
analyze_push_up_form(), after visibility validation (None for any angle that
was rejected). This module does not compare the angles with the numbers inside
`analysis`, so it cannot detect a frame mix-up for you.
"""

import math
import numbers


# ===========================================================================
# 1. COLUMN NAMES (one place, so other code never has to guess them)
# ===========================================================================

METADATA_COLUMNS = ["frame_number"]

MODEL_INPUT_COLUMNS = [
    "elbow_angle",
    "hip_angle",
    "shoulder_angle",
    "elbow_angle_available",
    "hip_angle_available",
    "shoulder_angle_available",
    "all_angles_available",
]

RULE_OUTPUT_COLUMNS = [
    "rule_movement_phase",
    "rule_form_status",
    "rule_feedback",
]

# The complete output, in order.
FEATURE_COLUMNS = METADATA_COLUMNS + MODEL_INPUT_COLUMNS + RULE_OUTPUT_COLUMNS

# Keys that must exist in the dictionary returned by analyze_push_up_form().
REQUIRED_ANALYSIS_KEYS = ("form_status", "movement_phase", "feedback")

# Placed between feedback messages when the list is turned into one string.
FEEDBACK_SEPARATOR = "; "


# ===========================================================================
# 2. PRIVATE HELPERS (each checks one input and returns a clean value)
# ===========================================================================

def _clean_angle(value, name):
    """
    Return the angle as a finite float, or None if it is unavailable.

    None, NaN and infinity all mean "unavailable" and become None.
    Anything that is not a number (text, a list, True/False, ...) is a
    mistake by the caller, so it raises TypeError.
    """
    if value is None:
        return None

    # bool is a kind of int in Python, so True would pass the number test.
    # We reject it explicitly: True is not an angle.
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{name} must be a number or None (use None when the angle is "
            f"unavailable), but got {type(value).__name__}: {value!r}."
        )

    try:
        number = float(value)       # also turns NumPy numbers into floats
    except OverflowError:           # an integer too big to become a float
        return None

    if not math.isfinite(number):   # NaN, +infinity, -infinity
        return None
    return number


def _check_analysis(analysis):
    """Check that `analysis` looks like analyze_push_up_form()'s result."""
    if not isinstance(analysis, dict):
        raise TypeError(
            "analysis must be the dictionary returned by "
            f"analyze_push_up_form(), but got {type(analysis).__name__}."
        )

    missing = [key for key in REQUIRED_ANALYSIS_KEYS if key not in analysis]
    if missing:
        raise ValueError(
            f"analysis is missing required key(s): {', '.join(missing)}. "
            "Pass the dictionary returned by analyze_push_up_form()."
        )

    for key in ("form_status", "movement_phase"):
        if not isinstance(analysis[key], str):
            raise TypeError(
                f"analysis['{key}'] must be a string, "
                f"but got {type(analysis[key]).__name__}."
            )

    feedback = analysis["feedback"]
    if not isinstance(feedback, list):
        raise TypeError(
            "analysis['feedback'] must be a list of strings, "
            f"but got {type(feedback).__name__}."
        )
    for message in feedback:
        if not isinstance(message, str):
            raise TypeError(
                "analysis['feedback'] must be a list of strings, but it "
                f"contains a {type(message).__name__}: {message!r}."
            )


def _clean_frame_number(frame_number):
    """Return frame_number as an int (or None if it was not given)."""
    if frame_number is None:
        return None

    if isinstance(frame_number, bool) or not isinstance(frame_number,
                                                        numbers.Integral):
        raise TypeError(
            "frame_number must be a whole number or None, "
            f"but got {type(frame_number).__name__}: {frame_number!r}."
        )
    if frame_number < 0:
        raise ValueError(
            f"frame_number must not be negative, but got {frame_number}."
        )
    return int(frame_number)


# ===========================================================================
# 3. MAIN FUNCTION
# ===========================================================================

def extract_features(elbow_angle, hip_angle, shoulder_angle, analysis,
                     frame_number=None):
    """
    Build one feature row (a dictionary) for one video frame.

    Parameters
    ----------
    elbow_angle, hip_angle, shoulder_angle : number or None
        The calculated joint angles in degrees, after visibility validation.
        Use None for an angle that is unavailable. NaN and infinity are also
        treated as unavailable. True/False and non-numbers raise TypeError.
    analysis : dict
        The dictionary returned by analyze_push_up_form(). It must contain
        "form_status", "movement_phase" (strings) and "feedback" (a list of
        strings). Any other keys are ignored.
    frame_number : int or None, optional
        Your own frame counter (a whole number, 0 or more). If omitted the
        row still has a "frame_number" column, with the value None.

    Returns
    -------
    dict
        Exactly the 11 columns in FEATURE_COLUMNS, in that order.

    Raises
    ------
    TypeError
        An input has the wrong type (for example text instead of a number).
    ValueError
        `analysis` lacks a required key, or frame_number is negative.
    """
    # Step 1: check and clean every input. If anything is wrong we stop here,
    # before building any row.
    elbow = _clean_angle(elbow_angle, "elbow_angle")
    hip = _clean_angle(hip_angle, "hip_angle")
    shoulder = _clean_angle(shoulder_angle, "shoulder_angle")
    _check_analysis(analysis)
    frame_number = _clean_frame_number(frame_number)

    # Step 2: an angle is "available" exactly when it is a number after
    # cleaning (cleaned angles are always finite floats or None).
    elbow_ok = elbow is not None
    hip_ok = hip is not None
    shoulder_ok = shoulder is not None

    # Step 3: build the row. The rule_* values are COPIED from the analyzer's
    # output, never recalculated. The only change is joining the feedback
    # list into one string.
    return {
        "frame_number": frame_number,
        "elbow_angle": elbow,
        "hip_angle": hip,
        "shoulder_angle": shoulder,
        "elbow_angle_available": elbow_ok,
        "hip_angle_available": hip_ok,
        "shoulder_angle_available": shoulder_ok,
        "all_angles_available": elbow_ok and hip_ok and shoulder_ok,
        "rule_movement_phase": analysis["movement_phase"],
        "rule_form_status": analysis["form_status"],
        "rule_feedback": FEEDBACK_SEPARATOR.join(analysis["feedback"]),
    }
