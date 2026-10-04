"""
form_analyzer.py

Rule-based push-up form analysis for the Sports Video Analytics & Action
Counter project.

Where this module fits in the pipeline:

    landmark coordinates
            |
            v
    angle_calculator.py     (calculate_angle)
            |
            v
    joint angles
            |
            v
    form_analyzer.py        <-- THIS FILE
            |
            v
    form status + feedback

This module does NOT calculate angles and does NOT see raw coordinates.
It only receives finished joint angles (in degrees) for ONE moment (one
video frame) and applies simple rules to them.

Joint angles expected (all in degrees, measured with calculate_angle):
    elbow_angle    : shoulder -> ELBOW    -> wrist
    hip_angle      : shoulder -> HIP      -> ankle
    shoulder_angle : elbow    -> SHOULDER -> hip
                     (angle between the upper arm and the torso)

IMPORTANT: every threshold below is an INITIAL ENGINEERING VALUE, chosen to
be consistent with our simulated demo data. They are NOT medical or
universal standards. They must be tuned later using real videos.
"""

import math


# ===========================================================================
# 1. THRESHOLDS (initial engineering values - tune later with real videos)
# ===========================================================================

# --- Movement phase (uses the ELBOW angle only) ---
# Elbow angle at or above this value  -> arms nearly straight -> TOP
TOP_ELBOW_MIN_ANGLE = 160
# Elbow angle at or below this value  -> arms strongly bent   -> BOTTOM
BOTTOM_ELBOW_MAX_ANGLE = 90
# Anything between the two values is MIDDLE.

# --- Back / hip alignment ---
# A straight body has a hip angle close to 180 degrees. If the hip angle
# falls below this value, the body is bending at the hip (sagging or piking).
HIP_MIN_ANGLE = 160

# --- Shoulder position (angle between upper arm and torso) ---
# The shoulder angle should stay inside this allowed range.
SHOULDER_MIN_ANGLE = 20
SHOULDER_MAX_ANGLE = 100

# --- Depth (needs the whole rep, see analyze_push_up_form) ---
# During a rep, the elbow angle should get down to at least this value.
DEPTH_ELBOW_MAX_ANGLE = 90


# ===========================================================================
# 2. NAMES AND MESSAGES (kept as constants to avoid typos elsewhere)
# ===========================================================================

# Movement phases
PHASE_TOP = "TOP"
PHASE_MIDDLE = "MIDDLE"
PHASE_BOTTOM = "BOTTOM"
PHASE_UNKNOWN = "UNKNOWN"          # elbow angle missing, phase can't be found

# Form status values
STATUS_GOOD = "GOOD"
STATUS_NEEDS_CORRECTION = "NEEDS_CORRECTION"
STATUS_INCOMPLETE_DATA = "INCOMPLETE_DATA"   # no errors found, but some
                                             # angles were missing, so we
                                             # can't confirm the form is good

# Feedback messages
MSG_GOOD_FORM = "Good form"
MSG_BACK_STRAIGHTER = "Keep your back straighter"
MSG_SHOULDER_POSITION = "Maintain proper shoulder position"
MSG_GO_LOWER = "Go lower"
MSG_ELBOW_MISSING = "Elbow angle missing: cannot determine movement phase"
MSG_HIP_MISSING = "Hip angle missing: cannot check back alignment"
MSG_SHOULDER_MISSING = "Shoulder angle missing: cannot check shoulder position"


# ===========================================================================
# 3. SMALL HELPER FUNCTIONS
# ===========================================================================

def is_missing(angle):
    """
    Return True if an angle value cannot be used.

    Missing means None (e.g. calculate_angle() returned None) or NaN
    ("not a number"). NaN needs special care: every comparison with NaN
    (like NaN >= 160) quietly returns False, which would give wrong answers
    instead of an error.
    """
    return angle is None or not math.isfinite(angle)


def determine_movement_phase(elbow_angle):
    """
    Work out the approximate movement phase from the elbow angle alone.

    Returns PHASE_TOP, PHASE_MIDDLE, PHASE_BOTTOM, or PHASE_UNKNOWN if the
    elbow angle is missing. This only says WHERE we are in the movement. It
    does not say whether the form is good or bad.
    """
    if is_missing(elbow_angle):
        return PHASE_UNKNOWN

    if elbow_angle >= TOP_ELBOW_MIN_ANGLE:
        return PHASE_TOP
    if elbow_angle <= BOTTOM_ELBOW_MAX_ANGLE:
        return PHASE_BOTTOM
    return PHASE_MIDDLE


# ---------------------------------------------------------------------------
# Form checks. Each one returns a feedback message if there is a problem,
# or None if the check passes. The caller makes sure the angle is not missing.
# ---------------------------------------------------------------------------

def check_hip_alignment(hip_angle):
    """Body should be close to a straight line: hip angle near 180 degrees."""
    # Note: a low hip angle can mean sagging hips OR hips piked up in the
    # air. This single number cannot tell them apart, so the message is
    # general.
    if hip_angle < HIP_MIN_ANGLE:
        return MSG_BACK_STRAIGHTER
    return None


def check_shoulder_position(shoulder_angle):
    """Shoulder angle should stay inside the allowed range."""
    if shoulder_angle < SHOULDER_MIN_ANGLE or shoulder_angle > SHOULDER_MAX_ANGLE:
        return MSG_SHOULDER_POSITION
    return None


def check_depth(lowest_elbow_angle):
    """
    Over a whole rep, the elbow should bend at least as far as the target.
    A BIGGER angle means a shallower push-up, so too high -> "Go lower".
    """
    if lowest_elbow_angle > DEPTH_ELBOW_MAX_ANGLE:
        return MSG_GO_LOWER
    return None


# ===========================================================================
# 4. MAIN FUNCTION
# ===========================================================================

def analyze_push_up_form(elbow_angle=None, hip_angle=None,
                         shoulder_angle=None, lowest_elbow_angle=None):
    """
    Analyze push-up form from joint angles (in degrees).

    Parameters
    ----------
    elbow_angle, hip_angle, shoulder_angle : float or None
        Joint angles for the current moment. None (or NaN) means "missing".
        Missing values never crash this function: the matching check is
        skipped and a message explains what could not be checked.
    lowest_elbow_angle : float or None, optional
        The smallest elbow angle reached during the current rep. This only
        makes sense for a whole rep, so a rep counter will supply it later.
        If None, the depth check is skipped (single-frame analysis).

    Returns
    -------
    dict with keys:
        "form_status"    : "GOOD", "NEEDS_CORRECTION" or "INCOMPLETE_DATA"
        "movement_phase" : "TOP", "MIDDLE", "BOTTOM" or "UNKNOWN"
        "feedback"       : list of message strings
        "elbow_angle", "hip_angle", "shoulder_angle" : the inputs, unchanged
    """
    corrections = []     # real form problems found
    missing_notes = []   # explanations of what could not be checked

    # --- Movement phase (needs only the elbow angle) ---
    movement_phase = determine_movement_phase(elbow_angle)
    if movement_phase == PHASE_UNKNOWN:
        missing_notes.append(MSG_ELBOW_MISSING)

    # --- Hip check ---
    if is_missing(hip_angle):
        missing_notes.append(MSG_HIP_MISSING)
    else:
        message = check_hip_alignment(hip_angle)
        if message:
            corrections.append(message)

    # --- Shoulder check ---
    if is_missing(shoulder_angle):
        missing_notes.append(MSG_SHOULDER_MISSING)
    else:
        message = check_shoulder_position(shoulder_angle)
        if message:
            corrections.append(message)

    # --- Depth check (optional, only if a whole-rep value was supplied) ---
    if not is_missing(lowest_elbow_angle):
        message = check_depth(lowest_elbow_angle)
        if message:
            corrections.append(message)

    # --- Decide the overall status ---
    # Real problems come first. If there are none but some data was missing,
    # we cannot honestly say the form is good.
    if corrections:
        form_status = STATUS_NEEDS_CORRECTION
        feedback = corrections + missing_notes
    elif missing_notes:
        form_status = STATUS_INCOMPLETE_DATA
        feedback = missing_notes
    else:
        form_status = STATUS_GOOD
        feedback = [MSG_GOOD_FORM]

    return {
        "form_status": form_status,
        "movement_phase": movement_phase,
        "feedback": feedback,
        "elbow_angle": elbow_angle,
        "hip_angle": hip_angle,
        "shoulder_angle": shoulder_angle,
    }
