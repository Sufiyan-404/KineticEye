"""
pushup_angle_demo.py

A small standalone demo for the Sports Video Analytics & Action Counter project.

It SIMULATES the body landmarks we will eventually receive from MediaPipe,
then uses calculate_angle() from angle_calculator.py to measure:
    - the elbow angle  (shoulder -> elbow -> wrist)
    - the hip angle    (shoulder -> hip   -> ankle)
    - the shoulder angle (elbow -> shoulder -> hip)

The coordinates below are hand-made and only meant to show how the geometry
works. They are NOT medically accurate.

Section 4 at the bottom shows how landmarks in Person 1's format
({"x": ..., "y": ..., "visibility": ...}) are converted and sent through
calculate_angle() and then analyze_push_up_form().

analyze_landmarks() does the measuring and form analysis without printing
(video_analysis_demo.py uses it); print_landmark_report() calls it and prints.

Real camera frames are not square, so print_landmark_report() can also be given
the image width and height. It then converts MediaPipe's 0-1 coordinates to
pixels before measuring angles. Without the size, the 0-1 values are used as
they are (the simulated demo does this).

Run with:
    python pushup_angle_demo.py
(angle_calculator.py must be in the same folder.)
"""

import math
import numbers

from angle_calculator import calculate_angle
from form_analyzer import analyze_push_up_form


# ---------------------------------------------------------------------------
# 1. SAMPLE DATA
# ---------------------------------------------------------------------------
# Coordinates are (x, y) between 0 and 1, like MediaPipe's normalised output:
#   x = 0 is the left edge of the image, x = 1 is the right edge
#   y = 0 is the TOP of the image,       y = 1 is the BOTTOM  (y grows downward!)
#
# The camera looks at the person from the side. The person's feet are on the
# left of the image and their head would be on the right, so the body is a
# long, nearly horizontal line from the ankle (left) to the shoulder (right).
#
# The landmark names follow MediaPipe's naming style (left_shoulder, ...).

POSES = {
    # Arms almost straight, body in a straight line.
    "Top Position": {
        "left_shoulder": (0.70, 0.45),
        "left_elbow":    (0.71, 0.565),
        "left_wrist":    (0.72, 0.68),
        "left_hip":      (0.43, 0.55),
        "left_ankle":    (0.15, 0.66),
    },
    # Halfway down: shoulder has dropped, elbow is bent, body still straight.
    "Middle Position": {
        "left_shoulder": (0.70, 0.50),
        "left_elbow":    (0.64, 0.60),
        "left_wrist":    (0.72, 0.68),
        "left_hip":      (0.43, 0.58),
        "left_ankle":    (0.15, 0.66),
    },
    # Chest near the floor: elbow strongly bent, body still straight.
    "Bottom Position": {
        "left_shoulder": (0.70, 0.55),
        "left_elbow":    (0.62, 0.62),
        "left_wrist":    (0.72, 0.68),
        "left_hip":      (0.43, 0.605),
        "left_ankle":    (0.15, 0.66),
    },
    # BONUS pose (bad form): same arms as the top position, but the hips
    # sag toward the floor. Shows how the hip angle reveals alignment errors.
    "Bonus: Top Position with Sagging Hips": {
        "left_shoulder": (0.70, 0.45),
        "left_elbow":    (0.71, 0.565),
        "left_wrist":    (0.72, 0.68),
        "left_hip":      (0.43, 0.64),
        "left_ankle":    (0.15, 0.66),
    },
}


# ---------------------------------------------------------------------------
# 2. FUNCTIONS THAT MEASURE THE BODY
# ---------------------------------------------------------------------------

def get_elbow_angle(landmarks):
    """Angle at the elbow, formed by the shoulder and the wrist."""
    return calculate_angle(
        landmarks["left_shoulder"],   # a: one end of the upper arm
        landmarks["left_elbow"],      # b: the vertex (where we measure)
        landmarks["left_wrist"],      # c: the end of the forearm
    )


def get_hip_angle(landmarks):
    """Angle at the hip, formed by the shoulder and the ankle."""
    return calculate_angle(
        landmarks["left_shoulder"],   # a: the upper body
        landmarks["left_hip"],        # b: the vertex (where we measure)
        landmarks["left_ankle"],      # c: the legs
    )


def get_shoulder_angle(landmarks):
    """Angle at the shoulder, formed by the elbow and the hip."""
    return calculate_angle(
        landmarks["left_elbow"],      # a: the upper arm
        landmarks["left_shoulder"],   # b: the vertex (where we measure)
        landmarks["left_hip"],        # c: the torso
    )


def format_angle(angle):
    """Turn an angle into text. calculate_angle() returns None if undefined."""
    if angle is None:
        return "undefined (a landmark is missing or overlaps another)"
    return f"{angle:.1f}\u00b0"


# ---------------------------------------------------------------------------
# 3. PRINTING
# ---------------------------------------------------------------------------

def print_pose_report(pose_name, landmarks):
    """Measure one pose and print the results."""
    elbow_angle = get_elbow_angle(landmarks)
    hip_angle = get_hip_angle(landmarks)

    print(pose_name)
    print(f"Elbow Angle: {format_angle(elbow_angle)}")
    print(f"Hip Angle: {format_angle(hip_angle)}")
    print()  # blank line between poses


# ---------------------------------------------------------------------------
# 4. USING PERSON 1'S LANDMARK FORMAT
# ---------------------------------------------------------------------------
# Person 1's get_landmarks(results) returns every landmark as a dictionary:
#     {"x": ..., "y": ..., "visibility": ...}
# The example below copies that exact format (a top push-up position).
# The right-side landmarks have low visibility because, in a side view,
# the right side of the body is mostly hidden behind the left side.

LANDMARK_EXAMPLE = {
    "left_shoulder":  {"x": 0.70, "y": 0.45,  "visibility": 0.98},
    "left_elbow":     {"x": 0.71, "y": 0.565, "visibility": 0.97},
    "left_wrist":     {"x": 0.72, "y": 0.68,  "visibility": 0.95},
    "left_hip":       {"x": 0.43, "y": 0.55,  "visibility": 0.96},
    "left_knee":      {"x": 0.29, "y": 0.605, "visibility": 0.94},
    "left_ankle":     {"x": 0.15, "y": 0.66,  "visibility": 0.93},
    "right_shoulder": {"x": 0.69, "y": 0.45,  "visibility": 0.41},
    "right_elbow":    {"x": 0.70, "y": 0.565, "visibility": 0.38},
    "right_wrist":    {"x": 0.71, "y": 0.68,  "visibility": 0.35},
    "right_hip":      {"x": 0.42, "y": 0.55,  "visibility": 0.40},
    "right_knee":     {"x": 0.28, "y": 0.605, "visibility": 0.37},
    "right_ankle":    {"x": 0.14, "y": 0.66,  "visibility": 0.36},
}


def check_image_size(image_width, image_height):
    """
    Check the image size and return (width, height) in pixels.

    Returns None if BOTH are None (no size given, so no scaling).
    Raises ValueError if only one is given, or if either is not above 0.
    Raises TypeError if either is not a whole number (text, decimals, True...).
    """
    if image_width is None and image_height is None:
        return None
    if image_width is None or image_height is None:
        raise ValueError(
            "image_width and image_height must be given together, "
            "but only one of them was provided.")

    for name, value in (("image_width", image_width),
                        ("image_height", image_height)):
        # bool is a kind of int in Python, so we reject True/False explicitly.
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(
                f"{name} must be a whole number of pixels, "
                f"but got {type(value).__name__}: {value!r}.")
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0, but got {value}.")
    return int(image_width), int(image_height)


def landmark_to_point(landmark, image_width=None, image_height=None):
    """
    Convert one Person 1 landmark into the (x, y) point calculate_angle() needs.

        {"x": 0.70, "y": 0.45, "visibility": 0.98}  ->  (0.70, 0.45)

    If the image size is given, x and y are first converted from MediaPipe's
    0-1 values to pixels (x uses the width, y uses the height). For a
    640 x 480 image:

        {"x": 0.50, "y": 0.25, "visibility": 0.98}  ->  (320.0, 120.0)

    Only x and y are used. visibility is never changed or scaled.
    """
    size = check_image_size(image_width, image_height)
    x, y = landmark["x"], landmark["y"]
    if size is None:
        return (x, y)
    width, height = size
    return (x * width, y * height)


# ---------------------------------------------------------------------------
# VISIBILITY CHECK
# ---------------------------------------------------------------------------
# MediaPipe gives every landmark a visibility between 0 and 1. A low value
# means MediaPipe is not sure the body part is really visible, so any angle
# that uses it could be wrong. We only trust an angle if ALL landmarks it
# needs are visible enough.
#
# 0.5 is a provisional engineering value for testing. Tune it with real videos.
VISIBILITY_THRESHOLD = 0.5

# The landmarks each angle needs (the same ones get_*_angle() uses above).
ELBOW_LANDMARKS = ["left_shoulder", "left_elbow", "left_wrist"]
HIP_LANDMARKS = ["left_shoulder", "left_hip", "left_ankle"]
SHOULDER_LANDMARKS = ["left_elbow", "left_shoulder", "left_hip"]


def is_valid_number(value):
    """
    True only for a real, finite number (int or float).

    Text like "high", None, True/False, NaN and infinity all return False.
    We check the TYPE first because math.isfinite("high") would crash.
    """
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:   # an integer too big to turn into a float
        return False


def has_valid_coordinates(landmark):
    """True if the landmark is a dictionary with a usable x and y."""
    return (isinstance(landmark, dict)
            and is_valid_number(landmark.get("x"))
            and is_valid_number(landmark.get("y")))


def find_visibility_problems(landmark_dict, required_names,
                             threshold=VISIBILITY_THRESHOLD):
    """
    Check the required landmarks and return a list of problems (text).

    An empty list means every required landmark is present, has usable x and
    y coordinates, and is visible enough. This function never crashes on bad
    data: anything unusable is reported as a problem instead.
    """
    problems = []
    for name in required_names:
        landmark = landmark_dict.get(name)

        # The landmark itself must exist and be a dictionary.
        if landmark is None:
            problems.append(f"{name} is missing")
            continue
        if not isinstance(landmark, dict):
            problems.append(f"{name} is not a valid landmark dictionary")
            continue

        # The coordinates must be real numbers.
        for key in ("x", "y"):
            if key not in landmark:
                problems.append(f"{name} has no '{key}' coordinate")
            elif not is_valid_number(landmark[key]):
                problems.append(
                    f"{name} has an invalid '{key}' coordinate "
                    f"({landmark[key]!r})")

        # The visibility must be a real number that reaches the threshold.
        if "visibility" not in landmark:
            problems.append(f"{name} has no visibility value")
        elif not is_valid_number(landmark["visibility"]):
            problems.append(
                f"{name} has an invalid visibility value "
                f"({landmark['visibility']!r})")
        elif landmark["visibility"] < threshold:
            problems.append(
                f"{name} visibility {landmark['visibility']:.3f} "
                f"is below {threshold}")
    return problems


def measure_angle_if_reliable(angle_name, measure_function, required_names,
                              landmark_dict, points):
    """
    Measure one angle only if its required landmarks are reliable.

    Returns (angle, note):
        reliable   -> (the angle, None)
        unreliable -> (None, a message explaining why it was not assessed)
    Passing None on is safe: analyze_push_up_form() treats None as "missing".
    """
    problems = find_visibility_problems(landmark_dict, required_names)
    if problems:
        return None, f"{angle_name} angle not assessed: " + "; ".join(problems)
    return measure_function(points), None


def angle_text(angle, note):
    """Text for printing: 'not assessed' if the angle was rejected."""
    if note is not None:
        return "not assessed"
    return format_angle(angle)


def _note_for(notes, angle_name):
    """
    Pick the "not assessed" note for one angle ("Elbow", "Hip", "Shoulder")
    out of the notes list, or None if that angle had no problem.
    measure_angle_if_reliable() starts every note with "<Name> angle not
    assessed", which is what we look for here.
    """
    prefix = f"{angle_name} angle not assessed"
    for note in notes:
        if note.startswith(prefix):
            return note
    return None


def analyze_landmarks(landmark_dict, image_width=None, image_height=None):
    """
    Convert Person 1's landmarks, measure the angles and analyze the form.
    This does exactly what print_landmark_report() does, but PRINTS NOTHING.

    image_width and image_height (in pixels) are optional. Give BOTH to
    convert the landmarks to pixel coordinates before measuring angles (use
    this for real camera or video frames). Leave BOTH out to use the 0-1
    values as they are, like the simulated demo does.

    An empty dictionary (nobody detected) is fine: every angle is then None
    and the analyzer reports INCOMPLETE_DATA.

    Returns a tuple (angles, result, notes):
        angles : dict with the keys "elbow_angle", "hip_angle" and
                 "shoulder_angle". Each value is the angle in degrees that was
                 sent to the form analyzer, or None if the visibility check
                 rejected it.
        result : the dictionary returned by analyze_push_up_form().
        notes  : list of texts explaining why an angle was not assessed
                 (empty when every angle was reliable).
    """
    # Check the image size first, so a wrong size stops everything before
    # anything is calculated.
    check_image_size(image_width, image_height)

    # If we did not receive a dictionary at all, treat it as "no landmarks".
    if not isinstance(landmark_dict, dict):
        landmark_dict = {}

    # Step 1: convert landmarks from {"x","y","visibility"} to (x, y), in
    # pixels if the image size was given. Only landmarks with usable x and y
    # are converted, so one broken landmark cannot crash the report. After
    # this, the dictionary looks like the ones in POSES, so the existing
    # get_elbow_angle() and get_hip_angle() work unchanged.
    # (Visibility is checked below on the ORIGINAL landmark_dict, so it is
    # never scaled.)
    points = {name: landmark_to_point(lm, image_width, image_height)
              for name, lm in landmark_dict.items()
              if has_valid_coordinates(lm)}

    # Step 2: calculate each joint angle (uses calculate_angle()), but only
    # if the landmarks it needs are usable and visible enough. Each angle is
    # checked on its own, so one unreliable angle does not stop the others.
    elbow_angle, elbow_note = measure_angle_if_reliable(
        "Elbow", get_elbow_angle, ELBOW_LANDMARKS, landmark_dict, points)
    hip_angle, hip_note = measure_angle_if_reliable(
        "Hip", get_hip_angle, HIP_LANDMARKS, landmark_dict, points)
    shoulder_angle, shoulder_note = measure_angle_if_reliable(
        "Shoulder", get_shoulder_angle, SHOULDER_LANDMARKS,
        landmark_dict, points)
    notes = [n for n in (elbow_note, hip_note, shoulder_note) if n]

    # Step 3: send the angles to the form analyzer. A rejected angle is None,
    # which form_analyzer already treats as "missing".
    result = analyze_push_up_form(
        elbow_angle=elbow_angle,
        hip_angle=hip_angle,
        shoulder_angle=shoulder_angle,
    )

    # The three angles exactly as they were just sent to the analyzer (None
    # for any angle the visibility check rejected).
    angles = {
        "elbow_angle": elbow_angle,
        "hip_angle": hip_angle,
        "shoulder_angle": shoulder_angle,
    }
    return angles, result, notes


def print_landmark_report(title, landmark_dict,
                          image_width=None, image_height=None):
    """
    Convert Person 1's landmarks, measure the angles, analyze the form, print.

    image_width and image_height (in pixels) are optional. Give BOTH to
    convert the landmarks to pixel coordinates before measuring angles (use
    this for real camera frames). Leave BOTH out to use the 0-1 values as they
    are, like the simulated demo does.

    The measuring and analysis are done by analyze_landmarks(); this function
    only prints the result.

    Returns a tuple (angles, result):
        angles : dict with the keys "elbow_angle", "hip_angle" and
                 "shoulder_angle". Each value is the angle in degrees that was
                 sent to the form analyzer, or None if the visibility check
                 rejected it.
        result : the dictionary returned by analyze_push_up_form().
    Callers that only want the printed report can simply ignore the return
    value. Nothing is measured again to build it.
    """
    angles, result, notes = analyze_landmarks(
        landmark_dict, image_width, image_height)

    print(title)
    elbow_text = angle_text(angles["elbow_angle"], _note_for(notes, "Elbow"))
    hip_text = angle_text(angles["hip_angle"], _note_for(notes, "Hip"))
    shoulder_text = angle_text(angles["shoulder_angle"],
                               _note_for(notes, "Shoulder"))
    print(f"Elbow Angle: {elbow_text}")
    print(f"Hip Angle: {hip_text}")
    print(f"Shoulder Angle: {shoulder_text}")
    print(f"Visibility Check (threshold {VISIBILITY_THRESHOLD}):")
    if notes:
        for note in notes:
            print(f"  - {note}")
    else:
        print("  All required landmarks are reliable.")
    print("Form Analysis:")
    print(f"  Status: {result['form_status']}")
    print(f"  Movement Phase: {result['movement_phase']}")
    print("  Feedback:")
    for message in result["feedback"]:
        print(f"    - {message}")
    print()

    return angles, result


def main():
    for pose_name, landmarks in POSES.items():
        print_pose_report(pose_name, landmarks)

    print_landmark_report("Person 1 Landmark Format Example", LANDMARK_EXAMPLE)


if __name__ == "__main__":
    main()
