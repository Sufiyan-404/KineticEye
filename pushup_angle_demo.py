"""
pushup_angle_demo.py

A small standalone demo for the Sports Video Analytics & Action Counter project.

It SIMULATES the body landmarks we will eventually receive from MediaPipe,
then uses calculate_angle() from angle_calculator.py to measure:
    - the elbow angle  (shoulder -> elbow -> wrist)
    - the hip angle    (shoulder -> hip   -> ankle)

The coordinates below are hand-made and only meant to show how the geometry
works. They are NOT medically accurate.

Run with:
    python pushup_angle_demo.py
(angle_calculator.py must be in the same folder.)
"""

from angle_calculator import calculate_angle


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


def main():
    for pose_name, landmarks in POSES.items():
        print_pose_report(pose_name, landmarks)


if __name__ == "__main__":
    main()
