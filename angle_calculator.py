"""
angle_calculator.py

Reusable helper for the Sports Video Analytics & Action Counter project.

Main function:
    calculate_angle(a, b, c) -> angle at point b, in degrees (0 to 180).

Example: for a bicep curl, a = shoulder, b = elbow, c = wrist.
The returned value is the elbow angle.
"""

import numpy as np

# Vectors shorter than this are treated as "zero length".
# We use a tiny number instead of exactly 0 because computer arithmetic on
# decimals is never perfectly exact (e.g. 0.1 + 0.2 != 0.3 exactly).
EPSILON = 1e-9


def calculate_angle(a, b, c):
    """
    Calculate the angle at point b formed by the segments b->a and b->c.

    Parameters
    ----------
    a, b, c : sequence of two numbers
        2D points such as (x, y), [x, y], or a NumPy array of length 2.
        b is the vertex (the "corner") where the angle is measured.

    Returns
    -------
    float or None
        Angle in degrees, between 0 and 180.
        Returns None if the angle is undefined because a == b or c == b
        (one of the two vectors has zero length).
    """
    # Step 1: convert inputs to NumPy float arrays so we can do vector maths.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    # Step 2: build the two vectors that start at the vertex b.
    # "Arrow from b to a" = a - b   (head minus tail)
    ba = a - b
    bc = c - b

    # Step 3: compute the length (magnitude) of each vector.
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # Step 4: guard against zero-length vectors.
    # The formula divides by (norm_ba * norm_bc); dividing by zero is not
    # allowed, and an angle with no direction is meaningless anyway.
    if norm_ba < EPSILON or norm_bc < EPSILON:
        return None

    # Step 5: dot product of the two vectors.
    # dot(ba, bc) = |ba| * |bc| * cos(theta)
    dot_product = np.dot(ba, bc)

    # Step 6: rearrange the formula above to isolate cos(theta).
    cos_theta = dot_product / (norm_ba * norm_bc)

    # Step 7: clip to the valid range of cosine, [-1, 1].
    # Because of tiny floating-point errors, cos_theta can come out as
    # 1.0000000000000002 or -1.0000000000000002. np.arccos would then
    # return NaN (not a number). Clipping prevents that.
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Step 8: arccos gives the angle in radians; convert to degrees.
    theta_radians = np.arccos(cos_theta)
    theta_degrees = np.degrees(theta_radians)

    return float(theta_degrees)
