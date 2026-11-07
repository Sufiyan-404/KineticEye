"""
test_landmark_coordinate_scaling.py

Tests for converting MediaPipe's normalised (0-1) landmark coordinates to
pixels before angles are calculated (pushup_angle_demo.py), and for passing
the real frame size from single_frame_demo.py.

Why this matters: x is a fraction of the image WIDTH and y is a fraction of
the image HEIGHT. On a non-square image one step in x is not the same length
as one step in y, so angles measured on the raw 0-1 values are distorted.

Run either way (put this file next to pushup_angle_demo.py):
    python test_landmark_coordinate_scaling.py
    pytest test_landmark_coordinate_scaling.py     (if you have pytest installed)
"""

import contextlib
import copy
import enum
import importlib
import inspect
import io
import math
import sys
import types

import numpy as np

import pushup_angle_demo
from angle_calculator import calculate_angle
from pushup_angle_demo import (
    LANDMARK_EXAMPLE,
    check_image_size,
    landmark_to_point,
    main,
    print_landmark_report,
)

# What `python pushup_angle_demo.py` printed BEFORE the scaling change.
# The simulated demo gives no image size, so it must still print exactly this.
GOLDEN_DEMO_LINES = [
    "Top Position",
    "Elbow Angle: 180.0\u00b0",
    "Hip Angle: 178.9\u00b0",
    "",
    "Middle Position",
    "Elbow Angle: 104.0\u00b0",
    "Hip Angle: 179.4\u00b0",
    "",
    "Bottom Position",
    "Elbow Angle: 72.1\u00b0",
    "Hip Angle: 179.6\u00b0",
    "",
    "Bonus: Top Position with Sagging Hips",
    "Elbow Angle: 180.0\u00b0",
    "Hip Angle: 149.0\u00b0",
    "",
    "Person 1 Landmark Format Example",
    "Elbow Angle: 180.0\u00b0",
    "Hip Angle: 178.9\u00b0",
    "Shoulder Angle: 74.6\u00b0",
    "Visibility Check (threshold 0.5):",
    "  All required landmarks are reliable.",
    "Form Analysis:",
    "  Status: GOOD",
    "  Movement Phase: TOP",
    "  Feedback:",
    "    - Good form",
    "",
]


# ---- Helpers (their names do not start with "test_") --------------------

def make_landmarks(**changes):
    """
    Copy of the good example landmarks with changes applied.

    make_landmarks(left_wrist={"visibility": 0.3}) updates just that field.
    """
    landmarks = copy.deepcopy(LANDMARK_EXAMPLE)
    for name, change in changes.items():
        landmarks[name].update(change)
    return landmarks


@contextlib.contextmanager
def recording():
    """
    While active: record what analyze_push_up_form() receives and capture
    everything that is printed. Yields (sent, output).
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
            yield sent, output
    finally:
        pushup_angle_demo.analyze_push_up_form = real_analyzer


def run_report(landmarks, **size):
    """Run print_landmark_report(); return (angles_sent, printed_text)."""
    with recording() as (sent, output):
        print_landmark_report("test", landmarks, **size)
    return sent, output.getvalue()


def expect_error(error_type, function, *args, **kwargs):
    """Check that function(...) raises error_type; return its message."""
    try:
        function(*args, **kwargs)
    except error_type as error:
        return str(error)
    raise AssertionError(
        f"expected {error_type.__name__}, but no error was raised")


def same(a, b):
    """Compare two angles (floats) or two Nones."""
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, abs_tol=1e-9)


def angles_assessed(landmarks, **size):
    """Which of (elbow, hip, shoulder) were passed on as real numbers."""
    sent, _ = run_report(landmarks, **size)
    return tuple(sent[name] is not None
                 for name in ("elbow_angle", "hip_angle", "shoulder_angle"))


# ---- 1. Converting normalised coordinates to pixels ----------------------

def test_landmark_to_point_converts_normalised_values_to_pixels():
    landmark = {"x": 0.5, "y": 0.25, "visibility": 0.9}
    assert landmark_to_point(landmark, 640, 480) == (320.0, 120.0)

    landmark = {"x": 0.70, "y": 0.45, "visibility": 0.98}
    for width, height in [(640, 480), (1280, 720), (1920, 1080)]:
        x, y = landmark_to_point(landmark, width, height)
        assert math.isclose(x, 0.70 * width)
        assert math.isclose(y, 0.45 * height)


def test_x_uses_the_width_and_y_uses_the_height():
    landmark = {"x": 0.5, "y": 0.25, "visibility": 0.9}
    # 200 wide, 100 high: x = 0.5 * 200 and y = 0.25 * 100.
    assert landmark_to_point(landmark, 200, 100) == (100.0, 25.0)
    # Swapping the sizes must give a different (here: 50, 50) answer, which
    # proves the width and height are not mixed up.
    assert landmark_to_point(landmark, 100, 200) == (50.0, 50.0)


def test_visibility_is_not_part_of_the_point():
    landmark = {"x": 0.5, "y": 0.25, "visibility": 0.9}
    before = copy.deepcopy(landmark)
    point = landmark_to_point(landmark, 640, 480)
    assert len(point) == 2                      # only x and y
    assert landmark == before                   # nothing was changed


def test_landmark_dictionary_is_not_modified_by_the_report():
    landmarks = make_landmarks()
    before = copy.deepcopy(landmarks)
    run_report(landmarks, image_width=640, image_height=480)
    assert landmarks == before


# ---- 2. Angles measured in pixels ----------------------------------------

def test_pixel_scaling_corrects_the_angle_on_a_non_square_image():
    # On a 200 x 100 image, the shoulder is 50 pixels right and 50 pixels
    # down from the elbow, and the wrist is 50 pixels right and 50 pixels up.
    # In real pixels those two arms are exactly 90 degrees apart.
    # In the stretched 0-1 values the same arms look like ~126.87 degrees.
    landmarks = make_landmarks(
        left_elbow={"x": 0.5, "y": 0.5},
        left_shoulder={"x": 0.75, "y": 1.0},
        left_wrist={"x": 0.75, "y": 0.0},
    )
    with_size, _ = run_report(landmarks, image_width=200, image_height=100)
    without_size, _ = run_report(landmarks)

    assert math.isclose(with_size["elbow_angle"], 90.0, abs_tol=1e-6)
    assert math.isclose(without_size["elbow_angle"],
                        math.degrees(math.acos(-0.6)), abs_tol=1e-6)
    assert not math.isclose(with_size["elbow_angle"],
                            without_size["elbow_angle"], abs_tol=1.0)


def test_all_three_angles_match_a_hand_calculation_in_pixels():
    width, height = 640, 480

    def pixel(name):
        landmark = LANDMARK_EXAMPLE[name]
        return (landmark["x"] * width, landmark["y"] * height)

    expected = {
        "elbow_angle": calculate_angle(pixel("left_shoulder"),
                                       pixel("left_elbow"),
                                       pixel("left_wrist")),
        "hip_angle": calculate_angle(pixel("left_shoulder"),
                                     pixel("left_hip"),
                                     pixel("left_ankle")),
        "shoulder_angle": calculate_angle(pixel("left_elbow"),
                                          pixel("left_shoulder"),
                                          pixel("left_hip")),
    }
    sent, _ = run_report(LANDMARK_EXAMPLE, image_width=width,
                         image_height=height)
    for name, value in expected.items():
        assert math.isclose(sent[name], value, abs_tol=1e-9), name


def test_a_square_image_gives_the_same_angles_as_the_raw_values():
    # Scaling x and y by the SAME number does not change any angle.
    scaled, _ = run_report(LANDMARK_EXAMPLE, image_width=500, image_height=500)
    raw, _ = run_report(LANDMARK_EXAMPLE)
    for name in ("elbow_angle", "hip_angle", "shoulder_angle"):
        assert same(scaled[name], raw[name]), name


# ---- 3. Image size omitted -----------------------------------------------

def test_omitted_size_keeps_the_original_behaviour():
    landmark = {"x": 0.70, "y": 0.45, "visibility": 0.98}
    assert landmark_to_point(landmark) == (0.70, 0.45)

    omitted, _ = run_report(LANDMARK_EXAMPLE)
    explicit_none, _ = run_report(LANDMARK_EXAMPLE, image_width=None,
                                  image_height=None)
    assert omitted == explicit_none
    assert math.isclose(omitted["elbow_angle"], 180.0, abs_tol=1e-6)
    assert math.isclose(omitted["hip_angle"], 178.875, abs_tol=0.01)
    assert math.isclose(omitted["shoulder_angle"], 74.6466, abs_tol=0.001)


def test_the_old_way_of_calling_still_works():
    # Existing callers pass only (title, landmarks); the new parameters
    # must be optional.
    parameters = inspect.signature(print_landmark_report).parameters
    assert list(parameters) == ["title", "landmark_dict",
                                "image_width", "image_height"]
    assert parameters["image_width"].default is None
    assert parameters["image_height"].default is None
    landmark_parameters = inspect.signature(landmark_to_point).parameters
    assert landmark_parameters["image_width"].default is None
    assert landmark_parameters["image_height"].default is None


# ---- 4. Visibility is not scaled -----------------------------------------

def test_visibility_is_checked_on_the_original_values():
    # If visibility were scaled by the width (0.491 * 640 = 314), this wrist
    # would wrongly look "very visible". It must still be rejected.
    landmarks = make_landmarks(left_wrist={"visibility": 0.491},
                               left_ankle={"visibility": 0.137})
    sent, text = run_report(landmarks, image_width=640, image_height=480)
    assert sent["elbow_angle"] is None
    assert sent["hip_angle"] is None
    assert sent["shoulder_angle"] is not None
    assert "left_wrist visibility 0.491 is below 0.5" in text
    assert "left_ankle visibility 0.137 is below 0.5" in text


def test_the_same_angles_are_assessed_with_and_without_a_size():
    cases = [
        make_landmarks(),
        make_landmarks(left_wrist={"visibility": 0.2}),
        make_landmarks(left_hip={"visibility": 0.2}),
        make_landmarks(left_shoulder={"visibility": 0.2}),
    ]
    for landmarks in cases:
        assert (angles_assessed(landmarks, image_width=640, image_height=480)
                == angles_assessed(landmarks))


# ---- 5. Invalid or incomplete image sizes --------------------------------

def test_giving_only_one_dimension_raises_value_error():
    for size in ({"image_width": 640}, {"image_height": 480}):
        message = expect_error(ValueError, print_landmark_report,
                               "t", LANDMARK_EXAMPLE, **size)
        assert "together" in message


def test_zero_or_negative_dimensions_raise_value_error():
    for bad in (0, -1, -640):
        message = expect_error(ValueError, print_landmark_report, "t",
                               LANDMARK_EXAMPLE, image_width=bad,
                               image_height=480)
        assert "image_width" in message
        message = expect_error(ValueError, print_landmark_report, "t",
                               LANDMARK_EXAMPLE, image_width=640,
                               image_height=bad)
        assert "image_height" in message


def test_wrong_dimension_types_raise_type_error():
    bad_values = [640.0, 480.5, "640", True, False, [640],
                  float("nan"), float("inf")]
    for bad in bad_values:
        message = expect_error(TypeError, print_landmark_report, "t",
                               LANDMARK_EXAMPLE, image_width=bad,
                               image_height=480)
        assert "image_width" in message, bad
        message = expect_error(TypeError, print_landmark_report, "t",
                               LANDMARK_EXAMPLE, image_width=640,
                               image_height=bad)
        assert "image_height" in message, bad


def test_numpy_integer_sizes_are_accepted():
    with_numpy, _ = run_report(LANDMARK_EXAMPLE, image_width=np.int64(640),
                               image_height=np.int32(480))
    with_python, _ = run_report(LANDMARK_EXAMPLE, image_width=640,
                                image_height=480)
    assert with_numpy == with_python


def test_a_bad_size_stops_everything_before_any_output():
    with recording() as (sent, output):
        expect_error(ValueError, print_landmark_report, "t",
                     LANDMARK_EXAMPLE, image_width=640)
    assert sent == {}                    # the form analyzer was never called
    assert output.getvalue() == ""       # nothing was printed


def test_a_bad_size_is_rejected_even_with_no_landmarks():
    expect_error(ValueError, print_landmark_report, "t", {},
                 image_width=0, image_height=480)


def test_landmark_to_point_also_rejects_a_bad_size():
    landmark = {"x": 0.5, "y": 0.5, "visibility": 0.9}
    expect_error(ValueError, landmark_to_point, landmark, 640)       # width only
    expect_error(ValueError, landmark_to_point, landmark, 0, 480)
    expect_error(TypeError, landmark_to_point, landmark, "640", 480)


def test_check_image_size_returns_the_size_or_none():
    assert check_image_size(None, None) is None
    assert check_image_size(640, 480) == (640, 480)
    assert type(check_image_size(np.int64(640), 480)[0]) is int


# ---- 6. The simulated demo is unchanged ----------------------------------

def test_simulated_demo_output_is_exactly_what_it_was_before():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main()
    assert output.getvalue().splitlines() == GOLDEN_DEMO_LINES


# ---- 7. single_frame_demo.py passes the real frame size ------------------
# single_frame_demo.py needs a webcam and MediaPipe. These tests swap in tiny
# fake versions (and put the real modules back afterwards), so no camera is
# used and the tests run anywhere.

@contextlib.contextmanager
def fake_camera_and_mediapipe(frame_shape):
    """Yield a freshly imported single_frame_demo using a fake camera/MediaPipe."""
    names = ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "LEFT_HIP",
             "LEFT_KNEE", "LEFT_ANKLE", "RIGHT_SHOULDER", "RIGHT_ELBOW",
             "RIGHT_WRIST", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]
    pose_landmark = enum.IntEnum("PoseLandmark",
                                 {name: i for i, name in enumerate(names)})
    frame = np.zeros(frame_shape, dtype=np.uint8)   # shape: (height, width, 3)

    class FakeCapture:
        def read(self):
            return True, frame

        def release(self):
            pass

    class FakePose:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def close(self):
            # Real MediaPipe Pose objects have close(); Person 1's
            # PoseDetector.close() calls it.
            pass

        def process(self, rgb_frame):
            landmarks = {pose_landmark[name.upper()]: types.SimpleNamespace(**lm)
                         for name, lm in LANDMARK_EXAMPLE.items()}
            return types.SimpleNamespace(
                pose_landmarks=types.SimpleNamespace(landmark=landmarks))

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.COLOR_BGR2RGB = 4
    fake_cv2.VideoCapture = lambda index: FakeCapture()
    fake_cv2.cvtColor = lambda image, code: image

    fake_mediapipe = types.ModuleType("mediapipe")
    fake_mediapipe.solutions = types.SimpleNamespace(
        pose=types.SimpleNamespace(PoseLandmark=pose_landmark, Pose=FakePose,
                                   POSE_CONNECTIONS=None),
        drawing_utils=None)

    module_names = ["cv2", "mediapipe", "pose_detector", "single_frame_demo"]
    saved = {name: sys.modules.get(name) for name in module_names}
    try:
        for name in module_names:
            sys.modules.pop(name, None)          # force a fresh import
        sys.modules["cv2"] = fake_cv2
        sys.modules["mediapipe"] = fake_mediapipe
        yield importlib.import_module("single_frame_demo")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_single_frame_demo_passes_the_real_frame_size():
    # A 640-wide, 480-high frame has shape (480, 640, 3): height comes first.
    # Person 1's detector already reports PIXELS of that real frame, so the
    # size is not passed on (that would scale the pixels a second time); x
    # must come from the width and y from the height, not swapped.
    with fake_camera_and_mediapipe((480, 640, 3)) as demo:
        calls = []

        def recorder(title, landmarks, image_width=None, image_height=None):
            calls.append((image_width, image_height, landmarks))

        demo.print_landmark_report = recorder
        with contextlib.redirect_stdout(io.StringIO()):
            demo.main()
    assert [call[:2] for call in calls] == [(None, None)]
    elbow = calls[0][2]["left_elbow"]
    assert elbow["x"] == round(LANDMARK_EXAMPLE["left_elbow"]["x"] * 640)
    assert elbow["y"] == round(LANDMARK_EXAMPLE["left_elbow"]["y"] * 480)


def test_single_frame_demo_runs_end_to_end_with_a_non_square_frame():
    with fake_camera_and_mediapipe((480, 640, 3)) as demo:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            demo.main()
    text = output.getvalue()
    assert "Frame size: 640 x 480 pixels" in text
    assert "Real MediaPipe Frame" in text
    assert "Elbow Angle:" in text and "Status:" in text


# ---- Simple runner so `python test_landmark_coordinate_scaling.py` works -

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
