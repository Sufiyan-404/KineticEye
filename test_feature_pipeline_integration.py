"""
test_feature_pipeline_integration.py

Integration tests for the single-frame pipeline:

    landmarks -> print_landmark_report() -> (angles, analysis)
                                          -> extract_features() -> one row

They check that:
    - print_landmark_report() still prints the same report and now also
      returns the validated angles and the analyzer result,
    - single_frame_demo.py passes THOSE SAME values to extract_features()
      (None stays None for an angle rejected by the visibility check),
    - the row has the 11 expected columns, in order, with consistent values,
    - frame_number is None, the 0.5 threshold and the pixel scaling still
      apply, and the simulated demo output has not changed,
    - nothing is extracted when no person is detected.

No webcam and no real MediaPipe are used: single_frame_demo.py is imported
with tiny fake versions of cv2 and mediapipe (the real modules are put back
afterwards), so the tests run anywhere.

Run either way (put this file next to the project files):
    python test_feature_pipeline_integration.py
    pytest test_feature_pipeline_integration.py     (if you have pytest installed)
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
from pushup_angle_demo import LANDMARK_EXAMPLE, main, print_landmark_report

ANGLE_NAMES = ["elbow_angle", "hip_angle", "shoulder_angle"]

# The 11 columns we promised, written out here on purpose (not imported) so
# the test fails if the extractor's column list ever changes by accident.
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

# What print_landmark_report("Person 1 Landmark Format Example",
# LANDMARK_EXAMPLE) printed BEFORE it started returning values.
GOLDEN_REPORT_LINES = [
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

# What `python pushup_angle_demo.py` printed BEFORE this change.
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
] + GOLDEN_REPORT_LINES


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


def same(a, b):
    """Compare two angles (floats) or two Nones."""
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, abs_tol=1e-9)


def expected_angles(landmarks, width=None, height=None):
    """
    The three angles worked out independently, straight from
    calculate_angle() (not through pushup_angle_demo's helpers). With a
    width and height the 0-1 values are converted to pixels first.
    """
    def point(name):
        landmark = landmarks[name]
        if width is None:
            return (landmark["x"], landmark["y"])
        return (landmark["x"] * width, landmark["y"] * height)

    return {
        "elbow_angle": calculate_angle(point("left_shoulder"),
                                       point("left_elbow"),
                                       point("left_wrist")),
        "hip_angle": calculate_angle(point("left_shoulder"),
                                     point("left_hip"),
                                     point("left_ankle")),
        "shoulder_angle": calculate_angle(point("left_elbow"),
                                          point("left_shoulder"),
                                          point("left_hip")),
    }


def whole_pixel_points(landmarks, width, height):
    """
    {name: (x_px, y_px)}: the pixel positions exactly as Person 1's
    PoseDetector reports them (rounded to whole pixels, kept inside the image).
    """
    return {name: (min(width - 1, max(0, int(round(lm["x"] * width)))),
                   min(height - 1, max(0, int(round(lm["y"] * height)))))
            for name, lm in landmarks.items()}


def whole_pixel_fractions(landmarks, width, height):
    """The same whole-pixel positions written back as 0-1 fractions."""
    return {name: {"x": x / width, "y": y / height}
            for name, (x, y) in whole_pixel_points(
                landmarks, width, height).items()}


def expected_angles_whole_pixels(landmarks, width, height):
    """
    Like expected_angles(), but for the whole-pixel positions above. The
    single-frame demo now gets its landmarks from Person 1's detector, which
    reports whole pixels, so this is what its angles must match.
    """
    return expected_angles(whole_pixel_fractions(landmarks, width, height),
                           width, height)


def run_report(landmarks, **size):
    """
    Run print_landmark_report() while recording what analyze_push_up_form()
    receives. Returns (returned_value, analyzer_kwargs, printed_text).
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
            returned = print_landmark_report("test", landmarks, **size)
    finally:
        pushup_angle_demo.analyze_push_up_form = real_analyzer
    return returned, sent, output.getvalue()


# ---- Fake camera and MediaPipe (same idea as the scaling tests) ----------

@contextlib.contextmanager
def fake_camera_and_mediapipe(frame_shape=(480, 640, 3), landmarks=None,
                              person_detected=True, camera_works=True):
    """
    Yield a freshly imported single_frame_demo that uses a fake camera and a
    fake MediaPipe. frame_shape is (height, width, colours).
    person_detected=False makes MediaPipe find nobody; camera_works=False
    makes the camera return no frame.
    """
    if landmarks is None:
        landmarks = LANDMARK_EXAMPLE
    names = ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "LEFT_HIP",
             "LEFT_KNEE", "LEFT_ANKLE", "RIGHT_SHOULDER", "RIGHT_ELBOW",
             "RIGHT_WRIST", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]
    pose_landmark = enum.IntEnum("PoseLandmark",
                                 {name: i for i, name in enumerate(names)})
    frame = np.zeros(frame_shape, dtype=np.uint8)   # (height, width, 3)

    class FakeCapture:
        def read(self):
            if camera_works:
                return True, frame
            return False, None

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
            if not person_detected:
                return types.SimpleNamespace(pose_landmarks=None)
            found = {pose_landmark[name.upper()]: types.SimpleNamespace(**lm)
                     for name, lm in landmarks.items()}
            return types.SimpleNamespace(
                pose_landmarks=types.SimpleNamespace(landmark=found))

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


def run_single_frame(landmarks=None, frame_shape=(480, 640, 3), **fake_options):
    """
    Run single_frame_demo.main() with the fake camera and record the calls.

    Returns a namespace with:
        text            everything main() printed
        report_calls    arguments of each print_landmark_report() call
        report_returns  what each print_landmark_report() call returned
        extract_calls   arguments of each extract_features() call
        rows            what each extract_features() call returned
    """
    with fake_camera_and_mediapipe(frame_shape, landmarks,
                                   **fake_options) as demo:
        report_calls, report_returns = [], []
        extract_calls, rows = [], []
        real_report = demo.print_landmark_report
        real_extract = demo.extract_features
        report_signature = inspect.signature(real_report)
        extract_signature = inspect.signature(real_extract)

        def recording_report(*args, **kwargs):
            report_calls.append(
                dict(report_signature.bind(*args, **kwargs).arguments))
            returned = real_report(*args, **kwargs)
            report_returns.append(returned)
            return returned

        def recording_extract(*args, **kwargs):
            extract_calls.append(
                dict(extract_signature.bind(*args, **kwargs).arguments))
            row = real_extract(*args, **kwargs)
            rows.append(row)
            return row

        demo.print_landmark_report = recording_report
        demo.extract_features = recording_extract

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            demo.main()

    return types.SimpleNamespace(text=output.getvalue(),
                                 report_calls=report_calls,
                                 report_returns=report_returns,
                                 extract_calls=extract_calls,
                                 rows=rows)


# ---- 1. print_landmark_report() prints the same report and returns -------

def test_report_still_prints_the_existing_report():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        print_landmark_report("Person 1 Landmark Format Example",
                              LANDMARK_EXAMPLE)
    assert output.getvalue().splitlines() == GOLDEN_REPORT_LINES


def test_report_returns_the_validated_angles_and_the_analyzer_result():
    returned, sent, _ = run_report(LANDMARK_EXAMPLE)

    # A tuple of exactly two things: (angles, analyzer result).
    assert isinstance(returned, tuple) and len(returned) == 2
    angles, result = returned

    # The angle dictionary has exactly the three documented keys.
    assert list(angles) == ANGLE_NAMES
    expected = expected_angles(LANDMARK_EXAMPLE)
    for name in ANGLE_NAMES:
        assert same(angles[name], expected[name]), name

    # The second item is the analyzer's result dictionary.
    assert result["form_status"] == "GOOD"
    assert result["movement_phase"] == "TOP"
    assert result["feedback"] == ["Good form"]


def test_nothing_is_measured_or_analyzed_twice():
    calculate_calls = []
    analyzer_results = []
    real_calculate = pushup_angle_demo.calculate_angle
    real_analyzer = pushup_angle_demo.analyze_push_up_form

    def counting_calculate(a, b, c):
        calculate_calls.append((a, b, c))
        return real_calculate(a, b, c)

    def counting_analyzer(**kwargs):
        result = real_analyzer(**kwargs)
        analyzer_results.append(result)
        return result

    pushup_angle_demo.calculate_angle = counting_calculate
    pushup_angle_demo.analyze_push_up_form = counting_analyzer
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _, result = print_landmark_report("test", LANDMARK_EXAMPLE)
    finally:
        pushup_angle_demo.calculate_angle = real_calculate
        pushup_angle_demo.analyze_push_up_form = real_analyzer

    assert len(calculate_calls) == 3          # elbow, hip, shoulder: once each
    assert len(analyzer_results) == 1         # one analysis
    assert result is analyzer_results[0]      # the very same dictionary


def test_returned_angles_are_exactly_what_the_analyzer_received():
    scenarios = {
        "all good": make_landmarks(),
        "wrist hidden": make_landmarks(left_wrist={"visibility": 0.491}),
        "ankle hidden": make_landmarks(left_ankle={"visibility": 0.212}),
        "hip hidden": make_landmarks(left_hip={"visibility": 0.2}),
        "shoulder hidden": make_landmarks(left_shoulder={"visibility": 0.2}),
        "no landmarks": {},
        "not a dictionary": None,
    }
    for label, landmarks in scenarios.items():
        (angles, _), sent, _ = run_report(landmarks)
        for name in ANGLE_NAMES:
            assert same(angles[name], sent[name]), (label, name)


def test_a_rejected_angle_is_none_in_the_returned_angles():
    (angles, result), _, _ = run_report(
        make_landmarks(left_ankle={"visibility": 0.212}))
    assert angles["hip_angle"] is None
    assert angles["elbow_angle"] is not None
    assert angles["shoulder_angle"] is not None
    assert result["form_status"] == "INCOMPLETE_DATA"

    (angles, result), _, _ = run_report({})
    assert [angles[name] for name in ANGLE_NAMES] == [None, None, None]
    assert result["movement_phase"] == "UNKNOWN"


def test_callers_that_ignore_the_return_value_still_work():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        print_landmark_report("test", LANDMARK_EXAMPLE)   # result ignored
        assert main() is None                             # main() unchanged
    assert "Status: GOOD" in output.getvalue()


# ---- 2. The simulated demo is unchanged ----------------------------------

def test_simulated_demo_output_is_exactly_what_it_was_before():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main()
    assert output.getvalue().splitlines() == GOLDEN_DEMO_LINES


# ---- 3. single_frame_demo.py passes the same values to the extractor -----

def test_extract_features_is_called_once_with_the_reports_own_values():
    run = run_single_frame()
    assert len(run.report_calls) == 1
    assert len(run.extract_calls) == 1

    angles, analysis = run.report_returns[0]
    call = run.extract_calls[0]
    for name in ANGLE_NAMES:
        assert same(call[name], angles[name]), name
    assert call["analysis"] is analysis        # the very same dictionary


def test_a_rejected_angle_reaches_the_extractor_as_none():
    # The real webcam case: ankle visibility 0.212 is below 0.5.
    run = run_single_frame(make_landmarks(left_ankle={"visibility": 0.212}))
    call = run.extract_calls[0]
    row = run.rows[0]

    assert call["hip_angle"] is None
    assert call["elbow_angle"] is not None
    assert call["shoulder_angle"] is not None

    assert row["hip_angle"] is None
    assert row["hip_angle_available"] is False
    assert row["elbow_angle_available"] is True
    assert row["shoulder_angle_available"] is True
    assert row["all_angles_available"] is False
    assert row["rule_form_status"] == "INCOMPLETE_DATA"
    assert row["rule_feedback"] == (
        "Hip angle missing: cannot check back alignment")


def test_every_rejected_angle_stays_none_in_the_row():
    cases = {
        "elbow_angle": make_landmarks(left_wrist={"visibility": 0.1}),
        "hip_angle": make_landmarks(left_ankle={"visibility": 0.1}),
    }
    for name, landmarks in cases.items():
        row = run_single_frame(landmarks).rows[0]
        assert row[name] is None, name
        assert row[f"{name}_available"] is False, name
        assert row["all_angles_available"] is False, name

    # A hidden shoulder rejects all three angles.
    row = run_single_frame(
        make_landmarks(left_shoulder={"visibility": 0.1})).rows[0]
    assert [row[name] for name in ANGLE_NAMES] == [None, None, None]
    assert row["rule_movement_phase"] == "UNKNOWN"


# ---- 4. The feature row ---------------------------------------------------

def test_the_row_has_the_11_expected_columns_in_order():
    row = run_single_frame().rows[0]
    assert list(row) == EXPECTED_COLUMNS
    assert len(row) == 11


def test_row_values_are_consistent_with_the_inputs_and_the_analysis():
    scenarios = [
        make_landmarks(),                                   # everything fine
        make_landmarks(left_ankle={"visibility": 0.212}),   # hip rejected
        make_landmarks(left_wrist={"visibility": 0.491}),   # elbow rejected
        make_landmarks(left_shoulder={"visibility": 0.1}),  # all rejected
    ]
    for landmarks in scenarios:
        run = run_single_frame(landmarks)
        angles, analysis = run.report_returns[0]
        row = run.rows[0]

        for name in ANGLE_NAMES:
            assert same(row[name], angles[name]), name
            assert row[f"{name}_available"] is (angles[name] is not None)
        assert row["all_angles_available"] is all(
            angles[name] is not None for name in ANGLE_NAMES)

        # The rule_* columns are copied from the analyzer, not recalculated.
        assert row["rule_movement_phase"] == analysis["movement_phase"]
        assert row["rule_form_status"] == analysis["form_status"]
        assert row["rule_feedback"] == "; ".join(analysis["feedback"])


def test_all_angles_available_is_true_only_when_all_three_are():
    row = run_single_frame().rows[0]
    assert row["all_angles_available"] is True
    assert [row[f"{name}_available"] for name in ANGLE_NAMES] == [True] * 3

    for hidden in ("left_wrist", "left_ankle"):
        row = run_single_frame(
            make_landmarks(**{hidden: {"visibility": 0.1}})).rows[0]
        assert row["all_angles_available"] is False


def test_a_fully_visible_frame_gives_the_expected_good_row():
    row = run_single_frame().rows[0]
    expected = expected_angles_whole_pixels(LANDMARK_EXAMPLE, 640, 480)
    for name in ANGLE_NAMES:
        assert same(row[name], expected[name]), name
    assert row["rule_movement_phase"] == "TOP"
    assert row["rule_form_status"] == "GOOD"
    assert row["rule_feedback"] == "Good form"


def test_frame_number_is_none_and_is_not_invented():
    run = run_single_frame()
    call = run.extract_calls[0]
    assert "frame_number" in call            # passed explicitly...
    assert call["frame_number"] is None      # ...as None
    assert run.rows[0]["frame_number"] is None


# ---- 5. Compatible with pixel scaling and visibility validation ----------

def test_the_row_uses_pixel_coordinates_of_the_real_frame():
    # On a 640 x 480 frame the angles must be measured in pixels.
    row = run_single_frame(frame_shape=(480, 640, 3)).rows[0]
    in_pixels = expected_angles_whole_pixels(LANDMARK_EXAMPLE, 640, 480)
    raw_values = expected_angles(LANDMARK_EXAMPLE)
    assert same(row["shoulder_angle"], in_pixels["shoulder_angle"])
    # The shoulder angle in pixels is clearly different from the raw 0-1
    # value, so this fails if the scaling is lost on the way to the row.
    assert abs(row["shoulder_angle"] - raw_values["shoulder_angle"]) > 1.0


def test_the_real_frame_size_is_passed_with_width_and_height_not_swapped():
    # Person 1's detector already reports PIXELS of the real frame, so the
    # frame size must NOT be passed on (the report would scale the pixels a
    # second time). The pixels must still come from the right dimensions:
    # x from the width, y from the height, never swapped.
    for shape, width, height in (((480, 640, 3), 640, 480),
                                 ((720, 1280, 3), 1280, 720)):
        call = run_single_frame(frame_shape=shape).report_calls[0]
        assert "image_width" not in call and "image_height" not in call
        expected = whole_pixel_points(LANDMARK_EXAMPLE, width, height)
        for name, (x_px, y_px) in expected.items():
            assert call["landmark_dict"][name]["x"] == x_px, name
            assert call["landmark_dict"][name]["y"] == y_px, name


def test_different_frame_sizes_change_the_row_as_expected():
    wide = run_single_frame(frame_shape=(480, 640, 3)).rows[0]
    square = run_single_frame(frame_shape=(500, 500, 3)).rows[0]
    expected_square = expected_angles_whole_pixels(LANDMARK_EXAMPLE, 500, 500)
    # The raw (0-1) values of the same whole-pixel positions the detector
    # reports.
    raw_values = expected_angles(
        whole_pixel_fractions(LANDMARK_EXAMPLE, 500, 500))
    # A square frame scales x and y equally, so it matches the raw values
    # exactly (the same strict comparison as before).
    assert same(square["shoulder_angle"], expected_square["shoulder_angle"])
    assert same(square["shoulder_angle"], raw_values["shoulder_angle"])
    assert not same(wide["shoulder_angle"], square["shoulder_angle"])


def test_the_visibility_threshold_is_still_half():
    assert pushup_angle_demo.VISIBILITY_THRESHOLD == 0.5

    # Exactly 0.5 is accepted, just below is rejected, all the way to the row.
    at_threshold = run_single_frame(
        make_landmarks(left_wrist={"visibility": 0.5})).rows[0]
    assert at_threshold["elbow_angle_available"] is True

    just_below = run_single_frame(
        make_landmarks(left_wrist={"visibility": 0.499})).rows[0]
    assert just_below["elbow_angle"] is None
    assert just_below["elbow_angle_available"] is False


def test_each_angle_still_depends_on_its_own_landmarks():
    # Right-side landmarks are not used by any angle.
    row = run_single_frame(make_landmarks(
        right_shoulder={"visibility": 0.0},
        right_wrist={"visibility": 0.0})).rows[0]
    assert row["all_angles_available"] is True

    # The knee is not used either.
    row = run_single_frame(
        make_landmarks(left_knee={"visibility": 0.0})).rows[0]
    assert row["all_angles_available"] is True

    # The wrist is used only by the elbow angle.
    row = run_single_frame(
        make_landmarks(left_wrist={"visibility": 0.1})).rows[0]
    assert [row[f"{name}_available"] for name in ANGLE_NAMES] == [
        False, True, True]


# ---- 6. Printing the feature row -----------------------------------------

def test_the_existing_report_is_printed_before_the_feature_row():
    text = run_single_frame().text
    assert "Real MediaPipe Frame" in text
    assert "Visibility Check (threshold 0.5):" in text
    assert text.index("Real MediaPipe Frame") < text.index("Feature row")


def test_the_printed_feature_row_shows_all_11_fields_in_order():
    run = run_single_frame(make_landmarks(left_ankle={"visibility": 0.212}))
    row = run.rows[0]
    lines = run.text.splitlines()

    start = lines.index("Feature row (one frame):")
    printed = lines[start + 1:start + 1 + 11]
    assert len(printed) == 11

    for line, (name, value) in zip(printed, row.items()):
        assert line.startswith(f"  {name}"), (line, name)
        assert line.endswith(str(value)), (line, value)
    assert [line.split()[0] for line in printed] == EXPECTED_COLUMNS


# ---- 7. Paths where nothing may be extracted -----------------------------

def test_no_person_detected_does_not_attempt_feature_extraction():
    run = run_single_frame(person_detected=False)
    assert "No person detected" in run.text
    assert run.report_calls == []
    assert run.extract_calls == []
    assert "Feature row" not in run.text


def test_camera_failure_does_not_attempt_feature_extraction():
    run = run_single_frame(camera_works=False)
    assert "Could not read a frame" in run.text
    assert run.report_calls == []
    assert run.extract_calls == []


def test_a_report_that_returns_nothing_is_tolerated_without_extraction():
    # An existing test replaces print_landmark_report() with a stand-in that
    # returns None. main() must not crash, and must not build a row from
    # measurements that do not exist.
    with fake_camera_and_mediapipe() as demo:
        extract_calls = []
        demo.print_landmark_report = lambda *args, **kwargs: None
        demo.extract_features = lambda *args, **kwargs: extract_calls.append(1)
        with contextlib.redirect_stdout(io.StringIO()):
            demo.main()
    assert extract_calls == []


# ---- Simple runner so `python test_feature_pipeline_integration.py` works -

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
