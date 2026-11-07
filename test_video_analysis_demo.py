"""
test_video_analysis_demo.py

Tests for video_analysis_demo.py (video file -> one feature row per frame ->
CSV file) and for the small non-printing helper analyze_landmarks() that was
added to pushup_angle_demo.py.

No real video, webcam or MediaPipe is used. video_analysis_demo.py is imported
with tiny fake versions of cv2 and mediapipe (the real modules are put back
afterwards), and a fake "video" is just a list of small NumPy arrays. The CSV
files are written to a temporary folder that is deleted after each test.

Run either way (put this file next to the project files):
    python test_video_analysis_demo.py
    pytest -v test_video_analysis_demo.py       (if you have pytest installed)
"""

import contextlib
import copy
import csv
import enum
import importlib
import inspect
import io
import math
import os
import sys
import tempfile
import types

import numpy as np

import pushup_angle_demo
from angle_calculator import calculate_angle
from pushup_angle_demo import (
    LANDMARK_EXAMPLE,
    analyze_landmarks,
    main as simulated_demo_main,
    print_landmark_report,
)

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
# LANDMARK_EXAMPLE) printed BEFORE analyze_landmarks() existed.
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


def make_frame(height=480, width=640):
    """A fake BGR video frame: shape (height, width, 3), distinct channels."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = 10      # blue
    frame[..., 1] = 20      # green
    frame[..., 2] = 30      # red
    return frame


def same(a, b):
    """Compare two angles (floats) or two Nones."""
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, abs_tol=1e-9)


def expected_angles(landmarks, width, height, whole_pixels=False):
    """
    The three angles worked out independently, straight from
    calculate_angle(), after converting the 0-1 values to pixels.
    whole_pixels=True first rounds every coordinate to a whole pixel inside
    the image, exactly as Person 1's PoseDetector reports it.
    """
    def point(name):
        landmark = landmarks[name]
        x, y = landmark["x"] * width, landmark["y"] * height
        if whole_pixels:
            x = float(min(width - 1, max(0, int(round(x)))))
            y = float(min(height - 1, max(0, int(round(y)))))
        return (x, y)

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


def expect_error(error_type, function, *args, **kwargs):
    """Check that function(...) raises error_type; return its message."""
    try:
        function(*args, **kwargs)
    except error_type as error:
        return str(error)
    raise AssertionError(
        f"expected {error_type.__name__}, but no error was raised")


# ---- Fake camera/video and MediaPipe -------------------------------------

class FakeCapture:
    """Stands in for cv2.VideoCapture: hands out the given frames, then ends."""

    def __init__(self, frames=(), opened=True):
        self.frames = list(frames)
        self.opened = opened
        self.released = False
        self.position = 0

    def isOpened(self):
        return self.opened

    def read(self):
        if self.position < len(self.frames):
            frame = self.frames[self.position]
            self.position += 1
            return True, frame
        return False, None

    def get(self, prop):
        return len(self.frames)

    def release(self):
        self.released = True


@contextlib.contextmanager
def fake_environment(frames=(), per_frame_landmarks=None, opened=True,
                     process_error=None, process_error_at=0):
    """
    Yield a namespace with a freshly imported video_analysis_demo (as `demo`)
    that uses a fake OpenCV and a fake MediaPipe.

    frames               list of fake BGR frames (the "video")
    per_frame_landmarks  one entry per frame: a landmark dictionary like
                         LANDMARK_EXAMPLE, or None for "nobody detected".
                         Default: the good example landmarks in every frame.
    opened               False makes the fake video refuse to open
    process_error        an exception that the fake MediaPipe raises when it
                         reaches frame number `process_error_at`

    Also available: capture, poses (every fake Pose created), cv2_calls,
    capture_requests (paths passed to cv2.VideoCapture) and import_module()
    for importing another project module with the same fakes.
    """
    frames = list(frames)
    if per_frame_landmarks is None:
        per_frame_landmarks = [LANDMARK_EXAMPLE] * len(frames)

    names = ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "LEFT_HIP",
             "LEFT_KNEE", "LEFT_ANKLE", "RIGHT_SHOULDER", "RIGHT_ELBOW",
             "RIGHT_WRIST", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"]
    pose_landmark = enum.IntEnum("PoseLandmark",
                                 {name: i for i, name in enumerate(names)})

    capture = FakeCapture(frames, opened)
    env = types.SimpleNamespace(capture=capture, poses=[], cv2_calls=[],
                                capture_requests=[], demo=None,
                                import_module=importlib.import_module)

    class FakePose:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.entered = True             # created = started
            self.exited = False
            self.received = []              # every array given to process()
            env.poses.append(self)

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *args):
            self.exited = True
            return False

        def close(self):                    # what PoseDetector.close() calls
            self.exited = True

        def process(self, rgb_frame):
            index = len(self.received)
            self.received.append(rgb_frame)
            if process_error is not None and index == process_error_at:
                raise process_error
            landmarks = per_frame_landmarks[index]
            if landmarks is None:
                return types.SimpleNamespace(pose_landmarks=None)
            found = {pose_landmark[name.upper()]: types.SimpleNamespace(**lm)
                     for name, lm in landmarks.items()}
            return types.SimpleNamespace(
                pose_landmarks=types.SimpleNamespace(landmark=found))

    def fake_video_capture(path):
        env.capture_requests.append(path)
        return capture

    def fake_cvt_color(image, code):
        env.cv2_calls.append(code)
        return image[..., ::-1].copy()      # reverse the colour channels

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.COLOR_BGR2RGB = 4
    fake_cv2.CAP_PROP_FRAME_COUNT = 7
    fake_cv2.VideoCapture = fake_video_capture
    fake_cv2.cvtColor = fake_cvt_color

    fake_mediapipe = types.ModuleType("mediapipe")
    fake_mediapipe.solutions = types.SimpleNamespace(
        pose=types.SimpleNamespace(PoseLandmark=pose_landmark, Pose=FakePose,
                                   POSE_CONNECTIONS=None),
        drawing_utils=None)

    module_names = ["cv2", "mediapipe", "pose_detector",
                    "video_analysis_demo", "single_frame_demo"]
    saved = {name: sys.modules.get(name) for name in module_names}
    try:
        for name in module_names:
            sys.modules.pop(name, None)          # force a fresh import
        sys.modules["cv2"] = fake_cv2
        sys.modules["mediapipe"] = fake_mediapipe
        env.demo = importlib.import_module("video_analysis_demo")
        yield env
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def read_csv_file(path):
    """Return (header, records): the header list and a dict per data row."""
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        records = [dict(zip(header, line)) for line in reader]
    return header, records


def run_video(frames, per_frame_landmarks=None, **fake_options):
    """
    Run process_video() on a fake video and return what happened.

    Returns a namespace with: summary, header, records (the CSV, as text),
    text (what was printed), analyzer_calls (the keyword arguments given to
    analyze_landmarks for each frame) and env (see fake_environment).
    """
    with tempfile.TemporaryDirectory() as folder, \
            fake_environment(frames, per_frame_landmarks,
                             **fake_options) as env:
        video_path = os.path.join(folder, "clip.mp4")
        output_path = os.path.join(folder, "out.csv")
        with open(video_path, "wb") as file:
            file.write(b"not a real video")      # only needs to exist

        analyzer_calls = []
        analyzer_landmarks = []
        real_analyze = env.demo.analyze_landmarks

        def recording_analyze(landmarks, **kwargs):
            analyzer_calls.append(dict(kwargs))
            analyzer_landmarks.append(landmarks)
            return real_analyze(landmarks, **kwargs)

        env.demo.analyze_landmarks = recording_analyze

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            summary = env.demo.process_video(video_path, output_path)
        header, records = read_csv_file(output_path)
        return types.SimpleNamespace(
            summary=summary, header=header, records=records,
            text=output.getvalue(), analyzer_calls=analyzer_calls,
            analyzer_landmarks=analyzer_landmarks, env=env)


def run_main(argv_builder, frames=(), **fake_options):
    """
    Run demo.main() with a fake video. argv_builder(folder) returns the
    command-line arguments. Returns (exit_code, stdout, stderr, folder_files,
    env). folder_files lists the files left in the temporary folder.
    """
    with tempfile.TemporaryDirectory() as folder, \
            fake_environment(frames, **fake_options) as env:
        with open(os.path.join(folder, "clip.mp4"), "wb") as file:
            file.write(b"not a real video")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = env.demo.main(argv_builder(folder))
        return code, out.getvalue(), err.getvalue(), sorted(
            os.listdir(folder)), env


# ---- 1. One row per decoded frame ----------------------------------------

def test_every_decoded_frame_gives_exactly_one_row():
    run = run_video([make_frame() for _ in range(5)])
    assert len(run.records) == 5
    assert run.summary.frames_processed == 5
    assert len(run.env.poses[0].received) == 5      # MediaPipe saw each frame


def test_frame_numbers_start_at_zero_and_increase_by_one():
    run = run_video([make_frame() for _ in range(6)])
    assert [r["frame_number"] for r in run.records] == [
        "0", "1", "2", "3", "4", "5"]


def test_frame_numbers_keep_counting_through_frames_without_a_person():
    landmarks = [LANDMARK_EXAMPLE, None, None, LANDMARK_EXAMPLE, None]
    run = run_video([make_frame() for _ in landmarks], landmarks)
    assert [r["frame_number"] for r in run.records] == [
        "0", "1", "2", "3", "4"]
    assert run.summary.frames_processed == 5
    assert run.summary.frames_with_person == 2


def test_a_single_frame_video_gives_one_row_numbered_zero():
    run = run_video([make_frame()])
    assert [r["frame_number"] for r in run.records] == ["0"]


# ---- 2. Real frame size, colour conversion, video mode -------------------

def test_actual_width_and_height_are_passed_to_the_coordinate_processing():
    frames = [make_frame(height=480, width=640),
              make_frame(height=720, width=1280),
              make_frame(height=500, width=500)]
    run = run_video(frames)
    # Person 1's detector already reports PIXELS of each real frame, so no
    # size is passed on (that would scale them a second time) ...
    assert run.analyzer_calls == [{}, {}, {}]
    # ... and the pixels come from each frame's own width (x) and height (y).
    for (height, width), landmarks in zip(((480, 640), (720, 1280), (500, 500)),
                                          run.analyzer_landmarks):
        elbow = LANDMARK_EXAMPLE["left_elbow"]
        assert landmarks["left_elbow"]["x"] == round(elbow["x"] * width)
        assert landmarks["left_elbow"]["y"] == round(elbow["y"] * height)


def test_angles_are_measured_in_pixels_of_the_real_frame():
    run = run_video([make_frame(height=480, width=640)])
    expected = expected_angles(LANDMARK_EXAMPLE, 640, 480, whole_pixels=True)
    for name in ANGLE_NAMES:
        assert same(float(run.records[0][name]), expected[name]), name

    # In pixels the shoulder angle is clearly different from the raw 0-1
    # value, so this fails if the scaling is lost on the way to the CSV.
    raw = expected_angles(LANDMARK_EXAMPLE, width=1, height=1)
    assert abs(float(run.records[0]["shoulder_angle"])
               - raw["shoulder_angle"]) > 1.0


def test_frames_are_converted_from_bgr_to_rgb_before_mediapipe():
    run = run_video([make_frame()])
    assert run.env.cv2_calls == [4]                  # cv2.COLOR_BGR2RGB
    received = run.env.poses[0].received[0]
    # Original frame is (B, G, R) = (10, 20, 30); MediaPipe must get (R, G, B).
    assert tuple(received[0, 0]) == (30, 20, 10)


def test_mediapipe_runs_once_in_video_mode_and_is_closed():
    run = run_video([make_frame() for _ in range(4)])
    assert len(run.env.poses) == 1                   # one model, not one/frame
    pose = run.env.poses[0]
    assert pose.kwargs["static_image_mode"] is False
    assert pose.entered and pose.exited


# ---- 3. Frames without a usable pose -------------------------------------

def test_frame_without_a_person_still_produces_a_row_with_no_angles():
    run = run_video([make_frame(), make_frame()],
                    [LANDMARK_EXAMPLE, None])
    assert len(run.records) == 2
    row = run.records[1]
    assert row["frame_number"] == "1"
    for name in ANGLE_NAMES:
        assert row[name] == ""                       # None -> empty cell
        assert row[f"{name}_available"] == "False"
    assert row["all_angles_available"] == "False"
    assert row["rule_movement_phase"] == "UNKNOWN"
    assert row["rule_form_status"] == "INCOMPLETE_DATA"
    assert "Elbow angle missing" in row["rule_feedback"]

    # The neighbouring good frame is not affected.
    assert run.records[0]["all_angles_available"] == "True"
    assert run.records[0]["rule_form_status"] == "GOOD"


def test_a_video_where_nobody_is_detected_still_gives_all_rows():
    run = run_video([make_frame() for _ in range(3)], [None, None, None])
    assert len(run.records) == 3
    assert run.summary.frames_with_person == 0
    assert all(r["rule_form_status"] == "INCOMPLETE_DATA"
               for r in run.records)


# ---- 4. Visibility validation (threshold 0.5) ----------------------------

def test_visibility_failures_give_unavailable_angles_and_false_flags():
    cases = [
        # (changed landmarks, available flags for elbow, hip, shoulder)
        (make_landmarks(left_wrist={"visibility": 0.491}),
         ("False", "True", "True")),
        (make_landmarks(left_ankle={"visibility": 0.137}),
         ("True", "False", "True")),
        (make_landmarks(left_hip={"visibility": 0.2}),
         ("True", "False", "False")),
        (make_landmarks(left_shoulder={"visibility": 0.2}),
         ("False", "False", "False")),
    ]
    run = run_video([make_frame() for _ in cases],
                    [landmarks for landmarks, _ in cases])
    for row, (_, flags) in zip(run.records, cases):
        assert (row["elbow_angle_available"], row["hip_angle_available"],
                row["shoulder_angle_available"]) == flags
        for name, flag in zip(ANGLE_NAMES, flags):
            assert (row[name] == "") == (flag == "False"), name
        assert row["all_angles_available"] == "False"
        assert row["rule_form_status"] == "INCOMPLETE_DATA"


def test_threshold_is_still_half_exactly_at_half_is_accepted():
    assert pushup_angle_demo.VISIBILITY_THRESHOLD == 0.5
    landmarks = [make_landmarks(left_wrist={"visibility": 0.5}),
                 make_landmarks(left_wrist={"visibility": 0.499})]
    run = run_video([make_frame(), make_frame()], landmarks)
    assert run.records[0]["elbow_angle_available"] == "True"
    assert run.records[1]["elbow_angle_available"] == "False"
    assert run.records[1]["elbow_angle"] == ""


def test_right_side_and_knee_visibility_are_ignored():
    landmarks = make_landmarks(right_shoulder={"visibility": 0.0},
                               right_wrist={"visibility": 0.0},
                               left_knee={"visibility": 0.0})
    run = run_video([make_frame()], [landmarks])
    assert run.records[0]["all_angles_available"] == "True"
    assert run.records[0]["rule_form_status"] == "GOOD"


# ---- 5. The CSV file ------------------------------------------------------

def test_csv_has_the_exact_11_columns_in_the_agreed_order():
    run = run_video([make_frame() for _ in range(3)])
    assert run.header == EXPECTED_COLUMNS
    assert len(run.header) == 11
    for record in run.records:
        assert list(record) == EXPECTED_COLUMNS


def test_demo_uses_the_extractors_column_list():
    with fake_environment() as env:
        assert env.demo.FEATURE_COLUMNS == EXPECTED_COLUMNS


def test_csv_values_for_a_good_frame():
    run = run_video([make_frame(height=480, width=640)])
    row = run.records[0]
    expected = expected_angles(LANDMARK_EXAMPLE, 640, 480, whole_pixels=True)
    assert row["frame_number"] == "0"
    for name in ANGLE_NAMES:
        assert same(float(row[name]), expected[name]), name
        assert row[f"{name}_available"] == "True"
    assert row["all_angles_available"] == "True"
    assert row["rule_movement_phase"] == "TOP"
    assert row["rule_form_status"] == "GOOD"
    assert row["rule_feedback"] == "Good form"


def test_every_row_has_every_column_even_with_a_semicolon_in_feedback():
    # Feedback with several messages contains "; " and must stay one cell.
    run = run_video([make_frame()], [None])
    assert len(run.records[0]) == 11
    assert run.records[0]["rule_feedback"].count("; ") == 2


# ---- 6. Invalid or unreadable input --------------------------------------

def test_missing_video_path_is_reported_before_anything_is_opened():
    with tempfile.TemporaryDirectory() as folder, fake_environment() as env:
        missing = os.path.join(folder, "nope.mp4")
        message = expect_error(env.demo.VideoAnalysisError,
                               env.demo.process_video, missing,
                               os.path.join(folder, "out.csv"))
        assert "not found" in message and "nope.mp4" in message
        assert env.capture_requests == []            # video never opened
        assert not os.path.exists(os.path.join(folder, "out.csv"))


def test_a_folder_instead_of_a_video_is_reported():
    with tempfile.TemporaryDirectory() as folder, fake_environment() as env:
        message = expect_error(env.demo.VideoAnalysisError,
                               env.demo.process_video, folder,
                               os.path.join(folder, "out.csv"))
        assert "folder" in message
        assert env.capture_requests == []


def test_bad_video_argument_types_are_reported_not_crashed():
    with fake_environment() as env:
        for bad in (None, "", 42):
            expect_error(env.demo.VideoAnalysisError,
                         env.demo.process_video, bad, "out.csv")


def test_output_folder_that_does_not_exist_is_reported_early():
    with tempfile.TemporaryDirectory() as folder, \
            fake_environment([make_frame()]) as env:
        video = os.path.join(folder, "clip.mp4")
        with open(video, "wb") as file:
            file.write(b"x")
        bad_output = os.path.join(folder, "no_such_folder", "out.csv")
        message = expect_error(env.demo.VideoAnalysisError,
                               env.demo.process_video, video, bad_output)
        assert "does not exist" in message
        assert env.capture_requests == []


def test_output_path_may_not_be_a_folder_or_the_video_itself():
    with tempfile.TemporaryDirectory() as folder, \
            fake_environment([make_frame()]) as env:
        video = os.path.join(folder, "clip.mp4")
        with open(video, "wb") as file:
            file.write(b"precious video")
        expect_error(env.demo.VideoAnalysisError,
                     env.demo.process_video, video, folder)
        message = expect_error(env.demo.VideoAnalysisError,
                               env.demo.process_video, video, video)
        assert "overwrite" in message
        with open(video, "rb") as file:
            assert file.read() == b"precious video"   # untouched
        assert env.capture_requests == []


def test_a_video_opencv_cannot_open_is_reported_and_released():
    with tempfile.TemporaryDirectory() as folder, \
            fake_environment(opened=False) as env:
        video = os.path.join(folder, "broken.mp4")
        output = os.path.join(folder, "out.csv")
        with open(video, "wb") as file:
            file.write(b"garbage")
        message = expect_error(env.demo.VideoAnalysisError,
                               env.demo.process_video, video, output)
        assert "could not open" in message
        assert env.capture.released is True
        assert env.poses == []                       # MediaPipe never started
        assert not os.path.exists(output)


def test_an_empty_video_is_reported_and_no_csv_is_written():
    with tempfile.TemporaryDirectory() as folder, \
            fake_environment(frames=[]) as env:      # opens, but 0 frames
        video = os.path.join(folder, "empty.mp4")
        output = os.path.join(folder, "out.csv")
        with open(video, "wb") as file:
            file.write(b"x")
        message = expect_error(env.demo.VideoAnalysisError,
                               env.demo.process_video, video, output)
        assert "No frames" in message
        assert env.capture.released is True
        assert not os.path.exists(output)


def test_a_frame_that_decodes_as_none_ends_the_video_safely():
    class NoneFrameCapture(FakeCapture):
        def read(self):
            return True, None                         # "success" but no image

    with tempfile.TemporaryDirectory() as folder, fake_environment() as env:
        env.demo.cv2.VideoCapture = lambda path: NoneFrameCapture([], True)
        video = os.path.join(folder, "clip.mp4")
        with open(video, "wb") as file:
            file.write(b"x")
        expect_error(env.demo.VideoAnalysisError, env.demo.process_video,
                     video, os.path.join(folder, "out.csv"))


def test_unwritable_csv_gives_a_friendly_error():
    with tempfile.TemporaryDirectory() as folder, \
            fake_environment([make_frame()]) as env:
        video = os.path.join(folder, "clip.mp4")
        with open(video, "wb") as file:
            file.write(b"x")

        def refuse(*args, **kwargs):
            raise PermissionError(13, "Permission denied")

        original_open = env.demo.open if hasattr(env.demo, "open") else None
        env.demo.open = refuse          # only affects video_analysis_demo
        try:
            message = expect_error(env.demo.VideoAnalysisError,
                                   env.demo.process_video, video,
                                   os.path.join(folder, "out.csv"))
        finally:
            if original_open is None:
                del env.demo.open
        assert "Could not write" in message
        assert env.capture.released is True


# ---- 7. Resources are released -------------------------------------------

def test_video_and_model_are_released_after_a_normal_run():
    run = run_video([make_frame() for _ in range(3)])
    assert run.env.capture.released is True
    assert run.env.poses[0].exited is True


def test_resources_are_released_when_processing_fails_half_way():
    with tempfile.TemporaryDirectory() as folder, fake_environment(
            [make_frame() for _ in range(5)],
            process_error=RuntimeError("boom"), process_error_at=2) as env:
        video = os.path.join(folder, "clip.mp4")
        output = os.path.join(folder, "out.csv")
        with open(video, "wb") as file:
            file.write(b"x")
        message = expect_error(RuntimeError, env.demo.process_video,
                               video, output)
        assert message == "boom"
        assert env.capture.released is True
        assert env.poses[0].exited is True
        assert not os.path.exists(output)            # no half-written CSV


# ---- 8. Command line (main) ----------------------------------------------

def test_main_success_prints_a_short_summary_and_saves_the_csv():
    frames = [make_frame() for _ in range(4)]
    code, out, err, files, env = run_main(
        lambda folder: [os.path.join(folder, "clip.mp4"),
                        os.path.join(folder, "result.csv")], frames)
    assert code == 0
    assert "Processed 4 frames" in out
    assert "result.csv" in out
    assert "result.csv" in files
    assert err == ""


def test_main_without_arguments_prints_usage_and_exits_with_code_2():
    with fake_environment() as env:
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                env.demo.main([])
        except SystemExit as stop:
            assert stop.code == 2
        else:
            raise AssertionError("expected SystemExit")
        assert "usage" in err.getvalue().lower()
        assert "VIDEO" in err.getvalue()


def test_main_help_explains_how_to_run_the_script():
    with fake_environment() as env:
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                env.demo.main(["--help"])
        except SystemExit as stop:
            assert stop.code == 0
        text = out.getvalue()
        assert "VIDEO" in text and "OUTPUT_CSV" in text
        assert "python video_analysis_demo.py" in text


def test_main_with_a_bad_path_prints_a_friendly_error_not_a_traceback():
    code, out, err, files, env = run_main(
        lambda folder: [os.path.join(folder, "typo.mp4")])
    assert code == 1
    assert err.startswith("Error: ")
    assert "not found" in err
    assert "Traceback" not in err + out
    assert env.capture_requests == []


def test_main_with_an_unopenable_video_returns_1():
    code, out, err, files, env = run_main(
        lambda folder: [os.path.join(folder, "clip.mp4"),
                        os.path.join(folder, "out.csv")], opened=False)
    assert code == 1
    assert "could not open" in err
    assert "out.csv" not in files
    assert env.capture.released is True


def test_main_default_output_name_comes_from_the_video_name():
    with fake_environment() as env:
        assert env.demo.default_output_path("a/b/my_pushups.mp4") == (
            "my_pushups_features.csv")

    # End to end: with no output argument the CSV lands in the current folder.
    with tempfile.TemporaryDirectory() as folder, \
            fake_environment([make_frame(), make_frame()]) as env:
        video = os.path.join(folder, "clip.mp4")
        with open(video, "wb") as file:
            file.write(b"x")
        previous = os.getcwd()
        os.chdir(folder)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                code = env.demo.main([video])
        finally:
            os.chdir(previous)
        assert code == 0
        assert "clip_features.csv" in os.listdir(folder)


def test_main_warns_when_nobody_was_detected_in_the_whole_video():
    code, out, err, files, env = run_main(
        lambda folder: [os.path.join(folder, "clip.mp4"),
                        os.path.join(folder, "out.csv")],
        frames=[make_frame(), make_frame()], per_frame_landmarks=[None, None])
    assert code == 0
    assert "Warning: nobody was detected" in out
    assert "out.csv" in files


def test_ctrl_c_stops_cleanly_without_a_csv():
    code, out, err, files, env = run_main(
        lambda folder: [os.path.join(folder, "clip.mp4"),
                        os.path.join(folder, "out.csv")],
        frames=[make_frame() for _ in range(3)],
        process_error=KeyboardInterrupt(), process_error_at=1)
    assert code == 130
    assert "Stopped by the user" in err
    assert "out.csv" not in files
    assert env.capture.released is True


# ---- 9. Concise output ----------------------------------------------------

def test_progress_output_is_short_and_has_no_per_frame_report():
    run = run_video([make_frame() for _ in range(250)])
    lines = run.text.splitlines()
    assert len(lines) == 2                           # after frames 100 and 200
    assert "Processed 100" in lines[0] and "Processed 200" in lines[1]
    for forbidden in ("Form Analysis", "Elbow Angle", "Visibility Check"):
        assert forbidden not in run.text


# ---- 10. Existing behaviour is preserved ---------------------------------

def test_print_landmark_report_prints_exactly_what_it_printed_before():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        returned = print_landmark_report("Person 1 Landmark Format Example",
                                         LANDMARK_EXAMPLE)
    assert output.getvalue().splitlines() == GOLDEN_REPORT_LINES
    assert isinstance(returned, tuple) and len(returned) == 2
    angles, result = returned
    assert list(angles) == ANGLE_NAMES
    assert result["form_status"] == "GOOD"


def test_print_landmark_report_keeps_its_parameters():
    parameters = inspect.signature(print_landmark_report).parameters
    assert list(parameters) == ["title", "landmark_dict", "image_width",
                                "image_height"]
    assert parameters["image_width"].default is None
    assert parameters["image_height"].default is None


def test_print_landmark_report_still_says_not_assessed_for_rejected_angles():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        print_landmark_report(
            "t", make_landmarks(left_wrist={"visibility": 0.491}))
    text = output.getvalue()
    assert "Elbow Angle: not assessed" in text
    assert ("  - Elbow angle not assessed: "
            "left_wrist visibility 0.491 is below 0.5") in text
    assert "Hip Angle: 178.9" in text


def test_analyze_landmarks_prints_nothing_and_matches_the_report():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        angles, result, notes = analyze_landmarks(
            LANDMARK_EXAMPLE, image_width=640, image_height=480)
    assert output.getvalue() == ""
    assert notes == []

    with contextlib.redirect_stdout(io.StringIO()):
        report_angles, report_result = print_landmark_report(
            "t", LANDMARK_EXAMPLE, image_width=640, image_height=480)
    assert angles == report_angles
    assert result == report_result


def test_analyze_landmarks_handles_no_landmarks_and_rejected_angles():
    angles, result, notes = analyze_landmarks({})
    assert [angles[name] for name in ANGLE_NAMES] == [None, None, None]
    assert result["form_status"] == "INCOMPLETE_DATA"
    assert result["movement_phase"] == "UNKNOWN"

    angles, result, notes = analyze_landmarks(
        make_landmarks(left_ankle={"visibility": 0.2}))
    assert angles["hip_angle"] is None
    assert angles["elbow_angle"] is not None
    assert len(notes) == 1 and notes[0].startswith("Hip angle not assessed")

    # An image size must still be given as a pair.
    expect_error(ValueError, analyze_landmarks, LANDMARK_EXAMPLE,
                 image_width=640)


def test_simulated_demo_output_is_unchanged():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        simulated_demo_main()
    lines = output.getvalue().splitlines()
    assert lines[0] == "Top Position"
    assert lines[-len(GOLDEN_REPORT_LINES):] == GOLDEN_REPORT_LINES
    for pose_name in pushup_angle_demo.POSES:
        assert pose_name in output.getvalue()


def test_single_frame_demo_still_works_and_leaves_frame_number_empty():
    with fake_environment([make_frame()]) as env:
        single_frame_demo = env.import_module("single_frame_demo")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            single_frame_demo.main()
        text = output.getvalue()
    assert "Real MediaPipe Frame" in text
    assert "Feature row (one frame):" in text
    lines = text.splitlines()
    start = lines.index("Feature row (one frame):")
    assert [line.split()[0] for line in lines[start + 1:start + 12]] == (
        EXPECTED_COLUMNS)
    assert lines[start + 1].rstrip().endswith("None")   # frame_number


# ---- Simple runner so `python test_video_analysis_demo.py` works ---------

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
