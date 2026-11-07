"""
test_video_repetition_integration.py

End-to-end tests of the video pipeline with Person 1's CLASS-BASED detector:

    fake video -> BGR frame -> PoseDetector.detect_pose() (Person 1's REAL
    code, which converts BGR -> RGB itself and returns rounded pixel
    coordinates) -> pose_adapter.pose_data_to_landmarks() ->
    analyze_landmarks() without an image size (pixels are already final) ->
    visibility check (0.5) -> angles -> form analysis -> feature row ->
    RepetitionCounter (Person 2's REAL counter) -> CSV + summary

Only the outside world is faked: a tiny "cv2" and a tiny "mediapipe" are put
in sys.modules while a module is imported (the real ones are restored
afterwards). The fake MediaPipe `Pose` has the lifecycle of the real one
(process(), close(), `with`), so Person 1's detector can start, run and close
it exactly as it would for real. No webcam, video file or MediaPipe install is
needed.

This file also provides fake_environment(), reused by test_pose_adapter.py.

Run either way (from person2/src):
    python test_video_repetition_integration.py
    pytest -v test_video_repetition_integration.py
"""

from __future__ import annotations

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
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

import pose_adapter
import pushup_angle_demo
import repetition_counter
from angle_calculator import calculate_angle
from feature_extractor import extract_features
from pushup_angle_demo import LANDMARK_EXAMPLE, analyze_landmarks

ANGLE_NAMES = ["elbow_angle", "hip_angle", "shoulder_angle"]

# The 11 columns, written out here on purpose (not imported) so the test fails
# if the schema ever changes by accident.
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

LANDMARK_NAME_ORDER = [
    "left_shoulder", "left_elbow", "left_wrist", "left_hip", "left_knee",
    "left_ankle", "right_shoulder", "right_elbow", "right_wrist", "right_hip",
    "right_knee", "right_ankle",
]

WIDTH, HEIGHT = 640, 480

# Elbow angles (degrees) that give each movement phase.
TOP_ANGLE, MIDDLE_ANGLE, BOTTOM_ANGLE = 175, 120, 80

Landmarks = Dict[str, Dict[str, float]]


# ===========================================================================
# Fakes: a video and MediaPipe (Person 1's real detector runs on top)
# ===========================================================================

class FakeCapture:
    """Stands in for cv2.VideoCapture: hands out the frames, then ends."""

    def __init__(self, frames: Sequence[Any] = (), opened: bool = True):
        self.frames = list(frames)
        self.opened = opened
        self.released = False
        self.release_calls = 0
        self.position = 0

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        if self.position < len(self.frames):
            frame = self.frames[self.position]
            self.position += 1
            return True, frame
        return False, None

    def get(self, prop: int) -> int:
        return len(self.frames)

    def release(self) -> None:
        self.released = True
        self.release_calls += 1


@contextlib.contextmanager
def fake_environment(
    frames: Sequence[Any] = (),
    results_per_frame: Optional[Sequence[Any]] = None,
    *,
    opened: bool = True,
    process_error: Optional[BaseException] = None,
    process_error_at: int = 0,
    close_error: Optional[BaseException] = None,
    person1_path: Optional[str] = None,
    import_demo: bool = True,
) -> Iterator[types.SimpleNamespace]:
    """
    Fake cv2 + fake mediapipe, with a freshly imported video_analysis_demo.

    frames             the fake video: BGR arrays (see make_frame)
    results_per_frame  what the fake MediaPipe "detects" in each frame: a
                       landmark dictionary like LANDMARK_EXAMPLE (0-1 values),
                       None for "nobody in the frame", or a ready-made results
                       object
    opened             False makes the fake video refuse to open
    process_error      exception raised by the fake pose at frame
                       `process_error_at`
    close_error        exception raised when the fake pose is closed
    person1_path       use this file instead of person1/src/pose_detector.py
    import_demo        False: only install the fakes (no demo import)

    The yielded namespace has: demo, capture, poses (every fake MediaPipe Pose
    created, i.e. every model started), cv2_calls (colour codes given to
    cvtColor), capture_requests, make_results(), import_module(), PoseLandmark.
    """
    frames = list(frames)
    if results_per_frame is None:
        results_per_frame = [LANDMARK_EXAMPLE] * len(frames)
    results_per_frame = list(results_per_frame)      # kept: env.results

    pose_landmark = enum.IntEnum(
        "PoseLandmark",
        {name.upper(): index for index, name in enumerate(LANDMARK_NAME_ORDER)})

    def make_results(landmarks):
        """What MediaPipe's pose.process() returns for one frame."""
        if landmarks is None:
            return types.SimpleNamespace(pose_landmarks=None)
        if not isinstance(landmarks, dict):
            return landmarks                       # already a results object
        listed = [types.SimpleNamespace(x=0.0, y=0.0, visibility=0.0)
                  for _ in LANDMARK_NAME_ORDER]
        for name, values in landmarks.items():
            listed[LANDMARK_NAME_ORDER.index(name)] = types.SimpleNamespace(
                **values)
        return types.SimpleNamespace(
            pose_landmarks=types.SimpleNamespace(landmark=listed))

    env = types.SimpleNamespace(
        capture=FakeCapture(frames, opened), poses=[], cv2_calls=[],
        results=results_per_frame, capture_requests=[], demo=None,
        make_results=make_results,
        PoseLandmark=pose_landmark, import_module=importlib.import_module)

    class FakePose:
        """Same lifecycle as mediapipe's Pose: process(), close(), `with`."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            self.close_calls = 0
            self.received: List[Any] = []        # every array given to process()
            env.poses.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()                         # like the real class
            return False

        def close(self):
            self.closed = True
            self.close_calls += 1
            if close_error is not None:
                raise close_error

        def process(self, rgb_frame):
            index = len(self.received)
            self.received.append(rgb_frame)
            if process_error is not None and index == process_error_at:
                raise process_error
            return make_results(results_per_frame[index])

    def fake_video_capture(path):
        env.capture_requests.append(path)
        return env.capture

    def fake_cvt_color(image, code):
        env.cv2_calls.append(code)
        return image[..., ::-1].copy()           # BGR <-> RGB: reverse channels

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
    saved_path = pose_adapter.DEFAULT_PERSON1_DETECTOR_PATH
    try:
        for name in module_names:
            sys.modules.pop(name, None)           # force fresh imports
        sys.modules["cv2"] = fake_cv2
        sys.modules["mediapipe"] = fake_mediapipe
        if person1_path is not None:
            pose_adapter.DEFAULT_PERSON1_DETECTOR_PATH = person1_path
        if import_demo:
            env.demo = importlib.import_module("video_analysis_demo")
        yield env
    finally:
        pose_adapter.DEFAULT_PERSON1_DETECTOR_PATH = saved_path
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class FakeDetector:
    """
    A fake of Person 1's PoseDetector with its real lifecycle: detect_pose(),
    close() and `with`. Each detect_pose() call returns the next pose_data.
    """

    def __init__(self, results: Sequence[Dict[str, Any]] = ()):
        self.results = list(results)
        self.calls: List[Tuple[Any, bool]] = []     # (frame, draw) per call
        self.closed = False

    def detect_pose(self, frame, draw=True):
        self.calls.append((frame, draw))
        return frame, self.results[len(self.calls) - 1]

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


# ===========================================================================
# Helpers (their names do not start with "test_")
# ===========================================================================

def make_frame(height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    """A fake BGR frame (height, width, 3) with distinct B, G, R values."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0], frame[..., 1], frame[..., 2] = 10, 20, 30
    return frame


def make_landmarks(**changes) -> Landmarks:
    """Copy of the good example landmarks; each change updates one landmark."""
    landmarks = copy.deepcopy(LANDMARK_EXAMPLE)
    for name, change in changes.items():
        landmarks[name].update(change)
    return landmarks


def landmarks_for_elbow_angle(angle: float, width: int = WIDTH,
                              height: int = HEIGHT) -> Landmarks:
    """
    Good example landmarks (0-1 values, like MediaPipe's) whose ELBOW angle is
    about `angle` degrees in a width x height frame. Person 1's detector rounds
    them to whole pixels, which moves the angle by under a degree: far less
    than the gaps between the phase thresholds used by the tests.
    """
    shoulder, elbow = (300.0, 200.0), (400.0, 200.0)
    direction = math.radians(180.0 - angle)
    wrist = (elbow[0] + 100.0 * math.cos(direction),
             elbow[1] + 100.0 * math.sin(direction))
    landmarks = copy.deepcopy(LANDMARK_EXAMPLE)
    for name, (x, y) in (("left_shoulder", shoulder), ("left_elbow", elbow),
                         ("left_wrist", wrist)):
        landmarks[name] = {"x": x / width, "y": y / height, "visibility": 0.95}
    return landmarks


def whole_pixel_points(landmarks: Landmarks, width: int,
                       height: int) -> Dict[str, Tuple[int, int]]:
    """
    {name: (x_px, y_px)}: the pixel positions exactly as Person 1's
    PoseDetector reports them (rounded to whole pixels, kept inside the image).
    """
    return {name: (min(width - 1, max(0, int(round(lm["x"] * width)))),
                   min(height - 1, max(0, int(round(lm["y"] * height)))))
            for name, lm in landmarks.items()}


def pose_data_for(landmarks: Optional[Landmarks], width: int = WIDTH,
                  height: int = HEIGHT) -> Dict[str, Any]:
    """What Person 1's detect_pose() returns as pose_data for these landmarks."""
    if landmarks is None:
        return {"pose_detected": False,
                "landmarks": {n: None for n in LANDMARK_NAME_ORDER},
                "visibility": {n: None for n in LANDMARK_NAME_ORDER}}
    return {"pose_detected": True,
            "landmarks": whole_pixel_points(landmarks, width, height),
            "visibility": {n: float(lm["visibility"])
                           for n, lm in landmarks.items()}}


def expected_angles_for_pixels(points: Dict[str, Tuple[float, float]]
                               ) -> Dict[str, float]:
    """The three angles worked out independently from pixel positions."""
    return {
        "elbow_angle": calculate_angle(points["left_shoulder"],
                                       points["left_elbow"],
                                       points["left_wrist"]),
        "hip_angle": calculate_angle(points["left_shoulder"],
                                     points["left_hip"],
                                     points["left_ankle"]),
        "shoulder_angle": calculate_angle(points["left_elbow"],
                                          points["left_shoulder"],
                                          points["left_hip"]),
    }


def close_to(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, abs_tol=1e-9)


def expect_error(error_type, function: Callable, *args, **kwargs) -> str:
    """Check that function(...) raises error_type; return its message."""
    try:
        function(*args, **kwargs)
    except error_type as error:
        return str(error)
    raise AssertionError(
        f"expected {error_type.__name__}, but no error was raised")


def write_video_placeholder(folder: str, name: str = "clip.mp4") -> str:
    """The fake VideoCapture never reads it, but the path must exist."""
    path = os.path.join(folder, name)
    with open(path, "wb") as file:
        file.write(b"not a real video")
    return path


def read_csv_file(path: str):
    """Return (header, records): the header list and one dict per data row."""
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)
        return header, [dict(zip(header, line)) for line in reader]


def run_video(items: Sequence[Any], frame_size=(HEIGHT, WIDTH),
              **fake_options) -> types.SimpleNamespace:
    """
    Run video_analysis_demo.process_video() on a fake video.

    Each item is a number (an elbow angle in degrees), a landmark dictionary,
    or None (nobody in the frame). Returns a namespace with: summary, header,
    records (the CSV as text), text (printed output), counter_updates (the
    phase given to the counter for each frame), analyzer_calls (keyword
    arguments given to analyze_landmarks per frame) and env.
    """
    results = [landmarks_for_elbow_angle(item)
               if isinstance(item, (int, float)) else item for item in items]
    frames = [make_frame(*frame_size) for _ in results]
    with tempfile.TemporaryDirectory() as folder, \
            fake_environment(frames, results, **fake_options) as env:
        video = write_video_placeholder(folder)
        output = os.path.join(folder, "out.csv")

        updates: List[str] = []
        real_counter = env.demo.RepetitionCounter

        class RecordingCounter(real_counter):
            def update(self, phase):
                updates.append(phase)
                return super().update(phase)

        env.demo.RepetitionCounter = RecordingCounter

        analyzer_calls: List[Dict[str, Any]] = []
        real_analyze = env.demo.analyze_landmarks

        def recording_analyze(landmarks, **kwargs):
            analyzer_calls.append(dict(kwargs))
            return real_analyze(landmarks, **kwargs)

        env.demo.analyze_landmarks = recording_analyze

        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            summary = env.demo.process_video(video, output)
        header, records = read_csv_file(output)
        return types.SimpleNamespace(
            summary=summary, header=header, records=records,
            text=printed.getvalue(), counter_updates=updates,
            analyzer_calls=analyzer_calls, env=env)


def count_for(angles: Sequence[Optional[float]]) -> int:
    return run_video(angles).summary.repetitions


# ===========================================================================
# 1. Repetition counting through the whole pipeline (Person 2's real counter)
# ===========================================================================

def test_the_real_repetition_counter_is_used():
    with fake_environment() as env:
        assert env.demo.RepetitionCounter is repetition_counter.RepetitionCounter


def test_one_complete_push_up_counts_once():
    run = run_video([TOP_ANGLE, MIDDLE_ANGLE, BOTTOM_ANGLE, MIDDLE_ANGLE,
                     TOP_ANGLE])
    assert [r["rule_movement_phase"] for r in run.records] == [
        "TOP", "MIDDLE", "BOTTOM", "MIDDLE", "TOP"]
    assert run.summary.repetitions == 1


def test_several_complete_push_ups():
    one = [MIDDLE_ANGLE, BOTTOM_ANGLE, MIDDLE_ANGLE, TOP_ANGLE]
    assert count_for([TOP_ANGLE] + one * 3) == 3


def test_repeated_top_frames_do_not_increase_the_count():
    assert count_for([TOP_ANGLE] * 12) == 0
    assert count_for([TOP_ANGLE, BOTTOM_ANGLE, TOP_ANGLE]
                     + [TOP_ANGLE] * 12) == 1


def test_incomplete_movements_do_not_count():
    assert count_for([TOP_ANGLE, MIDDLE_ANGLE, TOP_ANGLE]) == 0     # not deep
    assert count_for([TOP_ANGLE, MIDDLE_ANGLE, BOTTOM_ANGLE,
                      BOTTOM_ANGLE]) == 0                           # no return
    assert count_for([TOP_ANGLE, MIDDLE_ANGLE, BOTTOM_ANGLE,
                      MIDDLE_ANGLE]) == 0                           # no top
    assert count_for([BOTTOM_ANGLE, MIDDLE_ANGLE, TOP_ANGLE]) == 0  # no start


def test_a_video_that_starts_mid_movement_waits_for_a_first_top():
    mid_start = [MIDDLE_ANGLE, BOTTOM_ANGLE, MIDDLE_ANGLE, TOP_ANGLE]
    assert count_for(mid_start) == 0
    assert count_for(mid_start + [MIDDLE_ANGLE, BOTTOM_ANGLE, TOP_ANGLE]) == 1


def test_unknown_frames_leave_the_counter_state_and_total_unchanged():
    # None = nobody detected, which gives the UNKNOWN phase.
    assert count_for([TOP_ANGLE, BOTTOM_ANGLE, None, None, TOP_ANGLE]) == 1
    assert count_for([None, TOP_ANGLE, None, BOTTOM_ANGLE, None,
                      TOP_ANGLE, None]) == 1
    assert count_for([TOP_ANGLE, BOTTOM_ANGLE, TOP_ANGLE] + [None] * 5) == 1


def test_there_is_no_unknown_timeout_even_after_a_very_long_gap():
    gap = [None] * 200
    assert count_for([TOP_ANGLE, BOTTOM_ANGLE] + gap + [TOP_ANGLE]) == 1
    assert count_for([TOP_ANGLE] + gap + [BOTTOM_ANGLE, TOP_ANGLE]) == 1


def test_a_hidden_wrist_makes_the_phase_unknown_without_resetting():
    hidden_wrist = make_landmarks(left_wrist={"visibility": 0.3})
    run = run_video([TOP_ANGLE, BOTTOM_ANGLE, hidden_wrist, hidden_wrist,
                     TOP_ANGLE])
    assert [r["rule_movement_phase"] for r in run.records] == [
        "TOP", "BOTTOM", "UNKNOWN", "UNKNOWN", "TOP"]
    assert run.summary.repetitions == 1


def test_the_counter_is_updated_exactly_once_per_decoded_frame():
    angles = [TOP_ANGLE, BOTTOM_ANGLE, TOP_ANGLE, None, None, TOP_ANGLE]
    run = run_video(angles)
    assert len(run.counter_updates) == len(angles) == len(run.records)
    assert run.counter_updates == [r["rule_movement_phase"]
                                   for r in run.records]
    assert run.counter_updates.count("UNKNOWN") == 2


def test_the_repetition_count_is_not_a_csv_column():
    run = run_video([TOP_ANGLE, BOTTOM_ANGLE, TOP_ANGLE])
    assert run.summary.repetitions == 1
    assert run.header == EXPECTED_COLUMNS
    assert not any("repetition" in name for name in run.header)


# ===========================================================================
# 2. Rows, frame numbers and the CSV
# ===========================================================================

def test_one_row_per_decoded_frame_numbered_from_zero():
    angles = [TOP_ANGLE, None, BOTTOM_ANGLE, None, None, TOP_ANGLE]
    run = run_video(angles)
    assert len(run.records) == len(angles) == run.summary.frames_processed
    assert [r["frame_number"] for r in run.records] == [
        str(i) for i in range(len(angles))]
    assert len(run.env.poses[0].received) == len(angles)    # every frame seen


def test_a_one_frame_video_gives_one_row_numbered_zero():
    assert [r["frame_number"] for r in run_video([TOP_ANGLE]).records] == ["0"]


def test_csv_has_exactly_the_11_agreed_columns_in_order():
    run = run_video([TOP_ANGLE, None, BOTTOM_ANGLE])
    assert run.header == EXPECTED_COLUMNS and len(run.header) == 11
    for record in run.records:
        assert list(record) == EXPECTED_COLUMNS


def test_the_demo_uses_the_extractors_column_list():
    with fake_environment() as env:
        assert env.demo.FEATURE_COLUMNS == EXPECTED_COLUMNS


def test_a_frame_with_no_person_still_gives_a_row_of_empty_angles():
    run = run_video([TOP_ANGLE, None])
    row = run.records[1]
    assert row["frame_number"] == "1"
    for name in ANGLE_NAMES:
        assert row[name] == ""                       # None -> empty CSV cell
        assert row[f"{name}_available"] == "False"
    assert row["all_angles_available"] == "False"
    assert row["rule_movement_phase"] == "UNKNOWN"
    assert row["rule_form_status"] == "INCOMPLETE_DATA"
    assert "Elbow angle missing" in row["rule_feedback"]
    assert run.summary.frames_with_person == 1


def test_a_video_where_nobody_is_detected_still_gives_every_row():
    run = run_video([None, None, None])
    assert len(run.records) == 3
    assert run.summary.frames_with_person == 0 and run.summary.repetitions == 0


def test_unavailable_landmarks_make_only_the_dependent_angles_unavailable():
    cases = [
        (make_landmarks(left_wrist={"visibility": 0.491}), (False, True, True)),
        (make_landmarks(left_ankle={"visibility": 0.137}), (True, False, True)),
        (make_landmarks(left_hip={"visibility": 0.2}), (True, False, False)),
        (make_landmarks(left_shoulder={"visibility": 0.2}),
         (False, False, False)),
    ]
    run = run_video([landmarks for landmarks, _ in cases])
    for row, (_, flags) in zip(run.records, cases):
        got = tuple(row[f"{name}_available"] == "True" for name in ANGLE_NAMES)
        assert got == flags
        for name, available in zip(ANGLE_NAMES, flags):
            assert (row[name] != "") == available, name


def test_landmarks_missing_from_the_detector_result_are_unavailable():
    # MediaPipe returned fewer landmarks than the detector expects, so Person
    # 1's detector reports them as None. The row must still be produced.
    with fake_environment() as env:
        short = types.SimpleNamespace(
            pose_landmarks=types.SimpleNamespace(landmark=[
                types.SimpleNamespace(x=0.5, y=0.5, visibility=0.9)] * 3))
    run = run_video([short])
    row = run.records[0]
    assert len(run.records) == 1
    assert row["all_angles_available"] == "False"
    assert row["rule_form_status"] == "INCOMPLETE_DATA"


# ===========================================================================
# 3. The real Person 1 interface: pixels in, scaled exactly once
# ===========================================================================

def test_the_demo_modules_do_not_load_person_1s_file_or_start_a_model_on_import():
    with tempfile.TemporaryDirectory() as folder:
        trap = os.path.join(folder, "pose_detector.py")
        with open(trap, "w", encoding="utf-8") as file:
            file.write("raise RuntimeError('Person 1 file was executed')\n")
        with fake_environment(person1_path=trap) as env:
            env.import_module("single_frame_demo")      # imports fine as well
            assert env.poses == []                       # no model was created


def test_the_detector_is_person_1s_class_loaded_from_its_path():
    run = run_video([TOP_ANGLE])
    # Person 1's class ran (it created exactly one fake MediaPipe Pose) and
    # Person 2's pose_detector module was never imported.
    assert len(run.env.poses) == 1
    assert "pose_detector" not in sys.modules
    assert pose_adapter.PERSON1_MODULE_NAME not in sys.modules


def test_low_visibility_coordinates_reach_person_2s_visibility_check():
    # Person 1's detector would drop the coordinates of a landmark below its
    # visibility_threshold. process_video() creates it with 0.0, so the
    # landmark is kept and Person 2's 0.5 rule is what rejects it.
    run = run_video([make_landmarks(left_wrist={"visibility": 0.2})])
    assert run.records[0]["elbow_angle_available"] == "False"
    assert run.records[0]["hip_angle_available"] == "True"
    assert len(run.env.poses) == 1


def test_the_half_threshold_decides_visibility_exactly_as_before():
    assert pushup_angle_demo.VISIBILITY_THRESHOLD == 0.5
    cases = [(0.491, False), (0.499, False), (0.5, True), (0.9, True)]
    run = run_video([make_landmarks(left_wrist={"visibility": v})
                     for v, _ in cases])
    for row, (_, accepted) in zip(run.records, cases):
        assert (row["elbow_angle_available"] == "True") == accepted


def test_exactly_one_bgr_to_rgb_conversion_per_frame():
    run = run_video([TOP_ANGLE, BOTTOM_ANGLE, None])
    assert run.env.cv2_calls == [4, 4, 4]          # one cv2.COLOR_BGR2RGB each
    received = run.env.poses[0].received[0]
    assert tuple(received[0, 0]) == (30, 20, 10)   # MediaPipe got (R, G, B)


def test_pixels_are_not_scaled_a_second_time():
    seen: List[Tuple[tuple, tuple, tuple]] = []
    real_calculate = pushup_angle_demo.calculate_angle

    def recording(a, b, c):
        seen.append((tuple(a), tuple(b), tuple(c)))
        return real_calculate(a, b, c)

    pushup_angle_demo.calculate_angle = recording
    try:
        run = run_video([MIDDLE_ANGLE])
    finally:
        pushup_angle_demo.calculate_angle = real_calculate

    # No image size reached the angle code ...
    assert run.analyzer_calls == [{}]
    # ... so the angle calculation received exactly the pixels Person 1's
    # detector reported (whole pixels), and nothing is bigger than the frame.
    pixels = whole_pixel_points(landmarks_for_elbow_angle(MIDDLE_ANGLE),
                                WIDTH, HEIGHT)
    elbow_call = seen[0]                            # (shoulder, elbow, wrist)
    for got, name in zip(elbow_call, ("left_shoulder", "left_elbow",
                                      "left_wrist")):
        assert got == tuple(map(float, pixels[name])), name
    assert all(0 <= value < max(WIDTH, HEIGHT)
               for call in seen for point in call for value in point)
    expected = expected_angles_for_pixels(pixels)
    for name in ANGLE_NAMES:
        assert close_to(float(run.records[0][name]), expected[name]), name


def test_the_pixels_come_from_each_frames_own_width_and_height():
    landmarks = landmarks_for_elbow_angle(MIDDLE_ANGLE)
    for height, width in ((480, 640), (720, 1280), (640, 480), (500, 500)):
        run = run_video([landmarks], frame_size=(height, width))
        pixels = whole_pixel_points(landmarks, width, height)
        expected = expected_angles_for_pixels(pixels)
        for name in ANGLE_NAMES:
            assert close_to(float(run.records[0][name]), expected[name]), (
                name, width, height)


def test_angles_and_form_analysis_are_computed_once_per_frame():
    calculations: List[int] = []
    analyses: List[int] = []
    real_calculate = pushup_angle_demo.calculate_angle
    real_analyze = pushup_angle_demo.analyze_push_up_form

    pushup_angle_demo.calculate_angle = (
        lambda a, b, c: calculations.append(1) or real_calculate(a, b, c))
    pushup_angle_demo.analyze_push_up_form = (
        lambda **kw: analyses.append(1) or real_analyze(**kw))
    try:
        run_video([TOP_ANGLE, None, BOTTOM_ANGLE, TOP_ANGLE])   # 3 with person
    finally:
        pushup_angle_demo.calculate_angle = real_calculate
        pushup_angle_demo.analyze_push_up_form = real_analyze
    assert len(analyses) == 4                  # one analysis per frame
    assert len(calculations) == 3 * 3          # 3 angles x 3 frames with a person


def test_the_frame_is_not_drawn_on_or_changed():
    frame = make_frame()
    original = frame.copy()
    with fake_environment([frame], [LANDMARK_EXAMPLE]) as env, \
            tempfile.TemporaryDirectory() as folder:
        with contextlib.redirect_stdout(io.StringIO()):
            env.demo.process_video(write_video_placeholder(folder),
                                   os.path.join(folder, "o.csv"))
    # Drawing would need MediaPipe's drawing utilities, which the fake does
    # not have, and would change the pixels. Neither happened.
    assert np.array_equal(frame, original)


# ===========================================================================
# 4. analyze_frame(frame, frame_number, pose): the supported interface
# ===========================================================================

def test_analyze_frame_keeps_its_public_signature():
    with fake_environment() as env:
        assert list(inspect.signature(env.demo.analyze_frame).parameters) == [
            "frame", "frame_number", "pose"]


def test_analyze_frame_with_a_class_based_detector():
    frame = make_frame()
    landmarks = make_landmarks(left_ankle={"visibility": 0.137})   # hip rejected
    data = pose_data_for(landmarks)
    detector = FakeDetector([data, data])
    with fake_environment([frame], [None]) as env:
        by_position = env.demo.analyze_frame(frame, 7, detector)
        by_keyword = env.demo.analyze_frame(frame=frame, frame_number=7,
                                            pose=detector)
        conversions = list(env.cv2_calls)
    row, person_found = by_position
    assert by_position == by_keyword and person_found is True

    # Independent route: reshape -> analyze (no size) -> extract.
    expected_landmarks = pose_adapter.pose_data_to_landmarks(data)
    angles, analysis, _ = analyze_landmarks(expected_landmarks)
    assert row == extract_features(angles["elbow_angle"], angles["hip_angle"],
                                   angles["shoulder_angle"], analysis,
                                   frame_number=7)
    assert row["hip_angle"] is None and row["hip_angle_available"] is False
    assert list(row) == EXPECTED_COLUMNS
    # The detector got the ORIGINAL frame, with drawing switched off, and this
    # path did not convert the colours (the detector does that itself).
    assert all(call[0] is frame and call[1] is False
               for call in detector.calls)
    assert conversions == []


def test_analyze_frame_with_a_class_based_detector_and_nobody_in_view():
    frame = make_frame()
    detector = FakeDetector([pose_data_for(None)])
    with fake_environment() as env:
        row, person_found = env.demo.analyze_frame(frame, 0, detector)
    assert person_found is False
    assert [row[name] for name in ANGLE_NAMES] == [None, None, None]
    assert row["rule_movement_phase"] == "UNKNOWN"
    assert row["rule_form_status"] == "INCOMPLETE_DATA"


def test_analyze_frame_still_accepts_a_mediapipe_compatible_pose_object():
    # The original contract: `pose` only needs process(rgb_frame). The colours
    # are converted once here and the 0-1 landmarks are scaled once, with the
    # real frame size, exactly as before.
    class MinimalPose:
        def __init__(self, results):
            self.results = results
            self.received: List[Any] = []

        def process(self, rgb_frame):
            self.received.append(rgb_frame)
            return self.results

    frame = make_frame(height=480, width=640)
    original = frame.copy()
    landmarks = landmarks_for_elbow_angle(130)
    with fake_environment([frame], [None]) as env:
        pose = MinimalPose(env.make_results(landmarks))
        sizes: List[Dict[str, Any]] = []
        real_analyze = env.demo.analyze_landmarks
        env.demo.analyze_landmarks = lambda lm, **kw: (
            sizes.append(dict(kw)) or real_analyze(lm, **kw))
        by_position = env.demo.analyze_frame(frame, 4, pose)
        by_keyword = env.demo.analyze_frame(frame=frame, frame_number=4,
                                            pose=pose)
        conversions = list(env.cv2_calls)
    row, person_found = by_position
    assert by_position == by_keyword
    assert person_found is True and row["frame_number"] == 4
    assert list(row) == EXPECTED_COLUMNS
    assert close_to(row["elbow_angle"], 130)        # unrounded: 0-1 x real size
    assert conversions == [4, 4]                    # one conversion per call
    assert sizes == [{"image_width": 640, "image_height": 480}] * 2
    assert tuple(pose.received[0][0, 0]) == (30, 20, 10)       # RGB order
    assert np.array_equal(frame, original)


def test_an_object_with_neither_method_is_rejected_clearly():
    with fake_environment() as env:
        message = expect_error(TypeError, env.demo.analyze_frame,
                               make_frame(), 0, object())
    assert "detect_pose" in message and "process" in message


def test_a_class_based_detector_wins_if_an_object_offers_both_methods():
    class Both(FakeDetector):
        def process(self, rgb_frame):
            raise AssertionError("process() must not be used")

    detector = Both([pose_data_for(LANDMARK_EXAMPLE)])
    with fake_environment() as env:
        row, person_found = env.demo.analyze_frame(make_frame(), 0, detector)
        assert env.cv2_calls == []
    assert person_found is True


# ===========================================================================
# 5. Cleanup of the video and of the detector
# ===========================================================================

def test_one_session_per_video_in_video_mode_and_everything_is_released():
    run = run_video([TOP_ANGLE] * 4)
    assert len(run.env.poses) == 1                       # not one per frame
    pose = run.env.poses[0]
    assert pose.kwargs["static_image_mode"] is False     # tracking mode
    assert pose.kwargs["min_detection_confidence"] == 0.5
    assert pose.kwargs["min_tracking_confidence"] == 0.5
    assert pose.closed and pose.close_calls == 1
    assert run.env.capture.released is True
    assert len(pose.received) == 4


def test_everything_is_released_when_processing_fails_half_way():
    with tempfile.TemporaryDirectory() as folder, fake_environment(
            [make_frame() for _ in range(5)], [LANDMARK_EXAMPLE] * 5,
            process_error=RuntimeError("boom"), process_error_at=2) as env:
        video = write_video_placeholder(folder)
        output = os.path.join(folder, "out.csv")
        assert expect_error(RuntimeError, env.demo.process_video,
                            video, output) == "boom"          # not hidden
        assert env.poses[0].closed is True
        assert env.capture.released is True
        assert not os.path.exists(output)                     # no half CSV


def test_the_video_is_released_even_if_closing_the_detector_fails():
    with tempfile.TemporaryDirectory() as folder, fake_environment(
            [make_frame()], [LANDMARK_EXAMPLE],
            close_error=RuntimeError("close failed")) as env:
        video = write_video_placeholder(folder)
        output = os.path.join(folder, "out.csv")
        assert expect_error(RuntimeError, env.demo.process_video,
                            video, output) == "close failed"  # not hidden
        assert env.capture.released is True
        assert not os.path.exists(output)


def test_a_video_that_cannot_be_opened_is_reported_and_released():
    with tempfile.TemporaryDirectory() as folder, \
            fake_environment(opened=False) as env:
        output = os.path.join(folder, "out.csv")
        message = expect_error(env.demo.VideoAnalysisError,
                               env.demo.process_video,
                               write_video_placeholder(folder), output)
        assert "could not open" in message
        assert env.capture.released is True
        assert env.poses == []                    # no model was ever started
        assert not os.path.exists(output)


def test_an_empty_video_is_reported_and_everything_is_released():
    with tempfile.TemporaryDirectory() as folder, \
            fake_environment(frames=[]) as env:
        output = os.path.join(folder, "out.csv")
        message = expect_error(env.demo.VideoAnalysisError,
                               env.demo.process_video,
                               write_video_placeholder(folder), output)
        assert "No frames" in message
        assert env.capture.released is True
        assert env.poses[0].closed is True        # started, then closed
        assert not os.path.exists(output)


def test_a_missing_person_1_file_is_a_friendly_error_and_the_video_is_released():
    missing = os.path.join(tempfile.gettempdir(), "no_such_folder",
                           "pose_detector.py")
    with tempfile.TemporaryDirectory() as folder, fake_environment(
            [make_frame()], [LANDMARK_EXAMPLE], person1_path=missing) as env:
        output = os.path.join(folder, "out.csv")
        message = expect_error(env.demo.VideoAnalysisError,
                               env.demo.process_video,
                               write_video_placeholder(folder), output)
        assert "not found" in message
        assert env.capture.released is True and env.poses == []
        assert not os.path.exists(output)


def test_ctrl_c_stops_cleanly_without_a_csv_and_releases_everything():
    with tempfile.TemporaryDirectory() as folder, fake_environment(
            [make_frame() for _ in range(3)], [LANDMARK_EXAMPLE] * 3,
            process_error=KeyboardInterrupt(), process_error_at=1) as env:
        video = write_video_placeholder(folder)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = env.demo.main([video, os.path.join(folder, "o.csv")])
        assert code == 130 and "Stopped by the user" in err.getvalue()
        assert env.poses[0].closed and env.capture.released
        assert not os.path.exists(os.path.join(folder, "o.csv"))


# ===========================================================================
# 7. A fresh counter for each video
# ===========================================================================

def run_videos_in_one_session(*videos: Sequence[Any],
                              **fake_options) -> types.SimpleNamespace:
    """
    Run process_video() once per video inside ONE imported video_analysis_demo
    module, so that any state wrongly kept between videos would show up.
    Each video is a list of items as in run_video(). A video that raises is
    recorded instead of stopping the test. Returns a namespace with:
    summaries (None for a video that failed), errors, counters (every
    RepetitionCounter that was created) and env.
    """
    with tempfile.TemporaryDirectory() as folder, \
            fake_environment([], [], **fake_options) as env:
        counters: List[Any] = []
        real_counter = env.demo.RepetitionCounter

        class TrackedCounter(real_counter):
            def __init__(self):
                super().__init__()
                counters.append(self)

        env.demo.RepetitionCounter = TrackedCounter

        summaries: List[Any] = []
        errors: List[Optional[BaseException]] = []
        for index, items in enumerate(videos):
            results = [landmarks_for_elbow_angle(item)
                       if isinstance(item, (int, float)) else item
                       for item in items]
            env.results[:] = results                   # what MediaPipe "sees"
            env.capture.frames[:] = [make_frame() for _ in results]
            env.capture.position = 0
            env.capture.released = False
            video = write_video_placeholder(folder, f"clip{index}.mp4")
            output = os.path.join(folder, f"out{index}.csv")
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    summaries.append(env.demo.process_video(video, output))
                errors.append(None)
            except BaseException as error:             # recorded, not hidden
                summaries.append(None)
                errors.append(error)
        return types.SimpleNamespace(summaries=summaries, errors=errors,
                                     counters=counters, env=env)


def test_every_video_gets_its_own_new_counter():
    run = run_videos_in_one_session(
        [TOP_ANGLE, BOTTOM_ANGLE, TOP_ANGLE],
        [TOP_ANGLE, BOTTOM_ANGLE, TOP_ANGLE],
        [TOP_ANGLE, BOTTOM_ANGLE, TOP_ANGLE])
    assert len(run.counters) == 3
    assert len({id(counter) for counter in run.counters}) == 3   # all different
    assert all(counter.count == 1 for counter in run.counters)


def test_the_total_of_one_video_is_not_added_to_the_next():
    run = run_videos_in_one_session(
        [TOP_ANGLE, BOTTOM_ANGLE, TOP_ANGLE, BOTTOM_ANGLE, TOP_ANGLE],   # 2
        [TOP_ANGLE, BOTTOM_ANGLE, TOP_ANGLE],                            # 1
        [TOP_ANGLE, TOP_ANGLE])                                          # 0
    assert [s.repetitions for s in run.summaries] == [2, 1, 0]


def test_an_unfinished_repetition_does_not_leak_into_the_next_video():
    # Video 1 ends at the bottom of a push-up. If its counter were reused, the
    # single TOP frame of video 2 would complete that repetition.
    run = run_videos_in_one_session(
        [TOP_ANGLE, BOTTOM_ANGLE],
        [TOP_ANGLE])
    assert [s.repetitions for s in run.summaries] == [0, 0]


def test_frame_numbers_and_counts_restart_for_each_video():
    with tempfile.TemporaryDirectory() as folder, \
            fake_environment([], []) as env:
        header_rows = []
        for index in range(2):
            items = [TOP_ANGLE, BOTTOM_ANGLE, TOP_ANGLE]
            env.results[:] = [landmarks_for_elbow_angle(a) for a in items]
            env.capture.frames[:] = [make_frame() for _ in items]
            env.capture.position, env.capture.released = 0, False
            output = os.path.join(folder, f"out{index}.csv")
            with contextlib.redirect_stdout(io.StringIO()):
                env.demo.process_video(
                    write_video_placeholder(folder, f"c{index}.mp4"), output)
            header_rows.append(read_csv_file(output))
    for header, records in header_rows:
        assert header == EXPECTED_COLUMNS
        assert [r["frame_number"] for r in records] == ["0", "1", "2"]


def test_a_video_that_fails_does_not_poison_the_next_one():
    # Video 1 fails half way, at the bottom of a push-up. Video 2 must start
    # with a clean counter and a clean detector, and must be fully released.
    run = run_videos_in_one_session(
        [TOP_ANGLE, BOTTOM_ANGLE, TOP_ANGLE],       # fails at frame 2
        [TOP_ANGLE],
        process_error=RuntimeError("boom"), process_error_at=2)
    assert isinstance(run.errors[0], RuntimeError) and run.summaries[0] is None
    assert run.errors[1] is None
    assert run.summaries[1].repetitions == 0        # a shared counter would give 1
    assert len(run.env.poses) == 2                  # one detector per video
    assert all(pose.closed and pose.close_calls == 1 for pose in run.env.poses)
    assert run.env.capture.released is True


def test_a_failed_first_video_still_releases_before_the_second_starts():
    run = run_videos_in_one_session(
        [TOP_ANGLE, BOTTOM_ANGLE],                  # fails at frame 1
        [TOP_ANGLE, BOTTOM_ANGLE, TOP_ANGLE],
        process_error=RuntimeError("boom"), process_error_at=1)
    # Video 2 also hits the error at ITS frame 1, so both fail: each failure
    # releases its own detector and the shared video exactly once.
    assert [type(e) for e in run.errors] == [RuntimeError, RuntimeError]
    assert len(run.env.poses) == 2
    assert all(pose.closed and pose.close_calls == 1 for pose in run.env.poses)


# ===========================================================================
# 6. Command line and the single-frame demo
# ===========================================================================

def test_main_prints_a_short_summary_with_the_repetition_total():
    angles = [TOP_ANGLE, BOTTOM_ANGLE, TOP_ANGLE, BOTTOM_ANGLE, TOP_ANGLE]
    frames = [make_frame() for _ in angles]
    results = [landmarks_for_elbow_angle(a) for a in angles]
    with tempfile.TemporaryDirectory() as folder, \
            fake_environment(frames, results) as env:
        video = write_video_placeholder(folder)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = env.demo.main([video, os.path.join(folder, "r.csv")])
        assert code == 0 and os.path.exists(os.path.join(folder, "r.csv"))
    text = out.getvalue()
    assert "Processed 5 frames" in text
    assert "Repetitions counted: 2" in text
    assert "Form Analysis" not in text           # no per-frame report


def test_main_reports_a_bad_path_without_a_traceback():
    with tempfile.TemporaryDirectory() as folder, fake_environment() as env:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), \
                contextlib.redirect_stdout(io.StringIO()):
            code = env.demo.main([os.path.join(folder, "typo.mp4")])
        assert code == 1 and err.getvalue().startswith("Error: ")
        assert "Traceback" not in err.getvalue()
        assert env.capture_requests == [] and env.poses == []


def test_single_frame_demo_uses_person_1s_detector_and_keeps_its_output():
    with fake_environment([make_frame()], [LANDMARK_EXAMPLE]) as env:
        demo = env.import_module("single_frame_demo")
        assert env.poses == []                    # importing starts no model
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            demo.main()
        pose = env.poses[0]
        conversions = list(env.cv2_calls)
    text = out.getvalue()
    assert pose.kwargs["static_image_mode"] is True     # one still image
    assert pose.closed is True and len(env.poses) == 1
    assert conversions == [4]                            # one conversion
    assert "Frame size: 640 x 480 pixels" in text
    lines = text.splitlines()
    start = lines.index("Feature row (one frame):")
    assert [line.split()[0] for line in lines[start + 1:start + 12]] == (
        EXPECTED_COLUMNS)
    assert lines[start + 1].rstrip().endswith("None")   # frame_number


def test_single_frame_demo_with_nobody_in_the_frame_stops_politely():
    with fake_environment([make_frame()], [None]) as env:
        demo = env.import_module("single_frame_demo")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            demo.main()
        assert env.poses[0].closed is True
    assert "No person detected in the frame" in out.getvalue()
    assert "Feature row" not in out.getvalue()


def test_single_frame_demo_reports_a_missing_person_1_file_without_crashing():
    missing = os.path.join(tempfile.gettempdir(), "no_such_folder",
                           "pose_detector.py")
    with fake_environment([make_frame()], [LANDMARK_EXAMPLE],
                          person1_path=missing) as env:
        demo = env.import_module("single_frame_demo")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            demo.main()
    assert out.getvalue().startswith("Error: ")
    assert "Feature row" not in out.getvalue()


def test_single_frame_demo_stops_when_the_camera_gives_no_frame():
    with fake_environment([], []) as env:
        demo = env.import_module("single_frame_demo")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            demo.main()
        assert env.poses == []                    # no detector was created
    assert "Could not read a frame from the camera." in out.getvalue()


# ---- Simple runner so `python test_video_repetition_integration.py` works --

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
