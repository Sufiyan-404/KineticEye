"""
test_pose_adapter.py

Tests for pose_adapter.py, which connects Person 1's class-based
PoseDetector (person1/src/pose_detector.py) to Person 2's pipeline:

  * loading the right file by its path, without shadowing Person 2's
    pose_detector.py and without starting a MediaPipe model on import
  * creating the detector with the real constructor options (and with
    visibility_threshold forced to 0.0)
  * reshaping pose_data (pixel tuples + separate visibility) into the landmark
    dictionary Person 2 uses, with visibility unchanged and no second scaling

Person 1's REAL file runs here, on top of a fake cv2 and a fake mediapipe (from
test_video_repetition_integration.py, which this file reuses), so nothing needs
to be installed and no webcam is used.

Run either way (from person2/src):
    python test_pose_adapter.py
    pytest -v test_pose_adapter.py
"""

from __future__ import annotations

import ast
import contextlib
import copy
import io
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Dict

import pose_adapter
import pushup_angle_demo
from pose_adapter import (
    DEFAULT_PERSON1_DETECTOR_PATH,
    PERSON1_MODULE_NAME,
    Person1DetectorError,
    create_person1_detector,
    load_person1_detector_class,
    pose_data_to_landmarks,
)
from pushup_angle_demo import LANDMARK_EXAMPLE, analyze_landmarks
from test_video_repetition_integration import (
    LANDMARK_NAME_ORDER,
    close_to,
    expect_error,
    expected_angles_for_pixels,
    fake_environment,
    make_frame,
    make_landmarks,
    pose_data_for,
    whole_pixel_points,
)

ANGLE_NAMES = ["elbow_angle", "hip_angle", "shoulder_angle"]

# A minimal class-based detector file with the interface Person 2 needs.
VALID_SOURCE = (
    "class PoseDetector:\n"
    "    def __init__(self, static_image_mode=False, model_complexity=1,\n"
    "                 visibility_threshold=0.0):\n"
    "        self.static_image_mode = static_image_mode\n"
    "        self.model_complexity = model_complexity\n"
    "        self.visibility_threshold = visibility_threshold\n"
    "    def detect_pose(self, frame, draw=True):\n"
    "        return frame, {}\n"
    "    def close(self):\n"
    "        pass\n"
)


# ---- Helpers (their names do not start with "test_") --------------------

@contextlib.contextmanager
def person1_file(source: str):
    """A temporary 'pose_detector.py' with the given source; yields its path."""
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "pose_detector.py")
        with open(path, "w", encoding="utf-8") as file:
            file.write(source)
        yield path


def error_for(source: str) -> str:
    with fake_environment(import_demo=False), person1_file(source) as path:
        return expect_error(Person1DetectorError,
                            load_person1_detector_class, path)


def detector_output(landmarks, width=640, height=480) -> Dict[str, Any]:
    """
    pose_data produced by Person 1's REAL detect_pose() (running on the fake
    MediaPipe) for these landmarks. Also returns the fake Pose for inspection.
    """
    with fake_environment([make_frame(height, width)], [landmarks],
                          import_demo=False) as env:
        with create_person1_detector() as detector:
            _, pose_data = detector.detect_pose(make_frame(height, width),
                                                draw=False)
        return pose_data


# ---- 1. The expected location ---------------------------------------------

def test_the_default_location_is_person1_src_pose_detector():
    assert DEFAULT_PERSON1_DETECTOR_PATH.parts[-3:] == (
        "person1", "src", "pose_detector.py")
    assert DEFAULT_PERSON1_DETECTOR_PATH.parent.parent.parent == (
        Path(pose_adapter.__file__).resolve().parents[2])


def test_person_1s_real_file_is_present_and_is_the_class_based_detector():
    assert DEFAULT_PERSON1_DETECTOR_PATH.is_file(), (
        f"expected Person 1's detector at {DEFAULT_PERSON1_DETECTOR_PATH}")
    text = DEFAULT_PERSON1_DETECTOR_PATH.read_text(encoding="utf-8")
    assert "class PoseDetector" in text and "def detect_pose" in text


# ---- 2. Loading the real file: no wrong file, no model on import -------------

def test_the_real_file_loads_the_poseDetector_class_without_starting_a_model():
    with fake_environment(import_demo=False) as env:
        detector_class = load_person1_detector_class()
        assert isinstance(detector_class, type)
        assert detector_class.__name__ == "PoseDetector"
        assert detector_class.__module__ == PERSON1_MODULE_NAME
        assert callable(detector_class.detect_pose)
        assert callable(detector_class.close)
        assert env.poses == []              # loading created NO MediaPipe model


def test_loading_never_touches_sys_modules_and_never_shadows_person_2s_file():
    with fake_environment(import_demo=False):
        person2_module = types.ModuleType("pose_detector")     # "Person 2's"
        sys.modules["pose_detector"] = person2_module
        before = dict(sys.modules)
        load_person1_detector_class()
        assert sys.modules["pose_detector"] is person2_module
        assert PERSON1_MODULE_NAME not in sys.modules
        assert sys.modules.keys() == before.keys()
        assert all(sys.modules[n] is m for n, m in before.items())


def test_the_loader_picks_the_file_at_the_given_path_not_a_same_named_one():
    other = VALID_SOURCE.replace("model_complexity=1", "model_complexity=7")
    with fake_environment(import_demo=False), \
            person1_file(VALID_SOURCE) as first, person1_file(other) as second:
        one = load_person1_detector_class(first)()
        two = load_person1_detector_class(second)()
        assert one.model_complexity == 1 and two.model_complexity == 7


def test_person_2s_own_mp_pose_style_file_is_refused_with_an_explanation():
    person2_file = Path(pose_adapter.__file__).parent / "pose_detector.py"
    assert person2_file.is_file()
    with fake_environment(import_demo=False):
        message = expect_error(Person1DetectorError,
                               load_person1_detector_class, person2_file)
    assert "PoseDetector class" in message and "mp_pose" in message


def test_each_load_runs_the_file_again_so_nothing_goes_stale():
    with fake_environment(import_demo=False):
        assert load_person1_detector_class() is not load_person1_detector_class()


def test_a_path_may_be_text_or_a_path_object_and_the_default_is_read_late():
    with fake_environment(import_demo=False), person1_file(VALID_SOURCE) as p:
        assert load_person1_detector_class(p).__name__ == "PoseDetector"
        assert load_person1_detector_class(Path(p)).__name__ == "PoseDetector"
        original = pose_adapter.DEFAULT_PERSON1_DETECTOR_PATH
        pose_adapter.DEFAULT_PERSON1_DETECTOR_PATH = p
        try:
            assert load_person1_detector_class()().model_complexity == 1
        finally:
            pose_adapter.DEFAULT_PERSON1_DETECTOR_PATH = original


# ---- 3. Missing or wrong files -----------------------------------------------

def test_a_missing_file_gives_a_clear_error_with_the_path():
    missing = os.path.join(tempfile.gettempdir(), "no_such_folder",
                           "pose_detector.py")
    message = expect_error(Person1DetectorError,
                           load_person1_detector_class, missing)
    assert "not found" in message and "no_such_folder" in message
    assert "person1/src/pose_detector.py" in message


def test_a_folder_instead_of_a_file_is_reported():
    with tempfile.TemporaryDirectory() as folder:
        expect_error(Person1DetectorError, load_person1_detector_class, folder)


def test_a_file_without_a_poseDetector_class_is_reported():
    assert "does not define a PoseDetector class" in error_for("x = 1\n")
    assert "does not define a PoseDetector class" in error_for(
        "PoseDetector = 5\n")


def test_a_class_without_detect_pose_or_close_is_reported():
    message = error_for("class PoseDetector:\n    pass\n")
    assert "detect_pose" in message and "close" in message
    message = error_for("class PoseDetector:\n"
                        "    def detect_pose(self, frame, draw=True): pass\n")
    assert "close" in message and "detect_pose" not in message.split("lacks")[1]


def test_an_import_error_inside_the_file_is_explained_and_chained():
    with fake_environment(import_demo=False), \
            person1_file("import a_module_that_does_not_exist_xyz\n") as path:
        try:
            load_person1_detector_class(path)
        except Person1DetectorError as error:
            assert "opencv-python and mediapipe" in str(error)
            assert isinstance(error.__cause__, ImportError)
        else:
            raise AssertionError("expected Person1DetectorError")


def test_other_errors_inside_the_file_are_not_hidden():
    with fake_environment(import_demo=False), \
            person1_file("raise RuntimeError('broken file')\n") as path:
        assert expect_error(RuntimeError, load_person1_detector_class,
                            path) == "broken file"


# ---- 4. Creating the detector with the real constructor ---------------------

def test_the_detector_is_created_with_the_real_constructor_options():
    with fake_environment(import_demo=False) as env:
        detector = create_person1_detector(
            static_image_mode=True, model_complexity=0, smooth_landmarks=False,
            enable_segmentation=False, min_detection_confidence=0.6,
            min_tracking_confidence=0.7)
        try:
            assert len(env.poses) == 1            # the model starts HERE
            kwargs = env.poses[0].kwargs
            assert kwargs["static_image_mode"] is True
            assert kwargs["model_complexity"] == 0
            assert kwargs["smooth_landmarks"] is False
            assert kwargs["min_detection_confidence"] == 0.6
            assert kwargs["min_tracking_confidence"] == 0.7
        finally:
            detector.close()
        assert env.poses[0].closed is True


def test_visibility_threshold_is_always_zero_so_no_coordinates_are_dropped():
    with fake_environment(import_demo=False):
        with create_person1_detector() as detector:
            assert detector.visibility_threshold == 0.0
        with create_person1_detector(visibility_threshold=0.0) as detector:
            assert detector.visibility_threshold == 0.0
        message = expect_error(ValueError, create_person1_detector,
                               visibility_threshold=0.5)
        assert "0.0" in message


def test_unknown_constructor_options_are_reported_with_the_allowed_ones():
    with fake_environment(import_demo=False):
        message = expect_error(Person1DetectorError, create_person1_detector,
                               frobnicate=True)
    assert "frobnicate" in message and "static_image_mode" in message
    assert "visibility_threshold" not in message.split("Allowed")[1]


def test_the_detector_works_as_a_context_manager_and_closes_once():
    with fake_environment(import_demo=False) as env:
        with create_person1_detector() as detector:
            assert env.poses[0].closed is False
        assert env.poses[0].closed and env.poses[0].close_calls == 1


def test_creating_the_detector_with_a_bad_file_fails_before_any_model_starts():
    with fake_environment(import_demo=False) as env, \
            person1_file("x = 1\n") as path:
        expect_error(Person1DetectorError, create_person1_detector, path)
        assert env.poses == []


# ---- 5. Landmark and visibility mapping (real detector output) ---------------

def test_pixels_and_visibility_map_to_person_2s_dictionary():
    landmarks = make_landmarks(left_wrist={"visibility": 0.491})
    data = detector_output(landmarks)
    result = pose_data_to_landmarks(data)
    expected_pixels = whole_pixel_points(landmarks, 640, 480)
    assert set(result) == set(LANDMARK_NAME_ORDER)
    for name, entry in result.items():
        assert list(entry) == ["x", "y", "visibility"], name
        assert (entry["x"], entry["y"]) == tuple(map(float, expected_pixels[name]))
        assert entry["x"] == data["landmarks"][name][0]      # reshaped only
        assert entry["visibility"] == data["visibility"][name]


def test_visibility_values_are_copied_unchanged():
    for value in (0.0, 0.137, 0.491, 0.5, 0.999, 1.0):
        data = {"pose_detected": True, "landmarks": {"left_wrist": (10, 20)},
                "visibility": {"left_wrist": value}}
        assert pose_data_to_landmarks(data)["left_wrist"]["visibility"] is value


def test_unusual_visibility_values_are_passed_on_not_invented_or_fixed():
    for odd in (None, "high", float("nan"), True):
        data = {"pose_detected": True, "landmarks": {"left_wrist": (10, 20)},
                "visibility": {"left_wrist": odd}}
        assert pose_data_to_landmarks(data)["left_wrist"]["visibility"] is odd
    # No visibility entry at all: the key is simply absent (nothing invented).
    data = {"pose_detected": True, "landmarks": {"left_wrist": (10, 20)}}
    assert "visibility" not in pose_data_to_landmarks(data)["left_wrist"]


def test_the_detector_keeps_low_visibility_coordinates_for_the_half_rule():
    data = detector_output(make_landmarks(left_wrist={"visibility": 0.2}))
    assert data["landmarks"]["left_wrist"] is not None       # not discarded
    assert data["visibility"]["left_wrist"] == 0.2
    landmarks = pose_data_to_landmarks(data)
    angles, _, notes = analyze_landmarks(landmarks)
    assert angles["elbow_angle"] is None                      # rejected by 0.5
    assert any("left_wrist visibility 0.200 is below 0.5" in n for n in notes)


def test_the_half_threshold_is_applied_by_the_existing_pipeline():
    assert pushup_angle_demo.VISIBILITY_THRESHOLD == 0.5
    for visibility, accepted in ((0.491, False), (0.499, False),
                                 (0.5, True), (0.9, True)):
        data = detector_output(make_landmarks(
            left_wrist={"visibility": visibility}))
        angles, _, notes = analyze_landmarks(pose_data_to_landmarks(data))
        assert (angles["elbow_angle"] is not None) == accepted, visibility
        assert (not notes) == accepted, visibility


def test_no_pose_gives_an_empty_dictionary():
    assert pose_data_to_landmarks(pose_data_for(None)) == {}
    assert pose_data_to_landmarks(detector_output(None)) == {}
    # pose_detected False wins even if stray landmark data is present.
    stray = pose_data_for(LANDMARK_EXAMPLE)
    stray["pose_detected"] = False
    assert pose_data_to_landmarks(stray) == {}


def test_missing_landmarks_are_left_out_and_the_rest_are_kept():
    data = pose_data_for(LANDMARK_EXAMPLE)
    data["landmarks"]["left_wrist"] = None
    del data["landmarks"]["left_ankle"]
    result = pose_data_to_landmarks(data)
    assert "left_wrist" not in result and "left_ankle" not in result
    assert "left_elbow" in result and "left_hip" in result
    angles, _, notes = analyze_landmarks(result)
    assert angles["elbow_angle"] is None and angles["hip_angle"] is None
    assert angles["shoulder_angle"] is not None
    assert any("left_wrist is missing" in n for n in notes)


def test_unusable_coordinates_are_skipped_without_crashing():
    bad_points = [None, "xy", (1,), (1, 2, 3), (None, 5), ("a", "b"),
                  (True, False), (float("nan"), 3), (3, float("inf")), [], 7,
                  {"x": 1, "y": 2}]
    for bad in bad_points:
        data = {"pose_detected": True,
                "landmarks": {"left_elbow": bad, "left_hip": (5, 5)},
                "visibility": {"left_elbow": 0.9, "left_hip": 0.9}}
        result = pose_data_to_landmarks(data)
        assert "left_elbow" not in result, repr(bad)
        assert "left_hip" in result, repr(bad)


def test_incomplete_pose_data_dictionaries_are_safe():
    assert pose_data_to_landmarks({}) == {}
    assert pose_data_to_landmarks({"pose_detected": True}) == {}
    assert pose_data_to_landmarks({"pose_detected": True,
                                   "landmarks": None}) == {}
    only_points = {"pose_detected": True, "landmarks": {"left_elbow": (1, 2)}}
    assert pose_data_to_landmarks(only_points) == {
        "left_elbow": {"x": 1.0, "y": 2.0}}


def test_pose_data_that_is_not_a_dictionary_is_rejected():
    for bad in (None, [], "pose", 5):
        message = expect_error(TypeError, pose_data_to_landmarks, bad)
        assert "pose_data" in message, repr(bad)


def test_the_conversion_does_not_modify_its_input():
    data = pose_data_for(LANDMARK_EXAMPLE)
    before = copy.deepcopy(data)
    pose_data_to_landmarks(data)
    assert data == before


# ---- 6. Coordinates are scaled exactly once -----------------------------------

def test_pixel_landmarks_give_the_angles_of_those_exact_pixels_with_no_size():
    landmarks = make_landmarks()
    data = detector_output(landmarks)
    result = pose_data_to_landmarks(data)
    angles, _, notes = analyze_landmarks(result)       # NO image size
    assert notes == []
    expected = expected_angles_for_pixels(
        {n: tuple(map(float, xy)) for n, xy in data["landmarks"].items()})
    for name in ANGLE_NAMES:
        assert close_to(angles[name], expected[name]), name


def test_passing_an_image_size_would_scale_the_pixels_a_second_time():
    # This is the mistake the adapter exists to prevent: the landmarks are
    # already pixels, so a size must never be given with them.
    data = detector_output(make_landmarks())
    elbow = pose_data_to_landmarks(data)["left_elbow"]
    once = pushup_angle_demo.landmark_to_point(elbow)
    twice = pushup_angle_demo.landmark_to_point(elbow, 640, 480)
    assert once == (elbow["x"], elbow["y"])
    assert twice[0] == elbow["x"] * 640 and twice[0] > 640   # far outside


# ---- 7. Purity -----------------------------------------------------------------

def test_the_adapter_prints_nothing():
    output = io.StringIO()
    with fake_environment([make_frame()], [LANDMARK_EXAMPLE],
                          import_demo=False), contextlib.redirect_stdout(output):
        with create_person1_detector() as detector:
            pose_data_to_landmarks(detector.detect_pose(
                make_frame(), draw=False)[1])
    assert output.getvalue() == ""


def test_the_adapter_does_no_colour_conversion_scaling_or_angle_work():
    # Inspect the CODE (not the explanatory docstring).
    tree = ast.parse(Path(pose_adapter.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, float):
            if node.value != 0.0:
                raise AssertionError(f"unexpected number {node.value}")
    for forbidden in ("VISIBILITY_THRESHOLD", "image_width", "image_height",
                      "cvtColor", "cv2", "numpy", "calculate_angle",
                      "analyze_landmarks", "landmark_to_point"):
        assert forbidden not in names, forbidden


# ---- Simple runner so `python test_pose_adapter.py` works ------------------

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
