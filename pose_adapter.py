"""
pose_adapter.py

Connects Person 1's class-based pose detector (person1/src/pose_detector.py)
to Person 2's existing angle / form / feature pipeline.

WHAT PERSON 1'S DETECTOR PROVIDES (checked against the real file)
    PoseDetector(static_image_mode, model_complexity, smooth_landmarks,
                 enable_segmentation, min_detection_confidence,
                 min_tracking_confidence, visibility_threshold)
    detector.detect_pose(frame, draw=True) -> (annotated_frame, pose_data)
    detector.close()                         (also usable with `with`)

    It takes an OpenCV BGR frame and converts it to RGB ITSELF. pose_data is:
        "pose_detected"  True / False
        "landmarks"      {"left_elbow": (x_px, y_px) or None, ...}
                         x_px / y_px are PIXELS: whole numbers, already scaled
                         to the frame size, rounded and kept inside the image
        "visibility"     {"left_elbow": 0.93 or None, ...}  (a separate dict)

WHAT PERSON 2'S PIPELINE EXPECTS
    {"left_elbow": {"x": ..., "y": ..., "visibility": ...}, ...}
    pushup_angle_demo.analyze_landmarks(landmarks) uses x and y as they are
    when no image size is given, so landmarks that are ALREADY in pixels are
    passed WITHOUT image_width / image_height. (With a size it would multiply
    by the size again, which would be a bug.) That is how coordinates are
    scaled exactly once: Person 1's detector does it, Person 2's code does not.

WHAT THIS MODULE DOES
    1. load_person1_detector_class() loads Person 1's file from its PATH under
       a private name. Both people have a file called pose_detector.py, so a
       normal import could pick the wrong one. The file is not added to
       sys.modules, so it can never shadow Person 2's pose_detector.py.
       Loading only imports the file; it does NOT create a MediaPipe model.
    2. create_person1_detector() creates the detector (this is where the model
       is created) and always sets visibility_threshold=0.0. Person 1's
       detector would otherwise throw away the coordinates of low-visibility
       landmarks; with 0.0 it keeps them, so Person 2's existing visibility
       rule (VISIBILITY_THRESHOLD = 0.5 in pushup_angle_demo.py) is the one
       and only place where that decision is made.
    3. pose_data_to_landmarks() reshapes pose_data into the dictionary above.
       Coordinates are only reshaped (not scaled, not rounded again) and each
       visibility value is copied unchanged, never made up.

It does no angle maths, no colour conversion and no form analysis, and prints
nothing.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Protocol, Tuple, Union

from pushup_angle_demo import is_valid_number

# The private name Person 1's file is loaded under (never put in sys.modules).
PERSON1_MODULE_NAME = "person1_pose_detector"

# Expected location: <project root>/person1/src/pose_detector.py, next to
# <project root>/person2/src/ (where this file lives).
DEFAULT_PERSON1_DETECTOR_PATH: Path = (
    Path(__file__).resolve().parents[2] / "person1" / "src" / "pose_detector.py"
)

# {"left_elbow": {"x": 412.0, "y": 233.0, "visibility": 0.93}, ...}
LandmarkDict = Dict[str, Dict[str, Any]]
PathLike = Union[str, "os.PathLike[str]"]


class Person1DetectorError(Exception):
    """Person 1's detector could not be found, loaded, understood or created."""


class PoseDetectorLike(Protocol):
    """What Person 2's code needs from a class-based detector."""

    def detect_pose(self, frame: Any, draw: bool = True) -> Tuple[Any, Dict[str, Any]]:
        ...

    def close(self) -> None:
        ...


# ===========================================================================
# 1. LOADING PERSON 1'S FILE (no model is created here)
# ===========================================================================

def _import_from_path(path: Path) -> ModuleType:
    """Execute the file at `path` as a private module (not in sys.modules)."""
    spec = importlib.util.spec_from_file_location(PERSON1_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise Person1DetectorError(f"{path} cannot be loaded as a Python file.")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as error:
        raise Person1DetectorError(
            f"Person 1's pose detector ({path}) could not be imported: "
            f"{error}\nCheck that opencv-python and mediapipe are installed."
        ) from error
    return module


def load_person1_detector_class(path: Optional[PathLike] = None) -> type:
    """
    Load Person 1's file and return its PoseDetector CLASS (not an object).

    Nothing is created or started: no MediaPipe model exists afterwards.
    Raises Person1DetectorError if the file is missing, cannot be imported, or
    has no PoseDetector class with detect_pose() and close() methods.
    """
    file_path = Path(path if path is not None else DEFAULT_PERSON1_DETECTOR_PATH)
    if not file_path.is_file():
        raise Person1DetectorError(
            f"Person 1's pose detector was not found at: {file_path}\n"
            "It should be person1/src/pose_detector.py, next to the person2 "
            "folder.")

    module = _import_from_path(file_path)

    detector_class = getattr(module, "PoseDetector", None)
    if not isinstance(detector_class, type):
        hint = ""
        if hasattr(module, "get_landmarks") or hasattr(module, "mp_pose"):
            hint = (" It defines mp_pose / get_landmarks instead, which is a "
                    "different detector design that this project does not "
                    "use here.")
        raise Person1DetectorError(
            f"{file_path} does not define a PoseDetector class.{hint}")

    missing = [name for name in ("detect_pose", "close")
               if not callable(getattr(detector_class, name, None))]
    if missing:
        raise Person1DetectorError(
            f"The PoseDetector class in {file_path} lacks the method(s): "
            f"{', '.join(missing)}.")
    return detector_class


def create_person1_detector(path: Optional[PathLike] = None,
                            **options: Any) -> PoseDetectorLike:
    """
    Create Person 1's PoseDetector (this starts the MediaPipe model).

    options : any PoseDetector constructor options (static_image_mode,
        model_complexity, smooth_landmarks, enable_segmentation,
        min_detection_confidence, min_tracking_confidence). Unknown names are
        reported with the list of allowed ones.
    visibility_threshold is ALWAYS 0.0 (see the module description). Asking
    for another value is an error: a second threshold would hide data from
    Person 2's visibility check.

    The caller owns the returned detector and must close() it (or use `with`).
    """
    detector_class = load_person1_detector_class(path)

    if "visibility_threshold" in options:
        if options.pop("visibility_threshold") != 0.0:
            raise ValueError(
                "visibility_threshold must be 0.0 here. The existing 0.5 "
                "visibility check in pushup_angle_demo.py is the only "
                "visibility rule; a second threshold would hide data from it.")

    allowed = [name for name in inspect.signature(detector_class.__init__)
               .parameters if name not in ("self", "visibility_threshold")]
    unknown = sorted(set(options) - set(allowed))
    if unknown:
        raise Person1DetectorError(
            f"PoseDetector does not accept: {', '.join(unknown)}. "
            f"Allowed options: {', '.join(allowed)}.")

    return detector_class(visibility_threshold=0.0, **options)


# ===========================================================================
# 2. RESHAPING ONE RESULT (pure function: no camera, no MediaPipe)
# ===========================================================================

def _pixel_point(point: Any) -> Optional[Tuple[float, float]]:
    """(x_px, y_px) as floats, or None if `point` is not two usable numbers."""
    try:
        x_px, y_px = point
    except (TypeError, ValueError):
        return None                       # None, wrong length, not a sequence
    if is_valid_number(x_px) and is_valid_number(y_px):
        return float(x_px), float(y_px)
    return None


def pose_data_to_landmarks(pose_data: Dict[str, Any]) -> LandmarkDict:
    """
    Turn Person 1's pose_data into Person 2's landmark dictionary.

    The x and y values stay PIXELS (they are only reshaped), so the result must
    be given to analyze_landmarks() / print_landmark_report() WITHOUT an image
    size. Each visibility is copied unchanged from pose_data["visibility"].

    * No person detected                      -> {}  (every angle will be None)
    * A landmark that is None or has unusable coordinates is left OUT, so
      Person 2's check reports it as missing and the angles that need it
      become unavailable.
    * A landmark with no visibility entry gets no "visibility" key, which
      Person 2's check also rejects. Nothing is ever invented.
    """
    if not isinstance(pose_data, dict):
        raise TypeError(
            "pose_data must be the dictionary returned by "
            f"PoseDetector.detect_pose(), but got {type(pose_data).__name__}.")

    if not pose_data.get("pose_detected"):
        return {}
    points = pose_data.get("landmarks")
    if not isinstance(points, dict):
        return {}
    visibilities = pose_data.get("visibility")
    if not isinstance(visibilities, dict):
        visibilities = {}

    landmarks: LandmarkDict = {}
    for name, point in points.items():
        pixel = _pixel_point(point)
        if pixel is None:
            continue
        entry: Dict[str, Any] = {"x": pixel[0], "y": pixel[1]}
        if name in visibilities:
            entry["visibility"] = visibilities[name]     # unchanged
        landmarks[name] = entry
    return landmarks
