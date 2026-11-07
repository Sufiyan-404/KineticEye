"""
video_analysis_demo.py

Processes a LOCAL VIDEO FILE frame by frame and saves one row of push-up
features per frame to a CSV file.

It is the video version of single_frame_demo.py. It only connects pieces that
already exist:

    video file
        -> OpenCV reads one frame                 (this file)
        -> PoseDetector.detect_pose(frame)        (Person 1: person1/src/
              pose_detector.py, loaded by pose_adapter.py; it converts
              BGR -> RGB itself and returns landmarks in PIXELS)
        -> pose_data_to_landmarks()               (pose_adapter.py: reshape
              only; visibility copied unchanged)
        -> analyze_landmarks()                    (pushup_angle_demo.py: NO
              image size, the pixels are already final; visibility check
              (threshold 0.5), joint angles, rule-based form analysis)
        -> extract_features()                     (feature_extractor.py)
        -> one row per frame
        -> RepetitionCounter.update(phase)        (repetition_counter.py: once
              per decoded frame; the total is only printed, it is NOT a CSV
              column)
        -> CSV file                               (Python's built-in csv module)

This file contains NO maths and NO form rules.

Rules of the output:
    * Exactly ONE row for every frame OpenCV could decode, even when no
      person is detected (the angles are then empty and the form status is
      INCOMPLETE_DATA).
    * frame_number is the zero-based position of the frame in the video:
      0 for the first decoded frame, 1 for the next, and so on.
    * The columns are exactly feature_extractor.FEATURE_COLUMNS (11 columns,
      same order as always). An unavailable angle is an EMPTY cell in the CSV.

Usage (all project files must be in the same folder):
    python video_analysis_demo.py VIDEO [OUTPUT_CSV]

Examples:
    python video_analysis_demo.py my_pushups.mp4
    python video_analysis_demo.py my_pushups.mp4 results/pushups.csv
    python video_analysis_demo.py "C:\\Videos\\my pushups.mp4" features.csv

If OUTPUT_CSV is left out, the file is named after the video (my_pushups.mp4
-> my_pushups_features.csv) and saved in the folder you run the command from.
An existing CSV with the same name is overwritten.

Film yourself SIDE-ON with your LEFT side facing the camera, so the left
shoulder, elbow, wrist, hip and ankle are visible (the existing angle code
only uses the left side).
"""

import argparse
import csv
import os
import sys
from collections import namedtuple

import cv2

from feature_extractor import extract_features, FEATURE_COLUMNS
from pose_adapter import (
    Person1DetectorError,
    create_person1_detector,
    pose_data_to_landmarks,
)
# Person 2's own detector helpers: only used by the MediaPipe-compatible path
# of analyze_frame() (callers that pass mp_pose.Pose objects). Importing them
# does not start a MediaPipe model.
from pose_detector import mp_pose, get_landmarks
from pushup_angle_demo import analyze_landmarks
from repetition_counter import RepetitionCounter

# Print a short progress line after every this-many frames.
PROGRESS_EVERY = 100

# What process_video() returns.
VideoSummary = namedtuple(
    "VideoSummary",
    ["frames_processed", "frames_with_person", "output_csv", "repetitions"],
    defaults=[0])


class VideoAnalysisError(Exception):
    """A problem we can explain to the user in one friendly sentence."""


# ===========================================================================
# 1. CHECKING THE PATHS (before any slow work starts)
# ===========================================================================

def default_output_path(video_path):
    """my_pushups.mp4 -> my_pushups_features.csv (in the current folder)."""
    stem = os.path.splitext(os.path.basename(os.fspath(video_path)))[0]
    return (stem or "video") + "_features.csv"


def _same_file(path_a, path_b):
    """True if both paths point at the same file on disk."""
    try:
        if os.path.exists(path_a) and os.path.exists(path_b):
            return os.path.samefile(path_a, path_b)
    except OSError:
        pass
    return os.path.normcase(os.path.abspath(path_a)) == os.path.normcase(
        os.path.abspath(path_b))


def check_paths(video_path, output_csv):
    """
    Check both paths and raise VideoAnalysisError with a clear message if
    something is wrong. Nothing is opened or created here, so a typo is
    reported straight away instead of after minutes of processing.
    """
    if not isinstance(video_path, (str, os.PathLike)) or not str(video_path):
        raise VideoAnalysisError(
            "No video file was given. Pass the path of a video, for example: "
            "python video_analysis_demo.py my_pushups.mp4")

    video_path = os.fspath(video_path)
    if not os.path.exists(video_path):
        raise VideoAnalysisError(
            f"Video file not found: {video_path}\n"
            "Check the spelling and the folder. If the path contains spaces, "
            "put it in quotes.")
    if os.path.isdir(video_path):
        raise VideoAnalysisError(
            f"{video_path} is a folder, not a video file. "
            "Give the path of the video itself.")

    output_csv = os.fspath(output_csv)
    if os.path.isdir(output_csv):
        raise VideoAnalysisError(
            f"The output path {output_csv} is a folder. "
            "Give a file name ending in .csv.")
    output_folder = os.path.dirname(os.path.abspath(output_csv))
    if not os.path.isdir(output_folder):
        raise VideoAnalysisError(
            f"The folder for the output file does not exist: {output_folder}\n"
            "Create it first or choose another output path.")
    if _same_file(video_path, output_csv):
        raise VideoAnalysisError(
            "The output CSV would overwrite the video itself. "
            "Choose a different output path.")


# ===========================================================================
# 2. WORKING ON ONE FRAME
# ===========================================================================

def _analyze_with_detector(frame, pose):
    """
    Person 1's class-based detector (PoseDetector): it takes the original BGR
    frame, converts it to RGB ITSELF and returns landmarks that are already in
    PIXELS. So there is no colour conversion here and NO image size is given
    to analyze_landmarks() (giving one would scale the pixels a second time).
    Returns (landmarks, (angles, analysis, notes)).
    """
    _, pose_data = pose.detect_pose(frame, draw=False)
    landmarks = pose_data_to_landmarks(pose_data)
    return landmarks, analyze_landmarks(landmarks)


def _analyze_with_mediapipe(frame, pose):
    """
    A MediaPipe-compatible pose object (anything with process(rgb_frame), such
    as mp_pose.Pose): this function converts BGR to RGB (once), and the 0-1
    landmarks are scaled to pixels by analyze_landmarks() (once), using the
    real size of the frame.
    Returns (landmarks, (angles, analysis, notes)).
    """
    # The real size of the frame. OpenCV frames are NumPy arrays whose shape
    # is (height, width, colours): the HEIGHT comes first.
    frame_height, frame_width = frame.shape[:2]

    # OpenCV gives BGR; MediaPipe expects RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    landmarks = get_landmarks(results)

    # Convert to pixels, check visibility, measure angles, analyze the form.
    return landmarks, analyze_landmarks(
        landmarks, image_width=frame_width, image_height=frame_height)


def analyze_frame(frame, frame_number, pose):
    """
    Run one decoded BGR frame through the whole pipeline.

    `pose` is the object that finds the person. Two kinds are supported:
      * Person 1's class-based PoseDetector (has detect_pose()): used by
        process_video(). It does its own BGR -> RGB conversion and returns
        pixel coordinates.
      * a MediaPipe-compatible object (has process()), as before: this
        function converts the colours and scales the 0-1 landmarks.
    Either way the colours are converted once and the coordinates are scaled
    once.

    Returns (row, person_found):
        row          : the 11-column feature dictionary for this frame
        person_found : True if a person was detected in the frame
    This never skips a frame: with nobody detected, the landmark dictionary is
    empty, every angle becomes None and the analyzer reports INCOMPLETE_DATA.
    """
    if hasattr(pose, "detect_pose"):
        landmarks, (angles, analysis, _notes) = _analyze_with_detector(
            frame, pose)
    elif hasattr(pose, "process"):
        landmarks, (angles, analysis, _notes) = _analyze_with_mediapipe(
            frame, pose)
    else:
        raise TypeError(
            "pose must have a detect_pose() method (Person 1's PoseDetector) "
            f"or a process() method (MediaPipe), but got {type(pose).__name__}.")

    row = extract_features(
        elbow_angle=angles["elbow_angle"],
        hip_angle=angles["hip_angle"],
        shoulder_angle=angles["shoulder_angle"],
        analysis=analysis,
        frame_number=frame_number,
    )
    return row, bool(landmarks)


# ===========================================================================
# 3. OPENING THE VIDEO AND WRITING THE CSV
# ===========================================================================

def open_video(video_path):
    """Open the video with OpenCV, or raise VideoAnalysisError."""
    capture = cv2.VideoCapture(os.fspath(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoAnalysisError(
            f"OpenCV could not open {video_path} as a video. The file may be "
            "damaged, not a video, or in a format your OpenCV cannot read. "
            "Try an .mp4 file.")
    return capture


def get_total_frames(capture):
    """Approximate number of frames (only used for progress), or None."""
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    except (TypeError, ValueError, OverflowError, AttributeError):
        return None
    return total if total > 0 else None


def write_csv(rows, output_csv):
    """Write the feature rows to a CSV file (header first, FEATURE_COLUMNS)."""
    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FEATURE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as error:
        raise VideoAnalysisError(
            f"Could not write the CSV file {output_csv}: {error.strerror or error}\n"
            "Is the file open in another program (for example Excel)?"
        ) from error


# ===========================================================================
# 4. THE MAIN LOOP
# ===========================================================================

def process_video(video_path, output_csv=None, progress_every=PROGRESS_EVERY):
    """
    Read every frame of the video, build one feature row per frame and save
    them all to output_csv. Returns a VideoSummary.

    Raises VideoAnalysisError (with a friendly message) if the path is wrong,
    the video cannot be opened, it contains no readable frame, or the CSV
    cannot be written. The video and the MediaPipe model are always released,
    even if something goes wrong half way. No CSV is written in that case.
    """
    if output_csv is None:
        output_csv = default_output_path(video_path)
    check_paths(video_path, output_csv)

    capture = open_video(video_path)
    detector = None
    rows = []
    frames_with_person = 0
    counter = RepetitionCounter()
    try:
        total_frames = get_total_frames(capture)

        # ONE detector for the whole video (this starts the MediaPipe model).
        # static_image_mode=False: this is a VIDEO, so MediaPipe tracks the
        # person from one frame to the next instead of searching from scratch
        # every time. The other settings match Person 1's webcam script.
        try:
            detector = create_person1_detector(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Person1DetectorError as error:
            raise VideoAnalysisError(str(error)) from error

        frame_number = 0            # zero-based, counts EVERY decoded frame
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break               # end of the video

            row, person_found = analyze_frame(frame, frame_number, detector)
            rows.append(row)
            frames_with_person += person_found
            # Exactly one update per decoded frame. A frame with no usable
            # phase is UNKNOWN, which leaves the counter unchanged.
            counter.update(row["rule_movement_phase"])
            frame_number += 1

            if progress_every and frame_number % progress_every == 0:
                of_total = f" of about {total_frames}" if total_frames else ""
                print(f"  Processed {frame_number}{of_total} frames...",
                      flush=True)
    finally:
        # Release BOTH resources, even if closing the detector fails.
        try:
            if detector is not None:
                detector.close()
        finally:
            capture.release()

    if not rows:
        raise VideoAnalysisError(
            f"No frames could be read from {video_path}. "
            "The video may be empty or damaged.")

    write_csv(rows, output_csv)
    return VideoSummary(len(rows), frames_with_person, output_csv,
                        counter.count)


# ===========================================================================
# 5. COMMAND LINE
# ===========================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        prog="video_analysis_demo.py",
        description=("Analyse a video file frame by frame and save one row of "
                     "push-up features per frame to a CSV file."),
        epilog=("examples:\n"
                "  python video_analysis_demo.py my_pushups.mp4\n"
                "  python video_analysis_demo.py my_pushups.mp4 results.csv\n"
                "\n"
                "Film yourself side-on with your LEFT side facing the camera."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "video_path", metavar="VIDEO",
        help="path of the video file to analyse (put it in quotes if it "
             "contains spaces)")
    parser.add_argument(
        "output_csv", metavar="OUTPUT_CSV", nargs="?", default=None,
        help="where to save the CSV (default: <video name>_features.csv in "
             "the current folder)")
    return parser


def main(argv=None):
    """Run the script. Returns 0 on success, 1 on a problem, 130 if stopped."""
    args = build_parser().parse_args(argv)
    output_csv = args.output_csv or default_output_path(args.video_path)

    print(f"Analysing {args.video_path} ...")
    try:
        summary = process_video(args.video_path, output_csv)
    except VideoAnalysisError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped by the user. No CSV file was written.",
              file=sys.stderr)
        return 130

    print(f"Done. Processed {summary.frames_processed} frames.")
    print(f"A person was detected in {summary.frames_with_person} of them.")
    print(f"Repetitions counted: {summary.repetitions}")
    if summary.frames_with_person == 0:
        print("Warning: nobody was detected, so every angle in the CSV is "
              "empty. Make sure your whole body is visible in the video.")
    print(f"Saved the feature rows to: {summary.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
