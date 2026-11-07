"""
single_frame_demo.py

Smallest possible integration of Person 1 (MediaPipe) and Person 2 (angles +
form analysis) for the Sports Video Analytics & Action Counter project.

It takes ONE real webcam frame and sends it through the whole pipeline:

    webcam frame
        -> PoseDetector.detect_pose(frame)  (Person 1: person1/src/
                                              pose_detector.py, loaded by
                                              pose_adapter.py; converts
                                              BGR -> RGB itself, returns
                                              landmarks in PIXELS)
        -> pose_data_to_landmarks()         (Person 2: pose_adapter.py,
                                              reshape only)
        -> landmark_to_point()              (Person 2: pushup_angle_demo.py,
                                              no image size: pixels are final)
        -> calculate_angle()                (Person 2: angle_calculator.py)
        -> analyze_push_up_form()           (Person 2: form_analyzer.py)
        -> printed result
        -> extract_features()               (Person 2: feature_extractor.py)
        -> printed feature row (frame_number is None: one frame only)

This file contains NO maths and NO form rules. It only connects the existing
pieces. The angle and form-analysis steps happen inside
print_landmark_report(), which returns the angles and the analysis so they can
be passed to extract_features() without being calculated again.

Run with (all project files must be in the same folder):
    python single_frame_demo.py

Stand SIDE-ON to the camera with your LEFT side facing it, in a push-up
position, so the left shoulder, elbow, wrist, hip and ankle are all visible.
"""

import cv2

from feature_extractor import extract_features
from pose_adapter import (
    Person1DetectorError,
    create_person1_detector,
    pose_data_to_landmarks,
)
from pushup_angle_demo import print_landmark_report

# A webcam often gives dark or blank frames for the first moment after it
# opens. We read and throw away this many frames so the camera can adjust.
WARMUP_FRAMES = 30

# The landmarks our angle functions use (just for printing the raw values).
USED_LANDMARKS = ["left_shoulder", "left_elbow", "left_wrist",
                  "left_hip", "left_ankle"]


def capture_one_frame():
    """Open the webcam, let it warm up, and return one BGR frame (or None)."""
    cap = cv2.VideoCapture(0)
    try:
        frame = None
        for _ in range(WARMUP_FRAMES):
            success, image = cap.read()
            if success:
                frame = image
        return frame
    finally:
        cap.release()


def print_feature_row(features):
    """Print every field of one feature row, in column order."""
    print("Feature row (one frame):")
    for name, value in features.items():
        print(f"  {name:<26}{value}")


def main():
    # Step 1: get one frame from the camera.
    frame = capture_one_frame()
    if frame is None:
        print("Could not read a frame from the camera.")
        return

    # The real size of the frame. OpenCV frames are NumPy arrays whose shape
    # is (height, width, colours): the HEIGHT comes first.
    frame_height, frame_width = frame.shape[:2]

    # Step 2: run Person 1's detector on the ORIGINAL (BGR) frame. It does the
    # BGR -> RGB conversion itself, so we must not convert the colours here.
    # static_image_mode=True because we analyse a single still image. The
    # `with` block closes the detector even if something goes wrong.
    try:
        with create_person1_detector(static_image_mode=True,
                                     min_detection_confidence=0.5) as detector:
            _, pose_data = detector.detect_pose(frame, draw=False)
    except Person1DetectorError as error:
        print(f"Error: {error}")
        return

    # Step 3: reshape Person 1's result into the landmark dictionary the rest
    # of the pipeline uses. The coordinates are PIXELS already.
    landmarks = pose_data_to_landmarks(pose_data)

    # The dictionary is empty if nobody was detected.
    if not landmarks:
        print("No person detected in the frame. Stand further back so your")
        print("whole body is visible, then run the script again.")
        return

    # Show the raw values Person 1 sent us (so you can see the real format).
    print(f"Frame size: {frame_width} x {frame_height} pixels")
    print("Landmarks from Person 1's detector, in pixels (the ones we use):")
    for name in USED_LANDMARKS:
        print(f"  {name}: {landmarks.get(name, 'missing')}")
    print()

    # Step 4: Person 2's existing code does the rest (angles -> form analysis
    # -> print). NO image size is passed: the landmarks are already in pixels,
    # so passing the frame size would scale them a second time.
    report = print_landmark_report("Real MediaPipe Frame", landmarks)

    # print_landmark_report() returns (angles, analysis). If it returned
    # nothing (for example a stand-in used in a test), there is no
    # measurement to turn into a feature row, so we stop here.
    if report is None:
        return
    angles, analysis = report

    # Step 5: turn the SAME angles and the SAME analysis into one feature
    # row. Nothing is measured again. An angle rejected by the visibility
    # check is still None here, so it becomes None in the row too.
    # frame_number is None: this script handles one frame and has no frame
    # counter, and we do not invent one.
    features = extract_features(
        elbow_angle=angles["elbow_angle"],
        hip_angle=angles["hip_angle"],
        shoulder_angle=angles["shoulder_angle"],
        analysis=analysis,
        frame_number=None,
    )
    print_feature_row(features)


if __name__ == "__main__":
    main()
