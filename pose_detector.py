import cv2
import mediapipe as mp


# -----------------------------
# MediaPipe setup
# -----------------------------

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# Landmarks we want to send to Person 2
LANDMARK_NAMES = {
    mp_pose.PoseLandmark.LEFT_SHOULDER: "left_shoulder",
    mp_pose.PoseLandmark.LEFT_ELBOW: "left_elbow",
    mp_pose.PoseLandmark.LEFT_WRIST: "left_wrist",
    mp_pose.PoseLandmark.LEFT_HIP: "left_hip",
    mp_pose.PoseLandmark.LEFT_KNEE: "left_knee",
    mp_pose.PoseLandmark.LEFT_ANKLE: "left_ankle",

    mp_pose.PoseLandmark.RIGHT_SHOULDER: "right_shoulder",
    mp_pose.PoseLandmark.RIGHT_ELBOW: "right_elbow",
    mp_pose.PoseLandmark.RIGHT_WRIST: "right_wrist",
    mp_pose.PoseLandmark.RIGHT_HIP: "right_hip",
    mp_pose.PoseLandmark.RIGHT_KNEE: "right_knee",
    mp_pose.PoseLandmark.RIGHT_ANKLE: "right_ankle",
}


def get_landmarks(results):
    """
    Convert MediaPipe results into a dictionary
    that Person 2 can use.
    """

    landmarks = {}

    if results.pose_landmarks is None:
        return landmarks

    for landmark_id, name in LANDMARK_NAMES.items():
        landmark = results.pose_landmarks.landmark[landmark_id]

        landmarks[name] = {
            "x": landmark.x,
            "y": landmark.y,
            "visibility": landmark.visibility
        }

    return landmarks


# -----------------------------
# Webcam execution
# -----------------------------

def main():
    cap = cv2.VideoCapture(0)

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            while cap.isOpened():
                success, frame = cap.read()

                if not success:
                    print("Could not read camera frame")
                    break

                # OpenCV uses BGR; MediaPipe expects RGB
                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                # Process frame with MediaPipe
                results = pose.process(rgb_frame)

                # Draw skeleton
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )

                    # Get landmarks for Person 2
                    landmarks = get_landmarks(results)

                    # Uncomment to print extracted landmarks
                    # print(landmarks)

                # Display webcam
                cv2.imshow("Pose Detection", frame)

                # Press Q to quit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
