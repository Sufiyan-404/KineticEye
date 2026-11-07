# KineticEye 

link to the prototype website : https://sable-modal-29074405.figma.site/

## About the Project

KineticEye is an AI-powered sports video analytics and exercise-form analysis project developed as a first-year Artificial Intelligence and Machine Learning (AIML) mini project.

The project currently focuses on **push-up analysis**, using Python, OpenCV, MediaPipe and NumPy to detect body landmarks, calculate elbow, shoulder and hip angles, validate landmark visibility, and evaluate exercise form through rule-based analysis.

A repetition counter recognizes complete **TOP → BOTTOM → TOP** movement cycles, reducing incorrect counts from incomplete repetitions. The system also extracts frame-by-frame movement features into CSV files for further inspection and analysis.

KineticEye aims to go beyond traditional fitness trackers by combining repetition counting with measurable joint angles, movement phases, landmark reliability and form feedback. Its developing web interface aims to make these insights accessible through interactive visualizations and workout reports.

## Features

1. **AI-Powered Pose Estimation:** Uses MediaPipe to identify body landmarks, including shoulders, elbows, hips, knees and ankles.
2. **Real-Time Skeleton Overlay:** Visualizes detected body joints and connections in supported camera demonstrations.
3. **Joint-Angle Calculation:** Uses geometric mathematics and NumPy to calculate elbow, hip and shoulder angles.
4. **Automatic Rep Counting:** Counts completed push-ups using a state-based TOP → BOTTOM → TOP sequence.
5. **Intelligent Form Analysis:** Evaluates selected posture indicators using angle thresholds and movement rules.
6. **Real-Time Feedback:** Rule-based posture guidance, such as maintaining a straighter back or improving movement depth, with live presentation planned.
7. **Exercise Support:** Currently focused on push-ups; squats, bicep curls and other exercises are planned.
8. **Interactive Fitness Dashboard:** A developing interface for exercise videos, repetition counts, joint angles, form status and progress.
9. **Workout Analytics:** Frame-level CSV analysis is implemented; long-term workout progress and repetition-quality analytics are planned.
10. **Machine Learning Integration:** Future classification of exercises and form quality using extracted movement features.
11. **Face Authentication System:** Planned identity verification to check whether the registered user is performing the exercise.

## Technology Stack

**Python, OpenCV, MediaPipe, NumPy, Pandas, Matplotlib, Scikit-learn and Streamlit** are part of the project's development stack or planned extensions. The current core video-analysis pipeline primarily uses Python, OpenCV, MediaPipe and NumPy.

## Installation & Setup

The following instructions are intended for Windows laptops.

### Step 1: Prerequisites

Install **Python 3.12** and Git. A recorded push-up video is required for the video-analysis demonstration; a webcam is useful for camera testing.

### Step 2: Download the Project

Open PowerShell or the VS Code terminal:

```powershell
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the example URL with the actual KineticEye GitHub repository address.

### Step 3: Create a Virtual Environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

If the repository includes `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

Otherwise, install the core video-analysis dependencies:

```powershell
python -m pip install opencv-python mediapipe numpy pytest
```

### Step 5: Run Video Analysis

Keep the original `person1/src` and `person2/src` folder structure intact.

From the project root:

```powershell
cd person2\src
python video_analysis_demo.py "C:\Videos\pushups.mp4"
```

Replace the example path with your own video. For best results, record from the side with the left shoulder, elbow, wrist, hip and ankle visible.

The program processes the frames, calculates angles, assesses form, counts repetitions and generates a CSV file named `pushups_features.csv` in the current working directory.

### Step 6: Run Tests

From `person2/src`:

```powershell
python -m pytest -v
```

This runs the available automated tests. Real-video validation is still recommended.

## How to Use

1. **Select an Exercise:** Push-ups are the current exercise focus.
2. **Provide Video Input:** Use a recorded video for the Python pipeline; webcam demonstrations depend on the available implementation.
3. **Start Analysis:** Run the video-analysis command.
4. **Monitor Joint Angles:** Review calculated elbow, shoulder and hip angles in the generated results.
5. **Track Repetitions:** Check the completed repetition count displayed in the terminal.
6. **Review Form Feedback:** Inspect the form-status and feedback fields in the CSV.
7. **Evaluate Results:** Use the extracted features to study movement consistency and identify potential improvements.

### Prototype Testing

Test different push-up videos and movement speeds, compare calculated angles with expected values, verify that incomplete repetitions are not counted, and observe how missing landmarks or incorrect posture affect results.

> **Note:** KineticEye is an evolving prototype. The Python pipeline and web dashboard are still being integrated, and some features remain planned.

## Future Improvements

Expand to squats, bicep curls and other exercises; implement machine-learning classification, personalized feedback, advanced workout analytics, improved dashboard integration and face-based user authentication.

## Acknowledgements & AI-Assisted Development

KineticEye was developed with assistance from **Claude (Anthropic)** for drafting and refining code, debugging, improving test coverage and understanding technical concepts.

The project team guided requirements, integration, implementation decisions and validation. AI-assisted development has supported the team's learning in computer vision, software engineering and artificial intelligence.
