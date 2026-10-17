# 🚀 KineticEye

> Stop dreaming and starting being Kinetic in becoming a fitter you !!!

---

## 📌 Table of Contents
- [About the Project](#-about-the-project)
- [Features](#-features)
- [Getting Started](#-getting-started)
- [How to Use](#-how-to-use)
- [Collaborators](#-collaborators)

---

## 📋 About the Project

KineticEye is an AI-powered sports video analytics and exercise-form analysis project developed as part of a first-year Artificial Intelligence and Machine Learning (AIML) mini project. The project focuses on analyzing push-up movements using computer vision to make exercise tracking more informative and measurable. It uses Python, OpenCV, MediaPipe and NumPy to detect body landmarks, calculate elbow, shoulder and hip angles, validate landmark visibility, and evaluate exercise form using rule-based analysis. A repetition-counting algorithm tracks movement phases and counts a repetition only after a complete TOP → BOTTOM → TOP cycle, helping reduce incorrect counts caused by incomplete movements. The system also extracts structured, frame-by-frame features and exports them to CSV files for further inspection and data analysis.

The KineticEye web interface is designed to present exercise insights through an intuitive dashboard, interactive visualizations and analysis reports, making technical results easier to understand. Unlike conventional fitness websites that primarily offer workout routines, exercise demonstrations or manually maintained progress trackers, KineticEye aims to connect movement analysis with measurable data. It considers individual joint angles, landmark reliability, movement phases and form feedback rather than relying solely on repetition totals. Its modular architecture also makes the analysis process easier to test, maintain and extend. The project is being developed with future integration, dataset analysis and machine-learning capabilities in mind. By combining computer vision, mathematical analysis and an accessible user interface, KineticEye aims to bridge the gap between exercise tracking and data-driven fitness insights while providing a practical foundation for further development in intelligent sports analytics.
## ✨ Features

- **Feature 1: AI-Powered Pose Estimation** — Uses MediaPipe to detect and track important body landmarks, including shoulders, elbows, hips, knees, and ankles.

- **Feature 2: Real-Time Skeleton Overlay** — Displays a digital skeleton over the user's body to visualize movement and joint positions.

- **Feature 3: Joint-Angle Calculation** — Uses Python, NumPy, and geometric mathematics to calculate joint angles and analyze exercise movements.

- **Feature 4: Automatic Rep Counting** — Uses a state machine to count completed repetitions while reducing half-rep errors and double counting.

- **Feature 5: Intelligent Form Analysis** — Evaluates selected posture indicators and identifies potential issues using configurable angle thresholds and movement rules.

- **Feature 6: Real-Time Feedback** — Provides guidance such as "Keep your back straighter" or "Complete the full range of motion."

- **Feature 7: Exercise Support** — Initially focuses on push-ups, with plans to extend the system to squats, bicep curls, and other exercises.

- **Feature 8: Interactive Fitness Dashboard** — Displays the exercise video, repetition count, joint angles, form status, and workout progress in a modern interface.

- **Feature 9: Workout Analytics** — Aims to track repetition quality, movement consistency, and workout progress over time.

- **Feature 10: Machine Learning Integration** — Plans to use extracted pose and movement features to train and evaluate models for exercise or form classification.

## 🚀 Getting Started
Tell your co-workers how to set up the project on their local computer.


## 💻 How to Use

The **Sports Video Analytics & Action Counter** prototype demonstrates how computer vision and geometric analysis can be used to evaluate exercise movements, count repetitions, and provide posture-related feedback.

### 🏋️ Using the Application

1. **Select an Exercise:** Choose a supported exercise, such as push-ups.
2. **Provide Video Input:** Use the webcam or upload a recorded exercise video, depending on the available prototype functionality.
3. **Start Analysis:** Initiate the exercise-analysis process to examine body landmarks and movement patterns.
4. **Monitor Joint Angles:** View calculated joint angles, such as the elbow and hip angles, to understand body positioning.
5. **Track Repetitions:** Observe the repetition counter as the system identifies completed movement cycles.
6. **Review Form Feedback:** Check the displayed form status and feedback messages for potential posture improvements.
7. **Evaluate Results:** Review the repetition count, movement analysis, and available workout statistics.

### 🧪 Prototype Testing

- Test the application with different exercise videos and movement speeds.
- Compare the calculated joint angles with expected values.
- Verify that incomplete movements are not counted as full repetitions.
- Observe how missing landmarks and incorrect posture affect the analysis.
- Use the results to identify areas for improving accuracy and reliability.

> **Note:** This is an evolving prototype. The availability and accuracy of individual features depend on the current level of implementation and integration. Additional exercises, machine-learning-based classification, personalized feedback, and advanced workout analytics are planned enhancements.
