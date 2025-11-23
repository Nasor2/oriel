# ORIEL: Intelligent Assistance System Based on Computer Vision

**ORIEL** is a technological assistance prototype designed to enhance the autonomy of visually impaired individuals. Leveraging advanced **Artificial Intelligence**, **Computer Vision**, and **Natural Language Processing (NLP)** techniques, the system interprets the environment in real-time and provides auditory feedback regarding obstacles, risks, and walkable paths.

This project was developed within an academic research context to explore assisted navigation systems and sensor fusion strategies.

---

## 🚀 Key Features

### 👁️ Intelligent Alert System (SAI)
* **Object Detection:** Identifies and classifies dynamic obstacles (vehicles, pedestrians, animals) using **YOLOv8**.
* **Hybrid Depth Estimation:** Combines **MiDaS (Monocular Depth Estimation)** with a mathematical **Pinhole Camera Model** to correct perspective distortions and calculate precise metric distances.
* **Contextual Risk Assessment:** Analyzes object trajectory, velocity, and proximity to issue prioritized alerts (e.g., "Imminent Danger" vs. "Caution").
* **Temporal Tracking:** Implements a tracking system (`TemporalTracker`) that "remembers" objects across frames to analyze behavioral trends over time.

### 🛣️ Footpath Analysis
* **Sidewalk Segmentation:** Detects walkable areas using a custom semantic segmentation model (`.keras`).
* **Safety Validation (SODD-CA):** A proprietary algorithm that verifies path continuity, confidence, and coverage before suggesting movement.
* **No-Path Detection:** Intelligent capability to inform the user if no sidewalk is detected or if the surface is unsafe for traversal.

### 🗣️ Natural Voice Interface
* **Voice Control:** Activation via *wakeword* ("Lucy") and processing of complex intents using **spaCy**.
* **Prioritized Auditory Feedback:** Text-to-Speech (TTS) system with queue management and priority handling, capable of interrupting standard messages if a safety alert occurs.

---

## 🛠️ System Architecture

ORIEL utilizes an **Event-Driven Architecture** to decouple subsystems and ensure real-time performance:

1.  **Input Layer:** Handles Audio (Microphone) and Video (IP Camera/Webcam/DroidCam).
2.  **Core Layer:** A central `Orchestrator` manages global state (`CoreContext`) and coordinates communication via an asynchronous `EventBus`.
3.  **Visual Layer:** Independent inference engines (SAI and Footpath) process frames and publish analysis events.
4.  **Output Layer:** TTS Manager that verbalizes notifications to the user, handling mutex locks to prevent system self-feedback.

---

## 📋 Prerequisites

* **Python 3.9+**
* **Hardware:**
    * Camera: USB Webcam or Smartphone (via DroidCam/IP Webcam).
    * Microphone: Integrated or external.
    * CPU: Intel i5/Ryzen 5 or higher recommended (system optimized for CPU inference).

### Dependencies
The project relies on the following main libraries (see `requirements.txt`):
* `torch`, `ultralytics`, `timm` (Computer Vision)
* `tensorflow` (Segmentation Model)
* `opencv-python` (Image Processing)
* `SpeechRecognition`, `pyaudio` (Audio Input)
* `pyttsx3` (Audio Output)
* `spacy` (NLP)
* `scikit-learn` (Regression analysis for direction)

---

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/oriel-system.git](https://github.com/your-username/oriel-system.git)
    cd oriel-system
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLP resources:**
    ```bash
    python -m spacy download es_core_news_sm
    ```

5.  **Configure Models:**
    Ensure the following weight files are placed in the `models/` directory:
    * `yolov8n.pt` (Object Detection).
    * `dpt_hybrid_384.pt` (Depth Estimation).
    * `qpulm_pruned_final.keras` (Path Segmentation).

---

## ▶️ Execution

### 1. Configure Video Source
Edit `main.py` to select your camera input.

* **Option A: DroidCam (USB/Wi-Fi) or Webcam (Recommended):**
    ```python
    video_manager = VideoManager(source=0)
    ```
* **Option B: IP Webcam (Wi-Fi):**
    ```python
    video_manager = VideoManager(source="[http://192.168.1.](http://192.168.1.)XX:8080/video")
    ```

### 2. Start the System
Run the main script:
```bash
python main.py
