# ORIEL: Intelligent Assistance System Based on Computer Vision

<p align="center">
    <img src="https://github.com/Nasor2/oriel/blob/main/oriel.jpeg" alt="ORIEL Logo Banner" width="100%"/>
</p>

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)
![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-ff8f00?logo=tensorflow&logoColor=white)

**ORIEL** is a technological assistance prototype designed to enhance the autonomy of visually impaired individuals. Leveraging advanced **Artificial Intelligence**, **Computer Vision**, and **Natural Language Processing (NLP)** techniques, the system interprets the environment in real-time and provides auditory feedback regarding obstacles, risks, and walkable paths.

This project was developed within an academic research context, emphasizing the exploration of **sensor fusion** and **event-driven architectures** for assisted navigation.

> [!NOTE]
> The system is currently configured for real-time operation using **CPU inference** for portability, with models optimized for speed (YOLOv8n, MiDaS DPT-Hybrid).

---

## 🚀 Key Features

### 👁️ Intelligent Alert System (SAI)
* **Object Detection:** Identifies and classifies dynamic obstacles (vehicles, pedestrians, animals) using **YOLOv8**.
* **Hybrid Depth Estimation:** Combines **MiDaS** (Monocular Depth) with a mathematical **Pinhole Camera Model** to calculate stable metric distances.
* **Contextual Risk Assessment:** Prioritizes alerts based on trajectory, velocity, and distance using a proprietary risk scoring function.
* **Temporal Tracking:** Employs a `TemporalTracker` to maintain object identity across frames, enabling true velocity analysis.

### 🛣️ Footpath Analysis
* **Sidewalk Segmentation:** Detects and analyzes walkable areas using a specialized semantic segmentation model (`.keras`).
* **Safety Validation (SODD-CA):** Verifies path continuity, confidence, and **coverage area** before issuing guidance.

> [!IMPORTANT]
> The depth calculation is a **Hybrid Sensor Fusion** solution: it combines the object size from YOLO (Geometry) with the relative depth map from MiDaS (AI) to achieve greater metric accuracy than either method alone.

---

## 🏛️ System Architecture

ORIEL utilizes a robust **Event-Driven Architecture (EDA)** to decouple the high-demand Visual processing from the interactive Audio components.

| Component | Description |
| :--- | :--- |
| **EventBus** | Central nervous system for asynchronous communication (Pub/Sub). |
| **Orchestrator** | Manages the global state (`CoreContext`), handling command intents and coordinating subsystem activation/deactivation. |
| **Visual Pipelines** | Independent threads for SAI (Alerts) and Footpath (Navigation). |
| **Output (TTS)** | Manages audio output priorities, preventing new speech commands from interrupting critical safety alerts. |

> [!TIP]
> This Event-Driven design makes the project highly **scalable** and **maintainable**, as new features (e.g., OCR, GPS integration) can be added by simply subscribing to existing events.

---

## 📋 Prerequisites

* **Python 3.9+**
* **Hardware:**
    * Camera: USB Webcam or Smartphone (via DroidCam/IP Webcam is highly recommended).
    * Microphone: Integrated or external.

### Dependencies
The project relies on the libraries listed in `requirements.txt`: `torch`, `tensorflow`, `ultralytics`, `scikit-learn`, `spacy`, `pyttsx3`, etc.

---

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/oriel-system.git](https://github.com/your-username/oriel-system.git)
    cd oriel-system
    ```

2.  **Create and activate a virtual environment (Essential):**
    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate
    # Linux/Mac: source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLP resources:**
    ```bash
    python -m spacy download es_core_news_sm
    ```

> [!CAUTION]
> **Model Configuration:** You must manually place the weight files (`yolov8n.pt`, `dpt_hybrid_384.pt`, `qpulm_pruned_final.keras`) into the `models/` directory for the system to boot successfully.

---

## ▶️ Execution

### 1. Configure Video Source
Edit the `main.py` file to set your camera input source.

* **Option A: DroidCam (USB/Wi-Fi) or Webcam (Recommended for Demos):**
    ```python
    video_manager = VideoManager(source=0) 
    ```
* **Option B: IP Webcam (Wi-Fi):**
    ```python
    # Ensure the IP matches the stream URL provided by your phone app
    video_manager = VideoManager(source="[http://192.168.1.](http://192.168.1.)XX:8080/video")
    ```

> [!WARNING]
> **Live Demo Risk:** The IP address is often hardcoded in the example. Ensure the IP address matches your local Wi-Fi network's configuration *before* any live demonstration, or use `source=0` with a stable USB connection.

### 2. Start the System
Run the main script:
```bash
python main.py
```

---

## 🎤 Usage Guide (Voice Commands)

The system is activated by the *wakeword* **"Lucy"** followed by a command (currently configured for Spanish commands only):

| Intent | Spanish Command Example | Action |
| :--- | :--- | :--- |
| **Activate Alerts** | *"Lucy... activar sistema de alerta"* | Initiates continuous monitoring and risk assessment. |
| **Check Path** | *"Lucy... ¿por dónde camino?"* | Analyzes path safety and provides directional guidance. |
| **Deactivate Alerts** | *"Lucy... desactivar alertas"* | Stops the main alert loop to conserve resources. |
| **System Status** | *"Lucy... estado del sistema"* | Confirms that ORIEL is operational and reports any locked resources. |

> [!TIP]
> **Activation:** Always wait for the system's auditory confirmation ("Sí, te escucho") after saying the wakeword to ensure the microphone is open for your command.

---

## 🌟 Technical Highlights

* **Scalability:** The **Event-Driven Architecture** ensures that both the SAI and Footpath systems can scale their processing threads independently.
* **Maintainability:** The use of distinct, single-responsibility modules (e.g., `PathAnalyzer`, `CommandProcessor`) simplifies debugging and future updates.
* **Safety Protocol:** Implements a strict **Safety Validation** in the Footpath system, returning "No-Path Detected" if sidewalk coverage drops below 5%.

> [!IMPORTANT]
> **Sensor Fusion:** The system's robustness stems from its **Hybrid Depth Estimation**, which combines machine learning output (MiDaS) with classical projective geometry (Pinhole Model) to achieve reliable, metric distance reporting.

---

## 📂 Project Structure

The codebase is organized into logical layers, reflecting the system's architecture:

```text
oriel/
├── core/                   # Orchestrator, EventBus, CoreContext (System State)
├── input/                  # Audio Pipeline (STT/Wakeword), Video Manager
├── visual/
│   ├── sai/                # Intelligent Alert System (YOLO, MiDaS, Risk Logic)
│   └── footpath/           # Path Analysis (Segmentation, SODD-CA Logic)
├── output/                 # TTS Manager (Text-to-Speech)
├── models/                 # AI Model Weights (.pt, .keras)
├── main.py                 # System Entry Point
└── requirements.txt        # Python Dependencies
