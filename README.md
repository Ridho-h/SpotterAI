# SpotterAI 🏋️‍♂️

*Real-time AI-powered fitness trainer that sees your workouts*

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![TensorFlow 2.13](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Google-blue.svg)](https://mediapipe.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI Status](https://github.com/Ridho-h/SpotterAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Ridho-h/SpotterAI/actions)

![Demo](assets/demo.gif)
*Demo GIF coming soon — run the app to see it live!*

## Overview
SpotterAI is an intelligent, real-time fitness trainer that uses your webcam to analyze body poses, automatically classify exercises, and count repetitions. By leveraging Google's MediaPipe for lightweight pose estimation and a custom Bidirectional LSTM model with an Attention mechanism, it delivers accurate, on-device AI analysis directly in your browser. 

**Key Highlights**
- ⚡ **Real-time inference** directly in the browser via WebRTC
- 🧠 **Advanced deep learning** using Bi-LSTM + Attention for sequence modeling
- 🔒 **Privacy-first**, processing video streams locally without saving to the cloud
- 🚀 **Deploy anywhere** with seamless Docker integration

## Features
- 🎯 **Real-time exercise classification** (curl, press, squat)
- 🔢 **Automatic rep counting** with precise stage detection (up/down)
- 🧠 **Custom Bi-LSTM + Attention** deep learning model
- 📹 **Browser-based webcam access** via WebRTC (no install needed)
- 🐳 **Docker support** for one-command deployment
- ⚙️ **Configurable confidence thresholds** for robust detection

## Architecture
```mermaid
flowchart LR
    A[Webcam Feed] --> B[MediaPipe Pose]
    B --> C[33 Keypoints x4]
    C --> D[30-Frame Sequence Buffer]
    D --> E[Bi-LSTM + Attention]
    E --> F[Exercise Classifier]
    F --> G[Rep Counter]
    G --> H[Live Overlay]
```
The architecture leverages a streaming pipeline where individual frames are processed by MediaPipe to extract 33 body keypoints (x, y, z, and visibility). These keypoints are buffered into a rolling window of 30 frames. The structured sequence data is fed into a custom **Bidirectional LSTM + Luong Attention mechanism** to classify the ongoing movement dynamically, while a geometric tracker counts reps based on joint angles.

## Model Details
SpotterAI's core intelligence relies on sequence modeling of skeletal data. The Attention mechanism drastically improves performance by allowing the model to focus on the most critical frames of a movement (e.g., the bottom of a squat).

| Feature | Baseline LSTM | Bi-LSTM + Attention |
|---------|---------------|---------------------|
| **File Size** | ~9 MB | ~100 MB |
| **Accuracy** | Good | Excellent |
| **Context** | Single direction | Bidirectional context with learned frame focus |

- **Input Shape**: `(batch, 30, 132)` (30 frames, 33 landmarks × 4 metrics)
- **Architecture**: `Input -> Bi-LSTM (256 units) -> Luong Multiplicative Attention -> Dense -> Softmax`

## Quick Start

### Docker (Recommended)
```bash
git clone https://github.com/Ridho-h/SpotterAI.git
cd SpotterAI
# Download model weights (see Models section)
docker-compose up
# Open http://localhost:8501
```

### Local Setup
```bash
git clone https://github.com/Ridho-h/SpotterAI.git
cd SpotterAI
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

## Models
Due to GitHub file size limits, the trained model files are not included in the repository.
1. Download the models from the [Releases page](https://github.com/Ridho-h/SpotterAI/releases).
2. Place the `.h5` files in the `models/` directory.
   - `models/LSTM.h5`
   - `models/LSTM_Attention.h5`

## How It Works
1. **Webcam Capture**: The browser captures video frames and streams them via WebRTC.
2. **Pose Extraction**: MediaPipe extracts 33 body landmarks, giving x, y, z coordinates and visibility scores.
3. **Sequence Buffering**: Keypoints are flattened and stacked into a rolling 30-frame sequence buffer.
4. **Classification**: Once the buffer is full, the Bi-LSTM+Attention model predicts the current exercise probability.
5. **Rep Counting**: A dedicated state machine uses angle geometry (e.g., elbow/knee angles) to detect up/down stages and counts reps.
6. **Visualization**: Predictions, counters, and pose skeletons are overlaid on the video stream in real-time.

## Development
To run tests and linting:
```bash
pip install -r requirements-dev.txt
pytest tests/ -v
ruff check .
```

## Project Structure
```text
SpotterAI/
├── app.py                  # Streamlit entrypoint
├── spotter/
│   ├── model.py            # Bi-LSTM + Attention architecture
│   ├── pose.py             # MediaPipe keypoint extraction
│   ├── tracker.py          # Rep counting state machine
│   └── visualizer.py       # Drawing helpers
├── tests/                  # Unit tests
├── notebooks/              # Training notebook
├── models/                 # Model weights (not in repo)
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/      # CI/CD
```

## Roadmap
- [ ] Add more exercises (deadlift, lunges, push-ups, pull-ups)
- [ ] Workout session history and progress tracking
- [ ] Form correction feedback using angle thresholds
- [ ] Mobile-responsive UI
- [ ] Export workout data to CSV/JSON
- [ ] Deploy to Hugging Face Spaces for public demo

## License & Author
- Released under the [MIT License](https://opensource.org/licenses/MIT).
- Built by [Ridho-h](https://github.com/Ridho-h).
- ⭐ If you find this useful, please give it a star!
