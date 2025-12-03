# Face Liveness Detection App

A streamlined, efficient face liveness detection application for vKYC.

## Features

- **Real-time Face Detection**: Ensures face is present and properly positioned.
- **Liveness Verification**: Uses eye blink detection (EAR) to verify live presence.
- **Visibility Checks**:
  - Ensures forehead and ears are visible.
  - Detects if face is too close, too far, or not looking straight.
- **Clear UI**:
  - Color-coded status bar (Green=Verified, Red=Warning, Orange=Pending).
  - Clear text instructions (e.g., "Face Not Fully Visible", "Look Straight").

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python liveness_app.py
```

## How it Works

1. **Align Face**: Position your face within the frame. The oval guide will turn green when verified.
2. **Visibility Check**: The app checks if your forehead and ears are visible. If obstructed (e.g., by a mask or poor framing), it warns "Face Not Fully Visible".
3. **Blink to Verify**: Blink your eyes naturally. The system counts blinks to verify liveness.
4. **Success**: Status changes to "LIVENESS VERIFIED".
