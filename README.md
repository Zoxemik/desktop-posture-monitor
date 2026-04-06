# Posture Guard

Posture Guard is a lightweight desktop application that monitors sitting posture using a webcam and MediaPipe Pose.

The application does not enforce a single "correct" posture. Instead, it builds a personalized baseline and detects meaningful deviations and prolonged static positions.

## Overview

The system works in four main stages:

1. Calibration  
   At startup, the application records the user's natural sitting posture and builds a baseline using median values.

2. Tracking  
   The webcam feed is processed with MediaPipe Pose to extract body landmarks.

3. Metrics  
   The system computes posture-related metrics such as:
   - head forward distance
   - torso angle
   - neck compression
   - shoulder tilt
   - head tilt
   - distance from screen (estimated)

4. Evaluation  
   Current posture is compared against the baseline and classified into:
   - green (neutral)
   - yellow (drifting)
   - red (overload)

The application also tracks movement and detects prolonged stillness.

## Features

- Personalized baseline calibration
- Real-time posture evaluation
- Movement detection and stillness reminders
- Configurable thresholds and alert timing
- Desktop notifications, sound alerts, and overlay indicator
- System tray integration
- Local session statistics

## Requirements

Install dependencies:

pip install opencv-python mediapipe pystray Pillow win10toast plyer

Additional requirements:
- Webcam
- Windows (recommended for full notification support)

## Running

Run the application:

python main.py

The app runs in the background. The preview window can be toggled from the tray.

## Configuration

Configuration file:

%LOCALAPPDATA%\PostureGuard\data\config.json

Main configuration areas:
- camera settings (resolution, FPS)
- detection and smoothing parameters
- posture thresholds
- alert timing and cooldowns
- movement detection sensitivity
- notification settings

Baseline data is stored separately and reused between runs.

## Data

Session data is stored locally in JSON format and includes:
- posture alert count
- stillness reminders
- time spent in bad posture
- reposition count

## Notes

- Best results require head and shoulders to be visible
- Tracking quality depends on lighting and camera placement
- Recalibration is recommended after changing desk or seating setup

## Status

The application includes a complete monitoring engine and notification system.

Future improvements may include:
- graphical settings UI
- extended analytics
- improved posture classification
