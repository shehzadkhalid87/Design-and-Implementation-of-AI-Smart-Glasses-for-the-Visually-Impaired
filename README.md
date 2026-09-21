# AI Smart Glasses Using ESP32-S3 and Edge Impulse

## 1. Project Overview

This project implements an AI-assisted smart-glasses prototype for visual
object recognition and spoken audio feedback. An OV3660 camera captures the
scene, an Edge Impulse FOMO model identifies objects, and a YX5300 MP3 module
plays the corresponding audio announcement.

The system also provides a local Wi-Fi web interface for viewing the latest
camera frame and detection information.

## 2. System Objectives

- Capture images using an OV3660 camera.
- Detect trained object classes on an ESP32-S3.
- Provide audio feedback for recognized objects.
- Display camera and detection information through a local web page.
- Operate as a self-contained embedded system without cloud connectivity during
  normal operation.

## 3. Recognized Object Classes

The current Edge Impulse FOMO model contains five classes:

1. `car`
2. `chair`
3. `door`
4. `person`
5. `stop sign`

The model input is 96 x 96 pixels. The camera captures frames at 320 x 240
pixels, which are prepared for the model by the firmware.

## 4. System Architecture

The system consists of the following functional modules:

1. **Image acquisition**  
   The OV3660 camera captures JPEG frames.

2. **Image preprocessing**  
   The firmware decodes the selected frame, converts it to RGB format, and
   prepares the 96 x 96 model input.

3. **Machine-learning inference**  
   The Edge Impulse FOMO model evaluates the image and returns detected classes,
   confidence values, and approximate object locations.

4. **Detection selection**  
   The firmware selects the strongest valid detection and updates the current
   web status.

5. **Audio feedback**  
   The selected class is mapped to an MP3 track and played through the YX5300
   module.

6. **Web monitoring**  
   The ESP32-S3 creates a local access point and serves the camera frame and
   detection status.

## 5. Hardware Components

- ESP32-S3 development board with PSRAM
- OV3660 camera module
- YX5300 MP3 player module
- TF card for MP3 files
- Speaker or audio amplifier
- Power supply and connecting wires

## 6. Hardware Connections

### 6.1 OV3660 Camera

The camera pin assignments are defined in `src/main.cpp`.

| Camera signal | ESP32-S3 GPIO |
|---|---:|
| D0 | 11 |
| D1 | 9 |
| D2 | 8 |
| D3 | 10 |
| D4 | 12 |
| D5 | 18 |
| D6 | 17 |
| D7 | 16 |
| XCLK | 15 |
| PCLK | 13 |
| VSYNC | 6 |
| HREF | 7 |
| SCCB SDA | 4 |
| SCCB SCL | 5 |

### 6.2 YX5300 Audio Module

| YX5300 signal | ESP32-S3 GPIO |
|---|---:|
| RX | 2 |
| TX | 1 |

The ESP32-S3, YX5300, and audio power circuit use a common ground.

## 7. Audio File Organization

The MP3 files are stored on the YX5300 TF card in folder `01`.

| Object class | Track file |
|---|---|
| car | `01/001.mp3` |
| chair | `01/002.mp3` |
| door | `01/003.mp3` |
| person | `01/004.mp3` |
| stop sign | `01/005.mp3` |

The mapping is implemented in the `audioMap` table in `src/main.cpp`.

## 8. Software and Machine-Learning Components

- PlatformIO
- Arduino framework for ESP32
- Edge Impulse FOMO object-detection model
- Edge Impulse inference SDK
- ESP-NN acceleration support
- YX5300 ESP32 library
- Wi-Fi and WebServer libraries

The current model export uses the Edge Impulse compiled TensorFlow Lite
runtime. The firmware is configured for ESP32-S3 PSRAM and optimized compiler
options.

## 9. Folder Structure

```text
eps32-s3-devkittc-1/
|
|-- README.md
|-- platformio.ini
|-- partitions_pico.csv
|
|-- src/
|   `-- main.cpp
|
|-- model-parameters/
|   |-- model_metadata.h
|   `-- model_variables.h
|
|-- tflite-model/
|   |-- tflite_learn_1117170_3_compiled.cpp
|   |-- tflite_learn_1117170_3_compiled.h
|   `-- trained_model_ops_define.h
|
|-- edge-impulse-sdk/
|   |-- classifier/
|   |-- dsp/
|   |-- tensorflow/
|   |-- CMSIS/
|   `-- porting/
|
|-- include/
|   `-- README
|
|-- lib/
|   `-- README
|
`-- .pio/
    `-- PlatformIO build files and downloaded libraries
```

The `model-parameters`, `tflite-model`, and `edge-impulse-sdk` directories are
the generated Edge Impulse deployment files used by the firmware.

## 10. Main Firmware File

`src/main.cpp` contains:

- Camera initialization and configuration
- JPEG frame capture
- Image decoding and resizing
- Edge Impulse signal preparation
- FOMO inference
- Detection result processing
- Audio track selection
- Wi-Fi access-point setup
- Web page and JSON status handlers
- Runtime performance reporting

## 11. Important Firmware Parameters

The main operational parameters are defined near the beginning of
`src/main.cpp`.

| Parameter | Purpose | Current value |
|---|---|---:|
| `DETECTION_THRESHOLD` | Minimum confidence for announcements | `0.50` |
| `AUDIO_COOLDOWN_MS` | Delay between repeated announcements | `3000 ms` |
| `INFERENCE_INTERVAL_MS` | Minimum interval between inference attempts | `250 ms` |
| `CAMERA_WIDTH` | Camera frame width | `320` |
| `CAMERA_HEIGHT` | Camera frame height | `240` |
| `MODEL_WIDTH` | Model input width | `96` |
| `MODEL_HEIGHT` | Model input height | `96` |

## 12. Web Interface

At startup, the ESP32-S3 creates the following Wi-Fi access point:

```text
SSID: AISmartGlasses
Password: smartglass
```

Connect a computer or mobile device to this network and open:

```text
http://192.168.4.1/
```

The web service provides:

| Address | Function |
|---|---|
| `/` | Camera and detection dashboard |
| `/snapshot.jpg` | Latest JPEG camera frame |
| `/status` | JSON detection and timing data |

The JSON status includes the detected class, confidence, number of boxes,
selected box coordinates, inference time, and camera-frame age.

## 13. Build and Upload Procedure

From the project directory:

```bash
~/.platformio/penv/bin/platformio run
~/.platformio/penv/bin/platformio run -t upload
```

The serial monitor uses 115200 baud:

```bash
~/.platformio/penv/bin/platformio device monitor -b 115200
```

## 14. Runtime Sequence

1. The ESP32-S3 initializes the camera.
2. The Edge Impulse classifier is initialized.
3. The Wi-Fi access point and web server are started.
4. The YX5300 audio module is initialized.
5. The camera captures frames continuously.
6. Selected frames are prepared for FOMO inference.
7. The strongest detection is stored as the current result.
8. The matching MP3 track is played when the confidence threshold and cooldown
   conditions are satisfied.
9. The web interface displays the latest frame and detection status.

## 15. Performance Measurement

The firmware prints timing information in the following format:

```text
[TIME] grab=... decode=... resize=... DSP=... NN=... total=...
```

These values represent camera capture, JPEG decoding, image resizing, Edge
Impulse DSP, neural-network inference, and total processing time.

The web server operates independently from the main inference loop so that the
latest camera frame remains available while the classifier is processing.

## 16. Model Development

The model is trained in Edge Impulse using the FOMO object-detection learning
block. The training dataset should contain varied examples for every class,
including different distances, lighting conditions, viewpoints, backgrounds,
and object sizes.

Model files downloaded from Edge Impulse must be extracted into the project
root while preserving the expected directory names:

```text
edge-impulse-sdk/
model-parameters/
tflite-model/
```

The model labels and input dimensions must match the firmware configuration.

## 17. Project Reproducibility

To reproduce the project:

1. Install PlatformIO and the ESP32 platform.
2. Obtain the project source files.
3. Place the matching Edge Impulse export directories at the project root.
4. Prepare the audio files on the YX5300 TF card.
5. Connect the camera and audio module according to the pin tables.
6. Build and upload the firmware.
7. Open the serial monitor to observe startup and timing information.
8. Connect to the `AISmartGlasses` Wi-Fi network to view the web interface.

