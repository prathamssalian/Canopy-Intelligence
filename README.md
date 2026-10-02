# 🌴 Arecanut Yellow Leaf Disease Detection

An AI-powered system for detecting **Yellow Leaf Disease (YLD)** in arecanut palms using **drone imagery** and estimating the fraction of a plantation plot affected by the disease.

The system uses a **YOLO11s object detection model** with two classes:

- `healthy`
- `yld`

Inference is performed using **ONNX Runtime on CPU**, without PyTorch at inference time. This keeps the deployment lightweight enough for a **Vercel serverless environment**.

---

## 🚀 Key Features

- 🛩️ Drone-based arecanut plantation monitoring
- 🤖 YOLO11s object detection
- 🌴 Healthy and YLD palm classification
- 📊 Bias-corrected YLD rate estimation
- 📍 GPS-based tree coordinate estimation
- 🗺️ Interactive Leaflet maps
- 🧭 Navigation from user/operator location to detected trees
- 📷 Live drone camera analysis
- 📤 Image upload and analysis
- ⚡ ONNX Runtime CPU inference
- ☁️ Vercel-compatible deployment
- 📱 Responsive web interface
- 🔄 Real-time publishing of drone detection results

---

# 🧠 System Architecture

```text
                    ┌──────────────────────┐
                    │   Drone / Camera     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Image Acquisition  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Client-side Image     │
                    │ Downscaling           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Flask API             │
                    │ api/index.py          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ YOLO11s ONNX Model    │
                    │ ONNX Runtime          │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐        ┌─────────────────┐
        │ Healthy Palms   │        │ YLD Palms       │
        └─────────────────┘        └─────────────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ GPS Coordinate       │
                    │ Estimation            │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Web Dashboard         │
                    │ Maps + Statistics     │
                    └──────────────────────┘
