# ATM Intelligent Surveillance System (xVision)

An AI-powered computer vision system for real-time ATM surveillance using **YOLOv8s** and a custom **Spatial-Memory Tracker**. The system detects **people** and **trash**, tracks stationary trash during human occlusions, monitors dwell time, and automatically triggers alerts when trash remains unattended beyond a predefined threshold. It also generates automated reports and supports Docker-based deployment with GitHub Actions CI/CD.

---

## Features

- YOLOv8s-based person and trash detection
- Custom Spatial-Memory Tracker for static object tracking
- Occlusion-aware dwell-time monitoring
- Automatic alert generation
- Automated dwell-time report generation
- Docker support
- GitHub Actions CI/CD pipeline

---

## Project Structure

```text
ATM_Surveillance_Project/
├── .github/
│   └── workflows/
│       └── docker-build.yaml
├── src/
│   ├── config.py
│   ├── tracker.py
│   ├── trainer.py
│   ├── xvision_detect.py
│   ├── xVision_evalute.py
│   └── dataset_preprocessor.py
├── data.yaml
├── Dockerfile
├── requirements.txt
├── xVision_main.py
├── yolov8s.pt
└── README.md
```

---

## Tracking Logic

The custom **Spatial-Memory Tracker** is designed for static objects and overcomes the limitations of conventional motion-based trackers by:

- Preserving object IDs during human occlusions
- Continuing dwell-time counting while trash is temporarily hidden
- Using a configurable grace period before removing objects
- Restoring previous IDs when objects reappear nearby

---

## Performance

- **mAP@50:** **99.08%**
- **mAP@50-95:** **88.77%**

---

## Installation

```bash
git clone https://github.com/your-username/ATM-Intelligent-Surveillance-System.git

cd ATM-Intelligent-Surveillance-System

pip install -r requirements.txt
```

---

## Usage

```bash
# Dataset preprocessing
python xVision_main.py --task preprocess

# Train model
python xVision_main.py --task train

# Evaluate model
python xVision_main.py --task eval

# Run tracking
python xVision_main.py --task track

# Run object detection
python xVision_main.py --task detect
```

---

## Docker

```bash
docker build -t atm-surveillance .

docker run --rm atm-surveillance
```

---

## CI/CD

GitHub Actions automatically builds and pushes the Docker image to Docker Hub whenever changes are pushed to the `main` branch.

**Docker Image**

```text
abeera22imtiaz/atm-intelligent-surveillance-system:latest
```

---

## Technologies

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Docker
- GitHub Actions

---

## Author

**Abeera Imtiaz**
