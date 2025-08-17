# 🛡️ Optimizing Workplace Compliance and Safety using AI

## 📌 Project Overview
Manual monitoring of workplace safety incidents is inefficient, error-prone, and inconsistent. This project leverages **AI-powered Object Detection and Pose Estimation** to automate compliance tracking in real time, helping industries ensure worker safety and reduce risks.

The system detects **safety gear compliance (gloves, masks, lab coats, shoes, goggles, hair caps)** and evaluates **worker posture** for unsafe actions. It provides real-time monitoring through a **Streamlit-based deployment**.

---

## 🎯 Objectives
- **Maximize workplace safety** by detecting PPE (Personal Protective Equipment).
- **Evaluate posture compliance** using pose estimation.
- **Automate real-time monitoring** to reduce dependency on manual reporting.
- **Minimize deployment costs** while ensuring high accuracy.

---

## 🏗️ Project Workflow
1. **Image Collection & Preprocessing**
   - Collected 540 workplace images.
   - Extracted frames from videos using OpenCV.

2. **Image Annotation & Data Augmentation**
   - Annotated safety gear with **Roboflow**.
   - Applied augmentations: rotation, scaling, flipping, brightness/contrast changes, noise, shear.

3. **Object Detection**
   - Model: **YOLOv11n**
   - Training: 100 epochs, 640px input size.
   - Metrics: Loss, mAP.

4. **Pose Estimation**
   - Model: **YOLOv11n-Pose**
   - Tracked unsafe activities (bending, running, laying, arm-raising, face-touching, etc.).

5. **Deployment**
   - Framework: **Streamlit**
   - Features:
     - Upload images (JPG/PNG).
     - Real-time detection & compliance status display.

6. **Analysis & Monitoring**
   - Automated reports for safety compliance.
   - Identifies non-compliance and unsafe posture events.

---

## 🖼️ Project Architecture
### High-Level
- Data Collection → Preprocessing → Object Detection → Pose Estimation → Deployment → Monitoring

### Low-Level
- OpenCV for frame extraction  
- Roboflow for annotation  
- YOLOv11n for detection & pose estimation  
- Streamlit for deployment  

---

## ⚙️ Tech Stack
- Python
- YOLOv11n / YOLOv11n-Pose**
- OpenCV
- Roboflow
- Streamlit
- NumPy / Pandas / Matplotlib**

---
🚀 Deployment
To run locally:

```bash
# Clone the repository
git clone https://github.com/<your-username>/workplace-safety-ai.git
cd workplace-safety-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

📊 Challenges Faced

Limited dataset & diversity of workplace conditions.

Time-consuming annotation and class imbalance.

High GPU requirements for YOLOv11n training.

Pose estimation errors due to occlusions.

Deployment latency with large images.

Ensuring cost-effectiveness and regulatory compliance.
