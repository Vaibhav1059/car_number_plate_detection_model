# 🚗 Automatic Number Plate Recognition (ANPR)

## 📌 Project Overview

This project implements an **Automatic Number Plate Recognition (ANPR)** system using advanced Computer Vision and OCR techniques. The application automatically detects vehicle license plates from images and extracts the registration number with confidence scores through an interactive web interface.

### 🔗 Live Demo

https://carnumberplatedetectionmodel-hmlev3rcv9q5vbsbehmgrg.streamlit.app/

### 🔗 GitHub Repository

https://github.com/Vaibhav1059/car_number_plate_detection_model

### 🛠️ Technologies Used

* YOLOv8 (Ultralytics)
* EasyOCR
* OpenCV
* Python
* Streamlit
* NumPy

---

## ⚙️ Features

* 📍 License Plate Detection using YOLOv8
* 🔤 OCR-based Text Extraction
* 🌐 English and Hindi Number Plate Support
* 📊 Detection & OCR Confidence Scores
* 🖼️ User-Friendly Streamlit Interface
* ⚠️ Low-Confidence Detection Warning System
* 🚀 End-to-End Automated ANPR Pipeline

---

## 📂 Project Structure

ANPR/
│
├── data/
│ └── yolo_dataset/
│
├── models/
│ └── yolov8/
│ └── best.pt
│
├── src/
│ ├── detector.py
│ ├── ocr.py
│ └── pipeline.py
│
├── streamlit_app/
│ └── app.py
│
├── notebooks/
│
└── requirements.txt

---

## 🚀 Installation

### Clone Repository

git clone https://github.com/Vaibhav1059/car_number_plate_detection_model.git

cd car_number_plate_detection_model

### Create Virtual Environment

conda create -n anpr python=3.10

conda activate anpr

### Install Dependencies

pip install -r requirements.txt

---

## ▶️ Run Application

streamlit run streamlit_app/app.py

---

## 🧪 Usage

1. Upload a vehicle image.
2. The system detects the license plate.
3. OCR extracts the plate number.
4. Detection and OCR confidence scores are displayed.
5. Review results directly in the Streamlit interface.

---

## 🧠 Technical Details

### Detection Model

* YOLOv8 (Ultralytics)
* Custom-trained License Plate Dataset
* Bounding Box Detection

### OCR Engine

* EasyOCR
* Multi-stage Image Preprocessing
* Confidence-Based Text Recognition

---

## 📈 Future Improvements

* Real-Time Video Processing
* Vehicle Entry/Exit Logging System
* Database Integration (MySQL/PostgreSQL)
* Parking Management Dashboard
* PaddleOCR Integration
* Cloud Deployment (AWS, Azure, GCP)
* Mobile-Friendly Interface

---

## ⚠️ Limitations

* OCR accuracy depends on image quality.
* Performance may decrease under poor lighting conditions.
* Best results are achieved with clear front-facing number plates.
* Currently optimized for image-based processing.

---

## 📌 Conclusion

This project demonstrates a complete end-to-end AI-powered Automatic Number Plate Recognition (ANPR) pipeline, integrating object detection and optical character recognition to automate vehicle identification tasks. It showcases practical applications of Computer Vision, Deep Learning, and OCR technologies in real-world scenarios such as traffic monitoring, parking management, and security systems.

---

### 👨‍💻 Author

**Vaibhav Soni**

GitHub: https://github.com/Vaibhav1059

LinkedIn: https://www.linkedin.com/in/vaibhav-soni/
