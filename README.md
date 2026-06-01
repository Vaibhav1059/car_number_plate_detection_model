# 🚗 Automatic Number Plate Recognition (ANPR)

## 📌 Project Overview

This project implements an **Automatic Number Plate Recognition (ANPR)** system using:

- YOLOv8 for license plate detection
- EasyOCR for text extraction
- Streamlit for web interface

The system detects vehicle number plates from images and extracts the text along with confidence scores.

---

## ⚙️ Features

- 📍 License plate detection using YOLOv8
- 🔤 OCR text extraction (English + Hindi support)
- 📊 Confidence scoring for detection and OCR
- 🖼️ Image upload interface using Streamlit
- ⚠️ Low-confidence detection warning system

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

### 1. Clone the repository
git clone <repo-url>
cd ANPR

### 2. Create environment


conda create -n anpr python=3.10
conda activate anpr


### 3. Install dependencies


pip install -r requirements.txt


---

## ▶️ Run the Application


streamlit run streamlit_app/app.py


---

## 🧪 Usage

1. Upload an image of a vehicle
2. The system will:
   - Detect the number plate
   - Extract text
   - Show confidence scores

---

## ⚠️ Limitations

- OCR accuracy may vary depending on:
  - Lighting conditions
  - Image quality
  - Font variations
- Works best on clear and front-facing number plates
- Not optimized for real-time video

---

## 🧠 Technical Details

### Detection
- Model: YOLOv8 (Ultralytics)
- Trained on custom license plate dataset

### OCR
- Engine: EasyOCR
- Multi-preprocessing approach used for better accuracy

---

## 📈 Future Improvements

- Improve OCR using PaddleOCR or custom model
- Add real-time video processing
- Deploy on cloud (AWS / Render)
- Enhance dataset for better generalization

---

## 👨‍💻 Authors

- Shrestha Swami
- Bhumi Porwal
- Vaibhav Soni

---

## 📌 Conclusion

This project demonstrates a complete end-to-end pipeline for number plate detection and recogniti