# Face Recognition Attendance System

An automated attendance management system built using Python, FaceNet, MTCNN, and OpenCV. The system identifies registered students through facial recognition and records their attendance automatically.

## Project Overview

This project uses deep learning-based facial recognition to automate student attendance.

It uses the pretrained FaceNet model to generate 512-dimensional facial embeddings and compares them using cosine similarity to recognize registered students.

## Key Features

- Student registration using face recognition.
- Face detection using MTCNN.
- Facial feature extraction using FaceNet.
- Recognition using cosine similarity.
- Automatic attendance marking.
- Duplicate attendance prevention.
- Unknown face detection.
- CSV-based student and attendance management.
- Real-time webcam integration.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework |
| FaceNet | Facial feature extraction |
| MTCNN | Face detection |
| OpenCV | Webcam and image processing |
| NumPy | Numerical operations |
| Pandas | CSV data management |
| Scikit-learn | Machine learning utilities |

## Deep Learning Architecture

The system uses:

- **MTCNN:** Detects faces from webcam frames.
- **InceptionResnetV1 (FaceNet):** A pretrained CNN-based architecture that generates 512-dimensional facial embeddings.
- **Cosine Similarity:** Compares facial embeddings to identify registered students.

This project uses transfer learning with a pretrained FaceNet model rather than training a custom CNN classifier from scratch.

## Project Structure

```text
face_recog_atten/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── attendance.py
│   ├── config.py
│   ├── database.py
│   ├── embedding_database.py
│   ├── face_embedding.py
│   ├── recognition.py
│   └── registration.py
│
└── notebooks/
    └── face_attendance.ipynb
```

## Installation

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project

```bash
cd face_recog_atten
```

### 3. Create a Virtual Environment

```bash
python -m venv face_env
```

### 4. Activate the Environment

Windows PowerShell:

```powershell
.\face_env\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

**Note:** Install compatible PyTorch, torchvision, NumPy, Pillow, and FaceNet dependencies before running the application. Refer to their official installation documentation if dependency conflicts occur.

## Run the Application

```bash
python main.py
```

The application provides three options:

1. Register New Student
2. Mark Attendance
3. Exit

## Attendance Management

Student and attendance information is stored locally in CSV files.

The system prevents duplicate attendance entries for the same student on the same date.

## Important Notes

- A working webcam is required.
- Registered students' face embeddings are stored locally.
- Student records
