from pathlib import Path

# Project base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Project directories
STUDENTS_DIR = BASE_DIR / "students"
ATTENDANCE_DIR = BASE_DIR / "attendance"
MODELS_DIR = BASE_DIR / "models"

# CSV files
STUDENTS_FILE = STUDENTS_DIR / "students.csv"
ATTENDANCE_FILE = ATTENDANCE_DIR / "attendance.csv"

# Face Recognition Settings
MAX_FACE_IMAGES = 30
CAMERA_INDEX = 0

# Create required directories
for directory in [STUDENTS_DIR, ATTENDANCE_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)