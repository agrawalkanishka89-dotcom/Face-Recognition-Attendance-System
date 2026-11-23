import pandas as pd

from datetime import datetime

from src.config import (
    STUDENTS_FILE,
    ATTENDANCE_FILE
)


# ---------------------------------
# 1. Initialize CSV Files
# ---------------------------------

def initialize_files():

    STUDENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ATTENDANCE_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not STUDENTS_FILE.exists():

        students_df = pd.DataFrame(
            columns=["ID", "Name"]
        )

        students_df.to_csv(STUDENTS_FILE, index=False)

    if not ATTENDANCE_FILE.exists():

        attendance_df = pd.DataFrame(
            columns=["ID", "Name", "Date", "Time", "Status"]
        )

        attendance_df.to_csv(ATTENDANCE_FILE, index=False)


# ---------------------------------
# 2. Read Student Records
# ---------------------------------

def get_students():

    initialize_files()

    return pd.read_csv(STUDENTS_FILE)


# ---------------------------------
# 3. Generate Unique Student ID
# ---------------------------------

def generate_student_id():

    students_df = get_students()

    if students_df.empty:
        return 1

    student_ids = pd.to_numeric(
        students_df["ID"],
        errors="coerce"
    ).dropna()

    if student_ids.empty:
        return 1

    return int(student_ids.max()) + 1


# ---------------------------------
# 4. Register New Student
# ---------------------------------

def register_student(name):

    name = name.strip()

    if not name:
        raise ValueError("Student name cannot be empty.")

    students_df = get_students()

    existing_names = students_df["Name"].astype(str).str.strip().str.lower()

    if name.lower() in existing_names.values:
        raise ValueError("Student is already registered.")

    student_id = generate_student_id()

    new_student = pd.DataFrame([
        {
            "ID": student_id,
            "Name": name
        }
    ])

    students_df = pd.concat(
        [students_df, new_student],
        ignore_index=True
    )

    students_df.to_csv(STUDENTS_FILE, index=False)

    return student_id


# ---------------------------------
# 5. Mark Attendance
# ---------------------------------

def mark_attendance(student_id):

    initialize_files()

    students_df = get_students()

    student = students_df[
        students_df["ID"].astype(str) == str(student_id)
    ]

    if student.empty:
        raise ValueError("Student not found.")

    student_name = student.iloc[0]["Name"]

    attendance_df = pd.read_csv(ATTENDANCE_FILE)

    today = datetime.now().strftime("%Y-%m-%d")

    existing_record = attendance_df[
        (attendance_df["ID"].astype(str) == str(student_id)) &
        (attendance_df["Date"].astype(str) == today)
    ]

    if not existing_record.empty:
        return "Already Marked"

    current_time = datetime.now().strftime("%H:%M:%S")

    new_record = pd.DataFrame([
        {
            "ID": student_id,
            "Name": student_name,
            "Date": today,
            "Time": current_time,
            "Status": "Present"
        }
    ])

    attendance_df = pd.concat(
        [attendance_df, new_record],
        ignore_index=True
    )

    attendance_df.to_csv(ATTENDANCE_FILE, index=False)

    return "Attendance Marked Successfully"
