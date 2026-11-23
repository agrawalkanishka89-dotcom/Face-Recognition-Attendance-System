from src.database import mark_attendance


# ---------------------------------
# 1. Process Student Attendance
# ---------------------------------

def process_attendance(student_id):

    try:

        result = mark_attendance(student_id)

        if result == "Already Marked":

            print("Already Marked!")

        else:

            print("Attendance Marked Successfully!")

        return result

    except ValueError as error:

        print("Attendance Error:", error)

        return None

    except Exception as error:

        print("Unexpected Error:", error)

        return None
