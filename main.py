from src.registration import register_new_student
from src.recognition import start_recognition


# ---------------------------------
# 1. Main Application
# ---------------------------------

def main():

    while True:

        print("\n================================")
        print("   FACE RECOGNITION ATTENDANCE")
        print("================================")

        print("1. Register New Student")
        print("2. Mark Attendance")
        print("3. Exit")

        print("================================")

        choice = input("Enter your choice (1-3): ").strip()

        # Register Student
        if choice == "1":

            try:
                register_new_student()

            except Exception as error:
                print("Registration Error:", error)

        # Mark Attendance
        elif choice == "2":

            try:
                start_recognition()

            except Exception as error:
                print("Recognition Error:", error)

        # Exit Application
        elif choice == "3":

            print("Thank you for using the system!")
            break

        else:

            print("Invalid choice. Please select 1, 2 or 3.")


# ---------------------------------
# 2. Application Entry Point
# ---------------------------------

if __name__ == "__main__":

    main()
