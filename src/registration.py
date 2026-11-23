import cv2
import time

from src.config import MAX_FACE_IMAGES, CAMERA_INDEX
from src.database import get_students, register_student
from src.face_embedding import FaceEmbedding
from src.embedding_database import save_embeddings


def register_new_student():

    name = input("\nEnter Student Name: ").strip()

    if not name:
        print("Student name cannot be empty.")
        return

    students_df = get_students()

    existing_names = (
        students_df["Name"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if name.lower() in existing_names.values:
        print("Student is already registered.")
        return

    face_system = FaceEmbedding()

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("Camera could not be opened!")
        return

    embeddings_list = []
    frame_number = 0
    cancelled = False

    print("\nFace Registration Started!")
    print("Look at the camera.")
    print("Move your face slightly for different samples.")
    print("Press Q to cancel.")

    try:

        while len(embeddings_list) < MAX_FACE_IMAGES:

            success, frame = camera.read()

            if not success:
                print("Camera frame could not be read.")
                break

            embeddings, faces = face_system.extract_embeddings(frame)

            if faces is not None and len(faces) == 1:

                box = faces[0]

                if box is not None:

                    x1, y1, x2, y2 = map(int, box)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    if (
                        len(embeddings) == 1
                        and frame_number % 5 == 0
                    ):

                        embeddings_list.append(embeddings[0])

            elif faces is not None and len(faces) > 1:

                cv2.putText(
                    frame,
                    "Only one person should be visible",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.putText(
                frame,
                f"Samples: {len(embeddings_list)}/{MAX_FACE_IMAGES}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("Student Registration", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                cancelled = True
                break

            frame_number += 1

    finally:

        camera.release()
        cv2.destroyAllWindows()

    if cancelled or len(embeddings_list) < MAX_FACE_IMAGES:

        print("\nRegistration cancelled.")
        print("Student was not registered.")
        return

    try:

        student_id = register_student(name)

        save_embeddings(
            student_id,
            embeddings_list
        )

        print("\nStudent Registered Successfully!")
        print("Student ID:", student_id)
        print("Student Name:", name)
        print("Face Samples:", len(embeddings_list))

    except Exception as error:

        print("Registration failed:", error)


if __name__ == "__main__":
    register_new_student()