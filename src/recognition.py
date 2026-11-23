import cv2
import numpy as np

from src.config import CAMERA_INDEX
from src.face_embedding import FaceEmbedding
from src.embedding_database import load_embeddings
from src.database import get_students
from src.attendance import process_attendance


# Initial thresholds - testing ke baad calibrate karenge
SIMILARITY_THRESHOLD = 0.70
MINIMUM_MARGIN = 0.05
STABLE_FRAMES = 5


def predict_student(face_embedding, saved_embeddings, saved_ids):

    if saved_embeddings is None or saved_ids is None:
        return None, 0.0

    # Cosine similarity calculate karna
    similarities = saved_embeddings @ face_embedding

    # Har student ki maximum similarity nikalna
    student_scores = {}

    for student_id in np.unique(saved_ids):

        student_similarities = similarities[
            saved_ids == student_id
        ]

        student_scores[str(student_id)] = float(
            np.max(student_similarities)
        )

    if not student_scores:
        return None, 0.0

    sorted_scores = sorted(
        student_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    best_id, best_score = sorted_scores[0]

    # Similarity threshold check
    if best_score < SIMILARITY_THRESHOLD:
        return None, best_score

    # Second-best student se sufficient difference hona chahiye
    if len(sorted_scores) > 1:

        second_score = sorted_scores[1][1]

        if best_score - second_score < MINIMUM_MARGIN:
            return None, best_score

    return best_id, best_score


def start_recognition():

    print("Loading FaceNet...")

    face_system = FaceEmbedding()

    saved_embeddings, saved_ids = load_embeddings()

    if saved_embeddings is None:
        print("No registered face embeddings found.")
        print("Please register students first.")
        return

    students_df = get_students()

    student_names = dict(
        zip(
            students_df["ID"].astype(str),
            students_df["Name"].astype(str)
        )
    )

    saved_embeddings = np.asarray(
        saved_embeddings,
        dtype=np.float32
    )

    saved_ids = saved_ids.astype(str)

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError("Camera could not be opened.")

    print("\nFace Recognition Started!")
    print("Press Q to exit.")

    previous_candidate = None
    stable_count = 0
    attendance_handled_for = None
    last_message = ""

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print("Camera frame could not be read.")
                break

            embeddings, faces = face_system.extract_embeddings(frame)

            candidate_id = None
            similarity = 0.0

            if faces is not None and len(faces) == 1 and len(embeddings) == 1:

                box = faces[0]

                if box is not None:

                    x1, y1, x2, y2 = map(int, box)

                    candidate_id, similarity = predict_student(
                        embeddings[0],
                        saved_embeddings,
                        saved_ids
                    )

                    if candidate_id is not None:

                        student_name = student_names.get(candidate_id)

                        if student_name is not None:

                            display_text = (
                                f"{student_name} | Similarity: "
                                f"{similarity:.2f}"
                            )

                            color = (0, 255, 0)

                        else:
                            candidate_id = None
                            display_text = "Unknown Face"
                            color = (0, 0, 255)

                    else:

                        display_text = "Unknown Face"
                        color = (0, 0, 255)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    cv2.putText(
                        frame,
                        display_text,
                        (x1, max(y1 - 10, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )

            elif faces is not None and len(faces) > 1:

                cv2.putText(
                    frame,
                    "Only one face should be visible",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # Stable prediction check
            if candidate_id is not None:

                if candidate_id == previous_candidate:
                    stable_count += 1

                else:
                    previous_candidate = candidate_id
                    stable_count = 1
                    last_message = ""

                # Attendance only after stable recognition
                if (
                    stable_count >= STABLE_FRAMES
                    and candidate_id != attendance_handled_for
                ):

                    result = process_attendance(candidate_id)

                    last_message = result or "Attendance Error"

                    attendance_handled_for = candidate_id

            else:

                previous_candidate = None
                stable_count = 0
                attendance_handled_for = None
                last_message = ""

            if last_message:

                cv2.putText(
                    frame,
                    last_message,
                    (20, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

            cv2.imshow(
                "Face Recognition Attendance",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start_recognition()