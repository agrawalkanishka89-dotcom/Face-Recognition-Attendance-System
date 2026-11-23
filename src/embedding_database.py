import numpy as np

from src.config import MODELS_DIR


EMBEDDING_FILE = MODELS_DIR / "face_embeddings.npz"


def save_embeddings(student_id, embeddings):

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    student_ids = np.array(
        [str(student_id)] * len(embeddings)
    )

    if EMBEDDING_FILE.exists():

        data = np.load(EMBEDDING_FILE)

        old_embeddings = data["embeddings"]
        old_student_ids = data["student_ids"]

        embeddings = np.vstack([
            old_embeddings,
            embeddings
        ])

        student_ids = np.concatenate([
            old_student_ids,
            student_ids
        ])

    np.savez_compressed(
        EMBEDDING_FILE,
        embeddings=embeddings,
        student_ids=student_ids
    )

    print("Face embeddings saved successfully!")


def load_embeddings():

    if not EMBEDDING_FILE.exists():
        return None, None

    data = np.load(EMBEDDING_FILE)

    embeddings = data["embeddings"]
    student_ids = data["student_ids"]

    return embeddings, student_ids
