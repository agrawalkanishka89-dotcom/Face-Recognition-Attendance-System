
import cv2
import torch
import numpy as np

from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.nn.functional import normalize


DEVICE = torch.device("cpu")


class FaceEmbedding:

    def __init__(self):

        print("Loading FaceNet models...")

        self.detector = MTCNN(
            image_size=160,
            margin=20,
            keep_all=True,
            min_face_size=40,
            device=DEVICE
        )

        self.model = InceptionResnetV1(
            pretrained="vggface2"
        ).eval().to(DEVICE)

        print("FaceNet models loaded successfully!")

    def extract_embeddings(self, frame):

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        faces, probabilities = self.detector.detect(rgb_frame)

        face_tensors = self.detector(rgb_frame)

        if face_tensors is None:
            return [], faces

        if face_tensors.ndim == 3:
            face_tensors = face_tensors.unsqueeze(0)

        with torch.no_grad():

            embeddings = self.model(
                face_tensors.to(DEVICE)
            )

            embeddings = normalize(
                embeddings,
                p=2,
                dim=1
            )

        return embeddings.cpu().numpy(), faces