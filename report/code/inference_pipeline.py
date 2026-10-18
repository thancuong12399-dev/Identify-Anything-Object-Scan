import cv2
import torch
import numpy as np

class FaceVerificationSystem:
    """
    End-to-End Real-Time Face Recognition & Anti-Spoofing Verification Pipeline.
    Combines RetinaFace detection, MiniFASNet liveness checking, and ArcFace feature extraction.
    """
    def __init__(self, embedding_model, liveness_model, database_features: dict, threshold: float = 0.65):
        self.embedding_model = embedding_model
        self.liveness_model = liveness_model
        self.database = database_features  # Dictionary: {person_id: 512D_normalized_vector}
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def verify_face(self, aligned_face: np.ndarray) -> dict:
        """
        Processes an aligned 112x112 face image and returns identification results.
        """
        # 1. Preprocess Tensor
        tensor = aligned_face.astype(np.float32) / 127.5 - 1.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)

        # 2. Anti-Spoofing / Liveness Check
        with torch.no_grad():
            liveness_score = float(torch.softmax(self.liveness_model(tensor), dim=1)[0][1])
        
        if liveness_score < 0.85:
            return {"status": "SPOOF_REJECTED", "liveness_score": liveness_score, "identity": None}

        # 3. Deep Feature Extraction (ArcFace 512-D)
        with torch.no_grad():
            embedding = self.embedding_model(tensor)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1).cpu().numpy()[0]

        # 4. Cosine Similarity Matching Against Database
        best_match_id = None
        max_similarity = -1.0

        for person_id, ref_vector in self.database.items():
            cosine_sim = np.dot(embedding, ref_vector)
            if cosine_sim > max_similarity:
                max_similarity = float(cosine_sim)
                best_match_id = person_id

        if max_similarity >= self.threshold:
            return {
                "status": "VERIFIED",
                "identity": best_match_id,
                "confidence": max_similarity,
                "liveness_score": liveness_score
            }
        else:
            return {
                "status": "UNKNOWN_PERSON",
                "identity": None,
                "confidence": max_similarity,
                "liveness_score": liveness_score
            }
