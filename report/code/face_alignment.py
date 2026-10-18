import cv2
import numpy as np

def align_face_landmarks(image: np.ndarray, landmarks: np.ndarray, output_size=(112, 112)) -> np.ndarray:
    """
    Align facial image using 5-point facial landmarks via Similarity Transformation.
    Landmarks array shape: (5, 2) corresponding to:
    [Left Eye, Right Eye, Nose Tip, Left Mouth Corner, Right Mouth Corner]
    """
    # Standard ArcFace canonical 5-point reference landmarks for 112x112 resolution
    reference_landmarks = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    # Estimate affine partial similarity transformation matrix (rigid: scale + rotation + translation)
    transform_matrix, _ = cv2.estimateAffinePartial2D(landmarks, reference_landmarks)
    
    if transform_matrix is None:
        # Fallback to direct resize if transformation estimation fails
        return cv2.resize(image, output_size)

    # Warp affine transformation to obtain canonical frontal alignment
    aligned_face = cv2.warpAffine(
        image, 
        transform_matrix, 
        output_size, 
        borderMode=cv2.BORDER_REFLECT_101
    )
    return aligned_face
