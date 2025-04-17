# coding: utf-8

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import numpy as np


def viz_lmk(img_, vps, **kwargs):
    """可视化点"""
    lineType = kwargs.get("lineType", cv2.LINE_8)  # cv2.LINE_AA
    img_for_viz = img_.copy()
    for pt in vps:
        cv2.circle(
            img_for_viz,
            (int(pt[0]), int(pt[1])),
            radius=kwargs.get("radius", 1),
            color=(0, 255, 0),
            thickness=kwargs.get("thickness", 1),
            lineType=lineType,
        )
    return img_for_viz

def draw_landmarks(image: np.ndarray, landmarks: np.ndarray, color=(0, 255, 0), radius=2) -> np.ndarray:
    """Draw landmarks on an image.

    Args:
        image (np.ndarray): The input image (RGB or BGR).
        landmarks (np.ndarray): Landmarks coordinates (N, 2).
        color (tuple, optional): Landmark color (BGR). Defaults to (0, 255, 0) (green).
        radius (int, optional): Landmark radius. Defaults to 2.

    Returns:
        np.ndarray: Image with landmarks drawn.
    """
    output_image = image.copy()
    if landmarks is not None and len(landmarks) > 0:
        for i in range(landmarks.shape[0]):
            x = int(round(landmarks[i, 0]))
            y = int(round(landmarks[i, 1]))
            cv2.circle(output_image, (x, y), radius, color, -1) # -1 fills the circle
    return output_image
