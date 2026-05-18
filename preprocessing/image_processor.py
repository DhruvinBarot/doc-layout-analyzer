"""
preprocessing/image_processor.py
---------------------------------
Converts raw page images into analysis-ready form:
  1. Resize to canonical width
  2. Denoise (fast Non-Local Means)
  3. Grayscale conversion
  4. Deskew (correct slight page rotation via Hough transform)
  5. Binarize (Otsu) — used for region proposal step only
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from skimage.transform import rotate as sk_rotate
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PREPROCESS, PreprocessConfig


class ImageProcessor:
    """
    Stateless image preprocessor.  Each method is usable independently.

    Usage
    -----
    proc  = ImageProcessor()
    gray  = proc.to_gray(rgb_image)
    clean = proc.full_preprocess(rgb_image)   # returns dict of views
    """

    def __init__(self, cfg: PreprocessConfig = PREPROCESS):
        self.cfg = cfg

    # ── 1. Resize ─────────────────────────────────────────────────────────────

    def resize(self, img: np.ndarray) -> np.ndarray:
        """
        Resize to cfg.target_width while preserving aspect ratio.
        Only downscales — small pages are left untouched.
        """
        h, w = img.shape[:2]
        if w <= self.cfg.target_width:
            return img
        scale  = self.cfg.target_width / w
        new_wh = (self.cfg.target_width, int(h * scale))
        return cv2.resize(img, new_wh, interpolation=cv2.INTER_AREA)

    # ── 2. Denoise ────────────────────────────────────────────────────────────

    def denoise(self, img: np.ndarray) -> np.ndarray:
        """
        Fast Non-Local Means denoising.
        Works on RGB or grayscale input.
        """
        if img.ndim == 3:
            return cv2.fastNlMeansDenoisingColored(
                img,
                None,
                h=self.cfg.denoise_h,
                hColor=self.cfg.denoise_h,
                templateWindowSize=7,
                searchWindowSize=21,
            )
        return cv2.fastNlMeansDenoising(
            img,
            None,
            h=self.cfg.denoise_h,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    # ── 3. Grayscale ──────────────────────────────────────────────────────────

    @staticmethod
    def to_gray(img: np.ndarray) -> np.ndarray:
        """RGB → grayscale.  Handles already-gray images gracefully."""
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ── 4. Deskew ─────────────────────────────────────────────────────────────

    def deskew(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect and correct page skew using Probabilistic Hough Lines.
        Returns the (possibly rotated) grayscale image.
        """
        if not self.cfg.deskew:
            return gray

        # Edge map → Hough lines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=gray.shape[1] // 4,
            maxLineGap=20,
        )

        if lines is None:
            return gray

        # Collect angles of nearly-horizontal lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < self.cfg.deskew_max_angle:
                angles.append(angle)

        if not angles:
            return gray

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.1:            # negligible skew
            return gray

        logger.debug(f"Deskewing by {median_angle:.2f}°")
        corrected = sk_rotate(
            gray,
            angle=-median_angle,
            resize=False,
            mode="edge",
            preserve_range=True,
        ).astype(np.uint8)
        return corrected

    # ── 5. Binarize (Otsu) ────────────────────────────────────────────────────

    @staticmethod
    def binarize(gray: np.ndarray) -> np.ndarray:
        """
        Otsu global thresholding → binary image (255 = foreground / ink).
        Input must be grayscale.
        """
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary

    @staticmethod
    def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
        """
        Adaptive thresholding — better for uneven illumination / scanned docs.
        """
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=25,
            C=10,
        )

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def full_preprocess(self, rgb: np.ndarray) -> dict:
        """
        Run the complete preprocessing stack.

        Returns
        -------
        dict with keys:
          "rgb"    : resized colour image (for feature extraction)
          "gray"   : grayscale, deskewed
          "binary" : Otsu binary (for region proposal)
        """
        rgb    = self.resize(rgb)
        gray   = self.to_gray(rgb)
        gray   = self.deskew(gray)
        binary = self.binarize(gray) if self.cfg.binarize else gray
        return {"rgb": rgb, "gray": gray, "binary": binary}


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python image_processor.py <image_path>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    img      = cv2.imread(str(img_path))
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    proc   = ImageProcessor()
    result = proc.full_preprocess(img_rgb)
    for k, v in result.items():
        print(f"{k}: shape={v.shape}  dtype={v.dtype}")
