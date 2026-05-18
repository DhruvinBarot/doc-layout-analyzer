"""
features/feature_extractor.py
-------------------------------
Extracts a rich, fixed-length feature vector from each region proposal.

Feature groups:
  A) HOG (Histogram of Oriented Gradients)   — captures local structure
  B) LBP (Local Binary Patterns)             — texture descriptor
  C) Intensity histogram                     — tonal distribution
  D) Geometric features                      — shape, fill, density
  E) Positional features                     — location on page

Total dimensionality: ~400 floats (depends on HOG/LBP settings).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEATURES, FeatureConfig

BBox = Tuple[int, int, int, int]


class FeatureExtractor:
    """
    Extracts a fixed-length feature vector per region crop.

    Usage
    -----
    extractor = FeatureExtractor()
    feat_vec  = extractor.extract(gray_image, binary_image, bbox, page_shape)
    mat       = extractor.extract_batch(gray, binary, boxes, page_shape)
    """

    def __init__(self, cfg: FeatureConfig = FEATURES):
        self.cfg        = cfg
        self._feat_dim  = None          # computed on first call

    # ── Public: single region ─────────────────────────────────────────────────

    def extract(
        self,
        gray:       np.ndarray,
        binary:     np.ndarray,
        bbox:       BBox,
        page_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Return a 1-D float32 feature vector for one region.

        Parameters
        ----------
        gray       : full-page grayscale image
        binary     : full-page Otsu binary image
        bbox       : (x, y, w, h) of the region
        page_shape : (height, width) of the full page
        """
        x, y, w, h = bbox
        page_h, page_w = page_shape[:2]

        # Clamp crop to image bounds
        x1 = max(x, 0);  y1 = max(y, 0)
        x2 = min(x + w, page_w);  y2 = min(y + h, page_h)

        gray_crop   = gray[y1:y2, x1:x2]
        binary_crop = binary[y1:y2, x1:x2]

        if gray_crop.size == 0:
            logger.warning(f"Empty crop at {bbox} — returning zeros")
            return np.zeros(self._ensure_dim(gray, binary, page_shape), dtype=np.float32)

        parts = []

        # A) HOG
        parts.append(self._hog_features(gray_crop))

        # B) LBP texture
        parts.append(self._lbp_features(gray_crop))

        # C) Intensity histogram
        if self.cfg.include_histogram:
            parts.append(self._intensity_hist(gray_crop))

        # D) Geometric
        if self.cfg.include_geometric:
            parts.append(self._geometric_features(
                gray_crop, binary_crop, w, h
            ))

        # E) Positional
        if self.cfg.include_positional:
            parts.append(self._positional_features(
                x1, y1, x2, y2, page_w, page_h
            ))

        return np.concatenate(parts).astype(np.float32)

    # ── Public: batch ─────────────────────────────────────────────────────────

    def extract_batch(
        self,
        gray:       np.ndarray,
        binary:     np.ndarray,
        boxes:      List[BBox],
        page_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Extract features for all boxes in one page.

        Returns
        -------
        np.ndarray of shape (N, D) where D is feature dimensionality.
        """
        feats = [self.extract(gray, binary, bbox, page_shape) for bbox in boxes]
        if not feats:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack(feats)

    # ── A) HOG ────────────────────────────────────────────────────────────────

    def _hog_features(self, crop: np.ndarray) -> np.ndarray:
        """HOG on a fixed-size thumbnail."""
        resized = cv2.resize(
            crop, self.cfg.hog_resize, interpolation=cv2.INTER_AREA
        )
        feat = hog(
            resized,
            orientations=self.cfg.hog_orientations,
            pixels_per_cell=self.cfg.hog_pixels_per_cell,
            cells_per_block=self.cfg.hog_cells_per_block,
            block_norm="L2-Hys",
            visualize=False,
            feature_vector=True,
        )
        return feat.astype(np.float32)

    # ── B) LBP texture ────────────────────────────────────────────────────────

    def _lbp_features(self, crop: np.ndarray) -> np.ndarray:
        """LBP uniform histogram."""
        resized = cv2.resize(
            crop, (64, 64), interpolation=cv2.INTER_AREA
        )
        lbp = local_binary_pattern(
            resized,
            P=self.cfg.lbp_n_points,
            R=self.cfg.lbp_radius,
            method="uniform",
        )
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=self.cfg.lbp_n_bins,
            range=(0, self.cfg.lbp_n_bins),
            density=True,
        )
        return hist.astype(np.float32)

    # ── C) Intensity histogram ────────────────────────────────────────────────

    @staticmethod
    def _intensity_hist(crop: np.ndarray, bins: int = 16) -> np.ndarray:
        hist, _ = np.histogram(
            crop.ravel(), bins=bins, range=(0, 256), density=True
        )
        return hist.astype(np.float32)

    # ── D) Geometric features ─────────────────────────────────────────────────

    @staticmethod
    def _geometric_features(
        gray_crop:   np.ndarray,
        binary_crop: np.ndarray,
        w: int,
        h: int,
    ) -> np.ndarray:
        """
        13 geometric descriptors:
          area, aspect_ratio, fill_ratio, ink_density,
          mean, std, min, max,
          h_proj_entropy, v_proj_entropy,
          runs_h, runs_v, edge_density
        """
        area   = float(w * h)
        aspect = w / max(h, 1)
        fill   = float(np.count_nonzero(binary_crop)) / max(area, 1)

        mean_v = float(gray_crop.mean())
        std_v  = float(gray_crop.std())
        min_v  = float(gray_crop.min())
        max_v  = float(gray_crop.max())

        # Projection profiles → entropy
        h_proj = gray_crop.mean(axis=1) + 1e-6
        v_proj = gray_crop.mean(axis=0) + 1e-6
        h_proj /= h_proj.sum()
        v_proj /= v_proj.sum()
        h_ent  = float(-np.sum(h_proj * np.log2(h_proj)))
        v_ent  = float(-np.sum(v_proj * np.log2(v_proj)))

        # Run-length (horizontal / vertical ink runs)
        runs_h = _count_runs(binary_crop, axis=1)
        runs_v = _count_runs(binary_crop, axis=0)

        # Edge density
        edges = cv2.Canny(gray_crop, 50, 150)
        edge_dens = float(edges.mean()) / 255.0

        return np.array(
            [
                np.log1p(area),
                aspect,
                fill,
                fill,
                mean_v / 255.0,
                std_v / 128.0,
                min_v / 255.0,
                max_v / 255.0,
                h_ent,
                v_ent,
                runs_h,
                runs_v,
                edge_dens,
            ],
            dtype=np.float32,
        )

    # ── E) Positional features ────────────────────────────────────────────────

    @staticmethod
    def _positional_features(
        x1: int, y1: int, x2: int, y2: int,
        page_w: int, page_h: int,
    ) -> np.ndarray:
        """
        8 normalised positional descriptors.
        Useful for header/footer/margin classification.
        """
        cx  = (x1 + x2) / 2.0
        cy  = (y1 + y2) / 2.0
        rw  = (x2 - x1) / max(page_w, 1)
        rh  = (y2 - y1) / max(page_h, 1)
        return np.array(
            [
                x1 / page_w,            # left edge
                y1 / page_h,            # top edge
                x2 / page_w,            # right edge
                y2 / page_h,            # bottom edge
                cx / page_w,            # centre x
                cy / page_h,            # centre y
                rw,                     # relative width
                rh,                     # relative height
            ],
            dtype=np.float32,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _ensure_dim(
        self,
        gray:       np.ndarray,
        binary:     np.ndarray,
        page_shape: Tuple[int, int],
    ) -> int:
        """Return feature dimensionality by running one dummy extraction."""
        if self._feat_dim is None:
            h, w = gray.shape[:2]
            dummy = self.extract(gray, binary, (0, 0, w // 2, h // 2), page_shape)
            self._feat_dim = dummy.shape[0]
        return self._feat_dim


# ─── Helper ───────────────────────────────────────────────────────────────────

def _count_runs(binary: np.ndarray, axis: int) -> float:
    """Average number of foreground runs per row/column (normalised)."""
    total = 0
    arr   = binary if axis == 1 else binary.T
    for row in arr:
        transitions = np.diff((row > 0).astype(int))
        total += int((transitions == 1).sum())
    return float(total) / max(len(arr), 1)


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng    = np.random.default_rng(0)
    gray   = rng.integers(0, 256, (1000, 800), dtype=np.uint8)
    binary = (gray > 128).astype(np.uint8) * 255

    ext    = FeatureExtractor()
    bbox   = (50, 50, 200, 100)
    feat   = ext.extract(gray, binary, bbox, gray.shape)
    print(f"Feature vector shape: {feat.shape}")
    print(f"First 10 values: {feat[:10]}")

    mat = ext.extract_batch(gray, binary, [(50,50,200,100),(300,200,150,80)], gray.shape)
    print(f"Batch matrix shape: {mat.shape}")
