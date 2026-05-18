"""
output/exporter.py
-------------------
Saves pipeline results in all configured output formats:
  1. JSON   — structured per-page layout records
  2. CSV    — flat table for quick analysis
  3. Annotated PDF — colour-coded overlay for visual QA
  4. Crops  — individual region images (optional)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import fitz                 # PyMuPDF — for annotated PDF creation
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OUTPUT, OutputConfig

BBox = Tuple[int, int, int, int]   # (x, y, w, h)


# ─── Data structures ──────────────────────────────────────────────────────────

def make_region(
    pdf_path:  str,
    page_idx:  int,
    bbox:      BBox,
    label:     str,
    confidence: float,
) -> Dict[str, Any]:
    x, y, w, h = bbox
    return {
        "pdf":        pdf_path,
        "page":       page_idx,
        "label":      label,
        "confidence": round(float(confidence), 4),
        "x":          x, "y": y, "w": w, "h": h,
        "area":       w * h,
        "aspect":     round(w / max(h, 1), 4),
    }


# ─── Exporter ─────────────────────────────────────────────────────────────────

class Exporter:
    """
    Exports layout analysis results to disk.

    Usage
    -----
    exp = Exporter(output_dir="data/results")
    exp.export_page(
        pdf_path="doc.pdf",
        page_idx=0,
        rgb_image=rgb,
        boxes=boxes,
        labels=labels,
        confs=confs,
    )
    exp.flush_csv()   # write accumulated CSV at end
    """

    def __init__(
        self,
        output_dir: str | Path = OUTPUT,
        cfg:        OutputConfig = OUTPUT,
    ):
        # Accept either a path string or the config object
        if isinstance(output_dir, OutputConfig):
            cfg        = output_dir
            output_dir = Path("data/results")
        self.cfg        = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._records: List[Dict] = []

    # ── Per-page export ───────────────────────────────────────────────────────

    def export_page(
        self,
        pdf_path:  str,
        page_idx:  int,
        rgb_image: np.ndarray,
        boxes:     List[BBox],
        labels:    List[str],
        confs:     np.ndarray,
    ) -> Dict:
        """
        Export one page's results in all configured formats.

        Returns the per-page result dict (also appended to internal list).
        """
        stem       = Path(pdf_path).stem
        page_dir   = self.output_dir / stem / f"page_{page_idx:04d}"
        page_dir.mkdir(parents=True, exist_ok=True)

        # Build records
        regions = [
            make_region(pdf_path, page_idx, bbox, lbl, conf)
            for bbox, lbl, conf in zip(boxes, labels, confs)
        ]

        result = {
            "pdf":     pdf_path,
            "page":    page_idx,
            "n_regions": len(regions),
            "regions": regions,
        }

        # JSON
        if self.cfg.save_json:
            json_path = page_dir / "layout.json"
            json_path.write_text(json.dumps(result, indent=2))

        # Accumulate for CSV
        self._records.extend(regions)

        # Annotated PNG
        if self.cfg.save_annotated_pdf:
            ann_path = page_dir / "annotated.png"
            annotated = self._annotate_image(rgb_image, boxes, labels, confs)
            cv2.imwrite(str(ann_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        # Crops
        if self.cfg.save_crops:
            crops_dir = page_dir / "crops"
            crops_dir.mkdir(exist_ok=True)
            self._save_crops(rgb_image, boxes, labels, crops_dir)

        return result

    # ── Flush CSV ─────────────────────────────────────────────────────────────

    def flush_csv(self, path: Optional[Path] = None) -> Path:
        """Write all accumulated records to a single CSV file."""
        if not self._records:
            logger.warning("No records to write.")
            return

        path = path or (self.output_dir / "layout_results.csv")
        df   = pd.DataFrame(self._records)
        df.to_csv(path, index=False)
        logger.info(f"CSV written → {path}  ({len(df):,} rows)")
        return path

    # ── Annotated PDF (multi-page) ────────────────────────────────────────────

    def create_annotated_pdf(
        self,
        source_pdf: str | Path,
        all_results: List[Dict],
        out_path:   Optional[Path] = None,
    ) -> Path:
        """
        Overlay coloured bounding boxes on the original PDF pages
        and save as a new annotated PDF.

        Parameters
        ----------
        source_pdf  : original PDF path
        all_results : list of per-page result dicts from export_page()
        out_path    : destination path (default: <stem>_annotated.pdf)
        """
        source_pdf = Path(source_pdf)
        out_path   = out_path or (
            self.output_dir / f"{source_pdf.stem}_annotated.pdf"
        )

        doc   = fitz.open(str(source_pdf))
        index = {r["page"]: r for r in all_results}

        for page_idx in range(len(doc)):
            if page_idx not in index:
                continue
            page    = doc[page_idx]
            pwidth  = page.rect.width
            pheight = page.rect.height

            result  = index[page_idx]
            img_h, img_w = result.get("image_shape", (pheight, pwidth))

            for region in result["regions"]:
                x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                label       = region["label"]
                hex_color   = self.cfg.label_colors.get(label, "#9E9E9E")
                color       = _hex_to_fitz(hex_color)

                # Scale pixel coords → PDF points
                sx = pwidth  / max(img_w, 1)
                sy = pheight / max(img_h, 1)
                rect = fitz.Rect(x*sx, y*sy, (x+w)*sx, (y+h)*sy)

                # Semi-transparent fill
                page.draw_rect(rect, color=color, fill=color,
                               fill_opacity=self.cfg.annotation_alpha, width=0)
                # Solid border
                page.draw_rect(rect, color=color, width=1.2)
                # Label text
                page.insert_text(
                    (rect.x0 + 2, rect.y0 + 8),
                    label,
                    fontsize=6,
                    color=color,
                )

        doc.save(str(out_path))
        doc.close()
        logger.info(f"Annotated PDF → {out_path}")
        return out_path

    # ── Internal ──────────────────────────────────────────────────────────────

    def _annotate_image(
        self,
        rgb:    np.ndarray,
        boxes:  List[BBox],
        labels: List[str],
        confs:  np.ndarray,
    ) -> np.ndarray:
        """Draw coloured boxes + labels on an RGB image copy."""
        canvas  = rgb.copy().astype(np.float32)
        overlay = rgb.copy().astype(np.float32)
        alpha   = self.cfg.annotation_alpha

        for (x, y, w, h), label, conf in zip(boxes, labels, confs):
            hex_col = self.cfg.label_colors.get(label, "#9E9E9E")
            bgr     = _hex_to_bgr(hex_col)
            rgb_col = bgr[::-1]

            # Semi-transparent fill
            cv2.rectangle(overlay, (x, y), (x+w, y+h), rgb_col.tolist(), -1)
            # Border
            cv2.rectangle(canvas, (x, y), (x+w, y+h), rgb_col.tolist(), 2)
            # Label
            txt = f"{label} {conf:.2f}"
            cv2.putText(
                canvas, txt, (x + 2, max(y - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, rgb_col.tolist(), 1,
                cv2.LINE_AA,
            )

        blended = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _save_crops(
        rgb:       np.ndarray,
        boxes:     List[BBox],
        labels:    List[str],
        crops_dir: Path,
    ) -> None:
        h, w = rgb.shape[:2]
        for i, ((x, y, bw, bh), label) in enumerate(zip(boxes, labels)):
            x2 = min(x + bw, w)
            y2 = min(y + bh, h)
            crop = rgb[y:y2, x:x2]
            if crop.size == 0:
                continue
            fname = crops_dir / f"{i:04d}_{label}.png"
            cv2.imwrite(str(fname), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))


# ─── Colour helpers ───────────────────────────────────────────────────────────

def _hex_to_bgr(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return np.array([b, g, r], dtype=np.uint8)

def _hex_to_fitz(hex_color: str) -> Tuple[float, float, float]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r / 255, g / 255, b / 255)
