"""
pipeline.py
------------
Orchestrates the end-to-end Document Layout Analysis pipeline:

  PDF Folder → [Load] → [Preprocess] → [Propose] → [Extract] → [Classify] → [Export]

Designed for 80K+ PDFs:
  - Streams pages; never loads the full corpus into memory
  - Uses joblib Parallel for page-level parallelism
  - Detailed progress tracking via loguru + tqdm
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger
from tqdm import tqdm

from config import (
    INGESTION, PREPROCESS, PROPOSAL, FEATURES, CLASSIFIER, OUTPUT,
    RESULTS_DIR, LGBM_MODEL_PATH,
)
from ingestion.pdf_loader        import PDFLoader
from preprocessing.image_processor import ImageProcessor
from segmentation.region_proposer  import RegionProposer
from features.feature_extractor    import FeatureExtractor
from classification.lgbm_classifier import LayoutClassifier
from output.exporter               import Exporter


class LayoutPipeline:
    """
    Full document layout analysis pipeline.

    Usage — Inference (model already trained)
    -----------------------------------------
    pipe = LayoutPipeline.from_saved()
    pipe.run(pdf_folder="path/to/pdfs", output_dir="data/results")

    Usage — Training
    ----------------
    pipe = LayoutPipeline()
    pipe.train_from_annotations("annotations.csv")
    pipe.run(...)
    """

    def __init__(
        self,
        classifier: Optional[LayoutClassifier] = None,
        output_dir: str | Path = RESULTS_DIR,
    ):
        self.loader     = PDFLoader(INGESTION)
        self.processor  = ImageProcessor(PREPROCESS)
        self.proposer   = RegionProposer(PROPOSAL)
        self.extractor  = FeatureExtractor(FEATURES)
        self.classifier = classifier or LayoutClassifier(CLASSIFIER)
        self.exporter   = Exporter(output_dir)
        self.output_dir = Path(output_dir)

        self._stats = {
            "pdfs":    0, "pages": 0,
            "proposals_raw": 0, "proposals_kept": 0,
            "regions": 0, "errors": 0,
        }

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_saved(
        cls,
        model_path: Path = LGBM_MODEL_PATH,
        output_dir: str | Path = RESULTS_DIR,
    ) -> "LayoutPipeline":
        """Load an already-trained classifier and return a ready pipeline."""
        clf = LayoutClassifier.load(model_path)
        return cls(classifier=clf, output_dir=output_dir)

    # ── Main inference run ────────────────────────────────────────────────────

    def run(
        self,
        pdf_folder: str | Path,
        recursive:  bool = True,
        max_pdfs:   Optional[int] = None,
    ) -> dict:
        """
        Process all PDFs in a folder and save results.

        Parameters
        ----------
        pdf_folder : path to folder containing PDF files
        recursive  : scan sub-folders
        max_pdfs   : limit number of PDFs (None = all)

        Returns
        -------
        summary statistics dict
        """
        if not self.classifier.is_fitted():
            raise RuntimeError(
                "Classifier not trained. Call .train_from_annotations() "
                "or use LayoutPipeline.from_saved()."
            )

        pdf_folder = Path(pdf_folder)
        pattern    = "**/*.pdf" if recursive else "*.pdf"
        pdf_list   = sorted(pdf_folder.glob(pattern))
        if max_pdfs:
            pdf_list = pdf_list[:max_pdfs]

        if not pdf_list:
            logger.warning(f"No PDFs found in {pdf_folder}")
            return self._stats

        logger.info(f"Processing {len(pdf_list):,} PDFs from {pdf_folder}")
        t0 = time.perf_counter()

        seen_pdfs = set()
        for pdf_path in tqdm(pdf_list, desc="PDFs", unit="pdf"):
            try:
                self._process_pdf(pdf_path)
                seen_pdfs.add(str(pdf_path))
            except Exception as exc:
                logger.error(f"PDF failed: {pdf_path}  →  {exc}")
                self._stats["errors"] += 1

        # Finalize outputs
        self.exporter.flush_csv()

        elapsed = time.perf_counter() - t0
        self._stats["elapsed_s"]         = round(elapsed, 2)
        self._stats["pages_per_second"]  = round(
            self._stats["pages"] / max(elapsed, 1), 2
        )
        reduction = 100 * (
            1 - self._stats["proposals_kept"] /
            max(self._stats["proposals_raw"], 1)
        )
        self._stats["proposal_reduction_pct"] = round(reduction, 1)

        logger.info("─" * 60)
        logger.info(f"  PDFs processed : {self._stats['pdfs']:,}")
        logger.info(f"  Pages          : {self._stats['pages']:,}")
        logger.info(f"  Regions found  : {self._stats['regions']:,}")
        logger.info(f"  Proposal reduc.: {reduction:.1f}%")
        logger.info(f"  Throughput     : {self._stats['pages_per_second']} pages/s")
        logger.info(f"  Errors         : {self._stats['errors']}")
        logger.info(f"  Elapsed        : {elapsed:.1f}s")
        logger.info("─" * 60)

        return self._stats

    # ── Single-PDF processing ─────────────────────────────────────────────────

    def _process_pdf(self, pdf_path: Path) -> None:
        self._stats["pdfs"] += 1
        page_results = []

        for _, page_idx, rgb in self.loader.load_pdf(pdf_path):
            try:
                result = self._process_page(str(pdf_path), page_idx, rgb)
                if result:
                    page_results.append(result)
            except Exception as exc:
                logger.warning(f"  Page {page_idx} error: {exc}")
                self._stats["errors"] += 1

        # Annotated PDF (whole-document overlay)
        if self._stats.get("save_annotated_pdf", True) and page_results:
            try:
                self.exporter.create_annotated_pdf(pdf_path, page_results)
            except Exception as exc:
                logger.debug(f"Annotated PDF failed for {pdf_path}: {exc}")

    # ── Single-page processing ────────────────────────────────────────────────

    def _process_page(
        self,
        pdf_path:  str,
        page_idx:  int,
        rgb:       np.ndarray,
    ) -> Optional[dict]:
        self._stats["pages"] += 1

        # 1. Preprocess
        views  = self.processor.full_preprocess(rgb)
        gray   = views["gray"]
        binary = views["binary"]
        rgb    = views["rgb"]

        # 2. Region proposals
        boxes_raw = self.proposer.propose(binary, gray)
        self._stats["proposals_raw"] += len(boxes_raw)

        if not boxes_raw:
            return None

        # 3. Feature extraction
        X = self.extractor.extract_batch(gray, binary, boxes_raw, gray.shape)

        if X.shape[0] == 0:
            return None

        # 4. Classify + confidence filter
        labels, confs, boxes = self.classifier.predict_filtered(X, boxes_raw)
        self._stats["proposals_kept"] += len(boxes)
        self._stats["regions"]        += len(boxes)

        if not boxes:
            return None

        # 5. Export
        result = self.exporter.export_page(
            pdf_path, page_idx, rgb, boxes, labels, confs
        )
        result["image_shape"] = gray.shape  # needed for PDF annotation scaling
        return result

    # ── Training ──────────────────────────────────────────────────────────────

    def train_from_annotations(
        self,
        annotations_csv: str | Path,
        pdf_root:        str | Path = ".",
        max_samples:     Optional[int] = None,
    ) -> "LayoutPipeline":
        """
        Train the LightGBM classifier from a CSV of labelled bounding boxes.

        CSV format
        ----------
        pdf_path, page, x, y, w, h, label
        path/to/doc.pdf, 0, 120, 80, 400, 60, title
        ...

        The method renders each page, runs preprocessing + proposal,
        then extracts features for each annotated box.
        """
        import pandas as pd

        ann = pd.read_csv(annotations_csv)
        required = {"pdf_path", "page", "x", "y", "w", "h", "label"}
        if not required.issubset(ann.columns):
            raise ValueError(f"CSV must have columns: {required}")

        if max_samples:
            ann = ann.sample(min(max_samples, len(ann)), random_state=42)

        logger.info(f"Building training features from {len(ann):,} annotations …")

        X_list, y_list = [], []
        page_cache: dict = {}

        for _, row in tqdm(ann.iterrows(), total=len(ann), desc="Annotated regions"):
            key = (row.pdf_path, int(row.page))
            if key not in page_cache:
                page_cache[key] = self._load_page_views(
                    row.pdf_path, int(row.page)
                )

            views = page_cache[key]
            if views is None:
                continue

            gray, binary = views
            bbox  = (int(row.x), int(row.y), int(row.w), int(row.h))
            feat  = self.extractor.extract(gray, binary, bbox, gray.shape)
            X_list.append(feat)
            y_list.append(row.label)

        if not X_list:
            raise RuntimeError("No training samples extracted.")

        X = np.stack(X_list)
        self.classifier.train(X, y_list)
        self.classifier.save()
        logger.success(f"Training complete — model saved to {LGBM_MODEL_PATH}")
        return self

    def _load_page_views(
        self, pdf_path: str, page_idx: int
    ) -> Optional[tuple]:
        for _, idx, rgb in self.loader.load_pdf(pdf_path, page_limit=page_idx + 1):
            if idx == page_idx:
                views = self.processor.full_preprocess(rgb)
                return views["gray"], views["binary"]
        return None

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return dict(self._stats)
