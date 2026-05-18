"""
ingestion/pdf_loader.py
-----------------------
Converts PDF files → page images using PyMuPDF (fitz).
Supports parallel processing of large PDF corpora (80K+ files).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import fitz                       # PyMuPDF
import numpy as np
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import INGESTION, IngestionConfig


# ─── Types ────────────────────────────────────────────────────────────────────

PageImage = np.ndarray            # HxWx3 uint8
PageRecord = Tuple[str, int, PageImage]   # (pdf_path, page_index, image)


# ─── Core loader ──────────────────────────────────────────────────────────────

class PDFLoader:
    """
    Renders PDF pages to NumPy arrays.

    Usage
    -----
    loader = PDFLoader()
    for pdf_path, page_idx, image in loader.load_pdf("doc.pdf"):
        ...  # image is HxWx3 uint8 RGB
    """

    def __init__(self, cfg: IngestionConfig = INGESTION):
        self.cfg = cfg
        self._matrix = fitz.Matrix(cfg.dpi / 72, cfg.dpi / 72)   # scale factor

    # ── single PDF ────────────────────────────────────────────────────────────

    def load_pdf(
        self,
        pdf_path: str | Path,
        page_limit: Optional[int] = None,
    ) -> Generator[PageRecord, None, None]:
        """
        Yield (pdf_path_str, page_index, rgb_image) for every page.

        Parameters
        ----------
        pdf_path   : path to the PDF file
        page_limit : override cfg.max_pages for this call
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            return

        limit = page_limit or self.cfg.max_pages

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            logger.error(f"Cannot open {pdf_path}: {exc}")
            return

        n_pages = min(len(doc), limit) if limit else len(doc)

        for page_idx in range(n_pages):
            try:
                page = doc[page_idx]
                pix  = page.get_pixmap(matrix=self._matrix, colorspace=fitz.csRGB)
                img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if img.shape[2] == 4:               # drop alpha channel if present
                    img = img[:, :, :3]
                yield str(pdf_path), page_idx, img
            except Exception as exc:
                logger.warning(f"Page {page_idx} error in {pdf_path}: {exc}")

        doc.close()

    # ── batch / folder ────────────────────────────────────────────────────────

    def load_folder(
        self,
        folder: str | Path,
        recursive: bool = True,
    ) -> Generator[PageRecord, None, None]:
        """
        Stream all PDF pages from a folder (optionally recursive).
        Pages from all PDFs are yielded in sequence.
        """
        folder   = Path(folder)
        pattern  = "**/*.pdf" if recursive else "*.pdf"
        pdf_list = sorted(folder.glob(pattern))

        if not pdf_list:
            logger.warning(f"No PDFs found in {folder}")
            return

        logger.info(f"Found {len(pdf_list):,} PDFs in {folder}")

        for pdf_path in tqdm(pdf_list, desc="Loading PDFs", unit="pdf"):
            yield from self.load_pdf(pdf_path)

    # ── parallel batch (for 80K scale) ────────────────────────────────────────

    def load_folder_parallel(
        self,
        folder: str | Path,
        recursive: bool = True,
    ) -> Generator[List[PageRecord], None, None]:
        """
        Load PDFs in parallel using ProcessPoolExecutor.
        Yields *lists* of PageRecords per PDF (suitable for downstream batching).

        Note: Each worker loads one full PDF; results are unordered.
        """
        folder   = Path(folder)
        pattern  = "**/*.pdf" if recursive else "*.pdf"
        pdf_list = sorted(folder.glob(pattern))

        logger.info(
            f"Parallel loading {len(pdf_list):,} PDFs "
            f"with {self.cfg.n_workers} workers"
        )

        with ProcessPoolExecutor(max_workers=self.cfg.n_workers) as pool:
            futures = {
                pool.submit(_load_pdf_worker, str(p), self.cfg): p
                for p in pdf_list
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="PDF workers",
                unit="pdf",
            ):
                pdf_path = futures[fut]
                try:
                    records = fut.result()
                    yield records
                except Exception as exc:
                    logger.error(f"Worker failed for {pdf_path}: {exc}")

    # ── convenience: count pages ──────────────────────────────────────────────

    @staticmethod
    def count_pages(pdf_path: str | Path) -> int:
        try:
            doc = fitz.open(str(pdf_path))
            n   = len(doc)
            doc.close()
            return n
        except Exception:
            return 0

    @staticmethod
    def count_folder(folder: str | Path, recursive: bool = True) -> dict:
        folder  = Path(folder)
        pattern = "**/*.pdf" if recursive else "*.pdf"
        stats   = {"pdfs": 0, "pages": 0, "errors": 0}
        for p in folder.glob(pattern):
            stats["pdfs"] += 1
            n = PDFLoader.count_pages(p)
            if n:
                stats["pages"] += n
            else:
                stats["errors"] += 1
        return stats


# ─── Worker (must be top-level for pickling) ──────────────────────────────────

def _load_pdf_worker(pdf_path: str, cfg: IngestionConfig) -> List[PageRecord]:
    loader = PDFLoader(cfg)
    return list(loader.load_pdf(pdf_path))


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_loader.py <pdf_or_folder>")
        sys.exit(1)

    target = Path(sys.argv[1])
    loader = PDFLoader()

    if target.is_dir():
        for pdf_path, page_idx, img in loader.load_folder(target):
            logger.info(f"{pdf_path}  page {page_idx}  shape={img.shape}")
    else:
        for pdf_path, page_idx, img in loader.load_pdf(target):
            logger.info(f"page {page_idx}  shape={img.shape}  dtype={img.dtype}")
