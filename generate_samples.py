"""
generate_samples.py
--------------------
Creates sample PDFs and a matching annotations.csv for testing the pipeline.
Run once before training:  python generate_samples.py
"""

import csv
import random
from pathlib import Path
import fitz  # PyMuPDF

OUTPUT_PDF_DIR = Path("data/sample_pdfs")
ANNOTATIONS_CSV = Path("annotations.csv")

OUTPUT_PDF_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["title", "text", "figure", "table", "caption", "header", "footer"]

LABEL_COLORS = {
    "title":   (0.6, 0.0, 0.8),
    "text":    (0.1, 0.4, 0.9),
    "figure":  (1.0, 0.6, 0.0),
    "table":   (0.0, 0.7, 0.3),
    "caption": (0.0, 0.6, 0.6),
    "header":  (0.9, 0.1, 0.1),
    "footer":  (0.9, 0.1, 0.5),
}

SAMPLE_TEXTS = {
    "title":   ["Introduction", "Related Work", "Methodology",
                "Experimental Results", "Conclusion", "Abstract"],
    "text":    [
        "This paper presents a novel approach to document layout analysis "
        "using classical computer vision techniques combined with gradient "
        "boosted trees for classification.",
        "The proposed method significantly outperforms baseline approaches "
        "in terms of both speed and accuracy across a diverse set of "
        "document types including academic papers, forms, and reports.",
        "We evaluate our system on a dataset of over 80,000 PDF documents "
        "spanning multiple domains. Results demonstrate robust performance "
        "even under challenging conditions such as skewed scans.",
    ],
    "caption": ["Figure 1: Overview of the proposed pipeline.",
                "Table 2: Comparison with state-of-the-art methods.",
                "Figure 3: Sample output on a research paper page."],
    "header":  ["Document Layout Analysis — CVPR 2024"],
    "footer":  ["Page 1", "Page 2", "Page 3", "Page 4"],
}


def make_pdf(pdf_path: Path, n_pages: int = 3) -> list:
    """Create a PDF with realistic layout regions. Returns annotation rows."""
    doc   = fitz.open()
    rows  = []
    rng   = random.Random(hash(str(pdf_path)))

    for page_num in range(n_pages):
        page  = doc.new_page(width=595, height=842)   # A4 points
        regions = _layout_regions(page_num, rng)

        for (x, y, w, h, label) in regions:
            color = LABEL_COLORS.get(label, (0.5, 0.5, 0.5))
            rect  = fitz.Rect(x, y, x + w, y + h)

            # Draw filled rectangle to simulate a region
            page.draw_rect(rect, color=color, fill=color,
                           fill_opacity=0.15, width=1)

            # Insert representative text
            texts = SAMPLE_TEXTS.get(label, ["Lorem ipsum dolor sit amet."])
            txt   = rng.choice(texts)
            fontsize = 14 if label == "title" else (8 if label in ("header","footer","caption") else 10)
            page.insert_textbox(
                rect, txt,
                fontsize=fontsize,
                color=(0, 0, 0),
                align=fitz.TEXT_ALIGN_LEFT,
            )

            # Convert PDF points → pixel coords at 150 DPI
            scale = 150 / 72
            rows.append({
                "pdf_path": str(pdf_path),
                "page":     page_num,
                "x":        int(x * scale),
                "y":        int(y * scale),
                "w":        int(w * scale),
                "h":        int(h * scale),
                "label":    label,
            })

    doc.save(str(pdf_path))
    doc.close()
    return rows


def _layout_regions(page_num: int, rng: random.Random) -> list:
    """
    Return a list of (x, y, w, h, label) in PDF points for one page.
    Simulates a realistic academic paper layout.
    """
    regions = []

    # Header
    regions.append((30, 10, 535, 20, "header"))

    # Title (first page only)
    if page_num == 0:
        regions.append((80, 50, 435, 35, "title"))

    # Two-column text blocks
    col_w = 250
    y_pos = 110 if page_num == 0 else 40
    for col_x in (30, 315):
        for _ in range(rng.randint(2, 4)):
            h = rng.randint(60, 130)
            if y_pos + h > 760:
                break
            regions.append((col_x, y_pos, col_w, h, "text"))
            y_pos += h + rng.randint(8, 18)

        # Occasionally add a figure or table in this column
        if rng.random() > 0.5 and y_pos + 80 < 760:
            kind = rng.choice(["figure", "table"])
            fh   = rng.randint(80, 140)
            regions.append((col_x, y_pos, col_w, fh, kind))
            y_pos += fh + 5
            cap_h = 18
            regions.append((col_x, y_pos, col_w, cap_h, "caption"))
            y_pos += cap_h + 10

    # Footer
    regions.append((30, 815, 535, 16, "footer"))

    return regions


def main():
    all_rows = []
    n_pdfs   = 10   # generate 10 sample PDFs

    print(f"Generating {n_pdfs} sample PDFs in {OUTPUT_PDF_DIR}/ ...")
    for i in range(n_pdfs):
        pdf_path = OUTPUT_PDF_DIR / f"sample_{i+1:03d}.pdf"
        rows     = make_pdf(pdf_path, n_pages=random.randint(2, 4))
        all_rows.extend(rows)
        print(f"  Created {pdf_path}  ({len(rows)} annotated regions)")

    # Write CSV
    with open(ANNOTATIONS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pdf_path","page","x","y","w","h","label"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone! {len(all_rows)} annotations written to {ANNOTATIONS_CSV}")
    print(f"\nNext step:")
    print(f"  python main.py train annotations.csv")


if __name__ == "__main__":
    main()
