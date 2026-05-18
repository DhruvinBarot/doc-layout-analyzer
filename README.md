# DocScan — Document Layout Analysis at Scale

DocScan is a pipeline that takes a folder of PDFs and automatically figures out what's on each page — titles, paragraphs, figures, tables, headers, footers, and more. It processes 80,000+ PDFs without needing a GPU or any deep learning setup.

The core idea is simple: instead of training a heavy neural network, we use OpenCV to find regions on a page, extract descriptive features from each region, and let LightGBM classify them. It's fast, interpretable, and works on any machine.

---

## What's inside

- **Ingestion** — renders PDF pages to images using PyMuPDF
- **Preprocessing** — resizes, deskews, and binarizes each page
- **Region Proposal** — finds candidate regions using connected components and MSER, then cuts duplicates down by ~60% with NMS
- **Feature Extraction** — describes each region with HOG, LBP texture, intensity histograms, geometric shape stats, and page position
- **Classification** — LightGBM model trained with 5-fold CV predicts one of 10 region types
- **Export** — saves results as JSON, CSV, and colour-coded annotated PDFs

---

## Getting started

```bash
git clone https://github.com/YOUR_USERNAME/doc-layout-analyzer.git
cd doc-layout-analyzer
pip install -r requirements.txt
```

Generate sample PDFs to test with:
```bash
python generate_samples.py
```

Train the model:
```bash
python main.py train annotations.csv
```

Run on your PDFs:
```bash
python main.py run data/sample_pdfs/
```

Results land in `data/results/` — a CSV with every detected region and an annotated PDF for each file.

---

## Region types

`text` · `title` · `figure` · `table` · `caption` · `list` · `header` · `footer` · `equation` · `other`

---

## CLI

```bash
python main.py run <folder>          # run on a folder of PDFs
python main.py train <csv>           # train from labelled annotations
python main.py demo <file.pdf>       # run on a single PDF
python main.py stats <folder>        # count pages without processing
```

---

## Training on your own data

Make a CSV with one row per labelled region:

```
pdf_path,page,x,y,w,h,label
docs/paper.pdf,0,120,80,400,60,title
docs/paper.pdf,0,120,160,620,480,text
```

Then run `python main.py train annotations.csv`. The model saves to `data/models/` and is ready to use immediately.

---

## Results

Ran on 15 real arXiv papers (169 pages) — extracted 43,052 regions with zero errors. Training on the synthetic dataset takes under 2 minutes on a standard laptop CPU.

---

## Tech stack

PyMuPDF · OpenCV · scikit-image · LightGBM · pandas · typer · loguru

---

## License

MIT
