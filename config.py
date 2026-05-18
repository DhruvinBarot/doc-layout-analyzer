"""
config.py
Centralized configuration for the Document Layout Analysis pipeline.
All tunable knobs live here — no magic numbers buried in modules.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple


# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR        = Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data"
MODELS_DIR      = DATA_DIR / "models"
RESULTS_DIR     = DATA_DIR / "results"
SAMPLE_PDF_DIR  = DATA_DIR / "sample_pdfs"

LGBM_MODEL_PATH = MODELS_DIR / "lgbm_layout.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"


# ─── PDF Ingestion ─────────────────────────────────────────────────────────────

@dataclass
class IngestionConfig:
    dpi: int = 150                  # Render DPI (150 is a good speed/quality balance)
    max_pages: int = 50             # Cap pages per PDF (None = no cap)
    colorspace: str = "RGB"         # "RGB" or "GRAY"
    n_workers: int = 4              # Parallel PDF workers


# ─── Preprocessing ─────────────────────────────────────────────────────────────

@dataclass
class PreprocessConfig:
    target_width: int   = 1200      # Resize width (preserves aspect ratio)
    denoise_h: int      = 10        # Luminance denoising strength
    binarize: bool      = True      # Otsu binarization for proposal step
    deskew: bool        = True      # Correct page rotation
    deskew_max_angle: float = 10.0  # Max correction angle (degrees)


# ─── Region Proposal ──────────────────────────────────────────────────────────

@dataclass
class ProposalConfig:
    # Morphological closing kernel to merge nearby blobs
    morph_kernel_size: Tuple[int, int] = (5, 5)
    morph_iterations: int = 2

    # Connected-component filters (area in pixels²)
    min_area: int  = 400
    max_area: int  = 800_000

    # Aspect ratio guard (width/height)
    min_aspect: float = 0.05
    max_aspect: float = 20.0

    # Non-max suppression overlap threshold (IoU)
    nms_iou_thresh: float = 0.4

    # MSER parameters (for additional text-region hints)
    mser_delta: int       = 5
    mser_min_area: int    = 60
    mser_max_area: int    = 14_400
    mser_max_variation: float = 0.25

    # Padding added around each proposal bbox
    bbox_padding: int = 4


# ─── Feature Extraction ────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    # HOG
    hog_orientations: int      = 9
    hog_pixels_per_cell: Tuple = (8, 8)
    hog_cells_per_block: Tuple = (2, 2)
    hog_resize: Tuple[int,int] = (64, 64)

    # LBP (Local Binary Patterns) texture
    lbp_radius: int      = 3
    lbp_n_points: int    = 24          # = 8 * radius
    lbp_n_bins: int      = 26          # histogram bins

    # Geometric / positional features
    include_geometric: bool    = True   # area, aspect ratio, fill ratio …
    include_positional: bool   = True   # normalized x,y position on page
    include_histogram: bool    = True   # pixel intensity histogram (16 bins)


# ─── Classification ────────────────────────────────────────────────────────────

# Region labels used during training / inference
REGION_LABELS: List[str] = [
    "text",
    "title",
    "figure",
    "table",
    "caption",
    "list",
    "header",
    "footer",
    "equation",
    "other",
]

@dataclass
class ClassifierConfig:
    # LightGBM hyper-parameters
    n_estimators: int    = 600
    num_leaves: int      = 63
    learning_rate: float = 0.05
    max_depth: int       = -1          # -1 = unlimited
    min_child_samples: int = 20
    subsample: float     = 0.8
    colsample_bytree: float = 0.8
    n_jobs: int          = -1
    random_state: int    = 42
    class_weight: str    = "balanced"

    # Inference
    confidence_threshold: float = 0.35 # Discard low-confidence proposals


# ─── Output ────────────────────────────────────────────────────────────────────

@dataclass
class OutputConfig:
    save_json: bool          = True
    save_csv: bool           = True
    save_annotated_pdf: bool = True
    save_crops: bool         = False    # Cropped region images (can be large)
    annotation_alpha: float  = 0.25    # Opacity of region fill in annotated PDF

    # Per-label colours (BGR for OpenCV, hex for PDF)
    label_colors: dict = field(default_factory=lambda: {
        "text":     "#2196F3",
        "title":    "#9C27B0",
        "figure":   "#FF9800",
        "table":    "#4CAF50",
        "caption":  "#009688",
        "list":     "#00BCD4",
        "header":   "#F44336",
        "footer":   "#E91E63",
        "equation": "#FFEB3B",
        "other":    "#9E9E9E",
    })


# ─── Global defaults (instantiated) ───────────────────────────────────────────

INGESTION   = IngestionConfig()
PREPROCESS  = PreprocessConfig()
PROPOSAL    = ProposalConfig()
FEATURES    = FeatureConfig()
CLASSIFIER  = ClassifierConfig()
OUTPUT      = OutputConfig()
