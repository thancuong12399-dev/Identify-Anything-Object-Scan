import fitz  # PyMuPDF
import shutil
from pathlib import Path

# Paths
report_pdf = Path(r"d:\AI\report\main.pdf")
figures_dir = Path(r"d:\AI\report\figures")
artifact_dir = Path(r"C:\Users\thanp\.gemini\antigravity-ide\brain\198ab228-5a4f-460f-aa45-bd321189b2cc")

artifact_dir.mkdir(parents=True, exist_ok=True)

# 1. Copy 5 key figures to artifact directory
figures = ["system_design.png", "eda_plots.png", "roc_curve.png", "ui_screenshot.png", "team_photo.png"]
for fig in figures:
    src = figures_dir / fig
    dst = artifact_dir / fig
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied {fig} to artifact dir: {dst}")

# 2. Render PDF pages
doc = fitz.open(report_pdf)
print(f"Total pages in PDF: {len(doc)}")

# Render selected pages with 2x zoom (144 dpi) for crisp viewing
zoom = 2.0
mat = fitz.Matrix(zoom, zoom)

pages_to_render = [
    (0, "page_01_cover.png"),           # Cover page
    (1, "page_02_title_detail.png"),    # Title detail page with 2 authors
    (5, "page_06_team_roles.png"),      # Roles and milestones table
    (10, "page_11_architecture.png"),   # System design figure in report
    (15, "page_16_benchmark.png"),      # Model benchmarks and ROC/mAP curves
    (17, "page_18_ui.png"),             # UI dashboard in report
    (19, "page_20_team_review.png"),    # Team photo and comments
]

for page_idx, out_name in pages_to_render:
    if page_idx < len(doc):
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=mat)
        out_path = artifact_dir / out_name
        pix.save(str(out_path))
        print(f"Rendered Page {page_idx+1} -> {out_path}")

print("All visuals successfully generated!")
