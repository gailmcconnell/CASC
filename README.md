# CASC
Content-Aware Compaction of Sparse Microscopy Images

An open-source Python pipeline for the content-aware spatial compaction of sparse microscopy datasets.

CASC reorganises segmented foreground structures into a compact image representation by removing empty background regions and repacking objects into a reduced spatial domain while preserving object-local pixel intensities exactly.

Repository Structure
CASC/
│
├── CASC_v1.0.py
├── README.md
├── example_data/
├── figures/
├── results/
└── LICENSE
Installation
Requirements

Python 3.10+

Dependencies
pip install numpy pandas tifffile pillow scipy scikit-image matplotlib

Usage

Edit the input/output paths in the script:

INPUT_PATH = r"path_to_input_image.tif"

OUTPUT_TIF = r"path_to_output_image.tif"
OUTPUT_METRICS = r"path_to_metrics.json"
OUTPUT_COMPONENTS = r"path_to_components.csv"

Run:

python CASC_v1.0.py
Output Files

CASC produces:

Output	Description
.tif	Compacted microscopy image
.json	Quantitative metrics
.csv	Component metadata
Quantitative Metrics

If you use CASC in academic work, please cite:

McConnell G. Content-Aware Spatial Compaction of Sparse Microscopy Images (CASC). University of Strathclyde.


Gail McConnell
Strathclyde Institute of Pharmacy and Biomedical Sciences (SIPBS)
University of Strathclyde
