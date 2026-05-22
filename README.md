# CASC
Content-Aware Compaction of Sparse Microscopy Images

An open-source Python pipeline for the content-aware spatial compaction of sparse microscopy datasets.

CASC reorganises segmented foreground structures into a compact image representation by removing empty background regions and repacking objects into a reduced spatial domain while preserving object-local pixel intensities exactly.

Overview

Microscopy datasets are frequently spatially sparse, where biologically relevant structures occupy only a small fraction of the field of view (FOV). Large image regions therefore contain little or no useful signal, resulting in:

inflated file sizes,
inefficient storage,
increased computational overhead,
slower image processing and visualisation,
and unnecessary data transfer costs.

CASC addresses this problem by treating spatial arrangement itself as a removable source of redundancy.

Rather than preserving the original topology of the image, CASC:

segments foreground objects,
extracts connected components with contextual padding,
and repacks them into a compact image domain.

The method intentionally destroys global spatial organisation while preserving object-local morphology and pixel intensity information.

Key Features
Adaptive foreground segmentation
Connected-component extraction
Contextual object padding
Heuristic shelf-based irregular object packing
Exact preservation of object pixel intensities
Quantitative spatial preservation metrics
Open-source Python implementation
TIFF-compatible workflow
Pipeline Overview

The CASC workflow consists of:

Intensity normalisation
Adaptive histogram equalisation
Gaussian smoothing
Otsu thresholding
Morphological cleanup
Connected-component extraction
Object cropping with contextual padding
Irregular spatial packing
Compact image rendering
Quantitative metric generation

The implementation supports:

grayscale microscopy images,
RGB/RGBA microscopy images,
multi-channel image stacks,
and mesoscale microscopy datasets.
Conceptual Illustration
Original sparse image
┌─────────────────────────────┐
│  cell       cell            │
│                             │
│                             │
│               nucleus       │
│                             │
│   organelle                 │
└─────────────────────────────┘

↓

Compacted representation
┌──────────────┐
│cellcell      │
│nucleus       │
│organelle     │
└──────────────┘
Scientific Scope

CASC is intended for:

✅ Object-centric microscopy workflows

Examples include:

cell morphology analysis,
object detection,
particle quantification,
organelle analysis,
segmentation workflows,
fluorescence microscopy,
sparse mesoscale imaging.

CASC is not suitable for:

❌ Spatial-topology-dependent applications

Examples include:

spatial transcriptomics,
tissue architecture analysis,
neighbourhood analysis,
spatial ecology,
topology-preserving reconstruction tasks.

The method intentionally destroys:

local neighbourhood relationships,
global spatial organisation,
and pairwise spatial ordering.
Example Results

Reported performance includes:

390-fold reduction in file size in sparse microscopy datasets

substantial reduction in background area
high object-level preservation
exact retention of object pixel intensities
substantial increase in foreground density

Spatial preservation metrics demonstrated:

near-zero spatial Jaccard retention,
near-zero spatial rank correlation,

confirming deliberate destruction of spatial topology.

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

The pipeline computes:

foreground retention fraction,
area reduction fraction,
density gain,
nearest-neighbour retention,
pairwise distance rank correlation,
nearest-neighbour distortion,
object count preservation,
intensity preservation metrics.
Important Parameters
Parameter	Description
PADDING	Context margin around extracted objects
MIN_OBJECT_AREA	Minimum connected-component size
ALLOW_ROTATIONS	Allow object rotations during packing
DILATE_MASK_FOR_PACKING	Packing collision spacing
VISUAL_HALO_RADIUS	Rendering halo around objects
Current Limitations
Spatial topology is intentionally destroyed
Packing is heuristic rather than globally optimal
Performance depends on segmentation quality
Large datasets may require substantial RAM
Dense images provide limited compaction benefit
Citation

If you use CASC in academic work, please cite:

McConnell G. Content-Aware Spatial Compaction of Sparse Microscopy Images (CASC). University of Strathclyde.

License

MIT License

Author

Gail McConnell
Strathclyde Institute of Pharmacy and Biomedical Sciences (SIPBS)
University of Strathclyde
