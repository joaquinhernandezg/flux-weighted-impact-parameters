# Flux-Weighted Impact Parameters

Small, reusable Python utilities for working with flux-weighted impact parameters, azimuthal angles, lensing deflection maps, and segmentation products in gravitationally lensed systems.

This repository is intentionally lightweight: it is structured so the code can be imported from notebooks, scripts, or other projects, without forcing a large library or framework around it.

## Repository layout

```text
flux-weighted-impact-parameters/
├── pyproject.toml
├── src/
│   └── flux_weighted_impact_parameters/
├── notebooks/
├── examples/
└── scripts/
```

## Installation

Clone the repository and install it in editable mode:

```bash
pip install -e .
```

That makes the package importable from anywhere in your environment:

```python
from flux_weighted_impact_parameters import make_impact_parameter_map
from flux_weighted_impact_parameters import make_azimuthal_angle_map
from flux_weighted_impact_parameters import get_binned_pixel_stats
```

If you do not want to install it, you can also add the repository to your `PYTHONPATH`, but editable installation is the cleanest option for notebooks and external scripts.

## Current modules

- `lensing.py`: lensing-matrix scaling, delensing, magnification, shear, and critical-line helpers
- `impact_parameter_map.py`: source-plane impact-parameter maps
- `azimuthal_angle_map.py`: azimuthal-angle maps relative to a galaxy major axis
- `flux_contribution_maps.py`: contribution kernels and flux-fraction utilities
- `impact_parameter_distribution.py`: weighted summary statistics for binned pixels
- `binned_pixel_stats.py`: per-spaxel summary products
- `correlation.py`: kernel correlation and overlap metrics
- `segmentation_utils.py`: segmentation helpers
- `make_segmentation_mask.py`: segmentation-map and mask workflow

## Adding examples later

The repository already includes placeholder directories for:

- `notebooks/` for worked examples and walkthroughs
- `examples/` for small sample inputs or reference outputs
- `scripts/` for reproducible command-line workflows

Large FITS files and generated figures are intentionally excluded by `.gitignore`. If you want to share a few compact example files, add only curated, lightweight datasets.

## Suggested next steps

1. Initialize git inside this directory.
2. Review the README title and project description.
3. Add one notebook showing a minimal end-to-end example.
4. Add a short script in `scripts/` that reproduces one figure or map.

## Suggested repository names

- `flux-weighted-impact-parameters`
- `lensing-impact-parameter-tools`
- `arc-flux-mapping`
- `lensed-spaxel-footprints`
- `source-plane-impact-maps`
