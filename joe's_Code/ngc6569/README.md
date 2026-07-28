# NGC 6569 workspace

This folder contains the scripts, notebooks, data, and generated output for the
NGC 6569 tidal-stream experiments.

## Layout

- `NGC6569_Dev3_original.ipynb`: original notebook kept at the top level.
- `helpers/`: reusable Python helper code for orbits, leapfrog integration,
  cluster setup, snapshot storage, validation utilities, and initial-condition
  fitting.
- `experiments/dev3/`: Dev3 exploratory and plotting notebooks.
- `experiments/chen/`: Chen-style validation and comparison notebooks.
- `data/`: tabular inputs and generated text data used by the notebooks.
- `data/legacy_root_data/`: older top-level text data preserved because the
  canonical `data/` copies differ.
- `milkyway/`: Agama Milky Way potential configuration files.
- `output/`: generated figures, animations, cached runs, and simulation products.
- `output/figures/orbits/`: orbit figure PDFs.
- `output/figures/mass_loss/`: mass-loss and parameter-study figure PDFs.
- `output/figures/observables/`: observable-density figure PDFs.
- `output/figures/tidal_tail/`: tidal-tail and stream-comparison figure PDFs.
- `output/legacy_root_outputs/`: older generated media files that used to live
  at the top level but were kept instead of deleted because newer output copies
  differ.

Keep new generated files under `data/` or `output/` rather than at the top level.
