# NGC 6569 workspace

This folder contains the scripts, notebooks, data, and generated output for the
NGC 6569 tidal-stream experiments.

## Layout

- `*.py`: reusable helper code for orbits, leapfrog integration, cluster setup,
  and initial conditions.
- `*.ipynb`: exploratory and validation notebooks.
- `data/`: tabular inputs and generated text data used by the notebooks.
- `milkyway/`: Agama Milky Way potential configuration files.
- `output/`: generated figures, animations, cached runs, and simulation products.
- `output/legacy_root_outputs/`: older generated media files that used to live
  at the top level but were kept instead of deleted because newer `output/`
  copies differ.

Keep new generated files under `data/` or `output/` rather than at the top level.
