# Spectra Reductions 2026

Research workspace for the NGC 6569 AAT Ca II triplet spectroscopy analysis.

## Directory Layout

- `notebooks/` - Jupyter notebooks for spectrum reduction, plotting, SED fitting, and the integrated CaT + SPACE/SP_Ace + SED pipeline.
- `scripts/` - Batch shell scripts for SPACE/SP_Ace runs.
- `data/raw/` - Original/raw data archives, including the AAT FITS spectra tarball.
- `data/interim/` - Intermediate reduction products such as CaT summary tables.
- `data/final/` - Final machine-readable catalogs, paper tables, candidate-member tables, and quality summaries.
- `docs/manuals/` - Workflow manuals, student guides, and setup instructions.
- `docs/papers/` - Reference papers and project writeups.
- `docs/presentations/` - Slide decks.
- `docs/correspondence/` - Email or correspondence exports.

## Main Workflow

1. Unpack `data/raw/aat_spectra_2.tar.gz` when FITS spectra are needed locally.
2. Use `notebooks/AAT_SPEC_student_clean_voigt_v3.ipynb` for one-spectrum CaT reduction and SP_Ace text export.
3. Use scripts in `scripts/` to run SPACE/SP_Ace over exported `aat_*.txt` spectra.
4. Use `notebooks/NGC6569_CaT_SPACE_SED_full_pipeline_v9.ipynb` for the integrated catalog-generation workflow.
5. Final tables are written under `data/final/`.

## Local Environment

This repo has a Python 3.11 virtual environment in `.venv` and project paths in `.env`.

Activate it:

```bash
source .venv/bin/activate
set -a; source .env; set +a
```

Launch Jupyter:

```bash
jupyter notebook
```

In Jupyter, choose the kernel named `Python (.venv: SpectraReductions2026)`.

## Notes

Some notebooks and scripts contain hard-coded local paths from the original analysis environment. Update those paths before rerunning the workflow on a new machine.
