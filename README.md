# Multistage Digital Health Adoption Using HINTS 6 and 7

This repository contains the core Python code used to analyze stage-specific digital health engagement in the U.S. National Cancer Institute's Health Information National Trends Survey (HINTS) 6 and HINTS 7.

The analysis distinguishes four outcomes: portal activation, routinization, functional depth, and cross-platform integration. It combines survey-weighted regression, official jackknife replicate-weight inference, robustness and heterogeneity analyses, out-of-time validation, XGBoost, and SHAP interpretation.

## Repository structure

```text
analysis/
  run_experiment.py             Main harmonization, regression, ML, and figures
  run_extended_experiments.py   Robustness, heterogeneity, calibration, and validation
  regenerate_figures.py         Rebuild figures from generated result tables
data_extracted/                  Local HINTS Stata files (not tracked)
requirements.txt                Python dependencies
```

Raw data, processed data, fitted models, and generated outputs are intentionally excluded from this repository.

## Requirements

- Python 3.12 recommended
- HINTS 6 and HINTS 7 public-use Stata datasets

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data setup

Download the HINTS 6 and HINTS 7 public-use Stata packages from the official NCI HINTS download page:

https://hints.cancer.gov/data/download-data.aspx

Extract the files anywhere under `data_extracted/`. The scripts search recursively for these filenames:

```text
hints6_public.dta
hints7_public.dta
```

The expected minimal layout is:

```text
data_extracted/
  HINTS6/hints6_public.dta
  HINTS7/hints7_public.dta
```

The HINTS public-use data are not redistributed in this repository. Users are responsible for following the NCI data-use terms and documentation.

## Run the analyses

Run commands from the repository root.

Primary analysis:

```bash
python analysis/run_experiment.py
```

Extended robustness and validation analyses:

```bash
python analysis/run_extended_experiments.py
```

After both analyses have completed, regenerate all publication figures without refitting the models:

```bash
python analysis/regenerate_figures.py
```

Generated artifacts are written to:

```text
data_processed/
outputs/tables/
outputs/figures/
outputs/models/
```

## Reproducibility notes

- HINTS final survey weights and all 50 official jackknife replicate weights are used.
- HINTS 6 is used for model training and HINTS 7 for out-of-time testing.
- Randomized procedures use the fixed seed `20260915`.
- The scripts create required output directories automatically.
- Figure styling requests Times New Roman and falls back to an available serif font if it is not installed.

## Study scope

HINTS is a repeated cross-sectional survey. The stage-specific outcomes represent conceptually ordered dimensions of engagement and should not be interpreted as longitudinal transitions or causal effects.
