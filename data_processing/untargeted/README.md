# Untargeted Benchmark Analysis

This directory contains parameterized QE HF and TripleTOF 6600 benchmark
workflows together with two XCMS ablation workflows. Dataset-specific file
names and calculation settings are defined in
`/path/to/paper_supplement/untargeted/datasets.json`.

## Run

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r untargeted/requirements.txt
python untargeted/run_all.py \
  --data-dir /path/to/paper_supplement/untargeted \
  --output-dir results/untargeted
```

The command performs the following steps:

1. verifies analytical inputs against `source_manifest.tsv`;
2. calculates qualitative metrics, CV values, differential-metabolite
   recovery, fold-change errors, and overall quantitative accuracy;
3. analyzes the supplied XCMS alignment and peak-picking ablation matrices;
4. writes calculated tables as TSV and figures as 300-dpi PNG and SVG files.

Use `--skip-ablation` to run only the main benchmark analysis.

## Main definitions

- Detection is defined as a non-missing, non-zero area. Rows missing in more
  than half of the samples are removed before qualitative evaluation.
- Fold change is `(mean_B + 1e-6) / (mean_A + 1e-6)`. An absolute relative
  fold-change error of at least 20% is a quantitative error.
- Overall error IDs are the union of qualitative and quantitative error IDs.
- Differential metabolites use a two-sided Welch test with unadjusted
  `p < 0.05`, FC up/down thresholds from `datasets.json`, and no
  multiple-testing correction.
- CV uses population standard deviation (`ddof=0`). The TOF analysis uses the
  940 features shared by all three available methods; QE uses the
  method-specific feature scopes defined in `datasets.json`.

## Evaluation universes

- QE HF: 1,551 evaluated entries, consisting of 836 true targets and 715
  curated decoys.
- TripleTOF 6600: 1,700 evaluated entries, consisting of 970 true targets and
  730 decoys.

## False-positive dataset construction

`data_preparation/generate_false_positive_dataset.py` constructs a controlled
false-positive set from one centroided mzML file. It iterates over scan-level
centroid peaks from MS1 spectra, keeps peaks whose absolute m/z difference
from every true positive is at least 1 Da, excludes intensities above 100,000,
and samples 836 candidates with random seed 42.

```bash
python untargeted/data_preparation/generate_false_positive_dataset.py \
  --tp-file peak_list.csv \
  --mzml-file mzML/SA1.mzML \
  --output-csv TP_FP_eval_strict.csv \
  --plot-output TP_FPs_mzRT.png \
  --n-fp 836 \
  --min-mz-diff-da 1.0 \
  --max-intensity 100000 \
  --seed 42
```

The CSV output columns are `Compound Name`, `mz`, `RT`, and `Intensity`.
True-positive intensity values are empty because the reference list contains
only compound names, m/z, and RT. Candidate peaks are not grouped across scans
or deduplicated into chromatographic features.

For development checks, install `requirements-dev.txt` and run `pytest` from
the `untargeted/` directory.

## Output figures

The main workflow writes the following figures for both datasets:

- Precision, Recall, and F1 grouped bars;
- peak-area CV boxplots;
- differential-metabolite TP, FP, and FN stacked bars;
- fold-change relative-error boxplots;
- combined QE and TOF overall-accuracy bars.

The ablation workflow writes individual accuracy panels and a combined QE/TOF
three-variant accuracy panel.

## Optional XCMS workflows

The scripts in `ablation/xcms_modules/` require mzML inputs, mapping support
tables, R, and the xcms package.

Generated `results/` directories are ignored by Git.

