# Manual Peak Curator

This is a standalone manual LC-MS peak integration tool extracted from MetSynQ development. It does not import MetSynQ source code.

## Requirements

- Windows with Python 3.10 or newer.
- Python packages listed in `requirements.txt`.
- `tkinter` is required for the GUI. It is included with the standard Windows Python installer from python.org.

## Install

Double-click:

```text
install_dependencies.bat
```

Or run:

```powershell
python -m pip install -r requirements.txt
```

## Run

Double-click:

```text
run_tool.bat
```

Or run:

```powershell
python manual_peak_curator.py
```

## Input Feature Table

Required columns:

```text
feature_id,mz,rt
```

Recommended columns:

```text
feature_id,mz,rt,rtmin,rtmax,name,polarity
```

`rt`, `rtmin`, and `rtmax` are in minutes. Extra sample area columns are allowed and ignored by the GUI.

## Workflow

1. Load the POS or NEG mzML folder.
2. Load the matching POS or NEG feature table.
3. Select a feature in the left list.
4. The chromatogram panel shows samples in a scrollable 3-column grid. Enlarge the window to see more samples at once, or use the right scrollbar/mouse wheel to scroll down.
5. Each small EIC plot is titled by sample name.
6. Drag the RT range inside a sample plot to update that sample only.
7. Use `Apply To All Samples` only when one boundary should be shared by all samples for the selected feature.
8. Export curated results.

## Outputs

The tool exports:

- `manual_curated_peak_areas_long.csv`
- `manual_curated_peak_areas_matrix.csv`
- `manual_curation_edits.csv`
- `manual_curation_metadata.json`
