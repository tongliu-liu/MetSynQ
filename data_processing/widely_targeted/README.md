# Widely Targeted Benchmark Analysis

This directory contains the widely targeted benchmark calculation and figure
generation workflow.

## Expected layout

```text
repository_root/widely_targeted/                # analysis code
/path/to/paper_supplement/widely_targeted/      # analysis data
repository_root/results/widely_targeted/        # generated output
```

## Run

```bash
pip install -r widely_targeted/requirements.txt
python widely_targeted/run_all.py \
  --data-dir /path/to/paper_supplement/widely_targeted \
  --output-dir results/widely_targeted
```

The command verifies input files against `source_manifest.tsv`, executes the
five benchmark stages, writes supporting TSV tables, and exports seven figures
as 300-dpi PNG and SVG files. An input checksum mismatch causes a non-zero exit
status.

