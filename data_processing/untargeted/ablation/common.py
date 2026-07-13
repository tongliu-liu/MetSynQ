"""Shared helpers for the final AB error-summary calculations."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


FEATURE_COLUMN_CANDIDATES = ["feature_id", "Compound Name", "mw ID", "Feature number"]


def read_table(path: Path) -> pd.DataFrame:
    """Read a supported tabular input without changing missing values."""
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    """Write a deterministic, machine-readable TSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, na_rep="NA", lineterminator="\n")


def find_feature_column(df: pd.DataFrame) -> str:
    for column in FEATURE_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(
        "No feature ID column found. Tried: " + ", ".join(FEATURE_COLUMN_CANDIDATES)
    )


def clean_feature_ids(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(value))
    )


def sorted_feature_ids(values: set[str]) -> list[str]:
    return sorted(values, key=natural_key)


def sample_columns(df: pd.DataFrame, feature_column: str) -> list[str]:
    columns = [column for column in df.columns if column != feature_column]
    if len(columns) < 2 or len(columns) % 2 != 0:
        raise ValueError(
            f"Expected an even number of sample columns after '{feature_column}', "
            f"found {len(columns)}: "
            + ", ".join(map(str, columns))
        )
    return columns


def read_feature_set(path: Path) -> set[str]:
    df = read_table(path)
    feature_column = find_feature_column(df)
    values = clean_feature_ids(df[feature_column]).dropna()
    values = values[values != ""]
    result = set(values.astype(str))
    if not result:
        raise ValueError(f"No feature IDs found in {path}")
    return result
