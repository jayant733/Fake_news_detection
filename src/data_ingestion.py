from __future__ import annotations

from pathlib import Path

from src.data_utils import resolve_data_path


def load_data(spark, path: str | Path | None = None):
    csv_path = resolve_data_path(path)
    return spark.read.csv(str(csv_path), header=True, inferSchema=True)
