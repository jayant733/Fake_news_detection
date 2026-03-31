from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


DEFAULT_MIN_TEXT_LENGTH = 50


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(*parts: str) -> Path:
    return project_root().joinpath(*parts)


def resolve_data_path(data_path: str | Path | None = None) -> Path:
    if data_path is not None:
        candidate = Path(data_path)
        if not candidate.is_absolute():
            candidate = resolve_repo_path(str(candidate))
        return candidate

    candidates = (
        resolve_repo_path("data", "train.csv"),
        resolve_repo_path("DSCI632-Project", "data", "train.csv"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Could not find train.csv in the repository.")


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_training_dataframe(
    df: pd.DataFrame,
    *,
    min_text_length: int = DEFAULT_MIN_TEXT_LENGTH,
) -> pd.DataFrame:
    text_col = "text" if "text" in df.columns else "content"
    title_col = "title" if "title" in df.columns else None

    if text_col not in df.columns:
        raise KeyError("Expected a text column named 'text' or 'content'.")
    if "label" not in df.columns:
        raise KeyError("Expected a 'label' column in the dataset.")

    clean_df = df.dropna(subset=[text_col, "label"]).copy()
    clean_df[text_col] = clean_df[text_col].map(normalize_text)

    if title_col is not None:
        clean_df[title_col] = clean_df[title_col].fillna("").map(normalize_text)
        clean_df["full_text"] = (
            clean_df[title_col].str.cat(clean_df[text_col], sep=" ").map(normalize_text)
        )
    else:
        clean_df["full_text"] = clean_df[text_col]

    clean_df["label"] = pd.to_numeric(clean_df["label"], errors="coerce")
    clean_df = clean_df.dropna(subset=["full_text", "label"])
    clean_df = clean_df[clean_df["full_text"].str.len() > min_text_length]
    clean_df = clean_df[["full_text", "label"]].drop_duplicates().reset_index(drop=True)
    clean_df["label"] = clean_df["label"].astype(int)
    return clean_df


def load_training_dataframe(
    *,
    data_path: str | Path | None = None,
    sample_size: int | None = None,
    random_state: int = 42,
    min_text_length: int = DEFAULT_MIN_TEXT_LENGTH,
) -> pd.DataFrame:
    csv_path = resolve_data_path(data_path)
    df = pd.read_csv(csv_path)
    clean_df = prepare_training_dataframe(df, min_text_length=min_text_length)

    if sample_size is not None and sample_size < len(clean_df):
        clean_df = (
            clean_df.sample(sample_size, random_state=random_state)
            .reset_index(drop=True)
        )

    return clean_df
