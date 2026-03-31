import pandas as pd

from src.data_utils import normalize_text, prepare_training_dataframe


def test_normalize_text_collapses_extra_whitespace():
    assert normalize_text(" hello \n  world\t") == "hello world"


def test_prepare_training_dataframe_builds_full_text_and_filters_rows():
    raw = pd.DataFrame(
        {
            "title": ["Breaking", None],
            "text": [
                "This is a long enough text body for the fake news detector to keep.",
                "short",
            ],
            "label": [1, 0],
        }
    )

    cleaned = prepare_training_dataframe(raw, min_text_length=20)

    assert list(cleaned.columns) == ["full_text", "label"]
    assert cleaned.shape[0] == 1
    assert cleaned.iloc[0]["full_text"].startswith("Breaking")
    assert cleaned.iloc[0]["label"] == 1
