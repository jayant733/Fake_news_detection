from pathlib import Path

from src.data_utils import project_root, resolve_data_path


def test_resolve_data_path_finds_repo_dataset():
    data_path = resolve_data_path()
    assert data_path.exists()
    assert data_path.name == "train.csv"


def test_project_root_is_repo_root():
    root = project_root()
    assert root == Path(__file__).resolve().parents[1]
