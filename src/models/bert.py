from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data_utils import load_training_dataframe, resolve_repo_path


@dataclass
class BertTrainingConfig:
    model_name: str = "bert-base-uncased"
    output_dir: str = "artifacts/bert"
    data_path: str | None = None
    sample_size: int = 4000
    min_text_length: int = 50
    test_size: float = 0.2
    max_length: int = 256
    epochs: int = 1
    train_batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    random_state: int = 42


class NewsTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    losses: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            losses.append(outputs.loss.item())
            predictions.extend(outputs.logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())

    correct = sum(int(pred == label) for pred, label in zip(predictions, labels))
    true_positive = sum(int(pred == 1 and label == 1) for pred, label in zip(predictions, labels))
    false_positive = sum(int(pred == 1 and label == 0) for pred, label in zip(predictions, labels))
    false_negative = sum(int(pred == 0 and label == 1) for pred, label in zip(predictions, labels))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": correct / len(labels) if labels else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def split_dataframe(df, test_size: float, random_state: int):
    indices = np.arange(len(df))
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)

    split_index = max(1, int(len(df) * (1 - test_size)))
    split_index = min(split_index, len(df) - 1)

    train_indices = indices[:split_index]
    eval_indices = indices[split_index:]
    return df.iloc[train_indices].reset_index(drop=True), df.iloc[eval_indices].reset_index(drop=True)


def train_bert(config: BertTrainingConfig) -> dict[str, object]:
    set_seed(config.random_state)

    df = load_training_dataframe(
        data_path=config.data_path,
        sample_size=config.sample_size,
        random_state=config.random_state,
        min_text_length=config.min_text_length,
    )

    train_df, eval_df = split_dataframe(df, config.test_size, config.random_state)

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=2,
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to load the BERT tokenizer/model. Pass a local model path with "
            "--model-name or allow a one-time download of the pretrained weights."
        ) from exc

    train_dataset = NewsTextDataset(
        train_df["full_text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
        config.max_length,
    )
    eval_dataset = NewsTextDataset(
        eval_df["full_text"].tolist(),
        eval_df["label"].tolist(),
        tokenizer,
        config.max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.eval_batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        train_losses: list[float] = []

        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            train_losses.append(outputs.loss.item())

        metrics = evaluate(model, eval_loader, device)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = float(np.mean(train_losses)) if train_losses else 0.0
        history.append(metrics)

    output_dir = resolve_repo_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    report = {
        "config": asdict(config),
        "train_rows": len(train_df),
        "eval_rows": len(eval_df),
        "history": history,
        "final_metrics": history[-1] if history else {},
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
