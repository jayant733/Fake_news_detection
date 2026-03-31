from __future__ import annotations

import argparse
import json


def train_classical(args: argparse.Namespace) -> dict[str, float]:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split

    from src.data_utils import load_training_dataframe, resolve_repo_path

    df = load_training_dataframe(
        data_path=args.data_path,
        sample_size=args.sample_size,
        random_state=args.random_state,
        min_text_length=args.min_text_length,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df["full_text"],
        df["label"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=args.max_features,
        ngram_range=(1, 1),
        min_df=args.min_df,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=args.max_iter,
        class_weight="balanced",
        random_state=args.random_state,
    )
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }

    vectorizer_path = resolve_repo_path(args.vectorizer_out)
    model_path = resolve_repo_path(args.model_out)
    metrics_path = resolve_repo_path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train fake news detection models.")
    subparsers = parser.add_subparsers(dest="model", required=False)

    classical = subparsers.add_parser("classical", help="Train the TF-IDF + Logistic Regression model.")
    classical.add_argument("--data-path", default=None)
    classical.add_argument("--sample-size", type=int, default=None)
    classical.add_argument("--min-text-length", type=int, default=50)
    classical.add_argument("--test-size", type=float, default=0.2)
    classical.add_argument("--random-state", type=int, default=42)
    classical.add_argument("--max-features", type=int, default=5000)
    classical.add_argument("--min-df", type=int, default=5)
    classical.add_argument("--max-iter", type=int, default=2000)
    classical.add_argument("--vectorizer-out", default="vectorizer.pkl")
    classical.add_argument("--model-out", default="fake_news_model.pkl")
    classical.add_argument("--metrics-out", default="artifacts/classical_metrics.json")

    bert = subparsers.add_parser("bert", help="Train a BERT classifier without notebook-only dependencies.")
    bert.add_argument("--data-path", default=None)
    bert.add_argument("--sample-size", type=int, default=4000)
    bert.add_argument("--min-text-length", type=int, default=50)
    bert.add_argument("--test-size", type=float, default=0.2)
    bert.add_argument("--random-state", type=int, default=42)
    bert.add_argument("--model-name", default="bert-base-uncased")
    bert.add_argument("--output-dir", default="artifacts/bert")
    bert.add_argument("--max-length", type=int, default=256)
    bert.add_argument("--epochs", type=int, default=1)
    bert.add_argument("--train-batch-size", type=int, default=8)
    bert.add_argument("--eval-batch-size", type=int, default=16)
    bert.add_argument("--learning-rate", type=float, default=2e-5)
    bert.add_argument("--weight-decay", type=float, default=0.01)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.model in (None, "classical"):
        metrics = train_classical(args)
    else:
        from src.models.bert import BertTrainingConfig, train_bert

        config = BertTrainingConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            data_path=args.data_path,
            sample_size=args.sample_size,
            min_text_length=args.min_text_length,
            test_size=args.test_size,
            max_length=args.max_length,
            epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            random_state=args.random_state,
        )
        metrics = train_bert(config)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
