"""训练并评估中文影评二分类模型。"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from matplotlib import rcParams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]

# 避免 Windows 上导出的中文混淆矩阵出现方框或缺字。
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required_columns = {"text", "label"}
    if not required_columns.issubset(data.columns):
        raise ValueError("数据文件必须包含 text 和 label 两列。")
    data = data.dropna(subset=["text", "label"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]
    if not set(data["label"].unique()).issubset({0, 1}):
        raise ValueError("label 只能取 0（差评）或 1（好评）。")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "reviews.csv")
    parser.add_argument("--model-out", type=Path, default=ROOT / "models" / "sentiment_model.joblib")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    data = load_data(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=args.test_size,
        random_state=42, stratify=data["label"]
    )
    # char n-gram 对中文无需先分词；当前由对比实验选出朴素贝叶斯作为最佳模型。
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("classifier", MultinomialNB()),
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)

    report = classification_report(y_test, predictions, target_names=["差评", "好评"], digits=4)
    print(f"训练样本：{len(x_train)}，测试样本：{len(x_test)}")
    print(f"F1-score：{f1_score(y_test, predictions):.4f}")
    print("\n分类报告：\n" + report)
    pd.DataFrame({"text": x_test, "actual": y_test, "predicted": predictions, "positive_probability": probabilities}).to_csv(
        args.report_dir / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    display = ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=["差评", "好评"])
    display.figure_.savefig(args.report_dir / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    print(f"\n模型已保存：{args.model_out}")
    print(f"评估产物：{args.report_dir}")


if __name__ == "__main__":
    main()
