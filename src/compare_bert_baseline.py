"""在与 BERT 相同的数据切分上评估调参后的传统朴素贝叶斯基线。"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from train import load_data

ROOT = Path(__file__).resolve().parents[1]


def stratified_limit(texts: pd.Series, labels: pd.Series, maximum: int, random_state: int) -> tuple[pd.Series, pd.Series]:
    if len(texts) <= maximum:
        return texts, labels
    chosen_texts, _, chosen_labels, _ = train_test_split(
        texts, labels, train_size=maximum, random_state=random_state, stratify=labels
    )
    return chosen_texts, chosen_labels


def extract_metric(report: str, title: str) -> float:
    matched = re.search(rf"- {re.escape(title)}：([0-9.]+)", report)
    if not matched:
        raise ValueError(f"未能从 BERT 报告中读取“{title}”。")
    return float(matched.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "douban_reviews.csv")
    parser.add_argument("--bert-report", type=Path, default=ROOT / "reports" / "bert_report.md")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "bert_vs_tfidf.md")
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=1000)
    args = parser.parse_args()

    data = load_data(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.25, random_state=42, stratify=data["label"]
    )
    x_train, y_train = stratified_limit(x_train, y_train, args.train_samples, 42)
    x_test, y_test = stratified_limit(x_test, y_test, args.test_samples, 43)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("classifier", MultinomialNB(alpha=0.3)),
    ])
    started_at = time.perf_counter()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    elapsed_seconds = time.perf_counter() - started_at
    tfidf_accuracy = accuracy_score(y_test, prediction)
    tfidf_f1 = f1_score(y_test, prediction)

    bert_report = args.bert_report.read_text(encoding="utf-8")
    bert_accuracy = extract_metric(bert_report, "Accuracy")
    bert_f1 = extract_metric(bert_report, "好评 F1")
    bert_seconds = extract_metric(bert_report, "耗时")
    report = f"""# TF-IDF 与轻量中文 RoBERTa 公平对比

## 公平性控制

- 相同数据源：`{args.data.name}`
- 相同训练样本：{len(x_train)} 条（随机种子 42，分层抽样）
- 相同测试样本：{len(x_test)} 条（随机种子 43，分层抽样）
- BERT：1 个 epoch、CPU；传统模型：TF-IDF 字符 1–2 gram + MultinomialNB(alpha=0.3)。

## 实验结果

| 模型 | Accuracy | 好评 F1 | 训练/评估耗时 | 资源特点 |
|---|---:|---:|---:|---|
| TF-IDF + 朴素贝叶斯 | {tfidf_accuracy:.4f} | {tfidf_f1:.4f} | {elapsed_seconds:.1f} 秒 | CPU 轻量、模型小、可解释性较强 |
| 轻量中文 RoBERTa | {bert_accuracy:.4f} | {bert_f1:.4f} | {bert_seconds:.1f} 秒 | 预训练语义能力、更高内存与算力需求 |

## 结论

在这次 CPU 小样本实验中，{'轻量中文 RoBERTa' if bert_f1 > tfidf_f1 else 'TF-IDF + 朴素贝叶斯'} 的好评 F1 更高。\
这不代表模型在所有规模下都更优：BERT 往往需要更多训练数据、更多 epoch 或 GPU 才能发挥优势；传统模型则非常适合快速迭代与资源受限的场景。
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"TF-IDF Accuracy：{tfidf_accuracy:.4f}，好评 F1：{tfidf_f1:.4f}，耗时：{elapsed_seconds:.1f} 秒")
    print(f"对比报告：{args.output}")


if __name__ == "__main__":
    main()
