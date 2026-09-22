"""使用轻量中文 RoBERTa 微调豆瓣影评二分类模型。"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train import load_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "hfl/rbt3"


class ReviewDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int) -> None:
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def stratified_limit(texts: pd.Series, labels: pd.Series, maximum: int, random_state: int) -> tuple[pd.Series, pd.Series]:
    if len(texts) <= maximum:
        return texts, labels
    selected_texts, _, selected_labels, _ = train_test_split(
        texts, labels, train_size=maximum, random_state=random_state, stratify=labels
    )
    return selected_texts, selected_labels


def evaluate(model, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            target = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(target.cpu().tolist())
    return accuracy_score(labels, predictions), f1_score(labels, predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "douban_reviews.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "rbt3_sentiment")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "bert_report.md")
    parser.add_argument("--max-train-samples", type=int, default=2000, help="CPU 友好上限；设为 0 表示全量")
    parser.add_argument("--max-test-samples", type=int, default=1000, help="CPU 友好上限；设为 0 表示全量")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("epochs、batch-size 和 max-length 必须大于 0。")

    torch.manual_seed(42)
    data = load_data(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.25, random_state=42, stratify=data["label"]
    )
    if args.max_train_samples > 0:
        x_train, y_train = stratified_limit(x_train, y_train, args.max_train_samples, 42)
    if args.max_test_samples > 0:
        x_test, y_test = stratified_limit(x_test, y_test, args.max_test_samples, 43)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}；训练样本：{len(x_train)}；测试样本：{len(x_test)}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    train_loader = DataLoader(
        ReviewDataset(x_train.tolist(), y_train.tolist(), tokenizer, args.max_length),
        batch_size=args.batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        ReviewDataset(x_test.tolist(), y_test.tolist(), tokenizer, args.max_length),
        batch_size=args.batch_size,
    )
    optimizer = AdamW(model.parameters(), lr=2e-5)
    started_at = time.perf_counter()
    model.train()
    for epoch in range(args.epochs):
        loss_sum = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f"Epoch {epoch + 1}/{args.epochs}，平均 loss：{loss_sum / len(train_loader):.4f}")

    accuracy, f1 = evaluate(model, test_loader, device)
    elapsed_seconds = time.perf_counter() - started_at
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    report = f"""# 轻量中文 RoBERTa 微调报告

## 实验设置

- 基础模型：[{MODEL_NAME}](https://huggingface.co/{MODEL_NAME})
- 设备：`{device}`
- 训练样本：{len(x_train)}；测试样本：{len(x_test)}
- Epoch：{args.epochs}；批大小：{args.batch_size}；最大长度：{args.max_length}
- 耗时：{elapsed_seconds:.1f} 秒

## 独立测试集结果

- Accuracy：{accuracy:.4f}
- 好评 F1：{f1:.4f}

## 注意

此脚本为 CPU 友好的快速实验，默认只抽取部分数据；与全量传统模型比较时，需确保两者使用相同训练/测试样本。模型已保存至 `{args.output_dir}`。
"""
    args.report.write_text(report, encoding="utf-8")
    print(f"测试集 Accuracy：{accuracy:.4f}，好评 F1：{f1:.4f}")
    print(f"模型已保存：{args.output_dir}")
    print(f"报告已保存：{args.report}")


if __name__ == "__main__":
    main()
