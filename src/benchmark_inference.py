"""比较传统模型和轻量 RoBERTa 在 CPU 上的批量推理速度与模型体积。"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train import load_data

ROOT = Path(__file__).resolve().parents[1]


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "douban_reviews.csv")
    parser.add_argument("--tfidf-model", type=Path, default=ROOT / "models" / "sentiment_model.joblib")
    parser.add_argument("--bert-model", type=Path, default=ROOT / "models" / "rbt3_sentiment")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "inference_benchmark.md")
    args = parser.parse_args()
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples 和 batch-size 必须大于 0。")
    if not args.tfidf_model.exists() or not args.bert_model.exists():
        raise FileNotFoundError("未找到传统模型或 BERT 模型，请先完成两种模型的训练。")

    texts = load_data(args.data)["text"].head(args.samples).tolist()
    tfidf = joblib.load(args.tfidf_model)
    started_at = time.perf_counter()
    tfidf.predict_proba(texts)
    tfidf_seconds = time.perf_counter() - started_at

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    bert = AutoModelForSequenceClassification.from_pretrained(args.bert_model).to(device).eval()
    # 预热一次，避免首次算子初始化影响计时。
    with torch.no_grad():
        warmup = tokenizer(texts[: min(len(texts), args.batch_size)], return_tensors="pt", padding=True, truncation=True, max_length=128)
        bert(**{key: value.to(device) for key, value in warmup.items()})
    started_at = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch = tokenizer(texts[start:start + args.batch_size], return_tensors="pt", padding=True, truncation=True, max_length=128)
            bert(**{key: value.to(device) for key, value in batch.items()})
    bert_seconds = time.perf_counter() - started_at

    tfidf_size_mb = args.tfidf_model.stat().st_size / 1024 / 1024
    bert_size_mb = directory_size(args.bert_model) / 1024 / 1024
    report = f"""# 推理部署基准测试

## 测试设置

- 文本数：{len(texts)} 条；批大小：{args.batch_size}。
- 设备：`{device}`。
- BERT 已进行一次预热，计时不包含加载模型时间。

## 结果

| 模型 | 批量耗时 | 平均每条耗时 | 磁盘体积 |
|---|---:|---:|---:|
| TF-IDF + 朴素贝叶斯 | {tfidf_seconds:.4f} 秒 | {tfidf_seconds / len(texts) * 1000:.2f} ms | {tfidf_size_mb:.2f} MB |
| 轻量中文 RoBERTa | {bert_seconds:.4f} 秒 | {bert_seconds / len(texts) * 1000:.2f} ms | {bert_size_mb:.2f} MB |

## 部署建议

当前 CPU 本地网页优先使用 TF-IDF + 朴素贝叶斯：启动快、推理快、模型小且便于解释。\
RoBERTa 适合具备 GPU 或可接受更高延迟与资源开销的服务场景。
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"传统模型：{tfidf_seconds:.4f} 秒；BERT：{bert_seconds:.4f} 秒")
    print(f"报告：{args.report}")


if __name__ == "__main__":
    main()
