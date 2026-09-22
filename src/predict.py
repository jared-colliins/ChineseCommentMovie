"""在命令行预测一条中文影评。"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="要分析的影评")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "sentiment_model.joblib")
    args = parser.parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"未找到模型：{args.model}。请先运行 python src/train.py")

    model = joblib.load(args.model)
    positive_probability = float(model.predict_proba([args.text])[0, 1])
    label = "好评" if positive_probability >= 0.5 else "差评"
    confidence = positive_probability if label == "好评" else 1 - positive_probability
    print(f"输入：{args.text}")
    print(f"输出：{label}（{confidence:.1%}）")


if __name__ == "__main__":
    main()
