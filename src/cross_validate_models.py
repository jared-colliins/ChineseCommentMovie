"""用分层 K 折交叉验证更稳定地比较传统文本分类模型。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from compare_models import build_models
from train import load_data

ROOT = Path(__file__).resolve().parents[1]


def write_report(result: pd.DataFrame, folds: int, path: Path) -> None:
    headers = list(result.columns)
    lines = [
        "# 交叉验证模型选择报告",
        "",
        "## 为什么使用交叉验证？",
        "",
        f"同一模型会进行 {folds} 次不同的训练/验证划分，最终看平均 F1 与波动范围，避免一次随机切分恰好偏高或偏低。",
        "",
        "## 结果",
        "",
        "| " + " | ".join(headers) + " |",
        "|---|" + "|".join(["---:"] * (len(headers) - 1)) + "|",
    ]
    for _, row in result.iterrows():
        lines.append(
            f"| {row['模型']} | {row['平均准确率']:.4f} | {row['平均好评 F1']:.4f} | {row['F1 标准差']:.4f} |"
        )
    best = result.iloc[0]
    lines.extend([
        "",
        "## 当前建议",
        "",
        f"以平均好评 F1 为准，当前建议继续评估 **{best['模型']}**（平均 F1：**{best['平均好评 F1']:.4f}**）。"
        "样本量较小时，标准差越小代表结果越稳定；最终切换网页模型前仍建议结合可解释性与更多真实数据判断。",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "reviews.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "cross_validation.md")
    args = parser.parse_args()

    data = load_data(args.data)
    if args.folds > data["label"].value_counts().min():
        raise ValueError("折数不能超过任一类别的样本数。")
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    rows: list[dict[str, float | str]] = []
    for name, model in build_models().items():
        scores = cross_validate(
            model, data["text"], data["label"], cv=splitter,
            scoring={"accuracy": "accuracy", "f1": "f1"}, n_jobs=1,
        )
        rows.append({
            "模型": name,
            "平均准确率": scores["test_accuracy"].mean(),
            "平均好评 F1": scores["test_f1"].mean(),
            "F1 标准差": scores["test_f1"].std(),
        })

    result = pd.DataFrame(rows).sort_values("平均好评 F1", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(result, args.folds, args.output)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\n交叉验证报告：{args.output}")


if __name__ == "__main__":
    main()
