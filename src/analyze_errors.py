"""找出逻辑回归模型的错误预测，辅助人工分析与迭代。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from train import load_data

ROOT = Path(__file__).resolve().parents[1]
LABEL_NAME = {0: "差评", 1: "好评"}


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
    ])


def possible_reason(text: str, confidence: float) -> str:
    """提供供人工验证的初步线索，不把它当作模型解释。"""
    if len(text) <= 8:
        return "文本很短，几乎没有可供模型判断的情感线索。"
    if confidence < 0.6:
        return "模型置信度较低，说明正负面词的证据接近。"
    return "模型较有把握但仍出错；建议检查训练集中是否缺少相似表达。"


def write_markdown(errors: pd.DataFrame, test_count: int, path: Path) -> None:
    if errors.empty:
        content = "# 错误案例分析\n\n本次测试集没有错误预测。"
    else:
        lines = [
            "# 错误案例分析",
            "",
            "## 概览",
            "",
            f"- 测试样本数：{test_count}",
            f"- 预测错误数：{len(errors)}",
            f"- 错误率：{len(errors) / test_count:.1%}",
            "",
            "## 错误样本",
            "",
            "| 影评 | 真实标签 | 预测标签 | 预测置信度 | 初步分析线索 |",
            "|---|---|---|---:|---|",
        ]
        for _, row in errors.iterrows():
            clean_text = str(row["影评"]).replace("|", "\\|")
            lines.append(
                f"| {clean_text} | {row['真实标签']} | {row['预测标签']} | "
                f"{row['预测置信度']:.1%} | {row['初步分析线索']} |"
            )
        lines.extend([
            "",
            "## 下一步怎么改进",
            "",
            "1. 为每条错误样本人工判断情感与错误原因。",
            "2. 补充同类表达及其正确标签到训练数据。",
            "3. 重新训练后，对比错误数和 F1 是否改善。",
        ])
        content = "\n".join(lines)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "reviews.csv")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    data = load_data(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.25, random_state=42, stratify=data["label"]
    )
    model = build_model().fit(x_train, y_train)
    predicted = model.predict(x_test)
    positive_probability = model.predict_proba(x_test)[:, 1]

    rows: list[dict[str, object]] = []
    for text, actual, prediction, probability in zip(x_test, y_test, predicted, positive_probability):
        if actual == prediction:
            continue
        confidence = float(probability if prediction == 1 else 1 - probability)
        rows.append({
            "影评": text,
            "真实标签": LABEL_NAME[int(actual)],
            "预测标签": LABEL_NAME[int(prediction)],
            "预测置信度": confidence,
            "初步分析线索": possible_reason(str(text), confidence),
        })

    errors = pd.DataFrame(rows)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.report_dir / "error_cases.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    write_markdown(errors, len(x_test), args.report_dir / "error_analysis.md")
    print(f"测试样本：{len(x_test)}，错误预测：{len(errors)}")
    print(f"已生成：{args.report_dir / 'error_cases.csv'}")
    print(f"已生成：{args.report_dir / 'error_analysis.md'}")


if __name__ == "__main__":
    main()
