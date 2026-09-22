"""在同一个中文影评数据集上比较三种传统分类模型。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from train import load_data

ROOT = Path(__file__).resolve().parents[1]


def build_models() -> dict[str, Pipeline]:
    """构建保持相同 TF-IDF 特征的三个可比较模型。"""
    def vectorizer() -> TfidfVectorizer:
        return TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=1, sublinear_tf=True)

    return {
        "朴素贝叶斯": Pipeline([
            ("tfidf", vectorizer()),
            ("classifier", MultinomialNB()),
        ]),
        "逻辑回归": Pipeline([
            ("tfidf", vectorizer()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ]),
        "线性 SVM": Pipeline([
            ("tfidf", vectorizer()),
            ("classifier", LinearSVC(class_weight="balanced", random_state=42)),
        ]),
    }


def save_markdown_report(result: pd.DataFrame, path: Path) -> None:
    """保存适合在 VS Code 预览的简洁实验报告。"""
    best = result.iloc[0]
    headers = list(result.columns)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "|---|" + "|".join(["---:"] * (len(headers) - 1)) + "|",
    ]
    for _, row in result.iterrows():
        values = [row["模型"]] + [f"{row[column]:.4f}" for column in headers[1:]]
        table_lines.append("| " + " | ".join(values) + " |")
    table = "\n".join(table_lines)
    content = f"""# 模型对比实验报告

## 实验设置

- 数据划分：训练集 75%，测试集 25%；随机种子为 42。
- 文本特征：字符级 TF-IDF（1–2 gram）。
- 评价重点：好评类别的 F1 分数。

## 实验结果

{table}

## 当前结论

当前最优模型是 **{best['模型']}**，其好评 F1 为 **{best['F1（好评）']:.4f}**。\
但示例数据集很小，分数只用于学习阶段的模型比较，不能代表实际业务效果。
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "reviews.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "model_comparison.csv")
    args = parser.parse_args()

    data = load_data(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.25, random_state=42, stratify=data["label"]
    )

    rows: list[dict[str, float | str]] = []
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        rows.append({
            "模型": name,
            "准确率": accuracy_score(y_test, prediction),
            "Precision（好评）": precision_score(y_test, prediction, zero_division=0),
            "Recall（好评）": recall_score(y_test, prediction, zero_division=0),
            "F1（好评）": f1_score(y_test, prediction, zero_division=0),
        })

    result = pd.DataFrame(rows).sort_values("F1（好评）", ascending=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.4f")
    markdown_output = args.output.with_suffix(".md")
    save_markdown_report(result, markdown_output)
    print("模型对比结果（按好评 F1 从高到低排序）：")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\n结果已保存：{args.output}")
    print(f"可读报告已保存：{markdown_output}")


if __name__ == "__main__":
    main()
