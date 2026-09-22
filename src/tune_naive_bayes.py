"""在独立测试集评估前提下，为 TF-IDF + 朴素贝叶斯选择参数。"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from train import load_data

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "douban_reviews.csv")
    parser.add_argument("--model-out", type=Path, default=ROOT / "models" / "sentiment_model.joblib")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "tuning_report.md")
    args = parser.parse_args()

    data = load_data(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.25, random_state=42, stratify=data["label"]
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", sublinear_tf=True)),
        ("classifier", MultinomialNB()),
    ])
    search = GridSearchCV(
        pipeline,
        param_grid={
            "tfidf__ngram_range": [(1, 2), (2, 3)],
            "tfidf__min_df": [1, 2],
            "classifier__alpha": [0.3, 1.0],
        },
        scoring="f1",
        cv=3,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(x_train, y_train)
    predictions = search.predict(x_test)
    test_f1 = f1_score(y_test, predictions)
    test_accuracy = accuracy_score(y_test, predictions)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, args.model_out)
    parameters = search.best_params_
    report = f"""# 朴素贝叶斯调参报告

## 实验设计

- 数据：`{args.data.name}`
- 外层划分：75% 训练集、25% 独立测试集，随机种子 42。
- 参数选择：仅在训练集上进行 3 折交叉验证，以好评 F1 为评分标准。
- 参数组合数：{len(search.cv_results_['params'])}。

## 最优参数

- 字符 n-gram：`{parameters['tfidf__ngram_range']}`
- 最小文档频率：`{parameters['tfidf__min_df']}`
- 平滑参数 alpha：`{parameters['classifier__alpha']}`
- 训练集交叉验证最佳 F1：{search.best_score_:.4f}

## 独立测试集结果

- Accuracy：{test_accuracy:.4f}
- 好评 F1：{test_f1:.4f}

```
{classification_report(y_test, predictions, target_names=['差评', '好评'], digits=4)}
```

最优模型已保存为 `{args.model_out.name}`，网页将自动加载它。
"""
    args.report.write_text(report, encoding="utf-8")
    print(f"最优参数：{parameters}")
    print(f"独立测试集 Accuracy：{test_accuracy:.4f}，好评 F1：{test_f1:.4f}")
    print(f"报告已保存：{args.report}")


if __name__ == "__main__":
    main()
