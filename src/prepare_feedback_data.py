"""审核并合并人工反馈，生成新的训练数据文件而不覆盖原始数据。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from train import load_data

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data", type=Path, default=ROOT / "data" / "douban_reviews.csv")
    parser.add_argument("--feedback", type=Path, default=ROOT / "data" / "user_feedback.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "douban_reviews_with_feedback.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "feedback_merge_report.md")
    args = parser.parse_args()

    if not args.feedback.exists():
        raise FileNotFoundError("尚未找到人工反馈文件。请先在网页中保存至少一条人工标注。")

    base = load_data(args.base_data)
    feedback = pd.read_csv(args.feedback)
    if not {"text", "label"}.issubset(feedback.columns):
        raise ValueError("反馈文件必须包含 text 和 label 两列。")
    feedback = feedback[["text", "label"]].dropna().copy()
    feedback["text"] = feedback["text"].astype(str).str.strip()
    feedback = feedback[(feedback["text"] != "") & (feedback["label"].isin([0, 1]))]
    feedback["label"] = feedback["label"].astype(int)
    feedback = feedback.drop_duplicates(subset=["text", "label"])

    base_labels = base.set_index("text")["label"].to_dict()
    conflicts = feedback[feedback.apply(lambda row: row["text"] in base_labels and base_labels[row["text"]] != row["label"], axis=1)]
    accepted = feedback[~feedback["text"].isin(base["text"])].copy()
    merged = pd.concat([base, accepted], ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False, encoding="utf-8-sig")
    report = f"""# 人工反馈合并报告

- 原始训练样本：{len(base)}
- 收到的有效人工标注：{len(feedback)}
- 接受并新增的标注：{len(accepted)}
- 与原始数据标签冲突的标注：{len(conflicts)}（未合并，需要人工复核）
- 新训练数据样本总数：{len(merged)}

新文件：`{args.output.name}`。请用它重新训练：

```powershell
.\\.venv\\Scripts\\python.exe src\\train.py --data data\\{args.output.name}
```
"""
    args.report.write_text(report, encoding="utf-8")
    print(f"已生成训练数据：{args.output}")
    print(f"已生成合并报告：{args.report}")


if __name__ == "__main__":
    main()
