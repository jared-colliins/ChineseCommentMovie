"""下载并校验公开的豆瓣中文影评情感数据集。"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET_URL = "https://huggingface.co/datasets/GT610/douban/resolve/main/douban_reviews.parquet?download=true"
DATASET_PAGE = "https://huggingface.co/datasets/GT610/douban"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "douban_reviews.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "douban_dataset_profile.md")
    parser.add_argument("--max-rows", type=int, default=None, help="仅用于快速实验；默认保留全量数据")
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        print("正在下载公开豆瓣影评数据集……")
        urlretrieve(DATASET_URL, temporary_path)
        data = pd.read_parquet(temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("下载的数据缺少 text 或 label 列，已停止导入。")
    data = data[["text", "label"]].dropna().copy()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[(data["text"] != "") & (data["label"].isin([0, 1]))]
    data["label"] = data["label"].astype(int)
    before_deduplication = len(data)
    data = data.drop_duplicates(subset=["text", "label"])
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("--max-rows 必须大于 0。")
        data = data.sample(n=min(args.max_rows, len(data)), random_state=42)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False, encoding="utf-8-sig")
    distribution = data["label"].value_counts().to_dict()
    report = f"""# 公开豆瓣影评数据集质量报告

## 来源

- 数据集：[GT610/douban]({DATASET_PAGE})
- 任务：中文电影评论二分类情感分析。
- 数据卡声明：CC0-1.0；使用前请自行确认其当前许可和适用范围。

## 导入结果

- 有效且去重后的样本数：{len(data)}
- 去重前有效样本数：{before_deduplication}
- 差评（0）：{distribution.get(0, 0)}
- 好评（1）：{distribution.get(1, 0)}
- 输出文件：`{args.output.name}`

## 使用建议

该文件与示例数据分开保存。训练时显式传入 `--data data/{args.output.name}`；不要把个人反馈直接混入公开数据，除非已完成标签审核与数据版本记录。
"""
    args.report.write_text(report, encoding="utf-8")
    print(f"数据已保存：{args.output}")
    print(f"质量报告已保存：{args.report}")


if __name__ == "__main__":
    main()
