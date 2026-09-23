# 人工反馈合并报告

- 原始训练样本：39997
- 收到的有效人工标注：6
- 接受并新增的标注：6
- 与原始数据标签冲突的标注：0（未合并，需要人工复核）
- 新训练数据样本总数：40003

新文件：`douban_reviews_with_feedback.csv`。请用它重新训练：

```powershell
.\.venv\Scripts\python.exe src\train.py --data data\douban_reviews_with_feedback.csv
```
