# 中文影评情感分析器

输入一句中文影评，输出“好评 / 差评”与置信度。项目从 `TF-IDF + Logistic Regression` 基线开始；在当前示例数据的模型对比中，朴素贝叶斯的 F1 更高，因此训练脚本现采用 `TF-IDF + Multinomial Naive Bayes`。

## 第 1 天：训练与命令行预测

### 1. 安装 Python 与依赖

请安装 Python 3.10 或更高版本，并在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

若 PowerShell 拒绝激活虚拟环境，可在当前窗口先执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### 2. 训练、评估、保存

```powershell
python src/train.py
```

训练后会生成：

- `models/sentiment_model.joblib`：可复用的模型；
- `reports/confusion_matrix.png`：混淆矩阵；
- `reports/test_predictions.csv`：测试集的每条预测，供分析错误案例。

### 3. 手动预测

```powershell
python src/predict.py "这部电影节奏拖沓，演员演得也很生硬"
python src/predict.py "画面很美，故事真挚，值得推荐"
```

## 数据格式

`data/reviews.csv` 必须有两列：`text`（评论文本）与 `label`（`1`=好评，`0`=差评）。仓库的示例数据仅供跑通流程；正式实验应替换为规模更大、来源合规且标签可靠的数据集。

## 为什么不用分词？

本版用字符级 1–2 gram TF-IDF，因此不依赖分词工具也能处理中文。第 3 天可以加入 `jieba` 分词与停用词，比较两种特征方案。

## 第 2 天：启动网页界面

确认已经训练模型后，执行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

终端会显示一个本地地址（通常是 `http://localhost:8501`）。按住 `Ctrl` 点击该地址，或在浏览器中打开它；输入影评后点击“分析情感”即可看到预测类别、置信度与正负面概率。

## 第 3 天：比较不同模型

在同一个训练集、测试集划分和 TF-IDF 特征下，比较朴素贝叶斯、逻辑回归与线性 SVM：

```powershell
.\.venv\Scripts\python.exe src\compare_models.py
```

终端会打印 Accuracy、Precision、Recall 和 F1；原始结果保存至 `reports/model_comparison.csv`，更适合阅读的报告保存至 `reports/model_comparison.md`。在 VS Code 打开 `.md` 文件后按 `Ctrl+Shift+V` 可预览表格。这一步应以 **F1（好评）** 为主排序，而不是仅看准确率。

## 第 3 天：分析错误案例

```powershell
.\.venv\Scripts\python.exe src\analyze_errors.py
```

脚本会生成 `reports/error_cases.csv` 与 `reports/error_analysis.md`。打开 Markdown 报告并按 `Ctrl+Shift+V`，查看每个错误样本的真实标签、预测标签、置信度和初步分析线索；然后再决定该补什么数据。

## 第 4 天：收集人工反馈

网页结果下方可以保存“实际标签”。反馈会写入本地 `data/user_feedback.csv`，不会自动改变原始训练数据。

当标注积累后，执行下列命令生成审核后的新数据文件：

```powershell
.\.venv\Scripts\python.exe src\prepare_feedback_data.py
```

它会生成 `data/douban_reviews_with_feedback.csv` 和 `reports/feedback_merge_report.md`。脚本会排除与原始标签冲突的反馈，原始 `douban_reviews.csv` 不会被修改。

## 第 4 天：用交叉验证选择模型

单次训练/测试切分很容易受小样本影响。用合并后的数据运行 5 折交叉验证：

```powershell
.\.venv\Scripts\python.exe src\cross_validate_models.py --data data\douban_reviews_with_feedback.csv
```

结果保存为 `reports/cross_validation.md`。选择模型时优先看“平均好评 F1”，再结合 F1 标准差判断稳定性。

## 第 5 天：导入公开豆瓣影评数据

项目提供脚本下载并校验公开的 [GT610/douban 数据集](https://huggingface.co/datasets/GT610/douban)。其数据卡标注约 4 万条中文影评、`text`/`label` 两列及 CC0-1.0 许可；请在使用前自行确认许可与适用范围。

```powershell
.\.venv\Scripts\python.exe src\download_douban_dataset.py
```

随后使用新数据训练传统模型：

```powershell
.\.venv\Scripts\python.exe src\train.py --data data\douban_reviews.csv
```

下载脚本还会生成 `reports/douban_dataset_profile.md`，记录样本数、去重情况与类别分布。

## 第 5 天：调参并保存最佳传统模型

该脚本在训练集内使用 3 折交叉验证选择字符 n-gram、最小文档频率和朴素贝叶斯 alpha；随后只用独立测试集做最终评估：

```powershell
.\.venv\Scripts\python.exe src\tune_naive_bayes.py --data data\douban_reviews.csv
```

最优模型会覆盖 `models/sentiment_model.joblib`，报告保存至 `reports/tuning_report.md`。

## 进阶：微调轻量中文 RoBERTa

`src/train_bert.py` 使用 [hfl/rbt3](https://huggingface.co/hfl/rbt3)（3 层中文 RoBERTa/BERT 架构，模型卡标注 Apache-2.0）进行二分类微调。它和传统模型依赖分开，先安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-bert.txt
```

无 GPU 时先运行 CPU 友好的 2,000 条样本、1 个 epoch 实验：

```powershell
.\.venv\Scripts\python.exe src\train_bert.py --data data\douban_reviews.csv
```

模型保存到 `models/rbt3_sentiment`，报告保存到 `reports/bert_report.md`。全量训练应优先使用 GPU，并增大 `--max-train-samples`、`--max-test-samples` 与 `--epochs`。

## 进阶：公平比较传统模型与 BERT

在 BERT 实验完成后，使用相同的 2,000 条训练样本和 1,000 条测试样本训练传统基线：

```powershell
.\.venv\Scripts\python.exe src\compare_bert_baseline.py --data data\douban_reviews.csv
```

结果保存为 `reports/bert_vs_tfidf.md`，包含效果、耗时与资源消耗的公平对比。

## 进阶：推理部署基准

完成两种模型训练后，测量相同影评批次的 CPU 推理耗时、单条延迟与模型体积：

```powershell
.\.venv\Scripts\python.exe src\benchmark_inference.py --data data\douban_reviews.csv
```

报告保存至 `reports/inference_benchmark.md`，用于决定网页或服务端应部署哪种模型。
