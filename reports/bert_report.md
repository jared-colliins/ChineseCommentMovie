# 轻量中文 RoBERTa 微调报告

## 实验设置

- 基础模型：[hfl/rbt3](https://huggingface.co/hfl/rbt3)
- 设备：`cpu`
- 训练样本：2000；测试样本：1000
- Epoch：1；批大小：8；最大长度：128
- 耗时：191.3 秒

## 独立测试集结果

- Accuracy：0.7180
- 好评 F1：0.7294

## 注意

此脚本为 CPU 友好的快速实验，默认只抽取部分数据；与全量传统模型比较时，需确保两者使用相同训练/测试样本。模型已保存至 `D:\_vscode\_ChineseCommentMovie\models\rbt3_sentiment`。
