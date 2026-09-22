# 云端部署与人工反馈

## 架构

```text
外部用户 → Streamlit Community Cloud 网页 → Supabase review_feedback 表
项目所有者 → Supabase Dashboard / 导出脚本 → 人工审核 → 重新训练
```

## 1. 创建 Supabase 表

在 Supabase 新建项目，进入 **SQL Editor**，执行 [`../database/schema.sql`](../database/schema.sql)。该策略只允许已登录用户提交自己的反馈，不能读取已有反馈。

## 2. 配置 Streamlit Cloud 密钥

从 Supabase 项目设置取得 **Project URL** 与 **anon public key**。在 Streamlit Cloud 的 **App settings → Secrets** 填入：

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_ANON_KEY = "your-anon-key"
```

不要填写 `service_role` key；它只能由项目所有者在本地导出数据时使用。

## 3. 推送并部署

将项目推送到 GitHub，再在 Streamlit Community Cloud 创建应用并指定入口文件 `app.py`。

部署仓库应包含：

- `app.py`、`src/`、`requirements.txt`；
- `models/sentiment_model.joblib`，供云端网页加载预测模型；
- `database/schema.sql`，便于复现数据库结构。

不要提交：

- `.streamlit/secrets.toml`；
- `data/user_feedback.csv`；
- `models/rbt3_sentiment/`；
- `SUPABASE_SERVICE_ROLE_KEY`。

## 4. 审核与重新训练

在本地安全地设置环境变量后导出云端反馈：

```powershell
$env:SUPABASE_URL = "https://your-project.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY = "your-service-role-key"
.\.venv\Scripts\python.exe src\export_supabase_feedback.py
```

再运行：

```powershell
.\.venv\Scripts\python.exe src\prepare_feedback_data.py
.\.venv\Scripts\python.exe src\tune_naive_bayes.py --data data\douban_reviews_with_feedback.csv
```

重新训练后，将新的 `models/sentiment_model.joblib` 提交并重新部署。先人工审核反馈；公开匿名表单可能收到无关或恶意输入。
