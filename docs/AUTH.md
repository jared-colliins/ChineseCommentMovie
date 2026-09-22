# 注册登录配置

本项目在未配置 Supabase Secrets 时保持本地开发模式；部署环境配置了 `SUPABASE_URL` 与 `SUPABASE_ANON_KEY` 后，访客必须注册并登录才能进入情感分析器。

## Supabase 设置

1. 在 Supabase Dashboard 的 **Authentication → Providers → Email** 启用 Email 登录。
2. 建议保持邮箱确认开启，以减少滥用账号。
3. 在 **Authentication → URL Configuration** 中添加 Streamlit 云端应用的 URL 作为 Redirect URL。
4. 在 SQL Editor 执行 [`../database/schema.sql`](../database/schema.sql)。其中的 RLS 策略只允许 `authenticated` 用户插入自己的反馈，不能读取反馈表。

## Streamlit Cloud Secrets

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_ANON_KEY = "your-anon-key"
```

不要放入 `SUPABASE_SERVICE_ROLE_KEY`。它只能由项目所有者在本地导出和审核反馈时使用。
