"""由项目所有者从 Supabase 导出人工反馈，供审核和训练使用。"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from supabase import create_client

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    url = os.environ.get("SUPABASE_URL")
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not service_key:
        raise EnvironmentError("请设置 SUPABASE_URL 与 SUPABASE_SERVICE_ROLE_KEY；不要在网页或 Git 中暴露 service role key。")

    response = create_client(url, service_key).table("review_feedback").select("text,label,created_at").execute()
    feedback = pd.DataFrame(response.data)
    if feedback.empty:
        print("云端暂无反馈，不生成训练文件。")
        return
    feedback = feedback.drop_duplicates(subset=["text", "label"])
    output = ROOT / "data" / "user_feedback.csv"
    feedback.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"已导出 {len(feedback)} 条反馈：{output}")


if __name__ == "__main__":
    main()
