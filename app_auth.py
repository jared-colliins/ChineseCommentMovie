"""Streamlit 与 Supabase Auth 的最小集成，密钥仅从 Streamlit Secrets 读取。"""
from __future__ import annotations

import streamlit as st


def get_supabase_client():
    """未配置云端时返回 None，保留本地开发模式。"""
    try:
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_ANON_KEY")
    except Exception:
        return None
    if not url and not key:
        return None
    if not url or not key:
        raise RuntimeError("Supabase 配置不完整，请同时设置 SUPABASE_URL 与 SUPABASE_ANON_KEY。")
    try:
        from supabase import create_client
    except ModuleNotFoundError as error:
        raise RuntimeError("缺少 supabase 依赖，请执行 pip install -r requirements.txt。") from error
    return create_client(url, key)


def current_user(client):
    """恢复本次浏览器会话中的登录状态，失败时要求重新登录。"""
    session = st.session_state.get("supabase_session")
    if not session:
        return None
    try:
        client.auth.set_session(session["access_token"], session["refresh_token"])
        return client.auth.get_user().user
    except Exception:
        st.session_state.pop("supabase_session", None)
        return None


def show_auth_page(client) -> None:
    """展示登录与注册界面；成功后保存短期会话令牌而非密码。"""
    st.subheader("登录后提交人工标注")
    st.caption("分析功能无需登录。仅自愿提交人工标注时需要账户；密码不会写入本项目的文件或数据库。")
    login_tab, signup_tab = st.tabs(["登录", "注册"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("邮箱", key="login_email")
            password = st.text_input("密码", type="password", key="login_password")
            submitted = st.form_submit_button("登录", use_container_width=True)
        if submitted:
            try:
                response = client.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state["supabase_session"] = {
                    "access_token": response.session.access_token,
                    "refresh_token": response.session.refresh_token,
                }
                st.rerun()
            except Exception:
                st.error("登录失败，请检查邮箱、密码或邮箱验证状态。")
    with signup_tab:
        with st.form("signup_form"):
            email = st.text_input("邮箱", key="signup_email")
            password = st.text_input("密码", type="password", key="signup_password", help="建议至少使用 8 位并包含字母和数字。")
            submitted = st.form_submit_button("创建账号", use_container_width=True)
        if submitted:
            try:
                client.auth.sign_up({"email": email, "password": password})
                st.success("注册请求已提交。请检查邮箱并完成验证，然后返回此页面登录。")
            except Exception:
                st.error("注册失败。邮箱可能已注册，或密码不符合 Supabase 的安全规则。")


def show_logged_in_user(client, user) -> None:
    """在侧栏显示当前用户并提供退出入口。"""
    with st.sidebar:
        st.divider()
        st.caption(f"已登录：{user.email}")
        if st.button("退出登录", use_container_width=True):
            try:
                client.auth.sign_out()
            finally:
                st.session_state.pop("supabase_session", None)
            st.rerun()
