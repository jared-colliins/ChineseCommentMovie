"""Streamlit 网页版中文影评情感分析器。"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from postgrest import ReturnMethod

from app_auth import current_user, get_supabase_client, show_auth_page, show_logged_in_user

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "sentiment_model.joblib"
FEEDBACK_PATH = ROOT / "data" / "user_feedback.csv"
UNCERTAINTY_THRESHOLD = 0.60

st.set_page_config(page_title="中文影评情感分析器", page_icon="🎬", layout="centered")

st.markdown(
    """
    <style>
      .stApp {
        background: radial-gradient(circle at 15% 0%, #252c4e 0%, #0e1117 38%, #0e1117 100%);
        color: #f8fafc;
      }
      .block-container { max-width: 920px; padding-top: 3.2rem; padding-bottom: 4rem; }
      .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label,
      .stApp [data-testid="stMarkdownContainer"],
      .stApp [data-testid="stMarkdownContainer"] *,
      .stApp [data-testid="stCaptionContainer"],
      .stApp [data-testid="stMetricLabel"],
      .stApp [data-testid="stMetricValue"],
      .stApp [data-testid="stExpander"] summary,
      .stApp [data-testid="stExpander"] summary * {
        color: #f8fafc !important;
      }
      .hero {
        padding: 2.1rem 2.2rem;
        margin: 0 0 2rem 0;
        border: 1px solid rgba(167, 139, 250, .42);
        border-radius: 22px;
        background: linear-gradient(120deg, rgba(91, 67, 175, .55), rgba(31, 41, 55, .72));
        box-shadow: 0 18px 45px rgba(0, 0, 0, .22);
      }
      .hero h1 { margin: 0; font-size: 2.55rem; letter-spacing: .02em; }
      .hero p { margin: .55rem 0 0; color: #c4c9d4 !important; font-size: 1.05rem; }
      [data-testid="stTextArea"] textarea {
        border-radius: 14px; border: 1px solid #4b5563;
        background: rgba(31, 35, 46, .92); color: #f8fafc !important;
      }
      [data-testid="stTextInput"] input {
        background: rgba(31, 35, 46, .92) !important;
        border-color: #4b5563 !important; color: #f8fafc !important;
      }
      [data-testid="stTextArea"] textarea::placeholder,
      [data-testid="stTextInput"] input::placeholder { color: #b8c0cf !important; opacity: 1; }
      [data-testid="stTextArea"] label, [data-testid="stTextInput"] label,
      [data-testid="stRadio"] label, [data-testid="stRadio"] p { color: #f8fafc !important; }
      [data-testid="stTextArea"] textarea:focus { border-color: #a78bfa; box-shadow: 0 0 0 1px #a78bfa; }
      .stButton > button {
        border: 0; border-radius: 12px; min-height: 3.1rem; font-size: 1.05rem; font-weight: 650;
        background: linear-gradient(90deg, #7c3aed, #db2777); color: white;
      }
      .stButton > button:hover { background: linear-gradient(90deg, #6d28d9, #be185d); color: white; }
      [data-testid="stFormSubmitButton"] > button {
        border: 0; border-radius: 12px; min-height: 3.1rem;
        background: linear-gradient(90deg, #7c3aed, #db2777) !important;
        color: #ffffff !important;
      }
      [data-testid="stFormSubmitButton"] > button p { color: #ffffff !important; }
      [data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(90deg, #6d28d9, #be185d) !important;
      }
      [data-testid="stMetric"] { padding: .6rem 0; }
      [data-testid="stExpander"] { border-radius: 12px; border-color: #3e4655; }
      [data-testid="stSidebar"] { background: rgba(22, 25, 35, .96); }
      @media (max-width: 640px) {
        .block-container { padding: 1.25rem 1.25rem 3rem; }
        .hero { padding: 1.45rem 1.35rem; margin-bottom: 1.5rem; }
        .hero h1 { font-size: 2rem; line-height: 1.25; }
        .hero p { font-size: 1rem; line-height: 1.6; }
        [data-testid="stTextArea"] textarea { font-size: 1rem !important; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(model_path: str, modified_at: int):
    """只在应用启动时加载一次模型。"""
    return joblib.load(model_path)


def predict_sentiment(model, text: str) -> tuple[str, float, float]:
    """返回类别、该类别置信度和好评概率。"""
    positive_probability = float(model.predict_proba([text])[0, 1])
    if positive_probability >= 0.5:
        return "好评", positive_probability, positive_probability
    return "差评", 1 - positive_probability, positive_probability


def feature_clues(model, text: str, limit: int = 6) -> tuple[list[str], list[str]]:
    """返回当前预测中最支持好评和差评的字符特征，仅作辅助解释。"""
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    matrix = vectorizer.transform([text]).tocoo()
    # 朴素贝叶斯中，两类的 log probability 差可作为每个特征的方向线索。
    direction = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
    features = vectorizer.get_feature_names_out()
    # 单字特征通常缺少独立语义，例如“一”；解释时仅保留至少两个字符的片段。
    signals = [
        (features[index], float(value * direction[index]))
        for index, value in zip(matrix.col, matrix.data)
        if len(features[index]) >= 2
    ]
    positive = [feature for feature, score in sorted(signals, key=lambda item: item[1], reverse=True) if score > 0][:limit]
    negative = [feature for feature, score in sorted(signals, key=lambda item: item[1]) if score < 0][:limit]
    return positive, negative


def save_feedback(text: str, label: int, supabase_client=None) -> bool:
    """保存人工标注到 Supabase；本地开发未配置时回退到 CSV。"""
    if supabase_client:
        try:
            supabase_client.table("review_feedback").insert(
                {"text": text, "label": label},
                returning=ReturnMethod.minimal,
            ).execute()
            return True
        except Exception as error:
            raise RuntimeError(f"云端反馈提交失败：{error}") from error

    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FEEDBACK_PATH.exists():
        with FEEDBACK_PATH.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("text") == text and row.get("label") == str(label):
                    return False

    write_header = not FEEDBACK_PATH.exists()
    with FEEDBACK_PATH.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "label", "created_at"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "text": text,
            "label": label,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })
    return True


def main() -> None:
    st.markdown(
        """
        <section class="hero">
          <h1>🎬 中文影评情感分析器</h1>
          <p>输入一句观后感，获得情感倾向、置信度与模型判断线索。</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.header("系统说明")
        st.write("**当前模型**：TF-IDF + 朴素贝叶斯")
        st.write("**训练数据**：公开豆瓣影评")
        st.write("**输出范围**：好评 / 差评 / 低置信度提醒")
        st.divider()
        st.caption("每次人工标注都会先进入独立反馈文件，审核后才会参与下一轮训练。")

    try:
        supabase_client = get_supabase_client()
    except RuntimeError as error:
        st.error(str(error))
        return
    if supabase_client:
        user = current_user(supabase_client)
        if not user:
            show_auth_page(supabase_client)
            return
        show_logged_in_user(supabase_client, user)

    if not MODEL_PATH.exists():
        st.error("尚未找到训练好的模型。请先在终端运行：python src/train.py")
        st.code(".\\.venv\\Scripts\\python.exe src\\train.py", language="powershell")
        return

    try:
        # 将修改时间作为缓存键：重新训练模型后，网页无需重启也会加载新文件。
        model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
    except Exception as error:
        st.error(f"模型加载失败：{error}")
        return

    text = st.text_area(
        "输入一条电影评价",
        placeholder="例如：这部电影节奏紧凑，演员演技自然，看得很过瘾。",
        height=140,
        max_chars=500,
    )
    analyze = st.button("分析情感", type="primary", use_container_width=True)

    if analyze:
        review = text.strip()
        if not review:
            st.warning("请先输入一条影评。")
        else:
            label, confidence, positive_probability = predict_sentiment(model, review)
            st.session_state["analysis"] = {
                "review": review,
                "label": label,
                "confidence": confidence,
                "positive_probability": positive_probability,
            }

    analysis = st.session_state.get("analysis")
    if analysis:
        review = analysis["review"]
        label = analysis["label"]
        confidence = analysis["confidence"]
        positive_probability = analysis["positive_probability"]
        if confidence < UNCERTAINTY_THRESHOLD:
            st.warning(f"模型倾向：{label}（置信度偏低，建议结合人工判断）")
        elif label == "好评":
            st.success(f"预测结果：{label}")
        else:
            st.error(f"预测结果：{label}")
        st.metric("置信度", f"{confidence:.1%}")
        st.progress(confidence)
        if confidence < UNCERTAINTY_THRESHOLD:
            st.info("当前评论的有效情感线索较少，模型只能给出弱倾向，而非可靠结论。")
        with st.expander("查看概率详情"):
            st.write(f"好评概率：{positive_probability:.1%}")
            st.write(f"差评概率：{1 - positive_probability:.1%}")
        with st.expander("查看模型判断线索"):
            positive, negative = feature_clues(model, review)
            clues = pd.DataFrame({
                "支持好评的字符线索": pd.Series(positive or ["未发现明显线索"]),
                "支持差评的字符线索": pd.Series(negative or ["未发现明显线索"]),
            })
            st.dataframe(clues, use_container_width=True, hide_index=True)
            st.caption("提示：这里显示的是 TF-IDF 字符片段（如“好看”“不值”），并非模型对完整语义的理解。")

        st.divider()
        st.subheader("帮助模型迭代")
        st.caption("如果你知道这条影评的实际情感，可以保存人工标注；它会先进入独立反馈文件，后续审核后再合并到训练数据。")
        with st.form("feedback_form", clear_on_submit=True):
            actual_label = st.radio("实际标签", ["好评", "差评"], horizontal=True)
            submitted = st.form_submit_button("保存人工标注")
        if submitted:
            try:
                saved = save_feedback(review, 1 if actual_label == "好评" else 0, supabase_client)
                if saved:
                    st.success("感谢反馈！标注已提交，审核后才会参与下一轮训练。")
                else:
                    st.info("相同的影评与标签已经保存过，无需重复添加。")
            except RuntimeError as error:
                st.error(str(error))

    st.divider()
    st.caption("提示：当前模型基于小型示例数据，结果用于学习和演示，不代表真实口碑判断。")


if __name__ == "__main__":
    main()
