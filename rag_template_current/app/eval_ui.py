# 評価用質問CSVを使ってRAG APIの回答品質を確認するStreamlitアプリです。
from pathlib import Path
import datetime as dt
import json
import pandas as pd
import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="RAG評価UI", layout="wide")
st.title("RAG評価UI")

api_url = st.sidebar.text_input("API URL", value="http://127.0.0.1:8000/ask")
uploaded = st.file_uploader("評価用CSVをアップロード", type=["csv"])

st.markdown("""
評価用CSVには最低限 `question_id`, `question`, `expected_answer` を含めてください。
任意で `corpus_ids`, `expected_source` を指定できます。
""")

if uploaded:
    df = pd.read_csv(uploaded)
    if df.empty:
        st.warning("評価用CSVに行がありません。")
        st.stop()
    st.dataframe(df, use_container_width=True)

    idx = st.number_input("評価する行番号", min_value=0, max_value=len(df)-1, value=0)
    row = df.iloc[int(idx)]

    st.subheader("質問")
    st.write(row["question"])

    st.subheader("期待回答")
    st.write(row.get("expected_answer", ""))

    corpus_ids = []
    if "corpus_ids" in row and pd.notna(row["corpus_ids"]):
        corpus_ids = [x.strip() for x in str(row["corpus_ids"]).split("|") if x.strip()]

    if st.button("この質問を実行", type="primary"):
        payload = {
            "question": str(row["question"]),
            "corpus_ids": corpus_ids or None,
            "show_debug": True,
        }
        with st.spinner("実行中..."):
            try:
                res = requests.post(api_url, json=payload, timeout=300)
                res.raise_for_status()
                st.session_state["last_result"] = res.json()
            except Exception as e:
                st.error(f"API呼び出しに失敗しました: {e}")
                st.stop()

    result = st.session_state.get("last_result")
    if result:
        st.subheader("回答")
        st.markdown(result["answer"])

        st.subheader("根拠")
        for i, s in enumerate(result.get("sources", []), start=1):
            with st.expander(f'根拠{i}: {s.get("heading_path")} / {s.get("score", 0):.3f}'):
                st.caption(f'{s.get("corpus_id")} / {s.get("source_file")}')
                st.write(s.get("child_text", ""))
                forms = s.get("forms") or []
                if forms:
                    st.markdown("#### 関連様式")
                    for form in forms:
                        exists_label = "あり" if form.get("exists") else "未配置"
                        st.write(f'{form.get("form_name")} / {form.get("file_path")} / ファイル: {exists_label}')

        st.subheader("人手評価")
        score = st.radio("評価", ["○", "△", "×"], horizontal=True)
        comment = st.text_area("コメント")

        if st.button("評価ログ保存"):
            log = {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "question_id": row.get("question_id", ""),
                "question": row.get("question", ""),
                "expected_answer": row.get("expected_answer", ""),
                "answer": result.get("answer", ""),
                "corpus_ids": corpus_ids,
                "score": score,
                "comment": comment,
                "sources": result.get("sources", []),
            }
            out = LOG_DIR / "eval_logs.jsonl"
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log, ensure_ascii=False) + "\n")
            st.success(f"保存しました: {out}")
