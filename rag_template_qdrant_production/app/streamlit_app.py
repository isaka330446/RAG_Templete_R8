# 利用者が会話形式でRAGへ質問し、根拠と回答ログを確認するStreamlitアプリです。
from pathlib import Path
import json
import uuid

import requests
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_SETTINGS = BASE_DIR / "config" / "corpus_settings.json"


def api_base_url(ask_url: str) -> str:
    base = ask_url.rstrip("/")
    if base.endswith("/ask"):
        base = base[: -len("/ask")]
    return base


def source_display_title(source: dict) -> str:
    return str(
        source.get("heading_path")
        or source.get("title")
        or source.get("source_file")
        or source.get("child_id")
        or "根拠"
    )


def short_text(value: str, limit: int = 42) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def source_button_key(key_prefix: str, index: int, source: dict) -> str:
    raw = str(source.get("child_id") or source.get("parent_id") or index)
    safe = "".join(ch if ch.isalnum() else "_" for ch in raw)[:80]
    return f"{key_prefix}_source_{index}_{safe}"


def render_source_dialog(source: dict, index: int) -> None:
    @st.dialog("根拠詳細")
    def _dialog():
        score = source.get("score")
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "-"
        st.markdown(f"### 根拠{index}: {source_display_title(source)}")
        st.caption(f"score: {score_text}")

        metadata = [
            ("corpus_id", source.get("corpus_id")),
            ("source_file", source.get("source_file")),
            ("heading_path", source.get("heading_path")),
            ("parent_id", source.get("parent_id")),
            ("child_id", source.get("child_id")),
            ("document_type", source.get("document_type")),
            ("source_type", source.get("source_type")),
            ("meeting_id", source.get("meeting_id")),
            ("meeting_name", source.get("meeting_name")),
            ("meeting_date", source.get("meeting_date")),
            ("slide_no", source.get("slide_no")),
            ("slide_title", source.get("slide_title")),
            ("agenda", source.get("agenda")),
            ("topic", source.get("topic")),
            ("section_title", source.get("section_title")),
            ("content_type", source.get("content_type")),
            ("auxiliary_reason", source.get("auxiliary_reason")),
        ]
        for label, value in metadata:
            if value not in (None, ""):
                st.caption(f"{label}: {value}")

        st.markdown("#### ヒットした子チャンク")
        st.markdown(source.get("child_text") or source.get("text") or "")

        parent = source.get("parent_text")
        if parent:
            with st.expander("回答生成に使った親チャンク全文", expanded=False):
                st.markdown(parent)

        tags = source.get("search_tags") or []
        if tags:
            st.markdown("#### SearchTag")
            st.write(" / ".join(str(tag) for tag in tags))

        forms = source.get("forms") or []
        if forms:
            st.markdown("#### 関連様式")
            for form in forms:
                exists_label = "あり" if form.get("exists") else "未配置"
                st.write(f'{form.get("form_name")} / {form.get("file_path")} / ファイル: {exists_label}')

    _dialog()


def render_sources(sources: list[dict], key_prefix: str) -> None:
    if not sources:
        return
    st.caption(f"根拠: {len(sources)}件")
    column_count = min(4, max(1, len(sources)))
    for start in range(0, len(sources), column_count):
        cols = st.columns(column_count)
        for offset, source in enumerate(sources[start : start + column_count]):
            index = start + offset + 1
            title = source_display_title(source)
            score = source.get("score")
            score_text = f" / score={score:.3f}" if isinstance(score, (int, float)) else ""
            with cols[offset]:
                if st.button(
                    f"根拠{index}",
                    key=source_button_key(key_prefix, index, source),
                    help=f"{title}{score_text}",
                ):
                    render_source_dialog(source, index)
                st.caption(short_text(f"{title}{score_text}"))


def render_answer_source(message: dict) -> None:
    if message.get("cache_hit"):
        similarity = message.get("cache_similarity")
        sim_text = f" / similarity={similarity:.3f}" if isinstance(similarity, (int, float)) else ""
        st.caption(f'回答元: 承認済みQAキャッシュ / qa_id={message.get("qa_cache_id")}{sim_text}')
    elif message.get("answer_source"):
        st.caption(f'回答元: {message.get("answer_source")}')


def render_hallucination_report(message: dict, api_url: str, idx: int) -> None:
    log_id = message.get("log_id")
    if not log_id:
        return

    if log_id in st.session_state.reported_log_ids:
        st.caption("この回答は報告済みです。")
        return

    with st.expander("ハルシネーション疑いを報告", expanded=False):
        comment = st.text_area("コメント", key=f"report_comment_{log_id}_{idx}", height=90)
        if st.button("報告する", key=f"report_button_{log_id}_{idx}"):
            payload = {
                "question": message.get("question") or "",
                "answer": message.get("content") or "",
                "session_id": message.get("session_id") or st.session_state.session_id,
                "log_id": log_id,
                "comment": comment,
            }
            try:
                res = requests.post(f"{api_base_url(api_url)}/feedback/hallucination", json=payload, timeout=30)
                res.raise_for_status()
                report_id = res.json().get("report_id")
            except Exception as exc:
                st.error(f"報告に失敗しました: {exc}")
                return
            st.session_state.reported_log_ids.add(log_id)
            st.success(f"報告を受け付けました。report_id={report_id}")


st.set_page_config(page_title="RAGチャット", layout="wide")
st.title("RAGチャット")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "reported_log_ids" not in st.session_state:
    st.session_state.reported_log_ids = set()

api_url = st.sidebar.text_input("API URL", value="http://127.0.0.1:8000/ask")
st.sidebar.caption(f"session_id: {st.session_state.session_id}")

if st.sidebar.button("新しい会話"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.reported_log_ids = set()
    st.rerun()

settings = json.loads(CORPUS_SETTINGS.read_text(encoding="utf-8"))
corpora = [c for c in settings.get("corpora", []) if c.get("enabled", True)]

st.sidebar.subheader("検索対象文書")
selected = []
for c in sorted(corpora, key=lambda x: x.get("priority", 999)):
    checked = st.sidebar.checkbox(c["display_name"], value=True, key=c["corpus_id"])
    if checked:
        selected.append(c["corpus_id"])

top_k = st.sidebar.slider("根拠数", min_value=3, max_value=15, value=8)

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_answer_source(message)
            render_sources(message.get("sources", []), key_prefix=f"history_{idx}")
            if message.get("log_id"):
                st.caption(f'log_id: {message["log_id"]}')
            render_hallucination_report(message, api_url, idx)

question = st.chat_input("質問を入力してください")
if question:
    if not selected:
        st.warning("検索対象文書を1つ以上選択してください。")
        st.stop()

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ]
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    payload = {
        "question": question,
        "corpus_ids": selected,
        "top_k": top_k,
        "show_debug": False,
        "session_id": st.session_state.session_id,
        "history": history,
    }

    with st.chat_message("assistant"):
        with st.spinner("検索・回答生成中..."):
            try:
                res = requests.post(api_url, json=payload, timeout=300)
                res.raise_for_status()
                data = res.json()
            except Exception as exc:
                st.error(f"API呼び出しに失敗しました: {exc}")
                st.stop()
        st.markdown(data["answer"])
        render_answer_source(data)
        render_sources(data.get("sources", []), key_prefix=f"current_{len(st.session_state.messages)}")
        if data.get("log_id"):
            st.caption(f'log_id: {data["log_id"]}')

    st.session_state.session_id = data.get("session_id") or st.session_state.session_id
    st.session_state.messages.append({
        "role": "assistant",
        "content": data["answer"],
        "sources": data.get("sources", []),
        "log_id": data.get("log_id"),
        "session_id": st.session_state.session_id,
        "question": question,
        "answer_source": data.get("answer_source"),
        "cache_hit": data.get("cache_hit", False),
        "qa_cache_id": data.get("qa_cache_id"),
        "cache_similarity": data.get("cache_similarity"),
    })
