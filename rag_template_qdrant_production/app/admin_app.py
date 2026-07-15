# 管理者がハルシネーション報告を確認し、根拠付きQAを承認登録するStreamlitアプリです。
from __future__ import annotations

import json
from urllib.parse import quote

import requests
import streamlit as st


def normalize_base_url(value: str) -> str:
    return value.rstrip("/")


def get_json(base_url: str, path: str, params: dict | None = None) -> dict:
    res = requests.get(f"{base_url}{path}", params=params, timeout=60)
    res.raise_for_status()
    return res.json()


def post_json(base_url: str, path: str, payload: dict, timeout: int = 60) -> dict:
    res = requests.post(f"{base_url}{path}", json=payload, timeout=timeout)
    res.raise_for_status()
    return res.json()


def patch_json(base_url: str, path: str, payload: dict) -> dict:
    res = requests.patch(f"{base_url}{path}", json=payload, timeout=60)
    res.raise_for_status()
    return res.json()


def source_label(source: dict, idx: int) -> str:
    heading = source.get("heading_path") or source.get("title") or "見出しなし"
    try:
        score = float(source.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return f'{idx}. {heading} / スコア={score:.3f} / Corpus ID={source.get("corpus_id")} / 子ID={source.get("child_id")}'


REPORT_STATUS_LABELS = {
    "open": "未対応",
    "resolved": "対応済み",
    "ignored": "対応不要",
    "all": "すべて",
}

QA_STATUS_LABELS = {
    "approved": "有効",
    "disabled": "無効",
    "all": "すべて",
}

COLUMN_LABELS = {
    "id": "ID",
    "rank": "順位",
    "hits": "ヒット数",
    "count": "件数",
    "question": "質問",
    "question_preview": "質問",
    "answer": "回答",
    "answer_preview": "回答",
    "status": "状態",
    "enabled": "有効",
    "priority": "優先度",
    "corpus_id": "Corpus ID",
    "corpus_version": "Corpusバージョン",
    "index_version": "Indexバージョン",
    "display_name": "表示名",
    "description": "説明",
    "markdown_dir": "Markdownフォルダ",
    "markdown_files": "Markdown件数",
    "glob": "対象パターン",
    "file_pattern": "対象パターン",
    "created_at": "作成日時",
    "updated_at": "更新日時",
    "modified_at": "更新日時",
    "session_id": "セッションID",
    "log_id": "ログID",
    "parent_id": "親ID",
    "child_id": "子ID",
    "source_file": "参照元",
    "source_url": "参照URL",
    "heading_path": "見出し",
    "source_count": "根拠数",
    "top_k": "根拠取得数",
    "avg_score": "平均スコア",
    "max_score": "最大スコア",
    "last_hit_at": "最終ヒット日時",
    "warning_count": "警告数",
    "warning": "警告",
    "message": "内容",
    "path": "パス",
    "size_bytes": "サイズbytes",
    "row_count": "行数",
    "tagged_count": "タグ付き件数",
    "tag_count": "タグ数",
    "text_preview": "本文プレビュー",
}


def report_status_label(value: str | None) -> str:
    return REPORT_STATUS_LABELS.get(str(value or ""), str(value or ""))


def qa_status_label(value: str | None) -> str:
    return QA_STATUS_LABELS.get(str(value or ""), str(value or ""))


def enabled_label(value: object) -> str:
    return "有効" if bool(value) else "無効"


def localized_value(key: str, value: object) -> object:
    if key == "enabled":
        return enabled_label(value)
    if key == "status":
        raw = str(value or "")
        return REPORT_STATUS_LABELS.get(raw, QA_STATUS_LABELS.get(raw, raw))
    return value


def localized_table_rows(items: list[dict]) -> list[dict]:
    return [
        {
            COLUMN_LABELS.get(str(key), str(key)): localized_value(str(key), value)
            for key, value in item.items()
        }
        for item in items
    ]


def render_source_picker(sources: list[dict], key_prefix: str) -> list[dict]:
    selected = []
    for idx, source in enumerate(sources, start=1):
        checkbox_key = f"{key_prefix}_source_{idx}_{source.get('child_id', idx)}"
        with st.expander(source_label(source, idx), expanded=False):
            checked = st.checkbox("このチャンクを根拠に使う", key=checkbox_key)
            st.caption(
                f'参照元: {source.get("source_file")} / 親ID: {source.get("parent_id")} / '
                f'子ID: {source.get("child_id")}'
            )
            st.markdown("ヒットした子チャンク")
            st.write(source.get("child_text") or source.get("text") or "")
            if source.get("parent_text"):
                st.markdown("親チャンク")
                st.write(source.get("parent_text"))
            tags = source.get("search_tags") or []
            if tags:
                st.markdown("SearchTag")
                st.write(" / ".join(tags))
        if checked:
            selected.append(source)
    return selected


def search_evidence(base_url: str, query: str, top_k: int) -> list[dict]:
    data = post_json(
        base_url,
        "/admin/evidence/search",
        {"query": query, "top_k": top_k},
        timeout=180,
    )
    return data.get("sources", [])


def register_qa(
    base_url: str,
    *,
    question: str,
    answer: str,
    evidence: list[dict],
    approved_by: str,
    memo: str,
    source_report_id: int | None = None,
    corpus_version: str = "",
    index_version: str = "",
) -> int:
    payload = {
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "approved_by": approved_by,
        "memo": memo,
        "source_report_id": source_report_id,
        "corpus_version": corpus_version.strip() or None,
        "index_version": index_version.strip() or None,
    }
    data = post_json(base_url, "/admin/qa-cache", payload, timeout=180)
    return int(data["qa_id"])


def update_qa(base_url: str, qa_id: int, payload: dict) -> dict:
    return patch_json(base_url, f"/admin/qa-cache/{qa_id}", payload)


def short_text(value: object, limit: int = 120) -> str:
    text = str(value or "").strip().replace("\r\n", "\n")
    if len(text) <= limit:
        return text
    return text[:limit] + "...[省略]"


def log_label(log: dict) -> str:
    return f'#{log.get("id")} {log.get("created_at")} {short_text(log.get("question"), 80)}'


def log_table_rows(logs: list[dict]) -> list[dict]:
    return [
        {
            "ID": log.get("id"),
            "作成日時": log.get("created_at"),
            "セッションID": log.get("session_id"),
            "質問": log.get("question_preview") or short_text(log.get("question"), 120),
            "回答": log.get("answer_preview") or short_text(log.get("answer"), 160),
            "根拠数": log.get("source_count"),
            "根拠取得数": log.get("top_k"),
        }
        for log in logs
    ]


def qa_table_rows(items: list[dict]) -> list[dict]:
    return [
        {
            "id": item.get("id"),
            "状態": qa_status_label(item.get("status")),
            "更新日時": item.get("updated_at"),
            "質問": short_text(item.get("question"), 140),
            "回答": short_text(item.get("answer"), 180),
            "根拠数": item.get("evidence_count"),
            "Corpusバージョン": item.get("corpus_version"),
            "Indexバージョン": item.get("index_version"),
            "承認者": item.get("approved_by"),
        }
        for item in items
    ]


def evidence_json_text(evidence: list[dict]) -> str:
    return json.dumps(evidence or [], ensure_ascii=False, indent=2)


def parse_evidence_json(raw: str) -> list[dict]:
    value = json.loads(raw or "[]")
    if not isinstance(value, list):
        raise ValueError("根拠JSONは配列にしてください。")
    return value


def chunk_rank_rows(chunks: list[dict]) -> list[dict]:
    return [
        {
            "順位": item.get("rank"),
            "ヒット数": item.get("hits"),
            "平均スコア": item.get("avg_score"),
            "最大スコア": item.get("max_score"),
            "Corpus ID": item.get("corpus_id"),
            "子ID": item.get("child_id"),
            "見出し": short_text(item.get("heading_path"), 120),
            "参照元": short_text(item.get("source_file"), 120),
            "最終ヒット日時": item.get("last_hit_at"),
        }
        for item in chunks
    ]


def source_file_rank_rows(items: list[dict]) -> list[dict]:
    return [
        {
            "順位": item.get("rank"),
            "ヒット数": item.get("hits"),
            "Corpus ID": item.get("corpus_id"),
            "参照元": short_text(item.get("source_file") or item.get("source_url"), 160),
            "最終ヒット日時": item.get("last_hit_at"),
        }
        for item in items
    ]


def chunk_chart_rows(chunks: list[dict]) -> list[dict]:
    return [
        {
            "チャンク": f'{item.get("rank")}. {short_text(item.get("heading_path") or item.get("child_id"), 48)}',
            "ヒット数": item.get("hits", 0),
        }
        for item in chunks[:10]
    ]


def source_file_chart_rows(items: list[dict]) -> list[dict]:
    return [
        {
            "参照元": f'{item.get("rank")}. {short_text(item.get("source_file") or item.get("source_url"), 48)}',
            "ヒット数": item.get("hits", 0),
        }
        for item in items[:10]
    ]


def question_chart_rows(items: list[dict]) -> list[dict]:
    return [
        {
            "質問": f'Q{item.get("rank")}',
            "回数": item.get("hits", 0),
        }
        for item in items[:10]
    ]


def quality_chart_rows(items: list[dict]) -> list[dict]:
    return [
        {
            "日付": item.get("date"),
            "根拠なし": item.get("no_hit", 0),
            "低信頼": item.get("low_confidence", 0),
        }
        for item in items
    ]


def quality_log_rows(items: list[dict]) -> list[dict]:
    return [
        {
            "ID": item.get("id"),
            "作成日時": item.get("created_at"),
            "セッションID": item.get("session_id"),
            "最大スコア": item.get("max_score"),
            "根拠数": item.get("source_count"),
            "質問": item.get("question_preview") or short_text(item.get("question"), 160),
            "回答": item.get("answer_preview"),
            "理由": item.get("debug_reason"),
        }
        for item in items
    ]


def chunk_file_rows(chunks: dict) -> list[dict]:
    labels = {
        "parent_chunks": "親チャンク",
        "child_chunks": "子チャンク",
        "child_chunks_with_tags": "SearchTag付き子チャンク",
    }
    rows = []
    for key, label in labels.items():
        item = chunks.get(key, {})
        rows.append({
            "種別": label,
            "有無": "あり" if item.get("exists") else "なし",
            "行数": item.get("row_count"),
            "タグ付き件数": item.get("tagged_count"),
            "サイズbytes": item.get("size_bytes"),
            "更新日時": item.get("modified_at"),
            "パス": item.get("path"),
        })
    report = chunks.get("chunk_report", {})
    rows.append({
        "種別": "チャンク監査レポート",
        "有無": "あり" if report.get("exists") else "なし",
        "行数": report.get("row_count"),
        "タグ付き件数": None,
        "サイズbytes": report.get("size_bytes"),
        "更新日時": report.get("modified_at"),
        "パス": report.get("path"),
    })
    return rows


def storage_rows(storage: dict) -> list[dict]:
    return [
        {
            "保存領域": name,
            "有無": "あり" if item.get("exists") else "なし",
            "サイズbytes": item.get("size_bytes"),
            "更新日時": item.get("modified_at"),
            "パス": item.get("path"),
        }
        for name, item in storage.items()
    ]


def child_corpus_chart_rows(chunks: dict) -> list[dict]:
    return [
        {"Corpus ID": row.get("corpus_id"), "子チャンク数": row.get("count", 0)}
        for row in chunks.get("child_chunks", {}).get("by_corpus", [])
    ]


def chunk_report_corpus_rows(chunks: dict) -> list[dict]:
    return localized_table_rows(chunks.get("chunk_report", {}).get("by_corpus", []))


def search_tag_table_rows(items: list[dict]) -> list[dict]:
    return [
        {
            "子ID": item.get("child_id"),
            "Corpus ID": item.get("corpus_id"),
            "タグ数": item.get("tag_count"),
            "見出し": short_text(item.get("heading_path"), 140),
            "参照元": short_text(item.get("source_file") or item.get("source_url"), 120),
            "SearchTag": " / ".join(item.get("search_tags") or []),
            "本文プレビュー": short_text(item.get("text_preview"), 160),
        }
        for item in items
    ]


def search_tag_label(item: dict) -> str:
    return f'{item.get("child_id")} / {short_text(item.get("heading_path"), 90)}'


def corpus_table_rows(corpora: list[dict]) -> list[dict]:
    return localized_table_rows(corpora)


def daily_question_rows(items: list[dict]) -> list[dict]:
    return [{"日付": item.get("date"), "質問数": item.get("questions", 0)} for item in items]


def hourly_question_rows(items: list[dict]) -> list[dict]:
    return [{"時間": item.get("hour"), "質問数": item.get("questions", 0)} for item in items]


def top_question_rows(items: list[dict]) -> list[dict]:
    return [
        {
            "順位": item.get("rank"),
            "回数": item.get("hits"),
            "質問": item.get("question_preview") or short_text(item.get("question"), 180),
            "最終質問日時": item.get("last_asked_at") or item.get("last_hit_at"),
        }
        for item in items
    ]


def tags_to_editor_text(tags: list[str]) -> str:
    return "\n".join(str(tag) for tag in tags or [])


def parse_tag_editor_text(raw: str) -> list[str]:
    tags = []
    seen = set()
    for part in (raw or "").replace("\r", "\n").replace(",", "\n").replace("、", "\n").replace("，", "\n").split("\n"):
        tag = part.strip()
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


st.set_page_config(page_title="RAG管理者アプリ", layout="wide")
st.title("RAG管理者アプリ")

base_url = normalize_base_url(st.sidebar.text_input("APIベースURL", value="http://127.0.0.1:8000"))
approved_by = st.sidebar.text_input("承認者", value="admin")
default_top_k = st.sidebar.slider("根拠検索数", min_value=3, max_value=20, value=8)

report_tab, manual_tab, approved_tab, logs_tab, dashboard_tab, index_tab, search_tag_tab = st.tabs(
    ["報告対応", "QA登録", "承認済みQA", "ログ閲覧", "ログ分析", "文書/Index", "SearchTag編集"]
)

with report_tab:
    status_filter = st.selectbox(
        "表示する報告",
        ["open", "resolved", "ignored", "all"],
        index=0,
        format_func=report_status_label,
    )
    params = {"limit": 100}
    if status_filter != "all":
        params["status"] = status_filter

    try:
        reports = get_json(base_url, "/admin/reports", params=params).get("reports", [])
    except Exception as exc:
        st.error(f"報告一覧を取得できませんでした: {exc}")
        reports = []

    if not reports:
        st.info("対象の報告はありません。")
    else:
        labels = [f'#{r["id"]} [{report_status_label(r.get("status"))}] {r["question"][:80]}' for r in reports]
        selected_label = st.selectbox("報告を選択", labels)
        report = reports[labels.index(selected_label)]
        report_id = int(report["id"])

        col1, col2 = st.columns(2)
        with col1:
            st.text_area("報告された質問", value=report.get("question", ""), height=120, disabled=True)
            st.text_area("報告された回答", value=report.get("answer", ""), height=220, disabled=True)
        with col2:
            st.text_area("ユーザーコメント", value=report.get("comment") or "", height=120, disabled=True)
            st.write(f'セッションID: {report.get("session_id")}')
            st.write(f'ログID: {report.get("log_id")}')
            st.write(f'作成日時: {report.get("created_at")}')

        correct_question = st.text_area("承認QAとして登録する質問", value=report.get("question", ""), key=f"report_q_{report_id}")
        correct_answer = st.text_area("正しい回答", value="", height=220, key=f"report_a_{report_id}")

        search_query = st.text_input("根拠検索クエリ", value=correct_question, key=f"report_search_q_{report_id}")
        if st.button("根拠検索", key=f"report_search_button_{report_id}"):
            try:
                st.session_state[f"report_sources_{report_id}"] = search_evidence(base_url, search_query, default_top_k)
            except Exception as exc:
                st.error(f"根拠検索に失敗しました: {exc}")

        report_sources = st.session_state.get(f"report_sources_{report_id}", [])
        selected_evidence = render_source_picker(report_sources, f"report_{report_id}") if report_sources else []

        corpus_version = st.text_input("Corpusバージョン（空欄なら設定値）", value="", key=f"report_corpus_version_{report_id}")
        index_version = st.text_input("Indexバージョン（空欄なら設定値）", value="", key=f"report_index_version_{report_id}")
        memo = st.text_area("管理メモ", value="", key=f"report_memo_{report_id}")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("承認QAとして登録", type="primary", key=f"register_report_{report_id}"):
                if not correct_question.strip() or not correct_answer.strip():
                    st.warning("質問と正しい回答を入力してください。")
                elif not selected_evidence:
                    st.warning("根拠チャンクを1つ以上選択してください。")
                else:
                    try:
                        qa_id = register_qa(
                            base_url,
                            question=correct_question,
                            answer=correct_answer,
                            evidence=selected_evidence,
                            approved_by=approved_by,
                            memo=memo,
                            source_report_id=report_id,
                            corpus_version=corpus_version,
                            index_version=index_version,
                        )
                        st.success(f"承認QAを登録しました。qa_id={qa_id}")
                    except Exception as exc:
                        st.error(f"承認QA登録に失敗しました: {exc}")
        with col2:
            if st.button("未対応に戻す", key=f"report_open_{report_id}"):
                try:
                    patch_json(base_url, f"/admin/reports/{report_id}/status", {"status": "open"})
                    st.success("ステータスを未対応にしました。")
                except Exception as exc:
                    st.error(f"ステータス更新に失敗しました: {exc}")
        with col3:
            if st.button("対応不要にする", key=f"report_ignored_{report_id}"):
                try:
                    patch_json(base_url, f"/admin/reports/{report_id}/status", {"status": "ignored"})
                    st.success("ステータスを対応不要にしました。")
                except Exception as exc:
                    st.error(f"ステータス更新に失敗しました: {exc}")

with manual_tab:
    manual_question = st.text_area("質問", height=120, key="manual_question")
    manual_answer = st.text_area("回答", height=220, key="manual_answer")
    manual_search_query = st.text_input("根拠検索クエリ", value=manual_question, key="manual_search_query")

    if st.button("根拠検索", key="manual_search_button"):
        try:
            st.session_state["manual_sources"] = search_evidence(base_url, manual_search_query, default_top_k)
        except Exception as exc:
            st.error(f"根拠検索に失敗しました: {exc}")

    manual_sources = st.session_state.get("manual_sources", [])
    selected_manual_evidence = render_source_picker(manual_sources, "manual") if manual_sources else []
    manual_corpus_version = st.text_input("Corpusバージョン（空欄なら設定値）", value="", key="manual_corpus_version")
    manual_index_version = st.text_input("Indexバージョン（空欄なら設定値）", value="", key="manual_index_version")
    manual_memo = st.text_area("管理メモ", value="", key="manual_memo")

    if st.button("承認QAとして登録", type="primary", key="register_manual"):
        if not manual_question.strip() or not manual_answer.strip():
            st.warning("質問と回答を入力してください。")
        elif not selected_manual_evidence:
            st.warning("根拠チャンクを1つ以上選択してください。")
        else:
            try:
                qa_id = register_qa(
                    base_url,
                    question=manual_question,
                    answer=manual_answer,
                    evidence=selected_manual_evidence,
                    approved_by=approved_by,
                    memo=manual_memo,
                    corpus_version=manual_corpus_version,
                    index_version=manual_index_version,
                )
                st.success(f"承認QAを登録しました。qa_id={qa_id}")
            except Exception as exc:
                st.error(f"承認QA登録に失敗しました: {exc}")

with approved_tab:
    col1, col2 = st.columns(2)
    with col1:
        limit = st.slider("表示件数", min_value=10, max_value=200, value=50)
    with col2:
        qa_status_filter = st.selectbox(
            "QA状態",
            ["all", "approved", "disabled"],
            index=0,
            format_func=qa_status_label,
        )
    try:
        items = get_json(base_url, "/admin/qa-cache", params={"limit": limit, "status": qa_status_filter}).get("items", [])
    except Exception as exc:
        st.error(f"承認済みQAを取得できませんでした: {exc}")
        items = []

    if not items:
        st.info("承認済みQAはまだありません。")
    else:
        st.dataframe(qa_table_rows(items), use_container_width=True, hide_index=True)
        labels = [f'#{item["id"]} [{qa_status_label(item.get("status"))}] {short_text(item.get("question"), 90)}' for item in items]
        selected_qa_label = st.selectbox("編集するQA", labels)
        selected_qa_id = int(items[labels.index(selected_qa_label)]["id"])

        try:
            qa_detail = get_json(base_url, f"/admin/qa-cache/{selected_qa_id}")
        except Exception as exc:
            st.error(f"承認済みQAの詳細を取得できませんでした: {exc}")
            qa_detail = {}

        if qa_detail:
            edit_status = st.selectbox(
                "QA状態",
                ["approved", "disabled"],
                index=0 if qa_detail.get("status") == "approved" else 1,
                format_func=qa_status_label,
                key=f"qa_status_{selected_qa_id}",
            )
            edit_question = st.text_area(
                "質問",
                value=qa_detail.get("question", ""),
                height=120,
                key=f"qa_question_{selected_qa_id}",
            )
            edit_answer = st.text_area(
                "回答",
                value=qa_detail.get("answer", ""),
                height=220,
                key=f"qa_answer_{selected_qa_id}",
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                edit_corpus_version = st.text_input(
                    "Corpusバージョン",
                    value=qa_detail.get("corpus_version", ""),
                    key=f"qa_corpus_version_{selected_qa_id}",
                )
            with col2:
                edit_index_version = st.text_input(
                    "Indexバージョン",
                    value=qa_detail.get("index_version", ""),
                    key=f"qa_index_version_{selected_qa_id}",
                )
            with col3:
                edit_approved_by = st.text_input(
                    "承認者",
                    value=qa_detail.get("approved_by") or approved_by,
                    key=f"qa_approved_by_{selected_qa_id}",
                )
            edit_memo = st.text_area("管理メモ", value=qa_detail.get("memo") or "", key=f"qa_memo_{selected_qa_id}")

            edit_search_query = st.text_input(
                "根拠再検索クエリ",
                value=edit_question,
                key=f"qa_search_query_{selected_qa_id}",
            )
            if st.button("根拠再検索", key=f"qa_search_button_{selected_qa_id}"):
                try:
                    st.session_state[f"qa_edit_sources_{selected_qa_id}"] = search_evidence(
                        base_url,
                        edit_search_query,
                        default_top_k,
                    )
                except Exception as exc:
                    st.error(f"根拠検索に失敗しました: {exc}")

            qa_edit_sources = st.session_state.get(f"qa_edit_sources_{selected_qa_id}", [])
            selected_edit_evidence = render_source_picker(qa_edit_sources, f"qa_edit_{selected_qa_id}") if qa_edit_sources else []
            evidence_raw = st.text_area(
                "根拠JSON（再検索で選択した場合は選択根拠が優先されます）",
                value=evidence_json_text(qa_detail.get("evidence", [])),
                height=260,
                key=f"qa_evidence_json_{selected_qa_id}",
            )

            def build_update_payload(next_status: str) -> dict | None:
                try:
                    evidence = selected_edit_evidence or parse_evidence_json(evidence_raw)
                except Exception as exc:
                    st.error(f"根拠JSONを読み込めませんでした: {exc}")
                    return None
                if not edit_question.strip() or not edit_answer.strip():
                    st.warning("質問と回答を入力してください。")
                    return None
                return {
                    "question": edit_question,
                    "answer": edit_answer,
                    "evidence": evidence,
                    "status": next_status,
                    "corpus_version": edit_corpus_version.strip() or None,
                    "index_version": edit_index_version.strip() or None,
                    "approved_by": edit_approved_by,
                    "memo": edit_memo,
                }

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("QAを更新", type="primary", key=f"qa_update_{selected_qa_id}"):
                    payload = build_update_payload(edit_status)
                    if payload:
                        try:
                            update_qa(base_url, selected_qa_id, payload)
                            st.success("承認済みQAを更新しました。")
                        except Exception as exc:
                            st.error(f"承認済みQAの更新に失敗しました: {exc}")
            with col2:
                if st.button("無効化", key=f"qa_disable_{selected_qa_id}"):
                    payload = build_update_payload("disabled")
                    if payload:
                        try:
                            update_qa(base_url, selected_qa_id, payload)
                            st.success("承認済みQAを無効化しました。")
                        except Exception as exc:
                            st.error(f"承認済みQAの無効化に失敗しました: {exc}")
            with col3:
                if st.button("再有効化", key=f"qa_enable_{selected_qa_id}"):
                    payload = build_update_payload("approved")
                    if payload:
                        try:
                            update_qa(base_url, selected_qa_id, payload)
                            st.success("承認済みQAを再有効化しました。")
                        except Exception as exc:
                            st.error(f"承認済みQAの再有効化に失敗しました: {exc}")

with logs_tab:
    log_limit = st.slider("ログ表示件数", min_value=10, max_value=500, value=100, key="admin_log_limit")
    try:
        logs = get_json(base_url, "/admin/logs/recent", params={"limit": log_limit}).get("logs", [])
    except Exception as exc:
        st.error(f"ログを取得できませんでした: {exc}")
        logs = []

    if not logs:
        st.info("ログはまだありません。")
    else:
        st.dataframe(log_table_rows(logs), use_container_width=True, hide_index=True)
        selected_log_label = st.selectbox("ログ詳細", [log_label(log) for log in logs], key="selected_log_detail")
        selected_log = logs[[log_label(log) for log in logs].index(selected_log_label)]

        col1, col2 = st.columns(2)
        with col1:
            st.text_area("質問", value=selected_log.get("question", ""), height=140, disabled=True)
            st.write(f'ID: {selected_log.get("id")}')
            st.write(f'セッションID: {selected_log.get("session_id")}')
            st.write(f'作成日時: {selected_log.get("created_at")}')
        with col2:
            st.text_area("回答", value=selected_log.get("answer", ""), height=260, disabled=True)
            st.write(f'根拠取得数: {selected_log.get("top_k")}')
            st.write(f'検索対象Corpus: {selected_log.get("corpus_ids")}')
            st.write(f'根拠数: {selected_log.get("source_count")}')

        sources = selected_log.get("sources") or []
        if sources:
            st.markdown("回答時の根拠チャンク")
            for idx, source in enumerate(sources, start=1):
                with st.expander(source_label(source, idx), expanded=False):
                    st.caption(
                        f'参照元: {source.get("source_file")} / 親ID: {source.get("parent_id")} / '
                        f'子ID: {source.get("child_id")}'
                    )
                    st.write(source.get("child_text") or source.get("text") or "")
        debug = selected_log.get("debug")
        if debug:
            with st.expander("デバッグ情報", expanded=False):
                st.json(debug)

with dashboard_tab:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sample_limit = st.slider("ランキング集計対象ログ数", min_value=100, max_value=5000, value=1000, step=100)
    with col2:
        top_n = st.slider("ランキング表示件数", min_value=5, max_value=50, value=20)
    with col3:
        dashboard_days = st.slider("分析対象日数", min_value=1, max_value=90, value=7)
    with col4:
        low_score_threshold = st.slider("低信頼しきい値", min_value=0.0, max_value=1.0, value=0.35, step=0.05)

    try:
        dashboard = get_json(
            base_url,
            "/admin/logs/dashboard",
            params={
                "sample_limit": sample_limit,
                "top_n": top_n,
                "days": dashboard_days,
                "low_score_threshold": low_score_threshold,
            },
        )
    except Exception as exc:
        st.error(f"ログ分析を取得できませんでした: {exc}")
        dashboard = {}

    if dashboard:
        overview = dashboard.get("overview", {})
        sample = dashboard.get("sample", {})
        metric_cols = st.columns(6)
        metric_cols[0].metric("総質問数", overview.get("total_questions", 0))
        metric_cols[1].metric(f"{dashboard_days}日", overview.get("window_questions", 0))
        metric_cols[2].metric(f"{dashboard_days}日セッション", overview.get("window_sessions", 0))
        metric_cols[3].metric("24時間", overview.get("last_24h_questions", 0))
        metric_cols[4].metric("7日", overview.get("last_7d_questions", 0))
        metric_cols[5].metric("根拠ありログ", overview.get("logs_with_sources", 0))
        st.caption(
            f'集計対象: 直近{sample.get("sampled_questions", 0)}件 / '
            f'根拠なし: {sample.get("sampled_no_source_questions", 0)}件 / '
            f'期間: {overview.get("first_log_at")} - {overview.get("last_log_at")}'
        )

        quality = dashboard.get("quality", {})
        quality_cols = st.columns(3)
        quality_cols[0].metric("根拠なし", quality.get("no_hit_count", 0))
        quality_cols[1].metric("低信頼", quality.get("low_confidence_count", 0))
        quality_cols[2].metric("低信頼しきい値", quality.get("low_score_threshold", low_score_threshold))

        daily_questions = dashboard.get("daily_questions", [])
        hourly_questions = dashboard.get("hourly_questions", [])
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("日次質問数")
            if daily_questions:
                st.line_chart(daily_question_rows(daily_questions), x="日付", y="質問数", use_container_width=True)
            else:
                st.info("日次質問数はまだありません。")
        with chart_col2:
            st.markdown("直近24時間の時間別質問数")
            if hourly_questions:
                st.bar_chart(hourly_question_rows(hourly_questions), x="時間", y="質問数", use_container_width=True)
            else:
                st.info("直近24時間のログはまだありません。")

        st.markdown("根拠なし / 低信頼")
        daily_quality = quality.get("daily_quality", [])
        if daily_quality:
            st.bar_chart(quality_chart_rows(daily_quality), x="日付", use_container_width=True)
        no_hit_logs = quality.get("no_hit_logs", [])
        low_confidence_logs = quality.get("low_confidence_logs", [])
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("根拠なしログ")
            if no_hit_logs:
                st.dataframe(quality_log_rows(no_hit_logs), use_container_width=True, hide_index=True)
            else:
                st.info("根拠なしログはありません。")
        with col2:
            st.markdown("低信頼ログ")
            if low_confidence_logs:
                st.dataframe(quality_log_rows(low_confidence_logs), use_container_width=True, hide_index=True)
            else:
                st.info("低信頼ログはありません。")

        st.markdown("LLM傾向レポート")
        if st.button("直近ログの傾向レポート生成", type="primary"):
            try:
                report_data = post_json(
                    base_url,
                    "/admin/logs/report",
                    {
                        "days": dashboard_days,
                        "sample_limit": sample_limit,
                        "top_n": top_n,
                        "low_score_threshold": low_score_threshold,
                    },
                    timeout=240,
                )
                st.session_state["log_trend_report"] = report_data.get("report", "")
            except Exception as exc:
                st.error(f"傾向レポート生成に失敗しました: {exc}")
        if st.session_state.get("log_trend_report"):
            st.markdown(st.session_state["log_trend_report"])

        chunks = dashboard.get("top_hit_chunks", [])
        st.markdown("子チャンクヒットランキング")
        if chunks:
            st.bar_chart(chunk_chart_rows(chunks), x="チャンク", y="ヒット数", use_container_width=True)
            st.dataframe(chunk_rank_rows(chunks), use_container_width=True, hide_index=True)
            selected_chunk_label = st.selectbox(
                "チャンク詳細",
                [
                    f'#{item.get("rank")} ヒット数={item.get("hits")} {short_text(item.get("heading_path"), 90)}'
                    for item in chunks
                ],
                key="selected_dashboard_chunk",
            )
            selected_chunk = chunks[
                [
                    f'#{item.get("rank")} ヒット数={item.get("hits")} {short_text(item.get("heading_path"), 90)}'
                    for item in chunks
                ].index(selected_chunk_label)
            ]
            with st.expander("選択チャンクの詳細", expanded=True):
                st.write(f'Corpus ID: {selected_chunk.get("corpus_id")}')
                st.write(f'親ID: {selected_chunk.get("parent_id")}')
                st.write(f'子ID: {selected_chunk.get("child_id")}')
                st.write(f'参照元: {selected_chunk.get("source_file")}')
                st.write(f'最終ヒット日時: {selected_chunk.get("last_hit_at")}')
                st.write(f'サンプルログID: {selected_chunk.get("sample_log_ids")}')
                st.markdown("子チャンク本文プレビュー")
                st.write(selected_chunk.get("child_text_preview") or "")
        else:
            st.info("根拠チャンクのヒットログはまだありません。")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("参照ファイルランキング")
            source_files = dashboard.get("top_source_files", [])
            if source_files:
                st.bar_chart(source_file_chart_rows(source_files), x="参照元", y="ヒット数", use_container_width=True)
                st.dataframe(source_file_rank_rows(source_files), use_container_width=True, hide_index=True)
            else:
                st.info("参照ファイルのヒットログはまだありません。")
        with col2:
            st.markdown("頻出質問ランキング")
            top_questions = dashboard.get("top_questions", [])
            if top_questions:
                st.bar_chart(question_chart_rows(top_questions), x="質問", y="回数", use_container_width=True)
                st.dataframe(top_question_rows(top_questions), use_container_width=True, hide_index=True)
            else:
                st.info("質問ログはまだありません。")

with index_tab:
    try:
        index_status = get_json(base_url, "/admin/index/status")
    except Exception as exc:
        st.error(f"文書/Index状態を取得できませんでした: {exc}")
        index_status = {}

    if index_status:
        corpora = index_status.get("corpora", [])
        chunks = index_status.get("chunks", {})
        child_chunks = chunks.get("child_chunks", {})
        tagged_chunks = chunks.get("child_chunks_with_tags", {})
        chunk_report = chunks.get("chunk_report", {})
        vector_collection = index_status.get("vector_collection", {})

        metric_cols = st.columns(6)
        metric_cols[0].metric("有効Corpus", sum(1 for corpus in corpora if corpus.get("enabled")))
        metric_cols[1].metric("Markdown", sum(int(corpus.get("markdown_files") or 0) for corpus in corpora))
        metric_cols[2].metric("親チャンク", chunks.get("parent_chunks", {}).get("row_count", 0))
        metric_cols[3].metric("子チャンク", child_chunks.get("row_count", 0))
        metric_cols[4].metric("SearchTag付き", tagged_chunks.get("tagged_count", 0))
        metric_cols[5].metric("監査警告", chunk_report.get("warning_count", 0))

        st.markdown("Corpus一覧")
        if corpora:
            st.dataframe(corpus_table_rows(corpora), use_container_width=True, hide_index=True)
        else:
            st.info("corpus_settings.json が見つからないか、Corpusが未設定です。")

        chart_rows = child_corpus_chart_rows(chunks)
        if chart_rows:
            st.markdown("Corpus別 子チャンク数")
            st.bar_chart(chart_rows, x="Corpus ID", y="子チャンク数", use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("チャンクファイル")
            st.dataframe(chunk_file_rows(chunks), use_container_width=True, hide_index=True)
        with col2:
            st.markdown("保存領域")
            st.dataframe(storage_rows(index_status.get("storage", {})), use_container_width=True, hide_index=True)

        report_rows = chunk_report_corpus_rows(chunks)
        if report_rows:
            st.markdown("チャンク監査 Corpus別集計")
            st.dataframe(report_rows, use_container_width=True, hide_index=True)

        warnings = chunk_report.get("warnings", [])
        if warnings:
            with st.expander("チャンク監査警告", expanded=False):
                st.dataframe(localized_table_rows(warnings), use_container_width=True, hide_index=True)

        st.markdown("ベクトルDB / バージョン")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.json(vector_collection)
        with col2:
            st.markdown("回答キャッシュ設定")
            st.json(index_status.get("answer_cache", {}))
        with col3:
            st.markdown("ベクトルDB設定")
            st.json(index_status.get("vector_db", {}))

        with st.expander("検索設定", expanded=False):
            st.json(index_status.get("retrieval_settings", {}))

with search_tag_tab:
    st.info(
        "SearchTagはBM25検索と検索候補文に効きます。保存後に検索器を再読み込みすると即時反映されますが、"
        "Dense/Vector検索へ完全に反映するには scripts/03_build_index.py の再実行が必要です。"
    )
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        tag_query = st.text_input("検索キーワード", key="search_tag_query")
    with col2:
        tag_corpus_id = st.text_input("Corpus ID", key="search_tag_corpus_id")
    with col3:
        tag_limit = st.slider("表示件数", min_value=10, max_value=500, value=100, step=10, key="search_tag_limit")

    tag_params = {"limit": tag_limit}
    if tag_query.strip():
        tag_params["query"] = tag_query.strip()
    if tag_corpus_id.strip():
        tag_params["corpus_id"] = tag_corpus_id.strip()

    try:
        tag_data = get_json(base_url, "/admin/search-tags", params=tag_params)
        tag_items = tag_data.get("items", [])
    except Exception as exc:
        st.error(f"SearchTag一覧を取得できませんでした: {exc}")
        tag_data = {}
        tag_items = []

    if tag_data:
        st.caption(
            f'対象ファイル: {tag_data.get("source_file")} / 編集先: {tag_data.get("editable_file")} / '
            f'該当: {tag_data.get("total_matches", 0)}件'
        )

    if not tag_items:
        st.info("条件に一致する子チャンクがありません。")
    else:
        st.dataframe(search_tag_table_rows(tag_items), use_container_width=True, hide_index=True)
        tag_labels = [search_tag_label(item) for item in tag_items]
        selected_tag_label = st.selectbox("編集する子チャンク", tag_labels, key="selected_search_tag_child")
        selected_tag_item = tag_items[tag_labels.index(selected_tag_label)]
        selected_child_id = str(selected_tag_item.get("child_id") or "")
        encoded_child_id = quote(selected_child_id, safe="")

        try:
            tag_detail_data = get_json(base_url, f"/admin/search-tags/{encoded_child_id}")
            tag_detail = tag_detail_data.get("item", {})
        except Exception as exc:
            st.error(f"SearchTag詳細を取得できませんでした: {exc}")
            tag_detail = {}

        if tag_detail:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f'子ID: {tag_detail.get("child_id")}')
                st.write(f'親ID: {tag_detail.get("parent_id")}')
                st.write(f'Corpus ID: {tag_detail.get("corpus_id")}')
                st.write(f'参照元: {tag_detail.get("source_file") or tag_detail.get("source_url")}')
                st.write(f'見出し: {tag_detail.get("heading_path")}')
            with col2:
                st.text_area(
                    "子チャンク本文",
                    value=tag_detail.get("text") or tag_detail.get("child_text") or "",
                    height=220,
                    disabled=True,
                    key=f"search_tag_child_text_{selected_child_id}",
                )

            tag_editor_text = st.text_area(
                "SearchTag（1行1件。カンマ区切りも可）",
                value=tags_to_editor_text(tag_detail.get("search_tags") or []),
                height=220,
                key=f"search_tag_editor_{selected_child_id}",
            )
            parsed_tags = parse_tag_editor_text(tag_editor_text)
            st.caption(f"保存予定タグ数: {len(parsed_tags)}")
            reload_after_save = st.checkbox(
                "保存後に検索器を再読み込みする",
                value=True,
                key=f"search_tag_reload_{selected_child_id}",
            )

            if st.button("SearchTagを保存", type="primary", key=f"search_tag_save_{selected_child_id}"):
                try:
                    result = patch_json(
                        base_url,
                        f"/admin/search-tags/{encoded_child_id}",
                        {"search_tags": parsed_tags, "reload_retriever": reload_after_save},
                    )
                    st.success("SearchTagを更新しました。")
                    if result.get("reload_error"):
                        st.warning(f'検索器の再読み込みに失敗しました: {result.get("reload_error")}')
                    if result.get("warning"):
                        st.caption(result["warning"])
                except Exception as exc:
                    st.error(f"SearchTag更新に失敗しました: {exc}")
