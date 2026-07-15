from __future__ import annotations


SYSTEM_PROMPT = """あなたは、登録済みの業務文書を根拠に回答するRAGアシスタントです。

必ず守ること:
- 回答は提示された根拠文書に基づいてください。
- 根拠にない内容を断定しないでください。
- 個別判断や追加確認が必要な場合は、その旨を明示してください。
- 文書ID、文書種別、タイトル、見出し、版日、参照URLが根拠に含まれる場合は示してください。
- 根拠を示すときは、本文中に [根拠1] の形式で根拠番号を書いてください。
- 複数の根拠を使う場合は [根拠1][根拠2] のように番号を分け、番号を作り変えないでください。
- 回答は日本語で簡潔かつ実務的に書いてください。
"""


def build_history_text(history: list[dict], max_messages: int = 10) -> str:
    rows: list[str] = []
    for item in history[-max_messages:]:
        role = "ユーザー" if item.get("role") == "user" else "アシスタント"
        content = str(item.get("content", "")).strip()
        if content:
            rows.append(f"{role}: {content}")
    return "\n".join(rows)


def build_retrieval_query(question: str, history: list[dict], max_messages: int = 4) -> str:
    history_text = build_history_text(history, max_messages=max_messages)
    if not history_text:
        return question
    return f"{history_text}\nユーザー: {question}"


def build_user_prompt(question: str, contexts: list[dict], history: list[dict] | None = None) -> str:
    history_text = build_history_text(history or [])
    blocks: list[str] = []
    for i, c in enumerate(contexts, start=1):
        blocks.append(
            f"""[根拠{i}]
corpus_id: {c.get('corpus_id')}
document_type: {c.get('document_type')}
document_id: {c.get('document_id')}
document_series: {c.get('document_series')}
version_date: {c.get('version_date')}
source_url: {c.get('source_url')}
source_file: {c.get('source_file')}
heading_path: {c.get('heading_path')}
parent_id: {c.get('parent_id')}

{c.get('parent_text') or c.get('child_text')}
"""
        )

    context_text = "\n\n".join(blocks)

    return f"""以下の根拠文書だけを使って質問に回答してください。会話履歴は質問の意図理解にだけ使い、根拠文書にない内容は断定しないでください。

# 会話履歴
{history_text or "なし"}

# 質問
{question}

# 根拠文書
{context_text}

# 回答条件
- まず結論を短く述べてください。
- 重要な主張、条件、例外、注意点には、使った根拠番号を [根拠1] の形式で付けてください。
- 根拠欄を作る場合も、各項目の先頭に [根拠1] の形式を付けてから、文書ID、文書種別、タイトル、URL、版日を書いてください。
- 個別判断や追加確認が必要な場合は、その理由を根拠に基づいて説明してください。
- 根拠が不足する場合は、「登録済み文書だけでは確認できません」と書いてください。
"""
