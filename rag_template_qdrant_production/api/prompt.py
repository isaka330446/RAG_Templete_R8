# LLMへ渡すシステムプロンプト、検索クエリ、ユーザープロンプトを組み立てます。
SYSTEM_PROMPT = """あなたは、社内文書に基づいて回答するRAGアシスタントです。

必ず守ること:
- 回答は根拠文書に基づく。
- 根拠にないことは断定しない。
- 根拠が不足している場合は、不足していると明示する。
- 可能な限り、条・章・見出し・様式名を示す。
- 推測と根拠に基づく記述を混ぜない。
- 回答は日本語で簡潔かつ実務的に書く。
"""

def build_history_text(history: list[dict], max_messages: int = 10) -> str:
    rows = []
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
    blocks = []
    for i, c in enumerate(contexts, start=1):
        forms = c.get("forms") or []
        form_lines = "\n".join(
            f"- {f.get('form_name')} ({f.get('file_path')})"
            for f in forms
            if f.get("form_name")
        )
        blocks.append(
            f"""[根拠{i}]
corpus_id: {c.get('corpus_id')}
source_file: {c.get('source_file')}
heading_path: {c.get('heading_path')}
parent_id: {c.get('parent_id')}
関連様式:
{form_lines or "なし"}

{c.get('parent_text') or c.get('child_text')}
"""
        )

    context_text = "\n\n".join(blocks)

    return f"""以下の根拠文書だけを使って質問に回答してください。
会話履歴は更問の意図理解にだけ使い、根拠文書にない内容は断定しないでください。

# 会話履歴
{history_text or "なし"}

# 質問
{question}

# 根拠文書
{context_text}

# 回答条件
- 根拠に基づく結論を先に述べる。
- 必要なら箇条書きで整理する。
- 根拠が不足する場合は「このテンプレート内の文書だけでは確認できません」と書く。
- 関連する様式名が根拠に含まれる場合は、様式名も示す。
"""
