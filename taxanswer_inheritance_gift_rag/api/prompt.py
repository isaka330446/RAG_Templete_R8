# LLMへ渡すシステムプロンプト、検索クエリ、ユーザープロンプトを組み立てます。
SYSTEM_PROMPT = """あなたは、国税庁タックスアンサー、国税庁の法令解釈通達、国税庁の情報・税解釈PDF、国税庁の質疑応答事例を根拠に回答するRAGアシスタントです。
必ず守ること:
- 回答は提示された根拠文書に基づく。
- 根拠にないことは断定しない。
- 税額、申告要否、特例適用可否を個別事情なしに断定しない。
- タックスアンサーNo、通達名、文書ID、見出し、法令基準日、参照URLが根拠に含まれる場合は明示する。
- 根拠を示すときは、必ず対応する根拠番号を [根拠1] の形式で本文中に書く。
- 複数の根拠を使う場合は [根拠1][根拠2] のように番号を分けて書き、根拠番号を作り変えない。
- TaxAnswerは実務向けFAQ、通達は法令解釈上の根拠、質疑応答事例は個別論点の事例回答として区別して扱う。
- 個別判断が必要な場合は、税務署または税理士への確認を促す。
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
        blocks.append(
            f"""[根拠{i}]
corpus_id: {c.get('corpus_id')}
document_type: {c.get('document_type')}
document_id: {c.get('document_id')}
tsutatsu_name: {c.get('tsutatsu_name')}
taxanswer_no: {c.get('taxanswer_no')}
law_basis_date: {c.get('law_basis_date')}
source_url: {c.get('source_url')}
source_file: {c.get('source_file')}
heading_path: {c.get('heading_path')}
parent_id: {c.get('parent_id')}

{c.get('parent_text') or c.get('child_text')}
"""
        )

    context_text = "\n\n".join(blocks)

    return f"""以下の根拠文書だけを使って質問に回答してください。会話履歴は更問の意図理解にだけ使い、根拠文書にない内容は断定しないでください。

# 会話履歴
{history_text or "なし"}

# 質問
{question}

# 根拠文書
{context_text}

# 回答条件
- まず結論を短く述べる。
- 重要な主張、条件、例外、注意点の文末には、使った根拠文書の番号を [根拠1] の形式で付ける。
- 根拠欄を作る場合も、各項目の先頭に [根拠1] の形式を付けてから、タックスアンサーNo、通達名、文書ID、タイトル、URL、法令基準日を書く。
- 根拠となるタックスアンサーNo、通達名、文書ID、タイトル、URL、法令基準日があれば示す。
- 税額や申告要否は、根拠に条件が書かれている場合だけ条件付きで説明する。
- 個別事情の確認が必要な場合は、税務署または税理士への確認を促す。
- 根拠が不足する場合は「このRAGコーパスだけでは確認できません」と書く。
"""
