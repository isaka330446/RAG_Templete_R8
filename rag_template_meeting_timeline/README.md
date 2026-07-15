# RAGテンプレート（承認済みQAキャッシュ版）

`rag_template_hybrid_reranker/` をベースに、承認済みQAキャッシュを追加した3つ目のテンプレートです。親子RAG、SearchTag、ハイブリッド検索、任意のreranker、更問対応チャット、SQLiteログを維持しつつ、管理者が承認したQAだけを再利用します。

## 基本方針

- 回答済みQAの自動登録はしません。
- LLM回答はハルシネーションを含む可能性があるため、再利用対象は管理者承認済みQAだけです。
- `/ask` は最初に承認済みQAを意味検索し、高類似度で一致した場合だけLLMを使わず返答します。
- 承認済みQAに一致しない場合は、通常どおりハイブリッド検索 + reranker + LLMで回答します。
- 承認済みQAは `corpus_version` と `index_version` ごとに分けます。文書やインデックスを更新した後、古いQAは自動では流用しません。
- 管理者アプリはユーザー用チャットアプリと分けています。権限制御を後から入れやすくするためです。

## 回答フロー

1. ユーザーがチャットで質問する。
2. 質問と会話履歴から検索クエリを作る。
3. 承認済みQAキャッシュをEmbedding類似度で検索する。
4. 類似度が `ANSWER_CACHE_HIGH_SIMILARITY_THRESHOLD` 以上なら、承認済み回答と登録済み根拠を返す。
5. ヒットしなければ、従来どおりRAG検索してLLMで回答する。
6. ユーザーがハルシネーション疑いを報告した場合、管理者アプリの報告対応キューに入る。
7. 管理者が正しいQAを作成し、根拠検索で選んだチャンクを添付して承認QAとして登録する。

## セットアップ

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
copy config\settings.example.json config\settings.json
```

Linux / WSL の場合:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
cp config/settings.example.json config/settings.json
```

## 文書登録

Markdownを `data/markdown/<corpus_id>/` に配置し、`config/corpus_settings.json` に文書群を登録します。

```bash
python scripts/01_make_chunks.py
python scripts/02_make_search_tags.py
python scripts/03_build_index.py
```

SearchTag生成にはLLMを使います。完全オフラインでLLMを起動していない場合、この工程は省略できます。その場合、インデックス作成と検索は `child_chunks.jsonl` を使います。

## 起動

API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

ユーザー用チャット:

```bash
streamlit run app/streamlit_app.py
```

管理者アプリ:

```bash
streamlit run app/admin_app.py
```

評価UI:

```bash
streamlit run app/eval_ui.py
```

## 主要設定

`.env` または `config/settings.json` で設定します。

- `ANSWER_CACHE_ENABLED=true`
- `ANSWER_CACHE_SQLITE_PATH=logs/answer_cache.sqlite`
- `ANSWER_CACHE_HIGH_SIMILARITY_THRESHOLD=0.88`
- `ANSWER_CACHE_CORPUS_VERSION=default`
- `ANSWER_CACHE_INDEX_VERSION=default`
- `RAG_LOG_SQLITE_PATH=logs/rag_chat_logs.sqlite`
- `RERANKER_ENABLED=false`

しきい値は高めを推奨します。会社導入時は誤ヒットのリスクを避けるため、最初は `0.88` 以上から検証してください。

## 管理者ワークフロー

1. ユーザーがチャット画面から「ハルシネーション疑い」を報告する。
2. 管理者アプリの「報告対応」で報告内容を確認する。
3. 正しい回答を書く。
4. 根拠検索で関連チャンクを探す。
5. 根拠として使うチャンクを選ぶ。
6. 承認QAとして登録する。

登録されたQAは `approved_qa` テーブルに保存され、次回以降の類似質問でだけ再利用されます。

## API

- `POST /ask`
- `GET /logs/recent`
- `POST /feedback/hallucination`
- `GET /admin/reports`
- `PATCH /admin/reports/{report_id}/status`
- `POST /admin/evidence/search`
- `POST /admin/qa-cache`
- `GET /admin/qa-cache`

現在の管理者APIには認証を入れていません。実運用では管理者アプリを社内ネットワークや認証プロキシの内側に置く想定です。

## ログ

- チャットログ: `logs/rag_chat_logs.sqlite`
- 承認済みQAと報告キュー: `logs/answer_cache.sqlite`

SQLiteはローカル運用を前提にしています。複数サーバー運用にする場合は、DBをPostgreSQLなどに置き換えてください。

## 会議資料・議事録・時系列イベントRAG

このテンプレートでは、承認済みQAキャッシュ版の構成を保ったまま、PowerPoint会議資料のMarkdown化結果と議事録Markdownを取り込めます。

- 配置先: `data/markdown/meeting_documents/`
- PPTX資料Markdown: frontmatter の `document_type: meeting_slide_deck`
- 議事録Markdown: frontmatter の `document_type: meeting_minutes`
- 紐づけキー: `meeting_id`
- 会議日: `meeting_date` は必須
- MeetingEvent出力: `data/meeting_events/meeting_events.jsonl`

取り込み手順:

```bash
python scripts/01_make_chunks.py
python scripts/02_make_search_tags.py
python scripts/03_build_index.py
```

時系列イベントだけ確認する場合:

```bash
python scripts/07_query_meeting_events.py --topic 生成AI --event-type decision
```

APIの `/ask` では `answer_mode` に `auto`、`rag`、`timeline` を指定できます。時系列回答は `event_date`、`meeting_date` の昇順でアプリ側がソートし、根拠として資料名、スライド番号、議事録セクションを返します。

詳細は `docs/meeting_timeline_rag.md` を参照してください。
