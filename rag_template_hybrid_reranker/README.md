# RAG 強化テンプレート（ハイブリッド検索 + reranker版）

このテンプレートは、標準版の親子RAG / SearchTag / 複数文書対応に、BM25 + Denseのハイブリッド検索とreranker接続を加えた強化構成です。
会社導入の標準ベースは `../rag_template_current/` とし、検索精度を追加で詰める案件ではこのディレクトリを使います。

## 基本思想

- 検索は **子チャンク**
- 回答生成は **親チャンク**
- SearchTag は **子チャンク単位**
- 検索は **Dense + BM25 のハイブリッド**
- 任意で **reranker** を使って候補を再順位付け
- UI は **チャット形式**
- 更問用に会話履歴をAPIへ渡す
- `/ask` の実行ログはローカルSQLiteへ保存
- 複数文書は `corpus_id` で管理
- UI 側で検索対象文書群を選択
- 回答根拠として、ヒットした子チャンクと復元した親チャンクを表示
- 様式ファイルは全文Markdown化せず、まず `form_catalog.csv` で様式名・ファイルパス・説明を管理
- 本文中に出てくる正式な様式名から、様式ファイルへのリンク候補を提示

## ディレクトリ構成

```text
rag_template_current/
  api/
    main.py
    schemas.py
    rag_engine.py
    llm_client.py
    retriever.py
    prompt.py
  app/
    streamlit_app.py
    eval_ui.py
  scripts/
    01_make_chunks.py
    02_make_search_tags.py
    03_build_index.py
    04_make_form_catalog.py
    05_batch_eval.py
  config/
    corpus_settings.json
    settings.example.json
  data/
    markdown/
      sample_corpus/
        sample.md
    forms/
      form_catalog.csv
  chunks/
  indexes/
    chroma/
  logs/
  eval/
  docs/
    evaluation_csv_format.md
    markdown_conversion_prompt.txt
    operation_guide.md
  requirements.txt
  .env.example
```

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

`.env` と `config/settings.json` はローカル環境用です。Git管理対象にしない前提です。

## 1. Markdown配置

`data/markdown/<corpus_id>/` 配下に Markdown を配置します。

例:

```text
data/markdown/travel_rules/
  001_overview.md
  002_allowance.md
```

`config/corpus_settings.json` に文書群の設定を追加します。

## 2. 親子チャンク作成

```bash
python scripts/01_make_chunks.py
```

出力:

```text
chunks/parent_chunks.jsonl
chunks/child_chunks.jsonl
```

## 3. SearchTag 作成

```bash
python scripts/02_make_search_tags.py
```

出力:

```text
chunks/child_chunks_with_tags.jsonl
```

LLMが未起動の場合はこの工程を省略できます。その場合、インデックス作成と検索は `child_chunks.jsonl` を使います。

## 4. インデックス作成

```bash
python scripts/03_build_index.py
```

出力:

```text
indexes/chroma/
```

Embedding APIが全バッチで成功してから既存のChroma collectionを置き換えるため、Embedding失敗時に既存インデックスを先に削除しません。

## 5. API起動

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 6. Streamlit UI起動

```bash
streamlit run app/streamlit_app.py
```

UIはチャット形式です。同じ会話内の過去メッセージをAPIへ渡すため、「それはいつまで？」のような更問にも対応します。

## 7. 評価UI起動

```bash
streamlit run app/eval_ui.py
```

## 注意

- `.env` や `settings.json` には環境固有のURLやAPIキーを入れるため、このテンプレートでは `.env.example` と `settings.example.json` のみ同梱しています。
- API/LLM/Embedding/検索重み/reranker/CORSは `config/settings.json` と環境変数から読み込まれます。環境変数がある場合は環境変数を優先します。
- `RAG_LOG_ENABLED=true` の場合、`/ask` の質問、回答、根拠、会話履歴、session_idを `logs/rag_chat_logs.sqlite` に保存します。保存先は `RAG_LOG_SQLITE_PATH` または `config/settings.json` で変更できます。
- 直近ログは `GET /logs/recent?limit=50` でも確認できます。
- `corpus_ids` を未指定にした場合は全件検索、空リストにした場合は検索対象なしとして扱います。Streamlit UIでは検索対象を1つ以上選択してください。
- rerankerは `POST {RERANKER_BASE_URL}/rerank` に `query`, `documents`, `top_k` を送るOpenAI互換外の任意HTTPサービスとして扱います。未起動や失敗時はハイブリッド検索結果のまま回答します。
- 様式ファイルは `data/forms/` に実体ファイルを置いて `python scripts/04_make_form_catalog.py` で `form_catalog.csv` を生成してください。RAG結果の根拠本文に様式名が含まれる場合、APIレスポンスとUIに関連様式として表示されます。
- vLLM の OpenAI互換APIを利用する前提です。
- 完全オフライン環境では、pip wheelhouse やモデルファイルを別途準備してください。
