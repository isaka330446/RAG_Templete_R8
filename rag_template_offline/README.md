# Offline RAG Template

オフライン運用を前提にした、再利用可能なRAGアプリテンプレートです。

含まれるもの:

- FastAPI RAG API
- FastHTMLユーザー向けチャットUI
- FastHTML管理者アプリ
- FastHTML開発者評価UI
- SQLiteログ
- 承認済みQAキャッシュとalias検索
- CSV/JSONL評価スクリプト
- 最小サンプルMarkdown文書

## 設定

実運用の設定は `config/settings.json` に集約します。

- URL、host、port、LLM、Embedding、Reranker、SQLiteパスは `config/settings.json` を変更してください。
- `.env` は原則 `SETTINGS_PATH=config/settings.json` など最小限にします。
- SQLiteは必須機能です。標準の `journal_mode` は `DELETE` です。これはDBを削除する設定ではなく、rollback journal方式です。

## 文書投入

1. `config/corpus_settings.json` にcorpusを定義します。
2. Markdown文書を `data/markdown/<corpus_id>/` に配置します。
3. 正式タイトルや文書IDはファイル名ではなく、Markdown先頭のYAMLメタデータに保存します。

例:

```markdown
---
title: "文書タイトル"
document_id: "doc_001"
document_type: "manual"
version_date: "2026-01-01"
---

# 文書タイトル
本文...
```

## 実行順

### 1. 文書処理

```powershell
python scripts/01_make_chunks.py
python scripts/02_make_search_tags.py
python scripts/03_build_index.py
```

### 2. 任意の補助データ

```powershell
python scripts/04_make_form_catalog.py
python scripts/08_import_approved_qa_seed.py
python scripts/09_backfill_approved_qa_aliases.py --ensure-original
```

### 3. アプリ起動

```powershell
python scripts/run_api.py
python -m app_fasthtml_modern.main
python -m app_fasthtml_admin_common.main
python -m app_dev_eval.main
```

### 4. 評価

```powershell
python scripts/10_batch_inference.py --input eval/golden/rag_golden_dataset.jsonl --show-debug
python scripts/13_eval_rules.py --input eval/runs/<run>/predictions.jsonl
python scripts/14_check_eval_env.py
python scripts/11_eval_ragas.py --input eval/runs/<run>/predictions.jsonl
python scripts/12_eval_deepeval.py --input eval/runs/<run>/predictions.jsonl --local-openai
```

`11_eval_ragas.py` と `12_eval_deepeval.py` は任意依存です。必要な場合だけ `requirements_eval.txt` を入れてください。
RAGAS import が `langchain_community` で失敗する場合は `python -m pip install -U -r requirements_eval.txt` を実行し、`python scripts/14_check_eval_env.py` で確認してください。

## オンライン環境とオフライン環境

ソースコードは分けません。

- オンライン環境: 文書取得、変換、検証、評価に使います。
- オフライン環境: `data/markdown`、`chunks`、`indexes`、`logs`、`config/settings.json` を配置して運用します。

オンライン取得用スクリプトはこのテンプレートには含めません。必要なプロジェクト側で `scripts/00_*` として追加してください。

## 動作確認

```powershell
python -m compileall -q api app_dev_eval app_fasthtml_admin_common app_fasthtml_modern scripts
python scripts/check_no_hardcoded_urls.py
python scripts/check_docs_no_stale_urls.py
python scripts/10_batch_inference.py --input eval/golden/rag_golden_dataset.jsonl --limit 2 --dry-run
```

## 生成物

以下はテンプレートに同梱しません。

- SQLite DB
- Chroma index
- 生成済みchunk
- 評価run出力
- 実運用文書

必要な生成物は各環境で作成してください。
