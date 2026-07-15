# Operation Guide

## 設定管理

実運用設定は `config/settings.json` に集約します。URL、host、port、LLM/Embedding/Reranker接続先、CORS許可URLは `urls` セクションだけを変更してください。

`.env` は `SETTINGS_PATH=config/settings.json` と非URLの上書きに限定します。URLを `.env` やPythonコードへ書かないでください。

## SQLite

SQLiteは本番必須です。

- `logging.required=true`
- `answer_cache.required=true`
- `journal_mode=DELETE`
- `startup_write_check=true`

`DELETE` はDB削除ではありません。rollback journal方式です。会社Windowsオフライン環境ではWALより安定しやすいため標準にしています。

API起動時には、ログDBと承認済みQA DBの両方でschema作成と書き込みチェックを行います。失敗した場合、APIは起動しません。開発時だけ `required=false` を明示して継続を許可できます。

## 起動手順

1. `config/settings.json` を確認します。
2. LLM、Embedding、必要ならRerankerのローカルAPIを起動します。
3. APIを起動します。

```bash
python scripts/run_api.py
```

4. ユーザー向けFastHTMLフロントを起動します。

```bash
python -m app_fasthtml_modern.main
```

ユーザー向けフロントはこの1つだけです。旧ラッパーは廃止済みです。

## オンライン取得とオフライン運用

オンライン取得用スクリプトは開発・更新用です。オフライン運用PCでは通常実行しません。

オンライン取得:

1. `python scripts/00_fetch_nta_taxanswer.py`
2. `python scripts/00_fetch_nta_tsutatsu.py --clean`
3. `python scripts/00_fetch_nta_pdf_joho.py --clean`
4. `python scripts/00_fetch_nta_shitsugi.py --clean`

オフライン運用で使う主な生成手順:

1. `python scripts/01_make_chunks.py`
2. `python scripts/02_make_taxanswer_search_tags.py`
3. 任意: `python scripts/02_make_search_tags.py`
4. `python scripts/03_build_index.py`

## 承認済みQAとalias

回答本体は `approved_qa`、検索入口は `approved_qa_aliases` です。aliasにヒットしても返す回答は必ず親の `approved_qa.answer` です。

既存運用DBへaliasを補完する場合:

```bash
python scripts/09_backfill_approved_qa_aliases.py --check
python scripts/09_backfill_approved_qa_aliases.py --migrate-schema --backup
python scripts/09_backfill_approved_qa_aliases.py --ensure-original
python scripts/09_backfill_approved_qa_aliases.py --apply --backup --ensure-original
python scripts/09_backfill_approved_qa_aliases.py --generate-llm --limit 10
python scripts/09_backfill_approved_qa_aliases.py --apply --backup --generate-llm --only-without-llm-aliases --limit 50 --max-aliases-per-qa 8
```

運用DBに書き込む前に必ずバックアップしてください。

## 管理者アプリ

管理者アプリは日常運用と改善作業用です。

- ダッシュボード
- 改善キュー
- 報告対応
- 承認QA
- ログ探索
- 文書/評価

SearchTag編集は補助機能です。主導線はQA alias追加、承認QA化、文書不足確認です。

## 評価

評価データはCSV/JSONL両対応です。JSONLは `eval/golden/rag_golden_dataset.jsonl` の形式を使えます。

```bash
python scripts/10_batch_inference.py --input eval/golden/rag_golden_dataset.jsonl --show-debug
python scripts/14_check_eval_env.py
python scripts/11_eval_ragas.py --input eval/runs/<run>/predictions.jsonl
python scripts/12_eval_deepeval.py --input eval/runs/<run>/predictions.jsonl --local-openai
python scripts/13_eval_rules.py --input eval/runs/<run>/predictions.jsonl
```

RAGAS/DeepEvalは任意依存です。本番 `requirements.txt` ではなく `requirements_eval.txt` を使ってください。
If RAGAS fails while importing `langchain_community`, refresh the optional evaluation environment with `python -m pip install -U -r requirements_eval.txt`, then run `python scripts/14_check_eval_env.py`.

## 開発者向け評価UI

開発者向け評価UIはFastHTML版です。Streamlit版は廃止しています。

```bash
python -m app_dev_eval.main
```

このUIでは、バッチ推論、RAGAS、DeepEval、ルール採点、結果サマリ、詳細確認、手動採点を扱えます。手動採点は4基準です。

- 正確性
- 根拠整合性
- 網羅性
- 回答適切性

保存先は `eval/runs/<run>/manual_scores.jsonl`、`manual_scores.csv`、`manual_summary.json` です。

## チェック

```bash
python scripts/check_no_hardcoded_urls.py
python -m compileall -q api app_dev_eval app_fasthtml_admin_common app_fasthtml_modern scripts
```
