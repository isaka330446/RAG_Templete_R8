# TaxAnswer Inheritance/Gift RAG

国税庁タックスアンサー、通達、質疑応答事例、相続税・贈与税関連PDFを対象にした、オフライン運用前提のRAGアプリです。

## 基本方針

- 実運用設定は `config/settings.json` に集約します。
- URL、host、port、LLM/Embedding/Reranker接続先、CORS許可URLは `config/settings.json` の `urls` だけを変更します。
- `.env` は原則 `SETTINGS_PATH=config/settings.json` と非URLのモデル名/APIキー/DBフラグだけに使います。
- SQLiteは本番必須です。`logging.required=true` と `answer_cache.required=true` が標準です。
- SQLite標準の `journal_mode` は `DELETE` です。これはDB削除ではなく、rollback journal方式です。
- SQLite初期化、schema作成、起動時書き込みチェックに失敗した場合、APIは起動しません。
- ユーザー向けFastHTMLフロントは `app_fasthtml_modern/main.py` だけです。
- 根拠表示は大型ポップアップ方式だけです。旧ラッパーとUIモード切替は廃止しました。
- 管理者アプリと開発者評価UIは、一般ユーザー向けフロントとは別用途です。

## 初期設定

1. `config/settings.example.json` を参考に `config/settings.json` を作成します。
2. `config/settings.json` の `urls` を運用PCのローカルAPI構成に合わせます。
3. `logging` と `answer_cache` のSQLite保存先を確認します。
4. 会社Windows環境では `journal_mode: "DELETE"` のまま使うことを推奨します。

URLを変える場合、Pythonコードや `.env.example` は変更しません。`config/settings.json` だけ変更してください。

## 起動

API:

```bash
python scripts/run_api.py
```

ユーザー向けFastHTMLフロント:

```bash
python -m app_fasthtml_modern.main
```

管理者アプリ:

```bash
python -m app_fasthtml_admin_common.main
```

管理者アプリのhost/portも `config/settings.json` の `urls.admin_bind_host` / `urls.admin_bind_port` を正としてください。

## スクリプト実行順

オンライン取得系です。オフライン運用PCでは通常実行しません。

1. `python scripts/00_fetch_nta_taxanswer.py`
2. `python scripts/00_fetch_nta_tsutatsu.py --clean`
3. `python scripts/00_fetch_nta_pdf_joho.py --clean`
4. `python scripts/00_fetch_nta_shitsugi.py --clean`

ナレッジ生成系です。Markdownが揃った状態で実行します。

1. `python scripts/01_make_chunks.py`
2. `python scripts/02_make_taxanswer_search_tags.py`
3. 任意: `python scripts/02_make_search_tags.py`
4. `python scripts/03_build_index.py`
5. 任意: `python scripts/04_make_form_catalog.py`

承認済みQA投入・alias補完です。運用DBへ書く前に必ずdry-runとバックアップを行います。

1. `python scripts/07_make_approved_qa_seed.py`
2. `python scripts/08_import_approved_qa_seed.py`
3. `python scripts/08_import_approved_qa_seed.py --apply --approved-by admin`
4. `python scripts/09_backfill_approved_qa_aliases.py --check`
5. `python scripts/09_backfill_approved_qa_aliases.py --migrate-schema --backup`
6. `python scripts/09_backfill_approved_qa_aliases.py --ensure-original`
7. `python scripts/09_backfill_approved_qa_aliases.py --apply --backup --ensure-original`
8. 任意: `python scripts/09_backfill_approved_qa_aliases.py --generate-llm --limit 10`
9. 任意: `python scripts/09_backfill_approved_qa_aliases.py --apply --backup --generate-llm --only-without-llm-aliases --limit 50 --max-aliases-per-qa 8`

評価系です。開発・検証用で、通常運用手順には含めません。

1. `python scripts/10_batch_inference.py --input eval/golden/rag_golden_dataset.jsonl --show-debug`
2. `python scripts/14_check_eval_env.py`
3. `python scripts/11_eval_ragas.py --input eval/runs/<run>/predictions.jsonl`
4. `python scripts/12_eval_deepeval.py --input eval/runs/<run>/predictions.jsonl --local-openai`
5. `python scripts/13_eval_rules.py --input eval/runs/<run>/predictions.jsonl`

RAGAS import failures such as missing `langchain_community` usually mean the optional evaluation environment is incomplete. Refresh it with `python -m pip install -U -r requirements_eval.txt`, then run `python scripts/14_check_eval_env.py`.

## SQLite運用

- `logs/rag_chat_logs.sqlite` はチャットログDBです。
- `logs/answer_cache.sqlite` は承認済みQAキャッシュ本体です。削除・初期化しないでください。
- `journal_mode=DELETE` はDB削除ではありません。
- `startup_write_check=true` の場合、API起動時に専用チェックテーブルへINSERT/DELETEを行い、実際に書き込み可能か確認します。
- 本番標準ではSQLiteが使えない場合、APIは起動しません。
- 開発用途でのみ `required=false` にできます。この場合はDB異常時に機能無効のまま継続できます。

状態確認:

```bash
python - <<'PY'
from api.main import engine
print(engine.log_store.status())
print(engine.answer_cache.status())
PY
```

## URL直書きチェック

コードにURL、host、portを直書きしていないか確認します。

```bash
python scripts/check_no_hardcoded_urls.py
```

`config/settings.json` と `config/settings.example.json` はURL定義の正規置き場なのでチェック対象外です。

## Markdownファイル名

`data/markdown/**/*.md` はZIP展開事故を避けるため、短いIDベースのファイル名に統一します。
正式タイトル、出典URL、文書ID、元ファイル名はMarkdown先頭のYAMLメタデータへ保存します。
長い日本語タイトルをファイル名へ詰め込まないでください。

既存Markdownを短縮名へ揃える場合:

```bash
python scripts/00_normalize_markdown_filenames.py
python scripts/00_normalize_markdown_filenames.py --apply
```

`--apply` 実行時はMarkdownのリネームに加え、`chunks/*.jsonl` と `chunks/chunk_report.csv` の `source_file` 参照も更新します。

## 開発者向け評価UI

開発者向け評価UIはFastHTML版に統一しました。Streamlit版は廃止しています。

```bash
python -m app_dev_eval.main
```

このUIでは、バッチ推論、RAGAS、DeepEval、ルール採点、結果サマリ、質問単位の詳細確認、4基準の手動採点を扱えます。起動host/portは `config/settings.json` の `urls.dev_eval_bind_host` / `urls.dev_eval_bind_port` で変更します。

手動採点の基準は「正確性」「根拠整合性」「網羅性」「回答適切性」です。採点結果は `eval/runs/<run>/manual_scores.jsonl`、`manual_scores.csv`、`manual_summary.json` に保存します。

クラウド生成AIへ `predictions.jsonl` を貼り付けて評価する場合は、必要項目だけに絞ったJSONLを10問単位に分割できます。

```bash
python scripts/15_prepare_llm_eval_batches.py --input eval/runs/<run>/predictions.jsonl
```

出力先は既定で `eval/runs/<run>/llm_eval_batches/` です。`batch_001.jsonl`、`batch_002.jsonl` のように、評価データだけを分割して出力します。評価プロンプト本文はこのプロジェクトでは管理しません。根拠本文は長くなりやすいため、既定では評価に必要な範囲に短縮します。全文を渡す場合は `--no-truncate` を付けてください。

## Reranker evidence filtering

通常のUIは `/ask` を使うため、QAキャッシュ未ヒット時は `config/settings.json` の検索設定に従います。`reranker.enabled=true` にした場合は、リランカー後の根拠をそのまま8件LLMへ渡さず、次の設定で絞り込みます。

- `reranker.top_k`: リランカーへ渡す候補数
- `reranker.min_score`: このrerank score未満の根拠は原則LLMへ渡さない
- `reranker.min_keep`: scoreが低くても最低限残す根拠数
- `reranker.max_keep`: LLMへ渡す最大根拠数
- `reranker.dedupe_parent`: 同じparent chunk由来の根拠を優先的に1件へまとめる

推奨初期値は `top_k=20`, `min_score=0.1`, `min_keep=2`, `max_keep=4`, `dedupe_parent=true` です。リランカーを使わない比較実験では `reranker.enabled=false` のままにしてください。

## SearchTag

SearchTagは削除していません。BM25、dense embedding、form matchingの補助として使われます。

通常の改善導線は、まず以下を優先します。

1. QA alias追加
2. 承認QA化
3. 文書不足確認
4. その後、必要に応じてSearchTag補強

SearchTag利用は `config/settings.json` の `search_tags.enabled_in_retrieval` で切り替えられます。

## ユーザー向けUI

- 初期画面では「今日は何を確認しますか？」の直下に入力フォームを表示します。
- 「新規チャット」ボタンは入力フォーム左側にあります。
- 回答中の根拠リンクと回答下の根拠チップは、大型ポップアップで根拠本文を開きます。
- 旧サイド表示方式は廃止しました。
