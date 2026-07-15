# Offline RAG Template 運用ガイド

## 基本方針

- 実運用設定は `config/settings.json` に集約します。
- URL、host、port、SQLite、LLM、Embedding、Rerankerの接続先はコードに直書きしません。
- SQLiteは本番運用の必須機能です。標準の `journal_mode` は `DELETE` です。
- `DELETE` はDB削除ではなく、rollback journal方式です。
- ユーザー向けフロントは `app_fasthtml_modern.main` の1つです。

## 初期セットアップ

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item config\settings.example.json config\settings.json
```

`config/settings.json` を環境に合わせて編集します。

## 文書投入からindex作成まで

1. `config/corpus_settings.json` にcorpusを追加します。
2. `data/markdown/<corpus_id>/` にMarkdownを配置します。
3. 次の順に実行します。

```powershell
python scripts/01_make_chunks.py
python scripts/02_make_search_tags.py
python scripts/03_build_index.py
```

## アプリ起動

```powershell
python scripts/run_api.py
python -m app_fasthtml_modern.main
python -m app_fasthtml_admin_common.main
```

## 承認済みQA

承認済みQAは、よく使う質問に対して安定した回答を返すためのキャッシュです。

- `approved_qa`: 回答本体
- `approved_qa_aliases`: 検索入口

別名質問を増やすとヒットしやすくなりますが、回答範囲が広がるaliasは無効化してください。

## 評価

```powershell
python scripts/10_batch_inference.py --input eval/golden/rag_golden_dataset.jsonl --show-debug
python scripts/13_eval_rules.py --input eval/runs/<run>/predictions.jsonl
```

RAGAS/DeepEvalを使う場合だけ、評価用依存を追加します。

```powershell
python -m pip install -r requirements_eval.txt
python scripts/14_check_eval_env.py
python scripts/11_eval_ragas.py --input eval/runs/<run>/predictions.jsonl
python scripts/12_eval_deepeval.py --input eval/runs/<run>/predictions.jsonl --local-openai
```

RAGAS import が `langchain_community` で失敗する場合は `python -m pip install -U -r requirements_eval.txt` を実行し、`python scripts/14_check_eval_env.py` で確認してください。

## オフライン運用

オフライン環境では、オンライン取得スクリプトは実行しません。
必要な文書、chunk、index、SQLite、設定ファイルを配置して運用します。

ソースコードはオンライン用とオフライン用で分岐させません。
