# RAG Developer Evaluation UI

FastHTMLで作成した開発者向けの評価アプリです。日常運用の管理者アプリとは分け、RAGAS / DeepEval / ルール採点 / 手動採点を検証時だけ使う前提にしています。

## Setup

```bash
pip install -r requirements_fasthtml.txt
pip install -r requirements_eval.txt
```

RAGAS / DeepEvalは任意依存です。バッチ推論、ルール採点、手動採点だけならRAGAS / DeepEvalの実行環境がなくても画面自体は起動できます。

## Run

```bash
python -m app_dev_eval.main
```

起動host/portは `config/settings.json` の `urls.dev_eval_bind_host` / `urls.dev_eval_bind_port` を使います。

## 機能

- バッチ推論: CSV/JSONL評価セットをRAG API `/ask` に投入し、`predictions.jsonl/csv` を作成します。
- RAGAS実行: `predictions.jsonl` をRAGASで評価します。
- DeepEval実行: `predictions.jsonl` をDeepEvalで評価します。
- ルール採点: `must_include` / `must_not_include` / `no_answer` の簡易採点を実行します。
- 結果サマリ: runごとの `summary.json`、各評価summary、CSV結果を確認します。
- 詳細・手動採点: 質問ごとの回答、根拠、debugを見ながら4基準で手動採点します。

## 手動採点

手動採点は次の4基準です。

- 正確性
- 根拠整合性
- 網羅性
- 回答適切性

各基準は `○=1.0`、`△=0.5`、`×=0.0` で保存します。出力先は対象runディレクトリです。

```text
eval/runs/<run>/manual_scores.jsonl
eval/runs/<run>/manual_scores.csv
eval/runs/<run>/manual_summary.json
```

同じ `run_id + question_id + reviewer` の採点は上書き保存されます。
