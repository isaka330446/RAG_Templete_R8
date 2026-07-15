# 内部文書取り込みガイド

このテンプレートでは、取り込み対象をMarkdownに変換してからRAG indexを作成します。

## 1. corpusを決める

`config/corpus_settings.json` にcorpusを追加します。

```json
{
  "corpus_id": "company_internal",
  "display_name": "社内規程",
  "description": "社内規程と運用マニュアル",
  "markdown_dir": "data/markdown/company_internal",
  "priority": 10,
  "enabled": true
}
```

## 2. Markdownを配置する

Markdownファイル名は短いIDにしてください。正式タイトルはYAMLメタデータに保存します。

```markdown
---
title: "正式な文書タイトル"
document_id: "company_rule_001"
document_type: "rule"
source_site: "internal"
version_date: "2026-01-01"
---

# 正式な文書タイトル

本文...
```

推奨metadata:

- `title`
- `document_id`
- `document_type`
- `source_site`
- `source_url`
- `version_date`

`source_url` は社内ファイルパスや文書管理システムIDでも構いません。

## 3. chunkとindexを作る

```powershell
python scripts/01_make_chunks.py
python scripts/02_make_search_tags.py
python scripts/03_build_index.py
```

SearchTagは補助機能です。不要な場合は `config/settings.json` の `search_tags.enabled_in_retrieval` を `false` にしてください。

## 4. 検索対象を絞る

APIやUIでは `corpus_ids` に対象corpusを指定できます。
複数の部署や文書セットを扱う場合は、corpusを分けて管理してください。

## 5. 承認済みQAを育てる

安定して使う質問は管理者アプリから承認済みQAとして登録します。
aliasは検索入口なので、表現揺れを追加できます。
ただし、対象文書や条件が変わるaliasは無効化してください。

## 6. 評価する

評価データはCSVまたはJSONLで管理できます。
JSONLは `eval/golden/rag_golden_dataset.jsonl` を参考にしてください。

```powershell
python scripts/10_batch_inference.py --input eval/golden/rag_golden_dataset.jsonl --show-debug
python scripts/13_eval_rules.py --input eval/runs/<run>/predictions.jsonl
```
