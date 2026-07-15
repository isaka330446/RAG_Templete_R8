# 評価用CSVフォーマット

RAG評価では、質問・期待回答・期待根拠をCSVで管理します。

## 推奨カラム

| カラム名 | 必須 | 内容 |
|---|---:|---|
| question_id | 任意 | 質問ID |
| question | 必須 | RAGに投げる質問 |
| expected_answer | 必須 | 期待される回答 |
| expected_source | 任意 | 期待される根拠ファイル・章・条 |
| corpus_ids | 任意 | 検索対象文書群。複数指定は `|` 区切り |
| category | 任意 | 分類 |
| difficulty | 任意 | 難易度 |
| memo | 任意 | 補足 |

## 例

```csv
question_id,question,expected_answer,expected_source,corpus_ids,category,difficulty,memo
Q001,サンプル申請書はいつ提出しますか？,申請者は必要事項を記載したサンプル申請書を提出する。,sample.md 第3条,sample_corpus,手続,低,
```

## 評価の観点

- ○: 正答。根拠も妥当。
- △: 一部正答。根拠不足、説明不足、表現の曖昧さあり。
- ×: 誤答。根拠違い、幻覚、質問未回答。

## 自動評価の前提

RAGASやLangSmithに渡す場合も、このCSVを起点にできます。

最低限必要な項目:

- question
- expected_answer
- retrieved_contexts
- generated_answer

本テンプレートの `scripts/05_batch_eval.py` は、まずRAG回答と根拠をCSVに出します。
その後、○△×の人手評価、またはRAGAS等に連携してください。
