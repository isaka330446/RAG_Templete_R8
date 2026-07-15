# 運用ガイド

## 1. 新しい文書を追加する

1. `data/markdown/<corpus_id>/` を作成
2. Markdownファイルを配置
3. `config/corpus_settings.json` に追加
4. `scripts/01_make_chunks.py`
5. `scripts/02_make_search_tags.py`
6. `scripts/03_build_index.py`
7. UI/APIで確認

`scripts/02_make_search_tags.py` はLLMを使う任意工程です。LLMが使えない環境では省略できます。

Streamlit UIはチャット形式です。サイドバーの「新しい会話」で `session_id` を更新し、会話履歴をクリアします。

## 2. 様式ファイルを追加する

1. `data/forms/` にExcel/Word様式を配置
2. `python scripts/04_make_form_catalog.py`
3. `data/forms/form_catalog.csv` を確認
4. 必要に応じて description を手修正

根拠本文に `form_name` が含まれる場合、APIレスポンスの `sources[].forms` とStreamlit UIに関連様式として表示されます。

## 3. 精度評価

1. `eval/questions.csv` を作成
2. `python scripts/05_batch_eval.py --input eval/questions.csv`
3. `streamlit run app/eval_ui.py`
4. ○△×で評価
5. NG質問は、原因を以下に分類する

- Markdown化ミス
- チャンク分割ミス
- SearchTag不足
- 検索対象文書の選択ミス
- 根拠文書不足
- プロンプト不備
- LLMの回答生成ミス

## 4. チャットログ

`/ask` の実行ごとに、質問、回答、根拠、会話履歴、`session_id` を `logs/rag_chat_logs.sqlite` に保存します。
保存先は `config/settings.json` の `logging.sqlite_path` または `.env` の `RAG_LOG_SQLITE_PATH` で変更できます。
直近ログは `GET /logs/recent?limit=50` で確認できます。

## 5. 改善の優先順位

1. Markdown構造の修正
2. 子チャンク粒度の調整
3. SearchTagの改善
4. BM25 / Dense / Reranker の重み調整
5. プロンプト修正
6. LLMモデル変更

## 6. 強化版の標準

- 親チャンク: 回答生成に十分な章・節・条単位
- 子チャンク: 検索しやすい 800〜1200文字程度
- SearchTag: 子チャンク単位
- UI: チャット形式、更問対応
- 複数文書: corpus_idで選択
- 評価ログ: corpus_idsも保存
- チャットログ: SQLiteに保存
- 設定: `config/settings.json` と環境変数でAPI/LLM/Embedding/検索/reranker/CORSを管理
- 検索対象: `corpus_ids=None` は全件、`corpus_ids=[]` は検索対象なし
- 検索: Dense + BM25 のハイブリッド検索
- reranker: `POST {RERANKER_BASE_URL}/rerank` に接続し、失敗時はハイブリッド検索結果にフォールバック
