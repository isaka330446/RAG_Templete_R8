# RAG Template

会社導入向けのRAGテンプレートを、用途別に段階分けして配置しています。

## テンプレート

- `rag_template_current/`
  - 標準版
  - 親子RAG、SearchTag、Dense検索、チャットUI、更問対応、SQLiteログ、複数文書選択、評価UIを含む
  - まず社内展開する基本テンプレート

- `rag_template_hybrid_reranker/`
  - 強化版
  - 標準版に加えて、BM25 + Dense のハイブリッド検索と reranker 接続を含む
  - 検索精度を追加で詰めたい案件向け

- `rag_template_answer_cache/`
  - 承認済みQAキャッシュ版
  - `rag_template_hybrid_reranker/` をベースに、管理者承認済みQAの再利用を追加
  - 類似質問が承認済みQAに高スコア一致した場合だけ、LLMを使わずに回答する
  - LLM回答の自動登録はしない。ハルシネーション混入を避けるため、再利用対象は管理者が承認したQAのみ
  - ユーザー用チャットアプリとは別に、報告対応、根拠検索、承認QA登録用の管理者アプリを持つ

- `rag_template_meeting_timeline/`
  - 会議資料・議事録・時系列イベントRAG版
  - `rag_template_answer_cache/` をベースに、PowerPoint会議資料Markdown、議事録Markdown、MeetingEvent検索を追加
  - `meeting_id` で資料と議事録を紐づけ、既存の親子チャンク、ハイブリッド検索、reranker、承認済みQAキャッシュに接続
  - 議事録上の決定、懸念、宿題、保留、却下をMeetingEventとして保存し、時系列モードで event_date / meeting_date 昇順に表示

- `rag_template_qdrant_production/`
  - 本番運用寄りQdrant版
  - 承認済みQAキャッシュ版をベースに、ベクトルDBを抽象化してQdrantをデフォルト化
  - `VECTOR_DB_PROVIDER=qdrant/chroma` で切替可能
  - Qdrantのpayload filterで `corpus_id` 絞り込みに対応
  - Release Managerにより、versioned collectionをstagingで作成し、評価後にactive化できる
  - 永続化、バックアップ、監視、権限分離を重視する本番導入向け

## 個別プロジェクト

- `taxanswer_inheritance_gift_rag/`
  - 国税庁タックスアンサー、相続税法基本通達、財産評価基本通達、相続税の措置法通達、相続税・贈与税関係PDF、質疑応答事例を対象にした個別RAG
  - ベースは `rag_template_answer_cache/`
  - 現時点では会社PCでEmbedding APIを動かす前段として、NTAページ取得、Markdown化、親子チャンク化、LLM不要SearchTag生成まで作成済み
  - Chroma index作成は `scripts/03_build_index.py` で後から実行する

## 使い分け

最初は `rag_template_current/` を使います。
キーワード検索やrerankerが必要になったら `rag_template_hybrid_reranker/`、承認済みの定型質問をLLMなしで返したい場合は `rag_template_answer_cache/` を使います。
本番運用でベクトルDBの永続化、バックアップ、監視、フィルタ性能を重視する場合は `rag_template_qdrant_production/` を使います。
