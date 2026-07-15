# 運用ガイド

## 1. 新しい文書を追加する

1. `data/markdown/<corpus_id>/` を作成
2. Markdownファイルを配置
3. `config/corpus_settings.json` に追加
4. `python scripts/01_make_chunks.py`
5. `python scripts/02_make_search_tags.py`
6. Qdrantを起動
7. `python scripts/03_build_index.py`
8. 評価して問題なければ `python scripts/06_manage_releases.py --activate <release_id>`
9. APIとUIで確認

SearchTag生成は任意工程です。完全オフラインでLLMを起動していない場合は省略できます。

## 2. Qdrant運用

このテンプレートは `VECTOR_DB_PROVIDER=qdrant` をデフォルトにしています。

最小検証:

```bash
docker run -p 6333:6333 -p 6334:6334 -v %cd%\indexes\qdrant:/qdrant/storage qdrant/qdrant:latest
```

本番では以下を必ず決めてください。

- 永続ディスクの場所
- snapshot / backup / restore 手順
- Qdrant API key
- TLS終端
- Qdrantへの接続元制限
- Qdrantの監視とログ保存
- 再インデックス時の切替手順

## 3. Vector DB切替

Qdrant:

```env
VECTOR_DB_PROVIDER=qdrant
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION_NAME=rag_children
```

Chroma:

```env
VECTOR_DB_PROVIDER=chroma
VECTOR_DB_CHROMA_PATH=indexes/chroma
```

どちらを選んでも、API側は `HybridRetriever` から同じ形で呼び出します。

## 4. Index Release Manager

本番用の検索collectionを直接作り直すと、build失敗時に検索不能になります。このテンプレートでは、`03_build_index.py` が毎回versioned collectionを作り、`indexes/release_manifest.json` にrelease状態を保存します。

状態:

- `building`: build中
- `staging`: build成功、まだ本番未反映
- `active`: APIが検索に使うrelease
- `archived`: 旧release
- `failed`: build失敗

基本コマンド:

```bash
python scripts/03_build_index.py --index-version v20260610_001
python scripts/06_manage_releases.py --list
python scripts/06_manage_releases.py --activate default_v20260610_001
python scripts/06_manage_releases.py --active
```

検証環境で即active化する場合:

```bash
python scripts/03_build_index.py --index-version v20260610_001 --activate
```

APIから確認・切替する場合:

```text
GET  /admin/releases
POST /admin/releases/{release_id}/activate
```

`release_manifest.json` は環境ごとの状態なのでGit管理しません。APIはactive releaseの `collection_name`、`corpus_version`、`index_version` を使って検索し、承認済みQAキャッシュも同じversionに揃えます。

## 5. 承認済みQAキャッシュ

LLM回答は自動登録しません。再利用できる回答は管理者が登録したQAだけです。

標準フロー:

1. ユーザーがチャットで質問
2. 承認済みQAに高類似度一致すれば、その回答を返す
3. 一致しなければ通常RAGで回答
4. ユーザーがハルシネーション疑いを報告
5. 管理者が報告を確認
6. 管理者が根拠検索でチャンクを選択
7. 正しいQAと根拠チャンクを承認登録

`corpus_version` と `index_version` が一致する承認済みQAだけが検索対象になります。

## 6. 管理者アプリ

```bash
streamlit run app/admin_app.py
```

主な機能:

- ユーザー報告の一覧表示
- 報告内容の確認
- 正しいQAの登録
- 根拠チャンク検索
- 選択した根拠チャンクのQA添付
- 承認済みQA一覧
- チャットログ閲覧
- 子チャンク、参照ファイル、頻出質問のランキング表示
- 質問数推移とランキングのグラフ表示
- LLMによる直近ログ傾向レポート生成
- No Hit / Low Confidence の検知
- 承認済みQAの編集、無効化、再有効化
- 文書、チャンク、SearchTag、インデックス状態の確認
- SearchTagの検索、編集、検索器再読み込み

認証は入れていません。実運用では管理者アプリを別URLにし、VPN、リバースプロキシ、SSOなどの外側で保護してください。

## 7. ログ

- チャットログ: `logs/rag_chat_logs.sqlite`
- 承認済みQAと報告キュー: `logs/answer_cache.sqlite`

管理者アプリの「ログ閲覧」では、質問、回答、session_id、根拠チャンクを確認できます。

管理者アプリの「ログ分析」では、総質問数、セッション数、24時間/7日間の質問数、よく回答根拠に使われた子チャンク、参照ファイル、頻出質問を確認できます。よくヒットするチャンクや文書は、RAG改善だけでなく、研修やマニュアル整備の優先候補として扱えます。

No Hit は根拠チャンクが0件だった質問です。Low Confidence は回答時の最大検索スコアが管理画面のしきい値を下回った質問です。これらは、文書不足、チャンク粒度不一致、SearchTag不足、ユーザー教育テーマの候補として優先確認してください。

「直近ログの傾向レポート生成」ボタンを押すと、集計済みKPI、日次推移、ランキングをローカルLLMに渡し、RAG改善、追加すべき文書、研修やマニュアル化すべき業務領域、次のアクションを自然言語で出力します。個別ログ全文ではなく集計データを渡します。

APIで確認する場合:

```bash
curl http://127.0.0.1:8000/admin/logs/recent?limit=50
curl "http://127.0.0.1:8000/admin/logs/dashboard?sample_limit=1000&top_n=20&days=7"
curl -X POST http://127.0.0.1:8000/admin/logs/report -H "Content-Type: application/json" -d "{\"days\":7,\"sample_limit\":1000,\"top_n\":20}"
```

## 承認済みQAの編集・無効化

管理者アプリの「承認済みQA」では、登録済みQAを選択して質問、回答、根拠、version、メモ、statusを更新できます。

- `approved`: 承認済みQAキャッシュの検索対象
- `disabled`: 検索対象外。削除ではないため、後から再有効化できます。

質問を編集すると、更新時にEmbeddingも作り直します。根拠は既存JSONを編集するか、根拠再検索で選んだチャンクに差し替えられます。

## 文書/Index管理

管理者アプリの「文書/Index」では、以下を確認できます。

- `corpus_settings.json` のCorpus一覧とMarkdown件数
- 親チャンク、子チャンク、SearchTag付き子チャンクの件数
- `chunk_report.csv` の警告件数とCorpus別集計
- Chroma/Qdrant/logsなどの保存領域サイズ
- Vector DBのcollection名、件数、version情報

文書追加後は、Markdown件数、子チャンク件数、SearchTag付き件数、Vector DB件数が想定どおり増えているかを確認してください。

複数APIサーバー構成にする場合は、SQLiteではなくPostgreSQLなどへ移行してください。

## SearchTag編集
管理者アプリの「SearchTag編集」では、子チャンクを検索してSearchTagを直接編集できます。

- 編集対象は `chunks/child_chunks_with_tags.jsonl` です。
- `child_chunks_with_tags.jsonl` がまだ無い場合は、`child_chunks.jsonl` を元に編集済みファイルを作成します。
- 保存後に検索器を再読み込みすると、BM25とSearchTagによる候補検索には即時反映されます。
- Dense/Vector検索にも完全に反映するには、`python scripts/03_build_index.py` を再実行してください。
- SearchTagは同義語、略語、口語表現、社内用語、似た語句の区別に使うと効果が出やすいです。

## 8. 精度改善の優先順位

1. Markdown構造の修正
2. 親チャンク粒度の調整
3. 子チャンク粒度の調整
4. SearchTagの改善
5. BM25 / Dense / reranker の重み調整
6. Qdrant payload index / filter設計
7. 承認済みQAキャッシュのしきい値調整
8. プロンプト修正
9. LLMモデル変更
