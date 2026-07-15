# RAGテンプレート（Qdrant本番運用寄り版）

`rag_template_answer_cache/` をベースに、ベクトルDBを本番運用しやすいQdrantへ切り替えた4つ目のテンプレートです。親子RAG、SearchTag、ハイブリッド検索、任意のreranker、承認済みQAキャッシュ、ユーザー報告、管理者アプリは維持しています。

## 位置づけ

- 標準: `rag_template_current/`
- ハイブリッド検索 + reranker: `rag_template_hybrid_reranker/`
- 承認済みQAキャッシュ: `rag_template_answer_cache/`
- 本番運用寄りQdrant: `rag_template_qdrant_production/`

## 追加した設計

- `api/vector_store.py` でベクトルDBを抽象化
- `VECTOR_DB_PROVIDER=qdrant` をデフォルト化
- `VECTOR_DB_PROVIDER=chroma` に戻せる互換経路を維持
- `api/release_manager.py` で index release を管理
- `03_build_index.py` はactive collectionを直接壊さず、versioned collectionへbuildする
- build直後は `staging`、`--activate` または管理API/CLIで `active` に切り替える
- Qdrantのpayload filterで `corpus_id` 絞り込みに対応
- Qdrant登録時に `corpus_id`、`parent_id`、`source_file` のpayload indexを作成
- Qdrant point IDは `child_id` から決定的UUIDを生成し、再インデックス時のID揺れを避ける

## セットアップ

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
copy config\settings.example.json config\settings.json
```

Linux / WSL の場合:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
cp config/settings.example.json config/settings.json
```

完全オフライン環境では、Qdrant本体、Python wheelhouse、LLM、Embedding、rerankerのモデルを事前に社内配布してください。

## Qdrant起動

開発・検証の最小例:

```bash
docker run -p 6333:6333 -p 6334:6334 -v %cd%\indexes\qdrant:/qdrant/storage qdrant/qdrant:latest
```

PowerShellで別パスにする場合は、`%cd%` の代わりに絶対パスを指定してください。本番では永続ディスク、バックアップ、監視、認証、TLS、冗長化を別途設計してください。

## 主要設定

`.env` または `config/settings.json` で切り替えます。

```env
VECTOR_DB_PROVIDER=qdrant
QDRANT_URL=http://127.0.0.1:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=rag_children
QDRANT_LOCAL_PATH=
QDRANT_PREFER_GRPC=false
```

Chromaへ戻す場合:

```env
VECTOR_DB_PROVIDER=chroma
VECTOR_DB_CHROMA_PATH=indexes/chroma
```

## 文書登録とインデックス

```bash
python scripts/01_make_chunks.py
python scripts/02_make_search_tags.py
python scripts/03_build_index.py
```

`03_build_index.py` は `VECTOR_DB_PROVIDER` を見て、QdrantまたはChromaへ登録します。Qdrantを使う場合は先にQdrantサーバーを起動してください。

デフォルトでは新しいindex releaseは `staging` になります。本番検索に使うには、評価後にactive化します。

```bash
python scripts/06_manage_releases.py --list
python scripts/06_manage_releases.py --activate <release_id>
```

検証環境などでbuild成功後すぐactive化したい場合:

```bash
python scripts/03_build_index.py --index-version v20260610_001 --activate
```

APIから確認・切替する場合:

```text
GET  /admin/releases
POST /admin/releases/{release_id}/activate
```

## 起動

API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

ユーザー用チャット:

```bash
streamlit run app/streamlit_app.py
```

管理者アプリ:

```bash
streamlit run app/admin_app.py
```

## 本番化チェック

- Qdrantの永続ディスクを用意する
- Qdrant snapshot / backup / restore 手順を決める
- API key、TLS、ネットワーク境界を設定する
- `corpus_version` と `index_version` の更新ルールを決める
- `staging` releaseを評価し、合格したreleaseだけ `active` にする
- 管理者アプリをVPN、SSO、認証プロキシ配下に置く
- 監視対象にQdrant、LLM、Embedding、reranker、API、SQLite/DBを入れる
- SQLiteログを複数サーバーで共有する場合はPostgreSQL等へ移行する

## 注意

このテンプレートのBM25は従来どおりアプリ内で計算します。Qdrant native sparse vector / RRF への完全移行は次段階の拡張候補です。まずはDense vector storeをQdrantへ移し、payload filter、永続化、バックアップ、運用境界を整えることを優先しています。
