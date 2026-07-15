# 社内文書をRAGへ追加する手順書

この手順書は、会社PCまたは社内の完全オフライン環境で、社内文書をこのRAGプロジェクトへ追加するための標準手順です。

## 0. 重要な前提

- 社内文書を追加した後は、社外GitHubへpushしないでください。
- `data/markdown/`、`chunks/*.jsonl`、`indexes/chroma/` には社内文書本文または本文由来の情報が入ります。
- 権限が異なる文書を同じRAGに混ぜないでください。このテンプレート単体にはユーザー別ACL制御はありません。
- 全社員が閲覧してよい文書だけを1つのRAGに入れるのが安全です。部署別や機密区分別に分ける場合は、別corpus、別インデックス、別アプリ、または認証・認可の追加実装が必要です。
- 回答済みQAキャッシュは文書・インデックスのversion単位で管理してください。文書を大きく更新した後に古いQAをそのまま流用すると、古い根拠で回答する可能性があります。

## 1. 全体フロー

1. 社内文書の投入対象を決める
2. Word/PDF/ExcelなどをMarkdownへ変換する
3. `data/markdown/company_internal/` にMarkdownを配置する
4. `config/corpus_settings.json` に社内文書corpusを追加する
5. chunkを再生成する
6. SearchTagまたは検索用textを再生成する
7. Chromaインデックスを再作成する
8. API/UIで検索・回答を確認する
9. 評価質問で精度を確認する
10. 必要なQAだけ管理者承認済みQAキャッシュへ登録する

## 2. 社内文書の準備

投入前に、対象文書を棚卸ししてください。

最低限決める項目:

- 文書名
- 文書ID
- 文書種別
- 所管部署
- 適用開始日または版
- 閲覧可能な対象者
- 最新版かどうか
- RAGに入れてよい機密区分かどうか

推奨する投入単位:

- 1つの規程、マニュアル、FAQ、手順書を1Markdownファイルにする
- 長すぎる文書は章単位で分割する
- 表はできるだけMarkdown表として残す
- ヘッダー、フッター、ページ番号、目次だけのページは削除する
- 図表だけで意味が成立する箇所は、図の内容を短い文章で補足する

## 3. Markdown配置先

社内文書用ディレクトリを作成します。

```powershell
New-Item -ItemType Directory -Force data\markdown\company_internal
```

ファイル名は、文書IDとタイトルが分かる名前にします。

```text
data/markdown/company_internal/
  HR-RULE-001_就業規則.md
  IT-MANUAL-003_パスワード管理手順.md
  ACC-FAQ-010_経費精算FAQ.md
```

## 4. Markdown形式

各Markdownの先頭にfrontmatterを付けます。現行スクリプトでRAGの根拠メタデータとして扱いやすい項目は、主に `title`、`document_type`、`document_id`、`category`、`source_site`、`source_url` です。

```markdown
---
title: "就業規則"
document_type: "internal_policy"
document_id: "HR-RULE-001"
category: "人事"
source_site: "company_internal"
source_url: "internal://hr/HR-RULE-001"
law_basis_date: "2026-04-01"
---

# 就業規則

## 第1章 総則

本文...

## 第2章 労働時間

本文...
```

注意:

- `law_basis_date` は税務向けの項目名ですが、現行レスポンスに出せる日付項目として使えます。社内文書では適用開始日や版の日付として使ってください。
- `owner`、`department`、`confidentiality`、`effective_date` などを検索結果へ出したい場合は、`scripts/01_make_chunks.py` の `base_row()` と `api/schemas.py` の拡張が必要です。
- 見出しは `#`、`##`、`###` を使ってください。見出しが弱い文書は検索根拠が読みにくくなります。

## 5. corpus設定を追加する

`config/corpus_settings.json` の `corpora` に、社内文書用corpusを追加します。

```json
{
  "corpus_id": "company_internal",
  "display_name": "社内文書",
  "description": "社内規程、業務マニュアル、FAQを対象にしたRAG corpus",
  "markdown_dir": "data/markdown/company_internal",
  "priority": 90,
  "enabled": true,
  "chunking": {
    "parent_split_levels": [2, 3],
    "max_parent_chars": 8000,
    "min_parent_chars": 1200,
    "child_max_chars": 900,
    "child_overlap_chars": 120,
    "force_heading_split": true,
    "auto_descend": true
  }
}
```

既存の国税庁データと一緒に検索したくない場合は、APIリクエストやUI側で `corpus_ids` に `company_internal` を指定してください。

chunk設定の意味:

- `parent_split_levels`: 親チャンクを切る見出しレベルです。`[2]` はH2、`[2, 3]` はH2/H3で切ります。
- `max_parent_chars`: 親チャンクの目安上限です。超えた場合は段落分割、または `auto_descend` により下位見出しへ降ります。
- `min_parent_chars`: 小さすぎる親チャンクを隣接セクションと結合する目安です。
- `child_max_chars`: 検索用の子チャンク上限です。
- `child_overlap_chars`: 長い子チャンクを分割する際の重なり幅です。
- `force_heading_split`: 文書全体が `max_parent_chars` 以下でも、指定見出しで親チャンクを切ります。
- `auto_descend`: H2で大きすぎる場合にH3/H4へ自動的に降りて切ります。

文書ごとに切り方が違う場合は、Markdown frontmatterで上書きできます。

```markdown
---
title: "経費精算マニュアル"
document_id: "ACC-MANUAL-001"
parent_split_levels: "2,3"
max_parent_chars: "6000"
min_parent_chars: "1000"
force_heading_split: "true"
auto_descend: "true"
---
```

## 6. versionを決める

`config/settings.json` を作成して、社内文書版のversionを明示します。`settings.example.json` はテンプレートなので、社内環境では `settings.json` を使ってください。

```powershell
Copy-Item config\settings.example.json config\settings.json
```

例:

```json
{
  "answer_cache": {
    "corpus_version": "company_internal_2026_06_v1",
    "index_version": "bge-m3_chroma_company_2026_06_v1"
  }
}
```

文書を追加・差し替えしたら、少なくとも `corpus_version` を更新してください。Embeddingモデルやchunk設計を変えた場合は `index_version` も更新してください。

## 7. chunkを再生成する

Markdownを配置したら、親チャンクと子チャンクを再生成します。

```powershell
python scripts\01_make_chunks.py
```

出力:

```text
chunks/parent_chunks.jsonl
chunks/child_chunks.jsonl
chunks/chunk_report.csv
```

確認ポイント:

- `parents` が増えていること
- `children` が増えていること
- 親チャンク数と子チャンク数がほぼ同じになっていないこと
- 1文書が巨大すぎる場合は、Markdown側で章単位に分割すること
- `chunks/chunk_report.csv` の `warnings` を確認すること

`chunk_report.csv` の主な警告:

- `parent_child_counts_too_close`: 親チャンクと子チャンクの数が近く、親子RAGの効果が薄い可能性があります。
- `parent_over_max_chars`: 親チャンクが設定上限を超えています。
- `parent_under_min_chars`: 小さすぎる親チャンクが残っています。
- `large_document_not_split`: 大きい文書がほぼ分割されていません。
- `configured_headings_not_found`: 指定したH2/H3などの見出しが文書内にありません。

`chunk_report.csv` は社内文書名やパスを含むためGit管理対象外です。社内環境で確認用に使ってください。

## 8. SearchTagまたは検索用textを再生成する

`chunks/child_chunks_with_tags.jsonl` が古いままだと、新しく追加した社内文書が検索対象から漏れます。`01_make_chunks.py` の後は、必ず検索用ファイルを更新してください。

現時点で確実に使える手順:

```powershell
python scripts\02_make_taxanswer_search_tags.py
```

このスクリプトは税務向けの簡易タグ生成ですが、少なくとも `title`、`heading_path`、`text` を含む `search_text` を全子チャンクに再付与します。社内文書でも最低限の検索対象化には使えます。

社内文書専用のSearchTag品質を上げる場合は、次のどちらかを追加してください。

- 社内用キーワード辞書を作り、`02_make_taxanswer_search_tags.py` を社内用に分岐する
- ローカルLLMを使って、文書ごとに社内業務用SearchTagを生成する

ローカルLLMでSearchTagを生成する場合は、8並列がデフォルトです。

```powershell
python scripts\02_make_search_tags.py
```

明示的に8並列で実行する場合:

```powershell
python scripts\02_make_search_tags.py --workers 8
```

このスクリプトは `chunks/child_chunks_with_tags.partial.jsonl` に途中保存します。途中で止まった場合は、同じコマンドを再実行すると既存の完了行を再利用して続きから処理します。最初から作り直す場合だけ `--no-resume` を付けます。

```powershell
python scripts\02_make_search_tags.py --workers 8 --no-resume
```

少数件だけで接続確認する場合:

```powershell
python scripts\02_make_search_tags.py --workers 8 --limit 20
```

SearchTagを使わずに本文・見出しだけで始める場合は、古い `child_chunks_with_tags.jsonl` を削除してからインデックスを作成します。

```powershell
Remove-Item chunks\child_chunks_with_tags.jsonl
```

ただし、その場合もAPI起動時のBM25は `child_chunks.jsonl` を読む状態にしてください。

## 9. Chromaインデックスを再作成する

Embedding APIを起動した状態で、Chromaを再作成します。

前提:

- Embedding APIの接続先は `config/settings.json` の `urls.embedding_base_url` で管理します。
- Embeddingモデル名は `config/settings.json` の `embedding.model` で管理します。
- URL、host、portはコードやコマンドに直書きせず、必ず `config/settings.json` の `urls` セクションを確認してください。

実行:

```powershell
python scripts\03_build_index.py
```

出力先:

```text
indexes/chroma/
```

注意:

- 現行の `03_build_index.py` はEmbeddingを作ってからcollectionを作り直します。
- 大量文書を入れる場合はメモリ使用量が増えます。数万チャンク規模になるなら、バッチごとに直接Chromaへ投入する方式へ改修してください。

## 10. APIとUIを起動する

LLM、Embedding、必要ならrerankerを起動した後、RAG APIを起動します。
APIのhost/portは `config/settings.json` の `urls.api_bind_host` / `urls.api_bind_port` から読みます。

```powershell
python scripts\run_api.py
```

チャットUI:

```powershell
python -m app_fasthtml_modern.main
```

管理者アプリ:

```powershell
python -m app_fasthtml_admin_common.main
```

## 11. 検索確認

まず、管理者用の根拠検索APIで、社内文書の該当チャンクが取れるか確認します。
確認先は `config/settings.json` の `urls.rag_api_base_url` を読み、その値に `/admin/evidence/search` を付けてください。

```powershell
$settings = Get-Content config\settings.json -Raw | ConvertFrom-Json
$apiBase = $settings.urls.rag_api_base_url.TrimEnd("/")
$body = @{
  query = "パスワードを忘れた場合の申請手順"
  corpus_ids = @("company_internal")
  top_k = 5
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "$apiBase/admin/evidence/search" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

確認ポイント:

- `corpus_id` が `company_internal` になっている
- `heading_path` が質問意図に近い
- `child_text` に根拠として使える文章がある
- 同じ親文書ばかりに偏りすぎていない

## 12. 回答確認

RAG回答を確認します。
確認先は `config/settings.json` の `urls.rag_api_base_url` を読み、その値に `/ask` を付けてください。

```powershell
$settings = Get-Content config\settings.json -Raw | ConvertFrom-Json
$apiBase = $settings.urls.rag_api_base_url.TrimEnd("/")
$body = @{
  question = "パスワードを忘れた場合はどう申請しますか"
  corpus_ids = @("company_internal")
  top_k = 5
  show_debug = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "$apiBase/ask" `
  -ContentType "application/json; charset=utf-8" `
  -Body $body
```

確認ポイント:

- `answer_source` が `rag` になっている
- `sources` に社内文書の根拠が入っている
- 回答が根拠外の推測をしていない
- `debug.retrieval_query` が更問の文脈を含めて自然になっている

## 13. 評価質問を作る

本番利用前に、社内文書向けの評価質問CSVを作ってください。

推奨する質問:

- 単一文書で答えられる質問
- 複数文書を横断しないと答えられない質問
- 似た制度名や似た手順名で間違えやすい質問
- 口語の質問
- 文書に答えがない質問
- 古い版と新版を混同しやすい質問

最低でも30問、本番前は100問程度を推奨します。

既存の形式を参考にします。

```text
eval/qa_100_questions.csv
```

社内文書用には、別ファイルを作るのが安全です。

```text
eval/company_internal_qa_questions.csv
```

## 14. 承認済みQAキャッシュへ登録する

回答済みQAキャッシュは、自動登録しません。管理者が正しいQAと根拠チャンクを確認してから登録します。

標準フロー:

1. ユーザーが質問する
2. RAGが回答する
3. ユーザーまたは管理者がハルシネーション疑いを報告する
4. 管理者アプリで正しい回答を書く
5. 管理者アプリで根拠チャンクを検索して選ぶ
6. 承認済みQAとして登録する

既存seedのように一括登録する場合も、必ず管理者レビュー済みのものだけを登録してください。

```powershell
python scripts\08_import_approved_qa_seed.py --apply --approved-by admin
```

文書更新後に同じQAを使い続ける場合は、根拠チャンクが現行文書と一致するか確認してから、新しい `corpus_version` と `index_version` で再登録してください。

## 15. 本番前チェックリスト

- 社内文書が社外GitHubにpushされない運用になっている
- `git remote -v` でpush先を確認した
- 権限が違う文書を同じRAGに混ぜていない
- Markdownの見出しが整理されている
- `python scripts\01_make_chunks.py` が成功した
- `chunks/child_chunks_with_tags.jsonl` を再生成した、または削除して stale を避けた
- `python scripts\03_build_index.py` が成功した
- `/admin/evidence/search` で社内文書が検索できる
- `/ask` で根拠付き回答が返る
- 評価質問で検索漏れ・誤回答を確認した
- 承認済みQAキャッシュのversionが現行indexと一致している
- ログSQLiteに機密情報が入る前提で保存場所と保護を決めた

## 16. 足りなくなりやすい実装

社内文書を本番運用する場合、次の実装は早めに検討してください。

- 社内文書専用SearchTag生成
- Word/PDF/ExcelからMarkdownへの変換バッチ
- 文書ID、版、所管部署、機密区分などの汎用メタデータ保持
- 部署別・権限別の検索制御
- 大量チャンク向けのChroma投入処理
- 文書更新時の差分検知
- 古い承認済みQAキャッシュの棚卸し
- 評価質問のバッチ推論とスコア集計
- インデックス作成結果のmanifest出力
