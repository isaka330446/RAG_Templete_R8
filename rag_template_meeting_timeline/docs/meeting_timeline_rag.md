# 会議資料・議事録・時系列イベントRAG

このテンプレートは `rag_template_answer_cache` をベースに、PowerPoint会議資料のMarkdown化結果、議事録のMarkdown化結果、MeetingEventの時系列検索を追加したものです。

## 入力Markdown

Markdownファイルは `data/markdown/meeting_documents/` に配置します。PPTXや議事録そのものではなく、クラウドAIなどでMarkdown化した結果を置きます。

対応する `document_type` は次の2種類です。

- `meeting_slide_deck`: PowerPoint会議資料
- `meeting_minutes`: 議事録

両方のMarkdownで `meeting_id` を同じ値にすると、資料と議事録が同じ会議として紐づきます。`meeting_date` は必須です。未設定の場合、そのファイルはエラーとしてスキップされます。

## 生成AIプロンプト出力との互換

Claude等に投げる変換プロンプトの出力をそのまま取り込めるよう、パーサは次の揺れを吸収します。

- frontmatter の値が `"..."` で囲まれていても通常値として扱う
- `meeting_date: ""` は空欄として扱い、エラーにする
- 分割出力の先頭にある `<!-- part: ... -->`、`<!-- slide_range: ... -->`、`<!-- agenda_range: ... -->` を無視する
- `なし`、`不明`、`明記なし`、`取得できない` は任意項目では空欄として扱う
- PPTX側の `proposed_events` で `event_type: decision` が出ても `proposal` に降格する
- 議事録側に `# meeting_events` がある場合、その明示イベントをMeetingEventの正として使う
- `# meeting_events` がない場合だけ、`decisions`、`concerns`、`action_items`、`pending_items`、`rejected_items`、`discussion` からイベントを生成する

これにより、PPTX資料側の「案・説明・提案・報告」と、議事録側の「決定・懸念・宿題・保留・見送り」を分離して扱います。

## 取り込み

```bash
python scripts/01_make_chunks.py
python scripts/02_make_search_tags.py
python scripts/03_build_index.py
```

`01_make_chunks.py` は通常Markdownと会議Markdownを自動判定します。会議Markdownの場合は、既存の親子チャンクJSONLに加えて `data/meeting_events/meeting_events.jsonl` を生成します。

出力先:

- `chunks/parent_chunks.jsonl`
- `chunks/child_chunks.jsonl`
- `chunks/chunk_report.csv`
- `data/meeting_events/meeting_events.jsonl`

## チャンク設計

PPTX資料:

- 親チャンク: スライド1枚
- 子チャンク: `visible_text`、`tables_markdown`、`figures_and_charts`、`visual_summary`、`speaker_notes`、`proposed_events`、`search_tags`
- PPTX資料の `proposed_events` は原則として提案・報告扱いです。`decision` と書かれていても `proposal` に落とします。

議事録:

- 親チャンク: 議題・セクション単位
- 子チャンク: `explanation`、`discussion`、`decisions`、`concerns`、`action_items`、`pending_items`、`rejected_items`、`search_tags`
- `decisions` だけを決定事項として扱います。

## MeetingEvent

議事録から次のイベントを作ります。

- `decisions` -> `decision`
- `concerns` -> `concern`
- `action_items` -> `action_item`
- `pending_items` -> `pending`
- `rejected_items` -> `rejection`
- `discussion` -> `discussion`

PPTX資料の `proposed_events` は `proposal`、`discussion`、`concern`、`change`、`report` などとして扱えますが、決定事項にはしません。

## 時系列検索

APIの `/ask` は `answer_mode` を受け取れます。

- `auto`: 質問文から時系列系の質問か自動判定
- `rag`: 通常RAGのみ
- `timeline`: MeetingEventの時系列検索を明示

例:

```json
{
  "question": "生成AI活用方針のこれまでの経緯を時系列で教えて",
  "corpus_ids": ["meeting_documents"],
  "answer_mode": "timeline",
  "event_type": "decision"
}
```

時系列回答は、`event_date`、`meeting_date` の昇順でアプリ側がソートします。LLMにはソートを任せません。

## MeetingEvent CLI

ベクトル化前でもイベント抽出結果を確認できます。

```bash
python scripts/07_query_meeting_events.py --topic 生成AI --event-type decision
python scripts/07_query_meeting_events.py --meeting-id 2024-09-12_dx_suishin_03 --json
```

主なフィルタ:

- `--meeting-id`
- `--topic`
- `--event-type`
- `--date-from`
- `--date-to`
- `--status`
- `--owner`
- `--source-type slide|minutes`

## API

通常の `/ask` に加えて、MeetingEventを直接見るためのAPIがあります。

```bash
GET /meeting-events?topic=生成AI&event_type=decision
```

## 動作確認

```bash
python -m unittest discover -s tests
python -m compileall api app scripts tests
```

実データがない段階では、テストはインラインのサンプルMarkdownでパーサとイベント生成を確認します。
