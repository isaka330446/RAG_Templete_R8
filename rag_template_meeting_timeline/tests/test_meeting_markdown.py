import tempfile
import unittest
from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from api.meeting_event_store import MeetingEventStore
from api.meeting_markdown import build_meeting_chunks, parse_meeting_markdown_text
from api.meeting_timeline import format_timeline_answer


SLIDE_MARKDOWN = """---
document_type: meeting_slide_deck
meeting_id: 2024-09-12_dx_suishin_03
meeting_name: 第3回DX推進会議
meeting_date: 2024-09-12
fiscal_year: 令和6年度
department: 情報システム担当
source_file: 第3回DX推進会議資料.pptx
confidentiality: 内部資料
language: ja
---

---

# Slide 1: 生成AI PoC案

## slide_metadata

- slide_no: 1
- slide_title: 生成AI PoC案
- agenda: 議題1
- section: 方針案
- source_type: slide

## visible_text

生成AIを2業務でPoCする案を提示する。

## tables

| 業務 | 目的 |
|---|---|
| FAQ | 回答支援 |

## figures_and_charts

対象業務の比較図。

## visual_summary

PoC候補を比較している。

## speaker_notes

決定ではなく提案として説明する。

## proposed_events

- event_type: decision
  topic: 生成AI活用
  summary: 対象業務を2件に絞る案を提示した
  confidence: 0.8

## search_tags

- 生成AI
- PoC
"""


MINUTES_MARKDOWN = """---
document_type: meeting_minutes
meeting_id: 2024-09-12_dx_suishin_03
meeting_name: 第3回DX推進会議
meeting_date: 2024-09-12
fiscal_year: 令和6年度
department: 情報システム担当
source_file: 第3回DX推進会議_議事録.docx
confidentiality: 内部資料
language: ja
---

# meeting_overview

DX推進会議の議事録。

---

# 議題1: 生成AI活用方針

## section_metadata

- agenda: 議題1
- topic: 生成AI活用
- source_type: minutes
- section_type: agenda

## explanation

資料の提案内容を説明した。

## discussion

対象業務の優先順位について議論した。

## decisions

- decision: 対象業務を2件に絞ってPoCを開始する
  topic: 生成AI活用
  status: 決定
  confidence: 0.95

## concerns

- concern: 個人情報を含む資料の扱いに注意が必要
  topic: セキュリティ
  confidence: 0.9

## action_items

- action_item: 対象業務の候補を一覧化する
  owner: 情報システム担当
  due_date: 2024-09-30
  status: 未完了
  confidence: 0.9

## pending_items

- pending_item: 本番運用時の監査ログ保存期間
  status: 保留

## rejected_items

- rejected_item: 全業務を同時にPoC対象にする案
  status: 見送り

## search_tags

- 生成AI
- PoC
"""


class MeetingMarkdownTest(unittest.TestCase):
    def test_parse_slide_records_and_proposal_event(self):
        parsed = parse_meeting_markdown_text(SLIDE_MARKDOWN)
        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.slides), 1)
        self.assertEqual(parsed.slides[0].slide_no, 1)
        self.assertEqual(parsed.slides[0].source_type, "slide")
        self.assertEqual(len(parsed.events), 1)
        self.assertEqual(parsed.events[0].event_type, "proposal")
        self.assertIn("downgraded", "; ".join(parsed.warnings))

    def test_parse_minutes_records_and_events(self):
        parsed = parse_meeting_markdown_text(MINUTES_MARKDOWN)
        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.minutes_sections), 1)
        event_types = {event.event_type for event in parsed.events}
        self.assertIn("decision", event_types)
        self.assertIn("action_item", event_types)
        self.assertIn("concern", event_types)
        self.assertIn("pending", event_types)
        self.assertIn("rejection", event_types)
        self.assertIn("discussion", event_types)

    def test_build_parent_child_chunks(self):
        parsed = parse_meeting_markdown_text(SLIDE_MARKDOWN)
        parents, children, reports, events = build_meeting_chunks(parsed, "meeting_documents", "deck.md")
        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0]["source_type"], "slide")
        self.assertEqual(parents[0]["slide_no"], 1)
        content_types = {child["content_type"] for child in children}
        self.assertIn("visible_text", content_types)
        self.assertIn("tables_markdown", content_types)
        self.assertIn("figures_and_charts", content_types)
        self.assertIn("visual_summary", content_types)
        self.assertIn("speaker_notes", content_types)
        self.assertEqual(reports[0]["parent_count"], 1)
        self.assertEqual(events[0]["event_type"], "proposal")

    def test_meeting_id_links_slide_and_minutes(self):
        slide = parse_meeting_markdown_text(SLIDE_MARKDOWN).slides[0]
        minutes = parse_meeting_markdown_text(MINUTES_MARKDOWN).minutes_sections[0]
        self.assertEqual(slide.meeting_id, minutes.meeting_id)

    def test_missing_meeting_date_is_error(self):
        broken = SLIDE_MARKDOWN.replace("meeting_date: 2024-09-12\n", "")
        parsed = parse_meeting_markdown_text(broken)
        self.assertTrue(parsed.errors)
        self.assertIn("meeting_date", parsed.errors[0])

    def test_quoted_frontmatter_and_part_comments_are_supported(self):
        markdown = '''<!-- part: 1 / unknown -->
<!-- slide_range: 1-1 -->
---
document_type: meeting_slide_deck
meeting_id: "2024-09-12_dx_suishin_03"
meeting_name: "第3回DX推進会議"
meeting_date: "2024-09-12"
fiscal_year: "令和6年度"
department: "情報システム担当"
source_file: "第3回DX推進会議資料.pptx"
deck_title: "第3回DX推進会議資料"
confidentiality: "内部資料"
language: ja
---

---

# Slide 1: 表紙

## slide_metadata

- slide_no: 1
- slide_title: 表紙
- agenda:
- section:
- source_type: slide
- page_role: cover

## visible_text

第3回DX推進会議

## tables

なし

## figures_and_charts

なし

## visual_summary

- 事実:
  - 表紙である
- 推測:
  - なし

## speaker_notes

なし

## proposed_events

- event_type: none
  topic:
  summary:
  confidence:

## search_tags

- DX推進会議
'''
        parsed = parse_meeting_markdown_text(markdown)
        self.assertEqual(parsed.errors, [])
        self.assertEqual(parsed.meta.meeting_id, "2024-09-12_dx_suishin_03")
        self.assertEqual(parsed.meta.meeting_date, "2024-09-12")
        self.assertEqual(len(parsed.events), 0)
        parents, children, _, _ = build_meeting_chunks(parsed, "meeting_documents", "deck.md")
        self.assertEqual(len(parents), 1)
        self.assertNotIn("tables_markdown", {child["content_type"] for child in children})
        self.assertNotIn("speaker_notes", {child["content_type"] for child in children})

    def test_quoted_blank_meeting_date_is_error(self):
        markdown = '''---
document_type: meeting_minutes
meeting_id: ""
meeting_name: "第3回DX推進会議"
meeting_date: ""
source_file: "minutes.docx"
language: ja
---

# meeting_overview
'''
        parsed = parse_meeting_markdown_text(markdown)
        self.assertTrue(parsed.errors)
        self.assertIn("meeting_date", parsed.errors[0])

    def test_explicit_meeting_events_are_authoritative(self):
        markdown = '''---
document_type: meeting_minutes
meeting_id: "2024-09-12_dx_suishin_03"
meeting_name: "第3回DX推進会議"
meeting_date: "2024-09-12"
source_file: "minutes.docx"
language: ja
---

# 議題1: 生成AI活用方針

## section_metadata

- agenda: 議題1
- topic: 生成AI活用
- source_type: minutes
- section_type: agenda

## explanation

説明した。

## discussion

議論した。

## decisions

- decision: 対象業務を2件に絞ってPoCを開始する
  topic: 生成AI活用
  status: 決定
  confidence: 0.95

## concerns

なし

## action_items

なし

## pending_items

なし

## rejected_items

なし

## search_tags

- 生成AI

# meeting_events

- event_type: decision
  event_date: 2024-09-12
  topic: 生成AI活用
  subtopic:
  event_summary: 対象業務を2件に絞ってPoCを開始する
  owner:
  due_date:
  status: 決定
  source_section: 議題1: 生成AI活用方針
  confidence: 0.95
'''
        parsed = parse_meeting_markdown_text(markdown)
        self.assertEqual(parsed.errors, [])
        self.assertEqual(len(parsed.minutes_sections), 1)
        self.assertEqual(len(parsed.events), 1)
        self.assertEqual(parsed.events[0].event_type, "decision")
        self.assertEqual(parsed.events[0].source_refs[0]["section_title"], "議題1: 生成AI活用方針")

    def test_event_store_sorts_and_filters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = MeetingEventStore(Path(temp_dir) / "events.jsonl")
            events = [
                {
                    "event_id": "b",
                    "meeting_id": "m1",
                    "meeting_name": "第2回",
                    "meeting_date": "2024-09-20",
                    "event_date": "2024-09-20",
                    "topic": "生成AI活用",
                    "event_type": "action_item",
                    "event_summary": "宿題",
                    "source_refs": [],
                },
                {
                    "event_id": "a",
                    "meeting_id": "m1",
                    "meeting_name": "第1回",
                    "meeting_date": "2024-09-12",
                    "event_date": "2024-09-12",
                    "topic": "生成AI活用",
                    "event_type": "decision",
                    "event_summary": "決定",
                    "source_refs": [],
                },
            ]
            store.replace_all(events)
            rows = store.query(meeting_id="m1")
            self.assertEqual([row["event_id"] for row in rows], ["a", "b"])
            decisions = store.query(meeting_id="m1", event_type="decision")
            self.assertEqual(len(decisions), 1)
            self.assertEqual(decisions[0]["event_summary"], "決定")

    def test_timeline_formatter_outputs_decision_table(self):
        answer = format_timeline_answer([
            {
                "event_id": "a",
                "meeting_id": "m1",
                "meeting_name": "第1回",
                "meeting_date": "2024-09-12",
                "event_date": "2024-09-12",
                "topic": "生成AI活用",
                "event_type": "decision",
                "event_summary": "対象業務を2件に絞ってPoCを開始する",
                "source_refs": [{"source_type": "minutes", "source_file": "minutes.md", "agenda": "議題1", "section_title": "生成AI"}],
            }
        ])
        self.assertIn("詳細タイムライン表", answer)
        self.assertIn("decision", answer)
        self.assertIn("対象業務を2件", answer)


if __name__ == "__main__":
    unittest.main()
