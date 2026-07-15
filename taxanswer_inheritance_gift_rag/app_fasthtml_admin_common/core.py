from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlencode

import httpx
from api.config import allow_runtime_api_url_override, load_settings, runtime_rag_api_base_url
from fasthtml.common import fast_app, serve
from markdown_it import MarkdownIt
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent.parent
MARKDOWN = MarkdownIt("commonmark", {"html": False, "breaks": False}).enable(["table", "strikethrough"])


@dataclass(frozen=True)
class AdminVariant:
    key: str
    title: str
    subtitle: str
    port: int


VARIANTS = {
    "command_bi": AdminVariant("command_bi", "RAG改善コンソール", "運用状態を見て、改善アクションへつなげる管理画面", 8611),
    "command_saas": AdminVariant("command_saas", "RAG改善コンソール", "問題検知、優先順位付け、QA/SearchTag/評価改善の入口", 8612),
    "analyst_bi": AdminVariant("analyst_bi", "RAG改善コンソール", "ログと報告から品質改善ポイントを探す管理画面", 8613),
    "analyst_saas": AdminVariant("analyst_saas", "RAG改善コンソール", "RAG運用を日次で確認する実用コンソール", 8614),
}

STATUS_LABELS = {
    "open": "未対応",
    "resolved": "対応済み",
    "ignored": "対応不要",
    "approved": "有効",
    "disabled": "無効",
    "all": "すべて",
}
PRIORITY_LABELS = {"high": "高", "medium": "中", "low": "低"}
TASK_LABELS = {
    "hallucination_report": "報告あり",
    "no_hit": "根拠なし",
    "low_confidence": "低信頼",
    "frequent_question": "頻出質問",
    "unstable_retrieval": "検索不安定",
    "stale_qa": "古いQA",
    "index_warning": "Index警告",
    "eval_failed": "評価失敗",
    "search_tag_candidate": "SearchTag候補",
}
TASK_LABELS.update({
    "near_cache_miss": "惜しいQAキャッシュミス",
    "qa_alias_missing": "QA alias未生成",
})
ISSUE_TYPES = {
    "": "未分類",
    "retrieval_miss": "正しい文書はあるが検索できていない",
    "corpus_missing": "コーパスに根拠がない",
    "generation_error": "根拠はあるが生成が誤った",
    "out_of_scope": "対象外質問",
    "user_misunderstanding": "ユーザー誤認/回答は妥当",
    "other": "その他",
}
RESOLUTION_TYPES = {
    "": "未選択",
    "qa_created": "承認QAを作成",
    "search_tag_updated": "SearchTagを更新",
    "document_update_needed": "文書追加/更新が必要",
    "prompt_or_generation_fix_needed": "生成/プロンプト側の修正が必要",
    "marked_out_of_scope": "対象外として整理",
    "no_action": "対応不要",
    "other": "その他",
}


def esc(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def markdown_block(value: object) -> str:
    return MARKDOWN.render(str(value or ""))


def text_block(value: object) -> str:
    return esc(value).replace("\n", "<br>")


def short_text(value: object, limit: int = 90) -> str:
    text = str(value or "").strip().replace("\r\n", "\n").replace("\n", " ")
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "..."


def js_string(value: object) -> str:
    text = str(value or "")
    text = text.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n").replace("'", "\\'")
    return esc(text)


def int_value(value: object, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def float_value(value: object, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def percent_text(value: object) -> str:
    return f"{float_value(value) * 100:.1f}%"


def normalize_base_url(value: object) -> str:
    return runtime_rag_api_base_url(str(value or "").strip() or None)


def selected_base(request: Request) -> str:
    candidate = request.query_params.get("base_url") if allow_runtime_api_url_override() else None
    return normalize_base_url(candidate)


def with_base(path: str, base_url: str, **params: object) -> str:
    clean = clean_query_params(params) or {}
    if allow_runtime_api_url_override():
        clean["base_url"] = base_url
    return f"{path}?{urlencode(clean, doseq=True)}" if clean else path


def redirect(path: str, base_url: str, **params: object) -> RedirectResponse:
    return RedirectResponse(with_base(path, base_url, **params), status_code=303)


def clean_query_params(params: dict | None) -> dict | None:
    if not params:
        return None
    clean: dict[str, object] = {}
    for key, value in params.items():
        if value in (None, "", []):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        clean[key] = value
    return clean or None


async def api_get(base_url: str, path: str, params: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=90) as client:
        res = await client.get(f"{base_url}{path}", params=clean_query_params(params))
        res.raise_for_status()
        return res.json()


async def api_post(base_url: str, path: str, payload: dict, timeout: int = 180) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        res = await client.post(f"{base_url}{path}", json=payload)
        res.raise_for_status()
        return res.json()


async def api_patch(base_url: str, path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=90) as client:
        res = await client.patch(f"{base_url}{path}", json=payload)
        res.raise_for_status()
        return res.json()


def api_error_message(error: object) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        response = error.response
        detail: object
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        if isinstance(detail, (dict, list)):
            detail_text = json.dumps(detail, ensure_ascii=False, indent=2)
        else:
            detail_text = str(detail or "")
        if len(detail_text) > 4000:
            detail_text = detail_text[:4000] + "\n..."
        return f"HTTP {response.status_code} from RAG API\n{detail_text}"
    return str(error)


def page_css() -> str:
    return """
    @font-face {
      font-family:"Noto Sans JP Local";
      src:url("/static/fonts/NotoSansJP-wght.ttf") format("truetype");
      font-weight:100 900;
      font-style:normal;
      font-display:swap;
    }
    :root {
      --bg:#f4f7fb;
      --panel:#ffffff;
      --panel-soft:#f8fbff;
      --ink:#172033;
      --muted:#667085;
      --line:#e2e8f0;
      --primary:#2557d6;
      --primary-soft:#eef4ff;
      --green:#0f766e;
      --green-soft:#ecfdf5;
      --warn:#b45309;
      --warn-soft:#fff7ed;
      --bad:#b42318;
      --bad-soft:#fef3f2;
      --shadow:0 18px 54px rgba(15,23,42,.10);
      --radius:16px;
    }
    * { box-sizing:border-box; }
    body {
      margin:0;
      min-height:100vh;
      color:var(--ink);
      background:
        radial-gradient(circle at top left, rgba(37,87,214,.13), transparent 34%),
        radial-gradient(circle at bottom right, rgba(15,118,110,.10), transparent 30%),
        var(--bg);
      font-family:"Noto Sans JP Local","Noto Sans JP","Noto Sans CJK JP","Yu Gothic UI","Meiryo",sans-serif;
      font-size:15px;
      letter-spacing:0;
    }
    a { color:inherit; text-decoration:none; }
    button,input,select,textarea { font:inherit; }
    .topbar {
      height:68px;
      display:flex;
      align-items:center;
      justify-content:space-between;
      padding:0 22px;
      background:rgba(255,255,255,.86);
      border-bottom:1px solid var(--line);
      backdrop-filter:blur(14px);
      position:sticky;
      top:0;
      z-index:10;
    }
    .brand strong { display:block; font-size:18px; }
    .brand span { display:block; color:var(--muted); font-size:12px; margin-top:2px; }
    .api-note { color:var(--muted); font-size:12px; }
    .layout {
      width:min(1840px, calc(100vw - 32px));
      margin:18px auto 30px;
      display:grid;
      grid-template-columns:240px minmax(0,1fr);
      gap:16px;
      align-items:start;
    }
    .side {
      position:sticky;
      top:86px;
      display:grid;
      gap:6px;
      padding:12px;
      background:rgba(255,255,255,.88);
      border:1px solid var(--line);
      border-radius:var(--radius);
      box-shadow:var(--shadow);
    }
    .side a {
      display:flex;
      align-items:center;
      justify-content:space-between;
      min-height:40px;
      padding:9px 11px;
      color:var(--muted);
      border-radius:12px;
      font-weight:700;
    }
    .side a.active,.side a:hover { color:var(--primary); background:var(--primary-soft); }
    .main { display:grid; gap:16px; min-width:0; }
    .panel,.card,.kpi {
      background:rgba(255,255,255,.9);
      border:1px solid var(--line);
      border-radius:var(--radius);
      box-shadow:var(--shadow);
    }
    .panel { padding:18px; }
    .panel h2 { margin:0 0 13px; font-size:18px; }
    .panel h3 { margin:18px 0 10px; font-size:15px; }
    .lead { color:var(--muted); line-height:1.75; margin:.2rem 0 1rem; }
    .grid-2 { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
    .grid-3 { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }
    .grid-4 { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }
    .grid-6 { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:12px; }
    .kpi { padding:14px; }
    .kpi span { color:var(--muted); font-size:12px; font-weight:800; }
    .kpi strong { display:block; font-size:28px; margin-top:4px; }
    .kpi small { color:var(--muted); display:block; margin-top:4px; }
    .status-banner {
      display:flex;
      justify-content:space-between;
      gap:16px;
      align-items:flex-start;
      padding:16px 18px;
      border-radius:var(--radius);
      border:1px solid var(--line);
      background:linear-gradient(135deg,#fff,#f8fbff);
      box-shadow:var(--shadow);
    }
    .status-title { font-size:22px; font-weight:900; }
    .status-ok { color:var(--green); }
    .status-warn { color:var(--warn); }
    .status-bad { color:var(--bad); }
    .alert-list { display:grid; gap:8px; margin-top:10px; }
    .notice {
      padding:10px 12px;
      border-radius:12px;
      background:var(--warn-soft);
      border:1px solid #fed7aa;
      color:#9a3412;
      line-height:1.65;
    }
    .empty {
      padding:16px;
      border:1px dashed var(--line);
      border-radius:12px;
      color:var(--muted);
      background:var(--panel-soft);
      line-height:1.7;
    }
    .filters { display:flex; flex-wrap:wrap; gap:10px; align-items:end; margin-bottom:14px; }
    .field { display:grid; gap:5px; }
    .field label { color:var(--muted); font-size:12px; font-weight:800; }
    .field input,.field select,.field textarea {
      border:1px solid var(--line);
      border-radius:12px;
      padding:9px 10px;
      background:#fff;
      min-width:140px;
    }
    textarea { width:100%; min-height:110px; resize:vertical; }
    .btn {
      display:inline-flex;
      align-items:center;
      justify-content:center;
      min-height:38px;
      border:1px solid var(--primary);
      border-radius:12px;
      padding:8px 12px;
      color:#fff;
      background:var(--primary);
      cursor:pointer;
      font-weight:800;
      gap:6px;
    }
    .btn.secondary { background:#fff; color:var(--primary); }
    .btn.ghost { background:#fff; color:var(--ink); border-color:var(--line); }
    .btn.danger { background:var(--bad); border-color:var(--bad); }
    .btn.small { min-height:30px; padding:5px 9px; font-size:12px; }
    .table-wrap { overflow:auto; border:1px solid var(--line); border-radius:14px; background:#fff; }
    table { width:100%; border-collapse:collapse; }
    th,td { padding:10px 11px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
    th { background:var(--panel-soft); color:var(--muted); font-size:12px; white-space:nowrap; }
    tr:hover td { background:#fbfdff; }
    .badge {
      display:inline-flex;
      align-items:center;
      min-height:24px;
      padding:3px 8px;
      border-radius:999px;
      font-size:12px;
      font-weight:900;
      color:#1e3a8a;
      background:#dbeafe;
      white-space:nowrap;
    }
    .badge.ok { color:#067647; background:var(--green-soft); }
    .badge.warn { color:#c2410c; background:var(--warn-soft); }
    .badge.bad { color:var(--bad); background:var(--bad-soft); }
    .badge.muted { color:var(--muted); background:#f1f5f9; }
    .split { display:grid; grid-template-columns:minmax(0,1fr) 460px; gap:14px; align-items:start; }
    .cards { display:grid; gap:10px; }
    .card { padding:13px; display:grid; gap:8px; }
    .card-title { font-weight:900; }
    .muted { color:var(--muted); }
    .pre {
      white-space:pre-wrap;
      line-height:1.75;
      padding:13px;
      border:1px solid var(--line);
      border-radius:12px;
      background:var(--panel-soft);
      overflow:auto;
    }
    .markdown-body { line-height:1.8; word-break:break-word; }
    .markdown-body > :first-child { margin-top:0; }
    .markdown-body > :last-child { margin-bottom:0; }
    .markdown-body h1,.markdown-body h2,.markdown-body h3 { line-height:1.35; margin:1em 0 .55em; letter-spacing:0; }
    .markdown-body h1 { font-size:1.34rem; }
    .markdown-body h2 { font-size:1.15rem; }
    .markdown-body h3 { font-size:1.02rem; }
    .markdown-body p { margin:.55em 0; }
    .markdown-body ul,.markdown-body ol { padding-left:1.35em; margin:.55em 0; }
    .markdown-body table { width:100%; border-collapse:collapse; margin:.8em 0; }
    .markdown-body th,.markdown-body td { border:1px solid var(--line); padding:7px 8px; }
    .source-grid { display:grid; gap:10px; }
    .source-card { border:1px solid var(--line); border-radius:12px; padding:12px; background:#fff; }
    .source-card header { display:flex; justify-content:space-between; gap:10px; margin-bottom:8px; }
    .source-meta { display:flex; flex-wrap:wrap; gap:8px; margin:8px 0; }
    .source-meta span { display:inline-flex; padding:4px 8px; border-radius:999px; background:#f1f5f9; color:var(--muted); font-size:12px; }
    details { margin-top:10px; }
    summary { cursor:pointer; color:var(--primary); font-weight:900; }
    .bars { display:grid; gap:8px; }
    .bar-row { display:grid; grid-template-columns:170px minmax(0,1fr) 56px; gap:9px; align-items:center; }
    .bar { height:10px; border-radius:999px; background:#e5e7eb; overflow:hidden; }
    .bar i { display:block; height:100%; background:linear-gradient(90deg,var(--primary),var(--green)); }
    .spark { width:100%; height:128px; }
    .right-actions { display:flex; flex-wrap:wrap; gap:8px; }
    .admin-modal { position:fixed; inset:0; z-index:80; display:grid; place-items:center; padding:24px; }
    .admin-modal-backdrop { position:absolute; inset:0; background:rgba(15,23,42,.58); backdrop-filter:blur(2px); }
    .admin-modal-card {
      position:relative;
      width:min(1500px, calc(100vw - 42px));
      height:min(92vh, 1040px);
      overflow:hidden;
      background:#fff;
      border:1px solid var(--line);
      border-radius:22px;
      box-shadow:0 24px 80px rgba(15,23,42,.24);
      display:flex;
      flex-direction:column;
    }
    .admin-modal-card.qa-detail-modal { width:min(1560px, calc(100vw - 36px)); }
    .admin-modal-card.alias-modal { width:min(1240px, calc(100vw - 42px)); }
    .admin-modal-card.log-detail-modal { width:min(1500px, calc(100vw - 36px)); height:min(90vh, 980px); }
    .admin-modal-head { flex:0 0 auto; padding:18px 22px; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; gap:12px; background:#fff; position:relative; z-index:1; }
    .admin-modal-body { overflow:auto; height:100%; padding:22px; background:#fff; }
    .admin-modal-grid { display:grid; grid-template-columns:minmax(520px, 1.1fr) minmax(420px, .9fr); gap:16px; align-items:start; }
    .qa-detail-wide { display:grid; grid-template-columns:minmax(0, 1fr); gap:14px; }
    .qa-detail-wide textarea[name="question"] { min-height:88px; font-size:16px; }
    .qa-detail-wide textarea[name="answer"] { min-height:260px; font-size:15px; line-height:1.8; }
    .qa-answer-preview {
      margin-top:10px;
      padding:16px 18px;
      border:1px solid var(--line);
      border-radius:14px;
      background:#f8fbff;
      max-height:360px;
      overflow:auto;
    }
    .qa-answer-preview h1,.qa-answer-preview h2,.qa-answer-preview h3 { margin-top:0; }
    .qa-answer-preview p { line-height:1.75; }
    .qa-answer-preview ul,.qa-answer-preview ol { padding-left:1.5em; }
    .qa-register-tools { display:grid; grid-template-columns:minmax(280px, .8fr) minmax(360px, 1.2fr); gap:14px; align-items:start; margin:14px 0; }
    .qa-register-actions { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:10px; }
    .qa-workspace-grid { display:grid; grid-template-columns:minmax(480px, 1fr) minmax(420px, .9fr); gap:16px; align-items:start; }
    .qa-workspace-main,.qa-workspace-side,.qa-similar-box { display:grid; gap:12px; }
    .qa-similar-box { margin-top:10px; }
    .modal-close { border:1px solid var(--line); border-radius:999px; background:#fff; color:var(--ink); padding:9px 13px; text-decoration:none; font-weight:900; }
    .admin-loading { position:fixed; inset:0; z-index:140; display:none; place-items:center; padding:24px; background:rgba(15,23,42,.46); backdrop-filter:blur(3px); }
    body.admin-loading-active .admin-loading { display:grid; }
    .admin-loading-card { width:min(420px, calc(100vw - 40px)); border:1px solid rgba(255,255,255,.72); border-radius:22px; background:rgba(255,255,255,.94); box-shadow:0 24px 80px rgba(15,23,42,.24); padding:24px; display:grid; justify-items:center; gap:12px; text-align:center; }
    .admin-loading-spinner { width:48px; height:48px; border-radius:999px; border:4px solid #dbeafe; border-top-color:var(--primary); animation:admin-spin .85s linear infinite; }
    .admin-loading-card strong { font-size:17px; }
    .admin-loading-card small { color:var(--muted); line-height:1.6; }
    @keyframes admin-spin { to { transform:rotate(360deg); } }
    @media (prefers-reduced-motion: reduce) { .admin-loading-spinner { animation:none; } }
    .alias-row.alias-disabled { opacity:.48; background:#f3f4f6; color:#6b7280; }
    .alias-row.alias-active { background:#fff; }
    .alias-row.alias-risk { background:#fff7ed; }
    @media (max-width:1180px) {
      .layout,.split { grid-template-columns:1fr; }
      .admin-modal-grid { grid-template-columns:1fr; }
      .qa-workspace-grid,.qa-register-tools { grid-template-columns:1fr; }
      .side { position:static; grid-template-columns:repeat(3,minmax(0,1fr)); }
      .grid-6,.grid-4,.grid-3,.grid-2 { grid-template-columns:repeat(2,minmax(0,1fr)); }
    }
    @media (max-width:720px) {
      .layout { width:min(100vw - 18px, 100%); margin:10px auto 18px; }
      .topbar { height:auto; padding:12px; align-items:flex-start; }
      .side,.grid-6,.grid-4,.grid-3,.grid-2 { grid-template-columns:1fr; }
      .status-banner { display:grid; }
    }
    """


def nav_html(active: str, base_url: str) -> str:
    items = [
        ("/", "dashboard", "ダッシュボード"),
        ("/queue", "queue", "改善キュー"),
        ("/reports", "reports", "報告対応"),
        ("/qa-cache", "qa", "承認QA"),
        ("/logs", "logs", "ログ探索"),
        ("/index-eval", "index_eval", "文書/評価"),
    ]
    return '<aside class="side">' + "".join(
        f'<a class="{"active" if key == active else ""}" href="{with_base(path, base_url)}"><span>{esc(label)}</span><span>›</span></a>'
        for path, key, label in items
    ) + "</aside>"


def page_shell(variant: AdminVariant, active: str, base_url: str, content: str) -> str:
    return f"""<!doctype html>
    <html lang="ja">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>{esc(variant.title)}</title>
      <style>{page_css()}</style>
      <script>
        function fillEvidence(id, jsonText) {{
          const el = document.getElementById(id);
          if (!el) return;
          let current = [];
          try {{
            const parsed = JSON.parse(el.value || "[]");
            current = Array.isArray(parsed) ? parsed : [];
          }} catch (e) {{
            current = [];
          }}
          let next = [];
          try {{
            const parsedNext = JSON.parse(jsonText || "[]");
            next = Array.isArray(parsedNext) ? parsedNext : [];
          }} catch (e) {{
            next = [];
          }}
          const seen = new Set(current.map((item) => String(item.child_id || item.parent_id || item.source_file || JSON.stringify(item))));
          next.forEach((item) => {{
            const key = String(item.child_id || item.parent_id || item.source_file || JSON.stringify(item));
            if (!seen.has(key)) {{
              current.push(item);
              seen.add(key);
            }}
          }});
          el.value = JSON.stringify(current, null, 2);
        }}
        function showAdminLoading(message) {{
          const label = document.getElementById("admin-loading-label");
          if (label) label.textContent = message || "API処理中...";
          document.body.classList.add("admin-loading-active");
        }}
        window.addEventListener("pageshow", () => {{
          document.body.classList.remove("admin-loading-active");
        }});
        document.addEventListener("DOMContentLoaded", () => {{
          document.querySelectorAll("form").forEach((form) => {{
            form.addEventListener("submit", (event) => {{
              const submitter = event.submitter;
              const text = submitter && submitter.textContent ? submitter.textContent.trim() : "";
              const message = (submitter && submitter.dataset.loadingMessage) || form.dataset.loadingMessage || (text ? text + "中..." : "API処理中...");
              showAdminLoading(message);
            }});
          }});
          document.querySelectorAll("a[href]").forEach((link) => {{
            link.addEventListener("click", (event) => {{
              if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
              if (link.target === "_blank" || link.hasAttribute("download") || link.dataset.noLoading === "true") return;
              if (link.closest(".admin-modal-backdrop") || link.closest(".modal-close")) return;
              const href = link.getAttribute("href") || "";
              if (!href || href.startsWith("#") || href.startsWith("javascript:")) return;
              showAdminLoading(link.dataset.loadingMessage || "画面を読み込んでいます...");
            }});
          }});
        }});
      </script>
    </head>
    <body>
      <header class="topbar">
        <div class="brand">
          <strong>{esc(variant.title)}</strong>
          <span>{esc(variant.subtitle)}</span>
        </div>
        <div class="api-note">接続先: {esc(base_url)} / config/settings.json の urls.rag_api_base_url で変更</div>
      </header>
      <div class="layout">
        {nav_html(active, base_url)}
        <main class="main">{content}</main>
      </div>
      <div class="admin-loading" role="status" aria-live="polite" aria-label="処理中">
        <div class="admin-loading-card">
          <span class="admin-loading-spinner" aria-hidden="true"></span>
          <strong id="admin-loading-label">API処理中...</strong>
          <small>RAG APIへ問い合わせています。画面を閉じずにお待ちください。</small>
        </div>
      </div>
    </body>
    </html>"""


def error_panel(title: str, error: object) -> str:
    return f"""
    <section class="panel">
      <h2>{esc(title)}</h2>
      <div class="notice">画面の表示に必要なデータを取得できませんでした。RAG APIが起動しているか確認してください。</div>
      <details><summary>開発者向け詳細</summary><div class="pre">{esc(error)}</div></details>
    </section>
    """


def kpi_card(label: str, value: object, note: str = "", tone: str = "") -> str:
    return f'<div class="kpi"><span>{esc(label)}</span><strong class="{esc(tone)}">{esc(value)}</strong><small>{esc(note)}</small></div>'


def badge(label: object, tone: str = "") -> str:
    return f'<span class="badge {esc(tone)}">{esc(label)}</span>'


def status_badge(status: object) -> str:
    text = STATUS_LABELS.get(str(status or ""), str(status or ""))
    tone = "ok" if status in {"resolved", "approved"} else "bad" if status == "open" else "muted"
    return badge(text, tone)


def priority_badge(priority: object) -> str:
    key = str(priority or "")
    tone = "bad" if key == "high" else "warn" if key == "medium" else "muted"
    return badge(PRIORITY_LABELS.get(key, key), tone)


def data_table(headers: list[str], rows: list[list[object]], empty: str) -> str:
    if not rows:
        return f'<div class="empty">{esc(empty)}</div>'
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def validity_summary(corpus: dict) -> str:
    counts = corpus.get("validity") or {}
    if not counts:
        return '<span class="muted">未集計</span>'
    labels = {
        "current": "現在有効",
        "unbounded": "期限なし",
        "not_started": "開始前",
        "expired": "期限切れ",
        "invalid_period": "日付不正",
    }
    parts = [
        f'{esc(label)}:{esc(counts.get(key, 0))}'
        for key, label in labels.items()
        if int_value(counts.get(key, 0))
    ]
    return '<span class="muted">' + " / ".join(parts or ["0"]) + "</span>"


def select_options(options: dict[str, str], selected: object) -> str:
    selected_text = str(selected or "")
    return "".join(
        f'<option value="{esc(value)}" {"selected" if str(value) == selected_text else ""}>{esc(label)}</option>'
        for value, label in options.items()
    )


def source_title(source: dict) -> str:
    return str(
        source.get("heading_path")
        or source.get("title")
        or source.get("source_file")
        or source.get("source_url")
        or source.get("child_id")
        or "根拠"
    )


def source_file_name(source: dict) -> str:
    text = str(source.get("source_file") or source.get("source_url") or "").strip()
    if not text:
        return ""
    if text.startswith(("http" + "://", "https" + "://")):
        return text
    return text.replace("\\", "/").split("/")[-1]


def score_label(score: object) -> str:
    value = float_value(score, -1.0)
    if value < 0:
        return "未取得"
    if value >= 0.75:
        return "高一致"
    if value >= 0.45:
        return "中一致"
    if value > 0:
        return "要確認"
    return "未取得"


def render_sources(sources: list[dict], target_textarea_id: str | None = None) -> str:
    if not sources:
        return '<div class="empty">回答時の根拠は記録されていません。ログのdebug情報、または根拠再検索で確認してください。</div>'
    cards = []
    for idx, source in enumerate(sources, start=1):
        payload = js_string(json.dumps([source], ensure_ascii=False))
        use_button = (
            f'<button class="btn secondary small" type="button" onclick="fillEvidence(\'{esc(target_textarea_id)}\', \'{payload}\')">この根拠をQAに使う</button>'
            if target_textarea_id
            else ""
        )
        child_text = source.get("child_text") or source.get("text") or ""
        parent_text = source.get("parent_text") or ""
        cards.append(f"""
        <article class="source-card">
          <header>
            <strong>根拠{idx}: {esc(short_text(source_title(source), 120))}</strong>
            <div class="right-actions">{badge(score_label(source.get("score")), "ok" if float_value(source.get("score")) >= 0.75 else "warn")}{use_button}</div>
          </header>
          <div class="source-meta">
            <span>参照元: {esc(short_text(source_file_name(source), 80))}</span>
            <span>Corpus: {esc(source.get("corpus_id"))}</span>
            <span>Score: {esc(source.get("score"))}</span>
          </div>
          <details open><summary>ヒットした本文</summary><div class="pre markdown-body">{markdown_block(child_text)}</div></details>
          <details><summary>親チャンク</summary><div class="pre markdown-body">{markdown_block(parent_text)}</div></details>
          <details><summary>内部情報</summary><div class="pre">{esc(json.dumps(source, ensure_ascii=False, indent=2))}</div></details>
        </article>
        """)
    return '<div class="source-grid">' + "".join(cards) + "</div>"


def svg_line_chart(items: list[dict], x_key: str, y_key: str, color: str = "#2557d6") -> str:
    values = [float_value(item.get(y_key)) for item in items]
    if not values:
        return '<div class="empty">まだ推移を表示できるログがありません。チャット利用後にここへ表示されます。</div>'
    width, height, pad = 720, 128, 12
    max_value = max(values) or 1.0
    step = (width - pad * 2) / max(1, len(values) - 1)
    points = []
    for idx, value in enumerate(values):
        x = pad + step * idx
        y = height - pad - ((height - pad * 2) * value / max_value)
        points.append(f"{x:.1f},{y:.1f}")
    return f"""
    <svg class="spark" viewBox="0 0 {width} {height}" preserveAspectRatio="none" aria-hidden="true">
      <polyline points="{' '.join(points)}" fill="none" stroke="{color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    """


def bar_chart(items: list[dict], label_key: str, value_key: str, limit: int = 10) -> str:
    rows = items[:limit]
    if not rows:
        return '<div class="empty">ランキング対象のデータがまだありません。</div>'
    max_value = max(float_value(item.get(value_key)) for item in rows) or 1.0
    html_rows = []
    for item in rows:
        value = float_value(item.get(value_key))
        width = max(2, min(100, value / max_value * 100))
        html_rows.append(
            f'<div class="bar-row"><span title="{esc(item.get(label_key))}">{esc(short_text(item.get(label_key), 26))}</span><div class="bar"><i style="width:{width:.1f}%"></i></div><strong>{esc(int(value))}</strong></div>'
        )
    return f'<div class="bars">{"".join(html_rows)}</div>'


def task_rows(tasks: list[dict], base_url: str, limit: int | None = None) -> list[list[object]]:
    selected = tasks[:limit] if limit else tasks
    rows = []
    for task in selected:
        links = task.get("links") or {}
        operations = []
        if task.get("task_type") == "hallucination_report":
            operations.append(f'<a class="btn small" href="{with_base("/reports", base_url)}">報告対応</a>')
        if task.get("task_type") in {"no_hit", "low_confidence"}:
            operations.append(f'<a class="btn secondary small" href="{with_base("/logs", base_url, query=task.get("question"), source_state="all")}">ログを見る</a>')
            operations.append(f'<a class="btn ghost small" href="{with_base("/qa-cache", base_url, question=task.get("question"))}">QA候補化</a>')
        if task.get("task_type") == "near_cache_miss":
            candidate_qa_id = task.get("qa_id") or task.get("cache_candidate_qa_id")
            log_id = task.get("log_id")
            if candidate_qa_id:
                operations.append(f'<a class="btn secondary small" href="{with_base("/qa-cache", base_url, qa_id=candidate_qa_id, question=task.get("question"))}">候補QAを開く</a>')
            operations.append(f'<a class="btn ghost small" href="{with_base("/qa-cache", base_url, question=task.get("question"))}">match-debug</a>')
            if log_id:
                operations.append(f'<a class="btn ghost small" href="{with_base("/logs", base_url, log_id=log_id)}">ログ詳細</a>')
            else:
                operations.append(f'<a class="btn ghost small" href="{with_base("/logs", base_url, query=task.get("question"))}">ログ詳細</a>')
            if candidate_qa_id:
                operations.append(f'<a class="btn ghost small" href="{with_base("/qa-cache", base_url, qa_id=candidate_qa_id, question=task.get("question"))}">alias追加</a>')
        if task.get("task_type") == "search_tag_candidate":
            operations.append(f'<a class="btn secondary small" href="{with_base("/search-tags", base_url)}">SearchTag編集</a>')
        if task.get("task_type") in {"index_warning", "eval_failed"}:
            operations.append(f'<a class="btn secondary small" href="{with_base("/index-eval", base_url)}">文書/評価</a>')
        if not operations and links:
            operations.append(f'<a class="btn ghost small" href="{with_base("/logs", base_url, query=task.get("question"))}">確認</a>')
        rows.append([
            priority_badge(task.get("priority")),
            esc(TASK_LABELS.get(str(task.get("task_type") or ""), task.get("task_type"))),
            f'<strong>{esc(task.get("title"))}</strong><br><span class="muted">{esc(short_text(task.get("question"), 130))}</span>',
            esc(task.get("count")),
            esc(task.get("last_seen_at") or "-"),
            esc(task.get("related_corpus") or "-"),
            esc(task.get("max_score") if task.get("max_score") is not None else "-"),
            esc(task.get("reason_hint")),
            esc(task.get("suggested_action")),
            '<div class="right-actions">' + "".join(operations) + "</div>",
        ])
    return rows


async def dashboard_content(base_url: str) -> str:
    try:
        summary = await api_get(base_url, "/admin/ops/summary", {"days": 7, "top_n": 20})
        actions = await api_get(base_url, "/admin/actions/improvement-candidates", {"days": 7, "top_n": 20})
    except Exception as exc:
        return error_panel("ダッシュボードを表示できません", exc)
    health = summary.get("health", {})
    kpis = summary.get("kpis", {})
    dashboard = summary.get("dashboard", {})
    alerts = health.get("alerts", [])
    status = str(health.get("status") or "OK")
    status_class = "status-bad" if "異常" in status else "status-warn" if "警告" in status else "status-ok"
    alert_html = (
        '<div class="alert-list">' + "".join(f'<div class="notice"><strong>{esc(a.get("message"))}</strong><br>{esc(a.get("action"))}</div>' for a in alerts) + "</div>"
        if alerts
        else '<p class="lead">重大な未対応項目はありません。下段の頻出質問や評価結果を見て、次の改善候補を確認してください。</p>'
    )
    top_tasks = task_rows(actions.get("tasks", []), base_url, limit=10)
    return f"""
    <section class="status-banner">
      <div>
        <div class="status-title {status_class}">状態: {esc(status)}</div>
        <div class="muted">最終ログ: {esc(health.get("last_log_at") or "未記録")}</div>
        {alert_html}
      </div>
      <div class="right-actions">
        <a class="btn" href="{with_base("/queue", base_url)}">改善キューを見る</a>
        <a class="btn secondary" href="{with_base("/reports", base_url)}">報告対応へ</a>
      </div>
    </section>
    <section class="grid-6">
      {kpi_card("期間内質問数", kpis.get("questions_in_window", 0), f'セッション {kpis.get("sessions_in_window", 0)}')}
      {kpi_card("根拠なし率", percent_text(kpis.get("no_hit_rate", 0)), f'{kpis.get("no_hit_count", 0)}件')}
      {kpi_card("低信頼率", percent_text(kpis.get("low_confidence_rate", 0)), f'{kpis.get("low_confidence_count", 0)}件')}
      {kpi_card("未対応報告", kpis.get("open_reports", 0), "ユーザー報告")}
      {kpi_card("QAヒット率", percent_text(kpis.get("qa_cache_hit_rate", 0)), f'ヒット {kpis.get("qa_cache_hits", 0)}件')}
      {kpi_card("Index警告", kpis.get("chunk_warnings", 0), f'タグ網羅率 {percent_text(kpis.get("tag_coverage_rate", 0))}')}
    </section>
    <section class="panel">
      <h2>最優先対応</h2>
      {data_table(["優先度","種別","対象","件数","最終発生","Corpus","最大スコア","原因候補","推奨対応","操作"], top_tasks, "今すぐ対応すべき項目はありません。日次の評価結果と頻出質問を確認してください。")}
    </section>
    <section class="grid-2">
      <div class="panel"><h2>日次質問数</h2>{svg_line_chart(dashboard.get("daily_questions", []), "date", "questions")}</div>
      <div class="panel"><h2>根拠なし/低信頼の推移</h2>{bar_chart(dashboard.get("quality", {}).get("daily_quality", []), "date", "no_hit")}</div>
    </section>
    <section class="grid-3">
      <div class="panel"><h2>頻出質問</h2>{bar_chart(dashboard.get("top_questions", []), "question", "hits")}</div>
      <div class="panel"><h2>頻出参照チャンク</h2>{bar_chart(dashboard.get("top_hit_chunks", []), "heading_path", "hits")}</div>
      <div class="panel"><h2>頻出参照ファイル</h2>{bar_chart(dashboard.get("top_source_files", []), "source_file", "hits")}</div>
    </section>
    """


async def queue_content(base_url: str, request: Request) -> str:
    q = request.query_params
    task_type = q.get("task_type") or ""
    priority = q.get("priority") or ""
    try:
        data = await api_get(base_url, "/admin/actions/improvement-candidates", {"days": q.get("days") or 30, "top_n": 50})
    except Exception as exc:
        return error_panel("改善キューを表示できません", exc)
    tasks = data.get("tasks", [])
    if task_type:
        tasks = [task for task in tasks if str(task.get("task_type") or "") == task_type]
    if priority:
        tasks = [task for task in tasks if str(task.get("priority") or "") == priority]
    search_tag_settings = load_settings().get("search_tags", {})
    search_tag_panel = (
        f'<div class="panel"><h2>SearchTag候補</h2>{bar_chart(data.get("search_tag_candidates", []), "heading_path", "tag_count")}</div>'
        if bool(search_tag_settings.get("show_improvement_candidates", False))
        else ""
    )
    return f"""
    <section class="panel">
      <h2>改善キュー</h2>
      <p class="lead">未対応報告、根拠なし、低信頼、頻出質問、Index警告、評価失敗を優先度順にまとめています。まず上から処理してください。</p>
      <form class="filters" method="get" action="/queue">
        <div class="field"><label>種別</label><select name="task_type">{select_options({"":"すべて", **TASK_LABELS}, task_type)}</select></div>
        <div class="field"><label>優先度</label><select name="priority">{select_options({"":"すべて","high":"高","medium":"中","low":"低"}, priority)}</select></div>
        <div class="field"><label>期間(日)</label><input name="days" value="{esc(q.get("days") or 30)}"></div>
        <button class="btn secondary" type="submit">絞り込む</button>
      </form>
      {data_table(["優先度","種別","対象","件数","最終発生","Corpus","最大スコア","原因候補","推奨対応","操作"], task_rows(tasks, base_url), "改善キューに表示する項目はありません。ログが増えると候補が表示されます。")}
    </section>
    <section class="grid-3">
      <div class="panel"><h2>No Hit</h2>{bar_chart(data.get("no_hit_logs", []), "question_preview", "source_count")}</div>
      <div class="panel"><h2>低信頼</h2>{bar_chart(data.get("low_confidence_logs", []), "question_preview", "max_score")}</div>
      {search_tag_panel}
    </section>
    """


async def evidence_search_panel(base_url: str, query: str, target_id: str) -> str:
    if not query:
        return '<div class="empty">質問やキーワードを入力すると、QA登録に使える根拠候補を検索できます。</div>'
    try:
        data = await api_post(base_url, "/admin/evidence/search", {"query": query, "top_k": 8}, timeout=180)
    except Exception as exc:
        return f'<div class="notice">根拠検索に失敗しました。<details><summary>詳細</summary><div class="pre">{esc(exc)}</div></details></div>'
    return render_sources(data.get("sources", []), target_id)


async def reports_content(base_url: str, request: Request) -> str:
    q = request.query_params
    status = q.get("status") or "open"
    report_id = q.get("report_id")
    try:
        params = {"limit": 100}
        if status != "all":
            params["status"] = status
        data = await api_get(base_url, "/admin/reports", params)
        reports = data.get("reports", [])
        selected_id = report_id
        detail = await api_get(base_url, f"/admin/reports/{selected_id}") if selected_id else {}
    except Exception as exc:
        return error_panel("報告対応を表示できません", exc)
    rows = []
    for report in reports:
        rows.append([
            esc(report.get("id")),
            status_badge(report.get("status")),
            esc(report.get("created_at")),
            esc(short_text(report.get("question"), 120)),
            esc(short_text(report.get("comment"), 80)),
            f'<a class="btn small" href="{with_base("/reports", base_url, status=status, report_id=report.get("id"))}">詳細</a>',
        ])
    detail_modal = report_detail_modal(base_url, detail, {"status": status}) if detail else ""
    return f"""
    <section class="panel">
      <h2>報告対応</h2>
      <p class="lead">ハルシネーション疑いを、原因分類と対応方法に分けて処理します。必要なら根拠検索から承認QAへ登録してください。</p>
      <form class="filters" method="get" action="/reports">
        <div class="field"><label>状態</label><select name="status">{select_options({"open":"未対応","resolved":"対応済み","ignored":"対応不要","all":"すべて"}, status)}</select></div>
        <button class="btn secondary" type="submit">表示</button>
      </form>
    </section>
    <section class="panel">
      <h2>報告一覧</h2>
      {data_table(["ID","状態","日時","質問","ユーザーコメント","操作"], rows, "この条件の報告はありません。")}
    </section>
    {detail_modal}
    """


def report_detail_modal(base_url: str, detail: dict, close_params: dict | None = None) -> str:
    report = detail.get("report") or {}
    if not report:
        return ""
    close_href = with_base("/reports", base_url, **(close_params or {}))
    return f"""
    <div class="admin-modal" role="dialog" aria-modal="true" aria-label="報告詳細">
      <a class="admin-modal-backdrop" href="{close_href}" aria-label="閉じる"></a>
      <section class="admin-modal-card log-detail-modal">
        <header class="admin-modal-head">
          <div>
            <h2>報告詳細 #{esc(report.get("id"))}</h2>
            <p class="lead">質問・回答・ユーザーコメント・原因分類・承認QA登録を広い画面で確認します。</p>
          </div>
          <a class="modal-close" href="{close_href}">閉じる</a>
        </header>
        <div class="admin-modal-body">
          {report_detail_html(base_url, detail)}
        </div>
      </section>
    </div>
    """


def report_detail_html(base_url: str, detail: dict) -> str:
    report = detail.get("report") or {}
    log = detail.get("log") or {}
    sources = log.get("sources") or []
    similar_logs = detail.get("similar_logs") or []
    evidence_json = esc(json.dumps(sources[:1], ensure_ascii=False, indent=2))
    similar_rows = [
        [esc(row.get("id")), esc(row.get("created_at")), esc(short_text(row.get("question"), 100)), esc(row.get("source_count")), esc(row.get("max_score"))]
        for row in similar_logs[:8]
    ]
    return f"""
    <div class="panel">
      <h2>報告詳細 #{esc(report.get("id"))}</h2>
      <div class="source-meta">
        <span>状態: {STATUS_LABELS.get(str(report.get("status") or ""), report.get("status"))}</span>
        <span>log_id: {esc(report.get("log_id") or "-")}</span>
        <span>作成: {esc(report.get("created_at"))}</span>
      </div>
      <h3>質問</h3><div class="pre">{text_block(report.get("question"))}</div>
      <h3>実際の回答</h3><div class="pre markdown-body">{markdown_block(report.get("answer"))}</div>
      <h3>ユーザーコメント</h3><div class="pre">{text_block(report.get("comment") or "コメントなし")}</div>
      <h3>回答時の根拠</h3>{render_sources(sources, "report_evidence_json")}
      <h3>原因分類と対応メモ</h3>
      <form method="post" action="/report-analysis">
        <input type="hidden" name="report_id" value="{esc(report.get("id"))}">
        <div class="grid-2">
          <div class="field"><label>状態</label><select name="status">{select_options({"open":"未対応","resolved":"対応済み","ignored":"対応不要"}, report.get("status"))}</select></div>
          <div class="field"><label>原因分類</label><select name="issue_type">{select_options(ISSUE_TYPES, report.get("issue_type") or "")}</select></div>
          <div class="field"><label>対応方法</label><select name="resolution_type">{select_options(RESOLUTION_TYPES, report.get("resolution_type") or "")}</select></div>
          <div class="field"><label>紐づけ子チャンクID</label><input name="linked_child_id" value="{esc(report.get("linked_child_id") or "")}"></div>
        </div>
        <div class="field"><label>管理メモ</label><textarea name="admin_memo">{esc(report.get("admin_memo") or "")}</textarea></div>
        <button class="btn" type="submit">分類とメモを保存</button>
      </form>
      <h3>承認QA登録</h3>
      <form method="post" action="/qa-register">
        <input type="hidden" name="source_report_id" value="{esc(report.get("id"))}">
        <div class="field"><label>質問</label><textarea name="question">{esc(report.get("question"))}</textarea></div>
        <div class="field"><label>正しい回答</label><textarea name="answer"></textarea></div>
        <details open><summary>選択済み根拠JSON</summary><textarea id="report_evidence_json" name="evidence_json">{evidence_json or "[]"}</textarea></details>
        <div class="field"><label>承認者</label><input name="approved_by" value="admin"></div>
        <div class="field"><label>メモ</label><textarea name="memo">報告 #{esc(report.get("id"))} から登録</textarea></div>
        <button class="btn" type="submit">承認QAとして登録</button>
      </form>
      <h3>類似ログ</h3>
      {data_table(["ID","日時","質問","根拠数","最大スコア"], similar_rows, "類似ログは見つかりません。")}
    </div>
    """


async def qa_content(base_url: str, request: Request, generated: dict | None = None) -> str:
    q = request.query_params
    status = q.get("status") or "all"
    keyword = q.get("q") or ""
    qa_id = q.get("qa_id")
    alias_qa_id = q.get("alias_qa_id")
    generated = generated or {}
    draft_question = str(generated.get("question") or q.get("question") or keyword)
    draft_answer = str(generated.get("answer") or q.get("answer") or "")
    draft_sources = generated.get("sources") or []
    draft_evidence_json = esc(json.dumps(draft_sources, ensure_ascii=False, indent=2)) if draft_sources else "[]"
    try:
        summary = await api_get(base_url, "/admin/qa-cache/summary", {"limit": 20})
        data = await api_get(base_url, "/admin/qa-cache", {"status": status, "limit": 200})
        items = data.get("items", [])
    except Exception as exc:
        return error_panel("承認QAを表示できません", exc)
    alias_summary = summary.get("aliases", {})
    alias_index = alias_summary.get("alias_index") or {}
    if keyword:
        folded = keyword.casefold()
        items = [item for item in items if folded in str(item.get("question") or "").casefold() or folded in str(item.get("answer") or "").casefold()]
    selected = next((item for item in items if str(item.get("id")) == str(qa_id)), None) if qa_id else None
    if qa_id and selected is None:
        try:
            selected = await api_get(base_url, f"/admin/qa-cache/{int_value(qa_id)}")
        except Exception:
            selected = None
    alias_selected = next((item for item in items if str(item.get("id")) == str(alias_qa_id)), None) if alias_qa_id else None
    if alias_qa_id and alias_selected is None:
        try:
            alias_selected = await api_get(base_url, f"/admin/qa-cache/{int_value(alias_qa_id)}")
        except Exception:
            alias_selected = None
    similar_html = await qa_similar_panel(base_url, draft_question) if draft_question else '<div class="empty">質問を入力すると、類似QAと現在のキャッシュ判定を確認できます。</div>'
    evidence_html = await evidence_search_panel(base_url, draft_question, "qa_evidence_json") if draft_question else '<div class="empty">質問を入力すると、根拠候補を検索できます。</div>'
    selected_detail_html = await qa_detail_modal(base_url, selected) if selected else ""
    alias_modal_html = await qa_alias_modal(base_url, alias_selected) if alias_selected else ""
    generated_notice = '<div class="notice">生成した回答と根拠を下の登録フォームへ反映しました。内容と根拠を確認してから登録してください。</div>' if (draft_answer or draft_sources) else ""
    rows = [
        [
            esc(item.get("id")),
            status_badge(item.get("status")),
            esc(short_text(item.get("question"), 120)),
            esc(item.get("evidence_count")),
            esc(item.get("corpus_version")),
            esc(item.get("index_version")),
            f'<a class="btn small" href="{with_base("/qa-cache", base_url, status=status, q=keyword, qa_id=item.get("id"))}">詳細</a>',
        ]
        for item in items
    ]
    return f"""
    <section class="panel">
      <h2>承認QA</h2>
      <p class="lead">よく聞かれる質問や正しい回答が確定したものを、根拠付きでLLMに渡さず回答するための管理画面です。</p>
      <div class="grid-4">
        {kpi_card("総QA", summary.get("total", 0))}
        {kpi_card("有効QA", sum(int_value(row.get("count")) for row in summary.get("by_status", []) if row.get("status") == "approved"))}
        {kpi_card("無効QA", sum(int_value(row.get("count")) for row in summary.get("by_status", []) if row.get("status") == "disabled"))}
        {kpi_card("Version", short_text((summary.get("by_version") or [{}])[0].get("index_version"), 24) if summary.get("by_version") else "-")}
      </div>
      <div class="grid-4">
        {kpi_card("active alias", alias_summary.get("active", 0))}
        {kpi_card("disabled alias", alias_summary.get("disabled", 0))}
        {kpi_card("alias未生成QA", alias_summary.get("original_missing_qa", 0))}
        {kpi_card("alias index", alias_index.get("count", 0))}
      </div>
      <form class="filters" method="get" action="/qa-cache">
        <div class="field"><label>キーワード</label><input name="q" value="{esc(keyword)}"></div>
        <div class="field"><label>状態</label><select name="status">{select_options({"all":"すべて","approved":"有効","disabled":"無効"}, status)}</select></div>
        <button class="btn secondary" type="submit">検索</button>
      </form>
    </section>
    <section class="panel">
      <h2>QA一覧</h2>
      {data_table(["ID","状態","質問","根拠数","Corpus","Index","操作"], rows, "承認QAはまだ登録されていません。報告対応または根拠検索から登録してください。")}
    </section>
    {selected_detail_html}
    {alias_modal_html}
    <section class="panel">
      <h2>QA候補作成・類似確認</h2>
      <p class="lead">質問を起点に、既存QAとの近さ、RAG回答生成、根拠候補を同じ場所で確認します。</p>
      <div class="qa-workspace-grid">
        <div class="qa-workspace-main">
          <form method="post" action="/qa-generate-answer" data-loading-message="回答と根拠を生成しています...">
            <div class="field"><label>質問</label><textarea name="question">{esc(draft_question)}</textarea></div>
            <div class="field"><label>根拠数</label><input name="top_k" value="8"></div>
            <div class="right-actions">
              <button class="btn" type="submit">回答と根拠を生成</button>
              <button class="btn secondary" type="submit" formaction="/qa-cache" formmethod="get" data-loading-message="類似QAを確認しています...">類似QAだけ確認</button>
            </div>
          </form>
          {generated_notice}
          <div class="qa-similar-box">
            <h3>類似QA / キャッシュ判定</h3>
            {similar_html}
          </div>
        </div>
        <div class="qa-workspace-side">
          <h3>根拠候補</h3>
          <p class="lead">候補の「追加」ボタンで、下の登録フォームの根拠JSONへ追加できます。</p>
          <form method="get" action="/qa-cache" data-loading-message="根拠候補を検索しています...">
            <div class="field"><label>根拠検索クエリ</label><textarea name="question">{esc(draft_question)}</textarea></div>
            <button class="btn secondary" type="submit">根拠候補を検索</button>
          </form>
          {evidence_html}
        </div>
      </div>
    </section>
    <section class="panel">
      <h2>新規QA登録</h2>
      <p class="lead">質問、回答、根拠を確認してから承認QAとして登録します。根拠候補は検索結果のボタンで下のJSONに追加できます。</p>
      <form method="post" action="/qa-register" data-loading-message="承認QAを登録しています...">
        <div class="field"><label>質問</label><textarea name="question">{esc(draft_question)}</textarea></div>
        <div class="field"><label>回答</label><textarea name="answer">{esc(draft_answer)}</textarea></div>
        <div class="notice">根拠は1件以上選ぶ運用を推奨します。根拠なしで登録する場合は、管理メモに理由を残してください。</div>
        <details open><summary>選択済み根拠JSON</summary><textarea id="qa_evidence_json" name="evidence_json">{draft_evidence_json}</textarea></details>
        <div class="field"><label>承認者</label><input name="approved_by" value="admin"></div>
        <div class="field"><label>管理メモ</label><textarea name="memo"></textarea></div>
        <button class="btn" type="submit">承認QAを登録</button>
      </form>
    </section>
    """

async def qa_similar_panel(base_url: str, question: str) -> str:
    try:
        debug = await api_post(base_url, "/admin/qa-cache/match-debug", {"question": question, "top_n": 8}, timeout=180)
    except Exception as exc:
        return f'<div class="notice">match-debugに失敗しました。EmbeddingサーバーとRAG APIを確認してください。<details><summary>詳細</summary><div class="pre">{esc(exc)}</div></details></div>'
    best = debug.get("best") or {}
    decision = str(debug.get("decision") or "miss")
    reason = str(debug.get("miss_reason") or "")
    match_html = (
        f'<div class="notice">判定: {esc(decision)} / QA#{esc(best.get("qa_id"))} alias#{esc(best.get("alias_id"))} similarity {float_value(best.get("similarity")):.3f} margin {float_value(best.get("margin")):.3f}</div>'
        if best
        else f'<div class="empty">判定: {esc(decision)} / reason={esc(reason)}</div>'
    )
    rows = [
        [
            esc(item.get("qa_id")),
            esc(item.get("alias_id")),
            esc(item.get("alias_type")),
            f'{float_value(item.get("similarity")):.3f}',
            f'{float_value(item.get("margin")):.3f}',
            esc(short_text(item.get("alias_text"), 90)),
            esc(short_text(item.get("question"), 80)),
            "hit" if item.get("would_hit") else "-",
        ]
        for item in debug.get("candidates", [])
    ]
    return match_html + data_table(["QA", "Alias", "種別", "similarity", "margin", "alias_text", "正式質問", "hit"], rows, "alias候補は見つかりません。")


def alias_table(aliases: list[dict], qa_id: int) -> str:
    if not aliases:
        return '<div class="empty">aliasはまだありません。</div>'
    rows = []
    for alias in aliases:
        status = str(alias.get("status") or "")
        risk_flags = alias.get("risk_flags") or alias.get("risk_flags_json") or ""
        risk_text = risk_flags if isinstance(risk_flags, str) else ", ".join(str(x) for x in risk_flags)
        row_class = "alias-row " + ("alias-disabled" if status == "disabled" else "alias-active")
        if risk_text:
            row_class += " alias-risk"
        next_status = "disabled" if status == "active" else "active"
        rows.append(
            f"""
            <tr class="{esc(row_class)}">
              <td>{esc(alias.get("id"))}</td>
              <td>{status_badge(status)}</td>
              <td>{esc(alias.get("alias_type"))}</td>
              <td><strong>{esc(short_text(alias.get("alias_text"), 120))}</strong>{f'<div class="notice">{esc(risk_text)}</div>' if risk_text else ''}</td>
              <td>{esc(alias.get("updated_at"))}</td>
              <td>
                <form method="post" action="/qa-alias-update" class="inline-form">
                  <input type="hidden" name="qa_id" value="{qa_id}">
                  <input type="hidden" name="alias_id" value="{esc(alias.get("id"))}">
                  <input type="hidden" name="status" value="{esc(next_status)}">
                  <button class="btn small secondary" type="submit">{'無効化' if status == 'active' else '有効化'}</button>
                </form>
              </td>
            </tr>
            """
        )
    return f"""
    <div class="table-wrap">
      <table>
        <thead><tr><th>ID</th><th>状態</th><th>種別</th><th>別名質問</th><th>更新日時</th><th>操作</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


async def qa_detail_modal(base_url: str, item: dict) -> str:
    return f"""
    <div class="admin-modal" role="dialog" aria-modal="true" aria-label="QA詳細">
      <a class="admin-modal-backdrop" href="{with_base('/qa-cache', base_url)}" aria-label="閉じる"></a>
      <section class="admin-modal-card qa-detail-modal">
        <header class="admin-modal-head">
          <div>
            <h2>QA #{esc(item.get("id"))}</h2>
            <p class="lead">質問・回答・根拠を広い画面で確認、編集します。Aliasは下部のボタンから別画面で管理します。</p>
          </div>
          <a class="modal-close" href="{with_base('/qa-cache', base_url)}">閉じる</a>
        </header>
        <div class="admin-modal-body">
          <div class="qa-detail-wide">
            {qa_detail_panel(base_url, item)}
          </div>
        </div>
      </section>
    </div>
    """


async def qa_alias_modal(base_url: str, item: dict) -> str:
    qa_id = int_value(item.get("id"))
    return f"""
    <div class="admin-modal" role="dialog" aria-modal="true" aria-label="Alias管理">
      <a class="admin-modal-backdrop" href="{with_base('/qa-cache', base_url, qa_id=qa_id)}" aria-label="閉じる"></a>
      <section class="admin-modal-card alias-modal">
        <header class="admin-modal-head">
          <div>
            <h2>Alias管理 / QA #{esc(qa_id)}</h2>
            <p class="lead">検索入口になる別名質問を管理します。回答本体は承認QA詳細で編集してください。</p>
          </div>
          <a class="modal-close" href="{with_base('/qa-cache', base_url, qa_id=qa_id)}">QA詳細へ戻る</a>
        </header>
        <div class="admin-modal-body">
          {await qa_alias_panel(base_url, item)}
        </div>
      </section>
    </div>
    """

async def qa_alias_panel(base_url: str, item: dict) -> str:
    qa_id = int_value(item.get("id"))
    try:
        data = await api_get(base_url, f"/admin/qa-cache/{qa_id}/aliases")
    except Exception as exc:
        return f'<div class="panel"><h2>Alias管理</h2><div class="notice">alias一覧を取得できません。<details><summary>詳細</summary><div class="pre">{esc(exc)}</div></details></div></div>'
    aliases = data.get("aliases") or []
    rows = [
        [
            esc(alias.get("id")),
            status_badge(alias.get("status")),
            esc(alias.get("alias_type")),
            esc(short_text(alias.get("alias_text"), 120)),
            esc(alias.get("updated_at")),
            f"""
            <form method="post" action="/qa-alias-update" class="inline-form">
              <input type="hidden" name="qa_id" value="{qa_id}">
              <input type="hidden" name="alias_id" value="{esc(alias.get("id"))}">
              <input type="hidden" name="status" value="{'disabled' if alias.get("status") == 'active' else 'active'}">
              <button class="btn small secondary" type="submit">{'無効化' if alias.get("status") == 'active' else '有効化'}</button>
            </form>
            """,
        ]
        for alias in aliases
    ]
    return f"""
    <div class="panel">
      <h2>Alias管理</h2>
      <p class="lead">approved_qaは回答本体、aliasは検索入口です。税目違い・制度違い・条件追加のaliasは無効化してください。</p>
      {alias_table(aliases, qa_id)}
      <form method="post" action="/qa-alias-add" style="margin-top:14px">
        <input type="hidden" name="qa_id" value="{qa_id}">
        <div class="field"><label>手動alias 1行1件</label><textarea name="aliases"></textarea></div>
        <button class="btn secondary" type="submit">手動aliasを追加</button>
      </form>
      <div class="grid-2" style="margin-top:14px">
        <form method="post" action="/qa-alias-generate" data-loading-message="Alias候補を生成しています...">
          <input type="hidden" name="qa_id" value="{qa_id}">
          <input type="hidden" name="dry_run" value="true">
          <div class="field"><label>最大件数</label><input name="max_aliases" value="8"></div>
          <button class="btn ghost" type="submit">LLM alias dry-run</button>
        </form>
        <form method="post" action="/qa-alias-generate" data-loading-message="Aliasを生成して保存しています...">
          <input type="hidden" name="qa_id" value="{qa_id}">
          <div class="field"><label>最大件数</label><input name="max_aliases" value="8"></div>
          <label><input type="checkbox" name="replace_existing_generated" value="true"> 既存LLM aliasを置換</label><br><br>
          <button class="btn" type="submit">LLM aliasを生成して保存</button>
        </form>
      </div>
      <div class="grid-2" style="margin-top:14px">
        <form method="post" action="/qa-alias-backfill">
          <input type="hidden" name="ensure_original" value="true">
          <input type="hidden" name="dry_run" value="true">
          <div class="field"><label>limit</label><input name="limit" value="50"></div>
          <button class="btn secondary" type="submit">original alias補完を確認</button>
        </form>
        <form method="post" action="/qa-cache-reload-index" data-loading-message="Alias indexを再読込しています...">
          <button class="btn ghost" type="submit">alias index reload</button>
        </form>
      </div>
      <details style="margin-top:14px">
        <summary>original alias補完を実行する</summary>
        <div class="notice">運用DBに書き込む前に、必ずSQLiteのバックアップを取得してください。推奨: python scripts/09_backfill_approved_qa_aliases.py --apply --backup --ensure-original</div>
        <form method="post" action="/qa-alias-backfill">
          <input type="hidden" name="ensure_original" value="true">
          <input type="hidden" name="dry_run" value="false">
          <div class="field"><label>limit</label><input name="limit" value="50"></div>
          <button class="btn danger" type="submit">original alias補完を実行</button>
        </form>
      </details>
    </div>
    """


def qa_detail_panel(base_url: str, item: dict) -> str:
    evidence = esc(json.dumps(item.get("evidence", []), ensure_ascii=False, indent=2))
    return f"""
    <div class="panel">
      <h2>QA詳細 #{esc(item.get("id"))}</h2>
      <form method="post" action="/qa-update">
        <input type="hidden" name="qa_id" value="{esc(item.get("id"))}">
        <div class="field"><label>状態</label><select name="status">{select_options({"approved":"有効","disabled":"無効"}, item.get("status"))}</select></div>
        <div class="field"><label>質問</label><textarea name="question">{esc(item.get("question"))}</textarea></div>
        <div class="field">
          <label>回答</label>
          <div class="qa-answer-preview markdown-body">{markdown_block(item.get("answer"))}</div>
        </div>
        <details><summary>回答Markdownを編集</summary><textarea name="answer">{esc(item.get("answer"))}</textarea></details>
        <details><summary>根拠JSON</summary><textarea name="evidence_json">{evidence}</textarea></details>
        <div class="grid-2">
          <div class="field"><label>Corpus version</label><input name="corpus_version" value="{esc(item.get("corpus_version"))}"></div>
          <div class="field"><label>Index version</label><input name="index_version" value="{esc(item.get("index_version"))}"></div>
        </div>
        <div class="field"><label>承認者</label><input name="approved_by" value="{esc(item.get("approved_by") or "admin")}"></div>
        <div class="field"><label>管理メモ</label><textarea name="memo">{esc(item.get("memo"))}</textarea></div>
        <div class="right-actions">
          <a class="btn secondary" href="{with_base('/qa-cache', base_url, alias_qa_id=item.get('id'))}">Alias管理を開く</a>
          <button class="btn" type="submit">更新</button>
        </div>
      </form>
      <form method="post" action="/qa-disable" style="margin-top:10px">
        <input type="hidden" name="qa_id" value="{esc(item.get("id"))}">
        <button class="btn danger" type="submit">このQAを無効化</button>
      </form>
    </div>
    """


async def logs_content(base_url: str, request: Request) -> str:
    q = request.query_params
    params = {
        "limit": q.get("limit") or 100,
        "days": q.get("days") or 30,
        "query": q.get("query") or None,
        "corpus_id": q.get("corpus_id") or None,
        "source_state": q.get("source_state") or "all",
        "answer_source": q.get("answer_source") or None,
        "qa_cache_id": q.get("qa_cache_id") or None,
        "session_id": q.get("session_id") or None,
        "log_id": q.get("log_id") or None,
        "min_score": q.get("min_score") or None,
        "max_score": q.get("max_score") or None,
    }
    try:
        data = await api_get(base_url, "/admin/logs/search", params)
    except Exception as exc:
        return error_panel("ログ探索を表示できません", exc)
    logs = data.get("logs", [])
    rows = []
    for row in logs:
        quality = row.get("quality_state") or ("no_sources" if int_value(row.get("source_count")) == 0 else "normal")
        tone = "bad" if quality == "no_sources" else "warn" if quality == "low_confidence" else "ok"
        rows.append([
            esc(row.get("id")),
            esc(row.get("created_at")),
            badge(quality, tone),
            esc(row.get("answer_source") or "rag"),
            esc(short_text(row.get("question"), 130)),
            esc(row.get("source_count")),
            esc(row.get("max_score")),
            esc(row.get("qa_cache_id") or "-"),
            f'<a class="btn small" href="{with_base("/logs", base_url, **{**{k: v for k, v in params.items() if v}, "log_id": row.get("id")})}">詳細</a>',
        ])
    selected = None
    if q.get("log_id"):
        selected = next((row for row in logs if str(row.get("id")) == str(q.get("log_id"))), None)
    close_params = {k: v for k, v in params.items() if v and k != "log_id"}
    modal_html = log_detail_modal(base_url, selected, close_params) if selected else ""
    return f"""
    <section class="panel">
      <h2>ログ探索</h2>
      <p class="lead">改善キューや報告対応からドリルダウンして、回答元・根拠数・最大スコア・QAキャッシュを確認します。</p>
      <form class="filters" method="get" action="/logs">
        <div class="field"><label>キーワード</label><input name="query" value="{esc(q.get("query") or "")}"></div>
        <div class="field"><label>期間(日)</label><input name="days" value="{esc(q.get("days") or 30)}"></div>
        <div class="field"><label>品質</label><select name="source_state">{select_options({"all":"すべて","with_sources":"根拠あり","no_sources":"根拠なし","low_confidence":"低信頼"}, q.get("source_state") or "all")}</select></div>
        <div class="field"><label>回答元</label><select name="answer_source">{select_options({"":"すべて","rag":"RAG","approved_qa_cache":"承認QA"}, q.get("answer_source") or "")}</select></div>
        <div class="field"><label>Corpus</label><input name="corpus_id" value="{esc(q.get("corpus_id") or "")}"></div>
        <div class="field"><label>QA ID</label><input name="qa_cache_id" value="{esc(q.get("qa_cache_id") or "")}"></div>
        <button class="btn secondary" type="submit">検索</button>
      </form>
    </section>
    <section class="panel"><h2>ログ一覧</h2>{data_table(["ID","日時","品質","回答元","質問","根拠数","最大スコア","QA ID","操作"], rows, "条件に一致するログはありません。")}</section>
    {modal_html}
    """


def log_detail_modal(base_url: str, row: dict | None, close_params: dict | None = None) -> str:
    if not row:
        return ""
    close_href = with_base("/logs", base_url, **(close_params or {}))
    return f"""
    <div class="admin-modal" role="dialog" aria-modal="true" aria-label="ログ詳細">
      <a class="admin-modal-backdrop" href="{close_href}" aria-label="閉じる"></a>
      <section class="admin-modal-card log-detail-modal">
        <header class="admin-modal-head">
          <div>
            <h2>ログ詳細 #{esc(row.get("id"))}</h2>
            <p class="lead">質問、回答、根拠、QAキャッシュ候補、debug情報を大きな画面で確認します。</p>
          </div>
          <a class="modal-close" href="{close_href}">閉じる</a>
        </header>
        <div class="admin-modal-body">
          {log_detail_panel(row, base_url)}
        </div>
      </section>
    </div>
    """


def log_detail_panel(row: dict | None, base_url: str = "") -> str:
    if not row:
        return ""
    evidence_json = esc(json.dumps(row.get("sources") or [], ensure_ascii=False, indent=2))
    log_id = row.get("id")
    return_to = with_base("/logs", base_url, log_id=log_id)
    register_notice = (
        '<div class="notice">このログには根拠がありません。承認QAとして登録する場合は、根拠検索で裏取りしてから登録する運用を推奨します。</div>'
        if int_value(row.get("source_count")) == 0
        else '<div class="notice">ログの質問・回答・回答時根拠を下のフォームへ入れています。内容と根拠を確認してから登録してください。</div>'
    )
    return f"""
    <section class="panel">
      <h2>ログ詳細 #{esc(row.get("id"))}</h2>
      <div class="source-meta">
        <span>回答元: {esc(row.get("answer_source") or "rag")}</span>
        <span>根拠数: {esc(row.get("source_count"))}</span>
        <span>最大スコア: {esc(row.get("max_score"))}</span>
        <span>QA ID: {esc(row.get("qa_cache_id") or "-")}</span>
        <span>候補QA: {esc(row.get("cache_candidate_qa_id") or "-")}</span>
        <span>候補alias: {esc(row.get("cache_candidate_alias_id") or "-")}</span>
        <span>候補similarity: {esc(row.get("cache_candidate_similarity") or "-")}</span>
        <span>miss理由: {esc(row.get("cache_miss_reason") or "-")}</span>
        <span>match方法: {esc(row.get("cache_match_method") or "-")}</span>
      </div>
      <h3>質問</h3><div class="pre">{text_block(row.get("question"))}</div>
      <h3>回答</h3><div class="pre markdown-body">{markdown_block(row.get("answer"))}</div>
      <h3>根拠</h3>{render_sources(row.get("sources") or [])}
      <h3>承認QA登録</h3>
      {register_notice}
      <form method="post" action="/qa-register" data-loading-message="ログから承認QAを登録しています...">
        <div class="field"><label>質問</label><textarea name="question">{esc(row.get("question"))}</textarea></div>
        <div class="field"><label>回答</label><textarea name="answer">{esc(row.get("answer"))}</textarea></div>
        <details open><summary>回答時根拠JSON</summary><textarea name="evidence_json">{evidence_json}</textarea></details>
        <div class="field"><label>承認者</label><input name="approved_by" value="admin"></div>
        <div class="field"><label>管理メモ</label><textarea name="memo">ログ #{esc(log_id)} から承認QA登録</textarea></div>
        <button class="btn" type="submit">このログから承認QAとして登録</button>
      </form>
      <details><summary>debug情報</summary><div class="pre">{esc(json.dumps(row.get("debug") or {}, ensure_ascii=False, indent=2))}</div></details>
    </section>
    """


def corpus_toggle_form(corpus: dict, base_url: str) -> str:
    corpus_id = str(corpus.get("corpus_id") or "").strip()
    if not corpus_id:
        return "-"
    enabled = bool(corpus.get("enabled"))
    next_enabled = "false" if enabled else "true"
    label = "無効化" if enabled else "有効化"
    tone = "secondary" if enabled else ""
    base_input = f'<input type="hidden" name="base_url" value="{esc(base_url)}">' if allow_runtime_api_url_override() else ""
    return f"""
      <form method="post" action="/index-corpus-toggle" data-loading-message="文書セットの状態を更新しています..." style="display:inline">
        {base_input}
        <input type="hidden" name="corpus_id" value="{esc(corpus_id)}">
        <input type="hidden" name="enabled" value="{next_enabled}">
        <button class="btn {tone} small" type="submit">{label}</button>
      </form>
    """


def document_register_panel(corpora: list[dict], base_url: str) -> str:
    options = "\n".join(
        f'<option value="{esc(corpus.get("corpus_id"))}">{esc(corpus.get("display_name") or corpus.get("corpus_id"))}</option>'
        for corpus in corpora
        if corpus.get("corpus_id")
    )
    base_input = f'<input type="hidden" name="base_url" value="{esc(base_url)}">' if allow_runtime_api_url_override() else ""
    return f"""
    <section class="panel">
      <h2>Markdown文書登録</h2>
      <p class="lead">Markdown化済みの文書を保存し、必要に応じてチャンキングとベクトルIndex再構築まで実行します。</p>
      <form method="post" action="/documents-register" data-loading-message="Markdown文書を登録し、チャンキング/ベクトル化を実行しています...">
        {base_input}
        <div class="grid-2">
          <div class="field"><label>Corpus ID</label><input name="corpus_id" list="corpus-id-options" placeholder="例: internal_manual" required><datalist id="corpus-id-options">{options}</datalist></div>
          <div class="field"><label>表示名</label><input name="display_name" placeholder="例: 社内マニュアル"></div>
          <div class="field"><label>Markdown保存先</label><input name="markdown_dir" placeholder="既存corpusは空欄で既存設定を使用"></div>
          <div class="field"><label>優先度</label><input name="priority" value="100"></div>
          <div class="field"><label>文書種別</label><input name="document_type" value="manual"></div>
          <div class="field"><label>文書ID</label><input name="document_id" placeholder="例: manual_001"></div>
          <div class="field"><label>ファイル名</label><input name="filename" placeholder="空欄なら文書IDまたはタイトルから自動生成"></div>
          <div class="field"><label>出典URL/メモ</label><input name="source_url" placeholder="任意"></div>
          <div class="field"><label>有効開始日</label><input type="date" name="valid_from"></div>
          <div class="field"><label>有効終了日</label><input type="date" name="valid_until"></div>
        </div>
        <div class="field"><label>タイトル</label><input name="title" required></div>
        <div class="field"><label>Markdown本文</label><textarea name="markdown_text" rows="16" placeholder="# 見出し&#10;&#10;本文..." required></textarea></div>
        <div class="qa-register-actions">
          <label><input type="checkbox" name="enabled" checked> このcorpusを検索対象にする</label>
          <label><input type="checkbox" name="run_chunking" checked> チャンキングを実行</label>
          <label><input type="checkbox" name="run_indexing" checked> ベクトル化を実行</label>
          <button class="btn" type="submit">登録してIndex更新</button>
        </div>
      </form>
    </section>
    """


async def index_eval_content(base_url: str) -> str:
    try:
        index = await api_get(base_url, "/admin/index/status")
        eval_status = await api_get(base_url, "/admin/eval/status")
    except Exception as exc:
        return error_panel("文書/評価を表示できません", exc)
    chunks = index.get("chunks", {})
    vector = index.get("vector_collection", {})
    forms = index.get("forms", {})
    child_count = int_value(chunks.get("child_chunks", {}).get("row_count"))
    vector_count = vector.get("count")
    mismatch = vector_count is not None and int_value(vector_count) != child_count
    corpus_rows = [
        [
            esc(corpus.get("corpus_id")),
            esc(corpus.get("display_name")),
            esc(corpus.get("markdown_files")),
            validity_summary(corpus),
            esc(corpus.get("priority")),
            badge("有効", "ok") if corpus.get("enabled") else badge("無効", "muted"),
            corpus_toggle_form(corpus, base_url),
        ]
        for corpus in index.get("corpora", [])
    ]
    file_rows = [
        ["parent_chunks.jsonl", esc(chunks.get("parent_chunks", {}).get("row_count")), "-", esc(chunks.get("parent_chunks", {}).get("modified_at"))],
        ["child_chunks.jsonl", esc(chunks.get("child_chunks", {}).get("row_count")), "-", esc(chunks.get("child_chunks", {}).get("modified_at"))],
        ["child_chunks_with_tags.jsonl", esc(chunks.get("child_chunks_with_tags", {}).get("row_count")), esc(chunks.get("child_chunks_with_tags", {}).get("tagged_count")), esc(chunks.get("child_chunks_with_tags", {}).get("modified_at"))],
        ["chunk_report.csv", esc(chunks.get("chunk_report", {}).get("row_count")), esc(chunks.get("chunk_report", {}).get("warning_count")), esc(chunks.get("chunk_report", {}).get("modified_at"))],
    ]
    warnings = chunks.get("chunk_report", {}).get("warnings", [])
    warning_rows = [[esc(w.get("corpus_id")), esc(short_text(w.get("source_file"), 80)), esc(w.get("warnings"))] for w in warnings[:30]]
    latest = eval_status.get("latest", {})
    failed_rows = [
        [esc(row.get("question_id")), esc(short_text(row.get("question"), 110)), esc(row.get("status") or "source_count=0"), esc(short_text(row.get("error"), 80))]
        for row in latest.get("failed_items", [])[:30]
    ]
    form_html = (
        f'{badge("設定済み", "ok")} 関連様式 {esc(forms.get("row_count"))}件'
        if forms.get("configured")
        else f'{badge("未設定", "muted")} data/forms/form_catalog.csv はヘッダーのみです。通常UIでは関連様式導線を表示しません。'
    )
    return f"""
    <section class="grid-4">
      {kpi_card("親チャンク", chunks.get("parent_chunks", {}).get("row_count", 0))}
      {kpi_card("子チャンク", child_count)}
      {kpi_card("ベクトル数", vector_count if vector_count is not None else "取得不能", "Chroma collection")}
      {kpi_card("評価失敗", latest.get("failed_count", 0), latest.get("path") or "結果未生成")}
    </section>
    <section class="panel">
      <h2>Index状態</h2>
      <div class="notice" style="display:{'block' if mismatch else 'none'}">子チャンク数とベクトル数が一致していません。Index再構築を検討してください。</div>
      <p class="lead">{form_html}</p>
      {data_table(["Corpus ID","表示名","Markdown件数","有効期間","優先度","状態","操作"], corpus_rows, "corpus_settings.json が未設定です。")}
    </section>
    {document_register_panel(index.get("corpora", []), base_url)}
    <section class="grid-2">
      <div class="panel"><h2>チャンク/タグ</h2>{data_table(["ファイル","行数","タグ/警告","更新日時"], file_rows, "チャンクファイルがありません。scripts/01_make_chunks.py から実行してください。")}</div>
      <div class="panel"><h2>chunk_report警告</h2>{data_table(["Corpus","ファイル","警告"], warning_rows, "chunk_reportに警告はありません。")}</div>
    </section>
    <section id="eval" class="panel">
      <h2>評価</h2>
      <p class="lead">qa_100_questions.csv: {esc("あり" if eval_status.get("qa_100_questions", {}).get("exists") else "なし")} / 最新結果: {esc(latest.get("path") or "未生成")}</p>
      <div class="grid-3">
        {kpi_card("評価行数", latest.get("row_count", 0))}
        {kpi_card("OK", latest.get("ok_count", 0))}
        {kpi_card("失敗/要確認", latest.get("failed_count", 0))}
      </div>
      <h3>評価コマンド</h3>
      <div class="pre">{text_block(chr(10).join(eval_status.get("commands", [])))}</div>
      <h3>失敗質問</h3>
      {data_table(["ID","質問","状態","エラー"], failed_rows, "評価失敗は記録されていません。評価未実行の場合は上のコマンドで実行してください。")}
    </section>
    """


async def tags_content(base_url: str, request: Request) -> str:
    q = request.query_params
    query = q.get("query") or ""
    corpus_id = q.get("corpus_id") or ""
    child_id = q.get("child_id") or ""
    try:
        data = await api_get(base_url, "/admin/search-tags", {"query": query, "corpus_id": corpus_id, "limit": 100})
        items = data.get("items", [])
    except Exception as exc:
        return error_panel("SearchTag編集を表示できません", exc)
    selected = next((item for item in items if str(item.get("child_id")) == child_id), None) if child_id else None
    selected = selected or (items[0] if items else None)
    rows = [
        [
            esc(short_text(item.get("child_id"), 30)),
            esc(item.get("corpus_id")),
            esc(short_text(item.get("heading_path"), 110)),
            esc(item.get("tag_count")),
            esc(item.get("candidate_reason") or ""),
            f'<a class="btn small" href="{with_base("/search-tags", base_url, query=query, corpus_id=corpus_id, child_id=item.get("child_id"))}">編集</a>',
        ]
        for item in items
    ]
    return f"""
    <section class="panel">
      <h2>SearchTag編集</h2>
      <p class="lead">全件棚卸しではなく、改善キュー・ログ詳細・報告詳細から必要なチャンクだけ補強する補助画面です。</p>
      <form class="filters" method="get" action="/search-tags">
        <div class="field"><label>キーワード</label><input name="query" value="{esc(query)}"></div>
        <div class="field"><label>Corpus</label><input name="corpus_id" value="{esc(corpus_id)}"></div>
        <button class="btn secondary" type="submit">検索</button>
      </form>
    </section>
    <section class="split">
      <div class="panel"><h2>候補一覧</h2>{data_table(["子ID","Corpus","見出し","タグ数","理由","操作"], rows, "候補がありません。改善キューや低信頼ログから遷移してください。")}</div>
      {await tag_detail_panel(base_url, selected) if selected else '<div class="panel"><h2>編集</h2><div class="empty">編集対象を選択してください。</div></div>'}
    </section>
    """


async def tag_detail_panel(base_url: str, summary: dict) -> str:
    child_id = str(summary.get("child_id") or "")
    try:
        data = await api_get(base_url, f"/admin/search-tags/{quote(child_id, safe='')}")
        item = data.get("item", {})
    except Exception:
        item = summary
    tags = "\n".join(str(tag) for tag in item.get("search_tags") or [])
    text = item.get("text") or item.get("child_text") or summary.get("text_preview") or ""
    return f"""
    <div class="panel">
      <h2>SearchTag編集</h2>
      <p><strong>{esc(item.get("heading_path"))}</strong></p>
      <div class="source-meta"><span>子ID: {esc(child_id)}</span><span>Corpus: {esc(item.get("corpus_id"))}</span></div>
      <div class="pre markdown-body">{markdown_block(text)}</div>
      <form method="post" action="/search-tags-save">
        <input type="hidden" name="child_id" value="{esc(child_id)}">
        <div class="field"><label>SearchTag 1行1件</label><textarea name="search_tags">{esc(tags)}</textarea></div>
        <label><input type="checkbox" name="reload_retriever" value="true" checked> 保存後に検索器を再読み込み</label>
        <br><br><button class="btn" type="submit">保存</button>
      </form>
    </div>
    """


async def trend_report_content(base_url: str, request: Request) -> str:
    q = request.query_params
    payload = {
        "days": int_value(q.get("days"), 7),
        "sample_limit": int_value(q.get("sample_limit"), 1000),
        "top_n": int_value(q.get("top_n"), 20),
        "low_score_threshold": float_value(q.get("low_score_threshold"), 0.35),
    }
    try:
        data = await api_post(base_url, "/admin/logs/report", payload, timeout=240)
    except Exception as exc:
        return error_panel("傾向レポートを生成できません", exc)
    return f'<section class="panel"><h2>LLM傾向レポート</h2><p class="lead">日常の主画面ではなく、必要な時だけ生成する分析補助です。</p><div class="pre markdown-body">{markdown_block(data.get("report"))}</div></section>'


def parse_evidence_json(raw: object) -> list[dict]:
    try:
        value = json.loads(str(raw or "[]"))
    except json.JSONDecodeError as exc:
        raise ValueError("根拠JSONの形式が正しくありません") from exc
    if not isinstance(value, list):
        raise ValueError("根拠JSONは配列で指定してください")
    return value


async def handle_report_analysis(request: Request) -> RedirectResponse:
    form = await request.form()
    base_url = normalize_base_url(form.get("base_url"))
    report_id = int_value(form.get("report_id"))
    payload = {
        "status": str(form.get("status") or "open"),
        "issue_type": str(form.get("issue_type") or "") or None,
        "resolution_type": str(form.get("resolution_type") or "") or None,
        "admin_memo": str(form.get("admin_memo") or ""),
        "linked_child_id": str(form.get("linked_child_id") or "") or None,
    }
    await api_post(base_url, f"/admin/reports/{report_id}/analysis", payload)
    return redirect("/reports", base_url, status="all", report_id=report_id)


async def handle_report_status(request: Request) -> RedirectResponse:
    form = await request.form()
    base_url = normalize_base_url(form.get("base_url"))
    report_id = int_value(form.get("report_id"))
    status = str(form.get("status") or "open")
    await api_patch(base_url, f"/admin/reports/{report_id}/status", {"status": status})
    return redirect("/reports", base_url, status="all", report_id=report_id)


async def handle_qa_register(request: Request) -> HTMLResponse | RedirectResponse:
    form = await request.form()
    base_url = normalize_base_url(form.get("base_url"))
    variant = getattr(request.app.state, "variant", next(iter(VARIANTS.values())))
    return_to = str(form.get("return_to") or request.headers.get("referer") or "").strip()
    try:
        question = str(form.get("question") or "").strip()
        answer = str(form.get("answer") or "").strip()
        if not question:
            raise ValueError("質問が空のため、承認QAとして登録できません。")
        if not answer:
            raise ValueError("回答が空のため、承認QAとして登録できません。")
        evidence = parse_evidence_json(form.get("evidence_json"))
        payload = {
            "question": question,
            "answer": answer,
            "evidence": evidence,
            "approved_by": str(form.get("approved_by") or "admin"),
            "memo": str(form.get("memo") or ""),
            "source_report_id": int_value(form.get("source_report_id")) or None,
            "corpus_version": str(form.get("corpus_version") or "") or None,
            "index_version": str(form.get("index_version") or "") or None,
        }
        result = await api_post(base_url, "/admin/qa-cache", payload, timeout=240)
        return redirect("/qa-cache", base_url, qa_id=result.get("qa_id"))
    except Exception as exc:
        active = "logs" if "/logs" in return_to else "reports" if "/reports" in return_to else "qa"
        back_href = return_to or with_base("/qa-cache", base_url)
        content = (
            error_panel("承認QA登録に失敗しました", api_error_message(exc))
            + f"""
            <section class="panel">
              <h2>次の操作</h2>
              <p class="lead">質問・回答・根拠JSON、Embedding API の起動状態を確認してから再実行してください。</p>
              <a class="btn secondary" href="{esc(back_href)}">元の画面に戻る</a>
            </section>
            """
        )
        return HTMLResponse(page_shell(variant, active, base_url, content))


async def handle_qa_generate_answer(request: Request) -> HTMLResponse:
    form = await request.form()
    base_url = normalize_base_url(form.get("base_url"))
    question = str(form.get("question") or "").strip()
    top_k = int_value(form.get("top_k"), 8)
    variant = getattr(request.app.state, "variant", next(iter(VARIANTS.values())))
    if not question:
        content = await qa_content(base_url, request)
        return HTMLResponse(page_shell(variant, "qa", base_url, content))
    try:
        result = await api_post(
            base_url,
            "/ask",
            {
                "question": question,
                "top_k": top_k,
                "show_debug": True,
            },
            timeout=360,
        )
    except Exception as exc:
        content = error_panel("質問から回答を生成できませんでした", exc)
        return HTMLResponse(page_shell(variant, "qa", base_url, content))
    generated = {
        "question": question,
        "answer": result.get("answer") or "",
        "sources": result.get("sources") or [],
    }
    content = await qa_content(base_url, request, generated=generated)
    return HTMLResponse(page_shell(variant, "qa", base_url, content))


async def handle_qa_update(request: Request) -> RedirectResponse:
    form = await request.form()
    base_url = normalize_base_url(form.get("base_url"))
    qa_id = int_value(form.get("qa_id"))
    evidence = parse_evidence_json(form.get("evidence_json"))
    payload = {
        "question": str(form.get("question") or ""),
        "answer": str(form.get("answer") or ""),
        "evidence": evidence,
        "status": str(form.get("status") or "approved"),
        "approved_by": str(form.get("approved_by") or "admin"),
        "memo": str(form.get("memo") or ""),
        "corpus_version": str(form.get("corpus_version") or "") or None,
        "index_version": str(form.get("index_version") or "") or None,
    }
    await api_patch(base_url, f"/admin/qa-cache/{qa_id}", payload)
    return redirect("/qa-cache", base_url, qa_id=qa_id)


async def handle_qa_disable(request: Request) -> RedirectResponse:
    form = await request.form()
    base_url = normalize_base_url(form.get("base_url"))
    qa_id = int_value(form.get("qa_id"))
    await api_post(base_url, f"/admin/qa-cache/{qa_id}/disable", {})
    return redirect("/qa-cache", base_url, status="disabled", qa_id=qa_id)


async def handle_qa_alias_update(request: Request) -> RedirectResponse:
    form = await request.form()
    base_url = normalize_base_url(form.get("base_url"))
    qa_id = int_value(form.get("qa_id"))
    alias_id = int_value(form.get("alias_id"))
    payload = {
        "status": str(form.get("status") or "") or None,
        "memo": "管理画面から状態変更",
    }
    await api_patch(base_url, f"/admin/qa-cache/aliases/{alias_id}", payload)
    return redirect("/qa-cache", base_url, qa_id=qa_id)


async def handle_qa_alias_backfill(request: Request) -> HTMLResponse | RedirectResponse:
    form = await request.form()
    base_url = normalize_base_url(form.get("base_url"))
    payload = {
        "ensure_original": bool(form.get("ensure_original")),
        "generate_llm_aliases": bool(form.get("generate_llm_aliases")),
        "only_without_llm_aliases": True,
        "limit": int_value(form.get("limit"), 50),
        "dry_run": bool(form.get("dry_run")),
    }
    result = await api_post(base_url, "/admin/qa-cache/backfill-aliases", payload, timeout=600)
    if payload["dry_run"]:
        items = result.get("items") or []
        rows = [
            [esc(item.get("qa_id")), esc(short_text(item.get("question"), 120)), esc(item.get("corpus_version")), esc(item.get("index_version"))]
            for item in items[:100]
        ]
        content = f"""
        <section class="panel">
          <h2>original alias補完 dry-run</h2>
          <p class="lead">追加予定: {esc(result.get("would_insert"))} / 対象: {esc(result.get("target_count"))}</p>
          <div class="notice">運用DBに書き込む前に、必ずSQLiteのバックアップを取得してください。CLI推奨: python scripts/09_backfill_approved_qa_aliases.py --apply --backup --ensure-original</div>
          {data_table(["QA ID", "質問", "Corpus", "Index"], rows, "対象QAはありません。")}
          <details open><summary>raw result</summary><div class="pre">{esc(json.dumps(result, ensure_ascii=False, indent=2))}</div></details>
          <p style="margin-top:14px"><a class="btn ghost" href="{with_base('/qa-cache', base_url)}">承認QAへ戻る</a></p>
        </section>
        """
        variant = getattr(request.app.state, "variant", next(iter(VARIANTS.values())))
        return HTMLResponse(page_shell(variant, "qa", base_url, content))
    return redirect("/qa-cache", base_url)


async def handle_qa_cache_reload_index(request: Request) -> RedirectResponse:
    form = await request.form()
    base_url = normalize_base_url(form.get("base_url"))
    await api_post(base_url, "/admin/qa-cache/reload-index", {}, timeout=120)
    return redirect("/qa-cache", base_url)


async def handle_index_corpus_toggle(request: Request) -> RedirectResponse:
    form = await request.form()
    base_url = normalize_base_url(form.get("base_url"))
    corpus_id = str(form.get("corpus_id") or "").strip()
    enabled = str(form.get("enabled") or "").strip().casefold() in {"1", "true", "yes", "on", "enabled"}
    await api_post(base_url, f"/admin/index/corpora/{quote(corpus_id, safe='')}/enabled", {"enabled": enabled}, timeout=90)
    return redirect("/index-eval", base_url)


async def handle_document_register(request: Request) -> HTMLResponse:
    form = await request.form()
    base_url = normalize_base_url(form.get("base_url"))
    payload = {
        "corpus_id": str(form.get("corpus_id") or "").strip(),
        "display_name": str(form.get("display_name") or "").strip(),
        "markdown_dir": str(form.get("markdown_dir") or "").strip(),
        "priority": str(form.get("priority") or "").strip(),
        "enabled": bool(form.get("enabled")),
        "document_type": str(form.get("document_type") or "").strip(),
        "document_id": str(form.get("document_id") or "").strip(),
        "filename": str(form.get("filename") or "").strip(),
        "source_url": str(form.get("source_url") or "").strip(),
        "valid_from": str(form.get("valid_from") or "").strip(),
        "valid_until": str(form.get("valid_until") or "").strip(),
        "title": str(form.get("title") or "").strip(),
        "markdown_text": str(form.get("markdown_text") or ""),
        "run_chunking": bool(form.get("run_chunking")),
        "run_indexing": bool(form.get("run_indexing")),
    }
    variant = getattr(request.app.state, "variant", next(iter(VARIANTS.values())))
    try:
        result = await api_post(base_url, "/admin/documents/register-markdown", payload, timeout=7200)
    except Exception as exc:
        return HTMLResponse(page_shell(variant, "index_eval", base_url, error_panel("Markdown文書を登録できません", api_error_message(exc))))

    step_rows = [
        [esc(step.get("script")), esc(step.get("returncode")), esc(short_text(step.get("stdout"), 120)), esc(short_text(step.get("stderr"), 120))]
        for step in result.get("steps", [])
    ]
    content = f"""
    <section class="panel">
      <h2>Markdown文書を登録しました</h2>
      <p class="lead">保存先: {esc((result.get("document") or {}).get("path"))}</p>
      <div class="grid-3">
        {kpi_card("Corpus", (result.get("corpus") or {}).get("corpus_id") or "-")}
        {kpi_card("新規Corpus", "はい" if result.get("corpus_created") else "いいえ")}
        {kpi_card("実行Step", len(result.get("steps") or []))}
      </div>
      {data_table(["Script","終了コード","stdout","stderr"], step_rows, "スクリプトは実行していません。")}
      <details><summary>実行結果JSON</summary><div class="pre">{esc(json.dumps(result, ensure_ascii=False, indent=2))}</div></details>
      <p style="margin-top:14px"><a class="btn" href="{with_base('/index-eval', base_url)}">文書/評価へ戻る</a></p>
    </section>
    """
    return HTMLResponse(page_shell(variant, "index_eval", base_url, content))


async def handle_qa_alias_add(request: Request) -> RedirectResponse:
    form = await request.form()
    base_url = normalize_base_url(form.get("base_url"))
    qa_id = int_value(form.get("qa_id"))
    aliases = [line.strip() for line in str(form.get("aliases") or "").splitlines() if line.strip()]
    payload = {
        "aliases": aliases,
        "alias_type": str(form.get("alias_type") or "admin_alias"),
        "status": str(form.get("status") or "active"),
        "memo": "admin UI",
    }
    await api_post(base_url, f"/admin/qa-cache/{qa_id}/aliases", payload, timeout=240)
    return redirect("/qa-cache", base_url, qa_id=qa_id)


async def handle_qa_alias_generate(request: Request) -> HTMLResponse | RedirectResponse:
    form = await request.form()
    base_url = normalize_base_url(form.get("base_url"))
    qa_id = int_value(form.get("qa_id"))
    payload = {
        "max_aliases": int_value(form.get("max_aliases"), 8),
        "replace_existing_generated": bool(form.get("replace_existing_generated")),
        "dry_run": bool(form.get("dry_run")),
        "status": str(form.get("status") or "disabled"),
    }
    result = await api_post(base_url, f"/admin/qa-cache/{qa_id}/aliases/generate", payload, timeout=300)
    if not payload["dry_run"]:
        return redirect("/qa-cache", base_url, qa_id=qa_id)

    candidates = result.get("candidates") or []
    alias_text = "\n".join(str(item.get("alias_text") or "") for item in candidates if str(item.get("alias_text") or "").strip())
    rows = [
        [
            esc(item.get("alias_text")),
            esc(item.get("normalized_text")),
            esc(", ".join(item.get("risk_flags") or [])),
            "yes" if item.get("duplicate") else "-",
        ]
        for item in candidates
    ]
    content = f"""
    <section class="panel">
      <h2>LLM alias dry-run result</h2>
      <p class="lead">QA ID: {qa_id} / {esc(result.get("question"))}</p>
      {data_table(["alias候補", "normalized_text", "risk_flags", "duplicate"], rows, "alias候補は生成されませんでした。")}
      <details><summary>raw/debug</summary><div class="pre">{esc(json.dumps(result, ensure_ascii=False, indent=2))}</div></details>
      <div class="grid-2" style="margin-top:14px">
        <form method="post" action="/qa-alias-add" data-loading-message="Alias候補をdisabledで保存しています...">
          <input type="hidden" name="qa_id" value="{qa_id}">
          <input type="hidden" name="alias_type" value="llm_paraphrase">
          <input type="hidden" name="status" value="disabled">
          <div class="field"><label>disabled保存する候補</label><textarea name="aliases">{esc(alias_text)}</textarea></div>
          <button class="btn secondary" type="submit">disabledで保存</button>
        </form>
        <form method="post" action="/qa-alias-add" data-loading-message="Alias候補をactiveで保存しています...">
          <input type="hidden" name="qa_id" value="{qa_id}">
          <input type="hidden" name="alias_type" value="llm_paraphrase">
          <input type="hidden" name="status" value="active">
          <div class="field"><label>active保存する候補</label><textarea name="aliases">{esc(alias_text)}</textarea></div>
          <button class="btn" type="submit">activeで保存</button>
        </form>
      </div>
      <p style="margin-top:14px"><a class="btn ghost" href="{with_base("/qa-cache", base_url, qa_id=qa_id)}">破棄してQA詳細へ戻る</a></p>
    </section>
    """
    variant = getattr(request.app.state, "variant", next(iter(VARIANTS.values())))
    return HTMLResponse(page_shell(variant, "qa", base_url, content))


async def handle_search_tags_save(request: Request) -> RedirectResponse:
    form = await request.form()
    base_url = normalize_base_url(form.get("base_url"))
    child_id = str(form.get("child_id") or "")
    tags = [
        line.strip()
        for line in str(form.get("search_tags") or "").replace(",", "\n").replace("、", "\n").splitlines()
        if line.strip()
    ]
    await api_patch(
        base_url,
        f"/admin/search-tags/{quote(child_id, safe='')}",
        {"search_tags": tags, "reload_retriever": bool(form.get("reload_retriever"))},
    )
    return redirect("/search-tags", base_url, child_id=child_id)


def create_admin_app(variant_key: str):
    variant = VARIANTS[variant_key]
    app, rt = fast_app()
    app.state.variant = variant
    app.routes.insert(0, Mount("/static", app=StaticFiles(directory=BASE_DIR / "static"), name="static"))

    @rt("/", methods=["GET"])
    async def dashboard(request: Request):
        base_url = selected_base(request)
        return HTMLResponse(page_shell(variant, "dashboard", base_url, await dashboard_content(base_url)))

    @rt("/queue", methods=["GET"])
    async def queue(request: Request):
        base_url = selected_base(request)
        return HTMLResponse(page_shell(variant, "queue", base_url, await queue_content(base_url, request)))

    @rt("/actions", methods=["GET"])
    async def actions(request: Request):
        return RedirectResponse(with_base("/queue", selected_base(request)), status_code=303)

    @rt("/reports", methods=["GET"])
    async def reports(request: Request):
        base_url = selected_base(request)
        return HTMLResponse(page_shell(variant, "reports", base_url, await reports_content(base_url, request)))

    @rt("/qa-cache", methods=["GET"])
    async def qa_cache(request: Request):
        base_url = selected_base(request)
        return HTMLResponse(page_shell(variant, "qa", base_url, await qa_content(base_url, request)))

    @rt("/logs", methods=["GET"])
    async def logs(request: Request):
        base_url = selected_base(request)
        return HTMLResponse(page_shell(variant, "logs", base_url, await logs_content(base_url, request)))

    @rt("/index-eval", methods=["GET"])
    async def index_eval(request: Request):
        base_url = selected_base(request)
        return HTMLResponse(page_shell(variant, "index_eval", base_url, await index_eval_content(base_url)))

    @rt("/index", methods=["GET"])
    async def index_alias(request: Request):
        return RedirectResponse(with_base("/index-eval", selected_base(request)), status_code=303)

    @rt("/index-corpus-toggle", methods=["POST"])
    async def index_corpus_toggle(request: Request):
        return await handle_index_corpus_toggle(request)

    @rt("/documents-register", methods=["POST"])
    async def documents_register(request: Request):
        return await handle_document_register(request)

    @rt("/search-tags", methods=["GET"])
    async def tags(request: Request):
        base_url = selected_base(request)
        return HTMLResponse(page_shell(variant, "queue", base_url, await tags_content(base_url, request)))

    @rt("/trend-report", methods=["GET"])
    async def trend_report(request: Request):
        base_url = selected_base(request)
        return HTMLResponse(page_shell(variant, "dashboard", base_url, await trend_report_content(base_url, request)))

    @rt("/report-analysis", methods=["POST"])
    async def report_analysis(request: Request):
        return await handle_report_analysis(request)

    @rt("/report-status", methods=["POST"])
    async def report_status(request: Request):
        return await handle_report_status(request)

    @rt("/qa-register", methods=["POST"])
    async def qa_register(request: Request):
        return await handle_qa_register(request)

    @rt("/qa-generate-answer", methods=["POST"])
    async def qa_generate_answer(request: Request):
        return await handle_qa_generate_answer(request)

    @rt("/qa-update", methods=["POST"])
    async def qa_update(request: Request):
        return await handle_qa_update(request)

    @rt("/qa-disable", methods=["POST"])
    async def qa_disable(request: Request):
        return await handle_qa_disable(request)

    @rt("/qa-alias-add", methods=["POST"])
    async def qa_alias_add(request: Request):
        return await handle_qa_alias_add(request)

    @rt("/qa-alias-update", methods=["POST"])
    async def qa_alias_update(request: Request):
        return await handle_qa_alias_update(request)

    @rt("/qa-alias-generate", methods=["POST"])
    async def qa_alias_generate(request: Request):
        return await handle_qa_alias_generate(request)

    @rt("/qa-alias-backfill", methods=["POST"])
    async def qa_alias_backfill(request: Request):
        return await handle_qa_alias_backfill(request)

    @rt("/qa-cache-reload-index", methods=["POST"])
    async def qa_cache_reload_index(request: Request):
        return await handle_qa_cache_reload_index(request)

    @rt("/search-tags-save", methods=["POST"])
    async def search_tags_save(request: Request):
        return await handle_search_tags_save(request)

    @rt("/health", methods=["GET"])
    async def health(_: Request):
        return HTMLResponse("ok")

    return app


def run() -> None:
    serve()
