from __future__ import annotations

import csv
import html
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fasthtml.common import fast_app
from markdown_it import MarkdownIt
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from api.config import get_required_url, get_required_url_value, get_url_number, load_settings, rag_ask_url
from eval_common import DEFAULT_EVAL_JSONL, DEFAULT_RUNS_DIR, read_jsonl, write_csv, write_jsonl

app, rt = fast_app()
MARKDOWN = MarkdownIt("commonmark", {"html": False, "breaks": True}).enable(["table", "strikethrough"])

SETTINGS = load_settings()
EVAL_LLM_SETTINGS = SETTINGS.get("eval_llm") or SETTINGS.get("alias_llm") or {}
EMBEDDING_SETTINGS = SETTINGS.get("embedding") or {}
DEFAULT_EVAL_BASE_URL = get_required_url("eval_llm_base_url")
DEFAULT_EMBEDDING_BASE_URL = get_required_url("embedding_base_url")

MANUAL_CRITERIA: list[tuple[str, str, str]] = [
    ("accuracy", "正確性", "回答の結論・条件・数値が根拠文書と合っているか"),
    ("evidence", "根拠整合性", "提示根拠と回答内容が対応しているか"),
    ("coverage", "網羅性", "質問に必要な論点を過不足なく扱っているか"),
    ("appropriateness", "回答妥当性", "実務利用しやすい表現・抑制・注意喚起になっているか"),
]
SCORE_OPTIONS: list[tuple[str, str]] = [
    ("1.0", "良い"),
    ("0.5", "一部不足"),
    ("0.0", "要修正"),
]
TAB_LABELS = {
    "chunks": "チャンキング",
    "batch": "バッチ推論",
    "ragas": "RAGAS",
    "deepeval": "DeepEval",
    "rules": "ルール採点",
    "summary": "結果サマリ",
    "detail": "詳細・手動採点",
}


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def markdown_block(value: Any) -> str:
    return MARKDOWN.render(str(value if value is not None else ""))


def rel_path(path: Path) -> str:
    try:
        return path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_repo_path(value: str | None, default: Path | None = None) -> Path:
    if not value:
        if default is None:
            return BASE_DIR
        return default
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


def default_eval_input() -> str:
    return rel_path(DEFAULT_EVAL_JSONL) if DEFAULT_EVAL_JSONL.exists() else "eval/qa_100_questions.csv"


def list_prediction_files() -> list[Path]:
    if not DEFAULT_RUNS_DIR.exists():
        return []
    return sorted(DEFAULT_RUNS_DIR.glob("*/predictions.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def list_run_dirs() -> list[Path]:
    runs = {path.parent for path in list_prediction_files()}
    return sorted(runs, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)


def load_csv_rows(path: Path, limit: int = 80) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit]


def parse_json_field(value: Any, fallback: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    if not text:
        return fallback
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return fallback


def is_checked(form: Any, name: str) -> bool:
    return str(form.get(name) or "").lower() in {"1", "true", "yes", "on"}


def int_form(form: Any, name: str, default: int) -> int:
    try:
        return int(str(form.get(name) or "").strip() or default)
    except ValueError:
        return default


def text_form(form: Any, name: str, default: str = "") -> str:
    value = form.get(name)
    if value is None:
        return default
    return str(value).strip()


def run_command(args: list[str], timeout_sec: int) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(args, cwd=BASE_DIR, text=True, capture_output=True, timeout=timeout_sec)
        return {
            "args": args,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timeout": False,
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "args": args,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timeout": True,
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "timeout_sec": timeout_sec,
        }


def command_result_html(result: dict[str, Any] | None) -> str:
    if not result:
        return ""
    command = subprocess.list2cmdline([str(part) for part in result.get("args", [])])
    timed_out = bool(result.get("timeout"))
    returncode = result.get("returncode")
    if timed_out:
        status_class = "danger"
        status_text = f"timeout ({esc(result.get('timeout_sec'))} sec)"
    elif returncode == 0:
        status_class = "ok"
        status_text = "completed"
    else:
        status_class = "danger"
        status_text = f"exit code: {esc(returncode)}"
    return f"""
    <section class="panel command-result">
      <div class="section-head">
        <h2>実行結果</h2>
        <span class="status {status_class}">{status_text}</span>
      </div>
      <label>Command</label>
      <pre class="cmd">{esc(command)}</pre>
      <div class="grid two">
        <div>
          <label>stdout</label>
          <pre>{esc(result.get("stdout") or "")}</pre>
        </div>
        <div>
          <label>stderr</label>
          <pre>{esc(result.get("stderr") or "")}</pre>
        </div>
      </div>
    </section>
    """


def nav_html(active: str) -> str:
    items = []
    for key, label in TAB_LABELS.items():
        cls = "active" if key == active else ""
        items.append(f'<a class="{cls}" href="/?tab={esc(key)}">{esc(label)}</a>')
    return '<nav class="tabs">' + "".join(items) + "</nav>"


def layout(active: str, body: str) -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Dev Eval</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --surface: rgba(255, 255, 255, .88);
      --surface-strong: #fff;
      --text: #172033;
      --muted: #667085;
      --line: rgba(23, 32, 51, .10);
      --primary: #305cff;
      --primary-soft: rgba(48, 92, 255, .10);
      --accent: #00b8a9;
      --danger: #ef4444;
      --shadow: 0 18px 60px rgba(15, 23, 42, .10);
      --radius: 16px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(48, 92, 255, .14), transparent 32%),
        radial-gradient(circle at bottom right, rgba(0, 184, 169, .10), transparent 28%),
        var(--bg);
      color: var(--text);
      font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    header {{ max-width: 1760px; margin: 0 auto; padding: 32px 28px 18px; }}
    header h1 {{ margin: 0; font-size: clamp(30px, 4vw, 48px); letter-spacing: 0; }}
    header p {{ margin: 8px 0 0; color: var(--muted); line-height: 1.7; }}
    main {{ max-width: 1760px; margin: 0 auto; padding: 0 28px 48px; }}
    .tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0 18px; }}
    .tabs a {{
      display: inline-flex; align-items: center; min-height: 40px; padding: 0 14px;
      border-radius: 999px; text-decoration: none; color: var(--muted);
      background: rgba(255,255,255,.58); border: 1px solid var(--line);
    }}
    .tabs a.active {{ color: var(--primary); background: var(--primary-soft); border-color: rgba(48,92,255,.25); font-weight: 700; }}
    .panel {{
      background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
      box-shadow: var(--shadow); padding: 20px; margin-bottom: 18px; backdrop-filter: blur(12px);
    }}
    .section-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 14px; }}
    h2 {{ margin: 0; font-size: 20px; }}
    h3 {{ margin: 18px 0 10px; font-size: 16px; }}
    label {{ display: block; color: var(--muted); font-size: 13px; margin: 0 0 6px; }}
    input, select, textarea {{
      width: 100%; border: 1px solid var(--line); background: var(--surface-strong);
      color: var(--text); border-radius: 12px; min-height: 42px; padding: 10px 12px;
      font: inherit;
    }}
    textarea {{ min-height: 96px; resize: vertical; }}
    form .row {{ margin-bottom: 14px; }}
    .grid {{ display: grid; gap: 14px; }}
    .two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .three {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .four {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .actions {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-top: 16px; }}
    button, .button {{
      border: 0; border-radius: 999px; padding: 11px 18px; min-height: 44px;
      background: var(--primary); color: #fff; font-weight: 700; cursor: pointer;
      text-decoration: none; display: inline-flex; align-items: center; justify-content: center;
    }}
    .button.secondary, button.secondary {{ background: #eef2ff; color: var(--primary); }}
    .status {{ padding: 6px 10px; border-radius: 999px; font-size: 13px; font-weight: 700; }}
    .status.ok {{ background: rgba(0,184,169,.12); color: #087f77; }}
    .status.danger {{ background: rgba(239,68,68,.12); color: var(--danger); }}
    .muted {{ color: var(--muted); }}
    .small {{ font-size: 13px; }}
    pre {{
      overflow: auto; max-height: 360px; white-space: pre-wrap; word-break: break-word;
      background: #101828; color: #f8fafc; padding: 14px; border-radius: 12px;
    }}
    pre.cmd {{ max-height: 120px; background: #eef2ff; color: #1d2b53; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 10px; border-bottom: 1px solid var(--line); vertical-align: top; text-align: left; }}
    th {{ color: var(--muted); font-weight: 700; background: rgba(255,255,255,.48); }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 14px; background: var(--surface-strong); }}
    .table-wrap.wide-table {{ max-height: 560px; }}
    .table-wrap.wide-table table {{ width: max-content; min-width: 100%; table-layout: auto; }}
    .table-wrap.wide-table th {{
      position: sticky; top: 0; z-index: 1; background: #f8fbff;
      white-space: nowrap;
    }}
    .table-wrap.wide-table td {{
      min-width: 132px; max-width: 280px; white-space: nowrap;
      overflow: hidden; text-overflow: ellipsis; line-height: 1.45;
    }}
    .detail-cols {{ display: grid; grid-template-columns: minmax(680px, 1.35fr) minmax(520px, .85fr); gap: 20px; align-items: start; }}
    .answer-panel {{ min-width: 0; }}
    .answer-render {{
      border: 1px solid var(--line); border-radius: 14px; background: #fff;
      padding: 16px 18px; line-height: 1.82; overflow-wrap: anywhere;
    }}
    .markdown-body h1,.markdown-body h2,.markdown-body h3 {{ margin: 18px 0 10px; line-height: 1.35; }}
    .markdown-body h1:first-child,.markdown-body h2:first-child,.markdown-body h3:first-child {{ margin-top: 0; }}
    .markdown-body p {{ margin: 0 0 12px; }}
    .markdown-body ul,.markdown-body ol {{ padding-left: 1.55em; margin: 8px 0 14px; }}
    .markdown-body li {{ margin: 4px 0; }}
    .markdown-body table {{ margin: 12px 0; }}
    .markdown-body code {{ background: #eef2ff; color: #1d2b53; border-radius: 6px; padding: 1px 5px; }}
    .markdown-body blockquote {{ margin: 10px 0; padding: 8px 12px; border-left: 4px solid var(--primary); background: var(--primary-soft); border-radius: 8px; }}
    .score-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .score-card {{ border: 1px solid var(--line); border-radius: 14px; padding: 12px; background: rgba(255,255,255,.66); }}
    details {{ border: 1px solid var(--line); border-radius: 12px; padding: 12px; background: rgba(255,255,255,.58); margin-top: 10px; }}
    summary {{ cursor: pointer; font-weight: 700; }}
    @media (max-width: 900px) {{
      .two, .three, .four, .detail-cols, .score-grid {{ grid-template-columns: 1fr; }}
      header, main {{ padding-left: 14px; padding-right: 14px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>RAG Developer Evaluation</h1>
    <p>バッチ推論、RAGAS、DeepEval、ルール採点、手動採点を開発者向けにまとめたFastHTMLアプリです。</p>
  </header>
  <main>
    {nav_html(active)}
    {body}
  </main>
</body>
</html>""")


def checkbox(name: str, label: str, checked: bool = False) -> str:
    value = " checked" if checked else ""
    return f'<label class="small"><input type="checkbox" name="{esc(name)}"{value} style="width:auto; min-height:auto; margin-right:8px;">{esc(label)}</label>'


def input_row(label: str, name: str, value: Any = "", input_type: str = "text", extra: str = "") -> str:
    return f'<div class="row"><label>{esc(label)}</label><input type="{esc(input_type)}" name="{esc(name)}" value="{esc(value)}" {extra}></div>'


def select_row(label: str, name: str, options: list[str], selected: str = "") -> str:
    opts = []
    for option in options:
        sel = " selected" if option == selected else ""
        opts.append(f'<option value="{esc(option)}"{sel}>{esc(option)}</option>')
    return f'<div class="row"><label>{esc(label)}</label><select name="{esc(name)}">{"".join(opts)}</select></div>'


def predictions_options(selected: str = "") -> str:
    files = [rel_path(path) for path in list_prediction_files()]
    return select_row("predictions.jsonl", "prediction_file", files or [""], selected or (files[0] if files else ""))


def render_batch(result: dict[str, Any] | None = None) -> HTMLResponse:
    body = f"""
    <section class="panel">
      <div class="section-head">
        <h2>バッチ推論</h2>
        <span class="muted small">CSV/JSONLをRAG APIへ投入します</span>
      </div>
      <form method="post" action="/run-batch">
        {input_row("RAG API /ask", "api", rag_ask_url())}
        {input_row("input CSV/JSONL", "input_path", default_eval_input())}
        <div class="grid four">
          {input_row("limit", "limit", 1, "number", 'min="0"')}
          {input_row("start", "start", 1, "number", 'min="1"')}
          {input_row("workers", "workers", 1, "number", 'min="1"')}
          {input_row("top_k", "top_k", 0, "number", 'min="0"')}
        </div>
        <div class="grid three">
          {input_row("timeout sec", "timeout_sec", 3600, "number", 'min="60" step="60"')}
          {input_row("retries", "retries", 1, "number", 'min="0"')}
          {input_row("output dir", "output_dir", "")}
        </div>
        <div class="actions">
          {checkbox("show_debug", "show-debugを付ける", True)}
          {checkbox("resume", "既存runを再開")}
          {checkbox("disable_search_tags", "SearchTag無効A/B runとして記録")}
        </div>
        <div class="actions"><button type="submit">バッチ推論を実行</button></div>
      </form>
    </section>
    {command_result_html(result)}
    """
    return layout("batch", body)


def render_ragas(result: dict[str, Any] | None = None) -> HTMLResponse:
    body = f"""
    <section class="panel">
      <div class="section-head">
        <h2>RAGAS</h2>
        <span class="muted small">任意依存が必要です</span>
      </div>
      <form method="post" action="/run-ragas">
        {predictions_options()}
        <div class="grid three">
          {input_row("limit", "limit", 1, "number", 'min="0"')}
          {input_row("評価LLM base URL", "base_url", DEFAULT_EVAL_BASE_URL)}
          {input_row("評価LLM model", "model", EVAL_LLM_SETTINGS.get("model") or "")}
        </div>
        <div class="grid three">
          {input_row("評価LLM API key", "api_key", EVAL_LLM_SETTINGS.get("api_key") or "dummy", "password")}
          {input_row("Embedding base URL", "embedding_base_url", DEFAULT_EMBEDDING_BASE_URL)}
          {input_row("Embedding model", "embedding_model", EMBEDDING_SETTINGS.get("model") or "")}
        </div>
        <div class="grid two">
          {input_row("Embedding API key", "embedding_api_key", EMBEDDING_SETTINGS.get("api_key") or "dummy", "password")}
          {input_row("timeout sec", "timeout_sec", 3600, "number", 'min="60" step="60"')}
        </div>
        <div class="actions">{checkbox("include_no_answer", "answer_type=no_answerも評価対象に含める")}</div>
        <div class="actions"><button type="submit">RAGASを実行</button></div>
      </form>
    </section>
    {command_result_html(result)}
    """
    return layout("ragas", body)


def render_deepeval(result: dict[str, Any] | None = None) -> HTMLResponse:
    body = f"""
    <section class="panel">
      <div class="section-head">
        <h2>DeepEval</h2>
        <span class="muted small">local OpenAI-compatible endpoint向け</span>
      </div>
      <form method="post" action="/run-deepeval">
        {predictions_options()}
        <div class="grid three">
          {input_row("limit", "limit", 1, "number", 'min="0"')}
          {input_row("評価LLM base URL", "base_url", DEFAULT_EVAL_BASE_URL)}
          {input_row("評価LLM model", "model", EVAL_LLM_SETTINGS.get("model") or "local-eval-model")}
        </div>
        <div class="grid two">
          {input_row("評価LLM API key", "api_key", EVAL_LLM_SETTINGS.get("api_key") or "dummy", "password")}
          {input_row("timeout sec", "timeout_sec", 3600, "number", 'min="60" step="60"')}
        </div>
        <div class="actions">{checkbox("include_no_answer", "answer_type=no_answerも評価対象に含める")}</div>
        <div class="actions"><button type="submit">DeepEvalを実行</button></div>
      </form>
    </section>
    {command_result_html(result)}
    """
    return layout("deepeval", body)


def render_rules(result: dict[str, Any] | None = None) -> HTMLResponse:
    body = f"""
    <section class="panel">
      <div class="section-head">
        <h2>ルール採点</h2>
        <span class="muted small">must_include / must_not_include / no_answer抑制を確認します</span>
      </div>
      <form method="post" action="/run-rules">
        {predictions_options()}
        <div class="grid two">
          {input_row("limit", "limit", 0, "number", 'min="0"')}
          {input_row("timeout sec", "timeout_sec", 600, "number", 'min="60" step="60"')}
        </div>
        <div class="actions"><button type="submit">ルール採点を実行</button></div>
      </form>
    </section>
    {command_result_html(result)}
    """
    return layout("rules", body)


def render_chunks(result: dict[str, Any] | None = None) -> HTMLResponse:
    body = f"""
    <section class="panel">
      <div class="section-head">
        <h2>チャンキング</h2>
        <span class="muted small">Markdownから親チャンク・子チャンクを再生成します</span>
      </div>
      <form method="post" action="/run-chunks">
        {select_row("HTMLタグ処理", "html_tags", ["strip", "preserve"], "strip")}
        <p class="muted small">
          strip: HTMLコメント・HTMLタグ・作業用ラベルを除去して保存します。通常はこちらです。<br>
          preserve: HTMLコメントと作業用ラベルは除去し、HTMLタグは保存します。チャンキング検証用です。
        </p>
        <div class="grid two">
          {input_row("timeout sec", "timeout_sec", 600, "number", 'min="60" step="60"')}
          <div class="row"><label>出力</label><input value="chunks/parent_chunks.jsonl, chunks/child_chunks.jsonl" readonly></div>
        </div>
        <div class="actions"><button type="submit">チャンキングを実行</button></div>
      </form>
    </section>
    {command_result_html(result)}
    """
    return layout("chunks", body)


def table_html(rows: list[dict[str, Any]], empty_text: str = "表示できるデータがありません。", class_name: str = "") -> str:
    if not rows:
        return f'<p class="muted">{esc(empty_text)}</p>'
    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)
    head = "".join(f"<th>{esc(col)}</th>" for col in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{esc(row.get(col, ''))}</td>" for col in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    classes = "table-wrap" + (f" {esc(class_name)}" if class_name else "")
    return f'<div class="{classes}"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'


def json_panel(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        data = {"error": str(exc)}
    return f"""
    <section class="panel">
      <div class="section-head"><h2>{esc(path.name)}</h2></div>
      <pre>{esc(json.dumps(data, ensure_ascii=False, indent=2))}</pre>
    </section>
    """


def csv_panel(path: Path) -> str:
    rows = load_csv_rows(path)
    if not rows:
        return ""
    return f"""
    <section class="panel">
      <div class="section-head"><h2>{esc(path.name)}</h2><span class="muted small">先頭{len(rows)}件</span></div>
      {table_html(rows, class_name="wide-table")}
    </section>
    """


def render_summary(request: Request | None = None) -> HTMLResponse:
    run_dirs = [rel_path(path) for path in list_run_dirs()]
    selected = ""
    if request:
        selected = request.query_params.get("run") or ""
    selected = selected or (run_dirs[0] if run_dirs else "")
    run_dir = resolve_repo_path(selected) if selected else None
    selector = select_row("run", "run", run_dirs or [""], selected)
    panels = ""
    if run_dir and run_dir.exists():
        for name in ["summary.json", "ragas_summary.json", "deepeval_summary.json", "rule_summary.json", "manual_summary.json"]:
            panels += json_panel(run_dir / name)
        for name in ["predictions.csv", "ragas_scores.csv", "deepeval_scores.csv", "rule_scores.csv", "manual_scores.csv"]:
            panels += csv_panel(run_dir / name)
    body = f"""
    <section class="panel">
      <div class="section-head">
        <h2>結果サマリ</h2>
        <span class="muted small">run単位で自動評価と手動採点を確認します</span>
      </div>
      <form method="get" action="/">
        <input type="hidden" name="tab" value="summary">
        {selector}
        <div class="actions"><button type="submit" class="secondary">表示</button></div>
      </form>
    </section>
    {panels or '<section class="panel"><p class="muted">まだ評価runがありません。</p></section>'}
    """
    return layout("summary", body)


def manual_paths(run_dir: Path) -> tuple[Path, Path, Path]:
    return run_dir / "manual_scores.jsonl", run_dir / "manual_scores.csv", run_dir / "manual_summary.json"


def load_manual_scores(run_dir: Path) -> list[dict[str, Any]]:
    path, _, _ = manual_paths(run_dir)
    return read_jsonl(path)


def score_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def write_manual_summary(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    _, _, summary_path = manual_paths(run_dir)
    summary: dict[str, Any] = {
        "count": len(rows),
        "question_count": len({str(row.get("question_id") or "") for row in rows}),
        "reviewer_count": len({str(row.get("reviewer") or "") for row in rows}),
    }
    if rows:
        summary["avg_total_score"] = sum(score_float(row.get("total_score")) for row in rows) / len(rows)
        for key, label, _ in MANUAL_CRITERIA:
            field = f"{key}_score"
            summary[f"avg_{field}"] = sum(score_float(row.get(field)) for row in rows) / len(rows)
    else:
        summary["avg_total_score"] = 0.0
        for key, _, _ in MANUAL_CRITERIA:
            summary[f"avg_{key}_score"] = 0.0
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def save_manual_score(run_dir: Path, record: dict[str, Any]) -> None:
    jsonl_path, csv_path, _ = manual_paths(run_dir)
    existing = load_manual_scores(run_dir)
    key = (str(record.get("run_id") or ""), str(record.get("question_id") or ""), str(record.get("reviewer") or ""))
    rows = [
        row
        for row in existing
        if (str(row.get("run_id") or ""), str(row.get("question_id") or ""), str(row.get("reviewer") or "")) != key
    ]
    rows.append(record)
    rows.sort(key=lambda row: (str(row.get("question_id") or ""), str(row.get("reviewer") or "")))
    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)
    write_manual_summary(run_dir, rows)


def existing_manual_score(run_dir: Path, question_id: str, reviewer: str) -> dict[str, Any] | None:
    for row in load_manual_scores(run_dir):
        if str(row.get("question_id") or "") == question_id and str(row.get("reviewer") or "") == reviewer:
            return row
    return None


def score_select(name: str, selected: Any = "1.0") -> str:
    selected_text = str(selected)
    opts = []
    for value, label in SCORE_OPTIONS:
        sel = " selected" if value == selected_text else ""
        opts.append(f'<option value="{esc(value)}"{sel}>{esc(label)}</option>')
    return f'<select name="{esc(name)}">{"".join(opts)}</select>'


def render_detail(request: Request) -> HTMLResponse:
    files = [rel_path(path) for path in list_prediction_files()]
    selected_file = request.query_params.get("prediction_file") or (files[0] if files else "")
    reviewer = request.query_params.get("reviewer") or "developer"
    path = resolve_repo_path(selected_file) if selected_file else None
    rows = read_jsonl(path) if path and path.exists() else []
    ids = [str(row.get("question_id") or "") for row in rows]
    selected_id = request.query_params.get("question_id") or (ids[0] if ids else "")
    row = next((item for item in rows if str(item.get("question_id") or "") == selected_id), None)
    selector_form = f"""
    <section class="panel">
      <div class="section-head">
        <h2>詳細確認</h2>
        <span class="muted small">回答・根拠・debugを見ながら手動採点します</span>
      </div>
      <form method="get" action="/">
        <input type="hidden" name="tab" value="detail">
        {select_row("prediction file", "prediction_file", files or [""], selected_file)}
        {select_row("question_id", "question_id", ids or [""], selected_id)}
        {input_row("reviewer", "reviewer", reviewer)}
        <div class="actions"><button type="submit" class="secondary">表示</button></div>
      </form>
    </section>
    """
    if not row or not path:
        return layout("detail", selector_form + '<section class="panel"><p class="muted">確認できる予測結果がありません。</p></section>')
    run_dir = path.parent
    saved = existing_manual_score(run_dir, selected_id, reviewer) or {}
    sources = parse_json_field(row.get("sources_json"), [])
    debug = parse_json_field(row.get("debug_json"), {})
    criteria_html = []
    for key, label, help_text in MANUAL_CRITERIA:
        criteria_html.append(f"""
        <div class="score-card">
          <label>{esc(label)}</label>
          {score_select(f"score_{key}", saved.get(f"{key}_score", "1.0"))}
          <p class="muted small">{esc(help_text)}</p>
        </div>
        """)
    score_form = f"""
    <form method="post" action="/manual-score">
      <input type="hidden" name="prediction_file" value="{esc(selected_file)}">
      <input type="hidden" name="question_id" value="{esc(selected_id)}">
      {input_row("reviewer", "reviewer", reviewer)}
      <div class="score-grid">{"".join(criteria_html)}</div>
      <div class="row">
        <label>コメント</label>
        <textarea name="comment" placeholder="修正方針、気になった根拠、採点理由など">{esc(saved.get("comment") or "")}</textarea>
      </div>
      <div class="actions"><button type="submit">手動採点を保存</button></div>
    </form>
    """
    source_rows = []
    if isinstance(sources, list):
        for idx, source in enumerate(sources, start=1):
            if isinstance(source, dict):
                source_rows.append({
                    "#": idx,
                    "title": source.get("title") or "",
                    "score": source.get("score") or "",
                    "source_file": source.get("source_file") or "",
                    "heading_path": source.get("heading_path") or "",
                })
    detail = f"""
    <div class="detail-cols">
      <section class="panel answer-panel">
        <h2>回答確認</h2>
        <h3>Question</h3>
        <p>{esc(row.get("question"))}</p>
        <h3>Expected</h3>
        <div class="answer-render markdown-body">{markdown_block(row.get("expected_answer"))}</div>
        <h3>Actual</h3>
        <div class="answer-render markdown-body">{markdown_block(row.get("actual_answer"))}</div>
        <h3>Sources</h3>
        {table_html(source_rows, "根拠がありません。")}
        <details>
          <summary>sources_json</summary>
          <pre>{esc(json.dumps(sources, ensure_ascii=False, indent=2))}</pre>
        </details>
        <details>
          <summary>debug_json</summary>
          <pre>{esc(json.dumps(debug, ensure_ascii=False, indent=2))}</pre>
        </details>
      </section>
      <section class="panel">
        <h2>手動採点</h2>
        <p class="muted small">4基準を ○=1.0 / △=0.5 / ×=0.0 で採点します。保存先はこのrunの `manual_scores.*` です。</p>
        {score_form}
      </section>
    </div>
    """
    return layout("detail", selector_form + detail)


@rt("/", methods=["GET"])
async def home(request: Request) -> HTMLResponse:
    tab = request.query_params.get("tab") or "batch"
    if tab == "chunks":
        return render_chunks()
    if tab == "ragas":
        return render_ragas()
    if tab == "deepeval":
        return render_deepeval()
    if tab == "rules":
        return render_rules()
    if tab == "summary":
        return render_summary(request)
    if tab == "detail":
        return render_detail(request)
    return render_batch()


@rt("/run-batch", methods=["POST"])
async def run_batch(request: Request) -> HTMLResponse:
    form = await request.form()
    args = [
        sys.executable,
        "scripts/10_batch_inference.py",
        "--api",
        text_form(form, "api", rag_ask_url()),
        "--input",
        text_form(form, "input_path", default_eval_input()),
        "--limit",
        str(int_form(form, "limit", 1)),
        "--start",
        str(int_form(form, "start", 1)),
        "--workers",
        str(int_form(form, "workers", 1)),
        "--timeout-sec",
        str(int_form(form, "timeout_sec", 240)),
        "--retries",
        str(int_form(form, "retries", 1)),
    ]
    top_k = int_form(form, "top_k", 0)
    if top_k:
        args += ["--top-k", str(top_k)]
    if is_checked(form, "show_debug"):
        args.append("--show-debug")
    if is_checked(form, "resume"):
        args.append("--resume")
    if is_checked(form, "disable_search_tags"):
        args.append("--disable-search-tags")
    output_dir = text_form(form, "output_dir", "")
    if output_dir:
        args += ["--output-dir", output_dir]
    result = run_command(args, int_form(form, "timeout_sec", 3600))
    return render_batch(result)


@rt("/run-ragas", methods=["POST"])
async def run_ragas(request: Request) -> HTMLResponse:
    form = await request.form()
    args = [
        sys.executable,
        "scripts/11_eval_ragas.py",
        "--input",
        text_form(form, "prediction_file"),
        "--limit",
        str(int_form(form, "limit", 1)),
        "--base-url",
        text_form(form, "base_url", DEFAULT_EVAL_BASE_URL),
        "--model",
        text_form(form, "model", str(EVAL_LLM_SETTINGS.get("model") or "")),
        "--api-key",
        text_form(form, "api_key", str(EVAL_LLM_SETTINGS.get("api_key") or "dummy")),
        "--embedding-base-url",
        text_form(form, "embedding_base_url", DEFAULT_EMBEDDING_BASE_URL),
        "--embedding-model",
        text_form(form, "embedding_model", str(EMBEDDING_SETTINGS.get("model") or "")),
        "--embedding-api-key",
        text_form(form, "embedding_api_key", str(EMBEDDING_SETTINGS.get("api_key") or "dummy")),
    ]
    if is_checked(form, "include_no_answer"):
        args.append("--include-no-answer")
    result = run_command(args, int_form(form, "timeout_sec", 3600))
    return render_ragas(result)


@rt("/run-deepeval", methods=["POST"])
async def run_deepeval(request: Request) -> HTMLResponse:
    form = await request.form()
    args = [
        sys.executable,
        "scripts/12_eval_deepeval.py",
        "--input",
        text_form(form, "prediction_file"),
        "--limit",
        str(int_form(form, "limit", 1)),
        "--local-openai",
        "--base-url",
        text_form(form, "base_url", DEFAULT_EVAL_BASE_URL),
        "--model",
        text_form(form, "model", str(EVAL_LLM_SETTINGS.get("model") or "local-eval-model")),
        "--api-key",
        text_form(form, "api_key", str(EVAL_LLM_SETTINGS.get("api_key") or "dummy")),
    ]
    if is_checked(form, "include_no_answer"):
        args.append("--include-no-answer")
    result = run_command(args, int_form(form, "timeout_sec", 3600))
    return render_deepeval(result)


@rt("/run-rules", methods=["POST"])
async def run_rules(request: Request) -> HTMLResponse:
    form = await request.form()
    args = [
        sys.executable,
        "scripts/13_eval_rules.py",
        "--input",
        text_form(form, "prediction_file"),
    ]
    limit = int_form(form, "limit", 0)
    if limit:
        args += ["--limit", str(limit)]
    result = run_command(args, int_form(form, "timeout_sec", 600))
    return render_rules(result)


@rt("/run-chunks", methods=["POST"])
async def run_chunks(request: Request) -> HTMLResponse:
    form = await request.form()
    html_tags = text_form(form, "html_tags", "strip")
    if html_tags not in {"strip", "preserve"}:
        html_tags = "strip"
    args = [
        sys.executable,
        "scripts/01_make_chunks.py",
        "--html-tags",
        html_tags,
    ]
    result = run_command(args, int_form(form, "timeout_sec", 600))
    return render_chunks(result)


@rt("/manual-score", methods=["POST"])
async def manual_score(request: Request) -> RedirectResponse:
    form = await request.form()
    prediction_file = text_form(form, "prediction_file")
    question_id = text_form(form, "question_id")
    reviewer = text_form(form, "reviewer", "developer") or "developer"
    path = resolve_repo_path(prediction_file)
    run_dir = path.parent
    scores = {key: score_float(form.get(f"score_{key}")) for key, _, _ in MANUAL_CRITERIA}
    total = sum(scores.values()) / len(MANUAL_CRITERIA)
    record: dict[str, Any] = {
        "run_id": run_dir.name,
        "question_id": question_id,
        "reviewer": reviewer,
        "total_score": round(total, 4),
        "comment": text_form(form, "comment", ""),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for key, _, _ in MANUAL_CRITERIA:
        record[f"{key}_score"] = scores[key]
    save_manual_score(run_dir, record)
    query = urlencode({
        "tab": "detail",
        "prediction_file": prediction_file,
        "question_id": question_id,
        "reviewer": reviewer,
    })
    return RedirectResponse(f"/?{query}", status_code=303)


@rt("/health", methods=["GET"])
async def health(_: Request) -> HTMLResponse:
    return HTMLResponse("ok")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=get_required_url_value("dev_eval_bind_host"),
        port=get_url_number("dev_eval_bind_port"),
    )
