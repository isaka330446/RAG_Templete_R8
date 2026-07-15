# FastHTMLチャットアプリ

`rag_template_meeting_timeline` には、Streamlit版とは別にFastHTML版のチャットアプリを用意しています。

## セットアップ

```powershell
cd rag_template_meeting_timeline
python -m venv .venv-fasthtml
.\.venv-fasthtml\Scripts\activate
pip install -r requirements_fasthtml.txt
```

## 起動

先にRAG APIを起動してください。

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

左側固定の根拠表示版:

```powershell
python -m uvicorn app_fasthtml_saas.main:app --host 0.0.0.0 --port 8622
python -m uvicorn app_fasthtml_modern.main:app --host 0.0.0.0 --port 8623
```

ポップアップ根拠表示版:

```powershell
python -m uvicorn app_fasthtml_saas_popup.main:app --host 0.0.0.0 --port 8632
python -m uvicorn app_fasthtml_modern_popup.main:app --host 0.0.0.0 --port 8633
```

RAG APIのURLを変える場合は環境変数を指定します。

```powershell
$env:RAG_API_URL = "http://127.0.0.1:8000/ask"
```

## 根拠リンク

回答本文中の `根拠1`、`根拠①`、`[2]`、`根拠: 会議資料名` のような表記を、表示中の根拠へリンク化します。
タイトルが曖昧で複数候補に当たる場合は誤リンクを避け、回答末尾の `根拠リンク` 一覧から確認できるようにします。
