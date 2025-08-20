# dashboard/app.py
import os, json, base64
from typing import List, Dict, Any, Tuple

import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go

# ---------------------- Config ----------------------
USE_LLM = bool(os.environ.get("OPENAI_API_KEY"))
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# ------------------ Data helpers -------------------
def normalize_history(raw: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    needed = ["timestamp", "inventory", "cash", "mid_price", "pnl"]
    for col in needed:
        if col not in df.columns:
            df[col] = 0.0
    # coerce numerics
    for c in ["timestamp", "inventory", "cash", "mid_price", "pnl"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.fillna(0.0)
    return df

def parse_upload(contents: str) -> List[Dict[str, Any]]:
    if not contents:
        return []
    _type, b64 = contents.split(",", 1)
    decoded = base64.b64decode(b64)

    # JSON array or {"history": [...]}
    try:
        data = json.loads(decoded)
        if isinstance(data, dict) and "history" in data:
            data = data["history"]
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # NDJSON fallback
    rows = []
    try:
        for line in decoded.decode("utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    except Exception:
        return []

def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return dict(steps=0, final_pnl=0.0, avg_abs_inv=0.0,
                    max_drawdown=0.0, vol=0.0, sharpe=0.0)
    pnl = df["pnl"].astype(float)
    inv = df["inventory"].astype(float)

    steps = int(len(df))
    final_pnl = float(pnl.iloc[-1])
    avg_abs_inv = float(inv.abs().mean())

    # drawdown on PnL path
    roll_max = pnl.cummax()
    dd = pnl - roll_max
    max_dd = float(dd.min())

    # naive return stats from pnl differences
    rets = pnl.diff().dropna()
    vol = float(rets.std()) if not rets.empty else 0.0
    sharpe = float(rets.mean() / vol) if vol > 1e-12 else 0.0

    return dict(steps=steps, final_pnl=final_pnl, avg_abs_inv=avg_abs_inv,
                max_drawdown=max_dd, vol=vol, sharpe=sharpe)

def build_context_text(df: pd.DataFrame, metrics: Dict[str, Any], max_rows: int = 120):
    """
    Provide the model with a compact, machine-usable context.
    We include: schema, summary metrics, and a tail slice of rows.
    """
    ctx = {
        "schema": {
            "columns": list(df.columns) if not df.empty else ["timestamp","inventory","cash","mid_price","pnl"],
            "notes": "Rows represent strategy state per step."
        },
        "metrics": metrics,
        "tail_rows": df.tail(max_rows).to_dict(orient="records") if not df.empty else [],
        "first_timestamp": float(df["timestamp"].iloc[0]) if not df.empty else None,
        "last_timestamp": float(df["timestamp"].iloc[-1]) if not df.empty else None,
    }
    # keep prompt small enough for any small model
    text = json.dumps(ctx, separators=(",", ":"), ensure_ascii=False)
    MAX = 14000
    return text[:MAX]

# ------------------- Unified LLM -------------------
def llm_answer_unified(question: str, context_json: str) -> str:
    """
    Single path: always call the LLM.
    - If the question is data-related, the model can compute from context_json.
    - If it is conceptual, it ignores the data and answers generally.
    No hard-coded answers or keyword routing.
    """
    if not USE_LLM:
        return ("LLM is not configured (missing OPENAI_API_KEY). "
                "Set it and retry to get automatic answers from data and concepts.")

    try:
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY
        system_msg = (
            "You are a quantitative trading assistant for a market-making dashboard. "
            "You will receive a JSON context (schema, summary metrics, and recent rows). "
            "When the user's question can be answered from the JSON, compute it precisely and report the value(s). "
            "When the question is conceptual or the data is insufficient, answer based on your general knowledge. "
            "Be concise and do NOT reveal chain-of-thought—just provide the final answer and key figures."
        )
        user_msg = (
            f"CONTEXT_JSON:\n{context_json}\n\n"
            f"QUESTION:\n{question}\n\n"
            "Instructions:\n"
            "- If the answer depends on the JSON, quote the computed figures clearly.\n"
            "- If the JSON is irrelevant, give a clear conceptual answer.\n"
            "- Keep it under ~6 lines unless more detail is necessary."
        )
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.25,
            max_tokens=500,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        return content if content else "I couldn't generate an answer."
    except Exception as e:
        return f"(LLM error: {e})"

# ------------------- Plotting ---------------------
def make_figures(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure, go.Figure, str]:
    if df.empty:
        empty = go.Figure()
        return empty, empty, empty, "Upload a history JSON (e.g., data/history_rule_based.json)."

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df["timestamp"], y=df["pnl"], mode="lines+markers", name="P&L"))
    fig1.add_trace(go.Scatter(x=df["timestamp"], y=df["inventory"], mode="lines+markers",
                              name="Inventory", yaxis="y2"))
    fig1.update_layout(
        height=360,
        title="P&L and Inventory over Time",
        xaxis_title="Timestamp",
        yaxis_title="P&L",
        yaxis2=dict(title="Inventory", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, b=40),
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["timestamp"], y=df["mid_price"], mode="lines", name="Mid Price"))
    fig2.update_layout(title="Mid Price Over Time", xaxis_title="Timestamp",
                       yaxis_title="Mid Price", height=320)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df["inventory"], y=df["pnl"], mode="markers", name="Inv vs P&L"))
    fig3.update_layout(title="Inventory vs P&L", xaxis_title="Inventory",
                       yaxis_title="P&L", height=320)

    m = compute_metrics(df)
    summary = (
        f"Steps: {m['steps']} | Final P&L: {m['final_pnl']:.2f} | "
        f"Avg |Inventory|: {m['avg_abs_inv']:.2f} | Max DD: {m['max_drawdown']:.2f} | "
        f"Vol(P&L diffs): {m['vol']:.4f} | Sharpe: {m['sharpe']:.3f}"
    )
    return fig1, fig2, fig3, summary

# ---------------------- UI -----------------------
app = dash.Dash(__name__)
app.title = "Market Maker Dashboard"

app.layout = html.Div(
    style={"fontFamily": "Segoe UI, Arial, sans-serif", "maxWidth": "1200px", "margin": "34px auto"},
    children=[
        html.H2("Limit Order Book Market Maker Dashboard", style={"marginBottom": "14px"}),

        # Upload
        html.Div(
            [
                html.Label("Upload strategy history JSON", style={"display": "block", "marginBottom": "6px"}),
                dcc.Upload(
                    id="upload-history",
                    children=html.Div(["Drag & drop or ", html.A("select a file")]),
                    style={
                        "width": "100%",
                        "height": "64px",
                        "lineHeight": "64px",
                        "borderWidth": "2px",
                        "borderStyle": "dashed",
                        "borderRadius": "6px",
                        "textAlign": "center",
                    },
                    multiple=False,
                ),
                html.Div(id="upload-status", style={"marginTop": "8px", "color": "#444"}),
            ],
            style={"marginBottom": "16px"},
        ),

        # Stores
        dcc.Store(id="history-store"),
        dcc.Store(id="chat-store", data=[]),

        # Tabs: Analytics (charts) | Chat
        dcc.Tabs(
            id="main-tabs",
            value="tab-analytics",
            children=[
                dcc.Tab(label="Analytics", value="tab-analytics", children=[
                    html.Div([
                        dcc.Graph(id="pnl-inv-graph"),
                        dcc.Graph(id="midprice-graph"),
                        dcc.Graph(id="inv-vs-pnl"),
                        html.Div(id="summary", style={"marginTop": "10px", "fontSize": "14px", "color": "#333"}),
                    ])
                ]),
                dcc.Tab(label="Ask the Data", value="tab-chat", children=[
                    html.Div(
                        style={"maxWidth": "900px", "margin": "16px auto",
                               "border": "1px solid #e5e5e5", "borderRadius": "8px", "padding": "12px"},
                        children=[
                            html.H4("Ask the Data", style={"margin": "0 0 6px 0"}),
                            html.Small(
                                ("LLM is ON" if USE_LLM else "LLM is OFF"),
                                style={"color": "#666"}
                            ),
                            html.Div(
                                id="chat-thread",
                                style={
                                    "height": "520px",
                                    "overflowY": "auto",
                                    "border": "1px solid #f0f0f0",
                                    "borderRadius": "6px",
                                    "padding": "8px",
                                    "background": "#fafafa",
                                    "marginTop": "8px",
                                },
                            ),
                            html.Div(
                                style={"display": "flex", "gap": "8px", "marginTop": "10px"},
                                children=[
                                    dcc.Input(
                                        id="user-input",
                                        placeholder="Ask anything (data or concept). e.g., 'What is the final PnL?' or 'Why cancel and replace quotes?'",
                                        type="text",
                                        style={"flex": 1},
                                    ),
                                    html.Button("Ask", id="send-btn", n_clicks=0),
                                ],
                            ),
                        ],
                    )
                ]),
            ],
        ),
    ],
)

# -------------------- Callbacks -------------------
@app.callback(
    Output("history-store", "data"),
    Output("upload-status", "children"),
    Input("upload-history", "contents"),
    State("upload-history", "filename"),
)
def load_history(contents, filename):
    if not contents:
        return dash.no_update, "No file uploaded yet."
    rows = parse_upload(contents)
    if not rows:
        return dash.no_update, "Could not parse file. Expecting a JSON array of state dicts."
    df = normalize_history(rows)
    return df.to_dict("list"), f"Loaded {len(df)} steps from {filename}."

@app.callback(
    Output("pnl-inv-graph", "figure"),
    Output("midprice-graph", "figure"),
    Output("inv-vs-pnl", "figure"),
    Output("summary", "children"),
    Input("history-store", "data"),
)
def update_plots(data):
    if not data:
        empty = go.Figure()
        return empty, empty, empty, "Upload a history JSON (e.g., data/history_rule_based.json)."
    df = pd.DataFrame(data)
    return make_figures(df)

def _render_chat(messages: List[Dict[str, str]]) -> List[html.Div]:
    bubbles = []
    for m in messages[-200:]:
        is_user = (m.get("role") == "user")
        style = {
            "padding": "8px 10px",
            "borderRadius": "8px",
            "margin": "6px 0",
            "maxWidth": "85%",
            "whiteSpace": "pre-wrap",
            "background": "#dfe9ff" if is_user else "#fff",
            "border": "1px solid #e8e8e8",
            "alignSelf": "flex-end" if is_user else "flex-start",
        }
        who = "You" if is_user else "Analyst"
        bubbles.append(html.Div([html.Small(who, style={"color": "#666"}), html.Div(m.get("content",""))], style=style))
    return bubbles

@app.callback(
    Output("chat-thread", "children"),
    Output("chat-store", "data"),
    Output("user-input", "value"),
    Input("send-btn", "n_clicks"),
    State("user-input", "value"),
    State("chat-store", "data"),
    State("history-store", "data"),
    prevent_initial_call=True,
)
def handle_chat(n_clicks, user_text, chat_hist, data):
    if not user_text or not user_text.strip():
        return _render_chat(chat_hist or []), (chat_hist or []), ""

    chat_hist = (chat_hist or []) + [{"role": "user", "content": user_text.strip()}]

    # Build context from uploaded data (if any)
    df = pd.DataFrame(data) if data else pd.DataFrame()
    df = normalize_history(df.to_dict("records")) if not df.empty else df
    metrics = compute_metrics(df)
    context_text = build_context_text(df, metrics)

    # Single unified LLM call (no keyword routing, no hard-coded answers)
    reply = llm_answer_unified(user_text, context_text)

    chat_hist.append({"role": "assistant", "content": reply})
    return _render_chat(chat_hist), chat_hist, ""

if __name__ == "__main__":
    # Dash 3+: app.run
    app.run(debug=True, port=8050)
