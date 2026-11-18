"""Page 2 - Layer 2: Base Portfolio."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc

from utils.data_loader import (
    load_portfolio_returns,
    compute_portfolio_stats,
    compute_monthly_heatmap,
    load_tearsheet,
)
from utils.chart_helpers import (
    apply_layout,
    GOLD, BLUE, TEAL, RED, GREEN, WHITE, SUBTEXT, SLATE, ORANGE,
    CHART_CFG, CARD_BG,
    cumulative, drawdown, rolling_sharpe, add_drawdown_shading,
)

MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
              7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _cumulative_return_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    s = df["base_return"].dropna()
    cum = cumulative(s) * 100
    fig.add_scatter(
        x=cum.index, y=cum.round(3),
        name="Base CB Portfolio", mode="lines",
        line=dict(color=GOLD, width=2.5),
        hovertemplate="<b>Base CB Portfolio</b><br>%{x|%b %Y}: %{y:+.2f}%<extra></extra>",
    )
    add_drawdown_shading(fig, s, threshold=-0.05)

    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.2)")
    apply_layout(fig, "Cumulative Return - Base CB Portfolio", height=400)
    fig.update_yaxes(title_text="Cumulative Return (%)", title_font=dict(color=SUBTEXT, size=11),
                     ticksuffix="%")
    fig.update_layout(
        showlegend=False,
        annotations=[dict(
            x=0.01, y=0.97, xref="paper", yref="paper",
            text="<i>Shaded = drawdown > 5%</i>",
            showarrow=False, font=dict(color=SUBTEXT, size=10),
        )],
    )
    return fig


def _performance_table(df: pd.DataFrame) -> dbc.Table:
    cols = {
        "base_return": "Base Portfolio",
    }
    metrics = ["Ann. Return", "Ann. Vol", "Sharpe", "Max DD", "Calmar", "Hit Rate"]
    header = html.Thead(html.Tr(
        [html.Th("Metric")] + [html.Th(label) for label in cols.values()],
        className="table-dark",
    ))

    stats = {col: compute_portfolio_stats(df, col) for col in cols if col in df.columns}
    rows = []
    for m in metrics:
        cells = [html.Td(m, style={"fontWeight": 600, "color": SUBTEXT})]
        for col in cols:
            if col not in stats:
                cells.append(html.Td("N/A"))
                continue
            val = stats[col].get(m, "N/A")
            # Color logic for key metrics
            style = {}
            if m == "Sharpe":
                v = float(val) if val != "N/A" else 0
                style = {"color": GREEN if v > 1.0 else (GOLD if v > 0.4 else RED), "fontWeight": 700}
            elif m == "Max DD":
                style = {"color": RED, "fontWeight": 600}
            elif m == "Ann. Return":
                v = float(val.replace("%", "").replace("+", "")) if val != "N/A" else 0
                style = {"color": GREEN if v > 10 else (GOLD if v > 0 else RED), "fontWeight": 600}
            cells.append(html.Td(val, style=style))
        rows.append(html.Tr(cells))

    return dbc.Table(
        [header, html.Tbody(rows)],
        bordered=False, hover=True, size="sm",
        className="table-dark table-hover align-middle mb-0",
        style={"fontSize": "0.82rem"},
    )


def _monthly_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = compute_monthly_heatmap(df, "base_return")

    months = [MONTH_ABBR.get(m, str(m)) for m in pivot.columns]
    years = [str(y) for y in pivot.index]
    z = pivot.values.tolist()

    text = [
        [f"{v:+.1f}%" if not np.isnan(v) else "" for v in row]
        for row in z
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=months,
        y=years,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        colorscale=[
            [0.0, "#C62828"],
            [0.35, "#EF5350"],
            [0.5, "#455A64"],
            [0.65, "#43A047"],
            [1.0, "#1B5E20"],
        ],
        zmid=0,
        showscale=True,
        colorbar=dict(
            title=dict(text="%", font=dict(color=SUBTEXT)),
            tickfont=dict(color=SUBTEXT),
            len=0.8,
        ),
        hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:+.2f}%<extra></extra>",
    ))

    apply_layout(fig, "Monthly Return Heatmap - Base Portfolio (%)", height=360)
    fig.update_xaxes(side="top", tickfont=dict(size=11))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
    fig.update_layout(margin=dict(l=50, r=80, t=60, b=20))
    return fig


def _rolling_sharpe_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    rs = rolling_sharpe(df["base_return"].dropna(), window=12).dropna()
    fig.add_scatter(
        x=rs.index, y=rs.round(3),
        name="Base Portfolio", mode="lines",
        line=dict(color=GOLD, width=2),
        hovertemplate="<b>Base Portfolio</b><br>%{x|%b %Y}: %{y:.2f}<extra></extra>",
    )
    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.2)")
    fig.add_hline(y=1.0, line_width=1,
                  line_color=f"rgba({int(GOLD[1:3],16)},{int(GOLD[3:5],16)},{int(GOLD[5:7],16)},0.3)",
                  line_dash="dot")
    apply_layout(fig, "Rolling 12-Month Sharpe Ratio", height=320)
    fig.update_yaxes(title_text="Sharpe", title_font=dict(color=SUBTEXT, size=11))
    fig.update_layout(showlegend=False)
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _card(title: str, content) -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(
            title,
            style={"background": "rgba(255,255,255,0.04)", "fontWeight": 600,
                   "color": WHITE, "fontSize": "0.85rem", "letterSpacing": "0.03em"},
        ),
        dbc.CardBody(content, className="p-0"),
    ], className="mb-3 border-0",
       style={"background": CARD_BG, "borderRadius": "8px", "overflow": "hidden"})


def layout() -> html.Div:
    df = load_portfolio_returns()
    ts = load_tearsheet()

    return html.Div([
        html.Div([
            html.H4("Layer 2 - Base CB Portfolio", className="mb-1",
                    style={"color": GOLD, "fontWeight": 700}),
            html.P("Long-only quality + momentum and low-vol sleeves · net of 30 bp costs · vol-scaled · "
                   "Sharpe 0.49 · Max DD -22.9%",
                   className="mb-0", style={"color": SUBTEXT, "fontSize": "0.88rem"}),
        ], className="mb-4"),

        # Cumulative return chart
        dbc.Row([
            dbc.Col(_card("Cumulative Return - All Strategy Layers",
                          dcc.Graph(figure=_cumulative_return_chart(df), config=CHART_CFG)),
                    md=12),
        ]),

        # Performance table + rolling Sharpe
        dbc.Row([
            dbc.Col(_card("Performance Statistics",
                          html.Div(_performance_table(df), className="p-2")),
                    md=5),
            dbc.Col(_card("Rolling 12-Month Sharpe Ratio",
                          dcc.Graph(figure=_rolling_sharpe_chart(df), config=CHART_CFG)),
                    md=7),
        ]),

        # Monthly heatmap
        dbc.Row([
            dbc.Col(_card("Monthly Return Heatmap - Base Portfolio",
                          dcc.Graph(figure=_monthly_heatmap(df), config=CHART_CFG)),
                    md=12),
        ]),
    ], className="p-3")


def register_callbacks(app):
    pass
