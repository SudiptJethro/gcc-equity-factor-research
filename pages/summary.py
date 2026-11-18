"""Page 4 - Research Summary."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc

from utils.data_loader import (
    load_tearsheet,
    load_fm_summary,
    compute_thesis_scorecard,
)
from utils.chart_helpers import (
    apply_layout,
    GOLD, BLUE, TEAL, RED, GREEN, WHITE, SUBTEXT, SLATE, ORANGE, PURPLE,
    CHART_CFG, CARD_BG,
)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _strategy_waterfall(ts: pd.DataFrame) -> go.Figure:
    def _parse_pct(s):
        if isinstance(s, (int, float)):
            return float(s)
        return float(str(s).replace("%", "").replace("+", "").replace("−", "-").replace("−", "-"))

    portfolios = []
    for _, r in ts.iterrows():
        name = str(r.get("Portfolio", r.iloc[0]))
        if "alpha" in name.lower() or "ceiling" in name.lower() or "vs" in name.lower():
            continue
        portfolios.append({
            "name": name.replace("Layer ", "L").split("(")[0].strip(),
            "sharpe": _parse_pct(r.get("Sharpe", 0)),
            "ann_ret": _parse_pct(r.get("Ann. Ret", r.get("Ann. Ret%", 0))),
            "max_dd": _parse_pct(r.get("Max DD", 0)),
        })

    if not portfolios:
        portfolios = [
            {"name": "Base Portfolio", "sharpe": 0.49, "ann_ret": 5.35, "max_dd": -22.9},
        ]

    names = [p["name"] for p in portfolios]
    fig = go.Figure()

    # Sharpe bars
    fig.add_bar(
        x=[p["sharpe"] for p in portfolios],
        y=names, orientation="h",
        name="Sharpe Ratio",
        marker_color=[GOLD][:len(portfolios)],
        text=[f"{p['sharpe']:.2f}" for p in portfolios],
        textposition="outside",
        textfont=dict(color=WHITE, size=12),
        hovertemplate="<b>%{y}</b><br>Sharpe: %{x:.2f}<extra></extra>",
    )

    apply_layout(fig, "Strategy Comparison - Sharpe Ratio", height=260)
    fig.update_xaxes(title_text="Sharpe Ratio", title_font=dict(color=SUBTEXT, size=11))
    fig.update_yaxes(tickfont=dict(size=11))
    fig.update_layout(showlegend=False, margin=dict(l=180, r=60, t=44, b=40))
    return fig


def _return_comparison_chart(ts: pd.DataFrame) -> go.Figure:
    def _parse_pct(s):
        if isinstance(s, (int, float)):
            return float(s)
        return float(str(s).replace("%", "").replace("+", "").replace("−", "-").replace("−", "-"))

    portfolios = []
    for _, r in ts.iterrows():
        name = str(r.get("Portfolio", r.iloc[0]))
        if "alpha" in name.lower() or "vs" in name.lower():
            continue
        portfolios.append({
            "name": name.replace("Layer ", "L").split("(")[0].strip(),
            "ann_ret": _parse_pct(r.get("Ann. Ret", r.get("Ann. Ret%", 0))),
            "max_dd": abs(_parse_pct(r.get("Max DD", 0))),
        })

    if not portfolios:
        portfolios = [
            {"name": "Base Portfolio", "ann_ret": 5.35, "max_dd": 22.9},
        ]

    names = [p["name"] for p in portfolios]
    colors = [GOLD][:len(portfolios)]
    fig = go.Figure()

    fig.add_bar(
        x=[p["ann_ret"] for p in portfolios],
        y=names, orientation="h",
        name="Ann. Return (%)",
        marker_color=colors,
        text=[f"{p['ann_ret']:+.1f}%" for p in portfolios],
        textposition="outside",
        textfont=dict(color=WHITE, size=12),
        hovertemplate="<b>%{y}</b><br>Ann. Return: %{x:+.1f}%<extra></extra>",
    )

    apply_layout(fig, "Strategy Comparison - Annualised Return", height=260)
    fig.update_xaxes(title_text="Ann. Return (%)", ticksuffix="%",
                     title_font=dict(color=SUBTEXT, size=11))
    fig.update_layout(showlegend=False, margin=dict(l=180, r=80, t=44, b=40))
    return fig


# ---------------------------------------------------------------------------
# GAP 4 / Thesis Scorecard helpers
# ---------------------------------------------------------------------------

def _scorecard_panel_a(panel_a: pd.DataFrame) -> dbc.Table:
    header = html.Thead(html.Tr([
        html.Th("Factor"),
        html.Th("Significant (|t|>2)"),
        html.Th("t-stat (NW-4)"),
        html.Th("Direction"),
        html.Th("Strongest Exchange"),
    ], className="table-dark"))

    rows = []
    for _, r in panel_a.iterrows():
        sig = r["Significant (|t|>2)"] == "Yes"
        rows.append(html.Tr([
            html.Td(r["Factor"], style={
                "fontWeight": 700,
                "color": GOLD if sig else SUBTEXT,
            }),
            html.Td(
                r["Significant (|t|>2)"],
                style={"color": GREEN if sig else RED, "fontWeight": 700},
            ),
            html.Td(r["t-stat (NW-4)"]),
            html.Td(
                r["Direction"],
                style={"color": TEAL if r["Direction"] == "Positive" else ORANGE},
            ),
            html.Td(r["Strongest Exchange"]),
        ]))

    return dbc.Table(
        [header, html.Tbody(rows)],
        bordered=False, hover=True, size="sm",
        className="table-dark table-hover align-middle mb-0",
        style={"fontSize": "0.82rem"},
    )


def _scorecard_panel_b(panel_b: pd.DataFrame) -> dbc.Table:
    header = html.Thead(html.Tr([
        html.Th("Factor"),
        html.Th("Max Interaction |t|"),
        html.Th("Cross-Exch. Significant (|t|>2)"),
        html.Th("Outlier Exchange"),
    ], className="table-dark"))

    rows = []
    for _, r in panel_b.iterrows():
        sig = r["Cross-Exch. Significant (|t|>2)"] == "Yes"
        rows.append(html.Tr([
            html.Td(r["Factor"], style={"fontWeight": 700, "color": GOLD}),
            html.Td(r["Max Interaction |t|"],
                    style={"color": GOLD if sig else WHITE, "fontWeight": 600 if sig else 400}),
            html.Td(
                r["Cross-Exch. Significant (|t|>2)"],
                style={"color": GREEN if sig else SUBTEXT, "fontWeight": 700 if sig else 400},
            ),
            html.Td(r["Outlier Exchange"]),
        ]))

    return dbc.Table(
        [header, html.Tbody(rows)],
        bordered=False, hover=True, size="sm",
        className="table-dark table-hover align-middle mb-0",
        style={"fontSize": "0.82rem"},
    )


# ---------------------------------------------------------------------------
# Stat cards
# ---------------------------------------------------------------------------

def _stat_card(title: str, value: str, subtitle: str, color: str = GOLD) -> dbc.Col:
    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.P(title, className="mb-1",
                       style={"color": SUBTEXT, "fontSize": "0.78rem",
                              "fontWeight": 600, "textTransform": "uppercase",
                              "letterSpacing": "0.06em"}),
                html.H3(value, className="mb-1",
                        style={"color": color, "fontWeight": 800,
                               "fontFamily": "Inter, sans-serif"}),
                html.P(subtitle, className="mb-0",
                       style={"color": SUBTEXT, "fontSize": "0.76rem"}),
            ], className="p-3"),
        ], className="border-0 h-100",
           style={"background": CARD_BG, "borderRadius": "8px"}),
        md=3, className="mb-3",
    )


# ---------------------------------------------------------------------------
# Methodology text
# ---------------------------------------------------------------------------

METHODOLOGY_TEXT = html.Div([
    html.H6("Research Framework", style={"color": GOLD, "fontWeight": 700, "marginBottom": "12px"}),
    html.P([
        html.Strong("Universe & Data: "), "167 GCC-listed equities across Tadawul (Saudi Arabia), "
        "ADX, DFM, QSE, and Boursa Kuwait. Monthly data March 2015 – May 2026 (~135 months). "
        "Prices from Yahoo Finance (TradingView for ADX/DFM gaps); fundamentals are a single "
        "December 2025 Yahoo Finance snapshot, so value and quality results are upper bounds.",
    ], className="mb-2", style={"fontSize": "0.84rem", "color": WHITE}),
    html.P([
        html.Strong("Layer 1 - Factor Analysis: "), "Four factors - Momentum (12-1 month), "
        "Quality (ROE, earnings stability and operating margin), Value (P/B + EV/EBITDA, inverted for GCC), "
        "Low-Vol (rolling 252-day σ). Fama-MacBeth (1973) cross-sectional OLS each month; "
        "time-series means tested with Newey-West (4-lag) t-statistics.",
    ], className="mb-2", style={"fontSize": "0.84rem", "color": WHITE}),
    html.P([
        html.Strong("Layer 2 - CB Portfolio: "), "Sleeve A buys top-20 quality+momentum stocks "
        "(vol-scaled to 10% target), Sleeve B holds top-20 lowest-vol stocks (8% target). "
        "30 bps round-trip transaction costs and a drawdown circuit breaker.",
    ], className="mb-2", style={"fontSize": "0.84rem", "color": WHITE}),
    html.P([
        html.Strong("Key Finding: "), "Quality is the strongest GCC factor premium (t = 2.98, "
        "look-ahead inflated); momentum is marginal (t = 1.81) but has the cleanest quintile "
        "spread. Value exhibits a sign-flip (expensive stocks outperform). "
        "Base CB portfolio achieves Sharpe 0.49 with Max DD -22.9% over the full period.",
    ], className="mb-0", style={"fontSize": "0.84rem", "color": TEAL, "fontStyle": "italic"}),
], style={"lineHeight": "1.65"})


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _card(title, content) -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(title, style={
            "background": "rgba(255,255,255,0.04)", "fontWeight": 600,
            "color": WHITE, "fontSize": "0.85rem", "letterSpacing": "0.03em",
        }),
        dbc.CardBody(content, className="p-3"),
    ], className="mb-3 border-0",
       style={"background": CARD_BG, "borderRadius": "8px", "overflow": "hidden"})


def layout() -> html.Div:
    ts = load_tearsheet()
    panel_a, panel_b = compute_thesis_scorecard()
    fm = load_fm_summary().set_index("factor")

    def _p(t):
        t = abs(t)
        return "p < 1%" if t >= 2.576 else "p < 5%" if t >= 1.96 else "p < 10%" if t >= 1.645 else "n.s."

    return html.Div([
        html.Div([
            html.H4("Research Summary", className="mb-1",
                    style={"color": GOLD, "fontWeight": 700}),
            html.P("GCC Equity Factor Research - factor premia significance and portfolio performance",
                   className="mb-0", style={"color": SUBTEXT, "fontSize": "0.88rem"}),
        ], className="mb-4"),

        # Stat cards
        dbc.Row([
            _stat_card("Base Sharpe", "0.49", "CB portfolio (vol-scaled)", GOLD),
            _stat_card("FM t-stat Quality", f"{fm.loc['QUALITY', 't_uni_nw3']:+.2f}",
                       f"{_p(fm.loc['QUALITY', 't_uni_nw3'])} (univariate, NW-4 lag)", GREEN),
            _stat_card("FM t-stat Momentum", f"{fm.loc['MOMENTUM', 't_uni_nw3']:+.2f}",
                       f"{_p(fm.loc['MOMENTUM', 't_uni_nw3'])} (univariate, NW-4 lag)", TEAL),
            _stat_card("Quality IC", f"{fm.loc['QUALITY', 'IC_mean']:.3f}",
                       f"ICIR annualised {fm.loc['QUALITY', 'ICIR_ann']:.2f}", BLUE),
        ], className="mb-2"),

        # Strategy comparison charts
        dbc.Row([
            dbc.Col(_card("Sharpe Ratio Comparison",
                          dcc.Graph(figure=_strategy_waterfall(ts), config=CHART_CFG,
                                    className="p-0")),
                    md=6),
            dbc.Col(_card("Annualised Return Comparison",
                          dcc.Graph(figure=_return_comparison_chart(ts), config=CHART_CFG,
                                    className="p-0")),
                    md=6),
        ]),

        # GAP 4: Thesis Scorecard
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Thesis Scorecard", style={
                    "background": "rgba(201,168,76,0.10)",
                    "fontWeight": 700, "color": GOLD,
                    "fontSize": "0.88rem", "letterSpacing": "0.04em",
                }),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6(
                                "Panel A - Claim 1: Factor Premia Significance",
                                style={"color": WHITE, "fontWeight": 600,
                                       "marginBottom": "10px", "fontSize": "0.84rem"},
                            ),
                            _scorecard_panel_a(panel_a),
                        ], md=6),
                        dbc.Col([
                            html.H6(
                                "Panel B - Claim 2: Cross-Exchange Heterogeneity",
                                style={"color": WHITE, "fontWeight": 600,
                                       "marginBottom": "10px", "fontSize": "0.84rem"},
                            ),
                            _scorecard_panel_b(panel_b),
                        ], md=6),
                    ]),
                ], className="p-3"),
            ], className="mb-3 border-0",
               style={"background": CARD_BG, "borderRadius": "8px"}),
                    md=12),
        ]),

        # Methodology
        dbc.Row([
            dbc.Col(_card("Methodology", METHODOLOGY_TEXT), md=12),
        ]),
    ], className="p-3")


def register_callbacks(app):
    pass
