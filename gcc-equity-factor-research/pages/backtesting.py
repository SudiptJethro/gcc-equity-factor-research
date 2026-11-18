"""Page 3 - Layer 3: Backtesting Engine.

Monthly rebalancing, 50bps round-trip TC, Sharpe/Sortino/MaxDD,
Fama-MacBeth cross-sectional OLS, IC t-stats, sub-period stability.
Research question: do GCC factors generate statistically significant alpha?
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

from utils.data_loader import (
    load_backtest_summary,
    load_fm_summary,
    load_ic_monthly,
    load_sub_period_analysis,
    load_quintile_timeseries,
)
from utils.chart_helpers import (
    apply_layout,
    GOLD, BLUE, TEAL, RED, GREEN, WHITE, SUBTEXT, SLATE, ORANGE, PURPLE,
    CHART_CFG, CARD_BG, FACTOR_COLORS, QUINTILE_COLORS,
    cumulative,
)

EXCHANGES = ["GCC", "ADX", "DFM", "Tadawul", "QSE", "Kuwait"]
FACTORS_ORDERED = ["MOMENTUM", "QUALITY", "VALUE", "LOW-VOL"]
PERIODS_ORDERED = [
    "Full sample", "Pre-COVID 15-19", "COVID 2020",
    "Post-COVID 21-23", "Recent 24-26",
]


# ── stat helpers ──────────────────────────────────────────────────────────────

def _sortino(s: pd.Series) -> float:
    if len(s) < 5:
        return np.nan
    ann_ret  = s.mean() * 12
    downside = s[s < 0].std() * np.sqrt(12)
    return ann_ret / downside if downside > 0 else np.nan


def _sig(t) -> str:
    if pd.isna(t):
        return ""
    a = abs(float(t))
    if a >= 2.576: return "***"
    if a >= 1.960: return "**"
    if a >= 1.645: return "*"
    return ""


def _fmt(v, fmt=".2f"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{float(v):{fmt}}"


# ── chart builders ────────────────────────────────────────────────────────────

def _empty_fig(title: str, height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="No data for selected exchange",
        x=0.5, y=0.5, xref="paper", yref="paper",
        font=dict(color=SUBTEXT, size=13), showarrow=False,
    )
    apply_layout(fig, title, height=height)
    return fig


def _quintile_waterfall(exchange: str) -> go.Figure:
    """Grouped bar Q1–Q5 + L/S lines per factor for selected exchange."""
    bs = load_backtest_summary()
    exch_data = bs[bs["exchange"] == exchange]

    factor_rows: dict = {}
    for f in FACTORS_ORDERED:
        row = exch_data[exch_data["factor"] == f]
        if len(row):
            factor_rows[f] = row.iloc[0]

    if not factor_rows:
        return _empty_fig(f"Quintile Return Monotonicity - {exchange}", 400)

    x_labels = [f.title() for f in FACTORS_ORDERED if f in factor_rows]
    q_cols   = ["Q1_ann_pct", "Q2_ann_pct", "Q3_ann_pct", "Q4_ann_pct", "Q5_ann_pct"]
    q_names  = ["Q1 (Best)", "Q2", "Q3", "Q4", "Q5 (Worst)"]

    fig = go.Figure()
    for qcol, qname, qcolor in zip(q_cols, q_names, QUINTILE_COLORS):
        y = [
            float(factor_rows[f].get(qcol, np.nan))
            for f in FACTORS_ORDERED if f in factor_rows
        ]
        fig.add_bar(
            x=x_labels, y=y, name=qname,
            marker_color=qcolor, opacity=0.88,
            hovertemplate=f"<b>{qname}</b><br>%{{x}}: %{{y:+.1f}}%<extra></extra>",
        )

    ls_y = [
        float(factor_rows[f].get("LS_ann_pct", np.nan))
        for f in FACTORS_ORDERED if f in factor_rows
    ]
    ls_net_y = [
        float(factor_rows[f].get("LS_net_ann_pct", np.nan))
        for f in FACTORS_ORDERED if f in factor_rows
    ]

    fig.add_scatter(
        x=x_labels, y=ls_y, name="L/S Gross",
        mode="markers+lines",
        line=dict(color=GOLD, width=2.5),
        marker=dict(size=9, color=GOLD, symbol="diamond"),
        hovertemplate="<b>L/S Gross</b><br>%{x}: %{y:+.1f}%<extra></extra>",
    )
    fig.add_scatter(
        x=x_labels, y=ls_net_y, name="L/S Net (50bps TC)",
        mode="markers+lines",
        line=dict(color=TEAL, width=2, dash="dot"),
        marker=dict(size=7, color=TEAL),
        hovertemplate="<b>L/S Net (50bps TC)</b><br>%{x}: %{y:+.1f}%<extra></extra>",
    )

    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.18)")
    fig.update_layout(barmode="group")
    apply_layout(fig, f"Quintile Return Monotonicity - {exchange}", height=440)
    fig.update_yaxes(
        title_text="Annualised Return (%)", ticksuffix="%",
        title_font=dict(color=SUBTEXT, size=11),
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.18,
            xanchor="left", x=0,
            font=dict(size=11),
            itemwidth=80,
            tracegroupgap=4,
        ),
        margin=dict(b=90),
    )
    return fig


def _risk_metrics_table(exchange: str) -> dbc.Table:
    bs = load_backtest_summary()
    exch_data = bs[bs["exchange"] == exchange]

    header = html.Thead(html.Tr([
        html.Th(col, style={"fontSize": "0.72rem"})
        for col in ["Factor", "L/S Ann%", "Sharpe", "Sortino",
                    "MaxDD%", "Net Shp", "IC", "IC t-stat", "Hit%"]
    ], className="table-dark"))

    rows = []
    for factor in FACTORS_ORDERED:
        row = exch_data[exch_data["factor"] == factor]
        if not len(row):
            continue
        r      = row.iloc[0]
        ls     = float(r.get("LS_ann_pct",    np.nan))
        sharpe = float(r.get("LS_sharpe",     np.nan))
        maxdd  = float(r.get("LS_maxdd_pct",  np.nan))
        net_sh = float(r.get("LS_net_sharpe", np.nan))
        ic     = float(r.get("IC_mean",       np.nan))
        ic_t   = float(r.get("IC_t",          np.nan))
        hit    = float(r.get("LS_hit_pct",    np.nan))

        qt     = load_quintile_timeseries(exchange, factor)
        sort_v = (
            _sortino(qt["LS"].dropna())
            if qt is not None and "LS" in qt.columns else np.nan
        )

        sharpe_c = GREEN if sharpe > 0.5 else (GOLD if sharpe > 0 else RED)
        ls_c     = GREEN if ls > 5 else (GOLD if ls > 0 else RED)
        ic_t_c   = GREEN if abs(ic_t) >= 1.96 else (GOLD if abs(ic_t) >= 1.645 else SUBTEXT)

        cells = [
            html.Td(factor.title(),
                    style={"fontWeight": 600,
                           "color": FACTOR_COLORS.get(factor, WHITE),
                           "fontSize": "0.78rem"}),
            html.Td(f"{ls:+.1f}%" if not np.isnan(ls) else "N/A",
                    style={"color": ls_c, "fontSize": "0.78rem"}),
            html.Td(_fmt(sharpe),
                    style={"color": sharpe_c, "fontWeight": 700, "fontSize": "0.78rem"}),
            html.Td(_fmt(sort_v),
                    style={"fontSize": "0.78rem"}),
            html.Td(f"{maxdd:.1f}%" if not np.isnan(maxdd) else "N/A",
                    style={"color": RED, "fontSize": "0.78rem"}),
            html.Td(_fmt(net_sh),
                    style={"fontSize": "0.78rem"}),
            html.Td(f"{ic:+.3f}" if not np.isnan(ic) else "N/A",
                    style={"fontSize": "0.78rem"}),
            html.Td(
                f"{ic_t:+.2f}{_sig(ic_t)}" if not np.isnan(ic_t) else "N/A",
                style={"color": ic_t_c, "fontWeight": 600, "fontSize": "0.78rem"},
            ),
            html.Td(f"{hit:.0f}%" if not np.isnan(hit) else "N/A",
                    style={"fontSize": "0.78rem"}),
        ]
        rows.append(html.Tr(cells))

    return dbc.Table(
        [header, html.Tbody(rows)],
        bordered=False, hover=True, size="sm",
        className="table-dark table-hover align-middle mb-0",
    )


def _cumulative_ls_chart(exchange: str) -> go.Figure:
    fig = go.Figure()
    has_any = False

    for factor in FACTORS_ORDERED:
        qt = load_quintile_timeseries(exchange, factor)
        if qt is None or "LS" not in qt.columns:
            continue
        ls = qt["LS"].dropna()
        if len(ls) < 5:
            continue
        cum   = cumulative(ls) * 100
        color = FACTOR_COLORS.get(factor, SLATE)

        fig.add_scatter(
            x=cum.index, y=cum.round(2),
            name=factor.title(), mode="lines",
            line=dict(color=color, width=2.5),
            hovertemplate=(
                f"<b>{factor.title()}</b><br>"
                "%{x|%b %Y}: %{y:+.1f}%<extra></extra>"
            ),
        )
        if "LS_net" in qt.columns:
            cum_net = cumulative(qt["LS_net"].dropna()) * 100
            fig.add_scatter(
                x=cum_net.index, y=cum_net.round(2),
                name=f"{factor.title()} Net",
                mode="lines",
                line=dict(color=color, width=1.2, dash="dot"),
                opacity=0.50, showlegend=False,
                hovertemplate=(
                    f"<b>{factor.title()} Net (50bps)</b><br>"
                    "%{x|%b %Y}: %{y:+.1f}%<extra></extra>"
                ),
            )
        has_any = True

    if not has_any:
        return _empty_fig(f"Cumulative L/S Returns - {exchange}", 360)

    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.18)")
    apply_layout(
        fig,
        f"Cumulative L/S Returns - {exchange}  (solid = gross · dot = net 50bps TC)",
        height=360,
    )
    fig.update_yaxes(
        title_text="Cumulative Return (%)", ticksuffix="%",
        title_font=dict(color=SUBTEXT, size=11),
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.18,
            xanchor="left", x=0,
            font=dict(size=11),
        ),
        margin=dict(b=70),
    )
    return fig


def _fm_significance_chart() -> go.Figure:
    """Univariate + Joint FM lambda bars with NW-4 t-stat significance."""
    fm = load_fm_summary()

    factors, lambdas_uni, lambdas_jt, t_uni, t_jt = [], [], [], [], []
    for f in FACTORS_ORDERED:
        row = fm[fm["factor"] == f]
        if not len(row):
            continue
        r = row.iloc[0]
        factors.append(f.title())
        lam_u = float(r.get("lambda_uni_bps",   np.nan))
        lam_j = float(r.get("lambda_joint_bps", np.nan))
        tu    = float(r.get("t_uni_nw3",         np.nan))
        tj    = float(r.get("t_joint_nw3",       np.nan))
        lambdas_uni.append(lam_u)
        lambdas_jt.append(lam_j)
        t_uni.append(tu)
        t_jt.append(tj)

    uni_colors = [
        GREEN if (not pd.isna(t) and abs(t) >= 1.96)
        else (GOLD if (not pd.isna(t) and abs(t) >= 1.645)
              else SLATE)
        for t in t_uni
    ]

    text_uni = [
        f"{lam:+.0f} {_sig(t)}" if not pd.isna(lam) else "N/A"
        for lam, t in zip(lambdas_uni, t_uni)
    ]

    fig = go.Figure()
    fig.add_bar(
        x=factors, y=lambdas_uni, name="Univariate FM",
        marker_color=uni_colors, opacity=0.92,
        text=text_uni, textposition="outside",
        textfont=dict(color=WHITE, size=11),
        customdata=[[t] for t in t_uni],
        hovertemplate=(
            "<b>%{x} (Univariate)</b><br>"
            "λ: %{y:+.0f} bps/mo<br>"
            "NW t-stat: %{customdata[0]:+.2f}<extra></extra>"
        ),
    )
    fig.add_bar(
        x=factors, y=lambdas_jt, name="Joint FM",
        marker_color=BLUE, opacity=0.60,
        customdata=[[t] for t in t_jt],
        hovertemplate=(
            "<b>%{x} (Joint)</b><br>"
            "λ: %{y:+.0f} bps/mo<br>"
            "NW t-stat: %{customdata[0]:+.2f}<extra></extra>"
        ),
    )

    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.18)")
    fig.update_layout(barmode="group")
    apply_layout(
        fig,
        "Fama-MacBeth λ (bps/month) - Univariate & Joint OLS  (Newey-West 4-lag t-stats)",
        height=380,
    )
    fig.update_yaxes(
        title_text="λ (bps/month)",
        title_font=dict(color=SUBTEXT, size=11),
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.18,
            xanchor="left", x=0,
            font=dict(size=11),
        ),
        margin=dict(b=70),
        annotations=[dict(
            text="Stars: ***p<1%  **p<5%  *p<10%  (green = significant at 5%)",
            x=0.99, y=0.03, xref="paper", yref="paper",
            showarrow=False, font=dict(color=SUBTEXT, size=10), xanchor="right",
        )],
    )
    return fig


def _sub_period_table() -> dbc.Table:
    df = load_sub_period_analysis()

    header = html.Thead(html.Tr(
        [html.Th("Period", style={"fontSize": "0.72rem", "minWidth": "105px"})] +
        [
            html.Th(f.title(), style={
                "fontSize": "0.72rem",
                "color": FACTOR_COLORS.get(f, WHITE),
            })
            for f in FACTORS_ORDERED
        ],
        className="table-dark",
    ))

    rows = []
    for period in PERIODS_ORDERED:
        block = df[df["period"] == period]
        if block.empty:
            continue
        period_short = (
            period.replace(" 15-19", "")
                  .replace(" 21-23", "")
                  .replace(" 24-26", "")
        )
        cells = [html.Td(
            period_short,
            style={"fontWeight": 600, "color": SUBTEXT, "fontSize": "0.72rem"},
        )]
        for factor in FACTORS_ORDERED:
            fb = block[block["factor"] == factor]
            if fb.empty:
                cells.append(html.Td("-", style={"color": SUBTEXT, "fontSize": "0.72rem"}))
                continue
            r   = fb.iloc[0]
            t   = float(r.get("t_stat",     np.nan))
            lam = float(r.get("lambda_bps", np.nan))
            sh  = float(r.get("sharpe",     np.nan))
            sig = _sig(t)
            color = (GREEN if abs(t) >= 1.96
                     else (GOLD if abs(t) >= 1.645 else SUBTEXT))
            hover = (f"Sharpe: {sh:.2f}  t: {t:+.2f}"
                     if not (np.isnan(sh) or np.isnan(t)) else "")
            cells.append(html.Td(
                f"{lam:+.0f} {sig}" if not np.isnan(lam) else "N/A",
                title=hover,
                style={"color": color,
                       "fontWeight": 600 if sig else 400,
                       "fontSize": "0.72rem"},
            ))
        rows.append(html.Tr(cells))

    return dbc.Table(
        [header, html.Tbody(rows)],
        bordered=False, hover=True, size="sm",
        className="table-dark table-hover align-middle mb-0",
    )


def _ic_timeseries_chart() -> go.Figure:
    ic_df = load_ic_monthly()
    fig = go.Figure()

    for factor in FACTORS_ORDERED:
        col = next((c for c in ic_df.columns if c.upper() == factor), None)
        if col is None:
            continue
        ic = ic_df[col].dropna()
        if len(ic) < 5:
            continue
        color = FACTOR_COLORS.get(factor, SLATE)

        fig.add_scatter(
            x=ic.index, y=ic.round(3),
            name=factor.title(), mode="lines",
            line=dict(color=color, width=0.8),
            opacity=0.30, showlegend=False, hoverinfo="skip",
        )
        roll = ic.rolling(12, min_periods=6).mean()
        fig.add_scatter(
            x=roll.index, y=roll.round(3),
            name=f"{factor.title()} (12m avg)", mode="lines",
            line=dict(color=color, width=2.5),
            hovertemplate=(
                f"<b>{factor.title()} 12m avg IC</b><br>"
                "%{x|%b %Y}: %{y:.3f}<extra></extra>"
            ),
        )

    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.22)")
    apply_layout(
        fig,
        "Monthly Spearman IC - GCC Pooled  (thin = monthly · bold = 12m rolling avg)",
        height=320,
    )
    fig.update_yaxes(
        title_text="Spearman IC",
        title_font=dict(color=SUBTEXT, size=11),
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.18,
            xanchor="left", x=0,
            font=dict(size=11),
        ),
        margin=dict(b=70),
    )
    return fig


def _tc_impact_chart(exchange: str) -> go.Figure:
    bs = load_backtest_summary()
    exch_data = bs[bs["exchange"] == exchange]

    x_labels, gross_vals, net_vals = [], [], []
    for factor in FACTORS_ORDERED:
        row = exch_data[exch_data["factor"] == factor]
        if not len(row):
            continue
        r = row.iloc[0]
        x_labels.append(factor.title())
        gross_vals.append(float(r.get("LS_sharpe",     np.nan)))
        net_vals.append(float(r.get("LS_net_sharpe",   np.nan)))

    if not x_labels:
        return _empty_fig(f"TC Impact - {exchange}", 300)

    fig = go.Figure()
    fig.add_bar(
        x=x_labels, y=gross_vals, name="Gross",
        marker_color=BLUE, opacity=0.88,
        hovertemplate="<b>Gross Sharpe</b><br>%{x}: %{y:.2f}<extra></extra>",
    )
    fig.add_bar(
        x=x_labels, y=net_vals, name="Net (50bps)",
        marker_color=TEAL, opacity=0.88,
        hovertemplate="<b>Net Sharpe (50bps TC)</b><br>%{x}: %{y:.2f}<extra></extra>",
    )
    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.18)")
    fig.update_layout(barmode="group")
    apply_layout(fig, f"TC Impact: Gross vs Net Sharpe - {exchange}", height=300)
    fig.update_yaxes(
        title_text="Sharpe Ratio",
        title_font=dict(color=SUBTEXT, size=11),
    )
    return fig


def _sharpe_heatmap() -> go.Figure:
    bs = load_backtest_summary()

    z, text_z = [], []
    for exch in EXCHANGES:
        row_z, row_t = [], []
        exch_data = bs[bs["exchange"] == exch]
        for factor in FACTORS_ORDERED:
            r = exch_data[exch_data["factor"] == factor]
            if len(r):
                sh = float(r.iloc[0].get("LS_sharpe", np.nan))
                row_z.append(sh if not np.isnan(sh) else None)
                row_t.append(f"{sh:.2f}" if not np.isnan(sh) else "")
            else:
                row_z.append(None)
                row_t.append("")
        z.append(row_z)
        text_z.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f.title() for f in FACTORS_ORDERED],
        y=EXCHANGES,
        text=text_z,
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        colorscale=[
            [0.0, "#C62828"], [0.35, "#EF5350"],
            [0.5,  "#455A64"],
            [0.65, "#43A047"], [1.0,  "#1B5E20"],
        ],
        zmid=0, zmin=-1.2, zmax=1.2,
        showscale=True,
        colorbar=dict(
            title=dict(text="Sharpe", font=dict(color=SUBTEXT, size=11)),
            tickfont=dict(color=SUBTEXT, size=10),
            len=0.85,
        ),
        hovertemplate="<b>%{y} / %{x}</b><br>L/S Sharpe: %{z:.2f}<extra></extra>",
    ))
    apply_layout(fig, "L/S Sharpe Heatmap - All Exchanges × All Factors", height=340)
    fig.update_layout(margin=dict(l=80, r=80, t=44, b=30))
    return fig


def _kpi_cards(exchange: str) -> list:
    bs   = load_backtest_summary()
    fm   = load_fm_summary()
    exch = bs[bs["exchange"] == exchange]

    best_factor, best_sharpe, max_abs_ic, sig_count = "N/A", np.nan, np.nan, 0

    if len(exch):
        valid_sh = exch["LS_sharpe"].dropna()
        if len(valid_sh):
            idx         = valid_sh.idxmax()
            best_factor = str(exch.loc[idx, "factor"]).title()
            best_sharpe = float(valid_sh[idx])
        valid_ic = exch["IC_mean"].dropna()
        if len(valid_ic):
            max_abs_ic = float(valid_ic.abs().max())

    if len(fm) and "t_uni_nw3" in fm.columns:
        sig_count = int((fm["t_uni_nw3"].abs() >= 1.96).sum())

    kpi = [
        ("Best Factor",      best_factor,                                        GOLD),
        ("Peak Sharpe",      _fmt(best_sharpe),                                  GREEN if best_sharpe > 0.5 else GOLD),
        ("Max |IC|",         f"{max_abs_ic:.3f}" if not np.isnan(max_abs_ic) else "N/A", BLUE),
        ("FM Sig. |t|>2",    f"{sig_count}/4",                                   GREEN if sig_count >= 2 else GOLD),
    ]

    return [
        html.Div([
            html.Div(label, style={
                "color": SUBTEXT, "fontSize": "0.67rem",
                "textTransform": "uppercase", "letterSpacing": "0.06em",
            }),
            html.Div(value, style={
                "color": color, "fontSize": "1.3rem",
                "fontWeight": 700, "lineHeight": 1.2, "marginTop": "2px",
            }),
        ], style={
            "background": CARD_BG, "borderRadius": "6px",
            "padding": "8px 14px", "flex": "1", "minWidth": "100px",
        })
        for label, value, color in kpi
    ]


# ── shared card wrapper ───────────────────────────────────────────────────────

def _card(title: str, content) -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(
            title,
            style={
                "background": "rgba(255,255,255,0.04)",
                "fontWeight": 600, "color": WHITE,
                "fontSize": "0.83rem", "letterSpacing": "0.03em",
            },
        ),
        dbc.CardBody(content, className="p-0"),
    ], className="mb-3 border-0",
       style={"background": CARD_BG, "borderRadius": "8px", "overflow": "hidden"})


# ── layout ────────────────────────────────────────────────────────────────────

def layout() -> html.Div:
    return html.Div([

        # Header
        html.Div([
            html.H4("Layer 3 - Backtesting Engine", className="mb-1",
                    style={"color": GOLD, "fontWeight": 700}),
            html.P(
                "Monthly rebalancing · 50bps round-trip TC · Q1–Q5 quintile attribution · "
                "Sharpe / Sortino / Max DD · Fama-MacBeth OLS cross-sectional t-stats · "
                "Sub-period stability · IC significance",
                className="mb-0",
                style={"color": SUBTEXT, "fontSize": "0.86rem"},
            ),
        ], className="mb-3"),

        # Exchange selector + KPI strip
        dbc.Row([
            dbc.Col([
                html.Label("Exchange", style={
                    "color": SUBTEXT, "fontSize": "0.73rem",
                    "textTransform": "uppercase", "letterSpacing": "0.06em",
                    "marginBottom": "4px", "display": "block",
                }),
                dcc.Dropdown(
                    id="bt-exchange-dd",
                    options=[{"label": ex, "value": ex} for ex in EXCHANGES],
                    value="GCC",
                    clearable=False,
                    style={"fontSize": "0.85rem"},
                ),
            ], md=2, className="pe-3"),
            dbc.Col(
                html.Div(
                    id="bt-kpi-row",
                    children=_kpi_cards("GCC"),
                    style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
                ),
                md=10,
            ),
        ], className="mb-3 align-items-end"),

        # Quintile waterfall – full width
        dbc.Row([
            dbc.Col(
                _card(
                    "Quintile Return Monotonicity  (Q1 = best score / long · "
                    "Q5 = worst / short · annualised %)",
                    dcc.Graph(
                        id="bt-quintile-fig",
                        figure=_quintile_waterfall("GCC"),
                        config=CHART_CFG,
                    ),
                ),
                md=12,
            ),
        ]),

        # Risk table + cumulative chart
        dbc.Row([
            dbc.Col(
                _card(
                    "Risk-Adjusted Performance  (L/S long-short)",
                    html.Div(
                        id="bt-metrics-table",
                        children=html.Div(_risk_metrics_table("GCC"), className="p-2"),
                    ),
                ),
                md=5,
            ),
            dbc.Col(
                _card(
                    "Cumulative L/S Returns  (solid = gross · dot = net 50bps TC)",
                    dcc.Graph(
                        id="bt-cum-fig",
                        figure=_cumulative_ls_chart("GCC"),
                        config=CHART_CFG,
                    ),
                ),
                md=7,
            ),
        ]),

        # FM significance (static) + sub-period stability
        dbc.Row([
            dbc.Col(
                _card(
                    "Fama-MacBeth Factor Premia  (OLS cross-sectional)",
                    dcc.Graph(figure=_fm_significance_chart(), config=CHART_CFG),
                ),
                md=6,
            ),
            dbc.Col(
                _card(
                    "Sub-Period Stability  (λ bps/month · colour = significance)",
                    html.Div(_sub_period_table(), className="p-2"),
                ),
                md=6,
            ),
        ]),

        # IC time series (static) + TC impact (dynamic)
        dbc.Row([
            dbc.Col(
                _card(
                    "Spearman IC Time Series - GCC Pooled",
                    dcc.Graph(figure=_ic_timeseries_chart(), config=CHART_CFG),
                ),
                md=8,
            ),
            dbc.Col(
                _card(
                    "Transaction Cost Impact  (50bps round-trip)",
                    dcc.Graph(
                        id="bt-tc-fig",
                        figure=_tc_impact_chart("GCC"),
                        config=CHART_CFG,
                    ),
                ),
                md=4,
            ),
        ]),

        # Sharpe heatmap – full width, static
        dbc.Row([
            dbc.Col(
                _card(
                    "L/S Sharpe Heatmap - All Exchanges × All Factors",
                    dcc.Graph(figure=_sharpe_heatmap(), config=CHART_CFG),
                ),
                md=12,
            ),
        ]),

    ], className="p-3")


# ── callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):
    @app.callback(
        [
            Output("bt-kpi-row",       "children"),
            Output("bt-quintile-fig",  "figure"),
            Output("bt-metrics-table", "children"),
            Output("bt-cum-fig",       "figure"),
            Output("bt-tc-fig",        "figure"),
        ],
        Input("bt-exchange-dd", "value"),
    )
    def update_exchange(exchange):
        ex = exchange or "GCC"
        return (
            _kpi_cards(ex),
            _quintile_waterfall(ex),
            html.Div(_risk_metrics_table(ex), className="p-2"),
            _cumulative_ls_chart(ex),
            _tc_impact_chart(ex),
        )
