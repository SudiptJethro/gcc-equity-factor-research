"""Page 1 - Layer 1: Factor Analysis."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc
import dash_bootstrap_components as dbc

from utils.data_loader import (
    load_fm_summary,
    load_quintile_spreads,
    load_ic_monthly,
    load_backtest_summary,
    compute_fm_interactions,
    compute_intra_exchange_corr,
    compute_cross_exchange_corr,
    load_quintile_stock_membership,
    load_ticker_names,
    FACTORS,
    EXCHANGES,
)
from utils.chart_helpers import (
    apply_layout, subplot_layout,
    FACTOR_COLORS, EXCHANGE_COLORS, QUINTILE_COLORS,
    GOLD, BLUE, TEAL, RED, GREEN, WHITE, SUBTEXT, SLATE, ORANGE, PURPLE,
    CHART_CFG, CARD_BG, BORDER,
    cumulative, rolling_sharpe,
)

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

CORR_COLORSCALE = [
    [0.0, "#4A9FD4"],   # blue  (negative)
    [0.5, "#1C2840"],   # navy  (zero)
    [1.0, "#C9A84C"],   # gold  (positive)
]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _factor_summary_table(fm: pd.DataFrame) -> dbc.Table:
    factor_order = ["MOMENTUM", "QUALITY", "VALUE", "LOW-VOL"]
    fm = fm.set_index("factor").reindex(factor_order).reset_index()

    header = html.Thead(html.Tr([
        html.Th("Factor", style={"width": "18%"}),
        html.Th("L/S Ann. Return"),
        html.Th("Ann. Vol"),
        html.Th("Sharpe"),
        html.Th("IC Mean"),
        html.Th("ICIR (ann.)"),
        html.Th("t-stat (NW)"),
        html.Th("Sig."),
    ], className="table-dark"))

    rows = []
    for _, r in fm.iterrows():
        fac = r["factor"]
        col = FACTOR_COLORS.get(fac, WHITE)
        ret = r.get("port_ann_ret_pct", np.nan)
        sharpe = r.get("port_sharpe", np.nan)
        ic = r.get("IC_mean", np.nan)
        t = r.get("t_uni_nw3", np.nan)
        sig = r.get("sig_stars", r.get("sig_uni", ""))

        def ret_color(v):
            if pd.isna(v):
                return WHITE
            return GREEN if v > 8 else (GOLD if v > 0 else RED)

        def sharpe_color(v):
            if pd.isna(v):
                return WHITE
            return GREEN if v > 0.8 else (GOLD if v > 0.4 else RED)

        rows.append(html.Tr([
            html.Td(fac.title(), style={"fontWeight": 700, "color": col}),
            html.Td(f"{ret:+.1f}%" if not pd.isna(ret) else "N/A",
                    style={"color": ret_color(ret), "fontWeight": 600}),
            html.Td(f"{r.get('port_ann_vol_pct', np.nan):.1f}%"
                    if not pd.isna(r.get("port_ann_vol_pct", np.nan)) else "N/A"),
            html.Td(f"{sharpe:.2f}" if not pd.isna(sharpe) else "N/A",
                    style={"color": sharpe_color(sharpe), "fontWeight": 600}),
            html.Td(f"{ic:+.4f}" if not pd.isna(ic) else "N/A"),
            html.Td(f"{r.get('ICIR_ann', np.nan):+.2f}"
                    if not pd.isna(r.get("ICIR_ann", np.nan)) else "N/A"),
            html.Td(f"{t:+.2f}" if not pd.isna(t) else "N/A"),
            html.Td(sig if sig else "N/A", style={"color": GOLD, "fontWeight": 700}),
        ]))

    return dbc.Table(
        [header, html.Tbody(rows)],
        bordered=False, hover=True, size="sm",
        className="table-dark table-hover align-middle mb-0",
        style={"fontSize": "0.82rem"},
    )


def _fm_tstat_table(fm: pd.DataFrame) -> dbc.Table:
    factor_order = ["MOMENTUM", "QUALITY", "VALUE", "LOW-VOL"]
    fm = fm.set_index("factor").reindex(factor_order).reset_index()

    header = html.Thead(html.Tr([
        html.Th("Factor"),
        html.Th("λ Joint (bps/mo)"),
        html.Th("t Joint"),
        html.Th("λ Uni (bps/mo)"),
        html.Th("t Uni"),
        html.Th("Sig."),
    ], className="table-dark"))

    rows = []
    for _, r in fm.iterrows():
        fac = r["factor"]
        col = FACTOR_COLORS.get(fac, WHITE)
        tj = r.get("t_joint_nw3", np.nan)
        tu = r.get("t_uni_nw3", np.nan)
        sig = r.get("sig_stars", r.get("sig_uni", ""))
        rows.append(html.Tr([
            html.Td(fac.title(), style={"fontWeight": 700, "color": col}),
            html.Td(f"{r.get('lambda_joint_bps', np.nan):+.1f}"
                    if not pd.isna(r.get("lambda_joint_bps", np.nan)) else "N/A"),
            html.Td(f"{tj:+.2f}" if not pd.isna(tj) else "N/A",
                    style={"color": GOLD if abs(tj) >= 1.645 else WHITE} if not pd.isna(tj) else {}),
            html.Td(f"{r.get('lambda_uni_bps', np.nan):+.1f}"
                    if not pd.isna(r.get("lambda_uni_bps", np.nan)) else "N/A"),
            html.Td(f"{tu:+.2f}" if not pd.isna(tu) else "N/A",
                    style={"color": GOLD if abs(tu) >= 1.645 else WHITE} if not pd.isna(tu) else {}),
            html.Td(sig if sig else "N/A", style={"color": GOLD, "fontWeight": 700}),
        ]))

    return dbc.Table(
        [header, html.Tbody(rows)],
        bordered=False, hover=True, size="sm",
        className="table-dark table-hover align-middle mb-0",
        style={"fontSize": "0.82rem"},
    )


def _quintile_chart(qs: pd.DataFrame) -> go.Figure:
    factor_order = ["MOMENTUM", "QUALITY", "VALUE", "LOW-VOL"]
    q_cols = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    fig = go.Figure()

    for qi, qcol in enumerate(q_cols):
        y_vals = []
        x_vals = []
        for fac in factor_order:
            row = qs[qs["factor"] == fac]
            if len(row):
                y_vals.append(row[qcol].values[0])
                x_vals.append(fac.title())
            else:
                y_vals.append(0)
                x_vals.append(fac.title())
        fig.add_bar(
            name=qcol,
            x=x_vals,
            y=y_vals,
            marker_color=QUINTILE_COLORS[qi],
            text=[f"{v:+.1f}%" for v in y_vals],
            textposition="outside",
            textfont=dict(size=10, color=WHITE),
        )

    fig.update_layout(barmode="group")
    apply_layout(fig, "Quintile Return Spreads - Annualised (%)", height=380)
    fig.update_yaxes(title_text="Ann. Return (%)", title_font=dict(color=SUBTEXT, size=11))
    fig.update_xaxes(title_text="")
    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.2)")
    return fig


def _exchange_breakdown_chart(bs: pd.DataFrame) -> go.Figure:
    factor_order = ["MOMENTUM", "QUALITY", "VALUE", "LOW-VOL"]
    exchanges_plot = ["GCC", "ADX", "DFM", "Tadawul", "QSE"]
    fig = go.Figure()

    for fac in factor_order:
        fdf = bs[bs["factor"] == fac]
        y_vals, x_vals, text_vals = [], [], []
        for exch in exchanges_plot:
            row = fdf[fdf["exchange"] == exch]
            if len(row):
                v = row["LS_sharpe"].values[0]
                y_vals.append(v)
                text_vals.append(f"{v:.2f}")
            else:
                y_vals.append(None)
                text_vals.append("")
            x_vals.append(exch)
        fig.add_bar(
            name=fac.title(),
            x=x_vals,
            y=y_vals,
            marker_color=FACTOR_COLORS.get(fac, WHITE),
            text=text_vals,
            textposition="outside",
            textfont=dict(size=9, color=WHITE),
        )

    fig.update_layout(barmode="group")
    apply_layout(fig, "L/S Sharpe Ratio by Exchange × Factor", height=380)
    fig.update_yaxes(title_text="L/S Sharpe Ratio", title_font=dict(color=SUBTEXT, size=11))
    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.2)")
    return fig


def _rolling_ic_chart(ic_df: pd.DataFrame) -> go.Figure:
    col_map = {"VALUE": "Value", "MOMENTUM": "Momentum", "QUALITY": "Quality", "LOW-VOL": "Low-Vol"}
    fig = go.Figure()
    for col, label in col_map.items():
        if col not in ic_df.columns:
            continue
        roll = ic_df[col].rolling(12, min_periods=6).mean().dropna()
        fig.add_scatter(
            x=roll.index,
            y=roll.round(4),
            name=f"{label} (12m avg)",
            mode="lines",
            line=dict(color=FACTOR_COLORS.get(col, WHITE), width=2),
        )
    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.25)")
    apply_layout(fig, "Rolling 12-Month Average IC (Spearman)", height=340)
    fig.update_yaxes(title_text="IC", title_font=dict(color=SUBTEXT, size=11))
    fig.update_layout(legend=dict(orientation="h", y=-0.25, x=0))
    return fig


def _interaction_table(idf: pd.DataFrame) -> dbc.Table:
    header = html.Thead(html.Tr([
        html.Th("Factor"),
        html.Th("Exchange"),
        html.Th("Differential vs GCC (ann. %)"),
        html.Th("t-stat"),
        html.Th("Sig."),
    ], className="table-dark"))

    rows = []
    for _, r in idf.iterrows():
        fac = r["Factor"]
        sig = r["Sig."]
        coef = r["Coef (ann. %)"]
        tstat = r["t-stat"]
        rows.append(html.Tr([
            html.Td(fac, style={"color": FACTOR_COLORS.get(fac.upper(), WHITE), "fontWeight": 700}),
            html.Td(r["Exchange"]),
            html.Td(f"{coef:+.2f}%",
                    style={"color": GREEN if coef > 0 else RED, "fontWeight": 600}),
            html.Td(f"{tstat:+.2f}",
                    style={"color": GOLD if abs(tstat) >= 1.645 else SUBTEXT}),
            html.Td(sig if sig else "N/A", style={"color": GOLD, "fontWeight": 700}),
        ]))

    note = html.Tr([
        html.Td("Coefficient = exchange LS annual return minus GCC pooled LS return. "
                "t-stat uses LS vol and T = 124 months. |t|>1.96 = ** (5%).",
                colSpan=5,
                style={"color": SUBTEXT, "fontSize": "0.75rem", "fontStyle": "italic"}),
    ])
    rows.append(note)

    return dbc.Table(
        [header, html.Tbody(rows)],
        bordered=False, hover=True, size="sm",
        className="table-dark table-hover align-middle mb-0",
        style={"fontSize": "0.82rem"},
    )


# ---------------------------------------------------------------------------
# GAP 2 / Oil Macro Control helpers
# ---------------------------------------------------------------------------

def _oil_table(oil_df: pd.DataFrame, status: str) -> html.Div:
    def _td(v, fmt):
        return "N/A" if pd.isna(v) else format(float(v), fmt)

    header = html.Thead(html.Tr([
        html.Th("Factor"),
        html.Th("Oil Beta"),
        html.Th("t-stat (NW-4)"),
        html.Th("R²"),
    ], className="table-dark"))

    rows = []
    for _, r in oil_df.iterrows():
        beta = r["Oil Beta"]
        t = r["t-stat"]
        r2 = r["R²"]
        sig = (not pd.isna(t)) and abs(float(t)) >= 1.645
        factor_key = str(r["Factor"]).upper().replace("LOW-VOL", "LOW-VOL")
        rows.append(html.Tr([
            html.Td(r["Factor"], style={
                "fontWeight": 700,
                "color": FACTOR_COLORS.get(r["Factor"], WHITE),
            }),
            html.Td(_td(beta, "+.3f")),
            html.Td(_td(t, "+.2f"), style={
                "color": GOLD if sig else WHITE,
                "fontWeight": 600 if sig else 400,
            }),
            html.Td(_td(r2, ".3f")),
        ]))

    rows.append(html.Tr([html.Td(
        f"Source: {status}. NW-4 Newey-West SE. |t|>1.645 highlighted.",
        colSpan=4,
        style={"color": SUBTEXT, "fontSize": "0.75rem", "fontStyle": "italic"},
    )]))

    return html.Div(dbc.Table(
        [header, html.Tbody(rows)],
        bordered=False, hover=True, size="sm",
        className="table-dark table-hover align-middle mb-0",
        style={"fontSize": "0.82rem"},
    ))


# ---------------------------------------------------------------------------
# GAP 1 / Factor Crowding helpers
# ---------------------------------------------------------------------------

def _corr_heatmap_fig(corr_df: pd.DataFrame, exchange: str) -> go.Figure:
    factors = corr_df.columns.tolist()
    z = corr_df.values
    text = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z.tolist(),
        x=factors,
        y=factors,
        zmin=-1,
        zmax=1,
        colorscale=CORR_COLORSCALE,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=12, color=WHITE),
        showscale=True,
        colorbar=dict(thickness=12, len=0.8, tickfont=dict(color=SUBTEXT, size=10)),
        hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
    ))
    apply_layout(fig, f"{exchange} - Factor-Factor Correlation", height=320)
    fig.update_layout(margin=dict(l=100, r=60, t=50, b=80))
    return fig


def _cross_exchange_table(df: pd.DataFrame) -> dbc.Table:
    if df is None or len(df) == 0:
        return html.Div("No cross-exchange data available.",
                        style={"color": SUBTEXT, "padding": "20px"})
    factor_cols = [c for c in df.columns if c != "Exchange Pair"]

    header = html.Thead(html.Tr(
        [html.Th("Exchange Pair")] + [html.Th(c) for c in factor_cols],
        className="table-dark",
    ))
    rows = []
    for _, r in df.iterrows():
        cells = [html.Td(r["Exchange Pair"], style={"fontWeight": 600, "color": WHITE})]
        for col in factor_cols:
            val = r[col]
            if pd.isna(val):
                cells.append(html.Td("N/A", style={"color": SUBTEXT}))
            else:
                crowded = abs(float(val)) > 0.6
                cells.append(html.Td(
                    f"{val:.3f}",
                    style={
                        "color": RED if crowded else WHITE,
                        "fontWeight": 700 if crowded else 400,
                        "background": "rgba(255,107,107,0.12)" if crowded else "transparent",
                    },
                ))
        rows.append(html.Tr(cells))

    rows.append(html.Tr([html.Td(
        "Red cells: cross-exchange correlation > 0.6 - crowding signal. "
        "Based on aligned monthly L/S return series.",
        colSpan=1 + len(factor_cols),
        style={"color": SUBTEXT, "fontSize": "0.75rem", "fontStyle": "italic"},
    )]))

    return dbc.Table(
        [header, html.Tbody(rows)],
        bordered=False, hover=True, size="sm",
        className="table-dark table-hover align-middle mb-0",
        style={"fontSize": "0.82rem"},
    )


def _quintile_membership_tabs() -> html.Div:
    """Tabbed display: for each factor, show which stocks sit in Q1–Q5 (latest date)."""
    membership = load_quintile_stock_membership()

    FACTOR_ORDER = ["Momentum", "Quality", "Value", "Low-Vol"]
    Q_LABELS = {
        "Q1": ("Best", "#1A4480"),
        "Q2": ("2nd",  "#2E6DA4"),
        "Q3": ("3rd",  "#5C7A8A"),
        "Q4": ("4th",  "#C97B4B"),
        "Q5": ("Worst","#9B2626"),
    }

    if not membership:
        return html.Div(
            "Quintile assignment data unavailable - run factor_construction.py first.",
            style={"color": SUBTEXT, "padding": "24px", "fontSize": "0.84rem",
                   "textAlign": "center"},
        )

    ticker_names = load_ticker_names()

    tabs = []
    for factor in FACTOR_ORDER:
        qmap = membership.get(factor)
        if not qmap:
            tabs.append(dbc.Tab(
                html.Div(f"No data for {factor}.",
                         style={"color": SUBTEXT, "padding": "20px", "fontSize": "0.83rem"}),
                label=factor, tab_id=f"qm-{factor.lower().replace('-','').replace(' ','')}",
            ))
            continue

        fac_color = FACTOR_COLORS.get(factor, WHITE)
        cols = []
        for q, (q_label, q_bg) in Q_LABELS.items():
            tickers = qmap.get(q, [])
            badges = []
            for tk in tickers:
                short = tk.split(".")[0]
                name = ticker_names.get(tk, "")
                label_text = f"{name} ({short})" if name and name != short else short
                tooltip_text = f"{tk}" if not name or name == short else f"{tk} - {name}"
                badges.append(
                    html.Span(
                        label_text,
                        title=tooltip_text,
                        style={
                            "display": "inline-block",
                            "background": q_bg,
                            "color": WHITE,
                            "borderRadius": "4px",
                            "padding": "2px 7px",
                            "margin": "2px 2px",
                            "fontSize": "0.72rem",
                            "fontFamily": "Inter, monospace",
                            "fontWeight": 600,
                            "letterSpacing": "0.02em",
                            "cursor": "default",
                        },
                    )
                )

            cols.append(dbc.Col([
                html.Div([
                    html.Span(q, style={
                        "fontWeight": 800, "color": WHITE,
                        "fontSize": "0.88rem", "marginRight": "6px",
                    }),
                    html.Span(f"({q_label})", style={
                        "color": SUBTEXT, "fontSize": "0.74rem",
                    }),
                    html.Span(
                        f" {len(tickers)}",
                        style={"color": GOLD, "fontWeight": 700,
                               "fontSize": "0.80rem", "marginLeft": "4px"},
                    ),
                ], className="mb-2"),
                html.Div(
                    badges if badges else html.Span("-", style={"color": SUBTEXT}),
                    style={"lineHeight": "1.8"},
                ),
            ], xs=12, sm=6, md=True, className="mb-3",
               style={
                   "borderLeft": f"3px solid {q_bg}",
                   "paddingLeft": "10px",
               }))

        tabs.append(dbc.Tab(
            html.Div(
                dbc.Row(cols, className="g-2 mt-1"),
                className="p-2",
            ),
            label=factor,
            tab_id=f"qm-{factor.lower().replace('-','').replace(' ','')}",
            label_style={"color": fac_color, "fontWeight": 600},
            active_label_style={"color": fac_color, "borderBottomColor": fac_color,
                                 "fontWeight": 700},
        ))

    first_factor = FACTOR_ORDER[0].lower().replace("-", "").replace(" ", "")
    return html.Div([
        dbc.Tabs(tabs, active_tab=f"qm-{first_factor}"),
        html.Div(
            "Hover over a ticker badge to see full symbol. "
            "Quintile 1 = top-ranked (best signal); Quintile 5 = bottom-ranked. "
            "Based on latest available cross-sectional snapshot.",
            className="mt-2",
            style={"color": SUBTEXT, "fontSize": "0.76rem", "fontStyle": "italic"},
        ),
    ])


def _crowding_tabs_section() -> html.Div:
    intra = compute_intra_exchange_corr()
    cross_df = compute_cross_exchange_corr()

    tab_configs = [
        ("ADX",    "Abu Dhabi Securities Exchange"),
        ("DFM",    "Dubai Financial Market"),
        ("Tadawul","Saudi Exchange"),
        ("QSE",    "Qatar Stock Exchange"),
        ("Kuwait", "Boursa Kuwait"),
    ]

    tabs = []
    for exch, full_name in tab_configs:
        corr_df = intra.get(exch)
        if corr_df is None or len(corr_df) < 2:
            content = html.Div(
                f"Insufficient factor data for {full_name} "
                "(fewer than 2 factor return series available for this exchange).",
                style={"color": SUBTEXT, "padding": "48px 24px",
                       "fontSize": "0.85rem", "textAlign": "center"},
            )
        else:
            content = dcc.Graph(
                figure=_corr_heatmap_fig(corr_df, exch),
                config=CHART_CFG,
            )
        tabs.append(dbc.Tab(content, label=exch, tab_id=f"crowd-{exch.lower()}"))

    return html.Div([
        dbc.Tabs(tabs, active_tab="crowd-adx"),
        html.Div(className="mt-4"),
        html.H6("Cross-Exchange Factor Overlap",
                style={"color": GOLD, "fontWeight": 600, "marginBottom": "10px"}),
        html.Div(_cross_exchange_table(cross_df), style={"overflowX": "auto"}),
        html.Div(className="mt-3"),
        dbc.Alert(
            [
                html.I(className="bi bi-info-circle-fill me-2"),
                html.Strong("Factor Crowding: "),
                "High cross-exchange correlation (>0.6, shown in red) suggests the same macro "
                "driver prices the factor across markets - a crowding signal. "
                "Low correlation confirms exchange-specific alpha and supports diversification.",
            ],
            color="info",
            className="mb-0",
            style={
                "fontSize": "0.83rem", "borderRadius": "6px",
                "background": "rgba(74,159,212,0.10)",
                "borderColor": "rgba(74,159,212,0.28)",
                "color": WHITE,
            },
        ),
    ])


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _card(title: str, content, className: str = "") -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(
            title,
            style={"background": "rgba(255,255,255,0.04)", "fontWeight": 600,
                   "color": WHITE, "fontSize": "0.85rem", "letterSpacing": "0.03em"},
        ),
        dbc.CardBody(content, className=f"p-0 {className}"),
    ], className="mb-3 border-0",
       style={"background": CARD_BG, "borderRadius": "8px", "overflow": "hidden"})


def layout() -> html.Div:
    fm = load_fm_summary()
    ic = load_ic_monthly()
    bs = load_backtest_summary()
    idf = compute_fm_interactions()

    return html.Div([
        # Page header
        html.Div([
            html.H4("Layer 1 - Factor Analysis", className="mb-1",
                    style={"color": GOLD, "fontWeight": 700}),
            html.P("Fama-MacBeth cross-sectional regressions across 167 GCC stocks · "
                   "2015–2025 · Quintile analysis per exchange",
                   className="mb-0", style={"color": SUBTEXT, "fontSize": "0.88rem"}),
        ], className="mb-4"),

        # Row 1: summary table + FM t-stat table
        dbc.Row([
            dbc.Col(_card("Factor Summary - L/S Portfolio Statistics",
                          html.Div(_factor_summary_table(fm), className="p-2")),
                    md=7),
            dbc.Col(_card("Fama-MacBeth Coefficients (Newey-West 4-lag)",
                          html.Div(_fm_tstat_table(fm), className="p-2")),
                    md=5),
        ]),

        # Row 2: quintile spread chart (dynamic / reacts to oil-adjusted toggle)
        dbc.Row([
            dbc.Col(_card("Quintile Return Spreads",
                          dcc.Graph(id="quintile-chart-main", config=CHART_CFG,
                                    className="p-0")),
                    md=12),
        ]),

        # Row 2b: quintile stock membership - collapsible
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(
                        html.Div(
                            dbc.Row([
                                dbc.Col(
                                    html.Span(
                                        "Quintile Stock Membership - Latest Snapshot",
                                        style={"fontWeight": 600, "color": WHITE,
                                               "fontSize": "0.85rem", "letterSpacing": "0.03em"},
                                    ),
                                    className="d-flex align-items-center",
                                ),
                                dbc.Col(
                                    html.I(
                                        id="qm-collapse-icon",
                                        className="bi bi-chevron-up",
                                        style={"color": SUBTEXT, "fontSize": "1rem"},
                                    ),
                                    width="auto",
                                    className="d-flex align-items-center",
                                ),
                            ], className="g-0 justify-content-between"),
                            id="qm-collapse-header",
                            n_clicks=0,
                            style={"cursor": "pointer", "userSelect": "none"},
                        ),
                        style={"background": "rgba(255,255,255,0.04)"},
                    ),
                    dbc.Collapse(
                        dbc.CardBody(
                            _quintile_membership_tabs(),
                            className="p-0",
                        ),
                        id="qm-collapse-body",
                        is_open=True,
                    ),
                ], className="mb-3 border-0",
                   style={"background": CARD_BG, "borderRadius": "8px", "overflow": "hidden"}),
                md=12,
            ),
        ]),

        # Row 3: exchange breakdown
        dbc.Row([
            dbc.Col(_card("L/S Sharpe Ratio by Exchange × Factor",
                          dcc.Graph(figure=_exchange_breakdown_chart(bs), config=CHART_CFG)),
                    md=12),
        ]),

        # Row 4: rolling IC + interaction terms
        dbc.Row([
            dbc.Col(_card("Rolling 12-Month IC - Factor Predictability",
                          dcc.Graph(figure=_rolling_ic_chart(ic), config=CHART_CFG)),
                    md=7),
            dbc.Col(_card("Interaction Terms: Factor × Exchange (FM differential)",
                          html.Div(_interaction_table(idf), className="p-2",
                                   style={"maxHeight": "340px", "overflowY": "auto"})),
                    md=5),
        ]),

        # Row 5: Macroeconomic Control / Brent Crude Sensitivity (GAP 2)
        dbc.Row([
            dbc.Col(_card(
                "Macroeconomic Control: Brent Crude Sensitivity",
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dcc.Loading(
                                html.Div(id="oil-table-container", className="p-2"),
                                type="dot",
                            ),
                        ], md=7),
                        dbc.Col([
                            html.Div([
                                html.P(
                                    "Oil-Adjusted Returns",
                                    style={"color": WHITE, "fontWeight": 600,
                                           "marginBottom": "6px", "fontSize": "0.88rem"},
                                ),
                                dbc.Switch(
                                    id="oil-adjusted-toggle",
                                    label="Residualize factor returns vs oil - "
                                          "re-draws quintile bars above",
                                    value=False,
                                    style={"color": SUBTEXT, "fontSize": "0.83rem"},
                                ),
                                html.Div(
                                    id="oil-status-text",
                                    className="mt-2",
                                    style={"color": SUBTEXT, "fontSize": "0.77rem"},
                                ),
                            ], className="p-3"),
                        ], md=5),
                    ]),
                ]),
            ), md=12),
        ]),

        # Row 6: Factor Crowding / Cross-Exchange Correlation (GAP 1)
        dbc.Row([
            dbc.Col(_card(
                "Factor Crowding - Within-Exchange Factor Correlations",
                html.Div(_crowding_tabs_section(), className="p-2"),
            ), md=12),
        ]),
    ], className="p-3")


def register_callbacks(app):
    from dash import Input, Output, State
    from utils.data_loader import (
        load_quintile_spreads,
        compute_oil_adjusted_quintile_spreads,
        compute_oil_regression,
    )

    @app.callback(
        Output("quintile-chart-main", "figure"),
        Output("oil-status-text", "children"),
        Output("oil-table-container", "children"),
        Input("oil-adjusted-toggle", "value"),
    )
    def update_quintile_chart(oil_on: bool):
        if oil_on:
            qs, msg = compute_oil_adjusted_quintile_spreads()
            status = f"Oil-adjusted mode active. {msg}"
        else:
            qs = load_quintile_spreads()
            status = "Showing raw (unadjusted) quintile returns."
        oil_df, oil_status = compute_oil_regression()
        return _quintile_chart(qs), status, _oil_table(oil_df, oil_status)

    @app.callback(
        Output("qm-collapse-body", "is_open"),
        Output("qm-collapse-icon", "className"),
        Input("qm-collapse-header", "n_clicks"),
        State("qm-collapse-body", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_quintile_membership(n_clicks, is_open):
        new_open = not is_open
        icon = "bi bi-chevron-up" if new_open else "bi bi-chevron-down"
        return new_open, icon
