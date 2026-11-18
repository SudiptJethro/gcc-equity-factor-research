"""Page 3 - Layer 3: VRP Overlay."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

from utils.data_loader import (
    load_portfolio_returns,
    load_vrp_data,
    load_vol_surface,
    load_regime_analysis,
    compute_vrp_alpha_regression,
    TICKERS,
)
from utils.chart_helpers import (
    apply_layout, subplot_layout,
    GOLD, BLUE, TEAL, RED, GREEN, WHITE, SUBTEXT, SLATE, ORANGE, PURPLE,
    CHART_CFG, CARD_BG,
    cumulative,
)

TICKER_NAMES = {
    "1120.SR": "Al Rajhi Bank",
    "2010.SR": "SABIC",
    "2222.SR": "Saudi Aramco",
    "7010.SR": "STC",
}
TICKER_COLORS = {
    "1120.SR": GOLD,
    "2010.SR": BLUE,
    "2222.SR": TEAL,
    "7010.SR": ORANGE,
}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _cumulative_overlay_chart(df: pd.DataFrame, mode: str = "dynamic") -> go.Figure:
    fig = go.Figure()
    series_all = {
        "base_return": ("Base Only", SLATE, dict(width=1.5, dash="dot")),
        "static_overlay_return": ("+ Static (3b ceiling)", BLUE, dict(width=2)),
        "dynamic_overlay_return": ("+ Dynamic (3c)", GOLD, dict(width=2.5)),
    }
    if mode == "static":
        series = {k: v for k, v in series_all.items()
                  if k in ("base_return", "static_overlay_return")}
    else:
        series = series_all

    for col, (label, color, lkw) in series.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        cum = cumulative(s) * 100
        fig.add_scatter(
            x=cum.index, y=cum.round(3),
            name=label, mode="lines",
            line=dict(color=color, **lkw),
            hovertemplate=f"<b>{label}</b><br>%{{x|%b %Y}}: %{{y:+.2f}}%<extra></extra>",
        )

    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.2)")
    apply_layout(fig, "Cumulative Return Comparison - VRP Overlay Layers", height=370)
    fig.update_yaxes(title_text="Cumulative Return (%)", ticksuffix="%",
                     title_font=dict(color=SUBTEXT, size=11))
    fig.update_layout(legend=dict(orientation="h", y=-0.22, x=0))
    return fig


def _vol_surface_panel(surf_df: pd.DataFrame) -> go.Figure:
    tickers = [t for t in TICKERS if t in surf_df["ticker"].unique()]
    n = len(tickers)
    rows, cols = 2, 2

    iv_col = "iv_pct" if "iv_pct" in surf_df.columns else "iv_synthetic"
    rv_col = "rv_pct" if "rv_pct" in surf_df.columns else "rv_blended"

    subplot_titles = [TICKER_NAMES.get(t, t) for t in tickers]
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    for i, ticker in enumerate(tickers):
        r, c = divmod(i, cols)
        tdf = surf_df[surf_df["ticker"] == ticker].sort_values("date")
        tc = TICKER_COLORS.get(ticker, WHITE)

        x = tdf["date"]
        iv = tdf[iv_col]
        rv = tdf[rv_col]

        # IV line
        fig.add_scatter(
            x=x, y=iv.round(2),
            name=f"IV ({TICKER_NAMES.get(ticker, ticker)})",
            mode="lines",
            line=dict(color=tc, width=1.8),
            legendgroup=ticker,
            showlegend=(i == 0),
            hovertemplate=f"<b>IV {ticker}</b><br>%{{x|%b %Y}}: %{{y:.1f}}%<extra></extra>",
            row=r + 1, col=c + 1,
        )
        # RV line
        fig.add_scatter(
            x=x, y=rv.round(2),
            name=f"RV ({TICKER_NAMES.get(ticker, ticker)})",
            mode="lines",
            line=dict(color=tc, width=1.5, dash="dot"),
            legendgroup=ticker,
            showlegend=(i == 0),
            hovertemplate=f"<b>RV {ticker}</b><br>%{{x|%b %Y}}: %{{y:.1f}}%<extra></extra>",
            row=r + 1, col=c + 1,
        )
        # VRP gap shading (fill between IV and RV)
        fig.add_scatter(
            x=list(x) + list(x[::-1]),
            y=list(iv.round(2)) + list(rv.round(2)[::-1]),
            fill="toself",
            fillcolor=f"rgba({int(tc[1:3], 16)},{int(tc[3:5], 16)},{int(tc[5:7], 16)},0.15)",
            line=dict(width=0),
            name=f"VRP gap ({ticker})",
            showlegend=False,
            hoverinfo="skip",
            row=r + 1, col=c + 1,
        )

    subplot_layout(fig, "Implied vs Realised Volatility - VRP Gap (shaded)", height=500)
    for i in range(1, 5):
        axis_id = "" if i == 1 else str(i)
        if hasattr(fig.layout, f"yaxis{axis_id}"):
            fig.layout[f"yaxis{axis_id}"].update(
                ticksuffix="%",
                title_text="Vol (%)" if (i - 1) % 2 == 0 else "",
                title_font=dict(color=SUBTEXT, size=10),
            )
    for ann in fig.layout.annotations:
        ann.font.color = SUBTEXT
        ann.font.size = 11
    fig.update_layout(showlegend=False)
    return fig


def _regime_table(regime_df: pd.DataFrame) -> dbc.Table:
    cols = regime_df.columns.tolist()
    header = html.Thead(html.Tr(
        [html.Th(c) for c in cols], className="table-dark"
    ))
    regime_colors = {"calm": TEAL, "normal": BLUE, "stressed": RED}
    rows = []
    for _, r in regime_df.iterrows():
        regime_name = str(r.get("Regime", r.iloc[0])).lower()
        color = regime_colors.get(regime_name, WHITE)
        cells = []
        for ci, col in enumerate(cols):
            val = r[col]
            if ci == 0:
                cells.append(html.Td(str(val).title(),
                                     style={"color": color, "fontWeight": 700}))
            elif isinstance(val, float):
                cells.append(html.Td(f"{val:.2f}"))
            else:
                cells.append(html.Td(str(val)))
        rows.append(html.Tr(cells))

    return dbc.Table(
        [header, html.Tbody(rows)],
        bordered=False, hover=True, size="sm",
        className="table-dark table-hover align-middle mb-0",
        style={"fontSize": "0.82rem"},
    )


def _earnings_effect_chart(vrp_df: pd.DataFrame) -> go.Figure:
    cc_col = "cc_premium_pct"
    earn_col = "earnings_month"
    if cc_col not in vrp_df.columns or earn_col not in vrp_df.columns:
        return go.Figure()

    grp = (
        vrp_df.groupby(earn_col)[cc_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grp["label"] = grp[earn_col].map({True: "Earnings Month", False: "Non-Earnings"})

    colors = [GOLD, SLATE]
    fig = go.Figure()
    for i, row in grp.iterrows():
        se = row["std"] / np.sqrt(max(row["count"], 1))
        fig.add_bar(
            x=[row["label"]],
            y=[round(row["mean"], 4)],
            name=row["label"],
            marker_color=colors[i % len(colors)],
            error_y=dict(type="data", array=[se], color="rgba(255,255,255,0.4)"),
            text=[f"{row['mean']:.3f}%"],
            textposition="outside",
            textfont=dict(color=WHITE, size=11),
        )

    apply_layout(fig, "CC Yield: Earnings vs Non-Earnings Months", height=320)
    fig.update_yaxes(title_text="Avg CC Premium (%/mo)", title_font=dict(color=SUBTEXT, size=11),
                     ticksuffix="%")
    fig.update_layout(showlegend=False, bargap=0.5)
    return fig


def _overlay_firing_chart(vrp_df: pd.DataFrame) -> go.Figure:
    if "overlay_fired" not in vrp_df.columns and "signal_fire" not in vrp_df.columns:
        return go.Figure()

    fired_col = "overlay_fired" if "overlay_fired" in vrp_df.columns else "signal_fire"
    cc_col = "cc_premium_pct"
    regime_col = "regime" if "regime" in vrp_df.columns else "sso_regime"

    monthly = (
        vrp_df[vrp_df[fired_col] == True]
        .groupby(["date", regime_col])
        .agg(avg_cc=(cc_col, "mean"))
        .reset_index()
    )

    regime_colors = {"normal": BLUE, "stressed": RED, "calm": TEAL}
    fig = go.Figure()
    for regime, rdf in monthly.groupby(regime_col):
        color = regime_colors.get(str(regime).lower(), SLATE)
        fig.add_bar(
            x=rdf["date"],
            y=rdf["avg_cc"].round(4),
            name=str(regime).title(),
            marker_color=color,
            hovertemplate=f"<b>{str(regime).title()}</b><br>%{{x|%b %Y}}: %{{y:.3f}}%<extra></extra>",
        )

    apply_layout(fig, "Monthly Overlay Firing: CC Premium Collected by Regime", height=320)
    fig.update_yaxes(title_text="Avg CC Premium (%/mo)", title_font=dict(color=SUBTEXT, size=11),
                     ticksuffix="%")
    fig.update_layout(barmode="stack", legend=dict(orientation="h", y=-0.28, x=0))
    return fig


# ---------------------------------------------------------------------------
# GAP 3 helpers
# ---------------------------------------------------------------------------

def _vrp_alpha_table(df: pd.DataFrame) -> dbc.Table:
    header = html.Thead(html.Tr(
        [html.Th(c) for c in df.columns], className="table-dark"
    ))
    rows = []
    for _, r in df.iterrows():
        t_val = r.get("t-stat", "N/A")
        try:
            sig = abs(float(str(t_val).replace("N/A", "nan"))) >= 1.96
        except (ValueError, TypeError):
            sig = False
        cells = []
        for col in df.columns:
            v = r[col]
            style = {}
            if col == "Metric":
                style = {"fontWeight": 600, "color": WHITE}
            elif col == "t-stat" and sig:
                style = {"color": GOLD, "fontWeight": 700}
            cells.append(html.Td(str(v), style=style))
        rows.append(html.Tr(cells))
    return dbc.Table(
        [header, html.Tbody(rows)],
        bordered=False, hover=True, size="sm",
        className="table-dark table-hover align-middle mb-0",
        style={"fontSize": "0.82rem"},
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _card(title, content) -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader(title, style={
            "background": "rgba(255,255,255,0.04)", "fontWeight": 600,
            "color": WHITE, "fontSize": "0.85rem", "letterSpacing": "0.03em",
        }),
        dbc.CardBody(content, className="p-0"),
    ], className="mb-3 border-0",
       style={"background": CARD_BG, "borderRadius": "8px", "overflow": "hidden"})


def layout() -> html.Div:
    df_port = load_portfolio_returns()
    vrp_df = load_vrp_data()
    surf_df = load_vol_surface()
    regime_df = load_regime_analysis()
    vrp_alpha_df = compute_vrp_alpha_regression()

    # Compute incremental Sharpe (VRP dynamic minus base)
    base = df_port["base_return"].dropna()
    dyn_col = "dynamic_overlay_return"
    if dyn_col in df_port.columns:
        dyn = df_port[dyn_col].dropna()
        base_sharpe = (base.mean() * 12) / (base.std() * np.sqrt(12))
        vrp_sharpe = (dyn.mean() * 12) / (dyn.std() * np.sqrt(12))
        incr = vrp_sharpe - base_sharpe
        incr_str = f"{incr:+.2f}"
        incr_color = GREEN if incr > 0 else RED
    else:
        incr_str, incr_color = "N/A", SUBTEXT

    toggle = dbc.RadioItems(
        id="vrp-overlay-toggle",
        options=[
            {"label": " Static Overlay (3b ceiling)", "value": "static"},
            {"label": " Dynamic Overlay (3c)", "value": "dynamic"},
        ],
        value="dynamic",
        inline=True,
        inputStyle={"marginRight": "6px"},
        style={"color": WHITE, "fontSize": "0.88rem"},
    )

    # GAP 3 / Applied extension banner
    banner = dbc.Alert(
        [
            html.I(className="bi bi-lightbulb-fill me-2"),
            html.Strong("Applied Implementation Extension: "),
            "This section presents an applied implementation extension. The VRP overlay is "
            "exploratory - it tests whether volatility risk premium signals enhance the base "
            "factor strategy in practice, beyond the core thesis. "
            "Core thesis is answered by Pages 1–2 and the Summary scorecard.",
        ],
        color="warning",
        className="mb-3",
        style={
            "fontSize": "0.84rem", "borderRadius": "8px",
            "background": "rgba(201,168,76,0.12)",
            "borderColor": "rgba(201,168,76,0.40)",
            "color": WHITE,
        },
    )

    # Incremental Sharpe stat card
    incr_card = dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.P("Incremental Sharpe vs Base", className="mb-1",
                       style={"color": SUBTEXT, "fontSize": "0.78rem",
                              "fontWeight": 600, "textTransform": "uppercase",
                              "letterSpacing": "0.06em"}),
                html.H3(incr_str, className="mb-1",
                        style={"color": incr_color, "fontWeight": 800}),
                html.P("Dynamic (3c) Sharpe minus Base Sharpe",
                       className="mb-0",
                       style={"color": SUBTEXT, "fontSize": "0.76rem"}),
            ], className="p-3"),
        ], className="border-0 mb-3",
           style={"background": "#1C2840", "borderRadius": "8px"}),
        md=4,
    )

    return html.Div([
        html.Div([
            html.H4("Layer 3 - VRP Overlay Strategy", className="mb-1",
                    style={"color": GOLD, "fontWeight": 700}),
            html.P("Synthetic vol surface on 4 Saudi SSOs · "
                   "Dynamic 3c: Sharpe 1.49 · MaxDD -15.8% · Ann. Ret 18.1%",
                   className="mb-0", style={"color": SUBTEXT, "fontSize": "0.88rem"}),
        ], className="mb-2"),

        # GAP 3: Applied extension banner
        banner,

        # GAP 3: Incremental Sharpe stat card
        dbc.Row([incr_card]),

        # Toggle row
        dbc.Row([
            dbc.Col([
                html.Label("Overlay Mode:", style={"color": SUBTEXT, "fontSize": "0.82rem",
                                                   "marginRight": "12px"}),
                toggle,
            ], className="d-flex align-items-center mb-3"),
        ]),

        # Cumulative return (reacts to toggle)
        dbc.Row([
            dbc.Col(_card("Cumulative Return - Strategy Layers",
                          dcc.Graph(id="vrp-cum-chart", config=CHART_CFG)),
                    md=12),
        ]),

        # Vol surface
        dbc.Row([
            dbc.Col(_card("Implied vs Realised Volatility - VRP Gap",
                          dcc.Graph(figure=_vol_surface_panel(surf_df) if surf_df is not None
                                    else go.Figure(), config=CHART_CFG)),
                    md=12),
        ]),

        # Regime table + earnings effect
        dbc.Row([
            dbc.Col(_card("Regime Breakdown",
                          html.Div(_regime_table(regime_df), className="p-2",
                                   style={"overflowX": "auto"})),
                    md=5),
            dbc.Col(_card("CC Yield: Earnings vs Non-Earnings",
                          dcc.Graph(figure=_earnings_effect_chart(vrp_df) if vrp_df is not None
                                    else go.Figure(), config=CHART_CFG)),
                    md=7),
        ]),

        # Overlay firing rate
        dbc.Row([
            dbc.Col(_card("Monthly Overlay Firing Rate - CC Premium by Regime",
                          dcc.Graph(figure=_overlay_firing_chart(vrp_df) if vrp_df is not None
                                    else go.Figure(), config=CHART_CFG)),
                    md=12),
        ]),

        # GAP 3: VRP alpha regression vs base portfolio
        dbc.Row([
            dbc.Col(_card(
                "VRP Alpha Test - Regression of Overlay Excess Returns on Base Portfolio (NW-4)",
                html.Div([
                    html.P(
                        "Formal test: does the VRP overlay generate returns independent of the base "
                        "factor portfolio? Excess return = dynamic overlay return − base return. "
                        "OLS with Newey-West (4-lag) standard errors.",
                        style={"color": SUBTEXT, "fontSize": "0.82rem", "marginBottom": "12px"},
                    ),
                    html.Div(_vrp_alpha_table(vrp_alpha_df), className="p-0"),
                ], className="p-3"),
            ), md=12),
        ]),

        # Store for shared data
        dcc.Store(id="vrp-port-store"),
    ], className="p-3")


def register_callbacks(app):
    from dash import Input, Output

    @app.callback(
        Output("vrp-cum-chart", "figure"),
        Input("vrp-overlay-toggle", "value"),
    )
    def update_cum_chart(mode):
        df = load_portfolio_returns()
        return _cumulative_overlay_chart(df, mode=mode or "dynamic")
