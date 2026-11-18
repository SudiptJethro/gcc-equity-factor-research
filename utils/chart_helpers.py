"""Shared chart styling and helper utilities."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
NAVY = "#0D1B2A"
GOLD = "#C9A84C"
SLATE = "#7A8BA0"
TEAL = "#4ECCA3"
RED = "#FF6B6B"
BLUE = "#4A9FD4"
PURPLE = "#9B7FD4"
GREEN = "#52D68A"
ORANGE = "#F5A623"
WHITE = "#E8EAF0"
SUBTEXT = "#A0AEC0"
CARD_BG = "#1C2840"
BORDER = "rgba(255,255,255,0.08)"

FACTOR_COLORS = {
    "MOMENTUM": GOLD,
    "QUALITY": BLUE,
    "VALUE": TEAL,
    "LOW-VOL": RED,
    "Momentum": GOLD,
    "Quality": BLUE,
    "Value": TEAL,
    "Low-Vol": RED,
}

EXCHANGE_COLORS = {
    "GCC": WHITE,
    "Tadawul": GOLD,
    "ADX": BLUE,
    "DFM": TEAL,
    "QSE": PURPLE,
    "Kuwait": ORANGE,
}

PORTFOLIO_COLORS = {
    "base_return": SLATE,
    "static_overlay_return": BLUE,
    "dynamic_overlay_return": GOLD,
}

QUINTILE_COLORS = ["#1565C0", "#5E92C8", "#8899AA", "#D4826B", "#C62828"]

CHART_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]}

# ---------------------------------------------------------------------------
# Base layout
# ---------------------------------------------------------------------------
_FONT = "Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif"

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=_FONT, color=WHITE, size=12),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.07)",
        showline=False,
        zeroline=False,
        tickfont=dict(color=SUBTEXT, size=11),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.07)",
        showline=False,
        zeroline=False,
        tickfont=dict(color=SUBTEXT, size=11),
    ),
    margin=dict(l=50, r=20, t=44, b=50),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        font=dict(color=WHITE, size=11),
    ),
    hoverlabel=dict(
        bgcolor=NAVY,
        bordercolor=GOLD,
        font=dict(color=WHITE, size=12),
    ),
)


def apply_layout(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    kw = {**BASE_LAYOUT, "height": height}
    if title:
        kw["title"] = dict(text=title, font=dict(color=WHITE, size=13, family=_FONT), x=0.01, y=0.97)
    fig.update_layout(**kw)
    return fig


def subplot_layout(fig: go.Figure, title: str = "", height: int = 440) -> go.Figure:
    """Apply base layout to multi-axis subplot figure."""
    kw = {**BASE_LAYOUT, "height": height}
    kw.pop("xaxis", None)
    kw.pop("yaxis", None)
    if title:
        kw["title"] = dict(text=title, font=dict(color=WHITE, size=13, family=_FONT), x=0.01, y=0.99)
    fig.update_layout(**kw)
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(
                gridcolor="rgba(255,255,255,0.07)",
                showline=False,
                zeroline=False,
                tickfont=dict(color=SUBTEXT, size=10),
            )
    return fig


# ---------------------------------------------------------------------------
# Finance helpers
# ---------------------------------------------------------------------------

def cumulative(s: pd.Series) -> pd.Series:
    return (1 + s).cumprod() - 1


def drawdown(s: pd.Series) -> pd.Series:
    cum = (1 + s).cumprod()
    return cum / cum.cummax() - 1


def rolling_sharpe(s: pd.Series, window: int = 12) -> pd.Series:
    ann = 12
    return (s.rolling(window).mean() * ann) / (s.rolling(window).std() * np.sqrt(ann))


def drawdown_periods(s: pd.Series, threshold: float = -0.05):
    """Return list of (start, end) date pairs where DD exceeds threshold."""
    dd = drawdown(s)
    in_dd = dd < threshold
    periods = []
    start = None
    for d, v in in_dd.items():
        if v and start is None:
            start = d
        elif not v and start is not None:
            periods.append((start, d))
            start = None
    if start is not None:
        periods.append((start, s.index[-1]))
    return periods


def add_drawdown_shading(fig: go.Figure, s: pd.Series, threshold: float = -0.05) -> go.Figure:
    for start, end in drawdown_periods(s, threshold):
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="rgba(255,107,107,0.12)",
            layer="below",
            line_width=0,
        )
    return fig
