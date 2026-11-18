"""GCC Equity Factor Research - Plotly Dash Dashboard.

Run:  python app.py
"""

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

from pages import factor_model, portfolio, summary, backtesting

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "GCC Equity Factor Research"
server = app.server  # expose WSGI for deployment

# Register page callbacks
factor_model.register_callbacks(app)
portfolio.register_callbacks(app)
backtesting.register_callbacks(app)
summary.register_callbacks(app)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
NAV_LINKS = [
    ("/factor-model",  "bi-bar-chart-line", "Layer 1", "Factor Analysis"),
    ("/portfolio",     "bi-graph-up",       "Layer 2", "Base Portfolio"),
    ("/backtesting",   "bi-activity",       "Layer 3", "Backtesting"),
    ("/summary",       "bi-clipboard-data", "",        "Research Summary"),
]

GOLD = "#C9A84C"
SUBTEXT = "#A0AEC0"


def _nav_item(href: str, icon: str, badge: str, label: str) -> dbc.NavItem:
    badge_el = dbc.Badge(
        badge, color="warning", pill=True, className="ms-1",
        style={"fontSize": "0.65rem", "fontWeight": 700},
    ) if badge else ""
    return dbc.NavItem(
        dbc.NavLink(
            [html.I(className=f"bi {icon} me-2"), label, badge_el],
            href=href,
            id=f"nav-{href.strip('/')}",
            className="nav-link-sidebar",
            style={"fontSize": "0.86rem", "padding": "10px 16px",
                   "borderRadius": "6px", "marginBottom": "2px"},
        )
    )


sidebar = html.Div([
    # Branding
    html.Div([
        html.Div("GCC", style={"fontSize": "1.6rem", "fontWeight": 900,
                                "color": GOLD, "letterSpacing": "-0.02em",
                                "lineHeight": 1}),
        html.Div("Equity Factor Research",
                 style={"fontSize": "0.7rem", "color": SUBTEXT,
                        "textTransform": "uppercase", "letterSpacing": "0.1em",
                        "marginTop": "2px"}),
    ], className="px-3 pt-3 pb-3"),

    html.Hr(style={"borderColor": "rgba(255,255,255,0.08)", "margin": "0 0 8px 0"}),

    # Navigation
    dbc.Nav(
        [_nav_item(*args) for args in NAV_LINKS],
        vertical=True,
        pills=True,
        className="px-2",
    ),

    html.Hr(style={"borderColor": "rgba(255,255,255,0.08)", "margin": "16px 0 8px 0"}),

    # Footer stats
    html.Div([
        html.Div("167 stocks · 5 exchanges",
                 style={"color": SUBTEXT, "fontSize": "0.72rem"}),
        html.Div("2015 – 2025 · ~135 months",
                 style={"color": SUBTEXT, "fontSize": "0.72rem"}),
        html.Div("Fama-MacBeth (1973) methodology",
                 style={"color": SUBTEXT, "fontSize": "0.72rem", "marginTop": "4px"}),
    ], className="px-3 pb-3"),
], id="sidebar", className="sidebar-wrapper")

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dbc.Row([
        # Sidebar column
        dbc.Col(sidebar, width=2, className="p-0",
                style={"minHeight": "100vh"}),
        # Content column
        dbc.Col([
            html.Div(id="page-content", className="content-area"),
        ], width=10, className="p-0"),
    ], className="g-0"),
], style={"minHeight": "100vh"})

# ---------------------------------------------------------------------------
# Routing callback
# ---------------------------------------------------------------------------
PAGE_MAP = {
    "/factor-model": factor_model.layout,
    "/portfolio":    portfolio.layout,
    "/backtesting":  backtesting.layout,
    "/summary":      summary.layout,
}


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def render_page(pathname: str):
    fn = PAGE_MAP.get(pathname, factor_model.layout)
    return fn()


# Active nav link highlighting
@app.callback(
    [Output(f"nav-{href.strip('/')}", "active") for href, *_ in NAV_LINKS],
    Input("url", "pathname"),
)
def set_active_nav(pathname):
    return [pathname == href for href, *_ in NAV_LINKS]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
