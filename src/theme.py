"""Visual theme for the Stock Market AI Streamlit dashboard (Phase 13 UI).

A single CSS string + a few HTML helpers that restyle the stock default look
into a dark, red-and-orange "control room" dashboard.  Everything here is
presentation only - no data logic - so ``app.py`` stays the single source of
truth for computation.

Only *stable* Streamlit hooks are targeted (``data-testid`` attributes that
have not changed across recent releases): ``stAppViewContainer``, ``stSidebar``,
``stHeader``, ``stMetric`` + its label/value/delta children, ``stButton``,
``stTabs`` / ``data-baseweb="tab"``, ``stSlider``, ``stSelectbox`` and
``stPlotlyChart``.  The richer "cards" (hero panel, news list) are plain HTML
blocks (``.sma-panel`` / ``.sma-panel-accent``) that need no Streamlit DOM.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Palette (sampled from the reference dashboard)
# --------------------------------------------------------------------------
BG_BASE = "#0d0605"          # page background - near-black maroon
CARD_TOP = "#1e100d"         # card gradient - top
CARD_BOTTOM = "#160b09"      # card gradient - bottom
CARD_BORDER = "rgba(255, 92, 42, 0.16)"
ACCENT = "#ff5c1a"           # vibrant orange
ACCENT_SOFT = "#ff7a3d"
ACCENT_RED = "#e03a1a"       # deep red for gradients
GOLD = "#ffc24a"             # gauge highlight
TEXT = "#f4ece8"
TEXT_MUTED = "#a48d84"
GREEN = "#3ddc84"
RED = "#ff5a4e"

FONT_STACK = (
    "'Poppins', 'Segoe UI', system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif"
)


def _gradient_panel() -> str:
    """Soft radial red glows layered over the base colour."""
    return f"""
    background:
        radial-gradient(1100px 620px at 88% -8%, rgba(224, 58, 26, 0.16), transparent 60%),
        radial-gradient(900px 520px at -12% 12%, rgba(255, 92, 26, 0.10), transparent 55%),
        radial-gradient(700px 700px at 50% 118%, rgba(120, 20, 12, 0.28), transparent 60%),
        linear-gradient(180deg, #120807 0%, {BG_BASE} 100%);
    background-attachment: fixed;
    """


THEME_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

:root {{
    --sma-accent: {ACCENT};
    --sma-accent-soft: {ACCENT_SOFT};
    --sma-accent-red: {ACCENT_RED};
    --sma-gold: {GOLD};
    --sma-text: {TEXT};
    --sma-muted: {TEXT_MUTED};
    --sma-card-top: {CARD_TOP};
    --sma-card-bottom: {CARD_BOTTOM};
    --sma-card-border: {CARD_BORDER};
    --sma-green: {GREEN};
    --sma-red: {RED};
}}

html, body, [class*="css"] {{
    font-family: {FONT_STACK};
}}

/* --------------------------------------------------------------- page -- */
.stApp {{
    color: {TEXT};
    font-family: {FONT_STACK};
}}

[data-testid="stAppViewContainer"] {{
    {_gradient_panel()}
}}

[data-testid="stMainBlockContainer"] {{
    padding-top: 2.4rem;
    padding-bottom: 4rem;
    max-width: 1180px;
}}

/* -------------------------------------------------------------- header -- */
[data-testid="stHeader"] {{
    background: transparent;
}}

[data-testid="stDecoration"] {{
    background: linear-gradient(90deg, transparent, {ACCENT}, transparent);
    opacity: 0.0;
}}

#MainMenu, footer {{visibility: hidden;}}

/* ------------------------------------------------------------ sidebar --- */
[data-testid="stSidebar"] {{
    background:
        linear-gradient(180deg, #170c0a 0%, #0f0706 100%);
    border-right: 1px solid rgba(255, 92, 42, 0.14);
}}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {{
    color: {TEXT};
    font-weight: 600;
    letter-spacing: 0.2px;
}}

[data-testid="stSidebar"] label, .stMarkdown p {{
    color: {TEXT_MUTED};
}}

/* ----------------------------------------------------------- headings --- */
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2 {{
    color: {TEXT};
    font-weight: 600;
    letter-spacing: 0.2px;
}}

[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5 {{
    color: {ACCENT_SOFT};
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.78rem;
    letter-spacing: 1.4px;
}}

p, li, label {{
    color: {TEXT};
}}

div[data-testid="stCaptionContainer"], .stCaption {{
    color: {TEXT_MUTED} !important;
}}
[data-testid="stCaptionContainer"] p {{ color: {TEXT_MUTED}; }}

/* --------------------------------------------------------------- cards -- */
/* The visual "cards" of the dashboard are (a) the metrics themselves and
   (b) the pure-HTML ``.sma-panel`` / ``.sma-panel-accent`` blocks, both of
   which use stable hooks. Any bordered st.container just gets a matching
   accent border + radius so it blends with the theme. */
[data-testid="stVerticalBlock"] {{
    border-color: {CARD_BORDER} !important;
    border-radius: 16px;
}}

/* ------------------------------------------------------------ metrics --- */
[data-testid="stMetric"] {{
    background: linear-gradient(165deg, {CARD_TOP} 0%, {CARD_BOTTOM} 100%);
    border: 1px solid {CARD_BORDER};
    border-radius: 16px;
    padding: 16px 18px 14px 18px;
    box-shadow: 0 14px 30px -22px rgba(0, 0, 0, 0.9);
    height: 100%;
}}

[data-testid="stMetricLabel"] > div, [data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED} !important;
    font-weight: 500 !important;
    font-size: 0.74rem !important;
    text-transform: uppercase;
    letter-spacing: 1.1px;
}}

[data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-weight: 600 !important;
    font-size: 1.65rem !important;
    letter-spacing: 0.3px;
}}

[data-testid="stMetricDelta"] {{
    font-weight: 500 !important;
}}

/* ------------------------------------------------------------ buttons --- */
div[data-testid="stButton"] > button {{
    background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_RED} 100%);
    color: #fff;
    border: none;
    border-radius: 999px;
    padding: 0.45rem 1.3rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
    box-shadow: 0 10px 24px -14px rgba(255, 92, 26, 0.9);
}}

div[data-testid="stButton"] > button:hover {{
    transform: translateY(-1px);
    filter: brightness(1.08);
    box-shadow: 0 14px 30px -12px rgba(255, 92, 26, 0.95);
    color: #fff;
}}

div[data-testid="stButton"] > button:focus {{
    box-shadow: 0 0 0 2px rgba(255, 92, 26, 0.45);
}}

/* --------------------------------------------------------------- tabs --- */
div[data-baseweb="tab-list"] {{
    background: linear-gradient(165deg, {CARD_TOP}, {CARD_BOTTOM});
    border: 1px solid {CARD_BORDER};
    border-radius: 14px;
    padding: 6px;
    gap: 6px;
}}

div[data-baseweb="tab"] {{
    background: transparent;
    border-radius: 10px;
    color: {TEXT_MUTED};
    font-weight: 500;
    letter-spacing: 0.3px;
}}

div[data-baseweb="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_RED} 100%);
    color: #fff;
}}

div[data-baseweb="tab-highlight"] {{ display: none; }}
div[data-baseweb="tab-borders"] {{ display: none; }}

/* ------------------------------------------------------- select / slider */
div[data-baseweb="select"] > div {{
    background: {CARD_TOP};
    border: 1px solid {CARD_BORDER};
    border-radius: 12px;
    color: {TEXT};
}}

div[data-baseweb="singleSelect"] div[data-baseweb="input"] {{
    background: transparent;
    color: {TEXT};
}}

[data-testid="stSlider"] div[data-baseweb="thumb"] {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
[data-testid="stSlider"] div[data-baseweb="track"] {{
    background: rgba(255, 92, 42, 0.35);
}}
[data-testid="stSlider"] div[data-baseweb="track"]::before {{
    background: {ACCENT};
}}

/* -------------------------------------------------------------- alerts --- */
[data-testid="stAlert"] {{
    border-radius: 14px;
    border: 1px solid {CARD_BORDER};
}}

/* ------------------------------------------------------------- charts ---- */
[data-testid="stPlotlyChart"] {{
    border-radius: 14px;
}}

/* ------------------------------------------- custom HTML card helpers ---- */
.sma-panel {{
    background: linear-gradient(165deg, {CARD_TOP} 0%, {CARD_BOTTOM} 100%);
    border: 1px solid {CARD_BORDER};
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 18px 40px -26px rgba(0, 0, 0, 0.9);
    height: 100%;
}}

.sma-panel-accent {{
    background: linear-gradient(150deg, {ACCENT} 0%, {ACCENT_RED} 82%);
    border: none;
    border-radius: 18px;
    padding: 22px 24px;
    box-shadow: 0 22px 46px -20px rgba(255, 92, 26, 0.55);
    color: #fff;
    height: 100%;
}}

/* Force crisp, fully-opaque white text inside the orange panel so nothing is
   washed out by the global <p> colour or the muted opacity used elsewhere. */
.sma-panel-accent p,
.sma-panel-accent .sma-eyebrow,
.sma-panel-accent .sma-big,
.sma-panel-accent .sma-sub {{
    color: #fff !important;
    opacity: 1 !important;
    text-shadow: 0 1px 2px rgba(90, 20, 6, 0.35);
}}

.sma-eyebrow {{
    text-transform: uppercase;
    letter-spacing: 1.6px;
    font-size: 0.7rem;
    font-weight: 700;
    opacity: 0.9;
    margin: 0 0 6px 0;
}}

.sma-big {{
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.05;
    margin: 0;
}}

.sma-sub {{
    font-size: 0.82rem;
    font-weight: 500;
    opacity: 0.95;
    margin: 6px 0 0 0;
}}

.sma-chip {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.4px;
    margin: 3px 6px 3px 0;
}}

.sma-chip-up    {{ background: rgba(61, 220, 132, 0.16); color: {GREEN}; border: 1px solid rgba(61, 220, 132, 0.4); }}
.sma-chip-down  {{ background: rgba(255, 90, 78, 0.16);  color: {RED};   border: 1px solid rgba(255, 90, 78, 0.4); }}
.sma-chip-neutral{{ background: rgba(255, 194, 74, 0.14); color: {GOLD};  border: 1px solid rgba(255, 194, 74, 0.4); }}
.sma-chip-onaccent {{ background: rgba(255, 255, 255, 0.18); color: #fff; border: 1px solid rgba(255, 255, 255, 0.35); }}

.sma-news-row {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 4px;
    border-bottom: 1px solid rgba(255, 92, 42, 0.10);
    font-size: 0.86rem;
    line-height: 1.4;
    color: {TEXT};
}}
.sma-news-row:last-child {{ border-bottom: none; }}
.sma-news-date {{ color: {TEXT_MUTED}; white-space: nowrap; font-size: 0.78rem; padding-top: 1px; }}
.sma-news-head {{ flex: 1; }}
</style>
"""


# --------------------------------------------------------------------------
# Small HTML helpers used by app.py
# --------------------------------------------------------------------------
def eyebrow(text: str) -> str:
    """A small uppercase section label."""
    return f'<p class="sma-eyebrow">{text}</p>'


def accent_panel(title: str, big: str, sub: str = "", chips: str = "") -> str:
    """The orange hero panel (mirrors the reference 'Areas of life' card)."""
    return (
        '<div class="sma-panel-accent">'
        f'<p class="sma-eyebrow">{title}</p>'
        f'<p class="sma-big">{big}</p>'
        f'<p class="sma-sub">{sub}</p>'
        f'<div style="margin-top:12px">{chips}</div>'
        "</div>"
    )


def chip(text: str, kind: str = "neutral", on_accent: bool = False) -> str:
    """A pill badge. ``kind`` is up / down / neutral."""
    if on_accent:
        cls = "sma-chip sma-chip-onaccent"
    else:
        cls = f"sma-chip sma-chip-{kind}"
    return f'<span class="{cls}">{text}</span>'
