from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as _components

_ASSETS = Path(__file__).resolve().parent / "assets"


def _load_orbital_bg() -> str:
    svg = (_ASSETS / "bg_orbital.svg").read_text()
    b64 = base64.b64encode(svg.encode()).decode()
    return f"data:image/svg+xml;base64,{b64}"


_ORBITAL_BG_URI = _load_orbital_bg()

_CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&display=swap');

/* ── CSS variables ── */
:root {
    --cyan: #4FC3F7;
    --cyan-bright: #00D4FF;
    --purple: #AB82FF;
    --bg-deep: #060910;
    --bg-card: rgba(12,20,38,0.65);
    --border-cyan: rgba(0,212,255,0.22);
    --border-purple: rgba(171,130,255,0.18);
    --glow-cyan: rgba(79,195,247,0.35);
    --glow-purple: rgba(171,130,255,0.30);
    --text-primary: #E8ECF0;
    --text-muted: #8899AA;
    --font-display: 'Orbitron', sans-serif;
    --font-body: 'Rajdhani', sans-serif;
    --font-mono: 'Share Tech Mono', monospace;
}

/* ── Data-grid + radial depth background ── */
body::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: -3;
    background:
        /* Radial depth gradient – soft vignette from center */
        radial-gradient(ellipse at 50% 40%, rgba(10,24,50,0.5) 0%, transparent 65%),
        radial-gradient(ellipse at 80% 20%, rgba(79,195,247,0.04) 0%, transparent 40%),
        radial-gradient(ellipse at 15% 70%, rgba(171,130,255,0.03) 0%, transparent 40%),
        /* Grid lines */
        linear-gradient(rgba(79,195,247,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(79,195,247,0.035) 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 100% 100%, 60px 60px, 60px 60px;
}

body::after {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: -2;
    background:
        radial-gradient(1px 1px at 10% 20%, rgba(255,255,255,0.45), transparent),
        radial-gradient(1px 1px at 30% 60%, rgba(255,255,255,0.3), transparent),
        radial-gradient(1.5px 1.5px at 50% 10%, rgba(79,195,247,0.6), transparent),
        radial-gradient(1px 1px at 70% 80%, rgba(255,255,255,0.3), transparent),
        radial-gradient(1px 1px at 90% 40%, rgba(255,255,255,0.4), transparent),
        radial-gradient(2px 2px at 15% 75%, rgba(79,195,247,0.5), transparent),
        radial-gradient(1px 1px at 85% 15%, rgba(255,255,255,0.35), transparent),
        radial-gradient(1.5px 1.5px at 45% 45%, rgba(171,130,255,0.4), transparent),
        radial-gradient(1px 1px at 65% 35%, rgba(255,255,255,0.35), transparent),
        radial-gradient(2px 2px at 80% 55%, rgba(171,130,255,0.35), transparent);
}

/* Force stApp background transparent so overlays show */
.stApp {
    background: transparent !important;
}

/* ── Orbital trajectory + starfield SVG overlay ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none;
    z-index: 0;
    background-image: url('{{ORBITAL_BG}}');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

/* Ensure main content stays above the overlay */
.stApp > * {
    position: relative;
    z-index: 1;
}

/* Edge coordinate markers */
.edge-coord {
    position: fixed;
    font-family: var(--font-mono);
    font-size: 7px;
    letter-spacing: 1px;
    color: rgba(79,195,247,0.06);
    pointer-events: none;
    z-index: -1;
    user-select: none;
}

.edge-coord.tl { top: 10px; left: 24px; }
.edge-coord.tr { top: 10px; right: 24px; color: rgba(171,130,255,0.06); }
.edge-coord.bl { bottom: 10px; left: 24px; }
.edge-coord.br { bottom: 10px; right: 24px; color: rgba(171,130,255,0.06); }

/* ── Orbitron headers with glow ── */
h1, h2, h3 {
    font-family: var(--font-display) !important;
    letter-spacing: 1.5px;
}

h1::after {
    content: '';
    display: block;
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, var(--cyan), var(--purple));
    box-shadow: 0 0 8px var(--glow-cyan);
    margin-top: 8px;
}

/* ── Body text ── */
p, li, span, div {
    font-family: var(--font-body);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c14 0%, #060910 100%) !important;
    border-right: 1px solid var(--border-cyan);
    box-shadow: 1px 0 20px rgba(79,195,247,0.05);
}

/* Sidebar nav items – command prompt style */
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] span {
    font-family: var(--font-mono) !important;
    letter-spacing: 1px;
    font-size: 0.9em !important;
}

/* ── Pulsing status dots ── */
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 4px rgba(102,187,106,0.4); }
    50% { box-shadow: 0 0 12px rgba(102,187,106,0.8), 0 0 20px rgba(102,187,106,0.3); }
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 4px rgba(239,83,80,0.4); }
    50% { box-shadow: 0 0 12px rgba(239,83,80,0.8), 0 0 20px rgba(239,83,80,0.3); }
}
@keyframes pulse-amber {
    0%, 100% { box-shadow: 0 0 4px rgba(255,167,38,0.4); }
    50% { box-shadow: 0 0 12px rgba(255,167,38,0.8), 0 0 20px rgba(255,167,38,0.3); }
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 10px;
    flex-shrink: 0;
}
.status-dot.green { background: #66BB6A; animation: pulse-green 2s ease-in-out infinite; }
.status-dot.red   { background: #EF5350; animation: pulse-red 1.5s ease-in-out infinite; }
.status-dot.amber { background: #FFA726; animation: pulse-amber 1.8s ease-in-out infinite; }

/* ── Status panel – telemetry style ── */
.status-panel {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border-cyan);
    border-radius: 4px;
    padding: 14px;
    margin: 8px 0;
    position: relative;
}

.status-panel::before {
    content: '//SYS.TELEMETRY';
    position: absolute;
    top: -8px;
    right: 12px;
    font-family: var(--font-mono);
    font-size: 0.55em;
    color: rgba(79,195,247,0.3);
    letter-spacing: 1px;
}

.status-panel-title {
    font-family: var(--font-mono);
    font-size: 0.72em;
    color: var(--cyan);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 12px;
    text-shadow: 0 0 8px var(--glow-cyan);
}

.status-row {
    display: flex;
    align-items: center;
    padding: 5px 0;
    font-size: 0.82em;
    font-family: var(--font-mono);
    border-bottom: 1px solid rgba(79,195,247,0.05);
}
.status-row:last-child { border-bottom: none; }

.status-label {
    color: var(--text-muted);
    font-size: 0.9em;
}

.status-value {
    margin-left: auto;
    font-weight: 500;
    letter-spacing: 1px;
    font-size: 0.85em;
}

.status-value.online  { color: #66BB6A; text-shadow: 0 0 6px rgba(102,187,106,0.4); }
.status-value.offline { color: #EF5350; text-shadow: 0 0 6px rgba(239,83,80,0.4); }
.status-value.degraded { color: #FFA726; text-shadow: 0 0 6px rgba(255,167,38,0.4); }

/* ── Sidebar brand – minimal crosshair + acronym ── */
.sidebar-brand {
    text-align: center;
    padding: 10px 0 6px;
    position: relative;
}

.sidebar-brand-mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 44px; height: 44px;
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 2px;
    position: relative;
    margin-bottom: 8px;
}

/* Crosshair corners on the brand mark */
.sidebar-brand-mark::before,
.sidebar-brand-mark::after {
    content: '';
    position: absolute;
    width: 10px; height: 10px;
}
.sidebar-brand-mark::before {
    top: -2px; left: -2px;
    border-top: 2px solid var(--cyan);
    border-left: 2px solid var(--cyan);
}
.sidebar-brand-mark::after {
    bottom: -2px; right: -2px;
    border-bottom: 2px solid var(--purple);
    border-right: 2px solid var(--purple);
}

.sidebar-brand-acronym {
    font-family: var(--font-display);
    font-size: 0.85em;
    font-weight: 700;
    color: var(--cyan);
    letter-spacing: 2px;
    text-shadow: 0 0 10px var(--glow-cyan);
}

.sidebar-brand-sub {
    font-family: var(--font-mono);
    font-size: 0.58em;
    color: rgba(79,195,247,0.35);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── Version badge – neon outline + hover glow ── */
.version-badge {
    display: inline-block;
    background: rgba(79,195,247,0.06);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 2px;
    padding: 3px 14px;
    font-size: 0.68em;
    color: var(--cyan);
    letter-spacing: 2px;
    font-family: var(--font-mono);
    box-shadow: 0 0 6px rgba(79,195,247,0.15), inset 0 0 6px rgba(79,195,247,0.05);
    transition: box-shadow 0.3s ease, border-color 0.3s ease;
}

.version-badge:hover {
    border-color: rgba(0,212,255,0.6);
    box-shadow: 0 0 14px rgba(79,195,247,0.35), inset 0 0 10px rgba(79,195,247,0.08);
}

/* ── Hero ── */
.hero-container {
    text-align: center;
    padding: 50px 20px 36px;
    position: relative;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: 30px; left: 50%;
    transform: translateX(-50%);
    width: 500px; height: 80px;
    background: radial-gradient(ellipse, rgba(79,195,247,0.08) 0%, transparent 70%);
    pointer-events: none;
}

.hero-title {
    font-family: var(--font-display) !important;
    font-size: 2.8em;
    font-weight: 900;
    background: linear-gradient(135deg, var(--cyan) 0%, var(--purple) 50%, var(--cyan) 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient-shift 4s ease infinite;
    margin-bottom: 8px;
    filter: drop-shadow(0 0 20px var(--glow-cyan));
}

.hero-subtitle {
    font-family: var(--font-mono);
    font-size: 1em;
    color: var(--text-muted);
    margin-bottom: 40px;
    letter-spacing: 4px;
    text-transform: uppercase;
}

@keyframes gradient-shift {
    0% { background-position: 0% center; }
    50% { background-position: 100% center; }
    100% { background-position: 0% center; }
}

/* ── Dashboard frame – outer bounding box ── */
.dashboard-frame {
    border: 1px solid rgba(0,212,255,0.1);
    border-radius: 4px;
    padding: 28px 20px 24px;
    position: relative;
    margin: 0 8px;
}

/* Corner accents on the dashboard frame */
.dashboard-frame::before,
.dashboard-frame::after {
    content: '';
    position: absolute;
    width: 24px; height: 24px;
    border-color: rgba(0,212,255,0.3);
    border-style: solid;
}
.dashboard-frame::before {
    top: -1px; left: -1px;
    border-width: 2px 0 0 2px;
}
.dashboard-frame::after {
    bottom: -1px; right: -1px;
    border-width: 0 2px 2px 0;
}

/* Extra corners via inner spans */
.dashboard-corner-tr {
    position: absolute;
    top: -1px; right: -1px;
    width: 24px; height: 24px;
    border-top: 2px solid rgba(171,130,255,0.3);
    border-right: 2px solid rgba(171,130,255,0.3);
}
.dashboard-corner-bl {
    position: absolute;
    bottom: -1px; left: -1px;
    width: 24px; height: 24px;
    border-bottom: 2px solid rgba(171,130,255,0.3);
    border-left: 2px solid rgba(171,130,255,0.3);
}

/* Coordinate stamps */
.coord-stamp {
    font-family: var(--font-mono);
    font-size: 0.5em;
    color: rgba(79,195,247,0.2);
    letter-spacing: 1px;
    position: absolute;
}
.coord-stamp.top-left  { top: 6px; left: 28px; }
.coord-stamp.bot-right { bottom: 6px; right: 28px; }

/* Section label inside frame */
.frame-label {
    font-family: var(--font-mono);
    font-size: 0.62em;
    color: rgba(79,195,247,0.25);
    letter-spacing: 2px;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 18px;
}

/* ── Feature cards – glassmorphism + segmented neon brackets ── */
.feature-card {
    background: linear-gradient(165deg, rgba(14,26,48,0.75) 0%, rgba(8,16,32,0.6) 100%);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--border-cyan);
    border-radius: 4px;
    padding: 32px 24px 28px;
    text-align: center;
    transition: all 0.35s ease;
    height: 100%;
    position: relative;
    overflow: hidden;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}

/* Segmented neon corner brackets */
.feature-card::before,
.feature-card::after {
    content: '';
    position: absolute;
    width: 20px; height: 20px;
    border-color: var(--cyan-bright);
    border-style: solid;
    opacity: 0.25;
    transition: opacity 0.35s ease, border-color 0.35s ease;
}
.feature-card::before {
    top: 6px; left: 6px;
    border-width: 1.5px 0 0 1.5px;
}
.feature-card::after {
    bottom: 6px; right: 6px;
    border-width: 0 1.5px 1.5px 0;
}

/* Extra corners via inner spans */
.feature-card .corner-tr {
    position: absolute;
    top: 6px; right: 6px;
    width: 20px; height: 20px;
    border-top: 1.5px solid var(--purple);
    border-right: 1.5px solid var(--purple);
    opacity: 0.2;
    transition: opacity 0.35s ease;
}
.feature-card .corner-bl {
    position: absolute;
    bottom: 6px; left: 6px;
    width: 20px; height: 20px;
    border-bottom: 1.5px solid var(--purple);
    border-left: 1.5px solid var(--purple);
    opacity: 0.2;
    transition: opacity 0.35s ease;
}

.feature-card:hover {
    border-color: rgba(0,212,255,0.45);
    box-shadow:
        0 0 30px rgba(79,195,247,0.12),
        inset 0 0 30px rgba(79,195,247,0.03);
    transform: translateY(-3px);
    background: linear-gradient(165deg, rgba(18,32,56,0.8) 0%, rgba(10,20,38,0.7) 100%);
}
.feature-card:hover::before,
.feature-card:hover::after {
    opacity: 0.7;
    border-color: var(--cyan-bright);
}
.feature-card:hover .corner-tr,
.feature-card:hover .corner-bl {
    opacity: 0.6;
}

.feature-icon {
    margin-bottom: 14px;
    filter: drop-shadow(0 0 10px var(--glow-cyan));
}

.feature-title {
    font-family: var(--font-display);
    font-size: 1em;
    font-weight: 600;
    color: var(--cyan);
    margin-bottom: 10px;
    letter-spacing: 1px;
    text-shadow: 0 0 10px var(--glow-cyan);
}

.feature-desc {
    font-family: var(--font-body);
    font-size: 0.92em;
    color: var(--text-muted);
    line-height: 1.6;
}

/* Micro-data serial – small, monospace, low opacity, top-right */
.feature-card-serial {
    position: absolute;
    top: 10px;
    right: 14px;
    font-family: var(--font-mono);
    font-size: 0.48em;
    color: rgba(79,195,247,0.18);
    letter-spacing: 1.5px;
}

/* ── Page header ── */
.page-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 24px;
}

.page-header-icon {
    filter: drop-shadow(0 0 10px var(--glow-cyan));
}

.page-header-text h1 {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 1.8em !important;
    text-shadow: 0 0 15px var(--glow-cyan);
}

.page-header-sub {
    font-family: var(--font-mono);
    color: var(--text-muted);
    font-size: 0.82em;
    margin: 4px 0 0 0;
    letter-spacing: 1px;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border-cyan);
    border-radius: 4px;
    padding: 16px 20px;
    position: relative;
}

div[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--cyan), transparent);
    border-radius: 4px 0 0 4px;
}

div[data-testid="stMetric"] label {
    font-family: var(--font-mono) !important;
    font-size: 0.68em !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted) !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    color: var(--cyan) !important;
    text-shadow: 0 0 8px var(--glow-cyan);
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    border: 1px solid var(--border-cyan) !important;
    border-radius: 4px !important;
    background: var(--bg-card) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

div[data-testid="stExpander"] summary {
    font-family: var(--font-mono);
    font-size: 0.88em;
    color: var(--cyan) !important;
    letter-spacing: 0.5px;
}

/* ── Metadata pills ── */
.chat-meta {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(79,195,247,0.06);
    border: 1px solid rgba(79,195,247,0.12);
    border-radius: 2px;
    padding: 3px 12px;
    font-family: var(--font-mono);
    font-size: 0.72em;
    color: var(--text-muted);
}

/* ── Documents table ── */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-cyan) !important;
    border-radius: 4px !important;
    overflow: hidden;
}

/* ── Pagination ── */
button[kind="secondary"] {
    border-color: rgba(79,195,247,0.3) !important;
    font-family: var(--font-mono) !important;
    letter-spacing: 1px;
}

/* ── Chat bubble layout (applied by JS) ── */
.chat-bubble-user {
    flex-direction: row-reverse !important;
    margin-left: auto !important;
    max-width: 75% !important;
    width: fit-content !important;
    border-radius: 4px !important;
    border: 1px solid var(--border-cyan) !important;
    background: rgba(79,195,247,0.03) !important;
}

.chat-bubble-assistant {
    margin-right: auto !important;
    max-width: 85% !important;
    width: fit-content !important;
    border-radius: 4px !important;
    border: 1px solid var(--border-purple) !important;
    background: var(--bg-card) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

/* ── Scanline overlay on hero (subtle) ── */
@keyframes scanline {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
}

.hero-container::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(79,195,247,0.15), transparent);
    animation: scanline 8s linear infinite;
    pointer-events: none;
}

/* ── Chat input ── */
div[data-testid="stChatInput"] {
    border: 1px solid var(--border-cyan) !important;
    border-radius: 4px !important;
}

/* ── Buttons ── */
button[data-testid="stBaseButton-secondary"] {
    font-family: var(--font-mono) !important;
    letter-spacing: 1px;
    border-radius: 3px !important;
}
</style>
"""

_JS_COMPONENT = """
<script>
(function() {
    const root = window.parent.document;

    function styleChatBubbles() {
        const messages = root.querySelectorAll('[data-testid="stChatMessage"]');
        messages.forEach(function(msg) {
            if (msg.classList.contains('chat-styled')) return;
            msg.classList.add('chat-styled');
            var isUser = msg.querySelector('[data-testid="chatAvatarIcon-user"]')
                      || msg.innerHTML.indexOf('chatAvatarIcon-user') !== -1;
            if (!isUser) {
                var imgs = msg.querySelectorAll('img');
                imgs.forEach(function(img) {
                    if (img.alt && img.alt.toLowerCase().indexOf('user') !== -1) isUser = true;
                });
            }
            if (isUser) {
                msg.classList.add('chat-bubble-user');
            } else {
                msg.classList.add('chat-bubble-assistant');
            }
        });
    }

    var observer = new MutationObserver(function() {
        setTimeout(styleChatBubbles, 100);
    });
    observer.observe(root.body, { childList: true, subtree: true });
    styleChatBubbles();
    setTimeout(styleChatBubbles, 500);
    setTimeout(styleChatBubbles, 1500);
})();
</script>
"""


_EDGE_COORDS = """
<div class="edge-coord tl">+47.3769 // -122.3420</div>
<div class="edge-coord tr">ALT 408.2km // INC 51.64°</div>
<div class="edge-coord bl">EPOCH 2026.176</div>
<div class="edge-coord br">TLM // NOMINAL</div>
"""


def inject_css() -> None:
    css = _CSS_TEMPLATE.replace("{{ORBITAL_BG}}", _ORBITAL_BG_URI)
    st.markdown(css + _EDGE_COORDS, unsafe_allow_html=True)


def inject_chat_js() -> None:
    try:
        st.html(_JS_COMPONENT)
    except AttributeError:
        _components.html(_JS_COMPONENT, height=0)
