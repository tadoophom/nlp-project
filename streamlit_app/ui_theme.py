"""Shared Streamlit theming helpers for a cohesive UI."""

from __future__ import annotations

from contextlib import contextmanager
from textwrap import dedent
from typing import Iterable, Mapping

import streamlit as st
from streamlit import components


_BASE_CSS = """
<style>
:root {
    --brand-primary: #1f2937;
    --brand-accent: #2563eb;
    --surface: #ffffff;
    --surface-muted: #f3f4f6;
    --surface-border: rgba(15, 23, 42, 0.08);
    --text-strong: #111827;
    --text-muted: #6b7280;
    --radius-lg: 16px;
    --radius-md: 12px;
    --shadow-sm: 0 10px 30px rgba(15, 23, 42, 0.06);
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 0% 0%, rgba(37, 99, 235, 0.04), transparent 55%),
                #f8fafc;
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] > div {
    background: #f8fafc;
    color: #1f2937;
    border-right: 1px solid rgba(15, 23, 42, 0.08);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #1f2937 !important;
}

.themed-hero {
    background: var(--surface);
    border: 1px solid var(--surface-border);
    border-radius: var(--radius-lg);
    padding: 26px 30px;
    box-shadow: var(--shadow-sm);
    margin-bottom: 1.5rem;
    display: flex;
    gap: 22px;
    align-items: flex-start;
}

.themed-hero .hero-copy {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
}

.themed-hero-icon {
    width: 54px;
    height: 54px;
    border-radius: 14px;
    background: rgba(37, 99, 235, 0.12);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    color: var(--brand-accent);
}

.themed-hero h1 {
    font-size: 1.9rem;
    line-height: 1.2;
    margin: 0 0 0.35rem 0;
    color: var(--brand-primary);
}

.themed-hero p {
    margin: 0;
    font-size: 0.95rem;
    color: var(--text-muted);
}

.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 1.05rem;
}

.pill {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: var(--brand-primary);
    background: rgba(37, 99, 235, 0.08);
    border: 1px solid rgba(37, 99, 235, 0.12);
}

.stat-card {
    background: #ffffff;
    border-radius: var(--radius-md);
    border: 1px solid rgba(15, 23, 42, 0.1);
    padding: 18px 20px;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
    min-height: 118px;
}

.stat-card span.value {
    display: block;
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--brand-accent);
}

.stat-card .description {
    color: var(--text-muted);
    font-size: 0.82rem;
    margin-top: 6px;
    line-height: 1.35;
}

.section-wrap {
    background: #ffffff;
    border-radius: var(--radius-lg);
    border: 1px solid rgba(15, 23, 42, 0.08);
    padding: 24px 26px;
    margin-bottom: 1.4rem;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}

.section-wrap h2 {
    color: var(--brand-primary);
    font-size: 1.25rem;
    margin-bottom: 0.2rem;
}

.section-wrap p.section-subtitle {
    color: var(--text-muted);
    margin-bottom: 1.2rem;
    font-size: 0.92rem;
}

.section-divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(17, 24, 39, 0.15), transparent);
    margin: 1.4rem 0 1.1rem;
}
</style>
"""


def apply_theme() -> None:
    """Inject shared CSS once per session."""
    st.markdown(_BASE_CSS, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str, *, tags: Iterable[str] | None = None, icon: str | None = None) -> None:
    """Display a compact hero header at the top of the page."""

    icon_html = f"<div class='themed-hero-icon'>{icon}</div>" if icon else ""
    tags_html = (
        "<div class='hero-tags'>"
        + "".join(f"<span class='pill'>{pill}</span>" for pill in tags or [])
        + "</div>"
        if tags
        else ""
    )
    markup = dedent(
        """
        <div class='themed-hero'>
            {icon_html}
            <div class='hero-copy'>
                <h1>{title}</h1>
                <p>{subtitle}</p>
                {tags_html}
            </div>
        </div>
        """
    ).format(icon_html=icon_html, title=title, subtitle=subtitle, tags_html=tags_html)
    components.v1.html(markup, height=150, scrolling=False)


def render_stat_cards(cards: Iterable[Mapping[str, str | int | float]]) -> None:
    """Render a row of statistic cards."""

    cards = list(cards)
    if not cards:
        return
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        label = str(card.get("label", ""))
        value = card.get("value", "—")
        description = str(card.get("description", ""))
        col.markdown(
            """
            <div class='stat-card'>
                <span class='value'>{value}</span>
                <div style='font-weight:600;color:var(--text-strong);margin-bottom:4px;'>{label}</div>
                <div class='description'>{description}</div>
            </div>
            """.format(value=value, label=label, description=description),
            unsafe_allow_html=True,
        )


@contextmanager
def section(title: str, subtitle: str | None = None):
    """Context manager that wraps Streamlit elements in a styled surface."""

    st.markdown(
        "<div class='section-wrap'>" +
        f"<h2>{title}</h2>" +
        (f"<p class='section-subtitle'>{subtitle}</p>" if subtitle else ""),
        unsafe_allow_html=True,
    )
    container = st.container()
    with container:
        yield
    st.markdown("</div>", unsafe_allow_html=True)


def divider() -> None:
    """Styled horizontal divider."""

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
