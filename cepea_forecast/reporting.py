from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from cepea_forecast.forecasting import ForecastBundle

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
BLUE = colors.HexColor("#204a87")
RED = colors.HexColor("#c73e1d")
ORANGE = colors.HexColor("#f4a261")
GREEN = colors.HexColor("#2e7d32")
GRAY = colors.HexColor("#6c757d")
LIGHT_BG = colors.HexColor("#f7fafc")
BORDER = colors.HexColor("#d0d7de")
CARD_BG = colors.HexColor("#f0f4f8")

_BLUE_HEX = "#204a87"
_RED_HEX = "#c73e1d"
_ORANGE_HEX = "#f4a261"
_GREEN_HEX = "#2e7d32"
_GRAY_HEX = "#888888"

_TABLE_STYLE = [
    ("BACKGROUND", (0, 0), (-1, 0), BLUE),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("LEADING", (0, 0), (-1, -1), 10),
    ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, LIGHT_BG]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_json_loads(value: str, default=None):
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else value


def _color_for_change(value: float) -> colors.Color:
    return GREEN if value >= 0 else RED


def _compute_rsi(history: pd.Series, period: int = 14) -> float:
    delta = history.diff()
    gain = delta.clip(lower=0).ewm(span=period, min_periods=1, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(span=period, min_periods=1, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    last = rsi.iloc[-1]
    if pd.isna(last):
        return 50.0
    return float(last)


def _categorize_past_covariates(covariates: list[str]) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {}
    for c in covariates:
        if any(k in c for k in ("bezerro", "ratio_momentum", "ratio_range")):
            cat = "Cross-Asset (Bezerro)"
        elif any(k in c for k in ("usd", "fx_adjusted", "boi_usd")):
            cat = "Cross-Asset (FX)"
        elif any(k in c for k in ("rolling_mean", "ma_ratio")):
            cat = "Trend"
        elif any(k in c for k in ("momentum", "acceleration", "divergence")):
            cat = "Momentum"
        elif any(k in c for k in ("vol", "log_return")):
            cat = "Volatility"
        elif any(k in c for k in ("rsi", "range_position", "yoy")):
            cat = "Oscillators & Regime"
        else:
            cat = "Other"
        categories.setdefault(cat, []).append(c)
    return categories


# ---------------------------------------------------------------------------
# Page 1: Executive Summary
# ---------------------------------------------------------------------------

def _build_executive_summary(
    bundle: ForecastBundle, source_file: Path, styles: dict
) -> list:
    body_style = styles["body"]
    meta = bundle.metadata
    history = bundle.history
    forecast = bundle.forecast_frame
    last_price = float(history.iloc[-1])
    p50_end = float(forecast["0.5"].iloc[-1])
    p10_end = float(forecast["0.1"].iloc[-1])
    p90_end = float(forecast["0.9"].iloc[-1])
    change_pct = (p50_end / last_price - 1) * 100
    ci_width_end = p90_end - p10_end

    # Header bar
    header = Table(
        [[Paragraph(
            "CEPEA BOI GORDO &mdash; Forecast Report<br/>"
            f"<font size=9>{datetime.now().strftime('%d %B %Y, %H:%M')}</font>",
            ParagraphStyle("HeaderTitle", parent=styles["title"], textColor=colors.white, fontSize=18, leading=22),
        )]],
        colWidths=[10.5 * inch],
        rowHeights=[50],
    )
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BLUE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
    ]))

    # Metric cards
    def _card(label: str, value: str, color=BLUE):
        return Paragraph(
            f"<font size=7 color='#6c757d'>{label}</font><br/>"
            f"<font size=14 color='{color}'><b>{value}</b></font>",
            ParagraphStyle("Card", alignment=1, leading=18),
        )

    change_color = _GREEN_HEX if change_pct >= 0 else _RED_HEX
    cards = Table(
        [[
            _card("Current Price (R$/@)", f"{last_price:,.2f}"),
            _card("52-Week Forecast (P50)", f"R$ {p50_end:,.2f}"),
            _card("Expected Change", f"{change_pct:+.1f}%", change_color),
            _card("CI Width (Step 52)", f"R$ {ci_width_end:,.2f}"),
        ]],
        colWidths=[2.5 * inch] * 4,
        rowHeights=[55],
    )
    cards.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
        ("BOX", (0, 0), (0, 0), 1, BLUE),
        ("BOX", (1, 0), (1, 0), 1, BLUE),
        ("BOX", (2, 0), (2, 0), 1, BLUE),
        ("BOX", (3, 0), (3, 0), 1, BLUE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    # Data + Model overview
    history_years = len(history) / 52
    known_count = len(_safe_json_loads(meta.get("known_covariates", "[]"), []))
    past_count = meta.get("covariate_count", "0")

    data_info = Paragraph(
        f"<b><font color='{_BLUE_HEX}'>Data Overview</font></b><br/>"
        f"Source: {source_file.name}<br/>"
        f"Target: {meta.get('target_column', 'Target')}<br/>"
        f"Range: {history.index.min().strftime('%d %b %Y')} to {history.index.max().strftime('%d %b %Y')}<br/>"
        f"History: {len(history):,} weeks ({history_years:.1f} years)<br/>"
        f"Aggregation: Weekly mean (W-FRI)",
        body_style,
    )
    model_info = Paragraph(
        f"<b><font color='{_BLUE_HEX}'>Model Overview</font></b><br/>"
        f"Preset: {meta.get('preset', 'N/A')}<br/>"
        f"Horizon: {bundle.spec.prediction_length} weeks<br/>"
        f"Validation windows: {meta.get('num_val_windows', 'N/A')}<br/>"
        f"Eval metric: WQL (Weighted Quantile Loss)<br/>"
        f"Known covariates: {known_count} | Past covariates: {past_count}",
        body_style,
    )
    overview = Table([[data_info, model_info]], colWidths=[5.0 * inch, 5.0 * inch])
    overview.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))

    # Summary table
    trained_at = meta.get("model_trained_at", "N/A")
    try:
        trained_at = pd.Timestamp(trained_at).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass

    summary_data = [
        ["Model", "Granularity", "Periods", "History End", "Forecast End", "Covariates", "Trained At"],
        [
            bundle.spec.model_id,
            bundle.spec.granularity,
            str(bundle.spec.prediction_length),
            history.index.max().date().isoformat(),
            forecast["timestamp"].max().date().isoformat(),
            past_count,
            trained_at,
        ],
    ]
    summary_table = Table(summary_data, repeatRows=1)
    summary_table.setStyle(TableStyle(_TABLE_STYLE + [("ALIGN", (2, 1), (-1, -1), "CENTER")]))

    # Model count
    model_family = _safe_json_loads(meta.get("model_family", ""), [])
    n_models = len(model_family) if isinstance(model_family, list) else "preset default"

    footer_text = Paragraph(
        f"<i><font size=7 color='#6c757d'>"
        f"Based on {history_years:.0f} years of weekly CEPEA data, "
        f"using {n_models} model families with {known_count + int(past_count)} features "
        f"({known_count} known + {past_count} past covariates)."
        f"</font></i>",
        body_style,
    )

    return [
        header, Spacer(1, 0.2 * inch),
        cards, Spacer(1, 0.2 * inch),
        overview, Spacer(1, 0.15 * inch),
        summary_table, Spacer(1, 0.1 * inch),
        footer_text,
        PageBreak(),
    ]


# ---------------------------------------------------------------------------
# Page 2: Model & Variables
# ---------------------------------------------------------------------------

def _build_model_variables_page(bundle: ForecastBundle, styles: dict) -> list:
    subtitle_style = styles["subtitle"]
    body_style = styles["body"]
    meta = bundle.metadata

    elements: list = [Paragraph("Model &amp; Variables", subtitle_style)]

    # Training configuration
    elements.append(Paragraph("<b>Training Configuration</b>", body_style))
    time_limit = meta.get("time_limit", "unlimited")
    time_limit_str = "Unlimited" if time_limit == "unlimited" else f"{time_limit}s"
    config_data = [
        ["Parameter", "Value"],
        ["Preset", meta.get("preset", "N/A")],
        ["Time limit", time_limit_str],
        ["Validation windows", meta.get("num_val_windows", "N/A")],
        ["Seasonal period", f"{meta.get('seasonal_period', '52')} weeks"],
        ["Quantile levels", "10%, 50%, 90%"],
        ["Recalibration", "Per-horizon isotonic regression"],
    ]
    config_table = Table(config_data, colWidths=[2.5 * inch, 3.0 * inch])
    config_table.setStyle(TableStyle(_TABLE_STYLE))
    elements.extend([Spacer(1, 0.05 * inch), config_table, Spacer(1, 0.15 * inch)])

    # Model ensemble
    model_family = _safe_json_loads(meta.get("model_family", ""), [])
    if isinstance(model_family, list) and model_family:
        elements.append(Paragraph(f"<b>Model Ensemble ({len(model_family)} families)</b>", body_style))
        # Display as comma-separated in a paragraph
        model_text = ", ".join(model_family)
        elements.append(Paragraph(f"<font size=7>{model_text}</font>", body_style))
    else:
        elements.append(Paragraph(f"<b>Model family:</b> {meta.get('model_family', 'N/A')}", body_style))
    elements.append(Spacer(1, 0.15 * inch))

    # Known covariates
    known_list = _safe_json_loads(meta.get("known_covariates", "[]"), [])
    if known_list:
        elements.append(Paragraph(f"<b>Known Covariates ({len(known_list)} features)</b>", body_style))
        known_data = [["Feature", "Description"]]
        for f in known_list:
            if "sin_yearly" in f:
                k = f.split("_")[-1]
                desc = f"Yearly seasonality, sine harmonic {k}"
            elif "cos_yearly" in f:
                k = f.split("_")[-1]
                desc = f"Yearly seasonality, cosine harmonic {k}"
            else:
                desc = f
            known_data.append([f, desc])
        known_table = Table(known_data, colWidths=[2.0 * inch, 5.0 * inch])
        known_table.setStyle(TableStyle(_TABLE_STYLE + [("FONTSIZE", (0, 0), (-1, -1), 7)]))
        elements.extend([Spacer(1, 0.05 * inch), known_table])
        elements.append(Paragraph(
            "<i><font size=6 color='#6c757d'>Fourier features encode yearly seasonality as continuous signals. "
            "Because they are deterministic functions of the date, they are available for the full forecast horizon.</font></i>",
            body_style,
        ))
    elements.append(Spacer(1, 0.1 * inch))

    # Past covariates
    past_list = _safe_json_loads(meta.get("past_covariates", "[]"), [])
    if past_list:
        categories = _categorize_past_covariates(past_list)
        elements.append(Paragraph(f"<b>Past Covariates ({len(past_list)} features, history only)</b>", body_style))
        for cat, features in categories.items():
            elements.append(Paragraph(
                f"<font size=7><b>{cat}:</b> {', '.join(features)}</font>", body_style,
            ))
    elements.append(Spacer(1, 0.1 * inch))

    # Source data columns
    labels = _safe_json_loads(meta.get("covariate_labels", "{}"), {})
    if labels:
        elements.append(Paragraph("<b>Source Data Columns</b>", body_style))
        label_data = [["Internal Name", "Label"]]
        for internal, label in labels.items():
            label_data.append([internal, label])
        label_table = Table(label_data, colWidths=[2.0 * inch, 4.0 * inch])
        label_table.setStyle(TableStyle(_TABLE_STYLE + [("FONTSIZE", (0, 0), (-1, -1), 7)]))
        elements.extend([Spacer(1, 0.05 * inch), label_table])

    elements.append(PageBreak())
    return elements


# ---------------------------------------------------------------------------
# Page 3: Main forecast plot (improved)
# ---------------------------------------------------------------------------

def _build_main_plot(bundle: ForecastBundle, image_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    history = bundle.history
    forecast = bundle.forecast_frame

    # Show last 104 weeks (2 years) + forecast
    history_window = history.iloc[-104:]

    # 52-week rolling mean
    rolling_52 = history.rolling(52, min_periods=26).mean()
    rolling_window = rolling_52.iloc[-104:]

    # CI fill
    ax.fill_between(
        forecast["timestamp"], forecast["0.1"], forecast["0.9"],
        color=_ORANGE_HEX, alpha=0.18, label="80% CI (P10–P90)",
    )

    # Rolling mean
    ax.plot(rolling_window.index, rolling_window.values,
            color=_GRAY_HEX, linewidth=1.2, linestyle="--", alpha=0.7, label="52-week MA")

    # History
    ax.plot(history_window.index, history_window.values,
            color=_BLUE_HEX, linewidth=1.5, label="History")

    # Forecast P50
    ax.plot(forecast["timestamp"], forecast["0.5"],
            color=_RED_HEX, linewidth=2.0, label="Forecast (P50)")

    # Forecast mean (secondary)
    ax.plot(forecast["timestamp"], forecast["mean"],
            color=_RED_HEX, linewidth=1.0, linestyle=":", alpha=0.5, label="Forecast (mean)")

    # Vertical divider
    ax.axvline(forecast["timestamp"].iloc[0], color="#555555", linewidth=0.8, linestyle="--", alpha=0.5)

    # Annotations
    last_price = float(history.iloc[-1])
    last_date = history.index[-1]
    ax.annotate(f"R$ {last_price:,.2f}", xy=(last_date, last_price),
                xytext=(10, 12), textcoords="offset points", fontsize=8,
                color=_BLUE_HEX, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=_BLUE_HEX, lw=0.8))

    end_p50 = float(forecast["0.5"].iloc[-1])
    end_date = forecast["timestamp"].iloc[-1]
    ax.annotate(f"R$ {end_p50:,.2f}", xy=(end_date, end_p50),
                xytext=(10, -15), textcoords="offset points", fontsize=8,
                color=_RED_HEX, fontweight="bold")

    # RSI box
    rsi = _compute_rsi(history)
    rsi_color = _RED_HEX if rsi > 70 or rsi < 30 else _GREEN_HEX
    ax.text(0.98, 0.95, f"RSI(14): {rsi:.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color=rsi_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=_GRAY_HEX, alpha=0.85))

    # Axes
    ax.set_title(bundle.spec.title, fontsize=12, fontweight="bold", color=_BLUE_HEX)
    ax.set_ylabel(f"{bundle.metadata.get('target_column', 'Target')} (R$/@)", fontsize=9)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.legend(loc="upper left", frameon=True, fancybox=True, fontsize=7.5, framealpha=0.9)
    ax.margins(x=0.01)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(image_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 4: Confidence Interval Analysis
# ---------------------------------------------------------------------------

def _build_ci_analysis(
    bundle: ForecastBundle, image_dir: Path, styles: dict
) -> list:
    subtitle_style = styles["subtitle"]
    body_style = styles["body"]
    forecast = bundle.forecast_frame
    last_price = float(bundle.history.iloc[-1])

    elements: list = [Paragraph("Confidence Interval Analysis", subtitle_style)]

    # Horizon summary table
    key_steps = [1, 13, 26, 52]
    horizon_data = [["Horizon", "Step", "Period End", "Median (P50)", "Change %", "CI Width", "CI %"]]
    labels = ["1 week", "13 weeks", "26 weeks", "52 weeks"]
    for label, step in zip(labels, key_steps):
        idx = min(step - 1, len(forecast) - 1)
        row = forecast.iloc[idx]
        p50 = float(row["0.5"])
        p10 = float(row["0.1"])
        p90 = float(row["0.9"])
        change = (p50 / last_price - 1) * 100
        ci = p90 - p10
        ci_pct = (ci / p50) * 100 if p50 != 0 else 0
        horizon_data.append([
            label, str(step), str(row["timestamp"].date()),
            f"R$ {p50:,.2f}", f"{change:+.1f}%", f"R$ {ci:,.2f}", f"{ci_pct:.1f}%",
        ])

    horizon_table = Table(
        horizon_data, repeatRows=1,
        colWidths=[1.1 * inch, 0.6 * inch, 1.0 * inch, 1.1 * inch, 0.9 * inch, 1.1 * inch, 0.8 * inch],
    )
    style_cmds = list(_TABLE_STYLE) + [("ALIGN", (1, 1), (-1, -1), "CENTER")]
    # Color change % column
    for row_idx in range(1, len(horizon_data)):
        val = float(horizon_data[row_idx][4].replace("%", "").replace("+", ""))
        style_cmds.append(("TEXTCOLOR", (4, row_idx), (4, row_idx), _color_for_change(val)))
    horizon_table.setStyle(TableStyle(style_cmds))
    elements.extend([Spacer(1, 0.05 * inch), horizon_table, Spacer(1, 0.2 * inch)])

    # CI width evolution plot
    ci_plot_path = image_dir / "ci_evolution.png"
    steps = list(range(1, len(forecast) + 1))
    ci_widths = (forecast["0.9"] - forecast["0.1"]).values

    fig, ax = plt.subplots(figsize=(10.0, 3.2))
    ax.fill_between(steps, 0, ci_widths, color=_ORANGE_HEX, alpha=0.25)
    ax.plot(steps, ci_widths, color=_RED_HEX, linewidth=1.5)
    for s in key_steps:
        if s <= len(ci_widths):
            ax.axvline(s, color=_GRAY_HEX, linewidth=0.5, linestyle=":", alpha=0.5)
            ax.plot(s, ci_widths[s - 1], "o", color=_RED_HEX, markersize=5)
    ax.set_xlabel("Forecast Step", fontsize=9)
    ax.set_ylabel("CI Width (R$)", fontsize=9)
    ax.set_title("80% Confidence Interval Width by Horizon", fontsize=10, color=_BLUE_HEX)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(ci_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    elements.append(Image(str(ci_plot_path), width=9.5 * inch, height=3.0 * inch))
    elements.append(Spacer(1, 0.1 * inch))

    # Summary sentence
    p50_52 = float(forecast["0.5"].iloc[-1])
    p10_52 = float(forecast["0.1"].iloc[-1])
    p90_52 = float(forecast["0.9"].iloc[-1])
    change_52 = (p50_52 / last_price - 1) * 100
    elements.append(Paragraph(
        f"At step 52, the model expects the price to be <b>R$ {p50_52:,.2f}</b> "
        f"({change_52:+.1f}%), with 80% probability between "
        f"<b>R$ {p10_52:,.2f}</b> and <b>R$ {p90_52:,.2f}</b>.",
        body_style,
    ))

    elements.append(PageBreak())
    return elements


# ---------------------------------------------------------------------------
# Page 5+: Forecast data table (enhanced)
# ---------------------------------------------------------------------------

def _forecast_table_data(bundle: ForecastBundle) -> list[list[str]]:
    last_price = float(bundle.history.iloc[-1])
    rows = [["Step", "Period End", "Mean", "P10", "P50", "P90", "Change %"]]
    forecast_rows = bundle.rows.sort_values("step")
    for row in forecast_rows.to_dict(orient="records"):
        p50 = float(row["0.5"])
        change = (p50 / last_price - 1) * 100
        rows.append([
            str(row["step"]),
            str(row["target_period_end"]),
            f"R$ {float(row['mean']):,.2f}",
            f"R$ {float(row['0.1']):,.2f}",
            f"R$ {p50:,.2f}",
            f"R$ {float(row['0.9']):,.2f}",
            f"{change:+.1f}%",
        ])
    return rows


def _build_forecast_table(bundle: ForecastBundle) -> LongTable:
    data = _forecast_table_data(bundle)
    table = LongTable(
        data,
        repeatRows=1,
        colWidths=[0.55 * inch, 1.0 * inch, 1.05 * inch, 1.05 * inch, 1.05 * inch, 1.05 * inch, 0.8 * inch],
    )
    style_cmds = list(_TABLE_STYLE) + [
        ("ALIGN", (0, 1), (1, -1), "CENTER"),
        ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
    ]
    # Color change % column per row
    for row_idx in range(1, len(data)):
        val = float(data[row_idx][6].replace("%", "").replace("+", ""))
        style_cmds.append(("TEXTCOLOR", (6, row_idx), (6, row_idx), _color_for_change(val)))
    table.setStyle(TableStyle(style_cmds))
    return table


# ---------------------------------------------------------------------------
# Page footer
# ---------------------------------------------------------------------------

def _page_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(36, 18, f"CEPEA Boi Gordo Forecast Report — {datetime.now().strftime('%Y-%m-%d')}")
    canvas.drawRightString(doc.pagesize[0] - 36, 18, f"Page {doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_forecast_report(
    output_path: Path, bundles: list[ForecastBundle], source_file: Path
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = output_path.parent.parent.parent / "tmp" / "pdfs"
    tmp_root.mkdir(parents=True, exist_ok=True)

    raw_styles = getSampleStyleSheet()
    styles = {
        "title": raw_styles["Title"],
        "subtitle": ParagraphStyle(
            "Subtitle", parent=raw_styles["Heading2"],
            textColor=BLUE, spaceAfter=10,
        ),
        "body": ParagraphStyle(
            "Body", parent=raw_styles["BodyText"], leading=14, fontSize=9,
        ),
        "small": ParagraphStyle(
            "Small", parent=raw_styles["BodyText"], fontSize=7, leading=9,
        ),
    }

    story: list = []

    # Page 1: Executive Summary
    bundle = bundles[0]
    story.extend(_build_executive_summary(bundle, source_file, styles))

    # Page 2: Model & Variables
    story.extend(_build_model_variables_page(bundle, styles))

    with TemporaryDirectory(dir=tmp_root) as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        for index, bundle in enumerate(bundles, start=1):
            # Page 3: Main forecast plot
            image_path = tmp_dir_path / f"{bundle.spec.model_id}.png"
            _build_main_plot(bundle, image_path)

            last_price = float(bundle.history.iloc[-1])
            last_date = bundle.history.index.max().strftime("%d %b %Y")

            story.extend([
                Paragraph(bundle.spec.title, styles["subtitle"]),
                Paragraph(
                    f"Last 2 years of history + {bundle.spec.prediction_length}-week forecast. "
                    f"Gray dashed line is the 52-week moving average.",
                    styles["small"],
                ),
                Spacer(1, 0.1 * inch),
                Image(str(image_path), width=10.5 * inch, height=5.5 * inch),
                PageBreak(),
            ])

            # Page 4: CI analysis
            story.extend(_build_ci_analysis(bundle, tmp_dir_path, styles))

            # Page 5+: Forecast table
            story.extend([
                Paragraph(f"{bundle.spec.title} — Detailed Forecast Table", styles["subtitle"]),
                Paragraph(
                    f"All {bundle.spec.prediction_length} forecast steps. "
                    f"Change % is relative to the last observed value of R$ {last_price:,.2f} on {last_date}.",
                    styles["body"],
                ),
                Spacer(1, 0.1 * inch),
                _build_forecast_table(bundle),
            ])
            if index < len(bundles):
                story.append(PageBreak())

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=landscape(A4),
            leftMargin=36, rightMargin=36,
            topMargin=36, bottomMargin=28,
        )
        doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)

    return output_path
