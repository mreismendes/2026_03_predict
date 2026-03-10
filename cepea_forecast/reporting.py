from __future__ import annotations

from datetime import datetime
from tempfile import TemporaryDirectory
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, LongTable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from cepea_forecast.forecasting import ForecastBundle


def _summary_table_data(bundles: list[ForecastBundle]) -> list[list[str]]:
    rows = [["Model", "Granularity", "Periods", "History End", "Forecast End", "Covariates"]]
    for bundle in bundles:
        rows.append(
            [
                bundle.spec.model_id,
                bundle.spec.granularity,
                str(bundle.spec.prediction_length),
                bundle.history.index.max().date().isoformat(),
                bundle.forecast_frame["timestamp"].max().date().isoformat(),
                bundle.metadata.get("covariate_count", "0"),
            ]
        )
    return rows


def _build_plot(bundle: ForecastBundle, image_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    history = bundle.history
    forecast = bundle.forecast_frame

    ax.plot(history.index, history.values, color="#204a87", linewidth=1.5, label="History")
    ax.plot(forecast["timestamp"], forecast["mean"], color="#c73e1d", linewidth=2.1, label="Forecast mean")
    ax.fill_between(
        forecast["timestamp"],
        forecast["0.1"],
        forecast["0.9"],
        color="#f4a261",
        alpha=0.24,
        label="80% interval",
    )
    ax.axvline(forecast["timestamp"].iloc[0], color="#555555", linewidth=1.0, linestyle="--", label="Forecast start")
    ax.set_title(bundle.spec.title)
    ax.set_ylabel(bundle.metadata.get("target_column", "Target"))
    ax.grid(True, alpha=0.2, linewidth=0.8)
    ax.legend(loc="upper left", frameon=False)
    ax.margins(x=0.01)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _forecast_table_data(bundle: ForecastBundle) -> list[list[str]]:
    rows = [["Step", "Period End", "Mean", "P10", "P50", "P90"]]
    forecast_rows = bundle.rows.sort_values("step")
    for row in forecast_rows.to_dict(orient="records"):
        rows.append(
            [
                str(row["step"]),
                str(row["target_period_end"]),
                f"{float(row['mean']):.4f}",
                f"{float(row['0.1']):.4f}",
                f"{float(row['0.5']):.4f}",
                f"{float(row['0.9']):.4f}",
            ]
        )
    return rows


def _build_forecast_table(bundle: ForecastBundle) -> LongTable:
    table = LongTable(
        _forecast_table_data(bundle),
        repeatRows=1,
        colWidths=[0.7 * inch, 1.3 * inch, 1.15 * inch, 1.15 * inch, 1.15 * inch, 1.15 * inch],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#204a87")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d0d7de")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f7fafc")]),
                ("ALIGN", (0, 1), (1, -1), "CENTER"),
                ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _page_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(doc.pagesize[0] - 36, 18, f"Page {doc.page}")
    canvas.restoreState()


def generate_forecast_report(output_path: Path, bundles: list[ForecastBundle], source_file: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = output_path.parent.parent.parent / "tmp" / "pdfs"
    tmp_root.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    body_style = styles["BodyText"]
    body_style.leading = 15
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#204a87"),
        spaceAfter=10,
    )

    story = [
        Paragraph("CEPEA Boi Gordo Forecast Report", title_style),
        Spacer(1, 0.2 * inch),
        Paragraph(
            (
                f"Source file: {source_file.name}<br/>"
                f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
                f"Target series: {bundles[0].metadata.get('target_column', 'Target')}<br/>"
                f"Past covariates available: {bundles[0].metadata.get('covariate_count', '0')}<br/>"
                "Aggregation: closed weekly mean (W-FRI) and closed monthly mean for numeric columns."
            ),
            body_style,
        ),
        Spacer(1, 0.2 * inch),
    ]

    summary_table = Table(_summary_table_data(bundles), repeatRows=1)
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#204a87")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7de")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f7fafc")]),
                ("ALIGN", (2, 1), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(summary_table)
    story.append(PageBreak())

    with TemporaryDirectory(dir=tmp_root) as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        for index, bundle in enumerate(bundles, start=1):
            image_path = tmp_dir_path / f"{bundle.spec.model_id}.png"
            _build_plot(bundle, image_path)
            story.extend(
                [
                    Paragraph(bundle.spec.title, subtitle_style),
                    Paragraph(
                        (
                            f"Model ID: {bundle.spec.model_id}<br/>"
                            f"Model family: {bundle.metadata.get('model_family', '')}<br/>"
                            f"Prediction length: {bundle.spec.prediction_length}<br/>"
                            f"Past covariates used: {bundle.metadata.get('covariate_count', '0')}<br/>"
                            f"Last observed period: {bundle.history.index.max().date().isoformat()}<br/>"
                            f"Forecast through: {bundle.forecast_frame['timestamp'].max().date().isoformat()}"
                        ),
                        body_style,
                    ),
                    Spacer(1, 0.15 * inch),
                    Image(str(image_path), width=10.1 * inch, height=5.3 * inch),
                    PageBreak(),
                    Paragraph(f"{bundle.spec.title} forecast table", subtitle_style),
                    Paragraph(
                        (
                            f"Forecast values for the complete {bundle.spec.prediction_length}-step horizon.<br/>"
                            f"Columns show the mean forecast and the 10th, 50th, and 90th percentiles."
                        ),
                        body_style,
                    ),
                    Spacer(1, 0.12 * inch),
                    _build_forecast_table(bundle),
                ]
            )
            if index < len(bundles):
                story.append(PageBreak())

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=landscape(A4),
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=28,
        )
        doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)

    return output_path
