# app.py — FINAL VERSION (Control Charts + Scrap Forecast)
import io
import os
import time
import sys
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, request, Response, jsonify

app = Flask(__name__)

# Optional API key
API_KEY = os.getenv("API_KEY", None)

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def control_chart_figure(series, dates, wc_name, figsize=(11, 6)):
    """Creates a control chart figure."""
    values = np.array(series, dtype=float)
    mean = float(np.mean(values)) if len(values) else 0.0
    std = float(np.std(values, ddof=0)) if len(values) else 0.0
    UCL = mean + 3 * std
    LCL = max(mean - 3 * std, 0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dates, values, marker="o", linestyle="-", label="Scrap Qty")
    ax.axhline(mean, color="green", linestyle="--", label=f"CL = {mean:.2f}")
    ax.axhline(UCL, color="red", linestyle="--", label=f"UCL = {UCL:.2f}")
    ax.axhline(LCL, color="orange", linestyle="--", label=f"LCL = {LCL:.2f}")

    ax.set_title(f"Control Chart – Work Center: {wc_name}", fontsize=14)
    ax.set_ylabel("Scrap Quantity")
    ax.set_xlabel("Posting Date")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def fig_to_base64_png(fig, dpi=120, pad_inches=0.3):
    """Converts a Matplotlib figure into a Base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def analyze_scrap_reasons(df):
    """
    3-month trend analysis + prediction for scrap reasons.
    Uses column 'Scrap Reason' (mapped from 'Scrap Description').
    """
    if "Scrap Reason" not in df.columns:
        return "<p><em>No Scrap Reason column found.</em></p>"

    df2 = df.copy()
    df2["Posting Date"] = pd.to_datetime(df2["Posting Date"], errors="coerce")
    df2 = df2.dropna(subset=["Posting Date"])
    df2["Scrap Reason"] = df2["Scrap Reason"].astype(str).str.strip().fillna("Unknown")

    df2["year_month"] = df2["Posting Date"].dt.to_period("M")

    monthly = (df2.groupby(["Scrap Reason", "year_month"])["Scrap Qty Breakup"]
               .sum()
               .reset_index())

    if monthly.empty:
        return "<p><em>No monthly scrap data found.</em></p>"

    months = sorted(monthly["year_month"].unique())
    if len(months) < 3:
        return "<p><em>Not enough data (need 3+ months).</em></p>"

    last3 = months[-3:]
    mlabs = [m.strftime("%Y-%m") for m in last3]

    pivot = monthly.pivot(index="Scrap Reason",
                          columns="year_month",
                          values="Scrap Qty Breakup").reindex(columns=last3,
                                                              fill_value=0)

    results = []
    for reason, row in pivot.iterrows():
        vals = [float(row[m]) for m in last3]       # list of 3 values
        x = np.array([0, 1, 2])
        y = np.array(vals)

        if np.all(y == y[0]):
            slope = 0.0
            intercept = float(y[0])
        else:
            slope, intercept = np.polyfit(x, y, 1)

        pred = max(slope * 3 + intercept, 0)
        increasing = (vals[0] < vals[1] < vals[2]) or (slope > 0.01 and vals[2] > vals[0])

        if increasing:
            results.append({
                "reason": reason,
                "m1": vals[0],
                "m2": vals[1],
                "m3": vals[2],
                "slope": slope,
                "prediction": pred
            })

    if not results:
        return "<p><strong>No increasing scrap reasons detected.</strong></p>"

    # Sort by slope descending
    results = sorted(results, key=lambda r: r["slope"], reverse=True)

    html = """
    <h2>Scrap Reasons Increasing (3-Month Trend + Forecast)</h2>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
    <tr>
        <th>Scrap Reason</th>
        <th>{}</th><th>{}</th><th>{}</th>
        <th>Slope</th>
        <th>Predicted Next Month</th>
    </tr>
    """.format(*mlabs)

    for r in results:
        html += f"""
        <tr>
            <td>{r['reason']}</td>
            <td align="right">{r['m1']:.2f}</td>
            <td align="right">{r['m2']:.2f}</td>
            <td align="right">{r['m3']:.2f}</td>
            <td align="right">{r['slope']:.3f}</td>
            <td align="right">{r['prediction']:.2f}</td>
        </tr>
        """

    html += "</table>"
    return html


# =====================================================================
# ROUTES
# =====================================================================

@app.route("/", methods=["GET"])
def home():
    return "Control Chart + Scrap Forecast API", 200


@app.route("/process", methods=["POST"])
def process():
    start = time.time()

    # API Key check
    if API_KEY:
        if request.headers.get("x-api-key") != API_KEY:
            return jsonify({"error": "Invalid API key"}), 401

    if "file" not in request.files:
        return jsonify({"error": "Upload Excel file using 'file' field"}), 400

    try:
        df = pd.read_excel(request.files["file"])
    except Exception as e:
        return jsonify({"error": f"Error reading Excel: {str(e)}"}), 400

    # Normalize columns
    df.columns = df.columns.astype(str).str.strip()

    # 🔥 Map "Scrap Description" → "Scrap Reason"
    if "Scrap Description" in df.columns:
        df["Scrap Reason"] = df["Scrap Description"]
    else:
        df["Scrap Reason"] = None  # safe fallback

    # Required columns
    required = ["Actual Work Center", "Posting Date", "Scrap Qty Breakup"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing required columns: {missing}"}), 400

    # Clean types
    df["Posting Date"] = pd.to_datetime(df["Posting Date"], errors="coerce")
    df = df.dropna(subset=["Posting Date"])
    df["Scrap Qty Breakup"] = pd.to_numeric(df["Scrap Qty Breakup"], errors="coerce").fillna(0)

    # Scrap analysis block
    scrap_block = analyze_scrap_reasons(df)

    # Build control charts
    charts_html = []
    for wc in df["Actual Work Center"].dropna().unique():
        sub = df[df["Actual Work Center"] == wc]
        if sub.empty:
            continue

        fig = control_chart_figure(sub["Scrap Qty Breakup"], sub["Posting Date"], wc)
        img = fig_to_base64_png(fig)

        charts_html.append(f"""
        <div class="chart-page">
            <h3>Work Center: {wc}</h3>
            <img class="chart-img" src="{img}" />
        </div>
        """)

    if not charts_html:
        return jsonify({"error": "No Work Center data found"}), 400

    # Final HTML
    html = f"""
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Control Charts + Scrap Forecast</title>
        <style>
            body {{ font-family: Arial; margin: 16px; }}
            .chart-page {{ page-break-after: always; margin-bottom: 25px; }}
            img.chart-img {{ max-width: 100%; height: auto; border: 1px solid #ccc; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ccc; padding: 6px; }}
        </style>
    </head>
    <body>
        <h1>Control Charts & Scrap Forecast</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        {scrap_block}

        {''.join(charts_html)}
    </body>
    </html>
    """

    elapsed = time.time() - start
    print(f"/process done in {elapsed:.2f}s", file=sys.stderr)

    return Response(html, mimetype="text/html")


# =====================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
