# app.py
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

# Optional API key protection: set environment variable API_KEY
API_KEY = os.getenv("API_KEY", None)

# -------------------- Helper: chart creation & conversion --------------------
def control_chart_figure(series, dates, wc_name, figsize=(11, 6)):
    """Return a matplotlib Figure for a control chart (Scrap Qty vs Posting Date)."""
    values = np.array(series, dtype=float)
    # compute CL, UCL, LCL
    mean = float(np.mean(values)) if len(values) else 0.0
    std = float(np.std(values, ddof=0)) if len(values) else 0.0
    UCL = mean + 3 * std
    LCL = max(mean - 3 * std, 0.0)

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
    """Convert a matplotlib Figure to a base64-encoded PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return "data:image/png;base64," + b64

# -------------------- Helper: scrap reason analysis & forecast --------------------
def analyze_scrap_reasons(df, qty_col="Scrap Qty Breakup", date_col="Posting Date", reason_col="Scrap Reason"):
    """
    Analyze scrap reasons over the last 3 months, flag increasing reasons, and predict next month.
    Returns HTML snippet (string).
    """
    # Defensive checks
    if reason_col not in df.columns or date_col not in df.columns or qty_col not in df.columns:
        return "<p><strong>Scrap Reason analysis:</strong> required columns missing.</p>"

    df_local = df.copy()
    # Ensure date col is datetime
    df_local[date_col] = pd.to_datetime(df_local[date_col], errors="coerce")
    df_local = df_local.dropna(subset=[date_col])
    df_local[reason_col] = df_local[reason_col].astype(str).str.strip().fillna("Unknown")
    # Period for grouping
    df_local["year_month"] = df_local[date_col].dt.to_period("M")
    monthly = (df_local
               .groupby([reason_col, "year_month"], observed=True)[qty_col]
               .sum()
               .reset_index())
    if monthly.empty:
        return "<p><strong>Scrap Reason analysis:</strong> no monthly data available.</p>"

    months_sorted = sorted(monthly["year_month"].unique())
    if len(months_sorted) < 3:
        return "<p><strong>Scrap Reason analysis:</strong> not enough months of data (need >= 3 months).</p>"

    last_three = months_sorted[-3:]  # ascending order
    pivot = (monthly[monthly["year_month"].isin(last_three)]
             .pivot(index=reason_col, columns="year_month", values=qty_col)
            )
    pivot = pivot.reindex(columns=last_three, fill_value=0)

    rows = []
    for reason, row in pivot.iterrows():
        vals = [float(row.get(m, 0.0) or 0.0) for m in last_three]  # [m-2, m-1, m]
        strictly_increasing = (vals[0] < vals[1] < vals[2])
        x = np.array([0.0, 1.0, 2.0])
        y = np.array(vals)
        if np.all(y == y[0]):
            slope = 0.0
            intercept = float(y[0])
        else:
            coeffs = np.polyfit(x, y, 1)
            slope = float(coeffs[0])
            intercept = float(coeffs[1])
        pred = slope * 3.0 + intercept
        pred = max(pred, 0.0)
        is_increasing = strictly_increasing or (slope > 0.01 and vals[2] > vals[0])
        rows.append({
            "reason": reason,
            "m1": vals[0],
            "m2": vals[1],
            "m3": vals[2],
            "slope": slope,
            "prediction": pred,
            "increasing": is_increasing
        })

    increasing_rows = [r for r in rows if r["increasing"]]
    month_labels = [m.strftime("%Y-%m") for m in last_three]

    if not increasing_rows:
        return "<p><strong>Scrap Reason analysis:</strong> no scrap reasons with clear increasing trend in the last 3 months.</p>"

    # Build HTML table
    html = "<div class='scrap-analysis'><h2>Scrap Reason Trends (last 3 months)</h2>"
    html += "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;'>"
    html += "<thead><tr><th>Scrap Reason</th>"
    html += f"<th>{month_labels[0]}</th><th>{month_labels[1]}</th><th>{month_labels[2]}</th>"
    html += "<th>Slope</th><th>Predicted Next Month</th><th>Trend</th></tr></thead><tbody>"

    increasing_rows = sorted(increasing_rows, key=lambda r: r["slope"], reverse=True)

    for r in increasing_rows:
        html += "<tr>"
        html += f"<td>{r['reason']}</td>"
        html += f"<td align='right'>{r['m1']:.2f}</td>"
        html += f"<td align='right'>{r['m2']:.2f}</td>"
        html += f"<td align='right'>{r['m3']:.2f}</td>"
        html += f"<td align='right'>{r['slope']:.3f}</td>"
        html += f"<td align='right'>{r['prediction']:.2f}</td>"
        html += f"<td>{'Increasing' if r['increasing'] else '—'}</td>"
        html += "</tr>"

    html += "</tbody></table></div>"
    return html

# -------------------- Routes --------------------
@app.route("/", methods=["GET"])
def home():
    return "Control Chart HTML API", 200

@app.route("/process", methods=["POST"])
def process():
    start = time.time()
    # API key optional
    if API_KEY:
        recv_key = request.headers.get("x-api-key")
        if recv_key != API_KEY:
            return jsonify({"error": "invalid api key"}), 401

    if "file" not in request.files:
        return jsonify({"error": "Upload Excel file using 'file' field"}), 400

    try:
        df = pd.read_excel(request.files["file"])
    except Exception as e:
        return jsonify({"error": f"Error reading Excel: {str(e)}"}), 400

    # Normalize columns
    df.columns = df.columns.astype(str).str.strip()

    required_cols = ["Actual Work Center", "Posting Date", "Scrap Qty Breakup"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing required columns: {missing}"}), 400

    # Clean data
    df["Posting Date"] = pd.to_datetime(df["Posting Date"], errors="coerce")
    df = df.dropna(subset=["Posting Date"])
    df["Scrap Qty Breakup"] = pd.to_numeric(df["Scrap Qty Breakup"], errors="coerce").fillna(0)
    df = df.sort_values("Posting Date")

    # Prepare scrap-reason analysis block (will handle missing Scrap Reason gracefully)
    scrap_block = analyze_scrap_reasons(df) if "Scrap Reason" in df.columns else "<p><em>Scrap Reason column not present — skipping analysis.</em></p>"

    # Build charts - one per Actual Work Center
    charts_html = []
    workcenters = df["Actual Work Center"].dropna().unique()
    for wc in workcenters:
        subset = df[df["Actual Work Center"] == wc]
        if subset.empty:
            continue
        fig = control_chart_figure(subset["Scrap Qty Breakup"], subset["Posting Date"], wc, figsize=(11, 6))
        img_b64 = fig_to_base64_png(fig, dpi=120, pad_inches=0.3)
        block = f"""
        <div class="chart-page">
          <h3>Work Center: {wc}</h3>
          <img class="chart-img" src="{img_b64}" alt="chart-{wc}" />
        </div>
        """
        charts_html.append(block)

    if len(charts_html) == 0:
        return jsonify({"error": "No workcenters with data found."}), 400

    # Full HTML template
    html_template = f"""<!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Control Charts - Scrap Qty</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 16px; }}
          h1 {{ margin-bottom: 6px; }}
          .generated {{ font-size: 12px; color: #555; margin-top:0; }}
          .scrap-analysis {{ margin: 12px 0 20px 0; }}
          table {{ border-collapse: collapse; width: 100%; max-width: 100%; }}
          table th, table td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
          .chart-page {{ page-break-after: always; margin-bottom: 28px; }}
          .chart-img {{ display:block; max-width: 100%; height: auto; border:1px solid #ccc; margin-top: 8px; }}
          @page {{ size: A4 portrait; margin: 12mm; }}
        </style>
      </head>
      <body>
        <h1>Control Charts — Scrap Quantity</h1>
        <p class="generated">Generated: {datetime.now().isoformat()}</p>
        {scrap_block}
        {"".join(charts_html)}
      </body>
    </html>"""

    elapsed = time.time() - start
    print(f"/process finished in {elapsed:.2f}s charts={len(charts_html)}", file=sys.stderr)
    sys.stderr.flush()

    return Response(html_template, mimetype="text/html")

# -------------------- main --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
