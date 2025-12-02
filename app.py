# app.py (return HTML with embedded charts)
import io
import base64
import sys
import time
from datetime import datetime

from flask import Flask, request, Response, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

# small helper: render matplotlib figure to base64 PNG
def fig_to_base64_png(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return "data:image/png;base64," + b64

def control_chart_figure(series, dates, wc_name, figsize=(11, 6)):
    values = np.array(series, dtype=float)
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
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig

@app.route("/", methods=["GET"])
def health():
    return "Control Chart HTML API", 200

@app.route("/process", methods=["POST"])
def process():
    start = time.time()

    if "file" not in request.files:
        return jsonify({"error": "Upload Excel file using 'file' field"}), 400

    try:
        df = pd.read_excel(request.files["file"])
    except Exception as e:
        return jsonify({"error": f"Error reading Excel: {str(e)}"}), 400

    df.columns = df.columns.astype(str).str.strip()
    required = ["Actual Work Center", "Posting Date", "Scrap Qty Breakup"]
    for col in required:
        if col not in df.columns:
            return jsonify({"error": f"Missing column: {col}"}), 400

    df["Posting Date"] = pd.to_datetime(df["Posting Date"], errors="coerce")
    df = df.dropna(subset=["Posting Date"])
    df["Scrap Qty Breakup"] = pd.to_numeric(df["Scrap Qty Breakup"], errors="coerce").fillna(0)
    df = df.sort_values("Posting Date")

    workcenters = df["Actual Work Center"].dropna().unique()
    if len(workcenters) == 0:
        return jsonify({"error": "No work centers found"}), 400

    charts_html = []
    for wc in workcenters:
        subset = df[df["Actual Work Center"] == wc]
        if subset.empty:
            continue
        fig = control_chart_figure(subset["Scrap Qty Breakup"], subset["Posting Date"], wc, figsize=(11,6))
        img_b64 = fig_to_base64_png(fig, dpi=120)
        # each chart block: title + timestamp + img
        block = f"""
        <div class="chart-page">
          <h2 class="page-title">Control Chart - Scrap Qty</h2>
          <p class="generated">Generated: {datetime.now().isoformat()}</p>
          <h3 class="wc-name">{wc}</h3>
          <img class="chart-img" src="{img_b64}" alt="chart-{wc}" />
        </div>
        """
        charts_html.append(block)

    html_template = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Control Charts - Scrap Qty</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 16px; }}
          .chart-page {{ page-break-after: always; margin-bottom: 30px; }}
          .page-title {{ text-align: left; font-size: 28px; margin: 8px 0; }}
          .generated {{ font-size: 12px; color: #333; margin: 0 0 8px 0; }}
          .wc-name {{ font-size: 18px; margin: 4px 0 12px 0; }}
          .chart-img {{ display: block; max-width: 100%; height: auto; border: 1px solid #ddd; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
          /* Helpful when printing to PDF */
          @page {{ size: A4 portrait; margin: 12mm; }}
        </style>
      </head>
      <body>
        <h1>Control Charts - Scrap Qty</h1>
        {"".join(charts_html)}
      </body>
    </html>
    """

    elapsed = time.time() - start
    # small log for diagnostics
    print(f"/process generated {len(charts_html)} charts in {elapsed:.2f}s", file=sys.stderr)

    return Response(html_template, mimetype="text/html")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
