# app.py
import os
import io
import time
import sys
from datetime import datetime

from flask import Flask, request, send_file, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

app = Flask(__name__)

# Optional: simple API key protection. Set API_KEY env var on the host to enable.
API_KEY = os.getenv("API_KEY")

def control_chart_figure(series, dates, wc_name, figsize=(11, 8.5)):
    """
    Create a matplotlib Figure for the control chart.
    Returns a matplotlib.figure.Figure object.
    """
    values = np.array(series, dtype=float)
    # calculate CL, UCL, LCL
    mean = float(np.mean(values)) if len(values) else 0.0
    std = float(np.std(values, ddof=0)) if len(values) else 0.0
    UCL = mean + 3 * std
    LCL = max(mean - 3 * std, 0.0)

    fig, ax = plt.subplots(figsize=figsize)
    # plot points and line
    ax.plot(dates, values, marker="o", linestyle="-", label="Scrap Qty")
    # control lines
    ax.axhline(mean, color="green", linestyle="--", label=f"CL = {mean:.2f}")
    ax.axhline(UCL, color="red", linestyle="--", label=f"UCL = {UCL:.2f}")
    ax.axhline(LCL, color="orange", linestyle="--", label=f"LCL = {LCL:.2f}")
    # labels and title
    ax.set_title(f"Control Chart – Work Center: {wc_name}", fontsize=14)
    ax.set_ylabel("Scrap Quantity")
    ax.set_xlabel("Posting Date")
    # rotate x ticks for readability
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig

@app.route("/", methods=["GET", "HEAD"])
def health():
    return "OK - control-charts-api", 200

@app.route("/process", methods=["POST"])
def process():
    start_time = time.time()
    print("START /process", file=sys.stderr)
    sys.stderr.flush()

    # API key check (optional)
    if API_KEY:
        recv_key = request.headers.get("x-api-key")
        if recv_key != API_KEY:
            return jsonify({"error": "invalid api key"}), 401

    if "file" not in request.files:
        return jsonify({"error": "Upload Excel file using 'file' field"}), 400

    file = request.files["file"]
    try:
        df = pd.read_excel(file)
    except Exception as e:
        return jsonify({"error": f"Error reading Excel: {str(e)}"}), 400

    # normalize column names
    df.columns = df.columns.astype(str).str.strip()

    required = ["Actual Work Center", "Posting Date", "Scrap Qty Breakup"]
    for col in required:
        if col not in df.columns:
            return jsonify({"error": f"Missing column: {col}"}), 400

    # parse & clean
    df["Posting Date"] = pd.to_datetime(df["Posting Date"], errors="coerce")
    df = df.dropna(subset=["Posting Date"])
    # ensure numeric scrap qty
    df["Scrap Qty Breakup"] = pd.to_numeric(df["Scrap Qty Breakup"], errors="coerce").fillna(0)
    df = df.sort_values("Posting Date")

    workcenters = df["Actual Work Center"].dropna().unique()
    if len(workcenters) == 0:
        return jsonify({"error": "No work centers found in the data"}), 400

    # Create figures for each workcenter
    figs = []
    # Optional: you can filter workcenters by minimum rows, e.g., >=3
    MIN_ROWS = 1  # change to 5 if you want to skip sparse groups
    for wc in workcenters:
        subset = df[df["Actual Work Center"] == wc]
        if subset.shape[0] < MIN_ROWS:
            continue
        # use posting date and scrap qty
        dates = subset["Posting Date"]
        values = subset["Scrap Qty Breakup"]
        try:
            fig = control_chart_figure(values, dates, wc, figsize=(11, 8.5))
            figs.append((wc, fig))
        except Exception as e:
            # skip on plotting failure but log
            print(f"Failed to plot workcenter {wc}: {e}", file=sys.stderr)
            sys.stderr.flush()
            continue

    if len(figs) == 0:
        return jsonify({"error": "No charts produced (maybe no workcenters met MIN_ROWS)"}), 400

    # Write all figures into a single PDF in-memory using PdfPages
    pdf_buffer = io.BytesIO()
    try:
        with PdfPages(pdf_buffer) as pdf:
            for wc, fig in figs:
                # optional: adjust figure size before saving
                # fig.set_size_inches(11, 8.5)  # landscape (already set)
                fig.tight_layout(pad=1.0)
                pdf.savefig(fig, bbox_inches="tight", dpi=150)
                plt.close(fig)
    except Exception as e:
        return jsonify({"error": f"PDF creation failed: {str(e)}"}), 500

    pdf_buffer.seek(0)
    elapsed = time.time() - start_time
    print(f"FINISH /process time={elapsed:.2f}s pages={len(figs)}", file=sys.stderr)
    sys.stderr.flush()

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="control_charts_scrap_qty.pdf"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # bind to all interfaces so Render/CloudRun can route traffic
    app.run(host="0.0.0.0", port=port)
