# app.py (A4-safe PDF output)
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

API_KEY = os.getenv("API_KEY")

# A4 sizes in inches
A4_PORTRAIT = (8.27, 11.69)
A4_LANDSCAPE = (11.69, 8.27)

# Settings you can tweak
FIGSIZE = A4_LANDSCAPE           # use A4 landscape as page size
DPI = 120                        # reduce for speed (120 is fine); raise for higher quality
PAD_INCHES = 0.6                 # padding when saving PDF to avoid clipping
MIN_ROWS = 1                     # min rows to generate chart for a workcenter

def control_chart_figure(series, dates, wc_name, figsize=FIGSIZE):
    """Create and return a matplotlib Figure sized for an A4 page."""
    values = np.array(series, dtype=float)
    mean = float(np.mean(values)) if len(values) else 0.0
    std = float(np.std(values, ddof=0)) if len(values) else 0.0
    UCL = mean + 3 * std
    LCL = max(mean - 3 * std, 0.0)

    # create fig with exact page dimensions
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

    ax.plot(dates, values, marker="o", linestyle="-", label="Scrap Qty")
    ax.axhline(mean, color="green", linestyle="--", label=f"CL = {mean:.2f}")
    ax.axhline(UCL, color="red", linestyle="--", label=f"UCL = {UCL:.2f}")
    ax.axhline(LCL, color="orange", linestyle="--", label=f"LCL = {LCL:.2f}")

    ax.set_title(f"Control Chart – Work Center: {wc_name}", fontsize=16)
    ax.set_ylabel("Scrap Quantity", fontsize=12)
    ax.set_xlabel("Posting Date", fontsize=12)

    # improve tick label fit
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    ax.tick_params(axis="y", labelsize=10)

    # place legend inside plot but with enough room
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98))

    # Adjust subplot params so title + ticks fit comfortably within A4 margins
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14)

    return fig

@app.route("/", methods=["GET", "HEAD"])
def health():
    return "OK - control-charts-api", 200

@app.route("/process", methods=["POST"])
def process():
    start_time = time.time()
    print("START /process", file=sys.stderr)
    sys.stderr.flush()

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
        return jsonify({"error": "No work centers found in the data"}), 400

    figs = []
    for wc in workcenters:
        subset = df[df["Actual Work Center"] == wc]
        if subset.shape[0] < MIN_ROWS:
            continue
        dates = subset["Posting Date"]
        values = subset["Scrap Qty Breakup"]
        try:
            fig = control_chart_figure(values, dates, wc, figsize=FIGSIZE)
            figs.append((wc, fig))
        except Exception as e:
            print(f"Failed to create chart for {wc}: {e}", file=sys.stderr)
            sys.stderr.flush()
            continue

    if len(figs) == 0:
        return jsonify({"error": "No charts produced (maybe no workcenters met MIN_ROWS)"}), 400

    pdf_buffer = io.BytesIO()
    try:
        with PdfPages(pdf_buffer) as pdf:
            for wc, fig in figs:
                # ensure layout and then save with padding to prevent any clipping
                fig.tight_layout()
                # Save to the PDF with pad_inches (prevents cropping of titles/labels)
                pdf.savefig(fig, bbox_inches="tight", pad_inches=PAD_INCHES)
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
    app.run(host="0.0.0.0", port=port)
