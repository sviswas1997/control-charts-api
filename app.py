from flask import Flask, request, send_file, jsonify
import pandas as pd
import io, base64, os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from weasyprint import HTML

app = Flask(__name__)

def to_base64_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

def control_chart(series, dates, wc_name):
    values = np.array(series)
    mean = np.mean(values) if len(values) else 0.0
    std = np.std(values, ddof=0) if len(values) else 0.0
    UCL = mean + 3 * std
    LCL = max(mean - 3 * std, 0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, values, marker='o', linestyle='-', label='Scrap Qty')
    ax.axhline(mean, color='green', linestyle='--', label=f'CL = {mean:.2f}')
    ax.axhline(UCL, color='red', linestyle='--', label=f'UCL = {UCL:.2f}')
    ax.axhline(LCL, color='orange', linestyle='--', label=f'LCL = {LCL:.2f}')
    ax.set_title(f"Control Chart – Work Center: {wc_name}")
    ax.set_ylabel("Scrap Quantity")
    ax.set_xlabel("Posting Date")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    return fig

@app.route("/process", methods=["POST"])
def process():
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

    charts = []
    for wc in workcenters:
        subset = df[df["Actual Work Center"] == wc]
        if subset.empty:
            continue
        fig = control_chart(subset["Scrap Qty Breakup"], subset["Posting Date"], wc)
        charts.append((wc, to_base64_png(fig)))

    html = "<html><body>"
    html += f"<h1>Control Charts – Scrap Qty</h1><p>Generated: {datetime.now()}</p>"
    for wc, img in charts:
        html += f"<h2>{wc}</h2><img src='{img}' />"
    html += "</body></html>"

    try:
        pdf = HTML(string=html).write_pdf()
    except Exception as e:
        return jsonify({"error": f"PDF conversion failed: {str(e)}"}), 500

    return send_file(io.BytesIO(pdf), mimetype="application/pdf", as_attachment=True, download_name="control_charts_scrap_qty.pdf")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
