import os, io, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from flask import Flask, jsonify, render_template, request, send_file
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

app = Flask(__name__)
RAW_CSV_PATH = "cafe.csv"

SERVICE_CONFIG = {
    "Browsing":        {"rate": 15.0,  "weight": 0.25, "avg_dur": 90,  "std_dur": 60,  "prints": False},
    "Gaming":          {"rate": 25.0,  "weight": 0.20, "avg_dur": 160, "std_dur": 70,  "prints": False},
    "Printing":        {"rate": 10.0,  "weight": 0.15, "avg_dur": 60,  "std_dur": 50,  "prints": True},
    "Scanning":        {"rate": 12.0,  "weight": 0.10, "avg_dur": 45,  "std_dur": 30,  "prints": False},
    "Online Class":    {"rate": 15.0,  "weight": 0.15, "avg_dur": 120, "std_dur": 45,  "prints": False},
    "Downloading":     {"rate": 30.0,  "weight": 0.08, "avg_dur": 75,  "std_dur": 40,  "prints": False},
    "Video Streaming": {"rate": 20.0,  "weight": 0.07, "avg_dur": 110, "std_dur": 50,  "prints": False}
}
SERVICE_RATE_MAP = {k: v["rate"] for k, v in SERVICE_CONFIG.items()}
SERVICE_RATE_MAP["Scanning"] = 10.0 
MEMBERSHIPS, TIMES_OF_DAY, PAYMENTS, STATUSES = ["Regular", "Student", "Member"], ["Morning", "Afternoon", "Evening", "Night"], ["Cash", "Credit Card", "Mobile Wallet", "Debit Card"], ["Completed", "Canceled"]
MEMBERSHIP_DISCOUNT_MAP = {'Student': 0.05, 'Member': 0.10, 'Regular': 0.00}
TIME_OF_DAY_MIDPOINT = {'Morning': '09:00', 'Afternoon': '14:00', 'Evening': '18:00', 'Night': '21:00'}

LABEL_META = {
    "Document Taskers": ("Short visits focused on printing/scanning.", "Promote print bundles or bulk prepay packages."),
    "Heavy Gamers / Streamers": ("Long-stay, high data-consumption sessions.", "Offer multi-hour night packages or tier extensions."),
    "VIP Consumers": ("High transactional spend across combined resources.", "Provide priority seating or premium lounge options."),
    "Casual Browsers": ("Standard short-duration digital sessions.", "Deploy snack or drink loyalty coupons."),
    "Budget Users": ("Low-fee, short sessions — price-sensitive customers.", "Offer starter packs or hourly promos."),
    "Power Downloaders": ("Long sessions with above-average spend.", "Upsell unlimited data bundles."),
    "Night Regulars": ("Extended evening/night usage patterns.", "Create night-owl discount packages."),
    "Student Workers": ("Document + research focused, budget-conscious.", "Partner with schools for student discount cards."),
}
LABEL_POOL = [
    ("Document Taskers",        lambda s: s["avg_pages"]),
    ("Heavy Gamers / Streamers",lambda s: s["avg_duration"]),
    ("VIP Consumers",           lambda s: s["avg_fee"]),
    ("Casual Browsers",         lambda s: -s["avg_duration"]),
    ("Budget Users",            lambda s: -s["avg_fee"]),
    ("Power Downloaders",       lambda s: s["avg_duration"] * 0.5 + s["avg_fee"] * 0.5),
    ("Night Regulars",          lambda s: s["avg_duration"] * 0.4 + s["avg_pages"] * 0.6),
    ("Student Workers",         lambda s: s["avg_pages"] * 0.7 - s["avg_fee"] * 0.3),
]

def initialize_synthetic_csv():
    if os.path.exists(RAW_CSV_PATH): return
    random.seed(42); np.random.seed(42)
    records, services = [], list(SERVICE_CONFIG.keys())
    s_weights = [SERVICE_CONFIG[s]["weight"] for s in services]
    for i in range(1, 201):
        srv = random.choice(services) if random.random() < 0.05 else np.random.choice(services, p=s_weights)
        cfg = SERVICE_CONFIG[srv]
        duration = max(10, int(np.random.normal(cfg["avg_dur"], cfg["std_dur"])))
        pages = max(1, int(np.random.normal(12, 6))) if cfg["prints"] else 0
        final_fee = max(5.0, ((duration / 60) * cfg["rate"] + pages * 3.50) * (1.0 - (0.20 if random.choice(MEMBERSHIPS) == "Student" else 0.0)))
        records.append({
            "sessionId": f"SESS-{1000 + i}", "customer_id": f"CUST-{random.randint(100, 350)}",
            "serviceType": random.choice([srv.lower(), srv.upper()]) if random.random() < 0.15 else srv,
            "durationMinutes": None if random.random() < 0.08 else float(duration), "finalFee": None if random.random() < 0.08 else float(final_fee),
            "paymentMethod": None if random.random() < 0.05 else random.choice(PAYMENTS),
            "sessionStatus": None if random.random() < 0.05 else np.random.choice(STATUSES, p=[0.92, 0.08]),
            "pagesPrinted": None if random.random() < 0.08 else float(pages), "membershipType": random.choice(MEMBERSHIPS), "time_of_day": random.choice(TIMES_OF_DAY)
        })
    for _ in range(10): records.append(random.choice(records).copy())
    pd.DataFrame(records).to_csv(RAW_CSV_PATH, index=False)

def get_cleaned_dataframe():
    initialize_synthetic_csv()
    if not os.path.exists(RAW_CSV_PATH): return pd.DataFrame()
    df = pd.read_csv(RAW_CSV_PATH, na_values=['nan', 'NaN', 'NAN', 'null', 'NULL', ''])
    if df.empty: return pd.DataFrame()
    
    df.columns = df.columns.str.strip()
    df['missing_count'] = df.isnull().sum(axis=1)
    df_clean = df.sort_values(by=['sessionId', 'missing_count']).drop_duplicates(subset=['sessionId'], keep='first').copy()
    df_clean.drop(columns=['missing_count'], inplace=True)

    for col in ['serviceType', 'sessionStatus', 'paymentMethod', 'membershipType', 'time_of_day']:
        df_clean[col] = df_clean[col].astype(str).str.strip().str.title() if col in df_clean.columns else 'Unknown'
        df_clean.loc[df_clean[col].str.lower() == 'nan', col] = 'Unknown'

    extra_cols = ['rate_per_hour', 'base_fee', 'discount_rate', 'time', 'computer_number']
    for col in ['durationMinutes', 'finalFee', 'pagesPrinted'] + extra_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce') if col in df_clean.columns else np.nan
    df_clean['date'] = pd.to_datetime(df_clean.get('date'), errors='coerce').dt.strftime('%Y-%m-%d')
    df_clean['date'] = df_clean['date'].fillna(pd.Timestamp.today().strftime('%Y-%m-%d'))

    df_clean['durationMinutes'] = df_clean['durationMinutes'].fillna(df_clean['durationMinutes'].median())
    df_clean['rate_per_hour'] = df_clean['rate_per_hour'].fillna(df_clean['serviceType'].map(SERVICE_RATE_MAP)).fillna(df_clean['rate_per_hour'].median())
    df_clean['base_fee'] = df_clean['base_fee'].fillna((df_clean['durationMinutes'] / 60.0 * df_clean['rate_per_hour']).round(2))
    df_clean['discount_rate'] = df_clean['discount_rate'].fillna(df_clean['membershipType'].map(MEMBERSHIP_DISCOUNT_MAP)).fillna(0.0)
    df_clean['finalFee'] = df_clean['finalFee'].fillna(df_clean['finalFee'].median())
    df_clean['pagesPrinted'] = df_clean['pagesPrinted'].fillna(0.0)
    df_clean['computer_number'] = df_clean['computer_number'].fillna(df_clean['computer_number'].mode()[0] if not df_clean['computer_number'].mode().empty else 0)
    df_clean['time'] = df_clean['time'].astype(str).replace('nan', np.nan).fillna(df_clean['time_of_day'].map(TIME_OF_DAY_MIDPOINT).fillna('12:00'))

    for col, fallback in [('paymentMethod', 'Cash'), ('sessionStatus', 'Completed'), ('serviceType', 'Unknown'), ('membershipType', 'Unknown'), ('time_of_day', 'Unknown')]:
        fill_val = df_clean[col].dropna().mode().to_list()[0] if not df_clean[col].dropna().empty else fallback
        df_clean[col] = df_clean[col].fillna(fill_val).replace('Unknown', fill_val)
    return df_clean

def assign_archetypes(stats_list):
    assigned = {}
    for label, score_fn in LABEL_POOL:
        if len(assigned) == len(stats_list): break
        best_c, best_sc = None, None
        for s in stats_list:
            c = s["cluster"]
            if c in assigned: continue
            sc = score_fn(s)
            if best_sc is None or sc > best_sc: best_sc, best_c = sc, c
        if best_c is not None: assigned[best_c] = label
    for s in stats_list:
        if s["cluster"] not in assigned: assigned[s["cluster"]] = f"Mixed Users (C{s['cluster']})"
    return assigned

def get_cluster_stats(df, k):
    cols = ['durationMinutes', 'finalFee', 'pagesPrinted']
    df['cluster'] = KMeans(n_clusters=k, random_state=42, n_init=15).fit_predict(MinMaxScaler().fit_transform(df[cols]))
    stats_list = []
    for c in range(k):
        sub = df[df['cluster'] == c]
        stats_list.append({
            "cluster": c, "count": len(sub),
            "avg_duration": float(sub['durationMinutes'].mean() or 0),
            "avg_fee": float(sub['finalFee'].mean() or 0),
            "avg_pages": float(sub['pagesPrinted'].mean() or 0)
        })
    assigned = assign_archetypes(stats_list)
    for s in stats_list: s["label"] = assigned[s["cluster"]]
    return df, stats_list

@app.route('/')
def index_view_portal(): return render_template('index.html')

@app.route('/api/clean-data', methods=['GET'])
def get_cleaned_logs_table():
    df = get_cleaned_dataframe()
    if df.empty: return jsonify([])
    renames = {'sessionId': 'session_id', 'serviceType': 'service_type', 'finalFee': 'final_fee', 'paymentMethod': 'payment_method', 'sessionStatus': 'session_status'}
    col_order = ['session_id', 'customer_id', 'date', 'time', 'service_type', 'durationMinutes', 'rate_per_hour', 'base_fee', 'membershipType', 'discount_rate', 'final_fee', 'payment_method', 'pagesPrinted', 'computer_number', 'time_of_day', 'session_status']
    return jsonify([{k: r.get(k, '') for k in col_order} for r in df.rename(columns=renames).fillna('').to_dict(orient='records')])

@app.route('/api/analytics/chart', methods=['POST'])
def get_eda_chart_payload():
    df = get_cleaned_dataframe()
    if df.empty: return jsonify({"labels": [], "values": [], "scatter_data": []})
    p = request.json or {}; col, c_type = p.get('column', 'serviceType'), p.get('chart_type', 'bar')
    if c_type == 'scatter':
        y = 'finalFee' if col != 'finalFee' else 'durationMinutes'
        return jsonify({"labels": [], "values": [], "scatter_data": df[[col, y]].dropna().rename(columns={col: 'x', y: 'y'}).to_dict(orient='records'), "x_axis_title": col, "y_axis_title": y})
    if c_type == 'histogram' and pd.api.types.is_numeric_dtype(df[col]):
        counts, edges = np.histogram(df[col].dropna(), bins=10)
        return jsonify({"labels": [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(counts))], "values": counts.tolist(), "scatter_data": []})
    counts = df[col].value_counts()
    return jsonify({"labels": counts.index.tolist(), "values": counts.values.tolist(), "scatter_data": []})

@app.route('/api/analytics/correlation', methods=['GET'])
def get_eda_correlation_matrix():
    df = get_cleaned_dataframe()
    cols = ['durationMinutes', 'finalFee', 'pagesPrinted']
    return jsonify({"columns": cols, "matrix": [] if df.empty else df[cols].corr().fillna(0).round(4).values.tolist()})

@app.route('/api/analytics/features', methods=['POST'])
def get_scaled_feature_vectors():
    df = get_cleaned_dataframe()
    if df.empty: return jsonify({"headers": [], "raw": [], "scaled": []})
    cols = ['durationMinutes', 'finalFee', 'pagesPrinted']
    scaled = MinMaxScaler().fit_transform(df[cols])
    return jsonify({"headers": cols, "raw": df[cols].round(4).to_dict(orient='records'), "scaled": pd.DataFrame(scaled, columns=cols).round(4).to_dict(orient='records'), "stats": {c: {"min": float(df[c].min()), "max": float(df[c].max())} for c in cols}})

@app.route('/api/analytics/clustering-metrics', methods=['GET'])
def get_clustering_evaluation_curves():
    df = get_cleaned_dataframe()
    if df.empty or len(df) < 10: return jsonify({"k_values": [], "inertia": [], "silhouette": []})
    X = MinMaxScaler().fit_transform(df[['durationMinutes', 'finalFee', 'pagesPrinted']])
    k_arr, inertia, sil = [], [], []
    for k in range(2, min(9, len(df))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        k_arr.append(k); inertia.append(float(km.inertia_)); sil.append(round(float(silhouette_score(X, labels)), 4))
    return jsonify({"k_values": k_arr, "inertia": inertia, "silhouette": sil})

@app.route('/api/analytics/kmeans', methods=['POST'])
def execute_kmeans_segmentation():
    df = get_cleaned_dataframe()
    if df.empty: return jsonify({"clusters": {}, "centroids": [], "summary": []})
    k = int((request.json or {}).get('k', 3))
    df, stats_list = get_cluster_stats(df, k)
    c_data = {str(c): df[df['cluster'] == c][['durationMinutes', 'finalFee']].rename(columns={'durationMinutes': 'x', 'finalFee': 'y'}).to_dict(orient='records') for c in range(k)}
    centroids = [{"cluster": s["cluster"], "x": s["avg_duration"], "y": s["avg_fee"]} for s in stats_list]
    return jsonify({"clusters": c_data, "centroids": centroids, "summary": [{**s, "avg_duration": round(s["avg_duration"], 1), "avg_fee": round(s["avg_fee"], 2), "avg_pages": round(s["avg_pages"], 1)} for s in stats_list]})

@app.route('/api/export/pdf', methods=['POST'])
def export_analytical_pdf_report():
    df = get_cleaned_dataframe()
    if df.empty: return "No data.", 400
    k = int((request.json or {}).get('k', 3))
    df, stats_list = get_cluster_stats(df, k)
    
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    stys = getSampleStyleSheet()
    ts = ParagraphStyle('T', parent=stys['Heading1'], fontSize=22, leading=26, textColor=colors.HexColor('#1e293b'), spaceAfter=8)
    h2s = ParagraphStyle('H2', parent=stys['Heading2'], fontSize=13, leading=16, textColor=colors.HexColor('#475569'), spaceBefore=14, spaceAfter=8)
    bs = ParagraphStyle('B', parent=stys['BodyText'], fontSize=9.5, leading=13.5, textColor=colors.HexColor('#334155'))

    story = [Paragraph("CyberCafé Operation Segmentation Report", ts), Paragraph("Automated behavioral clustering across operational logs.", bs), Spacer(1, 15)]
    table_data = [[Paragraph(f"<b>{h}</b>", bs) for h in ["Cluster", "Archetype", "Avg Duration", "Avg Fee", "Avg Pages"]]]
    
    for s in stats_list:
        desc, strat = LABEL_META.get(s["label"], ("Mixed usage pattern.", "Review usage trends regularly."))
        table_data.append([Paragraph(f"#{s['cluster']}", bs), Paragraph(f"<b>{s['label']}</b>", bs), Paragraph(f"{s['avg_duration']:.1f} min", bs), Paragraph(f"₱{s['avg_fee']:.2f}", bs), Paragraph(f"{s['avg_pages']:.1f} pgs", bs)])

    t = Table(table_data, colWidths=[45, 180, 90, 90, 90])
    t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f8fafc")), ("ALIGN", (0,0), (-1,-1), "LEFT"), ("BOTTOMPADDING", (0,0), (-1,-1), 8), ("TOPPADDING", (0,0), (-1,-1), 8), ("LINEBELOW", (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1"))]))
    story += [Paragraph("Cluster Profile Matrix", h2s), t, Spacer(1, 15), Paragraph("Strategic Persona Insights", h2s)]
    
    for s in stats_list:
        desc, strat = LABEL_META.get(s["label"], ("Mixed usage pattern.", "Review usage trends regularly."))
        story += [Paragraph(f"<b>Cluster {s['cluster']} — {s['label']}</b>", bs), Paragraph(f"<i>Behavior:</i> {desc}", bs), Paragraph(f"<i>Action Plan:</i> {strat}", bs), Spacer(1, 8)]

    doc.build(story); buf.seek(0)
    return send_file(buf, mimetype='application/pdf', as_attachment=True, download_name="cybercafe_cluster_report.pdf")

if __name__ == '__main__':
    initialize_synthetic_csv()
    app.run(debug=True, port=5000)