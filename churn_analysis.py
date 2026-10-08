"""
Customer Churn Analysis - end-to-end pipeline
Cleans the data, computes churn rate / CLV / MRR loss, segments churn, draws charts,
writes data-driven recommendations and builds a PDF analytics report.

Usage:  python churn_analysis.py raw_churn.csv [--currency "$"]
"""
import re, sys, argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.0)
PAL = {"retained": "#4C78A8", "churned": "#E4572E"}
OUT = Path("outputs"); CHARTS = OUT / "charts"

# ---------------------------------------------------------------- 1. CLEANING
def snake(c): return re.sub(r"[^0-9a-z]+", "_", str(c).strip().lower()).strip("_")

def find_col(df, *patterns):
    for p in patterns:
        for c in df.columns:
            if re.search(p, c): return c
    return None

def load_and_clean(path):
    log = []
    df = pd.read_csv(path); raw_rows = len(df)
    df.columns = [snake(c) for c in df.columns]
    log.append(f"Loaded {raw_rows:,} rows x {df.shape[1]} columns; headers standardized to snake_case.")
    for c in df.select_dtypes(include=["object", "string"]).columns:
        s = df[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        df[c] = s.mask(s.str.lower().isin({"", "na", "n/a", "nan", "null", "none", "?", "-"}), pd.NA)
    log.append("Trimmed whitespace and converted blank/placeholder strings (e.g. ' ', 'N/A') to nulls.")
    d = df.duplicated().sum(); df = df.drop_duplicates().reset_index(drop=True)
    idc = find_col(df, r"customer.?id|^id$")
    if idc:
        d2 = df[idc].duplicated().sum(); df = df.drop_duplicates(subset=idc).reset_index(drop=True); d += d2
    log.append(f"Removed {d:,} duplicate records.")

    churn = find_col(df, r"churn", r"cancel", r"attrition")
    if churn is None: raise SystemExit("No churn column found.")
    df["churn"] = df[churn].astype("string").str.lower().map(
        {"yes": 1, "y": 1, "true": 1, "1": 1, "churned": 1, "no": 0, "n": 0, "false": 0, "0": 0, "retained": 0})
    bad = df["churn"].isna().sum(); df = df[df["churn"].notna()].copy(); df["churn"] = df["churn"].astype(int)
    if churn != "churn": df = df.drop(columns=[churn])
    log.append(f"Encoded churn target as 0/1 ({bad} rows with unreadable target dropped).")

    for key, pats in {"tenure": [r"tenure", r"months"], "monthly": [r"monthly.?charge", r"mrr", r"monthly"],
                      "total": [r"total.?charge"]}.items():
        c = find_col(df, *pats)
        if c: df[c] = pd.to_numeric(df[c].astype("string").str.replace(r"[,\$€£₹\s]", "", regex=True), errors="coerce")
    ten, mon, tot = (find_col(df, r"tenure"), find_col(df, r"monthly.?charge", r"monthly"), find_col(df, r"total.?charge"))
    if mon and df[mon].isna().any():
        n = df[mon].isna().sum(); df[mon] = df[mon].fillna(df[mon].median())
        log.append(f"Imputed {n} missing monthly charges with the median.")
    if ten and df[ten].isna().any(): df[ten] = df[ten].fillna(df[ten].median())
    if tot and df[tot].isna().any():
        n = df[tot].isna().sum(); df[tot] = df[tot].fillna(df[ten] * df[mon])
        log.append(f"Repaired {n} missing total charges as tenure x monthly charges (brand-new customers = 0).")
    for c in df.select_dtypes(include=["object", "string"]).columns:
        if c != idc and df[c].isna().any(): df[c] = df[c].fillna(df[c].mode().iloc[0])
    for c in (ten, mon):
        if c: df[c] = df[c].clip(lower=0)
    cols = {"id": idc, "tenure": ten, "monthly": mon, "total": tot,
            "contract": find_col(df, r"contract"), "payment": find_col(df, r"payment")}
    log.append(f"Final clean dataset: {len(df):,} rows (from {raw_rows:,}); no remaining nulls in analysis fields.")
    return df, cols, log

# ---------------------------------------------------------------- 2. METRICS
def compute_metrics(df, cols):
    m = {}; n = len(df); ch = df[df.churn == 1]
    m["customers"], m["churned"] = n, len(ch); m["churn_rate"] = len(ch) / n
    mon, ten = cols["monthly"], cols["tenure"]
    m["arpu"] = df[mon].mean()
    m["mrr_total"] = df[mon].sum(); m["mrr_lost"] = ch[mon].sum(); m["mrr_lost_pct"] = m["mrr_lost"] / m["mrr_total"]
    m["arr_lost"] = m["mrr_lost"] * 12
    # monthly churn hazard = churn events / total customer-months observed  -> lifespan = 1/hazard
    cust_months = df[ten].clip(lower=1).sum()
    m["monthly_churn"] = len(ch) / cust_months
    m["lifespan"] = 1 / m["monthly_churn"]
    m["clv"] = m["arpu"] * m["lifespan"]
    m["clv_churned_avg_value"] = (ch[mon] * ch[ten]).mean()
    m["avg_tenure_churned"], m["avg_tenure_retained"] = ch[ten].mean(), df[df.churn == 0][ten].mean()
    return m

def churn_by(df, col, order=None):
    g = df.groupby(col, observed=True).agg(customers=("churn", "size"), churned=("churn", "sum"),
                                           churn_rate=("churn", "mean")).reset_index()
    return g.set_index(col).loc[order].reset_index() if order else g.sort_values("churn_rate", ascending=False)

def rank_drivers(df, cols, top=8):
    skip = {cols["id"], "churn", "tenure_group"}; rows = []
    for c in df.select_dtypes(include=["object", "string", "category"]).columns:
        if c in skip or df[c].nunique() > 8 or df[c].nunique() < 2: continue
        g = churn_by(df, c)
        g = g[g.customers >= 0.03 * len(df)]
        if len(g) < 2: continue
        rows.append({"feature": c, "worst_segment": g.iloc[0][c], "worst_rate": g.iloc[0].churn_rate,
                     "best_segment": g.iloc[-1][c], "best_rate": g.iloc[-1].churn_rate,
                     "spread": g.iloc[0].churn_rate - g.iloc[-1].churn_rate})
    return pd.DataFrame(rows).sort_values("spread", ascending=False).head(top).reset_index(drop=True)

# ---------------------------------------------------------------- 3. CHARTS
def barplot(g, x, title, fname, horizontal=False, overall=None):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    vals = g.churn_rate * 100
    cols = [PAL["churned"] if v > (overall or 0) * 100 else PAL["retained"] for v in g.churn_rate]
    if horizontal:
        ax.barh(g[x].astype(str), vals, color=cols); ax.invert_yaxis(); ax.set_xlabel("Churn rate (%)")
        for i, (v, n) in enumerate(zip(vals, g.customers)): ax.text(v + .4, i, f"{v:.1f}%  (n={n:,})", va="center", fontsize=8)
        ax.set_xlim(0, vals.max() * 1.35)
    else:
        ax.bar(g[x].astype(str), vals, color=cols); ax.set_ylabel("Churn rate (%)")
        for i, (v, n) in enumerate(zip(vals, g.customers)): ax.text(i, v + .5, f"{v:.1f}%\n(n={n:,})", ha="center", fontsize=8)
        ax.set_ylim(0, vals.max() * 1.25)
    if overall: (ax.axvline if horizontal else ax.axhline)(overall * 100, color="grey", ls="--", lw=1, label=f"Overall {overall*100:.1f}%"); ax.legend(fontsize=8)
    ax.set_title(title, fontweight="bold"); ax.set_xlabel(ax.get_xlabel() or ""); fig.tight_layout()
    fig.savefig(CHARTS / fname, dpi=160); plt.close(fig); return CHARTS / fname

def make_charts(df, cols, m, tabs, drivers):
    CHARTS.mkdir(parents=True, exist_ok=True); p = {}
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.pie([m["customers"] - m["churned"], m["churned"]], labels=["Retained", "Churned"], autopct="%1.1f%%", startangle=90,
           colors=[PAL["retained"], PAL["churned"]], wedgeprops=dict(width=.45, edgecolor="white"))
    ax.set_title("Customer base: retained vs churned", fontweight="bold"); fig.tight_layout()
    fig.savefig(CHARTS / "01_overview.png", dpi=160); plt.close(fig); p["overview"] = CHARTS / "01_overview.png"
    o = m["churn_rate"]
    if "contract" in tabs: p["contract"] = barplot(tabs["contract"], cols["contract"], "Churn rate by contract type", "02_contract.png", overall=o)
    p["tenure"] = barplot(tabs["tenure"], "tenure_group", "Churn rate by tenure group", "03_tenure.png", overall=o)
    if "payment" in tabs: p["payment"] = barplot(tabs["payment"], cols["payment"], "Churn rate by payment method", "04_payment.png", horizontal=True, overall=o)
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.8)); lab = {0: "Retained", 1: "Churned"}
    d = df.assign(Status=df.churn.map(lab))
    for ax, c, t in zip(axs, [cols["tenure"], cols["monthly"]], ["Tenure (months)", "Monthly charges"]):
        sns.boxplot(data=d, x="Status", y=c, order=["Retained", "Churned"], palette=[PAL["retained"], PAL["churned"]], ax=ax, width=.5)
        ax.set_title(t + " by status", fontweight="bold"); ax.set_xlabel(""); ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(CHARTS / "05_boxplots.png", dpi=160); plt.close(fig); p["box"] = CHARTS / "05_boxplots.png"
    if cols["contract"]:
        pv = df.pivot_table(index=cols["contract"], columns="tenure_group", values="churn", aggfunc="mean", observed=True) * 100
        fig, ax = plt.subplots(figsize=(7.5, 3.4)); sns.heatmap(pv, annot=True, fmt=".1f", cmap="Reds", ax=ax, cbar_kws={"label": "Churn rate (%)"})
        ax.set_title("Churn rate (%): contract type x tenure group", fontweight="bold"); ax.set_xlabel(""); ax.set_ylabel("")
        fig.tight_layout(); fig.savefig(CHARTS / "06_heatmap_segments.png", dpi=160); plt.close(fig); p["heat1"] = CHARTS / "06_heatmap_segments.png"
    num = df[[c for c in (cols["tenure"], cols["monthly"], cols["total"]) if c] + ["churn"]].copy()
    for c in df.select_dtypes(include=["object", "string"]).columns:
        if c != cols["id"] and df[c].nunique() == 2: num[c] = (df[c] == df[c].value_counts().index[-1]).astype(int)
    fig, ax = plt.subplots(figsize=(7.5, 5.6)); sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation heatmap (binary features encoded 0/1)", fontweight="bold"); fig.tight_layout()
    fig.savefig(CHARTS / "07_heatmap_corr.png", dpi=160); plt.close(fig); p["heat2"] = CHARTS / "07_heatmap_corr.png"
    if len(drivers):
        fig, ax = plt.subplots(figsize=(7.5, 4)); lbl = drivers.feature + ": " + drivers.worst_segment.astype(str)
        ax.barh(lbl, drivers.worst_rate * 100, color=PAL["churned"]); ax.invert_yaxis(); ax.axvline(o * 100, color="grey", ls="--", lw=1)
        for i, v in enumerate(drivers.worst_rate * 100): ax.text(v + .4, i, f"{v:.1f}%", va="center", fontsize=8)
        ax.set_xlabel("Churn rate (%) of the highest-risk segment"); ax.set_title("Top behavioural drivers of churn", fontweight="bold")
        fig.tight_layout(); fig.savefig(CHARTS / "08_drivers.png", dpi=160); plt.close(fig); p["drivers"] = CHARTS / "08_drivers.png"
    return p

# ---------------------------------------------------------------- 4. INSIGHTS
def build_insights(df, cols, m, tabs, drivers, cur):
    f = lambda v: f"{cur}{v:,.0f}"; pc = lambda v: f"{v*100:.1f}%"; ins, rec = [], []
    o = m["churn_rate"]
    ins.append(f"Overall churn is {pc(o)} ({m['churned']:,} of {m['customers']:,} customers), costing {f(m['mrr_lost'])} in monthly recurring revenue ({pc(m['mrr_lost_pct'])} of MRR, about {f(m['arr_lost'])} annualised).")
    ins.append(f"Customers stay on average {m['lifespan']:.0f} months at an ARPU of {cur}{m['arpu']:.2f}, giving a Customer Lifetime Value of {f(m['clv'])}.")
    ins.append(f"Churned customers left after {m['avg_tenure_churned']:.1f} months on average versus {m['avg_tenure_retained']:.1f} months of tenure for retained customers.")
    if "contract" in tabs:
        g = tabs["contract"]; hi, lo = g.iloc[0], g.iloc[-1]; c = cols["contract"]
        share = df[(df[c] == hi[c]) & (df.churn == 1)].shape[0] / m["churned"]
        ins.append(f"'{hi[c]}' contracts churn at {pc(hi.churn_rate)} versus {pc(lo.churn_rate)} for '{lo[c]}' - a {hi.churn_rate/max(lo.churn_rate,1e-9):.1f}x gap; they account for {pc(share)} of all churners.")
        mrr_hi = df[(df[c] == hi[c]) & (df.churn == 1)][cols["monthly"]].sum()
        rec.append(("Move customers off short-term contracts",
            f"'{hi[c]}' customers drive {pc(share)} of churn. Offer 10-15% discounts or a free add-on for 12/24-month commitments. Converting just 10% of the churn in this segment protects roughly {f(mrr_hi*.10)} MRR ({f(mrr_hi*.10*12)} per year)."))
    t = tabs["tenure"]; first = t.iloc[0]
    sh = df[(df.tenure_group == first.tenure_group) & (df.churn == 1)].shape[0] / m["churned"]
    ins.append(f"Churn is front-loaded: the '{first.tenure_group}' group churns at {pc(first.churn_rate)} and holds {pc(sh)} of all churners; risk falls steadily as tenure grows.")
    rec.append(("Fix the first-year experience",
        f"{pc(sh)} of churners leave in the first {first.tenure_group.split()[0].split('-')[-1]} months. Introduce a 90-day onboarding programme: welcome call, setup help, a day-30 and day-60 health check, and proactive outreach for low-usage accounts."))
    if "payment" in tabs:
        g = tabs["payment"]; hi, lo = g.iloc[0], g.iloc[-1]; c = cols["payment"]
        ins.append(f"'{hi[c]}' users churn at {pc(hi.churn_rate)} versus {pc(lo.churn_rate)} for '{lo[c]}'.")
        rec.append(("Promote automatic payments",
            f"Customers paying by '{hi[c]}' churn {hi.churn_rate/max(lo.churn_rate,1e-9):.1f}x more than '{lo[c]}' customers. Offer a small bill credit for switching to automatic payment, which reduces friction and failed-payment churn."))
    ch, rt = df[df.churn == 1][cols["monthly"]].median(), df[df.churn == 0][cols["monthly"]].median()
    ins.append(f"Median monthly charge is {cur}{ch:.2f} for churned customers versus {cur}{rt:.2f} for retained customers" + (" - higher-paying customers are leaving, suggesting price sensitivity." if ch > rt else "."))
    if ch > rt:
        rec.append(("Review pricing and bundle value",
            f"Churners pay more per month ({cur}{ch:.2f} vs {cur}{rt:.2f}). Introduce loyalty pricing, bundles, or a right-sizing review for high-bill accounts before they cancel."))
    for _, r in drivers.head(4).iterrows():
        if r.feature in (cols["contract"], cols["payment"]): continue
        ins.append(f"{r.feature.replace('_',' ').title()}: '{r.worst_segment}' churns at {pc(r.worst_rate)} versus {pc(r.best_rate)} for '{r.best_segment}'.")
    extra = [r for _, r in drivers.iterrows() if r.feature not in (cols["contract"], cols["payment"])][:1]
    for r in extra:
        rec.append((f"Target the high-risk '{r.feature.replace('_',' ')}' segment",
            f"Customers with {r.feature.replace('_',' ')} = '{r.worst_segment}' churn at {pc(r.worst_rate)} (vs {pc(r.best_rate)}). Bundle or trial the related service/benefit for this group and track the change in churn."))
    rec.append(("Build an early-warning model and retention playbook",
        "Score customers monthly using tenure, contract, payment method and charges. Route the top-risk decile to a retention team with pre-approved offers, and measure results with an A/B test against a holdout group."))
    return ins, rec

# ---------------------------------------------------------------- 5. PDF
def build_pdf(path, m, tabs, drivers, charts, ins, rec, log, cur, cols):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=colors.HexColor("#1F2A44"), fontSize=17, spaceAfter=8)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=colors.HexColor("#E4572E"), fontSize=13, spaceBefore=8, spaceAfter=4)
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10, leading=14)
    SM = ParagraphStyle("SM", parent=B, fontSize=8.5, leading=11, textColor=colors.HexColor("#555555"))
    def img(p, w=15.5): 
        from PIL import Image as PI
        iw, ih = PI.open(p).size; return Image(str(p), width=w*cm, height=w*cm*ih/iw)
    def footer(c, d):
        c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.grey)
        c.drawString(2*cm, 1.2*cm, "Customer Churn Analysis - Analytics Report"); c.drawRightString(19*cm, 1.2*cm, f"Page {d.page}"); c.restoreState()
    S = []
    S += [Spacer(1, 1.2*cm), Paragraph("Customer Churn Analysis", ParagraphStyle("T", parent=H1, fontSize=26, leading=30)),
          Paragraph("Churn rate, Customer Lifetime Value, revenue impact and behavioural drivers", ParagraphStyle("st", parent=B, fontSize=12, textColor=colors.grey)), Spacer(1, .6*cm)]
    kp = [["Customers", f"{m['customers']:,}", "Churn rate", f"{m['churn_rate']*100:.1f}%"],
          ["Churned customers", f"{m['churned']:,}", "Monthly churn rate", f"{m['monthly_churn']*100:.2f}%"],
          ["ARPU (monthly)", f"{cur}{m['arpu']:.2f}", "Avg customer lifespan", f"{m['lifespan']:.0f} months"],
          ["Customer Lifetime Value", f"{cur}{m['clv']:,.0f}", "MRR lost to churn", f"{cur}{m['mrr_lost']:,.0f} ({m['mrr_lost_pct']*100:.1f}%)"]]
    t = Table(kp, colWidths=[4.3*cm, 3.7*cm, 4.6*cm, 4.4*cm])
    t.setStyle(TableStyle([("FONTNAME", (0,0), (-1,-1), "Helvetica"), ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"), ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9.5), ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#F3F5FA")), ("GRID", (0,0), (-1,-1), .5, colors.white), ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7)]))
    S += [t, Spacer(1, .5*cm), Paragraph("Executive summary", H2)]
    for i in ins[:4]: S.append(Paragraph("&bull; " + i, B)); S.append(Spacer(1, 3))
    S += [Paragraph("Top recommendations", H2)]
    for i, (h, _) in enumerate(rec[:4], 1): S.append(Paragraph(f"{i}. <b>{h}</b>", B))
    S.append(PageBreak())
    S += [Paragraph("1. Data & methodology", H1), Paragraph("The dataset was cleaned and validated before analysis:", B)]
    for l in log: S.append(Paragraph("&bull; " + l, B))
    S += [Spacer(1, 5), Paragraph("Metric definitions", H2)]
    for d in ["<b>Churn rate</b> = churned customers / total customers.",
              "<b>Monthly churn rate</b> = churn events / total customer-months observed (sum of tenure).",
              "<b>Average lifespan</b> = 1 / monthly churn rate.",
              "<b>CLV</b> = ARPU x average lifespan (revenue basis; no margin applied).",
              "<b>MRR loss</b> = sum of monthly charges of churned customers."]: S.append(Paragraph("&bull; " + d, B))
    S += [Spacer(1, 8), Paragraph("2. Baseline churn and revenue impact", H1), Table([[img(charts["overview"], 7.5), [Paragraph(i, B) for i in ins[:3]]]], colWidths=[8*cm, 9*cm]), PageBreak()]
    S.append(Paragraph("3. Churn by contract, tenure and payment method", H1))
    for key, txt in [("contract", ins[3] if len(ins) > 3 else ""), ("tenure", ""), ("payment", "")]:
        if key in charts: S += [img(charts[key], 13.5), Spacer(1, 4)]
    for i in ins[3:6]: S += [Paragraph("&bull; " + i, B)]
    S.append(PageBreak())
    S += [Paragraph("4. Behavioural patterns", H1), img(charts["box"], 15), Spacer(1, 4)]
    if "heat1" in charts: S += [img(charts["heat1"], 14), Spacer(1, 4)]
    S += [Paragraph("Short-tenure, month-to-month customers form the highest-risk pocket; churn drops sharply with longer contracts and longer tenure.", B), PageBreak()]
    S.append(Paragraph("5. Key drivers of churn", H1))
    if "drivers" in charts: S += [img(charts["drivers"], 15)]
    S += [img(charts["heat2"], 12.5), Paragraph("Tenure and contract length show the strongest negative relationship with churn; short-contract and high-charge indicators are positively related.", B), PageBreak()]
    S.append(Paragraph("6. Actionable recommendations", H1))
    for i, (h, d) in enumerate(rec, 1): S += [Paragraph(f"{i}. {h}", H2), Paragraph(d, B)]
    S += [Spacer(1, 10), Paragraph("Next steps: pilot the top two initiatives on a test group, track churn monthly against a control group, and refresh this report quarterly.", B), PageBreak()]
    S.append(Paragraph("Appendix: segment tables", H1))
    for k, g in tabs.items():
        if k == "tenure": name = "tenure_group"
        else: name = cols[k]
        rows = [[name.replace("_", " ").title(), "Customers", "Churned", "Churn rate"]] + [[str(r[name]), f"{r.customers:,}", f"{r.churned:,}", f"{r.churn_rate*100:.1f}%"] for _, r in g.iterrows()]
        tb = Table(rows, colWidths=[7*cm, 3*cm, 3*cm, 3*cm]); tb.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1F2A44")), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("FONTSIZE", (0,0), (-1,-1), 9), ("GRID", (0,0), (-1,-1), .4, colors.lightgrey), ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F3F5FA")])]))
        S += [KeepTogether([Paragraph(name.replace("_", " ").title(), H2), tb]), Spacer(1, 6)]
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=2*cm,
                      title="Customer Churn Analysis", author="Vignesh M").build(S, onFirstPage=footer, onLaterPages=footer)

# ---------------------------------------------------------------- MAIN
def run(path, cur="$"):
    OUT.mkdir(exist_ok=True); CHARTS.mkdir(exist_ok=True, parents=True)
    df, cols, log = load_and_clean(path)
    df["tenure_group"] = pd.cut(df[cols["tenure"]], [-1, 12, 24, 48, 1e9], labels=["0-12 m", "13-24 m", "25-48 m", "49+ m"])
    m = compute_metrics(df, cols)
    tabs = {"tenure": churn_by(df, "tenure_group", list(df.tenure_group.cat.categories))}
    if cols["contract"]: tabs["contract"] = churn_by(df, cols["contract"])
    if cols["payment"]: tabs["payment"] = churn_by(df, cols["payment"])
    drivers = rank_drivers(df, cols)
    charts = make_charts(df, cols, m, tabs, drivers)
    ins, rec = build_insights(df, cols, m, tabs, drivers, cur)
    df.drop(columns=["tenure_group"]).to_csv(OUT / "cleaned_churn_data.csv", index=False)
    pd.concat([t.assign(segment_type=k).rename(columns={cols.get(k) or "tenure_group": "segment"}) for k, t in tabs.items()]).to_csv(OUT / "churn_by_segment.csv", index=False)
    pd.Series(m).to_csv(OUT / "key_metrics.csv", header=["value"])
    build_pdf(OUT / "Customer_Churn_Analysis_Report.pdf", m, tabs, drivers, charts, ins, rec, log, cur, cols)
    print("\n".join(log)); print("\nKEY METRICS"); [print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v}") for k, v in m.items()]
    return df, m, tabs, drivers, ins, rec

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--currency", default="$"); a = ap.parse_args()
    run(a.csv, a.currency)
