"""AI-Powered Fraud Detection System - Streamlit UI.
Run:  streamlit run app.py
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from data_generator import generate_transactions, FOREIGN, MERCHANTS
from fraud_model import FraudDetector, engineer_features, explain, FEATURES

load_dotenv(Path(__file__).with_name(".env"))

st.set_page_config(page_title="AI Fraud Detection", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Training AI models on historical transactions...")
def train(seed: int = 42):
    hist = generate_transactions(seed=seed)
    return FraudDetector(seed).fit(hist), hist


@st.cache_resource(show_spinner=False)
def get_mongo_collection(uri: str):
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except (PyMongoError, ValueError):
        client.close()
        raise
    return client["fraud_detection"]["transactions"]


detector, history = train()

# ---------------- Sidebar ----------------
st.sidebar.title("🛡️ Fraud Detection")
src = st.sidebar.radio("Transactions to analyse", ["Sample data", "Upload CSV"])
threshold = st.sidebar.slider("Alert threshold (risk score)", 10, 90, 50, 5,
                              help="Transactions at or above this score are flagged for review.")
if src == "Upload CSV":
    up = st.sidebar.file_uploader("CSV with columns: txn_id, customer_id, timestamp, merchant_category, "
                                  "amount, country, card_present", type="csv")
    if up is None:
        st.info("Upload a CSV or switch to Sample data.")
        st.stop()
    raw = pd.read_csv(up)
    if "is_fraud" not in raw.columns:
        raw["is_fraud"] = 0
else:
    raw = generate_transactions(seed=7)  # unseen data (different seed from training)

scored = detector.score_transactions(raw)
scored["flagged"] = scored["risk_score"] >= threshold
flagged = scored[scored["flagged"]].sort_values("risk_score", ascending=False)

if st.sidebar.button("Save analyzed transactions to MongoDB"):
    mongo_uri = os.environ.get("MONGODB_URI", "mongodb://127.0.0.1:27017")
    try:
        collection = get_mongo_collection(mongo_uri)
        documents = json.loads(scored.to_json(orient="records", date_format="iso"))
        saved_at = datetime.now(timezone.utc)
        for document in documents:
            document["timestamp"] = pd.to_datetime(
                document["timestamp"], utc=True
            ).to_pydatetime()
            document["source"] = src
            document["risk_threshold"] = threshold
            document["saved_at"] = saved_at
        result = collection.insert_many(documents)
        st.sidebar.success(f"Saved {len(result.inserted_ids):,} transactions.")
    except (PyMongoError, ValueError) as exc:
        st.sidebar.error(
            f"MongoDB save failed ({type(exc).__name__}). Verify the server is reachable and the "
            "URI credentials and network access are configured correctly."
        )

st.title("AI-Powered Fraud Detection System")
st.caption("Hybrid model: Isolation Forest (anomaly detection) + Random Forest (supervised learning)")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🚨 Flagged transactions",
                                  "🔎 Check a transaction", "🧠 Model & AI explained"])

# ---------------- Dashboard ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions analysed", f"{len(scored):,}")
    c2.metric("Flagged for review", f"{len(flagged):,}", f"{len(flagged)/len(scored):.1%} of total")
    c3.metric("Amount at risk", f"₹{flagged['amount'].sum():,.0f}")
    c4.metric("Critical alerts", int((scored["risk_level"] == "Critical").sum()))

    a, b = st.columns(2)
    with a:
        st.subheader("Risk score distribution")
        st.altair_chart(alt.Chart(scored).mark_bar().encode(
            x=alt.X("risk_score:Q", bin=alt.Bin(maxbins=30), title="Risk score"),
            y=alt.Y("count()", title="Transactions"),
            color=alt.condition(alt.datum.risk_score >= threshold, alt.value("#d62728"), alt.value("#4c78a8")),
        ).properties(height=300), width='stretch')
    with b:
        st.subheader("Flagged transactions by hour")
        by_hour = flagged.groupby("hour").size().reindex(range(24), fill_value=0).reset_index(name="flagged")
        st.altair_chart(alt.Chart(by_hour).mark_bar(color="#d62728").encode(
            x=alt.X("hour:O", title="Hour of day"), y=alt.Y("flagged:Q", title="Flagged")
        ).properties(height=300), width='stretch')

    a, b = st.columns(2)
    with a:
        st.subheader("Flag rate by merchant category")
        m = scored.groupby("merchant_category")["flagged"].mean().reset_index()
        st.altair_chart(alt.Chart(m).mark_bar().encode(
            x=alt.X("flagged:Q", axis=alt.Axis(format="%"), title="Flag rate"),
            y=alt.Y("merchant_category:N", sort="-x", title=None)).properties(height=300),
            width='stretch')
    with b:
        st.subheader("Amount vs customer's usual spend")
        samp = scored.sample(min(2500, len(scored)), random_state=1)
        st.altair_chart(alt.Chart(samp).mark_circle(size=40, opacity=.6).encode(
            x=alt.X("amount_ratio:Q", scale=alt.Scale(type="log"), title="Amount ÷ customer average (log)"),
            y=alt.Y("risk_score:Q", title="Risk score"),
            color=alt.Color("flagged:N", scale=alt.Scale(range=["#4c78a8", "#d62728"]), title="Flagged"),
            tooltip=["txn_id", "amount", "merchant_category", "country", "risk_score"]
        ).properties(height=300), width='stretch')

# ---------------- Flagged ----------------
with tab2:
    st.subheader(f"{len(flagged)} transactions need review")
    level = st.multiselect("Risk level", ["Medium", "High", "Critical"], default=["High", "Critical"])
    view = flagged[flagged["risk_level"].isin(level)]
    st.dataframe(view[["txn_id", "customer_id", "timestamp", "merchant_category", "amount", "country",
                       "card_present", "risk_score", "risk_level", "reasons"]]
                 .style.format({"risk_score": "{:.0f}", "amount": "₹{:,.0f}"}),
                 width='stretch', height=480, hide_index=True)
    st.download_button("Download flagged transactions (CSV)", view.to_csv(index=False),
                       "flagged_transactions.csv", "text/csv")

# ---------------- Live check ----------------
with tab3:
    st.subheader("Score a single transaction")
    st.caption("Enter a transaction below. Compared against the customer's typical spend of ₹1,500 (demo).")
    c1, c2, c3 = st.columns(3)
    amt = c1.number_input("Amount (₹)", 1.0, 1_000_000.0, 45_000.0, step=500.0)
    hr = c2.slider("Hour of day", 0, 23, 2)
    merch = c3.selectbox("Merchant category", MERCHANTS, index=MERCHANTS.index("Crypto Exchange"))
    c1, c2, c3 = st.columns(3)
    ctry = c1.selectbox("Country", ["India"] + FOREIGN, index=6)
    cp = c2.selectbox("Card present?", ["Yes", "No"], index=1)
    burst = c3.selectbox("Transactions in the last 10 min", [0, 1, 2], index=1)

    avg = 1500.0
    row = pd.DataFrame([{
        "amount": amt, "amount_ratio": amt / avg, "amount_zscore": (amt - avg) / 800, "hour": hr,
        "is_night": int(hr <= 4), "is_foreign": int(ctry != "India"),
        "high_risk_country": int(ctry in {"Nigeria", "Russia", "Brazil"}),
        "card_not_present": int(cp == "No"),
        "risky_merchant": int(merch in {"Electronics", "Jewellery", "Crypto Exchange", "Online Retail"}),
        "tx_last_10min": burst, "mins_since_last_tx": 5 if burst else 2000,
        "country": ctry}])
    s = float(detector.score(row)[0])
    level = "Critical" if s > 70 else "High" if s > 50 else "Medium" if s > 30 else "Low"
    colour = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}[level]
    st.markdown(f"### {colour} Risk score: **{s:.0f}/100** ({level})")
    st.progress(min(int(s), 100))
    if s >= threshold:
        st.error("Recommended action: HOLD transaction and trigger OTP / customer verification.")
    else:
        st.success("Recommended action: Approve.")
    st.write("**Why:**", explain(row.iloc[0]))

# ---------------- Model ----------------
with tab4:
    st.subheader("How AI is used in this project")
    st.markdown("""
1. **Feature engineering** – raw transactions are converted into behavioural signals: amount relative to the
   customer's own average, z-score, night-time flag, foreign / high-risk country, card-not-present, merchant risk
   and **velocity** (bursts of transactions in minutes).
2. **Unsupervised AI (Isolation Forest)** – learns what "normal" looks like and isolates rare, unusual
   transactions. Works even when no fraud labels exist, so it can catch *new* fraud types.
3. **Supervised AI (Random Forest)** – trained on labelled history to recognise known fraud patterns.
4. **Hybrid risk score (0–100)** = 70% Random Forest probability + 30% anomaly score.
5. **Explainability layer** – every alert comes with plain-language reasons so analysts can act quickly.
""")
    m = detector.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{m['precision']:.1%}")
    c2.metric("Recall", f"{m['recall']:.1%}")
    c3.metric("F1-score", f"{m['f1']:.1%}")
    c4.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    st.caption(f"Evaluated on a held-out test set of {m['n_test']:,} transactions at threshold 50.")

    a, b = st.columns(2)
    with a:
        st.subheader("Confusion matrix")
        cm = pd.DataFrame(m["confusion"], index=["Actual genuine", "Actual fraud"],
                          columns=["Predicted genuine", "Predicted fraud"])
        st.dataframe(cm, width='stretch')
    with b:
        st.subheader("Feature importance")
        imp = detector.importances.reset_index()
        imp.columns = ["feature", "importance"]
        st.altair_chart(alt.Chart(imp).mark_bar().encode(
            x="importance:Q", y=alt.Y("feature:N", sort="-x", title=None)).properties(height=300),
            width='stretch')
    st.caption("Note: data is synthetic and illustrative. Real-world performance would need real labelled transactions.")
