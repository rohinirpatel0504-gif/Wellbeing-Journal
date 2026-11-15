# AI-Powered Fraud Detection System

Streamlit app that scores card transactions for fraud risk using a hybrid of unsupervised
(Isolation Forest) and supervised (Random Forest) machine learning, with plain-language explanations.

## Run

In VS Code, open **Terminal → New Terminal** and run these PowerShell commands:

```powershell
cd "C:\Users\Rohini R Patel\Downloads\fraud_detection\fraud_detection"
pip install -r requirements.txt
streamlit run app.py
```

The app trains on synthetic history at start-up (a few seconds) and analyses a separate, unseen batch of transactions.

## MongoDB

The app loads `MONGODB_URI` from a `.env` file in the project directory (or from the process environment) and
connects to `mongodb://127.0.0.1:27017` if it is not set. For local MongoDB, install MongoDB Community Server
for Windows and make sure its **MongoDB** service is running before saving transactions. In PowerShell, you can
check and start the service with:

```powershell
Get-Service MongoDB
Start-Service MongoDB
```

In the running app, click **Save analyzed transactions to MongoDB** in the sidebar. This writes the current
scored batch to the `fraud_detection.transactions` collection; it does not save automatically. For MongoDB Atlas,
set `MONGODB_URI` in `.env` to the connection string from Atlas and ensure the database user and network access
list are configured. Never commit `.env` or credentials to source control.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface (dashboard, flagged list, single-transaction checker, model explanation) |
| `fraud_model.py` | Feature engineering, hybrid model, risk scoring, explanations |
| `data_generator.py` | Synthetic transaction data with injected fraud patterns |
| `sample_transactions.csv` | Sample dataset (also usable via the "Upload CSV" option) |

## CSV format for uploads

`txn_id, customer_id, timestamp, merchant_category, amount, country, card_present` (optional: `is_fraud`)

## Demo script

1. **Dashboard** - totals, amount at risk, risk-score histogram, flags by hour/merchant.
2. **Flagged transactions** - sorted by risk, with reasons; download as CSV. Move the sidebar threshold to see the trade-off.
3. **Check a transaction** - the default inputs (₹45,000, 2 AM, crypto exchange, Russia, card not present) score as Critical; set India / card present / Grocery / ₹800 and it drops to Low.
4. **Model & AI explained** - precision, recall, F1, ROC-AUC, confusion matrix, feature importance.
