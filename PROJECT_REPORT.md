# AI-Powered Fraud Detection System
**Course:** AI for Finance - CIA-2 | **Project option:** 3 - AI-Powered Fraud Detection System

## 1. Problem Statement
Digital payments are growing rapidly, and so is card and online fraud. Banks cannot manually review millions of transactions, and fixed rules ("block anything above ₹50,000") either miss clever fraud or block genuine customers. This project builds an AI system that scores every transaction for fraud risk by learning each customer's normal behaviour and the patterns fraudsters leave behind.

## 2. Objectives
- Analyse financial transactions and detect unusual or potentially fraudulent activity.
- Give each transaction a risk score (0-100) and a plain-language reason.
- Provide a user-friendly dashboard for fraud analysts.
- Demonstrate the system with sample data and measure its accuracy.

## 3. How AI Is Used
| Stage | Technique | Role |
|---|---|---|
| Feature engineering | Behavioural profiling | Compares each transaction with the customer's own average (amount ratio, z-score) and adds time, geography, channel, merchant-risk and velocity features |
| Anomaly detection | **Isolation Forest** (unsupervised) | Learns what "normal" looks like and isolates rare transactions; can catch new fraud types without labels |
| Classification | **Random Forest** (supervised) | Learns known fraud patterns from labelled history; class-weighted for the heavy imbalance (about 3-4% fraud) |
| Risk scoring | Hybrid ensemble | Risk score = 70% Random Forest probability + 30% anomaly score |
| Explainability | Rule-based reason generator + feature importance | Tells analysts *why* a transaction was flagged |

**AI-assisted development:** Claude (Anthropic) was used as an AI coding assistant to design the architecture, generate the code, create the synthetic dataset, test the pipeline and draft this report.

## 4. Features
1. **Dashboard** - transactions analysed, number flagged, amount at risk, critical alerts, risk-score distribution, flags by hour and merchant category.
2. **Flagged transactions** - sortable list with risk level (Medium / High / Critical), reasons, and CSV download.
3. **Adjustable alert threshold** - lets the bank trade off missed fraud against false alarms.
4. **Single-transaction checker** - enter a transaction and get an instant score, reason and recommended action (approve / hold and verify).
5. **CSV upload** - analyse your own transaction file.
6. **Model page** - precision, recall, F1, ROC-AUC, confusion matrix, feature importance.

## 5. Fraud Patterns Detected (in the sample data)
1. Amounts far above the customer's usual spend.
2. High-value, card-not-present activity at night (00:00-04:59).
3. Transactions from foreign / high-risk countries.
4. Rapid bursts of repeated transactions (velocity attacks) at risky merchants.

## 6. Data
Synthetic data was generated for 300 customers over 90 days (about 12,000 transactions, about 4.4% fraud after injected burst patterns). Each customer has their own typical spend. Fraud was injected using the four patterns above. Real bank data is confidential, so synthetic data is standard for a prototype; results on real data would differ.

## 7. Results
Evaluated on a held-out 25% test set (3,045 transactions) at an alert threshold of 50:

| Metric | Value |
|---|---|
| Precision | 93.8% |
| Recall | 89.6% |
| F1-score | 91.7% |
| ROC-AUC | 0.999 |
| Confusion matrix | 2,902 genuine approved, 8 false alarms, 14 missed frauds, 121 frauds caught |

Most important features: amount relative to customer average (about 32%), amount z-score (about 29%), raw amount (about 16%), card-not-present (about 8%).

*Caveat:* the high scores partly reflect that the synthetic fraud patterns are cleaner than real fraud. Real-world deployment would need real labelled data, ongoing retraining and monitoring.

## 8. Practical Applications and Benefits
- **Banks and card issuers:** real-time screening of card and UPI transactions, triggering OTP or call-back for risky ones.
- **Payment gateways and e-commerce:** reducing chargebacks and merchant losses.
- **Fraud-analyst teams:** prioritised alert queue with reasons, which cuts manual review time.
- **Customers:** fewer false declines than blunt rules, and faster protection of their money.
- **Benefits:** personalised per-customer baselines, detection of new patterns through anomaly detection, adjustable sensitivity and transparent explanations.

## 9. Limitations and Future Scope
- Synthetic data; to be validated on real transaction data.
- Add graph-based detection of fraud rings and device / IP features.
- Use SHAP for deeper explanations, and online learning that adapts as fraud changes.
- Handle concept drift and fairness checks; integrate with a real-time streaming API.

## 10. Tools and Technologies
Python, pandas, NumPy, scikit-learn (Isolation Forest, Random Forest), Altair, Streamlit; Claude as the AI development assistant.

## 11. How to Run
`pip install -r requirements.txt` then `streamlit run app.py` (see README.md).
