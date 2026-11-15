"""Synthetic card-transaction generator with realistic fraud patterns.

Fraud patterns injected:
  1. Large amounts far above the customer's normal spend
  2. Night-time (00:00-04:59) high-value activity
  3. Foreign / high-risk country transactions
  4. Rapid bursts (many transactions within minutes)
  5. Card-not-present purchases at risky merchant categories
"""
import numpy as np
import pandas as pd

MERCHANTS = ["Grocery", "Fuel", "Restaurant", "Electronics", "Travel",
             "Online Retail", "Utilities", "Entertainment", "Jewellery", "Crypto Exchange"]
RISKY = {"Electronics", "Jewellery", "Crypto Exchange", "Online Retail"}
COUNTRIES = ["India"] * 1  # home country
FOREIGN = ["USA", "UK", "UAE", "Singapore", "Nigeria", "Russia", "Brazil"]


def generate_transactions(n_customers=300, n_tx=12000, fraud_rate=0.03, seed=42):
    rng = np.random.default_rng(seed)
    cust_ids = np.arange(1000, 1000 + n_customers)
    cust_avg = rng.lognormal(mean=7.0, sigma=0.5, size=n_customers)  # avg spend in INR
    cust_map = dict(zip(cust_ids, cust_avg))

    start = pd.Timestamp("2026-08-01")
    rows = []
    for i in range(n_tx):
        cid = rng.choice(cust_ids)
        ts = start + pd.Timedelta(seconds=int(rng.integers(0, 90 * 24 * 3600)))
        # genuine customers mostly transact 7am-11pm
        hour = int(rng.choice(np.arange(24), p=_hour_probs()))
        ts = ts.replace(hour=hour, minute=int(rng.integers(0, 60)))
        merchant = rng.choice(MERCHANTS, p=[.22, .12, .16, .06, .05, .14, .1, .08, .04, .03])
        amount = float(rng.lognormal(np.log(cust_map[cid]), 0.6))
        country = "India" if rng.random() < 0.97 else rng.choice(FOREIGN[:4])
        card_present = int(rng.random() < 0.65)
        rows.append([f"T{i:06d}", cid, ts, merchant, round(amount, 2), country, card_present, 0])

    df = pd.DataFrame(rows, columns=["txn_id", "customer_id", "timestamp", "merchant_category",
                                     "amount", "country", "card_present", "is_fraud"])

    # inject fraud
    n_fraud = int(n_tx * fraud_rate)
    idx = rng.choice(df.index, n_fraud, replace=False)
    for j, k in enumerate(idx):
        pattern = j % 4
        cid = df.at[k, "customer_id"]
        base = cust_map[cid]
        df.at[k, "is_fraud"] = 1
        if pattern == 0:      # unusually large
            df.at[k, "amount"] = round(base * rng.uniform(8, 30), 2)
            df.at[k, "merchant_category"] = rng.choice(list(RISKY))
        elif pattern == 1:    # night time, card not present
            t = df.at[k, "timestamp"].replace(hour=int(rng.integers(0, 5)))
            df.at[k, "timestamp"] = t
            df.at[k, "amount"] = round(base * rng.uniform(4, 12), 2)
            df.at[k, "card_present"] = 0
        elif pattern == 2:    # foreign high-risk country
            df.at[k, "country"] = rng.choice(FOREIGN[4:])
            df.at[k, "amount"] = round(base * rng.uniform(3, 10), 2)
            df.at[k, "card_present"] = 0
        else:                 # burst: duplicate quick transactions
            df.at[k, "amount"] = round(base * rng.uniform(2, 6), 2)
            df.at[k, "merchant_category"] = rng.choice(list(RISKY))
            df.at[k, "card_present"] = 0
            for b in range(2):
                extra = df.loc[k].copy()
                extra["txn_id"] = f"B{k}_{b}"
                extra["timestamp"] = df.at[k, "timestamp"] + pd.Timedelta(minutes=int(rng.integers(1, 6)) * (b + 1))
                df.loc[len(df)] = extra

    return df.sort_values("timestamp").reset_index(drop=True)


def _hour_probs():
    p = np.array([0.5] * 5 + [1] * 2 + [4, 5, 5, 5, 5, 6, 6, 6, 5, 5, 5, 6, 7, 7, 6, 4, 2][:17], dtype=float)
    p = p[:24] if len(p) >= 24 else np.pad(p, (0, 24 - len(p)), constant_values=1)
    return p / p.sum()


if __name__ == "__main__":
    d = generate_transactions()
    d.to_csv("sample_transactions.csv", index=False)
    print(d.shape, "fraud:", d.is_fraud.sum())
