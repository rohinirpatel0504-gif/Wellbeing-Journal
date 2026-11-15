"""Feature engineering + hybrid AI model for fraud detection.

Two complementary models are combined:
  * Isolation Forest  (unsupervised) - flags statistically unusual transactions,
    so it can catch NEW fraud patterns with no labels.
  * Random Forest     (supervised)   - learns known fraud patterns from labelled history.
The final risk score is a weighted blend of both (0-100).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split

RISKY_MERCHANTS = {"Electronics", "Jewellery", "Crypto Exchange", "Online Retail"}
HIGH_RISK_COUNTRIES = {"Nigeria", "Russia", "Brazil"}

FEATURES = ["amount", "amount_ratio", "amount_zscore", "hour", "is_night", "is_foreign",
            "high_risk_country", "card_not_present", "risky_merchant",
            "tx_last_10min", "mins_since_last_tx"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    g = d.groupby("customer_id")["amount"]
    d["cust_mean"] = g.transform("mean")
    d["cust_std"] = g.transform("std").fillna(1).replace(0, 1)
    d["amount_ratio"] = d["amount"] / d["cust_mean"]
    d["amount_zscore"] = (d["amount"] - d["cust_mean"]) / d["cust_std"]

    d["hour"] = d["timestamp"].dt.hour
    d["is_night"] = d["hour"].between(0, 4).astype(int)
    d["is_foreign"] = (d["country"] != "India").astype(int)
    d["high_risk_country"] = d["country"].isin(HIGH_RISK_COUNTRIES).astype(int)
    d["card_not_present"] = (1 - d["card_present"]).astype(int)
    d["risky_merchant"] = d["merchant_category"].isin(RISKY_MERCHANTS).astype(int)

    # velocity features
    gap = d.groupby("customer_id")["timestamp"].diff().dt.total_seconds() / 60
    d["mins_since_last_tx"] = gap.fillna(60 * 24 * 7).clip(upper=60 * 24 * 7)
    d["tx_last_10min"] = ((d["mins_since_last_tx"] <= 10).astype(int)
                          + (d.groupby("customer_id")["mins_since_last_tx"].shift(1).fillna(1e6) <= 10).astype(int))
    return d


class FraudDetector:
    def __init__(self, seed=42):
        self.seed = seed
        self.iso = IsolationForest(n_estimators=200, contamination=0.04, random_state=seed)
        self.rf = RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample",
                                         min_samples_leaf=2, random_state=seed, n_jobs=-1)
        self._iso_min = self._iso_max = None
        self.metrics = {}

    def _iso_score(self, X):
        raw = -self.iso.score_samples(X)          # higher = more anomalous
        return np.clip((raw - self._iso_min) / (self._iso_max - self._iso_min + 1e-9), 0, 1)

    def fit(self, df: pd.DataFrame):
        d = engineer_features(df)
        X, y = d[FEATURES], d["is_fraud"]
        Xtr, Xte, ytr, yte, _, idx_te = train_test_split(
            X, y, d.index, test_size=0.25, stratify=y, random_state=self.seed)

        self.rf.fit(Xtr, ytr)
        self.iso.fit(Xtr)
        raw = -self.iso.score_samples(Xtr)
        self._iso_min, self._iso_max = raw.min(), raw.max()

        score = self.score(Xte)
        pred = (score >= 50).astype(int)
        self.metrics = {
            "precision": precision_score(yte, pred, zero_division=0),
            "recall": recall_score(yte, pred, zero_division=0),
            "f1": f1_score(yte, pred, zero_division=0),
            "roc_auc": roc_auc_score(yte, score),
            "confusion": confusion_matrix(yte, pred).tolist(),
            "report": classification_report(yte, pred, target_names=["Genuine", "Fraud"], zero_division=0),
            "n_test": int(len(yte)),
        }
        self.importances = pd.Series(self.rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Risk score 0-100 = 70% supervised probability + 30% anomaly score."""
        p = self.rf.predict_proba(X[FEATURES])[:, 1]
        a = self._iso_score(X[FEATURES])
        return 100 * (0.7 * p + 0.3 * a)

    def score_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        d = engineer_features(df)
        d["risk_score"] = self.score(d)
        d["risk_level"] = pd.cut(d["risk_score"], [-1, 30, 50, 70, 101],
                                 labels=["Low", "Medium", "High", "Critical"])
        d["reasons"] = d.apply(explain, axis=1)
        return d


def explain(r) -> str:
    """Human-readable reasons for a flag (rule-based explanation layer)."""
    why = []
    if r["amount_ratio"] > 5:
        why.append(f"amount is {r['amount_ratio']:.1f}x the customer's average")
    if r["is_night"] and r["amount_ratio"] > 2:
        why.append("high-value transaction at night")
    if r["high_risk_country"]:
        why.append(f"high-risk country ({r['country']})")
    elif r["is_foreign"]:
        why.append(f"foreign transaction ({r['country']})")
    if r["tx_last_10min"] >= 1:
        why.append("rapid repeat transactions (velocity)")
    if r["card_not_present"] and r["risky_merchant"]:
        why.append("card-not-present at risky merchant")
    return "; ".join(why) if why else "pattern deviates from customer's normal behaviour"
