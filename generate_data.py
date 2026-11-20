"""Generates dataset 2 (synthetic, seeded, reproducible). Not for fraud claims.

This dataset differs from dataset 1: different seed, more rows, different
cities and channel mix, more credit-heavy activity, different amount scale,
and different missing/duplicate counts.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(777)
n = 1500

cities = ["Ahmedabad", "Lucknow", "Kochi", "Chandigarh", "Delhi", "Mumbai", "Bengaluru", "Pune"]
city_p = [.12, .10, .08, .09, .20, .17, .14, .10]
channels = ["ATM", "Mobile App", "Net Banking", "Branch", "POS", "UPI"]
ch_p = [.15, .22, .10, .12, .20, .21]
accounts = ["Savings", "Current", "Salary", "Business"]
ac_p = [.38, .17, .30, .15]

txn_type = rng.choice(["Credit", "Debit"], n, p=[.52, .48])
account_type = rng.choice(accounts, n, p=ac_p)
channel = rng.choice(channels, n, p=ch_p)
city = rng.choice(cities, n, p=city_p)

base = rng.lognormal(mean=7.9, sigma=0.9, size=n)
mult = pd.Series(account_type).map(
    {"Savings": 1.0, "Current": 1.6, "Salary": 1.4, "Business": 2.2}).values
credit_boost = np.where(txn_type == "Credit", 1.25, 1.0)
amt = base * mult * credit_boost

amt[channel == "ATM"] = np.minimum(amt[channel == "ATM"], 50_000)
amt[channel == "POS"] = np.minimum(amt[channel == "POS"], 80_000)
amt[channel == "UPI"] = np.minimum(amt[channel == "UPI"], 100_000)

big = rng.choice(n, 18, replace=False)
amt[big] = rng.uniform(200_000, 750_000, 18)
amt = np.round(amt, 2)

risk = (0.03 + 0.08 * (amt > 120_000) + 0.05 * (channel == "UPI")
        + 0.03 * (channel == "POS") + 0.02 * (city == "Delhi"))
flagged = np.where(rng.random(n) < risk, "Yes", "No")

dates = pd.to_datetime("2026-03-01") + pd.to_timedelta(rng.integers(0, 240 * 24 * 60, n), unit="m")

df = pd.DataFrame({
    "Transaction_ID": [f"BNK{500000 + i}" for i in range(n)],
    "Transaction_Date": dates.strftime("%Y-%m-%d %H:%M"),
    "Account_ID": [f"AC{rng.integers(2000, 2600)}" for _ in range(n)],
    "Account_Type": account_type,
    "Transaction_Type": txn_type,
    "Channel": channel,
    "City": city,
    "Amount": amt,
    "Flagged": flagged,
})
df = df.sort_values("Transaction_Date").reset_index(drop=True)

df.loc[rng.choice(n, 37, replace=False), "Amount"] = np.nan          # missing amounts
dups = df.sample(12, random_state=5)                                  # repeated IDs
df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=9).reset_index(drop=True)

out = Path(__file__).resolve().parent.parent / "data" / "10_bank_transactions.csv"
df.to_csv(out, index=False)
print(df.shape, "missing:", int(df["Amount"].isna().sum()),
      "dup IDs:", int(df["Transaction_ID"].duplicated().sum()),
      "flagged:", int((df["Flagged"] == "Yes").sum()))
