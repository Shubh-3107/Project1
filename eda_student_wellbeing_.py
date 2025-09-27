# Data cleaninh, processing and EDA

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (7.5, 4.5)

# ----------------------------
# Load
# ----------------------------
df = pd.read_csv("student_wellbeing_dataset.csv")  # attached file

# ----------------------------
# Data Exploration (audit)
# ----------------------------
print("Shape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nHead:")
print(df.head(5))

# Missing values
na_counts = df.isna().sum().sort_values(ascending=False)
na_pct = (df.isna().mean()*100).round(2)
print("\nMissing by column:\n", na_counts)
print("\nMissing % by column:\n", na_pct)

# Duplicates
dup_total = df.duplicated().sum()
dup_id = df.duplicated(subset=["Student_ID"]).sum()
print("\nExact duplicate rows:", dup_total)
print("Duplicate Student_ID:", dup_id)

# Optional: visualize missingness
plt.figure(figsize=(7,3))
sns.heatmap(df.isna(), cbar=False)
plt.title("Missingness Heatmap")
plt.tight_layout()
plt.show()

# ----------------------------
# Data Preprocessing
# ----------------------------
# Normalize categorical labels
if "Extracurricular" in df.columns:
    df["Extracurricular"] = df["Extracurricular"].astype(str).str.strip().str.title()
if "Stress_Level" in df.columns:
    df["Stress_Level"] = df["Stress_Level"].astype(str).str.strip().str.title()

# Coerce numerics
num_cols = ["Hours_Study","Sleep_Hours","Screen_Time","Attendance","CGPA"]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Range checks (report only)
ranges = {
    "Hours_Study": (0, 24),
    "Sleep_Hours": (0, 24),
    "Screen_Time": (0, 24),
    "Attendance": (0, 100),
    "CGPA": (0, 10),
}
for col, (lo, hi) in ranges.items():
    if col in df.columns:
        bad = df[(df[col] < lo) | (df[col] > hi)]
        print(f"{col}: out-of-range rows = {len(bad)}")

# Simple median imputations for numeric columns
for col in num_cols:
    if col in df.columns:
        med = df[col].median()
        df[col] = df[col].fillna(med)

# Encode (for correlation only; keep labels for plots)
if "Extracurricular" in df.columns:
    df["Extra_bin"] = df["Extracurricular"].map({"Yes":1, "No":0})
if "Stress_Level" in df.columns:
    df["Stress_ord"] = df["Stress_Level"].map({"Low":1,"Medium":2,"High":3})

# Drop exact duplicates if any
df = df.drop_duplicates()
print("Post-clean shape:", df.shape)

# ----------------------------
# EDA Task 1: Relations of study/sleep/screen with CGPA
# ----------------------------
corr_cols = [c for c in ["Hours_Study","Sleep_Hours","Screen_Time","Attendance","Stress_ord","Extra_bin","CGPA"] if c in df.columns]
corr = df[corr_cols].corr().round(3)
print("\nCorrelation with CGPA:\n", corr["CGPA"].sort_values(ascending=False))

plt.figure(figsize=(7.5,5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

for x in [c for c in ["Hours_Study","Sleep_Hours","Screen_Time","Attendance"] if c in df.columns]:
    ax = sns.regplot(data=df, x=x, y="CGPA", scatter_kws={"alpha":0.4}, line_kws={"color":"red"})
    ax.set_title(f"{x} vs CGPA")
    plt.tight_layout()
    plt.show()

# ----------------------------
# EDA Task 2: Compare CGPA across stress levels
# ----------------------------
if "Stress_Level" in df.columns:
    order = [g for g in ["Low","Medium","High"] if g in df["Stress_Level"].unique()]
    if len(order) >= 2:
        ax = sns.boxplot(data=df, x="Stress_Level", y="CGPA", order=order)
        sns.stripplot(data=df, x="Stress_Level", y="CGPA", order=order, color="k", alpha=0.15)
        ax.set_title("CGPA across Stress Levels")
        plt.tight_layout()
        plt.show()
        print("\nCGPA by Stress Level (group stats):")
        print(df.groupby("Stress_Level")["CGPA"].agg(["count","mean","median","std"]).round(3))

# ----------------------------
# Insights (from steps up to Task 2 only)
# ----------------------------
insights = []

# Correlation highlights (excluding CGPA with itself)
if "CGPA" in corr.columns:
    corr_series = corr["CGPA"].drop("CGPA").sort_values(ascending=False)
    # pick top 2 non-trivial correlates if available
    for feat in corr_series.index[:2]:
        insights.append(f"{feat} shows correlation r={corr_series.loc[feat]:.2f} with CGPA.")

# Directional notes for key continuous features
if "Hours_Study" in df.columns:
    r_study = df[["Hours_Study","CGPA"]].corr().iloc[0,1]
    dir_study = "positive" if r_study >= 0 else "negative"
    insights.append(f"Study hours vs CGPA correlation is {dir_study} at r={r_study:.2f}.")

if "Sleep_Hours" in df.columns:
    r_sleep = df[["Sleep_Hours","CGPA"]].corr().iloc[0,1]
    dir_sleep = "positive" if r_sleep >= 0 else "negative"
    insights.append(f"Sleep hours vs CGPA correlation is {dir_sleep} at r={r_sleep:.2f}.")

if "Screen_Time" in df.columns:
    r_screen = df[["Screen_Time","CGPA"]].corr().iloc[0,1]
    dir_screen = "negative" if r_screen < 0 else "positive"
    insights.append(f"Screen time vs CGPA correlation is {dir_screen} at r={r_screen:.2f}.")

# Stress levels comparison
if "Stress_Level" in df.columns:
    grp = df.groupby("Stress_Level")["CGPA"].mean()
    if {"Low","High"}.issubset(grp.index):
        diff = grp["Low"] - grp["High"]
        insights.append(f"Average CGPA in Low-stress exceeds High-stress by ≈ {diff:.2f} points.")

print("\nInsights (from Tasks 1–2):")
for i, s in enumerate(insights, 1):
    print(f"{i}. {s}")

