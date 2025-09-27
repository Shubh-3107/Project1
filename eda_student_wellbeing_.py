#Project 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (7.5, 4.5)

# Load
df = pd.read_csv(r"student_wellbeing_dataset.csv")

# Data Exploration
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

# Missingness heatmap
plt.figure(figsize=(7,3))
sns.heatmap(df.isna(), cbar=False)
plt.title("Missingness Heatmap")
plt.tight_layout()
plt.show()


# Data Preprocessing
# Normalize categorical labels
if "Extracurricular" in df.columns:
    df["Extracurricular"] = df["Extracurricular"].astype(str).str.strip().str.title()
if "Stress_Level" in df.columns:
    df["Stress_Level"] = df["Stress_Level"].astype(str).str.strip().str.title()

# Numeric columns
num_cols = ["Hours_Study","Sleep_Hours","Screen_Time","Attendance","CGPA"]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Range checks
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

# Impute medians for numeric
for col in num_cols:
    if col in df.columns:
        med = df[col].median()
        df[col] = df[col].fillna(med)

# Ordinal encode Stress_Level for correlation (keep labels for plots)
if "Stress_Level" in df.columns:
    stress_map = {"Low":1, "Medium":2, "High":3}
    df["Stress_ord"] = df["Stress_Level"].map(stress_map)

# Drop exact duplicates
df = df.drop_duplicates()
print("Post-clean shape:", df.shape)

# EDA Part 1: Relationships with CGPA
corr_cols = [c for c in ["Hours_Study","Sleep_Hours","Screen_Time","Attendance","Stress_ord","CGPA"] if c in df.columns]
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

# EDA Part 2: CGPA across Stress Levels
if "Stress_Level" in df.columns:
    order = [g for g in ["Low","Medium","High"] if g in df["Stress_Level"].unique()]
    if len(order) >= 2:
        ax = sns.boxplot(data=df, x="Stress_Level", y="CGPA", order=order)
        sns.stripplot(data=df, x="Stress_Level", y="CGPA", order=order, color="k", alpha=0.15)
        ax.set_title("CGPA across Stress Levels")
        plt.tight_layout()
        plt.show()
        print(df.groupby("Stress_Level")["CGPA"].agg(["count","mean","median","std"]).round(3))

fig, axes = plt.subplots(2, 2, figsize=(12,8))
sns.regplot(data=df, x="Hours_Study", y="CGPA", ax=axes[0,0])
sns.regplot(data=df, x="Sleep_Hours", y="CGPA", ax=axes[0,1])
sns.regplot(data=df, x="Screen_Time", y="CGPA", ax=axes[1,0])
sns.regplot(data=df, x="Attendance", y="CGPA", ax=axes[1,1])
plt.tight_layout()
plt.show()
