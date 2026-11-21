"""Run Exploratory Data Analysis (EDA) on STATS19 data.

Generates plots and saves them to reports/figures/.
Handles both legacy ('accident_*') and modern ('collision_*') schemas.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RAW_CSV = Path("dataset/raw/collision_last_5_years.csv")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def load_and_normalize() -> pd.DataFrame:
    """Load raw CSV and normalize schema to 'collision_*' naming."""
    print(f"Loading {RAW_CSV}...")
    df = pd.read_csv(RAW_CSV, low_memory=False)

    # Normalize legacy schema to modern schema
    rename_map = {
        "accident_index": "collision_index",
        "accident_severity": "collision_severity",
        "accident_year": "collision_year",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Map target labels
    severity_map = {1: "Fatal", 2: "Serious", 3: "Slight"}
    if "collision_severity" in df.columns:
        df["severity_label"] = df["collision_severity"].map(severity_map)
    else:
        print("WARNING: Target column 'collision_severity' not found.")

    return df


def run_eda(df: pd.DataFrame) -> None:
    print("\n--- Dataset Overview ---")
    print(f"Shape: {df.shape}")
    print(f"Duplicates: {df.duplicated(subset=['collision_index']).sum()} rows")
    
    # 1. Target Distribution (Bar Chart)
    plt.figure()
    sns.countplot(data=df, x="severity_label", order=["Slight", "Serious", "Fatal"], palette="viridis")
    plt.title("Distribution of Accident Severity (Target)")
    plt.ylabel("Count")
    plt.savefig(FIG_DIR / "01_target_distribution.png", bbox_inches="tight")
    plt.close()

    # 2. Missing Values (Bar Chart)
    plt.figure(figsize=(12, 8))
    missing = df.isnull().sum().sort_values(ascending=False).head(20)
    sns.barplot(x=missing.values, y=missing.index, palette="magma")
    plt.title("Top 20 Features with Missing Values")
    plt.xlabel("Number of Missing Records")
    plt.savefig(FIG_DIR / "02_missing_values.png", bbox_inches="tight")
    plt.close()

    # 3. Speed Limit vs Severity (Boxplot)
    if "speed_limit" in df.columns:
        plt.figure()
        sns.boxplot(data=df, x="severity_label", y="speed_limit", order=["Slight", "Serious", "Fatal"], palette="Set2")
        plt.title("Speed Limit vs Accident Severity")
        plt.savefig(FIG_DIR / "03_speed_vs_severity.png", bbox_inches="tight")
        plt.close()

    # 4. Weather Conditions (Pie Chart)
    if "weather_conditions" in df.columns:
        plt.figure()
        weather_counts = df["weather_conditions"].value_counts().head(5)
        weather_counts.plot.pie(autopct="%1.1f%%", startangle=90, cmap="tab20c")
        plt.title("Top 5 Weather Conditions")
        plt.ylabel("")
        plt.savefig(FIG_DIR / "04_weather_pie.png", bbox_inches="tight")
        plt.close()

    # 5. Time Series (Monthly Counts)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month_year"] = df["date"].dt.to_period("M")
        monthly_counts = df.groupby("month_year").size()
        
        plt.figure(figsize=(12, 6))
        monthly_counts.plot(kind="line", marker="o", color="b")
        plt.title("Accidents Over Time (Monthly)")
        plt.xlabel("Month")
        plt.ylabel("Number of Accidents")
        plt.xticks(rotation=45)
        plt.savefig(FIG_DIR / "05_time_series.png", bbox_inches="tight")
        plt.close()

    # 6. Geographical Distribution (Scatter Map)
    if "longitude" in df.columns and "latitude" in df.columns:
        # Filter out invalid coordinates (UK bounding box approx)
        geo_df = df.dropna(subset=["longitude", "latitude"])
        geo_df = geo_df[(geo_df["longitude"] > -8) & (geo_df["longitude"] < 2) & 
                        (geo_df["latitude"] > 50) & (geo_df["latitude"] < 59)]
        
        plt.figure(figsize=(8, 10))
        sns.scatterplot(
            data=geo_df.sample(min(10000, len(geo_df)), random_state=42), 
            x="longitude", y="latitude", hue="severity_label", 
            palette={"Slight": "blue", "Serious": "orange", "Fatal": "red"},
            alpha=0.4, s=10
        )
        plt.title("Geographical Distribution of Accidents (UK)")
        plt.savefig(FIG_DIR / "06_geo_map.png", bbox_inches="tight")
        plt.close()

    # 7. Correlation Heatmap (Numerical Features)
    num_cols = ["speed_limit", "longitude", "latitude", "collision_severity"]
    available_nums = [c for c in num_cols if c in df.columns]
    
    if len(available_nums) > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[available_nums].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap (Numerical Features)")
        plt.savefig(FIG_DIR / "07_correlation_heatmap.png", bbox_inches="tight")
        plt.close()

    print(f"\n[success] All EDA plots saved to {FIG_DIR}/")


if __name__ == "__main__":
    df = load_and_normalize()
    run_eda(df)