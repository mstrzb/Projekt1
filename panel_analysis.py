"""Panel data-quality checks, projection and assortment metrics (SQL + pandas).

Pipeline
    1. Load the CSVs into an in-memory SQLite database.
    2. Run data-quality checks in SQL (duplicates, bad values, orphan keys).
    3. Clean the sales table and flag product-weeks that look wrong
       (robust rolling-median outlier detection).
    4. Compute expansion factors (universe stores / panel stores per stratum)
       and project the panel to the market.
    5. Calculate assortment metrics per product: projected value, share,
       ABC class, numeric / weighted distribution and sales velocity.

Usage:
    python generate_data.py
    python panel_analysis.py
"""
import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DQ_CHECKS = {
    "duplicate_rows": """
        SELECT COALESCE(SUM(c - 1), 0) FROM (
            SELECT COUNT(*) AS c FROM sales GROUP BY store_id, product_id, week HAVING COUNT(*) > 1)""",
    "units_without_value": "SELECT COUNT(*) FROM sales WHERE units > 0 AND value <= 0",
    "negative_units": "SELECT COUNT(*) FROM sales WHERE units < 0",
    "orphan_products": """
        SELECT COUNT(*) FROM sales s LEFT JOIN products p USING (product_id)
        WHERE p.product_id IS NULL""",
    "orphan_stores": """
        SELECT COUNT(*) FROM sales s LEFT JOIN stores st USING (store_id)
        WHERE st.store_id IS NULL""",
    "strata_without_panel_stores": """
        SELECT COUNT(*) FROM universe u
        LEFT JOIN (SELECT region, store_format, COUNT(*) AS n FROM stores GROUP BY 1, 2) s
        USING (region, store_format) WHERE s.n IS NULL""",
}

PREPARE_SQL = """
CREATE TEMP TABLE clean_sales AS
    SELECT DISTINCT store_id, product_id, week, units, value
    FROM sales WHERE units > 0 AND value > 0;

CREATE TEMP TABLE store_factors AS
    SELECT s.store_id, s.region, s.store_format, s.turnover,
           u.universe_stores * 1.0 / n.sample_stores AS expansion_factor
    FROM stores s
    JOIN (SELECT region, store_format, COUNT(*) AS sample_stores
          FROM stores GROUP BY region, store_format) n USING (region, store_format)
    JOIN universe u USING (region, store_format);
"""

PRODUCT_METRICS_SQL = """
SELECT p.product_id, p.brand, p.segment,
       SUM(c.value * f.expansion_factor) AS projected_value,
       SUM(c.units * f.expansion_factor) AS projected_units
FROM clean_sales c
JOIN store_factors f USING (store_id)
JOIN products p USING (product_id)
GROUP BY p.product_id
"""

DISTRIBUTION_SQL = """
SELECT d.product_id,
       100.0 * SUM(f.expansion_factor)
             / (SELECT SUM(expansion_factor) FROM store_factors) AS numeric_distribution,
       100.0 * SUM(f.turnover * f.expansion_factor)
             / (SELECT SUM(turnover * expansion_factor) FROM store_factors) AS weighted_distribution
FROM (SELECT DISTINCT product_id, store_id FROM clean_sales) d
JOIN store_factors f USING (store_id)
GROUP BY d.product_id
"""


def load_db(data_dir):
    con = sqlite3.connect(":memory:")
    for table in ["universe", "stores", "products", "sales"]:
        pd.read_csv(Path(data_dir) / f"{table}.csv").to_sql(table, con, index=False)
    return con


def run_dq_checks(con):
    rows = [(name, con.execute(sql).fetchone()[0]) for name, sql in DQ_CHECKS.items()]
    return pd.DataFrame(rows, columns=["check", "issues_found"])


def detect_anomalies(weekly, window=9, z_threshold=4.0, min_rel_change=0.3):
    """Flag product-weeks that deviate strongly from a rolling median.

    `weekly` has columns product_id, week, units (panel total per week).
    Missing weeks count as zero, so a file that never arrived shows up as a drop.
    The noise scale is the larger of the robust spread (1.4826 * MAD) and
    sqrt(baseline), the Poisson floor, so low-volume products are not over-flagged.
    """
    grid = (
        weekly.pivot(index="week", columns="product_id", values="units")
        .reindex(range(int(weekly["week"].min()), int(weekly["week"].max()) + 1))
        .fillna(0)
    )
    flagged = []
    for product_id, series in grid.items():
        baseline = series.rolling(window, center=True, min_periods=window // 2 + 1).median()
        resid = series - baseline
        scale = np.maximum(1.4826 * resid.abs().median(), np.sqrt(baseline.clip(lower=1)))
        z = resid / scale
        rel = resid / baseline.replace(0, np.nan)
        hits = (z.abs() > z_threshold) & (rel.abs() > min_rel_change)
        for week in series.index[hits]:
            flagged.append(
                {
                    "product_id": product_id,
                    "week": week,
                    "units": series[week],
                    "expected_units": round(baseline[week], 1),
                    "robust_z": round(z[week], 1),
                    "type": "spike" if resid[week] > 0 else "drop",
                }
            )
    cols = ["product_id", "week", "units", "expected_units", "robust_z", "type"]
    return pd.DataFrame(flagged, columns=cols)


def build_assortment_table(con):
    """Projected sales, distribution and ABC class per product."""
    metrics = pd.read_sql(PRODUCT_METRICS_SQL, con).merge(
        pd.read_sql(DISTRIBUTION_SQL, con), on="product_id"
    )
    metrics = metrics.sort_values("projected_value", ascending=False).reset_index(drop=True)
    metrics["value_share_pct"] = 100 * metrics["projected_value"] / metrics["projected_value"].sum()
    metrics["cum_share_pct"] = metrics["value_share_pct"].cumsum()
    metrics["abc_class"] = np.select(
        [metrics["cum_share_pct"] <= 80, metrics["cum_share_pct"] <= 95], ["A", "B"], default="C"
    )
    # Value sold per point of weighted distribution: how well a product sells
    # where it is actually listed.
    metrics["velocity"] = metrics["projected_value"] / metrics["weighted_distribution"]
    high_velocity = metrics["velocity"] >= metrics["velocity"].quantile(0.75)
    low_distribution = metrics["weighted_distribution"] < metrics["weighted_distribution"].median()
    metrics["distribution_gap"] = high_velocity & low_distribution
    return metrics.round(2)


def plot_pareto(assortment, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = assortment["abc_class"].map({"A": "#2a6f97", "B": "#89c2d9", "C": "#cfd8dc"})
    ax.bar(range(len(assortment)), assortment["value_share_pct"], color=colors)
    ax.set_ylabel("Share of projected value (%)")
    ax.set_xlabel("Products, ranked by projected value")
    ax2 = ax.twinx()
    ax2.plot(range(len(assortment)), assortment["cum_share_pct"], color="#c1121f", lw=1.8)
    ax2.axhline(80, color="grey", ls="--", lw=0.8)
    ax2.set_ylabel("Cumulative share (%)")
    ax2.set_ylim(0, 105)
    ax.set_title("Assortment Pareto (synthetic data): blue = A, light blue = B, grey = C")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(data_dir="data", out_dir="output"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    con = load_db(data_dir)

    dq = run_dq_checks(con)
    print("\nData-quality checks (SQL)\n", dq.to_string(index=False))

    con.executescript(PREPARE_SQL)
    weekly = pd.read_sql(
        "SELECT product_id, week, SUM(units) AS units FROM clean_sales GROUP BY 1, 2", con
    )
    anomalies = detect_anomalies(weekly)
    print(f"\nFlagged product-weeks: {len(anomalies)}\n", anomalies.to_string(index=False))

    assortment = build_assortment_table(con)
    class_summary = assortment.groupby("abc_class").agg(
        products=("product_id", "count"), value_share_pct=("value_share_pct", "sum")
    )
    print("\nABC classes\n", class_summary.round(1).to_string())
    gaps = assortment[assortment["distribution_gap"]]
    print(f"\nDistribution gaps (high velocity, low weighted distribution): {len(gaps)} products")
    print(gaps[["product_id", "brand", "weighted_distribution", "velocity"]].to_string(index=False))

    dq.to_csv(out / "dq_report.csv", index=False)
    anomalies.to_csv(out / "anomalies.csv", index=False)
    assortment.to_csv(out / "assortment_summary.csv", index=False)
    plot_pareto(assortment, out / "pareto.png")
    print(f"\nResults written to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="output")
    args = parser.parse_args()
    main(args.data_dir, args.out_dir)
