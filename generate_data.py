"""Generate a small SYNTHETIC retail-panel dataset.

Everything here is made up. The data mimics the shape of a store panel
(a sample of stores drawn from a known universe, weekly sales per product)
and has a few data-quality problems injected on purpose, so that
panel_analysis.py has something to find. The injected problems are saved
to data/injected_issues.csv so detection can be checked against the truth.

Usage:
    python generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_PRODUCTS = 60
N_WEEKS = 52
SAMPLING_RATE = 0.2
BASE_UNITS = 30  # weekly units of a product with popularity 1.0 in a store of size 1.0

REGIONS = ["North", "South", "East", "West", "Central"]
FORMATS = ["Hyper", "Super", "Discount", "Convenience"]
UNIVERSE_BASE = {"Hyper": 10, "Super": 40, "Discount": 35, "Convenience": 60}
STORE_SIZE = {"Hyper": 3.0, "Super": 1.5, "Discount": 1.0, "Convenience": 0.4}
RANGE_BREADTH = {"Hyper": 0.9, "Super": 0.7, "Discount": 0.5, "Convenience": 0.3}


def build_universe(rng):
    """Number of stores in the whole market, per region x format stratum."""
    rows = [
        {
            "region": region,
            "store_format": fmt,
            "universe_stores": int(UNIVERSE_BASE[fmt] * rng.uniform(0.7, 1.4)),
        }
        for region in REGIONS
        for fmt in FORMATS
    ]
    return pd.DataFrame(rows)


def build_stores(universe, rng):
    """The panel: a ~20% sample of stores in every stratum."""
    rows, store_id = [], 1
    for u in universe.itertuples():
        n_sample = max(2, round(u.universe_stores * SAMPLING_RATE))
        for _ in range(n_sample):
            rows.append(
                {
                    "store_id": store_id,
                    "region": u.region,
                    "store_format": u.store_format,
                    "turnover": round(STORE_SIZE[u.store_format] * rng.lognormal(0, 0.25), 3),
                }
            )
            store_id += 1
    return pd.DataFrame(rows)


def build_products(rng):
    popularity = rng.lognormal(0, 1, N_PRODUCTS)
    popularity = popularity / popularity.max()
    # A few strong sellers are listed in far fewer stores than their sales justify
    # (the "distribution gaps" an assortment study should surface).
    listing_factor = np.ones(N_PRODUCTS)
    top = np.argsort(-popularity)[3:15]
    listing_factor[rng.choice(top, size=4, replace=False)] = 0.25
    return pd.DataFrame(
        {
            "product_id": np.arange(1, N_PRODUCTS + 1),
            "brand": rng.choice(list("ABCDEF"), N_PRODUCTS),
            "segment": rng.choice(["Basic", "Premium", "Value"], N_PRODUCTS),
            "price": np.round(rng.uniform(0.8, 6.0, N_PRODUCTS), 2),
            "popularity": np.round(popularity, 4),
            "listing_factor": listing_factor,
        }
    )


def build_sales(stores, products, rng):
    """Weekly units and value per store x product (only weeks with sales)."""
    weeks = np.arange(1, N_WEEKS + 1)
    season = 1 + 0.15 * np.sin(2 * np.pi * weeks / N_WEEKS)
    pop = products["popularity"].to_numpy()
    listing_factor = products["listing_factor"].to_numpy()
    frames = []
    for s in stores.itertuples():
        p_list = np.clip(
            RANGE_BREADTH[s.store_format] * np.sqrt(pop) * 1.3 * listing_factor, 0.03, 1.0
        )
        listed = rng.random(len(products)) < p_list
        for p in products[listed].itertuples():
            units = rng.poisson(BASE_UNITS * s.turnover * p.popularity * season)
            keep = units > 0
            frames.append(
                pd.DataFrame(
                    {
                        "store_id": s.store_id,
                        "product_id": p.product_id,
                        "week": weeks[keep],
                        "units": units[keep],
                        "value": np.round(units[keep] * p.price * rng.normal(1, 0.03, keep.sum()), 2),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def inject_issues(sales, products, rng):
    """Break the data in a few known ways and record what was broken."""
    issues = []
    mid = products.sort_values("popularity", ascending=False).iloc[5:30]["product_id"].to_numpy()
    chosen = rng.choice(mid, size=5, replace=False)
    spikes = [(chosen[0], 14), (chosen[1], 27), (chosen[2], 40)]
    drops = [(chosen[3], 20), (chosen[4], 33)]

    for pid, week in spikes:  # e.g. a unit-conversion error in a supplier file
        m = (sales["product_id"] == pid) & (sales["week"] == week)
        sales.loc[m, ["units", "value"]] *= 8
        issues.append({"issue": "spike", "product_id": int(pid), "week": week})

    for pid, week in drops:  # e.g. a file that never arrived
        m = (sales["product_id"] == pid) & (sales["week"] == week)
        sales = sales[~m]
        issues.append({"issue": "drop", "product_id": int(pid), "week": week})

    dupes = sales.sample(12, random_state=1)
    sales = pd.concat([sales, dupes], ignore_index=True)
    issues.append({"issue": "duplicate_rows", "product_id": None, "week": None})

    zero_value = rng.choice(sales.index, size=6, replace=False)
    sales.loc[zero_value, "value"] = 0
    issues.append({"issue": "units_without_value", "product_id": None, "week": None})

    for pid in products.loc[products["listing_factor"] < 1, "product_id"]:
        issues.append({"issue": "under_distributed", "product_id": int(pid), "week": None})

    return sales, pd.DataFrame(issues)


def main(out_dir="data"):
    rng = np.random.default_rng(SEED)
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    universe = build_universe(rng)
    stores = build_stores(universe, rng)
    products = build_products(rng)
    sales = build_sales(stores, products, rng)
    sales, issues = inject_issues(sales, products, rng)

    universe.to_csv(out / "universe.csv", index=False)
    stores.to_csv(out / "stores.csv", index=False)
    products.drop(columns=["popularity", "listing_factor"]).to_csv(out / "products.csv", index=False)
    sales.to_csv(out / "sales.csv", index=False)
    issues.to_csv(out / "injected_issues.csv", index=False)
    print(f"{len(stores)} panel stores, {len(products)} products, {len(sales):,} sales rows -> {out}/")


if __name__ == "__main__":
    main()
