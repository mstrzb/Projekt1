import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panel_analysis import PREPARE_SQL, detect_anomalies  # noqa: E402


def test_detect_anomalies_finds_spike_and_drop():
    rng = np.random.default_rng(0)
    units = rng.poisson(200, 52).astype(float)
    units[10] *= 8  # spike
    units[30] = 0  # missing file
    weekly = pd.DataFrame({"product_id": 1, "week": np.arange(1, 53), "units": units})
    flagged = detect_anomalies(weekly)
    assert set(flagged["week"]) == {11, 31}
    assert dict(zip(flagged["week"], flagged["type"])) == {11: "spike", 31: "drop"}


def test_detect_anomalies_ignores_normal_noise():
    rng = np.random.default_rng(1)
    weekly = pd.DataFrame({"product_id": 1, "week": np.arange(1, 53), "units": rng.poisson(200, 52)})
    assert detect_anomalies(weekly).empty


def test_expansion_factor_is_universe_over_sample():
    con = sqlite3.connect(":memory:")
    pd.DataFrame({"region": ["N"], "store_format": ["Super"], "universe_stores": [50]}).to_sql(
        "universe", con, index=False
    )
    pd.DataFrame(
        {"store_id": [1, 2], "region": ["N", "N"], "store_format": ["Super", "Super"], "turnover": [1.0, 2.0]}
    ).to_sql("stores", con, index=False)
    pd.DataFrame({"store_id": [1], "product_id": [1], "week": [1], "units": [1], "value": [1.0]}).to_sql(
        "sales", con, index=False
    )
    con.executescript(PREPARE_SQL)
    factors = pd.read_sql("SELECT expansion_factor FROM store_factors", con)
    assert (factors["expansion_factor"] == 25.0).all()  # 50 universe stores / 2 panel stores
