"""Checks for the updated FoodWise workflow. Run from the Smart_Mess folder: python test_updated_model.py"""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from history import FEATURES, REQUIRED_COLS, build_row, lookup, menus_before, prepare

BASE = Path(__file__).resolve().parent
warnings.filterwarnings("ignore")

# 1. Model loads and its schema matches the app (Actual excluded).
model = joblib.load(BASE / "food_waste_model_updated.pkl")
assert list(model.feature_names_in_) == FEATURES and "Actual" not in FEATURES
print("PASS model schema:", FEATURES)

# 2. Smoke prediction with the notebook's example input.
ex = pd.DataFrame([["Lunch", "Rice + Rajma + Roti", 5000, 4500, 2800.0, 180.0, 210.0, "Normal day", "Monday"]], columns=FEATURES)
print(f"PASS smoke prediction: {float(model.predict(ex)[0]):.2f} kg (raw)")

# 3. History lookup against the notebook's audit algorithm, on a SYNTHETIC TEST FIXTURE (not your data).
rng = np.random.default_rng(0)
rows = [dict(Date=pd.Timestamp("2025-01-01") + pd.Timedelta(days=i), Meal=m, Menu=f"{m} menu {rng.integers(0, 4)}",
             Actual=int(rng.integers(3000, 5000)), **{"Waste kg": float(rng.uniform(50, 300))})
        for i in range(60) for m in ["Breakfast", "Lunch"]]
fx = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)  # shuffled input order
d = prepare(fx)
checked = 0
for _, r in d.iterrows():
    prev_same = d[(d.Meal == r.Meal) & (d.Menu == r.Menu) & (d.Date < r.Date)]
    prev_cat = d[(d.Meal == r.Meal) & (d.Date < r.Date)]
    h, why = lookup(d, r.Meal, r.Menu, r.Date)
    if prev_same.empty:
        assert h is None and "not enough history" in why  # missing history is reported, never zero-filled
        continue
    s, c = prev_same.iloc[-1], prev_cat.iloc[-1]
    assert h["Previous Attendance (Same Menu)"] == s.Actual and h["Previous Waste (Same Menu)"] == s["Waste kg"]
    assert h["Previous Waste kg"] == c["Waste kg"] and h["same_date"] < r.Date and h["cat_date"] < r.Date
    future = d[d.Date >= r.Date].copy()  # corrupt every record on/after the date: result must not change
    future["Actual"], future["Waste kg"] = -999, -999
    h2, _ = lookup(pd.concat([d[d.Date < r.Date], future]), r.Meal, r.Menu, r.Date)
    assert h2 == h
    checked += 1
print(f"PASS history lookup: {checked} rows match the notebook definition, use only earlier records, ignore future data")
row = build_row("Lunch", h, 5000, 2800.0, "Normal day", "Monday")
assert list(row.columns) == FEATURES and "Actual" not in row.columns and not row.isna().any().any()
nan_fx = d.copy(); nan_fx.loc[nan_fx.Meal == "Lunch", "Waste kg"] = np.nan
assert lookup(nan_fx, "Lunch", nan_fx[nan_fx.Meal == "Lunch"].Menu.iloc[-1], "2026-01-01")[0] is None
print("PASS model input built with exact feature order; NaN history rejected")


# 3b. Transparency: source rows, feature table and input summary all come from the same lookup / model-input row.
from history import historical_features_table, input_summary, source_frames


class Spy:  # stands in for the model to capture exactly what predict() receives
    def predict(self, X):
        self.X = X.copy()
        return [1.0]


diff_rows = 0
for _, r in d.iloc[40:].iterrows():
    h, _why = lookup(d, r.Meal, r.Menu, r.Date)
    if h is None:
        continue
    row = build_row(r.Meal, h, 5000, 2800.0, "Normal day", "Monday")
    spy = Spy(); spy.predict(row)
    s_, c_ = h["same_src"], h["cat_src"]
    raw_same = fx.iloc[s_["Data row"] - 1]  # independent check against the untouched input table
    assert (raw_same.Date, raw_same.Menu, raw_same.Meal) == (s_["Date"], r.Menu, r.Meal) and raw_same.Actual == s_["Actual"] and raw_same["Waste kg"] == s_["Waste kg"]
    latest = fx[(fx.Meal == r.Meal) & (fx.Menu == r.Menu) & (fx.Date < r.Date)].Date.max()
    latest_cat = fx[(fx.Meal == r.Meal) & (fx.Date < r.Date)].Date.max()
    assert s_["Date"] == latest and c_["Date"] == latest_cat and s_["Date"] < r.Date and c_["Date"] < r.Date
    assert fx.iloc[c_["Data row"] - 1]["Waste kg"] == c_["Waste kg"]
    feat = historical_features_table(row, h).set_index("Feature")["Value"]
    for f in ["Previous Attendance (Same Menu)", "Previous Waste kg", "Previous Waste (Same Menu)"]:
        assert feat[f] == f"{spy.X.iloc[0][f]:.10g}"
    summ = input_summary(spy.X, h)
    assert list(summ.Feature) == FEATURES and "Actual" not in set(summ.Feature)
    assert list(summ.Value) == [f"{v:.10g}" if isinstance(v, (int, float, np.number)) else str(v) for v in spy.X.iloc[0]]
    same_df, cat_df = source_frames(h)
    assert "Actual" in same_df.columns and "Actual" not in cat_df.columns
    diff_rows += int(s_["Data row"] != c_["Data row"])
assert diff_rows > 0
print(f"PASS transparency: source rows verified against raw data; tables match the predict() input; {diff_rows} cases with different same-menu and category rows")

# 4. Real dataset checks (skipped if the file is not present next to this script).
path = BASE / "data" / "FoodWise_Updated_Dataset.xlsx"
if path.exists():
    real = pd.read_excel(path, sheet_name="Mess Data")
    assert not [c for c in REQUIRED_COLS if c not in real.columns], "missing columns"
    print(f"PASS dataset: {len(real):,} rows, {real['Date'].min():%Y-%m-%d} to {real['Date'].max():%Y-%m-%d}")
else:
    print("SKIPPED dataset checks: data/FoodWise_Updated_Dataset.xlsx not found")