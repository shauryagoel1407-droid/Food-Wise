"""Pre-meal history lookup for FoodWise.

Mirrors the definitions audited in FoodWise_Model_Training_Updated.ipynb:
- Previous Attendance (Same Menu) / Previous Waste (Same Menu): the Actual and Waste kg of the most
  recent STRICTLY EARLIER record with the same Meal and the same Menu.
- Previous Waste kg: Waste kg of the most recent strictly earlier record of the same Meal.
Only records dated before the intended meal date are used. Missing history is never filled with zero.
"""
import numbers

import pandas as pd

FEATURES = ["Meal", "Menu", "Expected", "Previous Attendance (Same Menu)", "Prepared kg",
            "Previous Waste kg", "Previous Waste (Same Menu)", "Remark", "Day"]
REQUIRED_COLS = ["Date", "Day", "Meal", "Menu", "Expected", "Actual", "Prepared kg", "Waste kg", "Previous Waste kg",
                 "Remark", "Previous Attendance (Same Menu)", "Previous Waste (Same Menu)"]


def norm(text):
    """Collapse harmless whitespace differences only (case and wording are left untouched)."""
    return " ".join(str(text).split())


def prepare(df):
    """Return a sorted working copy; the original dataframe is not modified."""
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["Meal"] = d["Meal"].map(norm)
    d["_menu"] = d["Menu"].map(norm)
    d["_ord"] = range(len(d))
    return d.sort_values(["Date", "_ord"], kind="stable").reset_index(drop=True)


def _before(d, meal, date):
    return d[(d["Date"] < pd.Timestamp(date).normalize()) & (d["Meal"] == norm(meal))]


def menus_before(d, meal, date):
    return sorted(_before(d, meal, date)["_menu"].unique())


def _src(r):
    """The exact dataset record a historical value came from ('Data row' = 1-based position among the data rows)."""
    return {"Data row": int(r["_ord"]) + 1, "Date": r["Date"], "Meal": r["Meal"], "Menu": r["Menu"],
            "Actual": r["Actual"], "Waste kg": r["Waste kg"]}


def lookup(d, meal, menu, date):
    """Return (history dict, None) or (None, explanation) when history is insufficient."""
    when = f"{pd.Timestamp(date):%d %b %Y}"
    h = _before(d, meal, date)
    if h.empty:
        return None, f"No {meal} records exist before {when}, so there is not enough history to predict."
    same = h[h["_menu"] == norm(menu)]
    if same.empty:
        return None, f"'{norm(menu)}' has no earlier {meal} record before {when}, so there is not enough history to predict."
    sm, cat = same.iloc[-1], h.iloc[-1]
    vals = {"Previous Attendance (Same Menu)": sm["Actual"], "Previous Waste kg": cat["Waste kg"],
            "Previous Waste (Same Menu)": sm["Waste kg"]}
    missing = [k for k, v in vals.items() if pd.isna(v)]
    if missing:
        return None, f"The dataset has missing values in the history records needed for: {', '.join(missing)}."
    out = {k: float(v) for k, v in vals.items()}
    out.update(menu_value=sm["Menu"], same_date=sm["Date"], cat_date=cat["Date"], same_src=_src(sm), cat_src=_src(cat))
    return out, None


def build_row(meal, hist, expected, prepared, remark, day):
    """One-row DataFrame with the exact feature names and order the saved model expects (no Actual)."""
    row = {"Meal": meal, "Menu": hist["menu_value"], "Expected": expected,
           "Previous Attendance (Same Menu)": hist["Previous Attendance (Same Menu)"], "Prepared kg": prepared,
           "Previous Waste kg": hist["Previous Waste kg"], "Previous Waste (Same Menu)": hist["Previous Waste (Same Menu)"],
           "Remark": remark, "Day": day}
    return pd.DataFrame([row], columns=FEATURES)


# ---- Transparency tables. They only format the lookup result and the model-input row; nothing is recomputed. ----
def _d(ts):
    return f"{pd.Timestamp(ts):%d %b %Y}"


def _v(v):
    return f"{v:.10g}" if isinstance(v, numbers.Number) else str(v)


def source_frames(hist):
    """(same-menu source table, same-meal-category source table), one row each, from the lookup's own source rows."""
    def one(src, cols):
        return pd.DataFrame([{c: (_d(src[c]) if c == "Date" else _v(src[c])) for c in cols}])
    return (one(hist["same_src"], ["Data row", "Date", "Meal", "Menu", "Actual", "Waste kg"]),
            one(hist["cat_src"], ["Data row", "Date", "Meal", "Menu", "Waste kg"]))


def historical_features_table(row, hist):
    r, s, c = row.iloc[0], hist["same_src"], hist["cat_src"]
    items = [("Previous Attendance (Same Menu)", s, "Actual attendance from the latest earlier occurrence of the selected menu in this meal category."),
             ("Previous Waste kg", c, "Waste kg from the latest earlier record of this meal category."),
             ("Previous Waste (Same Menu)", s, "Waste kg from the latest earlier occurrence of the selected menu in this meal category.")]
    return pd.DataFrame([{"Feature": f, "Value": _v(r[f]), "Source date": _d(src["Date"]), "How it was obtained": how} for f, src, how in items])


def input_summary(row, hist):
    """Feature / Value / Source for every column of the exact DataFrame passed to model.predict()."""
    s, c = hist["same_src"], hist["cat_src"]
    src = {"Meal": "User input", "Menu": "User input (the dataset's spelling of the selected menu)", "Expected": "User input",
           "Prepared kg": "User input (planned food)", "Remark": "User input", "Day": "Derived from the selected meal date",
           "Previous Attendance (Same Menu)": f"Actual attendance from the same-menu source record (data row {s['Data row']}, {_d(s['Date'])})",
           "Previous Waste (Same Menu)": f"Waste kg from the same-menu source record (data row {s['Data row']}, {_d(s['Date'])})",
           "Previous Waste kg": f"Waste kg from the previous meal-category source record (data row {c['Data row']}, {_d(c['Date'])})"}
    return pd.DataFrame([{"Feature": f, "Value": _v(row.iloc[0][f]), "Source": src[f]} for f in row.columns])