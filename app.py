"""FoodWise - Food Waste Prediction System (Streamlit implementation of the approved design)."""
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from history import (FEATURES, REQUIRED_COLS, build_row, historical_features_table, input_summary, lookup, menus_before,
                     prepare, source_frames)

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "food_waste_model_updated.pkl"
DATASET_NAME = "FoodWise_Updated_Dataset.xlsx"
DATA_DIR = BASE / "data"
MEALS = ["Breakfast", "Lunch", "Snacks", "Dinner"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
REMARKS = ["Normal day", "Weekend", "Holiday", "Special event"]
PAL = ["#2fae80", "#bfeed6", "#c9b27a", "#5f9f86"]
MEAL_COLOR = dict(zip(MEALS, PAL))

st.set_page_config(page_title="FoodWise", page_icon="🍃", layout="wide", initial_sidebar_state="collapsed")

CSS = """
:root{--ink:#f5f5f7;--mute:#9db5a9;--line:rgba(143,209,179,.16);--em:#2fae80;--mint:#8fd1b3;--bg:#080d0b;--sur:#0d1511}
header[data-testid=stHeader],#MainMenu,[data-testid=stToolbar],[data-testid=stDecoration],[data-testid=stSidebar],[data-testid=stSidebarCollapsedControl]{display:none!important}
html,[data-testid=stMain]{scroll-behavior:smooth;scroll-padding-top:64px}
.anchor{scroll-margin-top:64px}
.stApp{background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display','Inter','Helvetica Neue',Arial,sans-serif;-webkit-font-smoothing:antialiased}
.block-container{padding:0!important;max-width:100%!important}
.block-container>[data-testid=stVerticalBlock]{gap:0}
[data-testid=stMarkdownContainer] p{margin:0}
.nav{position:fixed;top:0;left:0;right:0;z-index:999;height:48px;display:flex;align-items:center;justify-content:center;gap:clamp(18px,4vw,44px);background:rgba(8,13,11,.72);backdrop-filter:saturate(180%) blur(20px);-webkit-backdrop-filter:saturate(180%) blur(20px);border-bottom:1px solid var(--line)}
.nav a{font-size:12px;color:#b7cbc0!important;text-decoration:none!important;transition:color .2s}.nav a:hover{color:#fff!important}
.nav .lg{display:flex;gap:7px;align-items:center;font-weight:600;color:#fff;font-size:14px;letter-spacing:-.2px}
.nav svg{width:16px;height:16px;fill:none;stroke:var(--em);stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
.hero{background:radial-gradient(120% 90% at 50% 100%,#1c5a42 0%,#0f3a2b 38%,#080d0b 80%);color:var(--ink);text-align:center;padding:140px 22px 0;overflow:hidden}
.eyebrow{font-size:12px;font-weight:600;letter-spacing:.14em;color:var(--mint)}
.h1{font-size:clamp(48px,9vw,104px);line-height:1.02;letter-spacing:-.045em;font-weight:700;margin:18px 0 22px}
.h1 span{background:linear-gradient(180deg,#bfeed6,#5fc59a);-webkit-background-clip:text;background-clip:text;color:transparent}
.hero .lead {
    display: block;
    width: 100%;
    max-width: 680px;
    margin: 0 auto;
    padding: 0;
    text-align: center !important;
    font-size: clamp(17px, 2vw, 21px);
    line-height: 1.5;
    color: #b9cbc2;
    text-wrap: balance;
    position: relative;
    left: 0;
}
.btns{display:flex;gap:14px;justify-content:center;margin-top:32px;flex-wrap:wrap}
.b{display:inline-block;padding:12px 24px;border-radius:980px;font-size:16px;text-decoration:none!important;transition:.25s;font-weight:500}
.b.p{background:var(--em);color:#04271a!important}.b.p:hover{background:#4cc99a;transform:scale(1.03)}.b.s{color:var(--mint)!important}.b.s:hover{color:#fff!important}.b.s:after{content:' \\203A'}
.vis{width:min(900px,100%);margin:58px auto 0;display:block}
.ring{transform-origin:450px 330px;animation:spin 90s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}
.fl{animation:fl 7s ease-in-out infinite}@keyframes fl{50%{transform:translateY(-10px)}}
.rv{animation:fu 1.1s cubic-bezier(.2,.7,.2,1) both}.d1{animation-delay:.12s}.d2{animation-delay:.24s}.d3{animation-delay:.36s}
@keyframes fu{from{opacity:0;transform:translateY(28px)}to{opacity:1;transform:none}}
.banner{background:var(--sur);color:var(--mute);text-align:center;font-size:12px;padding:10px 16px;border-bottom:1px solid var(--line)}
.sec{padding:clamp(80px,12vw,150px) 22px}.in{max-width:1000px;margin:0 auto}.c{text-align:center}
.prob{background:#0a1210}.sus{background:radial-gradient(90% 70% at 50% 0%,#12382a,#080d0b 72%)}
.tag{font-size:12px;font-weight:600;letter-spacing:.12em;color:var(--mint);text-transform:uppercase}
.h2{font-size:clamp(36px,6vw,72px);line-height:1.05;letter-spacing:-.04em;font-weight:700;margin:14px 0 20px;color:var(--ink)}
.sub{font-size:clamp(18px,2.2vw,24px);line-height:1.35;color:var(--mute);max-width:640px}.c .sub{margin:0 auto}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:40px;margin-top:80px;text-align:left}
.stats b{display:block;font-size:clamp(40px,6vw,72px);letter-spacing:-.04em;font-weight:700;color:var(--mint);line-height:1}.stats p{margin-top:12px;color:var(--mute);font-size:16px;max-width:260px}
.plate{width:min(420px,80%);margin:60px auto 0;display:block}
.sg{display:grid;grid-template-columns:repeat(3,1fr);gap:48px;margin-top:70px;text-align:left}
.sg div{border-top:1px solid rgba(143,209,179,.3);padding-top:22px}.sg h4{font-size:21px;font-weight:600;letter-spacing:-.02em;margin-bottom:8px;color:var(--ink)}.sg p{color:var(--mute);font-size:16px}
.disc{max-width:760px;margin:64px auto 0;font-size:13px;line-height:1.6;color:var(--mute);text-align:center}
.st-key-predict{padding:clamp(80px,12vw,140px) max(22px,calc((100vw - 1000px)/2));background:var(--bg)}
.st-key-analytics{background:var(--sur);border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:clamp(80px,12vw,140px) max(22px,calc((100vw - 1000px)/2))}
.stApp label p{font-size:13px!important;color:#c9ddd3!important}
[data-baseweb=input],[data-baseweb=select]>div{background:rgba(255,255,255,.04)!important;border:1px solid var(--line)!important;border-radius:12px!important;transition:border-color .2s}
[data-baseweb=base-input]{background:transparent!important}
[data-baseweb=input]:focus-within,[data-baseweb=select]>div:hover,[data-baseweb=select]>div:focus-within{border-color:var(--em)!important}
[data-testid=stCaptionContainer]{color:var(--mute)}
.st-key-predict [data-testid=stButton] button{width:100%;margin-top:14px;border:0;border-radius:980px;padding:.8rem;background:var(--em);color:#04271a;font-size:17px;font-weight:600;transition:.25s}
.st-key-predict [data-testid=stButton] button:hover{background:#4cc99a;color:#04271a;transform:scale(1.015)}
.st-key-predict [data-testid=stExpander]{border:1px solid var(--line);border-radius:14px;background:rgba(255,255,255,.02);margin-top:14px}
.res{display:flex;flex-direction:column;justify-content:center;min-height:560px;text-align:center;padding:48px 28px;border-radius:28px;border:1px solid var(--line);background:radial-gradient(closest-side at 50% 115%,rgba(47,174,128,.38),transparent),#0b2e22;box-shadow:0 0 80px rgba(47,174,128,.12);animation:pop .8s cubic-bezier(.2,.8,.2,1) both}
@keyframes pop{from{opacity:0;transform:scale(.97)}to{opacity:1;transform:none}}
.res small{font-size:12px;letter-spacing:.14em;color:var(--mint);font-weight:600}
.num{font-size:clamp(56px,7vw,92px);font-weight:700;letter-spacing:-.05em;line-height:1;margin:16px 0 4px;font-variant-numeric:tabular-nums;color:#fff}.num i{font-style:normal;font-size:.32em;color:var(--mint);letter-spacing:0;margin-left:8px;font-weight:500}
.pct{display:inline-block;align-self:center;margin:18px 0 12px;padding:6px 16px;border-radius:980px;border:1px solid rgba(143,209,179,.4);color:#bfeed6;font-size:14px}
.res p{color:#b9cbc2;font-size:14px;max-width:320px;margin:0 auto}
.flag{margin:18px auto 0;max-width:340px;padding:10px 14px;border-radius:14px;background:rgba(201,178,122,.16);border:1px solid rgba(201,178,122,.5);color:#ecd9a6;font-size:13px}
.chip{display:block;text-align:center;margin-top:14px;font-size:12px;color:var(--mute)}
.ct{font-size:21px;font-weight:600;letter-spacing:-.02em;margin-top:2.2rem;color:var(--ink)}.cs{font-size:14px;color:var(--mute);margin-bottom:.4rem}
.foot{padding:36px 22px;text-align:center;font-size:12px;color:var(--mute);border-top:1px solid var(--line)}
@media (max-width:820px){.stats,.sg{grid-template-columns:1fr;gap:32px}.res{min-height:0}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

HERO = """<div class="nav"><span class="lg"><svg viewBox="0 0 24 24"><path d="M5 19c0-9 5-14 15-14 0 10-5 15-14 15"/><path d="M5 19c2-5 5-8 9-10"/></svg>FoodWise</span><a target="_self" href="#top">Overview</a><a target="_self" href="#predict">Predict</a><a target="_self" href="#analytics">Analytics</a><a target="_self" href="#about">About</a></div>
<div id="top" class="hero">
<div class="eyebrow rv">FOODWISE · INTELLIGENT FOOD MANAGEMENT</div>
<div class="h1 rv d1">Less Waste.<br><span>More Tomorrow.</span></div>
<p class="lead rv d2">Meet a smarter way to understand food waste, explore meal patterns, and move towards a more sustainable campus.</p>
<div class="btns rv d3"><a class="b p" target="_self" href="#predict">Explore FoodWise</a><a class="b s" target="_self" href="#analytics">Discover the Technology</a></div>
<svg class="vis rv d3" viewBox="0 0 900 380" fill="none" aria-hidden="true"><defs><radialGradient id="pg" cx="50%" cy="50%" r="50%"><stop offset="0" stop-color="#2a7a5b"/><stop offset="1" stop-color="#0f3a2b"/></radialGradient><linearGradient id="lf" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#bfeed6"/><stop offset="1" stop-color="#2fae80"/></linearGradient></defs>
<circle cx="450" cy="330" r="330" fill="url(#pg)" opacity=".55"/>
<g class="ring" stroke="#8fd1b3" stroke-opacity=".35"><circle cx="450" cy="330" r="300"/><circle cx="450" cy="330" r="240" stroke-dasharray="2 8"/><circle cx="450" cy="330" r="180" stroke-opacity=".5"/><circle cx="450" cy="30" r="5" fill="#8fd1b3"/><circle cx="210" cy="330" r="4" fill="#8fd1b3"/><circle cx="690" cy="330" r="4" fill="#8fd1b3"/></g>
<circle cx="450" cy="330" r="120" fill="#0b2e22" stroke="#8fd1b3" stroke-opacity=".5"/><circle cx="450" cy="330" r="92" stroke="#8fd1b3" stroke-opacity=".3"/>
<g class="fl"><path d="M450 300c0-52 26-86 80-92-2 54-26 88-80 92z" fill="url(#lf)"/><path d="M450 300c0-40-22-68-64-74 2 42 22 68 64 74z" fill="#7fd1ae"/><path d="M450 300v50" stroke="#e8f7ef" stroke-width="3" stroke-linecap="round"/></g></svg>
</div>
<div class="banner">Experimental estimates · The model is trained on synthetic data and its real-world accuracy has not been established.</div>
<div class="sec prob"><div class="in"><div class="tag">The Problem</div>
<div class="h2">Every Meal<br>Has an Impact.</div>
<p class="sub">Behind every serving is a decision about how much to prepare. Understanding food waste is the first step towards planning meals more thoughtfully on campus.</p>
<div class="stats"><div><b>Prepared</b><p>Quantities set before the first student arrives.</p></div><div><b>Attended</b><p>Attendance that rarely matches what was expected.</p></div><div><b>Wasted</b><p>The gap between the two is where patterns hide.</p></div></div></div></div>
<div id="predict" class="anchor"></div>"""

FOOT = """<div id="about" class="anchor"></div><div class="sec sus c"><div class="in"><div class="tag">About &amp; Sustainability</div>
<div class="h2">Small Insights.<br>A Bigger Difference.</div>
<p class="sub">FoodWise explores how attendance, preparation and previous waste relate to food waste in a campus mess, using a machine learning model and a synthetic dataset. Better visibility supports better questions. FoodWise is built to help a campus explore planning decisions and raise awareness of food waste, with no promises beyond that.</p>
<svg class="plate" viewBox="0 0 420 200" fill="none" aria-hidden="true"><g stroke="#8fd1b3" stroke-opacity=".4"><path d="M10 190c40-120 140-170 200-170s160 50 200 170"/><path d="M60 190c30-80 90-115 150-115s120 35 150 115" stroke-opacity=".6"/><path d="M115 190c20-45 55-65 95-65s75 20 95 65"/></g><circle cx="210" cy="190" r="6" fill="#2fae80"/><circle cx="210" cy="20" r="4" fill="#8fd1b3"/></svg>
<div class="sg"><div><h4>Attendance</h4><p>Compare what was expected with who actually came.</p></div><div><h4>Preparation</h4><p>See how quantities relate to attendance, meal by meal.</p></div><div><h4>Patterns</h4><p>Notice which days and meals deserve a closer look.</p></div></div>
<p class="disc">Disclaimer: FoodWise produces experimental estimates from a model trained on synthetic data because real mess records are unavailable. Real-world prediction accuracy has not been established, and the tool is not validated for operational use. Charts reflect the loaded dataset, which is synthetic if file in the data folder is the synthetic one.</p></div></div>
<div class="foot">FoodWise · Food Waste Prediction System · Bennett University</div>"""


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None, f"Model file not found: {MODEL_PATH.name} should sit next to app.py."
    try:
        return joblib.load(MODEL_PATH), None
    except Exception as e:  # version mismatch, corrupt file, etc.
        return None, f"The model could not be loaded: {e}"


FONT = "-apple-system, Inter, Helvetica Neue, Arial"
TXT = "#c9ddd3"


def pick_dataset():
    """Return (filename, error). Only the updated dataset is accepted; there is no fallback to older files."""
    if not DATA_DIR.is_dir():
        return None, f"The data folder was not found. Checked: {DATA_DIR}"
    if not (DATA_DIR / DATASET_NAME).is_file():
        found = sorted(p.name for p in DATA_DIR.glob("*.xlsx") if not p.name.startswith("~$"))
        return None, f"{DATASET_NAME} was not found in {DATA_DIR}. Files present: {found or 'no .xlsx files'}."
    return DATASET_NAME, None


@st.cache_data(show_spinner=False)
def read_dataset(name):
    df = pd.read_excel(DATA_DIR / name, sheet_name="Mess Data")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns {missing}")
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    return df


def col(df, *keys):
    norm = {c: c.lower().replace(" ", "").replace("_", "") for c in df.columns}
    for k in keys:
        for c, n in norm.items():
            if n == k:
                return c
    for k in keys:
        for c, n in norm.items():
            if k in n:
                return c
    return None


def waste_col(df):
    c = [c for c in df.columns if "waste" in c.lower() and "previous" not in c.lower() and "%" not in c]
    c.sort(key=lambda x: "kg" not in x.lower())
    return c[0] if c else None


def style(fig, h=340):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT, size=13, color=TXT), legend=dict(orientation="h", y=-0.15, font=dict(color=TXT)),
                      hoverlabel=dict(bgcolor="#0b2e22", bordercolor="#2fae80", font_color="#ffffff"))
    fig.update_xaxes(showgrid=False, linecolor="rgba(143,209,179,.3)", tickfont=dict(color=TXT), title_font=dict(color=TXT), zeroline=False)
    fig.update_yaxes(gridcolor="rgba(143,209,179,.12)", tickfont=dict(color=TXT), title_font=dict(color=TXT), zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def head(title, sub):
    st.markdown(f'<div class="ct">{title}</div><div class="cs">{sub}</div>', unsafe_allow_html=True)


def result_html(r):
    if r is None:
        num, pct, note, flag = "&mdash;", "Awaiting input", "Enter meal details, then choose Estimate Food Waste.", ""
    else:
        num, pct = f"{r['pred']:,.2f}<i>kg</i>", f"{r['pct']:.2f}% of planned food"
        note = f"An experimental pre-meal estimate. Planned food: {r['planned']:,.0f} kg. {r['hist']}"
        flag = f'<div class="flag">{r["flag"]}</div>' if r["flag"] else ""
    return (f'<div class="res"><small>ESTIMATED FOOD WASTE</small><div class="num">{num}</div>'
            f'<div class="pct">{pct}</div><p>{note}</p>{flag}</div>'
            '<span class="chip">Experimental estimate from a model trained on synthetic data &middot; not a measurement of discarded food</span>')


def show_transparency(r):
    """Expanders built from the lookup result and the very DataFrame that was passed to model.predict()."""
    h, row = r["src"], r["row"]
    same_df, cat_df = source_frames(h)
    with st.expander("Historical Data Used — View Source Records"):
        for title, tbl in (("Previous Occurrence of the Same Menu", same_df), ("Previous Occurrence of the Same Meal Category", cat_df)):
            st.markdown(f"**{title}**")
            st.markdown(f'<span class="pct" style="margin:2px 0 10px">Source date: {tbl.iloc[0]["Date"]}</span>', unsafe_allow_html=True)
            st.dataframe(tbl, hide_index=True, width="stretch")
        if h["same_src"]["Data row"] == h["cat_src"]["Data row"]:
            st.caption("Both historical features happen to come from the same dataset row.")
        st.markdown("**Historical Features Used by the Model**")
        st.dataframe(historical_features_table(row, h), hide_index=True, width="stretch")
    with st.expander("Model Input Summary — View All Features"):
        st.dataframe(input_summary(row, h), hide_index=True, width="stretch")


model, model_err = load_model()
if model is not None and list(getattr(model, "feature_names_in_", FEATURES)) != FEATURES:
    model, model_err = None, f"The model's features {list(model.feature_names_in_)} do not match the app's {FEATURES}."
ds_name, ds_err = pick_dataset()
df, read_err = None, None
if ds_name:
    try:
        df = read_dataset(ds_name)
    except Exception as e:
        read_err = f"Could not read {ds_name}: {e}"
st.markdown(HERO, unsafe_allow_html=True)

# ---------------- Prediction ----------------
with st.container(key="predict"):
    st.markdown('<div class="c"><div class="tag">Intelligence</div><div class="h2">See Beyond<br>the Numbers.</div>'
                '<p class="sub">Describe a meal. FoodWise&rsquo;s machine learning model returns an experimental estimate of the food waste to expect.</p></div>'
                '<div style="height:56px"></div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large", vertical_alignment="top")
    hist_df = prepare(df) if df is not None else None
    last_date = hist_df["Date"].max().date() if hist_df is not None else date.today()
    remarks = sorted(df["Remark"].dropna().astype(str).str.strip().unique()) if df is not None else REMARKS
    with left:
        a, b = st.columns(2)
        meal = a.selectbox("Meal", MEALS, index=1)
        mdate = b.date_input("Meal date", value=last_date + timedelta(days=1), format="DD/MM/YYYY",
                             help="Only records dated before this day are used as history.")
        menus = menus_before(hist_df, meal, mdate) if hist_df is not None else []
        if menus:
            menu = st.selectbox("Menu", menus)
        else:
            menu = None
            st.selectbox("Menu", ["No menu history before this date"], disabled=True)
        a, b = st.columns(2)
        expected = a.number_input("Expected attendance", min_value=0, value=5000, step=50)
        prepared = b.number_input("Planned food (kg)", min_value=0.0, value=2800.0, step=10.0)
        remark = st.selectbox("Remark", remarks, index=remarks.index("Normal day") if "Normal day" in remarks else 0)
        go_btn = st.button("Estimate Food Waste", width="stretch")
    if go_btn:
        errs, res = [], None
        if model is None:
            errs.append(model_err)
        if hist_df is None:
            errs.append(ds_err or read_err)
        if expected <= 0:
            errs.append("Expected attendance must be greater than 0.")
        if prepared <= 0:
            errs.append("Planned food must be greater than 0 kg.")
        if not menu:
            errs.append("Please choose a menu that has an earlier record for this meal and date.")
        if not errs:
            hist, why = lookup(hist_df, meal, menu, mdate)
            if hist is None:
                errs.append(why)
            else:
                row = build_row(meal, hist, expected, prepared, remark, pd.Timestamp(mdate).day_name())
                try:
                    pred = float(model.predict(row)[0])
                    flag = ""
                    # Raw model output is always shown; invalid values are flagged, never clipped or hidden.
                    if pred < 0:
                        flag = "Flag: the model returned a negative value, which cannot be real waste. It is shown unaltered and is not a valid estimate."
                    elif pred > prepared:
                        flag = "Flag: the predicted waste exceeds the planned food quantity, which is not possible. It is shown unaltered and is not a valid estimate."
                    hs = f"History: same menu on {hist['same_date']:%d %b %Y}, previous {meal} on {hist['cat_date']:%d %b %Y}."
                    res = dict(pred=pred, pct=pred / prepared * 100, flag=flag, planned=prepared, hist=hs, src=hist, row=row)
                except Exception as e:
                    errs.append(f"The model could not produce an estimate: {e}")
        st.session_state["out"] = dict(res=res, errs=errs)
    out = st.session_state.get("out", dict(res=None, errs=[]))
    with right:
        st.markdown(result_html(out["res"]), unsafe_allow_html=True)
        if out["res"]:
            show_transparency(out["res"])
        for e in out["errs"]:
            st.error(e)
        if model_err and not out["errs"]:
            st.error(model_err)

# ---------------- Analytics ----------------
with st.container(key="analytics"):
    st.markdown('<div id="analytics" class="anchor"></div><div class="c"><div class="tag">Analytics</div><div class="h2">Patterns Worth<br>Discovering.</div>'
                '<p class="sub">Quiet, focused visualisations that turn records into something you can read at a glance.</p></div>'
                '<div style="height:40px"></div>', unsafe_allow_html=True)
    if df is None:
        st.error(ds_err or read_err)
    else:
        st.caption(f"Dataset: {ds_name} · {len(df):,} records · synthetic data, not real mess records. Folder: {DATA_DIR}")
        mc, wc, dc = col(df, "meal"), waste_col(df), col(df, "date")
        ac, pc, wd = col(df, "actual"), col(df, "preparedkg", "prepared"), col(df, "day")
        d = df.copy()
        if mc:
            d[mc] = d[mc].astype(str).str.strip().str.title()
        if wc:
            d[wc] = pd.to_numeric(d[wc], errors="coerce")
        c1, c2 = st.columns(2, gap="large")
        c3, c4 = st.columns(2, gap="large")
        with c1:
            head("Waste by meal", "Share of total recorded waste")
            if mc and wc:
                s = d.groupby(mc)[wc].sum().reindex(MEALS).dropna()
                show(style(go.Figure(go.Pie(labels=s.index, values=s.values, hole=0.7, sort=False, textinfo="percent", textfont=dict(color="#04271a", size=13),
                                            marker=dict(colors=[MEAL_COLOR[m] for m in s.index], line=dict(color="#0d1511", width=3)))), 360))
            else:
                st.info("This chart needs a Meal column and a waste column, which were not found.")
        with c2:
            fig, ttl, sub = None, "Waste trend", ""
        with c2:
            fig, ttl, sub = None, "Waste trend", ""

            if wc and (dc or wd):
                if dc:
                    dd = d.assign(
                        _d=pd.to_datetime(d[dc], errors="coerce")
                    ).dropna(subset=["_d", wc])

                    if dd["_d"].nunique() > 1:
                        daily = (
                            dd.groupby(dd["_d"].dt.normalize())[wc]
                            .sum()
                            .sort_index()
                        )

                        if len(daily) > 120:
                            s = daily.resample("W").mean().dropna()
                            ttl = "Weekly Waste Trend"
                            sub = "Average daily waste across all meals"
                        else:
                            s = daily
                            ttl = "Daily Waste Trend"
                            sub = "Total recorded waste across all meals"

                        fig = go.Figure(
                            go.Scatter(
                                x=s.index,
                                y=s.values,
                                mode="lines+markers",
                                line=dict(color=PAL[0], width=2.5),
                                marker=dict(size=5, color=PAL[0]),
                                fill="tozeroy",
                                fillcolor="rgba(47,174,128,0.10)"
                            )
                        )

                        fig.update_yaxes(
                            rangemode="tozero",
                            tickformat=",.0f",
                            title="Waste (kg)"
                        )

                        fig.update_xaxes(
                            title="Date",
                            tickformat="%b %Y" if len(s) > 45 else "%b %d",
                            nticks=7
                        )

                if fig is None and wd:
                    weekday_order = {day: i for i, day in enumerate(DAYS)}
                    weekday_data = d.copy()
                    weekday_data["_day"] = (
                        weekday_data[wd].astype(str).str.strip().str.title()
                    )

                    s = (
                        weekday_data.groupby("_day")[wc]
                        .sum()
                        .reindex(DAYS)
                        .dropna()
                    )

                    ttl = "Waste by Weekday"
                    sub = "Total recorded waste across meals for each weekday"

                    fig = go.Figure(
                        go.Scatter(
                            x=s.index,
                            y=s.values,
                            mode="lines+markers",
                            line=dict(color=PAL[0], width=2.5),
                            marker=dict(size=7, color=PAL[1])
                        )
                    )

                    fig.update_yaxes(
                        rangemode="tozero",
                        tickformat=",.0f",
                        title="Waste (kg)"
                    )
                    fig.update_xaxes(title="Day")

                head(ttl, sub or "Waste over time")

                if fig is not None:
                    show(style(fig))
                else:
                    st.info(
                        "Not enough valid date records to draw the waste trend."
                    )
            else:
                st.info(
                    "This chart needs a date or weekday column and a waste column."
                )
        with c3:
            head("Average by meal", "Average waste (kg) per meal record")
            if mc and wc:
                s = d.groupby(mc)[wc].mean().reindex(MEALS).dropna()
                show(style(go.Figure(go.Bar(x=s.index, y=s.values, marker_color=[MEAL_COLOR[m] for m in s.index])), 340))
            else:
                st.info("This chart needs a Meal column and a waste column, which were not found.")
        with c4:
            head("Attendance and preparation", "Actual students against food prepared (kg)")
            if ac and pc:
                sm = d[[ac, pc]].dropna()
                sm = sm.sample(min(len(sm), 1500), random_state=1)
                fig = go.Figure(go.Scatter(x=sm[ac], y=sm[pc], mode="markers", marker=dict(color=PAL[0], size=7, opacity=0.55)))
                fig.update_xaxes(title="Actual students")
                fig.update_yaxes(title="Food prepared (kg)")
                show(style(fig))
            else:
                st.info("This chart needs Actual and Prepared kg columns, which were not found.")

st.markdown(FOOT, unsafe_allow_html=True)