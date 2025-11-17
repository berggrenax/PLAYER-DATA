# pages/matchdata.py
# Körs automatiskt som multipage-sida i Streamlit

import re
import unicodedata
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gspread
from google.oauth2.service_account import Credentials

# ----------------------- SIDKONFIG / HEADER -----------------------
st.title("MATCH DASHBOARD")

# ----------------------- KONTROLLERA INLOGGNING -------------------
# Använd samma flagga som sätts i app.py
if not st.session_state.get("logged_in", False):
    st.warning("Du måste logga in på huvudsidan först.")
    st.stop()
# ----------------------- HJÄLPARE -----------------------
def _norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _coerce_numeric_cols(frame: pd.DataFrame, min_ratio: float = 0.7) -> pd.DataFrame:
    f = frame.copy()
    for c in f.columns:
        if f[c].dtype == object:
            s = f[c].astype(str).str.strip()
            looks_num = (
                s.str.match(r'^-?\d{1,3}([ .]?\d{3})*(,\d+|\.\d+)?$') |
                s.str.match(r'^-?\d+([,\.]\d+)?$')
            )
            if looks_num.mean() >= min_ratio:
                s = (s.replace({'\u00a0': ''}, regex=True)
                     .str.replace(' ', '', regex=False)
                     .str.replace('.', '', regex=False)
                     .str.replace(',', '.', regex=False))
                f[c] = pd.to_numeric(s, errors='coerce')
    return f

def _num(x):
    return pd.to_numeric(x, errors="coerce")

def _pct(series: pd.Series) -> pd.Series:
    """Hantera 75%, '75', '0.75' -> 75.0"""
    s = series.astype(str).str.strip()
    has_pct = s.str.contains("%")
    s = s.str.replace("%", "", regex=False).str.replace(",", ".", regex=False)
    out = pd.to_numeric(s, errors="coerce")
    if pd.notna(out.median()) and out.median() <= 1 and has_pct.any():
        out = out * 100.0
    return out

def parse_date_series(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip()
    # 1) EXAKT 'YYYY-MM-DD'
    dt = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
    # 2) Fallback för ev. Exceltal
    if dt.isna().any():
        as_num = pd.to_numeric(raw, errors='coerce')
        dt2 = pd.to_datetime(as_num, unit='d', origin='1899-12-30', errors='coerce')
        dt = dt.fillna(dt2)
    return dt

# ----------------------- GOOGLE SHEETS -----------------------
SHEET_ID = st.secrets.get("SHEET_ID", "1Pk4Ru2D0HI-gOqib_XVfXftadX_OCrXR2WZK85eFdRg")
WORKSHEET_NAME = "Blad2"

@st.cache_data(show_spinner=True)
def load_matches(sheet_id: str, worksheet_name: str) -> pd.DataFrame:
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    client = gspread.authorize(creds)
    ws = client.open_by_key(sheet_id).worksheet(worksheet_name)
    rows = ws.get_all_values()
    if not rows:
        raise RuntimeError("Google Sheet är tomt.")
    df = pd.DataFrame(rows[1:], columns=rows[0])

    # Numerifiera försiktigt
    df = _coerce_numeric_cols(df)

    # Datumkolumn
    date_col = next((c for c in df.columns if _norm(c) in {"date","datum"}), None)
    if date_col:
        df["_date"] = parse_date_series(df[date_col])
    else:
        df["_date"] = pd.NaT

    # Procentkolumner (inkl. Possession)
    pct_candidates = [c for c in df.columns if c.endswith("%") or _norm(c).endswith("%") or "accuracy" in _norm(c)]
    pct_candidates += [
        "Possession",
        "Pass_accuracy","SoTPB%","corners_shot%","att_w_shots%",
        "dribbles%","fwd_pass%","back_pass%","lat_pass%","prog_pass%","long_pass%",
        "pass_finalthird%","crosses%","attacks_left%","attacks_mid%","attacks_right%"
    ]
    for c in set(pct_candidates):
        if c in df.columns:
            df[c] = _pct(df[c])

    # xG/xGA om de finns
    if "xG" in df.columns:  df["_xg"]  = _num(df["xG"])
    if "xGA" in df.columns: df["_xga"] = _num(df["xGA"])

    return df

with st.sidebar:
    st.header("Data")
    try:
        matches = load_matches(SHEET_ID, WORKSHEET_NAME)
        st.success(f"Läst in {len(matches)} rader från {WORKSHEET_NAME}")
    except Exception as e:
        st.error("Kunde inte läsa matchdatan.")
        st.code(repr(e))
        st.stop()

    # Filter: Datum, Location, Opponent
    work = matches.copy()
    if "_date" in work.columns and work["_date"].notna().any():
        dmin, dmax = work["_date"].min().date(), work["_date"].max().date()
        dr = st.date_input("Datumintervall", (dmin, dmax))
        if isinstance(dr, tuple) and len(dr) == 2:
            start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
            work = work[(work["_date"].isna()) | ((work["_date"] >= start) & (work["_date"] <= end))]

    if "Location" in work.columns:
        locs = sorted([x for x in work["Location"].dropna().astype(str).unique().tolist()])
        sel = st.multiselect("Location", options=locs, default=locs)
        if sel: work = work[work["Location"].astype(str).isin(sel)]

    if "Opponent" in work.columns:
        opps = sorted([x for x in work["Opponent"].dropna().astype(str).unique().tolist()])
        sel_opp = st.multiselect("Motståndare", options=opps, default=opps)
        if sel_opp: work = work[work["Opponent"].astype(str).isin(sel_opp)]

st.caption(f"Rader efter filter: {len(work)}")
if work.empty:
    st.info("Inget att visa för nuvarande filter.")
    st.stop()

# Sortera på datum + nollställ index
if "_date" in work.columns and work["_date"].notna().any():
    work = work.sort_values("_date").reset_index(drop=True)
else:
    work = work.reset_index(drop=True)

# ----- Månad + färg + X-etikett ("Opponent mm-dd") -----
work["_month_num"]  = work["_date"].dt.month.fillna(0).astype(int)
work["_month_name"] = work["_date"].dt.strftime("%b").fillna("?")

MONTH_PALETTE = (
    px.colors.qualitative.Set2
    + px.colors.qualitative.Set3
    + px.colors.qualitative.Pastel
    + px.colors.qualitative.Safe
)
month_values = [m for m in work["_month_num"].unique().tolist() if m >= 0]
month_color_map = {m: MONTH_PALETTE[(m-1) % len(MONTH_PALETTE)] for m in month_values if m > 0}
month_color_map[0] = "#9e9e9e"
work["_month_color"] = work["_month_num"].map(month_color_map).fillna("#9e9e9e")

def _fmt_xlabel(row):
    opp = str(row.get("Opponent", "")).strip()
    if pd.notna(row.get("_date")):
        mmdd = pd.to_datetime(row["_date"]).strftime("%m-%d")
        return f"{opp} {mmdd}" if opp else mmdd
    return opp or "Match"

work["_label"] = work.apply(_fmt_xlabel, axis=1)

# ======================= SAMMANFATTNING =======================
st.subheader("Sammanfattning")

def _safe_mean(col):
    if col not in work.columns: return np.nan
    return _num(work[col]).mean()

def _safe_sum(col):
    if col not in work.columns: return np.nan
    return _num(work[col]).sum()

sum_cols = {
    "Matcher": len(work),
    "Poäng (sum)": int(_safe_sum("points")) if "points" in work.columns else int(0),
    "GS": int(_safe_sum("GS")) if "GS" in work.columns else int(_safe_sum("_gf")),
    "GC": int(_safe_sum("GC")) if "GC" in work.columns else int(_safe_sum("_ga")),
    "Possession (medel %)": _safe_mean("Possession"),
    "Avg age (medel)": _safe_mean("avg_age"),
    "xG (sum)": float(_safe_sum("_xg") if "_xg" in work.columns else _safe_sum("xG")),
    "xGA (sum)": float(_safe_sum("_xga") if "_xga" in work.columns else _safe_sum("xGA")),
}

c1, c2, c3, c4 = st.columns(4)
c1.metric("Matcher", sum_cols["Matcher"])
c2.metric("Poäng (sum)", sum_cols["Poäng (sum)"])
c3.metric("GS", sum_cols["GS"])
c4.metric("GC", sum_cols["GC"])

c5, c6, c7, c8 = st.columns(4)
c5.metric("Possession (medel)", "—" if np.isnan(sum_cols["Possession (medel %)"]) else f"{sum_cols['Possession (medel %)']:.1f}%")
c6.metric("Snittålder", "—" if np.isnan(sum_cols["Avg age (medel)"]) else f"{sum_cols['Avg age (medel)']:.2f}")
c7.metric("xG (sum)", "—" if np.isnan(sum_cols["xG (sum)"]) else f"{sum_cols['xG (sum)']:.2f}")
c8.metric("xGA (sum)", "—" if np.isnan(sum_cols["xGA (sum)"]) else f"{sum_cols['xGA (sum)']:.2f}")

st.markdown("---")

# ======================= KLICKBAR MATCHLISTA =======================
st.subheader("Games")

def _fmt_label_for_select(i: int) -> str:
    r = work.iloc[i]
    parts = []
    if pd.notna(r.get("_date")):
        parts.append(pd.to_datetime(r["_date"]).strftime("%Y-%m-%d"))
    if pd.notna(r.get("Opponent", None)):
        parts.append(str(r["Opponent"]))
    if pd.notna(r.get("Location", None)):
        parts.append(str(r["Location"]))
    if pd.notna(r.get("Matchday2", None)):
        parts.append(f"MD {r['Matchday2']}")
    return " | ".join(parts) if parts else f"Match {i+1}"

labels = [_fmt_label_for_select(i) for i in range(len(work))]

idx = st.selectbox(
    "Choose Game",
    options=list(range(len(work))),
    format_func=lambda i: labels[i],
    index=0,
    key="match_select",
)

row = work.iloc[idx]

st.write("### Game details")
detail_candidates = ["Opponent","date","Location","points","Possession","avg_age",
                     "xG","xGA","Shots","SoT","SoT_penaltybox","shots_outsidebox","SoT_outsidebox","Avg_shot_distance"]
detail_cols = [c for c in detail_candidates if c in work.columns]
st.dataframe(row[detail_cols].to_frame(name="Värde"))

st.markdown("---")

# ======================= HJÄLPFUNKTIONER FÖR DIAGRAM =======================
def _x_and_colors(plot: pd.DataFrame):
    """Gemensam X (kategori) och färg per rad."""
    x = plot["_label"] if "_label" in plot.columns else pd.Series([f"Match {i+1}" for i in range(len(plot))])
    colors = plot.get("_month_color", pd.Series(["#777777"] * len(plot))).tolist()
    return x, colors

def bar_with_mean(df, col, title):
    plot = df.copy()
    y = pd.to_numeric(plot[col], errors="coerce").fillna(0)
    x, colors = _x_and_colors(plot)

    fig = go.Figure(go.Bar(x=x, y=y, marker=dict(color=colors)))
    fig.update_layout(
        template="plotly_white", height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        title=title, xaxis_title="Match", yaxis_title=col,
        xaxis=dict(type="category", categoryorder="array", categoryarray=list(x))
    )

    mean_val = float(y.mean()) if len(y) else np.nan
    if pd.notna(mean_val):
        fig.add_hline(
            y=mean_val, line=dict(dash="dash", width=2),
            annotation_text=f"Medel: {mean_val:.1f}",
            annotation_position="top left"
        )

    # Månad-legend via dummy-spår
    legend_added = set()
    month_nums = plot.get("_month_num", pd.Series([0]*len(plot)))
    month_names = plot.get("_month_name", pd.Series(["?"]*len(plot)))
    for mn in month_nums.unique():
        if mn in legend_added: continue
        mask = month_nums == mn
        name  = month_names[mask].iloc[0] if mask.any() else "?"
        color = plot.loc[mask, "_month_color"].iloc[0] if mask.any() else "#777777"
        fig.add_bar(x=[None], y=[None], name=str(name), marker=dict(color=color),
                    showlegend=True, hoverinfo="skip")
        legend_added.add(mn)

    return fig

def stacked_total_fill_pct(df, total_col, success_col, pct_col, title, hover_cols=None):
    """
    Draw a stacked bar where SUCCESS is the bottom (darker) part and the REST on top (lighter).
    - Shows % as text centered on the bar (if pct_col provided).
    - Hover shows: Match, attempts (total), successful, % and optional extra fields.
    - No duplicate hover tooltips and no vertical spikeline.
    """
    plot = df.copy()
    plot[total_col]   = pd.to_numeric(plot[total_col],  errors="coerce").fillna(0)
    plot[success_col] = pd.to_numeric(plot[success_col], errors="coerce").fillna(0)
    plot["_rest"]     = (plot[total_col] - plot[success_col]).clip(lower=0)

    x, colors = _x_and_colors(plot)

    # Build hover text for the SUCCESS trace only
    pct_vals = None
    if pct_col and pct_col in plot.columns:
        pct_vals = pd.to_numeric(plot[pct_col], errors="coerce")

    hover_success = []
    for i in range(len(plot)):
        lines = [
            f"{total_col}: {int(plot.iloc[i][total_col])}",
            f"{success_col}: {int(plot.iloc[i][success_col])}",
        ]
        if pct_vals is not None and pd.notna(pct_vals.iloc[i]):
            lines.append(f"{pct_col}: {pct_vals.iloc[i]:.1f}%")
        if hover_cols:
            for k in hover_cols:
                v = plot.iloc[i].get(k, 0)
                try:
                    # pretty number if numeric
                    v = f"{float(v):.1f}" if isinstance(v, (int, float, np.floating)) else v
                except Exception:
                    pass
                lines.append(f"{k}: {v}")
        hover_success.append("<br>".join(lines))

    # % label centered on the success part
    pct_label = None
    if pct_vals is not None:
        pct_label = pct_vals.apply(lambda v: f"{v:.1f}%" if pd.notna(v) else "")

    fig = go.Figure()

    # SUCCESS (bottom, darker) – carries the hover + the visible % text
    fig.add_bar(
        x=x,
        y=plot[success_col],
        name="Lyckat",
        marker=dict(color=colors, opacity=0.9),
        hovertemplate="Match: %{x}<br>%{text}<extra></extra>",
        text=hover_success,                 # used in hovertemplate
        textposition="none",                # don't render hover text on bars
        customdata=np.stack([pct_label.fillna("")], axis=-1) if pct_label is not None else None
    )

    # REST (top, lighter) – no hover (prevents duplicate)
    fig.add_bar(
        x=x,
        y=plot["_rest"],
        name="Övrigt",
        marker=dict(color=colors, opacity=0.35),
        hoverinfo="skip"
    )

    # Add % as visible text on the WHOLE height (anchor at total height)
    if pct_label is not None:
        fig.add_scatter(
            x=x,
            y=plot[total_col],
            mode="text",
            text=pct_label,
            textposition="middle center",
            textfont=dict(size=12, color="#111"),
            showlegend=False,
            hoverinfo="skip"
        )

    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        title=title,
        xaxis_title="Match",
        yaxis_title=total_col,
        legend_title="Del",
        xaxis=dict(type="category", categoryorder="array", categoryarray=list(x)),
        hovermode="x unified",           # one hover box per x
    )
    # remove vertical spikeline
    fig.update_xaxes(showspikes=False)

    # Month legend (dummy traces)
    legend_added = set()
    month_nums = plot.get("_month_num", pd.Series([0]*len(plot)))
    month_names = plot.get("_month_name", pd.Series(["?"]*len(plot)))
    for mn in month_nums.unique():
        if mn in legend_added:
            continue
        mask  = month_nums == mn
        name  = month_names[mask].iloc[0] if mask.any() else "?"
        color = plot.loc[mask, "_month_color"].iloc[0] if mask.any() else "#777777"
        fig.add_bar(x=[None], y=[None], name=str(name), marker=dict(color=color),
                    showlegend=True, hoverinfo="skip")
        legend_added.add(mn)

    return fig

def two_color_stack(df, left_col, right_col, title,
                    left_color="#CC4C4C", right_color="#2BA84A"):
    plot = df.copy()
    plot[left_col]  = pd.to_numeric(plot[left_col], errors="coerce").fillna(0)
    plot[right_col] = pd.to_numeric(plot[right_col], errors="coerce").fillna(0)

    x, _ = _x_and_colors(plot)

    fig = go.Figure()
    fig.add_bar(x=x, y=plot[left_col],  name=left_col,  marker=dict(color=left_color))
    fig.add_bar(x=x, y=plot[right_col], name=right_col, marker=dict(color=right_color))
    fig.update_layout(
        barmode="stack", template="plotly_white", height=420,
        margin=dict(l=10,r=10,t=40,b=10),
        title=title, xaxis_title="Match", yaxis_title="Antal",
        xaxis=dict(type="category", categoryorder="array", categoryarray=list(x))
    )
    return fig

def triple_stack(df, c1, c2, c3, title, names=None):
    plot = df.copy()
    for c in [c1, c2, c3]:
        plot[c] = pd.to_numeric(plot[c], errors="coerce").fillna(0)

    x, _ = _x_and_colors(plot)

    n1, n2, n3 = names or (c1, c2, c3)
    fig = go.Figure()
    fig.add_bar(x=x, y=plot[c1], name=n1)
    fig.add_bar(x=x, y=plot[c2], name=n2)
    fig.add_bar(x=x, y=plot[c3], name=n3)
    fig.update_layout(
        barmode="stack", template="plotly_white", height=420,
        margin=dict(l=10,r=10,t=40,b=10),
        title=title, xaxis_title="Match", yaxis_title="Summa",
        xaxis=dict(type="category", categoryorder="array", categoryarray=list(x))
    )
    return fig

def pie_sum(df, cols, title):
    vals = [pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0).sum() for c in cols]
    fig = go.Figure(go.Pie(labels=cols, values=vals, hole=0.45))
    fig.update_layout(template="plotly_white", height=420,
                      margin=dict(l=10,r=10,t=40,b=10), title=title)
    return fig

# ======================= DIAGRAM =======================
st.header("Diagram")

# Possession – enkel stapel med medel (0–100%)
if "Possession" in work.columns:
    st.subheader("Possession (%)")
    fig = bar_with_mean(work, "Possession", "Possession per match")
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)

# Passningar: total/accurate + % inuti stapeln, hover visar attempts/lyckade/%
if {"Passes_total","Accurate_passes"}.issubset(work.columns):
    st.subheader("Total Passes")
    fig = stacked_total_fill_pct(work, "Passes_total", "Accurate_passes", "Pass_accuracy", "")
    st.plotly_chart(fig, use_container_width=True)

# Skott: Shots (total) + SoT (fyll), ingen procentsiffra på stapeln, med hover-info
shots_hover = [
    "SoT_penaltybox", "SoTPB%", "shots_outsidebox",
    "SoT_outsidebox", "SoToutside%", "Avg_shot_distance"
]
fig = stacked_total_fill_pct(
    work,
    total_col="Shots",
    success_col="SoT",
    pct_col=None,           # ingen % på stapeln
    title="Shots",
    hover_cols=shots_hover  # visas i hover
)
st.plotly_chart(fig, use_container_width=True)

# Skott emot: Shots_allowed/SoT_allowed
if {"Shots_allowed","SoT_allowed"}.issubset(work.columns):
    st.subheader("Shots Against")
    fig = stacked_total_fill_pct(work, "Shots_allowed", "SoT_allowed", None, "")
    st.plotly_chart(fig, use_container_width=True)

# Hörnor: corners / corners_shot + % inne i stapeln
if {"corners","corners_shot"}.issubset(work.columns):
    st.subheader("Corners")
    fig = stacked_total_fill_pct(work, "corners", "corners_shot", "corners_shot%", "")
    st.plotly_chart(fig, use_container_width=True)

# Attacker: total_att / att_w_shots + % inne i stapeln
if {"total_att","att_w_shots"}.issubset(work.columns):
    st.subheader("Attacks")
    fig = stacked_total_fill_pct(work, "total_att", "att_w_shots", "att_w_shots%", "")
    st.plotly_chart(fig, use_container_width=True)

# PPDA – enkel stapel med medel
if "PPDA" in work.columns:
    st.subheader("PPDA")
    st.plotly_chart(bar_with_mean(work, "PPDA", "PPDA"), use_container_width=True)

# Duels% – enkel stapel med medel (0–100%)
if "Duels%" in work.columns:
    st.subheader("Duels %")
    fig = bar_with_mean(work, "Duels%", "Duels%")
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)

# Fouls vs suffered – stack (röd/grön)
if {"fouls","fouls_suffered"}.issubset(work.columns):
    st.subheader("Fouls")
    st.plotly_chart(two_color_stack(work, "fouls", "fouls_suffered", "Fouls / Suffered"), use_container_width=True)

# Dribbles – total, beräknad lyckade från dribbles%
if "dribbles" in work.columns:
    st.subheader("Dribbles")
    tmp = work.copy()
    tmp["dribbles"] = _num(tmp["dribbles"])
    if "dribbles%" in tmp.columns:
        tmp["dribbles%"] = _pct(tmp["dribbles%"])
        tmp["_dribbles_succ"] = (tmp["dribbles"] * (tmp["dribbles%"]/100.0)).round()
        fig = stacked_total_fill_pct(tmp, "dribbles", "_dribbles_succ", "dribbles%", "")
    else:
        fig = bar_with_mean(tmp, "dribbles", "Dribblingar (totalt)")
    st.plotly_chart(fig, use_container_width=True)

# Pass-detaljer (alla med % i stapeln och hover)
pass_sets = [
    ("fwd_pass",        "fwd_pass_acc",        "fwd_pass%",          "Forward Passes"),
    ("back_passes",     "back_pass_acc",       "back_pass%",         "Back Passes"),
    ("lat_pass",        "lat_passacc",         "lat_pass%",          "Lateral Passes"),
    ("prog_pass",       "prog_pass_acc",       "prog_pass%",         "Progressive Passes"),
    ("pass_finalthird", "pass_finalthirdacc",  "pass_finalthird%",   "Passes Final Third"),
    ("crosses",         "crosses_acc",         "crosses%",           "Crosses"),
]
for total_col, success_col, pct_col, title in pass_sets:
    if {total_col, success_col}.issubset(work.columns):
        st.subheader(title)
        fig = stacked_total_fill_pct(work, total_col, success_col, pct_col, title)
        st.plotly_chart(fig, use_container_width=True)

# Genomsnittlig passlängd – enkel stapel med medel
if "avgpass_length" in work.columns:
    st.subheader("Genomsnittlig passlängd (m)")
    st.plotly_chart(bar_with_mean(work, "avgpass_length", "Avg pass length"), use_container_width=True)

# Attack-fördelning %
if {"attacks_left%","attacks_mid%","attacks_right%"}.issubset(work.columns):
    st.subheader("Attackfördelning (%)")
    st.plotly_chart(triple_stack(work, "attacks_left%", "attacks_mid%", "attacks_right%", "Attack origin %", names=("Left%","Mid%","Right%")), use_container_width=True)

# xG per riktning
if {"xG_left","xG_mid","xG_right"}.issubset(work.columns):
    st.subheader("xG per riktning")
    st.plotly_chart(triple_stack(work, "xG_left", "xG_mid", "xG_right", "xG origin", names=("xG Left","xG Mid","xG Right")), use_container_width=True)

# Attacker (antal) per riktning
if {"attleft","attmid","attright"}.issubset(work.columns):
    st.subheader("Attacker (antal) per riktning")
    st.plotly_chart(triple_stack(work, "attleft", "attmid", "attright", "Attacks divided", names=("Left","Mid","Right")), use_container_width=True)

# Pie: Målzon (goals_inside vs goals_out)
if {"goals_inside","goals_out"}.issubset(work.columns):
    st.subheader("Målzon (summerat)")
    st.plotly_chart(pie_sum(work, ["goals_inside","goals_out"], "Goals Inside vs Outside box"), use_container_width=True)

# Pie: måltyper
goal_type_cols = [c for c in ["goals_cross","g_play_left","g_play_right","g_play_mid","g_long_ball","g_penalty","g_set_piece","g_press","g_owngoal"] if c in work.columns]
if goal_type_cols:
    st.subheader("Måltyper (summerat)")
    st.plotly_chart(pie_sum(work, goal_type_cols, "Goal types"), use_container_width=True)
