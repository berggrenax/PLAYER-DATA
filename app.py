# app.py
# Kör: streamlit run app.py

import re
import unicodedata
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pandas.api.types import is_numeric_dtype
import gspread
from google.oauth2.service_account import Credentials

# ----------------------- SIDKONFIG -----------------------
st.set_page_config(page_title="PLAYER DATA", layout="wide")
st.title("PLAYER DATA")

# ---- SIMPLE PASSWORD PROTECTION ----
import streamlit as st

APP_PASSWORD = st.secrets["APP_PASSWORD"]

def check_password():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        return True

    password = st.text_input("Ange lösenord:", type="password")
    if st.button("Logga in"):
        if password == APP_PASSWORD:
            st.session_state.logged_in = True
            return True
        else:
            st.error("Fel lösenord")

    return False

if not check_password():
    st.stop()
# ---- END PASSWORD PROTECTION ----

# ----------------------- HJÄLPFUNKTIONER -----------------------
def _norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _coerce_numeric_cols(frame: pd.DataFrame, min_ratio: float = 0.7) -> pd.DataFrame:
    """Försök göra objekt-kolumner numeriska (hanterar t.ex. 1 234,56)."""
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

def _detect_ratio_for_pct(colname: str, cols: set[str]) -> tuple[str|None, str|None]:
    """
    Returnerar (numerator_col, denominator_col) för pct-kolumn.
    Täcker: *_accuracy_pct, *_acc_pct, *_win_pct, *_success_pct, *_succ_pct + generiskt mönster.
    """
    n = colname
    n_norm = _norm(n)

    def pick(firsts, seconds):
        for a in firsts:
            for b in seconds:
                if a in cols and b in cols:
                    return a, b
        return None, None

    if n_norm.endswith("accuracy_pct"):
        base = n[:len(n) - len("_accuracy_pct")]
        return pick([f"{base}_accurate", f"{base}_acc"], [f"{base}_total", f"{base}_att"])

    if n_norm.endswith("_acc_pct"):
        base = n[:len(n) - len("_acc_pct")]
        return pick([f"{base}_acc", f"{base}_accurate"], [f"{base}_att", f"{base}_total"])

    if n_norm.endswith("_win_pct"):
        base = n[:len(n) - len("_win_pct")]
        return pick([f"{base}_won"], [f"{base}_att", f"{base}_total"])

    if n_norm.endswith("_success_pct"):
        base = n[:len(n) - len("_success_pct")]
        return pick([f"{base}_successful", f"{base}_succ"], [f"{base}_total", f"{base}_att"])

    if n_norm.endswith("_succ_pct"):
        base = n[:len(n) - len("_succ_pct")]
        return pick([f"{base}_succ", f"{base}_successful"], [f"{base}_att", f"{base}_total"])

    patterns = [
        (["_acc", "_accurate"], ["_att", "_total"]),
        (["_won"], ["_att", "_total"]),
        (["_succ", "_successful"], ["_att", "_total"]),
    ]
    for num_tokens, den_tokens in patterns:
        for token_num in num_tokens:
            if token_num in n_norm:
                base = n[:n_norm.index(token_num)]
                nums = [base + t for t in num_tokens]
                dens = [base + t for t in den_tokens]
                a, b = pick(nums, dens)
                if a and b:
                    return a, b
    return None, None

# ----------------------- GOOGLE SHEETS -----------------------
with st.sidebar:
    st.header("Data")
    st.caption("Läser automatiskt från Google.")

SHEET_ID = "1Pk4Ru2D0HI-gOqib_XVfXftadX_OCrXR2WZK85eFdRg"   # bara ID
WORKSHEET_NAME = "Blad1"

@st.cache_data(show_spinner=True)
def load_from_sheets(sheet_id: str, worksheet_name: str) -> pd.DataFrame:
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
    return df

try:
    df = load_from_sheets(SHEET_ID, WORKSHEET_NAME)
    st.sidebar.success("Ansluten till Google Sheets")
except Exception as e:
    import traceback
    st.sidebar.error("Kunde inte läsa Google Sheet.")
    st.code(f"{type(e).__name__}: {e}")
    st.code(''.join(traceback.format_exc())[-2000:])
    st.stop()

with st.sidebar:
    if st.button("Uppdatera data"):
        load_from_sheets.clear()
        st.rerun()
# Efter df = load_from_sheets(...)

# --- DIFFERENTIALS ---
if "Goals" in df.columns and ("xG" in df.columns or "XG" in df.columns):
    g_col  = "Goals"
    xg_col = "xG" if "xG" in df.columns else "XG"
    df["Goal differential"] = pd.to_numeric(df[g_col], errors="coerce") - pd.to_numeric(df[xg_col], errors="coerce")

if "Assists" in df.columns and ("xA" in df.columns or "XAssists" in df.columns):
    a_col  = "Assists"
    xa_col = "xA" if "xA" in df.columns else "XAssists"
    df["Assist differential"] = pd.to_numeric(df[a_col], errors="coerce") - pd.to_numeric(df[xa_col], errors="coerce")

# ----------------------- DERIVERA % & TYPER -----------------------
def _norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _find_col(df: pd.DataFrame, name_or_aliases) -> str | None:
    """Hitta kolumn via normaliserad jämförelse; tar en str eller en lista alias."""
    targets = name_or_aliases if isinstance(name_or_aliases, (list, tuple)) else [name_or_aliases]
    targets = [_norm(t) for t in targets]
    for c in df.columns:
        if _norm(c) in targets:
            return c
    return None

def _ensure_col(df: pd.DataFrame, name: str):
    """Skapa tom kolumn (NaN) om den saknas, så att den blir valbar i X/Y."""
    if name not in df.columns:
        df[name] = np.nan

def _safe_pct(df: pd.DataFrame, num_hints, den_hints, out_name: str):
    """
    out = (num/den)*100 med skydd:
      - saknas num/den → skapa out som NaN
      - den <= 0 eller ej numerisk → NaN (inte ∞)
    """
    num_col = _find_col(df, num_hints)
    den_col = _find_col(df, den_hints)
    if not num_col or not den_col:
        _ensure_col(df, out_name)
        return
    num = pd.to_numeric(df[num_col], errors="coerce")
    den = pd.to_numeric(df[den_col], errors="coerce")
    out = np.where((den > 0) & np.isfinite(den), (num / den) * 100.0, np.nan)
    df[out_name] = out

# ✅ De fyra du vill ha – säkerställ att de finns på matchnivå
_safe_pct(df, ["Aerial_duels_won","Aerials_won"],
             ["Aerial_duels_att","Aerials_att","Aerial_duels_total"],
             "Aerial_duels_win_pct")

_safe_pct(df, ["Dribbles_succ","Dribbles_successful"],
             ["Dribbles_att","Dribbles_total"],
             "Dribbles_succ_pct")

_safe_pct(df, ["Crosses_acc","Crosses_accurate"],
             ["Crosses_att","Crosses_total"],
             "Crosses_acc_pct")

_safe_pct(df, ["Long_passes_acc","Long_balls_acc","Long_balls_accurate"],
             ["Long_passes_att","Long_balls_att","Long_balls_total"],
             "Long_passes_acc_pct")

# (behåll gärna fler _safe_pct-anrop om du vill skapa andra procent)

# Rensa ev. % och gör *_pct numeriska
for c in df.columns:
    if str(c).endswith("_pct"):
        df[c] = (
            df[c].astype(str)
                 .str.replace("%", "", regex=False)
                 .str.strip()
        )

# Allmän numerisk konvertering + städa ∞
df = _coerce_numeric_cols(df)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Position alltid kategori (även 1–12)
if "Position" in df.columns:
    df["Position"] = df["Position"].astype(str)

# ----------------------- FÄLTMAPPNING -----------------------
cols = df.columns.tolist()

def _guess_col(explicit_name, keywords):
    if explicit_name in cols:
        return explicit_name
    for k in keywords:
        for c in cols:
            if k in str(c).lower():
                return c
    return None

player_col   = _guess_col("Player",   ["spelare", "player", "namn", "name"]) or _guess_col("A", [])
position_col = _guess_col("Position", ["position", "pos"]) or _guess_col("B", [])
date_col     = _guess_col("Datum",    ["datum", "date"]) or _guess_col("C", [])
opponent_col = _guess_col("Opponent", ["opponent", "motst", "lag", "team"]) or _guess_col("D", [])

num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
minutes_col  = "Minutes" if "Minutes" in num_cols_all else _guess_col("E", ["minut", "minutes", "mins", "played"])
if minutes_col not in num_cols_all:
    minutes_col = None

# ----------------------- FILTER -----------------------
with st.sidebar:
    st.header("Filter")

work = df.copy()
# Se till att dessa fyra är float i 'work' (så de kommer med i numeric_cols_base)
for _col in ["Aerial_duels_win_pct","Dribbles_succ_pct","Crosses_acc_pct","Long_passes_acc_pct"]:
    if _col in work.columns:
        work[_col] = pd.to_numeric(work[_col], errors="coerce")
    else:
        work[_col] = np.nan  # skapa kolumnen om den saknas efter filter

# Position med select-all/rensa
if position_col and position_col in work.columns:
    pos_vals = sorted(work[position_col].dropna().astype(str).unique().tolist(), key=lambda x: x.lower())
    colA, colB = st.sidebar.columns(2)
    if colA.button("Välj alla positioner"):
        st.session_state["pos_sel_all"] = pos_vals
    if colB.button("Rensa positioner"):
        st.session_state["pos_sel_all"] = []
    default_pos = st.session_state.get("pos_sel_all", pos_vals)
    pos_sel = st.sidebar.multiselect("Position(er)", options=pos_vals, default=default_pos)
    if pos_sel:
        work = work[work[position_col].astype(str).isin([str(v) for v in pos_sel])]

# Motståndare
if opponent_col and opponent_col in work.columns:
    opp_vals = sorted(work[opponent_col].dropna().astype(str).unique().tolist(), key=lambda x: x.lower())
    opp_sel = st.sidebar.multiselect("Motståndare", options=opp_vals, default=opp_vals)
    if opp_sel:
        work = work[work[opponent_col].astype(str).isin([str(v) for v in opp_sel])]

# Spelare med select-all/rensa
if player_col and player_col in work.columns:
    st.sidebar.subheader("Spelare")
    player_vals = (
        work[player_col]
        .dropna().astype(str)
        .sort_values(key=lambda s: s.str.lower())
        .unique().tolist()
    )
    colP1, colP2 = st.sidebar.columns(2)
    if colP1.button("Välj alla spelare"):
        st.session_state["player_sel_all"] = player_vals
    if colP2.button("Rensa spelare"):
        st.session_state["player_sel_all"] = []
    default_players = st.session_state.get("player_sel_all", player_vals)

    search = st.sidebar.text_input("Sök spelare", value="")
    options = [p for p in player_vals if search.lower() in p.lower()] if search else player_vals
    # sync default med ev. sökresultat
    default_players = [p for p in default_players if p in options] or options
    selected_players = st.sidebar.multiselect("Välj spelare", options=options, default=default_players)
    if selected_players:
        work = work[work[player_col].astype(str).isin(selected_players)]

# Datum
if date_col and date_col in work.columns:
    raw = work[date_col].astype(str).str.strip()
    dt = pd.to_datetime(raw, errors='coerce', dayfirst=True, infer_datetime_format=True)
    if dt.notna().sum() < max(1, int(0.5 * len(raw))):
        as_num = pd.to_numeric(raw, errors='coerce')
        dt2 = pd.to_datetime(as_num, unit='d', origin='1899-12-30', errors='coerce')
        dt = dt.fillna(dt2)
    work = work.assign(_date=dt)
    parsed_ok = int(work['_date'].notna().sum()); total_rows = int(len(work))
    st.sidebar.caption(f"Datum tolkade: {parsed_ok}/{total_rows}")
    if work['_date'].notna().any():
        dmin = work['_date'].min().date(); dmax = work['_date'].max().date()
        drange = st.sidebar.date_input("Datumintervall", value=(dmin, dmax))
        include_unknown = st.sidebar.checkbox("Inkludera rader med okänt datum", value=True)
        if isinstance(drange, tuple) and len(drange) == 2:
            start, end = pd.to_datetime(drange[0]), pd.to_datetime(drange[1])
            known_mask = work['_date'].notna()
            in_range = (work['_date'] >= start) & (work['_date'] <= end)
            if include_unknown:
                work = work[(~known_mask) | in_range]
            else:
                work = work[known_mask & in_range]

st.caption(f"Rader efter filter: {len(work)}")

# ----------------------- KOLUMNLISTOR -----------------------
numeric_cols_base = work.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols_base = work.select_dtypes(exclude=[np.number]).columns.tolist()
if len(numeric_cols_base) < 1:
    st.warning("Hittade inga numeriska kolumner efter filter.")
    st.stop()
# --- VYER ---
tab_scatter, tab_leader = st.tabs([" Scatterplot", " Leaderboards"])
# ----------------------- UI PLOTT -----------------------
with tab_scatter:
    with st.sidebar:
        st.header("Plottvisning")
        level = st.radio("Spelare / Match", ["Match", "Spelare"], index=0)
        agg_label = st.selectbox("Aggregat för övriga numeriska kolumner", ["Summa", "Medel", "Median"], index=0)
        position_agg = st.selectbox("Positions-aggregation", ["Mode (vanligast)", "Median (numerisk)", "Första", "Sista"], index=0)
        x_col = st.selectbox("X-axel", options=numeric_cols_base, index=0)
        y_col = st.selectbox("Y-axel", options=numeric_cols_base, index=1 if len(numeric_cols_base) > 1 else 0)
        x_per90 = st.checkbox("X per 90", value=False)
        y_per90 = st.checkbox("Y per 90", value=False)
        color_col = st.selectbox("Färg efter... (valfritt)", options=[None] + categorical_cols_base + numeric_cols_base, index=0)
        size_col  = st.selectbox("Punktstorlek (valfritt)", options=[None] + numeric_cols_base, index=0)

    # ----------------------- AGGREGATION (SPELARE) -----------------------
    plot_df = work.copy()

    if level == "Spelare" and player_col:
        agg_map = {"Summa": np.sum, "Medel": np.mean, "Median": np.median}
        agg_fn = agg_map.get(agg_label or "Summa", np.sum)

        all_cols = plot_df.columns.tolist()
        col_set = set(all_cols)
        num_cols = plot_df.select_dtypes(include=[np.number]).columns.tolist()

        def looks_like_percent_name(name: str) -> bool:
            s = str(name).lower()
            return ("%" in s) or ("pct" in s) or s.endswith("_pct")

        excluded = {position_col} if position_col else set()
        skip_cols = {"Avg_pass_length_m"}  # viktas separat
        base_cols = [c for c in num_cols if (c not in excluded) and (c not in skip_cols) and (not looks_like_percent_name(c))]

        grp = plot_df.groupby(player_col, dropna=False)

        def _agg_for(col):
            # minutes ska alltid summeras
            if minutes_col and col == minutes_col:
                return np.sum
            return agg_fn

        result = grp[base_cols].agg({c: _agg_for(c) for c in base_cols}).reset_index()
        result["matches"] = grp.size().values

        # Position separat
        if position_col and position_col in plot_df.columns:
            if position_agg == "Median (numerisk)" and is_numeric_dtype(plot_df[position_col]):
                pos_series = grp[position_col].median()
            elif position_agg == "Första":
                pos_series = grp[position_col].first()
            elif position_agg == "Sista":
                pos_series = grp[position_col].last()
            else:
                pos_series = grp[position_col].agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan)
            result = result.merge(
                pos_series.reset_index().rename(columns={position_col: position_col}),
                on=player_col,
                how="left"
            )

        # Viktat medel för Avg_pass_length_m (vikter = Passes_total)
        if ("Avg_pass_length_m" in plot_df.columns) and ("Passes_total" in plot_df.columns):
            tmp = plot_df[[player_col, "Avg_pass_length_m", "Passes_total"]].copy()
            tmp["Avg_pass_length_m"] = pd.to_numeric(tmp["Avg_pass_length_m"], errors="coerce")
            tmp["Passes_total"]      = pd.to_numeric(tmp["Passes_total"], errors="coerce")
            tmp["_prod"] = tmp["Avg_pass_length_m"] * tmp["Passes_total"]
            num = tmp.groupby(player_col)["_prod"].sum(min_count=1)
            den = tmp.groupby(player_col)["Passes_total"].sum(min_count=1)
            wavg = (num / den).rename("Avg_pass_length_m").reset_index()
            result = result.drop(columns=["Avg_pass_length_m"], errors="ignore").merge(wavg, on=player_col, how="left")

        # Räkna om ALLA *_pct: sum(tälj)/sum(nämn) * 100
        pct_cols = [c for c in all_cols if str(c).endswith("_pct")]
        if pct_cols:
            pct_frames = []
            sums_cache = {}
            for pct_col in pct_cols:
                num_col, den_col = _detect_ratio_for_pct(pct_col, col_set)
                if not num_col or not den_col:
                    continue
                key_num = ("num", num_col); key_den = ("den", den_col)
                if key_num not in sums_cache:
                    sums_cache[key_num] = grp[num_col].sum(min_count=1)
                if key_den not in sums_cache:
                    sums_cache[key_den] = grp[den_col].sum(min_count=1)
                num_s = sums_cache[key_num]; den_s = sums_cache[key_den]
                pct = (num_s / den_s) * 100.0
                pct_frames.append(pct.rename(pct_col))
            if pct_frames:
                pct_df = pd.concat(pct_frames, axis=1).reset_index()
                result = result.merge(pct_df, on=player_col, how="left")

        plot_df = result

    # ----------------------- PER 90 -----------------------
    if (x_per90 or y_per90) and minutes_col is None:
        st.warning("Per 90 är valt men minutkolumn kunde inte hittas (ha t.ex. 'Minutes').")

    if (x_per90 or y_per90) and (minutes_col is not None) and (minutes_col in plot_df.columns):
        plot_df = plot_df[plot_df[minutes_col] > 0].copy()

    if x_per90 and (minutes_col in plot_df.columns if minutes_col else False):
        x_data = plot_df[x_col] / plot_df[minutes_col] * 90
        x_title = f"{x_col} (per90)"
    else:
        x_data = plot_df[x_col]; x_title = x_col

    if y_per90 and (minutes_col in plot_df.columns if minutes_col else False):
        y_data = plot_df[y_col] / plot_df[minutes_col] * 90
        y_title = f"{y_col} (per90)"
    else:
        y_data = plot_df[y_col]; y_title = y_col

    if plot_df.empty:
        st.warning("Inga rader att visa efter filter")
        st.stop()

    # ----------------------- HOVER-DATA -----------------------
    custom_cols = []; custom_headers = []
    for col, label in [(player_col, "Spelare"), (position_col, "Position"),
                    (date_col, "Datum"), (opponent_col, "Motståndare"),
                    (minutes_col, "Minuter")]:
        if col and col in plot_df.columns:
            custom_cols.append(plot_df[col]); custom_headers.append(label)
    customdata = np.column_stack(custom_cols) if custom_cols else None

    # ----------------------- SCATTERPLOT -----------------------
    fig = px.scatter(
        plot_df,
        x=x_data,
        y=y_data,
        color=color_col if color_col not in (None, "None") else None,
        size=size_col if size_col not in (None, "None") else None,
        hover_data=[],
        hover_name=None,
        template="plotly_white",
        height=520,
    )

    # Grundfärg röd om ingen färgdimension är vald
    if color_col in (None, "None"):
        fig.update_traces(marker=dict(color="red"))

    # Hovertemplate
    if customdata is not None:
        fig.update_traces(customdata=customdata)
        lines = [f"{lbl}: %{{customdata[{i}]}}" for i, lbl in enumerate(custom_headers)]
        hover_lines = "<br>".join(lines)
        fig.update_traces(hovertemplate=f"{hover_lines}<br>{x_title}: %{{x}}<br>{y_title}: %{{y}}<extra></extra>")
    else:
        fig.update_traces(hovertemplate=f"{x_title}: %{{x}}<br>{y_title}: %{{y}}<extra></extra>")

    # Svarta, större axelrubriker
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis_title_font=dict(size=18, color="black"),
        yaxis_title_font=dict(size=18, color="black"),
    )

    # Sätt procentformat på axlar om titel ser ut som %
    def _looks_pct_title(t):
        s = str(t).lower()
        return ("%" in s) or ("pct" in s) or ("procent" in s)

    if _looks_pct_title(x_title):
        fig.update_xaxes(ticksuffix="%", tickformat=".1f")
    if _looks_pct_title(y_title):
        fig.update_yaxes(ticksuffix="%", tickformat=".1f")

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------- DATAFÖRHANDSVISNING + EXPORT -----------------------
    with st.expander("Visa dataförhandsvisning"):
        st.dataframe(plot_df.head(200))

    @st.cache_data
    def to_csv_bytes(frame: pd.DataFrame) -> bytes:
        return frame.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Ladda ned data (CSV)",
        data=to_csv_bytes(plot_df),
        file_name="data_export.csv",
        mime="text/csv",
    )

    # ----------------------- leaderboard -----------------------
with tab_leader:
    st.markdown("---")
    st.header("Leaderboards")

    # --- Kategorival ---
    category = st.radio(
        "Välj kategori",
        ["Passningar", "Offensivt", "Dueller/Försvar"],
        horizontal=True,
    )

    # --- Globala kontroller ---
    colA, colB, colC = st.columns([1,1,2])
    with colA:
        per90_on = st.toggle("per90", value=False, help="Visa totals per 90 min där det är relevant.")
    with colB:
        min_minutes = st.slider("Min minuter (global)", 0, 3000, 0, step=30, help="Spelare under detta filtreras bort i alla listor.")
    with colC:
        min_attempts = st.slider(
            "Min försök (global)", 0, 500, 0, step=10,
            help=("För procentsatser används relevant nämnare per KPI (t.ex. Passes_total för Pass% / "
                  "Crosses_att för Cross% / Duels_total för Duels%). För totals utan % ignoreras detta."),
        )

    show_nonqual = st.checkbox("Visa även icke-kvalificerade rader", value=False)

    # --- Minutes-kolumn ---
    minutes_col = "Minutes" if "Minutes" in work.columns else None

    # ===================== Hjälpare för kolumnmatchning =====================
    all_cols_list = list(work.columns)

    def _resolve_col(name: str) -> str | None:
        target = _norm(name)
        for c in all_cols_list:
            if _norm(c) == target:
                return c
        return None

    def _first_existing(candidates: list[str]) -> str | None:
        for cand in candidates:
            rc = _resolve_col(cand)
            if rc is not None and rc in work.columns:
                return rc
        return None

    # ===================== UI-etiketter =====================
    DISPLAY_LABELS = {
        "Passes_total": "Total passes",
        "Passes_accurate": "Accurate passes",
        "passes_acc_pct": "Pass accuracy %",
        "Actions_total": "Total actions",
        "Actions_successful": "Actions successful",
        "Actions_success_pct": "Actions success %",
        "Progressive_passes_att": "Progressive passes attempted",
        "Progressive_passes_acc": "Progressive passes accurate",
        "Progressive_passes_acc_pct": "Progressive pass percent %",
        "Forward_passes_att": "Forward passes attempts",
        "Forward_passes_acc": "Forward passes accurate",
        "Forward_passes_acc_pct": "Forward pass %",
        "Back_passes_att": "Back passes attempted",
        "Back_passes_acc": "Back passes accurate",
        "Back_passes_acc_pct": "Back passes accuracy %",
        "Lateral_passes_att": "Lateral passes attempted",
        "Lateral_passes_acc": "Lateral passes accurate",
        "Lateral_passes_acc_pct": "Lateral pass %",
        "ShortMed_passes_att": "Short/Medium pass attempted",
        "ShortMed_passes_acc": "Short/Medium pass accurate attempts",
        "ShortMed_passes_acc_pct": "Short/Medium passes percent %",
        "Long_passes_att": "Long passes attempted",
        "Long_passes_acc": "Long passes accurate",
        "Long_passes_acc_pct": "Long passes accuracy %",
        "Passes_final_third_att": "Final 1/3 passes attempts",
        "Passes_final_third_acc": "Final 1/3 passes accurate",
        "Passes_final_third_acc_pct": "Final 1/3 pass %",
        "Through_passes_att": "Through passes attempts",
        "Through_passes_acc": "Through passes accurate",
        "Aerial_duels_att": "Aerial duels",
        "Aerial_duels_won": "Aerial duels won",
        "Aerial_duels_win_pct": "Aerial duels win %",
        "Def_duels_att": "Defensive duels",
        "Def_duels_won": "Defensive duels won",
        "Def_duels_win_pct": "Defensive duels win %",
        "Off_duels_att": "Offensive duels",
        "Off_duels_won": "Offensive duels won",
        "Off_duels_win_pct": "Offensive duels win %",
        "Dribbles_att": "Dribbles attempted",
        "Dribbles_succ": "Dribbles successful",
        "Dribbles_succ_pct": "Dribbles success %",
        "Crosses_att": "Crosses attempted",
        "Crosses_acc": "Crosses accurate",
        "Crosses_acc_pct": "Crosses accuracy %",
        "Goal differential": "Goal differential",
        "Assists": "Assists",
        "xA": "xA",
        "XAssists": "xA", 
        "Assist differential": "Assist differential",
        "Loose_ball_duels_att": "Loose ball duels",
        "Loose_ball_duels_won": "Loose ball duels won",
        "Sliding_tackles_att": "Slide tackles",
        "Sliding_tackles_won": "Slide tackles won",
    }

    def pretty_label(name: str) -> str:
        if name in DISPLAY_LABELS:
            return DISPLAY_LABELS[name]
        s = name.replace("_pct", " %").replace("_att", " attempted").replace("_acc", " accurate")
        s = s.replace("_", " ").replace("xg", "xG").replace("xa", "xA")
        return s[0].upper() + s[1:]

    # ===================== KPI-grupper (din ordning) =====================
    KPI_GROUPS = {
        "Passningar": [
            "Actions_total", "Actions_successful", "Actions_success_pct",
            "Passes_total", "Passes_accurate", "passes_acc_pct",
            "Forward_passes_att", "Forward_passes_acc", "Forward_passes_acc_pct",
            "Back_passes_att", "Back_passes_acc", "Back_passes_acc_pct",
            "Lateral_passes_att", "Lateral_passes_acc", "Lateral_passes_acc_pct",
            "ShortMed_passes_att", "ShortMed_passes_acc", "ShortMed_passes_acc_pct",
            "Long_passes_att", "Long_passes_acc", "Long_passes_acc_pct",
            "Progressive_passes_att", "Progressive_passes_acc", "Progressive_passes_acc_pct",
            "Passes_final_third_att", "Passes_final_third_acc", "Passes_final_third_acc_pct",
            "Through_passes_att", "Through_passes_acc",
            "Deep_completions", "Key_passes",
            "Shot_assists",
            "Second_assists",
            "Avg_pass_length_m",
        ],
        "Offensivt": [
            "Goals", "xG", "Goal differential",
            "Assists", "xA", "Assist differential",
            "Dribbles_att", "Dribbles_succ", "Dribbles_succ_pct",
            "Crosses_att", "Crosses_acc", "Crosses_acc_pct",
            "Shots_total", "Shots_on_target",
            "Touches_in_box",
            "Fouls_suffered",
        ],
        "Dueller/Försvar": [
            "Def_duels_att", "Def_duels_won", "Def_duels_win_pct",
            "Off_duels_att", "Off_duels_won", "Off_duels_win_pct",
            "Aerial_duels_att", "Aerial_duels_won", "Aerial_duels_win_pct",
            "Loose_ball_duels_att", "Loose_ball_duels_won",
            "Interceptions", "Recoveries",
            "Sliding_tackles_att", "Sliding_tackles_won",
            "Clearances",
            "Fouls",
            "Losses", "Losses_ownhalf",
        ],
    }
    ALWAYS_COMPUTED = {"passes_acc_pct", "Actions_success_pct", "Goal differential", "Assist differential"}

    # Bygg valda KPI (med alias-fallback för xA -> XAssists)
    wanted = KPI_GROUPS.get(category, [])
    selected_kpis, missing_kpis = [], []
    for k in wanted:
        rc = _resolve_col(k)
        if rc is None and k == "xA":
            rc = _resolve_col("XAssists")  # alias-fallback
        if (rc is not None) or (k in ALWAYS_COMPUTED):
            selected_kpis.append(rc if rc is not None else k)
        else:
            missing_kpis.append(k)

    if selected_kpis:
        st.caption("Visar KPI: " + ", ".join(pretty_label(str(x)) for x in selected_kpis))
    else:
        st.warning("Inga KPI:er hittades i datan för denna kategori.")

    if missing_kpis:
        with st.expander("KPI som inte matchade några kolumner just nu"):
            st.write(", ".join(missing_kpis))

    # ===================== Numerik-konvertering =====================
    work_numeric = work.copy()
    cols_to_num = [c for c in selected_kpis if isinstance(c, str) and (c in work_numeric.columns)]
    if minutes_col:
        cols_to_num.append(minutes_col)
    for c in cols_to_num:
        work_numeric[c] = pd.to_numeric(work_numeric[c], errors="coerce")

    # ===================== Spelarkolumn =====================
    player_col = next((c for c in ["Player","player","Spelare","Name","namn"] if c in work_numeric.columns), None)
    if not player_col:
        st.warning("Kunde inte hitta spelarkolumn (Player/Spelare).")
        st.stop()

    grp = work_numeric.groupby(player_col, dropna=False)

    # ===================== Basram (matcher/minuter) =====================
    base = pd.DataFrame({player_col: grp.size().index, "matches": grp.size().values})
    base["Minutes"] = grp[minutes_col].sum(min_count=1).values if (minutes_col and minutes_col in work_numeric.columns) else np.nan

    from pandas.api.types import is_numeric_dtype

    # ===================== Nämnare-funktion (för kvalificering) =====================
    def _denominator_for_kpi(kpi: str) -> str | None:
        kl = kpi.lower()

        # Exakta suffix – klipp korrekt
        if kl.endswith("_acc_pct"):
            base_name = kpi[:-len("_acc_pct")]
            return _first_existing([f"{base_name}_att", f"{base_name}_total"])
        if kl.endswith("_win_pct"):
            base_name = kpi[:-len("_win_pct")]
            return _first_existing([f"{base_name}_att", f"{base_name}_total"])
        if kl.endswith("_succ_pct"):
            base_name = kpi[:-len("_succ_pct")]
            return _first_existing([f"{base_name}_att", f"{base_name}_total"])

        # Pass%
        if ("pass" in kl) and (kl.endswith("_acc_pct") or kl in {"passes_acc_pct", "pass_accuracy", "pass_pct", "pass_percent", "passes_accuracy_pct"}):
            return _first_existing(["Passes_total", "Passes_att", "Passes_attempted", "Total_passes", "Passes"])

        # Actions%
        if kl in {"actions_success_pct", "actions success %"}:
            return _first_existing(["Actions_total"])

        # Crosses%
        if kl in {"crosses_acc_pct", "cross_accuracy", "cross_pct"}:
            return _first_existing(["Crosses_att", "Crosses_total"])

        # Duels / Aerials / Dribbles / Long passes
        if kl in {"duels_win_pct", "duels_pct"}:
            return _first_existing(["Duels_total", "Duels_att"])
        if kl in {"aerial_duels_win_pct", "aerials_win_pct"}:
            return _first_existing(["Aerial_duels_att", "Aerials_att", "Aerial_duels_total"])
        if kl in {"dribbles_succ_pct", "dribbles_success_pct"}:
            return _first_existing(["Dribbles_att", "Dribbles_total"])
        if kl in {"long_passes_acc_pct"}:
            return _first_existing(["Long_passes_att", "Long_balls_att", "Long_balls_total"])

        # Progressive passes %
        if kl in {"progressive_passes_acc_pct"}:
            return _first_existing(["Progressive_passes_att", "Progressive_passes_total"])

        return None

    # ===================== KPI-beräkning per spelare =====================
    def kpi_series(kpi_name: str) -> pd.Series:
        """
        Per-spelare serie.
        - Procent-KPI: viktad % = sum(num)/sum(den)*100 (ingen mean av %-kolumn).
        - Pass% & Actions% har egna alias-regler.
        - Goal/Assist differential: sum(Goals/Assists) - sum(xG/xA) (+ per90 om på).
        - Övriga numerics: sum (+ per90 om på).
        """
        name = kpi_name
        low = str(name).lower()

        # minutes per player (beräknas här; undviker yttre 'base' i funktionen)
        mins = None
        if per90_on and minutes_col and (minutes_col in work_numeric.columns):
            mins = grp[minutes_col].sum(min_count=1)

        # --- Pass accuracy % (bara totalpassar, inte alla *_passes_...) ---
        if low in {"passes_acc_pct", "pass_accuracy", "pass_pct", "pass_percent", "passes_accuracy_pct"}:
            num_col = _first_existing([
                "Passes_acc",
                "Passes_accurate",
                "Passes_completed",
                "Successful_passes",
                "Accurate_passes",
            ])
            den_col = _denominator_for_kpi("passes_acc_pct")
            if num_col and den_col:
                work_numeric[num_col] = pd.to_numeric(work_numeric[num_col], errors="coerce")
                work_numeric[den_col] = pd.to_numeric(work_numeric[den_col], errors="coerce")
                num_s = grp[num_col].sum(min_count=1)
                den_s = grp[den_col].sum(min_count=1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    return (num_s / den_s) * 100.0
            return pd.Series(np.nan, index=grp.size().index, dtype=float)

        # --- Actions success % (endast procent-KPI) ---
        if low.endswith("_success_pct") or low in {"actions_success_pct", "actions success %"}:
            num_col = _first_existing(["Actions_successful"])
            den_col = _first_existing(["Actions_total"])
            if num_col and den_col:
                work_numeric[num_col] = pd.to_numeric(work_numeric[num_col], errors="coerce")
                work_numeric[den_col] = pd.to_numeric(work_numeric[den_col], errors="coerce")
                num_s = grp[num_col].sum(min_count=1)
                den_s = grp[den_col].sum(min_count=1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    return (num_s / den_s) * 100.0
            return pd.Series(np.nan, index=grp.size().index, dtype=float)

        # --- Generic procent (alla andra) ---
        is_pct = (low.endswith("_acc_pct") or low.endswith("_win_pct") or low.endswith("_succ_pct") or "%" in low)
        if is_pct:
            den_col = _denominator_for_kpi(name)
            num_col = None
            if low.endswith("_acc_pct"):
                root = name[:-len("_acc_pct")]
                num_col = _first_existing([f"{root}_acc", f"{root}_accurate"])
            elif low.endswith("_win_pct"):
                root = name[:-len("_win_pct")]
                num_col = _first_existing([f"{root}_won"])
            elif low.endswith("_succ_pct"):
                root = name[:-len("_succ_pct")]
                num_col = _first_existing([f"{root}_succ", f"{root}_successful"])

            if num_col and den_col and (num_col in work_numeric.columns) and (den_col in work_numeric.columns):
                work_numeric[num_col] = pd.to_numeric(work_numeric[num_col], errors="coerce")
                work_numeric[den_col] = pd.to_numeric(work_numeric[den_col], errors="coerce")
                num_s = grp[num_col].sum(min_count=1)
                den_s = grp[den_col].sum(min_count=1)
                with np.errstate(divide='ignore', invalid='ignore'):
                    return (num_s / den_s) * 100.0
            return pd.Series(np.nan, index=grp.size().index, dtype=float)

        # --- Goal differential ---
        if low == "goal differential":
            g_col  = _first_existing(["Goals"])
            xg_col = _first_existing(["xG", "XG"])
            if g_col and xg_col:
                work_numeric[g_col]  = pd.to_numeric(work_numeric[g_col],  errors="coerce")
                work_numeric[xg_col] = pd.to_numeric(work_numeric[xg_col], errors="coerce")
                diff = grp[g_col].sum(min_count=1) - grp[xg_col].sum(min_count=1)
                if mins is not None:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        return (diff / mins) * 90.0
                return diff
            return pd.Series(np.nan, index=grp.size().index, dtype=float)

        # --- Assist differential ---
        if low == "assist differential":
            a_col  = _first_existing(["Assists"])
            xa_col = _first_existing(["xA", "XAssists"])  # stöd båda
            if a_col and xa_col:
                work_numeric[a_col]  = pd.to_numeric(work_numeric[a_col],  errors="coerce")
                work_numeric[xa_col] = pd.to_numeric(work_numeric[xa_col], errors="coerce")
                diff = grp[a_col].sum(min_count=1) - grp[xa_col].sum(min_count=1)
                if mins is not None:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        return (diff / mins) * 90.0
                return diff
            return pd.Series(np.nan, index=grp.size().index, dtype=float)
            
        # --- Average pass length (viktat snitt, inte summa) ---
        if low in {"avg_pass_length_m", "avg pass length", "average_pass_length"}:
            len_col = _first_existing(["Avg_pass_length_m"])
            wt_col  = _first_existing(["Passes_total", "Passes_att", "Passes"])
            if len_col and wt_col:
                work_numeric[len_col] = pd.to_numeric(work_numeric[len_col], errors="coerce")
                work_numeric[wt_col]  = pd.to_numeric(work_numeric[wt_col],  errors="coerce")

                # produkt = längd * antal pass → summera produkt och antal
                work_numeric["_prod_avg_len"] = work_numeric[len_col] * work_numeric[wt_col]
                num_s = grp["_prod_avg_len"].sum(min_count=1)
                den_s = grp[wt_col].sum(min_count=1)

                with np.errstate(divide='ignore', invalid='ignore'):
                    res = num_s / den_s

                # städa temp-kolumn
                work_numeric.drop(columns=["_prod_avg_len"], inplace=True, errors="ignore")
                return res

            # om något saknas → bara NaN
            return pd.Series(np.nan, index=grp.size().index, dtype=float)

        # --- Vanliga numeriska KPI (inkl. Actions_successful) ---
        if (name in work_numeric.columns) and is_numeric_dtype(work_numeric[name]):
            s = grp[name].sum(min_count=1)
            if mins is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    return (s / mins) * 90.0
            return s

        return pd.Series(dtype=float)

    # Bygg wide-tabell
    wide = base.copy()
    for k in selected_kpis:
        ser = kpi_series(k)
        if not ser.empty:
            wide = wide.merge(ser.rename(k), left_on=player_col, right_index=True, how="left")

    # ===================== Kvalificering =====================
    qual_mask = pd.Series(True, index=wide.index)

    if minutes_col and "Minutes" in wide.columns:
        qual_mask &= (wide["Minutes"].fillna(0) >= min_minutes)

    if min_attempts > 0:
        denom_cols = {}
        for k in selected_kpis:
            k_str = k if isinstance(k, str) else str(k)
            den = _denominator_for_kpi(k_str)
            if den and den in work_numeric.columns:
                denom_cols[den] = grp[den].sum(min_count=1)
        if denom_cols:
            den_df = pd.concat(denom_cols, axis=1)
            den_df.index.name = player_col
            wide = wide.merge(den_df.reset_index(), on=player_col, how="left")
            atleast_one_ok = np.zeros(len(wide), dtype=bool)
            for dcol in denom_cols.keys():
                atleast_one_ok |= (wide.get(dcol, 0).fillna(0).values >= min_attempts)
            qual_mask &= atleast_one_ok

    # ===================== Render-funktion =====================
    def render_leaderboard(kpi_name: str):
        dfk = wide[[player_col, "Minutes", kpi_name]].copy()

        den = _denominator_for_kpi(kpi_name if isinstance(kpi_name, str) else str(kpi_name))
        if den and (den in wide.columns):
            dfk["_den"] = wide[den]
            kpi_mask = (dfk["_den"].fillna(0) >= min_attempts)
        else:
            kpi_mask = pd.Series(True, index=dfk.index)

        mask = qual_mask & kpi_mask
        dfk["_qual"] = mask

        qdf = dfk.copy() if show_nonqual else dfk[dfk["_qual"]].copy()

        label = pretty_label(str(kpi_name))
        st.subheader(label)

        if qdf.empty:
            st.info("Inga rader att visa för denna KPI med nuvarande filter.")
            return

        qdf = qdf.sort_values(by=[kpi_name, "Minutes", player_col], ascending=[False, False, True])
        qdf["Rank"] = qdf[kpi_name].rank(method="dense", ascending=False).astype("Int64")

        disp = qdf[["Rank", player_col, "Minutes", kpi_name]].reset_index(drop=True)
        disp = disp.rename(columns={kpi_name: label})

        # ---- 1 decimal på alla floats, 0 på Minutes + centrera allt ----
        fmt = {}
        for c in disp.columns:
            if pd.api.types.is_float_dtype(disp[c]):
                fmt[c] = "{:.0f}" if c == "Minutes" else "{:.1f}"

        vals = pd.to_numeric(disp[label], errors="coerce")
        pctl = vals.rank(pct=True, method="average", ascending=True).fillna(0.0)

        def _row_bg(val):
            if pd.isna(val): return ""
            if val <= 0.5:
                t = val / 0.5
                r, g, b = 255, int(255*t), 0
            else:
                t = (val - 0.5) / 0.5
                r, g, b = int(255*(1-t)), 255, 0
            return f"background-color: rgb({r},{g},{b}); color: black;"

        def row_style(row):
            return [_row_bg(pctl.iloc[row.name])] * len(row)

        styler = (
            disp.style
            .apply(row_style, axis=1)
            .format(fmt)
            .set_table_styles([
                {'selector': 'th', 'props': 'text-align: center; vertical-align: middle;'},
                {'selector': 'td', 'props': 'text-align: center; vertical-align: middle;'},
            ])
            .set_properties(**{'text-align': 'center', 'vertical-align': 'middle'})
            .set_table_styles([
                {"selector":"tbody td","props":[("text-align","center"),("vertical-align","middle")]},
                {"selector":"thead th","props":[("text-align","center"),("vertical-align","middle")]},
            ], overwrite=False)
        )


        st.dataframe(styler, use_container_width=True, hide_index=True)

    # ===================== Grid-rendering =====================
    if selected_kpis:
        for i in range(0, len(selected_kpis), 3):
            rcols = st.columns(min(3, len(selected_kpis) - i))
            for j, col in enumerate(rcols):
                with col:
                    render_leaderboard(selected_kpis[i + j])

    # ===================== Export (inkl. rank) =====================
    export_wide = wide[[player_col, "Minutes"] + selected_kpis].copy()
    for k in selected_kpis:
        k_str = k if isinstance(k, str) else str(k)
        den = _denominator_for_kpi(k_str)
        if den and (den in wide.columns):
            kpi_mask = (wide[den].fillna(0) >= min_attempts)
        else:
            kpi_mask = pd.Series(True, index=wide.index)
        mask = qual_mask & kpi_mask
        ranks = export_wide.loc[mask, k].rank(method="dense", ascending=False).astype("Int64")
        export_wide[f"{k}__rank"] = pd.Series(index=export_wide.index, dtype="Int64")
        export_wide.loc[mask, f"{k}__rank"] = ranks

    st.download_button(
        "Ladda ned kategori som CSV",
        data=export_wide.to_csv(index=False).encode("utf-8"),
        file_name=f"leaderboards_{category.lower()}.csv",
        mime="text/csv",
    )

    st.caption("Procent = sum(numerator)/sum(denominator)*100. Minutes visas utan decimal, övriga tal med 1 decimal. XAssists visas som xA.")
