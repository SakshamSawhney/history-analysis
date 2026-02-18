import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io
import re
import scipy.stats as stats
import datetime
import ast

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="EUR Rates Analysis (Pro)")

# --- HELPER FUNCTIONS ---
def calculate_var(series, confidence=0.95):
    """Calculates Historical Value at Risk."""
    if series.empty: return 0, 0
    returns = series.pct_change().dropna()
    if returns.empty: return 0, 0
    var = np.percentile(returns, (1 - confidence) * 100)
    cvar = returns[returns < var].mean() if not returns[returns < var].empty else var
    return var, cvar

def get_contract_num(col_name):
    match = re.search(r'(\d+)', str(col_name))
    return int(match.group(1)) if match else 0

def _validate_expr_ast(node):
    if isinstance(node, ast.Expression):
        _validate_expr_ast(node.body)
        return

    if isinstance(node, ast.Lambda):
        _validate_expr_ast(node.body)
        return

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)):
            raise ValueError("Only basic arithmetic operators are allowed")
        _validate_expr_ast(node.left)
        _validate_expr_ast(node.right)
        return

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError("Only unary + and - are allowed")
        _validate_expr_ast(node.operand)
        return

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Attribute):
            raise ValueError("Only np.<func>(...) calls are allowed")
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "np":
            raise ValueError("Only np.<func>(...) calls are allowed")
        allowed_funcs = {"abs", "log", "exp", "sqrt", "clip", "where", "maximum", "minimum"}
        if node.func.attr not in allowed_funcs:
            raise ValueError(f"np.{node.func.attr} is not allowed")
        for arg in node.args:
            _validate_expr_ast(arg)
        for kw in node.keywords:
            _validate_expr_ast(kw.value)
        return

    if isinstance(node, ast.Attribute):
        if not isinstance(node.value, ast.Name) or node.value.id != "np":
            raise ValueError("Only np.<const> attributes are allowed")
        return

    if isinstance(node, ast.Name):
        if node.id not in {"x", "np"}:
            raise ValueError("Only 'x' and 'np' are allowed")
        return

    if isinstance(node, ast.Constant):
        return

    raise ValueError("Expression contains unsupported syntax")


def apply_expression(values, expr):
    if expr is None:
        return values
    expr = expr.strip()
    if not expr:
        return values

    values = np.asarray(values, dtype=float)
    try:
        tree = ast.parse(expr, mode="eval")
        body = tree.body
        if isinstance(body, ast.Lambda):
            body = body.body
        _validate_expr_ast(body)
        code = compile(ast.Expression(body), "<expr>", "eval")
        result = eval(code, {"np": np, "__builtins__": {}}, {"x": values})
        return np.asarray(result, dtype=float)
    except Exception as exc:
        st.warning(f"Expression error: {exc}")
        return values


def apply_transforms(values, y_offset=0, y_scale=1.0, normalize_to=None, expr=None):
    """
    Apply transformations to y-values.
    
    Args:
        values: array-like y-values
        y_offset: constant to add to all values
        y_scale: factor to multiply all values by
        normalize_to: tuple (min, max) to normalize values to this range
    
    Returns:
        Transformed values
    """
    values = np.asarray(values, dtype=float)
    
    # Apply custom expression first
    transformed = apply_expression(values, expr)

    # Apply offset and scale
    transformed = (transformed * y_scale) + y_offset
    
    # Normalize to range if specified
    if normalize_to is not None:
        target_min, target_max = normalize_to
        current_min = np.nanmin(transformed)
        current_max = np.nanmax(transformed)
        current_range = current_max - current_min
        
        if current_range > 0:
            # Scale to [0, 1] first
            normalized = (transformed - current_min) / current_range
            # Scale to target range
            transformed = normalized * (target_max - target_min) + target_min
    
    return transformed

# --- INITIALIZE ECB CALENDAR DATA ---
def get_default_ecb_data():
    data = [
        {'Date': '2026-02-05', 'Action': 'Hold', 'Change': '0.00%', 'Rate': '2.00%'},
        {'Date': '2025-12-18', 'Action': 'Hold', 'Change': '0.00%', 'Rate': '2.00%'},
        {'Date': '2025-10-30', 'Action': 'Hold', 'Change': '0.00%', 'Rate': '2.00%'},
        {'Date': '2025-09-11', 'Action': 'Hold', 'Change': '0.00%', 'Rate': '2.00%'},
        {'Date': '2025-07-24', 'Action': 'Hold', 'Change': '0.00%', 'Rate': '2.00%'},
        {'Date': '2025-06-05', 'Action': 'Cut', 'Change': '-0.25%', 'Rate': '2.00%'},
        {'Date': '2025-04-17', 'Action': 'Cut', 'Change': '-0.25%', 'Rate': '2.25%'},
        {'Date': '2025-03-06', 'Action': 'Cut', 'Change': '-0.25%', 'Rate': '2.50%'},
        {'Date': '2025-01-30', 'Action': 'Cut', 'Change': '-0.25%', 'Rate': '2.75%'},
        {'Date': '2024-12-12', 'Action': 'Cut', 'Change': '-0.25%', 'Rate': '3.00%'},
        {'Date': '2024-10-17', 'Action': 'Cut', 'Change': '-0.25%', 'Rate': '3.25%'},
        {'Date': '2024-09-12', 'Action': 'Cut', 'Change': '-0.25%', 'Rate': '3.50%'},
        {'Date': '2024-06-06', 'Action': 'Cut', 'Change': '-0.25%', 'Rate': '3.75%'},
        {'Date': '2023-09-14', 'Action': 'Hike', 'Change': '+0.25%', 'Rate': '4.00%'},
        {'Date': '2023-07-27', 'Action': 'Hike', 'Change': '+0.25%', 'Rate': '3.75%'},
        {'Date': '2023-06-15', 'Action': 'Hike', 'Change': '+0.25%', 'Rate': '3.50%'},
        {'Date': '2023-05-04', 'Action': 'Hike', 'Change': '+0.25%', 'Rate': '3.25%'},
        {'Date': '2023-03-16', 'Action': 'Hike', 'Change': '+0.50%', 'Rate': '3.00%'},
        {'Date': '2023-02-02', 'Action': 'Hike', 'Change': '+0.50%', 'Rate': '2.50%'},
        {'Date': '2022-12-15', 'Action': 'Hike', 'Change': '+0.50%', 'Rate': '2.00%'},
        {'Date': '2022-10-27', 'Action': 'Hike', 'Change': '+0.75%', 'Rate': '1.50%'},
        {'Date': '2022-09-08', 'Action': 'Hike', 'Change': '+0.75%', 'Rate': '0.75%'},
        {'Date': '2022-07-21', 'Action': 'Hike', 'Change': '+0.50%', 'Rate': '0.00%'},
    ]
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# --- FILE LOADING LOGIC ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
FILE_NAME = "LOIS 12th Feb.xlsx"
FILE_PATH = os.path.join(SCRIPT_DIR, FILE_NAME)

@st.cache_data
def load_sheet(file_path, sheet_name, backup_data=None):
    df = None
    source = "File"
    
    if os.path.exists(file_path):
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            if df.empty: df = None
        except Exception: df = None
    
    if df is None and backup_data is not None:
        try:
            lines = [line.strip() for line in backup_data.strip().split('\n') if line.strip() and not line.strip().startswith('[')]
            cleaned_data = "\n".join(lines)
            df = pd.read_csv(io.StringIO(cleaned_data), sep='|', skipinitialspace=True)
            source = "Backup Data"
        except Exception as e:
            st.error(f"Error parsing backup data: {e}")
            return None, "Error"

    if df is None: return None, "Not Found"

    # --- CLEANING LOGIC ---
    df = df.dropna(how='all').dropna(axis=1, how='all')
    df.columns = [str(c).strip().replace('*', '') for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Find Date Column
    date_col = None
    for col in df.columns:
        if 'date' in col.lower(): date_col = col; break
            
    if date_col:
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        if date_col in df.columns and date_col != 'Date': df = df.drop(columns=[date_col])
    else:
        if len(df.columns) > 0:
             try:
                 df['Date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                 df = df.drop(columns=[df.columns[0]])
             except: pass

    if 'Date' not in df.columns: return None, "Error"

    # --- SMART COLUMN RENAMING ---
    rename_map = {}
    spot_col = None
    for col in df.columns:
        if col == 'Date': continue
        col_lower = col.lower()
        num = get_contract_num(col)
        
        if 'open interest' in col_lower: rename_map[col] = f'OI_{num}' if num > 0 else 'OI'
        elif 'smavg' in col_lower or 'moving average' in col_lower: rename_map[col] = f'MA_{num}' if num > 0 else 'MA'
        elif 'spot' in col_lower or 'eur003m' in col_lower or 'eeswec' in col_lower:
            if spot_col is None: rename_map[col] = 'Spot'; spot_col = col
        else:
            if num > 0: rename_map[col] = f'F{num}'
            else: rename_map[col] = col
            
    df = df.rename(columns=rename_map)
    
    # Clean Numeric Data
    cols_to_numeric = [c for c in df.columns if c != 'Date']
    for col in cols_to_numeric:
        if df[col].dtype == object: df[col] = df[col].astype(str).str.replace('*', '', regex=False).str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.dropna(subset=['Date'])
    df = df.set_index('Date').sort_index(ascending=True)

    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty: df = df.dropna(subset=numeric_cols, how='all')
    
    return df, source

# --- HARDCODED BACKUP DATA ---
LOIS_BACKUP_DATA = """HEAD | Date | EEFOSC1 BGN Curncy (L1) | EEFOSC2 BGN Curncy (L1) | EEFOSC3 BGN Curncy (L1) | EEFOSC4 BGN Curncy (L1) | EEFOSC5 BGN Curncy (L1) | EEFOSC6 BGN Curncy (L1) | EEFOSC7 BGN Curncy (L1) | EEFOSC8 BGN Curncy (L1) | EUR003M Index - EESWEC Curncy (R1) | 
0 | 2026/2/12 | 11.519 | 12.3 | 12.8 | 13.219 | 13.6 | 13.847 | 13.858 | 13.696 | 0.048 | 
1 | 2026/2/11 | 11.55 | 12.328 | 12.817 | 13.203 | 13.661 | 13.892 | 13.877 | 13.797 | 0.0476 | 
2 | 2026/2/10 | 11.1 | 12.141 | 12.783 | 13.192 | 13.653 | 13.932 | 13.899 | 13.885 | 0.047 | 
3 | 2026/2/9 | 11.325 | 12.375 | 13 | 13.5 | 14 | 14.108 | 14.074 | 13.903 | 0.0474 | 
4 | 2026/2/6 | 11.6 | 12.55 | 12.875 | 13.567 | 14 | 14.125 | 14.136 | 14.086 | 0.0659 | 
5 | 2026/2/5 | 12.09 | 13.068 | 13.4 | 13.815 | 14.207 | 14.375 | 14.352 | 14.12 | 0.0865 | 
6 | 2026/2/4 | 12.3 | 13.0565 | 13.4 | 13.826 | 14.294 | 14.424 | 14.392 | 14.326 | 0.107 | 
7 | 2026/2/3 | 12.15 | 13 | 13.444 | 13.798 | 14.084 | 14.397 | 14.357 | 14.38 | 0.0968 | 
8 | 2026/2/2 | 12.125 | 12.948 | 13.4 | 13.7 | 14.078 | 14.159 | 14.167 | 14.16 | 0.0913 | 
9 | 2026/1/30 | 12.375 | 13.05 | 13.562 | 13.835 | 14.086 | 14.355 | 14.347 | 14.312 | 0.1018 | 
10 | 2026/1/29 | 12.3 | 12.65 | 13.317 | 13.599 | 13.885 | 14.053 | 13.93 | 13.941 | 0.0899 | 
11 | 2026/1/28 | 12.1 | 12.6 | 13.057 | 13.375 | 13.6 | 13.787 | 13.804 | 13.868 | 0.0938 | 
12 | 2026/1/27 | 12.579 | 13.044 | 13.172 | 13.445 | 13.683 | 13.875 | 13.893 | 13.852 | 0.1075 | 
13 | 2026/1/26 | * | 13.2 | 13.4 | 13.5925 | 13.879 | 13.844 | 13.891 | 13.892 | 0.1056 | 
"""

# --- MAIN APP LOGIC ---
st.title("📊 EUR Rates Analysis & Hedging Engine")

# Initialize ECB Calendar in Session State
if 'ecb_calendar' not in st.session_state:
    st.session_state['ecb_calendar'] = get_default_ecb_data()

# ---------------------------------------------------------
# SIDEBAR: GLOBAL TOGGLES (Early Setup)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔄 Data Conversion (ER)")
    er_subtract_toggle = st.checkbox("100 - value", value=False, key="er_subtract")
    
    er_multiply_toggle = False
    multiplier_value = 1
    if er_subtract_toggle:
        er_multiply_toggle = st.checkbox("Multiply by", value=False, key="er_multiply")
        if er_multiply_toggle:
            multiplier_value = st.number_input("Multiplier", value=100, step=1, key="multiplier_input")
        st.caption("ER values will be converted for LOIS comparison")
    
    st.markdown("### 📅 Event Overlay")
    show_events_on_charts = st.checkbox("Show ECB Dates on Charts", value=False, key="show_events_global")
    if show_events_on_charts:
        st.caption("Red=Hike, Green=Cut, Grey=Hold")

# Load Market Data
df_lois, source_lois = load_sheet(FILE_PATH, "LOIS", backup_data=LOIS_BACKUP_DATA)
df_er, source_er = load_sheet(FILE_PATH, "ER", backup_data=None)

# Apply ER conversion if enabled
if er_subtract_toggle and df_er is not None:
    # Convert ER values: (100 - value) for price columns
    numeric_cols_er = df_er.select_dtypes(include=[np.number]).columns
    # Exclude OI and MA columns from conversion (only convert prices like F1, F2, Spot)
    cols_to_convert = [c for c in numeric_cols_er if not any(skip in c for skip in ['OI', 'MA'])]
    
    if er_multiply_toggle:
        # Apply (100 - value) * multiplier
        df_er[cols_to_convert] = df_er[cols_to_convert].apply(lambda x: (100 - x) * multiplier_value)
    else:
        # Apply only (100 - value)
        df_er[cols_to_convert] = df_er[cols_to_convert].apply(lambda x: 100 - x)

# Add prefixes
if df_lois is not None: df_lois = df_lois.add_prefix("LOIS_")
if df_er is not None: df_er = df_er.add_prefix("ER_")

# Create Master DataFrame (after ER conversion is applied)
if df_lois is not None and df_er is not None:
    df_master = pd.concat([df_lois, df_er], axis=1)
elif df_lois is not None:
    df_master = df_lois.copy()
elif df_er is not None:
    df_master = df_er.copy()
else:
    st.error("No data found.")
    st.stop()

# --- GET COLS FUNCTION ---
def get_cols(df, prefix=None):
    cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
    if prefix: cols = [c for c in cols if c.startswith(prefix)]
    return sorted(cols, key=lambda x: get_contract_num(x))

available_cols = get_cols(df_master)

# =====================================================================
# FLEXIBLE INSTRUMENT BUILDER FUNCTION FOR ALL TABS
# =====================================================================
def build_series_from_selection(df_master, dataset, inst_type, key_base, include_spot=True):
    """Build a time series from any combination of dataset/type/legs."""
    
    if inst_type == "Outright":
        cols = get_cols(df_master, dataset + "_")
        cols = [c for c in cols if "_F" in c or "Spot" in c]
        if not include_spot:
            cols = [c for c in cols if "Spot" not in c]
        l1 = st.selectbox("Instrument", cols, key=f"{key_base}_l1")
        series = df_master[l1].dropna()
        title = f"{dataset} {inst_type}: {l1}"
        return series, title
    
    elif inst_type == "Spread":
        cols = get_cols(df_master, dataset + "_")
        cols = [c for c in cols if "_F" in c]
        c1, c2 = st.columns(2)
        with c1:
            l1 = st.selectbox("Leg 1", cols, key=f"{key_base}_l1")
        with c2:
            l2 = st.selectbox("Leg 2", cols, index=min(1, len(cols)-1), key=f"{key_base}_l2")
        series = (df_master[l1] - df_master[l2]).dropna()
        title = f"{dataset} {inst_type}: {l1}-{l2}"
        return series, title
    
    else:  # Fly
        cols = get_cols(df_master, dataset + "_")
        cols = [c for c in cols if "_F" in c]
        c1, c2, c3 = st.columns(3)
        with c1:
            l1 = st.selectbox("Leg 1", cols, key=f"{key_base}_l1")
        with c2:
            l2 = st.selectbox("Leg 2", cols, index=min(1, len(cols)-1), key=f"{key_base}_l2")
        with c3:
            l3 = st.selectbox("Leg 3", cols, index=min(2, len(cols)-1), key=f"{key_base}_l3")
        series = ((df_master[l1] + df_master[l3]) - (2 * df_master[l2])).dropna()
        title = f"{dataset} {inst_type}: {l1}-{l2}-{l3}"
        return series, title


def align_series_to_common_index(series_list):
    if not series_list:
        return series_list, None

    common_index = series_list[0].dropna().index
    for s in series_list[1:]:
        common_index = common_index.intersection(s.dropna().index)

    if common_index.empty:
        return series_list, None

    return [s.loc[common_index] for s in series_list], common_index


def get_dataset_available_dates(df_master, dataset):
    cols = [c for c in df_master.columns if c.startswith(dataset + "_")]
    if not cols:
        return df_master.index
    return df_master[cols].dropna(how="all").index


def get_instrument_legs(inst_name):
    return inst_name.split("-")


def shared_legs_only_if_exact_match(inst_a, inst_b):
    legs_a = get_instrument_legs(inst_a)
    legs_b = get_instrument_legs(inst_b)
    shared = set(legs_a) & set(legs_b)
    if not shared:
        return True
    return legs_a == legs_b

# Create Tabs
tab_zscore, tab_corr, tab_regress, tab_range, tab_risk, tab_calendar = st.tabs([
    "📈 Z-Score & Technical",
    "🔗 Correlation Analysis",
    "📊 Regression & Hedging",
    "📉 Range & Curves",
    "⚖️ Risk Management",
    "📅 Calendar"
])

# =====================================================================
# TAB 1: Z-SCORE & TECHNICAL CHARTS
# =====================================================================
with tab_zscore:
    st.header("📈 Z-Score & Technical Analysis")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("1. Select Instrument")
        compare_mode = st.checkbox("Enable Comparison Mode", value=False, key="zscore_compare")
    
    with col_right:
        st.subheader("2. Parameters")
        z_window = st.number_input("Window (Days)", 5, 252, 20, key="zscore_window")
        entry_z = st.slider("Entry Threshold (Z)", 1.0, 4.0, 2.0, 0.1, key="zscore_entry")
        atr_window = st.number_input("ATR Window (Days)", 5, 120, 14, key="zscore_atr")
    
    series_list = []
    titles_list = []
    datasets_list = []
    
    if compare_mode:
        num_compare = st.number_input("Number of instruments to compare", 2, 10, 2, 1, key="zscore_num_compare")
        for i in range(int(num_compare)):
            with st.expander(f"Instrument {i+1}", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    ds = st.selectbox("Dataset", ["LOIS", "ER"], key=f"zscore_comp_ds_{i}")
                with c2:
                    it = st.selectbox("Type", ["Outright", "Spread", "Fly"], key=f"zscore_comp_type_{i}")
                
                ser, tit = build_series_from_selection(df_master, ds, it, f"zscore_comp_{i}")
                series_list.append(ser)
                titles_list.append(tit)
                datasets_list.append(ds)
    else:
        c1, c2 = st.columns(2)
        with c1:
            dataset = st.selectbox("Dataset", ["LOIS", "ER"], key="zscore_ds")
        with c2:
            inst_type = st.selectbox("Type", ["Outright", "Spread", "Fly"], key="zscore_type")
        
        series, title = build_series_from_selection(df_master, dataset, inst_type, "zscore")
        series_list.append(series)
        titles_list.append(title)
        datasets_list.append(dataset)

    # Align to common timeframe when comparing LOIS and ER together
    use_common_timeframe = False
    if compare_mode and len(set(datasets_list)) > 1:
        use_common_timeframe = st.checkbox(
            "Use only common timeframe values (LOIS vs ER)",
            value=True,
            key="zscore_common_timeframe"
        )
        if use_common_timeframe:
            series_list, common_index = align_series_to_common_index(series_list)
            if common_index is None:
                st.warning("No overlapping dates between selected series.")
    
    # Calculations for primary series
    series = series_list[0]
    title = titles_list[0]
    rolling_mean = series.rolling(z_window).mean()
    rolling_std = series.rolling(z_window).std()
    z_score = (series - rolling_mean) / rolling_std
    daily_vol = series.diff().abs()
    atr_val = daily_vol.rolling(atr_window).mean().iloc[-1]
    
    # Calculate Z-scores for comparison series if comparison mode is enabled
    z_score_list = [z_score]
    if compare_mode:
        for s in series_list[1:]:
            s_mean = s.rolling(z_window).mean()
            s_std = s.rolling(z_window).std()
            s_z = (s - s_mean) / s_std
            z_score_list.append(s_z)
    
    # Transform Controls for Price Series
    with st.expander("📐 Price Transform Controls (optional)", expanded=True):
        z_expr = st.text_input("Expression (use x)", value="", key="zscore_expr", help="Example: x*100-2 or np.log(x)")
        col_zt1, col_zt2, col_zt3 = st.columns(3)
        with col_zt1:
            z_y_offset = st.number_input("Y offset", value=0.0, step=0.1, format="%.4f", key="zscore_y_offset")
        with col_zt2:
            z_y_scale = st.number_input("Y scale", value=1.0, step=0.1, format="%.2f", key="zscore_y_scale")
        with col_zt3:
            z_normalize = st.checkbox("Normalize", key="zscore_normalize")
        
        if z_normalize:
            col_zn1, col_zn2 = st.columns(2)
            with col_zn1:
                z_y_min = st.number_input("Min", value=0.0, step=1.0, key="zscore_y_min")
            with col_zn2:
                z_y_max = st.number_input("Max", value=100.0, step=1.0, key="zscore_y_max")
        else:
            z_y_min, z_y_max = None, None
    
    # Apply transforms to all price series
    series_transformed = pd.Series(
        apply_transforms(
            series.values,
            y_offset=z_y_offset,
            y_scale=z_y_scale,
            normalize_to=(z_y_min, z_y_max) if z_normalize else None,
            expr=z_expr
        ),
        index=series.index
    )
    
    rolling_mean_transformed = pd.Series(
        apply_transforms(
            rolling_mean.values,
            y_offset=z_y_offset,
            y_scale=z_y_scale,
            normalize_to=(z_y_min, z_y_max) if z_normalize else None,
            expr=z_expr
        ),
        index=rolling_mean.index
    )
    
    rolling_std_transformed = rolling_std * z_y_scale  # Scale std dev
    
    series_list_transformed = []
    if compare_mode:
        for s in series_list:
            s_trans = pd.Series(
                apply_transforms(
                    s.values,
                    y_offset=z_y_offset,
                    y_scale=z_y_scale,
                    normalize_to=(z_y_min, z_y_max) if z_normalize else None,
                    expr=z_expr
                ),
                index=s.index
            )
            series_list_transformed.append(s_trans)
    
    # Chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        subplot_titles=("Price & Bands", "Z-Score"))
    
    # Add primary series
    fig.add_trace(go.Scatter(x=series_transformed.index, y=series_transformed, name=f"{title} (Price)", line=dict(color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling_mean_transformed.index, y=rolling_mean_transformed, name=f"{title} (Mean)", line=dict(color='blue', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=series_transformed.index, y=rolling_mean_transformed + rolling_std_transformed*entry_z, name="Upper Band", line=dict(color='red', dash='dash'), opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=series_transformed.index, y=rolling_mean_transformed - rolling_std_transformed*entry_z, name="Lower Band", fill='tonexty', fillcolor='rgba(255,0,0,0.1)', line=dict(color='green', dash='dash'), opacity=0.5), row=1, col=1)
    
    # Add comparison series prices if in comparison mode
    if compare_mode:
        colors = ['orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
        for i, s in enumerate(series_list_transformed[1:]):
            fig.add_trace(go.Scatter(x=s.index, y=s, name=f"{titles_list[i+1]} (Price)", line=dict(color=colors[i % len(colors)]), opacity=0.7), row=1, col=1)
    
    # Z-Score main
    fig.add_trace(go.Scatter(x=z_score.index, y=z_score, name=f"{title} (Z-Score)", line=dict(color='purple')), row=2, col=1)
    signals = z_score[((z_score > entry_z) | (z_score < -entry_z))]
    fig.add_trace(go.Scatter(x=signals.index, y=signals, mode='markers', name='Signal', marker=dict(color='red', size=8)), row=2, col=1)
    
    # Add comparison Z-scores
    if compare_mode:
        colors = ['orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
        for i, z_comp in enumerate(z_score_list[1:]):
            fig.add_trace(go.Scatter(x=z_comp.index, y=z_comp, name=f"{titles_list[i+1]} (Z-Score)", line=dict(color=colors[i % len(colors)]), opacity=0.7), row=2, col=1)
    
    fig.add_hline(y=entry_z, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-entry_z, line_dash="dash", line_color="green", row=2, col=1)
    
    # Add ECB events if enabled
    if show_events_on_charts:
        for _, row in st.session_state['ecb_calendar'].iterrows():
            color = 'red' if row['Action'] == 'Hike' else ('green' if row['Action'] == 'Cut' else 'grey')
            fig.add_vline(x=row['Date'], line=dict(color=color, width=1, dash="dot"), row=1, col=1)
            fig.add_vline(x=row['Date'], line=dict(color=color, width=1, dash="dot"), row=2, col=1)
    
    fig.update_layout(height=1000, template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Current Price", f"{series.iloc[-1]:.4f}")
    col_m2.metric("Current Z-Score", f"{z_score.iloc[-1]:.2f}")
    col_m3.metric("ATR (Volatility)", f"{atr_val:.4f}")

# =====================================================================
# TAB 2: CORRELATION ANALYSIS
# =====================================================================
with tab_corr:
    st.header("🔗 Correlation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Instrument 1")
        ds1 = st.selectbox("Dataset", ["LOIS", "ER"], key="corr_ds1")
        type1 = st.selectbox("Type", ["Outright", "Spread", "Fly"], key="corr_type1")
        series1, title1 = build_series_from_selection(df_master, ds1, type1, "corr1")
    
    with col2:
        st.subheader("Instrument 2")
        ds2 = st.selectbox("Dataset", ["LOIS", "ER"], key="corr_ds2", index=1)
        type2 = st.selectbox("Type", ["Outright", "Spread", "Fly"], key="corr_type2", index=1)
        series2, title2 = build_series_from_selection(df_master, ds2, type2, "corr2")
    
    # Align and calculate correlation
    if ds1 != ds2:
        use_common_timeframe = st.checkbox(
            "Use only common timeframe values (LOIS vs ER)",
            value=True,
            key="corr_common_timeframe"
        )
    else:
        use_common_timeframe = False

    if use_common_timeframe:
        aligned_series, common_index = align_series_to_common_index([series1, series2])
        if common_index is None:
            aligned_df = pd.DataFrame()
        else:
            series1 = aligned_series[0]
            series2 = aligned_series[1]
            aligned_df = pd.concat([series1, series2], axis=1).dropna()
    else:
        aligned_df = pd.concat([series1, series2], axis=1).dropna()
    if len(aligned_df) > 1:
        corr = aligned_df.corr().iloc[0, 1]
        
        # Rolling correlation
        max_corr_window = max(10, len(aligned_df))
        rolling_corr_window = st.number_input(
            "Rolling Correlation Window",
            min_value=10,
            max_value=max_corr_window,
            value=min(30, max_corr_window),
            key="corr_window"
        )
        rolling_corr = aligned_df.iloc[:, 0].rolling(rolling_corr_window).corr(aligned_df.iloc[:, 1])
        
        # Metrics
        col_c1, col_c2 = st.columns(2)
        col_c1.metric("Correlation (Overall)", f"{corr:.4f}")
        col_c2.metric("Sample Size", len(aligned_df))
        
        # Detect if comparing LOIS vs ER
        is_lois_vs_er = (ds1 != ds2)
        
        # Transform Controls
        with st.expander("📐 Transform Controls (optional)", expanded=True):
            if is_lois_vs_er:
                # Enhanced controls for LOIS vs ER comparison
                st.info("🔄 LOIS vs ER comparison: Configure separate transformations and ranges for each dataset")
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown(f"**{title1}**")
                expr1 = st.text_input("Expression (use x)", value="", key="corr_expr1", help="Example: x*100-2 or np.log(x)")
                y1_offset = st.number_input("Add (+/-)", value=0.0, step=0.1, format="%.4f", key="corr_y1_offset", help="Constant to add/subtract from values")
                y1_scale = st.number_input("Multiply (×)", value=1.0, step=0.1, format="%.2f", key="corr_y1_scale", help="Factor to multiply values by")
                norm1 = st.checkbox("Normalize to range", value=is_lois_vs_er, key="corr_norm1")
                if norm1:
                    y1_min = st.number_input("Min", value=0.0, step=1.0, key="corr_y1_min")
                    y1_max = st.number_input("Max", value=100.0, step=1.0, key="corr_y1_max")
                else:
                    y1_min, y1_max = None, None
            
            with col_t2:
                st.markdown(f"**{title2}**")
                expr2 = st.text_input("Expression (use x)", value="", key="corr_expr2", help="Example: x*100-2 or np.log(x)")
                y2_offset = st.number_input("Add (+/-)", value=0.0, step=0.1, format="%.4f", key="corr_y2_offset", help="Constant to add/subtract from values")
                y2_scale = st.number_input("Multiply (×)", value=1.0, step=0.1, format="%.2f", key="corr_y2_scale", help="Factor to multiply values by")
                norm2 = st.checkbox("Normalize to range", value=is_lois_vs_er, key="corr_norm2")
                if norm2:
                    y2_min = st.number_input("Min", value=0.0, step=1.0, key="corr_y2_min")
                    y2_max = st.number_input("Max", value=100.0, step=1.0, key="corr_y2_max")
                else:
                    y2_min, y2_max = None, None
        
        # Apply transforms
        series1_transformed = pd.Series(
            apply_transforms(
                series1.values,
                y_offset=y1_offset,
                y_scale=y1_scale,
                normalize_to=(y1_min, y1_max) if norm1 else None,
                expr=expr1
            ),
            index=series1.index
        )
        
        series2_transformed = pd.Series(
            apply_transforms(
                series2.values,
                y_offset=y2_offset,
                y_scale=y2_scale,
                normalize_to=(y2_min, y2_max) if norm2 else None,
                expr=expr2
            ),
            index=series2.index
        )
        
        # Chart 1: Time Series with dual y-axes
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=series1_transformed.index, y=series1_transformed, name=title1, yaxis='y', line=dict(color='blue')))
        fig1.add_trace(go.Scatter(x=series2_transformed.index, y=series2_transformed, name=title2, yaxis='y2', line=dict(color='red')))
        fig1.update_layout(
            title="Time Series Comparison (Dual Axes)",
            yaxis=dict(
                title=dict(text=f"{title1} (Left)", font=dict(color='blue')),
                side='left',
                tickfont=dict(color='blue')
            ),
            yaxis2=dict(
                title=dict(text=f"{title2} (Right)", font=dict(color='red')),
                overlaying='y',
                side='right',
                tickfont=dict(color='red')
            ),
            height=600, template="plotly_white"
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Chart 2: Scatter
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=aligned_df.iloc[:, 0], y=aligned_df.iloc[:, 1], mode='markers', name='Observations'))
        fig2.update_layout(
            title=f"Scatter: {title1} vs {title2}",
            xaxis_title=title1, yaxis_title=title2,
            height=600, template="plotly_white"
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Chart 3: Rolling Correlation
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, name='Rolling Correlation', line=dict(color='purple')))
        fig3.add_hline(y=0, line_dash="dash", line_color="grey")
        fig3.update_layout(
            title=f"Rolling Correlation ({rolling_corr_window}D)",
            yaxis_title="Correlation", height=600, template="plotly_white"
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Not enough data to calculate correlation.")

# =====================================================================
# TAB 3: REGRESSION & HEDGING
# =====================================================================
with tab_regress:
    st.header("📊 Regression & Hedging")
    
    tab_direct, tab_hedge = st.tabs(["Direct Regression", "Hedging Finder"])
    
    with tab_direct:
        st.subheader("Regression: Any Two Instruments")
        
        reg_window = st.slider("Regression Window (Days)", 30, 252, 90, key="reg_window")
        use_bp_change = st.checkbox("Use bp change", value=True, key="reg_use_bp_change")
        
        col_x, col_y = st.columns(2)
        
        with col_x:
            st.markdown("**X Variable (Independent)**")
            ds_x = st.selectbox("Dataset", ["LOIS", "ER"], key="reg_ds_x")
            type_x = st.selectbox("Type", ["Outright", "Spread", "Fly"], key="reg_type_x")
            series_x, title_x = build_series_from_selection(df_master, ds_x, type_x, "reg_x")
        
        with col_y:
            st.markdown("**Y Variable (Dependent)**")
            ds_y = st.selectbox("Dataset", ["LOIS", "ER"], key="reg_ds_y", index=1)
            type_y = st.selectbox("Type", ["Outright", "Spread", "Fly"], key="reg_type_y", index=1)
            series_y, title_y = build_series_from_selection(df_master, ds_y, type_y, "reg_y")
        
        # Prepare data
        if use_bp_change:
            series_x = series_x.diff() * 100
            series_y = series_y.diff() * 100
        
        reg_df = pd.concat([series_x, series_y], axis=1).iloc[-reg_window:].dropna()
        
        if len(reg_df) > 1:
            x_vals = reg_df.iloc[:, 0].to_numpy()
            y_vals = reg_df.iloc[:, 1].to_numpy()
            
            if not (np.isclose(np.nanstd(x_vals), 0) or np.isclose(np.nanstd(y_vals), 0)):
                m, b, r, p, err = stats.linregress(x_vals, y_vals)
                
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Beta (Slope)", f"{float(m):.4f}")
                col_m2.metric("R-Squared", f"{float(r**2):.4f}")
                col_m3.metric("P-Value", f"{float(p):.4f}")
                
                # Scatter + Regression Line
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=reg_df.iloc[:, 0], y=reg_df.iloc[:, 1], mode='markers', name='Observations'))
                x_line = np.array([reg_df.iloc[:, 0].min(), reg_df.iloc[:, 0].max()])
                y_line = m * x_line + b
                fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Regression', line=dict(color='red')))
                fig.update_layout(
                    title=f"{title_y} vs {title_x}",
                    xaxis_title=title_x, yaxis_title=title_y,
                    height=750, template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need variation in both X and Y.")
        else:
            st.warning("Not enough overlapping data.")
    
    with tab_hedge:
        st.subheader("Find Best Hedging Instrument")
        
        col_h1, col_h2 = st.columns(2)
        
        with col_h1:
            st.markdown("**Target Position**")
            ds_target = st.selectbox("Dataset", ["LOIS", "ER"], key="hedge_ds_target")
            type_target = st.selectbox("Type", ["Outright", "Spread", "Fly"], key="hedge_type_target")
            series_target, title_target = build_series_from_selection(df_master, ds_target, type_target, "hedge_target")
            
            hedge_window = st.slider("Regression Window", 30, 252, 90, key="hedge_window")
            search_dataset = st.selectbox("Search in", ["ALL", "LOIS", "ER"], key="hedge_search")
            
            if st.button("Find Best Hedges"):
                # Get candidate instruments
                if search_dataset == "ALL":
                    candidates = [c for c in available_cols if c not in [series_target.name]]
                else:
                    candidates = get_cols(df_master, search_dataset + "_")
                    candidates = [c for c in candidates if c not in [series_target.name]]
                
                target_chg = (series_target.diff() * 100).iloc[-hedge_window:].dropna()
                results = []
                
                for cand in candidates:
                    cand_chg = (df_master[cand].diff() * 100).iloc[-hedge_window:].dropna()
                    df_align = pd.concat([target_chg, cand_chg], axis=1).dropna()
                    
                    if len(df_align) < 30:
                        continue
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(df_align.iloc[:, 1], df_align.iloc[:, 0])
                    corr = df_align.corr().iloc[0, 1]
                    
                    results.append({
                        "Instrument": cand,
                        "R-Squared": r_value**2,
                        "Beta": slope,
                        "Correlation": corr
                    })
                
                if results:
                    df_results = pd.DataFrame(results).sort_values("R-Squared", ascending=False)
                    st.session_state['hedge_results'] = df_results.head(10)
                    st.session_state['hedge_target_title'] = title_target
                    st.success(f"Found {len(df_results)} candidates!")
                else:
                    st.warning("No suitable candidates found.")
        
        with col_h2:
            if 'hedge_results' in st.session_state:
                st.markdown("**Top 10 Hedging Candidates**")
                st.dataframe(st.session_state['hedge_results'], use_container_width=True)
                
                best = st.session_state['hedge_results'].iloc[0]
                st.success(f"**{best['Instrument']}** with Beta={best['Beta']:.4f} (R²={best['R-Squared']:.2f})")

# =====================================================================
# TAB 4: RANGE & CURVE ANALYSIS
# =====================================================================
with tab_range:
    st.header("📉 Range & Curve Analysis")
    
    tab_r1, tab_r2, tab_r3, tab_r4 = st.tabs(["Range Analysis", "Front-to-Back Transmission", "Curve Comparison", "Correlation & Regression Matrix"])
    
    with tab_r1:
        st.subheader("Historical Range")
        
        col_r1_1, col_r1_2 = st.columns(2)
        with col_r1_1:
            market = st.selectbox("Market", ["LOIS", "ER"], key="range_market")
        with col_r1_2:
            type_range = st.selectbox("Type", ["Outright", "Spread", "Fly"], key="range_type")
        
        def build_range_table(df_master, prefix, type_r):
            cols = get_cols(df_master, prefix + "_")
            cols = [c for c in cols if "_F" in c or "Spot" in c]
            rows = []
            
            if type_r == "Outright":
                for col in cols:
                    ser = df_master[col].dropna()
                    if not ser.empty:
                        rows.append({
                            "Instrument": col,
                            "Min": ser.min(),
                            "Max": ser.max(),
                            "P5": ser.quantile(0.05),
                            "P95": ser.quantile(0.95),
                            "Range Width": ser.quantile(0.95) - ser.quantile(0.05)
                        })
            elif type_r == "Spread":
                for i in range(len(cols) - 1):
                    for j in range(i+1, len(cols)):
                        ser = (df_master[cols[i]] - df_master[cols[j]]).dropna()
                        if not ser.empty:
                            rows.append({
                                "Instrument": f"{cols[i]}-{cols[j]}",
                                "Min": ser.min(),
                                "Max": ser.max(),
                                "P5": ser.quantile(0.05),
                                "P95": ser.quantile(0.95),
                                "Range Width": ser.quantile(0.95) - ser.quantile(0.05)
                            })
            else:  # Fly
                for i in range(len(cols) - 2):
                    for j in range(i+1, len(cols)-1):
                        for k in range(j+1, len(cols)):
                            ser = ((df_master[cols[i]] + df_master[cols[k]]) - (2 * df_master[cols[j]])).dropna()
                            if not ser.empty:
                                rows.append({
                                    "Instrument": f"{cols[i]}-{cols[j]}-{cols[k]}",
                                    "Min": ser.min(),
                                    "Max": ser.max(),
                                    "P5": ser.quantile(0.05),
                                    "P95": ser.quantile(0.95),
                                    "Range Width": ser.quantile(0.95) - ser.quantile(0.05)
                                })
            
            return pd.DataFrame(rows)
        
        df_ranges = build_range_table(df_master, market, type_range)
        if not df_ranges.empty:
            styled = df_ranges.style.format({c: "{:.4f}" for c in df_ranges.columns if c != "Instrument"})
            styled = styled.background_gradient(subset=["Range Width"], cmap="YlOrRd")
            st.dataframe(styled, use_container_width=True)
        else:
            st.info("No data available.")
    
    with tab_r2:
        st.subheader("Front-to-Back Transmission (Curve Slope Analysis)")
        
        col_fb1, col_fb2 = st.columns(2)
        
        with col_fb1:
            st.markdown("**Front (X)**")
            ds_f = st.selectbox("Dataset", ["LOIS", "ER"], key="fb_ds_f")
            type_f = st.selectbox("Type", ["Outright", "Spread", "Fly"], key="fb_type_f")
            ser_f, title_f = build_series_from_selection(df_master, ds_f, type_f, "fb_f")
        
        with col_fb2:
            st.markdown("**Back (Y)**")
            ds_b = st.selectbox("Dataset", ["LOIS", "ER"], key="fb_ds_b", index=1)
            type_b = st.selectbox("Type", ["Outright", "Spread", "Fly"], key="fb_type_b", index=1)
            ser_b, title_b = build_series_from_selection(df_master, ds_b, type_b, "fb_b")
        
        # Calculate changes
        x_chg = ser_f.diff().dropna()
        y_chg = ser_b.diff().dropna()
        fb_df = pd.concat([x_chg, y_chg], axis=1).dropna()
        
        if len(fb_df) > 1:
            # --- 1. STANDARD CONTEMPORANEOUS REGRESSION ---
            m, b, r, p, err = stats.linregress(fb_df.iloc[:, 0], fb_df.iloc[:, 1])
            
            st.markdown("### Contemporaneous Relationship (Same Day)")
            col_fb_m1, col_fb_m2, col_fb_m3 = st.columns(3)
            col_fb_m1.metric("Beta (Same Day)", f"{float(m):.4f}")
            col_fb_m2.metric("R-Squared", f"{float(r**2):.4f}")
            col_fb_m3.metric("P-Value", f"{float(p):.4f}")
            
            fig_fb = go.Figure()
            fig_fb.add_trace(go.Scatter(x=fb_df.iloc[:, 0], y=fb_df.iloc[:, 1], mode='markers', name='Observations'))
            x_line = np.array([fb_df.iloc[:, 0].min(), fb_df.iloc[:, 0].max()])
            y_line = m * x_line + b
            fig_fb.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Regression', line=dict(color='red')))
            fig_fb.update_layout(
                title="Same Day Movement Correlation",
                xaxis_title=f"Δ {title_f}", yaxis_title=f"Δ {title_b}",
                height=550, template="plotly_white"
            )
            st.plotly_chart(fig_fb, use_container_width=True)

            # --- 2. LEAD / LAG ANALYSIS ---
            st.markdown("### Lead/Lag Analysis")
            st.caption("Analyzes if movements in Front predict movements in Back with a time delay (or vice versa).")
            
            max_lag = st.slider("Max Lag Days to Test", 1, 30, 10, key="fb_max_lag")
            
            lags = np.arange(-max_lag, max_lag + 1)
            corrs = []
            
            # Calculate Cross-Correlation for each lag
            for lag in lags:
                # Lag > 0: Front leads Back (Corr(Front_t, Back_{t+lag}))
                # Lag < 0: Back leads Front
                # To correlate Front(t) with Back(t+lag), shift Back by -lag
                shifted_y = y_chg.shift(-lag)
                c = x_chg.corr(shifted_y)
                corrs.append(c)
            
            # Find optimal lag
            best_idx = np.argmax(np.abs(corrs))
            best_lag = lags[best_idx]
            best_corr = corrs[best_idx]
            
            # Plot Cross-Correlation Function
            fig_lag = go.Figure()
            fig_lag.add_trace(go.Bar(x=lags, y=corrs, name="Correlation", marker_color='lightblue'))
            fig_lag.add_trace(go.Scatter(x=lags, y=corrs, mode='lines', line=dict(color='blue'), name='Trend'))
            
            # Highlight peak
            fig_lag.add_trace(go.Scatter(
                x=[best_lag],
                y=[best_corr],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='Peak Correlation'
            ))
            
            fig_lag.update_layout(
                title="Cross-Correlation Function (Front vs Back)",
                xaxis_title="Lag (Days) [Positive = Front Leads]",
                yaxis_title="Correlation Coefficient",
                height=550, template="plotly_white"
            )
            st.plotly_chart(fig_lag, use_container_width=True)
            
            # Interpretation Logic
            col_l1, col_l2, col_l3 = st.columns(3)
            col_l1.metric("Optimal Lag (Days)", best_lag)
            col_l2.metric("Correlation at Lag", f"{best_corr:.4f}")
            
            # Calculate R² and p-value for optimal lag
            if best_lag > 0:
                y_lagged_temp = y_chg.shift(-best_lag)
                x_lagged_temp = x_chg
            elif best_lag < 0:
                y_lagged_temp = x_chg.shift(-best_lag)
                x_lagged_temp = y_chg
            else:
                y_lagged_temp = y_chg
                x_lagged_temp = x_chg
            
            df_temp_reg = pd.concat([x_lagged_temp, y_lagged_temp], axis=1).dropna()
            if len(df_temp_reg) > 2:
                _, _, r_temp, p_temp, _ = stats.linregress(df_temp_reg.iloc[:, 0], df_temp_reg.iloc[:, 1])
                r_squared_lag = r_temp**2
                col_l3.metric("R² at Optimal Lag", f"{r_squared_lag:.4f}")
                
                # Add p-value below the metrics
                st.caption(f"**p-value at Optimal Lag:** {p_temp:.4e}")

            # Comparison table for optimal lag
            st.markdown("### Lagged Comparison Table")
            shifted_y_best = y_chg.shift(-best_lag)
            df_compare = pd.concat([x_chg, shifted_y_best], axis=1).dropna()
            df_compare.columns = [f"Δ {title_f} (t)", f"Δ {title_b} (t+{best_lag})"]
            rows_to_show = st.number_input(
                "Rows to display",
                min_value=5,
                max_value=min(200, len(df_compare)) if len(df_compare) > 0 else 5,
                value=min(20, len(df_compare)) if len(df_compare) > 0 else 5,
                key="fb_lag_table_rows"
            )
            st.dataframe(df_compare.head(int(rows_to_show)), use_container_width=True)
            
            # --- 3. REGRESSION AT OPTIMAL LAG ---
            # Regress the lagger against the leader shifted by the lag
            if best_lag != 0:
                st.markdown("### Regression at Optimal Lag")
                
                if best_lag > 0:
                    # Front Leads Back. Predict Back(t+lag) using Front(t)
                    st.info(f"Front contract leads Back contract by {best_lag} days.")
                    st.latex(r"\Delta \text{Back}_{t+" + str(best_lag) + r"} = \beta \cdot \Delta \text{Front}_t + \alpha")
                    
                    y_lagged = y_chg.shift(-best_lag)
                    x_lagged = x_chg
                    x_label = f"Δ {title_f} (t)"
                    y_label = f"Δ {title_b} (t+{best_lag})"
                else:
                    # Back Leads Front. Predict Front(t+lag) using Back(t)
                    st.info(f"Back contract leads Front contract by {abs(best_lag)} days.")
                    st.latex(r"\Delta \text{Front}_{t+" + str(abs(best_lag)) + r"} = \beta \cdot \Delta \text{Back}_t + \alpha")
                    
                    y_lagged = x_chg.shift(-best_lag)
                    x_lagged = y_chg
                    x_label = f"Δ {title_b} (t)"
                    y_label = f"Δ {title_f} (t+{abs(best_lag)})"
                
                # Create aligned dataframe for regression
                df_lag_reg = pd.concat([x_lagged, y_lagged], axis=1).dropna()
                
                if len(df_lag_reg) > 10:
                    m_lag, b_lag, r_lag, p_lag, _ = stats.linregress(df_lag_reg.iloc[:, 0], df_lag_reg.iloc[:, 1])
                    
                    col_l3.metric("Lagged Beta", f"{m_lag:.4f}")
                    
                    # Plot Lagged Regression
                    fig_lag_reg = go.Figure()
                    fig_lag_reg.add_trace(go.Scatter(x=df_lag_reg.iloc[:, 0], y=df_lag_reg.iloc[:, 1], mode='markers', name='Observations'))
                    
                    # Add regression line
                    x_line_lag = np.array([df_lag_reg.iloc[:, 0].min(), df_lag_reg.iloc[:, 0].max()])
                    y_line_lag = m_lag * x_line_lag + b_lag
                    fig_lag_reg.add_trace(go.Scatter(x=x_line_lag, y=y_line_lag, mode='lines', name='Regression', line=dict(color='red')))
                    
                    fig_lag_reg.update_layout(
                        title="Lagged Regression Plot",
                        xaxis_title=x_label, yaxis_title=y_label,
                        height=550, template="plotly_white"
                    )
                    st.plotly_chart(fig_lag_reg, use_container_width=True)
            else:
                col_l3.metric("Lagged Beta", "N/A (Same Day)")
    
    with tab_r3:
        st.subheader("Curve Comparison (Multi-Date)")
        
        curve_dates = df_master.index.sort_values()
        if not curve_dates.empty:
            # Build curves helper function
            def get_curve_data(df_master, prefix, c_type, date_val, aggregation='Daily'):
                cols = sorted([c for c in df_master.columns if c.startswith(prefix + "_") and "_F" in c], 
                             key=lambda x: get_contract_num(x))
                if not cols:
                    return [], []
                
                # Get the date range for aggregation
                if aggregation == 'Daily':
                    if date_val not in df_master.index:
                        return [], []
                    row = df_master.loc[date_val]
                elif aggregation == 'Weekly':
                    # Get the week containing date_val (Monday to Sunday)
                    week_start = date_val - pd.Timedelta(days=date_val.weekday())
                    week_end = week_start + pd.Timedelta(days=6)
                    week_data = df_master.loc[(df_master.index >= week_start) & (df_master.index <= week_end)]
                    if week_data.empty:
                        return [], []
                    row = week_data.mean()
                else:  # Monthly
                    # Get the month containing date_val
                    month_start = date_val.replace(day=1)
                    if date_val.month == 12:
                        month_end = date_val.replace(year=date_val.year + 1, month=1, day=1) - pd.Timedelta(days=1)
                    else:
                        month_end = date_val.replace(month=date_val.month + 1, day=1) - pd.Timedelta(days=1)
                    month_data = df_master.loc[(df_master.index >= month_start) & (df_master.index <= month_end)]
                    if month_data.empty:
                        return [], []
                    row = month_data.mean()
                
                labels, values = [], []
                
                if c_type == "Outright":
                    for col in cols:
                        labels.append(col.split("_")[1])
                        values.append(row.get(col, np.nan))
                elif c_type == "Spread":
                    for i in range(len(cols)-1):
                        labels.append(f"{cols[i].split('_')[1]}-{cols[i+1].split('_')[1]}")
                        values.append(row.get(cols[i], np.nan) - row.get(cols[i+1], np.nan))
                else:  # Fly
                    for i in range(len(cols)-2):
                        labels.append(f"{cols[i].split('_')[1]}-{cols[i+1].split('_')[1]}-{cols[i+2].split('_')[1]}")
                        values.append((row.get(cols[i], np.nan) + row.get(cols[i+2], np.nan)) - (2*row.get(cols[i+1], np.nan)))
                
                return labels, values
            
            # Number of curves to compare
            num_curves = st.number_input("Number of curves to compare", 1, 10, 2, 1, key="num_curves")
            
            curves_config = []
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            for i in range(int(num_curves)):
                with st.expander(f"Curve {i+1}", expanded=(i < 2)):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        date_idx = min(len(curve_dates)-1-i, len(curve_dates)-1)
                        curve_date = st.selectbox(
                            f"Date",
                            curve_dates,
                            index=date_idx,
                            key=f"curve_date_{i}"
                        )
                    with col2:
                        aggregation = st.selectbox(
                            f"Period",
                            ["Daily", "Weekly", "Monthly"],
                            key=f"curve_agg_{i}"
                        )
                    with col3:
                        dataset = st.selectbox(
                            f"Dataset",
                            ["LOIS", "ER"],
                            key=f"curve_ds_{i}"
                        )
                    with col4:
                        curve_type = st.selectbox(
                            f"Type",
                            ["Outright", "Spread", "Fly"],
                            key=f"curve_type_{i}"
                        )
                    
                    # Individual transform controls for each curve
                    st.markdown("**Transforms (optional)**")
                    col_t1, col_t2, col_t3 = st.columns(3)
                    with col_t1:
                        curve_expr = st.text_input("Expression (use x)", value="", key=f"curve_expr_{i}", help="Example: x*100-2 or np.log(x)")
                        y_offset = st.number_input("Add (+/-)", value=0.0, step=0.1, format="%.4f", key=f"curve_y_offset_{i}", help="Constant to add/subtract")
                    with col_t2:
                        y_scale = st.number_input("Multiply (×)", value=1.0, step=0.1, format="%.2f", key=f"curve_y_scale_{i}", help="Factor to multiply by")
                    with col_t3:
                        normalize = st.checkbox("Normalize", key=f"curve_norm_{i}")
                    
                    if normalize:
                        col_n1, col_n2 = st.columns(2)
                        with col_n1:
                            y_min = st.number_input("Min", value=0.0, step=1.0, key=f"curve_y_min_{i}")
                        with col_n2:
                            y_max = st.number_input("Max", value=100.0, step=1.0, key=f"curve_y_max_{i}")
                        normalize_to = (y_min, y_max)
                    else:
                        normalize_to = None
                    
                    curves_config.append({
                        'date': curve_date,
                        'dataset': dataset,
                        'type': curve_type,
                        'aggregation': aggregation,
                        'color': colors[i % len(colors)],
                        'y_offset': y_offset,
                        'y_scale': y_scale,
                        'normalize_to': normalize_to,
                        'expr': curve_expr
                    })

            # Align curves to common timeframe when LOIS and ER are both selected
            datasets_selected = {c['dataset'] for c in curves_config}
            use_common_timeframe = False
            if "LOIS" in datasets_selected and "ER" in datasets_selected:
                use_common_timeframe = st.checkbox(
                    "Use only common timeframe values (LOIS vs ER)",
                    value=True,
                    key="curve_common_timeframe"
                )
                lois_dates = get_dataset_available_dates(df_master, "LOIS")
                er_dates = get_dataset_available_dates(df_master, "ER")
                common_curve_dates = lois_dates.intersection(er_dates)
            else:
                common_curve_dates = None
            
            # Build the figure
            fig_curve = go.Figure()
            
            info_shown = False
            for i, config in enumerate(curves_config):
                if use_common_timeframe and config['aggregation'] != 'Daily' and not info_shown:
                    st.info("Common timeframe alignment applies only to Daily curves.")
                    info_shown = True

                if use_common_timeframe and config['aggregation'] == 'Daily' and common_curve_dates is not None:
                    if config['date'] not in common_curve_dates:
                        st.warning(f"Skipping {config['dataset']} curve on {config['date'].date()} (not in common timeframe).")
                        continue
                labels, values = get_curve_data(
                    df_master,
                    config['dataset'],
                    config['type'],
                    config['date'],
                    config['aggregation']
                )
                if labels:
                    # Apply transforms
                    values_transformed = apply_transforms(
                        values,
                        y_offset=config['y_offset'],
                        y_scale=config['y_scale'],
                        normalize_to=config['normalize_to'],
                        expr=config.get('expr')
                    )
                    
                    # Build name with aggregation info
                    if config['aggregation'] == 'Daily':
                        name = f"{config['dataset']} {config['type']} ({config['date'].date()})"
                    else:
                        name = f"{config['dataset']} {config['type']} ({config['aggregation']} @ {config['date'].date()})"
                    
                    fig_curve.add_trace(go.Scatter(
                        x=labels,
                        y=values_transformed,
                        mode='lines+markers',
                        name=name,
                        line=dict(color=config['color']),
                        marker=dict(size=6)
                    ))
            
            fig_curve.update_layout(
                title="Yield Curve Comparison",
                xaxis_title="Contract",
                yaxis_title="Value",
                height=750,
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig_curve, use_container_width=True)
    
    with tab_r4:
        st.subheader("Correlation & Regression Matrix")
        st.markdown("Calculate correlations and regression metrics for all instrument pairs or anchor vs all.")

        def build_all_instruments(df_master, prefix, inst_type, include_spot=True):
            """Build all instruments of a given type."""
            cols = get_cols(df_master, prefix + "_")
            cols = [c for c in cols if "_F" in c or "Spot" in c]
            if not include_spot:
                cols = [c for c in cols if "Spot" not in c]
            inst_dict = {}

            if inst_type == "Outright":
                for col in cols:
                    ser = df_master[col].dropna()
                    if not ser.empty:
                        inst_dict[col] = ser
            elif inst_type == "Spread":
                for i in range(len(cols) - 1):
                    for j in range(i + 1, len(cols)):
                        name = f"{cols[i]}-{cols[j]}"
                        ser = (df_master[cols[i]] - df_master[cols[j]]).dropna()
                        if not ser.empty:
                            inst_dict[name] = ser
            else:  # Fly
                for i in range(len(cols) - 2):
                    for j in range(i + 1, len(cols) - 1):
                        for k in range(j + 1, len(cols)):
                            name = f"{cols[i]}-{cols[j]}-{cols[k]}"
                            ser = ((df_master[cols[i]] + df_master[cols[k]]) - (2 * df_master[cols[j]])).dropna()
                            if not ser.empty:
                                inst_dict[name] = ser

            return inst_dict

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            matrix_mode = st.selectbox(
                "Matrix Mode",
                ["All pairs (same type)", "Anchor vs all"],
                key="corr_reg_mode"
            )
        with col_m2:
            max_corr_window = max(10, len(df_master))
            corr_window = st.number_input(
                "Correlation Window (Days)",
                min_value=10,
                max_value=max_corr_window,
                value=min(60, max_corr_window),
                key="corr_window_input"
            )

        max_reg_window = max(30, len(df_master))
        reg_window = st.number_input(
            "Regression Window (Days)",
            min_value=30,
            max_value=max_reg_window,
            value=min(90, max_reg_window),
            key="reg_window_input"
        )

        include_spot = st.checkbox("Include Spot", value=True, key="matrix_include_spot")
        require_same_side = st.checkbox(
            "Disallow shared-leg mismatches",
            value=False,
            key="matrix_same_side"
        )

        if matrix_mode == "All pairs (same type)":
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                corr_market = st.selectbox("Market", ["LOIS", "ER"], key="corr_market")
            with col_a2:
                corr_type = st.selectbox("Instrument Type", ["Outright", "Spread", "Fly"], key="corr_type")
        else:
            st.markdown("**Anchor**")
            col_anchor1, col_anchor2 = st.columns(2)
            with col_anchor1:
                anchor_market = st.selectbox("Anchor Market", ["LOIS", "ER"], key="anchor_market")
            with col_anchor2:
                anchor_type = st.selectbox("Anchor Type", ["Outright", "Spread", "Fly"], key="anchor_type")

            anchor_series, anchor_title = build_series_from_selection(
                df_master, anchor_market, anchor_type, "anchor_matrix", include_spot=include_spot
            )
            anchor_name = anchor_title.split(": ", 1)[1] if ": " in anchor_title else anchor_title

            st.markdown("**Targets**")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                target_market = st.selectbox("Target Market", ["LOIS", "ER"], key="target_market")
            with col_t2:
                target_types = st.multiselect(
                    "Target Types",
                    ["Outright", "Spread", "Fly"],
                    default=["Outright", "Spread", "Fly"],
                    key="target_types"
                )

        # Analysis type selection
        analysis_type = st.radio(
            "Analysis Type",
            ["Contemporaneous", "Lead/Lag"],
            key="matrix_analysis_type",
            horizontal=True
        )
        
        max_lag_matrix = 10
        if analysis_type == "Lead/Lag":
            max_lag_matrix = st.slider(
                "Max Lag Days to Test",
                1, 30, 10,
                key="matrix_max_lag"
            )

        if st.button("Calculate Correlation & Regression Matrix", key="calc_corr_reg"):
            if matrix_mode == "All pairs (same type)":
                inst_dict = build_all_instruments(df_master, corr_market, corr_type, include_spot=include_spot)
                inst_names = list(inst_dict.keys())

                if len(inst_names) < 2:
                    st.warning(f"Need at least 2 {corr_type} instruments to calculate correlations.")
                    st.stop()

                with st.spinner(f"Calculating {'lead/lag' if analysis_type == 'Lead/Lag' else 'contemporaneous'} analysis..."):
                    results = []

                    for i in range(len(inst_names)):
                        for j in range(i + 1, len(inst_names)):
                            inst1_name = inst_names[i]
                            inst2_name = inst_names[j]

                            if require_same_side and not shared_legs_only_if_exact_match(inst1_name, inst2_name):
                                continue

                            ser1 = inst_dict[inst1_name]
                            ser2 = inst_dict[inst2_name]

                            pct1 = ser1.diff() * 100
                            pct2 = ser2.diff() * 100

                            if analysis_type == "Lead/Lag":
                                # Lead/Lag Analysis
                                analysis_df = pd.concat([pct1, pct2], axis=1).iloc[-reg_window:].dropna()
                                
                                if len(analysis_df) < 30:
                                    continue
                                
                                x_chg = analysis_df.iloc[:, 0]
                                y_chg = analysis_df.iloc[:, 1]
                                
                                # Calculate cross-correlation for all lags
                                lags = np.arange(-max_lag_matrix, max_lag_matrix + 1)
                                corrs = []
                                
                                for lag in lags:
                                    shifted_y = y_chg.shift(-lag)
                                    c = x_chg.corr(shifted_y)
                                    corrs.append(c)
                                
                                # Find optimal lag
                                best_idx = np.argmax(np.abs(corrs))
                                best_lag = lags[best_idx]
                                best_corr = corrs[best_idx]
                                
                                # Calculate regression at optimal lag
                                if best_lag > 0:
                                    y_lagged = y_chg.shift(-best_lag)
                                    x_lagged = x_chg
                                elif best_lag < 0:
                                    y_lagged = x_chg.shift(-best_lag)
                                    x_lagged = y_chg
                                else:
                                    y_lagged = y_chg
                                    x_lagged = x_chg
                                
                                df_lag_reg = pd.concat([x_lagged, y_lagged], axis=1).dropna()
                                
                                if len(df_lag_reg) > 10:
                                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                                        df_lag_reg.iloc[:, 0], df_lag_reg.iloc[:, 1]
                                    )
                                    
                                    results.append({
                                        "Instrument 1": inst1_name,
                                        "Instrument 2": inst2_name,
                                        "Best Lag": best_lag,
                                        "Correlation": best_corr,
                                        "Beta": slope,
                                        "R-Squared": r_value**2,
                                        "P-Value": p_value,
                                        "Observations": len(df_lag_reg)
                                    })
                            else:
                                # Contemporaneous Analysis
                                corr_df = pd.concat([pct1, pct2], axis=1).iloc[-corr_window:].dropna()
                                reg_df = pd.concat([pct1, pct2], axis=1).iloc[-reg_window:].dropna()

                                if len(reg_df) < 30:
                                    continue

                                if len(corr_df) < 2:
                                    continue

                                corr = corr_df.corr().iloc[0, 1]
                                x_vals = reg_df.iloc[:, 0].to_numpy()
                                y_vals = reg_df.iloc[:, 1].to_numpy()
                                slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)

                                results.append({
                                    "Instrument 1": inst1_name,
                                    "Instrument 2": inst2_name,
                                    "Correlation": corr,
                                    "Beta": slope,
                                    "R-Squared": r_value**2,
                                    "P-Value": p_value,
                                    "Observations": len(reg_df)
                                })
            else:
                # Anchor vs All mode
                if anchor_series.empty:
                    st.warning("Anchor series has no data.")
                    st.stop()

                if not target_types:
                    st.warning("Select at least one target type.")
                    st.stop()

                target_dict = {}
                for t_type in target_types:
                    for name, ser in build_all_instruments(df_master, target_market, t_type, include_spot=include_spot).items():
                        target_dict[name] = (t_type, ser)

                if not target_dict:
                    st.warning("No target instruments found for the selected types.")
                    st.stop()

                with st.spinner(f"Calculating {'lead/lag' if analysis_type == 'Lead/Lag' else 'contemporaneous'} analysis..."):
                    results = []
                    anchor_pct = anchor_series.diff() * 100

                    for target_name, (target_type, target_ser) in target_dict.items():
                        if target_name == anchor_name:
                            continue

                        if require_same_side and not shared_legs_only_if_exact_match(anchor_name, target_name):
                            continue

                        target_pct = target_ser.diff() * 100
                        
                        if analysis_type == "Lead/Lag":
                            # Lead/Lag Analysis
                            analysis_df = pd.concat([anchor_pct, target_pct], axis=1).iloc[-reg_window:].dropna()
                            
                            if len(analysis_df) < 30:
                                continue
                            
                            x_chg = analysis_df.iloc[:, 0]
                            y_chg = analysis_df.iloc[:, 1]
                            
                            # Calculate cross-correlation for all lags
                            lags = np.arange(-max_lag_matrix, max_lag_matrix + 1)
                            corrs = []
                            
                            for lag in lags:
                                shifted_y = y_chg.shift(-lag)
                                c = x_chg.corr(shifted_y)
                                corrs.append(c)
                            
                            # Find optimal lag
                            best_idx = np.argmax(np.abs(corrs))
                            best_lag = lags[best_idx]
                            best_corr = corrs[best_idx]
                            
                            # Calculate regression at optimal lag
                            if best_lag > 0:
                                y_lagged = y_chg.shift(-best_lag)
                                x_lagged = x_chg
                            elif best_lag < 0:
                                y_lagged = x_chg.shift(-best_lag)
                                x_lagged = y_chg
                            else:
                                y_lagged = y_chg
                                x_lagged = x_chg
                            
                            df_lag_reg = pd.concat([x_lagged, y_lagged], axis=1).dropna()
                            
                            if len(df_lag_reg) > 10:
                                slope, intercept, r_value, p_value, std_err = stats.linregress(
                                    df_lag_reg.iloc[:, 0], df_lag_reg.iloc[:, 1]
                                )
                                
                                results.append({
                                    "Anchor": anchor_title,
                                    "Instrument": target_name,
                                    "Type": target_type,
                                    "Best Lag": best_lag,
                                    "Correlation": best_corr,
                                    "Beta": slope,
                                    "R-Squared": r_value**2,
                                    "P-Value": p_value,
                                    "Observations": len(df_lag_reg)
                                })
                        else:
                            # Contemporaneous Analysis
                            corr_df = pd.concat([anchor_pct, target_pct], axis=1).iloc[-corr_window:].dropna()
                            reg_df = pd.concat([anchor_pct, target_pct], axis=1).iloc[-reg_window:].dropna()

                            if len(reg_df) < 30:
                                continue

                            if len(corr_df) < 2:
                                continue

                            corr = corr_df.corr().iloc[0, 1]
                            x_vals = reg_df.iloc[:, 0].to_numpy()
                            y_vals = reg_df.iloc[:, 1].to_numpy()
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)

                            results.append({
                                "Anchor": anchor_title,
                                "Instrument": target_name,
                                "Type": target_type,
                                "Correlation": corr,
                                "Beta": slope,
                                "R-Squared": r_value**2,
                                "P-Value": p_value,
                                "Observations": len(reg_df)
                            })

            if results:
                df_results = pd.DataFrame(results).sort_values("Correlation", ascending=False, key=abs)

                # Format based on analysis type
                if analysis_type == "Lead/Lag":
                    format_dict = {
                        "Best Lag": "{:.0f}",
                        "Correlation": "{:.4f}",
                        "Beta": "{:.4f}",
                        "R-Squared": "{:.4f}",
                        "P-Value": "{:.4f}"
                    }
                    styled = df_results.style.format(format_dict)
                    styled = styled.background_gradient(subset=["Best Lag"], cmap="RdYlGn_r", vmin=-max_lag_matrix, vmax=max_lag_matrix)
                else:
                    format_dict = {
                        "Correlation": "{:.4f}",
                        "Beta": "{:.4f}",
                        "R-Squared": "{:.4f}",
                        "P-Value": "{:.4f}"
                    }
                    styled = df_results.style.format(format_dict)
                
                styled = styled.background_gradient(subset=["Correlation", "R-Squared"], cmap="RdYlGn", vmin=-1, vmax=1)
                styled = styled.background_gradient(subset=["Beta"], cmap="RdBu_r")

                st.dataframe(styled, use_container_width=True, height=700)
                
                if analysis_type == "Lead/Lag":
                    st.info(f"**Total pairs calculated:** {len(df_results)} | Positive lag = Instrument 1 leads, Negative lag = Instrument 2 leads")
                else:
                    st.info(f"**Total pairs calculated:** {len(df_results)}")
            else:
                st.warning("No valid pairs found for analysis.")

# =====================================================================
# TAB 5: RISK MANAGEMENT
# =====================================================================
with tab_risk:
    st.header("⚖️ Risk Management")
    
    col_r1, col_r2, col_r3 = st.columns(3)
    
    with col_r1:
        st.subheader("1. Position")
        market_risk = st.selectbox("Market", ["LOIS", "ER"], key="risk_market")
        type_risk = st.selectbox("Type", ["Outright", "Spread", "Fly"], key="risk_type")
        ser_risk, title_risk = build_series_from_selection(df_master, market_risk, type_risk, "risk")
        
        direction = st.radio("Direction", ["Long", "Short"], horizontal=True)
        entry_price = st.number_input("Entry Price", value=float(ser_risk.iloc[-1]))
    
    with col_r2:
        st.subheader("2. Account")
        account_size = st.number_input("Account (EUR)", value=1000000, step=100000)
        risk_pct = st.slider("Risk %", 0.1, 10.0, 1.0, 0.1)
        var_conf = st.slider("VaR Confidence %", 90.0, 99.9, 95.0)
    
    with col_r3:
        st.subheader("3. Stop Loss")
        sl_method = st.radio("Method", ["ATR", "Manual"], horizontal=True)
        if sl_method == "ATR":
            atr_win_risk = st.number_input("ATR Window", 5, 50, 20)
            atr_mult_risk = st.number_input("ATR Mult", 0.5, 5.0, 2.0, 0.1)
            atr_risk = ser_risk.diff().abs().rolling(atr_win_risk).mean().iloc[-1]
            sl_dist = atr_risk * atr_mult_risk
        else:
            sl_price = st.number_input("SL Price", value=float(entry_price*0.95))
            sl_dist = abs(entry_price - sl_price)
    
    # Calcs
    var_95, cvar_95 = calculate_var(ser_risk, var_conf/100)
    var_eur = account_size * abs(var_95)
    risk_amount = account_size * (risk_pct / 100)
    pos_size = risk_amount / sl_dist if sl_dist > 0 else 0
    notional = pos_size * entry_price
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Daily VaR", f"€{var_eur:,.0f}", f"{var_95*100:.2f}%")
    col_m2.metric("Max Position Size", f"{pos_size:,.2f}", f"€{notional:,.0f}")
    col_m3.metric("Stop Distance", f"{sl_dist:.4f}")
    
    st.info(f"Risk **€{risk_amount:,.0f}** with SL {sl_dist:.4f} → **{pos_size:,.2f}** units")

# =====================================================================
# TAB 6: CALENDAR MANAGEMENT
# =====================================================================
with tab_calendar:
    st.header("📅 ECB Calendar")
    
    edited_df = st.data_editor(
        st.session_state['ecb_calendar'],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "Action": st.column_config.SelectboxColumn("Action", options=["Hike", "Cut", "Hold"]),
            "Change": st.column_config.TextColumn("Change"),
            "Rate": st.column_config.TextColumn("Rate")
        },
        hide_index=True,
        key="ecb_calendar_editor"
    )
    
    if not edited_df.equals(st.session_state['ecb_calendar']):
        st.session_state['ecb_calendar'] = edited_df
        st.success("Calendar updated!")
