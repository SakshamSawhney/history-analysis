import streamlit as st
import pandas as pd
import numpy as np
import re
import io

# --- Configuration & Styling ---
st.set_page_config(
    page_title="FI Trading Dashboard - Case Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Financial Terminal Look
st.markdown("""
<style>
    /* Font */
    .main {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        font-size: 13px;
    }
    
    /* Table Header & Cell tweaks */
    .stDataFrame {
        width: 100%;
        font-size: 12px;
    }
    th {
        text-align: center !important;
        background-color: #f0f0f0;
        font-weight: bold;
    }
    
    /* Hide generic streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #e0e0e0;
        padding: 5px 5px 0 5px;
        border-radius: 5px 5px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #d0d0d0;
        border-radius: 4px 4px 0 0;
        padding: 8px 15px;
        font-weight: 600;
        color: #333;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #d32f2f; /* Active color red */
        border-bottom: 2px solid #d32f2f;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #fafafa;
        border: 1px solid #ddd;
        padding: 8px;
        border-radius: 4px;
    }

    /* Number Input styling */
    input[type="number"] {
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Parsing (Embedded from your text) ---

RAW_DATA_STRING = """
HEAD | Meeting Date | Effective Date | Hike/Cut (bps) | Implied Rate | Meeting Code | contr start | contr end | * | srl contr start | srl contr end | 
0 | 11-Feb-26 | 11-Feb-26 | * | * | Feb-26 | 3/18/2026 | 6/16/2026 | Tuesday | 2/18/2026 | 5/19/2026 | 
1 | 25-Mar-26 | 25-Mar-26 | * | * | Mar-26 | 6/17/2026 | 9/15/2026 | Tuesday | 3/18/2026 | 6/16/2026 | 90 | 
2 | 6-May-26 | 6-May-26 | * | * | May-26 | 9/16/2026 | 12/15/2026 | Tuesday | 4/15/2026 | 7/14/2026 | 90 | 
3 | 17-Jun-26 | 17-Jun-26 | * | * | Jun-26 | 12/16/2026 | 3/16/2027 | Tuesday | 5/20/2026 | 8/18/2026 | 90 | 
4 | 29-Jul-26 | 29-Jul-26 | * | * | Jul-26 | 3/17/2027 | 6/15/2027 | Tuesday | 6/17/2026 | 9/15/2026 | 90 | * | * | * | * | * | * | * | 2026/3/18 | 
5 | 16-Sep-26 | 16-Sep-26 | * | * | Sep-26 | 6/16/2027 | 9/14/2027 | Tuesday | 7/15/2026 | 10/20/2026 | 90 | * | * | * | * | * | * | * | 2026/6/17 | 
6 | 4-Nov-26 | 4-Nov-26 | * | * | Nov-26 | 9/15/2027 | 12/14/2027 | Tuesday | 8/19/2026 | 11/17/2026 | 90 | 
7 | 23-Dec-26 | 23-Dec-26 | * | * | Dec-26 | 12/15/2027 | 3/14/2028 | Tuesday | 9/16/2026 | 12/15/2026 | 90 | 15-Apr | * | * | #VALUE! | 
8 | 27-Jan-27 | 27-Jan-27 | * | * | Jan-27 | * | 3/20/1900 | Tuesday | 10/21/2026 | 1/19/2027 | 80 | * | * | * | #VALUE! | 
9 | 17-Mar-27 | 17-Mar-27 | * | * | Mar-27 | 2022/9/21 | 12/20/2022 | Tuesday | 11/18/2026 | 2/16/2027 | 90 | * | * | * | 14 | 
10 | 28-Apr-27 | 28-Apr-27 | * | * | Apr-27 | 2023/3/15 | 6/20/2023 | Tuesday | 12/16/2026 | 3/16/2027 | 97 | 
11 | 16-Jun-27 | 16-Jun-27 | * | * | Jun-27 | 2024/6/19 | 9/17/2024 | Tuesday | * | * | 90 | 
12 | 28-Jul-27 | 28-Jul-27 | * | * | Jul-27 | 2025/6/18 | 9/16/2025 | Tuesday | * | * | 90 | 
13 | 15-Sep-27 | 15-Sep-27 | * | * | Sep-27 | 2024/3/20 | 6/18/2024 | Tuesday | * | * | 90 | 
14 | 27-Oct-27 | 27-Oct-27 | * | * | Oct-27 | 2022/12/21 | 3/14/2023 | Tuesday | * | * | 83 | * | * | * | #VALUE! | 
15 | 15-Dec-27 | 15-Dec-27 | * | * | Dec-27 | * | * | * | * | 
39 | * | Meeting Impact on 3M outright | * | * | * | * | * | * | * | * | * | * | * | * | * | * | 
40 | * | Out/Meet | 11-Feb-26 | 25-Mar-26 | 6-May-26 | 17-Jun-26 | 29-Jul-26 | 16-Sep-26 | 4-Nov-26 | 23-Dec-26 | 27-Jan-27 | 17-Mar-27 | 28-Apr-27 | 16-Jun-27 | 28-Jul-27 | 15-Sep-27 | 
41 | * | Mar-26 | 1 | 0.9222222222 | 0.4555555556 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
42 | * | Jun-26 | 1 | 1 | 1 | 1 | 0.5333333333 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
43 | * | Sep-26 | 1 | 1 | 1 | 1 | 1 | 1 | 0.4555555556 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
44 | * | Dec-26 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0.9222222222 | 0.5333333333 | 0 | 0 | 0 | 0 | 0 | 
45 | * | Mar-27 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0.5333333333 | 0 | 0 | 0 | 
46 | * | Jun-27 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0.5333333333 | 0 | 
47 | * | Sep-27 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 
48 | * | Dec-27 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 
49 | * | Meeting Impact on 3M Spread | * | * | * | * | * | * | * | * | * | * | * | * | * | * | 
50 | * | Out/Meet | 11-Feb-26 | 25-Mar-26 | 6-May-26 | 17-Jun-26 | 29-Jul-26 | 16-Sep-26 | 4-Nov-26 | 23-Dec-26 | 27-Jan-27 | 17-Mar-27 | 28-Apr-27 | 16-Jun-27 | 28-Jul-27 | 15-Sep-27 | 
51 | * | Mar-26/Jun-26 | 0 | 0.0777777778 | 0.5444444444 | 1 | 0.5333333333 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
52 | * | Jun-26/Sep-26 | 0 | 0 | 0 | 0 | 0.4666666667 | 1 | 0.4555555556 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
53 | * | Sep-26/Dec-26 | 0 | 0 | 0 | 0 | 0 | 0 | 0.5444444444 | 0.9222222222 | 0.5333333333 | 0 | 0 | 0 | 0 | 0 | 
54 | * | Dec-26/Mar-27 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0777777778 | 0.4666666667 | 1 | 0.5333333333 | 0 | 0 | 0 | 
55 | * | Mar-27/Jun-27 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.4666666667 | 1 | 0.5333333333 | 0 | 
56 | * | Jun-27/Sep-27 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.4666666667 | 1 | 
57 | * | Sep-27/Dec-27 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
58 | * | Meeting Impact on 3M Fly | * | * | * | * | * | * | * | * | * | * | * | * | * | * | 
59 | * | Out/Meet | 11-Feb-26 | 25-Mar-26 | 6-May-26 | 17-Jun-26 | 29-Jul-26 | 16-Sep-26 | 4-Nov-26 | 23-Dec-26 | 27-Jan-27 | 17-Mar-27 | 28-Apr-27 | 16-Jun-27 | 28-Jul-27 | 15-Sep-27 | 
60 | * | Mar-26/Jun-26/Sep-26 | 0 | 0.0777777778 | 0.5444444444 | 1 | 0.0666666667 | -1 | -0.4555555556 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
61 | * | Jun-26/Sep-26/Dec-26 | 0 | 0 | 0 | 0 | 0.4666666667 | 1 | -0.0888888889 | -0.9222222222 | -0.5333333333 | 0 | 0 | 0 | 0 | 0 | 
62 | * | Sep-26/Dec-26/Mar-27 | 0 | 0 | 0 | 0 | 0 | 0 | 0.5444444444 | 0.8444444444 | 0.0666666667 | -1 | -0.5333333333 | 0 | 0 | 0 | 
63 | * | Dec-26/Mar-27/Jun-27 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0777777778 | 0.4666666667 | 1 | 0.0666666667 | -1 | -0.5333333333 | 0 | 
64 | * | Mar-27/Jun-27/Sep-27 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.4666666667 | 1 | 0.0666666667 | -1 | 
65 | * | Jun-27/Sep-27/Dec-27 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.4666666667 | 1 | 
66 | * | Meeting Impact on Serial Outright | * | * | * | * | * | * | * | * | * | * | * | * | * | * | 
67 | * | Out/Meet | 11-Feb-26 | 25-Mar-26 | 6-May-26 | 17-Jun-26 | 29-Jul-26 | 16-Sep-26 | 4-Nov-26 | 23-Dec-26 | 27-Jan-27 | 17-Mar-27 | 28-Apr-27 | 16-Jun-27 | 28-Jul-27 | 15-Sep-27 | 
68 | * | Feb-26 | 1 | 0.6111111111 | 0.1444444444 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
69 | * | Mar-26 | 1 | 0.9222222222 | 0.4555555556 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
70 | * | Apr-26 | 1 | 1 | 0.7666666667 | 0.3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
71 | * | May-26 | 1 | 1 | 1 | 0.6888888889 | 0.2222222222 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
72 | * | Jun-26 | 1 | 1 | 1 | 1 | 0.5333333333 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
73 | * | Jul-26 | 1 | 1 | 1 | 1 | 0.8556701031 | 0.3505154639 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
74 | * | Aug-26 | 1 | 1 | 1 | 1 | 1 | 0.6888888889 | 0.1444444444 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
75 | * | Sep-26 | 1 | 1 | 1 | 1 | 1 | 1 | 0.4555555556 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
76 | * | Oct-26 | 1 | 1 | 1 | 1 | 1 | 1 | 0.8444444444 | 0.3 | 0 | 0 | 0 | 0 | 0 | 0 | 
77 | * | Nov-26 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0.6111111111 | 0.2222222222 | 0 | 0 | 0 | 0 | 0 | 
78 | * | Dec-26 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0.9222222222 | 0.5333333333 | 0 | 0 | 0 | 0 | 0 | 
79 | * | Meeting Impact on Serial Spread | * | * | * | * | * | * | * | * | * | * | * | * | * | * | 
80 | * | Out/Meet | 11-Feb-26 | 25-Mar-26 | 6-May-26 | 17-Jun-26 | 29-Jul-26 | 16-Sep-26 | 4-Nov-26 | 23-Dec-26 | 27-Jan-27 | 17-Mar-27 | 28-Apr-27 | 16-Jun-27 | 28-Jul-27 | 15-Sep-27 | 
81 | * | Feb-26/Mar-26 | 0 | 0.3111111111 | 0.3111111111 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
82 | * | Mar-26/Apr-26 | 0 | 0.0777777778 | 0.3111111111 | 0.3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
83 | * | Apr-26/May-26 | 0 | 0 | 0.2333333333 | 0.3888888889 | 0.2222222222 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
84 | * | May-26/Jun-26 | 0 | 0 | 0 | 0.3111111111 | 0.3111111111 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
85 | * | Jun-26/Jul-26 | 0 | 0 | 0 | 0 | 0.3223367698 | 0.3505154639 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
86 | * | Jul-26/Aug-26 | 0 | 0 | 0 | 0 | 0.1443298969 | 0.338373425 | 0.1444444444 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
87 | * | Aug-26/Sep-26 | 0 | 0 | 0 | 0 | 0 | 0.3111111111 | 0.3111111111 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
88 | * | Sep-26/Oct-26 | 0 | 0 | 0 | 0 | 0 | 0 | 0.3888888889 | 0.3 | 0 | 0 | 0 | 0 | 0 | 0 | 
89 | * | Oct-26/Nov-26 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1555555556 | 0.3111111111 | 0.2222222222 | 0 | 0 | 0 | 0 | 0 | 
90 | * | Nov-26/Dec-26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.3111111111 | 0.3111111111 | 0 | 0 | 0 | 0 | 0 | 
91 | * | Meeting Impact on Serial Fly | * | * | * | * | * | * | * | * | * | * | * | * | * | * | 
92 | * | Out/Meet | 11-Feb-26 | 25-Mar-26 | 6-May-26 | 17-Jun-26 | 29-Jul-26 | 16-Sep-26 | 4-Nov-26 | 23-Dec-26 | 27-Jan-27 | 17-Mar-27 | 28-Apr-27 | 16-Jun-27 | 28-Jul-27 | 15-Sep-27 | 
93 | * | Feb-26/Mar-26/Apr-26 | 0 | 0.2333333333 | 0 | -0.3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
94 | * | Mar-26/Apr-26/May-26 | 0 | 0.0777777778 | 0.0777777778 | -0.0888888889 | -0.2222222222 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
95 | * | Apr-26/May-26/Jun-26 | 0 | 0 | 0.2333333333 | 0.0777777778 | -0.0888888889 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
96 | * | May-26/Jun-26/Jul-26 | 0 | 0 | 0 | 0.3111111111 | -0.0112256586 | -0.3505154639 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
97 | * | Jun-26/Jul-26/Aug-26 | 0 | 0 | 0 | 0 | 0.1780068729 | 0.0121420389 | -0.1444444444 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
98 | * | Jul-26/Aug-26/Sep-26 | 0 | 0 | 0 | 0 | 0.1443298969 | 0.0272623139 | -0.1666666667 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 
99 | * | Aug-26/Sep-26/Oct-26 | 0 | 0 | 0 | 0 | 0 | 0.3111111111 | -0.0777777778 | -0.3 | 0 | 0 | 0 | 0 | 0 | 0 | 
100 | * | Sep-26/Oct-26/Nov-26 | 0 | 0 | 0 | 0 | 0 | 0 | 0.2333333333 | -0.0111111111 | -0.2222222222 | 0 | 0 | 0 | 0 | 0 | 
101 | * | Oct-26/Nov-26/Dec-26 | 0 | 0 | 0 | 0 | 0 | 0 | 0.1555555556 | 0 | -0.0888888889 | 0 | 0 | 0 | 0 | 0 | 
"""

def parse_embedded_data(text_data):
    """Parses the raw string text into a dictionary of DataFrames."""
    lines = text_data.strip().split('\n')
    data_blocks = {}
    current_key = None
    headers = []
    data_rows = []
    
    block_pattern = re.compile(r"Meeting Impact on (.*)")
    
    for line in lines:
        parts = [p.strip() for p in line.split('|')]
        
        if len(parts) > 2 and "Meeting Impact on" in parts[2]:
            if current_key and headers and data_rows:
                df = pd.DataFrame(data_rows, columns=headers)
                data_blocks[current_key] = df
            current_key = parts[2].strip()
            headers = []
            data_rows = []
            continue
            
        if current_key and len(parts) > 2 and parts[2] == "Out/Meet":
            headers = [h for h in parts[2:] if h]
            continue
            
        if current_key and headers:
            if parts[0].isdigit():
                row_label = parts[2]
                row_values = [v for v in parts[3:]]
                full_row = [row_label] + row_values[:len(headers)-1]
                while len(full_row) < len(headers):
                    full_row.append('')
                data_rows.append(full_row)
    
    if current_key and headers and data_rows:
        df = pd.DataFrame(data_rows, columns=headers)
        data_blocks[current_key] = df
        
    return data_blocks

def clean_and_convert_df(df):
    df = df.set_index(df.columns[0])
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def generate_mock_prices(instruments):
    """Generates random Settle and WWAP prices for demo purposes."""
    np.random.seed(42)
    # Base prices around 97-99
    settles = np.random.uniform(97.0, 99.5, size=len(instruments))
    # WWAP slightly different
    wwaps = settles + np.random.uniform(-0.05, 0.05, size=len(instruments))
    return pd.DataFrame({
        "Settle": settles,
        "WWAP": wwaps
    }, index=instruments)

# --- Main Application ---

def main():
    st.title("📊 ECB Meeting Impact Dashboard")
    st.markdown("### Scenario Analysis: Outrights, Spreads, and Flies")

    # 1. Load Data
    raw_blocks = parse_embedded_data(RAW_DATA_STRING)
    impact_data = {k: clean_and_convert_df(v) for k, v in raw_blocks.items()}
    
    # Identify meeting dates (columns)
    # Assuming all impact tables share the same meeting columns (Meeting Dates)
    sample_df = list(impact_data.values())[0]
    meeting_dates = sample_df.columns.tolist()
    
    # 2. Session State for Cases
    # We initialize 5 cases with default 0 hikes, or some presets
    if 'cases' not in st.session_state:
        st.session_state.cases = {
            "Case 1": {d: 0.0 for d in meeting_dates}, # No Hikes
            "Case 2": {d: 0.0 for d in meeting_dates}, # Custom
            "Case 3": {d: 0.0 for d in meeting_dates}, # Custom
            "Case 4": {d: 0.0 for d in meeting_dates}, # Custom
            "Case 5": {d: 0.0 for d in meeting_dates}, # Custom
        }
        # Set a default scenario for Case 2 (e.g., 25bps in Feb)
        st.session_state.cases["Case 2"]["11-Feb-26"] = 25.0

    # 3. Sidebar - Scenario Manager
    st.sidebar.header("⚙️ Scenario Manager")
    selected_tab_name = st.sidebar.selectbox(
        "Select Data View", 
        ["3M Outright", "3M Spread", "3M Fly", "Serial Outright", "Serial Spread", "Serial Fly"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Edit Cases")
    
    # Select which Case to edit
    case_to_edit = st.sidebar.selectbox("Select Case to Edit", list(st.session_state.cases.keys()))
    
    st.sidebar.markdown(f"**Define Hikes/Cuts for {case_to_edit}:**")
    
    # Create inputs for each meeting date
    edited_case = {}
    cols = st.sidebar.columns(2) # 2 columns of inputs to save space
    for i, date in enumerate(meeting_dates):
        with cols[i % 2]:
            # Use number input for precise bps entry
            val = st.number_input(
                f"{date}", 
                value=float(st.session_state.cases[case_to_edit][date]), 
                key=f"input_{case_to_edit}_{date}",
                step=5.0, format="%0.1f"
            )
            edited_case[date] = val
    
    # Save button
    if st.sidebar.button("Save Case Configuration"):
        st.session_state.cases[case_to_edit] = edited_case
        st.sidebar.success(f"{case_to_edit} updated!")
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Percent View")
    case_for_percent = st.sidebar.selectbox(
        "Case for % Impact",
        list(st.session_state.cases.keys()),
        index=list(st.session_state.cases.keys()).index("Case 2") if "Case 2" in st.session_state.cases else 0,
        key="case_for_percent"
    )

    # 4. Calculation Logic
    # Function to calculate impact for a specific case
    def calculate_case_impact(impact_df, case_bps):
        """
        impact_df: Rows = Instruments, Cols = Meetings, Values = Impact Factors
        case_bps: Dict {Meeting: bps_change}
        Returns: Series {Instrument: total_price_change}
        """
        # Create a series of bps aligned to columns
        bps_series = pd.Series(case_bps)
        
        # Multiply Impact Factors by BPS (Broadcasting)
        # result[row, col] = impact[row, col] * bps[col]
        weighted_impact = impact_df.multiply(bps_series, axis=1)
        
        # Sum across columns to get total shift per instrument
        total_shift = weighted_impact.sum(axis=1)
        return total_shift

    # 5. Main Display Area
    
    # Get the correct impact table based on selection
    # Map selectbox value to data key
    data_key_map = {
        "3M Outright": "Meeting Impact on 3M outright",
        "3M Spread": "Meeting Impact on 3M Spread",
        "3M Fly": "Meeting Impact on 3M Fly",
        "Serial Outright": "Meeting Impact on Serial Outright",
        "Serial Spread": "Meeting Impact on Serial Spread",
        "Serial Fly": "Meeting Impact on Serial Fly"
    }
    
    current_impact_df = impact_data.get(data_key_map[selected_tab_name])
    
    if current_impact_df is None:
        st.error("Data not found for selection.")
        return

    # Generate Mock Market Data for this view
    instruments = current_impact_df.index.tolist()
    market_df = generate_mock_prices(instruments)
    
    # Calculate shifts for all 5 cases
    case_results = {}
    for case_name, case_bps in st.session_state.cases.items():
        case_results[case_name] = calculate_case_impact(current_impact_df, case_bps)

    # Combine into one master DataFrame for display
    # Columns: Settle, WWAP, Case 1, Case 2, Case 3, Case 4, Case 5
    display_df = market_df.copy()
    
    for case_name in ["Case 1", "Case 2", "Case 3", "Case 4", "Case 5"]:
        display_df[case_name] = case_results[case_name]

    # 6. Filtering
    col_search, col_filter_val = st.columns([2, 1])
    with col_search:
        search_term = st.text_input("Filter Instruments (e.g., Mar-26)", "")
    with col_filter_val:
        min_impact = st.number_input("Min Impact (bps)", value=None)

    if search_term:
        display_df = display_df.filter(regex=re.compile(search_term, re.IGNORECASE), axis=0)
    
    # 7. Styling and Output
    def highlight_cases(val):
        """Color negative shifts red, positive green."""
        if isinstance(val, (int, float)):
            if val < 0:
                return 'color: red; font-weight: bold;'
            elif val > 0:
                return 'color: green; font-weight: bold;'
        return ''

    st.subheader(f"Market Impact Analysis: {selected_tab_name}")

    tab_impact, tab_percent = st.tabs(["Impact (bps)", "Meeting % Impact"])

    with tab_impact:
        # Format numbers
        # Settle/WWAP usually 4 decimals (e.g., 98.5000)
        # Cases are bps (e.g., 12.5)
        formatted_df = display_df.style.format({
            "Settle": "{:.4f}",
            "WWAP": "{:.4f}",
            "Case 1": "{:.2f}",
            "Case 2": "{:.2f}",
            "Case 3": "{:.2f}",
            "Case 4": "{:.2f}",
            "Case 5": "{:.2f}",
        }).map(highlight_cases, subset=["Case 1", "Case 2", "Case 3", "Case 4", "Case 5"])

        st.dataframe(formatted_df, use_container_width=True, height=500)

        # 8. Charts
        st.subheader("Visual Comparison")
        chart_case_select = st.multiselect(
            "Select Cases to Plot", 
            ["Case 1", "Case 2", "Case 3", "Case 4", "Case 5"], 
            default=["Case 1", "Case 2"]
        )

        if chart_case_select:
            chart_data = display_df[chart_case_select]
            st.bar_chart(chart_data, use_container_width=True)

    with tab_percent:
        st.caption("Toggle between actual bps impact (case-weighted) and percent impact from the source table.")

        view_mode = st.radio(
            "Meeting impact view",
            ["Actual impact (bps multiplied)", "Percent impact (from excel)"],
            horizontal=True
        )

        selected_case_bps = st.session_state.cases[case_for_percent]
        bps_series = pd.Series(selected_case_bps)

        weighted_meeting_impact = current_impact_df.multiply(bps_series, axis=1)
        total_impact = weighted_meeting_impact.sum(axis=1)

        if view_mode == "Actual impact (bps multiplied)":
            meeting_view_df = weighted_meeting_impact.copy()
            meeting_view_df.insert(0, "Actual Impact (bps)", total_impact)

            formatted_view = meeting_view_df.style.format(
                {"Actual Impact (bps)": "{:.2f}"}
            ).format("{:.2f}", subset=meeting_view_df.columns[1:]).map(
                highlight_cases, subset=["Actual Impact (bps)"]
            )
        else:
            percent_view_df = current_impact_df.copy() * 100.0
            percent_view_df.insert(0, "Actual Impact (bps)", total_impact)

            formatted_view = percent_view_df.style.format(
                {"Actual Impact (bps)": "{:.2f}"}
            ).format("{:.2f}%", subset=percent_view_df.columns[1:]).map(
                highlight_cases, subset=["Actual Impact (bps)"]
            )

            heatmap_df = percent_view_df.drop(columns=["Actual Impact (bps)"])
            heatmap_styled = heatmap_df.style.format("{:.1f}%").background_gradient(
                cmap="RdYlGn", axis=None, vmin=-100, vmax=100
            )

        st.dataframe(formatted_view, use_container_width=True, height=500)

        if view_mode == "Percent impact (from excel)":
            st.subheader("Meeting Impact Heatmap (%)")
            st.dataframe(heatmap_styled, use_container_width=True, height=500)

if __name__ == "__main__":
    main()