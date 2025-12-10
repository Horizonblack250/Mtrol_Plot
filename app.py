import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import timedelta, datetime

# ----------------------------
# CONFIG / PATH
# ----------------------------
DATA_PATH = "data/df_clean.csv"   # put your CSV here
EPS_VFM = 1e-6

# set page config before other Streamlit calls
st.set_page_config(page_title="Mtrol — Pressures & Flows (interactive)", layout="wide")
st.title("Mtrol — Pressures & Flows (interactive)")

# ----------------------------
# Load data (cached)
# ----------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    # detect and standardize column names (flexible)
    possible_vfm = ["VFM Flow Rate (kg/h)", "VFM_flow_kgph", "VFM_flow_kg/h", "VFM_flow", "VFM Flow (kg per h)", "VFM Flow (kg/h)"]
    possible_mtrol = ["Mtrol revised flow", "Mtrol_flow_kgph", "Mtrol Flow (kg/h)", "Mtrol_flow"]
    possible_p1 = ["Inlet Pressure P1 (barg)", "P1", "Inlet Pressure P1 (bg)"]
    possible_p2 = ["Outlet Pressure P2 (barg)", "P2", "Outlet Pressure P2 (bg)"]
    possible_p2sp = ["Setpoint P2 (bg)", "P2_SP", "Setpoint P2 (barg)"]

    vfm = next((c for c in possible_vfm if c in df.columns), None)
    mtrol = next((c for c in possible_mtrol if c in df.columns), None)
    p1 = next((c for c in possible_p1 if c in df.columns), None)
    p2 = next((c for c in possible_p2 if c in df.columns), None)
    p2sp = next((c for c in possible_p2sp if c in df.columns), None)

    if vfm is None or mtrol is None or p2 is None or p1 is None:
        raise ValueError("Required columns missing. Found columns: " + ", ".join(df.columns))

    df = df.rename(columns={vfm: "VFM_flow_kgph", mtrol: "Mtrol_flow_kgph", p1: "P1", p2: "P2"})
    if p2sp:
        df = df.rename(columns={p2sp: "P2_SP"})
    else:
        df["P2_SP"] = np.nan

    # elapsed seconds and dt seconds between rows
    df["t_seconds"] = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds()
    df["dt_s"] = df["Timestamp"].diff().dt.total_seconds().fillna(method="bfill")
    df["dt_s"] = df["dt_s"].fillna(df["dt_s"].median())

    # error percentage (guard)
    df["Error_pct"] = np.where(df["VFM_flow_kgph"].abs() > EPS_VFM,
                               100.0 * (df["VFM_flow_kgph"] - df["Mtrol_flow_kgph"]) / df["VFM_flow_kgph"],
                               np.nan)

    return df

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Controls")

# Convert pandas Timestamps to native python datetimes for the slider
min_ts = df["Timestamp"].min().to_pydatetime()
max_ts = df["Timestamp"].max().to_pydatetime()

sel_start, sel_end = st.sidebar.slider(
    "Select time window (datetime)",
    min_value=min_ts,
    max_value=max_ts,
    value=(min_ts, max_ts),
    format="YYYY-MM-DD HH:mm:ss"
)

# smoothing seconds
smooth_seconds = st.sidebar.selectbox("Smoothing window (seconds, 0 = none)", [0, 5, 10, 30, 60, 120, 300], index=0)
median_dt = float(df["dt_s"].median() or 1.0)
smooth_rows = max(1, int(round(smooth_seconds / median_dt))) if smooth_seconds > 0 else 0

# toggle error % trace
show_error_pct = st.sidebar.checkbox("Show Error % (VFM - Mtrol) on right axis (bottom plot)", value=True)

# downsample parameter
max_points = st.sidebar.slider("Max points to plot (for performance)", 1000, 200000, 25000, step=500)

# thresholds (barg) for P2 in-spec KPI: 0.1 to 1.0 barg step 0.1
p2_thresholds = [round(x, 2) for x in np.arange(0.1, 1.01, 0.1)]
p2_thresholds += [2.0, 5.0]

# ----------------------------
# Filter to selected datetime window
# ----------------------------
# sel_start and sel_end are python datetimes — use directly
mask = (df["Timestamp"] >= pd.to_datetime(sel_start)) & (df["Timestamp"] <= pd.to_datetime(sel_end))
window_df = df.loc[mask].copy()
if window_df.empty:
    st.warning("No data in selected time window.")
    st.stop()

# optional smoothing (centered rolling)
if smooth_rows and smooth_rows > 1:
    window_df = window_df.sort_values("Timestamp")
    window_df["VFM_sm"] = window_df["VFM_flow_kgph"].rolling(window=smooth_rows, min_periods=1, center=True).mean()
    window_df["Mtrol_sm"] = window_df["Mtrol_flow_kgph"].rolling(window=smooth_rows, min_periods=1, center=True).mean()
    window_df["Err_sm"] = window_df["Error_pct"].rolling(window=smooth_rows, min_periods=1, center=True).mean()
    vfm_col_plot = "VFM_sm"
    mtrol_col_plot = "Mtrol_sm"
    err_col_plot = "Err_sm"
else:
    vfm_col_plot = "VFM_flow_kgph"
    mtrol_col_plot = "Mtrol_flow_kgph"
    err_col_plot = "Error_pct"

# downsample uniformly if too many points (safe)
if len(window_df) > max_points:
    window_df = window_df.sample(n=max_points, random_state=42).sort_values("Timestamp")

# ----------------------------
# Integration function to compute total mass (kg) over the window using timestamps (trapezoidal)
# ----------------------------
def integrate_mass_kg(df_slice, flow_col):
    if len(df_slice) < 2:
        return 0.0
    ts = pd.to_datetime(df_slice["Timestamp"])
    flows = df_slice[flow_col].values
    total = 0.0
    for i in range(len(ts) - 1):
        dt_hours = (ts.iloc[i+1] - ts.iloc[i]).total_seconds() / 3600.0
        avg_flow = 0.5 * (flows[i] + flows[i+1])
        total += avg_flow * dt_hours
    return total

vfm_total_kg = integrate_mass_kg(window_df.sort_values("Timestamp"), vfm_col_plot)
mtrol_total_kg = integrate_mass_kg(window_df.sort_values("Timestamp"), mtrol_col_plot)

# ----------------------------
# P2 in-spec KPIs (absolute barg thresholds) for the window
# ----------------------------
kpi_table = []
if window_df["P2_SP"].notna().sum() > 0:
    abs_diff = (window_df["P2"] - window_df["P2_SP"]).abs()
    total_rows_with_sp = (~window_df["P2_SP"].isna()).sum()
    for thr in p2_thresholds:
        pct_in_spec = 100.0 * (abs_diff <= thr).sum() / total_rows_with_sp if total_rows_with_sp > 0 else np.nan
        kpi_table.append((thr, pct_in_spec))
else:
    kpi_table = []

# ----------------------------
# Build Plotly subplot (two rows, shared xaxis)
# ----------------------------
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.35, 0.65],
                    vertical_spacing=0.06,
                    specs=[[{"secondary_y": False}], [{"secondary_y": True}]])

# Top plot: P1, P2, P2_SP (pressures)
fig.add_trace(go.Scatter(
    x=window_df["Timestamp"],
    y=window_df["P1"],
    mode="lines",
    name="P1 (barg)",
    line=dict(color="#2ECC40", dash="dash"),
    hovertemplate="%{x}<br>P1: %{y:.3f} barg"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=window_df["Timestamp"],
    y=window_df["P2"],
    mode="lines",
    name="P2 (barg)",
    line=dict(color="#800080", dash="dot"),
    hovertemplate="%{x}<br>P2: %{y:.3f} barg"
), row=1, col=1)

if window_df["P2_SP"].notna().any():
    fig.add_trace(go.Scatter(
        x=window_df["Timestamp"],
        y=window_df["P2_SP"],
        mode="lines",
        name="P2_SP (barg)",
        line=dict(color="#AAAAAA", dash="dashdot"),
        hovertemplate="%{x}<br>P2_SP: %{y:.3f} barg"
    ), row=1, col=1)

fig.update_yaxes(title_text="Pressure (barg)", row=1, col=1)

# Bottom plot: flows + optional error%
fig.add_trace(go.Scatter(
    x=window_df["Timestamp"],
    y=window_df[vfm_col_plot],
    mode="lines",
    name="VFM Flow (kg/h)",
    line=dict(color="#0B62FF", width=1.6),
    hovertemplate="%{x}<br>VFM: %{y:.2f} kg/h"
), row=2, col=1, secondary_y=False)

fig.add_trace(go.Scatter(
    x=window_df["Timestamp"],
    y=window_df[mtrol_col_plot],
    mode="lines",
    name="Mtrol Flow (kg/h)",
    line=dict(color="#FF4136", width=1.6),
    hovertemplate="%{x}<br>Mtrol: %{y:.2f} kg/h"
), row=2, col=1, secondary_y=False)

if show_error_pct:
    fig.add_trace(go.Scatter(
        x=window_df["Timestamp"],
        y=window_df[err_col_plot],
        mode="lines",
        name="Error % (VFM - Mtrol)",
        line=dict(color="#00CC96", width=1.2),
        hovertemplate="%{x}<br>Error: %{y:.2f}%"
    ), row=2, col=1, secondary_y=True)

fig.update_yaxes(title_text="Flow (kg/h)", row=2, col=1, secondary_y=False)
if show_error_pct:
    fig.update_yaxes(title_text="Error %", row=2, col=1, secondary_y=True)

fig.update_xaxes(title_text="Timestamp (datetime)", row=2, col=1)
fig.update_layout(height=800,
                  margin=dict(l=60, r=80, t=90, b=80),
                  legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
                  hovermode="x unified")

# ----------------------------
# Show KPIs and totals
# ----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Points shown", f"{len(window_df):,}")
k2.metric("Window duration", f"{timedelta(seconds=int((window_df['Timestamp'].max() - window_df['Timestamp'].min()).total_seconds()))}")
k3.metric("Total VFM (kg)", f"{vfm_total_kg:,.1f}")
k4.metric("Total Mtrol (kg)", f"{mtrol_total_kg:,.1f}")

# P2 in-spec table
st.markdown("### P2 in-spec fractions (absolute barg thresholds)")
if not kpi_table:
    st.info("No P2 setpoints (P2_SP) available in this window to evaluate.")
else:
    df_kpi = pd.DataFrame(kpi_table, columns=["threshold_barg", "pct_in_spec"])
    df_kpi["Threshold"] = df_kpi["threshold_barg"].apply(lambda x: f"±{x:.2f} barg")
    df_kpi = df_kpi[["Threshold", "pct_in_spec"]].rename(columns={"pct_in_spec": "% in spec"})
    st.dataframe(df_kpi.style.format({"% in spec": "{:.2f}"}), use_container_width=True)

# ----------------------------
# Display the plot
# ----------------------------
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Optional: show raw table
# ----------------------------
if st.checkbox("Show sample of window data (first 300 rows)"):
    show_cols = ["Timestamp", "P1", "P2", "P2_SP", "VFM_flow_kgph", "Mtrol_flow_kgph", "Error_pct"]
    st.dataframe(window_df[show_cols].head(300), use_container_width=True)

st.markdown("---")
st.caption("Top: pressures (P1, P2, P2_SP). Bottom: flows (VFM, Mtrol) and optional Error % on right. Use the datetime slider to choose a precise time window for analysis.")
