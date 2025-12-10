import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import timedelta

# ----------------------
# Config
# ----------------------
DATA_PATH = "data/df_clean.csv"   # <-- put your CSV here (see repo layout below)

st.set_page_config(page_title="50NB Mtrol — VFM vs Mtrol Flow", layout="wide")

st.title("50NB Mtrol — VFM Flow vs Mtrol Flow (elapsed seconds)")

# ----------------------
# Load data with cache
# ----------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    # compute elapsed seconds from start
    start = df["Timestamp"].iloc[0]
    df["t_seconds"] = (df["Timestamp"] - start).dt.total_seconds()
    return df

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Data file not found at `{DATA_PATH}`. Put your CSV at that path (see README instructions).")
    st.stop()

# ----------------------
# Columns detection
# ----------------------
# Accept a few possible column name variants (based on your datasets)
possible_vfm_cols = ["VFM Flow Rate (kg/h)", "VFM_flow_kgph", "VFM_flow_kgph", "VFM Flow (kg per h)", "VFM Flow (kg/h)"]
possible_mtrol_cols = ["Mtrol revised flow", "Mtrol_flow_kgph", "Mtrol revised flow", "Mtrol Flow (kg/h)"]

vfm_col = next((c for c in possible_vfm_cols if c in df.columns), None)
mtrol_col = next((c for c in possible_mtrol_cols if c in df.columns), None)

if vfm_col is None or mtrol_col is None:
    st.error("Couldn't find expected flow columns in the CSV. Columns found:\n\n" + ", ".join(df.columns))
    st.stop()

# ----------------------
# Sidebar controls
# ----------------------
st.sidebar.header("View controls")

# time range slider (in seconds)
t_min, t_max = int(df["t_seconds"].min()), int(df["t_seconds"].max())
default_range = (t_min, t_max)
sel_range = st.sidebar.slider("Elapsed time (seconds) range", min_value=t_min, max_value=t_max, value=default_range, step=1)

# smoothing
smooth_window = st.sidebar.selectbox("Rolling smoothing (seconds, 0 = no smoothing)", options=[0,5,10,30,60,120,300], index=0)
# convert smoothing seconds to rows using median dt estimate
median_dt = float(df["t_seconds"].diff().median() or 1.0)

# show difference trace
show_diff = st.sidebar.checkbox("Show (VFM - Mtrol) difference", value=True)

# downsample for plotting if huge
max_points = st.sidebar.slider("Max points to display (downsamples for performance)", 2000, 200000, 20000, step=500)
st.sidebar.markdown("Tip: Increase smoothing and reduce max points for faster interaction.")

# ----------------------
# Filter data by selected time range
# ----------------------
mask = (df["t_seconds"] >= sel_range[0]) & (df["t_seconds"] <= sel_range[1])
plot_df = df.loc[mask].copy()

if plot_df.empty:
    st.warning("No data in selected time range.")
    st.stop()

# ----------------------
# Apply smoothing if requested
# ----------------------
if smooth_window and smooth_window > 0:
    # convert smoothing seconds to approximate window in rows
    win = max(1, int(round(smooth_window / median_dt)))
    plot_df[f"{vfm_col}_sm"] = plot_df[vfm_col].rolling(window=win, min_periods=1, center=True).mean()
    plot_df[f"{mtrol_col}_sm"] = plot_df[mtrol_col].rolling(window=win, min_periods=1, center=True).mean()
    vfm_plot_col = f"{vfm_col}_sm"
    mtrol_plot_col = f"{mtrol_col}_sm"
else:
    vfm_plot_col = vfm_col
    mtrol_plot_col = mtrol_col

# ----------------------
# Downsample for performance (uniform sampling)
# ----------------------
if len(plot_df) > max_points:
    frac = max_points / len(plot_df)
    plot_df = plot_df.sample(n=max_points, random_state=42).sort_values("t_seconds")

# ----------------------
# Build interactive Plotly figure
# ----------------------
fig = go.Figure()

# VFM trace (blue)
fig.add_trace(go.Scatter(
    x=plot_df["t_seconds"],
    y=plot_df[vfm_plot_col],
    mode="lines",
    name="VFM Flow (kg/h)",
    line=dict(color="#CEDCCA", width=1.5),
    hovertemplate="t: %{x:.0f}s<br>VFM: %{y:.2f} kg/h<br>Time: %{customdata}",
    customdata=plot_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
))

# Mtrol trace (red)
fig.add_trace(go.Scatter(
    x=plot_df["t_seconds"],
    y=plot_df[mtrol_plot_col],
    mode="lines",
    name="Mtrol Revised Flow (kg/h)",
    line=dict(color="red", width=1.5),
    hovertemplate="t: %{x:.0f}s<br>Mtrol: %{y:.2f} kg/h<br>Time: %{customdata}",
    customdata=plot_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
))

# Difference trace (green) optional
if show_diff:
    diff = plot_df[vfm_plot_col] - plot_df[mtrol_plot_col]
    fig.add_trace(go.Scatter(
        x=plot_df["t_seconds"],
        y=diff,
        mode="lines",
        name="VFM - Mtrol (kg/h)",
        line=dict(color="green", width=1),
        yaxis="y2",
        hovertemplate="t: %{x:.0f}s<br>Diff: %{y:.2f} kg/h<br>Time: %{customdata}",
        customdata=plot_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    ))

# Layout: add a second y-axis for difference
yaxis2 = dict(
    title="Diff (kg/h)",
    overlaying="y",
    side="right",
    showgrid=False,
    zeroline=True,
    zerolinecolor="gray",
    zerolinewidth=1,
)

fig.update_layout(
    xaxis=dict(title="Elapsed time (seconds)"),
    yaxis=dict(title="Flow (kg/h)"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=50, r=60, t=80, b=50),
    height=600
)

if show_diff:
    fig.update_layout(yaxis2=yaxis2)

# Improve x-axis tick formatting: show human-friendly step if needed
# (Plotly will handle numeric ticks; hover shows real timestamp)
fig.update_xaxes(tickformat="d", nticks=10)

# ----------------------
# Show stats above the chart
# ----------------------
col1, col2, col3 = st.columns(3)
col1.metric("Data points (shown)", f"{len(plot_df):,}")
col2.metric("Time span", f"{timedelta(seconds=int(plot_df['t_seconds'].max() - plot_df['t_seconds'].min()))}")
mean_vfm = plot_df[vfm_plot_col].mean()
mean_mtrol = plot_df[mtrol_plot_col].mean()
col3.metric("Mean flows (VFM / Mtrol)", f"{mean_vfm:.1f} / {mean_mtrol:.1f} kg/h")

# ----------------------
# Display figure
# ----------------------
st.plotly_chart(fig, use_container_width=True)

# ----------------------
# Optional: show raw table (small)
# ----------------------
if st.checkbox("Show sample data (first 200 rows)"):
    st.dataframe(plot_df[["Timestamp","t_seconds", vfm_col, mtrol_col]].head(200))

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.caption("Colors: blue = VFM, red = Mtrol. X axis is elapsed seconds since start of dataset.")
