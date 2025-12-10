import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import timedelta

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "data/df_clean.csv"   # place your CSV here
EPS_VFM = 1e-6  # guard for division by zero when computing error %

st.set_page_config(page_title="50NB Mtrol — VFM vs Mtrol (interactive)", layout="wide")
st.title("50NB Mtrol — VFM vs Mtrol Flow (elapsed seconds) — interactive")

# ----------------------------
# Load data (cached)
# ----------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # normalize column names (common variants)
    df = df.rename(columns=lambda c: c.strip())
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    # detect common column names, raise if missing
    possible_vfm = ["VFM Flow Rate (kg/h)", "VFM_flow_kgph", "VFM Flow (kg per h)", "VFM_flow_kg/h", "VFM_flow"]
    possible_mtrol = ["Mtrol revised flow", "Mtrol_flow_kgph", "Mtrol revised flow", "Mtrol Flow (kg/h)", "Mtrol_flow"]
    possible_p1 = ["Inlet Pressure P1 (barg)", "P1", "Inlet Pressure P1 (bg)", "Inlet Pressure P1 (barg)"]
    possible_p2 = ["Outlet Pressure P2 (barg)", "P2", "Outlet Pressure P2 (bg)", "P2"]
    possible_p2sp = ["Setpoint P2 (bg)", "P2_SP", "Setpoint P2 (barg)", "P2_SP"]

    vfm = next((c for c in possible_vfm if c in df.columns), None)
    mtrol = next((c for c in possible_mtrol if c in df.columns), None)
    p1 = next((c for c in possible_p1 if c in df.columns), None)
    p2 = next((c for c in possible_p2 if c in df.columns), None)
    p2sp = next((c for c in possible_p2sp if c in df.columns), None)

    # require flows and pressures
    if vfm is None or mtrol is None or p2 is None:
        raise ValueError("Required columns not found. Found columns: " + ", ".join(df.columns))

    # standardize to convenient names
    df = df.rename(columns={vfm: "VFM_flow_kgph", mtrol: "Mtrol_flow_kgph",
                            p1: "P1", p2: "P2"})
    if p2sp:
        df = df.rename(columns={p2sp: "P2_SP"})
    else:
        # create P2_SP column of NaN if absent
        df["P2_SP"] = np.nan

    # compute elapsed seconds
    start = df["Timestamp"].iloc[0]
    df["t_seconds"] = (df["Timestamp"] - start).dt.total_seconds()

    # compute error pct guard
    df["Error_pct"] = np.where(df["VFM_flow_kgph"].abs() > EPS_VFM,
                               100.0 * (df["VFM_flow_kgph"] - df["Mtrol_flow_kgph"]) / df["VFM_flow_kgph"],
                               np.nan)

    # compute dt seconds between rows (for integration)
    df["dt_s"] = df["Timestamp"].diff().dt.total_seconds().fillna(method="bfill")
    median_dt = df["dt_s"].median()
    df["dt_s"] = df["dt_s"].fillna(median_dt)

    return df

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("View & analysis controls")

# time window slider (seconds)
t_min, t_max = int(df["t_seconds"].min()), int(df["t_seconds"].max())
default_range = (t_min, t_max)
sel_range = st.sidebar.slider("Elapsed time range (seconds)", min_value=t_min, max_value=t_max,
                              value=default_range, step=1, help="Select the time window to display and analyze (acts like zoom).")

# smoothing (seconds)
smooth_seconds = st.sidebar.selectbox("Smoothing window (seconds, 0 = no smoothing)",
                                      options=[0, 5, 10, 30, 60, 120, 300], index=0)
median_dt = float(df["dt_s"].median() or 1.0)
if smooth_seconds > 0:
    smooth_window_rows = max(1, int(round(smooth_seconds / median_dt)))
else:
    smooth_window_rows = 0

# toggle error % trace
show_error_pct = st.sidebar.checkbox("Show Error % (VFM - Mtrol) on right axis", value=True)

# toggle P1/P2 traces
show_pressures = st.sidebar.checkbox("Show P1 / P2 (pressure) traces", value=True)

# max points to plot
max_points = st.sidebar.slider("Max points to display (for performance)", min_value=1000, max_value=200000,
                               value=20000, step=500)

# thresholds for P2-in-spec KPIs: 0.1%..1.0% and also some larger buckets
kpi_small_percentiles = [0.001 * i for i in range(1, 11)]  # 0.1% to 1.0%
kpi_wide = [0.02, 0.05, 0.10, 1.00]  # 2%, 5%, 10%, 100%
# Combine
kpi_thresholds = kpi_small_percentiles + kpi_wide

# ----------------------------
# Filter data to selected range
# ----------------------------
mask = (df["t_seconds"] >= sel_range[0]) & (df["t_seconds"] <= sel_range[1])
plot_df = df.loc[mask].copy()
if plot_df.empty:
    st.warning("No data in selected range.")
    st.stop()

# apply smoothing if requested
if smooth_window_rows > 1:
    plot_df = plot_df.sort_values("t_seconds")
    plot_df["VFM_smooth"] = plot_df["VFM_flow_kgph"].rolling(window=smooth_window_rows, min_periods=1, center=True).mean()
    plot_df["Mtrol_smooth"] = plot_df["Mtrol_flow_kgph"].rolling(window=smooth_window_rows, min_periods=1, center=True).mean()
    plot_df["Error_smooth"] = plot_df["Error_pct"].rolling(window=smooth_window_rows, min_periods=1, center=True).mean()
    vfm_plot_col = "VFM_smooth"
    mtrol_plot_col = "Mtrol_smooth"
    err_plot_col = "Error_smooth"
else:
    vfm_plot_col = "VFM_flow_kgph"
    mtrol_plot_col = "Mtrol_flow_kgph"
    err_plot_col = "Error_pct"

# downsample uniformly if too many points
if len(plot_df) > max_points:
    plot_df = plot_df.sample(n=max_points, random_state=42).sort_values("t_seconds")

# ----------------------------
# Compute totals inside selected window (trapezoidal integration)
# mass (kg) = integral(flow_kgph * dt_hours)
def integrate_flow_kg(df_slice, flow_col):
    # if only one point, approximate as flow * dt_total
    if len(df_slice) <= 1:
        return 0.0
    # compute dt in hours between successive rows (use dt_s)
    dt_h = df_slice["dt_s"].values / 3600.0
    # approximate average flow between points using trapezoid: sum( (f[i]+f[i+1])/2 * dt )
    f = df_slice[flow_col].values
    # ensure same length: compute mid-sum for pairs
    # do pairwise trapezoid
    total = 0.0
    for i in range(len(f)-1):
        avg = 0.5 * (f[i] + f[i+1])
        total += avg * (dt_h[i+1])  # dt between i and i+1 - using dt_s shifted; simpler: use Timestamp
    # The above uses dt_h[1..], to be safe we will compute using timestamps instead:
    # fallback calculation using timestamps (more robust)
    ts = pd.to_datetime(df_slice["Timestamp"])
    seconds = (ts.diff().dt.total_seconds().fillna(0).values)
    total2 = 0.0
    for i in range(len(f)-1):
        dt_hours = seconds[i+1] / 3600.0
        avg = 0.5 * (f[i] + f[i+1])
        total2 += avg * dt_hours
    return total2

vfm_total_kg = integrate_flow_kg(plot_df, vfm_plot_col)
mtrol_total_kg = integrate_flow_kg(plot_df, mtrol_plot_col)

# ----------------------------
# KPIs: P2 within thresholds of setpoint (fraction)
# ----------------------------
kpi_results = []
if plot_df["P2_SP"].notna().sum() > 0:
    # compute relative abs error |P2 - SP| / SP
    # avoid div by zero: if P2_SP == 0 use absolute difference instead (mark separately)
    eps = 1e-6
    # mask where P2_SP > eps
    mask_sp = plot_df["P2_SP"].abs() > eps
    rel_err = np.full(len(plot_df), np.nan)
    rel_err[mask_sp] = np.abs(plot_df.loc[mask_sp, "P2"] - plot_df.loc[mask_sp, "P2_SP"]) / np.abs(plot_df.loc[mask_sp, "P2_SP"])
    # also compute absolute errors for rows where SP is zero
    abs_err = np.abs(plot_df["P2"] - plot_df["P2_SP"])
    for thr in kpi_thresholds:
        if thr <= 0.01:  # small thresholds: interpret as fraction (0.001 -> 0.1%)
            frac = np.nanmean(rel_err <= thr) * 100.0
        else:
            # larger thresholds such as 0.02 etc. also treat as fraction
            frac = np.nanmean(rel_err <= thr) * 100.0
        kpi_results.append((thr, frac))
else:
    # no P2_SP available
    kpi_results = []

# ----------------------------
# Build interactive Plotly figure
# ----------------------------
fig = go.Figure()

# VFM (blue)
fig.add_trace(go.Scatter(
    x=plot_df["t_seconds"],
    y=plot_df[vfm_plot_col],
    mode="lines",
    name="VFM Flow (kg/h)",
    line=dict(color="#0B62FF", width=1.6),
    hovertemplate="t: %{x:.0f}s<br>VFM: %{y:.2f} kg/h<br>Time: %{customdata}",
    customdata=plot_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
))

# Mtrol (red)
fig.add_trace(go.Scatter(
    x=plot_df["t_seconds"],
    y=plot_df[mtrol_plot_col],
    mode="lines",
    name="Mtrol Revised Flow (kg/h)",
    line=dict(color="#FF4136", width=1.6),
    hovertemplate="t: %{x:.0f}s<br>Mtrol: %{y:.2f} kg/h<br>Time: %{customdata}",
    customdata=plot_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
))

# P1 / P2 (pressures) — plotted on a separate right-hand pressure axis
if show_pressures:
    fig.add_trace(go.Scatter(
        x=plot_df["t_seconds"],
        y=plot_df["P1"],
        mode="lines",
        name="P1 (barg)",
        line=dict(color="#2ECC40", dash="dash", width=1.2),
        yaxis="y3",
        hovertemplate="t: %{x:.0f}s<br>P1: %{y:.3f} barg<br>Time: %{customdata}",
        customdata=plot_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["t_seconds"],
        y=plot_df["P2"],
        mode="lines",
        name="P2 (barg)",
        line=dict(color="#800080", dash="dot", width=1.4),
        yaxis="y3",
        hovertemplate="t: %{x:.0f}s<br>P2: %{y:.3f} barg<br>Time: %{customdata}",
        customdata=plot_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    ))
    # if P2_SP exists, plot setpoint dashed
    if "P2_SP" in plot_df.columns and plot_df["P2_SP"].notna().any():
        fig.add_trace(go.Scatter(
            x=plot_df["t_seconds"],
            y=plot_df["P2_SP"],
            mode="lines",
            name="P2_SP (barg)",
            line=dict(color="#AAAAAA", dash="dashdot", width=1.0),
            yaxis="y3",
            hovertemplate="t: %{x:.0f}s<br>P2_SP: %{y:.3f} barg<br>Time: %{customdata}",
            customdata=plot_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        ))

# Error % trace (optional) — will be on its own right axis (y2)
if show_error_pct:
    fig.add_trace(go.Scatter(
        x=plot_df["t_seconds"],
        y=plot_df[err_plot_col],
        mode="lines",
        name="Error % (VFM - Mtrol)",
        line=dict(color="#00CC96", width=1.2),
        yaxis="y2",
        hovertemplate="t: %{x:.0f}s<br>Error: %{y:.2f} %<br>Time: %{customdata}",
        customdata=plot_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    ))

# layout with multiple axes
fig.update_layout(
    xaxis=dict(title="Elapsed time (seconds)"),
    yaxis=dict(title="Flow (kg/h)", side="left"),
    yaxis2=dict(title="Error %", overlaying="y", side="right", showgrid=False, zeroline=True,
                zerolinecolor="gray", zerolinewidth=1, position=0.92),
    # pressure axis (y3) on right but offset so it doesn't overlap error axis
    yaxis3=dict(title="Pressure (barg)", overlaying="y", side="right", position=0.82),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=50, r=80, t=90, b=60),
    height=650,
    hovermode="x unified",
    title=f"VFM vs Mtrol — elapsed seconds [{int(sel_range[0])}s → {int(sel_range[1])}s]"
)

# Improve ticks
fig.update_xaxes(nticks=10)

# ----------------------------
# Top-line KPIs
# ----------------------------
col1, col2, col3, col4 = st.columns([1,1,1,1])
col1.metric("Points shown", f"{len(plot_df):,}")
col2.metric("Window duration", f"{timedelta(seconds=int(plot_df['t_seconds'].max()-plot_df['t_seconds'].min()))}")
col3.metric("Total VFM (kg)", f"{vfm_total_kg:,.1f}")
col4.metric("Total Mtrol (kg)", f"{mtrol_total_kg:,.1f}")

# ----------------------------
# P2-in-spec KPIs (table)
# ----------------------------
st.markdown("### P2 control quality (fraction within ±threshold of P2 setpoint) — selected window")
if len(kpi_results) == 0:
    st.info("No P2 setpoint (P2_SP) values found in the selected window.")
else:
    kpi_table = pd.DataFrame(kpi_results, columns=["threshold_fraction", "percent_in_spec"])
    # format threshold as percent text
    def fmt_thr(x):
        if x < 0.01:
            return f"±{x*100:.1f}%"
        else:
            return f"±{x*100:.0f}%"
    kpi_table["threshold"] = kpi_table["threshold_fraction"].apply(fmt_thr)
    st.dataframe(kpi_table[["threshold","percent_in_spec"]].rename(columns={"percent_in_spec":"% in spec"}), use_container_width=True)

# ----------------------------
# Show the interactive chart
# ----------------------------
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Show sample table toggle
# ----------------------------
if st.checkbox("Show table (first 300 rows)"):
    st.dataframe(plot_df[["Timestamp","t_seconds","VFM_flow_kgph","Mtrol_flow_kgph","Error_pct","P1","P2","P2_SP"]].head(300))

# ----------------------------
# Explain zoom vs slider note
# ----------------------------
st.markdown(
    """
    **Note:** You can zoom/pan interactively on the plot for visual inspection.
    To **compute totals** (VFM & Mtrol mass in kg) for a zoomed region, use the *Elapsed time range* slider in the sidebar — the totals and KPIs above update for that selected window.
    (Plotly's client-side zoom box is for exploration; programmatic access to the client zoom in Streamlit requires additional components.)
    """
)

# ----------------------------
# Footer / credits
# ----------------------------
st.markdown("---")
st.caption("Colors: blue = VFM, red = Mtrol, purple/orange = pressures. Error % is green. Orders from Emyr Var Emreis, Emperor of Nilfgaard. ")

