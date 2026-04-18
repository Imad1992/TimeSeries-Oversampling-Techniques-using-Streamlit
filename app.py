# TimeSeries Oversampling Lab (Streamlit)
# --------------------------------------
# Attractive Streamlit UI + fixed "synthetic_rows" KeyError.
#
# Run in VS Code terminal (Windows / macOS / Linux):
#   python -m venv .venv
#   # Windows PowerShell:
#   #   .\.venv\Scripts\Activate.ps1
#   # macOS/Linux:
#   #   source .venv/bin/activate
#   pip install streamlit numpy pandas matplotlib scipy
#   streamlit run app.py
#
# If you previously used Anaconda, ensure VS Code uses the same interpreter
# where you installed packages (Python: Select Interpreter).

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# ----------------------------
# Page + UI styling
# ----------------------------
st.set_page_config(page_title="TimeSeries Oversampling Lab", page_icon="📈", layout="wide")

st.markdown(
    """
<style>
/* widen main container a bit */
.block-container { padding-top: 1.4rem; padding-bottom: 2.2rem; }

/* nicer metric cards */
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 14px 14px;
}

/* section headers */
.section-title {
  font-size: 1.05rem;
  font-weight: 800;
  margin: 0.6rem 0 0.4rem 0;
  opacity: 0.95;
}

/* soft panels for better readability */
.panel {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 14px;
}

.small-muted { opacity: 0.8; font-size: 0.9rem; }
hr { border-color: rgba(255,255,255,0.12); }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📈 TimeSeries Oversampling Lab")
st.caption("Upload time-series data → generate synthetic samples with multiple techniques → visualize → download CSV outputs.")


# ----------------------------
# Techniques (classical augmentations)
# ----------------------------
def _rng(seed: int):
    return np.random.default_rng(seed)

def _prepare(ts: pd.DataFrame):
    t = ts["timestamp"].astype("int64").to_numpy()  # ns since epoch
    y = ts["value"].astype(float).to_numpy()
    return t, y

def _build_df(t_ns: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    return (
        pd.DataFrame({"timestamp": pd.to_datetime(t_ns), "value": y})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

def _sample_indices(n: int, add_n: int, rng):
    return rng.integers(0, n, size=add_n)

def jitter_noise(ts: pd.DataFrame, add_n: int, seed: int = 42, sigma: float = 0.02):
    """Add gaussian noise to values."""
    rng = _rng(seed)
    t, y = _prepare(ts)
    idx = _sample_indices(len(y), add_n, rng)
    scale = max(1e-9, np.std(y))
    y_syn = y[idx] + rng.normal(0, sigma * scale, size=add_n)
    t_syn = t[idx]
    return _build_df(t_syn, y_syn)

def scaling(ts: pd.DataFrame, add_n: int, seed: int = 42, scale_sigma: float = 0.05):
    """Random multiplicative scaling."""
    rng = _rng(seed + 1)
    t, y = _prepare(ts)
    idx = _sample_indices(len(y), add_n, rng)
    factors = rng.normal(1.0, scale_sigma, size=add_n)
    return _build_df(t[idx], y[idx] * factors)

def magnitude_warp(ts: pd.DataFrame, add_n: int, seed: int = 42, knot_count: int = 6, warp_sigma: float = 0.20):
    """Smoothly warp magnitude across time with a random curve."""
    rng = _rng(seed + 2)
    t, y = _prepare(ts)

    knots_x = np.linspace(t.min(), t.max(), knot_count)
    knots_y = rng.normal(1.0, warp_sigma, size=knot_count)
    f = interp1d(knots_x, knots_y, kind="cubic", fill_value="extrapolate")

    idx = _sample_indices(len(y), add_n, rng)
    t_syn = t[idx]
    y_syn = y[idx] * f(t_syn)
    return _build_df(t_syn, y_syn)

def time_warp(ts: pd.DataFrame, add_n: int, seed: int = 42, knot_count: int = 6, warp_sigma: float = 0.15):
    """Slightly shift timestamps (smoothly) then resample values by interpolation."""
    rng = _rng(seed + 3)
    t, y = _prepare(ts)

    f_val = interp1d(t, y, kind="linear", fill_value="extrapolate")

    knots_x = np.linspace(t.min(), t.max(), knot_count)
    knots_y = rng.normal(0.0, warp_sigma, size=knot_count)
    f_warp = interp1d(knots_x, knots_y, kind="cubic", fill_value="extrapolate")

    idx = _sample_indices(len(t), add_n, rng)
    t_base = t[idx]
    span = max(1, t.max() - t.min())
    t_syn = t_base + (f_warp(t_base) * span * 0.02).astype(np.int64)

    y_syn = f_val(t_syn)
    return _build_df(t_syn, y_syn)

def window_slicing(ts: pd.DataFrame, add_n: int, seed: int = 42, window_frac: float = 0.35):
    """
    Pick random windows and use them to produce local values around random timestamps.
    Good for preserving local patterns.
    """
    rng = _rng(seed + 4)
    t, y = _prepare(ts)
    n = len(y)
    if n < 10:
        return jitter_noise(ts, add_n, seed=seed)

    w = max(10, int(n * window_frac))
    t_min, t_max = t.min(), t.max()
    synth_t = rng.integers(t_min, t_max, size=add_n)

    y_syn = np.empty(add_n, dtype=float)
    for i in range(add_n):
        start = rng.integers(0, n - w)
        end = start + w
        t_win = t[start:end]
        y_win = y[start:end]

        tn = (t_win - t_win.min()) / max(1, (t_win.max() - t_win.min()))
        local_span = (t_max - t_min) * 0.08
        t_local = synth_t[i] + (tn - 0.5) * local_span

        f_win = interp1d(t_local, y_win, kind="linear", fill_value="extrapolate")
        y_syn[i] = f_win(synth_t[i])

    return _build_df(synth_t, y_syn)

def bootstrap_blocks(ts: pd.DataFrame, add_n: int, seed: int = 42, block_size: int = 24):
    """
    Block bootstrap values from contiguous blocks to preserve short-term autocorrelation.
    """
    rng = _rng(seed + 5)
    t, y = _prepare(ts)
    n = len(y)
    if n < block_size + 2:
        return jitter_noise(ts, add_n, seed=seed)

    starts = rng.integers(0, n - block_size, size=max(1, add_n // block_size + 1))
    y_pool = np.concatenate([y[s:s + block_size] for s in starts])

    idx_t = _sample_indices(n, add_n, rng)
    t_syn = t[idx_t]
    y_syn = y_pool[rng.integers(0, len(y_pool), size=add_n)]
    return _build_df(t_syn, y_syn)

TECHNIQUES = {
    "jitter": jitter_noise,
    "scaling": scaling,
    "magwarp": magnitude_warp,
    "timewarp": time_warp,
    "winslice": window_slicing,
    "bootstrap": bootstrap_blocks,
}


# ----------------------------
# Helpers
# ----------------------------
def parse_timeseries(df: pd.DataFrame, time_col: str, value_col: str) -> pd.DataFrame:
    ts = df[[time_col, value_col]].copy()
    ts.columns = ["timestamp", "value"]
    ts["timestamp"] = pd.to_datetime(ts["timestamp"], errors="coerce")
    ts["value"] = pd.to_numeric(ts["value"], errors="coerce")
    ts = ts.dropna(subset=["timestamp", "value"]).sort_values("timestamp").reset_index(drop=True)
    return ts

def describe_arr(y: np.ndarray):
    return {
        "count": int(len(y)),
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "p05": float(np.quantile(y, 0.05)),
        "p50": float(np.quantile(y, 0.50)),
        "p95": float(np.quantile(y, 0.95)),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
    }

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def split_budget(total_to_add: int, chosen: List[str]) -> Dict[str, int]:
    if total_to_add <= 0:
        return {name: 0 for name in chosen}
    k = max(1, len(chosen))
    base = total_to_add // k
    rem = total_to_add % k
    return {name: base + (1 if i < rem else 0) for i, name in enumerate(chosen)}

def overlay_fig(orig: pd.DataFrame, synth_by_tech: Dict[str, pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(orig["timestamp"], orig["value"], label="original", linewidth=1.8)
    for name, syn in synth_by_tech.items():
        ax.scatter(syn["timestamp"], syn["value"], s=10, alpha=0.30, label=f"synthetic:{name}")
    ax.set_title("Overlay: Original (line) vs Synthetic (points)")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    return fig

def hist_fig(orig: pd.DataFrame, syn: pd.DataFrame, name: str):
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.hist(orig["value"].to_numpy(), bins=40, alpha=0.65, label="original")
    ax.hist(syn["value"].to_numpy(), bins=40, alpha=0.65, label=f"synthetic:{name}")
    ax.set_title(f"Value Distribution: {name}")
    ax.legend()
    fig.tight_layout()
    return fig

def make_merged_all(orig: pd.DataFrame, synth_by_tech: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = [orig.assign(source="original")]
    for name, syn in synth_by_tech.items():
        parts.append(syn.assign(source=f"synthetic_{name}"))
    return pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


# ----------------------------
# Sidebar: Inputs
# ----------------------------
with st.sidebar:
    st.markdown("## Controls")
    st.markdown('<div class="small-muted">Upload data and configure augmentation.</div>', unsafe_allow_html=True)
    st.divider()

    use_sample = st.checkbox("Use built-in sample data", value=False)
    uploaded = None if use_sample else st.file_uploader("Upload CSV", type=["csv"])

    st.divider()
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    min_rows = st.number_input("Small-data threshold (min rows)", min_value=10, value=500, step=10)

    st.divider()
    st.markdown("### Techniques")
    chosen = st.multiselect(
        "Select techniques",
        options=list(TECHNIQUES.keys()),
        default=["jitter", "scaling", "magwarp", "timewarp", "winslice", "bootstrap"],
    )

    st.divider()
    run = st.button("▶ Run oversampling", type="primary")


# ----------------------------
# Load data
# ----------------------------
if use_sample:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=120, freq="H"),
        "value": (np.sin(np.linspace(0, 10*np.pi, 120)) * 8 + 100) + rng.normal(0, 0.6, 120)
    })
    st.success("Using built-in sample data.")
else:
    if not uploaded:
        st.info("Upload a CSV (or enable the sample data) to begin.")
        st.stop()
    df = pd.read_csv(uploaded)

st.markdown('<div class="section-title">1) Data Preview</div>', unsafe_allow_html=True)
st.dataframe(df.head(30), use_container_width=True)

cols = list(df.columns)
if len(cols) < 2:
    st.error("Your CSV must have at least 2 columns (time + value).")
    st.stop()

cA, cB, cC = st.columns([1.0, 1.0, 1.2])
with cA:
    time_col = st.selectbox("Time column", options=cols, index=0)
with cB:
    value_col = st.selectbox("Value column", options=cols, index=1)
with cC:
    st.markdown('<div class="small-muted">Tip: time should be parseable dates; value numeric.</div>', unsafe_allow_html=True)

try:
    ts = parse_timeseries(df, time_col, value_col)
except Exception as e:
    st.error(f"Failed to parse your time/value columns: {e}")
    st.stop()

if len(ts) < 2:
    st.error("After parsing/cleaning, there are not enough valid rows. Check column selection and data format.")
    st.stop()

st.markdown('<div class="section-title">2) Oversampling Settings</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Parsed rows", f"{len(ts)}")
m2.metric("Small-data threshold", f"{int(min_rows)}")
m3.metric("Seed", f"{int(seed)}")
m4.metric("Techniques selected", f"{len(chosen)}")

target_rows = st.number_input(
    "Target total rows (after oversampling)",
    min_value=int(len(ts)),
    value=int(max(len(ts), 2000)),
    step=100
)

to_add = max(0, int(target_rows) - len(ts))
st.info(f"Original rows: {len(ts)} • Target rows: {int(target_rows)} • Synthetic rows to add: {to_add}")

if not chosen:
    st.warning("Select at least 1 technique in the sidebar.")
    st.stop()

# If user hasn't clicked Run yet, stop here
if not run:
    st.stop()


# ----------------------------
# Generate synthetic data
# ----------------------------
plan = split_budget(to_add, chosen)

synth_by_tech: Dict[str, pd.DataFrame] = {}
summary_rows: List[dict] = []

for name in chosen:
    add_n = int(plan.get(name, 0))
    if add_n <= 0:
        continue
    synth = TECHNIQUES[name](ts, add_n=add_n, seed=int(seed))
    synth_by_tech[name] = synth

    summary_rows.append({
        "technique": name,
        "original_rows": int(len(ts)),
        "synthetic_rows": int(len(synth)),
        "total_if_merged": int(len(ts) + len(synth)),
        "synthetic_ratio": round(len(synth) / max(1, len(ts)), 4),
    })

st.markdown('<div class="section-title">3) Synthetic Generation Summary</div>', unsafe_allow_html=True)

summary_df = pd.DataFrame(summary_rows)

# ✅ Fix: only sort if the column exists (i.e., if we generated synthetic rows)
if not summary_df.empty and "synthetic_rows" in summary_df.columns:
    summary_df = summary_df.sort_values("synthetic_rows", ascending=False)
    st.dataframe(summary_df, use_container_width=True)
else:
    st.warning(
        "No synthetic rows were generated.\n\n"
        "Most common reasons:\n"
        "- Your target rows equals parsed rows (to_add = 0)\n"
        "- Budget split results in 0 rows per technique\n\n"
        "Increase Target total rows and run again."
    )

# Small-data note (informational)
if len(ts) >= int(min_rows) and to_add == 0:
    st.info("Dataset is not considered small (and target already reached). Showing original only.")


# ----------------------------
# Visualizations
# ----------------------------
st.markdown('<div class="section-title">4) Visual Preview</div>', unsafe_allow_html=True)

left, right = st.columns([1.5, 1.0])
with left:
    st.pyplot(overlay_fig(ts, synth_by_tech), use_container_width=True)

with right:
    st.markdown("**Original value stats**")
    st.json(describe_arr(ts["value"].to_numpy()))

if synth_by_tech:
    st.markdown('<div class="section-title">5) Technique Comparisons</div>', unsafe_allow_html=True)
    for name, syn in synth_by_tech.items():
        st.markdown(f"#### {name}")
        c1, c2 = st.columns([1.2, 1.0])
        with c1:
            st.pyplot(hist_fig(ts, syn, name), use_container_width=True)
        with c2:
            st.markdown("**Synthetic value stats**")
            st.json(describe_arr(syn["value"].to_numpy()))
else:
    st.info("No synthetic datasets to compare. Increase Target total rows to generate synthetic samples.")


# ----------------------------
# Downloads
# ----------------------------
st.markdown('<div class="section-title">6) Download Files</div>', unsafe_allow_html=True)

st.download_button(
    "⬇️ Download original.csv",
    data=to_csv_bytes(ts),
    file_name="original.csv",
    mime="text/csv"
)

# per-tech
for name, syn in synth_by_tech.items():
    st.download_button(
        f"⬇️ Download synthetic_{name}.csv",
        data=to_csv_bytes(syn),
        file_name=f"synthetic_{name}.csv",
        mime="text/csv"
    )

# merged per tech
for name, syn in synth_by_tech.items():
    merged = (
        pd.concat(
            [ts.assign(source="original"), syn.assign(source=f"synthetic_{name}")],
            ignore_index=True
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    st.download_button(
        f"⬇️ Download merged_original_plus_{name}.csv",
        data=to_csv_bytes(merged),
        file_name=f"merged_original_plus_{name}.csv",
        mime="text/csv"
    )

# merged all
if synth_by_tech:
    merged_all = make_merged_all(ts, synth_by_tech)
    st.download_button(
        "⬇️ Download merged_all_techniques.csv",
        data=to_csv_bytes(merged_all),
        file_name="merged_all_techniques.csv",
        mime="text/csv"
    )

st.success("Finished! If you don’t see synthetic outputs, increase Target total rows and rerun.")