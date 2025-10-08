"""
Probabilistic plume model, Streamlit app (batched engine)

Run locally:
1) pip install -r requirements.txt
2) streamlit run probabilistic_plume_app.py

requirements.txt content:
-------------------------
streamlit>=1.50
numpy>=2.3
matplotlib
# Optional for later features
# pandas
# plotly
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st

# Try to import matplotlib and warn if unavailable
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception as e:
    HAVE_MPL = False
    st.warning(f"Matplotlib not available: {e}. Install it via requirements.txt.")

# -----------------------------
# Streamlit page setup
# -----------------------------

st.set_page_config(
    page_title="Probabilistic plume model",
    page_icon="💨",
    layout="wide",
)

# -----------------------------
# Data classes and parameters
# -----------------------------

@dataclass
class SimParams:
    N: int = 99
    T_a: float = 20.0
    k: float = 5.0
    alpha: float = 0.5  # mixing fraction
    steps: int = 200
    parcels_per_step: int = 10
    seed: int = 42
    snapshot_stride: int = 10  # take a field snapshot every this many steps


@dataclass
class SimResults:
    T: np.ndarray  # final temperature field
    snapshots: List[Tuple[int, np.ndarray]]  # list of (t, field)
    diagnostics: Dict[str, List[Tuple[int, float]]]  # time series per metric
    params: SimParams


# -----------------------------
# Core model helpers (batched engine)
# -----------------------------

def init_grid(params: SimParams) -> np.ndarray:
    """Create initial temperature field with a hot source at the centre."""
    N = params.N
    T = np.full((N, N), params.T_a, dtype=np.float64)
    c = N // 2
    T[c, c] = params.k * params.T_a
    return T


def choose_move_weights(T_particle: float, T_neighbors: Dict[str, float], T_a: float) -> Dict[str, float]:
    """Return unnormalised move weights for up, left, right, down based on the basic rule.
    Cooler neighbors only. Up gets bias b = T_particle / T_a. Others get weight 1.
    """
    b = T_particle / T_a
    weights: Dict[str, float] = {"up": 0.0, "left": 0.0, "right": 0.0, "down": 0.0}
    for d, Tn in T_neighbors.items():
        if Tn < T_particle:
            weights[d] = b if d == "up" else 1.0
    return weights


def step_once_batched(T: np.ndarray, params: SimParams, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
    """Advance the simulation by one global step on a read at t, write to t+1 schedule.
    Batched mixing: accumulate arrivals per destination cell, then apply a closed form update.
    Parcels are *not* persistent; they are injected at the source and mixed once per step.
    """
    N = params.N
    c = N // 2

    # Read source temperature from time t once for all parcels in this step
    T_source = T[c, c]

    # Arrival counters per cell
    arrivals = np.zeros((N, N), dtype=np.int32)

    # Diagnostics
    moved_up = 0
    total_moves = 0

    # Neighbor map and temps around the source at time t
    i, j = c, c
    nbrs = {
        "up": (i - 1, j) if i - 1 >= 0 else None,
        "down": (i + 1, j) if i + 1 < N else None,
        "left": (i, j - 1) if j - 1 >= 0 else None,
        "right": (i, j + 1) if j + 1 < N else None,
    }
    T_neighbors: Dict[str, float] = {d: T[idx] for d, idx in nbrs.items() if idx is not None}

    weights = choose_move_weights(T_source, T_neighbors, params.T_a)
    total_w = sum(weights.values())

    if total_w > 0.0:
        dirs = list(weights.keys())
        probs = np.array([weights[d] for d in dirs], dtype=np.float64)
        probs /= probs.sum()
        # Draw destinations for r parcels in one vectorised call
        choices = rng.choice(len(dirs), size=params.parcels_per_step, p=probs)
        for ch in choices:
            dest = nbrs[dirs[ch]]
            di, dj = dest
            arrivals[di, dj] += 1
            total_moves += 1
            if dirs[ch] == "up":
                moved_up += 1
    else:
        # No cooler neighbors, all parcels stay at the source cell
        arrivals[c, c] += params.parcels_per_step
        total_moves += params.parcels_per_step

    # Now apply batched mixing to produce T_next
    T_next = T.copy()

    # For cells with m arrivals, apply: T_new = (1-α)^m * T_old + [1-(1-α)^m] * T_source
    if params.alpha != 0.0:
        m_mask = arrivals > 0
        if np.any(m_mask):
            m = arrivals[m_mask].astype(np.float64)
            factor = np.power(1.0 - params.alpha, m)
            T_old = T[m_mask]
            T_new = factor * T_old + (1.0 - factor) * T_source
            T_next[m_mask] = T_new

    # Diagnostics
    diag: Dict[str, float] = {}
    diag["max_T_over_Ta"] = float(T_next.max() / params.T_a)
    diag["mean_T_over_Ta"] = float(T_next.mean() / params.T_a)

    # Vertical centroid of heat using origin at bottom (imshow origin="lower")
    y = np.arange(N, dtype=np.float64)
    weighted_sum = (T_next * y[:, None]).sum()
    total_temp = T_next.sum()
    diag["vertical_centroid"] = float(weighted_sum / total_temp) if total_temp > 0 else float(c)

    frac_up = moved_up / total_moves if total_moves > 0 else 0.0
    diag["frac_moves_up"] = float(frac_up)

    return T_next, diag


def run_simulation(params: SimParams) -> SimResults:
    rng = np.random.default_rng(params.seed)
    T = init_grid(params)

    snapshots: List[Tuple[int, np.ndarray]] = [(0, T.copy())]
    diagnostics: Dict[str, List[Tuple[int, float]]] = {
        "max_T_over_Ta": [(0, float(T.max() / params.T_a))],
        "mean_T_over_Ta": [(0, float(T.mean() / params.T_a))],
        "vertical_centroid": [(0, float((np.arange(params.N)[:, None] * T).sum() / T.sum()))],
        "frac_moves_up": [(0, 0.0)],
    }

    for t in range(1, params.steps + 1):
        T, diag = step_once_batched(T, params, rng)
        if t % params.snapshot_stride == 0 or t == params.steps:
            snapshots.append((t, T.copy()))
        for k, v in diag.items():
            diagnostics.setdefault(k, []).append((t, v))

    return SimResults(T=T, snapshots=snapshots, diagnostics=diagnostics, params=params)

# -----------------------------
# UI controls
# -----------------------------

st.title("Probabilistic buoyant plume: minimal model")
st.caption("Batched, non-persistent parcels; read t then write t+1")

with st.sidebar:
    st.header("Controls")
    N = st.number_input("Grid size N", min_value=21, max_value=401, value=99, step=2, help="Prefer odd so the centre is unique")
    T_a = st.number_input("Ambient temperature T_a", min_value=0.1, value=20.0)
    k = st.number_input("Source multiplier k", min_value=1.0, value=5.0)
    alpha = st.slider("Mixing fraction alpha", min_value=0.0, max_value=1.0, value=0.5)
    steps = st.number_input("Time steps", min_value=1, value=200)
    parcels_per_step = st.number_input("Parcels per step r", min_value=1, value=10)
    seed = st.number_input("Random seed", min_value=0, value=42)
    snapshot_stride = st.number_input("Snapshot stride", min_value=1, value=10)

    run_btn = st.button("Run simulation", type="primary")

# Store params in session state for reproducibility and reruns
if "params" not in st.session_state:
    st.session_state.params = None
if "results" not in st.session_state:
    st.session_state.results = None

if run_btn:
    params = SimParams(
        N=int(N), T_a=float(T_a), k=float(k), alpha=float(alpha),
        steps=int(steps), parcels_per_step=int(parcels_per_step),
        seed=int(seed), snapshot_stride=int(snapshot_stride)
    )
    st.session_state.params = params
    with st.spinner("Running simulation..."):
        st.session_state.results = run_simulation(params)

# -----------------------------
# Visuals
# -----------------------------

res: Optional[SimResults] = st.session_state.results
col1, col2 = st.columns([2, 1])

if res is None:
    with col1:
        st.info("Press Run simulation to generate a field. A heatmap will appear here.")
    with col2:
        st.info("Diagnostics and profiles will appear here.")
else:
    if not HAVE_MPL:
        st.error("Matplotlib is required for plots. Install it via requirements.txt.")
    else:
        # Heatmap of the latest snapshot
        t_last, T_last = res.snapshots[-1]
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        im = ax1.imshow(T_last / res.params.T_a, origin="lower", interpolation="nearest")
        ax1.set_title(f"Field at t = {t_last} (T/T_a)")
        ax1.set_xlabel("x index")
        ax1.set_ylabel("y index")
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="T/T_a")
        with col1:
            st.pyplot(fig1, clear_figure=True)

        # Centreline profile above the source
        c = res.params.N // 2
        profile = T_last[c:, c] / res.params.T_a  # from centre upward
        y = np.arange(profile.size)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(profile, y)
        ax2.set_xlabel("T/T_a")
        ax2.set_ylabel("vertical distance above centre")
        ax2.set_title("Centreline profile above source")
        with col2:
            st.pyplot(fig2, clear_figure=True)

        # Diagnostics time series
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        for key in ["max_T_over_Ta", "mean_T_over_Ta", "vertical_centroid", "frac_moves_up"]:
            ts = np.array(res.diagnostics[key])
            ax3.plot(ts[:, 0], ts[:, 1], label=key)
        ax3.set_xlabel("time step")
        ax3.set_ylabel("value")
        ax3.set_title("Diagnostics time series")
        ax3.legend()
        with col2:
            st.pyplot(fig3, clear_figure=True)

# -----------------------------
# Export area
# -----------------------------

if res is not None:
    st.subheader("Export")
    snaps = {f"t{t}": arr for t, arr in res.snapshots}
    npz_bytes = None
    try:
        import io
        buf = io.BytesIO()
        np.savez_compressed(buf, **snaps)
        npz_bytes = buf.getvalue()
    except Exception as e:
        st.warning(f"Could not package snapshots: {e}")

    if npz_bytes is not None:
        st.download_button(
            label="Download snapshots (npz)",
            data=npz_bytes,
            file_name="plume_snapshots.npz",
            mime="application/zip",
        )

st.markdown(
    """
    ---
    Notes: non-persistent parcels with batched mixing. Later we will add the Brownian floor and directional persistence as optional features.
    """
)