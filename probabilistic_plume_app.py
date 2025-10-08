"""
Probabilistic plume model, Streamlit scaffold

To run locally:
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

import math
import time
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

st.set_page_config(
    page_title="Probabilistic plume model",
    page_icon="💨",
    layout="wide",
)

@dataclass
class SimParams:
    N: int = 99
    T_a: float = 20.0
    k: float = 5.0
    alpha: float = 0.5
    steps: int = 200
    parcels_per_step: int = 1
    seed: int = 42
    snapshot_stride: int = 10

@dataclass
class SimResults:
    T: np.ndarray
    snapshots: List[Tuple[int, np.ndarray]]
    diagnostics: Dict[str, List[Tuple[int, float]]]
    params: SimParams

def init_grid(params: SimParams) -> np.ndarray:
    N = params.N
    T = np.full((N, N), params.T_a, dtype=np.float64)
    c = N // 2
    T[c, c] = params.k * params.T_a
    return T

def choose_move_weights(T_particle: float, T_neighbors: Dict[str, float], T_a: float) -> Dict[str, float]:
    b = T_particle / T_a
    weights: Dict[str, float] = {"up": 0.0, "left": 0.0, "right": 0.0, "down": 0.0}
    for d, Tn in T_neighbors.items():
        if Tn < T_particle:
            weights[d] = b if d == "up" else 1.0
    return weights

def step_once(T: np.ndarray, params: SimParams, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
    N = params.N
    c = N // 2
    T_next = T.copy()
    diag: Dict[str, float] = {}

    for _ in range(params.parcels_per_step):
        i, j = c, c
        T_particle = T[i, j]

        nbrs = {
            "up": (max(i - 1, 0), j) if i - 1 >= 0 else None,
            "down": (min(i + 1, N - 1), j) if i + 1 < N else None,
            "left": (i, max(j - 1, 0)) if j - 1 >= 0 else None,
            "right": (i, min(j + 1, N - 1)) if j + 1 < N else None,
        }
        T_neighbors: Dict[str, float] = {d: T[idx] for d, idx in nbrs.items() if idx is not None}
        weights = choose_move_weights(T_particle, T_neighbors, params.T_a)
        total = sum(weights.values())

        if total <= 0.0:
            dest = (i, j)
        else:
            dirs = list(weights.keys())
            probs = np.array([weights[d] for d in dirs], dtype=np.float64)
            probs /= probs.sum()
            choice = rng.choice(len(dirs), p=probs)
            dest = nbrs[dirs[choice]]

        di, dj = dest
        T_next[di, dj] = (1.0 - params.alpha) * T_next[di, dj] + params.alpha * T[i, j]

    diag["max_T_over_Ta"] = float(T_next.max() / params.T_a)
    diag["mean_T_over_Ta"] = float(T_next.mean() / params.T_a)

    return T_next, diag

def run_simulation(params: SimParams) -> SimResults:
    rng = np.random.default_rng(params.seed)
    T = init_grid(params)
    snapshots: List[Tuple[int, np.ndarray]] = [(0, T.copy())]
    diagnostics: Dict[str, List[Tuple[int, float]]] = {
        "max_T_over_Ta": [(0, float(T.max() / params.T_a))],
        "mean_T_over_Ta": [(0, float(T.mean() / params.T_a))],
    }

    for t in range(1, params.steps + 1):
        T, diag = step_once(T, params, rng)
        if t % params.snapshot_stride == 0 or t == params.steps:
            snapshots.append((t, T.copy()))
        for k, v in diag.items():
            diagnostics.setdefault(k, []).append((t, v))

    return SimResults(T=T, snapshots=snapshots, diagnostics=diagnostics, params=params)

st.title("Probabilistic buoyant plume: minimal model")
with st.sidebar:
    st.header("Controls")
    N = st.number_input("Grid size N", min_value=21, max_value=401, value=99, step=2)
    T_a = st.number_input("Ambient temperature T_a", min_value=0.1, value=20.0)
    k = st.number_input("Source multiplier k", min_value=1.0, value=5.0)
    alpha = st.slider("Mixing fraction alpha", 0.0, 1.0, 0.5)
    steps = st.number_input("Time steps", min_value=1, value=200)
    parcels_per_step = st.number_input("Parcels per step r", min_value=1, value=1)
    seed = st.number_input("Random seed", min_value=0, value=42)
    snapshot_stride = st.number_input("Snapshot stride", min_value=1, value=10)
    run_btn = st.button("Run simulation", type="primary")

if "params" not in st.session_state:
    st.session_state.params = None
if "results" not in st.session_state:
    st.session_state.results = None

if run_btn:
    params = SimParams(int(N), float(T_a), float(k), float(alpha), int(steps), int(parcels_per_step), int(seed), int(snapshot_stride))
    st.session_state.params = params
    with st.spinner("Running simulation..."):
        st.session_state.results = run_simulation(params)

res: Optional[SimResults] = st.session_state.results
col1, col2 = st.columns([2, 1])

if res is None:
    with col1:
        st.info("Press Run simulation to generate a field.")
else:
    if not HAVE_MPL:
        st.error("Matplotlib is required for plots. Install it via requirements.txt.")
    else:
        t_last, T_last = res.snapshots[-1]
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        im = ax1.imshow(T_last / res.params.T_a, origin="lower", interpolation="nearest")
        ax1.set_title(f"Field at t={t_last} (T/T_a)")
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="T/T_a")
        with col1:
            st.pyplot(fig1, clear_figure=True)

        c = res.params.N // 2
        profile = T_last[: c + 1, c] / res.params.T_a
        y = np.arange(profile.size)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(profile, y)
        ax2.set_xlabel("T/T_a")
        ax2.set_ylabel("vertical index")
        ax2.set_title("Centreline profile above source")
        with col2:
            st.pyplot(fig2, clear_figure=True)

        fig3, ax3 = plt.subplots(figsize=(6, 3))
        for key, series in res.diagnostics.items():
            ts = np.array(series)
            ax3.plot(ts[:, 0], ts[:, 1], label=key)
        ax3.set_xlabel("time step")
        ax3.set_title("Diagnostics time series")
        ax3.legend()
        with col2:
            st.pyplot(fig3, clear_figure=True)

st.markdown("---\nIf you see a Matplotlib import error, add a requirements.txt with streamlit, numpy, matplotlib and redeploy.")