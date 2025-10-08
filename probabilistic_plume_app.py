"""
Probabilistic plume model, Streamlit app (persistent engine) with progress bar and correct 'up' orientation

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
# Stop flag helper
# -----------------------------

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

def request_stop():
    st.session_state.stop_requested = True

# -----------------------------
# Core model helpers (persistent engine)
# ----------------------------- (persistent engine)
# -----------------------------

def init_grid(params: SimParams) -> np.ndarray:
    """Create initial temperature field with a hot source at the centre."""
    N = params.N
    T = np.full((N, N), params.T_a, dtype=np.float64)
    c = N // 2
    T[c, c] = params.k * params.T_a
    return T

# IMPORTANT: display uses origin="lower", so increasing row index i moves visually UP.
# Therefore, define 'up' as (i + 1, j) so that 'up' in the model matches 'up' on the heatmap.

def neighbor_map(i: int, j: int, N: int) -> Dict[str, Optional[Tuple[int, int]]]:
    return {
        "up": (i + 1, j) if i + 1 < N else None,
        "down": (i - 1, j) if i - 1 >= 0 else None,
        "left": (i, j - 1) if j - 1 >= 0 else None,
        "right": (i, j + 1) if j + 1 < N else None,
    }


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

# Persistent parcel representation
from dataclasses import dataclass
@dataclass
class Parcel:
    i: int
    j: int
    T: float


def step_once_persistent(T: np.ndarray, parcels: List[Parcel], params: SimParams, rng: np.random.Generator) -> Tuple[np.ndarray, List[Parcel], Dict[str, float]]:
    N = params.N
    c = N // 2

    # Inject r new parcels at the centre with temperature read from T at time t
    T_source = T[c, c]
    for _ in range(params.parcels_per_step):
        parcels.append(Parcel(c, c, T_source))

    # Prepare arrival buckets per cell: list of incoming parcel temps
    arrivals_T: List[List[List[float]]] = [[[] for _ in range(N)] for __ in range(N)]

    moved_up = 0
    total_moves = 0

    new_parcels: List[Parcel] = []
    for p in parcels:
        nbrs = neighbor_map(p.i, p.j, N)
        T_neighbors: Dict[str, float] = {d: T[idx] for d, idx in nbrs.items() if idx is not None}
        weights = choose_move_weights(p.T, T_neighbors, params.T_a)
        total_w = sum(weights.values())

        if total_w <= 0.0:
            di, dj = p.i, p.j
            chosen_dir = None
        else:
            dirs = list(weights.keys())
            probs = np.array([weights[d] for d in dirs], dtype=np.float64)
            probs /= probs.sum()
            choice = rng.choice(len(dirs), p=probs)
            chosen_dir = dirs[choice]
            di, dj = neighbor_map(p.i, p.j, N)[chosen_dir]
            total_moves += 1
            if chosen_dir == "up":
                moved_up += 1

        arrivals_T[di][dj].append(p.T)
        new_parcels.append(Parcel(di, dj, p.T))

    # Apply mixing at destinations to produce T_next and update parcel temps
    T_next = T.copy()
    for di in range(N):
        for dj in range(N):
            if arrivals_T[di][dj]:
                T_cell = T_next[di][dj]
                for Tin in arrivals_T[di][dj]:
                    T_cell = (1.0 - params.alpha) * T_cell + params.alpha * Tin
                T_next[di][dj] = T_cell

    # Update parcel temperatures to the mixed destination cell temperature
    final_parcels: List[Parcel] = []
    for p in new_parcels:
        p.T = T_next[p.i, p.j]
        final_parcels.append(p)

    # Diagnostics
    diag: Dict[str, float] = {}
    diag["max_T_over_Ta"] = float(T_next.max() / params.T_a)
    diag["mean_T_over_Ta"] = float(T_next.mean() / params.T_a)
    y = np.arange(N, dtype=np.float64)
    weighted_sum = (T_next * y[:, None]).sum()
    total_temp = T_next.sum()
    diag["vertical_centroid"] = float(weighted_sum / total_temp) if total_temp > 0 else float(c)
    frac_up = moved_up / total_moves if total_moves > 0 else 0.0
    diag["frac_moves_up"] = float(frac_up)
    diag["active_parcels"] = float(len(final_parcels))

    return T_next, final_parcels, diag


def run_simulation(
    params: SimParams,
    *,
    progress_cb=None,
    live_update_stride: int = 0,
    live_placeholder: Optional[st.delta_generator.DeltaGenerator] = None,
    stop_check=None,
) -> SimResults:
    """Run the persistent-parcel simulation.

    progress_cb: callable(step:int, steps:int, diag:dict) -> None
    live_update_stride: every n steps, draw a live heatmap into live_placeholder (0 disables)
    live_placeholder: st.empty() container to render live frames
    stop_check: callable() -> bool, if True at a step, break early
    """
    rng = np.random.default_rng(params.seed)
    T = init_grid(params)
    parcels: List[Parcel] = []

    snapshots: List[Tuple[int, np.ndarray]] = [(0, T.copy())]
    diagnostics: Dict[str, List[Tuple[int, float]]] = {
        "max_T_over_Ta": [(0, float(T.max() / params.T_a))],
        "mean_T_over_Ta": [(0, float(T.mean() / params.T_a))],
        "vertical_centroid": [(0, float((np.arange(params.N)[:, None] * T).sum() / T.sum()))],
        "frac_moves_up": [(0, 0.0)],
        "active_parcels": [(0, 0.0)],
    }

    for t in range(1, params.steps + 1):
        # Stop request check
        if stop_check is not None and stop_check():
            break

        T, parcels, diag = step_once_persistent(T, parcels, params, rng)
        if t % params.snapshot_stride == 0 or t == params.steps:
            snapshots.append((t, T.copy()))
        for k, v in diag.items():
            diagnostics.setdefault(k, []).append((t, v))

        # Progress callback
        if progress_cb is not None:
            progress_cb(t, params.steps, diag)

        # Live update frame
        if live_update_stride and live_placeholder is not None and (t % live_update_stride == 0):
            try:
                import matplotlib.pyplot as plt
                fig_live, ax_live = plt.subplots(figsize=(5, 5))
                im = ax_live.imshow(T / params.T_a, origin="lower", interpolation="nearest")
                ax_live.set_title(f"Live field at t = {t}")
                live_placeholder.pyplot(fig_live, clear_figure=True)
            except Exception:
                pass

    return SimResults(T=T, snapshots=snapshots, diagnostics=diagnostics, params=params)

# -----------------------------
# UI controls
# -----------------------------
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

    st.markdown("---")
    live_update_stride = st.number_input(
        "Live update every n steps",
        min_value=0,
        value=10,
        help="0 disables live plotting during a run"
    )
    stop_btn = st.button("Stop", on_click=request_stop)
    run_btn = st.button("Run simulation", type="primary")

# Store params in session state for reproducibility and reruns
if "params" not in st.session_state:
    st.session_state.params = None
if "results" not in st.session_state:
    st.session_state.results = None

if run_btn:
    # Reset stop flag at the start of a run
    st.session_state.stop_requested = False

    params = SimParams(
        N=int(N), T_a=float(T_a), k=float(k), alpha=float(alpha),
        steps=int(steps), parcels_per_step=int(parcels_per_step),
        seed=int(seed), snapshot_stride=int(snapshot_stride)
    )
    st.session_state.params = params

    # Progress UI
    prog = st.progress(0, text="Starting simulation…")
    status = st.empty()
    live_placeholder = st.empty() if live_update_stride else None

    def progress_cb(t, total, diag):
        pct = int(100 * t / total)
        txt = f"Step {t}/{total} · active parcels={int(diag['active_parcels'])} · frac_up={diag['frac_moves_up']:.2f}"
        prog.progress(pct, text=txt)
        status.write(txt)

    def stop_check():
        return st.session_state.stop_requested

    with st.spinner("Running simulation..."):
        st.session_state.results = run_simulation(
            params,
            progress_cb=progress_cb,
            live_update_stride=int(live_update_stride),
            live_placeholder=live_placeholder,
            stop_check=stop_check,
        )

    # Clear progress when done
    prog.empty()

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
        # Post-run snapshot slider for animation/scrubbing
        snap_labels = [t for t, _ in res.snapshots]
        sel = st.slider("View snapshot", 0, len(res.snapshots) - 1, len(res.snapshots) - 1)
        t_sel, T_sel = res.snapshots[sel]

        fig1, ax1 = plt.subplots(figsize=(6, 6))
        im = ax1.imshow(T_sel / res.params.T_a, origin="lower", interpolation="nearest")
        ax1.set_title(f"Field at t = {t_sel} (T/T_a)")
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
