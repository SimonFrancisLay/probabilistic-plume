"""
Probabilistic plume model, Streamlit app
Persistent parcels, exponential dT movement with optional diagonals, scheduled source profiles,
live progress and updates, snapshot slider, configurable boundary modes, rectangular adiabatic barrier,
colormap controls, and a gravity toggle.

Run locally:
1) pip install -r requirements.txt
2) streamlit run probabilistic_plume_app.py

requirements.txt content:
-------------------------
streamlit>=1.50
numpy>=2.3
matplotlib
# Optional later
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
    from matplotlib.colors import PowerNorm
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
    k: float = 5.0            # peak or source multiplier
    alpha: float = 0.5        # mixing fraction on arrival
    steps: int = 200
    parcels_per_step: int = 10
    seed: int = 42
    snapshot_stride: int = 10 # snapshot every this many steps
    # Source scheduling
    source_mode: str = "Persistent"  # options: Persistent, Grow, Grow-plateau-decay
    tau_g: float = 20.0              # growth time constant in steps
    plateau_steps: int = 50          # plateau duration in steps
    tau_d: float = 40.0              # decay time constant in steps
    # Movement model parameters, exponential dT weighting
    allow_diagonals: bool = True
    epsilon_baseline: float = 0.005  # small floor so motion never stalls
    lambda_per_Ta: float = 1.4       # effective lambda is lambda_per_Ta / T_a
    distance_penalty: bool = True    # multiply diagonal weights by 1/sqrt(2)
    disable_gravity: bool = False    # if True, remove directional buoyancy bias
    # Boundary handling
    boundary_mode: str = "Outflow"   # options: Outflow, Blocked, Periodic
    # Adiabatic barrier block
    barrier_enabled: bool = False
    barrier_y0: int = 50   # bottom row i, inclusive 0..N-1, origin lower so larger is higher
    barrier_x0: int = 20   # left col j, inclusive
    barrier_y1: int = 70   # top row i, inclusive
    barrier_x1: int = 80   # right col j, inclusive


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
# Core model helpers, persistent engine with exponential dT movement
# -----------------------------

def init_grid(params: SimParams) -> np.ndarray:
    """Create initial temperature field with a hot source at the centre."""
    N = params.N
    T = np.full((N, N), params.T_a, dtype=np.float64)
    c = N // 2
    T[c, c] = params.k * params.T_a
    return T

# Display uses origin lower, so increasing row index i moves visually up.
# Define up as (i + 1, j) so that up in the model matches the heatmap.

def neighbor_map(
    i: int,
    j: int,
    N: int,
    *,
    allow_diagonals: bool = True,
    boundary_mode: str = "Outflow",
) -> Dict[str, Optional[Tuple[int, int]]]:
    moves = {"up": (1, 0), "down": (-1, 0), "left": (0, -1), "right": (0, 1)}
    if allow_diagonals:
        moves.update({
            "up_left": (1, -1), "up_right": (1, 1),
            "down_left": (-1, -1), "down_right": (-1, 1),
        })

    nbrs: Dict[str, Optional[Tuple[int, int]]] = {}
    for d, (di, dj) in moves.items():
        ii, jj = i + di, j + dj
        if boundary_mode == "Periodic":
            ii %= N
            jj %= N
            nbrs[d] = (ii, jj)
            continue
        if 0 <= ii < N and 0 <= jj < N:
            nbrs[d] = (ii, jj)
        else:
            # Blocked and Outflow both mark as None here. Outflow is handled in the step logic.
            nbrs[d] = None
    return nbrs


def compute_move_weights_exp(T_particle: float, T_neighbors: Dict[str, float], params: SimParams) -> Dict[str, float]:
    """Exponential dT weighting with directional priors and optional distance penalty.
    w_d = [eps + max(exp(lambda * dT) - 1, 0)] * P_dir(d) * C_dist(d),  dT = T_p - T_n
    Direction priors favour up, then up diagonals, then lateral, then down. Can be disabled with disable_gravity.
    """
    eps = max(0.0, params.epsilon_baseline)
    lam_eff = params.lambda_per_Ta / max(params.T_a, 1e-12)

    if params.disable_gravity:
        base = 1.0
        P_dir = {
            "up": base, "up_left": base, "up_right": base,
            "left": base, "right": base,
            "down": base, "down_left": base, "down_right": base,
        }
    else:
        b = T_particle / params.T_a  # buoyancy factor
        P_dir = {
            "up": b,
            "up_left": 0.7 * b,
            "up_right": 0.7 * b,
            "left": 0.5,
            "right": 0.5,
            "down": 0.3,
            "down_left": 0.2,
            "down_right": 0.2,
        }
    C_dist = {
        "up": 1.0, "down": 1.0, "left": 1.0, "right": 1.0,
        "up_left": (1.0 / np.sqrt(2)) if params.distance_penalty else 1.0,
        "up_right": (1.0 / np.sqrt(2)) if params.distance_penalty else 1.0,
        "down_left": (1.0 / np.sqrt(2)) if params.distance_penalty else 1.0,
        "down_right": (1.0 / np.sqrt(2)) if params.distance_penalty else 1.0,
    }

    weights: Dict[str, float] = {}
    for d, Tn in T_neighbors.items():
        if d not in P_dir:
            continue
        dT = T_particle - Tn
        g = np.exp(lam_eff * dT) - 1.0 if dT > 0 else 0.0
        w = (eps + g) * P_dir[d] * C_dist.get(d, 1.0)
        weights[d] = max(0.0, float(w))

    # Add a small stay option so we never run out of probability mass
    weights["stay"] = eps
    return weights


@dataclass
class Parcel:
    i: int
    j: int
    T: float


def source_multiplier(t: int, p: SimParams) -> float:
    """Return source temperature multiplier s(t) so that T_source(t) = s(t) * T_a."""
    import math
    if p.source_mode == "Persistent":
        return p.k
    if p.source_mode == "Grow":
        return 1.0 + (p.k - 1.0) * (1.0 - math.exp(-max(t, 0) / max(p.tau_g, 1e-9)))
    # Grow, plateau, decay
    s_grow = 1.0 + (p.k - 1.0) * (1.0 - math.exp(-max(t, 0) / max(p.tau_g, 1e-9)))
    if p.k <= 1.0:
        t_plateau_start = 0
    else:
        t_star = -p.tau_g * math.log(max(1e-9, (p.k - 1.0) * 0.01 / max(p.k - 1.0, 1e-9)))
        t_plateau_start = int(max(0.0, math.ceil(t_star)))
    t_plateau_end = t_plateau_start + int(max(0, p.plateau_steps))
    if t < t_plateau_start:
        return s_grow
    if t <= t_plateau_end:
        return p.k
    td = max(p.tau_d, 1e-9)
    return 1.0 + (p.k - 1.0) * math.exp(-(t - t_plateau_end) / td)


def build_barrier_mask(N: int, p: SimParams) -> Optional[np.ndarray]:
    """Return a boolean mask of shape (N, N) where True marks adiabatic barrier cells.
    Rectangle from (y0, x0) to (y1, x1) inclusive. If disabled or empty, return None.
    """
    if not p.barrier_enabled:
        return None

    y0 = int(np.clip(min(p.barrier_y0, p.barrier_y1), 0, N-1))
    y1 = int(np.clip(max(p.barrier_y0, p.barrier_y1), 0, N-1))
    x0 = int(np.clip(min(p.barrier_x0, p.barrier_x1), 0, N-1))
    x1 = int(np.clip(max(p.barrier_x0, p.barrier_x1), 0, N-1))

    if y1 < y0 or x1 < x0:
        return None

    mask = np.zeros((N, N), dtype=bool)
    mask[y0:y1+1, x0:x1+1] = True
    return mask


def compute_default_barrier(N: int) -> Tuple[int, int, int, int]:
    """Default rectangular barrier based on N.
    Vertically: centred halfway between centre and top, thickness = 5 percent of N (at least 1).
    Horizontally: spans from 1/4 to 2/3 of the width.
    Returns (y0, y1, x0, x1), clamped to grid.
    """
    if N <= 0:
        return 0, 0, 0, 0
    # Vertical placement
    centre = N // 2
    top = N - 1
    centre_to_top_mid = int(round((centre + top) / 2))
    thickness = max(1, int(round(0.05 * N)))
    y0 = centre_to_top_mid - thickness // 2
    y1 = y0 + thickness - 1
    y0 = max(0, min(N - 1, y0))
    y1 = max(0, min(N - 1, y1))
    if y1 < y0:
        y1 = y0
    # Horizontal span
    x0 = int(round(0.25 * (N - 1)))
    x1 = int(round((2.0 / 3.0) * (N - 1)))
    x0 = max(0, min(N - 1, x0))
    x1 = max(0, min(N - 1, x1))
    if x1 < x0:
        x1 = x0
    return y0, y1, x0, x1


def step_once_persistent(
    T: np.ndarray,
    parcels: List[Parcel],
    params: SimParams,
    rng: np.random.Generator,
    T_source: float,
    barrier_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Parcel], Dict[str, float]]:
    N = params.N
    c = N // 2

    # Clamp source cell and inject r new parcels only if source is hotter than ambient
    if barrier_mask is not None and barrier_mask[c, c]:
        st.warning("Barrier overlaps the source cell; source cell will be exempt from the barrier.")
        barrier_mask = barrier_mask.copy()
        barrier_mask[c, c] = False
    if T_source > params.T_a + 1e-6:
        T[c, c] = T_source
        for _ in range(params.parcels_per_step):
            parcels.append(Parcel(c, c, T_source))

    # Arrival buckets per destination cell
    arrivals_T: List[List[List[float]]] = [[[] for _ in range(N)] for __ in range(N)]

    moved_up = 0
    total_moves = 0

    new_parcels: List[Parcel] = []
    for p in parcels:
        nbrs = neighbor_map(
            p.i, p.j, N,
            allow_diagonals=params.allow_diagonals,
            boundary_mode=params.boundary_mode,
        )
        # Build temperature map from in-bounds neighbours, skipping barrier cells
        T_neighbors: Dict[str, float] = {}
        for d, idx in nbrs.items():
            if idx is None:
                continue
            ii, jj = idx
            if barrier_mask is not None and barrier_mask[ii, jj]:
                continue
            T_neighbors[d] = T[ii, jj]
        # For Outflow, create synthetic neighbours for off-grid directions using ambient
        if params.boundary_mode == "Outflow":
            for d, idx in nbrs.items():
                if idx is None:
                    T_neighbors[d] = params.T_a

        weights = compute_move_weights_exp(p.T, T_neighbors, params)
        dirs = list(weights.keys())
        probs = np.array([weights[d] for d in dirs], dtype=np.float64)
        s = probs.sum()
        if s <= 0:
            chosen_dir = "stay"
        else:
            probs /= s
            choice = rng.choice(len(dirs), p=probs)
            chosen_dir = dirs[choice]

        if chosen_dir == "stay":
            di, dj = p.i, p.j
            arrivals_T[di][dj].append(p.T)
            new_parcels.append(Parcel(di, dj, p.T))
            continue

        dest = nbrs.get(chosen_dir, None)
        # If destination is a barrier cell, treat as blocked
        if dest is not None and barrier_mask is not None and barrier_mask[dest[0], dest[1]]:
            dest = None

        if dest is None:
            if params.boundary_mode == "Outflow":
                # Parcel leaves the domain
                total_moves += 1
                continue
            else:
                # Blocked: treat as stay
                di, dj = p.i, p.j
        else:
            di, dj = dest
            total_moves += 1
            if str(chosen_dir).startswith("up"):
                moved_up += 1

        arrivals_T[di][dj].append(p.T)
        new_parcels.append(Parcel(di, dj, p.T))

    # Mix arrivals to produce T_next and update parcel temperatures
    T_next = T.copy()
    for di in range(N):
        for dj in range(N):
            if barrier_mask is not None and barrier_mask[di, dj]:
                # adiabatic barrier holds ambient and ignores arrivals
                T_next[di, dj] = params.T_a
                continue
            if arrivals_T[di][dj]:
                T_cell = T_next[di][dj]
                for Tin in arrivals_T[di][dj]:
                    T_cell = (1.0 - params.alpha) * T_cell + params.alpha * Tin
                T_next[di][dj] = T_cell

    # Update parcels to mixed destination cell temperature
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
    rng = np.random.default_rng(params.seed)
    T = init_grid(params)
    parcels: List[Parcel] = []

    # Build adiabatic barrier mask once per run
    barrier_mask = build_barrier_mask(params.N, params)

    snapshots: List[Tuple[int, np.ndarray]] = [(0, T.copy())]
    diagnostics: Dict[str, List[Tuple[int, float]]] = {
        "max_T_over_Ta": [(0, float(T.max() / params.T_a))],
        "mean_T_over_Ta": [(0, float(T.mean() / params.T_a))],
        "vertical_centroid": [(0, float((np.arange(params.N)[:, None] * T).sum() / T.sum()))],
        "frac_moves_up": [(0, 0.0)],
        "active_parcels": [(0, 0.0)],
    }

    for t in range(1, params.steps + 1):
        if stop_check is not None and stop_check():
            break

        T_source = source_multiplier(t, params) * params.T_a
        T, parcels, diag = step_once_persistent(T, parcels, params, rng, T_source, barrier_mask)

        if t % params.snapshot_stride == 0 or t == params.steps:
            snapshots.append((t, T.copy()))
        for k, v in diag.items():
            diagnostics.setdefault(k, []).append((t, v))

        if progress_cb is not None:
            progress_cb(t, params.steps, diag)

        if live_update_stride and live_placeholder is not None and (t % live_update_stride == 0):
            try:
                fig_live, ax_live = plt.subplots(figsize=(5, 5))
                ax_live.imshow(T / params.T_a, origin="lower", interpolation="nearest")
                ax_live.set_title(f"Live field at t = {t}")
                live_placeholder.pyplot(fig_live, clear_figure=True)
                plt.close(fig_live)
            except Exception:
                pass

    return SimResults(T=T, snapshots=snapshots, diagnostics=diagnostics, params=params)

# -----------------------------
# UI controls
# -----------------------------

st.title("Probabilistic buoyant plume: minimal model")
st.caption("Persistent parcels with scheduled source and exponential dT movement. No background diffusion.")

with st.sidebar:
    st.header("Controls")
    N   = st.number_input("Grid size N", min_value=21, max_value=401, value=99, step=2, help="Prefer odd so the centre is unique")
    T_a = st.number_input("Ambient temperature T_a", min_value=0.1, value=20.0)
    k   = st.number_input("Source peak multiplier k", min_value=1.0, value=5.0)

    source_mode = st.selectbox("Source profile", ["Persistent", "Grow", "Grow-plateau-decay"], index=0, help="Schedule for source temperature at the centre")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        tau_g = st.number_input("Growth tau (steps)", min_value=1.0, value=20.0)
    with col_s2:
        plateau_steps = st.number_input("Plateau steps", min_value=0, value=50)
    with col_s3:
        tau_d = st.number_input("Decay tau (steps)", min_value=1.0, value=40.0)

    alpha = st.slider("Mixing fraction alpha", 0.0, 1.0, 0.5)

    st.subheader("Movement model")
    allow_diagonals  = st.checkbox("Allow diagonal moves", value=True)
    epsilon_baseline = st.slider("Baseline eps (floor)", 0.0, 0.05, 0.005, step=0.001, help="Small floor so motion never stalls")
    lambda_per_Ta    = st.slider("dT sensitivity lambda (per T_a)", 0.0, 3.0, 1.4, step=0.05, help="Effective lambda is lambda/T_a")
    distance_penalty = st.checkbox("Distance penalty for diagonals (1/sqrt(2))", value=True)
    disable_gravity  = st.checkbox("Disable gravity bias", value=False, help="Treat all directions equally; only dT and distance penalty apply")

    boundary_mode = st.selectbox("Boundary mode", ["Outflow", "Blocked", "Periodic"], index=0, help="Outflow removes parcels that step out; Blocked ignores off-grid moves; Periodic wraps around")

    # --- Auto default barrier whenever N changes or first load ---
    if "prev_N" not in st.session_state:
        st.session_state.prev_N = int(N)
        by0, by1, bx0, bx1 = compute_default_barrier(int(N))
        st.session_state.barrier_y0 = by0
        st.session_state.barrier_y1 = by1
        st.session_state.barrier_x0 = bx0
        st.session_state.barrier_x1 = bx1
    elif st.session_state.prev_N != int(N):
        st.session_state.prev_N = int(N)
        by0, by1, bx0, bx1 = compute_default_barrier(int(N))
        st.session_state.barrier_y0 = by0
        st.session_state.barrier_y1 = by1
        st.session_state.barrier_x0 = bx0
        st.session_state.barrier_x1 = bx1

    st.subheader("Adiabatic barrier block")
    barrier_enabled = st.checkbox("Enable rectangular barrier", value=False, help="Internal block that does not receive heat and cannot be entered")
    # Use session defaults that auto update on N change, but remain user editable afterwards
    barrier_y0 = st.number_input("Barrier bottom row i0 (0=bottom)", min_value=0, max_value=int(N-1), value=int(st.session_state.barrier_y0))
    barrier_y1 = st.number_input("Barrier top row i1",                 min_value=0, max_value=int(N-1), value=int(st.session_state.barrier_y1))
    barrier_x0 = st.number_input("Barrier left col j0",                min_value=0, max_value=int(N-1), value=int(st.session_state.barrier_x0))
    barrier_x1 = st.number_input("Barrier right col j1",               min_value=0, max_value=int(N-1), value=int(st.session_state.barrier_x1))

    st.subheader("Color mapping")
    cmap_name = st.selectbox("Colormap", ["inferno", "magma", "viridis", "plasma", "cividis"], index=0, help="Perceptually uniform maps give smooth gradation")
    gamma = st.slider("Contrast (gamma)", min_value=0.3, max_value=2.0, value=1.0, step=0.1, help="Less than 1 brightens warm regions, greater than 1 compresses highlights")

    steps            = st.number_input("Time steps", min_value=1, value=200)
    parcels_per_step = st.number_input("Parcels per step r", min_value=1, value=10)
    seed             = st.number_input("Random seed", min_value=0, value=42)
    snapshot_stride  = st.number_input("Snapshot stride", min_value=1, value=10)

    st.markdown("---")
    live_update_stride = st.number_input("Live update every n steps", min_value=0, value=10, help="Zero disables live plotting during a run")
    stop_btn = st.button("Stop", on_click=request_stop)
    run_btn  = st.button("Run simulation", type="primary")

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
        seed=int(seed), snapshot_stride=int(snapshot_stride),
        source_mode=str(source_mode), tau_g=float(tau_g), plateau_steps=int(plateau_steps), tau_d=float(tau_d),
        allow_diagonals=bool(allow_diagonals), epsilon_baseline=float(epsilon_baseline),
        lambda_per_Ta=float(lambda_per_Ta), distance_penalty=bool(distance_penalty), disable_gravity=bool(disable_gravity),
        boundary_mode=str(boundary_mode),
        barrier_enabled=bool(barrier_enabled), barrier_y0=int(barrier_y0), barrier_x0=int(barrier_x0), barrier_y1=int(barrier_y1), barrier_x1=int(barrier_x1),
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
        # Post run snapshot slider
        sel = st.slider("View snapshot", 0, len(res.snapshots) - 1, len(res.snapshots) - 1)
        t_sel, T_sel = res.snapshots[sel]

        # Rebuild barrier mask for this run for plotting overlay
        barrier_mask_plot = build_barrier_mask(res.params.N, res.params)

        # Prepare data with barrier masked so it shows as black
        data = (T_sel / res.params.T_a).copy()
        if barrier_mask_plot is not None:
            data = np.ma.array(data, mask=barrier_mask_plot)

        # Build colormap with adjustable contrast and black for masked cells
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color="black")
        norm = PowerNorm(gamma=gamma)

        # Heatmap for selected snapshot
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        im = ax1.imshow(data, origin="lower", interpolation="nearest", cmap=cmap, norm=norm)
        ax1.set_title(f"Field at t = {t_sel} (T/T_a)")
        ax1.set_xlabel("x index")
        ax1.set_ylabel("y index")
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="T/T_a")
        with col1:
            st.pyplot(fig1, clear_figure=True)
        plt.close(fig1)

        # Centreline profile above the source, from centre upward
        c = res.params.N // 2
        profile = T_sel[c:, c] / res.params.T_a
        y = np.arange(profile.size)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(profile, y)
        ax2.set_xlabel("T/T_a")
        ax2.set_ylabel("vertical distance above centre")
        ax2.set_title("Centreline profile above source")
        with col2:
            st.pyplot(fig2, clear_figure=True)
        plt.close(fig2)

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
        plt.close(fig3)

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
    Notes: persistent parcels with scheduled source options (persistent, grow, grow plateau decay) and exponential dT movement with optional diagonals. No global diffusion in this version. Supports Outflow, Blocked, or Periodic boundaries, plus an optional adiabatic rectangular barrier. Includes a gravity toggle to remove directional priors if desired.
    """
)
