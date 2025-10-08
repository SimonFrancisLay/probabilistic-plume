"""
Probabilistic plume model, Streamlit app
Flux (conservative) engine with exponential dT weighting, optional diagonals, gravity toggle,
vertical tilt with lateral clamp, source schedules, boundary modes, rectangular adiabatic barrier,
background diffusion, progress/live updates, snapshot slider, diagnostics, export, and UI helper refactor.

New in this version
- Source region is a centered square covering configurable percent of total cells (default 5%).
- Vertical tilt μ also clamps pure lateral moves, tightening plume angle.

Run locally:
1) pip install -r requirements.txt
2) streamlit run probabilistic_plume_app.py

requirements.txt content:
-------------------------
streamlit>=1.50
numpy>=2.3
matplotlib
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import streamlit as st

# Matplotlib
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    HAVE_MPL = True
except Exception as e:
    HAVE_MPL = False
    st.warning(f"Matplotlib not available: {e}. Install it via requirements.txt.")

st.set_page_config(page_title="Probabilistic plume model", page_icon="💨", layout="wide")

# =============================
# Parameters and results
# =============================

@dataclass
class SimParams:
    N: int = 99
    T_a: float = 20.0
    k: float = 5.0                    # source peak multiplier
    alpha: float = 0.5                # optional post-flux smoothing (not used by default)
    steps: int = 200
    parcels_per_step: int = 10        # legacy (unused by flux engine)
    seed: int = 42
    snapshot_stride: int = 10
    # Source scheduling
    source_mode: str = "Persistent"   # Persistent, Grow, Grow-plateau-decay
    tau_g: float = 20.0
    plateau_steps: int = 50
    tau_d: float = 40.0
    # Source geometry
    source_area_frac: float = 0.05    # centered square area as fraction of total cells
    # Movement model parameters
    allow_diagonals: bool = True
    epsilon_baseline: float = 0.005   # floor in directional weights
    lambda_per_Ta: float = 1.4        # λ_eff = lambda_per_Ta / T_a
    distance_penalty: bool = True     # diagonals get 1/√2 if True
    disable_gravity: bool = False     # flattens directional priors
    mu_vertical_tilt: float = 1.0     # strength of temperature dependent vertical tilt
    # Flux engine control
    beta_transfer: float = 0.3        # fraction of excess exported per step
    # Background diffusion
    epsilon_diffusion: float = 0.02   # global Moore-neighbour mixing per step
    # Boundaries
    boundary_mode: str = "Outflow"    # Outflow, Blocked, Periodic
    # Adiabatic barrier block
    barrier_enabled: bool = False
    barrier_y0: int = 50
    barrier_y1: int = 70
    barrier_x0: int = 20
    barrier_x1: int = 80


@dataclass
class SimResults:
    T: np.ndarray
    snapshots: List[Tuple[int, np.ndarray]]
    diagnostics: Dict[str, List[Tuple[int, float]]]
    params: SimParams

# =============================
# Helpers: source schedule, source region, neighbours, weights
# =============================

def source_multiplier(t: int, p: SimParams) -> float:
    """Return s(t) so that T_source = s(t) * T_a."""
    import math
    if p.source_mode == "Persistent":
        return p.k
    if p.source_mode == "Grow":
        return 1.0 + (p.k - 1.0) * (1.0 - math.exp(-max(t, 0) / max(p.tau_g, 1e-9)))
    # Grow-plateau-decay
    s_grow = 1.0 + (p.k - 1.0) * (1.0 - math.exp(-max(t, 0) / max(p.tau_g, 1e-9)))
    # Heuristic plateau start: when growth reaches ~99% of k
    t_star = 0 if p.k <= 1.0 else int(max(0.0, np.ceil(-p.tau_g * np.log(max(1e-9, (p.k - 1.0) * 0.01 / max(p.k - 1.0, 1e-9))))))
    t_plateau_end = t_star + int(max(0, p.plateau_steps))
    if t < t_star:
        return s_grow
    if t <= t_plateau_end:
        return p.k
    return 1.0 + (p.k - 1.0) * np.exp(-(t - t_plateau_end) / max(p.tau_d, 1e-9))


def source_region_coords(N: int, frac: float) -> Tuple[int, int, int, int]:
    """Return (y0, y1, x0, x1) for a centered square whose area ~ frac * N*N.
    Ensures at least 1×1 and uses odd side for symmetry about centre.
    """
    frac = float(np.clip(frac, 0.0, 1.0))
    if frac <= 0.0:
        c = N // 2
        return c, c, c, c
    side = max(1, int(round(np.sqrt(frac) * N)))
    if side % 2 == 0:
        side = min(side + 1, N)
    c = N // 2
    half = side // 2
    y0 = max(0, c - half)
    y1 = min(N - 1, c + half)
    x0 = max(0, c - half)
    x1 = min(N - 1, c + half)
    return y0, y1, x0, x1


def init_grid(params: SimParams) -> np.ndarray:
    N = params.N
    T = np.full((N, N), params.T_a, dtype=np.float64)
    y0, y1, x0, x1 = source_region_coords(N, params.source_area_frac)
    T[y0:y1+1, x0:x1+1] = params.k * params.T_a
    return T

# origin="lower": increasing i is visually up

def neighbor_map(i: int, j: int, N: int, *, allow_diagonals: bool, boundary_mode: str) -> Dict[str, Optional[Tuple[int, int]]]:
    moves = {"up": (1, 0), "down": (-1, 0), "left": (0, -1), "right": (0, 1)}
    if allow_diagonals:
        moves.update({"up_left": (1, -1), "up_right": (1, 1), "down_left": (-1, -1), "down_right": (-1, 1)})
    nbrs: Dict[str, Optional[Tuple[int, int]]] = {}
    for d, (di, dj) in moves.items():
        ii, jj = i + di, j + dj
        if boundary_mode == "Periodic":
            nbrs[d] = (ii % N, jj % N)
        elif 0 <= ii < N and 0 <= jj < N:
            nbrs[d] = (ii, jj)
        else:
            nbrs[d] = None
    return nbrs


def _direction_priors(T_particle: float, params: SimParams) -> Dict[str, float]:
    if params.disable_gravity:
        base = 1.0
        return {
            "up": base, "up_left": base, "up_right": base,
            "left": base, "right": base,
            "down": base, "down_left": base, "down_right": base,
        }
    b = T_particle / params.T_a
    return {
        "up": b,
        "up_left": 0.7 * b, "up_right": 0.7 * b,
        "left": 0.5, "right": 0.5,
        "down": 0.2, "down_left": 0.1, "down_right": 0.1,  # reduced down priors
    }


def _distance_penalty(params: SimParams) -> Dict[str, float]:
    diag = (1.0 / np.sqrt(2)) if params.distance_penalty else 1.0
    return {
        "up": 1.0, "down": 1.0, "left": 1.0, "right": 1.0,
        "up_left": diag, "up_right": diag, "down_left": diag, "down_right": diag,
    }


def compute_flux_weights(T_particle: float, T_neighbors: Dict[str, float], *, params: SimParams) -> Dict[str, float]:
    """Weights for directional flux out of a cell.
    Exponential ΔT term, directional priors, diagonal penalty, and temperature dependent vertical tilt V(T).
    Up directions × V, down and lateral directions ÷ V, where V = exp(mu * max(T/T_a - 1, 0)).
    """
    eps = max(0.0, params.epsilon_baseline)
    lam_eff = params.lambda_per_Ta / max(params.T_a, 1e-12)
    P = _direction_priors(T_particle, params)
    C = _distance_penalty(params)

    # Vertical tilt B with lateral clamp
    rel = max(T_particle / max(params.T_a, 1e-12) - 1.0, 0.0)
    V = float(np.exp(params.mu_vertical_tilt * rel))
    up_dirs = {"up", "up_left", "up_right"}
    down_dirs = {"down", "down_left", "down_right"}
    lateral_dirs = {"left", "right"}

    w: Dict[str, float] = {}
    for d, Tn in T_neighbors.items():
        if d not in P:
            continue
        dT = T_particle - Tn
        g = np.exp(lam_eff * dT) - 1.0 if dT > 0 else 0.0
        weight = (eps + g) * P[d] * C.get(d, 1.0)
        if d in up_dirs:
            weight *= V
        elif d in down_dirs and V > 0:
            weight /= V
        elif d in lateral_dirs and V > 0:
            weight /= V
        w[d] = max(0.0, float(weight))
    return w

# =============================
# Barrier helpers and defaults
# =============================

def compute_default_barrier(N: int) -> Tuple[int, int, int, int]:
    """Default rectangular barrier.
    Vertically centered halfway between centre and top; thickness = 5 percent of N (>=1).
    Horizontally spans from quarter to two thirds of width. Returns (y0, y1, x0, x1).
    """
    if N <= 0:
        return 0, 0, 0, 0
    centre = N // 2
    top = N - 1
    mid = int(round((centre + top) / 2))
    thickness = max(1, int(round(0.05 * N)))
    y0 = max(0, min(N - 1, mid - thickness // 2))
    y1 = max(0, min(N - 1, y0 + thickness - 1))
    x0 = int(round(0.25 * (N - 1)))
    x1 = int(round((2.0 / 3.0) * (N - 1)))
    x0 = max(0, min(N - 1, x0))
    x1 = max(0, min(N - 1, x1))
    if y1 < y0:
        y1 = y0
    if x1 < x0:
        x1 = x0
    return y0, y1, x0, x1


def ensure_barrier_state_defaults(N: int) -> None:
    if "prev_N" not in st.session_state:
        st.session_state.prev_N = int(N)
    missing = any(k not in st.session_state for k in ("barrier_y0", "barrier_y1", "barrier_x0", "barrier_x1"))
    if missing or st.session_state.prev_N != int(N):
        st.session_state.prev_N = int(N)
        by0, by1, bx0, bx1 = compute_default_barrier(int(N))
        st.session_state.barrier_y0 = by0
        st.session_state.barrier_y1 = by1
        st.session_state.barrier_x0 = bx0
        st.session_state.barrier_x1 = bx1


def build_barrier_mask(N: int, p: SimParams) -> Optional[np.ndarray]:
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

# =============================
# Diffusion and flux step
# =============================

def background_diffuse(T: np.ndarray, params: SimParams, barrier_mask: Optional[np.ndarray]) -> np.ndarray:
    eps = max(0.0, float(params.epsilon_diffusion))
    if eps <= 0.0:
        return T
    if params.boundary_mode == "Periodic":
        up = np.roll(T, -1, axis=0); down = np.roll(T, 1, axis=0)
        left = np.roll(T, 1, axis=1); right = np.roll(T, -1, axis=1)
        ul = np.roll(up, 1, axis=1); ur = np.roll(up, -1, axis=1)
        dl = np.roll(down, 1, axis=1); dr = np.roll(down, -1, axis=1)
        neigh_mean = (up + down + left + right + ul + ur + dl + dr) / 8.0
    else:
        Tp = np.pad(T, ((1, 1), (1, 1)), mode='edge')
        up = Tp[2:, 1:-1]; down = Tp[:-2, 1:-1]
        left = Tp[1:-1, :-2]; right = Tp[1:-1, 2:]
        ul = Tp[2:, :-2]; ur = Tp[2:, 2:]
        dl = Tp[:-2, :-2]; dr = Tp[:-2, 2:]
        neigh_mean = (up + down + left + right + ul + ur + dl + dr) / 8.0
    T_new = (1.0 - eps) * T + eps * neigh_mean
    if barrier_mask is not None:
        T_new[barrier_mask] = params.T_a
    return T_new


def flux_step(T: np.ndarray, *, params: SimParams, barrier_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, float]]:
    N = params.N
    beta = float(params.beta_transfer)
    outflow = np.zeros_like(T)
    inflow = np.zeros_like(T)
    sink_loss = 0.0

    for i in range(N):
        for j in range(N):
            if barrier_mask is not None and barrier_mask[i, j]:
                continue
            Tij = T[i, j]
            Xij = max(Tij - params.T_a, 0.0)  # export only excess over ambient
            export = beta * Xij
            if export <= 0.0:
                continue
            nbrs = neighbor_map(i, j, N, allow_diagonals=params.allow_diagonals, boundary_mode=params.boundary_mode)
            Tn: Dict[str, float] = {}
            sink_dirs: List[str] = []
            for d, idx in nbrs.items():
                if idx is None:
                    if params.boundary_mode == "Outflow":
                        Tn[d] = params.T_a  # use ambient for weight calc
                        sink_dirs.append(d)
                    continue
                ii, jj = idx
                if barrier_mask is not None and barrier_mask[ii, jj]:
                    continue
                Tn[d] = T[ii, jj]
            if not Tn:
                continue
            w = compute_flux_weights(Tij, Tn, params=params)
            total_w = sum(w.values())
            if total_w <= 0.0:
                continue
            for d, wd in w.items():
                frac = wd / total_w
                if d in sink_dirs:
                    sink_loss += export * frac
                else:
                    ii, jj = nbrs[d]
                    inflow[ii, jj] += export * frac
            outflow[i, j] += export

    T_next = T - outflow + inflow
    if barrier_mask is not None:
        T_next[barrier_mask] = params.T_a
    T_next = background_diffuse(T_next, params, barrier_mask)

    diag: Dict[str, float] = {}
    diag["max_T_over_Ta"] = float(T_next.max() / params.T_a)
    diag["mean_T_over_Ta"] = float(T_next.mean() / params.T_a)
    y = np.arange(N, dtype=np.float64)
    weighted_sum = (T_next * y[:, None]).sum(); total_temp = T_next.sum()
    diag["vertical_centroid"] = float(weighted_sum / total_temp) if total_temp > 0 else float(N // 2)
    diag["sink_loss"] = float(sink_loss)
    return T_next, diag

# =============================
# Simulation driver
# =============================

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
    barrier_mask = build_barrier_mask(params.N, params)

    snapshots: List[Tuple[int, np.ndarray]] = [(0, T.copy())]
    diagnostics: Dict[str, List[Tuple[int, float]]] = {
        "max_T_over_Ta": [(0, float(T.max() / params.T_a))],
        "mean_T_over_Ta": [(0, float(T.mean() / params.T_a))],
        "vertical_centroid": [(0, float((np.arange(params.N)[:, None] * T).sum() / max(T.sum(), 1e-12)))],
        "frac_moves_up": [(0, 0.0)],  # retained names for continuity
        "active_parcels": [(0, 0.0)],
        "sink_loss": [(0, 0.0)],
    }

    # Centre index for profiles only
    c = params.N // 2

    for t in range(1, params.steps + 1):
        if stop_check is not None and stop_check():
            break
        # Source forcing while hotter than ambient
        T_source = source_multiplier(t, params) * params.T_a
        if T_source > params.T_a + 1e-6:
            y0, y1, x0, x1 = source_region_coords(params.N, params.source_area_frac)
            if barrier_mask is not None:
                block = T[y0:y1+1, x0:x1+1]
                mask_block = barrier_mask[y0:y1+1, x0:x1+1]
                block[~mask_block] = T_source
                T[y0:y1+1, x0:x1+1] = block
            else:
                T[y0:y1+1, x0:x1+1] = T_source

        T, diag = flux_step(T, params=params, barrier_mask=barrier_mask)
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

# =============================
# UI helper components
# =============================

def barrier_controls(N: int) -> Tuple[bool, int, int, int, int]:
    st.subheader("Adiabatic barrier block")
    barrier_enabled = st.checkbox(
        "Enable rectangular barrier",
        value=False,
        help="Internal block that does not receive heat and cannot be entered",
    )
    y0 = st.number_input(
        "Barrier bottom row i0 (0=bottom)",
        min_value=0, max_value=int(N - 1),
        value=int(st.session_state.barrier_y0),
    )
    y1 = st.number_input(
        "Barrier top row i1",
        min_value=0, max_value=int(N - 1),
        value=int(st.session_state.barrier_y1),
    )
    x0 = st.number_input(
        "Barrier left col j0",
        min_value=0, max_value=int(N - 1),
        value=int(st.session_state.barrier_x0),
    )
    x1 = st.number_input(
        "Barrier right col j1",
        min_value=0, max_value=int(N - 1),
        value=int(st.session_state.barrier_x1),
    )
    return bool(barrier_enabled), int(y0), int(y1), int(x0), int(x1)


def color_and_diffusion_controls() -> Tuple[str, float, float]:
    st.subheader("Color mapping")
    cmap_name = st.selectbox(
        "Colormap",
        ["inferno", "magma", "viridis", "plasma", "cividis"],
        index=0,
        help="Perceptually uniform maps give smooth gradation",
    )
    gamma = st.slider(
        "Contrast (gamma)",
        min_value=0.3, max_value=2.0, value=1.0, step=0.1,
        help="Less than 1 brightens warm regions; greater than 1 compresses highlights",
    )
    st.subheader("Background diffusion")
    epsilon_diffusion = st.slider(
        "Background diffusion ε",
        min_value=0.0, max_value=0.2, value=0.02, step=0.005,
        help="Moore neighbour mixing per step. Set to 0 to disable.",
    )
    return str(cmap_name), float(gamma), float(epsilon_diffusion)

# =============================
# Sidebar
# =============================

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

def request_stop():
    st.session_state.stop_requested = True

st.title("Probabilistic buoyant plume: flux model")
st.caption("Conservative flux transport with exponential dT weighting, vertical tilt, and extended source region.")

with st.sidebar:
    st.header("Controls")
    N   = st.number_input("Grid size N", min_value=21, max_value=401, value=99, step=2, help="Prefer odd so the centre is unique")
    T_a = st.number_input("Ambient temperature T_a", min_value=0.1, value=20.0)
    k   = st.number_input("Source peak multiplier k", min_value=1.0, value=5.0)

    source_mode = st.selectbox("Source profile", ["Persistent", "Grow", "Grow-plateau-decay"], index=0)
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        tau_g = st.number_input("Growth tau (steps)", min_value=1.0, value=20.0)
    with col_s2:
        plateau_steps = st.number_input("Plateau steps", min_value=0, value=50)
    with col_s3:
        tau_d = st.number_input("Decay tau (steps)", min_value=1.0, value=40.0)

    source_area_pct = st.slider(
        "Source area (% of grid cells)",
        min_value=0.5, max_value=25.0, value=5.0, step=0.5,
        help="The source is a centered square covering this percent of total cells",
    )

    alpha = st.slider("Mixing fraction alpha (optional)", 0.0, 1.0, 0.5, help="Only used if you later enable post-flux smoothing")

    st.subheader("Movement model")
    allow_diagonals  = st.checkbox("Allow diagonal moves", value=True)
    epsilon_baseline = st.slider("Baseline eps (floor)", 0.0, 0.05, 0.005, step=0.001)
    lambda_per_Ta    = st.slider("dT sensitivity lambda (per T_a)", 0.0, 3.0, 1.4, step=0.05)
    distance_penalty = st.checkbox("Distance penalty for diagonals (1/sqrt(2))", value=True)
    disable_gravity  = st.checkbox("Disable gravity bias", value=False)

    st.subheader("Flux transfer")
    beta_transfer = st.slider("Export fraction per step β", 0.0, 0.8, 0.3, 0.05)
    mu_vertical_tilt = st.slider("Vertical tilt μ", 0.0, 3.0, 1.0, 0.1, help="Up weights ×V, down/lateral weights ÷V as T rises above T_a")

    boundary_mode = st.selectbox("Boundary mode", ["Outflow", "Blocked", "Periodic"], index=0)

    # Ensure barrier defaults now that N is known
    ensure_barrier_state_defaults(int(N))
    barrier_enabled, barrier_y0, barrier_y1, barrier_x0, barrier_x1 = barrier_controls(int(N))

    cmap_name, gamma, epsilon_diffusion = color_and_diffusion_controls()

    steps            = st.number_input("Time steps", min_value=1, value=200)
    parcels_per_step = st.number_input("Parcels per step r (legacy)", min_value=1, value=10)
    seed             = st.number_input("Random seed", min_value=0, value=42)
    snapshot_stride  = st.number_input("Snapshot stride", min_value=1, value=10)

    st.markdown("---")
    live_update_stride = st.number_input("Live update every n steps", min_value=0, value=10)
    stop_btn = st.button("Stop", on_click=request_stop)
    run_btn  = st.button("Run simulation", type="primary")

# Session storage
if "params" not in st.session_state:
    st.session_state.params = None
if "results" not in st.session_state:
    st.session_state.results = None

if run_btn:
    st.session_state.stop_requested = False
    params = SimParams(
        N=int(N), T_a=float(T_a), k=float(k), alpha=float(alpha),
        steps=int(steps), parcels_per_step=int(parcels_per_step),
        seed=int(seed), snapshot_stride=int(snapshot_stride),
        source_mode=str(source_mode), tau_g=float(tau_g), plateau_steps=int(plateau_steps), tau_d=float(tau_d),
        source_area_frac=float(source_area_pct) / 100.0,
        allow_diagonals=bool(allow_diagonals), epsilon_baseline=float(epsilon_baseline),
        lambda_per_Ta=float(lambda_per_Ta), distance_penalty=bool(distance_penalty), disable_gravity=bool(disable_gravity),
        mu_vertical_tilt=float(mu_vertical_tilt),
        beta_transfer=float(beta_transfer), boundary_mode=str(boundary_mode), epsilon_diffusion=float(epsilon_diffusion),
        barrier_enabled=bool(barrier_enabled), barrier_y0=int(barrier_y0), barrier_y1=int(barrier_y1), barrier_x0=int(barrier_x0), barrier_x1=int(barrier_x1),
    )
    st.session_state.params = params

    prog = st.progress(0, text="Starting simulation…")
    status = st.empty()
    live_placeholder = st.empty() if live_update_stride else None

    def progress_cb(t, total, diag):
        pct = int(100 * t / total)
        txt = f"Step {t}/{total} · sink_loss={diag.get('sink_loss', 0.0):.3f}"
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
    prog.empty()

# =============================
# Visuals
# =============================

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
        sel = st.slider("View snapshot", 0, len(res.snapshots) - 1, len(res.snapshots) - 1)
        t_sel, T_sel = res.snapshots[sel]

        barrier_mask_plot = build_barrier_mask(res.params.N, res.params)
        data = (T_sel / res.params.T_a).copy()
        if barrier_mask_plot is not None:
            data = np.ma.array(data, mask=barrier_mask_plot)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color="black")
        norm = PowerNorm(gamma=gamma)

        fig1, ax1 = plt.subplots(figsize=(6, 6))
        im = ax1.imshow(data, origin="lower", interpolation="nearest", cmap=cmap, norm=norm)
        ax1.set_title(f"Field at t = {t_sel} (T/T_a)")
        ax1.set_xlabel("x index"); ax1.set_ylabel("y index")
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="T/T_a")
        with col1:
            st.pyplot(fig1, clear_figure=True)
        plt.close(fig1)

        # Centreline profile above centre
        c = res.params.N // 2
        profile = T_sel[c:, c] / res.params.T_a
        y = np.arange(profile.size)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(profile, y)
        ax2.set_xlabel("T/T_a"); ax2.set_ylabel("vertical distance above centre")
        ax2.set_title("Centreline profile above source")
        with col2:
            st.pyplot(fig2, clear_figure=True)
        plt.close(fig2)

        # Diagnostics
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        for key in ["max_T_over_Ta", "mean_T_over_Ta", "vertical_centroid", "sink_loss"]:
            ts = np.array(res.diagnostics.get(key, []))
            if ts.size:
                ax3.plot(ts[:, 0], ts[:, 1], label=key)
        ax3.set_xlabel("time step"); ax3.set_ylabel("value"); ax3.set_title("Diagnostics time series"); ax3.legend()
        with col2:
            st.pyplot(fig3, clear_figure=True)
        plt.close(fig3)

# =============================
# Export
# =============================

if res is not None:
    st.subheader("Export")
    snaps = {f"t{t}": arr for t, arr in res.snapshots}
    try:
        import io
        buf = io.BytesIO()
        np.savez_compressed(buf, **snaps)
        st.download_button("Download snapshots (npz)", data=buf.getvalue(), file_name="plume_snapshots.npz", mime="application/zip")
    except Exception as e:
        st.warning(f"Could not package snapshots: {e}")

# ---
# Legacy parcel engine retained below as comments for reference
# def step_once_persistent(...):
#     ... parcel implementation ...
# def run_simulation(...):
#     ... parcel based loop ...
