"""
Probabilistic plume model, Streamlit app
Stochastic token flux engine with A/B sampling toggle:
A) Per token categorical sampling
B) Single-step multinomial allocation

Now hardened for large grids and hot cores:
- Overflow safe exponentials in weight and tilt calculations
- Probability vectors sanitised before sampling
- Fixed colour scale in live and snapshot views (T/T_a from 1 to k)

Features
- Exponential dT weighting with directional priors and distance penalty
- Temperature dependent vertical tilt with lateral clamp
  up: ×V, up-diagonals: ×0.7 V, down and lateral: ÷V, where V = exp(mu * max(T/T_a - 1, 0))
- Source schedules and a centered source strip one cell tall, spanning a percent of width
- Boundary modes: Outflow, Blocked, Periodic
- Rectangular adiabatic barrier mask
- Optional background diffusion after stochastic transfer
- Progress bar, live updates, snapshots, diagnostics, export

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
    k: float = 5.0
    steps: int = 200
    seed: int = 42
    snapshot_stride: int = 10
    # Source schedule
    source_mode: str = "Persistent"   # Persistent, Grow, Grow-plateau-decay
    tau_g: float = 20.0
    plateau_steps: int = 50
    tau_d: float = 40.0
    # Source geometry: centered horizontal strip one cell tall
    source_span_frac: float = 0.05     # percent of width as fraction 0..1
    # Movement model parameters
    allow_diagonals: bool = True
    epsilon_baseline: float = 0.005
    lambda_per_Ta: float = 1.8
    distance_penalty: bool = True
    disable_gravity: bool = False
    mu_vertical_tilt: float = 1.5
    # Stochastic token engine
    eta_token_quanta: float = 1e-3     # token size q = eta * T_a
    sampling_mode: str = "Per token"   # "Per token" or "Multinomial"
    # Background diffusion
    epsilon_diffusion: float = 0.01
    # Boundaries
    boundary_mode: str = "Outflow"     # Outflow, Blocked, Periodic
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
# Helpers: source schedule, source strip, neighbours, weights
# =============================

def source_multiplier(t: int, p: SimParams) -> float:
    import math
    if p.source_mode == "Persistent":
        return p.k
    if p.source_mode == "Grow":
        return 1.0 + (p.k - 1.0) * (1.0 - math.exp(-max(t, 0) / max(p.tau_g, 1e-9)))
    # Grow-plateau-decay
    s_grow = 1.0 + (p.k - 1.0) * (1.0 - math.exp(-max(t, 0) / max(p.tau_g, 1e-9)))
    t_star = 0 if p.k <= 1.0 else int(max(0.0, np.ceil(-p.tau_g * np.log(max(1e-9, (p.k - 1.0) * 0.01 / max(p.k - 1.0, 1e-9))))))
    t_plateau_end = t_star + int(max(0, p.plateau_steps))
    if t < t_star:
        return s_grow
    if t <= t_plateau_end:
        return p.k
    return 1.0 + (p.k - 1.0) * np.exp(-(t - t_plateau_end) / max(p.tau_d, 1e-9))


def source_region_coords(N: int, frac: float) -> Tuple[int, int, int, int]:
    """Centered horizontal strip one cell tall, spanning about frac * N columns."""
    frac = float(np.clip(frac, 0.0, 1.0))
    c = N // 2
    span = max(1, int(round(frac * N)))
    if span % 2 == 0:
        span = min(span + 1, N)
    half = span // 2
    y0 = y1 = c
    x0 = max(0, c - half)
    x1 = min(N - 1, c + half)
    return y0, y1, x0, x1


def init_grid(params: SimParams) -> np.ndarray:
    N = params.N
    T = np.full((N, N), params.T_a, dtype=np.float64)
    y0, y1, x0, x1 = source_region_coords(N, params.source_span_frac)
    T[y0:y1+1, x0:x1+1] = params.k * params.T_a
    return T

# origin lower: increasing i is visually up

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


def _direction_priors(Tp: float, p: SimParams) -> Dict[str, float]:
    if p.disable_gravity:
        base = 1.0
        return {"up": base, "up_left": base, "up_right": base, "left": base, "right": base, "down": base, "down_left": base, "down_right": base}
    b = Tp / p.T_a
    return {"up": b, "up_left": 0.7 * b, "up_right": 0.7 * b, "left": 0.5, "right": 0.5, "down": 0.2, "down_left": 0.1, "down_right": 0.1}


def _distance_penalty(p: SimParams) -> Dict[str, float]:
    diag = (1.0 / np.sqrt(2)) if p.distance_penalty else 1.0
    return {"up": 1.0, "down": 1.0, "left": 1.0, "right": 1.0, "up_left": diag, "up_right": diag, "down_left": diag, "down_right": diag}


def compute_flux_weights(Tp: float, Tn: Dict[str, float], *, params: SimParams) -> Dict[str, float]:
    """Directional weights, overflow safe.
    Up ×V, up-diagonals ×0.7 V, down and lateral ÷V. Uses expm1 and clamps exponent args.
    """
    eps = max(0.0, params.epsilon_baseline)
    lam_eff = params.lambda_per_Ta / max(params.T_a, 1e-12)

    # Vertical tilt with clamp
    rel = max(Tp / max(params.T_a, 1e-12) - 1.0, 0.0)
    tilt_arg = np.clip(params.mu_vertical_tilt * rel, 0.0, 50.0)
    V = float(np.exp(tilt_arg))

    P = _direction_priors(Tp, params)
    C = _distance_penalty(params)

    up_dirs       = {"up"}
    up_diag_dirs  = {"up_left", "up_right"}
    down_dirs     = {"down", "down_left", "down_right"}
    lateral_dirs  = {"left", "right"}

    w: Dict[str, float] = {}
    for d, Tdj in Tn.items():
        if d not in P:
            continue
        dT = Tp - Tdj
        # Only positive gradient contributes
        arg = np.clip(lam_eff * max(dT, 0.0), 0.0, 50.0)
        g = np.expm1(arg)  # stable exp(arg) - 1
        weight = (eps + g) * P[d] * C.get(d, 1.0)

        if d in up_dirs:
            weight *= V
        elif d in up_diag_dirs:
            weight *= 0.7 * V
        elif d in down_dirs and V > 0:
            weight /= V
        elif d in lateral_dirs and V > 0:
            weight /= V

        if not np.isfinite(weight) or weight < 0.0:
            weight = 0.0
        w[d] = float(weight)
    return w

# =============================
# Barrier helpers and defaults
# =============================

def compute_default_barrier(N: int) -> Tuple[int, int, int, int]:
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
# Background diffusion
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

# =============================
# Stochastic token flux step (A or B)
# =============================

def stochastic_flux_step(T: np.ndarray, *, params: SimParams, barrier_mask: Optional[np.ndarray], rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
    N = params.N
    q = max(1e-12, params.eta_token_quanta * params.T_a)  # token heat
    use_multinomial = (params.sampling_mode == "Multinomial")

    inflow = np.zeros_like(T)
    out_tokens = 0
    sink_tokens = 0

    for i in range(N):
        for j in range(N):
            if barrier_mask is not None and barrier_mask[i, j]:
                continue
            Tij = T[i, j]
            Xij = max(Tij - params.T_a, 0.0)
            E = Xij  # exported heat; q controls effective step size
            # Evaluate every cell, even if Xij is zero, to keep scheme general
            if E <= 0.0:
                continue

            nbrs = neighbor_map(i, j, N, allow_diagonals=params.allow_diagonals, boundary_mode=params.boundary_mode)
            Tn: Dict[str, float] = {}
            sink_dirs: List[str] = []
            for d, idx in nbrs.items():
                if idx is None:
                    if params.boundary_mode == "Outflow":
                        Tn[d] = params.T_a
                        sink_dirs.append(d)
                    continue
                ii, jj = idx
                if barrier_mask is not None and barrier_mask[ii, jj]:
                    continue
                Tn[d] = T[ii, jj]
            if not Tn:
                continue

            w = compute_flux_weights(Tij, Tn, params=params)

            # Build dirs and raw weights, sanitise
            dirs = list(w.keys())
            raw = np.array([w[d] for d in dirs], dtype=np.float64)
            raw[~np.isfinite(raw)] = 0.0
            raw[raw < 0.0] = 0.0
            s = raw.sum()
            if s <= 0.0:
                continue
            pvec = raw / s
            pvec = np.clip(pvec, 0.0, 1.0)
            sum_p = pvec.sum()
            if not np.isfinite(sum_p) or sum_p <= 0.0:
                continue
            pvec /= sum_p

            n_float = E / q
            n_base = int(np.floor(n_float))
            n = n_base + (1 if rng.random() < (n_float - n_base) else 0)
            if n <= 0:
                continue

            out_tokens += n

            if use_multinomial:
                alloc = rng.multinomial(n, pvec)  # shape (K,)
                for k, m in enumerate(alloc):
                    if m == 0:
                        continue
                    d = dirs[k]
                    if d in sink_dirs:
                        sink_tokens += m
                        continue
                    ii, jj = nbrs[d]
                    inflow[ii, jj] += m * q
            else:
                choices = rng.choice(len(dirs), size=n, p=pvec)
                for idx_choice in choices:
                    d = dirs[idx_choice]
                    if d in sink_dirs:
                        sink_tokens += 1
                        continue
                    ii, jj = nbrs[d]
                    inflow[ii, jj] += q

            # Remove exported heat from origin
            T[i, j] = max(params.T_a, T[i, j] - n * q)

    T_next = T + inflow

    if barrier_mask is not None:
        T_next[barrier_mask] = params.T_a
    T_next = background_diffuse(T_next, params, barrier_mask)

    diag: Dict[str, float] = {}
    diag["tokens_out"] = float(out_tokens)
    diag["tokens_sink"] = float(sink_tokens)
    diag["max_T_over_Ta"] = float(T_next.max() / params.T_a)
    diag["mean_T_over_Ta"] = float(T_next.mean() / params.T_a)
    y = np.arange(N, dtype=np.float64)
    weighted_sum = (T_next * y[:, None]).sum(); total_temp = T_next.sum()
    diag["vertical_centroid"] = float(weighted_sum / total_temp) if total_temp > 0 else float(N // 2)
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
        "tokens_out": [(0, 0.0)],
        "tokens_sink": [(0, 0.0)],
    }

    for t in range(1, params.steps + 1):
        if stop_check is not None and stop_check():
            break
        T_source = source_multiplier(t, params) * params.T_a
        if T_source > params.T_a + 1e-6:
            y0, y1, x0, x1 = source_region_coords(params.N, params.source_span_frac)
            if barrier_mask is not None:
                block = T[y0:y1+1, x0:x1+1]
                mask_block = barrier_mask[y0:y1+1, x0:x1+1]
                block[~mask_block] = T_source
                T[y0:y1+1, x0:x1+1] = block
            else:
                T[y0:y1+1, x0:x1+1] = T_source

        T, diag = stochastic_flux_step(T, params=params, barrier_mask=barrier_mask, rng=rng)

        if t % params.snapshot_stride == 0 or t == params.steps:
            snapshots.append((t, T.copy()))
        for k, v in diag.items():
            diagnostics.setdefault(k, []).append((t, v))
        if progress_cb is not None:
            progress_cb(t, params.steps, diag)
        if live_update_stride and live_placeholder is not None and (t % live_update_stride == 0):
            try:
                fig_live, ax_live = plt.subplots(figsize=(5, 5))
                im = ax_live.imshow(
                    T / params.T_a,
                    origin="lower",
                    interpolation="nearest",
                    cmap="inferno",
                    vmin=1.0,
                    vmax=params.k,
                )
                ax_live.set_title(f"Live field at t = {t}  (T/Tₐ fixed scale)")
                fig_live.colorbar(im, ax=ax_live, fraction=0.046, pad=0.04, label="T/Tₐ")
                live_placeholder.pyplot(fig_live, clear_figure=True)
                plt.close(fig_live)
            except Exception:
                pass

    return SimResults(T=T, snapshots=snapshots, diagnostics=diagnostics, params=params)

# =============================
# UI helpers
# =============================

def barrier_controls(N: int) -> Tuple[bool, int, int, int, int]:
    st.subheader("Adiabatic barrier block")
    barrier_enabled = st.checkbox("Enable rectangular barrier", value=False)
    y0 = st.number_input("Barrier bottom row i0 (0=bottom)", min_value=0, max_value=int(N - 1), value=int(st.session_state.barrier_y0))
    y1 = st.number_input("Barrier top row i1", min_value=0, max_value=int(N - 1), value=int(st.session_state.barrier_y1))
    x0 = st.number_input("Barrier left col j0", min_value=0, max_value=int(N - 1), value=int(st.session_state.barrier_x0))
    x1 = st.number_input("Barrier right col j1", min_value=0, max_value=int(N - 1), value=int(st.session_state.barrier_x1))
    return bool(barrier_enabled), int(y0), int(y1), int(x0), int(x1)


def color_and_diffusion_controls() -> Tuple[str, float, float]:
    st.subheader("Color mapping")
    cmap_name = st.selectbox("Colormap", ["inferno", "magma", "viridis", "plasma", "cividis"], index=0)
    gamma = st.slider("Contrast (gamma)", min_value=0.3, max_value=2.0, value=1.0, step=0.1)
    st.subheader("Background diffusion")
    epsilon_diffusion = st.slider("Background diffusion ε", min_value=0.0, max_value=0.2, value=0.01, step=0.005)
    return str(cmap_name), float(gamma), float(epsilon_diffusion)

# =============================
# Sidebar
# =============================

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

def request_stop():
    st.session_state.stop_requested = True

st.title("Probabilistic buoyant plume: stochastic token flux")
st.caption("Toggle between per token sampling and single-shot multinomial allocation. Overflow safe and fixed colour scale.")

with st.sidebar:
    st.header("Controls")
    N   = st.number_input("Grid size N", min_value=21, max_value=401, value=99, step=2)
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

    source_span_pct = st.slider(
        "Source span (% of width)",
        min_value=0.5, max_value=50.0, value=5.0, step=0.5,
        help="Centered horizontal strip, one cell tall, spanning this percent of the grid width",
    )

    st.subheader("Movement model")
    allow_diagonals  = st.checkbox("Allow diagonal moves", value=True)
    epsilon_baseline = st.slider("Baseline eps (floor)", 0.0, 0.05, 0.005, step=0.001)
    lambda_per_Ta    = st.slider("dT sensitivity lambda (per T_a)", 0.0, 3.5, 1.8, step=0.05)
    distance_penalty = st.checkbox("Distance penalty for diagonals (1/sqrt(2))", value=True)
    disable_gravity  = st.checkbox("Disable gravity bias", value=False)
    mu_vertical_tilt = st.slider("Vertical tilt μ", 0.0, 3.0, 1.5, 0.1)

    st.subheader("Stochastic tokens")
    eta_token_quanta = st.select_slider(
        "Token quantum η (in units of T_a)",
        options=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2], value=1e-3,
        help="Token heat is q = η T_a. Smaller η gives more tokens and smoother motion; larger η is chunkier and faster.",
    )

    sampling_mode = st.selectbox("Sampling mode", ["Per token", "Multinomial"], index=0, help="A: per token draws. B: one multinomial draw per cell.")

    boundary_mode = st.selectbox("Boundary mode", ["Outflow", "Blocked", "Periodic"], index=0)

    # Ensure barrier defaults now that N is known
    ensure_barrier_state_defaults(int(N))
    barrier_enabled, barrier_y0, barrier_y1, barrier_x0, barrier_x1 = barrier_controls(int(N))

    cmap_name, gamma, epsilon_diffusion = color_and_diffusion_controls()

    steps           = st.number_input("Time steps", min_value=1, value=200)
    seed            = st.number_input("Random seed", min_value=0, value=42)
    snapshot_stride = st.number_input("Snapshot stride", min_value=1, value=10)

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
        N=int(N), T_a=float(T_a), k=float(k), steps=int(steps), seed=int(seed), snapshot_stride=int(snapshot_stride),
        source_mode=str(source_mode), tau_g=float(tau_g), plateau_steps=int(plateau_steps), tau_d=float(tau_d),
        source_span_frac=float(source_span_pct) / 100.0,
        allow_diagonals=bool(allow_diagonals), epsilon_baseline=float(epsilon_baseline),
        lambda_per_Ta=float(lambda_per_Ta), distance_penalty=bool(distance_penalty), disable_gravity=bool(disable_gravity),
        mu_vertical_tilt=float(mu_vertical_tilt), eta_token_quanta=float(eta_token_quanta), sampling_mode=str(sampling_mode),
        epsilon_diffusion=float(epsilon_diffusion), boundary_mode=str(boundary_mode),
        barrier_enabled=bool(barrier_enabled), barrier_y0=int(barrier_y0), barrier_y1=int(barrier_y1), barrier_x0=int(barrier_x0), barrier_x1=int(barrier_x1),
    )
    st.session_state.params = params

    prog = st.progress(0, text="Starting simulation…")
    status = st.empty()
    live_placeholder = st.empty() if live_update_stride else None

    def progress_cb(t, total, diag):
        pct = int(100 * t / total)
        txt = f"Step {t}/{total} · tokens_out={int(diag.get('tokens_out', 0))}"
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
        cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad(color="white")
        norm = PowerNorm(gamma=gamma)

        fig1, ax1 = plt.subplots(figsize=(6, 6))
        im = ax1.imshow(
            data,
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            vmin=1.0,
            vmax=res.params.k,
        )
        ax1.set_title(f"Field at t = {t_sel} (T/Tₐ, fixed scale)")
        ax1.set_xlabel("x index"); ax1.set_ylabel("y index")
        fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="T/Tₐ")
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
        for key in ["max_T_over_Ta", "mean_T_over_Ta", "vertical_centroid", "tokens_out", "tokens_sink"]:
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
