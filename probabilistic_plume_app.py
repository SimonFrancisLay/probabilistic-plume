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
    k: float = 5.0            # peak/source multiplier
    alpha: float = 0.5        # mixing fraction
    steps: int = 200
    parcels_per_step: int = 10
    seed: int = 42
    snapshot_stride: int = 10 # take a field snapshot every this many steps
    # Source scheduling
    source_mode: str = "Persistent"   # options: Persistent, Grow, Grow-plateau-decay
    tau_g: float = 20.0               # growth time constant (steps)
    plateau_steps: int = 50           # plateau duration (steps)
    tau_d: float = 40.0               # decay time constant (steps)
    # Movement model parameters (exponential ΔT weighting)
    allow_diagonals: bool = True
    epsilon_baseline: float = 0.005   # small floor to avoid stalling
    lambda_per_Ta: float = 1.4        # λ_eff = lambda_per_Ta / T_a
    distance_penalty: bool = True
    # Boundary handling
    boundary_mode: str = "Outflow"  # options: "Blocked", "Periodic", "Reflect", "Outflow"

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
# Core model helpers (persistent engine, exponential ΔT movement)
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

def neighbor_map(
    i: int, j: int, N: int,
    allow_diagonals: bool = True,
    boundary_mode: str = "Outflow"
) -> Dict[str, Optional[Tuple[int, int]]]:
    # candidate displacements
    moves = {
        "up": (1, 0), "down": (-1, 0), "left": (0, -1), "right": (0, 1)
    }
    if allow_diagonals:
        moves.update({
            "up_left": (1, -1), "up_right": (1, 1),
            "down_left": (-1, -1), "down_right": (-1, 1),
        })

    nbrs: Dict[str, Optional[Tuple[int, int]]] = {}
    for d, (di, dj) in moves.items():
        ii, jj = i + di, j + dj

        if boundary_mode == "Periodic":
            # wrap around both axes
            ii %= N
            jj %= N
            nbrs[d] = (ii, jj)
            continue

        # for Blocked and Outflow we test bounds
        if 0 <= ii < N and 0 <= jj < N:
            nbrs[d] = (ii, jj)
        else:
            # Blocked: mark as None (unavailable)
            # Outflow: also mark as None here; we will treat it specially in the move step
            nbrs[d] = None

    return nbrs


def compute_move_weights_exp(T_particle: float, T_neighbors: Dict[str, float], params: SimParams) -> Dict[str, float]:
    """Exponential ΔT weighting with directional priors and optional distance penalty.
    w_d = [ε + max(exp(λ ΔT) - 1, 0)] * P_dir(d) * C_dist(d),  ΔT = T_p - T_n
    Direction priors favour up, then up-diagonals, then lateral, then down.
    """
    eps = max(0.0, params.epsilon_baseline)
    lam_eff = params.lambda_per_Ta / max(params.T_a, 1e-12)

    b = T_particle / params.T_a  # buoyancy factor for priors
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
        # Only include priors we defined; skip unknown keys
        if d not in P_dir:
            continue
        dT = T_particle - Tn
        g = np.exp(lam_eff * dT) - 1.0 if dT > 0 else 0.0
        w = (eps + g) * P_dir[d] * C_dist.get(d, 1.0)
        weights[d] = max(0.0, float(w))

    # Add a small 'stay' option so we never run out of probability mass
    weights["stay"] = eps
    return weights

# Persistent parcel representation
from dataclasses import dataclass
@dataclass
class Parcel:
    i: int
    j: int
    T: float


def step_once_persistent(T: np.ndarray, parcels: List[Parcel], params: SimParams, rng: np.random.Generator, T_source: float) -> Tuple[np.ndarray, List[Parcel], Dict[str, float]]:
    N = params.N
    c = N // 2

    # Clamp the source cell temperature for this step and inject r new parcels
    T[c, c] = T_source
    for _ in range(params.parcels_per_step):
        parcels.append(Parcel(c, c, T_source))

    # Prepare arrival buckets per cell: list of incoming parcel temps
    arrivals_T: List[List[List[float]]] = [[[] for _ in range(N)] for __ in range(N)]

    moved_up = 0
    total_moves = 0

    new_parcels: List[Parcel] = []
    # before loop: keep a list to rebuild only the parcels that stay in-domain
new_parcels: List[Parcel] = []

for p in parcels:
    nbrs = neighbor_map(p.i, p.j, N,
                        allow_diagonals=params.allow_diagonals,
                        boundary_mode=params.boundary_mode)

    # Build T_neighbors from in-bounds entries only
    T_neighbors: Dict[str, float] = {d: T[idx] for d, idx in nbrs.items() if idx is not None}

    # If Outflow: add synthetic neighbours for off-grid directions using ambient T_a
    out_dirs = [d for d, idx in nbrs.items() if idx is None]
    if params.boundary_mode == "Outflow":
        for d in out_dirs:
            # treat outside as ambient to compute ΔT; we will handle removal after choosing
            T_neighbors[d] = params.T_a

    # Compute weights
    weights = compute_move_weights_exp(p.T, T_neighbors, params)

    # Normalise
    dirs = list(weights.keys())
    probs = np.array([weights[d] for d in dirs], dtype=np.float64)
    s = probs.sum()
    if s <= 0:
        chosen_dir = "stay"
    else:
        probs /= s
        choice = rng.choice(len(dirs), p=probs)
        chosen_dir = dirs[choice]

    # Resolve destination
    if chosen_dir == "stay":
        di, dj = p.i, p.j
        arrivals_T[di][dj].append(p.T)
        new_parcels.append(Parcel(di, dj, p.T))
        continue

    dest = nbrs.get(chosen_dir, None)

    if dest is None:
        # Off-grid direction chosen
        if params.boundary_mode == "Outflow":
            # parcel leaves the domain: do not add to arrivals, do not keep parcel
            total_moves += 1
            # optional: track an outflow counter if you wish for diagnostics
            continue
        else:
            # Blocked: treat as 'stay'
            di, dj = p.i, p.j
    else:
        di, dj = dest
        total_moves += 1
        if chosen_dir.startswith("up"):
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


def source_multiplier(t: int, p: SimParams) -> float:
    """Return source temperature multiplier s(t) such that T_source(t) = s(t) * T_a.
    Persistent: s(t) = k.
    Grow: s(t) = 1 + (k-1)*(1 - exp(-t/tau_g)).
    Grow-plateau-decay: piecewise growth to k, plateau, then decay back toward 1.
    """
    import math
    if p.source_mode == "Persistent":
        return p.k
    if p.source_mode == "Grow":
        return 1.0 + (p.k - 1.0) * (1.0 - math.exp(-max(t,0) / max(p.tau_g, 1e-9)))
    # Grow-plateau-decay
    s_grow = 1.0 + (p.k - 1.0) * (1.0 - math.exp(-max(t,0) / max(p.tau_g, 1e-9)))
    # define plateau start when within 99% of k
    t_star = -p.tau_g * math.log(max(1e-9, (p.k - 1.0) * 0.01 / max(p.k - 1.0, 1e-9)))
    t_plateau_start = int(max(0.0, math.ceil(t_star)))
    t_plateau_end = t_plateau_start + int(max(0, p.plateau_steps))
    if t < t_plateau_start:
        return s_grow
    if t <= t_plateau_end:
        return p.k
    # decay
    td = max(p.tau_d, 1e-9)
    return 1.0 + (p.k - 1.0) * math.exp(-(t - t_plateau_end) / td)

def diffuse8(T: np.ndarray, eps: float) -> np.ndarray:
    """Apply background diffusion toward the 8-neighbour (Moore) average.
    T_new = (1-eps)*T + eps*mean(neighbours_8). Uses edge padding to mimic reflecting boundaries.
    """
    if eps <= 0.0:
        return T
    pad = np.pad(T, 1, mode='edge')
    # Sum of 8 neighbours (exclude center)
    nsum = (
        pad[0:-2, 0:-2] + pad[0:-2, 1:-1] + pad[0:-2, 2:  ] +
        pad[1:-1, 0:-2] +                     pad[1:-1, 2:  ] +
        pad[2:  , 0:-2] + pad[2:  , 1:-1] + pad[2:  , 2:  ]
    )
    navg = nsum / 8.0
    return (1.0 - eps) * T + eps * navg

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

        # compute source temp for this step
        T_source = source_multiplier(t, params) * params.T_a
        T, parcels, diag = step_once_persistent(T, parcels, params, rng, T_source)
        
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
                plt.close(fig_live)
                plt.close(fig_live)
            except Exception:
                pass

    return SimResults(T=T, snapshots=snapshots, diagnostics=diagnostics, params=params)

# -----------------------------
# UI controls
# -----------------------------
# -----------------------------

st.title("Probabilistic buoyant plume: minimal model")
st.caption("Persistent parcels with scheduled source and exponential-ΔT movement; no background diffusion")

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

    alpha = st.slider("Mixing fraction alpha", 0.0, 1.0, 0.5)

    st.subheader("Movement model")
    allow_diagonals  = st.checkbox("Allow diagonal moves", value=True)
    epsilon_baseline = st.slider("Baseline ε (floor)", 0.0, 0.05, 0.005, step=0.001)
    lambda_per_Ta    = st.slider("ΔT sensitivity λ (per T_a)", 0.0, 3.0, 1.4, step=0.05)
    distance_penalty = st.checkbox("Distance penalty for diagonals (1/√2)", value=True)

    boundary_mode = st.selectbox(
        "Boundary mode",
        ["Outflow", "Blocked", "Periodic"],
        index=0,
        help="Outflow removes parcels that step out; Blocked ignores off-grid moves; Periodic wraps around"
    )

    steps            = st.number_input("Time steps", min_value=1, value=200)
    parcels_per_step = st.number_input("Parcels per step r", min_value=1, value=10)
    seed             = st.number_input("Random seed", min_value=0, value=42)
    snapshot_stride  = st.number_input("Snapshot stride", min_value=1, value=10)

    st.markdown("---")
    live_update_stride = st.number_input("Live update every n steps", min_value=0, value=10)
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
        lambda_per_Ta=float(lambda_per_Ta), distance_penalty=bool(distance_penalty),
        boundary_mode=str(boundary_mode),
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
        plt.close(fig1)
        plt.close(fig1)

        # Centreline profile above the source
        c = res.params.N // 2
        profile = T_sel[c:, c] / res.params.T_a  # centreline using selected snapshot  # from centre upward
        y = np.arange(profile.size)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(profile, y)
        ax2.set_xlabel("T/T_a")
        ax2.set_ylabel("vertical distance above centre")
        ax2.set_title("Centreline profile above source")
        with col2:
            st.pyplot(fig2, clear_figure=True)
        plt.close(fig2)
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
    Notes: persistent parcels with scheduled source options (persistent, grow, grow–plateau–decay) and exponential-ΔT movement with optional diagonals; no global diffusion. Later we will add the Brownian floor and directional persistence as optional features.
    """
)
