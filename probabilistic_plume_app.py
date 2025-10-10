"""
SSM model: Stochastic Smoke Model, full Streamlit app
- Stochastic token flux plume engine with A or B sampling
- Barriers Phase 1 editor with Solid or Hole rows, ordered application
- Barriers moved to dedicated tab with Apply button, rows disabled by default
- Robust parsing of barrier rows, no crashes on partial edits
- Fixed colour scale for T/T_a in live and snapshots, white barriers
- Gravity aware source placement and defaults
- Optional incompressible cap: soft back pressure plus final clamp
- Startup defaults, including barriers, can be supplied via ssm_config.json

Run locally:
  pip install -r requirements.txt
  streamlit run probabilistic_plume_app.py

requirements.txt
----------------
streamlit>=1.50
numpy>=2.3
matplotlib
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import os
import json
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

st.set_page_config(page_title="SSM model: stochastic smoke", page_icon="💨", layout="wide")

# =============================================================
# Config loading
# =============================================================

def load_config(path: str = "ssm_config.json") -> Optional[dict]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load config {path}: {e}")
    return None

CFG = load_config()

# =============================================================
# Parameters and results
# =============================================================

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
    source_span_frac: float = 0.05     # fraction of width 0..1
    # Movement model
    allow_diagonals: bool = True
    epsilon_baseline: float = 0.005
    lambda_per_Ta: float = 1.8
    distance_penalty: bool = True
    disable_gravity: bool = False
    mu_vertical_tilt: float = 1.5
    # Stochastic token engine
    eta_token_quanta: float = 1e-3     # token size q = eta * T_a
    sampling_mode: str = "Per token"   # Per token or Multinomial
    # Background diffusion
    epsilon_diffusion: float = 0.01
    # Boundaries
    boundary_mode: str = "Outflow"     # Outflow, Blocked, Periodic
    # Legacy rectangle barrier fallback
    barrier_enabled: bool = False
    barrier_y0: int = 50
    barrier_y1: int = 70
    barrier_x0: int = 20
    barrier_x1: int = 80
    # Incompressible cap
    cap_enabled: bool = False
    cap_softness_per_Ta: float = 0.2


@dataclass
class SimResults:
    T: np.ndarray
    snapshots: List[Tuple[int, np.ndarray]]
    diagnostics: Dict[str, List[Tuple[int, float]]]
    params: SimParams

# =============================================================
# Source schedule and geometry
# =============================================================

def source_multiplier(t: int, p: SimParams) -> float:
    import math
    if p.source_mode == "Persistent":
        return p.k
    if p.source_mode == "Grow":
        return 1.0 + (p.k - 1.0) * (1.0 - math.exp(-max(t, 0) / max(p.tau_g, 1e-9)))
    s_grow = 1.0 + (p.k - 1.0) * (1.0 - math.exp(-max(t, 0) / max(p.tau_g, 1e-9)))
    t_star = 0 if p.k <= 1.0 else int(max(0.0, np.ceil(-p.tau_g * np.log(max(1e-9, (p.k - 1.0) * 0.01 / max(p.k - 1.0, 1e-9))))))
    t_plateau_end = t_star + int(max(0, p.plateau_steps))
    if t < t_star:
        return s_grow
    if t <= t_plateau_end:
        return p.k
    return 1.0 + (p.k - 1.0) * np.exp(-(t - t_plateau_end) / max(p.tau_d, 1e-9))


def source_row(N: int, disable_gravity: bool) -> int:
    if disable_gravity:
        return N // 2
    return int(round(0.25 * (N - 1)))


def source_region_coords(N: int, frac: float, *, row: int) -> Tuple[int, int, int, int]:
    frac = float(np.clip(frac, 0.0, 1.0))
    c = N // 2
    span = max(1, int(round(frac * N)))
    if span % 2 == 0:
        span = min(span + 1, N)
    half = span // 2
    y0 = y1 = int(np.clip(row, 0, N - 1))
    x0 = max(0, c - half)
    x1 = min(N - 1, c + half)
    return y0, y1, x0, x1


def init_grid(params: SimParams) -> np.ndarray:
    N = params.N
    T = np.full((N, N), params.T_a, dtype=np.float64)
    r = source_row(N, params.disable_gravity)
    y0, y1, x0, x1 = source_region_coords(N, params.source_span_frac, row=r)
    T[y0:y1+1, x0:x1+1] = params.k * params.T_a
    return T

# =============================================================
# Neighbours and weights
# =============================================================

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
    b = max(Tp / max(p.T_a, 1e-12), 0.0)
    return {"up": b, "up_left": 0.7 * b, "up_right": 0.7 * b, "left": 0.5, "right": 0.5, "down": 0.2, "down_left": 0.1, "down_right": 0.1}


def _distance_penalty(p: SimParams) -> Dict[str, float]:
    diag = (1.0 / np.sqrt(2)) if p.distance_penalty else 1.0
    return {"up": 1.0, "down": 1.0, "left": 1.0, "right": 1.0, "up_left": diag, "up_right": diag, "down_left": diag, "down_right": diag}


def _backpressure_factor(Tn: float, T_cap: float, delta: float) -> float:
    if delta <= 0.0:
        return 0.0 if Tn >= T_cap else 1.0
    z = (Tn - T_cap) / delta
    z = float(np.clip(z, -50.0, 50.0))
    return 1.0 / (1.0 + np.exp(z))


def compute_flux_weights(
    Tp: float,
    Tn: Dict[str, float],
    *,
    params: SimParams,
    T_cap: Optional[float] = None,
    delta_soft: Optional[float] = None,
) -> Dict[str, float]:
    eps = max(0.0, params.epsilon_baseline)
    lam_eff = params.lambda_per_Ta / max(params.T_a, 1e-12)

    if params.disable_gravity:
        V = 1.0
    else:
        rel = max(Tp / max(params.T_a, 1e-12) - 1.0, 0.0)
        tilt_arg = np.clip(params.mu_vertical_tilt * rel, 0.0, 50.0)
        V = float(np.exp(tilt_arg))

    P = _direction_priors(Tp, params)
    C = _distance_penalty(params)

    up_dirs       = {"up"}
    up_diag_dirs  = {"up_left", "up_right"}
    down_dirs     = {"down", "down_left", "down_right"}
    lateral_dirs  = {"left", "right"}

    use_cap = params.cap_enabled and (T_cap is not None) and (delta_soft is not None)

    w: Dict[str, float] = {}
    for d, Tdj in Tn.items():
        if d not in P:
            continue
        dT = max(Tp - Tdj, 0.0)
        arg = np.clip(lam_eff * dT, 0.0, 50.0)
        g = np.expm1(arg)
        weight = (eps + g) * P[d] * C.get(d, 1.0)
        if not params.disable_gravity:
            if d in up_dirs:
                weight *= V
            elif d in up_diag_dirs:
                weight *= 0.7 * V
            elif d in down_dirs and V > 0:
                weight /= V
            elif d in lateral_dirs and V > 0:
                weight /= V
        if use_cap:
            weight *= _backpressure_factor(Tdj, T_cap, delta_soft)
        if not np.isfinite(weight) or weight < 0.0:
            weight = 0.0
        w[d] = float(weight)
    return w

# =============================================================
# Barrier editor, helpers
# =============================================================

def _bresenham(i0: int, j0: int, i1: int, j1: int) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    di = abs(i1 - i0)
    dj = abs(j1 - j0)
    si = 1 if i0 < i1 else -1
    sj = 1 if j0 < j1 else -1
    i, j = i0, j0
    if dj <= di:
        err = di // 2
        while True:
            points.append((i, j))
            if i == i1 and j == j1:
                break
            i += si
            err -= dj
            if err < 0:
                j += sj
                err += di
    else:
        err = dj // 2
        while True:
            points.append((i, j))
            if i == i1 and j == j1:
                break
            j += sj
            err -= di
            if err < 0:
                i += si
                err += dj
    return points


def _apply_thickness(mask: np.ndarray, cells: List[Tuple[int, int]], radius: int, value: bool = True) -> None:
    N = mask.shape[0]
    r = max(0, int(radius))
    if r == 0:
        for i, j in cells:
            if 0 <= i < N and 0 <= j < N:
                mask[i, j] = value
        return
    for i, j in cells:
        i0 = max(0, i - r); i1 = min(N - 1, i + r)
        j0 = max(0, j - r); j1 = min(N - 1, j + r)
        mask[i0:i1+1, j0:j1+1] = value


def default_barrier_rows(N: int, *, gravity_on: bool) -> List[Dict]:
    # Prefer config if present and matching grid size
    if CFG and CFG.get("barriers", {}).get("enabled", False):
        rows = CFG.get("barriers", {}).get("rows", [])
        # trust user provided rows for any N, they are bounds clipped later
        return rows
    # Fallback: gravity based
    thickness = 1
    if gravity_on:
        i = N // 2
        j0 = int(round(0.25 * (N - 1)))
        j1 = int(round((2.0 / 3.0) * (N - 1)))
        return [{"type": "Line", "mode": "Solid", "j0": j0, "i0": i, "j1": j1, "i1": i, "thickness": thickness, "enabled": False}]
    else:
        centre = N // 2
        top = N - 1
        mid = int(round((centre + top) / 2))
        h = max(1, int(round(0.05 * N)))
        y0 = max(0, min(N - 1, mid - h // 2))
        y1 = max(0, min(N - 1, y0 + h - 1))
        j0 = int(round(0.25 * (N - 1)))
        j1 = int(round((2.0 / 3.0) * (N - 1)))
        return [{"type": "Rect", "mode": "Solid", "j0": j0, "i0": y0, "j1": j1, "i1": y1, "thickness": thickness, "enabled": False}]


def ensure_barrier_rows_defaults(N: int, *, gravity_on: bool) -> None:
    if "barrier_rows" not in st.session_state:
        st.session_state.barrier_rows = default_barrier_rows(N, gravity_on=gravity_on)
    if "prev_N" not in st.session_state:
        st.session_state.prev_N = int(N)
    if "prev_gravity_on" not in st.session_state:
        st.session_state.prev_gravity_on = bool(gravity_on)
    if st.session_state.prev_N != int(N) or st.session_state.prev_gravity_on != bool(gravity_on):
        st.session_state.prev_N = int(N)
        st.session_state.prev_gravity_on = bool(gravity_on)
        st.session_state.barrier_rows = default_barrier_rows(N, gravity_on=gravity_on)


def _safe_int(x, default=None):
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return int(x)
    except Exception:
        return default


def build_barrier_mask_from_rows(N: int, rows: List[Dict]) -> Optional[np.ndarray]:
    if not rows:
        return None

    def _row_complete(r: Dict) -> bool:
        if not r.get("enabled", False):
            return False
        needed = ["j0", "i0", "j1", "i1", "thickness"]
        parsed = {k: _safe_int(r.get(k)) for k in needed}
        if any(v is None for v in parsed.values()):
            return False
        j0, i0, j1, i1, thick = parsed["j0"], parsed["i0"], parsed["j1"], parsed["i1"], parsed["thickness"]
        if not (0 <= i0 < N and 0 <= i1 < N and 0 <= j0 < N and 0 <= j1 < N):
            return False
        if thick < 1:
            return False
        return True

    mask = np.zeros((N, N), dtype=bool)

    def draw_row(target: np.ndarray, row: Dict, value: bool) -> None:
        typ = str(row.get("type", "Line"))
        j0 = _safe_int(row.get("j0"), 0); i0 = _safe_int(row.get("i0"), 0)
        j1 = _safe_int(row.get("j1"), j0); i1 = _safe_int(row.get("i1"), i0)
        # clip to bounds
        j0 = int(np.clip(j0, 0, N - 1)); j1 = int(np.clip(j1, 0, N - 1))
        i0 = int(np.clip(i0, 0, N - 1)); i1 = int(np.clip(i1, 0, N - 1))
        thick = max(1, _safe_int(row.get("thickness"), 1) or 1)
        radius = (thick - 1) // 2
        if typ.lower().startswith("line"):
            pts = _bresenham(i0, j0, i1, j1)
            _apply_thickness(target, pts, radius, value)
        elif typ.lower().startswith("rect"):
            cells = [(i, j) for i in range(min(i0, i1), max(i0, i1) + 1)
                              for j in range(min(j0, j1), max(j0, j1) + 1)]
            _apply_thickness(target, cells, radius, value)

    for row in rows:
        if not _row_complete(row):
            continue
        mode = str(row.get("mode", "Solid")).lower()
        draw_row(mask, row, False if mode == "hole" else True)

    return mask if mask.any() else None


def build_barrier_mask(N: int, p: SimParams) -> Optional[np.ndarray]:
    rows = st.session_state.get("barrier_rows_applied", None)
    if rows is None:
        # fall back to current editable rows, but they may be incomplete, so build will ignore as needed
        rows = st.session_state.get("barrier_rows", [])
    m = build_barrier_mask_from_rows(N, rows)
    if m is not None:
        return m
    if not p.barrier_enabled:
        return None
    y0 = int(np.clip(min(p.barrier_y0, p.barrier_y1), 0, N-1))
    y1 = int(np.clip(max(p.barrier_y0, p.barrier_y1), 0, N-1))
    x0 = int(np.clip(min(p.barrier_x0, p.barrier_x1), 0, N-1))
    x1 = int(np.clip(max(p.barrier_x0, p.barrier_x1), 0, N-1))
    mask = np.zeros((N, N), dtype=bool)
    mask[y0:y1+1, x0:x1+1] = True
    return mask


def barrier_table_editor(N: int, gravity_on: bool) -> None:
    st.subheader("Barrier designer")
    ensure_barrier_rows_defaults(N, gravity_on=gravity_on)
    rows = st.session_state.barrier_rows

    colconf = {
        "type": st.column_config.SelectboxColumn("type", options=["Line", "Rect"], width="small"),
        "mode": st.column_config.SelectboxColumn("mode", options=["Solid", "Hole"], width="small"),
        "j0": st.column_config.NumberColumn("j0 (x0)", step=1, min_value=0, max_value=N-1),
        "i0": st.column_config.NumberColumn("i0 (y0)", step=1, min_value=0, max_value=N-1),
        "j1": st.column_config.NumberColumn("j1 (x1)", step=1, min_value=0, max_value=N-1),
        "i1": st.column_config.NumberColumn("i1 (y1)", step=1, min_value=0, max_value=N-1),
        "thickness": st.column_config.NumberColumn("thickness (cells)", step=1, min_value=1, max_value=25),
        "enabled": st.column_config.CheckboxColumn("enabled", default=False),
    }

    edited = st.data_editor(
        rows,
        num_rows="dynamic",
        column_config=colconf,
        key="barrier_rows_editor",
        width="stretch",
        hide_index=True,
    )

    # Clean without forcing ints until present; leave None for partial edits
    cleaned: List[Dict] = []
    for r in edited:
        cleaned.append({
            "type": r.get("type", "Line"),
            "mode": r.get("mode", "Solid"),
            "j0": _safe_int(r.get("j0"), None),
            "i0": _safe_int(r.get("i0"), None),
            "j1": _safe_int(r.get("j1"), None),
            "i1": _safe_int(r.get("i1"), None),
            "thickness": max(1, _safe_int(r.get("thickness"), 1) or 1),
            "enabled": bool(r.get("enabled", False)),
        })
    st.session_state.barrier_rows = cleaned

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Add default barriers"):
            st.session_state.barrier_rows += default_barrier_rows(N, gravity_on=gravity_on)
    with c2:
        if st.button("Clear all"):
            st.session_state.barrier_rows = []
    with c3:
        if st.button("Disable all"):
            for r in st.session_state.barrier_rows:
                r["enabled"] = False
    with c4:
        if st.button("Apply barriers", type="primary"):
            st.session_state.barrier_rows_applied = [r.copy() for r in st.session_state.barrier_rows]
            st.success("Applied barrier set for the next run")

    with st.expander("Preview", expanded=False):
        try:
            preview_rows = st.session_state.get("barrier_rows", [])
            preview_mask = build_barrier_mask_from_rows(int(N), preview_rows)
            if HAVE_MPL:
                fig_prev, ax_prev = plt.subplots(figsize=(4, 4))
                if preview_mask is None:
                    ax_prev.text(0.5, 0.5, "No enabled barriers", ha="center", va="center")
                    ax_prev.set_axis_off()
                else:
                    ax_prev.imshow(preview_mask, origin="lower", interpolation="nearest", cmap="gray_r")
                    ax_prev.set_title("Barrier mask (white = barrier)")
                    ax_prev.set_xticks([]); ax_prev.set_yticks([])
                st.pyplot(fig_prev, clear_figure=True)
                plt.close(fig_prev)
            else:
                st.write("Matplotlib not available for preview.")
        except Exception as e:
            st.info(f"Preview unavailable: {e}")

# =============================================================
# Background diffusion
# =============================================================

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

# =============================================================
# Stochastic token flux step
# =============================================================

def stochastic_flux_step(
    T: np.ndarray,
    *,
    params: SimParams,
    barrier_mask: Optional[np.ndarray],
    rng: np.random.Generator,
    T_cap: Optional[float] = None,
    delta_soft: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    N = params.N
    q = max(1e-12, params.eta_token_quanta * params.T_a)
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
            E = Xij
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

            w = compute_flux_weights(
                Tij,
                Tn,
                params=params,
                T_cap=T_cap,
                delta_soft=delta_soft,
            )

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
                alloc = rng.multinomial(n, pvec)
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

            T[i, j] = max(params.T_a, T[i, j] - n * q)

    T_next = T + inflow

    if barrier_mask is not None:
        T_next[barrier_mask] = params.T_a
    T_next = background_diffuse(T_next, params, barrier_mask)

    if params.cap_enabled and (T_cap is not None):
        T_next = np.minimum(T_next, T_cap)

    diag: Dict[str, float] = {}
    diag["tokens_out"] = float(out_tokens)
    diag["tokens_sink"] = float(sink_tokens)
    diag["max_T_over_Ta"] = float(T_next.max() / params.T_a)
    diag["mean_T_over_Ta"] = float(T_next.mean() / params.T_a)
    y = np.arange(N, dtype=np.float64)
    weighted_sum = (T_next * y[:, None]).sum(); total_temp = T_next.sum()
    diag["vertical_centroid"] = float(weighted_sum / total_temp) if total_temp > 0 else float(N // 2)
    if params.cap_enabled and (T_cap is not None):
        diag["T_cap_over_Ta"] = float(T_cap / params.T_a)
    return T_next, diag

# =============================================================
# Simulation driver
# =============================================================

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

    T_cap_running = params.k * params.T_a if params.cap_enabled else None
    delta_soft = params.cap_softness_per_Ta * params.T_a if params.cap_enabled else None

    snapshots: List[Tuple[int, np.ndarray]] = [(0, T.copy())]
    diagnostics: Dict[str, List[Tuple[int, float]]] = {
        "max_T_over_Ta": [(0, float(T.max() / params.T_a))],
        "mean_T_over_Ta": [(0, float(T.mean() / params.T_a))],
        "vertical_centroid": [(0, float((np.arange(params.N)[:, None] * T).sum() / max(T.sum(), 1e-12)))],
        "tokens_out": [(0, 0.0)],
        "tokens_sink": [(0, 0.0)],
    }
    if params.cap_enabled and T_cap_running is not None:
        diagnostics["T_cap_over_Ta"] = [(0, float(T_cap_running / params.T_a))]

    for t in range(1, params.steps + 1):
        if stop_check is not None and stop_check():
            break
        T_source = source_multiplier(t, params) * params.T_a
        if params.cap_enabled:
            if T_cap_running is None:
                T_cap_running = T_source
            else:
                T_cap_running = max(T_cap_running, T_source)

        if T_source > params.T_a + 1e-6:
            r = source_row(params.N, params.disable_gravity)
            y0, y1, x0, x1 = source_region_coords(params.N, params.source_span_frac, row=r)
            if barrier_mask is not None:
                block = T[y0:y1+1, x0:x1+1]
                mask_block = barrier_mask[y0:y1+1, x0:x1+1]
                block[~mask_block] = T_source
                T[y0:y1+1, x0:x1+1] = block
            else:
                T[y0:y1+1, x0:x1+1] = T_source

        T, diag = stochastic_flux_step(
            T,
            params=params,
            barrier_mask=barrier_mask,
            rng=rng,
            T_cap=T_cap_running,
            delta_soft=delta_soft,
        )

        if t % params.snapshot_stride == 0 or t == params.steps:
            snapshots.append((t, T.copy()))
        for k, v in diag.items():
            diagnostics.setdefault(k, []).append((t, v))
        if progress_cb is not None:
            progress_cb(t, params.steps, diag)
        if live_update_stride and live_placeholder is not None and (t % live_update_stride == 0):
            try:
                fig_live, ax_live = plt.subplots(figsize=(5, 5))
                live_mask = build_barrier_mask(params.N, params)
                live_data = T / params.T_a
                if live_mask is not None:
                    live_data = np.ma.array(live_data, mask=live_mask)
                cmap_live = plt.get_cmap(st.session_state.get("cmap_name_live", "inferno")).copy()
                try:
                    cmap_live.set_bad(color="white")
                except Exception:
                    pass
                norm_live = PowerNorm(gamma=st.session_state.get("gamma_live", 1.0))
                im = ax_live.imshow(
                    live_data,
                    origin="lower",
                    interpolation="nearest",
                    cmap=cmap_live,
                    norm=norm_live,
                )
                ax_live.set_title(f"Live field at t = {t}  (T/T_a fixed scale)")
                fig_live.colorbar(im, ax=ax_live, fraction=0.046, pad=0.04, label="T/T_a")
                live_placeholder.pyplot(fig_live, clear_figure=True)
                plt.close(fig_live)
            except Exception:
                pass

    return SimResults(T=T, snapshots=snapshots, diagnostics=diagnostics, params=params)

# =============================================================
# UI helpers
# =============================================================

def color_and_diffusion_controls() -> Tuple[str, float, float]:
    st.subheader("Color mapping")
    cmap_name = st.selectbox("Colormap", ["inferno", "magma", "viridis", "plasma", "cividis"], index=0, key="cmap_select")
    gamma = st.slider("Contrast (gamma)", min_value=0.3, max_value=2.0, value=1.0, step=0.1, key="gamma_slider")
    st.session_state.cmap_name_live = cmap_name
    st.session_state.gamma_live = gamma
    st.subheader("Background diffusion")
    epsilon_diffusion = st.slider("Background diffusion ε", min_value=0.0, max_value=0.2, value=0.01, step=0.005)
    return str(cmap_name), float(gamma), float(epsilon_diffusion)

# =============================================================
# Tabs: Run, Barriers, Results
# =============================================================

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

def request_stop():
    st.session_state.stop_requested = True

st.title("SSM model: stochastic smoke plume")
st.caption("Per token vs multinomial sampling, Solid or Hole barriers on their own tab, fixed T/T_a scale, gravity aware defaults, optional incompressible cap, config driven defaults.")

# Defaults from config, if present
_defaults = CFG.get("app_defaults", {}) if CFG else {}

tab_run, tab_bar, tab_res = st.tabs(["Run", "Barriers", "Results"]) 

with tab_run:
    st.header("Run controls")

    with st.expander("Basic setup", expanded=False):
        cols = st.columns([1,1,1,1])
        with cols[0]:
            N   = st.number_input("Grid size N", min_value=21, max_value=401, value=int(_defaults.get("N", 99)), step=2)
        with cols[1]:
            T_a = st.number_input("Ambient temperature T_a", min_value=0.1, value=float(_defaults.get("T_a", 20.0)))
        with cols[2]:
            k   = st.number_input("Source peak multiplier k", min_value=1.0, value=float(_defaults.get("k", 5.0)))
        with cols[3]:
            steps = st.number_input("Time steps", min_value=1, value=int(_defaults.get("steps", 200)))

        source_mode = st.selectbox(
            "Source profile",
            ["Persistent", "Grow", "Grow-plateau-decay"],
            index=["Persistent","Grow","Grow-plateau-decay"].index(_defaults.get("source_mode","Grow-plateau-decay"))
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            tau_g = st.number_input("Growth tau (steps)", min_value=1.0, value=float(_defaults.get("tau_g", 20.0)))
        with c2:
            plateau_steps = st.number_input("Plateau steps", min_value=0, value=int(_defaults.get("plateau_steps", 50)))
        with c3:
            tau_d = st.number_input("Decay tau (steps)", min_value=1.0, value=float(_defaults.get("tau_d", 40.0)))

        source_span_pct = st.slider(
            "Source span (% of width)",
            min_value=0.5, max_value=50.0, value=float(_defaults.get("source_span_pct", 5.0)), step=0.5,
            help="Centered horizontal strip one cell tall, given as percent of grid width"
        )

        c4, c5 = st.columns(2)
        with c4:
            snapshot_stride = st.number_input("Snapshot stride", min_value=1, value=int(_defaults.get("snapshot_stride", 10)))
        with c5:
            live_update_stride = st.number_input("Live update every n steps", min_value=0, value=int(_defaults.get("live_update_stride", 10)))

    with st.expander("Advanced options", expanded=False):
        st.markdown("**Movement model**")
        allow_diagonals  = st.checkbox("Allow diagonal moves", value=bool(_defaults.get("allow_diagonals", True)))
        epsilon_baseline = st.slider("Baseline eps (floor)", 0.0, 0.05, float(_defaults.get("epsilon_baseline", 0.005)), step=0.001)
        lambda_per_Ta    = st.slider("dT sensitivity lambda (per T_a)", 0.0, 3.5, float(_defaults.get("lambda_per_Ta", 1.8)), step=0.05)
        distance_penalty = st.checkbox("Distance penalty for diagonals (1/sqrt(2))", value=bool(_defaults.get("distance_penalty", True)))
        disable_gravity  = st.checkbox("Disable gravity bias", value=bool(_defaults.get("disable_gravity", False)))
        mu_vertical_tilt = st.slider("Vertical tilt μ", 0.0, 3.0, float(_defaults.get("mu_vertical_tilt", 1.5)), 0.1)

        st.markdown("---")
        st.markdown("**Stochastic tokens**")
        eta_token_quanta = st.select_slider(
            "Token quantum η (in units of T_a)",
            options=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2], value=float(_defaults.get("eta_token_quanta", 1e-3)),
            help="Token heat is q = η T_a. Smaller η gives more tokens and smoother motion; larger η is chunkier and faster."
        )
        sampling_mode = st.selectbox(
            "Sampling mode", ["Per token", "Multinomial"],
            index=["Per token","Multinomial"].index(_defaults.get("sampling_mode","Per token"))
        )

        st.markdown("---")
        st.markdown("**Boundaries and diffusion**")
        boundary_mode = st.selectbox(
            "Boundary mode", ["Outflow", "Blocked", "Periodic"],
            index=["Outflow","Blocked","Periodic"].index(_defaults.get("boundary_mode","Outflow"))
        )
        # Diffusion only here, colour mapping moved to Results tab
        epsilon_diffusion = st.slider(
            "Background diffusion ε",
            min_value=0.0, max_value=0.2,
            value=float(_defaults.get("epsilon_diffusion", 0.01)), step=0.005,
            help="Moore neighbour averaging per step. Set to 0 to disable."
        )

        st.markdown("---")
        st.markdown("**Incompressible cap**")
        cap_enabled = st.checkbox("Enable soft cap and final clamp", value=bool(_defaults.get("cap_enabled", True)))
        cap_softness_per_Ta = st.slider(
            "Cap softness δ in units of T_a",
            min_value=0.02, max_value=1.0, value=float(_defaults.get("cap_softness_per_Ta", 0.2)), step=0.02,
            help="Back pressure starts to bite as neighbors approach the running max source temp. Smaller δ is sharper."
        )

        st.markdown("---")
        seed = st.number_input("Random seed", min_value=0, value=int(_defaults.get("seed", 42)))

    # Run button at the bottom, unchanged logic
    btn_cols = st.columns([1,1])
    with btn_cols[0]:
        run_clicked = st.button("Run simulation", type="primary")
    with btn_cols[1]:
        st.button("Stop", on_click=request_stop)

    if run_clicked:
        st.session_state.stop_requested = False
        params = SimParams(
            N=int(N), T_a=float(T_a), k=float(k), steps=int(steps), seed=int(seed), snapshot_stride=int(snapshot_stride),
            source_mode=str(source_mode), tau_g=float(tau_g), plateau_steps=int(plateau_steps), tau_d=float(tau_d),
            source_span_frac=float(source_span_pct) / 100.0,
            allow_diagonals=bool(allow_diagonals), epsilon_baseline=float(epsilon_baseline),
            lambda_per_Ta=float(lambda_per_Ta), distance_penalty=bool(distance_penalty), disable_gravity=bool(disable_gravity),
            mu_vertical_tilt=float(mu_vertical_tilt), eta_token_quanta=float(eta_token_quanta), sampling_mode=str(sampling_mode),
            epsilon_diffusion=float(epsilon_diffusion), boundary_mode=str(boundary_mode),
            barrier_enabled=False,
            cap_enabled=bool(cap_enabled), cap_softness_per_Ta=float(cap_softness_per_Ta),
        )
        st.session_state.params = params

        prog = st.progress(0, text="Starting simulation…")
        status = st.empty()
        live_placeholder = st.empty() if live_update_stride else None

        def progress_cb(t, total, diag):
            pct = int(100 * t / total)
            cap_txt = f" · cap={(diag.get('T_cap_over_Ta', 0.0)):.2f} T_a" if params.cap_enabled else ""
            txt = f"Step {t}/{total} · tokens_out={int(diag.get('tokens_out', 0))}{cap_txt}"
            prog.progress(pct, text=txt)
            status.write(txt)

        def stop_check():
            return st.session_state.stop_requested

        with st.spinner("Running simulation..."):
            res = run_simulation(
                params,
                progress_cb=progress_cb,
                live_update_stride=int(live_update_stride),
                live_placeholder=live_placeholder,
                stop_check=stop_check,
            )
        st.session_state.results = res
        prog.empty()

with tab_bar:
    st.header("Barrier setup")
    # seed barrier rows from config on first load only
    if CFG and CFG.get("barriers") and "seeded_from_cfg" not in st.session_state:
        st.session_state.barrier_rows = CFG["barriers"].get("rows", [])
        st.session_state.barrier_rows_applied = [r.copy() for r in st.session_state.barrier_rows]
        st.session_state["seeded_from_cfg"] = True
    # Use grid size and gravity from current or default params for sensible defaults if no config
    current_N = int(st.session_state.get("params", SimParams(N=int(_defaults.get("N", 99)))).N)
    current_grav_on = not bool(st.session_state.get("params", SimParams(disable_gravity=bool(_defaults.get("disable_gravity", False)))).disable_gravity)
    barrier_table_editor(current_N, current_grav_on)

with tab_res:
    st.header("Results")
    # Display controls live on Results tab
    disp_cols = st.columns([1,3])
    with disp_cols[0]:
        st.subheader("Display")
        cmap_name_sel = st.selectbox(
            "Colormap",
            ["inferno", "magma", "viridis", "plasma", "cividis"],
            index=["inferno","magma","viridis","plasma","cividis"].index(
                st.session_state.get("cmap_name_live", "inferno")
            )
        )
        gamma_sel = st.slider(
            "Contrast (gamma)",
            min_value=0.3, max_value=2.0,
            value=float(st.session_state.get("gamma_live", 1.0)),
            step=0.1
        )
        # Persist selections so both live and snapshots use them
        st.session_state["cmap_name_live"] = cmap_name_sel
        st.session_state["gamma_live"] = gamma_sel

    res: Optional[SimResults] = st.session_state.get("results")
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
            cmap = plt.get_cmap(st.session_state.get("cmap_name_live", "inferno")).copy()
            try:
                cmap.set_bad(color="white")
            except Exception:
                pass
            norm = PowerNorm(gamma=st.session_state.get("gamma_live", 1.0))

            fig1, ax1 = plt.subplots(figsize=(6, 6))
            im = ax1.imshow(
                data,
                origin="lower",
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
            )
            ax1.set_title(f"Field at t = {t_sel} (T/T_a, fixed scale)")
            ax1.set_xlabel("x index"); ax1.set_ylabel("y index")
            fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="T/T_a")
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
            keys = ["max_T_over_Ta", "mean_T_over_Ta", "vertical_centroid", "tokens_out", "tokens_sink"]
            if "T_cap_over_Ta" in res.diagnostics:
                keys.append("T_cap_over_Ta")
            for key in keys:
                ts = np.array(res.diagnostics.get(key, []))
                if ts.size:
                    ax3.plot(ts[:, 0], ts[:, 1], label=key)
            ax3.set_xlabel("time step"); ax3.set_ylabel("value"); ax3.set_title("Diagnostics time series"); ax3.legend()
            with col2:
                st.pyplot(fig3, clear_figure=True)
            plt.close(fig3)

    # Export
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