"""
Probabilistic plume model, Streamlit app with exponential ΔT weighting, optional diagonals, adjustable barrier block, colormap controls, and gravity toggle.
Run locally:
    pip install -r requirements.txt
    streamlit run probabilistic_plume_app.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

st.set_page_config(page_title="Probabilistic plume model", page_icon="💨", layout="wide")

@dataclass
class SimParams:
    N: int = 99
    T_a: float = 20.0
    k: float = 5.0
    alpha: float = 0.5
    steps: int = 200
    parcels_per_step: int = 10
    seed: int = 42
    snapshot_stride: int = 10
    source_mode: str = "Persistent"
    tau_g: float = 20.0
    plateau_steps: int = 50
    tau_d: float = 40.0
    allow_diagonals: bool = True
    epsilon_baseline: float = 0.005
    lambda_per_Ta: float = 1.4
    distance_penalty: bool = True
    disable_gravity: bool = False
    # Barrier block
    barrier_enabled: bool = False
    barrier_y0: int = 50
    barrier_x0: int = 20
    barrier_y1: int = 70
    barrier_x1: int = 80

@dataclass
class SimResults:
    T: np.ndarray
    snapshots: List[Tuple[int, np.ndarray]]
    diagnostics: Dict[str, List[Tuple[int, float]]]
    params: SimParams

def build_barrier_mask(N: int, p: SimParams) -> Optional[np.ndarray]:
    if not p.barrier_enabled:
        return None
    y0 = int(np.clip(min(p.barrier_y0, p.barrier_y1), 0, N-1))
    y1 = int(np.clip(max(p.barrier_y0, p.barrier_y1), 0, N-1))
    x0 = int(np.clip(min(p.barrier_x0, p.barrier_x1), 0, N-1))
    x1 = int(np.clip(max(p.barrier_x0, p.barrier_x1), 0, N-1))
    mask = np.zeros((N, N), dtype=bool)
    mask[y0:y1+1, x0:x1+1] = True
    return mask

def init_grid(params: SimParams) -> np.ndarray:
    N = params.N
    T = np.full((N, N), params.T_a)
    c = N // 2
    T[c, c] = params.k * params.T_a
    return T

def neighbor_map(i, j, N, allow_diagonals=True):
    m = {
        "up": (i + 1, j) if i + 1 < N else None,
        "down": (i - 1, j) if i - 1 >= 0 else None,
        "left": (i, j - 1) if j - 1 >= 0 else None,
        "right": (i, j + 1) if j + 1 < N else None,
    }
    if allow_diagonals:
        m.update({
            "up_left": (i + 1, j - 1) if (i + 1 < N and j - 1 >= 0) else None,
            "up_right": (i + 1, j + 1) if (i + 1 < N and j + 1 < N) else None,
            "down_left": (i - 1, j - 1) if (i - 1 >= 0 and j - 1 >= 0) else None,
            "down_right": (i - 1, j + 1) if (i - 1 >= 0 and j + 1 < N) else None,
        })
    return m

def compute_move_weights_exp(T_particle, T_neighbors, params: SimParams):
    eps = max(0.0, params.epsilon_baseline)
    lam_eff = params.lambda_per_Ta / max(params.T_a, 1e-12)
    if params.disable_gravity:
        base = 1.0
        P_dir = {d: base for d in ["up","up_left","up_right","left","right","down","down_left","down_right"]}
    else:
        b = T_particle / params.T_a
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
    C_dist = {d: (1/np.sqrt(2) if ("_" in d and params.distance_penalty) else 1.0) for d in P_dir}
    weights = {}
    for d, Tn in T_neighbors.items():
        if d not in P_dir: continue
        dT = T_particle - Tn
        g = np.exp(lam_eff * dT) - 1.0 if dT > 0 else 0.0
        weights[d] = (eps + g) * P_dir[d] * C_dist[d]
    weights["stay"] = eps
    return weights

def run_simulation(params: SimParams, progress_cb=None):
    np.random.seed(params.seed)
    N = params.N
    T = init_grid(params)
    barrier_mask = build_barrier_mask(N, params)
    snapshots = []
    for t in range(params.steps):
        T_next = T.copy()
        for _ in range(params.parcels_per_step):
            i, j = np.random.randint(0, N, 2)
            if barrier_mask is not None and barrier_mask[i,j]:
                continue
            T_particle = T[i,j]
            neigh = neighbor_map(i,j,N,params.allow_diagonals)
            Tn = {d:T[x,y] for d,(x,y) in neigh.items() if (x is not None and (barrier_mask is None or not barrier_mask[x,y]))}
            w = compute_move_weights_exp(T_particle,Tn,params)
            dirs, probs = list(w.keys()), np.array(list(w.values()))
            probs /= probs.sum()
            move = np.random.choice(dirs,p=probs)
            if move=="stay": continue
            x,y = neigh[move]
            if barrier_mask is not None and barrier_mask[x,y]:
                continue
            T_next[x,y] = (1-params.alpha)*T[x,y] + params.alpha*T_particle
        T = T_next
        if t % params.snapshot_stride == 0:
            snapshots.append((t, T.copy()))
        if progress_cb:
            progress_cb(t, params.steps)
    return SimResults(T, snapshots, {}, params)

st.sidebar.header("Controls")
N = st.sidebar.number_input("Grid size N",21,401,99,step=2)
T_a = st.sidebar.number_input("Ambient temperature T_a",0.1,100.0,20.0)
k = st.sidebar.number_input("Source peak multiplier k",1.0,10.0,5.0)
source_mode = st.sidebar.selectbox("Source profile",["Persistent","Grow","Grow-plateau-decay"],0)
col_s1,col_s2,col_s3=st.sidebar.columns(3)
with col_s1: tau_g=st.number_input("Growth tau (steps)",1.0,200.0,20.0)
with col_s2: plateau_steps=st.number_input("Plateau steps",0,200,50)
with col_s3: tau_d=st.number_input("Decay tau (steps)",1.0,200.0,40.0)
alpha=st.sidebar.slider("Mixing fraction alpha",0.0,1.0,0.5)

st.sidebar.subheader("Movement model")
allow_diagonals=st.sidebar.checkbox("Allow diagonal moves",True)
epsilon_baseline=st.sidebar.slider("Baseline eps (floor)",0.0,0.05,0.005,step=0.001)
lambda_per_Ta=st.sidebar.slider("dT sensitivity lambda (per T_a)",0.0,3.0,1.4,step=0.05)
distance_penalty=st.sidebar.checkbox("Distance penalty for diagonals (1/sqrt(2))",True)
disable_gravity=st.sidebar.checkbox("Disable gravity bias",False,help="Treat all directions equally; only ΔT and distance penalty apply")

st.sidebar.subheader("Adiabatic barrier")
barrier_enabled=st.sidebar.checkbox("Enable rectangular barrier",False)
barrier_y0=st.sidebar.number_input("Barrier bottom row i0",0,int(N-1),max(0,(N//2)-10))
barrier_y1=st.sidebar.number_input("Barrier top row i1",0,int(N-1),min(N-1,(N//2)+10))
barrier_x0=st.sidebar.number_input("Barrier left col j0",0,int(N-1),20)
barrier_x1=st.sidebar.number_input("Barrier right col j1",0,int(N-1),int(N-20))

steps=st.sidebar.number_input("Time steps",1,2000,200)
parcels_per_step=st.sidebar.number_input("Parcels per step r",1,100,10)
seed=st.sidebar.number_input("Random seed",0,9999,42)
snapshot_stride=st.sidebar.number_input("Snapshot stride",1,100,10)

st.sidebar.subheader("Color mapping")
cmap_name=st.sidebar.selectbox("Colormap",["inferno","magma","viridis","plasma","cividis"],0)
gamma=st.sidebar.slider("Contrast (gamma)",0.3,2.0,1.0,0.1)

run_btn=st.sidebar.button("Run simulation",type="primary")
if run_btn:
    params=SimParams(N=int(N),T_a=float(T_a),k=float(k),alpha=float(alpha),steps=int(steps),parcels_per_step=int(parcels_per_step),seed=int(seed),snapshot_stride=int(snapshot_stride),source_mode=str(source_mode),tau_g=float(tau_g),plateau_steps=int(plateau_steps),tau_d=float(tau_d),allow_diagonals=bool(allow_diagonals),epsilon_baseline=float(epsilon_baseline),lambda_per_Ta=float(lambda_per_Ta),distance_penalty=bool(distance_penalty),disable_gravity=bool(disable_gravity),barrier_enabled=bool(barrier_enabled),barrier_y0=int(barrier_y0),barrier_x0=int(barrier_x0),barrier_y1=int(barrier_y1),barrier_x1=int(barrier_x1))
    prog=st.progress(0)
    def progress_cb(t,total): prog.progress(int(100*t/total))
    res=run_simulation(params,progress_cb=progress_cb)
    prog.empty()
    if res.snapshots:
        tvals=[s[0] for s in res.snapshots]
        t_sel=st.slider("Snapshot",min_value=min(tvals),max_value=max(tvals),value=max(tvals),step=params.snapshot_stride)
        T_sel=[T for (tt,T) in res.snapshots if tt==t_sel][0]
        barrier_mask_plot=build_barrier_mask(res.params.N,res.params)
        data=(T_sel/res.params.T_a).copy()
        if barrier_mask_plot is not None:
            data=np.ma.array(data,mask=barrier_mask_plot)
        cmap=plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color="black")
        norm=PowerNorm(gamma=gamma)
        fig,ax=plt.subplots(figsize=(6,6))
        im=ax.imshow(data,origin="lower",interpolation="nearest",cmap=cmap,norm=norm)
        ax.set_title(f"Field at t={t_sel} (T/T_a)")
        ax.set_xlabel("x index"); ax.set_ylabel("y index")
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04,label="T/T_a")
        st.pyplot(fig,clear_figure=True)
        plt.close(fig)
