from __future__ import annotations

"""
https://github.com/gajaka/luces-pvs-theories
"""

"""
stosic_v14.py — 7-node krug (K=7 / prilagodjenje 7/39) — Transport structure v2 / spectral full structure (GHQ 7/39)

Izvor (Stosić / LUCES):
  luces-pvs-theories-main/transport_structure_v2.pvs
  — ax_vertex_invariant, ax_monge_invariant (∀ metrike)
  — ax_rank_spectral_only: rank_preserving IFF is_spectral
  — thm_spectral_full_structure: spectral ⇒ vertex+Monge+rank

Mapiranje na 7/39:
  samo spectral metrike m∈{0,1} (Euclidean + var-weight ko-pojavljivanje)
  matching A→B; zadrži plan samo ako Spearman ρ(src,tgt) > 0.99 (rank)
  Π[i,j] += 1 za te ivice
  skor[j] = Σ_{i∈last} Π(i,j)
  next = top 7; bez randoma; stop ako uzastopni/AP
"""

from typing import List, Tuple

import numpy as np
from scipy.stats import spearmanr

from stosic_v1 import EPS, MAX_NUM, load_draws
from stosic_v2 import top7_from_freq
from stosic_v8 import cooccurrence_features, cost_matrix
from stosic_v9 import optimal_matching_support
from stosic_v10 import is_degenerate
from stosic_v13 import cost_var_weighted

RANK_RHO_MIN = 0.99


def is_rank_preserving(pairs: List[Tuple[int, int]]) -> bool:
    if len(pairs) < 2:
        return False
    ordered = sorted(pairs, key=lambda ij: ij[0])
    src_rank = np.arange(len(ordered), dtype=np.float64)
    tgt_vals = np.array([j for _, j in ordered], dtype=np.float64)
    rho, _ = spearmanr(src_rank, tgt_vals)
    if rho is None or np.isnan(rho):
        return False
    return float(rho) > RANK_RHO_MIN


def spectral_costs(draws: np.ndarray) -> List[np.ndarray]:
    feats = cooccurrence_features(draws)
    return [cost_matrix(feats), cost_var_weighted(feats)]


def accumulate_spectral_rank_support(
    draws: np.ndarray, costs: List[np.ndarray]
) -> np.ndarray:
    Pi = np.zeros((MAX_NUM, MAX_NUM), dtype=np.float64)
    for t in range(len(draws) - 1):
        src = [int(n) - 1 for n in draws[t]]
        tgt = [int(n) - 1 for n in draws[t + 1]]
        for C in costs:
            pairs = optimal_matching_support(src, tgt, C)
            if not is_rank_preserving(pairs):
                continue
            for i, j in pairs:
                Pi[i, j] += 1.0
    return Pi


def predict_next(draws: np.ndarray) -> List[int]:
    costs = spectral_costs(draws)
    Pi = accumulate_spectral_rank_support(draws, costs)
    skor = np.zeros(MAX_NUM, dtype=np.float64)
    for n in draws[-1]:
        skor += Pi[int(n) - 1, :]
    if float(skor.sum()) <= 0:
        for d in draws:
            for n in d:
                skor[int(n) - 1] += 1.0
    combo = top7_from_freq(skor + EPS)
    if is_degenerate(combo):
        nu = np.zeros(MAX_NUM, dtype=np.float64)
        for d in draws:
            for n in d:
                nu[int(n) - 1] += 1.0
        combo = top7_from_freq(nu)
    return combo


def main():
    draws = load_draws()
    next_combo = predict_next(draws)
    if is_degenerate(next_combo):
        raise SystemExit("degenerisan next (uzastopni/AP) — zaustavljen pre ispisa")
    print(next_combo)


if __name__ == "__main__":
    main()



"""
[1, x, 3, y, 14, z, 39]
"""



"""
v14: transport_structure_v2 — samo spectral + rank ρ>0.99.
"""



"""
21 teorija

fisher_voronoi → v1, v2
dual_observability → v3
v4 se pozivao na W₂/stabilnost — slabo / nije strogo
entropy_along_geodesic → v5
velocity_asymmetry (+ delom lie_generator_structure) → v6
brenier_uniqueness (+ delom rank_orientation) → v7

kantorovich_duality
cyclical_monotonicity
displacement_interpolation
displacement_concavity
wasserstein_metric (strogo)
transport_structure
transport_structure_v2
transport_stability
stability_of_maps
monge_kantorovich_equivalence
lie_generator_structure (pun T10)
fisher_boundary
hybrid_observability
tangent_bundle
global_optimality
"""



"""
Kratko, o repou:

21 PVS teorija — sve su prošle kroz v1–v22 (neke ranije labavo: naročito v3/v4; rank_orientation je ušao uz Brenier u v7).
Repo je o spektralnom OT / LUCES (ESP32), ne o lotou — 7/39 je naša mapa, ne Stosićev domen.
Najčistije jezgro oko Fisher–Voronoi, Brenier/CM, W₂, T10 (lie_generator_structure). global_optimality je samo aksiomi + lema (bez teorema).
Empirija u PVS-u (bootovi, κ, Monge fraction) ne prenosi se automatski na CSV — samo struktura ideja.
"""
