"""Recover a data-driven spatial clustering of each day's nodes directly from the
travel-time matrix. The only remaining consumer is simulation.py's Monte Carlo
ground-truth model, which gives edges sharing a cluster a correlated random shock rather
than treating every edge's delay risk as independent -- applied identically when scoring
every method (B0, B1), so it is a property of how outcomes are honestly evaluated, not
of any one method's routing decision. The dataset's own 6-sector maps (Athens Center/
West/North/East/South, Piraeus) are shipped only as pre-rendered images with day-level
aggregate counts, not as a field joinable to individual orders -- so clusters are
recovered from the matrices themselves, which is also what the diagnosed SD/CV structure
(Section 01 of the research brief) is actually built from.

Clustering runs on the corrected most-likely matrix (not raw distance): travel time
correlates more strongly with the diagnosed uncertainty (r=0.76-0.90 vs r=0.36-0.83 for
raw distance across the 9 days), so it is the more relevant basis for a clustering whose
purpose is identifying which edges plausibly share a correlated local traffic shock.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


@dataclass(frozen=True)
class ClusterAssignment:
    n_clusters: int
    labels: np.ndarray  # (n_nodes,), includes depot at index 0
    silhouette: float


def _symmetrize(mat: np.ndarray) -> np.ndarray:
    sym = (mat + mat.T) / 2.0
    np.fill_diagonal(sym, 0.0)
    return sym


def cluster_nodes(time_mostlikely: np.ndarray, n_clusters: int) -> ClusterAssignment:
    """Cluster all nodes (including depot, index 0) via complete-linkage agglomerative
    clustering on the symmetrized travel-time matrix. Complete linkage is used instead of
    average/single because this is the least prone to chaining a couple of outlier nodes
    off into their own trivial cluster -- which average linkage did in early testing
    (e.g. one day silhouette-selected a 67-node + 1-node split)."""
    dist = _symmetrize(time_mostlikely)
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="complete")
    labels = model.fit_predict(dist)
    sil = silhouette_score(dist, labels, metric="precomputed") if n_clusters > 1 else float("nan")
    return ClusterAssignment(n_clusters=n_clusters, labels=labels, silhouette=sil)


def select_n_clusters(time_mostlikely: np.ndarray, k_range: range = range(2, 9),
                       min_cluster_size: int = 5) -> ClusterAssignment:
    """Pick k by silhouette score, restricted to candidates where every cluster has at
    least `min_cluster_size` nodes. A cluster with only 1-2 members has no real
    cluster-mates to share a correlated shock with in the Monte Carlo evaluation, which
    defeats the point of clustering at all -- so those candidates are excluded from
    consideration entirely rather than merely disfavoured. If no k in range satisfies
    the constraint, falls back to the smallest k in range."""
    candidates = []
    for k in k_range:
        assign = cluster_nodes(time_mostlikely, k)
        min_size = int(np.bincount(assign.labels).min())
        if min_size >= min_cluster_size:
            candidates.append(assign)
    if not candidates:
        return cluster_nodes(time_mostlikely, k_range.start)
    return max(candidates, key=lambda a: a.silhouette)


def same_cluster_mask(labels: np.ndarray) -> np.ndarray:
    """(n,n) boolean matrix, True where nodes i and j share a cluster."""
    return labels[:, None] == labels[None, :]


if __name__ == "__main__":
    from src.data_loading import load_all_days
    from src.reconstruction import reconstruct

    days = load_all_days()
    for d, inst in days.items():
        recon = reconstruct(inst.time_optimistic, inst.time_mostlikely, inst.time_pessimistic, "order_statistics")
        best = select_n_clusters(recon.mostlikely)
        sizes = np.bincount(best.labels)
        print(f"Day {d}: best k={best.n_clusters}, silhouette={best.silhouette:.3f}, cluster sizes={list(sizes)}")
