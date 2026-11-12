"""Uniform safety-margin allocation for the global-robust baseline (B1): every edge's
planned travel time is inflated by the same multiple of that edge's own standard
deviation, independently of any other edge. z is tuned by cross-validation (see
src/tuning.py); it is the only free parameter here.
"""

import numpy as np


def effective_time_matrix(mostlikely: np.ndarray, sd: np.ndarray, safety_z: float) -> np.ndarray:
    """M + z*SD on every off-diagonal edge, diagonal left at 0."""
    eff = mostlikely + safety_z * sd
    np.fill_diagonal(eff, 0.0)
    return eff


if __name__ == "__main__":
    from src.data_loading import load_all_days
    from src.reconstruction import reconstruct, calibrate_beta_pert

    days = load_all_days()
    for d, inst in list(days.items())[:3]:
        recon = reconstruct(inst.time_optimistic, inst.time_mostlikely, inst.time_pessimistic, "order_statistics")
        params = calibrate_beta_pert(recon)
        eff = effective_time_matrix(recon.mostlikely, params.sd, safety_z=1.0)
        delta = eff - recon.mostlikely
        offdiag = ~np.eye(len(delta), dtype=bool)
        print(f"Day {d}: z=1.0 margin adds mean={delta[offdiag].mean():.3f}min, "
              f"max={delta[offdiag].max():.3f}min over the raw most-likely estimate")
