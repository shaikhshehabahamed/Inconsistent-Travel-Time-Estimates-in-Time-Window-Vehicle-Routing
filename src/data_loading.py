"""Load the Attica last-mile delivery dataset (Vrani et al. 2025, Data in Brief) into
clean, validated per-day structures. Depot is index 0 in every matrix; customers are
NODE_ID 1..n, matching the row order in orders.xlsx."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
N_DAYS = 9
LAT_TIERS = (180, 300, 360)  # deadline classes observed in the data (minutes)


@dataclass(frozen=True)
class DayInstance:
    day: int
    orders: pd.DataFrame  # index = NODE_ID (1..n), columns WEIGHT, VOLUME, SERVICE_TIME, EAT, LAT
    distance: np.ndarray  # (n+1, n+1) km, index 0 = depot, asymmetric
    time_optimistic: np.ndarray  # (n+1, n+1) minutes, raw (may violate O<=M<=P)
    time_mostlikely: np.ndarray
    time_pessimistic: np.ndarray

    @property
    def n_customers(self) -> int:
        return len(self.orders)

    @property
    def n_nodes(self) -> int:
        return len(self.orders) + 1


def _load_orders(day: int) -> pd.DataFrame:
    xl_path = DATA_ROOT / "orders" / "orders.xlsx"
    df = pd.read_excel(xl_path, sheet_name=f"Day {day}")
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df = df.set_index("NODE_ID").sort_index()
    df = df[["WEIGHT", "VOLUME", "SERVICE_TIME", "EAT", "LAT"]]
    if not df.index.equals(pd.RangeIndex(1, len(df) + 1)):
        raise ValueError(f"Day {day}: NODE_ID is not a contiguous 1..n range")
    bad_lat = ~df.LAT.isin(LAT_TIERS)
    if bad_lat.any():
        raise ValueError(f"Day {day}: unexpected LAT values {df.LAT[bad_lat].unique()}")
    if (df.EAT != 0).any():
        raise ValueError(f"Day {day}: unexpected nonzero EAT")
    return df


def _load_matrix(day: int, name: str) -> np.ndarray:
    path = DATA_ROOT / "time_and_distance_matrices" / f"day_{day}" / f"{name}_{day}.xlsx"
    df = pd.read_excel(path, header=0, index_col=0)
    return df.to_numpy(dtype=float)


def load_day(day: int) -> DayInstance:
    orders = _load_orders(day)
    n = len(orders)
    expected_shape = (n + 1, n + 1)

    distance = _load_matrix(day, "distance_matrix")
    o = _load_matrix(day, "time_matrix_optimistic")
    m = _load_matrix(day, "time_matrix_mostlikely")
    p = _load_matrix(day, "time_matrix_pessimistic")

    for name, mat in [("distance", distance), ("optimistic", o), ("mostlikely", m), ("pessimistic", p)]:
        if mat.shape != expected_shape:
            raise ValueError(f"Day {day}: {name} matrix shape {mat.shape} != expected {expected_shape}")
        if np.any(np.diag(mat) != 0):
            raise ValueError(f"Day {day}: {name} matrix has nonzero diagonal")
        if np.any(np.isnan(mat)):
            raise ValueError(f"Day {day}: {name} matrix contains NaN")
        if np.any(mat < 0):
            raise ValueError(f"Day {day}: {name} matrix contains negative values")

    return DayInstance(
        day=day, orders=orders, distance=distance,
        time_optimistic=o, time_mostlikely=m, time_pessimistic=p,
    )


def load_all_days() -> dict[int, DayInstance]:
    return {d: load_day(d) for d in range(1, N_DAYS + 1)}


if __name__ == "__main__":
    days = load_all_days()
    for d, inst in days.items():
        print(f"Day {d}: {inst.n_customers} customers, {inst.n_nodes} nodes "
              f"(incl. depot), weight sum={inst.orders.WEIGHT.sum():.1f}kg, "
              f"volume sum={inst.orders.VOLUME.sum():.3f}m3")
