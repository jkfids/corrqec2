"""
Script to generate parameter CSV file for experiment 3.
"""

import csv
from pathlib import Path
import numpy as np
from itertools import product

DIR = Path(__file__).resolve().parent
OUT = DIR / "experiment3_params.csv"

# DISTANCES = [7, 11]
DISTANCES = [15]
ROUNDS = 1_000_000
SHOTS = 20

THETAS_MAJOR_MIN = 0.0
THETAS_MAJOR_MAX = 1.0
THETAS_MAJOR_N = 11
THETAS_MINOR_MIN = 0.40
THETAS_MINOR_MAX = 0.60
THETAS_MINOR_N = 11

A_LIST = [0.001]
B_LIST = [0.5]


def gen_thetas(
    thetas_major_max: int,
    thetas_major_min: int,
    thetas_major_n: int,
    thetas_minor_max: int,
    thetas_minor_min: int,
    thetas_minor_n: int,
) -> list:
    thetas_major = np.linspace(
        thetas_major_min, thetas_major_max, thetas_major_n
    ).round(5)
    thetas_minor = np.linspace(
        thetas_minor_min, thetas_minor_max, thetas_minor_n
    ).round(5)
    thetas = np.union1d(thetas_major, thetas_minor).tolist()
    return thetas


def gen_rows(
    distances: list,
    rounds: int,
    shots: int,
    thetas: list,
    a_list: list,
    b_list: list,
):
    for d, theta, a, b in product(distances, thetas, a_list, b_list):
        yield (d, rounds, shots, theta, a, b)


def write_csv(rows: list, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["distance", "rounds", "shots", "theta", "a", "b"])
        w.writerows(rows)


def main():
    thetas = gen_thetas(
        THETAS_MAJOR_MAX,
        THETAS_MAJOR_MIN,
        THETAS_MAJOR_N,
        THETAS_MINOR_MAX,
        THETAS_MINOR_MIN,
        THETAS_MINOR_N,
    )
    rows = gen_rows(
        DISTANCES,
        ROUNDS,
        SHOTS,
        thetas,
        A_LIST,
        B_LIST,
    )
    rows_list = list(rows)
    if OUT.exists():
        print(f"Overwriting existing file {OUT}")
    write_csv(rows_list, OUT)
    print(f"Wrote {len(rows_list)} rows to {OUT} for parameters:")
    print(f"DISTANCES: {DISTANCES}")
    print(f"ROUNDS: {ROUNDS}")
    print(f"SHOTS: {SHOTS}")
    print(f"THETAS: {thetas}")
    print(f"A_LIST: {A_LIST}")
    print(f"B_LIST: {B_LIST}")
    print()


if __name__ == "__main__":
    main()
