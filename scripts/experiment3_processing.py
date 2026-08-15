from pathlib import Path
import argparse
import pickle

import numpy as np

# Sweep outputs are written here; on a cluster this is typically scratch
# space. Change this one line to run elsewhere.
RESULTS_DIR = Path.home() / "mx95_scratch2" / "jkam" / "corrqec2_results"


def load_error_matrix(filepath):
    with np.load(filepath) as data:
        bitpacked = data["error_matrix"]
        shape = tuple(data["shape"])
        pack_axis = int(data["pack_axis"])
        dtype = np.dtype(str(data["dtype"]))

        error_matrix = np.unpackbits(bitpacked, axis=pack_axis)
        error_matrix = error_matrix.take(
            indices=range(shape[pack_axis]), axis=pack_axis
        ).astype(dtype)

        distance = int(data["distance"])
        rounds = int(data["rounds"])
        shots = int(data["shots"])
        theta = float(data["theta"])
        a = float(data["a"])
        b = float(data["b"])

    return error_matrix, distance, rounds, shots, theta, a, b


def crop_burnin(error_matrix, burnin):
    return error_matrix[:, :, burnin:]


def calc_mean_density(error_matrix):
    """Calculate the mean error density and its variance per shot."""
    # shots, qubits, rounds = error_matrix.shape
    error_matrix = error_matrix.astype(np.float64)  # ensure float
    rho_t = error_matrix.mean(axis=1)  # shape (shots, rounds)
    rho_mean = rho_t.mean(axis=1)  # shape (shots,)
    rho_var = rho_t.var(axis=1)  # shape (shots,)
    return rho_mean, rho_var


def calc_autocorr_1d(row, max_lag):
    """Calculate the autocorrelation of a 1D array up to a maximum lag via FFT."""
    row = row.astype(np.float64)  # ensure float
    row = row - np.mean(row)  # zero-mean
    L = len(row)
    n = 1 << (2 * L - 1).bit_length()  # n is the next power of 2 >= 2*L-1
    f = np.fft.rfft(row, n=n)
    acov = np.fft.irfft(f * np.conj(f), n=n)[:L]  # autocovariance
    acov /= L - np.arange(L)  # normalize by number of terms

    if acov[0] == 0:
        acorr = np.zeros(max_lag + 1)
    else:
        acorr = acov / acov[0]  # autocorrelation
    return acorr[: max_lag + 1]


def calc_autocorr(error_matrix, max_lag):
    """Calculate the mean autocorrelation per shot."""
    shots, qubits, rounds = error_matrix.shape

    rho_ts = error_matrix.mean(axis=1).astype(np.float64)  # shape (shots, rounds)
    rho_acorrs = np.empty((shots, max_lag + 1), dtype=np.float64)
    for i in range(shots):
        rho_acorrs[i] = calc_autocorr_1d(rho_ts[i], max_lag)

    return rho_acorrs


def calc_stats(error_matrix, burnin, max_lag):
    error_matrix = crop_burnin(error_matrix, burnin)
    rho_mean, rho_var = calc_mean_density(error_matrix)
    rho_acorrs = calc_autocorr(error_matrix, max_lag)
    return rho_mean, rho_var, rho_acorrs


def main():
    print("Processing experiment 3 error matrices...")
    resultsdir = RESULTS_DIR / "error_matrices"
    savedir = RESULTS_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--burnin", type=int, default=200_000, help="Burn-in period")
    parser.add_argument(
        "--max_lag", type=int, default=200, help="Autocorrelation max lag"
    )

    args = parser.parse_args()
    burnin = args.burnin
    max_lag = args.max_lag

    data = []
    for filepath in sorted(resultsdir.glob("*.npz")):
        error_matrix, distance, rounds, shots, theta, a, b = load_error_matrix(filepath)
        mean, var, corr = calc_stats(error_matrix, burnin=burnin, max_lag=max_lag)
        print(
            f"Processed {filepath.name}: d={distance}, r={rounds}, shots={shots}, theta={theta/np.pi}π, a={a}, b={b}"
        )

        result_dict = {
            "distance": distance,
            "rounds": rounds,
            "shots": shots,
            "theta": theta,
            "a": a,
            "b": b,
            "mean": mean,
            "var": var,
            "corr": corr,
        }
        data.append(result_dict)

    savedir.mkdir(parents=True, exist_ok=True)
    with open(savedir / "experiment3_results.pkl", "wb") as f:
        pickle.dump(data, f)
    print(f"Saved processed results to {savedir / 'experiment3_results.pkl'}")


if __name__ == "__main__":
    main()
