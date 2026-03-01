import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns


def load_results(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    # Get all distances and thetas assuming all distances have the same set of thetas
    distances = []
    thetas = []
    for result in data:
        distances.append(result["distance"])
        thetas.append(result["theta"])  # Convert to units of π
    distances = sorted(set(distances))
    thetas = sorted(set(thetas))

    # Organise results by distance and theta
    results_dict = {d: {t: {} for t in thetas} for d in distances}
    for result in data:
        d = result["distance"]
        t = np.round(result["theta"] / np.pi, 4)  # Convert to units of π
        mean = result["mean"]
        var = result["var"]
        corr = result["corr"]
        results_dict[d][t] = {"mean": mean, "var": var, "corr": corr}

    # Round thetas to 5 decimal places to avoid floating point issues in dict keys
    thetas = [np.round(t / np.pi, 4) for t in thetas]

    return results_dict, distances, thetas


def n_qubits(distance):
    return 2 * (distance**2) - 1


def calc_mean_sem(x):
    x = np.asarray(x, dtype=np.float64)
    mean = float(x.mean())
    sem = float(x.std(ddof=1) / np.sqrt(len(x)))
    return mean, sem


def process_results(results_dict, distances, thetas):
    Y1_dict = {d: [] for d in distances}
    Y2_dict = {d: [] for d in distances}
    Y3_dict = {d: [] for d in distances}

    for d in distances:
        for t in thetas:
            rho_mean = results_dict[d][t]["mean"]
            rho_mean_m, _ = calc_mean_sem(rho_mean)
            Y1_dict[d].append(rho_mean_m)

            rho_var = results_dict[d][t]["var"]
            rho_var_m, _ = calc_mean_sem(rho_var)
            # Scale variance by number of qubits
            Y2_dict[d].append(rho_var_m * n_qubits(d))

            rho_acorr = results_dict[d][t]["corr"]
            corr_means = rho_acorr.mean(axis=0)
            tau = fit_autocorrs(corr_means, hi=0.8, lo=0.01)
            Y3_dict[d].append(tau)

    return Y1_dict, Y2_dict, Y3_dict


def fit_autocorrs(autocorrs, hi=0.5, lo=0.01):

    autocorrs = np.asarray(autocorrs, dtype=np.float64)
    ts = np.arange(autocorrs.size)

    mask = (autocorrs >= lo) & (autocorrs <= hi) & (ts >= 1)
    if mask.sum() < 3:
        raise ValueError("Not enough points for fitting.")

    print(autocorrs[mask])
    y = np.log(autocorrs[mask])
    t = ts[mask]

    # Least squares fit
    b, a = np.polyfit(t, y, 1)
    tau = -1.0 / b
    return tau


def plot_autocorr(results_dict, distance, thetas):
    Ys = []
    for t in thetas:
        corrs = results_dict[distance][t]["corr"]  # shape (shots, max_lag+1)
        corr_means = corrs.mean(axis=0)  # shape (max_lag+1,)
        Ys.append(corr_means[:20])

    fig, ax = plt.subplots()
    for i, t in enumerate(thetas):
        ax.plot(np.arange(len(Ys[i])), Ys[i], label=f"θ={t:.2f}π")
    ax.semilogy()
    ax.legend()
    fig.savefig("test2.png", dpi=300)


def plot_figures(Y1_dict, Y2_dict, Y3_dict, distances, thetas):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.2, 3.2))
    axs = [ax1, ax2, ax3]
    colors = sns.color_palette("muted")

    for i, d in enumerate(distances):
        ax1.plot(thetas, Y1_dict[d], label=f"d={d}", color=colors[i])
        ax2.plot(thetas, Y2_dict[d], label=f"d={d}", color=colors[i])
        # ax1.errorbar(
        #     thetas, Y1_dict[d][0], yerr=Y1_dict[d][1], label=f"d={d}", color=colors[i]
        # )
        # ax2.errorbar(
        #     thetas, Y2_dict[d][0], yerr=Y2_dict[d][1], label=f"d={d}", color=colors[i]
        # )
        ax3.plot(thetas, Y3_dict[d], label=f"d={d}", color=colors[i])

    ax3.legend()

    fig.savefig("test1.png", dpi=300)


def main():
    filepath = "/home/fidel/Projects/corrqec2/data/experiment3_results.pkl"
    results_dict, distances, thetas = load_results(filepath)
    Y1_dict, Y2_dict, Y3_dict = process_results(results_dict, distances, thetas)
    plot_figures(Y1_dict, Y2_dict, Y3_dict, distances, thetas)
    plot_autocorr(
        results_dict, distance=15, thetas=[0.0, 0.38, 0.5, 0.6, 0.7, 0.8, 0.9]
    )


if __name__ == "__main__":
    main()
