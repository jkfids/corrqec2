import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import sinter


def calc_xi(a, b):
    return -1 / np.log(1 - a - b)


def xi_to_Delta(xi):
    return 1 - np.exp(-1 / xi)


def Delta_to_xi(Delta):
    return -1 / np.log(1 - Delta)


def calc_per_round(per_shot: float, rounds: int):
    return 0.5 * (1 - (1 - 2 * per_shot) ** (1 / rounds))


def binomial_interval(failures, shots, level=0.68):
    alpha = 0.5
    a_post = failures + alpha
    b_post = shots - failures + alpha
    center = a_post / (a_post + b_post)
    lower = beta.ppf((1 - level) / 2, a_post, b_post)
    upper = beta.ppf(1 - (1 - level) / 2, a_post, b_post)
    return np.array([center, lower, upper])


def binomial_interval_per_round(failures, shots, rounds, level=0.95):
    per_shot_interval = binomial_interval(failures, shots, level)
    per_round_interval = calc_per_round(per_shot_interval, rounds)
    return per_round_interval


if __name__ == "__main__":
    # Configure matplotlib
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )

    # Load data from CSV
    tasks = sinter.stats_from_csv_files(
        "/home/fidel/Projects/corrqec2/data/experiment1_results.csv"
    )

    # Prepare data for plotting
    distances = sorted(
        set(task.json_metadata["experiment_args"]["distance"] for task in tasks)
    )
    xis = sorted(
        set(
            int(
                np.round(
                    calc_xi(
                        task.json_metadata["noise_model_args"]["model_params"]["a"],
                        task.json_metadata["noise_model_args"]["model_params"]["b"],
                    ),
                    1,
                )
            )
            for task in tasks
        )
    )

    results_dict = {distance: {xi: None for xi in xis} for distance in distances}

    for task in tasks:
        distance = task.json_metadata["experiment_args"]["distance"]
        model_params = task.json_metadata["noise_model_args"]["model_params"]
        rounds_mult = int(task.json_metadata["experiment_args"]["rounds"][0])
        n_rounds = distance * rounds_mult
        a, b = model_params["a"], model_params["b"]
        xi = int(np.round(calc_xi(a, b)))
        shots = task.shots
        errors = task.errors
        per_round_interval = binomial_interval_per_round(errors, shots, n_rounds)
        # per_shot = errors / shots
        # per_round = calc_per_round(per_shot, n_rounds)
        results_dict[distance][xi] = per_round_interval

    fig1, ax1 = plt.subplots()
    for distance in distances:
        xi_sorted = sorted(results_dict[distance].keys())
        center = [results_dict[distance][xi][0] for xi in xi_sorted]
        lower = [results_dict[distance][xi][1] for xi in xi_sorted]
        upper = [results_dict[distance][xi][2] for xi in xi_sorted]
        ax1.plot(xi_sorted, center, marker="o", label=f"D={distance}")
        ax1.fill_between(xi_sorted, lower, upper, alpha=0.5, linewidth=0)
        ax1.legend()
        ax1.semilogy()
        ax1.grid(True)
        ax1.set_xlabel("Correlation length $\\xi$")
        ax1.set_ylabel("Logical error rate per round $p_{L}$")

        fig1.savefig("./project/paper/figures/experiment1a.png", dpi=600)

    fig2, ax2 = plt.subplots()
    for xi in xis:
        distance_sorted = sorted(results_dict.keys())
        center = [results_dict[distance][xi][0] for distance in distance_sorted]
        lower = [results_dict[distance][xi][1] for distance in distance_sorted]
        upper = [results_dict[distance][xi][2] for distance in distance_sorted]
        ax2.scatter(distance_sorted, center, marker="o", label=f"$\\xi$={xi}")
        ax2.fill_between(distance_sorted, lower, upper, alpha=0.5, linewidth=0)
        ax2.legend()
        ax2.semilogy()
        ax2.grid(True)
        ax2.set_xlabel("Distance $d$")
        ax2.set_ylabel("Logical error rate per round $p_{L}$")

        fig2.savefig("./project/paper/figures/experiment1b.png", dpi=600)
