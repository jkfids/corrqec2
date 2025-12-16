import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sinter
import seaborn as sns


def calc_xi(a, b):
    return -1 / np.log(1 - a - b)


def xi_to_Delta(xi):
    if isinstance(xi, np.ndarray):
        out = np.empty_like(xi)
        for i in range(len(xi)):
            if xi[i] == 0:
                out[i] = 1
            else:
                out[i] = 1 - np.exp(-1 / xi[i])
    else:
        if xi == 0:
            out = 1
        else:
            out = 1 - np.exp(-1 / xi)
    return out


def Delta_to_xi(Delta):
    if isinstance(Delta, np.ndarray):
        out = np.empty_like(Delta)
        for i in range(len(Delta)):
            if Delta[i] == 0:
                out[i] = np.inf
            else:
                out[i] = -1 / np.log(1 - Delta[i])
    else:
        if Delta == 0:
            out = np.inf
        else:
            out = -1 / np.log(1 - Delta)
    return out


def calc_per_round(per_shot: float, rounds: int):
    return 0.5 * (1 - (1 - 2 * per_shot) ** (1 / rounds))


def binomial_interval(failures, shots, level=0.95):
    alpha = 0.01
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
        "/home/fidel/Projects/corrqec2/data/experiment2_results.csv"
    )

    # Prepare data for plotting
    rounds_list = sorted(
        set(task.json_metadata["experiment_args"]["rounds"] for task in tasks)
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

    # Set cutoff for logical error rate
    cutoff = 1e-8

    results_dict = {rounds: {xi: None for xi in xis} for rounds in rounds_list}

    for task in tasks:
        model_params = task.json_metadata["noise_model_args"]["model_params"]
        rounds = task.json_metadata["experiment_args"]["rounds"]
        a, b = model_params["a"], model_params["b"]
        xi = int(np.round(calc_xi(a, b)))
        shots = task.shots
        errors = task.errors
        per_shot_interval = binomial_interval(errors, shots)
        # per_round_interval = binomial_interval_per_round(errors, shots, rounds)
        results_dict[rounds][xi] = per_shot_interval

    # Remove no. rounds = 40 due to insufficient data
    del results_dict[40]
    rounds_list.remove(40)

    xi_sorted = sorted(results_dict[rounds_list[0]].keys())
    distance_sorted = sorted(results_dict.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)
    colors = sns.color_palette("muted")

    for i, rounds in enumerate(rounds_list):
        # Cut off missing data points
        xi_sorted_i = [xi for xi in xi_sorted if results_dict[rounds][xi][0] > cutoff]
        center = [results_dict[rounds][xi][0] for xi in xi_sorted_i]
        lower = [results_dict[rounds][xi][1] for xi in xi_sorted_i]
        upper = [results_dict[rounds][xi][2] for xi in xi_sorted_i]
        ax1.plot(
            xi_sorted_i,
            center,
            marker=".",
            label=f"$N_r={rounds}$",
            linestyle="-",
            linewidth=0.8,
            color=colors[i],
        )
        ax1.fill_between(
            xi_sorted_i, lower, upper, alpha=0.5, linewidth=0, color=colors[i]
        )
    ax1.set_ylabel("Logical error rate (per shot)")
    ax1.set_xlabel("Correlation length, $\\xi$")
    ax1.legend(loc="lower right")
    ax1.set_xticks([1, 5, 10, 15, 20, 25])
    ax1.set_ylim(5e-8, 2.5e-2)

    secax1 = ax1.secondary_xaxis("top", functions=(xi_to_Delta, Delta_to_xi))
    secax1.set_xlabel("Spectral gap, $\\Delta$")
    delta_ticks = [0.6, 0.2, 0.1, 0.06, 0.04]
    secax1.set_xticks(delta_ticks)
    secax1.tick_params(direction="in", width=0.6)

    for j, xi in enumerate(xis):
        # Cut off missing data points
        distance_sorted_i = [
            distance
            for distance in distance_sorted
            if results_dict[distance][xi][0] > cutoff
        ]
        center = [results_dict[distance][xi][0] for distance in distance_sorted_i]
        lower = [results_dict[distance][xi][1] for distance in distance_sorted_i]
        upper = [results_dict[distance][xi][2] for distance in distance_sorted_i]
        ax2.errorbar(
            distance_sorted_i,
            center,
            yerr=[
                np.array(center) - np.array(lower),
                np.array(upper) - np.array(center),
            ],
            fmt=".",
            linewidth=0.8,
            color=colors[j],
        )

        # Exponentional fits
        logy = np.log(center)

        # Fit excluding the first point when xi>1
        if xi == 1:
            b, a = np.polyfit(distance_sorted_i[:4], logy[:4], 1)
        else:
            b, a = np.polyfit(distance_sorted_i[:], logy[:], 1)

        x_fit = np.linspace(5, 40, 100)
        y_fit = np.exp(a + b * x_fit)

        ax2.plot(
            x_fit,
            y_fit,
            linestyle="--",
            linewidth=0.8,
            color=colors[j],
        )

        # For legend
        ax2.errorbar(
            [],
            [],
            yerr=[[], []],
            fmt=".",
            linestyle="--",
            linewidth=0.8,
            color=colors[j],
            label=f"$\\xi={xi}$",
        )

    ax2.legend(loc="lower left")
    ax2.set_xlabel("Number of rounds, $N_r$")
    ax2.set_xlim(3.5, 36.5)

    ax1.text(-0.155, 1.14, "(a)", transform=ax1.transAxes, va="top", ha="left", size=9)
    ax2.text(-0.07, 1.14, "(b)", transform=ax2.transAxes, va="top", ha="left", size=9)

    for ax in (ax1, ax2):
        ax.grid(axis="y", alpha=0.5)
        ax.semilogy()
        # ax.spines["top"].set_visible(False)
        # ax.spines["right"].set_visible(False)
        ax.tick_params(direction="in", which="both", width=0.6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.09)
    fig.savefig(
        "./project/paper/figures/experiment2.pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.00,
    )
