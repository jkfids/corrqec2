import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sinter
import seaborn as sns

FIGURE_DIR = "./manuscript/figures"
DATA_DIR = "./data"


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


def distance_to_qubits(distance):
    return 2 * distance**2 - 1


def qubits_to_distance(qubits):
    return np.round(np.sqrt((qubits + 1) / 2)).astype(int)


def distance_to_data_qubits(distance):
    return distance * distance


def data_qubits_to_distance(qubits):
    return np.round(np.sqrt(qubits)).astype(int)


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
        f"{DATA_DIR}/experiment1_results.csv"
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
        results_dict[distance][xi] = per_round_interval

    xi_sorted = sorted(results_dict[distance].keys())
    distance_sorted = sorted(results_dict.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)
    # cycle = plt.cycler(color=plt.cm.tab10.colors)
    # colors = cycle.by_key()["color"]
    colors = sns.color_palette("muted")

    for i, distance in enumerate(distances):
        # Cut off missing data points
        xi_sorted_i = [xi for xi in xi_sorted if results_dict[distance][xi] is not None]
        center = [results_dict[distance][xi][0] for xi in xi_sorted_i]
        lower = [results_dict[distance][xi][1] for xi in xi_sorted_i]
        upper = [results_dict[distance][xi][2] for xi in xi_sorted_i]
        ax1.plot(
            xi_sorted_i,
            center,
            marker=".",
            label=f"$d={distance}$",
            linestyle="-",
            linewidth=0.8,
            color=colors[i],
        )
        ax1.fill_between(
            xi_sorted_i, lower, upper, alpha=0.5, linewidth=0, color=colors[i]
        )
    ax1.set_ylabel("Logical error rate (per round)")
    ax1.set_xlabel("Correlation time, $\\xi$")
    ax1.legend(loc="lower right")
    ax1.set_xticks([1, 5, 10, 15, 20, 25])
    ax1.set_ylim(2e-9, 1.5e-4)

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
            if results_dict[distance][xi] is not None
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
            # label=f"$\\xi={xi}$",
        )

        # Exponentional fits
        logy = np.log(center)
        # sigma_logy = 0.5 * (np.log(upper) - np.log(lower))
        # w = 1.0 / sigma_logy**2

        # Fit excluding the first point when xi>1
        if xi == 1:
            b, a = np.polyfit(distance_sorted_i, logy, 1)
        else:
            b, a = np.polyfit(distance_sorted_i[1:], logy[1:], 1)

        x_fit = np.linspace(7, 20, 100)
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
    ax2.set_xlabel("Code distance, $d$")
    ax2.set_xlim(4, 20)
    ax2.set_xticks([5, 10, 15, 19])

    secax2 = ax2.secondary_xaxis(
        "top", functions=(distance_to_qubits, qubits_to_distance)
    )
    secax2.set_xlabel("Total no. qubits")
    qubit_ticks = [50, 100, 200, 400, 750]
    secax2.set_xticks(qubit_ticks)
    secax2.tick_params(direction="in", width=0.6)

    ax1.text(
        -0.155,
        1.14,
        "(a)",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        size=9,
        fontweight="bold",
    )
    ax2.text(
        -0.07,
        1.14,
        "(b)",
        transform=ax2.transAxes,
        va="top",
        ha="left",
        size=9,
        fontweight="bold",
    )

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
        f"{FIGURE_DIR}/experiment1.pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.00,
    )
