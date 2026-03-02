import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sinter
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


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
        "/home/fidel/Projects/corrqec2/data/experiment4_results.csv"
    )

    # Prepare data for plotting
    distances = sorted(
        set(task.json_metadata["experiment_args"]["distance"] for task in tasks)
    )
    thetas = sorted(
        set(
            float(
                np.round(
                    task.json_metadata["noise_model_args"]["model_params"]["theta"]
                    / np.pi,
                    5,
                )
            )
            for task in tasks
        )
    )

    results_dict = {
        distance: {theta: None for theta in thetas} for distance in distances
    }

    for task in tasks:
        distance = task.json_metadata["experiment_args"]["distance"]
        theta = float(
            np.round(
                task.json_metadata["noise_model_args"]["model_params"]["theta"] / np.pi,
                5,
            )
        )
        rounds_mult = int(task.json_metadata["experiment_args"]["rounds"][0])
        n_rounds = distance * rounds_mult
        shots = task.shots
        errors = task.errors
        per_round_interval = binomial_interval_per_round(errors, shots, n_rounds)
        results_dict[distance][theta] = per_round_interval

    theta_sorted = sorted(results_dict[distance].keys())
    distance_sorted = sorted(results_dict.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3), sharey=True)
    axs = [ax1, ax2]
    colors = sns.color_palette("muted")

    for i, distance in enumerate(distance_sorted):
        # Cut off missing data points
        theta_sorted_i = [
            theta for theta in theta_sorted if results_dict[distance][theta] is not None
        ]
        center = [results_dict[distance][theta][0] for theta in theta_sorted_i]
        lower = [results_dict[distance][theta][1] for theta in theta_sorted_i]
        upper = [results_dict[distance][theta][2] for theta in theta_sorted_i]
        ax1.plot(
            theta_sorted_i,
            center,
            marker=".",
            label=f"$d={distance}$",
            linestyle="-",
            linewidth=0.8,
            color=colors[i],
        )
        ax1.fill_between(
            theta_sorted_i, lower, upper, alpha=0.5, linewidth=0, color=colors[i]
        )
    ax1.set_ylim(2e-9, 1.2e-2)
    ax1.set_xlim(0.185, 0.5)
    ax1.legend(loc="lower right")
    ax1.set_ylabel("Logical error rate (per round)")
    ax1.set_xlabel("Controlled-rotation angle, $\\theta$")
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%g$\\pi$"))
    ax1.xaxis.set_major_locator(MultipleLocator(base=0.1))

    for j, theta in enumerate(theta_sorted):
        # Cut off missing data points
        distance_sorted_j = [
            distance
            for distance in distance_sorted
            if results_dict[distance][theta] is not None
        ]
        center = [results_dict[distance][theta][0] for distance in distance_sorted_j]
        lower = [results_dict[distance][theta][1] for distance in distance_sorted_j]
        upper = [results_dict[distance][theta][2] for distance in distance_sorted_j]
        ax2.errorbar(
            distance_sorted_j,
            center,
            yerr=[
                np.array(center) - np.array(lower),
                np.array(upper) - np.array(center),
            ],
            fmt=".",
            linewidth=0.8,
            color=colors[j],
        )

        # Exponential fits
        logy = np.log(center)
        b, a = np.polyfit(distance_sorted_j, logy, 1)
        x_fit = np.linspace(4, 18, 100)
        y_fit = np.exp(a + b * x_fit)
        ax2.plot(
            x_fit,
            y_fit,
            linestyle="--",
            color=colors[j],
            linewidth=0.8,
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
            label=f"$\\theta={theta}\\pi$",
        )

    ax2.legend(loc="lower left")
    ax2.set_xlabel("Code distance, $d$")
    ax2.set_xlim(4.5, 17.5)
    ax2.set_xticks([5, 7, 9, 11, 13, 15, 17])

    ax1.text(-0.155, 1.08, "(a)", transform=ax1.transAxes, va="top", ha="left", size=10)
    ax2.text(-0.07, 1.08, "(b)", transform=ax2.transAxes, va="top", ha="left", size=10)

    for ax in (ax1, ax2):
        ax.semilogy()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="in", which="both", width=0.6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.09)
    fig.savefig(
        "./project/paper/figures/experiment4.pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.00,
    )
