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
        thetas.append(result["theta"] / np.pi)  # Convert to units of π
    distances = sorted(set(distances))
    thetas = sorted(set(thetas))

    # Organise results by distance and theta
    results_dict = {d: {t: {} for t in thetas} for d in distances}
    for result in data:
        d = result["distance"]
        t = result["theta"] / np.pi  # Convert to units of π
        mean = result["mean"]
        var = result["var"]
        corr = result["corr"]
        results_dict[d][t] = {"mean": mean, "var": var, "corr": corr}

    return results_dict, distances, thetas


def n_qubits(distance):
    return 2 * (distance**2) - 1


def calc_mean_sem(x):
    x = np.asarray(x, dtype=np.float64)
    mean = float(x.mean())
    sem = float(x.std(ddof=1) / np.sqrt(len(x)))
    return mean, sem


def process_results(results_dict, distances, thetas):
    Y0_dict = {d: [] for d in distances}
    Y1_dict = {d: [] for d in distances}

    for d in distances:
        for t in thetas:
            rho_mean = results_dict[d][t]["mean"]
            rho_mean_m, _ = calc_mean_sem(rho_mean)
            Y0_dict[d].append(rho_mean_m)

            rho_var = results_dict[d][t]["var"]
            rho_var_m, _ = calc_mean_sem(rho_var)
            Y1_dict[d].append(
                rho_var_m * n_qubits(d)
            )  # Scale variance by number of qubits

    return Y0_dict, Y1_dict


def plot_figures(Y0_dict, Y1_dict, distances, thetas):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.2, 3.2))
    axs = [ax1, ax2, ax3]
    colors = sns.color_palette("muted")

    for i, d in enumerate(distances):
        ax1.plot(thetas, Y0_dict[d], label=f"d={d}", color=colors[i])
        ax2.plot(thetas, Y1_dict[d], label=f"d={d}", color=colors[i])
        # ax1.errorbar(
        #     thetas, Y0_dict[d][0], yerr=Y0_dict[d][1], label=f"d={d}", color=colors[i]
        # )
        # ax2.errorbar(
        #     thetas, Y1_dict[d][0], yerr=Y1_dict[d][1], label=f"d={d}", color=colors[i]
        # )

    ax2.legend()

    fig.savefig("test.png", dpi=300)


def main():
    filepath = "/home/fidel/Projects/corrqec2/data/experiment3_results.pkl"
    results_dict, distances, thetas = load_results(filepath)
    Y0_dict, Y1_dict = process_results(results_dict, distances, thetas)
    plot_figures(Y0_dict, Y1_dict, distances, thetas)


# def main():
#     with open("/home/fidel/Projects/corrqec2/data/experiment3_results.pkl", "rb") as f:
#         data = pickle.load(f)

#     results_dict, distances, thetas = load_results(
#         "/home/fidel/Projects/corrqec2/data/experiment3_results.pkl"
#     )
#     print(process_mean_density(results_dict, distances, thetas))
# for result in data:
#     distance = result["distance"]
#     theta = result["theta"]
#     mean = result["mean"]
#     var = result["var"]
#     corr = result["corr"]

#     if distance == 13:
#         if theta in [0.1, 0.2, 0.3, 0.4, 0.5]:
#             mean_corr = corr.mean(axis=0)
#             results_dict[theta] = [mean_corr[:20]]

# print(results_dict.keys())

# fig, ax = plt.subplots()

# thetas = list(results_dict.keys())
# corr_lists = list(results_dict.values())
# pairs = sorted(zip(thetas, corr_lists), key=lambda x: x[0])
# theta_sorted, corr_sorted = map(list, zip(*pairs))

# for theta, corr_list in zip(theta_sorted, corr_sorted):
#     for corr in corr_list:
#         # sns.lineplot(x=np.arange(len(corr)), y=corr, label=f"θ={theta}π", ax=ax)
#         ax.plot(
#             np.arange(len(corr)),
#             corr,
#             "o",
#             markersize=4,
#             label=f"θ={theta}π",
#             linestyle="-",
#         )
#         ax.semilogy()
#         ax.legend()

# fig.savefig("test.png", dpi=300)


if __name__ == "__main__":
    main()
