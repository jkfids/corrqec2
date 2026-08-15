"""Worked examples: single-qubit non-Markovian dynamics (Sec. 4.3).

Builds the single-slot process tensors of three system-environment Hamiltonian
families, applies the multi-time Pauli twirl to obtain the induced SPPs, and
plots the relative-entropy diagnostics and Pauli trajectory distributions.
"""

from itertools import product

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

from processtensor import (
    ProcessTensor,
    QuantumChannel,
    pauli,
    relative_entropy,
    unitary_from_hamiltonian,
)

FIGURE_DIR = "./manuscript/figures"

RHO_ENV = np.ones((2, 2), dtype=complex) / 2  # environment in |+><+|


def H_heisenberg(theta=0.5 * np.pi):
    return -theta * 0.5 * (pauli("XX") + pauli("YY") + pauli("ZZ"))


def H_crx(theta=0.5 * np.pi):
    return -theta * 0.5 * (pauli("XI") - pauli("XZ"))


def H_magnetic(theta=0.5 * np.pi):
    return H_heisenberg(0.5 * np.pi) - theta * (
        pauli("XI") + pauli("YI") + pauli("ZI")
    )


def gen_process_tensor(Hamiltonian, theta):
    """Single-slot process tensor of the system-environment unitary exp(-iH)."""
    U = unitary_from_hamiltonian(Hamiltonian(theta))
    return ProcessTensor.from_stinespring([U, U], RHO_ENV)


def multitime_twirl(pt):
    """Multi-time Pauli twirl of a process tensor.

    Averages the Choi matrix over conjugation by a Pauli at each time step,
    each acting jointly on that step's input and output space. Pauli matrices
    are symmetric up to a sign, so kron(P, P) fixes the conjugation.
    """
    choi = np.zeros_like(pt.choi)
    for labels in product("IXYZ", repeat=pt.steps):
        V = np.array([[1.0]], dtype=complex)
        for p in labels:
            V = np.kron(V, np.kron(pauli(p), pauli(p)))
        choi += V @ pt.choi @ V.conj().T
    return ProcessTensor.from_choi(choi / 4**pt.steps, pt.dims)


def trajectory_choi(pauli_string):
    """Unit-trace Choi matrix of a Pauli trajectory, one Pauli per time step."""
    choi = np.array([[1.0]], dtype=complex)
    for p in pauli_string:
        channel = QuantumChannel.from_unitary(pauli(p))
        choi = np.kron(choi, channel.choi / channel.trace)
    return choi


def calc_pauli_probs(pt):
    """Pauli trajectory probabilities Pr(P) of a process tensor."""
    strings = ["".join(s) for s in product("IXYZ", repeat=pt.steps)]
    rho = pt.choi / pt.trace
    probs = [np.trace(rho @ trajectory_choi(s)).real for s in strings]
    return probs, strings


def run_entropy_calcs(Hamiltonian, thetas):
    """GQMI of the process, GQMI of its twirl, and the twirl relative entropy."""
    gqmi = []
    gqmi_twirled = []
    twirl_entropy = []
    for theta in thetas:
        pt = gen_process_tensor(Hamiltonian, theta)
        twirled = multitime_twirl(pt)
        gqmi.append(pt.gqmi())
        gqmi_twirled.append(twirled.gqmi())
        twirl_entropy.append(
            relative_entropy(pt.choi / pt.trace, twirled.choi / twirled.trace)
        )

    return [gqmi, gqmi_twirled, twirl_entropy]


def run_pauli_prob_calcs(Hamiltonian, theta):
    pt = gen_process_tensor(Hamiltonian, theta)
    probs, strings = calc_pauli_probs(pt)
    return probs, strings


def plot_combined(S_lists1, S_lists2, S_lists3, thetas, probs1, probs2, probs3):
    fig, axs = plt.subplots(2, 3, figsize=(10.2, 6.2), sharey="row")
    colors = sns.color_palette("muted")

    entropy_axes = axs[0]
    probs_axes = axs[1]

    S_array = [S_lists1, S_lists2, S_lists3]
    probs_array = [probs1, probs2, probs3]
    strings = [A + B for A in ["I", "X", "Y", "Z"] for B in ["I", "X", "Y", "Z"]]

    for i in range(3):
        entropy_axes[i].plot(
            thetas / np.pi,
            S_array[i][0],
            color=colors[0],
            label="$\\rho:$ original, $\\sigma:$ marginalized",
        )
        entropy_axes[i].plot(
            thetas / np.pi,
            S_array[i][1],
            color=colors[1],
            label="$\\rho:$ twirled, $\\sigma:$ twirled marginalized",
        )
        entropy_axes[i].plot(
            thetas / np.pi,
            S_array[i][2],
            color=colors[2],
            linestyle="--",
            label="$\\rho:$ original, $\\sigma:$ twirled",
        )

        entropy_axes[i].xaxis.set_major_formatter(FormatStrFormatter("%g$\\pi$"))
        entropy_axes[i].xaxis.set_major_locator(
            matplotlib.ticker.MultipleLocator(base=1.0)
        )
        entropy_axes[i].set_xticks([0, 0.5, 1.0, 1.5, 2.0])
        entropy_axes[i].spines["top"].set_visible(False)
        entropy_axes[i].spines["right"].set_visible(False)
        entropy_axes[i].tick_params(direction="in", which="both", width=0.6)

        probs_axes[i].bar(strings, probs_array[i], color=colors[i])
        probs_axes[i].spines["top"].set_visible(False)
        probs_axes[i].spines["right"].set_visible(False)
        probs_axes[i].tick_params(axis="x", labelrotation=45)
        probs_axes[i].tick_params(direction="in", which="both", axis="y", width=0.6)

        for spine in entropy_axes[i].spines.values():
            spine.set_linewidth(0.5)
        for spine in probs_axes[i].spines.values():
            spine.set_linewidth(0.5)

    entropy_axes[0].set_xlabel("Interaction strength, $\\theta_{J}$")
    entropy_axes[1].set_xlabel("Interaction strength, $\\theta_{RX}$")
    entropy_axes[2].set_xlabel("Interaction strength, $\\theta_{F}$")

    probs_axes[0].set_xlabel("Pauli trajectory, $\\mathcal{P}$")
    probs_axes[1].set_xlabel("Pauli trajectory, $\\mathcal{P}$")
    probs_axes[2].set_xlabel("Pauli trajectory, $\\mathcal{P}$")

    entropy_axes[0].set_ylabel("Relative entropy, $S(\\rho \\| \\sigma)$")
    probs_axes[0].set_ylabel("Probability, $\\Pr(\\mathcal{P})$")

    entropy_axes[0].set_title("Heisenberg interaction")
    entropy_axes[1].set_title("Controlled-$X$ rotation")
    entropy_axes[2].set_title("Heisenberg + local field")

    probs_axes[0].set_title("Heisenberg interaction ($\\theta_J = \\frac{\\pi}{2}$)")
    probs_axes[1].set_title("Controlled-$X$ rotation ($\\theta_{RX} = \\frac{\\pi}{2}$)")
    probs_axes[2].set_title("Heisenberg + local field ($\\theta_F = \\pi$)")

    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for idx, ax in enumerate(axs.flatten()):
        x_pos = -0.17 if idx % 3 == 0 else -0.09
        ax.text(
            x_pos,
            1.12,
            labels[idx],
            transform=ax.transAxes,
            va="top",
            ha="left",
            size=12,
            fontweight="bold",
        )

    entropy_axes[1].legend(loc="upper center")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.07, hspace=0.35)
    fig.savefig(
        f"{FIGURE_DIR}/worked_examples.pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.0,
    )


def main():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    thetas = np.linspace(0, 2 * np.pi, 401)
    S_lists1 = run_entropy_calcs(H_heisenberg, thetas)
    S_lists2 = run_entropy_calcs(H_crx, thetas)
    S_lists3 = run_entropy_calcs(H_magnetic, thetas)

    probs1, _ = run_pauli_prob_calcs(H_heisenberg, 0.5 * np.pi)
    probs2, _ = run_pauli_prob_calcs(H_crx, 0.5 * np.pi)
    probs3, _ = run_pauli_prob_calcs(H_magnetic, np.pi)

    plot_combined(S_lists1, S_lists2, S_lists3, thetas, probs1, probs2, probs3)


if __name__ == "__main__":
    main()
