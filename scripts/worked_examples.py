import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import seaborn as sns

import qiskit.quantum_info as qi
from processtensor import QTensor, QuantumState, UnitarySE, ProcessTensor

I = qi.Pauli("I").to_matrix()
X = qi.Pauli("X").to_matrix()
Y = qi.Pauli("Y").to_matrix()
Z = qi.Pauli("Z").to_matrix()

II = qi.Pauli("II").to_matrix()
IX = qi.Pauli("IX").to_matrix()
IY = qi.Pauli("IY").to_matrix()
IZ = qi.Pauli("IZ").to_matrix()
XI = qi.Pauli("XI").to_matrix()
XX = qi.Pauli("XX").to_matrix()
XY = qi.Pauli("XY").to_matrix()
XZ = qi.Pauli("XZ").to_matrix()
YI = qi.Pauli("YI").to_matrix()
YX = qi.Pauli("YX").to_matrix()
YY = qi.Pauli("YY").to_matrix()
YZ = qi.Pauli("YZ").to_matrix()
ZI = qi.Pauli("ZI").to_matrix()
ZX = qi.Pauli("ZX").to_matrix()
ZY = qi.Pauli("ZY").to_matrix()
ZZ = qi.Pauli("ZZ").to_matrix()


def H_heisenberg(theta=0.5 * np.pi):
    return -theta * 0.5 * (XX + YY + ZZ)


def H_crx(theta=0.5 * np.pi):
    return -theta * 0.5 * (XI - XZ)


def H_magnetic(theta=0.5 * np.pi):
    return H_heisenberg(0.5 * np.pi) - theta * (XI + YI + ZI)


def multi_kron(Ms):
    """Compute the tensor product of a list of matrices."""
    prod = Ms[0]
    for i in range(1, len(Ms)):
        prod = np.kron(prod, Ms[i])
    return prod


def choi_pauli(pauli: str):

    bell_vector = np.array([1, 0, 0, 1]) / np.sqrt(2)
    bell_matrix = np.outer(bell_vector, bell_vector)

    pauli_matrix = qi.Pauli("I" + pauli).to_matrix()
    choi = pauli_matrix @ bell_matrix @ pauli_matrix
    return choi


def double_choi_from_paulis(pauli_string):

    s1, s2 = pauli_string
    choi1 = choi_pauli(s1)
    choi2 = choi_pauli(s2)

    return np.kron(choi1, choi2)


def calc_pauli_expectation(choi, pauli_string):
    choi = choi / np.trace(choi)  # Ensure it's a valid density matrix
    pauli_choi = double_choi_from_paulis(pauli_string)
    return np.trace(choi @ pauli_choi).real


def process_tensor_from_stinespring(unitary):
    rho_init = np.ones((2, 2), dtype=np.complex128) / 2
    q0 = QuantumState.from_matrix(rho_init, dS=1)
    q1 = UnitarySE.from_unitary(unitary, dS=2, dE=2)
    q2 = q0 @ q1 @ q1
    pt = q2.trace_subsystem(q2.ndim - 1)
    pt = ProcessTensor(pt)
    return pt


def multitime_twirl(choi, d=2):
    d2 = d * d
    pst = pauli_superop_tensor(d)
    pt_data = ProcessTensor.from_choi(choi, (d, d, d, d)).data.reshape(d2, d2, d2, d2)
    twirled = (
        np.einsum("abcd, xai, xbj, yck, ydl -> ijkl", pt_data, pst, pst, pst, pst) / 16
    )

    return ProcessTensor(twirled.reshape(1, d2, d2, d2, d2, 1)).choi_matrix


def pauli_superop_tensor(d):
    n_qubits = int(np.log2(d))
    paulis = [p.to_matrix() for p in qi.pauli_basis(n_qubits)]
    return np.array([np.kron(p.T, p) for p in paulis])


def calc_gqmi(choi):
    """Compute the generalised quantum mutual information of a process tensor given its Choi matrix."""
    choi_marginalised = marginalise_choi(choi)
    gqmi = calc_relative_entropy(choi, choi_marginalised)
    return gqmi


def marginalise_choi(choi):
    choi = choi / np.trace(choi)  # Ensure it's a valid density matrix
    choi0 = qi.partial_trace(choi, (0, 1)) * 2
    choi1 = qi.partial_trace(choi, (2, 3)) * 2
    return np.kron(choi0, choi1)


def calc_relative_entropy(rho, sigma):
    """Compute the quantum relative entropy S(rho || sigma)."""
    rho = 1e-8 + rho / np.trace(rho)  # Ensure rho is trace 1
    sigma = 1e-8 + sigma / np.trace(sigma)  # Ensure sigma is trace 1
    log_rho = scipy.linalg.logm(rho)
    log_sigma = scipy.linalg.logm(sigma)
    return np.trace(rho @ (log_rho - log_sigma)).real


def choi_svd(choi):
    d = int(np.sqrt(choi.shape[0]))
    shuffled = choi.reshape(d, d, d, d).transpose(0, 2, 1, 3).reshape(d * d, d * d)
    U, S, Vh = np.linalg.svd(shuffled)
    n_nonzero = np.sum(S > 1e-10)
    U = U[:, :n_nonzero]
    S = S[:n_nonzero]
    Vh = Vh[:n_nonzero, :]
    V = np.diag(S) @ Vh
    return U, S, V


def gen_process_tensor(Hamiltonian, theta):
    U = scipy.linalg.expm(-1j * Hamiltonian(theta))
    pt = process_tensor_from_stinespring(U)
    return pt


def run_entropy_calcs(Hamiltonian, thetas):
    S_lists = [[] for _ in range(5)]
    for theta in thetas:
        # U = scipy.linalg.expm(-1j * Hamiltonian(theta))
        # pt = process_tensor_from_stinespring(U)
        pt = gen_process_tensor(Hamiltonian, theta)
        choi = pt.choi_matrix
        choi_twirled = multitime_twirl(choi)
        S_lists[0].append(calc_gqmi(choi))
        S_lists[1].append(calc_gqmi(choi_twirled))
        S_lists[2].append(calc_relative_entropy(choi, choi_twirled))
        # _, U, _ = choi_svd(choi)
        # _, V, _ = choi_svd(choi_twirled)
        # S_lists[3].append(len(U))
        # S_lists[4].append(len(V))
        # S_lists[3].append(1 - calc_pauli_expectation(choi, "II"))

    return S_lists


def plot_entropies(S_lists1, S_lists2, S_lists3, thetas):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.2, 3.2), sharey=True)
    axs = [ax1, ax2, ax3]
    colors = sns.color_palette("muted")

    S_array = [S_lists1, S_lists2, S_lists3]

    for i in range(3):
        axs[i].plot(
            thetas / np.pi,
            S_array[i][0],
            color=colors[0],
            label="$\\rho:$ original, $\\sigma:$ marginalised",
            # marker="o",
            # markersize=3,
        )
        axs[i].plot(
            thetas / np.pi,
            S_array[i][1],
            color=colors[1],
            label="$\\rho:$ twirled, $\\sigma:$ twirled marginalised",
            # marker="^",
            # markersize=3,
        )
        axs[i].plot(
            thetas / np.pi,
            S_array[i][2],
            color=colors[2],
            linestyle="--",
            label="$\\rho:$ original, $\\sigma:$ twirled",
            # marker="x",
            # markersize=3,
        )

        # axs[i].plot(
        #     thetas / np.pi,
        #     S_array[i][3],
        #     color=colors[3],
        #     linestyle=":",
        #     # label="log(SV count)",
        #     # marker="s",
        # )

        # axs[i].plot(
        #     thetas / np.pi,
        #     S_array[i][4],
        #     color=colors[4],
        #     linestyle=":",
        # )

        axs[i].xaxis.set_major_formatter(FormatStrFormatter("%g$\\pi$"))
        axs[i].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
        axs[i].set_xticks([0, 0.5, 1.0, 1.5, 2.0])
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)

        axs[i].tick_params(direction="in", which="both", width=0.6)

        for spine in axs[i].spines.values():
            spine.set_linewidth(0.5)

    ax1.set_xlabel("Interaction strength, $\\theta_{J}$")
    ax2.set_xlabel("Interaction strength, $\\theta_{RX}$")
    ax3.set_xlabel("Interaction strength, $\\theta_{F}$")

    ax1.text(-0.15, 1.11, "(a)", transform=ax1.transAxes, va="top", ha="left", size=12)
    ax2.text(-0.09, 1.11, "(b)", transform=ax2.transAxes, va="top", ha="left", size=12)
    ax3.text(-0.09, 1.11, "(c)", transform=ax3.transAxes, va="top", ha="left", size=12)
    ax1.set_ylabel("Quantum relative entropy, $S(\\rho \\| \\sigma)$")
    ax1.set_title("Heisenberg interaction")
    ax2.set_title("Controlled-$X$ rotation")
    ax3.set_title("Heisenberg + local field")
    ax2.legend(loc="upper center")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.07)
    fig.savefig(
        "./project/paper/figures/worked_examples1.pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.0,
    )


def calc_pauli_probs(choi):
    probs = []
    strings = []

    for s1 in ["I", "X", "Y", "Z"]:
        for s2 in ["I", "X", "Y", "Z"]:
            pauli_string = s1 + s2
            prob = calc_pauli_expectation(choi, pauli_string)
            probs.append(prob)
            strings.append(pauli_string)
    return probs, strings


def run_pauli_prob_calcs(Hamiltonian, theta):
    pt = gen_process_tensor(Hamiltonian, theta)
    choi = pt.choi_matrix
    probs, strings = calc_pauli_probs(choi)
    return probs, strings


def plot_pauli_probs(probs1, probs2, probs3):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.2, 3.0), sharey=True)
    axs = [ax1, ax2, ax3]
    colors = sns.color_palette("muted")

    probs_array = [probs1, probs2, probs3]
    strings = [A + B for A in ["I", "X", "Y", "Z"] for B in ["I", "X", "Y", "Z"]]

    for i in range(3):
        axs[i].bar(strings, probs_array[i], color=colors[i])

        axs[i].set_xlabel("Pauli trajectory, $\\mathcal{P}$")
        axs[i].spines["top"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].tick_params(axis="x", labelrotation=45)

        axs[i].tick_params(direction="in", which="both", axis="y", width=0.6)
        for spine in axs[i].spines.values():
            spine.set_linewidth(0.5)

    ax1.text(-0.15, 1.11, "(a)", transform=ax1.transAxes, va="top", ha="left", size=12)
    ax2.text(-0.09, 1.11, "(b)", transform=ax2.transAxes, va="top", ha="left", size=12)
    ax3.text(-0.09, 1.11, "(c)", transform=ax3.transAxes, va="top", ha="left", size=12)
    ax1.set_ylabel("Probability, $\\Pr(\\mathcal{P})$")
    ax1.set_title("Heisenberg interaction ($\\theta_J = \\frac{\\pi}{2}$)")
    ax2.set_title("Controlled-$X$ rotation ($\\theta_{RX} = \\frac{\\pi}{2}$)")
    ax3.set_title("Heisenberg + local field ($\\theta_F = \\pi$)")

    # fig.subplots_adjust(wspace=-0.01)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.07)
    fig.savefig(
        "./project/paper/figures/worked_examples2.pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.0,
    )


def main():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 9,
        }
    )

    thetas = np.linspace(0, 2 * np.pi, 401)
    S_lists1 = run_entropy_calcs(H_heisenberg, thetas)
    S_lists2 = run_entropy_calcs(H_crx, thetas)
    S_lists3 = run_entropy_calcs(H_magnetic, thetas)
    plot_entropies(S_lists1, S_lists2, S_lists3, thetas)

    probs1, _ = run_pauli_prob_calcs(H_heisenberg, 0.5 * np.pi)
    probs2, _ = run_pauli_prob_calcs(H_crx, 0.5 * np.pi)
    probs3, _ = run_pauli_prob_calcs(H_magnetic, np.pi)
    plot_pauli_probs(probs1, probs2, probs3)


if __name__ == "__main__":
    main()
