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
XX = qi.Pauli("XX").to_matrix()
YY = qi.Pauli("YY").to_matrix()
ZZ = qi.Pauli("ZZ").to_matrix()
XI = qi.Pauli("XI").to_matrix()
YI = qi.Pauli("YI").to_matrix()
ZI = qi.Pauli("ZI").to_matrix()
XZ = qi.Pauli("XZ").to_matrix()


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
    rho = 1e-10 + rho / np.trace(rho)  # Ensure rho is trace 1
    sigma = 1e-10 + sigma / np.trace(sigma)  # Ensure sigma is trace 1
    log_rho = scipy.linalg.logm(rho)
    log_sigma = scipy.linalg.logm(sigma)
    return np.trace(rho @ (log_rho - log_sigma)).real


def run_entropy_calcs(Hamiltonian, thetas):
    S_lists = [[] for _ in range(3)]
    for theta in thetas:
        U = scipy.linalg.expm(-1j * Hamiltonian(theta))
        pt = process_tensor_from_stinespring(U)
        choi = pt.choi_matrix
        choi_twirled = multitime_twirl(choi)
        S_lists[0].append(calc_gqmi(choi))
        S_lists[1].append(calc_gqmi(choi_twirled))
        S_lists[2].append(calc_relative_entropy(choi, choi_twirled))

    return S_lists


def plot_entropies(S_lists1, S_lists2, S_lists3, thetas):
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
        }
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.2, 3.2), sharey=True)
    axs = [ax1, ax2, ax3]
    colors = sns.color_palette("muted")

    S_array = [S_lists1, S_lists2, S_lists3]

    for i in range(3):
        axs[i].plot(
            thetas / np.pi,
            S_array[i][0],
            color=colors[0],
            # marker="o",
            # markersize=3,
        )
        axs[i].plot(
            thetas / np.pi,
            S_array[i][1],
            color=colors[1],
            label="$\\Upsilon, \\Upsilon$",
            # marker="^",
            # markersize=3,
        )
        axs[i].plot(
            thetas / np.pi,
            S_array[i][2],
            color=colors[2],
            linestyle="--",
            # marker="x",
            # markersize=3,
        )

        axs[i].xaxis.set_major_formatter(FormatStrFormatter("%g$\pi$"))
        axs[i].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
        axs[i].set_xticks([0, 0.5, 1.0, 1.5, 2.0])
        axs[i].set_xlabel("Interaction strength, $\\theta$")

    ax1.set_ylabel("Quantum relative entropy, S")
    ax1.set_title("Heisenberg interaction")
    ax2.set_title("Controlled-$X$ rotation")
    ax3.set_title("Heisenberg with local field")
    ax2.legend(loc="upper center")

    fig.tight_layout()
    fig.savefig(
        "./project/paper/figures/worked_examples1.pdf",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.0,
    )


def main():
    thetas = np.linspace(0, 2 * np.pi, 11)
    S_lists1 = run_entropy_calcs(H_heisenberg, thetas)
    S_lists2 = run_entropy_calcs(H_crx, thetas)
    S_lists3 = run_entropy_calcs(H_magnetic, thetas)

    plot_entropies(S_lists1, S_lists2, S_lists3, thetas)


if __name__ == "__main__":
    main()
