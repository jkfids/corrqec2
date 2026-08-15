from pathlib import Path
import numpy as np
from corrqec2.sampling import create_task, run_tasks_to_csv, get_default_parser

# Sweep outputs are written here; on a cluster this is typically scratch
# space. Change this one line to run elsewhere.
RESULTS_DIR = Path.home() / "mx95_scratch2" / "jkam" / "corrqec2_results"


def calc_a_b(p_bar, Delta):
    """Calculate storm model parameters a, b for given spectral gap Delta (=a+b) and fixed marginal error rate p_bar."""
    a = 4 * p_bar * Delta / 3
    b = Delta * (1 - 4 * p_bar / 3)
    return a, b


def calc_Delta_from_xi(xi):
    if isinstance(xi, (int, float)):
        xi = [xi]
    out = np.zeros_like(xi, dtype=float)
    for i in range(len(xi)):
        if xi[i] == 0:
            out[i] = 1.0
        else:
            out[i] = 1 - np.exp(-1 / xi[i])
    return out if len(out) > 1 else out[0]


def calc_a_b_from_xi(p_bar, xi):
    Delta = calc_Delta_from_xi(xi)
    return calc_a_b(p_bar, Delta)


def main():

    # Parse command-line arguments
    parser = get_default_parser()
    args = parser.parse_args()

    # Configure output path
    output_dir = RESULTS_DIR

    # Sweep over number of rounds and correlation lengths
    tasks = []
    xis = [1, 2, 4, 6, 8, 12, 16, 20, 28]
    rounds_list = [5, 10, 15, 20, 25, 30, 35, 40]

    # Fixed noise model parameters
    p_gate = 0.001
    gate_noise = {
        "after_identity_depolarization": p_gate,
        "after_clifford_depolarization": p_gate,
        "before_measure_flip_probability": p_gate,
    }

    for rounds in rounds_list:
        for xi in xis:
            a, b = calc_a_b_from_xi(p_bar=p_gate, xi=xi)

            task = create_task(
                experiment="SurfaceCodeStability",
                experiment_args={"distance": 6, "rounds": rounds, "basis": "Z"},
                noise_model="StormModel",
                noise_model_args={
                    "model_params": {
                        "a": a,
                        "b": b,
                        "emissions": [[1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]],
                    },
                    "gate_noise": gate_noise,
                    "noisy_qubit_types": "all",
                },
                decoder="Pymatching",
                marginalized_detector_error_model=True,
            )
            tasks.append(task)

    print(f"Starting {Path(__file__).stem}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Number of tasks: {len(tasks)}")
    print(f"Rounds: {rounds}")
    print(f"Correlation lengths (ξ): {xis}")
    print(f"Marginal error rate (p_gate and p_bar): {p_gate}")
    print(f"Total shots per task: {args.max_shots}")
    print(f"Minimum batch size: {args.batch_size}")

    run_tasks_to_csv(
        tasks=tasks,
        n_workers=args.num_workers,
        max_shots=args.max_shots,
        min_batch_size=args.batch_size,
        output_dir=output_dir,
        filename=Path(__file__).stem + "_results",
        print_progress=True,
    )


if __name__ == "__main__":
    main()
