from pathlib import Path
import numpy as np
from corrqec2.sampling import create_task, run_tasks_to_csv, get_default_parser


def calc_a_b(p_bar, Delta):
    """Calculate storm model parameters a, b for given spectral gap Delta (=a+b) and fixed marginal error rate p_bar."""
    a = 4 * p_bar * Delta / 3
    b = Delta * (1 - 4 * p_bar / 3)
    return a, b


def calc_Delta_from_xi(xi):
    return 1 - np.exp(-1 / xi)


def calc_a_b_from_xi(p_bar, xi):
    Delta = calc_Delta_from_xi(xi)
    return calc_a_b(p_bar, Delta)


def main():

    # Parse command-line arguments
    parser = get_default_parser()
    args = parser.parse_args()

    # Configure output path
    output_dir = Path.home() / "mx95_scratch2" / "jkam" / "corrqec2_results"

    # Sweep over distances and correlation lengths
    tasks = []
    xis = [2, 4, 6, 8, 12, 16, 20, 28]
    distances = [5, 7, 9, 11, 13, 15, 17]

    # Fixed noise model parameters
    p_gate = 0.001
    gate_noise = {
        "after_identity_depolarization": p_gate,
        "after_clifford_depolarization": p_gate,
        "before_measure_flip_probability": p_gate,
    }

    for distance in distances:
        for xi in xis:
            a, b = calc_a_b_from_xi(p_bar=p_gate, xi=xi)

            task = create_task(
                experiment="SurfaceCodeMemory",
                experiment_args={"distance": distance, "rounds": "3d", "basis": "Z"},
                noise_model="StormModel",
                noise_model_args={
                    "model_params": {
                        "a": a,
                        "b": b,
                        "emissions": [[1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]],
                    },
                    "gate_noise": gate_noise,
                    "noisy_qubit_types": "syndrome",
                },
                decoder="Pymatching",
                marginalized_detector_error_model=True,
            )
            tasks.append(task)

    print(f"Starting {Path(__file__).stem}: Distance & correlation length sweep")
    print(f"Number of workers: {args.num_workers}")
    print(f"Number of tasks: {len(tasks)}")
    print(f"Distances: {distances}")
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
