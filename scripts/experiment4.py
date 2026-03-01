from pathlib import Path
import numpy as np
from corrqec2.sampling import create_task, run_tasks_to_csv, get_default_parser


def main():
    # Parse command-line arguments
    parser = get_default_parser()
    args = parser.parse_args()

    # Configure output path
    output_dir = Path.home() / "mx95_scratch2" / "jkam" / "corrqec2_results"

    # Sweep over distances and controlled-rotation angles
    tasks = []
    thetas = [0.0, 0.1, 0.2, 0.3, 0.34, 0.36, 0.38, 0.4, 0.42, 0.5]
    thetas = [t * np.pi for t in thetas]
    distances = [5, 7, 9, 11, 13, 15]

    # Fixed noise model parameters
    p_gate = 0.001
    gate_noise = {
        "after_identity_depolarization": p_gate,
        "after_clifford_depolarization": p_gate,
        "before_measure_flip_probability": p_gate,
        "after_reset_flip_probability": p_gate,
    }

    for distance in distances:
        for theta in thetas:
            task = create_task(
                experiment="SurfaceCodeMemory",
                experiment_args={"distance": distance, "rounds": "3d", "basis": "Z"},
                noise_model="QCAModel",
                noise_model_args={
                    "model_params": {
                        "a": 0.0001,
                        "b": 0.5,
                        "theta": theta,
                        "emissions": [[1.0, 0.0, 0.0, 0.0], [0.0, 1 / 3, 1 / 3, 1 / 3]],
                    },
                    "gate_noise": gate_noise,
                },
                decoder="Pymatching",
                marginalized_detector_error_model=True,
            )
            tasks.append(task)

    print(f"Starting {Path(__file__).stem}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Number of tasks: {len(tasks)}")
    print(f"Distances: {distances}")
    print(f"Thetas (θ): {thetas}")
    # print(f"Thetas (θ): {list(np.round([t / np.pi for t in thetas], 4))}π")
    print(f"Gate error rate (p_gate): {p_gate}")
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
