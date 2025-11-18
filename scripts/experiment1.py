from pathlib import Path
import argparse
import numpy as np
import stim
import sinter
from corrqec2.sampling import SinterSampler, save_stats_to_csv


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
    parser = argparse.ArgumentParser(description="Test script for sampling via Sinter.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--max-shots",
        type=int,
        default=1000,
        help="Maximum number of shots to simulate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Minimum batch size for each worker",
    )
    args = parser.parse_args()

    # Configure output path
    save_dir = Path.home() / "mx95_scratch2" / "jkam" / "corrqec2_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    resume_path = save_dir / "experiment1.tmp.csv"
    output_path = save_dir / "experiment1.csv"

    # Fixed noise model parameters
    p_gate = 0.001
    gate_noise = {
        "after_identity_depolarization": p_gate,
        "after_clifford_depolarization": p_gate,
        "before_measure_flip_probability": p_gate,
        #'after_reset_flip_probability': p_gate,
    }
    noisy_qubit_types = "syndrome"

    # Sweep over distances and correlation lengths
    tasks = []
    xis = [2, 4, 6, 8, 12, 16, 24]
    distances = [5, 7, 9, 11, 13]
    for distance in distances:
        for xi in xis:
            a, b = calc_a_b_from_xi(p_bar=p_gate, xi=xi)
            experiment_args = {"distance": distance, "rounds": "3d", "basis": "Z"}
            noise_model_args = {
                "model_params": {
                    "a": a,
                    "b": b,
                    "emissions": [[1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]],
                },
                "gate_noise": gate_noise,
                "noisy_qubit_types": noisy_qubit_types,
            }

            meta_data = {
                "experiment": "SurfaceCodeMemory",
                "experiment_args": experiment_args,
                "noise_model": "StormModel",
                "noise_model_args": noise_model_args,
                "decoder": "Pymatching",
                "min_batch_size": args.batch_size,
                "marginalized_detector_error_model": True,
            }
            task = sinter.Task(circuit=stim.Circuit(), json_metadata=meta_data)
            tasks.append(task)

    print("Starting test sampling run...")
    print("Starting experiment1: Distance & correlation length sweep")
    print(f"Number of workers: {args.num_workers}")
    print(f"Number of tasks: {len(tasks)}")
    print(f"Distances: {distances}")
    print(f"Correlation lengths (ξ): {xis}")
    print(f"Marginal error rate (p_gate and p_bar): {p_gate}")
    print(f"Total shots per task: {args.max_shots}")

    if resume_path.exists():
        print(f"Resuming from existing file: {resume_path}")

    sampler = SinterSampler(print_progress=True)
    stats = sinter.collect(
        tasks=tasks,
        num_workers=args.num_workers,
        decoders="custom_sampler",
        custom_decoders={"custom_sampler": sampler},
        max_shots=args.max_shots,
        print_progress=False,
        save_resume_filepath=resume_path,
    )

    # Save results to CSV
    save_stats_to_csv(stats, output_path)
    if resume_path.exists():
        resume_path.unlink()

    print("Test script successful!")
    # print(f"Collected {stats[0].shots} shots")
    # print(f"Observed {stats[0].errors} errors")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
