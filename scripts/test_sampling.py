from pathlib import Path
import argparse
import stim
import sinter
from corrqec2.sampling import SinterSampler


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test script for sampling via Sinter.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes (default: 4)",
    )
    parser.add_argument(
        "--total-shots",
        type=int,
        default=50,
        help="Maximum number of shots to simulate (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Minimum batch size for each worker (default: 10)",
    )
    args = parser.parse_args()

    # Configure output path
    save_dir = Path.home() / "mx95_scratch2" / "corrqec2_results"
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "test_run.csv"

    # Noise model parameters
    p_gate = 0.01
    gate_noise = {
        #'after_identity_depolarization': p_gate,
        "after_clifford_depolarization": p_gate,
        "before_measure_flip_probability": p_gate,
        #'after_reset_flip_probability': p_gate,
    }
    noise_model_params = {
        "a": 0.01,
        "b": 0.1,
        "emissions": [[1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]],
    }
    noisy_qubit_types = "syndrome"
    noise_model_args = {
        "model_params": noise_model_params,
        "gate_noise": gate_noise,
        "noisy_qubit_types": noisy_qubit_types,
    }

    metadata = {
        "experiment": "SurfaceCodeMemory",
        "experiment_args": {"distance": 5, "rounds": "1d", "basis": "Z"},
        "noise_model": "StormModel",
        "noise_model_args": noise_model_args,
        "decoder": "Pymatching",
        "min_batch_size": args.batch_size,
        "marginalized_detector_error_model": True,
    }

    print("Starting test sampling run...")
    print(f"Number of workers: {args.num_workers}")

    sampler = SinterSampler()
    task = sinter.Task(circuit=stim.Circuit(), json_metadata=metadata)
    stats = sinter.collect(
        tasks=[task],
        num_workers=args.num_workers,
        decoders="custom_sampler",
        custom_decoders={"custom_sampler": sampler},
        max_shots=args.total_shots,
        print_progress=False,
        save_resume_filepath=output_path,
    )

    print("Test script successful!")
    print(f"Collected {stats[0].shots} shots")
    print(f"Observed {stats[0].errors} errors")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
