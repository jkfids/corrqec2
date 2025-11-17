import stim
import sinter
from corrqec2.sampling import SinterSampler


def main():
    sampler = SinterSampler()
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
        "min_batch_size": 10,
        "marginalized_detector_error_model": True,
    }

    task = sinter.Task(circuit=stim.Circuit(), json_metadata=metadata)
    stats = sinter.collect(
        tasks=[task],
        num_workers=4,
        decoders="custom_sampler",
        custom_decoders={"custom_sampler": sampler},
        max_shots=80,
        print_progress=False,
    )

    print("Test script successful!")
    print(f"Collected {stats[0].shots} shots")
    print(f"Observed {stats[0].errors} errors")


if __name__ == "__main__":
    main()
