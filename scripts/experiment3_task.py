import numpy as np
import os
import argparse
from pathlib import Path

from corrqec2.noisemodels import StormQCAModel
from corrqec2.experiments import SurfaceCodeMemory


def sample_error_matrix_to_file(distance, rounds, shots, theta, a, b, outdir):
    experiment = SurfaceCodeMemory(distance=distance, rounds=rounds)
    model_params = {
        "a": a,
        "b": b,
        "theta": theta,
        "emissions": [[1, 0, 0, 0], [0, 1 / 3, 1 / 3, 1 / 3]],
    }
    model = StormQCAModel(model_params)
    error_matrix = model.gen_error_matrix(experiment, n_samples=shots).astype(np.uint8)
    pack_axis = 1

    job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "local")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    filename = f"{job_id}_{task_id}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    filepath = os.path.join(outdir, filename)
    # tmp = filepath + ".tmp"
    bitpacked = np.packbits(error_matrix, axis=pack_axis)
    np.savez_compressed(
        filepath,
        error_matrix=bitpacked,
        shape=error_matrix.shape,
        pack_axis=pack_axis,
        dtype=str(error_matrix.dtype),
        distance=int(distance),
        rounds=int(rounds),
        shots=int(shots),
        theta=float(theta),
        a=float(a),
        b=float(b),
    )
    # os.replace(tmp, filepath)


def main():
    outdir = (
        Path.home() / "mx95_scratch2" / "jkam" / "corrqec2_results" / "error_matrices"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distance", type=int, required=True, help="Surface code distance"
    )
    parser.add_argument("--rounds", type=int, required=True, help="Number of rounds")
    parser.add_argument("--shots", type=int, default=1, help="Number of shots")
    parser.add_argument(
        "--theta", type=float, required=True, help="Parameter theta in π radians"
    )
    parser.add_argument("--a", type=float, default=0.001, help="Parameter a")
    parser.add_argument("--b", type=float, default=0.5, help="Parameter b")

    args = parser.parse_args()
    sample_error_matrix_to_file(
        distance=args.distance,
        rounds=args.rounds,
        shots=args.shots,
        theta=args.theta * np.pi,
        a=args.a,
        b=args.b,
        outdir=outdir,
    )

    print(
        f"Saved error matrix for d={args.distance}, r={args.rounds}, shots={args.shots}, theta={args.theta}π, a={args.a}, b={args.b} to {outdir}"
    )


if __name__ == "__main__":
    main()
